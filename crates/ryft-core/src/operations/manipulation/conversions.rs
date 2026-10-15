//! Operations that change a value's type representation, convert or reinterpret its elements as another [`DataType`],
//! or reduce the precision of its floating-point elements within their own [`DataType`]. Numerical conversion keeps
//! the shape, sharding, and memory space of its input and follows the destination format's rounding, saturation, and
//! exceptional-value rules, whereas a bitcast keeps every encoding bit and adds or consumes a trailing axis when the
//! two element widths differ. A type-representation cast preserves the logical value while adapting its shape
//! information and nominal type identities through a relationship that is checked or assumed. This module
//! provides the following:
//!
//!   - The [`ConvertElementType`] value capability, whose
//!     [`convert_element_type`](ConvertElementType::convert_element_type) converts elements numerically, whose
//!     [`bitcast_element_type`](ConvertElementType::bitcast_element_type) reinterprets their bits, and whose
//!     [`promote_element_type`](ConvertElementType::promote_element_type) converts only along the type promotion
//!     lattice. A numerical conversion to the input's own element type returns the input without staging anything,
//!     whereas a bitcast is always staged, since its derivative is zero rather than the identity.
//!   - The [`ConvertElementTypeOperation`] that the capability stages, which carries the destination
//!     [`data_type`](ConvertElementTypeOperation::data_type) and whether it is a
//!     [`bitcast`](ConvertElementTypeOperation::bitcast). It is generic over the [`Type`] universe it operates in,
//!     so one operation serves bare [`DataType`]s and [`ArrayType`]s alike.
//!   - The [`ElementType`] type capability, through which the operation infers its output type. Replacing the element
//!     type of a [`Type`] is a metadata-only change that clears byte-stride layouts when the element width changes,
//!     and reinterpreting it conserves the number of encoding bits by adjusting the trailing shape.
//!   - The [`ReducePrecision`] value capability and the [`ReducePrecisionOperation`] that it stages, which round real
//!     floating-point elements to the values of a format with fewer exponent or mantissa bits while keeping their
//!     element type, so that the precision of a narrower format can be simulated without converting to it.
//!   - The [`Cast`] value capability and the [`CastOperation`] that it stages, which preserve element values while
//!     changing their type representation. A checked cast validates the required type relationships at runtime.
//!     An unchecked cast assumes those relationships, for example after a [`GuardOperation`](crate::GuardOperation)
//!     has checked them. Differentiation and transposition use unchecked casts for tangent and cotangent
//!     representations whose relationships the primal computation has established. The [`CastType`] type capability
//!     validates the output type's requirements, such as its element type and layout. The operation checks dimension
//!     relationships separately.
//!
//! Numerical conversion is linear, so differentiation converts the primal and aligns the tangent with the output's
//! tangent type, transposition converts the cotangent back to the input's cotangent type, and conversions into or
//! out of a type without a tangent space (e.g., integers) produce structural zeros. A bitcast has a structural zero
//! derivative and cannot be transposed. Precision reduction is treated as linear and as its own transpose, so that
//! tangents and cotangents carry the same precision as the primal. Batching keeps the mapped axis in place, except that
//! a widening bitcast moves it away from the trailing position it would otherwise consume. Backends lower numerical
//! conversion, bitcasts, and precision reduction to their element-conversion, bit-reinterpretation, and
//! precision-reduction constructs (e.g., [`stablehlo.convert`](https://openxla.org/stablehlo/spec#convert),
//! [`stablehlo.bitcast_convert`](https://openxla.org/stablehlo/spec#bitcast_convert), and
//! [`stablehlo.reduce_precision`](https://openxla.org/stablehlo/spec#reduce_precision) in the XLA backend).
//!
//! # Example
//!
//! Converting a traced value stages one instruction per conversion, and only a bitcast records its mode:
//!
//! ```rust
//! # use indoc::indoc;
//! # use ryft_core::{Array, ArrayOperation, ArrayType, ConvertElementType, DataType, ProgramError, TracingContext};
//! # fn main() -> Result<(), ProgramError> {
//! let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
//!     |input| input.convert_element_type(DataType::F32)?.bitcast_element_type(DataType::U32),
//!     ArrayType::new_static(DataType::F64, [3]),
//! )?;
//!
//! assert_eq!(
//!     program.to_string(),
//!     indoc! {"
//!         lambda %0:f64[3] .
//!         let %1:f32[3] = convert_element_type [data_type=f32] %0
//!             %2:u32[3] = convert_element_type [data_type=u32, bitcast=true] %1
//!         in (%2)"},
//! );
//!
//! assert_eq!(
//!     program.interpret(Array::vector(vec![1.0f64, -2.0, 0.5])?)?,
//!     Array::vector(vec![0x3f800000u32, 0xc0000000, 0x3f000000])?,
//! );
//! # Ok(())
//! # }
//! ```

use std::borrow::Cow;
use std::fmt::Display;
use std::marker::PhantomData;

use ryft_macros::capability;

use crate::arrays::{
    Array, ArrayAddressing, ArrayBatch, ArrayBatchingPolicy, ArrayExtentBatchingPolicy, ArrayIrType, ArrayIrValue,
    ArrayType, BatchingTypePolicy, DataType, Dimension, DimensionType, Layout, ShardingDimension, bf16, f16,
};
use crate::batching::{
    BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError, InterpretableBatchableOperation,
};
use crate::contexts::{Context, Domain, StagingContext};
use crate::differentiation::{
    CotangentAccumulator, DifferentiableOperation, DifferentiableType, DifferentiationContext, DifferentiationDriver,
    DifferentiationDual, DifferentiationError, DifferentiationPolicy, ElementwiseDerivativeAlignment,
    TransposableOperation, TranspositionContext, TranspositionDriver,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_differentiable_operation, impl_reference_dischargeable_operation};
use crate::operations::Capability;
use crate::operations::arithmetic::AddOperation;
use crate::operations::manipulation::transposition::Transpose;
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::{
    EffectClass, EffectClasses, Effects, MaybeZero, Operation, OperationFormatter, ProgramError, RegionInterface, Type,
    TypeError, TypeIdentity, TypeIdentityPosition, TypeIdentityRenaming, TypeRefinements, Typed, Value,
    ValueDomainDispatch,
};
use crate::tracing::{Tracer, TracingContext};

/// Canonical operation name for [`ConvertElementTypeOperation`].
pub const CONVERT_ELEMENT_TYPE_OPERATION_NAME: &str = "convert_element_type";

/// [`Operation`] that converts the elements of its input to a requested [`DataType`], either numerically or by
/// reinterpreting their encoding bits. Numerical conversion keeps the input's shape, sharding, and memory space,
/// uses the destination format's rounding, saturation, and exceptional-value rules, and rejects tokens as well as
/// conversions between structural zeros and materialized element types. A bitcast keeps every encoding bit and adds
/// or consumes a trailing axis when the two element widths differ. Type inference goes through the [`ElementType`]
/// capability of the type universe `T`, so the same operation serves bare [`DataType`]s and [`ArrayType`]s.
/// Refer to the documentation of [`ConvertElementType`] for more information.
///
/// Interpretation converts the values through the same capability. Batching keeps the mapped axis in place, except that
/// a widening bitcast moves it away from the trailing position it would otherwise consume. Differentiation converts the
/// primal and aligns the tangent with the output's tangent type, producing a structural zero when either side has no
/// tangent space, and transposition converts the cotangent back to the input's cotangent type. A bitcast has a
/// structural zero derivative and cannot be transposed.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ConvertElementTypeOperation<T: ElementType> {
    /// Refer to the documentation of [`data_type`](Self::data_type) for more information.
    data_type: DataType,

    /// Refer to the documentation of [`bitcast`](Self::bitcast) for more information.
    bitcast: bool,

    /// [`PhantomData`] marker tying this [`Operation`] to the [`Type`] universe in which it is valid.
    marker: PhantomData<fn() -> T>,
}

impl<T: ElementType> ConvertElementTypeOperation<T> {
    /// Creates a new [`ConvertElementTypeOperation`] that converts its input to `data_type`. The source element type
    /// comes from the input, so the conversion is validated during type inference rather than here.
    ///
    /// # Parameters
    ///
    ///   - `data_type`: Destination element type.
    ///   - `bitcast`: Whether to reinterpret encoding bits instead of converting numerical values.
    #[inline]
    pub fn new(data_type: DataType, bitcast: bool) -> Self {
        Self { data_type, bitcast, marker: PhantomData }
    }

    /// Returns the destination element [`DataType`] of this [`ConvertElementTypeOperation`].
    #[inline]
    pub fn data_type(&self) -> DataType {
        self.data_type
    }

    /// Returns whether this [`ConvertElementTypeOperation`] reinterprets encoding bits
    /// rather than converting numerical values.
    #[inline]
    pub fn bitcast(&self) -> bool {
        self.bitcast
    }
}

// Implemented manually because deriving `Copy` would require `T: Copy`, although the marker is `Copy` for every `T`.
impl<T: ElementType> Copy for ConvertElementTypeOperation<T> {}

impl<T: ElementType> Display for ConvertElementTypeOperation<T> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<T: ElementType> Operation for ConvertElementTypeOperation<T> {
    type Type = T;

    #[inline]
    fn name(&self) -> &'static str {
        CONVERT_ELEMENT_TYPE_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[T],
        region_interfaces: &[RegionInterface<T>],
    ) -> Result<Vec<T>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        check_count!("region", region_interfaces, 0, TypeError);
        if self.bitcast {
            return Ok(vec![input_types[0].with_bitcast_element_type(self.data_type)?]);
        }

        if input_types[0].element_type().is_token() || self.data_type.is_token() {
            return Err(TypeError::invalid(format!(
                "cannot convert values to or from the `{}` data type",
                DataType::Token,
            )));
        }

        // Structural zeros have no materialized elements to convert, so they only convert to themselves.
        if input_types[0].element_type().is_zero() != self.data_type.is_zero() {
            return Err(TypeError::invalid(format!(
                "cannot convert values to or from the `{}` data type",
                DataType::Zero,
            )));
        }

        Ok(vec![input_types[0].with_element_type(self.data_type)])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("data_type", self.data_type)?;
            if self.bitcast {
                operation.field("bitcast", true)?;
            }
            Ok(())
        })
    }
}

impl<C: Domain<Type: ElementType, Value: ConvertElementType>> InterpretableOperation<C>
    for ConvertElementTypeOperation<C::Type>
{
    #[inline]
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![if self.bitcast {
            inputs[0].bitcast_element_type(self.data_type)?
        } else {
            inputs[0].convert_element_type(self.data_type)?
        }])
    }
}

impl<C: Context<Type: ElementType, Operation: From<ConvertElementTypeOperation<C::Type>>>>
    PartiallyEvaluatableOperation<C> for ConvertElementTypeOperation<C::Type>
{
}

impl<C: Context<Type = ArrayType, Value: ConvertElementType + Transpose>, P: ArrayExtentBatchingPolicy<C>>
    BatchableOperation<C, ArrayBatchingPolicy<P>> for ConvertElementTypeOperation<ArrayType>
{
    fn batch<D: BatchingDriver<C, ArrayBatchingPolicy<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatchingPolicy<P>>, BatchingError> {
        check_count!("input", inputs, 1, ProgramError);

        // Numerical conversion and equal-width bitcasts keep the mapped axis in place. A widening bitcast consumes
        // the packed trailing axis, so a bounded ragged axis in that position cannot be consumed at all, because its
        // padding would be folded into the wider elements, and a mapped axis in that position moves to the front first.
        let widening = self.bitcast && inputs[0].value().r#type().data_type().bit_width() < self.data_type.bit_width();
        let rank = inputs[0].value().r#type().rank();
        if widening && inputs[0].ragged_axes().iter().any(|axis| axis.axis() + 1 == rank) {
            return Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "`{CONVERT_ELEMENT_TYPE_OPERATION_NAME}` in bitcast mode does not support consuming a bounded \
                     ragged trailing dimension",
                ),
            });
        }

        self.infer_output_types(&[inputs[0].unbatched_type()], &[])?;

        let input = if widening && inputs[0].batch_axis_position() == Some(rank - 1) {
            inputs[0].move_axis(0)?
        } else {
            inputs[0].clone()
        };

        let mut outputs =
            self.interpret_with_batch_axes(context, std::slice::from_ref(&input), &[input.batch_axis()])?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(vec![outputs.remove(0).with_ragged_axes(input.ragged_axes().to_vec())?].into())
    }
}

impl_differentiable_operation! {
    <T> ConvertElementTypeOperation<T>,
    jvp<C>
    where
        T: DifferentiableType + ElementType,
        C: Context<Type = T>,
        C::Value: ConvertElementType + ElementwiseDerivativeAlignment<T>,
    {
        |operation, context, _driver, inputs| {
            // The primal converts to the requested element type and a live tangent is aligned with the output's tangent
            // type. A bitcast, or a conversion into a type with no tangent space, has a structural zero tangent.
            check_count!("input", inputs, 1, ProgramError);
            if operation.bitcast {
                let primal = inputs[0].primal().bitcast_element_type(operation.data_type)?;
                return Ok(vec![DifferentiationDual::new_with_zero_tangent(primal)?]);
            }
            let primal = inputs[0].primal().convert_element_type(operation.data_type)?;
            let output_tangent_type = primal.r#type().tangent()?;
            let tangent = match inputs[0].tangent() {
                _ if output_tangent_type.is_zero_space() => MaybeZero::Zero(output_tangent_type),
                MaybeZero::Zero(_) => MaybeZero::Zero(output_tangent_type),
                MaybeZero::Value(tangent) => MaybeZero::Value(tangent.align_tangent(
                    &output_tangent_type,
                    &context.primal_to_tangent(primal.clone())?,
                )?),
            };
            Ok(vec![DifferentiationDual::new(primal, tangent)?])
        }
    },
    transpose<V, O>
    where
        T: DifferentiableType + ElementType,
        V: Value<Type = T>,
        O: From<ConvertElementTypeOperation<T>>,
        Tracer<TracingContext<V, O>>: ElementwiseDerivativeAlignment<T>,
    {
        |operation, context, _driver, inputs, outputs, accumulators| {
            if operation.bitcast {
                return Err(ProgramError::UnsupportedOperation {
                    message: format!("operation `{}` is not transposable in bitcast mode", operation.name()),
                }.into());
            }

            // A live output cotangent converts back to the input's cotangent type. Structural zeros stay structural,
            // and an input with no cotangent space receives nothing.
            check_count!("input", inputs, 1, ProgramError);
            check_count!("output", outputs, 1, ProgramError);
            check_count!("accumulator", accumulators, 1, DifferentiationError);

            let input_cotangent_type = inputs[0].r#type().cotangent()?;
            if input_cotangent_type.is_zero_space() {
                return Ok(());
            }

            let contribution = match &outputs[0] {
                MaybeZero::Zero(_) => MaybeZero::Zero(input_cotangent_type),
                MaybeZero::Value(cotangent) => {
                    MaybeZero::Value(cotangent.unalign_cotangent(&input_cotangent_type)?)
                }
            };

            accumulators[0].accumulate(context, contribution)?;
            Ok(())
        }
    },
}

/// Represents a [`Type`] that has an element [`DataType`] which can be replaced independently of the rest of its
/// structure and placement metadata. Replacing the element type changes metadata only: it does not convert values,
/// validate their representability, or enforce the promotion lattice. [`ConvertElementTypeOperation`] infers its
/// output type through this capability.
///
/// [`with_element_type`](Self::with_element_type) returns the requested [`DataType`] itself for a bare data type and
/// preserves an [`ArrayType`]'s shape, sharding, memory space, and element-based tiled layout, clearing a byte-stride
/// layout only when the element storage width changes, since its offsets need not accommodate the new elements.
/// [`with_bitcast_element_type`](Self::with_bitcast_element_type) instead conserves encoding bits by adjusting the
/// trailing shape, which a bare data type cannot describe, so bare data types support only equal-width
/// reinterpretation.
///
/// # Example
///
/// ```rust
/// # use ryft_core::{ArrayType, DataType, ElementType, Shape};
/// let input = ArrayType::new(DataType::F64, Shape::new(vec![2.into(), 3.into()]));
/// let output = input.with_element_type(DataType::F32);
/// assert_eq!(output.element_type(), DataType::F32);
/// assert_eq!(output.shape(), input.shape());
/// assert_eq!(input.element_type(), DataType::F64);
/// ```
pub trait ElementType: Type {
    /// Returns the element [`DataType`].
    fn element_type(&self) -> DataType;

    /// Returns a type derived from this one with `data_type` as its element type, preserving structural and placement
    /// metadata while clearing byte-stride layouts that a changed element storage width invalidates. This does not
    /// validate the numerical conversion itself; [`ConvertElementType::convert_element_type`] converts actual values.
    ///
    /// # Parameters
    ///
    ///   - `data_type`: Element type of the returned type.
    fn with_element_type(&self, data_type: DataType) -> Self;

    /// Returns the type obtained by reinterpreting the encoding bits of this type's elements as `data_type`. Array
    /// types add or remove a trailing axis to conserve the meaningful encoding bits, whereas bare data types support
    /// only equal-width reinterpretation. Element types without a bit representation, boolean or complex elements
    /// paired with a different type, widths that do not divide each other, trailing extents that do not match the
    /// width ratio, and partitioned trailing axes that would have to be consumed are rejected with a [`TypeError`].
    ///
    /// # Parameters
    ///
    ///   - `data_type`: Element type of the returned type.
    fn with_bitcast_element_type(&self, data_type: DataType) -> Result<Self, TypeError>;
}

impl ElementType for DataType {
    #[inline]
    fn element_type(&self) -> DataType {
        *self
    }

    #[inline]
    fn with_element_type(&self, data_type: DataType) -> Self {
        data_type
    }

    fn with_bitcast_element_type(&self, data_type: DataType) -> Result<Self, TypeError> {
        let output = ArrayType::scalar(*self).with_bitcast_element_type(data_type)?;
        if output.rank() != 0 {
            return Err(TypeError::invalid("a bare element type cannot represent rank-changing bit reinterpretation"));
        }
        Ok(data_type)
    }
}

impl ElementType for ArrayType {
    #[inline]
    fn element_type(&self) -> DataType {
        self.data_type()
    }

    fn with_element_type(&self, data_type: DataType) -> Self {
        let output = self.clone().with_data_type(data_type);
        if matches!(self.layout(), Some(Layout::Strided(_)))
            && ArrayAddressing::element_byte_width_for_data_type(self.data_type())
                != ArrayAddressing::element_byte_width_for_data_type(data_type)
        {
            output.with_layout(None)
        } else {
            output
        }
    }

    fn with_bitcast_element_type(&self, data_type: DataType) -> Result<Self, TypeError> {
        let source = self.data_type();
        if source.is_token() || source.is_zero() || data_type.is_token() || data_type.is_zero() {
            return Err(TypeError::invalid(format!(
                "`{CONVERT_ELEMENT_TYPE_OPERATION_NAME}` in bitcast mode requires element types \
                 with a bit representation",
            )));
        }

        if source != data_type
            && (source == DataType::Boolean
                || data_type == DataType::Boolean
                || source.is_complex()
                || data_type.is_complex())
        {
            return Err(TypeError::invalid(format!(
                "`{CONVERT_ELEMENT_TYPE_OPERATION_NAME}` in bitcast mode requires identical source \
                 and destination types for boolean or complex elements",
            )));
        }

        let input_bits = source.bit_width();
        let output_bits = data_type.bit_width();
        if !input_bits.max(output_bits).is_multiple_of(input_bits.min(output_bits)) {
            return Err(TypeError::invalid(format!(
                "`{CONVERT_ELEMENT_TYPE_OPERATION_NAME}` in bitcast mode requires one element bit width \
                 to divide the other",
            )));
        }

        let output = if input_bits > output_bits {
            self.with_inserted_dimension(self.rank(), Dimension::Static(input_bits / output_bits))?
        } else if input_bits < output_bits {
            let ratio = output_bits / input_bits;
            if self.shape().dimensions().last() != Some(&Dimension::Static(ratio)) {
                return Err(TypeError::invalid(format!(
                    "`{CONVERT_ELEMENT_TYPE_OPERATION_NAME}` in bitcast mode requires a trailing dimension \
                     of size {ratio} when widening elements",
                )));
            }

            // Every piece of an output element must live on one device. Dropping a partitioned trailing axis here
            // would silently require a gather, including over manual mesh axes whose placement would otherwise turn
            // into varying-axis metadata.
            let trailing_dimension = self.sharding().and_then(|sharding| sharding.dimensions().last());
            if let Some(ShardingDimension::Sharded(_) | ShardingDimension::Unconstrained) = trailing_dimension {
                return Err(TypeError::invalid(format!(
                    "`{CONVERT_ELEMENT_TYPE_OPERATION_NAME}` in bitcast mode requires a replicated trailing dimension \
                     when widening elements",
                )));
            }

            self.without_dimension(self.rank() - 1)?.0
        } else {
            self.clone()
        };

        Ok(output.with_data_type(data_type))
    }
}

/// Converts elements numerically or reinterprets their encoding bits as another [`DataType`]. Numerical conversion
/// preserves shape, sharding, and memory space. Byte-stride layouts are cleared when the element storage width
/// changes; tiled layouts are preserved. Numerical conversion may narrow precision or change numerical category.
/// [`Self::bitcast_element_type`] instead preserves encoding bits and changes the trailing shape when source and
/// destination bit widths differ. [`Self::promote_element_type`] additionally checks that the requested conversion
/// is permitted by the type promotion lattice.
///
/// For the reference [`Array`] backend, [`ArrayElement::convert_to`](crate::ArrayElement::convert_to) defines
/// per-element rounding, truncation, saturation, and exceptional-value handling. [`Array::converted_to`] documents
/// the handling of same-type conversions, tokens, and structural zeros. Type inference rejects token conversions and
/// numerical conversions between structural zero and materialized element types.
///
/// [`ConvertElementType`] is the value capability that stages [`ConvertElementTypeOperation`]. Context-carrying values
/// bind that operation through their own context, except for a numerical conversion to the input's own element type,
/// which returns the input once validated, whereas concrete values execute their backend's conversion directly.
///
/// The universe parameter `T` defaults to the [`Capability`] universe of the implementor, so that homogeneous array
/// values implement this capability for [`ArrayType`] and composite array IR values implement it for [`ArrayIrType`].
///
/// # Examples
///
/// ```rust
/// # use ryft_core::{Array, ConvertElementType, DataType, ProgramError};
/// #
/// # fn main() -> Result<(), ProgramError> {
/// let input = Array::vector(vec![1.75f64, -2.5]).unwrap();
/// let output = input.convert_element_type(DataType::I32)?;
/// assert_eq!(output, Array::vector(vec![1i32, -2]).unwrap());
///
/// let input = Array::vector(vec![1.0f32, 2.0]).unwrap();
/// assert_eq!(input.promote_element_type(DataType::F64)?, Array::vector(vec![1.0f64, 2.0]).unwrap());
/// # Ok(())
/// # }
/// ```
#[capability(projection(ArrayIrType => ArrayType))]
pub trait ConvertElementType<T = <Self as Capability>::Universe>: Capability + Sized {
    /// Converts each element to `data_type`, following [`ElementType::with_element_type`] for layout and placement
    /// metadata. Narrowing and conversions between numerical categories are allowed, subject to the backend's element
    /// conversion rules. Real-to-integer conversion truncates toward zero and saturates in the destination range.
    /// Integer-to-integer narrowing retains the low bits. Complex-to-Boolean conversion tests both components for
    /// nonzero, while other noncomplex destinations discard the imaginary component. Finite-only microscaling formats
    /// map NaN to positive maximum finite and saturate infinities. [`f8e8m0fnu`](crate::f8e8m0fnu) maps zero to NaN.
    /// The input is unchanged and the result carries the requested element type. Unsupported conversions return a
    /// [`ProgramError`].
    ///
    /// # Parameters
    ///
    ///   - `data_type`: Destination element type. For example, converting `f64` elements to `i32` truncates fractional
    ///     parts toward zero; converting to a lower-precision floating-point format may round the values.
    fn convert_element_type(&self, data_type: DataType) -> Result<Self, ProgramError>;

    /// Reinterprets this array as `data_type`. Equal bit widths retain the shape. Narrowing adds a trailing axis with
    /// one element per piece of the input encoding; widening combines the trailing axis into one output element and
    /// requires its static size to equal the output-to-input bit-width ratio. The widths must divide exactly.
    ///
    /// Core eager arrays order pieces from least to most significant bits, including padded sub-byte encodings.
    /// Equal-width conversions preserve layout; rank changes clear layout and preserve compatible sharding and memory.
    /// Widening requires the consumed axis to be replicated when sharding is specified. Partitioned or unconstrained
    /// trailing axes must be resharded first so all pieces of each output element are available on the same device.
    /// Boolean and complex elements support only identity reinterpretation. Tokens and structural zeros are rejected.
    /// Reinterpretation has a zero derivative, including when source and destination element types are equal. In the
    /// Ryft XLA backend, XLA lowering supports equal-width casts without finite dimension bounds. Rank-changing casts
    /// require finite bounds for physical allocation but this does not restrict core eager or symbolic type inference.
    ///
    /// # Example
    ///
    /// ```
    /// # use ryft_core::{Array, ConvertElementType, DataType};
    /// let pieces = Array::scalar(0x12345678u32).unwrap().bitcast_element_type(DataType::U16)?;
    /// assert_eq!(pieces, Array::vector(vec![0x5678u16, 0x1234]).unwrap());
    /// assert_eq!(pieces.bitcast_element_type(DataType::U32)?, Array::scalar(0x12345678u32).unwrap());
    /// # Ok::<(), ryft_core::ProgramError>(())
    /// ```
    ///
    /// # Parameters
    ///
    ///   - `data_type`: Destination element encoding; it determines both the values and any trailing-axis change.
    fn bitcast_element_type(&self, data_type: DataType) -> Result<Self, ProgramError>;

    /// Converts each element to `data_type` after checking [`DataType::promote_to`]. A conversion outside the promotion
    /// lattice returns a [`TypeError`] before any conversion is dispatched; an accepted conversion delegates to
    /// [`ConvertElementType::convert_element_type`]. Promotion follows the lattice's numerical-category rules and does
    /// not guarantee exact representation of every source value.
    ///
    /// # Parameters
    ///
    ///   - `data_type`: Requested destination type, such as `f64` for an `f32` input. The destination must be reachable
    ///     from the source in the promotion lattice. The same type is accepted; narrowing from `f64` to `f32` is not.
    #[inline]
    fn promote_element_type(&self, data_type: DataType) -> Result<Self, ProgramError>
    where
        Self: Typed<Type: ElementType>,
    {
        self.r#type()
            .element_type()
            .promote_to(data_type)
            .map_err(|error| TypeError::invalid(error.to_string()))?;
        self.convert_element_type(data_type)
    }
}

impl<T: ElementType, V: Value<Type = T, Dispatch = ValueDomainDispatch>> ConvertElementType<T> for V
where
    V::Domain: Context<Operation: From<ConvertElementTypeOperation<T>>>,
{
    fn convert_element_type(&self, data_type: DataType) -> Result<Self, ProgramError> {
        let operation = ConvertElementTypeOperation::<V::Type>::new(data_type, false);
        let input_type = self.r#type();
        Operation::infer_output_types(&operation, std::slice::from_ref(input_type.as_ref()), &[])?;
        if input_type.element_type() == data_type {
            return Ok(self.clone());
        }
        let mut outputs = self.domain().bind(operation, Vec::new(), std::slice::from_ref(self))?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }

    fn bitcast_element_type(&self, data_type: DataType) -> Result<Self, ProgramError> {
        // Even an identity bitcast must be staged. Its declared derivative is zero rather than the identity.
        let operation = ConvertElementTypeOperation::<V::Type>::new(data_type, true);
        let mut outputs = self.domain().bind(operation, Vec::new(), std::slice::from_ref(self))?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

impl ConvertElementType for Array {
    #[inline]
    fn convert_element_type(&self, data_type: DataType) -> Result<Self, ProgramError> {
        self.converted_to(data_type)
    }

    fn bitcast_element_type(&self, data_type: DataType) -> Result<Self, ProgramError> {
        let output_type = ConvertElementTypeOperation::<ArrayType>::new(data_type, true)
            .infer_output_types(&[self.r#type().into_owned()], &[])?
            .remove(0);
        if self.r#type().data_type() == data_type {
            return Ok(self.clone());
        }

        let input_bits = self.r#type().data_type().bit_width();
        let output_bits = data_type.bit_width();
        let bytes = self.logical_bytes();
        if input_bits >= 8 && output_bits >= 8 || input_bits == output_bits {
            return Array::from_logical_bytes(output_type, &bytes);
        }

        // Sub-byte host elements occupy separate padded bytes, so the meaningful bits are walked in logical order and
        // the padding is restored at each output element boundary instead of being reinterpreted as input data.
        let output_count = Array::element_count(&output_type)?;
        let input_bytes = input_bits.div_ceil(8);
        let output_bytes = output_bits.div_ceil(8);
        let mut output = vec![0u8; output_count * output_bytes];
        for (output_index, element) in output.chunks_exact_mut(output_bytes).enumerate() {
            for bit in 0..output_bits {
                let source_bit = output_index * output_bits + bit;
                let source_index = source_bit / input_bits;
                let source_offset = source_bit % input_bits;
                let value = (bytes[source_index * input_bytes + source_offset / 8] >> (source_offset % 8)) & 1;
                element[bit / 8] |= value << (bit % 8);
            }
        }

        Array::from_logical_bytes(output_type, &output)
    }
}

/// Canonical operation name for [`ReducePrecisionOperation`].
pub const REDUCE_PRECISION_OPERATION_NAME: &str = "reduce_precision";

/// [`Operation`] that rounds each real floating-point element of its input to the values
/// representable by a floating-point format with [`exponent_bits`](Self::exponent_bits) exponent bits and
/// [`mantissa_bits`](Self::mantissa_bits) mantissa bits, while keeping the input's element type, shape, sharding, and
/// memory space. This simulates the precision of a narrower format (e.g., `bfloat16` inside `f32` arithmetic) without
/// changing how the values are stored. Type inference goes through the [`ElementType`] capability of the type
/// universe `T`, so the same operation serves bare [`DataType`]s and [`ArrayType`]s. Refer to the documentation
/// of [`ReducePrecision`] for the rounding semantics.
///
/// Interpretation reduces the values through the same capability, and batching keeps the mapped axis in place.
/// Differentiation treats the operation as linear and as its own transpose, although it is piecewise constant,
/// so that derivatives carry the same simulated precision as the primal computation (this is also the rule of
/// [`jax.lax.reduce_precision`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.reduce_precision.html)):
/// the tangent and the cotangent are both reduced to the same precision as the primal, and symbolic zeros stay
/// symbolic. Backends lower the operation to their precision-reduction constructs (e.g.,
/// [`stablehlo.reduce_precision`](https://openxla.org/stablehlo/spec#reduce_precision) in the XLA backend).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ReducePrecisionOperation<T: ElementType> {
    /// Refer to the documentation of [`exponent_bits`](Self::exponent_bits) for more information.
    exponent_bits: u32,

    /// Refer to the documentation of [`mantissa_bits`](Self::mantissa_bits) for more information.
    mantissa_bits: u32,

    /// [`PhantomData`] marker tying this [`Operation`] to the [`Type`] universe in which it is valid.
    marker: PhantomData<fn() -> T>,
}

impl<T: ElementType> ReducePrecisionOperation<T> {
    /// Creates a new [`ReducePrecisionOperation`] that rounds its input to `exponent_bits` exponent bits and
    /// `mantissa_bits` mantissa bits. The bit counts are validated during type inference rather than here.
    ///
    /// # Parameters
    ///
    ///   - `exponent_bits`: Number of exponent bits of the simulated format, which must be at least one.
    ///   - `mantissa_bits`: Number of explicitly stored mantissa bits of the simulated format (i.e., excluding the
    ///     implicit leading bit), which may be zero.
    #[inline]
    pub fn new(exponent_bits: u32, mantissa_bits: u32) -> Self {
        Self { exponent_bits, mantissa_bits, marker: PhantomData }
    }

    /// Returns the number of exponent bits of the format simulated by this [`ReducePrecisionOperation`].
    #[inline]
    pub fn exponent_bits(&self) -> u32 {
        self.exponent_bits
    }

    /// Returns the number of explicitly stored mantissa bits of the format simulated by this
    /// [`ReducePrecisionOperation`].
    #[inline]
    pub fn mantissa_bits(&self) -> u32 {
        self.mantissa_bits
    }
}

// Implemented manually because deriving `Copy` would require `T: Copy`, although the marker is `Copy` for every `T`.
impl<T: ElementType> Copy for ReducePrecisionOperation<T> {}

impl<T: ElementType> Display for ReducePrecisionOperation<T> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<T: ElementType> Operation for ReducePrecisionOperation<T> {
    type Type = T;

    #[inline]
    fn name(&self) -> &'static str {
        REDUCE_PRECISION_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[T],
        region_interfaces: &[RegionInterface<T>],
    ) -> Result<Vec<T>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        check_count!("region", region_interfaces, 0, TypeError);
        let data_type = input_types[0].element_type();
        if !data_type.is_floating_point() {
            return Err(TypeError::invalid(format!(
                "`{REDUCE_PRECISION_OPERATION_NAME}` requires real floating-point elements but got `{data_type}`",
            )));
        }
        if self.exponent_bits == 0 {
            return Err(TypeError::invalid(format!(
                "`{REDUCE_PRECISION_OPERATION_NAME}` requires at least one exponent bit",
            )));
        }
        Ok(vec![input_types[0].clone()])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("exponent_bits", self.exponent_bits)?;
            operation.field("mantissa_bits", self.mantissa_bits)
        })
    }
}

impl<C: Domain<Type: ElementType, Value: ReducePrecision>> InterpretableOperation<C>
    for ReducePrecisionOperation<C::Type>
{
    #[inline]
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![inputs[0].reduce_precision(self.exponent_bits, self.mantissa_bits)?])
    }
}

impl<C: Context<Type: ElementType, Operation: From<ReducePrecisionOperation<C::Type>>>> PartiallyEvaluatableOperation<C>
    for ReducePrecisionOperation<C::Type>
{
}

impl<C: Context<Type = ArrayType, Value: ReducePrecision>, P: ArrayExtentBatchingPolicy<C>>
    BatchableOperation<C, ArrayBatchingPolicy<P>> for ReducePrecisionOperation<ArrayType>
{
    fn batch<D: BatchingDriver<C, ArrayBatchingPolicy<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatchingPolicy<P>>, BatchingError> {
        // Precision reduction is elementwise, so it applies to the packed value directly, keeping the mapped axis and
        // any ragged axes in place. Rounding padding elements is harmless because they remain padding.
        check_count!("input", inputs, 1, ProgramError);
        self.infer_output_types(&[inputs[0].unbatched_type()], &[])?;
        let mut outputs = self.interpret_with_batch_axes(context, inputs, &[inputs[0].batch_axis()])?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(vec![outputs.remove(0).with_ragged_axes(inputs[0].ragged_axes().to_vec())?].into())
    }
}

impl_differentiable_operation! {
    <T> ReducePrecisionOperation<T>,
    jvp<C>
    where
        T: DifferentiableType + ElementType,
        C: Context<Type = T, Value: ReducePrecision>,
    {
        |operation, _context, _driver, inputs| {
            // Precision reduction is treated as linear, so a live tangent is reduced to the same precision as the
            // primal and a symbolic-zero tangent stays symbolic.
            check_count!("input", inputs, 1, ProgramError);
            let primal = inputs[0].primal().reduce_precision(operation.exponent_bits, operation.mantissa_bits)?;
            let tangent = match inputs[0].tangent() {
                MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().tangent()?),
                MaybeZero::Value(tangent) => {
                    MaybeZero::Value(tangent.reduce_precision(operation.exponent_bits, operation.mantissa_bits)?)
                }
            };
            Ok(vec![DifferentiationDual::new(primal, tangent)?])
        }
    },
    transpose<V, O>
    where
        T: DifferentiableType + ElementType,
        V: Value<Type = T>,
        O: Operation<Type = T> + From<ReducePrecisionOperation<T>>,
    {
        |operation, context, _driver, inputs, outputs, accumulators| {
            // The operation is its own transpose, so a live output cotangent is reduced to the same precision before
            // it is accumulated into the input cotangent. Symbolic-zero cotangents contribute nothing.
            check_count!("input", inputs, 1, ProgramError);
            check_count!("output", outputs, 1, ProgramError);
            check_count!("accumulator", accumulators, 1, DifferentiationError);
            let MaybeZero::Value(cotangent) = &outputs[0] else { return Ok(()); };
            let mut contributions =
                context.stage_operation(*operation, Vec::new(), std::slice::from_ref(cotangent))?;
            check_count!("output", contributions, 1, ProgramError);
            accumulators[0].accumulate(context, MaybeZero::Value(contributions.remove(0)))?;
            Ok(())
        }
    },
}

/// Rounds real floating-point elements to the values representable by a narrower floating-point format while keeping
/// their element type, which simulates the precision of that format (e.g., `bfloat16` inside `f32` arithmetic).
///
/// The semantics are those of [`stablehlo.reduce_precision`](https://openxla.org/stablehlo/spec#reduce_precision)
/// as implemented by XLA, applied to the bit encoding of each element in its own [`DataType`]:
///
///   - The mantissa is rounded to `mantissa_bits` bits, to nearest with ties to even. Components whose requested bit
///     count is at least the element type's own are left unchanged.
///   - Values whose rounded exponent exceeds the largest finite exponent of the reduced format overflow to an
///     infinity with the input's sign.
///   - Values whose exponent is at or below the smallest exponent of the reduced format, including every subnormal
///     of the reduced format, are flushed to a zero with the input's sign rather than being rounded to a subnormal.
///   - NaN inputs are returned unchanged, even when `mantissa_bits = 0` leaves the reduced format without NaNs.
///     Infinities and signed zeros are preserved.
///
/// [`ReducePrecision`] is the value capability that stages [`ReducePrecisionOperation`]. Context-carrying values bind
/// that operation through their own context, whereas concrete values execute their backend's reduction directly. For
/// the reference [`Array`] backend, reduction supports [`DataType::F16`], [`DataType::BF16`], [`DataType::F32`], and
/// [`DataType::F64`] elements.
///
/// The universe parameter `T` defaults to the [`Capability`] universe of the implementor, so that homogeneous array
/// values implement this capability for [`ArrayType`] and composite array IR values implement it for [`ArrayIrType`].
///
/// # Example
///
/// ```rust
/// # use ryft_core::{Array, ProgramError, ReducePrecision};
/// # fn main() -> Result<(), ProgramError> {
/// // With `bfloat16` precision (8 exponent bits and 7 mantissa bits), `1 + 2^-8` is a tie that rounds to the even
/// // mantissa of `1`, whereas `1 + 3 * 2^-8` is a tie that rounds up to `1 + 2^-6`. `1e38` keeps its exponent.
/// let input = Array::vector(vec![1.00390625f32, 1.01171875, 1e38])?;
/// let expected = Array::vector(vec![1.0f32, 1.015625, f32::from_bits(0x7e960000)])?;
/// assert_eq!(input.reduce_precision(8, 7)?, expected);
///
/// // With `float16` precision (5 exponent bits and 10 mantissa bits), `1e38` overflows to infinity.
/// let expected = Array::vector(vec![1.00390625f32, 1.01171875, f32::INFINITY])?;
/// assert_eq!(input.reduce_precision(5, 10)?, expected);
/// # Ok(())
/// # }
/// ```
#[capability(projection(ArrayIrType => ArrayType))]
pub trait ReducePrecision<T = <Self as Capability>::Universe>: Capability + Sized {
    /// Returns this value with each element rounded to a floating-point format with `exponent_bits` exponent bits and
    /// `mantissa_bits` mantissa bits. Refer to the documentation of [`ReducePrecision`] for the rounding semantics.
    /// Non-floating-point elements and `exponent_bits = 0` return a [`ProgramError`].
    ///
    /// # Parameters
    ///
    ///   - `exponent_bits`: Number of exponent bits of the simulated format, which must be at least one.
    ///   - `mantissa_bits`: Number of explicitly stored mantissa bits of the simulated format (i.e., excluding the
    ///     implicit leading bit), which may be zero.
    fn reduce_precision(&self, exponent_bits: u32, mantissa_bits: u32) -> Result<Self, ProgramError>;
}

impl<
    T: ElementType,
    V: Value<Type = T, Dispatch = ValueDomainDispatch, Domain: Context<Operation: From<ReducePrecisionOperation<T>>>>,
> ReducePrecision<T> for V
{
    fn reduce_precision(&self, exponent_bits: u32, mantissa_bits: u32) -> Result<Self, ProgramError> {
        let operation = ReducePrecisionOperation::<V::Type>::new(exponent_bits, mantissa_bits);
        let mut outputs = self.domain().bind(operation, Vec::new(), std::slice::from_ref(self))?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

impl ReducePrecision for Array {
    fn reduce_precision(&self, exponent_bits: u32, mantissa_bits: u32) -> Result<Self, ProgramError> {
        let output_type = ReducePrecisionOperation::<ArrayType>::new(exponent_bits, mantissa_bits)
            .infer_output_types(&[self.r#type().into_owned()], &[])?
            .remove(0);

        // Reduce an IEEE-style encoding in the low `bit_width` bits, preserving its storage format, following OpenXLA's
        // [`reducePrecision`](https://github.com/openxla/xla/blob/main/xla/mlir_hlo/mhlo/transforms/transformation_helpers.h),
        // used by the CPU backend's MLIR-based fusion emitters: round the mantissa with a ties-to-even bias and
        // truncate (including a carry into the exponent), select signed infinity or zero from the rounded exponent,
        // and restore NaN inputs. The legacy `EmitReducePrecisionIR` in `xla/service/llvm_ir/llvm_util.cc` computes
        // the same bits except that it maps NaNs to positive infinity when `mantissa_bits = 0`. Input bits above
        // `bit_width` must be cleared; `source_mantissa_bits` excludes the implicit leading bit. Type inference
        // has already validated that `exponent_bits` is positive.
        let reduce = |bits: u64, bit_width: u32, source_mantissa_bits: u32| {
            let width_mask = u64::MAX >> (64 - bit_width);
            let source_exponent_bits = bit_width - 1 - source_mantissa_bits;
            let sign_mask = 1u64 << (bit_width - 1);
            let exponent_mask = ((1u64 << source_exponent_bits) - 1) << source_mantissa_bits;
            let is_nan = (bits & !sign_mask) > exponent_mask;

            let mut result = bits;
            if mantissa_bits < source_mantissa_bits {
                // The rounding bias is `0111...` below the last retained mantissa bit, plus one when that bit is set,
                // so that adding it and truncating rounds to nearest with ties to even.
                let shift = source_mantissa_bits - mantissa_bits;
                let last_mantissa_bit = 1u64 << shift;
                let rounding_bias = (last_mantissa_bit >> 1) - 1 + ((result & last_mantissa_bit) >> shift);
                result = result.wrapping_add(rounding_bias) & width_mask & !(last_mantissa_bit - 1);
            }

            if exponent_bits < source_exponent_bits {
                // A biased exponent of `2^(n - 1) - 1` denotes `1.0` for every exponent width `n`, so the reduced
                // format's largest finite and smallest (i.e., zero or subnormal) exponents are that bias offset by
                // the reduced bias.
                let exponent_bias = (1u64 << (source_exponent_bits - 1)) - 1;
                let reduced_exponent_bias = (1u64 << (exponent_bits - 1)) - 1;
                let reduced_max_exponent = (exponent_bias + reduced_exponent_bias) << source_mantissa_bits;
                let reduced_min_exponent = (exponent_bias - reduced_exponent_bias) << source_mantissa_bits;
                let exponent = result & exponent_mask;
                let signed_zero = result & sign_mask;
                if exponent > reduced_max_exponent {
                    result = signed_zero | exponent_mask;
                } else if exponent <= reduced_min_exponent {
                    result = signed_zero;
                }
            }

            // The exponent handling turns NaNs into infinities and mantissa rounding may carry a NaN payload into the
            // sign bit, so NaN inputs are restored here.
            if is_nan { bits } else { result }
        };

        match output_type.data_type() {
            DataType::F16 => self.map_elements::<f16, f16>(output_type, |value| {
                Ok(f16::from_bits(reduce(value.to_bits().into(), 16, 10) as u16))
            }),
            DataType::BF16 => self.map_elements::<bf16, bf16>(output_type, |value| {
                Ok(bf16::from_bits(reduce(value.to_bits().into(), 16, 7) as u16))
            }),
            DataType::F32 => self.map_elements::<f32, f32>(output_type, |value| {
                Ok(f32::from_bits(reduce(value.to_bits().into(), 32, 23) as u32))
            }),
            DataType::F64 => {
                self.map_elements::<f64, f64>(output_type, |value| Ok(f64::from_bits(reduce(value.to_bits(), 64, 52))))
            }
            data_type => Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "`{REDUCE_PRECISION_OPERATION_NAME}` is not supported for `{data_type}` elements \
                     of reference arrays",
                ),
            }),
        }
    }
}

/// Defines which output types a [`CastOperation`] can use for this type family. For arrays, the input and output
/// must agree on their element [`DataType`], [`Shape`](crate::Shape) rank, and [`Memory`](crate::Memory). Optional
/// [`Layout`] and [`Sharding`](crate::Sharding) constraints must agree only when both types specify them. An
/// unspecified field imposes no constraint. For example, an input specifying a layout and an output specifying a
/// sharding can both describe a value that carries that layout and sharding; two different specified layouts cannot.
/// This differs from requiring one complete type to refine the other.
///
/// Custom type families implement this capability to use [`CastOperation`]. The operation validates dimension
/// extents and identity relationships separately through [`TypeRefinements`].
pub trait CastType: Type {
    /// Validates the requirements on `output_type` described by this trait. [`CastOperation`] separately validates
    /// dimension extents and identity relationships; this method does not inspect runtime values.
    fn validate_cast_output_type(&self, output_type: &Self) -> Result<(), TypeError>;
}

impl CastType for DataType {
    #[inline]
    fn validate_cast_output_type(&self, output_type: &Self) -> Result<(), TypeError> {
        if self == output_type {
            Ok(())
        } else {
            Err(TypeError::invalid(format!(
                "`{CAST_OPERATION_NAME}` cannot convert type `{self}` to unrelated type `{output_type}`",
            )))
        }
    }
}

impl CastType for ArrayType {
    fn validate_cast_output_type(&self, output_type: &Self) -> Result<(), TypeError> {
        let layout_matches = self.layout().zip(output_type.layout()).is_none_or(|(input, output)| input == output);
        let sharding_matches =
            self.sharding().zip(output_type.sharding()).is_none_or(|(input, output)| input == output);
        if self.data_type() == output_type.data_type()
            && self.rank() == output_type.rank()
            && self.memory() == output_type.memory()
            && layout_matches
            && sharding_matches
        {
            Ok(())
        } else {
            Err(TypeError::invalid(format!(
                "`{CAST_OPERATION_NAME}` cannot convert type `{self}` to unrelated type `{output_type}`",
            )))
        }
    }
}

impl CastType for DimensionType {
    #[inline]
    fn validate_cast_output_type(&self, _output_type: &Self) -> Result<(), TypeError> {
        // A first-class dimension only needs the dimension and identity checks performed by refinement validation.
        Ok(())
    }
}

impl CastType for ArrayIrType {
    fn validate_cast_output_type(&self, output_type: &Self) -> Result<(), TypeError> {
        match (self, output_type) {
            (Self::Array(input), Self::Array(output)) => input.validate_cast_output_type(output),
            (Self::Dimension(input), Self::Dimension(output)) => input.validate_cast_output_type(output),
            _ => Err(TypeError::invalid(format!(
                "`{CAST_OPERATION_NAME}` cannot convert type `{self}` to unrelated type `{output_type}`",
            ))),
        }
    }
}

// TODO(eaplatanios): Review from here onwards.

/// Canonical operation name for [`CastOperation`].
pub const CAST_OPERATION_NAME: &str = "cast";

/// [`Operation`] that returns its first input at a different type representation (e.g., `f32[n]` as `f32[2]`). Without
/// an explicit [relation](Self::relation), one type must refine the other (e.g., `f32[n]` and `f32[2]`, or
/// `dimension<m>` and `dimension<2>`). A relation also permits renamed identities and casts that narrow some components
/// while widening others, but does not permit incompatible layouts, shardings, or memory placements.
/// Further inputs are _witnesses_: values whose types make the output's identities available to
/// the instruction (e.g., a value of type `f32[n]` lets `f32[2]` widen to `f32[n]`). Witnesses carry no derivatives
/// and are otherwise unused. The output type may not define a type identity.
///
/// A cast states a relationship between its first input at the output type and its witnesses at their
/// [declared types](Self::witness_types): every relationship that their shared identities express (e.g., widening
/// `f32[2]` to `f32[n]` next to a witness of type `f32[n]` states that `n = 2`, and so does narrowing `f32[n]` to
/// `f32[2]`). Type inference rejects relationships that the static types already contradict, and interpretation checks
/// the complete relationship over the concrete types of its inputs (e.g., a witness of extent 5 rejects the widening
/// above). With an explicit relation, it also retains the [declared input type](Self::input_type), so the original
/// identity assignments survive concrete execution. For example, casting `f32[a, b]` to `f32[b, a]` requires equal
/// axis extents; a value with shape `[2, 3]` cannot change `a = 2, b = 3` into `a = 3, b = 2`.
/// Staged and compiled programs differ by mode:
///
///   - A [checked](Self::checked) cast requires the relationship at runtime. It declares
///     [`EffectClass::OrderedAssertion`], so it is ordered like an assertion and is never removed as dead code, and a
///     violated relationship fails the program instead of producing a value.
///   - An [unchecked](Self::unchecked) cast assumes the relationship. It is pure and has no runtime cost beyond the
///     representation change (e.g., none at all for first-class dimensions). Transforms emit unchecked casts right
///     after proving their relationship: a [`GuardOperation`](crate::operations::assertions::GuardOperation) converts
///     the tangents of its outputs with them, and the transpose of a cast is the unchecked cast of its output cotangent
///     back to its input's cotangent type, with witnesses that carry the required input identities.
///
/// The [`Cast`] capability applies both modes to values. Transposition assumes the forward relationship has already
/// been established. Identities needed by the input cotangent must be carried by a live output cotangent, a known
/// witness, or a retained dimension source. For example, directly transposing `f32[n]` to `f32[2]` without any saved
/// value carrying `n` is unsupported. Linearize the program before transposing it to retain the primal witnesses and
/// execute its checks; the resulting tangent program uses unchecked casts.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CastOperation<T: Type> {
    /// Refer to the documentation of [`output_type`](Self::output_type) for more information.
    output_type: T,

    /// Refer to the documentation of [`witness_types`](Self::witness_types) for more information.
    witness_types: Vec<T>,

    /// Declared first input type and local relation, boxed to keep operation families from storing both types inline.
    /// Refer to [`Self::input_type`] and [`Self::relation`].
    relation: Option<Box<(T, T)>>,

    /// Refer to the documentation of [`is_checked`](Self::is_checked) for more information.
    checked: bool,
}

impl<T: Type> CastOperation<T> {
    /// Creates a new checked [`CastOperation`] that returns its first input at `output_type`, with one further witness
    /// input for each of the `witness_types`. Staged and compiled programs check the relationship that the cast states
    /// at runtime (refer to the type documentation).
    #[inline]
    pub fn checked(output_type: T, witness_types: Vec<T>) -> Self {
        Self { output_type, witness_types, relation: None, checked: true }
    }

    /// Creates a new unchecked [`CastOperation`] that returns its first input at `output_type`, with one further
    /// witness input for each of the `witness_types`. Use this only right after proving the relationship that the cast
    /// states (refer to the type documentation), and use [`Self::checked`] otherwise.
    ///
    /// <div class="warning">
    ///
    /// Staged and compiled programs assume the relationship of an unchecked cast; a program that executes one whose
    /// relationship does not hold silently computes with the wrong extents (e.g., reading padding as data).
    ///
    /// </div>
    #[inline]
    pub fn unchecked(output_type: T, witness_types: Vec<T>) -> Self {
        Self { output_type, witness_types, relation: None, checked: false }
    }

    /// Returns this [`CastOperation`] with its [declared input type](Self::input_type) set to `input_type` and its
    /// [local relation](Self::relation) set to `relation`. Retaining the input declaration preserves the original
    /// identity assignments when the operation executes on concrete values (refer to the type documentation).
    #[inline]
    pub fn with_relation(mut self, input_type: T, relation: T) -> Self {
        self.relation = Some(Box::new((input_type, relation)));
        self
    }

    /// Returns the declared type of the first input when this cast has an explicit [relation](Self::relation).
    /// Its identities belong to the surrounding program, like those in the output and witness types. The actual
    /// input must refine this declaration, and its original identity assignments must agree with the output and
    /// witnesses. Ordinary casts without an explicit relation need no separate input declaration.
    #[inline]
    pub fn input_type(&self) -> Option<&T> {
        self.relation.as_deref().map(|(input_type, _)| input_type)
    }

    /// Returns the requested type of the first input's new representation. Type inference preserves identities
    /// carried by the inputs. When concrete operands no longer carry names from the input or witness declarations,
    /// it substitutes their known exact extents so they do not become unbound names in a staged program.
    #[inline]
    pub fn output_type(&self) -> &T {
        &self.output_type
    }

    /// Returns the declared types of the witness inputs, whose identities the relationship that this cast states
    /// shares with the [output type](Self::output_type).
    #[inline]
    pub fn witness_types(&self) -> &[T] {
        &self.witness_types
    }

    /// Returns the type that the first input and the output must both instantiate, if any. Without a relation, one of
    /// the two types must refine the other. A relation also admits casts that rename identities or that narrow some
    /// components while widening others (e.g., `f32[q]` to `f32[n]` when both instantiate a declared `f32[p]` whose
    /// relationships a [`GuardOperation`](crate::operations::assertions::GuardOperation) checked). Like the declared
    /// types of a guard, the relation's identities are local to the cast, so type-identity renamings of the surrounding
    /// program leave it unchanged.
    #[inline]
    pub fn relation(&self) -> Option<&T> {
        self.relation.as_deref().map(|(_, relation)| relation)
    }

    /// Returns `true` if staged and compiled programs check the relationship that this cast states at runtime (refer to
    /// [`Self::checked`] and [`Self::unchecked`]).
    #[inline]
    pub fn is_checked(&self) -> bool {
        self.checked
    }

    /// Returns `value` at `output_type` through an unchecked [`CastOperation`] bound in `context` with the provided
    /// `witnesses` (at their current types) and optional [relation](Self::relation), or `value` itself when it already
    /// has that type, so that no-op conversions stage nothing. Transformation rules use this to bind into the specific
    /// context that they target (e.g., the tangent context of a forward-mode rule), and must have proven the
    /// relationship that the cast states.
    pub(crate) fn bind_unchecked<C: Context<Type = T, Operation: From<Self>>>(
        context: &C,
        value: &C::Value,
        witnesses: &[C::Value],
        output_type: T,
        relation: Option<T>,
    ) -> Result<C::Value, ProgramError> {
        if value.r#type().as_ref() == &output_type {
            return Ok(value.clone());
        }
        let witness_types = witnesses.iter().map(|witness| witness.r#type().into_owned()).collect();
        let inputs = std::iter::once(value).chain(witnesses).cloned().collect::<Vec<_>>();
        let mut operation = Self::unchecked(output_type, witness_types);
        if let Some(relation) = relation {
            operation = operation.with_relation(value.r#type().into_owned(), relation);
        }
        let mut outputs = context.bind(operation, Vec::new(), &inputs)?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }

    /// Checks the complete relationship over `input_types` (the value followed by the witnesses), in both modes.
    /// Concrete extents complete the check; symbolic types must prove the relationship without runtime checks.
    fn validate_concrete_inputs(
        &self,
        input_types: &[T],
        region_interfaces: &[RegionInterface<T>],
    ) -> Result<(), ProgramError>
    where
        T: CastType,
    {
        self.infer_output_types(input_types, region_interfaces)?;
        let refinements = T::Refinements::new(
            std::iter::once(&self.output_type)
                .chain(&self.witness_types)
                .chain(self.input_type())
                .collect::<Vec<_>>(),
            input_types.iter().chain(self.input_type().map(|_| &input_types[0])).collect::<Vec<_>>(),
        )?;
        let mut needs_runtime_checks = !refinements.identities_needing_runtime_checks().is_empty();
        if !needs_runtime_checks && let Some(relation) = self.relation() {
            // The output represents this same value. Check the local relation against its actual input type, rather
            // than treating the requested output type as an independent observation of runtime geometry.
            let local = T::Refinements::from_local_declarations([relation], [&input_types[0]])?;
            needs_runtime_checks = !local.identities_needing_runtime_checks().is_empty();
        }
        if needs_runtime_checks {
            return Err(ProgramError::Concretization {
                message: "`cast` cannot check symbolic type relationships during interpretation; \
                          bind the operation through its context"
                    .to_owned(),
            });
        }
        Ok(())
    }
}

impl<T: CastType> Display for CastOperation<T> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<T: CastType> Operation for CastOperation<T> {
    type Type = T;

    #[inline]
    fn name(&self) -> &'static str {
        CAST_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[T],
        region_interfaces: &[RegionInterface<T>],
    ) -> Result<Vec<T>, TypeError> {
        check_count!("input", input_types, 1 + self.witness_types.len(), TypeError);
        check_count!("region", region_interfaces, 0, TypeError);
        let input_type = &input_types[0];
        if let Some(reference) = input_types
            .iter()
            .chain(&self.witness_types)
            .chain([&self.output_type])
            .chain(self.input_type())
            .chain(self.relation())
            .find(|r#type| r#type.is_reference())
        {
            return Err(TypeError::invalid(format!(
                "`{CAST_OPERATION_NAME}` does not support reference type `{reference}`",
            )));
        }
        if self.output_type.identities().any(|(position, _)| position == TypeIdentityPosition::Definition) {
            return Err(TypeError::invalid(format!(
                "`{CAST_OPERATION_NAME}` output type `{}` cannot define a type identity",
                self.output_type,
            )));
        }

        // A relation is a local declaration that the value and the output must both instantiate, through scoped
        // correspondences that admit differently named identities, while the witnesses must instantiate their declared
        // types.
        let (refinements, local) = if let Some((declared_input, relation)) = self.relation.as_deref() {
            let refinements = T::Refinements::new(
                std::iter::once(declared_input).chain(&self.witness_types).collect::<Vec<_>>(),
                input_types.iter(),
            )?;
            // Witness identities belong to the surrounding program, while the relation's identities are local.
            // Apply ambient facts to both sides before checking the local relation, so a concrete witness cannot
            // contradict an extent determined by the value (e.g., output `a = 5` next to a value of extent 3).
            let input = refinements.refine_actual(input_type, &[])?;
            let output = refinements.refine_declared(&self.output_type, &[])?;
            input.validate_cast_output_type(&output)?;
            let local = T::Refinements::from_local_declarations([relation, relation], [&input, &output])?;
            (refinements, Some(local))
        } else {
            // The cast relates the value at the more general of its two types to the witnesses at their declared
            // types. Static types must not contradict that relationship (e.g., a witness of static extent 5 next to
            // widening `f32[2]` to `f32[n]`).
            let (general, specific) = if self.output_type.is_refined_by(input_type) {
                (&self.output_type, input_type)
            } else if input_type.is_refined_by(&self.output_type) {
                (input_type, &self.output_type)
            } else {
                return Err(TypeError::invalid(format!(
                    "`{CAST_OPERATION_NAME}` cannot convert type `{input_type}` to unrelated type `{}`",
                    self.output_type,
                )));
            };
            let refinements = T::Refinements::new(
                std::iter::once(general).chain(&self.witness_types).collect::<Vec<_>>(),
                std::iter::once(specific).chain(&input_types[1..]).collect::<Vec<_>>(),
            )?;
            (refinements, None)
        };

        // Preserve names carried by live operands, including nominal cotangent types. Resolve names lost when
        // declared operands became concrete, but leave names absent from those declarations alone: ordinary graph
        // validation must still reject widening to an identity for which the caller supplied no witness.
        let preserved = self
            .output_type
            .identities()
            .filter_map(|(_, identity)| {
                let carries = |r#type: &T| r#type.identities().any(|(_, available)| available == identity);
                (input_types.iter().any(carries)
                    || !self.input_type().into_iter().chain(&self.witness_types).any(carries))
                .then(|| identity.clone())
            })
            .collect::<Vec<_>>();
        let output = refinements.refine_declared(&self.output_type, &preserved)?;
        let output = match local {
            Some(local) => local.refine_actual(&output, &preserved)?,
            None => output,
        };
        Ok(vec![output])
    }

    #[inline]
    fn effects(&self) -> Cow<'_, Effects> {
        if self.checked {
            Cow::Owned(Effects::explicit(EffectClasses::single(EffectClass::OrderedAssertion)))
        } else {
            Cow::Borrowed(Effects::empty())
        }
    }

    #[inline]
    fn rename_type_identities(&self, renaming: &TypeIdentityRenaming<T::Identity>) -> Result<Self, TypeError> {
        // Unlike the declared types of a guard, the input, output and witness types name identities of the
        // surrounding program, so they follow its renamings. The relation is a local declaration and stays unchanged.
        Ok(Self {
            output_type: self.output_type.rename_identities(renaming)?,
            witness_types: self
                .witness_types
                .iter()
                .map(|r#type| r#type.rename_identities(renaming))
                .collect::<Result<_, _>>()?,
            relation: self
                .relation
                .as_deref()
                .map(|(input_type, relation)| {
                    input_type.rename_identities(renaming).map(|input_type| Box::new((input_type, relation.clone())))
                })
                .transpose()?,
            checked: self.checked,
        })
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("type", &self.output_type)?;
            if let Some(input_type) = self.input_type() {
                operation.field("input_type", input_type)?;
            }
            if !self.witness_types.is_empty() {
                operation.field(
                    "witnesses",
                    format_args!(
                        "[{}]",
                        self.witness_types.iter().map(ToString::to_string).collect::<Vec<_>>().join(", "),
                    ),
                )?;
            }
            if let Some(relation) = self.relation() {
                operation.field("relation", relation)?;
            }
            if self.checked {
                operation.field("checked", true)?;
            }
            Ok(())
        })
    }
}

impl_reference_dischargeable_operation!(@reference_free <T> CastOperation<T> where T: CastType);

impl<C: Domain<Type: CastType>> InterpretableOperation<C> for CastOperation<C::Type> {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        // Concrete values carry their geometry in their types. For symbolic values, conditional facts cannot replace
        // a runtime check: interpretation accepts only relationships already proved by the types, in both modes.
        self.validate_concrete_inputs(
            &inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>(),
            &driver.regions().map(|region| region.interface()).collect::<Vec<_>>(),
        )?;
        Ok(vec![inputs[0].clone()])
    }
}

impl<C: Context<Type: CastType, Operation: From<CastOperation<C::Type>>>> PartiallyEvaluatableOperation<C>
    for CastOperation<C::Type>
{
}

impl<C: Context<Type: CastType, Operation: From<CastOperation<C::Type>>>, P: BatchingTypePolicy<C>>
    BatchableOperation<C, P> for CastOperation<C::Type>
{
    fn batch<D: BatchingDriver<C, P>>(
        &self,
        context: &BatchingContext<C, P>,
        driver: &D,
        inputs: &[P::Batch],
    ) -> Result<BatchedOutputs<C, P>, BatchingError> {
        // The stacked value and witnesses are retyped with the batch axis of each one inserted into its type, and the
        // output keeps the batch axis of the value.
        self.infer_output_types(
            &inputs.iter().map(|input| P::unbatched_type(input).into_owned()).collect::<Vec<_>>(),
            &driver.regions().map(|region| region.interface()).collect::<Vec<_>>(),
        )?;
        let extent = P::batch_extent(context)?;
        let output_type =
            P::batched_declared_type(context, CAST_OPERATION_NAME, &inputs[0], &self.output_type, &extent)?;
        let witness_types = self
            .witness_types
            .iter()
            .zip(&inputs[1..])
            .map(|(witness_type, witness)| {
                P::batched_declared_type(context, CAST_OPERATION_NAME, witness, witness_type, &extent)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut operation = CastOperation { output_type, witness_types, relation: None, checked: self.checked };
        if let Some((input_type, relation)) = self.relation.as_deref() {
            // Only the relation's names are local to this cast. Keep output and witness names in their ambient scope.
            let mut renaming = TypeIdentityRenaming::new();
            for (_, identity) in relation.identities() {
                if renaming.target(identity).is_none() {
                    renaming.insert(identity.clone(), identity.fresh())?;
                }
            }
            let relation = relation.rename_identities(&renaming)?;
            let relation = P::batched_declared_type(context, CAST_OPERATION_NAME, &inputs[0], &relation, &extent)?;
            let input_type = P::batched_declared_type(context, CAST_OPERATION_NAME, &inputs[0], input_type, &extent)?;
            operation = operation.with_relation(input_type, relation);
        }
        let values = inputs.iter().map(|input| P::value(input).clone()).collect::<Vec<_>>();
        let mut outputs = context.parent().bind(operation, Vec::new(), &values)?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(vec![P::batch(outputs.remove(0), P::batch_axis(&inputs[0]))?].into())
    }
}

impl<C: Context<Type = T>, T: DifferentiableType + CastType> DifferentiableOperation<C> for CastOperation<T>
where
    C::Operation: From<CastOperation<T>>,
{
    fn jvp<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        // A cast is the identity on values, so the tangent of its first input is retyped to the output's tangent type
        // by an unchecked cast, behind the primal cast (which checks the relationship when this cast is checked). The
        // primal inputs witness that tangent cast, which keeps the identities of the first input in scope for its
        // transpose. The primal output of a checked cast witnesses it too, so that the tangent conversion depends on
        // the check. Witnesses carry no derivatives.
        let primals = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        let mut outputs = context.primal().bind(self.clone(), Vec::new(), &primals)?;
        check_count!("output", outputs, 1, ProgramError);
        let output = outputs.remove(0);
        Ok(vec![match inputs[0].tangent() {
            MaybeZero::Zero(_) => DifferentiationDual::new_with_zero_tangent(output)?,
            MaybeZero::Value(tangent) => {
                let tangent_type = output.r#type().tangent()?;
                let mut witnesses = primals
                    .iter()
                    .map(|primal| context.primal_to_tangent(primal.clone()))
                    .collect::<Result<Vec<_>, _>>()?;
                if self.checked {
                    witnesses.push(context.primal_to_tangent(output.clone())?);
                }
                let relation = self.relation().map(DifferentiableType::tangent).transpose()?;
                let tangent =
                    CastOperation::bind_unchecked(context.tangent(), tangent, &witnesses, tangent_type, relation)?;
                DifferentiationDual::new(output, tangent)?
            }
        }])
    }
}

impl<
    T: DifferentiableType + CastType,
    V: Value<Type = T>,
    O: Operation<Type = T> + From<AddOperation<T>> + From<CastOperation<T>>,
> TransposableOperation<V, O> for CastOperation<T>
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TranspositionContext<V, O>,
        _driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
        accumulators: &[CotangentAccumulator],
    ) -> Result<(), DifferentiationError> {
        // The transpose of the identity is the identity, at the cotangent type of the linear first input, converted by
        // an unchecked cast because the forward program already established the relationship. The known witnesses keep
        // that type's identities in scope; a linear witness would have no value to witness with.
        check_count!("output", outputs, 1, ProgramError);
        check_count!("accumulator", accumulators, inputs.len(), DifferentiationError);
        let mut witnesses = inputs[1..]
            .iter()
            .map(|witness| {
                witness.as_known().cloned().ok_or_else(|| ProgramError::UnsupportedOperation {
                    message: format!("`{CAST_OPERATION_NAME}` cannot transpose a linear witness"),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let cotangent_type = inputs[0].r#type().cotangent()?;
        let relation = self.relation().map(DifferentiableType::cotangent).transpose()?;
        let contribution = match &outputs[0] {
            MaybeZero::Zero(_) => MaybeZero::Zero(cotangent_type),
            MaybeZero::Value(cotangent) => {
                // The accumulator requires the original nominal type, even when the forward cast fixed its extent.
                // Retained primal geometry can witness that name without changing the accumulator's contract.
                let mut missing = cotangent_type.identities().map(|(_, identity)| identity.clone()).collect::<Vec<_>>();
                for value in std::iter::once(cotangent).chain(&witnesses) {
                    let r#type = value.r#type();
                    missing.retain(|identity| !r#type.identities().any(|(_, available)| available == identity));
                }
                for source in context.dimension_sources() {
                    if missing.is_empty() {
                        break;
                    }
                    let r#type = source.r#type();
                    if !r#type.is_reference() && r#type.identities().any(|(_, identity)| missing.contains(identity)) {
                        missing.retain(|identity| !r#type.identities().any(|(_, available)| available == identity));
                        witnesses.push(source.clone());
                    }
                }
                if let Some(identity) = missing.first() {
                    return Err(ProgramError::UnsupportedOperation {
                        message: format!(
                            "`{CAST_OPERATION_NAME}` transposition needs a known witness for input identity \
                             `{identity}`; linearize the program before transposing it",
                        ),
                    }
                    .into());
                }
                MaybeZero::Value(CastOperation::bind_unchecked(
                    &**context,
                    cotangent,
                    &witnesses,
                    cotangent_type,
                    relation,
                )?)
            }
        };
        accumulators[0].accumulate(context, contribution)
    }
}

/// Represents the ability to return a value at a different but related type representation, through a [`CastOperation`]
/// (e.g., an `f32[n]` value whose runtime extent is 2 as `f32[2]`, or an `f32[2]` value as `f32[n]` next to a witness
/// of type `f32[n]`). The value at the output type and the witnesses at their own types must satisfy every relationship
/// that their shared identities express (refer to [`CastOperation`]). Concrete values have exact types, so a cast
/// checks immediately that the value's type refines the output type (their witnesses carry no identities).
/// Context-carrying values apply the operation through their context, where symbolic witness types relate the output's
/// identities to other values. An [`ArrayIrValue`] delegates array casts to its array member, including when that
/// member is a tracer; its concrete dimension witnesses have exact types and introduce no shared identities.
/// [`cast`](Self::cast) checks the relationship in staged and compiled programs too,
/// whereas [`cast_unchecked`](Self::cast_unchecked) assumes it there.
///
/// Casting is not elementwise: each value has its own output type and witnesses, so this capability applies to
/// individual values rather than to [`Parameterized`](crate::parameters::Parameterized) structures.
///
/// The universe parameter `T` defaults to the [`Capability`] universe of the implementor, so that homogeneous array
/// values implement this capability for [`ArrayType`] and composite array IR values implement it for [`ArrayIrType`].
///
/// # Example
///
/// ```rust
/// # use ryft_core::{Array, ArrayType, Cast, DataType, DimensionBounds, DimensionType, ProgramError, Shape};
/// # fn main() -> Result<(), ProgramError> {
/// let n = DimensionType::new("n", DimensionBounds::positive(Some(8))?);
/// let symbolic = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()]));
/// let value = Array::vector(vec![1f32, 2f32])?;
///
/// // A concrete value of extent 2 can be represented as `f32[n]` or `f32[2]`, but not as `f32[3]`.
/// assert_eq!(value.cast(symbolic, &[])?, value);
/// assert_eq!(value.cast(ArrayType::new_static(DataType::F32, [2]), &[])?, value);
/// assert!(value.cast(ArrayType::new_static(DataType::F32, [3]), &[]).is_err());
/// # Ok(())
/// # }
/// ```
#[capability]
pub trait Cast<T = <Self as Capability>::Universe>: Capability + Sized {
    /// Returns this value at `output_type`, requiring at runtime the relationship that a checked [`CastOperation`]
    /// states between this value at `output_type` and the `witnesses` at their own types.
    ///
    /// # Parameters
    ///
    ///   - `output_type`: Type at which to return this value. It must not define a type identity.
    ///   - `witnesses`: Values whose types bring the identities of `output_type` into scope, if any.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] if the types are unrelated, the static types or the concrete values contradict the
    /// relationship, or the cast cannot be applied in this value's context. A staged cast reports a violated
    /// relationship when the program executes.
    fn cast(&self, output_type: T, witnesses: &[Self]) -> Result<Self, ProgramError>;

    /// Returns this value at `output_type` like [`Self::cast`], but through an unchecked [`CastOperation`]: staged and
    /// compiled programs assume the relationship instead of checking it.
    ///
    /// <div class="warning">
    ///
    /// Use this only when the relationship is already proven (e.g., by an earlier check); a program that executes an
    /// unchecked cast whose relationship does not hold silently computes with the wrong extents.
    ///
    /// </div>
    ///
    /// # Parameters
    ///
    ///   - `output_type`: Type at which to return this value. It must not define a type identity.
    ///   - `witnesses`: Values whose types bring the identities of `output_type` into scope, if any.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] if the types are unrelated, the static types or the concrete values contradict the
    /// relationship, or the cast cannot be applied in this value's context.
    fn cast_unchecked(&self, output_type: T, witnesses: &[Self]) -> Result<Self, ProgramError>;
}

impl Cast for Array {
    #[inline]
    fn cast(&self, output_type: ArrayType, witnesses: &[Self]) -> Result<Self, ProgramError> {
        cast_array(CastOperation::checked(output_type, Vec::new()), self, witnesses)
    }

    #[inline]
    fn cast_unchecked(&self, output_type: ArrayType, witnesses: &[Self]) -> Result<Self, ProgramError> {
        cast_array(CastOperation::unchecked(output_type, Vec::new()), self, witnesses)
    }
}

impl<A: Value<Type = ArrayType> + Cast<ArrayType>> Cast for ArrayIrValue<A> {
    #[inline]
    fn cast(&self, output_type: ArrayIrType, witnesses: &[Self]) -> Result<Self, ProgramError> {
        cast_array_ir_value(CastOperation::checked(output_type, Vec::new()), self, witnesses)
    }

    #[inline]
    fn cast_unchecked(&self, output_type: ArrayIrType, witnesses: &[Self]) -> Result<Self, ProgramError> {
        cast_array_ir_value(CastOperation::unchecked(output_type, Vec::new()), self, witnesses)
    }
}

impl<T: Type, V: Value<Type = T, Dispatch = ValueDomainDispatch, Domain: Context<Operation: From<CastOperation<T>>>>>
    Cast<T> for V
{
    fn cast(&self, output_type: T, witnesses: &[Self]) -> Result<Self, ProgramError> {
        bind_cast(CastOperation::checked(output_type, Vec::new()), self, witnesses)
    }

    fn cast_unchecked(&self, output_type: T, witnesses: &[Self]) -> Result<Self, ProgramError> {
        bind_cast(CastOperation::unchecked(output_type, Vec::new()), self, witnesses)
    }
}

/// Applies `operation`, whose witness types are not yet set, to the concrete array `value` and `witnesses` by checking
/// its complete relationship over their exact types.
fn cast_array(operation: CastOperation<ArrayType>, value: &Array, witnesses: &[Array]) -> Result<Array, ProgramError> {
    let input_types =
        std::iter::once(value).chain(witnesses).map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
    CastOperation { witness_types: input_types[1..].to_vec(), ..operation }
        .validate_concrete_inputs(&input_types, &[])?;
    Ok(value.clone())
}

/// Applies `operation` through the array member's execution context, or validates a concrete dimension cast. Check
/// every member before omitting dimension witnesses: their exact types carry no identities, while reference witnesses
/// are invalid. Array members may themselves be tracers, so their symbolic relationships must not be checked as if
/// they were concrete values.
fn cast_array_ir_value<A: Value<Type = ArrayType> + Cast<ArrayType>>(
    operation: CastOperation<ArrayIrType>,
    value: &ArrayIrValue<A>,
    witnesses: &[ArrayIrValue<A>],
) -> Result<ArrayIrValue<A>, ProgramError> {
    let input_types =
        std::iter::once(value).chain(witnesses).map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
    let operation = CastOperation { witness_types: input_types[1..].to_vec(), ..operation };
    operation.infer_output_types(&input_types, &[])?;
    if let (ArrayIrValue::Array(value), ArrayIrType::Array(output_type)) = (value, operation.output_type()) {
        let witnesses = witnesses
            .iter()
            .filter_map(|witness| match witness {
                ArrayIrValue::Array(value) => Some(value.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();
        return Ok(ArrayIrValue::Array(if operation.is_checked() {
            value.cast(output_type.clone(), &witnesses)?
        } else {
            value.cast_unchecked(output_type.clone(), &witnesses)?
        }));
    }
    operation.validate_concrete_inputs(&input_types, &[])?;
    Ok(value.clone())
}

/// Binds `operation`, whose witness types are not yet set, to `value` and `witnesses` through the domain of `value`.
fn bind_cast<V: Value<Domain: Context<Operation: From<CastOperation<V::Type>>>>>(
    operation: CastOperation<V::Type>,
    value: &V,
    witnesses: &[V],
) -> Result<V, ProgramError> {
    let inputs = std::iter::once(value).chain(witnesses).cloned().collect::<Vec<_>>();
    let witness_types = witnesses.iter().map(|witness| witness.r#type().into_owned()).collect();
    let mut outputs = value.domain().bind(CastOperation { witness_types, ..operation }, Vec::new(), &inputs)?;
    check_count!("output", outputs, 1, ProgramError);
    Ok(outputs.remove(0))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use indoc::{formatdoc, indoc};
    use num_complex::Complex as ComplexNumber;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayElement, ArrayIrBatch, ArrayIrOperation, ArrayIrValue, ArrayOperation, ArrayReference, ArrayType,
        DataType, Dimension, DimensionBounds, DimensionError, DimensionType, DimensionValue, DimensionVariable, Layout,
        LogicalMesh, Memory, MeshAxis, MeshAxisType, RaggedAxis, Shape, Sharding, ShardingDimension, StridedLayout,
        Tile, TileDimension, TiledLayout, bf16, f8e4m3fn, f8e5m2, f8e8m0fnu, i4, u2, u4,
    };
    use crate::batching::BatchAxis;
    use crate::contexts::{EagerContext, StagingContext};
    use crate::differentiation::{
        DifferentiableOperation, DifferentiationContext, DifferentiationError, TransposableOperation,
        TranspositionContext, differentiate_at,
    };
    use crate::macros::{
        check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_transposition, check_operation_type_inference, dispatch_on_array_element_type,
    };
    use crate::operations::assertions::{AssertOperation, AssertionError, GuardOperation};
    use crate::parameters::Placeholder;
    use crate::partial::PartialValue;
    use crate::programs::{EffectClasses, Effects, EmptyRegionDriver, ProgramBuilder, ReferenceType, Typed};
    use crate::tests::ProjectedMemberType;

    use super::*;

    // An independent type family can implement the operation-specific contract without changing TypeRefinements.
    impl CastType for ProjectedMemberType<0> {
        fn validate_cast_output_type(&self, _output_type: &Self) -> Result<(), TypeError> {
            Ok(())
        }
    }

    #[test]
    fn test_convert_element_type() {
        // Operation identity, accessors, and rendering, which records the mode only for a bitcast.
        let operation = ConvertElementTypeOperation::<ArrayType>::new(DataType::F32, false);
        assert_eq!(operation.name(), CONVERT_ELEMENT_TYPE_OPERATION_NAME);
        assert_eq!(operation.data_type(), DataType::F32);
        assert!(!operation.bitcast());
        assert_eq!(operation.to_string(), "convert_element_type [data_type=f32]");
        let operation = ConvertElementTypeOperation::<ArrayType>::new(DataType::U32, true);
        assert_eq!(operation.name(), CONVERT_ELEMENT_TYPE_OPERATION_NAME);
        assert_eq!(operation.data_type(), DataType::U32);
        assert!(operation.bitcast());
        assert_eq!(operation.to_string(), "convert_element_type [data_type=u32, bitcast=true]");
    }

    #[test]
    fn test_convert_element_type_type_inference() {
        // Numerical conversion replaces the element type and validates arity and token inputs.
        let array_operation = ConvertElementTypeOperation::<ArrayType>::new(DataType::F32, false);
        check_operation_type_inference!(
            operation = array_operation,
            cases = [
                {
                    input_types = [ArrayType::scalar(DataType::F64)],
                    output_types = [ArrayType::scalar(DataType::F32)],
                },
                {
                    input_types = [],
                    error = "expected 1 input but got 0",
                },
                {
                    input_types = [ArrayType::scalar(DataType::F64), ArrayType::scalar(DataType::F64)],
                    error = "expected 1 input but got 2",
                },
                {
                    input_types = [ArrayType::scalar(DataType::Token)],
                    error = "cannot convert values to or from the `token` data type",
                },
            ],
        );

        // Shape and memory space are preserved, while byte strides are cleared when the element width changes.
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]))
            .with_layout(Layout::Strided(StridedLayout::new(vec![24, 8])))
            .with_memory(Memory::Host { pinned: true });
        check_operation_type_inference!(
            operation = array_operation,
            cases = [{
                input_types = [input_type.clone()],
                output_types = [input_type.with_data_type(DataType::F32).with_layout(None)],
            }],
        );

        // Tokens cannot be produced by conversion either.
        check_operation_type_inference!(
            operation = ConvertElementTypeOperation::<ArrayType>::new(DataType::Token, false),
            cases = [{
                input_types = [ArrayType::scalar(DataType::F64)],
                error = "cannot convert values to or from the `token` data type",
            }],
        );

        // The operation cannot own nested regions.
        assert_eq!(
            array_operation.infer_output_types(
                &[ArrayType::scalar(DataType::F64)],
                &[RegionInterface::new(vec![], vec![], EffectClasses::NONE)],
            ),
            Err(TypeError::invalid("expected 0 regions but got 1")),
        );
    }

    #[test]
    fn test_convert_element_type_type_inference_data_type() {
        // The same operation infers bare element types in the `DataType` universe.
        check_operation_type_inference!(
            operation = ConvertElementTypeOperation::<DataType>::new(DataType::F32, false),
            cases = [{
                input_types = [DataType::F64],
                output_types = [DataType::F32],
            }],
        );
    }

    #[test]
    fn test_convert_element_type_type_inference_structural_zero() {
        // Structural zeros have no materialized elements, so they only convert to themselves.
        check_operation_type_inference!(
            operation = ConvertElementTypeOperation::<ArrayType>::new(DataType::Zero, false),
            cases = [
                {
                    input_types = [ArrayType::scalar(DataType::F32)],
                    error = "cannot convert values to or from the `zero` data type",
                },
                {
                    input_types = [ArrayType::scalar(DataType::Zero)],
                    output_types = [ArrayType::scalar(DataType::Zero)],
                },
            ],
        );

        check_operation_type_inference!(
            operation = ConvertElementTypeOperation::<ArrayType>::new(DataType::F32, false),
            cases = [{
                input_types = [ArrayType::scalar(DataType::Zero)],
                error = "cannot convert values to or from the `zero` data type",
            }],
        );
    }

    #[test]
    fn test_convert_element_type_type_inference_bitcast() {
        // Equal-width reinterpretation keeps the shape, widening consumes a trailing axis of the width ratio, and only
        // element types with an unambiguous bit representation are accepted.
        check_operation_type_inference!(
            operation = ConvertElementTypeOperation::<ArrayType>::new(DataType::U32, true),
            cases = [
                {
                    input_types = [ArrayType::scalar(DataType::F32)],
                    output_types = [ArrayType::scalar(DataType::U32)],
                },
                {
                    input_types = [ArrayType::new_static(DataType::U16, [3, 2])],
                    output_types = [ArrayType::new_static(DataType::U32, [3])],
                },
                {
                    input_types = [ArrayType::new_static(DataType::U16, [3])],
                    error = "`convert_element_type` in bitcast mode requires a trailing dimension of size 2 when \
                             widening elements",
                },
                {
                    input_types = [ArrayType::scalar(DataType::U16)],
                    error = "`convert_element_type` in bitcast mode requires a trailing dimension of size 2 when \
                             widening elements",
                },
                {
                    input_types = [ArrayType::scalar(DataType::Boolean)],
                    error = "`convert_element_type` in bitcast mode requires identical source and destination types \
                             for boolean or complex elements",
                },
                {
                    input_types = [ArrayType::scalar(DataType::C64)],
                    error = "`convert_element_type` in bitcast mode requires identical source and destination types \
                             for boolean or complex elements",
                },
                {
                    input_types = [ArrayType::scalar(DataType::Token)],
                    error = "`convert_element_type` in bitcast mode requires element types with a bit representation",
                },
                {
                    input_types = [ArrayType::scalar(DataType::Zero)],
                    error = "`convert_element_type` in bitcast mode requires element types with a bit representation",
                },
                {
                    input_types = [],
                    error = "expected 1 input but got 0",
                },
            ],
        );

        // Narrowing adds a trailing axis, and widths that do not divide each other are rejected.
        check_operation_type_inference!(
            operation = ConvertElementTypeOperation::<ArrayType>::new(DataType::U16, true),
            cases = [{
                input_types = [ArrayType::new_static(DataType::F32, [3])],
                output_types = [ArrayType::new_static(DataType::U16, [3, 2])],
            }],
        );
        check_operation_type_inference!(
            operation = ConvertElementTypeOperation::<ArrayType>::new(DataType::U8, true),
            cases = [{
                input_types = [ArrayType::scalar(DataType::F6E2M3FN)],
                error = "`convert_element_type` in bitcast mode requires one element bit width to divide the other",
            }],
        );
    }

    #[test]
    fn test_convert_element_type_type_inference_bitcast_layout() {
        // Equal-width reinterpretation preserves layout and memory space, while rank-changing reinterpretation clears
        // the layout because its strides no longer describe the new shape.
        let input = ArrayType::new_static(DataType::F32, [2])
            .with_layout(Layout::Strided(StridedLayout::new(vec![8])))
            .with_memory(Memory::Host { pinned: true });
        assert_eq!(
            ConvertElementTypeOperation::<ArrayType>::new(DataType::U32, true)
                .infer_output_types(&[input.clone()], &[]),
            Ok(vec![input.clone().with_data_type(DataType::U32)]),
        );
        let output = input
            .clone()
            .with_data_type(DataType::U16)
            .with_shape(Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]))
            .with_layout(None);
        assert_eq!(
            ConvertElementTypeOperation::<ArrayType>::new(DataType::U16, true).infer_output_types(&[input], &[]),
            Ok(vec![output]),
        );
    }

    #[test]
    fn test_convert_element_type_type_inference_bitcast_sharding() {
        // The new axis is replicated, and widening removes exactly that axis without changing existing placement.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let input = ArrayType::new_static(DataType::U32, [4])
            .with_sharding(Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        let output = ArrayType::new_static(DataType::U16, [4, 2])
            .with_sharding(
                Sharding::new(mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()]).unwrap(),
            )
            .unwrap();
        assert_eq!(
            ConvertElementTypeOperation::<ArrayType>::new(DataType::U16, true)
                .infer_output_types(&[input.clone()], &[]),
            Ok(vec![output.clone()]),
        );
        assert_eq!(
            ConvertElementTypeOperation::<ArrayType>::new(DataType::U32, true).infer_output_types(&[output], &[]),
            Ok(vec![input]),
        );
    }

    #[test]
    fn test_convert_element_type_type_inference_bitcast_sharding_requires_replicated_trailing_axis() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("manual", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("explicit", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();

        // Neither manual partitioning nor a compiler-selected partition can be discarded while assembling bits.
        for dimension in [
            ShardingDimension::sharded(["manual"]),
            ShardingDimension::sharded(["explicit"]),
            ShardingDimension::unconstrained(),
        ] {
            let input = ArrayType::new_static(DataType::U16, [2])
                .with_sharding(Sharding::new(mesh.clone(), vec![dimension]).unwrap())
                .unwrap();
            assert_eq!(
                ConvertElementTypeOperation::<ArrayType>::new(DataType::U32, true).infer_output_types(&[input], &[]),
                Err(TypeError::invalid(
                    "`convert_element_type` in bitcast mode requires a replicated trailing dimension when widening \
                     elements",
                )),
            );
        }
    }

    #[test]
    fn test_convert_element_type_interpretation() {
        let output = ConvertElementTypeOperation::<ArrayType>::new(DataType::F32, false)
            .interpret(&EagerContext::<Array>::new(), &EmptyRegionDriver, &[Array::scalar(2.0f64).unwrap()])
            .unwrap();
        assert_eq!(output, vec![Array::scalar(2.0f32).unwrap()]);
    }

    #[test]
    fn test_convert_element_type_interpretation_invalid_inputs() {
        let context = EagerContext::<Array>::new();
        let operation = ConvertElementTypeOperation::<ArrayType>::new(DataType::F32, false);

        // Validate arity before accessing inputs or attempting element conversion.
        assert!(matches!(
            operation.interpret(&context, &EmptyRegionDriver, &[]),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        ));
        assert!(matches!(
            operation.interpret(
                &context,
                &EmptyRegionDriver,
                &[Array::scalar(2.0f64).unwrap(), Array::scalar(3.0f64).unwrap()]
            ),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 2 }),
        ));

        // Tokens cannot be converted into ordinary elements or produced by conversion.
        assert!(matches!(
            operation.interpret(
                &context,
                &EmptyRegionDriver,
                &[Array::new(ArrayType::scalar(DataType::Token), Vec::new()).unwrap()],
            ),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "cannot convert values to or from the `token` data type",
        ));
        assert!(matches!(
            ConvertElementTypeOperation::<ArrayType>::new(DataType::Token, false).interpret(
                &context,
                &EmptyRegionDriver,
                &[Array::scalar(2.0f64).unwrap()],
            ),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "cannot convert values to or from the `token` data type",
        ));
    }

    #[test]
    fn test_convert_element_type_interpretation_bitcast() {
        // Signed zero and NaN payloads survive without numeric conversion or canonicalization.
        let input = Array::vector(vec![0x80000000u32, 0x7fc01234]).unwrap();
        let output = ConvertElementTypeOperation::<ArrayType>::new(DataType::F32, true)
            .interpret(&EagerContext::<Array>::new(), &EmptyRegionDriver, &[input.clone()])
            .unwrap()
            .remove(0);
        assert_eq!(output.logical_bytes(), input.logical_bytes());
        assert_eq!(output.bitcast_element_type(DataType::U32), Ok(input));
    }

    #[test]
    fn test_convert_element_type_partial_evaluation() {
        let array_operation = ConvertElementTypeOperation::<ArrayType>::new(DataType::F32, false);

        // Known inputs fold to converted values; unknown inputs retain the conversion.
        check_operation_partial_evaluation!(
            operation = array_operation,
            inputs = [Array::scalar(2.0f64).unwrap()],
            expected = Array::scalar(2.0f32).unwrap(),
        );
    }

    #[test]
    fn test_convert_element_type_partial_evaluation_bitcast() {
        check_operation_partial_evaluation!(
            operation = ConvertElementTypeOperation::<ArrayType>::new(DataType::U32, true),
            inputs = [Array::scalar(1.0f32).unwrap()],
            expected = Array::scalar(0x3f800000u32).unwrap(),
        );
    }

    #[test]
    fn test_convert_element_type_batching() {
        // Numerical conversion preserves a leading mapped axis.
        let array_operation = ConvertElementTypeOperation::<ArrayType>::new(DataType::F32, false);
        check_operation_batching!(
            @exact,
            operation = array_operation,
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), Array::vector(vec![1.0, 2.0]).unwrap())],
                outputs = [(@mapped(axis = 0), Array::vector(vec![1.0f32, 2.0]).unwrap())],
            }],
        );

        // Numerical conversion preserves a non-leading mapped axis.
        check_operation_batching!(
            @exact,
            operation = array_operation,
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 1), Array::matrix(1, 2, vec![1.0f64, 2.0]).unwrap())],
                outputs = [(@mapped(axis = 1), Array::matrix(1, 2, vec![1.0f32, 2.0]).unwrap())],
            }],
        );
    }

    #[test]
    fn test_convert_element_type_batching_bitcast() {
        // Widening moves a trailing mapped axis to the front before consuming the trailing axis.
        check_operation_batching!(
            @exact,
            operation = ConvertElementTypeOperation::<ArrayType>::new(DataType::U32, true),
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 1), Array::matrix(2, 2, vec![0x5678u16, 0xcdef, 0x1234, 0x89ab]).unwrap())],
                outputs = [(@mapped(axis = 0), Array::vector(vec![0x12345678u32, 0x89abcdef]).unwrap())],
            }],
        );

        // Equal-width reinterpretation keeps the mapped position unchanged.
        check_operation_batching!(
            @exact,
            operation = ConvertElementTypeOperation::<ArrayType>::new(DataType::U32, true),
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 1), Array::matrix(1, 2, vec![1.0f32, -0.0]).unwrap())],
                outputs = [(@mapped(axis = 1), Array::matrix(1, 2, vec![0x3f800000u32, 0x80000000]).unwrap())],
            }],
        );

        // Narrowing appends the encoding-piece axis after the existing mapped axis.
        check_operation_batching!(
            @exact,
            operation = ConvertElementTypeOperation::<ArrayType>::new(DataType::U16, true),
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 1), Array::matrix(1, 2, vec![0x12345678u32, 0x89abcdef]).unwrap())],
                outputs = [(@mapped(axis = 1), Array::from_elements(
                    ArrayType::new_static(DataType::U16, [1, 2, 2]),
                    &[0x5678u16, 0x1234, 0xcdef, 0x89ab],
                ).unwrap())],
            }],
        );

        // Widening consumes the trailing axis, so a bounded ragged axis in that position cannot be batched, whereas a
        // ragged axis elsewhere carries over to the output.
        let context = BatchingContext::<_, ArrayBatchingPolicy>::new(EagerContext::<Array>::new(), 2);
        let array = Array::from_elements(ArrayType::new_static(DataType::U16, [2, 3, 2]), &[0u16; 12]).unwrap();
        let variable = DimensionVariable::new("length", DimensionBounds::new(0, Some(4)).unwrap());
        let extents = Array::from_elements(ArrayType::new_static(DataType::I32, [2]), &[1i32, 3]).unwrap();
        let input = ArrayBatch::new(array.clone(), BatchAxis::new(0))
            .unwrap()
            .with_ragged_axes(vec![RaggedAxis::new(1, extents.clone(), variable.clone(), vec![0])])
            .unwrap();
        let ragged_axes = input.ragged_axes().to_vec();
        let (outputs, _) = ConvertElementTypeOperation::<ArrayType>::new(DataType::U32, true)
            .batch(&context, &EmptyRegionDriver, &[input])
            .unwrap()
            .into_parts();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].r#type().as_ref(), &ArrayType::new_static(DataType::U32, [2, 3]));
        assert_eq!(outputs[0].ragged_axes(), ragged_axes.as_slice());
        let input = ArrayBatch::new(array, BatchAxis::new(0))
            .unwrap()
            .with_ragged_axes(vec![RaggedAxis::new(2, extents, variable, vec![0])])
            .unwrap();
        assert_eq!(
            ConvertElementTypeOperation::<ArrayType>::new(DataType::U32, true)
                .batch(&context, &EmptyRegionDriver, &[input])
                .unwrap_err(),
            BatchingError::UnsupportedOperation {
                message: format!(
                    "`{CONVERT_ELEMENT_TYPE_OPERATION_NAME}` in bitcast mode does not support consuming a bounded \
                     ragged trailing dimension",
                ),
            },
        );
    }

    #[test]
    fn test_convert_element_type_differentiation() {
        // Widening continuous values preserves their derivatives and agrees with finite differences.
        check_operation_differentiation!(
            @approx(step = 0.125, epsilon = 1e-6),
            operation = ConvertElementTypeOperation::new(DataType::F64, false),
            cases = [{
                primals = [Array::scalar(2.0f32).unwrap()],
                tangents = [Array::scalar(2.0f32).unwrap()],
                primal_outputs = [Array::scalar(2.0f64).unwrap()],
                tangent_outputs = [Array::scalar(2.0f64).unwrap()],
            }],
        );
    }

    #[test]
    fn test_convert_element_type_differentiation_low_precision() {
        // Low-precision primals use their wider differential representations in both conversion directions.
        let primal = Array::from_elements::<f8e8m0fnu>(
            ArrayType::scalar(DataType::F8E8M0FNU),
            &[2.0].map(|value| f8e8m0fnu::from_f64(value).unwrap()),
        )
        .unwrap();
        let tangent = Array::scalar(3.0f32).unwrap();
        let (output, output_tangent) =
            differentiate_at(primal).jvp(tangent, |value| value.convert_element_type(DataType::F32)).unwrap();
        assert_eq!(output.r#type().into_owned(), ArrayType::scalar(DataType::F32));
        assert_eq!(output_tangent, Array::scalar(3.0f32).unwrap());

        let primal = Array::scalar(2.0f32).unwrap();
        let tangent = Array::scalar(3.0f32).unwrap();
        let (_, output_tangent) = differentiate_at(primal)
            .jvp(tangent, |value| value.convert_element_type(DataType::F8E8M0FNU))
            .unwrap();
        assert_eq!(output_tangent, Array::scalar(3.0f32).unwrap());
    }

    #[test]
    fn test_convert_element_type_differentiation_layout() {
        // Conversions between different storage widths discard byte layouts in both primal and tangent outputs.
        let layout = Layout::Strided(StridedLayout::new(vec![1]));
        let laid_outf32 =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(1)])).with_layout(layout.clone());
        let laid_out_f8 =
            ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![Dimension::Static(1)])).with_layout(layout.clone());
        let plainf32 = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(1)]));

        let (_, tangent) = differentiate_at(Array::from_elements::<f32>(laid_outf32.clone(), &[2.0]).unwrap())
            .jvp(Array::from_elements::<f32>(laid_outf32.clone(), &[3.0]).unwrap(), |value| {
                value.convert_element_type(DataType::F8E8M0FNU)
            })
            .unwrap();
        assert_eq!(tangent, Array::from_elements::<f32>(plainf32.clone(), &[3.0]).unwrap());

        let (_, tangent) = differentiate_at(
            Array::from_elements::<f8e8m0fnu>(
                laid_out_f8.clone(),
                &[2.0].map(|value| f8e8m0fnu::from_f64(value).unwrap()),
            )
            .unwrap(),
        )
        .jvp(Array::from_elements::<f32>(plainf32.clone(), &[3.0]).unwrap(), |value| {
            value.convert_element_type(DataType::F32)
        })
        .unwrap();
        assert_eq!(tangent, Array::from_elements::<f32>(plainf32.clone(), &[3.0]).unwrap());
    }

    #[test]
    fn test_convert_element_type_differentiation_narrowing() {
        // Narrowing real and complex primals also narrows their concrete tangent values.
        let (_, tangent) = differentiate_at(Array::scalar(2.0f64).unwrap())
            .jvp(Array::scalar(3.0f64).unwrap(), |value| value.convert_element_type(DataType::F32))
            .unwrap();
        assert_eq!(tangent, Array::scalar(3.0f32).unwrap());

        let (_, tangent) = differentiate_at(Array::scalar(ComplexNumber::new(2.0f64, 0.0)).unwrap())
            .jvp(Array::scalar(ComplexNumber::new(3.0f64, 0.0)).unwrap(), |value| {
                value.convert_element_type(DataType::C64)
            })
            .unwrap();
        assert_eq!(tangent, Array::scalar(ComplexNumber::new(3.0f32, 0.0)).unwrap());
    }

    #[test]
    fn test_convert_element_type_differentiation_complex() {
        // Real-to-complex conversion inserts zero imaginary components in both primal and tangent values.
        let (primal, tangent) = differentiate_at(Array::scalar(2.0f32).unwrap())
            .jvp(Array::scalar(3.0f32).unwrap(), |value| value.convert_element_type(DataType::C64))
            .unwrap();
        assert_eq!(primal, Array::scalar(ComplexNumber::new(2.0f32, 0.0)).unwrap());
        assert_eq!(tangent, Array::scalar(ComplexNumber::new(3.0f32, 0.0)).unwrap());

        // Complex-to-real conversion discards the imaginary primal and tangent components independently.
        let (primal, tangent) = differentiate_at(Array::scalar(ComplexNumber::new(2.0f32, 5.0)).unwrap())
            .jvp(Array::scalar(ComplexNumber::new(3.0f32, -7.0)).unwrap(), |value| {
                value.convert_element_type(DataType::F32)
            })
            .unwrap();
        assert_eq!(primal, Array::scalar(2.0f32).unwrap());
        assert_eq!(tangent, Array::scalar(3.0f32).unwrap());
    }

    #[test]
    fn test_convert_element_type_differentiation_structural_zero() {
        // A structural-zero tangent stays structural and adopts the output's tangent type.
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let primal = context.input(ArrayType::scalar(DataType::C64));
        let outputs = ConvertElementTypeOperation::new(DataType::F64, false)
            .jvp(
                &DifferentiationContext::fused(context.clone()),
                &EmptyRegionDriver,
                &[DifferentiationDual::new_with_zero_tangent(primal).unwrap()],
            )
            .unwrap();
        assert!(outputs[0].tangent().is_zero());
        assert_eq!(outputs[0].tangent().r#type().as_ref(), &ArrayType::scalar(DataType::F64));

        // Only the primal conversion is staged: the zero tangent needs no materialization or conversion.
        assert_eq!(context.builder().borrow().instructions().len(), 1);
    }

    #[test]
    fn test_convert_element_type_differentiation_discrete_intermediate() {
        // Passing through an element type with a zero-dimensional tangent space erases the incoming tangent.
        let primal = Array::scalar(2.75f64).unwrap();
        let tangent = Array::scalar(3.0f64).unwrap();
        let (output, output_tangent) = differentiate_at(primal)
            .jvp(tangent, |value| value.convert_element_type(DataType::I32)?.convert_element_type(DataType::F64))
            .unwrap();
        assert_eq!(output.r#type().into_owned(), ArrayType::scalar(DataType::F64));
        assert_eq!(output_tangent, Array::scalar(0.0f64).unwrap());
    }

    #[test]
    fn test_convert_element_type_differentiation_bitcast() {
        // Even identity bit reinterpretation declares a structural zero derivative.
        let (output, tangent) = differentiate_at(Array::scalar(2.0f32).unwrap())
            .jvp(Array::scalar(3.0f32).unwrap(), |value| value.bitcast_element_type(DataType::F32))
            .unwrap();
        assert_eq!(output, Array::scalar(2.0f32).unwrap());
        assert_eq!(tangent, Array::scalar(0.0f32).unwrap());
        let gradient = differentiate_at(Array::scalar(2.0f32).unwrap())
            .gradient(|value| value.bitcast_element_type(DataType::F32))
            .unwrap();
        assert_eq!(gradient, Array::scalar(0.0f32).unwrap());
    }

    #[test]
    fn test_convert_element_type_transposition() {
        // Cotangents convert back to the input's cotangent type, and an input with no cotangent space receives a zero.
        check_operation_transposition!(
            @exact,
            operation = ConvertElementTypeOperation::new(DataType::F32, false),
            cases = [
                {
                    inputs = [(@linear(type = ArrayType::scalar(DataType::F64)))],
                    output_cotangents = [
                        Array::scalar(3.0f32).unwrap(),
                    ],
                    input_cotangents = [Array::scalar(3.0f64).unwrap()],
                },
                {
                    inputs = [(@linear(type = ArrayType::scalar(DataType::I32)))],
                    output_cotangents = [
                        Array::scalar(3.0f32).unwrap(),
                    ],
                    input_cotangents = [Array::new(ArrayType::scalar(DataType::Zero), Vec::new()).unwrap()],
                },
            ],
        );
    }

    #[test]
    fn test_convert_element_type_transposition_low_precision_layout() {
        // Cotangents recover the input differential layout in both low-precision conversion directions.
        let layout = Layout::Strided(StridedLayout::new(vec![1]));
        let laid_out_f32 =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(1)])).with_layout(layout.clone());
        let laid_out_f8 =
            ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![Dimension::Static(1)])).with_layout(layout.clone());
        let plain_f32 = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(1)]));

        check_operation_transposition!(
            @exact,
            operation = ConvertElementTypeOperation::new(DataType::F8E8M0FNU, false),
            cases = [{
                inputs = [(@linear(type = laid_out_f32.clone()))],
                output_cotangents = [Array::from_elements::<f32>(plain_f32.clone(), &[3.0]).unwrap()],
                input_cotangents = [Array::from_elements::<f32>(laid_out_f32.clone(), &[3.0]).unwrap()],
            }],
        );

        check_operation_transposition!(
            @exact,
            operation = ConvertElementTypeOperation::new(DataType::F32, false),
            cases = [{
                inputs = [(@linear(type = laid_out_f8))],
                output_cotangents = [Array::from_elements::<f32>(plain_f32.clone(), &[3.0]).unwrap()],
                input_cotangents = [Array::from_elements::<f32>(plain_f32, &[3.0]).unwrap()],
            }],
        );
    }

    #[test]
    fn test_convert_element_type_transposition_complex() {
        // Pulling a complex cotangent back through real-to-complex conversion discards its imaginary component.
        let (primal, pullback) = differentiate_at(Array::scalar(2.0f32).unwrap())
            .vjp(|value| value.convert_element_type(DataType::C64))
            .unwrap();
        assert_eq!(primal, Array::scalar(ComplexNumber::new(2.0f32, 0.0)).unwrap());
        assert_eq!(
            pullback.apply(Array::scalar(ComplexNumber::new(3.0f32, -7.0)).unwrap()).unwrap(),
            Array::scalar(3.0f32).unwrap(),
        );

        // Pulling back through complex-to-real conversion inserts a zero imaginary cotangent.
        let (primal, pullback) = differentiate_at(Array::scalar(ComplexNumber::new(2.0f32, 5.0)).unwrap())
            .vjp(|value| value.convert_element_type(DataType::F32))
            .unwrap();
        assert_eq!(primal, Array::scalar(2.0f32).unwrap());
        assert_eq!(
            pullback.apply(Array::scalar(3.0f32).unwrap()).unwrap(),
            Array::scalar(ComplexNumber::new(3.0f32, 0.0)).unwrap(),
        );
    }

    #[test]
    fn test_convert_element_type_transposition_structural_zero() {
        // A structural-zero cotangent stays structural, adopts the input's cotangent type, and stages nothing.
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let mut rule_context = TranspositionContext::new(context.clone());
        let inputs = [PartialValue::Unknown(ArrayType::scalar(DataType::C64))];
        let accumulators = rule_context.cotangent_accumulators(&inputs, &[]).unwrap();
        ConvertElementTypeOperation::new(DataType::F64, false)
            .transpose(
                &mut rule_context,
                &EmptyRegionDriver,
                &inputs,
                &[MaybeZero::Zero(ArrayType::scalar(DataType::F64))],
                &accumulators,
            )
            .unwrap();
        let cotangents = rule_context.take_cotangents(&accumulators).unwrap();
        assert!(cotangents[0].is_zero());
        assert_eq!(cotangents[0].r#type().as_ref(), &ArrayType::scalar(DataType::C64));
        assert!(context.builder().borrow().instructions().is_empty());
    }

    #[test]
    fn test_convert_element_type_transposition_bitcast() {
        // A bitcast has a zero derivative, so its transposition is rejected instead of treated as the identity.
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let mut rule_context = TranspositionContext::new(context);
        let inputs = [PartialValue::Unknown(ArrayType::scalar(DataType::F32))];
        let accumulators = rule_context.cotangent_accumulators(&inputs, &[]).unwrap();
        assert!(matches!(
            ConvertElementTypeOperation::<ArrayType>::new(DataType::F32, true).transpose(
                &mut rule_context,
                &EmptyRegionDriver,
                &inputs,
                &[MaybeZero::Zero(ArrayType::scalar(DataType::F32))],
                &accumulators,
            ),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "operation `convert_element_type` is not transposable in bitcast mode",
        ));
    }

    #[test]
    fn test_element_type_element_type() {
        assert_eq!(DataType::F64.element_type(), DataType::F64);
        assert_eq!(ArrayType::scalar(DataType::F32).element_type(), DataType::F32);
    }

    #[test]
    fn test_element_type_with_element_type() {
        assert_eq!(DataType::F64.with_element_type(DataType::F32), DataType::F32);

        // Replacing the descriptor preserves placement metadata without converting any physical values.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding = Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap();
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)]))
            .with_sharding(sharding)
            .unwrap()
            .with_layout(Layout::Strided(StridedLayout::new(vec![8])))
            .with_memory(Memory::Host { pinned: true });
        assert_eq!(
            input_type.with_element_type(DataType::F32),
            input_type.clone().with_data_type(DataType::F32).with_layout(None),
        );
        assert_eq!(input_type.with_element_type(DataType::F64), input_type);
        assert_eq!(input_type.element_type(), DataType::F64);

        // Byte strides stay valid when the element storage width is unchanged, and tiled layouts count elements.
        let strided =
            ArrayType::new_static(DataType::F32, [2]).with_layout(Layout::Strided(StridedLayout::new(vec![8])));
        assert_eq!(strided.with_element_type(DataType::I32), strided.clone().with_data_type(DataType::I32));
        let tiled = ArrayType::new_static(DataType::F32, [3])
            .with_layout(Layout::Tiled(TiledLayout::new(vec![0], vec![Tile::new(vec![TileDimension::Sized(2)])])));
        assert_eq!(tiled.with_element_type(DataType::F64), tiled.clone().with_data_type(DataType::F64));
    }

    #[test]
    fn test_element_type_with_bitcast_element_type() {
        assert_eq!(DataType::F32.with_bitcast_element_type(DataType::U32), Ok(DataType::U32));
        assert!(matches!(
            DataType::U32.with_bitcast_element_type(DataType::U16),
            Err(TypeError::Invalid { message })
                if message == "a bare element type cannot represent rank-changing bit reinterpretation",
        ));
        let input = ArrayType::new_static(DataType::U32, [3]);
        let pieces = input.with_bitcast_element_type(DataType::U16).unwrap();
        assert_eq!(pieces, ArrayType::new_static(DataType::U16, [3, 2]));
        assert_eq!(pieces.with_bitcast_element_type(DataType::U32), Ok(input));
    }

    #[test]
    fn test_convert_element_type_convert_element_type() {
        // Context-carrying values stage one numerical conversion through their own context.
        let (output_type, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |input| input.convert_element_type(DataType::F32),
            ArrayType::new_static(DataType::F64, [2]),
        )
        .unwrap();
        assert_eq!(output_type, ArrayType::new_static(DataType::F32, [2]));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[2] .
                let %1:f32[2] = convert_element_type [data_type=f32] %0
                in (%1)"},
        );
    }

    #[test]
    fn test_convert_element_type_convert_element_type_identity() {
        // A conversion to the input's own element type returns the input without staging anything.
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let input_type = ArrayType::new_static(DataType::F32, [2])
            .with_layout(Layout::Strided(StridedLayout::new(vec![4])))
            .with_memory(Memory::Host { pinned: true });
        let input = context.input(input_type);
        let output = input.convert_element_type(DataType::F32).unwrap();
        assert_eq!(output.atom_id(), input.atom_id());
        assert_eq!(output.r#type(), input.r#type());

        let zero = context.input(ArrayType::scalar(DataType::Zero));
        let output_zero = zero.convert_element_type(DataType::Zero).unwrap();
        assert_eq!(output_zero.atom_id(), zero.atom_id());

        // Validation runs before identity elimination, so an identity conversion cannot make tokens convertible.
        let token = context.input(ArrayType::scalar(DataType::Token));
        assert!(matches!(
            token.convert_element_type(DataType::Token),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "cannot convert values to or from the `token` data type",
        ));
        assert!(context.builder().borrow().instructions().is_empty());
    }

    #[test]
    fn test_convert_element_type_bitcast_element_type() {
        // Context-carrying values stage every bitcast, and rank-changing bitcasts record their trailing axis.
        let (output_type, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |input| input.bitcast_element_type(DataType::U16),
            ArrayType::new_static(DataType::U32, [3]),
        )
        .unwrap();
        assert_eq!(output_type, ArrayType::new_static(DataType::U16, [3, 2]));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:u32[3] .
                let %1:u16[3, 2] = convert_element_type [data_type=u16, bitcast=true] %0
                in (%1)"},
        );

        // Identity bitcasts remain explicit in staged code so their zero derivative cannot become an identity map.
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let input = context.input(ArrayType::scalar(DataType::F32));
        let output = input.bitcast_element_type(DataType::F32).unwrap();
        assert_ne!(output.atom_id(), input.atom_id());
        assert_eq!(output.r#type(), input.r#type());
        assert_eq!(context.builder().borrow().instructions().len(), 1);
    }

    #[test]
    fn test_convert_element_type_promote_element_type() {
        assert_eq!(
            Array::scalar(2.0f32).unwrap().promote_element_type(DataType::F64),
            Ok(Array::scalar(2.0f64).unwrap()),
        );

        // An already-promoted value retains its complete metadata and element contents.
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)]))
            .with_layout(Layout::Strided(StridedLayout::new(vec![4])))
            .with_memory(Memory::Host { pinned: true });
        let input = Array::from_elements::<f32>(input_type, &[1.0, 2.0]).unwrap();
        assert_eq!(input.promote_element_type(DataType::F32), Ok(input));
    }

    #[test]
    fn test_convert_element_type_promote_element_type_disallowed() {
        assert!(matches!(
            Array::scalar(2.0f64).unwrap().promote_element_type(DataType::F32),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "cannot promote type `f64` to type `f32`",
        ));
        assert!(matches!(
            Array::scalar(2.0f64).unwrap().promote_element_type(DataType::I32),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "cannot promote type `f64` to type `i32`",
        ));
    }

    #[test]
    fn test_array_convert_element_type() {
        // Every materialized element data type converts to every other one without falling back to a dynamic scalar
        // representation. The fixture uses each format's conversion of one, including signed one-bit narrowing.
        let data_types = [
            DataType::Boolean,
            DataType::I1,
            DataType::I2,
            DataType::I4,
            DataType::I8,
            DataType::I16,
            DataType::I32,
            DataType::I64,
            DataType::U1,
            DataType::U2,
            DataType::U4,
            DataType::U8,
            DataType::U16,
            DataType::U32,
            DataType::U64,
            DataType::F4E2M1FN,
            DataType::F6E2M3FN,
            DataType::F6E3M2FN,
            DataType::F8E3M4,
            DataType::F8E4M3,
            DataType::F8E4M3FN,
            DataType::F8E4M3FNUZ,
            DataType::F8E4M3B11FNUZ,
            DataType::F8E5M2,
            DataType::F8E5M2FNUZ,
            DataType::F8E8M0FNU,
            DataType::BF16,
            DataType::F16,
            DataType::F32,
            DataType::F64,
            DataType::C64,
            DataType::C128,
        ];
        for source_data_type in data_types {
            let source = dispatch_on_array_element_type!(source_data_type, |Element| {
                Array::from_elements(
                    ArrayType::new_static(source_data_type, [1]),
                    &[<Element as ArrayElement>::one().unwrap()],
                )
            })
            .unwrap();
            for target_data_type in data_types {
                let converted = source.convert_element_type(target_data_type).unwrap();
                assert_eq!(converted.r#type().into_owned(), ArrayType::new_static(target_data_type, [1]));
            }
        }
    }

    #[test]
    fn test_array_convert_element_type_integer_and_boolean() {
        // Representative values pin Boolean truth, integer truncation and sub-byte modular narrowing.
        let boundary = Array::vector(vec![20.0, -20.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN]).unwrap();
        assert_eq!(boundary.convert_element_type(DataType::I4).unwrap().to_f64s(), vec![7.0, -8.0, 7.0, -8.0, 0.0]);
        assert_eq!(boundary.convert_element_type(DataType::U4).unwrap().to_f64s(), vec![15.0, 0.0, 15.0, 0.0, 0.0]);
        let vector = Array::vector(vec![0.0, 1.5]).unwrap();
        assert_eq!(vector.convert_element_type(DataType::Boolean).unwrap(), Array::vector(vec![false, true]).unwrap());
        assert_eq!(vector.convert_element_type(DataType::I32).unwrap(), Array::vector(vec![0i32, 1]).unwrap());
        let signed = Array::from_elements(
            ArrayType::new_static(DataType::I4, [3]),
            &[i4::new(-8).unwrap(), i4::new(-1).unwrap(), i4::new(7).unwrap()],
        )
        .unwrap();
        assert_eq!(
            signed.convert_element_type(DataType::U2).unwrap().elements::<u2>(),
            Ok(vec![u2::new(0).unwrap(), u2::new(3).unwrap(), u2::new(3).unwrap()]),
        );
        assert_eq!(
            Array::from_elements(ArrayType::new_static(DataType::U4, [1]), &[u4::new(15).unwrap()])
                .unwrap()
                .convert_element_type(DataType::I4)
                .unwrap()
                .elements::<i4>(),
            Ok(vec![i4::new(-1).unwrap()]),
        );
    }

    #[test]
    fn test_array_convert_element_type_complex() {
        // Complex conversion preserves both components only for complex destinations and otherwise converts the real
        // component, except that Boolean conversion observes whether either component is nonzero.
        let complex = Array::vector(vec![ComplexNumber::new(0.0f32, 2.0), ComplexNumber::new(-1.5, 0.0)]).unwrap();
        assert_eq!(complex.convert_element_type(DataType::Boolean).unwrap().elements::<bool>(), Ok(vec![true, true]),);
        assert_eq!(complex.convert_element_type(DataType::I32).unwrap().elements::<i32>(), Ok(vec![0, -1]),);
        assert_eq!(
            complex.convert_element_type(DataType::C128).unwrap().elements::<ComplexNumber<f64>>(),
            Ok(vec![ComplexNumber::new(0.0, 2.0), ComplexNumber::new(-1.5, 0.0)]),
        );
    }

    #[test]
    fn test_array_convert_element_type_low_precision() {
        // Preserve low integer bits when rounding across a BF16 midpoint.
        let value = (1u64 << 60) + (1u64 << 52) + 1;
        let rounded = Array::scalar(value).unwrap().convert_element_type(DataType::BF16).unwrap();
        assert_eq!(rounded.elements::<bf16>().unwrap()[0].to_bits(), 0x5d81);

        // Conversions into low-precision floating-point element types produce exact encodings,
        // including their format-specific exceptional values.
        let vector = Array::vector(vec![0.0, 1.5]).unwrap();
        let low_precision = vector.convert_element_type(DataType::F8E5M2).unwrap();
        assert_eq!(low_precision.elements::<f8e5m2>().unwrap()[1].to_bits(), 0x3e);
        assert_eq!(
            Array::scalar(1e9f64)
                .unwrap()
                .convert_element_type(DataType::F8E4M3FN)
                .unwrap()
                .elements::<f8e4m3fn>()
                .unwrap()[0]
                .to_bits(),
            0x7f,
        );
        assert_eq!(
            Array::scalar(0.0f64)
                .unwrap()
                .convert_element_type(DataType::F8E8M0FNU)
                .unwrap()
                .elements::<f8e8m0fnu>()
                .unwrap()[0]
                .to_bits(),
            0xff,
        );
    }

    #[test]
    fn test_array_convert_element_type_encoding_fidelity() {
        // Exactly representable low-precision values survive a widening and narrowing round trip bit-for-bit.
        let array = Array::from_elements::<f8e8m0fnu>(
            ArrayType::new_static(DataType::F8E8M0FNU, [2]),
            &[2.0, 0.5].map(|value| f8e8m0fnu::from_f64(value).unwrap()),
        )
        .unwrap();
        let converted = array.convert_element_type(DataType::BF16).unwrap();
        assert_eq!(converted.to_f64s(), vec![2.0, 0.5]);
        let round_trip = converted.convert_element_type(DataType::F8E8M0FNU).unwrap();
        assert_eq!(round_trip, array);
        assert_eq!(round_trip.elements::<f8e8m0fnu>().unwrap()[0].to_bits(), 0x80);
        assert_eq!(round_trip.elements::<f8e8m0fnu>().unwrap()[1].to_bits(), 0x7e);
    }

    #[test]
    fn test_array_convert_element_type_layout() {
        // Cross-type conversion traverses the input's logical order and drops byte strides when element widths change.
        let input_type =
            ArrayType::new_static(DataType::F64, [2]).with_layout(Layout::Strided(StridedLayout::new(vec![-16])));
        let converted = Array::from_elements(input_type, &[1.9f64, -2.9])
            .unwrap()
            .convert_element_type(DataType::I32)
            .unwrap();
        assert_eq!(converted.r#type().into_owned(), ArrayType::new_static(DataType::I32, [2]),);
        assert_eq!(converted.elements::<i32>(), Ok(vec![1, -2]));
        assert_eq!(converted.storage_bytes().len(), 8);

        // A tightly packed input remains valid when widening would otherwise create overlapping byte strides.
        let input_type =
            ArrayType::new_static(DataType::F32, [2]).with_layout(Layout::Strided(StridedLayout::new(vec![4])));
        let widened = Array::from_elements(input_type, &[1.5f32, -2.5])
            .unwrap()
            .convert_element_type(DataType::F64)
            .unwrap();
        assert_eq!(widened.r#type().as_ref(), &ArrayType::new_static(DataType::F64, [2]));
        assert_eq!(widened.elements::<f64>(), Ok(vec![1.5, -2.5]));
        assert_eq!(widened.storage_bytes().len(), 16);

        // Equal-width conversion preserves byte strides, including their direction and unused storage bytes.
        let input_type =
            ArrayType::new_static(DataType::F32, [2]).with_layout(Layout::Strided(StridedLayout::new(vec![-8])));
        let converted = Array::from_elements(input_type.clone(), &[1.5f32, -2.5])
            .unwrap()
            .convert_element_type(DataType::I32)
            .unwrap();
        assert_eq!(converted.r#type().as_ref(), &input_type.with_data_type(DataType::I32));
        assert_eq!(converted.elements::<i32>(), Ok(vec![1, -2]));
        assert_eq!(converted.storage_bytes().len(), 12);

        // Tiled layouts count elements, so widening preserves their tile geometry and logical element order.
        let input_type = ArrayType::new_static(DataType::F32, [3])
            .with_layout(Layout::Tiled(TiledLayout::new(vec![0], vec![Tile::new(vec![TileDimension::Sized(2)])])));
        let widened = Array::from_elements(input_type.clone(), &[1.5f32, -2.5, 3.0])
            .unwrap()
            .convert_element_type(DataType::F64)
            .unwrap();
        assert_eq!(widened.r#type().as_ref(), &input_type.with_data_type(DataType::F64));
        assert_eq!(widened.elements::<f64>(), Ok(vec![1.5, -2.5, 3.0]));
    }

    #[test]
    fn test_array_convert_element_type_identity() {
        // Same-type conversion shares the original bytes, preserving NaN payloads and every unoccupied layout byte.
        let nan = Array::vector(vec![f32::from_bits(0x7fc0_1234)]).unwrap();
        let unchanged = nan.convert_element_type(DataType::F32).unwrap();
        assert!(Arc::ptr_eq(nan.shared_storage_bytes(), unchanged.shared_storage_bytes()));
        assert_eq!(unchanged.storage_bytes(), nan.storage_bytes());
    }

    #[test]
    fn test_array_convert_element_type_invalid_types() {
        let vector = Array::vector(vec![0.0, 1.5]).unwrap();
        // Token conversion is always rejected. Structural-zero conversion is valid only when it is a same-type no-op.
        assert!(matches!(
            vector.convert_element_type(DataType::Token),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "cannot convert values to or from the `token` data type",
        ));
        let token = Array::from_logical_bytes(ArrayType::new_static(DataType::Token, [1]), &[]).unwrap();
        assert!(matches!(
            token.convert_element_type(DataType::Token),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "cannot convert values to or from the `token` data type",
        ));
        let zero = Array::from_logical_bytes(ArrayType::new_static(DataType::Zero, [2]), &[]).unwrap();
        assert_eq!(zero.convert_element_type(DataType::Zero), Ok(zero.clone()));
        assert!(matches!(
            vector.convert_element_type(DataType::Zero),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "cannot convert values to or from the `zero` data type",
        ));
        assert!(matches!(
            zero.convert_element_type(DataType::F32),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "cannot convert values to or from the `zero` data type",
        ));
    }

    #[test]
    fn test_array_bitcast_element_type() {
        // Equal-width reinterpretation keeps the shape and every encoding bit.
        assert_eq!(
            Array::scalar(1.0f32).unwrap().bitcast_element_type(DataType::U32),
            Ok(Array::scalar(0x3f800000u32).unwrap()),
        );

        // Narrowing adds a trailing axis of pieces, ordered from least to most significant bits, and widening consumes
        // that axis.
        let pieces = Array::scalar(0x12345678u32).unwrap().bitcast_element_type(DataType::U16).unwrap();
        assert_eq!(pieces, Array::vector(vec![0x5678u16, 0x1234]).unwrap());
        assert_eq!(pieces.bitcast_element_type(DataType::U32), Ok(Array::scalar(0x12345678u32).unwrap()));

        // Identity reinterpretation returns the input unchanged, including for Boolean elements.
        let input = Array::scalar(true).unwrap();
        let output = input.bitcast_element_type(DataType::Boolean).unwrap();
        assert!(Arc::ptr_eq(input.shared_storage_bytes(), output.shared_storage_bytes()));
        assert_eq!(output, input);
    }

    #[test]
    fn test_array_bitcast_element_type_layout() {
        // Physical holes do not contribute bits when an element splits into a new logical trailing axis.
        let input_type =
            ArrayType::new_static(DataType::U32, [2]).with_layout(Layout::Strided(StridedLayout::new(vec![8])));
        let input = Array::from_elements(input_type, &[0x12345678u32, 0x89abcdef]).unwrap();
        assert_eq!(
            input.bitcast_element_type(DataType::U16),
            Ok(Array::matrix(2, 2, vec![0x5678u16, 0x1234, 0xcdef, 0x89ab]).unwrap()),
        );
        assert_eq!(
            Array::vector(Vec::<u32>::new()).unwrap().bitcast_element_type(DataType::U16),
            Ok(Array::matrix(0, 2, Vec::<u16>::new()).unwrap()),
        );
    }

    #[test]
    fn test_array_bitcast_element_type_sub_byte() {
        // FP6 occupies padded host bytes, but only its six meaningful bits participate in reinterpretation.
        let encodings = (0u8..64).collect::<Vec<_>>();
        for data_type in [DataType::F6E2M3FN, DataType::F6E3M2FN] {
            let input = Array::from_logical_bytes(ArrayType::new_static(data_type, [64]), &encodings).unwrap();
            for (pieces_type, width) in [(DataType::U1, 1), (DataType::U2, 2)] {
                let pieces = input.bitcast_element_type(pieces_type).unwrap();
                let expected = encodings
                    .iter()
                    .flat_map(|encoding| {
                        (0..6 / width).map(move |index| (encoding >> (width * index)) & ((1 << width) - 1))
                    })
                    .collect::<Vec<_>>();
                assert_eq!(pieces.r#type().as_ref(), &ArrayType::new_static(pieces_type, [64, 6 / width]));
                assert_eq!(pieces.logical_bytes(), expected);
                assert_eq!(pieces.bitcast_element_type(data_type), Ok(input.clone()));
            }
        }

        // Byte-wide elements split into sub-byte pieces from least to most significant bits.
        let pieces = Array::scalar(0xabu8).unwrap().bitcast_element_type(DataType::U4).unwrap();
        assert_eq!(pieces.elements::<u4>().unwrap(), vec![u4::new(11).unwrap(), u4::new(10).unwrap()]);
        assert_eq!(pieces.bitcast_element_type(DataType::U8), Ok(Array::scalar(0xabu8).unwrap()));
    }

    #[test]
    fn test_reduce_precision() {
        let operation = ReducePrecisionOperation::<ArrayType>::new(5, 10);
        assert_eq!(operation.name(), REDUCE_PRECISION_OPERATION_NAME);
        assert_eq!(operation.exponent_bits(), 5);
        assert_eq!(operation.mantissa_bits(), 10);
        assert_eq!(operation.to_string(), "reduce_precision [exponent_bits=5, mantissa_bits=10]");
    }

    #[test]
    fn test_reduce_precision_type_inference() {
        // Precision reduction keeps the complete input type and accepts only real floating-point element types.
        let array_operation = ReducePrecisionOperation::<ArrayType>::new(5, 10);
        let input_type = ArrayType::new_static(DataType::BF16, [2, 3]).with_memory(Memory::Host { pinned: true });
        check_operation_type_inference!(
            operation = array_operation,
            cases = [
                {
                    input_types = [input_type.clone()],
                    output_types = [input_type],
                },
                {
                    input_types = [ArrayType::scalar(DataType::F8E4M3FN)],
                    output_types = [ArrayType::scalar(DataType::F8E4M3FN)],
                },
                {
                    input_types = [],
                    error = "expected 1 input but got 0",
                },
                {
                    input_types = [ArrayType::scalar(DataType::F32), ArrayType::scalar(DataType::F32)],
                    error = "expected 1 input but got 2",
                },
                {
                    input_types = [ArrayType::scalar(DataType::I32)],
                    error = "`reduce_precision` requires real floating-point elements but got `i32`",
                },
                {
                    input_types = [ArrayType::scalar(DataType::Boolean)],
                    error = "`reduce_precision` requires real floating-point elements but got `bool`",
                },
                {
                    input_types = [ArrayType::scalar(DataType::C64)],
                    error = "`reduce_precision` requires real floating-point elements but got `c64`",
                },
                {
                    input_types = [ArrayType::scalar(DataType::Token)],
                    error = "`reduce_precision` requires real floating-point elements but got `token`",
                },
            ],
        );

        // The simulated format needs an exponent, but it may have no explicitly stored mantissa bits.
        check_operation_type_inference!(
            operation = ReducePrecisionOperation::<ArrayType>::new(0, 10),
            cases = [{
                input_types = [ArrayType::scalar(DataType::F32)],
                error = "`reduce_precision` requires at least one exponent bit",
            }],
        );

        check_operation_type_inference!(
            operation = ReducePrecisionOperation::<ArrayType>::new(5, 0),
            cases = [{
                input_types = [ArrayType::scalar(DataType::F32)],
                output_types = [ArrayType::scalar(DataType::F32)],
            }],
        );

        // The same operation infers bare element types in the `DataType` universe.
        check_operation_type_inference!(
            operation = ReducePrecisionOperation::<DataType>::new(8, 7),
            cases = [{
                input_types = [DataType::F64],
                output_types = [DataType::F64],
            }],
        );

        // The operation cannot own nested regions.
        assert_eq!(
            array_operation.infer_output_types(
                &[ArrayType::scalar(DataType::F32)],
                &[RegionInterface::new(vec![], vec![], EffectClasses::NONE)],
            ),
            Err(TypeError::invalid("expected 0 regions but got 1")),
        );
    }

    #[test]
    fn test_reduce_precision_interpretation() {
        // `1 + 2^-11` is a tie between two `f16` mantissas and rounds to the even one.
        let output = ReducePrecisionOperation::<ArrayType>::new(5, 10)
            .interpret(&EagerContext::<Array>::new(), &EmptyRegionDriver, &[Array::scalar(1.00048828125f32).unwrap()])
            .unwrap();
        assert_eq!(output, vec![Array::scalar(1.0f32).unwrap()]);
    }

    #[test]
    fn test_reduce_precision_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = ReducePrecisionOperation::<ArrayType>::new(5, 10),
            inputs = [Array::vector(vec![1.00048828125f32, 1e10]).unwrap()],
            expected = Array::vector(vec![1.0f32, f32::INFINITY]).unwrap(),
        );
    }

    #[test]
    fn test_reduce_precision_batching() {
        // Precision reduction is elementwise and so it preserves leading and non-leading mapped axes.
        let array_operation = ReducePrecisionOperation::<ArrayType>::new(5, 10);
        check_operation_batching!(
            @exact,
            operation = array_operation,
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), Array::vector(vec![1.00048828125f32, 1e10]).unwrap())],
                outputs = [(@mapped(axis = 0), Array::vector(vec![1.0f32, f32::INFINITY]).unwrap())],
            }],
        );
        check_operation_batching!(
            @exact,
            operation = array_operation,
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 1), Array::matrix(1, 2, vec![1.00048828125f32, 1e10]).unwrap())],
                outputs = [(@mapped(axis = 1), Array::matrix(1, 2, vec![1.0f32, f32::INFINITY]).unwrap())],
            }],
        );
    }

    #[test]
    fn test_reduce_precision_differentiation() {
        // The tangent is reduced to the same precision as the primal, so `1 + 2^-40` loses its low bits. The primal
        // shifts by the finite-difference step are exactly representable, so finite differences agree.
        check_operation_differentiation!(
            @approx(step = 0.125, epsilon = 0.0),
            operation = ReducePrecisionOperation::new(11, 30),
            cases = [{
                primals = [Array::scalar(2.0f64).unwrap()],
                tangents = [Array::scalar(1.0f64 + 2.0f64.powi(-40)).unwrap()],
                primal_outputs = [Array::scalar(2.0f64).unwrap()],
                tangent_outputs = [Array::scalar(1.0f64).unwrap()],
                jvp = indoc! {"
                    lambda %0:f64[], %1:f64[] .
                    let %2:f64[] = reduce_precision [exponent_bits=11, mantissa_bits=30] %0
                        %3:f64[] = reduce_precision [exponent_bits=11, mantissa_bits=30] %1
                    in (%2, %3)
                "},
            }],
        );

        // A symbolic-zero tangent stays symbolic, so only the primal reduction is staged.
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let primal = context.input(ArrayType::scalar(DataType::F32));
        let outputs = ReducePrecisionOperation::new(5, 10)
            .jvp(
                &DifferentiationContext::fused(context.clone()),
                &EmptyRegionDriver,
                &[DifferentiationDual::new_with_zero_tangent(primal).unwrap()],
            )
            .unwrap();
        assert!(outputs[0].tangent().is_zero());
        assert_eq!(outputs[0].tangent().r#type().as_ref(), &ArrayType::scalar(DataType::F32));
        assert_eq!(context.builder().borrow().instructions().len(), 1);
    }

    #[test]
    fn test_reduce_precision_transposition() {
        // The operation is its own transpose, so the cotangent is reduced to the same precision.
        check_operation_transposition!(
            @exact,
            operation = ReducePrecisionOperation::new(5, 10),
            cases = [{
                inputs = [(@linear(type = ArrayType::scalar(DataType::F32)))],
                output_cotangents = [Array::scalar(1.00048828125f32).unwrap()],
                input_cotangents = [Array::scalar(1.0f32).unwrap()],
                pullback = indoc! {"
                    lambda %0:f32[] .
                    let %1:f32[] = reduce_precision [exponent_bits=5, mantissa_bits=10] %0
                    in (%1)
                "},
            }],
        );

        // A symbolic-zero cotangent contributes nothing and stages nothing.
        let input_type = ArrayType::scalar(DataType::F32);
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let mut rule_context = TranspositionContext::new(context.clone());
        let inputs = [PartialValue::Unknown(input_type.clone())];
        let accumulators = rule_context.cotangent_accumulators(&inputs, &[]).unwrap();
        ReducePrecisionOperation::new(5, 10)
            .transpose(&mut rule_context, &EmptyRegionDriver, &inputs, &[MaybeZero::Zero(input_type)], &accumulators)
            .unwrap();
        let contributions = rule_context.take_cotangents(&accumulators).unwrap();
        assert!(contributions[0].is_zero());
        assert!(context.builder().borrow().instructions().is_empty());
    }

    #[test]
    fn test_reduce_precision_reduce_precision() {
        // Context-carrying values stage one reduction through their own context, even when it is a no-op.
        let (output_type, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |input| input.reduce_precision(8, 7)?.reduce_precision(8, 23),
            ArrayType::new_static(DataType::F32, [2]),
        )
        .unwrap();
        assert_eq!(output_type, ArrayType::new_static(DataType::F32, [2]));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2] .
                let %1:f32[2] = reduce_precision [exponent_bits=8, mantissa_bits=7] %0
                    %2:f32[2] = reduce_precision [exponent_bits=8, mantissa_bits=23] %1
                in (%2)"},
        );
    }

    #[test]
    fn test_array_reduce_precision() {
        // Reference arrays keep their type while rounding every element.
        let input = Array::from_elements(ArrayType::new_static(DataType::F32, [2]), &[1.00048828125f32, 1e10]).unwrap();
        let output = input.reduce_precision(5, 10).unwrap();
        assert_eq!(output.r#type(), input.r#type());
        assert_eq!(output, Array::vector(vec![1.0f32, f32::INFINITY]).unwrap());

        // Invalid requests fail type inference, and other floating-point element types are unsupported.
        assert_eq!(
            Array::scalar(1i32).unwrap().reduce_precision(5, 10),
            Err(TypeError::invalid("`reduce_precision` requires real floating-point elements but got `i32`").into()),
        );
        assert_eq!(
            Array::scalar(1.0f32).unwrap().reduce_precision(0, 10),
            Err(TypeError::invalid("`reduce_precision` requires at least one exponent bit").into()),
        );
        assert_eq!(
            Array::scalar(f8e4m3fn::from_f32(1.0).unwrap()).unwrap().reduce_precision(4, 2),
            Err(ProgramError::UnsupportedOperation {
                message: "`reduce_precision` is not supported for `f8e4m3fn` elements of reference arrays".to_string(),
            }),
        );
    }

    #[test]
    fn test_array_reduce_precision_f32() {
        // Each case lists an input encoding and its expected encoding after reduction to `f16` precision.
        let cases = [
            // Ties round to the even mantissa, and other values round to nearest.
            (1.00048828125f32.to_bits(), 1.0f32.to_bits()),
            (1.00146484375f32.to_bits(), 1.001953125f32.to_bits()),
            (0x3f801008, 1.0009765625f32.to_bits()),
            // Values above the largest finite `f16` overflow after rounding, keeping their sign.
            (65519.0f32.to_bits(), 65504.0f32.to_bits()),
            (65520.0f32.to_bits(), f32::INFINITY.to_bits()),
            ((-1e10f32).to_bits(), f32::NEG_INFINITY.to_bits()),
            // Values below the smallest normal `f16` flush to signed zero, unless rounding carries them up to it.
            (2.0f32.powi(-14).to_bits(), 2.0f32.powi(-14).to_bits()),
            (2.0f32.powi(-15).to_bits(), 0.0f32.to_bits()),
            ((-2.0f32.powi(-15)).to_bits(), (-0.0f32).to_bits()),
            ((2.0f32.powi(-14) * (1.0 - 2.0f32.powi(-12))).to_bits(), 2.0f32.powi(-14).to_bits()),
            // Infinities, signed zeros, and NaN payloads are preserved.
            (f32::INFINITY.to_bits(), f32::INFINITY.to_bits()),
            (f32::NEG_INFINITY.to_bits(), f32::NEG_INFINITY.to_bits()),
            ((-0.0f32).to_bits(), (-0.0f32).to_bits()),
            (0x7fc00001, 0x7fc00001),
            (0xff800001, 0xff800001),
            (0x7fffffff, 0x7fffffff),
        ];
        let input = Array::vector(cases.iter().map(|(input, _)| f32::from_bits(*input)).collect()).unwrap();
        let output = input.reduce_precision(5, 10).unwrap().elements::<f32>().unwrap();
        let expected = cases.iter().map(|(_, expected)| *expected).collect::<Vec<_>>();
        assert_eq!(output.into_iter().map(f32::to_bits).collect::<Vec<_>>(), expected);

        // Without mantissa bits, ties round toward an even exponent and NaNs are still preserved.
        let input = Array::vector(vec![1.5f32, 3.0, f32::from_bits(0xffc00000)]).unwrap();
        let output = input.reduce_precision(5, 0).unwrap().elements::<f32>().unwrap();
        assert_eq!(output.into_iter().map(f32::to_bits).collect::<Vec<_>>(), vec![0x40000000, 0x40000000, 0xffc00000]);

        // Requested widths at least as large as those of `f32` leave every encoding unchanged, including subnormals.
        let input = Array::vector(vec![f32::from_bits(1), 1.0 + f32::EPSILON, f32::from_bits(0x7fc00001)]).unwrap();
        assert_eq!(input.reduce_precision(8, 23).unwrap().logical_bytes(), input.logical_bytes());
        assert_eq!(input.reduce_precision(11, 52).unwrap().logical_bytes(), input.logical_bytes());
    }

    #[test]
    fn test_array_reduce_precision_f64() {
        // Within the normal `f32` range, reduction to `f32` precision agrees with rounding conversion to `f32`.
        let values = [1.0f64 + 2.0f64.powi(-24), 1.0 + 3.0 * 2.0f64.powi(-24), 0.1, -123.456, f32::MAX as f64];
        let input = Array::vector(values.to_vec()).unwrap();
        let expected = values.iter().map(|value| *value as f32 as f64).collect::<Vec<_>>();
        assert_eq!(input.reduce_precision(8, 23).unwrap().elements::<f64>().unwrap(), expected);
        assert_eq!(expected[..2], [1.0, 1.0 + 2.0f64.powi(-22)]);

        // Overflow, underflow, and exceptional values follow the same rules as for the other element types.
        let cases = [
            ((f32::MAX as f64 + 2.0f64.powi(103)).to_bits(), f64::INFINITY.to_bits()),
            (2.0f64.powi(128).to_bits(), f64::INFINITY.to_bits()),
            (2.0f64.powi(-126).to_bits(), 2.0f64.powi(-126).to_bits()),
            (2.0f64.powi(-127).to_bits(), 0.0f64.to_bits()),
            ((-2.0f64.powi(-127)).to_bits(), (-0.0f64).to_bits()),
            (f64::NEG_INFINITY.to_bits(), f64::NEG_INFINITY.to_bits()),
            (0x7ff0000000000001, 0x7ff0000000000001),
        ];
        let input = Array::vector(cases.iter().map(|(input, _)| f64::from_bits(*input)).collect()).unwrap();
        let output = input.reduce_precision(8, 23).unwrap().elements::<f64>().unwrap();
        let expected = cases.iter().map(|(_, expected)| *expected).collect::<Vec<_>>();
        assert_eq!(output.into_iter().map(f64::to_bits).collect::<Vec<_>>(), expected);
    }

    #[test]
    fn test_array_reduce_precision_f16() {
        // Reducing only the mantissa (to `f8e5m2` precision) rounds to nearest even, lets a mantissa carry overflow
        // into infinity, and keeps subnormals and NaN payloads.
        let cases = [
            (f16::from_f32(1.125).to_bits(), f16::from_f32(1.0).to_bits()),
            (f16::from_f32(1.375).to_bits(), f16::from_f32(1.5).to_bits()),
            (f16::MAX.to_bits(), f16::INFINITY.to_bits()),
            (0x0100, 0x0100),
            (0x7c01, 0x7c01),
        ];
        let input = Array::vector(cases.iter().map(|(input, _)| f16::from_bits(*input)).collect()).unwrap();
        let output = input.reduce_precision(5, 2).unwrap().elements::<f16>().unwrap();
        let expected = cases.iter().map(|(_, expected)| *expected).collect::<Vec<_>>();
        assert_eq!(output.into_iter().map(f16::to_bits).collect::<Vec<_>>(), expected);

        // Reducing only the exponent overflows above the largest and flushes below the smallest reduced exponent.
        let cases = [
            (f16::from_f32(255.875).to_bits(), f16::from_f32(255.875).to_bits()),
            (f16::from_f32(256.0).to_bits(), f16::INFINITY.to_bits()),
            (f16::from_f32(2.0f32.powi(-6)).to_bits(), f16::from_f32(2.0f32.powi(-6)).to_bits()),
            (f16::from_f32(2.0f32.powi(-7)).to_bits(), f16::ZERO.to_bits()),
            (f16::from_f32(-2.0f32.powi(-7)).to_bits(), f16::NEG_ZERO.to_bits()),
        ];
        let input = Array::vector(cases.iter().map(|(input, _)| f16::from_bits(*input)).collect()).unwrap();
        let output = input.reduce_precision(4, 10).unwrap().elements::<f16>().unwrap();
        let expected = cases.iter().map(|(_, expected)| *expected).collect::<Vec<_>>();
        assert_eq!(output.into_iter().map(f16::to_bits).collect::<Vec<_>>(), expected);
    }

    #[test]
    fn test_array_reduce_precision_bf16() {
        // Reducing only the mantissa rounds to nearest even and keeps subnormals and NaN payloads.
        let cases = [
            (bf16::from_f32(1.0625).to_bits(), bf16::from_f32(1.0).to_bits()),
            (bf16::from_f32(1.1875).to_bits(), bf16::from_f32(1.25).to_bits()),
            (0x0010, 0x0010),
            (0x7f81, 0x7f81),
        ];
        let input = Array::vector(cases.iter().map(|(input, _)| bf16::from_bits(*input)).collect()).unwrap();
        let output = input.reduce_precision(8, 3).unwrap().elements::<bf16>().unwrap();
        let expected = cases.iter().map(|(_, expected)| *expected).collect::<Vec<_>>();
        assert_eq!(output.into_iter().map(bf16::to_bits).collect::<Vec<_>>(), expected);

        // Reducing only the exponent (to `f16` range) overflows and flushes like the other element types.
        let cases = [
            (bf16::from_f32(65280.0).to_bits(), bf16::from_f32(65280.0).to_bits()),
            (bf16::from_f32(65536.0).to_bits(), bf16::INFINITY.to_bits()),
            (bf16::from_f32(2.0f32.powi(-14)).to_bits(), bf16::from_f32(2.0f32.powi(-14)).to_bits()),
            (bf16::from_f32(2.0f32.powi(-15)).to_bits(), bf16::ZERO.to_bits()),
            (bf16::from_f32(-2.0f32.powi(-15)).to_bits(), bf16::NEG_ZERO.to_bits()),
        ];
        let input = Array::vector(cases.iter().map(|(input, _)| bf16::from_bits(*input)).collect()).unwrap();
        let output = input.reduce_precision(5, 7).unwrap().elements::<bf16>().unwrap();
        let expected = cases.iter().map(|(_, expected)| *expected).collect::<Vec<_>>();
        assert_eq!(output.into_iter().map(bf16::to_bits).collect::<Vec<_>>(), expected);
    }

    // TODO(eaplatanios): Review from here onwards.

    #[test]
    fn test_cast_type_validate_cast_output_type() {
        check_operation_type_inference!(
            operation = CastOperation::checked(ProjectedMemberType::<0>, Vec::new())
                .with_relation(ProjectedMemberType::<0>, ProjectedMemberType::<0>),
            cases = [{ input_types = [ProjectedMemberType::<0>], output_types = [ProjectedMemberType::<0>] }],
        );
        assert_eq!(DataType::F32.validate_cast_output_type(&DataType::F32), Ok(()));
        assert_eq!(
            DataType::F32.validate_cast_output_type(&DataType::F64),
            Err(TypeError::invalid("`cast` cannot convert type `f32` to unrelated type `f64`")),
        );
        let array = ArrayType::new_static(DataType::F32, [2]);
        assert_eq!(
            array.validate_cast_output_type(&ArrayType::scalar(DataType::F32)),
            Err(TypeError::invalid("`cast` cannot convert type `f32[2]` to unrelated type `f32[]`")),
        );
        assert_eq!(
            array.validate_cast_output_type(&array.clone().with_memory(Memory::Host { pinned: true })),
            Err(TypeError::invalid("`cast` cannot convert type `f32[2]` to unrelated type `f32[2]@Host[Pinned]`")),
        );
        // Dimension extents are validated by refinement construction rather than this function.
        let dimension = DimensionType::from(Dimension::Static(2));
        assert_eq!(dimension.validate_cast_output_type(&DimensionType::from(Dimension::Static(3))), Ok(()));
        assert_eq!(
            ArrayIrType::Array(array).validate_cast_output_type(&ArrayIrType::Dimension(dimension)),
            Err(TypeError::invalid("`cast` cannot convert type `f32[2]` to unrelated type `dimension<2>`")),
        );
    }

    #[test]
    fn test_cast() {
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let m = DimensionType::new("m", DimensionBounds::positive(Some(8)).unwrap());
        let symbolic = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()]));
        let operation = CastOperation::unchecked(symbolic.clone(), vec![symbolic.clone()]);
        assert_eq!(operation.name(), CAST_OPERATION_NAME);
        assert_eq!(operation.output_type(), &symbolic);
        assert_eq!(operation.input_type(), None);
        assert_eq!(operation.witness_types(), std::slice::from_ref(&symbolic));
        assert!(!operation.is_checked());
        assert_eq!(operation.effects().classes(), Effects::empty().classes());
        assert_eq!(operation.to_string(), "cast [type=f32[n], witnesses=[f32[n]]]");
        assert_eq!(CastOperation::unchecked(symbolic.clone(), Vec::new()).to_string(), "cast [type=f32[n]]");

        // A checked cast is ordered like an assertion and renders its mode.
        let checked = CastOperation::checked(symbolic.clone(), vec![symbolic.clone()]);
        assert!(checked.is_checked());
        assert_eq!(checked.effects().classes(), EffectClasses::single(EffectClass::OrderedAssertion));
        assert_eq!(checked.to_string(), "cast [type=f32[n], witnesses=[f32[n]], checked=true]");

        // The output and witness types name identities of the surrounding program, so they follow its renamings.
        let mut renaming = TypeIdentityRenaming::new();
        renaming.insert(n.variable().unwrap().clone(), m.variable().unwrap().clone()).unwrap();
        assert_eq!(
            operation.rename_type_identities(&renaming).unwrap().to_string(),
            "cast [type=f32[m], witnesses=[f32[m]]]",
        );
        assert_eq!(
            checked.rename_type_identities(&renaming).unwrap().to_string(),
            "cast [type=f32[m], witnesses=[f32[m]], checked=true]",
        );
    }

    #[test]
    fn test_cast_type_inference() {
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let symbolic: ArrayIrType = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()])).into();
        let exact = |extent: usize| ArrayIrType::from(ArrayType::new_static(DataType::F32, [extent]));
        let static_dimension = |extent: usize| ArrayIrType::from(DimensionType::from(Dimension::Static(extent)));

        // Narrowing and widening are both admitted, for arrays and for first-class dimensions, and witnesses scope the
        // identities that a widened output type mentions.
        let narrowing = CastOperation::unchecked(exact(2), vec![symbolic.clone()]);
        assert_eq!(narrowing.infer_output_types(&[symbolic.clone(), symbolic.clone()], &[]), Ok(vec![exact(2)]));
        let widening = CastOperation::unchecked(symbolic.clone(), vec![symbolic.clone()]);
        assert_eq!(widening.infer_output_types(&[exact(2), symbolic.clone()], &[]), Ok(vec![symbolic.clone()]));
        assert_eq!(
            CastOperation::unchecked(static_dimension(2), Vec::new()).infer_output_types(&[n.clone().into()], &[]),
            Ok(vec![static_dimension(2)]),
        );

        // A relationship that the static types already contradict is rejected in both directions.
        let mismatch = Err(TypeError::custom(DimensionError::InputDimensionMismatch {
            dimension: "n".to_string(),
            expected: 2,
            actual: 5,
        }));
        assert_eq!(widening.infer_output_types(&[exact(2), exact(5)], &[]), mismatch);
        let widening_with_dimension = CastOperation::unchecked(symbolic.clone(), vec![n.clone().into()]);
        assert_eq!(widening_with_dimension.infer_output_types(&[exact(2), static_dimension(5)], &[]), mismatch);
        assert_eq!(narrowing.infer_output_types(&[symbolic.clone(), exact(5)], &[]), mismatch);

        // Unrelated types, identity-defining output types, references, and wrong arities are rejected.
        assert_eq!(
            CastOperation::unchecked(exact(2), Vec::new()).infer_output_types(&[exact(3)], &[]),
            Err(TypeError::invalid("`cast` cannot convert type `f32[3]` to unrelated type `f32[2]`")),
        );
        assert_eq!(
            CastOperation::unchecked(n.clone().into(), Vec::new()).infer_output_types(&[static_dimension(2)], &[]),
            Err(TypeError::invalid("`cast` output type `dimension<n ∈ [1, 8)>` cannot define a type identity")),
        );
        let reference: ArrayIrType = ReferenceType::new(ArrayType::new_static(DataType::F32, [2])).into();
        assert_eq!(
            CastOperation::unchecked(reference.clone(), Vec::new()).infer_output_types(&[reference], &[]),
            Err(TypeError::invalid("`cast` does not support reference type `ref<f32[2]>`")),
        );
        assert_eq!(
            widening.infer_output_types(&[exact(2)], &[]),
            Err(TypeError::invalid("expected 2 inputs but got 1"))
        );
    }

    #[test]
    fn test_cast_type_inference_relations() {
        // `f32[a]` and `f32[b, 3]`-style casts that rename an identity or narrow one component while widening another
        // relate neither type to the other, so they need a relation that both instantiate. The relation is local: it
        // renders, survives renamings of the surrounding program, and its bounds must contain both sides.
        let bounds = DimensionBounds::positive(Some(8)).unwrap();
        let [p, a, b] = ["p", "a", "b"].map(|name| DimensionType::new(name, bounds));
        let array = |dimensions: Vec<Dimension>| -> ArrayIrType {
            ArrayType::new(DataType::F32, Shape::new(dimensions)).into()
        };
        let relation = array(vec![p.simplified_dimension(), p.simplified_dimension()]);
        let input = array(vec![b.simplified_dimension(), Dimension::Static(3)]);
        let output = array(vec![Dimension::Static(3), a.simplified_dimension()]);
        let witness = array(vec![a.simplified_dimension()]);
        let plain = CastOperation::unchecked(output.clone(), vec![witness.clone()]);
        assert_eq!(
            plain.infer_output_types(&[input.clone(), witness.clone()], &[]),
            Err(TypeError::invalid("`cast` cannot convert type `f32[b, 3]` to unrelated type `f32[3, a]`")),
        );
        let related = plain.with_relation(input.clone(), relation.clone());
        assert_eq!(related.input_type(), Some(&input));
        assert_eq!(related.relation(), Some(&relation));
        assert_eq!(related.infer_output_types(&[input.clone(), witness.clone()], &[]), Ok(vec![output.clone()]));
        assert_eq!(
            related.to_string(),
            "cast [type=f32[3, a], input_type=f32[b, 3], witnesses=[f32[a]], relation=f32[p, p]]",
        );
        assert_eq!(
            related.infer_output_types(&[input.clone(), array(vec![Dimension::Static(5)])], &[]),
            Err(TypeError::custom(DimensionError::InputDimensionMismatch {
                dimension: "p".to_owned(),
                expected: 3,
                actual: 5,
            })),
        );
        let mut renaming = TypeIdentityRenaming::new();
        renaming.insert(a.to_variable(), b.to_variable()).unwrap();
        renaming.insert(b.to_variable(), a.to_variable()).unwrap();
        let renamed = related.rename_type_identities(&renaming).unwrap();
        assert_eq!(renamed.input_type(), Some(&array(vec![a.simplified_dimension(), Dimension::Static(3)])));
        assert_eq!(renamed.output_type(), &array(vec![Dimension::Static(3), b.simplified_dimension()]));
        assert_eq!(renamed.relation(), Some(&relation));

        // A relation whose bounds do not contain both sides, or that the static types contradict, is rejected.
        let narrow = DimensionType::new("narrow", DimensionBounds::positive(Some(3)).unwrap());
        let narrow_relation = array(vec![narrow.simplified_dimension(), narrow.simplified_dimension()]);
        assert!(
            CastOperation::unchecked(output.clone(), vec![witness.clone()])
                .with_relation(input.clone(), narrow_relation)
                .infer_output_types(&[input.clone(), witness.clone()], &[])
                .is_err(),
        );
        let contradicted = array(vec![Dimension::Static(4), a.simplified_dimension()]);
        assert!(
            CastOperation::unchecked(contradicted, vec![witness.clone()])
                .with_relation(input.clone(), relation)
                .infer_output_types(&[input, witness], &[])
                .is_err(),
        );
    }

    #[test]
    fn test_cast_type_inference_relation_metadata() {
        let plain = ArrayType::new_static(DataType::F32, [2]);
        let layouts = [
            None,
            Some(Layout::Strided(StridedLayout::new(vec![4]))),
            Some(Layout::Strided(StridedLayout::new(vec![8]))),
        ];
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let shardings = [
            None,
            Some(Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap()),
            Some(Sharding::new(mesh, vec![ShardingDimension::replicated()]).unwrap()),
        ];
        // Unspecified constraints admit either concrete value. Equal specified constraints agree; different ones
        // conflict. Apply this truth table independently to all 81 combinations of layout and sharding declarations.
        let compatible = [[true, true, true], [true, true, false], [true, false, true]];
        for checked in [true, false] {
            for (input_layout, layout) in layouts.iter().enumerate() {
                for (output_layout, output_layout_value) in layouts.iter().enumerate() {
                    for (input_sharding, sharding) in shardings.iter().enumerate() {
                        for (output_sharding, output_sharding_value) in shardings.iter().enumerate() {
                            let input =
                                plain.clone().with_layout(layout.clone()).with_sharding(sharding.clone()).unwrap();
                            let output = plain
                                .clone()
                                .with_layout(output_layout_value.clone())
                                .with_sharding(output_sharding_value.clone())
                                .unwrap();
                            let expected = if compatible[input_layout][output_layout]
                                && compatible[input_sharding][output_sharding]
                            {
                                Ok(vec![output.clone()])
                            } else {
                                Err(TypeError::invalid(format!(
                                    "`cast` cannot convert type `{input}` to unrelated type `{output}`",
                                )))
                            };
                            let operation = if checked {
                                CastOperation::checked(output.clone(), Vec::new())
                            } else {
                                CastOperation::unchecked(output.clone(), Vec::new())
                            }
                            .with_relation(input.clone(), plain.clone());
                            assert_eq!(operation.infer_output_types(&[input.clone()], &[]), expected);
                            let operation = if checked {
                                CastOperation::<ArrayIrType>::checked(output.into(), Vec::new())
                            } else {
                                CastOperation::<ArrayIrType>::unchecked(output.into(), Vec::new())
                            }
                            .with_relation(input.clone().into(), plain.clone().into());
                            assert_eq!(
                                operation.infer_output_types(&[input.into()], &[]),
                                expected.map(|types| types.into_iter().map(ArrayIrType::from).collect()),
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_cast_type_inference_relation_bounds() {
        let wide = DimensionBounds::positive(Some(8)).unwrap();
        let narrow = DimensionBounds::positive(Some(4)).unwrap();
        let [p, q, a, d] = ["p", "q", "a", "d"].map(|name| DimensionType::new(name, wide));
        let [b, c] = ["b", "c"].map(|name| DimensionType::new(name, narrow));
        let input = ArrayType::new(DataType::F32, Shape::new(vec![a.simplified_dimension(), c.simplified_dimension()]));
        let output =
            ArrayType::new(DataType::F32, Shape::new(vec![b.simplified_dimension(), d.simplified_dimension()]));

        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding = Sharding::new(mesh, vec![ShardingDimension::replicated(); 2]).unwrap();
        let input = input.with_layout(Layout::Strided(StridedLayout::new(vec![12, 4])));
        let output = output.with_sharding(sharding).unwrap();
        // Opposing symbolic bounds, repeated local names, and independent layout/sharding constraints remain supported.
        // Inference must preserve the requested nominal output names rather than selecting another representation.
        for relation in [
            ArrayType::new(DataType::F32, Shape::new(vec![p.simplified_dimension(), q.simplified_dimension()])),
            ArrayType::new(DataType::F32, Shape::new(vec![p.simplified_dimension(); 2])),
        ] {
            check_operation_type_inference!(
                operation = CastOperation::checked(output.clone(), vec![output.clone()])
                    .with_relation(input.clone(), relation),
                cases = [{ input_types = [input.clone(), output.clone()], output_types = [output.clone()] }],
            );
        }
    }

    #[test]
    fn test_cast_interpretation() {
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let symbolic: ArrayIrType = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()])).into();
        let vector = |extent: usize| ArrayIrValue::Array(Array::vector(vec![1f32; extent]).unwrap());
        let dimension = |extent: usize| ArrayIrValue::Dimension(DimensionValue::constant(extent).unwrap());
        let context = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();

        // Interpretation checks the complete relationship over the concrete types: widening `f32[2]` to `f32[n]` next
        // to a witness of extent 5 is rejected, and so is narrowing a value of the wrong extent.
        let widening = CastOperation::unchecked(symbolic.clone(), vec![n.clone().into()]);
        assert_eq!(widening.interpret(&context, &EmptyRegionDriver, &[vector(2), dimension(2)]), Ok(vec![vector(2)]));
        assert_eq!(
            widening.interpret(&context, &EmptyRegionDriver, &[vector(2), dimension(5)]),
            Err(ProgramError::Type(TypeError::custom(DimensionError::InputDimensionMismatch {
                dimension: "n".to_string(),
                expected: 2,
                actual: 5,
            }))),
        );
        let checked = CastOperation::checked(symbolic.clone(), vec![n.clone().into()]);
        assert_eq!(checked.interpret(&context, &EmptyRegionDriver, &[vector(2), dimension(2)]), Ok(vec![vector(2)]));
        assert!(checked.interpret(&context, &EmptyRegionDriver, &[vector(2), dimension(5)]).is_err());
        let narrowing = CastOperation::unchecked(ArrayType::new_static(DataType::F32, [2]).into(), Vec::new());
        assert_eq!(narrowing.interpret(&context, &EmptyRegionDriver, &[vector(2)]), Ok(vec![vector(2)]));
        assert_eq!(
            narrowing.interpret(&context, &EmptyRegionDriver, &[vector(3)]),
            Err(ProgramError::Type(TypeError::invalid(
                "`cast` cannot convert type `f32[3]` to unrelated type `f32[2]`",
            ))),
        );
    }

    #[test]
    fn test_cast_interpretation_symbolic_relationships() {
        let bounds = DimensionBounds::positive(Some(8)).unwrap();
        let [a, b, p] = ["a", "b", "p"].map(|name| DimensionType::new(name, bounds));
        let vector = ArrayType::new(DataType::F32, Shape::new(vec![a.simplified_dimension()]));
        let matrix =
            ArrayType::new(DataType::F32, Shape::new(vec![a.simplified_dimension(), b.simplified_dimension()]));
        let relation = ArrayType::new(DataType::F32, Shape::new(vec![p.simplified_dimension(); 2]));
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let exact = context.input(ArrayType::new_static(DataType::F32, [2]));
        let witness = context.input(vector.clone());
        let value = context.input(matrix.clone());
        let expected = Err(ProgramError::Concretization {
            message: "`cast` cannot check symbolic type relationships during interpretation; \
                          bind the operation through its context"
                .to_owned(),
        });
        for checked in [true, false] {
            let widening = if checked {
                CastOperation::checked(vector.clone(), vec![vector.clone()])
            } else {
                CastOperation::unchecked(vector.clone(), vec![vector.clone()])
            };
            assert_eq!(widening.interpret(&context, &EmptyRegionDriver, &[exact.clone(), witness.clone()]), expected,);
            let related = if checked {
                CastOperation::checked(matrix.clone(), Vec::new())
            } else {
                CastOperation::unchecked(matrix.clone(), Vec::new())
            }
            .with_relation(matrix.clone(), relation.clone());
            assert_eq!(related.interpret(&context, &EmptyRegionDriver, std::slice::from_ref(&value)), expected);
        }
        // A type-level proof is sufficient when no runtime relationship remains to be checked.
        assert_eq!(
            CastOperation::checked(matrix, Vec::new()).interpret(
                &context,
                &EmptyRegionDriver,
                std::slice::from_ref(&value),
            ),
            Ok(vec![value]),
        );
        // A requested symbolic output is not a second runtime observation. The exact input proves this relation.
        assert_eq!(
            CastOperation::checked(vector, Vec::new())
                .with_relation(
                    ArrayType::new_static(DataType::F32, [2]),
                    ArrayType::new(DataType::F32, Shape::new(vec![p.simplified_dimension()])),
                )
                .interpret(&context, &EmptyRegionDriver, std::slice::from_ref(&exact)),
            Ok(vec![exact]),
        );
        assert!(context.builder().borrow().instructions().is_empty());
    }

    #[test]
    fn test_cast_interpretation_relations() {
        let bounds = DimensionBounds::non_negative(Some(5)).unwrap();
        let [a, b, p, q] = ["a", "b", "p", "q"].map(|name| DimensionType::new(name, bounds));
        let input = ArrayType::new(DataType::F32, Shape::new(vec![a.simplified_dimension(), b.simplified_dimension()]));
        let output =
            ArrayType::new(DataType::F32, Shape::new(vec![b.simplified_dimension(), a.simplified_dimension()]));
        let relation =
            ArrayType::new(DataType::F32, Shape::new(vec![p.simplified_dimension(), q.simplified_dimension()]));
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let square = Array::matrix(2, 2, vec![1f32; 4]).unwrap();
        let rectangular = Array::matrix(2, 3, vec![1f32; 6]).unwrap();
        let empty = Array::matrix(0, 0, Vec::<f32>::new()).unwrap();
        let mismatch = Err(ProgramError::Type(TypeError::custom(DimensionError::InputDimensionMismatch {
            dimension: "p".to_owned(),
            expected: 2,
            actual: 3,
        })));

        // Concrete execution must retain a = 2, b = 3 from the input declaration. It cannot assign those same names
        // new extents from the requested output, even though the independently scoped relation has no repeated name.
        for operation in
            [CastOperation::checked(output.clone(), Vec::new()), CastOperation::unchecked(output, Vec::new())]
        {
            let operation = operation.with_relation(input.clone(), relation.clone());
            assert_eq!(
                operation.interpret(&context, &EmptyRegionDriver, std::slice::from_ref(&square)),
                Ok(vec![square.clone()])
            );
            assert_eq!(operation.interpret(&context, &EmptyRegionDriver, std::slice::from_ref(&rectangular)), mismatch);
            assert_eq!(
                operation.interpret(&context, &EmptyRegionDriver, std::slice::from_ref(&empty)),
                Ok(vec![empty.clone()])
            );
        }

        // Reusing ambient identity objects in the local relation does not link their two scopes. Keeping the input's
        // representation is valid for rectangular geometry, even when the local declaration uses the opposite names.
        let local = ArrayType::new(DataType::F32, Shape::new(vec![b.simplified_dimension(), a.simplified_dimension()]));
        let operation = CastOperation::checked(input.clone(), Vec::new()).with_relation(input, local);
        assert_eq!(operation.interpret(&context, &EmptyRegionDriver, &[rectangular.clone()]), Ok(vec![rectangular]));
    }

    #[test]
    fn test_cast_interpretation_relation_metadata() {
        let plain = ArrayType::new_static(DataType::F32, [2]);
        let input = plain.clone().with_layout(Layout::Strided(StridedLayout::new(vec![4])));
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding = Sharding::new(mesh, vec![ShardingDimension::replicated()]).unwrap();
        let output = plain.clone().with_sharding(sharding.clone()).unwrap();
        let actual_type = input.clone().with_sharding(sharding).unwrap();
        let actual = Array::from_elements(actual_type, &[1f32, 2f32]).unwrap();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let value = builder.add_input(input.clone());
        let mut outputs = Vec::new();
        for operation in
            [CastOperation::checked(output.clone(), Vec::new()), CastOperation::unchecked(output, Vec::new())]
        {
            outputs.push(
                builder
                    .add_instruction(
                        operation.with_relation(input.clone(), plain.clone()),
                        Vec::new(),
                        vec![value],
                        None,
                    )
                    .unwrap()[0],
            );
        }
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder], vec![Placeholder; 2]).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2][layout=strided{4}] .
                let %1:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}] = cast [
                    type=f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}],
                    input_type=f32[2][layout=strided{4}],
                    relation=f32[2],
                    checked=true,
                ] %0
                    %2:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}] = cast [
                        type=f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}],
                        input_type=f32[2][layout=strided{4}],
                        relation=f32[2],
                    ] %0
                in (%1, %2)"},
        );
        assert_eq!(program.interpret(vec![actual.clone()]), Ok(vec![actual.clone(); 2]));
        let context = EagerContext::new();
        let known = [PartialValue::Known(actual.clone())];
        let folded = program.partially_evaluate(&known).unwrap();
        assert_eq!(folded.program().to_string(), "lambda  .\nin ()");
        assert_eq!(folded.interpret(&context, &[]), Ok(vec![actual.clone(); 2]));
        let residual = program.partially_evaluate(&[PartialValue::Unknown(input)]).unwrap();
        assert_eq!(residual.program().to_string(), program.to_string());
        assert_eq!(residual.interpret(&context, &[actual.clone()]), Ok(vec![actual.clone(); 2]));
        let deferred = program.entry_region_ref().partially_evaluate_in_context(&context, &known, false).unwrap();
        assert_eq!(
            deferred.program().to_string(),
            indoc! {"
                lambda %0:f32[2][layout=strided{4}][sharding={mesh<['x'=1:explicit]>, [{}]}] .
                let %1:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}] = cast [
                    type=f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}],
                    input_type=f32[2][layout=strided{4}],
                    relation=f32[2],
                    checked=true,
                ] %0
                in (%1)"},
        );
        assert_eq!(deferred.interpret(&context, &[]), Ok(vec![actual; 2]));
    }

    #[test]
    fn test_cast_interpretation_relation_ordering() {
        let bounds = DimensionBounds::positive(Some(5)).unwrap();
        let [a, b, p, q] = ["a", "b", "p", "q"].map(|name| DimensionType::new(name, bounds));
        let input = ArrayType::new(DataType::F32, Shape::new(vec![a.simplified_dimension(), b.simplified_dimension()]));
        let output =
            ArrayType::new(DataType::F32, Shape::new(vec![b.simplified_dimension(), a.simplified_dimension()]));
        let relation =
            ArrayType::new(DataType::F32, Shape::new(vec![p.simplified_dimension(), q.simplified_dimension()]));
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let value = builder.add_input(input.clone());
        builder
            .add_instruction(
                CastOperation::checked(output, Vec::new()).with_relation(input, relation),
                Vec::new(),
                vec![value],
                None,
            )
            .unwrap();
        let condition = builder.add_constant(Array::scalar(false).unwrap());
        builder
            .add_instruction(AssertOperation::<ArrayType>::new("later assertion"), Vec::new(), vec![condition], None)
            .unwrap();
        let program = builder.build::<Vec<Array>, Vec<Array>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {r#"
                lambda %0:f32[a, b] .
                let %1:f32[b, a] = cast [type=f32[b, a], input_type=f32[a, b], relation=f32[p, q], checked=true] %0
                    %2:bool[] = const false
                    () = assert [message="later assertion", labels=[]] %2
                in ()"#},
        );

        // A discarded result must still check immediately, before the following ordered assertion. The valid square
        // control reaches that assertion, proving that the failing rectangular case stops at the cast itself.
        assert_eq!(
            program.interpret(vec![Array::matrix(2, 3, vec![1f32; 6]).unwrap()]),
            Err(ProgramError::Type(TypeError::custom(DimensionError::InputDimensionMismatch {
                dimension: "p".to_owned(),
                expected: 2,
                actual: 3,
            }))),
        );
        let error = program.interpret(vec![Array::matrix(2, 2, vec![1f32; 4]).unwrap()]).unwrap_err();
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::Failed { message: "later assertion".to_owned(), observations: Vec::new() }),
        );
        let error = program
            .partially_evaluate(&[PartialValue::Known(Array::matrix(2, 3, vec![1f32; 6]).unwrap())])
            .unwrap_err();
        assert_eq!(
            error,
            ProgramError::Type(TypeError::custom(DimensionError::InputDimensionMismatch {
                dimension: "p".to_owned(),
                expected: 2,
                actual: 3,
            })),
        );
    }

    #[test]
    fn test_cast_partial_evaluation() {
        // A checked cast of a known value checks its relationship while partial evaluation folds it, and a checked cast
        // of an unknown value stays checked in the residual program.
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let symbolic = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()]));
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |inputs: Vec<Tracer<_>>| Ok(vec![inputs[0].cast(ArrayType::new_static(DataType::F32, [2]), &[])?]),
            vec![symbolic.clone()],
        )
        .unwrap();
        let vector = |extent: usize| Array::vector(vec![1f32; extent]).unwrap();
        let folded = program.partially_evaluate(&[PartialValue::Known(vector(2))]).unwrap();
        assert_eq!(folded.program().to_string(), "lambda  .\nin ()");
        assert!(program.partially_evaluate(&[PartialValue::Known(vector(3))]).is_err());
        let residual = program.partially_evaluate(&[PartialValue::Unknown(symbolic)]).unwrap();
        assert_eq!(
            residual.program().to_string(),
            indoc! {"
                lambda %0:f32[n] .
                let %1:f32[2] = cast [type=f32[2], checked=true] %0
                in (%1)"},
        );
    }

    #[test]
    fn test_cast_partial_evaluation_relations() {
        let bounds = DimensionBounds::positive(Some(5)).unwrap();
        let [a, b, p, q] = ["a", "b", "p", "q"].map(|name| DimensionType::new(name, bounds));
        let input = ArrayType::new(DataType::F32, Shape::new(vec![a.simplified_dimension(), b.simplified_dimension()]));
        let output =
            ArrayType::new(DataType::F32, Shape::new(vec![b.simplified_dimension(), a.simplified_dimension()]));
        let relation =
            ArrayType::new(DataType::F32, Shape::new(vec![p.simplified_dimension(), q.simplified_dimension()]));
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let value = builder.add_input(input.clone());
        let result = builder
            .add_instruction(
                CastOperation::checked(output, Vec::new()).with_relation(input.clone(), relation),
                Vec::new(),
                vec![value],
                None,
            )
            .unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![result], vec![Placeholder], vec![Placeholder]).unwrap();
        let square = Array::matrix(2, 2, vec![1f32; 4]).unwrap();
        let rectangular = Array::matrix(2, 3, vec![1f32; 6]).unwrap();
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();

        // All-known folding checks before removing the instruction, while unknown operands keep the complete source
        // declaration in the residual. A folded valid output remains available through the evaluation's output map.
        let error = program.partially_evaluate(&[PartialValue::Known(rectangular.clone())]).unwrap_err();
        let expected = ProgramError::Type(TypeError::custom(DimensionError::InputDimensionMismatch {
            dimension: "p".to_owned(),
            expected: 2,
            actual: 3,
        }));
        assert_eq!(error, expected);
        let folded = program.partially_evaluate(&[PartialValue::Known(square.clone())]).unwrap();
        assert_eq!(folded.program().to_string(), "lambda  .\nin ()");
        assert_eq!(folded.interpret(&context, &[]), Ok(vec![square.clone()]));
        let residual = program.partially_evaluate(&[PartialValue::Unknown(input.clone())]).unwrap();
        assert_eq!(residual.program().to_string(), program.to_string());
        assert_eq!(residual.interpret(&context, &[square.clone()]), Ok(vec![square.clone()]));
        assert_eq!(residual.interpret(&context, &[rectangular.clone()]), Err(expected.clone()));

        // Disabling effect folding retains valid all-known checks. Concrete operands supply exact extents rather
        // than nominal names, while the payload retains the original declaration needed to validate the relationship.
        let deferred = program
            .entry_region_ref()
            .partially_evaluate_in_context(&context, &[PartialValue::Known(square.clone())], false)
            .unwrap();
        assert_eq!(
            deferred.program().to_string(),
            indoc! {"
                lambda %0:f32[2, 2] .
                let %1:f32[2, 2] = cast [type=f32[b, a], input_type=f32[a, b], relation=f32[p, q], checked=true] %0
                in (%1)"},
        );
        assert_eq!(deferred.interpret(&context, &[]), Ok(vec![square]));
        let error = program
            .entry_region_ref()
            .partially_evaluate_in_context(&context, &[PartialValue::Known(rectangular)], false)
            .unwrap_err();
        assert_eq!(error, expected);

        // Known operands in an outer trace are still symbolic. Folding into that trace must preserve the checked
        // primitive and its original ambient declaration rather than treating conditional facts as runtime checks.
        let (_, staged) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |values: Vec<Tracer<_>>| {
                let context = values[0].domain();
                let evaluation =
                    program.partially_evaluate_in_context(&context, &[PartialValue::Known(values[0].clone())])?;
                evaluation.interpret(&context, &[])
            },
            vec![input],
        )
        .unwrap();
        assert_eq!(staged.to_string(), program.to_string());
    }

    #[test]
    fn test_cast_partial_evaluation_widening() {
        let n = DimensionType::new("n", DimensionBounds::positive(Some(5)).unwrap());
        let symbolic = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()]));
        let exact = ArrayType::new_static(DataType::F32, [2]);
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |(value, witness)| value.cast(symbolic.clone(), &[witness]),
            (exact, symbolic.clone()),
        )
        .unwrap();
        let value = Array::vector(vec![1f32, 2f32]).unwrap();
        let context = EagerContext::new();
        let deferred = program
            .entry_region_ref()
            .partially_evaluate_in_context(
                &context,
                &[PartialValue::Known(value.clone()), PartialValue::Known(value.clone())],
                false,
            )
            .unwrap();
        assert_eq!(
            deferred.program().to_string(),
            indoc! {"
                lambda %0:f32[2], %1:f32[2] .
                let %2:f32[2] = cast [type=f32[n], witnesses=[f32[n]], checked=true] %0 %1
                in (%2)"},
        );
        assert_eq!(deferred.interpret(&context, &[]), Ok(vec![value]));
    }

    #[test]
    fn test_cast_batching() {
        // A mapped value is retyped with the batch axis inserted into the output type, and keeps that axis, while a
        // replicated witness keeps its declared type. Batching preserves the mode of the cast.
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let batch = DimensionType::new("batch", DimensionBounds::new(0, Some(5)).unwrap());
        let symbolic = ArrayIrType::from(ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()])));
        let exact: ArrayIrType = ArrayType::new_static(DataType::F32, [2]).into();
        for (operation, mode) in [
            (CastOperation::unchecked(exact.clone(), vec![symbolic.clone()]), ""),
            (CastOperation::checked(exact.clone(), vec![symbolic.clone()]), ", checked=true"),
        ] {
            let (_, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
                |(extent, mapped, witness)| {
                    let batching = BatchingContext::new(extent.domain(), extent);
                    let outputs = operation.batch(
                        &batching,
                        &EmptyRegionDriver,
                        &[ArrayIrBatch::new(mapped, BatchAxis::new(0))?, ArrayIrBatch::replicated(witness)],
                    )?;
                    let outputs = outputs.into_parts().0;
                    assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
                    Ok(outputs[0].value().clone())
                },
                (
                    ArrayIrType::Dimension(batch.clone()),
                    ArrayIrType::Array(ArrayType::new(
                        DataType::F32,
                        Shape::new(vec![batch.simplified_dimension(), n.simplified_dimension()]),
                    )),
                    symbolic.clone(),
                ),
            )
            .unwrap();
            assert_eq!(
                program.to_string(),
                formatdoc! {"
                    lambda %0:dimension<batch ∈ [0, 5)>, %1:f32[batch, n], %2:f32[n] .
                    let %3:f32[batch, 2] = cast [type=f32[batch, 2], witnesses=[f32[n]]{mode}] %1 %2
                    in (%3)"},
            );
        }
    }

    #[test]
    fn test_cast_batching_local_relation() {
        // The relation's repeated `n` means equal item dimensions, independently of an ambient batch named `n`.
        // The output and witness keep their ambient names, including a retained symbolic output dimension.
        let bounds = DimensionBounds::non_negative(Some(8)).unwrap();
        let [n, a, b] = ["n", "a", "b"].map(|name| DimensionType::new(name, bounds));
        let witness = ArrayIrType::from(ArrayType::new(DataType::F32, Shape::new(vec![a.simplified_dimension()])));
        let output = ArrayIrType::from(ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Static(3), a.simplified_dimension()]),
        ));
        let relation = ArrayIrType::from(ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension(); 2])));
        let (_, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |(extent, mapped, witness_value)| {
                let batching = BatchingContext::new(extent.domain(), extent);
                let outputs = CastOperation::unchecked(output.clone(), vec![witness.clone()])
                    .with_relation(
                        ArrayType::new(DataType::F32, Shape::new(vec![b.simplified_dimension(), Dimension::Static(3)]))
                            .into(),
                        relation.clone(),
                    )
                    .batch(
                        &batching,
                        &EmptyRegionDriver,
                        &[ArrayIrBatch::new(mapped, BatchAxis::new(0))?, ArrayIrBatch::replicated(witness_value)],
                    )?
                    .into_parts()
                    .0;
                Ok(outputs[0].value().clone())
            },
            (
                ArrayIrType::Dimension(n.clone()),
                ArrayIrType::Array(ArrayType::new(
                    DataType::F32,
                    Shape::new(vec![n.simplified_dimension(), b.simplified_dimension(), Dimension::Static(3)]),
                )),
                witness.clone(),
            ),
        )
        .unwrap();
        assert_eq!(
            program.output_types(),
            vec![ArrayIrType::Array(ArrayType::new(
                DataType::F32,
                Shape::new(vec![n.simplified_dimension(), Dimension::Static(3), a.simplified_dimension()]),
            ))],
        );
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:dimension<n ∈ [0, 8)>, %1:f32[n, b, 3], %2:f32[a] .
                let %3:f32[n, 3, a] = cast [
                    type=f32[n, 3, a],
                    input_type=f32[n, b, 3],
                    witnesses=[f32[a]],
                    relation=f32[n, n, n],
                ] %1 %2
                in (%3)"},
        );
        let ArrayIrOperation::Cast(cast) = program.instructions()[0].operation() else {
            panic!("expected a cast");
        };
        assert_eq!(
            cast.input_type(),
            Some(&ArrayIrType::Array(ArrayType::new(
                DataType::F32,
                Shape::new(vec![n.simplified_dimension(), b.simplified_dimension(), Dimension::Static(3)]),
            ))),
        );
        for extent in [0, 2] {
            let dimension = ArrayIrValue::Dimension(DimensionValue::constant(extent).unwrap());
            let mapped = ArrayIrValue::Array(
                Array::from_elements::<f32>(
                    ArrayType::new_static(DataType::F32, [extent, 3, 3]),
                    &vec![1.0; extent * 9],
                )
                .unwrap(),
            );
            let witness = |extent| ArrayIrValue::Array(Array::vector(vec![1f32; extent]).unwrap());
            assert_eq!(program.interpret((dimension.clone(), mapped.clone(), witness(3))), Ok(mapped.clone()),);
            assert!(program.interpret((dimension, mapped, witness(4))).is_err());
        }
    }

    #[test]
    fn test_cast_batching_manual_sharding() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let packed = ArrayType::new_static(DataType::F32, [2, 3])
            .with_sharding(
                Sharding::new(mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()]).unwrap(),
            )
            .unwrap();
        let varying = packed
            .clone()
            .with_sharding(packed.sharding().unwrap().clone().with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        for (packed, variation) in [(packed, ""), (varying, ", varying_manual={'x'}")] {
            let per_item = packed.unbatched(BatchAxis::new(0)).unwrap();
            let (_, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
                |(extent, mapped, witness)| {
                    let batching = BatchingContext::new(extent.domain(), extent)
                        .with_axis_sharding(ShardingDimension::sharded(["x"]));
                    let outputs = CastOperation::unchecked(per_item.clone().into(), vec![per_item.clone().into()])
                        .with_relation(per_item.clone().into(), per_item.clone().into())
                        .batch(
                            &batching,
                            &EmptyRegionDriver,
                            &[
                                ArrayIrBatch::new(mapped, BatchAxis::new(0))?,
                                ArrayIrBatch::new(witness, BatchAxis::new(0))?,
                            ],
                        )?
                        .into_parts()
                        .0;
                    Ok(outputs[0].value().clone())
                },
                (
                    ArrayIrType::Dimension(DimensionType::from(Dimension::Static(2))),
                    ArrayIrType::Array(packed.clone()),
                    ArrayIrType::Array(packed.clone()),
                ),
            )
            .unwrap();
            let expected_type = format!("f32[2, 3][sharding={{mesh<['x'=2:manual]>, [{{'x'}}, {{}}]{variation}}}]");
            assert_eq!(
                program.to_string(),
                formatdoc! {"
                    lambda %0:dimension<2>, %1:{expected_type}, %2:{expected_type} .
                    let %3:{expected_type} = cast [
                        type={expected_type},
                        input_type={expected_type},
                        witnesses=[{expected_type}],
                        relation={expected_type},
                    ] %1 %2
                    in (%3)"},
            );
        }
    }

    #[test]
    fn test_cast_differentiation() {
        // The primal cast keeps its mode, while the tangent is converted by an unchecked cast that the primal input
        // and, for a checked cast, the primal output witness, so that the tangent conversion depends on the check. The
        // pullback converts the cotangent back with an unchecked cast.
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let symbolic = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()]));
        let exact = ArrayType::new_static(DataType::F32, [2]);
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |inputs: Vec<Tracer<_>>| Ok(vec![inputs[0].cast(exact.clone(), &[])?]),
            vec![symbolic.clone()],
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[n] .
                let %1:f32[2] = cast [type=f32[2], checked=true] %0
                in (%1)"},
        );
        assert_eq!(
            program.jvp().unwrap().to_string(),
            indoc! {"
                lambda %0:f32[n], %1:f32[n] .
                let %2:f32[2] = cast [type=f32[2], checked=true] %0
                    %3:f32[2] = cast [type=f32[2], witnesses=[f32[n], f32[2]]] %1 %0 %2
                in (%2, %3)"},
        );
        let pullback = program.linearize().unwrap().tangent().transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f32[2], %1:f32[n], %2:f32[2] .
                let %3:f32[n] = cast [type=f32[n], witnesses=[f32[n], f32[2]]] %0 %1 %2
                in (%3)"},
        );
        let vector = |values: Vec<f32>| Array::vector(values).unwrap();
        let inputs = pullback.input_types().iter().map(|_| vector(vec![1.0, 2.0])).collect::<Vec<_>>();
        assert_eq!(pullback.interpret(inputs), Ok(vec![vector(vec![1.0, 2.0])]));
    }

    #[test]
    fn test_cast_differentiation_validates_witnesses() {
        let bounds = DimensionBounds::positive(Some(8)).unwrap();
        let [p, a, b] = ["p", "a", "b"].map(|name| DimensionType::new(name, bounds));
        let vector = |dimension: &DimensionType| {
            ArrayType::new(DataType::F32, Shape::new(vec![dimension.simplified_dimension()]))
        };
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let first = builder.add_input(vector(&a));
        let second = builder.add_input(vector(&b));
        let guarded = builder
            .add_instruction(
                GuardOperation::new("region", vec![vector(&p), vector(&p)]).with_retained(vec![a.to_variable()]),
                vec![],
                vec![first, second],
                None,
            )
            .unwrap()
            .to_vec();
        let sum = builder.add_instruction(AddOperation::new(), vec![], guarded, None).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![sum], vec![Placeholder; 2], vec![Placeholder])
            .unwrap()
            .jvp()
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {r#"
                lambda %0:f32[a], %1:f32[b], %2:f32[a], %3:f32[b] .
                let %4:f32[a], %5:f32[a] = guard [boundary="region", types=[f32[p], f32[p]], retained=[a]] %0 %1
                    %6:f32[a] = cast [type=f32[a], input_type=f32[b], witnesses=[f32[b], f32[a]], relation=f32[p]] %3 %1 %5
                    %7:f32[a] = add %4 %5
                    %8:f32[a] = add %2 %6
                in (%7, %8)"#},
        );
        let operation = program
            .instructions()
            .iter()
            .find_map(|instruction| match instruction.operation() {
                ArrayOperation::Cast(operation) => Some(operation),
                _ => None,
            })
            .unwrap();
        let exact = |extent| ArrayType::new_static(DataType::F32, [extent]);
        assert_eq!(
            operation.infer_output_types(&[exact(3), exact(3), exact(5)], &[]),
            Err(TypeError::custom(DimensionError::InputDimensionMismatch {
                dimension: "p".to_owned(),
                expected: 3,
                actual: 5,
            })),
        );
        assert_eq!(operation.infer_output_types(&[exact(3), exact(3), exact(3)], &[]), Ok(vec![exact(3)]));
    }

    #[test]
    fn test_cast_transposition() {
        let exact = ArrayType::new_static(DataType::F32, [2]);
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |inputs: Vec<Tracer<_>>| Ok(vec![inputs[0].cast(exact.clone(), &[])?]),
            vec![exact.clone()],
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2] .
                let %1:f32[2] = cast [type=f32[2], checked=true] %0
                in (%1)"},
        );
        let transposed = program.transpose_with_respect_to(&[0], &[]).unwrap();
        let value = Array::vector(vec![1f32, 2f32]).unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f32[2] .
                in (%0)"},
        );
        assert_eq!(transposed.interpret(vec![value.clone()]), Ok(vec![value]));

        // Structural zeros retain the original nominal cotangent type and require no materialized witness.
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let symbolic = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()]));
        let operation = CastOperation::checked(exact.clone(), Vec::new());
        let mut context = TranspositionContext::new(TracingContext::<Array, ArrayOperation<Array>>::new());
        let accumulator = context.cotangent_accumulator(symbolic.clone(), true);
        operation
            .transpose(
                &mut context,
                &EmptyRegionDriver,
                &[PartialValue::Unknown(symbolic.clone())],
                &[MaybeZero::Zero(exact)],
                std::slice::from_ref(&accumulator),
            )
            .unwrap();
        let cotangents = context.take_cotangents(&[accumulator]).unwrap();
        assert!(matches!(cotangents.as_slice(), [MaybeZero::Zero(r#type)] if *r#type == symbolic));
        assert!(context.builder().borrow().instructions().is_empty());
    }

    #[test]
    fn test_cast_transposition_input_identities() {
        // Bare narrowing has no saved value carrying n. Do not invent that identity or change the accumulator type.
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let symbolic = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()]));
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(symbolic);
        let output = builder
            .add_instruction(
                CastOperation::checked(ArrayType::new_static(DataType::F32, [2]), Vec::new()),
                Vec::new(),
                vec![input],
                None,
            )
            .unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[n] .
                let %1:f32[2] = cast [type=f32[2], checked=true] %0
                in (%1)"},
        );
        assert_eq!(
            program.transpose_with_respect_to(&[0], &[]).unwrap_err(),
            DifferentiationError::Program(ProgramError::UnsupportedOperation {
                message: "`cast` transposition needs a known witness for input identity `n`; linearize the program before transposing it".to_owned(),
            }),
        );

        // A known dimension can witness a narrowed axis without being an explicit operand of the forward cast.
        // The unchanged n is carried by the cotangent, and the unrelated known dimension must not become a witness.
        let m = DimensionType::new("m", DimensionBounds::positive(Some(8)).unwrap());
        let unrelated = DimensionType::new("unrelated", DimensionBounds::positive(Some(8)).unwrap());
        let input_type: ArrayIrType =
            ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension(), m.simplified_dimension()])).into();
        let output_type: ArrayIrType =
            ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension(), Dimension::Static(2)])).into();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        builder.add_input(m.clone().into());
        builder.add_input(unrelated.into());
        let input = builder.add_input(input_type.clone());
        let output = builder
            .add_instruction(CastOperation::checked(output_type.clone(), Vec::new()), Vec::new(), vec![input], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder; 3],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:dimension<m ∈ [1, 8)>, %1:dimension<unrelated ∈ [1, 8)>, %2:f32[n, m] .
                let %3:f32[n, 2] = cast [type=f32[n, 2], checked=true] %2
                in (%3)"},
        );
        let transposed = program.transpose_with_respect_to(&[2], &[]).unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f32[n, 2], %1:dimension<m ∈ [1, 8)>, %2:dimension<unrelated ∈ [1, 8)> .
                let %3:f32[n, m] = cast [type=f32[n, m], witnesses=[dimension<m ∈ [1, 8)>]] %0 %1
                in (%3)"},
        );
        assert_eq!(transposed.output_types(), vec![input_type]);
        let value = ArrayIrValue::Array(Array::matrix(3, 2, vec![1f32; 6]).unwrap());
        assert_eq!(
            transposed.interpret(vec![
                value.clone(),
                ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()),
                ArrayIrValue::Dimension(DimensionValue::constant(7).unwrap()),
            ]),
            Ok(vec![value]),
        );
    }

    #[test]
    fn test_cast_transposition_relations() {
        let bounds = DimensionBounds::positive(Some(5)).unwrap();
        let [a, b, p, q] = ["a", "b", "p", "q"].map(|name| DimensionType::new(name, bounds));
        let input = ArrayType::new(DataType::F32, Shape::new(vec![a.simplified_dimension(), b.simplified_dimension()]));
        let output =
            ArrayType::new(DataType::F32, Shape::new(vec![b.simplified_dimension(), a.simplified_dimension()]));
        let relation =
            ArrayType::new(DataType::F32, Shape::new(vec![p.simplified_dimension(), q.simplified_dimension()]));
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let value = builder.add_input(input.clone());
        let result = builder
            .add_instruction(
                CastOperation::checked(output, Vec::new()).with_relation(input, relation),
                Vec::new(),
                vec![value],
                None,
            )
            .unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![result], vec![Placeholder], vec![Placeholder]).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[a, b] .
                let %1:f32[b, a] = cast [type=f32[b, a], input_type=f32[a, b], relation=f32[p, q], checked=true] %0
                in (%1)"},
        );
        let transposed = program.transpose_with_respect_to(&[0], &[]).unwrap();

        // Reversing the cast swaps its declared input and requested output. The cotangent already carries both names,
        // so it needs no additional witness, and the nominal accumulator type must remain [a, b].
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f32[b, a] .
                let %1:f32[a, b] = cast [type=f32[a, b], input_type=f32[b, a], relation=f32[p, q]] %0
                in (%1)"},
        );
        let value = Array::matrix(2, 2, vec![1f32; 4]).unwrap();
        assert_eq!(transposed.interpret(vec![value.clone()]), Ok(vec![value.clone()]));
        let pullback = program.linearize().unwrap().tangent().transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f32[b, a], %1:f32[a, b], %2:f32[b, a] .
                let %3:f32[a, b] = cast [
                    type=f32[a, b],
                    input_type=f32[b, a],
                    witnesses=[f32[a, b], f32[b, a]],
                    relation=f32[p, q],
                ] %0 %1 %2
                in (%3)"},
        );
        let inputs = pullback.input_types().iter().map(|_| value.clone()).collect();
        assert_eq!(pullback.interpret(inputs), Ok(vec![value]));
    }

    #[test]
    fn test_array_cast() {
        // Concrete values have exact types, so a cast checks immediately that the value's type refines the output type,
        // in both modes. Concrete witnesses carry no identities, so they relate to nothing.
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let symbolic = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()]));
        let exact = |extent: usize| ArrayType::new_static(DataType::F32, [extent]);
        let vector = |extent: usize| Array::vector(vec![1f32; extent]).unwrap();
        assert_eq!(vector(2).cast(symbolic.clone(), &[]), Ok(vector(2)));
        assert_eq!(vector(2).cast_unchecked(symbolic.clone(), &[vector(3)]), Ok(vector(2)));
        assert_eq!(vector(2).cast(exact(2), &[]), Ok(vector(2)));
        let unrelated = Err(ProgramError::Type(TypeError::invalid(
            "`cast` cannot convert type `f32[3]` to unrelated type `f32[2]`",
        )));
        assert_eq!(vector(3).cast(exact(2), &[]), unrelated);
        assert_eq!(vector(3).cast_unchecked(exact(2), &[]), unrelated);
        assert_eq!(
            vector(2).cast(symbolic.clone(), &[vector(2), vector(2)]).map(|value| value.r#type().into_owned()),
            Ok(exact(2)),
        );
    }

    #[test]
    fn test_array_ir_value_cast() {
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let symbolic = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()]));
        let exact = |extent: usize| ArrayType::new_static(DataType::F32, [extent]);
        let vector = |extent: usize| Array::vector(vec![1f32; extent]).unwrap();

        // Composite values behave the same. A first-class dimension type defines its identity, which the output type of
        // a cast may not, so only exact dimension types are valid outputs for dimension values.
        let composite = |extent: usize| ArrayIrValue::<Array>::Array(vector(extent));
        assert_eq!(composite(2).cast(symbolic.clone().into(), &[]), Ok(composite(2)));
        assert!(composite(3).cast_unchecked(exact(2).into(), &[]).is_err());
        let reference = ArrayIrValue::Reference(ArrayReference::new(vector(2)));
        let expected = Err(ProgramError::Type(TypeError::invalid(format!(
            "`cast` does not support reference type `{}`",
            reference.r#type(),
        ))));
        assert_eq!(composite(2).cast(exact(2).into(), std::slice::from_ref(&reference)), expected);
        assert_eq!(composite(2).cast_unchecked(exact(2).into(), &[reference]), expected);
        let dimension = |extent: usize| ArrayIrValue::<Array>::Dimension(DimensionValue::constant(extent).unwrap());
        let exact_dimension = ArrayIrType::Dimension(DimensionType::from(Dimension::Static(2)));
        assert_eq!(dimension(2).cast(exact_dimension.clone(), &[]), Ok(dimension(2)));
        assert!(dimension(3).cast(exact_dimension, &[]).is_err());
        assert_eq!(
            dimension(2).cast(ArrayIrType::Dimension(n), &[]),
            Err(ProgramError::Type(TypeError::invalid(
                "`cast` output type `dimension<n ∈ [1, 8)>` cannot define a type identity",
            ))),
        );
    }

    #[test]
    fn test_array_ir_value_cast_wrapped_tracers() {
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let symbolic = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()]));
        let exact = ArrayType::new_static(DataType::F32, [2]);
        let vector = |extent: usize| Array::vector(vec![1f32; extent]).unwrap();
        for (checked, mode) in [(true, ", checked=true"), (false, "")] {
            // A dimension witness has no identity and introduces no relationship with the output. Keep the array
            // witness and let its context perform the cast, including its runtime check in checked mode.
            let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
                |(value, witness)| {
                    let value = ArrayIrValue::Array(value);
                    let witnesses =
                        [ArrayIrValue::Dimension(DimensionValue::constant(7)?), ArrayIrValue::Array(witness)];
                    let output = if checked {
                        value.cast(symbolic.clone().into(), &witnesses)?
                    } else {
                        value.cast_unchecked(symbolic.clone().into(), &witnesses)?
                    };
                    let ArrayIrValue::Array(output) = output else { panic!("expected an array") };
                    Ok(output)
                },
                (exact.clone(), symbolic.clone()),
            )
            .unwrap();
            let expected = formatdoc! {"
                lambda %0:f32[2], %1:f32[n] .
                let %2:f32[n] = cast [type=f32[n], witnesses=[f32[n]]{mode}] %0 %1
                in (%2)"};
            assert_eq!(program.to_string(), expected);
            assert_eq!(program.interpret((vector(2), vector(2))), Ok(vector(2)));
            // Interpretation validates concrete geometry in both modes; unchecked lowering alone assumes it.
            assert_eq!(
                program.interpret((vector(2), vector(3))),
                Err(ProgramError::Type(TypeError::custom(DimensionError::InputDimensionMismatch {
                    dimension: "n".to_owned(),
                    expected: 2,
                    actual: 3,
                }))),
            );
            let residual = program
                .to_flat_program()
                .partially_evaluate(&[PartialValue::Unknown(exact.clone()), PartialValue::Unknown(symbolic.clone())])
                .unwrap();
            assert_eq!(residual.program().to_string(), expected);

            // Narrowing a wrapped tracer must stage rather than attempting to inspect symbolic geometry eagerly.
            let (_, narrowed) = TracingContext::<Array, ArrayOperation<Array>>::trace(
                |value| {
                    let value = ArrayIrValue::Array(value);
                    let output = if checked {
                        value.cast(exact.clone().into(), &[])?
                    } else {
                        value.cast_unchecked(exact.clone().into(), &[])?
                    };
                    let ArrayIrValue::Array(output) = output else { panic!("expected an array") };
                    Ok(output)
                },
                symbolic.clone(),
            )
            .unwrap();
            assert_eq!(
                narrowed.to_string(),
                formatdoc! {"
                    lambda %0:f32[n] .
                    let %1:f32[2] = cast [type=f32[2]{mode}] %0
                    in (%1)"},
            );
            assert_eq!(narrowed.interpret(vector(2)), Ok(vector(2)));
            assert!(matches!(narrowed.interpret(vector(3)), Err(ProgramError::Type(_))));
        }
    }

    #[test]
    fn test_tracer_cast() {
        // Context-carrying values bind a checked or an unchecked cast through their context.
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let symbolic: ArrayIrType = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()])).into();
        let exact: ArrayIrType = ArrayType::new_static(DataType::F32, [2]).into();
        let (_, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |(value, witness)| {
                let checked = value.cast(exact.clone(), &[])?;
                let unchecked = value.cast_unchecked(exact.clone(), &[witness])?;
                Ok((checked, unchecked))
            },
            (symbolic.clone(), symbolic.clone()),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[n], %1:f32[n] .
                let %2:f32[2] = cast [type=f32[2], checked=true] %0
                    %3:f32[2] = cast [type=f32[2], witnesses=[f32[n]]] %0 %1
                in (%2, %3)"},
        );

        // A name absent from both the value and its declared witnesses is not a lost identity from concretization.
        // Do not erase it and admit a widening for which no operand supplies the requested relationship.
        let error = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |(value, _witness)| value.cast(symbolic.clone(), &[]),
            (exact, symbolic.clone()),
        )
        .unwrap_err();
        assert_eq!(
            error,
            ProgramError::Type(TypeError::invalid(
                "operation `cast` output type references identity n without consuming or defining it",
            )),
        );
    }
}
