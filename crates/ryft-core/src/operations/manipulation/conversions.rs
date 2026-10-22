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
//!     changing their type representation, including the layout and the placement over explicit mesh axes. A checked
//!     cast validates the required type relationships at runtime. An unchecked cast assumes those relationships, for
//!     example after a [`GuardOperation`](crate::GuardOperation) has checked them. Differentiation and transposition
//!     use unchecked casts for tangent and cotangent representations whose relationships the primal computation has
//!     proven. The [`CastType`] type trait infers the output type and validates the complete relationship, including
//!     dimension identities, extents, element types, and layout constraints.
//!
//! Numerical conversion is linear, so differentiation converts the primal and aligns the tangent with the output's
//! tangent type, transposition converts the cotangent back to the input's cotangent type, and conversions into or
//! out of a type without a tangent space (e.g., integers) produce structural zeros. A bitcast has a structural zero
//! derivative and cannot be transposed. Precision reduction is treated as linear and as its own transpose, so that
//! tangents and cotangents carry the same precision as the primal. Batching keeps the mapped axis in place, except that
//! a widening bitcast moves it away from the trailing position it would otherwise consume. A type-representation cast
//! is the identity: its tangent and cotangent are unchecked casts to the tangent and cotangent representations that
//! stay behind the primal check, and batching lifts its declarations by the mapped axis. Backends lower numerical
//! conversion, bitcasts, and precision reduction to their element-conversion, bit-reinterpretation, and
//! precision-reduction constructs (e.g., [`stablehlo.convert`](https://openxla.org/stablehlo/spec#convert),
//! [`stablehlo.bitcast_convert`](https://openxla.org/stablehlo/spec#bitcast_convert), and
//! [`stablehlo.reduce_precision`](https://openxla.org/stablehlo/spec#reduce_precision) in the XLA backend), and a
//! cast to the runtime checks that its actual input types leave unproven, followed by the shape, layout and placement
//! conversion.
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

use crate::arrays::types::dimensions::{DimensionComparison, DimensionRefinementFact};
use crate::arrays::{
    Array, ArrayAddressing, ArrayBatch, ArrayBatchingPolicy, ArrayExtentBatchingPolicy, ArrayIrType, ArrayIrValue,
    ArrayType, DataType, Dimension, DimensionRefinements, DimensionType, DimensionTypeRefinements, DimensionVariable,
    Layout, MeshAxisType, Shape, Sharding, ShardingDimension, bf16, f16,
};
use crate::batching::{
    BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError, BatchingPolicy,
    InterpretableBatchableOperation,
};
use crate::contexts::{Context, Domain, StagingContext};
use crate::differentiation::{
    CotangentAccumulator, DifferentiableOperation, DifferentiableType, DifferentiationContext, DifferentiationDriver,
    DifferentiationDual, DifferentiationError, DifferentiationPolicy, ElementwiseDerivativeAlignment,
    SupportsCotangentAdd, TransposableOperation, TranspositionContext, TranspositionDriver,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_differentiable_operation, impl_reference_dischargeable_operation};
use crate::operations::Capability;
use crate::operations::dimensions::dimension_size::DimensionSizeOperation;
use crate::operations::manipulation::transposition::Transpose;
use crate::operations::sharding::constrain_sharding::CONSTRAIN_SHARDING_OPERATION_NAME;
use crate::partial::{
    PartialEvaluationContext, PartialEvaluationDriver, PartialEvaluationValue, PartialValue,
    PartiallyEvaluatableOperation,
};
use crate::programs::{
    EffectClass, EffectClasses, Effects, MaybeZero, Operation, OperationFoldOutput, OperationFormatter, ProgramError,
    RegionInterface, Type, TypeError, TypeIdentityPosition, TypeIdentityRenaming, TypeRefinements, Typed, Value,
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

/// Canonical operation name for [`CastOperation`].
pub const CAST_OPERATION_NAME: &str = "cast";

/// [`Operation`] that returns its first input at a different type representation (e.g., `f32[n]` as `f32[2]`). The
/// input and output describe the same value, with equal extents on corresponding axes. Arrays require equal element
/// types, rank and memory space. Layout may change, as long as the requested layout addresses the output shape (refer
/// to [`ArrayType::validate_layout`]), and so may placement over [`MeshAxisType::Explicit`] mesh axes under the rules
/// of [`ReshardOperation`](crate::ReshardOperation): the output requests their new representation. Placement over
/// [`MeshAxisType::Auto`] and [`MeshAxisType::Manual`] axes is preserved, since the former belongs to backend compilers
/// (refer to [`ConstrainShardingOperation`](crate::ConstrainShardingOperation)) and the latter only changes through
/// manual collectives, and so are manual variation and reduction state. The axes that shard a dimension are ordered
/// from major to minor, so the axes up to and including an auto or manual axis determine which block of the dimension
/// each of its shards holds. Each dimension therefore keeps its axes up to and including its last auto or manual axis,
/// and only explicit axes minor to all of them may be added, removed, or reordered (e.g., `[m, x]` may become `[m]` or
/// `[m, y]` but not `[x, m]`). Since axis sizes fix those blocks, every mesh axis in such a preserved prefix keeps its
/// definition (i.e., its size and type) in the output mesh, explicit axes included (e.g., an explicit `x` in `[x, a]`
/// over an auto `a` cannot change its size), and so does every mesh axis that carries manual variation or reduction
/// state. Explicit axes may therefore be redefined or dropped only when they are minor to every auto and manual axis in
/// each dimension and carry no such state. Further inputs act as _witnesses_, meaning values whose types make output
/// identities available to the instruction (e.g., a value of type `f32[n]` lets `f32[2]` widen to `f32[n]`). Witnesses
/// carry no derivatives. Every output identity must occur in an input declaration, and the output type may not define
/// an identity. First-class dimension types define their names, so casts of dimension values can only target static
/// types.
///
/// The complete [input declarations](Self::input_types) preserve nominal bindings during concrete execution. Source,
/// output and witness declarations share their dimension identities: widening `f32[2]` to `f32[n]` next to a witness
/// of type `f32[n]` requires that witness to have extent `2`, and a static output next to two witnesses declared as
/// `dimension<n>` requires those witnesses to agree, so a cast can also act as a guard over its witnesses. Casting
/// `f32[a, b]` to `f32[b, a]` requires equal axis extents; a value with shape `[2, 3]` cannot change `a = 2, b = 3`
/// into `a = 3, b = 2`. Narrowing some axes while widening others uses the same rule. Array witnesses must carry the
/// declared identities nominally, whereas first-class dimension witnesses may instantiate them under another name
/// within their bounds, in which case the inferred output carries the supplied name (refer to [`Self::output_type`]).
/// Conditions beyond those expressed by these types belong in a [`GuardOperation`](crate::GuardOperation), which
/// checks declared type relationships. Type inference rejects static contradictions, and interpretation checks the
/// relationship over concrete inputs and materializes the requested representation, retaining exact concrete shapes.
/// Host arrays use dense row-major storage for an unspecified layout, share storage for sharding annotations, and
/// re-encode logical bytes when the layout changes. Optional layout and sharding constraints on inputs belong to their
/// declarations; the output independently requests its layout and sharding, and an output without a sharding is
/// untracked rather than replicated. Carrying an existing sharding forward requests no redistribution: backends
/// constrain placement only when the explicit placement changes. Derivative conversions request their tangent or exact
/// cotangent representation in one cast. Staged and compiled programs differ by mode:
///
///   - A [checked](Self::checked) cast requires the relationship at runtime. It declares
///     [`EffectClass::OrderedAssertion`], so conditional checks survive dead-code elimination and execute in assertion
///     order, and a violated relationship fails the program. Binding a checked cast to values whose types already prove
///     the relationship [normalizes](Self::normalized) it into an unchecked cast, so a checked cast in a program still
///     needs its check, and a proven single-input identity cast folds to its input.
///   - An [unchecked](Self::unchecked) cast assumes the relationship. It is pure and has no runtime cost beyond the
///     representation change (e.g., none at all for first-class dimensions). Transforms emit unchecked casts right
///     after proving their relationship: the tangent of a cast or of a [`GuardOperation`](crate::GuardOperation)
///     output is the unchecked cast of the input tangent, and the transpose of a cast is the unchecked cast of its
///     output cotangent back to its input's cotangent type, with witnesses that carry the required input identities.
///     A tangent cast is witnessed only for the identities that one of its sides does not carry, by a value whose type
///     carries each such identity, read from the primal input or output with [`CastType::IdentityWitnessOperation`]
///     (e.g., a `dimension<n>` scalar in the composite [`ArrayIrType`] family), or by that whole primal value in
///     families without such an operation. Forward-mode programs that interleave primal and tangent computations
///     also make it depend on the primal output, so that it stays behind the primal check.
///
/// Batching lifts the declared types by the mapped axis and binds the cast over the packed values. A proven one-input
/// identity keeps its carrier unchanged, including any bounded ragged axes. Every other cast over a bounded ragged
/// carrier is rejected, because the carrier's per-item extents live outside the packed types that the lifted cast
/// relates. For the same reason, the composite [`ArrayIrType`] policy rejects every other cast with a mapped
/// first-class dimension input (i.e., a source or witness holding one extent per batch item), since a lifted
/// dimension type cannot state per-item extents.
///
/// Raw transposition rejects conditional checked casts whose check needs an unavailable linear primal value or witness,
/// including cases whose shapes could be checked using known witnesses, because preserving assertion order requires
/// forward check scheduling (i.e., linearizing a checked program before transposing it, which executes its checks and
/// retains the necessary primal extents in the unchecked tangent program). An unchecked transpose needs every input
/// cotangent identity carried by its output cotangent, a known witness or a retained dimension source, and reports
/// unavailable extents rather than inventing names.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CastOperation<T: Type> {
    /// Refer to the documentation of [`input_types`](Self::input_types) for more information.
    input_types: Vec<T>,

    /// Refer to the documentation of [`output_type`](Self::output_type) for more information.
    output_type: T,

    /// Refer to the documentation of [`is_checked`](Self::is_checked) for more information.
    checked: bool,
}

impl<T: Type> CastOperation<T> {
    /// Creates a new checked [`CastOperation`] that returns its first input at `output_type`. `input_types` declares
    /// the value followed by its witnesses and must contain the first input. Staged and compiled programs check the
    /// relationship that the cast states at runtime. Refer to the documentation of [`CastOperation`] for more
    /// information.
    ///
    /// # Errors
    ///
    /// Returns a [`TypeError`] if `input_types` is empty.
    #[inline]
    pub fn checked(output_type: T, input_types: Vec<T>) -> Result<Self, TypeError> {
        if input_types.is_empty() {
            return Err(TypeError::invalid(format!("`{CAST_OPERATION_NAME}` requires a declared first input type")));
        }
        Ok(Self { input_types, output_type, checked: true })
    }

    /// Creates a new unchecked [`CastOperation`] that returns its first input at `output_type`. `input_types` declares
    /// the value followed by its witnesses and must contain the first input. Use this only right after proving the
    /// relationship that the cast states, and use [`Self::checked`] otherwise. Refer to the documentation of
    /// [`CastOperation`] for more information.
    ///
    /// <div class="warning">
    ///
    /// Staged and compiled programs assume the relationship of an unchecked cast; a program that executes one whose
    /// relationship does not hold silently computes with the wrong extents (e.g., reading padding as data).
    ///
    /// </div>
    ///
    /// # Errors
    ///
    /// Returns a [`TypeError`] if `input_types` is empty.
    #[inline]
    pub fn unchecked(output_type: T, input_types: Vec<T>) -> Result<Self, TypeError> {
        if input_types.is_empty() {
            return Err(TypeError::invalid(format!("`{CAST_OPERATION_NAME}` requires a declared first input type")));
        }
        Ok(Self { input_types, output_type, checked: false })
    }

    /// Returns the declared types of the value followed by its shape witnesses. These declarations retain the
    /// surrounding program's identities when actual values become concrete, and all follow its identity renamings.
    #[inline]
    pub fn input_types(&self) -> &[T] {
        &self.input_types
    }

    /// Returns the requested type of the first input's new representation. Type inference preserves identities carried
    /// by the inputs. When concrete inputs no longer carry names from the input or witness declarations, it substitutes
    /// their known exact extents so they do not become unbound names in a staged program, and when a first-class
    /// dimension witness instantiates a declared identity under another name, it carries that supplied name instead
    /// of the requested one.
    #[inline]
    pub fn output_type(&self) -> &T {
        &self.output_type
    }

    /// Returns whether this cast declares checked mode (refer to [`Self::checked`] and [`Self::unchecked`]). Every
    /// path that binds a cast to actual values normalizes it first (refer to [`Self::normalized`]), so a checked cast
    /// in a program still needs a runtime check, whereas an explicitly constructed checked cast that a program builder
    /// stages directly keeps its ordered effect even when backends omit its proven check.
    #[inline]
    pub fn is_checked(&self) -> bool {
        self.checked
    }

    /// Returns this cast normalized against the `input_types` it is about to be bound to (i.e., a checked cast whose
    /// relationship those types already prove needs no runtime check and becomes unchecked, so that it is pure and
    /// declares no ordered effect). Unchecked casts, casts that still need a check, and casts whose declaration the
    /// types do not admit (which binding then rejects) are returned unchanged. The proof uses the actual types rather
    /// than the stored declaration, because a declaration can admit weaker actual relationships that still require a
    /// check (e.g., `f32[n], f32[n]` admits `f32[2], f32[n]`), and later transforms only refine the bound types.
    #[inline]
    pub fn normalized(mut self, input_types: &[T]) -> Self
    where
        T: CastType,
    {
        if self.checked && matches!(T::infer_cast_output_type(&self, input_types), Ok((_, Some(_)))) {
            self.checked = false;
        }
        self
    }

    /// Validates the family-independent part of the cast contract against the supplied `input_types` (i.e., the value
    /// followed by its witnesses) and returns the candidate identities that the inferred output may preserve: every
    /// output identity that some supplied type carries. The supplied types must match the declaration in count and be
    /// admitted by it, no input or output may be a reference type, the output may not define a type identity, and
    /// every output identity must be carried by a declared input. A candidate is not necessarily preserved, because a
    /// supplied type can carry the same identity in an unrelated role (e.g., a supplied `dimension<n>` witnessing a
    /// declared `dimension<m>`). Family inference must therefore keep a candidate only when the correspondence class
    /// of the declared identity reaches that same supplied identity, and resolve the other output identities through
    /// their own classes. [`CastType::infer_cast_output_type`] implementations call this first and then validate and
    /// infer the family-specific relationship; the built-in families filter the candidates in this way and solve the
    /// dimension relationships.
    ///
    /// # Errors
    ///
    /// Returns a [`TypeError`] describing the first violated requirement.
    pub fn validate_declaration(&self, input_types: &[T]) -> Result<Vec<T::Identity>, TypeError> {
        check_count!("input", input_types, self.input_types.len(), TypeError);

        // Witnesses bring names into scope but are still cast inputs (i.e., they cannot bypass the reference
        // restriction). Inspect supplied types as well as declarations so neither side can hide a reference.
        if let Some(reference) = input_types
            .iter()
            .chain(&self.input_types)
            .chain([&self.output_type])
            .find(|r#type| r#type.is_reference())
        {
            return Err(TypeError::invalid(format!(
                "`{CAST_OPERATION_NAME}` does not support reference type `{reference}`",
            )));
        }

        // A cast reuses input identities; it cannot introduce a new binding. In particular, a named first-class
        // dimension defines its name, whereas an array shape only uses its dimension names.
        if self.output_type.identities().any(|(position, _)| position == TypeIdentityPosition::Definition) {
            return Err(TypeError::invalid(format!(
                "`{}` output type `{}` cannot define a type identity",
                CAST_OPERATION_NAME, self.output_type,
            )));
        }

        // Validate the complete signature together to reject conflicting refinements of repeated identities across
        // inputs and witnesses. Admission may leave runtime obligations, so success here does not prove the cast's
        // relationship. The caller's family-specific inference must still relate the inputs to the requested output.
        T::Refinements::new(&self.input_types, input_types)?;

        // Live names are preserved only in the final representation; all input and output constraints still participate
        // in the complete shape comparison. For example, a witness declared as `f32[n]` may arrive as `f32[2]`: the
        // declaration makes `n` legal in the requested output, but the inferred output must resolve it to `2` rather
        // than introduce an unbound `n` when no supplied input still carries that name.
        let mut preserved = Vec::new();
        for (_, identity) in self.output_type.identities() {
            let carries = |r#type: &T| r#type.identities().any(|(_, available)| available == identity);
            if !self.input_types.iter().any(carries) {
                return Err(TypeError::invalid(format!(
                    "`{CAST_OPERATION_NAME}` output identity `{identity}` is not carried by a declared input",
                )));
            }
            if input_types.iter().any(carries) {
                preserved.push(identity.clone());
            }
        }
        Ok(preserved)
    }

    /// Binds the tangent of a value whose `primal` was converted into `output` by a cast or by a guard, returning the
    /// `tangent` at the tangent type of `output` through an unchecked cast in the tangent context. The relationship is
    /// proven by the primal conversion, so the tangent cast only needs these witnesses:
    ///
    ///   - One witness per identity that the tangent carries but the output tangent type does not (e.g., `n` when
    ///     narrowing `f32[n]` to `f32[2]`), read from `primal`, so that the transpose of the tangent cast, which
    ///     converts a cotangent back to the tangent's type, finds that identity. One witness per identity that the
    ///     output tangent type carries but the tangent does not (e.g., `n` when widening `f32[2]` to `f32[n]`), read
    ///     from `output`, which the tangent cast itself requires. Each witness is the value of the identity's
    ///     [`CastType::IdentityWitnessOperation`] bound in the primal context (e.g., a `dimension<n>` scalar in the
    ///     composite [`ArrayIrType`] family). Families without such an operation retain the whole value instead.
    ///   - When `needs_ordering` holds, a data dependency on `output` so that the tangent stays behind the primal check
    ///     and its witness producers in a fused forward-mode program. A witness read from `output` already provides
    ///     it. Otherwise, `output` itself is a witness, unless a whole `primal` is retained, which is then re-cast at
    ///     its own type with `output` as its witness so that one value carries both.
    ///
    /// Witnesses reach the tangent context through [`DifferentiationContext::primal_to_tangent`]. A partitioned
    /// tangent program runs only after the primal program completes, so it retains only identity witnesses as its
    /// residuals. Scalar identity witnesses keep whole arrays out of those residuals and, having static types, can
    /// cross static region boundaries (e.g., those of [`ShardMapOperation`](crate::ShardMapOperation) bodies).
    ///
    /// # Parameters
    ///
    ///   - `context`: [`DifferentiationContext`] whose primal side produced `output` and whose tangent side receives
    ///     the converted tangent.
    ///   - `primal`: Primal value that was converted.
    ///   - `output`: Converted primal value, whose tangent type the returned tangent has.
    ///   - `tangent`: Tangent of `primal`.
    ///   - `needs_ordering`: Whether the tangent must depend on `output` even when the output tangent type carries
    ///     every identity of the tangent (i.e., in a fused program whose primal conversion still runs a check or
    ///     depends on witness producers).
    pub(crate) fn bind_tangent<
        C: Context<Type = T, Operation: From<Self> + From<T::IdentityWitnessOperation>>,
        P: DifferentiationPolicy<C>,
    >(
        context: &DifferentiationContext<C, P>,
        primal: &C::Value,
        output: &C::Value,
        tangent: &C::Value,
        needs_ordering: bool,
    ) -> Result<C::Value, ProgramError>
    where
        T: DifferentiableType + CastType,
    {
        let source_type = tangent.r#type().into_owned();
        let tangent_type = output.r#type().tangent()?;

        let lost = Self::identity_witness_operations(primal.r#type().as_ref(), &source_type, &tangent_type)?;
        let introduced = Self::identity_witness_operations(output.r#type().as_ref(), &tangent_type, &source_type)?;

        let read = |value: &C::Value, operation: T::IdentityWitnessOperation| -> Result<C::Value, ProgramError> {
            let mut outputs = context.primal().bind(operation, Vec::new(), std::slice::from_ref(value))?;
            check_count!("output", outputs, 1, ProgramError);
            Ok(context.primal_to_tangent(outputs.remove(0))?)
        };

        // A witness read from `output` (or `output` itself) orders the tangent behind the primal conversion. Without
        // one, a whole retained source is re-cast at its own type with `output` as its witness, so that one value
        // carries both its identities and that dependency, and otherwise `output` itself becomes the last witness.
        let mut ordered = !needs_ordering || introduced.as_ref().is_none_or(|operations| !operations.is_empty());
        let mut witnesses = Vec::new();
        match lost {
            Some(operations) => {
                for operation in operations {
                    witnesses.push(read(primal, operation)?);
                }
            }
            None if ordered => witnesses.push(context.primal_to_tangent(primal.clone())?),
            None => {
                ordered = true;
                let primal_type = primal.r#type().into_owned();
                let source = Self::bind_unchecked(context.primal(), primal, std::slice::from_ref(output), primal_type)?;
                witnesses.push(context.primal_to_tangent(source)?);
            }
        }

        match introduced {
            Some(operations) => {
                for operation in operations {
                    witnesses.push(read(output, operation)?);
                }
            }
            None => witnesses.push(context.primal_to_tangent(output.clone())?),
        }

        if !ordered {
            witnesses.push(context.primal_to_tangent(output.clone())?);
        }

        Self::bind_unchecked(context.tangent(), tangent, &witnesses, tangent_type)
    }

    /// Binds a checked [`CastOperation`] in `context`, capturing the current types of `value` and `witnesses` as its
    /// declaration. Relationships already proven by those types normalize to unchecked casts before binding, while
    /// conditional relationships retain their ordered check. Witness producers remain explicit dependencies.
    fn bind_checked<C: Context<Type = T, Operation: From<Self>>>(
        context: &C,
        value: &C::Value,
        witnesses: &[C::Value],
        output_type: T,
    ) -> Result<C::Value, ProgramError>
    where
        T: CastType,
    {
        let input_types =
            std::iter::once(value).chain(witnesses).map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        Self::checked(output_type, input_types.clone())?
            .normalized(&input_types)
            .bind(context, value, witnesses)
    }

    /// Returns `value` at `output_type` through an unchecked [`CastOperation`] bound in `context` with the provided
    /// `witnesses` (at their current types), or `value` itself when it already has that type and there are no
    /// witnesses. Witness producers remain explicit dependencies even for identity conversions. Transformation rules
    /// use this to bind into their target context (e.g., the tangent context of a forward-mode rule), and must have
    /// proven the relationship that the cast states.
    fn bind_unchecked<C: Context<Type = T, Operation: From<Self>>>(
        context: &C,
        value: &C::Value,
        witnesses: &[C::Value],
        output_type: T,
    ) -> Result<C::Value, ProgramError> {
        let input_types = std::iter::once(value).chain(witnesses).map(|input| input.r#type().into_owned()).collect();
        Self::unchecked(output_type, input_types)?.bind(context, value, witnesses)
    }

    /// Binds this [`CastOperation`] in `context`, preserving witnesses and validating an unchecked identity before
    /// returning its input directly. The context's operation family owns validation, so unchecked transformation
    /// helpers need no additional type-family bounds.
    fn bind<C: Context<Type = T, Operation: From<Self>>>(
        self,
        context: &C,
        value: &C::Value,
        witnesses: &[C::Value],
    ) -> Result<C::Value, ProgramError> {
        let input_type = value.r#type();
        let identity = !self.checked && witnesses.is_empty() && input_type.as_ref() == &self.output_type;
        let operation = C::Operation::from(self);
        if identity {
            // Equal types can still be invalid cast declarations (e.g., a reference or a named dimension output).
            operation.infer_output_types(std::slice::from_ref(input_type.as_ref()), &[])?;
            return Ok(value.clone());
        }
        let inputs = std::iter::once(value).chain(witnesses).cloned().collect::<Vec<_>>();
        let mut outputs = context.bind(operation, Vec::new(), &inputs)?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }

    /// Returns the [`CastType::IdentityWitnessOperation`]s that read every identity of `from` that `to` does not carry
    /// from a value of type `value_type`, one per identity in order of first occurrence, or `None` when one of them has
    /// no such operation and the whole value must be retained instead (refer to [`Self::bind_tangent`]).
    fn identity_witness_operations(
        value_type: &T,
        from: &T,
        to: &T,
    ) -> Result<Option<Vec<T::IdentityWitnessOperation>>, TypeError>
    where
        T: CastType,
    {
        let mut identities = Vec::new();
        for (_, identity) in from.identities() {
            if !to.identities().any(|(_, available)| available == identity) && !identities.contains(&identity) {
                identities.push(identity);
            }
        }
        let mut operations = Vec::with_capacity(identities.len());
        for identity in identities {
            match T::identity_witness_operation(value_type, identity)? {
                Some(operation) => operations.push(operation),
                None => return Ok(None),
            }
        }
        Ok(Some(operations))
    }

    /// Returns the inferred output type for direct interpretation over values of the supplied `input_types`,
    /// after proving the complete relationship in both modes. Concrete extents complete the proof, whereas symbolic
    /// relationships that still need a runtime check are rejected with [`ProgramError::Concretization`], since direct
    /// interpretation cannot emit a deferred runtime check.
    ///
    /// # Parameters
    ///
    ///   - `input_types`: Types of the value followed by its witnesses.
    ///   - `region_interfaces`: Interfaces of the attached regions, which must be empty.
    fn infer_interpreted_output_type(
        &self,
        input_types: &[T],
        region_interfaces: &[RegionInterface<T>],
    ) -> Result<T, ProgramError>
    where
        T: CastType,
    {
        check_count!("input", input_types, self.input_types.len(), TypeError);
        check_count!("region", region_interfaces, 0, TypeError);
        let (_, interpreted) = T::infer_cast_output_type(self, input_types)?;
        interpreted.ok_or_else(|| ProgramError::Concretization {
            message: format!(
                "`{CAST_OPERATION_NAME}` cannot check symbolic type relationships during interpretation; \
                 bind the operation through its context",
            ),
        })
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
        check_count!("input", input_types, self.input_types.len(), TypeError);
        check_count!("region", region_interfaces, 0, TypeError);
        let (output, _) = T::infer_cast_output_type(self, input_types)?;
        Ok(vec![output])
    }

    fn fold(
        &self,
        input_types: &[T],
        region_interfaces: &[RegionInterface<T>],
    ) -> Result<Option<Vec<OperationFoldOutput>>, TypeError> {
        check_count!("input", input_types, self.input_types.len(), TypeError);
        check_count!("region", region_interfaces, 0, TypeError);
        let (_, interpreted) = T::infer_cast_output_type(self, input_types)?;

        // Witnesses can carry the check that justifies a transform's cast. Keep their producers as explicit value
        // dependencies even when the returned representation is unchanged.
        Ok((input_types.len() == 1 && interpreted.as_ref() == Some(&input_types[0]))
            .then(|| vec![OperationFoldOutput::Input(0)]))
    }

    #[inline]
    fn effects(&self) -> Cow<'_, Effects> {
        if self.checked {
            Cow::Owned(Effects::explicit(EffectClasses::single(EffectClass::OrderedAssertion)))
        } else {
            Cow::Borrowed(Effects::empty())
        }
    }

    fn rename_type_identities(&self, renaming: &TypeIdentityRenaming<T::Identity>) -> Result<Self, TypeError> {
        // Every stored declaration names identities of the surrounding program.
        Ok(Self {
            input_types: self
                .input_types
                .iter()
                .map(|r#type| r#type.rename_identities(renaming))
                .collect::<Result<_, _>>()?,
            output_type: self.output_type.rename_identities(renaming)?,
            checked: self.checked,
        })
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("type", &self.output_type)?;
            operation.list("inputs", &self.input_types)?;
            if self.checked {
                operation.field("checked", true)?;
            }
            Ok(())
        })
    }
}

impl_reference_dischargeable_operation!(@reference_free <T> CastOperation<T> where T: CastType);

impl<C: Domain<Type: CastType, Value: Cast<C::Type>>> InterpretableOperation<C> for CastOperation<C::Type> {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        // Concrete values carry their extents in their types. For symbolic values, conditional facts cannot replace a
        // runtime check: interpretation accepts only relationships already proven by the types, in both modes.
        let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        let output_type = self.infer_interpreted_output_type(
            &input_types,
            &driver.regions().map(|region| region.interface()).collect::<Vec<_>>(),
        )?;
        if inputs.len() == 1 && inputs[0].r#type().as_ref() == &output_type {
            return Ok(vec![inputs[0].clone()]);
        }

        // The complete original declaration has been proven. Materialize the requested representation once;
        // the value capability retains every witness producer, including for an unchanged representation.
        Ok(vec![inputs[0].cast_unchecked(output_type, &inputs[1..])?])
    }
}

impl<C: Context<Type: CastType, Operation: From<CastOperation<C::Type>>>> PartiallyEvaluatableOperation<C>
    for CastOperation<C::Type>
{
    fn partially_evaluate<D: PartialEvaluationDriver<C>>(
        &self,
        context: &PartialEvaluationContext<C>,
        driver: &D,
        inputs: &[PartialEvaluationValue<C::Value>],
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
        // Residual inputs may be more refined than the original declaration (e.g., a known value with an exact shape),
        // so normalize against them before the default rule folds or residualizes the cast: a proven cast is pure and
        // must not be ordered behind earlier assertions.
        let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        context.fold_or_residualize(
            self.clone().normalized(&input_types),
            driver.regions().map(|region| region.to_program()).collect(),
            inputs,
        )
    }
}

impl<C: Context<Type: CastType, Operation: From<CastOperation<C::Type>>>, P: BatchingPolicy<C>> BatchableOperation<C, P>
    for CastOperation<C::Type>
{
    fn batch<D: BatchingDriver<C, P>>(
        &self,
        context: &BatchingContext<C, P>,
        driver: &D,
        inputs: &[P::Batch],
    ) -> Result<BatchedOutputs<C, P>, BatchingError> {
        // Validate the original declaration before lifting can erase its layout constraints. Reconstructed per-item
        // types have already lost layout and may carry manual variation introduced by the batch axis, so they are not
        // the original invocation's representations. The parent validates the supplied values against the lifted
        // declaration, preserving source/witness shapes, layout and sharding unrelated to the mapped axes.
        check_count!("input", inputs, self.input_types.len(), ProgramError);
        self.infer_output_types(
            &self.input_types,
            &driver.regions().map(|region| region.interface()).collect::<Vec<_>>(),
        )?;

        // An exact one-input identity has no relationship left to check. Keep its original carrier, including physical
        // layout and ragged extents that are absent from the logical per-item type. Witnesses or different declarations
        // can impose additional relationships, so they must continue through the lifted primitive.
        if let [input] = inputs
            && self.input_types[0] == self.output_type
            && P::unbatched_type(input).as_ref() == &self.output_type
        {
            return Ok(vec![input.clone()].into());
        }

        let input_types = self
            .input_types
            .iter()
            .zip(inputs)
            .map(|(r#type, input)| P::batched_type(context, input, r#type))
            .collect::<Result<Vec<_>, _>>()?;
        let output_type = P::batched_type(context, &inputs[0], &self.output_type)?;
        let values = inputs.iter().map(|input| P::value(input).clone()).collect::<Vec<_>>();
        let operation = CastOperation { input_types, output_type, checked: self.checked }
            .normalized(&values.iter().map(|value| value.r#type().into_owned()).collect::<Vec<_>>());
        let mut outputs = context.parent().bind(operation, Vec::new(), &values)?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(vec![P::batch(outputs.remove(0), P::batch_axis(&inputs[0]))?].into())
    }
}

impl<T: DifferentiableType + CastType, C: Context<Type = T>> DifferentiableOperation<C> for CastOperation<T>
where
    C::Operation: From<CastOperation<T>> + From<T::IdentityWitnessOperation>,
{
    fn jvp<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        // The tangent is an unchecked cast behind the primal conversion (refer to `CastOperation::bind_tangent`). Shape
        // witnesses carry no derivatives. In a fused Jacobian-Vector Product (JVP), the tangent must stay behind the
        // primal check and behind the witness producers, which may themselves be the checks that justify an unchecked
        // cast.
        let primals = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        let primal_types = primals.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        let operation = self.clone().normalized(&primal_types);
        let needs_ordering = !P::PRIMAL_COMPLETES_BEFORE_TANGENT && (primals.len() > 1 || operation.checked);
        let mut outputs = context.primal().bind(operation, Vec::new(), &primals)?;
        check_count!("output", outputs, 1, ProgramError);
        let output = outputs.remove(0);
        Ok(vec![match inputs[0].tangent() {
            MaybeZero::Zero(_) => DifferentiationDual::new_with_zero_tangent(output)?,
            MaybeZero::Value(tangent) => {
                let tangent = Self::bind_tangent(context, &primals[0], &output, tangent, needs_ordering)?;
                DifferentiationDual::new(output, tangent)?
            }
        }])
    }
}

impl<
    T: DifferentiableType + CastType,
    V: Value<Type = T>,
    O: Operation<Type = T> + SupportsCotangentAdd + From<CastOperation<T>>,
> TransposableOperation<V, O> for CastOperation<T>
where
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TranspositionContext<V, O>,
        driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
        accumulators: &[CotangentAccumulator],
    ) -> Result<(), DifferentiationError> {
        // The transpose of the identity is the identity, at the cotangent type of the linear first input, converted by
        // an unchecked cast after proving its relationship. Conditional checked primal casts must be linearized first.
        // Known witnesses keep that type's identities in scope. Linear witnesses have zero cotangents and need no
        // primal value unless their extents are required below; retained sources can supply missing identities.
        check_count!("output", outputs, 1, ProgramError);
        check_count!("accumulator", accumulators, inputs.len(), DifferentiationError);

        let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        let region_interfaces = driver.regions().map(|region| region.interface()).collect::<Vec<_>>();

        check_count!("input", input_types, self.input_types.len(), TypeError);
        check_count!("region", region_interfaces, 0, TypeError);

        let (_, interpreted) = T::infer_cast_output_type(self, &input_types)?;
        if self.checked && inputs.iter().any(PartialValue::is_unknown) && interpreted.is_none() {
            let missing = if inputs[0].is_unknown() { "value" } else { "witnesses" };
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "cannot transpose a checked `{CAST_OPERATION_NAME}` without the primal {missing} its check \
                     needs; linearize the program first, or use `cast_unchecked` with witnesses for a tangent that an \
                     earlier check already proved",
                ),
            }
            .into());
        }

        if !accumulators[0].is_needed() {
            return Ok(());
        }

        let mut witnesses = inputs[1..].iter().filter_map(|input| input.as_known().cloned()).collect::<Vec<_>>();
        let cotangent_type = inputs[0].r#type().cotangent()?;
        let contribution = match &outputs[0] {
            MaybeZero::Zero(_) => MaybeZero::Zero(cotangent_type),
            MaybeZero::Value(cotangent) => {
                // The accumulator requires the original nominal type, even when the forward cast fixed its extent.
                // Retained primal extents can witness that name without changing the accumulator's contract.
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
                MaybeZero::Value(CastOperation::bind_unchecked(&**context, cotangent, &witnesses, cotangent_type)?)
            }
        };

        accumulators[0].accumulate(context, contribution)
    }
}

/// Infers the output of a [`CastOperation`] for this type family, validating the complete relationship between its
/// input, output, and witness types. This includes dimension extents and identities as well as other type properties.
/// The value and representation constraints are described by [`CastOperation`]. This trait validates them together
/// with the complete source and witness declarations.
///
/// Custom type families implement this trait to use [`CastOperation`]. Their interpreted values implement [`Cast`]
/// to materialize any requested representation changes. The built-in implementations use [`TypeRefinements`] to
/// validate dimension and identity relationships and infer the output's representation.
pub trait CastType: Type {
    /// [`Operation`] that reads one identity of a value of this type into a value whose type carries that identity
    /// (e.g., the [`DimensionSizeOperation`] that reads the extent `n` of an `f32[n]` array as a `dimension<n>`).
    /// Differentiation binds it to give a tangent cast a scalar witness for an identity in place of the whole primal
    /// value (refer to [`CastOperation`]), which keeps whole arrays out of the residuals of partitioned programs.
    /// Families without such an operation name [`CastOperation<Self>`], which every operation family that
    /// differentiates casts already converts from, and never select it, so that generic rules that bind
    /// witnesses need no operation conversions beyond those of the cast itself.
    type IdentityWitnessOperation: Operation<Type = Self>;

    /// Validates `operation` against `input_types` (the value followed by its witnesses) and returns the inferred
    /// output type together with an optional output representation for direct interpretation. The latter is present
    /// only when the supplied types prove the complete shape without a runtime check; its identities must be carried by
    /// live inputs, with erased declared names resolved to proven extents. Every requested output name must occur in
    /// the declared input signature, consistently for direct interpretation and staging.
    ///
    /// Both checked and unchecked casts use the same type rules: statically contradictory relationships are errors,
    /// while relationships requiring runtime checks are permitted. A missing interpretation representation permits
    /// staging but requires concrete shapes before interpretation, including for unchecked casts. The mode determines
    /// whether staged and compiled programs check or assume those relationships. Optional layout and sharding changes
    /// do not require runtime shape checks and retain the requested representation fields.
    ///
    /// The declared input signature must contain the first input and admit the supplied types. The input and requested
    /// output must describe the same value: corresponding dimensions have equal extents and shared identities agree
    /// across the input, output, and witnesses. The remaining type requirements are described by [`CastOperation`].
    /// The inferred output preserves names carried by corresponding live inputs and resolves names lost during
    /// concretization to exact extents (refer to [`CastOperation::output_type`]). Invalid input counts must return an
    /// error, even when called directly.
    fn infer_cast_output_type(
        operation: &CastOperation<Self>,
        input_types: &[Self],
    ) -> Result<(Self, Option<Self>), TypeError>;

    /// Returns the [`Self::IdentityWitnessOperation`] that reads `identity` from a value of type `value_type`, or
    /// [`None`] when this family has no operation for that identity (e.g., because `value_type` does not carry it, or
    /// because the family has no such operation), in which case transforms retain the whole value as the witness. The
    /// selected operation takes the value as its only input and returns one value whose type carries `identity`. The
    /// default implementation returns [`None`].
    ///
    /// # Errors
    ///
    /// Returns a [`TypeError`] if `value_type` carries `identity` but the selected operation cannot be constructed to
    /// read it (e.g., in [`ArrayIrType`], when the lower bound of the identity exceeds the largest extent that a
    /// first-class dimension can represent). Transforms propagate this error rather than retaining the whole value, so
    /// that, for example, differentiating such a cast fails loudly instead of silently retaining a whole array as a
    /// residual.
    #[inline]
    fn identity_witness_operation(
        value_type: &Self,
        identity: &Self::Identity,
    ) -> Result<Option<Self::IdentityWitnessOperation>, TypeError> {
        let _ = (value_type, identity);
        Ok(None)
    }
}

impl CastType for DataType {
    // This family has no identities, so it keeps the default witness selector, which returns `None`. The unused
    // `CastOperation<Self>` placeholder reuses the conversion that cast differentiation already requires.
    type IdentityWitnessOperation = CastOperation<Self>;

    #[inline]
    fn infer_cast_output_type(
        operation: &CastOperation<Self>,
        input_types: &[Self],
    ) -> Result<(Self, Option<Self>), TypeError> {
        operation.validate_declaration(input_types)?;
        if input_types[0] == operation.output_type {
            Ok((operation.output_type, Some(operation.output_type)))
        } else {
            Err(TypeError::invalid(format!(
                "`{}` cannot convert type `{}` to unrelated type `{}`",
                CAST_OPERATION_NAME, input_types[0], operation.output_type,
            )))
        }
    }
}

impl CastType for ArrayType {
    // This universe has no first-class dimension values, so the default `None` selector retains the whole array.
    // `CastOperation<Self>` is an unused placeholder that reuses cast differentiation's existing conversion bound.
    type IdentityWitnessOperation = CastOperation<Self>;

    #[inline]
    fn infer_cast_output_type(
        operation: &CastOperation<Self>,
        input_types: &[Self],
    ) -> Result<(Self, Option<Self>), TypeError> {
        // Lift the complete declaration and supplied types so native and mixed casts share one inference contract.
        let operation = CastOperation {
            input_types: operation.input_types.iter().cloned().map(Into::into).collect(),
            output_type: operation.output_type.clone().into(),
            checked: operation.checked,
        };

        let input_types = input_types.iter().cloned().map(Into::into).collect::<Vec<_>>();
        let (output, interpreted) = ArrayIrType::infer_cast_output_type(&operation, &input_types)?;

        // The composite contract preserves the requested output's member kind.
        let project = |output| match output {
            ArrayIrType::Array(output) => output,
            _ => unreachable!(),
        };

        Ok((project(output), interpreted.map(project)))
    }
}

impl CastType for DimensionType {
    // A dimension value already carries its identity, so the default `None` selector retains the value itself.
    // `CastOperation<Self>` is an unused placeholder that reuses cast differentiation's existing conversion bound.
    type IdentityWitnessOperation = CastOperation<Self>;

    #[inline]
    fn infer_cast_output_type(
        operation: &CastOperation<Self>,
        input_types: &[Self],
    ) -> Result<(Self, Option<Self>), TypeError> {
        // Lift the complete declaration and supplied types so native and mixed casts share one inference contract.
        let operation = CastOperation {
            input_types: operation.input_types.iter().cloned().map(Into::into).collect(),
            output_type: operation.output_type.clone().into(),
            checked: operation.checked,
        };

        let input_types = input_types.iter().cloned().map(Into::into).collect::<Vec<_>>();
        let (output, interpreted) = ArrayIrType::infer_cast_output_type(&operation, &input_types)?;

        // The composite contract preserves the requested output's member kind.
        let project = |output| match output {
            ArrayIrType::Dimension(output) => output,
            _ => unreachable!(),
        };

        Ok((project(output), interpreted.map(project)))
    }
}

impl CastType for ArrayIrType {
    type IdentityWitnessOperation = DimensionSizeOperation;

    #[inline]
    fn infer_cast_output_type(
        operation: &CastOperation<Self>,
        input_types: &[Self],
    ) -> Result<(Self, Option<Self>), TypeError> {
        if let Some(r#type) = input_types
            .iter()
            .chain(&operation.input_types)
            .chain([&operation.output_type])
            .find(|r#type| matches!(r#type, Self::Ragged(_)))
        {
            return Err(TypeError::invalid(format!(
                "`{CAST_OPERATION_NAME}` does not support first-class ragged type `{type}`",
            )));
        }
        let preserved = operation.validate_declaration(input_types)?;
        match (&input_types[0], &operation.output_type) {
            (Self::Array(input), Self::Array(output)) => {
                // The validation above guarantees that the declared first input has the supplied input's member kind.
                let Self::Array(declared) = &operation.input_types[0] else { unreachable!() };

                // Arrays preserve the input value properties and static extents. Layout and explicit placement may
                // change, while auto/manual placement, manual variation, and reduction state remain constrained.
                if declared.unreduced_axes() != input.unreduced_axes()
                    || declared.reduced_axes() != input.reduced_axes()
                {
                    return Err(TypeError::invalid(format!(
                        "`{CAST_OPERATION_NAME}` supplied input changes the declared reduction state",
                    )));
                }

                if input.data_type() != output.data_type()
                    || input.rank() != output.rank()
                    || input.memory() != output.memory()
                    || input.unreduced_axes() != output.unreduced_axes()
                    || input.reduced_axes() != output.reduced_axes()
                    || input.shape().dimensions().iter().zip(output.shape().dimensions()).any(|(input, output)| {
                        matches!(
                            (input, output),
                            (Dimension::Static(input), Dimension::Static(output)) if input != output,
                        )
                    })
                {
                    return Err(TypeError::invalid(format!(
                        "`{CAST_OPERATION_NAME}` cannot convert type `{input}` to unrelated type `{output}`",
                    )));
                }

                // These are semantic dependencies, not optional placement constraints. An absent sharding is invariant,
                // and changing variation requires the corresponding operation (whose transpose performs the required
                // reduction).
                if ArrayType::check_matching_manual_variation(CAST_OPERATION_NAME, &[declared, input]).is_err() {
                    return Err(TypeError::invalid(format!(
                        "`{CAST_OPERATION_NAME}` supplied input changes the declared manual variation",
                    )));
                }

                if ArrayType::check_matching_manual_variation(CAST_OPERATION_NAME, &[input, output]).is_err() {
                    return Err(TypeError::invalid(format!(
                        "`{CAST_OPERATION_NAME}` cannot change varying manual axes; use `parallel_vary` to add \
                         variation or a reduction to remove it",
                    )));
                }

                // Placement follows the rules of `reshard` (i.e., the type system tracks placement over explicit
                // mesh axes, so the output may request a new one, whereas placement over auto axes belongs to backend
                // compilers and transitions over manual axes require manual collectives). The output therefore names
                // the same auto and manual axes as the input on every dimension, in the same position relative to the
                // explicit axes (refer to the check below). Carrying them forward requests no redistribution, and an
                // output without a sharding stays untracked rather than requesting replication.
                if let Some(output_sharding) = output.sharding() {
                    let named_axes = |sharding: &Sharding, dimension: usize, axis_type: MeshAxisType| match &sharding
                        .dimensions()[dimension]
                    {
                        ShardingDimension::Sharded(axis_names) => axis_names
                            .iter()
                            .filter(|name| sharding.mesh().axis_type(name) == Some(axis_type))
                            .cloned()
                            .collect::<Vec<_>>(),
                        ShardingDimension::Replicated | ShardingDimension::Unconstrained => Vec::new(),
                    };

                    for (axis_type, remedy) in [
                        (
                            MeshAxisType::Auto,
                            format!("use `{CONSTRAIN_SHARDING_OPERATION_NAME}` to constrain placement over them"),
                        ),
                        (MeshAxisType::Manual, "use manual collectives for transitions over them".to_owned()),
                    ] {
                        if (0..output.rank()).any(|dimension| {
                            named_axes(output_sharding, dimension, axis_type)
                                != input
                                    .sharding()
                                    .map(|sharding| named_axes(sharding, dimension, axis_type))
                                    .unwrap_or_default()
                        }) {
                            return Err(TypeError::invalid(format!(
                                "`{CAST_OPERATION_NAME}` cannot change placement over {axis_type} mesh axes; {remedy}",
                            )));
                        }
                    }

                    // Comparing names per axis type is not sufficient, because the axes that shard a dimension are
                    // ordered from major to minor: the sizes of all axes up to and including an auto or manual axis
                    // determine which block of the dimension each of its shards holds. Reordering those axes, or
                    // adding or removing an explicit axis major to one of them, would move data between auto or manual
                    // shards (i.e., perform an implicit collective). Each dimension must therefore keep its axes up to
                    // and including its last auto or manual axis, and only explicit axes minor to them may change.
                    fn preserved_prefix(sharding: &Sharding, dimension: usize) -> &[String] {
                        match &sharding.dimensions()[dimension] {
                            ShardingDimension::Sharded(axis_names) => {
                                let end = axis_names
                                    .iter()
                                    .rposition(|name| sharding.mesh().axis_type(name) != Some(MeshAxisType::Explicit))
                                    .map_or(0, |index| index + 1);
                                &axis_names[..end]
                            }
                            ShardingDimension::Replicated | ShardingDimension::Unconstrained => &[],
                        }
                    }

                    if let Some(dimension) = (0..output.rank()).find(|dimension| {
                        preserved_prefix(output_sharding, *dimension)
                            != input
                                .sharding()
                                .map(|sharding| preserved_prefix(sharding, *dimension))
                                .unwrap_or_default()
                    }) {
                        return Err(TypeError::invalid(format!(
                            "`{CAST_OPERATION_NAME}` cannot change the mesh axes up to the last auto or manual mesh \
                             axis of dimension {dimension}; only explicit mesh axes minor to every auto and manual \
                             axis may change",
                        )));
                    }

                    // The checks above compare axis names only, but the size and type of a mesh axis are semantic:
                    // the sizes of all axes in the preserved prefix of a dimension, explicit axes included, fix the
                    // blocks of its auto and manual shards. Every axis in such a prefix, and every axis of any type
                    // that carries manual variation or reduction state, must therefore keep its definition. An
                    // explicit axis is exempt only when it is minor to every auto and manual axis of each dimension
                    // that it shards and carries no such state. The output mesh may still add or drop other axes
                    // (e.g., to request an explicit redistribution).
                    if let Some(input_sharding) = input.sharding() {
                        let input_mesh = input_sharding.mesh();
                        let output_mesh = output_sharding.mesh();
                        let preserved_axes = (0..input_sharding.dimensions().len())
                            .flat_map(|dimension| preserved_prefix(input_sharding, dimension))
                            .chain(input_sharding.varying_manual_axes())
                            .chain(input_sharding.unreduced_axes())
                            .chain(input_sharding.reduced_axes())
                            .filter_map(|axis_name| {
                                input_mesh.axis_index(axis_name).map(|index| &input_mesh.axes()[index])
                            });
                        for input_axis in preserved_axes {
                            let output_axis =
                                output_mesh.axis_index(input_axis.name()).map(|index| &output_mesh.axes()[index]);
                            if output_axis != Some(input_axis) {
                                let output_axis = output_axis
                                    .map_or_else(|| "an undefined axis".to_owned(), |axis| format!("`{axis}`"));
                                return Err(TypeError::invalid(format!(
                                    "`{}` cannot redefine preserved {} mesh axis `{}` as {}",
                                    CAST_OPERATION_NAME,
                                    input_axis.r#type(),
                                    input_axis,
                                    output_axis,
                                )));
                            }
                        }
                    }
                }

                // A requested layout must address the output shape, so that a malformed request fails here rather than
                // only when interpretation materializes the output or a backend lowers it.
                output.validate_layout().map_err(|error| {
                    TypeError::invalid(format!("`{CAST_OPERATION_NAME}` output layout is invalid: {error}"))
                })?;
            }
            (Self::Dimension(input), Self::Dimension(output)) => {
                if matches!((input.extent(), output.extent()), (Some(input), Some(output)) if input != output) {
                    return Err(TypeError::invalid(format!(
                        "`{}` cannot convert type `{}` to unrelated type `{}`",
                        CAST_OPERATION_NAME, input, output,
                    )));
                }
            }
            _ => {
                return Err(TypeError::invalid(format!(
                    "`{}` cannot convert type `{}` to unrelated type `{}`",
                    CAST_OPERATION_NAME, &input_types[0], &operation.output_type,
                )));
            }
        }

        // Solve all dimensions together: input and output declarations share the declared scope, and supplied inputs
        // share the actual scope. The output observes the first actual input, admitting conditional narrowing and
        // differently bounded identities without changing the meaning of a shared declared name.
        let declared = operation.input_types.iter().map(ArrayIrType::dimensions).collect::<Vec<_>>();
        let actual = input_types.iter().map(ArrayIrType::dimensions).collect::<Vec<_>>();
        let output = operation.output_type.dimensions();
        let mut facts = declared
            .iter()
            .zip(&actual)
            .flat_map(|(declared, actual)| {
                declared.iter().zip(actual.iter()).flat_map(|(declared, actual)| {
                    DimensionRefinements::refinement_facts(declared, actual, DimensionComparison::LocalDeclarations)
                })
            })
            .collect::<Vec<_>>();
        for (declared, actual) in output.iter().zip(actual[0]) {
            // A singleton output declaration supplies an exact fact even when actual types no longer carry its name.
            if let Dimension::Dynamic(variable) = declared
                && let Some(extent) = variable.bounds().extent()
            {
                facts.push(Ok(DimensionRefinementFact::Declared(variable, extent)));
            }
            match (declared, actual) {
                (Dimension::Dynamic(declared), Dimension::Dynamic(actual)) => {
                    facts.push(Ok(DimensionRefinementFact::Correspondence(declared, actual)));
                }
                (Dimension::Static(_), Dimension::Dynamic(_)) => {
                    // Reversing ordinary refinement admits narrowing to a static axis. Its extent belongs to the
                    // supplied source variable, so retain the actual scope when recording that fact.
                    facts.extend(
                        DimensionRefinements::refinement_facts(
                            actual,
                            declared,
                            DimensionComparison::LocalDeclarations,
                        )
                        .map(|fact| {
                            fact.map(|fact| match fact {
                                DimensionRefinementFact::Declared(variable, extent) => {
                                    DimensionRefinementFact::Actual(variable, extent)
                                }
                                _ => unreachable!(),
                            })
                        }),
                    );
                }
                _ => facts.extend(DimensionRefinements::refinement_facts(
                    declared,
                    actual,
                    DimensionComparison::LocalDeclarations,
                )),
            }
        }

        let refinements = DimensionRefinements::from_refinement_facts(facts)?;
        for class in refinements.identity_classes() {
            // Every identity class has at least one member, so the maximal lower bound exists.
            let lower = class
                .declared()
                .iter()
                .chain(class.actual())
                .map(|variable| variable.bounds().lower())
                .max()
                .unwrap();
            if class
                .declared()
                .iter()
                .chain(class.actual())
                .any(|variable| variable.bounds().upper().is_some_and(|upper| upper <= lower))
            {
                return Err(TypeError::invalid(format!(
                    "`{}` cannot equate dimensions with disjoint bounds: {}",
                    CAST_OPERATION_NAME,
                    class
                        .declared()
                        .iter()
                        .chain(class.actual().iter().filter(|variable| !class.declared().contains(variable)))
                        .map(|variable| format!("`{variable}` ∈ {}", variable.bounds()))
                        .collect::<Vec<_>>()
                        .join(", "),
                )));
            }
        }

        // A formal identity may be the same object as an unrelated actual identity. Preserve its name only when its
        // scoped correspondence reaches that actual identity; otherwise resolve its own class below.
        let preserved_identities = preserved
            .iter()
            .filter(|variable| {
                refinements
                    .identity_classes()
                    .any(|class| class.declared().contains(variable) && class.actual().contains(variable))
            })
            .cloned()
            .collect::<Vec<_>>();

        // Full cast inference has validated compatibility. Ordinary refinement of shapes alone now distinguishes
        // relationships already proven by the supplied types from conditional equalities that need a runtime check.
        // Layout and sharding are intentionally absent: changing them adds no runtime shape constraint.
        let proof_declarations = std::iter::once(output)
            .chain(declared.iter().copied())
            .flatten()
            .cloned()
            .map(DimensionType::from)
            .collect::<Vec<_>>();
        let proof_actual = std::iter::once(actual[0])
            .chain(actual.iter().copied())
            .flatten()
            .cloned()
            .map(DimensionType::from)
            .collect::<Vec<_>>();

        // The complete scoped signature proves equality when every occurrence observes an already equal actual
        // dimension. A consistent formal-to-actual renaming does not require an additional runtime check. Ordinary
        // refinement does not admit a conditional relationship at all (e.g., a supplied `n` against a requested static
        // extent), so a failure here means that the relationship needs a runtime check, not that it is contradictory:
        // contradictions were rejected by the scoped facts above.
        let proven = match DimensionTypeRefinements::new(&proof_declarations, &proof_actual) {
            Ok(proof) => proof.identities_needing_runtime_checks().is_empty(),
            Err(_) => false,
        };
        let actual_identities = actual
            .iter()
            .flat_map(|dimensions| dimensions.iter())
            .filter_map(Dimension::variable)
            .cloned()
            .collect::<Vec<_>>();

        // The output observes the first actual input, so a symbolic output dimension must name a supplied identity of
        // the declared output identity's own correspondence class. Another supplied identity that merely shares the
        // declared name (e.g., a supplied `dimension<b>` witnessing a declared `dimension<a>`) belongs to a different
        // class and cannot represent it.
        let dimensions: Vec<_> = output
            .iter()
            .map(|dimension| {
                let refined = refinements.refine_declared_dimension(dimension, &preserved_identities);
                let Dimension::Dynamic(variable) = &refined else {
                    return Ok(refined);
                };
                let corresponding = refinements
                    .identity_classes()
                    .find(|class| class.declared().contains(variable))
                    .map(|class| class.actual())
                    .unwrap_or_default();
                if corresponding.contains(variable) {
                    return Ok(refined);
                }
                let represented = refinements.retype_actual_dimension(dimension, &refined, &actual_identities);
                match &represented {
                    Dimension::Dynamic(variable) if !corresponding.contains(variable) => {
                        Err(TypeError::invalid(format!(
                            "`{CAST_OPERATION_NAME}` output identity `{variable}` has no compatible \
                             supplied input identity",
                        )))
                    }
                    _ => Ok(represented),
                }
            })
            .collect::<Result<_, _>>()?;
        let interpreted = proven.then(|| dimensions.clone());

        // `validate_declaration` rejected reference types, and a dimension output has exactly one dimension.
        let represent = |dimensions: Vec<Dimension>| match &operation.output_type {
            Self::Array(output) => Self::Array(output.clone().with_shape(Shape::new(dimensions))),
            Self::Dimension(_) => Self::Dimension(DimensionType::from(dimensions.into_iter().next().unwrap())),
            Self::Reference(_) | Self::Ragged(_) => unreachable!(),
        };

        Ok((represent(dimensions), interpreted.map(represent)))
    }

    fn identity_witness_operation(
        value_type: &Self,
        identity: &DimensionVariable,
    ) -> Result<Option<DimensionSizeOperation>, TypeError> {
        // An array reads the extent of the first axis that carries the identity as a `dimension` scalar, whose type
        // carries that identity. A first-class dimension value already is such a scalar, and references are never
        // cast, so both keep the whole value.
        let Self::Array(array_type) = value_type else {
            return Ok(None);
        };
        array_type
            .shape()
            .dimensions()
            .iter()
            .position(|dimension| dimension.variable() == Some(identity))
            .map(|axis| DimensionSizeOperation::new(array_type, axis))
            .transpose()
    }
}

/// Represents the ability to return a value at a different but related type representation, through a [`CastOperation`]
/// (e.g., an `f32[n]` value whose runtime extent is 2 as `f32[2]`, or an `f32[2]` value as `f32[n]` next to a witness
/// of type `f32[n]`). The value at the output type and the witnesses at their own types must satisfy every relationship
/// that their shared identities express (refer to [`CastOperation`]). Concrete values have exact types, so a cast
/// checks shapes immediately and materializes the requested representation. Concrete witnesses carry no identities:
/// they cannot supply a nominal output name, so requesting an output name absent from the declared inputs is rejected
/// in both direct execution and staging. Host arrays retain exact shapes and use dense row-major storage when the
/// layout is unspecified. Context-carrying values apply the operation through their context, where symbolic witness
/// types relate the output's identities to other values. An [`ArrayIrValue`] delegates array casts to its array member,
/// including when that member is a tracer; its concrete dimension witnesses have exact types and introduce no shared
/// identities. Host dimension results reject array witnesses that their member context cannot resolve as constants; use
/// a composite context to retain staged dependencies or observe backend completion. [`cast`](Self::cast) checks the
/// relationship in staged and compiled programs too, whereas [`cast_unchecked`](Self::cast_unchecked) assumes it there.
///
/// Placement over explicit mesh axes may change under the rules of [`Reshard`](crate::Reshard), whereas placement
/// over auto and manual mesh axes is preserved (refer to [`CastOperation`]). The XLA backend requires finite dimension
/// bounds when changing physical allocation bounds, and constrains placement only when the explicit placement changes.
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
/// // Concrete inputs cannot supply the nominal name `n`; exact shapes still support representation changes.
/// assert!(value.cast(symbolic, &[]).is_err());
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
    /// Returns a [`ProgramError`] if the types are unrelated, the static types or the concrete values contradict
    /// the relationship, or the cast cannot be applied in this value's context. A staged cast reports a violated
    /// relationship when the program executes.
    fn cast(&self, output_type: T, witnesses: &[Self]) -> Result<Self, ProgramError>;

    /// Returns this value at `output_type` like [`Self::cast`], but through an unchecked [`CastOperation`]. This is
    /// because staged and compiled programs may assume the relationship instead of checking it.
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

impl<
    T: CastType,
    V: Value<Type = T, Dispatch = ValueDomainDispatch, Domain: Context<Operation: From<CastOperation<T>>>>,
> Cast<T> for V
{
    #[inline]
    fn cast(&self, output_type: T, witnesses: &[Self]) -> Result<Self, ProgramError> {
        CastOperation::bind_checked(&self.domain(), self, witnesses, output_type)
    }

    #[inline]
    fn cast_unchecked(&self, output_type: T, witnesses: &[Self]) -> Result<Self, ProgramError> {
        CastOperation::bind_unchecked(&self.domain(), self, witnesses, output_type)
    }
}

impl Cast for Array {
    #[inline]
    fn cast(&self, output_type: ArrayType, witnesses: &[Self]) -> Result<Self, ProgramError> {
        cast_array(self, output_type, witnesses, true)
    }

    #[inline]
    fn cast_unchecked(&self, output_type: ArrayType, witnesses: &[Self]) -> Result<Self, ProgramError> {
        cast_array(self, output_type, witnesses, false)
    }
}

impl<A: Value<Type = ArrayType, Domain: Context> + Cast<ArrayType>> Cast for ArrayIrValue<A> {
    #[inline]
    fn cast(&self, output_type: ArrayIrType, witnesses: &[Self]) -> Result<Self, ProgramError> {
        cast_array_ir_value(self, output_type, witnesses, true)
    }

    #[inline]
    fn cast_unchecked(&self, output_type: ArrayIrType, witnesses: &[Self]) -> Result<Self, ProgramError> {
        cast_array_ir_value(self, output_type, witnesses, false)
    }
}

/// Casts the concrete array `value` to `output_type`, validating its complete relationship with `witnesses`.
fn cast_array(
    value: &Array,
    output_type: ArrayType,
    witnesses: &[Array],
    checked: bool,
) -> Result<Array, ProgramError> {
    let inputs = std::iter::once(value).chain(witnesses);
    let input_types = inputs.map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
    let operation = CastOperation { input_types: input_types.clone(), output_type, checked };
    let output_type = operation.infer_interpreted_output_type(&input_types, &[])?;
    let input_type = value.r#type();

    // Materialized arrays retain exact shapes after declared names have been resolved to their known extents.
    // All other representation fields follow the request, including dense default storage for an unspecified layout.
    let representation = output_type.with_shape(input_type.shape().clone());
    if representation == *input_type {
        Ok(value.clone())
    } else if representation.layout() == input_type.layout() {
        // Sharding annotations do not change host-array storage.
        Ok(Array::new_unchecked(representation, value.shared_storage_bytes().clone()))
    } else {
        // Layout determines physical addressing. Re-encode logical bytes to preserve exact element bits,
        // including low-precision encodings and NaN payloads, rather than relabeling the original physical buffer.
        Array::from_logical_bytes(representation, &value.logical_bytes())
    }
}

/// Casts the provided [`ArrayIrValue`] through its execution context, or validates a concrete dimension cast. Rejects
/// reference witnesses before omitting concrete dimension witnesses, whose exact types carry no identities. Array
/// members may themselves be tracers, so their symbolic relationships must not be checked as if they were concrete
/// values.
fn cast_array_ir_value<A: Value<Type = ArrayType, Domain: Context> + Cast<ArrayType>>(
    value: &ArrayIrValue<A>,
    output_type: ArrayIrType,
    witnesses: &[ArrayIrValue<A>],
    checked: bool,
) -> Result<ArrayIrValue<A>, ProgramError> {
    if let (ArrayIrValue::Array(value), ArrayIrType::Array(output_type)) = (value, &output_type) {
        let witnesses = witnesses
            .iter()
            .filter_map(|witness| match witness {
                ArrayIrValue::Array(value) => Some(Ok(value.clone())),
                ArrayIrValue::Dimension(_) => None,
                ArrayIrValue::Ragged(_) => Some(Err(TypeError::invalid(format!(
                    "`{}` does not support first-class ragged type `{}`",
                    CAST_OPERATION_NAME,
                    witness.r#type(),
                )))),
                ArrayIrValue::Reference(_) => Some(Err(TypeError::invalid(format!(
                    "`{}` does not support reference type `{}`",
                    CAST_OPERATION_NAME,
                    witness.r#type(),
                )))),
            })
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(ArrayIrValue::Array(if checked {
            value.cast(output_type.clone(), &witnesses)?
        } else {
            value.cast_unchecked(output_type.clone(), &witnesses)?
        }));
    }

    let inputs = std::iter::once(value).chain(witnesses);
    let input_types = inputs.map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
    let operation = CastOperation { input_types: input_types.clone(), output_type, checked };
    operation.infer_interpreted_output_type(&input_types, &[])?;
    for witness in witnesses {
        if let ArrayIrValue::Array(witness) = witness
            && !witness
                .domain()
                .resolve(witness)
                .into_constant()
                .is_some_and(|constant| constant.capture_index().is_none())
        {
            return Err(ProgramError::Concretization {
                message: format!(
                    "host dimension `{CAST_OPERATION_NAME}` cannot retain an array witness producer; \
                     bind the cast through a composite context",
                ),
            });
        }
    }

    Ok(value.clone())
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::sync::Arc;

    use indoc::{formatdoc, indoc};
    use num_complex::Complex as ComplexNumber;
    use pretty_assertions::assert_eq;

    use crate::arrays::batching::tests::{ragged_array_batch, ragged_batch};
    use crate::arrays::{
        Array, ArrayBatch, ArrayElement, ArrayIrBatch, ArrayIrOperation, ArrayIrValue, ArrayOperation, ArrayReference,
        ArrayType, DataType, Dimension, DimensionBounds, DimensionError, DimensionType, DimensionValue,
        DimensionVariable, Layout, LogicalMesh, MAX_DIMENSION_EXTENT, Memory, MeshAxis, MeshAxisType, RaggedAxis,
        RaggedType, RaggedValueKind, Shape, Sharding, ShardingDimension, StridedLayout, Tile, TileDimension,
        TiledLayout, bf16, f8e4m3fn, f8e5m2, f8e8m0fnu, i4, u2, u4,
    };
    use crate::batching::{BatchAxis, BatchedProgram, BatchingTracer, ProgramBatchingOutputAxesPolicy};
    use crate::captures::CaptureReference;
    use crate::contexts::{EagerContext, StagingContext};
    use crate::differentiation::{
        DifferentiableOperation, DifferentiationContext, DifferentiationError, TransposableOperation,
        TranspositionContext, differentiate_at,
    };
    use crate::macros::{
        check_gradient, check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_transposition, check_operation_type_inference, dispatch_on_array_element_type,
    };
    use crate::operations::assertions::{AssertOperation, AssertionError, GuardOperation};
    use crate::operations::control_flow::condition::ConditionOperation;
    use crate::operations::dimensions::dimension_from_scalar::DimensionFromScalarOperation;
    use crate::parameters::Placeholder;
    use crate::partial::PartialValue;
    use crate::programs::{EffectClasses, Effects, EmptyRegionDriver, ProgramBuilder, ReferenceType, Typed};
    use crate::tests::ProjectedMemberType;

    use super::*;

    // An independent type family can implement the operation-specific contract without changing `TypeRefinements`.
    // The impl lives at module level because trait impls are never scoped to a function.
    impl CastType for ProjectedMemberType<0> {
        // This fixture has no identities and keeps the default witness selector, which returns `None`. The unused
        // `CastOperation<Self>` placeholder reuses the conversion that cast differentiation already requires.
        type IdentityWitnessOperation = CastOperation<Self>;

        fn infer_cast_output_type(
            operation: &CastOperation<Self>,
            input_types: &[Self],
        ) -> Result<(Self, Option<Self>), TypeError> {
            operation.validate_declaration(input_types)?;
            Ok((operation.output_type.clone(), Some(operation.output_type.clone())))
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
        let input = ragged_array_batch(array.clone(), extents.clone(), &variable, 1).unwrap();
        let ragged_axes = input.ragged_axes().to_vec();
        let (outputs, _) = ConvertElementTypeOperation::<ArrayType>::new(DataType::U32, true)
            .batch(&context, &EmptyRegionDriver, &[input])
            .unwrap()
            .into_parts();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].r#type().as_ref(), &ArrayType::new_static(DataType::U32, [2, 3]));
        assert_eq!(outputs[0].ragged_axes(), ragged_axes.as_slice());
        let input = ragged_array_batch(array, extents, &variable, 2).unwrap();
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
            MaybeZero::Value(Array::scalar(3.0f32).unwrap()),
        );

        // Pulling back through complex-to-real conversion inserts a zero imaginary cotangent.
        let (primal, pullback) = differentiate_at(Array::scalar(ComplexNumber::new(2.0f32, 5.0)).unwrap())
            .vjp(|value| value.convert_element_type(DataType::F32))
            .unwrap();
        assert_eq!(primal, Array::scalar(2.0f32).unwrap());
        assert_eq!(
            pullback.apply(Array::scalar(3.0f32).unwrap()).unwrap(),
            MaybeZero::Value(Array::scalar(ComplexNumber::new(3.0f32, 0.0)).unwrap()),
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

        // Tiled layouts count elements, so widening preserves their tile shape and logical element order.
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
        // Host arrays keep their type while rounding every element.
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

    #[test]
    fn test_cast() {
        // Both modes require a value to cast, independently of the type family's inference implementation.
        assert_eq!(
            CastOperation::checked(DataType::F32, Vec::new()),
            Err(TypeError::invalid("`cast` requires a declared first input type")),
        );
        assert_eq!(
            CastOperation::unchecked(DataType::F32, Vec::new()),
            Err(TypeError::invalid("`cast` requires a declared first input type")),
        );

        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let symbolic = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()]));
        let operation = CastOperation::unchecked(symbolic.clone(), vec![symbolic.clone(), symbolic.clone()]).unwrap();
        assert_eq!(operation.name(), CAST_OPERATION_NAME);
        assert_eq!(operation.input_types(), &[symbolic.clone(), symbolic.clone()]);
        assert_eq!(operation.output_type(), &symbolic);
        assert!(!operation.is_checked());
        assert_eq!(operation.effects().classes(), Effects::empty().classes());
        assert_eq!(operation.to_string(), "cast [type=f32[n], inputs=[f32[n], f32[n]]]");
        assert_eq!(
            CastOperation::unchecked(symbolic.clone(), vec![symbolic.clone()]).unwrap().to_string(),
            "cast [type=f32[n], inputs=[f32[n]]]",
        );

        // A checked cast declares the ordered-assertion effect and renders its mode.
        let checked = CastOperation::checked(symbolic.clone(), vec![symbolic.clone(), symbolic.clone()]).unwrap();
        assert!(checked.is_checked());
        assert_eq!(checked.effects().classes(), EffectClasses::single(EffectClass::OrderedAssertion));
        assert_eq!(checked.to_string(), "cast [type=f32[n], inputs=[f32[n], f32[n]], checked=true]");
        assert_eq!(
            format!("{:?}", CastOperation::checked(DataType::F32, vec![DataType::F32]).unwrap()),
            "CastOperation { input_types: [F32], output_type: F32, checked: true }",
        );

        // Equality and hashing follow the declarations and the mode.
        assert_eq!(
            operation,
            CastOperation::unchecked(symbolic.clone(), vec![symbolic.clone(), symbolic.clone()]).unwrap()
        );
        assert_ne!(operation, checked);
        assert_ne!(operation, CastOperation::unchecked(symbolic.clone(), vec![symbolic.clone()]).unwrap());
        let operations = HashSet::from([operation.clone(), checked.clone()]);
        assert!(operations.contains(&operation));
        assert!(operations.contains(&checked));
        assert!(!operations.contains(&CastOperation::unchecked(symbolic.clone(), vec![symbolic]).unwrap()));
    }

    #[test]
    fn test_cast_normalized() {
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let symbolic = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()]));
        let exact = ArrayType::new_static(DataType::F32, [2]);
        let laid_out = exact.clone().with_layout(Layout::Strided(StridedLayout::new(vec![8])));

        // Actual types that prove the relationship turn a checked cast into an unchecked one, an unproven narrowing
        // keeps its check, and an unchecked cast is returned unchanged.
        assert_eq!(
            CastOperation::checked(laid_out.clone(), vec![exact.clone()]).unwrap().normalized(&[exact.clone()]),
            CastOperation::unchecked(laid_out, vec![exact.clone()]).unwrap(),
        );
        assert_eq!(
            CastOperation::checked(exact.clone(), vec![symbolic.clone()]).unwrap().normalized(&[exact.clone()]),
            CastOperation::unchecked(exact.clone(), vec![symbolic.clone()]).unwrap(),
        );
        assert_eq!(
            CastOperation::checked(exact.clone(), vec![symbolic.clone()])
                .unwrap()
                .normalized(&[symbolic.clone()]),
            CastOperation::checked(exact.clone(), vec![symbolic.clone()]).unwrap(),
        );
        assert_eq!(
            CastOperation::unchecked(exact.clone(), vec![symbolic.clone()])
                .unwrap()
                .normalized(&[symbolic.clone()]),
            CastOperation::unchecked(exact.clone(), vec![symbolic.clone()]).unwrap(),
        );

        // The proof uses the actual types: a declaration admitting weaker actual types still needs its check.
        let widening = CastOperation::checked(symbolic.clone(), vec![symbolic.clone(), symbolic.clone()]).unwrap();
        assert_eq!(widening.clone().normalized(&[exact.clone(), symbolic]), widening);

        // A cast whose declaration the actual types do not admit is returned unchanged for binding to reject.
        let unrelated = CastOperation::checked(exact.clone(), vec![exact]).unwrap();
        assert_eq!(unrelated.clone().normalized(&[ArrayType::new_static(DataType::F32, [3])]), unrelated);
    }

    #[test]
    fn test_cast_validate_declaration() {
        let bounds = DimensionBounds::positive(Some(8)).unwrap();
        let [n, m] = ["n", "m"].map(|name| DimensionType::new(name, bounds));
        let symbolic: ArrayIrType = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()])).into();
        let exact: ArrayIrType = ArrayType::new_static(DataType::F32, [2]).into();
        let static_dimension: ArrayIrType = DimensionType::from(Dimension::Static(2)).into();

        // The returned identities are the output identities that live inputs still carry.
        let widening = CastOperation::checked(symbolic.clone(), vec![exact.clone(), n.clone().into()]).unwrap();
        assert_eq!(widening.validate_declaration(&[exact.clone(), n.clone().into()]), Ok(vec![n.to_variable()]));
        assert_eq!(widening.validate_declaration(&[exact.clone(), static_dimension]), Ok(Vec::new()));

        // Each family-independent requirement is reported in turn.
        assert_eq!(
            widening.validate_declaration(&[exact.clone()]),
            Err(TypeError::invalid("expected 2 inputs but got 1")),
        );
        let reference: ArrayIrType = ReferenceType::new(ArrayType::new_static(DataType::F32, [2])).into();
        assert_eq!(
            CastOperation::checked(exact.clone(), vec![exact.clone(), reference.clone()])
                .unwrap()
                .validate_declaration(&[exact.clone(), reference]),
            Err(TypeError::invalid("`cast` does not support reference type `ref<f32[2]>`")),
        );
        assert_eq!(
            CastOperation::checked(n.clone().into(), vec![exact.clone()])
                .unwrap()
                .validate_declaration(&[exact.clone()]),
            Err(TypeError::invalid("`cast` output type `dimension<n ∈ [1, 8)>` cannot define a type identity")),
        );
        let wrong: ArrayIrType = ArrayType::new_static(DataType::F32, [3]).into();
        assert_eq!(
            CastOperation::checked(exact.clone(), vec![exact.clone()]).unwrap().validate_declaration(&[wrong]),
            Err(TypeError::invalid("dimension 3 does not refine declared dimension 2")),
        );
        assert_eq!(
            CastOperation::checked(symbolic, vec![exact.clone(), m.clone().into()])
                .unwrap()
                .validate_declaration(&[exact, m.into()]),
            Err(TypeError::invalid("`cast` output identity `n` is not carried by a declared input")),
        );
    }

    #[test]
    fn test_cast_bind_checked() {
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let symbolic = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()]));
        let exact = ArrayType::new_static(DataType::F32, [2]);

        // Proven identities fold, proven witnessed relationships become pure, and conditional narrowing retains its
        // check. A witnessed identity still records both inputs so that their producers remain dependencies.
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |inputs: Vec<Tracer<_>>| {
                let context = inputs[0].context();
                let identity = CastOperation::bind_checked(context, &inputs[1], &[], exact.clone())?;
                assert_eq!(identity, inputs[1]);
                let witnessed = CastOperation::bind_checked(context, &inputs[1], &inputs[1..], exact.clone())?;
                let narrowed = CastOperation::bind_checked(context, &inputs[0], &[], exact.clone())?;
                Ok(vec![witnessed, narrowed])
            },
            vec![symbolic, exact.clone()],
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[n], %1:f32[2] .
                let %2:f32[2] = cast [type=f32[2], inputs=[f32[2], f32[2]]] %1 %1
                    %3:f32[2] = cast [type=f32[2], inputs=[f32[n]], checked=true] %0
                in (%2, %3)"},
        );
        let values = vec![Array::vector(vec![1f32, 2f32]).unwrap(), Array::vector(vec![3f32, 4f32]).unwrap()];
        assert_eq!(program.interpret(values.clone()), Ok(vec![values[1].clone(), values[0].clone()]));
    }

    #[test]
    fn test_cast_bind_unchecked() {
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let symbolic = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()]));
        let exact = ArrayType::new_static(DataType::F32, [2]);

        // A value that already has the output type is returned as is when there are no witnesses, whereas witnesses
        // stay explicit dependencies even for an identity conversion, and a representation change binds one unchecked
        // cast.
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |inputs: Vec<Tracer<_>>| {
                let context = inputs[0].context();
                let identity = CastOperation::bind_unchecked(context, &inputs[0], &[], exact.clone())?;
                assert_eq!(identity, inputs[0]);
                let witnessed = CastOperation::bind_unchecked(context, &inputs[0], &inputs[1..], exact.clone())?;
                let widened = CastOperation::bind_unchecked(context, &inputs[0], &inputs[1..], symbolic.clone())?;
                Ok(vec![witnessed, widened])
            },
            vec![exact.clone(), symbolic.clone()],
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2], %1:f32[n] .
                let %2:f32[2] = cast [type=f32[2], inputs=[f32[2], f32[n]]] %0 %1
                    %3:f32[n] = cast [type=f32[n], inputs=[f32[2], f32[n]]] %0 %1
                in (%2, %3)"},
        );
    }

    #[test]
    fn test_cast_bind_identity_validates_declaration() {
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        for checked in [true, false] {
            let result = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
                |input: Tracer<_>| {
                    let output_type = input.r#type().into_owned();
                    if checked {
                        CastOperation::bind_checked(input.context(), &input, &[], output_type)
                    } else {
                        CastOperation::bind_unchecked(input.context(), &input, &[], output_type)
                    }
                },
                ArrayIrType::Dimension(n.clone()),
            );
            assert_eq!(
                result.map(|_| ()),
                Err(ProgramError::Type(TypeError::invalid(
                    "`cast` output type `dimension<n ∈ [1, 8)>` cannot define a type identity",
                ))),
            );
        }
    }

    #[test]
    fn test_cast_infer_interpreted_output_type() {
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let symbolic = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()]));
        let exact = ArrayType::new_static(DataType::F32, [2]);
        let laid_out = exact.clone().with_layout(Layout::Strided(StridedLayout::new(vec![8])));

        // Concrete extents prove the relationship and the inferred representation is returned, in both modes.
        let narrowing = CastOperation::checked(exact.clone(), vec![symbolic.clone()]).unwrap();
        assert_eq!(narrowing.infer_interpreted_output_type(&[exact.clone()], &[]), Ok(exact.clone()));
        assert_eq!(
            CastOperation::unchecked(laid_out.clone(), vec![exact.clone()])
                .unwrap()
                .infer_interpreted_output_type(&[exact.clone()], &[]),
            Ok(laid_out),
        );

        // A relationship that still needs a runtime check has no interpretation, even for an unchecked cast.
        let unproven = Err(ProgramError::Concretization {
            message: "`cast` cannot check symbolic type relationships during interpretation; \
                      bind the operation through its context"
                .to_owned(),
        });
        assert_eq!(narrowing.infer_interpreted_output_type(&[symbolic.clone()], &[]), unproven);
        assert_eq!(
            CastOperation::unchecked(exact.clone(), vec![symbolic.clone()])
                .unwrap()
                .infer_interpreted_output_type(&[symbolic], &[]),
            unproven,
        );

        // Structural and type errors are reported before the proof.
        assert_eq!(
            narrowing.infer_interpreted_output_type(&[], &[]),
            Err(ProgramError::Type(TypeError::invalid("expected 1 input but got 0"))),
        );
        assert_eq!(
            narrowing.infer_interpreted_output_type(
                &[exact.clone()],
                &[RegionInterface::new(vec![], vec![], EffectClasses::NONE)],
            ),
            Err(ProgramError::Type(TypeError::invalid("expected 0 regions but got 1"))),
        );
        assert_eq!(
            CastOperation::checked(exact.clone(), vec![exact])
                .unwrap()
                .infer_interpreted_output_type(&[ArrayType::new_static(DataType::F32, [3])], &[]),
            Err(ProgramError::Type(TypeError::invalid("dimension 3 does not refine declared dimension 2"))),
        );
    }

    #[test]
    fn test_cast_rename_type_identities() {
        let bounds = DimensionBounds::positive(Some(8)).unwrap();
        let [n, m, a, b] = ["n", "m", "a", "b"].map(|name| DimensionType::new(name, bounds));
        let symbolic = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()]));
        let exact = ArrayType::new_static(DataType::F32, [2]);

        // The declarations name identities of the surrounding program, so they follow its renamings in both modes,
        // and renaming back restores the original operation.
        let operation = CastOperation::checked(symbolic.clone(), vec![exact.clone(), symbolic.clone()]).unwrap();
        let mut renaming = TypeIdentityRenaming::new();
        renaming.insert(n.to_variable(), m.to_variable()).unwrap();
        let renamed = operation.rename_type_identities(&renaming).unwrap();
        assert_eq!(renamed.to_string(), "cast [type=f32[m], inputs=[f32[2], f32[m]], checked=true]");
        let mut inverse = TypeIdentityRenaming::new();
        inverse.insert(m.to_variable(), n.to_variable()).unwrap();
        assert_eq!(renamed.rename_type_identities(&inverse), Ok(operation));

        // A swap is simultaneous.
        let array = |dimensions: Vec<Dimension>| ArrayType::new(DataType::F32, Shape::new(dimensions));
        let mixed = CastOperation::unchecked(
            array(vec![Dimension::Static(3), a.simplified_dimension()]),
            vec![array(vec![b.simplified_dimension(), Dimension::Static(3)]), array(vec![a.simplified_dimension()])],
        )
        .unwrap();
        let mut swap = TypeIdentityRenaming::new();
        swap.insert(a.to_variable(), b.to_variable()).unwrap();
        swap.insert(b.to_variable(), a.to_variable()).unwrap();
        assert_eq!(
            mixed.rename_type_identities(&swap).unwrap().to_string(),
            "cast [type=f32[3, b], inputs=[f32[a, 3], f32[b]]]",
        );

        // A first-class dimension declaration cannot be renamed to a variable with wider bounds.
        let wide = DimensionType::new("wide", DimensionBounds::positive(Some(16)).unwrap());
        let mut widening = TypeIdentityRenaming::new();
        widening.insert(n.to_variable(), wide.to_variable()).unwrap();
        assert_eq!(
            CastOperation::<ArrayIrType>::checked(exact.clone().into(), vec![exact.into(), n.into()])
                .unwrap()
                .rename_type_identities(&widening),
            Err(TypeError::invalid(
                "cannot rename dimension variable n with bounds [1, 8) to wide with bounds [1, 16)",
            )),
        );
    }

    #[test]
    fn test_cast_fold() {
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let symbolic = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()]));
        let exact = ArrayType::new_static(DataType::F32, [2]);
        let laid_out = exact.clone().with_layout(Layout::Strided(StridedLayout::new(vec![8])));

        // A proven single-input identity folds to its input, whereas a representation change, an unproven
        // relationship, and a witnessed cast do not: witness producers stay explicit dependencies even when the
        // representation is unchanged.
        let folded = Ok(Some(vec![OperationFoldOutput::Input(0)]));
        assert_eq!(
            CastOperation::checked(exact.clone(), vec![symbolic.clone()]).unwrap().fold(&[exact.clone()], &[]),
            folded
        );
        assert_eq!(
            CastOperation::unchecked(exact.clone(), vec![exact.clone()]).unwrap().fold(&[exact.clone()], &[]),
            folded
        );
        assert_eq!(
            CastOperation::checked(laid_out, vec![exact.clone()]).unwrap().fold(&[exact.clone()], &[]),
            Ok(None)
        );
        assert_eq!(
            CastOperation::checked(exact.clone(), vec![symbolic.clone()])
                .unwrap()
                .fold(&[symbolic.clone()], &[]),
            Ok(None),
        );
        assert_eq!(
            CastOperation::checked(exact.clone(), vec![exact.clone(), symbolic.clone()])
                .unwrap()
                .fold(&[exact.clone(), symbolic], &[]),
            Ok(None),
        );

        // Structural and type errors are reported.
        assert_eq!(
            CastOperation::checked(exact.clone(), vec![exact.clone()])
                .unwrap()
                .fold(&[exact.clone()], &[RegionInterface::new(vec![], vec![], EffectClasses::NONE)]),
            Err(TypeError::invalid("expected 0 regions but got 1")),
        );
        assert_eq!(
            CastOperation::checked(exact.clone(), vec![exact])
                .unwrap()
                .fold(&[ArrayType::new_static(DataType::F32, [3])], &[]),
            Err(TypeError::invalid("dimension 3 does not refine declared dimension 2")),
        );
    }

    #[test]
    fn test_cast_fold_preserves_witness_dependencies() {
        // Interpreting a cast through a staging context binds it, so a proven identity folds to its input only when
        // it has no witnesses: a witnessed identity keeps its checked witness producer in the staged program.
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let symbolic = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()]));
        let exact = ArrayType::new_static(DataType::F32, [2]);
        let vector = |extent| Array::vector(vec![1f32; extent]).unwrap();
        for constructor in [CastOperation::checked, CastOperation::unchecked] {
            let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
                |inputs: Vec<Tracer<_>>| {
                    let witness = inputs[1].cast(symbolic.clone(), &[inputs[0].clone()])?;
                    constructor(symbolic.clone(), vec![symbolic.clone(); 2]).unwrap().interpret(
                        inputs[0].context(),
                        &EmptyRegionDriver,
                        &[inputs[0].clone(), witness],
                    )
                },
                vec![symbolic.clone(), exact.clone()],
            )
            .unwrap();
            assert_eq!(
                program.to_string(),
                indoc! {"
                    lambda %0:f32[n], %1:f32[2] .
                    let %2:f32[n] = cast [type=f32[n], inputs=[f32[2], f32[n]], checked=true] %1 %0
                        %3:f32[n] = cast [type=f32[n], inputs=[f32[n], f32[n]]] %0 %2
                    in (%3)"},
            );
            assert_eq!(program.interpret(vec![vector(2), vector(2)]), Ok(vec![vector(2)]));
            assert_eq!(
                program.interpret(vec![vector(3), vector(2)]),
                Err(ProgramError::Type(TypeError::custom(DimensionError::InputDimensionMismatch {
                    dimension: "n".to_owned(),
                    expected: 3,
                    actual: 2,
                }))),
            );
        }
    }

    #[test]
    fn test_cast_type_inference() {
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let symbolic: ArrayIrType = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()])).into();
        let exact = |extent: usize| ArrayIrType::from(ArrayType::new_static(DataType::F32, [extent]));
        let static_dimension = |extent: usize| ArrayIrType::from(DimensionType::from(Dimension::Static(extent)));

        // Narrowing and widening are both admitted, for arrays and for first-class dimensions, and witnesses scope the
        // identities that a widened output type mentions.
        let narrowing = CastOperation::unchecked(exact(2), vec![symbolic.clone(), symbolic.clone()]).unwrap();
        check_operation_type_inference!(
            operation = narrowing.clone(),
            cases = [{ input_types = [symbolic.clone(), symbolic.clone()], output_types = [exact(2)] }],
        );
        let widening = CastOperation::unchecked(symbolic.clone(), vec![exact(2), symbolic.clone()]).unwrap();
        check_operation_type_inference!(
            operation = widening.clone(),
            cases = [
                { input_types = [exact(2), symbolic.clone()], output_types = [symbolic.clone()] },
                { input_types = [exact(2)], error = "expected 2 inputs but got 1" },
            ],
        );
        check_operation_type_inference!(
            operation = CastOperation::unchecked(static_dimension(2), vec![n.clone().into()]).unwrap(),
            cases = [{ input_types = [n.clone().into()], output_types = [static_dimension(2)] }],
        );

        // A relationship that the static types already contradict is rejected in both directions.
        let mismatch = Err(TypeError::custom(DimensionError::InputDimensionMismatch {
            dimension: "n".to_owned(),
            expected: 5,
            actual: 2,
        }));
        assert_eq!(widening.infer_output_types(&[exact(2), exact(5)], &[]), mismatch);
        let widening_with_dimension =
            CastOperation::unchecked(symbolic.clone(), vec![exact(2), n.clone().into()]).unwrap();
        assert_eq!(widening_with_dimension.infer_output_types(&[exact(2), static_dimension(5)], &[]), mismatch);
        assert_eq!(narrowing.infer_output_types(&[symbolic.clone(), exact(5)], &[]), mismatch);

        // Unrelated types, identity-defining output types, and references are rejected.
        check_operation_type_inference!(
            operation = CastOperation::unchecked(exact(2), vec![exact(3)]).unwrap(),
            cases = [{
                input_types = [exact(3)],
                error = "`cast` cannot convert type `f32[3]` to unrelated type `f32[2]`",
            }],
        );
        check_operation_type_inference!(
            operation = CastOperation::unchecked(n.clone().into(), vec![static_dimension(2)]).unwrap(),
            cases = [{
                input_types = [static_dimension(2)],
                error = "`cast` output type `dimension<n ∈ [1, 8)>` cannot define a type identity",
            }],
        );
        let reference: ArrayIrType = ReferenceType::new(ArrayType::new_static(DataType::F32, [2])).into();
        check_operation_type_inference!(
            operation = CastOperation::unchecked(reference.clone(), vec![reference.clone()]).unwrap(),
            cases = [{
                input_types = [reference],
                error = "`cast` does not support reference type `ref<f32[2]>`",
            }],
        );

        // The same operation infers bare element types in the `DataType` universe, where only identities are related.
        check_operation_type_inference!(
            operation = CastOperation::checked(DataType::F32, vec![DataType::F32]).unwrap(),
            cases = [
                { input_types = [DataType::F32], output_types = [DataType::F32] },
                { input_types = [DataType::F64], error = "type f64 does not refine declared type f32" },
            ],
        );
        check_operation_type_inference!(
            operation = CastOperation::checked(DataType::F64, vec![DataType::F32]).unwrap(),
            cases = [{
                input_types = [DataType::F32],
                error = "`cast` cannot convert type `f32` to unrelated type `f64`",
            }],
        );

        // The operation cannot own nested regions.
        assert_eq!(
            narrowing.infer_output_types(
                &[symbolic.clone(), symbolic],
                &[RegionInterface::new(vec![], vec![], EffectClasses::NONE)],
            ),
            Err(TypeError::invalid("expected 0 regions but got 1")),
        );
    }

    #[test]
    fn test_cast_type_inference_mixed_dimensions() {
        // Narrowing some axes while widening others uses the per-axis rule: the source fixes `b` and its static axis,
        // the witness must agree with that static axis through `a`, and a static output contradicting the declared
        // source is rejected.
        let bounds = DimensionBounds::positive(Some(8)).unwrap();
        let [a, b] = ["a", "b"].map(|name| DimensionType::new(name, bounds));
        let array = |dimensions: Vec<Dimension>| -> ArrayIrType {
            ArrayType::new(DataType::F32, Shape::new(dimensions)).into()
        };
        let input = array(vec![b.simplified_dimension(), Dimension::Static(3)]);
        let output = array(vec![Dimension::Static(3), a.simplified_dimension()]);
        let witness = array(vec![a.simplified_dimension()]);
        let operation = CastOperation::unchecked(output.clone(), vec![input.clone(), witness.clone()]).unwrap();
        check_operation_type_inference!(
            operation = operation.clone(),
            cases = [{ input_types = [input.clone(), witness.clone()], output_types = [output] }],
        );
        assert_eq!(
            operation.infer_output_types(&[input.clone(), array(vec![Dimension::Static(5)])], &[]),
            Err(TypeError::custom(DimensionError::InputDimensionMismatch {
                dimension: "a".to_owned(),
                expected: 5,
                actual: 3,
            })),
        );
        check_operation_type_inference!(
            operation = CastOperation::unchecked(
                array(vec![Dimension::Static(4), Dimension::Static(4)]),
                vec![input.clone(), witness.clone()],
            ).unwrap(),
            cases = [{
                input_types = [input, witness],
                error = "`cast` cannot convert type `f32[b, 3]` to unrelated type `f32[4, 4]`",
            }],
        );
    }

    #[test]
    fn test_cast_type_inference_repeated_identity() {
        // An identity repeated within one type relates every occurrence to its corresponding output axis: narrowing
        // `f32[n, n]` to `f32[n, 2]` requires `n = 2`, which symbolic inputs leave to a runtime check, exact inputs
        // prove, and contradicting exact inputs reject.
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let square =
            ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension(), n.simplified_dimension()]));
        let output = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension(), Dimension::Static(2)]));
        let exact = ArrayType::new_static(DataType::F32, [2, 2]);
        for constructor in [CastOperation::checked, CastOperation::unchecked] {
            let operation = constructor(output.clone(), vec![square.clone()]).unwrap();
            assert_eq!(
                ArrayType::infer_cast_output_type(&operation, std::slice::from_ref(&square)),
                Ok((output.clone(), None)),
            );
            assert_eq!(
                ArrayType::infer_cast_output_type(&operation, std::slice::from_ref(&exact)),
                Ok((exact.clone(), Some(exact.clone()))),
            );
            assert_eq!(
                operation.infer_output_types(&[ArrayType::new_static(DataType::F32, [2, 3])], &[]),
                Err(TypeError::custom(DimensionError::InputDimensionMismatch {
                    dimension: "n".to_owned(),
                    expected: 2,
                    actual: 3,
                })),
            );
        }
    }

    #[test]
    fn test_cast_type_inference_layout_and_sharding() {
        // Changing the layout or the explicit placement preserves logical values, so each kind of transition (adding,
        // changing, and dropping either annotation, or both at once) infers the requested output, in both modes and
        // in both universes.
        let plain = ArrayType::new_static(DataType::F32, [2]);
        let narrow = plain.clone().with_layout(Layout::Strided(StridedLayout::new(vec![4])));
        let wide = plain.clone().with_layout(Layout::Strided(StridedLayout::new(vec![8])));
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated = Sharding::new(mesh, vec![ShardingDimension::replicated()]).unwrap();
        let placed = plain.clone().with_sharding(sharded.clone()).unwrap();
        let unplaced = plain.clone().with_sharding(replicated.clone()).unwrap();
        let narrow_placed = narrow.clone().with_sharding(sharded).unwrap();
        let wide_unplaced = wide.clone().with_sharding(replicated).unwrap();
        for (input, output) in [
            (plain.clone(), plain.clone()),
            (plain.clone(), narrow.clone()),
            (narrow.clone(), wide.clone()),
            (wide, plain.clone()),
            (plain.clone(), placed.clone()),
            (placed.clone(), unplaced.clone()),
            (unplaced, plain),
            (narrow_placed.clone(), narrow_placed.clone()),
            (narrow_placed.clone(), wide_unplaced.clone()),
            (wide_unplaced, narrow_placed),
            (narrow, placed),
        ] {
            for constructor in [CastOperation::checked, CastOperation::unchecked] {
                assert_eq!(
                    constructor(output.clone(), vec![input.clone()]).unwrap().infer_output_types(&[input.clone()], &[]),
                    Ok(vec![output.clone()]),
                );
            }
            for constructor in [CastOperation::<ArrayIrType>::checked, CastOperation::<ArrayIrType>::unchecked] {
                assert_eq!(
                    constructor(output.clone().into(), vec![input.clone().into()])
                        .unwrap()
                        .infer_output_types(&[input.clone().into()], &[]),
                    Ok(vec![output.clone().into()]),
                );
            }
        }
    }

    #[test]
    fn test_cast_type_inference_tiled_layout() {
        // Tiled layouts are representation changes like strided ones.
        let plain = ArrayType::new_static(DataType::F32, [4]);
        let tiled = plain
            .clone()
            .with_layout(Layout::Tiled(TiledLayout::new(vec![0], vec![Tile::new(vec![TileDimension::Sized(2)])])));
        check_operation_type_inference!(
            operation = CastOperation::checked(tiled.clone(), vec![plain.clone()]).unwrap(),
            cases = [{ input_types = [plain.clone()], output_types = [tiled.clone()] }],
        );
        check_operation_type_inference!(
            operation = CastOperation::unchecked(plain.clone(), vec![tiled.clone()]).unwrap(),
            cases = [{ input_types = [tiled], output_types = [plain] }],
        );
    }

    #[test]
    fn test_cast_type_inference_output_layout_validation() {
        // The requested layout must address the output shape, so a malformed request is rejected during type inference
        // rather than only when interpretation materializes the output or a backend lowers it.
        let plain = ArrayType::new_static(DataType::F32, [2, 3]);
        let tiled = plain.clone().with_layout(Layout::Tiled(TiledLayout::new(
            vec![1, 0],
            vec![Tile::new(vec![TileDimension::Sized(2), TileDimension::Sized(2)])],
        )));
        let rank_mismatch = plain.clone().with_layout(Layout::Strided(StridedLayout::new(vec![4])));
        let overlapping = plain.clone().with_layout(Layout::Strided(StridedLayout::new(vec![1, 1])));
        for constructor in [CastOperation::checked, CastOperation::unchecked] {
            let infer = |output: &ArrayType| {
                constructor(output.clone(), vec![plain.clone()])
                    .unwrap()
                    .infer_output_types(std::slice::from_ref(&plain), &[])
            };
            assert_eq!(infer(&tiled), Ok(vec![tiled.clone()]));
            assert_eq!(
                infer(&rank_mismatch),
                Err(TypeError::invalid(
                    "`cast` output layout is invalid: strided layout rank 1 does not match array rank 2",
                )),
            );
            assert_eq!(
                infer(&overlapping),
                Err(TypeError::invalid(
                    "`cast` output layout is invalid: strided layout stride 1 on axis 0 is smaller than the 4-byte \
                     span occupied by more minor axes and may alias array elements",
                )),
            );
        }

        // The composite universe applies the same rule to its array members.
        assert_eq!(
            CastOperation::<ArrayIrType>::checked(rank_mismatch.into(), vec![plain.clone().into()])
                .unwrap()
                .infer_output_types(&[plain.into()], &[]),
            Err(TypeError::invalid(
                "`cast` output layout is invalid: strided layout rank 1 does not match array rank 2",
            )),
        );
    }

    #[test]
    fn test_cast_type_inference_distributed_semantics() {
        // Manual variation and reduction state are semantic rather than optional placement: a supplied input may not
        // carry variation or reduction state that the declared source lacks, and the output may not add or drop them,
        // even when the supplied input matches the requested output exactly.
        let plain = ArrayType::new_static(DataType::F32, [2]);
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let replicated = Sharding::replicated(mesh, 1);
        let varying = plain.clone().with_sharding(replicated.clone().with_varying_manual_axes(["x"]).unwrap()).unwrap();
        let unreduced = plain.clone().with_sharding(replicated.clone().with_unreduced_axes(["x"]).unwrap()).unwrap();
        let reduced = plain.clone().with_sharding(replicated.with_reduced_axes(["x"]).unwrap()).unwrap();
        let variation = TypeError::invalid("`cast` supplied input changes the declared manual variation");
        let reduction = TypeError::invalid("`cast` supplied input changes the declared reduction state");
        let variation_change = TypeError::invalid(
            "`cast` cannot change varying manual axes; use `parallel_vary` to add variation or a reduction to \
             remove it",
        );
        for constructor in [CastOperation::checked, CastOperation::unchecked] {
            let infer = |output: &ArrayType, declared: &ArrayType, actual: &ArrayType| {
                constructor(output.clone(), vec![declared.clone()])
                    .unwrap()
                    .infer_output_types(&[actual.clone()], &[])
            };
            assert_eq!(infer(&varying, &plain, &varying), Err(variation.clone()));
            assert_eq!(infer(&unreduced, &plain, &unreduced), Err(reduction.clone()));
            assert_eq!(infer(&reduced, &plain, &reduced), Err(reduction.clone()));
            assert_eq!(infer(&varying, &plain, &plain), Err(variation_change.clone()));
            assert_eq!(infer(&plain, &varying, &varying), Err(variation_change.clone()));
            assert_eq!(
                infer(&unreduced, &plain, &plain),
                Err(TypeError::invalid(format!(
                    "`{CAST_OPERATION_NAME}` cannot convert type `{}` to unrelated type `{}`",
                    &plain, &unreduced,
                )))
            );
            assert_eq!(
                infer(&plain, &unreduced, &unreduced),
                Err(TypeError::invalid(format!(
                    "`{CAST_OPERATION_NAME}` cannot convert type `{}` to unrelated type `{}`",
                    &unreduced, &plain,
                )))
            );
            assert_eq!(
                infer(&reduced, &plain, &plain),
                Err(TypeError::invalid(format!(
                    "`{CAST_OPERATION_NAME}` cannot convert type `{}` to unrelated type `{}`",
                    &plain, &reduced,
                )))
            );
            assert_eq!(
                infer(&plain, &reduced, &reduced),
                Err(TypeError::invalid(format!(
                    "`{CAST_OPERATION_NAME}` cannot convert type `{}` to unrelated type `{}`",
                    &reduced, &plain,
                )))
            );
        }

        // The composite universe applies the same rules to its array members.
        assert_eq!(
            CastOperation::<ArrayIrType>::checked(varying.clone().into(), vec![plain.clone().into()])
                .unwrap()
                .infer_output_types(&[varying.into()], &[]),
            Err(variation),
        );
        assert_eq!(
            CastOperation::<ArrayIrType>::unchecked(plain.clone().into(), vec![reduced.clone().into()])
                .unwrap()
                .infer_output_types(&[reduced.clone().into()], &[]),
            Err(TypeError::invalid(format!(
                "`{CAST_OPERATION_NAME}` cannot convert type `{}` to unrelated type `{}`",
                &reduced, &plain,
            ))),
        );
    }

    #[test]
    fn test_cast_type_inference_placement() {
        // Placement follows the rules of `reshard`: explicit placement may be requested, changed, and dropped from
        // tracking, whereas the auto and manual axes that each dimension names are carried forward unchanged, together
        // with every explicit axis major to them.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("a", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let plain = ArrayType::new_static(DataType::F32, [4, 4]);
        let placed = |dimensions: [ShardingDimension; 2]| {
            plain.clone().with_sharding(Sharding::new(mesh.clone(), dimensions.to_vec()).unwrap()).unwrap()
        };
        let sharded = |axes: &[&str]| ShardingDimension::sharded(axes.iter().copied());
        let replicated = ShardingDimension::replicated;
        let auto_error = TypeError::invalid(
            "`cast` cannot change placement over auto mesh axes; use `constrain_sharding` to constrain placement \
             over them",
        );
        let manual_error = TypeError::invalid(
            "`cast` cannot change placement over manual mesh axes; use manual collectives for transitions over them",
        );
        let order_error = TypeError::invalid(
            "`cast` cannot change the mesh axes up to the last auto or manual mesh axis of dimension 0; only explicit \
             mesh axes minor to every auto and manual axis may change",
        );
        let cases = [
            (plain.clone(), placed([sharded(&["x"]), replicated()]), Ok(())),
            (placed([sharded(&["x"]), replicated()]), placed([replicated(), sharded(&["x"])]), Ok(())),
            (placed([sharded(&["x"]), replicated()]), plain.clone(), Ok(())),
            // Replicating over a mesh with auto axes names none of them.
            (plain.clone(), placed([replicated(), replicated()]), Ok(())),
            // Auto placement is carried forward or left untracked, never introduced, dropped, or moved, and the
            // explicit axes major to it are fixed because they determine which block each auto shard holds.
            (
                placed([sharded(&["a"]), replicated()]),
                placed([sharded(&["x", "a"]), replicated()]),
                Err(order_error.clone()),
            ),
            (
                placed([sharded(&["x", "a"]), replicated()]),
                placed([sharded(&["a"]), replicated()]),
                Err(order_error.clone()),
            ),
            (placed([sharded(&["a", "x"]), replicated()]), placed([sharded(&["a", "y"]), replicated()]), Ok(())),
            (placed([sharded(&["a"]), replicated()]), plain.clone(), Ok(())),
            (plain.clone(), placed([sharded(&["a"]), replicated()]), Err(auto_error.clone())),
            (placed([sharded(&["a"]), replicated()]), placed([replicated(), replicated()]), Err(auto_error.clone())),
            (placed([sharded(&["a"]), replicated()]), placed([replicated(), sharded(&["a"])]), Err(auto_error)),
            // Manual placement is carried forward, never introduced or dropped, and only explicit axes minor to it
            // may change. Neither manual nor auto axes may be reordered relative to each other or to explicit axes.
            (placed([sharded(&["m"]), replicated()]), placed([sharded(&["m"]), sharded(&["x"])]), Ok(())),
            (placed([sharded(&["m", "x"]), replicated()]), placed([sharded(&["m"]), replicated()]), Ok(())),
            (placed([sharded(&["m"]), replicated()]), placed([sharded(&["m", "x"]), replicated()]), Ok(())),
            (
                placed([sharded(&["m", "x"]), replicated()]),
                placed([sharded(&["x", "m"]), replicated()]),
                Err(order_error.clone()),
            ),
            (
                placed([sharded(&["a", "m"]), replicated()]),
                placed([sharded(&["m", "a"]), replicated()]),
                Err(order_error),
            ),
            (plain.clone(), placed([sharded(&["m"]), replicated()]), Err(manual_error.clone())),
            (placed([sharded(&["m"]), replicated()]), placed([sharded(&["x"]), replicated()]), Err(manual_error)),
        ];
        for (input, output, expected) in cases {
            let expected = expected.map(|()| vec![output.clone()]);
            for constructor in [CastOperation::checked, CastOperation::unchecked] {
                assert_eq!(
                    constructor(output.clone(), vec![input.clone()]).unwrap().infer_output_types(&[input.clone()], &[]),
                    expected,
                );
            }
            for constructor in [CastOperation::<ArrayIrType>::checked, CastOperation::<ArrayIrType>::unchecked] {
                assert_eq!(
                    constructor(output.clone().into(), vec![input.clone().into()])
                        .unwrap()
                        .infer_output_types(&[input.clone().into()], &[]),
                    expected.clone().map(|types| types.into_iter().map(ArrayIrType::from).collect()),
                );
            }
        }
    }

    #[test]
    fn test_cast_type_inference_mesh_axis_definitions() {
        // Auto and manual axes that the output carries forward keep their mesh axis definitions (size and type), even
        // when only their names are compared for placement and manual variation, whereas the output mesh may still add
        // or drop unrelated axes. Axes that carry reduction state, and explicit axes major to an auto or manual axis in
        // a dimension (whose sizes fix the blocks of its shards), keep their definitions, whereas explicit axes minor to
        // every auto and manual axis and referenced only by dimension placement may be redefined or replaced.
        let mesh = |axes: &[(&str, usize, MeshAxisType)]| {
            LogicalMesh::new(
                axes.iter().map(|(name, size, r#type)| MeshAxis::new(*name, *size, *r#type).unwrap()).collect(),
            )
            .unwrap()
        };
        let placed = |mesh: LogicalMesh, dimension: ShardingDimension| {
            ArrayType::new_static(DataType::F32, [8])
                .with_sharding(Sharding::new(mesh, vec![dimension]).unwrap())
                .unwrap()
        };
        let varying = |size: usize| {
            ArrayType::new_static(DataType::F32, [8])
                .with_sharding(
                    Sharding::replicated(mesh(&[("x", size, MeshAxisType::Manual)]), 1)
                        .with_varying_manual_axes(["x"])
                        .unwrap(),
                )
                .unwrap()
        };
        let unreduced = |mesh: LogicalMesh| {
            ArrayType::new_static(DataType::F32, [8])
                .with_sharding(Sharding::replicated(mesh, 1).with_unreduced_axes(["x"]).unwrap())
                .unwrap()
        };
        let reduced = |mesh: LogicalMesh| {
            ArrayType::new_static(DataType::F32, [8])
                .with_sharding(Sharding::replicated(mesh, 1).with_reduced_axes(["x"]).unwrap())
                .unwrap()
        };
        let sharded = ShardingDimension::sharded(["x"]);
        let auto = |size: usize| mesh(&[("x", size, MeshAxisType::Auto)]);
        let manual = |size: usize| mesh(&[("x", size, MeshAxisType::Manual)]);
        let explicit = |size: usize| mesh(&[("x", size, MeshAxisType::Explicit)]);
        let with_explicit = |size: usize| mesh(&[("x", size, MeshAxisType::Auto), ("y", 2, MeshAxisType::Explicit)]);
        let major_explicit = ShardingDimension::sharded(["x", "a"]);
        let major_manual = ShardingDimension::sharded(["x", "m"]);
        let minor_explicit = ShardingDimension::sharded(["a", "x"]);
        let with_minor_explicit =
            mesh(&[("a", 2, MeshAxisType::Auto), ("x", 2, MeshAxisType::Explicit), ("y", 2, MeshAxisType::Explicit)]);
        let cases = [
            (
                placed(auto(2), sharded.clone()),
                placed(auto(4), sharded.clone()),
                Err(TypeError::invalid("`cast` cannot redefine preserved auto mesh axis `'x'=2:auto` as `'x'=4:auto`")),
            ),
            (
                placed(manual(2), sharded.clone()),
                placed(manual(4), sharded.clone()),
                Err(TypeError::invalid(
                    "`cast` cannot redefine preserved manual mesh axis `'x'=2:manual` as `'x'=4:manual`",
                )),
            ),
            (
                varying(2),
                varying(4),
                Err(TypeError::invalid(
                    "`cast` cannot redefine preserved manual mesh axis `'x'=2:manual` as `'x'=4:manual`",
                )),
            ),
            (placed(auto(2), sharded.clone()), placed(with_explicit(2), sharded.clone()), Ok(())),
            (placed(with_explicit(2), sharded.clone()), placed(auto(2), sharded.clone()), Ok(())),
            (
                unreduced(explicit(2)),
                unreduced(explicit(4)),
                Err(TypeError::invalid(
                    "`cast` cannot redefine preserved explicit mesh axis `'x'=2:explicit` as `'x'=4:explicit`",
                )),
            ),
            (
                reduced(explicit(2)),
                reduced(explicit(4)),
                Err(TypeError::invalid(
                    "`cast` cannot redefine preserved explicit mesh axis `'x'=2:explicit` as `'x'=4:explicit`",
                )),
            ),
            (
                unreduced(explicit(2)),
                unreduced(manual(2)),
                Err(TypeError::invalid(
                    "`cast` cannot redefine preserved explicit mesh axis `'x'=2:explicit` as `'x'=2:manual`",
                )),
            ),
            (
                unreduced(explicit(2)),
                unreduced(auto(2)),
                Err(TypeError::invalid(
                    "`cast` cannot redefine preserved explicit mesh axis `'x'=2:explicit` as `'x'=2:auto`",
                )),
            ),
            (
                unreduced(explicit(2)),
                unreduced(mesh(&[("x", 2, MeshAxisType::Explicit), ("y", 2, MeshAxisType::Explicit)])),
                Ok(()),
            ),
            (
                placed(mesh(&[("x", 2, MeshAxisType::Explicit), ("a", 2, MeshAxisType::Auto)]), major_explicit.clone()),
                placed(mesh(&[("x", 4, MeshAxisType::Explicit), ("a", 2, MeshAxisType::Auto)]), major_explicit.clone()),
                Err(TypeError::invalid(
                    "`cast` cannot redefine preserved explicit mesh axis `'x'=2:explicit` as `'x'=4:explicit`",
                )),
            ),
            (
                placed(mesh(&[("x", 2, MeshAxisType::Explicit), ("m", 2, MeshAxisType::Manual)]), major_manual.clone()),
                placed(mesh(&[("x", 4, MeshAxisType::Explicit), ("m", 2, MeshAxisType::Manual)]), major_manual),
                Err(TypeError::invalid(
                    "`cast` cannot redefine preserved explicit mesh axis `'x'=2:explicit` as `'x'=4:explicit`",
                )),
            ),
            (
                placed(mesh(&[("a", 2, MeshAxisType::Auto), ("x", 2, MeshAxisType::Explicit)]), minor_explicit.clone()),
                placed(mesh(&[("a", 2, MeshAxisType::Auto), ("x", 4, MeshAxisType::Explicit)]), minor_explicit.clone()),
                Ok(()),
            ),
            (
                placed(with_minor_explicit.clone(), minor_explicit),
                placed(with_minor_explicit, ShardingDimension::sharded(["a", "y"])),
                Ok(()),
            ),
            (placed(explicit(2), sharded.clone()), placed(explicit(4), sharded), Ok(())),
        ];
        for (input, output, expected) in cases {
            let expected = expected.map(|()| vec![output.clone()]);
            for constructor in [CastOperation::checked, CastOperation::unchecked] {
                assert_eq!(
                    constructor(output.clone(), vec![input.clone()]).unwrap().infer_output_types(&[input.clone()], &[]),
                    expected,
                );
            }
        }
    }

    #[test]
    fn test_cast_type_inference_bounds() {
        // Differently bounded identities may be equated as long as their bounds overlap; the output then carries the
        // requested names together with its own layout and sharding, and disjoint bounds are rejected.
        let wide = DimensionBounds::positive(Some(8)).unwrap();
        let narrow = DimensionBounds::positive(Some(4)).unwrap();
        let [a, d] = ["a", "d"].map(|name| DimensionType::new(name, wide));
        let [b, c] = ["b", "c"].map(|name| DimensionType::new(name, narrow));
        let input = ArrayType::new(DataType::F32, Shape::new(vec![a.simplified_dimension(), c.simplified_dimension()]))
            .with_layout(Layout::Strided(StridedLayout::new(vec![12, 4])));
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let output =
            ArrayType::new(DataType::F32, Shape::new(vec![b.simplified_dimension(), d.simplified_dimension()]))
                .with_sharding(Sharding::new(mesh, vec![ShardingDimension::replicated(); 2]).unwrap())
                .unwrap();
        check_operation_type_inference!(
            operation = CastOperation::checked(output.clone(), vec![input.clone(), output.clone()]).unwrap(),
            cases = [{ input_types = [input.clone(), output.clone()], output_types = [output.clone()] }],
        );
        let disjoint = DimensionType::new("disjoint", DimensionBounds::new(4, Some(8)).unwrap());
        let disjoint =
            ArrayType::new(DataType::F32, Shape::new(vec![b.simplified_dimension(), disjoint.simplified_dimension()]));
        check_operation_type_inference!(
            operation = CastOperation::checked(disjoint.clone(), vec![input.clone(), disjoint.clone()]).unwrap(),
            cases = [{
                input_types = [input, disjoint],
                error = "`cast` cannot equate dimensions with disjoint bounds: `c` ∈ [1, 4), `disjoint` ∈ [4, 8)",
            }],
        );
    }

    #[test]
    fn test_cast_type_inference_declared_witnesses() {
        // Supplied identities cannot fill a missing declaration: that association would disappear when those
        // inputs become concrete. The declaration must retain the original witness's nominal identity.
        let bounds = DimensionBounds::positive(Some(8)).unwrap();
        let n = DimensionType::new("n", bounds);
        let other = DimensionType::new("other", bounds);
        let exact: ArrayIrType = ArrayType::new_static(DataType::F32, [2]).into();
        let output: ArrayIrType = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()])).into();
        check_operation_type_inference!(
            operation = CastOperation::checked(output.clone(), vec![exact.clone(), other.into()]).unwrap(),
            cases = [{
                input_types = [exact.clone(), n.clone().into()],
                error = "`cast` output identity `n` is not carried by a declared input",
            }],
        );
        check_operation_type_inference!(
            operation = CastOperation::checked(output.clone(), vec![exact.clone(), n.clone().into()]).unwrap(),
            cases = [{ input_types = [exact.clone(), n.into()], output_types = [output] }],
        );

        // This also applies to singleton actual axes, which can instantiate a static declaration without retaining
        // their names in it. Recording that name keeps the cast valid after all-known partial evaluation.
        let fixed = DimensionType::new("fixed", DimensionBounds::new(2, Some(3)).unwrap());
        let symbolic: ArrayIrType =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(fixed.to_variable())])).into();
        check_operation_type_inference!(
            operation = CastOperation::checked(symbolic.clone(), vec![exact.clone()]).unwrap(),
            cases = [{
                input_types = [symbolic.clone()],
                error = "`cast` output identity `fixed` is not carried by a declared input",
            }],
        );
        check_operation_type_inference!(
            operation = CastOperation::checked(symbolic.clone(), vec![symbolic]).unwrap(),
            cases = [{ input_types = [exact.clone()], output_types = [exact] }],
        );
    }

    #[test]
    fn test_cast_type_inference_renamed_dimension_witness() {
        // A first-class dimension witness may instantiate the declared identity under another name within its bounds.
        // The source keeps its own name, which the inferred output then carries instead of the requested one.
        let bounds = DimensionBounds::positive(Some(8)).unwrap();
        let [k, n, m] = ["k", "n", "m"].map(|name| DimensionType::new(name, bounds));
        let source: ArrayIrType = ArrayType::new(DataType::F32, Shape::new(vec![k.simplified_dimension()])).into();
        let target: ArrayIrType = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()])).into();
        for constructor in [CastOperation::checked, CastOperation::unchecked] {
            assert_eq!(
                constructor(target.clone(), vec![source.clone(), n.clone().into()])
                    .unwrap()
                    .infer_output_types(&[source.clone(), m.clone().into()], &[]),
                Ok(vec![source.clone()]),
            );
        }
    }

    #[test]
    fn test_cast_type_inference_proven_renaming() {
        // A consistent formal-to-actual renaming proves the relationship without a runtime check, so the inferred
        // output also has an interpretation representation.
        let bounds = DimensionBounds::positive(Some(8)).unwrap();
        let [m, n] = ["m", "n"].map(|name| DimensionType::new(name, bounds));
        let source = ArrayType::new(DataType::F32, Shape::new(vec![m.simplified_dimension()]));
        let output = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()]));
        let operation =
            CastOperation::<ArrayIrType>::checked(output.into(), vec![source.clone().into(), n.into()]).unwrap();
        assert_eq!(
            ArrayIrType::infer_cast_output_type(&operation, &[source.clone().into(), m.into()]),
            Ok((source.clone().into(), Some(source.into()))),
        );
    }

    #[test]
    fn test_cast_type_inference_swapped_dimension_witnesses() {
        // First-class dimension witnesses supplied in the declared order prove the identity cast. Swapping them
        // does not form a consistent renaming (declared `n` would correspond to both supplied `n` and `m`), so
        // the relationship needs a runtime check instead of being proven.
        let bounds = DimensionBounds::positive(Some(8)).unwrap();
        let [m, n] = ["m", "n"].map(|name| DimensionType::new(name, bounds));
        let source: ArrayIrType = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()])).into();
        for constructor in [CastOperation::checked, CastOperation::unchecked] {
            let operation =
                constructor(source.clone(), vec![source.clone(), m.clone().into(), n.clone().into()]).unwrap();
            assert_eq!(
                ArrayIrType::infer_cast_output_type(&operation, &[source.clone(), m.clone().into(), n.clone().into()]),
                Ok((source.clone(), Some(source.clone()))),
            );
            assert_eq!(
                ArrayIrType::infer_cast_output_type(&operation, &[source.clone(), n.clone().into(), m.clone().into()]),
                Ok((source.clone(), None)),
            );
        }
    }

    #[test]
    fn test_cast_type_inference_refined_layout_and_sharding() {
        // A supplied input may carry a layout or sharding that the unannotated declaration leaves unspecified, and the
        // output still takes the requested representation rather than the supplied one.
        let plain = ArrayType::new_static(DataType::U8, [2]);
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let other_mesh = LogicalMesh::new(vec![MeshAxis::new("y", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let actual = plain.clone().with_layout(Layout::Strided(StridedLayout::new(vec![2])));
        let target = plain.clone().with_layout(Layout::Strided(StridedLayout::new(vec![3])));
        let actual_sharding = plain.clone().with_sharding(Sharding::replicated(mesh, 1)).unwrap();
        let target_sharding = plain.clone().with_sharding(Sharding::replicated(other_mesh, 1)).unwrap();
        for constructor in [CastOperation::checked, CastOperation::unchecked] {
            assert_eq!(
                constructor(target.clone(), vec![plain.clone()]).unwrap().infer_output_types(&[actual.clone()], &[]),
                Ok(vec![target.clone()]),
            );
            assert_eq!(
                constructor(target_sharding.clone(), vec![plain.clone()])
                    .unwrap()
                    .infer_output_types(&[actual_sharding.clone()], &[]),
                Ok(vec![target_sharding.clone()]),
            );
        }
    }

    #[test]
    fn test_cast_type_inference_output_identity_classes() {
        let bounds = DimensionBounds::positive(Some(8)).unwrap();
        let [n, m, other] = ["n", "m", "other"].map(|name| DimensionType::new(name, bounds));
        let exact: ArrayIrType = ArrayType::new_static(DataType::F32, [2]).into();
        let output: ArrayIrType = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()])).into();
        for constructor in [CastOperation::checked, CastOperation::unchecked] {
            let operation =
                constructor(output.clone(), vec![exact.clone(), n.clone().into(), other.clone().into()]).unwrap();
            // Declared n corresponds to supplied m. The other supplied n has a separate role and cannot supply the
            // output's nominal binding merely because it uses the same identity object as the declaration. Resolve
            // the formal n through its own class to the source's exact extent instead.
            assert_eq!(
                operation.infer_output_types(&[exact.clone(), m.clone().into(), n.clone().into()], &[]),
                Ok(vec![exact.clone()]),
            );
            assert_eq!(
                operation.infer_output_types(&[exact.clone(), n.clone().into(), m.clone().into()], &[]),
                Ok(vec![output.clone()]),
            );
        }
    }

    #[test]
    fn test_cast_type_inference_scoped_output_identity() {
        // Declared `b` corresponds to supplied `s`, whereas the supplied `b` witnesses the declared `a`. The output must
        // name the supplied identity of declared `b`'s own class even though a supplied identity shares its name;
        // otherwise the staged program would relate the source to the unrelated `a` witness.
        let bounds = DimensionBounds::new(0, Some(4)).unwrap();
        let [a, b, s] = ["a", "b", "s"].map(|name| DimensionType::new(name, bounds));
        let vector = |dimension: &DimensionType| {
            ArrayIrType::from(ArrayType::new(DataType::F32, Shape::new(vec![dimension.simplified_dimension()])))
        };
        let operation =
            CastOperation::checked(vector(&b), vec![vector(&s), a.clone().into(), b.clone().into()]).unwrap();
        let actual = vec![vector(&s), b.clone().into(), s.clone().into()];
        assert_eq!(ArrayIrType::infer_cast_output_type(&operation, &actual), Ok((vector(&s), Some(vector(&s)))));

        // These values satisfy the declared relationship (`s = b = 1` and `a = 3`), so the staged program does too.
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let inputs = actual.into_iter().map(|r#type| builder.add_input(r#type)).collect();
        let output = builder.add_instruction(operation, vec![], inputs, None).unwrap()[0];
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
                lambda %0:f32[s], %1:dimension<b ∈ [0, 4)>, %2:dimension<s ∈ [0, 4)> .
                let %3:f32[s] = cast [
                    type=f32[b],
                    inputs=[f32[s], dimension<a ∈ [0, 4)>, dimension<b ∈ [0, 4)>],
                    checked=true,
                ] %0 %1 %2
                in (%3)"},
        );
        let value = ArrayIrValue::Array(Array::vector(vec![1f32]).unwrap());
        let dimension = |extent: usize| ArrayIrValue::Dimension(DimensionValue::constant(extent).unwrap());
        assert_eq!(program.interpret(vec![value.clone(), dimension(3), dimension(1)]), Ok(vec![value]));
    }

    #[test]
    fn test_cast_type_inference_symbolic_witness_bounds() {
        // A differently named witness instantiates the declared identity only within its own bounds, so a static
        // output outside those bounds is rejected.
        let bounds = DimensionBounds::positive(Some(8)).unwrap();
        let n = DimensionType::new("n", bounds);
        let m = DimensionType::new("m", DimensionBounds::new(4, Some(8)).unwrap());
        let source: ArrayIrType = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()])).into();
        let exact = |extent| ArrayIrType::from(ArrayType::new_static(DataType::F32, [extent]));
        for constructor in [CastOperation::checked, CastOperation::unchecked] {
            let declarations = vec![source.clone(), n.clone().into()];
            let actual = [source.clone(), m.clone().into()];
            assert_eq!(
                constructor(exact(2), declarations.clone()).unwrap().infer_output_types(&actual, &[]),
                Err(TypeError::custom(DimensionError::BindingOutOfBounds {
                    variable: "m".to_owned(),
                    value: 2,
                    bounds: m.bounds(),
                })),
            );
            assert_eq!(
                constructor(exact(5), declarations).unwrap().infer_output_types(&actual, &[]),
                Ok(vec![exact(5)])
            );
        }

        // Neither witness class is exact. Joining them through the cast still requires a common admitted extent.
        let other = DimensionType::new("other", bounds);
        let lower = DimensionType::new("lower", DimensionBounds::positive(Some(4)).unwrap());
        let upper = DimensionType::new("upper", DimensionBounds::new(4, Some(8)).unwrap());
        let overlapping = DimensionType::new("overlapping", DimensionBounds::new(3, Some(8)).unwrap());
        let output: ArrayIrType = ArrayType::new(DataType::F32, Shape::new(vec![other.simplified_dimension()])).into();
        for constructor in [CastOperation::checked, CastOperation::unchecked] {
            let declarations = vec![source.clone(), n.clone().into(), output.clone(), other.clone().into()];
            let operation = constructor(output.clone(), declarations).unwrap();
            assert_eq!(
                operation.infer_output_types(
                    &[source.clone(), lower.clone().into(), output.clone(), upper.clone().into()],
                    &[],
                ),
                Err(TypeError::invalid(
                    "`cast` cannot equate dimensions with disjoint bounds: `n` ∈ [1, 8), `other` ∈ [1, 8), \
                     `lower` ∈ [1, 4), `upper` ∈ [4, 8)",
                )),
            );
            assert_eq!(
                operation.infer_output_types(
                    &[source.clone(), lower.clone().into(), output.clone(), overlapping.clone().into()],
                    &[],
                ),
                Ok(vec![output.clone()]),
            );
        }
    }

    #[test]
    fn test_cast_interpretation() {
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let symbolic: ArrayIrType = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()])).into();
        let exact: ArrayIrType = ArrayType::new_static(DataType::F32, [2]).into();
        let vector = |extent: usize| ArrayIrValue::Array(Array::vector(vec![1f32; extent]).unwrap());
        let dimension = |extent: usize| ArrayIrValue::Dimension(DimensionValue::constant(extent).unwrap());
        let context = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();

        // Interpretation checks the complete relationship over the concrete types in both modes: widening `f32[2]` to
        // `f32[n]` next to a witness of extent 5 is rejected, and so is narrowing a value of the wrong extent.
        let mismatch = Err(ProgramError::Type(TypeError::custom(DimensionError::InputDimensionMismatch {
            dimension: "n".to_owned(),
            expected: 5,
            actual: 2,
        })));
        let widening = CastOperation::unchecked(symbolic.clone(), vec![exact.clone(), n.clone().into()]).unwrap();
        assert_eq!(widening.interpret(&context, &EmptyRegionDriver, &[vector(2), dimension(2)]), Ok(vec![vector(2)]));
        assert_eq!(widening.interpret(&context, &EmptyRegionDriver, &[vector(2), dimension(5)]), mismatch);
        let checked = CastOperation::checked(symbolic.clone(), vec![exact.clone(), n.clone().into()]).unwrap();
        assert_eq!(checked.interpret(&context, &EmptyRegionDriver, &[vector(2), dimension(2)]), Ok(vec![vector(2)]));
        assert_eq!(checked.interpret(&context, &EmptyRegionDriver, &[vector(2), dimension(5)]), mismatch);
        let narrowing = CastOperation::unchecked(exact.clone(), vec![symbolic]).unwrap();
        assert_eq!(narrowing.interpret(&context, &EmptyRegionDriver, &[vector(2)]), Ok(vec![vector(2)]));
        assert_eq!(
            narrowing.interpret(&context, &EmptyRegionDriver, &[vector(3)]),
            Err(ProgramError::Type(TypeError::invalid(
                "`cast` cannot convert type `f32[3]` to unrelated type `f32[2]`",
            ))),
        );

        // Witnesses declared with a shared identity must agree with each other even when the output is static, so a
        // cast can act as a guard over its witnesses.
        let guard = CastOperation::checked(exact.clone(), vec![exact.clone(), n.clone().into(), n.into()]).unwrap();
        assert_eq!(
            guard.interpret(&context, &EmptyRegionDriver, &[vector(2), dimension(2), dimension(2)]),
            Ok(vec![vector(2)]),
        );
        assert_eq!(
            guard.interpret(&context, &EmptyRegionDriver, &[vector(2), dimension(2), dimension(3)]),
            Err(ProgramError::Type(TypeError::custom(DimensionError::InputDimensionMismatch {
                dimension: "n".to_owned(),
                expected: 2,
                actual: 3,
            }))),
        );

        // A reference cannot be the source of a cast.
        let reference = ArrayIrValue::Reference(ArrayReference::new(Array::vector(vec![1f32; 2]).unwrap()));
        assert_eq!(
            CastOperation::unchecked(exact.clone(), vec![exact]).unwrap().interpret(
                &context,
                &EmptyRegionDriver,
                &[reference]
            ),
            Err(ProgramError::Type(TypeError::invalid("`cast` does not support reference type `ref<f32[2]>`"))),
        );
    }

    #[test]
    fn test_cast_interpretation_symbolic_relationships() {
        let bounds = DimensionBounds::positive(Some(8)).unwrap();
        let [a, b] = ["a", "b"].map(|name| DimensionType::new(name, bounds));
        let vector = ArrayType::new(DataType::F32, Shape::new(vec![a.simplified_dimension()]));
        let matrix =
            ArrayType::new(DataType::F32, Shape::new(vec![a.simplified_dimension(), b.simplified_dimension()]));
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let exact = context.input(ArrayType::new_static(DataType::F32, [2]));
        let witness = context.input(vector.clone());
        let value = context.input(matrix.clone());
        let expected = Err(ProgramError::Concretization {
            message: "`cast` cannot check symbolic type relationships during interpretation; \
                      bind the operation through its context"
                .to_owned(),
        });

        // Symbolic values carry no concrete extents, so a relationship that needs a runtime check is rejected in both
        // modes rather than staged.
        for constructor in [CastOperation::checked, CastOperation::unchecked] {
            let widening =
                constructor(vector.clone(), vec![ArrayType::new_static(DataType::F32, [2]), vector.clone()]).unwrap();
            assert_eq!(widening.interpret(&context, &EmptyRegionDriver, &[exact.clone(), witness.clone()]), expected);
        }

        // A type-level proof is sufficient when no runtime relationship remains to be checked.
        assert_eq!(
            CastOperation::checked(matrix.clone(), vec![matrix]).unwrap().interpret(
                &context,
                &EmptyRegionDriver,
                std::slice::from_ref(&value),
            ),
            Ok(vec![value]),
        );

        // A requested symbolic output is not a second runtime observation. The exact input proves this relation.
        assert_eq!(
            CastOperation::checked(vector.clone(), vec![vector]).unwrap().interpret(
                &context,
                &EmptyRegionDriver,
                std::slice::from_ref(&exact),
            ),
            Ok(vec![exact]),
        );

        // Optional layout and sharding must not turn an unproven narrowing into a type-level proof or stage it before
        // rejecting the input.
        let narrowed =
            ArrayType::new_static(DataType::F32, [2]).with_layout(Layout::Strided(StridedLayout::new(vec![4])));
        let symbolic = context.input(ArrayType::new(DataType::F32, Shape::new(vec![a.simplified_dimension()])));
        let operation = CastOperation::checked(narrowed, vec![symbolic.r#type().into_owned()]).unwrap();
        assert_eq!(operation.interpret(&context, &EmptyRegionDriver, &[symbolic.clone()]), expected);

        // Equal bounds do not prove a different nominal output identity, and interpretation must not stage it.
        let fresh = DimensionType::new("fresh", bounds);
        let unrelated = ArrayType::new(DataType::F32, Shape::new(vec![fresh.simplified_dimension()]));
        for constructor in [CastOperation::checked, CastOperation::unchecked] {
            let operation = constructor(unrelated.clone(), vec![symbolic.r#type().into_owned()]).unwrap();
            assert_eq!(
                ArrayType::infer_cast_output_type(&operation, &[symbolic.r#type().into_owned()]),
                Err(TypeError::invalid("`cast` output identity `fresh` is not carried by a declared input")),
            );
            assert_eq!(
                operation.interpret(&context, &EmptyRegionDriver, &[symbolic.clone()]),
                Err(ProgramError::Type(TypeError::invalid(
                    "`cast` output identity `fresh` is not carried by a declared input",
                ))),
            );
        }
        assert!(context.builder().borrow().instructions().is_empty());
    }

    #[test]
    fn test_cast_interpretation_symbolic_layout_and_sharding() {
        let plain = ArrayType::new_static(DataType::F32, [2]);
        let name = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let output = ArrayType::new(DataType::F32, Shape::new(vec![name.simplified_dimension()]))
            .with_layout(Layout::Strided(StridedLayout::new(vec![8])))
            .with_sharding(Sharding::replicated(mesh, 1))
            .unwrap();
        let exact_output = output.clone().with_shape(plain.shape().clone());
        for constructor in [CastOperation::checked, CastOperation::unchecked] {
            let operation =
                constructor(output.clone(), vec![output.clone().with_layout(None).with_sharding(None).unwrap()])
                    .unwrap();
            assert_eq!(
                ArrayType::infer_cast_output_type(&operation, &[plain.clone()]),
                Ok((exact_output.clone(), Some(exact_output.clone()))),
            );
            // The original declaration carries the output name, and the exact input proves its extent. Only the
            // representation change is staged, with no unbound identity after concretization.
            let (inferred, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
                |value| {
                    let mut outputs = operation.interpret(value.context(), &EmptyRegionDriver, &[value.clone()])?;
                    Ok(outputs.remove(0))
                },
                plain.clone(),
            )
            .unwrap();
            assert_eq!(inferred, exact_output);
            assert_eq!(
                program.to_string(),
                indoc! {"
                    lambda %0:f32[2] .
                    let %1:f32[2][layout=strided{8}][sharding={mesh<['x'=1:explicit]>, [{}]}] = cast [
                        type=f32[2][layout=strided{8}][sharding={mesh<['x'=1:explicit]>, [{}]}],
                        inputs=[f32[2]],
                    ] %0
                    in (%1)"},
            );
            let value = Array::vector(vec![1f32, 2f32]).unwrap();
            assert_eq!(
                program.interpret(value),
                Ok(Array::from_elements(exact_output.clone(), &[1f32, 2f32]).unwrap()),
            );
        }
    }

    #[test]
    fn test_cast_interpretation_declared_dimensions() {
        let bounds = DimensionBounds::non_negative(Some(5)).unwrap();
        let [a, b] = ["a", "b"].map(|name| DimensionType::new(name, bounds));
        let input = ArrayType::new(DataType::F32, Shape::new(vec![a.simplified_dimension(), b.simplified_dimension()]));
        let output =
            ArrayType::new(DataType::F32, Shape::new(vec![b.simplified_dimension(), a.simplified_dimension()]));
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let square = Array::matrix(2, 2, vec![1f32; 4]).unwrap();
        let rectangular = Array::matrix(2, 3, vec![1f32; 6]).unwrap();
        let empty = Array::matrix(0, 0, Vec::<f32>::new()).unwrap();
        let mismatch = Err(ProgramError::Type(TypeError::custom(DimensionError::InputDimensionMismatch {
            dimension: "b".to_owned(),
            expected: 3,
            actual: 2,
        })));

        // Concrete execution must retain a = 2, b = 3 from the input declaration. It cannot assign those same names
        // new extents from the requested output, even when each axis would separately admit the requested extent.
        for constructor in [CastOperation::checked, CastOperation::unchecked] {
            let operation = constructor(output.clone(), vec![input.clone()]).unwrap();
            assert_eq!(
                operation.interpret(&context, &EmptyRegionDriver, std::slice::from_ref(&square)),
                Ok(vec![square.clone()]),
            );
            assert_eq!(operation.interpret(&context, &EmptyRegionDriver, std::slice::from_ref(&rectangular)), mismatch);
            assert_eq!(
                operation.interpret(&context, &EmptyRegionDriver, std::slice::from_ref(&empty)),
                Ok(vec![empty.clone()]),
            );
        }

        // A cast that preserves its declared type accepts rectangular shapes.
        let operation = CastOperation::checked(input.clone(), vec![input]).unwrap();
        assert_eq!(operation.interpret(&context, &EmptyRegionDriver, &[rectangular.clone()]), Ok(vec![rectangular]));
    }

    #[test]
    fn test_cast_interpretation_repeated_identity() {
        // Narrowing `f32[n, n]` to `f32[n, 2]` requires `n = 2`. Symbolic values leave it unproven, so interpretation
        // rejects the cast rather than staging it, whereas concrete values either prove or contradict it.
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let square =
            ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension(), n.simplified_dimension()]));
        let output = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension(), Dimension::Static(2)]));
        let tracing = TracingContext::<Array, ArrayOperation<Array>>::new();
        let symbolic = tracing.input(square.clone());
        let eager = EagerContext::<Array, ArrayOperation<Array>>::new();
        let proven = Array::matrix(2, 2, vec![1f32; 4]).unwrap();
        let contradicting = Array::matrix(2, 3, vec![1f32; 6]).unwrap();
        for constructor in [CastOperation::checked, CastOperation::unchecked] {
            let operation = constructor(output.clone(), vec![square.clone()]).unwrap();
            assert_eq!(
                operation.interpret(&tracing, &EmptyRegionDriver, std::slice::from_ref(&symbolic)),
                Err(ProgramError::Concretization {
                    message: "`cast` cannot check symbolic type relationships during interpretation; \
                              bind the operation through its context"
                        .to_owned(),
                }),
            );
            assert_eq!(
                operation.interpret(&eager, &EmptyRegionDriver, std::slice::from_ref(&proven)),
                Ok(vec![proven.clone()]),
            );
            assert_eq!(
                operation.interpret(&eager, &EmptyRegionDriver, std::slice::from_ref(&contradicting)),
                Err(ProgramError::Type(TypeError::custom(DimensionError::InputDimensionMismatch {
                    dimension: "n".to_owned(),
                    expected: 2,
                    actual: 3,
                }))),
            );
        }
        assert!(tracing.builder().borrow().instructions().is_empty());
    }

    #[test]
    fn test_cast_interpretation_dimension_matrix() {
        // Each requested axis is compared with the unchanged runtime shape under the names that the source and witness
        // declarations assign, so every combination of renaming, narrowing to a static axis, and witnessed axes
        // accepts exactly the shapes that agree with those names.
        let bounds = DimensionBounds::non_negative(Some(4)).unwrap();
        let [a, b, c] = ["a", "b", "c"].map(|name| DimensionType::new(name, bounds));
        let source =
            ArrayType::new(DataType::F32, Shape::new(vec![a.simplified_dimension(), b.simplified_dimension()]));
        let witness = ArrayType::new(DataType::F32, Shape::new(vec![c.simplified_dimension()]));
        let target = |dimensions: Vec<Dimension>| ArrayType::new(DataType::F32, Shape::new(dimensions));
        let matrix = |rows, columns| Array::matrix(rows, columns, vec![1f32; rows * columns]).unwrap();
        let vector = |extent| Array::vector(vec![1f32; extent]).unwrap();
        let mismatch = |dimension: &str, expected, actual| {
            Err(ProgramError::Type(TypeError::custom(DimensionError::InputDimensionMismatch {
                dimension: dimension.to_owned(),
                expected,
                actual,
            })))
        };
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        for constructor in [CastOperation::checked, CastOperation::unchecked] {
            let interpret = |dimensions: Vec<Dimension>, value: Array, witness_value: Array| {
                constructor(target(dimensions), vec![source.clone(), witness.clone()]).unwrap().interpret(
                    &context,
                    &EmptyRegionDriver,
                    &[value, witness_value],
                )
            };
            let identity = vec![a.simplified_dimension(), b.simplified_dimension()];
            assert_eq!(interpret(identity.clone(), matrix(2, 3), vector(2)), Ok(vec![matrix(2, 3)]));
            assert_eq!(interpret(identity, matrix(0, 0), vector(0)), Ok(vec![matrix(0, 0)]));
            let swapped = vec![b.simplified_dimension(), a.simplified_dimension()];
            assert_eq!(interpret(swapped.clone(), matrix(2, 2), vector(2)), Ok(vec![matrix(2, 2)]));
            assert_eq!(interpret(swapped.clone(), matrix(0, 0), vector(0)), Ok(vec![matrix(0, 0)]));
            assert_eq!(interpret(swapped, matrix(2, 3), vector(2)), mismatch("b", 3, 2));
            let narrowed_rows = vec![Dimension::Static(2), a.simplified_dimension()];
            assert_eq!(interpret(narrowed_rows.clone(), matrix(2, 2), vector(2)), Ok(vec![matrix(2, 2)]));
            assert_eq!(
                interpret(narrowed_rows, matrix(3, 3), vector(2)),
                Err(ProgramError::Type(TypeError::invalid(
                    "`cast` cannot convert type `f32[3, 3]` to unrelated type `f32[2, a]`",
                ))),
            );
            let narrowed_columns = vec![b.simplified_dimension(), Dimension::Static(2)];
            assert_eq!(interpret(narrowed_columns.clone(), matrix(2, 2), vector(2)), Ok(vec![matrix(2, 2)]));
            assert_eq!(interpret(narrowed_columns, matrix(3, 2), vector(2)), mismatch("b", 2, 3));
            let witnessed = vec![c.simplified_dimension(), a.simplified_dimension()];
            assert_eq!(interpret(witnessed.clone(), matrix(2, 2), vector(2)), Ok(vec![matrix(2, 2)]));
            assert_eq!(interpret(witnessed, matrix(2, 2), vector(3)), mismatch("c", 3, 2));
            let fixed = vec![Dimension::Static(2), Dimension::Static(2)];
            assert_eq!(interpret(fixed.clone(), matrix(2, 2), vector(2)), Ok(vec![matrix(2, 2)]));
            assert_eq!(
                interpret(fixed, matrix(2, 3), vector(2)),
                Err(ProgramError::Type(TypeError::invalid(
                    "`cast` cannot convert type `f32[2, 3]` to unrelated type `f32[2, 2]`",
                ))),
            );
        }
    }

    #[test]
    fn test_cast_interpretation_layout_and_sharding() {
        // A layout change re-encodes the logical bytes and a sharding annotation only relabels them, in both modes,
        // whether or not the supplied input already carries the requested sharding.
        let plain = ArrayType::new_static(DataType::F32, [2]);
        let input = plain.clone().with_layout(Layout::Strided(StridedLayout::new(vec![4])));
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding = Sharding::new(mesh, vec![ShardingDimension::replicated()]).unwrap();
        let output = plain.clone().with_sharding(sharding.clone()).unwrap();
        let expected_value = Array::from_elements(output.clone(), &[1f32, 2f32]).unwrap();
        let unannotated = Array::from_elements(input.clone(), &[1f32, 2f32]).unwrap();
        let actual = Array::from_elements(input.clone().with_sharding(sharding).unwrap(), &[1f32, 2f32]).unwrap();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let value = builder.add_input(input.clone());
        let checked = builder
            .add_instruction(
                CastOperation::checked(output.clone(), vec![input.clone()]).unwrap(),
                Vec::new(),
                vec![value],
                None,
            )
            .unwrap()[0];
        let unchecked = builder
            .add_instruction(CastOperation::unchecked(output, vec![input]).unwrap(), Vec::new(), vec![value], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![checked, unchecked], vec![Placeholder], vec![Placeholder; 2])
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2][layout=strided{4}] .
                let %1:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}] = cast [
                    type=f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}],
                    inputs=[f32[2][layout=strided{4}]],
                    checked=true,
                ] %0
                    %2:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}] = cast [
                        type=f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}],
                        inputs=[f32[2][layout=strided{4}]],
                    ] %0
                in (%1, %2)"},
        );
        assert_eq!(program.interpret(vec![actual]), Ok(vec![expected_value.clone(); 2]));
        assert_eq!(program.interpret(vec![unannotated]), Ok(vec![expected_value; 2]));
    }

    #[test]
    fn test_cast_interpretation_unspecified_layout_and_sharding() {
        // An output that adds a layout, a sharding, or both to an unannotated input materializes the requested
        // representation in both modes and in both universes.
        let input = Array::vector(vec![10u8, 20]).unwrap();
        let plain = input.r#type().into_owned();
        let layout = Layout::Strided(StridedLayout::new(vec![2]));
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding = Sharding::replicated(mesh, 1);
        for output_type in [
            plain.clone().with_layout(layout.clone()),
            plain.clone().with_sharding(sharding.clone()).unwrap(),
            plain.clone().with_layout(layout).with_sharding(sharding).unwrap(),
        ] {
            let expected = Array::from_elements(output_type.clone(), &[10u8, 20]).unwrap();
            for constructor in [CastOperation::checked, CastOperation::unchecked] {
                assert_eq!(
                    constructor(output_type.clone(), vec![plain.clone()]).unwrap().interpret(
                        &EagerContext::<Array, ArrayOperation<Array>>::new(),
                        &EmptyRegionDriver,
                        &[input.clone()],
                    ),
                    Ok(vec![expected.clone()]),
                );
            }
            for constructor in [CastOperation::<ArrayIrType>::checked, CastOperation::<ArrayIrType>::unchecked] {
                assert_eq!(
                    constructor(output_type.clone().into(), vec![plain.clone().into()]).unwrap().interpret(
                        &EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new(),
                        &EmptyRegionDriver,
                        &[ArrayIrValue::Array(input.clone())],
                    ),
                    Ok(vec![ArrayIrValue::Array(expected.clone())]),
                );
            }
        }
    }

    #[test]
    fn test_cast_interpretation_refined_layout_and_sharding() {
        // A supplied input whose layout or sharding the declaration leaves unspecified is re-encoded or relabeled into
        // the requested representation.
        let plain = ArrayType::new_static(DataType::U8, [2]);
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let other_mesh = LogicalMesh::new(vec![MeshAxis::new("y", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let actual = plain.clone().with_layout(Layout::Strided(StridedLayout::new(vec![2])));
        let target = plain.clone().with_layout(Layout::Strided(StridedLayout::new(vec![3])));
        let actual_sharding = plain.clone().with_sharding(Sharding::replicated(mesh, 1)).unwrap();
        let target_sharding = plain.clone().with_sharding(Sharding::replicated(other_mesh, 1)).unwrap();
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        for constructor in [CastOperation::checked, CastOperation::unchecked] {
            assert_eq!(
                constructor(target.clone(), vec![plain.clone()]).unwrap().interpret(
                    &context,
                    &EmptyRegionDriver,
                    &[Array::from_elements(actual.clone(), &[10u8, 20]).unwrap()],
                ),
                Ok(vec![Array::from_elements(target.clone(), &[10u8, 20]).unwrap()]),
            );
            assert_eq!(
                constructor(target_sharding.clone(), vec![plain.clone()]).unwrap().interpret(
                    &context,
                    &EmptyRegionDriver,
                    &[Array::from_elements(actual_sharding.clone(), &[10u8, 20]).unwrap()],
                ),
                Ok(vec![Array::from_elements(target_sharding.clone(), &[10u8, 20]).unwrap()]),
            );
        }
    }

    #[test]
    fn test_cast_interpretation_witnessed_storage_sharing() {
        // A witnessed cast that keeps the layout shares the input's storage, whether or not it adds a sharding.
        let plain = ArrayType::new_static(DataType::F32, [2]);
        let strided = plain.clone().with_layout(Layout::Strided(StridedLayout::new(vec![8])));
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let annotated = strided.clone().with_sharding(Sharding::replicated(mesh, 1)).unwrap();
        let value = Array::from_elements(strided.clone(), &[1f32, 2f32]).unwrap();
        let witness = Array::vector(vec![3f32, 4f32]).unwrap();
        for output in [strided, annotated] {
            for constructor in [CastOperation::checked, CastOperation::unchecked] {
                let operation = constructor(output.clone(), vec![plain.clone(), plain.clone()]).unwrap();
                let result = operation
                    .interpret(
                        &EagerContext::<Array, ArrayOperation<Array>>::new(),
                        &EmptyRegionDriver,
                        &[value.clone(), witness.clone()],
                    )
                    .unwrap();
                assert_eq!(result, vec![Array::from_elements(output.clone(), &[1f32, 2f32]).unwrap()]);
                assert!(Arc::ptr_eq(value.shared_storage_bytes(), result[0].shared_storage_bytes()));
            }
        }
    }

    #[test]
    fn test_cast_interpretation_ordering() {
        let bounds = DimensionBounds::positive(Some(5)).unwrap();
        let [a, b] = ["a", "b"].map(|name| DimensionType::new(name, bounds));
        let input = ArrayType::new(DataType::F32, Shape::new(vec![a.simplified_dimension(), b.simplified_dimension()]));
        let output =
            ArrayType::new(DataType::F32, Shape::new(vec![b.simplified_dimension(), a.simplified_dimension()]));
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let value = builder.add_input(input.clone());
        builder
            .add_instruction(CastOperation::checked(output, vec![input]).unwrap(), Vec::new(), vec![value], None)
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
                let %1:f32[b, a] = cast [type=f32[b, a], inputs=[f32[a, b]], checked=true] %0
                    %2:bool[] = const false
                    () = assert [message="later assertion", labels=[]] %2
                in ()"#},
        );

        // A discarded result must still check immediately, before the following ordered assertion. The valid square
        // control reaches that assertion, proving that the failing rectangular case stops at the cast itself.
        let mismatch = ProgramError::Type(TypeError::custom(DimensionError::InputDimensionMismatch {
            dimension: "b".to_owned(),
            expected: 3,
            actual: 2,
        }));
        assert_eq!(program.interpret(vec![Array::matrix(2, 3, vec![1f32; 6]).unwrap()]), Err(mismatch.clone()));
        let error = program.interpret(vec![Array::matrix(2, 2, vec![1f32; 4]).unwrap()]).unwrap_err();
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::Failed { message: "later assertion".to_owned(), observations: Vec::new() }),
        );
        assert_eq!(
            program
                .partially_evaluate(&[PartialValue::Known(Array::matrix(2, 3, vec![1f32; 6]).unwrap())])
                .unwrap_err(),
            mismatch,
        );
    }

    #[test]
    fn test_cast_interpretation_renamed_dimension_witness() {
        // A staged cast whose first-class dimension witness instantiates the declared identity under another name
        // keeps the source's name in its output, and concrete execution still checks the witness against the source.
        let bounds = DimensionBounds::positive(Some(8)).unwrap();
        let [k, n, m] = ["k", "n", "m"].map(|name| DimensionType::new(name, bounds));
        let source: ArrayIrType = ArrayType::new(DataType::F32, Shape::new(vec![k.simplified_dimension()])).into();
        let target: ArrayIrType = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()])).into();
        let value = ArrayIrValue::Array(Array::vector(vec![1f32, 2f32]).unwrap());
        let dimension = |extent: usize| ArrayIrValue::Dimension(DimensionValue::constant(extent).unwrap());
        let constructors: [(fn(ArrayIrType, Vec<ArrayIrType>) -> Result<CastOperation<ArrayIrType>, TypeError>, &str);
            2] = [(CastOperation::checked, ", checked=true"), (CastOperation::unchecked, "")];
        for (constructor, mode) in constructors {
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
            let input = builder.add_input(source.clone());
            let witness = builder.add_input(m.clone().into());
            let output = builder
                .add_instruction(
                    constructor(target.clone(), vec![source.clone(), n.clone().into()]).unwrap(),
                    vec![],
                    vec![input, witness],
                    None,
                )
                .unwrap()[0];
            let program = builder
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                    vec![output],
                    vec![Placeholder; 2],
                    vec![Placeholder],
                )
                .unwrap();
            assert_eq!(
                program.to_string(),
                formatdoc! {"
                    lambda %0:f32[k], %1:dimension<m ∈ [1, 8)> .
                    let %2:f32[k] = cast [type=f32[n], inputs=[f32[k], dimension<n ∈ [1, 8)>]{mode}] %0 %1
                    in (%2)"},
            );
            assert_eq!(program.interpret(vec![value.clone(), dimension(2)]), Ok(vec![value.clone()]));
            assert_eq!(
                program.interpret(vec![value.clone(), dimension(3)]),
                Err(ProgramError::Type(TypeError::custom(DimensionError::InputDimensionMismatch {
                    dimension: "n".to_owned(),
                    expected: 3,
                    actual: 2,
                }))),
            );
        }
    }

    #[test]
    fn test_cast_interpretation_refined_representation_witness_identity() {
        let bounds = DimensionBounds::positive(Some(8)).unwrap();
        let n = DimensionType::new("n", bounds);
        let m = DimensionType::new("m", bounds);
        let plain = ArrayType::new_static(DataType::F32, [2]);
        let source = plain.clone().with_layout(Layout::Strided(StridedLayout::new(vec![8])));
        let target = plain.clone().with_layout(Layout::Strided(StridedLayout::new(vec![12])));
        let declared = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()]));
        let operation =
            CastOperation::<ArrayIrType>::checked(target.clone().into(), vec![declared.into(), m.into()]).unwrap();
        // The witness's actual n instantiates declared m. It does not bind the source declaration's separate n,
        // whose extent is already 2, so replay must not acquire the witness's nominal identity.
        let (_, replayed) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |values: Vec<Tracer<_>>| operation.interpret(&values[0].domain(), &EmptyRegionDriver, &values),
            vec![ArrayIrType::from(source.clone()), ArrayIrType::from(n.clone())],
        )
        .unwrap();
        assert_eq!(
            replayed.to_string(),
            formatdoc! {"
                lambda %0:{source}, %1:{n} .
                let %2:{target} = cast [
                    type={target},
                    inputs=[{source}, {n}],
                ] %0 %1
                in (%2)"},
        );
        let value = Array::from_elements(source, &[1f32, 2f32]).unwrap();
        assert_eq!(
            replayed.interpret(vec![ArrayIrValue::Array(value), DimensionValue::constant(3).unwrap().into()]),
            Ok(vec![ArrayIrValue::Array(Array::from_elements(target, &[1f32, 2f32]).unwrap())]),
        );
    }

    #[test]
    fn test_cast_partial_evaluation() {
        let plain = ArrayType::new_static(DataType::F32, [2]);
        let laid_out = plain.clone().with_layout(Layout::Strided(StridedLayout::new(vec![8])));
        check_operation_partial_evaluation!(
            operation = CastOperation::checked(laid_out.clone(), vec![plain]).unwrap(),
            inputs = [Array::vector(vec![1f32, 2f32]).unwrap()],
            expected = Array::from_elements(laid_out, &[1f32, 2f32]).unwrap(),
        );

        // A checked cast of a known value checks its relationship while partial evaluation folds it, and a checked
        // cast of an unknown value stays checked in the residual program.
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let symbolic = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()]));
        let exact = ArrayType::new_static(DataType::F32, [2]);
        let vector = |extent: usize| Array::vector(vec![1f32; extent]).unwrap();
        check_operation_partial_evaluation!(
            operation = CastOperation::checked(exact.clone(), vec![symbolic.clone()]).unwrap(),
            cases = [
                {
                    inputs = [(@known, vector(2))],
                    outputs = [(@known, vector(2))],
                    residual_instructions = 0,
                },
                {
                    inputs = [(@unknown(type = symbolic.clone(), replay = vector(2)))],
                    outputs = [(@residual, vector(2))],
                    residual_instructions = 1,
                },
            ],
        );
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |inputs: Vec<Tracer<_>>| Ok(vec![inputs[0].cast(exact.clone(), &[])?]),
            vec![symbolic.clone()],
        )
        .unwrap();
        assert_eq!(
            program.partially_evaluate(&[PartialValue::Known(vector(3))]).unwrap_err(),
            ProgramError::Type(TypeError::invalid("`cast` cannot convert type `f32[3]` to unrelated type `f32[2]`")),
        );
        let residual = program.partially_evaluate(&[PartialValue::Unknown(symbolic)]).unwrap();
        assert_eq!(
            residual.program().to_string(),
            indoc! {"
                lambda %0:f32[n] .
                let %1:f32[2] = cast [type=f32[2], inputs=[f32[n]], checked=true] %0
                in (%1)"},
        );
    }

    #[test]
    fn test_cast_partial_evaluation_declared_dimensions() {
        let bounds = DimensionBounds::positive(Some(5)).unwrap();
        let [a, b] = ["a", "b"].map(|name| DimensionType::new(name, bounds));
        let input = ArrayType::new(DataType::F32, Shape::new(vec![a.simplified_dimension(), b.simplified_dimension()]));
        let output =
            ArrayType::new(DataType::F32, Shape::new(vec![b.simplified_dimension(), a.simplified_dimension()]));
        let square = Array::matrix(2, 2, vec![1f32; 4]).unwrap();
        let rectangular = Array::matrix(2, 3, vec![1f32; 6]).unwrap();

        // All-known folding checks before removing the instruction, while unknown inputs keep the complete source
        // declaration in the residual.
        check_operation_partial_evaluation!(
            operation = CastOperation::checked(output.clone(), vec![input.clone()]).unwrap(),
            cases = [
                {
                    inputs = [(@known, square.clone())],
                    outputs = [(@known, square.clone())],
                    residual_instructions = 0,
                },
                {
                    inputs = [(@unknown(type = input.clone(), replay = square.clone()))],
                    outputs = [(@residual, square.clone())],
                    residual_instructions = 1,
                },
            ],
        );
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let value = builder.add_input(input.clone());
        let result = builder
            .add_instruction(
                CastOperation::checked(output, vec![input.clone()]).unwrap(),
                Vec::new(),
                vec![value],
                None,
            )
            .unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![result], vec![Placeholder], vec![Placeholder]).unwrap();
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let mismatch = ProgramError::Type(TypeError::custom(DimensionError::InputDimensionMismatch {
            dimension: "b".to_owned(),
            expected: 3,
            actual: 2,
        }));
        assert_eq!(program.partially_evaluate(&[PartialValue::Known(rectangular.clone())]).unwrap_err(), mismatch);
        let residual = program.partially_evaluate(&[PartialValue::Unknown(input.clone())]).unwrap();
        assert_eq!(
            residual.program().to_string(),
            indoc! {"
                lambda %0:f32[a, b] .
                let %1:f32[b, a] = cast [type=f32[b, a], inputs=[f32[a, b]], checked=true] %0
                in (%1)"},
        );
        assert_eq!(residual.interpret(&context, &[rectangular.clone()]), Err(mismatch.clone()));

        // A fully proven identity also folds when effect folding is disabled. Contradictory concrete shapes must still
        // fail before the operation can be omitted.
        let deferred = program
            .entry_region_ref()
            .partially_evaluate_in_context(&context, &[PartialValue::Known(square.clone())], false)
            .unwrap();
        assert_eq!(deferred.program().to_string(), "lambda  .\nin ()");
        assert_eq!(deferred.interpret(&context, &[]), Ok(vec![square]));
        assert_eq!(
            program
                .entry_region_ref()
                .partially_evaluate_in_context(&context, &[PartialValue::Known(rectangular)], false)
                .unwrap_err(),
            mismatch,
        );

        // Known inputs in an outer trace are still symbolic. Folding into that trace must preserve the checked
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
        let vector = |extent: usize| Array::vector(vec![1f32; extent]).unwrap();

        // Known inputs with exact shapes prove the widening and fold it. An unknown witness keeps the checked cast in
        // the residual, whereas a known witness proves the relationship for an unknown source, which normalizes the
        // residual cast into an unchecked one.
        check_operation_partial_evaluation!(
            operation = CastOperation::checked(symbolic.clone(), vec![exact.clone(), symbolic.clone()]).unwrap(),
            cases = [
                {
                    inputs = [(@known, vector(2)), (@known, vector(2))],
                    outputs = [(@known, vector(2))],
                    residual_instructions = 0,
                },
                {
                    inputs = [(@known, vector(2)), (@unknown(type = symbolic.clone(), replay = vector(2)))],
                    outputs = [(@residual, vector(2))],
                    residual_instructions = 1,
                },
                {
                    inputs = [(@unknown(type = exact.clone(), replay = vector(2))), (@known, vector(2))],
                    outputs = [(@residual, vector(2))],
                    residual_instructions = 1,
                },
            ],
        );
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |(value, witness)| value.cast(symbolic.clone(), &[witness]),
            (exact.clone(), symbolic.clone()),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2], %1:f32[n] .
                let %2:f32[n] = cast [type=f32[n], inputs=[f32[2], f32[n]], checked=true] %0 %1
                in (%2)"},
        );
        let unknown_witness = program
            .to_flat_program()
            .partially_evaluate(&[PartialValue::Known(vector(2)), PartialValue::Unknown(symbolic.clone())])
            .unwrap();
        assert_eq!(
            unknown_witness.program().to_string(),
            indoc! {"
                lambda %0:f32[n], %1:f32[2] .
                let %2:f32[n] = cast [type=f32[n], inputs=[f32[2], f32[n]], checked=true] %1 %0
                in (%2)"},
        );
        let known_witness = program
            .to_flat_program()
            .partially_evaluate(&[PartialValue::Unknown(exact.clone()), PartialValue::Known(vector(2))])
            .unwrap();
        assert_eq!(
            known_witness.program().to_string(),
            indoc! {"
                lambda %0:f32[2], %1:f32[2] .
                let %2:f32[2] = cast [type=f32[n], inputs=[f32[2], f32[n]]] %0 %1
                in (%2)"},
        );

        // Known inputs with exact shapes prove the widening, so partial evaluation normalizes the checked cast and
        // folds it even when effect folding is disabled; contradictory known shapes are still rejected.
        let context = EagerContext::new();
        let deferred = program
            .entry_region_ref()
            .partially_evaluate_in_context(
                &context,
                &[PartialValue::Known(vector(2)), PartialValue::Known(vector(2))],
                false,
            )
            .unwrap();
        assert_eq!(deferred.program().to_string(), "lambda  .\nin ()");
        assert_eq!(deferred.interpret(&context, &[]), Ok(vec![vector(2)]));
        let mismatch = Err(ProgramError::Type(TypeError::custom(DimensionError::InputDimensionMismatch {
            dimension: "n".to_owned(),
            expected: 3,
            actual: 2,
        })));
        assert_eq!(
            program
                .entry_region_ref()
                .partially_evaluate_in_context(
                    &context,
                    &[PartialValue::Known(vector(2)), PartialValue::Known(vector(3))],
                    false,
                )
                .map(|_| ()),
            mismatch,
        );

        // An unchecked cast of contradictory known inputs is rejected as well, since folding checks the shapes.
        let (_, unchecked) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |(value, witness)| value.cast_unchecked(symbolic.clone(), &[witness]),
            (exact, symbolic.clone()),
        )
        .unwrap();
        assert_eq!(
            unchecked
                .to_flat_program()
                .partially_evaluate(&[PartialValue::Known(vector(2)), PartialValue::Known(vector(3))])
                .map(|_| ()),
            mismatch,
        );
    }

    #[test]
    fn test_cast_partial_evaluation_layout_and_sharding() {
        // Known inputs fold a representation change whether or not they already carry the requested sharding, also
        // when effect folding is disabled, because the proven relationship normalizes the explicitly checked cast
        // into the pure unchecked one. The residual of an unknown input is normalized the same way.
        let plain = ArrayType::new_static(DataType::F32, [2]);
        let input = plain.clone().with_layout(Layout::Strided(StridedLayout::new(vec![4])));
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding = Sharding::new(mesh, vec![ShardingDimension::replicated()]).unwrap();
        let output = plain.clone().with_sharding(sharding.clone()).unwrap();
        let expected_value = Array::from_elements(output.clone(), &[1f32, 2f32]).unwrap();
        let unannotated = Array::from_elements(input.clone(), &[1f32, 2f32]).unwrap();
        let actual = Array::from_elements(input.clone().with_sharding(sharding).unwrap(), &[1f32, 2f32]).unwrap();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let value = builder.add_input(input.clone());
        let checked = builder
            .add_instruction(
                CastOperation::checked(output.clone(), vec![input.clone()]).unwrap(),
                Vec::new(),
                vec![value],
                None,
            )
            .unwrap()[0];
        let unchecked = builder
            .add_instruction(
                CastOperation::unchecked(output, vec![input.clone()]).unwrap(),
                Vec::new(),
                vec![value],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![checked, unchecked], vec![Placeholder], vec![Placeholder; 2])
            .unwrap();
        let context = EagerContext::new();
        let known = [PartialValue::Known(actual.clone())];
        let unannotated_inputs = [PartialValue::Known(unannotated.clone())];
        let converted = program.partially_evaluate(&unannotated_inputs).unwrap();
        assert_eq!(converted.program().to_string(), "lambda  .\nin ()");
        assert_eq!(converted.interpret(&context, &[]), Ok(vec![expected_value.clone(); 2]));
        let folded = program.partially_evaluate(&known).unwrap();
        assert_eq!(folded.program().to_string(), "lambda  .\nin ()");
        assert_eq!(folded.interpret(&context, &[]), Ok(vec![expected_value.clone(); 2]));
        let residual = program.partially_evaluate(&[PartialValue::Unknown(input)]).unwrap();
        assert_eq!(
            residual.program().to_string(),
            indoc! {"
                lambda %0:f32[2][layout=strided{4}] .
                let %1:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}] = cast [
                    type=f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}],
                    inputs=[f32[2][layout=strided{4}]],
                ] %0
                    %2:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}] = cast [
                        type=f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}],
                        inputs=[f32[2][layout=strided{4}]],
                    ] %0
                in (%1, %2)"},
        );
        assert_eq!(residual.interpret(&context, &[actual]), Ok(vec![expected_value.clone(); 2]));
        assert_eq!(residual.interpret(&context, &[unannotated]), Ok(vec![expected_value.clone(); 2]));
        let deferred_missing = program
            .entry_region_ref()
            .partially_evaluate_in_context(&context, &unannotated_inputs, false)
            .unwrap();
        assert_eq!(deferred_missing.program().to_string(), "lambda  .\nin ()");
        assert_eq!(deferred_missing.interpret(&context, &[]), Ok(vec![expected_value.clone(); 2]));
        let deferred = program.entry_region_ref().partially_evaluate_in_context(&context, &known, false).unwrap();
        assert_eq!(deferred.program().to_string(), "lambda  .\nin ()");
        assert_eq!(deferred.interpret(&context, &[]), Ok(vec![expected_value; 2]));
    }

    #[test]
    fn test_cast_partial_evaluation_unspecified_layout_and_sharding() {
        let plain = ArrayType::new_static(DataType::U8, [2]);
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let output = plain
            .clone()
            .with_layout(Layout::Strided(StridedLayout::new(vec![2])))
            .with_sharding(Sharding::replicated(mesh, 1))
            .unwrap();
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |value| value.cast(output.clone(), &[]),
            plain.clone(),
        )
        .unwrap();
        let expected = indoc! {"
            lambda %0:u8[2] .
            let %1:u8[2][layout=strided{2}][sharding={mesh<['x'=1:explicit]>, [{}]}] = cast [
                type=u8[2][layout=strided{2}][sharding={mesh<['x'=1:explicit]>, [{}]}],
                inputs=[u8[2]],
            ] %0
            in (%1)"};
        assert_eq!(program.to_string(), expected);
        let input = Array::vector(vec![10u8, 20]).unwrap();
        let expected_value = Array::from_elements(output, &[10u8, 20]).unwrap();
        assert_eq!(program.interpret(input.clone()), Ok(expected_value.clone()));
        let context = EagerContext::new();
        let known = [PartialValue::Known(input.clone())];
        let folded = program.to_flat_program().partially_evaluate(&known).unwrap();
        assert_eq!(folded.program().to_string(), "lambda  .\nin ()");
        assert_eq!(folded.interpret(&context, &[]), Ok(vec![expected_value.clone()]));
        let residual = program.to_flat_program().partially_evaluate(&[PartialValue::Unknown(plain)]).unwrap();
        assert_eq!(residual.program().to_string(), expected);
        assert_eq!(residual.interpret(&context, &[input]), Ok(vec![expected_value.clone()]));
        let deferred = program.entry_region_ref().partially_evaluate_in_context(&context, &known, false).unwrap();
        assert_eq!(deferred.program().to_string(), "lambda  .\nin ()");
        assert_eq!(deferred.interpret(&context, &[]), Ok(vec![expected_value]));
    }

    #[test]
    fn test_cast_batching() {
        let scalar = ArrayType::scalar(DataType::F32);
        check_operation_batching!(
            @exact,
            operation = CastOperation::unchecked(scalar.clone(), vec![scalar]).unwrap(),
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), Array::vector(vec![1f32, 2f32]).unwrap())],
                outputs = [(@mapped(axis = 0), Array::vector(vec![1f32, 2f32]).unwrap())],
            }],
        );
        // A mapped value is retyped with the batch axis inserted into the output type, and keeps that axis, while a
        // replicated witness keeps its declared type. Batching preserves the mode of the cast.
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let batch = DimensionType::new("batch", DimensionBounds::new(0, Some(5)).unwrap());
        let symbolic = ArrayIrType::from(ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()])));
        let exact: ArrayIrType = ArrayType::new_static(DataType::F32, [2]).into();
        for (operation, mode) in [
            (CastOperation::unchecked(exact.clone(), vec![symbolic.clone(), symbolic.clone()]).unwrap(), ""),
            (
                CastOperation::checked(exact.clone(), vec![symbolic.clone(), symbolic.clone()]).unwrap(),
                ", checked=true",
            ),
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
                    let %3:f32[batch, 2] = cast [type=f32[batch, 2], inputs=[f32[batch, n], f32[n]]{mode}] %1 %2
                    in (%3)"},
            );
        }
    }

    #[test]
    fn test_cast_batching_mapped_witness() {
        // A witness mapped on another physical axis is lifted in place, so the output follows the source's batch
        // axis while the witness declaration gains the batch axis where its value carries it.
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let batch = DimensionType::new("batch", DimensionBounds::new(0, Some(5)).unwrap());
        let symbolic = ArrayIrType::from(ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()])));
        let exact: ArrayIrType = ArrayType::new_static(DataType::F32, [2]).into();
        let (_, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |(extent, mapped, witness)| {
                let batching = BatchingContext::new(extent.domain(), extent);
                let outputs = CastOperation::unchecked(exact.clone(), vec![symbolic.clone(), symbolic.clone()])
                    .unwrap()
                    .batch(
                        &batching,
                        &EmptyRegionDriver,
                        &[
                            ArrayIrBatch::new(mapped, BatchAxis::new(0))?,
                            ArrayIrBatch::new(witness, BatchAxis::new(1))?,
                        ],
                    )?
                    .into_parts()
                    .0;
                assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
                Ok(outputs[0].value().clone())
            },
            (
                ArrayIrType::Dimension(batch.clone()),
                ArrayIrType::Array(ArrayType::new(
                    DataType::F32,
                    Shape::new(vec![batch.simplified_dimension(), n.simplified_dimension()]),
                )),
                ArrayIrType::Array(ArrayType::new(
                    DataType::F32,
                    Shape::new(vec![n.simplified_dimension(), batch.simplified_dimension()]),
                )),
            ),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:dimension<batch ∈ [0, 5)>, %1:f32[batch, n], %2:f32[n, batch] .
                let %3:f32[batch, 2] = cast [type=f32[batch, 2], inputs=[f32[batch, n], f32[n, batch]]] %1 %2
                in (%3)"},
        );
    }

    #[test]
    fn test_cast_batching_replicated_source_mapped_witness() {
        // A replicated source stays replicated, and the mapped witness only contributes its per-item extents.
        let exact = ArrayType::new_static(DataType::F32, [2]);
        let n = DimensionType::new("n", DimensionBounds::positive(Some(5)).unwrap());
        let symbolic = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()]));
        check_operation_batching!(
            @exact,
            operation = CastOperation::checked(exact.clone(), vec![symbolic.clone(), symbolic.clone()]).unwrap(),
            axis_size = 3,
            cases = [{
                inputs = [
                    (@replicated, Array::vector(vec![1f32, 2f32]).unwrap()),
                    (@mapped(axis = 0), Array::matrix(3, 2, vec![0f32; 6]).unwrap()),
                ],
                outputs = [(@replicated, Array::vector(vec![1f32, 2f32]).unwrap())],
            }],
        );
        check_operation_batching!(
            @exact,
            operation = CastOperation::unchecked(exact, vec![symbolic.clone(), symbolic]).unwrap(),
            axis_size = 3,
            cases = [{
                inputs = [
                    (@replicated, Array::vector(vec![1f32, 2f32]).unwrap()),
                    (@mapped(axis = 0), Array::matrix(3, 2, vec![0f32; 6]).unwrap()),
                ],
                outputs = [(@replicated, Array::vector(vec![1f32, 2f32]).unwrap())],
            }],
        );
    }

    #[test]
    fn test_cast_batching_witnessed_tracer() {
        // A batching tracer casts through its context with its witnesses, so a mapped source narrowed next to a
        // replicated witness is checked per item and keeps its batch axis.
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let symbolic = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()]));
        let exact = ArrayType::new_static(DataType::F32, [2]);
        let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 3);
        let value = Array::matrix(3, 2, vec![1f32; 6]).unwrap();
        let source = BatchingTracer::new(context.clone(), ArrayBatch::new(value.clone(), BatchAxis::new(0)).unwrap());
        let witness = BatchingTracer::new(context, ArrayBatch::replicated(Array::vector(vec![0f32; 2]).unwrap()));
        assert_eq!(source.r#type().as_ref(), &exact);
        let output = source.cast(exact.clone(), &[witness.clone()]).unwrap().into_batch();
        assert_eq!(output, ArrayBatch::new(value.clone(), BatchAxis::new(0)).unwrap());
        let output = source.cast_unchecked(exact.clone(), &[witness.clone()]).unwrap().into_batch();
        assert_eq!(output, ArrayBatch::new(value, BatchAxis::new(0)).unwrap());
        let error = source.cast(symbolic, &[witness]).unwrap_err();
        assert_eq!(
            error.downcast_custom::<BatchingError>(),
            Some(&BatchingError::Type(TypeError::invalid(
                "`cast` output identity `n` is not carried by a declared input",
            ))),
        );
    }

    #[test]
    fn test_cast_batching_identity_ragged() {
        // A one-input identity keeps its original ragged carrier, including its storage, in both modes and universes.
        let variable = DimensionVariable::new("length", DimensionBounds::new(0, Some(4)).unwrap());
        for lengths in [vec![], vec![0], vec![1, 3]] {
            let size = lengths.len();
            let value = Array::matrix(size, 3, vec![1f32; size * 3]).unwrap();
            let extents = Array::vector(lengths).unwrap();
            let input = ragged_array_batch(value.clone(), extents.clone(), &variable, 1).unwrap();
            let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), size);
            let tracer = BatchingTracer::new(context, input.clone());
            let output_type = tracer.r#type().into_owned();
            let output = tracer.cast(output_type.clone(), &[]).unwrap().into_batch();
            assert_eq!(output, input);
            assert!(Arc::ptr_eq(output.value().shared_storage_bytes(), value.shared_storage_bytes()));
            let output = tracer.cast_unchecked(output_type, &[]).unwrap().into_batch();
            assert_eq!(output, input);
            assert!(Arc::ptr_eq(output.value().shared_storage_bytes(), value.shared_storage_bytes()));
            let composite =
                ragged_batch(ArrayIrValue::Array(value.clone()), ArrayIrValue::Array(extents), &variable, 1).unwrap();
            let context = BatchingContext::new(
                EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new(),
                ArrayIrValue::Dimension(DimensionValue::constant(size).unwrap()),
            );
            let composite_tracer = BatchingTracer::new(context, composite.clone());
            let output_type = composite_tracer.r#type().into_owned();
            assert_eq!(composite_tracer.cast(output_type.clone(), &[]).unwrap().into_batch(), composite);
            assert_eq!(composite_tracer.cast_unchecked(output_type, &[]).unwrap().into_batch(), composite);
        }
    }

    #[test]
    fn test_cast_batching_identity_ragged_staging() {
        let variable = DimensionVariable::new("length", DimensionBounds::new(0, Some(4)).unwrap());
        for checked in [true, false] {
            let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
                |(value, extents)| {
                    let context = BatchingContext::new(value.domain(), 2);
                    let input = ragged_array_batch(value, extents.clone(), &variable, 1)?;
                    let tracer = BatchingTracer::new(context, input.clone());
                    let output_type = tracer.r#type().into_owned();
                    let output = if checked {
                        tracer.cast(output_type, &[])?
                    } else {
                        tracer.cast_unchecked(output_type, &[])?
                    }
                    .into_batch();
                    assert_eq!(output, input);
                    Ok(vec![output.into_value(), extents])
                },
                (ArrayType::new_static(DataType::F32, [2, 3]), ArrayType::new_static(DataType::I32, [2])),
            )
            .unwrap();
            assert_eq!(
                program.to_string(),
                indoc! {"
                    lambda %0:f32[2, 3], %1:i32[2] .
                    in (%0, %1)"},
            );
            let value = Array::matrix(2, 3, vec![1f32; 6]).unwrap();
            let extents = Array::vector(vec![1i32, 3]).unwrap();
            assert_eq!(program.interpret((value.clone(), extents.clone())), Ok(vec![value, extents]));
        }
    }

    #[test]
    fn test_cast_batching_identity_validates_contract() {
        let variable = DimensionVariable::new("length", DimensionBounds::new(0, Some(4)).unwrap());
        let logical = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(variable.clone())]));
        let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 2);
        let input = ragged_array_batch(
            Array::matrix(2, 3, vec![1f32; 6]).unwrap(),
            Array::vector(vec![1i32, 3]).unwrap(),
            &variable,
            1,
        )
        .unwrap();
        let other = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Dynamic(DimensionVariable::new(
                "other",
                DimensionBounds::new(0, Some(4)).unwrap(),
            ))]),
        );
        let exact = ArrayType::new_static(DataType::F32, [3]);
        let ragged = Err(BatchingError::UnsupportedOperation {
            message: "cannot batch ragged array type `f32[length]`".to_owned(),
        });
        for constructor in [CastOperation::checked, CastOperation::unchecked] {
            // A different declaration or a narrowing is not an identity of this carrier's logical type. A foreign
            // declaration does not describe the carrier at all, and a narrowing would have to check the per-item
            // extents, which lifting the declarations cannot express, so batching rejects the ragged carrier.
            assert_eq!(
                constructor(other.clone(), vec![other.clone()]).unwrap().batch(
                    &context,
                    &EmptyRegionDriver,
                    &[input.clone()]
                ),
                Err(BatchingError::Type(TypeError::invalid(
                    "type f32[length] does not refine declared type f32[other]"
                ))),
            );
            assert_eq!(
                constructor(exact.clone(), vec![logical.clone()]).unwrap().batch(
                    &context,
                    &EmptyRegionDriver,
                    &[input.clone()]
                ),
                ragged,
            );
            // The witness would require every item to have length 3. Matching source and output declarations must not
            // erase that additional relationship, even though the first input otherwise looks like an identity.
            let witness = ArrayBatch::replicated(Array::vector(vec![1f32; 3]).unwrap());
            assert_eq!(
                constructor(logical.clone(), vec![logical.clone(), logical.clone()]).unwrap().batch(
                    &context,
                    &EmptyRegionDriver,
                    &[input.clone(), witness],
                ),
                ragged,
            );
        }

        // The composite policy rejects a non-identity cast over a ragged array carrier in the same way.
        let context = BatchingContext::new(
            EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new(),
            ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()),
        );
        let input = ragged_batch(
            ArrayIrValue::Array(Array::matrix(2, 3, vec![1f32; 6]).unwrap()),
            ArrayIrValue::Array(Array::vector(vec![1i32, 3]).unwrap()),
            &variable,
            1,
        )
        .unwrap();
        for constructor in [CastOperation::checked, CastOperation::unchecked] {
            assert_eq!(
                constructor(exact.clone().into(), vec![logical.clone().into()]).unwrap().batch(
                    &context,
                    &EmptyRegionDriver,
                    std::slice::from_ref(&input),
                ),
                Err(BatchingError::UnsupportedOperation {
                    message: "cannot batch ragged array type `f32[length]`".to_owned(),
                }),
            );
        }
    }

    #[test]
    fn test_cast_batching_replicated_ragged_source() {
        // A replicated carrier holds the same ragged item for every batch item, here with packed storage `[1, 2, 99]`
        // and live extent 2. Narrowing it to `f32[3]` would read the padding as data, so the composite policy rejects
        // it like a mapped ragged carrier rather than returning the padded storage.
        let variable = DimensionVariable::new("length", DimensionBounds::new(0, Some(4)).unwrap());
        let logical = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(variable.clone())]));
        let context = BatchingContext::new(
            EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new(),
            ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()),
        );
        let input = ArrayIrBatch::replicated(ArrayIrValue::Array(Array::vector(vec![1f32, 2f32, 99f32]).unwrap()))
            .with_ragged_axes(vec![RaggedAxis::new(
                0,
                ArrayIrValue::Array(Array::scalar(2i32).unwrap()),
                variable,
                vec![],
            )])
            .unwrap();
        for constructor in [CastOperation::checked, CastOperation::unchecked] {
            assert_eq!(
                constructor(ArrayType::new_static(DataType::F32, [3]).into(), vec![logical.clone().into()])
                    .unwrap()
                    .batch(&context, &EmptyRegionDriver, std::slice::from_ref(&input))
                    .map(|outputs| outputs.into_parts().0),
                Err(BatchingError::UnsupportedOperation {
                    message: "cannot batch ragged array type `f32[length]`".to_owned(),
                }),
            );
        }
    }

    #[test]
    fn test_cast_batching_layout() {
        let plain = ArrayType::new_static(DataType::F32, [2]);
        let declared = plain.clone().with_layout(Layout::Strided(StridedLayout::new(vec![4])));
        for checked in [true, false] {
            let (_, source) = TracingContext::<Array, ArrayOperation<Array>>::trace(
                |value| if checked { value.cast(plain.clone(), &[]) } else { value.cast_unchecked(plain.clone(), &[]) },
                declared.clone(),
            )
            .unwrap();
            let expected_source = indoc! {"
                lambda %0:f32[2][layout=strided{4}] .
                let %1:f32[2] = cast [type=f32[2], inputs=[f32[2][layout=strided{4}]]] %0
                in (%1)"};
            assert_eq!(source.to_string(), expected_source);
            let composite_declared: ArrayIrType = declared.clone().into();
            let (_, composite) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
                |value| {
                    if checked {
                        value.cast(plain.clone().into(), &[])
                    } else {
                        value.cast_unchecked(plain.clone().into(), &[])
                    }
                },
                composite_declared,
            )
            .unwrap();
            assert_eq!(composite.to_string(), expected_source);
            // Rank-changing packing clears layout. The complete original cast remains valid, while the packed
            // declaration and input agree on the new representation, including empty and singleton batches.
            for extent in [0, 1, 3] {
                let (program, axes) = source
                    .to_flat_program()
                    .batched(
                        extent,
                        ShardingDimension::replicated(),
                        &[BatchAxis::new(0)],
                        ProgramBatchingOutputAxesPolicy::Natural,
                    )
                    .unwrap()
                    .into_parts();
                assert_eq!(axes, vec![BatchAxis::new(0)]);
                assert_eq!(
                    program.to_string(),
                    formatdoc! {"
                        lambda %0:f32[{extent}, 2] .
                        in (%0)"},
                );
                let value = Array::matrix(extent, 2, vec![1f32; extent * 2]).unwrap();
                assert_eq!(program.interpret(vec![value.clone()]), Ok(vec![value.clone()]));
                let (program, axes) = composite
                    .to_flat_program()
                    .batched_with_threaded_extent(
                        DimensionType::from(Dimension::Static(extent)),
                        ShardingDimension::replicated(),
                        &[BatchAxis::new(0)],
                        ProgramBatchingOutputAxesPolicy::Natural,
                    )
                    .unwrap()
                    .into_parts();
                assert_eq!(axes, vec![BatchAxis::new(0)]);
                assert_eq!(
                    program.to_string(),
                    formatdoc! {"
                        lambda %0:dimension<{extent}>, %1:f32[{extent}, 2] .
                        in (%0, %1)"},
                );
                let dimension = ArrayIrValue::Dimension(DimensionValue::constant(extent).unwrap());
                let value = ArrayIrValue::Array(value);
                assert_eq!(program.interpret(vec![dimension.clone(), value.clone()]), Ok(vec![dimension, value]));
            }
        }
    }

    #[test]
    fn test_cast_batching_validates_declarations() {
        let plain = ArrayType::new_static(DataType::F32, [2]);
        let declared = plain.clone().with_layout(Layout::Strided(StridedLayout::new(vec![4])));
        let conflicting =
            ArrayType::new_static(DataType::F64, [2]).with_layout(Layout::Strided(StridedLayout::new(vec![8])));
        let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 3);
        let input = ArrayBatch::new(Array::matrix(3, 2, vec![1f32; 6]).unwrap(), BatchAxis::new(0)).unwrap();
        for constructor in [CastOperation::checked, CastOperation::unchecked] {
            // Lifting may erase layouts, but must still reject an invalid element-type change in the original cast.
            let operation = constructor(conflicting.clone(), vec![declared.clone()]).unwrap();
            assert_eq!(
                operation.batch(&context, &EmptyRegionDriver, &[input.clone()]).map(|_| ()),
                Err(TypeError::invalid(
                    "`cast` cannot convert type `f32[2][layout=strided{4}]` \
                     to unrelated type `f64[2][layout=strided{8}]`",
                )
                .into()),
            );
            let operation = constructor(plain.clone(), vec![declared.clone()]).unwrap();
            assert_eq!(
                operation.batch(&context, &EmptyRegionDriver, &[]).map(|_| ()),
                Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into()),
            );
            let wrong = ArrayBatch::new(Array::matrix(3, 4, vec![1f32; 12]).unwrap(), BatchAxis::new(0)).unwrap();
            assert_eq!(
                operation.batch(&context, &EmptyRegionDriver, &[wrong]).map(|_| ()),
                Err(ProgramError::Type(TypeError::invalid("dimension 4 does not refine declared dimension 2")).into()),
            );
        }
    }

    #[test]
    fn test_cast_batching_mixed_dimensions() {
        // Mixed item dimensions require both item extents to be 3, independently of the ambient batch extent.
        // Source, output and witness declarations keep their ambient names.
        let bounds = DimensionBounds::non_negative(Some(8)).unwrap();
        let [n, a, b] = ["n", "a", "b"].map(|name| DimensionType::new(name, bounds));
        let witness = ArrayIrType::from(ArrayType::new(DataType::F32, Shape::new(vec![a.simplified_dimension()])));
        let output = ArrayIrType::from(ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Static(3), a.simplified_dimension()]),
        ));
        let (_, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |(extent, mapped, witness_value)| {
                let batching = BatchingContext::new(extent.domain(), extent);
                let outputs = CastOperation::unchecked(
                    output.clone(),
                    vec![
                        ArrayType::new(DataType::F32, Shape::new(vec![b.simplified_dimension(), Dimension::Static(3)]))
                            .into(),
                        witness.clone(),
                    ],
                )
                .unwrap()
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
            program.to_string(),
            indoc! {"
                lambda %0:dimension<n ∈ [0, 8)>, %1:f32[n, b, 3], %2:f32[a] .
                let %3:f32[n, 3, a] = cast [type=f32[n, 3, a], inputs=[f32[n, b, 3], f32[a]]] %1 %2
                in (%3)"},
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
            assert_eq!(program.interpret((dimension.clone(), mapped.clone(), witness(3))), Ok(mapped.clone()));
            assert_eq!(
                program.interpret((dimension, mapped, witness(4))),
                Err(ProgramError::Type(TypeError::custom(DimensionError::InputDimensionMismatch {
                    dimension: "a".to_owned(),
                    expected: 4,
                    actual: 3,
                }))),
            );
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
                    let outputs = CastOperation::unchecked(
                        per_item.clone().into(),
                        vec![per_item.clone().into(), per_item.clone().into()],
                    )
                    .unwrap()
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
                        inputs=[{expected_type}, {expected_type}],
                    ] %1 %2
                    in (%3)"},
            );
        }
    }

    #[test]
    fn test_cast_batching_manual_variation() {
        // Batching over a manual mesh axis adds that axis's variation to every mapped value. An identity cast keeps
        // the item's own variation, an unspecified output placement still receives the batch axis's variation, and
        // existing variation on another axis cannot be dropped by the original cast.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let plain = ArrayType::new_static(DataType::F32, [2]).with_sharding(Sharding::replicated(mesh, 1)).unwrap();
        let varying = plain
            .clone()
            .with_sharding(plain.sharding().unwrap().clone().with_varying_manual_axes(["y"]).unwrap())
            .unwrap();
        let unannotated = ArrayType::new_static(DataType::F32, [2]);
        for (item, output) in
            [(plain.clone(), plain.clone()), (plain.clone(), unannotated.clone()), (varying.clone(), varying.clone())]
        {
            let packed = item.batched(0, Dimension::Static(4), ShardingDimension::sharded(["x"])).unwrap();
            let expected_source = if item == output {
                formatdoc! {"
                    lambda %0:{item} .
                    in (%0)"}
            } else {
                formatdoc! {"
                    lambda %0:{item} .
                    let %1:{output} = cast [
                        type={output},
                        inputs=[{item}],
                    ] %0
                    in (%1)"}
            };
            for checked in [true, false] {
                let (_, source) = TracingContext::<Array, ArrayOperation<Array>>::trace(
                    |value| {
                        if checked {
                            value.cast(output.clone(), &[])
                        } else {
                            value.cast_unchecked(output.clone(), &[])
                        }
                    },
                    item.clone(),
                )
                .unwrap();
                assert_eq!(source.to_string(), expected_source);
                let (program, axes) = source
                    .to_flat_program()
                    .batched(
                        4,
                        ShardingDimension::sharded(["x"]),
                        &[BatchAxis::new(0)],
                        ProgramBatchingOutputAxesPolicy::Natural,
                    )
                    .unwrap()
                    .into_parts();
                assert_eq!(axes, vec![BatchAxis::new(0)]);
                assert_eq!(
                    program.to_string(),
                    formatdoc! {"
                        lambda %0:{packed} .
                        in (%0)"},
                );
                let value = Array::from_elements(packed.clone(), &[1f32; 8]).unwrap();
                assert_eq!(program.interpret(vec![value.clone()]), Ok(vec![value]));
            }
        }

        // Only x belongs to this mapped axis. An unexpected change on y must remain an input-contract error.
        let packed = plain.batched(0, Dimension::Static(4), ShardingDimension::sharded(["x"])).unwrap();
        let wrong_type = packed
            .clone()
            .with_sharding(packed.sharding().unwrap().clone().with_varying_manual_axes(["x", "y"]).unwrap())
            .unwrap();
        let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 4)
            .with_axis_sharding(ShardingDimension::sharded(["x"]));
        let input =
            ArrayBatch::new(Array::from_elements(wrong_type.clone(), &[1f32; 8]).unwrap(), BatchAxis::new(0)).unwrap();
        for declared in [plain, unannotated] {
            assert_eq!(
                CastOperation::checked(declared.clone(), vec![declared])
                    .unwrap()
                    .batch(&context, &EmptyRegionDriver, &[input.clone()])
                    .map(|_| ()),
                Err(ProgramError::Type(TypeError::invalid(format!(
                    "type {wrong_type} does not refine declared type {packed}"
                )))
                .into()),
            );
        }
    }

    #[test]
    fn test_cast_differentiation() {
        let scalar = ArrayType::scalar(DataType::F64);
        check_operation_differentiation!(
            @approx(step = 0.125, epsilon = 0.0),
            operation = CastOperation::unchecked(scalar.clone(), vec![scalar.clone(), scalar]).unwrap(),
            cases = [{
                primals = [Array::scalar(2f64).unwrap(), Array::scalar(4f64).unwrap()],
                tangents = [Array::scalar(3f64).unwrap(), Array::scalar(5f64).unwrap()],
                primal_outputs = [Array::scalar(2f64).unwrap()],
                tangent_outputs = [Array::scalar(3f64).unwrap()],
                jvp = indoc! {"
                    lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[] .
                    let %4:f64[] = cast [type=f64[], inputs=[f64[], f64[]]] %0 %1
                        %5:f64[] = cast [type=f64[], inputs=[f64[], f64[]]] %2 %4
                    in (%4, %5)"},
            }],
        );

        // A source-typed identity retains both nominal shape and the checked result dependency in one residual. The
        // tangent conversion is unchecked because that residual carries the primal check.
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
                let %1:f32[2] = cast [type=f32[2], inputs=[f32[n]], checked=true] %0
                in (%1)"},
        );
        assert_eq!(
            program.jvp().unwrap().to_string(),
            indoc! {"
                lambda %0:f32[n], %1:f32[n] .
                let %2:f32[2] = cast [type=f32[2], inputs=[f32[n]], checked=true] %0
                    %3:f32[n] = cast [type=f32[n], inputs=[f32[n], f32[2]]] %0 %2
                    %4:f32[2] = cast [type=f32[2], inputs=[f32[n], f32[n]]] %1 %3
                in (%2, %4)"},
        );
        let linearization = program.linearize().unwrap();
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:f32[n] .
                let %1:f32[2] = cast [type=f32[2], inputs=[f32[n]], checked=true] %0
                in (%1, %0)"},
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f32[n], %1:f32[n] .
                let %2:f32[2] = cast [type=f32[2], inputs=[f32[n], f32[n]]] %0 %1
                in (%2)"},
        );
    }

    #[test]
    fn test_cast_differentiation_zero_tangent() {
        // A symbolic-zero source tangent stays symbolic: only the primal cast is bound, and witness tangents never
        // contribute, so differentiating with respect to the witness alone yields a zero output tangent.
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let symbolic = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()]));
        let exact = ArrayType::new_static(DataType::F32, [2]);
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let primal = context.input(exact.clone());
        let witness = context.input(symbolic.clone());
        let witness_tangent = context.input(symbolic.clone());
        let outputs = CastOperation::checked(symbolic.clone(), vec![exact.clone(), symbolic.clone()])
            .unwrap()
            .jvp(
                &DifferentiationContext::fused(context.clone()),
                &EmptyRegionDriver,
                &[
                    DifferentiationDual::new_with_zero_tangent(primal).unwrap(),
                    DifferentiationDual::new(witness, witness_tangent).unwrap(),
                ],
            )
            .unwrap();
        assert!(outputs[0].tangent().is_zero());
        assert_eq!(outputs[0].tangent().r#type().as_ref(), &symbolic);
        assert_eq!(context.builder().borrow().instructions().len(), 1);

        // At the program level, the zero output tangent of a static output is materialized at the boundary.
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |(value, witness)| value.cast(exact.clone(), &[witness]),
            (symbolic.clone(), symbolic),
        )
        .unwrap();
        let program = program.to_flat_program();
        assert_eq!(
            program.jvp_with_respect_to(&[1]).unwrap().to_string(),
            indoc! {"
                lambda %0:f32[n], %1:f32[n], %2:f32[n] .
                let %3:f32[2] = cast [type=f32[2], inputs=[f32[n], f32[n]], checked=true] %0 %1
                    %4:f32[2] = zero [type=f32[2]]
                in (%3, %4)"},
        );
        let linearization = program.linearize_with_respect_to(&[1]).unwrap();
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:f32[n], %1:f32[n] .
                let %2:f32[2] = cast [type=f32[2], inputs=[f32[n], f32[n]], checked=true] %0 %1
                in (%2)"},
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f32[n] .
                let %1:f32[2] = zero [type=f32[2]]
                in (%1)"},
        );
    }

    #[test]
    fn test_cast_differentiation_widening() {
        // Widening with a witness carries the witnessed name in the output tangent type, so the fused tangent only
        // depends on the checked result, and split linearization retains that result as its one residual.
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let symbolic = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()]));
        let exact = ArrayType::new_static(DataType::F32, [2]);
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |(value, witness)| value.cast(symbolic.clone(), &[witness]),
            (exact, symbolic.clone()),
        )
        .unwrap();
        let program = program.to_flat_program();
        assert_eq!(
            program.jvp().unwrap().to_string(),
            indoc! {"
                lambda %0:f32[2], %1:f32[n], %2:f32[2], %3:f32[n] .
                let %4:f32[n] = cast [type=f32[n], inputs=[f32[2], f32[n]], checked=true] %0 %1
                    %5:f32[n] = cast [type=f32[n], inputs=[f32[2], f32[n]]] %2 %4
                in (%4, %5)"},
        );
        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.residual_count(), 1);
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:f32[2], %1:f32[n] .
                let %2:f32[n] = cast [type=f32[n], inputs=[f32[2], f32[n]], checked=true] %0 %1
                in (%2, %2)"},
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f32[2], %1:f32[n], %2:f32[n] .
                let %3:f32[n] = cast [type=f32[n], inputs=[f32[2], f32[n]]] %0 %2
                in (%3)"},
        );
        let linearization = program.linearize_with_respect_to(&[0]).unwrap();
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f32[2], %1:f32[n] .
                let %2:f32[n] = cast [type=f32[n], inputs=[f32[2], f32[n]]] %0 %1
                in (%2)"},
        );
        let pullback = linearization.tangent().transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f32[n], %1:f32[n] .
                let %2:f32[2] = cast [type=f32[2], inputs=[f32[n], f32[n]]] %0 %1
                in (%2)"},
        );
        let vector = |values: Vec<f32>| Array::vector(values).unwrap();
        assert_eq!(
            pullback.interpret(vec![vector(vec![1.0, 2.0]), vector(vec![0.0, 0.0])]),
            Ok(vec![vector(vec![1.0, 2.0])]),
        );
    }

    #[test]
    fn test_cast_differentiation_identity_witnesses() {
        // In the composite family, a tangent cast is witnessed by the first-class dimension of each identity that one
        // of its sides does not carry, read with `dimension_size` from the primal source (narrowing) or the checked
        // primal output (widening), so linearization retains `dimension<n>` scalars rather than `f32[n]` arrays. The
        // fused narrowing tangent also depends on the checked output, whereas a witness read from the widened output
        // already provides that dependency.
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let symbolic = ArrayIrType::from(ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()])));
        let exact = ArrayIrType::from(ArrayType::new_static(DataType::F32, [2]));
        let vector = |values: Vec<f32>| ArrayIrValue::Array(Array::vector(values).unwrap());
        let (_, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |inputs: Vec<Tracer<_>>| Ok(vec![inputs[0].cast(exact.clone(), &[])?]),
            vec![symbolic.clone()],
        )
        .unwrap();
        assert_eq!(
            program.jvp().unwrap().to_string(),
            indoc! {"
                lambda %0:f32[n], %1:f32[n] .
                let %2:f32[2] = cast [type=f32[2], inputs=[f32[n]], checked=true] %0
                    %3:dimension<n ∈ [1, 8)> = dimension_size [axis=0] %0
                    %4:f32[2] = cast [type=f32[2], inputs=[f32[n], dimension<n ∈ [1, 8)>, f32[2]]] %1 %3 %2
                in (%2, %4)"},
        );
        let linearization = program.linearize().unwrap();
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:f32[n] .
                let %1:f32[2] = cast [type=f32[2], inputs=[f32[n]], checked=true] %0
                    %2:dimension<n ∈ [1, 8)> = dimension_size [axis=0] %0
                in (%1, %2)"},
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f32[n], %1:dimension<n ∈ [1, 8)> .
                let %2:f32[2] = cast [type=f32[2], inputs=[f32[n], dimension<n ∈ [1, 8)>]] %0 %1
                in (%2)"},
        );
        let pullback = linearization.tangent().transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f32[2], %1:dimension<n ∈ [1, 8)> .
                let %2:f32[n] = cast [type=f32[n], inputs=[f32[2], dimension<n ∈ [1, 8)>]] %0 %1
                in (%2)"},
        );
        let mut outputs = linearization.primal().interpret(vec![vector(vec![1.0, 2.0])]).unwrap();
        let residuals = outputs.split_off(1);
        assert_eq!(outputs, vec![vector(vec![1.0, 2.0])]);
        let mut pullback_inputs = vec![vector(vec![3.0, 4.0])];
        pullback_inputs.extend(residuals);
        assert_eq!(pullback.interpret(pullback_inputs), Ok(vec![vector(vec![3.0, 4.0])]));

        let (_, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |(value, witness)| value.cast(symbolic.clone(), &[witness]),
            (exact.clone(), symbolic.clone()),
        )
        .unwrap();
        let program = program.to_flat_program();
        assert_eq!(
            program.jvp().unwrap().to_string(),
            indoc! {"
                lambda %0:f32[2], %1:f32[n], %2:f32[2], %3:f32[n] .
                let %4:f32[n] = cast [type=f32[n], inputs=[f32[2], f32[n]], checked=true] %0 %1
                    %5:dimension<n ∈ [1, 8)> = dimension_size [axis=0] %4
                    %6:f32[n] = cast [type=f32[n], inputs=[f32[2], dimension<n ∈ [1, 8)>]] %2 %5
                in (%4, %6)"},
        );
        let linearization = program.linearize_with_respect_to(&[0]).unwrap();
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:f32[2], %1:f32[n] .
                let %2:f32[n] = cast [type=f32[n], inputs=[f32[2], f32[n]], checked=true] %0 %1
                    %3:dimension<n ∈ [1, 8)> = dimension_size [axis=0] %2
                in (%2, %3)"},
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f32[2], %1:dimension<n ∈ [1, 8)> .
                let %2:f32[n] = cast [type=f32[n], inputs=[f32[2], dimension<n ∈ [1, 8)>]] %0 %1
                in (%2)"},
        );
        let pullback = linearization.tangent().transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f32[n], %1:dimension<n ∈ [1, 8)> .
                let %2:f32[2] = cast [type=f32[2], inputs=[f32[n], dimension<n ∈ [1, 8)>]] %0 %1
                in (%2)"},
        );
        let mut outputs =
            linearization.primal().interpret(vec![vector(vec![1.0, 2.0]), vector(vec![0.0, 0.0])]).unwrap();
        let residuals = outputs.split_off(1);
        assert_eq!(outputs, vec![vector(vec![1.0, 2.0])]);
        let mut pullback_inputs = vec![vector(vec![3.0, 4.0])];
        pullback_inputs.extend(residuals);
        assert_eq!(pullback.interpret(pullback_inputs), Ok(vec![vector(vec![3.0, 4.0])]));
    }

    #[test]
    fn test_cast_differentiation_layout_and_sharding() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let output_type = ArrayType::scalar(DataType::F64)
            .with_layout(Layout::Strided(StridedLayout::new(Vec::new())))
            .with_sharding(Sharding::replicated(mesh, 0))
            .unwrap();
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |value| value.cast(output_type.clone(), &[]),
            ArrayType::scalar(DataType::F64),
        )
        .unwrap();
        // The static input proves the relationship, so the checked cast is bound unchecked and linearization keeps
        // no residual: the tangent program converts the tangent representation on its own.
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[][layout=strided{}][sharding={mesh<['x'=1:explicit]>, []}] = cast [
                    type=f64[][layout=strided{}][sharding={mesh<['x'=1:explicit]>, []}],
                    inputs=[f64[]],
                ] %0
                in (%1)"},
        );
        let linearization = program.to_flat_program().linearize().unwrap();
        assert_eq!(linearization.primal().to_string(), program.to_string());
        assert_eq!(linearization.tangent().to_string(), program.to_string());
        for checked in [true, false] {
            let input = Array::scalar(2f64).unwrap();
            let (primal, tangent) = differentiate_at(input.clone())
                .jvp(Array::scalar(3f64).unwrap(), |value| {
                    if checked {
                        value.cast(output_type.clone(), &[])
                    } else {
                        value.cast_unchecked(output_type.clone(), &[])
                    }
                })
                .unwrap();
            assert_eq!(primal, Array::from_elements(output_type.clone(), &[2f64]).unwrap());
            assert_eq!(tangent, Array::from_elements(output_type.clone(), &[3f64]).unwrap());
            // Reverse mode must restore its exact nominal accumulator descriptor, rather than inheriting the output's
            // optional annotations into the inverse cast. Concrete results must honor that representation too.
            check_gradient!(
                |value| if checked {
                    value.cast(output_type.clone(), &[])
                } else {
                    value.cast_unchecked(output_type.clone(), &[])
                },
                at = input,
                step = 1e-6,
                tolerance = 1e-6,
            );
        }
    }

    #[test]
    fn test_cast_differentiation_refined_representation() {
        // Layout and sharding change directly, with one tangent conversion and no erased intermediate. The static
        // input type proves the relationship, so the JVP normalizes the checked primal cast as well, and split
        // linearization needs no primal residual for ordering. The composite universe stages the same programs.
        let plain = ArrayType::new_static(DataType::F32, [2]);
        let input = plain.clone().with_layout(Layout::Strided(StridedLayout::new(vec![8])));
        let output = plain.clone().with_layout(Layout::Strided(StridedLayout::new(vec![12])));
        let first_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let second_mesh = LogicalMesh::new(vec![MeshAxis::new("y", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharded_input = plain.clone().with_sharding(Sharding::replicated(first_mesh, 1)).unwrap();
        let sharded_output = plain.clone().with_sharding(Sharding::replicated(second_mesh, 1)).unwrap();
        let expected_layout = formatdoc! {"
            lambda %0:{input}, %1:{input} .
            let %2:{output} = cast [type={output}, inputs=[f32[2]]] %0
                %3:{output} = cast [type={output}, inputs=[{input}]] %1
            in (%2, %3)"};
        let expected_sharding = formatdoc! {"
            lambda %0:{sharded_input}, %1:{sharded_input} .
            let %2:{sharded_output} = cast [type={sharded_output}, inputs=[f32[2]]] %0
                %3:{sharded_output} = cast [
                    type={sharded_output},
                    inputs=[{sharded_input}],
                ] %1
            in (%2, %3)"};
        let expected_tangent_layout = formatdoc! {"
            lambda %0:{input} .
            let %1:{output} = cast [type={output}, inputs=[{input}]] %0
            in (%1)"};
        let expected_tangent_sharding = formatdoc! {"
            lambda %0:{sharded_input} .
            let %1:{sharded_output} = cast [
                type={sharded_output},
                inputs=[{sharded_input}],
            ] %0
            in (%1)"};
        for (input, output, expected, expected_tangent) in [
            (input, output, expected_layout, expected_tangent_layout),
            (sharded_input, sharded_output, expected_sharding, expected_tangent_sharding),
        ] {
            for constructor in [CastOperation::checked, CastOperation::unchecked] {
                let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
                let value = builder.add_input(input.clone());
                let result = builder
                    .add_instruction(
                        constructor(output.clone(), vec![plain.clone()]).unwrap(),
                        vec![],
                        vec![value],
                        None,
                    )
                    .unwrap()[0];
                let program = builder
                    .build::<Vec<Array>, Vec<Array>>(vec![result], vec![Placeholder], vec![Placeholder])
                    .unwrap();
                let jvp = program.jvp().unwrap();
                assert_eq!(jvp.to_string(), expected);
                let primal = Array::from_elements(input.clone(), &[1f32, 2f32]).unwrap();
                let tangent = Array::from_elements(input.clone(), &[3f32, 4f32]).unwrap();
                let expected_values = vec![
                    Array::from_elements(output.clone(), &[1f32, 2f32]).unwrap(),
                    Array::from_elements(output.clone(), &[3f32, 4f32]).unwrap(),
                ];
                assert_eq!(jvp.interpret(vec![primal.clone(), tangent.clone()]), Ok(expected_values.clone()));
                let linearization = program.linearize().unwrap();
                assert_eq!(linearization.residual_count(), 0);
                assert_eq!(linearization.tangent().to_string(), expected_tangent);
                let composite = program.into_unprojected::<ArrayIrValue<Array>, ArrayIrOperation<Array>>().unwrap();
                let composite_jvp = composite.jvp().unwrap();
                assert_eq!(composite_jvp.to_string(), expected);
                assert_eq!(
                    composite_jvp.interpret(vec![ArrayIrValue::Array(primal), ArrayIrValue::Array(tangent)]),
                    Ok(expected_values.into_iter().map(ArrayIrValue::Array).collect()),
                );
                assert_eq!(composite.linearize().unwrap().tangent().to_string(), expected_tangent);
            }
        }
    }

    #[test]
    fn test_cast_differentiation_refined_representation_checked_shape() {
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let declared = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()]));
        let input = declared.clone().with_layout(Layout::Strided(StridedLayout::new(vec![8])));
        let output =
            ArrayType::new_static(DataType::F32, [2]).with_layout(Layout::Strided(StridedLayout::new(vec![12])));
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let value = builder.add_input(input.clone());
        let result = builder
            .add_instruction(
                CastOperation::checked(output.clone(), vec![declared.clone()]).unwrap(),
                vec![],
                vec![value],
                None,
            )
            .unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![result], vec![Placeholder], vec![Placeholder]).unwrap();
        let jvp = program.jvp().unwrap();
        // One retained source carries n and the primal check into the fused tangent conversion.
        assert_eq!(
            jvp.to_string(),
            formatdoc! {"
                lambda %0:{input}, %1:{input} .
                let %2:{output} = cast [type={output}, inputs=[{declared}], checked=true] %0
                    %3:{input} = cast [
                        type={input},
                        inputs=[{input}, {output}],
                    ] %0 %2
                    %4:{output} = cast [
                        type={output},
                        inputs=[{input}, {input}],
                    ] %1 %3
                in (%2, %4)"},
        );
        let exact = input.clone().with_shape(Shape::new(vec![Dimension::Static(2)]));
        let primal = Array::from_elements(exact.clone(), &[1f32, 2f32]).unwrap();
        let tangent = Array::from_elements(exact, &[3f32, 4f32]).unwrap();
        assert_eq!(
            jvp.interpret(vec![primal, tangent]),
            Ok(vec![
                Array::from_elements(output.clone(), &[1f32, 2f32]).unwrap(),
                Array::from_elements(output.clone(), &[3f32, 4f32]).unwrap(),
            ]),
        );
        let wrong_type = input.with_shape(Shape::new(vec![Dimension::Static(3)]));
        let wrong = Array::from_elements(wrong_type.clone(), &[1f32, 2f32, 3f32]).unwrap();
        assert_eq!(
            jvp.interpret(vec![wrong.clone(), wrong]),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`cast` cannot convert type `{wrong_type}` to unrelated type `{output}`",
            )))),
        );
    }

    #[test]
    fn test_cast_differentiation_refined_representation_witness_identity() {
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let declared = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()]));
        let plain = ArrayType::new_static(DataType::F32, [2]);
        let input = plain.clone().with_layout(Layout::Strided(StridedLayout::new(vec![8])));
        let output = plain.with_layout(Layout::Strided(StridedLayout::new(vec![12])));
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let value = builder.add_input(input.clone());
        let witness = builder.add_input(declared.clone());
        let result = builder
            .add_instruction(
                CastOperation::checked(output.clone(), vec![declared.clone(), declared.clone()]).unwrap(),
                vec![],
                vec![value, witness],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![result], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        // The source has an exact shape. The fused tangent depends on the checked result without introducing n, and
        // split linearization needs no residual once the primal has completed.
        let expected = formatdoc! {"
            lambda %0:{input}, %1:{declared}, %2:{input}, %3:{declared} .
            let %4:{output} = cast [type={output}, inputs=[{declared}, {declared}], checked=true] %0 %1
                %5:{output} = cast [
                    type={output},
                    inputs=[{input}, {output}],
                ] %2 %4
            in (%4, %5)"};
        assert_eq!(program.jvp().unwrap().to_string(), expected);
        let linearization = program.linearize_with_respect_to(&[0]).unwrap();
        assert_eq!(linearization.residual_count(), 0);
        assert_eq!(
            linearization.tangent().to_string(),
            formatdoc! {"
                lambda %0:{input} .
                let %1:{output} = cast [type={output}, inputs=[{input}]] %0
                in (%1)"},
        );
        let primal = Array::from_elements(input.clone(), &[1f32, 2f32]).unwrap();
        let tangent = Array::from_elements(input, &[3f32, 4f32]).unwrap();
        let witness = Array::vector(vec![0f32; 2]).unwrap();
        let inputs = vec![primal, witness.clone(), tangent, witness];
        let expected_values = vec![
            Array::from_elements(output.clone(), &[1f32, 2f32]).unwrap(),
            Array::from_elements(output, &[3f32, 4f32]).unwrap(),
        ];
        assert_eq!(program.jvp().unwrap().interpret(inputs.clone()), Ok(expected_values.clone()));
        let composite = program.into_unprojected::<ArrayIrValue<Array>, ArrayIrOperation<Array>>().unwrap();
        let composite_jvp = composite.jvp().unwrap();
        assert_eq!(composite_jvp.to_string(), expected);
        assert_eq!(
            composite_jvp.interpret(inputs.into_iter().map(ArrayIrValue::Array).collect()),
            Ok(expected_values.into_iter().map(ArrayIrValue::Array).collect()),
        );
    }

    #[test]
    fn test_cast_differentiation_validates_witnesses() {
        // A tangent cast staged by a guard's derivative witnesses the renamed output with the primal input and output,
        // so concrete witnesses must agree with the source extent that the output name was renamed from.
        let bounds = DimensionBounds::positive(Some(8)).unwrap();
        let [a, b] = ["a", "b"].map(|name| DimensionType::new(name, bounds));
        let vector = |dimension: &DimensionType| {
            ArrayType::new(DataType::F32, Shape::new(vec![dimension.simplified_dimension()]))
        };
        let exact = |extent| ArrayType::new_static(DataType::F32, [extent]);
        let operation = CastOperation::unchecked(vector(&a), vec![vector(&b), vector(&b), vector(&a)]).unwrap();
        assert_eq!(
            operation.infer_output_types(&[exact(3), exact(3), exact(5)], &[]),
            Err(TypeError::custom(DimensionError::InputDimensionMismatch {
                dimension: "a".to_owned(),
                expected: 5,
                actual: 3,
            })),
        );
        assert_eq!(operation.infer_output_types(&[exact(3), exact(3), exact(3)], &[]), Ok(vec![exact(3)]));
    }

    #[test]
    fn test_cast_differentiation_preserves_witness_dependencies() {
        // The tangent cast stays behind every witness producer, here a guard, through one source-typed residual that
        // retains n together with the checked result, in both modes.
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let symbolic = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()]));
        let exact = ArrayType::new_static(DataType::F32, [2]);
        for (checked, mode) in [(true, ", checked=true"), (false, "")] {
            let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
                |inputs: Vec<Tracer<_>>| {
                    let context = inputs[1].context();
                    let fixed = context.constant(Array::vector(vec![0f32; 2])?);
                    let witnesses = context.bind(
                        GuardOperation::new("witness", vec![symbolic.clone(); 2]),
                        vec![],
                        &[inputs[1].clone(), fixed],
                    )?;
                    Ok(vec![if checked {
                        inputs[0].cast(exact.clone(), &witnesses)?
                    } else {
                        inputs[0].cast_unchecked(exact.clone(), &witnesses)?
                    }])
                },
                vec![symbolic.clone(); 2],
            )
            .unwrap();
            assert_eq!(
                program.jvp().unwrap().to_string(),
                formatdoc! {r#"
                    lambda %0:f32[n], %1:f32[n], %2:f32[n], %3:f32[n] .
                    let %4:f32[2] = const [0.0, 0.0]
                        %5:f32[2], %6:f32[2] = guard [boundary="witness", declared_types=[f32[n], f32[n]]] %1 %4
                        %7:f32[n] = cast [type=f32[n], inputs=[f32[n], f32[2]]] %1 %5
                        %8:f32[2] = cast [type=f32[2], inputs=[f32[n], f32[n]]] %3 %7
                        %9:f32[2] = cast [type=f32[2], inputs=[f32[n], f32[2], f32[2]]{mode}] %0 %5 %6
                        %10:f32[n] = cast [type=f32[n], inputs=[f32[n], f32[2]]] %0 %9
                        %11:f32[2] = cast [type=f32[2], inputs=[f32[n], f32[n]]] %2 %10
                    in (%9, %11)"#},
            );
            let linearization = program.linearize_with_respect_to(&[0]).unwrap();
            assert_eq!(linearization.residual_count(), 1);
            assert_eq!(
                linearization.tangent().to_string(),
                indoc! {"
                    lambda %0:f32[n], %1:f32[n] .
                    let %2:f32[2] = cast [type=f32[2], inputs=[f32[n], f32[n]]] %0 %1
                    in (%2)"},
            );
        }
    }

    #[test]
    fn test_cast_transposition() {
        let exact = ArrayType::new_static(DataType::F32, [2]);
        let laid_out = exact.clone().with_layout(Layout::Strided(StridedLayout::new(vec![8])));
        check_operation_transposition!(
            @exact,
            operation = CastOperation::unchecked(laid_out.clone(), vec![exact.clone()]).unwrap(),
            cases = [{
                inputs = [(@linear(type = exact.clone()))],
                output_cotangents = [Array::from_elements(laid_out, &[1f32, 2f32]).unwrap()],
                input_cotangents = [Array::vector(vec![1f32, 2f32]).unwrap()],
                pullback = indoc! {"
                    lambda %0:f32[2][layout=strided{8}] .
                    let %1:f32[2] = cast [type=f32[2], inputs=[f32[2][layout=strided{8}]]] %0
                    in (%1)"},
            }],
        );

        // A proven identity binds nothing, so its transpose is the identity as well.
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |inputs: Vec<Tracer<_>>| Ok(vec![inputs[0].cast(exact.clone(), &[])?]),
            vec![exact.clone()],
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2] .
                in (%0)"},
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

        // The linearized tangent of a checked narrowing is an unchecked cast whose source-typed residual carries the
        // primal check, so its transpose widens the cotangent back with that residual as witness.
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let symbolic = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()]));
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |inputs: Vec<Tracer<_>>| Ok(vec![inputs[0].cast(exact.clone(), &[])?]),
            vec![symbolic],
        )
        .unwrap();
        let pullback = program.linearize().unwrap().tangent().transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f32[2], %1:f32[n] .
                let %2:f32[n] = cast [type=f32[n], inputs=[f32[2], f32[n]]] %0 %1
                in (%2)"},
        );
        let vector = |values: Vec<f32>| Array::vector(values).unwrap();
        assert_eq!(
            pullback.interpret(vec![vector(vec![1.0, 2.0]), vector(vec![3.0, 4.0])]),
            Ok(vec![vector(vec![1.0, 2.0])]),
        );
    }

    #[test]
    fn test_cast_transposition_zero_cotangents() {
        // Structural zeros retain the original nominal cotangent types and require no materialized witness.
        let exact = ArrayType::new_static(DataType::F32, [2]);
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let symbolic = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()]));
        let witness = ArrayType::new_static(DataType::F32, [3]);
        let operation = CastOperation::unchecked(exact.clone(), vec![symbolic.clone(), witness.clone()]).unwrap();
        let mut context = TranspositionContext::new(TracingContext::<Array, ArrayOperation<Array>>::new());
        let inputs = [PartialValue::Unknown(symbolic.clone()), PartialValue::Unknown(witness.clone())];
        let accumulators = context.cotangent_accumulators(&inputs, &[]).unwrap();
        operation
            .transpose(&mut context, &EmptyRegionDriver, &inputs, &[MaybeZero::Zero(exact)], &accumulators)
            .unwrap();
        let cotangents = context.take_cotangents(&accumulators).unwrap();
        assert!(matches!(cotangents.as_slice(), [MaybeZero::Zero(first), MaybeZero::Zero(second)]
            if *first == symbolic && *second == witness));
        assert!(context.builder().borrow().instructions().is_empty());
    }

    #[test]
    fn test_cast_transposition_unneeded_cotangent() {
        // A source whose cotangent is not needed stages nothing, even for a live output cotangent.
        let exact = ArrayType::new_static(DataType::F32, [2]);
        let laid_out = exact.clone().with_layout(Layout::Strided(StridedLayout::new(vec![8])));
        let operation = CastOperation::unchecked(laid_out.clone(), vec![exact.clone()]).unwrap();
        let tracing = TracingContext::<Array, ArrayOperation<Array>>::new();
        let cotangent = tracing.input(laid_out);
        let mut context = TranspositionContext::new(tracing.clone());
        let accumulator = context.cotangent_accumulator(exact.clone(), false);
        assert!(!accumulator.is_needed());
        operation
            .transpose(
                &mut context,
                &EmptyRegionDriver,
                &[PartialValue::Unknown(exact.clone())],
                &[MaybeZero::Value(cotangent)],
                std::slice::from_ref(&accumulator),
            )
            .unwrap();
        let cotangents = context.take_cotangents(&[accumulator]).unwrap();
        assert!(matches!(cotangents.as_slice(), [MaybeZero::Zero(r#type)] if *r#type == exact));
        assert!(tracing.builder().borrow().instructions().is_empty());
    }

    #[test]
    fn test_cast_transposition_refined_representation() {
        // Reversing a representation change restores the exact nominal input representation, in both modes and
        // universes, directly and through linearization.
        let plain = ArrayType::new_static(DataType::F32, [2]);
        let input = plain.clone().with_layout(Layout::Strided(StridedLayout::new(vec![8])));
        let output = plain.clone().with_layout(Layout::Strided(StridedLayout::new(vec![12])));
        let first_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let second_mesh = LogicalMesh::new(vec![MeshAxis::new("y", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharded_input = plain.clone().with_sharding(Sharding::replicated(first_mesh, 1)).unwrap();
        let sharded_output = plain.clone().with_sharding(Sharding::replicated(second_mesh, 1)).unwrap();
        let expected_layout = formatdoc! {"
            lambda %0:{output} .
            let %1:{input} = cast [type={input}, inputs=[{output}]] %0
            in (%1)"};
        let expected_sharding = formatdoc! {"
            lambda %0:{sharded_output} .
            let %1:{sharded_input} = cast [
                type={sharded_input},
                inputs=[{sharded_output}],
            ] %0
            in (%1)"};
        let cotangent = Array::from_elements(output.clone(), &[1f32, 2f32]).unwrap();
        let expected_cotangent = Array::from_elements(input.clone(), &[1f32, 2f32]).unwrap();
        let sharded_cotangent = Array::from_elements(sharded_output.clone(), &[1f32, 2f32]).unwrap();
        let expected_sharded_cotangent = Array::from_elements(sharded_input.clone(), &[1f32, 2f32]).unwrap();
        check_operation_transposition!(
            @exact,
            operation = CastOperation::checked(output.clone(), vec![plain.clone()]).unwrap(),
            cases = [{
                inputs = [(@linear(type = input.clone()))],
                output_cotangents = [cotangent.clone()],
                input_cotangents = [expected_cotangent.clone()],
                pullback = expected_layout.clone(),
            }],
        );
        check_operation_transposition!(
            @exact,
            operation = CastOperation::unchecked(output.clone(), vec![plain.clone()]).unwrap(),
            cases = [{
                inputs = [(@linear(type = input.clone()))],
                output_cotangents = [cotangent.clone()],
                input_cotangents = [expected_cotangent.clone()],
                pullback = expected_layout.clone(),
            }],
        );
        check_operation_transposition!(
            @exact,
            operation = CastOperation::checked(sharded_output.clone(), vec![plain.clone()]).unwrap(),
            cases = [{
                inputs = [(@linear(type = sharded_input.clone()))],
                output_cotangents = [sharded_cotangent.clone()],
                input_cotangents = [expected_sharded_cotangent.clone()],
                pullback = expected_sharding.clone(),
            }],
        );
        check_operation_transposition!(
            @exact,
            operation = CastOperation::unchecked(sharded_output.clone(), vec![plain.clone()]).unwrap(),
            cases = [{
                inputs = [(@linear(type = sharded_input.clone()))],
                output_cotangents = [sharded_cotangent.clone()],
                input_cotangents = [expected_sharded_cotangent.clone()],
                pullback = expected_sharding.clone(),
            }],
        );
        check_operation_transposition!(
            @exact,
            backend = (ArrayIrValue<Array>, ArrayIrOperation<Array>),
            operation = CastOperation::unchecked(output.clone().into(), vec![ArrayIrType::from(plain.clone())]).unwrap(),
            cases = [{
                inputs = [(@linear(type = ArrayIrType::from(input.clone())))],
                output_cotangents = [ArrayIrValue::Array(cotangent.clone())],
                input_cotangents = [ArrayIrValue::Array(expected_cotangent.clone())],
                pullback = expected_layout.clone(),
            }],
        );
        check_operation_transposition!(
            @exact,
            backend = (ArrayIrValue<Array>, ArrayIrOperation<Array>),
            operation = CastOperation::checked(sharded_output.clone().into(), vec![ArrayIrType::from(plain.clone())]).unwrap(),
            cases = [{
                inputs = [(@linear(type = ArrayIrType::from(sharded_input.clone())))],
                output_cotangents = [ArrayIrValue::Array(sharded_cotangent.clone())],
                input_cotangents = [ArrayIrValue::Array(expected_sharded_cotangent.clone())],
                pullback = expected_sharding.clone(),
            }],
        );

        // Linearizing first changes nothing: the shape is already proven, so the tangent program is the same unchecked
        // cast and its pullback is the same inverse cast.
        for (input, output, expected, cotangent, expected_cotangent) in [
            (input, output, expected_layout, cotangent, expected_cotangent),
            (sharded_input, sharded_output, expected_sharding, sharded_cotangent, expected_sharded_cotangent),
        ] {
            for constructor in [CastOperation::checked, CastOperation::unchecked] {
                let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
                let value = builder.add_input(input.clone());
                let result = builder
                    .add_instruction(
                        constructor(output.clone(), vec![plain.clone()]).unwrap(),
                        vec![],
                        vec![value],
                        None,
                    )
                    .unwrap()[0];
                let program = builder
                    .build::<Vec<Array>, Vec<Array>>(vec![result], vec![Placeholder], vec![Placeholder])
                    .unwrap();
                let pullback = program.linearize().unwrap().tangent().transpose_with_respect_to(&[0], &[]).unwrap();
                assert_eq!(pullback.to_string(), expected);
                assert_eq!(pullback.interpret(vec![cotangent.clone()]), Ok(vec![expected_cotangent.clone()]));
                let composite = program.into_unprojected::<ArrayIrValue<Array>, ArrayIrOperation<Array>>().unwrap();
                let composite_pullback =
                    composite.linearize().unwrap().tangent().transpose_with_respect_to(&[0], &[]).unwrap();
                assert_eq!(composite_pullback.to_string(), expected);
                assert_eq!(
                    composite_pullback.interpret(vec![ArrayIrValue::Array(cotangent.clone())]),
                    Ok(vec![ArrayIrValue::Array(expected_cotangent.clone())]),
                );
            }
        }
    }

    #[test]
    fn test_cast_transposition_checked_assertions() {
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let symbolic = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()]));
        let exact = ArrayType::new_static(DataType::F32, [2]);
        let layout = exact.clone().with_layout(Layout::Strided(StridedLayout::new(vec![8])));

        // A checked cast whose relationship the input types prove transposes like an unchecked one.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(exact.clone());
        let output = builder
            .add_instruction(
                CastOperation::checked(layout.clone(), vec![exact.clone()]).unwrap(),
                vec![],
                vec![input],
                None,
            )
            .unwrap()[0];
        let proven = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap()
            .transpose()
            .unwrap();
        assert_eq!(
            proven.to_string(),
            indoc! {"
                lambda %0:f32[2][layout=strided{8}] .
                let %1:f32[2] = cast [type=f32[2], inputs=[f32[2][layout=strided{8}]]] %0
                in (%1)"},
        );
        assert_eq!(
            proven.interpret(vec![Array::from_elements(layout, &[1f32, 2f32]).unwrap()]),
            Ok(vec![Array::vector(vec![1f32, 2f32]).unwrap()]),
        );

        // A live result, a discarded result and a zero-output program must all preserve the primal precondition.
        let expected = DifferentiationError::Program(ProgramError::UnsupportedOperation {
            message: "cannot transpose a checked `cast` without the primal value its check needs; linearize the \
                      program first, or use `cast_unchecked` with witnesses for a tangent that an earlier check \
                      already proved"
                .to_owned(),
        });
        for outputs in 0..3 {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let value = builder.add_input(symbolic.clone());
            let witness = builder.add_input(symbolic.clone());
            let result = builder
                .add_instruction(
                    CastOperation::checked(exact.clone(), vec![symbolic.clone(), symbolic.clone()]).unwrap(),
                    vec![],
                    vec![value, witness],
                    None,
                )
                .unwrap()[0];
            let outputs = match outputs {
                0 => vec![result],
                1 => vec![value],
                _ => vec![],
            };
            let output_structure = vec![Placeholder; outputs.len()];
            let program =
                builder.build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder; 2], output_structure).unwrap();
            assert_eq!(program.transpose_with_respect_to(&[0], &[]).unwrap_err(), expected);
        }

        // Known checks are replayed even when their results have no cotangents.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let value = builder.add_input(exact.clone());
        let known = builder.add_input(symbolic.clone());
        builder
            .add_instruction(CastOperation::checked(exact.clone(), vec![symbolic]).unwrap(), vec![], vec![known], None)
            .unwrap();
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![value], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let pullback = program.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f32[2], %1:f32[n] .
                let %2:f32[2] = cast [type=f32[2], inputs=[f32[n]], checked=true] %1
                in (%0)"},
        );
        let vector = |extent| Array::vector(vec![1f32; extent]).unwrap();
        assert_eq!(pullback.interpret(vec![vector(2), vector(2)]), Ok(vec![vector(2)]));
        assert_eq!(
            pullback.interpret(vec![vector(2), vector(5)]),
            Err(ProgramError::Type(TypeError::invalid(
                "`cast` cannot convert type `f32[5]` to unrelated type `f32[2]`",
            ))),
        );
    }

    #[test]
    fn test_cast_transposition_linear_witness_known_source() {
        // A checked cast whose source is known but whose linear witness its check needs cannot be transposed either,
        // since the witness has no primal value during the reverse walk.
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let symbolic = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()]));
        let exact = ArrayType::new_static(DataType::F32, [2]);
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let value = builder.add_input(symbolic.clone());
        let witness = builder.add_input(symbolic.clone());
        let result = builder
            .add_instruction(
                CastOperation::checked(exact, vec![symbolic.clone(), symbolic]).unwrap(),
                vec![],
                vec![value, witness],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![result], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.transpose_with_respect_to(&[1], &[]).unwrap_err(),
            DifferentiationError::Program(ProgramError::UnsupportedOperation {
                message: "cannot transpose a checked `cast` without the primal witnesses its check needs; linearize \
                          the program first, or use `cast_unchecked` with witnesses for a tangent that an earlier \
                          check already proved"
                    .to_owned(),
            }),
        );
    }

    #[test]
    fn test_cast_transposition_validates_inputs() {
        // The rule validates its input count and the supplied input types before using any accumulator.
        let exact = ArrayType::new_static(DataType::F32, [2]);
        let operation = CastOperation::unchecked(exact.clone(), vec![exact.clone()]).unwrap();
        let mut context = TranspositionContext::new(TracingContext::<Array, ArrayOperation<Array>>::new());
        assert_eq!(
            operation.transpose(&mut context, &EmptyRegionDriver, &[], &[MaybeZero::Zero(exact.clone())], &[]),
            Err(DifferentiationError::Program(ProgramError::Type(TypeError::invalid("expected 1 input but got 0")))),
        );
        let wrong = ArrayType::new_static(DataType::F32, [3]);
        let accumulator = context.cotangent_accumulator(wrong.clone(), false);
        assert_eq!(
            operation.transpose(
                &mut context,
                &EmptyRegionDriver,
                &[PartialValue::Unknown(wrong)],
                &[MaybeZero::Zero(exact)],
                &[accumulator],
            ),
            Err(DifferentiationError::Program(ProgramError::Type(TypeError::invalid(
                "dimension 3 does not refine declared dimension 2",
            )))),
        );
    }

    #[test]
    fn test_cast_transposition_known_shape() {
        let exact = ArrayType::new_static(DataType::F32, [2]);
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let symbolic = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()]));
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let source = builder.add_input(exact.clone());
        let witness = builder.add_input(symbolic.clone());
        let output = builder
            .add_instruction(
                CastOperation::checked(symbolic.clone(), vec![exact, symbolic]).unwrap(),
                vec![],
                vec![source, witness],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        // The witness could prove the shape, but checking it during the reverse walk could reorder assertions. Require
        // linearization, which executes the primal check in forward order.
        assert_eq!(
            program.transpose_with_respect_to(&[0], &[]).unwrap_err(),
            DifferentiationError::Program(ProgramError::UnsupportedOperation {
                message: "cannot transpose a checked `cast` without the primal value its check needs; linearize the \
                          program first, or use `cast_unchecked` with witnesses for a tangent that an earlier check \
                          already proved"
                    .to_owned(),
            }),
        );
    }

    #[test]
    fn test_cast_transposition_known_region_checks() {
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let symbolic = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()]));
        let exact = |extent| ArrayType::new_static(DataType::F32, [extent]);
        let mut checking_builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let checking_input = checking_builder.add_input(symbolic.clone());
        checking_builder
            .add_instruction(
                CastOperation::checked(exact(2), vec![symbolic.clone()]).unwrap(),
                vec![],
                vec![checking_input],
                None,
            )
            .unwrap();
        let checking_branch = checking_builder
            .build::<Vec<Array>, Vec<Array>>(vec![checking_input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut identity_builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let identity_input = identity_builder.add_input(symbolic.clone());
        let identity_branch = identity_builder
            .build::<Vec<Array>, Vec<Array>>(vec![identity_input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let linear = builder.add_input(exact(2));
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean));
        let known = builder.add_input(symbolic.clone());
        let regions = vec![builder.import_program(checking_branch), builder.import_program(identity_branch)];
        builder.add_instruction(ConditionOperation::new(), regions, vec![predicate, known], None).unwrap();
        builder
            .add_instruction(CastOperation::checked(exact(3), vec![symbolic]).unwrap(), vec![], vec![known], None)
            .unwrap();
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![linear], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        let pullback = program.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f32[2], %1:bool[], %2:f32[n] .
                let %3:f32[n] = condition %1 %2 [
                    true={
                        lambda %0:f32[n] .
                        let %1:f32[2] = cast [type=f32[2], inputs=[f32[n]], checked=true] %0
                        in (%0)
                    },
                    false={
                        lambda %0:f32[n] .
                        in (%0)
                    },
                ]
                    %4:f32[3] = cast [type=f32[3], inputs=[f32[n]], checked=true] %2
                in (%0)"},
        );
        let vector = |extent| Array::vector(vec![1f32; extent]).unwrap();
        // An unselected branch never checks, and a selected branch checks before the following assertion.
        assert_eq!(pullback.interpret(vec![vector(2), Array::scalar(false).unwrap(), vector(3)]), Ok(vec![vector(2)]));
        assert_eq!(
            pullback.interpret(vec![vector(2), Array::scalar(true).unwrap(), vector(5)]),
            Err(ProgramError::Type(TypeError::invalid(
                "`cast` cannot convert type `f32[5]` to unrelated type `f32[2]`",
            ))),
        );
    }

    #[test]
    fn test_cast_transposition_linear_witnesses() {
        let plain = ArrayType::new_static(DataType::F32, [2]);
        let witness = ArrayType::new_static(DataType::F32, [3]);
        for checked in [true, false] {
            let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
                |(value, witness)| {
                    if checked {
                        value.cast(plain.clone(), &[witness])
                    } else {
                        value.cast_unchecked(plain.clone(), &[witness])
                    }
                },
                (plain.clone(), witness.clone()),
            )
            .unwrap();
            assert_eq!(
                program.to_string(),
                indoc! {"
                    lambda %0:f32[2], %1:f32[3] .
                    let %2:f32[2] = cast [type=f32[2], inputs=[f32[2], f32[3]]] %0 %1
                    in (%2)"},
            );
            let transposed = program.transpose_with_respect_to(&[0, 1], &[]).unwrap();
            assert_eq!(
                transposed.to_string(),
                indoc! {"
                    lambda %0:f32[2] .
                    let %1:f32[3] = zero [type=f32[3]]
                    in (%0, %1)"},
            );
            let value = Array::vector(vec![1f32, 2f32]).unwrap();
            assert_eq!(
                transposed.interpret(vec![value.clone()]),
                Ok(vec![value, Array::vector(vec![0f32; 3]).unwrap()]),
            );
        }

        // A representation change still needs no value from an irrelevant static linear witness.
        let output = plain.clone().with_layout(Layout::Strided(StridedLayout::new(vec![8])));
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |(value, witness)| value.cast_unchecked(output.clone(), &[witness]),
            (plain.clone(), witness.clone()),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2], %1:f32[3] .
                let %2:f32[2][layout=strided{8}] = cast [type=f32[2][layout=strided{8}], inputs=[f32[2], f32[3]]] %0 %1
                in (%2)"},
        );
        let transposed = program.transpose_with_respect_to(&[0, 1], &[]).unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f32[2][layout=strided{8}] .
                let %1:f32[2] = cast [type=f32[2], inputs=[f32[2][layout=strided{8}]]] %0
                    %2:f32[3] = zero [type=f32[3]]
                in (%1, %2)"},
        );
        let value = Array::from_elements(output, &[1f32, 2f32]).unwrap();
        assert_eq!(
            transposed.interpret(vec![value]),
            Ok(vec![Array::vector(vec![1f32, 2f32]).unwrap(), Array::vector(vec![0f32; 3]).unwrap()]),
        );

        // A linear witness does not provide a value for a required nominal identity. Keep the existing useful error.
        let name = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let symbolic = ArrayType::new(DataType::F32, Shape::new(vec![name.simplified_dimension()]));
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |(value, witness)| value.cast_unchecked(plain.clone(), &[witness]),
            (symbolic.clone(), symbolic),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[n], %1:f32[n] .
                let %2:f32[2] = cast [type=f32[2], inputs=[f32[n], f32[n]]] %0 %1
                in (%2)"},
        );
        assert_eq!(
            program.transpose_with_respect_to(&[0, 1], &[]).unwrap_err(),
            DifferentiationError::Program(ProgramError::UnsupportedOperation {
                message: "`cast` transposition needs a known witness for input identity `n`; \
                          linearize the program before transposing it"
                    .to_owned(),
            }),
        );
    }

    #[test]
    fn test_cast_transposition_known_witnesses() {
        // Known witnesses are kept as witnesses of the cotangent cast, so their producers stay explicit dependencies.
        let exact = ArrayType::new_static(DataType::F32, [2]);
        let laid_out = exact.clone().with_layout(Layout::Strided(StridedLayout::new(vec![8])));
        check_operation_transposition!(
            @exact,
            operation = CastOperation::unchecked(
                laid_out.clone(),
                vec![exact.clone(), ArrayType::new_static(DataType::F32, [3])],
            ).unwrap(),
            cases = [{
                inputs = [(@linear(type = exact.clone())), (@known, Array::vector(vec![1f32; 3]).unwrap())],
                output_cotangents = [Array::from_elements(laid_out, &[1f32, 2f32]).unwrap()],
                input_cotangents = [Array::vector(vec![1f32, 2f32]).unwrap()],
                pullback = indoc! {"
                    lambda %0:f32[2][layout=strided{8}], %1:f32[3] .
                    let %2:f32[2] = cast [type=f32[2], inputs=[f32[2][layout=strided{8}], f32[3]]] %0 %1
                    in (%2)"},
            }],
        );
    }

    #[test]
    fn test_cast_transposition_input_identities() {
        // Bare narrowing has no saved value carrying n. Do not invent that identity or change the accumulator type.
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let symbolic = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()]));
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(symbolic.clone());
        let output = builder
            .add_instruction(
                CastOperation::unchecked(ArrayType::new_static(DataType::F32, [2]), vec![symbolic]).unwrap(),
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
                let %1:f32[2] = cast [type=f32[2], inputs=[f32[n]]] %0
                in (%1)"},
        );
        assert_eq!(
            program.transpose_with_respect_to(&[0], &[]).unwrap_err(),
            DifferentiationError::Program(ProgramError::UnsupportedOperation {
                message: "`cast` transposition needs a known witness for input identity `n`; \
                          linearize the program before transposing it"
                    .to_owned(),
            }),
        );

        // A known dimension can witness a narrowed axis without being an explicit input of the forward cast.
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
            .add_instruction(
                CastOperation::unchecked(output_type.clone(), vec![input_type.clone()]).unwrap(),
                Vec::new(),
                vec![input],
                None,
            )
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
                let %3:f32[n, 2] = cast [type=f32[n, 2], inputs=[f32[n, m]]] %2
                in (%3)"},
        );
        let transposed = program.transpose_with_respect_to(&[2], &[]).unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f32[n, 2], %1:dimension<m ∈ [1, 8)>, %2:dimension<unrelated ∈ [1, 8)> .
                let %3:f32[n, m] = cast [type=f32[n, m], inputs=[f32[n, 2], dimension<m ∈ [1, 8)>]] %0 %1
                in (%3)"},
        );
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
    fn test_cast_transposition_known_intermediate() {
        // A known intermediate that defines an identity (here, a dimension decoded from a known scalar) supplies that
        // identity to the cotangent cast of a narrowing when no known program input carries it, and the pullback
        // replays its pure producer.
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let symbolic: ArrayIrType = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()])).into();
        let exact: ArrayIrType = ArrayType::new_static(DataType::F32, [2]).into();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let scalar = builder.add_input(ArrayType::scalar(DataType::I32).into());
        let value = builder.add_input(exact.clone());
        let dimension = builder
            .add_instruction(DimensionFromScalarOperation::new(n.clone()), Vec::new(), vec![scalar], None)
            .unwrap()[0];
        let widened = builder
            .add_instruction(
                CastOperation::unchecked(symbolic.clone(), vec![exact.clone(), n.clone().into()]).unwrap(),
                Vec::new(),
                vec![value, dimension],
                None,
            )
            .unwrap()[0];
        let narrowed = builder
            .add_instruction(
                CastOperation::unchecked(exact.clone(), vec![symbolic]).unwrap(),
                Vec::new(),
                vec![widened],
                None,
            )
            .unwrap()[0];
        // The reverse walk materializes a known intermediate when a rule first consumes it, so a later consumer
        // makes the dimension available to the narrowing's transpose, which is not an explicit input of it.
        let witnessed = builder
            .add_instruction(
                CastOperation::unchecked(exact.clone(), vec![exact.clone(), n.clone().into()]).unwrap(),
                Vec::new(),
                vec![narrowed, dimension],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![witnessed],
                vec![Placeholder; 2],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:i32[], %1:f32[2] .
                let %2:dimension<n ∈ [1, 8)> = dimension_from_scalar [bounds=[1, 8)] %0
                    %3:f32[n] = cast [type=f32[n], inputs=[f32[2], dimension<n ∈ [1, 8)>]] %1 %2
                    %4:f32[2] = cast [type=f32[2], inputs=[f32[n]]] %3
                    %5:f32[2] = cast [type=f32[2], inputs=[f32[2], dimension<n ∈ [1, 8)>]] %4 %2
                in (%5)"},
        );
        let pullback = program.transpose_with_respect_to(&[1], &[]).unwrap();
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f32[2], %1:i32[] .
                let %2:dimension<n ∈ [1, 8)> = dimension_from_scalar [bounds=[1, 8)] %1
                    %3:f32[2] = cast [type=f32[2], inputs=[f32[2], dimension<n ∈ [1, 8)>]] %0 %2
                    %4:f32[n] = cast [type=f32[n], inputs=[f32[2], dimension<n ∈ [1, 8)>]] %3 %2
                    %5:f32[2] = cast [type=f32[2], inputs=[f32[n], dimension<n ∈ [1, 8)>]] %4 %2
                in (%5)"},
        );
        let cotangent = ArrayIrValue::Array(Array::vector(vec![1f32, 2f32]).unwrap());
        assert_eq!(
            pullback.interpret(vec![cotangent.clone(), ArrayIrValue::Array(Array::scalar(2i32).unwrap())]),
            Ok(vec![cotangent.clone()]),
        );
        assert_eq!(
            pullback.interpret(vec![cotangent, ArrayIrValue::Array(Array::scalar(3i32).unwrap())]),
            Err(ProgramError::Type(TypeError::custom(DimensionError::InputDimensionMismatch {
                dimension: "n".to_owned(),
                expected: 3,
                actual: 2,
            }))),
        );
    }

    #[test]
    fn test_cast_transposition_known_intermediate_without_consumer() {
        // A known intermediate whose only use is as the witness of a narrowing is never an input of a transpose rule.
        // Its producer is a known assertion, so the pullback replays it before the reverse walk, and that replay must
        // supply its identity to the cotangent cast of the narrowing.
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let symbolic: ArrayIrType = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()])).into();
        let exact: ArrayIrType = ArrayType::new_static(DataType::F32, [2]).into();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let scalar = builder.add_input(ArrayType::scalar(DataType::I32).into());
        let value = builder.add_input(exact.clone());
        let dimension = builder
            .add_instruction(DimensionFromScalarOperation::new(n.clone()), Vec::new(), vec![scalar], None)
            .unwrap()[0];
        let widened = builder
            .add_instruction(
                CastOperation::unchecked(symbolic.clone(), vec![exact.clone(), n.clone().into()]).unwrap(),
                Vec::new(),
                vec![value, dimension],
                None,
            )
            .unwrap()[0];
        let narrowed = builder
            .add_instruction(CastOperation::unchecked(exact, vec![symbolic]).unwrap(), Vec::new(), vec![widened], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![narrowed],
                vec![Placeholder; 2],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:i32[], %1:f32[2] .
                let %2:dimension<n ∈ [1, 8)> = dimension_from_scalar [bounds=[1, 8)] %0
                    %3:f32[n] = cast [type=f32[n], inputs=[f32[2], dimension<n ∈ [1, 8)>]] %1 %2
                    %4:f32[2] = cast [type=f32[2], inputs=[f32[n]]] %3
                in (%4)"},
        );
        let pullback = program.transpose_with_respect_to(&[1], &[]).unwrap();
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f32[2], %1:i32[] .
                let %2:dimension<n ∈ [1, 8)> = dimension_from_scalar [bounds=[1, 8)] %1
                    %3:f32[n] = cast [type=f32[n], inputs=[f32[2], dimension<n ∈ [1, 8)>]] %0 %2
                    %4:f32[2] = cast [type=f32[2], inputs=[f32[n], dimension<n ∈ [1, 8)>]] %3 %2
                in (%4)"},
        );
        let cotangent = ArrayIrValue::Array(Array::vector(vec![1f32, 2f32]).unwrap());
        assert_eq!(
            pullback.interpret(vec![cotangent.clone(), ArrayIrValue::Array(Array::scalar(2i32).unwrap())]),
            Ok(vec![cotangent]),
        );
    }

    #[test]
    fn test_cast_transposition_declared_dimensions() {
        let bounds = DimensionBounds::positive(Some(5)).unwrap();
        let [a, b] = ["a", "b"].map(|name| DimensionType::new(name, bounds));
        let input = ArrayType::new(DataType::F32, Shape::new(vec![a.simplified_dimension(), b.simplified_dimension()]));
        let output =
            ArrayType::new(DataType::F32, Shape::new(vec![b.simplified_dimension(), a.simplified_dimension()]));
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let value = builder.add_input(input.clone());
        let result = builder
            .add_instruction(CastOperation::unchecked(output, vec![input]).unwrap(), Vec::new(), vec![value], None)
            .unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![result], vec![Placeholder], vec![Placeholder]).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[a, b] .
                let %1:f32[b, a] = cast [type=f32[b, a], inputs=[f32[a, b]]] %0
                in (%1)"},
        );
        let transposed = program.transpose_with_respect_to(&[0], &[]).unwrap();

        // Reversing the cast swaps its declared input and requested output. The cotangent already carries both names,
        // so it needs no additional witness, and the nominal accumulator type must remain [a, b].
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f32[b, a] .
                let %1:f32[a, b] = cast [type=f32[a, b], inputs=[f32[b, a]]] %0
                in (%1)"},
        );
        let value = Array::matrix(2, 2, vec![1f32; 4]).unwrap();
        assert_eq!(transposed.interpret(vec![value.clone()]), Ok(vec![value.clone()]));
        let pullback = program.linearize().unwrap().tangent().transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f32[b, a] .
                let %1:f32[a, b] = cast [type=f32[a, b], inputs=[f32[b, a]]] %0
                in (%1)"},
        );
        assert_eq!(pullback.interpret(vec![value.clone()]), Ok(vec![value]));
    }

    #[test]
    fn test_cast_type_infer_cast_output_type() {
        // An independent type family can implement the operation-specific contract without changing
        // `TypeRefinements` (refer to the module-level fixture).
        check_operation_type_inference!(
            operation = CastOperation::checked(ProjectedMemberType::<0>, vec![ProjectedMemberType::<0>]).unwrap(),
            cases = [{ input_types = [ProjectedMemberType::<0>], output_types = [ProjectedMemberType::<0>] }],
        );

        // Bare data types relate only to themselves.
        assert_eq!(
            DataType::infer_cast_output_type(
                &CastOperation::checked(DataType::F32, vec![DataType::F32]).unwrap(),
                &[DataType::F32],
            ),
            Ok((DataType::F32, Some(DataType::F32))),
        );
        assert_eq!(
            DataType::infer_cast_output_type(
                &CastOperation::checked(DataType::F64, vec![DataType::F32]).unwrap(),
                &[DataType::F32],
            ),
            Err(TypeError::invalid("`cast` cannot convert type `f32` to unrelated type `f64`")),
        );

        // Arrays keep their element type, rank, and memory space, and the representation is interpretable exactly when
        // the supplied types prove the shape, in both modes.
        let array = ArrayType::new_static(DataType::F32, [2, 3]);
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let symbolic = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()]));
        let exact = |extent| ArrayType::new_static(DataType::F32, [extent]);
        for constructor in [CastOperation::checked, CastOperation::unchecked] {
            assert_eq!(
                ArrayType::infer_cast_output_type(
                    &constructor(array.clone(), vec![array.clone()]).unwrap(),
                    std::slice::from_ref(&array),
                ),
                Ok((array.clone(), Some(array.clone()))),
            );
            for output in [
                ArrayType::scalar(DataType::F32),
                ArrayType::new_static(DataType::F32, [1, 6]),
                array.clone().with_memory(Memory::Host { pinned: true }),
            ] {
                assert_eq!(
                    ArrayType::infer_cast_output_type(
                        &constructor(output.clone(), vec![array.clone()]).unwrap(),
                        std::slice::from_ref(&array),
                    ),
                    Err(TypeError::invalid(format!(
                        "`{CAST_OPERATION_NAME}` cannot convert type `{}` to unrelated type `{}`",
                        &array, &output,
                    ))),
                );
            }
            let operation = constructor(symbolic.clone(), vec![exact(2), symbolic.clone()]).unwrap();
            assert_eq!(
                ArrayType::infer_cast_output_type(&operation, &[exact(2), exact(2)]),
                Ok((exact(2), Some(exact(2)))),
            );
            assert_eq!(
                ArrayType::infer_cast_output_type(&operation, &[exact(2), exact(3)]),
                Err(TypeError::custom(DimensionError::InputDimensionMismatch {
                    dimension: "n".to_owned(),
                    expected: 3,
                    actual: 2,
                })),
            );
            assert_eq!(
                ArrayType::infer_cast_output_type(&operation, &[]),
                Err(TypeError::invalid("expected 2 inputs but got 0")),
            );
            assert_eq!(
                ArrayType::infer_cast_output_type(&operation, &[exact(2)]),
                Err(TypeError::invalid("expected 2 inputs but got 1")),
            );
        }

        // Conditional narrowing can be staged, while nominal identity and concrete extents prove the relationship.
        let narrowing = CastOperation::checked(exact(2), vec![symbolic.clone()]).unwrap();
        assert_eq!(ArrayType::infer_cast_output_type(&narrowing, &[symbolic.clone()]), Ok((exact(2), None)));
        assert_eq!(ArrayType::infer_cast_output_type(&narrowing, &[exact(2)]), Ok((exact(2), Some(exact(2)))));
        assert_eq!(
            ArrayType::infer_cast_output_type(
                &CastOperation::checked(symbolic.clone(), vec![symbolic.clone()]).unwrap(),
                &[symbolic.clone()],
            ),
            Ok((symbolic.clone(), Some(symbolic.clone()))),
        );

        // Representation changes alone never need a runtime check.
        let plain = ArrayType::new_static(DataType::U8, [2]);
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let annotated = plain
            .clone()
            .with_layout(Layout::Strided(StridedLayout::new(vec![2])))
            .with_sharding(Sharding::replicated(mesh, 1))
            .unwrap();
        for constructor in [CastOperation::checked, CastOperation::unchecked] {
            assert_eq!(
                ArrayType::infer_cast_output_type(
                    &constructor(annotated.clone(), vec![plain.clone()]).unwrap(),
                    &[plain.clone()],
                ),
                Ok((annotated.clone(), Some(annotated.clone()))),
            );
        }

        // First-class dimensions narrow to static types, which a differently named actual instantiates only within
        // its own bounds, and cannot target identity-defining types.
        let dimension = DimensionType::from(Dimension::Static(2));
        let static_narrowing = CastOperation::checked(dimension.clone(), vec![n.clone()]).unwrap();
        assert_eq!(
            DimensionType::infer_cast_output_type(&static_narrowing, std::slice::from_ref(&n)),
            Ok((dimension.clone(), None)),
        );
        assert_eq!(
            DimensionType::infer_cast_output_type(&static_narrowing, std::slice::from_ref(&dimension)),
            Ok((dimension.clone(), Some(dimension.clone()))),
        );
        let narrow = DimensionType::new("narrow", DimensionBounds::new(4, Some(8)).unwrap());
        assert_eq!(
            DimensionType::infer_cast_output_type(&static_narrowing, std::slice::from_ref(&narrow)),
            Err(TypeError::custom(DimensionError::BindingOutOfBounds {
                variable: "narrow".to_owned(),
                value: 2,
                bounds: narrow.bounds(),
            })),
        );
        assert_eq!(
            DimensionType::infer_cast_output_type(
                &CastOperation::checked(DimensionType::from(Dimension::Static(3)), vec![dimension.clone()]).unwrap(),
                std::slice::from_ref(&dimension),
            ),
            Err(TypeError::invalid(format!(
                "`{CAST_OPERATION_NAME}` cannot convert type `{}` to unrelated type `{}`",
                &dimension,
                &DimensionType::from(Dimension::Static(3))
            ))),
        );
        assert_eq!(
            DimensionType::infer_cast_output_type(
                &CastOperation::checked(n, vec![dimension.clone()]).unwrap(),
                &[dimension.clone()],
            ),
            Err(TypeError::invalid("`cast` output type `dimension<n ∈ [1, 8)>` cannot define a type identity")),
        );

        // The composite universe rejects casts across member kinds and references.
        assert_eq!(
            ArrayIrType::infer_cast_output_type(
                &CastOperation::checked(ArrayIrType::Dimension(dimension), vec![array.clone().into()]).unwrap(),
                &[array.clone().into()],
            ),
            Err(TypeError::invalid("`cast` cannot convert type `f32[2, 3]` to unrelated type `dimension<2>`")),
        );
        let reference: ArrayIrType = ReferenceType::new(array.clone()).into();
        assert_eq!(
            ArrayIrType::infer_cast_output_type(
                &CastOperation::checked(array.clone().into(), vec![array.into(), reference.clone()]).unwrap(),
                &[ArrayType::new_static(DataType::F32, [2, 3]).into(), reference],
            ),
            Err(TypeError::invalid("`cast` does not support reference type `ref<f32[2, 3]>`")),
        );
    }

    #[test]
    fn test_cast_type_infer_cast_output_type_rejects_first_class_ragged_witnesses() {
        let witness = ArrayIrType::Ragged(
            RaggedType::new(
                ArrayType::new_static(DataType::I64, [2]),
                RaggedValueKind::Dimension(DimensionType::from(Dimension::Static(3))),
                Vec::new(),
            )
            .unwrap(),
        );
        let array = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let operation = CastOperation::checked(array.clone(), vec![array.clone(), witness.clone()]).unwrap();
        assert_eq!(
            operation.infer_output_types(&[array, witness.clone()], &[]),
            Err(TypeError::invalid(format!("`cast` does not support first-class ragged type `{witness}`",))),
        );
    }

    #[test]
    fn test_cast_type_identity_witness_operation() {
        // The composite universe reads an array identity from the first axis that carries it as a first-class
        // dimension, whose type carries the same identity.
        let n = DimensionVariable::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let m = DimensionVariable::new("m", DimensionBounds::positive(Some(8)).unwrap());
        let array =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), n.clone().into(), n.clone().into()]));
        let operation = ArrayIrType::identity_witness_operation(&array.clone().into(), &n).unwrap().unwrap();
        assert_eq!(operation, DimensionSizeOperation::new(&array, 1).unwrap());
        assert_eq!(operation.to_string(), "dimension_size [axis=1]");
        assert_eq!(
            operation.infer_output_types(&[array.clone().into()], &[]),
            Ok(vec![DimensionType::from(Dimension::Dynamic(n.clone())).into()]),
        );

        // Values that do not carry the identity, first-class dimensions (which already are such scalars), and
        // references have no identity witness operation.
        assert_eq!(ArrayIrType::identity_witness_operation(&array.clone().into(), &m), Ok(None));
        let dimension = ArrayIrType::Dimension(DimensionType::from(Dimension::Dynamic(n.clone())));
        assert_eq!(ArrayIrType::identity_witness_operation(&dimension, &n), Ok(None));
        let reference = ArrayIrType::from(ReferenceType::new(array.clone()));
        assert_eq!(ArrayIrType::identity_witness_operation(&reference, &n), Ok(None));

        // An identity that the value carries but that cannot be read as a first-class dimension value is an error
        // rather than `None`, so that transforms fail instead of silently retaining the whole array.
        let wide = DimensionVariable::new("wide", DimensionBounds::new(MAX_DIMENSION_EXTENT + 1, None).unwrap());
        let wide_array = ArrayType::new(DataType::F32, Shape::new(vec![wide.clone().into()]));
        assert_eq!(
            ArrayIrType::identity_witness_operation(&wide_array.into(), &wide),
            Err(DimensionError::ExtentExceedsBackendWidth {
                value: MAX_DIMENSION_EXTENT + 1,
                maximum: MAX_DIMENSION_EXTENT,
            }
            .into()),
        );

        // Homogeneous families have no such operation and retain whole values instead.
        assert_eq!(ArrayType::identity_witness_operation(&array, &n), Ok(None));
        let dimension = DimensionType::from(Dimension::Dynamic(n.clone()));
        assert_eq!(DimensionType::identity_witness_operation(&dimension, &n), Ok(None));
    }

    #[test]
    fn test_cast_cast() {
        // Context-carrying values bind a checked cast through their context, which the static input type may already
        // prove, and a checked cast next to an unchecked one with a witness.
        let plain = ArrayType::new_static(DataType::F32, [2]);
        let laid_out = plain.clone().with_layout(Layout::Strided(StridedLayout::new(vec![8])));
        let (_, proven) =
            TracingContext::<Array, ArrayOperation<Array>>::trace(|value| value.cast(laid_out.clone(), &[]), plain)
                .unwrap();
        assert_eq!(
            proven.to_string(),
            indoc! {"
                lambda %0:f32[2] .
                let %1:f32[2][layout=strided{8}] = cast [type=f32[2][layout=strided{8}], inputs=[f32[2]]] %0
                in (%1)"},
        );
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
                let %2:f32[2] = cast [type=f32[2], inputs=[f32[n]], checked=true] %0
                    %3:f32[2] = cast [type=f32[2], inputs=[f32[n], f32[n]]] %0 %1
                in (%2, %3)"},
        );

        // A name absent from both the value and its declared witnesses is not a lost identity from concretization.
        // Do not erase it and admit a widening for which no input supplies the requested relationship.
        let error = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |(value, _witness)| value.cast(symbolic.clone(), &[]),
            (exact, symbolic.clone()),
        )
        .unwrap_err();
        assert_eq!(
            error,
            ProgramError::Type(TypeError::invalid("`cast` output identity `n` is not carried by a declared input")),
        );
    }

    #[test]
    fn test_cast_cast_unchecked() {
        // An unchecked cast binds without a check and materializes the requested representation when interpreted.
        let plain = ArrayType::new_static(DataType::F32, [2]);
        let laid_out = plain.clone().with_layout(Layout::Strided(StridedLayout::new(vec![8])));
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |value| value.cast_unchecked(laid_out.clone(), &[]),
            plain,
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2] .
                let %1:f32[2][layout=strided{8}] = cast [type=f32[2][layout=strided{8}], inputs=[f32[2]]] %0
                in (%1)"},
        );
        assert_eq!(
            program.interpret(Array::vector(vec![1f32, 2f32]).unwrap()),
            Ok(Array::from_elements(laid_out, &[1f32, 2f32]).unwrap()),
        );
    }

    #[test]
    fn test_array_cast() {
        // Concrete values cannot supply nominal names, with or without exact shape witnesses, in either mode.
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let symbolic = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()]));
        let exact = |extent: usize| ArrayType::new_static(DataType::F32, [extent]);
        let vector = |extent: usize| Array::vector(vec![1f32; extent]).unwrap();
        let missing_name = Err(ProgramError::Type(TypeError::invalid(
            "`cast` output identity `n` is not carried by a declared input",
        )));
        for witnesses in [vec![], vec![vector(2)], vec![vector(3)], vec![vector(2), vector(2)]] {
            assert_eq!(vector(2).cast(symbolic.clone(), &witnesses), missing_name);
            assert_eq!(vector(2).cast_unchecked(symbolic.clone(), &witnesses), missing_name);
        }
        assert_eq!(vector(2).cast(exact(2), &[]), Ok(vector(2)));
        let unrelated = Err(ProgramError::Type(TypeError::invalid(
            "`cast` cannot convert type `f32[3]` to unrelated type `f32[2]`",
        )));
        assert_eq!(vector(3).cast(exact(2), &[]), unrelated);
        assert_eq!(vector(3).cast_unchecked(exact(2), &[]), unrelated);
    }

    #[test]
    fn test_array_cast_unspecified_layout_and_sharding() {
        // A sharding annotation shares the input's storage, whereas a layout re-encodes the logical bytes into dense
        // padded storage, in both modes.
        let input = Array::vector(vec![10u8, 20]).unwrap();
        let plain = input.r#type().into_owned();
        let layout = Layout::Strided(StridedLayout::new(vec![2]));
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding = Sharding::replicated(mesh, 1);
        for output_type in [
            plain.clone().with_layout(layout.clone()),
            plain.clone().with_sharding(sharding.clone()).unwrap(),
            plain.with_layout(layout).with_sharding(sharding).unwrap(),
        ] {
            let expected = Array::from_elements(output_type.clone(), &[10u8, 20]).unwrap();
            for result in [input.cast(output_type.clone(), &[]), input.cast_unchecked(output_type.clone(), &[])] {
                let result = result.unwrap();
                assert_eq!(result, expected);
                assert_eq!(result.r#type().as_ref(), &output_type);
                assert_eq!(result.logical_bytes(), vec![10, 20]);
                if output_type.layout().is_some() {
                    assert_eq!(result.storage_bytes(), &[10, 0, 20]);
                } else {
                    assert!(Arc::ptr_eq(result.shared_storage_bytes(), input.shared_storage_bytes()));
                }
            }
        }

        // Layout materialization copies element encodings exactly, including a NaN payload and a signed zero.
        let input = Array::vector(vec![f32::from_bits(0x7fc12345), f32::from_bits(0x80000000)]).unwrap();
        let output_type = input.r#type().clone().into_owned().with_layout(Layout::Strided(StridedLayout::new(vec![8])));
        let result = input.cast(output_type.clone(), &[]).unwrap();
        assert_eq!(result.r#type().as_ref(), &output_type);
        assert_eq!(result.logical_bytes(), input.logical_bytes());
        assert_eq!(result.storage_bytes().len(), 12);
        assert_eq!(&result.storage_bytes()[4..8], &[0, 0, 0, 0]);
        // Removing a layout returns to canonical dense storage; relabeling a padded buffer would read holes as data.
        let plain = ArrayType::new_static(DataType::F32, [2]);
        let dense = result.cast(plain.clone(), &[]).unwrap();
        assert_eq!(dense.r#type().as_ref(), &plain);
        assert_eq!(dense.logical_bytes(), input.logical_bytes());
        assert_eq!(dense.storage_bytes().len(), 8);
    }

    #[test]
    fn test_array_ir_value_cast() {
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let symbolic = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()]));
        let exact = |extent: usize| ArrayType::new_static(DataType::F32, [extent]);
        let vector = |extent: usize| Array::vector(vec![1f32; extent]).unwrap();

        // Composite array members behave like arrays, including representation changes, and reference witnesses are
        // rejected.
        let composite = |extent: usize| ArrayIrValue::<Array>::Array(vector(extent));
        assert_eq!(
            composite(2).cast(symbolic.clone().into(), &[]),
            Err(ProgramError::Type(TypeError::invalid(
                "`cast` output identity `n` is not carried by a declared input",
            ))),
        );
        assert_eq!(
            composite(3).cast_unchecked(exact(2).into(), &[]),
            Err(ProgramError::Type(TypeError::invalid(
                "`cast` cannot convert type `f32[3]` to unrelated type `f32[2]`",
            ))),
        );
        let laid_out = exact(2).with_layout(Layout::Strided(StridedLayout::new(vec![8])));
        let expected = Ok(ArrayIrValue::Array(Array::from_elements(laid_out.clone(), &[1f32, 1f32]).unwrap()));
        assert_eq!(composite(2).cast(laid_out.clone().into(), &[]), expected);
        assert_eq!(composite(2).cast_unchecked(laid_out.into(), &[]), expected);
        let reference = ArrayIrValue::Reference(ArrayReference::new(vector(2)));
        let expected = Err(ProgramError::Type(TypeError::invalid(format!(
            "`cast` does not support reference type `{}`",
            reference.r#type(),
        ))));
        assert_eq!(composite(2).cast(exact(2).into(), std::slice::from_ref(&reference)), expected);
        assert_eq!(composite(2).cast_unchecked(exact(2).into(), &[reference]), expected);

        // A first-class dimension type defines its identity, which the output type of a cast may not, so only exact
        // dimension types are valid outputs for dimension values, and concrete witnesses carry no identities.
        let dimension = |extent: usize| ArrayIrValue::<Array>::Dimension(DimensionValue::constant(extent).unwrap());
        let exact_dimension = ArrayIrType::Dimension(DimensionType::from(Dimension::Static(2)));
        assert_eq!(dimension(2).cast(exact_dimension.clone(), &[]), Ok(dimension(2)));
        assert_eq!(dimension(2).cast(exact_dimension.clone(), &[composite(2)]), Ok(dimension(2)));
        assert_eq!(
            dimension(3).cast(exact_dimension, &[]),
            Err(ProgramError::Type(TypeError::invalid(
                "`cast` cannot convert type `dimension<3>` to unrelated type `dimension<2>`",
            ))),
        );
        assert_eq!(
            dimension(2).cast(ArrayIrType::Dimension(n), &[]),
            Err(ProgramError::Type(TypeError::invalid(
                "`cast` output type `dimension<n ∈ [1, 8)>` cannot define a type identity",
            ))),
        );
    }

    #[test]
    fn test_array_ir_value_cast_wrapped_tracers() {
        // A composite value whose array member is a tracer delegates array casts to that member's context, so symbolic
        // relationships are staged and checked at runtime rather than inspected eagerly, whereas a host dimension cast
        // cannot retain a staged array witness.
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let symbolic = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()]));
        let exact = ArrayType::new_static(DataType::F32, [2]);
        let vector = |extent: usize| Array::vector(vec![1f32; extent]).unwrap();
        let unretained = ProgramError::Concretization {
            message: "host dimension `cast` cannot retain an array witness producer; \
                      bind the cast through a composite context"
                .to_owned(),
        };
        let context = TracingContext::<CaptureReference<ArrayType>, ArrayOperation<CaptureReference<ArrayType>>>::new();
        let captured = context.constant(CaptureReference::new(0, exact.clone()));
        let dimension = ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap());
        assert_eq!(
            dimension.cast(DimensionType::from(Dimension::Static(2)).into(), &[ArrayIrValue::Array(captured)]),
            Err(unretained.clone()),
        );
        for (checked, mode) in [(true, ", checked=true"), (false, "")] {
            let rejected_dimension = TracingContext::<Array, ArrayOperation<Array>>::trace(
                |witness| {
                    let value = ArrayIrValue::Dimension(DimensionValue::constant(2)?);
                    let witnesses = [ArrayIrValue::Array(witness)];
                    let target = DimensionType::from(Dimension::Static(2)).into();
                    let result =
                        if checked { value.cast(target, &witnesses) } else { value.cast_unchecked(target, &witnesses) };
                    result.map(|_| ())
                },
                exact.clone(),
            );
            assert_eq!(rejected_dimension.map(|_| ()), Err(unretained.clone()));
            let rejected = TracingContext::<Array, ArrayOperation<Array>>::trace(
                |value| {
                    let reference = ArrayIrValue::Reference(ArrayReference::new(value.clone()));
                    let value = ArrayIrValue::Array(value);
                    let output = if checked {
                        value.cast(exact.clone().into(), &[reference])?
                    } else {
                        value.cast_unchecked(exact.clone().into(), &[reference])?
                    };
                    let ArrayIrValue::Array(output) = output else { panic!("expected an array") };
                    Ok(output)
                },
                exact.clone(),
            );
            assert_eq!(
                rejected.map(|_| ()),
                Err(ProgramError::Type(TypeError::invalid("`cast` does not support reference type `ref<f32[2]>`"))),
            );

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
                let %2:f32[n] = cast [type=f32[n], inputs=[f32[2], f32[n]]{mode}] %0 %1
                in (%2)"};
            assert_eq!(program.to_string(), expected);
            assert_eq!(program.interpret((vector(2), vector(2))), Ok(vector(2)));
            // Interpretation validates concrete shapes in both modes; unchecked lowering alone assumes them.
            // The input witness binds n = 3 before the output observes the value's extent 2.
            assert_eq!(
                program.interpret((vector(2), vector(3))),
                Err(ProgramError::Type(TypeError::custom(DimensionError::InputDimensionMismatch {
                    dimension: "n".to_owned(),
                    expected: 3,
                    actual: 2,
                }))),
            );
            let residual = program
                .to_flat_program()
                .partially_evaluate(&[PartialValue::Unknown(exact.clone()), PartialValue::Unknown(symbolic.clone())])
                .unwrap();
            assert_eq!(residual.program().to_string(), expected);

            // Narrowing a wrapped tracer must stage rather than attempting to inspect symbolic shapes eagerly.
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
                    let %1:f32[2] = cast [type=f32[2], inputs=[f32[n]]{mode}] %0
                    in (%1)"},
            );
            assert_eq!(narrowed.interpret(vector(2)), Ok(vector(2)));
            assert_eq!(
                narrowed.interpret(vector(3)),
                Err(ProgramError::Type(TypeError::invalid(
                    "`cast` cannot convert type `f32[3]` to unrelated type `f32[2]`",
                ))),
            );
        }
    }
}
