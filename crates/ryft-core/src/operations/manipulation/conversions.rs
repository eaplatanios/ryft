//! Operations that change the element type of a value, either by converting its elements numerically or by
//! reinterpreting their encoding bits as another [`DataType`]. Numerical conversion keeps the shape, sharding, and
//! memory space of its input and follows the destination format's rounding, saturation, and exceptional-value rules,
//! whereas a bitcast keeps every encoding bit and adds or consumes a trailing axis when the two element widths differ.
//! This module provides the following:
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
//!
//! Numerical conversion is linear, so differentiation converts the primal and aligns the tangent with the output's
//! tangent type, transposition converts the cotangent back to the input's cotangent type, and conversions into or out
//! of a type without a tangent space (e.g., integers) produce structural zeros. A bitcast has a structural zero
//! derivative and cannot be transposed. Batching keeps the mapped axis in place, except that a widening bitcast moves
//! it away from the trailing position it would otherwise consume. Backends lower numerical conversion and bitcasts to
//! their element-conversion and bit-reinterpretation constructs (e.g.,
//! [`stablehlo.convert`](https://openxla.org/stablehlo/spec#convert) and
//! [`stablehlo.bitcast_convert`](https://openxla.org/stablehlo/spec#bitcast_convert) in the XLA backend).
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
//!     program.interpret(Array::vector(vec![1.0_f64, -2.0, 0.5])?)?,
//!     Array::vector(vec![0x3f800000_u32, 0xc0000000, 0x3f000000])?,
//! );
//! # Ok(())
//! # }
//! ```

use std::fmt::Display;
use std::marker::PhantomData;

use crate::arrays::{
    Array, ArrayAddressing, ArrayBatch, ArrayBatchingPolicy, ArrayExtentBatchingPolicy, ArrayType, DataType, Dimension,
    Layout, ShardingDimension,
};
use crate::batching::{
    BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError, InterpretableBatchableOperation,
};
use crate::contexts::{Context, Domain};
use crate::differentiation::{DifferentiableType, DifferentiationDual, ElementwiseDerivativeAlignment};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_differentiable_operation};
use crate::operations::manipulation::transposition::Transpose;
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{
    MaybeZero, Operation, OperationFormatter, ProgramError, RegionInterface, Type, TypeError, Typed, Value,
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
        let widening = self.bitcast
            && element_bit_width(inputs[0].value().r#type().data_type()) < element_bit_width(self.data_type);
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

    /// Returns a copy of this type with `data_type` as its element type, preserving structural and placement metadata
    /// while clearing byte-stride layouts that a changed element storage width invalidates. This does not validate the
    /// numerical conversion itself; [`ConvertElementType::convert_element_type`] converts actual values.
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

        let input_bits = element_bit_width(source);
        let output_bits = element_bit_width(data_type);
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

// TODO(eaplatanios): Review from here onwards.

/// Converts elements numerically or reinterprets their encoding bits as another [`DataType`]. Numerical conversion
/// preserves shape, sharding, and memory space. Byte-stride layouts are cleared when the element storage width changes;
/// tiled layouts are preserved. Numerical conversion may narrow precision or change numerical category.
/// [`Self::bitcast_element_type`] instead preserves encoding bits and changes the trailing shape when source and
/// destination bit widths differ. [`Self::promote_element_type`] additionally checks that the requested conversion is
/// permitted by the type promotion lattice.
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
/// # Examples
///
/// ```rust
/// # use ryft_core::{Array, ConvertElementType, DataType, ProgramError};
/// #
/// # fn main() -> Result<(), ProgramError> {
/// let input = Array::vector(vec![1.75_f64, -2.5]).unwrap();
/// let output = input.convert_element_type(DataType::I32)?;
/// assert_eq!(output, Array::vector(vec![1_i32, -2]).unwrap());
///
/// let input = Array::vector(vec![1.0_f32, 2.0]).unwrap();
/// assert_eq!(input.promote_element_type(DataType::F64)?, Array::vector(vec![1.0_f64, 2.0]).unwrap());
/// # Ok(())
/// # }
/// ```
pub trait ConvertElementType: Sized {
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
    /// Widening requires the consumed axis to be replicated when sharding is specified; partitioned or unconstrained
    /// trailing axes must be resharded first so all pieces of each output element are available on the same device.
    /// Boolean and complex elements support only identity reinterpretation. Tokens and structural zeros are rejected.
    /// Reinterpretation has a zero derivative, including when source and destination element types are equal. In the
    /// Ryft XLA backend, XLA lowering supports equal-width casts without finite dimension bounds. Rank-changing casts
    /// require finite bounds for physical allocation but this does not restrict core eager or symbolic type inference.
    ///
    /// # Parameters
    ///
    ///   - `data_type`: Destination element encoding; it determines both the values and any trailing-axis change.
    ///
    /// # Example
    ///
    /// ```
    /// # use ryft_core::{Array, ConvertElementType, DataType};
    /// let pieces = Array::scalar(0x12345678_u32).unwrap().bitcast_element_type(DataType::U16)?;
    /// assert_eq!(pieces, Array::vector(vec![0x5678_u16, 0x1234]).unwrap());
    /// assert_eq!(pieces.bitcast_element_type(DataType::U32)?, Array::scalar(0x12345678_u32).unwrap());
    /// # Ok::<(), ryft_core::ProgramError>(())
    /// ```
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
        Self: Typed,
        Self::Type: ElementType,
    {
        self.r#type()
            .element_type()
            .promote_to(data_type)
            .map_err(|error| TypeError::invalid(error.to_string()))?;
        self.convert_element_type(data_type)
    }
}

impl<V: Value<Type: ElementType, DispatchDomain: Context<Operation: From<ConvertElementTypeOperation<V::Type>>>>>
    ConvertElementType for V
{
    #[inline]
    fn convert_element_type(&self, data_type: DataType) -> Result<Self, ProgramError> {
        let operation = ConvertElementTypeOperation::<V::Type>::new(data_type, false);
        let input_type = self.r#type();
        Operation::infer_output_types(&operation, std::slice::from_ref(input_type.as_ref()), &[])?;
        if input_type.element_type() == data_type {
            return Ok(self.clone());
        }
        let mut outputs = self.dispatch_domain().bind(operation, Vec::new(), std::slice::from_ref(self))?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }

    #[inline]
    fn bitcast_element_type(&self, data_type: DataType) -> Result<Self, ProgramError> {
        // Even an identity bitcast must be staged: its declared derivative is zero rather than the identity.
        let operation = ConvertElementTypeOperation::<V::Type>::new(data_type, true);
        let mut outputs = self.dispatch_domain().bind(operation, Vec::new(), std::slice::from_ref(self))?;
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
        let input_bits = element_bit_width(self.r#type().data_type());
        let output_bits = element_bit_width(data_type);
        let bytes = self.logical_bytes();
        if input_bits >= 8 && output_bits >= 8 || input_bits == output_bits {
            return Array::from_logical_bytes(output_type, &bytes);
        }
        // Sub-byte host elements occupy separate padded bytes, so the meaningful bits are walked in logical order and
        // the padding is restored at each output element boundary instead of being reinterpreted as input data.
        let output_count = Array::materialized_element_count(&output_type)?;
        let input_bytes = input_bits.div_ceil(8);
        let output_bytes = output_bits.div_ceil(8);
        let mut output = vec![0_u8; output_count * output_bytes];
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

/// Returns the number of meaningful encoding bits of one element of `data_type`, excluding the padding that sub-byte
/// elements occupy in host storage.
fn element_bit_width(data_type: DataType) -> usize {
    match data_type {
        DataType::I1 | DataType::U1 => 1,
        DataType::I2 | DataType::U2 => 2,
        DataType::I4 | DataType::U4 | DataType::F4E2M1FN => 4,
        DataType::F6E2M3FN | DataType::F6E3M2FN => 6,
        _ => ArrayAddressing::element_byte_width_for_data_type(data_type) * 8,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use num_complex::Complex as ComplexNumber;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayElement, ArrayOperation, ArrayType, DataType, Dimension, DimensionBounds, DimensionVariable,
        Layout, LogicalMesh, Memory, MeshAxis, MeshAxisType, RaggedAxis, Shape, Sharding, ShardingDimension,
        StridedLayout, Tile, TileDimension, TiledLayout, bf16, f8e4m3fn, f8e5m2, f8e8m0fnu, i4, u2, u4,
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
    use crate::partial::PartialValue;
    use crate::programs::{EffectClasses, EmptyRegionDriver, Typed};

    use super::*;

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
                     elements"
                )),
            );
        }
    }

    #[test]
    fn test_convert_element_type_interpretation() {
        let output = ConvertElementTypeOperation::<ArrayType>::new(DataType::F32, false)
            .interpret(&EagerContext::<Array>::new(), &EmptyRegionDriver, &[Array::scalar(2.0_f64).unwrap()])
            .unwrap();
        assert_eq!(output, vec![Array::scalar(2.0_f32).unwrap()]);
    }

    #[test]
    fn test_convert_element_type_interpretation_invalid_inputs() {
        let context = EagerContext::<Array>::new();
        let operation = ConvertElementTypeOperation::<ArrayType>::new(DataType::F32, false);

        // Validate arity before accessing operands or attempting element conversion.
        assert!(matches!(
            operation.interpret(&context, &EmptyRegionDriver, &[]),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        ));
        assert!(matches!(
            operation.interpret(
                &context,
                &EmptyRegionDriver,
                &[Array::scalar(2.0_f64).unwrap(), Array::scalar(3.0_f64).unwrap()]
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
                &[Array::scalar(2.0_f64).unwrap()],
            ),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "cannot convert values to or from the `token` data type",
        ));
    }

    #[test]
    fn test_convert_element_type_interpretation_bitcast() {
        // Signed zero and NaN payloads survive without numeric conversion or canonicalization.
        let input = Array::vector(vec![0x80000000_u32, 0x7fc01234]).unwrap();
        let output = ConvertElementTypeOperation::<ArrayType>::new(DataType::F32, true)
            .interpret(&EagerContext::<Array>::new(), &EmptyRegionDriver, &[input.clone()])
            .unwrap()
            .remove(0);
        assert_eq!(output.logical_bytes(), input.logical_bytes());
        assert_eq!(output.bitcast_element_type(DataType::U32), Ok(input));

        let pieces = Array::scalar(0x12345678_u32).unwrap().bitcast_element_type(DataType::U16).unwrap();
        assert_eq!(pieces, Array::vector(vec![0x5678_u16, 0x1234]).unwrap());
        assert_eq!(pieces.bitcast_element_type(DataType::U32), Ok(Array::scalar(0x12345678_u32).unwrap()));
    }

    #[test]
    fn test_convert_element_type_interpretation_bitcast_layout() {
        // Physical holes do not contribute bits when an element splits into a new logical trailing axis.
        let input_type =
            ArrayType::new_static(DataType::U32, [2]).with_layout(Layout::Strided(StridedLayout::new(vec![8])));
        let input = Array::from_elements(input_type, &[0x12345678_u32, 0x89abcdef]).unwrap();
        assert_eq!(
            input.bitcast_element_type(DataType::U16),
            Ok(Array::matrix(2, 2, vec![0x5678_u16, 0x1234, 0xcdef, 0x89ab]).unwrap()),
        );
        assert_eq!(
            Array::vector(Vec::<u32>::new()).unwrap().bitcast_element_type(DataType::U16),
            Ok(Array::matrix(0, 2, Vec::<u16>::new()).unwrap()),
        );
    }

    #[test]
    fn test_convert_element_type_interpretation_bitcast_subbyte() {
        // FP6 occupies padded host bytes, but only its six meaningful bits participate in reinterpretation.
        let encodings = (0_u8..64).collect::<Vec<_>>();
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

        let pieces = Array::scalar(0xab_u8).unwrap().bitcast_element_type(DataType::U4).unwrap();
        assert_eq!(pieces.elements::<u4>().unwrap(), vec![u4::new(11).unwrap(), u4::new(10).unwrap()]);
        assert_eq!(pieces.bitcast_element_type(DataType::U8), Ok(Array::scalar(0xab_u8).unwrap()));
        assert_eq!(
            Array::scalar(true).unwrap().bitcast_element_type(DataType::Boolean),
            Ok(Array::scalar(true).unwrap()),
        );
    }

    #[test]
    fn test_convert_element_type_partial_evaluation() {
        let array_operation = ConvertElementTypeOperation::<ArrayType>::new(DataType::F32, false);

        // Known inputs fold to converted values; unknown inputs retain the conversion.
        check_operation_partial_evaluation!(
            operation = array_operation,
            inputs = [Array::scalar(2.0_f64).unwrap()],
            expected = Array::scalar(2.0_f32).unwrap(),
        );
    }

    #[test]
    fn test_convert_element_type_partial_evaluation_bitcast() {
        check_operation_partial_evaluation!(
            operation = ConvertElementTypeOperation::<ArrayType>::new(DataType::U32, true),
            inputs = [Array::scalar(1.0_f32).unwrap()],
            expected = Array::scalar(0x3f800000_u32).unwrap(),
        );
    }

    #[test]
    fn test_convert_element_type_batching() {
        let array_operation = ConvertElementTypeOperation::<ArrayType>::new(DataType::F32, false);

        check_operation_batching!(
            @exact,
            operation = array_operation,
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), Array::vector(vec![1.0, 2.0]).unwrap())],
                outputs = [(
                    @mapped(axis = 0),
                    Array::from_elements::<f32>(
                        ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)])),
                        &[1.0, 2.0],
                    ).unwrap()
                )],
            }],
        );
        // Numerical conversion preserves a non-leading mapped axis.
        check_operation_batching!(
            @exact,
            operation = array_operation,
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 1), Array::matrix(1, 2, vec![1.0_f64, 2.0]).unwrap())],
                outputs = [(@mapped(axis = 1), Array::matrix(1, 2, vec![1.0_f32, 2.0]).unwrap())],
            }],
        );
    }

    #[test]
    fn test_convert_element_type_batching_bitcast() {
        check_operation_batching!(
            @exact,
            operation = ConvertElementTypeOperation::<ArrayType>::new(DataType::U32, true),
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 1), Array::matrix(2, 2, vec![0x5678_u16, 0xcdef, 0x1234, 0x89ab]).unwrap())],
                outputs = [(@mapped(axis = 0), Array::vector(vec![0x12345678_u32, 0x89abcdef]).unwrap())],
            }],
        );
        // Equal-width reinterpretation keeps the mapped position unchanged.
        check_operation_batching!(
            @exact,
            operation = ConvertElementTypeOperation::<ArrayType>::new(DataType::U32, true),
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 1), Array::matrix(1, 2, vec![1.0_f32, -0.0]).unwrap())],
                outputs = [(@mapped(axis = 1), Array::matrix(1, 2, vec![0x3f800000_u32, 0x80000000]).unwrap())],
            }],
        );
        // Narrowing appends the encoding-piece axis after the existing mapped axis.
        check_operation_batching!(
            @exact,
            operation = ConvertElementTypeOperation::<ArrayType>::new(DataType::U16, true),
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 1), Array::matrix(1, 2, vec![0x12345678_u32, 0x89abcdef]).unwrap())],
                outputs = [(@mapped(axis = 1), Array::from_elements(
                    ArrayType::new_static(DataType::U16, [1, 2, 2]),
                    &[0x5678_u16, 0x1234, 0xcdef, 0x89ab],
                ).unwrap())],
            }],
        );

        // Widening consumes the trailing axis, so a bounded ragged axis in that position cannot be batched, whereas a
        // ragged axis elsewhere carries over to the output.
        let context = BatchingContext::<_, ArrayBatchingPolicy>::new(EagerContext::<Array>::new(), 2);
        let array = Array::from_elements(ArrayType::new_static(DataType::U16, [2, 3, 2]), &[0_u16; 12]).unwrap();
        let variable = DimensionVariable::new("length", DimensionBounds::new(0, Some(4)).unwrap());
        let extents = Array::from_elements(ArrayType::new_static(DataType::I32, [2]), &[1_i32, 3]).unwrap();
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
                     ragged trailing dimension"
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
                primals = [Array::from_elements::<f32>(ArrayType::scalar(DataType::F32), &[2.0]).unwrap()],
                tangents = [Array::from_elements::<f32>(ArrayType::scalar(DataType::F32), &[2.0]).unwrap()],
                primal_outputs = [Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[2.0]).unwrap()],
                tangent_outputs = [Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[2.0]).unwrap()],
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
        let tangent = Array::from_elements::<f32>(ArrayType::scalar(DataType::F32), &[3.0]).unwrap();
        let (output, output_tangent) =
            differentiate_at(primal).jvp(tangent, |value| value.convert_element_type(DataType::F32)).unwrap();
        assert_eq!(output.r#type().into_owned(), ArrayType::scalar(DataType::F32));
        assert_eq!(output_tangent, Array::from_elements::<f32>(ArrayType::scalar(DataType::F32), &[3.0]).unwrap());

        let primal = Array::from_elements::<f32>(ArrayType::scalar(DataType::F32), &[2.0]).unwrap();
        let tangent = Array::from_elements::<f32>(ArrayType::scalar(DataType::F32), &[3.0]).unwrap();
        let (_, output_tangent) = differentiate_at(primal)
            .jvp(tangent, |value| value.convert_element_type(DataType::F8E8M0FNU))
            .unwrap();
        assert_eq!(output_tangent, Array::from_elements::<f32>(ArrayType::scalar(DataType::F32), &[3.0]).unwrap());
    }

    #[test]
    fn test_convert_element_type_differentiation_layout() {
        // Conversions between different storage widths discard byte layouts in both primal and tangent outputs.
        let layout = Layout::Strided(StridedLayout::new(vec![1]));
        let laid_out_f32 =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(1)])).with_layout(layout.clone());
        let laid_out_f8 =
            ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![Dimension::Static(1)])).with_layout(layout.clone());
        let plain_f32 = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(1)]));

        let (_, tangent) = differentiate_at(Array::from_elements::<f32>(laid_out_f32.clone(), &[2.0]).unwrap())
            .jvp(Array::from_elements::<f32>(laid_out_f32.clone(), &[3.0]).unwrap(), |value| {
                value.convert_element_type(DataType::F8E8M0FNU)
            })
            .unwrap();
        assert_eq!(tangent, Array::from_elements::<f32>(plain_f32.clone(), &[3.0]).unwrap());

        let (_, tangent) = differentiate_at(
            Array::from_elements::<f8e8m0fnu>(
                laid_out_f8.clone(),
                &[2.0].map(|value| f8e8m0fnu::from_f64(value).unwrap()),
            )
            .unwrap(),
        )
        .jvp(Array::from_elements::<f32>(plain_f32.clone(), &[3.0]).unwrap(), |value| {
            value.convert_element_type(DataType::F32)
        })
        .unwrap();
        assert_eq!(tangent, Array::from_elements::<f32>(plain_f32.clone(), &[3.0]).unwrap());
    }

    #[test]
    fn test_convert_element_type_differentiation_narrowing() {
        // Narrowing real and complex primals also narrows their concrete tangent values.
        let (_, tangent) =
            differentiate_at(Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[2.0]).unwrap())
                .jvp(Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[3.0]).unwrap(), |value| {
                    value.convert_element_type(DataType::F32)
                })
                .unwrap();
        assert_eq!(tangent, Array::from_elements::<f32>(ArrayType::scalar(DataType::F32), &[3.0]).unwrap());

        let (_, tangent) = differentiate_at(
            Array::from_elements::<ComplexNumber<f64>>(
                ArrayType::scalar(DataType::C128),
                &[2.0].map(|value| ComplexNumber::new(value, 0.0)),
            )
            .unwrap(),
        )
        .jvp(
            Array::from_elements::<ComplexNumber<f64>>(
                ArrayType::scalar(DataType::C128),
                &[3.0].map(|value| ComplexNumber::new(value, 0.0)),
            )
            .unwrap(),
            |value| value.convert_element_type(DataType::C64),
        )
        .unwrap();
        assert_eq!(
            tangent,
            Array::from_elements::<ComplexNumber<f32>>(
                ArrayType::scalar(DataType::C64),
                &[3.0].map(|value| ComplexNumber::new(value, 0.0))
            )
            .unwrap(),
        );
    }

    #[test]
    fn test_convert_element_type_differentiation_complex() {
        // Real-to-complex conversion inserts zero imaginary components in both primal and tangent values.
        let (primal, tangent) = differentiate_at(Array::scalar(2.0_f32).unwrap())
            .jvp(Array::scalar(3.0_f32).unwrap(), |value| value.convert_element_type(DataType::C64))
            .unwrap();
        assert_eq!(primal, Array::scalar(ComplexNumber::new(2.0_f32, 0.0)).unwrap());
        assert_eq!(tangent, Array::scalar(ComplexNumber::new(3.0_f32, 0.0)).unwrap());

        // Complex-to-real conversion discards the imaginary primal and tangent components independently.
        let (primal, tangent) = differentiate_at(Array::scalar(ComplexNumber::new(2.0_f32, 5.0)).unwrap())
            .jvp(Array::scalar(ComplexNumber::new(3.0_f32, -7.0)).unwrap(), |value| {
                value.convert_element_type(DataType::F32)
            })
            .unwrap();
        assert_eq!(primal, Array::scalar(2.0_f32).unwrap());
        assert_eq!(tangent, Array::scalar(3.0_f32).unwrap());
    }

    #[test]
    fn test_convert_element_type_differentiation_structural_zero() {
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
        let primal = Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[2.75]).unwrap();
        let tangent = Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[3.0]).unwrap();
        let (output, output_tangent) = differentiate_at(primal)
            .jvp(tangent, |value| value.convert_element_type(DataType::I32)?.convert_element_type(DataType::F64))
            .unwrap();
        assert_eq!(output.r#type().into_owned(), ArrayType::scalar(DataType::F64));
        assert_eq!(output_tangent, Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[0.0]).unwrap());
    }

    #[test]
    fn test_convert_element_type_differentiation_bitcast() {
        // Even identity bit reinterpretation declares a structural zero derivative.
        let (output, tangent) = differentiate_at(Array::scalar(2.0_f32).unwrap())
            .jvp(Array::scalar(3.0_f32).unwrap(), |value| value.bitcast_element_type(DataType::F32))
            .unwrap();
        assert_eq!(output, Array::scalar(2.0_f32).unwrap());
        assert_eq!(tangent, Array::scalar(0.0_f32).unwrap());
        let gradient = differentiate_at(Array::scalar(2.0_f32).unwrap())
            .gradient(|value| value.bitcast_element_type(DataType::F32))
            .unwrap();
        assert_eq!(gradient, Array::scalar(0.0_f32).unwrap());
    }

    #[test]
    fn test_convert_element_type_transposition() {
        check_operation_transposition!(
            @exact,
            operation = ConvertElementTypeOperation::new(DataType::F32, false),
            cases = [
                {
                    inputs = [(@linear(type = ArrayType::scalar(DataType::F64)))],
                    output_cotangents = [
                        Array::from_elements::<f32>(ArrayType::scalar(DataType::F32), &[3.0]).unwrap(),
                    ],
                    input_cotangents = [Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[3.0]).unwrap()],
                },
                {
                    inputs = [(@linear(type = ArrayType::scalar(DataType::I32)))],
                    output_cotangents = [
                        Array::from_elements::<f32>(ArrayType::scalar(DataType::F32), &[3.0]).unwrap(),
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
        let (primal, pullback) = differentiate_at(Array::scalar(2.0_f32).unwrap())
            .vjp(|value| value.convert_element_type(DataType::C64))
            .unwrap();
        assert_eq!(primal, Array::scalar(ComplexNumber::new(2.0_f32, 0.0)).unwrap());
        assert_eq!(
            pullback.apply(Array::scalar(ComplexNumber::new(3.0_f32, -7.0)).unwrap()).unwrap(),
            Array::scalar(3.0_f32).unwrap(),
        );

        // Pulling back through complex-to-real conversion inserts a zero imaginary cotangent.
        let (primal, pullback) = differentiate_at(Array::scalar(ComplexNumber::new(2.0_f32, 5.0)).unwrap())
            .vjp(|value| value.convert_element_type(DataType::F32))
            .unwrap();
        assert_eq!(primal, Array::scalar(2.0_f32).unwrap());
        assert_eq!(
            pullback.apply(Array::scalar(3.0_f32).unwrap()).unwrap(),
            Array::scalar(ComplexNumber::new(3.0_f32, 0.0)).unwrap(),
        );
    }

    #[test]
    fn test_convert_element_type_transposition_structural_zero() {
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
        // Explicit conversion permits narrowing independently of the promotion lattice.
        assert_eq!(
            Array::scalar(2.75_f64).unwrap().convert_element_type(DataType::I32),
            Ok(Array::scalar(2_i32).unwrap()),
        );
        let input = Array::vector(vec![1.0_f64, 2.0]).unwrap();
        assert_eq!(input.convert_element_type(DataType::F64), Ok(input));
    }

    #[test]
    fn test_convert_element_type_convert_element_type_identity() {
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
        assert_eq!(
            Array::scalar(1.0_f32).unwrap().bitcast_element_type(DataType::U32),
            Ok(Array::scalar(0x3f800000_u32).unwrap()),
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
            Array::scalar(2.0_f32).unwrap().promote_element_type(DataType::F64),
            Ok(Array::scalar(2.0_f64).unwrap()),
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
            Array::scalar(2.0_f64).unwrap().promote_element_type(DataType::F32),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "cannot promote type `f64` to type `f32`",
        ));
        assert!(matches!(
            Array::scalar(2.0_f64).unwrap().promote_element_type(DataType::I32),
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
        let value = (1_u64 << 60) + (1_u64 << 52) + 1;
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
        let widened = Array::from_elements(input_type, &[1.5_f32, -2.5])
            .unwrap()
            .convert_element_type(DataType::F64)
            .unwrap();
        assert_eq!(widened.r#type().as_ref(), &ArrayType::new_static(DataType::F64, [2]));
        assert_eq!(widened.elements::<f64>(), Ok(vec![1.5, -2.5]));
        assert_eq!(widened.storage_bytes().len(), 16);

        // Equal-width conversion preserves byte strides, including their direction and unused storage bytes.
        let input_type =
            ArrayType::new_static(DataType::F32, [2]).with_layout(Layout::Strided(StridedLayout::new(vec![-8])));
        let converted = Array::from_elements(input_type.clone(), &[1.5_f32, -2.5])
            .unwrap()
            .convert_element_type(DataType::I32)
            .unwrap();
        assert_eq!(converted.r#type().as_ref(), &input_type.with_data_type(DataType::I32));
        assert_eq!(converted.elements::<i32>(), Ok(vec![1, -2]));
        assert_eq!(converted.storage_bytes().len(), 12);

        // Tiled layouts count elements, so widening preserves their tile geometry and logical element order.
        let input_type = ArrayType::new_static(DataType::F32, [3])
            .with_layout(Layout::Tiled(TiledLayout::new(vec![0], vec![Tile::new(vec![TileDimension::Sized(2)])])));
        let widened = Array::from_elements(input_type.clone(), &[1.5_f32, -2.5, 3.0])
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
        assert!(Arc::ptr_eq(nan.shared_storage(), unchanged.shared_storage()));
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
}
