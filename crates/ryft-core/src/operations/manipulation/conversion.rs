use std::fmt::Display;
use std::marker::PhantomData;

use crate::arrays::{ArrayType, DataType};
use crate::contexts::{Context, Domain};
use crate::differentiation::{DifferentiableType, DifferentiationDual, ElementwiseDerivativeAlignment};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_differentiable_operation};
use crate::operations::ElementwiseOperation;
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{
    MaybeZero, Operation, OperationFormatter, ProgramError, RegionInterface, Type, TypeError, Typed, Value,
};
use crate::tracing::{Tracer, TracingContext};

/// Canonical operation name for [`ConvertElementTypeOperation`].
pub const CONVERT_ELEMENT_TYPE_OPERATION_NAME: &str = "convert_element_type";

/// Unary [`Operation`] that converts each element of a value to a requested [`DataType`], preserving shape, layout,
/// sharding, and memory space. Refer to [`ConvertElementType`] for conversion semantics and [`ElementType`] for the
/// metadata contract. Type inference rejects conversion to or from [`DataType::Token`]; representability of individual
/// values is checked when the conversion executes.
///
/// The `T` parameter fixes the type universe, so each concrete payload instantiation implements exactly one
/// [`Operation`] contract. Array conversion uses the shared elementwise batching rule. Differentiation converts the
/// primal and aligns its tangent with the result's differential type. Transposition aligns cotangents with the input's
/// cotangent type. Types with no tangent or cotangent space produce structural zeros.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ConvertElementTypeOperation<T: ElementType> {
    /// Element [`DataType`] produced by this [`ConvertElementTypeOperation`].
    data_type: DataType,

    /// [`PhantomData`] marker tying this [`Operation`] to the [`Type`] universe in which it is valid.
    marker: PhantomData<fn() -> T>,
}

// TODO(eaplatanios): Review from here onwards.

impl<T: ElementType> ConvertElementTypeOperation<T> {
    /// Creates a conversion to `data_type`. The source element type comes from the operand, and validation takes place
    /// during type inference and execution. Refer to [`ConvertElementType`] for the conversion contract.
    #[inline]
    pub fn new(data_type: DataType) -> Self {
        Self { data_type, marker: PhantomData }
    }

    /// Returns the output element [`DataType`] of this [`ConvertElementTypeOperation`].
    #[inline]
    pub fn data_type(&self) -> DataType {
        self.data_type
    }
}

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

    #[inline]
    fn infer_output_types(
        &self,
        input_types: &[T],
        _region_interfaces: &[RegionInterface<T>],
    ) -> Result<Vec<T>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        if input_types[0].element_type().is_token() || self.data_type.is_token() {
            return Err(TypeError::invalid(format!(
                "cannot convert values to or from the `{}` data type",
                DataType::Token,
            )));
        }
        Ok(vec![input_types[0].with_element_type(self.data_type)])
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?
            .bracketed(|operation| operation.field("data_type", self.data_type))
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
        Ok(vec![inputs[0].convert_element_type(self.data_type)?])
    }
}

impl<C: Context<Type: ElementType, Operation: From<ConvertElementTypeOperation<C::Type>>>>
    PartiallyEvaluatableOperation<C> for ConvertElementTypeOperation<C::Type>
{
}

// Element-type conversion is unary elementwise even though it changes the result data type. Its custom
// inference preserves the input's structure and placement while replacing only that data type. Implementing
// `ElementwiseOperation` also gives it the shared elementwise `BatchableOperation` implementation.
impl ElementwiseOperation for ConvertElementTypeOperation<ArrayType> {
    #[inline]
    fn input_count(&self) -> usize {
        1
    }

    #[inline]
    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        Operation::infer_output_types(self, input_types, &[])
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
            // Convert the primal to the requested element data type and align a live tangent to the resulting
            // differential data type. Converting into a type with no tangent space produces a structural zero tangent.
            check_count!("input", inputs, 1, ProgramError);
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
        |_operation, context, _driver, inputs, outputs, accumulators| {
            // Convert a live output cotangent back to the input's complete cotangent type. Structural zeros remain
            // structural, and an input with no cotangent space receives the structural zero of that space.
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

/// Describes a type's element [`DataType`] independently of its remaining structure and placement metadata. Replacing
/// the element type changes metadata only: it does not convert values, validate their representability, or enforce the
/// promotion lattice. [`ConvertElementTypeOperation`] uses this contract to infer its result type.
///
/// For [`DataType`], replacement returns the requested type itself. For [`ArrayType`], it preserves shape, physical
/// layout, sharding, and memory space while replacing the element data type.
///
/// # Examples
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

    /// Returns a copy with `data_type` as its element type and all other metadata unchanged. This is an unchecked
    /// metadata replacement; use [`ConvertElementType::convert_element_type`] to convert actual values.
    fn with_element_type(&self, data_type: DataType) -> Self;
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
}

impl ElementType for ArrayType {
    #[inline]
    fn element_type(&self) -> DataType {
        self.data_type()
    }

    #[inline]
    fn with_element_type(&self, data_type: DataType) -> Self {
        self.clone().with_data_type(data_type)
    }
}

/// Converts each element of a value to a requested [`DataType`] while preserving its shape, layout, sharding, and
/// memory space. Conversion may narrow precision or change numerical category; it does not reinterpret the input's
/// bytes. [`ConvertElementType::promote_element_type`] additionally checks that the requested conversion is permitted
/// by the type promotion lattice.
///
/// For the reference [`Array`](crate::arrays::Array) backend,
/// [`ArrayElement::convert_to`](crate::arrays::ArrayElement::convert_to) defines per-element rounding, truncation,
/// saturation, and representability checks. Conversion can fail for values unsupported by the destination format.
/// [`Array::converted_to`](crate::arrays::Array::converted_to) documents the handling of same-type conversions, tokens,
/// and structural zeros. Staged operation type inference rejects tokens; value-dependent conversion checks happen
/// during execution.
///
/// [`ConvertElementType`] fills the same role for [`ConvertElementTypeOperation`] that [`std::ops::Add`] and
/// [`std::ops::Neg`] fill for their corresponding arithmetic operations. Traced values bind that operation in their
/// dispatch context; concrete values execute their backend's conversion.
///
/// # Examples
///
/// ```rust
/// # use ryft_core::{Array, ConvertElementType, DataType, ProgramError};
/// #
/// # fn main() -> Result<(), ProgramError> {
/// let input = Array::vector(vec![1.75_f64, -2.5]);
/// let output = input.convert_element_type(DataType::I32)?;
/// assert_eq!(output, Array::vector(vec![1_i32, -2]));
///
/// let input = Array::vector(vec![1.0_f32, 2.0]);
/// assert_eq!(input.promote_element_type(DataType::F64)?, Array::vector(vec![1.0_f64, 2.0]));
/// # Ok(())
/// # }
/// ```
pub trait ConvertElementType: Sized {
    /// Converts each element to `data_type`, preserving the other type metadata. Narrowing and conversions between
    /// numerical categories are allowed, subject to the backend's element conversion rules. The input is unchanged;
    /// the result carries the requested element type. Unsupported conversions return a [`ProgramError`].
    ///
    /// # Parameters
    ///
    ///   - `data_type`: Destination element type. For example, converting `f64` elements to `i32` truncates fractional
    ///     parts toward zero; converting to a lower-precision floating-point format may round the values.
    fn convert_element_type(&self, data_type: DataType) -> Result<Self, ProgramError>;

    /// Converts each element to `data_type` after checking [`DataType::promote_to`]. A conversion outside the promotion
    /// lattice returns a [`TypeError`] before any conversion is dispatched; an accepted conversion delegates to
    /// [`ConvertElementType::convert_element_type`] and can still fail its value-dependent checks. Promotion follows
    /// the lattice's numerical-category rules and does not guarantee exact representation of every source value.
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
        Ok(self
            .dispatch_domain()
            .bind(ConvertElementTypeOperation::new(data_type), Vec::new(), std::slice::from_ref(self))?
            .remove(0))
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayType, DataType, Dimension, Layout, LogicalMesh, Memory, MeshAxis, MeshAxisType, Shape, Sharding,
        ShardingDimension, StridedLayout,
    };
    use crate::contexts::EagerContext;
    use crate::differentiation::differentiate_at;
    use crate::macros::{
        check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_transposition, check_operation_type_inference,
    };
    use crate::programs::{EmptyRegionDriver, Typed};

    use super::*;

    #[test]
    fn test_convert_element_type() {
        // Check operation identity and the requested output element type.
        let array_operation = ConvertElementTypeOperation::<ArrayType>::new(DataType::F32);
        assert_eq!(array_operation.name(), CONVERT_ELEMENT_TYPE_OPERATION_NAME);
        assert_eq!(array_operation.data_type(), DataType::F32);
        assert_eq!(array_operation.to_string(), "convert_element_type [data_type=f32]");
    }

    #[test]
    fn test_convert_element_type_type_inference() {
        let array_operation = ConvertElementTypeOperation::<ArrayType>::new(DataType::F32);

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
                    error = "cannot convert values to or from the token data type",
                },
            ],
        );

        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]))
            .with_layout(Layout::Strided(StridedLayout::new(vec![3, 1])))
            .with_memory(Memory::Host { pinned: true });
        check_operation_type_inference!(
            operation = array_operation,
            cases = [{
                input_types = [input_type.clone()],
                output_types = [input_type.with_data_type(DataType::F32)],
            }],
        );

        check_operation_type_inference!(
            operation = ConvertElementTypeOperation::<ArrayType>::new(DataType::Token),
            cases = [{
                input_types = [ArrayType::scalar(DataType::F64)],
                error = "cannot convert values to or from the token data type",
            }],
        );
    }

    #[test]
    fn test_convert_element_type_type_inference_data_type() {
        check_operation_type_inference!(
            operation = ConvertElementTypeOperation::<DataType>::new(DataType::F32),
            cases = [{
                input_types = [DataType::F64],
                output_types = [DataType::F32],
            }],
        );
    }

    #[test]
    fn test_convert_element_type_interpretation() {
        let output = ConvertElementTypeOperation::<ArrayType>::new(DataType::F32)
            .interpret(&EagerContext::<Array>::new(), &EmptyRegionDriver, &[Array::scalar(2.0_f64)])
            .unwrap();
        assert_eq!(output, vec![Array::scalar(2.0_f32)]);
    }

    #[test]
    fn test_convert_element_type_interpretation_invalid_inputs() {
        let context = EagerContext::<Array>::new();
        let operation = ConvertElementTypeOperation::<ArrayType>::new(DataType::F32);

        // Validate arity before accessing operands or attempting element conversion.
        assert!(matches!(
            operation.interpret(&context, &EmptyRegionDriver, &[]),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        ));
        assert!(matches!(
            operation.interpret(&context, &EmptyRegionDriver, &[Array::scalar(2.0_f64), Array::scalar(3.0_f64)]),
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
                if message == "cannot convert values to or from the token data type",
        ));
        assert!(matches!(
            ConvertElementTypeOperation::<ArrayType>::new(DataType::Token).interpret(
                &context,
                &EmptyRegionDriver,
                &[Array::scalar(2.0_f64)],
            ),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "cannot convert values to or from the token data type",
        ));
    }

    #[test]
    fn test_convert_element_type_partial_evaluation() {
        let array_operation = ConvertElementTypeOperation::<ArrayType>::new(DataType::F32);

        // Known inputs fold to converted values; unknown inputs retain the conversion.
        check_operation_partial_evaluation!(
            operation = array_operation,
            inputs = [Array::scalar(2.0_f64)],
            expected = Array::scalar(2.0_f32),
        );
    }

    #[test]
    fn test_convert_element_type_batching() {
        let array_operation = ConvertElementTypeOperation::<ArrayType>::new(DataType::F32);

        check_operation_batching!(
            @exact,
            operation = array_operation,
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), Array::vector(vec![1.0, 2.0]))],
                outputs = [(
                    @mapped(axis = 0),
                    Array::from_f64s(
                        ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)])),
                        vec![1.0, 2.0],
                    )
                )],
            }],
        );
    }

    #[test]
    fn test_convert_element_type_differentiation() {
        // Widening continuous values preserves their derivatives and agrees with finite differences.
        check_operation_differentiation!(
            @approx(step = 0.125, epsilon = 1e-6),
            operation = ConvertElementTypeOperation::new(DataType::F64),
            cases = [{
                primals = [Array::from_f64s(ArrayType::scalar(DataType::F32), vec![2.0])],
                tangents = [Array::from_f64s(ArrayType::scalar(DataType::F32), vec![2.0])],
                primal_outputs = [Array::from_f64s(ArrayType::scalar(DataType::F64), vec![2.0])],
                tangent_outputs = [Array::from_f64s(ArrayType::scalar(DataType::F64), vec![2.0])],
            }],
        );
    }

    #[test]
    fn test_convert_element_type_differentiation_low_precision() {
        // Low-precision primals use their wider differential representations in both conversion directions.
        let primal = Array::from_f64s(ArrayType::scalar(DataType::F8E8M0FNU), vec![2.0]);
        let tangent = Array::from_f64s(ArrayType::scalar(DataType::F32), vec![3.0]);
        let (output, output_tangent) =
            differentiate_at(primal).jvp(tangent, |value| value.convert_element_type(DataType::F32)).unwrap();
        assert_eq!(output.r#type().into_owned(), ArrayType::scalar(DataType::F32));
        assert_eq!(output_tangent, Array::from_f64s(ArrayType::scalar(DataType::F32), vec![3.0]));

        let primal = Array::from_f64s(ArrayType::scalar(DataType::F32), vec![2.0]);
        let tangent = Array::from_f64s(ArrayType::scalar(DataType::F32), vec![3.0]);
        let (_, output_tangent) = differentiate_at(primal)
            .jvp(tangent, |value| value.convert_element_type(DataType::F8E8M0FNU))
            .unwrap();
        assert_eq!(output_tangent, Array::from_f64s(ArrayType::scalar(DataType::F32), vec![3.0]));
    }

    #[test]
    fn test_convert_element_type_differentiation_layout() {
        // When the differential element representation changes, JVP aligns the complete derivative
        // type: byte-level layout metadata is removed when widening away from `F8E8M0FNU` and restored when returning
        // to a layout-bearing `F32` differential space.
        let layout = Layout::Strided(StridedLayout::new(vec![1]));
        let laid_out_f32 =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(1)])).with_layout(layout.clone());
        let laid_out_f8 =
            ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![Dimension::Static(1)])).with_layout(layout.clone());
        let plain_f32 = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(1)]));

        let (_, tangent) = differentiate_at(Array::from_f64s(laid_out_f32.clone(), vec![2.0]))
            .jvp(Array::from_f64s(laid_out_f32.clone(), vec![3.0]), |value| {
                value.convert_element_type(DataType::F8E8M0FNU)
            })
            .unwrap();
        assert_eq!(tangent, Array::from_f64s(plain_f32.clone(), vec![3.0]));

        let (_, tangent) = differentiate_at(Array::from_f64s(laid_out_f8.clone(), vec![2.0]))
            .jvp(Array::from_f64s(plain_f32.clone(), vec![3.0]), |value| value.convert_element_type(DataType::F32))
            .unwrap();
        assert_eq!(tangent, Array::from_f64s(laid_out_f32.clone(), vec![3.0]));
    }

    #[test]
    fn test_convert_element_type_differentiation_narrowing() {
        // Narrowing real and complex primals also narrows their concrete tangent values.
        let (_, tangent) = differentiate_at(Array::from_f64s(ArrayType::scalar(DataType::F64), vec![2.0]))
            .jvp(Array::from_f64s(ArrayType::scalar(DataType::F64), vec![3.0]), |value| {
                value.convert_element_type(DataType::F32)
            })
            .unwrap();
        assert_eq!(tangent, Array::from_f64s(ArrayType::scalar(DataType::F32), vec![3.0]));

        let (_, tangent) = differentiate_at(Array::from_f64s(ArrayType::scalar(DataType::C128), vec![2.0]))
            .jvp(Array::from_f64s(ArrayType::scalar(DataType::C128), vec![3.0]), |value| {
                value.convert_element_type(DataType::C64)
            })
            .unwrap();
        assert_eq!(tangent, Array::from_f64s(ArrayType::scalar(DataType::C64), vec![3.0]));
    }

    #[test]
    fn test_convert_element_type_differentiation_discrete_intermediate() {
        // Passing through an element type with a zero-dimensional tangent space erases the incoming tangent.
        let primal = Array::from_f64s(ArrayType::scalar(DataType::F64), vec![2.75]);
        let tangent = Array::from_f64s(ArrayType::scalar(DataType::F64), vec![3.0]);
        let (output, output_tangent) = differentiate_at(primal)
            .jvp(tangent, |value| value.convert_element_type(DataType::I32)?.convert_element_type(DataType::F64))
            .unwrap();
        assert_eq!(output.r#type().into_owned(), ArrayType::scalar(DataType::F64));
        assert_eq!(output_tangent, Array::from_f64s(ArrayType::scalar(DataType::F64), vec![0.0]));
    }

    #[test]
    fn test_convert_element_type_transposition() {
        check_operation_transposition!(
            @exact,
            operation = ConvertElementTypeOperation::new(DataType::F32),
            cases = [
                {
                    inputs = [(@linear(type = ArrayType::scalar(DataType::F64)))],
                    output_cotangents = [Array::from_f64s(ArrayType::scalar(DataType::F32), vec![3.0])],
                    input_cotangents = [Array::from_f64s(ArrayType::scalar(DataType::F64), vec![3.0])],
                },
                {
                    inputs = [(@linear(type = ArrayType::scalar(DataType::I32)))],
                    output_cotangents = [Array::from_f64s(ArrayType::scalar(DataType::F32), vec![3.0])],
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
            operation = ConvertElementTypeOperation::new(DataType::F8E8M0FNU),
            cases = [{
                inputs = [(@linear(type = laid_out_f32.clone()))],
                output_cotangents = [Array::from_f64s(plain_f32.clone(), vec![3.0])],
                input_cotangents = [Array::from_f64s(laid_out_f32.clone(), vec![3.0])],
            }],
        );

        check_operation_transposition!(
            @exact,
            operation = ConvertElementTypeOperation::new(DataType::F32),
            cases = [{
                inputs = [(@linear(type = laid_out_f8))],
                output_cotangents = [Array::from_f64s(laid_out_f32, vec![3.0])],
                input_cotangents = [Array::from_f64s(plain_f32, vec![3.0])],
            }],
        );
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
        assert_eq!(input_type.with_element_type(DataType::F32), input_type.clone().with_data_type(DataType::F32));
        assert_eq!(input_type.with_element_type(DataType::F64), input_type);
        assert_eq!(input_type.element_type(), DataType::F64);
    }

    #[test]
    fn test_convert_element_type_convert_element_type() {
        // Explicit conversion permits narrowing independently of the promotion lattice.
        assert_eq!(Array::scalar(2.75_f64).convert_element_type(DataType::I32), Ok(Array::scalar(2_i32)));
        let input = Array::vector(vec![1.0_f64, 2.0]);
        assert_eq!(input.convert_element_type(DataType::F64), Ok(input));
    }

    #[test]
    fn test_convert_element_type_promote_element_type() {
        assert_eq!(Array::scalar(2.0_f32).promote_element_type(DataType::F64), Ok(Array::scalar(2.0_f64)));

        // An already-promoted value retains its complete metadata and element contents.
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)]))
            .with_layout(Layout::Strided(StridedLayout::new(vec![4])))
            .with_memory(Memory::Host { pinned: true });
        let input = Array::from_f64s(input_type, vec![1.0, 2.0]);
        assert_eq!(input.promote_element_type(DataType::F32), Ok(input));
    }

    #[test]
    fn test_convert_element_type_promote_element_type_disallowed() {
        assert!(matches!(
            Array::scalar(2.0_f64).promote_element_type(DataType::F32),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "cannot promote type `f64` to type `f32`",
        ));
        assert!(matches!(
            Array::scalar(2.0_f64).promote_element_type(DataType::I32),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "cannot promote type `f64` to type `i32`",
        ));
    }
}
