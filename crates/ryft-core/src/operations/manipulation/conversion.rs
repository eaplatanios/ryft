use std::fmt::Display;

use crate::contexts::{Context, Domain};
use crate::differentiation::forward::DifferentiationDual;
use crate::differentiation::{DifferentiableType, ElementwiseDerivativeAlignment};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_differentiable_operation};
use crate::operations::ElementwiseOperation;
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::ProgramError;
use crate::programs::atoms::MaybeZero;
use crate::programs::operations::{Operation, OperationFormatter};
use crate::programs::regions::RegionInterface;
use crate::programs::types::{Type, TypeError, Typed};
use crate::programs::values::Value;
use crate::tracing::{Tracer, TracingContext};
use crate::types::{ArrayType, DataType};

/// Canonical operation name for [`ConvertElementTypeOperation`].
pub const CONVERT_ELEMENT_TYPE_OPERATION_NAME: &str = "convert_element_type";

/// Unary operation that converts the element [`DataType`] of a value while preserving its shape and placement metadata.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ConvertElementTypeOperation {
    /// Element [`DataType`] produced by this [`ConvertElementTypeOperation`].
    data_type: DataType,
}

impl ConvertElementTypeOperation {
    /// Creates a new [`ConvertElementTypeOperation`].
    #[inline]
    pub fn new(data_type: DataType) -> Self {
        Self { data_type }
    }

    /// Returns the output element [`DataType`] of this [`ConvertElementTypeOperation`].
    #[inline]
    pub fn data_type(&self) -> DataType {
        self.data_type
    }
}

impl Display for ConvertElementTypeOperation {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        OperationFormatter::new(formatter, 0, CONVERT_ELEMENT_TYPE_OPERATION_NAME)?
            .bracketed(|operation| operation.field("data_type", self.data_type))
    }
}

impl<T: ElementType> Operation<T> for ConvertElementTypeOperation {
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
            return Err(TypeError::invalid("cannot convert values to or from the token data type".to_string()));
        }
        Ok(vec![input_types[0].with_element_type(self.data_type)])
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, CONVERT_ELEMENT_TYPE_OPERATION_NAME)?
            .bracketed(|operation| operation.field("data_type", self.data_type))
    }
}

// Element-type conversion is unary elementwise even though it changes the result data type. Its custom
// inference preserves the input's structure and placement while replacing only that data type. Implementing
// `ElementwiseOperation` also gives it the shared elementwise `BatchableOperation` implementation.
impl ElementwiseOperation for ConvertElementTypeOperation {
    #[inline]
    fn input_count(&self) -> usize {
        1
    }

    #[inline]
    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        Operation::infer_output_types(self, input_types, &[])
    }
}

impl<C: Domain<Type: ElementType, Value: ConvertElementType>> InterpretableOperation<C>
    for ConvertElementTypeOperation
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

impl<C: Context<Type: ElementType, Operation: From<ConvertElementTypeOperation>>> PartiallyEvaluatableOperation<C>
    for ConvertElementTypeOperation
{
}

impl_differentiable_operation! {
    ConvertElementTypeOperation,
    jvp<C>
    where
        C::Type: DifferentiableType + ElementType,
        C::Value: ConvertElementType + ElementwiseDerivativeAlignment<C::Type>,
    {
        |operation, _context, _driver, inputs| {
            // Forward-mode differentiation rule for `ConvertElementTypeOperation`. The primal is converted to the
            // requested element data type, while a live tangent is converted to the output's differential element
            // data type. Converting into a type with no tangent space produces a structural zero tangent.
            check_count!("input", inputs, 1, ProgramError);
            let primal = inputs[0].primal().convert_element_type(operation.data_type)?;
            let output_tangent_type = primal.r#type().tangent();
            let tangent = match inputs[0].tangent() {
                _ if output_tangent_type.is_zero_space() => MaybeZero::Zero(output_tangent_type),
                MaybeZero::Zero(_) => MaybeZero::Zero(output_tangent_type),
                MaybeZero::Value(tangent) => MaybeZero::Value(tangent.align_tangent(&output_tangent_type)?),
            };
            Ok(vec![DifferentiationDual::new(primal, tangent)?])
        }
    },
    transpose<V, O>
    where
        V::Type: DifferentiableType + ElementType,
        O: From<ConvertElementTypeOperation>,
        Tracer<TracingContext<V, O>>: ElementwiseDerivativeAlignment<V::Type>,
    {
        |_operation, _context, _driver, inputs, outputs| {
            // Transposition rule for `ConvertElementTypeOperation`. A live output cotangent is converted back to the
            // input's complete cotangent type, while a structural zero remains structural. An input with no cotangent
            // space receives the structural zero of that space.
            check_count!("input", inputs, 1, ProgramError);
            check_count!("output", outputs, 1, ProgramError);
            let input_cotangent_type = inputs[0].r#type().cotangent();
            if input_cotangent_type.is_zero_space() {
                return Ok(vec![MaybeZero::Zero(input_cotangent_type)]);
            }
            Ok(vec![match &outputs[0] {
                MaybeZero::Zero(_) => MaybeZero::Zero(input_cotangent_type),
                MaybeZero::Value(cotangent) => MaybeZero::Value(cotangent.unalign_cotangent(&input_cotangent_type)?),
            }])
        }
    },
}

/// Type whose values have an element [`DataType`] that can be inspected and/or replaced independently of the type's
/// remaining structure and placement metadata. [`ElementType::with_element_type`] is effectively a _casting_ operation.
pub trait ElementType: Type {
    /// Returns the element [`DataType`].
    fn element_type(&self) -> DataType;

    /// Returns this type with its element [`DataType`] replaced by `data_type`.
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
        let mut output = self.clone();
        output.data_type = data_type;
        output
    }
}

/// Value-level capability for converting a value's [`DataType`].
pub trait ConvertElementType: Sized {
    /// Converts this value's elements to `data_type` by _casting_ them.
    ///
    /// # Parameters
    ///
    ///   - `data_type`: Element [`DataType`] of the returned value.
    fn convert_element_type(&self, data_type: DataType) -> Result<Self, ProgramError>;

    /// Promotes this value's elements to `data_type` by _casting_ them. Unlike [`Self::convert_element_type`],
    /// this method rejects conversions that are not permitted by the [`DataType`] promotion lattice.
    ///
    /// # Parameters
    ///
    ///   - `data_type`: Element [`DataType`] to which the returned value is promoted.
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

impl<V: Value<Type: ElementType, DispatchDomain: Context<Operation: From<ConvertElementTypeOperation>>>>
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

    use crate::backends::arrays::Array;
    use crate::backends::scalars::Scalar;
    use crate::differentiation::jvp;
    use crate::macros::{
        check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_transposition, check_operation_type_inference,
    };
    use crate::programs::types::Typed;
    use crate::types::{ArrayType, DataType, Dimension, Layout, Shape, StridedLayout};

    use super::*;

    #[test]
    fn test_convert_element_type() {
        // Check operation metadata and exact inference, including structural array metadata and token rejection.
        let operation = ConvertElementTypeOperation::new(DataType::F32);
        assert_eq!(Operation::<DataType>::name(&operation), CONVERT_ELEMENT_TYPE_OPERATION_NAME);
        assert_eq!(operation.data_type(), DataType::F32);
        assert_eq!(operation.to_string(), "convert_element_type [data_type=f32]");

        check_operation_type_inference!(
            @elementwise @unary,
            operation = operation,
            cases = [
                {
                    input_data_types = [DataType::F64],
                    output_data_types = [DataType::F32],
                },
                {
                    input_data_types = [DataType::Token],
                    error = "cannot convert values to or from the token data type",
                },
            ],
        );

        check_operation_type_inference!(
            operation = ConvertElementTypeOperation::new(DataType::Token),
            cases = [{
                input_types = [DataType::F64],
                error = "cannot convert values to or from the token data type",
            }],
        );

        // Check the default fold-or-residualize rule and preservation of mapped batch placement.
        check_operation_partial_evaluation!(
            operation = operation,
            inputs = [Scalar::from(2.0_f64)],
            expected = Scalar::from(2.0_f32),
        );

        check_operation_batching!(
            @exact,
            operation = operation,
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

        // Check the continuous conversion JVP against finite differences and its reverse conversion of cotangents.
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
                    input_cotangents = [Array::new(ArrayType::scalar(DataType::Zero), vec![Scalar::Zero]).unwrap()],
                },
            ],
        );

        // Low-precision primals use their wider differential representations in both conversion directions.
        let primal = Array::from_f64s(ArrayType::scalar(DataType::F8E8M0FNU), vec![2.0]);
        let tangent = Array::from_f64s(ArrayType::scalar(DataType::F32), vec![3.0]);
        let (output, output_tangent) = jvp(|value| value.convert_element_type(DataType::F32), primal, tangent).unwrap();
        assert_eq!(output.r#type().into_owned(), ArrayType::scalar(DataType::F32));
        assert_eq!(output_tangent, Array::from_f64s(ArrayType::scalar(DataType::F32), vec![3.0]));

        let primal = Array::from_f64s(ArrayType::scalar(DataType::F32), vec![2.0]);
        let tangent = Array::from_f64s(ArrayType::scalar(DataType::F32), vec![3.0]);
        let (_, output_tangent) =
            jvp(|value| value.convert_element_type(DataType::F8E8M0FNU), primal, tangent).unwrap();
        assert_eq!(output_tangent, Array::from_f64s(ArrayType::scalar(DataType::F32), vec![3.0]));

        // When the differential element representation changes, JVP and transposition align the complete derivative
        // type: byte-level layout metadata is removed when widening away from `F8E8M0FNU` and restored when returning
        // to a layout-bearing `F32` differential space.
        let layout = Layout::Strided(StridedLayout::new(vec![1]));
        let laid_out_f32 =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(1)])).with_layout(layout.clone());
        let laid_out_f8 =
            ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![Dimension::Static(1)])).with_layout(layout.clone());
        let plain_f32 = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(1)]));

        let (_, tangent) = jvp(
            |value| value.convert_element_type(DataType::F8E8M0FNU),
            Array::from_f64s(laid_out_f32.clone(), vec![2.0]),
            Array::from_f64s(laid_out_f32.clone(), vec![3.0]),
        )
        .unwrap();
        assert_eq!(tangent, Array::from_f64s(plain_f32.clone(), vec![3.0]));

        let (_, tangent) = jvp(
            |value| value.convert_element_type(DataType::F32),
            Array::from_f64s(laid_out_f8.clone(), vec![2.0]),
            Array::from_f64s(plain_f32.clone(), vec![3.0]),
        )
        .unwrap();
        assert_eq!(tangent, Array::from_f64s(laid_out_f32.clone(), vec![3.0]));

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

        // Narrowing real and complex primals also narrows their concrete tangent values.
        let (_, tangent) = jvp(
            |value| value.convert_element_type(DataType::F32),
            Array::from_f64s(ArrayType::scalar(DataType::F64), vec![2.0]),
            Array::from_f64s(ArrayType::scalar(DataType::F64), vec![3.0]),
        )
        .unwrap();
        assert_eq!(tangent, Array::from_f64s(ArrayType::scalar(DataType::F32), vec![3.0]));

        let (_, tangent) = jvp(
            |value| value.convert_element_type(DataType::C64),
            Array::from_f64s(ArrayType::scalar(DataType::C128), vec![2.0]),
            Array::from_f64s(ArrayType::scalar(DataType::C128), vec![3.0]),
        )
        .unwrap();
        assert_eq!(tangent, Array::from_f64s(ArrayType::scalar(DataType::C64), vec![3.0]));

        // Passing through an element type with a zero-dimensional tangent space erases the incoming tangent.
        let primal = Array::from_f64s(ArrayType::scalar(DataType::F64), vec![2.75]);
        let tangent = Array::from_f64s(ArrayType::scalar(DataType::F64), vec![3.0]);
        let (output, output_tangent) = jvp(
            |value| value.convert_element_type(DataType::I32)?.convert_element_type(DataType::F64),
            primal,
            tangent,
        )
        .unwrap();
        assert_eq!(output.r#type().into_owned(), ArrayType::scalar(DataType::F64));
        assert_eq!(output_tangent, Array::from_f64s(ArrayType::scalar(DataType::F64), vec![0.0]));
    }
}
