use std::marker::PhantomData;

use ryft_core::tracing::TracingError;
use ryft_core::tracing::engines::{Engine, Tracer, TracingEngine};
use ryft_core::tracing_v2::{DifferentiableEngine, DifferentiableTracingEngine, LinearizableEngine};
use ryft_core::types::{ArrayType, TypeError};

use crate::arrays::{Array, NdArrayElement};
use crate::operations::{LinearNdarrayOperation, NdarrayOperation};

/// Stateless `ndarray` backend token for `ryft-core` tracing transforms.
///
/// [`NdArrayEngine`] selects [`ArrayType`] as its abstract metadata, [`Array<T>`] as its concrete
/// value, and the backend-owned ndarray operation carriers. It has no device or runtime state
/// because all execution happens eagerly on host CPU buffers.
#[derive(Copy, Clone, Debug, Default)]
pub struct NdArrayEngine<T = f64> {
    /// Phantom marker tying the zero-sized engine to its element type.
    marker: PhantomData<fn() -> T>,
}

impl<T> NdArrayEngine<T> {
    /// Returns a new [`NdArrayEngine`].
    #[inline]
    pub const fn new() -> Self {
        Self { marker: PhantomData }
    }
}

impl<T: NdArrayElement> Engine for NdArrayEngine<T> {
    type Type = ArrayType;
    type Value = Array<T>;

    fn zero(&self, array_type: &ArrayType) -> Result<Self::Value, TracingError> {
        Array::zeros(array_type).map_err(array_error_to_tracing_error)
    }

    fn one(&self, array_type: &ArrayType) -> Result<Self::Value, TracingError> {
        Array::ones(array_type).map_err(array_error_to_tracing_error)
    }
}

impl<T: NdArrayElement> TracingEngine for NdArrayEngine<T> {
    type OperationCarrier = NdarrayOperation<Array<T>>;
}

impl<T: NdArrayElement> LinearizableEngine for NdArrayEngine<T> {
    type LinearOperationCarrier = LinearNdarrayOperation<Array<T>>;
}

impl<T: NdArrayElement> DifferentiableEngine for NdArrayEngine<T> {
    type DifferentiableOperationCarrier = NdarrayOperation<Array<T>>;
}

impl<T: NdArrayElement> DifferentiableTracingEngine for NdArrayEngine<T> {
    type LinearOperationCarrier<'engine>
        = LinearNdarrayOperation<Tracer<'engine, Self>>
    where
        Self: 'engine;
}

fn array_error_to_tracing_error(error: crate::arrays::ArrayError) -> TracingError {
    TypeError { message: error.to_string() }.into()
}

#[cfg(test)]
mod tests {
    use ndarray::{arr1, arr2};
    use pretty_assertions::assert_eq;
    use ryft_core::operations::Operation;
    use ryft_core::operations::arithmetic::ADD_OPERATION_NAME;
    use ryft_core::tracing::TracingError;
    use ryft_core::tracing::engines::{Engine, TracingEngine};
    use ryft_core::tracing_v2::{DifferentiationError, MatrixOps, Sin, compile_grad, grad, jvp};
    use ryft_core::types::{ArrayType, DataType, Shape, Size};

    use crate::Array;

    use super::NdArrayEngine;

    #[test]
    fn test_engine_synthesizes_zero_and_one_arrays() {
        let engine = NdArrayEngine::<f64>::new();
        let array_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(2)]), None, None).unwrap();

        let zero = engine.zero(&array_type).unwrap();
        let one = engine.one(&array_type).unwrap();

        assert_eq!(zero.as_ndarray().iter().copied().collect::<Vec<_>>(), vec![0.0, 0.0, 0.0, 0.0]);
        assert_eq!(one.as_ndarray().iter().copied().collect::<Vec<_>>(), vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_engine_rejects_dynamic_shapes() {
        let engine = NdArrayEngine::<f64>::new();
        let array_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Dynamic(None)]), None, None).unwrap();

        let error = engine.zero(&array_type).unwrap_err();

        assert_eq!(error.to_string(), "ndarray backend requires static shape dimensions, but dimension #0 is *");
    }

    #[test]
    fn test_trace_and_program_interpret_use_ndarray_values() {
        let engine = NdArrayEngine::<f64>::new();
        let input = Array::from_shape_vec([3], vec![1.0, 2.0, 3.0]).unwrap();

        let (output, program): (Array<f64>, _) =
            engine.interpret_and_trace(|x| Ok((x.clone() * x).sin()), input.clone()).unwrap();
        let replayed: Array<f64> = program.interpret(input).unwrap();

        assert_eq!(output.as_ndarray(), &arr1(&[1.0f64.sin(), 4.0f64.sin(), 9.0f64.sin()]).into_dyn());
        assert_eq!(replayed, output);
        assert_eq!(program.instructions.len(), 2);
    }

    #[test]
    fn test_symbolic_trace_records_ndarray_carrier() {
        let engine = NdArrayEngine::<f64>::new();
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2)]), None, None).unwrap();

        let (output_type, program): (ArrayType, _) = engine.trace(|x| Ok(x.clone() + x), input_type.clone()).unwrap();

        assert_eq!(output_type, input_type);
        assert_eq!(program.instructions.len(), 1);
        assert_eq!(program.instructions[0].operation.name(), ADD_OPERATION_NAME);
    }

    #[test]
    fn test_jvp_over_ndarray_values() {
        let engine = NdArrayEngine::<f64>::new();
        let primal = Array::from_shape_vec([2], vec![2.0, 3.0]).unwrap();
        let tangent = Array::from_shape_vec([2], vec![5.0, 7.0]).unwrap();

        let (primal_output, tangent_output): (Array<f64>, Array<f64>) =
            jvp(&engine, |x| x.clone() * x, primal, tangent).unwrap();

        assert_eq!(primal_output.as_ndarray(), &arr1(&[4.0, 9.0]).into_dyn());
        assert_eq!(tangent_output.as_ndarray(), &arr1(&[20.0, 42.0]).into_dyn());
    }

    #[test]
    fn test_grad_rejects_non_scalar_array_output() {
        let engine = NdArrayEngine::<f64>::new();
        let input = Array::from_shape_vec([2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let result = grad(&engine, |input| input, input);

        assert!(matches!(
            result,
            Err(TracingError::Differentiation(DifferentiationError::NonScalarGradientOutput { output_type }))
                if output_type.rank() == 2
        ));
    }

    #[test]
    fn test_compile_grad_rejects_non_scalar_array_output() {
        let engine = NdArrayEngine::<f64>::new();
        let input = Array::from_shape_vec([2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let result = compile_grad(&engine, |input| input, input);

        assert!(matches!(
            result,
            Err(TracingError::Differentiation(DifferentiationError::NonScalarGradientOutput { output_type }))
                if output_type.rank() == 2
        ));
    }

    #[test]
    fn test_matrix_program_interpret() {
        let engine = NdArrayEngine::<f64>::new();
        let left = Array::from_shape_vec([2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let right = Array::from_shape_vec([2, 2], vec![5.0, 6.0, 7.0, 8.0]).unwrap();

        let (output, _program): (Array<f64>, _) =
            engine.interpret_and_trace(|(left, right)| Ok(left.matmul(right)), (left, right)).unwrap();

        assert_eq!(output.as_ndarray(), &arr2(&[[19.0, 22.0], [43.0, 50.0]]).into_dyn());
    }
}
