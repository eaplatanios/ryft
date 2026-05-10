use std::marker::PhantomData;

use ryft_core::tracing::TracingError;
use ryft_core::tracing::domains::{Domain, RuntimeDomain, Tracer, TracingDomain};
use ryft_core::tracing_v2::{DifferentiableDomain, DifferentiableTracingDomain};
use ryft_core::types::{ArrayType, TypeError};

use crate::arrays::{Array, NdArrayElement};
use crate::operations::{LinearNdarrayOperation, NdarrayOperation};

/// Stateless `ndarray` domain token for `ryft-core` tracing transforms.
///
/// [`NdArrayDomain`] selects [`ArrayType`] as its abstract metadata, [`Array<T>`] as its concrete
/// value, and the backend-owned ndarray operation carriers. It has no device or runtime state
/// because all execution happens eagerly on host CPU buffers.
#[derive(Copy, Clone, Debug, Default)]
pub struct NdArrayDomain<T = f64> {
    /// Linear ndarray domain used by automatic-differentiation transforms.
    linear_domain: NdArrayLinearDomain<T>,

    /// Phantom marker tying the zero-sized domain to its element type.
    marker: PhantomData<fn() -> T>,
}

impl<T> NdArrayDomain<T> {
    /// Returns a new [`NdArrayDomain`].
    #[inline]
    pub const fn new() -> Self {
        Self { linear_domain: NdArrayLinearDomain::new(), marker: PhantomData }
    }
}

impl<T: NdArrayElement> Domain for NdArrayDomain<T> {
    type Type = ArrayType;
    type Value = Array<T>;
}

impl<T: NdArrayElement> RuntimeDomain for NdArrayDomain<T> {
    fn zero(&self, array_type: &ArrayType) -> Result<Self::Value, TracingError> {
        Array::zeros(array_type).map_err(array_error_to_tracing_error)
    }

    fn one(&self, array_type: &ArrayType) -> Result<Self::Value, TracingError> {
        Array::ones(array_type).map_err(array_error_to_tracing_error)
    }
}

impl<T: NdArrayElement> TracingDomain for NdArrayDomain<T> {
    type OperationCarrier = NdarrayOperation<Array<T>>;
}

/// Stateless `ndarray` domain token for linear tangent and cotangent programs.
#[derive(Copy, Clone, Debug, Default)]
pub struct NdArrayLinearDomain<T = f64> {
    /// Phantom marker tying the zero-sized linear domain to its element type.
    marker: PhantomData<fn() -> T>,
}

impl<T> NdArrayLinearDomain<T> {
    /// Returns a new [`NdArrayLinearDomain`].
    #[inline]
    pub const fn new() -> Self {
        Self { marker: PhantomData }
    }
}

impl<T: NdArrayElement> Domain for NdArrayLinearDomain<T> {
    type Type = ArrayType;
    type Value = Array<T>;
}

impl<T: NdArrayElement> RuntimeDomain for NdArrayLinearDomain<T> {
    fn zero(&self, array_type: &ArrayType) -> Result<Self::Value, TracingError> {
        Array::zeros(array_type).map_err(array_error_to_tracing_error)
    }

    fn one(&self, array_type: &ArrayType) -> Result<Self::Value, TracingError> {
        Array::ones(array_type).map_err(array_error_to_tracing_error)
    }
}

impl<T: NdArrayElement> TracingDomain for NdArrayLinearDomain<T> {
    type OperationCarrier = LinearNdarrayOperation<Array<T>>;
}

impl<T: NdArrayElement> DifferentiableDomain for NdArrayDomain<T> {
    type Tangent = Array<T>;
    type LinearDomain = NdArrayLinearDomain<T>;
    type LinearOperationCarrier = LinearNdarrayOperation<Array<T>>;
    type DifferentiableOperationCarrier = NdarrayOperation<Array<T>>;

    #[inline]
    fn linear_domain(&self) -> &Self::LinearDomain {
        &self.linear_domain
    }
}

impl<T: NdArrayElement> DifferentiableTracingDomain for NdArrayDomain<T> {
    type LinearOperationCarrier<'domain>
        = LinearNdarrayOperation<Tracer<'domain, NdArrayDomain<T>>>
    where
        Self: 'domain;
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
    use ryft_core::tracing::domains::{RuntimeDomain, TracingDomain};
    use ryft_core::tracing_v2::{DifferentiableDomain, DifferentiationError, MatMul, Sin};
    use ryft_core::types::{ArrayType, DataType, Shape, Size};

    use crate::Array;

    use super::NdArrayDomain;

    #[test]
    fn test_domain_synthesizes_zero_and_one_arrays() {
        let domain = NdArrayDomain::<f64>::new();
        let array_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(2)]), None, None).unwrap();

        let zero = domain.zero(&array_type).unwrap();
        let one = domain.one(&array_type).unwrap();

        assert_eq!(zero.as_ndarray().iter().copied().collect::<Vec<_>>(), vec![0.0, 0.0, 0.0, 0.0]);
        assert_eq!(one.as_ndarray().iter().copied().collect::<Vec<_>>(), vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_domain_rejects_dynamic_shapes() {
        let domain = NdArrayDomain::<f64>::new();
        let array_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Dynamic(None)]), None, None).unwrap();

        let error = domain.zero(&array_type).unwrap_err();

        assert_eq!(error.to_string(), "ndarray backend requires static shape dimensions, but dimension #0 is *");
    }

    #[test]
    fn test_trace_and_program_interpret_use_ndarray_values() {
        let domain = NdArrayDomain::<f64>::new();
        let input = Array::from_shape_vec([3], vec![1.0, 2.0, 3.0]).unwrap();

        let (output, program): (Array<f64>, _) =
            domain.interpret_and_trace(|x| Ok((x.clone() * x).sin()), input.clone()).unwrap();
        let replayed: Array<f64> = program.interpret(input).unwrap();

        assert_eq!(output.as_ndarray(), &arr1(&[1.0f64.sin(), 4.0f64.sin(), 9.0f64.sin()]).into_dyn());
        assert_eq!(replayed, output);
        assert_eq!(program.instructions.len(), 2);
    }

    #[test]
    fn test_symbolic_trace_records_ndarray_carrier() {
        let domain = NdArrayDomain::<f64>::new();
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2)]), None, None).unwrap();

        let (output_type, program): (ArrayType, _) = domain.trace(|x| Ok(x.clone() + x), input_type.clone()).unwrap();

        assert_eq!(output_type, input_type);
        assert_eq!(program.instructions.len(), 1);
        assert_eq!(program.instructions[0].operation.name(), ADD_OPERATION_NAME);
    }

    #[test]
    fn test_jvp_over_ndarray_values() {
        let domain = NdArrayDomain::<f64>::new();
        let primal = Array::from_shape_vec([2], vec![2.0, 3.0]).unwrap();
        let tangent = Array::from_shape_vec([2], vec![5.0, 7.0]).unwrap();

        let (primal_output, tangent_output): (Array<f64>, Array<f64>) =
            domain.jvp(|x| x.clone() * x, primal, tangent).unwrap();

        assert_eq!(primal_output.as_ndarray(), &arr1(&[4.0, 9.0]).into_dyn());
        assert_eq!(tangent_output.as_ndarray(), &arr1(&[20.0, 42.0]).into_dyn());
    }

    #[test]
    fn test_grad_rejects_non_scalar_array_output() {
        let domain = NdArrayDomain::<f64>::new();
        let input = Array::from_shape_vec([2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let result = domain.grad(|input| input, input);

        assert!(matches!(
            result,
            Err(TracingError::Differentiation(DifferentiationError::NonScalarGradientOutput { output_type }))
                if output_type.rank() == 2
        ));
    }

    #[test]
    fn test_matrix_program_interpret() {
        let domain = NdArrayDomain::<f64>::new();
        let left = Array::from_shape_vec([2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let right = Array::from_shape_vec([2, 2], vec![5.0, 6.0, 7.0, 8.0]).unwrap();

        let (output, _program): (Array<f64>, _) =
            domain.interpret_and_trace(|(left, right)| Ok(left.matmul(right)), (left, right)).unwrap();

        assert_eq!(output.as_ndarray(), &arr2(&[[19.0, 22.0], [43.0, 50.0]]).into_dyn());
    }
}
