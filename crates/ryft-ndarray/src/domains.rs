use std::marker::PhantomData;

use ryft_core::EagerContext;
use ryft_core::contexts::{Context, ProvidesContext};
use ryft_core::domains::Domain;
use ryft_core::operations::InterpretableOperation;
use ryft_core::programs::{ProgramError, Value};
use ryft_core::tracing_v2::DifferentiationContext;
use ryft_core::types::ArrayType;

use crate::arrays::{Array, NdArrayElement};
use crate::operations::{LinearNdarrayOperation, NdarrayOperation};

/// Stateless `ndarray` domain token for `ryft-core` tracing transforms.
///
/// [`NdArrayDomain`] selects [`ArrayType`] as its abstract metadata, [`Array<T>`] as its concrete
/// value, and the backend-owned ndarray operation types. It has no device or runtime state
/// because all execution happens eagerly on host CPU buffers.
#[derive(Copy, Clone, Debug, Default)]
pub struct NdArrayDomain<T = f64> {
    /// Phantom marker tying the zero-sized domain to its element type.
    marker: PhantomData<fn() -> T>,
}

impl<T> NdArrayDomain<T> {
    /// Returns a new [`NdArrayDomain`].
    #[inline]
    pub const fn new() -> Self {
        Self { marker: PhantomData }
    }
}

impl<T: NdArrayElement> Domain for NdArrayDomain<T> {
    type Type = ArrayType;
    type Value = Array<T>;
    type Constant = Array<T>;
    type Operation = NdarrayOperation<Array<T>>;
}

impl<T: NdArrayElement> Context for NdArrayDomain<T> {
    #[inline]
    fn lift(&self, constant: Array<T>) -> Result<Array<T>, ProgramError> {
        Ok(constant)
    }

    fn bind<P: Into<Self::Operation>>(
        &self,
        operation: P,
        inputs: &[Self::Value],
    ) -> Result<Vec<Self::Value>, ProgramError> {
        let operation = operation.into();
        operation.interpret(&EagerContext::new(), inputs)
    }
}

impl<T: NdArrayElement> DifferentiationContext for NdArrayDomain<T> {
    type Tangent = Array<T>;
    type LinearOperation<V: Value<ArrayType>, F: Value<ArrayType>> = LinearNdarrayOperation<V, Array<T>, F>;
}

impl<T: NdArrayElement> ProvidesContext<<Array<T> as Value<ArrayType>>::InterpretationContext> for NdArrayDomain<T> {
    #[inline]
    fn context(&self) -> <Array<T> as Value<ArrayType>>::InterpretationContext {
        EagerContext::new()
    }
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
    type Constant = Array<T>;
    type Operation = LinearNdarrayOperation<Array<T>>;
}

impl<T: NdArrayElement> Context for NdArrayLinearDomain<T> {
    #[inline]
    fn lift(&self, constant: Array<T>) -> Result<Array<T>, ProgramError> {
        Ok(constant)
    }

    fn bind<P: Into<Self::Operation>>(
        &self,
        operation: P,
        inputs: &[Self::Value],
    ) -> Result<Vec<Self::Value>, ProgramError> {
        let operation = operation.into();
        operation.interpret(&EagerContext::new(), inputs)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{arr1, arr2};
    use pretty_assertions::assert_eq;
    use ryft_core::contexts::Context;
    use ryft_core::operations::Operation;
    use ryft_core::operations::arithmetic::ADD_OPERATION_NAME;
    use ryft_core::operations::constants::{OneOperation, ZeroOperation};
    use ryft_core::tracing::TracingContext;
    use ryft_core::tracing_v2::operations::dot::{Dot, DotDimensionNumbers};
    use ryft_core::tracing_v2::{DifferentiableDomainExtension, DifferentiationContext, DifferentiationError, Sin};
    use ryft_core::types::{ArrayType, DataType, Shape, Size};

    use crate::Array;

    use super::NdArrayDomain;

    #[test]
    fn test_domain_synthesizes_zero_and_one_arrays() {
        let domain = NdArrayDomain::<f64>::new();
        let array_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(2)]));
        let scalar_type = ArrayType::new(DataType::F64, Shape::new(vec![]));

        let zero = domain.bind(ZeroOperation::new(array_type), &[]).unwrap().into_iter().next().unwrap();
        let one = domain.bind(OneOperation::new(scalar_type), &[]).unwrap().into_iter().next().unwrap();

        assert_eq!(zero.as_ndarray().iter().copied().collect::<Vec<_>>(), vec![0.0, 0.0, 0.0, 0.0]);
        assert_eq!(one.as_ndarray().iter().copied().collect::<Vec<_>>(), vec![1.0]);
    }

    #[test]
    fn test_domain_rejects_dynamic_shapes() {
        let domain = NdArrayDomain::<f64>::new();
        let array_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Dynamic(None)]));

        let error = domain.bind(ZeroOperation::new(array_type), &[]).unwrap_err();

        assert_eq!(error.to_string(), "ndarray backend requires static shape dimensions, but dimension #0 is *");
    }

    #[test]
    fn test_trace_and_program_interpret_use_ndarray_values() {
        let domain = NdArrayDomain::<f64>::new();
        let input = Array::from_shape_vec([3], vec![1.0, 2.0, 3.0]).unwrap();

        let (output, program): (Array<f64>, _) =
            TracingContext::interpret_and_trace(&domain, |x| Ok((x.clone() * x).sin()), input.clone()).unwrap();
        let replayed: Array<f64> = program.interpret(input).unwrap();

        assert_eq!(output.as_ndarray(), &arr1(&[1.0f64.sin(), 4.0f64.sin(), 9.0f64.sin()]).into_dyn());
        assert_eq!(replayed, output);
        assert_eq!(program.instructions().len(), 2);
    }

    #[test]
    fn test_symbolic_trace_records_ndarray_operation() {
        let domain = NdArrayDomain::<f64>::new();
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2)]));

        let (output_type, program): (ArrayType, _) =
            TracingContext::trace(&domain, |x| Ok(x.clone() + x), input_type.clone()).unwrap();

        assert_eq!(output_type, input_type);
        assert_eq!(program.instructions().len(), 1);
        assert_eq!(program.instructions()[0].operation().name(), ADD_OPERATION_NAME);
    }

    #[test]
    fn test_jacfwd_over_ndarray_vector_value() {
        // f(x: [3]) = x*x — Jacobian is a 3x3 diagonal matrix with 2*x_i on the diagonal.
        let domain = NdArrayDomain::<f64>::new();
        let input = Array::from_shape_vec([3], vec![1.0, 2.0, 3.0]).unwrap();

        let jacobian = domain.jacfwd(|x| Ok(x.clone() * x), input).unwrap();

        let blocks = jacobian.iter_blocks().collect::<Vec<_>>();
        assert_eq!(blocks.len(), 1);
        let (_, _, block) = blocks[0];
        assert_eq!(block.output_shape(), &[3]);
        assert_eq!(block.input_shape(), &[3]);

        // Diagonal Jacobian: 2*x_i on the diagonal, 0 off-diagonal.
        let values = block.values();
        for row in 0..3 {
            for col in 0..3 {
                let expected = if row == col { 2.0 * (row as f64 + 1.0) } else { 0.0 };
                let actual = values[row * 3 + col];
                assert!((actual - expected).abs() < 1e-9, "values[{row}, {col}] = {actual}, expected {expected}");
            }
        }
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
    fn test_value_and_gradient_rejects_non_scalar_array_output() {
        let domain = NdArrayDomain::<f64>::new();
        let input = Array::from_shape_vec([2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let result = domain.value_and_gradient(|input| input, input);

        assert!(
            matches!(result, Err(DifferentiationError::NonScalarGradientOutput { .. })),
            "expected a non-scalar gradient-output rejection but got {result:?}",
        );
    }

    #[test]
    fn test_matrix_program_interpret() {
        let domain = NdArrayDomain::<f64>::new();
        let left = Array::from_shape_vec([2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let right = Array::from_shape_vec([2, 2], vec![5.0, 6.0, 7.0, 8.0]).unwrap();

        let (output, _program): (Array<f64>, _) = TracingContext::interpret_and_trace(
            &domain,
            |(left, right)| Ok(left.dot(&right, &DotDimensionNumbers::matmul())),
            (left, right),
        )
        .unwrap();

        assert_eq!(output.as_ndarray(), &arr2(&[[19.0, 22.0], [43.0, 50.0]]).into_dyn());
    }
}
