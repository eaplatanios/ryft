use ndarray::Array2;
use ryft_core::tracing_v2::DenseJacobian;

/// `ndarray` conversion helpers for dense Jacobian values produced by `ryft-core`.
pub trait DenseJacobianNdArrayExt<S: Clone> {
    /// Converts this dense Jacobian into an [`Array2`] with the same row-major entries.
    fn to_array2(&self) -> Array2<S>;
}

impl<S: Clone, InputStructure, OutputStructure> DenseJacobianNdArrayExt<S>
    for DenseJacobian<S, InputStructure, OutputStructure>
{
    fn to_array2(&self) -> Array2<S> {
        Array2::from_shape_vec((self.rows(), self.cols()), self.values().to_vec())
            .expect("dense Jacobian dimensions should match the stored values")
    }
}

#[cfg(test)]
mod tests {
    use ndarray::arr2;
    use pretty_assertions::assert_eq;
    use ryft_core::ArrayType;
    use ryft_core::tracing::TracingError;
    use ryft_core::tracing_v2::engines::{Engine, TracingEngine};
    use ryft_core::tracing_v2::{DifferentiableEngine, LinearPrimitiveOperation, PrimitiveOperation, Sin, jacfwd};

    use super::DenseJacobianNdArrayExt;

    #[derive(Copy, Clone, Debug)]
    struct ArrayScalarEngine;

    impl Engine for ArrayScalarEngine {
        type Type = ArrayType;
        type Value = f64;

        fn zero(&self, _type: &ArrayType) -> Result<f64, TracingError> {
            Ok(0.0)
        }

        fn one(&self, _type: &ArrayType) -> Result<f64, TracingError> {
            Ok(1.0)
        }
    }

    impl TracingEngine for ArrayScalarEngine {
        type Operation = PrimitiveOperation<f64>;
    }

    impl DifferentiableEngine for ArrayScalarEngine {
        type DifferentiableOperation = PrimitiveOperation<f64>;
        type LinearOperation = LinearPrimitiveOperation<f64>;
    }

    #[test]
    fn test_dense_jacobian_converts_to_array2() {
        let engine = ArrayScalarEngine;
        let jacobian = jacfwd::<ArrayScalarEngine, _, (f64, f64), (f64, f64), f64>(
            &engine,
            |(x, y)| Ok((x.clone() * y.clone() + x.clone().sin(), x + y)),
            (2.0f64, 3.0f64),
        )
        .unwrap();

        assert_eq!(jacobian.to_array2(), arr2(&[[3.0 + 2.0f64.cos(), 2.0], [1.0, 1.0]]));
    }
}
