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
    use ryft_core::tracing_v2::{DifferentiableEngine, Sin};

    use super::DenseJacobianNdArrayExt;
    use crate::{Array, NdArrayEngine};

    #[test]
    fn test_dense_jacobian_converts_to_array2() {
        let engine = NdArrayEngine::<f64>::new();
        let jacobian = engine
            .jacfwd::<_, (Array<f64>, Array<f64>), (Array<f64>, Array<f64>), Array<f64>>(
                |(x, y)| Ok((x.clone() * y.clone() + x.clone().sin(), x + y)),
                (Array::scalar(2.0), Array::scalar(3.0)),
            )
            .unwrap();

        assert_eq!(jacobian.to_array2(), arr2(&[[3.0 + 2.0f64.cos(), 2.0], [1.0, 1.0]]));
    }

    #[test]
    fn test_hessian_accepts_original_scalar_function() {
        let engine = NdArrayEngine::<f64>::new();
        let hessian = engine
            .hessian::<_, Array<f64>, Array<f64>>(|x| x.clone() * x.clone() * x, Array::scalar(2.0))
            .unwrap();

        assert_eq!(hessian.to_array2(), arr2(&[[12.0]]));
    }

    #[test]
    fn test_hessian_preserves_structured_input_coordinates() {
        let engine = NdArrayEngine::<f64>::new();
        let hessian = engine
            .hessian::<_, (Array<f64>, Array<f64>), Array<f64>>(
                |(x, y)| x.clone() * y + x.sin(),
                (Array::scalar(2.0), Array::scalar(3.0)),
            )
            .unwrap();

        assert_eq!(hessian.to_array2(), arr2(&[[-2.0f64.sin(), 1.0], [1.0, 0.0]]));
    }
}
