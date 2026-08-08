use std::ops::{Add as StandardAdd, Div as StandardDiv, Mul as StandardMul, Neg as StandardNeg};

use crate::macros::{
    define_elementwise_capability, define_elementwise_operation, impl_differentiable_elementwise_operation,
};
use crate::programs::ProgramError;

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`RsqrtOperation`].
pub const RSQRT_OPERATION_NAME: &str = "rsqrt";

define_elementwise_operation!(
    @unary
    /// [`Operation`] that computes the elementwise reciprocal square root of one value (i.e., `x ↦ 1/√x`, the
    /// principal branch `1/√z` on complex operands) while preserving its array metadata. Only floating-point and
    /// complex operands are supported, and operands that still carry partial sums are rejected.
    RsqrtOperation, RSQRT_OPERATION_NAME,
    Rsqrt, rsqrt,
    check_data_types = [@float],
    check_array_types = [@no_unreduced],
);

impl_differentiable_elementwise_operation! {
    @unary
    RsqrtOperation,
    jvp<C>
    where
        C::Value: StandardAdd<Output = C::Value>
            + StandardDiv<Output = C::Value>
            + StandardMul<Output = C::Value>
            + StandardNeg<Output = C::Value>,
    {
        // d(rsqrt(x)) = -x^{-3/2} / 2 · dx = -(rsqrt(x) / (x + x)) · dx, reusing the primal output evaluated at
        // the tangent type.
        |(input, input_tangent) -> output| -(output / (input.clone() + input)) * input_tangent
    },
    transpose = @nonlinear,
}

define_elementwise_capability!(
    @unary
    /// Value-level elementwise reciprocal-square-root capability. [`Rsqrt`] fills the same role for
    /// [`RsqrtOperation`] that [`Sin`](crate::Sin) fills for [`SinOperation`](crate::SinOperation).
    Rsqrt,
    /// Computes [`RsqrtOperation`] elementwise for this value.
    rsqrt,
    RsqrtOperation,
);

/// Implements [`Rsqrt`] for one host primitive type.
macro_rules! impl_capability_for_primitive {
    ($type:ty) => {
        impl Rsqrt for $type {
            fn rsqrt(&self) -> Result<Self, ProgramError> {
                Ok(<$type>::sqrt(*self).recip())
            }
        }
    };
}

impl_capability_for_primitive!(f32);
impl_capability_for_primitive!(f64);

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use half::{bf16, f16};
    use num_complex::Complex as ComplexNumber;
    use pretty_assertions::assert_eq;

    use crate::arrays::{Array, ArrayType, DataType};
    use crate::differentiation::gradient_holomorphic;
    use crate::macros::{
        check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_transposition, check_operation_type_inference,
    };

    use super::*;

    #[test]
    fn test_rsqrt() {
        assert_eq!(Array::scalar(0.5f32).rsqrt().unwrap(), Array::scalar(1.0 / 0.5f32.sqrt()));
        assert_eq!(Array::scalar(0.5f64).rsqrt().unwrap(), Array::scalar(1.0 / 0.5f64.sqrt()));
        assert_eq!(
            Array::scalar(bf16::from_f32(0.5)).rsqrt().unwrap(),
            Array::scalar(bf16::from_f32(1.0 / 0.5f32.sqrt())),
        );
        assert_eq!(
            Array::scalar(f16::from_f32(0.5)).rsqrt().unwrap(),
            Array::scalar(f16::from_f32(1.0 / 0.5f32.sqrt())),
        );
        let input = ComplexNumber::new(0.7f64, -0.3f64);
        let expected = ComplexNumber::new(1.0, 0.0) / input.sqrt();
        assert_abs_diff_eq!(Array::scalar(input).rsqrt().unwrap(), Array::scalar(expected), epsilon = 1e-12);

        assert_eq!(Array::scalar(4.0).rsqrt().unwrap(), Array::scalar(0.5),);
    }

    #[test]
    fn test_rsqrt_type_inference() {
        check_operation_type_inference!(
            @elementwise @unary,
            operation = RsqrtOperation,
            cases = [
                {
                    input_data_types = [DataType::C64],
                    output_data_types = [DataType::C64],
                },
                {
                    input_data_types = [DataType::I32],
                    error = "'rsqrt' does not support input data type i32",
                },
            ],
        );
        check_operation_type_inference!(
            @reject @unreduced,
            operation = RsqrtOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_rsqrt_batching() {
        check_operation_batching!(
            @approx(epsilon = 1e-9),
            operation = RsqrtOperation::new(),
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), Array::vector(vec![0.5, 4.0]))],
                outputs = [(@mapped(axis = 0), Array::vector(vec![1.0 / 0.5f64.sqrt(), 0.5]))],
            }],
        );
    }

    #[test]
    fn test_rsqrt_differentiation() {
        let expected_tangent = 3.0 * -(1.0 / (2.0 * 4.0f64.powf(1.5)));
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = RsqrtOperation::new(),
            cases = [{
                primals = [Array::scalar(4.0)],
                tangents = [Array::scalar(3.0)],
                primal_outputs = [Array::scalar(0.5)],
                tangent_outputs = [Array::scalar(expected_tangent)],
            }],
        );
    }

    #[test]
    fn test_rsqrt_complex_differentiation() {
        let input = ComplexNumber::new(0.7f64, -0.3f64);
        // d(1/√z)/dz = -z^{-3/2} / 2 on the principal branch.
        let expected = input.powc(ComplexNumber::new(-1.5, 0.0)) * ComplexNumber::new(-0.5, 0.0);
        assert_abs_diff_eq!(
            Array::scalar(expected),
            gradient_holomorphic(|input| input.rsqrt().unwrap(), Array::scalar(input)).unwrap(),
            epsilon = 1e-12,
        );
    }

    #[test]
    fn test_rsqrt_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = RsqrtOperation::new(),
            inputs = [Array::scalar(4.0)],
            expected = Array::scalar(0.5),
        );
    }

    #[test]
    fn test_rsqrt_transposition() {
        check_operation_transposition!(
            @rejected,
            operation = RsqrtOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_rsqrt_for_primitives() {
        assert_eq!(Rsqrt::rsqrt(&4.0_f64), Ok(0.5));
    }
}
