use std::ops::{Mul as StandardMul, Sub as StandardSub};

use crate::macros::{
    define_elementwise_capability, define_elementwise_operation, impl_differentiable_elementwise_operation,
};
use crate::operations::constants::OneLike;
use crate::programs::ProgramError;

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`LogisticOperation`].
pub const LOGISTIC_OPERATION_NAME: &str = "logistic";

define_elementwise_operation!(
    @unary
    /// [`Operation`] that computes the elementwise logistic sigmoid of one value (i.e., `x ↦ 1 / (1 + e^{-x})`, the
    /// analytic continuation on complex operands) while preserving its array metadata. Only floating-point and
    /// complex operands are supported, and operands that still carry partial sums are rejected.
    LogisticOperation, LOGISTIC_OPERATION_NAME,
    Logistic, logistic,
    check_data_types = [@float],
    check_array_types = [@no_unreduced],
);

impl_differentiable_elementwise_operation! {
    @unary
    LogisticOperation,
    jvp<C>
    where
        C::Value: OneLike + StandardMul<Output = C::Value> + StandardSub<Output = C::Value>,
    {
        // d(logistic(x)) = logistic(x) · (1 - logistic(x)) · dx, reusing the primal output evaluated at the
        // tangent type.
        |(_, input_tangent) -> output| output.clone() * (output.one_like() - output) * input_tangent
    },
    transpose = @nonlinear,
}

define_elementwise_capability!(
    @unary
    /// Value-level elementwise logistic-sigmoid capability. [`Logistic`] fills the same role for
    /// [`LogisticOperation`] that [`Sin`](crate::Sin) fills for [`SinOperation`](crate::SinOperation).
    Logistic,
    /// Computes [`LogisticOperation`] elementwise for this value.
    logistic,
    LogisticOperation,
);

/// Implements [`Logistic`] for one host primitive type.
macro_rules! impl_capability_for_primitive {
    ($type:ty) => {
        impl Logistic for $type {
            fn logistic(&self) -> Result<Self, ProgramError> {
                Ok(((-*self).exp() + 1.0).recip())
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

    use crate::backends::arrays::Array;
    use crate::backends::scalars::Scalar;
    use crate::differentiation::gradient_holomorphic;
    use crate::macros::{
        check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_transposition, check_operation_type_inference,
    };
    use crate::types::{ArrayType, DataType};

    use super::*;

    #[test]
    fn test_logistic() {
        assert_eq!(Scalar::from(0.5f32).logistic().unwrap(), 1.0 / (1.0 + (-0.5f32).exp()));
        assert_eq!(Scalar::from(0.5f64).logistic().unwrap(), 1.0 / (1.0 + (-0.5f64).exp()));
        assert_eq!(
            Scalar::from(bf16::from_f32(0.5)).logistic().unwrap(),
            bf16::from_f32(1.0 / (1.0 + (-0.5f32).exp())),
        );
        assert_eq!(Scalar::from(f16::from_f32(0.5)).logistic().unwrap(), f16::from_f32(1.0 / (1.0 + (-0.5f32).exp())),);
        let input = ComplexNumber::new(0.7f64, -0.3f64);
        let expected = ComplexNumber::new(1.0, 0.0) / (ComplexNumber::new(1.0, 0.0) + (-input).exp());
        assert_abs_diff_eq!(Scalar::from(input).logistic().unwrap(), Scalar::from(expected), epsilon = 1e-12);

        assert_eq!(Array::scalar(0.7).logistic().unwrap(), Array::scalar(1.0 / (1.0 + (-0.7f64).exp())),);
    }

    #[test]
    fn test_logistic_type_inference() {
        check_operation_type_inference!(
            @elementwise @unary,
            operation = LogisticOperation,
            cases = [
                {
                    input_data_types = [DataType::C64],
                    output_data_types = [DataType::C64],
                },
                {
                    input_data_types = [DataType::I32],
                    error = "'logistic' does not support input data type i32",
                },
            ],
        );
        check_operation_type_inference!(
            @reject @unreduced,
            operation = LogisticOperation,
            input_types = [ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_logistic_batching() {
        check_operation_batching!(
            @approx(epsilon = 1e-9),
            operation = LogisticOperation,
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), Array::vector(vec![0.5, -1.0]))],
                outputs = [(
                    @mapped(axis = 0),
                    Array::vector(vec![1.0 / (1.0 + (-0.5f64).exp()), 1.0 / (1.0 + 1.0f64.exp())])
                )],
            }],
        );
    }

    #[test]
    fn test_logistic_differentiation() {
        let logistic = 1.0 / (1.0 + (-0.7f64).exp());
        let expected_tangent = 3.0 * logistic * (1.0 - logistic);
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = LogisticOperation,
            cases = [{
                primals = [Array::scalar(0.7)],
                tangents = [Array::scalar(3.0)],
                primal_outputs = [Array::scalar(logistic)],
                tangent_outputs = [Array::scalar(expected_tangent)],
            }],
        );
    }

    #[test]
    fn test_logistic_complex_differentiation() {
        let input = ComplexNumber::new(0.7f64, -0.3f64);
        let expected = {
            let logistic = ComplexNumber::new(1.0, 0.0) / (ComplexNumber::new(1.0, 0.0) + (-input).exp());
            logistic * (ComplexNumber::new(1.0, 0.0) - logistic)
        };
        assert_abs_diff_eq!(
            Scalar::from(expected),
            gradient_holomorphic(|input| input.logistic().unwrap(), Scalar::from(input)).unwrap(),
            epsilon = 1e-12,
        );
    }

    #[test]
    fn test_logistic_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = LogisticOperation,
            inputs = [0.7],
            expected = 1.0 / (1.0 + (-0.7f64).exp()),
        );
    }

    #[test]
    fn test_logistic_transposition() {
        check_operation_transposition!(
            @rejected,
            operation = LogisticOperation,
            input_types = [ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_logistic_for_primitives() {
        assert_eq!(Logistic::logistic(&0.0_f64), Ok(0.5));
    }
}
