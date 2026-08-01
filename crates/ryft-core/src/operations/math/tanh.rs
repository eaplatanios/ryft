use std::ops::{Mul as StandardMul, Sub as StandardSub};

use crate::macros::{
    define_elementwise_capability, define_elementwise_operation, impl_differentiable_elementwise_operation,
};
use crate::operations::constants::OneLike;
use crate::programs::ProgramError;

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`TanhOperation`].
pub const TANH_OPERATION_NAME: &str = "tanh";

define_elementwise_operation!(
    @unary
    /// [`Operation`] that computes the elementwise hyperbolic tangent of one value (i.e., `x ↦ tanh(x)`, the analytic
    /// continuation `tanh(z)` on complex operands) while preserving its array metadata. Only floating-point and
    /// complex operands are supported, and operands that still carry partial sums are rejected.
    TanhOperation, TANH_OPERATION_NAME,
    Tanh, tanh,
    check_data_types = [@float],
    check_array_types = [@no_unreduced],
);

impl_differentiable_elementwise_operation! {
    @unary
    TanhOperation,
    jvp<C>
    where
        C::Value: OneLike + StandardMul<Output = C::Value> + StandardSub<Output = C::Value>,
    {
        // d(tanh(x)) = (1 - tanh(x)²) · dx, reusing the primal output evaluated at the tangent type.
        |(_, input_tangent) -> output| (output.one_like() - output.clone() * output) * input_tangent
    },
    transpose = @nonlinear,
}

define_elementwise_capability!(
    @unary
    /// Value-level elementwise hyperbolic-tangent capability. [`Tanh`] fills the same role for
    /// [`TanhOperation`] that [`Sin`](crate::Sin) fills for [`SinOperation`](crate::SinOperation).
    Tanh,
    /// Computes [`TanhOperation`] elementwise for this value.
    tanh,
    TanhOperation,
);

/// Implements [`Tanh`] for one host primitive type.
macro_rules! impl_capability_for_primitive {
    ($type:ty) => {
        impl Tanh for $type {
            fn tanh(&self) -> Result<Self, ProgramError> {
                Ok(<$type>::tanh(*self))
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
    use indoc::indoc;
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
    fn test_tanh() {
        assert_eq!(Scalar::from(0.5f32).tanh().unwrap(), 0.5f32.tanh());
        assert_eq!(Scalar::from(0.5f64).tanh().unwrap(), 0.5f64.tanh());
        assert_eq!(Scalar::from(bf16::from_f32(0.5)).tanh().unwrap(), bf16::from_f32(0.5f32.tanh()));
        assert_eq!(Scalar::from(f16::from_f32(0.5)).tanh().unwrap(), f16::from_f32(0.5f32.tanh()));
        let input = ComplexNumber::new(0.7f64, -0.3f64);
        assert_abs_diff_eq!(Scalar::from(input).tanh().unwrap(), Scalar::from(input.tanh()), epsilon = 1e-12);

        assert_eq!(Array::scalar(0.7).tanh().unwrap(), Array::scalar(0.7f64.tanh()),);
    }

    #[test]
    fn test_tanh_type_inference() {
        check_operation_type_inference!(
            @elementwise @unary,
            operation = TanhOperation,
            cases = [
                {
                    input_data_types = [DataType::C64],
                    output_data_types = [DataType::C64],
                },
                {
                    input_data_types = [DataType::I32],
                    error = "'tanh' does not support input data type i32",
                },
            ],
        );
        check_operation_type_inference!(
            @reject @unreduced,
            operation = TanhOperation,
            input_types = [ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_tanh_batching() {
        check_operation_batching!(
            @approx(epsilon = 1e-9),
            operation = TanhOperation,
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), Array::vector(vec![0.5, -1.0]))],
                outputs = [(@mapped(axis = 0), Array::vector(vec![0.5f64.tanh(), (-1.0f64).tanh()]))],
            }],
        );
    }

    #[test]
    fn test_tanh_differentiation() {
        let expected_tangent = 3.0 * (1.0 - 0.7f64.tanh() * 0.7f64.tanh());
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = TanhOperation,
            cases = [{
                primals = [Array::scalar(0.7)],
                tangents = [Array::scalar(3.0)],
                primal_outputs = [Array::scalar(0.7f64.tanh())],
                tangent_outputs = [Array::scalar(expected_tangent)],
                jvp = indoc! {"
                    lambda %0:f64[], %1:f64[] .
                    let %2:f64[] = tanh %0
                        %3:f64[] = one_like %2
                        %4:f64[] = mul %2 %2
                        %5:f64[] = sub %3 %4
                        %6:f64[] = mul %5 %1
                    in (%2, %6)
                "},
            }],
        );
    }

    #[test]
    fn test_tanh_complex_differentiation() {
        let input = ComplexNumber::new(0.7f64, -0.3f64);
        let expected = {
            let tanh = input.tanh();
            ComplexNumber::new(1.0, 0.0) - tanh * tanh
        };
        assert_abs_diff_eq!(
            Scalar::from(expected),
            gradient_holomorphic(|input| input.tanh().unwrap(), Scalar::from(input)).unwrap(),
            epsilon = 1e-12,
        );
    }

    #[test]
    fn test_tanh_partial_evaluation() {
        check_operation_partial_evaluation!(operation = TanhOperation, inputs = [0.7], expected = 0.7f64.tanh(),);
    }

    #[test]
    fn test_tanh_transposition() {
        check_operation_transposition!(
            @rejected,
            operation = TanhOperation,
            input_types = [ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_tanh_for_primitives() {
        assert_eq!(Tanh::tanh(&0.0_f64), Ok(0.0));
    }
}
