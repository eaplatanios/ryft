use std::ops::{Mul as StandardMul, Sub as StandardSub};

use crate::macros::{
    define_elementwise_capability, define_elementwise_operation, impl_differentiable_elementwise_operation,
};
use crate::operations::compare::{Compare, ComparisonDirection};
use crate::operations::constants::one_like::OneLike;
use crate::operations::constants::zero_like::ZeroLike;
use crate::operations::control_flow::select::Select;
use crate::operations::math::log::Log;
use crate::programs::ProgramError;

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`PowOperation`].
pub const POW_OPERATION_NAME: &str = "pow";

define_elementwise_operation!(
    @binary
    /// [`Operation`] that raises one value to the power of another elementwise (i.e., `(x, y) ↦ x^y`, with the
    /// complex power defined as the principal value `exp(y · log(x))`), promoting their element types and
    /// broadcasting their shapes. Matching the operand constraints of
    /// [StableHLO's `power`](https://openxla.org/stablehlo/spec#power) for those types, only floating-point and
    /// complex operands are supported (Ryft restricts the integer forms to keep the operation differentiable).
    /// Array operands that still carry partial sums are rejected, and their reduced-axis markers must agree.
    PowOperation, POW_OPERATION_NAME,
    Pow, pow,
    check_data_types = [@float],
    check_array_types = [@no_unreduced, @same_reduced_axes],
);

impl_differentiable_elementwise_operation! {
    @binary
    PowOperation,
    jvp<C>
    where
        C::Value: Pow
            + Log
            + Compare<C::Value>
            + Select
            + OneLike
            + ZeroLike
            + StandardMul<Output = C::Value>
            + StandardSub<Output = C::Value>,
    {
        // d(x^y) = y · x^{y-1} · dx + x^y · log(x) · dy, with log(x) evaluated at a base of one when x = 0 so
        // that the exponent contribution vanishes instead of producing log(0) = -∞.
        |(left, left_tangent), (right, _)| {
            let exponent = right.clone() - right.one_like();
            right * left.pow(&exponent)? * left_tangent
        };
        |(left, _), (right, right_tangent)| {
            let base_is_zero = left.compare(&left.zero_like(), ComparisonDirection::Equal)?;
            let safe_base = C::Value::select(&base_is_zero, &left.one_like(), &left)?;
            left.pow(&right)? * safe_base.log()? * right_tangent
        };
    },
    transpose = @nonlinear,
}

define_elementwise_capability!(
    @binary
    /// Value-level elementwise power capability. [`Pow`] fills the same role for
    /// [`PowOperation`] that [`Atan2`](crate::Atan2) fills for [`Atan2Operation`](crate::Atan2Operation).
    Pow,
    /// Raises this value to the power `exponent` elementwise, promoting both operands to a common floating-point or
    /// complex element type and returning a [`ProgramError`](crate::ProgramError) if something goes wrong.
    pow(exponent),
    PowOperation,
);

/// Implements [`Pow`] for one host primitive type. Only floating-point primitives are supported, matching the
/// reference backends' float-only power operation.
macro_rules! impl_capability_for_primitive {
    ($type:ty) => {
        impl Pow for $type {
            fn pow(&self, exponent: &Self) -> Result<Self, ProgramError> {
                Ok(<$type>::powf(*self, *exponent))
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

    use crate::arrays::{Array, ArrayOperation, ArrayType, DataType};
    use crate::differentiation::differentiate_at;
    use crate::macros::{
        check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_transposition, check_operation_type_inference,
    };
    use crate::operations::constants::one_like::OneLike;
    use crate::parameters::Placeholder;
    use crate::programs::ProgramBuilder;

    use super::*;

    #[test]
    fn test_pow() {
        assert_eq!(Array::scalar(2.0f32).pow(&Array::scalar(3.0f32)).unwrap(), Array::scalar(2.0f32.powf(3.0)),);
        assert_eq!(Array::scalar(2.0f64).pow(&Array::scalar(3.0f64)).unwrap(), Array::scalar(2.0f64.powf(3.0)),);
        assert_eq!(
            Array::scalar(bf16::from_f32(2.0)).pow(&Array::scalar(bf16::from_f32(3.0))).unwrap(),
            Array::scalar(bf16::from_f32(2.0f32.powf(3.0))),
        );
        assert_eq!(
            Array::scalar(f16::from_f32(2.0)).pow(&Array::scalar(f16::from_f32(3.0))).unwrap(),
            Array::scalar(f16::from_f32(2.0f32.powf(3.0))),
        );
        // The complex power is the principal value `exp(y · log(x))`.
        let input = ComplexNumber::new(0.7f64, -0.3f64);
        let exponent = ComplexNumber::new(2.0f64, 0.0f64);
        assert_abs_diff_eq!(
            Array::scalar(input).pow(&Array::scalar(exponent)).unwrap(),
            Array::scalar(input.powc(exponent)),
            epsilon = 1e-12,
        );
        assert_eq!(Array::scalar(2.0).pow(&Array::scalar(3.0)).unwrap(), Array::scalar(8.0),);
    }

    #[test]
    fn test_pow_type_inference() {
        check_operation_type_inference!(
            @elementwise @binary,
            operation = PowOperation,
            cases = [
                {
                    input_data_types = [DataType::F32, DataType::F64],
                    output_data_types = [DataType::F64],
                },
                {
                    input_data_types = [DataType::I32, DataType::I32],
                    error = "'pow' does not support input data type i32",
                },
            ],
        );
        check_operation_type_inference!(
            @reject @unreduced,
            operation = PowOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64), ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_pow_batching() {
        check_operation_batching!(
            @approx(epsilon = 1e-9),
            operation = PowOperation::new(),
            axis_size = 2,
            cases = [{
                inputs = [
                    (@mapped(axis = 0), Array::vector(vec![2.0, 3.0])),
                    (@mapped(axis = 0), Array::vector(vec![3.0, 2.0])),
                ],
                outputs = [(@mapped(axis = 0), Array::vector(vec![8.0, 9.0]))],
            }],
        );
    }

    #[test]
    fn test_pow_differentiation() {
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = PowOperation::new(),
            cases = [
                {
                    primals = [Array::scalar(2.0), Array::scalar(3.0)],
                    tangents = [Array::scalar(3.0), Array::scalar(5.0)],
                    primal_outputs = [Array::scalar(8.0)],
                    // d(x^y) = y · x^{y-1} · dx + x^y · ln(x) · dy = 3.0 · (3.0 · 4.0) + 5.0 · (8.0 · ln(2)).
                    tangent_outputs = [Array::scalar(3.0 * (3.0 * 4.0) + 5.0 * (8.0 * 2.0f64.ln()))],
                },
            ],
        );
    }

    #[test]
    fn test_pow_differentiation_at_zero_base() {
        // A zero base exercises the guarded log factor: the exponent contribution is exactly zero instead of
        // `log(0) = -∞` turning the tangent into a NaN. The finite-difference oracle cannot check this boundary
        // point (perturbing the base below zero is undefined), so the guard is asserted on the staged jvp program
        // directly.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let base = builder.add_input(ArrayType::scalar(DataType::F64));
        let exponent = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(PowOperation::new(), Vec::new(), vec![base, exponent]).unwrap()[0];
        let jvp_program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap()
            .jvp()
            .unwrap();
        let outputs = jvp_program
            .interpret(vec![Array::scalar(0.0), Array::scalar(2.0), Array::scalar(1.0), Array::scalar(1.0)])
            .unwrap();
        assert_eq!(outputs, vec![Array::scalar(0.0), Array::scalar(0.0)]);
    }

    #[test]
    fn test_pow_complex_differentiation() {
        // The holomorphic gradient of z² is 2z.
        let input = ComplexNumber::new(0.7f64, -0.3f64);
        assert_abs_diff_eq!(
            differentiate_at(Array::scalar(input))
                .holomorphic()
                .gradient(|input| {
                    let exponent = input.one_like() + input.one_like();
                    input.pow(&exponent).unwrap()
                })
                .unwrap(),
            Array::scalar(input * 2.0),
            epsilon = 1e-12,
        );
    }

    #[test]
    fn test_pow_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = PowOperation::new(),
            inputs = [Array::scalar(2.0), Array::scalar(3.0)],
            expected = Array::scalar(8.0),
        );
    }

    #[test]
    fn test_pow_transposition() {
        check_operation_transposition!(
            @rejected,
            operation = PowOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64), ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_pow_for_primitives() {
        assert_eq!(Pow::pow(&2.0_f64, &3.0), Ok(8.0));
    }
}
