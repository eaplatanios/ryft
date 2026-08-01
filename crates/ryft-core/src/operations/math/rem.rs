use std::ops::{Div as StandardDiv, Mul as StandardMul, Neg as StandardNeg, Sub as StandardSub};

use crate::macros::{
    define_elementwise_capability, define_elementwise_operation, define_tracer_operator,
    impl_differentiable_elementwise_operation,
};

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`RemOperation`].
pub const REM_OPERATION_NAME: &str = "rem";

define_elementwise_operation!(
    @binary
    /// [`Operation`] that computes the elementwise remainder of a dividend (its left operand) and a divisor (its
    /// right operand), promoting their element types and broadcasting their shapes. The result takes the sign of the
    /// dividend and has magnitude less than the divisor's (i.e., truncation semantics, matching
    /// [StableHLO's `remainder`](https://openxla.org/stablehlo/spec#remainder) and Rust's `%`). Only integer and
    /// floating-point operands are supported. Array operands that still carry partial sums are rejected, and their
    /// reduced-axis markers must agree.
    RemOperation, REM_OPERATION_NAME,
    Rem, rem,
    check_data_types = [@numeric @real],
    check_array_types = [@no_unreduced, @same_reduced_axes],
);

impl_differentiable_elementwise_operation! {
    @binary
    RemOperation,
    jvp<C>
    where
        C::Value: Rem
            + StandardDiv<Output = C::Value>
            + StandardMul<Output = C::Value>
            + StandardNeg<Output = C::Value>
            + StandardSub<Output = C::Value>,
    {
        // d(rem(x, y)) = dx - trunc(x / y) · dy away from the discontinuities, with the truncated quotient
        // recovered exactly as (x - rem(x, y)) / y.
        |(_, left_tangent), (_, _)| left_tangent;
        |(left, _), (right, right_tangent)| {
            let truncated_quotient = (left.clone() - left.rem(&right)?) / right;
            -(truncated_quotient * right_tangent)
        };
    },
    transpose = @nonlinear,
}

define_elementwise_capability!(
    @binary
    /// Value-level elementwise remainder capability. [`Rem`] is the fallible Ryft counterpart to
    /// [`std::ops::Rem`] that [`RemOperation`] interprets through, surfacing a
    /// [`ProgramError`](crate::ProgramError) when something goes wrong, instead of panicking. Value types
    /// additionally provide [`std::ops::Rem`] as ergonomic (albeit panicking) sugar layered on top of this
    /// capability.
    Rem,
    /// Computes the remainder of dividing this value (the dividend) by `right` (the divisor), with the result taking
    /// the sign of the dividend, and returning a [`ProgramError`](crate::ProgramError) if something goes wrong.
    rem(right),
    RemOperation,
);

define_tracer_operator!(
    @binary std::ops::Rem,
    rem,
    capability = Rem,
    method = rem,
);

#[cfg(test)]
mod tests {
    use half::{bf16, f16};
    use pretty_assertions::assert_eq;

    use crate::backends::arrays::Array;
    use crate::backends::scalars::Scalar;
    use crate::macros::{
        check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_transposition, check_operation_type_inference,
    };
    use crate::types::{ArrayType, DataType};

    use super::*;

    #[test]
    fn test_rem() {
        assert_eq!(Scalar::from(7i32).rem(&Scalar::from(3i32)).unwrap(), Scalar::from(1i32));
        // The result takes the sign of the dividend.
        assert_eq!(Scalar::from(-7i32).rem(&Scalar::from(3i32)).unwrap(), Scalar::from(-1i32));
        assert_eq!(Scalar::from(7i64).rem(&Scalar::from(-3i64)).unwrap(), Scalar::from(1i64));
        assert_eq!(Scalar::from(7u32).rem(&Scalar::from(3u32)).unwrap(), Scalar::from(1u32));
        assert_eq!(Scalar::from(7.5f64).rem(&Scalar::from(2.0f64)).unwrap(), Scalar::from(1.5f64));
        assert_eq!(Scalar::from(-7.5f32).rem(&Scalar::from(2.0f32)).unwrap(), Scalar::from(-1.5f32));
        assert_eq!(
            Scalar::from(bf16::from_f32(7.5)).rem(&Scalar::from(bf16::from_f32(2.0))).unwrap(),
            Scalar::from(bf16::from_f32(7.5f32 % 2.0f32)),
        );
        assert_eq!(
            Scalar::from(f16::from_f32(7.5)).rem(&Scalar::from(f16::from_f32(2.0))).unwrap(),
            Scalar::from(f16::from_f32(7.5f32 % 2.0f32)),
        );
        // Division by an integer zero reports an error instead of panicking.
        assert!(matches!(Scalar::from(7i32).rem(&Scalar::from(0i32)), Err(_)));

        assert_eq!(
            Array::vector(vec![7.5, -7.5]).rem(&Array::vector(vec![2.0, 2.0])).unwrap(),
            Array::vector(vec![1.5, -1.5]),
        );
    }

    #[test]
    fn test_rem_type_inference() {
        check_operation_type_inference!(
            @elementwise @binary,
            operation = RemOperation,
            cases = [
                {
                    input_data_types = [DataType::I32, DataType::I32],
                    output_data_types = [DataType::I32],
                },
                {
                    input_data_types = [DataType::C64, DataType::C64],
                    error = "'rem' does not support input data type c64",
                },
                {
                    input_data_types = [DataType::Boolean, DataType::Boolean],
                    error = "'rem' does not support input data type bool",
                },
            ],
        );
        check_operation_type_inference!(
            @reject @unreduced,
            operation = RemOperation,
            input_types = [ArrayType::scalar(DataType::F64), ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_rem_batching() {
        check_operation_batching!(
            @exact,
            operation = RemOperation,
            axis_size = 2,
            cases = [{
                inputs = [
                    (@mapped(axis = 0), Array::vector(vec![7.5, -7.5])),
                    (@mapped(axis = 0), Array::vector(vec![2.0, 2.0])),
                ],
                outputs = [(@mapped(axis = 0), Array::vector(vec![1.5, -1.5]))],
            }],
        );
    }

    #[test]
    fn test_rem_differentiation() {
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = RemOperation,
            cases = [{
                primals = [Array::scalar(7.5), Array::scalar(2.0)],
                tangents = [Array::scalar(3.0), Array::scalar(5.0)],
                primal_outputs = [Array::scalar(1.5)],
                // 3.0 - trunc(7.5 / 2.0) · 5.0 = 3.0 - 3.0 · 5.0 = -12.0.
                tangent_outputs = [Array::scalar(-12.0)],
            }],
        );
    }

    #[test]
    fn test_rem_partial_evaluation() {
        check_operation_partial_evaluation!(operation = RemOperation, inputs = [7.5, 2.0], expected = 1.5,);
    }

    #[test]
    fn test_rem_transposition() {
        check_operation_transposition!(
            @rejected,
            operation = RemOperation,
            input_types = [ArrayType::scalar(DataType::F64), ArrayType::scalar(DataType::F64)],
        );
    }
}
