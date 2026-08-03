use crate::macros::{
    define_elementwise_capability, define_elementwise_operation, impl_differentiable_elementwise_operation,
};
use crate::operations::compare::{Compare, ComparisonDirection};
use crate::operations::constants::ZeroLike;
use crate::operations::control_flow::Select;
use crate::programs::ProgramError;

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`MinOperation`].
pub const MIN_OPERATION_NAME: &str = "min";

define_elementwise_operation!(
    @binary
    /// [`Operation`] that computes the elementwise minimum of two numeric values, promoting their element types and
    /// broadcasting their shapes. Matching the operand constraints of
    /// [StableHLO's `minimum`](https://openxla.org/stablehlo/spec#minimum), only real (non-complex) numeric operands
    /// are supported because complex numbers are unordered (Boolean minima are spelled [`And`](crate::And)). For
    /// floating-point operands, NaNs propagate (the minimum is NaN when either operand is NaN) and `-0.0` orders
    /// below `+0.0` (so `min(-0.0, +0.0)` is `-0.0`). Array operands that still carry partial sums are rejected,
    /// and their reduced-axis markers must agree.
    MinOperation, MIN_OPERATION_NAME,
    Min, min,
    check_data_types = [@numeric @real],
    check_array_types = [@no_unreduced, @same_reduced_axes],
);

impl_differentiable_elementwise_operation! {
    @binary
    MinOperation,
    jvp<C>
    where
        C::Value: Compare<C::Value> + Select + ZeroLike,
    {
        // The tangent follows the winning operand, with ties routing to the left operand: each contribution masks
        // its own tangent by the same `left <= right` predicate, so the combined tangent is
        // `select(left <= right, left_tangent, right_tangent)`.
        |(left, left_tangent), (right, _)| {
            let left_wins = left.compare(&right, ComparisonDirection::LessThanOrEqual)?;
            C::Value::select(&left_wins, &left_tangent, &left_tangent.zero_like())?
        };
        |(left, _), (right, right_tangent)| {
            let left_wins = left.compare(&right, ComparisonDirection::LessThanOrEqual)?;
            C::Value::select(&left_wins, &right_tangent.zero_like(), &right_tangent)?
        };
    },
    transpose = @nonlinear,
}

define_elementwise_capability!(
    @binary
    /// Value-level elementwise minimum capability. [`Min`] fills the same role for
    /// [`MinOperation`] that [`Atan2`](crate::Atan2) fills for [`Atan2Operation`](crate::Atan2Operation).
    Min,
    /// Computes the elementwise minimum of this value and `right`, promoting both operands to a common numeric
    /// element type and returning a [`ProgramError`](crate::ProgramError) if something goes wrong.
    min(right),
    MinOperation,
);

/// Implements [`Min`] for one host primitive type.
macro_rules! impl_capability_for_primitive {
    // Integer primitives use ordinary total-order comparison, which cannot fail.
    (@integer $type:ty) => {
        impl Min for $type {
            fn min(&self, right: &Self) -> Result<Self, ProgramError> {
                Ok(::std::cmp::Ord::min(*self, *right))
            }
        }
    };

    // Floating-point primitives mirror the reference backends: NaN operands propagate, and signed zeros order
    // through the IEEE 754 total order (so that `-0.0` sorts below `+0.0`).
    (@float $type:ty) => {
        impl Min for $type {
            fn min(&self, right: &Self) -> Result<Self, ProgramError> {
                Ok(if self.is_nan() {
                    *self
                } else if right.is_nan() {
                    *right
                } else if matches!(self.total_cmp(right), ::std::cmp::Ordering::Greater) {
                    *right
                } else {
                    *self
                })
            }
        }
    };
}

impl_capability_for_primitive!(@integer i8);
impl_capability_for_primitive!(@integer i16);
impl_capability_for_primitive!(@integer i32);
impl_capability_for_primitive!(@integer i64);
impl_capability_for_primitive!(@integer i128);
impl_capability_for_primitive!(@integer isize);
impl_capability_for_primitive!(@integer u8);
impl_capability_for_primitive!(@integer u16);
impl_capability_for_primitive!(@integer u32);
impl_capability_for_primitive!(@integer u64);
impl_capability_for_primitive!(@integer u128);
impl_capability_for_primitive!(@integer usize);
impl_capability_for_primitive!(@float f32);
impl_capability_for_primitive!(@float f64);

#[cfg(test)]
mod tests {
    use half::{bf16, f16};
    use pretty_assertions::assert_eq;

    use crate::backends::arrays::{Array, ArrayOperation};
    use crate::backends::scalars::Scalar;
    use crate::macros::{
        check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_transposition, check_operation_type_inference,
    };
    use crate::parameters::Placeholder;
    use crate::programs::builders::ProgramBuilder;
    use crate::types::{ArrayType, DataType};

    use super::*;

    #[test]
    fn test_min() {
        assert_eq!(Scalar::from(2i32).min(&Scalar::from(5i32)).unwrap(), Scalar::from(2i32));
        assert_eq!(Scalar::from(-2i64).min(&Scalar::from(-5i64)).unwrap(), Scalar::from(-5i64));
        assert_eq!(Scalar::from(3u32).min(&Scalar::from(7u32)).unwrap(), Scalar::from(3u32));
        assert_eq!(Scalar::from(2.5f32).min(&Scalar::from(1.5f32)).unwrap(), Scalar::from(1.5f32));
        // Mixed-precision operands promote before comparing.
        assert_eq!(Scalar::from(2.5f32).min(&Scalar::from(3.5f64)).unwrap(), Scalar::from(2.5f64));
        assert_eq!(
            Scalar::from(bf16::from_f32(2.0)).min(&Scalar::from(bf16::from_f32(3.0))).unwrap(),
            Scalar::from(bf16::from_f32(2.0)),
        );
        assert_eq!(
            Scalar::from(f16::from_f32(2.0)).min(&Scalar::from(f16::from_f32(3.0))).unwrap(),
            Scalar::from(f16::from_f32(2.0)),
        );
        // NaNs propagate and `-0.0` orders below `+0.0`.
        assert!(match Scalar::from(f64::NAN).min(&Scalar::from(1.0f64)).unwrap() {
            Scalar::F64(value) => value.is_nan(),
            _ => false,
        });
        assert!(match Scalar::from(1.0f64).min(&Scalar::from(f64::NAN)).unwrap() {
            Scalar::F64(value) => value.is_nan(),
            _ => false,
        });
        assert!(match Scalar::from(-0.0f64).min(&Scalar::from(0.0f64)).unwrap() {
            Scalar::F64(value) => value == 0.0 && value.is_sign_negative(),
            _ => false,
        });
        assert_eq!(
            Array::vector(vec![0.7, -1.0]).min(&Array::vector(vec![0.3, 2.0])).unwrap(),
            Array::vector(vec![0.3, -1.0]),
        );
    }

    #[test]
    fn test_min_type_inference() {
        check_operation_type_inference!(
            @elementwise @binary,
            operation = MinOperation,
            cases = [
                {
                    input_data_types = [DataType::I32, DataType::I32],
                    output_data_types = [DataType::I32],
                },
                {
                    input_data_types = [DataType::C64, DataType::C64],
                    error = "'min' does not support input data type c64",
                },
                {
                    input_data_types = [DataType::Boolean, DataType::Boolean],
                    error = "'min' does not support input data type bool",
                },
            ],
        );
        check_operation_type_inference!(
            @reject @unreduced,
            operation = MinOperation::new(),
            input_types = [ArrayType::scalar(DataType::F64), ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_min_batching() {
        check_operation_batching!(
            @exact,
            operation = MinOperation::new(),
            axis_size = 2,
            cases = [
                {
                    inputs = [
                        (@mapped(axis = 0), Array::vector(vec![0.5, -1.0])),
                        (@mapped(axis = 0), Array::vector(vec![0.3, 2.0])),
                    ],
                    outputs = [(@mapped(axis = 0), Array::vector(vec![0.3, -1.0]))],
                },
                {
                    inputs = [
                        (@replicated, Array::scalar(0.0)),
                        (@mapped(axis = 0), Array::vector(vec![1.0, -2.0])),
                    ],
                    outputs = [(@mapped(axis = 0), Array::vector(vec![0.0, -2.0]))],
                },
            ],
        );
    }

    #[test]
    fn test_min_differentiation() {
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = MinOperation::new(),
            cases = [
                {
                    primals = [Array::scalar(2.0), Array::scalar(1.0)],
                    tangents = [Array::scalar(3.0), Array::scalar(5.0)],
                    primal_outputs = [Array::scalar(1.0)],
                    tangent_outputs = [Array::scalar(5.0)],
                },
                {
                    primals = [Array::scalar(1.0), Array::scalar(2.0)],
                    tangents = [Array::scalar(3.0), Array::scalar(5.0)],
                    primal_outputs = [Array::scalar(1.0)],
                    tangent_outputs = [Array::scalar(3.0)],
                },
            ],
        );
    }

    #[test]
    fn test_min_differentiation_at_ties() {
        // Ties route the tangent to the left operand. The finite-difference oracle cannot check the
        // non-differentiable tie point, so the tie policy is asserted on the staged jvp program directly.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let left = builder.add_input(ArrayType::scalar(DataType::F64));
        let right = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(MinOperation::new(), Vec::new(), vec![left, right]).unwrap()[0];
        let jvp_program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap()
            .jvp()
            .unwrap();
        let outputs = jvp_program
            .interpret(vec![Array::scalar(2.0), Array::scalar(2.0), Array::scalar(3.0), Array::scalar(5.0)])
            .unwrap();
        assert_eq!(outputs, vec![Array::scalar(2.0), Array::scalar(3.0)]);
    }

    #[test]
    fn test_min_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = MinOperation::new(),
            inputs = [Scalar::from(0.7), Scalar::from(0.3)],
            expected = Scalar::from(0.3),
        );
    }

    #[test]
    fn test_min_transposition() {
        check_operation_transposition!(
            @rejected,
            operation = MinOperation::new(),
            input_types = [ArrayType::scalar(DataType::F64), ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_min_for_primitives() {
        assert_eq!(Min::min(&3_usize, &4), Ok(3));
        assert!(Min::min(&1.0_f64, &f64::NAN).unwrap().is_nan());
        assert_eq!(Min::min(&0.0_f64, &-0.0).unwrap().to_bits(), (-0.0_f64).to_bits());
    }
}
