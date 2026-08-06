use crate::macros::{
    define_elementwise_capability, define_elementwise_operation, impl_differentiable_elementwise_operation,
};
use crate::operations::compare::{Compare, ComparisonDirection};
use crate::operations::constants::ZeroLike;
use crate::operations::control_flow::Select;
use crate::programs::ProgramError;

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`MaxOperation`].
pub const MAX_OPERATION_NAME: &str = "max";

define_elementwise_operation!(
    @binary
    /// [`Operation`] that computes the elementwise maximum of two numeric values, promoting their element types and
    /// broadcasting their shapes. Matching the operand constraints of
    /// [StableHLO's `maximum`](https://openxla.org/stablehlo/spec#maximum), only real (non-complex) numeric operands
    /// are supported because complex numbers are unordered (Boolean maxima are spelled [`Or`](crate::Or)). For
    /// floating-point operands, NaNs propagate (the maximum is NaN when either operand is NaN) and `-0.0` orders
    /// below `+0.0`. Array operands that still carry partial sums are rejected, and their reduced-axis markers must
    /// agree.
    MaxOperation, MAX_OPERATION_NAME,
    Max, max,
    check_data_types = [@numeric @real],
    check_array_types = [@no_unreduced, @same_reduced_axes],
);

impl_differentiable_elementwise_operation! {
    @binary
    MaxOperation,
    jvp<C>
    where
        C::Value: Compare<C::Value> + Select + ZeroLike,
    {
        // The tangent follows the winning operand, with ties routing to the left operand: each contribution masks
        // its own tangent by the same `left >= right` predicate, so the combined tangent is
        // `select(left >= right, left_tangent, right_tangent)`.
        |(left, left_tangent), (right, _)| {
            let left_wins = left.compare(&right, ComparisonDirection::GreaterThanOrEqual)?;
            C::Value::select(&left_wins, &left_tangent, &left_tangent.zero_like())?
        };
        |(left, _), (right, right_tangent)| {
            let left_wins = left.compare(&right, ComparisonDirection::GreaterThanOrEqual)?;
            C::Value::select(&left_wins, &right_tangent.zero_like(), &right_tangent)?
        };
    },
    transpose = @nonlinear,
}

define_elementwise_capability!(
    @binary
    /// Value-level elementwise maximum capability. [`Max`] fills the same role for
    /// [`MaxOperation`] that [`Atan2`](crate::Atan2) fills for [`Atan2Operation`](crate::Atan2Operation).
    Max,
    /// Computes the elementwise maximum of this value and `right`, promoting both operands to a common numeric
    /// element type and returning a [`ProgramError`](crate::ProgramError) if something goes wrong.
    max(right),
    MaxOperation,
);

/// Implements [`Max`] for one host primitive type.
macro_rules! impl_capability_for_primitive {
    // Integer primitives use ordinary total-order comparison, which cannot fail.
    (@integer $type:ty) => {
        impl Max for $type {
            fn max(&self, right: &Self) -> Result<Self, ProgramError> {
                Ok(::std::cmp::Ord::max(*self, *right))
            }
        }
    };

    // Floating-point primitives mirror the reference backends: NaN operands propagate, and signed zeros order
    // through the IEEE 754 total order (so that `-0.0` sorts below `+0.0`).
    (@float $type:ty) => {
        impl Max for $type {
            fn max(&self, right: &Self) -> Result<Self, ProgramError> {
                Ok(if self.is_nan() {
                    *self
                } else if right.is_nan() {
                    *right
                } else if matches!(self.total_cmp(right), ::std::cmp::Ordering::Less) {
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
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::backends::arrays::{Array, ArrayOperation};
    use crate::macros::{
        check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_transposition, check_operation_type_inference,
    };
    use crate::parameters::Placeholder;
    use crate::programs::builders::ProgramBuilder;
    use crate::types::{ArrayType, DataType};

    use super::*;

    #[test]
    fn test_max() {
        assert_eq!(Array::scalar(2i32).max(&Array::scalar(5i32)).unwrap(), Array::scalar(5i32));
        assert_eq!(Array::scalar(-2i64).max(&Array::scalar(-5i64)).unwrap(), Array::scalar(-2i64));
        assert_eq!(Array::scalar(3u32).max(&Array::scalar(7u32)).unwrap(), Array::scalar(7u32));
        assert_eq!(Array::scalar(2.5f32).max(&Array::scalar(1.5f32)).unwrap(), Array::scalar(2.5f32));
        // Mixed-precision operands promote before comparing.
        assert_eq!(Array::scalar(2.5f32).max(&Array::scalar(3.5f64)).unwrap(), Array::scalar(3.5f64));
        assert_eq!(
            Array::scalar(bf16::from_f32(2.0)).max(&Array::scalar(bf16::from_f32(3.0))).unwrap(),
            Array::scalar(bf16::from_f32(3.0)),
        );
        assert_eq!(
            Array::scalar(f16::from_f32(2.0)).max(&Array::scalar(f16::from_f32(3.0))).unwrap(),
            Array::scalar(f16::from_f32(3.0)),
        );
        // NaNs propagate and `-0.0` orders below `+0.0`.
        assert!(Array::scalar(f64::NAN).max(&Array::scalar(1.0f64)).unwrap().to_f64s()[0].is_nan());
        assert!(Array::scalar(1.0f64).max(&Array::scalar(f64::NAN)).unwrap().to_f64s()[0].is_nan());
        let zero = Array::scalar(-0.0f64).max(&Array::scalar(0.0f64)).unwrap().to_f64s()[0];
        assert!(zero == 0.0 && zero.is_sign_positive());
        assert_eq!(
            Array::vector(vec![0.7, -1.0]).max(&Array::vector(vec![0.3, 2.0])).unwrap(),
            Array::vector(vec![0.7, 2.0]),
        );
    }

    #[test]
    fn test_max_type_inference() {
        check_operation_type_inference!(
            @elementwise @binary,
            operation = MaxOperation,
            cases = [
                {
                    input_data_types = [DataType::I32, DataType::I32],
                    output_data_types = [DataType::I32],
                },
                {
                    input_data_types = [DataType::C64, DataType::C64],
                    error = "'max' does not support input data type c64",
                },
                {
                    input_data_types = [DataType::Boolean, DataType::Boolean],
                    error = "'max' does not support input data type bool",
                },
            ],
        );
        check_operation_type_inference!(
            @reject @unreduced,
            operation = MaxOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64), ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_max_batching() {
        check_operation_batching!(
            @exact,
            operation = MaxOperation::new(),
            axis_size = 2,
            cases = [
                {
                    inputs = [
                        (@mapped(axis = 0), Array::vector(vec![0.5, -1.0])),
                        (@mapped(axis = 0), Array::vector(vec![0.3, 2.0])),
                    ],
                    outputs = [(@mapped(axis = 0), Array::vector(vec![0.5, 2.0]))],
                },
                {
                    inputs = [
                        (@replicated, Array::scalar(0.0)),
                        (@mapped(axis = 0), Array::vector(vec![1.0, -2.0])),
                    ],
                    outputs = [(@mapped(axis = 0), Array::vector(vec![1.0, 0.0]))],
                },
            ],
        );
    }

    #[test]
    fn test_max_differentiation() {
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = MaxOperation::new(),
            cases = [
                {
                    primals = [Array::scalar(2.0), Array::scalar(1.0)],
                    tangents = [Array::scalar(3.0), Array::scalar(5.0)],
                    primal_outputs = [Array::scalar(2.0)],
                    tangent_outputs = [Array::scalar(3.0)],
                    jvp = indoc! {"
                        lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[] .
                        let %4:f64[] = max %0 %1
                            %5:bool[] = compare [direction=GreaterThanOrEqual] %0 %1
                            %6:f64[] = zero_like %2
                            %7:f64[] = select %5 %2 %6
                            %8:bool[] = compare [direction=GreaterThanOrEqual] %0 %1
                            %9:f64[] = zero_like %3
                            %10:f64[] = select %8 %9 %3
                            %11:f64[] = add %7 %10
                        in (%4, %11)
                    "},
                },
                {
                    primals = [Array::scalar(1.0), Array::scalar(2.0)],
                    tangents = [Array::scalar(3.0), Array::scalar(5.0)],
                    primal_outputs = [Array::scalar(2.0)],
                    tangent_outputs = [Array::scalar(5.0)],
                },
            ],
        );
    }

    #[test]
    fn test_max_differentiation_at_ties() {
        // Ties route the tangent to the left operand. The finite-difference oracle cannot check the
        // non-differentiable tie point, so the tie policy is asserted on the staged jvp program directly.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let left = builder.add_input(ArrayType::scalar(DataType::F64));
        let right = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(MaxOperation::new(), Vec::new(), vec![left, right]).unwrap()[0];
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
    fn test_max_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = MaxOperation::new(),
            inputs = [Array::scalar(0.7), Array::scalar(0.3)],
            expected = Array::scalar(0.7),
        );
    }

    #[test]
    fn test_max_transposition() {
        check_operation_transposition!(
            @rejected,
            operation = MaxOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64), ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_max_for_primitives() {
        assert_eq!(Max::max(&3_usize, &4), Ok(4));
        assert!(Max::max(&1.0_f64, &f64::NAN).unwrap().is_nan());
        assert_eq!(Max::max(&0.0_f64, &-0.0).unwrap().to_bits(), 0.0_f64.to_bits());
    }
}
