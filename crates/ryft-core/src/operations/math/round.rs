use crate::macros::{
    define_elementwise_capability, define_elementwise_operation, impl_differentiable_elementwise_operation,
};
use crate::programs::ProgramError;

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`RoundOperation`].
pub const ROUND_OPERATION_NAME: &str = "round";

define_elementwise_operation!(
    @unary
    /// [`Operation`] that rounds one value elementwise to the nearest integer, with ties resolved toward the nearest
    /// even integer, while preserving its array metadata. Matching the operand constraints of
    /// [StableHLO's `round_nearest_even`](https://openxla.org/stablehlo/spec#round_nearest_even), only real
    /// floating-point operands are supported, and operands that still carry partial sums are rejected.
    RoundOperation, ROUND_OPERATION_NAME,
    Round, round,
    check_data_types = [@float @real],
    check_array_types = [@no_unreduced],
);

impl_differentiable_elementwise_operation!(@constant RoundOperation);

define_elementwise_capability!(
    @unary
    /// Value-level elementwise round-to-nearest-even capability. [`Round`] fills the same role for
    /// [`RoundOperation`] that [`Sin`](crate::Sin) fills for [`SinOperation`](crate::SinOperation).
    Round,
    /// Computes [`RoundOperation`] elementwise for this value.
    round,
    RoundOperation,
);

/// Implements [`Round`] for one host primitive type.
macro_rules! impl_capability_for_primitive {
    ($type:ty) => {
        impl Round for $type {
            fn round(&self) -> Result<Self, ProgramError> {
                Ok(<$type>::round_ties_even(*self))
            }
        }
    };
}

impl_capability_for_primitive!(f32);
impl_capability_for_primitive!(f64);

#[cfg(test)]
mod tests {
    use half::{bf16, f16};
    use pretty_assertions::assert_eq;

    use crate::arrays::{ArrayType, DataType};
    use crate::backends::Array;
    use crate::macros::{
        check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_type_inference,
    };

    use super::*;

    #[test]
    fn test_round() {
        // Ties resolve toward the nearest even integer.
        assert_eq!(Array::scalar(2.5f64).round().unwrap(), Array::scalar(2.0f64));
        assert_eq!(Array::scalar(3.5f64).round().unwrap(), Array::scalar(4.0f64));
        assert_eq!(Array::scalar(-2.5f32).round().unwrap(), Array::scalar(-2.0f32));
        assert_eq!(Array::scalar(2.3f64).round().unwrap(), Array::scalar(2.0f64));
        assert_eq!(Array::scalar(bf16::from_f32(2.5)).round().unwrap(), Array::scalar(bf16::from_f32(2.0)));
        assert_eq!(Array::scalar(f16::from_f32(3.5)).round().unwrap(), Array::scalar(f16::from_f32(4.0)));
        // NaNs pass through unchanged.
        assert!(Array::scalar(f64::NAN).round().unwrap().to_f64s()[0].is_nan());

        assert_eq!(Array::vector(vec![0.5, 1.5, -2.5]).round().unwrap(), Array::vector(vec![0.0, 2.0, -2.0]),);
    }

    #[test]
    fn test_round_type_inference() {
        check_operation_type_inference!(
            @elementwise @unary,
            operation = RoundOperation,
            cases = [
                {
                    input_data_types = [DataType::F64],
                    output_data_types = [DataType::F64],
                },
                {
                    input_data_types = [DataType::I32],
                    error = "'round' does not support input data type i32",
                },
                {
                    input_data_types = [DataType::C64],
                    error = "'round' does not support input data type c64",
                },
            ],
        );
        check_operation_type_inference!(
            @reject @unreduced,
            operation = RoundOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_round_batching() {
        check_operation_batching!(
            @exact,
            operation = RoundOperation::new(),
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), Array::vector(vec![0.5, 1.5]))],
                outputs = [(@mapped(axis = 0), Array::vector(vec![0.0, 2.0]))],
            }],
        );
    }

    #[test]
    fn test_round_differentiation() {
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = RoundOperation::new(),
            cases = [{
                primals = [Array::scalar(2.4)],
                tangents = [Array::scalar(1.0)],
                primal_outputs = [Array::scalar(2.0)],
                tangent_outputs = [Array::scalar(0.0)],
            }],
        );
    }

    #[test]
    fn test_round_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = RoundOperation::new(),
            inputs = [Array::scalar(2.5)],
            expected = Array::scalar(2.0),
        );
    }

    #[test]
    fn test_round_for_primitives() {
        assert_eq!(Round::round(&2.5_f64), Ok(2.0));
        assert_eq!(Round::round(&3.5_f64), Ok(4.0));
    }
}
