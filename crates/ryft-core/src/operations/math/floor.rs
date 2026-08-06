use crate::macros::{
    define_elementwise_capability, define_elementwise_operation, impl_differentiable_elementwise_operation,
};
use crate::programs::ProgramError;

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`FloorOperation`].
pub const FLOOR_OPERATION_NAME: &str = "floor";

define_elementwise_operation!(
    @unary
    /// [`Operation`] that computes the elementwise floor of one value (i.e., `x ↦ ⌊x⌋`, rounding toward negative
    /// infinity) while preserving its array metadata. Matching the operand constraints of
    /// [StableHLO's `floor`](https://openxla.org/stablehlo/spec#floor), only real floating-point operands are
    /// supported, and operands that still carry partial sums are rejected.
    FloorOperation, FLOOR_OPERATION_NAME,
    Floor, floor,
    check_data_types = [@float @real],
    check_array_types = [@no_unreduced],
);

impl_differentiable_elementwise_operation!(@constant FloorOperation);

define_elementwise_capability!(
    @unary
    /// Value-level elementwise floor capability. [`Floor`] fills the same role for
    /// [`FloorOperation`] that [`Sin`](crate::Sin) fills for [`SinOperation`](crate::SinOperation).
    Floor,
    /// Computes [`FloorOperation`] elementwise for this value.
    floor,
    FloorOperation,
);

/// Implements [`Floor`] for one host primitive type.
macro_rules! impl_capability_for_primitive {
    ($type:ty) => {
        impl Floor for $type {
            fn floor(&self) -> Result<Self, ProgramError> {
                Ok(<$type>::floor(*self))
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

    use crate::backends::arrays::Array;
    use crate::macros::{
        check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_type_inference,
    };
    use crate::types::{ArrayType, DataType};

    use super::*;

    #[test]
    fn test_floor() {
        assert_eq!(Array::scalar(2.7f32).floor().unwrap(), Array::scalar(2.0f32));
        assert_eq!(Array::scalar(-2.3f64).floor().unwrap(), Array::scalar(-3.0f64));
        assert_eq!(Array::scalar(bf16::from_f32(2.7)).floor().unwrap(), Array::scalar(bf16::from_f32(2.7f32.floor())),);
        assert_eq!(Array::scalar(f16::from_f32(2.7)).floor().unwrap(), Array::scalar(f16::from_f32(2.7f32.floor())),);
        // NaNs pass through unchanged.
        assert!(Array::scalar(f64::NAN).floor().unwrap().to_f64s()[0].is_nan());

        assert_eq!(Array::vector(vec![-0.7, 0.0, 2.5]).floor().unwrap(), Array::vector(vec![-1.0, 0.0, 2.0]),);
    }

    #[test]
    fn test_floor_type_inference() {
        check_operation_type_inference!(
            @elementwise @unary,
            operation = FloorOperation,
            cases = [
                {
                    input_data_types = [DataType::F64],
                    output_data_types = [DataType::F64],
                },
                {
                    input_data_types = [DataType::I32],
                    error = "'floor' does not support input data type i32",
                },
                {
                    input_data_types = [DataType::C64],
                    error = "'floor' does not support input data type c64",
                },
            ],
        );
        check_operation_type_inference!(
            @reject @unreduced,
            operation = FloorOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_floor_batching() {
        check_operation_batching!(
            @exact,
            operation = FloorOperation::new(),
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), Array::vector(vec![0.5, -1.5]))],
                outputs = [(@mapped(axis = 0), Array::vector(vec![0.0, -2.0]))],
            }],
        );
    }

    #[test]
    fn test_floor_differentiation() {
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = FloorOperation::new(),
            cases = [{
                primals = [Array::scalar(-2.5)],
                tangents = [Array::scalar(1.0)],
                primal_outputs = [Array::scalar(-3.0)],
                tangent_outputs = [Array::scalar(0.0)],
            }],
        );
    }

    #[test]
    fn test_floor_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = FloorOperation::new(),
            inputs = [Array::scalar(2.7)],
            expected = Array::scalar(2.0),
        );
    }

    #[test]
    fn test_floor_for_primitives() {
        assert_eq!(Floor::floor(&1.75_f64), Ok(1.0));
    }
}
