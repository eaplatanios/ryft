use crate::macros::{
    define_elementwise_capability, define_elementwise_operation, impl_differentiable_elementwise_operation,
};
use crate::programs::ProgramError;

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`CeilOperation`].
pub const CEIL_OPERATION_NAME: &str = "ceil";

define_elementwise_operation!(
    @unary
    /// [`Operation`] that computes the elementwise ceiling of one value (i.e., `x ↦ ⌈x⌉`, rounding toward positive
    /// infinity) while preserving its array metadata. Matching the operand constraints of
    /// [StableHLO's `ceil`](https://openxla.org/stablehlo/spec#ceil), only real floating-point operands are
    /// supported, and operands that still carry partial sums are rejected.
    CeilOperation, CEIL_OPERATION_NAME,
    Ceil, ceil,
    check_data_types = [@float @real],
    check_array_types = [@no_unreduced],
);

impl_differentiable_elementwise_operation!(@constant CeilOperation);

define_elementwise_capability!(
    @unary
    /// Value-level elementwise ceiling capability. [`Ceil`] fills the same role for
    /// [`CeilOperation`] that [`Sin`](crate::Sin) fills for [`SinOperation`](crate::SinOperation).
    Ceil,
    /// Computes [`CeilOperation`] elementwise for this value.
    ceil,
    CeilOperation,
);

/// Implements [`Ceil`] for one host primitive type.
macro_rules! impl_capability_for_primitive {
    ($type:ty) => {
        impl Ceil for $type {
            fn ceil(&self) -> Result<Self, ProgramError> {
                Ok(<$type>::ceil(*self))
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
    use crate::backends::scalars::Scalar;
    use crate::macros::{
        check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_type_inference,
    };
    use crate::types::{ArrayType, DataType};

    use super::*;

    #[test]
    fn test_ceil() {
        assert_eq!(Scalar::from(2.3f32).ceil().unwrap(), Scalar::from(3.0f32));
        assert_eq!(Scalar::from(-2.7f64).ceil().unwrap(), Scalar::from(-2.0f64));
        assert_eq!(Scalar::from(bf16::from_f32(2.3)).ceil().unwrap(), Scalar::from(bf16::from_f32(2.3f32.ceil())));
        assert_eq!(Scalar::from(f16::from_f32(2.3)).ceil().unwrap(), Scalar::from(f16::from_f32(2.3f32.ceil())));
        // NaNs pass through unchanged.
        assert!(match Scalar::from(f64::NAN).ceil().unwrap() {
            Scalar::F64(value) => value.is_nan(),
            _ => false,
        });

        assert_eq!(Array::vector(vec![0.7, 1.0, -1.5]).ceil().unwrap(), Array::vector(vec![1.0, 1.0, -1.0]),);
    }

    #[test]
    fn test_ceil_type_inference() {
        check_operation_type_inference!(
            @elementwise @unary,
            operation = CeilOperation,
            cases = [
                {
                    input_data_types = [DataType::F64],
                    output_data_types = [DataType::F64],
                },
                {
                    input_data_types = [DataType::I32],
                    error = "'ceil' does not support input data type i32",
                },
                {
                    input_data_types = [DataType::C64],
                    error = "'ceil' does not support input data type c64",
                },
            ],
        );
        check_operation_type_inference!(
            @reject @unreduced,
            operation = CeilOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_ceil_batching() {
        check_operation_batching!(
            @exact,
            operation = CeilOperation::new(),
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), Array::vector(vec![0.5, -1.5]))],
                outputs = [(@mapped(axis = 0), Array::vector(vec![1.0, -1.0]))],
            }],
        );
    }

    #[test]
    fn test_ceil_differentiation() {
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = CeilOperation::new(),
            cases = [{
                primals = [Array::scalar(2.5)],
                tangents = [Array::scalar(1.0)],
                primal_outputs = [Array::scalar(3.0)],
                tangent_outputs = [Array::scalar(0.0)],
            }],
        );
    }

    #[test]
    fn test_ceil_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = CeilOperation::new(),
            inputs = [Array::scalar(2.3)],
            expected = Array::scalar(3.0),
        );
    }

    #[test]
    fn test_ceil_for_primitives() {
        assert_eq!(Ceil::ceil(&1.25_f64), Ok(2.0));
    }
}
