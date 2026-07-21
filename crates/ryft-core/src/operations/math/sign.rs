use crate::macros::{
    define_elementwise_capability, define_elementwise_operation, impl_differentiable_elementwise_operation,
};
use crate::programs::types::TypeError;
use crate::types::DataType;

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`SignOperation`].
pub const SIGN_OPERATION_NAME: &str = "sign";

define_elementwise_operation!(
    @unary
    /// [`Operation`] that computes the elementwise sign of one value while preserving its array metadata. Matching
    /// the operand constraints of [StableHLO's `sign`](https://openxla.org/stablehlo/spec#sign), signed-integer,
    /// floating-point, and complex operands are supported, while unsigned-integer, Boolean, token, and
    /// structural-zero operands are rejected (unsigned magnitudes carry no sign to extract). Signed integers map to
    /// `-1`, `0`, or `1`; floating-point values map to `-1.0` or `1.0` away from zero while signed zeros and NaNs
    /// pass through unchanged; and complex values map to `z / |z|`, with `0` mapping to `0`. Operands that still
    /// carry partial sums are rejected because the sign of a partial sum is not the sign of the total.
    SignOperation, SIGN_OPERATION_NAME,
    Sign, sign,
    infer_data_types = |input_types: &[DataType]| {
        let input_type = input_types[0];
        if input_type.is_signed() || input_type.is_floating_point() || input_type.is_complex() {
            Ok(vec![input_type])
        } else {
            Err(TypeError { message: format!("cannot compute the sign of a value of data type {input_type}") })
        }
    },
    check_array_types = [@no_unreduced],
);

impl_differentiable_elementwise_operation!(@constant SignOperation);

define_elementwise_capability!(
    @unary
    /// Value-level elementwise sign capability. [`Sign`] fills the same role for
    /// [`SignOperation`] that [`Sin`](crate::Sin) fills for [`SinOperation`](crate::SinOperation).
    Sign,
    /// Computes [`SignOperation`] elementwise for this value.
    sign,
    SignOperation,
);

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use half::{bf16, f16};
    use num_complex::Complex as ComplexNumber;
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
    fn test_sign() {
        assert_eq!(Scalar::from(-3i32).sign().unwrap(), Scalar::from(-1i32));
        assert_eq!(Scalar::from(0i32).sign().unwrap(), Scalar::from(0i32));
        assert_eq!(Scalar::from(5i64).sign().unwrap(), Scalar::from(1i64));
        assert_eq!(Scalar::from(-2.5f32).sign().unwrap(), Scalar::from(-1.0f32));
        assert_eq!(Scalar::from(2.5f64).sign().unwrap(), Scalar::from(1.0f64));
        assert_eq!(Scalar::from(bf16::from_f32(-4.0)).sign().unwrap(), Scalar::from(bf16::from_f32(-1.0)));
        assert_eq!(Scalar::from(f16::from_f32(4.0)).sign().unwrap(), Scalar::from(f16::from_f32(1.0)));
        // Signed zeros and NaNs pass through unchanged.
        assert_eq!(Scalar::from(0.0f64).sign().unwrap(), Scalar::from(0.0f64));
        assert_eq!(Scalar::from(-0.0f64).sign().unwrap().to_string(), Scalar::from(-0.0f64).to_string());
        assert!(match Scalar::from(f64::NAN).sign().unwrap() {
            Scalar::F64(value) => value.is_nan(),
            _ => false,
        });
        // Complex signs normalize to `z / |z|` and map the origin to itself.
        let input = ComplexNumber::new(3.0f64, -4.0f64);
        assert_abs_diff_eq!(
            Scalar::from(input).sign().unwrap(),
            Scalar::from(ComplexNumber::new(0.6, -0.8)),
            epsilon = 1e-12,
        );
        assert_eq!(
            Scalar::from(ComplexNumber::new(0.0f64, 0.0f64)).sign().unwrap(),
            Scalar::from(ComplexNumber::new(0.0f64, 0.0f64)),
        );

        assert_eq!(Array::vector(vec![-0.7, 0.0, 2.0]).sign().unwrap(), Array::vector(vec![-1.0, 0.0, 1.0]),);
    }

    #[test]
    fn test_sign_type_inference() {
        check_operation_type_inference!(
            @elementwise @unary,
            operation = SignOperation,
            cases = [
                {
                    input_data_types = [DataType::I32],
                    output_data_types = [DataType::I32],
                },
                {
                    input_data_types = [DataType::C64],
                    output_data_types = [DataType::C64],
                },
                {
                    input_data_types = [DataType::U8],
                    error = "cannot compute the sign of a value of data type u8",
                },
                {
                    input_data_types = [DataType::Boolean],
                    error = "cannot compute the sign of a value of data type bool",
                },
            ],
        );
        check_operation_type_inference!(
            @reject @unreduced,
            operation = SignOperation,
            input_types = [ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_sign_batching() {
        check_operation_batching!(
            @exact,
            operation = SignOperation,
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), Array::vector(vec![0.5, -1.0]))],
                outputs = [(@mapped(axis = 0), Array::vector(vec![1.0, -1.0]))],
            }],
        );
    }

    #[test]
    fn test_sign_differentiation() {
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = SignOperation,
            cases = [{
                primals = [Array::scalar(-2.0)],
                tangents = [Array::scalar(1.0)],
                primal_outputs = [Array::scalar(-1.0)],
                tangent_outputs = [Array::scalar(0.0)],
            }],
        );
    }

    #[test]
    fn test_sign_partial_evaluation() {
        check_operation_partial_evaluation!(operation = SignOperation, inputs = [-0.7], expected = -1.0,);
    }
}
