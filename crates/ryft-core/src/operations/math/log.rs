use std::ops::Div as StandardDiv;

use crate::macros::{
    define_elementwise_capability, define_elementwise_operation, impl_differentiable_elementwise_operation,
};
use crate::programs::ProgramError;

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`LogOperation`].
pub const LOG_OPERATION_NAME: &str = "log";

define_elementwise_operation!(
    @unary
    /// [`Operation`] that computes the elementwise natural logarithm of one value (i.e.,
    /// `x ↦ ln(x)`, the principal branch `ln(z)` on complex operands) while preserving its array metadata. Only
    /// floating-point and complex operands are supported, and operands that still carry partial sums are rejected.
    LogOperation, LOG_OPERATION_NAME,
    Log, log,
    check_data_types = [@float],
    check_array_types = [@no_unreduced],
);

impl_differentiable_elementwise_operation! {
    @unary
    LogOperation,
    jvp<C> where C::Value: StandardDiv<Output = C::Value> {
        |(input, input_tangent)| input_tangent / input
    },
    transpose = @nonlinear,
}

define_elementwise_capability!(
    @unary
    /// Value-level elementwise natural-logarithm capability. [`Log`] fills the same role for
    /// [`LogOperation`] that [`Sin`](crate::Sin) fills for [`SinOperation`](crate::SinOperation).
    Log,
    /// Computes [`LogOperation`] elementwise for this value.
    log,
    LogOperation,
);

/// Implements [`Log`] for one host primitive type.
macro_rules! impl_capability_for_primitive {
    ($type:ty) => {
        impl Log for $type {
            fn log(&self) -> Result<Self, ProgramError> {
                Ok(<$type>::ln(*self))
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

    use crate::arrays::{Array, ArrayOperation, ArrayType, DataType};
    use crate::differentiation::{gradient_holomorphic, jvp};
    use crate::macros::{
        check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_transposition, check_operation_type_inference,
    };
    use crate::parameters::Placeholder;
    use crate::programs::{ProgramBuilder, Typed};

    use super::*;

    #[test]
    fn test_log() {
        assert_eq!(Array::scalar(0.5f32).log().unwrap(), Array::scalar(0.5f32.ln()));
        assert_eq!(Array::scalar(0.5f64).log().unwrap(), Array::scalar(0.5f64.ln()));
        assert_eq!(Array::scalar(bf16::from_f32(0.5)).log().unwrap(), Array::scalar(bf16::from_f32(0.5f32.ln())),);
        assert_eq!(Array::scalar(f16::from_f32(0.5)).log().unwrap(), Array::scalar(f16::from_f32(0.5f32.ln())),);
        let input = ComplexNumber::new(0.7f64, -0.3f64);
        assert_abs_diff_eq!(Array::scalar(input).log().unwrap(), Array::scalar(input.ln()), epsilon = 1e-12);
        // The principal branch maps the negative real axis to `ln|x| + iπ`.
        assert_abs_diff_eq!(
            Array::scalar(ComplexNumber::new(-1.0f64, 0.0)).log().unwrap(),
            Array::scalar(ComplexNumber::new(0.0f64, std::f64::consts::PI)),
            epsilon = 1e-12,
        );

        assert_eq!(Array::scalar(0.7).log().unwrap(), Array::scalar(0.7f64.ln()),);
    }

    #[test]
    fn test_log_type_inference() {
        check_operation_type_inference!(
            @elementwise @unary,
            operation = LogOperation,
            cases = [
                {
                    input_data_types = [DataType::C64],
                    output_data_types = [DataType::C64],
                },
                {
                    input_data_types = [DataType::I32],
                    error = "'log' does not support input data type i32",
                },
            ],
        );
        check_operation_type_inference!(
            @reject @unreduced,
            operation = LogOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_log_batching() {
        check_operation_batching!(
            @approx(epsilon = 1e-9),
            operation = LogOperation::new(),
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), Array::vector(vec![0.5, 2.0]))],
                outputs = [(@mapped(axis = 0), Array::vector(vec![0.5f64.ln(), 2.0f64.ln()]))],
            }],
        );
    }

    #[test]
    fn test_log_differentiation() {
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = LogOperation::new(),
            cases = [{
                primals = [Array::scalar(0.7)],
                tangents = [Array::scalar(3.0)],
                primal_outputs = [Array::scalar(0.7f64.ln())],
                tangent_outputs = [Array::scalar(3.0 / 0.7)],
                jvp = indoc! {"
                    lambda %0:f64[], %1:f64[] .
                    let %2:f64[] = log %0
                        %3:f64[] = div %1 %0
                    in (%2, %3)
                "},
            }],
        );
    }

    #[test]
    fn test_log_complex_differentiation() {
        let input = ComplexNumber::new(0.7f64, -0.3f64);
        assert_eq!(
            gradient_holomorphic(|input| input.log().unwrap(), Array::scalar(input)),
            Ok(Array::scalar(ComplexNumber::new(1.0, 0.0) / input)),
        );
    }

    #[test]
    fn test_log_low_precision_differentiation_uses_widened_tangents() {
        let primal = Array::from_f64s(ArrayType::scalar(DataType::F8E8M0FNU), vec![2.0]);
        let input_tangent = Array::from_f64s(ArrayType::scalar(DataType::F32), vec![3.0]);
        let (_, tangent) = jvp(|input| input.log(), primal, input_tangent).unwrap();
        assert_eq!(tangent.r#type().as_ref(), &ArrayType::scalar(DataType::F32));
        assert_abs_diff_eq!(tangent.to_f64s()[0], 1.5, epsilon = 1e-9);

        // The widened staged tangent program divides by the input converted to the widened differential
        // representation.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F8E8M0FNU));
        let output = builder.add_instruction(LogOperation::new(), Vec::new(), vec![input]).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap()
            .jvp()
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f8e8m0fnu[], %1:f32[] .
                let %2:f8e8m0fnu[] = log %0
                    %3:f32[] = convert_element_type [data_type=f32] %0
                    %4:f32[] = div %1 %3
                in (%2, %4)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_log_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = LogOperation::new(),
            inputs = [Array::scalar(0.7)],
            expected = Array::scalar(0.7f64.ln()),
        );
    }

    #[test]
    fn test_log_transposition() {
        check_operation_transposition!(
            @rejected,
            operation = LogOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_log_for_primitives() {
        assert_eq!(Log::log(&1.0_f64), Ok(0.0));
    }
}
