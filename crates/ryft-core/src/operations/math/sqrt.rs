use std::ops::{Add as StandardAdd, Div as StandardDiv};

use crate::macros::{
    define_elementwise_capability, define_elementwise_operation, impl_differentiable_elementwise_operation,
};
use crate::programs::ProgramError;

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`SqrtOperation`].
pub const SQRT_OPERATION_NAME: &str = "sqrt";

define_elementwise_operation!(
    @unary
    /// [`Operation`] that computes the elementwise square root of one value (i.e., `x ↦ √x`, the
    /// principal branch `√z` on complex operands) while preserving its array metadata. Only floating-point and
    /// complex operands are supported, and operands that still carry partial sums are rejected.
    SqrtOperation, SQRT_OPERATION_NAME,
    Sqrt, sqrt,
    check_data_types = [@float],
    check_array_types = [@no_unreduced],
);

impl_differentiable_elementwise_operation! {
    @unary
    SqrtOperation,
    jvp<C> where C::Value: StandardAdd<Output = C::Value> + StandardDiv<Output = C::Value> {
        |(_, input_tangent) -> output| input_tangent / (output.clone() + output)
    },
    transpose = @nonlinear,
}

define_elementwise_capability!(
    @unary
    /// Value-level elementwise square-root capability. [`Sqrt`] fills the same role for
    /// [`SqrtOperation`] that [`Sin`](crate::Sin) fills for [`SinOperation`](crate::SinOperation).
    Sqrt,
    /// Computes [`SqrtOperation`] elementwise for this value.
    sqrt,
    SqrtOperation,
);

/// Implements [`Sqrt`] for one host primitive type.
macro_rules! impl_capability_for_primitive {
    ($type:ty) => {
        impl Sqrt for $type {
            fn sqrt(&self) -> Result<Self, ProgramError> {
                Ok(<$type>::sqrt(*self))
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
    fn test_sqrt() {
        assert_eq!(Array::scalar(0.25f32).sqrt().unwrap(), Array::scalar(0.5f32));
        assert_eq!(Array::scalar(0.25f64).sqrt().unwrap(), Array::scalar(0.5f64));
        assert_eq!(Array::scalar(bf16::from_f32(0.25)).sqrt().unwrap(), Array::scalar(bf16::from_f32(0.5)));
        assert_eq!(Array::scalar(f16::from_f32(0.25)).sqrt().unwrap(), Array::scalar(f16::from_f32(0.5)));
        let input = ComplexNumber::new(0.7f64, -0.3f64);
        assert_abs_diff_eq!(Array::scalar(input).sqrt().unwrap(), Array::scalar(input.sqrt()), epsilon = 1e-12);
        // The principal branch maps the negative real axis to the positive imaginary axis.
        assert_abs_diff_eq!(
            Array::scalar(ComplexNumber::new(-4.0f64, 0.0)).sqrt().unwrap(),
            Array::scalar(ComplexNumber::new(0.0f64, 2.0)),
            epsilon = 1e-12,
        );

        assert_eq!(Array::scalar(4.0).sqrt().unwrap(), Array::scalar(2.0),);
    }

    #[test]
    fn test_sqrt_type_inference() {
        check_operation_type_inference!(
            @elementwise @unary,
            operation = SqrtOperation,
            cases = [
                {
                    input_data_types = [DataType::C64],
                    output_data_types = [DataType::C64],
                },
                {
                    input_data_types = [DataType::I32],
                    error = "'sqrt' does not support input data type i32",
                },
            ],
        );
        check_operation_type_inference!(
            @reject @unreduced,
            operation = SqrtOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_sqrt_batching() {
        check_operation_batching!(
            @approx(epsilon = 1e-9),
            operation = SqrtOperation::new(),
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), Array::vector(vec![0.5, 2.0]))],
                outputs = [(@mapped(axis = 0), Array::vector(vec![0.5f64.sqrt(), 2.0f64.sqrt()]))],
            }],
        );
    }

    #[test]
    fn test_sqrt_differentiation() {
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = SqrtOperation::new(),
            cases = [{
                primals = [Array::scalar(2.0)],
                tangents = [Array::scalar(3.0)],
                primal_outputs = [Array::scalar(2.0f64.sqrt())],
                tangent_outputs = [Array::scalar(3.0 / (2.0 * 2.0f64.sqrt()))],
                jvp = indoc! {"
                    lambda %0:f64[], %1:f64[] .
                    let %2:f64[] = sqrt %0
                        %3:f64[] = add %2 %2
                        %4:f64[] = div %1 %3
                    in (%2, %4)
                "},
            }],
        );
    }

    #[test]
    fn test_sqrt_complex_differentiation() {
        let input = ComplexNumber::new(0.7f64, -0.3f64);
        assert_eq!(
            gradient_holomorphic(|input| input.sqrt().unwrap(), Array::scalar(input)),
            Ok(Array::scalar(ComplexNumber::new(1.0, 0.0) / (input.sqrt() + input.sqrt()))),
        );
    }

    #[test]
    fn test_sqrt_low_precision_differentiation_uses_widened_tangents() {
        let primal = Array::from_f64s(ArrayType::scalar(DataType::F8E8M0FNU), vec![2.0]);
        let input_tangent = Array::from_f64s(ArrayType::scalar(DataType::F32), vec![3.0]);
        let (_, tangent) = jvp(|input| input.sqrt(), primal, input_tangent).unwrap();
        assert_eq!(tangent.r#type().as_ref(), &ArrayType::scalar(DataType::F32));
        // The tangent payload is honestly `f32`-encoded, so the comparison happens at `f32` precision.
        assert_abs_diff_eq!(tangent.to_f64s()[0], 3.0 / (2.0 * 2.0f64.sqrt()), epsilon = 1e-6);

        // The widened staged tangent program recomputes the denominator in the widened differential representation
        // instead of converting the narrower primal output.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F8E8M0FNU));
        let output = builder.add_instruction(SqrtOperation::new(), Vec::new(), vec![input]).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap()
            .jvp()
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f8e8m0fnu[], %1:f32[] .
                let %2:f8e8m0fnu[] = sqrt %0
                    %3:f32[] = convert_element_type [data_type=f32] %0
                    %4:f32[] = sqrt %3
                    %5:f32[] = add %4 %4
                    %6:f32[] = div %1 %5
                in (%2, %6)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_sqrt_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = SqrtOperation::new(),
            inputs = [Array::scalar(4.0)],
            expected = Array::scalar(2.0),
        );
    }

    #[test]
    fn test_sqrt_transposition() {
        check_operation_transposition!(
            @rejected,
            operation = SqrtOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_sqrt_for_primitives() {
        assert_eq!(Sqrt::sqrt(&4.0_f64), Ok(2.0));
    }
}
