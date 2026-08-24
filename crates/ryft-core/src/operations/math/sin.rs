use std::ops::Mul as StandardMul;

use crate::macros::{
    define_elementwise_capability, define_elementwise_operation, impl_differentiable_elementwise_operation,
};
use crate::operations::math::cos::Cos;
use crate::programs::ProgramError;

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`SinOperation`].
pub const SIN_OPERATION_NAME: &str = "sin";

define_elementwise_operation!(
    @unary
    /// [`Operation`] that computes the elementwise sine of a floating-point or complex value while
    /// preserving its array metadata. Array operands that still carry partial sums are rejected.
    SinOperation, SIN_OPERATION_NAME,
    Sin, sin,
    check_data_types = [@float],
    check_array_types = [@no_unreduced],
);

impl_differentiable_elementwise_operation! {
    @unary
    SinOperation,
    jvp<C> where C::Value: Cos + StandardMul<Output = C::Value> {
        |(input, input_tangent)| input.cos()? * input_tangent
    },
    transpose = @nonlinear,
}

define_elementwise_capability!(
    @unary
    /// Value-level elementwise sine capability. [`Sin`] fills the same role for [`SinOperation`] that
    /// [`std::ops::Add`] and [`std::ops::Neg`] fill for their corresponding arithmetic
    /// [`Operation`]s.
    Sin,
    /// Computes [`SinOperation`] elementwise for this value.
    sin,
    SinOperation,
);

/// Implements [`Sin`] for one host primitive type.
macro_rules! impl_capability_for_primitive {
    ($type:ty) => {
        impl Sin for $type {
            fn sin(&self) -> Result<Self, ProgramError> {
                Ok(<$type>::sin(*self))
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
    use crate::differentiation::differentiate_at;
    use crate::macros::{
        check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_transposition, check_operation_type_inference,
    };
    use crate::parameters::Placeholder;
    use crate::programs::{ProgramBuilder, Typed};

    use super::*;

    #[test]
    fn test_sin() {
        assert_eq!(Array::scalar(0.5f32).sin().unwrap(), Array::scalar(0.5f32.sin()));
        assert_eq!(Array::scalar(0.5f64).sin().unwrap(), Array::scalar(0.5f64.sin()));
        assert_eq!(Array::scalar(bf16::from_f32(0.5)).sin().unwrap(), Array::scalar(bf16::from_f32(0.5f32.sin())),);
        assert_eq!(Array::scalar(f16::from_f32(0.5)).sin().unwrap(), Array::scalar(f16::from_f32(0.5f32.sin())),);
        let extreme = Array::scalar(ComplexNumber::new(0.0f64, 1000.0))
            .sin()
            .unwrap()
            .elements::<ComplexNumber<f64>>()
            .unwrap()[0];
        assert_eq!(extreme.re, 0.0);
        assert!(extreme.im.is_infinite() && extreme.im.is_sign_positive());

        assert_eq!(Array::scalar(0.5).sin().unwrap(), Array::scalar(0.5f64.sin()),);
    }

    #[test]
    fn test_sin_type_inference() {
        check_operation_type_inference!(
            @elementwise @unary,
            operation = SinOperation,
            cases = [
                {
                    input_data_types = [DataType::F64],
                    output_data_types = [DataType::F64],
                },
                {
                    input_data_types = [DataType::I32],
                    error = "`sin` does not support input data type i32",
                },
            ],
        );
        check_operation_type_inference!(
            @reject @unreduced,
            operation = SinOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_sin_batching() {
        check_operation_batching!(
            @approx(epsilon = 1e-9),
            operation = SinOperation::new(),
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), Array::vector(vec![0.5, -1.0]))],
                outputs = [(@mapped(axis = 0), Array::vector(vec![0.5f64.sin(), (-1.0f64).sin()]))],
            }],
        );
    }

    #[test]
    fn test_sin_differentiation() {
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = SinOperation::new(),
            cases = [{
                primals = [Array::scalar(2.0)],
                tangents = [Array::scalar(3.0)],
                primal_outputs = [Array::scalar(2.0f64.sin())],
                tangent_outputs = [Array::scalar(3.0 * 2.0f64.cos())],
                jvp = indoc! {"
                    lambda %0:f64[], %1:f64[] .
                    let %2:f64[] = sin %0
                        %3:f64[] = cos %0
                        %4:f64[] = mul %3 %1
                    in (%2, %4)
                "},
            }],
        );
    }

    #[test]
    fn test_sin_complex_differentiation() {
        let input = ComplexNumber::new(0.7f64, -0.3f64);
        assert_eq!(
            differentiate_at(Array::scalar(input)).holomorphic().gradient(|input| input.sin().unwrap()),
            Ok(Array::scalar(input.cos())),
        );
    }

    #[test]
    fn test_sin_low_precision_differentiation_uses_widened_tangents() {
        let primal = Array::from_f64s(ArrayType::scalar(DataType::F8E8M0FNU), vec![2.0]);
        let input_tangent = Array::from_f64s(ArrayType::scalar(DataType::F32), vec![3.0]);
        let (_, tangent) = differentiate_at(primal).jvp(input_tangent, |input| input.sin()).unwrap();
        assert_eq!(tangent.r#type().as_ref(), &ArrayType::scalar(DataType::F32));
        // The tangent payload is honestly `f32`-encoded, so the comparison happens at `f32` precision.
        assert_abs_diff_eq!(tangent.to_f64s()[0], 3.0 * 2.0f64.cos(), epsilon = 1e-6);

        // The widened staged tangent program computes the coefficient in the widened differential representation.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F8E8M0FNU));
        let output = builder.add_instruction(SinOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap()
            .jvp()
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f8e8m0fnu[], %1:f32[] .
                let %2:f8e8m0fnu[] = sin %0
                    %3:f32[] = convert_element_type [data_type=f32] %0
                    %4:f32[] = cos %3
                    %5:f32[] = mul %4 %1
                in (%2, %5)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_sin_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = SinOperation::new(),
            inputs = [Array::scalar(0.5)],
            expected = Array::scalar(0.5f64.sin()),
        );
    }

    #[test]
    fn test_sin_transposition() {
        check_operation_transposition!(
            @rejected,
            operation = SinOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_sin_for_primitives() {
        assert_eq!(Sin::sin(&0.0_f64), Ok(0.0));
    }
}
