use std::ops::Mul as StandardMul;

use crate::macros::{
    define_elementwise_capability, define_elementwise_operation, impl_differentiable_elementwise_operation,
};
use crate::programs::ProgramError;

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`ExpOperation`].
pub const EXP_OPERATION_NAME: &str = "exp";

define_elementwise_operation!(
    @unary
    /// [`Operation`] that computes the elementwise natural exponential of one value (i.e.,
    /// `x ↦ eˣ`, the analytic continuation `e^z` on complex operands) while preserving its array metadata. Only
    /// floating-point and complex operands are supported, and operands that still carry partial sums are rejected.
    ExpOperation, EXP_OPERATION_NAME,
    Exp, exp,
    check_data_types = [@float],
    check_array_types = [@no_unreduced],
);

impl_differentiable_elementwise_operation! {
    @unary
    ExpOperation,
    jvp<C> where C::Value: StandardMul<Output = C::Value> {
        |(_, input_tangent) -> output| output * input_tangent
    },
    transpose = @nonlinear,
}

define_elementwise_capability!(
    @unary
    /// Value-level elementwise natural-exponential capability. [`Exp`] fills the same role for
    /// [`ExpOperation`] that [`Sin`](crate::Sin) fills for [`SinOperation`](crate::SinOperation).
    Exp,
    /// Computes [`ExpOperation`] elementwise for this value.
    exp,
    ExpOperation,
);

/// Implements [`Exp`] for one host primitive type.
macro_rules! impl_capability_for_primitive {
    ($type:ty) => {
        impl Exp for $type {
            fn exp(&self) -> Result<Self, ProgramError> {
                Ok(<$type>::exp(*self))
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
    fn test_exp() {
        assert_eq!(Array::scalar(0.5f32).exp().unwrap(), Array::scalar(0.5f32.exp()));
        assert_eq!(Array::scalar(0.5f64).exp().unwrap(), Array::scalar(0.5f64.exp()));
        assert_eq!(Array::scalar(bf16::from_f32(0.5)).exp().unwrap(), Array::scalar(bf16::from_f32(0.5f32.exp())),);
        assert_eq!(Array::scalar(f16::from_f32(0.5)).exp().unwrap(), Array::scalar(f16::from_f32(0.5f32.exp())),);
        let input = ComplexNumber::new(0.7f64, -0.3f64);
        assert_abs_diff_eq!(Array::scalar(input).exp().unwrap(), Array::scalar(input.exp()), epsilon = 1e-12);
        // Euler's identity: e^{iπ} = -1.
        assert_abs_diff_eq!(
            Array::scalar(ComplexNumber::new(0.0f64, std::f64::consts::PI)).exp().unwrap(),
            Array::scalar(ComplexNumber::new(-1.0f64, 0.0)),
            epsilon = 1e-12,
        );

        assert_eq!(Array::scalar(0.7).exp().unwrap(), Array::scalar(0.7f64.exp()),);
    }

    #[test]
    fn test_exp_type_inference() {
        check_operation_type_inference!(
            @elementwise @unary,
            operation = ExpOperation,
            cases = [
                {
                    input_data_types = [DataType::C64],
                    output_data_types = [DataType::C64],
                },
                {
                    input_data_types = [DataType::I32],
                    error = "`exp` does not support input data type i32",
                },
            ],
        );
        check_operation_type_inference!(
            @reject @unreduced,
            operation = ExpOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_exp_batching() {
        check_operation_batching!(
            @approx(epsilon = 1e-9),
            operation = ExpOperation::new(),
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), Array::vector(vec![0.5, -1.0]))],
                outputs = [(@mapped(axis = 0), Array::vector(vec![0.5f64.exp(), (-1.0f64).exp()]))],
            }],
        );
    }

    #[test]
    fn test_exp_differentiation() {
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = ExpOperation::new(),
            cases = [{
                primals = [Array::scalar(0.7)],
                tangents = [Array::scalar(3.0)],
                primal_outputs = [Array::scalar(0.7f64.exp())],
                tangent_outputs = [Array::scalar(3.0 * 0.7f64.exp())],
                jvp = indoc! {"
                    lambda %0:f64[], %1:f64[] .
                    let %2:f64[] = exp %0
                        %3:f64[] = mul %2 %1
                    in (%2, %3)
                "},
            }],
        );
    }

    #[test]
    fn test_exp_complex_differentiation() {
        let input = ComplexNumber::new(0.7f64, -0.3f64);
        assert_eq!(
            differentiate_at(Array::scalar(input)).holomorphic().gradient(|input| input.exp().unwrap()),
            Ok(Array::scalar(input.exp())),
        );
    }

    #[test]
    fn test_exp_low_precision_differentiation_uses_widened_tangents() {
        let primal = Array::from_f64s(ArrayType::scalar(DataType::F8E8M0FNU), vec![2.0]);
        let input_tangent = Array::from_f64s(ArrayType::scalar(DataType::F32), vec![3.0]);
        let (primal_output, tangent) = differentiate_at(primal).jvp(input_tangent, |input| input.exp()).unwrap();
        // The primal output stays genuinely `f8e8m0fnu`-encoded (not an `f64` pun): `exp(2) ≈ 7.39` rounds to the
        // nearest representable power of two, `8 = 2^3`, whose biased-exponent encoding is `0x82`.
        assert_eq!(primal_output.r#type().as_ref(), &ArrayType::scalar(DataType::F8E8M0FNU));
        assert_eq!(primal_output.logical_bytes(), vec![0x82]);
        assert_eq!(tangent.r#type().as_ref(), &ArrayType::scalar(DataType::F32));
        // The tangent payload is honestly `f32`-encoded, so the comparison happens at `f32` precision.
        assert_abs_diff_eq!(tangent.to_f64s()[0], 3.0 * 2.0f64.exp(), epsilon = 1e-6);

        // The widened staged tangent program recomputes the coefficient in the widened differential representation
        // instead of converting the narrower primal output.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F8E8M0FNU));
        let output = builder.add_instruction(ExpOperation::new(), Vec::new(), vec![input]).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap()
            .jvp()
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f8e8m0fnu[], %1:f32[] .
                let %2:f8e8m0fnu[] = exp %0
                    %3:f32[] = convert_element_type [data_type=f32] %0
                    %4:f32[] = exp %3
                    %5:f32[] = mul %4 %1
                in (%2, %5)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_exp_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = ExpOperation::new(),
            inputs = [Array::scalar(0.7)],
            expected = Array::scalar(0.7f64.exp()),
        );
    }

    #[test]
    fn test_exp_transposition() {
        check_operation_transposition!(
            @rejected,
            operation = ExpOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_exp_for_primitives() {
        assert_eq!(Exp::exp(&0.0_f64), Ok(1.0));
    }
}
