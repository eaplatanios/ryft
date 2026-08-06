use std::f64::consts::FRAC_2_SQRT_PI;
use std::ops::{Mul as StandardMul, Neg as StandardNeg};

use crate::macros::{
    define_elementwise_capability, define_elementwise_operation, impl_differentiable_elementwise_operation,
};
use crate::operations::constants::Fill;
use crate::operations::manipulation::conversion::ElementType;
use crate::programs::types::Typed;
use crate::programs::values::Value;

// TODO(eaplatanios): Review this module.

use super::Exp;

/// Canonical operation name for [`ErfOperation`].
pub const ERF_OPERATION_NAME: &str = "erf";

define_elementwise_operation!(
    @unary
    /// [`Operation`] that computes the elementwise Gauss error function of one value (i.e.,
    /// `x ↦ erf(x) = 2/√π · ∫₀ˣ e^{−t²} dt`) while preserving its array metadata. Only real floating-point operands
    /// are supported, and operands that still carry partial sums are rejected.
    ErfOperation, ERF_OPERATION_NAME,
    Erf, erf,
    check_data_types = [@float @real],
    check_array_types = [@no_unreduced],
);

impl_differentiable_elementwise_operation! {
    @unary
    ErfOperation,
    jvp<C>
    where
        C::Type: ElementType,
        C::Value: Exp + StandardMul<Output = C::Value> + StandardNeg<Output = C::Value>,
        <C::Value as Value>::DispatchDomain: Fill<f64, C::Value>,
    {
        // d(erf(x)) = (2/√π) · exp(-x²) · dx, with the coefficient `2/√π` rounded to the aligned input's element
        // data type and staged as a nullary fill of the aligned input type.
        |(input, input_tangent)| {
            let input_type = input.r#type().into_owned();
            let coefficient = input.dispatch_domain().fill(&input_type, FRAC_2_SQRT_PI)?;
            coefficient * (-(input.clone() * input)).exp()? * input_tangent
        }
    },
    transpose = @nonlinear,
}

define_elementwise_capability!(
    @unary
    /// Value-level elementwise error-function capability. [`Erf`] fills the same role for
    /// [`ErfOperation`] that [`Sin`](crate::Sin) fills for [`SinOperation`](crate::SinOperation).
    Erf,
    /// Computes [`ErfOperation`] elementwise for this value.
    erf,
    ErfOperation,
);

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use half::{bf16, f16};
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::backends::arrays::Array;
    use crate::backends::scalars::Scalar;
    use crate::macros::{
        check_gradient, check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_transposition, check_operation_type_inference,
    };
    use crate::operations::math::{Reduce, ReductionKind};
    use crate::types::{ArrayType, DataType};

    use super::*;

    /// `erf(0.5)`, `erf(1)`, `erf(2)`, and `erf(3)` correctly rounded to `f64`.
    const ERF_HALF: f64 = 0.5204998778130465;
    const ERF_ONE: f64 = 0.8427007929497149;
    const ERF_TWO: f64 = 0.9953222650189527;
    const ERF_THREE: f64 = 0.9999779095030014;

    #[test]
    fn test_erf() {
        // Exact fixed points and symmetry.
        assert_eq!(Scalar::from(0.0f64).erf().unwrap(), 0.0f64);
        assert_eq!(Scalar::from(f64::INFINITY).erf().unwrap(), 1.0f64);
        assert_eq!(Scalar::from(f64::NEG_INFINITY).erf().unwrap(), -1.0f64);
        assert_eq!(Scalar::from(1.5f64).erf().unwrap(), -Scalar::from(-1.5f64).erf().unwrap());
        let Scalar::F64(not_a_number) = Scalar::from(f64::NAN).erf().unwrap() else { panic!("expected an f64 result") };
        assert!(not_a_number.is_nan());

        // Known values covering every rational-approximation regime of the reference implementation: the small
        // series (|x| < 2⁻²⁸), the primary interval (|x| < 0.84375), the [0.84375, 1.25) interval, both tail
        // intervals of the complementary-function path, and the saturated |x| ≥ 6 regime.
        assert_abs_diff_eq!(Scalar::from(1e-12f64).erf().unwrap(), Scalar::from(FRAC_2_SQRT_PI * 1e-12));
        assert_eq!(Scalar::from(0.5f64).erf().unwrap(), ERF_HALF);
        assert_eq!(Scalar::from(1.0f64).erf().unwrap(), ERF_ONE);
        assert_eq!(Scalar::from(2.0f64).erf().unwrap(), ERF_TWO);
        assert_eq!(Scalar::from(3.0f64).erf().unwrap(), ERF_THREE);
        assert_eq!(Scalar::from(4.0f64).erf().unwrap(), 0.9999999845827421);
        assert_eq!(Scalar::from(6.5f64).erf().unwrap(), 1.0f64);
        assert_eq!(Scalar::from(-6.5f64).erf().unwrap(), -1.0f64);

        // The narrower variants round the double-precision evaluation to their own precision.
        assert_eq!(Scalar::from(0.5f32).erf().unwrap(), ERF_HALF as f32);
        assert_eq!(Scalar::from(bf16::from_f32(0.5)).erf().unwrap(), bf16::from_f64(ERF_HALF));
        assert_eq!(Scalar::from(f16::from_f32(0.5)).erf().unwrap(), f16::from_f64(ERF_HALF));

        assert_eq!(Array::scalar(0.5).erf().unwrap(), Array::scalar(ERF_HALF));
    }

    #[test]
    fn test_erf_type_inference() {
        check_operation_type_inference!(
            @elementwise @unary,
            operation = ErfOperation,
            cases = [
                {
                    input_data_types = [DataType::F64],
                    output_data_types = [DataType::F64],
                },
                {
                    input_data_types = [DataType::C64],
                    error = "'erf' does not support input data type c64",
                },
                {
                    input_data_types = [DataType::I32],
                    error = "'erf' does not support input data type i32",
                },
            ],
        );
        check_operation_type_inference!(
            @reject @unreduced,
            operation = ErfOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_erf_batching() {
        check_operation_batching!(
            @approx(epsilon = 1e-9),
            operation = ErfOperation::new(),
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), Array::vector(vec![0.5, -1.0]))],
                outputs = [(@mapped(axis = 0), Array::vector(vec![ERF_HALF, -ERF_ONE]))],
            }],
        );
    }

    #[test]
    fn test_erf_differentiation() {
        let expected_tangent = 3.0 * FRAC_2_SQRT_PI * (-0.7f64 * 0.7).exp();
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = ErfOperation::new(),
            cases = [{
                primals = [Array::scalar(0.7)],
                tangents = [Array::scalar(3.0)],
                primal_outputs = [Array::scalar(0.6778011938374184)],
                tangent_outputs = [Array::scalar(expected_tangent)],
                jvp = indoc! {"
                    lambda %0:f64[], %1:f64[] .
                    let %2:f64[] = erf %0
                        %3:f64[] = constant [value=[1.1283791670955126]]
                        %4:f64[] = mul %0 %0
                        %5:f64[] = neg %4
                        %6:f64[] = exp %5
                        %7:f64[] = mul %3 %6
                        %8:f64[] = mul %7 %1
                    in (%2, %8)
                "},
            }],
        );
        check_gradient!(
            |x| x.erf().map(|values| values.reduce(&[0], ReductionKind::Sum)),
            at = Array::vector(vec![-2.5f64, -0.3, 0.0, 0.9, 3.0]),
            step = 1e-6,
            tolerance = 1e-6,
        );
    }

    #[test]
    fn test_erf_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = ErfOperation::new(),
            inputs = [Array::scalar(0.5)],
            expected = Array::scalar(ERF_HALF),
        );
    }

    #[test]
    fn test_erf_transposition() {
        check_operation_transposition!(
            @rejected,
            operation = ErfOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64)],
        );
    }
}
