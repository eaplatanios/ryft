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

/// Evaluates a polynomial at `x` with the provided coefficients ordered from the highest-degree term down to the
/// constant term, using Horner's scheme.
fn evaluate_polynomial(x: f64, coefficients: &[f64]) -> f64 {
    coefficients.iter().fold(0.0, |accumulator, coefficient| accumulator * x + coefficient)
}

/// Computes the Gauss error function `erf(x) = 2/√π · ∫₀ˣ e^{−t²} dt` in double precision.
///
/// This is the rational Chebyshev approximation from FDLIBM 5.3 (`s_erf.c`, developed at SunSoft and also used by
/// musl), split by argument magnitude: a short odd series below `2⁻²⁸`, a primary rational approximation on
/// `|x| < 0.84375`, a rational correction around `erf(1)` on `[0.84375, 1.25)`, two rational tail regimes evaluated
/// through the complementary function `1 − erfc(x)` on `[1.25, 6)`, and saturation to `±1` for `|x| ≥ 6` (where
/// `1 − erf(|x|) < 2⁻⁵⁶` is not representable next to one). The expected accuracy is about 1 ulp in double precision.
/// NaN inputs propagate and the sign symmetry `erf(−x) = −erf(x)` is exact, including for signed zeros. All
/// polynomial coefficient arrays below list the FDLIBM constants from the highest-degree term down to the constant
/// term, matching the [`evaluate_polynomial`] contract.
pub(crate) fn erf_f64(x: f64) -> f64 {
    /// `erf(1)` rounded toward zero, used as the base value of the `[0.84375, 1.25)` regime.
    const ERX: f64 = 8.45062911510467529297e-01;

    /// Coefficient of the short odd series below `2⁻²⁸`, equal to `8 · (2/√π − 1)`.
    const EFX8: f64 = 1.02703333676410069053e+00;

    /// Numerator coefficients of the primary regime `erf(x) = x + x · PP(x²)/QQ(x²)` on `|x| < 0.84375`.
    const PP: [f64; 5] = [
        -2.37630166566501626084e-05,
        -5.77027029648944159157e-03,
        -2.84817495755985104766e-02,
        -3.25042107247001499370e-01,
        1.28379167095512558561e-01,
    ];

    /// Denominator coefficients of the primary regime on `|x| < 0.84375`.
    const QQ: [f64; 6] = [
        -3.96022827877536812320e-06,
        1.32494738004321644526e-04,
        5.08130628187576562776e-03,
        6.50222499887672944485e-02,
        3.97917223959155352819e-01,
        1.0,
    ];

    /// Numerator coefficients of the regime `erf(x) = sign(x) · (ERX + PA(|x|−1)/QA(|x|−1))` on `[0.84375, 1.25)`.
    const PA: [f64; 7] = [
        -2.16637559486879084300e-03,
        3.54783043256182359371e-02,
        -1.10894694282396677476e-01,
        3.18346619901161753674e-01,
        -3.72207876035701323847e-01,
        4.14856118683748331666e-01,
        -2.36211856075265944077e-03,
    ];

    /// Denominator coefficients of the `[0.84375, 1.25)` regime.
    const QA: [f64; 7] = [
        1.19844998467991074170e-02,
        1.36370839120290507362e-02,
        1.26171219808761642112e-01,
        7.18286544141962662868e-02,
        5.40397917702171048937e-01,
        1.06420880400844228286e-01,
        1.0,
    ];

    /// Numerator coefficients of the complementary-function tail on `[1.25, 1/0.35)`, in the variable `1/x²`.
    const RA: [f64; 8] = [
        -9.81432934416914548592e+00,
        -8.12874355063065934246e+01,
        -1.84605092906711035994e+02,
        -1.62396669462573470355e+02,
        -6.23753324503260060396e+01,
        -1.05586262253232909814e+01,
        -6.93858572707181764372e-01,
        -9.86494403484714822705e-03,
    ];

    /// Denominator coefficients of the complementary-function tail on `[1.25, 1/0.35)`, in the variable `1/x²`.
    const SA: [f64; 9] = [
        -6.04244152148580987438e-02,
        6.57024977031928170135e+00,
        1.08635005541779435134e+02,
        4.29008140027567833386e+02,
        6.45387271733267880336e+02,
        4.34565877475229228821e+02,
        1.37657754143519042600e+02,
        1.96512716674392571292e+01,
        1.0,
    ];

    /// Numerator coefficients of the complementary-function tail on `[1/0.35, 6)`, in the variable `1/x²`.
    const RB: [f64; 7] = [
        -4.83519191608651397019e+02,
        -1.02509513161107724954e+03,
        -6.37566443368389627722e+02,
        -1.60636384855821916062e+02,
        -1.77579549177547519889e+01,
        -7.99283237680523006574e-01,
        -9.86494292470009928597e-03,
    ];

    /// Denominator coefficients of the complementary-function tail on `[1/0.35, 6)`, in the variable `1/x²`.
    const SB: [f64; 8] = [
        -2.24409524465858183362e+01,
        4.74528541206955367215e+02,
        2.55305040643316442583e+03,
        3.19985821950859553908e+03,
        1.53672958608443695994e+03,
        3.25792512996573918826e+02,
        3.03380607434824582924e+01,
        1.0,
    ];

    if x.is_nan() {
        return x;
    }
    let negative = x.is_sign_negative();
    let magnitude = x.abs();
    if magnitude < 0.84375 {
        if magnitude < 3.725290298461914e-09 {
            // For |x| < 2⁻²⁸ the series truncates to its leading odd term `x · 2/√π`, evaluated in a
            // scaled form that avoids intermediate underflow and preserves signed zeros.
            return 0.125 * (8.0 * x + EFX8 * x);
        }
        let squared = x * x;
        return x + x * (evaluate_polynomial(squared, &PP) / evaluate_polynomial(squared, &QQ));
    }
    if magnitude < 1.25 {
        let shifted = magnitude - 1.0;
        let correction = evaluate_polynomial(shifted, &PA) / evaluate_polynomial(shifted, &QA);
        return if negative { -ERX - correction } else { ERX + correction };
    }
    if magnitude >= 6.0 {
        // Covers |x| ≥ 6 and infinities: 1 − erf(6) < 2⁻⁵⁶ already rounds to zero next to one.
        return if negative { -1.0 } else { 1.0 };
    }
    // On [1.25, 6) the error function is evaluated through its complement as `erf(x) = sign(x) · (1 − erfc(|x|))`
    // with `erfc(y) = exp(−z² − 0.5625) · exp((z − y)(z + y) + R(1/y²)/S(1/y²)) / y`, where `z` is `y` with the low
    // half of its mantissa cleared so that `z²` is exact and the argument reduction stays accurate.
    let inverse_squared = 1.0 / (magnitude * magnitude);
    let quotient = if magnitude < 1.0 / 0.35 {
        evaluate_polynomial(inverse_squared, &RA) / evaluate_polynomial(inverse_squared, &SA)
    } else {
        evaluate_polynomial(inverse_squared, &RB) / evaluate_polynomial(inverse_squared, &SB)
    };
    let truncated = f64::from_bits(magnitude.to_bits() & 0xffff_ffff_0000_0000);
    let complement = (-truncated * truncated - 0.5625).exp()
        * ((truncated - magnitude) * (truncated + magnitude) + quotient).exp()
        / magnitude;
    if negative { complement - 1.0 } else { 1.0 - complement }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use half::{bf16, f16};
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{ArrayType, DataType};
    use crate::backends::arrays::Array;
    use crate::macros::{
        check_gradient, check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_transposition, check_operation_type_inference,
    };
    use crate::operations::math::{Reduce, ReductionKind};

    use super::*;

    /// `erf(0.5)`, `erf(1)`, `erf(2)`, and `erf(3)` correctly rounded to `f64`.
    const ERF_HALF: f64 = 0.5204998778130465;
    const ERF_ONE: f64 = 0.8427007929497149;
    const ERF_TWO: f64 = 0.9953222650189527;
    const ERF_THREE: f64 = 0.9999779095030014;

    #[test]
    fn test_erf() {
        // Exact fixed points and symmetry.
        assert_eq!(Array::scalar(0.0f64).erf().unwrap(), Array::scalar(0.0f64));
        assert_eq!(Array::scalar(f64::INFINITY).erf().unwrap(), Array::scalar(1.0f64));
        assert_eq!(Array::scalar(f64::NEG_INFINITY).erf().unwrap(), Array::scalar(-1.0f64));
        assert_eq!(Array::scalar(1.5f64).erf().unwrap(), -Array::scalar(-1.5f64).erf().unwrap());
        assert!(Array::scalar(f64::NAN).erf().unwrap().to_f64s()[0].is_nan());

        // Known values covering every rational-approximation regime of the reference implementation: the small
        // series (|x| < 2⁻²⁸), the primary interval (|x| < 0.84375), the [0.84375, 1.25) interval, both tail
        // intervals of the complementary-function path, and the saturated |x| ≥ 6 regime.
        assert_abs_diff_eq!(Array::scalar(1e-12f64).erf().unwrap(), Array::scalar(FRAC_2_SQRT_PI * 1e-12));
        assert_eq!(Array::scalar(0.5f64).erf().unwrap(), Array::scalar(ERF_HALF));
        assert_eq!(Array::scalar(1.0f64).erf().unwrap(), Array::scalar(ERF_ONE));
        assert_eq!(Array::scalar(2.0f64).erf().unwrap(), Array::scalar(ERF_TWO));
        assert_eq!(Array::scalar(3.0f64).erf().unwrap(), Array::scalar(ERF_THREE));
        assert_eq!(Array::scalar(4.0f64).erf().unwrap(), Array::scalar(0.9999999845827421));
        assert_eq!(Array::scalar(6.5f64).erf().unwrap(), Array::scalar(1.0f64));
        assert_eq!(Array::scalar(-6.5f64).erf().unwrap(), Array::scalar(-1.0f64));

        // The narrower variants round the double-precision evaluation to their own precision.
        assert_eq!(Array::scalar(0.5f32).erf().unwrap(), Array::scalar(ERF_HALF as f32));
        assert_eq!(Array::scalar(bf16::from_f32(0.5)).erf().unwrap(), Array::scalar(bf16::from_f64(ERF_HALF)),);
        assert_eq!(Array::scalar(f16::from_f32(0.5)).erf().unwrap(), Array::scalar(f16::from_f64(ERF_HALF)));

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
