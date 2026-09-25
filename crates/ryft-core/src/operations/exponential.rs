//! Operations that compute exponential and logarithmic functions elementwise. Each operation is defined by an
//! [`Operation`](crate::Operation) type (e.g., [`ExpOperation`]) together with a value capability trait (e.g., [`Exp`])
//! whose functions apply it to eager [`Array`](crate::Array)s and traced values alike, so the same code executes
//! immediately or records into a program depending on the value it runs on:
//!
//!   - [`Exp`] and [`Log`] compute the natural exponential and logarithm (i.e., `x ↦ eˣ` and `x ↦ ln(x)`, with the
//!     principal branch of the logarithm for complex values).
//!   - [`Ln1p`] computes `ln(1 + x)` accurately near zero.
//!   - [`LogAddExp`] computes `log(exp(a) + exp(b))` without forming potentially overflowing exponentials.
//!   - [`Logistic`] computes the logistic sigmoid (i.e., `x ↦ 1 / (1 + e^{-x})`).
//!
//! All operations support real floating-point and complex inputs. Complex logarithms use principal branches.
//! Unary operations preserve input metadata. [`LogAddExp`] promotes element types and broadcasts shapes. Its
//! low-precision real inputs use `f32` intermediates before conversion back to the output type. Inputs that carry
//! partial sums over unreduced mesh axes are rejected. These operations are nonlinear, so reverse-mode
//! differentiation transposes their linearizations.
//!
//! # Examples
//!
//! ```rust
//! # use ryft_core::{Array, Exp, Log, ProgramError};
//! # fn main() -> Result<(), ProgramError> {
//! assert_eq!(Array::scalar(0.0f64)?.exp()?, Array::scalar(1.0)?);
//! assert_eq!(Array::scalar(1.0f64)?.log()?, Array::scalar(0.0)?);
//! # Ok(())
//! # }
//! ```

use crate::arrays::{DataType, FloatingPointArrayElement};
use crate::differentiation::{
    DifferentiableType, DifferentiationDual, DifferentiationError, ElementwiseDerivativeAlignment,
};
use crate::macros::{
    check_count, define_elementwise_capability, define_elementwise_operation, impl_array_elementwise_operation,
    impl_differentiable_elementwise_operation, impl_differentiable_operation,
};
use crate::operations::arithmetic::{Add, Div, Mul, Sub};
use crate::operations::comparisons::{Compare, ComparisonDirection};
use crate::operations::complex::Real;
use crate::operations::constants::fill::Fill;
use crate::operations::constants::one_like::OneLike;
use crate::operations::constants::zero_like::ZeroLike;
use crate::operations::control_flow::select::Select;
use crate::operations::logical::And;
use crate::programs::{MaybeZero, ProgramError, Type, Typed, Value};

/// Canonical operation name for [`ExpOperation`].
pub const EXP_OPERATION_NAME: &str = "exp";

define_elementwise_operation!(
    @unary
    /// [`Operation`](crate::Operation) that computes the elementwise natural exponential of one value (i.e., `x ↦ eˣ`,
    /// the analytic continuation `e^z` on complex inputs) while preserving its array metadata. Only floating-point and
    /// complex inputs are supported, and inputs that still carry partial sums are rejected.
    ExpOperation,
    EXP_OPERATION_NAME,
    Exp,
    exp,
    check_data_types = [@float],
    check_array_types = [@no_unreduced],
);

impl_differentiable_elementwise_operation! {
    @unary
    ExpOperation,
    jvp<C> where C::Value: Mul {
        |(_, input_tangent) -> output| output.mul(&input_tangent)?
    },
    transpose = @nonlinear,
}

define_elementwise_capability!(
    @unary
    /// Represents the ability to compute elementwise natural exponentials. Concrete arrays compute immediately while
    /// context-carrying values apply [`ExpOperation`] through their context.
    Exp,
    /// Computes the natural exponential of each floating-point or complex element. Returns an error if the input types
    /// or metadata are unsupported.
    exp,
    ExpOperation,
);

impl_array_elementwise_operation!(
    @unary
    Exp,
    exp,
    operation = "exp",
    inputs = @float,
    checks = [@no_unreduced],
    |input| FloatingPointArrayElement::exp(input),
);

/// Implements [`Exp`] for one host primitive type.
macro_rules! impl_exp_for_primitive {
    ($type:ty) => {
        impl Exp for $type {
            #[inline]
            fn exp(&self) -> Result<Self, ProgramError> {
                Ok(<$type>::exp(*self))
            }
        }
    };
}

impl_exp_for_primitive!(f32);
impl_exp_for_primitive!(f64);

/// Canonical operation name for [`LogOperation`].
pub const LOG_OPERATION_NAME: &str = "log";

define_elementwise_operation!(
    @unary
    /// [`Operation`](crate::Operation) that computes the elementwise natural logarithm of one value (i.e., `x ↦ ln(x)`,
    /// the principal branch `ln(z)` on complex inputs) while preserving its array metadata. Only floating-point and
    /// complex inputs are supported, and inputs that still carry partial sums are rejected.
    LogOperation,
    LOG_OPERATION_NAME,
    Log,
    log,
    check_data_types = [@float],
    check_array_types = [@no_unreduced],
);

impl_differentiable_elementwise_operation! {
    @unary
    LogOperation,
    jvp<C> where C::Value: Div {
        |(input, input_tangent)| input_tangent.div(&input)?
    },
    transpose = @nonlinear,
}

define_elementwise_capability!(
    @unary
    /// Represents the ability to compute elementwise natural logarithms. Concrete arrays compute immediately while
    /// context-carrying values apply [`LogOperation`] through their context.
    Log,
    /// Computes the natural logarithm of each element, using the principal branch for complex values. Returns an error
    /// if the input types or metadata are unsupported.
    log,
    LogOperation,
);

impl_array_elementwise_operation!(
    @unary
    Log,
    log,
    operation = "log",
    inputs = @float,
    checks = [@no_unreduced],
    |input| FloatingPointArrayElement::log(input),
);

/// Implements [`Log`] for one host primitive type.
macro_rules! impl_log_for_primitive {
    ($type:ty) => {
        impl Log for $type {
            #[inline]
            fn log(&self) -> Result<Self, ProgramError> {
                Ok(<$type>::ln(*self))
            }
        }
    };
}

impl_log_for_primitive!(f32);
impl_log_for_primitive!(f64);

/// Canonical operation name for [`Ln1pOperation`].
pub const LN_1P_OPERATION_NAME: &str = "ln_1p";

define_elementwise_operation!(
    @unary
    /// [`Operation`](crate::Operation) that computes `ln(1 + x)` elementwise while retaining accuracy near zero
    /// and preserving input metadata. Real inputs below `-1` produce NaN, and `-1` produces negative infinity,
    /// subject to the output format's representation. Complex inputs use the principal logarithm of `1 + x`,
    /// with a branch cut on the real axis below `-1` and signed imaginary zeros selecting the side of the cut.
    /// Only floating-point and complex inputs are supported; inputs that carry partial sums are rejected.
    Ln1pOperation,
    LN_1P_OPERATION_NAME,
    Ln1p,
    ln_1p,
    check_data_types = [@float],
    check_array_types = [@no_unreduced],
);

impl_differentiable_elementwise_operation! {
    @unary
    Ln1pOperation,
    jvp<C>
    where
        C::Value: OneLike + Add + Div,
    {
        // `d(ln_1p(x)) = dx / (1 + x)`. The denominator is formed from the aligned input primal so that it carries
        // the tangent's element data type, and `one_like` supplies the one at exactly that type.
        |(input, input_tangent)| input_tangent.div(&input.one_like()?.add(&input)?)?
    },
    transpose = @nonlinear,
}

define_elementwise_capability!(
    @unary
    /// Represents the ability to compute elementwise `ln(1 + input)` accurately near zero. Concrete arrays compute
    /// immediately while context-carrying values apply [`Ln1pOperation`] through their context.
    Ln1p,
    /// Computes `ln(1 + input)` elementwise, retaining accuracy near zero and using the principal branch for complex
    /// inputs. Returns an error
    /// if the input types or metadata are unsupported.
    ln_1p,
    Ln1pOperation,
);

impl_array_elementwise_operation!(
    @unary
    Ln1p,
    ln_1p,
    operation = "ln_1p",
    inputs = @float,
    checks = [@no_unreduced],
    |input| FloatingPointArrayElement::ln_1p(input),
);

/// Implements [`Ln1p`] for one host primitive type.
macro_rules! impl_ln_1p_for_primitive {
    ($type:ty) => {
        impl Ln1p for $type {
            #[inline]
            fn ln_1p(&self) -> Result<Self, ProgramError> {
                Ok(<$type>::ln_1p(*self))
            }
        }
    };
}

impl_ln_1p_for_primitive!(f32);
impl_ln_1p_for_primitive!(f64);

/// Canonical operation name for [`LogAddExpOperation`].
pub const LOG_ADD_EXP_OPERATION_NAME: &str = "log_add_exp";

define_elementwise_operation!(
    @binary
    /// [`Operation`](crate::Operation) that computes `log(exp(a) + exp(b))` elementwise, promoting element types
    /// and broadcasting shapes. Factoring out the larger input avoids overflow in the intermediate exponentials.
    /// Output overflow still follows the element format's numerical conversion rules.
    ///
    /// Real inputs use `max(a, b) + ln_1p(exp(-abs(a - b)))`. Equal-sign infinities return that infinity, mixed
    /// infinities return positive infinity, and NaNs propagate. Half-precision and smaller formats use `f32`
    /// intermediates and round only the final output back to their element type.
    ///
    /// Complex inputs factor out the lexicographically larger input, evaluate the correction in component precision,
    /// and wrap the imaginary output into `[-π, π)`. This selects a principal logarithm branch; derivatives apply
    /// away from its cut. Inputs that carry partial sums are rejected, and reduced-axis markers must agree.
    LogAddExpOperation,
    LOG_ADD_EXP_OPERATION_NAME,
    LogAddExp,
    log_add_exp,
    check_data_types = [@float],
    check_array_types = [@no_unreduced, @same_reduced_axes],
);

impl_differentiable_operation! {
    <T> LogAddExpOperation<T>,
    jvp<C>
    where
        T: Type,
        C::Type: DifferentiableType,
        C::Value: ZeroLike
            + Add
            + Sub
            + Mul
            + Real
            + Exp
            + LogAddExp
            + And
            + Compare<C::Value>
            + Select
            + ElementwiseDerivativeAlignment<C::Type>,
        <C::Value as Value>::DispatchDomain: Fill<f64, C::Value>,
    {
        |_operation, context, _driver, inputs| {
            // The partial derivative with respect to each input is the softmax weight `exp(x - log_add_exp(a, b))`, and
            // so the tangent is `w_a · da + w_b · db`. Both weights are formed against the shared primal output, which
            // is therefore computed once. Following JAX's `_logaddexp_jvp`, every input and the primal output pass
            // through a `replace_infinity` guard that rewrites *positive* infinity to zero before the subtraction, so
            // that a `+∞` input yields the finite weights `exp(a)` and `1` rather than `exp(∞ - ∞) = NaN`. Negative
            // infinity is deliberately left in place, which is what makes the `(-∞, -∞)` tangent NaN.
            check_count!("input", inputs, 2, ProgramError);
            let left = &inputs[0];
            let right = &inputs[1];
            let primal = left.primal().log_add_exp(right.primal())?;
            let target = primal.r#type().tangent()?;
            let has_left_tangent = left.tangent().as_value().is_some();
            let has_right_tangent = right.tangent().as_value().is_some();
            if !has_left_tangent && !has_right_tangent {
                return Ok(vec![DifferentiationDual::new(primal, MaybeZero::Zero(target))?]);
            }

            if target.is_zero_space() {
                return Err(ProgramError::UnsupportedOperation {
                    message: format!(
                        "`{}` output type `{}` has no tangent space",
                        LOG_ADD_EXP_OPERATION_NAME,
                        primal.r#type(),
                    ),
                }
                .into());
            }

            let output_primal = primal;
            let primal = context.primal_to_tangent(output_primal.clone())?;

            // As with unary derivatives, evaluate coefficients in the differential representation
            // instead of converting an already rounded output from a narrower primal format.
            let aligned_primal = if output_primal.r#type().as_ref() == &target {
                primal.align_tangent(&target, &primal)?
            } else {
                let left = context.primal_to_tangent(left.primal().clone())?.align_tangent(&target, &primal)?;
                let right = context.primal_to_tangent(right.primal().clone())?.align_tangent(&target, &primal)?;
                left.log_add_exp(&right)?
            };

            // Inspect the real component for complex values. A literal infinity saturates in finite-only
            // formats, so first check that halving this constant actually leaves it infinite. This prevents
            // a representable finite maximum from being mistaken for positive infinity.
            let real = |value: &C::Value| {
                if target.is_complex() { value.real() } else { Ok(value.clone()) }
            };
            let real_output = real(&aligned_primal)?;
            let real_type = real_output.r#type().into_owned();
            let infinity = aligned_primal.dispatch_domain().fill(&real_type, f64::INFINITY)?;
            let half = aligned_primal.dispatch_domain().fill(&real_type, 0.5)?;
            let has_infinity = infinity.compare(&infinity.mul(&half)?, ComparisonDirection::Equal)?;
            let replace_infinity = |value: C::Value, component: C::Value| -> Result<C::Value, DifferentiationError> {
                let is_positive_infinity = component.compare(&infinity, ComparisonDirection::Equal)?;
                let is_positive_infinity = is_positive_infinity.and(&has_infinity)?;
                Ok(C::Value::select(&is_positive_infinity, &value.zero_like()?, &value)?)
            };

            let output_exponent = replace_infinity(aligned_primal, real_output)?;
            let left_term = left
                .tangent()
                .as_value()
                .map(|tangent| {
                    let input = context.primal_to_tangent(left.primal().clone())?.align_tangent(&target, &primal)?;
                    let component = real(&input)?;
                    let input = replace_infinity(input, component)?;
                    let weight = input.sub(&output_exponent)?.exp()?;
                    Ok::<_, DifferentiationError>(weight.mul(&tangent.align_tangent(&target, &primal)?)?)
                })
                .transpose()?;
            let right_term = right
                .tangent()
                .as_value()
                .map(|tangent| {
                    let input = context.primal_to_tangent(right.primal().clone())?.align_tangent(&target, &primal)?;
                    let component = real(&input)?;
                    let input = replace_infinity(input, component)?;
                    let weight = input.sub(&output_exponent)?.exp()?;
                    Ok::<_, DifferentiationError>(weight.mul(&tangent.align_tangent(&target, &primal)?)?)
                })
                .transpose()?;
            let tangent = match (left_term, right_term) {
                (Some(left), Some(right)) => MaybeZero::Value(left.add(&right)?),
                (Some(value), None) | (None, Some(value)) => MaybeZero::Value(value),
                (None, None) => MaybeZero::Zero(target),
            };
            Ok(vec![DifferentiationDual::new(output_primal, tangent)?])
        }
    },
    transpose = @nonlinear,
}

define_elementwise_capability!(
    @binary
    /// Represents the ability to compute stable elementwise log-sums of exponentials. Concrete arrays compute
    /// immediately while context-carrying values apply [`LogAddExpOperation`] through their context.
    LogAddExp,
    /// Computes `log(exp(self) + exp(other))` without forming potentially overflowing exponentials, promoting and
    /// broadcasting the inputs. Returns an error if the input types or metadata are unsupported.
    log_add_exp(other),
    LogAddExpOperation,
);

impl_array_elementwise_operation!(
    @binary
    LogAddExp,
    log_add_exp,
    operation = "log_add_exp",
    inputs = @float,
    checks = [@no_unreduced, @same_reduced_axes],
    |left, right| FloatingPointArrayElement::log_add_exp(left, right),
);

/// Implements [`LogAddExp`] for one host primitive type.
macro_rules! impl_log_add_exp_for_primitive {
    ($type:ty) => {
        impl LogAddExp for $type {
            fn log_add_exp(&self, other: &Self) -> Result<Self, ProgramError> {
                let delta = *self - *other;
                Ok(if delta.is_nan() {
                    *self + *other
                } else {
                    <$type>::max(*self, *other) + (-delta.abs()).exp().ln_1p()
                })
            }
        }
    };
}

impl_log_add_exp_for_primitive!(f32);
impl_log_add_exp_for_primitive!(f64);

/// Canonical operation name for [`LogisticOperation`].
pub const LOGISTIC_OPERATION_NAME: &str = "logistic";

define_elementwise_operation!(
    @unary
    /// [`Operation`](crate::Operation) that computes the elementwise logistic sigmoid of one value (i.e.,
    /// `x ↦ 1 / (1 + e^{-x})`, the analytic continuation on complex inputs) while preserving its array metadata.
    /// Only floating-point and complex inputs are supported, and inputs that still carry partial sums are rejected.
    LogisticOperation,
    LOGISTIC_OPERATION_NAME,
    Logistic,
    logistic,
    check_data_types = [@float],
    check_array_types = [@no_unreduced],
);

impl_differentiable_elementwise_operation! {
    @unary
    LogisticOperation,
    jvp<C>
    where
        C::Value: OneLike + Sub + Mul,
    {
        // `d(logistic(x)) = logistic(x) · (1 - logistic(x)) · dx`, reusing the primal output
        // evaluated at the tangent type.
        |(_, input_tangent) -> output| output.mul(&output.one_like()?.sub(&output)?)?.mul(&input_tangent)?
    },
    transpose = @nonlinear,
}

define_elementwise_capability!(
    @unary
    /// Represents the ability to compute the elementwise logistic function. Concrete arrays compute immediately while
    /// context-carrying values apply [`LogisticOperation`] through their context.
    Logistic,
    /// Computes `1 / (1 + exp(-input))` elementwise for floating-point or complex values. Returns an error if the input
    /// types or metadata are unsupported.
    logistic,
    LogisticOperation,
);

impl_array_elementwise_operation!(
    @unary
    Logistic,
    logistic,
    operation = "logistic",
    inputs = @float,
    checks = [@no_unreduced],
    |input| FloatingPointArrayElement::logistic(input),
);

/// Implements [`Logistic`] for one host primitive type.
macro_rules! impl_logistic_for_primitive {
    ($type:ty) => {
        impl Logistic for $type {
            #[inline]
            fn logistic(&self) -> Result<Self, ProgramError> {
                Ok(((-*self).exp() + 1.0).recip())
            }
        }
    };
}

impl_logistic_for_primitive!(f32);
impl_logistic_for_primitive!(f64);

// TODO(eaplatanios): Review from here onwards.

/// Returns whether the lowest value of `data_type` is an identity of the rounded pairwise [`LogAddExp`].
/// A binary fold rounds after every pair: once combining two sentinel values returns the sentinel, an all-sentinel
/// subtree of any size does too. There is therefore no scan-length bound. This contract does not apply to a
/// max-shifted sum, whose padding must remain neutral after subtraction of an arbitrary maximum.
pub(crate) fn is_log_add_exp_identity_data_type(data_type: DataType) -> bool {
    data_type.is_floating_point() && !matches!(data_type, DataType::F8E8M0FNU | DataType::F6E2M3FN)
}

/// Returns the diagnostic for a type whose lowest value cannot seed a rounded pairwise [`LogAddExp`] fold.
pub(crate) fn log_add_exp_identity_data_type_error(operation_name: &str, data_type: DataType) -> String {
    match data_type {
        _ if data_type.is_floating_point() => format!(
            "`{operation_name}` requires a floating-point format whose lowest value is a `log_add_exp` identity but \
             got `{data_type}`"
        ),
        _ => format!("`{operation_name}` requires real floating-point inputs but got `{data_type}`"),
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use half::{bf16, f16};
    use indoc::indoc;
    use num_complex::Complex as ComplexNumber;
    use pretty_assertions::assert_eq;

    use crate::arrays::{Array, ArrayOperation, ArrayType, DataType, f4e2m1fn, f8e4m3fn, f8e8m0fnu};
    use crate::contexts::EagerContext;
    use crate::differentiation::differentiate_at;
    use crate::interpretation::InterpretableOperation;
    use crate::macros::{
        check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_transposition, check_operation_type_inference,
    };
    use crate::parameters::Placeholder;
    use crate::programs::{EmptyRegionDriver, ProgramBuilder, TypeError, Typed};

    use super::*;

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
                    error = "`exp` does not support input data type `i32`",
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
    fn test_exp_interpretation() {
        assert_eq!(
            ExpOperation::<ArrayType>::new().interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[Array::scalar(0.0f64).unwrap()],
            ),
            Ok(vec![Array::scalar(1.0f64).unwrap()]),
        );
    }

    #[test]
    fn test_exp_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = ExpOperation::new(),
            inputs = [Array::scalar(0.7).unwrap()],
            expected = Array::scalar(0.7f64.exp()).unwrap(),
        );
    }

    #[test]
    fn test_exp_batching() {
        check_operation_batching!(
            @approx(epsilon = 1e-9),
            operation = ExpOperation::new(),
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), Array::vector(vec![0.5, -1.0]).unwrap())],
                outputs = [(@mapped(axis = 0), Array::vector(vec![0.5f64.exp(), (-1.0f64).exp()]).unwrap())],
            }],
        );
    }

    #[test]
    fn test_exp_differentiation() {
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = ExpOperation::new(),
            cases = [{
                primals = [Array::scalar(0.7).unwrap()],
                tangents = [Array::scalar(3.0).unwrap()],
                primal_outputs = [Array::scalar(0.7f64.exp()).unwrap()],
                tangent_outputs = [Array::scalar(3.0 * 0.7f64.exp()).unwrap()],
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
    fn test_exp_differentiation_complex() {
        let input = ComplexNumber::new(0.7f64, -0.3f64);
        assert_eq!(
            differentiate_at(Array::scalar(input).unwrap()).holomorphic().gradient(|input| input.exp().unwrap()),
            Ok(Array::scalar(input.exp()).unwrap()),
        );
    }

    #[test]
    fn test_exp_differentiation_low_precision_uses_widened_tangents() {
        let primal = Array::from_elements::<f8e8m0fnu>(
            ArrayType::scalar(DataType::F8E8M0FNU),
            &[2.0].map(|value| f8e8m0fnu::from_f64(value).unwrap()),
        )
        .unwrap();
        let input_tangent = Array::from_elements::<f32>(ArrayType::scalar(DataType::F32), &[3.0]).unwrap();
        let (primal_output, tangent) = differentiate_at(primal).jvp(input_tangent, |input| input.exp()).unwrap();
        // The primal keeps its exponent-only encoding: `exp(2) ≈ 7.39` rounds to the
        // nearest representable power of two, `8 = 2^3`, whose biased-exponent encoding is `0x82`.
        assert_eq!(primal_output.r#type().as_ref(), &ArrayType::scalar(DataType::F8E8M0FNU));
        assert_eq!(primal_output.logical_bytes(), vec![0x82]);
        assert_eq!(tangent.r#type().as_ref(), &ArrayType::scalar(DataType::F32));
        // The tangent is evaluated in its widened `f32` representation.
        assert_abs_diff_eq!(tangent.elements::<f32>().unwrap()[0], 3.0 * 2.0f32.exp(), epsilon = 1e-6);

        // The widened staged tangent program recomputes the coefficient in the widened differential representation
        // instead of converting the narrower primal output.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F8E8M0FNU));
        let output = builder.add_instruction(ExpOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
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
    fn test_exp_transposition() {
        check_operation_transposition!(
            @rejected,
            operation = ExpOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_array_exp() {
        assert_eq!(Array::scalar(0.5f32).unwrap().exp().unwrap(), Array::scalar(0.5f32.exp()).unwrap());
        assert_eq!(Array::scalar(0.5f64).unwrap().exp().unwrap(), Array::scalar(0.5f64.exp()).unwrap());
        assert_eq!(
            Array::scalar(bf16::from_f32(0.5)).unwrap().exp().unwrap(),
            Array::scalar(bf16::from_f32(0.5f32.exp())).unwrap(),
        );
        assert_eq!(
            Array::scalar(f16::from_f32(0.5)).unwrap().exp().unwrap(),
            Array::scalar(f16::from_f32(0.5f32.exp())).unwrap(),
        );
        let input = ComplexNumber::new(0.7f64, -0.3f64);
        assert_abs_diff_eq!(
            Array::scalar(input).unwrap().exp().unwrap(),
            Array::scalar(input.exp()).unwrap(),
            epsilon = 1e-12,
        );
        // Euler's identity: e^{iπ} = -1.
        assert_abs_diff_eq!(
            Array::scalar(ComplexNumber::new(0.0f64, std::f64::consts::PI)).unwrap().exp().unwrap(),
            Array::scalar(ComplexNumber::new(-1.0f64, 0.0)).unwrap(),
            epsilon = 1e-12,
        );

        let vector = Array::vector(vec![0.0, 1.0]).unwrap();
        assert_abs_diff_eq!(vector.exp().unwrap(), Array::vector(vec![1.0, 1.0f64.exp()]).unwrap(), epsilon = 1e-12);
    }

    #[test]
    fn test_array_exp_complex() {
        // Elementwise complex math decodes and encodes the complex element types directly.
        let left = Array::vector(vec![ComplexNumber::new(1.0f64, 2.0), ComplexNumber::new(0.5f64, -1.0)]).unwrap();
        let left_values = [ComplexNumber::new(1.0f64, 2.0), ComplexNumber::new(0.5f64, -1.0)];
        assert_abs_diff_eq!(
            left.exp().unwrap(),
            Array::vector(vec![left_values[0].exp(), left_values[1].exp()]).unwrap(),
            epsilon = 1e-12,
        );
    }

    #[test]
    fn test_array_exp_low_precision() {
        // Low-precision formats decode, compute, and re-encode without constructing intermediary scalar values.
        let low_precision = Array::from_elements(
            ArrayType::new_static(DataType::F8E4M3FN, [2]),
            &[f8e4m3fn::from_f64(0.0).unwrap(), f8e4m3fn::from_f64(1.0).unwrap()],
        )
        .unwrap();
        assert_eq!(
            low_precision.exp().unwrap().elements::<f8e4m3fn>(),
            Ok(vec![f8e4m3fn::from_f64(1.0).unwrap(), f8e4m3fn::from_bits(0x43)]),
        );
    }

    #[test]
    fn test_exp_primitives() {
        assert_eq!(Exp::exp(&0.0f32), Ok(1.0));
        assert_eq!(Exp::exp(&0.0f64), Ok(1.0));
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
                    error = "`log` does not support input data type `i32`",
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
    fn test_log_interpretation() {
        assert_eq!(
            LogOperation::<ArrayType>::new().interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[Array::scalar(1.0f64).unwrap()],
            ),
            Ok(vec![Array::scalar(0.0f64).unwrap()]),
        );
    }

    #[test]
    fn test_log_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = LogOperation::new(),
            inputs = [Array::scalar(0.7).unwrap()],
            expected = Array::scalar(0.7f64.ln()).unwrap(),
        );
    }

    #[test]
    fn test_log_batching() {
        check_operation_batching!(
            @approx(epsilon = 1e-9),
            operation = LogOperation::new(),
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), Array::vector(vec![0.5, 2.0]).unwrap())],
                outputs = [(@mapped(axis = 0), Array::vector(vec![0.5f64.ln(), 2.0f64.ln()]).unwrap())],
            }],
        );
    }

    #[test]
    fn test_log_differentiation() {
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = LogOperation::new(),
            cases = [{
                primals = [Array::scalar(0.7).unwrap()],
                tangents = [Array::scalar(3.0).unwrap()],
                primal_outputs = [Array::scalar(0.7f64.ln()).unwrap()],
                tangent_outputs = [Array::scalar(3.0 / 0.7).unwrap()],
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
    fn test_log_differentiation_complex() {
        // The analytic quotient and the scalar division algorithm may round their intermediate values differently.
        let input = ComplexNumber::new(0.7f64, -0.3f64);
        assert_abs_diff_eq!(
            differentiate_at(Array::scalar(input).unwrap())
                .holomorphic()
                .gradient(|input| input.log().unwrap())
                .unwrap(),
            Array::scalar(ComplexNumber::new(1.0, 0.0) / input).unwrap(),
            epsilon = 1e-15,
        );
    }

    #[test]
    fn test_log_differentiation_low_precision_uses_widened_tangents() {
        let primal = Array::from_elements::<f8e8m0fnu>(
            ArrayType::scalar(DataType::F8E8M0FNU),
            &[2.0].map(|value| f8e8m0fnu::from_f64(value).unwrap()),
        )
        .unwrap();
        let input_tangent = Array::from_elements::<f32>(ArrayType::scalar(DataType::F32), &[3.0]).unwrap();
        let (_, tangent) = differentiate_at(primal).jvp(input_tangent, |input| input.log()).unwrap();
        assert_eq!(tangent.r#type().as_ref(), &ArrayType::scalar(DataType::F32));
        assert_abs_diff_eq!(tangent.elements::<f32>().unwrap()[0], 1.5, epsilon = 1e-9);

        // The widened staged tangent program divides by the input converted to the widened differential
        // representation.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F8E8M0FNU));
        let output = builder.add_instruction(LogOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
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
    fn test_log_transposition() {
        check_operation_transposition!(
            @rejected,
            operation = LogOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_array_log() {
        assert_eq!(Array::scalar(0.5f32).unwrap().log().unwrap(), Array::scalar(0.5f32.ln()).unwrap());
        assert_eq!(Array::scalar(0.5f64).unwrap().log().unwrap(), Array::scalar(0.5f64.ln()).unwrap());
        assert_eq!(
            Array::scalar(bf16::from_f32(0.5)).unwrap().log().unwrap(),
            Array::scalar(bf16::from_f32(0.5f32.ln())).unwrap(),
        );
        assert_eq!(
            Array::scalar(f16::from_f32(0.5)).unwrap().log().unwrap(),
            Array::scalar(f16::from_f32(0.5f32.ln())).unwrap(),
        );
        let input = ComplexNumber::new(0.7f64, -0.3f64);
        assert_abs_diff_eq!(
            Array::scalar(input).unwrap().log().unwrap(),
            Array::scalar(input.ln()).unwrap(),
            epsilon = 1e-12,
        );
        // The principal branch maps the negative real axis to `ln|x| + iπ`.
        assert_abs_diff_eq!(
            Array::scalar(ComplexNumber::new(-1.0f64, 0.0)).unwrap().log().unwrap(),
            Array::scalar(ComplexNumber::new(0.0f64, std::f64::consts::PI)).unwrap(),
            epsilon = 1e-12,
        );

        assert_abs_diff_eq!(
            Array::vector(vec![1.0, std::f64::consts::E]).unwrap().log().unwrap(),
            Array::vector(vec![0.0, 1.0]).unwrap(),
            epsilon = 1e-12,
        );

        // Complex arrays retain both components throughout elementwise decoding and encoding.
        let values = [ComplexNumber::new(1.0f64, 2.0), ComplexNumber::new(0.5f64, -1.0)];
        assert_abs_diff_eq!(
            Array::vector(values.to_vec()).unwrap().log().unwrap(),
            Array::vector(vec![values[0].ln(), values[1].ln()]).unwrap(),
            epsilon = 1e-12,
        );
    }

    #[test]
    fn test_log_primitives() {
        assert_eq!(Log::log(&1.0f32), Ok(0.0));
        assert_eq!(Log::log(&1.0f64), Ok(0.0));
    }

    #[test]
    fn test_ln_1p_type_inference() {
        check_operation_type_inference!(
            @elementwise @unary,
            operation = Ln1pOperation,
            cases = [
                {
                    input_data_types = [DataType::F64],
                    output_data_types = [DataType::F64],
                },
                {
                    input_data_types = [DataType::C64],
                    output_data_types = [DataType::C64],
                },
                {
                    input_data_types = [DataType::I32],
                    error = "`ln_1p` does not support input data type `i32`",
                },
            ],
        );
        check_operation_type_inference!(
            @reject @unreduced,
            operation = Ln1pOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_ln_1p_interpretation() {
        assert_eq!(
            Ln1pOperation::<ArrayType>::new().interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[Array::scalar(0.0f64).unwrap()],
            ),
            Ok(vec![Array::scalar(0.0f64).unwrap()]),
        );
    }

    #[test]
    fn test_ln_1p_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = Ln1pOperation::new(),
            inputs = [Array::scalar(0.7).unwrap()],
            expected = Array::scalar(0.7f64.ln_1p()).unwrap(),
        );
    }

    #[test]
    fn test_ln_1p_batching() {
        check_operation_batching!(
            @approx(epsilon = 1e-9),
            operation = Ln1pOperation::new(),
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), Array::vector(vec![0.5, -0.5]).unwrap())],
                outputs = [(@mapped(axis = 0), Array::vector(vec![0.5f64.ln_1p(), (-0.5f64).ln_1p()]).unwrap())],
            }],
        );
    }

    #[test]
    fn test_ln_1p_differentiation() {
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = Ln1pOperation::new(),
            cases = [{
                primals = [Array::scalar(0.7).unwrap()],
                tangents = [Array::scalar(3.0).unwrap()],
                primal_outputs = [Array::scalar(0.7f64.ln_1p()).unwrap()],
                tangent_outputs = [Array::scalar(3.0 / 1.7).unwrap()],
                jvp = indoc! {"
                    lambda %0:f64[], %1:f64[] .
                    let %2:f64[] = ln_1p %0
                        %3:f64[] = one_like %0
                        %4:f64[] = add %3 %0
                        %5:f64[] = div %1 %4
                    in (%2, %5)
                "},
            }],
        );
    }

    #[test]
    fn test_ln_1p_differentiation_complex() {
        let input = ComplexNumber::new(0.5f64, 0.5);
        assert_abs_diff_eq!(
            differentiate_at(Array::scalar(input).unwrap())
                .holomorphic()
                .gradient(|input| input.ln_1p())
                .unwrap(),
            Array::scalar(ComplexNumber::new(0.6, -0.2)).unwrap(),
            epsilon = 1e-15,
        );
    }

    #[test]
    fn test_ln_1p_differentiation_unrepresentable_tangent() {
        // The primal saturates to the finite minimum, but a live zero tangent requires an unrepresentable NaN.
        assert_eq!(
            differentiate_at(Array::scalar(f4e2m1fn::from_f64(-1.0).unwrap()).unwrap())
                .jvp(Array::scalar(f4e2m1fn::from_f64(0.0).unwrap()).unwrap(), |input| input.ln_1p(),),
            Err(DifferentiationError::Program(TypeError::invalid("data type `f4e2m1fn` cannot represent NaN").into())),
        );
    }

    #[test]
    fn test_ln_1p_transposition() {
        check_operation_transposition!(
            @rejected,
            operation = Ln1pOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_array_ln_1p() {
        // Native and half-precision inputs retain their element types.
        assert_eq!(Array::scalar(0.5f32).unwrap().ln_1p().unwrap(), Array::scalar(0.5f32.ln_1p()).unwrap());
        assert_eq!(Array::scalar(0.5f64).unwrap().ln_1p().unwrap(), Array::scalar(0.5f64.ln_1p()).unwrap());
        assert_eq!(
            Array::scalar(bf16::from_f32(0.5)).unwrap().ln_1p().unwrap(),
            Array::scalar(bf16::from_f32(0.5f32.ln_1p())).unwrap(),
        );
        assert_eq!(
            Array::scalar(f16::from_f32(0.5)).unwrap().ln_1p().unwrap(),
            Array::scalar(f16::from_f32(0.5f32.ln_1p())).unwrap(),
        );

        // The fixed point and the boundary values of the real domain.
        assert_eq!(Array::scalar(0.0f64).unwrap().ln_1p().unwrap(), Array::scalar(0.0f64).unwrap());
        assert_eq!(Array::scalar(-1.0f64).unwrap().ln_1p().unwrap(), Array::scalar(f64::NEG_INFINITY).unwrap());
        assert!(Array::scalar(-2.0f64).unwrap().ln_1p().unwrap().elements::<f64>().unwrap()[0].is_nan());

        // This input is lost by forming `1 + x`, but its logarithm rounds back to `x` itself.
        assert_eq!(Array::scalar(1e-20f64).unwrap().ln_1p().unwrap(), Array::scalar(1e-20f64).unwrap());
    }

    #[test]
    fn test_array_ln_1p_complex() {
        assert_abs_diff_eq!(
            Array::scalar(ComplexNumber::new(0.0f64, 1.0)).unwrap().ln_1p().unwrap(),
            Array::scalar(ComplexNumber::new(std::f64::consts::LN_2 / 2.0, std::f64::consts::FRAC_PI_4)).unwrap(),
            epsilon = 1e-15,
        );
    }

    #[test]
    fn test_ln_1p_primitives() {
        assert_eq!(Ln1p::ln_1p(&0.0f64), Ok(0.0));
        assert_eq!(Ln1p::ln_1p(&0.0f32), Ok(0.0));
    }

    #[test]
    fn test_log_add_exp_type_inference() {
        check_operation_type_inference!(
            @elementwise @binary,
            operation = LogAddExpOperation,
            cases = [
                {
                    input_data_types = [DataType::F32, DataType::F64],
                    output_data_types = [DataType::F64],
                },
                {
                    input_data_types = [DataType::C64, DataType::C64],
                    output_data_types = [DataType::C64],
                },
                {
                    input_data_types = [DataType::I32, DataType::F32],
                    error = "`log_add_exp` does not support input data type `i32`",
                },
            ],
        );
        check_operation_type_inference!(
            @reject @unreduced,
            operation = LogAddExpOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64), ArrayType::scalar(DataType::F64)],
        );
        check_operation_type_inference!(
            @reject @mismatched_reduced,
            operation = LogAddExpOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64), ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_log_add_exp_interpretation() {
        assert_eq!(
            LogAddExpOperation::<ArrayType>::new().interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[Array::scalar(0.0f64).unwrap(), Array::scalar(f64::NEG_INFINITY).unwrap()],
            ),
            Ok(vec![Array::scalar(0.0f64).unwrap()]),
        );
    }

    #[test]
    fn test_log_add_exp_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = LogAddExpOperation::new(),
            inputs = [Array::scalar(0.0f64).unwrap(), Array::scalar(0.0f64).unwrap()],
            expected = Array::scalar(std::f64::consts::LN_2).unwrap(),
        );
    }

    #[test]
    fn test_log_add_exp_batching() {
        check_operation_batching!(
            @approx(epsilon = 1e-9),
            operation = LogAddExpOperation::new(),
            axis_size = 2,
            cases = [{
                inputs = [
                    (@mapped(axis = 0), Array::vector(vec![0.5, -1.0]).unwrap()),
                    (@replicated, Array::scalar(2.0).unwrap()),
                ],
                outputs = [(
                    @mapped(axis = 0),
                    Array::vector(vec![2.2014132779827524f64, 2.048587351573742f64]).unwrap(),
                )],
            }],
        );
    }

    #[test]
    fn test_log_add_exp_differentiation() {
        // The tangent is the softmax-weighted combination of the input tangents.
        let (left, right) = (0.7f64, -0.3f64);
        let (left_tangent, right_tangent) = (0.4f64, -0.2f64);
        let output = 1.0132616875182228f64;
        let tangent = (left - output).exp() * left_tangent + (right - output).exp() * right_tangent;
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = LogAddExpOperation::new(),
            cases = [{
                primals = [Array::scalar(left).unwrap(), Array::scalar(right).unwrap()],
                tangents = [Array::scalar(left_tangent).unwrap(), Array::scalar(right_tangent).unwrap()],
                primal_outputs = [Array::scalar(output).unwrap()],
                tangent_outputs = [Array::scalar(tangent).unwrap()],
                jvp = indoc! {"
                    lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[] .
                    let %4:f64[] = log_add_exp %0 %1
                        %5:f64[] = constant [value=inf]
                        %6:f64[] = constant [value=0.5]
                        %7:f64[] = mul %5 %6
                        %8:bool[] = compare [direction=Equal] %5 %7
                        %9:bool[] = compare [direction=Equal] %4 %5
                        %10:bool[] = and %9 %8
                        %11:f64[] = zero_like %4
                        %12:f64[] = select %10 %11 %4
                        %13:bool[] = compare [direction=Equal] %0 %5
                        %14:bool[] = and %13 %8
                        %15:f64[] = zero_like %0
                        %16:f64[] = select %14 %15 %0
                        %17:f64[] = sub %16 %12
                        %18:f64[] = exp %17
                        %19:f64[] = mul %18 %2
                        %20:bool[] = compare [direction=Equal] %1 %5
                        %21:bool[] = and %20 %8
                        %22:f64[] = zero_like %1
                        %23:f64[] = select %21 %22 %1
                        %24:f64[] = sub %23 %12
                        %25:f64[] = exp %24
                        %26:f64[] = mul %25 %3
                        %27:f64[] = add %19 %26
                    in (%4, %27)
                "},
            }],
        );
    }

    #[test]
    fn test_log_add_exp_differentiation_exceptional_tangents() {
        // These five results are conventions of the differentiation rule rather than mathematical extensions of the
        // primal, so they are pinned exactly. They follow from replacing only *positive* infinity with zero before
        // the weight subtraction, exactly as JAX's `_logaddexp_jvp` does.
        let jvp = |primals: (f64, f64), tangents: (f64, f64)| {
            differentiate_at((Array::scalar(primals.0).unwrap(), Array::scalar(primals.1).unwrap()))
                .jvp((Array::scalar(tangents.0).unwrap(), Array::scalar(tangents.1).unwrap()), |(left, right)| {
                    left.log_add_exp(&right)
                })
                .unwrap()
                .1
                .elements::<f64>()
                .unwrap()[0]
        };

        // Both weights become `exp(0 - 0) = 1`, so the tangents simply add.
        assert_eq!(jvp((f64::INFINITY, f64::INFINITY), (2.0, 3.0)), 5.0);
        // Negative infinity is not replaced, so both weights are `exp(-∞ - -∞) = exp(NaN)`.
        assert!(jvp((f64::NEG_INFINITY, f64::NEG_INFINITY), (2.0, 3.0)).is_nan());
        // The replaced `+∞` output makes the finite input's weight `exp(a)` instead of zero.
        assert_eq!(jvp((1.0, f64::INFINITY), (2.0, 3.0)), 1.0f64.exp() * 2.0 + 3.0);
        // A `-∞` input contributes nothing and the finite input carries the whole tangent.
        assert_eq!(jvp((1.0, f64::NEG_INFINITY), (2.0, 3.0)), 2.0);
        // A NaN input propagates through both the primal and the weights.
        assert!(jvp((f64::NAN, 1.0), (2.0, 3.0)).is_nan());
        assert!(jvp((1.0, f64::NAN), (2.0, 3.0)).is_nan());
    }

    #[test]
    fn test_log_add_exp_differentiation_finite_maximum() {
        // Infinity literals saturate in this format; its finite maximum must retain the finite-input derivative.
        let left = Array::scalar(f4e2m1fn::from_f64(6.0).unwrap()).unwrap();
        let right = Array::scalar(f4e2m1fn::from_f64(0.0).unwrap()).unwrap();
        let tangent = Array::scalar(f4e2m1fn::from_f64(1.0).unwrap()).unwrap();
        let (output, tangent) = differentiate_at((left.clone(), right.clone()))
            .jvp((right.clone(), tangent), |(left, right)| left.log_add_exp(&right))
            .unwrap();
        assert_eq!(output, left);
        assert_eq!(tangent, right);
    }

    #[test]
    fn test_log_add_exp_differentiation_widened_tangent() {
        let left = Array::scalar(f8e8m0fnu::from_f64(1.0).unwrap()).unwrap();
        let right = Array::scalar(f8e8m0fnu::from_f64(0.5).unwrap()).unwrap();
        let (output, tangent) = differentiate_at((left.clone(), right))
            .jvp((Array::scalar(1.0f32).unwrap(), Array::scalar(0.0f32).unwrap()), |(left, right)| {
                left.log_add_exp(&right)
            })
            .unwrap();
        assert_eq!(output, left);
        assert_abs_diff_eq!(tangent.elements::<f32>().unwrap()[0], 0.62245935f32, epsilon = 1e-7);
    }

    #[test]
    fn test_log_add_exp_differentiation_complex() {
        // Equal complex inputs have two coefficients of one half, away from the principal branch cut.
        let input = Array::scalar(ComplexNumber::new(1.0f64, 0.5)).unwrap();
        let zero = Array::scalar(ComplexNumber::new(0.0f64, 0.0)).unwrap();
        let two = Array::scalar(ComplexNumber::new(2.0f64, 0.0)).unwrap();
        let (output, tangent) = differentiate_at((input.clone(), input))
            .jvp((two, zero), |(left, right)| left.log_add_exp(&right))
            .unwrap();
        assert_abs_diff_eq!(
            output,
            Array::scalar(ComplexNumber::new(1.0 + std::f64::consts::LN_2, 0.5)).unwrap(),
            epsilon = 1e-15,
        );
        assert_abs_diff_eq!(tangent, Array::scalar(ComplexNumber::new(1.0, 0.0)).unwrap(), epsilon = 1e-15);
    }

    #[test]
    fn test_log_add_exp_transposition() {
        check_operation_transposition!(
            @rejected,
            operation = LogAddExpOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64), ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_array_log_add_exp() {
        // Native and half-precision inputs retain their element types.
        assert_eq!(
            Array::scalar(1.0f64).unwrap().log_add_exp(&Array::scalar(2.0f64).unwrap()).unwrap(),
            Array::scalar(2.313261687518223f64).unwrap(),
        );
        assert_eq!(
            Array::scalar(1.0f32).unwrap().log_add_exp(&Array::scalar(2.0f32).unwrap()).unwrap(),
            Array::scalar(2.0f32 + (-1.0f32).exp().ln_1p()).unwrap(),
        );
        assert_eq!(
            Array::scalar(bf16::from_f32(1.0))
                .unwrap()
                .log_add_exp(&Array::scalar(bf16::from_f32(2.0)).unwrap())
                .unwrap(),
            Array::scalar(bf16::from_f32(2.0f32 + (-1.0f32).exp().ln_1p())).unwrap(),
        );
        assert_eq!(
            Array::scalar(f16::from_f32(1.0))
                .unwrap()
                .log_add_exp(&Array::scalar(f16::from_f32(2.0)).unwrap())
                .unwrap(),
            Array::scalar(f16::from_f32(2.0f32 + (-1.0f32).exp().ln_1p())).unwrap(),
        );

        // The operation is symmetric, and two equal inputs add exactly `log(2)`.
        assert_eq!(
            Array::scalar(2.0f64).unwrap().log_add_exp(&Array::scalar(1.0f64).unwrap()).unwrap(),
            Array::scalar(2.313261687518223f64).unwrap(),
        );
        assert_eq!(
            Array::scalar(0.0f64).unwrap().log_add_exp(&Array::scalar(0.0f64).unwrap()).unwrap(),
            Array::scalar(std::f64::consts::LN_2).unwrap(),
        );

        // The reason the primitive exists: neither exponential is ever formed, so inputs far outside the range of
        // `exp` still produce the exact shifted result instead of infinity.
        assert_eq!(
            Array::scalar(1000.0f64).unwrap().log_add_exp(&Array::scalar(1000.0f64).unwrap()).unwrap(),
            Array::scalar(1000.0 + std::f64::consts::LN_2).unwrap(),
        );
        assert!((1000.0f64.exp() + 1000.0f64.exp()).ln().is_infinite());

        // The pinned exceptional values: same-sign infinities saturate, mixed infinities return the larger input,
        // and NaN propagates from either input.
        assert_eq!(
            Array::scalar(f64::INFINITY).unwrap().log_add_exp(&Array::scalar(f64::INFINITY).unwrap()).unwrap(),
            Array::scalar(f64::INFINITY).unwrap(),
        );
        assert_eq!(
            Array::scalar(f64::NEG_INFINITY)
                .unwrap()
                .log_add_exp(&Array::scalar(f64::NEG_INFINITY).unwrap())
                .unwrap(),
            Array::scalar(f64::NEG_INFINITY).unwrap(),
        );
        assert_eq!(
            Array::scalar(1.0f64).unwrap().log_add_exp(&Array::scalar(f64::INFINITY).unwrap()).unwrap(),
            Array::scalar(f64::INFINITY).unwrap(),
        );
        assert_eq!(
            Array::scalar(1.0f64).unwrap().log_add_exp(&Array::scalar(f64::NEG_INFINITY).unwrap()).unwrap(),
            Array::scalar(1.0f64).unwrap(),
        );
        assert!(
            Array::scalar(f64::NAN)
                .unwrap()
                .log_add_exp(&Array::scalar(1.0f64).unwrap())
                .unwrap()
                .elements::<f64>()
                .unwrap()[0]
                .is_nan()
        );
        assert!(
            Array::scalar(1.0f64)
                .unwrap()
                .log_add_exp(&Array::scalar(f64::NAN).unwrap())
                .unwrap()
                .elements::<f64>()
                .unwrap()[0]
                .is_nan()
        );
    }

    #[test]
    fn test_array_log_add_exp_complex() {
        let input = Array::scalar(ComplexNumber::new(1.0f64, 4.0)).unwrap();
        assert_abs_diff_eq!(
            input.log_add_exp(&input).unwrap(),
            Array::scalar(ComplexNumber::new(1.0 + std::f64::consts::LN_2, 4.0 - std::f64::consts::TAU)).unwrap(),
            epsilon = 1e-15,
        );
    }

    #[test]
    fn test_log_add_exp_primitives() {
        assert_eq!(LogAddExp::log_add_exp(&0.0f64, &0.0), Ok(std::f64::consts::LN_2));
        assert_eq!(LogAddExp::log_add_exp(&0.0f32, &0.0), Ok(std::f32::consts::LN_2));
    }

    #[test]
    fn test_logistic_type_inference() {
        check_operation_type_inference!(
            @elementwise @unary,
            operation = LogisticOperation,
            cases = [
                {
                    input_data_types = [DataType::C64],
                    output_data_types = [DataType::C64],
                },
                {
                    input_data_types = [DataType::I32],
                    error = "`logistic` does not support input data type `i32`",
                },
            ],
        );
        check_operation_type_inference!(
            @reject @unreduced,
            operation = LogisticOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_logistic_interpretation() {
        assert_eq!(
            LogisticOperation::<ArrayType>::new().interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[Array::scalar(0.0f64).unwrap()],
            ),
            Ok(vec![Array::scalar(0.5f64).unwrap()]),
        );
    }

    #[test]
    fn test_logistic_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = LogisticOperation::new(),
            inputs = [Array::scalar(0.7).unwrap()],
            expected = Array::scalar(1.0 / (1.0 + (-0.7f64).exp())).unwrap(),
        );
    }

    #[test]
    fn test_logistic_batching() {
        check_operation_batching!(
            @approx(epsilon = 1e-9),
            operation = LogisticOperation::new(),
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), Array::vector(vec![0.5, -1.0]).unwrap())],
                outputs = [(
                    @mapped(axis = 0),
                    Array::vector(vec![1.0 / (1.0 + (-0.5f64).exp()), 1.0 / (1.0 + 1.0f64.exp())]).unwrap(),
                )],
            }],
        );
    }

    #[test]
    fn test_logistic_differentiation() {
        let logistic = 1.0 / (1.0 + (-0.7f64).exp());
        let expected_tangent = 3.0 * logistic * (1.0 - logistic);
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = LogisticOperation::new(),
            cases = [{
                primals = [Array::scalar(0.7).unwrap()],
                tangents = [Array::scalar(3.0).unwrap()],
                primal_outputs = [Array::scalar(logistic).unwrap()],
                tangent_outputs = [Array::scalar(expected_tangent).unwrap()],
            }],
        );
    }

    #[test]
    fn test_logistic_differentiation_complex() {
        let input = ComplexNumber::new(0.7f64, -0.3f64);
        let expected = {
            let logistic = ComplexNumber::new(1.0, 0.0) / (ComplexNumber::new(1.0, 0.0) + (-input).exp());
            logistic * (ComplexNumber::new(1.0, 0.0) - logistic)
        };
        assert_abs_diff_eq!(
            differentiate_at(Array::scalar(input).unwrap())
                .holomorphic()
                .gradient(|input| input.logistic())
                .unwrap(),
            Array::scalar(expected).unwrap(),
            epsilon = 1e-12,
        );
    }

    #[test]
    fn test_logistic_transposition() {
        check_operation_transposition!(
            @rejected,
            operation = LogisticOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_array_logistic() {
        assert_eq!(
            Array::scalar(0.5f32).unwrap().logistic().unwrap(),
            Array::scalar(1.0 / (1.0 + (-0.5f32).exp())).unwrap(),
        );
        assert_eq!(
            Array::scalar(0.5f64).unwrap().logistic().unwrap(),
            Array::scalar(1.0 / (1.0 + (-0.5f64).exp())).unwrap(),
        );
        assert_eq!(
            Array::scalar(bf16::from_f32(0.5)).unwrap().logistic().unwrap(),
            Array::scalar(bf16::from_f32(1.0 / (1.0 + (-0.5f32).exp()))).unwrap(),
        );
        assert_eq!(
            Array::scalar(f16::from_f32(0.5)).unwrap().logistic().unwrap(),
            Array::scalar(f16::from_f32(1.0 / (1.0 + (-0.5f32).exp()))).unwrap(),
        );
        let input = ComplexNumber::new(0.7f64, -0.3f64);
        let expected = ComplexNumber::new(1.0, 0.0) / (ComplexNumber::new(1.0, 0.0) + (-input).exp());
        assert_abs_diff_eq!(
            Array::scalar(input).unwrap().logistic().unwrap(),
            Array::scalar(expected).unwrap(),
            epsilon = 1e-12,
        );
    }

    #[test]
    fn test_logistic_primitives() {
        assert_eq!(Logistic::logistic(&0.0f32), Ok(0.5));
        assert_eq!(Logistic::logistic(&0.0f64), Ok(0.5));
    }

    #[test]
    fn test_is_log_add_exp_identity_data_type() {
        assert!(is_log_add_exp_identity_data_type(DataType::F32));
        assert!(is_log_add_exp_identity_data_type(DataType::F8E4M3FN));
        assert!(!is_log_add_exp_identity_data_type(DataType::F8E8M0FNU));
        assert!(is_log_add_exp_identity_data_type(DataType::F4E2M1FN));
        assert!(is_log_add_exp_identity_data_type(DataType::F6E3M2FN));
        assert!(is_log_add_exp_identity_data_type(DataType::F8E4M3B11FNUZ));
        assert!(!is_log_add_exp_identity_data_type(DataType::F6E2M3FN));
        assert!(!is_log_add_exp_identity_data_type(DataType::I32));
        assert!(!is_log_add_exp_identity_data_type(DataType::C64));
    }

    #[test]
    fn test_log_add_exp_identity_data_type_error() {
        assert_eq!(
            log_add_exp_identity_data_type_error("reduce_log_sum_exp", DataType::F8E8M0FNU),
            "`reduce_log_sum_exp` requires a floating-point format whose lowest value is a `log_add_exp` identity but got \
             `f8e8m0fnu`",
        );
        assert_eq!(
            log_add_exp_identity_data_type_error("reduce_log_sum_exp", DataType::F6E2M3FN),
            "`reduce_log_sum_exp` requires a floating-point format whose lowest value is a `log_add_exp` identity but got \
             `f6e2m3fn`",
        );
        assert_eq!(
            log_add_exp_identity_data_type_error("reduce_log_sum_exp", DataType::I32),
            "`reduce_log_sum_exp` requires real floating-point inputs but got `i32`",
        );
    }
}
