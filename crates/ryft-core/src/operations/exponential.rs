//! Operations that compute exponential and logarithmic functions elementwise. Each operation is defined by an
//! [`Operation`](crate::Operation) type (e.g., [`ExpOperation`]) together with a value capability trait (e.g., [`Exp`])
//! whose functions apply it to eager [`Array`](crate::Array)s and traced values alike, so the same code executes
//! immediately or records into a program depending on the value it runs on:
//!
//!   - [`Exp`] and [`Log`] compute the natural exponential and logarithm (i.e., `x ↦ eˣ` and `x ↦ ln(x)`, with the
//!     principal branch of the logarithm for complex values).
//!   - [`Log1p`] computes `log(1 + x)` as a single operation, which keeps full relative accuracy for inputs near zero.
//!   - [`LogAddExp`] computes `log(exp(a) + exp(b))` without forming either exponential, so that it cannot overflow.
//!   - [`Logistic`] computes the logistic sigmoid (i.e., `x ↦ 1 / (1 + e^{-x})`).
//!
//! [`Exp`], [`Log`], and [`Logistic`] support floating-point and complex values, same as StableHLO's
//! [`exponential`](https://openxla.org/stablehlo/spec#exponential), [`log`](https://openxla.org/stablehlo/spec#log),
//! and [`logistic`](https://openxla.org/stablehlo/spec#logistic), whereas [`Log1p`] and [`LogAddExp`] support only
//! real floating-point values. Unary operations preserve the metadata of their input, and [`LogAddExp`] promotes the
//! element types and broadcasts the shapes of its inputs. Inputs that carry partial sums over unreduced mesh axes are
//! rejected. Every operation is nonlinear, so reverse-mode differentiation transposes its linearization instead.
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

use crate::arrays::{DataType, FloatingPointArrayElement, RealFloatingPointArrayElement};
use crate::differentiation::{
    DifferentiableType, DifferentiationDual, DifferentiationError, ElementwiseDerivativeAlignment,
};
use crate::macros::{
    check_count, define_elementwise_capability, define_elementwise_operation, impl_array_elementwise_operation,
    impl_differentiable_elementwise_operation, impl_differentiable_operation,
};
use crate::operations::comparisons::{Compare, ComparisonDirection};
use crate::operations::constants::fill::Fill;
use crate::operations::constants::one_like::OneLike;
use crate::operations::constants::zero_like::ZeroLike;
use crate::operations::control_flow::select::Select;
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
    jvp<C> where C::Value: std::ops::Mul<Output = C::Value> {
        |(_, input_tangent) -> output| output * input_tangent
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
    jvp<C> where C::Value: std::ops::Div<Output = C::Value> {
        |(input, input_tangent)| input_tangent / input
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

// TODO(eaplatanios): Rename `Log1p` to `Ln1p` and `log1p` to `ln_1p` to match Rust's conventions.
/// Canonical operation name for [`Log1pOperation`].
pub const LOG1P_OPERATION_NAME: &str = "log1p";

define_elementwise_operation!(
    @unary
    /// [`Operation`](crate::Operation) that computes the elementwise natural logarithm of one plus its input (i.e.,
    /// `x ↦ log(1 + x)`) while preserving its array metadata. The name matches the canonical mathematical spelling that
    /// Rust's own [`f64::ln_1p`] uses.
    ///
    /// The point of the primitive is accuracy near zero. Evaluating `log(1 + x)` by first forming `1 + x` loses every
    /// bit of `x` below the precision of one, so a small `x` returns a result whose relative error grows without bound
    /// as `x` shrinks. Computing the composition as a single operation keeps full relative accuracy there, which is why
    /// `log1p` is the form used by log-likelihood and log-probability code.
    ///
    /// Only real floating-point inputs are supported, and inputs that still carry partial sums are rejected. The
    /// complex logarithm needs a different construction (i.e., a principal branch and a separate accurate magnitude
    /// near `-1`), and that is construction is not currently supported here.
    Log1pOperation,
    LOG1P_OPERATION_NAME,
    Log1p,
    log1p,
    check_data_types = [@float @real],
    check_array_types = [@no_unreduced],
);

impl_differentiable_elementwise_operation! {
    @unary
    Log1pOperation,
    jvp<C>
    where
        C::Value: OneLike + std::ops::Add<Output = C::Value> + std::ops::Div<Output = C::Value>,
    {
        // d(log1p(x)) = dx / (1 + x). The denominator is formed from the aligned input primal so that it carries the
        // tangent's element data type, and `one_like` supplies the one at exactly that type.
        |(input, input_tangent)| input_tangent / (input.one_like()? + input)
    },
    transpose = @nonlinear,
}

define_elementwise_capability!(
    @unary
    /// Represents the ability to compute elementwise `log(1 + input)` accurately near zero. Concrete arrays compute
    /// immediately while context-carrying values apply [`Log1pOperation`] through their context.
    Log1p,
    /// Computes `log(1 + input)` for each real floating-point element, retaining accuracy near zero. Returns an error
    /// if the input types or metadata are unsupported.
    log1p,
    Log1pOperation,
);

impl_array_elementwise_operation!(
    @unary
    Log1p,
    log1p,
    operation = "log1p",
    inputs = @float @real,
    checks = [@no_unreduced],
    |input| RealFloatingPointArrayElement::log1p(input),
);

/// Implements [`Log1p`] for one host primitive type.
macro_rules! impl_log1p_for_primitive {
    ($type:ty) => {
        impl Log1p for $type {
            #[inline]
            fn log1p(&self) -> Result<Self, ProgramError> {
                Ok(self.ln_1p())
            }
        }
    };
}

impl_log1p_for_primitive!(f32);
impl_log1p_for_primitive!(f64);

/// Canonical operation name for [`LogAddExpOperation`].
pub const LOG_ADD_EXP_OPERATION_NAME: &str = "log_add_exp";

define_elementwise_operation!(
    @binary
    /// [`Operation`](crate::Operation) that computes the elementwise `log(exp(a) + exp(b))` of its inputs without
    /// forming either exponential, promoting their element types and broadcasting their shapes.
    ///
    /// The semantics are:
    ///
    /// ```text
    /// log_add_exp(a, b) = select(is_nan(a - b), a + b, max(a, b) + log1p(exp(-|a - b|)))
    /// ```
    ///
    /// and are borrowed from JAX's [`logaddexp`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.logaddexp.html).
    ///
    /// Factoring the larger input out of the sum is what makes the primitive usable across the whole real range:
    /// `exp(-|a - b|)` never overflows and so `log_add_exp(1000, 1000)` is exactly `1000 + log(2)` where the naive
    /// composition returns infinity.
    ///
    /// The `is_nan(a - b)` guard is for the cases in which the difference itself is undefined, and it fixes the
    /// following cases:
    ///
    ///   - `(+∞, +∞) ↦ +∞` and `(-∞, -∞) ↦ -∞`, through the `a + b` branch,
    ///   - any NaN input propagates NaN, also through the `a + b` branch,
    ///   - mixed infinities return the larger input, through the ordinary branch: `-|a - b|` is `-∞`, so
    ///     `log1p(exp(-∞)) = log1p(0) = 0` and the result is `max(a, b)`.
    ///
    /// Only real floating-point inputs are supported, and array inputs that still carry partial sums are rejected,
    /// with their reduced-axis markers required to agree.
    LogAddExpOperation,
    LOG_ADD_EXP_OPERATION_NAME,
    LogAddExp,
    log_add_exp,
    check_data_types = [@float @real],
    check_array_types = [@no_unreduced, @same_reduced_axes],
);

impl_differentiable_operation! {
    <T> LogAddExpOperation<T>,
    jvp<C>
    where
        T: Type,
        C::Type: DifferentiableType,
        C::Value: ZeroLike
            + Exp
            + LogAddExp
            + Compare<C::Value>
            + Select
            + std::ops::Add<Output = C::Value>
            + std::ops::Sub<Output = C::Value>
            + std::ops::Mul<Output = C::Value>
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
            let aligned_primal = primal.align_tangent(&target, &primal)?;
            let infinity = aligned_primal.dispatch_domain().fill(&target, f64::INFINITY)?;
            let replace_infinity = |value: C::Value| -> Result<C::Value, DifferentiationError> {
                let is_positive_infinity = value.compare(&infinity, ComparisonDirection::Equal)?;
                Ok(C::Value::select(&is_positive_infinity, &value.zero_like()?, &value)?)
            };

            let output_exponent = replace_infinity(aligned_primal)?;
            let left_term = left
                .tangent()
                .as_value()
                .map(|tangent| {
                    let input = replace_infinity(
                        context.primal_to_tangent(left.primal().clone())?.align_tangent(&target, &primal)?,
                    )?;
                    let weight = (input - output_exponent.clone()).exp()?;
                    Ok::<_, DifferentiationError>(weight * tangent.align_tangent(&target, &primal)?)
                })
                .transpose()?;
            let right_term = right
                .tangent()
                .as_value()
                .map(|tangent| {
                    let input = replace_infinity(
                        context.primal_to_tangent(right.primal().clone())?.align_tangent(&target, &primal)?,
                    )?;
                    let weight = (input - output_exponent.clone()).exp()?;
                    Ok::<_, DifferentiationError>(weight * tangent.align_tangent(&target, &primal)?)
                })
                .transpose()?;
            let tangent = left_term
                .into_iter()
                .chain(right_term)
                .reduce(|left_term, right_term| left_term + right_term)
                .map_or_else(|| MaybeZero::Zero(target), MaybeZero::Value);
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
    inputs = @float @real,
    checks = [@no_unreduced, @same_reduced_axes],
    |lhs, rhs| RealFloatingPointArrayElement::log_add_exp(lhs, rhs),
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
        C::Value: OneLike + std::ops::Sub<Output = C::Value> + std::ops::Mul<Output = C::Value>,
    {
        // `d(logistic(x)) = logistic(x) · (1 - logistic(x)) · dx`, reusing the primal output
        // evaluated at the tangent type.
        |(_, input_tangent) -> output| output.clone() * (output.one_like()? - output) * input_tangent
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

/// Returns whether the lowest value of `data_type` acts as an identity of [`LogAddExp`]. That sentinel is what the
/// whole `log(sum(exp(x)))` family writes over the padding of a bounded ragged axis and over an empty accumulation,
/// so an operation that can be asked to write it accepts exactly the element types for which this returns `true`.
///
/// One accumulation folds one copy of the sentinel per padded position it covers, and folding `k` copies of the
/// lowest value `lowest` yields `lowest + ln(k)`, which is still exactly `lowest` only while the drift `ln(k)` stays
/// inside half the gap between `lowest` and its neighbor toward zero. Every format whose lowest value is finite
/// therefore holds its sentinel across a bounded number of copies, `floor(e^(half gap))`, called its *reach* below,
/// while a format with a true `-inf` has unbounded reach:
///
/// | data type       |   lowest |  neighbor | half gap |     reach |
/// | --------------- | -------- | --------- | -------- | --------- |
/// | `f8e8m0fnu`     | `2^-127` |         — |        — |      none |
/// | `f6e2m3fn`      |   `-7.5` |      `-7` |   `0.25` |         1 |
/// | `f4e2m1fn`      |     `-6` |      `-4` |      `1` |         2 |
/// | `f8e4m3b11fnuz` |    `-30` |     `-28` |      `1` |         2 |
/// | `f6e3m2fn`      |    `-28` |     `-24` |      `2` |         7 |
/// | `f8e4m3fnuz`    |   `-240` |    `-224` |      `8` |     2_980 |
/// | `f8e4m3fn`      |   `-448` |    `-416` |     `16` | 8_886_110 |
/// | `f8e5m2fnuz`    | `-57344` |  `-49152` |   `4096` |  `e^4096` |
///
/// A type-level predicate cannot compare a reach against the count an accumulation will actually fold, because that
/// count is the difference between a ragged axis's bound and its per-item extent, which is a runtime quantity. The
/// line this predicate draws is therefore reach alone: a format is rejected when its reach is short enough that any
/// ragged mask worth writing exceeds it, which is the case for the first five rows of the table.
///
/// [`DataType::F8E8M0FNU`] has no sentinel to begin with: it encodes bare positive exponents, so it has neither a
/// zero nor a sign, and its lowest value `2^-127` exponentiates to one rather than to zero. The other four rejected
/// formats do have a lowest value whose exponential underflows to zero in their own format, and three of them —
/// [`DataType::F4E2M1FN`], [`DataType::F8E4M3B11FNUZ`], and [`DataType::F6E3M2FN`] — even keep it a *pairwise*
/// identity, in that `log_add_exp(x, lowest)` returns `x` for every `x` they represent. What they lack is reach: two
/// sentinels are already enough to move [`DataType::F6E2M3FN`]'s `-7.5` to `-7.0` (`-7.5 + ln(2) = -6.807`), three
/// move [`DataType::F4E2M1FN`]'s `-6` to `-4` (`-6 + ln(3) = -4.901`) and [`DataType::F8E4M3B11FNUZ`]'s `-30` to
/// `-28`, and eight move [`DataType::F6E3M2FN`]'s `-28` to `-24`.
///
/// The three accepted finite-lowest formats carry a documented quantitative limit rather than a check, because there
/// is no count for this predicate to check against: an accumulation folding more than 2_980 sentinels in
/// [`DataType::F8E4M3FNUZ`], or more than 8_886_110 in [`DataType::F8E4M3FN`], reads high, and `f8e5m2fnuz`'s reach
/// exceeds every representable count. A consumer that does know the count checks it — the `stablehlo.reduce_window`
/// lowering of a `cumulative_log_sum_exp` seeds one window per output position and rejects a scanned extent past the
/// reach of these same three formats.
pub(crate) fn is_log_add_exp_identity_data_type(data_type: DataType) -> bool {
    data_type.is_floating_point()
        && !matches!(
            data_type,
            DataType::F8E8M0FNU
                | DataType::F6E2M3FN
                | DataType::F4E2M1FN
                | DataType::F8E4M3B11FNUZ
                | DataType::F6E3M2FN
        )
}

/// Returns the diagnostic that `operation_name` reports for an element type rejected by
/// [`is_log_add_exp_identity_data_type`]. The three cases are named apart: an element type that is not real
/// floating-point at all, the one floating-point format that represents neither zero nor negative infinity, and the
/// formats whose lowest value is representable but stops being an identity within a handful of folds.
pub(crate) fn log_add_exp_identity_data_type_error(operation_name: &str, data_type: DataType) -> String {
    match data_type {
        DataType::F8E8M0FNU => format!(
            "`{operation_name}` requires a floating-point format that represents zero and negative infinity but got \
             `{data_type}`"
        ),
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

    use crate::arrays::{Array, ArrayOperation, ArrayType, DataType, Layout, StridedLayout, f8e4m3fn, f8e8m0fnu};
    use crate::contexts::EagerContext;
    use crate::differentiation::differentiate_at;
    use crate::interpretation::InterpretableOperation;
    use crate::macros::{
        check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_transposition, check_operation_type_inference,
    };
    use crate::parameters::Placeholder;
    use crate::programs::{EmptyRegionDriver, ProgramBuilder, Typed};

    use super::*;

    /// Evaluates the pinned primal construction in double precision, for use as the tests' expected value.
    fn expected_log_add_exp(left: f64, right: f64) -> f64 {
        let delta = left - right;
        if delta.is_nan() { left + right } else { left.max(right) + (-delta.abs()).exp().ln_1p() }
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

        assert_eq!(Array::scalar(0.7).unwrap().exp().unwrap(), Array::scalar(0.7f64.exp()).unwrap(),);
    }

    #[test]
    fn test_array_exp_vector() {
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
    fn test_array_exp_layout() {
        // Unary kernels preserve arbitrary physical layouts while traversing elements in logical order.
        let input_type =
            ArrayType::new_static(DataType::F64, [2]).with_layout(Layout::Strided(StridedLayout::new(vec![-16])));
        let input = Array::from_elements(input_type.clone(), &[0.0f64, 1.0]).unwrap();
        let exponential = input.exp().unwrap();
        assert_eq!(exponential.r#type().as_ref(), &input_type);
        assert_eq!(exponential.elements::<f64>(), Ok(vec![1.0, 1.0f64.exp()]));
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
            low_precision.exp().unwrap().to_f64s(),
            vec![1.0, f8e4m3fn::from_f64(1.0f64.exp()).unwrap().to_f64()],
        );
    }

    #[test]
    fn test_exp_for_primitives() {
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
        assert_abs_diff_eq!(tangent.to_f64s()[0], 1.5, epsilon = 1e-9);

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

        assert_eq!(Array::scalar(0.7).unwrap().log().unwrap(), Array::scalar(0.7f64.ln()).unwrap(),);

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
    fn test_log_for_primitives() {
        assert_eq!(Log::log(&1.0f64), Ok(0.0));
    }

    #[test]
    fn test_log1p_type_inference() {
        check_operation_type_inference!(
            @elementwise @unary,
            operation = Log1pOperation,
            cases = [
                {
                    input_data_types = [DataType::F64],
                    output_data_types = [DataType::F64],
                },
                {
                    input_data_types = [DataType::C64],
                    error = "`log1p` does not support input data type `c64`",
                },
                {
                    input_data_types = [DataType::I32],
                    error = "`log1p` does not support input data type `i32`",
                },
            ],
        );
        check_operation_type_inference!(
            @reject @unreduced,
            operation = Log1pOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_log1p_interpretation() {
        assert_eq!(
            Log1pOperation::<ArrayType>::new().interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[Array::scalar(0.0f64).unwrap()],
            ),
            Ok(vec![Array::scalar(0.0f64).unwrap()]),
        );
    }

    #[test]
    fn test_log1p_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = Log1pOperation::new(),
            inputs = [Array::scalar(0.7).unwrap()],
            expected = Array::scalar(0.7f64.ln_1p()).unwrap(),
        );
    }

    #[test]
    fn test_log1p_batching() {
        check_operation_batching!(
            @approx(epsilon = 1e-9),
            operation = Log1pOperation::new(),
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), Array::vector(vec![0.5, -0.5]).unwrap())],
                outputs = [(@mapped(axis = 0), Array::vector(vec![0.5f64.ln_1p(), (-0.5f64).ln_1p()]).unwrap())],
            }],
        );
    }

    #[test]
    fn test_log1p_differentiation() {
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = Log1pOperation::new(),
            cases = [{
                primals = [Array::scalar(0.7).unwrap()],
                tangents = [Array::scalar(3.0).unwrap()],
                primal_outputs = [Array::scalar(0.7f64.ln_1p()).unwrap()],
                tangent_outputs = [Array::scalar(3.0 / 1.7).unwrap()],
                jvp = indoc! {"
                    lambda %0:f64[], %1:f64[] .
                    let %2:f64[] = log1p %0
                        %3:f64[] = one_like %0
                        %4:f64[] = add %3 %0
                        %5:f64[] = div %1 %4
                    in (%2, %5)
                "},
            }],
        );
    }

    #[test]
    fn test_log1p_transposition() {
        check_operation_transposition!(
            @rejected,
            operation = Log1pOperation::<ArrayType>::new(),
            input_types = [ArrayType::scalar(DataType::F64)],
        );
    }

    #[test]
    fn test_array_log1p() {
        // Ordinary values in every supported floating-point width, each evaluated in its own precision.
        assert_eq!(Array::scalar(0.5f32).unwrap().log1p().unwrap(), Array::scalar(0.5f32.ln_1p()).unwrap());
        assert_eq!(Array::scalar(0.5f64).unwrap().log1p().unwrap(), Array::scalar(0.5f64.ln_1p()).unwrap());
        assert_eq!(
            Array::scalar(bf16::from_f32(0.5)).unwrap().log1p().unwrap(),
            Array::scalar(bf16::from_f32(0.5f32.ln_1p())).unwrap(),
        );
        assert_eq!(
            Array::scalar(f16::from_f32(0.5)).unwrap().log1p().unwrap(),
            Array::scalar(f16::from_f32(0.5f32.ln_1p())).unwrap()
        );

        // The fixed point and the boundary values of the real domain.
        assert_eq!(Array::scalar(0.0f64).unwrap().log1p().unwrap(), Array::scalar(0.0f64).unwrap());
        assert_eq!(Array::scalar(-1.0f64).unwrap().log1p().unwrap(), Array::scalar(f64::NEG_INFINITY).unwrap());
        assert!(Array::scalar(-2.0f64).unwrap().log1p().unwrap().to_f64s()[0].is_nan());

        // The accuracy the primitive exists for: near zero, `log1p` keeps full relative precision while the naive
        // composition through `1 + x` has already lost most of it.
        assert_eq!(Array::scalar(1e-10f64).unwrap().log1p().unwrap(), Array::scalar(1e-10f64.ln_1p()).unwrap());
        assert_ne!(1e-10f64.ln_1p(), (1.0f64 + 1e-10).ln());
        assert!((1e-10f64.ln_1p() - 1e-10).abs() < 1e-20);

        assert_eq!(Array::scalar(0.5).unwrap().log1p().unwrap(), Array::scalar(0.5f64.ln_1p()).unwrap());
    }

    #[test]
    fn test_log1p_for_primitives() {
        assert_eq!(Log1p::log1p(&0.0f64), Ok(0.0));
        assert_eq!(Log1p::log1p(&0.0f32), Ok(0.0));
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
                    error = "`log_add_exp` does not support input data type `c64`",
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
            inputs = [Array::scalar(0.5).unwrap(), Array::scalar(-0.25).unwrap()],
            expected = Array::scalar(expected_log_add_exp(0.5, -0.25)).unwrap(),
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
                    Array::vector(vec![expected_log_add_exp(0.5, 2.0), expected_log_add_exp(-1.0, 2.0)]).unwrap(),
                )],
            }],
        );
    }

    #[test]
    fn test_log_add_exp_differentiation() {
        // The tangent is the softmax-weighted combination of the operand tangents.
        let (left, right) = (0.7f64, -0.3f64);
        let (left_tangent, right_tangent) = (0.4f64, -0.2f64);
        let output = expected_log_add_exp(left, right);
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
                        %6:bool[] = compare [direction=Equal] %4 %5
                        %7:f64[] = zero_like %4
                        %8:f64[] = select %6 %7 %4
                        %9:bool[] = compare [direction=Equal] %0 %5
                        %10:f64[] = zero_like %0
                        %11:f64[] = select %9 %10 %0
                        %12:f64[] = sub %11 %8
                        %13:f64[] = exp %12
                        %14:f64[] = mul %13 %2
                        %15:bool[] = compare [direction=Equal] %1 %5
                        %16:f64[] = zero_like %1
                        %17:f64[] = select %15 %16 %1
                        %18:f64[] = sub %17 %8
                        %19:f64[] = exp %18
                        %20:f64[] = mul %19 %3
                        %21:f64[] = add %14 %20
                    in (%4, %21)
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
                .to_f64s()[0]
        };

        // Both weights become `exp(0 - 0) = 1`, so the tangents simply add.
        assert_eq!(jvp((f64::INFINITY, f64::INFINITY), (2.0, 3.0)), 5.0);
        // Negative infinity is not replaced, so both weights are `exp(-∞ - -∞) = exp(NaN)`.
        assert!(jvp((f64::NEG_INFINITY, f64::NEG_INFINITY), (2.0, 3.0)).is_nan());
        // The replaced `+∞` output makes the finite operand's weight `exp(a)` instead of zero.
        assert_eq!(jvp((1.0, f64::INFINITY), (2.0, 3.0)), 1.0f64.exp() * 2.0 + 3.0);
        // A `-∞` operand contributes nothing and the finite operand carries the whole tangent.
        assert_eq!(jvp((1.0, f64::NEG_INFINITY), (2.0, 3.0)), 2.0);
        // A NaN operand propagates through both the primal and the weights.
        assert!(jvp((f64::NAN, 1.0), (2.0, 3.0)).is_nan());
        assert!(jvp((1.0, f64::NAN), (2.0, 3.0)).is_nan());
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
        // Ordinary values in every supported floating-point width, each evaluated in its own precision.
        assert_eq!(
            Array::scalar(1.0f64).unwrap().log_add_exp(&Array::scalar(2.0f64).unwrap()).unwrap(),
            Array::scalar(expected_log_add_exp(1.0, 2.0)).unwrap(),
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

        // The operation is symmetric, and two equal operands add exactly `log(2)`.
        assert_eq!(
            Array::scalar(2.0f64).unwrap().log_add_exp(&Array::scalar(1.0f64).unwrap()).unwrap(),
            Array::scalar(expected_log_add_exp(1.0, 2.0)).unwrap(),
        );
        assert_eq!(
            Array::scalar(0.0f64).unwrap().log_add_exp(&Array::scalar(0.0f64).unwrap()).unwrap(),
            Array::scalar(std::f64::consts::LN_2).unwrap(),
        );

        // The reason the primitive exists: neither exponential is ever formed, so operands far outside the range of
        // `exp` still produce the exact shifted result instead of infinity.
        assert_eq!(
            Array::scalar(1000.0f64).unwrap().log_add_exp(&Array::scalar(1000.0f64).unwrap()).unwrap(),
            Array::scalar(1000.0 + std::f64::consts::LN_2).unwrap(),
        );
        assert!((1000.0f64.exp() + 1000.0f64.exp()).ln().is_infinite());

        // The pinned exceptional values: same-sign infinities saturate, mixed infinities return the larger operand,
        // and NaN propagates from either operand.
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
            Array::scalar(f64::NAN).unwrap().log_add_exp(&Array::scalar(1.0f64).unwrap()).unwrap().to_f64s()[0]
                .is_nan()
        );
        assert!(
            Array::scalar(1.0f64).unwrap().log_add_exp(&Array::scalar(f64::NAN).unwrap()).unwrap().to_f64s()[0]
                .is_nan()
        );

        assert_eq!(
            Array::scalar(1.0).unwrap().log_add_exp(&Array::scalar(2.0).unwrap()).unwrap(),
            Array::scalar(expected_log_add_exp(1.0, 2.0)).unwrap()
        );
    }

    #[test]
    fn test_log_add_exp_for_primitives() {
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
                    Array::vector(vec![1.0 / (1.0 + (-0.5f64).exp()), 1.0 / (1.0 + 1.0f64.exp())]).unwrap()
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
            Array::scalar(expected).unwrap(),
            differentiate_at(Array::scalar(input).unwrap())
                .holomorphic()
                .gradient(|input| { input.logistic().unwrap() })
                .unwrap(),
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

        assert_eq!(
            Array::scalar(0.7).unwrap().logistic().unwrap(),
            Array::scalar(1.0 / (1.0 + (-0.7f64).exp())).unwrap(),
        );
    }

    #[test]
    fn test_logistic_for_primitives() {
        assert_eq!(Logistic::logistic(&0.0f64), Ok(0.5));
    }

    #[test]
    fn test_is_log_add_exp_identity_data_type() {
        assert!(is_log_add_exp_identity_data_type(DataType::F32));
        assert!(is_log_add_exp_identity_data_type(DataType::F8E4M3FN));
        assert!(!is_log_add_exp_identity_data_type(DataType::F8E8M0FNU));
        assert!(!is_log_add_exp_identity_data_type(DataType::F4E2M1FN));
        assert!(!is_log_add_exp_identity_data_type(DataType::I32));
        assert!(!is_log_add_exp_identity_data_type(DataType::C64));
    }

    #[test]
    fn test_log_add_exp_identity_data_type_error() {
        assert_eq!(
            log_add_exp_identity_data_type_error("reduce_log_sum_exp", DataType::F8E8M0FNU),
            "`reduce_log_sum_exp` requires a floating-point format that represents zero and negative infinity but got \
             `f8e8m0fnu`",
        );
        assert_eq!(
            log_add_exp_identity_data_type_error("reduce_log_sum_exp", DataType::F4E2M1FN),
            "`reduce_log_sum_exp` requires a floating-point format whose lowest value is a `log_add_exp` identity but got \
             `f4e2m1fn`",
        );
        assert_eq!(
            log_add_exp_identity_data_type_error("reduce_log_sum_exp", DataType::I32),
            "`reduce_log_sum_exp` requires real floating-point inputs but got `i32`",
        );
    }
}
