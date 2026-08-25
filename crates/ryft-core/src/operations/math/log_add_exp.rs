use std::ops::{Add as StandardAdd, Mul as StandardMul, Sub as StandardSub};

use crate::arrays::DataType;
use crate::differentiation::{
    DifferentiableType, DifferentiationDual, DifferentiationError, ElementwiseDerivativeAlignment,
};
use crate::macros::{
    check_count, define_elementwise_capability, define_elementwise_operation, impl_differentiable_operation,
};
use crate::operations::compare::{Compare, ComparisonDirection};
use crate::operations::constants::fill::Fill;
use crate::operations::constants::zero_like::ZeroLike;
use crate::operations::control_flow::select::Select;
use crate::operations::math::exp::Exp;
use crate::programs::{MaybeZero, ProgramError, Type, Typed, Value};

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`LogAddExpOperation`].
pub const LOG_ADD_EXP_OPERATION_NAME: &str = "log_add_exp";

define_elementwise_operation!(
    @binary
    /// [`Operation`] that computes the elementwise `log(exp(a) + exp(b))` of its operands without forming either
    /// exponential, promoting their element types and broadcasting their shapes.
    ///
    /// The semantics are pinned to JAX's `logaddexp` (`jax/_src/lax/other.py`), which evaluates
    ///
    /// ```text
    /// log_add_exp(a, b) = select(isnan(a - b), a + b, max(a, b) + log1p(exp(-|a - b|)))
    /// ```
    ///
    /// Factoring the larger operand out of the sum is what makes the primitive usable across the whole real range:
    /// `exp(-|a - b|)` never overflows, so `log_add_exp(1000, 1000)` is exactly `1000 + log(2)` where the naive
    /// composition returns infinity.
    ///
    /// The `isnan(a - b)` arm is the guard for the cases in which the difference itself is undefined, and it fixes
    /// the following results:
    ///
    ///   - `(+∞, +∞) ↦ +∞` and `(-∞, -∞) ↦ -∞`, through the `a + b` arm;
    ///   - any NaN operand propagates NaN, also through the `a + b` arm;
    ///   - mixed infinities return the larger operand, through the ordinary arm: `-|a - b|` is `-∞`, so
    ///     `log1p(exp(-∞)) = log1p(0) = 0` and the result is `max(a, b)`.
    ///
    /// Only real floating-point operands are supported, and array operands that still carry partial sums are
    /// rejected, with their reduced-axis markers required to agree.
    LogAddExpOperation, LOG_ADD_EXP_OPERATION_NAME,
    LogAddExp, log_add_exp,
    check_data_types = [@float @real],
    check_array_types = [@no_unreduced, @same_reduced_axes],
);

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
             {data_type}"
        ),
        _ if data_type.is_floating_point() => format!(
            "`{operation_name}` requires a floating-point format whose lowest value is a `log_add_exp` identity but \
             got {data_type}"
        ),
        _ => format!("`{operation_name}` requires real floating-point inputs but got {data_type}"),
    }
}

impl_differentiable_operation! {
    <T> LogAddExpOperation<T>,
    jvp<C>
    where
        T: Type,
        C::Type: DifferentiableType,
        C::Value: LogAddExp
            + Compare<C::Value>
            + Select
            + ZeroLike
            + Exp
            + StandardAdd<Output = C::Value>
            + StandardSub<Output = C::Value>
            + StandardMul<Output = C::Value>
            + ElementwiseDerivativeAlignment<C::Type>,
        <C::Value as Value>::DispatchDomain: Fill<f64, C::Value>,
    {
        |_operation, _context, _driver, inputs| {
            // The partial derivative with respect to each operand is the softmax weight `exp(x - log_add_exp(a, b))`,
            // so the tangent is `w_a · da + w_b · db`. Both weights are formed against the shared primal output, which
            // is therefore computed once. Following JAX's `_logaddexp_jvp`, every operand and the primal output pass
            // through a `replace_infinity` guard that rewrites *positive* infinity to zero before the subtraction, so
            // that a `+∞` operand yields the finite weights `exp(a)` and `1` rather than `exp(∞ - ∞) = NaN`. Negative
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
                        "`{}` output type {} has no tangent space",
                        LOG_ADD_EXP_OPERATION_NAME,
                        primal.r#type(),
                    ),
                }
                .into());
            }
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
                    let operand = replace_infinity(left.primal().align_tangent(&target, &primal)?)?;
                    let weight = (operand - output_exponent.clone()).exp()?;
                    Ok::<_, DifferentiationError>(weight * tangent.align_tangent(&target, &primal)?)
                })
                .transpose()?;
            let right_term = right
                .tangent()
                .as_value()
                .map(|tangent| {
                    let operand = replace_infinity(right.primal().align_tangent(&target, &primal)?)?;
                    let weight = (operand - output_exponent.clone()).exp()?;
                    Ok::<_, DifferentiationError>(weight * tangent.align_tangent(&target, &primal)?)
                })
                .transpose()?;
            let tangent = left_term
                .into_iter()
                .chain(right_term)
                .reduce(|left_term, right_term| left_term + right_term)
                .map_or_else(|| MaybeZero::Zero(target), MaybeZero::Value);
            Ok(vec![DifferentiationDual::new(primal, tangent)?])
        }
    },
    transpose = @nonlinear,
}

define_elementwise_capability!(
    @binary
    /// Value-level elementwise `log(exp(self) + exp(other))` capability. [`LogAddExp`] fills the same role for
    /// [`LogAddExpOperation`] that [`Sin`](crate::Sin) fills for [`SinOperation`](crate::SinOperation).
    LogAddExp,
    /// Computes the elementwise `log(exp(self) + exp(other))` without forming either exponential, promoting both
    /// operands to a common real floating-point element type and returning a [`ProgramError`] if something goes
    /// wrong.
    log_add_exp(other),
    LogAddExpOperation,
);

/// Implements [`LogAddExp`] for one host primitive type.
macro_rules! impl_capability_for_primitive {
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

impl_capability_for_primitive!(f32);
impl_capability_for_primitive!(f64);

#[cfg(test)]
mod tests {
    use half::{bf16, f16};
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{Array, ArrayType, DataType};
    use crate::differentiation::differentiate_at;
    use crate::macros::{
        check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_transposition, check_operation_type_inference,
    };

    use super::*;

    // Evaluates the pinned primal construction in double precision, for use as the tests' expected value.
    fn expected(left: f64, right: f64) -> f64 {
        let delta = left - right;
        if delta.is_nan() { left + right } else { left.max(right) + (-delta.abs()).exp().ln_1p() }
    }

    #[test]
    fn test_log_add_exp() {
        // Ordinary values in every supported floating-point width, each evaluated in its own precision.
        assert_eq!(
            Array::scalar(1.0f64).log_add_exp(&Array::scalar(2.0f64)).unwrap(),
            Array::scalar(expected(1.0, 2.0)),
        );
        assert_eq!(
            Array::scalar(1.0f32).log_add_exp(&Array::scalar(2.0f32)).unwrap(),
            Array::scalar(2.0f32 + (-1.0f32).exp().ln_1p()),
        );
        assert_eq!(
            Array::scalar(bf16::from_f32(1.0)).log_add_exp(&Array::scalar(bf16::from_f32(2.0))).unwrap(),
            Array::scalar(bf16::from_f32(2.0f32 + (-1.0f32).exp().ln_1p())),
        );
        assert_eq!(
            Array::scalar(f16::from_f32(1.0)).log_add_exp(&Array::scalar(f16::from_f32(2.0))).unwrap(),
            Array::scalar(f16::from_f32(2.0f32 + (-1.0f32).exp().ln_1p())),
        );

        // The operation is symmetric, and two equal operands add exactly `log(2)`.
        assert_eq!(
            Array::scalar(2.0f64).log_add_exp(&Array::scalar(1.0f64)).unwrap(),
            Array::scalar(expected(1.0, 2.0)),
        );
        assert_eq!(
            Array::scalar(0.0f64).log_add_exp(&Array::scalar(0.0f64)).unwrap(),
            Array::scalar(std::f64::consts::LN_2),
        );

        // The reason the primitive exists: neither exponential is ever formed, so operands far outside the range of
        // `exp` still produce the exact shifted result instead of infinity.
        assert_eq!(
            Array::scalar(1000.0f64).log_add_exp(&Array::scalar(1000.0f64)).unwrap(),
            Array::scalar(1000.0 + std::f64::consts::LN_2),
        );
        assert!((1000.0f64.exp() + 1000.0f64.exp()).ln().is_infinite());

        // The pinned exceptional values: same-sign infinities saturate, mixed infinities return the larger operand,
        // and NaN propagates from either operand.
        assert_eq!(
            Array::scalar(f64::INFINITY).log_add_exp(&Array::scalar(f64::INFINITY)).unwrap(),
            Array::scalar(f64::INFINITY),
        );
        assert_eq!(
            Array::scalar(f64::NEG_INFINITY).log_add_exp(&Array::scalar(f64::NEG_INFINITY)).unwrap(),
            Array::scalar(f64::NEG_INFINITY),
        );
        assert_eq!(
            Array::scalar(1.0f64).log_add_exp(&Array::scalar(f64::INFINITY)).unwrap(),
            Array::scalar(f64::INFINITY),
        );
        assert_eq!(
            Array::scalar(1.0f64).log_add_exp(&Array::scalar(f64::NEG_INFINITY)).unwrap(),
            Array::scalar(1.0f64),
        );
        assert!(Array::scalar(f64::NAN).log_add_exp(&Array::scalar(1.0f64)).unwrap().to_f64s()[0].is_nan());
        assert!(Array::scalar(1.0f64).log_add_exp(&Array::scalar(f64::NAN)).unwrap().to_f64s()[0].is_nan());

        assert_eq!(Array::scalar(1.0).log_add_exp(&Array::scalar(2.0)).unwrap(), Array::scalar(expected(1.0, 2.0)));
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
                    error = "`log_add_exp` does not support input data type c64",
                },
                {
                    input_data_types = [DataType::I32, DataType::F32],
                    error = "`log_add_exp` does not support input data type i32",
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
    fn test_log_add_exp_batching() {
        check_operation_batching!(
            @approx(epsilon = 1e-9),
            operation = LogAddExpOperation::new(),
            axis_size = 2,
            cases = [{
                inputs = [
                    (@mapped(axis = 0), Array::vector(vec![0.5, -1.0])),
                    (@replicated, Array::scalar(2.0)),
                ],
                outputs = [(@mapped(axis = 0), Array::vector(vec![expected(0.5, 2.0), expected(-1.0, 2.0)]))],
            }],
        );
    }

    #[test]
    fn test_log_add_exp_differentiation() {
        // The tangent is the softmax-weighted combination of the operand tangents.
        let (left, right) = (0.7f64, -0.3f64);
        let (left_tangent, right_tangent) = (0.4f64, -0.2f64);
        let output = expected(left, right);
        let tangent = (left - output).exp() * left_tangent + (right - output).exp() * right_tangent;
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = LogAddExpOperation::new(),
            cases = [{
                primals = [Array::scalar(left), Array::scalar(right)],
                tangents = [Array::scalar(left_tangent), Array::scalar(right_tangent)],
                primal_outputs = [Array::scalar(output)],
                tangent_outputs = [Array::scalar(tangent)],
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
    fn test_log_add_exp_exceptional_tangents() {
        // These five results are conventions of the differentiation rule rather than mathematical extensions of the
        // primal, so they are pinned exactly. They follow from replacing only *positive* infinity with zero before
        // the weight subtraction, exactly as JAX's `_logaddexp_jvp` does.
        let jvp = |primals: (f64, f64), tangents: (f64, f64)| {
            differentiate_at((Array::scalar(primals.0), Array::scalar(primals.1)))
                .jvp((Array::scalar(tangents.0), Array::scalar(tangents.1)), |(left, right)| left.log_add_exp(&right))
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
    fn test_log_add_exp_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = LogAddExpOperation::new(),
            inputs = [Array::scalar(0.5), Array::scalar(-0.25)],
            expected = Array::scalar(expected(0.5, -0.25)),
        );
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
    fn test_log_add_exp_for_primitives() {
        assert_eq!(LogAddExp::log_add_exp(&0.0_f64, &0.0), Ok(std::f64::consts::LN_2));
        assert_eq!(LogAddExp::log_add_exp(&0.0_f32, &0.0), Ok(std::f32::consts::LN_2));
    }
}
