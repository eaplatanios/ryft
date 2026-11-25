//! Learning rate schedules, which map the number of updates that an optimizer has applied so far to a hyperparameter
//! value (e.g., a learning rate), in the style of the
//! [Optax schedules](https://optax.readthedocs.io/en/latest/api/optimizer_schedules.html).
//!
//! A [`Schedule`] is evaluated with array operations on a rank-0 `i32` step count array, rather than on a host
//! integer, so that the step count can live in the optimizer state and the schedule can be evaluated inside compiled
//! training steps, where the step count is a traced value. Schedules convert the step count to `f32` and return
//! rank-0 `f32` values. [`ScaleBySchedule`](crate::ScaleBySchedule) evaluates a schedule at
//! its own step count and scales updates by the result, and [`AdamW`](crate::AdamW) takes its learning rate as a
//! schedule. A constant value is a schedule as well, because `f64` implements [`Schedule`].
//!
//! The schedules mirror those of Optax:
//!
//!   - **Constant:** `f64` itself, which is the analogue of `optax.constant_schedule`.
//!   - **Interpolations:** [`LinearSchedule`] and [`PolynomialSchedule`] interpolate between two values,
//!     [`CosineDecaySchedule`] decays a value along half a cosine period, and [`ExponentialDecaySchedule`] decays it
//!     exponentially, continuously or in a staircase.
//!   - **Piecewise schedules:** [`PiecewiseConstantSchedule`] scales a value at step boundaries, and
//!     [`PiecewiseInterpolateSchedule`] interpolates linearly or along cosines between the scaled values, which is
//!     what the one-cycle schedules [`LinearOneCycleSchedule`] and [`CosineOneCycleSchedule`] of
//!     [Smith and Topin (2017)](https://arxiv.org/abs/1708.07120) are built from.
//!   - **Warmups and restarts:** [`WarmupCosineDecaySchedule`] and [`WarmupExponentialDecaySchedule`] combine a linear
//!     warmup with a decay, which is a common choice for training transformers, and [`SgdrSchedule`] restarts warmup
//!     cosine decays, as in the SGDR schedule of [Loshchilov and Hutter (2017)](https://arxiv.org/abs/1608.03983).
//!     A warmup to a constant value (i.e., `optax.warmup_constant_schedule`) is a [`LinearSchedule`] from the initial
//!     value to the peak value over the warmup steps, because that is how Optax defines it.
//!   - **Joining:** [`JoinedSchedule`] switches from one schedule to another at a step boundary, restarting the step
//!     count of the second schedule there. Joined schedules nest, so they can hold schedules of different types.
//!
//! Their constructors validate their configurations, so a schedule that exists can always be evaluated.

use std::collections::BTreeMap;
use std::f64::consts::PI;

use ryft_core::{Add, ComparisonDirection, DataType, Div, Mul, Sub};

use crate::{OptimizerError, OptimizerValue, convert_to, validate_finite, validate_positive};

// TODO(eaplatanios): Review this module.

/// Function from a step count to a hyperparameter value, which is the analogue of an Optax
/// [`Schedule`](https://optax.readthedocs.io/en/latest/api/optimizer_schedules.html#optax.Schedule). The step count is
/// the number of updates that have been applied so far, starting at `0`, and it is represented as a rank-0 `i32` array
/// so that schedules can be evaluated inside compiled training steps. Implementations convert the step count to `f32`
/// (which is exact up to `2^24`), compute their values with array operations on it, and never read it back to the host.
///
/// `f64` implements this trait as the constant schedule, which is the analogue of [`optax.constant_schedule`](
/// https://optax.readthedocs.io/en/latest/api/optimizer_schedules.html#optax.constant_schedule) and whose value at
/// every step count `t` is the constant `c` itself, as in
///
/// ```text
/// η(t) = c
/// ```
pub trait Schedule {
    /// Returns the value of this schedule at `count`, as a rank-0 `f32` array.
    ///
    /// # Parameters
    ///
    ///   - `count`: Rank-0 integer array (i.e., the `i32` step count of an optimizer state) that holds the number of
    ///     updates applied so far. A rank-0 `f32` array is accepted as well.
    ///
    /// # Errors
    ///
    /// Returns an [`OptimizerError`] when one of the array operations that compute the value fails.
    fn value<P: OptimizerValue>(&self, count: &P) -> Result<P, OptimizerError>;
}

impl Schedule for f64 {
    fn value<P: OptimizerValue>(&self, count: &P) -> Result<P, OptimizerError> {
        Ok(convert_to(count, DataType::F32)?.fill_like(*self)?)
    }
}

/// [`Schedule`] that interpolates linearly from an initial value to an end value over a number of steps after an
/// optional delay, and that is constant outside of that interval, which is the analogue of [`optax.linear_schedule`](
/// https://optax.readthedocs.io/en/latest/api/optimizer_schedules.html#optax.linear_schedule). It is the usual linear
/// warmup of the learning rate (e.g., the gradual warmup of [Goyal et al. (2017)](https://arxiv.org/abs/1706.02677)),
/// and its value is
///
/// ```text
/// τ    = clip(t − b, 0, T)
/// η(t) = (η₀ − ηₑ) · (1 − τ / T) + ηₑ
/// ```
///
/// Here `t` is the step count (i.e., the number of updates applied so far, starting at `0`), which optimizers pass as
/// the count before the increment of their current update. The initial value is `η₀`, the end value is `ηₑ`, `b` is the
/// transition begin (i.e., the number of steps for which `η₀` is held), `T` is the number of transition steps, and `τ`
/// is the number of steps into the transition, so the value is `η₀` up to step `b` and `ηₑ` from step `b + T` on. The
/// default is `b = 0`, which [`Self::with_transition_begin`] changes. A unit power of [`PolynomialSchedule`] gives the
/// same values.
///
/// This schedule is a closed form expression of the step count rather than a chain of other schedules.
///
/// # References
///
///   - Goyal et al. (2017). [Accurate, Large Minibatch SGD: Training ImageNet in 1
///     Hour](https://arxiv.org/abs/1706.02677).
#[derive(Clone, Debug, PartialEq)]
pub struct LinearSchedule {
    /// Value up to the beginning of the transition.
    initial_value: f64,

    /// Value from the end of the transition on.
    end_value: f64,

    /// Number of steps over which the value changes, which is always positive.
    transition_steps: usize,

    /// Number of steps for which the initial value is held before the transition begins.
    transition_begin: usize,
}

impl LinearSchedule {
    /// Creates a new [`LinearSchedule`] whose transition begins at step `0`. Use [`Self::with_transition_begin`] to
    /// hold the initial value for a number of steps first.
    ///
    /// # Parameters
    ///
    ///   - `initial_value`: Value at the beginning of the transition.
    ///   - `end_value`: Value at the end of the transition.
    ///   - `transition_steps`: Number of steps over which the value changes.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when a value is not finite or when `transition_steps` is `0`.
    pub fn new(initial_value: f64, end_value: f64, transition_steps: usize) -> Result<Self, OptimizerError> {
        validate_finite("initial_value", initial_value)?;
        validate_finite("end_value", end_value)?;
        validate_positive_steps("transition_steps", transition_steps)?;
        Ok(Self { initial_value, end_value, transition_steps, transition_begin: 0 })
    }

    /// Returns this schedule with a transition that begins after `transition_begin` steps.
    #[inline]
    pub fn with_transition_begin(mut self, transition_begin: usize) -> Self {
        self.transition_begin = transition_begin;
        self
    }

    /// Returns the value of this schedule up to the beginning of its transition.
    #[inline]
    pub fn initial_value(&self) -> f64 {
        self.initial_value
    }

    /// Returns the value of this schedule from the end of its transition on.
    #[inline]
    pub fn end_value(&self) -> f64 {
        self.end_value
    }

    /// Returns the number of steps over which the value of this schedule changes.
    #[inline]
    pub fn transition_steps(&self) -> usize {
        self.transition_steps
    }

    /// Returns the number of steps for which this schedule holds its initial value before its transition begins.
    #[inline]
    pub fn transition_begin(&self) -> usize {
        self.transition_begin
    }
}

impl Schedule for LinearSchedule {
    fn value<P: OptimizerValue>(&self, count: &P) -> Result<P, OptimizerError> {
        // This follows the operation order of `optax.polynomial_schedule` with a unit power.
        let step = &convert_to(count, DataType::F32)?;
        let transition_steps = step.fill_like(self.transition_steps as f64)?;
        let count = Sub::sub(step, &step.fill_like(self.transition_begin as f64)?)?
            .max(&step.fill_like(0f64)?)?
            .min(&transition_steps)?;
        let fraction = Sub::sub(&step.fill_like(1f64)?, &Div::div(&count, &transition_steps)?)?;
        let scaled = Mul::mul(&step.fill_like(self.initial_value - self.end_value)?, &fraction)?;
        Ok(Add::add(&scaled, &step.fill_like(self.end_value)?)?)
    }
}

/// [`Schedule`] that decays an initial value along half a cosine period, which is the schedule of
/// [Loshchilov and Hutter (2017)](https://arxiv.org/abs/1608.03983) without restarts and the analogue of
/// [`optax.cosine_decay_schedule`](
/// https://optax.readthedocs.io/en/latest/api/optimizer_schedules.html#optax.cosine_decay_schedule). Its value is
///
/// ```text
/// τ    = min(t, T)
/// η(t) = η₀ · ((1 − α) · (½ · (1 + cos(π · τ / T)))ᵖ + α)
/// ```
///
/// Here `t` is the step count (i.e., the number of updates applied so far, starting at `0`), which optimizers pass as
/// the count before the increment of their current update. The initial value is `η₀`, `T` is the number of decay steps,
/// `α` is the fraction of `η₀` that remains at the end of the decay, and `p` is the exponent of the cosine factor, so
/// the value decays from `η₀` to `α · η₀` at step `T` and stays there afterwards. The defaults are `α = 0` and `p = 1`,
/// which [`Self::with_alpha`] and [`Self::with_exponent`] change. With `p = 1`, this is the annealing of Loshchilov and
/// Hutter (2017) from `η₀` to `α · η₀`.
///
/// This schedule is a closed form expression of the step count rather than a chain of other schedules.
///
/// # References
///
///   - Loshchilov and Hutter (2017). [SGDR: Stochastic Gradient Descent with Warm
///     Restarts](https://arxiv.org/abs/1608.03983).
#[derive(Clone, Debug, PartialEq)]
pub struct CosineDecaySchedule {
    /// Value at step `0`.
    initial_value: f64,

    /// Number of steps over which the value decays, which is always positive.
    decay_steps: usize,

    /// Fraction of the initial value that remains at the end of the decay.
    alpha: f64,

    /// Exponent to which the cosine factor is raised.
    exponent: f64,
}

impl CosineDecaySchedule {
    /// Creates a new [`CosineDecaySchedule`] that decays `initial_value` to zero over `decay_steps` steps. Use
    /// [`Self::with_alpha`] to decay to a fraction of the initial value instead.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `initial_value` is not finite or when `decay_steps` is
    /// `0`.
    pub fn new(initial_value: f64, decay_steps: usize) -> Result<Self, OptimizerError> {
        validate_finite("initial_value", initial_value)?;
        validate_positive_steps("decay_steps", decay_steps)?;
        Ok(Self { initial_value, decay_steps, alpha: 0.0, exponent: 1.0 })
    }

    /// Returns this schedule with `alpha` as the fraction of the initial value that remains at the end of the decay.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `alpha` is not finite.
    pub fn with_alpha(mut self, alpha: f64) -> Result<Self, OptimizerError> {
        validate_finite("alpha", alpha)?;
        self.alpha = alpha;
        Ok(self)
    }

    /// Returns this schedule with `exponent` as the exponent to which its cosine factor is raised.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `exponent` is not finite.
    pub fn with_exponent(mut self, exponent: f64) -> Result<Self, OptimizerError> {
        validate_finite("exponent", exponent)?;
        self.exponent = exponent;
        Ok(self)
    }

    /// Returns the value of this schedule at step `0`.
    #[inline]
    pub fn initial_value(&self) -> f64 {
        self.initial_value
    }

    /// Returns the number of steps over which the value of this schedule decays.
    #[inline]
    pub fn decay_steps(&self) -> usize {
        self.decay_steps
    }

    /// Returns the fraction of the initial value that remains at the end of the decay of this schedule.
    #[inline]
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Returns the exponent to which this schedule raises its cosine factor.
    #[inline]
    pub fn exponent(&self) -> f64 {
        self.exponent
    }
}

impl Schedule for CosineDecaySchedule {
    fn value<P: OptimizerValue>(&self, count: &P) -> Result<P, OptimizerError> {
        // This follows the operation order of `optax.cosine_decay_schedule`. A unit exponent leaves the cosine factor
        // unchanged, so no power is staged for it.
        let step = &convert_to(count, DataType::F32)?;
        let decay_steps = step.fill_like(self.decay_steps as f64)?;
        let count = step.min(&decay_steps)?;
        let angle = Div::div(&Mul::mul(&step.fill_like(PI)?, &count)?, &decay_steps)?;
        let cosine = Mul::mul(&step.fill_like(0.5f64)?, &Add::add(&step.fill_like(1f64)?, &angle.cos()?)?)?;
        let cosine = if self.exponent == 1.0 { cosine } else { cosine.pow(&step.fill_like(self.exponent)?)? };
        let decayed = Add::add(&Mul::mul(&step.fill_like(1.0 - self.alpha)?, &cosine)?, &step.fill_like(self.alpha)?)?;
        Ok(Mul::mul(&step.fill_like(self.initial_value)?, &decayed)?)
    }
}

/// [`Schedule`] that warms up linearly from an initial value to a peak value and then decays to an end value along half
/// a cosine period, which combines the linear warmup of [Goyal et al. (2017)](https://arxiv.org/abs/1706.02677) with
/// the cosine annealing of [Loshchilov and Hutter (2017)](https://arxiv.org/abs/1608.03983), and which is the analogue
/// of [`optax.warmup_cosine_decay_schedule`](
/// https://optax.readthedocs.io/en/latest/api/optimizer_schedules.html#optax.warmup_cosine_decay_schedule). Its value
/// is
///
/// ```text
/// η(t) = η₀ + (ηₚ − η₀) · t / W                                         if t < W
/// η(t) = ηₚ · ((1 − α) · (½ · (1 + cos(π · min(t − W, D) / D)))ᵖ + α)   if t ≥ W
/// ```
///
/// Here `t` is the step count (i.e., the number of updates applied so far, starting at `0`), which optimizers pass as
/// the count before the increment of their current update. The initial value is `η₀`, the peak value is `ηₚ`, `W` is
/// the number of warmup steps, `D` is the number of decay steps after the warmup (i.e., `decay_steps − W`, because
/// `decay_steps` counts the warmup steps), `α = ηₑ / ηₚ` is the fraction of the peak value that the end value `ηₑ`
/// represents (or `0` when `ηₚ = 0`), and `p` is the exponent of the cosine factor, so the value reaches `ηₑ` at step
/// `decay_steps` and stays there afterwards. The defaults are `ηₑ = 0` and `p = 1`, which [`Self::with_end_value`] and
/// [`Self::with_exponent`] change.
///
/// It joins a [`LinearSchedule`] for the warmup and a [`CosineDecaySchedule`] for the decay at step `W`, as a
/// [`JoinedSchedule`] does.
///
/// # References
///
///   - Goyal et al. (2017). [Accurate, Large Minibatch SGD: Training ImageNet in 1
///     Hour](https://arxiv.org/abs/1706.02677).
///   - Loshchilov and Hutter (2017). [SGDR: Stochastic Gradient Descent with Warm
///     Restarts](https://arxiv.org/abs/1608.03983).
#[derive(Clone, Debug, PartialEq)]
pub struct WarmupCosineDecaySchedule {
    /// Value at step `0`.
    initial_value: f64,

    /// Value at the end of the warmup.
    peak_value: f64,

    /// Number of warmup steps.
    warmup_steps: usize,

    /// Total number of steps of the warmup and the decay, which is always greater than the number of warmup steps.
    decay_steps: usize,

    /// Value at the end of the decay.
    end_value: f64,

    /// Exponent to which the cosine factor of the decay is raised.
    exponent: f64,
}

impl WarmupCosineDecaySchedule {
    /// Creates a new [`WarmupCosineDecaySchedule`] whose decay ends at zero. Use [`Self::with_end_value`] to decay
    /// to another value instead.
    ///
    /// # Parameters
    ///
    ///   - `initial_value`: Value at step `0`.
    ///   - `peak_value`: Value at the end of the warmup.
    ///   - `warmup_steps`: Number of steps of the linear warmup, which may be `0` to start at the peak value.
    ///   - `decay_steps`: Total number of steps of the warmup and the decay.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when a value is not finite or when `decay_steps` is not
    /// greater than `warmup_steps`.
    pub fn new(
        initial_value: f64,
        peak_value: f64,
        warmup_steps: usize,
        decay_steps: usize,
    ) -> Result<Self, OptimizerError> {
        validate_finite("initial_value", initial_value)?;
        validate_finite("peak_value", peak_value)?;
        if decay_steps <= warmup_steps {
            return Err(OptimizerError::InvalidConfiguration {
                message: format!(
                    "`decay_steps` must be greater than `warmup_steps` but got `{decay_steps}` and `{warmup_steps}`",
                ),
            });
        }
        Ok(Self { initial_value, peak_value, warmup_steps, decay_steps, end_value: 0.0, exponent: 1.0 })
    }

    /// Returns this schedule with `end_value` as its value at the end of the decay.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `end_value` is not finite.
    pub fn with_end_value(mut self, end_value: f64) -> Result<Self, OptimizerError> {
        validate_finite("end_value", end_value)?;
        self.end_value = end_value;
        Ok(self)
    }

    /// Returns this schedule with `exponent` as the exponent to which the cosine factor of its decay is raised (refer
    /// to [`CosineDecaySchedule::with_exponent`]).
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `exponent` is not finite.
    pub fn with_exponent(mut self, exponent: f64) -> Result<Self, OptimizerError> {
        validate_finite("exponent", exponent)?;
        self.exponent = exponent;
        Ok(self)
    }

    /// Returns the value of this schedule at step `0`.
    #[inline]
    pub fn initial_value(&self) -> f64 {
        self.initial_value
    }

    /// Returns the value of this schedule at the end of its warmup.
    #[inline]
    pub fn peak_value(&self) -> f64 {
        self.peak_value
    }

    /// Returns the number of warmup steps of this schedule.
    #[inline]
    pub fn warmup_steps(&self) -> usize {
        self.warmup_steps
    }

    /// Returns the total number of steps of the warmup and the decay of this schedule.
    #[inline]
    pub fn decay_steps(&self) -> usize {
        self.decay_steps
    }

    /// Returns the value of this schedule at the end of its decay.
    #[inline]
    pub fn end_value(&self) -> f64 {
        self.end_value
    }

    /// Returns the exponent to which this schedule raises the cosine factor of its decay.
    #[inline]
    pub fn exponent(&self) -> f64 {
        self.exponent
    }
}

impl Schedule for WarmupCosineDecaySchedule {
    fn value<P: OptimizerValue>(&self, count: &P) -> Result<P, OptimizerError> {
        // This joins the two schedules at the warmup boundary as `optax.join_schedules` does. The constructor
        // validated the step counts and values, so both schedules are built directly. The step count is converted
        // once, so the conversions of the two schedules are no-ops.
        let step = &convert_to(count, DataType::F32)?;
        let alpha = if self.peak_value == 0.0 { 0.0 } else { self.end_value / self.peak_value };
        let decay = CosineDecaySchedule {
            initial_value: self.peak_value,
            decay_steps: self.decay_steps - self.warmup_steps,
            alpha,
            exponent: self.exponent,
        };
        if self.warmup_steps == 0 {
            return decay.value(step);
        }
        let warmup = LinearSchedule {
            initial_value: self.initial_value,
            end_value: self.peak_value,
            transition_steps: self.warmup_steps,
            transition_begin: 0,
        };
        let boundary = step.fill_like(self.warmup_steps as f64)?;
        let in_warmup = step.compare(&boundary, ComparisonDirection::LessThan)?;
        Ok(in_warmup.select(&warmup.value(step)?, &decay.value(&Sub::sub(step, &boundary)?)?)?)
    }
}

/// [`Schedule`] that interpolates from an initial value to an end value along a polynomial over a number of steps after
/// an optional delay, and that is constant outside of that interval, which is the analogue of
/// [`optax.polynomial_schedule`](
/// https://optax.readthedocs.io/en/latest/api/optimizer_schedules.html#optax.polynomial_schedule). Its value is
///
/// ```text
/// τ    = clip(t − b, 0, T)
/// η(t) = (η₀ − ηₑ) · (1 − τ / T)ᵖ + ηₑ
/// ```
///
/// Here `t` is the step count (i.e., the number of updates applied so far, starting at `0`), which optimizers pass as
/// the count before the increment of their current update. The initial value is `η₀`, the end value is `ηₑ`, `p` is the
/// power of the polynomial, `b` is the transition begin (i.e., the number of steps for which `η₀` is held), `T` is the
/// number of transition steps, and `τ` is the number of steps into the transition, so the value is `η₀` up to step `b`
/// and `ηₑ` from step `b + T` on. The default is `b = 0`, which [`Self::with_transition_begin`] changes. A unit power
/// gives the values of [`LinearSchedule`].
///
/// This schedule is a closed form expression of the step count rather than a chain of other schedules. It has no
/// originating paper.
#[derive(Clone, Debug, PartialEq)]
pub struct PolynomialSchedule {
    /// Value up to the beginning of the transition.
    initial_value: f64,

    /// Value from the end of the transition on.
    end_value: f64,

    /// Power of the polynomial along which the value changes.
    power: f64,

    /// Number of steps over which the value changes, which is always positive.
    transition_steps: usize,

    /// Number of steps for which the initial value is held before the transition begins.
    transition_begin: usize,
}

impl PolynomialSchedule {
    /// Creates a new [`PolynomialSchedule`] whose transition begins at step `0`. Use [`Self::with_transition_begin`]
    /// to hold the initial value for a number of steps first.
    ///
    /// # Parameters
    ///
    ///   - `initial_value`: Value at the beginning of the transition.
    ///   - `end_value`: Value at the end of the transition.
    ///   - `power`: Power of the polynomial along which the value changes.
    ///   - `transition_steps`: Number of steps over which the value changes.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when a value or the power is not finite, or when
    /// `transition_steps` is `0`.
    pub fn new(
        initial_value: f64,
        end_value: f64,
        power: f64,
        transition_steps: usize,
    ) -> Result<Self, OptimizerError> {
        validate_finite("initial_value", initial_value)?;
        validate_finite("end_value", end_value)?;
        validate_finite("power", power)?;
        validate_positive_steps("transition_steps", transition_steps)?;
        Ok(Self { initial_value, end_value, power, transition_steps, transition_begin: 0 })
    }

    /// Returns this schedule with a transition that begins after `transition_begin` steps.
    #[inline]
    pub fn with_transition_begin(mut self, transition_begin: usize) -> Self {
        self.transition_begin = transition_begin;
        self
    }

    /// Returns the value of this schedule up to the beginning of its transition.
    #[inline]
    pub fn initial_value(&self) -> f64 {
        self.initial_value
    }

    /// Returns the value of this schedule from the end of its transition on.
    #[inline]
    pub fn end_value(&self) -> f64 {
        self.end_value
    }

    /// Returns the power of the polynomial along which the value of this schedule changes.
    #[inline]
    pub fn power(&self) -> f64 {
        self.power
    }

    /// Returns the number of steps over which the value of this schedule changes.
    #[inline]
    pub fn transition_steps(&self) -> usize {
        self.transition_steps
    }

    /// Returns the number of steps for which this schedule holds its initial value before its transition begins.
    #[inline]
    pub fn transition_begin(&self) -> usize {
        self.transition_begin
    }
}

impl Schedule for PolynomialSchedule {
    fn value<P: OptimizerValue>(&self, count: &P) -> Result<P, OptimizerError> {
        // This follows the operation order of `optax.polynomial_schedule`.
        let step = &convert_to(count, DataType::F32)?;
        let transition_steps = step.fill_like(self.transition_steps as f64)?;
        let count = Sub::sub(step, &step.fill_like(self.transition_begin as f64)?)?
            .max(&step.fill_like(0f64)?)?
            .min(&transition_steps)?;
        let fraction = Sub::sub(&step.fill_like(1f64)?, &Div::div(&count, &transition_steps)?)?;
        let fraction = fraction.pow(&step.fill_like(self.power)?)?;
        let scaled = Mul::mul(&step.fill_like(self.initial_value - self.end_value)?, &fraction)?;
        Ok(Add::add(&scaled, &step.fill_like(self.end_value)?)?)
    }
}

/// [`Schedule`] that decays an initial value exponentially after an optional delay, which is the analogue of
/// [`optax.exponential_decay`](
/// https://optax.readthedocs.io/en/latest/api/optimizer_schedules.html#optax.exponential_decay). Its value is
///
/// ```text
/// η(t) = η₀                     if t ≤ b
/// η(t) = η₀ · r^((t − b) / T)   if t > b
/// ```
///
/// Here `t` is the step count (i.e., the number of updates applied so far, starting at `0`), which optimizers pass as
/// the count before the increment of their current update. The initial value is `η₀`, `b` is the transition begin
/// (i.e., the number of steps for which `η₀` is held), `T` is the number of transition steps, and `r` is the decay
/// rate, by which the value is multiplied every `T` steps (so a rate above `1` grows the value). The defaults are
/// `b = 0`, a continuous decay, and no bound. [`Self::with_transition_begin`] sets `b`, [`Self::with_staircase`] rounds
/// the exponent down to `⌊(t − b) / T⌋` so that the value decays at discrete intervals, and [`Self::with_end_value`]
/// bounds the value by an end value `ηₑ` (i.e., `max(η(t), ηₑ)` when `r < 1` and `min(η(t), ηₑ)` otherwise).
///
/// This schedule is a closed form expression of the step count rather than a chain of other schedules. It has no
/// originating paper.
#[derive(Clone, Debug, PartialEq)]
pub struct ExponentialDecaySchedule {
    /// Value up to the beginning of the decay.
    initial_value: f64,

    /// Number of steps over which the value is multiplied by the decay rate, which is always positive.
    transition_steps: usize,

    /// Factor by which the value is multiplied every `transition_steps` steps, which is never zero.
    decay_rate: f64,

    /// Number of steps for which the initial value is held before the decay begins.
    transition_begin: usize,

    /// Whether the exponent is rounded down to an integer, so that the value decays at discrete intervals.
    staircase: bool,

    /// Bound at which the decay stops, which is a lower bound when the decay rate is less than `1` and an upper bound
    /// otherwise, or [`None`] if the decay never stops.
    end_value: Option<f64>,
}

impl ExponentialDecaySchedule {
    /// Creates a new [`ExponentialDecaySchedule`] that decays continuously from step `0` on without a bound.
    ///
    /// # Parameters
    ///
    ///   - `initial_value`: Value up to the beginning of the decay.
    ///   - `transition_steps`: Number of steps over which the value is multiplied by `decay_rate`.
    ///   - `decay_rate`: Factor by which the value is multiplied every `transition_steps` steps.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `initial_value` or `decay_rate` is not finite, when
    /// `decay_rate` is zero, or when `transition_steps` is `0`.
    pub fn new(initial_value: f64, transition_steps: usize, decay_rate: f64) -> Result<Self, OptimizerError> {
        validate_finite("initial_value", initial_value)?;
        validate_finite("decay_rate", decay_rate)?;
        if decay_rate == 0.0 {
            return Err(OptimizerError::InvalidConfiguration { message: "`decay_rate` must not be zero".to_string() });
        }
        validate_positive_steps("transition_steps", transition_steps)?;
        Ok(Self { initial_value, transition_steps, decay_rate, transition_begin: 0, staircase: false, end_value: None })
    }

    /// Returns this schedule with a decay that begins after `transition_begin` steps.
    #[inline]
    pub fn with_transition_begin(mut self, transition_begin: usize) -> Self {
        self.transition_begin = transition_begin;
        self
    }

    /// Returns this schedule with a decay at discrete intervals of `transition_steps` steps when `staircase` is `true`.
    #[inline]
    pub fn with_staircase(mut self, staircase: bool) -> Self {
        self.staircase = staircase;
        self
    }

    /// Returns this schedule with `end_value` as the bound at which its decay stops.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `end_value` is not finite.
    pub fn with_end_value(mut self, end_value: f64) -> Result<Self, OptimizerError> {
        validate_finite("end_value", end_value)?;
        self.end_value = Some(end_value);
        Ok(self)
    }

    /// Returns the value of this schedule up to the beginning of its decay.
    #[inline]
    pub fn initial_value(&self) -> f64 {
        self.initial_value
    }

    /// Returns the number of steps over which this schedule multiplies its value by its decay rate.
    #[inline]
    pub fn transition_steps(&self) -> usize {
        self.transition_steps
    }

    /// Returns the factor by which this schedule multiplies its value every [`Self::transition_steps`] steps.
    #[inline]
    pub fn decay_rate(&self) -> f64 {
        self.decay_rate
    }

    /// Returns the number of steps for which this schedule holds its initial value before its decay begins.
    #[inline]
    pub fn transition_begin(&self) -> usize {
        self.transition_begin
    }

    /// Returns whether this schedule decays at discrete intervals.
    #[inline]
    pub fn staircase(&self) -> bool {
        self.staircase
    }

    /// Returns the bound at which the decay of this schedule stops, if it has one.
    #[inline]
    pub fn end_value(&self) -> Option<f64> {
        self.end_value
    }
}

impl Schedule for ExponentialDecaySchedule {
    fn value<P: OptimizerValue>(&self, count: &P) -> Result<P, OptimizerError> {
        // This follows the operation order of `optax.exponential_decay`.
        let step = &convert_to(count, DataType::F32)?;
        let decreased_count = Sub::sub(step, &step.fill_like(self.transition_begin as f64)?)?;
        let exponent = Div::div(&decreased_count, &step.fill_like(self.transition_steps as f64)?)?;
        let exponent = if self.staircase { exponent.floor()? } else { exponent };
        let initial_value = step.fill_like(self.initial_value)?;
        let decayed = Mul::mul(&initial_value, &step.fill_like(self.decay_rate)?.pow(&exponent)?)?;
        let not_started = decreased_count.compare(&step.fill_like(0f64)?, ComparisonDirection::LessThanOrEqual)?;
        let value = not_started.select(&initial_value, &decayed)?;
        match self.end_value {
            None => Ok(value),
            Some(end_value) if self.decay_rate < 1.0 => Ok(value.max(&step.fill_like(end_value)?)?),
            Some(end_value) => Ok(value.min(&step.fill_like(end_value)?)?),
        }
    }
}

/// [`Schedule`] that scales an initial value by a factor at each of a set of step boundaries, which is the analogue of
/// [`optax.piecewise_constant_schedule`](
/// https://optax.readthedocs.io/en/latest/api/optimizer_schedules.html#optax.piecewise_constant_schedule). Its value is
///
/// ```text
/// η(t) = η₀ · ∏ { sᵢ : bᵢ ≤ t }
/// ```
///
/// Here `t` is the step count (i.e., the number of updates applied so far, starting at `0`), which optimizers pass as
/// the count before the increment of their current update. The initial value is `η₀`, and every boundary `bᵢ` has the
/// scale `sᵢ`, which multiplies the value from that boundary on, so the value is the initial value multiplied by the
/// scales of all boundaries that the step count has reached.
///
/// This schedule is a closed form expression of the step count rather than a chain of other schedules. It has no
/// originating paper.
#[derive(Clone, Debug, PartialEq)]
pub struct PiecewiseConstantSchedule {
    /// Value before the first boundary.
    initial_value: f64,

    /// Scales by step boundary, which are all finite and non-negative.
    boundaries_and_scales: BTreeMap<usize, f64>,
}

impl PiecewiseConstantSchedule {
    /// Creates a new [`PiecewiseConstantSchedule`] that starts at `initial_value` and multiplies its value by the scale
    /// of each boundary of `boundaries_and_scales` from that boundary on.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `initial_value` is not finite or when a scale is negative
    /// or not finite.
    pub fn new(initial_value: f64, boundaries_and_scales: BTreeMap<usize, f64>) -> Result<Self, OptimizerError> {
        validate_finite("initial_value", initial_value)?;
        validate_scales(&boundaries_and_scales)?;
        Ok(Self { initial_value, boundaries_and_scales })
    }

    /// Returns the value of this schedule before its first boundary.
    #[inline]
    pub fn initial_value(&self) -> f64 {
        self.initial_value
    }

    /// Returns the scales of this schedule by step boundary.
    #[inline]
    pub fn boundaries_and_scales(&self) -> &BTreeMap<usize, f64> {
        &self.boundaries_and_scales
    }
}

impl Schedule for PiecewiseConstantSchedule {
    fn value<P: OptimizerValue>(&self, count: &P) -> Result<P, OptimizerError> {
        // This follows the operation order of `optax.piecewise_constant_schedule`, in which the indicator of each
        // boundary `b` is `max(0, sign(b - t))`, which is `1` before the boundary and `0` from it on.
        let step = &convert_to(count, DataType::F32)?;
        let one = step.fill_like(1f64)?;
        let mut value = step.fill_like(self.initial_value)?;
        for (&boundary, &scale) in &self.boundaries_and_scales {
            let indicator = Sub::sub(&step.fill_like(boundary as f64)?, step)?.sign()?.max(&step.fill_like(0f64)?)?;
            let scaled = Mul::mul(&Mul::mul(&Sub::sub(&one, &indicator)?, &step.fill_like(scale)?)?, &value)?;
            value = Add::add(&Mul::mul(&value, &indicator)?, &scaled)?;
        }
        Ok(value)
    }
}

/// Interpolation that [`PiecewiseInterpolateSchedule`] uses between the values at consecutive boundaries.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Interpolation {
    /// Linear interpolation, which at the fraction `f` of an interval from `start` to `end` is
    /// `(end - start) · f + start`.
    Linear,

    /// Cosine interpolation, which at the fraction `f` of an interval from `start` to `end` is
    /// `end + (start - end) / 2 · (cos(π · f) + 1)` and so has zero slope at both ends.
    Cosine,
}

/// [`Schedule`] that interpolates between the values that a [`PiecewiseConstantSchedule`] with the same boundaries and
/// scales takes at its boundaries, which is the analogue of [`optax.piecewise_interpolate_schedule`](
/// https://optax.readthedocs.io/en/latest/api/optimizer_schedules.html#optax.piecewise_interpolate_schedule). With the
/// values `v₀ = η₀` and `vᵢ = vᵢ₋₁ · sᵢ` at the boundaries, its value is
///
/// ```text
/// fᵢ(t) = (t − bᵢ) / max(bᵢ₊₁ − bᵢ, 1)
/// η(t)  = vᵢ + (vᵢ₊₁ − vᵢ) · fᵢ(t)                        if bᵢ ≤ t < bᵢ₊₁, with linear interpolation
/// η(t)  = vᵢ₊₁ + (vᵢ − vᵢ₊₁) · (cos(π · fᵢ(t)) + 1) / 2   if bᵢ ≤ t < bᵢ₊₁, with cosine interpolation
/// η(t)  = vₙ                                              if t ≥ bₙ
/// ```
///
/// Here `t` is the step count (i.e., the number of updates applied so far, starting at `0`), which optimizers pass as
/// the count before the increment of their current update. The initial value is `η₀`, the boundaries are
/// `0 = b₀ ≤ b₁ < … < bₙ` with the scales `sᵢ`, `vᵢ` is the value at boundary `bᵢ`, which is computed in `f32` in
/// boundary order, and `fᵢ(t)` is the fraction of the interval from `bᵢ` to `bᵢ₊₁` that the step count has covered.
/// [`Interpolation::Linear`] interpolates linearly and [`Interpolation::Cosine`] interpolates along half a cosine
/// period, which has zero slope at both ends. Note that the values are interpolated between the accumulated values `vᵢ`
/// rather than between the scales themselves.
///
/// This schedule is a closed form expression of the step count rather than a chain of other schedules. It has no
/// originating paper, and the one-cycle schedules [`LinearOneCycleSchedule`] and [`CosineOneCycleSchedule`] are built
/// from it.
#[derive(Clone, Debug, PartialEq)]
pub struct PiecewiseInterpolateSchedule {
    /// Interpolation between the values at consecutive boundaries.
    interpolation: Interpolation,

    /// Value at step `0`.
    initial_value: f64,

    /// Scales by step boundary, which are all finite and non-negative.
    boundaries_and_scales: BTreeMap<usize, f64>,
}

impl PiecewiseInterpolateSchedule {
    /// Creates a new [`PiecewiseInterpolateSchedule`].
    ///
    /// # Parameters
    ///
    ///   - `interpolation`: [`Interpolation`] between the values at consecutive boundaries.
    ///   - `initial_value`: Value at step `0`.
    ///   - `boundaries_and_scales`: Scales by step boundary, whose running products with `initial_value` are the values
    ///     at the boundaries.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `initial_value` is not finite or when a scale is negative
    /// or not finite.
    pub fn new(
        interpolation: Interpolation,
        initial_value: f64,
        boundaries_and_scales: BTreeMap<usize, f64>,
    ) -> Result<Self, OptimizerError> {
        validate_finite("initial_value", initial_value)?;
        validate_scales(&boundaries_and_scales)?;
        Ok(Self { interpolation, initial_value, boundaries_and_scales })
    }

    /// Returns the [`Interpolation`] of this schedule between the values at consecutive boundaries.
    #[inline]
    pub fn interpolation(&self) -> Interpolation {
        self.interpolation
    }

    /// Returns the value of this schedule at step `0`.
    #[inline]
    pub fn initial_value(&self) -> f64 {
        self.initial_value
    }

    /// Returns the scales of this schedule by step boundary.
    #[inline]
    pub fn boundaries_and_scales(&self) -> &BTreeMap<usize, f64> {
        &self.boundaries_and_scales
    }
}

impl Schedule for PiecewiseInterpolateSchedule {
    fn value<P: OptimizerValue>(&self, count: &P) -> Result<P, OptimizerError> {
        // This follows `optax.piecewise_interpolate_schedule`, which sums the interpolated values of all intervals
        // weighted by the indicators of whether the step falls into them, and adds the last value from the last
        // boundary on. At most one indicator is set, so the weighted sum is a selection. The values are the
        // cumulative products of the initial value and the scales in `f32`, as `jnp.cumprod` computes them.
        let step = &convert_to(count, DataType::F32)?;
        let mut boundaries = vec![0usize];
        let mut values = vec![self.initial_value as f32];
        for (&boundary, &scale) in &self.boundaries_and_scales {
            boundaries.push(boundary);
            values.push(values[values.len() - 1] * scale as f32);
        }
        let zero = step.fill_like(0f64)?;
        let last_boundary = step.fill_like(boundaries[boundaries.len() - 1] as f64)?;
        let after_last = last_boundary.compare(step, ComparisonDirection::LessThanOrEqual)?;
        let mut value = after_last.select(&step.fill_like(f64::from(values[values.len() - 1]))?, &zero)?;
        for index in 0..boundaries.len() - 1 {
            let (start, end) =
                (step.fill_like(f64::from(values[index]))?, step.fill_like(f64::from(values[index + 1]))?);
            let begin = step.fill_like(boundaries[index] as f64)?;
            let limit = step.fill_like(boundaries[index + 1] as f64)?;
            let size = step.fill_like(boundaries[index + 1].saturating_sub(boundaries[index]).max(1) as f64)?;
            let fraction = Div::div(&Sub::sub(step, &begin)?, &size)?;
            let interpolated = match self.interpolation {
                Interpolation::Linear => Add::add(&Mul::mul(&Sub::sub(&end, &start)?, &fraction)?, &start)?,
                Interpolation::Cosine => {
                    let cosine = Add::add(&Mul::mul(&step.fill_like(PI)?, &fraction)?.cos()?, &step.fill_like(1f64)?)?;
                    let half_range = Div::div(&Sub::sub(&start, &end)?, &step.fill_like(2f64)?)?;
                    Add::add(&end, &Mul::mul(&half_range, &cosine)?)?
                }
            };
            let inside = begin
                .compare(step, ComparisonDirection::LessThanOrEqual)?
                .and(&step.compare(&limit, ComparisonDirection::LessThan)?)?;
            value = Add::add(&inside.select(&interpolated, &zero)?, &value)?;
        }
        Ok(value)
    }
}

/// [`Schedule`] with three linear phases, which is the one-cycle schedule of
/// [Smith and Topin (2017)](https://arxiv.org/abs/1708.07120) and the analogue of [`optax.linear_onecycle_schedule`](
/// https://optax.readthedocs.io/en/latest/api/optimizer_schedules.html#optax.linear_onecycle_schedule). Its value
/// interpolates linearly between
///
/// ```text
/// η(0)       = ηₚ / d
/// η(⌊s · T⌋) = ηₚ
/// η(⌊f · T⌋) = ηₚ / d
/// η(T)       = ηₚ / (d · D)
/// ```
///
/// and stays at `η(T)` after step `T`. Here `t` is the step count (i.e., the number of updates applied so far, starting
/// at `0`), which optimizers pass as the count before the increment of their current update. The peak value is `ηₚ`,
/// `T` is the number of steps of the cycle, `s` is the fraction of the cycle that the rising phase spans, `f` is the
/// fraction that the rising and falling phases span together, `d` is the division factor of the peak value that gives
/// the initial value, and `D` is the division factor of the initial value that gives the final value. The defaults are
/// `s = 0.3`, `f = 0.85`, `d = 25`, and `D = 1e4`, which [`Self::with_start_fraction`], [`Self::with_final_fraction`],
/// [`Self::with_division_factor`], and [`Self::with_final_division_factor`] change.
///
/// It is the [`PiecewiseInterpolateSchedule`] with [`Interpolation::Linear`], the initial value `ηₚ / d`, and the
/// scales `d`, `1 / d`, and `1 / D` at the boundaries `⌊s · T⌋`, `⌊f · T⌋`, and `T` (where a later boundary replaces an
/// earlier one at the same step).
///
/// # References
///
///   - Smith and Topin (2017). [Super-Convergence: Very Fast Training of Neural Networks Using Large Learning
///     Rates](https://arxiv.org/abs/1708.07120).
#[derive(Clone, Debug, PartialEq)]
pub struct LinearOneCycleSchedule {
    /// Number of steps of the cycle, which is always positive.
    transition_steps: usize,

    /// Value at the end of the rising phase.
    peak_value: f64,

    /// Fraction of the cycle that the rising phase spans, which is in `[0, 1]`.
    start_fraction: f64,

    /// Fraction of the cycle that the rising and falling phases span together, which is in `[0, 1]`.
    final_fraction: f64,

    /// Division factor of the peak value that gives the initial value, which is positive and finite.
    division_factor: f64,

    /// Division factor of the initial value that gives the final value, which is positive and finite.
    final_division_factor: f64,
}

impl LinearOneCycleSchedule {
    /// Creates a new [`LinearOneCycleSchedule`] with a cycle of `transition_steps` steps that peaks at `peak_value`,
    /// with the default fractions and division factors.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `peak_value` is not finite or when `transition_steps` is
    /// `0`.
    pub fn new(transition_steps: usize, peak_value: f64) -> Result<Self, OptimizerError> {
        validate_positive_steps("transition_steps", transition_steps)?;
        validate_finite("peak_value", peak_value)?;
        Ok(Self {
            transition_steps,
            peak_value,
            start_fraction: 0.3,
            final_fraction: 0.85,
            division_factor: 25.0,
            final_division_factor: 1e4,
        })
    }

    /// Returns this schedule with `start_fraction` as the fraction of the cycle that its rising phase spans.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `start_fraction` is not in `[0, 1]`.
    pub fn with_start_fraction(mut self, start_fraction: f64) -> Result<Self, OptimizerError> {
        validate_fraction("start_fraction", start_fraction)?;
        self.start_fraction = start_fraction;
        Ok(self)
    }

    /// Returns this schedule with `final_fraction` as the fraction of the cycle that its rising and falling phases
    /// span together.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `final_fraction` is not in `[0, 1]`.
    pub fn with_final_fraction(mut self, final_fraction: f64) -> Result<Self, OptimizerError> {
        validate_fraction("final_fraction", final_fraction)?;
        self.final_fraction = final_fraction;
        Ok(self)
    }

    /// Returns this schedule with `division_factor` as the factor by which its peak value is divided to give its
    /// initial value.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `division_factor` is not positive and finite.
    pub fn with_division_factor(mut self, division_factor: f64) -> Result<Self, OptimizerError> {
        validate_positive("division_factor", division_factor)?;
        self.division_factor = division_factor;
        Ok(self)
    }

    /// Returns this schedule with `final_division_factor` as the factor by which its initial value is divided to give
    /// its final value.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `final_division_factor` is not positive and finite.
    pub fn with_final_division_factor(mut self, final_division_factor: f64) -> Result<Self, OptimizerError> {
        validate_positive("final_division_factor", final_division_factor)?;
        self.final_division_factor = final_division_factor;
        Ok(self)
    }

    /// Returns the number of steps of the cycle of this schedule.
    #[inline]
    pub fn transition_steps(&self) -> usize {
        self.transition_steps
    }

    /// Returns the value of this schedule at the end of its rising phase.
    #[inline]
    pub fn peak_value(&self) -> f64 {
        self.peak_value
    }

    /// Returns the fraction of the cycle that the rising phase of this schedule spans.
    #[inline]
    pub fn start_fraction(&self) -> f64 {
        self.start_fraction
    }

    /// Returns the fraction of the cycle that the rising and falling phases of this schedule span together.
    #[inline]
    pub fn final_fraction(&self) -> f64 {
        self.final_fraction
    }

    /// Returns the factor by which this schedule divides its peak value to give its initial value.
    #[inline]
    pub fn division_factor(&self) -> f64 {
        self.division_factor
    }

    /// Returns the factor by which this schedule divides its initial value to give its final value.
    #[inline]
    pub fn final_division_factor(&self) -> f64 {
        self.final_division_factor
    }

    /// Returns the [`PiecewiseInterpolateSchedule`] that this schedule evaluates.
    fn piecewise(&self) -> PiecewiseInterpolateSchedule {
        // The boundaries are inserted in the order of the dictionary of Optax, so a later boundary at the same step
        // replaces an earlier one, as it does there.
        let steps = self.transition_steps as f64;
        let mut boundaries_and_scales = BTreeMap::new();
        boundaries_and_scales.insert((self.start_fraction * steps) as usize, self.division_factor);
        boundaries_and_scales.insert((self.final_fraction * steps) as usize, 1.0 / self.division_factor);
        boundaries_and_scales.insert(self.transition_steps, 1.0 / self.final_division_factor);
        PiecewiseInterpolateSchedule {
            interpolation: Interpolation::Linear,
            initial_value: self.peak_value / self.division_factor,
            boundaries_and_scales,
        }
    }
}

impl Schedule for LinearOneCycleSchedule {
    fn value<P: OptimizerValue>(&self, count: &P) -> Result<P, OptimizerError> {
        self.piecewise().value(count)
    }
}

/// [`Schedule`] that rises and then falls along half cosine periods, which is a cosine variant of the one-cycle
/// schedule of [Smith and Topin (2017)](https://arxiv.org/abs/1708.07120) and the analogue of
/// [`optax.cosine_onecycle_schedule`](
/// https://optax.readthedocs.io/en/latest/api/optimizer_schedules.html#optax.cosine_onecycle_schedule). Its value
/// interpolates along half cosine periods between
///
/// ```text
/// η(0)       = ηₚ / d
/// η(⌊s · T⌋) = ηₚ
/// η(T)       = ηₚ / (d · D)
/// ```
///
/// and stays at `η(T)` after step `T`. Here `t` is the step count (i.e., the number of updates applied so far, starting
/// at `0`), which optimizers pass as the count before the increment of their current update. The peak value is `ηₚ`,
/// `T` is the number of steps of the cycle, `s` is the fraction of the cycle that the rising phase spans, `d` is the
/// division factor of the peak value that gives the initial value, and `D` is the division factor of the initial value
/// that gives the final value. Between two of these steps `t₀ < t₁`, the value at the fraction
/// `φ = (t − t₀) / (t₁ − t₀)` is `η(t₁) + (η(t₀) − η(t₁)) · (cos(π · φ) + 1) / 2`. The defaults are `s = 0.3`,
/// `d = 25`, and `D = 1e4`, which [`Self::with_start_fraction`], [`Self::with_division_factor`], and
/// [`Self::with_final_division_factor`] change.
///
/// It is the [`PiecewiseInterpolateSchedule`] with [`Interpolation::Cosine`], the initial value `ηₚ / d`, and the
/// scales `d` and `1 / (d · D)` at the boundaries `⌊s · T⌋` and `T` (where the second boundary replaces the first one
/// at the same step).
///
/// # References
///
///   - Smith and Topin (2017). [Super-Convergence: Very Fast Training of Neural Networks Using Large Learning
///     Rates](https://arxiv.org/abs/1708.07120).
#[derive(Clone, Debug, PartialEq)]
pub struct CosineOneCycleSchedule {
    /// Number of steps of the cycle, which is always positive.
    transition_steps: usize,

    /// Value at the end of the rising phase.
    peak_value: f64,

    /// Fraction of the cycle that the rising phase spans, which is in `[0, 1]`.
    start_fraction: f64,

    /// Division factor of the peak value that gives the initial value, which is positive and finite.
    division_factor: f64,

    /// Division factor of the initial value that gives the final value, which is positive and finite.
    final_division_factor: f64,
}

impl CosineOneCycleSchedule {
    /// Creates a new [`CosineOneCycleSchedule`] with a cycle of `transition_steps` steps that peaks at `peak_value`,
    /// with the default fraction and division factors.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `peak_value` is not finite or when `transition_steps` is
    /// `0`.
    pub fn new(transition_steps: usize, peak_value: f64) -> Result<Self, OptimizerError> {
        validate_positive_steps("transition_steps", transition_steps)?;
        validate_finite("peak_value", peak_value)?;
        Ok(Self {
            transition_steps,
            peak_value,
            start_fraction: 0.3,
            division_factor: 25.0,
            final_division_factor: 1e4,
        })
    }

    /// Returns this schedule with `start_fraction` as the fraction of the cycle that its rising phase spans.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `start_fraction` is not in `[0, 1]`.
    pub fn with_start_fraction(mut self, start_fraction: f64) -> Result<Self, OptimizerError> {
        validate_fraction("start_fraction", start_fraction)?;
        self.start_fraction = start_fraction;
        Ok(self)
    }

    /// Returns this schedule with `division_factor` as the factor by which its peak value is divided to give its
    /// initial value.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `division_factor` is not positive and finite.
    pub fn with_division_factor(mut self, division_factor: f64) -> Result<Self, OptimizerError> {
        validate_positive("division_factor", division_factor)?;
        self.division_factor = division_factor;
        Ok(self)
    }

    /// Returns this schedule with `final_division_factor` as the factor by which its initial value is divided to give
    /// its final value.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `final_division_factor` is not positive and finite.
    pub fn with_final_division_factor(mut self, final_division_factor: f64) -> Result<Self, OptimizerError> {
        validate_positive("final_division_factor", final_division_factor)?;
        self.final_division_factor = final_division_factor;
        Ok(self)
    }

    /// Returns the number of steps of the cycle of this schedule.
    #[inline]
    pub fn transition_steps(&self) -> usize {
        self.transition_steps
    }

    /// Returns the value of this schedule at the end of its rising phase.
    #[inline]
    pub fn peak_value(&self) -> f64 {
        self.peak_value
    }

    /// Returns the fraction of the cycle that the rising phase of this schedule spans.
    #[inline]
    pub fn start_fraction(&self) -> f64 {
        self.start_fraction
    }

    /// Returns the factor by which this schedule divides its peak value to give its initial value.
    #[inline]
    pub fn division_factor(&self) -> f64 {
        self.division_factor
    }

    /// Returns the factor by which this schedule divides its initial value to give its final value.
    #[inline]
    pub fn final_division_factor(&self) -> f64 {
        self.final_division_factor
    }

    /// Returns the [`PiecewiseInterpolateSchedule`] that this schedule evaluates.
    fn piecewise(&self) -> PiecewiseInterpolateSchedule {
        let mut boundaries_and_scales = BTreeMap::new();
        boundaries_and_scales
            .insert((self.start_fraction * self.transition_steps as f64) as usize, self.division_factor);
        boundaries_and_scales.insert(self.transition_steps, 1.0 / (self.division_factor * self.final_division_factor));
        PiecewiseInterpolateSchedule {
            interpolation: Interpolation::Cosine,
            initial_value: self.peak_value / self.division_factor,
            boundaries_and_scales,
        }
    }
}

impl Schedule for CosineOneCycleSchedule {
    fn value<P: OptimizerValue>(&self, count: &P) -> Result<P, OptimizerError> {
        self.piecewise().value(count)
    }
}

/// [`Schedule`] that warms up linearly from an initial value to a peak value and then decays exponentially, which
/// combines the linear warmup of [Goyal et al. (2017)](https://arxiv.org/abs/1706.02677) with an exponential decay, and
/// which is the analogue of [`optax.warmup_exponential_decay_schedule`](
/// https://optax.readthedocs.io/en/latest/api/optimizer_schedules.html#optax.warmup_exponential_decay_schedule). Its
/// value is
///
/// ```text
/// η(t) = η₀ + (ηₚ − η₀) · t / W     if t < W
/// η(t) = ηₚ                         if W ≤ t ≤ W + b
/// η(t) = ηₚ · r^((t − W − b) / T)   if t > W + b
/// ```
///
/// Here `t` is the step count (i.e., the number of updates applied so far, starting at `0`), which optimizers pass as
/// the count before the increment of their current update. The initial value is `η₀`, the peak value is `ηₚ`, `W` is
/// the number of warmup steps, `b` is the transition begin of the decay (counted from the end of the warmup), `T` is
/// the number of transition steps of the decay, and `r` is its decay rate. The defaults are `b = 0`, a continuous
/// decay, and no bound. [`Self::with_transition_begin`] sets `b`, [`Self::with_staircase`] rounds the exponent down to
/// an integer, and [`Self::with_end_value`] bounds the decayed value by an end value, as [`ExponentialDecaySchedule`]
/// does.
///
/// It is the [`JoinedSchedule`] of a [`LinearSchedule`] for the warmup and an [`ExponentialDecaySchedule`] that decays
/// the peak value, joined at step `W`.
///
/// # References
///
///   - Goyal et al. (2017). [Accurate, Large Minibatch SGD: Training ImageNet in 1
///     Hour](https://arxiv.org/abs/1706.02677).
#[derive(Clone, Debug, PartialEq)]
pub struct WarmupExponentialDecaySchedule {
    /// Value at step `0`.
    initial_value: f64,

    /// Number of warmup steps.
    warmup_steps: usize,

    /// Decay that follows the warmup, whose initial value is the peak value.
    decay: ExponentialDecaySchedule,
}

impl WarmupExponentialDecaySchedule {
    /// Creates a new [`WarmupExponentialDecaySchedule`] whose decay begins right after the warmup, decays continuously,
    /// and has no bound.
    ///
    /// # Parameters
    ///
    ///   - `initial_value`: Value at step `0`.
    ///   - `peak_value`: Value at the end of the warmup.
    ///   - `warmup_steps`: Number of steps of the linear warmup, which may be `0` to start at the peak value.
    ///   - `transition_steps`: Number of steps over which the decay multiplies the value by `decay_rate`.
    ///   - `decay_rate`: Factor by which the value is multiplied every `transition_steps` steps.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] under the conditions of [`ExponentialDecaySchedule::new`] or
    /// when `initial_value` is not finite.
    pub fn new(
        initial_value: f64,
        peak_value: f64,
        warmup_steps: usize,
        transition_steps: usize,
        decay_rate: f64,
    ) -> Result<Self, OptimizerError> {
        validate_finite("initial_value", initial_value)?;
        let decay = ExponentialDecaySchedule::new(peak_value, transition_steps, decay_rate)?;
        Ok(Self { initial_value, warmup_steps, decay })
    }

    /// Returns this schedule with a decay that begins `transition_begin` steps after the end of the warmup.
    #[inline]
    pub fn with_transition_begin(mut self, transition_begin: usize) -> Self {
        self.decay = self.decay.with_transition_begin(transition_begin);
        self
    }

    /// Returns this schedule with a decay at discrete intervals when `staircase` is `true`.
    #[inline]
    pub fn with_staircase(mut self, staircase: bool) -> Self {
        self.decay = self.decay.with_staircase(staircase);
        self
    }

    /// Returns this schedule with `end_value` as the bound at which its decay stops (refer to
    /// [`ExponentialDecaySchedule::with_end_value`]).
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `end_value` is not finite.
    pub fn with_end_value(mut self, end_value: f64) -> Result<Self, OptimizerError> {
        self.decay = self.decay.with_end_value(end_value)?;
        Ok(self)
    }

    /// Returns the value of this schedule at step `0`.
    #[inline]
    pub fn initial_value(&self) -> f64 {
        self.initial_value
    }

    /// Returns the number of warmup steps of this schedule.
    #[inline]
    pub fn warmup_steps(&self) -> usize {
        self.warmup_steps
    }

    /// Returns the [`ExponentialDecaySchedule`] that follows the warmup of this schedule, whose initial value is the
    /// peak value.
    #[inline]
    pub fn decay(&self) -> &ExponentialDecaySchedule {
        &self.decay
    }
}

impl Schedule for WarmupExponentialDecaySchedule {
    fn value<P: OptimizerValue>(&self, count: &P) -> Result<P, OptimizerError> {
        // This joins the two schedules at the warmup boundary as `optax.join_schedules` does. The constructor
        // validated the values, so the warmup schedule is built directly.
        if self.warmup_steps == 0 {
            return self.decay.value(count);
        }
        let warmup = LinearSchedule {
            initial_value: self.initial_value,
            end_value: self.decay.initial_value,
            transition_steps: self.warmup_steps,
            transition_begin: 0,
        };
        JoinedSchedule::new(warmup, self.warmup_steps, &self.decay).value(count)
    }
}

/// [`Schedule`] that restarts a [`WarmupCosineDecaySchedule`] for every cycle, which is the SGDR schedule of
/// [Loshchilov and Hutter (2017)](https://arxiv.org/abs/1608.03983) and the analogue of
/// [`optax.sgdr_schedule`](https://optax.readthedocs.io/en/latest/api/optimizer_schedules.html#optax.sgdr_schedule).
/// With the cycle schedules `S₀, …, Sₙ₋₁`, its value is
///
/// ```text
/// Bⱼ   = D₀ + … + Dⱼ₋₁
/// η(t) = Sⱼ(t − Bⱼ)       if Bⱼ ≤ t < Bⱼ₊₁ and j < n − 1
/// η(t) = Sₙ₋₁(t − Bₙ₋₁)   if t ≥ Bₙ₋₁
/// ```
///
/// Here `t` is the step count (i.e., the number of updates applied so far, starting at `0`), which optimizers pass as
/// the count before the increment of their current update. The cycle `j` lasts for the decay steps `Dⱼ` of its schedule
/// `Sⱼ` (which include its warmup steps), `Bⱼ` is the step at which it begins (i.e., `B₀ = 0`), and its schedule is
/// evaluated at the number of steps since that step, so the last cycle continues after its end with its end value.
///
/// It joins the schedules of the cycles at their boundaries, as nested [`JoinedSchedule`]s do.
///
/// # References
///
///   - Loshchilov and Hutter (2017). [SGDR: Stochastic Gradient Descent with Warm
///     Restarts](https://arxiv.org/abs/1608.03983).
#[derive(Clone, Debug, PartialEq)]
pub struct SgdrSchedule {
    /// Schedules of the cycles in order, of which there is at least one.
    cycles: Vec<WarmupCosineDecaySchedule>,
}

impl SgdrSchedule {
    /// Creates a new [`SgdrSchedule`] with the schedules of its `cycles` in order.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `cycles` is empty.
    pub fn new(cycles: Vec<WarmupCosineDecaySchedule>) -> Result<Self, OptimizerError> {
        if cycles.is_empty() {
            return Err(OptimizerError::InvalidConfiguration {
                message: "`cycles` must contain at least one schedule".to_string(),
            });
        }
        Ok(Self { cycles })
    }

    /// Returns the schedules of the cycles of this schedule in order.
    #[inline]
    pub fn cycles(&self) -> &[WarmupCosineDecaySchedule] {
        &self.cycles
    }
}

impl Schedule for SgdrSchedule {
    fn value<P: OptimizerValue>(&self, count: &P) -> Result<P, OptimizerError> {
        // This follows `optax.join_schedules`, which evaluates the first schedule and then replaces its value with that
        // of every later schedule from that schedule's boundary on.
        let step = &convert_to(count, DataType::F32)?;
        let mut value = self.cycles[0].value(step)?;
        let mut boundary = 0usize;
        for (previous, cycle) in self.cycles.iter().zip(&self.cycles[1..]) {
            boundary += previous.decay_steps;
            let boundary_value = step.fill_like(boundary as f64)?;
            let before = step.compare(&boundary_value, ComparisonDirection::LessThan)?;
            value = before.select(&value, &cycle.value(&Sub::sub(step, &boundary_value)?)?)?;
        }
        Ok(value)
    }
}

/// [`Schedule`] that follows one schedule up to a step boundary and another schedule from it on, which is the analogue
/// of [`optax.join_schedules`](
/// https://optax.readthedocs.io/en/latest/api/optimizer_schedules.html#optax.join_schedules) for two schedules. Its
/// value is
///
/// ```text
/// η(t) = A(t)       if t < b
/// η(t) = B(t − b)   if t ≥ b
/// ```
///
/// Here `t` is the step count (i.e., the number of updates applied so far, starting at `0`), which optimizers pass as
/// the count before the increment of their current update. The first schedule is `A`, the second schedule is `B`, and
/// `b` is the boundary, so the second schedule is evaluated at the number of steps since the boundary (i.e., at step
/// `0` at the boundary).
///
/// Joined schedules nest, so [`Self::then`] joins more schedules of any [`Schedule`] types (e.g.,
/// `JoinedSchedule::new(warmup, 100, decay).then(1000, 1e-5)` follows `warmup` up to step `100`, `decay` from there to
/// step `1000`, and the constant `1e-5` afterwards). It has no originating paper.
#[derive(Clone, Debug, PartialEq)]
pub struct JoinedSchedule<A, B> {
    /// Schedule that is followed before the boundary.
    first: A,

    /// Step from which the second schedule is followed.
    boundary: usize,

    /// Schedule that is followed from the boundary on.
    second: B,
}

impl<A: Schedule, B: Schedule> JoinedSchedule<A, B> {
    /// Creates a new [`JoinedSchedule`] that follows `first` before step `boundary` and `second` from it on.
    #[inline]
    pub fn new(first: A, boundary: usize, second: B) -> Self {
        Self { first, boundary, second }
    }

    /// Returns a [`JoinedSchedule`] that follows this schedule before step `boundary` and `next` from it on, where
    /// `boundary` counts the steps of this schedule from its beginning.
    #[inline]
    pub fn then<N: Schedule>(self, boundary: usize, next: N) -> JoinedSchedule<Self, N> {
        JoinedSchedule::new(self, boundary, next)
    }

    /// Returns the schedule that this schedule follows before its boundary.
    #[inline]
    pub fn first(&self) -> &A {
        &self.first
    }

    /// Returns the step from which this schedule follows its second schedule.
    #[inline]
    pub fn boundary(&self) -> usize {
        self.boundary
    }

    /// Returns the schedule that this schedule follows from its boundary on.
    #[inline]
    pub fn second(&self) -> &B {
        &self.second
    }
}

impl<A: Schedule, B: Schedule> Schedule for JoinedSchedule<A, B> {
    fn value<P: OptimizerValue>(&self, count: &P) -> Result<P, OptimizerError> {
        let step = &convert_to(count, DataType::F32)?;
        let boundary = step.fill_like(self.boundary as f64)?;
        let before = step.compare(&boundary, ComparisonDirection::LessThan)?;
        Ok(before.select(&self.first.value(step)?, &self.second.value(&Sub::sub(step, &boundary)?)?)?)
    }
}

impl<Sc: Schedule> Schedule for &Sc {
    fn value<P: OptimizerValue>(&self, count: &P) -> Result<P, OptimizerError> {
        (*self).value(count)
    }
}

/// Validates that the step count hyperparameter `name` has a positive value `steps`.
fn validate_positive_steps(name: &str, steps: usize) -> Result<(), OptimizerError> {
    if steps == 0 {
        return Err(OptimizerError::InvalidConfiguration { message: format!("`{name}` must be positive but got `0`") });
    }
    Ok(())
}

/// Validates that every scale of `boundaries_and_scales` is finite and non-negative.
fn validate_scales(boundaries_and_scales: &BTreeMap<usize, f64>) -> Result<(), OptimizerError> {
    for (boundary, scale) in boundaries_and_scales {
        if !scale.is_finite() || *scale < 0.0 {
            return Err(OptimizerError::InvalidConfiguration {
                message: format!(
                    "the scale at boundary `{boundary}` must be finite and non-negative but got `{scale}`"
                ),
            });
        }
    }
    Ok(())
}

/// Validates that the fraction hyperparameter `name` has a `value` in `[0, 1]`.
fn validate_fraction(name: &str, value: f64) -> Result<(), OptimizerError> {
    if !(0.0..=1.0).contains(&value) {
        return Err(OptimizerError::InvalidConfiguration {
            message: format!("`{name}` must be in `[0, 1]` but got `{value}`"),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use ryft_core::{Array, ArrayType, Typed};

    use crate::tests::assert_parity;

    use super::*;

    /// Returns the value of `schedule` at the rank-0 `i32` step count `count` as an `f32`, checking that it is a
    /// rank-0 `f32` array.
    fn value_at<Sc: Schedule>(schedule: &Sc, count: i32) -> f32 {
        let value = schedule.value(&Array::scalar(count).unwrap()).unwrap();
        assert_eq!(value.r#type().as_ref(), &ArrayType::scalar(DataType::F32));
        value.elements::<f32>().unwrap()[0]
    }

    #[test]
    fn test_f64_schedule_value() {
        assert_eq!(value_at(&0.5f64, 0), 0.5);
        assert_eq!(value_at(&0.5f64, 1000), 0.5);
    }

    #[test]
    fn test_linear_schedule_new() {
        let schedule = LinearSchedule::new(1.0, 0.25, 4).unwrap();
        assert_eq!(schedule.initial_value(), 1.0);
        assert_eq!(schedule.end_value(), 0.25);
        assert_eq!(schedule.transition_steps(), 4);
        assert_eq!(schedule.transition_begin(), 0);
    }

    #[test]
    fn test_linear_schedule_new_invalid_configuration() {
        assert!(matches!(
            LinearSchedule::new(f64::NAN, 0.0, 4),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`initial_value` must be finite but got `NaN`",
        ));
        assert!(matches!(
            LinearSchedule::new(1.0, f64::INFINITY, 4),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`end_value` must be finite but got `inf`",
        ));
        assert!(matches!(
            LinearSchedule::new(1.0, 0.0, 0),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`transition_steps` must be positive but got `0`",
        ));
    }

    #[test]
    fn test_linear_schedule_with_transition_begin() {
        let schedule = LinearSchedule::new(1.0, 0.25, 4).unwrap().with_transition_begin(2);
        assert_eq!(schedule.transition_begin(), 2);
    }

    #[test]
    fn test_linear_schedule_value() {
        // The value is held before the transition, interpolated during it, and constant after it.
        let schedule = LinearSchedule::new(1.0, 0.25, 4).unwrap().with_transition_begin(2);
        assert_eq!(value_at(&schedule, 0), 1.0);
        assert_eq!(value_at(&schedule, 2), 1.0);
        assert_eq!(value_at(&schedule, 3), 0.8125);
        assert_eq!(value_at(&schedule, 4), 0.625);
        assert_eq!(value_at(&schedule, 6), 0.25);
        assert_eq!(value_at(&schedule, 100), 0.25);
    }

    #[test]
    fn test_cosine_decay_schedule_new() {
        let schedule = CosineDecaySchedule::new(2.0, 10).unwrap();
        assert_eq!(schedule.initial_value(), 2.0);
        assert_eq!(schedule.decay_steps(), 10);
        assert_eq!(schedule.alpha(), 0.0);
        assert!(matches!(
            CosineDecaySchedule::new(f64::INFINITY, 10),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`initial_value` must be finite but got `inf`",
        ));
        assert!(matches!(
            CosineDecaySchedule::new(2.0, 0),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`decay_steps` must be positive but got `0`",
        ));
    }

    #[test]
    fn test_cosine_decay_schedule_with_alpha() {
        assert_eq!(CosineDecaySchedule::new(2.0, 10).unwrap().with_alpha(0.25).unwrap().alpha(), 0.25);
        assert!(matches!(
            CosineDecaySchedule::new(2.0, 10).unwrap().with_alpha(f64::NAN),
            Err(OptimizerError::InvalidConfiguration { message }) if message == "`alpha` must be finite but got `NaN`",
        ));
    }

    #[test]
    fn test_cosine_decay_schedule_value() {
        // The expected values repeat the `f32` operations of the schedule in the same order.
        let schedule = CosineDecaySchedule::new(2.0, 10).unwrap().with_alpha(0.25).unwrap();
        let pi = PI as f32;
        assert_eq!(value_at(&schedule, 0), 2.0 * (0.75 * (0.5 * (1.0 + (pi * 0.0 / 10.0).cos())) + 0.25));
        assert_eq!(value_at(&schedule, 0), 2.0);
        assert_eq!(value_at(&schedule, 5), 2.0 * (0.75 * (0.5 * (1.0 + (pi * 5.0 / 10.0).cos())) + 0.25));
        assert_eq!(value_at(&schedule, 10), 2.0 * (0.75 * (0.5 * (1.0 + (pi * 10.0 / 10.0).cos())) + 0.25));
        assert_eq!(value_at(&schedule, 10), 0.5);
        assert_eq!(value_at(&schedule, 20), 0.5);
    }

    #[test]
    fn test_warmup_cosine_decay_schedule_new() {
        let schedule = WarmupCosineDecaySchedule::new(0.0, 1.0, 4, 14).unwrap();
        assert_eq!(schedule.initial_value(), 0.0);
        assert_eq!(schedule.peak_value(), 1.0);
        assert_eq!(schedule.warmup_steps(), 4);
        assert_eq!(schedule.decay_steps(), 14);
        assert_eq!(schedule.end_value(), 0.0);
        assert!(matches!(
            WarmupCosineDecaySchedule::new(0.0, f64::NAN, 4, 14),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`peak_value` must be finite but got `NaN`",
        ));
        assert!(matches!(
            WarmupCosineDecaySchedule::new(0.0, 1.0, 4, 4),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`decay_steps` must be greater than `warmup_steps` but got `4` and `4`",
        ));
    }

    #[test]
    fn test_warmup_cosine_decay_schedule_with_end_value() {
        let schedule = WarmupCosineDecaySchedule::new(0.0, 1.0, 4, 14).unwrap().with_end_value(0.5).unwrap();
        assert_eq!(schedule.end_value(), 0.5);
        assert!(matches!(
            WarmupCosineDecaySchedule::new(0.0, 1.0, 4, 14).unwrap().with_end_value(f64::NEG_INFINITY),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`end_value` must be finite but got `-inf`",
        ));
    }

    #[test]
    fn test_warmup_cosine_decay_schedule_value() {
        // The warmup rises linearly to the peak value, and the decay then reaches the end value after the remaining
        // ten steps. The expected decay values repeat the `f32` operations of the schedule in the same order.
        let schedule = WarmupCosineDecaySchedule::new(0.0, 2.0, 4, 14).unwrap().with_end_value(0.5).unwrap();
        let pi = PI as f32;
        assert_eq!(value_at(&schedule, 0), 0.0);
        assert_eq!(value_at(&schedule, 1), 0.5);
        assert_eq!(value_at(&schedule, 3), 1.5);
        assert_eq!(value_at(&schedule, 4), 2.0);
        assert_eq!(value_at(&schedule, 9), 2.0 * (0.75 * (0.5 * (1.0 + (pi * 5.0 / 10.0).cos())) + 0.25));
        assert_eq!(value_at(&schedule, 14), 0.5);
        assert_eq!(value_at(&schedule, 100), 0.5);
    }

    #[test]
    fn test_warmup_cosine_decay_schedule_value_without_warmup() {
        let schedule = WarmupCosineDecaySchedule::new(0.0, 2.0, 0, 10).unwrap();
        assert_eq!(value_at(&schedule, 0), 2.0);
        assert_eq!(value_at(&schedule, 10), 0.0);
    }

    #[test]
    fn test_cosine_decay_schedule_with_exponent() {
        assert_eq!(CosineDecaySchedule::new(2.0, 10).unwrap().exponent(), 1.0);
        assert_eq!(CosineDecaySchedule::new(2.0, 10).unwrap().with_exponent(2.0).unwrap().exponent(), 2.0);
        assert!(matches!(
            CosineDecaySchedule::new(2.0, 10).unwrap().with_exponent(f64::INFINITY),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`exponent` must be finite but got `inf`",
        ));

        // The cosine factor `0.5 · (1 + cos(π · 5 / 10))` is squared before it is scaled.
        let schedule = CosineDecaySchedule::new(2.0, 10).unwrap().with_exponent(2.0).unwrap();
        let pi = PI as f32;
        assert_eq!(value_at(&schedule, 0), 2.0);
        assert_eq!(value_at(&schedule, 5), 2.0 * (0.5f32 * (1.0 + (pi * 5.0 / 10.0).cos())).powf(2.0));
    }

    #[test]
    fn test_warmup_cosine_decay_schedule_with_exponent() {
        let schedule = WarmupCosineDecaySchedule::new(0.0, 2.0, 4, 14).unwrap().with_exponent(2.0).unwrap();
        assert_eq!(schedule.exponent(), 2.0);
        let pi = PI as f32;
        assert_eq!(value_at(&schedule, 4), 2.0);
        assert_eq!(value_at(&schedule, 9), 2.0 * (0.5f32 * (1.0 + (pi * 5.0 / 10.0).cos())).powf(2.0));
        assert!(matches!(
            schedule.with_exponent(f64::NAN),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`exponent` must be finite but got `NaN`",
        ));
    }

    #[test]
    fn test_polynomial_schedule_new() {
        let schedule = PolynomialSchedule::new(1.0, 0.0, 2.0, 4).unwrap().with_transition_begin(1);
        assert_eq!(schedule.initial_value(), 1.0);
        assert_eq!(schedule.end_value(), 0.0);
        assert_eq!(schedule.power(), 2.0);
        assert_eq!(schedule.transition_steps(), 4);
        assert_eq!(schedule.transition_begin(), 1);
        assert!(matches!(
            PolynomialSchedule::new(1.0, 0.0, f64::NAN, 4),
            Err(OptimizerError::InvalidConfiguration { message }) if message == "`power` must be finite but got `NaN`",
        ));
        assert!(matches!(
            PolynomialSchedule::new(1.0, 0.0, 2.0, 0),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`transition_steps` must be positive but got `0`",
        ));
    }

    #[test]
    fn test_polynomial_schedule_value() {
        // With `b = 1` and `T = 4`, the fraction at step `t` is `1 - (t - 1) / 4`, so the values at steps `2` and `3`
        // are `0.75² = 0.5625` and `0.5² = 0.25`.
        let schedule = PolynomialSchedule::new(1.0, 0.0, 2.0, 4).unwrap().with_transition_begin(1);
        assert_eq!(value_at(&schedule, 0), 1.0);
        assert_eq!(value_at(&schedule, 1), 1.0);
        assert_eq!(value_at(&schedule, 2), 0.5625);
        assert_eq!(value_at(&schedule, 3), 0.25);
        assert_eq!(value_at(&schedule, 5), 0.0);
        assert_eq!(value_at(&schedule, 9), 0.0);
    }

    #[test]
    fn test_exponential_decay_schedule_new() {
        let schedule = ExponentialDecaySchedule::new(2.0, 2, 0.5).unwrap();
        assert_eq!(schedule.initial_value(), 2.0);
        assert_eq!(schedule.transition_steps(), 2);
        assert_eq!(schedule.decay_rate(), 0.5);
        assert_eq!(schedule.transition_begin(), 0);
        assert!(!schedule.staircase());
        assert_eq!(schedule.end_value(), None);
        assert!(matches!(
            ExponentialDecaySchedule::new(2.0, 2, 0.0),
            Err(OptimizerError::InvalidConfiguration { message }) if message == "`decay_rate` must not be zero",
        ));
        assert!(matches!(
            ExponentialDecaySchedule::new(2.0, 0, 0.5),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`transition_steps` must be positive but got `0`",
        ));
    }

    #[test]
    fn test_exponential_decay_schedule_with_transition_begin() {
        assert_eq!(ExponentialDecaySchedule::new(2.0, 2, 0.5).unwrap().with_transition_begin(3).transition_begin(), 3);
    }

    #[test]
    fn test_exponential_decay_schedule_with_staircase() {
        assert!(ExponentialDecaySchedule::new(2.0, 2, 0.5).unwrap().with_staircase(true).staircase());
    }

    #[test]
    fn test_exponential_decay_schedule_with_end_value() {
        let schedule = ExponentialDecaySchedule::new(2.0, 2, 0.5).unwrap().with_end_value(0.75).unwrap();
        assert_eq!(schedule.end_value(), Some(0.75));
        assert!(matches!(
            schedule.with_end_value(f64::NAN),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`end_value` must be finite but got `NaN`",
        ));
    }

    #[test]
    fn test_exponential_decay_schedule_value() {
        // With `b = 1` and `T = 2`, the exponent at step `t` is `(t - 1) / 2`, so the value at step `2` is
        // `2 · 0.5^0.5` and the value at step `3` is `2 · 0.5 = 1`.
        let schedule = ExponentialDecaySchedule::new(2.0, 2, 0.5).unwrap().with_transition_begin(1);
        assert_eq!(value_at(&schedule, 0), 2.0);
        assert_eq!(value_at(&schedule, 1), 2.0);
        assert_eq!(value_at(&schedule, 2), 2.0 * 0.5f32.powf(0.5));
        assert_eq!(value_at(&schedule, 3), 1.0);

        // A staircase rounds the exponent down, so the value at step `2` is still `2` and that at step `4` is `1`.
        let staircase = schedule.clone().with_staircase(true);
        assert_eq!(value_at(&staircase, 2), 2.0);
        assert_eq!(value_at(&staircase, 4), 1.0);

        // The end value is a lower bound for decay rates below `1`: the value at step `5` would be `2 · 0.5² = 0.5`.
        let bounded = schedule.with_end_value(0.75).unwrap();
        assert_eq!(value_at(&bounded, 3), 1.0);
        assert_eq!(value_at(&bounded, 5), 0.75);

        // And it is an upper bound for decay rates above `1`: the value at step `2` would be `1 · 2² = 4`.
        let growing = ExponentialDecaySchedule::new(1.0, 1, 2.0).unwrap().with_end_value(3.0).unwrap();
        assert_eq!(value_at(&growing, 1), 2.0);
        assert_eq!(value_at(&growing, 2), 3.0);
    }

    #[test]
    fn test_piecewise_constant_schedule_new() {
        let boundaries_and_scales = BTreeMap::from([(2, 0.5), (4, 0.1)]);
        let schedule = PiecewiseConstantSchedule::new(1.0, boundaries_and_scales.clone()).unwrap();
        assert_eq!(schedule.initial_value(), 1.0);
        assert_eq!(schedule.boundaries_and_scales(), &boundaries_and_scales);
        assert!(matches!(
            PiecewiseConstantSchedule::new(1.0, BTreeMap::from([(2, -0.5)])),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "the scale at boundary `2` must be finite and non-negative but got `-0.5`",
        ));
    }

    #[test]
    fn test_piecewise_constant_schedule_value() {
        // The value is `1` before step `2`, `1 · 0.5` from step `2` on, and `(1 · 0.1) · 0.5` from step `4` on.
        let schedule = PiecewiseConstantSchedule::new(1.0, BTreeMap::from([(2, 0.5), (4, 0.1)])).unwrap();
        assert_eq!(value_at(&schedule, 0), 1.0);
        assert_eq!(value_at(&schedule, 1), 1.0);
        assert_eq!(value_at(&schedule, 2), 0.5);
        assert_eq!(value_at(&schedule, 3), 0.5);
        assert_eq!(value_at(&schedule, 4), 0.1f32 * 0.5);
        assert_eq!(value_at(&schedule, 100), 0.1f32 * 0.5);
        assert_eq!(value_at(&PiecewiseConstantSchedule::new(3.0, BTreeMap::new()).unwrap(), 7), 3.0);
    }

    #[test]
    fn test_piecewise_interpolate_schedule_new() {
        let boundaries_and_scales = BTreeMap::from([(4, 0.1)]);
        let schedule =
            PiecewiseInterpolateSchedule::new(Interpolation::Cosine, 1.0, boundaries_and_scales.clone()).unwrap();
        assert_eq!(schedule.interpolation(), Interpolation::Cosine);
        assert_eq!(schedule.initial_value(), 1.0);
        assert_eq!(schedule.boundaries_and_scales(), &boundaries_and_scales);
        assert!(matches!(
            PiecewiseInterpolateSchedule::new(Interpolation::Linear, 1.0, BTreeMap::from([(4, f64::NAN)])),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "the scale at boundary `4` must be finite and non-negative but got `NaN`",
        ));
    }

    #[test]
    fn test_piecewise_interpolate_schedule_value() {
        // This is the example of the Optax documentation, whose values at the boundaries are `1`, `0.1`, and
        // `0.1 · 0.01`, so the linear interpolation is `0.55` at step `2` and `0.0505` at step `6`.
        let boundaries_and_scales = BTreeMap::from([(4, 0.1), (8, 0.01)]);
        let schedule =
            PiecewiseInterpolateSchedule::new(Interpolation::Linear, 1.0, boundaries_and_scales.clone()).unwrap();
        let (first, second, third) = (1.0f32, 0.1f32, 0.1f32 * 0.01f32);
        assert_eq!(value_at(&schedule, 0), 1.0);
        assert_eq!(value_at(&schedule, 2), (second - first) * 0.5 + first);
        assert_eq!(value_at(&schedule, 4), second);
        assert_eq!(value_at(&schedule, 6), (third - second) * 0.5 + second);
        assert_eq!(value_at(&schedule, 8), third);
        assert_eq!(value_at(&schedule, 10), third);

        // The cosine interpolation is `end + (start - end) / 2 · (cos(π · fraction) + 1)`.
        let schedule = PiecewiseInterpolateSchedule::new(Interpolation::Cosine, 1.0, boundaries_and_scales).unwrap();
        let pi = PI as f32;
        assert_eq!(value_at(&schedule, 0), 1.0);
        assert_eq!(value_at(&schedule, 1), second + (first - second) / 2.0 * ((pi * 0.25).cos() + 1.0));
        assert_eq!(value_at(&schedule, 6), third + (second - third) / 2.0 * ((pi * 0.5).cos() + 1.0));
        assert_eq!(value_at(&schedule, 9), third);
    }

    #[test]
    fn test_linear_one_cycle_schedule_new() {
        let schedule = LinearOneCycleSchedule::new(10, 1.0).unwrap();
        assert_eq!(schedule.transition_steps(), 10);
        assert_eq!(schedule.peak_value(), 1.0);
        assert_eq!(schedule.start_fraction(), 0.3);
        assert_eq!(schedule.final_fraction(), 0.85);
        assert_eq!(schedule.division_factor(), 25.0);
        assert_eq!(schedule.final_division_factor(), 1e4);
        assert!(matches!(
            LinearOneCycleSchedule::new(0, 1.0),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`transition_steps` must be positive but got `0`",
        ));
    }

    #[test]
    fn test_linear_one_cycle_schedule_with_fractions_and_factors() {
        let schedule = LinearOneCycleSchedule::new(10, 1.0)
            .unwrap()
            .with_start_fraction(0.2)
            .unwrap()
            .with_final_fraction(0.6)
            .unwrap()
            .with_division_factor(10.0)
            .unwrap()
            .with_final_division_factor(100.0)
            .unwrap();
        assert_eq!(schedule.start_fraction(), 0.2);
        assert_eq!(schedule.final_fraction(), 0.6);
        assert_eq!(schedule.division_factor(), 10.0);
        assert_eq!(schedule.final_division_factor(), 100.0);
        assert!(matches!(
            schedule.clone().with_start_fraction(1.5),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`start_fraction` must be in `[0, 1]` but got `1.5`",
        ));
        assert!(matches!(
            schedule.with_division_factor(0.0),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`division_factor` must be positive and finite but got `0`",
        ));
    }

    #[test]
    fn test_linear_one_cycle_schedule_value() {
        // With `T = 10`, the boundaries are `3`, `8`, and `10`, and the values there are `1 / 25 · 25`,
        // `(1 / 25 · 25) · (1 / 25)`, and that times `1e-4`.
        let schedule = LinearOneCycleSchedule::new(10, 1.0).unwrap();
        let initial = (1.0f64 / 25.0) as f32;
        let peak = initial * 25.0;
        let low = peak * (1.0f64 / 25.0) as f32;
        let last = low * 1e-4f32;
        assert_eq!(value_at(&schedule, 0), initial);
        assert_eq!(value_at(&schedule, 3), peak);
        assert_eq!(value_at(&schedule, 4), (low - peak) * 0.2 + peak);
        assert_eq!(value_at(&schedule, 8), low);
        assert_eq!(value_at(&schedule, 9), (last - low) * 0.5 + low);
        assert_eq!(value_at(&schedule, 12), last);
    }

    #[test]
    fn test_cosine_one_cycle_schedule_new() {
        let schedule = CosineOneCycleSchedule::new(10, 1.0).unwrap();
        assert_eq!(schedule.transition_steps(), 10);
        assert_eq!(schedule.peak_value(), 1.0);
        assert_eq!(schedule.start_fraction(), 0.3);
        assert_eq!(schedule.division_factor(), 25.0);
        assert_eq!(schedule.final_division_factor(), 1e4);
        let schedule = schedule
            .with_start_fraction(0.5)
            .unwrap()
            .with_division_factor(5.0)
            .unwrap()
            .with_final_division_factor(2.0)
            .unwrap();
        assert_eq!(schedule.start_fraction(), 0.5);
        assert_eq!(schedule.division_factor(), 5.0);
        assert_eq!(schedule.final_division_factor(), 2.0);
        assert!(matches!(
            schedule.with_final_division_factor(f64::INFINITY),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`final_division_factor` must be positive and finite but got `inf`",
        ));
    }

    #[test]
    fn test_cosine_one_cycle_schedule_value() {
        // With `T = 10`, `η = 1`, `d = 5`, and `D = 2`, the values at the boundaries `5` and `10` are `0.2 · 5 = 1` and
        // `1 · (1 / 10) = 0.1`, and the cosine interpolation is halfway at steps `2.5` and `7.5`.
        let schedule = CosineOneCycleSchedule::new(10, 1.0)
            .unwrap()
            .with_start_fraction(0.5)
            .unwrap()
            .with_division_factor(5.0)
            .unwrap()
            .with_final_division_factor(2.0)
            .unwrap();
        let pi = PI as f32;
        let (initial, peak) = (0.2f32, 0.2f32 * 5.0);
        let last = peak * (1.0f64 / 10.0) as f32;
        // The cosine interpolation computes the initial value as `1 + (0.2 - 1) / 2 · 2`, which rounds in `f32`.
        assert_eq!(value_at(&schedule, 0), peak + (initial - peak) / 2.0 * ((pi * 0.0).cos() + 1.0));
        assert_eq!(value_at(&schedule, 1), peak + (initial - peak) / 2.0 * ((pi * 0.2).cos() + 1.0));
        assert_eq!(value_at(&schedule, 5), peak);
        assert_eq!(value_at(&schedule, 7), last + (peak - last) / 2.0 * ((pi * 0.4).cos() + 1.0));
        assert_eq!(value_at(&schedule, 10), last);
    }

    #[test]
    fn test_warmup_exponential_decay_schedule_new() {
        let schedule = WarmupExponentialDecaySchedule::new(0.0, 1.0, 2, 1, 0.5)
            .unwrap()
            .with_transition_begin(1)
            .with_staircase(true)
            .with_end_value(0.25)
            .unwrap();
        assert_eq!(schedule.initial_value(), 0.0);
        assert_eq!(schedule.warmup_steps(), 2);
        assert_eq!(
            schedule.decay(),
            &ExponentialDecaySchedule::new(1.0, 1, 0.5)
                .unwrap()
                .with_transition_begin(1)
                .with_staircase(true)
                .with_end_value(0.25)
                .unwrap(),
        );
        assert!(matches!(
            WarmupExponentialDecaySchedule::new(f64::NAN, 1.0, 2, 1, 0.5),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`initial_value` must be finite but got `NaN`",
        ));
    }

    #[test]
    fn test_warmup_exponential_decay_schedule_value() {
        // The warmup reaches `1` at step `2`, after which the decay halves the value every step.
        let schedule = WarmupExponentialDecaySchedule::new(0.0, 1.0, 2, 1, 0.5).unwrap();
        assert_eq!(value_at(&schedule, 0), 0.0);
        assert_eq!(value_at(&schedule, 1), 0.5);
        assert_eq!(value_at(&schedule, 2), 1.0);
        assert_eq!(value_at(&schedule, 3), 0.5);
        assert_eq!(value_at(&schedule, 4), 0.25);
        assert_eq!(value_at(&WarmupExponentialDecaySchedule::new(0.0, 1.0, 0, 1, 0.5).unwrap(), 1), 0.5);
    }

    #[test]
    fn test_sgdr_schedule_new() {
        let cycle = WarmupCosineDecaySchedule::new(0.0, 1.0, 0, 2).unwrap();
        assert_eq!(SgdrSchedule::new(vec![cycle.clone()]).unwrap().cycles(), &[cycle]);
        assert!(matches!(
            SgdrSchedule::new(Vec::new()),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`cycles` must contain at least one schedule",
        ));
    }

    #[test]
    fn test_sgdr_schedule_value() {
        // Each cycle decays over two steps, and the second one restarts at step `2` from `0.5`.
        let schedule = SgdrSchedule::new(vec![
            WarmupCosineDecaySchedule::new(0.0, 1.0, 0, 2).unwrap(),
            WarmupCosineDecaySchedule::new(0.0, 0.5, 0, 2).unwrap(),
        ])
        .unwrap();
        let half_decay = 0.5f32 * (1.0 + (PI as f32 * 1.0 / 2.0).cos());
        assert_eq!(value_at(&schedule, 0), 1.0);
        assert_eq!(value_at(&schedule, 1), half_decay);
        assert_eq!(value_at(&schedule, 2), 0.5);
        assert_eq!(value_at(&schedule, 3), 0.5 * half_decay);
        assert_eq!(value_at(&schedule, 5), 0.0);
    }

    #[test]
    fn test_joined_schedule_new() {
        let schedule = JoinedSchedule::new(1.0f64, 2, 0.5f64);
        assert_eq!(schedule.first(), &1.0);
        assert_eq!(schedule.boundary(), 2);
        assert_eq!(schedule.second(), &0.5);
    }

    #[test]
    fn test_joined_schedule_then() {
        // The second schedule restarts its step count at the boundary, so the linear warmup is followed by constants.
        let schedule = JoinedSchedule::new(LinearSchedule::new(0.0, 1.0, 2).unwrap(), 2, 0.5f64)
            .then(4, LinearSchedule::new(0.25, 0.0, 4).unwrap());
        assert_eq!(schedule.boundary(), 4);
        assert_eq!(value_at(&schedule, 0), 0.0);
        assert_eq!(value_at(&schedule, 1), 0.5);
        assert_eq!(value_at(&schedule, 2), 0.5);
        assert_eq!(value_at(&schedule, 3), 0.5);
        assert_eq!(value_at(&schedule, 4), 0.25);
        assert_eq!(value_at(&schedule, 6), 0.125);
    }

    /// Returns the values of `schedule` at each step of `steps`, for comparison with the values of Optax.
    fn values_at<Sc: Schedule>(schedule: &Sc, steps: impl IntoIterator<Item = i32>) -> Vec<f32> {
        steps.into_iter().map(|step| value_at(schedule, step)).collect()
    }

    #[test]
    fn test_f64_optax_parity() {
        // Optax 9ecde04b: `optax.constant_schedule(0.5)`.
        const CONSTANT: [f32; 3] = [0.5, 0.5, 0.5];
        assert_parity(&[values_at(&0.5f64, [0, 1, 100])], &[CONSTANT]);
    }

    #[test]
    fn test_linear_schedule_optax_parity() {
        // Optax 9ecde04b: `optax.linear_schedule(1.0, 0.25, 4, transition_begin=2)`.
        const LINEAR: [f32; 9] = [1.0, 1.0, 1.0, 0.8125, 0.625, 0.4375, 0.25, 0.25, 0.25];
        assert_parity(
            &[values_at(&LinearSchedule::new(1.0, 0.25, 4).unwrap().with_transition_begin(2), 0..9)],
            &[LINEAR],
        );
    }

    #[test]
    fn test_cosine_decay_schedule_optax_parity() {
        // Optax 9ecde04b: `optax.cosine_decay_schedule(1.0, 5, alpha=0.1)`.
        const COSINE_DECAY: [f32; 8] = [1.0, 0.9140576, 0.68905765, 0.41094232, 0.18594232, 0.1, 0.1, 0.1];
        assert_parity(
            &[values_at(&CosineDecaySchedule::new(1.0, 5).unwrap().with_alpha(0.1).unwrap(), 0..8)],
            &[COSINE_DECAY],
        );

        // Optax 9ecde04b: `optax.cosine_decay_schedule(1.0, 5, alpha=0.1, exponent=2.0)`.
        const COSINE_DECAY_EXPONENT: [f32; 8] = [1.0, 0.836322, 0.4855432, 0.20742792, 0.10820676, 0.1, 0.1, 0.1];
        assert_parity(
            &[values_at(
                &CosineDecaySchedule::new(1.0, 5).unwrap().with_alpha(0.1).unwrap().with_exponent(2.0).unwrap(),
                0..8,
            )],
            &[COSINE_DECAY_EXPONENT],
        );
    }

    #[test]
    fn test_warmup_cosine_decay_schedule_optax_parity() {
        // Optax 9ecde04b: `optax.warmup_cosine_decay_schedule(0.1, 1.0, 3, 8)`.
        const WARMUP_COSINE_DECAY: [f32; 11] =
            [0.100000024, 0.40000004, 0.70000005, 1.0, 0.9045085, 0.6545085, 0.34549147, 0.09549147, 0.0, 0.0, 0.0];
        assert_parity(
            &[values_at(&WarmupCosineDecaySchedule::new(0.1, 1.0, 3, 8).unwrap(), 0..11)],
            &[WARMUP_COSINE_DECAY],
        );

        // Optax 9ecde04b: `optax.warmup_cosine_decay_schedule(0.1, 1.0, 3, 8, end_value=0.05, exponent=1.5)`.
        const WARMUP_COSINE_DECAY_EXPONENT: [f32; 11] =
            [0.100000024, 0.40000004, 0.70000005, 1.0, 0.8672267, 0.55303305, 0.24292104, 0.07803306, 0.05, 0.05, 0.05];
        assert_parity(
            &[values_at(
                &WarmupCosineDecaySchedule::new(0.1, 1.0, 3, 8)
                    .unwrap()
                    .with_end_value(0.05)
                    .unwrap()
                    .with_exponent(1.5)
                    .unwrap(),
                0..11,
            )],
            &[WARMUP_COSINE_DECAY_EXPONENT],
        );
    }

    #[test]
    fn test_polynomial_schedule_optax_parity() {
        // Optax 9ecde04b: `optax.polynomial_schedule(1.0, 0.1, 2.0, 4, transition_begin=1)`.
        const POLYNOMIAL: [f32; 8] = [1.0, 1.0, 0.60625, 0.325, 0.15625, 0.1, 0.1, 0.1];
        assert_parity(
            &[values_at(&PolynomialSchedule::new(1.0, 0.1, 2.0, 4).unwrap().with_transition_begin(1), 0..8)],
            &[POLYNOMIAL],
        );
    }

    #[test]
    fn test_exponential_decay_schedule_optax_parity() {
        // Optax 9ecde04b: `optax.exponential_decay(1.0, 2, 0.5, transition_begin=1)`.
        const EXPONENTIAL_DECAY: [f32; 10] =
            [1.0, 1.0, 0.70710677, 0.5, 0.35355338, 0.25, 0.17677669, 0.125, 0.088388346, 0.0625];
        assert_parity(
            &[values_at(&ExponentialDecaySchedule::new(1.0, 2, 0.5).unwrap().with_transition_begin(1), 0..10)],
            &[EXPONENTIAL_DECAY],
        );

        // Optax 9ecde04b: `optax.exponential_decay(1.0, 2, 0.5, transition_begin=1, staircase=True, end_value=0.2)`.
        const EXPONENTIAL_DECAY_STAIRCASE: [f32; 10] = [1.0, 1.0, 1.0, 0.5, 0.5, 0.25, 0.25, 0.2, 0.2, 0.2];
        assert_parity(
            &[values_at(
                &ExponentialDecaySchedule::new(1.0, 2, 0.5)
                    .unwrap()
                    .with_transition_begin(1)
                    .with_staircase(true)
                    .with_end_value(0.2)
                    .unwrap(),
                0..10,
            )],
            &[EXPONENTIAL_DECAY_STAIRCASE],
        );

        // Optax 9ecde04b: `optax.exponential_decay(1.0, 2, 1.5, end_value=2.0)`.
        const EXPONENTIAL_GROWTH: [f32; 10] = [1.0, 1.2247449, 1.5, 1.8371173, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0];
        assert_parity(
            &[values_at(&ExponentialDecaySchedule::new(1.0, 2, 1.5).unwrap().with_end_value(2.0).unwrap(), 0..10)],
            &[EXPONENTIAL_GROWTH],
        );
    }

    #[test]
    fn test_piecewise_constant_schedule_optax_parity() {
        let boundaries_and_scales = BTreeMap::from([(2, 0.5), (5, 0.1), (7, 3.0)]);
        // Optax 9ecde04b: `optax.piecewise_constant_schedule(1.0, {2: 0.5, 5: 0.1, 7: 3.0})`.
        const PIECEWISE_CONSTANT: [f32; 10] = [1.0, 1.0, 0.5, 0.5, 0.5, 0.05, 0.05, 0.15, 0.15, 0.15];
        assert_parity(
            &[values_at(&PiecewiseConstantSchedule::new(1.0, boundaries_and_scales.clone()).unwrap(), 0..10)],
            &[PIECEWISE_CONSTANT],
        );
    }

    #[test]
    fn test_piecewise_interpolate_schedule_optax_parity() {
        let boundaries_and_scales = BTreeMap::from([(2, 0.5), (5, 0.1), (7, 3.0)]);
        // Optax 9ecde04b: `optax.piecewise_interpolate_schedule('linear', 1.0, {2: 0.5, 5: 0.1, 7: 3.0})`.
        const PIECEWISE_LINEAR: [f32; 10] = [1.0, 0.75, 0.5, 0.35, 0.19999999, 0.05, 0.10000001, 0.15, 0.15, 0.15];
        assert_parity(
            &[values_at(
                &PiecewiseInterpolateSchedule::new(Interpolation::Linear, 1.0, boundaries_and_scales.clone()).unwrap(),
                0..10,
            )],
            &[PIECEWISE_LINEAR],
        );

        // Optax 9ecde04b: `optax.piecewise_interpolate_schedule('cosine', 1.0, {2: 0.5, 5: 0.1, 7: 3.0})`.
        const PIECEWISE_COSINE: [f32; 10] =
            [1.0, 0.75, 0.5, 0.3875, 0.16249998, 0.049999997, 0.10000001, 0.15, 0.15, 0.15];
        assert_parity(
            &[values_at(
                &PiecewiseInterpolateSchedule::new(Interpolation::Cosine, 1.0, boundaries_and_scales).unwrap(),
                0..10,
            )],
            &[PIECEWISE_COSINE],
        );
    }

    #[test]
    fn test_linear_one_cycle_schedule_optax_parity() {
        // Optax 9ecde04b: `optax.linear_onecycle_schedule(10, 1.0)`.
        const LINEAR_ONE_CYCLE: [f32; 13] =
            [0.04, 0.35999998, 0.68, 1.0, 0.80799997, 0.616, 0.42399997, 0.232, 0.04, 0.020002, 4.0e-6, 4.0e-6, 4.0e-6];
        assert_parity(&[values_at(&LinearOneCycleSchedule::new(10, 1.0).unwrap(), 0..13)], &[LINEAR_ONE_CYCLE]);

        // Optax 9ecde04b: `optax.linear_onecycle_schedule(10, 1.0, pct_start=0.2, pct_final=0.7, div_factor=10.0,
        // final_div_factor=100.0)`.
        const LINEAR_ONE_CYCLE_CUSTOM: [f32; 13] =
            [0.1, 0.55, 1.0, 0.82, 0.64, 0.45999998, 0.28000003, 0.1, 0.067, 0.034, 0.001, 0.001, 0.001];
        assert_parity(
            &[values_at(
                &LinearOneCycleSchedule::new(10, 1.0)
                    .unwrap()
                    .with_start_fraction(0.2)
                    .unwrap()
                    .with_final_fraction(0.7)
                    .unwrap()
                    .with_division_factor(10.0)
                    .unwrap()
                    .with_final_division_factor(100.0)
                    .unwrap(),
                0..13,
            )],
            &[LINEAR_ONE_CYCLE_CUSTOM],
        );
    }

    #[test]
    fn test_cosine_one_cycle_schedule_optax_parity() {
        // Optax 9ecde04b: `optax.cosine_onecycle_schedule(10, 1.0)`.
        const COSINE_ONE_CYCLE: [f32; 13] = [
            0.04000002, 0.28000003, 0.76000005, 1.0, 0.9504846, 0.81174564, 0.61126196, 0.3887419, 0.18825826,
            0.04951935, 4.0e-6, 4.0e-6, 4.0e-6,
        ];
        assert_parity(&[values_at(&CosineOneCycleSchedule::new(10, 1.0).unwrap(), 0..13)], &[COSINE_ONE_CYCLE]);

        // Optax 9ecde04b: `optax.cosine_onecycle_schedule(10, 1.0, pct_start=0.4, div_factor=5.0,
        // final_div_factor=50.0)`.
        const COSINE_ONE_CYCLE_CUSTOM: [f32; 13] = [
            0.19999999,
            0.31715727,
            0.6,
            0.8828427,
            1.0,
            0.93328065,
            0.751,
            0.502,
            0.25299996,
            0.070719354,
            0.004,
            0.004,
            0.004,
        ];
        assert_parity(
            &[values_at(
                &CosineOneCycleSchedule::new(10, 1.0)
                    .unwrap()
                    .with_start_fraction(0.4)
                    .unwrap()
                    .with_division_factor(5.0)
                    .unwrap()
                    .with_final_division_factor(50.0)
                    .unwrap(),
                0..13,
            )],
            &[COSINE_ONE_CYCLE_CUSTOM],
        );
    }

    #[test]
    fn test_warmup_exponential_decay_schedule_optax_parity() {
        // Optax 9ecde04b: `optax.warmup_exponential_decay_schedule(0.1, 1.0, 3, 2, 0.5)`.
        const WARMUP_EXPONENTIAL_DECAY: [f32; 13] = [
            0.100000024,
            0.40000004,
            0.70000005,
            1.0,
            0.70710677,
            0.5,
            0.35355338,
            0.25,
            0.17677669,
            0.125,
            0.088388346,
            0.0625,
            0.044194173,
        ];
        assert_parity(
            &[values_at(&WarmupExponentialDecaySchedule::new(0.1, 1.0, 3, 2, 0.5).unwrap(), 0..13)],
            &[WARMUP_EXPONENTIAL_DECAY],
        );

        // Optax 9ecde04b: `optax.warmup_exponential_decay_schedule(0.1, 1.0, 3, 2, 0.5, transition_begin=1,
        // staircase=True, end_value=0.2)`.
        const WARMUP_EXPONENTIAL_DECAY_STAIRCASE: [f32; 13] =
            [0.100000024, 0.40000004, 0.70000005, 1.0, 1.0, 1.0, 0.5, 0.5, 0.25, 0.25, 0.2, 0.2, 0.2];
        assert_parity(
            &[values_at(
                &WarmupExponentialDecaySchedule::new(0.1, 1.0, 3, 2, 0.5)
                    .unwrap()
                    .with_transition_begin(1)
                    .with_staircase(true)
                    .with_end_value(0.2)
                    .unwrap(),
                0..13,
            )],
            &[WARMUP_EXPONENTIAL_DECAY_STAIRCASE],
        );
    }

    #[test]
    fn test_sgdr_schedule_optax_parity() {
        let cycles = vec![
            WarmupCosineDecaySchedule::new(0.0, 1.0, 1, 4).unwrap(),
            WarmupCosineDecaySchedule::new(0.1, 0.5, 2, 5).unwrap().with_end_value(0.05).unwrap(),
        ];
        // Optax 9ecde04b: `optax.sgdr_schedule([{'init_value': 0.0, 'peak_value': 1.0, 'warmup_steps': 1,
        // 'decay_steps': 4}, {'init_value': 0.1, 'peak_value': 0.5, 'warmup_steps': 2, 'decay_steps': 5, 'end_value':
        // 0.05}])`.
        const SGDR: [f32; 12] =
            [0.0, 1.0, 0.75, 0.24999997, 0.099999994, 0.3, 0.5, 0.3875, 0.16249998, 0.05, 0.05, 0.05];
        assert_parity(&[values_at(&SgdrSchedule::new(cycles).unwrap(), 0..12)], &[SGDR]);
    }

    #[test]
    fn test_joined_schedule_optax_parity() {
        // Optax 9ecde04b: `optax.join_schedules([optax.linear_schedule(0.0, 1.0, 3), optax.cosine_decay_schedule(1.0,
        // 4), optax.constant_schedule(0.01)], [3, 7])`.
        const JOINED: [f32; 10] =
            [0.0, 0.33333337, 0.6666667, 1.0, 0.8535534, 0.49999997, 0.14644662, 0.01, 0.01, 0.01];
        assert_parity(
            &[values_at(
                &JoinedSchedule::new(
                    LinearSchedule::new(0.0, 1.0, 3).unwrap(),
                    3,
                    CosineDecaySchedule::new(1.0, 4).unwrap(),
                )
                .then(7, 0.01f64),
                0..10,
            )],
            &[JOINED],
        );
    }
}
