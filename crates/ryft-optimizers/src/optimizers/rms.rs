//! Optimizers that rescale gradients by running averages or sums of their squares: RMSProp, Adagrad, Adadelta, and
//! NovoGrad.

use ryft_core::{Parameter, ParameterPath, Parameterized};

use crate::macros::parameterized_state;
use crate::{
    AddDecayedWeights, Optimizer, OptimizerError, OptimizerValue, ScaleByAdadelta, ScaleByAdadeltaState,
    ScaleByNovograd, ScaleByNovogradState, ScaleByRms, ScaleByRmsState, ScaleByRss, ScaleByRssState, ScaleBySchedule,
    ScaleByScheduleState, ScaleByStddev, ScaleByStddevState, Schedule, Trace, TraceState, UpdateArguments,
    validate_finite,
};

// TODO(eaplatanios): Review this module.

parameterized_state! {
    /// State of [`RmsProp`] for the parameter structure `S`, which holds the state of its rescaling (in `rms` without
    /// centering or in `stddev` with it, each with one element when it is used and none otherwise), the step count of
    /// its learning rate schedule, and the state of its momentum (one element with momentum and none without).
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct RmsPropState<P, S> with family RmsPropStateFamily {
        /// State of the [`ScaleByRms`] transformation, which has one element without centering and none with it.
        rms: Vec<ScaleByRmsState<P, S>>,

        /// State of the [`ScaleByStddev`] transformation, which has one element with centering and none without it.
        stddev: Vec<ScaleByStddevState<P, S>>,

        /// State of the [`ScaleBySchedule`] transformation that applies the learning rate.
        learning_rate: ScaleByScheduleState<P>,

        /// State of the [`Trace`] transformation of the momentum, which has one element with momentum and none without.
        momentum: Vec<TraceState<P, S>>,
    }
}

impl<P: Parameter, S> RmsPropState<P, S> {
    /// Returns the state of the [`ScaleByRms`] transformation, if the optimizer is not centered.
    #[inline]
    pub fn rms(&self) -> Option<&ScaleByRmsState<P, S>> {
        self.rms.first()
    }

    /// Returns the state of the [`ScaleByStddev`] transformation, if the optimizer is centered.
    #[inline]
    pub fn stddev(&self) -> Option<&ScaleByStddevState<P, S>> {
        self.stddev.first()
    }

    /// Returns the state of the [`ScaleBySchedule`] transformation that applies the learning rate.
    #[inline]
    pub fn learning_rate(&self) -> &ScaleByScheduleState<P> {
        &self.learning_rate
    }

    /// Returns the state of the [`Trace`] transformation of the momentum, if the optimizer has momentum.
    #[inline]
    pub fn momentum(&self) -> Option<&TraceState<P, S>> {
        self.momentum.first()
    }
}

/// RMSProp optimizer of Hinton (2012), which divides the gradients by the root mean square of their recent values, and
/// which is the analogue of
/// [`optax.rmsprop`](https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.rmsprop). Its update is
///
/// ```text
/// vₜ = β · vₜ₋₁ + (1 − β) · gₜ²
/// v̂ₜ = vₜ / (1 − βᵗ)
/// aₜ = gₜ / √(v̂ₜ + ε)
/// τₜ = −ηₜ · aₜ + μ · τₜ₋₁
/// uₜ = τₜ
/// ```
///
/// and, when it is centered as in [Graves (2013)](https://arxiv.org/abs/1308.0850), the rescaling subtracts the squared
/// mean of the gradients from their second moment, as in
///
/// ```text
/// mₜ = β · mₜ₋₁ + (1 − β) · gₜ
/// m̂ₜ = mₜ / (1 − βᵗ)
/// aₜ = gₜ / √(v̂ₜ − m̂ₜ² + ε)
/// ```
///
/// Here `t ≥ 1` counts the updates including the current one, `gₜ` is the gradient at the parameters `θₜ₋₁`, `uₜ` is
/// the update that is added to them (i.e., `θₜ = θₜ₋₁ + uₜ`), and `ηₜ` is the learning rate [`Schedule`] evaluated at
/// the `t − 1` updates applied before the current one. The second moment `vₜ` starts at the initial scale `v₀`, the
/// mean `mₜ` starts at zero, both decay with the rate `β`, `v̂ₜ` and `m̂ₜ` are their estimates, `ε` is added for
/// numerical stability, `aₜ` is the rescaled gradient, and `τₜ` is the momentum trace, which starts at zero and decays
/// with the rate `μ`. All operations are element-wise. The defaults are `β = 0.9`, `ε = 1e-8`, and `v₀ = 0`, without
/// bias correction (i.e., `v̂ₜ = vₜ` and `m̂ₜ = mₜ`), without centering, and without momentum (i.e., `uₜ = −ηₜ · aₜ`,
/// with no trace in the state). [`Self::with_bias_correction`] enables the bias corrections of the blocks,
/// [`Self::with_epsilon_in_sqrt`] with `false` moves `ε` outside the square root (e.g., `aₜ = gₜ / (√v̂ₜ + ε)`, which
/// is the form of PyTorch), [`Self::with_centered`] selects the centered rescaling, [`Self::with_momentum`] sets `μ`,
/// and [`Self::with_nesterov`] replaces the update with `uₜ = −ηₜ · aₜ + μ · τₜ`. Note that the learning rate is
/// applied before the momentum, so a learning rate schedule does not rescale the past updates in `τₜ`.
///
/// It is the chain of [`ScaleByRms`] (or of [`ScaleByStddev`] when it is centered), [`ScaleBySchedule::learning_rate`],
/// and, with momentum, [`Trace`].
///
/// # References
///
///   - Hinton (2012). Overview of mini-batch gradient descent. Lecture 6 of the course Neural Networks for Machine
///     Learning.
///   - Graves (2013). [Generating Sequences With Recurrent Neural Networks](https://arxiv.org/abs/1308.0850).
#[derive(Clone, Debug)]
pub struct RmsProp<Sc: Schedule> {
    /// Transformation that rescales the gradients without centering.
    rms: ScaleByRms,

    /// Transformation that rescales the gradients with centering, which has the same hyperparameters as `rms`.
    stddev: ScaleByStddev,

    /// Whether the gradients are rescaled by their standard deviation rather than their root mean square.
    centered: bool,

    /// Transformation that scales the updates by the negated learning rate.
    learning_rate: ScaleBySchedule<Sc>,

    /// Decay of the momentum, or [`None`] without momentum.
    momentum: Option<f64>,

    /// Whether the momentum is Nesterov momentum.
    nesterov: bool,
}

impl<Sc: Schedule> RmsProp<Sc> {
    /// Creates a new [`RmsProp`] with the learning rate schedule `learning_rate` (e.g., an `f64` for a constant
    /// learning rate) and default values for all other hyperparameters.
    pub fn new(learning_rate: Sc) -> Self {
        Self {
            rms: ScaleByRms::new(),
            stddev: ScaleByStddev::new(),
            centered: false,
            learning_rate: ScaleBySchedule::learning_rate(learning_rate),
            momentum: None,
            nesterov: false,
        }
    }

    /// Returns this optimizer with `decay` as the decay rate of its moving averages.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `decay` is not in `[0, 1)`.
    pub fn with_decay(mut self, decay: f64) -> Result<Self, OptimizerError> {
        self.rms = self.rms.with_decay(decay)?;
        self.stddev = self.stddev.with_decay(decay)?;
        Ok(self)
    }

    /// Returns this optimizer with `epsilon` as the term that is added for numerical stability.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `epsilon` is negative or not finite.
    pub fn with_epsilon(mut self, epsilon: f64) -> Result<Self, OptimizerError> {
        self.rms = self.rms.with_epsilon(epsilon)?;
        self.stddev = self.stddev.with_epsilon(epsilon)?;
        Ok(self)
    }

    /// Returns this optimizer with `initial_scale` as the initial value of its moving averages of the squared
    /// gradients.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `initial_scale` is negative or not finite.
    pub fn with_initial_scale(mut self, initial_scale: f64) -> Result<Self, OptimizerError> {
        self.rms = self.rms.with_initial_scale(initial_scale)?;
        self.stddev = self.stddev.with_initial_scale(initial_scale)?;
        Ok(self)
    }

    /// Returns this optimizer with `ε` added inside the square root when `epsilon_in_sqrt` is `true` and to the square
    /// root otherwise.
    #[inline]
    pub fn with_epsilon_in_sqrt(mut self, epsilon_in_sqrt: bool) -> Self {
        self.rms = self.rms.with_epsilon_in_sqrt(epsilon_in_sqrt);
        self.stddev = self.stddev.with_epsilon_in_sqrt(epsilon_in_sqrt);
        self
    }

    /// Returns this optimizer with the bias correction of its moving averages enabled when `bias_correction` is `true`.
    #[inline]
    pub fn with_bias_correction(mut self, bias_correction: bool) -> Self {
        self.rms = self.rms.with_bias_correction(bias_correction);
        self.stddev = self.stddev.with_bias_correction(bias_correction);
        self
    }

    /// Returns this optimizer centered (i.e., rescaling by the standard deviation of the gradients) when `centered` is
    /// `true`.
    #[inline]
    pub fn with_centered(mut self, centered: bool) -> Self {
        self.centered = centered;
        self
    }

    /// Returns this optimizer with the momentum decay `momentum`.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `momentum` is not finite.
    pub fn with_momentum(mut self, momentum: f64) -> Result<Self, OptimizerError> {
        validate_finite("momentum", momentum)?;
        self.momentum = Some(momentum);
        Ok(self)
    }

    /// Returns this optimizer with Nesterov momentum when `nesterov` is `true`, which has an effect only with momentum.
    #[inline]
    pub fn with_nesterov(mut self, nesterov: bool) -> Self {
        self.nesterov = nesterov;
        self
    }

    /// Returns the [`ScaleByRms`] transformation of this optimizer, which holds the hyperparameters of its rescaling.
    #[inline]
    pub fn rms(&self) -> &ScaleByRms {
        &self.rms
    }

    /// Returns whether this optimizer is centered.
    #[inline]
    pub fn centered(&self) -> bool {
        self.centered
    }

    /// Returns the [`Trace`] transformation of the momentum of this optimizer, if it has momentum.
    pub fn momentum(&self) -> Option<Trace> {
        // The momentum was validated by `Self::with_momentum`, so the trace can be built directly.
        self.momentum.map(|momentum| Trace::new(momentum).unwrap().with_nesterov(self.nesterov))
    }

    /// Returns the learning rate [`Schedule`] of this optimizer.
    #[inline]
    pub fn learning_rate(&self) -> &Sc {
        self.learning_rate.schedule()
    }
}

impl<P: OptimizerValue, Sc: Schedule> Optimizer<P> for RmsProp<Sc> {
    type State<S: Parameterized<P>> = RmsPropState<P, S>;

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        let (rms, stddev) = if self.centered {
            (Vec::new(), vec![self.stddev.initialize(parameters)?])
        } else {
            (vec![self.rms.initialize(parameters)?], Vec::new())
        };
        let learning_rate = self.learning_rate.initialize(parameters)?;
        let momentum =
            self.momentum().map(|trace| trace.initialize(parameters)).into_iter().collect::<Result<_, _>>()?;
        Ok(RmsPropState { rms, stddev, learning_rate, momentum })
    }

    fn update<S: Parameterized<P>>(
        &self,
        gradients: S,
        state: Self::State<S>,
        parameters: &S,
        arguments: &UpdateArguments<'_, P, S>,
    ) -> Result<(S, Self::State<S>), OptimizerError> {
        let RmsPropState { rms, stddev, learning_rate, momentum } = state;
        let mismatch = || OptimizerError::StructureMismatch {
            message: "the state of `RmsProp` does not match its configuration".to_string(),
        };
        let (updates, rms, stddev) = match (self.centered, rms.into_iter().next(), stddev.into_iter().next()) {
            (false, Some(rms_state), None) => {
                let (updates, rms_state) = self.rms.update(gradients, rms_state, parameters, arguments)?;
                (updates, vec![rms_state], Vec::new())
            }
            (true, None, Some(stddev_state)) => {
                let (updates, stddev_state) = self.stddev.update(gradients, stddev_state, parameters, arguments)?;
                (updates, Vec::new(), vec![stddev_state])
            }
            _ => return Err(mismatch()),
        };
        let (updates, learning_rate) = self.learning_rate.update(updates, learning_rate, parameters, arguments)?;
        let (updates, momentum) = match (self.momentum(), momentum.into_iter().next()) {
            (Some(trace), Some(momentum_state)) => {
                let (updates, momentum_state) = trace.update(updates, momentum_state, parameters, arguments)?;
                (updates, vec![momentum_state])
            }
            (None, None) => (updates, Vec::new()),
            _ => return Err(mismatch()),
        };
        Ok((updates, RmsPropState { rms, stddev, learning_rate, momentum }))
    }
}

parameterized_state! {
    /// State of [`Adagrad`] for the parameter structure `S`, which holds the state of its [`ScaleByRss`] transformation
    /// and the step count of its learning rate schedule.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct AdagradState<P, S> with family AdagradStateFamily {
        /// State of the [`ScaleByRss`] transformation of the optimizer.
        rss: ScaleByRssState<P, S>,

        /// State of the [`ScaleBySchedule`] transformation that applies the learning rate.
        learning_rate: ScaleByScheduleState<P>,
    }
}

impl<P: Parameter, S> AdagradState<P, S> {
    /// Returns the state of the [`ScaleByRss`] transformation of the optimizer.
    #[inline]
    pub fn rss(&self) -> &ScaleByRssState<P, S> {
        &self.rss
    }

    /// Returns the state of the [`ScaleBySchedule`] transformation that applies the learning rate.
    #[inline]
    pub fn learning_rate(&self) -> &ScaleByScheduleState<P> {
        &self.learning_rate
    }
}

/// Adagrad optimizer of [Duchi et al. (2011)](https://jmlr.org/papers/v12/duchi11a.html), which divides the gradients
/// by the root of the sum of all their past squares, and which is the analogue of
/// [`optax.adagrad`](https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.adagrad). Its update is
///
/// ```text
/// sₜ = sₜ₋₁ + gₜ²
/// uₜ = −ηₜ · gₜ / √(sₜ + ε)   where sₜ > 0
/// uₜ = 0                      where sₜ = 0
/// ```
///
/// Here `t ≥ 1` counts the updates including the current one, `gₜ` is the gradient at the parameters `θₜ₋₁`, `uₜ` is
/// the update that is added to them (i.e., `θₜ = θₜ₋₁ + uₜ`), and `ηₜ` is the learning rate [`Schedule`] evaluated at
/// the `t − 1` updates applied before the current one. The sum of squares `sₜ` starts at the initial accumulator value
/// `s₀`, and `ε` is added for numerical stability. All operations are element-wise. The defaults are `s₀ = 0.1` and
/// `ε = 1e-7`.
///
/// It is the chain of [`ScaleByRss`] and [`ScaleBySchedule::learning_rate`].
///
/// # References
///
///   - Duchi, Hazan, and Singer (2011). [Adaptive Subgradient Methods for Online Learning and Stochastic
///     Optimization](https://jmlr.org/papers/v12/duchi11a.html).
#[derive(Clone, Debug)]
pub struct Adagrad<Sc: Schedule> {
    /// Transformation that rescales the gradients.
    rss: ScaleByRss,

    /// Transformation that scales the updates by the negated learning rate.
    learning_rate: ScaleBySchedule<Sc>,
}

impl<Sc: Schedule> Adagrad<Sc> {
    /// Creates a new [`Adagrad`] with the learning rate schedule `learning_rate` (e.g., an `f64` for a constant
    /// learning rate) and default values for all other hyperparameters.
    pub fn new(learning_rate: Sc) -> Self {
        Self { rss: ScaleByRss::new(), learning_rate: ScaleBySchedule::learning_rate(learning_rate) }
    }

    /// Returns this optimizer with `initial_accumulator_value` as the initial value of its accumulators (refer to
    /// [`ScaleByRss::with_initial_accumulator_value`]).
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `initial_accumulator_value` is negative or not finite.
    pub fn with_initial_accumulator_value(mut self, initial_accumulator_value: f64) -> Result<Self, OptimizerError> {
        self.rss = self.rss.with_initial_accumulator_value(initial_accumulator_value)?;
        Ok(self)
    }

    /// Returns this optimizer with `epsilon` as the term that is added for numerical stability outside of square roots
    /// (refer to [`ScaleByRss::with_epsilon`]).
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `epsilon` is negative or not finite.
    pub fn with_epsilon(mut self, epsilon: f64) -> Result<Self, OptimizerError> {
        self.rss = self.rss.with_epsilon(epsilon)?;
        Ok(self)
    }

    /// Returns the [`ScaleByRss`] transformation of this optimizer, which holds its rescaling hyperparameters.
    #[inline]
    pub fn rss(&self) -> &ScaleByRss {
        &self.rss
    }

    /// Returns the learning rate [`Schedule`] of this optimizer.
    #[inline]
    pub fn learning_rate(&self) -> &Sc {
        self.learning_rate.schedule()
    }
}

impl<P: OptimizerValue, Sc: Schedule> Optimizer<P> for Adagrad<Sc> {
    type State<S: Parameterized<P>> = AdagradState<P, S>;

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        let rss = self.rss.initialize(parameters)?;
        let learning_rate = self.learning_rate.initialize(parameters)?;
        Ok(AdagradState { rss, learning_rate })
    }

    fn update<S: Parameterized<P>>(
        &self,
        gradients: S,
        state: Self::State<S>,
        parameters: &S,
        arguments: &UpdateArguments<'_, P, S>,
    ) -> Result<(S, Self::State<S>), OptimizerError> {
        let (updates, rss) = self.rss.update(gradients, state.rss, parameters, arguments)?;
        let (updates, learning_rate) =
            self.learning_rate.update(updates, state.learning_rate, parameters, arguments)?;
        Ok((updates, AdagradState { rss, learning_rate }))
    }
}

parameterized_state! {
    /// State of [`NovoGrad`] for the parameter structure `S`, which holds the state of its [`ScaleByNovograd`]
    /// transformation and the step count of its learning rate schedule.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct NovoGradState<P, S> with family NovoGradStateFamily {
        /// State of the [`ScaleByNovograd`] transformation of the optimizer.
        novograd: ScaleByNovogradState<P, S>,

        /// State of the [`ScaleBySchedule`] transformation that applies the learning rate.
        learning_rate: ScaleByScheduleState<P>,
    }
}

impl<P: Parameter, S> NovoGradState<P, S> {
    /// Returns the state of the [`ScaleByNovograd`] transformation of the optimizer.
    #[inline]
    pub fn novograd(&self) -> &ScaleByNovogradState<P, S> {
        &self.novograd
    }

    /// Returns the state of the [`ScaleBySchedule`] transformation that applies the learning rate.
    #[inline]
    pub fn learning_rate(&self) -> &ScaleByScheduleState<P> {
        &self.learning_rate
    }
}

/// NovoGrad optimizer of [Ginsburg et al. (2019)](https://arxiv.org/abs/1905.11286), which normalizes every leaf of the
/// gradients by a moving average of its squared norm before accumulating it with momentum, and which is the analogue of
/// [`optax.novograd`](https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.novograd). For every leaf, its
/// update is
///
/// ```text
/// vₜ = ‖gₜ‖²                          if t = 1
/// vₜ = β₂ · vₜ₋₁ + (1 − β₂) · ‖gₜ‖²   if t > 1
/// aₜ = gₜ / (√(vₜ + ε̄) + ε) + λ · θₜ₋₁
/// mₜ = aₜ                             if t = 1
/// mₜ = β₁ · mₜ₋₁ + aₜ                 if t > 1
/// uₜ = −ηₜ · mₜ
/// ```
///
/// Here `t ≥ 1` counts the updates including the current one, `gₜ` is the gradient at the parameters `θₜ₋₁`, `uₜ` is
/// the update that is added to them (i.e., `θₜ = θₜ₋₁ + uₜ`), and `ηₜ` is the learning rate [`Schedule`] evaluated at
/// the `t − 1` updates applied before the current one. The second moment `vₜ` is a scalar per leaf that tracks the
/// squared norm `‖gₜ‖²` of the gradient of that leaf with the decay rate `β₂`, `aₜ` is the normalized gradient with the
/// weight decay `λ`, the momentum `mₜ` accumulates it with the decay rate `β₁`, and `ε` and `ε̄` are added outside and
/// inside the square root for numerical stability. The operations on `aₜ` and `mₜ` are element-wise. The defaults are
/// `β₁ = 0.9`, `β₂ = 0.25`, `ε = 1e-6`, `ε̄ = 0`, and `λ = 0`.
///
/// It is the chain of [`ScaleByNovograd`] and [`ScaleBySchedule::learning_rate`].
///
/// # References
///
///   - Ginsburg et al. (2019). [Stochastic Gradient Methods with Layer-wise Adaptive Moments for Training of Deep
///     Networks](https://arxiv.org/abs/1905.11286).
#[derive(Clone, Debug)]
pub struct NovoGrad<Sc: Schedule> {
    /// Transformation that rescales the gradients.
    novograd: ScaleByNovograd,

    /// Transformation that scales the updates by the negated learning rate.
    learning_rate: ScaleBySchedule<Sc>,
}

impl<Sc: Schedule> NovoGrad<Sc> {
    /// Creates a new [`NovoGrad`] with the learning rate schedule `learning_rate` (e.g., an `f64` for a constant
    /// learning rate) and default values for all other hyperparameters.
    pub fn new(learning_rate: Sc) -> Self {
        Self {
            novograd: ScaleByNovograd::new().with_epsilon(1e-6).unwrap(),
            learning_rate: ScaleBySchedule::learning_rate(learning_rate),
        }
    }

    /// Returns this optimizer with `beta1` as the decay rate of the first moment estimates (refer to
    /// [`ScaleByNovograd::with_beta1`]).
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `beta1` is not in `[0, 1)`.
    pub fn with_beta1(mut self, beta1: f64) -> Result<Self, OptimizerError> {
        self.novograd = self.novograd.with_beta1(beta1)?;
        Ok(self)
    }

    /// Returns this optimizer with `beta2` as the decay rate of the second moment estimates (refer to
    /// [`ScaleByNovograd::with_beta2`]).
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `beta2` is not in `[0, 1)`.
    pub fn with_beta2(mut self, beta2: f64) -> Result<Self, OptimizerError> {
        self.novograd = self.novograd.with_beta2(beta2)?;
        Ok(self)
    }

    /// Returns this optimizer with `epsilon` as the term that is added for numerical stability outside of square roots
    /// (refer to [`ScaleByNovograd::with_epsilon`]).
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `epsilon` is negative or not finite.
    pub fn with_epsilon(mut self, epsilon: f64) -> Result<Self, OptimizerError> {
        self.novograd = self.novograd.with_epsilon(epsilon)?;
        Ok(self)
    }

    /// Returns this optimizer with `epsilon_root` as the term that is added inside square roots (refer to
    /// [`ScaleByNovograd::with_epsilon_root`]).
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `epsilon_root` is negative or not finite.
    pub fn with_epsilon_root(mut self, epsilon_root: f64) -> Result<Self, OptimizerError> {
        self.novograd = self.novograd.with_epsilon_root(epsilon_root)?;
        Ok(self)
    }

    /// Returns this optimizer with `weight_decay` as the multiple of the parameters that is added to the normalized
    /// gradients (refer to [`ScaleByNovograd::with_weight_decay`]).
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `weight_decay` is not finite.
    pub fn with_weight_decay(mut self, weight_decay: f64) -> Result<Self, OptimizerError> {
        self.novograd = self.novograd.with_weight_decay(weight_decay)?;
        Ok(self)
    }

    /// Returns the [`ScaleByNovograd`] transformation of this optimizer, which holds its rescaling hyperparameters.
    #[inline]
    pub fn novograd(&self) -> &ScaleByNovograd {
        &self.novograd
    }

    /// Returns the learning rate [`Schedule`] of this optimizer.
    #[inline]
    pub fn learning_rate(&self) -> &Sc {
        self.learning_rate.schedule()
    }
}

impl<P: OptimizerValue, Sc: Schedule> Optimizer<P> for NovoGrad<Sc> {
    type State<S: Parameterized<P>> = NovoGradState<P, S>;

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        let novograd = self.novograd.initialize(parameters)?;
        let learning_rate = self.learning_rate.initialize(parameters)?;
        Ok(NovoGradState { novograd, learning_rate })
    }

    fn update<S: Parameterized<P>>(
        &self,
        gradients: S,
        state: Self::State<S>,
        parameters: &S,
        arguments: &UpdateArguments<'_, P, S>,
    ) -> Result<(S, Self::State<S>), OptimizerError> {
        let (updates, novograd) = self.novograd.update(gradients, state.novograd, parameters, arguments)?;
        let (updates, learning_rate) =
            self.learning_rate.update(updates, state.learning_rate, parameters, arguments)?;
        Ok((updates, NovoGradState { novograd, learning_rate }))
    }
}

parameterized_state! {
    /// State of [`Adadelta`] for the parameter structure `S`, which holds the state of its [`ScaleByAdadelta`]
    /// transformation and the step count of its learning rate schedule. Weight decay is stateless.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct AdadeltaState<P, S> with family AdadeltaStateFamily {
        /// State of the [`ScaleByAdadelta`] transformation of the optimizer.
        adadelta: ScaleByAdadeltaState<P, S>,

        /// State of the [`ScaleBySchedule`] transformation that applies the learning rate.
        learning_rate: ScaleByScheduleState<P>,
    }
}

impl<P: Parameter, S> AdadeltaState<P, S> {
    /// Returns the state of the [`ScaleByAdadelta`] transformation of the optimizer.
    #[inline]
    pub fn adadelta(&self) -> &ScaleByAdadeltaState<P, S> {
        &self.adadelta
    }

    /// Returns the state of the [`ScaleBySchedule`] transformation that applies the learning rate.
    #[inline]
    pub fn learning_rate(&self) -> &ScaleByScheduleState<P> {
        &self.learning_rate
    }
}

/// Adadelta optimizer of [Zeiler (2012)](https://arxiv.org/abs/1212.5701), which scales the gradients by the ratio of
/// the root mean squares of the recent updates and the recent gradients, and which is the analogue of
/// [`optax.adadelta`](https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.adadelta). Its update is
///
/// ```text
/// dₜ = gₜ + λ · θₜ₋₁
/// Gₜ = ρ · Gₜ₋₁ + (1 − ρ) · dₜ²
/// aₜ = √(Dₜ₋₁ + ε) / √(Gₜ + ε) · dₜ
/// Dₜ = ρ · Dₜ₋₁ + (1 − ρ) · aₜ²
/// uₜ = −ηₜ · aₜ
/// ```
///
/// Here `t ≥ 1` counts the updates including the current one, `gₜ` is the gradient at the parameters `θₜ₋₁`, `uₜ` is
/// the update that is added to them (i.e., `θₜ = θₜ₋₁ + uₜ`), and `ηₜ` is the learning rate [`Schedule`] evaluated at
/// the `t − 1` updates applied before the current one. The averages `Gₜ` of the squared gradients and `Dₜ` of the
/// squared rescaled updates start at zero and decay with the rate `ρ`, `ε` is added inside both square roots, `aₜ` is
/// the rescaled update, and `λ` is the rate of a weight decay that is added to the gradients before they are rescaled
/// (i.e., L2 regularization rather than decoupled weight decay). All operations are element-wise. The defaults are
/// `ρ = 0.9`, `ε = 1e-6`, and `λ = 0`, and [`Self::with_weight_decay_mask`] restricts the weight decay to the leaves
/// that the mask selects. Optax also accepts no learning rate, in which case it applies neither a learning rate nor a
/// sign flip, while this optimizer always applies its learning rate. The original Adadelta of Zeiler (2012) has no
/// learning rate, which corresponds to `ηₜ = 1`.
///
/// It is the chain of [`AddDecayedWeights`], [`ScaleByAdadelta`], and [`ScaleBySchedule::learning_rate`].
///
/// # References
///
///   - Zeiler (2012). [ADADELTA: An Adaptive Learning Rate Method](https://arxiv.org/abs/1212.5701).
#[derive(Clone, Debug)]
pub struct Adadelta<Sc: Schedule> {
    /// Transformation that adds the decayed weights to the gradients.
    weight_decay: AddDecayedWeights,

    /// Transformation that rescales the gradients.
    adadelta: ScaleByAdadelta,

    /// Transformation that scales the updates by the negated learning rate.
    learning_rate: ScaleBySchedule<Sc>,
}

impl<Sc: Schedule> Adadelta<Sc> {
    /// Creates a new [`Adadelta`] with the learning rate schedule `learning_rate` (e.g., an `f64` for a constant
    /// learning rate) and default values for all other hyperparameters.
    pub fn new(learning_rate: Sc) -> Self {
        Self {
            weight_decay: AddDecayedWeights::new(0.0).unwrap(),
            adadelta: ScaleByAdadelta::new(),
            learning_rate: ScaleBySchedule::learning_rate(learning_rate),
        }
    }

    /// Returns this optimizer with `rho` as the decay rate of its moving averages.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `rho` is not in `[0, 1)`.
    pub fn with_rho(mut self, rho: f64) -> Result<Self, OptimizerError> {
        self.adadelta = self.adadelta.with_rho(rho)?;
        Ok(self)
    }

    /// Returns this optimizer with `epsilon` as the term that is added inside the square roots.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `epsilon` is negative or not finite.
    pub fn with_epsilon(mut self, epsilon: f64) -> Result<Self, OptimizerError> {
        self.adadelta = self.adadelta.with_epsilon(epsilon)?;
        Ok(self)
    }

    /// Returns this optimizer with `weight_decay` as the rate of its weight decay, keeping its mask.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `weight_decay` is not finite.
    pub fn with_weight_decay(mut self, weight_decay: f64) -> Result<Self, OptimizerError> {
        self.weight_decay = self.weight_decay.with_rate(weight_decay)?;
        Ok(self)
    }

    /// Returns this optimizer with `mask` as the predicate that selects the leaves that decay by their
    /// [`ParameterPath`]s (refer to [`AddDecayedWeights::with_mask`]).
    pub fn with_weight_decay_mask<Mask>(mut self, mask: Mask) -> Self
    where
        Mask: 'static + Fn(&ParameterPath) -> bool + Send + Sync,
    {
        self.weight_decay = self.weight_decay.with_mask(mask);
        self
    }

    /// Returns the [`ScaleByAdadelta`] transformation of this optimizer.
    #[inline]
    pub fn adadelta(&self) -> &ScaleByAdadelta {
        &self.adadelta
    }

    /// Returns the [`AddDecayedWeights`] transformation of this optimizer.
    #[inline]
    pub fn weight_decay(&self) -> &AddDecayedWeights {
        &self.weight_decay
    }

    /// Returns the learning rate [`Schedule`] of this optimizer.
    #[inline]
    pub fn learning_rate(&self) -> &Sc {
        self.learning_rate.schedule()
    }
}

impl<P: OptimizerValue, Sc: Schedule> Optimizer<P> for Adadelta<Sc> {
    type State<S: Parameterized<P>> = AdadeltaState<P, S>;

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        let adadelta = self.adadelta.initialize(parameters)?;
        let learning_rate = self.learning_rate.initialize(parameters)?;
        Ok(AdadeltaState { adadelta, learning_rate })
    }

    fn update<S: Parameterized<P>>(
        &self,
        gradients: S,
        state: Self::State<S>,
        parameters: &S,
        arguments: &UpdateArguments<'_, P, S>,
    ) -> Result<(S, Self::State<S>), OptimizerError> {
        let (updates, ()) = self.weight_decay.update(gradients, (), parameters, arguments)?;
        let (updates, adadelta) = self.adadelta.update(updates, state.adadelta, parameters, arguments)?;
        let (updates, learning_rate) =
            self.learning_rate.update(updates, state.learning_rate, parameters, arguments)?;
        Ok((updates, AdadeltaState { adadelta, learning_rate }))
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::tests::{assert_parity, parity_updates, state_paths, vector_updates};

    use super::*;

    #[test]
    fn test_rms_prop_new() {
        let optimizer = RmsProp::new(0.1f64);
        assert_eq!(optimizer.rms(), &ScaleByRms::new());
        assert!(!optimizer.centered());
        assert_eq!(optimizer.momentum(), None);
        let optimizer = optimizer
            .with_decay(0.5)
            .unwrap()
            .with_epsilon(0.0)
            .unwrap()
            .with_initial_scale(1.0)
            .unwrap()
            .with_epsilon_in_sqrt(false)
            .with_bias_correction(true)
            .with_centered(true)
            .with_nesterov(true)
            .with_momentum(0.9)
            .unwrap();
        assert_eq!(
            optimizer.rms(),
            &ScaleByRms::new()
                .with_decay(0.5)
                .unwrap()
                .with_epsilon(0.0)
                .unwrap()
                .with_initial_scale(1.0)
                .unwrap()
                .with_epsilon_in_sqrt(false)
                .with_bias_correction(true),
        );
        assert!(optimizer.centered());
        assert_eq!(optimizer.momentum(), Some(Trace::new(0.9).unwrap().with_nesterov(true)));
        assert!(matches!(
            RmsProp::new(0.1f64).with_momentum(f64::NAN),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`momentum` must be finite but got `NaN`",
        ));
    }

    #[test]
    fn test_rms_prop_initialize() {
        assert_eq!(
            state_paths(&RmsProp::new(0.1f64)),
            vec![
                "$.rms[0].count",
                "$.rms[0].second_moments[\"bias\"]",
                "$.rms[0].second_moments[\"weights\"]",
                "$.learning_rate.count",
            ],
        );
        assert_eq!(state_paths(&RmsProp::new(0.1f64).with_centered(true).with_momentum(0.9).unwrap()).len(), 8);
    }

    #[test]
    fn test_rms_prop_update() {
        // The RMS updates `2 / √2` and `1 / √1.5` of the transformation tests are scaled by `-0.5`, and the momentum
        // `0.5` accumulates them after the learning rate.
        let optimizer =
            RmsProp::new(0.5f64).with_decay(0.5).unwrap().with_epsilon(0.0).unwrap().with_epsilon_in_sqrt(false);
        let first = -0.5 * (1.0f32 / 2.0f32.sqrt() * 2.0);
        let second = -0.5 * (1.0f32 / 1.5f32.sqrt());
        assert_eq!(vector_updates(&optimizer, &[0.0], &[&[2.0], &[1.0]]), vec![vec![first], vec![second]]);
        let with_momentum = optimizer.clone().with_momentum(0.5).unwrap();
        assert_eq!(
            vector_updates(&with_momentum, &[0.0], &[&[2.0], &[1.0]]),
            vec![vec![first], vec![second + 0.5 * first],],
        );

        // The centered variant divides by the standard deviation, as in the transformation tests.
        let centered = optimizer.with_centered(true);
        assert_eq!(
            vector_updates(&centered, &[0.0], &[&[2.0], &[4.0]]),
            vec![vec![-1.0], vec![-0.5 * (1.0f32 / 2.75f32.sqrt() * 4.0)],],
        );
    }

    #[test]
    fn test_adagrad_new() {
        let optimizer = Adagrad::new(0.1f64);
        assert_eq!(optimizer.rss(), &ScaleByRss::new());
        assert!(matches!(
            Adagrad::new(0.1f64).with_initial_accumulator_value(f64::NAN),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`initial_accumulator_value` must be finite and non-negative but got `NaN`",
        ));
    }

    #[test]
    fn test_adagrad_update() {
        // The Adagrad updates `[1, 0]` and `[3 / √13, 0]` of the transformation tests are scaled by `-0.5`.
        let optimizer = Adagrad::new(0.5f64).with_initial_accumulator_value(0.0).unwrap().with_epsilon(0.0).unwrap();
        assert_eq!(state_paths(&optimizer)[0], "$.rss.sum_of_squares[\"bias\"]");
        assert_eq!(
            vector_updates(&optimizer, &[0.0, 0.0], &[&[2.0, 0.0], &[3.0, 0.0]]),
            vec![vec![-0.5, -0.0], vec![-0.5 * (13.0f32.sqrt().recip() * 3.0), -0.0],],
        );
    }

    #[test]
    fn test_adadelta_new() {
        let optimizer = Adadelta::new(0.1f64);
        assert_eq!(optimizer.adadelta(), &ScaleByAdadelta::new());
        assert_eq!(optimizer.weight_decay().rate(), 0.0);
        let optimizer = optimizer
            .with_rho(0.5)
            .unwrap()
            .with_epsilon(1.0)
            .unwrap()
            .with_weight_decay(0.5)
            .unwrap()
            .with_weight_decay_mask(|_| true);
        assert_eq!((optimizer.adadelta().rho(), optimizer.adadelta().epsilon()), (0.5, 1.0));
        assert_eq!(optimizer.weight_decay().rate(), 0.5);
        assert_eq!(optimizer.learning_rate(), &0.1);
    }

    #[test]
    fn test_adadelta_update() {
        // The Adadelta updates of the transformation tests are scaled by `-0.5`. The weight decay is added to the
        // gradients before they are rescaled, so the gradient `1` of the parameter `2` with the rate `0.5` behaves like
        // the gradient `2`.
        let optimizer = Adadelta::new(0.5f64).with_rho(0.5).unwrap().with_epsilon(1.0).unwrap();
        assert_eq!(state_paths(&optimizer).len(), 5);
        let first = 1.0f32.sqrt() / 3.0f32.sqrt() * 2.0;
        let update_average = 0.5f32 * (first * first) + 0.5 * 0.0;
        let second = (update_average + 1.0).sqrt() / 2.5f32.sqrt() * 1.0;
        assert_eq!(
            vector_updates(&optimizer, &[0.0], &[&[2.0], &[1.0]]),
            vec![vec![-0.5 * first], vec![-0.5 * second]],
        );
        let optimizer = optimizer.with_weight_decay(0.5).unwrap();
        assert_eq!(vector_updates(&optimizer, &[2.0], &[&[1.0]]), vec![vec![-0.5 * first]]);
    }

    #[test]
    fn test_novo_grad_new() {
        let optimizer = NovoGrad::new(0.1f64);
        assert_eq!(optimizer.novograd(), &ScaleByNovograd::new().with_epsilon(1e-6).unwrap());
        let optimizer = optimizer.with_weight_decay(0.5).unwrap().with_epsilon_root(1e-3).unwrap();
        assert_eq!((optimizer.novograd().weight_decay(), optimizer.novograd().epsilon_root()), (0.5, 1e-3));
    }

    #[test]
    fn test_novo_grad_update() {
        // The NovoGrad updates `[0.6, 0.8]` and `0.5 · [0.6, 0.8] + [0, 1] / √13` of the transformation tests are
        // scaled by `-0.5`.
        let optimizer =
            NovoGrad::new(0.5f64).with_beta1(0.5).unwrap().with_beta2(0.5).unwrap().with_epsilon(0.0).unwrap();
        assert_eq!(state_paths(&optimizer)[0], "$.novograd.count");
        assert_eq!(
            vector_updates(&optimizer, &[1.0, 1.0], &[&[3.0, 4.0], &[0.0, 1.0]]),
            vec![
                vec![-0.5 * (3.0f32 / 5.0), -0.5 * (4.0f32 / 5.0)],
                vec![-0.5 * (0.5f32 * (3.0f32 / 5.0) + 0.0), -0.5 * (0.5f32 * (4.0f32 / 5.0) + 1.0 / 13.0f32.sqrt())],
            ],
        );
    }

    #[test]
    fn test_rms_prop_optax_parity() {
        // Optax 9ecde04b: `optax.rmsprop(0.1)`.
        const RMSPROP: [[f32; 9]; 4] = [
            [
                -0.31622618,
                0.3162274,
                -0.3162276,
                -0.31622767,
                0.31622618,
                -0.3162274,
                0.3162276,
                -0.3162277,
                0.31622776,
            ],
            [
                0.14744131,
                -0.25189623,
                -0.10482843,
                -0.14744191,
                -0.30151117,
                0.28571424,
                -0.10482843,
                0.122859016,
                -0.16563697,
            ],
            [
                -0.29947624,
                0.10068186,
                0.18114567,
                0.109369636,
                -0.052242268,
                -0.23117995,
                -0.20804451,
                -0.20037603,
                0.04927543,
            ],
            [
                -0.1751994,
                -0.14219122,
                0.22968228,
                -0.07468302,
                0.22845577,
                -0.02697741,
                0.24679866,
                -0.100169174,
                -0.13977472,
            ],
        ];
        // Optax 9ecde04b: `optax.rmsprop(0.1, centered=True)`.
        const RMSPROP_CENTERED: [[f32; 9]; 4] = [
            [
                -0.3333315,
                0.33333287,
                -0.33333313,
                -0.33333322,
                0.3333315,
                -0.33333287,
                0.33333313,
                -0.33333325,
                0.33333328,
            ],
            [
                0.14847781,
                -0.25252512,
                -0.113731794,
                -0.16188091,
                -0.3084597,
                0.28930864,
                -0.10653373,
                0.124333955,
                -0.16631065,
            ],
            [
                -0.31789133,
                0.10075181,
                0.18247445,
                0.11320563,
                -0.053951528,
                -0.23314194,
                -0.20872565,
                -0.21098338,
                0.049720738,
            ],
            [
                -0.19511078,
                -0.14306936,
                0.23235111,
                -0.07862385,
                0.22893476,
                -0.027283395,
                0.2518313,
                -0.10838407,
                -0.13982248,
            ],
        ];
        // Optax 9ecde04b: `optax.rmsprop(0.1, momentum=0.9)`.
        const RMSPROP_MOMENTUM: [[f32; 9]; 4] = [
            [
                -0.31622618,
                0.3162274,
                -0.3162276,
                -0.31622767,
                0.31622618,
                -0.3162274,
                0.3162276,
                -0.3162277,
                0.31622776,
            ],
            [
                -0.13716225,
                0.032708436,
                -0.38943326,
                -0.43204683,
                -0.016907603,
                0.0011095703,
                0.17977642,
                -0.16174592,
                0.118967995,
            ],
            [
                -0.42292225,
                0.13011946,
                -0.16934425,
                -0.2794725,
                -0.06745911,
                -0.23018134,
                -0.04624574,
                -0.34594736,
                0.15634662,
            ],
            [
                -0.5558294,
                -0.025083706,
                0.07727246,
                -0.32620826,
                0.16774258,
                -0.2341406,
                0.2051775,
                -0.4115218,
                0.00093722343,
            ],
        ];
        // Optax 9ecde04b: `optax.rmsprop(0.1, centered=True, momentum=0.9, nesterov=True)`.
        const RMSPROP_NESTEROV: [[f32; 9]; 4] = [
            [
                -0.63332987,
                0.63333243,
                -0.63333297,
                -0.6333331,
                0.63332987,
                -0.63333243,
                0.63333297,
                -0.6333332,
                0.6333332,
            ],
            [
                0.012109324,
                -0.20979813,
                -0.48609024,
                -0.57757366,
                -0.3160749,
                0.2796868,
                0.06758573,
                -0.033765413,
                -0.04599029,
            ],
            [
                -0.72672516,
                0.22988275,
                0.011578858,
                -0.15903275,
                -0.109361604,
                -0.45162934,
                -0.23987122,
                -0.5431579,
                0.20275772,
            ],
            [
                -0.73866093,
                -0.15561394,
                0.28766108,
                -0.39439985,
                0.38510698,
                -0.24847712,
                0.45044845,
                -0.50488675,
                -0.12792943,
            ],
        ];
        // Optax 9ecde04b: `optax.rmsprop(0.1, decay=0.8, eps=1e-3, initial_scale=0.5, eps_in_sqrt=False,
        // bias_correction=True)`.
        const RMSPROP_BIAS_CORRECTION: [[f32; 9]; 4] = [
            [
                -0.007048484,
                0.013993004,
                -0.020737091,
                -0.027198046,
                0.007048484,
                -0.013993004,
                0.020737091,
                -0.033311125,
                0.039031267,
            ],
            [
                0.005280402,
                -0.025739972,
                -0.010334137,
                -0.020159844,
                -0.030856147,
                0.04004905,
                -0.010334137,
                0.019761866,
                -0.033085786,
            ],
            [
                -0.039861266,
                0.013349133,
                0.026505133,
                0.019521238,
                -0.0066860663,
                -0.054887373,
                -0.032865997,
                -0.04884995,
                0.01226363,
            ],
            [
                -0.032093775,
                -0.024367435,
                0.054116376,
                -0.0159304,
                0.047400985,
                -0.0074905343,
                0.066967584,
                -0.029579267,
                -0.043738592,
            ],
        ];
        assert_parity(&parity_updates(&RmsProp::new(0.1)), &RMSPROP);
        assert_parity(&parity_updates(&RmsProp::new(0.1).with_centered(true)), &RMSPROP_CENTERED);
        assert_parity(&parity_updates(&RmsProp::new(0.1).with_momentum(0.9).unwrap()), &RMSPROP_MOMENTUM);
        assert_parity(
            &parity_updates(&RmsProp::new(0.1).with_centered(true).with_momentum(0.9).unwrap().with_nesterov(true)),
            &RMSPROP_NESTEROV,
        );
        assert_parity(
            &parity_updates(
                &RmsProp::new(0.1)
                    .with_decay(0.8)
                    .unwrap()
                    .with_epsilon(1e-3)
                    .unwrap()
                    .with_initial_scale(0.5)
                    .unwrap()
                    .with_epsilon_in_sqrt(false)
                    .with_bias_correction(true),
            ),
            &RMSPROP_BIAS_CORRECTION,
        );
    }

    #[test]
    fn test_adagrad_optax_parity() {
        // Optax 9ecde04b: `optax.adagrad(0.1)`.
        const ADAGRAD: [[f32; 9]; 4] = [
            [
                -0.030151123,
                0.053452224,
                -0.06882471,
                -0.07844644,
                0.030151123,
                -0.053452224,
                0.06882471,
                -0.084515415,
                0.08846517,
            ],
            [
                0.014907114,
                -0.05555554,
                -0.022360673,
                -0.036514834,
                -0.067082025,
                0.07302967,
                -0.022360673,
                0.03202563,
                -0.04585852,
            ],
            [
                -0.066666655,
                0.02169304,
                0.04082482,
                0.026413525,
                -0.011111109,
                -0.06348109,
                -0.048794996,
                -0.053935982,
                0.012991396,
            ],
            [
                -0.04061384,
                -0.030942632,
                0.058131825,
                -0.017342197,
                0.05547001,
                -0.0070359753,
                0.0659912,
                -0.02603778,
                -0.03631365,
            ],
        ];
        assert_parity(&parity_updates(&Adagrad::new(0.1)), &ADAGRAD);
    }

    #[test]
    fn test_adadelta_optax_parity() {
        // Optax 9ecde04b: `optax.adadelta(0.1)`.
        const ADADELTA: [[f32; 9]; 4] = [
            [
                -0.00031606978,
                0.0003161883,
                -0.0003162102,
                -0.0003162179,
                0.00031606978,
                -0.0003161883,
                0.0003162102,
                -0.00031622147,
                0.00031622342,
            ],
            [
                0.00020837177,
                -0.00035619488,
                -0.0001482376,
                -0.0002085055,
                -0.0004262734,
                0.0004040255,
                -0.0001482376,
                0.0001737439,
                -0.00023424183,
            ],
            [
                -0.00045743055,
                0.0001792082,
                0.0002637192,
                0.000167109,
                -0.000100704194,
                -0.00043447368,
                -0.00030288156,
                -0.00029732584,
                7.710636e-5,
            ],
            [
                -0.00036296566,
                -0.00025722606,
                0.0003776191,
                -0.00011761893,
                0.00043018776,
                -6.1319195e-5,
                0.0004220881,
                -0.0001725006,
                -0.00021487172,
            ],
        ];
        // Optax 9ecde04b: `optax.adadelta(0.1, rho=0.8, eps=1e-4, weight_decay=0.01)`.
        const ADADELTA_WEIGHT_DECAY: [[f32; 9]; 4] = [
            [
                -0.0021870255,
                0.0022234987,
                -0.0022306286,
                -0.0022327497,
                0.0021870255,
                -0.0022225592,
                0.0022306286,
                -0.0022339632,
                0.0022344773,
            ],
            [
                0.0013219694,
                -0.0024767325,
                -0.0012178723,
                -0.0015679734,
                -0.0029726815,
                0.002865228,
                -0.0008480446,
                0.0011769217,
                -0.0017668443,
            ],
            [
                -0.0031023358,
                0.0014494241,
                0.0017562017,
                0.001173642,
                -0.00067186326,
                -0.0031333168,
                -0.0020338036,
                -0.0021982389,
                0.0005554406,
            ],
            [
                -0.0025440431,
                -0.0018256874,
                0.0026242116,
                -0.0009891967,
                0.0030434926,
                -0.00048345464,
                0.0029825221,
                -0.0013815857,
                -0.001691835,
            ],
        ];
        assert_parity(&parity_updates(&Adadelta::new(0.1)), &ADADELTA);
        assert_parity(
            &parity_updates(
                &Adadelta::new(0.1)
                    .with_rho(0.8)
                    .unwrap()
                    .with_epsilon(1e-4)
                    .unwrap()
                    .with_weight_decay(0.01)
                    .unwrap(),
            ),
            &ADADELTA_WEIGHT_DECAY,
        );
    }

    #[test]
    fn test_novo_grad_optax_parity() {
        // Optax 9ecde04b: `optax.novograd(0.1)`.
        const NOVOGRAD: [[f32; 9]; 4] = [
            [
                -0.02672605,
                0.0534521,
                -0.08017816,
                -0.041931346,
                0.0104828365,
                -0.020965673,
                0.03144851,
                -0.052414186,
                0.06289702,
            ],
            [
                -0.007501382,
                -0.03465342,
                -0.10526446,
                -0.06412774,
                -0.030149743,
                0.033909958,
                0.015108894,
                -0.020783236,
                0.010425641,
            ],
            [
                -0.090665035,
                -0.0032168177,
                -0.03879549,
                -0.03625561,
                -0.03428789,
                -0.033859108,
                -0.022167588,
                -0.07592986,
                0.023689313,
            ],
            [
                -0.13000554,
                -0.03920039,
                0.049796317,
                -0.047563743,
                0.013941991,
                -0.037940044,
                0.047250804,
                -0.09820426,
                -0.023480708,
            ],
        ];
        // Optax 9ecde04b: `optax.novograd(0.1, weight_decay=0.01)`.
        const NOVOGRAD_WEIGHT_DECAY: [[f32; 9]; 4] = [
            [
                -0.02722605,
                0.0544521,
                -0.08217815,
                -0.042931344,
                0.010982837,
                -0.021215674,
                0.033448514,
                -0.053914186,
                0.06214702,
            ],
            [
                -0.0084241545,
                -0.032807875,
                -0.10898228,
                -0.06598481,
                -0.029210726,
                0.033456173,
                0.018875446,
                -0.02357932,
                0.008938496,
            ],
            [
                -0.091959886,
                -0.00057746813,
                -0.043950364,
                -0.038818058,
                -0.032924544,
                -0.034529757,
                -0.016830014,
                -0.079868846,
                0.0215298,
            ],
            [
                -0.13154328,
                -0.035846047,
                0.043392044,
                -0.05072221,
                0.015720155,
                -0.03877134,
                0.054019123,
                -0.10309198,
                -0.026266888,
            ],
        ];
        assert_parity(&parity_updates(&NovoGrad::new(0.1)), &NOVOGRAD);
        assert_parity(&parity_updates(&NovoGrad::new(0.1).with_weight_decay(0.01).unwrap()), &NOVOGRAD_WEIGHT_DECAY);
    }
}
