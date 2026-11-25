//! The Adam family of optimizers, which rescale gradients by estimates of their moments.
//!
//! [`Adam`] covers Adam, NAdam, AdamW, and NAdamW, and [`AdamW`] is the same chain with the weight decay of
//! `optax.adamw` by default. [`Amsgrad`], [`Adamax`], [`Radam`], [`Yogi`], [`AdaBelief`], and [`Adan`] are chains of
//! the variants of [`ScaleByAdam`] in the [`adam`](crate::transformations::adam) transformations module with a learning
//! rate, and [`OptimisticAdam`] adds an optimistic gradient step for min-max problems. Their states are named structs
//! rather than nested tuples of the states of their components, so that checkpoints refer to them by stable paths that
//! follow the parameter structure (e.g., `$.adam.first_moments.weights` for the parameter at `$.weights`).
//!
//! AdamW is Adam with decoupled weight decay of [Loshchilov and Hutter (2019)](https://arxiv.org/abs/1711.05101).
//! Unlike L2 regularization, which adds the decay to the gradients before the adaptive rescaling of Adam, decoupled
//! weight decay adds it to the rescaled updates, so that every decaying parameter shrinks at the same relative rate
//! regardless of its gradient statistics.

use ryft_core::{DataType, Parameter, ParameterPath, Parameterized};

use crate::macros::parameterized_state;
use crate::optimizers::OptionalWeightDecay;
use crate::{
    AddDecayedWeights, Optimizer, OptimizerError, OptimizerValue, ScaleByAdam, ScaleByAdamState, ScaleByAdamax,
    ScaleByAdan, ScaleByAdanState, ScaleByAmsgrad, ScaleByAmsgradState, ScaleByBelief, ScaleByBeliefState,
    ScaleByOptimisticGradient, ScaleByOptimisticGradientState, ScaleByRadam, ScaleBySchedule, ScaleByScheduleState,
    ScaleByYogi, Schedule, UpdateArguments,
};

// TODO(eaplatanios): Review this module.

parameterized_state! {
    /// State of [`AdamW`] for the parameter structure `S`, which holds the state of its Adam moment estimates and the
    /// step count of its learning rate schedule. Weight decay is stateless.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct AdamWState<P, S> with family AdamWStateFamily {
        /// State of the [`ScaleByAdam`] transformation of the optimizer.
        adam: ScaleByAdamState<P, S>,

        /// State of the [`ScaleBySchedule`] transformation that applies the learning rate.
        learning_rate: ScaleByScheduleState<P>,
    }
}

impl<P: Parameter, S> AdamWState<P, S> {
    /// Returns the state of the [`ScaleByAdam`] transformation of the optimizer.
    #[inline]
    pub fn adam(&self) -> &ScaleByAdamState<P, S> {
        &self.adam
    }

    /// Returns the state of the [`ScaleBySchedule`] transformation that applies the learning rate.
    #[inline]
    pub fn learning_rate(&self) -> &ScaleByScheduleState<P> {
        &self.learning_rate
    }
}

/// Adam optimizer with the decoupled weight decay of [Loshchilov and Hutter (2019)](https://arxiv.org/abs/1711.05101),
/// which adds the decay to the rescaled gradients instead of to the gradients, and which is the analogue of
/// [`optax.adamw`](https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.adamw). Its update is
///
/// ```text
/// mₜ = β₁ · mₜ₋₁ + (1 − β₁) · gₜ
/// vₜ = β₂ · vₜ₋₁ + (1 − β₂) · gₜ²
/// m̂ₜ = mₜ / (1 − β₁ᵗ)
/// v̂ₜ = vₜ / (1 − β₂ᵗ)
/// uₜ = −ηₜ · (m̂ₜ / (√(v̂ₜ + ε̄) + ε) + λ · θₜ₋₁)
/// ```
///
/// With Nesterov momentum, the first moment estimate is instead
///
/// ```text
/// m̂ₜ = β₁ · mₜ / (1 − β₁ᵗ⁺¹) + (1 − β₁) · gₜ / (1 − β₁ᵗ)
/// ```
///
/// Here `t ≥ 1` counts the updates including the current one, `gₜ` is the gradient at the parameters `θₜ₋₁`, `uₜ` is
/// the update that is added to them (i.e., `θₜ = θₜ₋₁ + uₜ`), and `ηₜ` is the learning rate [`Schedule`] evaluated at
/// the `t − 1` updates applied before the current one. The moments `mₜ` and `vₜ` start at zero and decay with the rates
/// `β₁` and `β₂`, `m̂ₜ` and `v̂ₜ` are their bias corrected estimates, `ε` and `ε̄` are added outside and inside the
/// square root for numerical stability, and `λ` is the rate of the decoupled weight decay. All operations, including
/// the squares and the square root, are element-wise. The defaults are `β₁ = 0.9`, `β₂ = 0.999`, `ε = 1e-8`, `ε̄ = 0`,
/// and `λ = 1e-4`, without Nesterov momentum. [`Self::with_nesterov`] selects the Nesterov estimate of the second
/// block, which turns this optimizer into NAdamW, and [`Self::with_weight_decay_mask`] restricts the weight decay to
/// the leaves that the mask selects (i.e., `λ = 0` for all other leaves), while every leaf decays by default.
/// [`Self::with_first_moment_data_type`] only changes the data type in which `mₜ` is stored.
///
/// It is the chain of [`ScaleByAdam`], [`AddDecayedWeights`], and [`ScaleBySchedule::learning_rate`], which is also
/// what [`Adam`] builds once its weight decay is set.
///
/// # References
///
///   - Loshchilov and Hutter (2019). [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101).
///   - Kingma and Ba (2015). [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980).
///   - Dozat (2016). Incorporating Nesterov Momentum into Adam.
#[derive(Clone, Debug)]
pub struct AdamW<Sc: Schedule> {
    /// Transformation that rescales the gradients by their moment estimates.
    adam: ScaleByAdam,

    /// Transformation that adds the decayed weights to the rescaled gradients.
    weight_decay: AddDecayedWeights,

    /// Transformation that scales the updates by the negated learning rate.
    learning_rate: ScaleBySchedule<Sc>,
}

impl<Sc: Schedule> AdamW<Sc> {
    /// Creates a new [`AdamW`] with the learning rate schedule `learning_rate` and default values for all other
    /// hyperparameters. A constant learning rate is passed as an `f64` (e.g., `AdamW::new(1e-3)`).
    pub fn new(learning_rate: Sc) -> Self {
        Self {
            adam: ScaleByAdam::new(),
            weight_decay: AddDecayedWeights::new(1e-4).unwrap(),
            learning_rate: ScaleBySchedule::learning_rate(learning_rate),
        }
    }

    /// Returns this optimizer with `beta1` as the decay rate of the first moment estimates.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `beta1` is not in `[0, 1)`.
    pub fn with_beta1(mut self, beta1: f64) -> Result<Self, OptimizerError> {
        self.adam = self.adam.with_beta1(beta1)?;
        Ok(self)
    }

    /// Returns this optimizer with `beta2` as the decay rate of the second moment estimates.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `beta2` is not in `[0, 1)`.
    pub fn with_beta2(mut self, beta2: f64) -> Result<Self, OptimizerError> {
        self.adam = self.adam.with_beta2(beta2)?;
        Ok(self)
    }

    /// Returns this optimizer with `epsilon` as the term that is added to the square root of the second moment
    /// estimates.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `epsilon` is negative or not finite.
    pub fn with_epsilon(mut self, epsilon: f64) -> Result<Self, OptimizerError> {
        self.adam = self.adam.with_epsilon(epsilon)?;
        Ok(self)
    }

    /// Returns this optimizer with `epsilon_root` as the term that is added to the second moment estimates inside the
    /// square root.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `epsilon_root` is negative or not finite.
    pub fn with_epsilon_root(mut self, epsilon_root: f64) -> Result<Self, OptimizerError> {
        self.adam = self.adam.with_epsilon_root(epsilon_root)?;
        Ok(self)
    }

    /// Returns this optimizer with its first moments stored in `data_type`.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `data_type` is not a floating-point data type.
    pub fn with_first_moment_data_type(mut self, data_type: DataType) -> Result<Self, OptimizerError> {
        self.adam = self.adam.with_first_moment_data_type(data_type)?;
        Ok(self)
    }

    /// Returns this optimizer with Nesterov momentum when `nesterov` is `true`, which turns it into NAdamW (i.e.,
    /// [`optax.nadamw`](https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.nadamw)).
    #[inline]
    pub fn with_nesterov(mut self, nesterov: bool) -> Self {
        self.adam = self.adam.with_nesterov(nesterov);
        self
    }

    /// Returns this optimizer with `weight_decay` as the rate of its decoupled weight decay, keeping its mask.
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

    /// Returns the [`ScaleByAdam`] transformation of this optimizer, which holds its moment hyperparameters.
    #[inline]
    pub fn adam(&self) -> &ScaleByAdam {
        &self.adam
    }

    /// Returns the [`AddDecayedWeights`] transformation of this optimizer, which holds its weight decay rate and mask.
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

impl<P: OptimizerValue, Sc: Schedule> Optimizer<P> for AdamW<Sc> {
    type State<S: Parameterized<P>> = AdamWState<P, S>;

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        let adam = self.adam.initialize(parameters)?;
        let learning_rate = self.learning_rate.initialize(parameters)?;
        Ok(AdamWState { adam, learning_rate })
    }

    fn update<S: Parameterized<P>>(
        &self,
        gradients: S,
        state: Self::State<S>,
        parameters: &S,
        arguments: &UpdateArguments<'_, P, S>,
    ) -> Result<(S, Self::State<S>), OptimizerError> {
        let (updates, adam) = self.adam.update(gradients, state.adam, parameters, arguments)?;
        let (updates, ()) = self.weight_decay.update(updates, (), parameters, arguments)?;
        let (updates, learning_rate) =
            self.learning_rate.update(updates, state.learning_rate, parameters, arguments)?;
        Ok((updates, AdamWState { adam, learning_rate }))
    }
}

parameterized_state! {
    /// State of [`Adam`] for the parameter structure `S`, which holds the state of its Adam moment estimates and the
    /// step count of its learning rate schedule. Weight decay is stateless.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct AdamState<P, S> with family AdamStateFamily {
        /// State of the [`ScaleByAdam`] transformation of the optimizer.
        adam: ScaleByAdamState<P, S>,

        /// State of the [`ScaleBySchedule`] transformation that applies the learning rate.
        learning_rate: ScaleByScheduleState<P>,
    }
}

impl<P: Parameter, S> AdamState<P, S> {
    /// Returns the state of the [`ScaleByAdam`] transformation of the optimizer.
    #[inline]
    pub fn adam(&self) -> &ScaleByAdamState<P, S> {
        &self.adam
    }

    /// Returns the state of the [`ScaleBySchedule`] transformation that applies the learning rate.
    #[inline]
    pub fn learning_rate(&self) -> &ScaleByScheduleState<P> {
        &self.learning_rate
    }
}

/// Adam optimizer of [Kingma and Ba (2015)](https://arxiv.org/abs/1412.6980), which rescales the gradients by bias
/// corrected estimates of their first and second moments and which covers
/// [`optax.adam`](https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.adam),
/// [`optax.nadam`](https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.nadam),
/// [`optax.adamw`](https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.adamw), and
/// [`optax.nadamw`](https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.nadamw). Its update is
///
/// ```text
/// mₜ = β₁ · mₜ₋₁ + (1 − β₁) · gₜ
/// vₜ = β₂ · vₜ₋₁ + (1 − β₂) · gₜ²
/// m̂ₜ = mₜ / (1 − β₁ᵗ)
/// v̂ₜ = vₜ / (1 − β₂ᵗ)
/// uₜ = −ηₜ · (m̂ₜ / (√(v̂ₜ + ε̄) + ε) + λ · θₜ₋₁)
/// ```
///
/// With Nesterov momentum, the first moment estimate is instead
///
/// ```text
/// m̂ₜ = β₁ · mₜ / (1 − β₁ᵗ⁺¹) + (1 − β₁) · gₜ / (1 − β₁ᵗ)
/// ```
///
/// Here `t ≥ 1` counts the updates including the current one, `gₜ` is the gradient at the parameters `θₜ₋₁`, `uₜ` is
/// the update that is added to them (i.e., `θₜ = θₜ₋₁ + uₜ`), and `ηₜ` is the learning rate [`Schedule`] evaluated at
/// the `t − 1` updates applied before the current one. The moments `mₜ` and `vₜ` start at zero and decay with the rates
/// `β₁` and `β₂`, `m̂ₜ` and `v̂ₜ` are their bias corrected estimates, `ε` and `ε̄` are added outside and inside the
/// square root for numerical stability, and `λ` is the rate of the decoupled weight decay. All operations, including
/// the squares and the square root, are element-wise. The defaults are `β₁ = 0.9`, `β₂ = 0.999`, `ε = 1e-8`, and
/// `ε̄ = 0`, without Nesterov momentum and without weight decay (i.e., `λ = 0`). [`Self::with_nesterov`] selects the
/// Nesterov estimate of the second block, which is the NAdam of Dozat (2016). [`Self::with_weight_decay`] sets `λ`,
/// which turns Adam into AdamW (or NAdam into NAdamW), and [`Self::with_weight_decay_mask`] restricts the weight decay
/// to the leaves that the mask selects. [`Self::with_first_moment_data_type`] only changes the data type in which `mₜ`
/// is stored.
///
/// It is the chain of [`ScaleByAdam`], [`AddDecayedWeights`] (only when a weight decay is set), and
/// [`ScaleBySchedule::learning_rate`]. [`AdamW`] is the same chain with the weight decay `1e-4` of `optax.adamw` by
/// default.
///
/// # References
///
///   - Kingma and Ba (2015). [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980).
///   - Dozat (2016). Incorporating Nesterov Momentum into Adam.
///   - Loshchilov and Hutter (2019). [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101).
#[derive(Clone, Debug)]
pub struct Adam<Sc: Schedule> {
    /// Transformation that rescales the gradients by their moment estimates.
    adam: ScaleByAdam,

    /// Transformation that adds the decayed weights to the rescaled gradients, if the optimizer decays its weights.
    weight_decay: OptionalWeightDecay,

    /// Transformation that scales the updates by the negated learning rate.
    learning_rate: ScaleBySchedule<Sc>,
}

impl<Sc: Schedule> Adam<Sc> {
    /// Creates a new [`Adam`] with the learning rate schedule `learning_rate` (e.g., an `f64` for a constant learning
    /// rate) and default values for all other hyperparameters.
    pub fn new(learning_rate: Sc) -> Self {
        Self {
            adam: ScaleByAdam::new(),
            weight_decay: OptionalWeightDecay::new(),
            learning_rate: ScaleBySchedule::learning_rate(learning_rate),
        }
    }

    /// Returns this optimizer with `beta1` as the decay rate of the first moment estimates.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `beta1` is not in `[0, 1)`.
    pub fn with_beta1(mut self, beta1: f64) -> Result<Self, OptimizerError> {
        self.adam = self.adam.with_beta1(beta1)?;
        Ok(self)
    }

    /// Returns this optimizer with `beta2` as the decay rate of the second moment estimates.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `beta2` is not in `[0, 1)`.
    pub fn with_beta2(mut self, beta2: f64) -> Result<Self, OptimizerError> {
        self.adam = self.adam.with_beta2(beta2)?;
        Ok(self)
    }

    /// Returns this optimizer with `epsilon` as the term that is added to the square root of the second moment
    /// estimates.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `epsilon` is negative or not finite.
    pub fn with_epsilon(mut self, epsilon: f64) -> Result<Self, OptimizerError> {
        self.adam = self.adam.with_epsilon(epsilon)?;
        Ok(self)
    }

    /// Returns this optimizer with `epsilon_root` as the term that is added to the second moment estimates inside the
    /// square root.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `epsilon_root` is negative or not finite.
    pub fn with_epsilon_root(mut self, epsilon_root: f64) -> Result<Self, OptimizerError> {
        self.adam = self.adam.with_epsilon_root(epsilon_root)?;
        Ok(self)
    }

    /// Returns this optimizer with its first moments stored in `data_type`.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `data_type` is not a floating-point data type.
    pub fn with_first_moment_data_type(mut self, data_type: DataType) -> Result<Self, OptimizerError> {
        self.adam = self.adam.with_first_moment_data_type(data_type)?;
        Ok(self)
    }

    /// Returns this optimizer with Nesterov momentum when `nesterov` is `true`, which turns it into NAdam (or NAdamW).
    #[inline]
    pub fn with_nesterov(mut self, nesterov: bool) -> Self {
        self.adam = self.adam.with_nesterov(nesterov);
        self
    }

    /// Returns this optimizer with `weight_decay` as the rate of its decoupled weight decay, keeping its mask.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `weight_decay` is not finite.
    pub fn with_weight_decay(mut self, weight_decay: f64) -> Result<Self, OptimizerError> {
        self.weight_decay = self.weight_decay.with_rate(weight_decay)?;
        Ok(self)
    }

    /// Returns this optimizer with `mask` as the predicate that selects the leaves that decay by their
    /// [`ParameterPath`]s (refer to [`AddDecayedWeights::with_mask`]). It applies once a weight decay is set, whether
    /// before or after this mask.
    pub fn with_weight_decay_mask<Mask>(mut self, mask: Mask) -> Self
    where
        Mask: 'static + Fn(&ParameterPath) -> bool + Send + Sync,
    {
        self.weight_decay = self.weight_decay.with_mask(mask);
        self
    }

    /// Returns the [`ScaleByAdam`] transformation of this optimizer, which holds its moment hyperparameters.
    #[inline]
    pub fn adam(&self) -> &ScaleByAdam {
        &self.adam
    }

    /// Returns the [`AddDecayedWeights`] transformation of this optimizer, if it decays its weights.
    #[inline]
    pub fn weight_decay(&self) -> Option<&AddDecayedWeights> {
        self.weight_decay.transformation()
    }

    /// Returns the learning rate [`Schedule`] of this optimizer.
    #[inline]
    pub fn learning_rate(&self) -> &Sc {
        self.learning_rate.schedule()
    }
}

impl<P: OptimizerValue, Sc: Schedule> Optimizer<P> for Adam<Sc> {
    type State<S: Parameterized<P>> = AdamState<P, S>;

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        let adam = self.adam.initialize(parameters)?;
        let learning_rate = self.learning_rate.initialize(parameters)?;
        Ok(AdamState { adam, learning_rate })
    }

    fn update<S: Parameterized<P>>(
        &self,
        gradients: S,
        state: Self::State<S>,
        parameters: &S,
        arguments: &UpdateArguments<'_, P, S>,
    ) -> Result<(S, Self::State<S>), OptimizerError> {
        let (updates, adam) = self.adam.update(gradients, state.adam, parameters, arguments)?;
        let updates = self.weight_decay.apply(updates, parameters, arguments)?;
        let (updates, learning_rate) =
            self.learning_rate.update(updates, state.learning_rate, parameters, arguments)?;
        Ok((updates, AdamState { adam, learning_rate }))
    }
}

parameterized_state! {
    /// State of [`Amsgrad`] for the parameter structure `S`, which holds the state of its [`ScaleByAmsgrad`]
    /// transformation and the step count of its learning rate schedule.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct AmsgradState<P, S> with family AmsgradStateFamily {
        /// State of the [`ScaleByAmsgrad`] transformation of the optimizer.
        amsgrad: ScaleByAmsgradState<P, S>,

        /// State of the [`ScaleBySchedule`] transformation that applies the learning rate.
        learning_rate: ScaleByScheduleState<P>,
    }
}

impl<P: Parameter, S> AmsgradState<P, S> {
    /// Returns the state of the [`ScaleByAmsgrad`] transformation of the optimizer.
    #[inline]
    pub fn amsgrad(&self) -> &ScaleByAmsgradState<P, S> {
        &self.amsgrad
    }

    /// Returns the state of the [`ScaleBySchedule`] transformation that applies the learning rate.
    #[inline]
    pub fn learning_rate(&self) -> &ScaleByScheduleState<P> {
        &self.learning_rate
    }
}

/// AMSGrad optimizer of [Reddi et al. (2018)](https://arxiv.org/abs/1904.09237), which is Adam with a denominator that
/// never decreases, and which is the analogue of
/// [`optax.amsgrad`](https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.amsgrad). Its update is
///
/// ```text
/// mₜ    = β₁ · mₜ₋₁ + (1 − β₁) · gₜ
/// vₜ    = β₂ · vₜ₋₁ + (1 − β₂) · gₜ²
/// m̂ₜ    = mₜ / (1 − β₁ᵗ)
/// v̂ₜ    = vₜ / (1 − β₂ᵗ)
/// v̂ₜᵐᵃˣ = max(v̂ₜ₋₁ᵐᵃˣ, v̂ₜ)
/// uₜ    = −ηₜ · m̂ₜ / (√(v̂ₜᵐᵃˣ + ε̄) + ε)
/// ```
///
/// Here `t ≥ 1` counts the updates including the current one, `gₜ` is the gradient at the parameters `θₜ₋₁`, `uₜ` is
/// the update that is added to them (i.e., `θₜ = θₜ₋₁ + uₜ`), and `ηₜ` is the learning rate [`Schedule`] evaluated at
/// the `t − 1` updates applied before the current one. The moments `mₜ` and `vₜ` and the running maximum `v̂ₜᵐᵃˣ` start
/// at zero, `β₁` and `β₂` are the decay rates of the moments, `m̂ₜ` and `v̂ₜ` are their bias corrected estimates, and
/// `ε` and `ε̄` are added outside and inside the square root for numerical stability. All operations are element-wise.
/// The defaults are `β₁ = 0.9`, `β₂ = 0.999`, `ε = 1e-8`, and `ε̄ = 0`, with both bias corrections.
/// [`Self::with_first_moment_bias_correction`] and [`Self::with_second_moment_bias_correction`] disable the corrections
/// (i.e., `m̂ₜ = mₜ` or `v̂ₜ = vₜ`). Note that the running maximum is taken over the bias corrected estimates by
/// default, while Reddi et al. (2018) take it over the uncorrected `vₜ` without bias corrections, which disabling both
/// corrections reproduces.
///
/// It is the chain of [`ScaleByAmsgrad`] and [`ScaleBySchedule::learning_rate`].
///
/// # References
///
///   - Reddi, Kale, and Kumar (2018). [On the Convergence of Adam and Beyond](https://arxiv.org/abs/1904.09237).
///   - Kingma and Ba (2015). [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980).
#[derive(Clone, Debug)]
pub struct Amsgrad<Sc: Schedule> {
    /// Transformation that rescales the gradients.
    amsgrad: ScaleByAmsgrad,

    /// Transformation that scales the updates by the negated learning rate.
    learning_rate: ScaleBySchedule<Sc>,
}

impl<Sc: Schedule> Amsgrad<Sc> {
    /// Creates a new [`Amsgrad`] with the learning rate schedule `learning_rate` (e.g., an `f64` for a constant
    /// learning rate) and default values for all other hyperparameters.
    pub fn new(learning_rate: Sc) -> Self {
        Self { amsgrad: ScaleByAmsgrad::new(), learning_rate: ScaleBySchedule::learning_rate(learning_rate) }
    }

    /// Returns this optimizer with `beta1` as the decay rate of the first moment estimates (refer to
    /// [`ScaleByAmsgrad::with_beta1`]).
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `beta1` is not in `[0, 1)`.
    pub fn with_beta1(mut self, beta1: f64) -> Result<Self, OptimizerError> {
        self.amsgrad = self.amsgrad.with_beta1(beta1)?;
        Ok(self)
    }

    /// Returns this optimizer with `beta2` as the decay rate of the second moment estimates (refer to
    /// [`ScaleByAmsgrad::with_beta2`]).
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `beta2` is not in `[0, 1)`.
    pub fn with_beta2(mut self, beta2: f64) -> Result<Self, OptimizerError> {
        self.amsgrad = self.amsgrad.with_beta2(beta2)?;
        Ok(self)
    }

    /// Returns this optimizer with `epsilon` as the term that is added for numerical stability outside of square roots
    /// (refer to [`ScaleByAmsgrad::with_epsilon`]).
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `epsilon` is negative or not finite.
    pub fn with_epsilon(mut self, epsilon: f64) -> Result<Self, OptimizerError> {
        self.amsgrad = self.amsgrad.with_epsilon(epsilon)?;
        Ok(self)
    }

    /// Returns this optimizer with `epsilon_root` as the term that is added inside square roots (refer to
    /// [`ScaleByAmsgrad::with_epsilon_root`]).
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `epsilon_root` is negative or not finite.
    pub fn with_epsilon_root(mut self, epsilon_root: f64) -> Result<Self, OptimizerError> {
        self.amsgrad = self.amsgrad.with_epsilon_root(epsilon_root)?;
        Ok(self)
    }

    /// Returns this optimizer with its first moments stored in `data_type` (refer to
    /// [`ScaleByAmsgrad::with_first_moment_data_type`]).
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `data_type` is not a floating-point data type.
    pub fn with_first_moment_data_type(mut self, data_type: DataType) -> Result<Self, OptimizerError> {
        self.amsgrad = self.amsgrad.with_first_moment_data_type(data_type)?;
        Ok(self)
    }

    /// Returns this optimizer with the bias correction of its first moments enabled when `enabled` is `true` (refer to
    /// [`ScaleByAmsgrad::with_first_moment_bias_correction`]).
    #[inline]
    pub fn with_first_moment_bias_correction(mut self, enabled: bool) -> Self {
        self.amsgrad = self.amsgrad.with_first_moment_bias_correction(enabled);
        self
    }

    /// Returns this optimizer with the bias correction of its second moments enabled when `enabled` is `true` (refer to
    /// [`ScaleByAmsgrad::with_second_moment_bias_correction`]).
    #[inline]
    pub fn with_second_moment_bias_correction(mut self, enabled: bool) -> Self {
        self.amsgrad = self.amsgrad.with_second_moment_bias_correction(enabled);
        self
    }

    /// Returns the [`ScaleByAmsgrad`] transformation of this optimizer, which holds its rescaling hyperparameters.
    #[inline]
    pub fn amsgrad(&self) -> &ScaleByAmsgrad {
        &self.amsgrad
    }

    /// Returns the learning rate [`Schedule`] of this optimizer.
    #[inline]
    pub fn learning_rate(&self) -> &Sc {
        self.learning_rate.schedule()
    }
}

impl<P: OptimizerValue, Sc: Schedule> Optimizer<P> for Amsgrad<Sc> {
    type State<S: Parameterized<P>> = AmsgradState<P, S>;

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        let amsgrad = self.amsgrad.initialize(parameters)?;
        let learning_rate = self.learning_rate.initialize(parameters)?;
        Ok(AmsgradState { amsgrad, learning_rate })
    }

    fn update<S: Parameterized<P>>(
        &self,
        gradients: S,
        state: Self::State<S>,
        parameters: &S,
        arguments: &UpdateArguments<'_, P, S>,
    ) -> Result<(S, Self::State<S>), OptimizerError> {
        let (updates, amsgrad) = self.amsgrad.update(gradients, state.amsgrad, parameters, arguments)?;
        let (updates, learning_rate) =
            self.learning_rate.update(updates, state.learning_rate, parameters, arguments)?;
        Ok((updates, AmsgradState { amsgrad, learning_rate }))
    }
}

parameterized_state! {
    /// State of [`Adamax`] for the parameter structure `S`, which holds the state of its [`ScaleByAdamax`]
    /// transformation and the step count of its learning rate schedule. Weight decay is stateless.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct AdamaxState<P, S> with family AdamaxStateFamily {
        /// State of the [`ScaleByAdamax`] transformation of the optimizer.
        adamax: ScaleByAdamState<P, S>,

        /// State of the [`ScaleBySchedule`] transformation that applies the learning rate.
        learning_rate: ScaleByScheduleState<P>,
    }
}

impl<P: Parameter, S> AdamaxState<P, S> {
    /// Returns the state of the [`ScaleByAdamax`] transformation of the optimizer.
    #[inline]
    pub fn adamax(&self) -> &ScaleByAdamState<P, S> {
        &self.adamax
    }

    /// Returns the state of the [`ScaleBySchedule`] transformation that applies the learning rate.
    #[inline]
    pub fn learning_rate(&self) -> &ScaleByScheduleState<P> {
        &self.learning_rate
    }
}

/// Adamax optimizer of [Kingma and Ba (2015)](https://arxiv.org/abs/1412.6980), which is the variant of Adam that
/// divides by an exponentially weighted infinity norm of the gradients, and which covers
/// [`optax.adamax`](https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.adamax) and
/// [`optax.adamaxw`](https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.adamaxw). Its update is
///
/// ```text
/// mₜ = β₁ · mₜ₋₁ + (1 − β₁) · gₜ
/// vₜ = max(|gₜ| + ε, β₂ · vₜ₋₁)
/// uₜ = −ηₜ · ((mₜ / (1 − β₁ᵗ)) / vₜ + λ · θₜ₋₁)
/// ```
///
/// Here `t ≥ 1` counts the updates including the current one, `gₜ` is the gradient at the parameters `θₜ₋₁`, `uₜ` is
/// the update that is added to them (i.e., `θₜ = θₜ₋₁ + uₜ`), and `ηₜ` is the learning rate [`Schedule`] evaluated at
/// the `t − 1` updates applied before the current one. The first moment `mₜ` and the infinity norm estimate `vₜ` start
/// at zero, `β₁` and `β₂` are their decay rates, `ε` keeps the denominator positive, and `λ` is the rate of the
/// decoupled weight decay. All operations are element-wise. The defaults are `β₁ = 0.9`, `β₂ = 0.999`, and `ε = 1e-8`,
/// without weight decay (i.e., `λ = 0`). [`Self::with_weight_decay`] sets `λ`, which turns this optimizer into AdamaxW
/// (whose default weight decay in Optax is `1e-4`), and [`Self::with_weight_decay_mask`] restricts the weight decay to
/// the leaves that the mask selects. Note that Kingma and Ba (2015) take the maximum of `|gₜ|` and `β₂ · vₜ₋₁` without
/// adding `ε`.
///
/// It is the chain of [`ScaleByAdamax`], [`AddDecayedWeights`] (only when a weight decay is set), and
/// [`ScaleBySchedule::learning_rate`].
///
/// # References
///
///   - Kingma and Ba (2015). [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980).
///   - Loshchilov and Hutter (2019). [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101).
#[derive(Clone, Debug)]
pub struct Adamax<Sc: Schedule> {
    /// Transformation that rescales the gradients.
    adamax: ScaleByAdamax,

    /// Transformation that adds the decayed weights to the rescaled gradients, if the optimizer decays its weights.
    weight_decay: OptionalWeightDecay,

    /// Transformation that scales the updates by the negated learning rate.
    learning_rate: ScaleBySchedule<Sc>,
}

impl<Sc: Schedule> Adamax<Sc> {
    /// Creates a new [`Adamax`] with the learning rate schedule `learning_rate` (e.g., an `f64` for a constant learning
    /// rate) and default values for all other hyperparameters.
    pub fn new(learning_rate: Sc) -> Self {
        Self {
            adamax: ScaleByAdamax::new(),
            weight_decay: OptionalWeightDecay::new(),
            learning_rate: ScaleBySchedule::learning_rate(learning_rate),
        }
    }

    /// Returns this optimizer with `beta1` as the decay rate of the first moment estimates (refer to
    /// [`ScaleByAdamax::with_beta1`]).
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `beta1` is not in `[0, 1)`.
    pub fn with_beta1(mut self, beta1: f64) -> Result<Self, OptimizerError> {
        self.adamax = self.adamax.with_beta1(beta1)?;
        Ok(self)
    }

    /// Returns this optimizer with `beta2` as the decay rate of the second moment estimates (refer to
    /// [`ScaleByAdamax::with_beta2`]).
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `beta2` is not in `[0, 1)`.
    pub fn with_beta2(mut self, beta2: f64) -> Result<Self, OptimizerError> {
        self.adamax = self.adamax.with_beta2(beta2)?;
        Ok(self)
    }

    /// Returns this optimizer with `epsilon` as the term that is added for numerical stability outside of square roots
    /// (refer to [`ScaleByAdamax::with_epsilon`]).
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `epsilon` is negative or not finite.
    pub fn with_epsilon(mut self, epsilon: f64) -> Result<Self, OptimizerError> {
        self.adamax = self.adamax.with_epsilon(epsilon)?;
        Ok(self)
    }

    /// Returns this optimizer with `weight_decay` as the rate of its decoupled weight decay, keeping its mask.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `weight_decay` is not finite.
    pub fn with_weight_decay(mut self, weight_decay: f64) -> Result<Self, OptimizerError> {
        self.weight_decay = self.weight_decay.with_rate(weight_decay)?;
        Ok(self)
    }

    /// Returns this optimizer with `mask` as the predicate that selects the leaves that decay by their
    /// [`ParameterPath`]s (refer to [`AddDecayedWeights::with_mask`]). It applies once a weight decay is set, whether
    /// before or after this mask.
    pub fn with_weight_decay_mask<Mask>(mut self, mask: Mask) -> Self
    where
        Mask: 'static + Fn(&ParameterPath) -> bool + Send + Sync,
    {
        self.weight_decay = self.weight_decay.with_mask(mask);
        self
    }

    /// Returns the [`ScaleByAdamax`] transformation of this optimizer, which holds its rescaling hyperparameters.
    #[inline]
    pub fn adamax(&self) -> &ScaleByAdamax {
        &self.adamax
    }

    /// Returns the [`AddDecayedWeights`] transformation of this optimizer, if it decays its weights.
    #[inline]
    pub fn weight_decay(&self) -> Option<&AddDecayedWeights> {
        self.weight_decay.transformation()
    }

    /// Returns the learning rate [`Schedule`] of this optimizer.
    #[inline]
    pub fn learning_rate(&self) -> &Sc {
        self.learning_rate.schedule()
    }
}

impl<P: OptimizerValue, Sc: Schedule> Optimizer<P> for Adamax<Sc> {
    type State<S: Parameterized<P>> = AdamaxState<P, S>;

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        let adamax = self.adamax.initialize(parameters)?;
        let learning_rate = self.learning_rate.initialize(parameters)?;
        Ok(AdamaxState { adamax, learning_rate })
    }

    fn update<S: Parameterized<P>>(
        &self,
        gradients: S,
        state: Self::State<S>,
        parameters: &S,
        arguments: &UpdateArguments<'_, P, S>,
    ) -> Result<(S, Self::State<S>), OptimizerError> {
        let (updates, adamax) = self.adamax.update(gradients, state.adamax, parameters, arguments)?;
        let updates = self.weight_decay.apply(updates, parameters, arguments)?;
        let (updates, learning_rate) =
            self.learning_rate.update(updates, state.learning_rate, parameters, arguments)?;
        Ok((updates, AdamaxState { adamax, learning_rate }))
    }
}

parameterized_state! {
    /// State of [`Radam`] for the parameter structure `S`, which holds the state of its [`ScaleByRadam`] transformation
    /// and the step count of its learning rate schedule.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct RadamState<P, S> with family RadamStateFamily {
        /// State of the [`ScaleByRadam`] transformation of the optimizer.
        radam: ScaleByAdamState<P, S>,

        /// State of the [`ScaleBySchedule`] transformation that applies the learning rate.
        learning_rate: ScaleByScheduleState<P>,
    }
}

impl<P: Parameter, S> RadamState<P, S> {
    /// Returns the state of the [`ScaleByRadam`] transformation of the optimizer.
    #[inline]
    pub fn radam(&self) -> &ScaleByAdamState<P, S> {
        &self.radam
    }

    /// Returns the state of the [`ScaleBySchedule`] transformation that applies the learning rate.
    #[inline]
    pub fn learning_rate(&self) -> &ScaleByScheduleState<P> {
        &self.learning_rate
    }
}

/// RAdam optimizer of [Liu et al. (2020)](https://arxiv.org/abs/1908.03265), which is Adam with a rectification of the
/// variance of its adaptive learning rate that falls back to momentum SGD in the first steps, and which is the analogue
/// of [`optax.radam`](https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.radam). Its update is
///
/// ```text
/// mₜ = β₁ · mₜ₋₁ + (1 − β₁) · gₜ
/// vₜ = β₂ · vₜ₋₁ + (1 − β₂) · gₜ²
/// m̂ₜ = mₜ / (1 − β₁ᵗ)
/// v̂ₜ = vₜ / (1 − β₂ᵗ)
/// ρ∞ = 2 / (1 − β₂) − 1
/// ρₜ = ρ∞ − 2 · t · β₂ᵗ / (1 − β₂ᵗ)
/// rₜ = √((ρₜ − 4) · (ρₜ − 2) · ρ∞ / ((ρ∞ − 4) · (ρ∞ − 2) · ρₜ))
/// uₜ = −ηₜ · rₜ · m̂ₜ / (√(v̂ₜ + ε̄) + ε)   if ρₜ ≥ ρ₀
/// uₜ = −ηₜ · m̂ₜ                          otherwise
/// ```
///
/// Here `t ≥ 1` counts the updates including the current one, `gₜ` is the gradient at the parameters `θₜ₋₁`, `uₜ` is
/// the update that is added to them (i.e., `θₜ = θₜ₋₁ + uₜ`), and `ηₜ` is the learning rate [`Schedule`] evaluated at
/// the `t − 1` updates applied before the current one. The moments `mₜ` and `vₜ` start at zero and decay with the rates
/// `β₁` and `β₂`, `m̂ₜ` and `v̂ₜ` are their bias corrected estimates, `ρₜ` is the length of the approximated simple
/// moving average with its limit `ρ∞`, `rₜ` is the variance rectification, `ρ₀` is the threshold above which the
/// adaptive update is used, and `ε` and `ε̄` are added outside and inside the square root for numerical stability. All
/// operations on `mₜ`, `vₜ`, and `gₜ` are element-wise, while `ρₜ` and `rₜ` are scalars. The defaults are `β₁ = 0.9`,
/// `β₂ = 0.999`, `ε = 1e-8`, `ε̄ = 0`, and `ρ₀ = 5`, without Nesterov momentum. [`Self::with_nesterov`] replaces `m̂ₜ`
/// with the Nesterov estimate `β₁ · mₜ / (1 − β₁ᵗ⁺¹) + (1 − β₁) · gₜ / (1 − β₁ᵗ)` in both cases.
///
/// It is the chain of [`ScaleByRadam`] and [`ScaleBySchedule::learning_rate`].
///
/// # References
///
///   - Liu et al. (2020). [On the Variance of the Adaptive Learning Rate and Beyond](https://arxiv.org/abs/1908.03265).
///   - Kingma and Ba (2015). [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980).
#[derive(Clone, Debug)]
pub struct Radam<Sc: Schedule> {
    /// Transformation that rescales the gradients.
    radam: ScaleByRadam,

    /// Transformation that scales the updates by the negated learning rate.
    learning_rate: ScaleBySchedule<Sc>,
}

impl<Sc: Schedule> Radam<Sc> {
    /// Creates a new [`Radam`] with the learning rate schedule `learning_rate` (e.g., an `f64` for a constant learning
    /// rate) and default values for all other hyperparameters.
    pub fn new(learning_rate: Sc) -> Self {
        Self { radam: ScaleByRadam::new(), learning_rate: ScaleBySchedule::learning_rate(learning_rate) }
    }

    /// Returns this optimizer with `beta1` as the decay rate of the first moment estimates (refer to
    /// [`ScaleByRadam::with_beta1`]).
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `beta1` is not in `[0, 1)`.
    pub fn with_beta1(mut self, beta1: f64) -> Result<Self, OptimizerError> {
        self.radam = self.radam.with_beta1(beta1)?;
        Ok(self)
    }

    /// Returns this optimizer with `beta2` as the decay rate of the second moment estimates (refer to
    /// [`ScaleByRadam::with_beta2`]).
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `beta2` is not in `[0, 1)`.
    pub fn with_beta2(mut self, beta2: f64) -> Result<Self, OptimizerError> {
        self.radam = self.radam.with_beta2(beta2)?;
        Ok(self)
    }

    /// Returns this optimizer with `epsilon` as the term that is added for numerical stability outside of square roots
    /// (refer to [`ScaleByRadam::with_epsilon`]).
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `epsilon` is negative or not finite.
    pub fn with_epsilon(mut self, epsilon: f64) -> Result<Self, OptimizerError> {
        self.radam = self.radam.with_epsilon(epsilon)?;
        Ok(self)
    }

    /// Returns this optimizer with `epsilon_root` as the term that is added inside square roots (refer to
    /// [`ScaleByRadam::with_epsilon_root`]).
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `epsilon_root` is negative or not finite.
    pub fn with_epsilon_root(mut self, epsilon_root: f64) -> Result<Self, OptimizerError> {
        self.radam = self.radam.with_epsilon_root(epsilon_root)?;
        Ok(self)
    }

    /// Returns this optimizer with `threshold` as the smallest length of the approximated simple moving average for
    /// which the adaptive learning rate is used (refer to [`ScaleByRadam::with_threshold`]).
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `threshold` is not finite.
    pub fn with_threshold(mut self, threshold: f64) -> Result<Self, OptimizerError> {
        self.radam = self.radam.with_threshold(threshold)?;
        Ok(self)
    }

    /// Returns this optimizer with Nesterov momentum when `nesterov` is `true` (refer to
    /// [`ScaleByRadam::with_nesterov`]).
    #[inline]
    pub fn with_nesterov(mut self, nesterov: bool) -> Self {
        self.radam = self.radam.with_nesterov(nesterov);
        self
    }

    /// Returns the [`ScaleByRadam`] transformation of this optimizer, which holds its rescaling hyperparameters.
    #[inline]
    pub fn radam(&self) -> &ScaleByRadam {
        &self.radam
    }

    /// Returns the learning rate [`Schedule`] of this optimizer.
    #[inline]
    pub fn learning_rate(&self) -> &Sc {
        self.learning_rate.schedule()
    }
}

impl<P: OptimizerValue, Sc: Schedule> Optimizer<P> for Radam<Sc> {
    type State<S: Parameterized<P>> = RadamState<P, S>;

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        let radam = self.radam.initialize(parameters)?;
        let learning_rate = self.learning_rate.initialize(parameters)?;
        Ok(RadamState { radam, learning_rate })
    }

    fn update<S: Parameterized<P>>(
        &self,
        gradients: S,
        state: Self::State<S>,
        parameters: &S,
        arguments: &UpdateArguments<'_, P, S>,
    ) -> Result<(S, Self::State<S>), OptimizerError> {
        let (updates, radam) = self.radam.update(gradients, state.radam, parameters, arguments)?;
        let (updates, learning_rate) =
            self.learning_rate.update(updates, state.learning_rate, parameters, arguments)?;
        Ok((updates, RadamState { radam, learning_rate }))
    }
}

parameterized_state! {
    /// State of [`Yogi`] for the parameter structure `S`, which holds the state of its [`ScaleByYogi`] transformation
    /// and the step count of its learning rate schedule.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct YogiState<P, S> with family YogiStateFamily {
        /// State of the [`ScaleByYogi`] transformation of the optimizer.
        yogi: ScaleByAdamState<P, S>,

        /// State of the [`ScaleBySchedule`] transformation that applies the learning rate.
        learning_rate: ScaleByScheduleState<P>,
    }
}

impl<P: Parameter, S> YogiState<P, S> {
    /// Returns the state of the [`ScaleByYogi`] transformation of the optimizer.
    #[inline]
    pub fn yogi(&self) -> &ScaleByAdamState<P, S> {
        &self.yogi
    }

    /// Returns the state of the [`ScaleBySchedule`] transformation that applies the learning rate.
    #[inline]
    pub fn learning_rate(&self) -> &ScaleByScheduleState<P> {
        &self.learning_rate
    }
}

/// Yogi optimizer of
/// [Zaheer et al. (2018)](
/// https://proceedings.neurips.cc/paper/2018/hash/90365351ccc7437a1309dc64e4db32a3-Abstract.html),
/// which is Adam with an additive update of the second moments that changes them by at most `(1 − β₂) · gₜ²` per step,
/// and which is the analogue of [`optax.yogi`](https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.yogi).
/// Its update is
///
/// ```text
/// mₜ = β₁ · mₜ₋₁ + (1 − β₁) · gₜ
/// vₜ = vₜ₋₁ − (1 − β₂) · sign(vₜ₋₁ − gₜ²) · gₜ²
/// m̂ₜ = mₜ / (1 − β₁ᵗ)
/// v̂ₜ = vₜ / (1 − β₂ᵗ)
/// uₜ = −ηₜ · m̂ₜ / (√(v̂ₜ + ε̄) + ε)
/// ```
///
/// Here `t ≥ 1` counts the updates including the current one, `gₜ` is the gradient at the parameters `θₜ₋₁`, `uₜ` is
/// the update that is added to them (i.e., `θₜ = θₜ₋₁ + uₜ`), and `ηₜ` is the learning rate [`Schedule`] evaluated at
/// the `t − 1` updates applied before the current one. The first moment `mₜ` starts at zero and the second moment `vₜ`
/// starts at the initial accumulator value `v₀`, `β₁` and `β₂` are their decay rates, `m̂ₜ` and `v̂ₜ` are their bias
/// corrected estimates, and `ε` and `ε̄` are added outside and inside the square root for numerical stability. All
/// operations are element-wise. The defaults are `β₁ = 0.9`, `β₂ = 0.999`, `ε = 1e-3`, `ε̄ = 0`, and `v₀ = 1e-6`.
///
/// It is the chain of [`ScaleByYogi`] and [`ScaleBySchedule::learning_rate`].
///
/// # References
///
///   - Zaheer et al. (2018). [Adaptive Methods for Nonconvex
///     Optimization](https://proceedings.neurips.cc/paper/2018/hash/90365351ccc7437a1309dc64e4db32a3-Abstract.html).
#[derive(Clone, Debug)]
pub struct Yogi<Sc: Schedule> {
    /// Transformation that rescales the gradients.
    yogi: ScaleByYogi,

    /// Transformation that scales the updates by the negated learning rate.
    learning_rate: ScaleBySchedule<Sc>,
}

impl<Sc: Schedule> Yogi<Sc> {
    /// Creates a new [`Yogi`] with the learning rate schedule `learning_rate` (e.g., an `f64` for a constant learning
    /// rate) and default values for all other hyperparameters.
    pub fn new(learning_rate: Sc) -> Self {
        Self { yogi: ScaleByYogi::new(), learning_rate: ScaleBySchedule::learning_rate(learning_rate) }
    }

    /// Returns this optimizer with `beta1` as the decay rate of the first moment estimates (refer to
    /// [`ScaleByYogi::with_beta1`]).
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `beta1` is not in `[0, 1)`.
    pub fn with_beta1(mut self, beta1: f64) -> Result<Self, OptimizerError> {
        self.yogi = self.yogi.with_beta1(beta1)?;
        Ok(self)
    }

    /// Returns this optimizer with `beta2` as the decay rate of the second moment estimates (refer to
    /// [`ScaleByYogi::with_beta2`]).
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `beta2` is not in `[0, 1)`.
    pub fn with_beta2(mut self, beta2: f64) -> Result<Self, OptimizerError> {
        self.yogi = self.yogi.with_beta2(beta2)?;
        Ok(self)
    }

    /// Returns this optimizer with `epsilon` as the term that is added for numerical stability outside of square roots
    /// (refer to [`ScaleByYogi::with_epsilon`]).
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `epsilon` is negative or not finite.
    pub fn with_epsilon(mut self, epsilon: f64) -> Result<Self, OptimizerError> {
        self.yogi = self.yogi.with_epsilon(epsilon)?;
        Ok(self)
    }

    /// Returns this optimizer with `epsilon_root` as the term that is added inside square roots (refer to
    /// [`ScaleByYogi::with_epsilon_root`]).
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `epsilon_root` is negative or not finite.
    pub fn with_epsilon_root(mut self, epsilon_root: f64) -> Result<Self, OptimizerError> {
        self.yogi = self.yogi.with_epsilon_root(epsilon_root)?;
        Ok(self)
    }

    /// Returns this optimizer with `initial_accumulator_value` as the initial value of its accumulators (refer to
    /// [`ScaleByYogi::with_initial_accumulator_value`]).
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `initial_accumulator_value` is negative or not finite.
    pub fn with_initial_accumulator_value(mut self, initial_accumulator_value: f64) -> Result<Self, OptimizerError> {
        self.yogi = self.yogi.with_initial_accumulator_value(initial_accumulator_value)?;
        Ok(self)
    }

    /// Returns the [`ScaleByYogi`] transformation of this optimizer, which holds its rescaling hyperparameters.
    #[inline]
    pub fn yogi(&self) -> &ScaleByYogi {
        &self.yogi
    }

    /// Returns the learning rate [`Schedule`] of this optimizer.
    #[inline]
    pub fn learning_rate(&self) -> &Sc {
        self.learning_rate.schedule()
    }
}

impl<P: OptimizerValue, Sc: Schedule> Optimizer<P> for Yogi<Sc> {
    type State<S: Parameterized<P>> = YogiState<P, S>;

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        let yogi = self.yogi.initialize(parameters)?;
        let learning_rate = self.learning_rate.initialize(parameters)?;
        Ok(YogiState { yogi, learning_rate })
    }

    fn update<S: Parameterized<P>>(
        &self,
        gradients: S,
        state: Self::State<S>,
        parameters: &S,
        arguments: &UpdateArguments<'_, P, S>,
    ) -> Result<(S, Self::State<S>), OptimizerError> {
        let (updates, yogi) = self.yogi.update(gradients, state.yogi, parameters, arguments)?;
        let (updates, learning_rate) =
            self.learning_rate.update(updates, state.learning_rate, parameters, arguments)?;
        Ok((updates, YogiState { yogi, learning_rate }))
    }
}

parameterized_state! {
    /// State of [`AdaBelief`] for the parameter structure `S`, which holds the state of its [`ScaleByBelief`]
    /// transformation and the step count of its learning rate schedule. Weight decay is stateless.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct AdaBeliefState<P, S> with family AdaBeliefStateFamily {
        /// State of the [`ScaleByBelief`] transformation of the optimizer.
        belief: ScaleByBeliefState<P, S>,

        /// State of the [`ScaleBySchedule`] transformation that applies the learning rate.
        learning_rate: ScaleByScheduleState<P>,
    }
}

impl<P: Parameter, S> AdaBeliefState<P, S> {
    /// Returns the state of the [`ScaleByBelief`] transformation of the optimizer.
    #[inline]
    pub fn belief(&self) -> &ScaleByBeliefState<P, S> {
        &self.belief
    }

    /// Returns the state of the [`ScaleBySchedule`] transformation that applies the learning rate.
    #[inline]
    pub fn learning_rate(&self) -> &ScaleByScheduleState<P> {
        &self.learning_rate
    }
}

/// AdaBelief optimizer of [Zhuang et al. (2020)](https://arxiv.org/abs/2010.07468), which is Adam with a denominator
/// that tracks the deviation of the gradients from their moving average (i.e., the belief in the observed gradients)
/// instead of their magnitude, and which is the analogue of
/// [`optax.adabelief`](https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.adabelief). Its update is
///
/// ```text
/// mₜ = β₁ · mₜ₋₁ + (1 − β₁) · gₜ
/// sₜ = β₂ · sₜ₋₁ + (1 − β₂) · (gₜ − mₜ)² + ε̄
/// m̂ₜ = mₜ / (1 − β₁ᵗ)
/// ŝₜ = sₜ / (1 − β₂ᵗ)
/// uₜ = −ηₜ · (m̂ₜ / (√ŝₜ + ε) + λ · θₜ₋₁)
/// ```
///
/// Here `t ≥ 1` counts the updates including the current one, `gₜ` is the gradient at the parameters `θₜ₋₁`, `uₜ` is
/// the update that is added to them (i.e., `θₜ = θₜ₋₁ + uₜ`), and `ηₜ` is the learning rate [`Schedule`] evaluated at
/// the `t − 1` updates applied before the current one. The first moment `mₜ` and the second moment of the prediction
/// errors `sₜ` start at zero and decay with the rates `β₁` and `β₂`, `m̂ₜ` and `ŝₜ` are their bias corrected estimates,
/// `ε̄` is added to `sₜ` (and so accumulates in the state) and `ε` is added outside the square root for numerical
/// stability, and `λ` is the rate of the decoupled weight decay. All operations are element-wise. The defaults are
/// `β₁ = 0.9`, `β₂ = 0.999`, `ε = 1e-16`, and `ε̄ = 1e-16`, without Nesterov momentum and without weight decay (i.e.,
/// `λ = 0`). [`Self::with_nesterov`] replaces `m̂ₜ` with the Nesterov estimate
/// `β₁ · mₜ / (1 − β₁ᵗ⁺¹) + (1 − β₁) · gₜ / (1 − β₁ᵗ)`, [`Self::with_weight_decay`] sets `λ`, and
/// [`Self::with_weight_decay_mask`] restricts the weight decay to the leaves that the mask selects. Note that the
/// defaults of `ε` and `ε̄` are those of Optax, whose documentation states that Zhuang et al. (2020) use `ε = 1e-8` and
/// `ε̄ = 0`.
///
/// It is the chain of [`ScaleByBelief`], [`AddDecayedWeights`] (only when a weight decay is set), and
/// [`ScaleBySchedule::learning_rate`].
///
/// # References
///
///   - Zhuang et al. (2020). [AdaBelief Optimizer: Adapting Stepsizes by the Belief in Observed
///     Gradients](https://arxiv.org/abs/2010.07468).
#[derive(Clone, Debug)]
pub struct AdaBelief<Sc: Schedule> {
    /// Transformation that rescales the gradients.
    belief: ScaleByBelief,

    /// Transformation that adds the decayed weights to the rescaled gradients, if the optimizer decays its weights.
    weight_decay: OptionalWeightDecay,

    /// Transformation that scales the updates by the negated learning rate.
    learning_rate: ScaleBySchedule<Sc>,
}

impl<Sc: Schedule> AdaBelief<Sc> {
    /// Creates a new [`AdaBelief`] with the learning rate schedule `learning_rate` (e.g., an `f64` for a constant
    /// learning rate) and default values for all other hyperparameters.
    pub fn new(learning_rate: Sc) -> Self {
        Self {
            belief: ScaleByBelief::new(),
            weight_decay: OptionalWeightDecay::new(),
            learning_rate: ScaleBySchedule::learning_rate(learning_rate),
        }
    }

    /// Returns this optimizer with `beta1` as the decay rate of the first moment estimates (refer to
    /// [`ScaleByBelief::with_beta1`]).
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `beta1` is not in `[0, 1)`.
    pub fn with_beta1(mut self, beta1: f64) -> Result<Self, OptimizerError> {
        self.belief = self.belief.with_beta1(beta1)?;
        Ok(self)
    }

    /// Returns this optimizer with `beta2` as the decay rate of the second moment estimates (refer to
    /// [`ScaleByBelief::with_beta2`]).
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `beta2` is not in `[0, 1)`.
    pub fn with_beta2(mut self, beta2: f64) -> Result<Self, OptimizerError> {
        self.belief = self.belief.with_beta2(beta2)?;
        Ok(self)
    }

    /// Returns this optimizer with `epsilon` as the term that is added for numerical stability outside of square roots
    /// (refer to [`ScaleByBelief::with_epsilon`]).
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `epsilon` is negative or not finite.
    pub fn with_epsilon(mut self, epsilon: f64) -> Result<Self, OptimizerError> {
        self.belief = self.belief.with_epsilon(epsilon)?;
        Ok(self)
    }

    /// Returns this optimizer with `epsilon_root` as the term that is added inside square roots (refer to
    /// [`ScaleByBelief::with_epsilon_root`]).
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `epsilon_root` is negative or not finite.
    pub fn with_epsilon_root(mut self, epsilon_root: f64) -> Result<Self, OptimizerError> {
        self.belief = self.belief.with_epsilon_root(epsilon_root)?;
        Ok(self)
    }

    /// Returns this optimizer with Nesterov momentum when `nesterov` is `true` (refer to
    /// [`ScaleByBelief::with_nesterov`]).
    #[inline]
    pub fn with_nesterov(mut self, nesterov: bool) -> Self {
        self.belief = self.belief.with_nesterov(nesterov);
        self
    }

    /// Returns this optimizer with `weight_decay` as the rate of its decoupled weight decay, keeping its mask.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `weight_decay` is not finite.
    pub fn with_weight_decay(mut self, weight_decay: f64) -> Result<Self, OptimizerError> {
        self.weight_decay = self.weight_decay.with_rate(weight_decay)?;
        Ok(self)
    }

    /// Returns this optimizer with `mask` as the predicate that selects the leaves that decay by their
    /// [`ParameterPath`]s (refer to [`AddDecayedWeights::with_mask`]). It applies once a weight decay is set, whether
    /// before or after this mask.
    pub fn with_weight_decay_mask<Mask>(mut self, mask: Mask) -> Self
    where
        Mask: 'static + Fn(&ParameterPath) -> bool + Send + Sync,
    {
        self.weight_decay = self.weight_decay.with_mask(mask);
        self
    }

    /// Returns the [`ScaleByBelief`] transformation of this optimizer, which holds its rescaling hyperparameters.
    #[inline]
    pub fn belief(&self) -> &ScaleByBelief {
        &self.belief
    }

    /// Returns the [`AddDecayedWeights`] transformation of this optimizer, if it decays its weights.
    #[inline]
    pub fn weight_decay(&self) -> Option<&AddDecayedWeights> {
        self.weight_decay.transformation()
    }

    /// Returns the learning rate [`Schedule`] of this optimizer.
    #[inline]
    pub fn learning_rate(&self) -> &Sc {
        self.learning_rate.schedule()
    }
}

impl<P: OptimizerValue, Sc: Schedule> Optimizer<P> for AdaBelief<Sc> {
    type State<S: Parameterized<P>> = AdaBeliefState<P, S>;

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        let belief = self.belief.initialize(parameters)?;
        let learning_rate = self.learning_rate.initialize(parameters)?;
        Ok(AdaBeliefState { belief, learning_rate })
    }

    fn update<S: Parameterized<P>>(
        &self,
        gradients: S,
        state: Self::State<S>,
        parameters: &S,
        arguments: &UpdateArguments<'_, P, S>,
    ) -> Result<(S, Self::State<S>), OptimizerError> {
        let (updates, belief) = self.belief.update(gradients, state.belief, parameters, arguments)?;
        let updates = self.weight_decay.apply(updates, parameters, arguments)?;
        let (updates, learning_rate) =
            self.learning_rate.update(updates, state.learning_rate, parameters, arguments)?;
        Ok((updates, AdaBeliefState { belief, learning_rate }))
    }
}

parameterized_state! {
    /// State of [`Adan`] for the parameter structure `S`, which holds the state of its [`ScaleByAdan`] transformation
    /// and the step count of its learning rate schedule. Weight decay is stateless.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct AdanState<P, S> with family AdanStateFamily {
        /// State of the [`ScaleByAdan`] transformation of the optimizer.
        adan: ScaleByAdanState<P, S>,

        /// State of the [`ScaleBySchedule`] transformation that applies the learning rate.
        learning_rate: ScaleByScheduleState<P>,
    }
}

impl<P: Parameter, S> AdanState<P, S> {
    /// Returns the state of the [`ScaleByAdan`] transformation of the optimizer.
    #[inline]
    pub fn adan(&self) -> &ScaleByAdanState<P, S> {
        &self.adan
    }

    /// Returns the state of the [`ScaleBySchedule`] transformation that applies the learning rate.
    #[inline]
    pub fn learning_rate(&self) -> &ScaleByScheduleState<P> {
        &self.learning_rate
    }
}

/// Adan optimizer of [Xie et al. (2022)](https://arxiv.org/abs/2208.06677), which combines the moments of the gradients
/// with a moment of the differences between consecutive gradients to estimate Nesterov momentum without evaluating the
/// gradient at an extrapolated point, and which is the analogue of
/// [`optax.adan`](https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.adan). Its update is
///
/// ```text
/// dₜ = gₜ − gₜ₋₁
/// mₜ = β₁ · mₜ₋₁ + (1 − β₁) · gₜ
/// vₜ = β₂ · vₜ₋₁ + (1 − β₂) · dₜ
/// nₜ = β₃ · nₜ₋₁ + (1 − β₃) · (gₜ + (1 − β₂) · dₜ)²
/// m̂ₜ = mₜ / (1 − β₁ᵗ)
/// v̂ₜ = vₜ / (1 − β₂ᵗ)
/// n̂ₜ = nₜ / (1 − β₃ᵗ)
/// uₜ = −ηₜ · ((m̂ₜ + (1 − β₂) · v̂ₜ) / (√(n̂ₜ + ε̄) + ε) + λ · θₜ₋₁)
/// ```
///
/// Here `t ≥ 1` counts the updates including the current one, `gₜ` is the gradient at the parameters `θₜ₋₁`, `uₜ` is
/// the update that is added to them (i.e., `θₜ = θₜ₋₁ + uₜ`), and `ηₜ` is the learning rate [`Schedule`] evaluated at
/// the `t − 1` updates applied before the current one. The gradient difference `dₜ` is zero at the first update (i.e.,
/// `d₁ = 0`), the moments `mₜ`, `vₜ`, and `nₜ` start at zero and decay with the rates `β₁`, `β₂`, and `β₃`, `m̂ₜ`,
/// `v̂ₜ`, and `n̂ₜ` are their bias corrected estimates, `ε` and `ε̄` are added outside and inside the square root for
/// numerical stability, and `λ` is the rate of the decoupled weight decay. All operations are element-wise. The
/// defaults are `β₁ = 0.98`, `β₂ = 0.92`, `β₃ = 0.99`, `ε = 1e-8`, `ε̄ = 1e-8`, and `λ = 0`, and
/// [`Self::with_weight_decay_mask`] restricts the weight decay to the leaves that the mask selects. Note that the
/// documentation of Optax states the algorithm of Xie et al. (2022) with moments that start at the first gradient and
/// with a proximal weight decay that divides by `1 + λ · η`, while this implementation, like the code of Optax, starts
/// the moments at zero with bias corrections and adds `λ · θₜ₋₁` to the rescaled update.
///
/// It is the chain of [`ScaleByAdan`], [`AddDecayedWeights`], and [`ScaleBySchedule::learning_rate`].
///
/// # References
///
///   - Xie et al. (2022). [Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep
///     Models](https://arxiv.org/abs/2208.06677).
#[derive(Clone, Debug)]
pub struct Adan<Sc: Schedule> {
    /// Transformation that rescales the gradients.
    adan: ScaleByAdan,

    /// Transformation that adds the decayed weights to the rescaled gradients.
    weight_decay: AddDecayedWeights,

    /// Transformation that scales the updates by the negated learning rate.
    learning_rate: ScaleBySchedule<Sc>,
}

impl<Sc: Schedule> Adan<Sc> {
    /// Creates a new [`Adan`] with the learning rate schedule `learning_rate` (e.g., an `f64` for a constant learning
    /// rate) and default values for all other hyperparameters.
    pub fn new(learning_rate: Sc) -> Self {
        Self {
            adan: ScaleByAdan::new().with_epsilon_root(1e-8).unwrap(),
            weight_decay: AddDecayedWeights::new(0.0).unwrap(),
            learning_rate: ScaleBySchedule::learning_rate(learning_rate),
        }
    }

    /// Returns this optimizer with `beta1` as the decay rate of the first moment estimates (refer to
    /// [`ScaleByAdan::with_beta1`]).
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `beta1` is not in `[0, 1)`.
    pub fn with_beta1(mut self, beta1: f64) -> Result<Self, OptimizerError> {
        self.adan = self.adan.with_beta1(beta1)?;
        Ok(self)
    }

    /// Returns this optimizer with `beta2` as the decay rate of the second moment estimates (refer to
    /// [`ScaleByAdan::with_beta2`]).
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `beta2` is not in `[0, 1)`.
    pub fn with_beta2(mut self, beta2: f64) -> Result<Self, OptimizerError> {
        self.adan = self.adan.with_beta2(beta2)?;
        Ok(self)
    }

    /// Returns this optimizer with `beta3` as the decay rate of the second moment estimates of Adan (refer to
    /// [`ScaleByAdan::with_beta3`]).
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `beta3` is not in `[0, 1)`.
    pub fn with_beta3(mut self, beta3: f64) -> Result<Self, OptimizerError> {
        self.adan = self.adan.with_beta3(beta3)?;
        Ok(self)
    }

    /// Returns this optimizer with `epsilon` as the term that is added for numerical stability outside of square roots
    /// (refer to [`ScaleByAdan::with_epsilon`]).
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `epsilon` is negative or not finite.
    pub fn with_epsilon(mut self, epsilon: f64) -> Result<Self, OptimizerError> {
        self.adan = self.adan.with_epsilon(epsilon)?;
        Ok(self)
    }

    /// Returns this optimizer with `epsilon_root` as the term that is added inside square roots (refer to
    /// [`ScaleByAdan::with_epsilon_root`]).
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `epsilon_root` is negative or not finite.
    pub fn with_epsilon_root(mut self, epsilon_root: f64) -> Result<Self, OptimizerError> {
        self.adan = self.adan.with_epsilon_root(epsilon_root)?;
        Ok(self)
    }

    /// Returns this optimizer with `weight_decay` as the rate of its decoupled weight decay, keeping its mask.
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

    /// Returns the [`ScaleByAdan`] transformation of this optimizer, which holds its rescaling hyperparameters.
    #[inline]
    pub fn adan(&self) -> &ScaleByAdan {
        &self.adan
    }

    /// Returns the [`AddDecayedWeights`] transformation of this optimizer, which holds its weight decay rate and mask.
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

impl<P: OptimizerValue, Sc: Schedule> Optimizer<P> for Adan<Sc> {
    type State<S: Parameterized<P>> = AdanState<P, S>;

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        let adan = self.adan.initialize(parameters)?;
        let learning_rate = self.learning_rate.initialize(parameters)?;
        Ok(AdanState { adan, learning_rate })
    }

    fn update<S: Parameterized<P>>(
        &self,
        gradients: S,
        state: Self::State<S>,
        parameters: &S,
        arguments: &UpdateArguments<'_, P, S>,
    ) -> Result<(S, Self::State<S>), OptimizerError> {
        let (updates, adan) = self.adan.update(gradients, state.adan, parameters, arguments)?;
        let (updates, ()) = self.weight_decay.update(updates, (), parameters, arguments)?;
        let (updates, learning_rate) =
            self.learning_rate.update(updates, state.learning_rate, parameters, arguments)?;
        Ok((updates, AdanState { adan, learning_rate }))
    }
}

parameterized_state! {
    /// State of [`OptimisticAdam`] for the parameter structure `S`, which holds the states of its Adam moment
    /// estimates, of its optimistic gradient step, and of its learning rate schedule.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct OptimisticAdamState<P, S> with family OptimisticAdamStateFamily {
        /// State of the [`ScaleByAdam`] transformation of the optimizer.
        adam: ScaleByAdamState<P, S>,

        /// State of the [`ScaleByOptimisticGradient`] transformation of the optimizer.
        optimistic: ScaleByOptimisticGradientState<P, S>,

        /// State of the [`ScaleBySchedule`] transformation that applies the learning rate.
        learning_rate: ScaleByScheduleState<P>,
    }
}

impl<P: Parameter, S> OptimisticAdamState<P, S> {
    /// Returns the state of the [`ScaleByAdam`] transformation of the optimizer.
    #[inline]
    pub fn adam(&self) -> &ScaleByAdamState<P, S> {
        &self.adam
    }

    /// Returns the state of the [`ScaleByOptimisticGradient`] transformation of the optimizer.
    #[inline]
    pub fn optimistic(&self) -> &ScaleByOptimisticGradientState<P, S> {
        &self.optimistic
    }

    /// Returns the state of the [`ScaleBySchedule`] transformation that applies the learning rate.
    #[inline]
    pub fn learning_rate(&self) -> &ScaleByScheduleState<P> {
        &self.learning_rate
    }
}

/// Optimistic Adam optimizer of [Daskalakis et al. (2017)](https://arxiv.org/abs/1711.00141) for min-max problems
/// (e.g., the training of generative adversarial networks), which extrapolates from the previous Adam update, and which
/// is the analogue of
/// [`optax.optimistic_adam_v2`](https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.optimistic_adam_v2).
/// Its update is
///
/// ```text
/// mₜ = β₁ · mₜ₋₁ + (1 − β₁) · gₜ
/// vₜ = β₂ · vₜ₋₁ + (1 − β₂) · gₜ²
/// m̂ₜ = β₁ · mₜ / (1 − β₁ᵗ⁺¹) + (1 − β₁) · gₜ / (1 − β₁ᵗ)
/// v̂ₜ = vₜ / (1 − β₂ᵗ)
/// aₜ = m̂ₜ / (√(v̂ₜ + ε̄) + ε)
/// uₜ = −ηₜ · ((α + β) · aₜ − β · aₜ₋₁)
/// ```
///
/// Here `t ≥ 1` counts the updates including the current one, `gₜ` is the gradient at the parameters `θₜ₋₁`, `uₜ` is
/// the update that is added to them (i.e., `θₜ = θₜ₋₁ + uₜ`), and `ηₜ` is the learning rate [`Schedule`] evaluated at
/// the `t − 1` updates applied before the current one. The moments `mₜ` and `vₜ` start at zero and decay with the rates
/// `β₁` and `β₂`, `m̂ₜ` is the Nesterov estimate of the first moment and `v̂ₜ` is the bias corrected estimate of the
/// second moment, `ε` and `ε̄` are added outside and inside the square root for numerical stability, `aₜ` is the Adam
/// update, `α` is the coefficient of the current Adam update, and `β` is the coefficient of the optimism. The previous
/// Adam update is the current one at the first update (i.e., `a₀ = a₁`). All operations are element-wise. The defaults
/// are `β₁ = 0.9`, `β₂ = 0.999`, `ε = 1e-8`, `ε̄ = 0`, and `α = β = 1`. [`Self::with_adam`] replaces the Adam rescaling
/// (e.g., to change its decay rates or to use the bias corrected estimate `m̂ₜ = mₜ / (1 − β₁ᵗ)` without Nesterov
/// momentum). The deprecated `optax.optimistic_adam`, whose optimism was tied to the learning rate, is not provided.
///
/// It is the chain of [`ScaleByAdam`] (with Nesterov momentum by default), [`ScaleByOptimisticGradient`], and
/// [`ScaleBySchedule::learning_rate`].
///
/// # References
///
///   - Daskalakis et al. (2017). [Training GANs with Optimism](https://arxiv.org/abs/1711.00141).
///   - Mokhtari, Ozdaglar, and Pattathil (2019). [A Unified Analysis of Extra-gradient and Optimistic Gradient Methods
///     for Saddle Point Problems: Proximal Point Approach](https://arxiv.org/abs/1901.08511).
///   - Kingma and Ba (2015). [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980).
#[derive(Clone, Debug)]
pub struct OptimisticAdam<Sc: Schedule> {
    /// Transformation that rescales the gradients by their moment estimates.
    adam: ScaleByAdam,

    /// Transformation that extrapolates from the previous Adam update.
    optimistic: ScaleByOptimisticGradient,

    /// Transformation that scales the updates by the negated learning rate.
    learning_rate: ScaleBySchedule<Sc>,
}

impl<Sc: Schedule> OptimisticAdam<Sc> {
    /// Creates a new [`OptimisticAdam`] with the learning rate schedule `learning_rate` (e.g., an `f64` for a constant
    /// learning rate) and default values for all other hyperparameters.
    pub fn new(learning_rate: Sc) -> Self {
        Self {
            adam: ScaleByAdam::new().with_nesterov(true),
            optimistic: ScaleByOptimisticGradient::new(),
            learning_rate: ScaleBySchedule::learning_rate(learning_rate),
        }
    }

    /// Returns this optimizer with `adam` as its Adam rescaling (e.g., to change its decay rates or to disable its
    /// Nesterov momentum).
    #[inline]
    pub fn with_adam(mut self, adam: ScaleByAdam) -> Self {
        self.adam = adam;
        self
    }

    /// Returns this optimizer with `alpha` as the coefficient of the current Adam update.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `alpha` is not finite.
    pub fn with_alpha(mut self, alpha: f64) -> Result<Self, OptimizerError> {
        self.optimistic = self.optimistic.with_alpha(alpha)?;
        Ok(self)
    }

    /// Returns this optimizer with `beta` as the coefficient of the optimism.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `beta` is not finite.
    pub fn with_beta(mut self, beta: f64) -> Result<Self, OptimizerError> {
        self.optimistic = self.optimistic.with_beta(beta)?;
        Ok(self)
    }

    /// Returns the [`ScaleByAdam`] transformation of this optimizer.
    #[inline]
    pub fn adam(&self) -> &ScaleByAdam {
        &self.adam
    }

    /// Returns the [`ScaleByOptimisticGradient`] transformation of this optimizer.
    #[inline]
    pub fn optimistic(&self) -> &ScaleByOptimisticGradient {
        &self.optimistic
    }

    /// Returns the learning rate [`Schedule`] of this optimizer.
    #[inline]
    pub fn learning_rate(&self) -> &Sc {
        self.learning_rate.schedule()
    }
}

impl<P: OptimizerValue, Sc: Schedule> Optimizer<P> for OptimisticAdam<Sc> {
    type State<S: Parameterized<P>> = OptimisticAdamState<P, S>;

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        let adam = self.adam.initialize(parameters)?;
        let optimistic = self.optimistic.initialize(parameters)?;
        let learning_rate = self.learning_rate.initialize(parameters)?;
        Ok(OptimisticAdamState { adam, optimistic, learning_rate })
    }

    fn update<S: Parameterized<P>>(
        &self,
        gradients: S,
        state: Self::State<S>,
        parameters: &S,
        arguments: &UpdateArguments<'_, P, S>,
    ) -> Result<(S, Self::State<S>), OptimizerError> {
        let (updates, adam) = self.adam.update(gradients, state.adam, parameters, arguments)?;
        let (updates, optimistic) = self.optimistic.update(updates, state.optimistic, parameters, arguments)?;
        let (updates, learning_rate) =
            self.learning_rate.update(updates, state.learning_rate, parameters, arguments)?;
        Ok((updates, OptimisticAdamState { adam, optimistic, learning_rate }))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use pretty_assertions::assert_eq;

    use ryft_core::{
        Add, Array, ArrayOperations, Mul, ProgramError, ReductionKind, Sub, Typed, bf16, differentiate_at, trace,
    };

    use crate::tests::{
        PARITY_GRADIENTS, PARITY_PARAMETERS, assert_parity, parity_elements, parity_parameters, parity_updates,
        state_paths, vector_updates,
    };
    use crate::{Chain, apply_updates};

    use super::*;

    /// Returns the quadratic loss `Σ (w - 1.5)² + (b + 0.5)²` of the vector `weights` (i.e., `w`) and the scalar
    /// `bias` (i.e., `b`), whose minimum is `0`.
    fn quadratic_loss<V: ArrayOperations>(weights: &V, bias: &V) -> Result<V, ProgramError> {
        let weights = Sub::sub(weights, &weights.fill_like(1.5f32)?)?;
        let bias = Sub::sub(bias, &bias.fill_like(-0.5f32)?)?;
        Add::add(&Mul::mul(&weights, &weights)?.reduce(&[0], ReductionKind::Sum)?, &Mul::mul(&bias, &bias)?)
    }

    #[test]
    fn test_adamw_new() {
        let optimizer = AdamW::new(0.1f64);
        assert_eq!(optimizer.adam(), &ScaleByAdam::new());
        assert_eq!(optimizer.weight_decay().rate(), 1e-4);
        assert!(optimizer.weight_decay().decays(&ParameterPath::root()));
        assert_eq!(optimizer.learning_rate(), &0.1);
    }

    #[test]
    fn test_adamw_with_beta1() {
        assert_eq!(AdamW::new(0.1f64).with_beta1(0.5).unwrap().adam().beta1(), 0.5);
        assert!(matches!(
            AdamW::new(0.1f64).with_beta1(1.5),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`beta1` must be in `[0, 1)` but got `1.5`",
        ));
    }

    #[test]
    fn test_adamw_with_beta2() {
        assert_eq!(AdamW::new(0.1f64).with_beta2(0.5).unwrap().adam().beta2(), 0.5);
        assert!(matches!(
            AdamW::new(0.1f64).with_beta2(1.0),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`beta2` must be in `[0, 1)` but got `1`",
        ));
    }

    #[test]
    fn test_adamw_with_epsilon() {
        assert_eq!(AdamW::new(0.1f64).with_epsilon(1e-6).unwrap().adam().epsilon(), 1e-6);
        assert!(matches!(
            AdamW::new(0.1f64).with_epsilon(f64::INFINITY),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`epsilon` must be finite and non-negative but got `inf`",
        ));
    }

    #[test]
    fn test_adamw_with_weight_decay() {
        // Changing the rate keeps a mask that was set before.
        let optimizer = AdamW::new(0.1f64)
            .with_weight_decay_mask(|path| path.to_string() != "$")
            .with_weight_decay(0.5)
            .unwrap();
        assert_eq!(optimizer.weight_decay().rate(), 0.5);
        assert!(!optimizer.weight_decay().decays(&ParameterPath::root()));
        assert!(matches!(
            AdamW::new(0.1f64).with_weight_decay(f64::NAN),
            Err(OptimizerError::InvalidConfiguration { message }) if message == "`rate` must be finite but got `NaN`",
        ));
    }

    #[test]
    fn test_adamw_with_weight_decay_mask() {
        let optimizer = AdamW::new(0.1f64).with_weight_decay_mask(|path| path.to_string() != "$");
        assert!(!optimizer.weight_decay().decays(&ParameterPath::root()));
        assert_eq!(optimizer.weight_decay().rate(), 1e-4);
    }

    #[test]
    fn test_adamw_initialize() {
        // The leaf paths of the state follow those of the parameters.
        let parameters = BTreeMap::from([
            ("bias", Array::scalar(bf16::from_f32(3.0)).unwrap()),
            ("weights", Array::vector(vec![1.0f32, 2.0]).unwrap()),
        ]);
        let state = AdamW::new(0.1f64).initialize(&parameters).unwrap();
        assert_eq!(state.adam(), &ScaleByAdam::new().initialize(&parameters).unwrap());
        assert_eq!(state.learning_rate().count(), &Array::scalar(0i32).unwrap());
        assert_eq!(
            state.named_parameters().map(|(path, _)| path.to_string()).collect::<Vec<_>>(),
            vec![
                "$.adam.count",
                "$.adam.first_moments[\"bias\"]",
                "$.adam.first_moments[\"weights\"]",
                "$.adam.second_moments[\"bias\"]",
                "$.adam.second_moments[\"weights\"]",
                "$.learning_rate.count",
            ],
        );
    }

    #[test]
    fn test_adamw_update() {
        // AdamW computes the same updates as the explicit chain of its transformations.
        let optimizer = AdamW::new(0.1f64).with_weight_decay(0.5).unwrap();
        let chain = Chain::new(ScaleByAdam::new(), AddDecayedWeights::new(0.5).unwrap())
            .then(ScaleBySchedule::learning_rate(0.1f64));
        let arguments = UpdateArguments::new();
        let mut parameters =
            vec![Array::vector(vec![1.0f32, -2.0]).unwrap(), Array::scalar(bf16::from_f32(3.0)).unwrap()];
        let mut state = optimizer.initialize(&parameters).unwrap();
        let mut chain_state = chain.initialize(&parameters).unwrap();
        for gradients in [[0.5f32, -1.0, 2.0], [1.5, 0.25, -1.0]] {
            let gradients = vec![
                Array::vector(vec![gradients[0], gradients[1]]).unwrap(),
                Array::scalar(bf16::from_f32(gradients[2])).unwrap(),
            ];
            let (updates, next_state) = optimizer.update(gradients.clone(), state, &parameters, &arguments).unwrap();
            let (chain_updates, next_chain_state) =
                chain.update(gradients, chain_state, &parameters, &arguments).unwrap();
            assert_eq!(updates, chain_updates);
            assert_eq!(next_state.adam(), &next_chain_state.0.0);
            assert_eq!(next_state.learning_rate(), &next_chain_state.1);
            parameters = apply_updates(parameters, updates).unwrap();
            state = next_state;
            chain_state = next_chain_state;
        }
        assert_eq!(state.adam().count(), &Array::scalar(2i32).unwrap());
    }

    #[test]
    fn test_adamw_update_minimizes_quadratic() {
        // Minimizes the quadratic loss eagerly from `w = [0, 3]` and `b = 2`, where it is `10.75`.
        let optimizer = AdamW::new(0.05f64);
        let mut parameters = (Array::vector(vec![0.0f32, 3.0]).unwrap(), Array::scalar(2.0f32).unwrap());
        let mut state = optimizer.initialize(&parameters).unwrap();
        let initial_loss = quadratic_loss(&parameters.0, &parameters.1).unwrap();
        assert_eq!(initial_loss.elements::<f32>().unwrap(), vec![10.75]);
        for _ in 0..300 {
            let (_, gradients) = differentiate_at(parameters.clone())
                .value_and_gradient(|(weights, bias)| quadratic_loss(&weights, &bias))
                .unwrap();
            let (updates, next_state) =
                optimizer.update(gradients, state, &parameters, &UpdateArguments::new()).unwrap();
            parameters = apply_updates(parameters, updates).unwrap();
            state = next_state;
        }
        let final_loss = quadratic_loss(&parameters.0, &parameters.1).unwrap().elements::<f32>().unwrap()[0];
        assert!(final_loss < 1e-3, "final loss {final_loss} is not below `1e-3`");
    }

    #[test]
    fn test_adamw_update_staged() {
        // Tracing the update into a program and interpreting that program reproduces the eager update exactly.
        let optimizer = AdamW::new(0.1f64).with_weight_decay(0.5).unwrap();
        let parameters = vec![Array::vector(vec![1.0f32, -2.0]).unwrap(), Array::scalar(bf16::from_f32(3.0)).unwrap()];
        let gradients = vec![Array::vector(vec![0.5f32, -1.0]).unwrap(), Array::scalar(bf16::from_f32(2.0)).unwrap()];
        let state = optimizer.initialize(&parameters).unwrap();
        let (_, program) = trace(
            |(gradients, state, parameters)| {
                optimizer
                    .update(gradients, state, &parameters, &UpdateArguments::new())
                    .map_err(|error| ProgramError::InvalidArgument { message: error.to_string() })
            },
            (gradients.clone(), state.clone(), parameters.clone()),
        )
        .unwrap();
        let staged = program.interpret((gradients.clone(), state.clone(), parameters.clone())).unwrap();
        let eager = optimizer.update(gradients, state, &parameters, &UpdateArguments::new()).unwrap();
        assert_eq!(staged, eager);
        assert_eq!(staged.0[1].r#type().as_ref(), eager.0[1].r#type().as_ref());
    }

    #[test]
    fn test_adamw_with_nesterov() {
        let optimizer = AdamW::new(0.1f64)
            .with_nesterov(true)
            .with_epsilon_root(1e-3)
            .unwrap()
            .with_first_moment_data_type(DataType::F64)
            .unwrap();
        assert!(optimizer.adam().nesterov());
        assert_eq!(optimizer.adam().epsilon_root(), 1e-3);
        assert_eq!(optimizer.adam().first_moment_data_type(), Some(DataType::F64));
    }

    #[test]
    fn test_adam_new() {
        let optimizer = Adam::new(0.1f64);
        assert_eq!(optimizer.adam(), &ScaleByAdam::new());
        assert!(optimizer.weight_decay().is_none());
        assert_eq!(optimizer.learning_rate(), &0.1);
        let optimizer = optimizer
            .with_beta1(0.5)
            .unwrap()
            .with_beta2(0.6)
            .unwrap()
            .with_epsilon(1e-3)
            .unwrap()
            .with_epsilon_root(1e-4)
            .unwrap()
            .with_first_moment_data_type(DataType::BF16)
            .unwrap()
            .with_nesterov(true)
            .with_weight_decay(0.5)
            .unwrap()
            .with_weight_decay_mask(|path| path.to_string() != "$");
        assert_eq!(
            (
                optimizer.adam().beta1(),
                optimizer.adam().beta2(),
                optimizer.adam().epsilon(),
                optimizer.adam().epsilon_root()
            ),
            (0.5, 0.6, 1e-3, 1e-4),
        );
        assert!(optimizer.adam().nesterov());
        assert_eq!(optimizer.weight_decay().map(AddDecayedWeights::rate), Some(0.5));
        assert!(!optimizer.weight_decay().unwrap().decays(&ParameterPath::root()));
        assert!(matches!(
            Adam::new(0.1f64).with_weight_decay(f64::NAN),
            Err(OptimizerError::InvalidConfiguration { message }) if message == "`rate` must be finite but got `NaN`",
        ));
    }

    #[test]
    fn test_adam_initialize() {
        assert_eq!(
            state_paths(&Adam::new(0.1f64)),
            vec![
                "$.adam.count",
                "$.adam.first_moments[\"bias\"]",
                "$.adam.first_moments[\"weights\"]",
                "$.adam.second_moments[\"bias\"]",
                "$.adam.second_moments[\"weights\"]",
                "$.learning_rate.count",
            ],
        );
    }

    #[test]
    fn test_adam_update() {
        // With `β₁ = β₂ = 0.5`, `ε = 0`, and `η = 0.5`, the gradient `2` gives the Adam update `2 / √4 = 1`, and the
        // gradient `1` gives `(1 / 0.75) / √(1.5 / 0.75)`, both scaled by `-0.5`.
        let optimizer = Adam::new(0.5f64).with_beta1(0.5).unwrap().with_beta2(0.5).unwrap().with_epsilon(0.0).unwrap();
        let second = (1.0f32 / 0.75) / (1.5f32 / 0.75).sqrt();
        assert_eq!(vector_updates(&optimizer, &[2.0], &[&[2.0], &[1.0]]), vec![vec![-0.5], vec![-0.5 * second]]);

        // The weight decay `0.5` adds `0.5 · 2` to the Adam update before the learning rate.
        let excluded = |path: &ParameterPath| path.to_string() != "$[0]";
        let masked_first = optimizer.clone().with_weight_decay_mask(excluded).with_weight_decay(0.5).unwrap();
        let optimizer = optimizer.with_weight_decay(0.5).unwrap();
        assert_eq!(
            vector_updates(&optimizer, &[2.0], &[&[2.0], &[1.0]]),
            vec![vec![-0.5 * 2.0], vec![-0.5 * (second + 1.0)],],
        );

        // A weight decay mask is kept whether it is set before or after the weight decay rate.
        assert_eq!(
            vector_updates(&masked_first, &[2.0], &[&[2.0], &[1.0]]),
            vector_updates(&optimizer.with_weight_decay_mask(excluded), &[2.0], &[&[2.0], &[1.0]]),
        );
    }

    #[test]
    fn test_amsgrad_new() {
        let optimizer = Amsgrad::new(0.1f64);
        assert_eq!(optimizer.amsgrad(), &ScaleByAmsgrad::new());
        assert_eq!(optimizer.learning_rate(), &0.1);
        let optimizer = optimizer
            .with_epsilon_root(1e-3)
            .unwrap()
            .with_first_moment_data_type(DataType::F64)
            .unwrap()
            .with_first_moment_bias_correction(false)
            .with_second_moment_bias_correction(false);
        assert_eq!(optimizer.amsgrad().epsilon_root(), 1e-3);
        assert!(!optimizer.amsgrad().first_moment_bias_correction());
        assert!(!optimizer.amsgrad().second_moment_bias_correction());
        assert!(matches!(
            Amsgrad::new(0.1f64).with_beta1(1.0),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`beta1` must be in `[0, 1)` but got `1`",
        ));
    }

    #[test]
    fn test_amsgrad_update() {
        // The AMSGrad updates `1` and `(1 / 0.75) / √4` of the transformation tests are scaled by `-0.5`.
        let optimizer =
            Amsgrad::new(0.5f64).with_beta1(0.5).unwrap().with_beta2(0.5).unwrap().with_epsilon(0.0).unwrap();
        assert_eq!(state_paths(&optimizer).len(), 8);
        assert_eq!(
            vector_updates(&optimizer, &[0.0], &[&[2.0], &[1.0]]),
            vec![vec![-0.5], vec![-0.5 * (1.0f32 / 0.75 / 2.0)],],
        );
    }

    #[test]
    fn test_adamax_new() {
        let optimizer = Adamax::new(0.1f64);
        assert_eq!(optimizer.adamax(), &ScaleByAdamax::new());
        assert!(optimizer.weight_decay().is_none());
        let optimizer = optimizer.with_weight_decay(1e-4).unwrap();
        assert_eq!(optimizer.weight_decay().map(AddDecayedWeights::rate), Some(1e-4));
        assert!(matches!(
            Adamax::new(0.1f64).with_epsilon(-1.0),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`epsilon` must be finite and non-negative but got `-1`",
        ));
    }

    #[test]
    fn test_adamax_update() {
        // The Adamax updates `[1, 1]` and `[0, (1.5 / 0.75) / 2]` of the transformation tests are scaled by `-0.5`, and
        // the weight decay `0.5` of AdamaxW adds `0.5 · [2, 2]` before the learning rate.
        let optimizer =
            Adamax::new(0.5f64).with_beta1(0.5).unwrap().with_beta2(0.5).unwrap().with_epsilon(0.0).unwrap();
        let gradients: [&[f32]; 2] = [&[2.0, 4.0], &[-1.0, 1.0]];
        assert_eq!(
            state_paths(&optimizer),
            state_paths(&Adam::new(0.5f64))
                .iter()
                .map(|path| path.replace("$.adam", "$.adamax"))
                .collect::<Vec<_>>(),
        );
        assert_eq!(
            vector_updates(&optimizer, &[2.0, 2.0], &gradients),
            vec![vec![-0.5, -0.5], vec![-0.0, -0.5 * (1.5f32 / 0.75 / 2.0)],],
        );
        let excluded = |path: &ParameterPath| path.to_string() != "$[0]";
        let masked_first = optimizer.clone().with_weight_decay_mask(excluded).with_weight_decay(0.5).unwrap();
        let optimizer = optimizer.with_weight_decay(0.5).unwrap();
        assert_eq!(vector_updates(&optimizer, &[2.0, 2.0], &gradients[..1]), vec![vec![-1.0, -1.0]]);

        // A weight decay mask is kept whether it is set before or after the weight decay rate.
        assert_eq!(
            vector_updates(&masked_first, &[2.0, 2.0], &gradients),
            vector_updates(&optimizer.with_weight_decay_mask(excluded), &[2.0, 2.0], &gradients),
        );
    }

    #[test]
    fn test_radam_new() {
        let optimizer =
            Radam::new(0.1f64).with_threshold(1.5).unwrap().with_nesterov(true).with_epsilon_root(1e-3).unwrap();
        assert_eq!(optimizer.radam().threshold(), 1.5);
        assert!(optimizer.radam().nesterov());
        assert_eq!(optimizer.radam().epsilon_root(), 1e-3);
        assert!(matches!(
            Radam::new(0.1f64).with_threshold(f64::NAN),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`threshold` must be finite but got `NaN`",
        ));
    }

    #[test]
    fn test_radam_update() {
        // The RAdam updates of the transformation tests are scaled by `-0.5`: the first is the unrectified `m̂ = 2`.
        let optimizer = Radam::new(0.5f64)
            .with_beta1(0.5)
            .unwrap()
            .with_beta2(0.75)
            .unwrap()
            .with_epsilon(0.0)
            .unwrap()
            .with_threshold(1.5)
            .unwrap();
        let decay_power = 0.75f32.powf(2.0);
        let length = 7.0f32 - 2.0 * 2.0 * decay_power / (1.0 - decay_power);
        let rectification = ((length - 4.0) * (length - 2.0) * 7.0 / (15.0 * length)).sqrt();
        let second_moment = 0.25f32 * 1.0 + 0.75 * (0.25 * 4.0);
        let second = rectification * (1.0f32 / 0.75) / (second_moment / (1.0 - decay_power)).sqrt();
        assert_eq!(vector_updates(&optimizer, &[0.0], &[&[2.0], &[1.0]]), vec![vec![-1.0], vec![-0.5 * second]]);
    }

    #[test]
    fn test_yogi_new() {
        let optimizer = Yogi::new(0.1f64);
        assert_eq!(optimizer.yogi(), &ScaleByYogi::new());
        let optimizer = optimizer.with_initial_accumulator_value(1.0).unwrap().with_epsilon_root(1e-3).unwrap();
        assert_eq!(optimizer.yogi().initial_accumulator_value(), 1.0);
        assert!(matches!(
            Yogi::new(0.1f64).with_beta2(2.0),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`beta2` must be in `[0, 1)` but got `2`",
        ));
    }

    #[test]
    fn test_yogi_update() {
        // The Yogi updates of the transformation tests are scaled by `-0.5`.
        let optimizer = Yogi::new(0.5f64)
            .with_beta1(0.5)
            .unwrap()
            .with_beta2(0.5)
            .unwrap()
            .with_epsilon(0.0)
            .unwrap()
            .with_initial_accumulator_value(1.0)
            .unwrap();
        assert_eq!(
            vector_updates(&optimizer, &[0.0], &[&[2.0], &[1.0]]),
            vec![
                vec![-0.5 * (2.0f32 / (3.0f32 / 0.5).sqrt())],
                vec![-0.5 * ((1.0f32 / 0.75) / (2.5f32 / 0.75).sqrt())],
            ],
        );
    }

    #[test]
    fn test_ada_belief_new() {
        let optimizer = AdaBelief::new(0.1f64);
        assert_eq!(optimizer.belief(), &ScaleByBelief::new());
        assert!(optimizer.weight_decay().is_none());
        let optimizer = optimizer.with_nesterov(true).with_weight_decay(0.5).unwrap();
        assert!(optimizer.belief().nesterov());
        assert_eq!(optimizer.weight_decay().map(AddDecayedWeights::rate), Some(0.5));
    }

    #[test]
    fn test_ada_belief_update() {
        // The AdaBelief updates `2` and `(1 / 0.75) / √(0.25 / 0.75)` of the transformation tests are scaled by `-0.5`.
        let optimizer = AdaBelief::new(0.5f64)
            .with_beta1(0.5)
            .unwrap()
            .with_beta2(0.5)
            .unwrap()
            .with_epsilon(0.0)
            .unwrap()
            .with_epsilon_root(0.0)
            .unwrap();
        assert_eq!(state_paths(&optimizer)[0], "$.belief.count");
        assert_eq!(
            vector_updates(&optimizer, &[0.0], &[&[2.0], &[1.0]]),
            vec![vec![-1.0], vec![-0.5 * ((1.0f32 / 0.75) / (0.25f32 / 0.75).sqrt())],],
        );

        // A weight decay mask is kept whether it is set before or after the weight decay rate.
        let excluded = |path: &ParameterPath| path.to_string() != "$[0]";
        let masked_first = optimizer.clone().with_weight_decay_mask(excluded).with_weight_decay(0.5).unwrap();
        let masked_last = optimizer.with_weight_decay(0.5).unwrap().with_weight_decay_mask(excluded);
        assert_eq!(
            vector_updates(&masked_first, &[2.0], &[&[2.0], &[1.0]]),
            vector_updates(&masked_last, &[2.0], &[&[2.0], &[1.0]]),
        );
    }

    #[test]
    fn test_adan_new() {
        let optimizer = Adan::new(0.1f64);
        assert_eq!(optimizer.adan().epsilon_root(), 1e-8);
        assert_eq!(optimizer.weight_decay().rate(), 0.0);
        let optimizer = optimizer.with_beta3(0.5).unwrap().with_weight_decay(0.5).unwrap();
        assert_eq!(optimizer.adan().beta3(), 0.5);
        assert_eq!(optimizer.weight_decay().rate(), 0.5);
    }

    #[test]
    fn test_adan_update() {
        // The Adan updates `1` and `(1 / 0.75 + 0.5 · (-0.5 / 0.75)) / √(1.125 / 0.75)` of the transformation tests are
        // scaled by `-0.5`.
        let optimizer = Adan::new(0.5f64)
            .with_beta1(0.5)
            .unwrap()
            .with_beta2(0.5)
            .unwrap()
            .with_beta3(0.5)
            .unwrap()
            .with_epsilon(0.0)
            .unwrap()
            .with_epsilon_root(0.0)
            .unwrap();
        assert_eq!(
            vector_updates(&optimizer, &[0.0], &[&[2.0], &[1.0]]),
            vec![vec![-0.5], vec![-0.5 * ((1.0f32 / 0.75 + 0.5 * (-0.5f32 / 0.75)) / (1.125f32 / 0.75).sqrt())],],
        );
    }

    #[test]
    fn test_optimistic_adam_new() {
        let optimizer = OptimisticAdam::new(0.1f64);
        assert!(optimizer.adam().nesterov());
        assert_eq!((optimizer.optimistic().alpha(), optimizer.optimistic().beta()), (1.0, 1.0));
        let optimizer = optimizer.with_alpha(0.5).unwrap().with_beta(2.0).unwrap().with_adam(ScaleByAdam::new());
        assert!(!optimizer.adam().nesterov());
        assert_eq!((optimizer.optimistic().alpha(), optimizer.optimistic().beta()), (0.5, 2.0));
        assert_eq!(optimizer.learning_rate(), &0.1);
    }

    #[test]
    fn test_optimistic_adam_update() {
        // Without Nesterov momentum, the Adam updates are `u₁ = 1` and `u₂ = (1 / 0.75) / √2`, so the optimistic steps
        // are `2 · 1 - 1 = 1` and `2 · u₂ - 1`, both scaled by `-0.5`.
        let adam = ScaleByAdam::new().with_beta1(0.5).unwrap().with_beta2(0.5).unwrap().with_epsilon(0.0).unwrap();
        let optimizer = OptimisticAdam::new(0.5f64).with_adam(adam);
        assert_eq!(state_paths(&optimizer)[5], "$.optimistic.initial_step");
        let second = (1.0f32 / 0.75) / (1.5f32 / 0.75).sqrt();
        assert_eq!(
            vector_updates(&optimizer, &[0.0], &[&[2.0], &[1.0]]),
            vec![vec![-0.5], vec![-0.5 * (2.0 * second - 1.0 * 1.0)],],
        );
    }

    #[test]
    fn test_adam_optax_parity() {
        // Optax 9ecde04b: `optax.adam(0.1)`.
        const EXPECTED: [[f32; 9]; 4] = [
            [
                -0.09999933,
                0.09999933,
                -0.09999933,
                -0.09999934,
                0.09999933,
                -0.09999933,
                0.09999933,
                -0.09999933,
                0.09999933,
            ],
            [
                -0.026633432,
                -0.016273059,
                -0.087105505,
                -0.09321702,
                -0.04941849,
                0.036609985,
                0.040021453,
                -0.03456023,
                0.020361768,
            ],
            [
                -0.06704981,
                0.007051367,
                -0.022722436,
                -0.047980227,
                -0.047713544,
                -0.025381634,
                -0.015380538,
                -0.05955345,
                0.024689391,
            ],
            [
                -0.07737498,
                -0.018474897,
                0.026137862,
                -0.05051261,
                0.011251225,
                -0.025295947,
                0.034904275,
                -0.063364856,
                -0.0042764237,
            ],
        ];
        assert_parity(&parity_updates(&Adam::new(0.1)), &EXPECTED);
        // Optax 9ecde04b: `optax.nadam(0.1)`.
        const NADAM: [[f32; 9]; 4] = [
            [
                -0.14736742,
                0.14736742,
                -0.14736743,
                -0.14736743,
                0.14736742,
                -0.14736742,
                0.14736743,
                -0.14736743,
                0.14736743,
            ],
            [
                0.016486194,
                -0.06838632,
                -0.07850528,
                -0.09211139,
                -0.10178071,
                0.08966437,
                0.001711358,
                0.005840812,
                -0.024660315,
            ],
            [
                -0.10741883,
                0.024058696,
                0.01805392,
                -0.013697824,
                -0.043821212,
                -0.06332546,
                -0.050548714,
                -0.08035058,
                0.026620632,
            ],
            [
                -0.089282595,
                -0.03771009,
                0.05947834,
                -0.050244756,
                0.048260726,
                -0.023688179,
                0.069686405,
                -0.06451181,
                -0.026100714,
            ],
        ];
        // Optax 9ecde04b: `optax.adamw(0.1, weight_decay=0.01)`.
        const ADAMW: [[f32; 9]; 4] = [
            [
                -0.10049933,
                0.10099933,
                -0.101999335,
                -0.10099935,
                0.10049933,
                -0.10024934,
                0.101999335,
                -0.101499334,
                0.09924933,
            ],
            [
                -0.027032932,
                -0.015374059,
                -0.0890035,
                -0.094116025,
                -0.04901899,
                0.036460232,
                0.04191945,
                -0.035958733,
                0.019512517,
            ],
            [
                -0.06742228,
                0.007965742,
                -0.024531431,
                -0.04878511,
                -0.047265023,
                -0.025567845,
                -0.013524457,
                -0.060915995,
                0.02382063,
            ],
            [
                -0.07768002,
                -0.01756849,
                0.024353396,
                -0.051268704,
                0.01174701,
                -0.02545659,
                0.036773883,
                -0.06466649,
                -0.005169006,
            ],
        ];
        // Optax 9ecde04b: `optax.nadamw(0.1, weight_decay=0.01)`.
        const NADAMW: [[f32; 9]; 4] = [
            [
                -0.14786743,
                0.14836742,
                -0.14936744,
                -0.14836743,
                0.14786743,
                -0.14761743,
                0.14936744,
                -0.14886743,
                0.14661743,
            ],
            [
                0.01613406,
                -0.067534685,
                -0.08035591,
                -0.09296302,
                -0.10142858,
                0.08956199,
                0.0035619908,
                0.0044896794,
                -0.025556931,
            ],
            [
                -0.10778709,
                0.024977863,
                0.016283644,
                -0.014456493,
                -0.04336765,
                -0.06351741,
                -0.048701644,
                -0.0817062,
                0.025749573,
            ],
            [
                -0.08954307,
                -0.0368159,
                0.05769178,
                -0.05098897,
                0.048757654,
                -0.023816606,
                0.071582176,
                -0.06578573,
                -0.026997525,
            ],
        ];
        // Optax 9ecde04b: `optax.adamw(0.1, weight_decay=0.5, mask={'bias': False, 'weights': True})`.
        const ADAMW_MASKED: [[f32; 9]; 4] = [
            [
                -0.09999933,
                0.09999933,
                -0.09999933,
                -0.14999934,
                0.12499934,
                -0.112499334,
                0.19999933,
                -0.17499934,
                0.062499333,
            ],
            [
                -0.026633432,
                -0.016273059,
                -0.087105505,
                -0.13571706,
                -0.030668456,
                0.029734952,
                0.1300215,
                -0.10081027,
                -0.0202632,
            ],
            [
                -0.06704981,
                0.007051367,
                -0.022722436,
                -0.083694406,
                -0.027430087,
                -0.033743415,
                0.06811842,
                -0.12076297,
                -0.014922417,
            ],
            [
                -0.07737498,
                -0.018474897,
                0.026137862,
                -0.08204206,
                0.032906186,
                -0.031970557,
                0.11499732,
                -0.118536234,
                -0.04314211,
            ],
        ];
        assert_parity(&parity_updates(&Adam::new(0.1).with_nesterov(true)), &NADAM);
        assert_parity(&parity_updates(&Adam::new(0.1).with_weight_decay(0.01).unwrap()), &ADAMW);
        assert_parity(&parity_updates(&Adam::new(0.1).with_nesterov(true).with_weight_decay(0.01).unwrap()), &NADAMW);
        assert_parity(
            &parity_updates(
                &Adam::new(0.1)
                    .with_weight_decay(0.5)
                    .unwrap()
                    .with_weight_decay_mask(|path| path.to_string() == "$[\"weights\"]"),
            ),
            &ADAMW_MASKED,
        );
    }

    #[test]
    fn test_adam_w_optax_parity() {
        // Optax 9ecde04b: `optax.adamw(0.1)`.
        const ADAMW_DEFAULT: [[f32; 9]; 4] = [
            [
                -0.10000433,
                0.10000934,
                -0.100019336,
                -0.10000934,
                0.10000433,
                -0.100001834,
                0.100019336,
                -0.10001433,
                0.099991836,
            ],
            [
                -0.026637433,
                -0.016264059,
                -0.087124504,
                -0.09322602,
                -0.04941449,
                0.036608484,
                0.040040452,
                -0.034574233,
                0.020353269,
            ],
            [
                -0.06705354,
                0.007060529,
                -0.022740563,
                -0.04798829,
                -0.047709048,
                -0.0253835,
                -0.015361938,
                -0.059567105,
                0.024680687,
            ],
            [
                -0.07737804,
                -0.018465808,
                0.02611996,
                -0.050520193,
                0.011256197,
                -0.025297558,
                0.03492303,
                -0.06337792,
                -0.004285374,
            ],
        ];
        // Optax 9ecde04b: `optax.adamw(0.1, b1=0.8, b2=0.99, eps=1e-6, eps_root=1e-8, weight_decay=0.05,
        // nesterov=True)`.
        const ADAMW_NESTEROV: [[f32; 9]; 4] = [
            [
                -0.14694288,
                0.14944366,
                -0.1544439,
                -0.14944403,
                0.14694288,
                -0.14569366,
                0.1544439,
                -0.15194412,
                0.14069414,
            ],
            [
                0.020963354,
                -0.07009123,
                -0.0840765,
                -0.093437634,
                -0.104813755,
                0.0944913,
                0.004901739,
                0.0052401903,
                -0.03540172,
            ],
            [
                -0.11544323,
                0.029719232,
                0.020155348,
                -0.0076941266,
                -0.0419479,
                -0.07029202,
                -0.05093959,
                -0.088360146,
                0.019568393,
            ],
            [
                -0.094794735,
                -0.039820317,
                0.06444699,
                -0.04890251,
                0.06163767,
                -0.025152862,
                0.08646823,
                -0.06990867,
                -0.039676674,
            ],
        ];
        assert_parity(&parity_updates(&AdamW::new(0.1)), &ADAMW_DEFAULT);
        assert_parity(
            &parity_updates(
                &AdamW::new(0.1)
                    .with_beta1(0.8)
                    .unwrap()
                    .with_beta2(0.99)
                    .unwrap()
                    .with_epsilon(1e-6)
                    .unwrap()
                    .with_epsilon_root(1e-8)
                    .unwrap()
                    .with_weight_decay(0.05)
                    .unwrap()
                    .with_nesterov(true),
            ),
            &ADAMW_NESTEROV,
        );
    }

    #[test]
    fn test_amsgrad_optax_parity() {
        // Optax 9ecde04b: `optax.amsgrad(0.1)`.
        const AMSGRAD: [[f32; 9]; 4] = [
            [
                -0.09999933,
                0.09999933,
                -0.09999933,
                -0.09999934,
                0.09999933,
                -0.09999933,
                0.09999933,
                -0.09999933,
                0.09999933,
            ],
            [
                -0.021052489,
                -0.016273059,
                -0.064911835,
                -0.07368372,
                -0.04941849,
                0.036609985,
                0.029824361,
                -0.026315609,
                0.016666556,
            ],
            [
                -0.06704981,
                0.006030557,
                -0.016359052,
                -0.032656606,
                -0.039432324,
                -0.025381634,
                -0.011931038,
                -0.04612515,
                0.016666554,
            ],
            [
                -0.07737498,
                -0.014988683,
                0.022322232,
                -0.030430153,
                0.011038225,
                -0.021969607,
                0.034904275,
                -0.04434399,
                -0.0027187965,
            ],
        ];
        // Optax 9ecde04b: `optax.amsgrad(0.1, bias_correction_mu=False, bias_correction_nu=False)`.
        const AMSGRAD_UNCORRECTED: [[f32; 9]; 4] = [
            [
                -0.31622678,
                0.3162273,
                -0.3162274,
                -0.31622753,
                0.31622678,
                -0.3162273,
                0.3162274,
                -0.31622756,
                0.3162276,
            ],
            [
                -0.113182046,
                -0.06915449,
                -0.3701662,
                -0.396138,
                -0.21001029,
                0.15557894,
                0.17007637,
                -0.14686827,
                0.08653002,
            ],
            [
                -0.33191323,
                0.034906026,
                -0.11248173,
                -0.2375141,
                -0.2361939,
                -0.12564546,
                -0.0761375,
                -0.29480454,
                0.12221868,
            ],
            [
                -0.4210487,
                -0.1005342,
                0.14223352,
                -0.27487275,
                0.06122541,
                -0.13765211,
                0.18993741,
                -0.34481046,
                -0.023270873,
            ],
        ];
        assert_parity(&parity_updates(&Amsgrad::new(0.1)), &AMSGRAD);
        assert_parity(
            &parity_updates(
                &Amsgrad::new(0.1).with_first_moment_bias_correction(false).with_second_moment_bias_correction(false),
            ),
            &AMSGRAD_UNCORRECTED,
        );
    }

    #[test]
    fn test_adamax_optax_parity() {
        // Optax 9ecde04b: `optax.adamax(0.1)`.
        const ADAMAX: [[f32; 9]; 4] = [
            [
                -0.09999998,
                0.09999998,
                -0.09999999,
                -0.09999999,
                0.09999998,
                -0.09999998,
                0.09999999,
                -0.09999998,
                0.09999999,
            ],
            [
                -0.021073699,
                -0.014736836,
                -0.064977236,
                -0.073757954,
                -0.036842097,
                0.028947363,
                0.029854408,
                -0.026342122,
                0.016683346,
            ],
            [
                -0.0413284,
                0.0054667215,
                -0.016391924,
                -0.032722227,
                -0.029426713,
                -0.020664198,
                -0.0119550135,
                -0.046217836,
                0.016700044,
            ],
            [
                -0.048745062,
                -0.013600896,
                0.019133467,
                -0.030521825,
                0.008229134,
                -0.017904233,
                0.023437042,
                -0.044477575,
                -0.0027269865,
            ],
        ];
        // Optax 9ecde04b: `optax.adamaxw(0.1, weight_decay=0.01)`.
        const ADAMAXW: [[f32; 9]; 4] = [
            [
                -0.10049998,
                0.10099997,
                -0.10199999,
                -0.10099999,
                0.10049998,
                -0.10024998,
                0.10199999,
                -0.101499975,
                0.09924998,
            ],
            [
                -0.0214732,
                -0.013837836,
                -0.06687524,
                -0.074656956,
                -0.036442596,
                0.028797612,
                0.031752408,
                -0.027740622,
                0.015834097,
            ],
            [
                -0.041706428,
                0.0063795596,
                -0.01822305,
                -0.03354657,
                -0.02899077,
                -0.020842744,
                -0.010088766,
                -0.047588594,
                0.015834961,
            ],
            [
                -0.04908138,
                -0.012694438,
                0.017320564,
                -0.031312622,
                0.008694067,
                -0.018061938,
                0.025313377,
                -0.04580074,
                -0.0036079057,
            ],
        ];
        assert_parity(&parity_updates(&Adamax::new(0.1)), &ADAMAX);
        assert_parity(&parity_updates(&Adamax::new(0.1).with_weight_decay(0.01).unwrap()), &ADAMAXW);
    }

    #[test]
    fn test_radam_optax_parity() {
        // Optax 9ecde04b: `optax.radam(0.1)`.
        const RADAM: [[f32; 9]; 4] = [
            [
                -0.009999999,
                0.019999998,
                -0.029999996,
                -0.039999995,
                0.009999999,
                -0.019999998,
                0.029999996,
                -0.04999999,
                0.05999999,
            ],
            [
                -0.0021052628,
                -0.003684209,
                -0.019473678,
                -0.029473677,
                -0.01105263,
                0.011578945,
                0.008947367,
                -0.013157889,
                0.009999998,
            ],
            [
                -0.012398521,
                0.0013653138,
                -0.0049077473,
                -0.013062726,
                -0.008819186,
                -0.009298889,
                -0.003579335,
                -0.023062725,
                0.009999997,
            ],
            [
                -0.014608896,
                -0.0033934272,
                0.006696713,
                -0.01217214,
                0.0024687403,
                -0.008048848,
                0.010546668,
                -0.022172138,
                -0.0016312884,
            ],
        ];
        assert_parity(&parity_updates(&Radam::new(0.1)), &RADAM);

        // RAdam applies its rectified update once `ρₜ` exceeds the threshold, which happens only from the fifth step,
        // so this configuration runs six steps (the parity gradients followed by the first two again). With
        // `β₂ = 0.999`, `ρₜ` is the difference of two numbers near 2000, and the rounding of `β₂ᵗ` (an integer power in
        // JAX and a floating-point power here, which differ by one ulp at `t = 6`) changes the sixth update by 0.6%, so
        // `β₂ = 0.9` keeps the comparison well conditioned.
        // Optax 9ecde04b: `optax.radam(0.1, b2=0.9, threshold=4.5, nesterov=True)`.
        const RADAM_ADAPTED: [[f32; 9]; 6] = [
            [
                -0.014736839,
                0.029473677,
                -0.04421052,
                -0.058947355,
                0.014736839,
                -0.029473677,
                0.04421052,
                -0.07368419,
                0.08842104,
            ],
            [
                0.0013031655,
                -0.015482614,
                -0.017550975,
                -0.029124094,
                -0.022763638,
                0.028358897,
                0.0003825985,
                0.0022237338,
                -0.0121110855,
            ],
            [
                -0.01986336,
                0.004658341,
                0.0038994092,
                -0.003729264,
                -0.008099742,
                -0.023200098,
                -0.011763618,
                -0.031116638,
                0.010782212,
            ],
            [
                -0.01685713,
                -0.006926504,
                0.015238789,
                -0.012107595,
                0.010589353,
                -0.0075372765,
                0.021056429,
                -0.022573473,
                -0.009956402,
            ],
            [
                -0.011553569,
                0.0056605292,
                -0.005261546,
                -0.013757623,
                0.0045317435,
                -0.0072231567,
                0.009802517,
                -0.0141690625,
                0.009047783,
            ],
            [
                -0.010725248,
                -0.010395135,
                -0.00554052,
                -0.019040063,
                -0.009993631,
                0.006789078,
                0.005398694,
                -0.0077076117,
                -0.0030358788,
            ],
        ];
        let optimizer = Radam::new(0.1).with_beta2(0.9).unwrap().with_threshold(4.5).unwrap().with_nesterov(true);
        let mut parameters = parity_parameters(&PARITY_PARAMETERS);
        let mut state = optimizer.initialize(&parameters).unwrap();
        let mut updates = Vec::new();
        for gradients in PARITY_GRADIENTS.iter().chain(&PARITY_GRADIENTS[..2]) {
            let (step_updates, next_state) =
                optimizer.update(parity_parameters(gradients), state, &parameters, &UpdateArguments::new()).unwrap();
            updates.push(parity_elements(&step_updates));
            parameters = apply_updates(parameters, step_updates).unwrap();
            state = next_state;
        }
        assert_parity(&updates, &RADAM_ADAPTED);
    }

    #[test]
    fn test_yogi_optax_parity() {
        // Optax 9ecde04b: `optax.yogi(0.1)`.
        const YOGI: [[f32; 9]; 4] = [
            [
                -0.09444513,
                0.0982869,
                -0.0991198,
                -0.09944048,
                0.09444513,
                -0.0982869,
                0.0991198,
                -0.09960113,
                0.09969481,
            ],
            [
                -0.025309801,
                -0.016120251,
                -0.08625051,
                -0.092655465,
                -0.048952993,
                0.03640036,
                0.039628614,
                -0.03439578,
                0.02029185,
            ],
            [
                -0.066360705,
                0.0069799223,
                -0.02252252,
                -0.047659513,
                -0.047203,
                -0.02527373,
                -0.015259138,
                -0.059298843,
                0.02458269,
            ],
            [
                -0.07666664,
                -0.018290352,
                0.025970718,
                -0.050133865,
                0.011165198,
                -0.025165806,
                0.03472503,
                -0.063056536,
                -0.0042567262,
            ],
        ];
        assert_parity(&parity_updates(&Yogi::new(0.1)), &YOGI);
    }

    #[test]
    fn test_ada_belief_optax_parity() {
        // Optax 9ecde04b: `optax.adabelief(0.1)`.
        const ADABELIEF: [[f32; 9]; 4] = [
            [
                -0.11111038,
                0.11111038,
                -0.11111037,
                -0.11111038,
                0.11111038,
                -0.11111038,
                0.11111037,
                -0.11111037,
                0.11111037,
            ],
            [
                -0.028369794,
                -0.01722796,
                -0.099352896,
                -0.10752074,
                -0.053307462,
                0.03910571,
                0.043007966,
                -0.03699088,
                0.021624671,
            ],
            [
                -0.07497136,
                0.007452087,
                -0.024302995,
                -0.052657563,
                -0.05190291,
                -0.026998153,
                -0.016319294,
                -0.06594329,
                0.026328186,
            ],
            [
                -0.09039627,
                -0.019607054,
                0.027966678,
                -0.056159277,
                0.011916401,
                -0.026971666,
                0.037542302,
                -0.071737655,
                -0.004527089,
            ],
        ];
        // Optax 9ecde04b: `optax.adabelief(0.1, weight_decay=0.01, nesterov=True)`.
        const ADABELIEF_VARIANT: [[f32; 9]; 4] = [
            [
                -0.1642416,
                0.16474159,
                -0.1657416,
                -0.16474159,
                0.1642416,
                -0.1639916,
                0.1657416,
                -0.16524158,
                0.16299158,
            ],
            [
                0.017225249,
                -0.07156396,
                -0.0913777,
                -0.10708071,
                -0.109454565,
                0.09569084,
                0.0036733225,
                0.0049168426,
                -0.027102819,
            ],
            [
                -0.120462716,
                0.026332745,
                0.017566858,
                -0.01576133,
                -0.04722361,
                -0.067540266,
                -0.051803388,
                -0.09031155,
                0.027501727,
            ],
            [
                -0.1045403,
                -0.039140504,
                0.061879475,
                -0.056573898,
                0.051606346,
                -0.02537155,
                0.07683562,
                -0.07428553,
                -0.028544014,
            ],
        ];
        assert_parity(&parity_updates(&AdaBelief::new(0.1)), &ADABELIEF);
        assert_parity(
            &parity_updates(&AdaBelief::new(0.1).with_weight_decay(0.01).unwrap().with_nesterov(true)),
            &ADABELIEF_VARIANT,
        );
    }

    #[test]
    fn test_adan_optax_parity() {
        // Optax 9ecde04b: `optax.adan(0.1)`.
        const ADAN: [[f32; 9]; 4] = [
            [
                -0.099999994,
                0.100000024,
                -0.10000004,
                -0.100000046,
                0.099999994,
                -0.100000024,
                0.10000004,
                -0.100000046,
                0.10000004,
            ],
            [
                -0.021650082,
                -0.01863359,
                -0.08673416,
                -0.09351189,
                -0.04830747,
                0.03684333,
                0.035145007,
                -0.029574683,
                0.015506779,
            ],
            [
                -0.06146347,
                0.0066038542,
                -0.022284517,
                -0.047859456,
                -0.043683637,
                -0.022688309,
                -0.014598309,
                -0.055813145,
                0.023465795,
            ],
            [
                -0.07081819,
                -0.016527355,
                0.021431094,
                -0.052881796,
                0.008402775,
                -0.020513235,
                0.032104205,
                -0.059150703,
                -0.0026923318,
            ],
        ];
        // Optax 9ecde04b: `optax.adan(0.1, weight_decay=0.01)`.
        const ADAN_WEIGHT_DECAY: [[f32; 9]; 4] = [
            [
                -0.1005,
                0.101000026,
                -0.102000035,
                -0.10100005,
                0.1005,
                -0.10025003,
                0.102000035,
                -0.10150005,
                0.09925004,
            ],
            [
                -0.022049582,
                -0.017734589,
                -0.08863216,
                -0.09441089,
                -0.04790797,
                0.03669358,
                0.037043005,
                -0.030973185,
                0.014657529,
            ],
            [
                -0.06184092,
                0.0075205886,
                -0.024093885,
                -0.048664045,
                -0.04323623,
                -0.022874752,
                -0.0127373515,
                -0.057180673,
                0.022601886,
            ],
            [
                -0.0711338,
                -0.015618141,
                0.01964582,
                -0.05363772,
                0.00889342,
                -0.020676803,
                0.033977903,
                -0.06046105,
                -0.0035788415,
            ],
        ];
        assert_parity(&parity_updates(&Adan::new(0.1)), &ADAN);
        assert_parity(&parity_updates(&Adan::new(0.1).with_weight_decay(0.01).unwrap()), &ADAN_WEIGHT_DECAY);
    }

    #[test]
    fn test_optimistic_adam_optax_parity() {
        // Optax 9ecde04b: `optax.optimistic_adam_v2(0.1)`.
        const OPTIMISTIC_ADAM: [[f32; 9]; 4] = [
            [
                -0.14736742,
                0.14736742,
                -0.14736743,
                -0.14736743,
                0.14736742,
                -0.14736742,
                0.14736743,
                -0.14736743,
                0.14736743,
            ],
            [
                0.1803398,
                -0.28414008,
                -0.009643125,
                -0.03685534,
                -0.35092884,
                0.32669616,
                -0.14394473,
                0.15904906,
                -0.19668806,
            ],
            [
                -0.23132384,
                0.11650371,
                0.114613116,
                0.064715736,
                0.014138293,
                -0.2163153,
                -0.10280879,
                -0.16654198,
                0.07790158,
            ],
            [
                -0.07114636,
                -0.09947887,
                0.10090275,
                -0.08679169,
                0.14034267,
                0.015949104,
                0.18992153,
                -0.048673045,
                -0.07882206,
            ],
        ];
        // Optax 9ecde04b: `optax.optimistic_adam_v2(0.1, alpha=0.5, beta=2.0)`.
        const OPTIMISTIC_ADAM_VARIANT: [[f32; 9]; 4] = [
            [
                -0.07368372,
                0.07368372,
                -0.07368372,
                -0.07368372,
                0.07368372,
                -0.07368372,
                0.07368372,
                -0.07368372,
                0.07368372,
            ],
            [
                0.33595034,
                -0.46570063,
                0.098471664,
                0.064456396,
                -0.54918665,
                0.5188958,
                -0.29045647,
                0.3093369,
                -0.35638568,
            ],
            [
                -0.30151945,
                0.19691937,
                0.20214537,
                0.1499782,
                0.0940084,
                -0.3376424,
                -0.12979451,
                -0.21255808,
                0.115872204,
            ],
            [
                -0.0083688265,
                -0.14239262,
                0.112588,
                -0.09821624,
                0.20829426,
                0.067430474,
                0.27531344,
                -0.00057837967,
                -0.11849304,
            ],
        ];
        assert_parity(&parity_updates(&OptimisticAdam::new(0.1)), &OPTIMISTIC_ADAM);
        assert_parity(
            &parity_updates(&OptimisticAdam::new(0.1).with_alpha(0.5).unwrap().with_beta(2.0).unwrap()),
            &OPTIMISTIC_ADAM_VARIANT,
        );
    }
}
