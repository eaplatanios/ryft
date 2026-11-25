//! Optimizers with layer-wise adaptation, which scale the update of every leaf by the ratio of the norms of its
//! parameter and its update: LAMB, LARS, and Fromage.

use std::fmt::Debug;
use std::sync::Arc;

use ryft_core::{Parameter, ParameterPath, Parameterized};

use crate::macros::parameterized_state;
use crate::{
    AddDecayedWeights, Optimizer, OptimizerError, OptimizerValue, Scale, ScaleByAdam, ScaleByAdamState,
    ScaleBySchedule, ScaleByScheduleState, ScaleByTrustRatio, Schedule, Trace, TraceState, UpdateArguments, convert_to,
    statistic_data_type, validate_finite, validate_non_negative, validate_structure,
};

// TODO(eaplatanios): Review this module.

parameterized_state! {
    /// State of [`Lamb`] for the parameter structure `S`, which holds the state of its Adam moment estimates and the
    /// step count of its learning rate schedule. Weight decay and the trust ratio are stateless.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct LambState<P, S> with family LambStateFamily {
        /// State of the [`ScaleByAdam`] transformation of the optimizer.
        adam: ScaleByAdamState<P, S>,

        /// State of the [`ScaleBySchedule`] transformation that applies the learning rate.
        learning_rate: ScaleByScheduleState<P>,
    }
}

impl<P: Parameter, S> LambState<P, S> {
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

/// LAMB optimizer of [You et al. (2020)](https://arxiv.org/abs/1904.00962), which scales the Adam update of every leaf
/// by the ratio of the norms of the leaf and of its update, and which is the analogue of
/// [`optax.lamb`](https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.lamb). For every leaf, its update is
///
/// ```text
/// mₜ = β₁ · mₜ₋₁ + (1 − β₁) · gₜ
/// vₜ = β₂ · vₜ₋₁ + (1 − β₂) · gₜ²
/// m̂ₜ = mₜ / (1 − β₁ᵗ)
/// v̂ₜ = vₜ / (1 − β₂ᵗ)
/// aₜ = m̂ₜ / (√(v̂ₜ + ε̄) + ε) + λ · θₜ₋₁
/// rₜ = ‖θₜ₋₁‖ / ‖aₜ‖   if ‖θₜ₋₁‖ > 0 and ‖aₜ‖ > 0
/// rₜ = 1               otherwise
/// uₜ = −ηₜ · rₜ · aₜ
/// ```
///
/// Here `t ≥ 1` counts the updates including the current one, `gₜ` is the gradient at the parameters `θₜ₋₁`, `uₜ` is
/// the update that is added to them (i.e., `θₜ = θₜ₋₁ + uₜ`), and `ηₜ` is the learning rate [`Schedule`] evaluated at
/// the `t − 1` updates applied before the current one. The moments `mₜ` and `vₜ` start at zero and decay with the rates
/// `β₁` and `β₂`, `m̂ₜ` and `v̂ₜ` are their bias corrected estimates, `ε` and `ε̄` are added outside and inside the
/// square root for numerical stability, `λ` is the weight decay rate, and `rₜ` is the trust ratio of the leaf, whose
/// norms `‖·‖` are taken over that leaf. The other operations are element-wise. The defaults are `β₁ = 0.9`,
/// `β₂ = 0.999`, `ε = 1e-6`, `ε̄ = 0`, and `λ = 0`. [`Self::with_adam`] replaces the Adam rescaling (e.g., to change
/// its decay rates or to use Nesterov momentum), and [`Self::with_weight_decay_mask`] restricts the weight decay to the
/// leaves that the mask selects.
///
/// It is the chain of [`ScaleByAdam`], [`AddDecayedWeights`], [`ScaleByTrustRatio`] with its defaults, and
/// [`ScaleBySchedule::learning_rate`].
///
/// # References
///
///   - You et al. (2020). [Large Batch Optimization for Deep Learning: Training BERT in 76
///     minutes](https://arxiv.org/abs/1904.00962).
///   - Kingma and Ba (2015). [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980).
#[derive(Clone, Debug)]
pub struct Lamb<Sc: Schedule> {
    /// Transformation that rescales the gradients by their moment estimates.
    adam: ScaleByAdam,

    /// Transformation that adds the decayed weights to the rescaled gradients.
    weight_decay: AddDecayedWeights,

    /// Transformation that scales the update of every leaf by its trust ratio.
    trust_ratio: ScaleByTrustRatio,

    /// Transformation that scales the updates by the negated learning rate.
    learning_rate: ScaleBySchedule<Sc>,
}

impl<Sc: Schedule> Lamb<Sc> {
    /// Creates a new [`Lamb`] with the learning rate schedule `learning_rate` (e.g., an `f64` for a constant learning
    /// rate) and default values for all other hyperparameters.
    pub fn new(learning_rate: Sc) -> Self {
        Self {
            adam: ScaleByAdam::new().with_epsilon(1e-6).unwrap(),
            weight_decay: AddDecayedWeights::new(0.0).unwrap(),
            trust_ratio: ScaleByTrustRatio::new(),
            learning_rate: ScaleBySchedule::learning_rate(learning_rate),
        }
    }

    /// Returns this optimizer with `adam` as its Adam rescaling (e.g., to change its decay rates or its `ε`).
    #[inline]
    pub fn with_adam(mut self, adam: ScaleByAdam) -> Self {
        self.adam = adam;
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

    /// Returns the [`ScaleByAdam`] transformation of this optimizer.
    #[inline]
    pub fn adam(&self) -> &ScaleByAdam {
        &self.adam
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

impl<P: OptimizerValue, Sc: Schedule> Optimizer<P> for Lamb<Sc> {
    type State<S: Parameterized<P>> = LambState<P, S>;

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        let adam = self.adam.initialize(parameters)?;
        let learning_rate = self.learning_rate.initialize(parameters)?;
        Ok(LambState { adam, learning_rate })
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
        let (updates, ()) = self.trust_ratio.update(updates, (), parameters, arguments)?;
        let (updates, learning_rate) =
            self.learning_rate.update(updates, state.learning_rate, parameters, arguments)?;
        Ok((updates, LambState { adam, learning_rate }))
    }
}

parameterized_state! {
    /// State of [`Lars`] for the parameter structure `S`, which holds the step count of its learning rate schedule and
    /// the state of its momentum. Weight decay and the trust ratio are stateless.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct LarsState<P, S> with family LarsStateFamily {
        /// State of the [`ScaleBySchedule`] transformation that applies the learning rate.
        learning_rate: ScaleByScheduleState<P>,

        /// State of the [`Trace`] transformation of the momentum.
        momentum: TraceState<P, S>,
    }
}

impl<P: Parameter, S> LarsState<P, S> {
    /// Returns the state of the [`ScaleBySchedule`] transformation that applies the learning rate.
    #[inline]
    pub fn learning_rate(&self) -> &ScaleByScheduleState<P> {
        &self.learning_rate
    }

    /// Returns the state of the [`Trace`] transformation of the momentum.
    #[inline]
    pub fn momentum(&self) -> &TraceState<P, S> {
        &self.momentum
    }
}

/// LARS optimizer of [You et al. (2017)](https://arxiv.org/abs/1708.03888), which scales the gradient of every leaf by
/// a trust ratio of the norms of the leaf and of its gradient before applying momentum, and which is the analogue of
/// [`optax.lars`](https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.lars). For every leaf, its update is
///
/// ```text
/// dₜ = gₜ + λ · θₜ₋₁
/// rₜ = c · ‖θₜ₋₁‖ / (‖dₜ‖ + ε)   if ‖θₜ₋₁‖ > 0 and ‖dₜ‖ > 0
/// rₜ = 1                         otherwise
/// mₜ = −ηₜ · rₜ · dₜ + μ · mₜ₋₁
/// uₜ = mₜ
/// ```
///
/// Here `t ≥ 1` counts the updates including the current one, `gₜ` is the gradient at the parameters `θₜ₋₁`, `uₜ` is
/// the update that is added to them (i.e., `θₜ = θₜ₋₁ + uₜ`), and `ηₜ` is the learning rate [`Schedule`] evaluated at
/// the `t − 1` updates applied before the current one. The weight decay `λ` is added to the gradients before the trust
/// ratio `rₜ` of the leaf, whose norms `‖·‖` are taken over that leaf, `c` is the trust coefficient, `ε` keeps the
/// denominator positive, and the momentum `mₜ` starts at zero and decays with the rate `μ`. The other operations are
/// element-wise. The defaults are `λ = 0`, `c = 1e-3`, `ε = 0`, and `μ = 0.9`, without Nesterov momentum.
/// [`Self::with_weight_decay_mask`] restricts the weight decay to the leaves that the mask selects,
/// [`Self::with_trust_ratio_mask`] restricts the trust ratio to the leaves that its mask selects (i.e., `rₜ = 1` for
/// all other leaves), and [`Self::with_nesterov`] replaces the update with `uₜ = −ηₜ · rₜ · dₜ + μ · mₜ`. Optax applies
/// the trust ratio through `optax.masked`, which this optimizer replaces with a predicate on the [`ParameterPath`]s of
/// the leaves, because the trust ratio is stateless.
///
/// It is the chain of [`AddDecayedWeights`], [`ScaleByTrustRatio`] on the leaves that the trust ratio mask selects,
/// [`ScaleBySchedule::learning_rate`], and [`Trace`], so the learning rate is applied before the momentum.
///
/// # References
///
///   - You, Gitman, and Ginsburg (2017). [Large Batch Training of Convolutional
///     Networks](https://arxiv.org/abs/1708.03888).
#[derive(Clone)]
pub struct Lars<Sc: Schedule> {
    /// Transformation that adds the decayed weights to the gradients.
    weight_decay: AddDecayedWeights,

    /// Transformation that scales the update of every selected leaf by its trust ratio.
    trust_ratio: ScaleByTrustRatio,

    /// Predicate that selects the leaves whose updates are scaled by their trust ratio by their paths, or [`None`] if
    /// all leaves are.
    trust_ratio_mask: Option<Arc<dyn Fn(&ParameterPath) -> bool + Send + Sync>>,

    /// Transformation that scales the updates by the negated learning rate.
    learning_rate: ScaleBySchedule<Sc>,

    /// Transformation that accumulates the momentum of the scaled updates.
    momentum: Trace,
}

impl<Sc: Schedule> Lars<Sc> {
    /// Creates a new [`Lars`] with the learning rate schedule `learning_rate` (e.g., an `f64` for a constant learning
    /// rate) and default values for all other hyperparameters.
    pub fn new(learning_rate: Sc) -> Self {
        Self {
            weight_decay: AddDecayedWeights::new(0.0).unwrap(),
            trust_ratio: ScaleByTrustRatio::new().with_trust_coefficient(1e-3).unwrap(),
            trust_ratio_mask: None,
            learning_rate: ScaleBySchedule::learning_rate(learning_rate),
            momentum: Trace::new(0.9).unwrap(),
        }
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

    /// Returns this optimizer with `trust_coefficient` as the coefficient of its trust ratio.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `trust_coefficient` is negative or not finite.
    pub fn with_trust_coefficient(mut self, trust_coefficient: f64) -> Result<Self, OptimizerError> {
        self.trust_ratio = self.trust_ratio.with_trust_coefficient(trust_coefficient)?;
        Ok(self)
    }

    /// Returns this optimizer with `epsilon` as the term that is added to the norms of the updates in its trust ratio.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `epsilon` is negative or not finite.
    pub fn with_epsilon(mut self, epsilon: f64) -> Result<Self, OptimizerError> {
        self.trust_ratio = self.trust_ratio.with_epsilon(epsilon)?;
        Ok(self)
    }

    /// Returns this optimizer with `mask` as the predicate that selects the leaves whose updates are scaled by their
    /// trust ratio by their [`ParameterPath`]s, as returned by [`Parameterized::named_parameters`] for the parameters.
    #[inline]
    pub fn with_trust_ratio_mask<Mask: 'static + Fn(&ParameterPath) -> bool + Send + Sync>(
        mut self,
        mask: Mask,
    ) -> Self {
        self.trust_ratio_mask = Some(Arc::new(mask));
        self
    }

    /// Returns this optimizer with the momentum decay `momentum`, keeping its Nesterov setting.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `momentum` is not finite.
    pub fn with_momentum(mut self, momentum: f64) -> Result<Self, OptimizerError> {
        self.momentum = Trace::new(momentum)?.with_nesterov(self.momentum.nesterov());
        Ok(self)
    }

    /// Returns this optimizer with Nesterov momentum when `nesterov` is `true`.
    #[inline]
    pub fn with_nesterov(mut self, nesterov: bool) -> Self {
        self.momentum = self.momentum.with_nesterov(nesterov);
        self
    }

    /// Returns the [`AddDecayedWeights`] transformation of this optimizer.
    #[inline]
    pub fn weight_decay(&self) -> &AddDecayedWeights {
        &self.weight_decay
    }

    /// Returns the [`ScaleByTrustRatio`] transformation of this optimizer.
    #[inline]
    pub fn trust_ratio(&self) -> &ScaleByTrustRatio {
        &self.trust_ratio
    }

    /// Returns whether this optimizer scales the update of the leaf at `path` by its trust ratio, which is the case for
    /// every leaf when it has no trust ratio mask.
    #[inline]
    pub fn scales_by_trust_ratio(&self, path: &ParameterPath) -> bool {
        self.trust_ratio_mask.as_ref().is_none_or(|mask| mask(path))
    }

    /// Returns the [`Trace`] transformation of the momentum of this optimizer.
    #[inline]
    pub fn momentum(&self) -> &Trace {
        &self.momentum
    }

    /// Returns the learning rate [`Schedule`] of this optimizer.
    #[inline]
    pub fn learning_rate(&self) -> &Sc {
        self.learning_rate.schedule()
    }
}

impl<Sc: Schedule + Debug> Debug for Lars<Sc> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Masks are opaque functions, so only their presence is rendered.
        formatter
            .debug_struct("Lars")
            .field("weight_decay", &self.weight_decay)
            .field("trust_ratio", &self.trust_ratio)
            .field("trust_ratio_masked", &self.trust_ratio_mask.is_some())
            .field("learning_rate", &self.learning_rate)
            .field("momentum", &self.momentum)
            .finish()
    }
}

impl<P: OptimizerValue, Sc: Schedule> Optimizer<P> for Lars<Sc> {
    type State<S: Parameterized<P>> = LarsState<P, S>;

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        let learning_rate = self.learning_rate.initialize(parameters)?;
        let momentum = self.momentum.initialize(parameters)?;
        Ok(LarsState { learning_rate, momentum })
    }

    fn update<S: Parameterized<P>>(
        &self,
        gradients: S,
        state: Self::State<S>,
        parameters: &S,
        arguments: &UpdateArguments<'_, P, S>,
    ) -> Result<(S, Self::State<S>), OptimizerError> {
        // The trust ratio is stateless, so applying it to the selected leaves only is what `optax.masked` does.
        let (updates, ()) = self.weight_decay.update(gradients, (), parameters, arguments)?;
        validate_structure("parameters", &updates.parameter_paths().collect::<Vec<_>>(), parameters)?;
        let structure = updates.parameter_structure();
        let updates = updates
            .into_parameters()
            .zip(parameters.named_parameters())
            .map(|(update, (path, parameter))| {
                if !self.scales_by_trust_ratio(&path) {
                    return Ok(update);
                }
                let data_type = update.r#type().data_type();
                let statistic_type = statistic_data_type(data_type);
                let scaled = self
                    .trust_ratio
                    .scale_leaf(&convert_to(&update, statistic_type)?, &convert_to(parameter, statistic_type)?)?;
                Ok(convert_to(&scaled, data_type)?)
            })
            .collect::<Result<Vec<_>, OptimizerError>>()?;
        let updates = S::from_parameters(structure, updates)?;
        let (updates, learning_rate) =
            self.learning_rate.update(updates, state.learning_rate, parameters, arguments)?;
        let (updates, momentum) = self.momentum.update(updates, state.momentum, parameters, arguments)?;
        Ok((updates, LarsState { learning_rate, momentum }))
    }
}

/// Fromage optimizer of [Bernstein et al. (2020)](https://arxiv.org/abs/2002.03432), which takes steps of a size
/// relative to the norm of every leaf and shrinks the leaf to keep its norm from growing, and which is the analogue of
/// [`optax.fromage`](https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.fromage) for a constant learning
/// rate. For every leaf, its update is
///
/// ```text
/// rₜ = max(‖θₜ₋₁‖, δ) / max(‖gₜ‖, δ)
/// uₜ = −η / √(1 + η²) · rₜ · gₜ + (1 / √(1 + η²) − 1) · θₜ₋₁
/// ```
///
/// Here `t ≥ 1` counts the updates including the current one, `gₜ` is the gradient at the parameters `θₜ₋₁`, `uₜ` is
/// the update that is added to them, and `η` is the constant learning rate. The norms `‖·‖` are taken over the leaf and
/// are clamped from below at the minimum norm `δ` (and `rₜ = 1` when a clamped norm is zero, which only happens for
/// `δ = 0`). The other operations are element-wise. The update gives `θₜ = (θₜ₋₁ − η · rₜ · gₜ) / √(1 + η²)`. The
/// default is `δ = 1e-6`, which [`Self::with_minimum_norm`] changes. Learning rate schedules, which Optax also accepts,
/// are not supported, because the factor `1 / √(1 + η²)` and the weight decay would have to be schedules as well. This
/// optimizer is stateless.
///
/// It is the chain of [`ScaleByTrustRatio`] (with the minimum norm `δ`), [`Scale`] by `−η / √(1 + η²)`, and
/// [`AddDecayedWeights`] with the rate `1 / √(1 + η²) − 1`.
///
/// # References
///
///   - Bernstein et al. (2020). [On the distance between two neural networks and the stability of
///     learning](https://arxiv.org/abs/2002.03432).
#[derive(Clone, Debug, PartialEq)]
pub struct Fromage {
    /// Constant learning rate `η`.
    learning_rate: f64,

    /// Transformation that scales the update of every leaf by its trust ratio.
    trust_ratio: ScaleByTrustRatio,
}

impl Fromage {
    /// Creates a new [`Fromage`] with the constant learning rate `learning_rate` and the default minimum norm.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `learning_rate` is not finite.
    pub fn new(learning_rate: f64) -> Result<Self, OptimizerError> {
        validate_finite("learning_rate", learning_rate)?;
        Ok(Self { learning_rate, trust_ratio: ScaleByTrustRatio::new().with_minimum_norm(1e-6)? })
    }

    /// Returns this optimizer with `minimum_norm` as the norm at which its trust ratio clamps the norms from below.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `minimum_norm` is negative or not finite.
    pub fn with_minimum_norm(mut self, minimum_norm: f64) -> Result<Self, OptimizerError> {
        validate_non_negative("minimum_norm", minimum_norm)?;
        self.trust_ratio = self.trust_ratio.with_minimum_norm(minimum_norm)?;
        Ok(self)
    }

    /// Returns the constant learning rate `η` of this optimizer.
    #[inline]
    pub fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    /// Returns the [`ScaleByTrustRatio`] transformation of this optimizer.
    #[inline]
    pub fn trust_ratio(&self) -> &ScaleByTrustRatio {
        &self.trust_ratio
    }
}

impl<P: OptimizerValue> Optimizer<P> for Fromage {
    type State<S: Parameterized<P>> = ();

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        let _ = parameters;
        Ok(())
    }

    fn update<S: Parameterized<P>>(
        &self,
        gradients: S,
        state: Self::State<S>,
        parameters: &S,
        arguments: &UpdateArguments<'_, P, S>,
    ) -> Result<(S, Self::State<S>), OptimizerError> {
        // The factor is computed on the host in `f64`, as Optax computes it from a Python float.
        let factor = 1.0 / (1.0 + self.learning_rate * self.learning_rate).powf(0.5);
        let (updates, ()) = self.trust_ratio.update(gradients, (), parameters, arguments)?;
        let (updates, ()) = Scale::new(-(self.learning_rate * factor))?.update(updates, (), parameters, arguments)?;
        let (updates, ()) = AddDecayedWeights::new(factor - 1.0)?.update(updates, (), parameters, arguments)?;
        Ok((updates, state))
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::tests::{assert_parity, parity_updates, state_paths, vector_updates};

    use super::*;

    #[test]
    fn test_lamb_new() {
        let optimizer = Lamb::new(0.1f64);
        assert_eq!(optimizer.adam(), &ScaleByAdam::new().with_epsilon(1e-6).unwrap());
        assert_eq!(optimizer.weight_decay().rate(), 0.0);
        let optimizer = optimizer
            .with_weight_decay(0.5)
            .unwrap()
            .with_weight_decay_mask(|path| path.to_string() != "$")
            .with_adam(ScaleByAdam::new());
        assert_eq!(optimizer.adam(), &ScaleByAdam::new());
        assert!(!optimizer.weight_decay().decays(&ParameterPath::root()));
        assert_eq!(optimizer.learning_rate(), &0.1);
        assert_eq!(state_paths(&optimizer)[0], "$.adam.count");
    }

    #[test]
    fn test_lamb_update() {
        // The Adam updates `u₁ = [1, 1]` and `u₂ = [a, a]` with `a = (1 / 0.75) / √2` are scaled by the trust ratio
        // `‖[3, 4]‖ / ‖u‖` and by `-0.5`.
        let adam = ScaleByAdam::new().with_beta1(0.5).unwrap().with_beta2(0.5).unwrap().with_epsilon(0.0).unwrap();
        let optimizer = Lamb::new(0.5f64).with_adam(adam);
        let first = 1.0 * (1.0f32 * 5.0 / (2.0f32.sqrt() + 0.0));
        let second_update = (1.0f32 / 0.75) / (1.5f32 / 0.75).sqrt();
        let second = second_update
            * (1.0f32 * 5.0 / ((second_update * second_update + second_update * second_update).sqrt() + 0.0));
        assert_eq!(
            vector_updates(&optimizer, &[3.0, 4.0], &[&[2.0, 2.0], &[1.0, 1.0]]),
            vec![vec![-0.5 * first, -0.5 * first], vec![-0.5 * second, -0.5 * second],],
        );
    }

    #[test]
    fn test_lars_new() {
        let optimizer = Lars::new(0.1f64);
        assert_eq!(optimizer.weight_decay().rate(), 0.0);
        assert_eq!(optimizer.trust_ratio(), &ScaleByTrustRatio::new().with_trust_coefficient(1e-3).unwrap());
        assert_eq!(optimizer.momentum(), &Trace::new(0.9).unwrap());
        assert!(optimizer.scales_by_trust_ratio(&ParameterPath::root()));
        let optimizer = optimizer
            .with_weight_decay(0.5)
            .unwrap()
            .with_weight_decay_mask(|_| false)
            .with_trust_coefficient(1.0)
            .unwrap()
            .with_epsilon(1e-3)
            .unwrap()
            .with_trust_ratio_mask(|_| false)
            .with_nesterov(true)
            .with_momentum(0.5)
            .unwrap();
        assert_eq!(optimizer.weight_decay().rate(), 0.5);
        assert_eq!((optimizer.trust_ratio().trust_coefficient(), optimizer.trust_ratio().epsilon()), (1.0, 1e-3));
        assert!(!optimizer.scales_by_trust_ratio(&ParameterPath::root()));
        assert_eq!(optimizer.momentum(), &Trace::new(0.5).unwrap().with_nesterov(true));
        assert_eq!(optimizer.learning_rate(), &0.1);
        assert!(format!("{optimizer:?}").contains("trust_ratio_masked: true"));
    }

    #[test]
    fn test_lars_update() {
        // The gradient `[0, 2]` of the parameter `[3, 4]` has the trust ratio `5 / 2`, so the scaled updates are
        // `-0.5 · [0, 5]`, which the momentum `0.5` accumulates into `[0, -2.5]` and `[0, -3.75]`.
        let optimizer = Lars::new(0.5f64).with_trust_coefficient(1.0).unwrap().with_momentum(0.5).unwrap();
        assert_eq!(
            state_paths(&optimizer),
            vec!["$.learning_rate.count", "$.momentum.trace[\"bias\"]", "$.momentum.trace[\"weights\"]"],
        );
        assert_eq!(
            vector_updates(&optimizer, &[3.0, 4.0], &[&[0.0, 2.0], &[0.0, 2.0]]),
            vec![vec![-0.0, -2.5], vec![-0.0, -3.75],],
        );

        // A trust ratio mask that excludes the leaf keeps its gradient.
        let optimizer = optimizer.with_trust_ratio_mask(|_| false);
        assert_eq!(
            vector_updates(&optimizer, &[3.0, 4.0], &[&[0.0, 2.0], &[0.0, 2.0]]),
            vec![vec![-0.0, -1.0], vec![-0.0, -1.5],],
        );
    }

    #[test]
    fn test_fromage_new() {
        let optimizer = Fromage::new(1.0).unwrap();
        assert_eq!(optimizer.learning_rate(), 1.0);
        assert_eq!(optimizer.trust_ratio().minimum_norm(), 1e-6);
        assert_eq!(optimizer.with_minimum_norm(0.0).unwrap().trust_ratio().minimum_norm(), 0.0);
        assert!(matches!(
            Fromage::new(f64::NAN),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`learning_rate` must be finite but got `NaN`",
        ));
    }

    #[test]
    fn test_fromage_update() {
        // With `η = 1`, `μ = 1 / √2`. The gradient `[0, 2]` of the parameter `[3, 4]` is scaled by the trust ratio to
        // `[0, 5]`, then by `-μ`, and `(μ - 1) · [3, 4]` is added. The optimizer is stateless, so both steps agree.
        let factor = 1.0 / 2.0f64.powf(0.5);
        let (scale, decay) = ((-factor) as f32, (factor - 1.0) as f32);
        let expected = vec![scale * 0.0 + decay * 3.0, scale * 5.0 + decay * 4.0];
        assert_eq!(
            vector_updates(&Fromage::new(1.0).unwrap(), &[3.0, 4.0], &[&[0.0, 2.0], &[0.0, 2.0]]),
            vec![expected.clone(), expected],
        );
    }

    #[test]
    fn test_lamb_optax_parity() {
        // Optax 9ecde04b: `optax.lamb(0.1)`.
        const LAMB: [[f32; 9]; 4] = [
            [
                -0.13228706,
                0.13228773,
                -0.13228793,
                -0.11636885,
                0.116367996,
                -0.116368584,
                0.11636876,
                -0.11636891,
                0.116368964,
            ],
            [
                -0.0602159,
                -0.03679229,
                -0.19693966,
                -0.1991637,
                -0.10558541,
                0.07821941,
                0.0855081,
                -0.07384002,
                0.043504175,
            ],
            [
                -0.18138193,
                0.019075235,
                -0.06146837,
                -0.12502842,
                -0.1243333,
                -0.06614035,
                -0.040079083,
                -0.15518649,
                0.06433649,
            ],
            [
                -0.17013043,
                -0.0406222,
                0.057471436,
                -0.13773641,
                0.030679528,
                -0.068976365,
                0.09517611,
                -0.1727818,
                -0.011660854,
            ],
        ];
        // Optax 9ecde04b: `optax.lamb(0.1, weight_decay=0.01, mask={'bias': False, 'weights': True})`.
        const LAMB_WEIGHT_DECAY: [[f32; 9]; 4] = [
            [
                -0.13228706,
                0.13228773,
                -0.13228793,
                -0.11665306,
                0.1160747,
                -0.11578653,
                0.11780795,
                -0.11723061,
                0.11463191,
            ],
            [
                -0.0602159,
                -0.03679229,
                -0.19693966,
                -0.198785,
                -0.10358433,
                0.07705428,
                0.08852046,
                -0.07592886,
                0.04118732,
            ],
            [
                -0.18138193,
                0.019075235,
                -0.06146837,
                -0.1258911,
                -0.12216898,
                -0.06620643,
                -0.035147868,
                -0.15743987,
                0.06152587,
            ],
            [
                -0.17013043,
                -0.0406222,
                0.057471436,
                -0.13567962,
                0.03151053,
                -0.06758847,
                0.09758787,
                -0.17139342,
                -0.013931001,
            ],
        ];
        assert_parity(&parity_updates(&Lamb::new(0.1)), &LAMB);
        assert_parity(
            &parity_updates(
                &Lamb::new(0.1)
                    .with_weight_decay(0.01)
                    .unwrap()
                    .with_weight_decay_mask(|path| path.to_string() == "$[\"weights\"]"),
            ),
            &LAMB_WEIGHT_DECAY,
        );
    }

    #[test]
    fn test_lars_optax_parity() {
        // Optax 9ecde04b: `optax.lars(0.1)`.
        const LARS: [[f32; 9]; 4] = [
            [
                -6.1237246e-5,
                0.00012247449,
                -0.00018371175,
                -0.000119522854,
                2.9880714e-5,
                -5.9761427e-5,
                8.964215e-5,
                -0.00014940357,
                0.0001792843,
            ],
            [
                -1.3284658e-5,
                -9.891726e-5,
                -0.0002489983,
                -0.00019139363,
                -9.884195e-5,
                0.00011386083,
                3.87664e-5,
                -5.064015e-5,
                1.46655075e-5,
            ],
            [
                -0.0001956355,
                -2.7799098e-5,
                -0.00010164558,
                -0.00010921835,
                -0.00010996972,
                -8.663297e-5,
                -7.0170085e-5,
                -0.00021367188,
                5.5222896e-5,
            ],
            [
                -0.00028259034,
                -0.00010490797,
                9.492615e-5,
                -0.00014151003,
                3.0667812e-5,
                -9.957643e-5,
                0.00013130775,
                -0.00027873172,
                -7.993995e-5,
            ],
        ];
        // Optax 9ecde04b: `optax.lars(0.1, weight_decay=0.01, trust_coefficient=0.5, eps=1e-3, momentum=0.8,
        // nesterov=True)`.
        const LARS_VARIANT: [[f32; 9]; 4] = [
            [
                -0.054418165,
                0.10883633,
                -0.16584584,
                -0.10844278,
                0.027771933,
                -0.053560153,
                0.08463828,
                -0.1362147,
                0.15671305,
            ],
            [
                0.011945255,
                -0.12695725,
                -0.14028126,
                -0.113940224,
                -0.09667029,
                0.12457918,
                0.0009218957,
                0.01881296,
                -0.07384162,
            ],
            [
                -0.15341409,
                0.025982145,
                0.013506308,
                -0.0067561828,
                -0.045839,
                -0.12634307,
                -0.06890529,
                -0.16291729,
                0.031312846,
            ],
            [
                -0.14438456,
                -0.063824885,
                0.11512089,
                -0.06589008,
                0.07800563,
                -0.047259595,
                0.14676592,
                -0.13912031,
                -0.09803556,
            ],
        ];
        assert_parity(&parity_updates(&Lars::new(0.1)), &LARS);
        assert_parity(
            &parity_updates(
                &Lars::new(0.1)
                    .with_weight_decay(0.01)
                    .unwrap()
                    .with_trust_coefficient(0.5)
                    .unwrap()
                    .with_epsilon(1e-3)
                    .unwrap()
                    .with_momentum(0.8)
                    .unwrap()
                    .with_nesterov(true),
            ),
            &LARS_VARIANT,
        );
    }

    #[test]
    fn test_fromage_optax_parity() {
        // Optax 9ecde04b: `optax.fromage(0.1)`.
        const FROMAGE: [[f32; 9]; 4] = [
            [
                -0.06341474,
                0.12682948,
                -0.19272563,
                -0.12389249,
                0.032213826,
                -0.060705546,
                0.09912289,
                -0.15610632,
                0.17467244,
            ],
            [
                0.03514953,
                -0.1822477,
                -0.083601594,
                -0.08339295,
                -0.116245955,
                0.15715055,
                -0.030088805,
                0.0723755,
                -0.14291771,
            ],
            [
                -0.16792141,
                0.06043126,
                0.10183258,
                0.056071974,
                -0.017103583,
                -0.18173753,
                -0.09042709,
                -0.16704485,
                0.036124345,
            ],
            [
                -0.098723724,
                -0.06797403,
                0.16106828,
                -0.045320407,
                0.12630744,
                -0.021371434,
                0.19501798,
                -0.08841575,
                -0.1273831,
            ],
        ];
        assert_parity(&parity_updates(&Fromage::new(0.1).unwrap()), &FROMAGE);
    }
}
