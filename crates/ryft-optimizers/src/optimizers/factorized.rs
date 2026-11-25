//! Memory-efficient optimizers that keep second moment statistics per axis rather than per element: Adafactor and SM3.

use ryft_core::{DataType, Parameter, ParameterPath, Parameterized};

use crate::macros::parameterized_state;
use crate::optimizers::OptionalWeightDecay;
use crate::{
    AddDecayedWeights, ClipByBlockRms, Ema, EmaState, Optimizer, OptimizerError, OptimizerValue, Scale,
    ScaleByFactoredRms, ScaleByFactoredRmsState, ScaleByParamBlockRms, ScaleBySchedule, ScaleByScheduleState,
    ScaleBySm3, ScaleBySm3State, Schedule, UpdateArguments, validate_decay_rate, validate_finite,
    validate_floating_point,
};

// TODO(eaplatanios): Review this module.

parameterized_state! {
    /// State of [`Adafactor`] for the parameter structure `S`, which holds the state of its factored rescaling, the
    /// step count of its learning rate schedule (one element with a learning rate and none without), and the state of
    /// its momentum (one element with momentum and none without).
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct AdafactorState<P, S> with family AdafactorStateFamily {
        /// State of the [`ScaleByFactoredRms`] transformation of the optimizer.
        factored_rms: ScaleByFactoredRmsState<P, S>,

        /// State of the [`ScaleBySchedule`] transformation of the learning rate, which has one element with a learning
        /// rate and none without.
        learning_rate: Vec<ScaleByScheduleState<P>>,

        /// State of the [`Ema`] transformation of the momentum, which has one element with momentum and none without.
        momentum: Vec<EmaState<P, S>>,
    }
}

impl<P: Parameter, S> AdafactorState<P, S> {
    /// Returns the state of the [`ScaleByFactoredRms`] transformation of the optimizer.
    #[inline]
    pub fn factored_rms(&self) -> &ScaleByFactoredRmsState<P, S> {
        &self.factored_rms
    }

    /// Returns the state of the [`ScaleBySchedule`] transformation of the learning rate, if the optimizer has one.
    #[inline]
    pub fn learning_rate(&self) -> Option<&ScaleByScheduleState<P>> {
        self.learning_rate.first()
    }

    /// Returns the state of the [`Ema`] transformation of the momentum, if the optimizer has momentum.
    #[inline]
    pub fn momentum(&self) -> Option<&EmaState<P, S>> {
        self.momentum.first()
    }
}

/// Adafactor optimizer of [Shazeer and Stern (2018)](https://arxiv.org/abs/1804.04235), which saves memory by
/// estimating the second moments of the gradients of every matrix from the moving averages of their row and column
/// means, and which is the analogue of
/// [`optax.adafactor`](https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.adafactor). For a factored
/// leaf, written as a matrix `g` with rows `i` and columns `j`, its statistics are
///
/// ```text
/// β̂ₜ  = 1 − (t − o)^(−d)
/// rᵢ  = β̂ₜ · rᵢ + (1 − β̂ₜ) · meanⱼ(gᵢⱼ² + ε)
/// cⱼ  = β̂ₜ · cⱼ + (1 − β̂ₜ) · meanᵢ(gᵢⱼ² + ε)
/// aᵢⱼ = gᵢⱼ / √(rᵢ · cⱼ / meanₖ(rₖ))
/// ```
///
/// and its update is
///
/// ```text
/// âₜ = aₜ / max(1, rms(aₜ) / τ)
/// bₜ = ηₜ · max(rms(θₜ₋₁), 10⁻³) · âₜ
/// mₜ = β₁ · mₜ₋₁ + (1 − β₁) · bₜ
/// uₜ = −(mₜ + λ · θₜ₋₁)
/// ```
///
/// Here `t ≥ 1` counts the updates including the current one, `gₜ` is the gradient at the parameters `θₜ₋₁`, `uₜ` is
/// the update that is added to them (i.e., `θₜ = θₜ₋₁ + uₜ`), and `ηₜ` is the learning rate [`Schedule`] evaluated at
/// the `t − 1` updates applied before the current one. A leaf is factored when its second largest extent is at least
/// the minimum extent to factor, in which case `j` indexes its largest axis, `i` indexes its second largest axis, any
/// other axes keep separate statistics, and the row statistics `r` and the column statistics `c` start at zero and are
/// updated in place. Every other leaf keeps full statistics `vₜ = β̂ₜ · vₜ₋₁ + (1 − β̂ₜ) · (gₜ² + ε)`, which start at
/// zero, and is rescaled as `aₜ = gₜ / √vₜ`. The decay `β̂ₜ` grows towards one with the decay rate `d` and the step
/// offset `o`, `ε` is added to the squared gradients, `rms` is the root mean square of a leaf, `τ` is the clipping
/// threshold, `mₜ` is the optional momentum with the decay rate `β₁`, which starts at zero, and `λ` is the weight decay
/// rate. The defaults are `d = 0.8`, `o = 0`, a minimum extent to factor of `128`, `ε = 1e-30`, and `τ = 1`, with the
/// parameter scale `max(rms(θₜ₋₁), 10⁻³)`, without a learning rate (i.e., `ηₜ = 1`), without momentum (i.e.,
/// `mₜ = bₜ`), and without weight decay (i.e., `λ = 0`). [`Self::with_factored_rms`] changes the statistics,
/// [`Self::with_clipping_threshold`] with [`None`] disables the clipping (i.e., `âₜ = aₜ`),
/// [`Self::with_learning_rate`] sets the learning rate schedule, which is not negated,
/// [`Self::with_multiply_by_parameter_scale`] with `false` removes the parameter scale, [`Self::with_momentum`] sets
/// `β₁` (with the momentum stored in [`Self::with_momentum_data_type`], `f32` by default), and
/// [`Self::with_weight_decay`] and [`Self::with_weight_decay_mask`] set `λ` and the leaves that decay. Note that the
/// weight decay is not scaled by the learning rate. Note also that, as in Optax, the step size is the learning rate
/// schedule rather than the relative step size schedule of Shazeer and Stern (2018), whose parameter scale this
/// optimizer applies.
///
/// It is the chain of [`ScaleByFactoredRms`], [`ClipByBlockRms`] (unless the clipping is disabled),
/// [`ScaleBySchedule::new`] with the learning rate (only when one is set), [`ScaleByParamBlockRms`] (unless the
/// parameter scale is disabled), [`Ema`] without debiasing (only with momentum), [`AddDecayedWeights`] (only with a
/// weight decay), and [`Scale`] by `−1`.
///
/// # References
///
///   - Shazeer and Stern (2018). [Adafactor: Adaptive Learning Rates with Sublinear Memory
///     Cost](https://arxiv.org/abs/1804.04235).
#[derive(Clone, Debug)]
pub struct Adafactor<Sc: Schedule = f64> {
    /// Transformation that rescales the gradients by their factored second moments.
    factored_rms: ScaleByFactoredRms,

    /// Transformation that clips the root mean square of every leaf, or [`None`] without clipping.
    clipping: Option<ClipByBlockRms>,

    /// Transformation that scales the updates by the learning rate, or [`None`] without a learning rate.
    learning_rate: Option<ScaleBySchedule<Sc>>,

    /// Whether the updates are scaled by the root mean squares of the parameters.
    multiply_by_parameter_scale: bool,

    /// Decay of the momentum, or [`None`] without momentum.
    momentum: Option<f64>,

    /// Data type in which the momentum is stored.
    momentum_data_type: DataType,

    /// Transformation that adds the decayed weights to the updates, or [`None`] without weight decay.
    weight_decay: OptionalWeightDecay,
}

impl Adafactor<f64> {
    /// Creates a new [`Adafactor`] without a learning rate and with default values for all other hyperparameters.
    pub fn new() -> Self {
        Self {
            factored_rms: ScaleByFactoredRms::new(),
            clipping: Some(ClipByBlockRms::new(1.0).unwrap()),
            learning_rate: None,
            multiply_by_parameter_scale: true,
            momentum: None,
            momentum_data_type: DataType::F32,
            weight_decay: OptionalWeightDecay::new(),
        }
    }
}

impl Default for Adafactor<f64> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Sc: Schedule> Adafactor<Sc> {
    /// Returns this optimizer with the learning rate schedule `learning_rate`, which scales the updates before the
    /// parameter scale.
    pub fn with_learning_rate<Learning: Schedule>(self, learning_rate: Learning) -> Adafactor<Learning> {
        Adafactor {
            factored_rms: self.factored_rms,
            clipping: self.clipping,
            learning_rate: Some(ScaleBySchedule::new(learning_rate)),
            multiply_by_parameter_scale: self.multiply_by_parameter_scale,
            momentum: self.momentum,
            momentum_data_type: self.momentum_data_type,
            weight_decay: self.weight_decay,
        }
    }

    /// Returns this optimizer with `factored_rms` as its factored rescaling (e.g., to change its minimum extent to
    /// factor, its decay rate and offset, or its `ε`).
    #[inline]
    pub fn with_factored_rms(mut self, factored_rms: ScaleByFactoredRms) -> Self {
        self.factored_rms = factored_rms;
        self
    }

    /// Returns this optimizer with the root mean square of every leaf clipped to `threshold`, or without clipping when
    /// `threshold` is [`None`].
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `threshold` is not positive and finite.
    pub fn with_clipping_threshold(mut self, threshold: Option<f64>) -> Result<Self, OptimizerError> {
        self.clipping = threshold.map(ClipByBlockRms::new).transpose()?;
        Ok(self)
    }

    /// Returns this optimizer with the updates scaled by the root mean squares of the parameters when
    /// `multiply_by_parameter_scale` is `true`.
    #[inline]
    pub fn with_multiply_by_parameter_scale(mut self, multiply_by_parameter_scale: bool) -> Self {
        self.multiply_by_parameter_scale = multiply_by_parameter_scale;
        self
    }

    /// Returns this optimizer with the momentum decay `momentum`.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `momentum` is not in `[0, 1)`.
    pub fn with_momentum(mut self, momentum: f64) -> Result<Self, OptimizerError> {
        validate_decay_rate("momentum", momentum)?;
        self.momentum = Some(momentum);
        Ok(self)
    }

    /// Returns this optimizer with its momentum stored in `data_type`.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `data_type` is not a floating-point data type.
    pub fn with_momentum_data_type(mut self, data_type: DataType) -> Result<Self, OptimizerError> {
        validate_floating_point("momentum_data_type", data_type)?;
        self.momentum_data_type = data_type;
        Ok(self)
    }

    /// Returns this optimizer with `weight_decay` as the rate of its weight decay, keeping its mask.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `weight_decay` is not finite.
    pub fn with_weight_decay(mut self, weight_decay: f64) -> Result<Self, OptimizerError> {
        validate_finite("weight_decay", weight_decay)?;
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

    /// Returns the [`ScaleByFactoredRms`] transformation of this optimizer.
    #[inline]
    pub fn factored_rms(&self) -> &ScaleByFactoredRms {
        &self.factored_rms
    }

    /// Returns the [`ClipByBlockRms`] transformation of this optimizer, if it clips its updates.
    #[inline]
    pub fn clipping(&self) -> Option<&ClipByBlockRms> {
        self.clipping.as_ref()
    }

    /// Returns the learning rate [`Schedule`] of this optimizer, if it has one.
    #[inline]
    pub fn learning_rate(&self) -> Option<&Sc> {
        self.learning_rate.as_ref().map(ScaleBySchedule::schedule)
    }

    /// Returns whether this optimizer scales its updates by the root mean squares of the parameters.
    #[inline]
    pub fn multiply_by_parameter_scale(&self) -> bool {
        self.multiply_by_parameter_scale
    }

    /// Returns the [`Ema`] transformation of the momentum of this optimizer, if it has momentum.
    pub fn momentum(&self) -> Option<Ema> {
        // The configuration was validated by the builder functions, so the moving average can be built directly.
        self.momentum.map(|momentum| {
            Ema::new(momentum)
                .unwrap()
                .with_debias(false)
                .with_accumulator_data_type(self.momentum_data_type)
                .unwrap()
        })
    }

    /// Returns the [`AddDecayedWeights`] transformation of this optimizer, if it decays its weights.
    #[inline]
    pub fn weight_decay(&self) -> Option<&AddDecayedWeights> {
        self.weight_decay.transformation()
    }
}

impl<P: OptimizerValue, Sc: Schedule> Optimizer<P> for Adafactor<Sc> {
    type State<S: Parameterized<P>> = AdafactorState<P, S>;

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        Ok(AdafactorState {
            factored_rms: self.factored_rms.initialize(parameters)?,
            learning_rate: self
                .learning_rate
                .iter()
                .map(|scale| scale.initialize(parameters))
                .collect::<Result<_, _>>()?,
            momentum: self.momentum().map(|ema| ema.initialize(parameters)).into_iter().collect::<Result<_, _>>()?,
        })
    }

    fn update<S: Parameterized<P>>(
        &self,
        gradients: S,
        state: Self::State<S>,
        parameters: &S,
        arguments: &UpdateArguments<'_, P, S>,
    ) -> Result<(S, Self::State<S>), OptimizerError> {
        // This applies the transformations in the order of `optax.adafactor`.
        let AdafactorState { factored_rms, learning_rate, momentum } = state;
        let mismatch = || OptimizerError::StructureMismatch {
            message: "the state of `Adafactor` does not match its configuration".to_string(),
        };
        let (mut updates, factored_rms) = self.factored_rms.update(gradients, factored_rms, parameters, arguments)?;
        if let Some(clipping) = &self.clipping {
            updates = clipping.update(updates, (), parameters, arguments)?.0;
        }
        let learning_rate = match (&self.learning_rate, learning_rate.into_iter().next()) {
            (Some(scale), Some(learning_rate)) => {
                let (scaled, learning_rate) = scale.update(updates, learning_rate, parameters, arguments)?;
                updates = scaled;
                vec![learning_rate]
            }
            (None, None) => Vec::new(),
            _ => return Err(mismatch()),
        };
        if self.multiply_by_parameter_scale {
            updates = ScaleByParamBlockRms::new().update(updates, (), parameters, arguments)?.0;
        }
        let momentum = match (self.momentum(), momentum.into_iter().next()) {
            (Some(ema), Some(momentum)) => {
                let (averaged, momentum) = ema.update(updates, momentum, parameters, arguments)?;
                updates = averaged;
                vec![momentum]
            }
            (None, None) => Vec::new(),
            _ => return Err(mismatch()),
        };
        let updates = self.weight_decay.apply(updates, parameters, arguments)?;
        let (updates, ()) = Scale::new(-1.0)?.update(updates, (), parameters, arguments)?;
        Ok((updates, AdafactorState { factored_rms, learning_rate, momentum }))
    }
}

parameterized_state! {
    /// State of [`Sm3`] for the parameter structure `S`, which holds the state of its [`ScaleBySm3`] transformation.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct Sm3State<P, S> with family Sm3StateFamily {
        /// State of the [`ScaleBySm3`] transformation of the optimizer.
        sm3: ScaleBySm3State<P, S>,
    }
}

impl<P: Parameter, S> Sm3State<P, S> {
    /// Returns the state of the [`ScaleBySm3`] transformation of the optimizer.
    #[inline]
    pub fn sm3(&self) -> &ScaleBySm3State<P, S> {
        &self.sm3
    }
}

/// SM3 optimizer of [Anil et al. (2019)](https://arxiv.org/abs/1901.11150), which saves memory by keeping one second
/// moment accumulator per index of every axis of a leaf rather than one per element, and which is the analogue of
/// [`optax.sm3`](https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.sm3). For a matrix leaf `g` with rows
/// `i` and columns `j`, its update is
///
/// ```text
/// νᵢⱼ = κ · gᵢⱼ² + β₂ · min(rᵢ, cⱼ)
/// aᵢⱼ = gᵢⱼ / √(νᵢⱼ + ε)   where νᵢⱼ > 0
/// aᵢⱼ = 0                  where νᵢⱼ = 0
/// rᵢ  = maxⱼ νᵢⱼ
/// cⱼ  = maxᵢ νᵢⱼ
/// mₜ  = β₁ · mₜ₋₁ + (1 − β₁) · aₜ
/// uₜ  = −η · mₜ
/// ```
///
/// Here `t ≥ 1` counts the updates including the current one, `gₜ` (written `g`) is the gradient at the parameters
/// `θₜ₋₁`, `uₜ` is the update that is added to them, and `η` is the constant learning rate. The row accumulators `rᵢ`
/// and the column accumulators `cⱼ` start at zero, `νᵢⱼ` is the second moment estimate, `β₂` is its decay with
/// `κ = 1 − β₂` (or `κ = 1` when `β₂ = 1`), `ε` is added for numerical stability, and the momentum `mₜ` starts at zero
/// and decays with the rate `β₁`. A leaf of another rank keeps one accumulator per axis, takes the minimum over all of
/// them, and updates each one with the maximum over the other axes (or with `ν` itself for vectors). Rank-0 leaves are
/// rejected, as Optax rejects them. The defaults are `β₁ = 0.9` (i.e., the momentum), `β₂ = 1`, and `ε = 1e-8`.
///
/// It is the chain of [`ScaleBySm3`] and [`Scale`] by the negated constant learning rate, as in Optax.
///
/// # References
///
///   - Anil et al. (2019). [Memory-Efficient Adaptive Optimization](https://arxiv.org/abs/1901.11150).
#[derive(Clone, Debug, PartialEq)]
pub struct Sm3 {
    /// Constant learning rate.
    learning_rate: f64,

    /// Transformation that rescales the gradients by their per-axis accumulators.
    sm3: ScaleBySm3,
}

impl Sm3 {
    /// Creates a new [`Sm3`] with the constant learning rate `learning_rate` and the default hyperparameters.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `learning_rate` is not finite.
    pub fn new(learning_rate: f64) -> Result<Self, OptimizerError> {
        validate_finite("learning_rate", learning_rate)?;
        Ok(Self { learning_rate, sm3: ScaleBySm3::new() })
    }

    /// Returns this optimizer with the momentum decay `momentum` (i.e., the `β₁` of [`ScaleBySm3`]).
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `momentum` is not in `[0, 1)`.
    pub fn with_momentum(mut self, momentum: f64) -> Result<Self, OptimizerError> {
        self.sm3 = self.sm3.with_beta1(momentum)?;
        Ok(self)
    }

    /// Returns this optimizer with `epsilon` as the term that is added inside the square root.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `epsilon` is negative or not finite.
    pub fn with_epsilon(mut self, epsilon: f64) -> Result<Self, OptimizerError> {
        self.sm3 = self.sm3.with_epsilon(epsilon)?;
        Ok(self)
    }

    /// Returns the constant learning rate of this optimizer.
    #[inline]
    pub fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    /// Returns the [`ScaleBySm3`] transformation of this optimizer.
    #[inline]
    pub fn sm3(&self) -> &ScaleBySm3 {
        &self.sm3
    }
}

impl<P: OptimizerValue> Optimizer<P> for Sm3 {
    type State<S: Parameterized<P>> = Sm3State<P, S>;

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        Ok(Sm3State { sm3: self.sm3.initialize(parameters)? })
    }

    fn update<S: Parameterized<P>>(
        &self,
        gradients: S,
        state: Self::State<S>,
        parameters: &S,
        arguments: &UpdateArguments<'_, P, S>,
    ) -> Result<(S, Self::State<S>), OptimizerError> {
        let (updates, sm3) = self.sm3.update(gradients, state.sm3, parameters, arguments)?;
        let (updates, ()) = Scale::new(-self.learning_rate)?.update(updates, (), parameters, arguments)?;
        Ok((updates, Sm3State { sm3 }))
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::tests::{assert_parity, parity_updates, state_paths, vector_updates};

    use super::*;

    #[test]
    fn test_adafactor_new() {
        let optimizer = Adafactor::new();
        assert_eq!(optimizer.factored_rms(), &ScaleByFactoredRms::new());
        assert_eq!(optimizer.clipping(), Some(&ClipByBlockRms::new(1.0).unwrap()));
        assert_eq!(optimizer.learning_rate(), None);
        assert!(optimizer.multiply_by_parameter_scale());
        assert_eq!(optimizer.momentum(), None);
        assert!(optimizer.weight_decay().is_none());
        let optimizer = Adafactor::default()
            .with_learning_rate(0.5f64)
            .with_factored_rms(ScaleByFactoredRms::new().with_factored(false))
            .with_clipping_threshold(None)
            .unwrap()
            .with_multiply_by_parameter_scale(false)
            .with_momentum(0.9)
            .unwrap()
            .with_momentum_data_type(DataType::BF16)
            .unwrap()
            .with_weight_decay(0.5)
            .unwrap()
            .with_weight_decay_mask(|_| true);
        assert_eq!(optimizer.learning_rate(), Some(&0.5));
        assert!(!optimizer.factored_rms().factored());
        assert_eq!(optimizer.clipping(), None);
        assert!(!optimizer.multiply_by_parameter_scale());
        assert_eq!(
            optimizer.momentum(),
            Some(Ema::new(0.9).unwrap().with_debias(false).with_accumulator_data_type(DataType::BF16).unwrap()),
        );
        assert_eq!(optimizer.weight_decay().map(AddDecayedWeights::rate), Some(0.5));
        assert!(matches!(
            Adafactor::new().with_clipping_threshold(Some(0.0)),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`threshold` must be positive and finite but got `0`",
        ));
    }

    #[test]
    fn test_adafactor_initialize() {
        assert_eq!(state_paths(&Adafactor::new()).len(), 7);
        assert_eq!(
            state_paths(&Adafactor::new().with_learning_rate(0.5f64).with_momentum(0.9).unwrap())[7..],
            [
                "$.learning_rate[0].count",
                "$.momentum[0].count",
                "$.momentum[0].ema[\"bias\"]",
                "$.momentum[0].ema[\"weights\"]"
            ],
        );
    }

    #[test]
    fn test_adafactor_update() {
        // The vector parameter `[3]` is not factored. The gradient `2` gives the rescaled update `2 · 4^(-1/2) = 1`,
        // whose root mean square is not clipped, and which the parameter scale `3` and the sign flip turn into `-3`.
        // The gradient `1` gives `u = 1 · v^(-1/2)` with `v = β · 4 + (1 - β) · 1` and `β = 1 - 2^(-0.8)`.
        let decay = 1.0f32 - 2.0f32.powf(-0.8);
        let second = 1.0 * (decay * 4.0 + (1.0 - decay) * 1.0).powf(-0.5);
        assert_eq!(
            vector_updates(&Adafactor::new(), &[3.0], &[&[2.0], &[1.0]]),
            vec![vec![-(2.0 * 4.0f32.powf(-0.5) * 3.0)], vec![-(second * 3.0)],],
        );

        // A learning rate of `0.5` scales the updates before the parameter scale.
        assert_eq!(
            vector_updates(&Adafactor::new().with_learning_rate(0.5f64), &[3.0], &[&[2.0], &[1.0]]),
            vec![vec![-(0.5 * (2.0 * 4.0f32.powf(-0.5)) * 3.0)], vec![-(0.5 * second * 3.0)],],
        );

        // A weight decay mask is kept whether it is set before or after the weight decay rate.
        let excluded = |path: &ParameterPath| path.to_string() != "$[0]";
        let masked_first = Adafactor::new().with_weight_decay_mask(excluded).with_weight_decay(0.5).unwrap();
        let masked_last = Adafactor::new().with_weight_decay(0.5).unwrap().with_weight_decay_mask(excluded);
        assert_eq!(
            vector_updates(&masked_first, &[3.0], &[&[2.0], &[1.0]]),
            vector_updates(&masked_last, &[3.0], &[&[2.0], &[1.0]]),
        );
    }

    #[test]
    fn test_sm3_new() {
        let optimizer = Sm3::new(0.1).unwrap();
        assert_eq!(optimizer.learning_rate(), 0.1);
        assert_eq!(optimizer.sm3(), &ScaleBySm3::new());
        let optimizer = optimizer.with_momentum(0.5).unwrap().with_epsilon(0.0).unwrap();
        assert_eq!(optimizer.sm3(), &ScaleBySm3::new().with_beta1(0.5).unwrap().with_epsilon(0.0).unwrap());
        assert!(matches!(
            Sm3::new(f64::INFINITY),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`learning_rate` must be finite but got `inf`",
        ));
    }

    #[test]
    fn test_sm3_update() {
        // The SM3 momenta of the transformation tests are scaled by `-0.5`.
        let optimizer = Sm3::new(0.5).unwrap().with_momentum(0.5).unwrap().with_epsilon(0.0).unwrap();
        let second = [5.0f32.sqrt().recip() * 1.0, 9.0f32.sqrt().recip() * 3.0];
        assert_eq!(
            vector_updates(&optimizer, &[0.0, 0.0], &[&[2.0, 0.0], &[1.0, 3.0]]),
            vec![vec![-0.25, -0.0], vec![-0.5 * (0.5 * second[0] + 0.5 * 0.5), -0.5 * (0.5 * second[1] + 0.5 * 0.0)],],
        );
    }

    #[test]
    fn test_adafactor_optax_parity() {
        // Optax 9ecde04b: `optax.adafactor()`.
        const ADAFACTOR: [[f32; 9]; 4] = [
            [-1.3228756, 1.3228756, -1.3228756, -1.1636868, 1.1636868, -1.1636868, 1.1636868, -1.1636868, 1.1636868],
            [0.4260401, -0.6986278, -0.30629817, -0.6502671, -1.244509, 1.1892445, -0.46750435, 0.5455744, -0.7262876],
            [
                -0.5294913,
                0.19005878,
                0.34597233,
                0.5452712,
                -0.24752602,
                -1.0646795,
                -0.99989873,
                -0.965114,
                0.24763311,
            ],
            [
                -0.6305648,
                -0.54084456,
                0.84630877,
                -0.5678469,
                1.5844336,
                -0.19112726,
                1.7065395,
                -0.7336956,
                -1.0457053,
            ],
        ];
        // Optax 9ecde04b: `optax.adafactor(0.1, min_dim_size_to_factor=2)`.
        const ADAFACTOR_FACTORED: [[f32; 9]; 4] = [
            [
                -0.13228756,
                0.13228756,
                -0.13228756,
                -0.18970524,
                0.046505325,
                -0.07498758,
                0.077929385,
                -0.12736009,
                0.12321717,
            ],
            [
                0.080043346,
                -0.13125643,
                -0.057546526,
                -0.095419526,
                -0.12221555,
                0.12165738,
                -0.03830637,
                0.06541825,
                -0.08546931,
            ],
            [
                -0.169601,
                0.060877595,
                0.11081816,
                0.07578816,
                -0.020368319,
                -0.14458574,
                -0.10885752,
                -0.14042795,
                0.027689897,
            ],
            [
                -0.114628784,
                -0.09831876,
                0.15384834,
                -0.048134316,
                0.1379595,
                -0.019655736,
                0.16754217,
                -0.07114055,
                -0.09122155,
            ],
        ];
        // Optax 9ecde04b: `optax.adafactor(0.1, min_dim_size_to_factor=2, multiply_by_parameter_scale=False,
        // clipping_threshold=None, momentum=0.9, weight_decay_rate=0.01)`.
        const ADAFACTOR_VARIANT: [[f32; 9]; 4] = [
            [
                -0.015000001,
                0.02,
                -0.030000001,
                -0.026653327,
                0.009082483,
                -0.009082805,
                0.026841052,
                -0.02618034,
                0.0033166544,
            ],
            [
                -0.007222909,
                0.007932782,
                -0.033464495,
                -0.033411615,
                -0.0025471435,
                0.0027460228,
                0.022399856,
                -0.018842656,
                -0.005582136,
            ],
            [
                -0.021783732,
                0.013377825,
                -0.02203704,
                -0.023913812,
                -0.0036024535,
                -0.010761866,
                0.0121478485,
                -0.030835867,
                -0.0032384829,
            ],
            [
                -0.029807126,
                0.0043510827,
                -0.008206138,
                -0.02649051,
                0.009517895,
                -0.011564276,
                0.02761557,
                -0.035205767,
                -0.011717108,
            ],
        ];
        assert_parity(&parity_updates(&Adafactor::new()), &ADAFACTOR);
        assert_parity(
            &parity_updates(
                &Adafactor::new()
                    .with_learning_rate(0.1)
                    .with_factored_rms(ScaleByFactoredRms::new().with_minimum_dimension_size_to_factor(2)),
            ),
            &ADAFACTOR_FACTORED,
        );
        assert_parity(
            &parity_updates(
                &Adafactor::new()
                    .with_learning_rate(0.1)
                    .with_factored_rms(ScaleByFactoredRms::new().with_minimum_dimension_size_to_factor(2))
                    .with_multiply_by_parameter_scale(false)
                    .with_clipping_threshold(None)
                    .unwrap()
                    .with_momentum(0.9)
                    .unwrap()
                    .with_weight_decay(0.01)
                    .unwrap(),
            ),
            &ADAFACTOR_VARIANT,
        );
    }

    #[test]
    fn test_sm3_optax_parity() {
        // Optax 9ecde04b: `optax.sm3(0.1)`.
        const SM3: [[f32; 9]; 4] = [
            [
                -0.009999995,
                0.009999999,
                -0.010000001,
                -0.01,
                0.009999995,
                -0.009999999,
                0.010000001,
                -0.010000001,
                0.010000001,
            ],
            [
                -0.004527861,
                0.0011913114,
                -0.0121622775,
                -0.013472134,
                0.002999995,
                -0.0019289308,
                0.0065746433,
                -0.005286093,
                0.0039612893,
            ],
            [
                -0.013445499,
                0.0040536043,
                -0.005600824,
                -0.008944926,
                0.001775495,
                -0.007961468,
                0.0010376789,
                -0.010720332,
                0.0049901013,
            ],
            [
                -0.017399078,
                -0.00043423884,
                0.0017905597,
                -0.009966086,
                0.005680429,
                -0.0078553865,
                0.007533032,
                -0.012505442,
                0.00056036003,
            ],
        ];
        assert_parity(&parity_updates(&Sm3::new(0.1).unwrap()), &SM3);
    }
}
