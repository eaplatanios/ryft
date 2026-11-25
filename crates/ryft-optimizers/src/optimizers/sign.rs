//! Optimizers that follow the signs of the gradients or of their moments: sign SGD and Signum, Lion, and Rprop.

use ryft_core::{DataType, Parameter, ParameterPath, Parameterized};

use crate::macros::parameterized_state;
use crate::{
    AddDecayedWeights, Ema, EmaState, LionMode, Optimizer, OptimizerError, OptimizerValue, Scale, ScaleByLion,
    ScaleByLionState, ScaleByRprop, ScaleByRpropState, ScaleBySchedule, ScaleByScheduleState, ScaleBySign, Schedule,
    UpdateArguments, validate_decay_rate, validate_floating_point,
};

// TODO(eaplatanios): Review this module.

parameterized_state! {
    /// State of [`SignSgd`] for the parameter structure `S`, which holds the state of its momentum (one element for
    /// Signum and none for sign SGD) and the step count of its learning rate schedule.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct SignSgdState<P, S> with family SignSgdStateFamily {
        /// State of the [`Ema`] transformation of the momentum, which has one element with momentum and none without.
        momentum: Vec<EmaState<P, S>>,

        /// State of the [`ScaleBySchedule`] transformation that applies the learning rate.
        learning_rate: ScaleByScheduleState<P>,
    }
}

impl<P: Parameter, S> SignSgdState<P, S> {
    /// Returns the state of the [`Ema`] transformation of the momentum, if the optimizer has momentum.
    #[inline]
    pub fn momentum(&self) -> Option<&EmaState<P, S>> {
        self.momentum.first()
    }

    /// Returns the state of the [`ScaleBySchedule`] transformation that applies the learning rate.
    #[inline]
    pub fn learning_rate(&self) -> &ScaleByScheduleState<P> {
        &self.learning_rate
    }
}

/// Sign SGD of [Bernstein et al. (2018)](https://arxiv.org/abs/1802.04434), which descends along the signs of the
/// gradients, and which is the analogue of
/// [`optax.sign_sgd`](https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.sign_sgd) and, with momentum, of
/// its Signum variant [`optax.signum`](https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.signum). Its
/// update is
///
/// ```text
/// uₜ = −ηₜ · sign(gₜ)
/// ```
///
/// and with momentum (i.e., Signum) it is
///
/// ```text
/// mₜ = β · mₜ₋₁ + (1 − β) · gₜ
/// uₜ = −ηₜ · sign(mₜ)
/// ```
///
/// Here `t ≥ 1` counts the updates including the current one, `gₜ` is the gradient at the parameters `θₜ₋₁`, `uₜ` is
/// the update that is added to them (i.e., `θₜ = θₜ₋₁ + uₜ`), and `ηₜ` is the learning rate [`Schedule`] evaluated at
/// the `t − 1` updates applied before the current one. The momentum `mₜ` starts at zero and decays with the rate `β`,
/// and `sign` maps every element to `−1`, `0`, or `1`. All operations are element-wise. By default, this optimizer has
/// no momentum. [`Self::with_momentum`] sets `β` (Optax uses `β = 0.9` for Signum), and
/// [`Self::with_accumulator_data_type`] only changes the data type in which `mₜ` is stored.
///
/// It is the chain of [`ScaleBySign`] and [`ScaleBySchedule::learning_rate`] without momentum, preceded by an [`Ema`]
/// without debiasing with momentum.
///
/// # References
///
///   - Bernstein et al. (2018). [signSGD: Compressed Optimisation for Non-Convex
///     Problems](https://arxiv.org/abs/1802.04434).
#[derive(Clone, Debug)]
pub struct SignSgd<Sc: Schedule> {
    /// Decay of the momentum, or [`None`] without momentum.
    momentum: Option<f64>,

    /// Data type in which the momentum is stored, or [`None`] for the statistic data type of each leaf.
    accumulator_data_type: Option<DataType>,

    /// Transformation that scales the updates by the negated learning rate.
    learning_rate: ScaleBySchedule<Sc>,
}

impl<Sc: Schedule> SignSgd<Sc> {
    /// Creates a new [`SignSgd`] without momentum with the learning rate schedule `learning_rate` (e.g., an `f64` for
    /// a constant learning rate).
    pub fn new(learning_rate: Sc) -> Self {
        Self {
            momentum: None,
            accumulator_data_type: None,
            learning_rate: ScaleBySchedule::learning_rate(learning_rate),
        }
    }

    /// Returns this optimizer with the momentum decay `momentum`, which turns it into Signum.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `momentum` is not in `[0, 1)`.
    pub fn with_momentum(mut self, momentum: f64) -> Result<Self, OptimizerError> {
        validate_decay_rate("momentum", momentum)?;
        self.momentum = Some(momentum);
        Ok(self)
    }

    /// Returns this optimizer with its momentum stored in `data_type` (refer to [`Ema::with_accumulator_data_type`]).
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `data_type` is not a floating-point data type.
    pub fn with_accumulator_data_type(mut self, data_type: DataType) -> Result<Self, OptimizerError> {
        validate_floating_point("accumulator_data_type", data_type)?;
        self.accumulator_data_type = Some(data_type);
        Ok(self)
    }

    /// Returns the [`Ema`] transformation of the momentum of this optimizer, if it has momentum.
    pub fn momentum(&self) -> Option<Ema> {
        // The configuration was validated by the builder functions, so the moving average can be built directly.
        self.momentum.map(|momentum| {
            let ema = Ema::new(momentum).unwrap().with_debias(false);
            match self.accumulator_data_type {
                Some(data_type) => ema.with_accumulator_data_type(data_type).unwrap(),
                None => ema,
            }
        })
    }

    /// Returns the learning rate [`Schedule`] of this optimizer.
    #[inline]
    pub fn learning_rate(&self) -> &Sc {
        self.learning_rate.schedule()
    }
}

impl<P: OptimizerValue, Sc: Schedule> Optimizer<P> for SignSgd<Sc> {
    type State<S: Parameterized<P>> = SignSgdState<P, S>;

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        let momentum = self.momentum().map(|ema| ema.initialize(parameters)).into_iter().collect::<Result<_, _>>()?;
        let learning_rate = self.learning_rate.initialize(parameters)?;
        Ok(SignSgdState { momentum, learning_rate })
    }

    fn update<S: Parameterized<P>>(
        &self,
        gradients: S,
        state: Self::State<S>,
        parameters: &S,
        arguments: &UpdateArguments<'_, P, S>,
    ) -> Result<(S, Self::State<S>), OptimizerError> {
        let SignSgdState { momentum: momentum_states, learning_rate } = state;
        let (updates, momentum) = match (self.momentum(), momentum_states.into_iter().next()) {
            (Some(ema), Some(momentum_state)) => {
                let (updates, momentum_state) = ema.update(gradients, momentum_state, parameters, arguments)?;
                (updates, vec![momentum_state])
            }
            (None, None) => (gradients, Vec::new()),
            _ => {
                return Err(OptimizerError::StructureMismatch {
                    message: "the momentum state of `SignSgd` does not match its momentum configuration".to_string(),
                });
            }
        };
        let (updates, ()) = ScaleBySign::new().update(updates, (), parameters, arguments)?;
        let (updates, learning_rate) = self.learning_rate.update(updates, learning_rate, parameters, arguments)?;
        Ok((updates, SignSgdState { momentum, learning_rate }))
    }
}

parameterized_state! {
    /// State of [`Lion`] for the parameter structure `S`, which holds the state of its [`ScaleByLion`] transformation
    /// and the step count of its learning rate schedule. Weight decay is stateless.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct LionState<P, S> with family LionStateFamily {
        /// State of the [`ScaleByLion`] transformation of the optimizer.
        lion: ScaleByLionState<P, S>,

        /// State of the [`ScaleBySchedule`] transformation that applies the learning rate.
        learning_rate: ScaleByScheduleState<P>,
    }
}

impl<P: Parameter, S> LionState<P, S> {
    /// Returns the state of the [`ScaleByLion`] transformation of the optimizer.
    #[inline]
    pub fn lion(&self) -> &ScaleByLionState<P, S> {
        &self.lion
    }

    /// Returns the state of the [`ScaleBySchedule`] transformation that applies the learning rate.
    #[inline]
    pub fn learning_rate(&self) -> &ScaleByScheduleState<P> {
        &self.learning_rate
    }
}

/// Lion optimizer of [Chen et al. (2023)](https://arxiv.org/abs/2302.06675), which descends along the sign of an
/// interpolation between the gradient and its momentum, and which is the analogue of
/// [`optax.lion`](https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.lion). Its update is
///
/// ```text
/// cₜ = sign(β₁ · mₜ₋₁ + (1 − β₁) · gₜ)
/// mₜ = β₂ · mₜ₋₁ + (1 − β₂) · gₜ
/// uₜ = −ηₜ · (cₜ + λ · θₜ₋₁)
/// ```
///
/// Here `t ≥ 1` counts the updates including the current one, `gₜ` is the gradient at the parameters `θₜ₋₁`, `uₜ` is
/// the update that is added to them (i.e., `θₜ = θₜ₋₁ + uₜ`), and `ηₜ` is the learning rate [`Schedule`] evaluated at
/// the `t − 1` updates applied before the current one. The momentum `mₜ` starts at zero and decays with the rate `β₂`,
/// `β₁` is the interpolation coefficient of the update, which reads the momentum before it is updated, and `λ` is the
/// rate of the decoupled weight decay. All operations are element-wise. The defaults are `β₁ = 0.9`, `β₂ = 0.99`, and
/// `λ = 1e-3`, with [`LionMode::Hard`]. [`Self::with_mode`] replaces the sign of the interpolation `x` with
/// `tanh(κ · x)` for the sharpness `κ` of [`LionMode::Smooth`], or with `x` itself where `|x| < 1` and `sign(x)`
/// elsewhere for [`LionMode::Refined`]. [`Self::with_weight_decay_mask`] restricts the weight decay to the leaves that
/// the mask selects, and [`Self::with_first_moment_data_type`] only changes the data type in which `mₜ` is stored.
///
/// It is the chain of [`ScaleByLion`], [`AddDecayedWeights`], and [`ScaleBySchedule::learning_rate`].
///
/// # References
///
///   - Chen et al. (2023). [Symbolic Discovery of Optimization Algorithms](https://arxiv.org/abs/2302.06675).
#[derive(Clone, Debug)]
pub struct Lion<Sc: Schedule> {
    /// Transformation that rescales the gradients.
    lion: ScaleByLion,

    /// Transformation that adds the decayed weights to the rescaled gradients.
    weight_decay: AddDecayedWeights,

    /// Transformation that scales the updates by the negated learning rate.
    learning_rate: ScaleBySchedule<Sc>,
}

impl<Sc: Schedule> Lion<Sc> {
    /// Creates a new [`Lion`] with the learning rate schedule `learning_rate` (e.g., an `f64` for a constant learning
    /// rate) and default values for all other hyperparameters.
    pub fn new(learning_rate: Sc) -> Self {
        Self {
            lion: ScaleByLion::new(),
            weight_decay: AddDecayedWeights::new(1e-3).unwrap(),
            learning_rate: ScaleBySchedule::learning_rate(learning_rate),
        }
    }

    /// Returns this optimizer with `beta1` as the decay rate of the first moment estimates (refer to
    /// [`ScaleByLion::with_beta1`]).
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `beta1` is not in `[0, 1)`.
    pub fn with_beta1(mut self, beta1: f64) -> Result<Self, OptimizerError> {
        self.lion = self.lion.with_beta1(beta1)?;
        Ok(self)
    }

    /// Returns this optimizer with `beta2` as the decay rate of the second moment estimates (refer to
    /// [`ScaleByLion::with_beta2`]).
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `beta2` is not in `[0, 1)`.
    pub fn with_beta2(mut self, beta2: f64) -> Result<Self, OptimizerError> {
        self.lion = self.lion.with_beta2(beta2)?;
        Ok(self)
    }

    /// Returns this optimizer with its first moments stored in `data_type` (refer to
    /// [`ScaleByLion::with_first_moment_data_type`]).
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `data_type` is not a floating-point data type.
    pub fn with_first_moment_data_type(mut self, data_type: DataType) -> Result<Self, OptimizerError> {
        self.lion = self.lion.with_first_moment_data_type(data_type)?;
        Ok(self)
    }

    /// Returns this optimizer with `mode` as the function that maps the interpolation of the gradient and its momentum
    /// to the update (refer to [`ScaleByLion::with_mode`]).
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when the sharpness of [`LionMode::Smooth`] is not finite.
    pub fn with_mode(mut self, mode: LionMode) -> Result<Self, OptimizerError> {
        self.lion = self.lion.with_mode(mode)?;
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

    /// Returns the [`ScaleByLion`] transformation of this optimizer, which holds its rescaling hyperparameters.
    #[inline]
    pub fn lion(&self) -> &ScaleByLion {
        &self.lion
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

impl<P: OptimizerValue, Sc: Schedule> Optimizer<P> for Lion<Sc> {
    type State<S: Parameterized<P>> = LionState<P, S>;

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        let lion = self.lion.initialize(parameters)?;
        let learning_rate = self.learning_rate.initialize(parameters)?;
        Ok(LionState { lion, learning_rate })
    }

    fn update<S: Parameterized<P>>(
        &self,
        gradients: S,
        state: Self::State<S>,
        parameters: &S,
        arguments: &UpdateArguments<'_, P, S>,
    ) -> Result<(S, Self::State<S>), OptimizerError> {
        let (updates, lion) = self.lion.update(gradients, state.lion, parameters, arguments)?;
        let (updates, ()) = self.weight_decay.update(updates, (), parameters, arguments)?;
        let (updates, learning_rate) =
            self.learning_rate.update(updates, state.learning_rate, parameters, arguments)?;
        Ok((updates, LionState { lion, learning_rate }))
    }
}

parameterized_state! {
    /// State of [`Rprop`] for the parameter structure `S`, which holds the state of its [`ScaleByRprop`]
    /// transformation.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct RpropState<P, S> with family RpropStateFamily {
        /// State of the [`ScaleByRprop`] transformation of the optimizer.
        rprop: ScaleByRpropState<P, S>,
    }
}

impl<P: Parameter, S> RpropState<P, S> {
    /// Returns the state of the [`ScaleByRprop`] transformation of the optimizer.
    #[inline]
    pub fn rprop(&self) -> &ScaleByRpropState<P, S> {
        &self.rprop
    }
}

/// Resilient backpropagation (Rprop) of [Riedmiller and Braun (1993)](https://doi.org/10.1109/ICNN.1993.298623), in the
/// iRprop− form of [Igel and Hüsken (2003)](https://doi.org/10.1016/S0925-2312(01)00700-7), which adapts a step size
/// per element from the signs of consecutive gradients, and which is the analogue of
/// [`optax.rprop`](https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.rprop). Its update is
///
/// ```text
/// pₜ = gₜ · dₜ₋₁
/// sₜ = clip(η₊ · sₜ₋₁, sₘᵢₙ, sₘₐₓ)   if pₜ > 0
/// sₜ = clip(η₋ · sₜ₋₁, sₘᵢₙ, sₘₐₓ)   if pₜ < 0
/// sₜ = sₜ₋₁                          if pₜ = 0
/// dₜ = 0                             if pₜ < 0
/// dₜ = sₜ · sign(gₜ)                 otherwise
/// uₜ = 0                             if pₜ < 0
/// uₜ = −dₜ₋₁                         otherwise
/// ```
///
/// Here `t ≥ 1` counts the updates including the current one, `gₜ` is the gradient at the parameters `θₜ₋₁`, and `uₜ`
/// is the update that is added to them. The step sizes `sₜ` start at the constant learning rate `η` (i.e., `s₀ = η`)
/// and are clipped to `[sₘᵢₙ, sₘₐₓ]`, `η₊` and `η₋` are the factors by which they grow and shrink, and `dₜ` is the
/// step, which starts at zero, is reset to zero where the gradient changes its sign, and is stored for the next update.
/// All operations are element-wise. The defaults are `η₋ = 0.5`, `η₊ = 1.2`, `sₘᵢₙ = 1e-6`, and `sₘₐₓ = 50`. Note that,
/// as in Optax, the update applies the step `dₜ₋₁` of the previous update rather than the new step `dₜ`, so the updates
/// lag the steps of the paper by one and the first update is zero. As in Optax, the learning rate is a constant rather
/// than a [`Schedule`].
///
/// It is the chain of [`ScaleByRprop`] and [`Scale`] with the factor `−1`.
///
/// # References
///
///   - Riedmiller and Braun (1993). [A direct adaptive method for faster backpropagation learning: the RPROP
///     algorithm](https://doi.org/10.1109/ICNN.1993.298623).
///   - Igel and Hüsken (2003). [Empirical evaluation of the improved Rprop learning
///     algorithms](https://doi.org/10.1016/S0925-2312(01)00700-7).
#[derive(Clone, Debug)]
pub struct Rprop {
    /// Transformation that adapts the step sizes from the signs of consecutive gradients.
    rprop: ScaleByRprop,
}

impl Rprop {
    /// Creates a new [`Rprop`] whose step sizes start at `learning_rate`, with the defaults of [`ScaleByRprop`].
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `learning_rate` is negative or not finite.
    pub fn new(learning_rate: f64) -> Result<Self, OptimizerError> {
        Ok(Self { rprop: ScaleByRprop::new(learning_rate)? })
    }

    /// Returns this optimizer with `eta_minus` as the factor by which its step sizes shrink.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `eta_minus` is not positive and finite.
    pub fn with_eta_minus(mut self, eta_minus: f64) -> Result<Self, OptimizerError> {
        self.rprop = self.rprop.with_eta_minus(eta_minus)?;
        Ok(self)
    }

    /// Returns this optimizer with `eta_plus` as the factor by which its step sizes grow.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `eta_plus` is not positive and finite.
    pub fn with_eta_plus(mut self, eta_plus: f64) -> Result<Self, OptimizerError> {
        self.rprop = self.rprop.with_eta_plus(eta_plus)?;
        Ok(self)
    }

    /// Returns this optimizer with its step sizes clipped to `[minimum_step_size, maximum_step_size]`.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] under the conditions of
    /// [`ScaleByRprop::with_step_size_bounds`].
    pub fn with_step_size_bounds(
        mut self,
        minimum_step_size: f64,
        maximum_step_size: f64,
    ) -> Result<Self, OptimizerError> {
        self.rprop = self.rprop.with_step_size_bounds(minimum_step_size, maximum_step_size)?;
        Ok(self)
    }

    /// Returns the [`ScaleByRprop`] transformation of this optimizer.
    #[inline]
    pub fn rprop(&self) -> &ScaleByRprop {
        &self.rprop
    }
}

impl<P: OptimizerValue> Optimizer<P> for Rprop {
    type State<S: Parameterized<P>> = RpropState<P, S>;

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        Ok(RpropState { rprop: self.rprop.initialize(parameters)? })
    }

    fn update<S: Parameterized<P>>(
        &self,
        gradients: S,
        state: Self::State<S>,
        parameters: &S,
        arguments: &UpdateArguments<'_, P, S>,
    ) -> Result<(S, Self::State<S>), OptimizerError> {
        let (updates, rprop) = self.rprop.update(gradients, state.rprop, parameters, arguments)?;
        let (updates, ()) = Scale::new(-1.0)?.update(updates, (), parameters, arguments)?;
        Ok((updates, RpropState { rprop }))
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::tests::{assert_parity, parity_updates, state_paths, vector_updates};

    use super::*;

    #[test]
    fn test_sign_sgd_new() {
        let optimizer = SignSgd::new(0.1f64);
        assert_eq!(optimizer.momentum(), None);
        assert_eq!(optimizer.learning_rate(), &0.1);
        let optimizer = optimizer.with_momentum(0.9).unwrap().with_accumulator_data_type(DataType::BF16).unwrap();
        assert_eq!(
            optimizer.momentum(),
            Some(Ema::new(0.9).unwrap().with_debias(false).with_accumulator_data_type(DataType::BF16).unwrap()),
        );
        assert!(matches!(
            SignSgd::new(0.1f64).with_momentum(1.0),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`momentum` must be in `[0, 1)` but got `1`",
        ));
    }

    #[test]
    fn test_sign_sgd_initialize() {
        assert_eq!(state_paths(&SignSgd::new(0.1f64)), vec!["$.learning_rate.count"]);
        assert_eq!(
            state_paths(&SignSgd::new(0.1f64).with_momentum(0.9).unwrap()),
            vec![
                "$.momentum[0].count",
                "$.momentum[0].ema[\"bias\"]",
                "$.momentum[0].ema[\"weights\"]",
                "$.learning_rate.count",
            ],
        );
    }

    #[test]
    fn test_sign_sgd_update() {
        // Sign SGD scales the signs of the gradients by `-0.5`.
        let gradients: [&[f32]; 2] = [&[2.0, -1.0], &[-3.0, 0.0]];
        assert_eq!(
            vector_updates(&SignSgd::new(0.5f64), &[0.0, 0.0], &gradients),
            vec![vec![-0.5, 0.5], vec![0.5, -0.0],],
        );

        // Signum with the momentum `0.5` takes the signs of the averages `[1, -0.5]` and `[-1, -0.25]`.
        let optimizer = SignSgd::new(0.5f64).with_momentum(0.5).unwrap();
        assert_eq!(vector_updates(&optimizer, &[0.0, 0.0], &gradients), vec![vec![-0.5, 0.5], vec![0.5, 0.5]]);
    }

    #[test]
    fn test_lion_new() {
        let optimizer = Lion::new(0.1f64);
        assert_eq!(optimizer.lion(), &ScaleByLion::new());
        assert_eq!(optimizer.weight_decay().rate(), 1e-3);
        let optimizer = optimizer.with_mode(LionMode::Refined).unwrap().with_weight_decay(0.0).unwrap();
        assert_eq!(optimizer.lion().mode(), LionMode::Refined);
        assert_eq!(optimizer.weight_decay().rate(), 0.0);
        assert!(matches!(
            Lion::new(0.1f64).with_beta2(-1.0),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`beta2` must be in `[0, 1)` but got `-1`",
        ));
    }

    #[test]
    fn test_lion_update() {
        // Without weight decay, the Lion updates `[1, -1]` and `[1, 0]` of the transformation tests are scaled by
        // `-0.5`, and the weight decay `0.5` adds `0.5 · [2, 2]` before the learning rate.
        let optimizer =
            Lion::new(0.5f64).with_beta1(0.5).unwrap().with_beta2(0.5).unwrap().with_weight_decay(0.0).unwrap();
        assert_eq!(state_paths(&optimizer)[0], "$.lion.count");
        let gradients: [&[f32]; 2] = [&[2.0, -0.5], &[-0.5, 0.25]];
        assert_eq!(vector_updates(&optimizer, &[2.0, 2.0], &gradients), vec![vec![-0.5, 0.5], vec![-0.5, -0.0]]);
        let optimizer = optimizer.with_weight_decay(0.5).unwrap();
        assert_eq!(vector_updates(&optimizer, &[2.0, 2.0], &gradients[..1]), vec![vec![-1.0, -0.0]]);
    }

    #[test]
    fn test_rprop_new() {
        let optimizer = Rprop::new(0.5)
            .unwrap()
            .with_eta_minus(0.25)
            .unwrap()
            .with_eta_plus(2.0)
            .unwrap()
            .with_step_size_bounds(0.1, 0.8)
            .unwrap();
        assert_eq!(
            optimizer.rprop(),
            &ScaleByRprop::new(0.5)
                .unwrap()
                .with_eta_minus(0.25)
                .unwrap()
                .with_eta_plus(2.0)
                .unwrap()
                .with_step_size_bounds(0.1, 0.8)
                .unwrap(),
        );
        assert!(matches!(
            Rprop::new(f64::NAN),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`learning_rate` must be finite and non-negative but got `NaN`",
        ));
    }

    #[test]
    fn test_rprop_update() {
        // The Rprop updates `[0, 0]` and `[0.5, 0]` of the transformation tests are negated.
        let optimizer = Rprop::new(0.5)
            .unwrap()
            .with_eta_minus(0.25)
            .unwrap()
            .with_eta_plus(2.0)
            .unwrap()
            .with_step_size_bounds(0.1, 0.8)
            .unwrap();
        assert_eq!(state_paths(&optimizer)[0], "$.rprop.step_sizes[\"bias\"]");
        assert_eq!(
            vector_updates(&optimizer, &[0.0, 0.0], &[&[3.0, -1.0], &[1.0, 2.0]]),
            vec![vec![-0.0, -0.0], vec![-0.5, -0.0]],
        );

        // As in Optax, the first update is zero and the second one is the first stored step.
        assert_eq!(vector_updates(&Rprop::new(0.1).unwrap(), &[0.0], &[&[1.0], &[1.0]]), vec![vec![-0.0], vec![-0.1]]);
    }

    #[test]
    fn test_sign_sgd_optax_parity() {
        // Optax 9ecde04b: `optax.sign_sgd(0.1)`.
        const SIGN_SGD: [[f32; 9]; 4] = [
            [-0.1, 0.1, -0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1],
            [0.1, -0.1, -0.1, -0.1, -0.1, 0.1, -0.1, 0.1, -0.1],
            [-0.1, 0.1, 0.1, 0.1, -0.1, -0.1, -0.1, -0.1, 0.1],
            [-0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1, -0.1],
        ];
        // Optax 9ecde04b: `optax.signum(0.1, beta=0.9)`.
        const SIGNUM: [[f32; 9]; 4] = [
            [-0.1, 0.1, -0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1],
            [-0.1, -0.1, -0.1, -0.1, -0.1, 0.1, 0.1, -0.1, 0.1],
            [-0.1, 0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, 0.1],
            [-0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1, -0.1],
        ];
        assert_parity(&parity_updates(&SignSgd::new(0.1)), &SIGN_SGD);
        assert_parity(&parity_updates(&SignSgd::new(0.1).with_momentum(0.9).unwrap()), &SIGNUM);
    }

    #[test]
    fn test_lion_optax_parity() {
        // Optax 9ecde04b: `optax.lion(0.1)`.
        const LION: [[f32; 9]; 4] = [
            [-0.100049995, 0.1001, -0.1002, -0.1001, 0.100049995, -0.100025, 0.1002, -0.100150004, 0.099925004],
            [
                0.09996001,
                -0.09991001,
                -0.100189984,
                -0.10008999,
                -0.09996001,
                0.099985,
                -0.09981003,
                0.09986001,
                -0.100085,
            ],
            [
                -0.100049995,
                0.10009998,
                0.09982004,
                0.09992002,
                -0.09995001,
                -0.100025,
                -0.09980004,
                -0.10014997,
                0.09992502,
            ],
            [
                -0.10003998,
                -0.09991003,
                0.09981006,
                -0.10008997,
                0.100059986,
                -0.100015,
                0.100209944,
                -0.10013995,
                -0.100084975,
            ],
        ];
        // Optax 9ecde04b: `optax.lion(0.1, b1=0.5, b2=0.9, weight_decay=0.1, mask={'bias': False, 'weights': True})`.
        const LION_VARIANT: [[f32; 9]; 4] = [
            [-0.1, 0.1, -0.1, -0.11000001, 0.105, -0.1025, 0.120000005, -0.115, 0.0925],
            [0.1, -0.1, -0.1, -0.1089, -0.09605, 0.098525, -0.081199996, 0.086150005, -0.108425],
            [-0.1, 0.1, 0.1, 0.09218901, -0.0950895, -0.10246026, -0.080388, -0.1147115, 0.09265926],
            [-0.1, -0.1, 0.1, -0.108732894, 0.10586139, -0.101435654, 0.12041588, -0.11356439, -0.108267345],
        ];
        assert_parity(&parity_updates(&Lion::new(0.1)), &LION);
        assert_parity(
            &parity_updates(
                &Lion::new(0.1)
                    .with_beta1(0.5)
                    .unwrap()
                    .with_beta2(0.9)
                    .unwrap()
                    .with_weight_decay(0.1)
                    .unwrap()
                    .with_weight_decay_mask(|path| path.to_string() == "$[\"weights\"]"),
            ),
            &LION_VARIANT,
        );
    }

    #[test]
    fn test_rprop_optax_parity() {
        // Optax 9ecde04b: `optax.rprop(0.1)`.
        const RPROP: [[f32; 9]; 4] = [
            [-0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0],
            [-0.0, -0.0, -0.1, -0.1, -0.0, -0.0, -0.0, -0.0, -0.0],
            [-0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0],
            [-0.05, -0.0, -0.0, -0.0, -0.0, -0.05, -0.0, -0.05, -0.0],
        ];
        // Optax 9ecde04b: `optax.rprop(0.1, eta_minus=0.25, eta_plus=1.5, min_step_size=0.05, max_step_size=0.2)`.
        const RPROP_VARIANT: [[f32; 9]; 4] = [
            [-0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0],
            [-0.0, -0.0, -0.1, -0.1, -0.0, -0.0, -0.0, -0.0, -0.0],
            [-0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0],
            [-0.05, -0.0, -0.0, -0.0, -0.0, -0.05, -0.0, -0.05, -0.0],
        ];
        assert_parity(&parity_updates(&Rprop::new(0.1).unwrap()), &RPROP);
        assert_parity(
            &parity_updates(
                &Rprop::new(0.1)
                    .unwrap()
                    .with_eta_minus(0.25)
                    .unwrap()
                    .with_eta_plus(1.5)
                    .unwrap()
                    .with_step_size_bounds(0.05, 0.2)
                    .unwrap(),
            ),
            &RPROP_VARIANT,
        );
    }
}
