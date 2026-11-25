//! Stochastic gradient descent and its variants, which follow the gradients directly, with momentum, with gradient
//! noise, with optimistic steps, or with Polyak step sizes.

use std::marker::PhantomData;

use ryft_core::{DataType, Parameter, Parameterized, Random};

use crate::macros::parameterized_state;
use crate::{
    AddNoise, AddNoiseState, Optimizer, OptimizerError, OptimizerValue, PolyakVariant, ScaleByOptimisticGradient,
    ScaleByOptimisticGradientState, ScaleByPolyak, ScaleBySchedule, ScaleByScheduleState, Schedule, Trace, TraceState,
    UpdateArguments, validate_finite, validate_floating_point,
};

// TODO(eaplatanios): Review this module.

parameterized_state! {
    /// State of [`Sgd`] for the parameter structure `S`, which holds the state of its momentum (one element with
    /// momentum and none without) and the step count of its learning rate schedule.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct SgdState<P, S> with family SgdStateFamily {
        /// State of the [`Trace`] transformation of the momentum, which has one element with momentum and none without.
        momentum: Vec<TraceState<P, S>>,

        /// State of the [`ScaleBySchedule`] transformation that applies the learning rate.
        learning_rate: ScaleByScheduleState<P>,
    }
}

impl<P: Parameter, S> SgdState<P, S> {
    /// Returns the state of the [`Trace`] transformation of the momentum, if the optimizer has momentum.
    #[inline]
    pub fn momentum(&self) -> Option<&TraceState<P, S>> {
        self.momentum.first()
    }

    /// Returns the state of the [`ScaleBySchedule`] transformation that applies the learning rate.
    #[inline]
    pub fn learning_rate(&self) -> &ScaleByScheduleState<P> {
        &self.learning_rate
    }
}

/// Stochastic gradient descent, optionally with the heavy ball momentum of
/// [Polyak (1964)](https://doi.org/10.1016/0041-5553(64)90137-5) or with the Nesterov momentum of
/// [Sutskever et al. (2013)](https://proceedings.mlr.press/v28/sutskever13.html), which is the analogue of
/// [`optax.sgd`](https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.sgd). Its update is
///
/// ```text
/// mₜ = gₜ + μ · mₜ₋₁
/// uₜ = −ηₜ · mₜ              with heavy ball momentum
/// uₜ = −ηₜ · (gₜ + μ · mₜ)   with Nesterov momentum
/// ```
///
/// Here `t ≥ 1` counts the updates including the current one, `gₜ` is the gradient at the parameters `θₜ₋₁`, `uₜ` is
/// the update that is added to them (i.e., `θₜ = θₜ₋₁ + uₜ`), and `ηₜ` is the learning rate [`Schedule`] evaluated at
/// the `t − 1` updates applied before the current one. The momentum `mₜ` (i.e., the trace of the gradients) starts at
/// zero and decays with the rate `μ`. All operations are element-wise. By default, this optimizer has no momentum, so
/// its update is `uₜ = −ηₜ · gₜ` and it keeps no trace. [`Self::with_momentum`] sets `μ`, [`Self::with_nesterov`]
/// selects Nesterov momentum (which has an effect only with momentum), and [`Self::with_accumulator_data_type`] only
/// changes the data type in which `mₜ` is stored. Note that the learning rate scales the trace rather than being
/// accumulated in it, so a learning rate schedule also rescales the past gradients in `mₜ`.
///
/// It is [`ScaleBySchedule::learning_rate`] alone without momentum, and the chain of [`Trace`] and
/// [`ScaleBySchedule::learning_rate`] with momentum.
///
/// # References
///
///   - Polyak (1964). [Some methods of speeding up the convergence of iteration
///     methods](https://doi.org/10.1016/0041-5553(64)90137-5).
///   - Sutskever et al. (2013). [On the importance of initialization and momentum in deep
///     learning](https://proceedings.mlr.press/v28/sutskever13.html).
#[derive(Clone, Debug)]
pub struct Sgd<Sc: Schedule> {
    /// Decay of the momentum, or [`None`] without momentum.
    momentum: Option<f64>,

    /// Whether the momentum is Nesterov momentum.
    nesterov: bool,

    /// Data type in which the momentum is stored, or [`None`] for the statistic data type of each leaf.
    accumulator_data_type: Option<DataType>,

    /// Transformation that scales the updates by the negated learning rate.
    learning_rate: ScaleBySchedule<Sc>,
}

impl<Sc: Schedule> Sgd<Sc> {
    /// Creates a new [`Sgd`] without momentum with the learning rate schedule `learning_rate` (e.g., an `f64` for a
    /// constant learning rate).
    pub fn new(learning_rate: Sc) -> Self {
        Self {
            momentum: None,
            nesterov: false,
            accumulator_data_type: None,
            learning_rate: ScaleBySchedule::learning_rate(learning_rate),
        }
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

    /// Returns this optimizer with its momentum stored in `data_type` (refer to [`Trace::with_accumulator_data_type`]).
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `data_type` is not a floating-point data type.
    pub fn with_accumulator_data_type(mut self, data_type: DataType) -> Result<Self, OptimizerError> {
        validate_floating_point("accumulator_data_type", data_type)?;
        self.accumulator_data_type = Some(data_type);
        Ok(self)
    }

    /// Returns the [`Trace`] transformation of the momentum of this optimizer, if it has momentum.
    pub fn momentum(&self) -> Option<Trace> {
        // The configuration was validated by the builder functions, so the trace can be built directly.
        self.momentum.map(|momentum| {
            let trace = Trace::new(momentum).unwrap().with_nesterov(self.nesterov);
            match self.accumulator_data_type {
                Some(data_type) => trace.with_accumulator_data_type(data_type).unwrap(),
                None => trace,
            }
        })
    }

    /// Returns whether the momentum of this optimizer is Nesterov momentum.
    #[inline]
    pub fn nesterov(&self) -> bool {
        self.nesterov
    }

    /// Returns the learning rate [`Schedule`] of this optimizer.
    #[inline]
    pub fn learning_rate(&self) -> &Sc {
        self.learning_rate.schedule()
    }
}

impl<P: OptimizerValue, Sc: Schedule> Optimizer<P> for Sgd<Sc> {
    type State<S: Parameterized<P>> = SgdState<P, S>;

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        let momentum =
            self.momentum().map(|trace| trace.initialize(parameters)).into_iter().collect::<Result<_, _>>()?;
        let learning_rate = self.learning_rate.initialize(parameters)?;
        Ok(SgdState { momentum, learning_rate })
    }

    fn update<S: Parameterized<P>>(
        &self,
        gradients: S,
        state: Self::State<S>,
        parameters: &S,
        arguments: &UpdateArguments<'_, P, S>,
    ) -> Result<(S, Self::State<S>), OptimizerError> {
        let SgdState { momentum: momentum_states, learning_rate } = state;
        let (updates, momentum) = match (self.momentum(), momentum_states.into_iter().next()) {
            (Some(trace), Some(momentum_state)) => {
                let (updates, momentum_state) = trace.update(gradients, momentum_state, parameters, arguments)?;
                (updates, vec![momentum_state])
            }
            (None, None) => (gradients, Vec::new()),
            _ => {
                return Err(OptimizerError::StructureMismatch {
                    message: "the momentum state of `Sgd` does not match its momentum configuration".to_string(),
                });
            }
        };
        let (updates, learning_rate) = self.learning_rate.update(updates, learning_rate, parameters, arguments)?;
        Ok((updates, SgdState { momentum, learning_rate }))
    }
}

parameterized_state! {
    /// State of [`NoisySgd`] for the parameter structure `S`, which holds the state of its [`AddNoise`] transformation
    /// and the step count of its learning rate schedule.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct NoisySgdState<P, S> with family NoisySgdStateFamily and marker leaf {
        /// State of the [`AddNoise`] transformation of the optimizer.
        noise: AddNoiseState<P>,

        /// State of the [`ScaleBySchedule`] transformation that applies the learning rate.
        learning_rate: ScaleByScheduleState<P>,
    }
}

impl<P: Parameter, S> NoisySgdState<P, S> {
    /// Returns the state of the [`AddNoise`] transformation of the optimizer.
    #[inline]
    pub fn noise(&self) -> &AddNoiseState<P> {
        &self.noise
    }

    /// Returns the state of the [`ScaleBySchedule`] transformation that applies the learning rate.
    #[inline]
    pub fn learning_rate(&self) -> &ScaleByScheduleState<P> {
        &self.learning_rate
    }
}

/// Noisy SGD optimizer of [Neelakantan et al. (2015)](https://arxiv.org/abs/1511.06807), which adds annealed Gaussian
/// noise to the gradients, and which is the analogue of
/// [`optax.noisy_sgd`](https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.noisy_sgd). Its update is
///
/// ```text
/// σₜ = √(ν / tᵞ)
/// uₜ = −ηₜ · (gₜ + σₜ · ξₜ)
/// ```
///
/// Here `t ≥ 1` counts the updates including the current one, `gₜ` is the gradient at the parameters `θₜ₋₁`, `uₜ` is
/// the update that is added to them (i.e., `θₜ = θₜ₋₁ + uₜ`), and `ηₜ` is the learning rate [`Schedule`] evaluated at
/// the `t − 1` updates applied before the current one. The noise `ξₜ` has independent standard normal elements of the
/// shape of every leaf, `σₜ` is its standard deviation, `ν` is the scale of its variance (i.e., the `eta` of Optax),
/// and `γ` is the exponent with which the variance decays. All operations are element-wise. There are no defaults,
/// because the [`AddNoise`] transformation that holds `ν`, `γ`, and the seed of the noise is passed to [`Self::new`]
/// (Optax uses `ν = 0.01` and `γ = 0.55`). It requires leaf values with the [`Random`] capability.
///
/// It is the chain of [`AddNoise`] and [`ScaleBySchedule::learning_rate`].
///
/// # References
///
///   - Neelakantan et al. (2015). [Adding Gradient Noise Improves Learning for Very Deep
///     Networks](https://arxiv.org/abs/1511.06807).
#[derive(Clone, Debug)]
pub struct NoisySgd<Sc: Schedule> {
    /// Transformation that rescales the gradients.
    noise: AddNoise,

    /// Transformation that scales the updates by the negated learning rate.
    learning_rate: ScaleBySchedule<Sc>,
}

impl<Sc: Schedule> NoisySgd<Sc> {
    /// Creates a new [`NoisySgd`] with the learning rate schedule `learning_rate` (e.g., an `f64` for a constant
    /// learning rate) and the gradient noise `noise` (e.g., `AddNoise::new(0.01, 0.55, seed)?` for the defaults of
    /// `optax.noisy_sgd`).
    pub fn new(learning_rate: Sc, noise: AddNoise) -> Self {
        Self { noise, learning_rate: ScaleBySchedule::learning_rate(learning_rate) }
    }

    /// Returns the [`AddNoise`] transformation of this optimizer, which holds its rescaling hyperparameters.
    #[inline]
    pub fn noise(&self) -> &AddNoise {
        &self.noise
    }

    /// Returns the learning rate [`Schedule`] of this optimizer.
    #[inline]
    pub fn learning_rate(&self) -> &Sc {
        self.learning_rate.schedule()
    }
}

impl<P: OptimizerValue + Random, Sc: Schedule> Optimizer<P> for NoisySgd<Sc> {
    type State<S: Parameterized<P>> = NoisySgdState<P, S>;

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        let noise = self.noise.initialize(parameters)?;
        let learning_rate = self.learning_rate.initialize(parameters)?;
        Ok(NoisySgdState { noise, learning_rate, leaf: PhantomData })
    }

    fn update<S: Parameterized<P>>(
        &self,
        gradients: S,
        state: Self::State<S>,
        parameters: &S,
        arguments: &UpdateArguments<'_, P, S>,
    ) -> Result<(S, Self::State<S>), OptimizerError> {
        let (updates, noise) = self.noise.update(gradients, state.noise, parameters, arguments)?;
        let (updates, learning_rate) =
            self.learning_rate.update(updates, state.learning_rate, parameters, arguments)?;
        Ok((updates, NoisySgdState { noise, learning_rate, leaf: PhantomData }))
    }
}

parameterized_state! {
    /// State of [`OptimisticGradientDescent`] for the parameter structure `S`, which holds the state of its
    /// [`ScaleByOptimisticGradient`] transformation and the step count of its learning rate schedule.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct OptimisticGradientDescentState<P, S> with family OptimisticGradientDescentStateFamily {
        /// State of the [`ScaleByOptimisticGradient`] transformation of the optimizer.
        optimistic: ScaleByOptimisticGradientState<P, S>,

        /// State of the [`ScaleBySchedule`] transformation that applies the learning rate.
        learning_rate: ScaleByScheduleState<P>,
    }
}

impl<P: Parameter, S> OptimisticGradientDescentState<P, S> {
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

/// Optimistic gradient descent of [Mokhtari et al. (2019)](https://arxiv.org/abs/1901.08511) for min-max problems,
/// which extrapolates from the previous gradient, and which is the analogue of [`optax.optimistic_gradient_descent`](
/// https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.optimistic_gradient_descent). Its update is
///
/// ```text
/// uₜ = −ηₜ · ((α + β) · gₜ − β · gₜ₋₁)
/// ```
///
/// Here `t ≥ 1` counts the updates including the current one, `gₜ` is the gradient at the parameters `θₜ₋₁`, `uₜ` is
/// the update that is added to them (i.e., `θₜ = θₜ₋₁ + uₜ`), and `ηₜ` is the learning rate [`Schedule`] evaluated at
/// the `t − 1` updates applied before the current one. The coefficient `α` weights the current gradient and `β` weights
/// the optimism (i.e., the difference from the previous gradient), and the previous gradient is the current one at the
/// first update (i.e., `g₀ = g₁`). All operations are element-wise. The defaults are `α = 1` and `β = 1`, which give
/// the optimistic step `2 · gₜ − gₜ₋₁`.
///
/// It is the chain of [`ScaleByOptimisticGradient`] and [`ScaleBySchedule::learning_rate`].
///
/// # References
///
///   - Mokhtari, Ozdaglar, and Pattathil (2019). [A Unified Analysis of Extra-gradient and Optimistic Gradient Methods
///     for Saddle Point Problems: Proximal Point Approach](https://arxiv.org/abs/1901.08511).
#[derive(Clone, Debug)]
pub struct OptimisticGradientDescent<Sc: Schedule> {
    /// Transformation that rescales the gradients.
    optimistic: ScaleByOptimisticGradient,

    /// Transformation that scales the updates by the negated learning rate.
    learning_rate: ScaleBySchedule<Sc>,
}

impl<Sc: Schedule> OptimisticGradientDescent<Sc> {
    /// Creates a new [`OptimisticGradientDescent`] with the learning rate schedule `learning_rate` (e.g., an `f64` for
    /// a constant learning rate) and default values for all other hyperparameters.
    pub fn new(learning_rate: Sc) -> Self {
        Self {
            optimistic: ScaleByOptimisticGradient::new(),
            learning_rate: ScaleBySchedule::learning_rate(learning_rate),
        }
    }

    /// Returns this optimizer with `alpha` as the coefficient of the current gradient (refer to
    /// [`ScaleByOptimisticGradient::with_alpha`]).
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `alpha` is not finite.
    pub fn with_alpha(mut self, alpha: f64) -> Result<Self, OptimizerError> {
        self.optimistic = self.optimistic.with_alpha(alpha)?;
        Ok(self)
    }

    /// Returns this optimizer with `beta` as the coefficient of the optimism (refer to
    /// [`ScaleByOptimisticGradient::with_beta`]).
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `beta` is not finite.
    pub fn with_beta(mut self, beta: f64) -> Result<Self, OptimizerError> {
        self.optimistic = self.optimistic.with_beta(beta)?;
        Ok(self)
    }

    /// Returns the [`ScaleByOptimisticGradient`] transformation of this optimizer, which holds its rescaling
    /// hyperparameters.
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

impl<P: OptimizerValue, Sc: Schedule> Optimizer<P> for OptimisticGradientDescent<Sc> {
    type State<S: Parameterized<P>> = OptimisticGradientDescentState<P, S>;

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        let optimistic = self.optimistic.initialize(parameters)?;
        let learning_rate = self.learning_rate.initialize(parameters)?;
        Ok(OptimisticGradientDescentState { optimistic, learning_rate })
    }

    fn update<S: Parameterized<P>>(
        &self,
        gradients: S,
        state: Self::State<S>,
        parameters: &S,
        arguments: &UpdateArguments<'_, P, S>,
    ) -> Result<(S, Self::State<S>), OptimizerError> {
        let (updates, optimistic) = self.optimistic.update(gradients, state.optimistic, parameters, arguments)?;
        let (updates, learning_rate) =
            self.learning_rate.update(updates, state.learning_rate, parameters, arguments)?;
        Ok((updates, OptimisticGradientDescentState { optimistic, learning_rate }))
    }
}

parameterized_state! {
    /// State of [`PolyakSgd`] for the parameter structure `S`, which holds the step count of its scaling schedule. The
    /// Polyak step size is stateless.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct PolyakSgdState<P, S> with family PolyakSgdStateFamily and marker leaf {
        /// State of the [`ScaleBySchedule`] transformation that applies the scaling.
        learning_rate: ScaleByScheduleState<P>,
    }
}

impl<P: Parameter, S> PolyakSgdState<P, S> {
    /// Returns the state of the [`ScaleBySchedule`] transformation that applies the scaling.
    #[inline]
    pub fn learning_rate(&self) -> &ScaleByScheduleState<P> {
        &self.learning_rate
    }
}

/// SGD with the Polyak step size of [Loizou et al. (2021)](https://arxiv.org/abs/2002.10542), which sets the step size
/// from the gap between the value of the objective and its minimum, and which is the analogue of
/// [`optax.polyak_sgd`](https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.polyak_sgd). Its update is
///
/// ```text
/// dₜ = −sₜ · gₜ
/// γₜ = min((f(θₜ₋₁) − f*) / (‖dₜ‖² + ε), γₘₐₓ)   if ‖dₜ‖² + ε > ε₆₄
/// γₜ = 0                                         otherwise
/// uₜ = γₜ · dₜ
/// ```
///
/// Here `t ≥ 1` counts the updates including the current one, `gₜ` is the gradient at the parameters `θₜ₋₁`, `uₜ` is
/// the update that is added to them, and `sₜ` is the scaling [`Schedule`] evaluated at the `t − 1` updates applied
/// before the current one. The value `f(θₜ₋₁)` of the objective is read from the `value` [`UpdateArguments`], `f*` is
/// the minimum of the objective, `‖dₜ‖` is the global norm of `dₜ` over all leaves, `ε` keeps the denominator positive,
/// `γₘₐₓ` is the largest step size, and `ε₆₄` is the machine epsilon of `f64`. The scaling is passed to [`Self::new`]
/// (e.g., `1.0`, which is the default of Optax), and the other defaults are `f* = 0`, `ε = 0`, and `γₘₐₓ = 1`, with
/// [`PolyakVariant::Sps`]. [`Self::with_minimum_value`], [`Self::with_epsilon`], and
/// [`Self::with_maximum_learning_rate`] set `f*`, `ε`, and `γₘₐₓ`, and [`Self::with_variant`] with
/// [`PolyakVariant::SpsPlus`] clamps the gap `f(θₜ₋₁) − f*` from below at zero. Note that the Polyak step size is
/// computed from the scaled gradients `dₜ`, so the scaling enters it squared and the update is
/// `−min((f(θₜ₋₁) − f*) / (sₜ² · ‖gₜ‖² + ε), γₘₐₓ) · sₜ · gₜ`.
///
/// It is the chain of [`ScaleBySchedule::learning_rate`] with the scaling schedule and [`ScaleByPolyak`].
///
/// # References
///
///   - Loizou et al. (2021). [Stochastic Polyak Step-size for SGD: An Adaptive Learning Rate for Fast
///     Convergence](https://arxiv.org/abs/2002.10542).
///   - Berrada, Zisserman, and Kumar (2020). [Training Neural Networks for and by
///     Interpolation](https://arxiv.org/abs/1906.05661).
#[derive(Clone, Debug)]
pub struct PolyakSgd<Sc: Schedule> {
    /// Transformation that scales the gradients by the negated scaling.
    learning_rate: ScaleBySchedule<Sc>,

    /// Transformation that scales the updates by the Polyak step size.
    polyak: ScaleByPolyak,
}

impl<Sc: Schedule> PolyakSgd<Sc> {
    /// Creates a new [`PolyakSgd`] with the scaling schedule `scaling` (e.g., `1.0` for the default of Optax).
    pub fn new(scaling: Sc) -> Self {
        Self { learning_rate: ScaleBySchedule::learning_rate(scaling), polyak: ScaleByPolyak::new() }
    }

    /// Returns this optimizer with `maximum_learning_rate` as its largest step size.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `maximum_learning_rate` is negative or not finite.
    pub fn with_maximum_learning_rate(mut self, maximum_learning_rate: f64) -> Result<Self, OptimizerError> {
        self.polyak = self.polyak.with_maximum_learning_rate(maximum_learning_rate)?;
        Ok(self)
    }

    /// Returns this optimizer with `minimum_value` as the minimum of the objective.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `minimum_value` is not finite.
    pub fn with_minimum_value(mut self, minimum_value: f64) -> Result<Self, OptimizerError> {
        self.polyak = self.polyak.with_minimum_value(minimum_value)?;
        Ok(self)
    }

    /// Returns this optimizer with `epsilon` as the term that is added to the squared global norm.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `epsilon` is negative or not finite.
    pub fn with_epsilon(mut self, epsilon: f64) -> Result<Self, OptimizerError> {
        self.polyak = self.polyak.with_epsilon(epsilon)?;
        Ok(self)
    }

    /// Returns this optimizer with `variant` as the variant of its step size.
    #[inline]
    pub fn with_variant(mut self, variant: PolyakVariant) -> Self {
        self.polyak = self.polyak.with_variant(variant);
        self
    }

    /// Returns the [`ScaleByPolyak`] transformation of this optimizer.
    #[inline]
    pub fn polyak(&self) -> &ScaleByPolyak {
        &self.polyak
    }

    /// Returns the scaling [`Schedule`] of this optimizer.
    #[inline]
    pub fn scaling(&self) -> &Sc {
        self.learning_rate.schedule()
    }
}

impl<P: OptimizerValue, Sc: Schedule> Optimizer<P> for PolyakSgd<Sc> {
    type State<S: Parameterized<P>> = PolyakSgdState<P, S>;

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        Ok(PolyakSgdState { learning_rate: self.learning_rate.initialize(parameters)?, leaf: PhantomData })
    }

    fn update<S: Parameterized<P>>(
        &self,
        gradients: S,
        state: Self::State<S>,
        parameters: &S,
        arguments: &UpdateArguments<'_, P, S>,
    ) -> Result<(S, Self::State<S>), OptimizerError> {
        let (updates, learning_rate) =
            self.learning_rate.update(gradients, state.learning_rate, parameters, arguments)?;
        let (updates, ()) = self.polyak.update(updates, (), parameters, arguments)?;
        Ok((updates, PolyakSgdState { learning_rate, leaf: PhantomData }))
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use ryft_core::Array;

    use crate::tests::{assert_parity, parity_quadratic_updates, parity_updates, state_paths, vector_updates};

    use super::*;

    #[test]
    fn test_sgd_new() {
        let optimizer = Sgd::new(0.1f64);
        assert_eq!(optimizer.momentum(), None);
        assert!(!optimizer.nesterov());
        assert_eq!(optimizer.learning_rate(), &0.1);
        let optimizer = optimizer
            .with_momentum(0.9)
            .unwrap()
            .with_nesterov(true)
            .with_accumulator_data_type(DataType::BF16)
            .unwrap();
        assert_eq!(
            optimizer.momentum(),
            Some(Trace::new(0.9).unwrap().with_nesterov(true).with_accumulator_data_type(DataType::BF16).unwrap()),
        );
        assert!(matches!(
            Sgd::new(0.1f64).with_momentum(f64::INFINITY),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`momentum` must be finite but got `inf`",
        ));
    }

    #[test]
    fn test_sgd_initialize() {
        assert_eq!(state_paths(&Sgd::new(0.1f64)), vec!["$.learning_rate.count"]);
        assert_eq!(
            state_paths(&Sgd::new(0.1f64).with_momentum(0.9).unwrap()),
            vec!["$.momentum[0].trace[\"bias\"]", "$.momentum[0].trace[\"weights\"]", "$.learning_rate.count"],
        );
    }

    #[test]
    fn test_sgd_update() {
        // Without momentum, the updates are `-0.5 · g`. With the momentum `0.5`, the traces of the gradients `2` and
        // `1` are `2` and `2`, and with Nesterov momentum the updates before the learning rate are `3` and `2`.
        let optimizer = Sgd::new(0.5f64);
        assert_eq!(vector_updates(&optimizer, &[0.0], &[&[2.0], &[1.0]]), vec![vec![-1.0], vec![-0.5]]);
        let optimizer = optimizer.with_momentum(0.5).unwrap();
        assert_eq!(vector_updates(&optimizer, &[0.0], &[&[2.0], &[1.0]]), vec![vec![-1.0], vec![-1.0]]);
        let optimizer = optimizer.with_nesterov(true);
        assert_eq!(vector_updates(&optimizer, &[0.0], &[&[2.0], &[1.0]]), vec![vec![-1.5], vec![-1.0]]);

        // A state without momentum is rejected by an optimizer with momentum.
        let parameters = vec![Array::scalar(1.0f32).unwrap()];
        let state = Sgd::new(0.5f64).initialize(&parameters).unwrap();
        assert!(matches!(
            optimizer.update(parameters.clone(), state, &parameters, &UpdateArguments::new()),
            Err(OptimizerError::StructureMismatch { message })
                if message == "the momentum state of `Sgd` does not match its momentum configuration",
        ));
    }

    #[test]
    fn test_noisy_sgd_new() {
        let optimizer = NoisySgd::new(0.1f64, AddNoise::new(0.01, 0.55, 3).unwrap());
        assert_eq!(optimizer.noise(), &AddNoise::new(0.01, 0.55, 3).unwrap());
        assert_eq!(optimizer.learning_rate(), &0.1);
    }

    #[test]
    fn test_noisy_sgd_update() {
        // Without noise, the updates are `-0.5 · g`, and the state holds the noise state and the learning rate count.
        let optimizer = NoisySgd::new(0.5f64, AddNoise::new(0.0, 0.55, 3).unwrap());
        assert_eq!(state_paths(&optimizer), vec!["$.noise.count", "$.noise.random_state", "$.learning_rate.count"],);
        assert_eq!(vector_updates(&optimizer, &[0.0], &[&[2.0], &[1.0]]), vec![vec![-1.0], vec![-0.5]]);
    }

    #[test]
    fn test_optimistic_gradient_descent_new() {
        let optimizer = OptimisticGradientDescent::new(0.1f64).with_alpha(0.5).unwrap().with_beta(2.0).unwrap();
        assert_eq!((optimizer.optimistic().alpha(), optimizer.optimistic().beta()), (0.5, 2.0));
        assert!(matches!(
            OptimisticGradientDescent::new(0.1f64).with_alpha(f64::NAN),
            Err(OptimizerError::InvalidConfiguration { message }) if message == "`alpha` must be finite but got `NaN`",
        ));
    }

    #[test]
    fn test_optimistic_gradient_descent_update() {
        // The optimistic steps `1` and `-1.5` of the transformation tests are scaled by `-0.5`.
        let optimizer = OptimisticGradientDescent::new(0.5f64).with_alpha(0.5).unwrap().with_beta(2.0).unwrap();
        assert_eq!(vector_updates(&optimizer, &[0.0], &[&[2.0], &[1.0]]), vec![vec![-0.5], vec![0.75]]);
    }

    #[test]
    fn test_polyak_sgd_new() {
        let optimizer = PolyakSgd::new(1.0f64)
            .with_maximum_learning_rate(0.5)
            .unwrap()
            .with_minimum_value(-1.0)
            .unwrap()
            .with_epsilon(1e-3)
            .unwrap()
            .with_variant(PolyakVariant::SpsPlus);
        assert_eq!(
            optimizer.polyak(),
            &ScaleByPolyak::new()
                .with_maximum_learning_rate(0.5)
                .unwrap()
                .with_minimum_value(-1.0)
                .unwrap()
                .with_epsilon(1e-3)
                .unwrap()
                .with_variant(PolyakVariant::SpsPlus),
        );
        assert_eq!(optimizer.scaling(), &1.0);
    }

    #[test]
    fn test_polyak_sgd_update() {
        // The scaled gradients `-[3, 4]` have the squared norm `25`, so the objective `5` gives the step size `0.2` at
        // both steps.
        let optimizer = PolyakSgd::new(1.0f64);
        let parameters = vec![Array::vector(vec![0.0f32, 0.0]).unwrap()];
        let gradients = vec![Array::vector(vec![3.0f32, 4.0]).unwrap()];
        let value = Array::scalar(5.0f32).unwrap();
        let arguments = UpdateArguments::new().with_value(&value);
        let state = optimizer.initialize(&parameters).unwrap();
        assert_eq!(
            state.named_parameters().map(|(path, _)| path.to_string()).collect::<Vec<_>>(),
            vec!["$.learning_rate.count"],
        );
        let expected = vec![Array::vector(vec![0.2f32 * -3.0, 0.2f32 * -4.0]).unwrap()];
        let (updates, state) = optimizer.update(gradients.clone(), state, &parameters, &arguments).unwrap();
        assert_eq!(updates, expected);
        let (updates, state) = optimizer.update(gradients, state, &parameters, &arguments).unwrap();
        assert_eq!(updates, expected);
        assert_eq!(state.learning_rate().count(), &Array::scalar(2i32).unwrap());
    }

    #[test]
    fn test_sgd_optax_parity() {
        // Optax 9ecde04b: `optax.sgd(0.1)`.
        const SGD: [[f32; 9]; 4] = [
            [
                -0.010000001,
                0.020000001,
                -0.030000001,
                -0.040000003,
                0.010000001,
                -0.020000001,
                0.030000001,
                -0.05,
                0.060000002,
            ],
            [
                0.0050000004,
                -0.025,
                -0.010000001,
                -0.020000001,
                -0.030000001,
                0.040000003,
                -0.010000001,
                0.020000001,
                -0.035,
            ],
            [
                -0.030000001,
                0.010000001,
                0.020000001,
                0.015000001,
                -0.0050000004,
                -0.044999998,
                -0.025,
                -0.040000003,
                0.010000001,
            ],
            [
                -0.020000001,
                -0.015000001,
                0.035,
                -0.010000001,
                0.030000001,
                -0.0050000004,
                0.044999998,
                -0.020000001,
                -0.030000001,
            ],
        ];
        // Optax 9ecde04b: `optax.sgd(0.1, momentum=0.9)`.
        const SGD_MOMENTUM: [[f32; 9]; 4] = [
            [
                -0.010000001,
                0.020000001,
                -0.030000001,
                -0.040000003,
                0.010000001,
                -0.020000001,
                0.030000001,
                -0.05,
                0.060000002,
            ],
            [
                -0.0039999997,
                -0.0070000007,
                -0.037,
                -0.056,
                -0.021000002,
                0.022000002,
                0.017000003,
                -0.024999999,
                0.019000003,
            ],
            [-0.0336, 0.0036999993, -0.013300001, -0.0354, -0.0239, -0.025199998, -0.009699999, -0.0625, 0.027100002],
            [-0.05024, -0.011670001, 0.02303, -0.04186, 0.008490003, -0.027679998, 0.03627, -0.07625, -0.00561],
        ];
        // Optax 9ecde04b: `optax.sgd(0.1, momentum=0.9, nesterov=True)`.
        const SGD_NESTEROV: [[f32; 9]; 4] = [
            [-0.019, 0.038, -0.057000007, -0.076, 0.019, -0.038, 0.057000007, -0.095, 0.114000015],
            [
                0.0014000006,
                -0.031300005,
                -0.0433,
                -0.0704,
                -0.048900004,
                0.059800006,
                0.0053000012,
                -0.0024999976,
                -0.017899998,
            ],
            [-0.06024, 0.0133299995, 0.00803, -0.016859999, -0.02651, -0.067679994, -0.03373, -0.09625, 0.034390002],
            [
                -0.065216,
                -0.025503,
                0.055727,
                -0.047673997,
                0.037641004,
                -0.029911999,
                0.077643,
                -0.088625,
                -0.035049003,
            ],
        ];
        assert_parity(&parity_updates(&Sgd::new(0.1)), &SGD);
        assert_parity(&parity_updates(&Sgd::new(0.1).with_momentum(0.9).unwrap()), &SGD_MOMENTUM);
        assert_parity(&parity_updates(&Sgd::new(0.1).with_momentum(0.9).unwrap().with_nesterov(true)), &SGD_NESTEROV);
    }

    #[test]
    fn test_noisy_sgd_optax_parity() {
        // Optax 9ecde04b: `optax.noisy_sgd(0.1, eta=0.0, seed=3)`.
        const NOISY_SGD: [[f32; 9]; 4] = [
            [
                -0.010000001,
                0.020000001,
                -0.030000001,
                -0.040000003,
                0.010000001,
                -0.020000001,
                0.030000001,
                -0.05,
                0.060000002,
            ],
            [
                0.0050000004,
                -0.025,
                -0.010000001,
                -0.020000001,
                -0.030000001,
                0.040000003,
                -0.010000001,
                0.020000001,
                -0.035,
            ],
            [
                -0.030000001,
                0.010000001,
                0.020000001,
                0.015000001,
                -0.0050000004,
                -0.044999998,
                -0.025,
                -0.040000003,
                0.010000001,
            ],
            [
                -0.020000001,
                -0.015000001,
                0.035,
                -0.010000001,
                0.030000001,
                -0.0050000004,
                0.044999998,
                -0.020000001,
                -0.030000001,
            ],
        ];
        assert_parity(&parity_updates(&NoisySgd::new(0.1, AddNoise::new(0.0, 0.55, 3).unwrap())), &NOISY_SGD);
    }

    #[test]
    fn test_optimistic_gradient_descent_optax_parity() {
        // Optax 9ecde04b: `optax.optimistic_gradient_descent(0.1)`.
        const OPTIMISTIC_GRADIENT_DESCENT: [[f32; 9]; 4] = [
            [
                -0.010000001,
                0.020000001,
                -0.030000001,
                -0.040000003,
                0.010000001,
                -0.020000001,
                0.030000001,
                -0.05,
                0.060000002,
            ],
            [0.020000001, -0.07, 0.010000001, -0.0, -0.07000001, 0.1, -0.05, 0.089999996, -0.13],
            [-0.065000005, 0.044999998, 0.05, 0.05, 0.020000001, -0.13, -0.040000003, -0.1, 0.055000003],
            [-0.01, -0.040000003, 0.05, -0.035000004, 0.065000005, 0.035, 0.115, -0.0, -0.07000001],
        ];
        // Optax 9ecde04b: `optax.optimistic_gradient_descent(0.1, alpha=0.5, beta=0.25)`.
        const OPTIMISTIC_GRADIENT_DESCENT_VARIANT: [[f32; 9]; 4] = [
            [
                -0.0050000004,
                0.010000001,
                -0.015000001,
                -0.020000001,
                0.0050000004,
                -0.010000001,
                0.015000001,
                -0.025,
                0.030000001,
            ],
            [0.00625, -0.02375, -0.0, -0.0050000004, -0.025, 0.035000004, -0.015000001, 0.027500002, -0.04125],
            [
                -0.023750002,
                0.013750001,
                0.017500002,
                0.016250001,
                0.0037500001,
                -0.04375,
                -0.01625,
                -0.035000004,
                0.01625,
            ],
            [-0.0075000003, -0.013750001, 0.02125, -0.01125, 0.023750002, 0.0075, 0.04, -0.0050000004, -0.025],
        ];
        assert_parity(&parity_updates(&OptimisticGradientDescent::new(0.1)), &OPTIMISTIC_GRADIENT_DESCENT);
        assert_parity(
            &parity_updates(&OptimisticGradientDescent::new(0.1).with_alpha(0.5).unwrap().with_beta(0.25).unwrap()),
            &OPTIMISTIC_GRADIENT_DESCENT_VARIANT,
        );
    }

    #[test]
    fn test_polyak_sgd_optax_parity() {
        // Optax 9ecde04b: `optax.polyak_sgd()` on the parity quadratic.
        const POLYAK_SGD: [[f32; 9]; 4] = [
            [0.12391964, 0.99135715, -0.7435179, -0.0, 0.37175894, 0.2788192, 1.4870358, -0.3097991, 0.18587947],
            [0.10181439, 0.5461299, -0.20830844, -0.0, 0.30544317, 0.1913407, 0.8191948, -0.12873042, 0.05207711],
            [0.08315254, 0.28045136, -0.043816186, -0.0, 0.24945764, 0.1272638, 0.42067704, -0.046591904, 0.010954047],
            [0.06926876, 0.1319759, -0.004738061, -0.0, 0.2078063, 0.08295159, 0.19796385, -0.013481842, 0.0011845477],
        ];
        // Optax 9ecde04b: `optax.polyak_sgd(max_learning_rate=0.5, scaling=0.5, f_min=2.0, eps=1e-3, variant='sps+')`
        // on the parity quadratic.
        const POLYAK_SGD_PLUS: [[f32; 9]; 4] = [
            [0.125, 1.0, -0.75, -0.0, 0.375, 0.28125, 1.5, -0.3125, 0.1875],
            [0.09375, 0.5, -0.1875, -0.0, 0.28125, 0.17578125, 0.75, -0.1171875, 0.046875],
            [0.0, 0.0, -0.0, -0.0, 0.0, 0.0, 0.0, -0.0, 0.0],
            [0.0, 0.0, -0.0, -0.0, 0.0, 0.0, 0.0, -0.0, 0.0],
        ];
        assert_parity(&parity_quadratic_updates(&PolyakSgd::new(1.0), 4), &POLYAK_SGD);
        assert_parity(
            &parity_quadratic_updates(
                &PolyakSgd::new(0.5)
                    .with_maximum_learning_rate(0.5)
                    .unwrap()
                    .with_minimum_value(2.0)
                    .unwrap()
                    .with_epsilon(1e-3)
                    .unwrap()
                    .with_variant(PolyakVariant::SpsPlus),
                4,
            ),
            &POLYAK_SGD_PLUS,
        );
    }
}
