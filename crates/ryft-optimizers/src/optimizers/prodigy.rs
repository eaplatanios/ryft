//! Prodigy optimizer of [Mishchenko and Defazio (2024)](https://arxiv.org/abs/2306.06101), which is Adam with a
//! learning rate that is estimated from the distance between the parameters and their initial values.

use ryft_core::{Add, DataType, Div, Mul, Neg, Parameter, ParameterPath, Parameterized, ReductionKind, Sub};

use crate::macros::parameterized_state;
use crate::{
    AddDecayedWeights, Optimizer, OptimizerError, OptimizerValue, Schedule, UpdateArguments, all_axes, bias_correction,
    convert_to, filled_statistics, increment_count, initial_count, inner_product, statistic_data_type,
    validate_decay_rate, validate_finite, validate_non_negative, validate_positive, validate_structure,
};

// TODO(eaplatanios): Review this module.

parameterized_state! {
    /// State of [`Prodigy`], which is the analogue of [`optax.contrib.ProdigyState`](
    /// https://optax.readthedocs.io/en/latest/api/contrib.html#optax.contrib.ProdigyState). It holds the moment
    /// estimates, the decaying sum of the scaled gradients, and the initial parameters with the structure `S` of the
    /// parameters, and the learning rate estimate, the weighted numerator of the estimate, and the step count. Its
    /// rank-0 floating-point arrays have the statistic data type of the first parameter.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct ProdigyState<P, S> with family ProdigyStateFamily {
        /// Exponential moving averages of the scaled gradients, which are the `exp_avg` of Optax.
        first_moments: S,

        /// Exponential moving averages of the squared scaled gradients, which are the `exp_avg_sq` of Optax.
        second_moments: S,

        /// Decaying sums of the scaled gradients, which are the `grad_sum` of Optax.
        gradient_sum: S,

        /// Parameters at initialization, which are the `params0` of Optax.
        initial_parameters: S,

        /// Rank-0 array that holds the estimate of the learning rate, which is the `estim_lr` of Optax.
        learning_rate_estimate: P,

        /// Rank-0 array that holds the weighted numerator of the estimate, which is the `numerator_weighted` of Optax.
        weighted_numerator: P,

        /// Rank-0 `i32` array that holds the number of updates applied so far.
        count: P,
    }
}

impl<P: Parameter, S> ProdigyState<P, S> {
    /// Returns the exponential moving averages of the scaled gradients.
    #[inline]
    pub fn first_moments(&self) -> &S {
        &self.first_moments
    }

    /// Returns the exponential moving averages of the squared scaled gradients.
    #[inline]
    pub fn second_moments(&self) -> &S {
        &self.second_moments
    }

    /// Returns the decaying sums of the scaled gradients.
    #[inline]
    pub fn gradient_sum(&self) -> &S {
        &self.gradient_sum
    }

    /// Returns the parameters at initialization.
    #[inline]
    pub fn initial_parameters(&self) -> &S {
        &self.initial_parameters
    }

    /// Returns the rank-0 array that holds the estimate of the learning rate.
    #[inline]
    pub fn learning_rate_estimate(&self) -> &P {
        &self.learning_rate_estimate
    }

    /// Returns the rank-0 array that holds the weighted numerator of the estimate.
    #[inline]
    pub fn weighted_numerator(&self) -> &P {
        &self.weighted_numerator
    }

    /// Returns the rank-0 `i32` array that holds the number of updates applied so far.
    #[inline]
    pub fn count(&self) -> &P {
        &self.count
    }
}

/// Prodigy optimizer of [Mishchenko and Defazio (2023)](https://arxiv.org/abs/2306.06101), which is Adam with a step
/// size that is scaled by an estimate of the distance from the initial parameters to the solution, and which is the
/// analogue of
/// [`optax.contrib.prodigy`](https://optax.readthedocs.io/en/latest/api/contrib.html#optax.contrib.prodigy). Its update
/// is
///
/// ```text
/// γₜ = dₜ₋₁ · ηₜ · √(1 − β₂ᵗ) / (1 − β₁ᵗ)
/// mₜ = β₁ · mₜ₋₁ + (1 − β₁) · dₜ₋₁ · gₜ
/// vₜ = β₂ · vₜ₋₁ + (1 − β₂) · dₜ₋₁² · gₜ²
/// sₜ = β₃ · sₜ₋₁ + (γₜ / d₀) · dₜ₋₁ · gₜ
/// ρₜ = β₃ · ρₜ₋₁ + (dₜ₋₁ / d₀) · γₜ · ⟨gₜ, θ₀ − θₜ₋₁⟩
/// dₜ = max(dₜ₋₁, κ · ρₜ / ‖sₜ‖₁)
/// uₜ = −γₜ · (mₜ / (√vₜ + dₜ · ε) + λ · θₜ₋₁)
/// ```
///
/// Here `t ≥ 1` counts the updates including the current one, `gₜ` is the gradient at the parameters `θₜ₋₁`, `uₜ` is
/// the update that is added to them (i.e., `θₜ = θₜ₋₁ + uₜ`), and `ηₜ` is the learning rate [`Schedule`] evaluated at
/// the `t − 1` updates applied before the current one. The distance estimate `dₜ` starts at `d₀` and never decreases,
/// `γₜ` is the step size with the bias correction of Adam, the moments `mₜ` and `vₜ` of the scaled gradients
/// `dₜ₋₁ · gₜ` start at zero and decay with the rates `β₁` and `β₂`, the sums `sₜ` and `ρₜ` start at zero and decay
/// with the rate `β₃`, `θ₀` are the initial parameters, `⟨·, ·⟩` is the inner product over all leaves, `‖sₜ‖₁` is the
/// sum of the absolute values of `sₜ` over all leaves, `κ` is the coefficient of the estimate, `ε` is scaled by the new
/// estimate `dₜ`, and `λ` is the rate of the decoupled weight decay. The other operations are element-wise. The
/// defaults are `ηₜ = 1` for the learning rate `1.0`, `β₁ = 0.9`, `β₂ = 0.999`, `β₃ = √β₂`, `ε = 1e-8`, `d₀ = 1e-6`,
/// `κ = 1`, and `λ = 0`, without the safeguarded warmup. [`Self::with_safeguard_warmup`] replaces `γₜ` with `dₜ₋₁` in
/// the update of `sₜ`, which avoids overestimates of `dₜ` during a warmup of the learning rate, and
/// [`Self::with_weight_decay_mask`] restricts the weight decay to the leaves that the mask selects.
///
/// It is a single transformation rather than a chain, because the estimate couples all leaves, and it applies its
/// weight decay with [`AddDecayedWeights`].
///
/// # References
///
///   - Mishchenko and Defazio (2023). [Prodigy: An Expeditiously Adaptive Parameter-Free
///     Learner](https://arxiv.org/abs/2306.06101).
#[derive(Clone, Debug)]
pub struct Prodigy<Sc: Schedule = f64> {
    /// Learning rate schedule `η`.
    learning_rate: Sc,

    /// Decay `β₁` of the first moments.
    beta1: f64,

    /// Decay `β₂` of the second moments.
    beta2: f64,

    /// Decay `β₃` of the sums of the estimate, or [`None`] for `√β₂`.
    beta3: Option<f64>,

    /// Term `ε` that is added to the root of the second moments, scaled by the estimate.
    epsilon: f64,

    /// Initial estimate `d₀` of the learning rate, which is the `estim_lr0` of Optax.
    initial_learning_rate_estimate: f64,

    /// Coefficient `c` of the estimate, which is the `estim_lr_coef` of Optax.
    learning_rate_estimate_coefficient: f64,

    /// Weight decay that is added to the rescaled gradients.
    weight_decay: AddDecayedWeights,

    /// Whether the sums of the scaled gradients use the estimate rather than the step size, which avoids overestimates
    /// during a warmup of the learning rate.
    safeguard_warmup: bool,
}

impl<Sc: Schedule> Prodigy<Sc> {
    /// Creates a new [`Prodigy`] with the learning rate schedule `learning_rate` (e.g., `1.0`, which is the default of
    /// Optax) and default values for all other hyperparameters.
    pub fn new(learning_rate: Sc) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            beta3: None,
            epsilon: 1e-8,
            initial_learning_rate_estimate: 1e-6,
            learning_rate_estimate_coefficient: 1.0,
            weight_decay: AddDecayedWeights::new(0.0).unwrap(),
            safeguard_warmup: false,
        }
    }

    /// Returns this optimizer with `beta1` as the decay `β₁` of its first moments.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `beta1` is not in `[0, 1)`.
    pub fn with_beta1(mut self, beta1: f64) -> Result<Self, OptimizerError> {
        validate_decay_rate("beta1", beta1)?;
        self.beta1 = beta1;
        Ok(self)
    }

    /// Returns this optimizer with `beta2` as the decay `β₂` of its second moments.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `beta2` is not in `[0, 1)`.
    pub fn with_beta2(mut self, beta2: f64) -> Result<Self, OptimizerError> {
        validate_decay_rate("beta2", beta2)?;
        self.beta2 = beta2;
        Ok(self)
    }

    /// Returns this optimizer with `beta3` as the decay `β₃` of the sums of its estimate, or with `√β₂` when it is
    /// [`None`].
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `beta3` is not finite.
    pub fn with_beta3(mut self, beta3: Option<f64>) -> Result<Self, OptimizerError> {
        if let Some(beta3) = beta3 {
            validate_finite("beta3", beta3)?;
        }
        self.beta3 = beta3;
        Ok(self)
    }

    /// Returns this optimizer with `epsilon` as the term `ε` that it adds to the root of the second moments.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `epsilon` is negative or not finite.
    pub fn with_epsilon(mut self, epsilon: f64) -> Result<Self, OptimizerError> {
        validate_non_negative("epsilon", epsilon)?;
        self.epsilon = epsilon;
        Ok(self)
    }

    /// Returns this optimizer with `initial_learning_rate_estimate` as its initial estimate `d₀`.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `initial_learning_rate_estimate` is not positive and
    /// finite.
    pub fn with_initial_learning_rate_estimate(
        mut self,
        initial_learning_rate_estimate: f64,
    ) -> Result<Self, OptimizerError> {
        validate_positive("initial_learning_rate_estimate", initial_learning_rate_estimate)?;
        self.initial_learning_rate_estimate = initial_learning_rate_estimate;
        Ok(self)
    }

    /// Returns this optimizer with `learning_rate_estimate_coefficient` as the coefficient `c` of its estimate.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `learning_rate_estimate_coefficient` is not finite.
    pub fn with_learning_rate_estimate_coefficient(
        mut self,
        learning_rate_estimate_coefficient: f64,
    ) -> Result<Self, OptimizerError> {
        validate_finite("learning_rate_estimate_coefficient", learning_rate_estimate_coefficient)?;
        self.learning_rate_estimate_coefficient = learning_rate_estimate_coefficient;
        Ok(self)
    }

    /// Returns this optimizer with `weight_decay` as the rate `λ` of its weight decay, keeping its mask.
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

    /// Returns this optimizer with the safeguarded warmup when `safeguard_warmup` is `true`.
    #[inline]
    pub fn with_safeguard_warmup(mut self, safeguard_warmup: bool) -> Self {
        self.safeguard_warmup = safeguard_warmup;
        self
    }

    /// Returns the learning rate [`Schedule`] of this optimizer.
    #[inline]
    pub fn learning_rate(&self) -> &Sc {
        &self.learning_rate
    }

    /// Returns the decay `β₁` of the first moments of this optimizer.
    #[inline]
    pub fn beta1(&self) -> f64 {
        self.beta1
    }

    /// Returns the decay `β₂` of the second moments of this optimizer.
    #[inline]
    pub fn beta2(&self) -> f64 {
        self.beta2
    }

    /// Returns the decay `β₃` of the sums of the estimate of this optimizer.
    #[inline]
    pub fn beta3(&self) -> f64 {
        self.beta3.unwrap_or_else(|| self.beta2.sqrt())
    }

    /// Returns the term `ε` that this optimizer adds to the root of the second moments.
    #[inline]
    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    /// Returns the initial estimate `d₀` of the learning rate of this optimizer.
    #[inline]
    pub fn initial_learning_rate_estimate(&self) -> f64 {
        self.initial_learning_rate_estimate
    }

    /// Returns the coefficient `c` of the estimate of this optimizer.
    #[inline]
    pub fn learning_rate_estimate_coefficient(&self) -> f64 {
        self.learning_rate_estimate_coefficient
    }

    /// Returns the [`AddDecayedWeights`] transformation of this optimizer.
    #[inline]
    pub fn weight_decay(&self) -> &AddDecayedWeights {
        &self.weight_decay
    }

    /// Returns whether this optimizer uses the safeguarded warmup.
    #[inline]
    pub fn safeguard_warmup(&self) -> bool {
        self.safeguard_warmup
    }
}

impl<P: OptimizerValue, Sc: Schedule> Optimizer<P> for Prodigy<Sc> {
    type State<S: Parameterized<P>> = ProdigyState<P, S>;

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        let count = initial_count("Prodigy", parameters)?;
        let data_type = statistic_data_type(parameters.parameters().next().unwrap().r#type().data_type());
        let scalar = convert_to(&count, data_type)?;
        let initial_parameters = parameters
            .parameters()
            .map(|parameter| convert_to(parameter, statistic_data_type(parameter.r#type().data_type())))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(ProdigyState {
            first_moments: filled_statistics(parameters, 0.0, None)?,
            second_moments: filled_statistics(parameters, 0.0, None)?,
            gradient_sum: filled_statistics(parameters, 0.0, None)?,
            initial_parameters: S::from_parameters(parameters.parameter_structure(), initial_parameters)?,
            learning_rate_estimate: scalar.fill_like(self.initial_learning_rate_estimate)?,
            weighted_numerator: scalar.fill_like(0f64)?,
            count,
        })
    }

    fn update<S: Parameterized<P>>(
        &self,
        gradients: S,
        state: Self::State<S>,
        parameters: &S,
        arguments: &UpdateArguments<'_, P, S>,
    ) -> Result<(S, Self::State<S>), OptimizerError> {
        // This follows the operation order of `optax.contrib.prodigy`.
        let paths = gradients.parameter_paths().collect::<Vec<_>>();
        validate_structure("parameters", &paths, parameters)?;
        validate_structure("first_moments", &paths, &state.first_moments)?;
        validate_structure("second_moments", &paths, &state.second_moments)?;
        validate_structure("gradient_sum", &paths, &state.gradient_sum)?;
        validate_structure("initial_parameters", &paths, &state.initial_parameters)?;
        let ProdigyState {
            first_moments,
            second_moments,
            gradient_sum,
            initial_parameters,
            learning_rate_estimate: estimate,
            weighted_numerator,
            count,
        } = state;
        let data_type = estimate.r#type().data_type();
        let next_count = increment_count(&count)?;
        let step = convert_to(&next_count, DataType::F32)?;
        let learning_rate = convert_to(&self.learning_rate.value(&count)?, data_type)?;
        let correction = Div::div(
            &convert_to(&bias_correction(&step, self.beta2)?, data_type)?.sqrt()?,
            &convert_to(&bias_correction(&step, self.beta1)?, data_type)?,
        )?;
        let step_size = Mul::mul(&Mul::mul(&estimate, &learning_rate)?, &correction)?;
        let beta3 = self.beta3();
        let initial_estimate = estimate.fill_like(self.initial_learning_rate_estimate)?;

        let gradient_leaves = gradients
            .parameters()
            .map(|gradient| convert_to(gradient, statistic_data_type(gradient.r#type().data_type())))
            .collect::<Result<Vec<_>, _>>()?;
        let differences = initial_parameters
            .parameters()
            .zip(parameters.parameters())
            .zip(&gradient_leaves)
            .map(|((initial, parameter), gradient)| {
                let data_type = gradient.r#type().data_type();
                Ok(Sub::sub(&convert_to(initial, data_type)?, &convert_to(parameter, data_type)?)?)
            })
            .collect::<Result<Vec<_>, OptimizerError>>()?;
        let accumulated_numerator =
            convert_to(&inner_product("numerator", &gradient_leaves, &differences)?, data_type)?;
        let sum_scale = if self.safeguard_warmup { estimate.clone() } else { step_size.clone() };
        let structure = gradients.parameter_structure();
        let mut next_first_moments = Vec::with_capacity(paths.len());
        let mut next_second_moments = Vec::with_capacity(paths.len());
        let mut next_gradient_sum = Vec::with_capacity(paths.len());
        let mut stored_first_moments = Vec::with_capacity(paths.len());
        let mut stored_second_moments = Vec::with_capacity(paths.len());
        let mut denominator: Option<P> = None;
        for (((gradient, first_moment), second_moment), sum) in gradient_leaves
            .iter()
            .zip(first_moments.into_parameters())
            .zip(second_moments.into_parameters())
            .zip(gradient_sum.into_parameters())
        {
            let leaf_type = gradient.r#type().data_type();
            let first_moment_type = first_moment.r#type().data_type();
            let second_moment_type = second_moment.r#type().data_type();
            let sum_type = sum.r#type().data_type();
            let scaled = Mul::mul(&convert_to(&estimate, leaf_type)?, gradient)?;
            let first_moment = Add::add(
                &Mul::mul(&scaled.fill_like(self.beta1)?, &convert_to(&first_moment, leaf_type)?)?,
                &Mul::mul(&scaled.fill_like(1.0 - self.beta1)?, &scaled)?,
            )?;
            let second_moment = Add::add(
                &Mul::mul(&scaled.fill_like(self.beta2)?, &convert_to(&second_moment, leaf_type)?)?,
                &Mul::mul(&Mul::mul(&scaled.fill_like(1.0 - self.beta2)?, &scaled)?, &scaled)?,
            )?;
            let sum = Add::add(
                &Mul::mul(&scaled.fill_like(beta3)?, &convert_to(&sum, leaf_type)?)?,
                &Div::div(
                    &Mul::mul(&convert_to(&sum_scale, leaf_type)?, &scaled)?,
                    &convert_to(&initial_estimate, leaf_type)?,
                )?,
            )?;
            let leaf_denominator = convert_to(&sum.abs()?.reduce(&all_axes(&sum), ReductionKind::Sum)?, data_type)?;
            denominator = Some(match denominator {
                None => leaf_denominator,
                Some(denominator) => Add::add(&denominator, &leaf_denominator)?,
            });
            // The state keeps the data types of the statistics that it replaces, as `initialize` created them from the
            // parameters, while the update is computed from the statistics in the data types of the gradients.
            stored_first_moments.push(convert_to(&first_moment, first_moment_type)?);
            stored_second_moments.push(convert_to(&second_moment, second_moment_type)?);
            next_first_moments.push(first_moment);
            next_second_moments.push(second_moment);
            next_gradient_sum.push(convert_to(&sum, sum_type)?);
        }
        let weighted_numerator = Add::add(
            &Mul::mul(&weighted_numerator.fill_like(beta3)?, &weighted_numerator)?,
            &Mul::mul(&Mul::mul(&Div::div(&estimate, &initial_estimate)?, &step_size)?, &accumulated_numerator)?,
        )?;
        let estimated = Div::div(
            &Mul::mul(&weighted_numerator.fill_like(self.learning_rate_estimate_coefficient)?, &weighted_numerator)?,
            &denominator.unwrap(),
        )?;
        let estimate = estimate.max(&estimated)?;

        let ascent = next_first_moments
            .iter()
            .zip(&next_second_moments)
            .map(|(first_moment, second_moment)| {
                let leaf_type = first_moment.r#type().data_type();
                let epsilon = Mul::mul(&convert_to(&estimate, leaf_type)?, &first_moment.fill_like(self.epsilon)?)?;
                Ok(Div::div(first_moment, &Add::add(&second_moment.sqrt()?, &epsilon)?)?)
            })
            .collect::<Result<Vec<_>, OptimizerError>>()?;
        let (ascent, ()) =
            self.weight_decay
                .update(S::from_parameters(structure.clone(), ascent)?, (), parameters, arguments)?;
        let factor = Neg::neg(&step_size)?;
        let updates = ascent
            .into_parameters()
            .zip(gradients.parameters())
            .map(|(ascent, gradient)| {
                let leaf_type = ascent.r#type().data_type();
                Ok(convert_to(&Mul::mul(&convert_to(&factor, leaf_type)?, &ascent)?, gradient.r#type().data_type())?)
            })
            .collect::<Result<Vec<_>, OptimizerError>>()?;
        let state = ProdigyState {
            first_moments: S::from_parameters(structure.clone(), stored_first_moments)?,
            second_moments: S::from_parameters(structure.clone(), stored_second_moments)?,
            gradient_sum: S::from_parameters(structure.clone(), next_gradient_sum)?,
            initial_parameters,
            learning_rate_estimate: estimate,
            weighted_numerator,
            count: next_count,
        };
        Ok((S::from_parameters(structure, updates)?, state))
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use ryft_core::Array;

    use crate::apply_updates;
    use crate::tests::{
        PARITY_PARAMETERS, assert_parity, parity_elements, parity_parameters, parity_quadratic, state_data_types,
        state_paths,
    };

    use super::*;

    /// Returns the single `f32` element of the rank-0 or single element `value`.
    fn element(value: &Array) -> f32 {
        value.elements::<f32>().unwrap()[0]
    }

    #[test]
    fn test_prodigy_new() {
        let optimizer = Prodigy::new(1.0);
        assert_eq!(optimizer.learning_rate(), &1.0);
        assert_eq!((optimizer.beta1(), optimizer.beta2(), optimizer.beta3()), (0.9, 0.999, 0.999f64.sqrt()));
        assert_eq!(optimizer.epsilon(), 1e-8);
        assert_eq!(optimizer.initial_learning_rate_estimate(), 1e-6);
        assert_eq!(optimizer.learning_rate_estimate_coefficient(), 1.0);
        assert_eq!(optimizer.weight_decay().rate(), 0.0);
        assert!(!optimizer.safeguard_warmup());
    }

    #[test]
    fn test_prodigy_with_hyperparameters() {
        let optimizer = Prodigy::new(1.0)
            .with_beta1(0.8)
            .unwrap()
            .with_beta2(0.99)
            .unwrap()
            .with_beta3(Some(0.5))
            .unwrap()
            .with_epsilon(1e-6)
            .unwrap()
            .with_initial_learning_rate_estimate(1e-3)
            .unwrap()
            .with_learning_rate_estimate_coefficient(2.0)
            .unwrap()
            .with_weight_decay(0.1)
            .unwrap()
            .with_weight_decay_mask(|_| false)
            .with_safeguard_warmup(true);
        assert_eq!((optimizer.beta1(), optimizer.beta2(), optimizer.beta3()), (0.8, 0.99, 0.5));
        assert_eq!(optimizer.epsilon(), 1e-6);
        assert_eq!(optimizer.initial_learning_rate_estimate(), 1e-3);
        assert_eq!(optimizer.learning_rate_estimate_coefficient(), 2.0);
        assert_eq!(optimizer.weight_decay().rate(), 0.1);
        assert!(!optimizer.weight_decay().decays(&ParameterPath::root()));
        assert!(optimizer.safeguard_warmup());
        assert!(matches!(
            Prodigy::new(1.0).with_beta1(1.0),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`beta1` must be in `[0, 1)` but got `1`",
        ));
        assert!(matches!(
            Prodigy::new(1.0).with_initial_learning_rate_estimate(0.0),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`initial_learning_rate_estimate` must be positive and finite but got `0`",
        ));
    }

    #[test]
    fn test_prodigy_initialize() {
        assert_eq!(
            state_paths(&Prodigy::new(1.0)),
            vec![
                "$.first_moments[\"bias\"]",
                "$.first_moments[\"weights\"]",
                "$.second_moments[\"bias\"]",
                "$.second_moments[\"weights\"]",
                "$.gradient_sum[\"bias\"]",
                "$.gradient_sum[\"weights\"]",
                "$.initial_parameters[\"bias\"]",
                "$.initial_parameters[\"weights\"]",
                "$.learning_rate_estimate",
                "$.weighted_numerator",
                "$.count",
            ],
        );
    }

    #[test]
    fn test_prodigy_update() {
        // From `θ₀ = 0` with the gradient `1`, `η = 1`, and `d = 1e-6`, the first step has `b = √0.001 / 0.1` and the
        // step size `dη = 3.16228e-7`, the moments `m = 1e-7` and `v = 1e-15`, the sum `s = dη`, and the numerator `0`
        // (because `θ = θ₀`), so `d` stays `1e-6` and the update is `-dη · m / (√v + 1e-14) = -1e-6`.
        let optimizer = Prodigy::new(1.0);
        let arguments = UpdateArguments::new();
        let initial = vec![Array::scalar(0.0f32).unwrap()];
        let gradients = vec![Array::scalar(1.0f32).unwrap()];
        let state = optimizer.initialize(&initial).unwrap();
        let (updates, state) = optimizer.update(gradients.clone(), state, &initial, &arguments).unwrap();
        assert!((element(&updates[0]) + 1e-6).abs() < 1e-10);
        assert_eq!(state.learning_rate_estimate(), &Array::scalar(1e-6f32).unwrap());
        assert!((element(&state.gradient_sum()[0]) - 3.16228e-7).abs() < 1e-11);

        // At `θ = -10`, the second step has `b = √0.001999 / 0.19` and `dη = 2.35317e-7`, the numerator
        // `r = dη · 1 · 10 = 2.35317e-6` and the sum `Σ|s| = 0.9995 · 3.16228e-7 + 2.35317e-7 = 5.51387e-7`, so the
        // estimate grows to `d = 4.26774`. With `m = 1.9e-7` and `v = 1.999e-15`, the update is
        // `-dη · m / (√v + 4.26774e-8) = -5.1163e-7`.
        let parameters = vec![Array::scalar(-10.0f32).unwrap()];
        let (updates, state) = optimizer.update(gradients, state, &parameters, &arguments).unwrap();
        assert!((element(state.learning_rate_estimate()) - 4.26774).abs() < 1e-3);
        assert!((element(&updates[0]) + 5.1163e-7).abs() < 1e-10);
        assert_eq!(state.count(), &Array::scalar(2i32).unwrap());
        assert_eq!(state.initial_parameters(), &initial);

        // `f64` parameters with `f32` gradients keep the `f64` statistics that `initialize` creates.
        let (initial_data_types, updated_data_types) = state_data_types(&optimizer);
        assert_eq!(updated_data_types, initial_data_types);
    }

    /// Returns the updates of `steps` steps of `optimizer` on the quadratic of
    /// [`parity_quadratic`](crate::tests::parity_quadratic) and the learning rate estimate after every step.
    fn prodigy_parity_updates<Sc: Schedule>(optimizer: &Prodigy<Sc>) -> (Vec<Vec<f32>>, Vec<f32>) {
        let mut parameters = parity_parameters(&PARITY_PARAMETERS);
        let mut state = optimizer.initialize(&parameters).unwrap();
        let (mut updates, mut estimates) = (Vec::new(), Vec::new());
        for _ in 0..4 {
            let (_, gradients) = parity_quadratic(&parameters).unwrap();
            let (step_updates, next_state) =
                optimizer.update(gradients, state, &parameters, &UpdateArguments::new()).unwrap();
            updates.push(parity_elements(&step_updates));
            estimates.push(next_state.learning_rate_estimate().elements::<f32>().unwrap()[0]);
            parameters = apply_updates(parameters, step_updates).unwrap();
            state = next_state;
        }
        (updates, estimates)
    }

    #[test]
    fn test_prodigy_optax_parity() {
        // Optax 9ecde04b: `optax.contrib.prodigy(1.0, estim_lr0=0.1)` on the gradients of the parity quadratic, which
        // stay correlated with the distance travelled, so the learning rate estimate grows at the fourth step (it does
        // not grow over the four parity gradients).
        const DEFAULT: [[f32; 9]; 4] = [
            [0.09999927, 0.09999933, -0.09999932, -0.0, 0.099999316, 0.09999931, 0.09999932, -0.09999931, 0.09999929],
            [0.09881156, 0.099832505, -0.09958677, -0.0, 0.099759914, 0.09937341, 0.09989628, -0.09881158, 0.09574808],
            [0.09631489, 0.099541746, -0.098824635, -0.0, 0.099334836, 0.09816761, 0.09971983, -0.0963149, 0.08445845],
            [0.09195509, 0.09911754, -0.097646296, -0.0, 0.098702975, 0.09622519, 0.09946704, -0.09195511, 0.062929966],
        ];
        const DEFAULT_ESTIMATES: [f32; 4] = [0.1, 0.1, 0.1, 0.11452645];
        let (updates, estimates) =
            prodigy_parity_updates(&Prodigy::new(1.0).with_initial_learning_rate_estimate(0.1).unwrap());
        assert_parity(&updates, &DEFAULT);
        assert_parity(&[estimates], &[DEFAULT_ESTIMATES]);

        // Optax 9ecde04b: `optax.contrib.prodigy(0.5, betas=(0.8, 0.99), beta3=0.9, estim_lr0=0.1, estim_lr_coef=2.0,
        // weight_decay=0.01, safeguard_warmup=True)` on the gradients of the parity quadratic.
        const VARIANT: [[f32; 9]; 4] = [
            [
                0.049874973,
                0.050249983,
                -0.05049998,
                -0.0002499999,
                0.050124977,
                0.049937475,
                0.05049998,
                -0.050374974,
                0.04981247,
            ],
            [
                0.04954531,
                0.050114516,
                -0.05022778,
                0.038958147,
                0.04999076,
                0.049728967,
                0.0503351,
                -0.04993269,
                0.048955962,
            ],
            [
                0.048917037,
                0.049966536,
                -0.049915228,
                -0.035315543,
                0.04980762,
                0.049353316,
                0.050210066,
                -0.049258377,
                0.047162503,
            ],
            [
                0.047959108,
                0.049779978,
                -0.049506925,
                -0.029907929,
                0.04956027,
                0.04880146,
                0.050075583,
                -0.048272505,
                0.044144697,
            ],
        ];
        const VARIANT_ESTIMATES: [f32; 4] = [0.1, 0.1, 0.1, 0.1];
        let optimizer = Prodigy::new(0.5)
            .with_beta1(0.8)
            .unwrap()
            .with_beta2(0.99)
            .unwrap()
            .with_beta3(Some(0.9))
            .unwrap()
            .with_initial_learning_rate_estimate(0.1)
            .unwrap()
            .with_learning_rate_estimate_coefficient(2.0)
            .unwrap()
            .with_weight_decay(0.01)
            .unwrap()
            .with_safeguard_warmup(true);
        let (updates, estimates) = prodigy_parity_updates(&optimizer);
        assert_parity(&updates, &VARIANT);
        assert_parity(&[estimates], &[VARIANT_ESTIMATES]);
    }
}
