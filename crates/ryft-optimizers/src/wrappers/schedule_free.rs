//! Schedule-free optimization of [Defazio et al. (2024)](https://arxiv.org/abs/2405.15682), which replaces learning
//! rate schedules with an interpolation between an averaged sequence of iterates and the iterates of a base optimizer.

use ryft_core::{Add, ComparisonDirection, DataType, Div, Mul, Parameter, Parameterized, Sub};

use crate::macros::parameterized_state;
use crate::{
    AddDecayedWeights, Chain, Optimizer, OptimizerError, OptimizerValue, ScaleByRms, ScaleBySchedule, Schedule, Sgd,
    UpdateArguments, convert_to, increment_count, initial_count, statistic_data_type, validate_finite,
    validate_floating_point, validate_structure,
};

// TODO(eaplatanios): Review this module.

parameterized_state! {
    /// State of [`ScheduleFree`], which is the analogue of
    /// [`optax.contrib.ScheduleFreeState`](
    /// https://optax.readthedocs.io/en/latest/api/contrib.html#optax.contrib.ScheduleFreeState).
    /// It holds the interpolation coefficient, the sums of the weights and the maximum learning rate so far, the step
    /// count, the state `I` of the base optimizer (whose leaf paths start with `$.base_optimizer_state`), and the
    /// iterates `z` of the base optimizer with the structure `S` of the parameters. Its rank-0 floating-point arrays
    /// have the statistic data type of the first parameter.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct ScheduleFreeState<P, S, I> with family ScheduleFreeStateFamily {
        /// Rank-0 array that holds the interpolation coefficient `β₁`.
        b1: P,

        /// Rank-0 array that holds the sum of the weights of the iterates so far.
        weight_sum: P,

        /// Rank-0 `i32` array that holds the number of updates applied so far plus one, at which the learning rate
        /// schedule is evaluated.
        step_count: P,

        /// Rank-0 array that holds the maximum learning rate so far, which is the `max_lr` of Optax.
        maximum_learning_rate: P,

        /// State of the base optimizer.
        base_optimizer_state: I,

        /// Iterates `z` of the base optimizer, with the structure of the parameters.
        z: S,
    }
}

impl<P: Parameter, S, I> ScheduleFreeState<P, S, I> {
    /// Returns the rank-0 array that holds the interpolation coefficient `β₁`.
    #[inline]
    pub fn b1(&self) -> &P {
        &self.b1
    }

    /// Returns the rank-0 array that holds the sum of the weights of the iterates so far.
    #[inline]
    pub fn weight_sum(&self) -> &P {
        &self.weight_sum
    }

    /// Returns the rank-0 `i32` array that holds the number of updates applied so far plus one.
    #[inline]
    pub fn step_count(&self) -> &P {
        &self.step_count
    }

    /// Returns the rank-0 array that holds the maximum learning rate so far.
    #[inline]
    pub fn maximum_learning_rate(&self) -> &P {
        &self.maximum_learning_rate
    }

    /// Returns the state of the base optimizer.
    #[inline]
    pub fn base_optimizer_state(&self) -> &I {
        &self.base_optimizer_state
    }

    /// Returns the iterates `z` of the base optimizer, with the structure of the parameters.
    #[inline]
    pub fn z(&self) -> &S {
        &self.z
    }
}

impl<P: OptimizerValue, S: Parameterized<P>, I> ScheduleFreeState<P, S, I> {
    /// Returns the parameters at which a [`ScheduleFree`] optimizer with this state should be evaluated (i.e., the
    /// averaged iterates `x = (y - (1 - β₁) · z) / β₁` for the training parameters `y`), which is the analogue of
    /// [`optax.contrib.schedule_free_eval_params`](
    /// https://optax.readthedocs.io/en/latest/api/contrib.html#optax.contrib.schedule_free_eval_params).
    /// The evaluation parameters are computed in the statistic data type of every parameter and returned in its data
    /// type.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::StructureMismatch`] when `parameters` does not have the structure of the iterates of
    /// this state, and other [`OptimizerError`]s when the array operations fail.
    pub fn evaluation_parameters(&self, parameters: &S) -> Result<S, OptimizerError> {
        validate_structure("parameters", &self.z.parameter_paths().collect::<Vec<_>>(), parameters)?;
        let leaves = parameters
            .parameters()
            .zip(self.z.parameters())
            .map(|(parameter, z)| {
                let data_type = parameter.r#type().data_type();
                let statistic_type = statistic_data_type(data_type);
                let b1 = convert_to(&self.b1, statistic_type)?;
                let z = convert_to(z, statistic_type)?;
                let x = interpolate_back(&convert_to(parameter, statistic_type)?, &z, &b1)?;
                Ok(convert_to(&x, data_type)?)
            })
            .collect::<Result<Vec<_>, OptimizerError>>()?;
        Ok(S::from_parameters(parameters.parameter_structure(), leaves)?)
    }
}

/// Schedule-free wrapper of [Defazio et al. (2024)](https://arxiv.org/abs/2405.15682), which replaces a decaying
/// learning rate schedule with an interpolation between the iterates of a base optimizer (e.g., SGD or Adam with a
/// constant learning rate) and their weighted average, and which is the analogue of
/// [`optax.contrib.schedule_free`](
/// https://optax.readthedocs.io/en/latest/api/contrib.html#optax.contrib.schedule_free).
/// Its update is
///
/// ```text
/// γₜ    = η(t)
/// γₜᵐᵃˣ = max(γₜ₋₁ᵐᵃˣ, γₜ)
/// wₜ    = (γₜᵐᵃˣ)ʳ
/// cₜ    = wₜ / (w₁ + … + wₜ)
/// zₜ    = zₜ₋₁ + Δₜ
/// xₜ    = (1 − cₜ) · xₜ₋₁ + cₜ · zₜ
/// yₜ    = β₁ · xₜ + (1 − β₁) · zₜ
/// uₜ    = yₜ − yₜ₋₁
/// ```
///
/// Here `t ≥ 1` counts the updates including the current one, the training parameters `yₜ₋₁` are those at which the
/// gradient `gₜ` is computed and to which the update `uₜ` is added, `Δₜ` is the update of the base optimizer for `gₜ`
/// at `yₜ₋₁`, `zₜ` are the iterates of the base optimizer, and `xₜ` is their weighted average, which gives the
/// parameters to evaluate and which [`ScheduleFreeState::evaluation_parameters`] recovers as
/// `xₜ = (yₜ − (1 − β₁) · zₜ) / β₁`. All three sequences start at the initial parameters (i.e., `x₀ = y₀ = z₀`). The
/// weight `wₜ` of every iterate is the maximum `γₜᵐᵃˣ` of the learning rate [`Schedule`] `η` so far, which starts at
/// zero, raised to the weight learning rate power `r`, and `cₜ = 0` while all weights are zero. All operations are
/// element-wise. The defaults are `β₁ = 0.9` and `r = 2`. [`Self::with_b1`] sets `β₁`,
/// [`Self::with_weight_learning_rate_power`] sets `r`, and [`Self::with_state_data_type`] only changes the data type in
/// which `zₜ` is stored. Note that the schedule is evaluated at `t` (i.e., at the step count plus one) here, while
/// optimizers evaluate their learning rate schedules at `t − 1`, so the base optimizer that [`Self::sgd`] and
/// [`Self::adamw`] create evaluates the same schedule one step earlier. The warmup of Optax (i.e., `warmup_steps`) is
/// the learning rate schedule `LinearSchedule::new(0.0, learning_rate, warmup_steps)` (refer to
/// [`LinearSchedule`](crate::LinearSchedule)), which is passed as the learning rate.
///
/// [`Self::sgd`] creates schedule-free SGD, whose base update is `Δₜ = −η(t − 1) · (gₜ + λ · yₜ₋₁)` with the weight
/// decay `λ`, and [`Self::adamw`] creates schedule-free AdamW, whose base update is
///
/// ```text
/// vₜ = β₂ · vₜ₋₁ + (1 − β₂) · gₜ²
/// Δₜ = −η(t − 1) · (gₜ / (√(vₜ / (1 − β₂ᵗ)) + ε) + λ · yₜ₋₁)
/// ```
///
/// with `v₀ = 0` and the defaults `β₂ = 0.999`, `ε = 1e-8`, and `λ = 0`.
///
/// This optimizer wraps its base optimizer rather than chaining transformations, and the base optimizers of
/// [`Self::sgd`] and [`Self::adamw`] are the chains of [`AddDecayedWeights`] and [`Sgd`] and of [`ScaleByRms`],
/// [`AddDecayedWeights`], and [`ScaleBySchedule::learning_rate`].
///
/// # References
///
///   - Defazio et al. (2024). [The Road Less Scheduled](https://arxiv.org/abs/2405.15682).
#[derive(Clone, Debug, PartialEq)]
pub struct ScheduleFree<O, Sc: Schedule = f64> {
    /// Optimizer whose iterates are averaged.
    base_optimizer: O,

    /// Learning rate [`Schedule`] that determines the weights of the iterates.
    learning_rate: Sc,

    /// Interpolation coefficient `β₁` between the averaged iterates and the iterates of the base optimizer.
    b1: f64,

    /// Power of the maximum learning rate that weights every iterate.
    weight_learning_rate_power: f64,

    /// Data type in which the iterates `z` are stored, or [`None`] for the data type of each parameter.
    state_data_type: Option<DataType>,
}

impl<O, Sc: Schedule> ScheduleFree<O, Sc> {
    /// Creates a new [`ScheduleFree`] that averages the iterates of `base_optimizer` with weights derived from the
    /// learning rate schedule `learning_rate`, which should be the learning rate of `base_optimizer`.
    #[inline]
    pub fn new(base_optimizer: O, learning_rate: Sc) -> Self {
        Self { base_optimizer, learning_rate, b1: 0.9, weight_learning_rate_power: 2.0, state_data_type: None }
    }

    /// Returns this optimizer with `b1` as its interpolation coefficient `β₁`.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `b1` is not in `(0, 1]`.
    pub fn with_b1(mut self, b1: f64) -> Result<Self, OptimizerError> {
        if !(b1 > 0.0 && b1 <= 1.0) {
            return Err(OptimizerError::InvalidConfiguration {
                message: format!("`b1` must be in `(0, 1]` but got `{b1}`"),
            });
        }
        self.b1 = b1;
        Ok(self)
    }

    /// Returns this optimizer with `weight_learning_rate_power` as the power of the maximum learning rate that weights
    /// every iterate.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `weight_learning_rate_power` is not finite.
    pub fn with_weight_learning_rate_power(mut self, weight_learning_rate_power: f64) -> Result<Self, OptimizerError> {
        validate_finite("weight_learning_rate_power", weight_learning_rate_power)?;
        self.weight_learning_rate_power = weight_learning_rate_power;
        Ok(self)
    }

    /// Returns this optimizer with the iterates `z` stored in `data_type` (i.e., the `state_dtype` of Optax).
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `data_type` is not a floating-point data type.
    pub fn with_state_data_type(mut self, data_type: DataType) -> Result<Self, OptimizerError> {
        validate_floating_point("state_data_type", data_type)?;
        self.state_data_type = Some(data_type);
        Ok(self)
    }

    /// Returns the optimizer whose iterates this optimizer averages.
    #[inline]
    pub fn base_optimizer(&self) -> &O {
        &self.base_optimizer
    }

    /// Returns the learning rate [`Schedule`] that determines the weights of the iterates.
    #[inline]
    pub fn learning_rate(&self) -> &Sc {
        &self.learning_rate
    }

    /// Returns the interpolation coefficient `β₁` of this optimizer.
    #[inline]
    pub fn b1(&self) -> f64 {
        self.b1
    }

    /// Returns the power of the maximum learning rate that weights every iterate.
    #[inline]
    pub fn weight_learning_rate_power(&self) -> f64 {
        self.weight_learning_rate_power
    }

    /// Returns the data type in which this optimizer stores the iterates `z`, or [`None`] if it stores them in the
    /// data type of each parameter.
    #[inline]
    pub fn state_data_type(&self) -> Option<DataType> {
        self.state_data_type
    }
}

impl<Sc: Schedule + Clone> ScheduleFree<Chain<AddDecayedWeights, Sgd<Sc>>, Sc> {
    /// Creates a new [`ScheduleFreeSgd`], which is the analogue of [`optax.contrib.schedule_free_sgd`](
    /// https://optax.readthedocs.io/en/latest/api/contrib.html#optax.contrib.schedule_free_sgd) with the learning rate
    /// schedule `learning_rate` (e.g., `1.0`, which is the default of Optax) and without weight decay. Its base
    /// optimizer is [`Sgd`] after [`AddDecayedWeights`], whose update is `Δₜ = −η(t − 1) · (gₜ + λ · yₜ₋₁)` for the
    /// gradient `gₜ` at the training parameters `yₜ₋₁` and the weight decay `λ` (refer to [`ScheduleFree`] for the
    /// update of the wrapper).
    pub fn sgd(learning_rate: Sc) -> Self {
        Self::new(Chain::new(AddDecayedWeights::new(0.0).unwrap(), Sgd::new(learning_rate.clone())), learning_rate)
    }

    /// Returns this optimizer with `weight_decay` as the rate of the weight decay of its base optimizer.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `weight_decay` is not finite.
    pub fn with_weight_decay(mut self, weight_decay: f64) -> Result<Self, OptimizerError> {
        let sgd = self.base_optimizer.second().clone();
        let decay = self.base_optimizer.first().clone().with_rate(weight_decay)?;
        self.base_optimizer = Chain::new(decay, sgd);
        Ok(self)
    }
}

impl<Sc: Schedule + Clone> ScheduleFree<Chain<Chain<ScaleByRms, AddDecayedWeights>, ScaleBySchedule<Sc>>, Sc> {
    /// Creates a new [`ScheduleFreeAdamW`], which is the analogue of [`optax.contrib.schedule_free_adamw`](
    /// https://optax.readthedocs.io/en/latest/api/contrib.html#optax.contrib.schedule_free_adamw) with the learning
    /// rate schedule `learning_rate` (e.g., `0.0025`, which is the default of Optax). Its base optimizer is the chain
    /// of [`ScaleByRms`] (with the decay `β₂ = 0.999`, `ε = 1e-8` outside of the square root, and bias correction),
    /// [`AddDecayedWeights`] (without weight decay by default), and [`ScaleBySchedule::learning_rate`], whose update is
    /// `Δₜ = −η(t − 1) · (gₜ / (√v̂ₜ + ε) + λ · yₜ₋₁)` with the bias corrected moving average `v̂ₜ` of the squared
    /// gradients (refer to [`ScheduleFree`] for the update of the wrapper).
    pub fn adamw(learning_rate: Sc) -> Self {
        let rms = ScaleByRms::new()
            .with_decay(0.999)
            .unwrap()
            .with_epsilon(1e-8)
            .unwrap()
            .with_epsilon_in_sqrt(false)
            .with_bias_correction(true);
        let base_optimizer = Chain::new(rms, AddDecayedWeights::new(0.0).unwrap())
            .then(ScaleBySchedule::learning_rate(learning_rate.clone()));
        Self::new(base_optimizer, learning_rate)
    }

    /// Returns this optimizer with `b2` as the decay `β₂` of the second moment estimates of its base optimizer.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `b2` is not in `[0, 1)`.
    pub fn with_b2(mut self, b2: f64) -> Result<Self, OptimizerError> {
        let rms = self.base_optimizer.first().first().clone().with_decay(b2)?;
        self.base_optimizer =
            Chain::new(rms, self.base_optimizer.first().second().clone()).then(self.base_optimizer.second().clone());
        Ok(self)
    }

    /// Returns this optimizer with `epsilon` as the term that its base optimizer adds to the root mean squares.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `epsilon` is negative or not finite.
    pub fn with_epsilon(mut self, epsilon: f64) -> Result<Self, OptimizerError> {
        let rms = self.base_optimizer.first().first().clone().with_epsilon(epsilon)?;
        self.base_optimizer =
            Chain::new(rms, self.base_optimizer.first().second().clone()).then(self.base_optimizer.second().clone());
        Ok(self)
    }

    /// Returns this optimizer with `weight_decay` as the rate of the weight decay of its base optimizer.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `weight_decay` is not finite.
    pub fn with_weight_decay(mut self, weight_decay: f64) -> Result<Self, OptimizerError> {
        let decay = self.base_optimizer.first().second().clone().with_rate(weight_decay)?;
        self.base_optimizer =
            Chain::new(self.base_optimizer.first().first().clone(), decay).then(self.base_optimizer.second().clone());
        Ok(self)
    }
}

impl<P: OptimizerValue, O: Optimizer<P>, Sc: Schedule> Optimizer<P> for ScheduleFree<O, Sc> {
    type State<S: Parameterized<P>> = ScheduleFreeState<P, S, O::State<S>>;

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        let count = initial_count("ScheduleFree", parameters)?;
        let data_type = statistic_data_type(parameters.parameters().next().unwrap().r#type().data_type());
        let scalar = convert_to(&count, data_type)?;
        let z = parameters
            .parameters()
            .map(|parameter| {
                convert_to(parameter, self.state_data_type.unwrap_or_else(|| parameter.r#type().data_type()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(ScheduleFreeState {
            b1: scalar.fill_like(self.b1)?,
            weight_sum: scalar.fill_like(0f64)?,
            step_count: count.fill_like(1i32)?,
            maximum_learning_rate: scalar.fill_like(0f64)?,
            base_optimizer_state: self.base_optimizer.initialize(parameters)?,
            z: S::from_parameters(parameters.parameter_structure(), z)?,
        })
    }

    fn update<S: Parameterized<P>>(
        &self,
        gradients: S,
        state: Self::State<S>,
        parameters: &S,
        arguments: &UpdateArguments<'_, P, S>,
    ) -> Result<(S, Self::State<S>), OptimizerError> {
        // This follows the operation order of `optax.contrib.schedule_free`.
        let paths = gradients.parameter_paths().collect::<Vec<_>>();
        validate_structure("parameters", &paths, parameters)?;
        validate_structure("z", &paths, &state.z)?;
        let ScheduleFreeState { b1, weight_sum, step_count, maximum_learning_rate, base_optimizer_state, z } = state;
        let data_type = maximum_learning_rate.r#type().data_type();
        let learning_rate = convert_to(&self.learning_rate.value(&step_count)?, data_type)?;
        let maximum_learning_rate = maximum_learning_rate.max(&learning_rate)?;
        let weight = maximum_learning_rate.pow(&maximum_learning_rate.fill_like(self.weight_learning_rate_power)?)?;
        let next_weight_sum = Add::add(&weight_sum, &weight)?;
        let ratio = Div::div(&weight, &next_weight_sum)?;
        let nan = weight.fill_like(f64::NAN)?;
        let ratio = is_nan(&ratio)?.select(&ratio.fill_like(0f64)?, &ratio)?;
        let coefficient = is_nan(&weight)?.or(&is_nan(&next_weight_sum)?)?.select(&nan, &ratio)?;

        let (base_updates, base_optimizer_state) =
            self.base_optimizer.update(gradients, base_optimizer_state, parameters, arguments)?;
        let structure = base_updates.parameter_structure();
        let mut updates = Vec::with_capacity(paths.len());
        let mut iterates = Vec::with_capacity(paths.len());
        for ((base_update, previous_z), parameter) in
            base_updates.into_parameters().zip(z.into_parameters()).zip(parameters.parameters())
        {
            let update_type = base_update.r#type().data_type();
            let statistic_type = statistic_data_type(parameter.r#type().data_type());
            let z_type = previous_z.r#type().data_type();
            let next_z = convert_to(&Add::add(&previous_z, &convert_to(&base_update, z_type)?)?, z_type)?;
            let y = convert_to(parameter, statistic_type)?;
            let b1 = y.fill_like(self.b1)?;
            let one_minus_b1 = y.fill_like(1.0 - self.b1)?;
            let coefficient = convert_to(&coefficient, statistic_type)?;
            let previous_x = interpolate_back(&y, &convert_to(&previous_z, statistic_type)?, &b1)?;
            let z_next = convert_to(&next_z, statistic_type)?;
            let x = Add::add(
                &Mul::mul(&Sub::sub(&coefficient.fill_like(1f64)?, &coefficient)?, &previous_x)?,
                &Mul::mul(&coefficient, &z_next)?,
            )?;
            let new_y = Add::add(&Mul::mul(&b1, &x)?, &Mul::mul(&one_minus_b1, &z_next)?)?;
            updates.push(convert_to(&Sub::sub(&new_y, &y)?, update_type)?);
            iterates.push(next_z);
        }
        let state = ScheduleFreeState {
            b1,
            weight_sum: next_weight_sum,
            step_count: increment_count(&step_count)?,
            maximum_learning_rate,
            base_optimizer_state,
            z: S::from_parameters(structure.clone(), iterates)?,
        };
        Ok((S::from_parameters(structure, updates)?, state))
    }
}

/// Schedule-free SGD, which is the analogue of
/// [`optax.contrib.schedule_free_sgd`](
/// https://optax.readthedocs.io/en/latest/api/contrib.html#optax.contrib.schedule_free_sgd)
/// and which [`ScheduleFree::sgd`] creates.
pub type ScheduleFreeSgd<Sc = f64> = ScheduleFree<Chain<AddDecayedWeights, Sgd<Sc>>, Sc>;

/// Schedule-free AdamW, which is the analogue of
/// [`optax.contrib.schedule_free_adamw`](
/// https://optax.readthedocs.io/en/latest/api/contrib.html#optax.contrib.schedule_free_adamw)
/// and which [`ScheduleFree::adamw`] creates.
pub type ScheduleFreeAdamW<Sc = f64> =
    ScheduleFree<Chain<Chain<ScaleByRms, AddDecayedWeights>, ScaleBySchedule<Sc>>, Sc>;

/// Returns the averaged iterate `(y - (1 - β₁) · z) / β₁` of the training parameter `y` and the iterate `z` of the
/// base optimizer, for the rank-0 coefficient `b1`.
fn interpolate_back<P: OptimizerValue>(y: &P, z: &P, b1: &P) -> Result<P, OptimizerError> {
    let b1 = convert_to(b1, y.r#type().data_type())?;
    let one_minus_b1 = Sub::sub(&b1.fill_like(1f64)?, &b1)?;
    Ok(Div::div(&Sub::sub(y, &Mul::mul(&one_minus_b1, z)?)?, &b1)?)
}

/// Returns whether the elements of `value` are NaN.
fn is_nan<P: OptimizerValue>(value: &P) -> Result<P, OptimizerError> {
    Ok(value.compare(value, ComparisonDirection::NotEqual)?)
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use ryft_core::Array;

    use crate::tests::{
        ArgumentProbe, PARITY_GRADIENTS, PARITY_PARAMETERS, assert_parity, parity_elements, parity_parameters,
        state_paths,
    };
    use crate::{LinearSchedule, Scale, apply_updates};

    use super::*;

    /// Asserts that every element of `actual` is within `1e-5` of the corresponding element of `expected`.
    fn assert_close(actual: &Array, expected: &[f32]) {
        let actual = actual.elements::<f32>().unwrap();
        assert_eq!(actual.len(), expected.len());
        assert!(actual.iter().zip(expected).all(|(actual, expected)| (actual - expected).abs() < 1e-5), "{actual:?}");
    }

    #[test]
    fn test_schedule_free_state_evaluation_parameters() {
        // With `β₁ = 0.9`, `x = (y - 0.1 · z) / 0.9` (refer to `test_schedule_free_update`).
        let optimizer = ScheduleFree::sgd(1.0);
        let parameters = vec![Array::scalar(1.0f32).unwrap()];
        let state = optimizer.initialize(&parameters).unwrap();
        let (updates, state) = optimizer
            .update(vec![Array::scalar(2.0f32).unwrap()], state, &parameters, &UpdateArguments::new())
            .unwrap();
        let parameters = apply_updates(parameters, updates).unwrap();
        let (updates, state) = optimizer
            .update(vec![Array::scalar(1.0f32).unwrap()], state, &parameters, &UpdateArguments::new())
            .unwrap();
        let parameters = apply_updates(parameters, updates).unwrap();
        assert_close(&state.evaluation_parameters(&parameters).unwrap()[0], &[-1.5]);
        assert!(matches!(
            state.evaluation_parameters(&Vec::new()),
            Err(OptimizerError::StructureMismatch { message })
                if message == "expected the leaf count of `parameters` to be 1 but got 0",
        ));
    }

    #[test]
    fn test_schedule_free_new() {
        let optimizer = ScheduleFree::new(Scale::new(-1.0).unwrap(), 1.0);
        assert_eq!(optimizer.base_optimizer(), &Scale::new(-1.0).unwrap());
        assert_eq!(optimizer.learning_rate(), &1.0);
        assert_eq!(optimizer.b1(), 0.9);
        assert_eq!(optimizer.weight_learning_rate_power(), 2.0);
        assert_eq!(optimizer.state_data_type(), None);
    }

    #[test]
    fn test_schedule_free_with_hyperparameters() {
        let optimizer = ScheduleFree::new(Scale::new(-1.0).unwrap(), 1.0)
            .with_b1(0.5)
            .unwrap()
            .with_weight_learning_rate_power(1.0)
            .unwrap()
            .with_state_data_type(DataType::F64)
            .unwrap();
        assert_eq!(optimizer.b1(), 0.5);
        assert_eq!(optimizer.weight_learning_rate_power(), 1.0);
        assert_eq!(optimizer.state_data_type(), Some(DataType::F64));
        let state = optimizer.initialize(&vec![Array::scalar(1.0f32).unwrap()]).unwrap();
        assert_eq!(state.z(), &vec![Array::scalar(1.0f64).unwrap()]);
        assert!(matches!(
            ScheduleFree::new(Scale::new(-1.0).unwrap(), 1.0).with_b1(0.0),
            Err(OptimizerError::InvalidConfiguration { message }) if message == "`b1` must be in `(0, 1]` but got `0`",
        ));
    }

    #[test]
    fn test_schedule_free_sgd() {
        let optimizer = ScheduleFree::sgd(LinearSchedule::new(0.0, 1.0, 10).unwrap()).with_weight_decay(0.1).unwrap();
        assert_eq!(optimizer.base_optimizer().first().rate(), 0.1);
        assert_eq!(optimizer.learning_rate(), &LinearSchedule::new(0.0, 1.0, 10).unwrap());
    }

    #[test]
    fn test_schedule_free_adamw() {
        let optimizer = ScheduleFree::adamw(0.0025)
            .with_b2(0.99)
            .unwrap()
            .with_epsilon(1e-6)
            .unwrap()
            .with_weight_decay(0.1)
            .unwrap();
        let rms = optimizer.base_optimizer().first().first();
        assert_eq!(
            (rms.decay(), rms.epsilon(), rms.epsilon_in_sqrt(), rms.bias_correction()),
            (0.99, 1e-6, false, true)
        );
        assert_eq!(optimizer.base_optimizer().first().second().rate(), 0.1);
        assert!(matches!(ScheduleFree::adamw(0.0025).with_b2(1.0), Err(OptimizerError::InvalidConfiguration { .. }),));
    }

    #[test]
    fn test_schedule_free_initialize() {
        assert_eq!(
            state_paths(&ScheduleFree::sgd(1.0)),
            vec![
                "$.b1",
                "$.weight_sum",
                "$.step_count",
                "$.maximum_learning_rate",
                "$.base_optimizer_state.1.learning_rate.count",
                "$.z[\"bias\"]",
                "$.z[\"weights\"]",
            ],
        );
        let state = ScheduleFree::sgd(1.0).initialize(&vec![Array::scalar(2.0f32).unwrap()]).unwrap();
        assert_eq!(state.step_count(), &Array::scalar(1i32).unwrap());
        assert_eq!(state.z(), &vec![Array::scalar(2.0f32).unwrap()]);
    }

    #[test]
    fn test_schedule_free_update() {
        // Schedule-free SGD with `η = 1` from `y = z = 1`. The first gradient `2` gives `z = -1`, the weight `1² = 1`
        // with the coefficient `c = 1 / 1`, `x = (1 - 0.1 · 1) / 0.9 = 1` before and `x = -1` after the interpolation,
        // and `y = 0.9 · -1 + 0.1 · -1 = -1`, so the update is `-2`. The second gradient `1` gives `z = -2` and
        // `c = 1 / 2`, `x = (-1 + 0.1) / 0.9 = -1` before and `x = 0.5 · -1 + 0.5 · -2 = -1.5` after the
        // interpolation, and `y = 0.9 · -1.5 + 0.1 · -2 = -1.55`, so the update is `-0.55`.
        let optimizer = ScheduleFree::sgd(1.0);
        let parameters = vec![Array::scalar(1.0f32).unwrap()];
        let arguments = UpdateArguments::new();
        let state = optimizer.initialize(&parameters).unwrap();
        let (updates, state) =
            optimizer.update(vec![Array::scalar(2.0f32).unwrap()], state, &parameters, &arguments).unwrap();
        assert_close(&updates[0], &[-2.0]);
        assert_eq!(state.weight_sum(), &Array::scalar(1.0f32).unwrap());
        let parameters = apply_updates(parameters, updates).unwrap();
        let (updates, state) =
            optimizer.update(vec![Array::scalar(1.0f32).unwrap()], state, &parameters, &arguments).unwrap();
        assert_close(&updates[0], &[-0.55]);
        assert_eq!(state.z(), &vec![Array::scalar(-2.0f32).unwrap()]);
        assert_eq!(state.step_count(), &Array::scalar(3i32).unwrap());
        assert_eq!(state.maximum_learning_rate(), &Array::scalar(1.0f32).unwrap());
    }

    #[test]
    fn test_schedule_free_update_arguments() {
        // The base optimizer receives the arguments: its update is `10 + 0.5 + 1 = 11.5`, so `z = 12.5`, and with
        // `c = 1`, the new parameter is `z` itself, which is the update `11.5`.
        let optimizer = ScheduleFree::new(ArgumentProbe, 1.0);
        let parameters = vec![Array::scalar(1.0f32).unwrap()];
        let value = Array::scalar(0.5f32).unwrap();
        let argument_gradients = vec![Array::scalar(10.0f32).unwrap()];
        let objective = |parameters: &Vec<Array>| -> Result<Array, OptimizerError> { Ok(parameters[0].clone()) };
        let arguments = UpdateArguments::new()
            .with_value(&value)
            .with_gradients(&argument_gradients)
            .with_objective(&objective);
        let state = optimizer.initialize(&parameters).unwrap();
        let (updates, _) = optimizer.update(parameters.clone(), state, &parameters, &arguments).unwrap();
        assert_close(&updates[0], &[11.5]);
    }

    /// Returns the updates of `optimizer` for the parity gradients (refer to [`parity_updates`]), each followed by the
    /// evaluation parameters after that update has been applied.
    fn schedule_free_parity_updates<O: Optimizer<Array>, Sc: Schedule>(
        optimizer: &ScheduleFree<O, Sc>,
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let mut parameters = parity_parameters(&PARITY_PARAMETERS);
        let mut state = optimizer.initialize(&parameters).unwrap();
        let (mut updates, mut evaluation_parameters) = (Vec::new(), Vec::new());
        for gradients in &PARITY_GRADIENTS {
            let (step_updates, next_state) =
                optimizer.update(parity_parameters(gradients), state, &parameters, &UpdateArguments::new()).unwrap();
            updates.push(parity_elements(&step_updates));
            parameters = apply_updates(parameters, step_updates).unwrap();
            evaluation_parameters.push(parity_elements(&next_state.evaluation_parameters(&parameters).unwrap()));
            state = next_state;
        }
        (updates, evaluation_parameters)
    }

    #[test]
    fn test_schedule_free_optax_parity() {
        // Optax 9ecde04b: `optax.contrib.schedule_free_sgd(0.5, warmup_steps=2, b1=0.9, weight_decay=0.01)`, with the
        // evaluation parameters of `optax.contrib.schedule_free_eval_params` after every step. The warmup is the
        // learning rate schedule that goes from `0` to `0.5` over two steps, as in Optax.
        const SGD: [[f32; 9]; 4] = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.1920929e-7, -5.9604645e-8],
            [
                0.009225011,
                -0.04919994,
                -0.02459991,
                -0.04305005,
                -0.06047499,
                0.08148751,
                -0.01639986,
                0.037925124,
                -0.07328743,
            ],
            [
                -0.07537305,
                0.022822976,
                0.042661548,
                0.03090763,
                -0.016998827,
                -0.1053787,
                -0.059059143,
                -0.10014486,
                0.016158164,
            ],
            [
                -0.061632425,
                -0.019681811,
                0.07510018,
                -0.011520922,
                0.051942468,
                -0.04166189,
                0.07042408,
                -0.070931196,
                -0.05342245,
            ],
        ];
        const SGD_EVALUATION: [[f32; 9]; 4] = [
            [0.5, -1.0, 2.0, 1.0, -0.5, 0.25, -2.0, 1.4999999, 0.74999994],
            [0.509, -1.048, 1.9760001, 0.95799994, -0.559, 0.32950002, -2.0159998, 1.537, 0.6785],
            [0.44220173, -1.0287795, 2.0133882, 0.9845401, -0.5754212, 0.23759672, -2.0688524, 1.4488046, 0.69127387],
            [0.3850734, -1.0428851, 2.0785096, 0.9778434, -0.5346948, 0.19420913, -2.0167565, 1.3819021, 0.64896727],
        ];
        let optimizer = ScheduleFree::sgd(LinearSchedule::new(0.0, 0.5, 2).unwrap())
            .with_b1(0.9)
            .unwrap()
            .with_weight_decay(0.01)
            .unwrap();
        let (updates, evaluation_parameters) = schedule_free_parity_updates(&optimizer);
        assert_parity(&updates, &SGD);
        assert_parity(&evaluation_parameters, &SGD_EVALUATION);

        // Optax 9ecde04b: `optax.contrib.schedule_free_adamw(0.1, warmup_steps=2, b1=0.9, b2=0.999, eps=1e-8,
        // weight_decay=0.01)`, with the evaluation parameters after every step.
        const ADAMW: [[f32; 9]; 4] = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.1920929e-7, -5.9604645e-8],
            [
                0.025729299,
                -0.04486394,
                -0.019159198,
                -0.026344359,
                -0.054790735,
                0.05175054,
                -0.017519236,
                0.0209229,
                -0.029526591,
            ],
            [
                -0.07887098,
                0.021968603,
                0.04343927,
                0.024491012,
                -0.01859355,
                -0.056516543,
                -0.05441332,
                -0.05036378,
                0.009103835,
            ],
            [
                -0.0641703,
                -0.023972988,
                0.06395006,
                -0.008679986,
                0.045618415,
                -0.023016736,
                0.04004121,
                -0.037437797,
                -0.027338564,
            ],
        ];
        const ADAMW_EVALUATION: [[f32; 9]; 4] = [
            [0.5, -1.0, 2.0, 1.0, -0.5, 0.25, -2.0, 1.4999999, 0.74999994],
            [0.5251018, -1.0437697, 1.9813081, 0.97429824, -0.5534544, 0.30048835, -2.017092, 1.5204124, 0.7211935],
            [0.45555204, -1.0252148, 2.0195055, 0.9954968, -0.57116985, 0.25137338, -2.065839, 1.4760983, 0.7286457],
            [0.39607123, -1.0428911, 2.0756073, 0.99057424, -0.535738, 0.22757247, -2.0381267, 1.4410149, 0.7070889],
        ];
        let optimizer = ScheduleFree::adamw(LinearSchedule::new(0.0, 0.1, 2).unwrap())
            .with_b1(0.9)
            .unwrap()
            .with_b2(0.999)
            .unwrap()
            .with_epsilon(1e-8)
            .unwrap()
            .with_weight_decay(0.01)
            .unwrap();
        let (updates, evaluation_parameters) = schedule_free_parity_updates(&optimizer);
        assert_parity(&updates, &ADAMW);
        assert_parity(&evaluation_parameters, &ADAMW_EVALUATION);

        // Optax 9ecde04b: `optax.contrib.schedule_free_adamw(0.1)`, without warmup and without weight decay, with the
        // evaluation parameters after every step.
        const ADAMW_DEFAULT: [[f32; 9]; 4] = [
            [
                -0.09999937,
                0.09999943,
                -0.09999931,
                -0.09999943,
                0.09999937,
                -0.09999935,
                0.09999931,
                -0.09999931,
                0.09999931,
            ],
            [
                0.03478995,
                -0.060733438,
                -0.02460134,
                -0.03478992,
                -0.073774755,
                0.06955899,
                -0.02460134,
                0.028892279,
                -0.039196372,
            ],
            [
                -0.05540654,
                0.004095018,
                0.030329823,
                0.012550175,
                -0.030940771,
                -0.030160993,
                -0.049679756,
                -0.03343606,
                -0.00081425905,
            ],
            [
                -0.05401817,
                -0.027076006,
                0.054932714,
                -0.0099666715,
                0.030317545,
                -0.014046103,
                0.02893281,
                -0.030129552,
                -0.027201176,
            ],
        ];
        const ADAMW_DEFAULT_EVALUATION: [[f32; 9]; 4] = [
            [0.40000063, -0.9000006, 1.9000006, 0.9000006, -0.40000063, 0.15000063, -1.9000006, 1.4000007, 0.84999925],
            [0.43162784, -0.95521283, 1.877636, 0.86837333, -0.4670686, 0.21323608, -1.9223654, 1.4262663, 0.8143662],
            [0.38809136, -0.9564012, 1.901047, 0.87619627, -0.49844155, 0.19337155, -1.9656291, 1.4005919, 0.81071824],
            [0.33984098, -0.9774118, 1.9469045, 0.8697331, -0.47994697, 0.17951077, -1.950029, 1.3734654, 0.7892331],
        ];
        let (updates, evaluation_parameters) = schedule_free_parity_updates(&ScheduleFree::adamw(0.1));
        assert_parity(&updates, &ADAMW_DEFAULT);
        assert_parity(&evaluation_parameters, &ADAMW_DEFAULT_EVALUATION);
    }
}
