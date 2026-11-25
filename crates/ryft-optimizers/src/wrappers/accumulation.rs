//! Wrappers that accumulate gradients or updates over several steps.
//!
//! [`MultiSteps`] accumulates gradients over several mini-steps and applies another optimizer to their average (or
//! sum) once per gradient step, optionally rejecting mini-steps whose gradients an [`UpdateSkipCriterion`] flags.
//! [`ApplyEvery`] accumulates the updates and emits their sum every few steps.

use ryft_core::{Add, ComparisonDirection, DataType, Div, Mul, Not, Parameter, Parameterized, ReductionKind, Rem, Sub};

use crate::macros::parameterized_state;
use crate::transformations::normalization::global_squared_norm;
use crate::{
    Optimizer, OptimizerError, OptimizerValue, Schedule, UpdateArguments, all_axes, convert_to, filled_statistics,
    increment_count, initial_count, statistic_data_type, validate_floating_point, validate_non_negative,
    validate_structure,
};

use super::select_structure;

// TODO(eaplatanios): Review this module.

/// Criterion under which [`MultiSteps`] rejects the gradients of a mini-step, as if that mini-step never happened,
/// which is the analogue of the `should_skip_update_fn` of
/// [`optax.MultiSteps`](https://optax.readthedocs.io/en/latest/api/optimizer_wrappers.html#optax.MultiSteps) with the
/// functions that Optax provides for it.
#[derive(Clone, Debug, PartialEq)]
pub enum UpdateSkipCriterion {
    /// Rejects gradients that contain a NaN or an infinity, which is [`optax.skip_not_finite`](
    /// https://optax.readthedocs.io/en/latest/api/optimizer_wrappers.html#optax.skip_not_finite). The state of
    /// [`MultiSteps`] records the number of elements of the gradients that were not finite.
    NotFinite,

    /// Rejects gradients whose squared global norm is not less than `maximum_squared_norm`, which is
    /// [`optax.skip_large_updates`](
    /// https://optax.readthedocs.io/en/latest/api/optimizer_wrappers.html#optax.skip_large_updates).
    /// The state of [`MultiSteps`] records the squared global norm of the gradients.
    LargeUpdates {
        /// Squared global norm from which gradients are rejected.
        maximum_squared_norm: f64,
    },
}

parameterized_state! {
    /// State of [`MultiSteps`], which is the analogue of [`optax.MultiStepsState`](
    /// https://optax.readthedocs.io/en/latest/api/optimizer_wrappers.html#optax.MultiStepsState). It holds the step
    /// counters, the state `I` of the wrapped optimizer (whose leaf paths start with `$.inner_state`), the accumulated
    /// gradients with the structure `S` of the parameters, and the record of the last decision of the
    /// [`UpdateSkipCriterion`], whose fields hold one element when the criterion records them and none otherwise.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct MultiStepsState<P, S, I> with family MultiStepsStateFamily {
        /// Rank-0 `i32` array that holds the number of mini-steps accumulated since the last gradient step.
        mini_step: P,

        /// Rank-0 `i32` array that holds the number of gradient steps (i.e., updates of the wrapped optimizer) so far.
        gradient_step: P,

        /// State of the wrapped optimizer, which is updated once per gradient step.
        inner_state: I,

        /// Gradients accumulated since the last gradient step, with the structure of the parameters.
        accumulated_gradients: S,

        /// Rank-0 boolean array that holds whether the last mini-step was rejected, when there is a skip criterion.
        should_skip: Vec<P>,

        /// Rank-0 `i32` array that holds the number of elements of the last gradients that were not finite, when the
        /// skip criterion is [`UpdateSkipCriterion::NotFinite`].
        not_finite_count: Vec<P>,

        /// Rank-0 array that holds the squared global norm of the last gradients, when the skip criterion is
        /// [`UpdateSkipCriterion::LargeUpdates`].
        squared_norm: Vec<P>,
    }
}

impl<P: Parameter, S, I> MultiStepsState<P, S, I> {
    /// Returns the rank-0 `i32` array that holds the number of mini-steps accumulated since the last gradient step.
    #[inline]
    pub fn mini_step(&self) -> &P {
        &self.mini_step
    }

    /// Returns the rank-0 `i32` array that holds the number of gradient steps so far.
    #[inline]
    pub fn gradient_step(&self) -> &P {
        &self.gradient_step
    }

    /// Returns the state of the wrapped optimizer.
    #[inline]
    pub fn inner_state(&self) -> &I {
        &self.inner_state
    }

    /// Returns the gradients accumulated since the last gradient step, with the structure of the parameters.
    #[inline]
    pub fn accumulated_gradients(&self) -> &S {
        &self.accumulated_gradients
    }

    /// Returns the rank-0 boolean array that holds whether the last mini-step was rejected, or [`None`] when there is
    /// no skip criterion.
    #[inline]
    pub fn should_skip(&self) -> Option<&P> {
        self.should_skip.first()
    }

    /// Returns the rank-0 `i32` array that holds the number of elements of the last gradients that were not finite, or
    /// [`None`] when the skip criterion is not [`UpdateSkipCriterion::NotFinite`].
    #[inline]
    pub fn not_finite_count(&self) -> Option<&P> {
        self.not_finite_count.first()
    }

    /// Returns the rank-0 array that holds the squared global norm of the last gradients, or [`None`] when the skip
    /// criterion is not [`UpdateSkipCriterion::LargeUpdates`].
    #[inline]
    pub fn squared_norm(&self) -> Option<&P> {
        self.squared_norm.first()
    }
}

impl<P: OptimizerValue, S, I> MultiStepsState<P, S, I> {
    /// Returns a rank-0 boolean array that holds whether the last update of [`MultiSteps`] was a gradient step (i.e.,
    /// whether it applied the wrapped optimizer), which is `optax.MultiSteps.has_updated`.
    ///
    /// # Errors
    ///
    /// Returns an [`OptimizerError`] when one of the array operations that compute the result fails.
    pub fn has_updated(&self) -> Result<P, OptimizerError> {
        let zero = self.mini_step.fill_like(0i32)?;
        let accumulating = self.mini_step.compare(&zero, ComparisonDirection::Equal)?;
        Ok(accumulating.and(&self.gradient_step.compare(&zero, ComparisonDirection::GreaterThan)?)?)
    }
}

/// Optimizer that accumulates gradients over several mini-steps and applies another optimizer once per gradient step,
/// which is the analogue of
/// [`optax.MultiSteps`](https://optax.readthedocs.io/en/latest/api/optimizer_wrappers.html#optax.MultiSteps). The
/// number `k` of mini-steps per gradient step is the value of a [`Schedule`] at the current gradient step, which is
/// converted to `i32` by rounding toward zero, so that the effective batch size can vary over training. At each
/// mini-step, the gradients `g` are accumulated as `a = a + (g - a) / (n + 1)` (i.e., their running mean, where `n`
/// is the number of mini-steps accumulated so far) or, when the gradient mean is disabled, as `a = a + g`. When the
/// mini-step is the `k`-th one, the wrapped optimizer is applied to `a`, its updates are returned, and the accumulated
/// gradients are reset to zero. All other mini-steps return zero updates and keep the state of the wrapped optimizer.
///
/// With an [`UpdateSkipCriterion`], a mini-step whose gradients the criterion flags returns zero updates and leaves the
/// counters, the accumulated gradients, and the state of the wrapped optimizer unchanged. The decisions are made with
/// array operations (i.e., the inner update is always computed and its results are selected), so they are staged
/// inside compiled training steps. The accumulated gradients are stored in the statistic data type of each leaf by
/// default, where Optax stores them in `f32`.
#[derive(Clone, Debug, PartialEq)]
pub struct MultiSteps<O, Sc: Schedule = f64> {
    /// Optimizer that is applied to the accumulated gradients once per gradient step.
    inner: O,

    /// [`Schedule`] of the number of mini-steps per gradient step, which is evaluated at the gradient step.
    every_k_schedule: Sc,

    /// Whether the accumulated gradients are averaged rather than summed.
    gradient_mean: bool,

    /// Criterion under which the gradients of a mini-step are rejected, or [`None`] if none are rejected.
    skip_criterion: Option<UpdateSkipCriterion>,

    /// Data type in which the accumulated gradients are stored, or [`None`] for the statistic data type of each leaf.
    accumulator_data_type: Option<DataType>,
}

impl<O> MultiSteps<O> {
    /// Creates a new [`MultiSteps`] that applies `inner` to the average of the gradients of every `every_k`
    /// consecutive mini-steps, without a skip criterion.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `every_k` is zero or greater than `i32::MAX`.
    pub fn new(inner: O, every_k: usize) -> Result<Self, OptimizerError> {
        if every_k == 0 || i32::try_from(every_k).is_err() {
            return Err(OptimizerError::InvalidConfiguration {
                message: format!("`every_k` must be in `[1, {}]` but got `{every_k}`", i32::MAX),
            });
        }
        Ok(Self {
            inner,
            every_k_schedule: every_k as f64,
            gradient_mean: true,
            skip_criterion: None,
            accumulator_data_type: None,
        })
    }
}

impl<O, Sc: Schedule> MultiSteps<O, Sc> {
    /// Returns this optimizer with `schedule` as the [`Schedule`] of the number of mini-steps per gradient step, which
    /// is evaluated at the number of gradient steps so far and must return positive integers.
    #[inline]
    pub fn with_every_k_schedule<T: Schedule>(self, schedule: T) -> MultiSteps<O, T> {
        MultiSteps {
            inner: self.inner,
            every_k_schedule: schedule,
            gradient_mean: self.gradient_mean,
            skip_criterion: self.skip_criterion,
            accumulator_data_type: self.accumulator_data_type,
        }
    }

    /// Returns this optimizer with the accumulated gradients averaged when `gradient_mean` is `true` and summed when it
    /// is `false`.
    #[inline]
    pub fn with_gradient_mean(mut self, gradient_mean: bool) -> Self {
        self.gradient_mean = gradient_mean;
        self
    }

    /// Returns this optimizer with `skip_criterion` as the criterion under which it rejects the gradients of a
    /// mini-step, or without one when it is [`None`].
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when the maximum squared norm of
    /// [`UpdateSkipCriterion::LargeUpdates`] is negative or not finite.
    pub fn with_skip_criterion(mut self, skip_criterion: Option<UpdateSkipCriterion>) -> Result<Self, OptimizerError> {
        if let Some(UpdateSkipCriterion::LargeUpdates { maximum_squared_norm }) = &skip_criterion {
            validate_non_negative("maximum_squared_norm", *maximum_squared_norm)?;
        }
        self.skip_criterion = skip_criterion;
        Ok(self)
    }

    /// Returns this optimizer with the accumulated gradients stored in `data_type` (i.e., the `accumulator_dtype` of
    /// Optax).
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `data_type` is not a floating-point data type.
    pub fn with_accumulator_data_type(mut self, data_type: DataType) -> Result<Self, OptimizerError> {
        validate_floating_point("accumulator_data_type", data_type)?;
        self.accumulator_data_type = Some(data_type);
        Ok(self)
    }

    /// Returns the optimizer that this optimizer applies once per gradient step.
    #[inline]
    pub fn inner(&self) -> &O {
        &self.inner
    }

    /// Returns the [`Schedule`] of the number of mini-steps per gradient step of this optimizer.
    #[inline]
    pub fn every_k_schedule(&self) -> &Sc {
        &self.every_k_schedule
    }

    /// Returns whether this optimizer averages the accumulated gradients rather than summing them.
    #[inline]
    pub fn gradient_mean(&self) -> bool {
        self.gradient_mean
    }

    /// Returns the criterion under which this optimizer rejects the gradients of a mini-step, if any.
    #[inline]
    pub fn skip_criterion(&self) -> Option<&UpdateSkipCriterion> {
        self.skip_criterion.as_ref()
    }

    /// Returns the data type in which this optimizer stores the accumulated gradients, or [`None`] if it stores them in
    /// the statistic data type of each leaf.
    #[inline]
    pub fn accumulator_data_type(&self) -> Option<DataType> {
        self.accumulator_data_type
    }

    /// Returns whether the skip criterion of this optimizer rejects `gradients`, as a rank-0 boolean array, together
    /// with the not-finite count and the squared norm that the criterion records, or [`None`] when there is no
    /// criterion.
    #[allow(clippy::type_complexity)]
    fn evaluate_skip_criterion<P: OptimizerValue>(
        &self,
        gradients: &[P],
    ) -> Result<Option<(P, Vec<P>, Vec<P>)>, OptimizerError> {
        match &self.skip_criterion {
            None => Ok(None),
            Some(UpdateSkipCriterion::NotFinite) => {
                let mut count: Option<P> = None;
                for gradient in gradients {
                    let gradient = convert_to(gradient, statistic_data_type(gradient.r#type().data_type()))?;
                    let finite =
                        gradient.abs()?.compare(&gradient.fill_like(f64::INFINITY)?, ComparisonDirection::LessThan)?;
                    let leaf_count = convert_to(&Not::not(&finite)?, DataType::I32)?
                        .reduce(&all_axes(&gradient), ReductionKind::Sum)?;
                    count = Some(match count {
                        None => leaf_count,
                        Some(count) => Add::add(&count, &leaf_count)?,
                    });
                }
                let count = count.ok_or_else(|| OptimizerError::StructureMismatch {
                    message: "`MultiSteps` requires at least one gradient to evaluate its skip criterion".to_string(),
                })?;
                let should_skip = count.compare(&count.fill_like(0i32)?, ComparisonDirection::GreaterThan)?;
                Ok(Some((should_skip, vec![count], Vec::new())))
            }
            Some(UpdateSkipCriterion::LargeUpdates { maximum_squared_norm }) => {
                let squared_norm =
                    global_squared_norm(gradients)?.ok_or_else(|| OptimizerError::StructureMismatch {
                        message: "`MultiSteps` requires at least one gradient to evaluate its skip criterion"
                            .to_string(),
                    })?;
                let maximum = squared_norm.fill_like(*maximum_squared_norm)?;
                let should_skip = Not::not(&squared_norm.compare(&maximum, ComparisonDirection::LessThan)?)?;
                Ok(Some((should_skip, Vec::new(), vec![squared_norm])))
            }
        }
    }
}

impl<P: OptimizerValue, O: Optimizer<P>, Sc: Schedule> Optimizer<P> for MultiSteps<O, Sc> {
    type State<S: Parameterized<P>> = MultiStepsState<P, S, O::State<S>>;

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        // The record of the skip criterion is its decision on zero gradients, as in Optax.
        let zeros = filled_statistics(parameters, 0.0, None)?.into_parameters().collect::<Vec<_>>();
        let (should_skip, not_finite_count, squared_norm) = match self.evaluate_skip_criterion(&zeros)? {
            None => (Vec::new(), Vec::new(), Vec::new()),
            Some((should_skip, not_finite_count, squared_norm)) => (vec![should_skip], not_finite_count, squared_norm),
        };
        let mini_step = initial_count("MultiSteps", parameters)?;
        Ok(MultiStepsState {
            gradient_step: mini_step.clone(),
            mini_step,
            inner_state: self.inner.initialize(parameters)?,
            accumulated_gradients: filled_statistics(parameters, 0.0, self.accumulator_data_type)?,
            should_skip,
            not_finite_count,
            squared_norm,
        })
    }

    fn update<S: Parameterized<P>>(
        &self,
        gradients: S,
        state: Self::State<S>,
        parameters: &S,
        arguments: &UpdateArguments<'_, P, S>,
    ) -> Result<(S, Self::State<S>), OptimizerError> {
        // This follows `optax.MultiSteps.update`, except that both branches of its `lax.cond` are computed and
        // selected.
        let paths = gradients.parameter_paths().collect::<Vec<_>>();
        validate_structure("parameters", &paths, parameters)?;
        validate_structure("accumulated_gradients", &paths, &state.accumulated_gradients)?;
        let MultiStepsState {
            mini_step,
            gradient_step,
            inner_state,
            accumulated_gradients,
            squared_norm: previous_squared_norm,
            ..
        } = state;
        let every_k = convert_to(&self.every_k_schedule.value(&gradient_step)?, DataType::I32)?;
        let gradient_leaves = gradients.parameters().cloned().collect::<Vec<_>>();
        let skip = self.evaluate_skip_criterion(&gradient_leaves)?;

        // Accumulate the gradients of this mini-step and apply the wrapped optimizer to them.
        let structure = gradients.parameter_structure();
        let previous_accumulated_gradients = accumulated_gradients.parameters().cloned().collect::<Vec<_>>();
        let mut accumulated = Vec::with_capacity(paths.len());
        let mut inner_gradients = Vec::with_capacity(paths.len());
        for (gradient, accumulator) in gradients.into_parameters().zip(accumulated_gradients.into_parameters()) {
            let data_type = gradient.r#type().data_type();
            let statistic_type = statistic_data_type(data_type);
            let gradient = convert_to(&gradient, statistic_type)?;
            let accumulator = convert_to(&accumulator, statistic_type)?;
            let accumulator = if self.gradient_mean {
                let count = Add::add(&convert_to(&mini_step, statistic_type)?, &gradient.fill_like(1f64)?)?;
                Add::add(&accumulator, &Div::div(&Sub::sub(&gradient, &accumulator)?, &count)?)?
            } else {
                Add::add(&gradient, &accumulator)?
            };
            inner_gradients.push(convert_to(&accumulator, data_type)?);
            accumulated.push(accumulator);
        }
        let previous_inner_state = inner_state.parameters().cloned().collect::<Vec<_>>();
        let (inner_updates, next_inner_state) = self.inner.update(
            S::from_parameters(structure.clone(), inner_gradients)?,
            inner_state,
            parameters,
            arguments,
        )?;

        // Emit the updates and reset the accumulated gradients at the last mini-step of a gradient step.
        let emit = mini_step.compare(&Sub::sub(&every_k, &every_k.fill_like(1i32)?)?, ComparisonDirection::Equal)?;
        let next_mini_step = Rem::rem(&increment_count(&mini_step)?, &every_k)?;
        let next_gradient_step = emit.select(&increment_count(&gradient_step)?, &gradient_step)?;
        let next_inner_state = select_structure("inner_state", &emit, next_inner_state, previous_inner_state.clone())?;
        let next_accumulated = accumulated
            .iter()
            .zip(&previous_accumulated_gradients)
            .map(|(accumulator, previous)| {
                let keep =
                    Sub::sub(&accumulator.fill_like(1f64)?, &convert_to(&emit, accumulator.r#type().data_type())?)?;
                let data_type = self.accumulator_data_type.unwrap_or_else(|| previous.r#type().data_type());
                Ok(convert_to(&Mul::mul(&keep, accumulator)?, data_type)?)
            })
            .collect::<Result<Vec<_>, OptimizerError>>()?;
        let update_structure = inner_updates.parameter_structure();
        let updates = inner_updates
            .into_parameters()
            .map(|update| Ok(Mul::mul(&convert_to(&emit, update.r#type().data_type())?, &update)?))
            .collect::<Result<Vec<_>, OptimizerError>>()?;
        let updates = S::from_parameters(update_structure, updates)?;

        // Keep everything but the record of the skip criterion when the gradients of this mini-step are rejected.
        let Some((should_skip, not_finite_count, squared_norm)) = skip else {
            let state = MultiStepsState {
                mini_step: next_mini_step,
                gradient_step: next_gradient_step,
                inner_state: next_inner_state,
                accumulated_gradients: S::from_parameters(structure, next_accumulated)?,
                should_skip: Vec::new(),
                not_finite_count: Vec::new(),
                squared_norm: Vec::new(),
            };
            return Ok((updates, state));
        };
        let squared_norm = squared_norm
            .iter()
            .zip(&previous_squared_norm)
            .map(|(squared_norm, previous)| convert_to(squared_norm, previous.r#type().data_type()))
            .collect::<Result<Vec<_>, _>>()?;
        let apply = Not::not(&should_skip)?;
        let zeros = updates
            .parameters()
            .map(|update| Ok(update.fill_like(0f64)?))
            .collect::<Result<Vec<_>, OptimizerError>>()?;
        let accumulated_gradients = S::from_parameters(structure, next_accumulated)?;
        let state = MultiStepsState {
            mini_step: apply.select(&next_mini_step, &mini_step)?,
            gradient_step: apply.select(&next_gradient_step, &gradient_step)?,
            inner_state: select_structure("inner_state", &apply, next_inner_state, previous_inner_state)?,
            accumulated_gradients: select_structure(
                "accumulated_gradients",
                &apply,
                accumulated_gradients,
                previous_accumulated_gradients,
            )?,
            should_skip: vec![should_skip],
            not_finite_count,
            squared_norm,
        };
        Ok((select_structure("updates", &apply, updates, zeros)?, state))
    }
}

parameterized_state! {
    /// State of [`ApplyEvery`], which is the analogue of the `ApplyEvery` state of
    /// [`optax.apply_every`](https://optax.readthedocs.io/en/latest/api/transformations.html#optax.apply_every). It
    /// holds the step count modulo the number of steps per emission and the accumulated updates, with the structure `S`
    /// of the parameters.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct ApplyEveryState<P, S> with family ApplyEveryStateFamily {
        /// Rank-0 `i32` array that holds the number of updates since the last emission.
        count: P,

        /// Updates accumulated since the last emission, with the structure of the parameters.
        accumulated_updates: S,
    }
}

impl<P: Parameter, S> ApplyEveryState<P, S> {
    /// Returns the rank-0 `i32` array that holds the number of updates since the last emission.
    #[inline]
    pub fn count(&self) -> &P {
        &self.count
    }

    /// Returns the updates accumulated since the last emission, with the structure of the parameters.
    #[inline]
    pub fn accumulated_updates(&self) -> &S {
        &self.accumulated_updates
    }
}

/// Transformation that accumulates the updates and emits their sum every `k` steps, returning zero updates at all
/// other steps, which is the analogue of
/// [`optax.apply_every`](https://optax.readthedocs.io/en/latest/api/transformations.html#optax.apply_every). Unlike
/// [`MultiSteps`], it does not hold back the transformations that precede it in a chain, which still update their
/// states at every step.
#[derive(Clone, Debug, PartialEq)]
pub struct ApplyEvery {
    /// Number of steps per emission.
    every_k: usize,
}

impl ApplyEvery {
    /// Creates a new [`ApplyEvery`] that emits the sum of the updates of every `every_k` consecutive steps.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `every_k` is zero or greater than `i32::MAX`.
    pub fn new(every_k: usize) -> Result<Self, OptimizerError> {
        if every_k == 0 || i32::try_from(every_k).is_err() {
            return Err(OptimizerError::InvalidConfiguration {
                message: format!("`every_k` must be in `[1, {}]` but got `{every_k}`", i32::MAX),
            });
        }
        Ok(Self { every_k })
    }

    /// Returns the number of steps per emission of this transformation.
    #[inline]
    pub fn every_k(&self) -> usize {
        self.every_k
    }
}

impl<P: OptimizerValue> Optimizer<P> for ApplyEvery {
    type State<S: Parameterized<P>> = ApplyEveryState<P, S>;

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        Ok(ApplyEveryState {
            count: initial_count("ApplyEvery", parameters)?,
            accumulated_updates: filled_statistics(parameters, 0.0, None)?,
        })
    }

    fn update<S: Parameterized<P>>(
        &self,
        gradients: S,
        state: Self::State<S>,
        parameters: &S,
        arguments: &UpdateArguments<'_, P, S>,
    ) -> Result<(S, Self::State<S>), OptimizerError> {
        // This follows the operation order of `optax.apply_every`.
        let _ = arguments;
        let paths = gradients.parameter_paths().collect::<Vec<_>>();
        validate_structure("parameters", &paths, parameters)?;
        validate_structure("accumulated_updates", &paths, &state.accumulated_updates)?;
        let every_k = state.count.fill_like(self.every_k as i32)?;
        let count = Rem::rem(&state.count, &every_k)?;
        let accumulating = Not::not(&count.compare(&count.fill_like(0i32)?, ComparisonDirection::Equal)?)?;
        let emit = count.compare(&count.fill_like(self.every_k as i32 - 1)?, ComparisonDirection::Equal)?;
        let structure = gradients.parameter_structure();
        let mut updates = Vec::with_capacity(paths.len());
        let mut accumulated = Vec::with_capacity(paths.len());
        for (gradient, accumulator) in gradients.into_parameters().zip(state.accumulated_updates.into_parameters()) {
            let data_type = gradient.r#type().data_type();
            let accumulator_type = accumulator.r#type().data_type();
            let statistic_type = statistic_data_type(data_type);
            let gradient = convert_to(&gradient, statistic_type)?;
            let accumulator = Add::add(
                &Mul::mul(&convert_to(&accumulating, statistic_type)?, &convert_to(&accumulator, statistic_type)?)?,
                &gradient,
            )?;
            updates.push(convert_to(&Mul::mul(&convert_to(&emit, statistic_type)?, &accumulator)?, data_type)?);
            accumulated.push(convert_to(&accumulator, accumulator_type)?);
        }
        let state = ApplyEveryState {
            count: Rem::rem(&increment_count(&state.count)?, &every_k)?,
            accumulated_updates: S::from_parameters(structure.clone(), accumulated)?,
        };
        Ok((S::from_parameters(structure, updates)?, state))
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use ryft_core::Array;

    use crate::tests::{ArgumentProbe, assert_parity, parity_updates, state_data_types, state_paths, vector_updates};
    use crate::{Adam, Chain, LinearSchedule, Scale, ScaleBySchedule, Sgd};

    use super::*;

    #[test]
    fn test_multi_steps_state_has_updated() {
        let optimizer = MultiSteps::new(Scale::new(1.0).unwrap(), 2).unwrap();
        let parameters = vec![Array::scalar(0.0f32).unwrap()];
        let arguments = UpdateArguments::new();
        let state = optimizer.initialize(&parameters).unwrap();
        assert_eq!(state.has_updated(), Ok(Array::scalar(false).unwrap()));
        let (_, state) = optimizer.update(parameters.clone(), state, &parameters, &arguments).unwrap();
        assert_eq!(state.has_updated(), Ok(Array::scalar(false).unwrap()));
        let (_, state) = optimizer.update(parameters.clone(), state, &parameters, &arguments).unwrap();
        assert_eq!(state.has_updated(), Ok(Array::scalar(true).unwrap()));
    }

    #[test]
    fn test_multi_steps_new() {
        let optimizer = MultiSteps::new(Scale::new(2.0).unwrap(), 3).unwrap();
        assert_eq!(optimizer.inner(), &Scale::new(2.0).unwrap());
        assert_eq!(optimizer.every_k_schedule(), &3.0);
        assert!(optimizer.gradient_mean());
        assert_eq!(optimizer.skip_criterion(), None);
        assert_eq!(optimizer.accumulator_data_type(), None);
        assert!(matches!(
            MultiSteps::new(Scale::new(2.0).unwrap(), 0),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`every_k` must be in `[1, 2147483647]` but got `0`",
        ));
    }

    #[test]
    fn test_multi_steps_with_every_k_schedule() {
        // The schedule is `1` at gradient step `0` and `2` afterwards, so the first gradient step takes one mini-step
        // and the second one takes two, with the learning rate `0.5`: `-0.5 · 2` and then `-0.5 · (2 + 4) / 2`.
        let schedule = LinearSchedule::new(1.0, 2.0, 1).unwrap();
        let optimizer = MultiSteps::new(ScaleBySchedule::learning_rate(0.5f64), 1)
            .unwrap()
            .with_every_k_schedule(schedule.clone());
        assert_eq!(optimizer.every_k_schedule(), &schedule);
        assert_eq!(
            vector_updates(&optimizer, &[0.0], &[&[2.0], &[2.0], &[4.0]]),
            vec![vec![-1.0], vec![0.0], vec![-1.5]],
        );
    }

    #[test]
    fn test_multi_steps_with_gradient_mean() {
        // Summed gradients: `-0.5 · (2 + 4)`.
        let optimizer = MultiSteps::new(ScaleBySchedule::learning_rate(0.5f64), 2).unwrap().with_gradient_mean(false);
        assert!(!optimizer.gradient_mean());
        assert_eq!(vector_updates(&optimizer, &[0.0], &[&[2.0], &[4.0]]), vec![vec![0.0], vec![-3.0]]);
    }

    #[test]
    fn test_multi_steps_with_skip_criterion() {
        let optimizer = MultiSteps::new(Scale::new(1.0).unwrap(), 2)
            .unwrap()
            .with_skip_criterion(Some(UpdateSkipCriterion::NotFinite))
            .unwrap();
        assert_eq!(optimizer.skip_criterion(), Some(&UpdateSkipCriterion::NotFinite));
        assert!(matches!(
            optimizer.with_skip_criterion(Some(UpdateSkipCriterion::LargeUpdates { maximum_squared_norm: -1.0 })),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`maximum_squared_norm` must be finite and non-negative but got `-1`",
        ));
    }

    #[test]
    fn test_multi_steps_with_accumulator_data_type() {
        let optimizer = MultiSteps::new(Scale::new(1.0).unwrap(), 2)
            .unwrap()
            .with_accumulator_data_type(DataType::F64)
            .unwrap();
        assert_eq!(optimizer.accumulator_data_type(), Some(DataType::F64));
        let state = optimizer.initialize(&vec![Array::scalar(0.0f32).unwrap()]).unwrap();
        assert_eq!(state.accumulated_gradients(), &vec![Array::scalar(0.0f64).unwrap()]);
    }

    #[test]
    fn test_multi_steps_initialize() {
        let optimizer = MultiSteps::new(ScaleBySchedule::new(1.0f64), 2).unwrap();
        assert_eq!(
            state_paths(&optimizer),
            vec![
                "$.mini_step",
                "$.gradient_step",
                "$.inner_state.count",
                "$.accumulated_gradients[\"bias\"]",
                "$.accumulated_gradients[\"weights\"]",
            ],
        );
        let optimizer = optimizer.with_skip_criterion(Some(UpdateSkipCriterion::NotFinite)).unwrap();
        assert_eq!(state_paths(&optimizer)[5..], ["$.should_skip[0]", "$.not_finite_count[0]"]);
        let optimizer = optimizer
            .with_skip_criterion(Some(UpdateSkipCriterion::LargeUpdates { maximum_squared_norm: 1.0 }))
            .unwrap();
        assert_eq!(state_paths(&optimizer)[5..], ["$.should_skip[0]", "$.squared_norm[0]"]);
    }

    #[test]
    fn test_multi_steps_update() {
        // With `k = 2` and the learning rate `0.5`, the mini-steps with the gradients `2` and `4` accumulate the
        // averages `0 + (2 - 0) / 1 = 2` and `2 + (4 - 2) / 2 = 3`, so the updates are `0` and `-0.5 · 3`. The
        // accumulator is then reset, so the third mini-step starts over with the average `1`.
        let optimizer = MultiSteps::new(ScaleBySchedule::learning_rate(0.5f64), 2).unwrap();
        assert_eq!(
            vector_updates(&optimizer, &[0.0], &[&[2.0], &[4.0], &[1.0], &[3.0]]),
            vec![vec![0.0], vec![-1.5], vec![0.0], vec![-1.0]],
        );
        let parameters = vec![Array::vector(vec![0.0f32]).unwrap()];
        let arguments = UpdateArguments::new();
        let state = optimizer.initialize(&parameters).unwrap();
        let (_, state) = optimizer
            .update(vec![Array::vector(vec![2.0f32]).unwrap()], state, &parameters, &arguments)
            .unwrap();
        assert_eq!(state.mini_step(), &Array::scalar(1i32).unwrap());
        assert_eq!(state.gradient_step(), &Array::scalar(0i32).unwrap());
        assert_eq!(state.inner_state().count(), &Array::scalar(0i32).unwrap());
        assert_eq!(state.accumulated_gradients(), &vec![Array::vector(vec![2.0f32]).unwrap()]);
        let (_, state) = optimizer
            .update(vec![Array::vector(vec![4.0f32]).unwrap()], state, &parameters, &arguments)
            .unwrap();
        assert_eq!(state.mini_step(), &Array::scalar(0i32).unwrap());
        assert_eq!(state.gradient_step(), &Array::scalar(1i32).unwrap());
        assert_eq!(state.inner_state().count(), &Array::scalar(1i32).unwrap());
        assert_eq!(state.accumulated_gradients(), &vec![Array::vector(vec![0.0f32]).unwrap()]);
    }

    #[test]
    fn test_multi_steps_update_skip_not_finite() {
        // The mini-step with a NaN is rejected, so the accumulation continues with the next one: `(2 + 4) / 2 = 3`.
        let optimizer = MultiSteps::new(ScaleBySchedule::learning_rate(0.5f64), 2)
            .unwrap()
            .with_skip_criterion(Some(UpdateSkipCriterion::NotFinite))
            .unwrap();
        let parameters = vec![Array::vector(vec![0.0f32, 0.0]).unwrap()];
        let arguments = UpdateArguments::new();
        let state = optimizer.initialize(&parameters).unwrap();
        assert_eq!(state.should_skip(), Some(&Array::scalar(false).unwrap()));
        assert_eq!(state.not_finite_count(), Some(&Array::scalar(0i32).unwrap()));
        let gradients = vec![Array::vector(vec![2.0f32, 2.0]).unwrap()];
        let (updates, state) = optimizer.update(gradients, state, &parameters, &arguments).unwrap();
        assert_eq!(updates, vec![Array::vector(vec![0.0f32, 0.0]).unwrap()]);
        let gradients = vec![Array::vector(vec![f32::NAN, f32::INFINITY]).unwrap()];
        let (updates, state) = optimizer.update(gradients, state, &parameters, &arguments).unwrap();
        assert_eq!(updates, vec![Array::vector(vec![0.0f32, 0.0]).unwrap()]);
        assert_eq!(state.should_skip(), Some(&Array::scalar(true).unwrap()));
        assert_eq!(state.not_finite_count(), Some(&Array::scalar(2i32).unwrap()));
        assert_eq!(state.mini_step(), &Array::scalar(1i32).unwrap());
        assert_eq!(state.accumulated_gradients(), &vec![Array::vector(vec![2.0f32, 2.0]).unwrap()]);
        let gradients = vec![Array::vector(vec![4.0f32, 4.0]).unwrap()];
        let (updates, state) = optimizer.update(gradients, state, &parameters, &arguments).unwrap();
        assert_eq!(updates, vec![Array::vector(vec![-1.5f32, -1.5]).unwrap()]);
        assert_eq!(state.should_skip(), Some(&Array::scalar(false).unwrap()));
    }

    #[test]
    fn test_multi_steps_update_skip_large_updates() {
        // The squared norm `3² + 4² = 25` is not less than `25`, so the mini-step is rejected, while `1² = 1` is less.
        let optimizer = MultiSteps::new(Scale::new(1.0).unwrap(), 1)
            .unwrap()
            .with_skip_criterion(Some(UpdateSkipCriterion::LargeUpdates { maximum_squared_norm: 25.0 }))
            .unwrap();
        let parameters = vec![Array::vector(vec![0.0f32, 0.0]).unwrap()];
        let arguments = UpdateArguments::new();
        let state = optimizer.initialize(&parameters).unwrap();
        assert_eq!(state.squared_norm(), Some(&Array::scalar(0.0f32).unwrap()));
        let gradients = vec![Array::vector(vec![3.0f32, 4.0]).unwrap()];
        let (updates, state) = optimizer.update(gradients, state, &parameters, &arguments).unwrap();
        assert_eq!(updates, vec![Array::vector(vec![0.0f32, 0.0]).unwrap()]);
        assert_eq!(state.squared_norm(), Some(&Array::scalar(25.0f32).unwrap()));
        assert_eq!(state.gradient_step(), &Array::scalar(0i32).unwrap());
        let gradients = vec![Array::vector(vec![1.0f32, 0.0]).unwrap()];
        let (updates, state) = optimizer.update(gradients, state, &parameters, &arguments).unwrap();
        assert_eq!(updates, vec![Array::vector(vec![1.0f32, 0.0]).unwrap()]);
        assert_eq!(state.gradient_step(), &Array::scalar(1i32).unwrap());
    }

    #[test]
    fn test_multi_steps_update_arguments() {
        // The wrapped optimizer receives the arguments: `10 + 0.5 + 1 = 11.5`.
        let optimizer = MultiSteps::new(ArgumentProbe, 1).unwrap();
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
        assert_eq!(updates, vec![Array::scalar(11.5f32).unwrap()]);
    }

    #[test]
    fn test_apply_every_new() {
        assert_eq!(ApplyEvery::new(3).unwrap().every_k(), 3);
        assert!(matches!(
            ApplyEvery::new(0),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`every_k` must be in `[1, 2147483647]` but got `0`",
        ));
    }

    #[test]
    fn test_apply_every_initialize() {
        assert_eq!(
            state_paths(&ApplyEvery::new(2).unwrap()),
            vec!["$.count", "$.accumulated_updates[\"bias\"]", "$.accumulated_updates[\"weights\"]"],
        );
    }

    #[test]
    fn test_apply_every_update() {
        // With `k = 2`, the updates `1` and `2` are emitted as `0` and `1 + 2 = 3`, and the third update `4` starts a
        // new sum.
        let optimizer = ApplyEvery::new(2).unwrap();
        assert_eq!(
            vector_updates(&optimizer, &[0.0], &[&[1.0], &[2.0], &[4.0], &[8.0]]),
            vec![vec![0.0], vec![3.0], vec![0.0], vec![12.0]],
        );

        // `f64` parameters with `f32` gradients keep the `f64` accumulator that `initialize` creates.
        let (initial_data_types, updated_data_types) = state_data_types(&optimizer);
        assert_eq!(updated_data_types, initial_data_types);
    }

    #[test]
    fn test_multi_steps_optax_parity() {
        // Optax 9ecde04b: `optax.MultiSteps(optax.adam(0.1), every_k_schedule=2)`.
        const MEAN: [[f32; 9]; 4] = [
            [-0.0, 0.0, -0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0],
            [
                -0.09999929,
                -0.09999929,
                -0.09999933,
                -0.09999933,
                -0.09999933,
                0.09999932,
                0.09999932,
                -0.09999932,
                0.09999932,
            ],
            [-0.0, 0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0],
            [
                -0.08070796,
                -0.09999895,
                0.020793196,
                -0.060590733,
                0.016273065,
                -0.044221073,
                0.09999897,
                -0.09651723,
                0.0058124424,
            ],
        ];
        assert_parity(&parity_updates(&MultiSteps::new(Adam::new(0.1), 2).unwrap()), &MEAN);

        // Optax 9ecde04b: `optax.MultiSteps(optax.sgd(0.1, momentum=0.9), every_k_schedule=3, use_grad_mean=False)`.
        const SUM: [[f32; 9]; 4] = [
            [-0.0, 0.0, -0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0],
            [-0.0, -0.0, -0.0, -0.0, -0.0, 0.0, 0.0, -0.0, 0.0],
            [
                -0.035000004,
                0.0050000004,
                -0.020000001,
                -0.045,
                -0.025000004,
                -0.024999999,
                -0.0049999985,
                -0.07000001,
                0.035000004,
            ],
            [-0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0],
        ];
        let optimizer =
            MultiSteps::new(Sgd::new(0.1).with_momentum(0.9).unwrap(), 3).unwrap().with_gradient_mean(false);
        assert_parity(&parity_updates(&optimizer), &SUM);

        // Optax 9ecde04b: `optax.MultiSteps(optax.adam(0.1), every_k_schedule=2,
        // should_skip_update_fn=functools.partial(optax.skip_large_updates, max_squared_norm=1.0))`. The squared norm
        // of
        // the first gradients is `1.05`, so they are skipped and the first update is made after the third step.
        const SKIP: [[f32; 9]; 4] = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, 0.0, -0.0],
            [
                -0.09999932,
                -0.09999932,
                0.0999993,
                -0.09999929,
                -0.09999933,
                -0.09999929,
                -0.09999933,
                -0.09999932,
                -0.09999932,
            ],
            [-0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, -0.0],
        ];
        let optimizer = MultiSteps::new(Adam::new(0.1), 2)
            .unwrap()
            .with_skip_criterion(Some(UpdateSkipCriterion::LargeUpdates { maximum_squared_norm: 1.0 }))
            .unwrap();
        assert_parity(&parity_updates(&optimizer), &SKIP);
    }

    #[test]
    fn test_apply_every_optax_parity() {
        // Optax 9ecde04b: `optax.chain(optax.sgd(0.1), optax.apply_every(3))`.
        const EXPECTED: [[f32; 9]; 4] = [
            [-0.0, 0.0, -0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0],
            [-0.0, -0.0, -0.0, -0.0, -0.0, 0.0, 0.0, -0.0, 0.0],
            [-0.035, 0.0050000018, -0.020000001, -0.045, -0.025, -0.024999997, -0.005000001, -0.07, 0.035000004],
            [-0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, -0.0],
        ];
        assert_parity(&parity_updates(&Chain::new(Sgd::new(0.1), ApplyEvery::new(3).unwrap())), &EXPECTED);
    }
}
