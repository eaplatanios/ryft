//! Wrappers that apply another optimizer only when a condition holds.
//!
//! [`ApplyIfFinite`] skips the updates of another optimizer whose gradients are not finite, and
//! [`ConditionallyTransform`] and [`ConditionallyMask`] apply another optimizer only at the steps that a
//! [`StepCondition`] selects, passing the updates through unchanged or replacing them with zeros at all other steps.

use ryft_core::{ComparisonDirection, DataType, Parameter, Parameterized, ReductionKind};

use crate::macros::parameterized_state;
use crate::{
    Optimizer, OptimizerError, OptimizerValue, UpdateArguments, all_axes, convert_to, increment_count, initial_count,
    statistic_data_type, validate_leaf_count,
};

use super::select_structure;

// TODO(eaplatanios): Review this module.

parameterized_state! {
    /// State of [`ApplyIfFinite`], which is the analogue of
    /// [`optax.ApplyIfFiniteState`](
    /// https://optax.readthedocs.io/en/latest/api/optimizer_wrappers.html#optax.ApplyIfFiniteState).
    /// It holds counts of the updates whose gradients were not finite, whether the gradients of the last update were
    /// finite, and the state `S` of the wrapped optimizer, whose leaf paths start with `$.inner_state`.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct ApplyIfFiniteState<P, S> with family ApplyIfFiniteStateFamily {
        /// Rank-0 `i32` array that holds the number of consecutive updates whose gradients were not finite, up to and
        /// including the last one, which is the `notfinite_count` of Optax.
        not_finite_count: P,

        /// Rank-0 boolean array that holds whether the gradients of the last update were finite, which is `true`
        /// before the first update.
        last_finite: P,

        /// Rank-0 `i32` array that holds the total number of updates whose gradients were not finite, which is the
        /// `total_notfinite` of Optax.
        total_not_finite_count: P,

        /// State of the wrapped optimizer.
        inner_state: S,
    }
}

impl<P: Parameter, S> ApplyIfFiniteState<P, S> {
    /// Returns the rank-0 `i32` array that holds the number of consecutive updates whose gradients were not finite, up
    /// to and including the last one.
    #[inline]
    pub fn not_finite_count(&self) -> &P {
        &self.not_finite_count
    }

    /// Returns the rank-0 boolean array that holds whether the gradients of the last update were finite.
    #[inline]
    pub fn last_finite(&self) -> &P {
        &self.last_finite
    }

    /// Returns the rank-0 `i32` array that holds the total number of updates whose gradients were not finite.
    #[inline]
    pub fn total_not_finite_count(&self) -> &P {
        &self.total_not_finite_count
    }

    /// Returns the state of the wrapped optimizer.
    #[inline]
    pub fn inner_state(&self) -> &S {
        &self.inner_state
    }
}

/// Optimizer that applies another optimizer only when all gradients are finite, which is the analogue of
/// [`optax.apply_if_finite`](https://optax.readthedocs.io/en/latest/api/optimizer_wrappers.html#optax.apply_if_finite).
/// When any element of any gradient leaf is NaN or infinite, the updates are all zero and the state of the inner
/// optimizer is returned unchanged, unless more than the maximum number of consecutive updates (i.e., the
/// `max_consecutive_errors` of Optax) had gradients that were not finite, in which case the inner optimizer is applied
/// regardless. Otherwise, the updates and the state of the inner optimizer are returned. The state counts the
/// consecutive and the total number of updates whose gradients were not finite, as described in
/// [`ApplyIfFiniteState`].
///
/// The decision is made with array operations (i.e., the inner update is always computed and its results are
/// selected), so it is staged rather than taken on the host inside compiled training steps. This is useful with mixed
/// precision training, where an occasional overflowing gradient should skip a step instead of corrupting the optimizer
/// state.
#[derive(Clone, Debug, PartialEq)]
pub struct ApplyIfFinite<O> {
    /// Optimizer that is applied when all gradients are finite.
    inner: O,

    /// Number of consecutive updates whose gradients are not finite that are skipped before the inner optimizer is
    /// applied regardless, which is at most `i32::MAX`.
    maximum_consecutive_errors: usize,
}

impl<O> ApplyIfFinite<O> {
    /// Creates a new [`ApplyIfFinite`] that applies `inner` when all gradients are finite, or when more than
    /// `maximum_consecutive_errors` consecutive updates (including the current one) had gradients that were not
    /// finite.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `maximum_consecutive_errors` is greater than `i32::MAX`,
    /// which the counts of the state cannot exceed.
    pub fn new(inner: O, maximum_consecutive_errors: usize) -> Result<Self, OptimizerError> {
        if i32::try_from(maximum_consecutive_errors).is_err() {
            return Err(OptimizerError::InvalidConfiguration {
                message: format!(
                    "`maximum_consecutive_errors` must be at most `{}` but got `{maximum_consecutive_errors}`",
                    i32::MAX,
                ),
            });
        }
        Ok(Self { inner, maximum_consecutive_errors })
    }

    /// Returns the optimizer that this optimizer applies when all gradients are finite.
    #[inline]
    pub fn inner(&self) -> &O {
        &self.inner
    }

    /// Returns the number of consecutive updates whose gradients are not finite that this optimizer skips before it
    /// applies its inner optimizer regardless.
    #[inline]
    pub fn maximum_consecutive_errors(&self) -> usize {
        self.maximum_consecutive_errors
    }
}

impl<P: OptimizerValue, O: Optimizer<P>> Optimizer<P> for ApplyIfFinite<O> {
    type State<S: Parameterized<P>> = ApplyIfFiniteState<P, O::State<S>>;

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        // The initial count is zero, so comparing it with itself creates the initial rank-0 boolean `true`.
        let not_finite_count = initial_count("ApplyIfFinite", parameters)?;
        let last_finite = not_finite_count.compare(&not_finite_count, ComparisonDirection::Equal)?;
        Ok(ApplyIfFiniteState {
            total_not_finite_count: not_finite_count.clone(),
            not_finite_count,
            last_finite,
            inner_state: self.inner.initialize(parameters)?,
        })
    }

    fn update<S: Parameterized<P>>(
        &self,
        gradients: S,
        state: Self::State<S>,
        parameters: &S,
        arguments: &UpdateArguments<'_, P, S>,
    ) -> Result<(S, Self::State<S>), OptimizerError> {
        // This follows `optax.apply_if_finite`, except that the inner update is always computed and selected. An
        // element is finite when its magnitude is less than infinity, which is false for infinities and NaNs.
        let ApplyIfFiniteState { not_finite_count, last_finite, total_not_finite_count, inner_state } = state;
        let mut finite = last_finite.fill_like(true)?;
        for gradient in gradients.parameters() {
            let gradient = convert_to(gradient, statistic_data_type(gradient.r#type().data_type()))?;
            let leaf_finite = gradient
                .abs()?
                .compare(&gradient.fill_like(f64::INFINITY)?, ComparisonDirection::LessThan)?
                .reduce(&all_axes(&gradient), ReductionKind::All)?;
            finite = finite.and(&leaf_finite)?;
        }
        let not_finite_count =
            finite.select(&not_finite_count.fill_like(0i32)?, &increment_count(&not_finite_count)?)?;
        let total_not_finite_count =
            finite.select(&total_not_finite_count, &increment_count(&total_not_finite_count)?)?;
        let maximum_consecutive_errors = not_finite_count.fill_like(self.maximum_consecutive_errors as i32)?;
        let apply =
            finite.or(&not_finite_count.compare(&maximum_consecutive_errors, ComparisonDirection::GreaterThan)?)?;

        let previous_inner_state = inner_state.parameters().cloned().collect::<Vec<_>>();
        let (updates, next_inner_state) = self.inner.update(gradients, inner_state, parameters, arguments)?;
        validate_leaf_count("inner_state", previous_inner_state.len(), next_inner_state.parameter_count())?;
        let update_structure = updates.parameter_structure();
        let updates = updates
            .into_parameters()
            .map(|update| Ok(apply.select(&update, &update.fill_like(0f64)?)?))
            .collect::<Result<Vec<_>, OptimizerError>>()?;
        let inner_state_structure = next_inner_state.parameter_structure();
        let inner_state = next_inner_state
            .into_parameters()
            .zip(previous_inner_state)
            .map(|(next, previous)| Ok(apply.select(&convert_to(&next, previous.r#type().data_type())?, &previous)?))
            .collect::<Result<Vec<_>, OptimizerError>>()?;
        let state = ApplyIfFiniteState {
            not_finite_count,
            last_finite: finite,
            total_not_finite_count,
            inner_state: O::State::<S>::from_parameters(inner_state_structure, inner_state)?,
        };
        Ok((S::from_parameters(update_structure, updates)?, state))
    }
}

/// Condition on the step count under which [`ConditionallyTransform`] and [`ConditionallyMask`] apply the optimizer
/// that they wrap, which is the analogue of the
/// [`optax.ConditionFn`](https://optax.readthedocs.io/en/latest/api/optimizer_wrappers.html#optax.ConditionFn)
/// protocol. Conditions are evaluated with array operations, so that they are staged inside compiled training steps
/// rather than read back to the host, and so they are implemented for types (e.g., a unit struct per condition) rather
/// than for closures, whose argument types could not vary with the leaf type `P`.
pub trait StepCondition {
    /// Returns whether the wrapped optimizer is applied at `step`, as a rank-0 boolean array.
    ///
    /// # Parameters
    ///
    ///   - `step`: Rank-0 `i32` array that holds the number of updates applied so far, which is `0` for the first
    ///     update.
    ///
    /// # Errors
    ///
    /// Returns an [`OptimizerError`] when one of the array operations that compute the condition fails.
    fn evaluate<P: OptimizerValue>(&self, step: &P) -> Result<P, OptimizerError>;
}

impl<C: StepCondition> StepCondition for &C {
    fn evaluate<P: OptimizerValue>(&self, step: &P) -> Result<P, OptimizerError> {
        (*self).evaluate(step)
    }
}

parameterized_state! {
    /// State of [`ConditionallyTransform`], which is the analogue of
    /// [`optax.ConditionallyTransformState`](
    /// https://optax.readthedocs.io/en/latest/api/optimizer_wrappers.html#optax.ConditionallyTransformState).
    /// It holds the state `S` of the wrapped optimizer, whose leaf paths start with `$.inner_state`, and the step
    /// count.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct ConditionallyTransformState<P, S> with family ConditionallyTransformStateFamily {
        /// State of the wrapped optimizer, which is updated only at the steps at which the condition holds.
        inner_state: S,

        /// Rank-0 `i32` array that holds the number of updates applied so far, whether or not the condition held.
        step: P,
    }
}

impl<P: Parameter, S> ConditionallyTransformState<P, S> {
    /// Returns the state of the wrapped optimizer.
    #[inline]
    pub fn inner_state(&self) -> &S {
        &self.inner_state
    }

    /// Returns the rank-0 `i32` array that holds the number of updates applied so far.
    #[inline]
    pub fn step(&self) -> &P {
        &self.step
    }
}

/// Optimizer that applies another optimizer only at the steps at which a [`StepCondition`] holds and passes the
/// updates through unchanged at all other steps, which is the analogue of
/// [`optax.conditionally_transform`](
/// https://optax.readthedocs.io/en/latest/api/optimizer_wrappers.html#optax.conditionally_transform).
/// The state of the wrapped optimizer is updated only at the steps at which the condition holds, and the step count
/// increases at every step. The condition sees the step count only (i.e., `forward_extra_args=False` in Optax).
///
/// The inner update is always computed and its results are selected, rather than skipped with `lax.cond`, so that the
/// condition is staged inside compiled training steps.
#[derive(Clone, Debug, PartialEq)]
pub struct ConditionallyTransform<O, C> {
    /// Optimizer that is applied at the steps at which the condition holds.
    inner: O,

    /// Condition on the step count under which the wrapped optimizer is applied.
    condition: C,
}

impl<O, C: StepCondition> ConditionallyTransform<O, C> {
    /// Creates a new [`ConditionallyTransform`] that applies `inner` at the steps at which `condition` holds.
    #[inline]
    pub fn new(inner: O, condition: C) -> Self {
        Self { inner, condition }
    }

    /// Returns the optimizer that this optimizer applies at the steps at which its condition holds.
    #[inline]
    pub fn inner(&self) -> &O {
        &self.inner
    }

    /// Returns the condition on the step count under which this optimizer applies its wrapped optimizer.
    #[inline]
    pub fn condition(&self) -> &C {
        &self.condition
    }
}

impl<P: OptimizerValue, O: Optimizer<P>, C: StepCondition> Optimizer<P> for ConditionallyTransform<O, C> {
    type State<S: Parameterized<P>> = ConditionallyTransformState<P, O::State<S>>;

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        let step = initial_count("ConditionallyTransform", parameters)?;
        Ok(ConditionallyTransformState { inner_state: self.inner.initialize(parameters)?, step })
    }

    fn update<S: Parameterized<P>>(
        &self,
        gradients: S,
        state: Self::State<S>,
        parameters: &S,
        arguments: &UpdateArguments<'_, P, S>,
    ) -> Result<(S, Self::State<S>), OptimizerError> {
        // This follows `optax.conditionally_transform`, except that the inner update is always computed and selected.
        let ConditionallyTransformState { inner_state, step } = state;
        let condition = evaluate_condition(&self.condition, &step)?;
        let previous_updates = gradients.parameters().cloned().collect::<Vec<_>>();
        let previous_inner_state = inner_state.parameters().cloned().collect::<Vec<_>>();
        let (updates, inner_state) = self.inner.update(gradients, inner_state, parameters, arguments)?;
        let updates = select_structure("updates", &condition, updates, previous_updates)?;
        let inner_state = select_structure("inner_state", &condition, inner_state, previous_inner_state)?;
        Ok((updates, ConditionallyTransformState { inner_state, step: increment_count(&step)? }))
    }
}

parameterized_state! {
    /// State of [`ConditionallyMask`], which is the analogue of
    /// [`optax.ConditionallyMaskState`](
    /// https://optax.readthedocs.io/en/latest/api/optimizer_wrappers.html#optax.ConditionallyMaskState).
    /// It holds the step count and the state `S` of the wrapped optimizer, whose leaf paths start with `$.inner_state`.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct ConditionallyMaskState<P, S> with family ConditionallyMaskStateFamily {
        /// Rank-0 `i32` array that holds the number of updates applied so far, whether or not the condition held.
        step: P,

        /// State of the wrapped optimizer, which is updated only at the steps at which the condition holds.
        inner_state: S,
    }
}

impl<P: Parameter, S> ConditionallyMaskState<P, S> {
    /// Returns the rank-0 `i32` array that holds the number of updates applied so far.
    #[inline]
    pub fn step(&self) -> &P {
        &self.step
    }

    /// Returns the state of the wrapped optimizer.
    #[inline]
    pub fn inner_state(&self) -> &S {
        &self.inner_state
    }
}

/// Optimizer that applies another optimizer only at the steps at which a [`StepCondition`] holds and returns zero
/// updates at all other steps, which is the analogue of [`optax.conditionally_mask`](
/// https://optax.readthedocs.io/en/latest/api/optimizer_wrappers.html#optax.conditionally_mask). It differs from
/// [`ConditionallyTransform`] only in the updates that it returns when the condition does not hold.
#[derive(Clone, Debug, PartialEq)]
pub struct ConditionallyMask<O, C> {
    /// Optimizer that is applied at the steps at which the condition holds.
    inner: O,

    /// Condition on the step count under which the wrapped optimizer is applied.
    condition: C,
}

impl<O, C: StepCondition> ConditionallyMask<O, C> {
    /// Creates a new [`ConditionallyMask`] that applies `inner` at the steps at which `condition` holds.
    #[inline]
    pub fn new(inner: O, condition: C) -> Self {
        Self { inner, condition }
    }

    /// Returns the optimizer that this optimizer applies at the steps at which its condition holds.
    #[inline]
    pub fn inner(&self) -> &O {
        &self.inner
    }

    /// Returns the condition on the step count under which this optimizer applies its wrapped optimizer.
    #[inline]
    pub fn condition(&self) -> &C {
        &self.condition
    }
}

impl<P: OptimizerValue, O: Optimizer<P>, C: StepCondition> Optimizer<P> for ConditionallyMask<O, C> {
    type State<S: Parameterized<P>> = ConditionallyMaskState<P, O::State<S>>;

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        let step = initial_count("ConditionallyMask", parameters)?;
        Ok(ConditionallyMaskState { step, inner_state: self.inner.initialize(parameters)? })
    }

    fn update<S: Parameterized<P>>(
        &self,
        gradients: S,
        state: Self::State<S>,
        parameters: &S,
        arguments: &UpdateArguments<'_, P, S>,
    ) -> Result<(S, Self::State<S>), OptimizerError> {
        // This follows `optax.conditionally_mask`, except that the inner update is always computed and selected.
        let ConditionallyMaskState { step, inner_state } = state;
        let condition = evaluate_condition(&self.condition, &step)?;
        let zeros = gradients
            .parameters()
            .map(|gradient| Ok(gradient.fill_like(0f64)?))
            .collect::<Result<Vec<_>, OptimizerError>>()?;
        let previous_inner_state = inner_state.parameters().cloned().collect::<Vec<_>>();
        let (updates, inner_state) = self.inner.update(gradients, inner_state, parameters, arguments)?;
        let updates = select_structure("updates", &condition, updates, zeros)?;
        let inner_state = select_structure("inner_state", &condition, inner_state, previous_inner_state)?;
        Ok((updates, ConditionallyMaskState { step: increment_count(&step)?, inner_state }))
    }
}

/// Returns the value of `condition` at `step`, after validating that it is a rank-0 boolean array.
fn evaluate_condition<P: OptimizerValue, C: StepCondition>(condition: &C, step: &P) -> Result<P, OptimizerError> {
    let value = condition.evaluate(step)?;
    let r#type = value.r#type();
    if r#type.rank() != 0 || r#type.data_type() != DataType::Boolean {
        return Err(OptimizerError::InvalidConfiguration {
            message: format!("expected the step condition to return a rank-0 boolean array but got `{}`", r#type),
        });
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use ryft_core::{Array, Rem, bf16};

    use crate::tests::{
        ArgumentProbe, PARITY_GRADIENTS, PARITY_PARAMETERS, assert_parity, parity_elements, parity_parameters,
        parity_updates, state_data_types, state_paths, vector_updates,
    };
    use crate::{Adam, Scale, ScaleBySchedule, Sgd, Trace, apply_updates, filled_statistics};

    use super::*;

    /// Test optimizer whose state is the last gradients in their own data types, so that its state changes its data
    /// types when the gradients and the parameters have different data types. Its updates are the gradients.
    struct LastGradients;

    impl<P: OptimizerValue> Optimizer<P> for LastGradients {
        type State<S: Parameterized<P>> = S;

        fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
            filled_statistics(parameters, 0.0, None)
        }

        fn update<S: Parameterized<P>>(
            &self,
            gradients: S,
            state: Self::State<S>,
            parameters: &S,
            arguments: &UpdateArguments<'_, P, S>,
        ) -> Result<(S, Self::State<S>), OptimizerError> {
            let _ = (state, parameters, arguments);
            let updates = S::from_parameters(
                gradients.parameter_structure(),
                gradients.parameters().cloned().collect::<Vec<_>>(),
            )?;
            Ok((updates, gradients))
        }
    }

    /// [`StepCondition`] that holds at even steps.
    #[derive(Clone, Debug, PartialEq)]
    struct EvenStep;

    impl StepCondition for EvenStep {
        fn evaluate<P: OptimizerValue>(&self, step: &P) -> Result<P, OptimizerError> {
            let remainder = Rem::rem(step, &step.fill_like(2i32)?)?;
            Ok(remainder.compare(&step.fill_like(0i32)?, ComparisonDirection::Equal)?)
        }
    }

    /// [`StepCondition`] that holds at every step.
    struct EveryStep;

    impl StepCondition for EveryStep {
        fn evaluate<P: OptimizerValue>(&self, step: &P) -> Result<P, OptimizerError> {
            Ok(step.compare(step, ComparisonDirection::Equal)?)
        }
    }

    /// Invalid [`StepCondition`] that returns the step count itself rather than a boolean.
    struct StepCount;

    impl StepCondition for StepCount {
        fn evaluate<P: OptimizerValue>(&self, step: &P) -> Result<P, OptimizerError> {
            Ok(step.clone())
        }
    }

    #[test]
    fn test_apply_if_finite_state_parameterized() {
        // The leaves of the inner state are nested under `$.inner_state`.
        let parameters = vec![Array::scalar(1.0f32).unwrap()];
        let state = ApplyIfFinite::new(ScaleBySchedule::new(1.0f64), 0).unwrap().initialize(&parameters).unwrap();
        assert_eq!(
            state.named_parameters().map(|(path, _)| path.to_string()).collect::<Vec<_>>(),
            vec!["$.not_finite_count", "$.last_finite", "$.total_not_finite_count", "$.inner_state.count"],
        );
    }

    #[test]
    fn test_apply_if_finite_new() {
        let optimizer = ApplyIfFinite::new(Scale::new(2.0).unwrap(), 3).unwrap();
        assert_eq!(optimizer.inner(), &Scale::new(2.0).unwrap());
        assert_eq!(optimizer.maximum_consecutive_errors(), 3);
        assert!(matches!(
            ApplyIfFinite::new(Scale::new(2.0).unwrap(), i32::MAX as usize + 1),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`maximum_consecutive_errors` must be at most `2147483647` but got `2147483648`",
        ));
    }

    #[test]
    fn test_apply_if_finite_initialize() {
        let parameters = vec![Array::scalar(1.0f32).unwrap()];
        let state = ApplyIfFinite::new(ScaleBySchedule::new(1.0f64), 0).unwrap().initialize(&parameters).unwrap();
        assert_eq!(state.not_finite_count(), &Array::scalar(0i32).unwrap());
        assert_eq!(state.last_finite(), &Array::scalar(true).unwrap());
        assert_eq!(state.total_not_finite_count(), &Array::scalar(0i32).unwrap());
        assert_eq!(state.inner_state().count(), &Array::scalar(0i32).unwrap());
    }

    #[test]
    fn test_apply_if_finite_update() {
        // Finite gradients are delegated to the inner optimizer, which advances its step count.
        let optimizer = ApplyIfFinite::new(ScaleBySchedule::learning_rate(0.5f64), 5).unwrap();
        let parameters = (Array::vector(vec![1.0f32, 2.0]).unwrap(), Array::scalar(bf16::from_f32(3.0)).unwrap());
        let arguments = UpdateArguments::new();
        let state = optimizer.initialize(&parameters).unwrap();
        let gradients = (Array::vector(vec![2.0f32, -4.0]).unwrap(), Array::scalar(bf16::from_f32(1.0)).unwrap());
        let (updates, state) = optimizer.update(gradients, state, &parameters, &arguments).unwrap();
        assert_eq!(updates, (Array::vector(vec![-1.0f32, 2.0]).unwrap(), Array::scalar(bf16::from_f32(-0.5)).unwrap()));
        assert_eq!(state.inner_state().count(), &Array::scalar(1i32).unwrap());
        assert_eq!(state.last_finite(), &Array::scalar(true).unwrap());

        // A NaN or an infinity in any leaf produces zero updates, leaves the inner state unchanged, and is counted.
        let gradients = (Array::vector(vec![2.0f32, f32::NAN]).unwrap(), Array::scalar(bf16::from_f32(1.0)).unwrap());
        let (updates, state) = optimizer.update(gradients, state, &parameters, &arguments).unwrap();
        assert_eq!(updates, (Array::vector(vec![0f32, 0.0]).unwrap(), Array::scalar(bf16::from_f32(0.0)).unwrap()));
        assert_eq!(state.inner_state().count(), &Array::scalar(1i32).unwrap());
        assert_eq!(state.not_finite_count(), &Array::scalar(1i32).unwrap());
        assert_eq!(state.last_finite(), &Array::scalar(false).unwrap());
        assert_eq!(state.total_not_finite_count(), &Array::scalar(1i32).unwrap());
        let gradients = (Array::vector(vec![2.0f32, 1.0]).unwrap(), Array::scalar(bf16::INFINITY).unwrap());
        let (updates, state) = optimizer.update(gradients, state, &parameters, &arguments).unwrap();
        assert_eq!(updates, (Array::vector(vec![0f32, 0.0]).unwrap(), Array::scalar(bf16::from_f32(0.0)).unwrap()));
        assert_eq!(state.inner_state().count(), &Array::scalar(1i32).unwrap());
        assert_eq!(state.not_finite_count(), &Array::scalar(2i32).unwrap());
        assert_eq!(state.total_not_finite_count(), &Array::scalar(2i32).unwrap());

        // Finite gradients reset the consecutive count but not the total count.
        let gradients = (Array::vector(vec![2.0f32, -4.0]).unwrap(), Array::scalar(bf16::from_f32(1.0)).unwrap());
        let (_, state) = optimizer.update(gradients, state, &parameters, &arguments).unwrap();
        assert_eq!(state.inner_state().count(), &Array::scalar(2i32).unwrap());
        assert_eq!(state.not_finite_count(), &Array::scalar(0i32).unwrap());
        assert_eq!(state.last_finite(), &Array::scalar(true).unwrap());
        assert_eq!(state.total_not_finite_count(), &Array::scalar(2i32).unwrap());

        // `f64` parameters with `f32` gradients keep the data types of the inner state that `initialize` creates, even
        // when the inner optimizer would change them.
        let (initial_data_types, updated_data_types) = state_data_types(&ApplyIfFinite::new(LastGradients, 5).unwrap());
        assert_eq!(updated_data_types, initial_data_types);
    }

    #[test]
    fn test_apply_if_finite_update_maximum_consecutive_errors() {
        // With at most one consecutive error, the first update with a NaN is skipped and the second one is applied,
        // including its NaN, as `optax.apply_if_finite` does.
        let optimizer = ApplyIfFinite::new(ScaleBySchedule::new(1.0f64), 1).unwrap();
        let parameters = vec![Array::scalar(1.0f32).unwrap()];
        let arguments = UpdateArguments::new();
        let gradients = vec![Array::scalar(f32::NAN).unwrap()];
        let state = optimizer.initialize(&parameters).unwrap();
        let (updates, state) = optimizer.update(gradients.clone(), state, &parameters, &arguments).unwrap();
        assert_eq!(updates, vec![Array::scalar(0f32).unwrap()]);
        assert_eq!(state.inner_state().count(), &Array::scalar(0i32).unwrap());
        assert_eq!(state.not_finite_count(), &Array::scalar(1i32).unwrap());
        let (updates, state) = optimizer.update(gradients, state, &parameters, &arguments).unwrap();
        assert!(updates[0].elements::<f32>().unwrap()[0].is_nan());
        assert_eq!(state.inner_state().count(), &Array::scalar(1i32).unwrap());
        assert_eq!(state.not_finite_count(), &Array::scalar(2i32).unwrap());
        assert_eq!(state.total_not_finite_count(), &Array::scalar(2i32).unwrap());
    }

    #[test]
    fn test_conditionally_transform_new() {
        let optimizer = ConditionallyTransform::new(Scale::new(2.0).unwrap(), EvenStep);
        assert_eq!(optimizer.inner(), &Scale::new(2.0).unwrap());
        assert_eq!(optimizer.condition(), &EvenStep);
    }

    #[test]
    fn test_conditionally_transform_initialize() {
        let optimizer = ConditionallyTransform::new(Trace::new(0.5).unwrap(), EvenStep);
        assert_eq!(
            state_paths(&optimizer),
            vec!["$.inner_state.trace[\"bias\"]", "$.inner_state.trace[\"weights\"]", "$.step"],
        );
    }

    #[test]
    fn test_conditionally_transform_update() {
        // With `μ = 0.5`, the gradients `2`, `1`, and `1` give the traces `2` (step `0`), `2` (step `1`, which passes
        // the gradient `1` through and keeps the trace), and `1 + 0.5 · 2 = 2` (step `2`).
        let optimizer = ConditionallyTransform::new(Trace::new(0.5).unwrap(), EvenStep);
        assert_eq!(
            vector_updates(&optimizer, &[0.0], &[&[2.0], &[1.0], &[1.0]]),
            vec![vec![2.0], vec![1.0], vec![2.0]]
        );
        let parameters = vec![Array::vector(vec![0.0f32]).unwrap()];
        let state = optimizer.initialize(&parameters).unwrap();
        let gradients = vec![Array::vector(vec![2.0f32]).unwrap()];
        let (_, state) = optimizer.update(gradients.clone(), state, &parameters, &UpdateArguments::new()).unwrap();
        let (_, state) = optimizer.update(gradients, state, &parameters, &UpdateArguments::new()).unwrap();
        assert_eq!(state.step(), &Array::scalar(2i32).unwrap());
        assert_eq!(state.inner_state().trace(), &vec![Array::vector(vec![2.0f32]).unwrap()]);

        // Conditions must return rank-0 booleans.
        let optimizer = ConditionallyTransform::new(Scale::new(2.0).unwrap(), StepCount);
        assert!(matches!(
            optimizer.update(
                parameters.clone(),
                optimizer.initialize(&parameters).unwrap(),
                &parameters,
                &UpdateArguments::new(),
            ),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "expected the step condition to return a rank-0 boolean array but got `i32[]`",
        ));
    }

    #[test]
    fn test_conditionally_transform_update_arguments() {
        // The wrapped optimizer receives the arguments: `10 + 0.5 + 1 = 11.5`.
        let optimizer = ConditionallyTransform::new(ArgumentProbe, EveryStep);
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
    fn test_conditionally_mask_new() {
        let optimizer = ConditionallyMask::new(Scale::new(2.0).unwrap(), EvenStep);
        assert_eq!(optimizer.inner(), &Scale::new(2.0).unwrap());
        assert_eq!(optimizer.condition(), &EvenStep);
    }

    #[test]
    fn test_conditionally_mask_initialize() {
        let optimizer = ConditionallyMask::new(Trace::new(0.5).unwrap(), EvenStep);
        assert_eq!(
            state_paths(&optimizer),
            vec!["$.step", "$.inner_state.trace[\"bias\"]", "$.inner_state.trace[\"weights\"]"],
        );
    }

    #[test]
    fn test_conditionally_mask_update() {
        // As for `ConditionallyTransform`, except that the update of step `1` is zero.
        let optimizer = ConditionallyMask::new(Trace::new(0.5).unwrap(), EvenStep);
        assert_eq!(
            vector_updates(&optimizer, &[0.0], &[&[2.0], &[1.0], &[1.0]]),
            vec![vec![2.0], vec![0.0], vec![2.0]]
        );
    }

    #[test]
    fn test_conditionally_mask_update_arguments() {
        let optimizer = ConditionallyMask::new(ArgumentProbe, EveryStep);
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
    fn test_apply_if_finite_optax_parity() {
        // Optax 9ecde04b: `optax.apply_if_finite(optax.adam(0.1), max_consecutive_errors=2)`, where the second
        // gradients contain a NaN. That step is skipped and the inner state is kept, so the third step is the second
        // step of Adam.
        const EXPECTED: [[f32; 9]; 4] = [
            [
                -0.09999933,
                0.09999933,
                -0.09999933,
                -0.09999933,
                0.09999933,
                -0.09999933,
                0.09999933,
                -0.09999934,
                0.09999933,
            ],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [
                -0.09177718,
                0.09321701,
                -0.014451908,
                -0.03659542,
                0.026633432,
                -0.09520702,
                0.0038121657,
                -0.09881159,
                0.078331955,
            ],
            [
                -0.09410892,
                0.024214376,
                0.036501687,
                -0.04211546,
                0.067049816,
                -0.07967558,
                0.050194085,
                -0.09196281,
                0.026016414,
            ],
        ];
        // The values of `notfinite_count`, `last_finite`, and `total_notfinite` of the Optax state after every step.
        const COUNTS: [(i32, bool, i32); 4] = [(0, true, 0), (1, false, 1), (0, true, 1), (0, true, 1)];
        let optimizer = ApplyIfFinite::new(Adam::new(0.1), 2).unwrap();
        let mut parameters = parity_parameters(&PARITY_PARAMETERS);
        let mut state = optimizer.initialize(&parameters).unwrap();
        let mut updates = Vec::new();
        for (step, gradients) in PARITY_GRADIENTS.iter().enumerate() {
            let mut gradients = *gradients;
            if step == 1 {
                gradients[1] = f32::NAN;
            }
            let (step_updates, next_state) = optimizer
                .update(parity_parameters(&gradients), state, &parameters, &UpdateArguments::new())
                .unwrap();
            let (not_finite_count, last_finite, total_not_finite_count) = COUNTS[step];
            assert_eq!(next_state.not_finite_count(), &Array::scalar(not_finite_count).unwrap());
            assert_eq!(next_state.last_finite(), &Array::scalar(last_finite).unwrap());
            assert_eq!(next_state.total_not_finite_count(), &Array::scalar(total_not_finite_count).unwrap());
            updates.push(parity_elements(&step_updates));
            parameters = apply_updates(parameters, step_updates).unwrap();
            state = next_state;
        }
        assert_parity(&updates, &EXPECTED);
    }

    #[test]
    fn test_conditionally_transform_optax_parity() {
        // Optax 9ecde04b: `optax.transforms.conditionally_transform(optax.sgd(0.1, momentum=0.9),
        // lambda step: step % 2 == 0)`. The gradients pass through unchanged at odd steps.
        const EXPECTED: [[f32; 9]; 4] = [
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
            [-0.05, 0.25, 0.1, 0.2, 0.3, -0.4, 0.1, -0.2, 0.35],
            [-0.039, 0.028, -0.007, -0.021, 0.004, -0.063, 0.0020000003, -0.085, 0.064],
            [0.2, 0.15, -0.35, 0.1, -0.3, 0.05, -0.45, 0.2, 0.3],
        ];
        let optimizer = ConditionallyTransform::new(Sgd::new(0.1).with_momentum(0.9).unwrap(), EvenStep);
        assert_parity(&parity_updates(&optimizer), &EXPECTED);
    }

    #[test]
    fn test_conditionally_mask_optax_parity() {
        // Optax 9ecde04b: `optax.transforms.conditionally_mask(optax.sgd(0.1, momentum=0.9),
        // lambda step: step % 2 == 0)`. The updates are zero at odd steps.
        const EXPECTED: [[f32; 9]; 4] = [
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
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-0.039, 0.028, -0.007, -0.021, 0.004, -0.063, 0.0020000003, -0.085, 0.064],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        let optimizer = ConditionallyMask::new(Sgd::new(0.1).with_momentum(0.9).unwrap(), EvenStep);
        assert_parity(&parity_updates(&optimizer), &EXPECTED);
    }
}
