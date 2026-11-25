//! Learning rates that are stored in the optimizer state, which is the statically typed counterpart of
//! [`optax.inject_hyperparams`](
//! https://optax.readthedocs.io/en/latest/api/optimizer_schedules.html#optax.inject_hyperparams) for learning rates.
//!
//! Optax rebuilds the wrapped transformation from the hyperparameters that it stores in the state at every step, which
//! needs transformations whose hyperparameters may be traced arrays. The transformations of this crate take their
//! hyperparameters as host values when they are built, so [`InjectLearningRate`] injects the one hyperparameter that
//! every optimizer applies in the same way, the learning rate, by applying it after an unscaled optimizer (e.g.,
//! [`ScaleByAdam`](crate::ScaleByAdam) rather than [`Adam`](crate::Adam)). The learning rate is recorded in the state
//! at every step and, unless it follows a schedule, read from the state, so a caller can change it between steps with
//! [`InjectLearningRateState::with_learning_rate`]. Other hyperparameters (e.g., the decay rates of Adam) cannot be
//! injected.

use ryft_core::{Mul, Neg, Parameter, Parameterized};

use crate::macros::parameterized_state;
use crate::{
    Optimizer, OptimizerError, OptimizerValue, Schedule, UpdateArguments, convert_to, increment_count, initial_count,
    statistic_data_type, validate_finite,
};

// TODO(eaplatanios): Review this module.

parameterized_state! {
    /// State of [`InjectLearningRate`], which is the analogue of
    /// [`optax.InjectHyperparamsState`](
    /// https://optax.readthedocs.io/en/latest/api/optimizer_schedules.html#optax.InjectHyperparamsState)
    /// with the learning rate as its only hyperparameter. It holds the step count, the learning rate, and the state
    /// `S` of the wrapped optimizer, whose leaf paths start with `$.inner_state`.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct InjectLearningRateState<P, S> with family InjectLearningRateStateFamily {
        /// Rank-0 `i32` array that holds the number of updates applied so far.
        count: P,

        /// Rank-0 array that holds the learning rate of the last update (or of the first update before any update was
        /// applied), which is the next learning rate unless the learning rate follows a schedule.
        learning_rate: P,

        /// State of the wrapped optimizer.
        inner_state: S,
    }
}

impl<P: Parameter, S> InjectLearningRateState<P, S> {
    /// Returns this state with `learning_rate` as its learning rate, which [`InjectLearningRate`] applies at the next
    /// update unless its learning rate follows a schedule. `learning_rate` must be a rank-0 floating-point array (e.g.,
    /// `Array::scalar(0.01f32)`).
    #[inline]
    pub fn with_learning_rate(mut self, learning_rate: P) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Returns the rank-0 `i32` array that holds the number of updates applied so far.
    #[inline]
    pub fn count(&self) -> &P {
        &self.count
    }

    /// Returns the rank-0 array that holds the learning rate of the last update.
    #[inline]
    pub fn learning_rate(&self) -> &P {
        &self.learning_rate
    }

    /// Returns the state of the wrapped optimizer.
    #[inline]
    pub fn inner_state(&self) -> &S {
        &self.inner_state
    }
}

/// Optimizer that applies another optimizer and scales its updates by the negated learning rate `-η`, where `η` is
/// stored in the state, which is the analogue of
/// [`optax.inject_hyperparams`](
/// https://optax.readthedocs.io/en/latest/api/optimizer_schedules.html#optax.inject_hyperparams)
/// applied to the learning rate of an optimizer (e.g., `optax.inject_hyperparams(optax.adam)(learning_rate=0.1)` is
/// `InjectLearningRate::new(ScaleByAdam::new(), 0.1)`).
///
/// [`Self::new`] creates an injected learning rate, which is initialized to a constant and read from the state at
/// every update, so a caller can change it between updates with [`InjectLearningRateState::with_learning_rate`], as
/// Optax lets callers change the numeric hyperparameters in `state.hyperparams`. [`Self::scheduled`] creates a
/// learning rate that follows a [`Schedule`], which is evaluated at the step count before every update and recorded
/// in the state, where changes are ignored, as Optax ignores changes to scheduled hyperparameters. The learning rate
/// is stored as a rank-0 `f32` array (i.e., the value type of schedules), where Optax stores it in the data type of
/// the parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct InjectLearningRate<O, Sc: Schedule = f64> {
    /// Optimizer whose updates are scaled by the negated learning rate.
    inner: O,

    /// Initial learning rate of an injected learning rate, or the learning rate schedule of a scheduled one.
    learning_rate: Sc,

    /// Whether the learning rate follows [`Self::learning_rate`] at every step rather than being read from the state.
    scheduled: bool,
}

impl<O> InjectLearningRate<O> {
    /// Creates a new [`InjectLearningRate`] that scales the updates of `inner` by the negated learning rate that is
    /// stored in the state, which is initialized to `learning_rate`.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `learning_rate` is not finite.
    pub fn new(inner: O, learning_rate: f64) -> Result<Self, OptimizerError> {
        validate_finite("learning_rate", learning_rate)?;
        Ok(Self { inner, learning_rate, scheduled: false })
    }
}

impl<O, Sc: Schedule> InjectLearningRate<O, Sc> {
    /// Creates a new [`InjectLearningRate`] that scales the updates of `inner` by the negated value of the learning
    /// rate schedule `learning_rate` at the step count before every update, and that records that value in the state.
    #[inline]
    pub fn scheduled(inner: O, learning_rate: Sc) -> Self {
        Self { inner, learning_rate, scheduled: true }
    }

    /// Returns the optimizer whose updates this optimizer scales.
    #[inline]
    pub fn inner(&self) -> &O {
        &self.inner
    }

    /// Returns the initial learning rate of an injected learning rate, or the learning rate schedule of a scheduled
    /// one.
    #[inline]
    pub fn learning_rate(&self) -> &Sc {
        &self.learning_rate
    }

    /// Returns whether the learning rate of this optimizer follows a schedule rather than being read from the state.
    #[inline]
    pub fn is_scheduled(&self) -> bool {
        self.scheduled
    }
}

impl<P: OptimizerValue, O: Optimizer<P>, Sc: Schedule> Optimizer<P> for InjectLearningRate<O, Sc> {
    type State<S: Parameterized<P>> = InjectLearningRateState<P, O::State<S>>;

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        let count = initial_count("InjectLearningRate", parameters)?;
        Ok(InjectLearningRateState {
            learning_rate: self.learning_rate.value(&count)?,
            count,
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
        // This follows `optax.inject_hyperparams`, which computes the hyperparameters before the inner update.
        let InjectLearningRateState { count, learning_rate, inner_state } = state;
        let learning_rate = if self.scheduled { self.learning_rate.value(&count)? } else { learning_rate };
        let (updates, inner_state) = self.inner.update(gradients, inner_state, parameters, arguments)?;
        let factor = Neg::neg(&learning_rate)?;
        let structure = updates.parameter_structure();
        let updates = updates
            .into_parameters()
            .map(|update| {
                let data_type = update.r#type().data_type();
                let statistic_type = statistic_data_type(data_type);
                let update = Mul::mul(&convert_to(&factor, statistic_type)?, &convert_to(&update, statistic_type)?)?;
                Ok(convert_to(&update, data_type)?)
            })
            .collect::<Result<Vec<_>, OptimizerError>>()?;
        let state = InjectLearningRateState { count: increment_count(&count)?, learning_rate, inner_state };
        Ok((S::from_parameters(structure, updates)?, state))
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use ryft_core::Array;

    use crate::tests::{
        ArgumentProbe, PARITY_GRADIENTS, PARITY_PARAMETERS, assert_parity, parity_parameters, parity_updates,
        state_paths, vector_updates,
    };
    use crate::{LinearSchedule, Scale, ScaleByAdam, Trace};

    use super::*;

    #[test]
    fn test_inject_learning_rate_state_with_learning_rate() {
        // The learning rate that a caller stores in the state is applied at the next update: `-0.5 · 2` and then
        // `-0.25 · 2`.
        let optimizer = InjectLearningRate::new(Scale::new(1.0).unwrap(), 0.5).unwrap();
        let parameters = vec![Array::scalar(0.0f32).unwrap()];
        let gradients = vec![Array::scalar(2.0f32).unwrap()];
        let arguments = UpdateArguments::new();
        let state = optimizer.initialize(&parameters).unwrap();
        assert_eq!(state.learning_rate(), &Array::scalar(0.5f32).unwrap());
        let (updates, state) = optimizer.update(gradients.clone(), state, &parameters, &arguments).unwrap();
        assert_eq!(updates, vec![Array::scalar(-1.0f32).unwrap()]);
        let state = state.with_learning_rate(Array::scalar(0.25f32).unwrap());
        let (updates, state) = optimizer.update(gradients, state, &parameters, &arguments).unwrap();
        assert_eq!(updates, vec![Array::scalar(-0.5f32).unwrap()]);
        assert_eq!(state.learning_rate(), &Array::scalar(0.25f32).unwrap());
        assert_eq!(state.count(), &Array::scalar(2i32).unwrap());
    }

    #[test]
    fn test_inject_learning_rate_new() {
        let optimizer = InjectLearningRate::new(Scale::new(2.0).unwrap(), 0.1).unwrap();
        assert_eq!(optimizer.inner(), &Scale::new(2.0).unwrap());
        assert_eq!(optimizer.learning_rate(), &0.1);
        assert!(!optimizer.is_scheduled());
        assert!(matches!(
            InjectLearningRate::new(Scale::new(2.0).unwrap(), f64::INFINITY),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`learning_rate` must be finite but got `inf`",
        ));
    }

    #[test]
    fn test_inject_learning_rate_scheduled() {
        // The schedule goes from `1` to `0.5` over two steps, so the updates of the gradient `2` are `-2`, `-1.5`, and
        // `-1`, and changes to the learning rate in the state are ignored.
        let schedule = LinearSchedule::new(1.0, 0.5, 2).unwrap();
        let optimizer = InjectLearningRate::scheduled(Scale::new(1.0).unwrap(), schedule.clone());
        assert_eq!(optimizer.learning_rate(), &schedule);
        assert!(optimizer.is_scheduled());
        assert_eq!(
            vector_updates(&optimizer, &[0.0], &[&[2.0], &[2.0], &[2.0]]),
            vec![vec![-2.0], vec![-1.5], vec![-1.0]],
        );
        let parameters = vec![Array::scalar(0.0f32).unwrap()];
        let state = optimizer.initialize(&parameters).unwrap().with_learning_rate(Array::scalar(10.0f32).unwrap());
        let (updates, state) =
            optimizer.update(parameters.clone(), state, &parameters, &UpdateArguments::new()).unwrap();
        assert_eq!(updates, vec![Array::scalar(-0.0f32).unwrap()]);
        assert_eq!(state.learning_rate(), &Array::scalar(1.0f32).unwrap());
    }

    #[test]
    fn test_inject_learning_rate_initialize() {
        let optimizer = InjectLearningRate::new(Trace::new(0.9).unwrap(), 0.1).unwrap();
        assert_eq!(
            state_paths(&optimizer),
            vec!["$.count", "$.learning_rate", "$.inner_state.trace[\"bias\"]", "$.inner_state.trace[\"weights\"]"],
        );
    }

    #[test]
    fn test_inject_learning_rate_update() {
        // With `μ = 0.5` and `η = 0.1`, the gradients `2` and `1` give the traces `2` and `2`, so both updates are
        // `-0.2`.
        let optimizer = InjectLearningRate::new(Trace::new(0.5).unwrap(), 0.1).unwrap();
        assert_eq!(vector_updates(&optimizer, &[0.0], &[&[2.0], &[1.0]]), vec![vec![-0.2], vec![-0.2]]);
    }

    #[test]
    fn test_inject_learning_rate_update_arguments() {
        // The wrapped optimizer receives the arguments: `-(10 + 0.5 + 1)`.
        let optimizer = InjectLearningRate::new(ArgumentProbe, 1.0).unwrap();
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
        assert_eq!(updates, vec![Array::scalar(-11.5f32).unwrap()]);
    }

    #[test]
    fn test_inject_learning_rate_optax_parity() {
        // Optax 9ecde04b: `optax.inject_hyperparams(optax.adam)(learning_rate=0.1)`.
        const INJECTED: [[f32; 9]; 4] = [
            [-0.099999994, 0.099999994, -0.1, -0.1, 0.099999994, -0.099999994, 0.1, -0.1, 0.1],
            [
                -0.026633611,
                -0.016273169,
                -0.08710609,
                -0.09321765,
                -0.049418814,
                0.036610227,
                0.040021718,
                -0.034560464,
                0.020361902,
            ],
            [
                -0.06705026,
                0.007051412,
                -0.022722585,
                -0.04798055,
                -0.047713857,
                -0.02538181,
                -0.015380639,
                -0.05955385,
                0.024689557,
            ],
            [
                -0.077375494,
                -0.018475022,
                0.026138041,
                -0.050512947,
                0.011251303,
                -0.02529612,
                0.03490451,
                -0.06336529,
                -0.004276451,
            ],
        ];
        let optimizer = InjectLearningRate::new(ScaleByAdam::new(), 0.1).unwrap();
        assert_parity(&parity_updates(&optimizer), &INJECTED);

        // Optax 9ecde04b: `optax.inject_hyperparams(optax.sgd)(learning_rate=optax.linear_schedule(0.1, 0.01, 3))`,
        // whose state records the learning rates `0.1`, `0.07`, `0.04`, and `0.01` after the four steps.
        const SCHEDULED: [[f32; 9]; 4] = [
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
            [0.0035, -0.0175, -0.007, -0.014, -0.021000002, 0.028, -0.007, 0.014, -0.0245],
            [-0.012, 0.004, 0.008, 0.006, -0.002, -0.018, -0.01, -0.016, 0.004],
            [-0.002, -0.0015, 0.0034999999, -0.001, 0.003, -0.0005, 0.0045, -0.002, -0.003],
        ];
        const LEARNING_RATES: [f32; 4] = [0.1, 0.07, 0.04, 0.01];
        let optimizer =
            InjectLearningRate::scheduled(Scale::new(1.0).unwrap(), LinearSchedule::new(0.1, 0.01, 3).unwrap());
        assert_parity(&parity_updates(&optimizer), &SCHEDULED);
        let parameters = parity_parameters(&PARITY_PARAMETERS);
        let mut state = optimizer.initialize(&parameters).unwrap();
        let mut learning_rates = Vec::new();
        for gradients in &PARITY_GRADIENTS {
            let (_, next_state) =
                optimizer.update(parity_parameters(gradients), state, &parameters, &UpdateArguments::new()).unwrap();
            learning_rates.push(next_state.learning_rate().elements::<f32>().unwrap()[0]);
            state = next_state;
        }
        assert_parity(&[learning_rates], &[LEARNING_RATES]);
    }
}
