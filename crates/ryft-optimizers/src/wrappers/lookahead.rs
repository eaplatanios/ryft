//! Lookahead optimizer of [Zhang et al. (2019)](https://arxiv.org/abs/1907.08610), which takes steps with a fast
//! optimizer and periodically moves a set of slow parameters towards the fast ones.

use std::marker::PhantomData;

use ryft_core::{Add, ComparisonDirection, Mul, Parameter, Parameterized, Rem, Sub};

use crate::macros::parameterized_state;
use crate::{
    Optimizer, OptimizerError, OptimizerValue, UpdateArguments, convert_to, increment_count, initial_count,
    statistic_data_type, validate_finite, validate_structure,
};

use super::select_structure;

// TODO(eaplatanios): Review this module.

parameterized_state! {
    /// Pair of fast and slow parameters with the structure `S`, which is the analogue of
    /// [`optax.LookaheadParams`](https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.LookaheadParams).
    /// Gradients are computed at the fast parameters, while the slow parameters are the ones to evaluate and to use for
    /// inference, because they generalize better. Its leaf paths start with `$.fast` and `$.slow`, and [`Lookahead`]
    /// returns its updates as a pair of the same structure, which [`apply_updates`](crate::apply_updates) applies.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct LookaheadParameters<P, S> with family LookaheadParametersFamily and marker leaf {
        /// Fast parameters, at which gradients are computed.
        fast: S,

        /// Slow parameters, which are used for evaluation and inference.
        slow: S,
    }
}

impl<P: Parameter, S> LookaheadParameters<P, S> {
    /// Creates a new [`LookaheadParameters`] with the fast parameters `fast` and the slow parameters `slow`.
    #[inline]
    pub fn new(fast: S, slow: S) -> Self {
        Self { fast, slow, leaf: PhantomData }
    }

    /// Creates a new [`LookaheadParameters`] whose fast and slow parameters are both `parameters`, which is
    /// `optax.LookaheadParams.init_synced`.
    #[inline]
    pub fn synchronized(parameters: S) -> Self
    where
        S: Clone,
    {
        Self::new(parameters.clone(), parameters)
    }

    /// Returns the fast parameters, at which gradients are computed.
    #[inline]
    pub fn fast(&self) -> &S {
        &self.fast
    }

    /// Returns the slow parameters, which are used for evaluation and inference.
    #[inline]
    pub fn slow(&self) -> &S {
        &self.slow
    }

    /// Returns the fast and the slow parameters, in that order.
    #[inline]
    pub fn into_parts(self) -> (S, S) {
        (self.fast, self.slow)
    }
}

parameterized_state! {
    /// State of [`Lookahead`], which is the analogue of
    /// [`optax.LookaheadState`](https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.LookaheadState). It
    /// holds the state `S` of the fast optimizer, whose leaf paths start with `$.fast_state`, and the number of steps
    /// since the last synchronization.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct LookaheadState<P, S> with family LookaheadStateFamily {
        /// State of the fast optimizer.
        fast_state: S,

        /// Rank-0 `i32` array that holds the number of fast steps taken since the slow and the fast parameters were
        /// last synchronized.
        steps_since_sync: P,
    }
}

impl<P: Parameter, S> LookaheadState<P, S> {
    /// Returns the state of the fast optimizer.
    #[inline]
    pub fn fast_state(&self) -> &S {
        &self.fast_state
    }

    /// Returns the rank-0 `i32` array that holds the number of fast steps taken since the last synchronization.
    #[inline]
    pub fn steps_since_sync(&self) -> &P {
        &self.steps_since_sync
    }
}

/// Lookahead optimizer of [Zhang et al. (2019)](https://arxiv.org/abs/1907.08610), which takes `k` steps with a fast
/// optimizer and then moves a set of slow parameters part of the way towards the fast ones, and which is the analogue
/// of [`optax.lookahead`](https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.lookahead). With the fast
/// parameters `φ` and the slow parameters `ψ` of a [`LookaheadParameters`] pair, its update is
///
/// ```text
/// φ̃ₜ = φₜ₋₁ + vₜ
/// ψₜ = ψₜ₋₁ + α · (φ̃ₜ − ψₜ₋₁)   if t mod k = 0
/// ψₜ = ψₜ₋₁                     otherwise
/// φₜ = ψₜ                       if t mod k = 0
/// φₜ = φ̃ₜ                       otherwise
/// ```
///
/// Here `t ≥ 1` counts the updates including the current one, `vₜ` is the update of the fast optimizer for the gradient
/// at the fast parameters `φₜ₋₁`, `k` is the synchronization period, and `α` is the slow step size. The update that it
/// returns is the pair `(φₜ − φₜ₋₁, ψₜ − ψₜ₋₁)`, which must be applied unchanged with
/// [`apply_updates`](crate::apply_updates), or the two sets of parameters do not stay synchronized. All operations are
/// element-wise. There are no defaults for `k` and `α`, which [`Self::new`] takes, and [`Self::with_reset_state`]
/// reinitializes the state of the fast optimizer at every synchronization.
///
/// Optax passes the pair of parameters to the update function of a gradient transformation whose gradients are those of
/// the fast parameters only, which the [`Optimizer`] trait cannot express, because its gradients, parameters, and
/// updates share one structure. So this optimizer does not implement [`Optimizer`] and provides [`Self::initialize`]
/// and [`Self::update`] for [`LookaheadParameters`] instead, which cannot be chained with other optimizers. It wraps
/// the fast optimizer, which can be any chain, rather than building a chain itself.
///
/// # References
///
///   - Zhang et al. (2019). [Lookahead Optimizer: k steps forward, 1 step back](https://arxiv.org/abs/1907.08610).
#[derive(Clone, Debug, PartialEq)]
pub struct Lookahead<O> {
    /// Optimizer that takes the steps on the fast parameters.
    fast_optimizer: O,

    /// Number `k` of fast steps between two synchronizations, which is at most `i32::MAX`.
    sync_period: usize,

    /// Slow step size `α` by which the slow parameters move towards the fast ones.
    slow_step_size: f64,

    /// Whether the state of the fast optimizer is reinitialized at every synchronization.
    reset_state: bool,
}

impl<O> Lookahead<O> {
    /// Creates a new [`Lookahead`] that takes steps with `fast_optimizer` and synchronizes the parameters every
    /// `sync_period` fast steps with the slow step size `slow_step_size`, without resetting the state of the fast
    /// optimizer.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `sync_period` is zero or greater than `i32::MAX`, or when
    /// `slow_step_size` is not finite.
    pub fn new(fast_optimizer: O, sync_period: usize, slow_step_size: f64) -> Result<Self, OptimizerError> {
        if sync_period == 0 || i32::try_from(sync_period).is_err() {
            return Err(OptimizerError::InvalidConfiguration {
                message: format!("`sync_period` must be in `[1, {}]` but got `{sync_period}`", i32::MAX),
            });
        }
        validate_finite("slow_step_size", slow_step_size)?;
        Ok(Self { fast_optimizer, sync_period, slow_step_size, reset_state: false })
    }

    /// Returns this optimizer with the state of the fast optimizer reinitialized at every synchronization when
    /// `reset_state` is `true`.
    #[inline]
    pub fn with_reset_state(mut self, reset_state: bool) -> Self {
        self.reset_state = reset_state;
        self
    }

    /// Returns the optimizer that takes the steps on the fast parameters.
    #[inline]
    pub fn fast_optimizer(&self) -> &O {
        &self.fast_optimizer
    }

    /// Returns the number of fast steps between two synchronizations.
    #[inline]
    pub fn sync_period(&self) -> usize {
        self.sync_period
    }

    /// Returns the slow step size by which the slow parameters move towards the fast ones.
    #[inline]
    pub fn slow_step_size(&self) -> f64 {
        self.slow_step_size
    }

    /// Returns whether the state of the fast optimizer is reinitialized at every synchronization.
    #[inline]
    pub fn reset_state(&self) -> bool {
        self.reset_state
    }

    /// Creates the initial state of this optimizer for `parameters`, which initializes the fast optimizer for the fast
    /// parameters.
    ///
    /// # Errors
    ///
    /// Returns an [`OptimizerError`] when the fast optimizer cannot be initialized for the fast parameters or when the
    /// parameters have no leaves.
    pub fn initialize<P: OptimizerValue, S: Parameterized<P>>(
        &self,
        parameters: &LookaheadParameters<P, S>,
    ) -> Result<LookaheadState<P, O::State<S>>, OptimizerError>
    where
        O: Optimizer<P>,
    {
        Ok(LookaheadState {
            fast_state: self.fast_optimizer.initialize(&parameters.fast)?,
            steps_since_sync: initial_count("Lookahead", &parameters.fast)?,
        })
    }

    /// Transforms the `gradients` of the objective at the fast parameters into updates for the pair `parameters` and
    /// returns them together with the next state.
    ///
    /// # Parameters
    ///
    ///   - `gradients`: Gradients at the fast parameters, with their structure.
    ///   - `state`: State returned by [`Self::initialize`] or by the previous update.
    ///   - `parameters`: Current pair of fast and slow parameters.
    ///   - `arguments`: Optional arguments that the fast optimizer needs, which refer to the fast parameters.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::StructureMismatch`] when the structures of `gradients` and of the two sets of
    /// parameters differ, and other [`OptimizerError`]s when the fast optimizer or the array operations fail.
    #[allow(clippy::type_complexity)]
    pub fn update<P: OptimizerValue, S: Parameterized<P>>(
        &self,
        gradients: S,
        state: LookaheadState<P, O::State<S>>,
        parameters: &LookaheadParameters<P, S>,
        arguments: &UpdateArguments<'_, P, S>,
    ) -> Result<(LookaheadParameters<P, S>, LookaheadState<P, O::State<S>>), OptimizerError>
    where
        O: Optimizer<P>,
    {
        // This follows the operation order of `optax.lookahead`, with the selection of the reset state instead of its
        // arithmetic blend.
        let paths = gradients.parameter_paths().collect::<Vec<_>>();
        validate_structure("fast", &paths, &parameters.fast)?;
        validate_structure("slow", &paths, &parameters.slow)?;
        let LookaheadState { fast_state, steps_since_sync } = state;
        let (updates, fast_state) = self.fast_optimizer.update(gradients, fast_state, &parameters.fast, arguments)?;
        let sync_period = steps_since_sync.fill_like(self.sync_period as i32)?;
        let sync_next = steps_since_sync
            .compare(&steps_since_sync.fill_like(self.sync_period as i32 - 1)?, ComparisonDirection::Equal)?;

        let structure = updates.parameter_structure();
        let mut fast_updates = Vec::with_capacity(paths.len());
        let mut slow_updates = Vec::with_capacity(paths.len());
        for ((update, fast), slow) in
            updates.into_parameters().zip(parameters.fast.parameters()).zip(parameters.slow.parameters())
        {
            let data_type = update.r#type().data_type();
            let statistic_type = statistic_data_type(data_type);
            let update = convert_to(&update, statistic_type)?;
            let sync = convert_to(&sync_next, statistic_type)?;
            let slow_step_size = update.fill_like(self.slow_step_size)?;
            let last_difference =
                Sub::sub(&Add::add(&convert_to(fast, statistic_type)?, &update)?, &convert_to(slow, statistic_type)?)?;
            let slow_update = Mul::mul(&Mul::mul(&slow_step_size, &sync)?, &last_difference)?;
            let fast_update = Sub::sub(
                &update,
                &Mul::mul(&Mul::mul(&sync, &Sub::sub(&update.fill_like(1f64)?, &slow_step_size)?)?, &last_difference)?,
            )?;
            fast_updates.push(convert_to(&fast_update, data_type)?);
            slow_updates.push(convert_to(&slow_update, slow.r#type().data_type())?);
        }
        let updates = LookaheadParameters::new(
            S::from_parameters(structure.clone(), fast_updates)?,
            S::from_parameters(structure, slow_updates)?,
        );

        let fast_state = if self.reset_state {
            let current_state = fast_state.parameters().cloned().collect::<Vec<_>>();
            let initial_state = self.fast_optimizer.initialize(&parameters.fast)?;
            select_structure("fast_state", &sync_next, initial_state, current_state)?
        } else {
            fast_state
        };
        let steps_since_sync = Rem::rem(&increment_count(&steps_since_sync)?, &sync_period)?;
        Ok((updates, LookaheadState { fast_state, steps_since_sync }))
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use ryft_core::Array;

    use crate::tests::{
        ArgumentProbe, PARITY_GRADIENTS, PARITY_PARAMETERS, assert_parity, parity_elements, parity_parameters,
    };
    use crate::{Adam, Scale, ScaleBySchedule, Trace, apply_updates};

    use super::*;

    #[test]
    fn test_lookahead_parameters_new() {
        let parameters = LookaheadParameters::<Array, _>::new(
            vec![Array::scalar(1.0f32).unwrap()],
            vec![Array::scalar(2.0f32).unwrap()],
        );
        assert_eq!(parameters.fast(), &vec![Array::scalar(1.0f32).unwrap()]);
        assert_eq!(parameters.slow(), &vec![Array::scalar(2.0f32).unwrap()]);
        assert_eq!(
            parameters.named_parameters().map(|(path, _)| path.to_string()).collect::<Vec<_>>(),
            vec!["$.fast[0]", "$.slow[0]"],
        );
        assert_eq!(
            parameters.into_parts(),
            (vec![Array::scalar(1.0f32).unwrap()], vec![Array::scalar(2.0f32).unwrap()]),
        );
    }

    #[test]
    fn test_lookahead_parameters_synchronized() {
        let parameters = LookaheadParameters::<Array, _>::synchronized(vec![Array::scalar(1.0f32).unwrap()]);
        assert_eq!(parameters.fast(), parameters.slow());
    }

    #[test]
    fn test_lookahead_new() {
        let optimizer = Lookahead::new(Scale::new(-1.0).unwrap(), 2, 0.5).unwrap();
        assert_eq!(optimizer.fast_optimizer(), &Scale::new(-1.0).unwrap());
        assert_eq!(optimizer.sync_period(), 2);
        assert_eq!(optimizer.slow_step_size(), 0.5);
        assert!(!optimizer.reset_state());
        assert!(matches!(
            Lookahead::new(Scale::new(-1.0).unwrap(), 0, 0.5),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`sync_period` must be in `[1, 2147483647]` but got `0`",
        ));
        assert!(matches!(
            Lookahead::new(Scale::new(-1.0).unwrap(), 1, f64::NAN),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`slow_step_size` must be finite but got `NaN`",
        ));
    }

    #[test]
    fn test_lookahead_with_reset_state() {
        // The step count of the fast optimizer is reset to `0` at the synchronization after two steps.
        let optimizer = Lookahead::new(ScaleBySchedule::new(1.0f64), 2, 0.5).unwrap().with_reset_state(true);
        assert!(optimizer.reset_state());
        let parameters = LookaheadParameters::synchronized(vec![Array::scalar(1.0f32).unwrap()]);
        let gradients = vec![Array::scalar(1.0f32).unwrap()];
        let arguments = UpdateArguments::new();
        let state = optimizer.initialize(&parameters).unwrap();
        let (_, state) = optimizer.update(gradients.clone(), state, &parameters, &arguments).unwrap();
        assert_eq!(state.fast_state().count(), &Array::scalar(1i32).unwrap());
        let (_, state) = optimizer.update(gradients, state, &parameters, &arguments).unwrap();
        assert_eq!(state.fast_state().count(), &Array::scalar(0i32).unwrap());
    }

    #[test]
    fn test_lookahead_initialize() {
        let optimizer = Lookahead::new(Trace::new(0.9).unwrap(), 2, 0.5).unwrap();
        let parameters = LookaheadParameters::synchronized(vec![Array::scalar(1.0f32).unwrap()]);
        let state = optimizer.initialize(&parameters).unwrap();
        assert_eq!(
            state.named_parameters().map(|(path, _)| path.to_string()).collect::<Vec<_>>(),
            vec!["$.fast_state.trace[0]", "$.steps_since_sync"],
        );
    }

    #[test]
    fn test_lookahead_update() {
        // The fast optimizer takes the steps `-1` from `θ_f = θₛ = 1`. After the first step, `θ_f = 0` and `θₛ = 1`.
        // The second step synchronizes: the last difference is `0 - 1 - 1 = -2`, so `θₛ` moves by `0.5 · -2 = -1` to
        // `0` and the fast update is `-1 - 0.5 · -2 = 0`, which leaves `θ_f = 0` at the new slow parameters.
        let optimizer = Lookahead::new(Scale::new(-1.0).unwrap(), 2, 0.5).unwrap();
        let parameters = LookaheadParameters::synchronized(vec![Array::scalar(1.0f32).unwrap()]);
        let gradients = vec![Array::scalar(1.0f32).unwrap()];
        let arguments = UpdateArguments::new();
        let state = optimizer.initialize(&parameters).unwrap();
        let (updates, state) = optimizer.update(gradients.clone(), state, &parameters, &arguments).unwrap();
        assert_eq!(
            updates,
            LookaheadParameters::new(vec![Array::scalar(-1.0f32).unwrap()], vec![Array::scalar(0.0f32).unwrap()])
        );
        assert_eq!(state.steps_since_sync(), &Array::scalar(1i32).unwrap());
        let parameters = apply_updates(parameters, updates).unwrap();
        let (updates, state) = optimizer.update(gradients, state, &parameters, &arguments).unwrap();
        assert_eq!(
            updates,
            LookaheadParameters::new(vec![Array::scalar(0.0f32).unwrap()], vec![Array::scalar(-1.0f32).unwrap()])
        );
        assert_eq!(state.steps_since_sync(), &Array::scalar(0i32).unwrap());
        let parameters = apply_updates(parameters, updates).unwrap();
        assert_eq!(parameters, LookaheadParameters::synchronized(vec![Array::scalar(0.0f32).unwrap()]));

        // Gradients must have the structure of the parameters.
        assert!(matches!(
            optimizer.update(Vec::new(), state, &parameters, &arguments),
            Err(OptimizerError::StructureMismatch { message })
                if message == "expected the leaf count of `fast` to be 0 but got 1",
        ));
    }

    #[test]
    fn test_lookahead_update_arguments() {
        // The fast optimizer receives the arguments and the fast parameters: `10 + 0.5 + 1 = 11.5`.
        let optimizer = Lookahead::new(ArgumentProbe, 2, 0.5).unwrap();
        let parameters =
            LookaheadParameters::new(vec![Array::scalar(1.0f32).unwrap()], vec![Array::scalar(5.0f32).unwrap()]);
        let value = Array::scalar(0.5f32).unwrap();
        let argument_gradients = vec![Array::scalar(10.0f32).unwrap()];
        let objective = |parameters: &Vec<Array>| -> Result<Array, OptimizerError> { Ok(parameters[0].clone()) };
        let arguments = UpdateArguments::new()
            .with_value(&value)
            .with_gradients(&argument_gradients)
            .with_objective(&objective);
        let state = optimizer.initialize(&parameters).unwrap();
        let gradients = vec![Array::scalar(1.0f32).unwrap()];
        let (updates, _) = optimizer.update(gradients, state, &parameters, &arguments).unwrap();
        assert_eq!(updates.fast(), &vec![Array::scalar(11.5f32).unwrap()]);
    }

    #[test]
    fn test_lookahead_optax_parity() {
        // Optax 9ecde04b: `optax.lookahead(optax.adam(0.1), sync_period=2, slow_step_size=0.5, reset_state=False)`,
        // from `optax.LookaheadParams.init_synced(PARITY_PARAMETERS)`. Every row holds the updates of the fast
        // parameters followed by those of the slow parameters.
        const KEPT: [[f32; 18]; 4] = [
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
                -0.0,
                0.0,
                -0.0,
                -0.0,
                0.0,
                -0.0,
                0.0,
                -0.0,
                0.0,
            ],
            [
                0.036682956,
                -0.058136173,
                0.0064469054,
                0.0033911392,
                -0.07470892,
                0.068304665,
                -0.029988911,
                0.032719526,
                -0.03981878,
                -0.06331639,
                0.041863114,
                -0.09355241,
                -0.09660816,
                0.02529043,
                -0.03169468,
                0.070010364,
                -0.067279756,
                0.060180545,
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
                -0.0,
                0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                0.0,
            ],
            [
                -0.0051626116,
                -0.012763163,
                0.02443019,
                -0.0012661777,
                0.02948238,
                4.2844564e-5,
                0.025142405,
                -0.0019056723,
                -0.0144829145,
                -0.07221237,
                -0.0057117343,
                0.0017076731,
                -0.04924643,
                -0.018231153,
                -0.025338791,
                0.00976187,
                -0.061459184,
                0.010206491,
            ],
        ];

        // Optax 9ecde04b: the same call with `reset_state=True`, which restarts Adam after every synchronization.
        const RESET: [[f32; 18]; 4] = [
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
                -0.0,
                0.0,
                -0.0,
                -0.0,
                0.0,
                -0.0,
                0.0,
                -0.0,
                0.0,
            ],
            [
                0.036682956,
                -0.058136173,
                0.0064469054,
                0.0033911392,
                -0.07470892,
                0.068304665,
                -0.029988911,
                0.032719526,
                -0.03981878,
                -0.06331639,
                0.041863114,
                -0.09355241,
                -0.09660816,
                0.02529043,
                -0.03169468,
                0.070010364,
                -0.067279756,
                0.060180545,
            ],
            [
                -0.09999933,
                0.09999933,
                0.09999933,
                0.09999933,
                -0.099999316,
                -0.09999933,
                -0.09999933,
                -0.09999934,
                0.09999933,
                -0.0,
                0.0,
                0.0,
                0.0,
                -0.0,
                -0.0,
                -0.0,
                -0.0,
                0.0,
            ],
            [
                0.0014825389,
                -0.062384665,
                -0.0010757595,
                -0.042773683,
                0.08119528,
                0.012591407,
                0.06626377,
                0.0033911392,
                -0.074708894,
                -0.09851678,
                0.037614733,
                0.098923564,
                0.057225585,
                -0.018804044,
                -0.08740792,
                -0.033735633,
                -0.09660816,
                0.0252904,
            ],
        ];

        for (reset_state, expected) in [(false, KEPT), (true, RESET)] {
            let optimizer = Lookahead::new(Adam::new(0.1), 2, 0.5).unwrap().with_reset_state(reset_state);
            let mut parameters = LookaheadParameters::synchronized(parity_parameters(&PARITY_PARAMETERS));
            let mut state = optimizer.initialize(&parameters).unwrap();
            let mut updates = Vec::new();
            for gradients in &PARITY_GRADIENTS {
                let (step_updates, next_state) = optimizer
                    .update(parity_parameters(gradients), state, &parameters, &UpdateArguments::new())
                    .unwrap();
                let mut row = parity_elements(step_updates.fast());
                row.extend(parity_elements(step_updates.slow()));
                updates.push(row);
                parameters = apply_updates(parameters, step_updates).unwrap();
                state = next_state;
            }
            assert_parity(&updates, &expected);
        }
    }
}
