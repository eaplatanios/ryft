//! Limited memory BFGS optimizer, which preconditions the gradients with an approximation of the inverse Hessian and
//! chooses its step sizes with a line search.

use ryft_core::{Parameter, Parameterized};

use crate::macros::parameterized_state;
use crate::{
    InitialGuessStrategy, Optimizer, OptimizerError, OptimizerValue, ScaleByLbfgs, ScaleByLbfgsState, ScaleBySchedule,
    ScaleByScheduleState, ScaleByZoomLinesearch, Schedule, UpdateArguments,
};

// TODO(eaplatanios): Review this module.

parameterized_state! {
    /// State of [`Lbfgs`] for the parameter structure `S`, which holds the state of its L-BFGS preconditioning, the
    /// step count of its learning rate schedule, and the state `I` of its line search, which has one element when the
    /// optimizer has a line search and none otherwise.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct LbfgsState<P, S, I> with family LbfgsStateFamily {
        /// State of the [`ScaleByLbfgs`] transformation of the optimizer.
        lbfgs: ScaleByLbfgsState<P, S>,

        /// State of the [`ScaleBySchedule`] transformation that applies the learning rate.
        learning_rate: ScaleByScheduleState<P>,

        /// State of the line search, if the optimizer has one.
        linesearch: Vec<I>,
    }
}

impl<P: Parameter, S, I> LbfgsState<P, S, I> {
    /// Returns the state of the [`ScaleByLbfgs`] transformation of the optimizer.
    #[inline]
    pub fn lbfgs(&self) -> &ScaleByLbfgsState<P, S> {
        &self.lbfgs
    }

    /// Returns the state of the [`ScaleBySchedule`] transformation that applies the learning rate.
    #[inline]
    pub fn learning_rate(&self) -> &ScaleByScheduleState<P> {
        &self.learning_rate
    }

    /// Returns the state of the line search, or [`None`] when the optimizer has no line search.
    #[inline]
    pub fn linesearch(&self) -> Option<&I> {
        self.linesearch.first()
    }
}

/// L-BFGS optimizer of [Liu and Nocedal (1989)](https://doi.org/10.1007/BF01589116), which preconditions the gradients
/// with a limited memory approximation of the inverse Hessian and chooses its step sizes with a line search, and which
/// is the analogue of [`optax.lbfgs`](https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.lbfgs). Its
/// update is
///
/// ```text
/// sₜ = θₜ₋₁ − θₜ₋₂
/// yₜ = gₜ − gₜ₋₁
/// ρₜ = 1 / ⟨yₜ, sₜ⟩
/// γₜ = ⟨yₜ, sₜ⟩ / ‖yₜ‖²
/// q  ← gₜ
/// aᵢ ← ρᵢ · ⟨sᵢ, q⟩ and then q ← q − aᵢ · yᵢ          for i from t down to t − M + 1
/// p  ← γₜ · q
/// bᵢ ← ρᵢ · ⟨yᵢ, p⟩ and then p ← p + (aᵢ − bᵢ) · sᵢ   for i from t − M + 1 up to t
/// uₜ = −αₜ · ηₜ · p
/// ```
///
/// Here `t ≥ 1` counts the updates including the current one, `gₜ` is the gradient at the parameters `θₜ₋₁`, `uₜ` is
/// the update that is added to them (i.e., `θₜ = θₜ₋₁ + uₜ`), and `ηₜ` is the learning rate [`Schedule`] evaluated at
/// the `t − 1` updates applied before the current one. The differences `sₜ` of the parameters and `yₜ` of the gradients
/// are recorded for the last `M` updates (i.e., the memory size) with the weights `ρₜ` (which are zero where
/// `⟨yₜ, sₜ⟩ = 0`), the inner products `⟨·, ·⟩` and norms `‖·‖` are taken over all leaves, and the two loop recursion
/// of Nocedal and Wright (2006, Algorithm 7.4) computes the product `p` of the approximation of the inverse Hessian and
/// `gₜ`, starting from the scaled identity `γₜ · I`. The differences are zero at the first update, where
/// `γ₁ = min(1, 1 / ‖g₁‖)`, and `γₜ = 1` when `‖yₜ‖ = 0`. The step size `αₜ` is chosen by the line search along
/// `−ηₜ · p`, or is `1` without one. The defaults are `ηₜ = 1`, `M = 10` with the scaled initial preconditioner, and
/// the zoom line search with at most `20` steps and [`InitialGuessStrategy::One`]. [`Self::with_learning_rate`] sets
/// the learning rate schedule, [`Self::with_memory_size`] sets `M`, [`Self::with_scale_initial_preconditioner`] with
/// `false` uses `γₜ = 1`, and [`Self::with_linesearch`] replaces the line search (e.g., with
/// [`ScaleByBacktrackingLinesearch`](crate::ScaleByBacktrackingLinesearch)) or removes it.
///
/// The line search needs the [`UpdateArguments`] that it documents (e.g., the value and the gradients of the objective
/// at the current parameters and the objective with its gradients for the zoom line search). Optax applies
/// `optax.scale(-1)` without a learning rate, where this optimizer applies the constant learning rate `1` with a
/// [`ScaleBySchedule`], which adds a step count to the state and gives the same updates.
///
/// It is the chain of [`ScaleByLbfgs`], [`ScaleBySchedule::learning_rate`], and the optional line search (e.g.,
/// [`ScaleByZoomLinesearch`]).
///
/// # References
///
///   - Liu and Nocedal (1989). [On the limited memory BFGS method for large scale
///     optimization](https://doi.org/10.1007/BF01589116).
///   - Nocedal and Wright (2006). [Numerical Optimization](https://doi.org/10.1007/978-0-387-40065-5), second edition.
#[derive(Clone, Debug, PartialEq)]
pub struct Lbfgs<Sc: Schedule = f64, L = ScaleByZoomLinesearch> {
    /// Transformation that preconditions the gradients.
    lbfgs: ScaleByLbfgs,

    /// Transformation that scales the preconditioned gradients by the negated learning rate.
    learning_rate: ScaleBySchedule<Sc>,

    /// Line search that scales the descent direction, if any.
    linesearch: Option<L>,
}

impl Lbfgs {
    /// Creates a new [`Lbfgs`] with default values for all hyperparameters.
    pub fn new() -> Self {
        Self {
            lbfgs: ScaleByLbfgs::new(),
            learning_rate: ScaleBySchedule::learning_rate(1.0),
            linesearch: Some(
                ScaleByZoomLinesearch::new(20).unwrap().with_initial_guess_strategy(InitialGuessStrategy::One),
            ),
        }
    }
}

impl Default for Lbfgs {
    fn default() -> Self {
        Self::new()
    }
}

impl<Sc: Schedule, L> Lbfgs<Sc, L> {
    /// Returns this optimizer with the learning rate schedule `learning_rate` (e.g., an `f64` for a constant learning
    /// rate).
    #[inline]
    pub fn with_learning_rate<T: Schedule>(self, learning_rate: T) -> Lbfgs<T, L> {
        Lbfgs {
            lbfgs: self.lbfgs,
            learning_rate: ScaleBySchedule::learning_rate(learning_rate),
            linesearch: self.linesearch,
        }
    }

    /// Returns this optimizer with `memory_size` as the number of past differences that it keeps in memory.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `memory_size` is zero.
    pub fn with_memory_size(mut self, memory_size: usize) -> Result<Self, OptimizerError> {
        self.lbfgs = self.lbfgs.with_memory_size(memory_size)?;
        Ok(self)
    }

    /// Returns this optimizer with the scaled identity as its initial preconditioner when
    /// `scale_initial_preconditioner` is `true` and the identity otherwise.
    #[inline]
    pub fn with_scale_initial_preconditioner(mut self, scale_initial_preconditioner: bool) -> Self {
        self.lbfgs = self.lbfgs.with_scale_initial_preconditioner(scale_initial_preconditioner);
        self
    }

    /// Returns this optimizer with `linesearch` as its line search, or without a line search when it is [`None`].
    #[inline]
    pub fn with_linesearch<T>(self, linesearch: Option<T>) -> Lbfgs<Sc, T> {
        Lbfgs { lbfgs: self.lbfgs, learning_rate: self.learning_rate, linesearch }
    }

    /// Returns the [`ScaleByLbfgs`] transformation of this optimizer.
    #[inline]
    pub fn lbfgs(&self) -> &ScaleByLbfgs {
        &self.lbfgs
    }

    /// Returns the learning rate [`Schedule`] of this optimizer.
    #[inline]
    pub fn learning_rate(&self) -> &Sc {
        self.learning_rate.schedule()
    }

    /// Returns the line search of this optimizer, if any.
    #[inline]
    pub fn linesearch(&self) -> Option<&L> {
        self.linesearch.as_ref()
    }
}

impl<P: OptimizerValue, Sc: Schedule, L: Optimizer<P>> Optimizer<P> for Lbfgs<Sc, L> {
    type State<S: Parameterized<P>> = LbfgsState<P, S, L::State<S>>;

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        Ok(LbfgsState {
            lbfgs: self.lbfgs.initialize(parameters)?,
            learning_rate: self.learning_rate.initialize(parameters)?,
            linesearch: self
                .linesearch
                .iter()
                .map(|linesearch| linesearch.initialize(parameters))
                .collect::<Result<_, _>>()?,
        })
    }

    fn update<S: Parameterized<P>>(
        &self,
        gradients: S,
        state: Self::State<S>,
        parameters: &S,
        arguments: &UpdateArguments<'_, P, S>,
    ) -> Result<(S, Self::State<S>), OptimizerError> {
        let LbfgsState { lbfgs, learning_rate, linesearch } = state;
        let (updates, lbfgs) = self.lbfgs.update(gradients, lbfgs, parameters, arguments)?;
        let (mut updates, learning_rate) = self.learning_rate.update(updates, learning_rate, parameters, arguments)?;
        let mut linesearch_states = Vec::with_capacity(linesearch.len());
        if let (Some(search), Some(linesearch)) = (&self.linesearch, linesearch.into_iter().next()) {
            let (searched, linesearch) = search.update(updates, linesearch, parameters, arguments)?;
            updates = searched;
            linesearch_states.push(linesearch);
        }
        Ok((updates, LbfgsState { lbfgs, learning_rate, linesearch: linesearch_states }))
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use ryft_core::{Array, Mul, Reduce, ReductionKind};

    use crate::tests::{assert_parity, parity_quadratic_updates, state_paths};
    use crate::{ScaleByBacktrackingLinesearch, apply_updates};

    use super::*;

    /// Returns the parameters after `steps` updates of `optimizer` from `(1, 1)` on the quadratic
    /// `f(x) = ½ · (x₁² + 2 · x₂²)`, whose minimizer is `(0, 0)`, together with the values of `f` before every update.
    fn minimize<O: Optimizer<Array>>(optimizer: &O, steps: usize) -> (Vec<f32>, Vec<f32>) {
        let curvatures = Array::vector(vec![1.0f32, 2.0]).unwrap();
        let objective = |parameters: &Vec<Array>| -> Result<(Array, Vec<Array>), OptimizerError> {
            let gradients = Mul::mul(&curvatures, &parameters[0])?;
            let value = Mul::mul(&gradients, &parameters[0])?.reduce(&[0], ReductionKind::Sum)?;
            Ok((Mul::mul(&value, &Array::scalar(0.5f32)?)?, vec![gradients]))
        };
        let value_objective =
            |parameters: &Vec<Array>| -> Result<Array, OptimizerError> { Ok(objective(parameters)?.0) };
        let mut parameters = vec![Array::vector(vec![1.0f32, 1.0]).unwrap()];
        let mut state = optimizer.initialize(&parameters).unwrap();
        let mut values = Vec::new();
        for _ in 0..steps {
            let (value, gradients) = objective(&parameters).unwrap();
            values.push(value.elements::<f32>().unwrap()[0]);
            let arguments = UpdateArguments::new()
                .with_value(&value)
                .with_gradients(&gradients)
                .with_objective(&value_objective)
                .with_objective_with_gradients(&objective);
            let (updates, next_state) = optimizer.update(gradients.clone(), state, &parameters, &arguments).unwrap();
            parameters = apply_updates(parameters, updates).unwrap();
            state = next_state;
        }
        (parameters[0].elements::<f32>().unwrap(), values)
    }

    #[test]
    fn test_lbfgs_state_linesearch() {
        let parameters = vec![Array::scalar(1.0f32).unwrap()];
        let state = Lbfgs::new().initialize(&parameters).unwrap();
        assert!(state.linesearch().is_some());
        let state = Lbfgs::new().with_linesearch(None::<ScaleByZoomLinesearch>).initialize(&parameters).unwrap();
        assert!(state.linesearch().is_none());
    }

    #[test]
    fn test_lbfgs_new() {
        let optimizer = Lbfgs::new();
        assert_eq!(optimizer.lbfgs(), &ScaleByLbfgs::new());
        assert_eq!(optimizer.learning_rate(), &1.0);
        assert_eq!(
            optimizer.linesearch(),
            Some(&ScaleByZoomLinesearch::new(20).unwrap().with_initial_guess_strategy(InitialGuessStrategy::One)),
        );
        assert_eq!(Lbfgs::default(), optimizer);
    }

    #[test]
    fn test_lbfgs_with_hyperparameters() {
        let optimizer = Lbfgs::new()
            .with_learning_rate(0.5)
            .with_memory_size(3)
            .unwrap()
            .with_scale_initial_preconditioner(false)
            .with_linesearch(Some(ScaleByBacktrackingLinesearch::new(10)));
        assert_eq!(optimizer.learning_rate(), &0.5);
        assert_eq!(optimizer.lbfgs().memory_size(), 3);
        assert!(!optimizer.lbfgs().scale_initial_preconditioner());
        assert_eq!(optimizer.linesearch(), Some(&ScaleByBacktrackingLinesearch::new(10)));
        assert!(matches!(
            Lbfgs::new().with_memory_size(0),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`memory_size` must be positive but got `0`",
        ));
    }

    #[test]
    fn test_lbfgs_initialize() {
        let paths = state_paths(&Lbfgs::new().with_memory_size(2).unwrap());
        assert_eq!(paths[0], "$.lbfgs.count");
        assert_eq!(paths[10], "$.learning_rate.count");
        assert_eq!(paths[11], "$.linesearch[0].learning_rate");
        assert_eq!(paths.len(), 18);
    }

    #[test]
    fn test_lbfgs_update() {
        // Without a line search, the first update is `-(1, 2) / √5` (refer to `ScaleByLbfgs`).
        let (parameters, _) = minimize(&Lbfgs::new().with_linesearch(None::<ScaleByZoomLinesearch>), 1);
        assert!((parameters[0] - (1.0 - 1.0 / 5f32.sqrt())).abs() < 1e-6);
        assert!((parameters[1] - (1.0 - 2.0 / 5f32.sqrt())).abs() < 1e-6);

        // With the zoom line search, the objective decreases at every step and reaches the minimizer `(0, 0)`.
        let (parameters, values) = minimize(&Lbfgs::new(), 8);
        assert!(values.windows(2).all(|pair| pair[1] <= pair[0]), "{values:?}");
        assert!(parameters.iter().all(|parameter| parameter.abs() < 1e-4), "{parameters:?}");

        // So does it with the backtracking line search.
        let optimizer = Lbfgs::new().with_linesearch(Some(ScaleByBacktrackingLinesearch::new(20)));
        let (parameters, values) = minimize(&optimizer, 20);
        assert!(values.windows(2).all(|pair| pair[1] <= pair[0]), "{values:?}");
        assert!(parameters.iter().all(|parameter| parameter.abs() < 1e-3), "{parameters:?}");
    }

    #[test]
    fn test_lbfgs_optax_parity() {
        // Optax 9ecde04b: `optax.lbfgs()`, on the parity quadratic.
        const LBFGS: [[f32; 9]; 4] = [
            [0.061134644, 0.48907715, -0.36680788, -0.0, 0.18340394, 0.13755295, 0.73361576, -0.1528366, 0.09170197],
            [0.3051175, 1.5755347, -0.5325965, -0.0, 0.9153525, 0.56481665, 2.3633013, -0.35713497, 0.13314913],
            [
                0.06929869,
                -0.053464264,
                -0.15313381,
                -0.0,
                0.20789605,
                0.040015135,
                -0.080195576,
                0.010259286,
                0.038283452,
            ],
            [
                0.051127367,
                -0.012152201,
                0.03547733,
                -0.0,
                0.15338208,
                0.017887862,
                -0.018228143,
                -0.0026873348,
                -0.008869349,
            ],
        ];
        // Optax 9ecde04b: `optax.lbfgs(learning_rate=0.1, linesearch=None)`, on the parity quadratic.
        const LBFGS_WITHOUT_LINESEARCH: [[f32; 9]; 5] = [
            [
                0.0061134645,
                0.048907716,
                -0.036680788,
                -0.0,
                0.018340394,
                0.0137552945,
                0.073361576,
                -0.015283661,
                0.009170197,
            ],
            [0.036013864, 0.20157018, -0.0862722, -0.0, 0.1080417, 0.068861336, 0.30235562, -0.049468886, 0.021568146],
            [0.03913537, 0.1758203, -0.092676185, -0.0, 0.11740615, 0.065763354, 0.26373047, -0.043458644, 0.023169016],
            [0.03860979, 0.15870826, -0.07738377, -0.0, 0.11582937, 0.060231682, 0.23806217, -0.039662324, 0.019345937],
            [0.036414057, 0.14189716, -0.07148985, -0.0, 0.109242104, 0.05426351, 0.2128458, -0.035216924, 0.017872429],
        ];
        assert_parity(&parity_quadratic_updates(&Lbfgs::new(), 4), &LBFGS);
        let optimizer = Lbfgs::new().with_learning_rate(0.1).with_linesearch(None::<ScaleByZoomLinesearch>);
        assert_parity(&parity_quadratic_updates(&optimizer, 5), &LBFGS_WITHOUT_LINESEARCH);
    }
}
