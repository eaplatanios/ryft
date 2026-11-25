//! Line searches, which scale the updates by a step size that decreases the objective sufficiently.
//!
//! [`ScaleByBacktrackingLinesearch`] tries decreasing step sizes until the objective decreases sufficiently (i.e., the
//! Armijo criterion), and [`ScaleByZoomLinesearch`] brackets and then zooms into an interval of step sizes until one
//! also satisfies the curvature condition (i.e., the strong Wolfe conditions). Optax runs its line searches as
//! `jax.lax.while_loop`s that call the objective. The objective of [`UpdateArguments`] is a function of the leaf type
//! `P`, which cannot be called on the values that `ryft-core` traces the body of a staged loop with, so the line
//! searches of this crate unroll their loops to their maximum number of iterations instead. Every iteration evaluates
//! the objective, and iterations after the search has finished keep its results through selections. The cost of an
//! update is therefore always the maximum number of objective evaluations, rather than the number that the search
//! needs.

use ryft_core::{Add, ComparisonDirection, Div, Mul, Neg, Not, Parameter, Parameterized, Sub};

use crate::macros::parameterized_state;
use crate::{
    Optimizer, OptimizerError, OptimizerValue, UpdateArguments, convert_to, filled_statistics, initial_count,
    inner_product, statistic_data_type, validate_non_negative, validate_positive, validate_structure,
};

// TODO(eaplatanios): Review this module.

/// Information about the last search of [`ScaleByBacktrackingLinesearch`], which is the analogue of
/// [`optax.BacktrackingLinesearchInfo`](
/// https://optax.readthedocs.io/en/latest/api/transformations.html#optax.BacktrackingLinesearchInfo).
#[derive(Clone, Debug, PartialEq, Eq, Hash, ryft_macros::Parameterized)]
#[ryft(crate = "ryft_core")]
pub struct BacktrackingLinesearchInfo<P: Parameter> {
    /// Rank-0 `i32` array that holds the number of step sizes that the last search tried, which is the
    /// `num_linesearch_steps` of Optax.
    linesearch_step_count: P,

    /// Rank-0 array that holds the error of the sufficient decrease criterion at the step size that the last search
    /// ended with, which is zero when the search succeeded and positive when it failed.
    decrease_error: P,
}

impl<P: Parameter> BacktrackingLinesearchInfo<P> {
    /// Returns the rank-0 `i32` array that holds the number of step sizes that the last search tried.
    #[inline]
    pub fn linesearch_step_count(&self) -> &P {
        &self.linesearch_step_count
    }

    /// Returns the rank-0 array that holds the error of the sufficient decrease criterion at the end of the last
    /// search.
    #[inline]
    pub fn decrease_error(&self) -> &P {
        &self.decrease_error
    }
}

/// State of [`ScaleByBacktrackingLinesearch`], which is the analogue of
/// [`optax.ScaleByBacktrackingLinesearchState`](
/// https://optax.readthedocs.io/en/latest/api/transformations.html#optax.ScaleByBacktrackingLinesearchState)
/// without its optional gradient. Its rank-0 arrays have the statistic data type of the first parameter (i.e., `f64`
/// for `f64` parameters and `f32` otherwise), except for the step count of the search information.
#[derive(Clone, Debug, PartialEq, Eq, Hash, ryft_macros::Parameterized)]
#[ryft(crate = "ryft_core")]
pub struct ScaleByBacktrackingLinesearchState<P: Parameter> {
    /// Rank-0 array that holds the learning rate that the last search found, which is `1` initially.
    learning_rate: P,

    /// Rank-0 array that holds the value of the objective at the step size that the last search ended with, which is
    /// infinite initially.
    value: P,

    /// Information about the last search.
    info: BacktrackingLinesearchInfo<P>,
}

impl<P: Parameter> ScaleByBacktrackingLinesearchState<P> {
    /// Returns the rank-0 array that holds the learning rate that the last search found.
    #[inline]
    pub fn learning_rate(&self) -> &P {
        &self.learning_rate
    }

    /// Returns the rank-0 array that holds the value of the objective at the step size that the last search ended with.
    #[inline]
    pub fn value(&self) -> &P {
        &self.value
    }

    /// Returns the information about the last search.
    #[inline]
    pub fn info(&self) -> &BacktrackingLinesearchInfo<P> {
        &self.info
    }
}

/// Transformation that scales the updates `u` by a learning rate `η` that satisfies the sufficient decrease criterion
/// (i.e., the Armijo criterion) `f(w + η · u) ≤ (1 + δ) · f(w) + η · c · ⟨u, ∇f(w)⟩ + ε` at the parameters `w`, which
/// is the analogue of
/// [`optax.scale_by_backtracking_linesearch`](
/// https://optax.readthedocs.io/en/latest/api/transformations.html#optax.scale_by_backtracking_linesearch),
/// where `c` is the slope relative tolerance, `δ` is the relative tolerance, and `ε` is the absolute tolerance. The
/// search starts from the learning rate of the previous search multiplied by the increase factor and clipped at the
/// maximum learning rate, and multiplies it by the decrease factor until the criterion holds or until it has tried
/// `1 + maximum_backtracking_steps` learning rates. When the objective is NaN or infinite at the last learning rate
/// that it tries, the learning rate is `0`. The updates must be a descent direction (i.e., `⟨u, ∇f(w)⟩ < 0`, e.g.,
/// negated gradients) for the criterion to be satisfiable.
///
/// The update requires the `value`, `gradients`, and `objective` [`UpdateArguments`], which are the value and the
/// gradients of the objective at the current parameters and the objective itself. The search is unrolled (refer to the
/// [module documentation](self)), so every update evaluates the objective `1 + maximum_backtracking_steps` times. The
/// `store_grad` option of Optax, which reuses the gradient at the accepted step size in the next update, needs the
/// gradient of the objective at other parameters and is not provided.
#[derive(Clone, Debug, PartialEq)]
pub struct ScaleByBacktrackingLinesearch {
    /// Maximum number of times that the learning rate is decreased.
    maximum_backtracking_steps: usize,

    /// Relative tolerance `c` with respect to the slope `⟨u, ∇f(w)⟩`.
    slope_relative_tolerance: f64,

    /// Factor by which the learning rate is decreased.
    decrease_factor: f64,

    /// Factor by which the learning rate of the previous search is increased to start the next one.
    increase_factor: f64,

    /// Maximum learning rate from which a search starts.
    maximum_learning_rate: f64,

    /// Absolute tolerance `ε` of the criterion.
    absolute_tolerance: f64,

    /// Relative tolerance `δ` of the criterion.
    relative_tolerance: f64,
}

impl ScaleByBacktrackingLinesearch {
    /// Creates a new [`ScaleByBacktrackingLinesearch`] that decreases the learning rate at most
    /// `maximum_backtracking_steps` times, with the slope relative tolerance `c = 1e-4`, the decrease factor `0.8`, the
    /// increase factor `1.5`, the maximum learning rate `1`, and no absolute or relative tolerance.
    #[inline]
    pub fn new(maximum_backtracking_steps: usize) -> Self {
        Self {
            maximum_backtracking_steps,
            slope_relative_tolerance: 1e-4,
            decrease_factor: 0.8,
            increase_factor: 1.5,
            maximum_learning_rate: 1.0,
            absolute_tolerance: 0.0,
            relative_tolerance: 0.0,
        }
    }

    /// Returns this transformation with `slope_relative_tolerance` as the relative tolerance `c` with respect to the
    /// slope.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `slope_relative_tolerance` is negative or not finite.
    pub fn with_slope_relative_tolerance(mut self, slope_relative_tolerance: f64) -> Result<Self, OptimizerError> {
        validate_non_negative("slope_relative_tolerance", slope_relative_tolerance)?;
        self.slope_relative_tolerance = slope_relative_tolerance;
        Ok(self)
    }

    /// Returns this transformation with `decrease_factor` as the factor by which it decreases the learning rate.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `decrease_factor` is not positive and finite.
    pub fn with_decrease_factor(mut self, decrease_factor: f64) -> Result<Self, OptimizerError> {
        validate_positive("decrease_factor", decrease_factor)?;
        self.decrease_factor = decrease_factor;
        Ok(self)
    }

    /// Returns this transformation with `increase_factor` as the factor by which it increases the learning rate of the
    /// previous search to start the next one. `1` starts every search from the previous learning rate, and infinity
    /// starts every search from the maximum learning rate.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `increase_factor` is not positive or is NaN.
    pub fn with_increase_factor(mut self, increase_factor: f64) -> Result<Self, OptimizerError> {
        if increase_factor.is_nan() || increase_factor <= 0.0 {
            return Err(OptimizerError::InvalidConfiguration {
                message: format!("`increase_factor` must be positive but got `{increase_factor}`"),
            });
        }
        self.increase_factor = increase_factor;
        Ok(self)
    }

    /// Returns this transformation with `maximum_learning_rate` as the maximum learning rate from which it starts a
    /// search.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `maximum_learning_rate` is not positive and finite.
    pub fn with_maximum_learning_rate(mut self, maximum_learning_rate: f64) -> Result<Self, OptimizerError> {
        validate_positive("maximum_learning_rate", maximum_learning_rate)?;
        self.maximum_learning_rate = maximum_learning_rate;
        Ok(self)
    }

    /// Returns this transformation with `absolute_tolerance` as the absolute tolerance `ε` of its criterion.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `absolute_tolerance` is negative or not finite.
    pub fn with_absolute_tolerance(mut self, absolute_tolerance: f64) -> Result<Self, OptimizerError> {
        validate_non_negative("absolute_tolerance", absolute_tolerance)?;
        self.absolute_tolerance = absolute_tolerance;
        Ok(self)
    }

    /// Returns this transformation with `relative_tolerance` as the relative tolerance `δ` of its criterion.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `relative_tolerance` is negative or not finite.
    pub fn with_relative_tolerance(mut self, relative_tolerance: f64) -> Result<Self, OptimizerError> {
        validate_non_negative("relative_tolerance", relative_tolerance)?;
        self.relative_tolerance = relative_tolerance;
        Ok(self)
    }

    /// Returns the maximum number of times that this transformation decreases the learning rate in one search.
    #[inline]
    pub fn maximum_backtracking_steps(&self) -> usize {
        self.maximum_backtracking_steps
    }

    /// Returns the relative tolerance `c` of this transformation with respect to the slope.
    #[inline]
    pub fn slope_relative_tolerance(&self) -> f64 {
        self.slope_relative_tolerance
    }

    /// Returns the factor by which this transformation decreases the learning rate.
    #[inline]
    pub fn decrease_factor(&self) -> f64 {
        self.decrease_factor
    }

    /// Returns the factor by which this transformation increases the learning rate of the previous search.
    #[inline]
    pub fn increase_factor(&self) -> f64 {
        self.increase_factor
    }

    /// Returns the maximum learning rate from which this transformation starts a search.
    #[inline]
    pub fn maximum_learning_rate(&self) -> f64 {
        self.maximum_learning_rate
    }

    /// Returns the absolute tolerance `ε` of the criterion of this transformation.
    #[inline]
    pub fn absolute_tolerance(&self) -> f64 {
        self.absolute_tolerance
    }

    /// Returns the relative tolerance `δ` of the criterion of this transformation.
    #[inline]
    pub fn relative_tolerance(&self) -> f64 {
        self.relative_tolerance
    }

    /// Returns the error `max(f(w + η · u) - (1 + δ) · f(w) - η · c · s, 0)` of the sufficient decrease criterion for
    /// the learning rate `η`, the slope `s`, the value `f(w)`, and the new value `f(w + η · u)`, which is infinite when
    /// it would be NaN.
    fn decrease_error<P: OptimizerValue>(
        &self,
        learning_rate: &P,
        slope: &P,
        value: &P,
        new_value: &P,
    ) -> Result<P, OptimizerError> {
        let relative_value = Mul::mul(&value.fill_like(1.0 + self.relative_tolerance)?, value)?;
        let slope_term =
            Mul::mul(&Mul::mul(learning_rate, &learning_rate.fill_like(self.slope_relative_tolerance)?)?, slope)?;
        let error = Sub::sub(&Sub::sub(new_value, &relative_value)?, &slope_term)?;
        let error = error
            .compare(&error, ComparisonDirection::NotEqual)?
            .select(&error.fill_like(f64::INFINITY)?, &error)?;
        Ok(error.max(&error.fill_like(0f64)?)?)
    }
}

impl<P: OptimizerValue> Optimizer<P> for ScaleByBacktrackingLinesearch {
    type State<S: Parameterized<P>> = ScaleByBacktrackingLinesearchState<P>;

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        let count = initial_count("ScaleByBacktrackingLinesearch", parameters)?;
        let data_type = statistic_data_type(parameters.parameters().next().unwrap().r#type().data_type());
        let scalar = convert_to(&count, data_type)?;
        Ok(ScaleByBacktrackingLinesearchState {
            learning_rate: scalar.fill_like(1f64)?,
            value: scalar.fill_like(f64::INFINITY)?,
            info: BacktrackingLinesearchInfo {
                linesearch_step_count: count,
                decrease_error: scalar.fill_like(f64::INFINITY)?,
            },
        })
    }

    fn update<S: Parameterized<P>>(
        &self,
        gradients: S,
        state: Self::State<S>,
        parameters: &S,
        arguments: &UpdateArguments<'_, P, S>,
    ) -> Result<(S, Self::State<S>), OptimizerError> {
        // This follows `optax.scale_by_backtracking_linesearch` with its `while_loop` unrolled to its maximum number of
        // iterations, where an iteration is active when the previous one did not satisfy the criterion.
        let name = "ScaleByBacktrackingLinesearch";
        let value = arguments.validate_value(name)?;
        let objective_gradients = arguments.validate_gradients(name)?;
        let objective = arguments.validate_objective(name)?;
        let paths = gradients.parameter_paths().collect::<Vec<_>>();
        validate_structure("parameters", &paths, parameters)?;
        validate_structure("gradients", &paths, objective_gradients)?;
        let data_type = state.learning_rate.r#type().data_type();
        let updates = gradients.parameters().cloned().collect::<Vec<_>>();
        let objective_gradients = objective_gradients.parameters().cloned().collect::<Vec<_>>();
        let slope = convert_to(&inner_product("slope", &updates, &objective_gradients)?, data_type)?;
        let value = convert_to(value, data_type)?;

        let maximum_learning_rate = state.learning_rate.fill_like(self.maximum_learning_rate)?;
        let mut learning_rate = Mul::mul(&state.learning_rate.fill_like(self.increase_factor)?, &state.learning_rate)?
            .min(&maximum_learning_rate)?;
        let mut new_value = value.clone();
        let mut decrease_error = value.fill_like(f64::INFINITY)?;
        let mut steps = state.info.linesearch_step_count.fill_like(0i32)?;
        let absolute_tolerance = decrease_error.fill_like(self.absolute_tolerance)?;
        for iteration in 0..=self.maximum_backtracking_steps {
            let active = Not::not(&decrease_error.compare(&absolute_tolerance, ComparisonDirection::LessThanOrEqual)?)?;
            let candidate_learning_rate = if iteration > 0 {
                Mul::mul(&learning_rate.fill_like(self.decrease_factor)?, &learning_rate)?
            } else {
                learning_rate.clone()
            };
            let candidate_parameters = parameters
                .parameters()
                .zip(&updates)
                .map(|(parameter, update)| {
                    let parameter_type = parameter.r#type().data_type();
                    let statistic_type = statistic_data_type(parameter_type);
                    let step = Mul::mul(
                        &convert_to(&candidate_learning_rate, statistic_type)?,
                        &convert_to(update, statistic_type)?,
                    )?;
                    Ok(convert_to(&Add::add(&convert_to(parameter, statistic_type)?, &step)?, parameter_type)?)
                })
                .collect::<Result<Vec<_>, OptimizerError>>()?;
            let candidate_value = convert_to(
                &objective(&S::from_parameters(parameters.parameter_structure(), candidate_parameters)?)?,
                data_type,
            )?;
            let candidate_error = self.decrease_error(&candidate_learning_rate, &slope, &value, &candidate_value)?;
            learning_rate = active.select(&candidate_learning_rate, &learning_rate)?;
            new_value = active.select(&candidate_value, &new_value)?;
            decrease_error = active.select(&candidate_error, &decrease_error)?;
            steps = active.select(&Add::add(&steps, &steps.fill_like(1i32)?)?, &steps)?;
        }

        let infinite = decrease_error.compare(&decrease_error.fill_like(f64::INFINITY)?, ComparisonDirection::Equal)?;
        let learning_rate = infinite.select(&learning_rate.fill_like(0f64)?, &learning_rate)?;
        let structure = gradients.parameter_structure();
        let updates = updates
            .into_iter()
            .map(|update| {
                let update_type = update.r#type().data_type();
                let statistic_type = statistic_data_type(update_type);
                let update =
                    Mul::mul(&convert_to(&learning_rate, statistic_type)?, &convert_to(&update, statistic_type)?)?;
                Ok(convert_to(&update, update_type)?)
            })
            .collect::<Result<Vec<_>, OptimizerError>>()?;
        let state = ScaleByBacktrackingLinesearchState {
            learning_rate,
            value: new_value,
            info: BacktrackingLinesearchInfo { linesearch_step_count: steps, decrease_error },
        };
        Ok((S::from_parameters(structure, updates)?, state))
    }
}

/// Strategy for the first step size that [`ScaleByZoomLinesearch`] tries, which is the `initial_guess_strategy` of
/// [`optax.scale_by_zoom_linesearch`](
/// https://optax.readthedocs.io/en/latest/api/transformations.html#optax.scale_by_zoom_linesearch).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InitialGuessStrategy {
    /// Starts every search from the step size `1`, which suits quasi-Newton directions (e.g., those of L-BFGS).
    One,

    /// Starts every search from the step size that the previous search found.
    Keep,
}

/// Information about the last search of [`ScaleByZoomLinesearch`], which is the analogue of
/// [`optax.ZoomLinesearchInfo`](
/// https://optax.readthedocs.io/en/latest/api/transformations.html#optax.ZoomLinesearchInfo).
#[derive(Clone, Debug, PartialEq, Eq, Hash, ryft_macros::Parameterized)]
#[ryft(crate = "ryft_core")]
pub struct ZoomLinesearchInfo<P: Parameter> {
    /// Rank-0 `i32` array that holds the number of step sizes that the last search tried, which is the
    /// `num_linesearch_steps` of Optax.
    linesearch_step_count: P,

    /// Rank-0 array that holds the error of the sufficient decrease criterion at the step size that the last search
    /// ended with.
    decrease_error: P,

    /// Rank-0 array that holds the error of the curvature criterion at the step size that the last search ended with.
    curvature_error: P,
}

impl<P: Parameter> ZoomLinesearchInfo<P> {
    /// Returns the rank-0 `i32` array that holds the number of step sizes that the last search tried.
    #[inline]
    pub fn linesearch_step_count(&self) -> &P {
        &self.linesearch_step_count
    }

    /// Returns the rank-0 array that holds the error of the sufficient decrease criterion at the end of the last
    /// search.
    #[inline]
    pub fn decrease_error(&self) -> &P {
        &self.decrease_error
    }

    /// Returns the rank-0 array that holds the error of the curvature criterion at the end of the last search.
    #[inline]
    pub fn curvature_error(&self) -> &P {
        &self.curvature_error
    }
}

parameterized_state! {
    /// State of [`ScaleByZoomLinesearch`], which is the analogue of
    /// [`optax.ScaleByZoomLinesearchState`](
    /// https://optax.readthedocs.io/en/latest/api/transformations.html#optax.ScaleByZoomLinesearchState).
    /// Its rank-0 arrays have the statistic data type of the first parameter, except for the step count of the search
    /// information, and its gradients have the structure `S` of the parameters.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct ScaleByZoomLinesearchState<P, S> with family ScaleByZoomLinesearchStateFamily {
        /// Rank-0 array that holds the learning rate that the last search found, which is `1` initially.
        learning_rate: P,

        /// Rank-0 array that holds the value of the objective at the learning rate that the last search found, which
        /// is infinite initially.
        value: P,

        /// Gradients of the objective at the learning rate that the last search found, which are zero initially.
        gradients: S,

        /// Information about the last search.
        info: ZoomLinesearchInfo<P>,
    }
}

impl<P: Parameter, S> ScaleByZoomLinesearchState<P, S> {
    /// Returns the rank-0 array that holds the learning rate that the last search found.
    #[inline]
    pub fn learning_rate(&self) -> &P {
        &self.learning_rate
    }

    /// Returns the rank-0 array that holds the value of the objective at the learning rate that the last search found.
    #[inline]
    pub fn value(&self) -> &P {
        &self.value
    }

    /// Returns the gradients of the objective at the learning rate that the last search found, which a caller can use
    /// as the value and the gradients at the next parameters of a deterministic objective, as
    /// `optax.value_and_grad_from_state` does.
    #[inline]
    pub fn gradients(&self) -> &S {
        &self.gradients
    }

    /// Returns the information about the last search.
    #[inline]
    pub fn info(&self) -> &ZoomLinesearchInfo<P> {
        &self.info
    }
}

/// Transformation that scales the updates `u` by a learning rate `η` that satisfies the strong Wolfe conditions (i.e.,
/// sufficient decrease and small curvature) at the parameters `w`, found with the zoom line search of
/// [Nocedal and Wright (2006)](https://doi.org/10.1007/978-0-387-40065-5) (Algorithms 3.5 and 3.6), which is the
/// analogue of
/// [`optax.scale_by_zoom_linesearch`](
/// https://optax.readthedocs.io/en/latest/api/transformations.html#optax.scale_by_zoom_linesearch).
/// The search first increases the step size by the increase factor until it brackets an interval that contains an
/// acceptable step size, and then zooms into that interval with safeguarded cubic or quadratic interpolation or with
/// bisection. With the slope relative tolerance `c₁`, the curvature relative tolerance `c₂`, and the slope
/// `s(η) = ⟨∇f(w + η · u), u⟩`, the step size is accepted when `f(w + η · u) ≤ f(w) + c₁ · η · s(0)` (or the
/// approximate sufficient decrease condition of [Hager and Zhang (2006)](https://doi.org/10.1145/1132973.1132979),
/// which tolerates rounding errors near the minimum, with the relative tolerance `1e-6` by default) and
/// `|s(η)| ≤ c₂ · |s(0)|`, up to the tolerance. When the search fails, it falls back to the best step size that
/// decreased the objective sufficiently, if any. The defaults are `c₁ = 1e-4`, `c₂ = 0.9`, the increase factor `2`,
/// no maximum learning rate, the tolerance `0`, the interval precision `1e-5`, and [`InitialGuessStrategy::Keep`].
///
/// The update requires the `value`, `gradients`, and `objective_with_gradients` [`UpdateArguments`], because Optax
/// differentiates the objective at every step size that it tries. The state records the value and the gradients at
/// the accepted step size (refer to [`ScaleByZoomLinesearchState::gradients`]). The search is unrolled (refer to the
/// [module documentation](self)), so every update evaluates the objective and its gradients `maximum_linesearch_steps`
/// times. Each iteration evaluates one step size, chosen by whether the interval was found before it, and both of the
/// bracketing and the zooming updates of the search state are computed from it and selected.
#[derive(Clone, Debug, PartialEq)]
pub struct ScaleByZoomLinesearch {
    /// Maximum number of step sizes that a search tries.
    maximum_linesearch_steps: usize,

    /// Maximum learning rate, or [`None`] if the step size is not bounded.
    maximum_learning_rate: Option<f64>,

    /// Tolerance of the errors of the conditions.
    tolerance: f64,

    /// Factor by which the step size is increased while the interval is searched for.
    increase_factor: f64,

    /// Relative tolerance `c₁` with respect to the slope in the sufficient decrease condition.
    slope_relative_tolerance: f64,

    /// Relative tolerance `c₂` with respect to the slope in the curvature condition.
    curvature_relative_tolerance: f64,

    /// Relative tolerance of the approximate sufficient decrease condition, or [`None`] if it is not used.
    approximate_decrease_relative_tolerance: Option<f64>,

    /// Length of the interval below which the search is considered to have failed.
    stepsize_precision: f64,

    /// Strategy for the first step size of every search.
    initial_guess_strategy: InitialGuessStrategy,
}

impl ScaleByZoomLinesearch {
    /// Creates a new [`ScaleByZoomLinesearch`] that tries at most `maximum_linesearch_steps` step sizes per search,
    /// with default values for all other hyperparameters.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `maximum_linesearch_steps` is zero.
    pub fn new(maximum_linesearch_steps: usize) -> Result<Self, OptimizerError> {
        if maximum_linesearch_steps == 0 {
            return Err(OptimizerError::InvalidConfiguration {
                message: "`maximum_linesearch_steps` must be positive but got `0`".to_string(),
            });
        }
        Ok(Self {
            maximum_linesearch_steps,
            maximum_learning_rate: None,
            tolerance: 0.0,
            increase_factor: 2.0,
            slope_relative_tolerance: 1e-4,
            curvature_relative_tolerance: 0.9,
            approximate_decrease_relative_tolerance: Some(1e-6),
            stepsize_precision: 1e-5,
            initial_guess_strategy: InitialGuessStrategy::Keep,
        })
    }

    /// Returns this transformation with `maximum_learning_rate` as its maximum learning rate, or without one when it is
    /// [`None`].
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `maximum_learning_rate` is not positive and finite.
    pub fn with_maximum_learning_rate(mut self, maximum_learning_rate: Option<f64>) -> Result<Self, OptimizerError> {
        if let Some(maximum_learning_rate) = maximum_learning_rate {
            validate_positive("maximum_learning_rate", maximum_learning_rate)?;
        }
        self.maximum_learning_rate = maximum_learning_rate;
        Ok(self)
    }

    /// Returns this transformation with `tolerance` as the tolerance of the errors of its conditions.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `tolerance` is negative or not finite.
    pub fn with_tolerance(mut self, tolerance: f64) -> Result<Self, OptimizerError> {
        validate_non_negative("tolerance", tolerance)?;
        self.tolerance = tolerance;
        Ok(self)
    }

    /// Returns this transformation with `increase_factor` as the factor by which it increases the step size while it
    /// searches for an interval.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `increase_factor` is not positive and finite.
    pub fn with_increase_factor(mut self, increase_factor: f64) -> Result<Self, OptimizerError> {
        validate_positive("increase_factor", increase_factor)?;
        self.increase_factor = increase_factor;
        Ok(self)
    }

    /// Returns this transformation with `slope_relative_tolerance` as the relative tolerance `c₁` of its sufficient
    /// decrease condition.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `slope_relative_tolerance` is negative or not finite.
    pub fn with_slope_relative_tolerance(mut self, slope_relative_tolerance: f64) -> Result<Self, OptimizerError> {
        validate_non_negative("slope_relative_tolerance", slope_relative_tolerance)?;
        self.slope_relative_tolerance = slope_relative_tolerance;
        Ok(self)
    }

    /// Returns this transformation with `curvature_relative_tolerance` as the relative tolerance `c₂` of its curvature
    /// condition.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `curvature_relative_tolerance` is negative or not finite.
    pub fn with_curvature_relative_tolerance(
        mut self,
        curvature_relative_tolerance: f64,
    ) -> Result<Self, OptimizerError> {
        validate_non_negative("curvature_relative_tolerance", curvature_relative_tolerance)?;
        self.curvature_relative_tolerance = curvature_relative_tolerance;
        Ok(self)
    }

    /// Returns this transformation with `approximate_decrease_relative_tolerance` as the relative tolerance of its
    /// approximate sufficient decrease condition, or without that condition when it is [`None`].
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when the tolerance is negative or not finite.
    pub fn with_approximate_decrease_relative_tolerance(
        mut self,
        approximate_decrease_relative_tolerance: Option<f64>,
    ) -> Result<Self, OptimizerError> {
        if let Some(tolerance) = approximate_decrease_relative_tolerance {
            validate_non_negative("approximate_decrease_relative_tolerance", tolerance)?;
        }
        self.approximate_decrease_relative_tolerance = approximate_decrease_relative_tolerance;
        Ok(self)
    }

    /// Returns this transformation with `stepsize_precision` as the length of the interval below which a search is
    /// considered to have failed.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `stepsize_precision` is negative or not finite.
    pub fn with_stepsize_precision(mut self, stepsize_precision: f64) -> Result<Self, OptimizerError> {
        validate_non_negative("stepsize_precision", stepsize_precision)?;
        self.stepsize_precision = stepsize_precision;
        Ok(self)
    }

    /// Returns this transformation with `initial_guess_strategy` as the strategy for the first step size of every
    /// search.
    #[inline]
    pub fn with_initial_guess_strategy(mut self, initial_guess_strategy: InitialGuessStrategy) -> Self {
        self.initial_guess_strategy = initial_guess_strategy;
        self
    }

    /// Returns the maximum number of step sizes that a search of this transformation tries.
    #[inline]
    pub fn maximum_linesearch_steps(&self) -> usize {
        self.maximum_linesearch_steps
    }

    /// Returns the maximum learning rate of this transformation, if any.
    #[inline]
    pub fn maximum_learning_rate(&self) -> Option<f64> {
        self.maximum_learning_rate
    }

    /// Returns the tolerance of the errors of the conditions of this transformation.
    #[inline]
    pub fn tolerance(&self) -> f64 {
        self.tolerance
    }

    /// Returns the factor by which this transformation increases the step size while it searches for an interval.
    #[inline]
    pub fn increase_factor(&self) -> f64 {
        self.increase_factor
    }

    /// Returns the relative tolerance `c₁` of the sufficient decrease condition of this transformation.
    #[inline]
    pub fn slope_relative_tolerance(&self) -> f64 {
        self.slope_relative_tolerance
    }

    /// Returns the relative tolerance `c₂` of the curvature condition of this transformation.
    #[inline]
    pub fn curvature_relative_tolerance(&self) -> f64 {
        self.curvature_relative_tolerance
    }

    /// Returns the relative tolerance of the approximate sufficient decrease condition of this transformation, if any.
    #[inline]
    pub fn approximate_decrease_relative_tolerance(&self) -> Option<f64> {
        self.approximate_decrease_relative_tolerance
    }

    /// Returns the length of the interval below which a search of this transformation is considered to have failed.
    #[inline]
    pub fn stepsize_precision(&self) -> f64 {
        self.stepsize_precision
    }

    /// Returns the strategy for the first step size of every search of this transformation.
    #[inline]
    pub fn initial_guess_strategy(&self) -> InitialGuessStrategy {
        self.initial_guess_strategy
    }

    /// Returns the error of the sufficient decrease condition at the step size `stepsize` with the value `value` and
    /// the slope `slope`, given the value `initial_value` and the slope `initial_slope` at the step size `0`, which is
    /// `_compute_decrease_error` of Optax.
    fn decrease_error<P: OptimizerValue>(
        &self,
        stepsize: &P,
        value: &P,
        slope: &P,
        initial_value: &P,
        initial_slope: &P,
    ) -> Result<P, OptimizerError> {
        let slope_term =
            Mul::mul(&Mul::mul(&stepsize.fill_like(self.slope_relative_tolerance)?, stepsize)?, initial_slope)?;
        let mut error = Sub::sub(&Sub::sub(value, initial_value)?, &slope_term)?;
        if let Some(tolerance) = self.approximate_decrease_relative_tolerance {
            let approximate_error = Sub::sub(
                slope,
                &Mul::mul(&slope.fill_like(2.0 * self.slope_relative_tolerance - 1.0)?, initial_slope)?,
            )?;
            let value_difference = Sub::sub(
                &Sub::sub(value, initial_value)?,
                &Mul::mul(&value.fill_like(tolerance)?, &initial_value.abs()?)?,
            )?;
            error = approximate_error.max(&value_difference)?.min(&error)?;
        }
        not_nan_or_infinity(&error.max(&error.fill_like(0f64)?)?)
    }

    /// Returns the error of the curvature condition for the slope `slope` and the slope `initial_slope` at the step
    /// size `0`, which is `_compute_curvature_error` of Optax.
    fn curvature_error<P: OptimizerValue>(&self, slope: &P, initial_slope: &P) -> Result<P, OptimizerError> {
        let error = Sub::sub(
            &slope.abs()?,
            &Mul::mul(&slope.fill_like(self.curvature_relative_tolerance)?, &initial_slope.abs()?)?,
        )?;
        not_nan_or_infinity(&error.max(&error.fill_like(0f64)?)?)
    }
}

impl<P: OptimizerValue> Optimizer<P> for ScaleByZoomLinesearch {
    type State<S: Parameterized<P>> = ScaleByZoomLinesearchState<P, S>;

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        let count = initial_count("ScaleByZoomLinesearch", parameters)?;
        let data_type = statistic_data_type(parameters.parameters().next().unwrap().r#type().data_type());
        let scalar = convert_to(&count, data_type)?;
        Ok(ScaleByZoomLinesearchState {
            learning_rate: scalar.fill_like(1f64)?,
            value: scalar.fill_like(f64::INFINITY)?,
            gradients: filled_statistics(parameters, 0.0, None)?,
            info: ZoomLinesearchInfo {
                linesearch_step_count: count,
                decrease_error: scalar.fill_like(f64::INFINITY)?,
                curvature_error: scalar.fill_like(f64::INFINITY)?,
            },
        })
    }

    fn update<S: Parameterized<P>>(
        &self,
        gradients: S,
        state: Self::State<S>,
        parameters: &S,
        arguments: &UpdateArguments<'_, P, S>,
    ) -> Result<(S, Self::State<S>), OptimizerError> {
        // This follows `zoom_linesearch` and `scale_by_zoom_linesearch` of Optax with their `while_loop` unrolled to
        // its maximum number of iterations, where an iteration is active when the search has neither finished nor
        // failed.
        let name = "ScaleByZoomLinesearch";
        let value = arguments.validate_value(name)?;
        let objective_gradients = arguments.validate_gradients(name)?;
        let objective_with_gradients = arguments.validate_objective_with_gradients(name)?;
        let paths = gradients.parameter_paths().collect::<Vec<_>>();
        validate_structure("parameters", &paths, parameters)?;
        validate_structure("gradients", &paths, objective_gradients)?;
        validate_structure("gradients", &paths, &state.gradients)?;
        let data_type = state.learning_rate.r#type().data_type();
        let updates = gradients.parameters().cloned().collect::<Vec<_>>();
        let initial_gradients = objective_gradients
            .parameters()
            .zip(state.gradients.parameters())
            .map(|(gradient, stored)| convert_to(gradient, stored.r#type().data_type()))
            .collect::<Result<Vec<_>, _>>()?;
        let initial_slope = convert_to(&inner_product("slope", &updates, &initial_gradients)?, data_type)?;
        let initial_value = convert_to(value, data_type)?;
        let zero = initial_value.fill_like(0f64)?;
        let stepsize_guess = match self.initial_guess_strategy {
            InitialGuessStrategy::One => zero.fill_like(1f64)?,
            InitialGuessStrategy::Keep => state.learning_rate.clone(),
        };
        let mut search = ZoomSearch::new(
            &initial_value,
            &initial_slope,
            &initial_gradients,
            state.info.linesearch_step_count.fill_like(0i32)?,
        )?;
        let tolerance = zero.fill_like(self.tolerance)?;
        let maximum_steps = search.count.fill_like(self.maximum_linesearch_steps as i32)?;
        for _ in 0..self.maximum_linesearch_steps {
            let active = Not::not(&search.done.or(&search.failed)?)?;
            let first = search.count.compare(&search.count.fill_like(0i32)?, ComparisonDirection::Equal)?;
            let last = Add::add(&search.count, &search.count.fill_like(1i32)?)?
                .compare(&maximum_steps, ComparisonDirection::GreaterThanOrEqual)?;

            // Step size of the interval search.
            let larger_stepsize = Mul::mul(&zero.fill_like(self.increase_factor)?, &search.stepsize)?;
            let mut search_stepsize = first.select(&stepsize_guess, &larger_stepsize)?;
            let maximum_stepsize_reached = match self.maximum_learning_rate {
                Some(maximum) => {
                    let maximum = zero.fill_like(maximum)?;
                    let reached = search_stepsize.compare(&maximum, ComparisonDirection::GreaterThanOrEqual)?;
                    search_stepsize = search_stepsize.min(&maximum)?;
                    reached
                }
                None => first.compare(&first, ComparisonDirection::NotEqual)?,
            };

            // Step size of the zoom into the interval.
            let delta = Sub::sub(&search.high, &search.low)?.abs()?;
            let left = search.high.min(&search.low)?;
            let right = search.high.max(&search.low)?;
            let cubic_check = Mul::mul(&zero.fill_like(0.2)?, &delta)?;
            let quadratic_check = Mul::mul(&zero.fill_like(0.1)?, &delta)?;
            let too_small_interval =
                delta.compare(&zero.fill_like(self.stepsize_precision)?, ComparisonDirection::LessThanOrEqual)?;
            let cubic_middle = cubic_minimum(
                &search.low,
                &search.value_low,
                &search.slope_low,
                &search.high,
                &search.value_high,
                &search.cubic_reference,
                &search.value_cubic_reference,
            )?;
            let use_cubic = within(&cubic_middle, &left, &right, &cubic_check)?;
            let quadratic_middle =
                quadratic_minimum(&search.low, &search.value_low, &search.slope_low, &search.high, &search.value_high)?;
            let use_quadratic =
                Not::not(&use_cubic)?.and(&within(&quadratic_middle, &left, &right, &quadratic_check)?)?;
            let bisection_middle = Div::div(&Add::add(&search.low, &search.high)?, &zero.fill_like(2f64)?)?;
            let use_bisection = Not::not(&use_cubic.or(&use_quadratic)?)?;
            let middle = use_cubic.select(&cubic_middle, &search.cubic_reference)?;
            let middle = use_quadratic.select(&quadratic_middle, &middle)?;
            let middle = use_bisection.select(&bisection_middle, &middle)?;

            // Evaluate the objective and its gradients at the step size of the current phase of the search.
            let stepsize = search.interval_found.select(&middle, &search_stepsize)?;
            let candidate = parameters
                .parameters()
                .zip(&updates)
                .map(|(parameter, update)| {
                    let parameter_type = parameter.r#type().data_type();
                    let statistic_type = statistic_data_type(parameter_type);
                    let step = Mul::mul(&convert_to(&stepsize, statistic_type)?, &convert_to(update, statistic_type)?)?;
                    Ok(convert_to(&Add::add(&convert_to(parameter, statistic_type)?, &step)?, parameter_type)?)
                })
                .collect::<Result<Vec<_>, OptimizerError>>()?;
            let (new_value, new_gradients) =
                objective_with_gradients(&S::from_parameters(parameters.parameter_structure(), candidate)?)?;
            validate_structure("gradients", &paths, &new_gradients)?;
            let new_value = convert_to(&new_value, data_type)?;
            let new_gradients = new_gradients
                .into_parameters()
                .zip(&initial_gradients)
                .map(|(gradient, initial)| convert_to(&gradient, initial.r#type().data_type()))
                .collect::<Result<Vec<_>, _>>()?;
            let new_slope = convert_to(&inner_product("slope", &new_gradients, &updates)?, data_type)?;
            let decrease_error =
                self.decrease_error(&stepsize, &new_value, &new_slope, &initial_value, &initial_slope)?;
            let curvature_error = self.curvature_error(&new_slope, &initial_slope)?;
            let error = decrease_error.max(&curvature_error)?;
            let satisfied = error.compare(&tolerance, ComparisonDirection::LessThanOrEqual)?;
            let safe_decrease = decrease_error.compare(&tolerance, ComparisonDirection::LessThanOrEqual)?;
            let positive_decrease_error = decrease_error.compare(&zero, ComparisonDirection::GreaterThan)?;

            // Update of the interval search (i.e., `_search_interval`).
            let interval = {
                let mut next = search.clone();
                let after_first = Not::not(&first)?;
                let set_high_to_new = positive_decrease_error.or(&new_value
                    .compare(&search.value, ComparisonDirection::GreaterThanOrEqual)?
                    .and(&after_first)?)?;
                let set_low_to_new = new_slope
                    .compare(&zero, ComparisonDirection::GreaterThanOrEqual)?
                    .and(&Not::not(&set_high_to_new)?)?;
                next.low = set_low_to_new.select(&stepsize, &search.stepsize)?;
                next.value_low = set_low_to_new.select(&new_value, &search.value)?;
                next.slope_low = set_low_to_new.select(&new_slope, &search.slope)?;
                next.high = set_low_to_new.select(&search.stepsize, &stepsize)?;
                next.value_high = set_low_to_new.select(&search.value, &new_value)?;
                next.slope_high = set_low_to_new.select(&search.slope, &new_slope)?;
                next.cubic_reference = next.low.clone();
                next.value_cubic_reference = next.value_low.clone();
                next.interval_found = set_high_to_new.or(&set_low_to_new)?.or(&satisfied)?;
                next.done = satisfied.or(&maximum_stepsize_reached.and(&Not::not(&next.interval_found)?)?)?;
                next.failed = last.and(&Not::not(&next.done)?)?;
                next.select_safe(&safe_decrease, &stepsize, &new_value, &new_gradients)?;
                next
            };

            // Update of the zoom into the interval (i.e., `_zoom_into_interval`).
            let zoom = {
                let mut next = search.clone();
                let update_safe =
                    safe_decrease.and(&new_value.compare(&search.safe_value, ComparisonDirection::LessThan)?)?;
                next.select_safe(&update_safe, &stepsize, &new_value, &new_gradients)?;
                next.done = satisfied.clone();
                let set_high_to_middle = positive_decrease_error
                    .or(&new_value.compare(&search.value_low, ComparisonDirection::GreaterThanOrEqual)?)?;
                let secant = Mul::mul(&new_slope, &Sub::sub(&search.high, &search.low)?)?;
                let set_high_to_low = secant
                    .compare(&zero, ComparisonDirection::GreaterThanOrEqual)?
                    .and(&Not::not(&set_high_to_middle)?)?;
                let set_low_to_middle = Not::not(&set_high_to_middle)?;
                let high = set_high_to_middle.select(&stepsize, &search.high)?;
                let value_high = set_high_to_middle.select(&new_value, &search.value_high)?;
                let slope_high = set_high_to_middle.select(&new_slope, &search.slope_high)?;
                next.high = set_high_to_low.select(&search.low, &high)?;
                next.value_high = set_high_to_low.select(&search.value_low, &value_high)?;
                next.slope_high = set_high_to_low.select(&search.slope_low, &slope_high)?;
                next.low = set_low_to_middle.select(&stepsize, &search.low)?;
                next.value_low = set_low_to_middle.select(&new_value, &search.value_low)?;
                next.slope_low = set_low_to_middle.select(&new_slope, &search.slope_low)?;
                let high_moved = set_high_to_middle.or(&set_high_to_low)?;
                next.cubic_reference = high_moved.select(&search.high, &search.low)?;
                next.value_cubic_reference = high_moved.select(&search.value_high, &search.value_low)?;
                let positive_safe_stepsize = next.safe_stepsize.compare(&zero, ComparisonDirection::GreaterThan)?;
                let presumably_failed = last.or(&too_small_interval.and(&positive_safe_stepsize)?)?;
                next.failed = presumably_failed.and(&Not::not(&next.done)?)?;
                next
            };

            let mut next = ZoomSearch::select(&search.interval_found, zoom, interval)?;
            next.count = Add::add(&search.count, &search.count.fill_like(1i32)?)?;
            next.stepsize = stepsize;
            next.value = new_value;
            next.gradients = new_gradients;
            next.slope = new_slope;
            next.decrease_error = decrease_error;
            next.curvature_error = curvature_error;

            // A failed search falls back to the best step size with a sufficient decrease (i.e., `_try_safe_step`).
            let outside_domain =
                next.decrease_error.compare(&zero.fill_like(f64::INFINITY)?, ComparisonDirection::Equal)?;
            let use_safe = next
                .failed
                .and(&next.safe_stepsize.compare(&zero, ComparisonDirection::GreaterThan)?.or(&outside_domain)?)?;
            next.stepsize = use_safe.select(&next.safe_stepsize, &next.stepsize)?;
            next.value = use_safe.select(&next.safe_value, &next.value)?;
            next.gradients = next
                .gradients
                .iter()
                .zip(&next.safe_gradients)
                .map(|(gradient, safe)| Ok(use_safe.select(safe, gradient)?))
                .collect::<Result<Vec<_>, OptimizerError>>()?;
            search = ZoomSearch::select(&active, next, search)?;
        }

        let structure = gradients.parameter_structure();
        let scaled = updates
            .into_iter()
            .map(|update| {
                let update_type = update.r#type().data_type();
                let statistic_type = statistic_data_type(update_type);
                let update =
                    Mul::mul(&convert_to(&search.stepsize, statistic_type)?, &convert_to(&update, statistic_type)?)?;
                Ok(convert_to(&update, update_type)?)
            })
            .collect::<Result<Vec<_>, OptimizerError>>()?;
        let state = ScaleByZoomLinesearchState {
            learning_rate: search.stepsize,
            value: search.value,
            gradients: S::from_parameters(structure.clone(), search.gradients)?,
            info: ZoomLinesearchInfo {
                linesearch_step_count: search.count,
                decrease_error: search.decrease_error,
                curvature_error: search.curvature_error,
            },
        };
        Ok((S::from_parameters(structure, scaled)?, state))
    }
}

/// State of one search of [`ScaleByZoomLinesearch`] while its loop is unrolled, which is the `ZoomLinesearchState` of
/// Optax without the parameters, the updates, and the initial value and slope, which do not change during a search.
/// Scalars are rank-0 arrays of the data type of the learning rate, except for the `i32` count and the boolean flags.
#[derive(Clone)]
struct ZoomSearch<P> {
    /// Number of step sizes tried so far.
    count: P,

    /// Last step size tried.
    stepsize: P,

    /// Value of the objective at the last step size.
    value: P,

    /// Gradients of the objective at the last step size.
    gradients: Vec<P>,

    /// Slope of the objective along the updates at the last step size.
    slope: P,

    /// Error of the sufficient decrease condition at the last step size.
    decrease_error: P,

    /// Error of the curvature condition at the last step size.
    curvature_error: P,

    /// Whether an interval that contains an acceptable step size was found.
    interval_found: P,

    /// Whether the last step size satisfies both conditions or the search cannot continue.
    done: P,

    /// Whether the search failed.
    failed: P,

    /// Step size at the low end of the interval.
    low: P,

    /// Value of the objective at the low end of the interval.
    value_low: P,

    /// Slope of the objective at the low end of the interval.
    slope_low: P,

    /// Step size at the high end of the interval.
    high: P,

    /// Value of the objective at the high end of the interval.
    value_high: P,

    /// Slope of the objective at the high end of the interval.
    slope_high: P,

    /// Third step size of the cubic interpolation.
    cubic_reference: P,

    /// Value of the objective at the third step size of the cubic interpolation.
    value_cubic_reference: P,

    /// Best step size so far that satisfies the sufficient decrease condition, or `0` if there is none.
    safe_stepsize: P,

    /// Value of the objective at the safe step size.
    safe_value: P,

    /// Gradients of the objective at the safe step size.
    safe_gradients: Vec<P>,
}

impl<P: OptimizerValue> ZoomSearch<P> {
    /// Creates the state of a search at the step size `0`, with the value `value`, the slope `slope`, and the gradients
    /// `gradients` there and the `i32` zero `count`.
    fn new(value: &P, slope: &P, gradients: &[P], count: P) -> Result<Self, OptimizerError> {
        let zero = value.fill_like(0f64)?;
        let infinity = value.fill_like(f64::INFINITY)?;
        let no = count.compare(&count, ComparisonDirection::NotEqual)?;
        Ok(Self {
            count,
            stepsize: zero.clone(),
            value: value.clone(),
            gradients: gradients.to_vec(),
            slope: slope.clone(),
            decrease_error: infinity.clone(),
            curvature_error: infinity,
            interval_found: no.clone(),
            done: no.clone(),
            failed: no,
            low: zero.clone(),
            value_low: value.clone(),
            slope_low: slope.clone(),
            high: zero.clone(),
            value_high: value.clone(),
            slope_high: slope.clone(),
            cubic_reference: zero.clone(),
            value_cubic_reference: value.clone(),
            safe_stepsize: zero,
            safe_value: value.clone(),
            safe_gradients: gradients.to_vec(),
        })
    }

    /// Replaces the safe step size, value, and gradients with `stepsize`, `value`, and `gradients` where `condition`
    /// holds.
    fn select_safe(&mut self, condition: &P, stepsize: &P, value: &P, gradients: &[P]) -> Result<(), OptimizerError> {
        self.safe_stepsize = condition.select(stepsize, &self.safe_stepsize)?;
        self.safe_value = condition.select(value, &self.safe_value)?;
        self.safe_gradients = gradients
            .iter()
            .zip(&self.safe_gradients)
            .map(|(gradient, safe)| Ok(condition.select(gradient, safe)?))
            .collect::<Result<Vec<_>, OptimizerError>>()?;
        Ok(())
    }

    /// Returns the state that selects every field of `on_true` where `condition` holds and of `on_false` otherwise.
    fn select(condition: &P, on_true: Self, on_false: Self) -> Result<Self, OptimizerError> {
        let select = |on_true: &P, on_false: &P| condition.select(on_true, on_false);
        let select_all = |on_true: &[P], on_false: &[P]| {
            on_true
                .iter()
                .zip(on_false)
                .map(|(on_true, on_false)| select(on_true, on_false))
                .collect::<Result<Vec<_>, _>>()
        };
        Ok(Self {
            count: select(&on_true.count, &on_false.count)?,
            stepsize: select(&on_true.stepsize, &on_false.stepsize)?,
            value: select(&on_true.value, &on_false.value)?,
            gradients: select_all(&on_true.gradients, &on_false.gradients)?,
            slope: select(&on_true.slope, &on_false.slope)?,
            decrease_error: select(&on_true.decrease_error, &on_false.decrease_error)?,
            curvature_error: select(&on_true.curvature_error, &on_false.curvature_error)?,
            interval_found: select(&on_true.interval_found, &on_false.interval_found)?,
            done: select(&on_true.done, &on_false.done)?,
            failed: select(&on_true.failed, &on_false.failed)?,
            low: select(&on_true.low, &on_false.low)?,
            value_low: select(&on_true.value_low, &on_false.value_low)?,
            slope_low: select(&on_true.slope_low, &on_false.slope_low)?,
            high: select(&on_true.high, &on_false.high)?,
            value_high: select(&on_true.value_high, &on_false.value_high)?,
            slope_high: select(&on_true.slope_high, &on_false.slope_high)?,
            cubic_reference: select(&on_true.cubic_reference, &on_false.cubic_reference)?,
            value_cubic_reference: select(&on_true.value_cubic_reference, &on_false.value_cubic_reference)?,
            safe_stepsize: select(&on_true.safe_stepsize, &on_false.safe_stepsize)?,
            safe_value: select(&on_true.safe_value, &on_false.safe_value)?,
            safe_gradients: select_all(&on_true.safe_gradients, &on_false.safe_gradients)?,
        })
    }
}

/// Returns `value`, or infinity where it is NaN.
fn not_nan_or_infinity<P: OptimizerValue>(value: &P) -> Result<P, OptimizerError> {
    Ok(value
        .compare(value, ComparisonDirection::NotEqual)?
        .select(&value.fill_like(f64::INFINITY)?, value)?)
}

/// Returns whether `point` lies strictly inside `[left + margin, right - margin]`.
fn within<P: OptimizerValue>(point: &P, left: &P, right: &P, margin: &P) -> Result<P, OptimizerError> {
    let above = point.compare(&Add::add(left, margin)?, ComparisonDirection::GreaterThan)?;
    Ok(above.and(&point.compare(&Sub::sub(right, margin)?, ComparisonDirection::LessThan)?)?)
}

/// Returns the minimizer of the cubic that interpolates the value `fa` and the slope `fpa` at `a` and the values `fb`
/// at `b` and `fc` at `c`, which is NaN or infinite when there is none, as `_cubicmin` of Optax computes it.
fn cubic_minimum<P: OptimizerValue>(a: &P, fa: &P, fpa: &P, b: &P, fb: &P, c: &P, fc: &P) -> Result<P, OptimizerError> {
    let db = Sub::sub(b, a)?;
    let dc = Sub::sub(c, a)?;
    let db2 = Mul::mul(&db, &db)?;
    let dc2 = Mul::mul(&dc, &dc)?;
    let dbdc = Mul::mul(&db, &dc)?;
    let denominator = Mul::mul(&Mul::mul(&dbdc, &dbdc)?, &Sub::sub(&db, &dc)?)?;
    let residual_b = Sub::sub(&Sub::sub(fb, fa)?, &Mul::mul(fpa, &db)?)?;
    let residual_c = Sub::sub(&Sub::sub(fc, fa)?, &Mul::mul(fpa, &dc)?)?;
    let cubic = Div::div(&Sub::sub(&Mul::mul(&dc2, &residual_b)?, &Mul::mul(&db2, &residual_c)?)?, &denominator)?;
    let quadratic = Div::div(
        &Sub::sub(&Mul::mul(&Mul::mul(&db2, &db)?, &residual_c)?, &Mul::mul(&Mul::mul(&dc2, &dc)?, &residual_b)?)?,
        &denominator,
    )?;
    let radical =
        Sub::sub(&Mul::mul(&quadratic, &quadratic)?, &Mul::mul(&Mul::mul(&a.fill_like(3f64)?, &cubic)?, fpa)?)?;
    let numerator = Add::add(&Neg::neg(&quadratic)?, &radical.sqrt()?)?;
    Ok(Add::add(a, &Div::div(&numerator, &Mul::mul(&a.fill_like(3f64)?, &cubic)?)?)?)
}

/// Returns the minimizer of the quadratic that interpolates the value `fa` and the slope `fpa` at `a` and the value
/// `fb` at `b`, as `_quadmin` of Optax computes it.
fn quadratic_minimum<P: OptimizerValue>(a: &P, fa: &P, fpa: &P, b: &P, fb: &P) -> Result<P, OptimizerError> {
    let db = Sub::sub(b, a)?;
    let curvature = Div::div(&Sub::sub(&Sub::sub(fb, fa)?, &Mul::mul(fpa, &db)?)?, &Mul::mul(&db, &db)?)?;
    Ok(Sub::sub(a, &Div::div(fpa, &Mul::mul(&a.fill_like(2f64)?, &curvature)?)?)?)
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use ryft_core::{Array, FillLike, Neg, Reduce, ReductionKind};

    use crate::tests::{assert_parity, parity_quadratic_updates, state_paths};
    use crate::{Chain, Scale, apply_updates};

    use super::*;

    /// Returns the value of the quadratic objective `Σᵢ xᵢ²` at the single vector leaf of `parameters`.
    // Objectives take the parameter structure itself (i.e., `&Vec<Array>`), so the reference cannot be a slice.
    #[allow(clippy::ptr_arg)]
    fn quadratic(parameters: &Vec<Array>) -> Result<Array, OptimizerError> {
        Ok(Mul::mul(&parameters[0], &parameters[0])?.reduce(&[0], ReductionKind::Sum)?)
    }

    /// Returns the gradient `2 · x` of [`quadratic`] at `parameters`.
    fn quadratic_gradients(parameters: &[Array]) -> Vec<Array> {
        vec![Mul::mul(&parameters[0].fill_like(2f64).unwrap(), &parameters[0]).unwrap()]
    }

    #[test]
    fn test_scale_by_backtracking_linesearch_new() {
        let linesearch = ScaleByBacktrackingLinesearch::new(15);
        assert_eq!(linesearch.maximum_backtracking_steps(), 15);
        assert_eq!(linesearch.slope_relative_tolerance(), 1e-4);
        assert_eq!(linesearch.decrease_factor(), 0.8);
        assert_eq!(linesearch.increase_factor(), 1.5);
        assert_eq!(linesearch.maximum_learning_rate(), 1.0);
        assert_eq!(linesearch.absolute_tolerance(), 0.0);
        assert_eq!(linesearch.relative_tolerance(), 0.0);
    }

    #[test]
    fn test_scale_by_backtracking_linesearch_with_hyperparameters() {
        let linesearch = ScaleByBacktrackingLinesearch::new(15)
            .with_slope_relative_tolerance(0.1)
            .unwrap()
            .with_decrease_factor(0.5)
            .unwrap()
            .with_increase_factor(f64::INFINITY)
            .unwrap()
            .with_maximum_learning_rate(2.0)
            .unwrap()
            .with_absolute_tolerance(1e-3)
            .unwrap()
            .with_relative_tolerance(1e-2)
            .unwrap();
        assert_eq!(linesearch.slope_relative_tolerance(), 0.1);
        assert_eq!(linesearch.decrease_factor(), 0.5);
        assert_eq!(linesearch.increase_factor(), f64::INFINITY);
        assert_eq!(linesearch.maximum_learning_rate(), 2.0);
        assert_eq!(linesearch.absolute_tolerance(), 1e-3);
        assert_eq!(linesearch.relative_tolerance(), 1e-2);
        assert!(matches!(
            ScaleByBacktrackingLinesearch::new(15).with_decrease_factor(0.0),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`decrease_factor` must be positive and finite but got `0`",
        ));
        assert!(matches!(
            ScaleByBacktrackingLinesearch::new(15).with_increase_factor(f64::NAN),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`increase_factor` must be positive but got `NaN`",
        ));
        assert!(matches!(
            ScaleByBacktrackingLinesearch::new(15).with_absolute_tolerance(-1.0),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`absolute_tolerance` must be finite and non-negative but got `-1`",
        ));
    }

    #[test]
    fn test_scale_by_backtracking_linesearch_initialize() {
        assert_eq!(
            state_paths(&ScaleByBacktrackingLinesearch::new(15)),
            vec!["$.learning_rate", "$.value", "$.info.linesearch_step_count", "$.info.decrease_error"],
        );
        let state = ScaleByBacktrackingLinesearch::new(15).initialize(&vec![Array::scalar(1.0f64).unwrap()]).unwrap();
        assert_eq!(state.learning_rate(), &Array::scalar(1.0f64).unwrap());
        assert_eq!(state.value(), &Array::scalar(f64::INFINITY).unwrap());
    }

    #[test]
    fn test_scale_by_backtracking_linesearch_update() {
        // On `f(x) = x₁² + x₂²` at `x = (1, 2)` with the direction `u = -∇f(x) = (-2, -4)`, the slope is `-20`. The
        // first learning rate is `min(1.5 · 1, 1) = 1`, at which `f(x + u) = 5` is not less than
        // `5 - 1 · 1e-4 · 20`, so the search tries `0.8`, at which `f = 0.6² + 1.2² = 1.8` is accepted.
        let linesearch = ScaleByBacktrackingLinesearch::new(15);
        let parameters = vec![Array::vector(vec![1.0f32, 2.0]).unwrap()];
        let value = quadratic(&parameters).unwrap();
        let objective_gradients = quadratic_gradients(&parameters);
        let objective = quadratic;
        let arguments = UpdateArguments::new()
            .with_value(&value)
            .with_gradients(&objective_gradients)
            .with_objective(&objective);
        let direction = vec![Neg::neg(&objective_gradients[0]).unwrap()];
        let state = linesearch.initialize(&parameters).unwrap();
        let (updates, state) = linesearch.update(direction, state, &parameters, &arguments).unwrap();
        assert_eq!(updates, vec![Array::vector(vec![-2.0f32 * 0.8, -4.0f32 * 0.8]).unwrap()]);
        assert_eq!(state.learning_rate(), &Array::scalar(0.8f32).unwrap());
        assert_eq!(state.info().linesearch_step_count(), &Array::scalar(2i32).unwrap());
        assert_eq!(state.info().decrease_error(), &Array::scalar(0.0f32).unwrap());
        assert!((state.value().elements::<f32>().unwrap()[0] - 1.8).abs() < 1e-5);

        // The next search starts from `min(1.5 · 0.8, 1) = 1` at `x = (-0.6, -1.2)`, where `x - ∇f(x) = (0.6, 1.2)`
        // has the same value, so it backtracks once more to `0.8`.
        let parameters = apply_updates(parameters, updates).unwrap();
        let value = quadratic(&parameters).unwrap();
        let objective_gradients = quadratic_gradients(&parameters);
        let arguments = UpdateArguments::new()
            .with_value(&value)
            .with_gradients(&objective_gradients)
            .with_objective(&objective);
        let direction = vec![Neg::neg(&objective_gradients[0]).unwrap()];
        let (_, state) = linesearch.update(direction, state, &parameters, &arguments).unwrap();
        assert_eq!(state.learning_rate(), &Array::scalar(0.8f32).unwrap());
        assert_eq!(state.info().linesearch_step_count(), &Array::scalar(2i32).unwrap());
    }

    #[test]
    fn test_scale_by_backtracking_linesearch_update_failure() {
        // An ascent direction never decreases the objective, so the search tries all `1 + 2` learning rates `1`,
        // `0.8`, and `0.64` and keeps the last one with a positive decrease error.
        let linesearch = ScaleByBacktrackingLinesearch::new(2);
        let parameters = vec![Array::vector(vec![1.0f32, 2.0]).unwrap()];
        let value = quadratic(&parameters).unwrap();
        let objective_gradients = quadratic_gradients(&parameters);
        let objective = quadratic;
        let arguments = UpdateArguments::new()
            .with_value(&value)
            .with_gradients(&objective_gradients)
            .with_objective(&objective);
        let state = linesearch.initialize(&parameters).unwrap();
        let (_, state) = linesearch.update(objective_gradients.clone(), state, &parameters, &arguments).unwrap();
        assert_eq!(state.learning_rate(), &Array::scalar(0.8f32 * 0.8f32).unwrap());
        assert_eq!(state.info().linesearch_step_count(), &Array::scalar(3i32).unwrap());
        assert!(state.info().decrease_error().elements::<f32>().unwrap()[0] > 0.0);

        // A NaN objective gives an infinite decrease error and a zero learning rate.
        let objective = |parameters: &Vec<Array>| -> Result<Array, OptimizerError> {
            Ok(quadratic(parameters)?.fill_like(f64::NAN)?)
        };
        let arguments = UpdateArguments::new()
            .with_value(&value)
            .with_gradients(&objective_gradients)
            .with_objective(&objective);
        let state = linesearch.initialize(&parameters).unwrap();
        let (updates, state) = linesearch.update(objective_gradients.clone(), state, &parameters, &arguments).unwrap();
        assert_eq!(updates, vec![Array::vector(vec![0.0f32, 0.0]).unwrap()]);
        assert_eq!(state.learning_rate(), &Array::scalar(0.0f32).unwrap());
        assert_eq!(state.info().decrease_error(), &Array::scalar(f32::INFINITY).unwrap());
    }

    #[test]
    fn test_scale_by_backtracking_linesearch_update_arguments() {
        // Each argument is required.
        let linesearch = ScaleByBacktrackingLinesearch::new(2);
        let parameters = vec![Array::vector(vec![1.0f32, 2.0]).unwrap()];
        let value = quadratic(&parameters).unwrap();
        let objective_gradients = quadratic_gradients(&parameters);
        let objective = quadratic;
        let state = linesearch.initialize(&parameters).unwrap();
        let arguments = UpdateArguments::new().with_gradients(&objective_gradients).with_objective(&objective);
        assert!(matches!(
            linesearch.update(parameters.clone(), state.clone(), &parameters, &arguments),
            Err(OptimizerError::MissingArgument { message })
                if message == "`ScaleByBacktrackingLinesearch` requires the `value` update argument, which was not \
                               provided",
        ));
        let arguments = UpdateArguments::new().with_value(&value).with_objective(&objective);
        assert!(matches!(
            linesearch.update(parameters.clone(), state.clone(), &parameters, &arguments),
            Err(OptimizerError::MissingArgument { .. }),
        ));
        let arguments = UpdateArguments::new().with_value(&value).with_gradients(&objective_gradients);
        assert!(matches!(
            linesearch.update(parameters.clone(), state, &parameters, &arguments),
            Err(OptimizerError::MissingArgument { .. }),
        ));
    }

    /// Returns the value and the gradients of [`quadratic`] at `parameters`.
    fn quadratic_with_gradients(parameters: &Vec<Array>) -> Result<(Array, Vec<Array>), OptimizerError> {
        Ok((quadratic(parameters)?, quadratic_gradients(parameters)))
    }

    #[test]
    fn test_scale_by_zoom_linesearch_new() {
        let linesearch = ScaleByZoomLinesearch::new(20).unwrap();
        assert_eq!(linesearch.maximum_linesearch_steps(), 20);
        assert_eq!(linesearch.maximum_learning_rate(), None);
        assert_eq!(linesearch.tolerance(), 0.0);
        assert_eq!(linesearch.increase_factor(), 2.0);
        assert_eq!(linesearch.slope_relative_tolerance(), 1e-4);
        assert_eq!(linesearch.curvature_relative_tolerance(), 0.9);
        assert_eq!(linesearch.approximate_decrease_relative_tolerance(), Some(1e-6));
        assert_eq!(linesearch.stepsize_precision(), 1e-5);
        assert_eq!(linesearch.initial_guess_strategy(), InitialGuessStrategy::Keep);
        assert!(matches!(
            ScaleByZoomLinesearch::new(0),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`maximum_linesearch_steps` must be positive but got `0`",
        ));
    }

    #[test]
    fn test_scale_by_zoom_linesearch_with_hyperparameters() {
        let linesearch = ScaleByZoomLinesearch::new(20)
            .unwrap()
            .with_maximum_learning_rate(Some(2.0))
            .unwrap()
            .with_tolerance(1e-3)
            .unwrap()
            .with_increase_factor(3.0)
            .unwrap()
            .with_slope_relative_tolerance(1e-3)
            .unwrap()
            .with_curvature_relative_tolerance(0.5)
            .unwrap()
            .with_approximate_decrease_relative_tolerance(None)
            .unwrap()
            .with_stepsize_precision(1e-4)
            .unwrap()
            .with_initial_guess_strategy(InitialGuessStrategy::One);
        assert_eq!(linesearch.maximum_learning_rate(), Some(2.0));
        assert_eq!(linesearch.tolerance(), 1e-3);
        assert_eq!(linesearch.increase_factor(), 3.0);
        assert_eq!(linesearch.slope_relative_tolerance(), 1e-3);
        assert_eq!(linesearch.curvature_relative_tolerance(), 0.5);
        assert_eq!(linesearch.approximate_decrease_relative_tolerance(), None);
        assert_eq!(linesearch.stepsize_precision(), 1e-4);
        assert_eq!(linesearch.initial_guess_strategy(), InitialGuessStrategy::One);
        assert!(matches!(
            ScaleByZoomLinesearch::new(20).unwrap().with_maximum_learning_rate(Some(0.0)),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`maximum_learning_rate` must be positive and finite but got `0`",
        ));
        assert!(matches!(
            ScaleByZoomLinesearch::new(20).unwrap().with_curvature_relative_tolerance(f64::NAN),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`curvature_relative_tolerance` must be finite and non-negative but got `NaN`",
        ));
    }

    #[test]
    fn test_scale_by_zoom_linesearch_initialize() {
        assert_eq!(
            state_paths(&ScaleByZoomLinesearch::new(20).unwrap()),
            vec![
                "$.learning_rate",
                "$.value",
                "$.gradients[\"bias\"]",
                "$.gradients[\"weights\"]",
                "$.info.linesearch_step_count",
                "$.info.decrease_error",
                "$.info.curvature_error",
            ],
        );
    }

    #[test]
    fn test_scale_by_zoom_linesearch_update() {
        // On `f(x) = x₁² + x₂²` at `x = (1, 2)` with `u = -∇f(x) = (-2, -4)`, the slope is `-20`. The first step size
        // `1` reaches `(-1, -2)`, where `f = 5` does not decrease enough, so it closes the interval `[0, 1]`. The cubic
        // interpolation is degenerate (its third point is `0` again), and the quadratic one through `f(0) = 5`,
        // `f'(0) = -20`, and `f(1) = 5` has its minimum at `0.5`, which reaches the minimizer `(0, 0)`, where both
        // conditions hold.
        let linesearch = ScaleByZoomLinesearch::new(20).unwrap();
        let parameters = vec![Array::vector(vec![1.0f32, 2.0]).unwrap()];
        let value = quadratic(&parameters).unwrap();
        let objective_gradients = quadratic_gradients(&parameters);
        let objective = quadratic_with_gradients;
        let arguments = UpdateArguments::new()
            .with_value(&value)
            .with_gradients(&objective_gradients)
            .with_objective_with_gradients(&objective);
        let direction = vec![Neg::neg(&objective_gradients[0]).unwrap()];
        let state = linesearch.initialize(&parameters).unwrap();
        let (updates, state) = linesearch.update(direction.clone(), state, &parameters, &arguments).unwrap();
        assert_eq!(updates, vec![Array::vector(vec![-1.0f32, -2.0]).unwrap()]);
        assert_eq!(state.learning_rate(), &Array::scalar(0.5f32).unwrap());
        assert_eq!(state.value(), &Array::scalar(0.0f32).unwrap());
        assert_eq!(state.gradients(), &vec![Array::vector(vec![0.0f32, 0.0]).unwrap()]);
        assert_eq!(state.info().linesearch_step_count(), &Array::scalar(2i32).unwrap());
        assert_eq!(state.info().decrease_error(), &Array::scalar(0.0f32).unwrap());
        assert_eq!(state.info().curvature_error(), &Array::scalar(0.0f32).unwrap());

        // With the initial guess `Keep`, the next search starts from `0.5`, which is accepted at once from the same
        // parameters, while `One` starts from `1` again.
        let (_, kept) = linesearch.update(direction.clone(), state.clone(), &parameters, &arguments).unwrap();
        assert_eq!(kept.info().linesearch_step_count(), &Array::scalar(1i32).unwrap());
        let linesearch = linesearch.with_initial_guess_strategy(InitialGuessStrategy::One);
        let (_, restarted) = linesearch.update(direction, state, &parameters, &arguments).unwrap();
        assert_eq!(restarted.info().linesearch_step_count(), &Array::scalar(2i32).unwrap());
    }

    #[test]
    fn test_scale_by_zoom_linesearch_update_failure() {
        // With the ascent direction `u = ∇f(x) = (2, 4)`, the step size `1` (where `f = 45`) closes the interval
        // `[0, 1]`, whose quadratic interpolation has its minimum at `-0.5`, outside of the interval, so the search
        // bisects it to `0.5` (where `f = 20`) and then to `0.25` (where `f = 11.25`). It fails after its three step
        // sizes without a safe step size and so makes the unsafe step `0.25`, as Optax does.
        let linesearch = ScaleByZoomLinesearch::new(3).unwrap();
        let parameters = vec![Array::vector(vec![1.0f32, 2.0]).unwrap()];
        let value = quadratic(&parameters).unwrap();
        let objective_gradients = quadratic_gradients(&parameters);
        let objective = quadratic_with_gradients;
        let arguments = UpdateArguments::new()
            .with_value(&value)
            .with_gradients(&objective_gradients)
            .with_objective_with_gradients(&objective);
        let state = linesearch.initialize(&parameters).unwrap();
        let (updates, state) = linesearch.update(objective_gradients.clone(), state, &parameters, &arguments).unwrap();
        assert_eq!(updates, vec![Array::vector(vec![0.5f32, 1.0]).unwrap()]);
        assert_eq!(state.learning_rate(), &Array::scalar(0.25f32).unwrap());
        assert_eq!(state.value(), &Array::scalar(11.25f32).unwrap());
        assert_eq!(state.info().linesearch_step_count(), &Array::scalar(3i32).unwrap());

        // An objective that is NaN everywhere is outside of its domain, so the search falls back to the step size `0`.
        let objective = |parameters: &Vec<Array>| -> Result<(Array, Vec<Array>), OptimizerError> {
            Ok((quadratic(parameters)?.fill_like(f64::NAN)?, quadratic_gradients(parameters)))
        };
        let arguments = UpdateArguments::new()
            .with_value(&value)
            .with_gradients(&objective_gradients)
            .with_objective_with_gradients(&objective);
        let state = linesearch.initialize(&parameters).unwrap();
        let direction = vec![Neg::neg(&objective_gradients[0]).unwrap()];
        let (updates, state) = linesearch.update(direction, state, &parameters, &arguments).unwrap();
        assert_eq!(updates, vec![Array::vector(vec![-0.0f32, -0.0]).unwrap()]);
        assert_eq!(state.learning_rate(), &Array::scalar(0.0f32).unwrap());
        assert_eq!(state.value(), &Array::scalar(5.0f32).unwrap());
        assert_eq!(state.gradients(), &objective_gradients);
    }

    #[test]
    fn test_scale_by_zoom_linesearch_update_arguments() {
        let linesearch = ScaleByZoomLinesearch::new(5).unwrap();
        let parameters = vec![Array::vector(vec![1.0f32, 2.0]).unwrap()];
        let value = quadratic(&parameters).unwrap();
        let objective_gradients = quadratic_gradients(&parameters);
        let objective = quadratic;
        let arguments = UpdateArguments::new()
            .with_value(&value)
            .with_gradients(&objective_gradients)
            .with_objective(&objective);
        let state = linesearch.initialize(&parameters).unwrap();
        assert!(matches!(
            linesearch.update(parameters.clone(), state, &parameters, &arguments),
            Err(OptimizerError::MissingArgument { message })
                if message == "`ScaleByZoomLinesearch` requires the `objective_with_gradients` update argument, which \
                               was not provided",
        ));
    }

    #[test]
    fn test_scale_by_backtracking_linesearch_optax_parity() {
        // Optax 9ecde04b: `optax.chain(optax.scale(-1.0),
        // optax.scale_by_backtracking_linesearch(max_backtracking_steps=15))`,
        // on the parity quadratic.
        const EXPECTED: [[f32; 9]; 4] = [
            [0.4, 3.2, -2.4, -0.0, 1.2, 0.90000004, 4.8, -1.0, 0.6],
            [0.08000002, -1.9200001, 3.3600004, -0.0, 0.23999996, -0.18000011, -2.8800004, 1.0, -0.8400001],
            [0.0127999885, 0.9216001, -3.763201, -0.0, 0.038400006, 0.028800031, 1.3824003, -0.8000001, 0.94080025],
            [0.004423682, -0.2477262, 3.3236601, -0.0, 0.013271047, 0.0011059389, -0.3715893, 0.46080017, -0.83091503],
        ];
        let optimizer = Chain::new(Scale::new(-1.0).unwrap(), ScaleByBacktrackingLinesearch::new(15));
        assert_parity(&parity_quadratic_updates(&optimizer, 4), &EXPECTED);
    }

    #[test]
    fn test_scale_by_zoom_linesearch_optax_parity() {
        // Optax 9ecde04b: `optax.chain(optax.scale(-1.0), optax.scale_by_zoom_linesearch(max_linesearch_steps=15))`, on
        // the
        // parity quadratic.
        const EXPECTED: [[f32; 9]; 4] = [
            [0.23721394, 1.8977115, -1.4232836, -0.0, 0.7116418, 0.53373134, 2.8465672, -0.59303486, 0.3558209],
            [0.12467302, 0.097057015, 0.6024526, -0.0, 0.37401912, 0.15390582, 0.14558564, 0.11034583, -0.15061314],
            [
                0.06552467,
                0.0049639386,
                -0.25500822,
                -0.0,
                0.196574,
                0.044380013,
                0.0074458513,
                -0.020532025,
                0.063752055,
            ],
            [
                0.034437925,
                0.00025388083,
                0.10794081,
                -0.0,
                0.10331383,
                0.012797342,
                0.00038079298,
                0.003820372,
                -0.026985116,
            ],
        ];
        let optimizer = Chain::new(Scale::new(-1.0).unwrap(), ScaleByZoomLinesearch::new(15).unwrap());
        assert_parity(&parity_quadratic_updates(&optimizer, 4), &EXPECTED);
    }
}
