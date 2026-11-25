//! Rescaling by running averages or sums of the squared gradients, which is the core of RMSProp, Adagrad, Adadelta,
//! and NovoGrad.
//!
//! [`ScaleByRms`] divides the gradients by the root mean square of their recent values and [`ScaleByStddev`] by their
//! recent standard deviation, which are the plain and the centered rescalings of RMSProp. [`ScaleByRss`] divides them
//! by the root of the sum of all their squares, which is the rescaling of Adagrad. [`ScaleByAdadelta`] scales them by
//! the ratio of the root mean squares of the recent updates and gradients, and [`ScaleByNovograd`] normalizes them by
//! a running average of their norms per leaf.

use std::marker::PhantomData;

use ryft_core::{Add, ComparisonDirection, DataType, Div, Mul, Parameter, Parameterized, ReductionKind, Sub};

use crate::macros::parameterized_state;
use crate::{
    Optimizer, OptimizerError, OptimizerValue, UpdateArguments, all_axes, bias_corrected, bias_correction, convert_to,
    filled_statistics, increment_count, initial_count, statistic_data_type, update_moment, validate_decay_rate,
    validate_finite, validate_floating_point, validate_non_negative, validate_structure, zero_count_like,
};

// TODO(eaplatanios): Review this module.

parameterized_state! {
    /// State of [`ScaleByRms`], which holds the number of updates applied so far and the exponential moving averages of
    /// the squared gradients, with the structure `S` of the parameters.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct ScaleByRmsState<P, S> with family ScaleByRmsStateFamily {
        /// Rank-0 `i32` array that holds the number of updates applied so far.
        count: P,

        /// Exponential moving averages of the squared gradients, with the structure of the parameters.
        second_moments: S,
    }
}

impl<P: Parameter, S> ScaleByRmsState<P, S> {
    /// Returns the rank-0 `i32` array that holds the number of updates applied so far.
    #[inline]
    pub fn count(&self) -> &P {
        &self.count
    }

    /// Returns the exponential moving averages of the squared gradients, with the structure of the parameters.
    #[inline]
    pub fn second_moments(&self) -> &S {
        &self.second_moments
    }
}

/// Transformation that divides the gradients by the root mean square of their recent values, which is the rescaling
/// of RMSProp of [Hinton (2012)](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf) and the
/// analogue of
/// [`optax.scale_by_rms`](https://optax.readthedocs.io/en/latest/api/transformations.html#optax.scale_by_rms). For a
/// gradient `g`, it updates `v = (1 - β) · g² + β · v` (from the initial scale `v₀`) and returns `g / √(v̂ + ε)`, or
/// `g / (√v̂ + ε)` when `ε` is added outside the square root, where `v̂` is `v` or, with bias correction at the step
/// count `t`, `v / (1 - βᵗ)`. The defaults are `β = 0.9`, `ε = 1e-8`, and `v₀ = 0`, with `ε` inside the square root
/// and without bias correction. Note that the state always holds a step count, even without bias correction.
#[derive(Clone, Debug, PartialEq)]
pub struct ScaleByRms {
    /// Decay rate `β` of the moving averages.
    decay: f64,

    /// Term `ε` that is added for numerical stability.
    epsilon: f64,

    /// Initial value `v₀` of the moving averages.
    initial_scale: f64,

    /// Whether `ε` is added inside the square root rather than to it.
    epsilon_in_sqrt: bool,

    /// Whether the moving averages are corrected for their initialization bias.
    bias_correction: bool,
}

impl ScaleByRms {
    /// Creates a new [`ScaleByRms`] with the default hyperparameters.
    #[inline]
    pub fn new() -> Self {
        Self { decay: 0.9, epsilon: 1e-8, initial_scale: 0.0, epsilon_in_sqrt: true, bias_correction: false }
    }

    /// Returns this transformation with `decay` as the decay rate of its moving averages.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `decay` is not in `[0, 1)`.
    pub fn with_decay(mut self, decay: f64) -> Result<Self, OptimizerError> {
        validate_decay_rate("decay", decay)?;
        self.decay = decay;
        Ok(self)
    }

    /// Returns this transformation with `epsilon` as the term that is added for numerical stability.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `epsilon` is negative or not finite.
    pub fn with_epsilon(mut self, epsilon: f64) -> Result<Self, OptimizerError> {
        validate_non_negative("epsilon", epsilon)?;
        self.epsilon = epsilon;
        Ok(self)
    }

    /// Returns this transformation with `initial_scale` as the initial value of its moving averages.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `initial_scale` is negative or not finite.
    pub fn with_initial_scale(mut self, initial_scale: f64) -> Result<Self, OptimizerError> {
        validate_non_negative("initial_scale", initial_scale)?;
        self.initial_scale = initial_scale;
        Ok(self)
    }

    /// Returns this transformation with `ε` added inside the square root when `epsilon_in_sqrt` is `true` and to the
    /// square root otherwise.
    #[inline]
    pub fn with_epsilon_in_sqrt(mut self, epsilon_in_sqrt: bool) -> Self {
        self.epsilon_in_sqrt = epsilon_in_sqrt;
        self
    }

    /// Returns this transformation with the bias correction of its moving averages enabled when `bias_correction` is
    /// `true`.
    #[inline]
    pub fn with_bias_correction(mut self, bias_correction: bool) -> Self {
        self.bias_correction = bias_correction;
        self
    }

    /// Returns the decay rate `β` of the moving averages of this transformation.
    #[inline]
    pub fn decay(&self) -> f64 {
        self.decay
    }

    /// Returns the term `ε` that this transformation adds for numerical stability.
    #[inline]
    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    /// Returns the initial value `v₀` of the moving averages of this transformation.
    #[inline]
    pub fn initial_scale(&self) -> f64 {
        self.initial_scale
    }

    /// Returns whether this transformation adds `ε` inside the square root.
    #[inline]
    pub fn epsilon_in_sqrt(&self) -> bool {
        self.epsilon_in_sqrt
    }

    /// Returns whether this transformation corrects its moving averages for their initialization bias.
    #[inline]
    pub fn bias_correction(&self) -> bool {
        self.bias_correction
    }
}

impl Default for ScaleByRms {
    fn default() -> Self {
        Self::new()
    }
}

impl<P: OptimizerValue> Optimizer<P> for ScaleByRms {
    type State<S: Parameterized<P>> = ScaleByRmsState<P, S>;

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        Ok(ScaleByRmsState {
            count: initial_count("ScaleByRms", parameters)?,
            second_moments: filled_statistics(parameters, self.initial_scale, None)?,
        })
    }

    fn update<S: Parameterized<P>>(
        &self,
        gradients: S,
        state: Self::State<S>,
        parameters: &S,
        arguments: &UpdateArguments<'_, P, S>,
    ) -> Result<(S, Self::State<S>), OptimizerError> {
        // This follows the operation order of `optax.scale_by_rms`.
        let _ = arguments;
        let paths = gradients.parameter_paths().collect::<Vec<_>>();
        validate_structure("parameters", &paths, parameters)?;
        validate_structure("second_moments", &paths, &state.second_moments)?;
        let count = increment_count(&state.count)?;
        let correction = bias_correction(&convert_to(&count, DataType::F32)?, self.decay)?;
        let structure = gradients.parameter_structure();
        let mut updates = Vec::with_capacity(paths.len());
        let mut second_moments = Vec::with_capacity(paths.len());
        for (gradient, second_moment) in gradients.into_parameters().zip(state.second_moments.into_parameters()) {
            let data_type = gradient.r#type().data_type();
            let statistic_type = statistic_data_type(data_type);
            let second_moment_type = second_moment.r#type().data_type();
            let gradient = convert_to(&gradient, statistic_type)?;
            let second_moment = update_moment(
                &Mul::mul(&gradient, &gradient)?,
                &convert_to(&second_moment, statistic_type)?,
                self.decay,
            )?;
            let estimate =
                if self.bias_correction { bias_corrected(&second_moment, &correction)? } else { second_moment.clone() };
            let scaling = inverse_root(&estimate, self.epsilon, self.epsilon_in_sqrt)?;
            updates.push(convert_to(&Mul::mul(&scaling, &gradient)?, data_type)?);
            second_moments.push(convert_to(&second_moment, second_moment_type)?);
        }
        let state = ScaleByRmsState { count, second_moments: S::from_parameters(structure.clone(), second_moments)? };
        Ok((S::from_parameters(structure, updates)?, state))
    }
}

parameterized_state! {
    /// State of [`ScaleByStddev`], which holds the number of updates applied so far and the exponential moving averages
    /// of the gradients and of their squares, with the structure `S` of the parameters.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct ScaleByStddevState<P, S> with family ScaleByStddevStateFamily {
        /// Rank-0 `i32` array that holds the number of updates applied so far.
        count: P,

        /// Exponential moving averages of the gradients, with the structure of the parameters.
        first_moments: S,

        /// Exponential moving averages of the squared gradients, with the structure of the parameters.
        second_moments: S,
    }
}

impl<P: Parameter, S> ScaleByStddevState<P, S> {
    /// Returns the rank-0 `i32` array that holds the number of updates applied so far.
    #[inline]
    pub fn count(&self) -> &P {
        &self.count
    }

    /// Returns the exponential moving averages of the gradients, with the structure of the parameters.
    #[inline]
    pub fn first_moments(&self) -> &S {
        &self.first_moments
    }

    /// Returns the exponential moving averages of the squared gradients, with the structure of the parameters.
    #[inline]
    pub fn second_moments(&self) -> &S {
        &self.second_moments
    }
}

/// Transformation that divides the gradients by their recent standard deviation, which is the rescaling of centered
/// RMSProp of [Graves (2013)](https://arxiv.org/abs/1308.0850) and the analogue of
/// [`optax.scale_by_stddev`](https://optax.readthedocs.io/en/latest/api/transformations.html#optax.scale_by_stddev).
/// For a gradient `g`, it updates `m = (1 - β) · g + β · m` and `v = (1 - β) · g² + β · v` and returns
/// `g / √(v̂ - m̂² + ε)`, or `g / (√(v̂ - m̂²) + ε)` when `ε` is added outside the square root, where the estimates are
/// the moving averages themselves or, with bias correction, the moving averages divided by `1 - βᵗ`. The defaults are
/// those of [`ScaleByRms`].
#[derive(Clone, Debug, PartialEq)]
pub struct ScaleByStddev {
    /// Decay rate `β` of the moving averages.
    decay: f64,

    /// Term `ε` that is added for numerical stability.
    epsilon: f64,

    /// Initial value of the moving averages of the squared gradients.
    initial_scale: f64,

    /// Whether `ε` is added inside the square root rather than to it.
    epsilon_in_sqrt: bool,

    /// Whether the moving averages are corrected for their initialization bias.
    bias_correction: bool,
}

impl ScaleByStddev {
    /// Creates a new [`ScaleByStddev`] with the default hyperparameters.
    #[inline]
    pub fn new() -> Self {
        Self { decay: 0.9, epsilon: 1e-8, initial_scale: 0.0, epsilon_in_sqrt: true, bias_correction: false }
    }

    /// Returns this transformation with `decay` as the decay rate of its moving averages.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `decay` is not in `[0, 1)`.
    pub fn with_decay(mut self, decay: f64) -> Result<Self, OptimizerError> {
        validate_decay_rate("decay", decay)?;
        self.decay = decay;
        Ok(self)
    }

    /// Returns this transformation with `epsilon` as the term that is added for numerical stability.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `epsilon` is negative or not finite.
    pub fn with_epsilon(mut self, epsilon: f64) -> Result<Self, OptimizerError> {
        validate_non_negative("epsilon", epsilon)?;
        self.epsilon = epsilon;
        Ok(self)
    }

    /// Returns this transformation with `initial_scale` as the initial value of its moving averages of the squared
    /// gradients.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `initial_scale` is negative or not finite.
    pub fn with_initial_scale(mut self, initial_scale: f64) -> Result<Self, OptimizerError> {
        validate_non_negative("initial_scale", initial_scale)?;
        self.initial_scale = initial_scale;
        Ok(self)
    }

    /// Returns this transformation with `ε` added inside the square root when `epsilon_in_sqrt` is `true` and to the
    /// square root otherwise.
    #[inline]
    pub fn with_epsilon_in_sqrt(mut self, epsilon_in_sqrt: bool) -> Self {
        self.epsilon_in_sqrt = epsilon_in_sqrt;
        self
    }

    /// Returns this transformation with the bias correction of its moving averages enabled when `bias_correction` is
    /// `true`.
    #[inline]
    pub fn with_bias_correction(mut self, bias_correction: bool) -> Self {
        self.bias_correction = bias_correction;
        self
    }

    /// Returns the decay rate `β` of the moving averages of this transformation.
    #[inline]
    pub fn decay(&self) -> f64 {
        self.decay
    }

    /// Returns the term `ε` that this transformation adds for numerical stability.
    #[inline]
    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    /// Returns the initial value of the moving averages of the squared gradients of this transformation.
    #[inline]
    pub fn initial_scale(&self) -> f64 {
        self.initial_scale
    }

    /// Returns whether this transformation adds `ε` inside the square root.
    #[inline]
    pub fn epsilon_in_sqrt(&self) -> bool {
        self.epsilon_in_sqrt
    }

    /// Returns whether this transformation corrects its moving averages for their initialization bias.
    #[inline]
    pub fn bias_correction(&self) -> bool {
        self.bias_correction
    }
}

impl Default for ScaleByStddev {
    fn default() -> Self {
        Self::new()
    }
}

impl<P: OptimizerValue> Optimizer<P> for ScaleByStddev {
    type State<S: Parameterized<P>> = ScaleByStddevState<P, S>;

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        Ok(ScaleByStddevState {
            count: initial_count("ScaleByStddev", parameters)?,
            first_moments: filled_statistics(parameters, 0.0, None)?,
            second_moments: filled_statistics(parameters, self.initial_scale, None)?,
        })
    }

    fn update<S: Parameterized<P>>(
        &self,
        gradients: S,
        state: Self::State<S>,
        parameters: &S,
        arguments: &UpdateArguments<'_, P, S>,
    ) -> Result<(S, Self::State<S>), OptimizerError> {
        // This follows the operation order of `optax.scale_by_stddev`.
        let _ = arguments;
        let paths = gradients.parameter_paths().collect::<Vec<_>>();
        validate_structure("parameters", &paths, parameters)?;
        validate_structure("first_moments", &paths, &state.first_moments)?;
        validate_structure("second_moments", &paths, &state.second_moments)?;
        let count = increment_count(&state.count)?;
        let correction = bias_correction(&convert_to(&count, DataType::F32)?, self.decay)?;
        let structure = gradients.parameter_structure();
        let mut updates = Vec::with_capacity(paths.len());
        let mut first_moments = Vec::with_capacity(paths.len());
        let mut second_moments = Vec::with_capacity(paths.len());
        let moments = state.first_moments.into_parameters().zip(state.second_moments.into_parameters());
        for (gradient, (first_moment, second_moment)) in gradients.into_parameters().zip(moments) {
            let data_type = gradient.r#type().data_type();
            let statistic_type = statistic_data_type(data_type);
            let first_moment_type = first_moment.r#type().data_type();
            let second_moment_type = second_moment.r#type().data_type();
            let gradient = convert_to(&gradient, statistic_type)?;
            let first_moment = update_moment(&gradient, &convert_to(&first_moment, statistic_type)?, self.decay)?;
            let second_moment = update_moment(
                &Mul::mul(&gradient, &gradient)?,
                &convert_to(&second_moment, statistic_type)?,
                self.decay,
            )?;
            let (first_estimate, second_estimate) = if self.bias_correction {
                (bias_corrected(&first_moment, &correction)?, bias_corrected(&second_moment, &correction)?)
            } else {
                (first_moment.clone(), second_moment.clone())
            };
            let variance = Sub::sub(&second_estimate, &Mul::mul(&first_estimate, &first_estimate)?)?;
            let scaling = inverse_root(&variance, self.epsilon, self.epsilon_in_sqrt)?;
            updates.push(convert_to(&Mul::mul(&scaling, &gradient)?, data_type)?);
            first_moments.push(convert_to(&first_moment, first_moment_type)?);
            second_moments.push(convert_to(&second_moment, second_moment_type)?);
        }
        let state = ScaleByStddevState {
            count,
            first_moments: S::from_parameters(structure.clone(), first_moments)?,
            second_moments: S::from_parameters(structure.clone(), second_moments)?,
        };
        Ok((S::from_parameters(structure, updates)?, state))
    }
}

parameterized_state! {
    /// State of [`ScaleByRss`], which holds the sums of the squared gradients with the structure `S` of the parameters.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct ScaleByRssState<P, S> with family ScaleByRssStateFamily and marker leaf {
        /// Sums of the squared gradients (starting at the initial accumulator value), with the structure of the
        /// parameters.
        sum_of_squares: S,
    }
}

impl<P: Parameter, S> ScaleByRssState<P, S> {
    /// Returns the sums of the squared gradients, with the structure of the parameters.
    #[inline]
    pub fn sum_of_squares(&self) -> &S {
        &self.sum_of_squares
    }
}

/// Transformation that divides the gradients by the root of the sum of all their squares, which is the rescaling of
/// Adagrad of [Duchi et al. (2011)](https://jmlr.org/papers/v12/duchi11a.html) and the analogue of
/// [`optax.scale_by_rss`](https://optax.readthedocs.io/en/latest/api/transformations.html#optax.scale_by_rss). For a
/// gradient `g`, it updates `s = g² + s` (from the initial accumulator value) and returns `g / √(s + ε)` where `s` is
/// positive and `0` elsewhere. The defaults are an initial accumulator value of `0.1` and `ε = 1e-7`.
#[derive(Clone, Debug, PartialEq)]
pub struct ScaleByRss {
    /// Initial value of the sums of squares.
    initial_accumulator_value: f64,

    /// Term `ε` that is added to the sums of squares inside the square root.
    epsilon: f64,
}

impl ScaleByRss {
    /// Creates a new [`ScaleByRss`] with the default hyperparameters.
    #[inline]
    pub fn new() -> Self {
        Self { initial_accumulator_value: 0.1, epsilon: 1e-7 }
    }

    /// Returns this transformation with `initial_accumulator_value` as the initial value of its sums of squares.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `initial_accumulator_value` is negative or not finite.
    pub fn with_initial_accumulator_value(mut self, initial_accumulator_value: f64) -> Result<Self, OptimizerError> {
        validate_non_negative("initial_accumulator_value", initial_accumulator_value)?;
        self.initial_accumulator_value = initial_accumulator_value;
        Ok(self)
    }

    /// Returns this transformation with `epsilon` as the term that is added to the sums of squares inside the square
    /// root.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `epsilon` is negative or not finite.
    pub fn with_epsilon(mut self, epsilon: f64) -> Result<Self, OptimizerError> {
        validate_non_negative("epsilon", epsilon)?;
        self.epsilon = epsilon;
        Ok(self)
    }

    /// Returns the initial value of the sums of squares of this transformation.
    #[inline]
    pub fn initial_accumulator_value(&self) -> f64 {
        self.initial_accumulator_value
    }

    /// Returns the term `ε` that this transformation adds to the sums of squares inside the square root.
    #[inline]
    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }
}

impl Default for ScaleByRss {
    fn default() -> Self {
        Self::new()
    }
}

impl<P: OptimizerValue> Optimizer<P> for ScaleByRss {
    type State<S: Parameterized<P>> = ScaleByRssState<P, S>;

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        Ok(ScaleByRssState {
            sum_of_squares: filled_statistics(parameters, self.initial_accumulator_value, None)?,
            leaf: PhantomData,
        })
    }

    fn update<S: Parameterized<P>>(
        &self,
        gradients: S,
        state: Self::State<S>,
        parameters: &S,
        arguments: &UpdateArguments<'_, P, S>,
    ) -> Result<(S, Self::State<S>), OptimizerError> {
        // This follows the operation order of `optax.scale_by_rss`.
        let _ = arguments;
        let paths = gradients.parameter_paths().collect::<Vec<_>>();
        validate_structure("parameters", &paths, parameters)?;
        validate_structure("sum_of_squares", &paths, &state.sum_of_squares)?;
        let structure = gradients.parameter_structure();
        let mut updates = Vec::with_capacity(paths.len());
        let mut sums_of_squares = Vec::with_capacity(paths.len());
        for (gradient, sum_of_squares) in gradients.into_parameters().zip(state.sum_of_squares.into_parameters()) {
            let data_type = gradient.r#type().data_type();
            let statistic_type = statistic_data_type(data_type);
            let sum_of_squares_type = sum_of_squares.r#type().data_type();
            let gradient = convert_to(&gradient, statistic_type)?;
            let sum_of_squares =
                Add::add(&Mul::mul(&gradient, &gradient)?, &convert_to(&sum_of_squares, statistic_type)?)?;
            let zero = gradient.fill_like(0f64)?;
            let positive = sum_of_squares.compare(&zero, ComparisonDirection::GreaterThan)?;
            let inverse = Add::add(&sum_of_squares, &gradient.fill_like(self.epsilon)?)?.rsqrt()?;
            let scaling = positive.select(&inverse, &zero)?;
            updates.push(convert_to(&Mul::mul(&scaling, &gradient)?, data_type)?);
            sums_of_squares.push(convert_to(&sum_of_squares, sum_of_squares_type)?);
        }
        let state = ScaleByRssState {
            sum_of_squares: S::from_parameters(structure.clone(), sums_of_squares)?,
            leaf: PhantomData,
        };
        Ok((S::from_parameters(structure, updates)?, state))
    }
}

parameterized_state! {
    /// State of [`ScaleByAdadelta`], which holds the exponential moving averages of the squared gradients and of the
    /// squared updates, with the structure `S` of the parameters.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct ScaleByAdadeltaState<P, S> with family ScaleByAdadeltaStateFamily and marker leaf {
        /// Exponential moving averages of the squared gradients, with the structure of the parameters.
        squared_gradient_averages: S,

        /// Exponential moving averages of the squared updates, with the structure of the parameters.
        squared_update_averages: S,
    }
}

impl<P: Parameter, S> ScaleByAdadeltaState<P, S> {
    /// Returns the exponential moving averages of the squared gradients, with the structure of the parameters.
    #[inline]
    pub fn squared_gradient_averages(&self) -> &S {
        &self.squared_gradient_averages
    }

    /// Returns the exponential moving averages of the squared updates, with the structure of the parameters.
    #[inline]
    pub fn squared_update_averages(&self) -> &S {
        &self.squared_update_averages
    }
}

/// Transformation that scales the gradients by the ratio of the root mean squares of the recent updates and gradients,
/// which is the rescaling of Adadelta of [Zeiler (2012)](https://arxiv.org/abs/1212.5701) and the analogue of
/// [`optax.scale_by_adadelta`](
/// https://optax.readthedocs.io/en/latest/api/transformations.html#optax.scale_by_adadelta).
/// For a gradient `g`, it updates `E[g²] = (1 - ρ) · g² + ρ · E[g²]`, returns the update
/// `u = (√(E[Δ²] + ε) / √(E[g²] + ε)) · g` with the average `E[Δ²]` of the previous squared updates, and then updates
/// `E[Δ²] = (1 - ρ) · u² + ρ · E[Δ²]`. The defaults are `ρ = 0.9` and `ε = 1e-6`.
#[derive(Clone, Debug, PartialEq)]
pub struct ScaleByAdadelta {
    /// Decay rate `ρ` of the moving averages.
    rho: f64,

    /// Term `ε` that is added inside the square roots.
    epsilon: f64,
}

impl ScaleByAdadelta {
    /// Creates a new [`ScaleByAdadelta`] with the default hyperparameters.
    #[inline]
    pub fn new() -> Self {
        Self { rho: 0.9, epsilon: 1e-6 }
    }

    /// Returns this transformation with `rho` as the decay rate of its moving averages.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `rho` is not in `[0, 1)`.
    pub fn with_rho(mut self, rho: f64) -> Result<Self, OptimizerError> {
        validate_decay_rate("rho", rho)?;
        self.rho = rho;
        Ok(self)
    }

    /// Returns this transformation with `epsilon` as the term that is added inside the square roots.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `epsilon` is negative or not finite.
    pub fn with_epsilon(mut self, epsilon: f64) -> Result<Self, OptimizerError> {
        validate_non_negative("epsilon", epsilon)?;
        self.epsilon = epsilon;
        Ok(self)
    }

    /// Returns the decay rate `ρ` of the moving averages of this transformation.
    #[inline]
    pub fn rho(&self) -> f64 {
        self.rho
    }

    /// Returns the term `ε` that this transformation adds inside the square roots.
    #[inline]
    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }
}

impl Default for ScaleByAdadelta {
    fn default() -> Self {
        Self::new()
    }
}

impl<P: OptimizerValue> Optimizer<P> for ScaleByAdadelta {
    type State<S: Parameterized<P>> = ScaleByAdadeltaState<P, S>;

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        Ok(ScaleByAdadeltaState {
            squared_gradient_averages: filled_statistics(parameters, 0.0, None)?,
            squared_update_averages: filled_statistics(parameters, 0.0, None)?,
            leaf: PhantomData,
        })
    }

    fn update<S: Parameterized<P>>(
        &self,
        gradients: S,
        state: Self::State<S>,
        parameters: &S,
        arguments: &UpdateArguments<'_, P, S>,
    ) -> Result<(S, Self::State<S>), OptimizerError> {
        // This follows the operation order of `optax.scale_by_adadelta`.
        let _ = arguments;
        let paths = gradients.parameter_paths().collect::<Vec<_>>();
        validate_structure("parameters", &paths, parameters)?;
        validate_structure("squared_gradient_averages", &paths, &state.squared_gradient_averages)?;
        validate_structure("squared_update_averages", &paths, &state.squared_update_averages)?;
        let structure = gradients.parameter_structure();
        let mut updates = Vec::with_capacity(paths.len());
        let mut gradient_averages = Vec::with_capacity(paths.len());
        let mut update_averages = Vec::with_capacity(paths.len());
        let averages = state
            .squared_gradient_averages
            .into_parameters()
            .zip(state.squared_update_averages.into_parameters());
        for (gradient, (gradient_average, update_average)) in gradients.into_parameters().zip(averages) {
            let data_type = gradient.r#type().data_type();
            let statistic_type = statistic_data_type(data_type);
            let gradient_average_type = gradient_average.r#type().data_type();
            let update_average_type = update_average.r#type().data_type();
            let gradient = convert_to(&gradient, statistic_type)?;
            let epsilon = gradient.fill_like(self.epsilon)?;
            let gradient_average = update_moment(
                &Mul::mul(&gradient, &gradient)?,
                &convert_to(&gradient_average, statistic_type)?,
                self.rho,
            )?;
            let update_average = convert_to(&update_average, statistic_type)?;
            let ratio = Div::div(
                &Add::add(&update_average, &epsilon)?.sqrt()?,
                &Add::add(&gradient_average, &epsilon)?.sqrt()?,
            )?;
            let update = Mul::mul(&ratio, &gradient)?;
            let update_average = update_moment(&Mul::mul(&update, &update)?, &update_average, self.rho)?;
            updates.push(convert_to(&update, data_type)?);
            gradient_averages.push(convert_to(&gradient_average, gradient_average_type)?);
            update_averages.push(convert_to(&update_average, update_average_type)?);
        }
        let state = ScaleByAdadeltaState {
            squared_gradient_averages: S::from_parameters(structure.clone(), gradient_averages)?,
            squared_update_averages: S::from_parameters(structure.clone(), update_averages)?,
            leaf: PhantomData,
        };
        Ok((S::from_parameters(structure, updates)?, state))
    }
}

parameterized_state! {
    /// State of [`ScaleByNovograd`], which holds the number of updates applied so far, the first moments with the
    /// structure `S` of the parameters, and the moving averages of the squared norms of the gradients, which have the
    /// structure of the parameters but rank-0 leaves.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct ScaleByNovogradState<P, S> with family ScaleByNovogradStateFamily {
        /// Rank-0 `i32` array that holds the number of updates applied so far.
        count: P,

        /// Momenta of the normalized gradients, with the structure of the parameters.
        first_moments: S,

        /// Exponential moving averages of the squared norms of the gradient leaves, with the structure of the
        /// parameters and rank-0 leaves.
        second_moments: S,
    }
}

impl<P: Parameter, S> ScaleByNovogradState<P, S> {
    /// Returns the rank-0 `i32` array that holds the number of updates applied so far.
    #[inline]
    pub fn count(&self) -> &P {
        &self.count
    }

    /// Returns the momenta of the normalized gradients, with the structure of the parameters.
    #[inline]
    pub fn first_moments(&self) -> &S {
        &self.first_moments
    }

    /// Returns the exponential moving averages of the squared norms of the gradient leaves, which are rank-0.
    #[inline]
    pub fn second_moments(&self) -> &S {
        &self.second_moments
    }
}

/// Transformation that normalizes each gradient leaf by a running average of its squared norm and accumulates the
/// normalized gradients with momentum, which is the rescaling of NovoGrad of
/// [Ginsburg et al. (2019)](https://arxiv.org/abs/1905.11286) and the analogue of
/// [`optax.scale_by_novograd`](
/// https://optax.readthedocs.io/en/latest/api/transformations.html#optax.scale_by_novograd).
/// For a gradient leaf `g` with the norm `‖g‖` and the parameter `θ`, it updates the rank-0 second moment as `v = ‖g‖²`
/// at the first update and `v = (1 - β₂) · ‖g‖² + β₂ · v` afterwards, and the first moment as
/// `m = g / (√(v + ε̄) + ε) + λ · θ` at the first update and `m = β₁ · m + g / (√(v + ε̄) + ε) + λ · θ` afterwards, and
/// returns `m`. The defaults are `β₁ = 0.9`, `β₂ = 0.25`, `ε = 1e-8`, `ε̄ = 0`, and a weight decay `λ = 0`.
#[derive(Clone, Debug, PartialEq)]
pub struct ScaleByNovograd {
    /// Decay rate `β₁` of the first moments.
    beta1: f64,

    /// Decay rate `β₂` of the second moments.
    beta2: f64,

    /// Term `ε` that is added to the square roots of the second moments.
    epsilon: f64,

    /// Term `ε̄` that is added to the second moments inside the square roots.
    epsilon_root: f64,

    /// Weight decay `λ` that is added to the normalized gradients.
    weight_decay: f64,

    /// Data type in which the first moments are stored, or [`None`] for the statistic data type of each leaf.
    first_moment_data_type: Option<DataType>,
}

impl ScaleByNovograd {
    /// Creates a new [`ScaleByNovograd`] with the default hyperparameters.
    #[inline]
    pub fn new() -> Self {
        Self {
            beta1: 0.9,
            beta2: 0.25,
            epsilon: 1e-8,
            epsilon_root: 0.0,
            weight_decay: 0.0,
            first_moment_data_type: None,
        }
    }

    /// Returns this transformation with `beta1` as the decay rate of its first moments.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `beta1` is not in `[0, 1)`.
    pub fn with_beta1(mut self, beta1: f64) -> Result<Self, OptimizerError> {
        validate_decay_rate("beta1", beta1)?;
        self.beta1 = beta1;
        Ok(self)
    }

    /// Returns this transformation with `beta2` as the decay rate of its second moments.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `beta2` is not in `[0, 1)`.
    pub fn with_beta2(mut self, beta2: f64) -> Result<Self, OptimizerError> {
        validate_decay_rate("beta2", beta2)?;
        self.beta2 = beta2;
        Ok(self)
    }

    /// Returns this transformation with `epsilon` as the term that is added to the square roots of its second moments.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `epsilon` is negative or not finite.
    pub fn with_epsilon(mut self, epsilon: f64) -> Result<Self, OptimizerError> {
        validate_non_negative("epsilon", epsilon)?;
        self.epsilon = epsilon;
        Ok(self)
    }

    /// Returns this transformation with `epsilon_root` as the term that is added to its second moments inside the
    /// square roots.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `epsilon_root` is negative or not finite.
    pub fn with_epsilon_root(mut self, epsilon_root: f64) -> Result<Self, OptimizerError> {
        validate_non_negative("epsilon_root", epsilon_root)?;
        self.epsilon_root = epsilon_root;
        Ok(self)
    }

    /// Returns this transformation with `weight_decay` as the multiple of the parameters that is added to the
    /// normalized gradients.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `weight_decay` is not finite.
    pub fn with_weight_decay(mut self, weight_decay: f64) -> Result<Self, OptimizerError> {
        validate_finite("weight_decay", weight_decay)?;
        self.weight_decay = weight_decay;
        Ok(self)
    }

    /// Returns this transformation with its first moments stored in `data_type` (refer to
    /// [`ScaleByAdam::with_first_moment_data_type`](crate::ScaleByAdam::with_first_moment_data_type)).
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `data_type` is not a floating-point data type.
    pub fn with_first_moment_data_type(mut self, data_type: DataType) -> Result<Self, OptimizerError> {
        validate_floating_point("first_moment_data_type", data_type)?;
        self.first_moment_data_type = Some(data_type);
        Ok(self)
    }

    /// Returns the decay rate `β₁` of the first moments of this transformation.
    #[inline]
    pub fn beta1(&self) -> f64 {
        self.beta1
    }

    /// Returns the decay rate `β₂` of the second moments of this transformation.
    #[inline]
    pub fn beta2(&self) -> f64 {
        self.beta2
    }

    /// Returns the term `ε` that this transformation adds to the square roots of its second moments.
    #[inline]
    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    /// Returns the term `ε̄` that this transformation adds to its second moments inside the square roots.
    #[inline]
    pub fn epsilon_root(&self) -> f64 {
        self.epsilon_root
    }

    /// Returns the weight decay `λ` that this transformation adds to the normalized gradients.
    #[inline]
    pub fn weight_decay(&self) -> f64 {
        self.weight_decay
    }

    /// Returns the data type in which this transformation stores its first moments, or [`None`] if it stores them in
    /// the statistic data type of each leaf.
    #[inline]
    pub fn first_moment_data_type(&self) -> Option<DataType> {
        self.first_moment_data_type
    }
}

impl Default for ScaleByNovograd {
    fn default() -> Self {
        Self::new()
    }
}

impl<P: OptimizerValue> Optimizer<P> for ScaleByNovograd {
    type State<S: Parameterized<P>> = ScaleByNovogradState<P, S>;

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        let second_moments = parameters
            .parameters()
            .map(|parameter| {
                Ok(convert_to(&zero_count_like(parameter)?, statistic_data_type(parameter.r#type().data_type()))?)
            })
            .collect::<Result<Vec<_>, OptimizerError>>()?;
        Ok(ScaleByNovogradState {
            count: initial_count("ScaleByNovograd", parameters)?,
            first_moments: filled_statistics(parameters, 0.0, self.first_moment_data_type)?,
            second_moments: S::from_parameters(parameters.parameter_structure(), second_moments)?,
        })
    }

    fn update<S: Parameterized<P>>(
        &self,
        gradients: S,
        state: Self::State<S>,
        parameters: &S,
        arguments: &UpdateArguments<'_, P, S>,
    ) -> Result<(S, Self::State<S>), OptimizerError> {
        // This follows the operation order of `optax.scale_by_novograd`, whose two branches for the first update and
        // the later ones are both computed and selected. The squared norm is the square of `jnp.linalg.norm`, so the
        // square root of the sum of squares is squared again.
        let _ = arguments;
        let paths = gradients.parameter_paths().collect::<Vec<_>>();
        validate_structure("parameters", &paths, parameters)?;
        validate_structure("first_moments", &paths, &state.first_moments)?;
        validate_structure("second_moments", &paths, &state.second_moments)?;
        let count = increment_count(&state.count)?;
        let first_update = count.compare(&count.fill_like(1i32)?, ComparisonDirection::Equal)?;
        let structure = gradients.parameter_structure();
        let mut updates = Vec::with_capacity(paths.len());
        let mut first_moments = Vec::with_capacity(paths.len());
        let mut second_moments = Vec::with_capacity(paths.len());
        let statistics = state.first_moments.into_parameters().zip(state.second_moments.into_parameters());
        for ((gradient, parameter), (first_moment, second_moment)) in
            gradients.into_parameters().zip(parameters.parameters()).zip(statistics)
        {
            let data_type = gradient.r#type().data_type();
            let statistic_type = statistic_data_type(data_type);
            let first_moment_type = first_moment.r#type().data_type();
            let second_moment_type = second_moment.r#type().data_type();
            let gradient = convert_to(&gradient, statistic_type)?;
            let norm = Mul::mul(&gradient, &gradient)?.reduce(&all_axes(&gradient), ReductionKind::Sum)?.sqrt()?;
            let squared_norm = Mul::mul(&norm, &norm)?;
            let second_moment = first_update.select(
                &squared_norm,
                &update_moment(&squared_norm, &convert_to(&second_moment, statistic_type)?, self.beta2)?,
            )?;
            let shifted = if self.epsilon_root == 0.0 {
                second_moment.clone()
            } else {
                Add::add(&second_moment, &second_moment.fill_like(self.epsilon_root)?)?
            };
            let denominator = Add::add(&shifted.sqrt()?, &second_moment.fill_like(self.epsilon)?)?;
            let parameter = convert_to(parameter, statistic_type)?;
            let addition = Add::add(
                &Div::div(&gradient, &denominator)?,
                &Mul::mul(&parameter.fill_like(self.weight_decay)?, &parameter)?,
            )?;
            let first_moment = convert_to(&first_moment, statistic_type)?;
            let accumulated = Add::add(&Mul::mul(&first_moment.fill_like(self.beta1)?, &first_moment)?, &addition)?;
            let first_moment = first_update.select(&addition, &accumulated)?;

            // The updates are the stored first moments, which are converted to their storage data type first.
            let first_moment = convert_to(&first_moment, first_moment_type)?;
            updates.push(convert_to(&first_moment, data_type)?);
            first_moments.push(first_moment);
            second_moments.push(convert_to(&second_moment, second_moment_type)?);
        }
        let state = ScaleByNovogradState {
            count,
            first_moments: S::from_parameters(structure.clone(), first_moments)?,
            second_moments: S::from_parameters(structure.clone(), second_moments)?,
        };
        Ok((S::from_parameters(structure, updates)?, state))
    }
}

/// Returns `1 / √(value + ε)` when `epsilon_in_sqrt` is `true` and `1 / (√value + ε)` otherwise, which are the two
/// scalings of `optax.scale_by_rms` and `optax.scale_by_stddev`.
fn inverse_root<P: OptimizerValue>(value: &P, epsilon: f64, epsilon_in_sqrt: bool) -> Result<P, OptimizerError> {
    let epsilon = value.fill_like(epsilon)?;
    if epsilon_in_sqrt {
        Ok(Add::add(value, &epsilon)?.rsqrt()?)
    } else {
        Ok(Div::div(&value.fill_like(1f64)?, &Add::add(&value.sqrt()?, &epsilon)?)?)
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use ryft_core::{Array, ArrayType, Typed};

    use crate::tests::{assert_parity, parity_updates, state_paths, vector_updates};

    use super::*;

    #[test]
    fn test_scale_by_rms_new() {
        let rms = ScaleByRms::new();
        assert_eq!((rms.decay(), rms.epsilon(), rms.initial_scale()), (0.9, 1e-8, 0.0));
        assert!(rms.epsilon_in_sqrt());
        assert!(!rms.bias_correction());
        assert_eq!(ScaleByRms::default(), rms);
        let rms = rms
            .with_decay(0.5)
            .unwrap()
            .with_epsilon(0.0)
            .unwrap()
            .with_initial_scale(1.0)
            .unwrap()
            .with_epsilon_in_sqrt(false)
            .with_bias_correction(true);
        assert_eq!((rms.decay(), rms.epsilon(), rms.initial_scale()), (0.5, 0.0, 1.0));
        assert!(!rms.epsilon_in_sqrt());
        assert!(rms.bias_correction());
        assert!(matches!(
            ScaleByRms::new().with_initial_scale(-1.0),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`initial_scale` must be finite and non-negative but got `-1`",
        ));
    }

    #[test]
    fn test_scale_by_rms_initialize() {
        assert_eq!(
            state_paths(&ScaleByRms::new()),
            vec!["$.count", "$.second_moments[\"bias\"]", "$.second_moments[\"weights\"]"],
        );
    }

    #[test]
    fn test_scale_by_rms_update() {
        // With `β = 0.5` and `ε = 0` outside the square root, the gradient `2` gives `v = 2` and the update `2 / √2`,
        // and the gradient `1` gives `v = 0.5 + 1 = 1.5` and the update `1 / √1.5`.
        let rms = ScaleByRms::new().with_decay(0.5).unwrap().with_epsilon(0.0).unwrap().with_epsilon_in_sqrt(false);
        assert_eq!(
            vector_updates(&rms, &[0.0], &[&[2.0], &[1.0]]),
            vec![vec![1.0f32 / 2.0f32.sqrt() * 2.0], vec![1.0f32 / 1.5f32.sqrt()],],
        );

        // With bias correction and `ε` inside the square root, the gradient `2` gives `v̂ = 2 / 0.5 = 4`, so the update
        // is `2 / √4 = 1`.
        let rms = rms.with_epsilon_in_sqrt(true).with_bias_correction(true);
        assert_eq!(vector_updates(&rms, &[0.0], &[&[2.0]]), vec![vec![1.0]]);
    }

    #[test]
    fn test_scale_by_stddev_new() {
        let stddev = ScaleByStddev::new();
        assert_eq!((stddev.decay(), stddev.epsilon(), stddev.initial_scale()), (0.9, 1e-8, 0.0));
        assert!(stddev.epsilon_in_sqrt());
        assert!(!stddev.bias_correction());
        assert_eq!(ScaleByStddev::default(), stddev);
        let stddev = stddev
            .with_decay(0.5)
            .unwrap()
            .with_epsilon(1.0)
            .unwrap()
            .with_initial_scale(2.0)
            .unwrap()
            .with_epsilon_in_sqrt(false)
            .with_bias_correction(true);
        assert_eq!((stddev.decay(), stddev.epsilon(), stddev.initial_scale()), (0.5, 1.0, 2.0));
        assert!(!stddev.epsilon_in_sqrt());
        assert!(stddev.bias_correction());
        assert!(matches!(
            ScaleByStddev::new().with_decay(-0.5),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`decay` must be in `[0, 1)` but got `-0.5`",
        ));
    }

    #[test]
    fn test_scale_by_stddev_initialize() {
        assert_eq!(
            state_paths(&ScaleByStddev::new()),
            vec![
                "$.count",
                "$.first_moments[\"bias\"]",
                "$.first_moments[\"weights\"]",
                "$.second_moments[\"bias\"]",
                "$.second_moments[\"weights\"]",
            ],
        );
    }

    #[test]
    fn test_scale_by_stddev_update() {
        // With `β = 0.5` and `ε = 0` outside the square root, the gradient `2` gives `m = 1` and `v = 2`, so the
        // variance is `1` and the update is `2`. The gradient `4` gives `m = 2.5` and `v = 9`, so the variance is
        // `2.75`.
        let stddev =
            ScaleByStddev::new().with_decay(0.5).unwrap().with_epsilon(0.0).unwrap().with_epsilon_in_sqrt(false);
        assert_eq!(
            vector_updates(&stddev, &[0.0], &[&[2.0], &[4.0]]),
            vec![vec![2.0], vec![1.0f32 / 2.75f32.sqrt() * 4.0],],
        );

        // With bias correction, `m̂ = 2` and `v̂ = 4`, so the variance is `0` and the update is
        // `(1 / √(0 + 4)) · 2 = 1`.
        let stddev = stddev.with_epsilon(4.0).unwrap().with_epsilon_in_sqrt(true).with_bias_correction(true);
        assert_eq!(vector_updates(&stddev, &[0.0], &[&[2.0]]), vec![vec![1.0]]);
    }

    #[test]
    fn test_scale_by_rss_new() {
        let rss = ScaleByRss::new();
        assert_eq!((rss.initial_accumulator_value(), rss.epsilon()), (0.1, 1e-7));
        assert_eq!(ScaleByRss::default(), rss);
        let rss = rss.with_initial_accumulator_value(0.0).unwrap().with_epsilon(0.0).unwrap();
        assert_eq!((rss.initial_accumulator_value(), rss.epsilon()), (0.0, 0.0));
        assert!(matches!(
            ScaleByRss::new().with_epsilon(f64::INFINITY),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`epsilon` must be finite and non-negative but got `inf`",
        ));
    }

    #[test]
    fn test_scale_by_rss_initialize() {
        assert_eq!(
            state_paths(&ScaleByRss::new()),
            vec!["$.sum_of_squares[\"bias\"]", "$.sum_of_squares[\"weights\"]"],
        );
    }

    #[test]
    fn test_scale_by_rss_update() {
        // Without an initial value or `ε`, the gradient `2` gives `s = 4` and the update `2 / √4 = 1`, and the gradient
        // `3` gives `s = 13` and the update `3 / √13`. An element whose sum is still zero gets a zero update.
        let rss = ScaleByRss::new().with_initial_accumulator_value(0.0).unwrap().with_epsilon(0.0).unwrap();
        assert_eq!(
            vector_updates(&rss, &[0.0, 0.0], &[&[2.0, 0.0], &[3.0, 0.0]]),
            vec![vec![1.0, 0.0], vec![13.0f32.sqrt().recip() * 3.0, 0.0],],
        );
    }

    #[test]
    fn test_scale_by_adadelta_new() {
        let adadelta = ScaleByAdadelta::new();
        assert_eq!((adadelta.rho(), adadelta.epsilon()), (0.9, 1e-6));
        assert_eq!(ScaleByAdadelta::default(), adadelta);
        let adadelta = adadelta.with_rho(0.5).unwrap().with_epsilon(1.0).unwrap();
        assert_eq!((adadelta.rho(), adadelta.epsilon()), (0.5, 1.0));
        assert!(matches!(
            ScaleByAdadelta::new().with_rho(1.0),
            Err(OptimizerError::InvalidConfiguration { message }) if message == "`rho` must be in `[0, 1)` but got `1`",
        ));
    }

    #[test]
    fn test_scale_by_adadelta_initialize() {
        assert_eq!(
            state_paths(&ScaleByAdadelta::new()),
            vec![
                "$.squared_gradient_averages[\"bias\"]",
                "$.squared_gradient_averages[\"weights\"]",
                "$.squared_update_averages[\"bias\"]",
                "$.squared_update_averages[\"weights\"]",
            ],
        );
    }

    #[test]
    fn test_scale_by_adadelta_update() {
        // With `ρ = 0.5` and `ε = 1`, the gradient `2` gives `E[g²] = 2` and the update `u₁ = (√1 / √3) · 2`, after
        // which `E[Δ²] = 0.5 · u₁²`. The gradient `1` gives `E[g²] = 1.5` and the update `(√(E[Δ²] + 1) / √2.5) · 1`.
        let adadelta = ScaleByAdadelta::new().with_rho(0.5).unwrap().with_epsilon(1.0).unwrap();
        let first = 1.0f32.sqrt() / 3.0f32.sqrt() * 2.0;
        let update_average = 0.5f32 * (first * first) + 0.5 * 0.0;
        let second = (update_average + 1.0).sqrt() / 2.5f32.sqrt() * 1.0;
        assert_eq!(vector_updates(&adadelta, &[0.0], &[&[2.0], &[1.0]]), vec![vec![first], vec![second]]);
    }

    #[test]
    fn test_scale_by_novograd_new() {
        let novograd = ScaleByNovograd::new();
        assert_eq!(
            (novograd.beta1(), novograd.beta2(), novograd.epsilon(), novograd.epsilon_root(), novograd.weight_decay()),
            (0.9, 0.25, 1e-8, 0.0, 0.0),
        );
        assert_eq!(novograd.first_moment_data_type(), None);
        assert_eq!(ScaleByNovograd::default(), novograd);
        let novograd = novograd
            .with_beta1(0.5)
            .unwrap()
            .with_beta2(0.6)
            .unwrap()
            .with_epsilon(0.0)
            .unwrap()
            .with_epsilon_root(1e-3)
            .unwrap()
            .with_weight_decay(0.5)
            .unwrap()
            .with_first_moment_data_type(DataType::F64)
            .unwrap();
        assert_eq!(
            (novograd.beta1(), novograd.beta2(), novograd.epsilon(), novograd.epsilon_root(), novograd.weight_decay()),
            (0.5, 0.6, 0.0, 1e-3, 0.5),
        );
        assert_eq!(novograd.first_moment_data_type(), Some(DataType::F64));
        assert!(matches!(
            ScaleByNovograd::new().with_weight_decay(f64::NAN),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`weight_decay` must be finite but got `NaN`",
        ));
    }

    #[test]
    fn test_scale_by_novograd_initialize() {
        // The second moments are rank-0 for every leaf.
        assert_eq!(
            state_paths(&ScaleByNovograd::new()),
            vec![
                "$.count",
                "$.first_moments[\"bias\"]",
                "$.first_moments[\"weights\"]",
                "$.second_moments[\"bias\"]",
                "$.second_moments[\"weights\"]",
            ],
        );
        let parameters = vec![Array::vector(vec![1.0f32, 2.0]).unwrap()];
        let state = ScaleByNovograd::new().initialize(&parameters).unwrap();
        assert_eq!(state.second_moments()[0].r#type().as_ref(), &ArrayType::scalar(DataType::F32));
    }

    #[test]
    fn test_scale_by_novograd_update() {
        // With `β₁ = β₂ = 0.5` and `ε = 0`, the gradient `[3, 4]` has the squared norm `25`, so `v = 25` and the first
        // update is `m = [3, 4] / 5`. The gradient `[0, 1]` has the squared norm `1`, so `v = 0.5 + 12.5 = 13` and the
        // second update is `m = 0.5 · [0.6, 0.8] + [0, 1] / √13`.
        let novograd =
            ScaleByNovograd::new().with_beta1(0.5).unwrap().with_beta2(0.5).unwrap().with_epsilon(0.0).unwrap();
        assert_eq!(
            vector_updates(&novograd, &[1.0, 1.0], &[&[3.0, 4.0], &[0.0, 1.0]]),
            vec![
                vec![3.0f32 / 5.0, 4.0f32 / 5.0],
                vec![0.5f32 * (3.0f32 / 5.0) + 0.0, 0.5f32 * (4.0f32 / 5.0) + 1.0 / 13.0f32.sqrt()],
            ],
        );

        // The weight decay adds `0.5 · θ` to the normalized gradients.
        let novograd = novograd.with_weight_decay(0.5).unwrap();
        assert_eq!(
            vector_updates(&novograd, &[1.0, 2.0], &[&[3.0, 4.0]]),
            vec![vec![3.0f32 / 5.0 + 0.5, 4.0f32 / 5.0 + 1.0,]],
        );
    }

    #[test]
    fn test_scale_by_rms_optax_parity() {
        // Optax 9ecde04b: `optax.scale_by_rms(decay=0.8, eps=1e-3, initial_scale=0.5)`.
        const SCALE_BY_RMS: [[f32; 9]; 4] = [
            [
                0.15752427,
                -0.31272915,
                0.46346214,
                0.60787743,
                -0.15752427,
                0.31272915,
                -0.46346214,
                0.7445292,
                -0.8724102,
            ],
            [-0.087963305, 0.42880952, 0.1721581, 0.3358618, 0.51404244, -0.6672229, 0.1721581, -0.3292432, 0.55126655],
            [0.5703381, -0.19099836, -0.37923977, -0.2793243, 0.09566334, 0.78547066, 0.47026053, 0.69907, -0.17549802],
            [0.41744286, 0.3169357, -0.70395887, 0.20721038, -0.61656904, 0.097446084, -0.87121, 0.3848147, 0.56903976],
        ];
        let optimizer = ScaleByRms::new()
            .with_decay(0.8)
            .unwrap()
            .with_epsilon(1e-3)
            .unwrap()
            .with_initial_scale(0.5)
            .unwrap();
        assert_parity(&parity_updates(&optimizer), &SCALE_BY_RMS);

        // Optax 9ecde04b: `optax.scale_by_rms(decay=0.8, eps=1e-3, initial_scale=0.0, eps_in_sqrt=False,
        // bias_correction=True)`.
        const SCALE_BY_RMS_BIAS_CORRECTED: [[f32; 9]; 4] = [
            [
                0.99009895,
                -0.99502486,
                0.9966778,
                0.99750626,
                -0.99009895,
                0.99502486,
                -0.9966778,
                0.998004,
                -0.99833614,
            ],
            [-0.64619297, 1.0863482, 0.4663364, 0.6525178, 1.2802275, -1.2210063, 0.4663364, -0.5462267, 0.7313734],
            [1.4864877, -0.5310623, -0.95677614, -0.58920366, 0.27309397, 1.17486, 1.0863482, 1.0503683, -0.2678028],
            [0.9923525, 0.84959054, -1.3168156, 0.4645519, -1.3047079, 0.15973875, -1.3956803, 0.6041104, 0.8557635],
        ];
        let optimizer = ScaleByRms::new()
            .with_decay(0.8)
            .unwrap()
            .with_epsilon(1e-3)
            .unwrap()
            .with_initial_scale(0.0)
            .unwrap()
            .with_epsilon_in_sqrt(false)
            .with_bias_correction(true);
        assert_parity(&parity_updates(&optimizer), &SCALE_BY_RMS_BIAS_CORRECTED);
    }

    #[test]
    fn test_scale_by_stddev_optax_parity() {
        // Optax 9ecde04b: `optax.scale_by_stddev(decay=0.8, eps=1e-3, initial_scale=0.5)`.
        const SCALE_BY_STDDEV: [[f32; 9]; 4] = [
            [
                0.15760252,
                -0.31334266,
                0.46546605,
                0.61242026,
                -0.15760252,
                0.31334266,
                -0.46546605,
                0.75292325,
                -0.8860011,
            ],
            [-0.0879682, 0.42901406, 0.17335007, 0.3411043, 0.5155096, -0.6693719, 0.17235847, -0.32995933, 0.5517294],
            [
                0.57471585,
                -0.19100928,
                -0.3793812,
                -0.28070515,
                0.09602308,
                0.7886762,
                0.47089556,
                0.7128596,
                -0.17594965,
            ],
            [
                0.42533037,
                0.31739742,
                -0.70887953,
                0.20897363,
                -0.6173105,
                0.097936414,
                -0.8788411,
                0.3973672,
                0.5698076,
            ],
        ];
        let optimizer = ScaleByStddev::new()
            .with_decay(0.8)
            .unwrap()
            .with_epsilon(1e-3)
            .unwrap()
            .with_initial_scale(0.5)
            .unwrap();
        assert_parity(&parity_updates(&optimizer), &SCALE_BY_STDDEV);

        // Optax 9ecde04b: `optax.scale_by_stddev(decay=0.8, eps=1e-3, initial_scale=0.0, eps_in_sqrt=True,
        // bias_correction=True)`.
        const SCALE_BY_STDDEV_BIAS_CORRECTED: [[f32; 9]; 4] = [
            [3.162282, -6.3245907, 9.486904, 12.649393, -3.162282, 6.3245907, -9.486904, 15.811624, -18.97423],
            [-0.6175402, 1.1070186, 0.9588588, 1.9177179, 1.4905984, -1.334157, 0.49686614, -0.57262725, 0.73977494],
            [1.948894, -0.5274106, -0.95994276, -0.6489998, 0.31264213, 1.2213022, 1.1146809, 1.3145647, -0.2745156],
            [1.5330614, 0.8667279, -1.4135908, 0.5294052, -1.3183607, 0.16592452, -1.4910911, 0.8047948, 0.8622698],
        ];
        let optimizer = ScaleByStddev::new()
            .with_decay(0.8)
            .unwrap()
            .with_epsilon(1e-3)
            .unwrap()
            .with_initial_scale(0.0)
            .unwrap()
            .with_epsilon_in_sqrt(true)
            .with_bias_correction(true);
        assert_parity(&parity_updates(&optimizer), &SCALE_BY_STDDEV_BIAS_CORRECTED);
    }

    #[test]
    fn test_scale_by_rss_optax_parity() {
        // Optax 9ecde04b: `optax.scale_by_rss(initial_accumulator_value=0.2, eps=1e-3)`.
        const SCALE_BY_RSS: [[f32; 9]; 4] = [
            [0.21770017, -0.40740043, 0.556128, 0.6657427, -0.21770017, 0.40740043, -0.556128, 0.74452925, -0.80106884],
            [-0.108210914, 0.453796, 0.18227066, 0.31583324, 0.546812, -0.6316664, 0.18227066, -0.28542322, 0.42334926],
            [
                0.54455525,
                -0.17859991,
                -0.34249386,
                -0.23049663,
                0.0907592,
                0.57926047,
                0.41465583,
                0.4957578,
                -0.120081685,
            ],
            [
                0.34124526,
                0.2587746,
                -0.51409477,
                0.15188171,
                -0.4782433,
                0.06422938,
                -0.59814197,
                0.24059744,
                0.33892354,
            ],
        ];
        let optimizer = ScaleByRss::new().with_initial_accumulator_value(0.2).unwrap().with_epsilon(1e-3).unwrap();
        assert_parity(&parity_updates(&optimizer), &SCALE_BY_RSS);
    }

    #[test]
    fn test_scale_by_adadelta_optax_parity() {
        // Optax 9ecde04b: `optax.scale_by_adadelta(rho=0.8, eps=1e-3)`.
        const SCALE_BY_ADADELTA: [[f32; 9]; 4] = [
            [
                0.057735026,
                -0.06666667,
                0.06882472,
                0.06963106,
                -0.057735026,
                0.06666667,
                -0.06882472,
                0.07001401,
                -0.070224695,
            ],
            [
                -0.03666178,
                0.077022344,
                0.03345411,
                0.04771904,
                0.08533202,
                -0.08758205,
                0.03345411,
                -0.040207546,
                0.054111566,
            ],
            [
                0.0885608,
                -0.03998896,
                -0.059863176,
                -0.039374955,
                0.020857787,
                0.09539087,
                0.0682116,
                0.068724155,
                -0.018594522,
            ],
            [
                0.0720393,
                0.057674613,
                -0.08502728,
                0.028576512,
                -0.08687417,
                0.014048095,
                -0.09418711,
                0.041634664,
                0.051659364,
            ],
        ];
        let optimizer = ScaleByAdadelta::new().with_rho(0.8).unwrap().with_epsilon(1e-3).unwrap();
        assert_parity(&parity_updates(&optimizer), &SCALE_BY_ADADELTA);
    }

    #[test]
    fn test_scale_by_novograd_optax_parity() {
        // Optax 9ecde04b: `optax.scale_by_novograd(b1=0.8, b2=0.5, eps=1e-3, eps_root=1e-4, weight_decay=0.1)`.
        const SCALE_BY_NOVOGRAD: [[f32; 9]; 4] = [
            [0.31645396, -0.6329079, 0.9993619, 0.5188519, -0.15471296, 0.23442593, -0.5141389, 0.6735648, -0.5532778],
            [
                0.18284416,
                0.090204924,
                1.4033545,
                0.8080867,
                0.17243837,
                -0.24625666,
                -0.542165,
                0.5150883,
                -0.000989998,
            ],
            [1.0962481, -0.36545405, 0.9962599, 0.6813044, 0.15567619, 0.4203881, -0.40959767, 1.2085596, -0.113124676],
            [1.5941248, -0.10250023, 0.4486553, 0.9865061, -0.3300337, 0.47248536, -1.3071405, 1.6378456, 0.33967584],
        ];
        let optimizer = ScaleByNovograd::new()
            .with_beta1(0.8)
            .unwrap()
            .with_beta2(0.5)
            .unwrap()
            .with_epsilon(1e-3)
            .unwrap()
            .with_epsilon_root(1e-4)
            .unwrap()
            .with_weight_decay(0.1)
            .unwrap();
        assert_parity(&parity_updates(&optimizer), &SCALE_BY_NOVOGRAD);
    }
}
