//! Adaptive rescaling by moment estimates of the gradients, which is the core of the Adam family of optimizers.
//!
//! [`ScaleByAdam`] is the rescaling of Adam (and, with Nesterov momentum, of NAdam), and the other transformations of
//! this module are its variants: [`ScaleByAmsgrad`] divides by the running maximum of the second moments,
//! [`ScaleByAdamax`] by an exponentially weighted infinity norm, [`ScaleByRadam`] rectifies the variance of the
//! adaptive learning rate, [`ScaleByYogi`] updates the second moments additively, [`ScaleByBelief`] tracks the variance
//! of the gradients around their first moments, and [`ScaleByAdan`] adds a moment of the gradient differences. They
//! all compute their moments with the exponential moving average `m = (1 - β) · g + β · m` and correct its
//! initialization bias at the step count `t` (i.e., the number of updates including the current one) by dividing by
//! `1 - βᵗ`, following the operation order of Optax.

use ryft_core::{Add, ComparisonDirection, DataType, Div, Mul, Parameter, Parameterized, Sub};

use crate::macros::parameterized_state;
use crate::{
    Optimizer, OptimizerError, OptimizerValue, UpdateArguments, bias_corrected, bias_correction, convert_to,
    filled_statistics, increment_count, initial_count, statistic_data_type, update_moment, validate_decay_rate,
    validate_finite, validate_floating_point, validate_non_negative, validate_structure,
};

// TODO(eaplatanios): Review this module.

parameterized_state! {
    /// State of [`ScaleByAdam`], which holds the number of updates applied so far and exponential moving averages of
    /// the gradients and of their squares. The moments have the structure `S` of the parameters that the state was
    /// initialized for, so their leaves are named by the paths of the parameters (e.g., `$.first_moments.weights`),
    /// and they are stored in `f32`, or in `f64` for `f64` leaves (unless [`ScaleByAdam::with_first_moment_data_type`]
    /// selects another data type for the first moments). [`ScaleByAdamax`], [`ScaleByRadam`], and [`ScaleByYogi`] use
    /// this state as well, as Optax does, and [`ScaleByAdamax`] stores exponentially weighted infinity norms of the
    /// gradients as its second moments.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct ScaleByAdamState<P, S> with family ScaleByAdamStateFamily {
        /// Rank-0 `i32` array that holds the number of updates applied so far.
        count: P,

        /// Exponential moving averages of the gradients, with the structure of the parameters.
        first_moments: S,

        /// Exponential moving averages of the squared gradients, with the structure of the parameters.
        second_moments: S,
    }
}

impl<P: Parameter, S> ScaleByAdamState<P, S> {
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

/// Transformation that rescales gradients by bias corrected estimates of their first and second moments, which is the
/// core of the Adam optimizer of [Kingma and Ba (2015)](https://arxiv.org/abs/1412.6980) and the analogue of
/// [`optax.scale_by_adam`](https://optax.readthedocs.io/en/latest/api/transformations.html#optax.scale_by_adam). For a
/// gradient `g` at the step count `t` (i.e., the number of updates including the current one), it updates the moments
/// as `m = β₁ · m + (1 - β₁) · g` and `v = β₂ · v + (1 - β₂) · g²`, corrects their initialization bias as
/// `m̂ = m / (1 - β₁ᵗ)` and `v̂ = v / (1 - β₂ᵗ)`, and returns the update `m̂ / (√(v̂ + ε̄) + ε)`. With Nesterov
/// momentum, which is the NAdam rescaling of Dozat (2016), the first moment estimate is
/// `m̂ = β₁ · m / (1 - β₁ᵗ⁺¹) + (1 - β₁) · g / (1 - β₁ᵗ)` instead. The update does not include
/// the learning rate or its sign, which a following transformation applies (e.g.,
/// [`ScaleBySchedule::learning_rate`](crate::ScaleBySchedule::learning_rate)).
#[derive(Clone, Debug, PartialEq)]
pub struct ScaleByAdam {
    /// Decay rate `β₁` of the first moment estimates.
    beta1: f64,

    /// Decay rate `β₂` of the second moment estimates.
    beta2: f64,

    /// Term `ε` that is added to the square root of the second moment estimates for numerical stability.
    epsilon: f64,

    /// Term `ε̄` that is added to the second moment estimates inside the square root for numerical stability.
    epsilon_root: f64,

    /// Whether the first moment estimates use Nesterov momentum.
    nesterov: bool,

    /// Data type in which the first moments are stored, or [`None`] for the statistic data type of each leaf.
    first_moment_data_type: Option<DataType>,
}

impl ScaleByAdam {
    /// Creates a new [`ScaleByAdam`] with the default hyperparameters of Adam, which are `β₁ = 0.9`, `β₂ = 0.999`,
    /// `ε = 1e-8`, and `ε̄ = 0`, without Nesterov momentum.
    #[inline]
    pub fn new() -> Self {
        Self {
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            epsilon_root: 0.0,
            nesterov: false,
            first_moment_data_type: None,
        }
    }

    /// Returns this transformation with `beta1` as the decay rate of the first moment estimates.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `beta1` is not in `[0, 1)`.
    pub fn with_beta1(mut self, beta1: f64) -> Result<Self, OptimizerError> {
        validate_decay_rate("beta1", beta1)?;
        self.beta1 = beta1;
        Ok(self)
    }

    /// Returns this transformation with `beta2` as the decay rate of the second moment estimates.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `beta2` is not in `[0, 1)`.
    pub fn with_beta2(mut self, beta2: f64) -> Result<Self, OptimizerError> {
        validate_decay_rate("beta2", beta2)?;
        self.beta2 = beta2;
        Ok(self)
    }

    /// Returns this transformation with `epsilon` as the term that is added to the square root of the second moment
    /// estimates.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `epsilon` is negative or not finite.
    pub fn with_epsilon(mut self, epsilon: f64) -> Result<Self, OptimizerError> {
        validate_non_negative("epsilon", epsilon)?;
        self.epsilon = epsilon;
        Ok(self)
    }

    /// Returns this transformation with `epsilon_root` as the term that is added to the second moment estimates inside
    /// the square root (i.e., the `eps_root` of Optax), which keeps the derivative of the update finite when the
    /// update is differentiated (e.g., in meta-learning).
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `epsilon_root` is negative or not finite.
    pub fn with_epsilon_root(mut self, epsilon_root: f64) -> Result<Self, OptimizerError> {
        validate_non_negative("epsilon_root", epsilon_root)?;
        self.epsilon_root = epsilon_root;
        Ok(self)
    }

    /// Returns this transformation with Nesterov momentum in its first moment estimates when `nesterov` is `true`,
    /// which turns Adam into NAdam.
    #[inline]
    pub fn with_nesterov(mut self, nesterov: bool) -> Self {
        self.nesterov = nesterov;
        self
    }

    /// Returns this transformation with its first moments stored in `data_type` (i.e., the `mu_dtype` of Optax). The
    /// moments are still computed in the statistic data type of each leaf and converted to `data_type` when they are
    /// stored.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `data_type` is not a floating-point data type.
    pub fn with_first_moment_data_type(mut self, data_type: DataType) -> Result<Self, OptimizerError> {
        validate_floating_point("first_moment_data_type", data_type)?;
        self.first_moment_data_type = Some(data_type);
        Ok(self)
    }

    /// Returns the decay rate `β₁` of the first moment estimates of this transformation.
    #[inline]
    pub fn beta1(&self) -> f64 {
        self.beta1
    }

    /// Returns the decay rate `β₂` of the second moment estimates of this transformation.
    #[inline]
    pub fn beta2(&self) -> f64 {
        self.beta2
    }

    /// Returns the term `ε` that this transformation adds to the square root of the second moment estimates.
    #[inline]
    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    /// Returns the term `ε̄` that this transformation adds to the second moment estimates inside the square root.
    #[inline]
    pub fn epsilon_root(&self) -> f64 {
        self.epsilon_root
    }

    /// Returns whether the first moment estimates of this transformation use Nesterov momentum.
    #[inline]
    pub fn nesterov(&self) -> bool {
        self.nesterov
    }

    /// Returns the data type in which this transformation stores its first moments, or [`None`] if it stores them in
    /// the statistic data type of each leaf.
    #[inline]
    pub fn first_moment_data_type(&self) -> Option<DataType> {
        self.first_moment_data_type
    }
}

impl Default for ScaleByAdam {
    fn default() -> Self {
        Self::new()
    }
}

impl<P: OptimizerValue> Optimizer<P> for ScaleByAdam {
    type State<S: Parameterized<P>> = ScaleByAdamState<P, S>;

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        Ok(ScaleByAdamState {
            count: initial_count("ScaleByAdam", parameters)?,
            first_moments: filled_statistics(parameters, 0.0, self.first_moment_data_type)?,
            second_moments: filled_statistics(parameters, 0.0, None)?,
        })
    }

    fn update<S: Parameterized<P>>(
        &self,
        gradients: S,
        state: Self::State<S>,
        parameters: &S,
        arguments: &UpdateArguments<'_, P, S>,
    ) -> Result<(S, Self::State<S>), OptimizerError> {
        // The moment updates follow the operation order of `optax.scale_by_adam`, so that `f32` results match it.
        let _ = arguments;
        let paths = gradients.parameter_paths().collect::<Vec<_>>();
        validate_structure("parameters", &paths, parameters)?;
        validate_structure("first_moments", &paths, &state.first_moments)?;
        validate_structure("second_moments", &paths, &state.second_moments)?;
        let count = increment_count(&state.count)?;
        let step = convert_to(&count, DataType::F32)?;
        let first_correction = bias_correction(&step, self.beta1)?;
        let second_correction = bias_correction(&step, self.beta2)?;
        let next_first_correction = if self.nesterov {
            Some(bias_correction(&convert_to(&increment_count(&count)?, DataType::F32)?, self.beta1)?)
        } else {
            None
        };
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
            let first_moment = update_moment(&gradient, &convert_to(&first_moment, statistic_type)?, self.beta1)?;
            let second_moment = update_moment(
                &Mul::mul(&gradient, &gradient)?,
                &convert_to(&second_moment, statistic_type)?,
                self.beta2,
            )?;
            let first_estimate = first_moment_estimate(
                &first_moment,
                &gradient,
                self.beta1,
                &first_correction,
                next_first_correction.as_ref(),
            )?;
            let second_estimate = bias_corrected(&second_moment, &second_correction)?;
            let update =
                Div::div(&first_estimate, &root_denominator(&second_estimate, self.epsilon_root, self.epsilon)?)?;
            updates.push(convert_to(&update, data_type)?);
            first_moments.push(convert_to(&first_moment, first_moment_type)?);
            second_moments.push(convert_to(&second_moment, second_moment_type)?);
        }
        let state = ScaleByAdamState {
            count,
            first_moments: S::from_parameters(structure.clone(), first_moments)?,
            second_moments: S::from_parameters(structure.clone(), second_moments)?,
        };
        Ok((S::from_parameters(structure, updates)?, state))
    }
}

parameterized_state! {
    /// State of [`ScaleByAmsgrad`], which holds the state of [`ScaleByAdam`] together with the running maxima of the
    /// (bias corrected) second moments, with the structure `S` of the parameters.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct ScaleByAmsgradState<P, S> with family ScaleByAmsgradStateFamily {
        /// Rank-0 `i32` array that holds the number of updates applied so far.
        count: P,

        /// Exponential moving averages of the gradients, with the structure of the parameters.
        first_moments: S,

        /// Exponential moving averages of the squared gradients, with the structure of the parameters.
        second_moments: S,

        /// Running maxima of the second moment estimates, with the structure of the parameters.
        maximum_second_moments: S,
    }
}

impl<P: Parameter, S> ScaleByAmsgradState<P, S> {
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

    /// Returns the running maxima of the second moment estimates, with the structure of the parameters.
    #[inline]
    pub fn maximum_second_moments(&self) -> &S {
        &self.maximum_second_moments
    }
}

/// Transformation that rescales gradients like [`ScaleByAdam`] but divides by the running maximum of the second moment
/// estimates, which is the AMSGrad rescaling of [Reddi et al. (2018)](https://openreview.net/forum?id=ryQu7f-RZ) and
/// the analogue of
/// [`optax.scale_by_amsgrad`](https://optax.readthedocs.io/en/latest/api/transformations.html#optax.scale_by_amsgrad).
/// With the moments `m` and `v` of [`ScaleByAdam`] and the step count `t`, it computes `m̂ = m / (1 - β₁ᵗ)` and
/// `v̂ = v / (1 - β₂ᵗ)` (where either bias correction can be disabled), updates the running maximum as
/// `v̂ₘₐₓ = max(v̂ₘₐₓ, v̂)`, and returns the update `m̂ / (√(v̂ₘₐₓ + ε̄) + ε)`. The defaults are `β₁ = 0.9`,
/// `β₂ = 0.999`, `ε = 1e-8`, and `ε̄ = 0`, with both bias corrections.
#[derive(Clone, Debug, PartialEq)]
pub struct ScaleByAmsgrad {
    /// Decay rate `β₁` of the first moment estimates.
    beta1: f64,

    /// Decay rate `β₂` of the second moment estimates.
    beta2: f64,

    /// Term `ε` that is added to the square root of the maximum second moment estimates.
    epsilon: f64,

    /// Term `ε̄` that is added to the maximum second moment estimates inside the square root.
    epsilon_root: f64,

    /// Data type in which the first moments are stored, or [`None`] for the statistic data type of each leaf.
    first_moment_data_type: Option<DataType>,

    /// Whether the first moments are corrected for their initialization bias.
    first_moment_bias_correction: bool,

    /// Whether the second moments are corrected for their initialization bias before their maximum is taken.
    second_moment_bias_correction: bool,
}

impl ScaleByAmsgrad {
    /// Creates a new [`ScaleByAmsgrad`] with the default hyperparameters.
    #[inline]
    pub fn new() -> Self {
        Self {
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            epsilon_root: 0.0,
            first_moment_data_type: None,
            first_moment_bias_correction: true,
            second_moment_bias_correction: true,
        }
    }

    /// Returns this transformation with `beta1` as the decay rate of the first moment estimates.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `beta1` is not in `[0, 1)`.
    pub fn with_beta1(mut self, beta1: f64) -> Result<Self, OptimizerError> {
        validate_decay_rate("beta1", beta1)?;
        self.beta1 = beta1;
        Ok(self)
    }

    /// Returns this transformation with `beta2` as the decay rate of the second moment estimates.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `beta2` is not in `[0, 1)`.
    pub fn with_beta2(mut self, beta2: f64) -> Result<Self, OptimizerError> {
        validate_decay_rate("beta2", beta2)?;
        self.beta2 = beta2;
        Ok(self)
    }

    /// Returns this transformation with `epsilon` as the term that is added to the square root of the maximum second
    /// moment estimates.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `epsilon` is negative or not finite.
    pub fn with_epsilon(mut self, epsilon: f64) -> Result<Self, OptimizerError> {
        validate_non_negative("epsilon", epsilon)?;
        self.epsilon = epsilon;
        Ok(self)
    }

    /// Returns this transformation with `epsilon_root` as the term that is added to the maximum second moment
    /// estimates inside the square root.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `epsilon_root` is negative or not finite.
    pub fn with_epsilon_root(mut self, epsilon_root: f64) -> Result<Self, OptimizerError> {
        validate_non_negative("epsilon_root", epsilon_root)?;
        self.epsilon_root = epsilon_root;
        Ok(self)
    }

    /// Returns this transformation with its first moments stored in `data_type` (refer to
    /// [`ScaleByAdam::with_first_moment_data_type`]).
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `data_type` is not a floating-point data type.
    pub fn with_first_moment_data_type(mut self, data_type: DataType) -> Result<Self, OptimizerError> {
        validate_floating_point("first_moment_data_type", data_type)?;
        self.first_moment_data_type = Some(data_type);
        Ok(self)
    }

    /// Returns this transformation with the bias correction of its first moments enabled when `enabled` is `true`.
    #[inline]
    pub fn with_first_moment_bias_correction(mut self, enabled: bool) -> Self {
        self.first_moment_bias_correction = enabled;
        self
    }

    /// Returns this transformation with the bias correction of its second moments enabled when `enabled` is `true`.
    #[inline]
    pub fn with_second_moment_bias_correction(mut self, enabled: bool) -> Self {
        self.second_moment_bias_correction = enabled;
        self
    }

    /// Returns the decay rate `β₁` of the first moment estimates of this transformation.
    #[inline]
    pub fn beta1(&self) -> f64 {
        self.beta1
    }

    /// Returns the decay rate `β₂` of the second moment estimates of this transformation.
    #[inline]
    pub fn beta2(&self) -> f64 {
        self.beta2
    }

    /// Returns the term `ε` that this transformation adds to the square root of the maximum second moment estimates.
    #[inline]
    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    /// Returns the term `ε̄` that this transformation adds to the maximum second moment estimates inside the square
    /// root.
    #[inline]
    pub fn epsilon_root(&self) -> f64 {
        self.epsilon_root
    }

    /// Returns the data type in which this transformation stores its first moments, or [`None`] if it stores them in
    /// the statistic data type of each leaf.
    #[inline]
    pub fn first_moment_data_type(&self) -> Option<DataType> {
        self.first_moment_data_type
    }

    /// Returns whether this transformation corrects the initialization bias of its first moments.
    #[inline]
    pub fn first_moment_bias_correction(&self) -> bool {
        self.first_moment_bias_correction
    }

    /// Returns whether this transformation corrects the initialization bias of its second moments.
    #[inline]
    pub fn second_moment_bias_correction(&self) -> bool {
        self.second_moment_bias_correction
    }
}

impl Default for ScaleByAmsgrad {
    fn default() -> Self {
        Self::new()
    }
}

impl<P: OptimizerValue> Optimizer<P> for ScaleByAmsgrad {
    type State<S: Parameterized<P>> = ScaleByAmsgradState<P, S>;

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        Ok(ScaleByAmsgradState {
            count: initial_count("ScaleByAmsgrad", parameters)?,
            first_moments: filled_statistics(parameters, 0.0, self.first_moment_data_type)?,
            second_moments: filled_statistics(parameters, 0.0, None)?,
            maximum_second_moments: filled_statistics(parameters, 0.0, None)?,
        })
    }

    fn update<S: Parameterized<P>>(
        &self,
        gradients: S,
        state: Self::State<S>,
        parameters: &S,
        arguments: &UpdateArguments<'_, P, S>,
    ) -> Result<(S, Self::State<S>), OptimizerError> {
        // This follows the operation order of `optax.scale_by_amsgrad`.
        let _ = arguments;
        let paths = gradients.parameter_paths().collect::<Vec<_>>();
        validate_structure("parameters", &paths, parameters)?;
        validate_structure("first_moments", &paths, &state.first_moments)?;
        validate_structure("second_moments", &paths, &state.second_moments)?;
        validate_structure("maximum_second_moments", &paths, &state.maximum_second_moments)?;
        let count = increment_count(&state.count)?;
        let step = convert_to(&count, DataType::F32)?;
        let first_correction = bias_correction(&step, self.beta1)?;
        let second_correction = bias_correction(&step, self.beta2)?;
        let structure = gradients.parameter_structure();
        let mut updates = Vec::with_capacity(paths.len());
        let mut first_moments = Vec::with_capacity(paths.len());
        let mut second_moments = Vec::with_capacity(paths.len());
        let mut maximum_second_moments = Vec::with_capacity(paths.len());
        let moments = state
            .first_moments
            .into_parameters()
            .zip(state.second_moments.into_parameters())
            .zip(state.maximum_second_moments.into_parameters());
        for (gradient, ((first_moment, second_moment), maximum_second_moment)) in
            gradients.into_parameters().zip(moments)
        {
            let data_type = gradient.r#type().data_type();
            let statistic_type = statistic_data_type(data_type);
            let first_moment_type = first_moment.r#type().data_type();
            let second_moment_type = second_moment.r#type().data_type();
            let maximum_second_moment_type = maximum_second_moment.r#type().data_type();
            let gradient = convert_to(&gradient, statistic_type)?;
            let first_moment = update_moment(&gradient, &convert_to(&first_moment, statistic_type)?, self.beta1)?;
            let second_moment = update_moment(
                &Mul::mul(&gradient, &gradient)?,
                &convert_to(&second_moment, statistic_type)?,
                self.beta2,
            )?;
            let first_estimate = if self.first_moment_bias_correction {
                bias_corrected(&first_moment, &first_correction)?
            } else {
                first_moment.clone()
            };
            let second_estimate = if self.second_moment_bias_correction {
                bias_corrected(&second_moment, &second_correction)?
            } else {
                second_moment.clone()
            };
            let maximum_second_moment = convert_to(&maximum_second_moment, statistic_type)?.max(&second_estimate)?;
            let denominator = root_denominator(&maximum_second_moment, self.epsilon_root, self.epsilon)?;
            updates.push(convert_to(&Div::div(&first_estimate, &denominator)?, data_type)?);
            first_moments.push(convert_to(&first_moment, first_moment_type)?);
            second_moments.push(convert_to(&second_moment, second_moment_type)?);
            maximum_second_moments.push(convert_to(&maximum_second_moment, maximum_second_moment_type)?);
        }
        let state = ScaleByAmsgradState {
            count,
            first_moments: S::from_parameters(structure.clone(), first_moments)?,
            second_moments: S::from_parameters(structure.clone(), second_moments)?,
            maximum_second_moments: S::from_parameters(structure.clone(), maximum_second_moments)?,
        };
        Ok((S::from_parameters(structure, updates)?, state))
    }
}

/// Transformation that rescales gradients by a bias corrected first moment estimate divided by an exponentially
/// weighted infinity norm of the gradients, which is the Adamax rescaling of
/// [Kingma and Ba (2015)](https://arxiv.org/abs/1412.6980) and the analogue of
/// [`optax.scale_by_adamax`](https://optax.readthedocs.io/en/latest/api/transformations.html#optax.scale_by_adamax).
/// For a gradient `g` at the step count `t`, it updates `m = β₁ · m + (1 - β₁) · g` and `u = max(|g| + ε, β₂ · u)` and
/// returns the update `(m / (1 - β₁ᵗ)) / u`. Its state is a [`ScaleByAdamState`] whose second moments are the norms
/// `u`. The defaults are `β₁ = 0.9`, `β₂ = 0.999`, and `ε = 1e-8`.
#[derive(Clone, Debug, PartialEq)]
pub struct ScaleByAdamax {
    /// Decay rate `β₁` of the first moment estimates.
    beta1: f64,

    /// Decay rate `β₂` of the infinity norm estimates.
    beta2: f64,

    /// Term `ε` that is added to the absolute values of the gradients before their maximum is taken.
    epsilon: f64,
}

impl ScaleByAdamax {
    /// Creates a new [`ScaleByAdamax`] with the default hyperparameters.
    #[inline]
    pub fn new() -> Self {
        Self { beta1: 0.9, beta2: 0.999, epsilon: 1e-8 }
    }

    /// Returns this transformation with `beta1` as the decay rate of the first moment estimates.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `beta1` is not in `[0, 1)`.
    pub fn with_beta1(mut self, beta1: f64) -> Result<Self, OptimizerError> {
        validate_decay_rate("beta1", beta1)?;
        self.beta1 = beta1;
        Ok(self)
    }

    /// Returns this transformation with `beta2` as the decay rate of the infinity norm estimates.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `beta2` is not in `[0, 1)`.
    pub fn with_beta2(mut self, beta2: f64) -> Result<Self, OptimizerError> {
        validate_decay_rate("beta2", beta2)?;
        self.beta2 = beta2;
        Ok(self)
    }

    /// Returns this transformation with `epsilon` as the term that is added to the absolute values of the gradients.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `epsilon` is negative or not finite.
    pub fn with_epsilon(mut self, epsilon: f64) -> Result<Self, OptimizerError> {
        validate_non_negative("epsilon", epsilon)?;
        self.epsilon = epsilon;
        Ok(self)
    }

    /// Returns the decay rate `β₁` of the first moment estimates of this transformation.
    #[inline]
    pub fn beta1(&self) -> f64 {
        self.beta1
    }

    /// Returns the decay rate `β₂` of the infinity norm estimates of this transformation.
    #[inline]
    pub fn beta2(&self) -> f64 {
        self.beta2
    }

    /// Returns the term `ε` that this transformation adds to the absolute values of the gradients.
    #[inline]
    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }
}

impl Default for ScaleByAdamax {
    fn default() -> Self {
        Self::new()
    }
}

impl<P: OptimizerValue> Optimizer<P> for ScaleByAdamax {
    type State<S: Parameterized<P>> = ScaleByAdamState<P, S>;

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        Ok(ScaleByAdamState {
            count: initial_count("ScaleByAdamax", parameters)?,
            first_moments: filled_statistics(parameters, 0.0, None)?,
            second_moments: filled_statistics(parameters, 0.0, None)?,
        })
    }

    fn update<S: Parameterized<P>>(
        &self,
        gradients: S,
        state: Self::State<S>,
        parameters: &S,
        arguments: &UpdateArguments<'_, P, S>,
    ) -> Result<(S, Self::State<S>), OptimizerError> {
        // This follows the operation order of `optax.scale_by_adamax` and `optax.tree.update_infinity_moment`.
        let _ = arguments;
        let paths = gradients.parameter_paths().collect::<Vec<_>>();
        validate_structure("parameters", &paths, parameters)?;
        validate_structure("first_moments", &paths, &state.first_moments)?;
        validate_structure("second_moments", &paths, &state.second_moments)?;
        let count = increment_count(&state.count)?;
        let first_correction = bias_correction(&convert_to(&count, DataType::F32)?, self.beta1)?;
        let structure = gradients.parameter_structure();
        let mut updates = Vec::with_capacity(paths.len());
        let mut first_moments = Vec::with_capacity(paths.len());
        let mut norms = Vec::with_capacity(paths.len());
        let moments = state.first_moments.into_parameters().zip(state.second_moments.into_parameters());
        for (gradient, (first_moment, norm)) in gradients.into_parameters().zip(moments) {
            let data_type = gradient.r#type().data_type();
            let statistic_type = statistic_data_type(data_type);
            let first_moment_type = first_moment.r#type().data_type();
            let norm_type = norm.r#type().data_type();
            let gradient = convert_to(&gradient, statistic_type)?;
            let first_moment = update_moment(&gradient, &convert_to(&first_moment, statistic_type)?, self.beta1)?;
            let decayed_norm = Mul::mul(&gradient.fill_like(self.beta2)?, &convert_to(&norm, statistic_type)?)?;
            let norm = Add::add(&gradient.abs()?, &gradient.fill_like(self.epsilon)?)?.max(&decayed_norm)?;
            let update = Div::div(&bias_corrected(&first_moment, &first_correction)?, &norm)?;
            updates.push(convert_to(&update, data_type)?);
            first_moments.push(convert_to(&first_moment, first_moment_type)?);
            norms.push(convert_to(&norm, norm_type)?);
        }
        let state = ScaleByAdamState {
            count,
            first_moments: S::from_parameters(structure.clone(), first_moments)?,
            second_moments: S::from_parameters(structure.clone(), norms)?,
        };
        Ok((S::from_parameters(structure, updates)?, state))
    }
}

/// Transformation that rescales gradients like [`ScaleByAdam`] with a rectified adaptive learning rate, which is the
/// RAdam rescaling of [Liu et al. (2020)](https://arxiv.org/abs/1908.03265) and the analogue of
/// [`optax.scale_by_radam`](https://optax.readthedocs.io/en/latest/api/transformations.html#optax.scale_by_radam).
/// With the moments `m` and `v` and the estimates `m̂` and `v̂` of [`ScaleByAdam`] (including its optional Nesterov
/// momentum) at the step count `t`, it computes the length of the approximated simple moving average
/// `ρ = ρ∞ - 2 · t · β₂ᵗ / (1 - β₂ᵗ)` with `ρ∞ = 2 / (1 - β₂) - 1`. When `ρ` is at least the threshold, the update is
/// `r · m̂ / (√(v̂ + ε̄) + ε)` with the rectification `r = √((ρ - 4) · (ρ - 2) · ρ∞ / ((ρ∞ - 4) · (ρ∞ - 2) · ρ))`, and
/// otherwise it is `m̂`. The defaults are `β₁ = 0.9`, `β₂ = 0.999`, `ε = 1e-8`, `ε̄ = 0`, and a threshold of `5`,
/// without Nesterov momentum. Its state is a [`ScaleByAdamState`].
#[derive(Clone, Debug, PartialEq)]
pub struct ScaleByRadam {
    /// Decay rate `β₁` of the first moment estimates.
    beta1: f64,

    /// Decay rate `β₂` of the second moment estimates.
    beta2: f64,

    /// Term `ε` that is added to the square root of the second moment estimates.
    epsilon: f64,

    /// Term `ε̄` that is added to the second moment estimates inside the square root.
    epsilon_root: f64,

    /// Smallest length of the approximated simple moving average for which the adaptive learning rate is used.
    threshold: f64,

    /// Whether the first moment estimates use Nesterov momentum.
    nesterov: bool,
}

impl ScaleByRadam {
    /// Creates a new [`ScaleByRadam`] with the default hyperparameters.
    #[inline]
    pub fn new() -> Self {
        Self { beta1: 0.9, beta2: 0.999, epsilon: 1e-8, epsilon_root: 0.0, threshold: 5.0, nesterov: false }
    }

    /// Returns this transformation with `beta1` as the decay rate of the first moment estimates.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `beta1` is not in `[0, 1)`.
    pub fn with_beta1(mut self, beta1: f64) -> Result<Self, OptimizerError> {
        validate_decay_rate("beta1", beta1)?;
        self.beta1 = beta1;
        Ok(self)
    }

    /// Returns this transformation with `beta2` as the decay rate of the second moment estimates.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `beta2` is not in `[0, 1)`.
    pub fn with_beta2(mut self, beta2: f64) -> Result<Self, OptimizerError> {
        validate_decay_rate("beta2", beta2)?;
        self.beta2 = beta2;
        Ok(self)
    }

    /// Returns this transformation with `epsilon` as the term that is added to the square root of the second moment
    /// estimates.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `epsilon` is negative or not finite.
    pub fn with_epsilon(mut self, epsilon: f64) -> Result<Self, OptimizerError> {
        validate_non_negative("epsilon", epsilon)?;
        self.epsilon = epsilon;
        Ok(self)
    }

    /// Returns this transformation with `epsilon_root` as the term that is added to the second moment estimates inside
    /// the square root.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `epsilon_root` is negative or not finite.
    pub fn with_epsilon_root(mut self, epsilon_root: f64) -> Result<Self, OptimizerError> {
        validate_non_negative("epsilon_root", epsilon_root)?;
        self.epsilon_root = epsilon_root;
        Ok(self)
    }

    /// Returns this transformation with `threshold` as the smallest length of the approximated simple moving average
    /// for which the adaptive learning rate is used.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `threshold` is not finite.
    pub fn with_threshold(mut self, threshold: f64) -> Result<Self, OptimizerError> {
        validate_finite("threshold", threshold)?;
        self.threshold = threshold;
        Ok(self)
    }

    /// Returns this transformation with Nesterov momentum in its first moment estimates when `nesterov` is `true`.
    #[inline]
    pub fn with_nesterov(mut self, nesterov: bool) -> Self {
        self.nesterov = nesterov;
        self
    }

    /// Returns the decay rate `β₁` of the first moment estimates of this transformation.
    #[inline]
    pub fn beta1(&self) -> f64 {
        self.beta1
    }

    /// Returns the decay rate `β₂` of the second moment estimates of this transformation.
    #[inline]
    pub fn beta2(&self) -> f64 {
        self.beta2
    }

    /// Returns the term `ε` that this transformation adds to the square root of the second moment estimates.
    #[inline]
    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    /// Returns the term `ε̄` that this transformation adds to the second moment estimates inside the square root.
    #[inline]
    pub fn epsilon_root(&self) -> f64 {
        self.epsilon_root
    }

    /// Returns the smallest length of the approximated simple moving average for which this transformation uses the
    /// adaptive learning rate.
    #[inline]
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Returns whether the first moment estimates of this transformation use Nesterov momentum.
    #[inline]
    pub fn nesterov(&self) -> bool {
        self.nesterov
    }
}

impl Default for ScaleByRadam {
    fn default() -> Self {
        Self::new()
    }
}

impl<P: OptimizerValue> Optimizer<P> for ScaleByRadam {
    type State<S: Parameterized<P>> = ScaleByAdamState<P, S>;

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        Ok(ScaleByAdamState {
            count: initial_count("ScaleByRadam", parameters)?,
            first_moments: filled_statistics(parameters, 0.0, None)?,
            second_moments: filled_statistics(parameters, 0.0, None)?,
        })
    }

    fn update<S: Parameterized<P>>(
        &self,
        gradients: S,
        state: Self::State<S>,
        parameters: &S,
        arguments: &UpdateArguments<'_, P, S>,
    ) -> Result<(S, Self::State<S>), OptimizerError> {
        // This follows the operation order of `optax.scale_by_radam`, in which the constants that only involve `ρ∞` are
        // computed in `f64` before they meet the `f32` step count.
        let _ = arguments;
        let paths = gradients.parameter_paths().collect::<Vec<_>>();
        validate_structure("parameters", &paths, parameters)?;
        validate_structure("first_moments", &paths, &state.first_moments)?;
        validate_structure("second_moments", &paths, &state.second_moments)?;
        let count = increment_count(&state.count)?;
        let step = convert_to(&count, DataType::F32)?;
        let first_correction = bias_correction(&step, self.beta1)?;
        let second_correction = bias_correction(&step, self.beta2)?;
        let next_first_correction = if self.nesterov {
            Some(bias_correction(&convert_to(&increment_count(&count)?, DataType::F32)?, self.beta1)?)
        } else {
            None
        };
        let infinite_length = 2.0 / (1.0 - self.beta2) - 1.0;
        let decay_power = step.fill_like(self.beta2)?.pow(&step)?;
        let length = Sub::sub(
            &step.fill_like(infinite_length)?,
            &Div::div(
                &Mul::mul(&Mul::mul(&step.fill_like(2f64)?, &step)?, &decay_power)?,
                &Sub::sub(&step.fill_like(1f64)?, &decay_power)?,
            )?,
        )?;
        let numerator = Mul::mul(
            &Mul::mul(&Sub::sub(&length, &step.fill_like(4f64)?)?, &Sub::sub(&length, &step.fill_like(2f64)?)?)?,
            &step.fill_like(infinite_length)?,
        )?;
        let denominator = Mul::mul(&step.fill_like((infinite_length - 4.0) * (infinite_length - 2.0))?, &length)?;
        let rectification = Div::div(&numerator, &denominator)?.sqrt()?;
        let rectified = length.compare(&step.fill_like(self.threshold)?, ComparisonDirection::GreaterThanOrEqual)?;
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
            let first_moment = update_moment(&gradient, &convert_to(&first_moment, statistic_type)?, self.beta1)?;
            let second_moment = update_moment(
                &Mul::mul(&gradient, &gradient)?,
                &convert_to(&second_moment, statistic_type)?,
                self.beta2,
            )?;
            let first_estimate = first_moment_estimate(
                &first_moment,
                &gradient,
                self.beta1,
                &first_correction,
                next_first_correction.as_ref(),
            )?;
            let second_estimate = bias_corrected(&second_moment, &second_correction)?;
            let adaptive = Div::div(
                &Mul::mul(&convert_to(&rectification, statistic_type)?, &first_estimate)?,
                &root_denominator(&second_estimate, self.epsilon_root, self.epsilon)?,
            )?;
            let update = rectified.select(&adaptive, &first_estimate)?;
            updates.push(convert_to(&update, data_type)?);
            first_moments.push(convert_to(&first_moment, first_moment_type)?);
            second_moments.push(convert_to(&second_moment, second_moment_type)?);
        }
        let state = ScaleByAdamState {
            count,
            first_moments: S::from_parameters(structure.clone(), first_moments)?,
            second_moments: S::from_parameters(structure.clone(), second_moments)?,
        };
        Ok((S::from_parameters(structure, updates)?, state))
    }
}

/// Transformation that rescales gradients like [`ScaleByAdam`] but updates the second moments additively, which is the
/// Yogi rescaling of
/// [Zaheer et al. (2018)](
/// https://proceedings.neurips.cc/paper/2018/hash/90365351ccc7437a1309dc64e4db32a3-Abstract.html)
/// and the analogue of
/// [`optax.scale_by_yogi`](https://optax.readthedocs.io/en/latest/api/transformations.html#optax.scale_by_yogi). For a
/// gradient `g`, it updates the second moments as `v = v - (1 - β₂) · sign(v - g²) · g²`, so that they change by at
/// most `(1 - β₂) · g²` per step, and otherwise computes the update `m̂ / (√(v̂ + ε̄) + ε)` of [`ScaleByAdam`]. The
/// second moments start at an initial accumulator value. The defaults are `β₁ = 0.9`, `β₂ = 0.999`, `ε = 1e-3`,
/// `ε̄ = 0`, and an initial accumulator value of `1e-6`. Its state is a [`ScaleByAdamState`].
#[derive(Clone, Debug, PartialEq)]
pub struct ScaleByYogi {
    /// Decay rate `β₁` of the first moment estimates.
    beta1: f64,

    /// Decay rate `β₂` of the second moment estimates.
    beta2: f64,

    /// Term `ε` that is added to the square root of the second moment estimates.
    epsilon: f64,

    /// Term `ε̄` that is added to the second moment estimates inside the square root.
    epsilon_root: f64,

    /// Initial value of the second moments.
    initial_accumulator_value: f64,
}

impl ScaleByYogi {
    /// Creates a new [`ScaleByYogi`] with the default hyperparameters.
    #[inline]
    pub fn new() -> Self {
        Self { beta1: 0.9, beta2: 0.999, epsilon: 1e-3, epsilon_root: 0.0, initial_accumulator_value: 1e-6 }
    }

    /// Returns this transformation with `beta1` as the decay rate of the first moment estimates.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `beta1` is not in `[0, 1)`.
    pub fn with_beta1(mut self, beta1: f64) -> Result<Self, OptimizerError> {
        validate_decay_rate("beta1", beta1)?;
        self.beta1 = beta1;
        Ok(self)
    }

    /// Returns this transformation with `beta2` as the decay rate of the second moment estimates.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `beta2` is not in `[0, 1)`.
    pub fn with_beta2(mut self, beta2: f64) -> Result<Self, OptimizerError> {
        validate_decay_rate("beta2", beta2)?;
        self.beta2 = beta2;
        Ok(self)
    }

    /// Returns this transformation with `epsilon` as the term that is added to the square root of the second moment
    /// estimates.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `epsilon` is negative or not finite.
    pub fn with_epsilon(mut self, epsilon: f64) -> Result<Self, OptimizerError> {
        validate_non_negative("epsilon", epsilon)?;
        self.epsilon = epsilon;
        Ok(self)
    }

    /// Returns this transformation with `epsilon_root` as the term that is added to the second moment estimates inside
    /// the square root.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `epsilon_root` is negative or not finite.
    pub fn with_epsilon_root(mut self, epsilon_root: f64) -> Result<Self, OptimizerError> {
        validate_non_negative("epsilon_root", epsilon_root)?;
        self.epsilon_root = epsilon_root;
        Ok(self)
    }

    /// Returns this transformation with `initial_accumulator_value` as the initial value of its second moments.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `initial_accumulator_value` is negative or not finite.
    pub fn with_initial_accumulator_value(mut self, initial_accumulator_value: f64) -> Result<Self, OptimizerError> {
        validate_non_negative("initial_accumulator_value", initial_accumulator_value)?;
        self.initial_accumulator_value = initial_accumulator_value;
        Ok(self)
    }

    /// Returns the decay rate `β₁` of the first moment estimates of this transformation.
    #[inline]
    pub fn beta1(&self) -> f64 {
        self.beta1
    }

    /// Returns the decay rate `β₂` of the second moment estimates of this transformation.
    #[inline]
    pub fn beta2(&self) -> f64 {
        self.beta2
    }

    /// Returns the term `ε` that this transformation adds to the square root of the second moment estimates.
    #[inline]
    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    /// Returns the term `ε̄` that this transformation adds to the second moment estimates inside the square root.
    #[inline]
    pub fn epsilon_root(&self) -> f64 {
        self.epsilon_root
    }

    /// Returns the initial value of the second moments of this transformation.
    #[inline]
    pub fn initial_accumulator_value(&self) -> f64 {
        self.initial_accumulator_value
    }
}

impl Default for ScaleByYogi {
    fn default() -> Self {
        Self::new()
    }
}

impl<P: OptimizerValue> Optimizer<P> for ScaleByYogi {
    type State<S: Parameterized<P>> = ScaleByAdamState<P, S>;

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        Ok(ScaleByAdamState {
            count: initial_count("ScaleByYogi", parameters)?,
            first_moments: filled_statistics(parameters, 0.0, None)?,
            second_moments: filled_statistics(parameters, self.initial_accumulator_value, None)?,
        })
    }

    fn update<S: Parameterized<P>>(
        &self,
        gradients: S,
        state: Self::State<S>,
        parameters: &S,
        arguments: &UpdateArguments<'_, P, S>,
    ) -> Result<(S, Self::State<S>), OptimizerError> {
        // This follows the operation order of `optax.scale_by_yogi`.
        let _ = arguments;
        let paths = gradients.parameter_paths().collect::<Vec<_>>();
        validate_structure("parameters", &paths, parameters)?;
        validate_structure("first_moments", &paths, &state.first_moments)?;
        validate_structure("second_moments", &paths, &state.second_moments)?;
        let count = increment_count(&state.count)?;
        let step = convert_to(&count, DataType::F32)?;
        let first_correction = bias_correction(&step, self.beta1)?;
        let second_correction = bias_correction(&step, self.beta2)?;
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
            let first_moment = update_moment(&gradient, &convert_to(&first_moment, statistic_type)?, self.beta1)?;
            let second_moment = convert_to(&second_moment, statistic_type)?;
            let squared = Mul::mul(&gradient, &gradient)?;
            let direction = Sub::sub(&second_moment, &squared)?.sign()?;
            let change = Mul::mul(&Mul::mul(&gradient.fill_like(1.0 - self.beta2)?, &direction)?, &squared)?;
            let second_moment = Sub::sub(&second_moment, &change)?;
            let first_estimate = bias_corrected(&first_moment, &first_correction)?;
            let second_estimate = bias_corrected(&second_moment, &second_correction)?;
            let update =
                Div::div(&first_estimate, &root_denominator(&second_estimate, self.epsilon_root, self.epsilon)?)?;
            updates.push(convert_to(&update, data_type)?);
            first_moments.push(convert_to(&first_moment, first_moment_type)?);
            second_moments.push(convert_to(&second_moment, second_moment_type)?);
        }
        let state = ScaleByAdamState {
            count,
            first_moments: S::from_parameters(structure.clone(), first_moments)?,
            second_moments: S::from_parameters(structure.clone(), second_moments)?,
        };
        Ok((S::from_parameters(structure, updates)?, state))
    }
}

parameterized_state! {
    /// State of [`ScaleByBelief`], which holds the number of updates applied so far, exponential moving averages of the
    /// gradients, and exponential moving averages of the squared differences between the gradients and those
    /// averages, with the structure `S` of the parameters.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct ScaleByBeliefState<P, S> with family ScaleByBeliefStateFamily {
        /// Rank-0 `i32` array that holds the number of updates applied so far.
        count: P,

        /// Exponential moving averages of the gradients, with the structure of the parameters.
        first_moments: S,

        /// Exponential moving averages of the squared prediction errors (plus `ε̄`), with the structure of the
        /// parameters.
        second_moments: S,
    }
}

impl<P: Parameter, S> ScaleByBeliefState<P, S> {
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

    /// Returns the exponential moving averages of the squared prediction errors, with the structure of the parameters.
    #[inline]
    pub fn second_moments(&self) -> &S {
        &self.second_moments
    }
}

/// Transformation that rescales gradients by their first moment estimates divided by the deviation of the gradients
/// from those estimates, which is the AdaBelief rescaling of [Zhuang et al. (2020)](https://arxiv.org/abs/2010.07468)
/// and the analogue of
/// [`optax.scale_by_belief`](https://optax.readthedocs.io/en/latest/api/transformations.html#optax.scale_by_belief).
/// For a gradient `g`, it updates `m = β₁ · m + (1 - β₁) · g` and `s = β₂ · s + (1 - β₂) · (g - m)² + ε̄`, computes the
/// estimates `m̂` (including the optional Nesterov momentum of [`ScaleByAdam`]) and `ŝ = s / (1 - β₂ᵗ)`, and returns
/// the update `m̂ / (√ŝ + ε)`. The defaults are `β₁ = 0.9`, `β₂ = 0.999`, `ε = 1e-16`, and `ε̄ = 1e-16`, without
/// Nesterov momentum.
#[derive(Clone, Debug, PartialEq)]
pub struct ScaleByBelief {
    /// Decay rate `β₁` of the first moment estimates.
    beta1: f64,

    /// Decay rate `β₂` of the second moment estimates.
    beta2: f64,

    /// Term `ε` that is added to the square root of the second moment estimates.
    epsilon: f64,

    /// Term `ε̄` that is added to the second moments after every update.
    epsilon_root: f64,

    /// Whether the first moment estimates use Nesterov momentum.
    nesterov: bool,
}

impl ScaleByBelief {
    /// Creates a new [`ScaleByBelief`] with the default hyperparameters.
    #[inline]
    pub fn new() -> Self {
        Self { beta1: 0.9, beta2: 0.999, epsilon: 1e-16, epsilon_root: 1e-16, nesterov: false }
    }

    /// Returns this transformation with `beta1` as the decay rate of the first moment estimates.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `beta1` is not in `[0, 1)`.
    pub fn with_beta1(mut self, beta1: f64) -> Result<Self, OptimizerError> {
        validate_decay_rate("beta1", beta1)?;
        self.beta1 = beta1;
        Ok(self)
    }

    /// Returns this transformation with `beta2` as the decay rate of the second moment estimates.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `beta2` is not in `[0, 1)`.
    pub fn with_beta2(mut self, beta2: f64) -> Result<Self, OptimizerError> {
        validate_decay_rate("beta2", beta2)?;
        self.beta2 = beta2;
        Ok(self)
    }

    /// Returns this transformation with `epsilon` as the term that is added to the square root of the second moment
    /// estimates.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `epsilon` is negative or not finite.
    pub fn with_epsilon(mut self, epsilon: f64) -> Result<Self, OptimizerError> {
        validate_non_negative("epsilon", epsilon)?;
        self.epsilon = epsilon;
        Ok(self)
    }

    /// Returns this transformation with `epsilon_root` as the term that is added to the second moments after every
    /// update.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `epsilon_root` is negative or not finite.
    pub fn with_epsilon_root(mut self, epsilon_root: f64) -> Result<Self, OptimizerError> {
        validate_non_negative("epsilon_root", epsilon_root)?;
        self.epsilon_root = epsilon_root;
        Ok(self)
    }

    /// Returns this transformation with Nesterov momentum in its first moment estimates when `nesterov` is `true`.
    #[inline]
    pub fn with_nesterov(mut self, nesterov: bool) -> Self {
        self.nesterov = nesterov;
        self
    }

    /// Returns the decay rate `β₁` of the first moment estimates of this transformation.
    #[inline]
    pub fn beta1(&self) -> f64 {
        self.beta1
    }

    /// Returns the decay rate `β₂` of the second moment estimates of this transformation.
    #[inline]
    pub fn beta2(&self) -> f64 {
        self.beta2
    }

    /// Returns the term `ε` that this transformation adds to the square root of the second moment estimates.
    #[inline]
    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    /// Returns the term `ε̄` that this transformation adds to the second moments after every update.
    #[inline]
    pub fn epsilon_root(&self) -> f64 {
        self.epsilon_root
    }

    /// Returns whether the first moment estimates of this transformation use Nesterov momentum.
    #[inline]
    pub fn nesterov(&self) -> bool {
        self.nesterov
    }
}

impl Default for ScaleByBelief {
    fn default() -> Self {
        Self::new()
    }
}

impl<P: OptimizerValue> Optimizer<P> for ScaleByBelief {
    type State<S: Parameterized<P>> = ScaleByBeliefState<P, S>;

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        Ok(ScaleByBeliefState {
            count: initial_count("ScaleByBelief", parameters)?,
            first_moments: filled_statistics(parameters, 0.0, None)?,
            second_moments: filled_statistics(parameters, 0.0, None)?,
        })
    }

    fn update<S: Parameterized<P>>(
        &self,
        gradients: S,
        state: Self::State<S>,
        parameters: &S,
        arguments: &UpdateArguments<'_, P, S>,
    ) -> Result<(S, Self::State<S>), OptimizerError> {
        // This follows the operation order of `optax.scale_by_belief`.
        let _ = arguments;
        let paths = gradients.parameter_paths().collect::<Vec<_>>();
        validate_structure("parameters", &paths, parameters)?;
        validate_structure("first_moments", &paths, &state.first_moments)?;
        validate_structure("second_moments", &paths, &state.second_moments)?;
        let count = increment_count(&state.count)?;
        let step = convert_to(&count, DataType::F32)?;
        let first_correction = bias_correction(&step, self.beta1)?;
        let second_correction = bias_correction(&step, self.beta2)?;
        let next_first_correction = if self.nesterov {
            Some(bias_correction(&convert_to(&increment_count(&count)?, DataType::F32)?, self.beta1)?)
        } else {
            None
        };
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
            let first_moment = update_moment(&gradient, &convert_to(&first_moment, statistic_type)?, self.beta1)?;
            let prediction_error = Sub::sub(&gradient, &first_moment)?;
            let second_moment = update_moment(
                &Mul::mul(&prediction_error, &prediction_error)?,
                &convert_to(&second_moment, statistic_type)?,
                self.beta2,
            )?;
            let second_moment = Add::add(&second_moment, &gradient.fill_like(self.epsilon_root)?)?;
            let first_estimate = first_moment_estimate(
                &first_moment,
                &gradient,
                self.beta1,
                &first_correction,
                next_first_correction.as_ref(),
            )?;
            let second_estimate = bias_corrected(&second_moment, &second_correction)?;
            let denominator = Add::add(&second_estimate.sqrt()?, &gradient.fill_like(self.epsilon)?)?;
            updates.push(convert_to(&Div::div(&first_estimate, &denominator)?, data_type)?);
            first_moments.push(convert_to(&first_moment, first_moment_type)?);
            second_moments.push(convert_to(&second_moment, second_moment_type)?);
        }
        let state = ScaleByBeliefState {
            count,
            first_moments: S::from_parameters(structure.clone(), first_moments)?,
            second_moments: S::from_parameters(structure.clone(), second_moments)?,
        };
        Ok((S::from_parameters(structure, updates)?, state))
    }
}

parameterized_state! {
    /// State of [`ScaleByAdan`], which holds the number of updates applied so far, the three moments of Adan, and the
    /// gradients of the previous update, all but the count with the structure `S` of the parameters.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct ScaleByAdanState<P, S> with family ScaleByAdanStateFamily {
        /// Rank-0 `i32` array that holds the number of updates applied so far.
        count: P,

        /// Exponential moving averages of the gradients, with the structure of the parameters.
        first_moments: S,

        /// Exponential moving averages of the differences between consecutive gradients, with the structure of the
        /// parameters.
        difference_moments: S,

        /// Exponential moving averages of the squared corrected gradients, with the structure of the parameters.
        second_moments: S,

        /// Gradients of the previous update, with the structure of the parameters.
        previous_gradients: S,
    }
}

impl<P: Parameter, S> ScaleByAdanState<P, S> {
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

    /// Returns the exponential moving averages of the differences between consecutive gradients.
    #[inline]
    pub fn difference_moments(&self) -> &S {
        &self.difference_moments
    }

    /// Returns the exponential moving averages of the squared corrected gradients.
    #[inline]
    pub fn second_moments(&self) -> &S {
        &self.second_moments
    }

    /// Returns the gradients of the previous update, with the structure of the parameters.
    #[inline]
    pub fn previous_gradients(&self) -> &S {
        &self.previous_gradients
    }
}

/// Transformation that rescales gradients by Nesterov-corrected moment estimates, which is the Adan rescaling of
/// [Xie et al. (2022)](https://arxiv.org/abs/2208.06677) and the analogue of
/// [`optax.scale_by_adan`](https://optax.readthedocs.io/en/latest/api/transformations.html#optax.scale_by_adan).
/// For a gradient `g`, the gradient difference `d = g - g₋₁` (which is zero at the first update), and the step count
/// `t`, it updates `m = β₁ · m + (1 - β₁) · g`, `v = β₂ · v + (1 - β₂) · d`, and
/// `n = β₃ · n + (1 - β₃) · (g + (1 - β₂) · d)²`, and returns the update
/// `(m̂ + (1 - β₂) · v̂) / (√(n̂ + ε̄) + ε)` with the bias corrected estimates `m̂`, `v̂`, and `n̂`. The defaults are
/// `β₁ = 0.98`, `β₂ = 0.92`, `β₃ = 0.99`, `ε = 1e-8`, and `ε̄ = 0`.
#[derive(Clone, Debug, PartialEq)]
pub struct ScaleByAdan {
    /// Decay rate `β₁` of the first moment estimates.
    beta1: f64,

    /// Decay rate `β₂` of the gradient difference estimates.
    beta2: f64,

    /// Decay rate `β₃` of the second moment estimates.
    beta3: f64,

    /// Term `ε` that is added to the square root of the second moment estimates.
    epsilon: f64,

    /// Term `ε̄` that is added to the second moment estimates inside the square root.
    epsilon_root: f64,
}

impl ScaleByAdan {
    /// Creates a new [`ScaleByAdan`] with the default hyperparameters.
    #[inline]
    pub fn new() -> Self {
        Self { beta1: 0.98, beta2: 0.92, beta3: 0.99, epsilon: 1e-8, epsilon_root: 0.0 }
    }

    /// Returns this transformation with `beta1` as the decay rate of the first moment estimates.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `beta1` is not in `[0, 1)`.
    pub fn with_beta1(mut self, beta1: f64) -> Result<Self, OptimizerError> {
        validate_decay_rate("beta1", beta1)?;
        self.beta1 = beta1;
        Ok(self)
    }

    /// Returns this transformation with `beta2` as the decay rate of the gradient difference estimates.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `beta2` is not in `[0, 1)`.
    pub fn with_beta2(mut self, beta2: f64) -> Result<Self, OptimizerError> {
        validate_decay_rate("beta2", beta2)?;
        self.beta2 = beta2;
        Ok(self)
    }

    /// Returns this transformation with `beta3` as the decay rate of the second moment estimates.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `beta3` is not in `[0, 1)`.
    pub fn with_beta3(mut self, beta3: f64) -> Result<Self, OptimizerError> {
        validate_decay_rate("beta3", beta3)?;
        self.beta3 = beta3;
        Ok(self)
    }

    /// Returns this transformation with `epsilon` as the term that is added to the square root of the second moment
    /// estimates.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `epsilon` is negative or not finite.
    pub fn with_epsilon(mut self, epsilon: f64) -> Result<Self, OptimizerError> {
        validate_non_negative("epsilon", epsilon)?;
        self.epsilon = epsilon;
        Ok(self)
    }

    /// Returns this transformation with `epsilon_root` as the term that is added to the second moment estimates inside
    /// the square root.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `epsilon_root` is negative or not finite.
    pub fn with_epsilon_root(mut self, epsilon_root: f64) -> Result<Self, OptimizerError> {
        validate_non_negative("epsilon_root", epsilon_root)?;
        self.epsilon_root = epsilon_root;
        Ok(self)
    }

    /// Returns the decay rate `β₁` of the first moment estimates of this transformation.
    #[inline]
    pub fn beta1(&self) -> f64 {
        self.beta1
    }

    /// Returns the decay rate `β₂` of the gradient difference estimates of this transformation.
    #[inline]
    pub fn beta2(&self) -> f64 {
        self.beta2
    }

    /// Returns the decay rate `β₃` of the second moment estimates of this transformation.
    #[inline]
    pub fn beta3(&self) -> f64 {
        self.beta3
    }

    /// Returns the term `ε` that this transformation adds to the square root of the second moment estimates.
    #[inline]
    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    /// Returns the term `ε̄` that this transformation adds to the second moment estimates inside the square root.
    #[inline]
    pub fn epsilon_root(&self) -> f64 {
        self.epsilon_root
    }
}

impl Default for ScaleByAdan {
    fn default() -> Self {
        Self::new()
    }
}

impl<P: OptimizerValue> Optimizer<P> for ScaleByAdan {
    type State<S: Parameterized<P>> = ScaleByAdanState<P, S>;

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        Ok(ScaleByAdanState {
            count: initial_count("ScaleByAdan", parameters)?,
            first_moments: filled_statistics(parameters, 0.0, None)?,
            difference_moments: filled_statistics(parameters, 0.0, None)?,
            second_moments: filled_statistics(parameters, 0.0, None)?,
            previous_gradients: filled_statistics(parameters, 0.0, None)?,
        })
    }

    fn update<S: Parameterized<P>>(
        &self,
        gradients: S,
        state: Self::State<S>,
        parameters: &S,
        arguments: &UpdateArguments<'_, P, S>,
    ) -> Result<(S, Self::State<S>), OptimizerError> {
        // This follows the operation order of `optax.scale_by_adan`.
        let _ = arguments;
        let paths = gradients.parameter_paths().collect::<Vec<_>>();
        validate_structure("parameters", &paths, parameters)?;
        validate_structure("first_moments", &paths, &state.first_moments)?;
        validate_structure("difference_moments", &paths, &state.difference_moments)?;
        validate_structure("second_moments", &paths, &state.second_moments)?;
        validate_structure("previous_gradients", &paths, &state.previous_gradients)?;
        let first_update = state.count.compare(&state.count.fill_like(0i32)?, ComparisonDirection::Equal)?;
        let count = increment_count(&state.count)?;
        let step = convert_to(&count, DataType::F32)?;
        let first_correction = bias_correction(&step, self.beta1)?;
        let difference_correction = bias_correction(&step, self.beta2)?;
        let second_correction = bias_correction(&step, self.beta3)?;
        let structure = gradients.parameter_structure();
        let mut updates = Vec::with_capacity(paths.len());
        let mut first_moments = Vec::with_capacity(paths.len());
        let mut difference_moments = Vec::with_capacity(paths.len());
        let mut second_moments = Vec::with_capacity(paths.len());
        let mut previous_gradients = Vec::with_capacity(paths.len());
        let statistics = state
            .first_moments
            .into_parameters()
            .zip(state.difference_moments.into_parameters())
            .zip(state.second_moments.into_parameters())
            .zip(state.previous_gradients.into_parameters());
        for (gradient, (((first_moment, difference_moment), second_moment), previous_gradient)) in
            gradients.into_parameters().zip(statistics)
        {
            let data_type = gradient.r#type().data_type();
            let statistic_type = statistic_data_type(data_type);
            let first_moment_type = first_moment.r#type().data_type();
            let difference_moment_type = difference_moment.r#type().data_type();
            let second_moment_type = second_moment.r#type().data_type();
            let previous_gradient_type = previous_gradient.r#type().data_type();
            let gradient = convert_to(&gradient, statistic_type)?;
            let difference = Sub::sub(&gradient, &convert_to(&previous_gradient, statistic_type)?)?;
            let difference = first_update.select(&gradient.fill_like(0f64)?, &difference)?;
            let first_moment = update_moment(&gradient, &convert_to(&first_moment, statistic_type)?, self.beta1)?;
            let difference_moment =
                update_moment(&difference, &convert_to(&difference_moment, statistic_type)?, self.beta2)?;
            let corrected = Add::add(&gradient, &Mul::mul(&gradient.fill_like(1.0 - self.beta2)?, &difference)?)?;
            let second_moment = update_moment(
                &Mul::mul(&corrected, &corrected)?,
                &convert_to(&second_moment, statistic_type)?,
                self.beta3,
            )?;
            let first_estimate = bias_corrected(&first_moment, &first_correction)?;
            let difference_estimate = bias_corrected(&difference_moment, &difference_correction)?;
            let second_estimate = bias_corrected(&second_moment, &second_correction)?;
            let numerator =
                Add::add(&first_estimate, &Mul::mul(&gradient.fill_like(1.0 - self.beta2)?, &difference_estimate)?)?;
            let update = Div::div(&numerator, &root_denominator(&second_estimate, self.epsilon_root, self.epsilon)?)?;
            updates.push(convert_to(&update, data_type)?);
            first_moments.push(convert_to(&first_moment, first_moment_type)?);
            difference_moments.push(convert_to(&difference_moment, difference_moment_type)?);
            second_moments.push(convert_to(&second_moment, second_moment_type)?);
            previous_gradients.push(convert_to(&gradient, previous_gradient_type)?);
        }
        let state = ScaleByAdanState {
            count,
            first_moments: S::from_parameters(structure.clone(), first_moments)?,
            difference_moments: S::from_parameters(structure.clone(), difference_moments)?,
            second_moments: S::from_parameters(structure.clone(), second_moments)?,
            previous_gradients: S::from_parameters(structure.clone(), previous_gradients)?,
        };
        Ok((S::from_parameters(structure, updates)?, state))
    }
}

/// Returns the bias corrected first moment estimate of the Adam family for the first moment `first_moment` and the
/// gradient `gradient`, given the rank-0 bias `correction` at the current step count `t`. With the correction
/// `next_correction` at step `t + 1`, it uses Nesterov momentum and returns
/// `β₁ · m / (1 - β₁ᵗ⁺¹) + (1 - β₁) · g / (1 - β₁ᵗ)`, which is the operation order of Optax, and otherwise it returns
/// `m / (1 - β₁ᵗ)`.
fn first_moment_estimate<P: OptimizerValue>(
    first_moment: &P,
    gradient: &P,
    beta1: f64,
    correction: &P,
    next_correction: Option<&P>,
) -> Result<P, OptimizerError> {
    match next_correction {
        None => bias_corrected(first_moment, correction),
        Some(next_correction) => Ok(Add::add(
            &Mul::mul(&first_moment.fill_like(beta1)?, &bias_corrected(first_moment, next_correction)?)?,
            &Mul::mul(&gradient.fill_like(1.0 - beta1)?, &bias_corrected(gradient, correction)?)?,
        )?),
    }
}

/// Returns the denominator `√(v + ε̄) + ε` of the Adam family for the second moment estimate `second_estimate`. A zero
/// `epsilon_root` is not added, which leaves the result unchanged.
fn root_denominator<P: OptimizerValue>(
    second_estimate: &P,
    epsilon_root: f64,
    epsilon: f64,
) -> Result<P, OptimizerError> {
    let shifted = if epsilon_root == 0.0 {
        second_estimate.clone()
    } else {
        Add::add(second_estimate, &second_estimate.fill_like(epsilon_root)?)?
    };
    Ok(Add::add(&shifted.sqrt()?, &second_estimate.fill_like(epsilon)?)?)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use pretty_assertions::assert_eq;

    use ryft_core::{Array, ArrayType, Typed, bf16};

    use crate::apply_updates;
    use crate::tests::{
        PARITY_GRADIENTS, PARITY_PARAMETERS, assert_parity, parity_elements, parity_parameters, parity_updates,
        state_paths, vector_updates,
    };

    use super::*;

    /// Returns the updates that Adam computes for the sequence of gradients `gradients` of one element, repeating the
    /// `f32` operations of [`ScaleByAdam::update`] in the same order.
    fn reference_adam_updates(gradients: &[f32], beta1: f64, beta2: f64, epsilon: f64) -> Vec<f32> {
        let (mut first_moment, mut second_moment) = (0f32, 0f32);
        let mut updates = Vec::new();
        for (index, gradient) in gradients.iter().enumerate() {
            let step = (index + 1) as f32;
            first_moment = (1.0 - beta1) as f32 * gradient + beta1 as f32 * first_moment;
            second_moment = (1.0 - beta2) as f32 * (gradient * gradient) + beta2 as f32 * second_moment;
            let first_correction = 1f32 - (beta1 as f32).powf(step);
            let second_correction = 1f32 - (beta2 as f32).powf(step);
            let denominator = (second_moment / second_correction).sqrt() + epsilon as f32;
            updates.push(first_moment / first_correction / denominator);
        }
        updates
    }

    #[test]
    fn test_scale_by_adam_state_parameterized() {
        // The leaves of the state are named by the paths of the parameters, and the state can be rebuilt from its
        // structure and leaves.
        let parameters = BTreeMap::from([
            ("bias", Array::scalar(1.0f32).unwrap()),
            ("weights", Array::vector(vec![2.0f32, 3.0]).unwrap()),
        ]);
        let state = ScaleByAdam::new().initialize(&parameters).unwrap();
        assert_eq!(
            state.named_parameters().map(|(path, _)| path.to_string()).collect::<Vec<_>>(),
            vec![
                "$.count",
                "$.first_moments[\"bias\"]",
                "$.first_moments[\"weights\"]",
                "$.second_moments[\"bias\"]",
                "$.second_moments[\"weights\"]",
            ],
        );
        assert_eq!(
            state.paths().iter().map(|path| path.to_string()).collect::<Vec<_>>(),
            vec![
                "$",
                "$.count",
                "$.first_moments",
                "$.first_moments[\"bias\"]",
                "$.first_moments[\"weights\"]",
                "$.second_moments",
                "$.second_moments[\"bias\"]",
                "$.second_moments[\"weights\"]",
            ],
        );
        assert_eq!(state.parameter_count(), 5);
        let rebuilt = ScaleByAdamState::from_parameters(state.parameter_structure(), state.clone().into_parameters());
        assert_eq!(rebuilt, Ok(state.clone()));
        let types = state.clone().map_parameters(|leaf| leaf.r#type().into_owned()).unwrap();
        assert_eq!(types.count(), &ArrayType::scalar(DataType::I32));
        assert_eq!(types.first_moments()["weights"], ArrayType::new_static(DataType::F32, [2]));
    }

    #[test]
    fn test_scale_by_adam_new() {
        let adam = ScaleByAdam::new();
        assert_eq!(adam.beta1(), 0.9);
        assert_eq!(adam.beta2(), 0.999);
        assert_eq!(adam.epsilon(), 1e-8);
        assert_eq!(ScaleByAdam::default(), adam);
    }

    #[test]
    fn test_scale_by_adam_with_beta1() {
        assert_eq!(ScaleByAdam::new().with_beta1(0.0).unwrap().beta1(), 0.0);
        assert!(matches!(
            ScaleByAdam::new().with_beta1(1.0),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`beta1` must be in `[0, 1)` but got `1`",
        ));
        assert!(matches!(
            ScaleByAdam::new().with_beta1(f64::NAN),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`beta1` must be in `[0, 1)` but got `NaN`",
        ));
    }

    #[test]
    fn test_scale_by_adam_with_beta2() {
        assert_eq!(ScaleByAdam::new().with_beta2(0.95).unwrap().beta2(), 0.95);
        assert!(matches!(
            ScaleByAdam::new().with_beta2(-0.5),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`beta2` must be in `[0, 1)` but got `-0.5`",
        ));
    }

    #[test]
    fn test_scale_by_adam_with_epsilon() {
        assert_eq!(ScaleByAdam::new().with_epsilon(0.0).unwrap().epsilon(), 0.0);
        assert!(matches!(
            ScaleByAdam::new().with_epsilon(-1e-8),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`epsilon` must be finite and non-negative but got `-0.00000001`",
        ));
    }

    #[test]
    fn test_scale_by_adam_initialize() {
        // Moments are stored in `f32`, or in `f64` for `f64` leaves, and the step count is a rank-0 `i32` zero.
        let parameters = (
            Array::vector(vec![1.0f32, 2.0]).unwrap(),
            Array::scalar(bf16::from_f32(3.0)).unwrap(),
            Array::scalar(4.0f64).unwrap(),
        );
        let state = ScaleByAdam::new().initialize(&parameters).unwrap();
        let zeros =
            (Array::vector(vec![0f32, 0.0]).unwrap(), Array::scalar(0f32).unwrap(), Array::scalar(0f64).unwrap());
        assert_eq!(state.count(), &Array::scalar(0i32).unwrap());
        assert_eq!(state.first_moments(), &zeros);
        assert_eq!(state.second_moments(), &zeros);
    }

    #[test]
    fn test_scale_by_adam_initialize_without_parameters() {
        assert!(matches!(
            Optimizer::<Array>::initialize(&ScaleByAdam::new(), &Vec::<Array>::new()),
            Err(OptimizerError::StructureMismatch { message })
                if message == "`ScaleByAdam` requires at least one parameter to create its step count",
        ));
    }

    #[test]
    fn test_scale_by_adam_update() {
        // Three steps on an `f32` vector leaf and a `bf16` leaf, whose moments are accumulated in `f32` and whose
        // updates are returned in `bf16`.
        let adam = ScaleByAdam::new().with_beta1(0.8).unwrap().with_beta2(0.9).unwrap().with_epsilon(1e-3).unwrap();
        let parameters = (Array::vector(vec![1.0f32, 2.0]).unwrap(), Array::scalar(bf16::from_f32(3.0)).unwrap());
        let gradients = [[1.0f32, -2.0, 0.5], [0.5, 4.0, -0.25], [-1.5, 0.0, 2.0]];
        let mut state = adam.initialize(&parameters).unwrap();
        let mut updates = Vec::new();
        for [first, second, third] in gradients {
            let gradients =
                (Array::vector(vec![first, second]).unwrap(), Array::scalar(bf16::from_f32(third)).unwrap());
            let (step_updates, next_state) =
                adam.update(gradients, state, &parameters, &UpdateArguments::new()).unwrap();
            assert_eq!(step_updates.1.r#type().as_ref(), &ArrayType::scalar(DataType::BF16));
            updates.push((step_updates.0.elements::<f32>().unwrap(), step_updates.1.elements::<bf16>().unwrap()[0]));
            state = next_state;
        }
        let first = reference_adam_updates(&[1.0, 0.5, -1.5], 0.8, 0.9, 1e-3);
        let second = reference_adam_updates(&[-2.0, 4.0, 0.0], 0.8, 0.9, 1e-3);
        let third = reference_adam_updates(&[0.5, -0.25, 2.0], 0.8, 0.9, 1e-3);
        assert_eq!(
            updates,
            (0..3)
                .map(|step| (vec![first[step], second[step]], bf16::from_f32(third[step])))
                .collect::<Vec<_>>(),
        );
        assert_eq!(state.count(), &Array::scalar(3i32).unwrap());
        assert_eq!(state.second_moments().1.r#type().as_ref(), &ArrayType::scalar(DataType::F32));
    }

    #[test]
    fn test_scale_by_adam_update_structure_mismatch() {
        // A state for more leaves than the gradients is rejected.
        let adam = ScaleByAdam::new();
        let state = adam.initialize(&vec![Array::scalar(1f32).unwrap(), Array::scalar(2f32).unwrap()]).unwrap();
        let parameters = vec![Array::scalar(1f32).unwrap()];
        assert!(matches!(
            adam.update(parameters.clone(), state, &parameters, &UpdateArguments::new()),
            Err(OptimizerError::StructureMismatch { message })
                if message == "expected the leaf count of `first_moments` to be 1 but got 2",
        ));

        // So is a state whose structure has the same number of leaves at other paths.
        let state = adam.initialize(&BTreeMap::from([("a", Array::scalar(1f32).unwrap())])).unwrap();
        let parameters = BTreeMap::from([("b", Array::scalar(1f32).unwrap())]);
        assert!(matches!(
            adam.update(parameters.clone(), state, &parameters, &UpdateArguments::new()),
            Err(OptimizerError::StructureMismatch { message })
                if message == "expected leaf 0 of `first_moments` to have the path `$[\"b\"]` but got `$[\"a\"]`",
        ));

        // And so are parameters whose structure differs from that of the gradients.
        let gradients = BTreeMap::from([("a", Array::scalar(1f32).unwrap())]);
        let state = adam.initialize(&gradients).unwrap();
        assert!(matches!(
            adam.update(gradients, state, &parameters, &UpdateArguments::new()),
            Err(OptimizerError::StructureMismatch { message })
                if message == "expected leaf 0 of `parameters` to have the path `$[\"a\"]` but got `$[\"b\"]`",
        ));
    }

    #[test]
    fn test_scale_by_adam_with_epsilon_root() {
        assert_eq!(ScaleByAdam::new().epsilon_root(), 0.0);
        assert_eq!(ScaleByAdam::new().with_epsilon_root(1e-6).unwrap().epsilon_root(), 1e-6);
        assert!(matches!(
            ScaleByAdam::new().with_epsilon_root(-1.0),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`epsilon_root` must be finite and non-negative but got `-1`",
        ));

        // With `β₁ = β₂ = 0.5` and `ε = 0`, the first update of the gradient `2` has `m̂ = 2` and `v̂ = 4`, so it is
        // `2 / √(4 + 5) = 2 / 3`.
        let adam = ScaleByAdam::new()
            .with_beta1(0.5)
            .unwrap()
            .with_beta2(0.5)
            .unwrap()
            .with_epsilon(0.0)
            .unwrap()
            .with_epsilon_root(5.0)
            .unwrap();
        assert_eq!(vector_updates(&adam, &[0.0], &[&[2.0]]), vec![vec![2.0f32 / 3.0]]);
    }

    #[test]
    fn test_scale_by_adam_with_nesterov() {
        assert!(!ScaleByAdam::new().nesterov());
        assert!(ScaleByAdam::new().with_nesterov(true).nesterov());

        // With `β₁ = β₂ = 0.5` and `ε = 0`, the first update of the gradient `2` has `m = 1`, `v = 2`, and
        // `m̂ = 0.5 · 1 / (1 - 0.5²) + 0.5 · 2 / (1 - 0.5)`, and `v̂ = 2 / 0.5 = 4`. The second update of the gradient
        // `1` has `m = 1`, `v = 1.5`, `m̂ = 0.5 · 1 / (1 - 0.5³) + 0.5 · 1 / (1 - 0.5²)`, and `v̂ = 1.5 / 0.75`.
        let adam = ScaleByAdam::new()
            .with_beta1(0.5)
            .unwrap()
            .with_beta2(0.5)
            .unwrap()
            .with_epsilon(0.0)
            .unwrap()
            .with_nesterov(true);
        let first = (0.5f32 * (1.0 / 0.75) + 0.5 * (2.0 / 0.5)) / 4.0f32.sqrt();
        let second = (0.5f32 * (1.0 / 0.875) + 0.5 * (1.0 / 0.75)) / (1.5f32 / 0.75).sqrt();
        assert_eq!(vector_updates(&adam, &[0.0], &[&[2.0], &[1.0]]), vec![vec![first], vec![second]]);
    }

    #[test]
    fn test_scale_by_adam_with_first_moment_data_type() {
        assert_eq!(ScaleByAdam::new().first_moment_data_type(), None);
        let adam = ScaleByAdam::new().with_first_moment_data_type(DataType::BF16).unwrap();
        assert_eq!(adam.first_moment_data_type(), Some(DataType::BF16));
        assert!(matches!(
            ScaleByAdam::new().with_first_moment_data_type(DataType::I32),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`first_moment_data_type` must be a floating-point data type but got `i32`",
        ));

        // The first moments are stored in `bf16`, while the second moments keep the statistic data type.
        let parameters = vec![Array::vector(vec![1.0f32, 2.0]).unwrap()];
        let state = adam.initialize(&parameters).unwrap();
        assert_eq!(state.first_moments()[0].r#type().data_type(), DataType::BF16);
        let (_, state) = adam.update(parameters.clone(), state, &parameters, &UpdateArguments::new()).unwrap();
        assert_eq!(state.first_moments()[0], Array::vector(vec![bf16::from_f32(0.1), bf16::from_f32(0.2)]).unwrap());
        assert_eq!(state.second_moments()[0].r#type().data_type(), DataType::F32);
    }

    #[test]
    fn test_scale_by_amsgrad_new() {
        let amsgrad = ScaleByAmsgrad::new();
        assert_eq!(amsgrad.beta1(), 0.9);
        assert_eq!(amsgrad.beta2(), 0.999);
        assert_eq!(amsgrad.epsilon(), 1e-8);
        assert_eq!(amsgrad.epsilon_root(), 0.0);
        assert_eq!(amsgrad.first_moment_data_type(), None);
        assert!(amsgrad.first_moment_bias_correction());
        assert!(amsgrad.second_moment_bias_correction());
        assert_eq!(ScaleByAmsgrad::default(), amsgrad);
        let amsgrad = amsgrad
            .with_beta1(0.5)
            .unwrap()
            .with_beta2(0.6)
            .unwrap()
            .with_epsilon(1e-3)
            .unwrap()
            .with_epsilon_root(1e-4)
            .unwrap()
            .with_first_moment_data_type(DataType::F64)
            .unwrap()
            .with_first_moment_bias_correction(false)
            .with_second_moment_bias_correction(false);
        assert_eq!(
            (amsgrad.beta1(), amsgrad.beta2(), amsgrad.epsilon(), amsgrad.epsilon_root()),
            (0.5, 0.6, 1e-3, 1e-4),
        );
        assert_eq!(amsgrad.first_moment_data_type(), Some(DataType::F64));
        assert!(!amsgrad.first_moment_bias_correction());
        assert!(!amsgrad.second_moment_bias_correction());
        assert!(matches!(
            ScaleByAmsgrad::new().with_beta2(1.0),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`beta2` must be in `[0, 1)` but got `1`",
        ));
    }

    #[test]
    fn test_scale_by_amsgrad_initialize() {
        assert_eq!(
            state_paths(&ScaleByAmsgrad::new()),
            vec![
                "$.count",
                "$.first_moments[\"bias\"]",
                "$.first_moments[\"weights\"]",
                "$.second_moments[\"bias\"]",
                "$.second_moments[\"weights\"]",
                "$.maximum_second_moments[\"bias\"]",
                "$.maximum_second_moments[\"weights\"]",
            ],
        );
    }

    #[test]
    fn test_scale_by_amsgrad_update() {
        // With `β₁ = β₂ = 0.5` and `ε = 0`, the gradient `2` gives `m̂ = 1 / 0.5 = 2` and `v̂ = 2 / 0.5 = 4`, so the
        // first update is `2 / √4 = 1`. The gradient `1` then gives `m̂ = 1 / 0.75` and `v̂ = 1.5 / 0.75 = 2`, but the
        // maximum `4` is kept, so the second update is `(1 / 0.75) / √4`.
        let amsgrad =
            ScaleByAmsgrad::new().with_beta1(0.5).unwrap().with_beta2(0.5).unwrap().with_epsilon(0.0).unwrap();
        assert_eq!(vector_updates(&amsgrad, &[0.0], &[&[2.0], &[1.0]]), vec![vec![1.0], vec![1.0f32 / 0.75 / 2.0]]);

        // Without bias corrections, the gradient `2` gives `m = 1` and `v = 2`, so the first update is `1 / √2`.
        let amsgrad = amsgrad.with_first_moment_bias_correction(false).with_second_moment_bias_correction(false);
        assert_eq!(vector_updates(&amsgrad, &[0.0], &[&[2.0]]), vec![vec![1.0f32 / 2.0f32.sqrt()]]);
    }

    #[test]
    fn test_scale_by_adamax_new() {
        let adamax = ScaleByAdamax::new();
        assert_eq!((adamax.beta1(), adamax.beta2(), adamax.epsilon()), (0.9, 0.999, 1e-8));
        assert_eq!(ScaleByAdamax::default(), adamax);
        let adamax = adamax.with_beta1(0.5).unwrap().with_beta2(0.6).unwrap().with_epsilon(0.0).unwrap();
        assert_eq!((adamax.beta1(), adamax.beta2(), adamax.epsilon()), (0.5, 0.6, 0.0));
        assert!(matches!(
            ScaleByAdamax::new().with_epsilon(f64::NAN),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`epsilon` must be finite and non-negative but got `NaN`",
        ));
    }

    #[test]
    fn test_scale_by_adamax_update() {
        // With `β₁ = β₂ = 0.5` and `ε = 0`, the gradient `2` gives `m = 1` and `u = max(2, 0) = 2`, so the first update
        // is `(1 / 0.5) / 2 = 1`. The gradient `-1` gives `m = 0` and `u = max(1, 0.5 · 2) = 1`, so the second update
        // is `(0 / 0.75) / 1 = 0`. For the second element, the gradients `4` and `1` give `m = 2` and `u = 4` and then
        // `m = 1.5` and `u = max(1, 0.5 · 4) = 2`.
        let adamax = ScaleByAdamax::new().with_beta1(0.5).unwrap().with_beta2(0.5).unwrap().with_epsilon(0.0).unwrap();
        assert_eq!(
            vector_updates(&adamax, &[0.0, 0.0], &[&[2.0, 4.0], &[-1.0, 1.0]]),
            vec![vec![1.0, 1.0], vec![0.0, 1.5f32 / 0.75 / 2.0],],
        );
    }

    #[test]
    fn test_scale_by_radam_new() {
        let radam = ScaleByRadam::new();
        assert_eq!(
            (radam.beta1(), radam.beta2(), radam.epsilon(), radam.epsilon_root(), radam.threshold()),
            (0.9, 0.999, 1e-8, 0.0, 5.0),
        );
        assert!(!radam.nesterov());
        assert_eq!(ScaleByRadam::default(), radam);
        let radam = radam
            .with_beta1(0.5)
            .unwrap()
            .with_beta2(0.6)
            .unwrap()
            .with_epsilon(0.0)
            .unwrap()
            .with_epsilon_root(1e-3)
            .unwrap()
            .with_threshold(1.5)
            .unwrap()
            .with_nesterov(true);
        assert_eq!((radam.beta1(), radam.beta2(), radam.epsilon_root(), radam.threshold()), (0.5, 0.6, 1e-3, 1.5));
        assert!(radam.nesterov());
        assert!(matches!(
            ScaleByRadam::new().with_threshold(f64::INFINITY),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`threshold` must be finite but got `inf`",
        ));
    }

    #[test]
    fn test_scale_by_radam_update() {
        // With `β₁ = 0.5`, `β₂ = 0.75`, and `ε = 0`, `ρ∞ = 2 / 0.25 - 1 = 7`. At the first step,
        // `ρ = 7 - 2 · 0.75 / 0.25 = 1` is below the threshold `1.5`, so the update is `m̂ = 1 / 0.5 = 2` for the
        // gradient `2`. At the second step, `ρ = 7 - 4 · 0.5625 / 0.4375 ≈ 1.857`, so the update of the gradient `1` is
        // rectified by `r = √((ρ - 4) · (ρ - 2) · 7 / (3 · 5 · ρ))` and is `r · m̂ / √v̂` with `m̂ = 1 / 0.75` and
        // `v̂ = 1.75 / 0.4375`.
        let radam = ScaleByRadam::new()
            .with_beta1(0.5)
            .unwrap()
            .with_beta2(0.75)
            .unwrap()
            .with_epsilon(0.0)
            .unwrap()
            .with_threshold(1.5)
            .unwrap();
        let decay_power = 0.75f32.powf(2.0);
        let length = 7.0f32 - 2.0 * 2.0 * decay_power / (1.0 - decay_power);
        let rectification = ((length - 4.0) * (length - 2.0) * 7.0 / (15.0 * length)).sqrt();
        let second_moment = 0.25f32 * 1.0 + 0.75 * (0.25 * 4.0);
        let second = rectification * (1.0f32 / 0.75) / (second_moment / (1.0 - decay_power)).sqrt();
        assert_eq!(vector_updates(&radam, &[0.0], &[&[2.0], &[1.0]]), vec![vec![2.0], vec![second]]);
    }

    #[test]
    fn test_scale_by_yogi_new() {
        let yogi = ScaleByYogi::new();
        assert_eq!(
            (yogi.beta1(), yogi.beta2(), yogi.epsilon(), yogi.epsilon_root(), yogi.initial_accumulator_value()),
            (0.9, 0.999, 1e-3, 0.0, 1e-6),
        );
        assert_eq!(ScaleByYogi::default(), yogi);
        let yogi = yogi
            .with_beta1(0.5)
            .unwrap()
            .with_beta2(0.6)
            .unwrap()
            .with_epsilon(0.0)
            .unwrap()
            .with_epsilon_root(1e-3)
            .unwrap()
            .with_initial_accumulator_value(1.0)
            .unwrap();
        assert_eq!((yogi.beta2(), yogi.epsilon_root(), yogi.initial_accumulator_value()), (0.6, 1e-3, 1.0));
        assert!(matches!(
            ScaleByYogi::new().with_initial_accumulator_value(-1.0),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`initial_accumulator_value` must be finite and non-negative but got `-1`",
        ));
    }

    #[test]
    fn test_scale_by_yogi_update() {
        // With `β₁ = β₂ = 0.5`, `ε = 0`, and `v = 1` initially, the gradient `2` gives `m = 1` and
        // `v = 1 - 0.5 · sign(1 - 4) · 4 = 3`, so the first update is `(1 / 0.5) / √(3 / 0.5)`. The gradient `1` gives
        // `m = 1` and `v = 3 - 0.5 · sign(3 - 1) · 1 = 2.5`, so the second update is `(1 / 0.75) / √(2.5 / 0.75)`.
        let yogi = ScaleByYogi::new()
            .with_beta1(0.5)
            .unwrap()
            .with_beta2(0.5)
            .unwrap()
            .with_epsilon(0.0)
            .unwrap()
            .with_initial_accumulator_value(1.0)
            .unwrap();
        assert_eq!(
            vector_updates(&yogi, &[0.0], &[&[2.0], &[1.0]]),
            vec![vec![2.0f32 / (3.0f32 / 0.5).sqrt()], vec![(1.0f32 / 0.75) / (2.5f32 / 0.75).sqrt()],],
        );
    }

    #[test]
    fn test_scale_by_belief_new() {
        let belief = ScaleByBelief::new();
        assert_eq!(
            (belief.beta1(), belief.beta2(), belief.epsilon(), belief.epsilon_root()),
            (0.9, 0.999, 1e-16, 1e-16),
        );
        assert!(!belief.nesterov());
        assert_eq!(ScaleByBelief::default(), belief);
        let belief = belief
            .with_beta1(0.5)
            .unwrap()
            .with_beta2(0.6)
            .unwrap()
            .with_epsilon(0.0)
            .unwrap()
            .with_epsilon_root(0.0)
            .unwrap()
            .with_nesterov(true);
        assert_eq!((belief.beta1(), belief.beta2(), belief.epsilon(), belief.epsilon_root()), (0.5, 0.6, 0.0, 0.0));
        assert!(belief.nesterov());
    }

    #[test]
    fn test_scale_by_belief_initialize() {
        assert_eq!(
            state_paths(&ScaleByBelief::new()),
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
    fn test_scale_by_belief_update() {
        // With `β₁ = β₂ = 0.5` and `ε = ε̄ = 0`, the gradient `2` gives `m = 1` and `s = 0.5 · (2 - 1)² = 0.5`, so the
        // first update is `(1 / 0.5) / √(0.5 / 0.5) = 2`. The gradient `1` gives `m = 1` and `s = 0.5 · 0.5 = 0.25`, so
        // the second update is `(1 / 0.75) / √(0.25 / 0.75)`.
        let belief = ScaleByBelief::new()
            .with_beta1(0.5)
            .unwrap()
            .with_beta2(0.5)
            .unwrap()
            .with_epsilon(0.0)
            .unwrap()
            .with_epsilon_root(0.0)
            .unwrap();
        assert_eq!(
            vector_updates(&belief, &[0.0], &[&[2.0], &[1.0]]),
            vec![vec![2.0], vec![(1.0f32 / 0.75) / (0.25f32 / 0.75).sqrt()],],
        );
    }

    #[test]
    fn test_scale_by_adan_new() {
        let adan = ScaleByAdan::new();
        assert_eq!(
            (adan.beta1(), adan.beta2(), adan.beta3(), adan.epsilon(), adan.epsilon_root()),
            (0.98, 0.92, 0.99, 1e-8, 0.0),
        );
        assert_eq!(ScaleByAdan::default(), adan);
        let adan = adan
            .with_beta1(0.5)
            .unwrap()
            .with_beta2(0.6)
            .unwrap()
            .with_beta3(0.7)
            .unwrap()
            .with_epsilon(0.0)
            .unwrap()
            .with_epsilon_root(1e-3)
            .unwrap();
        assert_eq!((adan.beta1(), adan.beta2(), adan.beta3(), adan.epsilon_root()), (0.5, 0.6, 0.7, 1e-3));
        assert!(matches!(
            ScaleByAdan::new().with_beta3(1.5),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`beta3` must be in `[0, 1)` but got `1.5`",
        ));
    }

    #[test]
    fn test_scale_by_adan_initialize() {
        assert_eq!(
            state_paths(&ScaleByAdan::new()),
            vec![
                "$.count",
                "$.first_moments[\"bias\"]",
                "$.first_moments[\"weights\"]",
                "$.difference_moments[\"bias\"]",
                "$.difference_moments[\"weights\"]",
                "$.second_moments[\"bias\"]",
                "$.second_moments[\"weights\"]",
                "$.previous_gradients[\"bias\"]",
                "$.previous_gradients[\"weights\"]",
            ],
        );
    }

    #[test]
    fn test_scale_by_adan_update() {
        // With `β₁ = β₂ = β₃ = 0.5` and `ε = 0`, the first gradient `2` has no difference, so `m = 1`, `v = 0`, and
        // `n = 0.5 · 2² = 2`, and the first update is `(1 / 0.5) / √(2 / 0.5) = 1`. The gradient `1` then has the
        // difference `-1`, so `m = 1`, `v = -0.5`, and `n = 0.5 · (1 - 0.5)² + 0.5 · 2 = 1.125`, and the second update
        // is `(1 / 0.75 + 0.5 · (-0.5 / 0.75)) / √(1.125 / 0.75)`.
        let adan = ScaleByAdan::new()
            .with_beta1(0.5)
            .unwrap()
            .with_beta2(0.5)
            .unwrap()
            .with_beta3(0.5)
            .unwrap()
            .with_epsilon(0.0)
            .unwrap();
        assert_eq!(
            vector_updates(&adan, &[0.0], &[&[2.0], &[1.0]]),
            vec![vec![1.0], vec![(1.0f32 / 0.75 + 0.5 * (-0.5f32 / 0.75)) / (1.125f32 / 0.75).sqrt()],],
        );
    }

    /// Returns the updates of `optimizer` for the parity gradients repeated twice (eight steps), with the loop of
    /// [`parity_updates`]. RAdam needs more than four steps before its length estimate reaches its threshold.
    fn repeated_parity_updates<O: Optimizer<Array>>(optimizer: &O) -> Vec<Vec<f32>> {
        let mut parameters = parity_parameters(&PARITY_PARAMETERS);
        let mut state = optimizer.initialize(&parameters).unwrap();
        let mut updates = Vec::new();
        for gradients in PARITY_GRADIENTS.iter().chain(&PARITY_GRADIENTS) {
            let (step_updates, next_state) =
                optimizer.update(parity_parameters(gradients), state, &parameters, &UpdateArguments::new()).unwrap();
            updates.push(parity_elements(&step_updates));
            parameters = apply_updates(parameters, step_updates).unwrap();
            state = next_state;
        }
        updates
    }

    #[test]
    fn test_scale_by_adam_optax_parity() {
        // Optax 9ecde04b: `optax.scale_by_adam(b1=0.8, b2=0.95, eps=1e-3, eps_root=1e-4)`.
        const SCALE_BY_ADAM: [[f32; 9]; 4] = [
            [0.985234, -0.99378985, 0.99612665, 0.9971957, -0.985234, 0.99378985, -0.99612665, 0.9978052, -0.99819803],
            [
                0.20812199,
                0.21906674,
                0.84883475,
                0.9172677,
                0.53816366,
                -0.41691712,
                -0.3495202,
                0.29365784,
                -0.14764397,
            ],
            [0.6995922, -0.059288487, 0.13705069, 0.404963, 0.4987779, 0.28486985, 0.2427534, 0.59381217, -0.20960678],
            [0.8061629, 0.23602392, -0.3823491, 0.44883022, -0.18121804, 0.27271813, -0.3741584, 0.6335372, 0.12355825],
        ];
        let optimizer = ScaleByAdam::new()
            .with_beta1(0.8)
            .unwrap()
            .with_beta2(0.95)
            .unwrap()
            .with_epsilon(1e-3)
            .unwrap()
            .with_epsilon_root(1e-4)
            .unwrap();
        assert_parity(&parity_updates(&optimizer), &SCALE_BY_ADAM);

        // Optax 9ecde04b: `optax.scale_by_adam(b1=0.8, b2=0.95, eps=1e-3, eps_root=1e-4, nesterov=True)`.
        const SCALE_BY_ADAM_NESTEROV: [[f32; 9]; 4] = [
            [1.4231157, -1.4354742, 1.4388496, 1.4403938, -1.4231157, 1.4354742, -1.4388496, 1.4412742, -1.4418416],
            [
                -0.22404389,
                0.73780406,
                0.7506089,
                0.8941335,
                1.0514643,
                -0.9409113,
                0.043383054,
                -0.120351575,
                0.3103687,
            ],
            [1.1103735, -0.25094885, -0.290072, 0.0394178, 0.44016427, 0.68523526, 0.6002906, 0.8168078, -0.24135017],
            [0.9174639, 0.44326973, -0.7262771, 0.45880127, -0.5833955, 0.24477282, -0.7586477, 0.6406103, 0.35776362],
        ];
        let optimizer = ScaleByAdam::new()
            .with_beta1(0.8)
            .unwrap()
            .with_beta2(0.95)
            .unwrap()
            .with_epsilon(1e-3)
            .unwrap()
            .with_epsilon_root(1e-4)
            .unwrap()
            .with_nesterov(true);
        assert_parity(&parity_updates(&optimizer), &SCALE_BY_ADAM_NESTEROV);
    }

    #[test]
    fn test_scale_by_amsgrad_optax_parity() {
        // Optax 9ecde04b: `optax.scale_by_amsgrad(b1=0.8, b2=0.95, eps=1e-3, eps_root=1e-4)`.
        const SCALE_BY_AMSGRAD: [[f32; 9]; 4] = [
            [0.985234, -0.99378985, 0.99612665, 0.9971957, -0.985234, 0.99378985, -0.99612665, 0.9978052, -0.99819803],
            [0.16420567, 0.21906674, 0.6271909, 0.7201969, 0.53816366, -0.41691712, -0.2582551, 0.22173451, -0.1201535],
            [
                0.6995922,
                -0.050277628,
                0.09797967,
                0.27177668,
                0.4078334,
                0.28486985,
                0.18779434,
                0.45800892,
                -0.13909319,
            ],
            [0.8061629, 0.1893829, -0.3288927, 0.26416224, -0.17779686, 0.23400132, -0.3741584, 0.4380608, 0.077096574],
        ];
        let optimizer = ScaleByAmsgrad::new()
            .with_beta1(0.8)
            .unwrap()
            .with_beta2(0.95)
            .unwrap()
            .with_epsilon(1e-3)
            .unwrap()
            .with_epsilon_root(1e-4)
            .unwrap();
        assert_parity(&parity_updates(&optimizer), &SCALE_BY_AMSGRAD);

        // Optax 9ecde04b: `optax.scale_by_amsgrad(b1=0.8, b2=0.95, eps=1e-3, eps_root=1e-4, bias_correction_mu=False,
        // bias_correction_nu=False)`.
        const SCALE_BY_AMSGRAD_UNCORRECTED: [[f32; 9]; 4] = [
            [0.78447074, -0.8542308, 0.8717978, 0.8791209, -0.78447074, 0.8542308, -0.8717978, 0.88300437, -0.8853721],
            [
                0.21851942,
                0.24797107,
                0.9601645,
                1.0453461,
                0.6090889,
                -0.47524756,
                -0.39536187,
                0.33552572,
                -0.16913296,
            ],
            [0.8888558, -0.07536419, 0.1746234, 0.51796126, 0.6257005, 0.36567962, 0.30977136, 0.7625224, -0.26540864],
            [1.0911834, 0.31914994, -0.5197769, 0.6090913, -0.24586974, 0.36341187, -0.5095288, 0.8635908, 0.16851917],
        ];
        let optimizer = ScaleByAmsgrad::new()
            .with_beta1(0.8)
            .unwrap()
            .with_beta2(0.95)
            .unwrap()
            .with_epsilon(1e-3)
            .unwrap()
            .with_epsilon_root(1e-4)
            .unwrap()
            .with_first_moment_bias_correction(false)
            .with_second_moment_bias_correction(false);
        assert_parity(&parity_updates(&optimizer), &SCALE_BY_AMSGRAD_UNCORRECTED);
    }

    #[test]
    fn test_scale_by_adamax_optax_parity() {
        // Optax 9ecde04b: `optax.scale_by_adamax(b1=0.8, b2=0.95, eps=1e-3)`.
        const SCALE_BY_ADAMAX: [[f32; 9]; 4] = [
            [
                0.99009913,
                -0.99502504,
                0.9966779,
                0.99750644,
                -0.99009913,
                0.99502504,
                -0.9966779,
                0.99800414,
                -0.99833626,
            ],
            [
                0.17370161,
                0.19920321,
                0.6605663,
                0.7583382,
                0.40605396,
                -0.33250213,
                -0.27199787,
                0.23345129,
                -0.12649488,
            ],
            [
                0.44115248,
                -0.048125036,
                0.108624816,
                0.3012314,
                0.32391313,
                0.23445167,
                0.20819752,
                0.50759035,
                -0.1541412,
            ],
            [
                0.54399574,
                0.1908154,
                -0.28219798,
                0.30820185,
                -0.13415085,
                0.20272231,
                -0.25507906,
                0.5110344,
                0.089934096,
            ],
        ];
        let optimizer =
            ScaleByAdamax::new().with_beta1(0.8).unwrap().with_beta2(0.95).unwrap().with_epsilon(1e-3).unwrap();
        assert_parity(&parity_updates(&optimizer), &SCALE_BY_ADAMAX);
    }

    #[test]
    fn test_scale_by_radam_optax_parity() {
        // Optax 9ecde04b: `optax.scale_by_radam(b1=0.8, b2=0.95, eps=1e-3, eps_root=1e-4, threshold=5.0)`.
        const SCALE_BY_RADAM: [[f32; 9]; 8] = [
            [
                0.10000002,
                -0.20000003,
                0.30000004,
                0.40000007,
                -0.10000002,
                0.20000003,
                -0.30000004,
                0.50000006,
                -0.6000001,
            ],
            [
                0.01666667,
                0.050000004,
                0.18888892,
                0.28888893,
                0.12222224,
                -0.13333336,
                -0.077777795,
                0.111111134,
                -0.072222255,
            ],
            [
                0.1327869,
                -0.011475414,
                0.029508201,
                0.109016396,
                0.09262296,
                0.10573769,
                0.056557376,
                0.22950822,
                -0.08360658,
            ],
            [
                0.15555558,
                0.043224934,
                -0.09905149,
                0.10596206,
                -0.040379405,
                0.086856365,
                -0.11504065,
                0.21951221,
                0.046341456,
            ],
            [
                0.13902906,
                -0.02912899,
                0.019657312,
                0.1934317,
                -0.05811518,
                0.120514035,
                -0.17006187,
                0.30295098,
                -0.14593053,
            ],
            [
                0.0994735,
                0.042235684,
                0.030687887,
                0.13292824,
                0.031658154,
                -0.011763777,
                -0.063263066,
                0.08475388,
                -0.0049881735,
            ],
            [
                0.17226613,
                0.011706235,
                -0.019151349,
                0.09965176,
                0.04721911,
                0.06668053,
                -0.007702125,
                0.14376216,
                -0.020447122,
            ],
            [
                0.22274235,
                0.06521326,
                -0.10564284,
                0.12401156,
                -0.050070435,
                0.07535187,
                -0.10337974,
                0.17504594,
                0.034139074,
            ],
        ];
        let optimizer = ScaleByRadam::new()
            .with_beta1(0.8)
            .unwrap()
            .with_beta2(0.95)
            .unwrap()
            .with_epsilon(1e-3)
            .unwrap()
            .with_epsilon_root(1e-4)
            .unwrap()
            .with_threshold(5.0)
            .unwrap();
        assert_parity(&repeated_parity_updates(&optimizer), &SCALE_BY_RADAM);

        // Optax 9ecde04b: `optax.scale_by_radam(b1=0.8, b2=0.95, eps=1e-3, eps_root=1e-4, threshold=5.0,
        // nesterov=True)`.
        const SCALE_BY_RADAM_NESTEROV: [[f32; 9]; 8] = [
            [
                0.14444447,
                -0.28888893,
                0.4333334,
                0.57777786,
                -0.14444447,
                0.28888893,
                -0.4333334,
                0.7222223,
                -0.8666668,
            ],
            [
                -0.017941715,
                0.16839711,
                0.16703099,
                0.28160295,
                0.23879784,
                -0.3009108,
                0.009653915,
                -0.045537353,
                0.1518215,
            ],
            [
                0.2107557,
                -0.048571687,
                -0.06245502,
                0.010611307,
                0.08173842,
                0.2543449,
                0.1398574,
                0.31569597,
                -0.09626818,
            ],
            [
                0.177032,
                0.08117951,
                -0.1881496,
                0.10831608,
                -0.12999347,
                0.07795623,
                -0.23325768,
                0.22196296,
                0.13418195,
            ],
            [
                0.13109219,
                -0.080728896,
                0.10357232,
                0.25999194,
                -0.072110474,
                0.14734352,
                -0.21320888,
                0.36957315,
                -0.28486168,
            ],
            [
                0.058943268,
                0.093056194,
                0.042996883,
                0.1362023,
                0.08973298,
                -0.07077958,
                -0.029549189,
                0.035730142,
                0.037403557,
            ],
            [
                0.2232996,
                -0.022464786,
                -0.06383896,
                0.040627565,
                0.050182924,
                0.12773794,
                0.047821887,
                0.17369692,
                -0.030797072,
            ],
            [
                0.24011101,
                0.10453388,
                -0.17094757,
                0.12348973,
                -0.1279025,
                0.0683696,
                -0.17668018,
                0.17293702,
                0.07936459,
            ],
        ];
        let optimizer = ScaleByRadam::new()
            .with_beta1(0.8)
            .unwrap()
            .with_beta2(0.95)
            .unwrap()
            .with_epsilon(1e-3)
            .unwrap()
            .with_epsilon_root(1e-4)
            .unwrap()
            .with_threshold(5.0)
            .unwrap()
            .with_nesterov(true);
        assert_parity(&repeated_parity_updates(&optimizer), &SCALE_BY_RADAM_NESTEROV);
    }

    #[test]
    fn test_scale_by_yogi_optax_parity() {
        // Optax 9ecde04b: `optax.scale_by_yogi(b1=0.8, b2=0.95, eps=1e-3, eps_root=1e-4,
        // initial_accumulator_value=1e-2)`.
        const SCALE_BY_YOGI: [[f32; 9]; 4] = [
            [0.2176911, -0.4073321, 0.5559579, 0.66546535, -0.2176911, 0.4073321, -0.5559579, 0.74416417, -0.8006425],
            [
                0.05091251,
                0.12658548,
                0.49699208,
                0.6362896,
                0.3107137,
                -0.29367214,
                -0.20464382,
                0.22116947,
                -0.121854536,
            ],
            [
                0.40970802,
                -0.035706986,
                0.087799706,
                0.2824348,
                0.28578398,
                0.22951849,
                0.1626822,
                0.47967878,
                -0.17179945,
            ],
            [
                0.5137515,
                0.14774615,
                -0.28585574,
                0.3166817,
                -0.12449585,
                0.21537398,
                -0.2992635,
                0.50768167,
                0.10196876,
            ],
        ];
        let optimizer = ScaleByYogi::new()
            .with_beta1(0.8)
            .unwrap()
            .with_beta2(0.95)
            .unwrap()
            .with_epsilon(1e-3)
            .unwrap()
            .with_epsilon_root(1e-4)
            .unwrap()
            .with_initial_accumulator_value(1e-2)
            .unwrap();
        assert_parity(&parity_updates(&optimizer), &SCALE_BY_YOGI);
    }

    #[test]
    fn test_scale_by_belief_optax_parity() {
        // Optax 9ecde04b: `optax.scale_by_belief(b1=0.8, b2=0.95, eps=1e-3, eps_root=1e-4)`.
        const SCALE_BY_BELIEF: [[f32; 9]; 4] = [
            [1.0793135, -1.1966559, 1.2238351, 1.2341498, -1.0793135, 1.1966559, -1.2238351, 1.2391815, -1.2420336],
            [0.2007715, 0.24257827, 1.073901, 1.2090664, 0.61794066, -0.47566938, -0.3945681, 0.33478463, -0.1667404],
            [0.843601, -0.06465335, 0.15327366, 0.47624275, 0.5724257, 0.3222524, 0.271371, 0.7203175, -0.23751134],
            [1.052211, 0.25924006, -0.43995258, 0.5348233, -0.20093, 0.3085031, -0.42918217, 0.79564875, 0.13894428],
        ];
        let optimizer = ScaleByBelief::new()
            .with_beta1(0.8)
            .unwrap()
            .with_beta2(0.95)
            .unwrap()
            .with_epsilon(1e-3)
            .unwrap()
            .with_epsilon_root(1e-4)
            .unwrap();
        assert_parity(&parity_updates(&optimizer), &SCALE_BY_BELIEF);

        // Optax 9ecde04b: `optax.scale_by_belief(b1=0.8, b2=0.95, eps=1e-3, eps_root=1e-4, nesterov=True)`.
        const SCALE_BY_BELIEF_NESTEROV: [[f32; 9]; 4] = [
            [1.5590084, -1.7285029, 1.7677618, 1.7826607, -1.5590084, 1.7285029, -1.7677618, 1.7899289, -1.7940485],
            [
                -0.21613106,
                0.81698954,
                0.9496309,
                1.1785729,
                1.2073326,
                -1.0735053,
                0.048974477,
                -0.13720682,
                0.35051215,
            ],
            [1.3389401, -0.27365658, -0.3244084, 0.046355944, 0.5051574, 0.77515644, 0.67105734, 0.99082, -0.27348065],
            [1.1974821, 0.4868713, -0.8356957, 0.54670477, -0.6468542, 0.2768909, -0.87021446, 0.80453175, 0.40231392],
        ];
        let optimizer = ScaleByBelief::new()
            .with_beta1(0.8)
            .unwrap()
            .with_beta2(0.95)
            .unwrap()
            .with_epsilon(1e-3)
            .unwrap()
            .with_epsilon_root(1e-4)
            .unwrap()
            .with_nesterov(true);
        assert_parity(&parity_updates(&optimizer), &SCALE_BY_BELIEF_NESTEROV);
    }

    #[test]
    fn test_scale_by_adan_optax_parity() {
        // Optax 9ecde04b: `optax.scale_by_adan(b1=0.9, b2=0.8, b3=0.95, eps=1e-3, eps_root=1e-4)`.
        const SCALE_BY_ADAN: [[f32; 9]; 4] = [
            [
                0.9852337,
                -0.99378955,
                0.99612635,
                0.9971954,
                -0.9852337,
                0.99378955,
                -0.99612635,
                0.9978048,
                -0.99819773,
            ],
            [
                0.04775094,
                0.30812922,
                0.80242556,
                0.8995212,
                0.54932654,
                -0.45749998,
                -0.18223624,
                0.12610163,
                0.009727746,
            ],
            [0.6155227, -0.051589098, 0.048968703, 0.320581, 0.4126659, 0.252486, 0.28632936, 0.5063257, -0.15734862],
            [
                0.69136804,
                0.21702813,
                -0.36411574,
                0.44491658,
                -0.16289216,
                0.17574365,
                -0.3364249,
                0.5210704,
                0.13354705,
            ],
        ];
        let optimizer = ScaleByAdan::new()
            .with_beta1(0.9)
            .unwrap()
            .with_beta2(0.8)
            .unwrap()
            .with_beta3(0.95)
            .unwrap()
            .with_epsilon(1e-3)
            .unwrap()
            .with_epsilon_root(1e-4)
            .unwrap();
        assert_parity(&parity_updates(&optimizer), &SCALE_BY_ADAN);
    }
}
