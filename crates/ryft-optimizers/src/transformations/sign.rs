//! Transformations that keep only the signs of the gradients or of their moments, which are the cores of sign SGD,
//! Signum, Lion, and Rprop.
//!
//! [`ScaleBySign`] replaces every update with its sign. [`ScaleByLion`] takes the sign of an interpolation between the
//! gradient and its momentum, and [`ScaleByRprop`] adapts a step size per element from the signs of consecutive
//! gradients.

use std::marker::PhantomData;

use ryft_core::{Add, ComparisonDirection, DataType, Mul, Parameter, Parameterized};

use crate::macros::parameterized_state;
use crate::{
    Optimizer, OptimizerError, OptimizerValue, UpdateArguments, convert_to, filled_statistics, increment_count,
    initial_count, statistic_data_type, update_moment, validate_decay_rate, validate_finite, validate_floating_point,
    validate_non_negative, validate_positive, validate_structure,
};

// TODO(eaplatanios): Review this module.

/// Transformation that replaces every element of the updates with its sign (i.e., `-1`, `0`, or `1`), which is the
/// analogue of
/// [`optax.scale_by_sign`](https://optax.readthedocs.io/en/latest/api/transformations.html#optax.scale_by_sign) and
/// the rescaling of sign SGD of [Bernstein et al. (2018)](https://arxiv.org/abs/1802.04434). This transformation is
/// stateless.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ScaleBySign;

impl ScaleBySign {
    /// Creates a new [`ScaleBySign`].
    #[inline]
    pub fn new() -> Self {
        Self
    }
}

impl<P: OptimizerValue> Optimizer<P> for ScaleBySign {
    type State<S: Parameterized<P>> = ();

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        let _ = parameters;
        Ok(())
    }

    fn update<S: Parameterized<P>>(
        &self,
        gradients: S,
        state: Self::State<S>,
        parameters: &S,
        arguments: &UpdateArguments<'_, P, S>,
    ) -> Result<(S, Self::State<S>), OptimizerError> {
        let _ = arguments;
        validate_structure("parameters", &gradients.parameter_paths().collect::<Vec<_>>(), parameters)?;
        let structure = gradients.parameter_structure();
        let updates = gradients
            .into_parameters()
            .map(|gradient| Ok(gradient.sign()?))
            .collect::<Result<Vec<_>, OptimizerError>>()?;
        Ok((S::from_parameters(structure, updates)?, state))
    }
}

/// Function of [`ScaleByLion`] that maps the interpolation between the gradient and its momentum to the update.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum LionMode {
    /// The sign of the interpolation, which is the update of the original Lion optimizer.
    Hard,

    /// `tanh(β · x)` of the interpolation `x` for a sharpness `β`, which is a smooth approximation of the sign.
    Smooth {
        /// Sharpness `β` of the approximation, which is the `smooth_beta` of Optax.
        beta: f64,
    },

    /// The interpolation itself where its magnitude is less than `1` and its sign elsewhere.
    Refined,
}

parameterized_state! {
    /// State of [`ScaleByLion`], which holds the number of updates applied so far and the momenta of the gradients,
    /// with the structure `S` of the parameters.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct ScaleByLionState<P, S> with family ScaleByLionStateFamily {
        /// Rank-0 `i32` array that holds the number of updates applied so far.
        count: P,

        /// Exponential moving averages of the gradients, with the structure of the parameters.
        first_moments: S,
    }
}

impl<P: Parameter, S> ScaleByLionState<P, S> {
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
}

/// Transformation that takes the sign of an interpolation between the gradient and its momentum, which is the
/// rescaling of Lion of [Chen et al. (2023)](https://arxiv.org/abs/2302.06675) and the analogue of
/// [`optax.scale_by_lion`](https://optax.readthedocs.io/en/latest/api/transformations.html#optax.scale_by_lion). For a
/// gradient `g` and the momentum `m`, it returns `sign((1 - β₁) · g + β₁ · m)` (or the function of the [`LionMode`]
/// instead of the sign) and then updates the momentum as `m = (1 - β₂) · g + β₂ · m`. The defaults are `β₁ = 0.9` and
/// `β₂ = 0.99` with [`LionMode::Hard`].
#[derive(Clone, Debug, PartialEq)]
pub struct ScaleByLion {
    /// Interpolation rate `β₁` between the gradient and its momentum.
    beta1: f64,

    /// Decay rate `β₂` of the momentum.
    beta2: f64,

    /// Function that maps the interpolation to the update.
    mode: LionMode,

    /// Data type in which the momenta are stored, or [`None`] for the statistic data type of each leaf.
    first_moment_data_type: Option<DataType>,
}

impl ScaleByLion {
    /// Creates a new [`ScaleByLion`] with the default hyperparameters.
    #[inline]
    pub fn new() -> Self {
        Self { beta1: 0.9, beta2: 0.99, mode: LionMode::Hard, first_moment_data_type: None }
    }

    /// Returns this transformation with `beta1` as the interpolation rate between the gradient and its momentum.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `beta1` is not in `[0, 1)`.
    pub fn with_beta1(mut self, beta1: f64) -> Result<Self, OptimizerError> {
        validate_decay_rate("beta1", beta1)?;
        self.beta1 = beta1;
        Ok(self)
    }

    /// Returns this transformation with `beta2` as the decay rate of the momentum.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `beta2` is not in `[0, 1)`.
    pub fn with_beta2(mut self, beta2: f64) -> Result<Self, OptimizerError> {
        validate_decay_rate("beta2", beta2)?;
        self.beta2 = beta2;
        Ok(self)
    }

    /// Returns this transformation with `mode` as the function that maps the interpolation to the update.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when the sharpness of [`LionMode::Smooth`] is not finite.
    pub fn with_mode(mut self, mode: LionMode) -> Result<Self, OptimizerError> {
        if let LionMode::Smooth { beta } = mode {
            validate_finite("beta", beta)?;
        }
        self.mode = mode;
        Ok(self)
    }

    /// Returns this transformation with its momenta stored in `data_type` (refer to
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

    /// Returns the interpolation rate `β₁` of this transformation between the gradient and its momentum.
    #[inline]
    pub fn beta1(&self) -> f64 {
        self.beta1
    }

    /// Returns the decay rate `β₂` of the momentum of this transformation.
    #[inline]
    pub fn beta2(&self) -> f64 {
        self.beta2
    }

    /// Returns the function that this transformation applies to the interpolation.
    #[inline]
    pub fn mode(&self) -> LionMode {
        self.mode
    }

    /// Returns the data type in which this transformation stores its momenta, or [`None`] if it stores them in the
    /// statistic data type of each leaf.
    #[inline]
    pub fn first_moment_data_type(&self) -> Option<DataType> {
        self.first_moment_data_type
    }
}

impl Default for ScaleByLion {
    fn default() -> Self {
        Self::new()
    }
}

impl<P: OptimizerValue> Optimizer<P> for ScaleByLion {
    type State<S: Parameterized<P>> = ScaleByLionState<P, S>;

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        Ok(ScaleByLionState {
            count: initial_count("ScaleByLion", parameters)?,
            first_moments: filled_statistics(parameters, 0.0, self.first_moment_data_type)?,
        })
    }

    fn update<S: Parameterized<P>>(
        &self,
        gradients: S,
        state: Self::State<S>,
        parameters: &S,
        arguments: &UpdateArguments<'_, P, S>,
    ) -> Result<(S, Self::State<S>), OptimizerError> {
        // This follows the operation order of `optax.scale_by_lion`.
        let _ = arguments;
        let paths = gradients.parameter_paths().collect::<Vec<_>>();
        validate_structure("parameters", &paths, parameters)?;
        validate_structure("first_moments", &paths, &state.first_moments)?;
        let structure = gradients.parameter_structure();
        let mut updates = Vec::with_capacity(paths.len());
        let mut first_moments = Vec::with_capacity(paths.len());
        for (gradient, first_moment) in gradients.into_parameters().zip(state.first_moments.into_parameters()) {
            let data_type = gradient.r#type().data_type();
            let statistic_type = statistic_data_type(data_type);
            let first_moment_type = first_moment.r#type().data_type();
            let gradient = convert_to(&gradient, statistic_type)?;
            let first_moment = convert_to(&first_moment, statistic_type)?;
            let interpolation = Add::add(
                &Mul::mul(&gradient.fill_like(1.0 - self.beta1)?, &gradient)?,
                &Mul::mul(&gradient.fill_like(self.beta1)?, &first_moment)?,
            )?;
            let update = match self.mode {
                LionMode::Hard => interpolation.sign()?,
                LionMode::Smooth { beta } => Mul::mul(&interpolation.fill_like(beta)?, &interpolation)?.tanh()?,
                LionMode::Refined => {
                    let small =
                        interpolation.abs()?.compare(&interpolation.fill_like(1f64)?, ComparisonDirection::LessThan)?;
                    small.select(&interpolation, &interpolation.sign()?)?
                }
            };
            let first_moment = update_moment(&gradient, &first_moment, self.beta2)?;
            updates.push(convert_to(&update, data_type)?);
            first_moments.push(convert_to(&first_moment, first_moment_type)?);
        }
        let state = ScaleByLionState {
            count: increment_count(&state.count)?,
            first_moments: S::from_parameters(structure.clone(), first_moments)?,
        };
        Ok((S::from_parameters(structure, updates)?, state))
    }
}

parameterized_state! {
    /// State of [`ScaleByRprop`], which holds the step sizes and the updates of the previous step, with the structure
    /// `S` of the parameters.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct ScaleByRpropState<P, S> with family ScaleByRpropStateFamily and marker leaf {
        /// Step sizes of the elements, with the structure of the parameters.
        step_sizes: S,

        /// Updates of the previous step, with the structure of the parameters.
        previous_updates: S,
    }
}

impl<P: Parameter, S> ScaleByRpropState<P, S> {
    /// Returns the step sizes of the elements, with the structure of the parameters.
    #[inline]
    pub fn step_sizes(&self) -> &S {
        &self.step_sizes
    }

    /// Returns the updates of the previous step, with the structure of the parameters.
    #[inline]
    pub fn previous_updates(&self) -> &S {
        &self.previous_updates
    }
}

/// Transformation that adapts a step size per element from the signs of consecutive gradients, which is the resilient
/// backpropagation (Rprop) of [Riedmiller and Braun (1993)](https://ieeexplore.ieee.org/document/298623) and the
/// analogue of
/// [`optax.scale_by_rprop`](https://optax.readthedocs.io/en/latest/api/transformations.html#optax.scale_by_rprop). For
/// a gradient `g`, the previous update `u` (which is zero at the first step), and the step size `s` (which starts at
/// the learning rate), it computes the product `p = g · u`. Where `p` is zero, `s` is kept, and elsewhere it is
/// multiplied by `η₊` where `p > 0` and by `η₋` where `p < 0` and clipped to the interval of the step sizes. The step
/// `0` where `p < 0` and `s · sign(g)` elsewhere is kept as the previous update of the next step, and, as in Optax, the
/// returned update is the previous update `u` that this step replaces, which is `0` where `p < 0`. The updates
/// therefore lag the steps by one, and the first update is zero. Note that the update ascends the gradient, so
/// optimizers negate it. The defaults are `η₋ = 0.5`, `η₊ = 1.2`, and step sizes in `[1e-6, 50]`.
#[derive(Clone, Debug, PartialEq)]
pub struct ScaleByRprop {
    /// Initial step size of every element.
    learning_rate: f64,

    /// Factor `η₋` by which the step sizes shrink where the gradient changes its sign.
    eta_minus: f64,

    /// Factor `η₊` by which the step sizes grow where the gradient keeps its sign.
    eta_plus: f64,

    /// Smallest step size.
    minimum_step_size: f64,

    /// Largest step size.
    maximum_step_size: f64,
}

impl ScaleByRprop {
    /// Creates a new [`ScaleByRprop`] whose step sizes start at `learning_rate`, with the default factors and interval.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `learning_rate` is negative or not finite.
    pub fn new(learning_rate: f64) -> Result<Self, OptimizerError> {
        validate_non_negative("learning_rate", learning_rate)?;
        Ok(Self { learning_rate, eta_minus: 0.5, eta_plus: 1.2, minimum_step_size: 1e-6, maximum_step_size: 50.0 })
    }

    /// Returns this transformation with `eta_minus` as the factor by which its step sizes shrink.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `eta_minus` is not positive and finite.
    pub fn with_eta_minus(mut self, eta_minus: f64) -> Result<Self, OptimizerError> {
        validate_positive("eta_minus", eta_minus)?;
        self.eta_minus = eta_minus;
        Ok(self)
    }

    /// Returns this transformation with `eta_plus` as the factor by which its step sizes grow.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `eta_plus` is not positive and finite.
    pub fn with_eta_plus(mut self, eta_plus: f64) -> Result<Self, OptimizerError> {
        validate_positive("eta_plus", eta_plus)?;
        self.eta_plus = eta_plus;
        Ok(self)
    }

    /// Returns this transformation with its step sizes clipped to `[minimum_step_size, maximum_step_size]`.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when a bound is negative or not finite, or when
    /// `minimum_step_size` is greater than `maximum_step_size`.
    pub fn with_step_size_bounds(
        mut self,
        minimum_step_size: f64,
        maximum_step_size: f64,
    ) -> Result<Self, OptimizerError> {
        validate_non_negative("minimum_step_size", minimum_step_size)?;
        validate_non_negative("maximum_step_size", maximum_step_size)?;
        if minimum_step_size > maximum_step_size {
            return Err(OptimizerError::InvalidConfiguration {
                message: format!(
                    "`minimum_step_size` must not be greater than `maximum_step_size` but got `{minimum_step_size}` \
                     and `{maximum_step_size}`",
                ),
            });
        }
        self.minimum_step_size = minimum_step_size;
        self.maximum_step_size = maximum_step_size;
        Ok(self)
    }

    /// Returns the initial step size of every element of this transformation.
    #[inline]
    pub fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    /// Returns the factor `η₋` by which the step sizes of this transformation shrink.
    #[inline]
    pub fn eta_minus(&self) -> f64 {
        self.eta_minus
    }

    /// Returns the factor `η₊` by which the step sizes of this transformation grow.
    #[inline]
    pub fn eta_plus(&self) -> f64 {
        self.eta_plus
    }

    /// Returns the smallest step size of this transformation.
    #[inline]
    pub fn minimum_step_size(&self) -> f64 {
        self.minimum_step_size
    }

    /// Returns the largest step size of this transformation.
    #[inline]
    pub fn maximum_step_size(&self) -> f64 {
        self.maximum_step_size
    }
}

impl<P: OptimizerValue> Optimizer<P> for ScaleByRprop {
    type State<S: Parameterized<P>> = ScaleByRpropState<P, S>;

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        Ok(ScaleByRpropState {
            step_sizes: filled_statistics(parameters, self.learning_rate, None)?,
            previous_updates: filled_statistics(parameters, 0.0, None)?,
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
        // This follows the operation order of `optax.scale_by_rprop`, whose final updates are the previous updates
        // that are stored in the state (i.e., not the new steps that replace them), masked to zero where the gradient
        // changes its sign.
        let _ = arguments;
        let paths = gradients.parameter_paths().collect::<Vec<_>>();
        validate_structure("parameters", &paths, parameters)?;
        validate_structure("step_sizes", &paths, &state.step_sizes)?;
        validate_structure("previous_updates", &paths, &state.previous_updates)?;
        let structure = gradients.parameter_structure();
        let mut updates = Vec::with_capacity(paths.len());
        let mut step_sizes = Vec::with_capacity(paths.len());
        let mut previous_updates = Vec::with_capacity(paths.len());
        let statistics = state.step_sizes.into_parameters().zip(state.previous_updates.into_parameters());
        for (gradient, (step_size, previous_update)) in gradients.into_parameters().zip(statistics) {
            let data_type = gradient.r#type().data_type();
            let statistic_type = statistic_data_type(data_type);
            let gradient = convert_to(&gradient, statistic_type)?;
            let step_size_type = step_size.r#type().data_type();
            let previous_update_type = previous_update.r#type().data_type();
            let step_size = convert_to(&step_size, statistic_type)?;
            let previous_update = convert_to(&previous_update, statistic_type)?;
            let product = Mul::mul(&gradient, &previous_update)?;
            let zero = gradient.fill_like(0f64)?;
            let factor = product
                .compare(&zero, ComparisonDirection::GreaterThan)?
                .select(&gradient.fill_like(self.eta_plus)?, &gradient.fill_like(self.eta_minus)?)?;
            let scaled = Mul::mul(&step_size, &factor)?
                .max(&gradient.fill_like(self.minimum_step_size)?)?
                .min(&gradient.fill_like(self.maximum_step_size)?)?;
            let step_size = product.compare(&zero, ComparisonDirection::Equal)?.select(&step_size, &scaled)?;
            let changed_sign = product.compare(&zero, ComparisonDirection::LessThan)?;
            let step = changed_sign.select(&zero, &Mul::mul(&step_size, &gradient.sign()?)?)?;
            let update = changed_sign.select(&zero, &previous_update)?;
            updates.push(convert_to(&update, data_type)?);
            step_sizes.push(convert_to(&step_size, step_size_type)?);
            previous_updates.push(convert_to(&step, previous_update_type)?);
        }
        let state = ScaleByRpropState {
            step_sizes: S::from_parameters(structure.clone(), step_sizes)?,
            previous_updates: S::from_parameters(structure.clone(), previous_updates)?,
            leaf: PhantomData,
        };
        Ok((S::from_parameters(structure, updates)?, state))
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::tests::{assert_parity, parity_updates, state_paths, vector_updates};

    use super::*;

    #[test]
    fn test_scale_by_sign_update() {
        assert_eq!(ScaleBySign::new(), ScaleBySign);
        assert_eq!(
            vector_updates(&ScaleBySign::new(), &[0.0, 0.0, 0.0], &[&[2.5, -0.5, 0.0], &[-3.0, 1.0, 4.0]]),
            vec![vec![1.0, -1.0, 0.0], vec![-1.0, 1.0, 1.0],],
        );
    }

    #[test]
    fn test_scale_by_lion_new() {
        let lion = ScaleByLion::new();
        assert_eq!((lion.beta1(), lion.beta2(), lion.mode()), (0.9, 0.99, LionMode::Hard));
        assert_eq!(lion.first_moment_data_type(), None);
        assert_eq!(ScaleByLion::default(), lion);
        let lion = lion
            .with_beta1(0.5)
            .unwrap()
            .with_beta2(0.6)
            .unwrap()
            .with_mode(LionMode::Smooth { beta: 2.0 })
            .unwrap()
            .with_first_moment_data_type(DataType::BF16)
            .unwrap();
        assert_eq!((lion.beta1(), lion.beta2(), lion.mode()), (0.5, 0.6, LionMode::Smooth { beta: 2.0 }));
        assert_eq!(lion.first_moment_data_type(), Some(DataType::BF16));
        assert!(matches!(
            ScaleByLion::new().with_mode(LionMode::Smooth { beta: f64::NAN }),
            Err(OptimizerError::InvalidConfiguration { message }) if message == "`beta` must be finite but got `NaN`",
        ));
    }

    #[test]
    fn test_scale_by_lion_initialize() {
        assert_eq!(
            state_paths(&ScaleByLion::new()),
            vec!["$.count", "$.first_moments[\"bias\"]", "$.first_moments[\"weights\"]"],
        );
    }

    #[test]
    fn test_scale_by_lion_update() {
        // With `β₁ = β₂ = 0.5`, the gradient `[2, -0.5]` gives the interpolation `[1, -0.25]` and the momentum
        // `[1, -0.25]`. The gradient `[-0.5, 0.25]` then gives the interpolation `[0.25, 0]`.
        let lion = ScaleByLion::new().with_beta1(0.5).unwrap().with_beta2(0.5).unwrap();
        let gradients: [&[f32]; 2] = [&[2.0, -0.5], &[-0.5, 0.25]];
        assert_eq!(vector_updates(&lion, &[0.0, 0.0], &gradients), vec![vec![1.0, -1.0], vec![1.0, 0.0]]);

        // The refined mode keeps interpolations of magnitude below `1`, and the smooth mode applies `tanh(2 · x)`.
        let refined = lion.clone().with_mode(LionMode::Refined).unwrap();
        assert_eq!(vector_updates(&refined, &[0.0, 0.0], &gradients), vec![vec![1.0, -0.25], vec![0.25, 0.0]]);
        let smooth = lion.with_mode(LionMode::Smooth { beta: 2.0 }).unwrap();
        assert_eq!(
            vector_updates(&smooth, &[0.0, 0.0], &gradients[..1]),
            vec![vec![(2.0f32 * 1.0).tanh(), (2.0f32 * -0.25).tanh(),]],
        );
    }

    #[test]
    fn test_scale_by_rprop_new() {
        let rprop = ScaleByRprop::new(0.1).unwrap();
        assert_eq!(
            (rprop.learning_rate(), rprop.eta_minus(), rprop.eta_plus(), rprop.minimum_step_size()),
            (0.1, 0.5, 1.2, 1e-6),
        );
        assert_eq!(rprop.maximum_step_size(), 50.0);
        let rprop = rprop
            .with_eta_minus(0.25)
            .unwrap()
            .with_eta_plus(2.0)
            .unwrap()
            .with_step_size_bounds(0.05, 0.3)
            .unwrap();
        assert_eq!(
            (rprop.eta_minus(), rprop.eta_plus(), rprop.minimum_step_size(), rprop.maximum_step_size()),
            (0.25, 2.0, 0.05, 0.3),
        );
        assert!(matches!(
            ScaleByRprop::new(0.1).unwrap().with_step_size_bounds(1.0, 0.5),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`minimum_step_size` must not be greater than `maximum_step_size` but got `1` and `0.5`",
        ));
        assert!(matches!(
            ScaleByRprop::new(0.1).unwrap().with_eta_plus(0.0),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`eta_plus` must be positive and finite but got `0`",
        ));
    }

    #[test]
    fn test_scale_by_rprop_initialize() {
        assert_eq!(
            state_paths(&ScaleByRprop::new(0.1).unwrap()),
            vec![
                "$.step_sizes[\"bias\"]",
                "$.step_sizes[\"weights\"]",
                "$.previous_updates[\"bias\"]",
                "$.previous_updates[\"weights\"]",
            ],
        );
    }

    #[test]
    fn test_scale_by_rprop_update() {
        // The step sizes start at `0.5`. At the first step the previous updates are zero, so the step sizes are kept,
        // the steps `0.5 · sign(g)` are stored, and the updates are the previous updates `0`. At the second step, the
        // first element keeps its sign, so its step size grows to `min(0.5 · 2, 0.8) = 0.8` and its update is the
        // stored step `0.5`, and the second element changes its sign, so its step size shrinks to `0.5 · 0.25` and its
        // update and stored step are zero. At the third step, the updates are the stored steps `[0.8, 0]`.
        let rprop = ScaleByRprop::new(0.5)
            .unwrap()
            .with_eta_minus(0.25)
            .unwrap()
            .with_eta_plus(2.0)
            .unwrap()
            .with_step_size_bounds(0.1, 0.8)
            .unwrap();
        assert_eq!(
            vector_updates(&rprop, &[0.0, 0.0], &[&[3.0, -1.0], &[1.0, 2.0], &[1.0, 2.0]]),
            vec![vec![0.0, 0.0], vec![0.5, 0.0], vec![0.8, 0.0]],
        );
    }

    #[test]
    fn test_scale_by_sign_optax_parity() {
        // Optax 9ecde04b: `optax.scale_by_sign()`.
        const EXPECTED: [[f32; 9]; 4] = [
            [1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
            [1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0],
        ];
        assert_parity(&parity_updates(&ScaleBySign::new()), &EXPECTED);
    }

    #[test]
    fn test_scale_by_lion_optax_parity() {
        // Optax 9ecde04b: `optax.scale_by_lion(b1=0.9, b2=0.99)`.
        const LION_HARD: [[f32; 9]; 4] = [
            [1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
            [1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0],
        ];
        // Optax 9ecde04b: `optax.scale_by_lion(b1=0.9, b2=0.99, mode='smooth', smooth_beta=2.0)`.
        const LION_SMOOTH: [[f32; 9]; 4] = [
            [
                0.019997332,
                -0.03997868,
                0.059928104,
                0.07982977,
                -0.019997332,
                0.03997868,
                -0.059928104,
                0.09966799,
                -0.11942729,
            ],
            [
                -0.008199817,
                0.046366725,
                0.02539454,
                0.047164977,
                0.05813437,
                -0.0762517,
                0.014598964,
                -0.030990072,
                0.059130933,
            ],
            [
                0.060806885,
                -0.01906169,
                -0.03284218,
                -0.019269612,
                0.013617158,
                0.0861499,
                0.046420608,
                0.08510364,
                -0.024387164,
            ],
            [
                0.046240177,
                0.029118404,
                -0.06642749,
                0.027913468,
                -0.05546121,
                0.014499343,
                -0.0887762,
                0.052408833,
                0.053799924,
            ],
        ];
        // Optax 9ecde04b: `optax.scale_by_lion(b1=0.5, b2=0.9, mode='refined')`.
        const LION_REFINED: [[f32; 9]; 4] = [
            [0.05, -0.1, 0.15, 0.2, -0.05, 0.1, -0.15, 0.25, -0.3],
            [-0.02, 0.115, 0.065, 0.120000005, 0.14500001, -0.19, 0.035, -0.075, 0.145],
            [0.15200001, -0.0465, -0.0815, -0.047000002, 0.0355, 0.21399999, 0.1165, 0.2125, -0.0595],
            [0.1168, 0.07315, -0.16835, 0.0677, -0.13805, 0.0376, -0.22015, 0.13125, 0.13645001],
        ];
        let lion = ScaleByLion::new().with_beta1(0.9).unwrap().with_beta2(0.99).unwrap();
        assert_parity(&parity_updates(&lion.clone().with_mode(LionMode::Hard).unwrap()), &LION_HARD);
        assert_parity(&parity_updates(&lion.with_mode(LionMode::Smooth { beta: 2.0 }).unwrap()), &LION_SMOOTH);
        let lion = ScaleByLion::new().with_beta1(0.5).unwrap().with_beta2(0.9).unwrap();
        assert_parity(&parity_updates(&lion.with_mode(LionMode::Refined).unwrap()), &LION_REFINED);
    }

    #[test]
    fn test_scale_by_rprop_optax_parity() {
        // Optax 9ecde04b: `optax.scale_by_rprop(0.1)`.
        const RPROP: [[f32; 9]; 4] = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.05, 0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.05, 0.0],
        ];
        // Optax 9ecde04b: `optax.scale_by_rprop(0.1, eta_minus=0.25, eta_plus=2.0, min_step_size=0.05,
        // max_step_size=0.3)`.
        const RPROP_BOUNDED: [[f32; 9]; 4] = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.05, 0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.05, 0.0],
        ];
        assert_parity(&parity_updates(&ScaleByRprop::new(0.1).unwrap()), &RPROP);
        let rprop = ScaleByRprop::new(0.1)
            .unwrap()
            .with_eta_minus(0.25)
            .unwrap()
            .with_eta_plus(2.0)
            .unwrap()
            .with_step_size_bounds(0.05, 0.3)
            .unwrap();
        assert_parity(&parity_updates(&rprop), &RPROP_BOUNDED);
    }
}
