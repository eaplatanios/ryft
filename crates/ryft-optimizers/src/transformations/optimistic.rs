//! Optimistic gradient steps, which extrapolate from the previous gradient and are used for min-max problems.

use ryft_core::{ComparisonDirection, Mul, Parameter, Parameterized, Sub};

use crate::macros::parameterized_state;
use crate::{
    Optimizer, OptimizerError, OptimizerValue, UpdateArguments, convert_to, filled_statistics, initial_count,
    statistic_data_type, validate_finite, validate_structure,
};

// TODO(eaplatanios): Review this module.

parameterized_state! {
    /// State of [`ScaleByOptimisticGradient`], which holds whether no update has been applied yet and the gradients of
    /// the previous update, with the structure `S` of the parameters.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct ScaleByOptimisticGradientState<P, S> with family ScaleByOptimisticGradientStateFamily {
        /// Rank-0 boolean array that holds whether no update has been applied yet.
        initial_step: P,

        /// Gradients of the previous update, with the structure of the parameters.
        previous_gradients: S,
    }
}

impl<P: Parameter, S> ScaleByOptimisticGradientState<P, S> {
    /// Returns the rank-0 boolean array that holds whether no update has been applied yet.
    #[inline]
    pub fn initial_step(&self) -> &P {
        &self.initial_step
    }

    /// Returns the gradients of the previous update, with the structure of the parameters.
    #[inline]
    pub fn previous_gradients(&self) -> &S {
        &self.previous_gradients
    }
}

/// Transformation that extrapolates from the previous gradient, which is the optimistic gradient method of
/// [Mokhtari et al. (2019)](https://arxiv.org/abs/1901.08511) and the analogue of
/// [`optax.scale_by_optimistic_gradient`](
/// https://optax.readthedocs.io/en/latest/api/transformations.html#optax.scale_by_optimistic_gradient).
/// For a gradient `g` and the previous gradient `g₋₁` (which is `g` itself at the first update), it returns
/// `(α + β) · g - β · g₋₁`. The defaults are `α = 1` and `β = 1`.
#[derive(Clone, Debug, PartialEq)]
pub struct ScaleByOptimisticGradient {
    /// Coefficient `α` of the current gradient.
    alpha: f64,

    /// Coefficient `β` of the optimism, which is the difference between the current and the previous gradients.
    beta: f64,
}

impl ScaleByOptimisticGradient {
    /// Creates a new [`ScaleByOptimisticGradient`] with the default coefficients.
    #[inline]
    pub fn new() -> Self {
        Self { alpha: 1.0, beta: 1.0 }
    }

    /// Returns this transformation with `alpha` as the coefficient of the current gradient.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `alpha` is not finite.
    pub fn with_alpha(mut self, alpha: f64) -> Result<Self, OptimizerError> {
        validate_finite("alpha", alpha)?;
        self.alpha = alpha;
        Ok(self)
    }

    /// Returns this transformation with `beta` as the coefficient of the optimism.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `beta` is not finite.
    pub fn with_beta(mut self, beta: f64) -> Result<Self, OptimizerError> {
        validate_finite("beta", beta)?;
        self.beta = beta;
        Ok(self)
    }

    /// Returns the coefficient `α` of the current gradient of this transformation.
    #[inline]
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Returns the coefficient `β` of the optimism of this transformation.
    #[inline]
    pub fn beta(&self) -> f64 {
        self.beta
    }
}

impl Default for ScaleByOptimisticGradient {
    fn default() -> Self {
        Self::new()
    }
}

impl<P: OptimizerValue> Optimizer<P> for ScaleByOptimisticGradient {
    type State<S: Parameterized<P>> = ScaleByOptimisticGradientState<P, S>;

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        // Comparing a zero with itself for equality creates the rank-0 boolean `true` without another bound.
        let zero = initial_count("ScaleByOptimisticGradient", parameters)?;
        Ok(ScaleByOptimisticGradientState {
            initial_step: zero.compare(&zero, ComparisonDirection::Equal)?,
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
        // This follows the operation order of `optax.scale_by_optimistic_gradient`, in which `α + β` is computed on the
        // host.
        let _ = arguments;
        let paths = gradients.parameter_paths().collect::<Vec<_>>();
        validate_structure("parameters", &paths, parameters)?;
        validate_structure("previous_gradients", &paths, &state.previous_gradients)?;
        let structure = gradients.parameter_structure();
        let mut updates = Vec::with_capacity(paths.len());
        let mut previous_gradients = Vec::with_capacity(paths.len());
        for (gradient, previous_gradient) in gradients.into_parameters().zip(state.previous_gradients.into_parameters())
        {
            let data_type = gradient.r#type().data_type();
            let statistic_type = statistic_data_type(data_type);
            let previous_gradient_type = previous_gradient.r#type().data_type();
            let gradient = convert_to(&gradient, statistic_type)?;
            let previous_gradient =
                state.initial_step.select(&gradient, &convert_to(&previous_gradient, statistic_type)?)?;
            let update = Sub::sub(
                &Mul::mul(&gradient.fill_like(self.alpha + self.beta)?, &gradient)?,
                &Mul::mul(&gradient.fill_like(self.beta)?, &previous_gradient)?,
            )?;
            updates.push(convert_to(&update, data_type)?);
            previous_gradients.push(convert_to(&gradient, previous_gradient_type)?);
        }
        let state = ScaleByOptimisticGradientState {
            initial_step: state.initial_step.compare(&state.initial_step, ComparisonDirection::NotEqual)?,
            previous_gradients: S::from_parameters(structure.clone(), previous_gradients)?,
        };
        Ok((S::from_parameters(structure, updates)?, state))
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use ryft_core::Array;

    use crate::tests::{assert_parity, parity_updates, state_paths, vector_updates};

    use super::*;

    #[test]
    fn test_scale_by_optimistic_gradient_new() {
        let optimistic = ScaleByOptimisticGradient::new();
        assert_eq!((optimistic.alpha(), optimistic.beta()), (1.0, 1.0));
        assert_eq!(ScaleByOptimisticGradient::default(), optimistic);
        let optimistic = optimistic.with_alpha(0.5).unwrap().with_beta(2.0).unwrap();
        assert_eq!((optimistic.alpha(), optimistic.beta()), (0.5, 2.0));
        assert!(matches!(
            ScaleByOptimisticGradient::new().with_beta(f64::NAN),
            Err(OptimizerError::InvalidConfiguration { message }) if message == "`beta` must be finite but got `NaN`",
        ));
    }

    #[test]
    fn test_scale_by_optimistic_gradient_initialize() {
        assert_eq!(
            state_paths(&ScaleByOptimisticGradient::new()),
            vec!["$.initial_step", "$.previous_gradients[\"bias\"]", "$.previous_gradients[\"weights\"]"],
        );
        let parameters = vec![Array::scalar(1.0f32).unwrap()];
        let state = ScaleByOptimisticGradient::new().initialize(&parameters).unwrap();
        assert_eq!(state.initial_step(), &Array::scalar(true).unwrap());
    }

    #[test]
    fn test_scale_by_optimistic_gradient_update() {
        // With `α = 0.5` and `β = 2`, the first gradient `2` is its own previous gradient, so the update is
        // `2.5 · 2 - 2 · 2 = 1`, and the second gradient `1` gives `2.5 · 1 - 2 · 2 = -1.5`.
        let optimistic = ScaleByOptimisticGradient::new().with_alpha(0.5).unwrap().with_beta(2.0).unwrap();
        assert_eq!(vector_updates(&optimistic, &[0.0], &[&[2.0], &[1.0]]), vec![vec![1.0], vec![-1.5]]);
    }

    #[test]
    fn test_scale_by_optimistic_gradient_optax_parity() {
        // Optax 9ecde04b: `optax.scale_by_optimistic_gradient()`.
        const OPTIMISTIC: [[f32; 9]; 4] = [
            [0.1, -0.2, 0.3, 0.4, -0.1, 0.2, -0.3, 0.5, -0.6],
            [-0.2, 0.7, -0.10000001, 0.0, 0.70000005, -1.0, 0.5, -0.9, 1.3],
            [0.65000004, -0.45, -0.5, -0.5, -0.20000002, 1.3, 0.4, 1.0, -0.55],
            [0.099999994, 0.4, -0.5, 0.35000002, -0.65000004, -0.35, -1.15, 0.0, 0.70000005],
        ];
        // Optax 9ecde04b: `optax.scale_by_optimistic_gradient(alpha=0.5, beta=0.8)`.
        const OPTIMISTIC_CONFIGURED: [[f32; 9]; 4] = [
            [0.04999999, -0.09999998, 0.15, 0.19999996, -0.04999999, 0.09999998, -0.15, 0.24999997, -0.3],
            [-0.14500001, 0.485, -0.110000014, -0.060000032, 0.47000003, -0.68, 0.37, -0.65999997, 0.935],
            [0.43, -0.32999998, -0.34, -0.35500002, -0.17500001, 0.905, 0.24499997, 0.68, -0.41],
            [0.01999998, 0.275, -0.29499996, 0.25, -0.43, -0.295, -0.78499997, -0.060000032, 0.47000003],
        ];
        assert_parity(&parity_updates(&ScaleByOptimisticGradient::new()), &OPTIMISTIC);
        let optimistic = ScaleByOptimisticGradient::new().with_alpha(0.5).unwrap().with_beta(0.8).unwrap();
        assert_parity(&parity_updates(&optimistic), &OPTIMISTIC_CONFIGURED);
    }
}
