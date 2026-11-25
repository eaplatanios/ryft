//! Reduction of the scale of the updates when the objective stops improving.

use ryft_core::{Add, ComparisonDirection, Div, Mul, Not, Parameter, Parameterized, Sub};

use crate::{
    Optimizer, OptimizerError, OptimizerValue, UpdateArguments, convert_to, increment_count, initial_count,
    statistic_data_type, validate_finite, validate_structure,
};

// TODO(eaplatanios): Review this module.

/// State of [`ReduceOnPlateau`], which is the analogue of
/// [`optax.contrib.ReduceLROnPlateauState`](
/// https://optax.readthedocs.io/en/latest/api/contrib.html#optax.contrib.ReduceLROnPlateauState).
/// Its rank-0 floating-point arrays have the statistic data type of the first parameter.
#[derive(Clone, Debug, PartialEq, Eq, Hash, ryft_macros::Parameterized)]
#[ryft(crate = "ryft_core")]
pub struct ReduceOnPlateauState<P: Parameter> {
    /// Rank-0 array that holds the scale of the updates.
    scale: P,

    /// Rank-0 array that holds the best average value of the objective so far.
    best_value: P,

    /// Rank-0 `i32` array that holds the number of consecutive accumulations without an improvement.
    plateau_count: P,

    /// Rank-0 `i32` array that holds the number of accumulations left in the current cooldown.
    cooldown_count: P,

    /// Rank-0 `i32` array that holds the number of values accumulated since the last accumulation was complete.
    count: P,

    /// Rank-0 array that holds the average of the values accumulated since the last accumulation was complete, which
    /// is the `avg_value` of Optax.
    average_value: P,
}

impl<P: Parameter> ReduceOnPlateauState<P> {
    /// Returns the rank-0 array that holds the scale of the updates.
    #[inline]
    pub fn scale(&self) -> &P {
        &self.scale
    }

    /// Returns the rank-0 array that holds the best average value of the objective so far.
    #[inline]
    pub fn best_value(&self) -> &P {
        &self.best_value
    }

    /// Returns the rank-0 `i32` array that holds the number of consecutive accumulations without an improvement.
    #[inline]
    pub fn plateau_count(&self) -> &P {
        &self.plateau_count
    }

    /// Returns the rank-0 `i32` array that holds the number of accumulations left in the current cooldown.
    #[inline]
    pub fn cooldown_count(&self) -> &P {
        &self.cooldown_count
    }

    /// Returns the rank-0 `i32` array that holds the number of values accumulated since the last complete
    /// accumulation.
    #[inline]
    pub fn count(&self) -> &P {
        &self.count
    }

    /// Returns the rank-0 array that holds the average of the values accumulated since the last complete accumulation.
    #[inline]
    pub fn average_value(&self) -> &P {
        &self.average_value
    }
}

/// Transformation that scales the updates and reduces the scale by a factor when the objective stops improving, which
/// is the analogue of [`optax.contrib.reduce_on_plateau`](
/// https://optax.readthedocs.io/en/latest/api/contrib.html#optax.contrib.reduce_on_plateau). At the end of every
/// accumulation `k` of `A` updates, it computes
///
/// ```text
/// v̄ₖ = mean of the values f of the A updates of accumulation k
/// bₖ = v̄ₖ                    if v̄ₖ < (1 − δ) · bₖ₋₁ − ε
/// bₖ = bₖ₋₁                  otherwise
/// pₖ = 0                     if v̄ₖ < (1 − δ) · bₖ₋₁ − ε
/// pₖ = pₖ₋₁ + 1              otherwise
/// σₖ = max(ρ · σₖ₋₁, σₘᵢₙ)   if pₖ = P
/// σₖ = σₖ₋₁                  otherwise
/// ```
///
/// and it returns the updates `σₖ · gₜ` from the last update of accumulation `k` until the next accumulation completes.
/// Here `gₜ` are the incoming updates (e.g., those of a preceding optimizer), the values `f` are read from the `value`
/// [`UpdateArguments`], `v̄ₖ` is their average, `bₖ` is the best average so far, which starts at infinity, `δ` and `ε`
/// are the relative and absolute tolerances of an improvement, `pₖ` counts the accumulations without an improvement,
/// `P` is the patience, `σₖ` is the scale, which starts at `1`, `ρ` is the factor by which it is reduced, and `σₘᵢₙ` is
/// the minimum scale. A reduction resets `pₖ` to zero and starts a cooldown of `C` accumulations, during which `pₖ = 0`
/// and `σₖ = σₖ₋₁` (while `bₖ` is still updated). The scaling is element-wise. The defaults are `ρ = 0.1`, `P = 10`,
/// `δ = 1e-4`, `ε = 0`, `C = 0`, `A = 1`, and `σₘᵢₙ = 0`.
///
/// The decisions are made with array operations (i.e., both outcomes are computed and selected), so they are staged
/// inside compiled training steps. This transformation is not a chain, and it is typically placed after an optimizer
/// with [`Chain`](crate::Chain) to scale its updates.
#[derive(Clone, Debug, PartialEq)]
pub struct ReduceOnPlateau {
    /// Factor by which the scale is reduced on a plateau.
    factor: f64,

    /// Number of consecutive accumulations without an improvement after which the scale is reduced.
    patience: usize,

    /// Relative tolerance `δ` of an improvement.
    relative_tolerance: f64,

    /// Absolute tolerance `ε` of an improvement.
    absolute_tolerance: f64,

    /// Number of accumulations after a reduction during which no plateau is counted.
    cooldown: usize,

    /// Number of values that are averaged before they are compared with the best average.
    accumulation_size: usize,

    /// Minimum scale.
    minimum_scale: f64,
}

impl ReduceOnPlateau {
    /// Creates a new [`ReduceOnPlateau`] with default values for all hyperparameters.
    #[inline]
    pub fn new() -> Self {
        Self {
            factor: 0.1,
            patience: 10,
            relative_tolerance: 1e-4,
            absolute_tolerance: 0.0,
            cooldown: 0,
            accumulation_size: 1,
            minimum_scale: 0.0,
        }
    }

    /// Returns this transformation with `factor` as the factor by which it reduces the scale.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `factor` is not in `(0, 1)`.
    pub fn with_factor(mut self, factor: f64) -> Result<Self, OptimizerError> {
        if !(factor > 0.0 && factor < 1.0) {
            return Err(OptimizerError::InvalidConfiguration {
                message: format!("`factor` must be in `(0, 1)` but got `{factor}`"),
            });
        }
        self.factor = factor;
        Ok(self)
    }

    /// Returns this transformation with `patience` as the number of consecutive accumulations without an improvement
    /// after which it reduces the scale.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `patience` is greater than `i32::MAX`.
    pub fn with_patience(mut self, patience: usize) -> Result<Self, OptimizerError> {
        validate_count("patience", patience)?;
        self.patience = patience;
        Ok(self)
    }

    /// Returns this transformation with the relative tolerance `relative_tolerance` and the absolute tolerance
    /// `absolute_tolerance` of an improvement, which are set together because at least one of them must be positive.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when either tolerance is negative or not finite, when both are
    /// zero, or when `relative_tolerance` is greater than `1`.
    pub fn with_tolerances(mut self, relative_tolerance: f64, absolute_tolerance: f64) -> Result<Self, OptimizerError> {
        if !relative_tolerance.is_finite()
            || !absolute_tolerance.is_finite()
            || relative_tolerance < 0.0
            || absolute_tolerance < 0.0
        {
            return Err(OptimizerError::InvalidConfiguration {
                message: format!(
                    "`relative_tolerance` and `absolute_tolerance` must be finite and non-negative but got \
                     `{relative_tolerance}` and `{absolute_tolerance}`",
                ),
            });
        }
        if relative_tolerance == 0.0 && absolute_tolerance == 0.0 {
            return Err(OptimizerError::InvalidConfiguration {
                message: "at least one of `relative_tolerance` and `absolute_tolerance` must be positive".to_string(),
            });
        }
        if relative_tolerance > 1.0 {
            return Err(OptimizerError::InvalidConfiguration {
                message: format!("`relative_tolerance` must be at most `1` but got `{relative_tolerance}`"),
            });
        }
        self.relative_tolerance = relative_tolerance;
        self.absolute_tolerance = absolute_tolerance;
        Ok(self)
    }

    /// Returns this transformation with `cooldown` as the number of accumulations after a reduction during which it
    /// counts no plateau.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `cooldown` is greater than `i32::MAX`.
    pub fn with_cooldown(mut self, cooldown: usize) -> Result<Self, OptimizerError> {
        validate_count("cooldown", cooldown)?;
        self.cooldown = cooldown;
        Ok(self)
    }

    /// Returns this transformation with `accumulation_size` as the number of values that it averages before it
    /// compares them with the best average.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `accumulation_size` is zero or greater than `i32::MAX`.
    pub fn with_accumulation_size(mut self, accumulation_size: usize) -> Result<Self, OptimizerError> {
        validate_count("accumulation_size", accumulation_size)?;
        if accumulation_size == 0 {
            return Err(OptimizerError::InvalidConfiguration {
                message: "`accumulation_size` must be positive but got `0`".to_string(),
            });
        }
        self.accumulation_size = accumulation_size;
        Ok(self)
    }

    /// Returns this transformation with `minimum_scale` as the scale below which it does not reduce the scale.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `minimum_scale` is not finite.
    pub fn with_minimum_scale(mut self, minimum_scale: f64) -> Result<Self, OptimizerError> {
        validate_finite("minimum_scale", minimum_scale)?;
        self.minimum_scale = minimum_scale;
        Ok(self)
    }

    /// Returns the factor by which this transformation reduces the scale.
    #[inline]
    pub fn factor(&self) -> f64 {
        self.factor
    }

    /// Returns the number of consecutive accumulations without an improvement after which this transformation reduces
    /// the scale.
    #[inline]
    pub fn patience(&self) -> usize {
        self.patience
    }

    /// Returns the relative tolerance `δ` of an improvement.
    #[inline]
    pub fn relative_tolerance(&self) -> f64 {
        self.relative_tolerance
    }

    /// Returns the absolute tolerance `ε` of an improvement.
    #[inline]
    pub fn absolute_tolerance(&self) -> f64 {
        self.absolute_tolerance
    }

    /// Returns the number of accumulations after a reduction during which this transformation counts no plateau.
    #[inline]
    pub fn cooldown(&self) -> usize {
        self.cooldown
    }

    /// Returns the number of values that this transformation averages before it compares them with the best average.
    #[inline]
    pub fn accumulation_size(&self) -> usize {
        self.accumulation_size
    }

    /// Returns the scale below which this transformation does not reduce the scale.
    #[inline]
    pub fn minimum_scale(&self) -> f64 {
        self.minimum_scale
    }
}

impl Default for ReduceOnPlateau {
    fn default() -> Self {
        Self::new()
    }
}

impl<P: OptimizerValue> Optimizer<P> for ReduceOnPlateau {
    type State<S: Parameterized<P>> = ReduceOnPlateauState<P>;

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        let count = initial_count("ReduceOnPlateau", parameters)?;
        let data_type = statistic_data_type(parameters.parameters().next().unwrap().r#type().data_type());
        let scalar = convert_to(&count, data_type)?;
        Ok(ReduceOnPlateauState {
            scale: scalar.fill_like(1f64)?,
            best_value: scalar.fill_like(f64::INFINITY)?,
            plateau_count: count.clone(),
            cooldown_count: count.clone(),
            count,
            average_value: scalar.fill_like(0f64)?,
        })
    }

    fn update<S: Parameterized<P>>(
        &self,
        gradients: S,
        state: Self::State<S>,
        parameters: &S,
        arguments: &UpdateArguments<'_, P, S>,
    ) -> Result<(S, Self::State<S>), OptimizerError> {
        // This follows `optax.contrib.reduce_on_plateau`, except that both branches of its `lax.cond`s are computed and
        // selected.
        let value = arguments.validate_value("ReduceOnPlateau")?;
        validate_structure("parameters", &gradients.parameter_paths().collect::<Vec<_>>(), parameters)?;
        let data_type = state.average_value.r#type().data_type();
        let count = increment_count(&state.count)?;
        let average_value = Div::div(
            &Add::add(
                &Mul::mul(&convert_to(&state.count, data_type)?, &state.average_value)?,
                &convert_to(value, data_type)?,
            )?,
            &convert_to(&count, data_type)?,
        )?;
        let complete = count.compare(&count.fill_like(self.accumulation_size as i32)?, ComparisonDirection::Equal)?;

        // Update of the scale at the end of an accumulation (i.e., `_update_scale`).
        let threshold = Sub::sub(
            &Mul::mul(&state.best_value.fill_like(1.0 - self.relative_tolerance)?, &state.best_value)?,
            &state.best_value.fill_like(self.absolute_tolerance)?,
        )?;
        let improved = average_value.compare(&threshold, ComparisonDirection::LessThan)?;
        let best_value = improved.select(&average_value, &state.best_value)?;
        let zero = count.fill_like(0i32)?;
        let plateau_count = improved.select(&zero, &increment_count(&state.plateau_count)?)?;
        let in_cooldown = state.cooldown_count.compare(&zero, ComparisonDirection::GreaterThan)?;
        let reduce = plateau_count.compare(&zero.fill_like(self.patience as i32)?, ComparisonDirection::Equal)?;
        let reduced_scale = reduce
            .select(&Mul::mul(&state.scale.fill_like(self.factor)?, &state.scale)?, &state.scale)?
            .max(&state.scale.fill_like(self.minimum_scale)?)?;
        let scale = in_cooldown.select(&state.scale, &reduced_scale)?;
        let plateau_count = in_cooldown.or(&reduce)?.select(&zero, &plateau_count)?;
        let cooldown_count = in_cooldown.select(
            &Sub::sub(&state.cooldown_count, &zero.fill_like(1i32)?)?,
            &reduce.select(&zero.fill_like(self.cooldown as i32)?, &zero)?,
        )?;

        let accumulating = Not::not(&complete)?;
        let state = ReduceOnPlateauState {
            scale: accumulating.select(&state.scale, &scale)?,
            best_value: accumulating.select(&state.best_value, &best_value)?,
            plateau_count: accumulating.select(&state.plateau_count, &plateau_count)?,
            cooldown_count: accumulating.select(&state.cooldown_count, &cooldown_count)?,
            count: accumulating.select(&count, &zero)?,
            average_value: accumulating.select(&average_value, &average_value.fill_like(0f64)?)?,
        };
        let structure = gradients.parameter_structure();
        let updates = gradients
            .into_parameters()
            .map(|gradient| {
                let gradient_type = gradient.r#type().data_type();
                let statistic_type = statistic_data_type(gradient_type);
                let update =
                    Mul::mul(&convert_to(&state.scale, statistic_type)?, &convert_to(&gradient, statistic_type)?)?;
                Ok(convert_to(&update, gradient_type)?)
            })
            .collect::<Result<Vec<_>, OptimizerError>>()?;
        Ok((S::from_parameters(structure, updates)?, state))
    }
}

/// Validates that the count hyperparameter `name` has a `value` of at most `i32::MAX`, which the counts of the state
/// cannot exceed.
fn validate_count(name: &str, value: usize) -> Result<(), OptimizerError> {
    if i32::try_from(value).is_err() {
        return Err(OptimizerError::InvalidConfiguration {
            message: format!("`{name}` must be at most `{}` but got `{value}`", i32::MAX),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use ryft_core::{Array, DataType, Typed};

    use crate::apply_updates;
    use crate::tests::{
        PARITY_GRADIENTS, PARITY_PARAMETERS, assert_parity, parity_elements, parity_parameters, state_paths,
    };

    use super::*;

    /// Returns the scales and the updates of the gradient `2` that `optimizer` returns for the objective values
    /// `values` in order.
    fn scales(optimizer: &ReduceOnPlateau, values: &[f32]) -> (Vec<f32>, ReduceOnPlateauState<Array>) {
        let parameters = vec![Array::scalar(0.0f32).unwrap()];
        let mut state = optimizer.initialize(&parameters).unwrap();
        let mut updates = Vec::new();
        for value in values {
            let value = Array::scalar(*value).unwrap();
            let arguments = UpdateArguments::new().with_value(&value);
            let (step_updates, next_state) =
                optimizer.update(vec![Array::scalar(2.0f32).unwrap()], state, &parameters, &arguments).unwrap();
            updates.push(step_updates[0].elements::<f32>().unwrap()[0]);
            state = next_state;
        }
        (updates, state)
    }

    #[test]
    fn test_reduce_on_plateau_new() {
        let optimizer = ReduceOnPlateau::new();
        assert_eq!(optimizer.factor(), 0.1);
        assert_eq!(optimizer.patience(), 10);
        assert_eq!((optimizer.relative_tolerance(), optimizer.absolute_tolerance()), (1e-4, 0.0));
        assert_eq!(optimizer.cooldown(), 0);
        assert_eq!(optimizer.accumulation_size(), 1);
        assert_eq!(optimizer.minimum_scale(), 0.0);
        assert_eq!(ReduceOnPlateau::default(), optimizer);
    }

    #[test]
    fn test_reduce_on_plateau_with_hyperparameters() {
        let optimizer = ReduceOnPlateau::new()
            .with_factor(0.5)
            .unwrap()
            .with_patience(2)
            .unwrap()
            .with_tolerances(0.0, 0.1)
            .unwrap()
            .with_cooldown(1)
            .unwrap()
            .with_accumulation_size(3)
            .unwrap()
            .with_minimum_scale(0.2)
            .unwrap();
        assert_eq!(optimizer.factor(), 0.5);
        assert_eq!(optimizer.patience(), 2);
        assert_eq!((optimizer.relative_tolerance(), optimizer.absolute_tolerance()), (0.0, 0.1));
        assert_eq!(optimizer.cooldown(), 1);
        assert_eq!(optimizer.accumulation_size(), 3);
        assert_eq!(optimizer.minimum_scale(), 0.2);
        assert!(matches!(
            ReduceOnPlateau::new().with_factor(1.0),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`factor` must be in `(0, 1)` but got `1`",
        ));
        assert!(matches!(
            ReduceOnPlateau::new().with_tolerances(0.0, 0.0),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "at least one of `relative_tolerance` and `absolute_tolerance` must be positive",
        ));
        assert!(matches!(
            ReduceOnPlateau::new().with_tolerances(2.0, 0.0),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`relative_tolerance` must be at most `1` but got `2`",
        ));
        assert!(matches!(
            ReduceOnPlateau::new().with_accumulation_size(0),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`accumulation_size` must be positive but got `0`",
        ));
    }

    #[test]
    fn test_reduce_on_plateau_initialize() {
        assert_eq!(
            state_paths(&ReduceOnPlateau::new()),
            vec!["$.scale", "$.best_value", "$.plateau_count", "$.cooldown_count", "$.count", "$.average_value"],
        );
    }

    #[test]
    fn test_reduce_on_plateau_update() {
        // With the patience `2` and the factor `0.5`, the value `1` improves on the infinite best value, the next two
        // values `1` do not improve on it (`1 < 0.9999 · 1` is false), so the scale halves at the third value. The
        // cooldown `1` then skips the plateau count of the fourth value, and the fifth and sixth values count again,
        // so the scale halves once more at the sixth value. The updates are the gradient `2` times the scale.
        let optimizer =
            ReduceOnPlateau::new().with_factor(0.5).unwrap().with_patience(2).unwrap().with_cooldown(1).unwrap();
        let (updates, state) = scales(&optimizer, &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        assert_eq!(updates, vec![2.0, 2.0, 1.0, 1.0, 1.0, 0.5]);
        assert_eq!(state.best_value(), &Array::scalar(1.0f32).unwrap());
        assert_eq!(state.cooldown_count(), &Array::scalar(1i32).unwrap());

        // An improving value resets the plateau count, and the minimum scale bounds the reduction.
        let optimizer = ReduceOnPlateau::new()
            .with_factor(0.5)
            .unwrap()
            .with_patience(1)
            .unwrap()
            .with_minimum_scale(0.75)
            .unwrap();
        let (updates, state) = scales(&optimizer, &[1.0, 0.5, 0.5, 0.5]);
        assert_eq!(updates, vec![2.0, 2.0, 1.5, 1.5]);
        assert_eq!(state.best_value(), &Array::scalar(0.5f32).unwrap());
    }

    #[test]
    fn test_reduce_on_plateau_update_accumulation() {
        // With the accumulation size `2`, the values are compared in pairs: the average `(4 + 2) / 2 = 3` improves on
        // the infinite best value, and the average `(3 + 5) / 2 = 4` does not, so with the patience `1` the scale
        // halves after the fourth value.
        let optimizer = ReduceOnPlateau::new()
            .with_factor(0.5)
            .unwrap()
            .with_patience(1)
            .unwrap()
            .with_accumulation_size(2)
            .unwrap();
        let (updates, state) = scales(&optimizer, &[4.0, 2.0, 3.0, 5.0]);
        assert_eq!(updates, vec![2.0, 2.0, 2.0, 1.0]);
        assert_eq!(state.best_value(), &Array::scalar(3.0f32).unwrap());
        assert_eq!(state.count(), &Array::scalar(0i32).unwrap());
        let (_, state) = scales(&optimizer, &[4.0]);
        assert_eq!(state.count(), &Array::scalar(1i32).unwrap());
        assert_eq!(state.average_value(), &Array::scalar(4.0f32).unwrap());
    }

    #[test]
    fn test_reduce_on_plateau_update_arguments() {
        let optimizer = ReduceOnPlateau::new();
        let parameters = vec![Array::scalar(0.0f32).unwrap()];
        let state = optimizer.initialize(&parameters).unwrap();
        assert!(matches!(
            optimizer.update(parameters.clone(), state, &parameters, &UpdateArguments::new()),
            Err(OptimizerError::MissingArgument { message })
                if message == "`ReduceOnPlateau` requires the `value` update argument, which was not provided",
        ));
    }

    #[test]
    fn test_reduce_on_plateau_optax_parity() {
        // Optax 9ecde04b: `optax.contrib.reduce_on_plateau(factor=0.5, patience=1, rtol=1e-4, atol=0.0)` with the
        // values `[1.0, 1.0, 0.5, 0.6]`. Each row holds the scale, the best value, the plateau count, the cooldown
        // count, and the average value of the state after a step.
        const DEFAULT: [[f32; 9]; 4] = [
            [0.1, -0.2, 0.3, 0.4, -0.1, 0.2, -0.3, 0.5, -0.6],
            [-0.025, 0.125, 0.05, 0.1, 0.15, -0.2, 0.05, -0.1, 0.175],
            [0.15, -0.05, -0.1, -0.075, 0.025, 0.225, 0.125, 0.2, -0.05],
            [0.05, 0.0375, -0.0875, 0.025, -0.075, 0.0125, -0.1125, 0.05, 0.075],
        ];
        const DEFAULT_STATES: [[f32; 5]; 4] = [
            [1.0, 1.0, 0.0, 0.0, 0.0],
            [0.5, 1.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0, 0.0],
            [0.25, 0.5, 0.0, 0.0, 0.0],
        ];
        let optimizer = ReduceOnPlateau::new()
            .with_factor(0.5)
            .unwrap()
            .with_patience(1)
            .unwrap()
            .with_tolerances(1e-4, 0.0)
            .unwrap();
        let (updates, states) = reduce_on_plateau_parity_updates(&optimizer, [1.0, 1.0, 0.5, 0.6]);
        assert_parity(&updates, &DEFAULT);
        assert_parity(&states, &DEFAULT_STATES);

        // Optax 9ecde04b: `optax.contrib.reduce_on_plateau(factor=0.5, patience=1, rtol=0.0, atol=0.1, cooldown=1,
        // accumulation_size=2, min_scale=0.6)` with the values `[1.0, 1.0, 1.0, 0.98]`. The average of the last two
        // values is not better than the best one by `0.1`, so the scale drops to the minimum scale.
        const VARIANT: [[f32; 9]; 4] = [
            [0.1, -0.2, 0.3, 0.4, -0.1, 0.2, -0.3, 0.5, -0.6],
            [-0.05, 0.25, 0.1, 0.2, 0.3, -0.4, 0.1, -0.2, 0.35],
            [0.3, -0.1, -0.2, -0.15, 0.05, 0.45, 0.25, 0.4, -0.1],
            [0.120000005, 0.09, -0.21000001, 0.060000002, -0.18, 0.030000001, -0.27, 0.120000005, 0.18],
        ];
        const VARIANT_STATES: [[f32; 5]; 4] = [
            [1.0, f32::INFINITY, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 1.0],
            [0.6, 1.0, 0.0, 1.0, 0.0],
        ];
        let optimizer = ReduceOnPlateau::new()
            .with_factor(0.5)
            .unwrap()
            .with_patience(1)
            .unwrap()
            .with_tolerances(0.0, 0.1)
            .unwrap()
            .with_cooldown(1)
            .unwrap()
            .with_accumulation_size(2)
            .unwrap()
            .with_minimum_scale(0.6)
            .unwrap();
        let (updates, states) = reduce_on_plateau_parity_updates(&optimizer, [1.0, 1.0, 1.0, 0.98]);
        assert_parity(&updates, &VARIANT);
        assert_parity(&states, &VARIANT_STATES);
    }

    /// Returns the updates of `optimizer` for the parity gradients (refer to [`parity_updates`](
    /// crate::tests::parity_updates)) with the objective values `values`, and the scale, the best value, the plateau
    /// count, the cooldown count, and the average value of the state after every step, as `f32`s.
    fn reduce_on_plateau_parity_updates(
        optimizer: &ReduceOnPlateau,
        values: [f32; 4],
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let scalar = |value: &Array| -> f32 {
            match value.r#type().data_type() {
                DataType::I32 => value.elements::<i32>().unwrap()[0] as f32,
                _ => value.elements::<f32>().unwrap()[0],
            }
        };
        let mut parameters = parity_parameters(&PARITY_PARAMETERS);
        let mut state = optimizer.initialize(&parameters).unwrap();
        let (mut updates, mut states) = (Vec::new(), Vec::new());
        for (gradients, value) in PARITY_GRADIENTS.iter().zip(values) {
            let value = Array::scalar(value).unwrap();
            let arguments = UpdateArguments::new().with_value(&value);
            let (step_updates, next_state) =
                optimizer.update(parity_parameters(gradients), state, &parameters, &arguments).unwrap();
            updates.push(parity_elements(&step_updates));
            states.push(vec![
                scalar(next_state.scale()),
                scalar(next_state.best_value()),
                scalar(next_state.plateau_count()),
                scalar(next_state.cooldown_count()),
                scalar(next_state.average_value()),
            ]);
            parameters = apply_updates(parameters, step_updates).unwrap();
            state = next_state;
        }
        (updates, states)
    }
}
