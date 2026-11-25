//! Transformations that multiply the updates by a constant or by the value of a schedule, or that replace them with
//! zeros.

use ryft_core::{Mul, Neg, Parameter, Parameterized};

use crate::{
    Optimizer, OptimizerError, OptimizerValue, Schedule, UpdateArguments, convert_to, increment_count, initial_count,
    statistic_data_type, validate_finite, validate_structure,
};

// TODO(eaplatanios): Review this module.

/// Transformation that multiplies the updates by a constant factor, which is the analogue of
/// [`optax.scale`](https://optax.readthedocs.io/en/latest/api/transformations.html#optax.scale). A constant learning
/// rate `η` is applied as the factor `-η`, because updates are added to the parameters. This transformation is
/// stateless.
#[derive(Clone, Debug, PartialEq)]
pub struct Scale {
    /// Factor by which the updates are multiplied.
    factor: f64,
}

impl Scale {
    /// Creates a new [`Scale`] that multiplies the updates by `factor`.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `factor` is not finite.
    pub fn new(factor: f64) -> Result<Self, OptimizerError> {
        validate_finite("factor", factor)?;
        Ok(Self { factor })
    }

    /// Returns the factor by which this transformation multiplies the updates.
    #[inline]
    pub fn factor(&self) -> f64 {
        self.factor
    }
}

impl<P: OptimizerValue> Optimizer<P> for Scale {
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
            .map(|gradient| {
                let data_type = gradient.r#type().data_type();
                let gradient = convert_to(&gradient, statistic_data_type(data_type))?;
                Ok(convert_to(&Mul::mul(&gradient.fill_like(self.factor)?, &gradient)?, data_type)?)
            })
            .collect::<Result<Vec<_>, OptimizerError>>()?;
        Ok((S::from_parameters(structure, updates)?, state))
    }
}

/// Transformation that replaces the updates with zeros of the same shapes and data types, which is the analogue of
/// [`optax.set_to_zero`](https://optax.readthedocs.io/en/latest/api/transformations.html#optax.set_to_zero). Applied to
/// a subset of the leaves with [`Masked`](crate::Masked), it freezes the parameters of that subset (refer to
/// [`Freeze`](crate::Freeze)). This transformation is stateless.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SetToZero;

impl<P: OptimizerValue> Optimizer<P> for SetToZero {
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
            .map(|gradient| Ok(gradient.fill_like(0f64)?))
            .collect::<Result<Vec<_>, OptimizerError>>()?;
        Ok((S::from_parameters(structure, updates)?, state))
    }
}

/// State of [`ScaleBySchedule`], which holds the number of updates applied so far.
#[derive(Clone, Debug, PartialEq, Eq, Hash, ryft_macros::Parameterized)]
#[ryft(crate = "ryft_core")]
pub struct ScaleByScheduleState<P: Parameter> {
    /// Rank-0 `i32` array that holds the number of updates applied so far.
    count: P,
}

impl<P: Parameter> ScaleByScheduleState<P> {
    /// Returns the rank-0 `i32` array that holds the number of updates applied so far.
    #[inline]
    pub fn count(&self) -> &P {
        &self.count
    }
}

/// Transformation that multiplies the updates by the value of a [`Schedule`] at the number of updates applied before
/// the current one (i.e., at step `0` for the first update), which is the analogue of
/// [`optax.scale_by_schedule`](
/// https://optax.readthedocs.io/en/latest/api/transformations.html#optax.scale_by_schedule).
/// [`Self::learning_rate`] creates the variant that multiplies by the negated value, which is the analogue of
/// [`optax.scale_by_learning_rate`](
/// https://optax.readthedocs.io/en/latest/api/transformations.html#optax.scale_by_learning_rate)
/// and turns rescaled gradients into descent updates.
#[derive(Clone, Debug, PartialEq)]
pub struct ScaleBySchedule<Sc: Schedule> {
    /// [`Schedule`] whose values multiply the updates.
    schedule: Sc,

    /// Whether the updates are multiplied by the negated values of the schedule.
    negated: bool,
}

impl<Sc: Schedule> ScaleBySchedule<Sc> {
    /// Creates a new [`ScaleBySchedule`] that multiplies the updates by the values of `schedule`.
    #[inline]
    pub fn new(schedule: Sc) -> Self {
        Self { schedule, negated: false }
    }

    /// Creates a new [`ScaleBySchedule`] that multiplies the updates by the negated values of the learning rate
    /// schedule `learning_rate`, because updates are added to the parameters.
    #[inline]
    pub fn learning_rate(learning_rate: Sc) -> Self {
        Self { schedule: learning_rate, negated: true }
    }

    /// Returns the [`Schedule`] whose values multiply the updates of this transformation.
    #[inline]
    pub fn schedule(&self) -> &Sc {
        &self.schedule
    }

    /// Returns whether this transformation multiplies the updates by the negated values of its schedule.
    #[inline]
    pub fn negated(&self) -> bool {
        self.negated
    }
}

impl<P: OptimizerValue, Sc: Schedule> Optimizer<P> for ScaleBySchedule<Sc> {
    type State<S: Parameterized<P>> = ScaleByScheduleState<P>;

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        Ok(ScaleByScheduleState { count: initial_count("ScaleBySchedule", parameters)? })
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
        let value = self.schedule.value(&state.count)?;
        let value = if self.negated { Neg::neg(&value)? } else { value };
        let structure = gradients.parameter_structure();
        let updates = gradients
            .into_parameters()
            .map(|gradient| {
                let data_type = gradient.r#type().data_type();
                let statistic_type = statistic_data_type(data_type);
                let gradient = convert_to(&gradient, statistic_type)?;
                Ok(convert_to(&Mul::mul(&convert_to(&value, statistic_type)?, &gradient)?, data_type)?)
            })
            .collect::<Result<Vec<_>, OptimizerError>>()?;
        let count = increment_count(&state.count)?;
        Ok((S::from_parameters(structure, updates)?, ScaleByScheduleState { count }))
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use ryft_core::{Array, bf16};

    use crate::LinearSchedule;
    use crate::tests::{assert_parity, parity_updates};

    use super::*;

    #[test]
    fn test_scale_new() {
        assert_eq!(Scale::new(-0.5).unwrap().factor(), -0.5);
        assert!(matches!(
            Scale::new(f64::NAN),
            Err(OptimizerError::InvalidConfiguration { message }) if message == "`factor` must be finite but got `NaN`",
        ));
    }

    #[test]
    fn test_scale_update() {
        let gradients = (Array::vector(vec![1.0f32, -2.0]).unwrap(), Array::scalar(bf16::from_f32(3.0)).unwrap());
        assert_eq!(
            Scale::new(-0.5)
                .unwrap()
                .update(gradients.clone(), (), &gradients, &UpdateArguments::new())
                .unwrap(),
            ((Array::vector(vec![-0.5f32, 1.0]).unwrap(), Array::scalar(bf16::from_f32(-1.5)).unwrap()), ()),
        );
    }

    #[test]
    fn test_set_to_zero_update() {
        let gradients = (Array::vector(vec![1.0f32, f32::NAN]).unwrap(), Array::scalar(bf16::from_f32(3.0)).unwrap());
        assert_eq!(
            SetToZero.update(gradients.clone(), (), &gradients, &UpdateArguments::new()).unwrap(),
            ((Array::vector(vec![0f32, 0.0]).unwrap(), Array::scalar(bf16::from_f32(0.0)).unwrap()), ()),
        );
    }

    #[test]
    fn test_scale_by_schedule_new() {
        let scale = ScaleBySchedule::new(0.5f64);
        assert_eq!(scale.schedule(), &0.5);
        assert!(!scale.negated());
    }

    #[test]
    fn test_scale_by_schedule_learning_rate() {
        let scale = ScaleBySchedule::learning_rate(0.5f64);
        assert_eq!(scale.schedule(), &0.5);
        assert!(scale.negated());
    }

    #[test]
    fn test_scale_by_schedule_initialize() {
        let parameters = vec![Array::vector(vec![1.0f64, 2.0]).unwrap()];
        let state = ScaleBySchedule::new(0.5f64).initialize(&parameters).unwrap();
        assert_eq!(state.count(), &Array::scalar(0i32).unwrap());
    }

    #[test]
    fn test_scale_by_schedule_update() {
        // The schedule is evaluated at the step count before the update, which starts at `0`.
        let schedule = LinearSchedule::new(1.0, 0.5, 2).unwrap();
        let gradients = vec![Array::vector(vec![2.0f32, -4.0]).unwrap()];
        let arguments = UpdateArguments::new();
        let scale = ScaleBySchedule::new(schedule.clone());
        let state = scale.initialize(&gradients).unwrap();
        let (updates, state) = scale.update(gradients.clone(), state, &gradients, &arguments).unwrap();
        assert_eq!(updates, vec![Array::vector(vec![2.0f32, -4.0]).unwrap()]);
        let (updates, state) = scale.update(gradients.clone(), state, &gradients, &arguments).unwrap();
        assert_eq!(updates, vec![Array::vector(vec![1.5f32, -3.0]).unwrap()]);
        assert_eq!(state.count(), &Array::scalar(2i32).unwrap());

        // Learning rates are negated.
        let scale = ScaleBySchedule::learning_rate(schedule);
        let state = scale.initialize(&gradients).unwrap();
        let (updates, _) = scale.update(gradients.clone(), state, &gradients, &arguments).unwrap();
        assert_eq!(updates, vec![Array::vector(vec![-2.0f32, 4.0]).unwrap()]);
    }

    #[test]
    fn test_scale_optax_parity() {
        // Optax 9ecde04b: `optax.scale(-0.5)`.
        const EXPECTED: [[f32; 9]; 4] = [
            [-0.05, 0.1, -0.15, -0.2, 0.05, -0.1, 0.15, -0.25, 0.3],
            [0.025, -0.125, -0.05, -0.1, -0.15, 0.2, -0.05, 0.1, -0.175],
            [-0.15, 0.05, 0.1, 0.075, -0.025, -0.225, -0.125, -0.2, 0.05],
            [-0.1, -0.075, 0.175, -0.05, 0.15, -0.025, 0.225, -0.1, -0.15],
        ];
        assert_parity(&parity_updates(&Scale::new(-0.5).unwrap()), &EXPECTED);
    }

    #[test]
    fn test_set_to_zero_optax_parity() {
        // Optax 9ecde04b: `optax.set_to_zero()`.
        const EXPECTED: [[f32; 9]; 4] = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        assert_parity(&parity_updates(&SetToZero), &EXPECTED);
    }

    #[test]
    fn test_scale_by_schedule_optax_parity() {
        // Optax 9ecde04b: `optax.scale_by_schedule(optax.linear_schedule(1.0, 0.25, 3))`.
        const SCALE_BY_SCHEDULE: [[f32; 9]; 4] = [
            [0.1, -0.2, 0.3, 0.4, -0.1, 0.2, -0.3, 0.5, -0.6],
            [-0.0375, 0.1875, 0.075, 0.15, 0.22500001, -0.3, 0.075, -0.15, 0.2625],
            [0.15, -0.05, -0.1, -0.075, 0.025, 0.225, 0.125, 0.2, -0.05],
            [0.05, 0.0375, -0.0875, 0.025, -0.075, 0.0125, -0.1125, 0.05, 0.075],
        ];
        // Optax 9ecde04b: `optax.scale_by_learning_rate(optax.linear_schedule(1.0, 0.25, 3))`.
        const SCALE_BY_LEARNING_RATE: [[f32; 9]; 4] = [
            [-0.1, 0.2, -0.3, -0.4, 0.1, -0.2, 0.3, -0.5, 0.6],
            [0.0375, -0.1875, -0.075, -0.15, -0.22500001, 0.3, -0.075, 0.15, -0.2625],
            [-0.15, 0.05, 0.1, 0.075, -0.025, -0.225, -0.125, -0.2, 0.05],
            [-0.05, -0.0375, 0.0875, -0.025, 0.075, -0.0125, 0.1125, -0.05, -0.075],
        ];
        let schedule = LinearSchedule::new(1.0, 0.25, 3).unwrap();
        assert_parity(&parity_updates(&ScaleBySchedule::new(schedule.clone())), &SCALE_BY_SCHEDULE);
        assert_parity(&parity_updates(&ScaleBySchedule::learning_rate(schedule)), &SCALE_BY_LEARNING_RATE);
    }
}
