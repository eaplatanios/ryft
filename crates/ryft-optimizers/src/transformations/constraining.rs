//! Transformations that constrain the updates, by replacing NaNs or by keeping the parameters non-negative.
//!
//! [`ZeroNans`] replaces NaN elements of the updates with zeros and records which leaves had any, and
//! [`KeepParamsNonnegative`] changes the updates that would make parameters negative so that they make them zero.

use std::marker::PhantomData;

use ryft_core::{Add, ComparisonDirection, Neg, Parameter, Parameterized, ReductionKind};

use crate::macros::parameterized_state;
use crate::transformations::normalization::map_with_parameters;
use crate::{
    Optimizer, OptimizerError, OptimizerValue, UpdateArguments, all_axes, validate_structure, zero_count_like,
};

// TODO(eaplatanios): Review this module.

parameterized_state! {
    /// State of [`ZeroNans`], which holds a rank-0 boolean per leaf, with the structure `S` of the parameters, that
    /// records whether the updates of that leaf had NaN elements at the last update.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct ZeroNansState<P, S> with family ZeroNansStateFamily and marker leaf {
        /// Rank-0 booleans that record whether the last updates of each leaf had NaN elements, with the structure of
        /// the parameters.
        found_nan: S,
    }
}

impl<P: Parameter, S> ZeroNansState<P, S> {
    /// Returns the rank-0 booleans that record whether the last updates of each leaf had NaN elements.
    #[inline]
    pub fn found_nan(&self) -> &S {
        &self.found_nan
    }
}

/// Transformation that replaces the NaN elements of the updates with zeros, which is the analogue of
/// [`optax.zero_nans`](https://optax.readthedocs.io/en/latest/api/transformations.html#optax.zero_nans). Its state
/// records for every leaf whether its last updates had any NaN elements, which starts at `false`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ZeroNans;

impl ZeroNans {
    /// Creates a new [`ZeroNans`].
    #[inline]
    pub fn new() -> Self {
        Self
    }
}

impl<P: OptimizerValue> Optimizer<P> for ZeroNans {
    type State<S: Parameterized<P>> = ZeroNansState<P, S>;

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        // Comparing a zero with itself for inequality creates a rank-0 boolean `false` without another bound.
        let found_nan = parameters
            .parameters()
            .map(|parameter| {
                let zero = zero_count_like(parameter)?;
                Ok(zero.compare(&zero, ComparisonDirection::NotEqual)?)
            })
            .collect::<Result<Vec<_>, OptimizerError>>()?;
        Ok(ZeroNansState {
            found_nan: S::from_parameters(parameters.parameter_structure(), found_nan)?,
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
        // NaN is the only value that is not equal to itself.
        let _ = arguments;
        let paths = gradients.parameter_paths().collect::<Vec<_>>();
        validate_structure("parameters", &paths, parameters)?;
        validate_structure("found_nan", &paths, &state.found_nan)?;
        let structure = gradients.parameter_structure();
        let mut updates = Vec::with_capacity(paths.len());
        let mut found_nan = Vec::with_capacity(paths.len());
        for gradient in gradients.into_parameters() {
            let is_nan = gradient.compare(&gradient, ComparisonDirection::NotEqual)?;
            found_nan.push(if gradient.r#type().rank() == 0 {
                is_nan.clone()
            } else {
                is_nan.reduce(&all_axes(&is_nan), ReductionKind::Any)?
            });
            updates.push(is_nan.select(&gradient.fill_like(0f64)?, &gradient)?);
        }
        let state = ZeroNansState { found_nan: S::from_parameters(structure.clone(), found_nan)?, leaf: PhantomData };
        Ok((S::from_parameters(structure, updates)?, state))
    }
}

/// Transformation that changes every update that would make a parameter negative into the update that makes it zero,
/// which is the analogue of
/// [`optax.keep_params_nonnegative`](
/// https://optax.readthedocs.io/en/latest/api/transformations.html#optax.keep_params_nonnegative).
/// For an update `u` of a parameter `θ`, it returns `-θ` where `θ + u < 0` and `u` elsewhere. It does not make
/// negative parameters non-negative unless their updates do, so the parameters should start non-negative. This
/// transformation is stateless.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct KeepParamsNonnegative;

impl KeepParamsNonnegative {
    /// Creates a new [`KeepParamsNonnegative`].
    #[inline]
    pub fn new() -> Self {
        Self
    }
}

impl<P: OptimizerValue> Optimizer<P> for KeepParamsNonnegative {
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
        map_with_parameters(gradients, parameters, |update, parameter| {
            let negative =
                Add::add(parameter, update)?.compare(&parameter.fill_like(0f64)?, ComparisonDirection::LessThan)?;
            Ok(negative.select(&Neg::neg(parameter)?, update)?)
        })
        .map(|updates| (updates, state))
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use ryft_core::Array;

    use crate::apply_updates;
    use crate::tests::{
        PARITY_GRADIENTS, PARITY_PARAMETERS, assert_parity, parity_elements, parity_parameters, state_paths,
        vector_updates,
    };

    use super::*;

    #[test]
    fn test_zero_nans_initialize() {
        assert_eq!(ZeroNans::new(), ZeroNans);
        assert_eq!(state_paths(&ZeroNans::new()), vec!["$.found_nan[\"bias\"]", "$.found_nan[\"weights\"]"]);
        let parameters = vec![Array::vector(vec![1.0f32, 2.0]).unwrap()];
        assert_eq!(ZeroNans::new().initialize(&parameters).unwrap().found_nan(), &vec![Array::scalar(false).unwrap()]);
    }

    #[test]
    fn test_zero_nans_update() {
        let gradients = (Array::vector(vec![1.0f32, f32::NAN]).unwrap(), Array::scalar(2.0f32).unwrap());
        let zero_nans = ZeroNans::new();
        let state = zero_nans.initialize(&gradients).unwrap();
        let (updates, state) = zero_nans.update(gradients.clone(), state, &gradients, &UpdateArguments::new()).unwrap();
        assert_eq!(updates, (Array::vector(vec![1.0f32, 0.0]).unwrap(), Array::scalar(2.0f32).unwrap()));
        assert_eq!(state.found_nan(), &(Array::scalar(true).unwrap(), Array::scalar(false).unwrap()));
    }

    #[test]
    fn test_keep_params_nonnegative_update() {
        // The parameter `1` with the update `-3` would become `-2`, so the update becomes `-1`.
        assert_eq!(KeepParamsNonnegative::new(), KeepParamsNonnegative);
        assert_eq!(
            vector_updates(&KeepParamsNonnegative::new(), &[1.0, 2.0], &[&[-3.0, -1.0]]),
            vec![vec![-1.0, -1.0]],
        );
    }

    #[test]
    fn test_zero_nans_optax_parity() {
        // Optax 9ecde04b: `optax.zero_nans()`, with a NaN gradient at element 1 in the second step and at element 5 in
        // the
        // third step. The second constant holds the `found_nan` flags of the leaves `bias` and `weights` after every
        // step.
        const EXPECTED: [[f32; 9]; 4] = [
            [0.1, -0.2, 0.3, 0.4, -0.1, 0.2, -0.3, 0.5, -0.6],
            [-0.05, 0.0, 0.1, 0.2, 0.3, -0.4, 0.1, -0.2, 0.35],
            [0.3, -0.1, -0.2, -0.15, 0.05, 0.0, 0.25, 0.4, -0.1],
            [0.2, 0.15, -0.35, 0.1, -0.3, 0.05, -0.45, 0.2, 0.3],
        ];
        const EXPECTED_FOUND_NAN: [[f32; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]];
        let mut gradients = PARITY_GRADIENTS;
        gradients[1][1] = f32::NAN;
        gradients[2][5] = f32::NAN;
        let parameters = parity_parameters(&PARITY_PARAMETERS);
        let mut state = ZeroNans::new().initialize(&parameters).unwrap();
        let (mut updates, mut found_nan) = (Vec::new(), Vec::new());
        for gradients in &gradients {
            let (step_updates, next_state) = ZeroNans::new()
                .update(parity_parameters(gradients), state, &parameters, &UpdateArguments::new())
                .unwrap();
            updates.push(parity_elements(&step_updates));
            found_nan.push(
                next_state
                    .found_nan()
                    .values()
                    .map(|flag| if flag.elements::<bool>().unwrap()[0] { 1.0 } else { 0.0 })
                    .collect::<Vec<f32>>(),
            );
            state = next_state;
        }
        assert_parity(&updates, &EXPECTED);
        assert_parity(&found_nan, &EXPECTED_FOUND_NAN);
    }

    #[test]
    fn test_keep_params_nonnegative_optax_parity() {
        // Optax 9ecde04b: `optax.keep_params_nonnegative()`, on the absolute values of the parity parameters with ten
        // times
        // the parity gradients as updates, so that several updates would make parameters negative.
        const EXPECTED: [[f32; 9]; 4] = [
            [1.0, -1.0, 3.0, 4.0, -0.5, 2.0, -2.0, 5.0, -0.75],
            [-0.5, 2.5, 1.0, 2.0, 3.0, -2.25, 1.0, -2.0, 3.5],
            [3.0, -1.0, -2.0, -1.5, 0.5, 4.5, 2.5, 4.0, -1.0],
            [2.0, 1.5, -3.5, 1.0, -3.0, 0.5, -3.5, 2.0, 3.0],
        ];
        let mut parameters = parity_parameters(&PARITY_PARAMETERS.map(f32::abs));
        let mut updates = Vec::new();
        for gradients in &PARITY_GRADIENTS {
            let gradients = parity_parameters(&gradients.map(|gradient| 10.0 * gradient));
            let (step_updates, ()) =
                KeepParamsNonnegative::new().update(gradients, (), &parameters, &UpdateArguments::new()).unwrap();
            updates.push(parity_elements(&step_updates));
            parameters = apply_updates(parameters, step_updates).unwrap();
        }
        assert_parity(&updates, &EXPECTED);
    }
}
