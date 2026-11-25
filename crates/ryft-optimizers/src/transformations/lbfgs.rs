//! Limited memory BFGS preconditioning, which multiplies the updates by an approximation of the inverse Hessian that is
//! built from the differences of the last parameters and updates.

use std::iter::once;

use ryft_core::{
    Add, ComparisonDirection, Concatenate, DataType, Dimension, Div, Mul, Neg, Parameter, Parameterized, Shape, Sub,
};

use crate::macros::parameterized_state;
use crate::{
    Optimizer, OptimizerError, OptimizerValue, UpdateArguments, convert_to, filled_statistics, increment_count,
    initial_count, inner_product, statistic_data_type, validate_structure,
};

// TODO(eaplatanios): Review this module.

parameterized_state! {
    /// State of [`ScaleByLbfgs`], which is the analogue of [`optax.ScaleByLBFGSState`](
    /// https://optax.readthedocs.io/en/latest/api/transformations.html#optax.ScaleByLBFGSState). It holds the
    /// parameters and the updates of the last update and the memory of their differences, with the structure `S` of the
    /// parameters. Every leaf of a memory has the shape of its parameter with a leading history dimension whose extent
    /// is the memory size, ordered from the oldest difference to the newest one. The leaves are stored in the statistic
    /// data type of each parameter, and the weights in `f32`, or in `f64` when any parameter is `f64`.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct ScaleByLbfgsState<P, S> with family ScaleByLbfgsStateFamily {
        /// Rank-0 `i32` array that holds the number of updates applied so far.
        count: P,

        /// Parameters of the last update, which are zero before the first update.
        parameters: S,

        /// Updates (i.e., gradients) of the last update, which are zero before the first update.
        updates: S,

        /// Last differences of consecutive parameters, with a leading history dimension per leaf.
        parameter_difference_memory: S,

        /// Last differences of consecutive updates, with a leading history dimension per leaf.
        update_difference_memory: S,

        /// Vector of the weights `ρ = 1 / ⟨δu, δw⟩` of the differences in the memories, which are zero for differences
        /// that are orthogonal or not yet recorded.
        weight_memory: P,
    }
}

impl<P: Parameter, S> ScaleByLbfgsState<P, S> {
    /// Returns the rank-0 `i32` array that holds the number of updates applied so far.
    #[inline]
    pub fn count(&self) -> &P {
        &self.count
    }

    /// Returns the parameters of the last update.
    #[inline]
    pub fn parameters(&self) -> &S {
        &self.parameters
    }

    /// Returns the updates of the last update.
    #[inline]
    pub fn updates(&self) -> &S {
        &self.updates
    }

    /// Returns the last differences of consecutive parameters, with a leading history dimension per leaf.
    #[inline]
    pub fn parameter_difference_memory(&self) -> &S {
        &self.parameter_difference_memory
    }

    /// Returns the last differences of consecutive updates, with a leading history dimension per leaf.
    #[inline]
    pub fn update_difference_memory(&self) -> &S {
        &self.update_difference_memory
    }

    /// Returns the vector of the weights of the differences in the memories.
    #[inline]
    pub fn weight_memory(&self) -> &P {
        &self.weight_memory
    }
}

/// Transformation that multiplies the updates by the limited memory BFGS approximation of the inverse Hessian, which is
/// the analogue of
/// [`optax.scale_by_lbfgs`](https://optax.readthedocs.io/en/latest/api/transformations.html#optax.scale_by_lbfgs).
/// With the parameters `wₖ` and the updates (i.e., gradients) `uₖ` of update `k`, it records the differences
/// `δwₖ = wₖ - wₖ₋₁` and `δuₖ = uₖ - uₖ₋₁` and the weights `ρₖ = 1 / ⟨δuₖ, δwₖ⟩` of the last `m` updates (i.e., the
/// memory size), and returns the product `Pₖ uₖ` of the approximation `Pₖ` and the updates, which it computes with the
/// two loop recursion of [Nocedal and Wright (2006)](https://doi.org/10.1007/978-0-387-40065-5) (Algorithm 7.4)
/// starting from the scaled identity `γₖ I`. With the scaled initial preconditioner, `γₖ = ⟨δuₖ, δwₖ⟩ / ‖δuₖ‖²`, and
/// at the first update, `γ₀ = min(1, 1 / ‖u₀‖)`. Without it, `γₖ = 1`.
///
/// The result is a descent direction only once it is negated, so this transformation is followed by a negative
/// scaling and usually by a line search (refer to [`Lbfgs`](crate::Lbfgs)). The memory is a shift register (i.e., the
/// newest difference is appended and the oldest one is dropped), where Optax writes into a circular buffer, which
/// gives the same products. The default memory size is `10`, with the scaled initial preconditioner.
#[derive(Clone, Debug, PartialEq)]
pub struct ScaleByLbfgs {
    /// Number `m` of past differences that are kept in memory.
    memory_size: usize,

    /// Whether the initial approximation of the inverse Hessian is the scaled identity `γₖ I` rather than `I`.
    scale_initial_preconditioner: bool,
}

impl ScaleByLbfgs {
    /// Creates a new [`ScaleByLbfgs`] with the memory size `10` and the scaled initial preconditioner.
    #[inline]
    pub fn new() -> Self {
        Self { memory_size: 10, scale_initial_preconditioner: true }
    }

    /// Returns this transformation with `memory_size` as the number of past differences that it keeps in memory.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `memory_size` is zero.
    pub fn with_memory_size(mut self, memory_size: usize) -> Result<Self, OptimizerError> {
        if memory_size == 0 {
            return Err(OptimizerError::InvalidConfiguration {
                message: "`memory_size` must be positive but got `0`".to_string(),
            });
        }
        self.memory_size = memory_size;
        Ok(self)
    }

    /// Returns this transformation with the scaled identity as its initial preconditioner when
    /// `scale_initial_preconditioner` is `true` (i.e., the `scale_init_precond` of Optax) and the identity otherwise.
    #[inline]
    pub fn with_scale_initial_preconditioner(mut self, scale_initial_preconditioner: bool) -> Self {
        self.scale_initial_preconditioner = scale_initial_preconditioner;
        self
    }

    /// Returns the number of past differences that this transformation keeps in memory.
    #[inline]
    pub fn memory_size(&self) -> usize {
        self.memory_size
    }

    /// Returns whether the initial preconditioner of this transformation is the scaled identity.
    #[inline]
    pub fn scale_initial_preconditioner(&self) -> bool {
        self.scale_initial_preconditioner
    }

    /// Returns `memory` (whose leading dimension has the memory size as its extent) with its oldest entry dropped and
    /// `entry` (which has the shape of one entry) appended as its newest entry.
    fn push<P: OptimizerValue>(&self, memory: &P, entry: &P) -> Result<P, OptimizerError> {
        let entry = stacked(&convert_to(entry, memory.r#type().data_type())?)?;
        if self.memory_size == 1 {
            return Ok(entry);
        }
        let kept = memory.slice_axis(0, 1, self.memory_size, 1)?;
        Ok(Concatenate::concatenate([&kept, &entry], 0)?)
    }
}

impl Default for ScaleByLbfgs {
    fn default() -> Self {
        Self::new()
    }
}

impl<P: OptimizerValue> Optimizer<P> for ScaleByLbfgs {
    type State<S: Parameterized<P>> = ScaleByLbfgsState<P, S>;

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        let count = initial_count("ScaleByLbfgs", parameters)?;
        let structure = parameters.parameter_structure();
        let zeros = filled_statistics(parameters, 0.0, None)?.into_parameters().collect::<Vec<_>>();
        let memory = zeros
            .iter()
            .map(|zero| Ok(Concatenate::concatenate(std::iter::repeat_n(&stacked(zero)?, self.memory_size), 0)?))
            .collect::<Result<Vec<_>, OptimizerError>>()?;
        let weight = convert_to(&count, weight_data_type(&zeros))?.fill_like(0f64)?.reshape([1])?;
        Ok(ScaleByLbfgsState {
            count,
            parameters: S::from_parameters(structure.clone(), zeros.clone())?,
            updates: S::from_parameters(structure.clone(), zeros)?,
            parameter_difference_memory: S::from_parameters(structure.clone(), memory.clone())?,
            update_difference_memory: S::from_parameters(structure, memory)?,
            weight_memory: Concatenate::concatenate(std::iter::repeat_n(&weight, self.memory_size), 0)?,
        })
    }

    fn update<S: Parameterized<P>>(
        &self,
        gradients: S,
        state: Self::State<S>,
        parameters: &S,
        arguments: &UpdateArguments<'_, P, S>,
    ) -> Result<(S, Self::State<S>), OptimizerError> {
        // This follows the operation order of `optax.scale_by_lbfgs`, with the circular buffer replaced by a shift
        // register whose last element is the newest difference.
        let _ = arguments;
        let paths = gradients.parameter_paths().collect::<Vec<_>>();
        validate_structure("parameters", &paths, parameters)?;
        validate_structure("parameters", &paths, &state.parameters)?;
        validate_structure("updates", &paths, &state.updates)?;
        validate_structure("parameter_difference_memory", &paths, &state.parameter_difference_memory)?;
        validate_structure("update_difference_memory", &paths, &state.update_difference_memory)?;
        let data_types = gradients.parameters().map(|gradient| gradient.r#type().data_type()).collect::<Vec<_>>();
        let updates = gradients
            .parameters()
            .map(|gradient| convert_to(gradient, statistic_data_type(gradient.r#type().data_type())))
            .collect::<Result<Vec<_>, _>>()?;
        let current_parameters = parameters
            .parameters()
            .zip(&updates)
            .map(|(parameter, update)| convert_to(parameter, update.r#type().data_type()))
            .collect::<Result<Vec<_>, _>>()?;
        let weight_type = state.weight_memory.r#type().data_type();
        let started = state.count.compare(&state.count.fill_like(0i32)?, ComparisonDirection::GreaterThan)?;

        // 1. Record the differences of the parameters and of the updates and their weight, which are zero at the first
        //    update.
        let mut parameter_differences = Vec::with_capacity(paths.len());
        let mut update_differences = Vec::with_capacity(paths.len());
        for (((parameter, update), previous_parameter), previous_update) in current_parameters
            .iter()
            .zip(&updates)
            .zip(state.parameters.parameters())
            .zip(state.updates.parameters())
        {
            let parameter_difference =
                Sub::sub(parameter, &convert_to(previous_parameter, parameter.r#type().data_type())?)?;
            let update_difference = Sub::sub(update, &convert_to(previous_update, update.r#type().data_type())?)?;
            parameter_differences.push(started.select(&parameter_difference, &parameter_difference.fill_like(0f64)?)?);
            update_differences.push(started.select(&update_difference, &update_difference.fill_like(0f64)?)?);
        }
        let curvature =
            convert_to(&inner_product("curvature", &update_differences, &parameter_differences)?, weight_type)?;
        let zero = curvature.fill_like(0f64)?;
        let weight = curvature
            .compare(&zero, ComparisonDirection::Equal)?
            .select(&zero, &Div::div(&curvature.fill_like(1f64)?, &curvature)?)?;
        let parameter_difference_memory = state
            .parameter_difference_memory
            .parameters()
            .zip(&parameter_differences)
            .map(|(memory, difference)| self.push(memory, difference))
            .collect::<Result<Vec<_>, _>>()?;
        let update_difference_memory = state
            .update_difference_memory
            .parameters()
            .zip(&update_differences)
            .map(|(memory, difference)| self.push(memory, difference))
            .collect::<Result<Vec<_>, _>>()?;
        let weight_memory = self.push(&state.weight_memory, &weight)?;

        // 2. Compute the scaling `γₖ` of the identity that initializes the approximation of the inverse Hessian.
        let identity_scale = if self.scale_initial_preconditioner {
            let squared_norm =
                convert_to(&inner_product("norm", &update_differences, &update_differences)?, weight_type)?;
            let scale = squared_norm
                .compare(&zero, ComparisonDirection::GreaterThan)?
                .select(&Div::div(&curvature, &squared_norm)?, &zero.fill_like(1f64)?)?;
            let update_norm = convert_to(&inner_product("norm", &updates, &updates)?, weight_type)?.sqrt()?;
            let capped_inverse_norm =
                Div::div(&update_norm.fill_like(1f64)?, &update_norm)?.min(&update_norm.fill_like(1f64)?)?;
            started.select(&scale, &capped_inverse_norm)?
        } else {
            zero.fill_like(1f64)?
        };

        // 3. Compute the product of the approximation of the inverse Hessian and the updates with the two loop
        //    recursion, from the newest difference to the oldest one and back.
        let differences = |memory: &[P], index: usize| -> Result<Vec<P>, OptimizerError> {
            memory.iter().map(|memory| entry(memory, index)).collect()
        };
        let mut preconditioned = updates.clone();
        let mut alphas = vec![None; self.memory_size];
        for index in (0..self.memory_size).rev() {
            let parameter_difference = differences(&parameter_difference_memory, index)?;
            let update_difference = differences(&update_difference_memory, index)?;
            let weight = entry(&weight_memory, index)?;
            let alpha = Mul::mul(
                &weight,
                &convert_to(&inner_product("alpha", &parameter_difference, &preconditioned)?, weight_type)?,
            )?;
            preconditioned = add_scaled(&preconditioned, &Neg::neg(&alpha)?, &update_difference)?;
            alphas[index] = Some(alpha);
        }
        preconditioned = preconditioned
            .iter()
            .map(|leaf| Ok(Mul::mul(&convert_to(&identity_scale, leaf.r#type().data_type())?, leaf)?))
            .collect::<Result<Vec<_>, OptimizerError>>()?;
        for (index, alpha) in alphas.into_iter().enumerate() {
            let parameter_difference = differences(&parameter_difference_memory, index)?;
            let update_difference = differences(&update_difference_memory, index)?;
            let weight = entry(&weight_memory, index)?;
            let beta = Mul::mul(
                &weight,
                &convert_to(&inner_product("beta", &update_difference, &preconditioned)?, weight_type)?,
            )?;
            preconditioned = add_scaled(&preconditioned, &Sub::sub(&alpha.unwrap(), &beta)?, &parameter_difference)?;
        }

        let structure = gradients.parameter_structure();
        let preconditioned = preconditioned
            .into_iter()
            .zip(data_types)
            .map(|(leaf, data_type)| convert_to(&leaf, data_type))
            .collect::<Result<Vec<_>, _>>()?;

        // The state keeps the data types of the leaves that it replaces, as `initialize` created them from the
        // parameters.
        let stored = |leaves: Vec<P>, previous: &S| {
            leaves
                .into_iter()
                .zip(previous.parameters())
                .map(|(leaf, previous)| convert_to(&leaf, previous.r#type().data_type()))
                .collect::<Result<Vec<_>, _>>()
        };
        let current_parameters = stored(current_parameters, &state.parameters)?;
        let updates = stored(updates, &state.updates)?;
        let state = ScaleByLbfgsState {
            count: increment_count(&state.count)?,
            parameters: S::from_parameters(structure.clone(), current_parameters)?,
            updates: S::from_parameters(structure.clone(), updates)?,
            parameter_difference_memory: S::from_parameters(structure.clone(), parameter_difference_memory)?,
            update_difference_memory: S::from_parameters(structure.clone(), update_difference_memory)?,
            weight_memory,
        };
        Ok((S::from_parameters(structure, preconditioned)?, state))
    }
}

/// Returns `value` reshaped to have a leading dimension with extent `1`.
fn stacked<P: OptimizerValue>(value: &P) -> Result<P, OptimizerError> {
    let r#type = value.r#type();
    let dimensions = once(Dimension::Static(1)).chain(r#type.shape().dimensions().iter().cloned()).collect();
    Ok(value.reshape(Shape::new(dimensions))?)
}

/// Returns the entry at `index` along the leading dimension of `memory`, without that dimension.
fn entry<P: OptimizerValue>(memory: &P, index: usize) -> Result<P, OptimizerError> {
    let r#type = memory.r#type();
    let shape = Shape::new(r#type.shape().dimensions()[1..].to_vec());
    Ok(memory.slice_axis(0, index, index + 1, 1)?.reshape(shape)?)
}

/// Returns `leaves + scale · directions` leaf by leaf, for the rank-0 `scale`.
fn add_scaled<P: OptimizerValue>(leaves: &[P], scale: &P, directions: &[P]) -> Result<Vec<P>, OptimizerError> {
    leaves
        .iter()
        .zip(directions)
        .map(|(leaf, direction)| {
            let data_type = leaf.r#type().data_type();
            let direction = convert_to(direction, data_type)?;
            Ok(Add::add(leaf, &Mul::mul(&convert_to(scale, data_type)?, &direction)?)?)
        })
        .collect()
}

/// Returns the data type of the weights of the memories of `leaves`, which is `f64` when any leaf is `f64` and `f32`
/// otherwise, as for the inner products that the weights are computed from.
fn weight_data_type<P: OptimizerValue>(leaves: &[P]) -> DataType {
    if leaves.iter().any(|leaf| leaf.r#type().data_type() == DataType::F64) { DataType::F64 } else { DataType::F32 }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use ryft_core::Array;

    use crate::tests::{assert_parity, parity_quadratic_updates, state_paths};
    use crate::{Chain, Scale};

    use super::*;

    /// Returns the `f32` elements of the parameters after every one of `steps` updates `w = w - P u` from the vector
    /// `initial`, where `u` is the gradient of the separable quadratic `Σᵢ ½ aᵢ wᵢ²` with the curvatures `curvatures`.
    fn quadratic_iterates(lbfgs: &ScaleByLbfgs, initial: &[f32], curvatures: &[f32], steps: usize) -> Vec<Vec<f32>> {
        let curvatures = Array::vector(curvatures.to_vec()).unwrap();
        let mut parameters = vec![Array::vector(initial.to_vec()).unwrap()];
        let mut state = lbfgs.initialize(&parameters).unwrap();
        let mut iterates = Vec::new();
        for _ in 0..steps {
            let gradients = vec![Mul::mul(&curvatures, &parameters[0]).unwrap()];
            let (updates, next_state) = lbfgs.update(gradients, state, &parameters, &UpdateArguments::new()).unwrap();
            parameters = vec![Sub::sub(&parameters[0], &updates[0]).unwrap()];
            iterates.push(parameters[0].elements::<f32>().unwrap());
            state = next_state;
        }
        iterates
    }

    #[test]
    fn test_scale_by_lbfgs_new() {
        let lbfgs = ScaleByLbfgs::new();
        assert_eq!(lbfgs.memory_size(), 10);
        assert!(lbfgs.scale_initial_preconditioner());
        assert_eq!(ScaleByLbfgs::default(), lbfgs);
    }

    #[test]
    fn test_scale_by_lbfgs_with_memory_size() {
        assert_eq!(ScaleByLbfgs::new().with_memory_size(3).unwrap().memory_size(), 3);
        assert!(matches!(
            ScaleByLbfgs::new().with_memory_size(0),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`memory_size` must be positive but got `0`",
        ));
    }

    #[test]
    fn test_scale_by_lbfgs_with_scale_initial_preconditioner() {
        // Without scaling, the first update is the gradient itself: `w = 0.1 - 4 · 0.1 = -0.3`.
        let lbfgs = ScaleByLbfgs::new().with_scale_initial_preconditioner(false);
        assert!(!lbfgs.scale_initial_preconditioner());
        assert_eq!(quadratic_iterates(&lbfgs, &[0.1], &[4.0], 1), vec![vec![0.1f32 - 4.0 * 0.1]]);
    }

    #[test]
    fn test_scale_by_lbfgs_initialize() {
        assert_eq!(
            state_paths(&ScaleByLbfgs::new()),
            vec![
                "$.count",
                "$.parameters[\"bias\"]",
                "$.parameters[\"weights\"]",
                "$.updates[\"bias\"]",
                "$.updates[\"weights\"]",
                "$.parameter_difference_memory[\"bias\"]",
                "$.parameter_difference_memory[\"weights\"]",
                "$.update_difference_memory[\"bias\"]",
                "$.update_difference_memory[\"weights\"]",
                "$.weight_memory",
            ],
        );
        let state = ScaleByLbfgs::new()
            .with_memory_size(3)
            .unwrap()
            .initialize(&vec![Array::vector(vec![1.0f32, 2.0]).unwrap()])
            .unwrap();
        assert_eq!(state.parameter_difference_memory()[0], Array::matrix(3, 2, vec![0.0f32; 6]).unwrap());
        assert_eq!(state.weight_memory(), &Array::vector(vec![0.0f32; 3]).unwrap());
    }

    #[test]
    fn test_scale_by_lbfgs_update() {
        // On `f(w) = ½ · 4 · w²` from `w = 0.1`, the first update is `min(1, 1 / 0.4) · 0.4 = 0.4`, so `w = -0.3`. The
        // second update uses the pair `δw = -0.4` and `δu = -1.6`, with `ρ = 1 / 0.64` and `γ = 0.64 / 2.56 = 1 / 4`,
        // for which the two loop recursion returns the Newton step `u / 4 = w`, so `w = 0` exactly.
        let lbfgs = ScaleByLbfgs::new();
        let iterates = quadratic_iterates(&lbfgs, &[0.1], &[4.0], 2);
        assert!((iterates[0][0] + 0.3).abs() < 1e-6);
        assert!(iterates[1][0].abs() < 1e-6);

        // On `f(w) = ½ · (w₁² + 2 · w₂²)` from `w = (1, 1)`, the first update is `u / ‖u‖ = (1, 2) / √5`, so
        // `w = (0.55279, 0.10557)`. The second update uses `δw = (-0.44721, -0.89443)`, `δu = (-0.44721, -1.78885)`,
        // `ρ = 1 / 1.8`, and `γ = 1.8 / 3.4`. The first loop gives `α = ρ · ⟨δw, u⟩ = -0.24225` and
        // `v = γ · (u - α · δu) = (0.23530, -0.11766)`, and the second one gives `β = ρ · ⟨δu, v⟩ = 0.05847` and
        // `v + (α - β) · δw = (0.36978, 0.15131)`, so `w = (0.18301, -0.04575)`, whatever the memory size is.
        for memory_size in [1, 5] {
            let lbfgs = ScaleByLbfgs::new().with_memory_size(memory_size).unwrap();
            let iterates = quadratic_iterates(&lbfgs, &[1.0, 1.0], &[1.0, 2.0], 2);
            assert!((iterates[0][0] - 0.55279).abs() < 1e-5 && (iterates[0][1] - 0.10557).abs() < 1e-5);
            assert!((iterates[1][0] - 0.18301).abs() < 1e-5 && (iterates[1][1] + 0.04575).abs() < 1e-5);
        }
    }

    #[test]
    fn test_scale_by_lbfgs_optax_parity() {
        // Optax 9ecde04b: `optax.chain(optax.scale_by_lbfgs(memory_size=3), optax.scale(-0.1))`, on the parity
        // quadratic.
        const SCALE_BY_LBFGS: [[f32; 9]; 5] = [
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
            [0.036013864, 0.20157018, -0.08627221, -0.0, 0.1080417, 0.068861336, 0.30235562, -0.049468886, 0.021568146],
            [0.03913537, 0.1758203, -0.092676185, -0.0, 0.11740615, 0.065763354, 0.26373047, -0.043458644, 0.023169016],
            [0.03860979, 0.15870826, -0.07738377, -0.0, 0.11582937, 0.060231682, 0.23806217, -0.039662324, 0.019345937],
            [0.036424335, 0.141905, -0.07143996, -0.0, 0.10927295, 0.054276194, 0.21285756, -0.035207745, 0.017859958],
        ];
        // Optax 9ecde04b: `optax.chain(optax.scale_by_lbfgs(memory_size=3, scale_init_precond=False),
        // optax.scale(-0.1))`,
        // on the parity quadratic.
        const SCALE_BY_LBFGS_IDENTITY: [[f32; 9]; 5] = [
            [0.05, 0.4, -0.3, -0.0, 0.15, 0.112500004, 0.6, -0.125, 0.075],
            [
                0.047017634,
                0.18636994,
                0.002550888,
                -0.0,
                0.14105293,
                0.0791031,
                0.27955496,
                -0.028588852,
                -0.00063772337,
            ],
            [0.046976067, 0.13838315, -0.17232063, -0.0, 0.1409282, 0.060748786, 0.20757468, -0.04467864, 0.04308016],
            [
                0.039196193,
                0.15743689,
                0.0014202237,
                -0.0,
                0.117588595,
                0.053009685,
                0.23615544,
                -0.039248496,
                -0.0003550544,
            ],
            [
                0.030433595,
                0.113068365,
                -0.09420384,
                -0.0,
                0.09130078,
                0.046439055,
                0.16960232,
                -0.018099068,
                0.02355096,
            ],
        ];
        let lbfgs = ScaleByLbfgs::new().with_memory_size(3).unwrap();
        let optimizer = Chain::new(lbfgs.clone(), Scale::new(-0.1).unwrap());
        assert_parity(&parity_quadratic_updates(&optimizer, 5), &SCALE_BY_LBFGS);
        let optimizer = Chain::new(lbfgs.with_scale_initial_preconditioner(false), Scale::new(-0.1).unwrap());
        assert_parity(&parity_quadratic_updates(&optimizer, 5), &SCALE_BY_LBFGS_IDENTITY);
    }
}
