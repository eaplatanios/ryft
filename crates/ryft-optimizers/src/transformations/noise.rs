//! Gradient noise, which adds annealed Gaussian noise to the updates.

use ryft_core::{Add, Concatenate, DataType, Div, Mul, Parameter, Parameterized, Random};

use crate::{
    Optimizer, OptimizerError, OptimizerValue, UpdateArguments, convert_to, increment_count, initial_count,
    statistic_data_type, validate_finite, validate_non_negative, validate_structure,
};

// TODO(eaplatanios): Review this module.

/// State of [`AddNoise`], which holds the number of updates applied so far and the random generator state from which
/// the noise of the next update is drawn.
#[derive(Clone, Debug, PartialEq, Eq, Hash, ryft_macros::Parameterized)]
#[ryft(crate = "ryft_core")]
pub struct AddNoiseState<P: Parameter> {
    /// Rank-0 `i32` array that holds the number of updates applied so far.
    count: P,

    /// `u64[2]` ThreeFry generator state (i.e., `[key, counter]`) from which the next noise is drawn.
    random_state: P,
}

impl<P: Parameter> AddNoiseState<P> {
    /// Returns the rank-0 `i32` array that holds the number of updates applied so far.
    #[inline]
    pub fn count(&self) -> &P {
        &self.count
    }

    /// Returns the `u64[2]` ThreeFry generator state from which the next noise is drawn.
    #[inline]
    pub fn random_state(&self) -> &P {
        &self.random_state
    }
}

/// Transformation that adds annealed Gaussian noise to the updates, which is the gradient noise of
/// [Neelakantan et al. (2015)](https://arxiv.org/abs/1511.06807) and the analogue of
/// [`optax.add_noise`](https://optax.readthedocs.io/en/latest/api/transformations.html#optax.add_noise). At the step
/// count `t` (i.e., the number of updates including the current one), it returns `g + σ · n` with the standard
/// deviation `σ = √(η / tᵞ)` and standard normal noise `n` of the shape of every leaf.
///
/// The noise is drawn with the random operations of `ryft-core` from a ThreeFry generator state that the state of this
/// transformation threads, which starts at `[seed, 0]`. As in Optax, every update splits the generator state into the
/// state of the next update and a sampling state, which is split again into one state per leaf. The noise is therefore
/// deterministic for a seed, but it is not bit-identical with the noise that JAX draws for the same seed, because the
/// key derivation of JAX differs. The noise is drawn in the statistic data type of every leaf (i.e., `f32`, or `f64`
/// for `f64` leaves). This transformation requires leaf values with the [`Random`] capability.
#[derive(Clone, Debug, PartialEq)]
pub struct AddNoise {
    /// Scale `η` of the variance of the noise.
    eta: f64,

    /// Decay exponent `γ` of the variance of the noise.
    gamma: f64,

    /// Seed of the random generator state.
    seed: u64,
}

impl AddNoise {
    /// Creates a new [`AddNoise`] with the variance scale `eta`, the decay exponent `gamma`, and the generator seed
    /// `seed`.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `eta` is negative or not finite or when `gamma` is not
    /// finite.
    pub fn new(eta: f64, gamma: f64, seed: u64) -> Result<Self, OptimizerError> {
        validate_non_negative("eta", eta)?;
        validate_finite("gamma", gamma)?;
        Ok(Self { eta, gamma, seed })
    }

    /// Returns the scale `η` of the variance of the noise of this transformation.
    #[inline]
    pub fn eta(&self) -> f64 {
        self.eta
    }

    /// Returns the decay exponent `γ` of the variance of the noise of this transformation.
    #[inline]
    pub fn gamma(&self) -> f64 {
        self.gamma
    }

    /// Returns the seed of the random generator state of this transformation.
    #[inline]
    pub fn seed(&self) -> u64 {
        self.seed
    }
}

impl<P: OptimizerValue + Random> Optimizer<P> for AddNoise {
    type State<S: Parameterized<P>> = AddNoiseState<P>;

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        // The generator state `[seed, 0]` is assembled from rank-0 fills of the zero count, so that it is created in
        // the domain of the parameters without another bound.
        let count = initial_count("AddNoise", parameters)?;
        let zero = convert_to(&count, DataType::U64)?.reshape([1])?;
        let random_state = Concatenate::concatenate([&zero.fill_like(self.seed)?, &zero], 0)?;
        Ok(AddNoiseState { count, random_state })
    }

    fn update<S: Parameterized<P>>(
        &self,
        gradients: S,
        state: Self::State<S>,
        parameters: &S,
        arguments: &UpdateArguments<'_, P, S>,
    ) -> Result<(S, Self::State<S>), OptimizerError> {
        // This follows the operation order of `optax.add_noise`.
        let _ = arguments;
        let paths = gradients.parameter_paths().collect::<Vec<_>>();
        validate_structure("parameters", &paths, parameters)?;
        let count = increment_count(&state.count)?;
        let step = convert_to(&count, DataType::F32)?;
        let standard_deviation =
            Div::div(&step.fill_like(self.eta)?, &step.pow(&step.fill_like(self.gamma)?)?)?.sqrt()?;
        let (_, mut keys) = state.random_state.split_rng_key(2)?;
        let sampling_state = keys.remove(1);
        let random_state = keys.remove(0);
        let (_, leaf_states) = sampling_state.split_rng_key(paths.len())?;
        let structure = gradients.parameter_structure();
        let updates = gradients
            .into_parameters()
            .zip(leaf_states)
            .map(|(gradient, leaf_state)| {
                let data_type = gradient.r#type().data_type();
                let statistic_type = statistic_data_type(data_type);
                let gradient = convert_to(&gradient, statistic_type)?;
                let noise_type = gradient.r#type().into_owned().with_layout(None);
                let (_, noise) = leaf_state.random_normal(&noise_type)?;
                let scaled = Mul::mul(&convert_to(&standard_deviation, statistic_type)?, &noise)?;
                Ok(convert_to(&Add::add(&gradient, &scaled)?, data_type)?)
            })
            .collect::<Result<Vec<_>, OptimizerError>>()?;
        let state = AddNoiseState { count, random_state };
        Ok((S::from_parameters(structure, updates)?, state))
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use ryft_core::{Array, ArrayType, RandomAlgorithm};

    use super::*;

    #[test]
    fn test_add_noise_new() {
        let noise = AddNoise::new(0.01, 0.55, 7).unwrap();
        assert_eq!((noise.eta(), noise.gamma(), noise.seed()), (0.01, 0.55, 7));
        assert!(matches!(
            AddNoise::new(-1.0, 0.55, 7),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`eta` must be finite and non-negative but got `-1`",
        ));
    }

    #[test]
    fn test_add_noise_initialize() {
        let parameters = vec![Array::vector(vec![1.0f32, 2.0]).unwrap()];
        let state = AddNoise::new(0.01, 0.55, 7).unwrap().initialize(&parameters).unwrap();
        assert_eq!(state.count(), &Array::scalar(0i32).unwrap());
        assert_eq!(
            state.random_state(),
            &Array::from_elements(RandomAlgorithm::ThreeFry.state_type(), &[7u64, 0]).unwrap(),
        );
        assert_eq!(
            state.named_parameters().map(|(path, _)| path.to_string()).collect::<Vec<_>>(),
            vec!["$.count", "$.random_state"],
        );
    }

    #[test]
    fn test_add_noise_update() {
        // At the step count `t`, the noise is drawn from the second of two states split from the generator state, split
        // again into one state per leaf, and scaled by `√(η / tᵞ)`, while the first split state is kept.
        let noise = AddNoise::new(4.0, 2.0, 7).unwrap();
        let gradients = vec![Array::vector(vec![1.0f32, -1.0]).unwrap()];
        let state = noise.initialize(&gradients).unwrap();
        let (updates, next_state) =
            noise.update(gradients.clone(), state.clone(), &gradients, &UpdateArguments::new()).unwrap();
        let (_, keys) = state.random_state().split_rng_key(2).unwrap();
        let (_, leaf_keys) = keys[1].split_rng_key(1).unwrap();
        let (_, samples) = leaf_keys[0].random_normal(&ArrayType::new_static(DataType::F32, [2])).unwrap();
        let samples = samples.elements::<f32>().unwrap();
        let standard_deviation = (4.0f32 / 1.0f32.powf(2.0)).sqrt();
        assert_eq!(
            updates,
            vec![
                Array::vector(vec![1.0 + standard_deviation * samples[0], -1.0 + standard_deviation * samples[1]])
                    .unwrap()
            ],
        );
        assert_eq!(next_state.random_state(), &keys[0]);
        assert_eq!(next_state.count(), &Array::scalar(1i32).unwrap());

        // The second update draws different noise from the advanced state, and a zero variance adds no noise.
        let (second, _) = noise.update(gradients.clone(), next_state, &gradients, &UpdateArguments::new()).unwrap();
        assert_ne!(second, updates);
        let silent = AddNoise::new(0.0, 2.0, 7).unwrap();
        let state = silent.initialize(&gradients).unwrap();
        assert_eq!(silent.update(gradients.clone(), state, &gradients, &UpdateArguments::new()).unwrap().0, gradients);
    }
}
