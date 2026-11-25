//! Transformations that bound the magnitude of the updates.
//!
//! [`ClipByGlobalNorm`] bounds the global norm of all updates, [`Clip`] bounds every element, [`ClipByBlockRms`] bounds
//! the root mean square of every leaf, and [`AdaptiveGradClip`] bounds the norms of the units of every leaf relative
//! to the norms of the units of its parameter. All of these transformations are stateless.

use ryft_core::{Add, ComparisonDirection, DataType, Div, Mul, Parameterized, ReductionKind};

use crate::transformations::normalization::map_with_parameters;
use crate::{
    Optimizer, OptimizerError, OptimizerValue, UpdateArguments, all_axes, convert_to, validate_non_negative,
    validate_positive, validate_structure,
};

// TODO(eaplatanios): Review this module.

/// Transformation that rescales the updates so that their global norm is at most a maximum norm, which is the analogue
/// of [`optax.clip_by_global_norm`](
/// https://optax.readthedocs.io/en/latest/api/transformations.html#optax.clip_by_global_norm).
/// The global norm `‖g‖` is the square root of the sum of the squares of all elements of all leaves, accumulated in
/// `f32`, or in `f64` when any leaf is `f64`. Updates whose global norm is less than the maximum norm `c` are returned
/// unchanged, and all others are scaled as `(g / ‖g‖) · c`, so that they keep their direction and have norm `c`. A
/// zero global norm returns the updates unchanged and never produces NaNs. This transformation is stateless.
#[derive(Clone, Debug, PartialEq)]
pub struct ClipByGlobalNorm {
    /// Maximum global norm of the updates, which is always positive and finite.
    maximum_norm: f64,
}

impl ClipByGlobalNorm {
    /// Creates a new [`ClipByGlobalNorm`] that bounds the global norm of the updates by `maximum_norm`.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `maximum_norm` is not positive and finite.
    pub fn new(maximum_norm: f64) -> Result<Self, OptimizerError> {
        if !maximum_norm.is_finite() || maximum_norm <= 0.0 {
            return Err(OptimizerError::InvalidConfiguration {
                message: format!("`maximum_norm` must be positive and finite but got `{maximum_norm}`"),
            });
        }
        Ok(Self { maximum_norm })
    }

    /// Returns the maximum global norm of the updates of this transformation.
    #[inline]
    pub fn maximum_norm(&self) -> f64 {
        self.maximum_norm
    }
}

impl<P: OptimizerValue> Optimizer<P> for ClipByGlobalNorm {
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
        let gradients = gradients.into_parameters().collect::<Vec<_>>();
        let statistic_type = if gradients.iter().any(|gradient| gradient.r#type().data_type() == DataType::F64) {
            DataType::F64
        } else {
            DataType::F32
        };
        let mut squared_norm = None;
        for gradient in &gradients {
            let gradient = convert_to(gradient, statistic_type)?;
            let leaf_squared_norm = Mul::mul(&gradient, &gradient)?.reduce(&all_axes(&gradient), ReductionKind::Sum)?;
            squared_norm = Some(match squared_norm {
                None => leaf_squared_norm,
                Some(squared_norm) => Add::add(&squared_norm, &leaf_squared_norm)?,
            });
        }
        let Some(squared_norm) = squared_norm else {
            return Ok((S::from_parameters(structure, gradients)?, state));
        };

        // The scaled branch divides by the larger of the global norm and the maximum norm, which equals the global norm
        // wherever that branch is selected, so that a zero global norm never divides zero by zero.
        let norm = squared_norm.sqrt()?;
        let maximum_norm = norm.fill_like(self.maximum_norm)?;
        let unclipped = norm.compare(&maximum_norm, ComparisonDirection::LessThan)?;
        let denominator = norm.max(&maximum_norm)?;
        let updates = gradients
            .into_iter()
            .map(|gradient| {
                let data_type = gradient.r#type().data_type();
                let gradient = convert_to(&gradient, statistic_type)?;
                let clipped = Mul::mul(&Div::div(&gradient, &denominator)?, &maximum_norm)?;
                Ok(convert_to(&unclipped.select(&gradient, &clipped)?, data_type)?)
            })
            .collect::<Result<Vec<_>, OptimizerError>>()?;
        Ok((S::from_parameters(structure, updates)?, state))
    }
}

/// Transformation that clips every element of the updates to `[-δ, δ]` for a maximum magnitude `δ`, which is the
/// analogue of [`optax.clip`](https://optax.readthedocs.io/en/latest/api/transformations.html#optax.clip).
#[derive(Clone, Debug, PartialEq)]
pub struct Clip {
    /// Maximum magnitude `δ` of every element of the updates.
    maximum_delta: f64,
}

impl Clip {
    /// Creates a new [`Clip`] that clips every element of the updates to `[-maximum_delta, maximum_delta]`.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `maximum_delta` is negative or not finite.
    pub fn new(maximum_delta: f64) -> Result<Self, OptimizerError> {
        validate_non_negative("maximum_delta", maximum_delta)?;
        Ok(Self { maximum_delta })
    }

    /// Returns the maximum magnitude of every element of the updates of this transformation.
    #[inline]
    pub fn maximum_delta(&self) -> f64 {
        self.maximum_delta
    }
}

impl<P: OptimizerValue> Optimizer<P> for Clip {
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
        // This is `jnp.clip`, which takes the maximum with the lower bound before the minimum with the upper bound.
        let _ = arguments;
        map_with_parameters(gradients, parameters, |update, _| {
            Ok(update.max(&update.fill_like(-self.maximum_delta)?)?.min(&update.fill_like(self.maximum_delta)?)?)
        })
        .map(|updates| (updates, state))
    }
}

/// Transformation that rescales every leaf whose root mean square exceeds a threshold to have that root mean square,
/// which is the analogue of
/// [`optax.clip_by_block_rms`](
/// https://optax.readthedocs.io/en/latest/api/transformations.html#optax.clip_by_block_rms).
/// For a leaf `u` with the root mean square `rms(u)` and the threshold `τ`, it returns `u / max(1, rms(u) / τ)`.
#[derive(Clone, Debug, PartialEq)]
pub struct ClipByBlockRms {
    /// Largest root mean square `τ` of every leaf.
    threshold: f64,
}

impl ClipByBlockRms {
    /// Creates a new [`ClipByBlockRms`] that bounds the root mean square of every leaf by `threshold`.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `threshold` is not positive and finite.
    pub fn new(threshold: f64) -> Result<Self, OptimizerError> {
        validate_positive("threshold", threshold)?;
        Ok(Self { threshold })
    }

    /// Returns the largest root mean square of every leaf of the updates of this transformation.
    #[inline]
    pub fn threshold(&self) -> f64 {
        self.threshold
    }
}

impl<P: OptimizerValue> Optimizer<P> for ClipByBlockRms {
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
        // This follows the operation order of `optax.clip_by_block_rms`.
        let _ = arguments;
        map_with_parameters(gradients, parameters, |update, _| {
            let root_mean_square = Mul::mul(update, update)?.reduce(&all_axes(update), ReductionKind::Mean)?.sqrt()?;
            let denominator = Div::div(&root_mean_square, &root_mean_square.fill_like(self.threshold)?)?
                .max(&root_mean_square.fill_like(1f64)?)?;
            Ok(Div::div(update, &denominator)?)
        })
        .map(|updates| (updates, state))
    }
}

/// Transformation that clips the norms of the units of every leaf relative to the norms of the units of its parameter,
/// which is the adaptive gradient clipping of [Brock et al. (2021)](https://arxiv.org/abs/2102.06171) and the analogue
/// of [`optax.adaptive_grad_clip`](
/// https://optax.readthedocs.io/en/latest/api/transformations.html#optax.adaptive_grad_clip).
/// The unit norms of a leaf are the norms over the reduction axes, broadcast back to the shape of the leaf. Unless the
/// axes are set with [`Self::with_axes`], they are all axes for leaves with at most one axis of an extent other than
/// `1`, the first axis for leaves of rank two or three (i.e., linear layers), the first three axes for leaves of rank
/// four (i.e., convolution kernels), and the first four axes for leaves of rank five. For the unit norms `‖g‖` of an
/// update `g` and `‖θ‖` of its parameter, the clipping `λ`, and `ε`, the maximum norm is `m = λ · max(‖θ‖, ε)`, and the
/// update is `g` where `‖g‖ < m` and `g · (m / max(‖g‖, 1e-6))` elsewhere. The default `ε` is `1e-3`.
#[derive(Clone, Debug, PartialEq)]
pub struct AdaptiveGradClip {
    /// Largest ratio `λ` of the unit norms of the updates to those of the parameters.
    clipping: f64,

    /// Smallest unit norm `ε` of the parameters.
    epsilon: f64,

    /// Axes over which the unit norms are computed, or [`None`] for the axes that depend on the rank of each leaf.
    axes: Option<Vec<usize>>,
}

impl AdaptiveGradClip {
    /// Creates a new [`AdaptiveGradClip`] with the clipping ratio `clipping` and the default `ε` and axes.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `clipping` is negative or not finite.
    pub fn new(clipping: f64) -> Result<Self, OptimizerError> {
        validate_non_negative("clipping", clipping)?;
        Ok(Self { clipping, epsilon: 1e-3, axes: None })
    }

    /// Returns this transformation with `epsilon` as the smallest unit norm of the parameters.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `epsilon` is negative or not finite.
    pub fn with_epsilon(mut self, epsilon: f64) -> Result<Self, OptimizerError> {
        validate_non_negative("epsilon", epsilon)?;
        self.epsilon = epsilon;
        Ok(self)
    }

    /// Returns this transformation with `axes` as the axes over which the unit norms of every leaf are computed.
    #[inline]
    pub fn with_axes(mut self, axes: Vec<usize>) -> Self {
        self.axes = Some(axes);
        self
    }

    /// Returns the largest ratio of the unit norms of the updates to those of the parameters of this transformation.
    #[inline]
    pub fn clipping(&self) -> f64 {
        self.clipping
    }

    /// Returns the smallest unit norm of the parameters of this transformation.
    #[inline]
    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    /// Returns the axes over which this transformation computes unit norms, or [`None`] if they depend on the rank of
    /// each leaf.
    #[inline]
    pub fn axes(&self) -> Option<&[usize]> {
        self.axes.as_deref()
    }

    /// Returns the unit norms of `value` broadcast to its shape, which is `optax.unitwise_norm`.
    fn unit_norms<P: OptimizerValue>(&self, value: &P) -> Result<P, OptimizerError> {
        let r#type = value.r#type();
        let rank = r#type.rank();
        let axes = match &self.axes {
            Some(axes) => axes.clone(),
            None => {
                let non_unit_axes = (0..rank).filter(|&axis| r#type.dimension(axis).value() != Some(1)).count();
                match rank {
                    _ if non_unit_axes <= 1 => (0..rank).collect(),
                    2 | 3 => vec![0],
                    4 => vec![0, 1, 2],
                    5 => vec![0, 1, 2, 3],
                    _ => {
                        return Err(OptimizerError::InvalidConfiguration {
                            message: format!(
                                "`AdaptiveGradClip` requires leaves of rank at most `5` unless its axes are set, but \
                                 got a leaf of rank `{rank}`",
                            ),
                        });
                    }
                }
            }
        };
        let kept_axes = (0..rank).filter(|axis| !axes.contains(axis)).collect::<Vec<_>>();
        let norms = Mul::mul(value, value)?.reduce(&axes, ReductionKind::Sum)?.sqrt()?;
        Ok(norms.broadcast_like(value, &kept_axes)?)
    }
}

impl<P: OptimizerValue> Optimizer<P> for AdaptiveGradClip {
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
        // This follows the operation order of `optax.adaptive_grad_clip` and `optax.unitwise_clip`.
        let _ = arguments;
        map_with_parameters(gradients, parameters, |update, parameter| {
            let update_norms = self.unit_norms(update)?;
            let parameter_norms = self.unit_norms(parameter)?;
            let maximum_norms = Mul::mul(
                &parameter_norms.fill_like(self.clipping)?,
                &parameter_norms.max(&parameter_norms.fill_like(self.epsilon)?)?,
            )?;
            let clipped =
                Mul::mul(update, &Div::div(&maximum_norms, &update_norms.max(&update_norms.fill_like(1e-6f64)?)?)?)?;
            let unclipped = update_norms.compare(&maximum_norms, ComparisonDirection::LessThan)?;
            Ok(unclipped.select(update, &clipped)?)
        })
        .map(|updates| (updates, state))
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use ryft_core::{Array, ArrayType, bf16};

    use crate::tests::{assert_parity, parity_updates, vector_updates};

    use super::*;

    #[test]
    fn test_clip_by_global_norm_new() {
        assert_eq!(ClipByGlobalNorm::new(2.5).unwrap().maximum_norm(), 2.5);
        assert!(matches!(
            ClipByGlobalNorm::new(0.0),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`maximum_norm` must be positive and finite but got `0`",
        ));
        assert!(matches!(
            ClipByGlobalNorm::new(f64::INFINITY),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`maximum_norm` must be positive and finite but got `inf`",
        ));
    }

    #[test]
    fn test_clip_by_global_norm_update() {
        // The global norm of these gradients is `5`.
        let gradients = (Array::vector(vec![3.0f32, 0.0]).unwrap(), Array::scalar(bf16::from_f32(4.0)).unwrap());
        let arguments = UpdateArguments::new();

        // Gradients below the maximum norm are returned unchanged.
        let clip = ClipByGlobalNorm::new(10.0).unwrap();
        let (updates, ()) = clip.update(gradients.clone(), (), &gradients, &arguments).unwrap();
        assert_eq!(updates, gradients);

        // Gradients above the maximum norm are rescaled to it.
        let clip = ClipByGlobalNorm::new(2.5).unwrap();
        let (updates, ()) = clip.update(gradients.clone(), (), &gradients, &arguments).unwrap();
        assert_eq!(
            updates,
            (
                Array::vector(vec![3.0f32 / 5.0 * 2.5, 0.0]).unwrap(),
                Array::scalar(bf16::from_f32(4.0f32 / 5.0 * 2.5)).unwrap(),
            ),
        );
    }

    #[test]
    fn test_clip_by_global_norm_update_zero_norm() {
        let gradients = (Array::vector(vec![0f32, 0.0]).unwrap(), Array::scalar(0f32).unwrap());
        let clip = ClipByGlobalNorm::new(1.0).unwrap();
        let (updates, ()) = clip.update(gradients.clone(), (), &gradients, &UpdateArguments::new()).unwrap();
        assert_eq!(updates, gradients);
    }

    #[test]
    fn test_clip_by_global_norm_update_f64() {
        // A single `f64` leaf makes the global norm accumulate in `f64`, and every update keeps its data type.
        let gradients = (Array::scalar(3.0f32).unwrap(), Array::scalar(4.0f64).unwrap());
        let clip = ClipByGlobalNorm::new(1.0).unwrap();
        let (updates, ()) = clip.update(gradients.clone(), (), &gradients, &UpdateArguments::new()).unwrap();
        assert_eq!(updates, (Array::scalar((3.0f64 / 5.0) as f32).unwrap(), Array::scalar(4.0f64 / 5.0).unwrap()));
    }

    #[test]
    fn test_clip_new() {
        assert_eq!(Clip::new(0.5).unwrap().maximum_delta(), 0.5);
        assert!(matches!(
            Clip::new(-0.5),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`maximum_delta` must be finite and non-negative but got `-0.5`",
        ));
    }

    #[test]
    fn test_clip_update() {
        assert_eq!(
            vector_updates(&Clip::new(0.5).unwrap(), &[0.0, 0.0, 0.0], &[&[2.0, -1.0, 0.25]]),
            vec![vec![0.5, -0.5, 0.25,]],
        );
    }

    #[test]
    fn test_clip_by_block_rms_new() {
        assert_eq!(ClipByBlockRms::new(0.5).unwrap().threshold(), 0.5);
        assert!(matches!(
            ClipByBlockRms::new(0.0),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`threshold` must be positive and finite but got `0`",
        ));
    }

    #[test]
    fn test_clip_by_block_rms_update() {
        // The root mean square of `[3, 4]` is `√12.5`, so with the threshold `1` the leaf is divided by `√12.5`, while
        // with the threshold `10` it is divided by `max(1, √12.5 / 10) = 1`.
        let gradients: [&[f32]; 1] = [&[3.0, 4.0]];
        assert_eq!(
            vector_updates(&ClipByBlockRms::new(1.0).unwrap(), &[0.0, 0.0], &gradients),
            vec![vec![3.0 / (12.5f32.sqrt() / 1.0), 4.0 / (12.5f32.sqrt() / 1.0),]],
        );
        assert_eq!(vector_updates(&ClipByBlockRms::new(10.0).unwrap(), &[0.0, 0.0], &gradients), vec![vec![3.0, 4.0]]);
    }

    #[test]
    fn test_adaptive_grad_clip_new() {
        let clip = AdaptiveGradClip::new(0.5).unwrap();
        assert_eq!((clip.clipping(), clip.epsilon(), clip.axes()), (0.5, 1e-3, None));
        let clip = clip.with_epsilon(0.0).unwrap().with_axes(vec![1]);
        assert_eq!((clip.epsilon(), clip.axes()), (0.0, Some(&[1usize][..])));
        assert!(matches!(
            AdaptiveGradClip::new(f64::NAN),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`clipping` must be finite and non-negative but got `NaN`",
        ));
    }

    #[test]
    fn test_adaptive_grad_clip_update() {
        // For a vector, the unit norms are the norms of the whole leaf: the parameter `[3, 4]` has the norm `5`, so
        // with `λ = 0.5` the maximum norm is `2.5`. The update `[0, 2]` stays, and the update `[6, 8]` of norm `10` is
        // scaled by `2.5 / 10`.
        let clip = AdaptiveGradClip::new(0.5).unwrap();
        assert_eq!(
            vector_updates(&clip, &[3.0, 4.0], &[&[0.0, 2.0], &[6.0, 8.0]]),
            vec![vec![0.0, 2.0], vec![6.0 * (2.5 / 10.0), 8.0 * (2.5 / 10.0)],],
        );

        // For a matrix, the unit norms are the norms of its columns, so only the second column is clipped, to
        // `0.5 · max(1, 1e-3) = 0.5`.
        let parameters = vec![Array::matrix(2, 2, vec![3.0f32, 1.0, 4.0, 0.0]).unwrap()];
        let gradients = vec![Array::matrix(2, 2, vec![0.6f32, 2.0, 0.8, 0.0]).unwrap()];
        assert_eq!(
            clip.update(gradients, (), &parameters, &UpdateArguments::new()),
            Ok((vec![Array::matrix(2, 2, vec![0.6f32, 2.0 * (0.5 / 2.0), 0.8, 0.0]).unwrap()], ())),
        );

        // Leaves of rank six need explicit axes.
        let parameters =
            vec![Array::from_elements(ArrayType::new_static(DataType::F32, [2, 1, 1, 1, 1, 2]), &[1.0f32; 4]).unwrap()];
        assert!(matches!(
            clip.update(parameters.clone(), (), &parameters, &UpdateArguments::new()),
            Err(OptimizerError::InvalidConfiguration { message })
                if message
                    == "`AdaptiveGradClip` requires leaves of rank at most `5` unless its axes are set, but got a leaf \
                        of rank `6`",
        ));
    }

    #[test]
    fn test_clip_optax_parity() {
        // Optax 9ecde04b: `optax.clip(0.25)`.
        const EXPECTED: [[f32; 9]; 4] = [
            [0.1, -0.2, 0.25, 0.25, -0.1, 0.2, -0.25, 0.25, -0.25],
            [-0.05, 0.25, 0.1, 0.2, 0.25, -0.25, 0.1, -0.2, 0.25],
            [0.25, -0.1, -0.2, -0.15, 0.05, 0.25, 0.25, 0.25, -0.1],
            [0.2, 0.15, -0.25, 0.1, -0.25, 0.05, -0.25, 0.2, 0.25],
        ];
        assert_parity(&parity_updates(&Clip::new(0.25).unwrap()), &EXPECTED);
    }

    #[test]
    fn test_clip_by_global_norm_optax_parity() {
        // Optax 9ecde04b: `optax.clip_by_global_norm(0.5)`, whose threshold is below the global norm of every step.
        const EXPECTED: [[f32; 9]; 4] = [
            [0.048795, -0.09759, 0.146385, 0.19518, -0.048795, 0.09759, -0.146385, 0.243975, -0.29277],
            [
                -0.034099717,
                0.17049858,
                0.06819943,
                0.13639887,
                0.20459831,
                -0.27279773,
                0.06819943,
                -0.13639887,
                0.238698,
            ],
            [
                0.19364919,
                -0.06454973,
                -0.12909946,
                -0.096824594,
                0.032274865,
                0.29047376,
                0.16137432,
                0.25819892,
                -0.06454973,
            ],
            [
                0.12700012,
                0.0952501,
                -0.22225022,
                0.06350006,
                -0.1905002,
                0.03175003,
                -0.28575027,
                0.12700012,
                0.1905002,
            ],
        ];
        assert_parity(&parity_updates(&ClipByGlobalNorm::new(0.5).unwrap()), &EXPECTED);
    }

    #[test]
    fn test_clip_by_block_rms_optax_parity() {
        // Optax 9ecde04b: `optax.clip_by_block_rms(0.2)`.
        const EXPECTED: [[f32; 9]; 4] = [
            [0.092582, -0.185164, 0.27774602, 0.20542105, -0.05135526, 0.10271052, -0.15406579, 0.2567763, -0.30813158],
            [-0.05, 0.25, 0.1, 0.14407206, 0.2161081, -0.2881441, 0.07203603, -0.14407206, 0.2521261],
            [
                0.27774605,
                -0.09258202,
                -0.18516403,
                -0.108347274,
                0.036115754,
                0.3250418,
                0.18057878,
                0.28892604,
                -0.07223151,
            ],
            [
                0.16107745,
                0.1208081,
                -0.28188553,
                0.07427814,
                -0.22283442,
                0.03713907,
                -0.3342516,
                0.14855628,
                0.22283442,
            ],
        ];
        assert_parity(&parity_updates(&ClipByBlockRms::new(0.2).unwrap()), &EXPECTED);
    }

    #[test]
    fn test_adaptive_grad_clip_optax_parity() {
        // Optax 9ecde04b: `optax.adaptive_grad_clip(0.1, eps=1e-3)`.
        const EXPECTED: [[f32; 9]; 4] = [
            [0.061237242, -0.122474484, 0.18371172, 0.17888544, -0.031008685, 0.025, -0.1341641, 0.15504342, -0.075],
            [-0.045983817, 0.22991908, 0.091967635, 0.2, 0.14462219, -0.05485298, 0.1, -0.09641479, 0.047996357],
            [
                0.20029928,
                -0.066766426,
                -0.13353285,
                -0.1264357,
                0.01991759,
                0.0737773,
                0.21072614,
                0.15934072,
                -0.016394956,
            ],
            [
                0.11409912,
                0.085574344,
                -0.19967346,
                0.047988024,
                -0.14615972,
                0.01258138,
                -0.2159461,
                0.09743981,
                0.075488284,
            ],
        ];
        assert_parity(&parity_updates(&AdaptiveGradClip::new(0.1).unwrap().with_epsilon(1e-3).unwrap()), &EXPECTED);
    }
}
