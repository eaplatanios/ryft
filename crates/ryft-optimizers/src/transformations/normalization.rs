//! Transformations that rescale the updates by norms of the updates or of the parameters, which are the layer-wise
//! adaptations of LAMB, LARS, Fromage, and Adafactor, together with the centering of gradients.
//!
//! [`ScaleByTrustRatio`] scales every leaf by the ratio of the norms of its parameter and its update.
//! [`ScaleByParamBlockNorm`] and [`ScaleByParamBlockRms`] scale every leaf by the norm or the root mean square of its
//! parameter. [`NormalizeByUpdateNorm`] divides all updates by their global norm, and [`Centralize`] subtracts the
//! mean over all axes but the first from every leaf of rank two or more. All of these transformations are stateless.

use ryft_core::{Add, ComparisonDirection, DataType, Div, Mul, Parameterized, ReductionKind, Sub};

use crate::{
    Optimizer, OptimizerError, OptimizerValue, UpdateArguments, all_axes, convert_to, statistic_data_type,
    validate_non_negative, validate_positive, validate_structure,
};

// TODO(eaplatanios): Review this module.

/// Transformation that scales the update of every leaf by the ratio of the norms of its parameter and its update, which
/// is the layer-wise adaptation of LAMB of [You et al. (2020)](https://arxiv.org/abs/1904.00962) and the analogue of
/// [`optax.scale_by_trust_ratio`](
/// https://optax.readthedocs.io/en/latest/api/transformations.html#optax.scale_by_trust_ratio).
/// For an update `u` and its parameter `θ`, with the norms `‖θ‖` and `‖u‖` clamped from below at a minimum norm, it
/// returns `u · c · ‖θ‖ / (‖u‖ + ε)` with the trust coefficient `c`, or `u` itself when either norm is zero. The
/// defaults are a minimum norm of `0`, `c = 1`, and `ε = 0`.
#[derive(Clone, Debug, PartialEq)]
pub struct ScaleByTrustRatio {
    /// Smallest norm, at which the norms of the parameters and updates are clamped from below.
    minimum_norm: f64,

    /// Trust coefficient `c` that multiplies the ratio of the norms.
    trust_coefficient: f64,

    /// Term `ε` that is added to the norms of the updates.
    epsilon: f64,
}

impl ScaleByTrustRatio {
    /// Creates a new [`ScaleByTrustRatio`] with the default hyperparameters.
    #[inline]
    pub fn new() -> Self {
        Self { minimum_norm: 0.0, trust_coefficient: 1.0, epsilon: 0.0 }
    }

    /// Returns this transformation with `minimum_norm` as the norm at which it clamps the norms from below.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `minimum_norm` is negative or not finite.
    pub fn with_minimum_norm(mut self, minimum_norm: f64) -> Result<Self, OptimizerError> {
        validate_non_negative("minimum_norm", minimum_norm)?;
        self.minimum_norm = minimum_norm;
        Ok(self)
    }

    /// Returns this transformation with `trust_coefficient` as the coefficient that multiplies the ratio of the norms.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `trust_coefficient` is negative or not finite.
    pub fn with_trust_coefficient(mut self, trust_coefficient: f64) -> Result<Self, OptimizerError> {
        validate_non_negative("trust_coefficient", trust_coefficient)?;
        self.trust_coefficient = trust_coefficient;
        Ok(self)
    }

    /// Returns this transformation with `epsilon` as the term that is added to the norms of the updates.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `epsilon` is negative or not finite.
    pub fn with_epsilon(mut self, epsilon: f64) -> Result<Self, OptimizerError> {
        validate_non_negative("epsilon", epsilon)?;
        self.epsilon = epsilon;
        Ok(self)
    }

    /// Returns the norm at which this transformation clamps the norms from below.
    #[inline]
    pub fn minimum_norm(&self) -> f64 {
        self.minimum_norm
    }

    /// Returns the trust coefficient `c` of this transformation.
    #[inline]
    pub fn trust_coefficient(&self) -> f64 {
        self.trust_coefficient
    }

    /// Returns the term `ε` that this transformation adds to the norms of the updates.
    #[inline]
    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    /// Returns the update `update` of the leaf with the parameter `parameter` scaled by its trust ratio, both in the
    /// statistic data type of the update.
    pub(crate) fn scale_leaf<P: OptimizerValue>(&self, update: &P, parameter: &P) -> Result<P, OptimizerError> {
        // This follows the operation order of `optax.scale_by_trust_ratio`.
        let parameter_norm = safe_norm(parameter, self.minimum_norm)?;
        let update_norm = safe_norm(update, self.minimum_norm)?;
        let trust_ratio = Div::div(
            &Mul::mul(&parameter_norm.fill_like(self.trust_coefficient)?, &parameter_norm)?,
            &Add::add(&update_norm, &update_norm.fill_like(self.epsilon)?)?,
        )?;
        let zero = parameter_norm.fill_like(0f64)?;
        let zero_norm = parameter_norm
            .compare(&zero, ComparisonDirection::Equal)?
            .or(&update_norm.compare(&zero, ComparisonDirection::Equal)?)?;
        let trust_ratio = zero_norm.select(&parameter_norm.fill_like(1f64)?, &trust_ratio)?;
        Ok(Mul::mul(update, &trust_ratio)?)
    }
}

impl Default for ScaleByTrustRatio {
    fn default() -> Self {
        Self::new()
    }
}

impl<P: OptimizerValue> Optimizer<P> for ScaleByTrustRatio {
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
        map_with_parameters(gradients, parameters, |update, parameter| self.scale_leaf(update, parameter))
            .map(|updates| (updates, state))
    }
}

/// Transformation that scales the update of every leaf by the norm of its parameter, clamped from below at a minimum
/// scale, which is the analogue of
/// [`optax.scale_by_param_block_norm`](
/// https://optax.readthedocs.io/en/latest/api/transformations.html#optax.scale_by_param_block_norm).
/// The default minimum scale is `1e-3`.
#[derive(Clone, Debug, PartialEq)]
pub struct ScaleByParamBlockNorm {
    /// Smallest scale, at which the norms of the parameters are clamped from below.
    minimum_scale: f64,
}

impl ScaleByParamBlockNorm {
    /// Creates a new [`ScaleByParamBlockNorm`] with the default minimum scale.
    #[inline]
    pub fn new() -> Self {
        Self { minimum_scale: 1e-3 }
    }

    /// Returns this transformation with `minimum_scale` as the scale at which it clamps the norms from below.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `minimum_scale` is negative or not finite.
    pub fn with_minimum_scale(mut self, minimum_scale: f64) -> Result<Self, OptimizerError> {
        validate_non_negative("minimum_scale", minimum_scale)?;
        self.minimum_scale = minimum_scale;
        Ok(self)
    }

    /// Returns the scale at which this transformation clamps the norms of the parameters from below.
    #[inline]
    pub fn minimum_scale(&self) -> f64 {
        self.minimum_scale
    }
}

impl Default for ScaleByParamBlockNorm {
    fn default() -> Self {
        Self::new()
    }
}

impl<P: OptimizerValue> Optimizer<P> for ScaleByParamBlockNorm {
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
            Ok(Mul::mul(update, &safe_norm(parameter, self.minimum_scale)?)?)
        })
        .map(|updates| (updates, state))
    }
}

/// Transformation that scales the update of every leaf by the root mean square of its parameter, clamped from below at
/// a minimum scale, which is the analogue of
/// [`optax.scale_by_param_block_rms`](
/// https://optax.readthedocs.io/en/latest/api/transformations.html#optax.scale_by_param_block_rms)
/// and the relative step size of Adafactor. The default minimum scale is `1e-3`.
#[derive(Clone, Debug, PartialEq)]
pub struct ScaleByParamBlockRms {
    /// Smallest scale, at which the root mean squares of the parameters are clamped from below.
    minimum_scale: f64,
}

impl ScaleByParamBlockRms {
    /// Creates a new [`ScaleByParamBlockRms`] with the default minimum scale.
    #[inline]
    pub fn new() -> Self {
        Self { minimum_scale: 1e-3 }
    }

    /// Returns this transformation with `minimum_scale` as the scale at which it clamps the root mean squares from
    /// below.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `minimum_scale` is negative or not finite.
    pub fn with_minimum_scale(mut self, minimum_scale: f64) -> Result<Self, OptimizerError> {
        validate_non_negative("minimum_scale", minimum_scale)?;
        self.minimum_scale = minimum_scale;
        Ok(self)
    }

    /// Returns the scale at which this transformation clamps the root mean squares of the parameters from below.
    #[inline]
    pub fn minimum_scale(&self) -> f64 {
        self.minimum_scale
    }
}

impl Default for ScaleByParamBlockRms {
    fn default() -> Self {
        Self::new()
    }
}

impl<P: OptimizerValue> Optimizer<P> for ScaleByParamBlockRms {
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
            Ok(Mul::mul(update, &safe_root_mean_square(parameter, self.minimum_scale)?)?)
        })
        .map(|updates| (updates, state))
    }
}

/// Transformation that divides all updates by their global norm, which is the analogue of
/// [`optax.normalize_by_update_norm`](
/// https://optax.readthedocs.io/en/latest/api/transformations.html#optax.normalize_by_update_norm).
/// With the global norm `‖g‖` of all leaves (accumulated in `f32`, or in `f64` when any leaf is `f64`), the scale
/// factor `s`, and `ε`, it returns `g / ((‖g‖ + ε) / s)`. The defaults are `s = 1` and `ε = 1e-6`.
#[derive(Clone, Debug, PartialEq)]
pub struct NormalizeByUpdateNorm {
    /// Factor `s` by which the normalized updates are scaled.
    scale_factor: f64,

    /// Term `ε` that is added to the global norm.
    epsilon: f64,
}

impl NormalizeByUpdateNorm {
    /// Creates a new [`NormalizeByUpdateNorm`] with the default hyperparameters.
    #[inline]
    pub fn new() -> Self {
        Self { scale_factor: 1.0, epsilon: 1e-6 }
    }

    /// Returns this transformation with `scale_factor` as the factor by which it scales the normalized updates.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `scale_factor` is not positive and finite.
    pub fn with_scale_factor(mut self, scale_factor: f64) -> Result<Self, OptimizerError> {
        validate_positive("scale_factor", scale_factor)?;
        self.scale_factor = scale_factor;
        Ok(self)
    }

    /// Returns this transformation with `epsilon` as the term that is added to the global norm.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `epsilon` is negative or not finite.
    pub fn with_epsilon(mut self, epsilon: f64) -> Result<Self, OptimizerError> {
        validate_non_negative("epsilon", epsilon)?;
        self.epsilon = epsilon;
        Ok(self)
    }

    /// Returns the factor `s` by which this transformation scales the normalized updates.
    #[inline]
    pub fn scale_factor(&self) -> f64 {
        self.scale_factor
    }

    /// Returns the term `ε` that this transformation adds to the global norm.
    #[inline]
    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }
}

impl Default for NormalizeByUpdateNorm {
    fn default() -> Self {
        Self::new()
    }
}

impl<P: OptimizerValue> Optimizer<P> for NormalizeByUpdateNorm {
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
        let Some(squared_norm) = global_squared_norm(&gradients)? else {
            return Ok((S::from_parameters(structure, gradients)?, state));
        };
        let norm = squared_norm.sqrt()?;
        let norm = Div::div(&Add::add(&norm, &norm.fill_like(self.epsilon)?)?, &norm.fill_like(self.scale_factor)?)?;
        let updates = gradients
            .into_iter()
            .map(|gradient| {
                let data_type = gradient.r#type().data_type();
                let gradient = convert_to(&gradient, norm.r#type().data_type())?;
                Ok(convert_to(&Div::div(&gradient, &norm)?, data_type)?)
            })
            .collect::<Result<Vec<_>, OptimizerError>>()?;
        Ok((S::from_parameters(structure, updates)?, state))
    }
}

/// Transformation that subtracts from every leaf of rank two or more its mean over all axes but the first, which is the
/// gradient centralization of [Yong et al. (2020)](https://arxiv.org/abs/2004.01461) and the analogue of
/// [`optax.centralize`](https://optax.readthedocs.io/en/latest/api/transformations.html#optax.centralize). Leaves of
/// rank zero or one are returned unchanged.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Centralize;

impl Centralize {
    /// Creates a new [`Centralize`].
    #[inline]
    pub fn new() -> Self {
        Self
    }
}

impl<P: OptimizerValue> Optimizer<P> for Centralize {
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
        map_with_parameters(gradients, parameters, |update, _| {
            let rank = update.r#type().rank();
            if rank < 2 {
                return Ok(update.clone());
            }
            let mean =
                update.reduce(&(1..rank).collect::<Vec<_>>(), ReductionKind::Mean)?.broadcast_like(update, &[0])?;
            Ok(Sub::sub(update, &mean)?)
        })
        .map(|updates| (updates, state))
    }
}

/// Returns the updates that `function` computes for every leaf of `gradients` and its parameter in `parameters`, after
/// validating that they have the same structure. Both leaves are converted to the statistic data type of the gradient
/// leaf before `function` is called, and the result is converted back to the data type of the gradient leaf.
pub(crate) fn map_with_parameters<P: OptimizerValue, S: Parameterized<P>>(
    gradients: S,
    parameters: &S,
    function: impl Fn(&P, &P) -> Result<P, OptimizerError>,
) -> Result<S, OptimizerError> {
    validate_structure("parameters", &gradients.parameter_paths().collect::<Vec<_>>(), parameters)?;
    let structure = gradients.parameter_structure();
    let updates = gradients
        .into_parameters()
        .zip(parameters.parameters())
        .map(|(gradient, parameter)| {
            let data_type = gradient.r#type().data_type();
            let statistic_type = statistic_data_type(data_type);
            let update = function(&convert_to(&gradient, statistic_type)?, &convert_to(parameter, statistic_type)?)?;
            Ok(convert_to(&update, data_type)?)
        })
        .collect::<Result<Vec<_>, OptimizerError>>()?;
    Ok(S::from_parameters(structure, updates)?)
}

/// Returns the rank-0 squared global norm of `leaves`, accumulated in `f32`, or in `f64` when any leaf is `f64`, which
/// is `optax.tree.norm(leaves, squared=True)`, or [`None`] when there are no leaves.
pub(crate) fn global_squared_norm<P: OptimizerValue>(leaves: &[P]) -> Result<Option<P>, OptimizerError> {
    let statistic_type = if leaves.iter().any(|leaf| leaf.r#type().data_type() == DataType::F64) {
        DataType::F64
    } else {
        DataType::F32
    };
    let mut squared_norm = None;
    for leaf in leaves {
        let leaf = convert_to(leaf, statistic_type)?;
        let leaf_squared_norm = Mul::mul(&leaf, &leaf)?.reduce(&all_axes(&leaf), ReductionKind::Sum)?;
        squared_norm = Some(match squared_norm {
            None => leaf_squared_norm,
            Some(squared_norm) => Add::add(&squared_norm, &leaf_squared_norm)?,
        });
    }
    Ok(squared_norm)
}

/// Returns the rank-0 norm of `value` clamped from below at `minimum_norm`, which is `optax.safe_norm`. The elements
/// whose norm is at most the minimum are replaced with ones before the norm is recomputed, so that the derivative of
/// the result stays finite at zero.
fn safe_norm<P: OptimizerValue>(value: &P, minimum_norm: f64) -> Result<P, OptimizerError> {
    let norm = Mul::mul(value, value)?.reduce(&all_axes(value), ReductionKind::Sum)?.sqrt()?;
    let minimum = norm.fill_like(minimum_norm)?;
    let small = norm.compare(&minimum, ComparisonDirection::LessThanOrEqual)?;
    let masked = small.select(&value.fill_like(1f64)?, value)?;
    let masked_norm = Mul::mul(&masked, &masked)?.reduce(&all_axes(&masked), ReductionKind::Sum)?.sqrt()?;
    Ok(small.select(&minimum, &masked_norm)?)
}

/// Returns the rank-0 root mean square of `value` clamped from below at `minimum`, which is
/// `optax.safe_root_mean_squares`, with the same masking as [`safe_norm`].
fn safe_root_mean_square<P: OptimizerValue>(value: &P, minimum: f64) -> Result<P, OptimizerError> {
    let root_mean_square = Mul::mul(value, value)?.reduce(&all_axes(value), ReductionKind::Mean)?.sqrt()?;
    let minimum = root_mean_square.fill_like(minimum)?;
    let small = root_mean_square.compare(&minimum, ComparisonDirection::LessThanOrEqual)?;
    let masked = small.select(&value.fill_like(1f64)?, value)?;
    let masked_root_mean_square =
        Mul::mul(&masked, &masked)?.reduce(&all_axes(&masked), ReductionKind::Mean)?.sqrt()?;
    Ok(small.select(&minimum, &masked_root_mean_square)?)
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use ryft_core::Array;

    use crate::tests::{assert_parity, parity_updates, vector_updates};

    use super::*;

    #[test]
    fn test_scale_by_trust_ratio_new() {
        let trust_ratio = ScaleByTrustRatio::new();
        assert_eq!(
            (trust_ratio.minimum_norm(), trust_ratio.trust_coefficient(), trust_ratio.epsilon()),
            (0.0, 1.0, 0.0),
        );
        assert_eq!(ScaleByTrustRatio::default(), trust_ratio);
        let trust_ratio = trust_ratio
            .with_minimum_norm(1.0)
            .unwrap()
            .with_trust_coefficient(0.5)
            .unwrap()
            .with_epsilon(1e-3)
            .unwrap();
        assert_eq!(
            (trust_ratio.minimum_norm(), trust_ratio.trust_coefficient(), trust_ratio.epsilon()),
            (1.0, 0.5, 1e-3),
        );
        assert!(matches!(
            ScaleByTrustRatio::new().with_trust_coefficient(-1.0),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`trust_coefficient` must be finite and non-negative but got `-1`",
        ));
    }

    #[test]
    fn test_scale_by_trust_ratio_update() {
        // The parameter `[3, 4]` has the norm `5` and the update `[0, 2]` has the norm `2`, so the trust ratio is
        // `2.5`.
        let trust_ratio = ScaleByTrustRatio::new();
        assert_eq!(vector_updates(&trust_ratio, &[3.0, 4.0], &[&[0.0, 2.0]]), vec![vec![0.0, 5.0]]);

        // A zero parameter norm keeps the update, and a minimum norm of `1` replaces it with `1`, so the ratio is
        // `0.5`.
        assert_eq!(vector_updates(&trust_ratio, &[0.0, 0.0], &[&[0.0, 2.0]]), vec![vec![0.0, 2.0]]);
        let trust_ratio = trust_ratio.with_minimum_norm(1.0).unwrap();
        assert_eq!(vector_updates(&trust_ratio, &[0.0, 0.0], &[&[0.0, 2.0]]), vec![vec![0.0, 1.0]]);
    }

    #[test]
    fn test_scale_by_param_block_norm_new() {
        assert_eq!(ScaleByParamBlockNorm::new().minimum_scale(), 1e-3);
        assert_eq!(ScaleByParamBlockNorm::default(), ScaleByParamBlockNorm::new());
        assert_eq!(ScaleByParamBlockNorm::new().with_minimum_scale(0.5).unwrap().minimum_scale(), 0.5);
        assert!(matches!(
            ScaleByParamBlockNorm::new().with_minimum_scale(f64::NAN),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`minimum_scale` must be finite and non-negative but got `NaN`",
        ));
    }

    #[test]
    fn test_scale_by_param_block_norm_update() {
        // The parameter `[3, 4]` has the norm `5`, and a zero parameter is clamped at the minimum scale `1e-3`.
        let scale = ScaleByParamBlockNorm::new();
        assert_eq!(vector_updates(&scale, &[3.0, 4.0], &[&[1.0, 2.0]]), vec![vec![5.0, 10.0]]);
        assert_eq!(vector_updates(&scale, &[0.0, 0.0], &[&[1.0, 2.0]]), vec![vec![1e-3f32, 2.0 * 1e-3f32]]);
    }

    #[test]
    fn test_scale_by_param_block_rms_new() {
        assert_eq!(ScaleByParamBlockRms::new().minimum_scale(), 1e-3);
        assert_eq!(ScaleByParamBlockRms::default(), ScaleByParamBlockRms::new());
        assert_eq!(ScaleByParamBlockRms::new().with_minimum_scale(0.5).unwrap().minimum_scale(), 0.5);
    }

    #[test]
    fn test_scale_by_param_block_rms_update() {
        // The parameter `[3, 4]` has the root mean square `√((9 + 16) / 2) = √12.5`.
        let scale = ScaleByParamBlockRms::new();
        assert_eq!(
            vector_updates(&scale, &[3.0, 4.0], &[&[1.0, 2.0]]),
            vec![vec![12.5f32.sqrt(), 2.0 * 12.5f32.sqrt(),]],
        );
        assert_eq!(vector_updates(&scale, &[0.0, 0.0], &[&[1.0, 2.0]]), vec![vec![1e-3f32, 2.0 * 1e-3f32]]);
    }

    #[test]
    fn test_normalize_by_update_norm_new() {
        let normalize = NormalizeByUpdateNorm::new();
        assert_eq!((normalize.scale_factor(), normalize.epsilon()), (1.0, 1e-6));
        assert_eq!(NormalizeByUpdateNorm::default(), normalize);
        let normalize = normalize.with_scale_factor(2.0).unwrap().with_epsilon(0.0).unwrap();
        assert_eq!((normalize.scale_factor(), normalize.epsilon()), (2.0, 0.0));
        assert!(matches!(
            NormalizeByUpdateNorm::new().with_scale_factor(0.0),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`scale_factor` must be positive and finite but got `0`",
        ));
    }

    #[test]
    fn test_normalize_by_update_norm_update() {
        // The global norm of `([3], 4)` is `5`, so with `s = 2` and `ε = 0` the updates are divided by `2.5`.
        let normalize = NormalizeByUpdateNorm::new().with_scale_factor(2.0).unwrap().with_epsilon(0.0).unwrap();
        let gradients = (Array::vector(vec![3.0f32]).unwrap(), Array::scalar(4.0f64).unwrap());
        assert_eq!(
            normalize.update(gradients.clone(), (), &gradients, &UpdateArguments::new()),
            Ok(((Array::vector(vec![(3.0f64 / 2.5) as f32]).unwrap(), Array::scalar(4.0f64 / 2.5).unwrap()), ())),
        );
    }

    #[test]
    fn test_centralize_update() {
        // The rows of the matrix have the means `1.5` and `4`, and the vector is returned unchanged.
        assert_eq!(Centralize::new(), Centralize);
        let gradients =
            (Array::matrix(2, 2, vec![1.0f32, 2.0, 3.0, 5.0]).unwrap(), Array::vector(vec![1.0f32, 3.0]).unwrap());
        assert_eq!(
            Centralize::new().update(gradients.clone(), (), &gradients, &UpdateArguments::new()),
            Ok(((Array::matrix(2, 2, vec![-0.5f32, 0.5, -1.0, 1.0]).unwrap(), gradients.1.clone()), ())),
        );
    }

    #[test]
    fn test_scale_by_trust_ratio_optax_parity() {
        // Optax 9ecde04b: `optax.scale_by_trust_ratio()`.
        const TRUST_RATIO: [[f32; 9]; 4] = [
            [0.6123724, -1.2247448, 1.8371173, 1.1952286, -0.29880714, 0.5976143, -0.8964215, 1.4940357, -1.792843],
            [-0.83487195, 4.17436, 1.6697439, 1.4591105, 2.1886659, -2.918221, 0.72955525, -1.4591105, 2.5534432],
            [4.689134, -1.5630447, -3.1260893, -1.189078, 0.39635932, 3.5672338, 1.9817966, 3.1708746, -0.79271865],
            [2.5673616, 1.9255213, -4.4928827, 0.8867108, -2.6601326, 0.4433554, -3.9901986, 1.7734216, 2.6601326],
        ];
        // Optax 9ecde04b: `optax.scale_by_trust_ratio(min_norm=1.5, trust_coefficient=0.5, eps=1e-3)`.
        const TRUST_RATIO_CONFIGURED: [[f32; 9]; 4] = [
            [
                0.07632538,
                -0.15265076,
                0.22897613,
                0.37980527,
                -0.09495132,
                0.18990263,
                -0.28485397,
                0.47475657,
                -0.56970793,
            ],
            [
                -0.04288311,
                0.21441554,
                0.08576622,
                0.22694702,
                0.34042054,
                -0.45389405,
                0.11347351,
                -0.22694702,
                0.39715728,
            ],
            [
                0.25522915,
                -0.08507638,
                -0.17015275,
                -0.16382533,
                0.054608442,
                0.49147594,
                0.2730422,
                0.43686754,
                -0.109216884,
            ],
            [
                0.16680108,
                0.12510082,
                -0.2919019,
                0.1102077,
                -0.33062312,
                0.05510385,
                -0.49593464,
                0.2204154,
                0.33062312,
            ],
        ];
        assert_parity(&parity_updates(&ScaleByTrustRatio::new()), &TRUST_RATIO);
        let trust_ratio = ScaleByTrustRatio::new()
            .with_minimum_norm(1.5)
            .unwrap()
            .with_trust_coefficient(0.5)
            .unwrap()
            .with_epsilon(1e-3)
            .unwrap();
        assert_parity(&parity_updates(&trust_ratio), &TRUST_RATIO_CONFIGURED);
    }

    #[test]
    fn test_scale_by_param_block_norm_optax_parity() {
        // Optax 9ecde04b: `optax.scale_by_param_block_norm(min_scale=1e-3)`.
        const EXPECTED: [[f32; 9]; 4] = [
            [0.2291288, -0.4582576, 0.6873864, 1.1401755, -0.28504387, 0.57008773, -0.8551316, 1.4252193, -1.7102633],
            [-0.15716384, 0.78581923, 0.3143277, 0.9695355, 1.4543034, -1.939071, 0.48476776, -0.9695355, 1.6966872],
            [
                0.93865037,
                -0.31288347,
                -0.62576693,
                -0.69318795,
                0.23106265,
                2.0795639,
                1.1553133,
                1.8485012,
                -0.4621253,
            ],
            [0.5965884, 0.4474413, -1.0440296, 0.48576012, -1.4572804, 0.24288006, -2.1859205, 0.97152025, 1.4572804],
        ];
        assert_parity(&parity_updates(&ScaleByParamBlockNorm::new().with_minimum_scale(1e-3).unwrap()), &EXPECTED);
    }

    #[test]
    fn test_scale_by_param_block_rms_optax_parity() {
        // Optax 9ecde04b: `optax.scale_by_param_block_rms(min_scale=1e-3)`.
        const EXPECTED: [[f32; 9]; 4] = [
            [
                0.13228756,
                -0.26457512,
                0.39686272,
                0.4654747,
                -0.116368674,
                0.23273735,
                -0.34910604,
                0.5818434,
                -0.6982121,
            ],
            [
                -0.08033207,
                0.40166035,
                0.16066414,
                0.29000708,
                0.4350106,
                -0.58001417,
                0.14500354,
                -0.29000708,
                0.50751233,
            ],
            [
                0.47718617,
                -0.15906206,
                -0.31812412,
                -0.20771646,
                0.06923882,
                0.62314934,
                0.3461941,
                0.55391055,
                -0.13847764,
            ],
            [
                0.30807996,
                0.23105998,
                -0.5391399,
                0.14033374,
                -0.42100123,
                0.07016687,
                -0.6315018,
                0.28066748,
                0.42100123,
            ],
        ];
        assert_parity(&parity_updates(&ScaleByParamBlockRms::new().with_minimum_scale(1e-3).unwrap()), &EXPECTED);
    }

    #[test]
    fn test_normalize_by_update_norm_optax_parity() {
        // Optax 9ecde04b: `optax.normalize_by_update_norm(scale_factor=0.5, eps=1e-6)`.
        const EXPECTED: [[f32; 9]; 4] = [
            [
                0.048794955,
                -0.09758991,
                0.14638487,
                0.19517982,
                -0.048794955,
                0.09758991,
                -0.14638487,
                0.24397478,
                -0.29276973,
            ],
            [
                -0.03409967,
                0.17049834,
                0.06819934,
                0.13639867,
                0.20459802,
                -0.27279735,
                0.06819934,
                -0.13639867,
                0.23869768,
            ],
            [
                0.19364893,
                -0.06454964,
                -0.12909928,
                -0.09682447,
                0.03227482,
                0.29047337,
                0.1613741,
                0.25819856,
                -0.06454964,
            ],
            [
                0.12699996,
                0.09524998,
                -0.22224994,
                0.06349998,
                -0.19049996,
                0.03174999,
                -0.2857499,
                0.12699996,
                0.19049996,
            ],
        ];
        let normalize = NormalizeByUpdateNorm::new().with_scale_factor(0.5).unwrap().with_epsilon(1e-6).unwrap();
        assert_parity(&parity_updates(&normalize), &EXPECTED);
    }

    #[test]
    fn test_centralize_optax_parity() {
        // Optax 9ecde04b: `optax.centralize()`.
        const EXPECTED: [[f32; 9]; 4] = [
            [0.1, -0.2, 0.3, 0.23333333, -0.26666668, 0.03333333, -0.16666666, 0.6333333, -0.46666667],
            [-0.05, 0.25, 0.1, 0.16666667, 0.26666668, -0.43333334, 0.016666666, -0.28333333, 0.26666665],
            [0.3, -0.1, -0.2, -0.26666665, -0.06666666, 0.3333333, 0.06666668, 0.21666668, -0.28333333],
            [0.2, 0.15, -0.35, 0.15, -0.25, 0.10000001, -0.46666667, 0.18333332, 0.28333333],
        ];
        assert_parity(&parity_updates(&Centralize::new()), &EXPECTED);
    }
}
