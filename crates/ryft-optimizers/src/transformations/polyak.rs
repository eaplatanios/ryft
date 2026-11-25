//! Polyak step sizes, which scale the updates by the gap between the objective and its minimum.

use ryft_core::{Add, ComparisonDirection, Div, Mul, Parameterized, Sub};

use crate::transformations::normalization::global_squared_norm;
use crate::{
    Optimizer, OptimizerError, OptimizerValue, UpdateArguments, convert_to, validate_finite, validate_non_negative,
    validate_structure,
};

// TODO(eaplatanios): Review this module.

/// Variant of the Polyak step size of [`ScaleByPolyak`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum PolyakVariant {
    /// The stochastic Polyak step size (SPS) of [Loizou et al. (2021)](https://arxiv.org/abs/2002.10542), which uses
    /// the gap between the objective and its minimum as it is.
    Sps,

    /// The variant of SPS of [Garrigos et al. (2023)](https://arxiv.org/abs/2305.07283) (i.e., `sps+`), which clamps
    /// negative gaps to zero.
    SpsPlus,
}

/// Transformation that scales the updates by the Polyak step size, which is the analogue of
/// [`optax.scale_by_polyak`](https://optax.readthedocs.io/en/latest/api/transformations.html#optax.scale_by_polyak).
/// It requires the value `f` of the objective at the current parameters, which it reads with
/// [`UpdateArguments::validate_value`]. For the updates `g` with the squared global norm `‖g‖²`, the minimum `f*` of
/// the objective, and `ε`, the step size is `min((f - f*) / (‖g‖² + ε), η_max)` with the maximum learning rate `η_max`,
/// or `0` when `‖g‖² + ε` is at most the machine epsilon of `f64`, and the updates are scaled by it. With
/// [`PolyakVariant::SpsPlus`], negative gaps `f - f*` are clamped to zero. The defaults are `f* = 0`, `η_max = 1`, and
/// `ε = 0` with [`PolyakVariant::Sps`]. Note that the step size is positive, so the updates descend only when the
/// gradients were negated before (e.g., by a negative learning rate, as in [`PolyakSgd`](crate::PolyakSgd)). This
/// transformation is stateless.
#[derive(Clone, Debug, PartialEq)]
pub struct ScaleByPolyak {
    /// Minimum `f*` of the objective.
    minimum_value: f64,

    /// Largest step size `η_max`.
    maximum_learning_rate: f64,

    /// Term `ε` that is added to the squared global norm.
    epsilon: f64,

    /// Variant of the step size.
    variant: PolyakVariant,
}

impl ScaleByPolyak {
    /// Creates a new [`ScaleByPolyak`] with the default hyperparameters.
    #[inline]
    pub fn new() -> Self {
        Self { minimum_value: 0.0, maximum_learning_rate: 1.0, epsilon: 0.0, variant: PolyakVariant::Sps }
    }

    /// Returns this transformation with `minimum_value` as the minimum of the objective (i.e., the `f_min` of Optax).
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `minimum_value` is not finite.
    pub fn with_minimum_value(mut self, minimum_value: f64) -> Result<Self, OptimizerError> {
        validate_finite("minimum_value", minimum_value)?;
        self.minimum_value = minimum_value;
        Ok(self)
    }

    /// Returns this transformation with `maximum_learning_rate` as its largest step size.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `maximum_learning_rate` is negative or not finite.
    pub fn with_maximum_learning_rate(mut self, maximum_learning_rate: f64) -> Result<Self, OptimizerError> {
        validate_non_negative("maximum_learning_rate", maximum_learning_rate)?;
        self.maximum_learning_rate = maximum_learning_rate;
        Ok(self)
    }

    /// Returns this transformation with `epsilon` as the term that is added to the squared global norm.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `epsilon` is negative or not finite.
    pub fn with_epsilon(mut self, epsilon: f64) -> Result<Self, OptimizerError> {
        validate_non_negative("epsilon", epsilon)?;
        self.epsilon = epsilon;
        Ok(self)
    }

    /// Returns this transformation with `variant` as the variant of its step size.
    #[inline]
    pub fn with_variant(mut self, variant: PolyakVariant) -> Self {
        self.variant = variant;
        self
    }

    /// Returns the minimum `f*` of the objective of this transformation.
    #[inline]
    pub fn minimum_value(&self) -> f64 {
        self.minimum_value
    }

    /// Returns the largest step size `η_max` of this transformation.
    #[inline]
    pub fn maximum_learning_rate(&self) -> f64 {
        self.maximum_learning_rate
    }

    /// Returns the term `ε` that this transformation adds to the squared global norm.
    #[inline]
    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    /// Returns the variant of the step size of this transformation.
    #[inline]
    pub fn variant(&self) -> PolyakVariant {
        self.variant
    }
}

impl Default for ScaleByPolyak {
    fn default() -> Self {
        Self::new()
    }
}

impl<P: OptimizerValue> Optimizer<P> for ScaleByPolyak {
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
        // This follows the operation order of `optax.scale_by_polyak`, whose threshold `jnp.finfo(float).eps` is the
        // machine epsilon of `f64`.
        let value = arguments.validate_value("ScaleByPolyak")?;
        validate_structure("parameters", &gradients.parameter_paths().collect::<Vec<_>>(), parameters)?;
        let structure = gradients.parameter_structure();
        let gradients = gradients.into_parameters().collect::<Vec<_>>();
        let Some(squared_norm) = global_squared_norm(&gradients)? else {
            return Ok((S::from_parameters(structure, gradients)?, state));
        };
        let statistic_type = squared_norm.r#type().data_type();
        let gap = convert_to(&Sub::sub(value, &value.fill_like(self.minimum_value)?)?, statistic_type)?;
        let gap = match self.variant {
            PolyakVariant::Sps => gap,
            PolyakVariant::SpsPlus => gap.max(&gap.fill_like(0f64)?)?,
        };
        let denominator = Add::add(&squared_norm, &squared_norm.fill_like(self.epsilon)?)?;
        let vanishing =
            denominator.compare(&denominator.fill_like(f64::EPSILON)?, ComparisonDirection::LessThanOrEqual)?;
        let step = Div::div(&gap, &denominator)?.min(&gap.fill_like(self.maximum_learning_rate)?)?;
        let step = vanishing.select(&step.fill_like(0f64)?, &step)?;
        let updates = gradients
            .into_iter()
            .map(|gradient| {
                let data_type = gradient.r#type().data_type();
                let gradient = convert_to(&gradient, statistic_type)?;
                Ok(convert_to(&Mul::mul(&step, &gradient)?, data_type)?)
            })
            .collect::<Result<Vec<_>, OptimizerError>>()?;
        Ok((S::from_parameters(structure, updates)?, state))
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use ryft_core::Array;

    use crate::tests::{assert_parity, parity_quadratic_updates};
    use crate::{Chain, Scale};

    use super::*;

    #[test]
    fn test_scale_by_polyak_new() {
        let polyak = ScaleByPolyak::new();
        assert_eq!(
            (polyak.minimum_value(), polyak.maximum_learning_rate(), polyak.epsilon(), polyak.variant()),
            (0.0, 1.0, 0.0, PolyakVariant::Sps),
        );
        assert_eq!(ScaleByPolyak::default(), polyak);
        let polyak = polyak
            .with_minimum_value(-1.0)
            .unwrap()
            .with_maximum_learning_rate(0.5)
            .unwrap()
            .with_epsilon(1e-3)
            .unwrap()
            .with_variant(PolyakVariant::SpsPlus);
        assert_eq!(
            (polyak.minimum_value(), polyak.maximum_learning_rate(), polyak.epsilon(), polyak.variant()),
            (-1.0, 0.5, 1e-3, PolyakVariant::SpsPlus),
        );
        assert!(matches!(
            ScaleByPolyak::new().with_maximum_learning_rate(-1.0),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "`maximum_learning_rate` must be finite and non-negative but got `-1`",
        ));
    }

    #[test]
    fn test_scale_by_polyak_update() {
        // The gradients `[3, 4]` have the squared norm `25`, so the objective `5` gives the step size `5 / 25 = 0.2`.
        let polyak = ScaleByPolyak::new();
        let gradients = vec![Array::vector(vec![3.0f32, 4.0]).unwrap()];
        let value = Array::scalar(5.0f32).unwrap();
        let arguments = UpdateArguments::new().with_value(&value);
        assert_eq!(
            polyak.update(gradients.clone(), (), &gradients, &arguments),
            Ok((vec![Array::vector(vec![(5.0f32 / 25.0) * 3.0, (5.0f32 / 25.0) * 4.0]).unwrap()], ())),
        );

        // The objective `50` gives the step size `2`, which is clamped to the maximum learning rate `1`.
        let value = Array::scalar(50.0f32).unwrap();
        let arguments = UpdateArguments::new().with_value(&value);
        assert_eq!(polyak.update(gradients.clone(), (), &gradients, &arguments).unwrap().0, gradients);

        // A negative gap gives a negative step size, unless the `sps+` variant clamps it to zero.
        let value = Array::scalar(-5.0f32).unwrap();
        let arguments = UpdateArguments::new().with_value(&value);
        assert_eq!(
            polyak.update(gradients.clone(), (), &gradients, &arguments).unwrap().0,
            vec![Array::vector(vec![(-5.0f32 / 25.0) * 3.0, (-5.0f32 / 25.0) * 4.0]).unwrap()],
        );
        let plus = polyak.clone().with_variant(PolyakVariant::SpsPlus);
        assert_eq!(
            plus.update(gradients.clone(), (), &gradients, &arguments).unwrap().0,
            vec![Array::vector(vec![0.0f32, 0.0]).unwrap()],
        );

        // Zero gradients give a zero step size rather than a division by zero.
        let zeros = vec![Array::vector(vec![0.0f32, 0.0]).unwrap()];
        assert_eq!(polyak.update(zeros.clone(), (), &zeros, &arguments).unwrap().0, zeros);

        // The step size of `f64` gradients stays in `f64`, so `1 / 25` is not rounded to `f32`.
        let gradients = vec![Array::vector(vec![3.0f64, 4.0]).unwrap()];
        let value = Array::scalar(1.0f64).unwrap();
        let arguments = UpdateArguments::new().with_value(&value);
        assert_eq!(
            polyak.update(gradients.clone(), (), &gradients, &arguments).unwrap().0,
            vec![Array::vector(vec![(1.0f64 / 25.0) * 3.0, (1.0f64 / 25.0) * 4.0]).unwrap()],
        );
    }

    #[test]
    fn test_scale_by_polyak_update_missing_value() {
        let gradients = vec![Array::scalar(1.0f32).unwrap()];
        assert!(matches!(
            ScaleByPolyak::new().update(gradients.clone(), (), &gradients, &UpdateArguments::new()),
            Err(OptimizerError::MissingArgument { message })
                if message == "`ScaleByPolyak` requires the `value` update argument, which was not provided",
        ));
    }

    #[test]
    fn test_scale_by_polyak_optax_parity() {
        // Optax 9ecde04b: `optax.chain(optax.scale_by_polyak(max_learning_rate=0.5, eps=1e-3), optax.scale(-1.0))`, on
        // the
        // parity quadratic.
        const POLYAK_SPS: [[f32; 9]; 3] = [
            [0.123917796, 0.99134237, -0.7435068, -0.0, 0.3717534, 0.27881503, 1.4870136, -0.3097945, 0.1858767],
            [0.101808265, 0.5461024, -0.2083039, -0.0, 0.30542478, 0.19132996, 0.8191535, -0.12872519, 0.05207595],
            [0.08313133, 0.28039727, -0.043817937, -0.0, 0.24939401, 0.1272344, 0.42059597, -0.04658616, 0.010954511],
        ];
        // Optax 9ecde04b: `optax.chain(optax.scale_by_polyak(f_min=0.9, variant='sps+'), optax.scale(-1.0))`, on the
        // parity
        // quadratic.
        const POLYAK_SPS_PLUS: [[f32; 9]; 3] = [
            [0.117192246, 0.93753797, -0.7031535, -0.0, 0.35157675, 0.26368254, 1.406307, -0.2929806, 0.17578837],
            [0.083368905, 0.46277174, -0.19394411, -0.0, 0.25010672, 0.1588673, 0.69415754, -0.112713076, 0.04848601],
            [
                0.041406583,
                0.16585107,
                -0.04268822,
                -0.0,
                0.124219745,
                0.06792001,
                0.24877661,
                -0.032601807,
                0.010672055,
            ],
        ];
        let polyak = ScaleByPolyak::new().with_maximum_learning_rate(0.5).unwrap().with_epsilon(1e-3).unwrap();
        let optimizer = Chain::new(polyak, Scale::new(-1.0).unwrap());
        assert_parity(&parity_quadratic_updates(&optimizer, 3), &POLYAK_SPS);
        let polyak = ScaleByPolyak::new().with_minimum_value(0.9).unwrap().with_variant(PolyakVariant::SpsPlus);
        let optimizer = Chain::new(polyak, Scale::new(-1.0).unwrap());
        assert_parity(&parity_quadratic_updates(&optimizer, 3), &POLYAK_SPS_PLUS);
    }
}
