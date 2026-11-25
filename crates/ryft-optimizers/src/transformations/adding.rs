//! Transformations that add terms to the updates, such as the decoupled weight decay of AdamW.

use std::fmt::Debug;
use std::sync::Arc;

use ryft_core::{Add, Mul, ParameterPath, Parameterized};

use crate::{
    Optimizer, OptimizerError, OptimizerValue, UpdateArguments, convert_to, statistic_data_type, validate_finite,
    validate_structure,
};

// TODO(eaplatanios): Review this module.

/// Transformation that adds a multiple of the parameters to the updates (i.e., `update + rate · parameter`), which is
/// the analogue of
/// [`optax.add_decayed_weights`](
/// https://optax.readthedocs.io/en/latest/api/transformations.html#optax.add_decayed_weights).
/// Placed after the rescaling of an adaptive optimizer and before its learning rate, this is the decoupled weight
/// decay of [Loshchilov and Hutter (2019)](https://arxiv.org/abs/1711.05101), which [`AdamW`](crate::AdamW) uses.
/// An optional mask selects the leaves that decay by their [`ParameterPath`]s (e.g., to exclude biases and
/// normalization scales), and the updates of the other leaves are returned unchanged. This transformation is stateless.
#[derive(Clone)]
pub struct AddDecayedWeights {
    /// Multiple of the parameters that is added to the updates.
    rate: f64,

    /// Predicate that selects the leaves that decay by their paths, or [`None`] if all leaves decay.
    mask: Option<Arc<dyn Fn(&ParameterPath) -> bool + Send + Sync>>,
}

impl AddDecayedWeights {
    /// Creates a new [`AddDecayedWeights`] that adds `rate` times the parameters of all leaves to their updates.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `rate` is not finite.
    pub fn new(rate: f64) -> Result<Self, OptimizerError> {
        Self { rate: 0.0, mask: None }.with_rate(rate)
    }

    /// Returns this transformation with `rate` as the multiple of the parameters that it adds to the updates, keeping
    /// its mask. [`AdamW`](crate::AdamW) uses it to change its weight decay rate after setting a mask.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `rate` is not finite.
    pub(crate) fn with_rate(mut self, rate: f64) -> Result<Self, OptimizerError> {
        validate_finite("rate", rate)?;
        self.rate = rate;
        Ok(self)
    }

    /// Returns this transformation with `mask` as the predicate that selects the leaves that decay by their
    /// [`ParameterPath`]s, as returned by [`Parameterized::named_parameters`] for the parameters.
    #[inline]
    pub fn with_mask<Mask: 'static + Fn(&ParameterPath) -> bool + Send + Sync>(mut self, mask: Mask) -> Self {
        self.mask = Some(Arc::new(mask));
        self
    }

    /// Returns the multiple of the parameters that this transformation adds to the updates.
    #[inline]
    pub fn rate(&self) -> f64 {
        self.rate
    }

    /// Returns whether this transformation decays the leaf at `path`, which is the case for every leaf when it has no
    /// mask.
    #[inline]
    pub fn decays(&self, path: &ParameterPath) -> bool {
        self.mask.as_ref().is_none_or(|mask| mask(path))
    }
}

impl Debug for AddDecayedWeights {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Masks are opaque functions, so only their presence is rendered.
        formatter
            .debug_struct("AddDecayedWeights")
            .field("rate", &self.rate)
            .field("masked", &self.mask.is_some())
            .finish()
    }
}

impl<P: OptimizerValue> Optimizer<P> for AddDecayedWeights {
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
        // A zero rate returns the updates unchanged, as `optax.add_decayed_weights` does, so that parameters that are
        // not finite do not turn the updates into NaNs.
        let _ = arguments;
        validate_structure("parameters", &gradients.parameter_paths().collect::<Vec<_>>(), parameters)?;
        if self.rate == 0.0 {
            return Ok((gradients, state));
        }
        let structure = gradients.parameter_structure();
        let updates = gradients
            .into_parameters()
            .zip(parameters.named_parameters())
            .map(|(update, (path, parameter))| {
                if !self.decays(&path) {
                    return Ok(update);
                }
                let data_type = update.r#type().data_type();
                let statistic_type = statistic_data_type(data_type);
                let parameter = convert_to(parameter, statistic_type)?;
                let decay = Mul::mul(&parameter.fill_like(self.rate)?, &parameter)?;
                Ok(convert_to(&Add::add(&convert_to(&update, statistic_type)?, &decay)?, data_type)?)
            })
            .collect::<Result<Vec<_>, OptimizerError>>()?;
        Ok((S::from_parameters(structure, updates)?, state))
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use ryft_core::{Array, ParameterPathSegment, bf16};

    use crate::tests::{assert_parity, parity_updates};

    use super::*;

    #[test]
    fn test_add_decayed_weights_new() {
        assert_eq!(AddDecayedWeights::new(0.5).unwrap().rate(), 0.5);
        assert!(matches!(
            AddDecayedWeights::new(f64::INFINITY),
            Err(OptimizerError::InvalidConfiguration { message }) if message == "`rate` must be finite but got `inf`",
        ));
    }

    #[test]
    fn test_add_decayed_weights_with_mask() {
        let unmasked = AddDecayedWeights::new(0.5).unwrap();
        let masked = unmasked.clone().with_mask(|path| path.to_string() != "$.1");
        let first = ParameterPath::root().with_segment(ParameterPathSegment::TupleIndex(0));
        let second = ParameterPath::root().with_segment(ParameterPathSegment::TupleIndex(1));
        assert!(unmasked.decays(&first));
        assert!(unmasked.decays(&second));
        assert!(masked.decays(&first));
        assert!(!masked.decays(&second));
        assert_eq!(format!("{unmasked:?}"), "AddDecayedWeights { rate: 0.5, masked: false }");
        assert_eq!(format!("{masked:?}"), "AddDecayedWeights { rate: 0.5, masked: true }");
    }

    #[test]
    fn test_add_decayed_weights_update() {
        let parameters = (Array::vector(vec![1.0f32, 2.0]).unwrap(), Array::scalar(bf16::from_f32(4.0)).unwrap());
        let updates = (Array::vector(vec![0.5f32, 0.5]).unwrap(), Array::scalar(bf16::from_f32(0.25)).unwrap());
        let decay = AddDecayedWeights::new(0.5).unwrap();
        let expected = (Array::vector(vec![1.0f32, 1.5]).unwrap(), Array::scalar(bf16::from_f32(2.25)).unwrap());
        assert_eq!(decay.update(updates.clone(), (), &parameters, &UpdateArguments::new()).unwrap(), (expected, ()));

        // A zero rate keeps the updates even for parameters that are not finite.
        let infinite = (Array::vector(vec![f32::INFINITY, 2.0]).unwrap(), Array::scalar(bf16::from_f32(4.0)).unwrap());
        let zero = AddDecayedWeights::new(0.0).unwrap();
        assert_eq!(
            zero.update(updates.clone(), (), &infinite, &UpdateArguments::new()).unwrap(),
            (updates.clone(), ()),
        );

        // Leaves that the mask excludes keep their updates.
        let decay = decay.with_mask(|path| path.to_string() != "$.1");
        let expected = (Array::vector(vec![1.0f32, 1.5]).unwrap(), Array::scalar(bf16::from_f32(0.25)).unwrap());
        assert_eq!(decay.update(updates, (), &parameters, &UpdateArguments::new()).unwrap(), (expected, ()));
    }

    #[test]
    fn test_add_decayed_weights_optax_parity() {
        // Optax 9ecde04b: `optax.add_decayed_weights(0.1)`.
        const EXPECTED: [[f32; 9]; 4] = [
            [0.15, -0.3, 0.5, 0.5, -0.15, 0.22500001, -0.5, 0.65, -0.52500004],
            [0.014999997, 0.120000005, 0.35, 0.35000002, 0.23500001, -0.3525, -0.15, 0.0150000155, 0.3725],
            [0.36650002, -0.218, 0.08499999, 0.034999996, 0.008500002, 0.46225, -0.0150000155, 0.6165, -0.040250003],
            [0.30315, 0.010200009, -0.056499988, 0.2885, -0.34065002, 0.10847501, -0.71650004, 0.47815007, 0.35572502],
        ];
        assert_parity(&parity_updates(&AddDecayedWeights::new(0.1).unwrap()), &EXPECTED);
    }
}
