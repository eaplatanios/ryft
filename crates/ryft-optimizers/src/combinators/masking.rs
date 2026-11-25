//! Application of an optimizer to the subset of the leaves that a predicate on their paths selects.
//!
//! [`Masked`] hands the optimizer that it wraps a [`SelectedLeaves`] map from the [`LeafPath`]s of the selected leaves
//! to the leaves, so the state of that optimizer holds statistics for the selected leaves only, named by their paths.
//! [`Freeze`] is the [`Masked`] [`SetToZero`] transformation, which freezes the selected leaves.

use std::collections::BTreeMap;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use std::sync::Arc;

use ryft_core::{Parameter, ParameterPath, Parameterized};

use crate::macros::parameterized_state;
use crate::{Optimizer, OptimizerError, OptimizerValue, SetToZero, UpdateArguments, validate_structure};

// TODO(eaplatanios): Review this module.

/// Path of a leaf of the structure that a [`Masked`] optimizer receives, which keys the [`SelectedLeaves`] that it
/// hands to the optimizer that it wraps. Its [`Debug`] rendering is the [`Display`] rendering of the path (e.g.,
/// `$.weights`), so that the leaves of the state of the wrapped optimizer have paths such as
/// `$.inner_state.trace[$.weights]`, which name the original parameters.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LeafPath(ParameterPath);

impl LeafPath {
    /// Creates a new [`LeafPath`] for the leaf at `path`.
    #[inline]
    pub fn new(path: ParameterPath) -> Self {
        Self(path)
    }

    /// Returns the path of the leaf that this key refers to.
    #[inline]
    pub fn path(&self) -> &ParameterPath {
        &self.0
    }
}

impl Debug for LeafPath {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Map keys are rendered with `Debug` in parameter paths, so the path is rendered as it would be on its own.
        write!(formatter, "{}", self.0)
    }
}

impl Display for LeafPath {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.0)
    }
}

/// Map from the [`LeafPath`]s of the leaves that a [`Masked`] or [`Partition`](crate::Partition) optimizer selects to
/// those leaves, which is the structure that the optimizers that they wrap are initialized for and updated with.
pub type SelectedLeaves<P> = BTreeMap<LeafPath, P>;

parameterized_state! {
    /// State of [`Masked`], which is the analogue of
    /// [`optax.MaskedState`](https://optax.readthedocs.io/en/latest/api/optimizer_wrappers.html#optax.MaskedState).
    /// It holds the state `S` of the wrapped optimizer, which was initialized for the [`SelectedLeaves`] of the
    /// parameters, so its leaf paths start with `$.inner_state` and name the selected parameters by their paths (e.g.,
    /// `$.inner_state.trace[$.weights]`).
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct MaskedState<P, S> with family MaskedStateFamily and marker leaf {
        /// State of the wrapped optimizer for the selected leaves.
        inner_state: S,
    }
}

impl<P: Parameter, S> MaskedState<P, S> {
    /// Returns the state of the wrapped optimizer for the selected leaves.
    #[inline]
    pub fn inner_state(&self) -> &S {
        &self.inner_state
    }
}

/// Optimizer that applies another optimizer to the leaves that a predicate on their [`ParameterPath`]s selects and
/// passes the updates of all other leaves through unchanged, which is the analogue of
/// [`optax.masked`](https://optax.readthedocs.io/en/latest/api/optimizer_wrappers.html#optax.masked) with a callable
/// mask. The predicate sees the paths of the structure that this optimizer receives (e.g., `$.weights` for a field
/// `weights` of the parameters), as returned by [`Parameterized::named_parameters`], so a [`Masked`] optimizer nested
/// inside another one sees the paths of the [`SelectedLeaves`] of the outer one (e.g., `$[$.weights]`).
///
/// The wrapped optimizer is initialized for, and updated with, the [`SelectedLeaves`] map of the selected leaves of the
/// parameters and the gradients, so its state holds statistics for those leaves only (refer to [`MaskedState`]). Its
/// [`UpdateArguments`] are restricted in the same way: the argument gradients become the selected argument gradients,
/// and the objective (with or without gradients) becomes a function of the selected leaves that evaluates the objective
/// at the current parameters with those leaves replaced (and returns the selected gradients). This is what
/// `optax.masked` does with `mask_compatible_extra_args=True`, extended to the objective, whose arguments must match
/// the structure that the wrapped optimizer sees. Optimizers that need at least one leaf (e.g., to create a step count)
/// fail when the predicate selects none.
#[derive(Clone)]
pub struct Masked<O> {
    /// Optimizer that is applied to the selected leaves.
    inner: O,

    /// Predicate that selects the leaves to which the wrapped optimizer is applied, by their paths.
    mask: Arc<dyn Fn(&ParameterPath) -> bool + Send + Sync>,
}

impl<O> Masked<O> {
    /// Creates a new [`Masked`] that applies `inner` to the leaves whose paths satisfy `mask`.
    #[inline]
    pub fn new<Mask: 'static + Fn(&ParameterPath) -> bool + Send + Sync>(inner: O, mask: Mask) -> Self {
        Self { inner, mask: Arc::new(mask) }
    }

    /// Returns the optimizer that this optimizer applies to the selected leaves.
    #[inline]
    pub fn inner(&self) -> &O {
        &self.inner
    }

    /// Returns whether this optimizer applies its wrapped optimizer to the leaf at `path`.
    #[inline]
    pub fn selects(&self, path: &ParameterPath) -> bool {
        (self.mask)(path)
    }

    /// Returns the [`SelectedLeaves`] of `structure`, cloning the selected leaves.
    fn select<P: Parameter + Clone, S: Parameterized<P>>(&self, structure: &S) -> SelectedLeaves<P> {
        structure
            .named_parameters()
            .filter(|(path, _)| self.selects(path))
            .map(|(path, leaf)| (LeafPath(path), leaf.clone()))
            .collect()
    }

    /// Returns the structure `structure` with the leaves `leaves` (i.e., the named leaves of a structure with that
    /// structure), where every selected leaf is replaced with the leaf that `selected` returns for its path.
    fn merge<P: Parameter, S: Parameterized<P>>(
        &self,
        structure: S::ParameterStructure,
        leaves: impl Iterator<Item = (ParameterPath, P)>,
        mut selected: impl FnMut(&LeafPath) -> Option<P>,
    ) -> Result<S, OptimizerError> {
        let leaves = leaves
            .map(|(path, leaf)| {
                if !self.selects(&path) {
                    return Ok(leaf);
                }
                let path = LeafPath(path);
                selected(&path).ok_or_else(|| OptimizerError::StructureMismatch {
                    message: format!("expected the selected leaves to contain the leaf at `{path}`"),
                })
            })
            .collect::<Result<Vec<_>, OptimizerError>>()?;
        Ok(S::from_parameters(structure, leaves)?)
    }
}

impl Masked<SetToZero> {
    /// Creates a new [`Freeze`], which replaces the updates of the leaves whose paths satisfy `mask` with zeros and
    /// passes all other updates through unchanged, which is the analogue of
    /// [`optax.freeze`](https://optax.readthedocs.io/en/latest/api/optimizer_wrappers.html#optax.freeze).
    #[inline]
    pub fn freeze<Mask: 'static + Fn(&ParameterPath) -> bool + Send + Sync>(mask: Mask) -> Self {
        Self::new(SetToZero, mask)
    }
}

impl<O: Debug> Debug for Masked<O> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Masks are opaque functions, so they are not rendered.
        formatter.debug_struct("Masked").field("inner", &self.inner).finish_non_exhaustive()
    }
}

impl<P: OptimizerValue, O: Optimizer<P>> Optimizer<P> for Masked<O> {
    type State<S: Parameterized<P>> = MaskedState<P, O::State<SelectedLeaves<P>>>;

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        Ok(MaskedState { inner_state: self.inner.initialize(&self.select(parameters))?, leaf: PhantomData })
    }

    fn update<S: Parameterized<P>>(
        &self,
        gradients: S,
        state: Self::State<S>,
        parameters: &S,
        arguments: &UpdateArguments<'_, P, S>,
    ) -> Result<(S, Self::State<S>), OptimizerError> {
        // This follows `optax.masked`, with the extra arguments restricted to the selected leaves.
        validate_structure("parameters", &gradients.parameter_paths().collect::<Vec<_>>(), parameters)?;
        let selected_gradients = self.select(&gradients);
        let selected_parameters = self.select(parameters);
        let selected_argument_gradients = arguments.gradients().map(|gradients| self.select(gradients));
        let objective = arguments.objective();
        let selected_objective = |selected: &SelectedLeaves<P>| -> Result<P, OptimizerError> {
            let objective = objective.ok_or_else(|| OptimizerError::MissingArgument {
                message: "`Masked` requires the `objective` update argument, which was not provided".to_string(),
            })?;
            let leaves = parameters.named_parameters().map(|(path, leaf)| (path, leaf.clone()));
            objective(&self.merge(parameters.parameter_structure(), leaves, |path| selected.get(path).cloned())?)
        };
        let objective_with_gradients = arguments.objective_with_gradients();
        let selected_objective_with_gradients =
            |selected: &SelectedLeaves<P>| -> Result<(P, SelectedLeaves<P>), OptimizerError> {
                let objective = objective_with_gradients.ok_or_else(|| OptimizerError::MissingArgument {
                    message: "`Masked` requires the `objective_with_gradients` update argument, which was not provided"
                        .to_string(),
                })?;
                let leaves = parameters.named_parameters().map(|(path, leaf)| (path, leaf.clone()));
                let (value, gradients) = objective(
                    &self.merge(parameters.parameter_structure(), leaves, |path| selected.get(path).cloned())?,
                )?;
                Ok((value, self.select(&gradients)))
            };
        let mut selected_arguments = UpdateArguments::new();
        if let Some(value) = arguments.value() {
            selected_arguments = selected_arguments.with_value(value);
        }
        if let Some(gradients) = &selected_argument_gradients {
            selected_arguments = selected_arguments.with_gradients(gradients);
        }
        if objective.is_some() {
            selected_arguments = selected_arguments.with_objective(&selected_objective);
        }
        if objective_with_gradients.is_some() {
            selected_arguments = selected_arguments.with_objective_with_gradients(&selected_objective_with_gradients);
        }

        let (mut selected_updates, inner_state) =
            self.inner
                .update(selected_gradients, state.inner_state, &selected_parameters, &selected_arguments)?;
        let structure = gradients.parameter_structure();
        let updates = self.merge(structure, gradients.into_named_parameters(), |path| selected_updates.remove(path))?;
        Ok((updates, MaskedState { inner_state, leaf: PhantomData }))
    }
}

/// [`Masked`] [`SetToZero`] transformation, which replaces the updates of the selected leaves with zeros and so freezes
/// the corresponding parameters, and which [`Masked::freeze`] creates. It is the analogue of
/// [`optax.freeze`](https://optax.readthedocs.io/en/latest/api/optimizer_wrappers.html#optax.freeze).
pub type Freeze = Masked<SetToZero>;

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use pretty_assertions::assert_eq;

    use ryft_core::{Add, Array, Mul, Reduce, ReductionKind};

    use crate::tests::{ArgumentProbe, assert_parity, parity_updates, state_paths};
    use crate::{Chain, ScaleBySchedule, ScaleByZoomLinesearch, Sgd, Trace};

    use super::*;

    #[test]
    fn test_leaf_path() {
        let path = ParameterPath::root().field("weights");
        let leaf_path = LeafPath::new(path.clone());
        assert_eq!(leaf_path.path(), &path);
        assert_eq!(format!("{leaf_path:?}"), "$.weights");
        assert_eq!(leaf_path.to_string(), "$.weights");
    }

    #[test]
    fn test_masked_new() {
        let masked = Masked::new(Trace::new(0.5).unwrap(), |path| path.to_string() == "$.1");
        assert_eq!(masked.inner(), &Trace::new(0.5).unwrap());
        assert!(masked.selects(&ParameterPath::root().tuple_index(1)));
        assert!(!masked.selects(&ParameterPath::root().tuple_index(0)));
        assert_eq!(
            format!("{masked:?}"),
            "Masked { inner: Trace { decay: 0.5, nesterov: false, accumulator_data_type: None }, .. }",
        );
    }

    #[test]
    fn test_masked_freeze() {
        // Frozen leaves get zero updates and all other updates pass through unchanged.
        let freeze = Masked::freeze(|path| path.to_string() == "$.0");
        let parameters = (Array::scalar(1.0f32).unwrap(), Array::scalar(2.0f32).unwrap());
        let gradients = (Array::scalar(3.0f32).unwrap(), Array::scalar(4.0f32).unwrap());
        let state = freeze.initialize(&parameters).unwrap();
        assert_eq!(state.named_parameters().count(), 0);
        let (updates, _) = freeze.update(gradients, state, &parameters, &UpdateArguments::new()).unwrap();
        assert_eq!(updates, (Array::scalar(0f32).unwrap(), Array::scalar(4.0f32).unwrap()));
    }

    #[test]
    fn test_masked_initialize() {
        // The state of the wrapped optimizer holds statistics for the selected leaves only, named by their paths.
        let masked = Masked::new(Trace::new(0.5).unwrap(), |path| path.to_string().contains("weights"));
        assert_eq!(state_paths(&masked), vec!["$.inner_state.trace[$[\"weights\"]]"]);
        let masked = Masked::new(ScaleBySchedule::new(1.0f64), |_| false);
        assert!(matches!(
            masked.initialize(&vec![Array::scalar(1.0f32).unwrap()]),
            Err(OptimizerError::StructureMismatch { message })
                if message == "`ScaleBySchedule` requires at least one parameter to create its step count",
        ));
    }

    #[test]
    fn test_masked_update() {
        // With `μ = 0.5`, the selected leaf `$.1` gets the traces `2` and `1 + 0.5 · 2 = 2` of its gradients `2` and
        // `1`, while the gradients of `$.0` pass through unchanged.
        let masked = Masked::new(Trace::new(0.5).unwrap(), |path| path.to_string() == "$.1");
        let parameters = (Array::vector(vec![0.0f32, 0.0]).unwrap(), Array::scalar(0.0f32).unwrap());
        let arguments = UpdateArguments::new();
        let state = masked.initialize(&parameters).unwrap();
        let gradients = (Array::vector(vec![1.0f32, -1.0]).unwrap(), Array::scalar(2.0f32).unwrap());
        let (updates, state) = masked.update(gradients, state, &parameters, &arguments).unwrap();
        assert_eq!(updates, (Array::vector(vec![1.0f32, -1.0]).unwrap(), Array::scalar(2.0f32).unwrap()));
        assert_eq!(
            state.inner_state().trace(),
            &BTreeMap::from([(LeafPath::new(ParameterPath::root().tuple_index(1)), Array::scalar(2.0f32).unwrap())]),
        );
        let gradients = (Array::vector(vec![3.0f32, 4.0]).unwrap(), Array::scalar(1.0f32).unwrap());
        let (updates, _) = masked.update(gradients, state, &parameters, &arguments).unwrap();
        assert_eq!(updates, (Array::vector(vec![3.0f32, 4.0]).unwrap(), Array::scalar(2.0f32).unwrap()));

        // A state whose selected leaves differ from those of the gradients is rejected by the wrapped optimizer.
        let state = masked.initialize(&parameters).unwrap();
        let gradients = (Array::vector(vec![1.0f32, -1.0]).unwrap(), (Array::scalar(2.0f32).unwrap(),));
        let parameters = (Array::vector(vec![0.0f32, 0.0]).unwrap(), (Array::scalar(0.0f32).unwrap(),));
        assert!(matches!(
            masked.update(gradients, state, &parameters, &UpdateArguments::new()),
            Err(OptimizerError::StructureMismatch { message })
                if message == "expected the leaf count of `trace` to be 0 but got 1",
        ));
    }

    #[test]
    fn test_masked_update_arguments() {
        // The probe sees the argument gradient `20` of the selected leaf `$.1`, the value `0.5`, and the objective
        // `1 + 2` at the full parameters, so its update is `23.5`, and the gradient of `$.0` passes through.
        let masked = Masked::new(ArgumentProbe, |path| path.to_string() == "$.1");
        let parameters = (Array::scalar(1.0f32).unwrap(), Array::scalar(2.0f32).unwrap());
        let gradients = (Array::scalar(5.0f32).unwrap(), Array::scalar(6.0f32).unwrap());
        let value = Array::scalar(0.5f32).unwrap();
        let argument_gradients = (Array::scalar(10.0f32).unwrap(), Array::scalar(20.0f32).unwrap());
        let objective = |parameters: &(Array, Array)| -> Result<Array, OptimizerError> {
            Ok(Add::add(&parameters.0, &parameters.1)?)
        };
        let arguments = UpdateArguments::new()
            .with_value(&value)
            .with_gradients(&argument_gradients)
            .with_objective(&objective);
        let state = masked.initialize(&parameters).unwrap();
        let (updates, _) = masked.update(gradients.clone(), state, &parameters, &arguments).unwrap();
        assert_eq!(updates, (Array::scalar(5.0f32).unwrap(), Array::scalar(23.5f32).unwrap()));

        // Missing arguments stay missing for the wrapped optimizer.
        let state = masked.initialize(&parameters).unwrap();
        assert!(matches!(
            masked.update(gradients, state, &parameters, &UpdateArguments::new()),
            Err(OptimizerError::MissingArgument { message })
                if message == "`ArgumentProbe` requires the `value` update argument, which was not provided",
        ));
    }

    #[test]
    fn test_masked_update_objective_with_gradients() {
        // The zoom line search sees the objective `f(x) = x₀² + x₁₁² + x₁₂²` as a function of the selected leaf `$.1`,
        // with the selected gradients, so it finds the step size `0.5` along `-∇f = (-2, -4)` (refer to the tests of
        // `ScaleByZoomLinesearch`), while the update of `$.0` passes through.
        let masked = Masked::new(ScaleByZoomLinesearch::new(20).unwrap(), |path| path.to_string() == "$.1");
        let parameters = (Array::scalar(3.0f32).unwrap(), Array::vector(vec![1.0f32, 2.0]).unwrap());
        let objective = |parameters: &(Array, Array)| -> Result<(Array, (Array, Array)), OptimizerError> {
            let value = Add::add(
                &Mul::mul(&parameters.0, &parameters.0)?,
                &Mul::mul(&parameters.1, &parameters.1)?.reduce(&[0], ReductionKind::Sum)?,
            )?;
            let two = Array::scalar(2.0f32)?;
            Ok((value, (Mul::mul(&two, &parameters.0)?, Mul::mul(&Array::vector(vec![2.0f32, 2.0])?, &parameters.1)?)))
        };
        let (value, gradients) = objective(&parameters).unwrap();
        let arguments = UpdateArguments::new()
            .with_value(&value)
            .with_gradients(&gradients)
            .with_objective_with_gradients(&objective);
        let direction = (Array::scalar(7.0f32).unwrap(), Array::vector(vec![-2.0f32, -4.0]).unwrap());
        let state = masked.initialize(&parameters).unwrap();
        let (updates, state) = masked.update(direction, state, &parameters, &arguments).unwrap();
        assert_eq!(updates, (Array::scalar(7.0f32).unwrap(), Array::vector(vec![-1.0f32, -2.0]).unwrap()));
        assert_eq!(state.inner_state().learning_rate(), &Array::scalar(0.5f32).unwrap());
    }

    #[test]
    fn test_masked_optax_parity() {
        // Optax 9ecde04b: `optax.masked(optax.sgd(0.1, momentum=0.9), {'bias': False, 'weights': True})`. The leaves
        // that are not selected keep their gradients as updates.
        const EXPECTED: [[f32; 9]; 4] = [
            [0.1, -0.2, 0.3, -0.040000003, 0.010000001, -0.020000001, 0.030000001, -0.05, 0.060000002],
            [-0.05, 0.25, 0.1, -0.056, -0.021000002, 0.022000002, 0.017000003, -0.024999999, 0.019000003],
            [0.3, -0.1, -0.2, -0.0354, -0.0239, -0.025199998, -0.009699999, -0.0625, 0.027100002],
            [0.2, 0.15, -0.35, -0.04186, 0.008490003, -0.027679998, 0.03627, -0.07625, -0.00561],
        ];
        let optimizer =
            Masked::new(Sgd::new(0.1).with_momentum(0.9).unwrap(), |path| path.to_string().contains("weights"));
        assert_parity(&parity_updates(&optimizer), &EXPECTED);
    }

    #[test]
    fn test_freeze_optax_parity() {
        // Optax 9ecde04b: `optax.chain(optax.sgd(0.1), optax.transforms.freeze({'bias': True, 'weights': False}))`.
        const EXPECTED: [[f32; 9]; 4] = [
            [0.0, 0.0, 0.0, -0.040000003, 0.010000001, -0.020000001, 0.030000001, -0.05, 0.060000002],
            [0.0, 0.0, 0.0, -0.020000001, -0.030000001, 0.040000003, -0.010000001, 0.020000001, -0.035],
            [0.0, 0.0, 0.0, 0.015000001, -0.0050000004, -0.044999998, -0.025, -0.040000003, 0.010000001],
            [0.0, 0.0, 0.0, -0.010000001, 0.030000001, -0.0050000004, 0.044999998, -0.020000001, -0.030000001],
        ];
        let optimizer = Chain::new(Sgd::new(0.1), Masked::freeze(|path| path.to_string().contains("bias")));
        assert_parity(&parity_updates(&optimizer), &EXPECTED);
    }
}
