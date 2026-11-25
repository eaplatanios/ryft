//! Application of one optimizer per label to the leaves that carry that label.
//!
//! [`Partition`] labels every leaf with a function of its path and applies one [`Masked`] optimizer per label, chained
//! in the order in which the labels were added. [`SelectiveTransform`] is the partition that freezes the leaves that a
//! predicate selects and applies an optimizer to all others.

use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Arc;

use ryft_core::{Parameter, ParameterPath, Parameterized};

use crate::macros::parameterized_state;
use crate::{Chain, Masked, Optimizer, OptimizerError, OptimizerValue, SetToZero, UpdateArguments};

// TODO(eaplatanios): Review this module.

parameterized_state! {
    /// State of [`Partition`], which is the analogue of [`optax.PartitionState`](
    /// https://optax.readthedocs.io/en/latest/api/combining_optimizers.html#optax.PartitionState). It holds the state
    /// `S` of the chain of [`Masked`] optimizers of the partition, so the leaves of the state of the optimizer of the
    /// first label have paths that start with `$.inner_states.0` when there are two labels (or with
    /// `$.inner_states.0.0` when there are three, and so on, following the nesting of [`Chain`]), and those of the last
    /// label have paths that start with `$.inner_states.1`.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct PartitionState<P, S> with family PartitionStateFamily and marker leaf {
        /// States of the [`Masked`] optimizers of the labels, nested as their [`Chain`]s are.
        inner_states: S,
    }
}

impl<P: Parameter, S> PartitionState<P, S> {
    /// Returns the states of the [`Masked`] optimizers of the labels, nested as their [`Chain`]s are.
    #[inline]
    pub fn inner_states(&self) -> &S {
        &self.inner_states
    }
}

/// Optimizer that labels every leaf with a function of its [`ParameterPath`] and applies one optimizer per label to the
/// leaves with that label, which is the analogue of
/// [`optax.partition`](https://optax.readthedocs.io/en/latest/api/combining_optimizers.html#optax.partition) with a
/// labeling function. Optax takes a mapping from labels to optimizers, which this optimizer encodes statically so
/// that the optimizers of different labels can have different types. [`Self::new`] creates a partition with the
/// optimizer of one label, and every [`Self::with_group`] adds the optimizer of another label. The optimizers are
/// applied as the [`Chain`] of one [`Masked`] optimizer per label, in the order in which the labels were added, so
/// a partition with the optimizers `a`, `b`, and `c` has the groups `Chain<Chain<Masked<A>, Masked<B>>, Masked<C>>`.
/// Because the masks of different labels select disjoint leaves, this chain updates every leaf with the optimizer of
/// its label only, which is what Optax does.
///
/// Every leaf must carry the label of one of the optimizers, which [`Optimizer::initialize`] and [`Optimizer::update`]
/// validate. Like [`Masked`], the optimizer of every label sees the [`SelectedLeaves`](crate::SelectedLeaves) of its
/// leaves, so its state is named by the paths of those leaves, and it needs at least one leaf when it creates a step
/// count.
#[derive(Clone)]
pub struct Partition<G> {
    /// Function that labels every leaf by its path.
    labels: Arc<dyn Fn(&ParameterPath) -> String + Send + Sync>,

    /// Labels that have an optimizer, in the order in which their optimizers are applied.
    group_labels: Vec<String>,

    /// [`Masked`] optimizers of the labels, chained in the order of [`Self::group_labels`].
    groups: G,
}

impl<O> Partition<Masked<O>> {
    /// Creates a new [`Partition`] that labels every leaf with `labels` and applies `optimizer` to the leaves with the
    /// label `label`. Optimizers for other labels are added with [`Self::with_group`].
    pub fn new<Labels: 'static + Fn(&ParameterPath) -> String + Send + Sync>(
        labels: Labels,
        label: impl Into<String>,
        optimizer: O,
    ) -> Self {
        let labels: Arc<dyn Fn(&ParameterPath) -> String + Send + Sync> = Arc::new(labels);
        let label = label.into();
        let groups = Self::group(&labels, label.clone(), optimizer);
        Self { labels, group_labels: vec![label], groups }
    }
}

impl<O> Partition<Chain<Masked<O>, Masked<SetToZero>>> {
    /// Creates a new [`SelectiveTransform`] that freezes the leaves whose paths satisfy `freeze_mask` and applies
    /// `optimizer` to all other leaves, which is the analogue of
    /// [`optax.selective_transform`](
    /// https://optax.readthedocs.io/en/latest/api/optimizer_wrappers.html#optax.selective_transform).
    /// The leaves are labeled `train` and `freeze`, in that order.
    pub fn selective<Mask: 'static + Fn(&ParameterPath) -> bool + Send + Sync>(
        optimizer: O,
        freeze_mask: Mask,
    ) -> Self {
        let labels = move |path: &ParameterPath| if freeze_mask(path) { "freeze" } else { "train" }.to_string();
        Partition::new(labels, "train", optimizer).with_group("freeze", SetToZero).unwrap()
    }
}

impl<G> Partition<G> {
    /// Returns this partition with `optimizer` applied to the leaves with the label `label`, after the optimizers of
    /// all other labels.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::InvalidConfiguration`] when `label` already has an optimizer.
    pub fn with_group<O>(
        self,
        label: impl Into<String>,
        optimizer: O,
    ) -> Result<Partition<Chain<G, Masked<O>>>, OptimizerError> {
        let label = label.into();
        if self.group_labels.contains(&label) {
            return Err(OptimizerError::InvalidConfiguration {
                message: format!("the label `{label}` already has an optimizer"),
            });
        }
        let group = Self::group(&self.labels, label.clone(), optimizer);
        let mut group_labels = self.group_labels;
        group_labels.push(label);
        Ok(Partition { labels: self.labels, group_labels, groups: Chain::new(self.groups, group) })
    }

    /// Returns the labels that have an optimizer, in the order in which their optimizers are applied.
    #[inline]
    pub fn group_labels(&self) -> &[String] {
        &self.group_labels
    }

    /// Returns the [`Chain`] of the [`Masked`] optimizers of the labels.
    #[inline]
    pub fn groups(&self) -> &G {
        &self.groups
    }

    /// Returns the label of the leaf at `path`.
    #[inline]
    pub fn label(&self, path: &ParameterPath) -> String {
        (self.labels)(path)
    }

    /// Returns the [`Masked`] optimizer that applies `optimizer` to the leaves that `labels` labels with `label`.
    fn group<O>(
        labels: &Arc<dyn Fn(&ParameterPath) -> String + Send + Sync>,
        label: String,
        optimizer: O,
    ) -> Masked<O> {
        let labels = labels.clone();
        Masked::new(optimizer, move |path| labels(path) == label)
    }

    /// Validates that every leaf of `structure` carries a label that has an optimizer.
    fn validate_labels<P: Parameter, S: Parameterized<P>>(&self, structure: &S) -> Result<(), OptimizerError> {
        for path in structure.parameter_paths() {
            let label = self.label(&path);
            if !self.group_labels.contains(&label) {
                return Err(OptimizerError::InvalidConfiguration {
                    message: format!(
                        "the leaf at `{path}` has the label `{label}`, which has no optimizer (the labels with \
                         optimizers are {:?})",
                        self.group_labels,
                    ),
                });
            }
        }
        Ok(())
    }
}

impl<G: Debug> Debug for Partition<G> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The labeling function is opaque, so it is not rendered.
        formatter
            .debug_struct("Partition")
            .field("group_labels", &self.group_labels)
            .field("groups", &self.groups)
            .finish_non_exhaustive()
    }
}

impl<P: OptimizerValue, G: Optimizer<P>> Optimizer<P> for Partition<G> {
    type State<S: Parameterized<P>> = PartitionState<P, G::State<S>>;

    fn initialize<S: Parameterized<P>>(&self, parameters: &S) -> Result<Self::State<S>, OptimizerError> {
        self.validate_labels(parameters)?;
        Ok(PartitionState { inner_states: self.groups.initialize(parameters)?, leaf: PhantomData })
    }

    fn update<S: Parameterized<P>>(
        &self,
        gradients: S,
        state: Self::State<S>,
        parameters: &S,
        arguments: &UpdateArguments<'_, P, S>,
    ) -> Result<(S, Self::State<S>), OptimizerError> {
        self.validate_labels(&gradients)?;
        let (updates, inner_states) = self.groups.update(gradients, state.inner_states, parameters, arguments)?;
        Ok((updates, PartitionState { inner_states, leaf: PhantomData }))
    }
}

/// [`Partition`] that freezes the leaves that a predicate on their paths selects and applies an optimizer to all other
/// leaves, which [`Partition::selective`] creates. It is the analogue of
/// [`optax.selective_transform`](
/// https://optax.readthedocs.io/en/latest/api/optimizer_wrappers.html#optax.selective_transform).
pub type SelectiveTransform<O> = Partition<Chain<Masked<O>, Masked<SetToZero>>>;

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use pretty_assertions::assert_eq;

    use ryft_core::{Add, Array};

    use crate::tests::{ArgumentProbe, assert_parity, parity_updates, state_paths};
    use crate::{Adam, Scale, Sgd, Trace};

    use super::*;

    /// Returns the label of a leaf of the parameters of [`state_paths`], which is the name of the leaf.
    fn leaf_name(path: &ParameterPath) -> String {
        if path.to_string().contains("weights") { "weights" } else { "bias" }.to_string()
    }

    #[test]
    fn test_partition_new() {
        let partition = Partition::new(leaf_name, "weights", Trace::new(0.5).unwrap());
        assert_eq!(partition.group_labels(), &["weights".to_string()]);
        assert_eq!(partition.groups().inner(), &Trace::new(0.5).unwrap());
        assert_eq!(partition.label(&ParameterPath::root().key("weights")), "weights");
    }

    #[test]
    fn test_partition_selective() {
        // The frozen leaf `$.0` gets a zero update, and the scaling is applied to the trained leaf `$.1`.
        let partition = Partition::selective(Scale::new(2.0).unwrap(), |path| path.to_string() == "$.0");
        assert_eq!(partition.group_labels(), &["train".to_string(), "freeze".to_string()]);
        let parameters = (Array::scalar(1.0f32).unwrap(), Array::scalar(2.0f32).unwrap());
        let gradients = (Array::scalar(3.0f32).unwrap(), Array::scalar(4.0f32).unwrap());
        let state = partition.initialize(&parameters).unwrap();
        let (updates, _) = partition.update(gradients, state, &parameters, &UpdateArguments::new()).unwrap();
        assert_eq!(updates, (Array::scalar(0f32).unwrap(), Array::scalar(8.0f32).unwrap()));
    }

    #[test]
    fn test_partition_with_group() {
        let partition = Partition::new(leaf_name, "weights", Trace::new(0.5).unwrap())
            .with_group("bias", Scale::new(2.0).unwrap())
            .unwrap();
        assert_eq!(partition.group_labels(), &["weights".to_string(), "bias".to_string()]);
        assert_eq!(partition.groups().second().inner(), &Scale::new(2.0).unwrap());
        assert!(format!("{partition:?}").starts_with("Partition { group_labels: [\"weights\", \"bias\"], groups: "));
        assert!(matches!(
            partition.with_group("bias", Scale::new(3.0).unwrap()),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "the label `bias` already has an optimizer",
        ));
    }

    #[test]
    fn test_partition_initialize() {
        // Every optimizer holds statistics for the leaves of its label only, and unlabeled leaves are rejected.
        let partition = Partition::new(leaf_name, "weights", Trace::new(0.5).unwrap())
            .with_group("bias", Trace::new(0.9).unwrap())
            .unwrap();
        assert_eq!(
            state_paths(&partition),
            vec![
                "$.inner_states.0.inner_state.trace[$[\"weights\"]]",
                "$.inner_states.1.inner_state.trace[$[\"bias\"]]",
            ],
        );
        let partition = Partition::new(leaf_name, "weights", Trace::new(0.5).unwrap());
        let parameters = BTreeMap::from([("bias", Array::scalar(1.0f32).unwrap())]);
        assert!(matches!(
            partition.initialize(&parameters),
            Err(OptimizerError::InvalidConfiguration { message })
                if message == "the leaf at `$[\"bias\"]` has the label `bias`, which has no optimizer (the labels \
                               with optimizers are [\"weights\"])",
        ));
    }

    #[test]
    fn test_partition_update() {
        // With three labels, `$.0` is scaled by `2`, `$.1` gets the traces `1` and `1 + 0.5 · 1 = 1.5` of its gradients
        // `1` and `1`, and `$.2` is scaled by `-1`.
        let labels = |path: &ParameterPath| path.to_string();
        let partition = Partition::new(labels, "$.0", Scale::new(2.0).unwrap())
            .with_group("$.1", Trace::new(0.5).unwrap())
            .unwrap()
            .with_group("$.2", Scale::new(-1.0).unwrap())
            .unwrap();
        let parameters =
            (Array::scalar(0.0f32).unwrap(), Array::scalar(0.0f32).unwrap(), Array::scalar(0.0f32).unwrap());
        let gradients =
            (Array::scalar(1.0f32).unwrap(), Array::scalar(1.0f32).unwrap(), Array::scalar(1.0f32).unwrap());
        let arguments = UpdateArguments::new();
        let state = partition.initialize(&parameters).unwrap();
        assert_eq!(
            state.named_parameters().map(|(path, _)| path.to_string()).collect::<Vec<_>>(),
            vec!["$.inner_states.0.1.inner_state.trace[$.1]"],
        );
        let (updates, state) = partition.update(gradients.clone(), state, &parameters, &arguments).unwrap();
        assert_eq!(
            updates,
            (Array::scalar(2.0f32).unwrap(), Array::scalar(1.0f32).unwrap(), Array::scalar(-1.0f32).unwrap()),
        );
        let (updates, _) = partition.update(gradients, state, &parameters, &arguments).unwrap();
        assert_eq!(
            updates,
            (Array::scalar(2.0f32).unwrap(), Array::scalar(1.5f32).unwrap(), Array::scalar(-1.0f32).unwrap()),
        );
    }

    #[test]
    fn test_partition_update_arguments() {
        // The probe of `$.1` sees its argument gradient `20`, the value `0.5`, and the objective `1 + 2` at the full
        // parameters.
        let partition = Partition::selective(ArgumentProbe, |path| path.to_string() == "$.0");
        let parameters = (Array::scalar(1.0f32).unwrap(), Array::scalar(2.0f32).unwrap());
        let value = Array::scalar(0.5f32).unwrap();
        let argument_gradients = (Array::scalar(10.0f32).unwrap(), Array::scalar(20.0f32).unwrap());
        let objective = |parameters: &(Array, Array)| -> Result<Array, OptimizerError> {
            Ok(Add::add(&parameters.0, &parameters.1)?)
        };
        let arguments = UpdateArguments::new()
            .with_value(&value)
            .with_gradients(&argument_gradients)
            .with_objective(&objective);
        let state = partition.initialize(&parameters).unwrap();
        let (updates, _) = partition.update(parameters.clone(), state, &parameters, &arguments).unwrap();
        assert_eq!(updates, (Array::scalar(0f32).unwrap(), Array::scalar(23.5f32).unwrap()));
    }

    #[test]
    fn test_partition_optax_parity() {
        // Optax 9ecde04b: `optax.partition({'a': optax.sgd(0.1, momentum=0.9), 'b': optax.adam(0.05)},
        // {'bias': 'b', 'weights': 'a'})`.
        const EXPECTED: [[f32; 9]; 4] = [
            [
                -0.049999665,
                0.049999665,
                -0.049999665,
                -0.040000003,
                0.010000001,
                -0.020000001,
                0.030000001,
                -0.05,
                0.060000002,
            ],
            [
                -0.013316716,
                -0.0081365295,
                -0.043552753,
                -0.056,
                -0.021000002,
                0.022000002,
                0.017000003,
                -0.024999999,
                0.019000003,
            ],
            [
                -0.033524904,
                0.0035256834,
                -0.011361218,
                -0.0354,
                -0.0239,
                -0.025199998,
                -0.009699999,
                -0.0625,
                0.027100002,
            ],
            [-0.03868749, -0.009237449, 0.013068931, -0.04186, 0.008490003, -0.027679998, 0.03627, -0.07625, -0.00561],
        ];
        let labels = |path: &ParameterPath| if path.to_string().contains("weights") { "a" } else { "b" }.to_string();
        let optimizer = Partition::new(labels, "a", Sgd::new(0.1).with_momentum(0.9).unwrap())
            .with_group("b", Adam::new(0.05))
            .unwrap();
        assert_parity(&parity_updates(&optimizer), &EXPECTED);
    }

    #[test]
    fn test_selective_transform_optax_parity() {
        // Optax 9ecde04b: `optax.transforms.selective_transform(optax.adam(0.1),
        // freeze_mask={'bias': True, 'weights': False})`.
        const EXPECTED: [[f32; 9]; 4] = [
            [0.0, 0.0, 0.0, -0.09999934, 0.09999933, -0.09999933, 0.09999933, -0.09999933, 0.09999933],
            [0.0, 0.0, 0.0, -0.09321702, -0.04941849, 0.036609985, 0.040021453, -0.03456023, 0.020361768],
            [0.0, 0.0, 0.0, -0.047980227, -0.047713544, -0.025381634, -0.015380538, -0.05955345, 0.024689391],
            [0.0, 0.0, 0.0, -0.05051261, 0.011251225, -0.025295947, 0.034904275, -0.063364856, -0.0042764237],
        ];
        let optimizer = Partition::selective(Adam::new(0.1), |path| path.to_string().contains("bias"));
        assert_parity(&parity_updates(&optimizer), &EXPECTED);
    }
}
