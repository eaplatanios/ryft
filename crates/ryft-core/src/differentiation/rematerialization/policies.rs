//! Built-in [`ResidualPolicy`]s of [`rematerialize`](super::rematerialize), which are the analogues of the
//! [JAX checkpoint policies](https://docs.jax.dev/en/latest/gradient-checkpointing.html#list-of-policies) with
//! full-word names, together with [`PolicyFn`] for policies defined by closures and [`MemoryTransferStorage`] for
//! policies that offload the residuals that they save.
//!
//! Every built-in policy recognizes the producers of a residual by their payload operations (refer to
//! [`ResidualProducer::payload`]) rather than by their operation family, so the same policy works for every family,
//! including composite and backend families that hold array operations through projected members. A residual that
//! several producers may produce (e.g., the corresponding outputs of the two branches of a `condition`) matches a
//! policy when any of its producers does.
//!
//! The built-in policies are generic over the type universe and declare their instantiations for the array universes
//! [`ArrayType`] and [`ArrayIrType`] (refer to [`ResidualPolicy::native_instantiations`]), so that promoting a staged
//! rematerialization from one of these universes to the other re-instantiates its policy rather than projecting the
//! types of its candidates.

// TODO(eaplatanios): Review this module.

use std::borrow::Cow;
use std::fmt::Debug;
use std::marker::PhantomData;

use crate::arrays::{ArrayIrType, ArrayType, Memory};
use crate::operations::{DotOperation, TagOperation, TransferToMemoryOperation};
use crate::partial::{
    ErasedResidualStorage, NativeResidualPolicies, NoStorage, ResidualCandidate, ResidualDecision, ResidualPolicy,
    ResidualPolicyError, ResidualProducer, ResidualRejection, ResidualStorage,
};
use crate::programs::{ErasedOperation, ProgramError, Type};

/// Canonical policy name for [`NothingSavable`].
pub const NOTHING_SAVABLE_POLICY_NAME: &str = "nothing_savable";

/// [`ResidualPolicy`] that saves nothing, so that differentiation recomputes every residual from the inputs of the
/// rematerialized function. This is the default policy of [`rematerialize`](super::rematerialize). This is the Ryft
/// analogue of JAX's
/// [`nothing_saveable`](https://docs.jax.dev/en/latest/_autosummary/jax.checkpoint_policies.nothing_saveable.html).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct NothingSavable;

impl<T: 'static + Type> ResidualPolicy<T> for NothingSavable {
    type Storage = NoStorage;

    #[inline]
    fn name(&self) -> &str {
        NOTHING_SAVABLE_POLICY_NAME
    }

    #[inline]
    fn classify(
        &self,
        _candidate: &ResidualCandidate<'_, T>,
    ) -> Result<ResidualDecision<NoStorage>, ResidualRejection> {
        Ok(ResidualDecision::Recompute)
    }

    #[inline]
    fn native_instantiations(&self) -> NativeResidualPolicies {
        NativeResidualPolicies::default()
            .with::<ArrayType, _>(self.clone())
            .with::<ArrayIrType, _>(self.clone())
    }
}

/// Canonical policy name for [`EverythingSavable`].
pub const EVERYTHING_SAVABLE_POLICY_NAME: &str = "everything_savable";

/// [`ResidualPolicy`] that saves every residual, so that differentiation recomputes nothing. This is the Ryft analogue
/// of JAX's
/// [`everything_saveable`](https://docs.jax.dev/en/latest/_autosummary/jax.checkpoint_policies.everything_saveable.html).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct EverythingSavable;

impl<T: 'static + Type> ResidualPolicy<T> for EverythingSavable {
    type Storage = NoStorage;

    #[inline]
    fn name(&self) -> &str {
        EVERYTHING_SAVABLE_POLICY_NAME
    }

    #[inline]
    fn classify(
        &self,
        _candidate: &ResidualCandidate<'_, T>,
    ) -> Result<ResidualDecision<NoStorage>, ResidualRejection> {
        Ok(ResidualDecision::Save)
    }

    #[inline]
    fn native_instantiations(&self) -> NativeResidualPolicies {
        NativeResidualPolicies::default()
            .with::<ArrayType, _>(self.clone())
            .with::<ArrayIrType, _>(self.clone())
    }
}

/// Canonical policy name for [`DotsSavable`].
pub const DOTS_SAVABLE_POLICY_NAME: &str = "dots_savable";

/// [`ResidualPolicy`] that saves the residuals that [`DotOperation`]s produce and recomputes every other residual.
/// This is the Ryft analogue of JAX's
/// [`dots_saveable`](https://docs.jax.dev/en/latest/_autosummary/jax.checkpoint_policies.dots_saveable.html#jax.checkpoint_policies.dots_saveable).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct DotsSavable;

impl<T: 'static + Type> ResidualPolicy<T> for DotsSavable {
    type Storage = NoStorage;

    #[inline]
    fn name(&self) -> &str {
        DOTS_SAVABLE_POLICY_NAME
    }

    fn classify(&self, candidate: &ResidualCandidate<'_, T>) -> Result<ResidualDecision<NoStorage>, ResidualRejection> {
        let saved = candidate.producers().iter().any(|producer| producer.payload::<DotOperation>().is_some());
        Ok(match saved {
            true => ResidualDecision::Save,
            false => ResidualDecision::Recompute,
        })
    }

    #[inline]
    fn native_instantiations(&self) -> NativeResidualPolicies {
        NativeResidualPolicies::default()
            .with::<ArrayType, _>(self.clone())
            .with::<ArrayIrType, _>(self.clone())
    }
}

/// Canonical policy name for [`DotsWithNoBatchDimensionsSavable`].
pub const DOTS_WITH_NO_BATCH_DIMENSIONS_SAVABLE_POLICY_NAME: &str = "dots_with_no_batch_dimensions_savable";

/// [`ResidualPolicy`] that saves the residuals that [`DotOperation`]s without batching dimensions (e.g., matrix
/// multiplications) produce and recomputes every other residual, which is the analogue of JAX's
/// `dots_with_no_batch_dims_savable`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct DotsWithNoBatchDimensionsSavable;

impl<T: 'static + Type> ResidualPolicy<T> for DotsWithNoBatchDimensionsSavable {
    type Storage = NoStorage;

    #[inline]
    fn name(&self) -> &str {
        DOTS_WITH_NO_BATCH_DIMENSIONS_SAVABLE_POLICY_NAME
    }

    fn classify(&self, candidate: &ResidualCandidate<'_, T>) -> Result<ResidualDecision<NoStorage>, ResidualRejection> {
        let saved = candidate.producers().iter().filter_map(|producer| producer.payload::<DotOperation>()).any(|dot| {
            let dimensions = dot.dimensions();
            dimensions.lhs_batching_dimensions().is_empty() && dimensions.rhs_batching_dimensions().is_empty()
        });
        Ok(match saved {
            true => ResidualDecision::Save,
            false => ResidualDecision::Recompute,
        })
    }

    #[inline]
    fn native_instantiations(&self) -> NativeResidualPolicies {
        NativeResidualPolicies::default()
            .with::<ArrayType, _>(self.clone())
            .with::<ArrayIrType, _>(self.clone())
    }
}

/// Canonical policy name for [`OffloadDotsWithNoBatchDimensions`].
pub const OFFLOAD_DOTS_WITH_NO_BATCH_DIMENSIONS_POLICY_NAME: &str = "offload_dots_with_no_batch_dimensions";

/// [`ResidualPolicy`] that saves the residuals that [`DotOperation`]s without batching dimensions produce by
/// offloading them to the provided [`Memory`] (refer to [`MemoryTransferStorage`]) and recomputes every other residual,
/// which is the analogue of JAX's `offload_dot_with_no_batch_dims`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct OffloadDotsWithNoBatchDimensions {
    /// [`Memory`] that the saved residuals are offloaded to.
    destination: Memory,
}

impl OffloadDotsWithNoBatchDimensions {
    /// Creates a new [`OffloadDotsWithNoBatchDimensions`] policy that offloads the residuals that it saves to
    /// `destination`.
    #[inline]
    pub fn new(destination: Memory) -> Self {
        Self { destination }
    }

    /// Returns the [`Memory`] that the saved residuals are offloaded to.
    #[inline]
    pub fn destination(&self) -> Memory {
        self.destination
    }
}

impl<T: 'static + Type> ResidualPolicy<T> for OffloadDotsWithNoBatchDimensions
where
    for<'t> &'t ArrayType: TryFrom<&'t T>,
{
    type Storage = MemoryTransferStorage;

    #[inline]
    fn name(&self) -> &str {
        OFFLOAD_DOTS_WITH_NO_BATCH_DIMENSIONS_POLICY_NAME
    }

    fn classify(
        &self,
        candidate: &ResidualCandidate<'_, T>,
    ) -> Result<ResidualDecision<MemoryTransferStorage>, ResidualRejection> {
        let saved = candidate.producers().iter().filter_map(|producer| producer.payload::<DotOperation>()).any(|dot| {
            let dimensions = dot.dimensions();
            dimensions.lhs_batching_dimensions().is_empty() && dimensions.rhs_batching_dimensions().is_empty()
        });
        Ok(match saved {
            true => ResidualDecision::SaveWith(MemoryTransferStorage::new(self.destination)),
            false => ResidualDecision::Recompute,
        })
    }

    #[inline]
    fn native_instantiations(&self) -> NativeResidualPolicies {
        NativeResidualPolicies::default()
            .with::<ArrayType, _>(self.clone())
            .with::<ArrayIrType, _>(self.clone())
    }
}

/// Canonical policy name for [`SaveOnlyTheseNames`].
pub const SAVE_ONLY_THESE_NAMES_POLICY_NAME: &str = "save_only_these_names";

/// [`ResidualPolicy`] that saves the residuals that are tagged (refer to [`Tag`](crate::Tag)) with one of the provided
/// names and recomputes every other residual, which is the analogue of JAX's `save_only_these_names`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SaveOnlyTheseNames {
    /// Names of the tags whose values are saved.
    names: Vec<String>,
}

impl SaveOnlyTheseNames {
    /// Creates a new [`SaveOnlyTheseNames`] policy that saves the values that are tagged with one of `names`.
    #[inline]
    pub fn new<N: Into<String>, I: IntoIterator<Item = N>>(names: I) -> Self {
        Self { names: names.into_iter().map(Into::into).collect() }
    }

    /// Returns the names of the tags whose values are saved.
    #[inline]
    pub fn names(&self) -> &[String] {
        self.names.as_slice()
    }
}

impl<T: 'static + Type> ResidualPolicy<T> for SaveOnlyTheseNames {
    type Storage = NoStorage;

    #[inline]
    fn name(&self) -> &str {
        SAVE_ONLY_THESE_NAMES_POLICY_NAME
    }

    fn classify(&self, candidate: &ResidualCandidate<'_, T>) -> Result<ResidualDecision<NoStorage>, ResidualRejection> {
        let saved = candidate
            .producers()
            .iter()
            .filter_map(|producer| TagOperation::<T>::key_of(producer.operation()))
            .any(|key| self.names.iter().any(|name| name == key));
        Ok(if saved { ResidualDecision::Save } else { ResidualDecision::Recompute })
    }

    #[inline]
    fn native_instantiations(&self) -> NativeResidualPolicies {
        NativeResidualPolicies::default()
            .with::<ArrayType, _>(self.clone())
            .with::<ArrayIrType, _>(self.clone())
    }
}

/// Canonical policy name for [`SaveAnyNamesButThese`].
pub const SAVE_ANY_NAMES_BUT_THESE_POLICY_NAME: &str = "save_any_names_but_these";

/// [`ResidualPolicy`] that saves the residuals that are tagged (refer to [`Tag`](crate::Tag)) with any name other than
/// the provided ones and recomputes every other residual, including untagged ones, which is the analogue of JAX's
/// `save_any_names_but_these`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SaveAnyNamesButThese {
    /// Names of the tags whose values are not saved.
    names: Vec<String>,
}

impl SaveAnyNamesButThese {
    /// Creates a new [`SaveAnyNamesButThese`] policy that saves the tagged values whose tag names are not in `names`.
    #[inline]
    pub fn new<N: Into<String>, I: IntoIterator<Item = N>>(names: I) -> Self {
        Self { names: names.into_iter().map(Into::into).collect() }
    }

    /// Returns the names of the tags whose values are not saved.
    #[inline]
    pub fn names(&self) -> &[String] {
        self.names.as_slice()
    }
}

impl<T: 'static + Type> ResidualPolicy<T> for SaveAnyNamesButThese {
    type Storage = NoStorage;

    #[inline]
    fn name(&self) -> &str {
        SAVE_ANY_NAMES_BUT_THESE_POLICY_NAME
    }

    fn classify(&self, candidate: &ResidualCandidate<'_, T>) -> Result<ResidualDecision<NoStorage>, ResidualRejection> {
        let saved = candidate
            .producers()
            .iter()
            .filter_map(|producer| TagOperation::<T>::key_of(producer.operation()))
            .any(|key| !self.names.iter().any(|name| name == key));
        Ok(if saved { ResidualDecision::Save } else { ResidualDecision::Recompute })
    }

    #[inline]
    fn native_instantiations(&self) -> NativeResidualPolicies {
        NativeResidualPolicies::default()
            .with::<ArrayType, _>(self.clone())
            .with::<ArrayIrType, _>(self.clone())
    }
}

/// Canonical policy name for [`SaveAnythingExceptTheseNames`].
pub const SAVE_ANYTHING_EXCEPT_THESE_NAMES_POLICY_NAME: &str = "save_anything_except_these_names";

/// [`ResidualPolicy`] that saves every residual except the ones that are tagged (refer to [`Tag`](crate::Tag)) with
/// one of the provided names, which is the analogue of JAX's `save_anything_except_these_names`. Unlike
/// [`SaveAnyNamesButThese`], it also saves untagged residuals.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SaveAnythingExceptTheseNames {
    /// Names of the tags whose values are not saved.
    names: Vec<String>,
}

impl SaveAnythingExceptTheseNames {
    /// Creates a new [`SaveAnythingExceptTheseNames`] policy that saves every value except the ones that are tagged
    /// with one of `names`.
    #[inline]
    pub fn new<N: Into<String>, I: IntoIterator<Item = N>>(names: I) -> Self {
        Self { names: names.into_iter().map(Into::into).collect() }
    }

    /// Returns the names of the tags whose values are not saved.
    #[inline]
    pub fn names(&self) -> &[String] {
        self.names.as_slice()
    }
}

impl<T: 'static + Type> ResidualPolicy<T> for SaveAnythingExceptTheseNames {
    type Storage = NoStorage;

    #[inline]
    fn name(&self) -> &str {
        SAVE_ANYTHING_EXCEPT_THESE_NAMES_POLICY_NAME
    }

    fn classify(&self, candidate: &ResidualCandidate<'_, T>) -> Result<ResidualDecision<NoStorage>, ResidualRejection> {
        let saved = candidate.producers().iter().any(|producer| {
            TagOperation::<T>::key_of(producer.operation()).is_none_or(|key| !self.names.iter().any(|name| name == key))
        });
        Ok(if saved { ResidualDecision::Save } else { ResidualDecision::Recompute })
    }

    #[inline]
    fn native_instantiations(&self) -> NativeResidualPolicies {
        NativeResidualPolicies::default()
            .with::<ArrayType, _>(self.clone())
            .with::<ArrayIrType, _>(self.clone())
    }
}

/// Canonical policy name for [`SaveAndOffloadOnlyTheseNames`].
pub const SAVE_AND_OFFLOAD_ONLY_THESE_NAMES_POLICY_NAME: &str = "save_and_offload_only_these_names";

/// [`ResidualPolicy`] that saves the residuals that are tagged (refer to [`Tag`](crate::Tag)) with one of the provided
/// savable names, offloads the ones that are tagged with one of the provided offloadable names to the provided
/// [`Memory`] (refer to [`MemoryTransferStorage`]), and recomputes every other residual, which is the analogue of JAX's
/// `save_and_offload_only_these_names`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SaveAndOffloadOnlyTheseNames {
    /// Names of the tags whose values are saved.
    savable_names: Vec<String>,

    /// Names of the tags whose values are offloaded.
    offloadable_names: Vec<String>,

    /// [`Memory`] that the offloaded values are transferred to.
    destination: Memory,
}

impl SaveAndOffloadOnlyTheseNames {
    /// Creates a new [`SaveAndOffloadOnlyTheseNames`] policy that saves the values that are tagged with one of
    /// `savable_names` and offloads the ones that are tagged with one of `offloadable_names` to `destination`.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::InvalidArgument`] when a name is both savable and offloadable, because a value cannot
    /// be both kept in place and offloaded.
    pub fn new<S: Into<String>, O: Into<String>>(
        savable_names: impl IntoIterator<Item = S>,
        offloadable_names: impl IntoIterator<Item = O>,
        destination: Memory,
    ) -> Result<Self, ProgramError> {
        let savable_names = savable_names.into_iter().map(Into::into).collect::<Vec<String>>();
        let offloadable_names = offloadable_names.into_iter().map(Into::into).collect::<Vec<String>>();
        let overlapping_names = savable_names
            .iter()
            .filter(|name| offloadable_names.contains(name))
            .map(|name| format!("`{name}`"))
            .collect::<Vec<_>>();
        if !overlapping_names.is_empty() {
            return Err(ProgramError::InvalidArgument {
                message: format!(
                    "names {} cannot be both savable and offloadable by a \
                     `{SAVE_AND_OFFLOAD_ONLY_THESE_NAMES_POLICY_NAME}` policy",
                    overlapping_names.join(", "),
                ),
            });
        }
        Ok(Self { savable_names, offloadable_names, destination })
    }

    /// Returns the names of the tags whose values are saved.
    #[inline]
    pub fn savable_names(&self) -> &[String] {
        self.savable_names.as_slice()
    }

    /// Returns the names of the tags whose values are offloaded.
    #[inline]
    pub fn offloadable_names(&self) -> &[String] {
        self.offloadable_names.as_slice()
    }

    /// Returns the [`Memory`] that the offloaded values are transferred to.
    #[inline]
    pub fn destination(&self) -> Memory {
        self.destination
    }
}

impl<T: 'static + Type> ResidualPolicy<T> for SaveAndOffloadOnlyTheseNames
where
    for<'t> &'t ArrayType: TryFrom<&'t T>,
{
    type Storage = MemoryTransferStorage;

    #[inline]
    fn name(&self) -> &str {
        SAVE_AND_OFFLOAD_ONLY_THESE_NAMES_POLICY_NAME
    }

    fn classify(
        &self,
        candidate: &ResidualCandidate<'_, T>,
    ) -> Result<ResidualDecision<MemoryTransferStorage>, ResidualRejection> {
        let keys = candidate
            .producers()
            .iter()
            .filter_map(|producer| TagOperation::<T>::key_of(producer.operation()))
            .collect::<Vec<_>>();
        let tagged_with = |names: &[String]| keys.iter().any(|key| names.iter().any(|name| name == key));
        Ok(if tagged_with(&self.savable_names) {
            ResidualDecision::Save
        } else if tagged_with(&self.offloadable_names) {
            ResidualDecision::SaveWith(MemoryTransferStorage::new(self.destination))
        } else {
            ResidualDecision::Recompute
        })
    }

    #[inline]
    fn native_instantiations(&self) -> NativeResidualPolicies {
        NativeResidualPolicies::default()
            .with::<ArrayType, _>(self.clone())
            .with::<ArrayIrType, _>(self.clone())
    }
}

/// Canonical policy name for [`SaveFromBothPolicies`].
pub const SAVE_FROM_BOTH_POLICIES_POLICY_NAME: &str = "save_from_both_policies";

/// [`ResidualPolicy`] that combines two policies, saving each residual that either of them saves, which is the analogue
/// of JAX's `save_from_both_policies`. The first policy classifies each residual first, and the second policy is only
/// consulted for the residuals that the first one recomputes. A residual that the first policy saves through a
/// [`ResidualStorage`] therefore keeps that storage, which makes this a strict superset of JAX's policy, whose
/// combination of two policies rejects offloading. Rejections of either policy are returned as they are.
///
/// Unlike the other built-in policies, this policy declares no instantiations in other type universes, because its
/// two policies may be defined for one universe only (e.g., [`PolicyFn`]s). Promoting a staged rematerialization that
/// uses it to another universe therefore projects the types of its candidates (refer to
/// [`ResidualPolicyReference::lift`](crate::ResidualPolicyReference::lift)).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SaveFromBothPolicies<P1, P2> {
    /// Policy that classifies each residual first.
    first: P1,

    /// Policy that classifies the residuals that `first` recomputes.
    second: P2,
}

impl<P1, P2> SaveFromBothPolicies<P1, P2> {
    /// Creates a new [`SaveFromBothPolicies`] policy that saves each residual that `first` or `second` saves.
    #[inline]
    pub fn new(first: P1, second: P2) -> Self {
        Self { first, second }
    }

    /// Returns the policy that classifies each residual first.
    #[inline]
    pub fn first(&self) -> &P1 {
        &self.first
    }

    /// Returns the policy that classifies the residuals that [`first`](Self::first) recomputes.
    #[inline]
    pub fn second(&self) -> &P2 {
        &self.second
    }
}

impl<T: 'static + Type, P1: ResidualPolicy<T>, P2: ResidualPolicy<T>> ResidualPolicy<T>
    for SaveFromBothPolicies<P1, P2>
{
    type Storage = ErasedResidualStorage<T>;

    #[inline]
    fn name(&self) -> &str {
        SAVE_FROM_BOTH_POLICIES_POLICY_NAME
    }

    fn classify(
        &self,
        candidate: &ResidualCandidate<'_, T>,
    ) -> Result<ResidualDecision<ErasedResidualStorage<T>>, ResidualRejection> {
        match self.first.classify(candidate)?.into_erased() {
            ResidualDecision::Recompute => Ok(self.second.classify(candidate)?.into_erased()),
            decision => Ok(decision),
        }
    }
}

/// Default name for [`PolicyFn`].
pub const POLICY_FN_POLICY_NAME: &str = "policy_fn";

/// [`ResidualPolicy`] whose classifier is the provided closure, for policies that the built-in ones cannot express.
/// The closure receives each candidate residual and returns its decision, possibly with a [`ResidualStorage`] of type
/// `S`, or a rejection that forbids every placement of the residual. Policies that need different storages for
/// different residuals can return [`ErasedResidualStorage`]s.
///
/// A [`PolicyFn`] is defined for the type universe of the candidates that its closure accepts and declares no
/// instantiations in other universes. Promoting a staged rematerialization that uses it to another universe therefore
/// projects the types of its candidates (refer to
/// [`ResidualPolicyReference::lift`](crate::ResidualPolicyReference::lift)).
///
/// # Examples
///
/// ```rust
/// # use ryft_core::differentiation::rematerialization::PolicyFn;
/// # use ryft_core::{ArrayType, NoStorage, ResidualDecision, ResidualRejection};
/// // Saves the residuals that have one producer and recomputes the ones that several producers may produce.
/// let policy = PolicyFn::new::<ArrayType>(|candidate| {
///     Ok::<_, ResidualRejection>(match candidate.producers().len() {
///         1 => ResidualDecision::<NoStorage>::Save,
///         _ => ResidualDecision::Recompute,
///     })
/// })
/// .with_name("save_unique_producers");
/// # let _ = policy;
/// ```
pub struct PolicyFn<F, S = NoStorage> {
    /// Name of the policy.
    name: Cow<'static, str>,

    /// Closure that classifies each candidate residual.
    function: F,

    /// Storage of the decisions that the closure returns.
    marker: PhantomData<fn() -> S>,
}

impl<F, S> PolicyFn<F, S> {
    /// Creates a new [`PolicyFn`] named `policy_fn` whose classifier is `function`, which classifies candidates of the
    /// type universe `T`.
    #[inline]
    pub fn new<T: 'static + Type>(function: F) -> Self
    where
        F: Fn(&ResidualCandidate<'_, T>) -> Result<ResidualDecision<S>, ResidualRejection>,
    {
        Self { name: Cow::Borrowed(POLICY_FN_POLICY_NAME), function, marker: PhantomData }
    }

    /// Returns this [`PolicyFn`] with the provided name, which is used in diagnostics and in the rendering of the
    /// operations that carry the policy.
    #[inline]
    pub fn with_name<N: Into<Cow<'static, str>>>(mut self, name: N) -> Self {
        self.name = name.into();
        self
    }
}

impl<F: Clone, S> Clone for PolicyFn<F, S> {
    #[inline]
    fn clone(&self) -> Self {
        Self { name: self.name.clone(), function: self.function.clone(), marker: PhantomData }
    }
}

impl<F, S> Debug for PolicyFn<F, S> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("PolicyFn").field("name", &self.name).finish_non_exhaustive()
    }
}

impl<T: 'static + Type, S: ResidualStorage<T>, F> ResidualPolicy<T> for PolicyFn<F, S>
where
    F: 'static + Send + Sync + Fn(&ResidualCandidate<'_, T>) -> Result<ResidualDecision<S>, ResidualRejection>,
{
    type Storage = S;

    #[inline]
    fn name(&self) -> &str {
        self.name.as_ref()
    }

    #[inline]
    fn classify(&self, candidate: &ResidualCandidate<'_, T>) -> Result<ResidualDecision<S>, ResidualRejection> {
        (self.function)(candidate)
    }
}

/// [`ResidualStorage`] that offloads array residuals to the provided [`Memory`] with a [`TransferToMemoryOperation`],
/// and restores them by transferring them back to the memory of the residual. It supports every type universe whose
/// types project into [`ArrayType`] (e.g., [`ArrayType`] itself and [`ArrayIrType`]) and rejects residuals that are not
/// arrays (e.g., dimensions or references).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct MemoryTransferStorage {
    /// [`Memory`] that the residuals are offloaded to.
    destination: Memory,
}

impl MemoryTransferStorage {
    /// Creates a new [`MemoryTransferStorage`] that offloads residuals to `destination`.
    #[inline]
    pub fn new(destination: Memory) -> Self {
        Self { destination }
    }

    /// Returns the [`Memory`] that the residuals are offloaded to.
    #[inline]
    pub fn destination(&self) -> Memory {
        self.destination
    }
}

impl<T: 'static + Type> ResidualStorage<T> for MemoryTransferStorage
where
    for<'t> &'t ArrayType: TryFrom<&'t T>,
{
    #[inline]
    fn name(&self) -> String {
        "memory_transfer".to_owned()
    }

    fn store_payloads(&self, residual_type: &T) -> Result<Vec<ErasedOperation>, ResidualPolicyError> {
        <&ArrayType>::try_from(residual_type).map_err(|_| ResidualPolicyError::UnsupportedStorage {
            storage: ResidualStorage::<T>::name(self),
            residual_type: residual_type.to_string(),
            message: "the residual is not an array".to_owned(),
        })?;
        Ok(vec![ErasedOperation::new(TransferToMemoryOperation::new(self.destination))])
    }

    fn restore_payloads(
        &self,
        _stored_type: &T,
        residual_type: &T,
    ) -> Result<Vec<ErasedOperation>, ResidualPolicyError> {
        let residual_array_type =
            <&ArrayType>::try_from(residual_type).map_err(|_| ResidualPolicyError::UnsupportedStorage {
                storage: ResidualStorage::<T>::name(self),
                residual_type: residual_type.to_string(),
                message: "the residual is not an array".to_owned(),
            })?;
        Ok(vec![ErasedOperation::new(TransferToMemoryOperation::new(residual_array_type.memory()))])
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::{Array, ArrayIrOperation, ArrayOperation, DataType, DimensionBounds, DimensionType};
    use crate::operations::{DimensionSizeOperation, DotDimensionNumbers, SinOperation};
    use crate::partial::ResidualPolicyReference;

    use super::*;

    type TestOperation = ArrayIrOperation<Array>;

    /// Returns the [`ArrayIrType`] of `f64` scalars.
    fn scalar_type() -> ArrayIrType {
        ArrayType::scalar(DataType::F64).into()
    }

    /// Returns the [`ArrayIrType`] of dimensions named `n`, which does not project into [`ArrayType`].
    fn dimension_type() -> ArrayIrType {
        ArrayIrType::Dimension(DimensionType::new("n", DimensionBounds::non_negative(None).unwrap()))
    }

    /// Returns a dot product that contracts the leading dimensions of its inputs and has no batching dimensions.
    fn dot() -> TestOperation {
        ArrayOperation::<Array>::from(DotOperation::new(DotDimensionNumbers::new(vec![0], vec![0], vec![], vec![])))
            .into()
    }

    /// Returns a dot product that contracts the trailing dimensions of its inputs and batches their leading ones.
    fn batched_dot() -> TestOperation {
        ArrayOperation::<Array>::from(DotOperation::new(DotDimensionNumbers::new(vec![1], vec![1], vec![0], vec![0])))
            .into()
    }

    /// Returns a tag with the provided key.
    fn tag(key: &str) -> TestOperation {
        ArrayOperation::<Array>::Tag(TagOperation::new(key)).into()
    }

    /// Returns a sine, which is neither a dot product nor a tag.
    fn sine() -> TestOperation {
        ArrayOperation::<Array>::from(SinOperation::<ArrayType>::new()).into()
    }

    /// Returns a scalar candidate that output 0 of any of `operations` may produce. The built-in policies classify
    /// candidates by the payloads of their producers only, so the producers carry no input types.
    fn candidate(operations: &[TestOperation]) -> ResidualCandidate<'_, ArrayIrType> {
        let producers = operations
            .iter()
            .map(|operation| ResidualProducer::new(operation, 0, Vec::new(), vec![scalar_type()]))
            .collect();
        ResidualCandidate::new(producers, scalar_type())
    }

    #[test]
    fn test_nothing_savable() {
        assert_eq!(ResidualPolicy::<ArrayIrType>::name(&NothingSavable), "nothing_savable");
        assert_eq!(NothingSavable.classify(&candidate(&[dot()])), Ok(ResidualDecision::Recompute));
        assert_eq!(NothingSavable.classify(&candidate(&[tag("a")])), Ok(ResidualDecision::Recompute));
    }

    #[test]
    fn test_everything_savable() {
        assert_eq!(ResidualPolicy::<ArrayIrType>::name(&EverythingSavable), "everything_savable");
        assert_eq!(EverythingSavable.classify(&candidate(&[sine()])), Ok(ResidualDecision::Save));
        assert_eq!(EverythingSavable.classify(&candidate(&[dot()])), Ok(ResidualDecision::Save));
    }

    #[test]
    fn test_dots_savable() {
        assert_eq!(ResidualPolicy::<ArrayIrType>::name(&DotsSavable), "dots_savable");
        assert_eq!(DotsSavable.classify(&candidate(&[dot()])), Ok(ResidualDecision::Save));
        assert_eq!(DotsSavable.classify(&candidate(&[batched_dot()])), Ok(ResidualDecision::Save));
        assert_eq!(DotsSavable.classify(&candidate(&[sine()])), Ok(ResidualDecision::Recompute));

        // A residual that several producers may produce is saved when any of them is a dot product.
        assert_eq!(DotsSavable.classify(&candidate(&[sine(), dot()])), Ok(ResidualDecision::Save));

        // Dot products are recognized in the array operation family itself too.
        let dot = ArrayOperation::<Array>::from(DotOperation::new(DotDimensionNumbers::new(
            vec![0],
            vec![0],
            vec![],
            vec![],
        )));
        let scalar_type = ArrayType::scalar(DataType::F64);
        let producer = ResidualProducer::new(&dot, 0, Vec::new(), vec![scalar_type.clone()]);
        assert_eq!(
            DotsSavable.classify(&ResidualCandidate::new(vec![producer], scalar_type)),
            Ok(ResidualDecision::Save),
        );

        // Lifting a reference to the policy from `ArrayType` into `ArrayIrType` uses its native instantiation, which
        // classifies candidates whose types do not project into `ArrayType` instead of rejecting them.
        let lifted = ResidualPolicyReference::<ArrayType>::new(DotsSavable).lift::<ArrayIrType>();
        let dimension_size = TestOperation::DimensionSize(
            DimensionSizeOperation::new(&ArrayType::new_static(DataType::F64, [3]), 0).unwrap(),
        );
        let producer = ResidualProducer::new(&dimension_size, 0, Vec::new(), vec![dimension_type()]);
        assert!(matches!(
            lifted.classify(&ResidualCandidate::new(vec![producer], dimension_type())),
            Ok(ResidualDecision::Recompute),
        ));
    }

    #[test]
    fn test_dots_with_no_batch_dimensions_savable() {
        let policy = DotsWithNoBatchDimensionsSavable;
        assert_eq!(ResidualPolicy::<ArrayIrType>::name(&policy), "dots_with_no_batch_dimensions_savable");
        assert_eq!(policy.classify(&candidate(&[dot()])), Ok(ResidualDecision::Save));
        assert_eq!(policy.classify(&candidate(&[batched_dot()])), Ok(ResidualDecision::Recompute));
        assert_eq!(policy.classify(&candidate(&[sine()])), Ok(ResidualDecision::Recompute));
        assert_eq!(policy.classify(&candidate(&[batched_dot(), dot()])), Ok(ResidualDecision::Save));
    }

    #[test]
    fn test_offload_dots_with_no_batch_dimensions() {
        let host = Memory::Host { pinned: true };
        let policy = OffloadDotsWithNoBatchDimensions::new(host);
        assert_eq!(policy.destination(), host);
        assert_eq!(ResidualPolicy::<ArrayIrType>::name(&policy), "offload_dots_with_no_batch_dimensions");
        assert_eq!(
            policy.classify(&candidate(&[dot()])),
            Ok(ResidualDecision::SaveWith(MemoryTransferStorage::new(host))),
        );
        assert_eq!(policy.classify(&candidate(&[batched_dot()])), Ok(ResidualDecision::Recompute));
        assert_eq!(policy.classify(&candidate(&[sine()])), Ok(ResidualDecision::Recompute));
    }

    #[test]
    fn test_save_only_these_names() {
        let policy = SaveOnlyTheseNames::new(["a", "b"]);
        assert_eq!(policy.names(), &["a".to_owned(), "b".to_owned()]);
        assert_eq!(ResidualPolicy::<ArrayIrType>::name(&policy), "save_only_these_names");
        assert_eq!(policy.classify(&candidate(&[tag("a")])), Ok(ResidualDecision::Save));
        assert_eq!(policy.classify(&candidate(&[tag("c")])), Ok(ResidualDecision::Recompute));
        assert_eq!(policy.classify(&candidate(&[dot()])), Ok(ResidualDecision::Recompute));
        assert_eq!(policy.classify(&candidate(&[tag("c"), tag("b")])), Ok(ResidualDecision::Save));
    }

    #[test]
    fn test_save_any_names_but_these() {
        let policy = SaveAnyNamesButThese::new(["a"]);
        assert_eq!(policy.names(), &["a".to_owned()]);
        assert_eq!(ResidualPolicy::<ArrayIrType>::name(&policy), "save_any_names_but_these");
        assert_eq!(policy.classify(&candidate(&[tag("b")])), Ok(ResidualDecision::Save));
        assert_eq!(policy.classify(&candidate(&[tag("a")])), Ok(ResidualDecision::Recompute));

        // Untagged residuals are recomputed.
        assert_eq!(policy.classify(&candidate(&[dot()])), Ok(ResidualDecision::Recompute));
        assert_eq!(policy.classify(&candidate(&[tag("a"), tag("b")])), Ok(ResidualDecision::Save));
    }

    #[test]
    fn test_save_anything_except_these_names() {
        let policy = SaveAnythingExceptTheseNames::new(["a"]);
        assert_eq!(policy.names(), &["a".to_owned()]);
        assert_eq!(ResidualPolicy::<ArrayIrType>::name(&policy), "save_anything_except_these_names");
        assert_eq!(policy.classify(&candidate(&[tag("b")])), Ok(ResidualDecision::Save));
        assert_eq!(policy.classify(&candidate(&[tag("a")])), Ok(ResidualDecision::Recompute));

        // Unlike `SaveAnyNamesButThese`, untagged residuals are saved, including ones that an excluded tag may also
        // produce.
        assert_eq!(policy.classify(&candidate(&[dot()])), Ok(ResidualDecision::Save));
        assert_eq!(policy.classify(&candidate(&[tag("a"), sine()])), Ok(ResidualDecision::Save));
    }

    #[test]
    fn test_save_and_offload_only_these_names() {
        let host = Memory::Host { pinned: false };
        let policy = SaveAndOffloadOnlyTheseNames::new(["a"], ["b"], host).unwrap();
        assert_eq!(policy.savable_names(), &["a".to_owned()]);
        assert_eq!(policy.offloadable_names(), &["b".to_owned()]);
        assert_eq!(policy.destination(), host);
        assert_eq!(ResidualPolicy::<ArrayIrType>::name(&policy), "save_and_offload_only_these_names");
        assert_eq!(policy.classify(&candidate(&[tag("a")])), Ok(ResidualDecision::Save));
        assert_eq!(
            policy.classify(&candidate(&[tag("b")])),
            Ok(ResidualDecision::SaveWith(MemoryTransferStorage::new(host))),
        );
        assert_eq!(policy.classify(&candidate(&[tag("c")])), Ok(ResidualDecision::Recompute));
        assert_eq!(policy.classify(&candidate(&[dot()])), Ok(ResidualDecision::Recompute));

        // Saving takes precedence over offloading for residuals that both kinds of tags may produce.
        assert_eq!(policy.classify(&candidate(&[tag("b"), tag("a")])), Ok(ResidualDecision::Save));
    }

    #[test]
    fn test_save_and_offload_only_these_names_overlapping_names() {
        assert_eq!(
            SaveAndOffloadOnlyTheseNames::new(["a", "b", "c"], ["c", "a"], Memory::Device),
            Err(ProgramError::InvalidArgument {
                message: "names `a`, `c` cannot be both savable and offloadable by a \
                    `save_and_offload_only_these_names` policy"
                    .to_owned(),
            }),
        );
    }

    #[test]
    fn test_save_from_both_policies() {
        let host = Memory::Host { pinned: true };
        let policy =
            SaveFromBothPolicies::new(OffloadDotsWithNoBatchDimensions::new(host), SaveOnlyTheseNames::new(["a"]));
        assert_eq!(policy.first(), &OffloadDotsWithNoBatchDimensions::new(host));
        assert_eq!(policy.second(), &SaveOnlyTheseNames::new(["a"]));
        assert_eq!(ResidualPolicy::<ArrayIrType>::name(&policy), "save_from_both_policies");

        // The first policy decides first, keeping its storage, and the second one decides what the first recomputes.
        let Ok(ResidualDecision::SaveWith(storage)) = policy.classify(&candidate(&[dot()])) else {
            panic!("expected an offloaded residual");
        };
        assert_eq!(storage.name(), "memory_transfer");
        assert!(matches!(policy.classify(&candidate(&[tag("a")])), Ok(ResidualDecision::Save)));
        assert!(matches!(policy.classify(&candidate(&[sine()])), Ok(ResidualDecision::Recompute)));

        // Rejections of either policy are returned as they are.
        let rejecting =
            PolicyFn::new::<ArrayIrType>(|_| Err::<ResidualDecision<NoStorage>, _>(ResidualRejection::new("rejected")));
        let policy = SaveFromBothPolicies::new(NothingSavable, rejecting);
        assert_eq!(policy.classify(&candidate(&[sine()])).map(|_| ()), Err(ResidualRejection::new("rejected")));
        let policy = SaveFromBothPolicies::new(EverythingSavable, policy.second().clone());
        assert!(matches!(policy.classify(&candidate(&[sine()])), Ok(ResidualDecision::Save)));

        // The policy declares no native instantiations, so lifting a reference to it projects the types of each
        // candidate and cannot classify the ones that do not project.
        let lifted = ResidualPolicyReference::<ArrayType>::new(SaveFromBothPolicies::new(DotsSavable, NothingSavable))
            .lift::<ArrayIrType>();
        let dimension_size = TestOperation::DimensionSize(
            DimensionSizeOperation::new(&ArrayType::new_static(DataType::F64, [3]), 0).unwrap(),
        );
        let producer = ResidualProducer::new(&dimension_size, 0, Vec::new(), vec![dimension_type()]);
        assert_eq!(
            lifted.classify(&ResidualCandidate::new(vec![producer], dimension_type())).map(|_| ()),
            Err(ResidualPolicyError::UnsupportedProjection {
                policy: "save_from_both_policies".to_owned(),
                position: "the output 0 of producer `dimension_size`".to_owned(),
                residual_type: dimension_type().to_string(),
            }),
        );
    }

    #[test]
    fn test_policy_fn() {
        // Saves the residuals that have one producer and recomputes the ones that several producers may produce.
        let policy = PolicyFn::new::<ArrayIrType>(|candidate| {
            Ok::<_, ResidualRejection>(match candidate.producers().len() {
                1 => ResidualDecision::<NoStorage>::Save,
                _ => ResidualDecision::Recompute,
            })
        });
        assert_eq!(ResidualPolicy::<ArrayIrType>::name(&policy), "policy_fn");
        assert_eq!(policy.classify(&candidate(&[sine()])), Ok(ResidualDecision::Save));
        assert_eq!(policy.classify(&candidate(&[sine(), dot()])), Ok(ResidualDecision::Recompute));

        // Names apply to clones too, and the closure does not render.
        let policy = policy.with_name("save_unique_producers");
        assert_eq!(ResidualPolicy::<ArrayIrType>::name(&policy.clone()), "save_unique_producers");
        assert_eq!(format!("{policy:?}"), "PolicyFn { name: \"save_unique_producers\", .. }");

        // Closures may return storages too.
        let host = Memory::Host { pinned: true };
        let policy = PolicyFn::new::<ArrayIrType>(move |_| {
            Ok::<_, ResidualRejection>(ResidualDecision::SaveWith(MemoryTransferStorage::new(host)))
        });
        assert_eq!(
            policy.classify(&candidate(&[sine()])),
            Ok(ResidualDecision::SaveWith(MemoryTransferStorage::new(host))),
        );
    }

    #[test]
    fn test_memory_transfer_storage() {
        let host = Memory::Host { pinned: true };
        let storage = MemoryTransferStorage::new(host);
        assert_eq!(storage.destination(), host);
        assert_eq!(ResidualStorage::<ArrayIrType>::name(&storage), "memory_transfer");

        // Storing transfers the residual to the destination, and restoring transfers it back to its own memory.
        let mut store = storage.store_payloads(&scalar_type()).unwrap();
        assert_eq!(store.len(), 1);
        assert_eq!(store.remove(0).downcast::<TransferToMemoryOperation>().unwrap().destination(), host);
        let stored_type = ArrayIrType::from(ArrayType::scalar(DataType::F64).with_memory(host));
        let mut restore = storage.restore_payloads(&stored_type, &scalar_type()).unwrap();
        assert_eq!(restore.len(), 1);
        assert_eq!(restore.remove(0).downcast::<TransferToMemoryOperation>().unwrap().destination(), Memory::Device);

        // Residuals that are not arrays cannot be offloaded.
        let error = ResidualPolicyError::UnsupportedStorage {
            storage: "memory_transfer".to_owned(),
            residual_type: dimension_type().to_string(),
            message: "the residual is not an array".to_owned(),
        };
        assert_eq!(storage.store_payloads(&dimension_type()).map(|_| ()), Err(error.clone()));
        assert_eq!(storage.restore_payloads(&dimension_type(), &dimension_type()).map(|_| ()), Err(error));
    }
}
