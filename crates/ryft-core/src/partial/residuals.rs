//! Residual policies, which decide how the known values that residual work consumes cross a partition boundary,
//! and the planner that applies them to a [`PartitionedProgram`].
//!
//! Partitioning a program into known work and residual work makes every known value that residual work needs
//! a _residual edge_ (i.e., an output of the known program that the residual program consumes). Saving each such
//! value is not always desirable. Automatic differentiation, for example, saves the intermediate values of a primal
//! computation for its pullback, and saving all of them can require far more memory than recomputing the cheap ones
//! in the pullback. A [`ResidualPolicy`] decides, for each known value that residual work needs, whether to
//! [save](ResidualDecision::Save) it as an edge, to save it through a [`ResidualStorage`] (e.g., by offloading
//! it to host memory), or to [recompute](ResidualDecision::Recompute) it in the residual program.
//! [`PartitionedProgram::with_residual_policy`] then rewrites the partition accordingly.
//!
//! This is the mechanism behind [JAX's checkpoint policies](https://docs.jax.dev/en/latest/301/remat.html), which
//! `partial_eval_jaxpr_custom` applies when it partially evaluates a program. As there, a policy sees every value
//! that the residual program would need, not only the edges of the original partition, so that a policy which saves
//! only the outputs of dot products saves the dot in `sin(dot(x, x))` and recomputes its cosine in the residual
//! program, instead of saving the cosine.
//!
//! # Policies and Candidates
//!
//! A policy classifies one [`ResidualCandidate`] at a time. A candidate describes one known value together with the
//! [`ResidualProducer`]s that may have produced it, looking through the outputs of operations that forward the outputs
//! of their attached regions (e.g., a condition forwards the corresponding outputs of its branches). Producers expose
//! the payloads of their operations (via [`ResidualProducer::payload`]), so that policies recognize them by payload
//! type (e.g., a [`DotOperation`](crate::DotOperation)) regardless of the operation family that holds them. Policies
//! are therefore defined over a [`Type`] universe rather than an operation family, and [`ResidualPolicyReference`]
//! is the type-erased, identity-compared handle through which programs and transforms carry them.
//!
//! # Planning
//!
//! [`PartitionedProgram::with_residual_policy`] decides everything before emitting anything. Starting from the known
//! values that the residual program reads, it classifies each demanded value once, and it marks the producers of
//! recomputed values for replay in the residual program, which demands their own inputs in turn. Known inputs are
//! forwarded as edges and constants are re-created in the residual program without consulting the policy. Work that
//! cannot be replayed safely (e.g., reads of references that the known program shares with its caller, or other
//! observable effects) is saved regardless of the policy. Local reference state is replayed together with its
//! complete lifecycle prefix, and lifecycles that the known program no longer observes are removed from it.
//!
//! Residual demand is distinct from knownness. When partitioning places residuals (e.g., via
//! [`Program::partition_with_residual_policy`](crate::Program::partition_with_residual_policy)) and residual work
//! demands an output of an all-known region-carrying instruction (e.g., a `scan` whose inputs carry no gradients) whose
//! replay would recompute values that the policy saves, the planner asks the operation to split that instruction on
//! residual demand, as JAX's `partial_eval_jaxpr_custom` rules do. The known part of the split also returns the
//! interior values that the policy saves, and its replay part recomputes the demanded outputs from them in the residual
//! program. Refer to [`split_on_residual_demand`](crate::PartiallyEvaluatableOperation::split_on_residual_demand) for
//! more information.

use std::any::{Any, TypeId};
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use thiserror::Error;

use crate::parameters::Placeholder;
use crate::partial::partitions::PartitionedProgram;
use crate::partial::values::{PartialEvaluationOutput, ResidualInputSource};
use crate::programs::{
    Atom, AtomId, AttachedRegionLiveness, ErasedOperation, Instruction, Operation, OperationPayloadProjection, Program,
    ProgramBuilder, ProgramError, RegionDataFlow, RegionDataFlowBoundary, RegionDataFlowSource, RegionDataFlowSources,
    RegionId, RegionPruningAnalysis, RegionRef, RegionRole, Type, Typed, Value, ValueId,
};

/// Error returned when classifying residuals with a [`ResidualPolicy`], staging their [`ResidualStorage`],
/// or placing the residuals of a [`PartitionedProgram`] with [`PartitionedProgram::with_residual_policy`].
///
/// This error and [`ProgramError`] convert into each other without losing information. Converting a
/// [`ResidualPolicyError::Program`] into a [`ProgramError`] unwraps it and every other variant is wrapped in
/// [`ProgramError::Custom`], while converting a [`ProgramError`] into a [`ResidualPolicyError`] recovers a
/// wrapped residual-policy error and wraps any other program error in [`ResidualPolicyError::Program`].
#[derive(Clone, Debug, PartialEq, Eq, Hash, Error)]
pub enum ResidualPolicyError {
    #[error("residual policy `{policy}` rejected a residual: {}", .rejection.message())]
    Rejected {
        /// Name of the rejecting policy.
        policy: String,

        /// Rejection that the classifier of the policy returned.
        rejection: ResidualRejection,
    },

    #[error(
        "residual policy `{policy}` cannot classify {position} of type `{residual_type}`, which does not project into \
         the type universe of the policy; register a native instantiation or a projection fallback for this universe"
    )]
    UnsupportedProjection {
        /// Name of the lifted policy.
        policy: String,

        /// Description of the position of the unprojectable type (e.g., a producer or the residual itself).
        position: String,

        /// Rendering of the unprojectable type.
        residual_type: String,
    },

    #[error("residual storage `{storage}` cannot stage a residual of type `{residual_type}`: {message}")]
    UnsupportedStorage {
        /// Name of the storage.
        storage: String,

        /// Rendering of the residual type.
        residual_type: String,

        /// Description of the problem.
        message: String,
    },

    #[error("residual storage `{storage}` is invalid: {message}")]
    InvalidStorage {
        /// Name of the storage.
        storage: String,

        /// Description of the violated requirement.
        message: String,
    },

    #[error(transparent)]
    Program(ProgramError),
}

impl From<ProgramError> for ResidualPolicyError {
    #[inline]
    fn from(error: ProgramError) -> Self {
        if let Some(error) = error.downcast_custom::<ResidualPolicyError>() {
            error.clone()
        } else {
            ResidualPolicyError::Program(error)
        }
    }
}

impl From<ResidualPolicyError> for ProgramError {
    #[inline]
    fn from(error: ResidualPolicyError) -> Self {
        match error {
            ResidualPolicyError::Program(error) => error,
            error => ProgramError::custom(error),
        }
    }
}

/// One operation output that may have produced a residual, as described to a [`ResidualPolicy`] by a
/// [`ResidualCandidate`]. The types are those of the producer's application site, which may differ from
/// the type of the residual itself (e.g., when the producer is part of a loop body whose outputs are stacked).
pub struct ResidualProducer<'o, T: Type> {
    /// Name of the operation that produced the output.
    name: &'static str,

    /// Operation that produced the output, viewed through its [`OperationPayloadProjection`].
    operation: &'o dyn OperationPayloadProjection,

    /// Index of the output among the outputs of the producer.
    output_index: usize,

    /// Types of the producer's inputs.
    input_types: Vec<T>,

    /// Types of the producer's outputs.
    output_types: Vec<T>,
}

impl<'o, T: Type> ResidualProducer<'o, T> {
    /// Creates a new [`ResidualProducer`].
    #[inline]
    pub fn new<O: Operation<Type = T> + OperationPayloadProjection>(
        operation: &'o O,
        output_index: usize,
        input_types: Vec<T>,
        output_types: Vec<T>,
    ) -> Self {
        Self { name: operation.name(), operation, output_index, input_types, output_types }
    }

    /// Returns the name of the operation that produced the output.
    #[inline]
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// Returns the operation that produced the output, viewed through its [`OperationPayloadProjection`]
    /// (e.g., to read the key of a producing tag with [`TagOperation::key_of`](crate::TagOperation::key_of)).
    #[inline]
    pub fn operation(&self) -> &'o dyn OperationPayloadProjection {
        self.operation
    }

    /// Returns the index of the output among the outputs of the producer.
    #[inline]
    pub fn output_index(&self) -> usize {
        self.output_index
    }

    /// Returns the types of the producer's inputs at its application site.
    #[inline]
    pub fn input_types(&self) -> &[T] {
        self.input_types.as_slice()
    }

    /// Returns the types of the producer's outputs at its application site.
    #[inline]
    pub fn output_types(&self) -> &[T] {
        self.output_types.as_slice()
    }

    /// Returns the payload operation of type `P` of the producer, if its operation holds one directly or through a
    /// projected member of its operation family. For example, a policy that saves the outputs of dot products checks
    /// `producer.payload::<DotOperation>().is_some()`. Refer to [`OperationPayloadProjection`] for more information.
    #[inline]
    pub fn payload<P: 'static>(&self) -> Option<&'o P> {
        self.operation.project_payload(TypeId::of::<P>()).and_then(|payload| payload.downcast_ref::<P>())
    }
}

/// Description of one residual that a [`ResidualPolicy`] classifies. It contains every operation output that may have
/// produced the residual, in a stable order (most residuals have one producer, while the output of a condition, for
/// example, has one producer per branch). Built-in loops conservatively include the initial carry and every producer
/// reachable through the body carry dependencies, including producers that a particular finite trip count may not
/// reach. A statically empty scan contributes only its initial carries; stacked empty outputs have no producers.
/// Policies return one decision for the complete candidate.
pub struct ResidualCandidate<'o, T: Type> {
    /// Operation outputs that may have produced the residual, in semantic order.
    producers: Vec<ResidualProducer<'o, T>>,

    /// Type of the residual.
    r#type: T,
}

impl<'o, T: Type> ResidualCandidate<'o, T> {
    /// Creates a new [`ResidualCandidate`] with the provided producers, in semantic order, and residual type.
    #[inline]
    pub fn new(producers: Vec<ResidualProducer<'o, T>>, r#type: T) -> Self {
        Self { producers, r#type }
    }

    /// Returns the operation outputs that may have produced this residual, in semantic order.
    #[inline]
    pub fn producers(&self) -> &[ResidualProducer<'o, T>] {
        self.producers.as_slice()
    }
}

impl<T: Type> Typed for ResidualCandidate<'_, T> {
    type Type = T;

    #[inline]
    fn r#type(&self) -> Cow<'_, T> {
        Cow::Borrowed(&self.r#type)
    }
}

/// Decision of a [`ResidualPolicy`] for one [`ResidualCandidate`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResidualDecision<S> {
    /// Saves the residual as an edge from the known program to the residual program.
    Save,

    /// Saves the residual as an edge through the provided [`ResidualStorage`], whose store operations apply to the
    /// residual in the known program and whose restore operations reproduce it in the residual program before its
    /// first use.
    SaveWith(S),

    /// Recomputes the residual in the residual program by replaying the operation that produced it. That replay needs
    /// the operation's inputs in the residual program too, so each of them becomes a known value that residual work
    /// demands and is classified in the same way. For example, recomputing `cos(dot(x, x))` demands `dot(x, x)`, which
    /// the policy may save or recompute in turn, while known inputs such as `x` are always forwarded.
    Recompute,
}

impl<S> ResidualDecision<S> {
    /// Returns this decision with its storage (if any) type-erased.
    #[inline]
    pub fn into_erased<T: 'static + Type>(self) -> ResidualDecision<ErasedResidualStorage<T>>
    where
        S: ResidualStorage<T>,
    {
        match self {
            Self::Save => ResidualDecision::Save,
            Self::SaveWith(storage) => ResidualDecision::SaveWith(Arc::new(storage)),
            Self::Recompute => ResidualDecision::Recompute,
        }
    }
}

/// Rejection of a residual by the classifier of a [`ResidualPolicy`] (e.g., because the policy forbids saving values
/// that carry a particular name), which placing residuals reports as [`ResidualPolicyError::Rejected`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ResidualRejection {
    /// Refer to the documentation of [`message`](Self::message) for more information.
    message: String,
}

impl ResidualRejection {
    /// Creates a new [`ResidualRejection`] with the provided message.
    #[inline]
    pub fn new<M: Into<String>>(message: M) -> Self {
        Self { message: message.into() }
    }

    /// Returns the message that explains this rejection.
    #[inline]
    pub fn message(&self) -> &str {
        self.message.as_str()
    }
}

/// Reversible transformation that is applied to a saved residual, such as offloading it to host memory,
/// described without naming an operation family. Each transformation is a chain of payload operations
/// (e.g., [`TransferToMemoryOperation`](crate::TransferToMemoryOperation)s) that
/// [`PartitionedProgram::with_residual_policy`] stages into the family of the partitioned programs through
/// [`OperationPayloadProjection::from_payload`]. The store chain applies to the residual in the known program, and the
/// restore chain reproduces it in the residual program before its first use. Every staged operation must be a pure
/// unary operation with a single result, which leaves program simplification free to schedule it anywhere that its data
/// dependencies allow, and the restore chain must reproduce the type of the residual up to identity renaming. A storage
/// can recognize its own store operations through [`matches_store_operation`](Self::matches_store_operation), allowing
/// placement to reuse an existing identical store chain over the same atom when that representation is already a known
/// output. Storages that do not recognize their operations remain valid and are staged again.
pub trait ResidualStorage<T: Type>: 'static + Send + Sync + Debug {
    /// Returns the name of this storage, which is used in diagnostics.
    fn name(&self) -> String;

    /// Returns the payload operations that store a residual of type `residual_type`, in application order.
    fn store_payloads(&self, residual_type: &T) -> Result<Vec<ErasedOperation>, ResidualPolicyError>;

    /// Returns the payload operations that restore a residual of type `residual_type` from its stored form of type
    /// `stored_type`, in application order.
    fn restore_payloads(&self, stored_type: &T, residual_type: &T)
    -> Result<Vec<ErasedOperation>, ResidualPolicyError>;

    /// Returns whether `operation` does the same work, with the same settings, as the operation at zero-based position
    /// `index` returned by [`store_payloads`](Self::store_payloads) for `residual_type`. Placement uses this to reuse
    /// a stored value already returned by the known program, instead of storing that value again. For example, a memory
    /// transfer must use the same destination, including host pinning. The default returns [`false`], so placement
    /// stores the value again. Return [`false`] for unsupported residual types or an index outside the store chain.
    /// Checking a match must not run the operation or change storage state. Placement also checks that the complete
    /// chain starts from the same source value and that each operation is pure, takes one input, returns one output
    /// and has no regions. Matching names, types or chain lengths alone does not make a stored value safe to reuse.
    #[inline]
    fn matches_store_operation(
        &self,
        residual_type: &T,
        index: usize,
        operation: &dyn OperationPayloadProjection,
    ) -> bool {
        let _ = (residual_type, index, operation);
        false
    }
}

/// Type-erased [`ResidualStorage`], which [`ResidualPolicyReference::classify`] returns. It is a [`ResidualStorage`]
/// itself, so that policies whose decisions may use different storages (e.g., a policy that combines two other
/// policies) can return erased storage.
pub type ErasedResidualStorage<T> = Arc<dyn ResidualStorage<T>>;

impl<T: 'static + Type> ResidualStorage<T> for ErasedResidualStorage<T> {
    #[inline]
    fn name(&self) -> String {
        self.as_ref().name()
    }

    #[inline]
    fn store_payloads(&self, residual_type: &T) -> Result<Vec<ErasedOperation>, ResidualPolicyError> {
        self.as_ref().store_payloads(residual_type)
    }

    #[inline]
    fn restore_payloads(
        &self,
        stored_type: &T,
        residual_type: &T,
    ) -> Result<Vec<ErasedOperation>, ResidualPolicyError> {
        self.as_ref().restore_payloads(stored_type, residual_type)
    }

    #[inline]
    fn matches_store_operation(
        &self,
        residual_type: &T,
        index: usize,
        operation: &dyn OperationPayloadProjection,
    ) -> bool {
        self.as_ref().matches_store_operation(residual_type, index, operation)
    }
}

/// Uninhabited [`ResidualStorage`] of policies that never return [`ResidualDecision::SaveWith`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum NoStorage {}

impl<T: Type> ResidualStorage<T> for NoStorage {
    #[inline]
    fn name(&self) -> String {
        match *self {}
    }

    #[inline]
    fn store_payloads(&self, _residual_type: &T) -> Result<Vec<ErasedOperation>, ResidualPolicyError> {
        match *self {}
    }

    #[inline]
    fn restore_payloads(
        &self,
        _stored_type: &T,
        _residual_type: &T,
    ) -> Result<Vec<ErasedOperation>, ResidualPolicyError> {
        match *self {}
    }
}

/// [`ResidualStorage`] of a policy over type universe `S` used in a type universe whose types project into `S`. Its
/// payloads do not depend on an operation family, so they pass through unchanged, and only the residual types are
/// projected.
#[derive(Debug)]
struct LiftedResidualStorage<S: Type> {
    /// Storage in the source universe.
    source: ErasedResidualStorage<S>,
}

impl<S: 'static + Type> LiftedResidualStorage<S> {
    /// Returns `r#type` projected into the source universe, or an [`ResidualPolicyError::UnsupportedStorage`] error
    /// when it does not project into it.
    #[inline]
    fn project<'t, T: Type>(&self, r#type: &'t T) -> Result<&'t S, ResidualPolicyError>
    where
        &'t S: TryFrom<&'t T>,
    {
        <&S>::try_from(r#type).map_err(|_| ResidualPolicyError::UnsupportedStorage {
            storage: self.source.name(),
            residual_type: r#type.to_string(),
            message: "the type does not project into the type universe of the storage".to_owned(),
        })
    }
}

impl<S: 'static + Type, T: Type> ResidualStorage<T> for LiftedResidualStorage<S>
where
    for<'t> &'t S: TryFrom<&'t T>,
{
    #[inline]
    fn name(&self) -> String {
        self.source.name()
    }

    #[inline]
    fn store_payloads(&self, residual_type: &T) -> Result<Vec<ErasedOperation>, ResidualPolicyError> {
        self.source.store_payloads(self.project(residual_type)?)
    }

    #[inline]
    fn restore_payloads(
        &self,
        stored_type: &T,
        residual_type: &T,
    ) -> Result<Vec<ErasedOperation>, ResidualPolicyError> {
        self.source.restore_payloads(self.project(stored_type)?, self.project(residual_type)?)
    }

    #[inline]
    fn matches_store_operation(
        &self,
        residual_type: &T,
        index: usize,
        operation: &dyn OperationPayloadProjection,
    ) -> bool {
        <&S>::try_from(residual_type)
            .is_ok_and(|residual_type| self.source.matches_store_operation(residual_type, index, operation))
    }
}

/// Policy that decides how each known value that residual work needs crosses a partition boundary. Policies are
/// defined over a [`Type`] universe `T` rather than an operation family, and they recognize the producers of residuals
/// by payload type (i.e., using [`ResidualProducer::payload`]), so one policy value applies to every operation family
/// over `T`, including families that hold operations of another universe through projected members.
///
/// Programs and transforms carry policies through [`ResidualPolicyReference`]s. When an operation that holds a
/// policy moves into an operation family over a larger type universe (e.g., from [`ArrayType`](crate::ArrayType) to
/// [`ArrayIrType`](crate::ArrayIrType)), its reference is [lifted](ResidualPolicyReference::lift) into that universe.
/// By default, lifting projects the types of each candidate into `T` and cannot classify candidates whose types do not
/// project into it (e.g., dimensions). Policies that are defined for several universes should therefore declare their
/// instantiations in those universes through [`native_instantiations`](Self::native_instantiations), which lifting
/// then uses instead.
pub trait ResidualPolicy<T: 'static + Type>: 'static + Send + Sync {
    /// Storage that this policy uses when it returns [`ResidualDecision::SaveWith`]. This is set to [`NoStorage`]
    /// for policies that never do.
    type Storage: ResidualStorage<T>;

    /// Returns the name of this policy, which is used in diagnostics and in the rendering of operations that carry it.
    fn name(&self) -> &str;

    /// Classifies `candidate`, returning a [`ResidualRejection`] when this policy forbids every placement of it.
    fn classify(
        &self,
        candidate: &ResidualCandidate<'_, T>,
    ) -> Result<ResidualDecision<Self::Storage>, ResidualRejection>;

    /// Returns the instantiations of this policy in other type universes, which [`ResidualPolicyReference::lift`] uses
    /// instead of projecting candidate types into `T`. Every instantiation must make the same decisions as this policy
    /// for candidates whose types project into `T`, which is what makes lifting preserve the meaning of a policy.
    /// The default declares none.
    #[inline]
    fn native_instantiations(&self) -> NativeResidualPolicies {
        NativeResidualPolicies::default()
    }
}

/// Type-erased, object-safe form of a [`ResidualPolicy`] instantiated in type universe `T`, which is what a
/// [`ResidualPolicyReference`] stores so that programs and operations can carry policies of any type.
///
/// [`ResidualPolicy`] itself cannot serve as that trait object. Its associated [`Storage`](ResidualPolicy::Storage)
/// type lets policies type their decisions precisely (e.g., [`NoStorage`] for policies that never store residuals), but
/// a trait object must fix associated types, so every policy stored behind one trait object would need the same
/// storage. Furthermore, [`ResidualPolicy::native_instantiations`] is consulted once, when a reference is created, and
/// has no role afterwards. This trait instead returns type-erased storage.
///
/// It also differs from [`ResidualPolicy`] in its errors: [`classify`](Self::classify) returns
/// [`ResidualPolicyError`]s, which attach the name of the policy to rejections and let the implementation that lifts
/// a policy into another type universe (i.e., [`LiftedResidualPolicy`]) fail with
/// [`ResidualPolicyError::UnsupportedProjection`] and return [`LiftedResidualStorage`]. That adapter is not a policy
/// that users should implement, which is why this trait is private while [`ResidualPolicy`] stays small, typed, and
/// public. [`NativeResidualPolicy`] wraps a typed policy, [`LiftedResidualPolicy`] projects into another universe,
/// and [`CombinedNativeResidualPolicy`] combines two native instantiations.
trait ErasedResidualPolicy<T: Type>: Send + Sync {
    /// Returns the name of the policy.
    fn name(&self) -> &str;

    /// Classifies `candidate`, reporting rejections as [`ResidualPolicyError::Rejected`].
    fn classify(
        &self,
        candidate: &ResidualCandidate<'_, T>,
    ) -> Result<ResidualDecision<ErasedResidualStorage<T>>, ResidualPolicyError>;
}

/// [`ErasedResidualPolicy`] of a [`ResidualPolicy`] in its own type universe. It is _native_ because the wrapped
/// policy is implemented for that universe and classifies candidates in the universe's own types, unlike a
/// [`LiftedResidualPolicy`], which projects the types of each candidate into the universe of a policy implemented for
/// another universe (refer to the documentation of [`NativeResidualPolicies`] for why the distinction matters). It
/// erases the storage and rejections of the wrapped policy.
struct NativeResidualPolicy<P>(P);

impl<T: 'static + Type, P: ResidualPolicy<T>> ErasedResidualPolicy<T> for NativeResidualPolicy<P> {
    #[inline]
    fn name(&self) -> &str {
        self.0.name()
    }

    #[inline]
    fn classify(
        &self,
        candidate: &ResidualCandidate<'_, T>,
    ) -> Result<ResidualDecision<ErasedResidualStorage<T>>, ResidualPolicyError> {
        match self.0.classify(candidate) {
            Ok(decision) => Ok(decision.into_erased()),
            Err(rejection) => Err(ResidualPolicyError::Rejected { policy: self.0.name().to_owned(), rejection }),
        }
    }
}

/// [`ErasedResidualPolicy`] that applies a policy over type universe `S` to candidates of type universe `T`, whose
/// types project into `S`. Producer operations do not depend on a type universe, so they pass through unchanged, and
/// only the types of each candidate are projected. A candidate with a type that does not project into `S` is classified
/// by the registered projection fallback or fails with [`ResidualPolicyError::UnsupportedProjection`]; it never reaches
/// the source classifier partially projected.
struct LiftedResidualPolicy<S: Type, T: Type> {
    /// Policy in the source universe.
    source: Arc<dyn ErasedResidualPolicy<S>>,

    /// Classifier of the candidates that do not project into the source universe, if one is registered.
    fallback: Option<ProjectionFallback<T>>,
}

impl<S: 'static + Type, T: 'static + Type> ErasedResidualPolicy<T> for LiftedResidualPolicy<S, T>
where
    for<'t> &'t S: TryFrom<&'t T>,
{
    #[inline]
    fn name(&self) -> &str {
        self.source.name()
    }

    fn classify(
        &self,
        candidate: &ResidualCandidate<'_, T>,
    ) -> Result<ResidualDecision<ErasedResidualStorage<T>>, ResidualPolicyError> {
        // Project every producer's types into the source universe. Producer operations are family-agnostic payload
        // views, so they pass through unchanged. The projectability of each producer is recorded for the projection
        // fallback, and the first type that does not project is recorded for the diagnostic.
        let project = |r#type: &T| <&S>::try_from(r#type).ok().cloned();
        let mut unprojectable = None;
        let mut producers = Vec::with_capacity(candidate.producers().len());
        let mut producer_projectable = Vec::with_capacity(candidate.producers().len());
        for producer in candidate.producers() {
            let input_types = producer.input_types().iter().map(project).collect::<Option<Vec<_>>>();
            let output_types = producer.output_types().iter().map(project).collect::<Option<Vec<_>>>();
            if let (Some(input_types), Some(output_types)) = (input_types, output_types) {
                producer_projectable.push(true);
                producers.push(ResidualProducer {
                    name: producer.name,
                    operation: producer.operation,
                    output_index: producer.output_index(),
                    input_types,
                    output_types,
                });
            } else {
                producer_projectable.push(false);
                if unprojectable.is_none() {
                    // The `unwrap` is safe because one of the producer's types failed to project just above.
                    let r#type = producer
                        .input_types()
                        .iter()
                        .chain(producer.output_types())
                        .find(|r#type| project(r#type).is_none())
                        .unwrap();
                    unprojectable = Some((
                        format!("the output {} of producer `{}`", producer.output_index(), producer.name()),
                        r#type.to_string(),
                    ));
                }
            }
        }

        // The residual type must project as well. A producer type that failed to project takes precedence in the
        // diagnostic, because it is usually the root cause (e.g., a dimension-producing operation).
        let residual_type = project(candidate.r#type().as_ref());
        if residual_type.is_none() && unprojectable.is_none() {
            unprojectable = Some(("the residual".to_owned(), candidate.r#type().to_string()));
        }

        match (unprojectable, residual_type) {
            (None, Some(residual_type)) => {
                // A fully projectable candidate reaches the source policy unchanged apart from its types, and so the
                // source policy's decisions (including its rejections) carry over. Storage returned by the source
                // policy is defined over the source universe, and so it is wrapped to project the types that the
                // planner passes to it.
                Ok(match self.source.classify(&ResidualCandidate::new(producers, residual_type))? {
                    ResidualDecision::Save => ResidualDecision::Save,
                    ResidualDecision::SaveWith(storage) => {
                        ResidualDecision::SaveWith(Arc::new(LiftedResidualStorage { source: storage }))
                    }
                    ResidualDecision::Recompute => ResidualDecision::Recompute,
                })
            }
            (unprojectable, residual_type) => {
                // A candidate with any unprojectable type never reaches the source policy partially projected. It goes
                // to the projection fallback of the destination universe, if the policy registered one, together with
                // which of its types project. Otherwise, it cannot be classified at all. Note that the `unwrap` below
                // is safe because a candidate whose residual type does not project records it as unprojectable just
                // above.
                let (position, r#type) = unprojectable.unwrap();
                match &self.fallback {
                    Some(fallback) => fallback(&ProjectionFallbackCandidate {
                        candidate,
                        producer_projectable,
                        residual_projectable: residual_type.is_some(),
                    })
                    .map_err(|rejection| ResidualPolicyError::Rejected {
                        policy: self.source.name().to_owned(),
                        rejection,
                    }),
                    None => Err(ResidualPolicyError::UnsupportedProjection {
                        policy: self.source.name().to_owned(),
                        position,
                        residual_type: r#type,
                    }),
                }
            }
        }
    }
}

/// Native instantiation of a [`SaveFromBothPolicies`](crate::SaveFromBothPolicies) composition in one type universe
/// supported by both child policies, instantiated from a deferred [`NativeResidualPolicyComposition`]. Like the typed
/// composition, it takes the first saving decision and consults the second policy only when the first requests
/// recomputation.
struct CombinedNativeResidualPolicy<T: Type> {
    /// Name of the composed policy, matching its non-erased definition.
    name: &'static str,

    /// Policy that classifies a candidate first.
    first: Arc<dyn ErasedResidualPolicy<T>>,

    /// Policy that classifies candidates that the first policy recomputes.
    second: Arc<dyn ErasedResidualPolicy<T>>,
}

impl<T: Type> ErasedResidualPolicy<T> for CombinedNativeResidualPolicy<T> {
    #[inline]
    fn name(&self) -> &str {
        self.name
    }

    fn classify(
        &self,
        candidate: &ResidualCandidate<'_, T>,
    ) -> Result<ResidualDecision<ErasedResidualStorage<T>>, ResidualPolicyError> {
        let result = match self.first.classify(candidate) {
            Ok(ResidualDecision::Recompute) => self.second.classify(candidate),
            result => result,
        };

        // The typed composition returns its child's rejection as a rejection of the composed policy.
        // Keep the same diagnostic after erasure, preserving other error variants and all rejection details.
        result.map_err(|error| match error {
            ResidualPolicyError::Rejected { rejection, .. } => {
                ResidualPolicyError::Rejected { policy: self.name.to_owned(), rejection }
            }
            error => error,
        })
    }
}

/// Instantiations of one [`ResidualPolicy`] in other type universes, which [`ResidualPolicy::native_instantiations`]
/// returns and [`ResidualPolicyReference::lift`] prefers over projecting candidate types. For example, a policy that
/// is generic over its type universe returns `NativeResidualPolicies::default().with::<ArrayType, _>(
/// self.clone()).with::<ArrayIrType, _>(self.clone())`.
///
/// An instantiation is _native_ to its universe because it is implemented for that universe and so classifies its
/// candidates in the universe's own types. This is the alternative to lifting a policy by projection, where the types
/// of each candidate are projected into the universe of the policy before the policy classifies them. Projection can
/// only handle candidates whose types project (e.g., when lifting from [`ArrayType`](crate::ArrayType) into
/// [`ArrayIrType`](crate::ArrayIrType), candidates that produce dimensions or references have no `ArrayType`
/// representation, and so a policy that saves everything could not classify them unless it has a native
/// `ArrayIrType` instantiation, which simply saves them).
///
/// [`SaveFromBothPolicies`](crate::SaveFromBothPolicies) derives its instantiations from those of its two policies.
/// and so it has a native instantiation in every type universe in which both of them have one. Other combined policies
/// register their instantiations explicitly through [`with`](Self::with), which requires the combined policy to
/// implement [`ResidualPolicy`] in each registered universe.
#[derive(Clone, Default)]
pub struct NativeResidualPolicies {
    /// Instantiations keyed by the [`TypeId`] of their universe `U`, each holding an
    /// `Arc<dyn ErasedResidualPolicy<U>>`.
    entries: Vec<(TypeId, Arc<dyn Any + Send + Sync>)>,

    /// Deferred composition of child registries in their common type universes.
    composition: Option<Arc<NativeResidualPolicyComposition>>,
}

impl NativeResidualPolicies {
    /// Returns these instantiations with `policy` registered as the instantiation in type universe `U`,
    /// replacing any instantiation that was registered for `U` before.
    pub fn with<U: 'static + Type, P: ResidualPolicy<U>>(mut self, policy: P) -> Self {
        let policy: Arc<dyn ErasedResidualPolicy<U>> = Arc::new(NativeResidualPolicy(policy));
        self.entries.retain(|(universe, _)| *universe != TypeId::of::<U>());
        self.entries.push((TypeId::of::<U>(), Arc::new(policy)));
        self
    }

    /// Returns the native instantiations of the [`SaveFromBothPolicies`](crate::SaveFromBothPolicies) composition of a
    /// policy whose instantiations are `self` with a policy whose instantiations are `other`. A combined instantiation
    /// exists in each type universe for which both registries provide an instantiation. It classifies a candidate with
    /// the first policy's instantiation and consults the second one only when the first recomputes the candidate,
    /// reporting rejections under `name`. Combined instantiations are created when a lookup names their universe,
    /// because the erased entries of both registries cannot be instantiated in a universe that is not yet known.
    /// Explicit registrations added with [`with`](Self::with) override the combined instantiation of their universe.
    #[inline]
    pub(crate) fn save_from_both(self, other: Self, name: &'static str) -> Self {
        Self {
            entries: Vec::new(),
            composition: Some(Arc::new(NativeResidualPolicyComposition { name, first: self, second: other })),
        }
    }

    /// Returns the explicitly registered or composed native instantiation for type universe `U`, if any.
    fn get<U: 'static + Type>(&self) -> Option<Arc<dyn ErasedResidualPolicy<U>>> {
        self.entries
            .iter()
            .find(|(universe, _)| *universe == TypeId::of::<U>())
            .and_then(|(_, policy)| policy.downcast_ref::<Arc<dyn ErasedResidualPolicy<U>>>().cloned())
            .or_else(|| {
                let composition = self.composition.as_ref()?;
                Some(Arc::new(CombinedNativeResidualPolicy {
                    name: composition.name,
                    first: composition.first.get::<U>()?,
                    second: composition.second.get::<U>()?,
                }))
            })
    }
}

/// Deferred recipe for the native instantiations of a [`SaveFromBothPolicies`](crate::SaveFromBothPolicies)
/// composition, in the type universes supported by both child registries, as created by
/// [`NativeResidualPolicies::save_from_both`]. The registries erase their type universes, so the
/// composition is deferred until a lookup supplies the destination universe. The lookup then instantiates a
/// [`CombinedNativeResidualPolicy`] from the native policies of both children. This is the only supported way to
/// combine registries; another combined policy that needs instantiations derived from erased child registries
/// requires its own recipe.
struct NativeResidualPolicyComposition {
    /// Name of the composed policy, used in rejection diagnostics.
    name: &'static str,

    /// Registry of the first policy, whose saves take precedence.
    first: NativeResidualPolicies,

    /// Registry of the policy consulted when the first policy recomputes.
    second: NativeResidualPolicies,
}

/// Candidate of type universe `T` that does not fully project into the universe of a lifted policy, as passed to the
/// projection fallback registered through [`ResidualPolicyReference::with_projection_fallback`], together with which
/// of its types project.
pub struct ProjectionFallbackCandidate<'c, 'o, T: Type> {
    /// Complete candidate.
    candidate: &'c ResidualCandidate<'o, T>,

    /// Whether all input and output types of each producer project, in producer order.
    producer_projectable: Vec<bool>,

    /// Whether the residual type projects.
    residual_projectable: bool,
}

impl<'c, 'o, T: Type> ProjectionFallbackCandidate<'c, 'o, T> {
    /// Returns the complete candidate.
    #[inline]
    pub fn candidate(&self) -> &'c ResidualCandidate<'o, T> {
        self.candidate
    }

    /// Returns whether all input and output types of each producer of the candidate project into the universe of the
    /// lifted policy, in producer order.
    #[inline]
    pub fn producer_projectable(&self) -> &[bool] {
        self.producer_projectable.as_slice()
    }

    /// Returns whether the residual type of the candidate projects into the universe of the lifted policy.
    #[inline]
    pub fn residual_projectable(&self) -> bool {
        self.residual_projectable
    }
}

/// Classifier that a lifted policy uses for the candidates that do not fully project into its universe.
type ProjectionFallback<T> = Arc<
    dyn for<'c, 'o> Fn(
            &ProjectionFallbackCandidate<'c, 'o, T>,
        ) -> Result<ResidualDecision<ErasedResidualStorage<T>>, ResidualRejection>
        + Send
        + Sync,
>;

/// Everything that [`ResidualPolicyReference::lift`] needs to instantiate one policy definition in type universes
/// other than the one it was registered in. For each such universe `U`, it holds at most one native instantiation
/// (i.e., a version of the policy implemented for `U`, which `lift` uses directly) and at most one projection fallback
/// (i.e., the classifier that `lift` uses for the candidates that do not project into the policy's own universe, when
/// there is no native instantiation for `U`).
///
/// A [`ResidualPolicyReference`] shares these instantiations through an [`Arc`], so that its clones and every reference
/// lifted from it, which keep its definition identifier, carry the same instantiations and can themselves be lifted
/// further. Registering a new instantiation creates a new definition with its own identifier instead of changing an
/// existing one (refer to [`ResidualPolicyReference::with_native_instantiation`] and
/// [`ResidualPolicyReference::with_projection_fallback`] for more information).
#[derive(Clone, Default)]
struct ResidualPolicyInstantiations {
    /// Native instantiations, keyed by type universe.
    natives: NativeResidualPolicies,

    /// Projection fallbacks keyed by the [`TypeId`] of their universe `U`, each holding a `ProjectionFallback<U>`.
    fallbacks: Vec<(TypeId, Arc<dyn Any + Send + Sync>)>,
}

impl ResidualPolicyInstantiations {
    /// Returns the [`ProjectionFallback`] that is registered for type universe `U`, if any.
    fn fallback<U: 'static + Type>(&self) -> Option<ProjectionFallback<U>> {
        self.fallbacks
            .iter()
            .find(|(universe, _)| *universe == TypeId::of::<U>())
            .and_then(|(_, fallback)| fallback.downcast_ref::<ProjectionFallback<U>>().cloned())
    }
}

/// Next process-unique identifier of a residual policy definition.
static NEXT_RESIDUAL_POLICY_ID: AtomicU64 = AtomicU64::new(0);

/// Type-erased reference to a [`ResidualPolicy`] instantiated in type universe `T`, which is how programs and
/// transforms carry policies.
///
/// # Identity
///
/// Operations that carry a policy must support equality and hashing, because programs compare and deduplicate their
/// instructions, transform caches key on them, and tests compare program structure. Policies, however, are arbitrary
/// values, often closures or structs holding closures, and neither closures nor trait objects can be compared or
/// hashed. Each reference therefore identifies a policy _definition_ by a process-unique identifier that
/// [`new`](Self::new) assigns, and references compare and hash by that identifier. Operations that carry
/// a policy then compare equal exactly when they carry the same definition.
///
/// Identity is tracked by an identifier rather than by pointer identity because it must survive lifting.
/// For example, promoting an operation that carries a policy from an [`ArrayType`](crate::ArrayType) family into an
/// [`ArrayIrType`](crate::ArrayIrType) family [lifts](Self::lift) its policy into a different type-erased object (i.e.,
/// a native instantiation or a projecting wrapper), yet the promoted operation must still compare equal to its origin
/// and to any other promotion of it. Clones and lifts of a reference therefore keep its identifier.
///
/// Identifiers are conservative. Two separately registered references are distinct even when their policies
/// look equal, because the equality of two closures cannot be decided. The references that the opt-ins
/// [`Self::with_native_instantiation`] and [`Self::with_projection_fallback`] return are new definitions as well,
/// because those opt-ins change what the policy decides, and treating the result as the same definition would let
/// caches return results computed under the old behavior. Identifiers only need to be unique within one process, since
/// policies are never serialized, and they come from a global atomic counter because policies can be registered on any
/// thread.
pub struct ResidualPolicyReference<T: Type> {
    /// Identifier of the policy definition.
    id: u64,

    /// Policy instantiated in universe `T`.
    policy: Arc<dyn ErasedResidualPolicy<T>>,

    /// Instantiations of the policy definition in other type universes.
    instantiations: Arc<ResidualPolicyInstantiations>,
}

impl<T: 'static + Type> ResidualPolicyReference<T> {
    /// Registers `policy` as a new policy definition, together with the
    /// [native instantiations](ResidualPolicy::native_instantiations) that it declares.
    pub fn new<P: ResidualPolicy<T>>(policy: P) -> Self {
        let natives = policy.native_instantiations();
        Self {
            id: NEXT_RESIDUAL_POLICY_ID.fetch_add(1, Ordering::Relaxed),
            policy: Arc::new(NativeResidualPolicy(policy)),
            instantiations: Arc::new(ResidualPolicyInstantiations { natives, fallbacks: Vec::new() }),
        }
    }

    /// Returns a new policy definition that classifies candidates of type universe `U` with `policy` after lifting
    /// into `U`, instead of projecting their types into `T`. This is how custom policies classify candidates that exist
    /// only in `U` (e.g., dimensions or references). `policy` must make the same decisions as this policy for the
    /// candidates whose types project into `T`, which this function cannot check.
    pub fn with_native_instantiation<U: 'static + Type, P: ResidualPolicy<U>>(self, policy: P) -> Self {
        let mut instantiations = (*self.instantiations).clone();
        instantiations.natives = instantiations.natives.with(policy);
        self.redefined(instantiations)
    }

    /// Returns a new policy definition whose lift into type universe `U` classifies the candidates whose types do not
    /// all project into `T` with `fallback`. Candidates whose types project still reach this policy unchanged. The
    /// fallback receives the complete candidate together with which of its types project.
    pub fn with_projection_fallback<
        U: 'static + Type,
        F: 'static
            + Send
            + Sync
            + for<'c, 'o> Fn(
                &ProjectionFallbackCandidate<'c, 'o, U>,
            ) -> Result<ResidualDecision<ErasedResidualStorage<U>>, ResidualRejection>,
    >(
        self,
        fallback: F,
    ) -> Self {
        let mut instantiations = (*self.instantiations).clone();
        let fallback: ProjectionFallback<U> = Arc::new(fallback);
        instantiations.fallbacks.retain(|(universe, _)| *universe != TypeId::of::<U>());
        instantiations.fallbacks.push((TypeId::of::<U>(), Arc::new(fallback)));
        self.redefined(instantiations)
    }

    /// Returns the identifier of the policy definition of this reference.
    #[inline]
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Returns the name of the policy (refer to [`ResidualPolicy::name`]).
    #[inline]
    pub fn name(&self) -> &str {
        self.policy.name()
    }

    /// Classifies `candidate` with the policy, returning its decision with type-erased storage.
    ///
    /// # Errors
    ///
    /// Returns [`ResidualPolicyError::Rejected`] when the policy rejects the candidate and
    /// [`ResidualPolicyError::UnsupportedProjection`] when this is a lifted reference that cannot classify it.
    #[inline]
    pub fn classify(
        &self,
        candidate: &ResidualCandidate<'_, T>,
    ) -> Result<ResidualDecision<ErasedResidualStorage<T>>, ResidualPolicyError> {
        self.policy.classify(candidate)
    }

    /// Returns this policy lifted into a type universe `U` whose types project into `T` (e.g., from
    /// [`ArrayType`](crate::ArrayType) into [`ArrayIrType`](crate::ArrayIrType)), keeping the identifier of its
    /// definition. The lifted policy is the [native instantiation](ResidualPolicy::native_instantiations) of the
    /// policy in `U`, if there is one. Otherwise, it projects the types of each candidate into `T` and classifies
    /// the projected candidate with this policy, using the [projection fallback](Self::with_projection_fallback)
    /// for `U` for the candidates that do not project.
    #[inline]
    pub fn lift<U: 'static + Type>(&self) -> ResidualPolicyReference<U>
    where
        for<'t> &'t T: TryFrom<&'t U>,
    {
        ResidualPolicyReference {
            id: self.id,
            policy: self.instantiations.natives.get::<U>().unwrap_or_else(|| {
                Arc::new(LiftedResidualPolicy::<T, U> {
                    source: self.policy.clone(),
                    fallback: self.instantiations.fallback::<U>(),
                })
            }),
            instantiations: self.instantiations.clone(),
        }
    }

    /// Returns this reference as a new policy definition with the provided instantiations.
    fn redefined(self, instantiations: ResidualPolicyInstantiations) -> Self {
        Self {
            id: NEXT_RESIDUAL_POLICY_ID.fetch_add(1, Ordering::Relaxed),
            policy: self.policy,
            instantiations: Arc::new(instantiations),
        }
    }
}

impl<T: Type> Clone for ResidualPolicyReference<T> {
    #[inline]
    fn clone(&self) -> Self {
        Self { id: self.id, policy: self.policy.clone(), instantiations: self.instantiations.clone() }
    }
}

impl<T: Type> Debug for ResidualPolicyReference<T> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ResidualPolicyReference")
            .field("name", &self.policy.name())
            .field("id", &self.id)
            .finish()
    }
}

impl<T: Type> PartialEq for ResidualPolicyReference<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<T: Type> Eq for ResidualPolicyReference<T> {}

impl<T: Type> Hash for ResidualPolicyReference<T> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl<V: Value<Type: 'static>, O: Operation<Type = V::Type> + OperationPayloadProjection> PartitionedProgram<V, O> {
    /// Returns this partition with the known values that its residual program consumes placed
    /// according to `policy`. This places the residuals of an existing partition at its top level only.
    /// [`Program::partition_with_residual_policy`] instead partitions a program with `policy` in its
    /// [`PartialEvaluationContext`](crate::PartialEvaluationContext), which also places the residuals
    /// of the partitions that split rules construct for the bodies of region-carrying operations.
    ///
    /// Planning starts from the edges that the residual program reads and decides everything before it emits the
    /// resulting programs. It classifies each demanded known value once:
    ///
    ///   - Known inputs are forwarded as edges and constants are re-created in the residual program, without consulting
    ///     the policy. Known inputs are therefore the only inputs that are ever saved, and only when residual work
    ///     needs them.
    ///   - Values whose producers cannot be replayed safely are saved regardless of the policy. This covers producers
    ///     that access references shared with the caller of the known program (e.g., reference inputs, or local
    ///     references that escape through an output), producers with other observable effects, and deferred work.
    ///   - Values whose provenance contains only region inputs or constants are saved without consulting the policy,
    ///     because they have no operation producer to classify. Region-input correspondence does not establish value
    ///     identity: a loop carry, for example, can change on each iteration.
    ///   - Every other value is classified by `policy`. Saved values become edges, possibly through the storage that
    ///     the policy returns, whose store operations apply to the value in the known program and whose restore
    ///     operations reproduce it in the residual program before its first use. Recomputed values mark their
    ///     producers for replay in the residual program, which demands the inputs of those producers in turn. Each
    ///     producer is replayed once, even when several of its outputs are recomputed, and those of its outputs that
    ///     are saved still resolve to their edges.
    ///
    /// This function replays a recomputed output of a region-carrying instruction by replaying the whole instruction.
    /// Placement during partitioning (refer to [`Program::partition_with_residual_policy`]) instead preserves the
    /// values that the policy saves inside the regions of such an instruction. When replaying the instruction would
    /// recompute them, it splits the instruction on residual demand through the partial-evaluation rule of its
    /// operation, if the instruction is pure and that rule supports such a split (as the rules of `condition` and
    /// `scan` do), and saves the output otherwise. The known part of the split replaces the instruction and also
    /// returns the interior values that the policy saves, which become edges without being classified again, and its
    /// replay part recomputes the demanded outputs from those values and from the inputs of the instruction, which are
    /// planned like any other demanded value. A split condition, for example, saves the values of the branch that runs
    /// in shared residual slots and replays the condition over its predicate in the residual program, and a split scan
    /// saves per-iteration values in temporal order and recomputes its demanded carries from their initial values.
    ///
    /// A known region input of a loop body (e.g., a `scan` carry) is therefore saved by the split rule of the loop on
    /// every iteration in which residual work reads it. Under a residual placement, the split rule of `scan` instead
    /// extends the placement of its body to the chain of work that computes such a carry, with the inputs of the body
    /// and the values that the placement already saves available without consulting the policy, and recomputes the
    /// carry in the residual program from its initial value, as JAX's `_scan_partial_eval_custom` does. That extension
    /// splits the region-carrying instructions of the chain on residual demand like placement does (e.g., a `condition`
    /// or an inner `scan` in the body), so the loop saves their interior values on every iteration rather than their
    /// outputs. The initial value is then an ordinary known value of the enclosing partition, which the policy
    /// classifies like any other.
    ///
    /// Operations that declare [`RegionDataFlow::Atomic`] data flow (e.g., composite calls) are never looked through
    /// and never split. The policy classifies their outputs with the operation itself as the only producer, so it sees
    /// neither the work nor the tags inside them (a policy can still match a composite call by its name), and replaying
    /// one of their recomputed outputs replays the whole operation, whatever its regions compute. This is how JAX's
    /// policies treat `custom_jvp_call` and `custom_vjp_call`. A composite call whose rules differentiate it is
    /// replaced by those rules (e.g., by the replayed forward mode rule), whose operations are classified like any
    /// other, so only calls that differentiation leaves intact (e.g., calls whose inputs are all known) stay atomic.
    ///
    /// Replaying an access of local reference state (i.e., of a reference that the known program allocates and that
    /// does not escape it) also replays the allocation and every earlier mutation of that reference, in program order,
    /// so that the replayed access observes the same state. Reference handles never become new edges: residual work
    /// uses local state only through replayed lifecycles, apart from handle edges of the original partition, which are
    /// kept. Effectful known work stays in the known program regardless of demand, except for local reference
    /// lifecycles that nothing in the known program observes anymore, which are removed from it.
    ///
    /// The resulting partition keeps the known inputs, the outputs, and the effect-ordering constraints of this
    /// partition. The residual program keeps its unknown inputs and the edges that it still reads at their relative
    /// positions, followed by the new edges in known-program order, and edges that nothing reads are dropped. The
    /// effect-ordering constraints remain valid because the known program gains no effects and the residual program
    /// gains only complete local reference lifecycles that nothing outside it can observe. Both programs are finally
    /// pruned with [`Program::into_pruned`], so that their region-carrying instructions (e.g., a known `scan` whose
    /// stacked residuals are now recomputed) stop producing values that nothing uses. A policy that saves every value
    /// therefore reproduces this partition, apart from dropping edges that the residual program does not read and
    /// the unused boundaries of region-carrying instructions. Region-input provenance records correspondence rather
    /// than value identity, so it never substitutes an instruction input for one of those instructions' outputs.
    ///
    /// Placement requires a partition whose residual edges have not been
    /// [forwarded](PartitionedProgram::forward_residuals) yet, because it can introduce new edges that repeat known
    /// inputs (e.g., the known input of a recomputed producer), which forwarding then feeds to the residual program
    /// directly. Residuals are therefore placed first and forwarded afterwards.
    ///
    /// # Errors
    ///
    /// Returns the errors of classifying candidates with `policy` (refer to [`ResidualPolicyReference::classify`]),
    /// [`ResidualPolicyError::UnsupportedStorage`] or [`ResidualPolicyError::InvalidStorage`] when a storage cannot be
    /// staged, and [`ResidualPolicyError::Program`] when this partition has forwarded residual inputs (wrapping a
    /// [`ProgramError::InvalidArgument`]) or when rebuilding the programs fails (e.g., when residual work would
    /// require a local reference handle as a new edge).
    #[inline]
    pub fn with_residual_policy(self, policy: &ResidualPolicyReference<V::Type>) -> Result<Self, ResidualPolicyError> {
        self.with_residual_placement(policy, ResidualPlanningMode::TopLevelPlacement, None, None)
    }

    /// Places the residuals of this partition like [`with_residual_policy`](Self::with_residual_policy). When `mode`
    /// preserves nested placements, a requested recomputation of a region-carrying output preserves the existing nested
    /// cuts unless the policy recomputes every live producer needed for that output. Unused sibling outputs and dormant
    /// derivative regions do not contribute demand. Requested saves, storage, and errors are handled directly before
    /// checking whether replay would discard a nested cut. When replay would discard one, `demand_splitter` may split
    /// the instruction on residual demand instead of saving its output (refer to
    /// [`ResidualPlacement::place_residuals`]). When `rounding` is provided, the policy classifies each value that a
    /// rounding operation of `rounding` produces like the value that it rounds (refer to [`RoundedResidualPlacement`]).
    /// This function does not round any value itself.
    fn with_residual_placement(
        self,
        policy: &ResidualPolicyReference<V::Type>,
        mode: ResidualPlanningMode,
        demand_splitter: Option<&ResidualDemandSplitter<'_, V, O>>,
        rounding: Option<&dyn ResidualRounding<V::Type>>,
    ) -> Result<Self, ResidualPolicyError> {
        if self.has_forwarded_residual_inputs() {
            return Err(ProgramError::InvalidArgument {
                message: "cannot place the residuals of a partition whose residual inputs are already forwarded"
                    .to_string(),
            }
            .into());
        }

        let known_output_count = self.outputs().iter().filter(|output| output.is_known()).count();
        let residual_inputs = self.residual_inputs().to_vec();
        let (known_program, residual_program, metadata) = self.into_programs_and_metadata();
        let edges = known_program.output_ids()[known_output_count..].to_vec();

        // Residual demand starts from the edges that the residual program actually reads.
        let mut read = vec![false; residual_program.atoms().len()];
        residual_program
            .instructions()
            .iter()
            .flat_map(|instruction| instruction.inputs())
            .chain(residual_program.output_ids())
            .for_each(|atom| read[atom.index()] = true);
        let seeds = residual_inputs
            .iter()
            .zip(residual_program.input_ids())
            .filter_map(|(input, atom)| match input {
                ResidualInputSource::ResidualEdge(edge) if read[atom.index()] => Some(edges[*edge]),
                _ => None,
            })
            .collect::<Vec<_>>();

        // Plan every demanded known atom before emitting anything, so that emission sees the final decisions.
        let plan = Self::plan_known_atoms(&known_program, policy, mode, &edges, seeds, demand_splitter, rounding)?;
        let plans = &plan.atoms;
        let replay_instructions = &plan.replayed_instructions;
        let splits = &plan.splits;
        let atom_count = known_program.atoms().len();

        // The edges of the resulting partition are the saved atoms, in the order of the residual program inputs that
        // consume them: the original edges that remain edges, at their relative positions among the inputs of the
        // original residual program, followed by the new edges in known-program order, and then by the values that the
        // known parts of split instructions save, in instruction order.
        let is_edge = |atom: &AtomId| matches!(plans[atom.index()], Some(ResidualPlan::Edge(_)));
        let mut new_edges = Vec::new();
        let mut atom_edge_positions = vec![None; atom_count];
        let mut saved_edge_positions = HashMap::new();
        let original_edges = residual_inputs.iter().filter_map(|input| match input {
            ResidualInputSource::ResidualEdge(edge) => Some(edges[*edge]),
            _ => None,
        });
        for atom in original_edges.chain((0..atom_count).map(AtomId::new)) {
            if is_edge(&atom) && atom_edge_positions[atom.index()].is_none() {
                atom_edge_positions[atom.index()] = Some(new_edges.len());
                new_edges.push(ResidualEdgeSource::Atom(atom));
            }
        }
        for (&instruction, split) in splits {
            for value in 0..split.saved_store_counts.len() {
                let edge = ResidualEdgeSource::Saved { instruction, value };
                saved_edge_positions.insert(edge, new_edges.len());
                new_edges.push(edge);
            }
        }

        // Emit the resulting known program, which copies the known program with its known outputs followed by the new
        // edges as outputs. The store operations of stored edges are staged right after their producers, so that the
        // edge outputs carry the stored representations.
        let mut builder = ProgramBuilder::<V, O>::new();
        let mut atoms = vec![None; atom_count];
        let (saved_atoms, stored_atoms) = plan.emit_known_program(&known_program, &mut atoms, &mut builder)?;
        let saved_atoms = saved_atoms
            .into_iter()
            .map(|(instruction, value, atom)| (ResidualEdgeSource::Saved { instruction, value }, atom))
            .collect::<HashMap<_, _>>();

        let mut known_outputs = known_program.output_ids()[..known_output_count]
            .iter()
            .map(|output| copy_atom(&known_program, *output, &mut atoms, &mut builder))
            .collect::<Result<Vec<_>, _>>()?;

        // Each new edge records how many store operations produce it from the known value that it saves, so that
        // precision rounding can find that value later. An original edge that is stored again keeps counting the store
        // operations that already produced it.
        let mut original_store_counts = HashMap::new();
        for (edge, atom) in edges.iter().enumerate() {
            original_store_counts.entry(*atom).or_insert_with(|| metadata.edge_store_count(edge));
        }
        let mut edge_store_counts = Vec::with_capacity(new_edges.len());
        for edge in &new_edges {
            let (known_output, store_count) = match edge {
                ResidualEdgeSource::Atom(atom) => {
                    let original_store_count = original_store_counts.get(atom).copied().unwrap_or(0);
                    match stored_atoms.get(atom) {
                        Some((stored_atom, store_count)) => (*stored_atom, original_store_count + store_count),
                        None => (copy_atom(&known_program, *atom, &mut atoms, &mut builder)?, original_store_count),
                    }
                }
                ResidualEdgeSource::Saved { instruction, value } => {
                    (saved_atoms[edge], splits[instruction].saved_store_counts[*value])
                }
            };
            known_outputs.push(known_output);
            edge_store_counts.push(store_count);
        }

        let edge_types = known_outputs[known_output_count..]
            .iter()
            .map(|output| builder.atoms()[output.index()].r#type().into_owned())
            .collect::<Vec<_>>();

        // The residual program replays the lifecycles of the local references that replayed instructions allocate,
        // so the known program keeps those lifecycles only if it still observes them itself.
        let replayed_allocations = replay_instructions
            .iter()
            .flat_map(|index| {
                let instruction = &known_program.instructions()[*index];
                instruction
                    .operation()
                    .effects()
                    .allocation_output_indices()
                    .filter_map(|output_index| instruction.outputs().get(output_index))
                    .filter_map(|output| atoms[output.index()])
                    .collect::<Vec<_>>()
            })
            .collect::<HashSet<_>>();

        // Region-carrying instructions of the known program may produce values that were edges before and that nothing
        // demands anymore (e.g., per-iteration stacks of a `scan` whose residuals are now recomputed), so the unused
        // boundaries of those instructions are pruned to stop computing them.
        let input_count = known_program.input_ids().len();
        let output_count = known_outputs.len();
        let new_known_program = builder
            .build::<Vec<V>, Vec<V>>(known_outputs, vec![Placeholder; input_count], vec![Placeholder; output_count])?
            .without_unobserved_local_references(&replayed_allocations)?
            .into_simplified()?
            .into_pruned()?;

        // Emit the resulting residual program, starting with its inputs: the unknown inputs and the original edges that
        // remain edges at their original relative positions, followed by the new edges. Each edge becomes one input,
        // even when the original residual program received it more than once.
        let mut builder = ProgramBuilder::<V, O>::new();
        let mut new_residual_inputs = Vec::new();
        let mut residual_atoms = vec![None; residual_program.atoms().len()];
        let mut edge_inputs = HashMap::new();
        let mut add_edge_input = |edge: ResidualEdgeSource, builder: &mut ProgramBuilder<V, O>, inputs: &mut Vec<_>| {
            let position = match edge {
                ResidualEdgeSource::Atom(atom) => atom_edge_positions[atom.index()],
                ResidualEdgeSource::Saved { .. } => saved_edge_positions.get(&edge).copied(),
            };
            if let Some(position) = position
                && !edge_inputs.contains_key(&edge)
            {
                edge_inputs.insert(edge, builder.add_input(edge_types[position].clone()));
                inputs.push(ResidualInputSource::ResidualEdge(position));
            }
        };

        for (input, atom) in residual_inputs.iter().zip(residual_program.input_ids()) {
            match input {
                ResidualInputSource::UnknownInput(index) => {
                    let r#type = residual_program.atoms()[atom.index()].r#type().into_owned();
                    residual_atoms[atom.index()] = Some(builder.add_input(r#type));
                    new_residual_inputs.push(ResidualInputSource::UnknownInput(*index));
                }
                ResidualInputSource::ResidualEdge(edge) => {
                    add_edge_input(ResidualEdgeSource::Atom(edges[*edge]), &mut builder, &mut new_residual_inputs)
                }
                ResidualInputSource::KnownInput(_) | ResidualInputSource::KnownOutput(_) => {
                    unreachable!("forwarded residual inputs are rejected before placement")
                }
            }
        }

        for edge in &new_edges {
            add_edge_input(*edge, &mut builder, &mut new_residual_inputs);
        }

        // Replay the instructions of the known program that are marked for replay and the replay parts of the split
        // instructions, in program order, so that every replayed state access observes the same local reference state
        // as in the known program and every replay part follows the values that it reads.
        let mut known_atoms = vec![None; atom_count];
        plan.emit_replay(&known_program, &mut known_atoms, &mut builder, &edge_inputs)?;

        // Edge inputs of the original residual program resolve to the known atoms that they received. Edges that the
        // original residual program does not read were never demanded, so they have no plan and are dropped.
        for (input, atom) in residual_inputs.iter().zip(residual_program.input_ids()) {
            if let ResidualInputSource::ResidualEdge(edge) = input
                && plans[edges[*edge].index()].is_some()
            {
                residual_atoms[atom.index()] = Some(plan.resolve_atom(
                    &known_program,
                    edges[*edge],
                    &mut known_atoms,
                    &mut builder,
                    &edge_inputs,
                )?);
            }
        }

        // Copy the original residual program on top of the replayed instructions.
        let mut remapping = HashMap::new();
        for instruction in residual_program.instructions() {
            let inputs = instruction
                .inputs()
                .iter()
                .map(|input| copy_atom(&residual_program, *input, &mut residual_atoms, &mut builder))
                .collect::<Result<Vec<_>, _>>()?;
            let outputs = copy_instruction(&residual_program, instruction, inputs, &mut builder, &mut remapping)?;
            for (source, output) in instruction.outputs().iter().zip(outputs) {
                residual_atoms[source.index()] = Some(output);
            }
        }

        let residual_outputs = residual_program
            .output_ids()
            .iter()
            .map(|output| copy_atom(&residual_program, *output, &mut residual_atoms, &mut builder))
            .collect::<Result<Vec<_>, _>>()?;
        let input_count = new_residual_inputs.len();
        let output_count = residual_outputs.len();
        let new_residual_program = builder
            .build::<Vec<V>, Vec<V>>(residual_outputs, vec![Placeholder; input_count], vec![Placeholder; output_count])?
            .into_simplified()?
            .into_pruned()?;

        let metadata = metadata.with_residual_inputs(new_residual_inputs).with_edge_store_counts(edge_store_counts);
        Ok(Self::from_programs_and_metadata(new_known_program, new_residual_program, metadata))
    }

    /// Returns this partition with the known values that it saves for its residual program rounded right after their
    /// producers by the operations that `rounding` returns for their types, whenever the known program also consumes
    /// those values itself, so that the known work that consumes such a value and the residual work that receives it
    /// observe the same value. Backends may compute inexact values at a higher precision than their types (e.g., XLA
    /// computes `bf16` values in `f32` under its default `--xla_allow_excess_precision` setting), and only the edge is
    /// materialized at its declared type, so without the rounding the known consumers of a residual and its residual
    /// consumers could observe different values (refer to [JAX PR #22244](https://github.com/jax-ml/jax/pull/22244),
    /// whose `remat_partial_eval` applies the same rounding for more information).
    ///
    /// The value that an edge saves is the edge itself, unless [`with_residual_policy`](Self::with_residual_policy)
    /// saved it through a [`ResidualStorage`], in which case it is the value that the store operations of that storage
    /// receive. The partition records how many store operations produce each edge, so such a value is rounded before it
    /// is stored, whatever operations the storage uses, and its stored representation is never rounded. Each value is
    /// rounded once, even when several edges save it, and a value that `rounding` reports as the result of its own
    /// rounding operation (refer to [`ResidualRounding::is_rounding`]) is not rounded again, so rounding a partition
    /// whose saved values are already rounded leaves it unchanged. Values that only the residual program or their own
    /// store operations consume, constant edges, values whose types `rounding` returns no operation for, and known
    /// inputs, unless `round_inputs` is set, are left unchanged. Each rounding operation is constructed in the
    /// operation family of this partition through [`OperationPayloadProjection::from_payload`]. Rounding requires a
    /// partition whose residual edges have not been [forwarded](Self::forward_residuals) yet, because a forwarded known
    /// input or output would reach the residual program without passing through the edge that this function rounds.
    ///
    /// # Parameters
    ///
    ///   - `rounding`: Rounding that returns a type-preserving rounding operation for the type of a saved value, or
    ///     `None` when that type needs no rounding under the caller's policy.
    ///   - `round_inputs`: Whether the saved known inputs that the known program also consumes are rounded too, at the
    ///     beginning of the known program. The partitions of regions leave them unrounded, because the program that
    ///     computes their values rounds them when it saves them (refer to [`RoundedResidualPlacement`]).
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError::InvalidArgument`] when this partition has forwarded residual inputs, a
    /// [`ProgramError::MalformedProgram`] when an edge is not the result of the store operations recorded for it, and
    /// another [`ProgramError`] when the operation family cannot hold a rounding operation or when rebuilding the known
    /// program fails.
    pub(crate) fn with_rounded_residuals(
        self,
        rounding: &dyn ResidualRounding<V::Type>,
        round_inputs: bool,
    ) -> Result<Self, ProgramError> {
        if self.has_forwarded_residual_inputs() {
            return Err(ProgramError::InvalidArgument {
                message: "cannot round the residuals of a partition whose residual inputs are already forwarded"
                    .to_string(),
            });
        }

        let known_output_count = self.outputs().iter().filter(|output| output.is_known()).count();
        let (known_program, residual_program, metadata) = self.into_programs_and_metadata();
        let use_counts = known_program.atom_use_counts();
        let instruction_by_output = known_program.instruction_by_output();
        let mut rounded = HashMap::new();
        for (edge, edge_atom) in known_program.output_ids()[known_output_count..].iter().enumerate() {
            // Walk back through the store operations that produce the edge, if any, to the value that it saves.
            let store_count = metadata.edge_store_count(edge);
            let value =
                stored_source(&known_program, &instruction_by_output, *edge_atom, store_count).ok_or_else(|| {
                    ProgramError::MalformedProgram(format!(
                        "residual edge {edge} is not the result of the {store_count} store operations recorded for it",
                    ))
                })?;

            // The saved value needs rounding only when the known program also consumes it outside of its first store
            // operation, if any.
            if !rounded.contains_key(&value)
                && let Some(payload) = saved_value_rounding(
                    &known_program,
                    &instruction_by_output,
                    &use_counts,
                    value,
                    usize::from(store_count > 0),
                    rounding,
                    round_inputs,
                )
            {
                rounded.insert(value, payload);
            }
        }
        if rounded.is_empty() {
            return Ok(Self::from_programs_and_metadata(known_program, residual_program, metadata));
        }

        // The rounding of a value that a store operation receives precedes that store operation, and the rounding of a
        // value that the edge itself saves becomes the edge output.
        let (known_program, _) = round_atoms(&known_program, rounded, &[])?;
        Ok(Self::from_programs_and_metadata(known_program, residual_program, metadata))
    }

    /// Plans the known atoms of `known_program` that residual work demands, starting from `seeds`, as described in the
    /// documentation of [`with_residual_policy`](Self::with_residual_policy), and returns the plan of each atom (or
    /// [`None`] for atoms that nothing demands) together with the positions of the instructions that residual work
    /// replays and the splits of the instructions that `demand_splitter` splits on residual demand, keyed by position.
    /// The atoms of `edges` are the edges of the partition being placed. Reference handles among them stay edges. When
    /// `mode` extends a placement for recomputation, every atom of `edges` has already been placed, and the policy does
    /// not classify it again. When `rounding` is provided, the policy classifies each value that a rounding operation
    /// of `rounding` produces like the value that it rounds (refer to [`ResidualProvenanceAnalysis`]).
    fn plan_known_atoms(
        known_program: &Program<V, O, Vec<V>, Vec<V>>,
        policy: &ResidualPolicyReference<V::Type>,
        mode: ResidualPlanningMode,
        edges: &[AtomId],
        seeds: Vec<AtomId>,
        demand_splitter: Option<&ResidualDemandSplitter<'_, V, O>>,
        rounding: Option<&dyn ResidualRounding<V::Type>>,
    ) -> Result<ResidualProgramPlan<V, O>, ResidualPolicyError> {
        // Local references that escape through an output of the known program (including handle edges) are shared with
        // work outside it, so their lifecycles are not tracked and none of their accesses is ever replayed.
        let region = known_program.entry_region_ref();
        let analysis = region.reference_analysis(0).map_err(ProgramError::from)?;
        let escaping = analysis.output_roots().iter().flatten().copied().collect::<HashSet<_>>();
        let lifecycles = analysis.local_lifecycles(region, |root| !escaping.contains(&root))?;
        let atom_count = known_program.atoms().len();
        let instruction_by_output = known_program.instruction_by_output();
        let mut output_positions = vec![0; atom_count];
        for instruction in known_program.instructions() {
            for (position, output) in instruction.outputs().iter().enumerate() {
                output_positions[output.index()] = position;
            }
        }
        let mut is_edge = vec![false; atom_count];
        edges.iter().for_each(|edge| is_edge[edge.index()] = true);
        let mut is_input = vec![false; atom_count];
        known_program.input_ids().iter().for_each(|input| is_input[input.index()] = true);

        // State predecessors always precede their consumer, so one forward pass determines replayability of every
        // lifecycle prefix. Demanded outputs can then share the result without repeatedly walking those prefixes.
        let replayable = lifecycles.replayable_instructions();

        // Replaying the known half of a nested split must preserve its interior policy decisions. Operations determine
        // the demanded outputs of the regions that they execute, which the analysis uses to cache work per region.
        let mut replay_analysis = ResidualReplayAnalysis::new(known_program, policy);

        // Plans one demanded known atom, applying the rules listed in the documentation of `with_residual_policy`
        // in order: constants, known inputs, and fixed edges never consult the policy, reference handles are either
        // original edges or replayed, values whose producers cannot be replayed are saved, and the policy classifies
        // everything else. The provenance of the known program provides the candidates that the policy classifies,
        // and memoizes the provenance of every visited value across calls.
        let mut provenance = ResidualProvenanceAnalysis::new(known_program, rounding);
        let mut plan = |atom: AtomId| -> Result<ResidualPlan<V::Type>, ResidualPolicyError> {
            let atom_type = match &known_program.atoms()[atom.index()] {
                Atom::Constant(_) => return Ok(ResidualPlan::Constant),
                Atom::Variable(r#type) => r#type.clone(),
            };

            if is_input[atom.index()] || (mode == ResidualPlanningMode::Recomputation && is_edge[atom.index()]) {
                return Ok(ResidualPlan::Edge(None));
            }

            // The `unwrap` is safe because every variable that is not an input is produced by an instruction.
            let index = instruction_by_output[atom.index()].unwrap();
            if atom_type.is_reference() {
                // Residual work uses local reference state only through replayed lifecycles, and never through new
                // handle edges. The handle edges of the original partition stay edges.
                return if is_edge[atom.index()] {
                    Ok(ResidualPlan::Edge(None))
                } else if replayable[index] {
                    Ok(ResidualPlan::Recompute)
                } else {
                    Err(ProgramError::MalformedProgram(format!(
                        "residual work requires the reference produced by operation `{}` as a new edge",
                        known_program.instructions()[index].operation().name(),
                    ))
                    .into())
                };
            }

            if !replayable[index] {
                return Ok(ResidualPlan::Edge(None));
            }

            // Values that only forward region inputs or constants have no producer that a policy could classify.
            let value = ValueId::new(known_program.entry(), atom);
            let Some(candidate) = provenance.candidate(value, atom_type)? else {
                return Ok(ResidualPlan::Edge(None));
            };

            Ok(match policy.classify(&candidate)? {
                ResidualDecision::Save => ResidualPlan::Edge(None),
                ResidualDecision::SaveWith(storage) => ResidualPlan::Edge(Some(storage)),
                ResidualDecision::Recompute => {
                    // Classify the demanded output before consulting the nested replay guard, so a sibling's save
                    // cannot hide its rejection or its requested storage. Only recomputation can overwrite the existing
                    // interior cuts of a nested split and therefore needs this extra validation. When replay would
                    // discard such a cut, the instruction is split on residual demand if its operation supports that,
                    // and its output is saved otherwise.
                    let instruction = &known_program.instructions()[index];
                    let output_index = output_positions[atom.index()];
                    if mode != ResidualPlanningMode::TopLevelPlacement
                        && !instruction.regions().is_empty()
                        && !replay_analysis.allows_replay(index, output_index)?
                    {
                        if demand_splitter.is_some() { ResidualPlan::Split } else { ResidualPlan::Edge(None) }
                    } else {
                        ResidualPlan::Recompute
                    }
                }
            })
        };

        // Discover the plans of all demanded known atoms before emitting anything, so that emission sees the final
        // decisions. Atoms are planned depth-first in seed order, and each one only once. Recomputing an atom marks its
        // producer for replay, and replaying a producer demands its inputs and replays its state predecessors in turn.
        // Splitting an atom requests a split of its producer on residual demand. The loop therefore alternates between
        // draining the pending atoms and marking the next pending instruction for replay, and splits the latest
        // requested instruction once neither remains, until nothing remains. A split demands the inputs that its replay
        // part reads, which precede the instruction, so no demand ever reaches an instruction after it was split.
        let mut plans = (0..atom_count).map(|_| None).collect::<Vec<Option<ResidualPlan<V::Type>>>>();
        let mut replay_instructions = BTreeSet::new();
        let mut split_requests = BTreeMap::<usize, Vec<bool>>::new();
        let mut splits = BTreeMap::new();
        let mut pending_atoms = seeds;
        pending_atoms.reverse();
        let mut pending_instructions = Vec::new();
        loop {
            while let Some(atom) = pending_atoms.pop() {
                if plans[atom.index()].is_some() {
                    continue;
                }
                let atom_plan = plan(atom)?;
                match atom_plan {
                    // The `unwrap`s are safe because only atoms produced by instructions are ever recomputed or split.
                    ResidualPlan::Recompute => pending_instructions.push(instruction_by_output[atom.index()].unwrap()),
                    ResidualPlan::Split => {
                        let index = instruction_by_output[atom.index()].unwrap();
                        let outputs = known_program.instructions()[index].outputs();
                        let demanded_outputs =
                            split_requests.entry(index).or_insert_with(|| vec![false; outputs.len()]);
                        demanded_outputs[output_positions[atom.index()]] = true;
                    }
                    ResidualPlan::Edge(_) | ResidualPlan::Constant => {}
                }
                plans[atom.index()] = Some(atom_plan);
            }

            if let Some(index) = pending_instructions.pop() {
                if replay_instructions.insert(index) {
                    pending_atoms.extend(known_program.instructions()[index].inputs().iter().rev().copied());
                    pending_instructions.extend(lifecycles.state_predecessors(index));
                }
                continue;
            }

            let Some((index, demanded_outputs)) = split_requests.pop_last() else {
                break;
            };

            // An instruction whose operation does not split it saves its demanded outputs instead.
            let instruction = &known_program.instructions()[index];
            let partition = match demand_splitter {
                Some(demand_splitter) => demand_splitter(region, index, &demanded_outputs)?,
                None => None,
            };
            match partition {
                Some(partition) => {
                    let split = ResidualSplit::new(partition, instruction, &demanded_outputs)?;
                    pending_atoms.extend(split.inputs.iter().rev().filter_map(|input| match input {
                        ResidualSplitInput::Input(position) => Some(instruction.inputs()[*position]),
                        ResidualSplitInput::Saved(_) => None,
                    }));
                    splits.insert(index, split);
                }
                None => {
                    for (output, _) in
                        instruction.outputs().iter().zip(&demanded_outputs).filter(|(_, demanded)| **demanded)
                    {
                        plans[output.index()] = Some(ResidualPlan::Edge(None));
                    }
                }
            }
        }

        Ok(ResidualProgramPlan { atoms: plans, replayed_instructions: replay_instructions, splits })
    }
}

/// Scope of a residual-planning request, which determines whether nested placements and existing edges are fixed.
#[derive(Copy, Clone, PartialEq, Eq)]
enum ResidualPlanningMode {
    /// Place an existing partition at its top level, allowing whole-operation replay of region-carrying producers.
    TopLevelPlacement,

    /// Place a freshly partitioned program while preserving the policy decisions made inside its nested regions.
    NestedPlacement,

    /// Extend an existing placement to additional demanded values, preserving both nested decisions and saved edges.
    Recomputation,
}

/// Complete plan of the known work that a residual program demands, computed before any program is emitted.
struct ResidualProgramPlan<V: Value, O: Operation<Type = V::Type>> {
    /// Decision for each known atom, or no decision when nothing demands it.
    atoms: Vec<Option<ResidualPlan<V::Type>>>,

    /// Positions of instructions whose original applications are replayed, in program order.
    replayed_instructions: BTreeSet<usize>,

    /// Known and replay parts of applications split on residual demand, keyed by instruction position.
    splits: BTreeMap<usize, ResidualSplit<V, O>>,
}

impl<V: Value<Type: 'static>, O: Operation<Type = V::Type> + OperationPayloadProjection> ResidualProgramPlan<V, O> {
    /// Emits the known part of one source instruction, followed by the values saved by its split, if any.
    /// The caller assigns the original outputs and saved values to its own boundary wiring.
    fn emit_known_instruction(
        &self,
        source: &Program<V, O, Vec<V>, Vec<V>>,
        index: usize,
        inputs: Vec<AtomId>,
        builder: &mut ProgramBuilder<V, O>,
        remapping: &mut HashMap<RegionId, RegionId>,
    ) -> Result<Vec<AtomId>, ProgramError> {
        match self.splits.get(&index) {
            Some(split) => builder.splice_program(&split.known_program, &inputs),
            None => copy_instruction(source, &source.instructions()[index], inputs, builder, remapping),
        }
    }

    /// Copies `source` into `builder`, replacing each split instruction with its known part and staging the store
    /// operations of each stored edge right after its producer, unless an existing store chain already produces that
    /// representation (refer to [`reusable_store`]). Fills `atoms` with the copy of each source atom that is emitted
    /// and returns the values that the known parts save, as `(instruction, value, atom)` in instruction order,
    /// together with the stored representation and store count of each stored source atom.
    #[allow(clippy::type_complexity)]
    fn emit_known_program(
        &self,
        source: &Program<V, O, Vec<V>, Vec<V>>,
        atoms: &mut [Option<AtomId>],
        builder: &mut ProgramBuilder<V, O>,
    ) -> Result<(Vec<(usize, usize, AtomId)>, HashMap<AtomId, (AtomId, usize)>), ProgramError> {
        for input in source.input_ids() {
            atoms[input.index()] = Some(builder.add_input(source.atoms()[input.index()].r#type().into_owned()));
        }
        let instruction_by_output = source.instruction_by_output();
        let mut saved_atoms = Vec::new();
        let mut stored_atoms = HashMap::new();
        let mut reused_stores = Vec::new();
        let mut remapping = HashMap::new();
        for (index, instruction) in source.instructions().iter().enumerate() {
            let inputs = instruction
                .inputs()
                .iter()
                .map(|input| copy_atom(source, *input, atoms, builder))
                .collect::<Result<Vec<_>, _>>()?;
            let mut outputs = self.emit_known_instruction(source, index, inputs, builder, &mut remapping)?;
            let saved = outputs.split_off(instruction.outputs().len());
            saved_atoms.extend(saved.into_iter().enumerate().map(|(value, atom)| (index, value, atom)));
            for (atom, output) in instruction.outputs().iter().zip(outputs) {
                atoms[atom.index()] = Some(output);
                if let Some(ResidualPlan::Edge(Some(storage))) = &self.atoms[atom.index()] {
                    let residual_type = source.atoms()[atom.index()].r#type().into_owned();
                    let payloads = storage.store_payloads(&residual_type)?;
                    let store_count = payloads.len();
                    match reusable_store(source, *atom, store_count, &**storage, &instruction_by_output) {
                        Some(stored) => reused_stores.push((*atom, stored, store_count)),
                        None => {
                            let stored = stage_storage(builder, output, payloads, &**storage)?;
                            stored_atoms.insert(*atom, (stored, store_count));
                        }
                    }
                }
            }
        }
        for (atom, stored, store_count) in reused_stores {
            stored_atoms.insert(atom, (atoms[stored.index()].unwrap(), store_count));
        }
        Ok((saved_atoms, stored_atoms))
    }

    /// Emits the original applications marked for replay and the replay parts of splits in source order.
    /// Saved outputs keep resolving to their inputs, even when another output replays the same producer.
    fn emit_replay(
        &self,
        source: &Program<V, O, Vec<V>, Vec<V>>,
        atoms: &mut [Option<AtomId>],
        builder: &mut ProgramBuilder<V, O>,
        edge_inputs: &HashMap<ResidualEdgeSource, AtomId>,
    ) -> Result<(), ResidualPolicyError> {
        let mut remapping = HashMap::new();
        for index in self.replayed_instructions.iter().chain(self.splits.keys()).copied().collect::<BTreeSet<_>>() {
            let instruction = &source.instructions()[index];
            if let Some(split) = self.splits.get(&index) {
                let inputs = split
                    .inputs
                    .iter()
                    .map(|input| match input {
                        ResidualSplitInput::Input(position) => {
                            self.resolve_atom(source, instruction.inputs()[*position], atoms, builder, edge_inputs)
                        }
                        ResidualSplitInput::Saved(value) => {
                            Ok(edge_inputs[&ResidualEdgeSource::Saved { instruction: index, value: *value }])
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let outputs = builder.splice_program(&split.residual_program, &inputs)?;
                for (output, atom) in split.demanded_outputs.iter().zip(outputs) {
                    atoms[instruction.outputs()[*output].index()] = Some(atom);
                }
            }
            if !self.replayed_instructions.contains(&index) {
                continue;
            }
            let inputs = instruction
                .inputs()
                .iter()
                .map(|input| self.resolve_atom(source, *input, atoms, builder, edge_inputs))
                .collect::<Result<Vec<_>, _>>()?;
            let outputs = copy_instruction(source, instruction, inputs, builder, &mut remapping)?;
            for (output, atom) in instruction.outputs().iter().zip(outputs) {
                if !matches!(self.atoms[output.index()], Some(ResidualPlan::Edge(_) | ResidualPlan::Split)) {
                    atoms[output.index()] = Some(atom);
                }
            }
        }
        Ok(())
    }

    /// Resolves a demanded known atom through its saved input, replayed result, or re-created constant.
    /// Stored edges are restored once, at first use, and must reproduce the original residual type.
    fn resolve_atom(
        &self,
        known_program: &Program<V, O, Vec<V>, Vec<V>>,
        atom: AtomId,
        atoms: &mut [Option<AtomId>],
        builder: &mut ProgramBuilder<V, O>,
        edge_inputs: &HashMap<ResidualEdgeSource, AtomId>,
    ) -> Result<AtomId, ResidualPolicyError> {
        if let Some(resolved) = atoms[atom.index()] {
            return Ok(resolved);
        }
        let resolved = match &self.atoms[atom.index()] {
            Some(ResidualPlan::Edge(storage)) => {
                let input = edge_inputs[&ResidualEdgeSource::Atom(atom)];
                match storage {
                    None => input,
                    Some(storage) => {
                        let stored_type = builder.atoms()[input.index()].r#type().into_owned();
                        let residual_type = known_program.atoms()[atom.index()].r#type().into_owned();
                        let payloads = storage.restore_payloads(&stored_type, &residual_type)?;
                        let restored = stage_storage(builder, input, payloads, &**storage)?;
                        let restored_type = builder.atoms()[restored.index()].r#type().into_owned();
                        let reproduces = |left: &V::Type, right: &V::Type| {
                            V::Type::derive_identity_renaming(std::slice::from_ref(left), std::slice::from_ref(right))
                                .is_ok()
                        };
                        if !reproduces(&residual_type, &restored_type) || !reproduces(&restored_type, &residual_type) {
                            return Err(ResidualPolicyError::InvalidStorage {
                                storage: storage.name(),
                                message: format!(
                                    "its restore operations produce `{restored_type}` instead of the residual type \
                                     `{residual_type}`",
                                ),
                            });
                        }
                        restored
                    }
                }
            }
            Some(ResidualPlan::Constant) | None => match &known_program.atoms()[atom.index()] {
                Atom::Constant(value) => builder.add_constant(value.clone()),
                Atom::Variable(_) => {
                    return Err(ProgramError::MalformedProgram(format!(
                        "known atom {atom} is neither saved nor recomputed by the residual plan",
                    ))
                    .into());
                }
            },
            Some(ResidualPlan::Recompute | ResidualPlan::Split) => {
                return Err(ProgramError::MalformedProgram(format!(
                    "known atom {atom} is used before the residual program recomputes it",
                ))
                .into());
            }
        };
        atoms[atom.index()] = Some(resolved);
        Ok(resolved)
    }
}

/// Plan of [`PartitionedProgram::with_residual_policy`] for one known atom that residual work demands.
enum ResidualPlan<T: Type> {
    /// The atom is an edge of the resulting partition, stored through the provided storage, if any.
    Edge(Option<ErasedResidualStorage<T>>),

    /// The atom is recomputed in the residual program by replaying its producer.
    Recompute,

    /// The atom is a constant, which the residual program re-creates.
    Constant,

    /// The atom is an output of a region-carrying instruction that is split on residual demand, whose replay part
    /// recomputes it in the residual program.
    Split,
}

/// Split of an all-known region-carrying instruction of a known program on residual demand, as planned by
/// [`PartitionedProgram::plan_known_atoms`]. Its known part replaces the instruction in the known program, and its
/// replay part recomputes the demanded outputs of the instruction in the residual program.
struct ResidualSplit<V: Value, O: Operation<Type = V::Type>> {
    /// Known part, which receives the inputs of the instruction and returns its outputs followed by the values that
    /// the replay part reads from it.
    known_program: Program<V, O, Vec<V>, Vec<V>>,

    /// Replay part, which receives the values that [`inputs`](Self::inputs) describes and returns the demanded outputs.
    residual_program: Program<V, O, Vec<V>, Vec<V>>,

    /// Source of each input of the replay part.
    inputs: Vec<ResidualSplitInput>,

    /// Number of store operations that produce each value that the known part saves for the replay part, in the order
    /// of the outputs of the known part that follow the outputs of the instruction.
    saved_store_counts: Vec<usize>,

    /// Positions of the demanded outputs of the instruction, in output order.
    demanded_outputs: Vec<usize>,
}

impl<V: Value, O: Operation<Type = V::Type>> ResidualSplit<V, O> {
    /// Creates the split of `instruction` on residual demand for the outputs that `demanded_outputs` marks from
    /// `partition`, which a [`ResidualDemandSplitter`] returned for it, validating the wiring of that partition.
    fn new(
        partition: PartitionedProgram<V, O>,
        instruction: &Instruction<O>,
        demanded_outputs: &[bool],
    ) -> Result<Self, ProgramError> {
        let output_count = instruction.outputs().len();
        let demanded_outputs = demanded_outputs
            .iter()
            .enumerate()
            .filter_map(|(index, &demanded)| demanded.then_some(index))
            .collect();
        let expected_outputs = (0..output_count)
            .map(PartialEvaluationOutput::Known)
            .chain((0..Vec::len(&demanded_outputs)).map(PartialEvaluationOutput::Unknown))
            .collect::<Vec<_>>();
        if partition.outputs() != expected_outputs.as_slice()
            || !partition.known_input_indices().iter().copied().eq(0..instruction.inputs().len())
            || partition.has_forwarded_residual_inputs()
        {
            return Err(ProgramError::MalformedProgram(format!(
                "operation `{}` split on residual demand does not follow the wiring of its instruction",
                instruction.operation().name(),
            )));
        }

        // Edges that forward inputs of the instruction become demands on those inputs. Every other edge is a value that
        // the known part saves, which it returns once after the outputs of the instruction.
        let residual_inputs = partition.residual_inputs().to_vec();
        let (known_program, residual_program, metadata) = partition.into_programs_and_metadata();
        let mut inputs = Vec::with_capacity(residual_inputs.len());
        let mut saved_outputs = Vec::new();
        let mut saved_edges = HashMap::new();
        let mut saved_store_counts = Vec::new();
        for input in residual_inputs {
            let ResidualInputSource::ResidualEdge(edge) = input else {
                return Err(ProgramError::MalformedProgram(format!(
                    "operation `{}` split on residual demand reads an unknown input",
                    instruction.operation().name(),
                )));
            };
            let output = known_program.output_ids()[output_count + edge];
            inputs.push(match known_program.input_ids().iter().position(|input| *input == output) {
                Some(position) => ResidualSplitInput::Input(position),
                None => ResidualSplitInput::Saved(*saved_edges.entry(edge).or_insert_with(|| {
                    saved_outputs.push(output_count + edge);
                    saved_store_counts.push(metadata.edge_store_count(edge));
                    saved_outputs.len() - 1
                })),
            });
        }
        let known_outputs = (0..output_count).chain(saved_outputs).collect::<Vec<_>>();
        let known_program = known_program.with_outputs(&known_outputs)?;
        Ok(Self { known_program, residual_program, inputs, saved_store_counts, demanded_outputs })
    }
}

/// Source of one input of the replay part of a [`ResidualSplit`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum ResidualSplitInput {
    /// Input of the split instruction at this position, which is planned like any other demanded value.
    Input(usize),

    /// Value at this position among the values that the known part saves.
    Saved(usize),
}

/// Known value that becomes a residual edge of a partition whose residuals are placed.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
enum ResidualEdgeSource {
    /// Atom of the known program.
    Atom(AtomId),

    /// Value at position `value` among the values that the known part of the split instruction at position
    /// `instruction` saves.
    Saved { instruction: usize, value: usize },
}

/// Placement of the residuals of the partitions of programs over values of type `V`
/// and operations of family `O` according to one residual policy. It is type-erased so that a
/// [`PartialEvaluationContext`](crate::PartialEvaluationContext) can carry a policy into the partitions that the
/// split rules of region-carrying operations construct, without requiring every operation family to support residual
/// placement. [`ResidualPolicyReference`]s implement it for every family that does, and so do
/// [`RoundedResidualPlacement`]s, which also round the values that the placed partitions save.
pub(crate) trait ResidualPlacement<V: Value, O: Operation<Type = V::Type>> {
    /// Returns the policy that places this adapter's residuals, preserving its definition identity.
    fn policy(&self) -> &ResidualPolicyReference<V::Type>;

    /// Returns the type of the [`ResidualRounding`] with which this placement rounds the values that the partitions
    /// that it places save (refer to [`RoundedResidualPlacement`]), or [`None`] when it keeps them unrounded.
    fn rounding_type(&self) -> Option<TypeId>;

    /// Returns `partition` with its residuals placed according to the policy (refer to the documentation of
    /// [`PartitionedProgram::with_residual_policy`] for more information on that). Replaying a demanded output of a
    /// region-carrying operation preserves the saved values that its split rule already placed inside its body. Only
    /// producers needed for that output participate. Unused sibling outputs and dormant derivative regions do not
    /// contribute residual demand.
    ///
    /// When replaying a demanded output of an all-known region-carrying instruction would recompute values that the
    /// policy saves, `demand_splitter` may split that instruction on residual demand (refer to
    /// [`ResidualDemandSplitter`]). Its known part then replaces the instruction in the known program and also returns
    /// the values that the policy saves, which become edges without being classified again, and its replay part
    /// recomputes the demanded outputs in the residual program from those values and from the inputs of the
    /// instruction, which are planned like any other demanded value. Without `demand_splitter`, or when it does not
    /// split the instruction, the demanded outputs are saved.
    fn place_residuals(
        &self,
        partition: PartitionedProgram<V, O>,
        demand_splitter: Option<&ResidualDemandSplitter<'_, V, O>>,
    ) -> Result<PartitionedProgram<V, O>, ProgramError>;

    /// Plans how residual work recomputes the values of the atoms `values` of `known_program`, the known program of
    /// a partition whose residuals were already placed and whose edges are the atoms `saved`, and returns the program
    /// that recomputes them. This extends that placement as if the residual program also read `values`. The inputs of
    /// `known_program` and the atoms of `saved` are available without consulting the policy, and every other demanded
    /// value is planned like a demanded value of [`place_residuals`](Self::place_residuals). Recomputed values are
    /// replayed by the returned program, and the values that the policy saves become additional inputs of it, so that
    /// the decisions of the policy apply to the recomputation as well.
    ///
    /// When replaying a demanded output of an all-known region-carrying instruction would recompute
    /// values that the policy saves, `demand_splitter` may split that instruction on residual demand, as for
    /// [`place_residuals`](Self::place_residuals). The recomputation then also returns `known_program` with the known
    /// part of each split in place of its instruction (refer to [`ResidualRecomputation::known_program`]), and it
    /// receives the values that those known parts save as additional inputs. Without `demand_splitter`, or when it does
    /// not split the instruction, the demanded outputs are inputs of the recomputation instead.
    ///
    /// Policy-based placements save values through their requested [`ResidualStorage`] in the known program and
    /// restore them before their first use in the recomputation. The values that the known part of a split saves are likewise received in the
    /// representation that its replay part reads, which restores them if they are stored.
    fn plan_recomputation(
        &self,
        known_program: &Program<V, O, Vec<V>, Vec<V>>,
        saved: &[AtomId],
        values: &[AtomId],
        demand_splitter: Option<&ResidualDemandSplitter<'_, V, O>>,
    ) -> Result<ResidualRecomputation<V, O>, ProgramError>;
}

/// Function that splits the all-known region-carrying instruction at the provided position of the provided region
/// on residual demand for the outputs that the provided mask marks, and returns the split as a partition of that
/// instruction, or [`None`] when its operation keeps it whole. The partition receives every input of the instruction
/// as a known input. Its known program returns the outputs of the instruction followed by its residual edges, and its
/// residual program returns the demanded outputs, in output order. Partitioning supplies this function to
/// [`ResidualPlacement::place_residuals`] through the partial-evaluation rules of the operations (refer to
/// [`split_on_residual_demand`](crate::PartiallyEvaluatableOperation::split_on_residual_demand)),
/// which the planner cannot reach itself.
pub(super) type ResidualDemandSplitter<'o, V, O> =
    dyn 'o + Fn(RegionRef<'_, V, O>, usize, &[bool]) -> Result<Option<PartitionedProgram<V, O>>, ProgramError>;

/// Program that recomputes known values from values that residual work receives, as planned by
/// [`ResidualPlacement::plan_recomputation`].
pub(crate) struct ResidualRecomputation<V: Value, O: Operation<Type = V::Type>> {
    /// Known program in which the known part of each instruction that the recomputation splits on residual demand
    /// replaces that instruction and in which the values that the recomputation receives are rounded, if the
    /// placement rounds saved values (refer to [`RoundedResidualPlacement`]), or [`None`] when the recomputation
    /// neither splits an instruction, stores a value, nor rounds a value. It has the inputs of the original known
    /// program, and it returns the outputs of the original known program, at the same positions, followed by the
    /// values that the known parts of the splits save and the representations of values newly saved through storage.
    pub(crate) known_program: Option<Program<V, O, Vec<V>, Vec<V>>>,

    /// Atoms of the known program whose values the recomputation [`program`](Self::program) receives, in the order
    /// of its inputs. They are atoms of [`known_program`](Self::known_program) when it is provided, and of the
    /// original known program otherwise. A saved value with a storage is received in its stored representation and
    /// restored by the recomputation before its first use.
    pub(crate) inputs: Vec<AtomId>,

    /// Program that recomputes the requested values from the values of [`inputs`](Self::inputs) and returns them in
    /// request order.
    pub(crate) program: Program<V, O, Vec<V>, Vec<V>>,
}

impl<V: Value<Type: 'static>, O: Operation<Type = V::Type> + OperationPayloadProjection> ResidualRecomputation<V, O> {
    /// Plans the recomputation that [`ResidualPlacement::plan_recomputation`] describes with `policy`, and, when
    /// `rounding` is provided, rounds the values that the recomputation receives from the known program like the saved
    /// values of a partition placed by a [`RoundedResidualPlacement`] with that rounding. The known program then
    /// rounds each of them that it also consumes right after its producer, and the recomputation receives the rounded
    /// values, so that the recomputed values agree with the known ones. The values that `saved` describes and those
    /// that the known parts of splits save are rounded by the placements that saved them, and the policy classifies a
    /// value that a rounding operation produces like the value that it rounds.
    fn plan(
        policy: &ResidualPolicyReference<V::Type>,
        known_program: &Program<V, O, Vec<V>, Vec<V>>,
        saved: &[AtomId],
        values: &[AtomId],
        demand_splitter: Option<&ResidualDemandSplitter<'_, V, O>>,
        rounding: Option<&dyn ResidualRounding<V::Type>>,
    ) -> Result<Self, ProgramError> {
        let plan = PartitionedProgram::plan_known_atoms(
            known_program,
            policy,
            ResidualPlanningMode::Recomputation,
            saved,
            values.to_vec(),
            demand_splitter,
            rounding,
        )?;
        let plans = &plan.atoms;
        let splits = &plan.splits;
        let stores_values = plans.iter().any(|plan| matches!(plan, Some(ResidualPlan::Edge(Some(_)))));

        // The known part of each split replaces its instruction in a copy of the known program, which returns the
        // values that the known part saves after the original outputs. `atoms` maps the atoms of the original known
        // program to those of the copy, or to themselves when nothing is split.
        let mut atoms = (0..known_program.atoms().len()).map(|index| Some(AtomId::new(index))).collect::<Vec<_>>();
        let mut saved_atoms = Vec::new();
        let mut stored_atoms = HashMap::new();
        let rewritten_program = if splits.is_empty() && !stores_values {
            None
        } else {
            let mut builder = ProgramBuilder::<V, O>::new();
            atoms.fill(None);
            (saved_atoms, stored_atoms) = plan.emit_known_program(known_program, &mut atoms, &mut builder)?;
            let mut outputs = known_program
                .output_ids()
                .iter()
                .map(|output| copy_atom(known_program, *output, &mut atoms, &mut builder))
                .collect::<Result<Vec<_>, _>>()?;
            outputs.extend(saved_atoms.iter().map(|(_, _, atom)| *atom));
            outputs.extend(
                plans
                    .iter()
                    .enumerate()
                    .filter(|&(_, plan)| matches!(plan, Some(ResidualPlan::Edge(Some(_)))))
                    .map(|(index, _)| stored_atoms[&AtomId::new(index)].0),
            );
            let input_count = known_program.input_ids().len();
            let output_count = outputs.len();
            Some(builder.build::<Vec<V>, Vec<V>>(
                outputs,
                vec![Placeholder; input_count],
                vec![Placeholder; output_count],
            )?)
        };

        // Every demanded atom that is an edge becomes an input of the recomputation, in known-program order, followed
        // by the values that the known parts of the splits save, in instruction order. Each atom of the known program
        // that the recomputation receives becomes one input, even when it is received more than once.
        let mut builder = ProgramBuilder::<V, O>::new();
        let mut recomputed_atoms = vec![None; known_program.atoms().len()];
        let mut inputs = Vec::new();
        let mut input_store_counts = Vec::new();
        let mut input_atoms = HashMap::new();
        let mut add_input = |atom: AtomId, store_count: usize, r#type: V::Type, builder: &mut ProgramBuilder<V, O>| {
            *input_atoms.entry(atom).or_insert_with(|| {
                inputs.push(atom);
                input_store_counts.push(store_count);
                builder.add_input(r#type)
            })
        };
        let rewritten_region = rewritten_program.as_ref().unwrap_or(known_program).entry_region_ref();
        let mut edge_inputs = HashMap::new();
        for (index, plan) in plans.iter().enumerate() {
            if let Some(ResidualPlan::Edge(storage)) = plan {
                let source = AtomId::new(index);
                let (atom, store_count) = match storage {
                    Some(_) => stored_atoms[&source],
                    None => (atoms[index].unwrap(), 0),
                };
                let atom_type = rewritten_region.atoms()[atom.index()].r#type().into_owned();
                let input = add_input(atom, store_count, atom_type, &mut builder);
                edge_inputs.insert(ResidualEdgeSource::Atom(source), input);
            }
        }
        for &(instruction, value, atom) in &saved_atoms {
            let atom_type = rewritten_region.atoms()[atom.index()].r#type().into_owned();
            let store_count = splits[&instruction].saved_store_counts[value];
            let input = add_input(atom, store_count, atom_type, &mut builder);
            edge_inputs.insert(ResidualEdgeSource::Saved { instruction, value }, input);
        }

        // Placement and carry recomputation share the same ordered replay and saved-output resolution.
        plan.emit_replay(known_program, &mut recomputed_atoms, &mut builder, &edge_inputs)?;

        let outputs = values
            .iter()
            .map(|value| plan.resolve_atom(known_program, *value, &mut recomputed_atoms, &mut builder, &edge_inputs))
            .collect::<Result<Vec<_>, _>>()?;
        let input_count = builder.input_ids().len();
        let output_count = outputs.len();
        let program = builder.build::<Vec<V>, Vec<V>>(
            outputs,
            vec![Placeholder; input_count],
            vec![Placeholder; output_count],
        )?;
        let mut recomputation = ResidualRecomputation { known_program: rewritten_program, inputs, program };
        let Some(rounding) = rounding else {
            return Ok(recomputation);
        };

        // The values that the recomputation receives are rounded right after their producers in the known program,
        // like the saved values of a placed partition, whenever the known program also consumes them.
        let known_program = recomputation.known_program.as_ref().unwrap_or(known_program);
        let use_counts = known_program.atom_use_counts();
        let instruction_by_output = known_program.instruction_by_output();
        let mut rounded = HashMap::new();
        for (&input, &store_count) in recomputation.inputs.iter().zip(&input_store_counts) {
            let value = stored_source(known_program, &instruction_by_output, input, store_count).ok_or_else(|| {
                ProgramError::MalformedProgram(
                    "stored recomputation input is not produced by unary store operations".to_owned(),
                )
            })?;
            if !rounded.contains_key(&value)
                && let Some(payload) = saved_value_rounding(
                    known_program,
                    &instruction_by_output,
                    &use_counts,
                    value,
                    usize::from(store_count > 0),
                    rounding,
                    false,
                )
            {
                rounded.insert(value, payload);
            }
        }
        if !rounded.is_empty() {
            let (known_program, inputs) = round_atoms(known_program, rounded, &recomputation.inputs)?;
            recomputation.known_program = Some(known_program);
            recomputation.inputs = inputs;
        }
        Ok(recomputation)
    }
}

impl<V: Value<Type: 'static>, O: Operation<Type = V::Type> + OperationPayloadProjection> ResidualPlacement<V, O>
    for ResidualPolicyReference<V::Type>
{
    #[inline]
    fn policy(&self) -> &ResidualPolicyReference<V::Type> {
        self
    }

    #[inline]
    fn rounding_type(&self) -> Option<TypeId> {
        None
    }

    #[inline]
    fn place_residuals(
        &self,
        partition: PartitionedProgram<V, O>,
        demand_splitter: Option<&ResidualDemandSplitter<'_, V, O>>,
    ) -> Result<PartitionedProgram<V, O>, ProgramError> {
        Ok(partition.with_residual_placement(self, ResidualPlanningMode::NestedPlacement, demand_splitter, None)?)
    }

    #[inline]
    fn plan_recomputation(
        &self,
        known_program: &Program<V, O, Vec<V>, Vec<V>>,
        saved: &[AtomId],
        values: &[AtomId],
        demand_splitter: Option<&ResidualDemandSplitter<'_, V, O>>,
    ) -> Result<ResidualRecomputation<V, O>, ProgramError> {
        ResidualRecomputation::plan(self, known_program, saved, values, demand_splitter, None)
    }
}

/// Rounding of the values that partitions save for their residual programs, which
/// [`PartitionedProgram::with_rounded_residuals`] and [`RoundedResidualPlacement`] stage right after the producers of
/// those values. A rounding is identified by its type (e.g., in the keys of the partitions that backend call rules
/// retain across calls), so all values of an implementing type must round identically.
pub(crate) trait ResidualRounding<T: Type> {
    /// Returns the type-preserving operation that rounds a saved value of type `r#type`, or [`None`] when values of
    /// that type need no rounding.
    fn rounding_operation(&self, r#type: &T) -> Option<ErasedOperation>;

    /// Returns whether `operation`, which produces a value of type `r#type`, is the operation that
    /// [`rounding_operation`](Self::rounding_operation) returns for that type, so that its result needs no further
    /// rounding. This lets a partition whose saved values an earlier placement already rounded be rounded again without
    /// rounding any value twice.
    fn is_rounding(&self, r#type: &T, operation: &dyn OperationPayloadProjection) -> bool;
}

/// [`ResidualPlacement`] that places residuals according to a policy, like the [`ResidualPolicyReference`] of that
/// policy, and that then rounds the values that each placed partition computes and saves with a [`ResidualRounding`]
/// (refer to [`PartitionedProgram::with_rounded_residuals`]). A
/// [`PartialEvaluationContext`](crate::PartialEvaluationContext) configured with
/// [`with_rounded_residual_policy`](crate::PartialEvaluationContext::with_rounded_residual_policy) carries it into the
/// split rules of region-carrying operations, so it also rounds the values that the partitions of their regions save
/// (e.g., the per-branch values of a `condition`, the per-iteration values of a `scan`, or the values of a `jit_call`
/// callee), including the known parts of their splits on residual demand, right after their producers inside the known
/// region and before the known work of that region consumes them. Rounding them only where the enclosing program
/// receives them would leave that known work using values that a backend computed at a higher precision while the
/// residual program receives rounded ones. The values that the `scan` carry recomputation saves (refer to
/// [`ResidualPlacement::plan_recomputation`]) are rounded in the known body in the same way. The policy classifies a
/// value that a rounding operation produces like the value that it rounds, so the rounding operations that the
/// placements of nested partitions stage do not change the decisions of the enclosing placements.
///
/// The saved known inputs of a placed partition are left unrounded, because their values come from the enclosing
/// program, which rounds them itself when it saves them in turn (e.g., when an input that the known part of a split
/// saves becomes a residual of the enclosing partition), so each value is rounded once. The caller that places the
/// outermost partition rounds the known inputs of that partition itself, as the rematerialization rules do.
pub(crate) struct RoundedResidualPlacement<T: Type, R: ResidualRounding<T>> {
    /// Policy that places the residuals.
    policy: ResidualPolicyReference<T>,

    /// Rounding of the saved values.
    rounding: R,
}

impl<T: Type, R: ResidualRounding<T>> RoundedResidualPlacement<T, R> {
    /// Creates a new [`RoundedResidualPlacement`] that places residuals according to `policy` and rounds the saved
    /// values with `rounding`.
    #[inline]
    pub(crate) fn new(policy: ResidualPolicyReference<T>, rounding: R) -> Self {
        Self { policy, rounding }
    }
}

impl<V, O, R> ResidualPlacement<V, O> for RoundedResidualPlacement<V::Type, R>
where
    V: Value<Type: 'static>,
    O: Operation<Type = V::Type> + OperationPayloadProjection,
    R: 'static + ResidualRounding<V::Type>,
{
    #[inline]
    fn policy(&self) -> &ResidualPolicyReference<V::Type> {
        &self.policy
    }

    #[inline]
    fn rounding_type(&self) -> Option<TypeId> {
        Some(TypeId::of::<R>())
    }

    fn place_residuals(
        &self,
        partition: PartitionedProgram<V, O>,
        demand_splitter: Option<&ResidualDemandSplitter<'_, V, O>>,
    ) -> Result<PartitionedProgram<V, O>, ProgramError> {
        partition
            .with_residual_placement(
                &self.policy,
                ResidualPlanningMode::NestedPlacement,
                demand_splitter,
                Some(&self.rounding),
            )?
            .with_rounded_residuals(&self.rounding, false)
    }

    fn plan_recomputation(
        &self,
        known_program: &Program<V, O, Vec<V>, Vec<V>>,
        saved: &[AtomId],
        values: &[AtomId],
        demand_splitter: Option<&ResidualDemandSplitter<'_, V, O>>,
    ) -> Result<ResidualRecomputation<V, O>, ProgramError> {
        ResidualRecomputation::plan(&self.policy, known_program, saved, values, demand_splitter, Some(&self.rounding))
    }
}

/// Analysis that decides, for one residual policy, whether replaying an output of a region-carrying instruction of the
/// known program of a nested split preserves the residual decisions that the split already made inside the regions of
/// that instruction. Replay recomputes all the work that the regions execute for that output, so it agrees with the
/// policy only when the policy recomputes every value that this work produces, which
/// [`allows_replay`](Self::allows_replay) checks by classifying each of those values.
///
/// The executed work comes from the [`RegionDataFlow`] of each region-carrying operation. Operations that declare
/// [`RegionDataFlow::Atomic`] data flow (e.g., composite calls) are classified like operations without regions, whose
/// replay recomputes only their own outputs, so the analysis never enters their regions. For each region that an
/// instruction executes, it determines the _demanded outputs_ of that region: the outputs that the instruction needs
/// from the region, together with every output that the operation's own semantics need for them (e.g., the carries
/// that later iterations of a loop read). The answer for a region therefore depends only on the region, its demanded
/// outputs, and the policy. The policy is fixed for the lifetime of the analysis, so answers are cached by region and
/// demanded outputs, and a region that several instructions share is classified once for each distinct set of demanded
/// outputs.
///
/// Each distinct set of demanded outputs costs a liveness query whose time is proportional to the boundaries of the
/// visited regions, and every output of a call is asked about separately. A call whose callee returns the residuals of
/// nested calls (e.g., the known side of a nested `jit_call` split, which returns the residuals of every call that it
/// contains) would therefore cost time quadratic in its boundary. The analysis instead first bounds what the traversal
/// would classify with [`ReplayBounds`], which it computes once per region in time proportional to the size of that
/// region, and only traverses the regions when these bounds do not settle the answer.
struct ResidualReplayAnalysis<'o, V: Value, O: Operation<Type = V::Type>> {
    /// [`Program`] containing all source instructions and regions.
    program: &'o Program<V, O, Vec<V>, Vec<V>>,

    /// [`ResidualPolicyReference`] whose decisions replay must agree with.
    policy: &'o ResidualPolicyReference<V::Type>,

    /// [`RegionPruningAnalysis`] that caches the region-local liveness of each set of demanded outputs.
    analysis: RegionPruningAnalysis<'o, V, O>,

    /// Whether replay agrees with the policy for each region and its demanded outputs, accounting for the regions that
    /// the region executes in turn.
    resolved: HashMap<(RegionId, Vec<bool>), bool>,

    /// [`RegionReplayBounds`] indexed by the immutable program arena, allocated when first needed.
    bounds: Vec<Option<RegionReplayBounds>>,
}

impl<'o, V: Value<Type: 'static>, O: Operation<Type = V::Type> + OperationPayloadProjection>
    ResidualReplayAnalysis<'o, V, O>
{
    /// Creates a replay analysis of `program` under `policy`, with empty caches.
    #[inline]
    fn new(program: &'o Program<V, O, Vec<V>, Vec<V>>, policy: &'o ResidualPolicyReference<V::Type>) -> Self {
        Self {
            program,
            policy,
            analysis: RegionPruningAnalysis::new(program.regions()),
            resolved: HashMap::new(),
            bounds: Vec::new(),
        }
    }

    /// Returns whether replaying output `output_index` of the instruction at position `instruction_index` in the
    /// entry region of the program recomputes only values that the policy also recomputes. Returns `true` without
    /// classifying any value when the operation of the instruction declares [`RegionDataFlow::Atomic`] data flow,
    /// because the policy classifies the outputs of such an operation by the operation itself, like those of an
    /// operation without regions, and so its decision for the demanded output already covers the replay. Returns
    /// `false` without classifying any value when the operation of the instruction does not declare which of its
    /// regions produce that output. Otherwise, the [`ReplayBounds`] of the output settle most answers, and the exact
    /// [`traverse`](Self::traverse) settles the others.
    ///
    /// # Errors
    ///
    /// Returns the errors of classifying values with the policy and of the region data flow and liveness queries.
    fn allows_replay(&mut self, instruction_index: usize, output_index: usize) -> Result<bool, ResidualPolicyError> {
        let program = self.program;
        let instruction = &program.instructions()[instruction_index];
        let data_flow = instruction.operation().region_data_flow();
        if matches!(data_flow, RegionDataFlow::Atomic) {
            return Ok(true);
        }
        let regions = program.region_data_flow_boundaries(instruction)?;
        let boundary = RegionDataFlowBoundary {
            input_count: instruction.inputs().len(),
            output_count: instruction.outputs().len(),
            regions: &regions,
        };

        if matches!(
            data_flow.output_sources(instruction.operation(), output_index, boundary)?,
            RegionDataFlowSources::Unknown
        ) {
            // Whole-operation public replay is handled by the caller. A nested split cannot infer precise replay
            // permission for a root whose producing computations have not been declared.
            return Ok(false);
        }

        // The replay bounds of the demanded output usually settle the answer without the exact traversal. Replaying an
        // output of an operation that declares provenance executes every computation region of the operation for the
        // region outputs that produce that output, which the bounds of the output describe directly, so this case
        // also skips the execution demand query, whose time is proportional to the boundary of the instruction.
        let provenance = matches!(data_flow, RegionDataFlow::Provenance);
        if provenance {
            for (region_index, region) in instruction.regions().iter().enumerate() {
                if instruction.operation().region_role(region_index) == Some(RegionRole::Computation) {
                    self.compute_bounds(*region);
                }
            }
            let (outputs, _) = self.executed_bounds(instruction, ReplayBounds::CLEAN, output_index..output_index + 1);
            if let Some(answer) = outputs[0].answer() {
                return Ok(answer);
            }
        }

        let mut outputs = vec![false; instruction.outputs().len()];
        outputs[output_index] = true;
        let roots = self.demands(program.entry(), instruction, &outputs, boundary)?;
        if !provenance {
            let bounds = roots.iter().fold(ReplayBounds::CLEAN, |combined, (region, outputs)| {
                combined.union(self.demand_bounds(*region, outputs))
            });
            if let Some(answer) = bounds.answer() {
                return Ok(answer);
            }
        }
        self.traverse(&roots)
    }

    /// Returns whether executing each region in `roots` for its demanded outputs recomputes only values that the policy
    /// also recomputes, by classifying every value that the regions execute for them, whether or not an earlier value
    /// was saved, so that a saved sibling cannot hide a rejection. This exact traversal is what
    /// [`allows_replay`](Self::allows_replay) answers when the [`ReplayBounds`] of `roots` do not settle the answer.
    ///
    /// # Errors
    ///
    /// Returns the errors of classifying values with the policy and of the region data flow and liveness queries.
    fn traverse(&mut self, roots: &[(RegionId, Vec<bool>)]) -> Result<bool, ResidualPolicyError> {
        let program = self.program;
        let mut pending = roots
            .iter()
            .rev()
            .cloned()
            .map(|(region, outputs)| ResidualReplayFrame::Visit { region, outputs })
            .collect::<Vec<_>>();
        while let Some(frame) = pending.pop() {
            match frame {
                ResidualReplayFrame::Finish { region, outputs, mut recomputes, dependencies } => {
                    // Each dependency was pushed after this frame and therefore resolved before it was popped. Regions
                    // never contain themselves, so no dependency can still be waiting for its own `Finish` frame.
                    for dependency in dependencies {
                        recomputes &= self.resolved[&dependency];
                    }
                    self.resolved.insert((region, outputs), recomputes);
                }
                ResidualReplayFrame::Visit { region, outputs } => {
                    if self.resolved.contains_key(&(region, outputs.clone())) {
                        continue;
                    }

                    let source = program.region_ref(region)?;
                    let selected = outputs
                        .iter()
                        .enumerate()
                        .filter_map(|(index, used)| used.then_some(index))
                        .collect::<Vec<_>>();
                    let live = self.analysis.live_sets(region, &selected)?;
                    let mut recomputes = true;
                    let mut dependencies = Vec::new();
                    for (index, instruction) in source.instructions().iter().enumerate() {
                        if !live.instructions()[index] {
                            continue;
                        }

                        // Atomic operations (e.g., composite calls) are producers of their outputs themselves, like
                        // operations without regions, so their regions are never visited.
                        let used_outputs =
                            instruction.outputs().iter().map(|output| live.atoms()[output.index()]).collect::<Vec<_>>();
                        if instruction.regions().is_empty()
                            || matches!(instruction.operation().region_data_flow(), RegionDataFlow::Atomic)
                        {
                            let atom_type = |atom: &AtomId| source.atoms()[atom.index()].r#type().into_owned();
                            let input_types = instruction.inputs().iter().map(atom_type).collect::<Vec<_>>();
                            let output_types = instruction.outputs().iter().map(atom_type).collect::<Vec<_>>();
                            for (output_index, used) in used_outputs.iter().enumerate() {
                                if !used {
                                    continue;
                                }

                                // Saved siblings must not hide another demanded output's rejection.
                                recomputes &=
                                    self.recomputes_output(instruction, output_index, &input_types, &output_types)?;
                            }
                        } else {
                            let regions = program.region_data_flow_boundaries(instruction)?;
                            let boundary = RegionDataFlowBoundary {
                                input_count: instruction.inputs().len(),
                                output_count: instruction.outputs().len(),
                                regions: &regions,
                            };
                            dependencies.extend(self.demands(region, instruction, &used_outputs, boundary)?);
                        }
                    }

                    pending.push(ResidualReplayFrame::Finish {
                        region,
                        outputs,
                        recomputes,
                        dependencies: dependencies.clone(),
                    });

                    pending.extend(
                        dependencies
                            .into_iter()
                            .rev()
                            .map(|(region, outputs)| ResidualReplayFrame::Visit { region, outputs }),
                    );
                }
            }
        }

        Ok(roots.iter().all(|root| self.resolved[root]))
    }

    /// Returns the demanded outputs of each region that `instruction` executes when only the outputs that
    /// `used_outputs` marks are used, according to the [`RegionDataFlow`](crate::RegionDataFlow) of its operation,
    /// marked in the order of the outputs of each region. An operation that declares no region data flow conservatively
    /// demands every output of each of its computation regions.
    ///
    /// # Parameters
    ///
    ///   - `region`: Region of the program that contains `instruction`.
    ///   - `instruction`: Region-carrying instruction of the program.
    ///   - `used_outputs`: Outputs of `instruction` that are used, marked in the order of its outputs.
    ///   - `boundary`: Boundary of `instruction` and its attached regions, as constructed from
    ///     [`Program::region_data_flow_boundaries`].
    ///
    /// # Errors
    ///
    /// Returns the errors of the region data flow and region liveness queries.
    fn demands(
        &mut self,
        region: RegionId,
        instruction: &Instruction<O>,
        used_outputs: &[bool],
        boundary: RegionDataFlowBoundary<'_>,
    ) -> Result<Vec<(RegionId, Vec<bool>)>, ProgramError> {
        let operation = instruction.operation();
        let atoms = self.program.region_ref(region)?.atoms();
        let input_types = instruction
            .inputs()
            .iter()
            .map(|input| atoms[input.index()].r#type().into_owned())
            .collect::<Vec<_>>();
        let mut regions =
            AttachedRegionLiveness::new(&mut self.analysis, operation, &input_types, instruction.regions());
        let demands =
            operation.region_data_flow().execution_demands(operation, used_outputs, boundary, &mut regions)?;
        Ok(instruction
            .regions()
            .iter()
            .zip(demands)
            .filter_map(|(&region, outputs)| outputs.map(|outputs| (region, outputs)))
            .collect())
    }

    /// Returns the [`ReplayBounds`] of executing `region` for the demanded outputs that `outputs` marks, which combine
    /// the bounds of the work that the region retains whatever it demands with the bounds of each demanded output.
    fn demand_bounds(&mut self, region: RegionId, outputs: &[bool]) -> ReplayBounds {
        self.compute_bounds(region);
        match self.bounds.get(region.index()).and_then(Option::as_ref) {
            Some(bounds) if bounds.outputs.len() == outputs.len() => outputs
                .iter()
                .zip(&bounds.outputs)
                .filter(|(demanded, _)| **demanded)
                .fold(bounds.retained, |combined, (_, output)| combined.union(*output)),
            _ => ReplayBounds::UNKNOWN,
        }
    }

    /// Computes the [`RegionReplayBounds`] of `region` and of every computation region that it executes, unless they
    /// were computed already. Regions precede the regions that attach them in the arena, so computing the bounds in
    /// increasing identifier order finds the bounds of every executed region already computed without recursion. The
    /// regions of operations that declare [`RegionDataFlow::Atomic`] data flow are never executed by the analysis.
    fn compute_bounds(&mut self, region: RegionId) {
        if self.bounds.get(region.index()).is_some_and(Option::is_some) {
            return;
        }
        let region_count = self.program.regions().len();
        if region.index() >= region_count {
            return;
        }
        self.bounds.resize_with(region_count, || None);
        let mut regions = Vec::new();
        let mut pending = vec![region];
        let mut visited = vec![false; region_count];
        while let Some(region) = pending.pop() {
            if self.bounds[region.index()].is_some() || std::mem::replace(&mut visited[region.index()], true) {
                continue;
            }
            regions.push(region);
            let Ok(source) = self.program.region_ref(region) else {
                continue;
            };
            for instruction in source.instructions() {
                let operation = instruction.operation();
                if !matches!(operation.region_data_flow(), RegionDataFlow::Atomic) {
                    pending.extend(instruction.regions().iter().enumerate().filter_map(|(index, region)| {
                        (operation.region_role(index) == Some(RegionRole::Computation)).then_some(*region)
                    }));
                }
            }
        }
        regions.sort_unstable_by_key(|region| region.index());
        for region in regions {
            let bounds = self.region_bounds(region);
            self.bounds[region.index()] = Some(bounds);
        }
    }

    /// Returns the [`RegionReplayBounds`] of `region`, given the bounds of every computation region that its
    /// instructions execute. One forward pass over the instructions of the region bounds each of its atoms as follows,
    /// where the inputs of the region and constants classify nothing in the region itself:
    ///
    ///   - The output of an operation without regions, or of an operation that declares [`RegionDataFlow::Atomic`]
    ///     data flow, is classified by the policy, and replay may also classify whatever the inputs of the operation
    ///     may. Replay certainly classifies the inputs of an operation without regions, whose boundary is never pruned,
    ///     while it may prune the inputs of an atomic operation.
    ///   - The output of another region-carrying operation may classify whatever its inputs, the work that its
    ///     computation regions retain, and the region outputs that its [`RegionDataFlow::Provenance`] declaration names
    ///     as its producers may, and certainly classifies what those region outputs certainly classify. Without such a
    ///     declaration, it may classify whatever its regions may for any demand.
    ///
    /// Instructions that the region retains when their outputs are unused (e.g., for their effects) may classify
    /// whatever their inputs and regions may for every demand of the region. Errors of the policy and of region data
    /// flow queries are bounds that may fail rather than errors, so that only the exact traversal reports them.
    fn region_bounds(&self, region: RegionId) -> RegionReplayBounds {
        let Ok(source) = self.program.region_ref(region) else {
            return RegionReplayBounds {
                outputs: Vec::new(),
                retained: ReplayBounds::UNKNOWN,
                all: ReplayBounds::UNKNOWN,
            };
        };

        let mut atoms = vec![ReplayBounds::CLEAN; source.atoms().len()];
        let mut retained = ReplayBounds::CLEAN;
        for (index, instruction) in source.instructions().iter().enumerate() {
            let inputs = instruction
                .inputs()
                .iter()
                .fold(ReplayBounds::CLEAN, |combined, input| combined.union(atoms[input.index()]));
            let (outputs, executed) = if instruction.regions().is_empty()
                || matches!(instruction.operation().region_data_flow(), RegionDataFlow::Atomic)
            {
                let inputs = if instruction.regions().is_empty() { inputs } else { inputs.possible() };
                let atom_type = |atom: &AtomId| source.atoms()[atom.index()].r#type().into_owned();
                let input_types = instruction.inputs().iter().map(atom_type).collect::<Vec<_>>();
                let output_types = instruction.outputs().iter().map(atom_type).collect::<Vec<_>>();
                let outputs = (0..instruction.outputs().len())
                    .map(|output_index| {
                        let classified =
                            match self.recomputes_output(instruction, output_index, &input_types, &output_types) {
                                Ok(true) => ReplayBounds::CLEAN,
                                Ok(false) => ReplayBounds::SAVED,
                                Err(_) => ReplayBounds::UNKNOWN,
                            };
                        classified.union(inputs)
                    })
                    .collect::<Vec<_>>();
                (outputs, inputs.possible())
            } else {
                self.executed_bounds(instruction, inputs.possible(), 0..instruction.outputs().len())
            };
            match source.instruction_effects(index) {
                Ok(effects) if effects.is_retained_when_unused() => retained = retained.union(executed),
                Ok(_) => {}
                Err(_) => retained = retained.union(ReplayBounds::UNKNOWN),
            }
            for (output, bounds) in instruction.outputs().iter().zip(outputs) {
                atoms[output.index()] = bounds;
            }
        }

        let outputs =
            source.output_ids().iter().map(|output| atoms[output.index()].union(retained)).collect::<Vec<_>>();
        let all = outputs.iter().fold(retained, |combined, output| combined.union(*output));
        RegionReplayBounds { outputs, retained, all }
    }

    /// Returns the [`ReplayBounds`] of the outputs in `outputs` of a region-carrying `instruction` whose operation does
    /// not declare [`RegionDataFlow::Atomic`] data flow, in order, together with the bounds of executing it for any
    /// demand, given the possible bounds of its inputs (refer to [`region_bounds`](Self::region_bounds)). The bounds
    /// of every computation region of `instruction` must have been computed.
    fn executed_bounds(
        &self,
        instruction: &Instruction<O>,
        inputs: ReplayBounds,
        outputs: Range<usize>,
    ) -> (Vec<ReplayBounds>, ReplayBounds) {
        let operation = instruction.operation();
        let output_count = outputs.len();
        let mut regions = Vec::with_capacity(instruction.regions().len());
        let mut retained = inputs;
        let mut all = inputs;
        for (region_index, region) in instruction.regions().iter().enumerate() {
            if operation.region_role(region_index) != Some(RegionRole::Computation) {
                regions.push(None);
                continue;
            }
            let Some(bounds) = self.bounds.get(region.index()).and_then(Option::as_ref) else {
                return (vec![ReplayBounds::UNKNOWN; output_count], ReplayBounds::UNKNOWN);
            };
            retained = retained.union(bounds.retained);
            all = all.union(bounds.all.possible());
            regions.push(Some(bounds));
        }

        let data_flow = operation.region_data_flow();
        let Ok(boundaries) = self.program.region_data_flow_boundaries(instruction) else {
            return (vec![ReplayBounds::UNKNOWN; output_count], ReplayBounds::UNKNOWN);
        };
        let boundary = RegionDataFlowBoundary {
            input_count: instruction.inputs().len(),
            output_count: instruction.outputs().len(),
            regions: &boundaries,
        };
        if !matches!(data_flow, RegionDataFlow::Provenance) {
            return (vec![all; output_count], all);
        }

        // Executing the operation for a used output demands the region outputs that produce it.
        let outputs = outputs
            .map(|output_index| match data_flow.output_sources(operation, output_index, boundary) {
                Ok(RegionDataFlowSources::Known(sources)) => {
                    sources.iter().fold(retained, |combined, source| match source {
                        RegionDataFlowSource::RegionOutput(origin) => combined.union(
                            regions
                                .get(origin.region_index)
                                .copied()
                                .flatten()
                                .and_then(|bounds| bounds.outputs.get(origin.output_index).copied())
                                .unwrap_or(ReplayBounds::UNKNOWN),
                        ),
                        _ => combined,
                    })
                }
                Ok(RegionDataFlowSources::Unknown) => all,
                Err(_) => ReplayBounds::UNKNOWN,
            })
            .collect::<Vec<_>>();
        (outputs, all)
    }

    /// Returns whether the policy recomputes output `output_index` of `instruction`, classified as a value whose only
    /// producer is that output, given the types of the inputs and outputs of `instruction`.
    ///
    /// # Errors
    ///
    /// Returns the errors of classifying the value with the policy.
    fn recomputes_output(
        &self,
        instruction: &Instruction<O>,
        output_index: usize,
        input_types: &[V::Type],
        output_types: &[V::Type],
    ) -> Result<bool, ResidualPolicyError> {
        let producer =
            ResidualProducer::new(instruction.operation(), output_index, input_types.to_vec(), output_types.to_vec());
        let candidate = ResidualCandidate::new(vec![producer], output_types[output_index].clone());
        Ok(matches!(self.policy.classify(&candidate)?, ResidualDecision::Recompute))
    }
}

/// Bounds on what [`ResidualReplayAnalysis::allows_replay`] classifies when replay recomputes a value, which settle
/// most answers without its exact traversal. The bounds over-approximate the values that the traversal may classify,
/// by assuming that every region-carrying operation keeps all of its inputs, and under-approximate the values that it
/// certainly classifies, by following only the producers that liveness never prunes. If no value that the traversal
/// may classify is saved, replay is allowed, and if a value that it certainly classifies is saved while no value that
/// it may classify fails to be classified, replay is not allowed.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct ReplayBounds {
    /// Whether the traversal may classify a value that the policy does not recompute or fails to classify.
    may_save: bool,

    /// Whether the traversal may fail to classify a value.
    may_fail: bool,

    /// Whether the traversal certainly classifies a value that the policy does not recompute.
    must_save: bool,
}

impl ReplayBounds {
    /// Bounds of values whose replay classifies nothing.
    const CLEAN: Self = Self { may_save: false, may_fail: false, must_save: false };

    /// Bounds of values that the policy saves.
    const SAVED: Self = Self { may_save: true, may_fail: false, must_save: true };

    /// Bounds of values about which nothing is known.
    const UNKNOWN: Self = Self { may_save: true, may_fail: true, must_save: false };

    /// Returns the bounds of replaying the values bounded by `self` and by `other` together.
    #[inline]
    fn union(self, other: Self) -> Self {
        Self {
            may_save: self.may_save || other.may_save,
            may_fail: self.may_fail || other.may_fail,
            must_save: self.must_save || other.must_save,
        }
    }

    /// Returns the answer of [`ResidualReplayAnalysis::allows_replay`] that these bounds settle, if any. Settling a
    /// `false` answer is only exact when the exact traversal could not fail, because that traversal keeps classifying
    /// values after finding a saved one, so that a sibling's save cannot hide a rejection.
    #[inline]
    fn answer(self) -> Option<bool> {
        if !self.may_save {
            Some(true)
        } else if self.must_save && !self.may_fail {
            Some(false)
        } else {
            None
        }
    }

    /// Returns these bounds for values that replay may not reach (e.g., inputs that pruning may drop), which keeps
    /// what the traversal may classify and drops what it certainly classifies.
    #[inline]
    fn possible(self) -> Self {
        Self { must_save: false, ..self }
    }
}

/// [`ReplayBounds`] of executing one region of a [`ResidualReplayAnalysis`] (refer to
/// [`ResidualReplayAnalysis::region_bounds`]).
struct RegionReplayBounds {
    /// Bounds of executing the region for each of its outputs, in output order, which include those of `retained`.
    outputs: Vec<ReplayBounds>,

    /// Bounds of the work that the region executes whatever it demands.
    retained: ReplayBounds,

    /// Bounds of executing the region for any set of demanded outputs.
    all: ReplayBounds,
}

/// Work item of the iterative traversal in [`ResidualReplayAnalysis::allows_replay`]. Visiting a region classifies its
/// own demanded values and schedules the regions that its instructions execute, and finishing it, after those regions
/// have been resolved, combines their answers into its own. This postorder traversal avoids recursion and resolves a
/// region that several instructions share only once for each distinct set of demanded outputs.
enum ResidualReplayFrame {
    /// Visits a region with its demanded outputs.
    Visit {
        /// Region to visit.
        region: RegionId,

        /// Demanded outputs of the region, marked in the order of its outputs.
        outputs: Vec<bool>,
    },

    /// Publishes the answer for a region once the answers of all the regions that it executes are available.
    Finish {
        /// Region whose answer is published.
        region: RegionId,

        /// Demanded outputs of the region, marked in the order of its outputs.
        outputs: Vec<bool>,

        /// Whether the policy recomputes every demanded value that the region's own instructions produce.
        recomputes: bool,

        /// Regions, with their demanded outputs, whose answers contribute to the answer for this region.
        dependencies: Vec<(RegionId, Vec<bool>)>,
    },
}

/// Analysis that resolves the operation outputs that may have produced the values of a [`Program`].
/// It follows the [`RegionDataFlowSource`]s that region-carrying operations declare through their
/// [`RegionDataFlow`](crate::RegionDataFlow), across attached regions, initial bindings, and recurrent feedback
/// (e.g., loop carries). A region-carrying operation that declares no data flow, or whose declared sources are unknown,
/// is itself kept as a possible producer. This includes every operation that declares [`RegionDataFlow::Atomic`] data
/// flow (e.g., a composite call), which is never looked through.
///
/// Every visited value is summarized symbolically, in terms of its containing region's inputs, and each summary is
/// computed once. Region summaries are instantiated separately at every call site. A region that several operations
/// invoke (e.g., with differently tagged inputs) therefore resolves to each call site's own input producers.
///
/// The rounding operations of a [`ResidualRounding`], if one is provided, are looked through as well, so that a value
/// that such an operation rounds resolves to the producers of the value that it rounds. Placements that round saved
/// values (refer to [`RoundedResidualPlacement`]) stage those operations inside the regions whose residuals they place
/// before an enclosing placement classifies the outputs of those regions, which then classifies a rounded saved value
/// like the value that the policy saved (e.g., by its tag) rather than as the result of the rounding operation.
struct ResidualProvenanceAnalysis<'o, V: Value, O: Operation<Type = V::Type>> {
    /// Program whose values are resolved.
    program: &'o Program<V, O, Vec<V>, Vec<V>>,

    /// Rounding whose operations are looked through, if any.
    rounding: Option<&'o dyn ResidualRounding<V::Type>>,

    /// Producer and input positions, indexed once for each region that resolution visits.
    positions: Vec<Option<ResidualAtomPositions>>,

    /// Symbolic provenance indexed by containing region and atom, independent of callers. A region's atom slots
    /// are allocated only when its first value is resolved, alongside the already region-local position analysis.
    resolved: Vec<Vec<Option<ResidualProvenanceLeaves>>>,
}

impl<'o, V: Value<Type: 'static>, O: Operation<Type = V::Type> + OperationPayloadProjection>
    ResidualProvenanceAnalysis<'o, V, O>
{
    /// Creates a new [`ResidualProvenanceAnalysis`] of `program` that looks through the rounding operations of
    /// `rounding`, if provided, with empty caches.
    #[inline]
    fn new(program: &'o Program<V, O, Vec<V>, Vec<V>>, rounding: Option<&'o dyn ResidualRounding<V::Type>>) -> Self {
        Self {
            program,
            rounding,
            positions: (0..program.regions().len()).map(|_| None).collect(),
            resolved: (0..program.regions().len()).map(|_| Vec::new()).collect(),
        }
    }

    /// Returns the [`ResidualCandidate`] that describes `value` to a [`ResidualPolicy`], which classifies it to decide
    /// whether residual work receives `value` as a saved residual or recomputes it. The candidate lists every operation
    /// output that may have produced `value`, in the order of its [`resolve`](Self::resolve)d provenance. Operation
    /// outputs that only forward an output of an attached region are looked through, so that the candidate lists the
    /// operations inside those regions that may actually produce the value (e.g., one producer for each branch of a
    /// `condition`). Operations that declare [`RegionDataFlow::Atomic`] data flow (e.g., composite calls) are the
    /// producers of their outputs themselves, so policies classify them by their payloads and never see the work or
    /// the tags inside them.
    ///
    /// Returns [`None`] when no operation output may have produced `value` (i.e., when every path of its provenance
    /// ends at an input or a constant of the program, such as a `condition` output that both branches forward from
    /// the same input). Such a value has no producer that a policy could classify, so
    /// [`PartitionedProgram::with_residual_policy`] saves it without consulting the policy.
    ///
    /// # Parameters
    ///
    ///   - `value`: Value of the program whose producers are resolved.
    ///   - `residual_type`: Type of `value`, which the candidate reports as the type of the residual.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] for malformed region data flow declarations, and
    /// [`ProgramError::UnboundAtomId`] when `value` or one of its sources does not exist in the program.
    fn candidate(
        &mut self,
        value: ValueId,
        residual_type: V::Type,
    ) -> Result<Option<ResidualCandidate<'o, V::Type>>, ProgramError> {
        let program = self.program;
        let mut producers = Vec::new();
        for leaf in self.resolve(value)?.as_slice().iter().copied() {
            if let ResidualProvenanceLeaf::Producer(value) = leaf {
                // Producer leaves are instruction outputs, so their indexed producer always exists.
                let (instruction_index, output_index) =
                    self.positions(value.region())?.producers[value.atom().index()].unwrap();
                let region = program.region(value.region())?;
                let instruction = &region.instructions()[instruction_index];
                let atom_type = |atom: &AtomId| region.atoms()[atom.index()].r#type().into_owned();
                producers.push(ResidualProducer::new(
                    instruction.operation(),
                    output_index,
                    instruction.inputs().iter().map(atom_type).collect(),
                    instruction.outputs().iter().map(atom_type).collect(),
                ));
            }
        }
        Ok((!producers.is_empty()).then(|| ResidualCandidate::new(producers, residual_type)))
    }

    /// Returns the positions of the producers and inputs of `region`, indexing its atoms on the first visit.
    fn positions(&mut self, region: RegionId) -> Result<&ResidualAtomPositions, ProgramError> {
        if self.positions.get(region.index()).is_none_or(Option::is_none) {
            let source = self.program.region(region)?;
            let mut producers = vec![None; source.atoms().len()];
            let mut inputs = vec![None; source.atoms().len()];
            for (instruction_index, instruction) in source.instructions().iter().enumerate() {
                for (output_index, output) in instruction.outputs().iter().enumerate() {
                    producers[output.index()] = Some((instruction_index, output_index));
                }
            }
            for (input_index, input) in source.input_ids().iter().enumerate() {
                inputs[input.index()] = Some(input_index);
            }
            self.positions[region.index()] = Some(ResidualAtomPositions { producers, inputs });
        }
        Ok(self.positions[region.index()].as_ref().unwrap())
    }

    /// Returns the symbolic provenance of `value`, in semantic order and without duplicates. Every value is resolved
    /// once, including caller values that several branches share. Region inputs remain symbolic until their caller
    /// instantiates them, so caching never confuses callers with differently tagged inputs.
    fn resolve(&mut self, value: ValueId) -> Result<ResidualProvenanceLeaves, ProgramError> {
        if let Some(Some(leaves)) =
            self.resolved.get(value.region().index()).and_then(|region| region.get(value.atom().index()))
        {
            return Ok(leaves.clone());
        }
        let leaves = self.resolve_uncached(value)?;
        if self.resolved[value.region().index()].is_empty() {
            let atom_count = self.program.region(value.region())?.atoms().len();
            self.resolved[value.region().index()].resize_with(atom_count, || None);
        }
        self.resolved[value.region().index()][value.atom().index()] = Some(leaves.clone());
        Ok(leaves)
    }

    /// Computes the symbolic provenance of `value` for [`resolve`](Self::resolve), which caches it.
    fn resolve_uncached(&mut self, value: ValueId) -> Result<ResidualProvenanceLeaves, ProgramError> {
        let program = self.program;
        let positions = self.positions(value.region())?;
        let producer = positions
            .producers
            .get(value.atom().index())
            .ok_or(ProgramError::UnboundAtomId { id: value.atom() })?;

        // Inputs stay symbolic and constants have no provenance. The caller of this region substitutes each input
        // through its own input bindings, without changing the cached region-local summary.
        let Some((instruction_index, output_index)) = *producer else {
            return Ok(match positions.inputs[value.atom().index()] {
                Some(input) => ResidualProvenanceLeaves::One(ResidualProvenanceLeaf::Input(input)),
                None => ResidualProvenanceLeaves::Empty,
            });
        };
        let region = program.region(value.region())?;
        let instruction = &region.instructions()[instruction_index];

        // A value that a rounding operation rounds resolves to the provenance of the value that it rounds.
        if let Some(rounding) = self.rounding
            && let [input] = instruction.inputs()
            && rounding.is_rounding(&region.atoms()[value.atom().index()].r#type(), instruction.operation())
        {
            return self.resolve(ValueId::new(value.region(), *input));
        }

        // Region-local summaries stay symbolic. Initial bindings and recurrent edges belong to the attachment,
        // so a shared body can be used by ordinary and recurrent carriers without contaminating its cached summary.
        let operation = instruction.operation();
        let data_flow = operation.region_data_flow();
        let regions = program.region_data_flow_boundaries(instruction)?;
        let boundary = RegionDataFlowBoundary {
            input_count: instruction.inputs().len(),
            output_count: instruction.outputs().len(),
            regions: &regions,
        };

        let RegionDataFlowSources::Known(sources) = data_flow.output_sources(operation, output_index, boundary)? else {
            return Ok(ResidualProvenanceLeaves::One(ResidualProvenanceLeaf::Producer(value)));
        };

        // Forward a complete immutable summary when no caller-specific input binding needs substitution. This
        // avoids allocating an expansion worklist and deduplication sets for ordinary one-output call forwarding.
        match sources.as_slice() {
            [] => return Ok(ResidualProvenanceLeaves::Empty),
            [RegionDataFlowSource::InstructionInput(index)] => {
                return self.resolve(ValueId::new(value.region(), instruction.inputs()[*index]));
            }
            [RegionDataFlowSource::RegionOutput(origin)] => {
                let region = instruction.regions()[origin.region_index];
                let output = program.region(region)?.output_ids()[origin.output_index];
                let leaves = self.resolve(ValueId::new(region, output))?;
                if leaves.as_slice().iter().all(|leaf| matches!(leaf, ResidualProvenanceLeaf::Producer(_))) {
                    return Ok(leaves);
                }
            }
            _ => {}
        }

        let mut pending = sources.into_iter().rev().map(ResidualProvenanceSource::Boundary).collect::<Vec<_>>();
        let mut expanded_inputs = HashSet::new();
        let mut expanded_outputs = HashSet::new();
        let mut leaves = Vec::new();
        let mut seen = HashSet::new();
        while let Some(source) = pending.pop() {
            let replacements = match source {
                ResidualProvenanceSource::Producer(producer) => {
                    ResidualProvenanceLeaves::One(ResidualProvenanceLeaf::Producer(producer))
                }
                ResidualProvenanceSource::Boundary(RegionDataFlowSource::InstructionInput(index)) => {
                    self.resolve(ValueId::new(value.region(), instruction.inputs()[index]))?
                }
                ResidualProvenanceSource::Boundary(RegionDataFlowSource::RegionInput { region_index, input_index }) => {
                    if !expanded_inputs.insert((region_index, input_index)) {
                        continue;
                    }
                    match data_flow.input_sources(operation, region_index, input_index, boundary)? {
                        RegionDataFlowSources::Unknown => {
                            ResidualProvenanceLeaves::One(ResidualProvenanceLeaf::Producer(value))
                        }
                        RegionDataFlowSources::Known(sources) => {
                            // Complete each declared alternative before the next. Region input/output positions
                            // terminate cycles without changing the region-local symbolic summaries.
                            pending.extend(sources.into_iter().rev().map(ResidualProvenanceSource::Boundary));
                            ResidualProvenanceLeaves::Empty
                        }
                    }
                }
                ResidualProvenanceSource::Boundary(RegionDataFlowSource::RegionOutput(origin)) => {
                    if !expanded_outputs.insert((origin.region_index, origin.output_index)) {
                        continue;
                    }
                    let region = instruction.regions()[origin.region_index];
                    let output = program.region(region)?.output_ids()[origin.output_index];
                    for leaf in self.resolve(ValueId::new(region, output))?.as_slice().iter().rev().copied() {
                        pending.push(match leaf {
                            ResidualProvenanceLeaf::Producer(producer) => ResidualProvenanceSource::Producer(producer),
                            ResidualProvenanceLeaf::Input(input_index) => {
                                ResidualProvenanceSource::Boundary(RegionDataFlowSource::RegionInput {
                                    region_index: origin.region_index,
                                    input_index,
                                })
                            }
                        });
                    }
                    ResidualProvenanceLeaves::Empty
                }
            };

            for leaf in replacements.as_slice().iter().copied() {
                if seen.insert(leaf) {
                    leaves.push(leaf);
                }
            }
        }

        Ok(leaves.into())
    }
}

/// Work item of the expansion of the sources of one instruction output in
/// [`ResidualProvenanceAnalysis::resolve_uncached`].
enum ResidualProvenanceSource {
    /// Source that the [`RegionDataFlow`](crate::RegionDataFlow) of the instruction declares, relative to the
    /// instruction and its attached regions.
    Boundary(RegionDataFlowSource),

    /// Instruction output that is itself a producer of the resolved value.
    Producer(ValueId),
}

/// Positions at which the atoms of one region are produced or enter the region, indexed by atom.
struct ResidualAtomPositions {
    /// Position of the producing instruction in the region and of the atom among the outputs of that instruction, for
    /// each atom that an instruction produces.
    producers: Vec<Option<(usize, usize)>>,

    /// Position of the atom among the inputs of the region, for each atom that is a region input.
    inputs: Vec<Option<usize>>,
}

/// Leaf of the symbolic provenance of a value, as resolved by [`ResidualProvenanceAnalysis`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
enum ResidualProvenanceLeaf {
    /// Output of an instruction that produces it itself, rather than forwarding an output of an attached region.
    Producer(ValueId),

    /// Input at the provided position of the region that contains the value, which each call site of that region
    /// resolves through its own inputs.
    Input(usize),
}

/// Immutable symbolic provenance, keeping the common empty and singleton summaries inline. Larger ordered
/// summaries share their immutable storage across cache hits instead of copying a vector of leaves each time.
#[derive(Clone, Debug, PartialEq, Eq)]
enum ResidualProvenanceLeaves {
    /// No operation or region input produces the value (e.g., a constant).
    Empty,

    /// One producer or symbolic region input.
    One(ResidualProvenanceLeaf),

    /// Several deduplicated producers or symbolic region inputs, in semantic order.
    Many(Arc<[ResidualProvenanceLeaf]>),
}

impl ResidualProvenanceLeaves {
    /// Returns the symbolic leaves in semantic order.
    fn as_slice(&self) -> &[ResidualProvenanceLeaf] {
        match self {
            Self::Empty => &[],
            Self::One(leaf) => std::slice::from_ref(leaf),
            Self::Many(leaves) => leaves.as_ref(),
        }
    }
}

impl From<Vec<ResidualProvenanceLeaf>> for ResidualProvenanceLeaves {
    fn from(mut leaves: Vec<ResidualProvenanceLeaf>) -> Self {
        match leaves.len() {
            0 => Self::Empty,
            1 => Self::One(leaves.pop().unwrap()),
            _ => Self::Many(leaves.into()),
        }
    }
}

/// Copies one instruction with its attached regions and provenance into `builder` over already mapped inputs.
/// `remapping` belongs to this source program and destination builder, preserving sharing between its regions.
fn copy_instruction<V: Value, O: Operation<Type = V::Type>>(
    source: &Program<V, O, Vec<V>, Vec<V>>,
    instruction: &Instruction<O>,
    inputs: Vec<AtomId>,
    builder: &mut ProgramBuilder<V, O>,
    remapping: &mut HashMap<RegionId, RegionId>,
) -> Result<Vec<AtomId>, ProgramError> {
    let regions = instruction
        .regions()
        .iter()
        .map(|region| Ok(builder.import_region_with_remapping(source.region_ref(*region)?, remapping)))
        .collect::<Result<Vec<_>, ProgramError>>()?;
    Ok(builder
        .add_instruction(instruction.operation().clone(), regions, inputs, Some(instruction.provenance().clone()))?
        .to_vec())
}

/// Returns the atom of `builder` that copies atom `atom` of `program`, using `atoms` to map the atoms of `program` that
/// were already copied and copying constants on first use.
fn copy_atom<V: Value, O: Operation<Type = V::Type>>(
    program: &Program<V, O, Vec<V>, Vec<V>>,
    atom: AtomId,
    atoms: &mut [Option<AtomId>],
    builder: &mut ProgramBuilder<V, O>,
) -> Result<AtomId, ProgramError> {
    if let Some(copy) = atoms[atom.index()] {
        return Ok(copy);
    }
    match &program.atoms()[atom.index()] {
        Atom::Constant(value) => {
            let copy = builder.add_constant(value.clone());
            atoms[atom.index()] = Some(copy);
            Ok(copy)
        }
        Atom::Variable(_) => Err(ProgramError::UnboundAtomId { id: atom }),
    }
}

/// Walks back from `atom` through the `store_count` unary store operations that produce it to the value that it stores,
/// or returns [`None`] when `atom` is not the result of such a chain. Residual placement stages each store operation
/// with that value, or with the result of the previous store operation, as its only input.
fn stored_source<V: Value, O: Operation<Type = V::Type>>(
    program: &Program<V, O, Vec<V>, Vec<V>>,
    instruction_by_output: &[Option<usize>],
    mut atom: AtomId,
    store_count: usize,
) -> Option<AtomId> {
    for _ in 0..store_count {
        let instruction = &program.instructions()[instruction_by_output[atom.index()]?];
        if instruction.inputs().len() != 1 || instruction.outputs().len() != 1 {
            return None;
        }
        atom = instruction.inputs()[0];
    }
    Some(atom)
}

/// Returns the operation that rounds the saved value of atom `atom` of `program` according to `rounding`, or [`None`]
/// when that value needs no rounding (refer to [`PartitionedProgram::with_rounded_residuals`]). It needs rounding when
/// it is a variable that the instructions of `program` consume more than `own_uses` times (i.e., apart from the store
/// operations that save it), that an operation other than its own rounding operation produces, and that is not an
/// input of `program`, unless `round_inputs` is set.
///
/// # Parameters
///
///   - `instruction_by_output`: Position of the instruction of `program` that produces each atom, if any.
///   - `use_counts`: Number of instruction inputs of `program` that each atom feeds (refer to
///     [`Program::atom_use_counts`]).
fn saved_value_rounding<V: Value, O: Operation<Type = V::Type> + OperationPayloadProjection>(
    program: &Program<V, O, Vec<V>, Vec<V>>,
    instruction_by_output: &[Option<usize>],
    use_counts: &[usize],
    atom: AtomId,
    own_uses: usize,
    rounding: &dyn ResidualRounding<V::Type>,
    round_inputs: bool,
) -> Option<ErasedOperation> {
    let Atom::Variable(_) = &program.atoms()[atom.index()] else {
        return None;
    };
    if use_counts[atom.index()] <= own_uses {
        return None;
    }
    let r#type = program.atoms()[atom.index()].r#type();
    let skip = match instruction_by_output[atom.index()] {
        Some(index) => rounding.is_rounding(&r#type, program.instructions()[index].operation()),
        None => !round_inputs,
    };
    if skip { None } else { rounding.rounding_operation(&r#type) }
}

/// Returns a copy of `program` in which the value of each atom of `rounded` is rounded by the operation that `rounded`
/// maps it to, right after its producer (or at the beginning for an input), and every later use of that value,
/// including the outputs of `program`, uses the rounded value instead. Also returns the atoms of the copy that hold the
/// values of `atoms`, which are the rounded values for the rounded atoms. Each operation is constructed in the
/// operation family `O` through [`OperationPayloadProjection::from_payload`].
fn round_atoms<V: Value, O: Operation<Type = V::Type> + OperationPayloadProjection>(
    program: &Program<V, O, Vec<V>, Vec<V>>,
    mut rounded: HashMap<AtomId, ErasedOperation>,
    atoms: &[AtomId],
) -> Result<(Program<V, O, Vec<V>, Vec<V>>, Vec<AtomId>), ProgramError> {
    let mut builder = ProgramBuilder::<V, O>::new();
    let mut copies = vec![None; program.atoms().len()];
    let mut round = |builder: &mut ProgramBuilder<V, O>, atom: AtomId, copy: AtomId| match rounded.remove(&atom) {
        Some(payload) => {
            let operation = O::from_payload(payload).map_err(|payload| ProgramError::UnsupportedOperation {
                message: format!(
                    "the operation family of the program cannot hold the residual rounding operation `{}`",
                    payload.type_name(),
                ),
            })?;
            Ok(builder.add_instruction(operation, Vec::new(), vec![copy], None)?[0])
        }
        None => Ok::<_, ProgramError>(copy),
    };
    for input in program.input_ids() {
        let copy = builder.add_input(program.atoms()[input.index()].r#type().into_owned());
        copies[input.index()] = Some(round(&mut builder, *input, copy)?);
    }

    let mut remapping = HashMap::new();
    for instruction in program.instructions() {
        let inputs = instruction
            .inputs()
            .iter()
            .map(|input| copy_atom(program, *input, &mut copies, &mut builder))
            .collect::<Result<Vec<_>, _>>()?;
        let regions = instruction
            .regions()
            .iter()
            .map(|region| Ok(builder.import_region_with_remapping(program.region_ref(*region)?, &mut remapping)))
            .collect::<Result<Vec<_>, ProgramError>>()?;
        let outputs = builder
            .add_instruction(instruction.operation().clone(), regions, inputs, Some(instruction.provenance().clone()))?
            .to_vec();
        for (source, output) in instruction.outputs().iter().zip(outputs) {
            copies[source.index()] = Some(round(&mut builder, *source, output)?);
        }
    }

    let outputs = program
        .output_ids()
        .iter()
        .map(|output| copy_atom(program, *output, &mut copies, &mut builder))
        .collect::<Result<Vec<_>, _>>()?;
    let atoms = atoms
        .iter()
        .map(|atom| copy_atom(program, *atom, &mut copies, &mut builder))
        .collect::<Result<Vec<_>, _>>()?;
    let input_count = program.input_ids().len();
    let output_count = outputs.len();
    let program =
        builder.build::<Vec<V>, Vec<V>>(outputs, vec![Placeholder; input_count], vec![Placeholder; output_count])?;
    Ok((program, atoms))
}

/// Finds a known output that already applies the `store_count` operations of `storage` to `source`. Storage-owned
/// recognition, pure unary applications, and the identical source atom establish representation equality without
/// treating a stored value as its logical source or relying on names, types or store counts as evidence of equality.
fn reusable_store<V: Value<Type: 'static>, O: Operation<Type = V::Type> + OperationPayloadProjection>(
    program: &Program<V, O, Vec<V>, Vec<V>>,
    source: AtomId,
    store_count: usize,
    storage: &dyn ResidualStorage<V::Type>,
    instruction_by_output: &[Option<usize>],
) -> Option<AtomId> {
    let residual_type = program.atoms()[source.index()].r#type();
    program.output_ids().iter().copied().find(|&output| {
        let mut atom = output;
        for step in (0..store_count).rev() {
            let Some(index) = instruction_by_output[atom.index()] else {
                return false;
            };
            let instruction = &program.instructions()[index];
            if instruction.inputs().len() != 1
                || instruction.outputs().len() != 1
                || !instruction.regions().is_empty()
                || !instruction.operation().effects().summary().is_pure()
                || !storage.matches_store_operation(&residual_type, step, instruction.operation())
            {
                return false;
            }
            atom = instruction.inputs()[0];
        }
        atom == source
    })
}

/// Stages the chain of storage operations that `payloads` describe on atom `atom` of `builder` and returns the atom
/// of its final result. Each payload is constructed in the operation family of `builder` and must be a pure unary
/// operation with a single result.
fn stage_storage<V: Value<Type: 'static>, O: Operation<Type = V::Type> + OperationPayloadProjection>(
    builder: &mut ProgramBuilder<V, O>,
    atom: AtomId,
    payloads: Vec<ErasedOperation>,
    storage: &dyn ResidualStorage<V::Type>,
) -> Result<AtomId, ResidualPolicyError> {
    payloads.into_iter().try_fold(atom, |atom, payload| {
        let operation = O::from_payload(payload).map_err(|payload| ResidualPolicyError::UnsupportedStorage {
            storage: storage.name(),
            residual_type: builder.atoms()[atom.index()].r#type().to_string(),
            message: format!("the operation family of the program cannot hold its payload `{}`", payload.type_name()),
        })?;
        let name = operation.name();
        if !operation.effects().is_pure() {
            return Err(ResidualPolicyError::InvalidStorage {
                storage: storage.name(),
                message: format!("its operation `{name}` is not pure"),
            });
        }
        match builder.add_instruction(operation, Vec::new(), vec![atom], None)? {
            [output] => Ok(*output),
            outputs => Err(ResidualPolicyError::InvalidStorage {
                storage: storage.name(),
                message: format!("its operation `{}` has {} results instead of one", name, outputs.len()),
            }),
        }
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::rc::Rc;
    use std::sync::atomic::AtomicUsize;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayType, DataType, DimensionBounds,
        DimensionType, Memory,
    };
    use crate::differentiation::{MemoryTransferStorage, NothingSavable};
    use crate::operations::{
        AddOperation, CompareOperation, ComparisonDirection, CompositeJvpRule, CompositeOperation, ConditionOperation,
        CosOperation, DimensionSizeOperation, DotDimensionNumbers, DotOperation, ExpOperation, LinearCallOperation,
        MulOperation, NegOperation, ReducePrecisionOperation, ReferenceFreezeOperation, ReferenceNewOperation,
        ReferenceReadOperation, ReferenceWriteOperation, RematerializeOperation, ReshapeOperation, ScanOperation,
        SinOperation, TagOperation, TransferToMemoryOperation, WhileOperation,
    };
    use crate::partial::values::PartialEvaluationOutput;
    use crate::programs::{
        Effects, InputRegionProvenance, OutputRegionProvenance, ReferenceType, RegionDataFlow, RegionInterface,
        RegionSlot, TypeError,
    };
    use crate::tests::TestRegionOperation;

    use super::*;

    type TestValue = ArrayIrValue<Array>;
    type TestOperation = ArrayIrOperation<Array>;
    type TestProgram = Program<TestValue, TestOperation, Vec<TestValue>, Vec<TestValue>>;
    type TestPartition = PartitionedProgram<TestValue, TestOperation>;

    // Test roundings may be plain functions from types to optional rounding operations. Such a rounding does not
    // recognize its own operations, so it rounds values that are already rounded again.
    impl<T: Type, F: Fn(&T) -> Option<ErasedOperation>> ResidualRounding<T> for F {
        fn rounding_operation(&self, r#type: &T) -> Option<ErasedOperation> {
            self(r#type)
        }

        fn is_rounding(&self, _type: &T, _operation: &dyn OperationPayloadProjection) -> bool {
            false
        }
    }

    /// [`ResidualStorage`] that stores and restores residuals by negating them.
    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    struct NegationStorage;

    impl<T: Type> ResidualStorage<T> for NegationStorage {
        fn name(&self) -> String {
            "negation".to_owned()
        }

        fn store_payloads(&self, _residual_type: &T) -> Result<Vec<ErasedOperation>, ResidualPolicyError> {
            Ok(vec![ErasedOperation::new(NegOperation::<ArrayType>::new())])
        }

        fn restore_payloads(
            &self,
            _stored_type: &T,
            _residual_type: &T,
        ) -> Result<Vec<ErasedOperation>, ResidualPolicyError> {
            Ok(vec![ErasedOperation::new(NegOperation::<ArrayType>::new())])
        }

        fn matches_store_operation(
            &self,
            _residual_type: &T,
            index: usize,
            operation: &dyn OperationPayloadProjection,
        ) -> bool {
            index == 0 && operation.project_payload(TypeId::of::<NegOperation<ArrayType>>()).is_some()
        }
    }

    /// Negation storage that uses the default conservative store recognition.
    #[derive(Debug)]
    struct UnrecognizedNegationStorage;

    impl<T: Type> ResidualStorage<T> for UnrecognizedNegationStorage {
        fn name(&self) -> String {
            "unrecognized_negation".to_owned()
        }

        fn store_payloads(&self, residual_type: &T) -> Result<Vec<ErasedOperation>, ResidualPolicyError> {
            NegationStorage.store_payloads(residual_type)
        }

        fn restore_payloads(
            &self,
            stored_type: &T,
            residual_type: &T,
        ) -> Result<Vec<ErasedOperation>, ResidualPolicyError> {
            NegationStorage.restore_payloads(stored_type, residual_type)
        }
    }

    /// [`ResidualRounding`] that rounds `bf16` arrays to the precision of `bf16` with a [`ReducePrecisionOperation`]
    /// and recognizes its own rounding operations.
    struct Bfloat16Rounding;

    impl ResidualRounding<ArrayIrType> for Bfloat16Rounding {
        fn rounding_operation(&self, r#type: &ArrayIrType) -> Option<ErasedOperation> {
            matches!(r#type, ArrayIrType::Array(array_type) if array_type.data_type() == DataType::BF16)
                .then(|| ErasedOperation::new(ReducePrecisionOperation::<ArrayType>::new(8, 7)))
        }

        fn is_rounding(&self, r#type: &ArrayIrType, operation: &dyn OperationPayloadProjection) -> bool {
            self.rounding_operation(r#type).is_some()
                && operation
                    .project_payload(TypeId::of::<ReducePrecisionOperation<ArrayType>>())
                    .and_then(|payload| payload.downcast_ref::<ReducePrecisionOperation<ArrayType>>())
                    .is_some_and(|operation| (operation.exponent_bits(), operation.mantissa_bits()) == (8, 7))
        }
    }

    /// [`ResidualPolicy`] that returns whatever `classify` returns for each candidate, in any type universe.
    struct TestPolicy<F> {
        /// Name of the policy.
        name: &'static str,

        /// Classifier of the policy.
        classify: F,
    }

    impl<T: 'static + Type, S: ResidualStorage<T>, F> ResidualPolicy<T> for TestPolicy<F>
    where
        F: 'static + Send + Sync + Fn(&ResidualCandidate<'_, T>) -> Result<ResidualDecision<S>, ResidualRejection>,
    {
        type Storage = S;

        fn name(&self) -> &str {
            self.name
        }

        fn classify(&self, candidate: &ResidualCandidate<'_, T>) -> Result<ResidualDecision<S>, ResidualRejection> {
            (self.classify)(candidate)
        }
    }

    /// Returns a [`ResidualPolicyReference`] to a [`TestPolicy`] over [`ArrayIrType`].
    fn policy<S: ResidualStorage<ArrayIrType>, F>(
        name: &'static str,
        classify: F,
    ) -> ResidualPolicyReference<ArrayIrType>
    where
        F: 'static
            + Send
            + Sync
            + Fn(&ResidualCandidate<'_, ArrayIrType>) -> Result<ResidualDecision<S>, ResidualRejection>,
    {
        ResidualPolicyReference::new(TestPolicy { name, classify })
    }

    /// Returns a policy that recomputes every residual.
    fn save_nothing() -> ResidualPolicyReference<ArrayIrType> {
        policy("save_nothing", |_| Ok(ResidualDecision::<NoStorage>::Recompute))
    }

    /// Returns a policy over [`ArrayType`] that recomputes every residual.
    fn save_nothing_in_arrays() -> ResidualPolicyReference<ArrayType> {
        ResidualPolicyReference::new(TestPolicy {
            name: "save_nothing",
            classify: |_: &ResidualCandidate<'_, ArrayType>| Ok(ResidualDecision::<NoStorage>::Recompute),
        })
    }

    /// Returns a policy that saves every residual.
    fn save_everything() -> ResidualPolicyReference<ArrayIrType> {
        policy("save_everything", |_| Ok(ResidualDecision::<NoStorage>::Save))
    }

    /// Returns a policy that saves the residuals produced by dot products, through `storage` when one is provided, and
    /// recomputes every other residual.
    fn save_dots(storage: Option<NegationStorage>) -> ResidualPolicyReference<ArrayIrType> {
        policy("save_dots", move |candidate| {
            Ok(match candidate.producers().iter().any(|producer| producer.payload::<DotOperation>().is_some()) {
                true => storage.map_or(ResidualDecision::Save, ResidualDecision::SaveWith),
                false => ResidualDecision::Recompute,
            })
        })
    }

    /// Returns a policy that saves the residuals tagged with `saved`, saves those tagged with `stored` through a
    /// [`NegationStorage`], and recomputes every other residual.
    fn save_names(
        saved: &'static [&'static str],
        stored: &'static [&'static str],
    ) -> ResidualPolicyReference<ArrayIrType> {
        policy("save_names", move |candidate| {
            let keys = candidate
                .producers()
                .iter()
                .filter_map(|producer| producer.payload::<TagOperation<ArrayType>>().map(TagOperation::key))
                .collect::<Vec<_>>();
            Ok(if keys.iter().any(|key| saved.contains(key)) {
                ResidualDecision::Save
            } else if keys.iter().any(|key| stored.contains(key)) {
                ResidualDecision::SaveWith(NegationStorage)
            } else {
                ResidualDecision::Recompute
            })
        })
    }

    /// Returns the [`ArrayIrType`] of `f64` vectors of size 3.
    fn vector_type() -> ArrayIrType {
        ArrayType::new_static(DataType::F64, [3]).into()
    }

    /// Returns the [`ArrayIrType`] of `f64` scalars.
    fn scalar_type() -> ArrayIrType {
        ArrayType::scalar(DataType::F64).into()
    }

    /// Returns the [`ArrayIrType`] of dimensions named `n`, which does not project into [`ArrayType`].
    fn dimension_type() -> ArrayIrType {
        ArrayIrType::Dimension(DimensionType::new("n", DimensionBounds::non_negative(None).unwrap()))
    }

    /// Returns a dot product of two vectors.
    fn dot() -> ArrayOperation<Array> {
        DotOperation::new(DotDimensionNumbers::new(vec![0], vec![0], vec![], vec![])).into()
    }

    /// Returns an operation that produces the size of the leading dimension of an `f64` vector of size 3 as a
    /// dimension of type [`dimension_type`].
    fn dimension_size() -> TestOperation {
        TestOperation::DimensionSize(
            DimensionSizeOperation::new(&ArrayType::new_static(DataType::F64, [3]), 0).unwrap(),
        )
    }

    /// Returns a candidate of type `residual_type` that output 0 of `operation` produces from inputs of `input_types`.
    fn candidate(
        operation: &TestOperation,
        input_types: Vec<ArrayIrType>,
        residual_type: ArrayIrType,
    ) -> ResidualCandidate<'_, ArrayIrType> {
        ResidualCandidate::new(
            vec![ResidualProducer::new(operation, 0, input_types, vec![residual_type.clone()])],
            residual_type,
        )
    }

    /// Adds the array operation `operation` to `builder` and returns its first output.
    fn add(
        builder: &mut ProgramBuilder<TestValue, TestOperation>,
        operation: ArrayOperation<Array>,
        inputs: Vec<AtomId>,
    ) -> AtomId {
        builder.add_instruction(operation, Vec::new(), inputs, None).unwrap()[0]
    }

    /// Builds `outputs` of `builder` into a program.
    fn build(builder: ProgramBuilder<TestValue, TestOperation>, outputs: Vec<AtomId>) -> TestProgram {
        let input_count = builder.input_ids().len();
        let output_count = outputs.len();
        builder.build(outputs, vec![Placeholder; input_count], vec![Placeholder; output_count]).unwrap()
    }

    /// Runs the known program of `partition` and then its residual program on `inputs`, returning the original
    /// outputs.
    fn run(partition: &TestPartition, inputs: &[TestValue]) -> Vec<TestValue> {
        let known_inputs = partition.known_input_indices().iter().map(|index| inputs[*index].clone()).collect();
        let known_outputs = partition.known_program().interpret(known_inputs).unwrap();
        let known_output_count = partition.outputs().iter().filter(|output| output.is_known()).count();
        let residual_inputs = partition
            .residual_inputs()
            .iter()
            .map(|source| match source {
                ResidualInputSource::UnknownInput(index) | ResidualInputSource::KnownInput(index) => {
                    inputs[*index].clone()
                }
                ResidualInputSource::KnownOutput(index) => known_outputs[*index].clone(),
                ResidualInputSource::ResidualEdge(edge) => known_outputs[known_output_count + edge].clone(),
            })
            .collect();
        let residual_outputs = partition.residual_program().interpret(residual_inputs).unwrap();
        partition
            .outputs()
            .iter()
            .map(|output| match output {
                PartialEvaluationOutput::Known(index) => known_outputs[*index].clone(),
                PartialEvaluationOutput::Unknown(index) => residual_outputs[*index].clone(),
            })
            .collect()
    }

    /// Builds `f(x, t) = (sin(dot(x, x)), cos(dot(x, x)) * t)`, whose partition with `x` known and `t` unknown has the
    /// cosine as its only edge:
    ///
    /// ```text
    /// lambda %0:f64[3], %1:f64[] .
    /// let %2:f64[] = dot [
    ///     dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
    /// ] %0 %0
    ///     %3:f64[] = sin %2
    ///     %4:f64[] = cos %2
    ///     %5:f64[] = mul %4 %1
    /// in (%3, %5)
    /// ```
    fn sin_dot_program() -> TestProgram {
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let x = builder.add_input(vector_type());
        let t = builder.add_input(scalar_type());
        let product = add(&mut builder, dot(), vec![x, x]);
        let sine = add(&mut builder, SinOperation::<ArrayType>::new().into(), vec![product]);
        let cosine = add(&mut builder, CosOperation::<ArrayType>::new().into(), vec![product]);
        let tangent = add(&mut builder, MulOperation::<ArrayType>::new().into(), vec![cosine, t]);
        build(builder, vec![sine, tangent])
    }

    /// Returns the inputs of [`sin_dot_program`] used to check that partitions with placed residuals compute the same
    /// outputs.
    fn sin_dot_inputs() -> Vec<TestValue> {
        vec![
            ArrayIrValue::Array(Array::vector(vec![1.0f64, 2.0, 3.0]).unwrap()),
            ArrayIrValue::Array(Array::scalar(2.0f64).unwrap()),
        ]
    }

    /// Builds `f(p, x, t)`, whose condition on `p` produces `tag[first](sin(x))` and `tag[second](cos(x))` in both
    /// branches, and whose outputs multiply those two values by `t`. When `first_consumed_first` is `false`, the
    /// residual program consumes the second value first, which reverses the order of the edges of the partition.
    fn condition_program(first_consumed_first: bool) -> TestProgram {
        let mut branch = ProgramBuilder::<TestValue, TestOperation>::new();
        let a = branch.add_input(scalar_type());
        let sine = add(&mut branch, SinOperation::<ArrayType>::new().into(), vec![a]);
        let first = add(&mut branch, TagOperation::<ArrayType>::new("first").into(), vec![sine]);
        let cosine = add(&mut branch, CosOperation::<ArrayType>::new().into(), vec![a]);
        let second = add(&mut branch, TagOperation::<ArrayType>::new("second").into(), vec![cosine]);
        let branch = build(branch, vec![first, second]);

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let p = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let x = builder.add_input(scalar_type());
        let t = builder.add_input(scalar_type());
        let true_branch = builder.import_program(branch.clone());
        let false_branch = builder.import_program(branch);
        let outputs = builder
            .add_instruction(ConditionOperation::new(), vec![true_branch, false_branch], vec![p, x], None)
            .unwrap()
            .to_vec();
        let consumed = if first_consumed_first { [outputs[0], outputs[1]] } else { [outputs[1], outputs[0]] };
        let first = add(&mut builder, MulOperation::<ArrayType>::new().into(), vec![consumed[0], t]);
        let second = add(&mut builder, MulOperation::<ArrayType>::new().into(), vec![consumed[1], t]);
        build(builder, vec![first, second])
    }

    /// Builds a partition whose known program attaches one shared body to a condition and a scan. The ordinary call
    /// forwards an input; the scan feeds it through two carries and a sine of a dot. An unused stacked result is
    /// tagged separately. Residual outputs multiply the selected ordinary result, carry history, or final first
    /// carry by an unknown scalar. Building both halves explicitly preserves the shared region identity under test.
    fn feedback_partition(length: usize, outputs: &[usize]) -> TestPartition {
        let mut body = ProgramBuilder::<TestValue, TestOperation>::new();
        body.add_input(ArrayType::scalar(DataType::I64).into());
        let first = body.add_input(scalar_type());
        let second = body.add_input(scalar_type());
        let dot = add(
            &mut body,
            DotOperation::new(DotDimensionNumbers::new(vec![], vec![], vec![], vec![])).into(),
            vec![first, first],
        );
        let next = add(&mut body, SinOperation::<ArrayType>::new().into(), vec![dot]);
        let unused = add(&mut body, TagOperation::<ArrayType>::new("unused").into(), vec![first]);
        let body = build(body, vec![second, next, first, unused]);
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let input = builder.add_input(scalar_type());
        let initial = add(&mut builder, TagOperation::<ArrayType>::new("initial").into(), vec![input]);
        let index = builder.add_constant(ArrayIrValue::Array(Array::scalar(0i64).unwrap()));
        let body = builder.import_program(body);
        let ordinary = builder
            .add_instruction(ConditionOperation::new(), vec![body, body], vec![predicate, index, initial, input], None)
            .unwrap()[2];
        let scan = builder
            .add_instruction(ScanOperation::<ArrayIrType>::new(2, length), vec![body], vec![initial, input], None)
            .unwrap()
            .to_vec();
        let known = build(builder, vec![ordinary, scan[2], scan[0]]);
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let unknown = builder.add_input(scalar_type());
        let edges = [
            builder.add_input(scalar_type()),
            builder.add_input(ArrayType::new_static(DataType::F64, [length]).into()),
            builder.add_input(scalar_type()),
        ];
        let values = outputs
            .iter()
            .map(|&output| add(&mut builder, MulOperation::<ArrayType>::new().into(), vec![edges[output], unknown]))
            .collect();
        PartitionedProgram::from_parts(
            known,
            build(builder, values),
            3,
            vec![0, 1],
            vec![
                ResidualInputSource::UnknownInput(2),
                ResidualInputSource::ResidualEdge(0),
                ResidualInputSource::ResidualEdge(1),
                ResidualInputSource::ResidualEdge(2),
            ],
            (0..outputs.len()).map(PartialEvaluationOutput::Unknown).collect(),
        )
        .unwrap()
    }

    /// Builds a known condition whose branches use a scan carry, optionally invoking the same scan body separately.
    /// A zero-trip scan cannot demand the body's dot, while an ordinary invocation of the shared body still can.
    fn nested_scan_program(length: usize, invoke_shared_body: bool) -> TestProgram {
        let mut body = ProgramBuilder::<TestValue, TestOperation>::new();
        body.add_input(ArrayType::scalar(DataType::I64).into());
        let input = body.add_input(scalar_type());
        let output = add(
            &mut body,
            DotOperation::new(DotDimensionNumbers::new(vec![], vec![], vec![], vec![])).into(),
            vec![input, input],
        );
        let body = build(body, vec![output]);
        let mut branch = ProgramBuilder::<TestValue, TestOperation>::new();
        let input = branch.add_input(scalar_type());
        let body = branch.import_program(body);
        let mut output = branch
            .add_instruction(ScanOperation::<ArrayIrType>::new(1, length), vec![body], vec![input], None)
            .unwrap()[0];
        if invoke_shared_body {
            let predicate = branch.add_constant(ArrayIrValue::Array(Array::scalar(true).unwrap()));
            let index = branch.add_constant(ArrayIrValue::Array(Array::scalar(0i64).unwrap()));
            let called = branch
                .add_instruction(ConditionOperation::new(), vec![body, body], vec![predicate, index, input], None)
                .unwrap()[0];
            output = add(&mut branch, AddOperation::<ArrayType>::new().into(), vec![output, called]);
        }
        let output = add(&mut branch, SinOperation::<ArrayType>::new().into(), vec![output]);
        let branch = build(branch, vec![output]);
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let input = builder.add_input(scalar_type());
        let unknown = builder.add_input(scalar_type());
        let branch = builder.import_program(branch);
        let output = builder
            .add_instruction(ConditionOperation::new(), vec![branch, branch], vec![predicate, input], None)
            .unwrap()[0];
        let output = add(&mut builder, MulOperation::<ArrayType>::new().into(), vec![output, unknown]);
        build(builder, vec![output])
    }

    /// Builds a known condition whose branches apply the residual-parameterized linear map
    /// `(r, u) ↦ (sin(r) · u, tag(cos(r) · u, "unused"))` with `r = u` and return its first output. A `linear_call`
    /// cannot prune its boundary, so its forward region keeps the unused second output. Its transpose maps the two
    /// output cotangents to `tag(sin(r) · first + cos(r) · second, "rule")`; that region remains dormant during forward
    /// execution. The condition's result is multiplied by an unknown scalar.
    fn linear_call_condition_program() -> TestProgram {
        let mut forward = ProgramBuilder::<TestValue, TestOperation>::new();
        let residual = forward.add_input(scalar_type());
        let linear = forward.add_input(scalar_type());
        let sine = add(&mut forward, SinOperation::<ArrayType>::new().into(), vec![residual]);
        let cosine = add(&mut forward, CosOperation::<ArrayType>::new().into(), vec![residual]);
        let first = add(&mut forward, MulOperation::<ArrayType>::new().into(), vec![sine, linear]);
        let second = add(&mut forward, MulOperation::<ArrayType>::new().into(), vec![cosine, linear]);
        let unused = add(&mut forward, TagOperation::<ArrayType>::new("unused").into(), vec![second]);
        let forward = build(forward, vec![first, unused]);
        let mut transpose = ProgramBuilder::<TestValue, TestOperation>::new();
        let residual = transpose.add_input(scalar_type());
        let first = transpose.add_input(scalar_type());
        let second = transpose.add_input(scalar_type());
        let sine = add(&mut transpose, SinOperation::<ArrayType>::new().into(), vec![residual]);
        let cosine = add(&mut transpose, CosOperation::<ArrayType>::new().into(), vec![residual]);
        let first = add(&mut transpose, MulOperation::<ArrayType>::new().into(), vec![sine, first]);
        let second = add(&mut transpose, MulOperation::<ArrayType>::new().into(), vec![cosine, second]);
        let cotangent = add(&mut transpose, AddOperation::<ArrayType>::new().into(), vec![first, second]);
        let rule = add(&mut transpose, TagOperation::<ArrayType>::new("rule").into(), vec![cotangent]);
        let transpose = build(transpose, vec![rule]);
        let mut branch = ProgramBuilder::<TestValue, TestOperation>::new();
        let input = branch.add_input(scalar_type());
        let forward = branch.import_program(forward);
        let transpose = branch.import_program(transpose);
        let output = branch
            .add_instruction(
                ArrayOperation::<Array>::LinearCall(LinearCallOperation::new(1)),
                vec![forward, transpose],
                vec![input, input],
                None,
            )
            .unwrap()[0];
        let branch = build(branch, vec![output]);
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let input = builder.add_input(scalar_type());
        let unknown = builder.add_input(scalar_type());
        let branch = builder.import_program(branch);
        let output = builder
            .add_instruction(ConditionOperation::new(), vec![branch, branch], vec![predicate, input], None)
            .unwrap()[0];
        let output = add(&mut builder, MulOperation::<ArrayType>::new().into(), vec![output, unknown]);
        build(builder, vec![output])
    }

    /// Checks that replay of [`linear_call_condition_program`] preserves both outputs and the transpose rule of each
    /// linear call while saving only the condition's predicate and input.
    fn check_linear_call_condition_partition(partition: &TestPartition) {
        assert_eq!(
            partition.to_string(),
            indoc! {"
                partition [
                    known_inputs=[0, 1],
                    residual_inputs=[UnknownInput(2), ResidualEdge(0), ResidualEdge(1)],
                    outputs=[Unknown(0)],
                ]
                known={
                    lambda %0:bool[], %1:f64[] .
                    in (%0, %1)
                }
                residual={
                    lambda %0:f64[], %1:bool[], %2:f64[] .
                    let %3:f64[] = condition %1 %2 [
                        true={
                            lambda %0:f64[] .
                            let %1:f64[], %2:f64[] = linear_call [residual_count=1] %0 %0 [
                                forward={
                                    lambda %0:f64[], %1:f64[] .
                                    let %2:f64[] = sin %0
                                        %3:f64[] = mul %2 %1
                                        %4:f64[] = cos %0
                                        %5:f64[] = mul %4 %1
                                        %6:f64[] = tag [key=unused] %5
                                    in (%3, %6)
                                },
                                transpose={
                                    lambda %0:f64[], %1:f64[], %2:f64[] .
                                    let %3:f64[] = sin %0
                                        %4:f64[] = mul %3 %1
                                        %5:f64[] = cos %0
                                        %6:f64[] = mul %5 %2
                                        %7:f64[] = add %4 %6
                                        %8:f64[] = tag [key=rule] %7
                                    in (%8)
                                },
                            ]
                            in (%1)
                        },
                        false={
                            lambda %0:f64[] .
                            let %1:f64[], %2:f64[] = linear_call [residual_count=1] %0 %0 [
                                forward={
                                    lambda %0:f64[], %1:f64[] .
                                    let %2:f64[] = sin %0
                                        %3:f64[] = mul %2 %1
                                        %4:f64[] = cos %0
                                        %5:f64[] = mul %4 %1
                                        %6:f64[] = tag [key=unused] %5
                                    in (%3, %6)
                                },
                                transpose={
                                    lambda %0:f64[], %1:f64[], %2:f64[] .
                                    let %3:f64[] = sin %0
                                        %4:f64[] = mul %3 %1
                                        %5:f64[] = cos %0
                                        %6:f64[] = mul %5 %2
                                        %7:f64[] = add %4 %6
                                        %8:f64[] = tag [key=rule] %7
                                    in (%8)
                                },
                            ]
                            in (%1)
                        },
                    ]
                        %4:f64[] = mul %3 %0
                    in (%4)
                }"},
        );
    }

    /// Returns a policy named `name` that rejects the residuals tagged with `key` and recomputes every other residual.
    fn reject_name(name: &'static str, key: &'static str) -> ResidualPolicyReference<ArrayIrType> {
        policy(name, move |candidate| {
            if candidate
                .producers()
                .iter()
                .any(|producer| producer.payload::<TagOperation<ArrayType>>().is_some_and(|tag| tag.key() == key))
            {
                Err(ResidualRejection::new(format!("the `{key}` result cannot cross this boundary")))
            } else {
                Ok(ResidualDecision::<NoStorage>::Recompute)
            }
        })
    }

    #[test]
    fn test_residual_policy_error() {
        let rejection = ResidualRejection::new("never save `x`");
        let rejected = ResidualPolicyError::Rejected { policy: "names".to_owned(), rejection };
        assert_eq!(rejected.to_string(), "residual policy `names` rejected a residual: never save `x`");
        assert_eq!(
            ResidualPolicyError::UnsupportedProjection {
                policy: "names".to_owned(),
                position: "the residual".to_owned(),
                residual_type: "n".to_owned(),
            }
            .to_string(),
            "residual policy `names` cannot classify the residual of type `n`, which does not project into the type \
             universe of the policy; register a native instantiation or a projection fallback for this universe",
        );
        assert_eq!(
            ResidualPolicyError::UnsupportedStorage {
                storage: "negation".to_owned(),
                residual_type: "f64[]".to_owned(),
                message: "no payload".to_owned(),
            }
            .to_string(),
            "residual storage `negation` cannot stage a residual of type `f64[]`: no payload",
        );
        assert_eq!(
            ResidualPolicyError::InvalidStorage {
                storage: "negation".to_owned(),
                message: "its operation `neg` is not pure".to_owned(),
            }
            .to_string(),
            "residual storage `negation` is invalid: its operation `neg` is not pure",
        );

        // Residual-policy errors round trip through program errors, while program errors round trip through their
        // dedicated variant, which renders like the program error that it wraps.
        assert_eq!(ResidualPolicyError::from(ProgramError::from(rejected.clone())), rejected);
        let program_error = ProgramError::MalformedProgram("broken".to_owned());
        assert_eq!(ResidualPolicyError::Program(program_error.clone()).to_string(), program_error.to_string());
        assert_eq!(
            ResidualPolicyError::from(program_error.clone()),
            ResidualPolicyError::Program(program_error.clone()),
        );
        assert_eq!(ProgramError::from(ResidualPolicyError::Program(program_error.clone())), program_error);
    }

    #[test]
    fn test_residual_producer() {
        let operation = TestOperation::from(dot());
        let producer = ResidualProducer::new(&operation, 0, vec![vector_type(), vector_type()], vec![scalar_type()]);
        assert_eq!(producer.name(), "dot");
        assert_eq!(producer.output_index(), 0);
        assert_eq!(producer.input_types(), &[vector_type(), vector_type()]);
        assert_eq!(producer.output_types(), &[scalar_type()]);
    }

    #[test]
    fn test_residual_producer_payload() {
        // Payloads are recognized both through the projected array member of the operation family and among the
        // native operations of the family, while payloads of other operations are not.
        let dot = TestOperation::from(dot());
        let producer = ResidualProducer::new(&dot, 0, vec![vector_type(), vector_type()], vec![scalar_type()]);
        assert!(producer.payload::<DotOperation>().is_some());
        assert!(producer.payload::<SinOperation<ArrayType>>().is_none());
        let dimension_size = dimension_size();
        let producer = ResidualProducer::new(&dimension_size, 0, vec![vector_type()], vec![dimension_type()]);
        assert!(producer.payload::<DimensionSizeOperation>().is_some());
        assert!(producer.payload::<DotOperation>().is_none());
    }

    #[test]
    fn test_residual_candidate() {
        let operation = TestOperation::from(dot());
        let producer = ResidualProducer::new(&operation, 0, vec![vector_type(), vector_type()], vec![scalar_type()]);
        let candidate = ResidualCandidate::new(vec![producer], scalar_type());
        assert_eq!(candidate.producers().len(), 1);
        assert_eq!(candidate.producers()[0].name(), "dot");
        assert_eq!(candidate.r#type().as_ref(), &scalar_type());
    }

    #[test]
    fn test_residual_decision_into_erased() {
        assert!(matches!(ResidualDecision::<NoStorage>::Save.into_erased::<ArrayType>(), ResidualDecision::Save));
        assert!(matches!(
            ResidualDecision::<NoStorage>::Recompute.into_erased::<ArrayType>(),
            ResidualDecision::Recompute,
        ));
        let ResidualDecision::SaveWith(storage) =
            ResidualDecision::SaveWith(NegationStorage).into_erased::<ArrayType>()
        else {
            panic!("expected a stored residual");
        };
        assert_eq!(storage.name(), "negation");
    }

    #[test]
    fn test_residual_rejection() {
        let rejection = ResidualRejection::new("never save `x`");
        assert_eq!(rejection.message(), "never save `x`");
    }

    #[test]
    fn test_residual_storage_matches_store_operation() {
        let r#type = scalar_type();
        let direct = ArrayOperation::<Array>::from(NegOperation::<ArrayType>::new());
        let projected = TestOperation::from(direct.clone());
        let sine = TestOperation::from(ArrayOperation::<Array>::from(SinOperation::<ArrayType>::new()));

        assert!(!UnrecognizedNegationStorage.matches_store_operation(&r#type, 0, &direct));
        assert!(!UnrecognizedNegationStorage.matches_store_operation(&r#type, 0, &projected));
        assert!(NegationStorage.matches_store_operation(&r#type, 0, &direct));
        assert!(NegationStorage.matches_store_operation(&r#type, 0, &projected));
        assert!(!NegationStorage.matches_store_operation(&r#type, 1, &projected));
        assert!(!NegationStorage.matches_store_operation(&r#type, 0, &sine));

        let erased: ErasedResidualStorage<ArrayIrType> = Arc::new(NegationStorage);
        assert!(erased.matches_store_operation(&r#type, 0, &projected));
        assert!(!erased.matches_store_operation(&r#type, 1, &projected));
        let erased: ErasedResidualStorage<ArrayIrType> = Arc::new(UnrecognizedNegationStorage);
        assert!(!erased.matches_store_operation(&r#type, 0, &projected));
    }

    #[test]
    fn test_native_residual_policies_with() {
        // A policy over `ArrayType` that declares an `ArrayIrType` instantiation, which it registers twice so that the
        // second registration replaces the first.
        struct SavingPolicy;

        impl ResidualPolicy<ArrayType> for SavingPolicy {
            type Storage = NoStorage;

            fn name(&self) -> &str {
                "saving"
            }

            fn classify(
                &self,
                _candidate: &ResidualCandidate<'_, ArrayType>,
            ) -> Result<ResidualDecision<NoStorage>, ResidualRejection> {
                Ok(ResidualDecision::Save)
            }

            fn native_instantiations(&self) -> NativeResidualPolicies {
                NativeResidualPolicies::default()
                    .with(TestPolicy {
                        name: "replaced",
                        classify: |_: &ResidualCandidate<'_, ArrayIrType>| Ok(ResidualDecision::<NoStorage>::Recompute),
                    })
                    .with(TestPolicy {
                        name: "saving",
                        classify: |_: &ResidualCandidate<'_, ArrayIrType>| Ok(ResidualDecision::<NoStorage>::Save),
                    })
            }
        }

        // Lifting the policy into `ArrayIrType` uses the registered instantiation, which also classifies the candidates
        // that do not project into `ArrayType` (e.g., dimensions).
        let lifted = ResidualPolicyReference::new(SavingPolicy).lift::<ArrayIrType>();
        assert_eq!(lifted.name(), "saving");
        let dimension_size = dimension_size();
        assert!(matches!(
            lifted.classify(&candidate(&dimension_size, vec![vector_type()], dimension_type())),
            Ok(ResidualDecision::Save),
        ));
    }

    #[test]
    fn test_native_residual_policies_save_from_both() {
        /// Returns one native policy over the array IR universe with a fixed decision.
        fn native(decision: ResidualDecision<NoStorage>) -> NativeResidualPolicies {
            NativeResidualPolicies::default().with(TestPolicy {
                name: "native",
                classify: move |_: &ResidualCandidate<'_, ArrayIrType>| Ok(decision.clone()),
            })
        }
        let recompute = || native(ResidualDecision::Recompute);
        let save = || native(ResidualDecision::Save);
        let operation = dimension_size();
        let candidate = candidate(&operation, vec![vector_type()], dimension_type());
        let combined = recompute().save_from_both(save(), "combined");
        assert!(combined.get::<ArrayType>().is_none());
        assert!(matches!(combined.get::<ArrayIrType>().unwrap().classify(&candidate), Ok(ResidualDecision::Save)));
        assert!(
            recompute()
                .save_from_both(NativeResidualPolicies::default(), "missing_second")
                .get::<ArrayIrType>()
                .is_none()
        );
        assert!(
            NativeResidualPolicies::default()
                .save_from_both(save(), "missing_first")
                .get::<ArrayIrType>()
                .is_none()
        );

        // Derived registries can compose again, and a subsequent explicit native entry overrides the derived one.
        let nested = recompute().save_from_both(combined, "nested");
        assert_eq!(nested.get::<ArrayIrType>().unwrap().name(), "nested");
        assert!(matches!(nested.get::<ArrayIrType>().unwrap().classify(&candidate), Ok(ResidualDecision::Save)));
        let overridden = nested.with(TestPolicy {
            name: "override",
            classify: |_: &ResidualCandidate<'_, ArrayIrType>| Ok(ResidualDecision::<NoStorage>::Recompute),
        });
        assert_eq!(overridden.get::<ArrayIrType>().unwrap().name(), "override");
        assert!(matches!(
            overridden.get::<ArrayIrType>().unwrap().classify(&candidate),
            Ok(ResidualDecision::Recompute)
        ));
    }

    #[test]
    fn test_native_residual_policies_save_from_both_preserves_storage_and_rejections() {
        let operation = dimension_size();
        let candidate = candidate(&operation, vec![vector_type()], dimension_type());
        let store = || {
            NativeResidualPolicies::default().with(TestPolicy {
                name: "store",
                classify: |_: &ResidualCandidate<'_, ArrayIrType>| Ok(ResidualDecision::SaveWith(NegationStorage)),
            })
        };
        let reject = || {
            NativeResidualPolicies::default().with(TestPolicy {
                name: "child_rejection",
                classify: |_: &ResidualCandidate<'_, ArrayIrType>| -> Result<ResidualDecision<NoStorage>, _> {
                    Err(ResidualRejection::new("the candidate is forbidden"))
                },
            })
        };
        let recompute = || {
            NativeResidualPolicies::default().with(TestPolicy {
                name: "recompute",
                classify: |_: &ResidualCandidate<'_, ArrayIrType>| Ok(ResidualDecision::<NoStorage>::Recompute),
            })
        };
        // The second policy is not called when the first saves, and second-policy storage survives when it is used.
        for combined in [store().save_from_both(reject(), "combined"), recompute().save_from_both(store(), "combined")]
        {
            let ResidualDecision::SaveWith(storage) =
                combined.get::<ArrayIrType>().unwrap().classify(&candidate).unwrap()
            else {
                panic!("expected the selected storage");
            };
            assert_eq!(storage.name(), "negation");
        }
        for combined in [reject().save_from_both(store(), "combined"), recompute().save_from_both(reject(), "combined")]
        {
            assert_eq!(
                combined.get::<ArrayIrType>().unwrap().classify(&candidate).map(|_| ()),
                Err(ResidualPolicyError::Rejected {
                    policy: "combined".to_owned(),
                    rejection: ResidualRejection::new("the candidate is forbidden"),
                }),
            );
        }
    }

    #[test]
    fn test_projection_fallback_candidate() {
        // A candidate whose second producer does not project into `ArrayType` reaches the projection fallback, which
        // observes the complete candidate together with which of its types project.
        let dot = TestOperation::from(dot());
        let dimension_size = dimension_size();
        let candidate = ResidualCandidate::new(
            vec![
                ResidualProducer::new(&dot, 0, vec![vector_type(), vector_type()], vec![scalar_type()]),
                ResidualProducer::new(&dimension_size, 0, vec![vector_type()], vec![dimension_type()]),
            ],
            scalar_type(),
        );
        let lifted = save_nothing_in_arrays()
            .with_projection_fallback::<ArrayIrType, _>(
                |candidate: &ProjectionFallbackCandidate<'_, '_, ArrayIrType>| {
                    let producers = candidate.candidate().producers();
                    assert_eq!(
                        producers.iter().map(ResidualProducer::name).collect::<Vec<_>>(),
                        ["dot", "dimension_size"],
                    );
                    assert_eq!(candidate.producer_projectable(), &[true, false]);
                    assert!(candidate.residual_projectable());
                    Ok(ResidualDecision::Save)
                },
            )
            .lift::<ArrayIrType>();
        assert!(matches!(lifted.classify(&candidate), Ok(ResidualDecision::Save)));
    }

    #[test]
    fn test_residual_policy_reference() {
        let dots = save_dots(None);
        assert_eq!(dots.name(), "save_dots");
        assert_eq!(
            format!("{dots:?}"),
            format!("ResidualPolicyReference {{ name: \"save_dots\", id: {} }}", dots.id()),
        );

        // References compare and hash by definition: clones are the same definition, while separately registered
        // policies are distinct definitions even when they behave identically.
        let clone = dots.clone();
        let other = save_dots(None);
        assert_eq!(clone.id(), dots.id());
        assert_eq!(clone, dots);
        assert_ne!(other.id(), dots.id());
        assert_ne!(other, dots);
        let definitions = HashSet::from([dots, other]);
        assert_eq!(definitions.len(), 2);
        assert!(definitions.contains(&clone));
        assert!(!definitions.contains(&save_dots(None)));
    }

    #[test]
    fn test_residual_policy_reference_with_native_instantiation() {
        // Lifting a policy over `ArrayType` into `ArrayIrType` cannot classify dimensions by projection. Registering a
        // native `ArrayIrType` instantiation creates a new definition, whose lift uses that instantiation for every
        // candidate.
        let source = save_nothing_in_arrays();
        let dot = TestOperation::from(dot());
        let dimension_size = dimension_size();
        let dot_candidate = candidate(&dot, vec![vector_type(), vector_type()], scalar_type());
        let dimension_candidate = candidate(&dimension_size, vec![vector_type()], dimension_type());
        assert!(matches!(
            source.lift::<ArrayIrType>().classify(&dimension_candidate),
            Err(ResidualPolicyError::UnsupportedProjection { .. }),
        ));
        let native = source.clone().with_native_instantiation::<ArrayIrType, _>(TestPolicy {
            name: "native_recompute",
            classify: |candidate: &ResidualCandidate<'_, ArrayIrType>| {
                Ok(match candidate.r#type().as_ref() {
                    ArrayIrType::Dimension(_) => ResidualDecision::<NoStorage>::Save,
                    _ => ResidualDecision::Recompute,
                })
            },
        });
        assert_ne!(native.id(), source.id());
        let lifted = native.lift::<ArrayIrType>();
        assert_eq!(lifted.id(), native.id());
        assert_eq!(lifted.name(), "native_recompute");
        assert!(matches!(lifted.classify(&dimension_candidate), Ok(ResidualDecision::Save)));
        assert!(matches!(lifted.classify(&dot_candidate), Ok(ResidualDecision::Recompute)));
    }

    #[test]
    fn test_residual_policy_reference_with_projection_fallback() {
        // Registering a projection fallback creates a new definition, whose lift classifies the candidates that do not
        // project with the fallback while projectable candidates still reach the source policy. A later registration
        // for the same universe replaces an earlier one.
        let source = save_nothing_in_arrays();
        let fallback = source
            .clone()
            .with_projection_fallback::<ArrayIrType, _>(|_| Ok(ResidualDecision::Recompute))
            .with_projection_fallback::<ArrayIrType, _>(|_| Ok(ResidualDecision::Save));
        assert_ne!(fallback.id(), source.id());
        let lifted = fallback.lift::<ArrayIrType>();
        assert_eq!(lifted.id(), fallback.id());
        let dot = TestOperation::from(dot());
        let dimension_size = dimension_size();
        assert!(matches!(
            lifted.classify(&candidate(&dimension_size, vec![vector_type()], dimension_type())),
            Ok(ResidualDecision::Save),
        ));
        assert!(matches!(
            lifted.classify(&candidate(&dot, vec![vector_type(), vector_type()], scalar_type())),
            Ok(ResidualDecision::Recompute),
        ));
    }

    #[test]
    fn test_residual_policy_reference_classify() {
        let operation = TestOperation::from(dot());
        let candidate = candidate(&operation, vec![vector_type(), vector_type()], scalar_type());
        assert!(matches!(save_dots(None).classify(&candidate), Ok(ResidualDecision::Save)));
        assert!(matches!(save_nothing().classify(&candidate), Ok(ResidualDecision::Recompute)));
        let Ok(ResidualDecision::SaveWith(storage)) = save_dots(Some(NegationStorage)).classify(&candidate) else {
            panic!("expected a stored residual");
        };
        assert_eq!(storage.name(), "negation");

        // Rejections are reported together with the name of the rejecting policy.
        let rejecting = policy("rejecting", |_| Err::<ResidualDecision<NoStorage>, _>(ResidualRejection::new("never")));
        assert_eq!(
            rejecting.classify(&candidate).map(|_| ()),
            Err(ResidualPolicyError::Rejected {
                policy: "rejecting".to_owned(),
                rejection: ResidualRejection::new("never"),
            }),
        );
    }

    #[test]
    fn test_residual_policy_reference_lift() {
        // A policy over `ArrayType` that saves dot products through a `NegationStorage` and recomputes everything else.
        let source = ResidualPolicyReference::<ArrayType>::new(TestPolicy {
            name: "store_dots",
            classify: |candidate: &ResidualCandidate<'_, ArrayType>| {
                Ok(match candidate.producers().iter().any(|producer| producer.payload::<DotOperation>().is_some()) {
                    true => ResidualDecision::SaveWith(NegationStorage),
                    false => ResidualDecision::Recompute,
                })
            },
        });
        let dot = TestOperation::from(dot());
        let sine = TestOperation::from(ArrayOperation::<Array>::from(SinOperation::<ArrayType>::new()));
        let dimension_size = dimension_size();

        // Lifting by projection keeps the identifier and the name of the source policy, and its decisions for
        // projectable candidates, including its storage, whose residual types are projected too.
        let lifted = source.lift::<ArrayIrType>();
        assert_eq!(lifted.id(), source.id());
        assert_eq!(lifted.name(), "store_dots");
        let Ok(ResidualDecision::SaveWith(storage)) =
            lifted.classify(&candidate(&dot, vec![vector_type(), vector_type()], scalar_type()))
        else {
            panic!("expected a stored residual");
        };
        assert_eq!(storage.name(), "negation");
        assert_eq!(storage.store_payloads(&scalar_type()).unwrap().len(), 1);
        let negation = TestOperation::from(ArrayOperation::<Array>::from(NegOperation::<ArrayType>::new()));
        assert!(storage.matches_store_operation(&scalar_type(), 0, &negation));
        assert!(!storage.matches_store_operation(&scalar_type(), 1, &negation));
        assert!(!storage.matches_store_operation(&dimension_type(), 0, &negation));
        assert_eq!(
            storage.store_payloads(&dimension_type()).map(|_| ()),
            Err(ResidualPolicyError::UnsupportedStorage {
                storage: "negation".to_owned(),
                residual_type: dimension_type().to_string(),
                message: "the type does not project into the type universe of the storage".to_owned(),
            }),
        );
        assert!(matches!(
            lifted.classify(&candidate(&sine, vec![scalar_type()], scalar_type())),
            Ok(ResidualDecision::Recompute),
        ));

        // Candidates that do not project are unsupported, because the policy registers neither a native instantiation
        // nor a projection fallback for `ArrayIrType`.
        assert_eq!(
            lifted.classify(&candidate(&dimension_size, vec![vector_type()], dimension_type())).map(|_| ()),
            Err(ResidualPolicyError::UnsupportedProjection {
                policy: "store_dots".to_owned(),
                position: "the output 0 of producer `dimension_size`".to_owned(),
                residual_type: dimension_type().to_string(),
            }),
        );
    }

    #[test]
    fn test_partitioned_program_with_residual_policy() {
        let program = sin_dot_program();
        let partition = program.partition(&[true, false]).unwrap();
        let expected = program.interpret(sin_dot_inputs()).unwrap();

        // Saving everything reproduces the partition, which saves the cosine.
        let placed = program.partition(&[true, false]).unwrap().with_residual_policy(&save_everything()).unwrap();
        assert_eq!(placed.to_string(), partition.to_string());
        assert_eq!(
            placed.to_string(),
            indoc! {"
                partition [
                    known_inputs=[0],
                    residual_inputs=[UnknownInput(1), ResidualEdge(0)],
                    outputs=[Known(0), Unknown(0)],
                ]
                known={
                    lambda %0:f64[3] .
                    let %1:f64[] = dot [
                        dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                    ] %0 %0
                        %2:f64[] = sin %1
                        %3:f64[] = cos %1
                    in (%2, %3)
                }
                residual={
                    lambda %0:f64[], %1:f64[] .
                    let %2:f64[] = mul %1 %0
                    in (%2)
                }"},
        );

        // Saving only dot products saves the dot product and recomputes its cosine in the residual program.
        let placed = partition.with_residual_policy(&save_dots(None)).unwrap();
        assert_eq!(
            placed.to_string(),
            indoc! {"
                partition [
                    known_inputs=[0],
                    residual_inputs=[UnknownInput(1), ResidualEdge(0)],
                    outputs=[Known(0), Unknown(0)],
                ]
                known={
                    lambda %0:f64[3] .
                    let %1:f64[] = dot [
                        dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                    ] %0 %0
                        %2:f64[] = sin %1
                    in (%2, %1)
                }
                residual={
                    lambda %0:f64[], %1:f64[] .
                    let %2:f64[] = cos %1
                        %3:f64[] = mul %2 %0
                    in (%3)
                }"},
        );
        assert_eq!(run(&placed, &sin_dot_inputs()), expected);

        // Saving nothing recomputes the dot product too, which saves the known input that it needs.
        let placed = program.partition(&[true, false]).unwrap().with_residual_policy(&save_nothing()).unwrap();
        assert_eq!(
            placed.to_string(),
            indoc! {"
                partition [
                    known_inputs=[0],
                    residual_inputs=[UnknownInput(1), ResidualEdge(0)],
                    outputs=[Known(0), Unknown(0)],
                ]
                known={
                    lambda %0:f64[3] .
                    let %1:f64[] = dot [
                        dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                    ] %0 %0
                        %2:f64[] = sin %1
                    in (%2, %0)
                }
                residual={
                    lambda %0:f64[], %1:f64[3] .
                    let %2:f64[] = dot [
                        dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                    ] %1 %1
                        %3:f64[] = cos %2
                        %4:f64[] = mul %3 %0
                    in (%4)
                }"},
        );
        assert_eq!(run(&placed, &sin_dot_inputs()), expected);

        // A rejection by the policy fails the placement of residuals.
        let rejecting = policy("rejecting", |_| Err::<ResidualDecision<NoStorage>, _>(ResidualRejection::new("never")));
        assert_eq!(
            program.partition(&[true, false]).unwrap().with_residual_policy(&rejecting).map(|_| ()),
            Err(ResidualPolicyError::Rejected {
                policy: "rejecting".to_owned(),
                rejection: ResidualRejection::new("never"),
            }),
        );
    }

    #[test]
    fn test_partitioned_program_with_residual_policy_saves_no_unneeded_inputs() {
        // `f(x, t) = (exp(x), exp(x) * t)`: saving everything saves only `exp(x)`, never its input.
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let x = builder.add_input(scalar_type());
        let t = builder.add_input(scalar_type());
        let exponential = add(&mut builder, ExpOperation::<ArrayType>::new().into(), vec![x]);
        let tangent = add(&mut builder, MulOperation::<ArrayType>::new().into(), vec![exponential, t]);
        let program = build(builder, vec![exponential, tangent]);
        let placed = program.partition(&[true, false]).unwrap().with_residual_policy(&save_everything()).unwrap();
        assert_eq!(
            placed.to_string(),
            indoc! {"
                partition [
                    known_inputs=[0],
                    residual_inputs=[UnknownInput(1), ResidualEdge(0)],
                    outputs=[Known(0), Unknown(0)],
                ]
                known={
                    lambda %0:f64[] .
                    let %1:f64[] = exp %0
                    in (%1, %1)
                }
                residual={
                    lambda %0:f64[], %1:f64[] .
                    let %2:f64[] = mul %1 %0
                    in (%2)
                }"},
        );
    }

    #[test]
    fn test_partitioned_program_with_residual_policy_decides_per_producer_output() {
        let program = condition_program(true);
        let reversed = condition_program(false);
        let inputs = vec![
            ArrayIrValue::Array(Array::scalar(true).unwrap()),
            ArrayIrValue::Array(Array::scalar(0.5f64).unwrap()),
            ArrayIrValue::Array(Array::scalar(2.0f64).unwrap()),
        ];
        let expected = program.interpret(inputs.clone()).unwrap();

        // Saving the first output of the condition and recomputing the second replays the condition in the residual
        // program, where the saved first output still resolves to its edge. Pruning then leaves each program with a
        // condition that computes only the output that it uses.
        let placed = program
            .partition(&[true, true, false])
            .unwrap()
            .with_residual_policy(&save_names(&["first"], &[]))
            .unwrap();
        assert_eq!(
            placed.to_string(),
            indoc! {"
                partition [
                    known_inputs=[0, 1],
                    residual_inputs=[UnknownInput(2), ResidualEdge(0), ResidualEdge(1), ResidualEdge(2)],
                    outputs=[Unknown(0), Unknown(1)],
                ]
                known={
                    lambda %0:bool[], %1:f64[] .
                    let %2:f64[] = condition %0 %1 [
                        true={
                            lambda %0:f64[] .
                            let %1:f64[] = sin %0
                                %2:f64[] = tag [key=first] %1
                            in (%2)
                        },
                        false={
                            lambda %0:f64[] .
                            let %1:f64[] = sin %0
                                %2:f64[] = tag [key=first] %1
                            in (%2)
                        },
                    ]
                    in (%2, %0, %1)
                }
                residual={
                    lambda %0:f64[], %1:f64[], %2:bool[], %3:f64[] .
                    let %4:f64[] = mul %1 %0
                        %5:f64[] = condition %2 %3 [
                            true={
                                lambda %0:f64[] .
                                let %1:f64[] = cos %0
                                    %2:f64[] = tag [key=second] %1
                                in (%2)
                            },
                            false={
                                lambda %0:f64[] .
                                let %1:f64[] = cos %0
                                    %2:f64[] = tag [key=second] %1
                                in (%2)
                            },
                        ]
                        %6:f64[] = mul %5 %0
                    in (%4, %6)
                }"},
        );
        assert_eq!(run(&placed, &inputs), expected);

        // The saved output resolves to its edge in the other demand order too, where the residual program consumes the
        // second output of the condition first.
        let placed = reversed
            .partition(&[true, true, false])
            .unwrap()
            .with_residual_policy(&save_names(&["first"], &[]))
            .unwrap();
        assert_eq!(
            placed.to_string(),
            indoc! {"
                partition [
                    known_inputs=[0, 1],
                    residual_inputs=[UnknownInput(2), ResidualEdge(0), ResidualEdge(1), ResidualEdge(2)],
                    outputs=[Unknown(0), Unknown(1)],
                ]
                known={
                    lambda %0:bool[], %1:f64[] .
                    let %2:f64[] = condition %0 %1 [
                        true={
                            lambda %0:f64[] .
                            let %1:f64[] = sin %0
                                %2:f64[] = tag [key=first] %1
                            in (%2)
                        },
                        false={
                            lambda %0:f64[] .
                            let %1:f64[] = sin %0
                                %2:f64[] = tag [key=first] %1
                            in (%2)
                        },
                    ]
                    in (%2, %0, %1)
                }
                residual={
                    lambda %0:f64[], %1:f64[], %2:bool[], %3:f64[] .
                    let %4:f64[] = condition %2 %3 [
                        true={
                            lambda %0:f64[] .
                            let %1:f64[] = cos %0
                                %2:f64[] = tag [key=second] %1
                            in (%2)
                        },
                        false={
                            lambda %0:f64[] .
                            let %1:f64[] = cos %0
                                %2:f64[] = tag [key=second] %1
                            in (%2)
                        },
                    ]
                        %5:f64[] = mul %4 %0
                        %6:f64[] = mul %1 %0
                    in (%5, %6)
                }"},
        );
        assert_eq!(run(&placed, &inputs), reversed.interpret(inputs.clone()).unwrap());

        // Storing the first output instead of saving it stages its storage around its edge.
        let placed = program
            .partition(&[true, true, false])
            .unwrap()
            .with_residual_policy(&save_names(&[], &["first"]))
            .unwrap();
        assert_eq!(
            placed.to_string(),
            indoc! {"
                partition [
                    known_inputs=[0, 1],
                    residual_inputs=[UnknownInput(2), ResidualEdge(0), ResidualEdge(1), ResidualEdge(2)],
                    outputs=[Unknown(0), Unknown(1)],
                ]
                known={
                    lambda %0:bool[], %1:f64[] .
                    let %2:f64[] = condition %0 %1 [
                        true={
                            lambda %0:f64[] .
                            let %1:f64[] = sin %0
                                %2:f64[] = tag [key=first] %1
                            in (%2)
                        },
                        false={
                            lambda %0:f64[] .
                            let %1:f64[] = sin %0
                                %2:f64[] = tag [key=first] %1
                            in (%2)
                        },
                    ]
                        %3:f64[] = neg %2
                    in (%3, %0, %1)
                }
                residual={
                    lambda %0:f64[], %1:f64[], %2:bool[], %3:f64[] .
                    let %4:f64[] = neg %1
                        %5:f64[] = mul %4 %0
                        %6:f64[] = condition %2 %3 [
                            true={
                                lambda %0:f64[] .
                                let %1:f64[] = cos %0
                                    %2:f64[] = tag [key=second] %1
                                in (%2)
                            },
                            false={
                                lambda %0:f64[] .
                                let %1:f64[] = cos %0
                                    %2:f64[] = tag [key=second] %1
                                in (%2)
                            },
                        ]
                        %7:f64[] = mul %6 %0
                    in (%5, %7)
                }"},
        );
        assert_eq!(run(&placed, &inputs), expected);

        // Saving one output and storing the other replays nothing.
        let placed = program
            .partition(&[true, true, false])
            .unwrap()
            .with_residual_policy(&save_names(&["first"], &["second"]))
            .unwrap();
        assert_eq!(
            placed.to_string(),
            indoc! {"
                partition [
                    known_inputs=[0, 1],
                    residual_inputs=[UnknownInput(2), ResidualEdge(0), ResidualEdge(1)],
                    outputs=[Unknown(0), Unknown(1)],
                ]
                known={
                    lambda %0:bool[], %1:f64[] .
                    let %2:f64[], %3:f64[] = condition %0 %1 [
                        true={
                            lambda %0:f64[] .
                            let %1:f64[] = sin %0
                                %2:f64[] = tag [key=first] %1
                                %3:f64[] = cos %0
                                %4:f64[] = tag [key=second] %3
                            in (%2, %4)
                        },
                        false={
                            lambda %0:f64[] .
                            let %1:f64[] = sin %0
                                %2:f64[] = tag [key=first] %1
                                %3:f64[] = cos %0
                                %4:f64[] = tag [key=second] %3
                            in (%2, %4)
                        },
                    ]
                        %4:f64[] = neg %3
                    in (%2, %4)
                }
                residual={
                    lambda %0:f64[], %1:f64[], %2:f64[] .
                    let %3:f64[] = mul %1 %0
                        %4:f64[] = neg %2
                        %5:f64[] = mul %4 %0
                    in (%3, %5)
                }"},
        );
        assert_eq!(run(&placed, &inputs), expected);

        // Saving nothing replays the complete condition over the known inputs, which are saved instead.
        let placed = program.partition(&[true, true, false]).unwrap().with_residual_policy(&save_nothing()).unwrap();
        assert_eq!(
            placed.to_string(),
            indoc! {"
                partition [
                    known_inputs=[0, 1],
                    residual_inputs=[UnknownInput(2), ResidualEdge(0), ResidualEdge(1)],
                    outputs=[Unknown(0), Unknown(1)],
                ]
                known={
                    lambda %0:bool[], %1:f64[] .
                    in (%0, %1)
                }
                residual={
                    lambda %0:f64[], %1:bool[], %2:f64[] .
                    let %3:f64[], %4:f64[] = condition %1 %2 [
                        true={
                            lambda %0:f64[] .
                            let %1:f64[] = sin %0
                                %2:f64[] = tag [key=first] %1
                                %3:f64[] = cos %0
                                %4:f64[] = tag [key=second] %3
                            in (%2, %4)
                        },
                        false={
                            lambda %0:f64[] .
                            let %1:f64[] = sin %0
                                %2:f64[] = tag [key=first] %1
                                %3:f64[] = cos %0
                                %4:f64[] = tag [key=second] %3
                            in (%2, %4)
                        },
                    ]
                        %5:f64[] = mul %3 %0
                        %6:f64[] = mul %4 %0
                    in (%5, %6)
                }"},
        );
        assert_eq!(run(&placed, &inputs), expected);
    }

    #[test]
    fn test_partitioned_program_with_residual_policy_saves_external_reads() {
        // `f(r, t)` reads the external reference `r`, overwrites it with the cosine of the value that it read, and
        // reads it again, returning the sines of both reads multiplied by `t`. Reads of `r` cannot be replayed, so they
        // are saved in topological order even though the policy saves nothing, and the zero-output write stays in the
        // known program although nothing demands it.
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let r = builder.add_input(ReferenceType::new(ArrayType::scalar(DataType::F64)).into());
        let t = builder.add_input(scalar_type());
        let a = builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![r], None).unwrap()[0];
        let cosine = add(&mut builder, CosOperation::<ArrayType>::new().into(), vec![a]);
        builder.add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![r, cosine], None).unwrap();
        let b = builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![r], None).unwrap()[0];
        let sine_a = add(&mut builder, SinOperation::<ArrayType>::new().into(), vec![a]);
        let sine_b = add(&mut builder, SinOperation::<ArrayType>::new().into(), vec![b]);
        let tangent_a = add(&mut builder, MulOperation::<ArrayType>::new().into(), vec![sine_a, t]);
        let tangent_b = add(&mut builder, MulOperation::<ArrayType>::new().into(), vec![sine_b, t]);
        let program = build(builder, vec![tangent_a, tangent_b]);
        let placed = program.partition(&[true, false]).unwrap().with_residual_policy(&save_nothing()).unwrap();
        assert_eq!(
            placed.to_string(),
            indoc! {"
                partition [
                    known_inputs=[0],
                    residual_inputs=[UnknownInput(1), ResidualEdge(0), ResidualEdge(1)],
                    outputs=[Unknown(0), Unknown(1)],
                ]
                known={
                    lambda %0:ref<f64[]> .
                    let %1:f64[] = reference_read %0
                        %2:f64[] = cos %1
                        () = reference_write %0 %2
                        %3:f64[] = reference_read %0
                    in (%1, %3)
                }
                residual={
                    lambda %0:f64[], %1:f64[], %2:f64[] .
                    let %3:f64[] = sin %1
                        %4:f64[] = mul %3 %0
                        %5:f64[] = sin %2
                        %6:f64[] = mul %5 %0
                    in (%4, %6)
                }"},
        );
    }

    #[test]
    fn test_partitioned_program_with_residual_policy_replays_local_reference_lifecycles() {
        // `f(x, t)` allocates a reference holding `x`, reads it, overwrites it with `sin(x)`, reads it again, and
        // freezes it, returning each observed value multiplied by `t`. Saving nothing replays the complete lifecycle in
        // the residual program, in program order, and removes it from the known program, which no longer observes it.
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let x = builder.add_input(scalar_type());
        let t = builder.add_input(scalar_type());
        let r = builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![x], None).unwrap()[0];
        let a = builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![r], None).unwrap()[0];
        let sine = add(&mut builder, SinOperation::<ArrayType>::new().into(), vec![x]);
        builder.add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![r, sine], None).unwrap();
        let b = builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![r], None).unwrap()[0];
        let f = builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![r], None).unwrap()[0];
        let tangent_a = add(&mut builder, MulOperation::<ArrayType>::new().into(), vec![a, t]);
        let tangent_b = add(&mut builder, MulOperation::<ArrayType>::new().into(), vec![b, t]);
        let tangent_f = add(&mut builder, MulOperation::<ArrayType>::new().into(), vec![f, t]);
        let program = build(builder, vec![tangent_a, tangent_b, tangent_f]);
        let partition = program.partition(&[true, false]).unwrap();
        assert_eq!(
            partition.to_string(),
            indoc! {"
                partition [
                    known_inputs=[0],
                    residual_inputs=[UnknownInput(1), ResidualEdge(0), ResidualEdge(1), ResidualEdge(2)],
                    outputs=[Unknown(0), Unknown(1), Unknown(2)],
                ]
                known={
                    lambda %0:f64[] .
                    let %1:ref<f64[]> = reference_new %0
                        %2:f64[] = reference_read %1
                        %3:f64[] = sin %0
                        () = reference_write %1 %3
                        %4:f64[] = reference_read %1
                        %5:f64[] = reference_freeze %1
                    in (%2, %4, %5)
                }
                residual={
                    lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[] .
                    let %4:f64[] = mul %1 %0
                        %5:f64[] = mul %2 %0
                        %6:f64[] = mul %3 %0
                    in (%4, %5, %6)
                }"},
        );
        let placed = partition.with_residual_policy(&save_nothing()).unwrap();
        assert_eq!(
            placed.to_string(),
            indoc! {"
                partition [
                    known_inputs=[0],
                    residual_inputs=[UnknownInput(1), ResidualEdge(0)],
                    outputs=[Unknown(0), Unknown(1), Unknown(2)],
                ]
                known={
                    lambda %0:f64[] .
                    in (%0)
                }
                residual={
                    lambda %0:f64[], %1:f64[] .
                    let %2:ref<f64[]> = reference_new %1
                        %3:f64[] = reference_read %2
                        %4:f64[] = sin %1
                        () = reference_write %2 %4
                        %5:f64[] = reference_read %2
                        %6:f64[] = reference_freeze %2
                        %7:f64[] = mul %3 %0
                        %8:f64[] = mul %5 %0
                        %9:f64[] = mul %6 %0
                    in (%7, %8, %9)
                }"},
        );
        let inputs = vec![
            ArrayIrValue::Array(Array::scalar(0.5f64).unwrap()),
            ArrayIrValue::Array(Array::scalar(2.0f64).unwrap()),
        ];
        assert_eq!(run(&placed, &inputs), program.interpret(inputs.clone()).unwrap());
    }

    #[test]
    fn test_partitioned_program_with_residual_policy_replays_nested_reference_lifecycles() {
        // Each branch writes a different value to state allocated by the enclosing program and reads it back. Saving
        // nothing moves the complete lifecycle to residual execution, leaving neither its condition nor orphaned
        // branch regions in the known program.
        let reference_type: ArrayIrType = ReferenceType::new(ArrayType::scalar(DataType::F64)).into();
        let mut branch = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = branch.add_input(reference_type.clone());
        let input = branch.add_input(scalar_type());
        let sine = add(&mut branch, SinOperation::<ArrayType>::new().into(), vec![input]);
        branch
            .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![reference, sine], None)
            .unwrap();
        let read = branch.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let true_branch = build(branch, vec![read]);

        let mut branch = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = branch.add_input(reference_type);
        let input = branch.add_input(scalar_type());
        let cosine = add(&mut branch, CosOperation::<ArrayType>::new().into(), vec![input]);
        branch
            .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![reference, cosine], None)
            .unwrap();
        let read = branch.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let false_branch = build(branch, vec![read]);

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let input = builder.add_input(scalar_type());
        let unknown = builder.add_input(scalar_type());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        let true_branch = builder.import_program(true_branch);
        let false_branch = builder.import_program(false_branch);
        let selected = builder
            .add_instruction(
                ConditionOperation::<ArrayIrType>::new(),
                vec![true_branch, false_branch],
                vec![predicate, reference, input],
                None,
            )
            .unwrap()[0];
        let output = add(&mut builder, MulOperation::<ArrayType>::new().into(), vec![selected, unknown]);
        let program = build(builder, vec![output]);
        let placed = program.partition(&[true, true, false]).unwrap().with_residual_policy(&save_nothing()).unwrap();
        assert_eq!(
            placed.to_string(),
            indoc! {"
                partition [
                    known_inputs=[0, 1],
                    residual_inputs=[UnknownInput(2), ResidualEdge(0), ResidualEdge(1)],
                    outputs=[Unknown(0)],
                ]
                known={
                    lambda %0:bool[], %1:f64[] .
                    in (%0, %1)
                }
                residual={
                    lambda %0:f64[], %1:bool[], %2:f64[] .
                    let %3:ref<f64[]> = reference_new %2
                        %4:f64[] = condition %1 %3 %2 [
                            true={
                                lambda %0:ref<f64[]>, %1:f64[] .
                                let %2:f64[] = sin %1
                                    () = reference_write %0 %2
                                    %3:f64[] = reference_read %0
                                in (%3)
                            },
                            false={
                                lambda %0:ref<f64[]>, %1:f64[] .
                                let %2:f64[] = cos %1
                                    () = reference_write %0 %2
                                    %3:f64[] = reference_read %0
                                in (%3)
                            },
                        ]
                        %5:f64[] = mul %4 %0
                    in (%5)
                }"},
        );
        for (predicate, expected) in [(true, 2.0 * 0.5f64.sin()), (false, 2.0 * 0.5f64.cos())] {
            let inputs = vec![
                ArrayIrValue::Array(Array::scalar(predicate).unwrap()),
                ArrayIrValue::Array(Array::scalar(0.5f64).unwrap()),
                ArrayIrValue::Array(Array::scalar(2.0f64).unwrap()),
            ];
            let expected = vec![ArrayIrValue::Array(Array::scalar(expected).unwrap())];
            assert_eq!(program.interpret(inputs.clone()), Ok(expected.clone()));
            assert_eq!(run(&placed, &inputs), expected);
        }
    }

    #[test]
    fn test_partitioned_program_with_residual_policy_replays_many_reference_observations() {
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let input = builder.add_input(scalar_type());
        let unknown = builder.add_input(scalar_type());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        let mut outputs = Vec::new();
        for index in 0..2000 {
            let value = builder.add_constant(ArrayIrValue::Array(Array::scalar(index as f64).unwrap()));
            builder
                .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![reference, value], None)
                .unwrap();
            let read =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
            outputs.push(add(&mut builder, MulOperation::<ArrayType>::new().into(), vec![read, unknown]));
        }
        let program = build(builder, outputs);
        let placed = program.partition(&[true, false]).unwrap().with_residual_policy(&save_nothing()).unwrap();
        assert!(placed.known_program().instructions().is_empty());
        assert_eq!(placed.known_program().output_ids().len(), 1);
        let inputs =
            vec![ArrayIrValue::Array(Array::scalar(0f64).unwrap()), ArrayIrValue::Array(Array::scalar(2f64).unwrap())];
        let expected = (0..2000)
            .map(|index| ArrayIrValue::Array(Array::scalar(2f64 * index as f64).unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(run(&placed, &inputs), expected);
        assert_eq!(program.interpret(inputs).unwrap(), expected);
    }

    #[test]
    fn test_partitioned_program_with_residual_policy_stages_storage() {
        // The known program stores the dot product and the residual program restores it before the cosine uses it.
        let program = sin_dot_program();
        let placed = program
            .partition(&[true, false])
            .unwrap()
            .with_residual_policy(&save_dots(Some(NegationStorage)))
            .unwrap();
        assert_eq!(
            placed.to_string(),
            indoc! {"
                partition [
                    known_inputs=[0],
                    residual_inputs=[UnknownInput(1), ResidualEdge(0)],
                    outputs=[Known(0), Unknown(0)],
                ]
                known={
                    lambda %0:f64[3] .
                    let %1:f64[] = dot [
                        dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                    ] %0 %0
                        %2:f64[] = sin %1
                        %3:f64[] = neg %1
                    in (%2, %3)
                }
                residual={
                    lambda %0:f64[], %1:f64[] .
                    let %2:f64[] = neg %1
                        %3:f64[] = cos %2
                        %4:f64[] = mul %3 %0
                    in (%4)
                }"},
        );
        assert_eq!(run(&placed, &sin_dot_inputs()), program.interpret(sin_dot_inputs()).unwrap());

        // Storage that does not reproduce the residual type or whose payloads the operation family cannot hold fails.

        /// [`ResidualStorage`] that stores residuals by transferring them to host memory and whose restoration returns
        /// them unchanged, which violates the storage contract because it does not reproduce the residual type.
        #[derive(Copy, Clone, Debug, PartialEq, Eq)]
        struct ForgetfulStorage;

        impl<T: Type> ResidualStorage<T> for ForgetfulStorage {
            fn name(&self) -> String {
                "forgetful".to_owned()
            }

            fn store_payloads(&self, _residual_type: &T) -> Result<Vec<ErasedOperation>, ResidualPolicyError> {
                Ok(vec![ErasedOperation::new(TransferToMemoryOperation::new(Memory::Host { pinned: true }))])
            }

            fn restore_payloads(
                &self,
                _stored_type: &T,
                _residual_type: &T,
            ) -> Result<Vec<ErasedOperation>, ResidualPolicyError> {
                Ok(Vec::new())
            }
        }

        /// [`ResidualStorage`] whose payload the operation family of the tests cannot hold.
        #[derive(Copy, Clone, Debug, PartialEq, Eq)]
        struct UnsupportedStorage;

        impl<T: Type> ResidualStorage<T> for UnsupportedStorage {
            fn name(&self) -> String {
                "unsupported".to_owned()
            }

            fn store_payloads(&self, _residual_type: &T) -> Result<Vec<ErasedOperation>, ResidualPolicyError> {
                Ok(vec![ErasedOperation::new(TagOperation::<ArrayIrType>::new("unsupported"))])
            }

            fn restore_payloads(
                &self,
                _stored_type: &T,
                _residual_type: &T,
            ) -> Result<Vec<ErasedOperation>, ResidualPolicyError> {
                Ok(Vec::new())
            }
        }

        /// Returns a policy that saves the residuals produced by dot products through `storage`.
        fn store_dots<S: Copy + ResidualStorage<ArrayIrType>>(storage: S) -> ResidualPolicyReference<ArrayIrType> {
            policy("store_dots", move |candidate| {
                Ok(match candidate.producers().iter().any(|producer| producer.payload::<DotOperation>().is_some()) {
                    true => ResidualDecision::SaveWith(storage),
                    false => ResidualDecision::Recompute,
                })
            })
        }
        assert_eq!(
            program
                .partition(&[true, false])
                .unwrap()
                .with_residual_policy(&store_dots(ForgetfulStorage))
                .map(|_| ()),
            Err(ResidualPolicyError::InvalidStorage {
                storage: "forgetful".to_owned(),
                message: "its restore operations produce `f64[]@Host[Pinned]` instead of the residual type `f64[]`"
                    .to_owned(),
            }),
        );
        assert_eq!(
            program
                .partition(&[true, false])
                .unwrap()
                .with_residual_policy(&store_dots(UnsupportedStorage))
                .map(|_| ()),
            Err(ResidualPolicyError::UnsupportedStorage {
                storage: "unsupported".to_owned(),
                residual_type: "f64[]".to_owned(),
                message: format!(
                    "the operation family of the program cannot hold its payload `{}`",
                    std::any::type_name::<TagOperation<ArrayIrType>>(),
                ),
            }),
        );
    }

    #[test]
    fn test_partitioned_program_with_residual_policy_reuses_identical_stored_outputs() {
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let x = builder.add_input(scalar_type());
        let y = builder.add_input(scalar_type());
        let sine = add(&mut builder, SinOperation::<ArrayType>::new().into(), vec![x]);
        let stored = add(&mut builder, NegOperation::<ArrayType>::new().into(), vec![sine]);
        let product = add(&mut builder, MulOperation::<ArrayType>::new().into(), vec![sine, y]);
        let program = build(builder, vec![product, sine, stored]);
        let policy = policy("store_sines", |candidate| {
            Ok(
                if candidate.producers().iter().any(|producer| producer.payload::<SinOperation<ArrayType>>().is_some())
                {
                    ResidualDecision::SaveWith(NegationStorage)
                } else {
                    ResidualDecision::Recompute
                },
            )
        });

        // The known output already negates precisely the saved sine. Reuse that stored atom as the residual edge.
        let stored_output = program.clone().with_outputs(&[0, 2]).unwrap();
        let reused = stored_output.partition(&[true, false]).unwrap().with_residual_policy(&policy).unwrap();
        assert_eq!(
            reused.to_string(),
            indoc! {"
                partition [
                    known_inputs=[0],
                    residual_inputs=[UnknownInput(1), ResidualEdge(0)],
                    outputs=[Unknown(0), Known(0)],
                ]
                known={
                    lambda %0:f64[] .
                    let %1:f64[] = sin %0
                        %2:f64[] = neg %1
                    in (%2, %2)
                }
                residual={
                    lambda %0:f64[], %1:f64[] .
                    let %2:f64[] = neg %1
                        %3:f64[] = mul %2 %0
                    in (%3)
                }"},
        );

        // A known logical sine has the same type as its negated storage but a different representation. It must not
        // replace the stored edge, or the residual restore would negate the logical value a second time.
        let logical_output = program.with_outputs(&[0, 1]).unwrap();
        let distinct = logical_output.partition(&[true, false]).unwrap().with_residual_policy(&policy).unwrap();
        assert_eq!(
            distinct.to_string(),
            indoc! {"
                partition [
                    known_inputs=[0],
                    residual_inputs=[UnknownInput(1), ResidualEdge(0)],
                    outputs=[Unknown(0), Known(0)],
                ]
                known={
                    lambda %0:f64[] .
                    let %1:f64[] = sin %0
                        %2:f64[] = neg %1
                    in (%1, %2)
                }
                residual={
                    lambda %0:f64[], %1:f64[] .
                    let %2:f64[] = neg %1
                        %3:f64[] = mul %2 %0
                    in (%3)
                }"},
        );
        let inputs = vec![
            ArrayIrValue::Array(Array::scalar(0.5f64).unwrap()),
            ArrayIrValue::Array(Array::scalar(3.0f64).unwrap()),
        ];
        for (original, partition) in [(stored_output, reused), (logical_output, distinct)] {
            let known = partition.known_program().interpret(vec![inputs[0].clone()]).unwrap();
            let residual = partition.residual_program().interpret(vec![inputs[1].clone(), known[1].clone()]).unwrap();
            assert_eq!(original.interpret(inputs.clone()), Ok(vec![residual[0].clone(), known[0].clone()]));
        }
    }

    #[test]
    fn test_partitioned_program_with_residual_policy_resolves_provenance_through_shared_regions() {
        // Three levels of conditions share their branch regions and forward their inputs: the innermost region
        // returns its input, and each enclosing region returns the output of a condition over the next region. Two
        // top-level conditions invoke the shared regions with differently tagged inputs, so the provenance of each
        // resolves to the producer of its own input.
        let condition =
            |builder: &mut ProgramBuilder<TestValue, TestOperation>, branch: RegionId, inputs: Vec<AtomId>| {
                builder.add_instruction(ConditionOperation::new(), vec![branch, branch], inputs, None).unwrap()[0]
            };
        let predicate_type: ArrayIrType = ArrayType::scalar(DataType::Boolean).into();
        let mut inner = ProgramBuilder::<TestValue, TestOperation>::new();
        inner.add_input(predicate_type.clone());
        let a = inner.add_input(scalar_type());
        let mut region = build(inner, vec![a]);
        for _ in 0..2 {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let p = builder.add_input(predicate_type.clone());
            let b = builder.add_input(scalar_type());
            let branch = builder.import_program(region);
            let output = condition(&mut builder, branch, vec![p, p, b]);
            region = build(builder, vec![output]);
        }

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let p = builder.add_input(predicate_type);
        let x = builder.add_input(scalar_type());
        let t = builder.add_input(scalar_type());
        let sine = add(&mut builder, SinOperation::<ArrayType>::new().into(), vec![x]);
        let u = add(&mut builder, TagOperation::<ArrayType>::new("u").into(), vec![sine]);
        let cosine = add(&mut builder, CosOperation::<ArrayType>::new().into(), vec![x]);
        let v = add(&mut builder, TagOperation::<ArrayType>::new("v").into(), vec![cosine]);
        let branch = builder.import_program(region);
        let first = condition(&mut builder, branch, vec![p, p, u]);
        let second = condition(&mut builder, branch, vec![p, p, v]);
        let tangent_first = add(&mut builder, MulOperation::<ArrayType>::new().into(), vec![first, t]);
        let tangent_second = add(&mut builder, MulOperation::<ArrayType>::new().into(), vec![second, t]);
        let program = build(builder, vec![tangent_first, tangent_second]);

        // Saving only `u` saves the first condition's output and replays the second condition in the residual program.
        let placed = program
            .partition(&[true, true, false])
            .unwrap()
            .with_residual_policy(&save_names(&["u"], &[]))
            .unwrap();
        assert_eq!(
            placed.to_string(),
            indoc! {"
                partition [
                    known_inputs=[0, 1],
                    residual_inputs=[UnknownInput(2), ResidualEdge(0), ResidualEdge(1), ResidualEdge(2)],
                    outputs=[Unknown(0), Unknown(1)],
                ]
                known={
                    lambda %0:bool[], %1:f64[] .
                    let %2:f64[] = sin %1
                        %3:f64[] = tag [key=u] %2
                        %4:f64[] = condition %0 %0 %3 [
                            true={
                                lambda %0:bool[], %1:f64[] .
                                let %2:f64[] = condition %0 %0 %1 [
                                    true=^1={
                                        lambda %0:bool[], %1:f64[] .
                                        let %2:f64[] = condition %0 %1 [
                                            true=^0={
                                                lambda %0:f64[] .
                                                in (%0)
                                            },
                                            false=^0,
                                        ]
                                        in (%2)
                                    },
                                    false=^1,
                                ]
                                in (%2)
                            },
                            false={
                                lambda %0:bool[], %1:f64[] .
                                let %2:f64[] = condition %0 %0 %1 [
                                    true=^4={
                                        lambda %0:bool[], %1:f64[] .
                                        let %2:f64[] = condition %0 %1 [
                                            true=^3={
                                                lambda %0:f64[] .
                                                in (%0)
                                            },
                                            false=^3,
                                        ]
                                        in (%2)
                                    },
                                    false=^4,
                                ]
                                in (%2)
                            },
                        ]
                    in (%4, %0, %1)
                }
                residual={
                    lambda %0:f64[], %1:f64[], %2:bool[], %3:f64[] .
                    let %4:f64[] = mul %1 %0
                        %5:f64[] = cos %3
                        %6:f64[] = tag [key=v] %5
                        %7:f64[] = condition %2 %2 %6 [
                            true={
                                lambda %0:bool[], %1:f64[] .
                                let %2:f64[] = condition %0 %0 %1 [
                                    true=^1={
                                        lambda %0:bool[], %1:f64[] .
                                        let %2:f64[] = condition %0 %1 [
                                            true=^0={
                                                lambda %0:f64[] .
                                                in (%0)
                                            },
                                            false=^0,
                                        ]
                                        in (%2)
                                    },
                                    false=^1,
                                ]
                                in (%2)
                            },
                            false={
                                lambda %0:bool[], %1:f64[] .
                                let %2:f64[] = condition %0 %0 %1 [
                                    true=^4={
                                        lambda %0:bool[], %1:f64[] .
                                        let %2:f64[] = condition %0 %1 [
                                            true=^3={
                                                lambda %0:f64[] .
                                                in (%0)
                                            },
                                            false=^3,
                                        ]
                                        in (%2)
                                    },
                                    false=^4,
                                ]
                                in (%2)
                            },
                        ]
                        %8:f64[] = mul %7 %0
                    in (%4, %8)
                }"},
        );
        let inputs = vec![
            ArrayIrValue::Array(Array::scalar(false).unwrap()),
            ArrayIrValue::Array(Array::scalar(0.5f64).unwrap()),
            ArrayIrValue::Array(Array::scalar(2.0f64).unwrap()),
        ];
        assert_eq!(run(&placed, &inputs), program.interpret(inputs.clone()).unwrap());
    }

    #[test]
    fn test_partitioned_program_with_residual_policy_reproduces_partitions_that_save_everything() {
        // A policy that saves everything reproduces the ordinary partition, including when the residual program
        // consumes the outputs of a region-carrying operation in either order.
        let program = sin_dot_program();
        let partition = program.partition(&[true, false]).unwrap();
        let rendering = partition.to_string();
        assert_eq!(partition.with_residual_policy(&save_everything()).unwrap().to_string(), rendering);
        let program = condition_program(true);
        let partition = program.partition(&[true, true, false]).unwrap();
        let rendering = partition.to_string();
        assert_eq!(partition.with_residual_policy(&save_everything()).unwrap().to_string(), rendering);
        let program = condition_program(false);
        let partition = program.partition(&[true, true, false]).unwrap();
        let rendering = partition.to_string();
        assert_eq!(partition.with_residual_policy(&save_everything()).unwrap().to_string(), rendering);
    }

    #[test]
    fn test_partitioned_program_with_residual_policy_preserves_region_outputs() {
        // The branches of a condition map `x` to `(sin(x), x)`, so the second output of the condition forwards `x`,
        // which the residual program also reads directly. Provenance alone cannot prove that a region input keeps its
        // initial value (e.g., loop carries evolve), so both the condition output and the direct input remain edges.
        let mut branch = ProgramBuilder::<TestValue, TestOperation>::new();
        let a = branch.add_input(scalar_type());
        let sine = add(&mut branch, SinOperation::<ArrayType>::new().into(), vec![a]);
        let branch = build(branch, vec![sine, a]);
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let p = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let x = builder.add_input(scalar_type());
        let t = builder.add_input(scalar_type());
        let true_branch = builder.import_program(branch.clone());
        let false_branch = builder.import_program(branch);
        let outputs = builder
            .add_instruction(ConditionOperation::new(), vec![true_branch, false_branch], vec![p, x], None)
            .unwrap()
            .to_vec();
        let first = add(&mut builder, MulOperation::<ArrayType>::new().into(), vec![outputs[0], t]);
        let second = add(&mut builder, MulOperation::<ArrayType>::new().into(), vec![outputs[1], t]);
        let third = add(&mut builder, MulOperation::<ArrayType>::new().into(), vec![x, t]);
        let program = build(builder, vec![first, second, third]);
        let partition = program.partition(&[true, true, false]).unwrap().with_residual_policy(&save_everything());
        let partition = partition.unwrap();
        assert_eq!(
            partition.to_string(),
            indoc! {"
                partition [
                    known_inputs=[0, 1],
                    residual_inputs=[UnknownInput(2), ResidualEdge(0), ResidualEdge(1), ResidualEdge(2)],
                    outputs=[Unknown(0), Unknown(1), Unknown(2)],
                ]
                known={
                    lambda %0:bool[], %1:f64[] .
                    let %2:f64[], %3:f64[] = condition %0 %1 [
                        true={
                            lambda %0:f64[] .
                            let %1:f64[] = sin %0
                            in (%1, %0)
                        },
                        false={
                            lambda %0:f64[] .
                            let %1:f64[] = sin %0
                            in (%1, %0)
                        },
                    ]
                    in (%2, %3, %1)
                }
                residual={
                    lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[] .
                    let %4:f64[] = mul %1 %0
                        %5:f64[] = mul %2 %0
                        %6:f64[] = mul %3 %0
                    in (%4, %5, %6)
                }"},
        );
        let inputs = vec![
            ArrayIrValue::Array(Array::scalar(true).unwrap()),
            ArrayIrValue::Array(Array::scalar(0.5f64).unwrap()),
            ArrayIrValue::Array(Array::scalar(2.0f64).unwrap()),
        ];
        assert_eq!(run(&partition, &inputs), program.interpret(inputs.clone()).unwrap());
    }

    #[test]
    fn test_partitioned_program_with_residual_policy_rejects_forwarded_partitions() {
        // `(x, t) ↦ x · t` with `x` known saves `x`, which forwarding then feeds to the residual program directly.
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let x = builder.add_input(scalar_type());
        let t = builder.add_input(scalar_type());
        let product = add(&mut builder, MulOperation::<ArrayType>::new().into(), vec![x, t]);
        let program = build(builder, vec![product]);
        let forwarded = program.partition(&[true, false]).unwrap().forward_residuals().unwrap();
        assert_eq!(
            forwarded.residual_inputs(),
            &[ResidualInputSource::UnknownInput(1), ResidualInputSource::KnownInput(0)],
        );
        assert_eq!(
            forwarded.with_residual_policy(&save_everything()).map(|partition| partition.to_string()),
            Err(ResidualPolicyError::Program(ProgramError::InvalidArgument {
                message: "cannot place the residuals of a partition whose residual inputs are already forwarded"
                    .to_string(),
            })),
        );
    }

    #[test]
    fn test_partitioned_program_with_rounded_residuals() {
        // With `x` known and `y` unknown, `(x, y) ↦ (sin(x)² · y + sin(x) · y + x · y)` saves `x`, `sin(x)`, and
        // `sin(x)²` for the residual program. The known program also consumes `x` and `sin(x)`, which are therefore
        // rounded right where they become available, while `sin(x)²` only feeds the residual program and stays as is.
        let scalar_type = ArrayIrType::from(ArrayType::scalar(DataType::BF16));
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let x = builder.add_input(scalar_type.clone());
        let y = builder.add_input(scalar_type);
        let sine = add(&mut builder, SinOperation::new().into(), vec![x]);
        let square = add(&mut builder, MulOperation::new().into(), vec![sine, sine]);
        let first = add(&mut builder, MulOperation::new().into(), vec![square, y]);
        let second = add(&mut builder, MulOperation::new().into(), vec![sine, y]);
        let third = add(&mut builder, MulOperation::new().into(), vec![x, y]);
        let sum = add(&mut builder, AddOperation::new().into(), vec![first, second]);
        let sum = add(&mut builder, AddOperation::new().into(), vec![sum, third]);
        let program = build(builder, vec![sum]);
        let rounding = |r#type: &ArrayIrType| {
            Some(ErasedOperation::new(ReducePrecisionOperation::<ArrayType>::new(8, 7))).filter(
                |_| matches!(r#type, ArrayIrType::Array(array_type) if array_type.data_type() == DataType::BF16),
            )
        };
        let partition = program.partition(&[true, false]).unwrap();
        assert_eq!(
            partition.with_rounded_residuals(&rounding, true).unwrap().to_string(),
            indoc! {"
                partition [
                    known_inputs=[0],
                    residual_inputs=[UnknownInput(1), ResidualEdge(0), ResidualEdge(1), ResidualEdge(2)],
                    outputs=[Unknown(0)],
                ]
                known={
                    lambda %0:bf16[] .
                    let %1:bf16[] = reduce_precision [exponent_bits=8, mantissa_bits=7] %0
                        %2:bf16[] = sin %1
                        %3:bf16[] = reduce_precision [exponent_bits=8, mantissa_bits=7] %2
                        %4:bf16[] = mul %3 %3
                    in (%4, %3, %1)
                }
                residual={
                    lambda %0:bf16[], %1:bf16[], %2:bf16[], %3:bf16[] .
                    let %4:bf16[] = mul %1 %0
                        %5:bf16[] = mul %2 %0
                        %6:bf16[] = add %4 %5
                        %7:bf16[] = mul %3 %0
                        %8:bf16[] = add %6 %7
                    in (%8)
                }"},
        );

        // An edge that store operations produce is rounded before it is stored, if the known program also consumes the
        // value that they store.
        let host = Memory::Host { pinned: true };
        let offload = policy("offload_everything", move |_: &ResidualCandidate<'_, ArrayIrType>| {
            Ok(ResidualDecision::SaveWith(MemoryTransferStorage::new(host)))
        });
        let partition = program.partition(&[true, false]).unwrap().with_residual_policy(&offload).unwrap();
        assert_eq!(
            partition.with_rounded_residuals(&rounding, true).unwrap().to_string(),
            indoc! {"
                partition [
                    known_inputs=[0],
                    residual_inputs=[UnknownInput(1), ResidualEdge(0), ResidualEdge(1), ResidualEdge(2)],
                    outputs=[Unknown(0)],
                ]
                known={
                    lambda %0:bf16[] .
                    let %1:bf16[] = reduce_precision [exponent_bits=8, mantissa_bits=7] %0
                        %2:bf16[] = sin %1
                        %3:bf16[] = reduce_precision [exponent_bits=8, mantissa_bits=7] %2
                        %4:bf16[] = mul %3 %3
                        %5:bf16[]@Host[Pinned] = transfer_to_memory [destination=Host[Pinned]] %4
                        %6:bf16[]@Host[Pinned] = transfer_to_memory [destination=Host[Pinned]] %3
                    in (%5, %6, %1)
                }
                residual={
                    lambda %0:bf16[], %1:bf16[]@Host[Pinned], %2:bf16[]@Host[Pinned], %3:bf16[] .
                    let %4:bf16[] = transfer_to_memory [destination=Device] %1
                        %5:bf16[] = mul %4 %0
                        %6:bf16[] = transfer_to_memory [destination=Device] %2
                        %7:bf16[] = mul %6 %0
                        %8:bf16[] = add %5 %7
                        %9:bf16[] = mul %3 %0
                        %10:bf16[] = add %8 %9
                    in (%10)
                }"},
        );

        // A partition whose edges need no rounding is returned unchanged.
        let partition = program.partition(&[true, false]).unwrap();
        let expected = partition.to_string();
        assert_eq!(partition.with_rounded_residuals(&|_: &ArrayIrType| None, true).unwrap().to_string(), expected);

        // Without `round_inputs`, the saved known input `x` stays unrounded, because the partitions of regions leave
        // their inputs to the program that computes them.
        let partition = program.partition(&[true, false]).unwrap();
        assert_eq!(
            partition.with_rounded_residuals(&rounding, false).unwrap().to_string(),
            indoc! {"
                partition [
                    known_inputs=[0],
                    residual_inputs=[UnknownInput(1), ResidualEdge(0), ResidualEdge(1), ResidualEdge(2)],
                    outputs=[Unknown(0)],
                ]
                known={
                    lambda %0:bf16[] .
                    let %1:bf16[] = sin %0
                        %2:bf16[] = reduce_precision [exponent_bits=8, mantissa_bits=7] %1
                        %3:bf16[] = mul %2 %2
                    in (%3, %2, %0)
                }
                residual={
                    lambda %0:bf16[], %1:bf16[], %2:bf16[], %3:bf16[] .
                    let %4:bf16[] = mul %1 %0
                        %5:bf16[] = mul %2 %0
                        %6:bf16[] = add %4 %5
                        %7:bf16[] = mul %3 %0
                        %8:bf16[] = add %6 %7
                    in (%8)
                }"},
        );

        // A rounding that recognizes its own operations does not round the values that they produce again, so rounding
        // a partition twice rounds each value once.
        let partition = program.partition(&[true, false]).unwrap();
        let rounded = partition.with_rounded_residuals(&Bfloat16Rounding, true).unwrap();
        let expected = rounded.to_string();
        assert_eq!(rounded.with_rounded_residuals(&Bfloat16Rounding, true).unwrap().to_string(), expected);

        // Forwarding would feed `x` to the residual program without passing through the edge that rounds it, so
        // rounding must precede forwarding.
        let forwarded = program.partition(&[true, false]).unwrap().forward_residuals().unwrap();
        assert!(forwarded.has_forwarded_residual_inputs());
        assert_eq!(
            forwarded.with_rounded_residuals(&rounding, true).map(|partition| partition.to_string()),
            Err(ProgramError::InvalidArgument {
                message: "cannot round the residuals of a partition whose residual inputs are already forwarded"
                    .to_string(),
            }),
        );
    }

    #[test]
    fn test_partitioned_program_with_rounded_residuals_through_custom_storage() {
        // With `x` known and `y` unknown, `(x, y) ↦ sin(x)² · y + sin(x) · y` saves `sin(x)` and `sin(x)²` through a
        // `NegationStorage`. The known program also squares `sin(x)`, which is therefore rounded once, right after it
        // is computed and before its store operation negates it, whatever operations the storage uses. `sin(x)²` only
        // feeds its store operation and stays as is.
        let scalar_type = ArrayIrType::from(ArrayType::scalar(DataType::BF16));
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let x = builder.add_input(scalar_type.clone());
        let y = builder.add_input(scalar_type);
        let sine = add(&mut builder, SinOperation::new().into(), vec![x]);
        let square = add(&mut builder, MulOperation::new().into(), vec![sine, sine]);
        let first = add(&mut builder, MulOperation::new().into(), vec![square, y]);
        let second = add(&mut builder, MulOperation::new().into(), vec![sine, y]);
        let sum = add(&mut builder, AddOperation::new().into(), vec![first, second]);
        let program = build(builder, vec![sum]);
        let rounding = |r#type: &ArrayIrType| {
            matches!(r#type, ArrayIrType::Array(array_type) if array_type.data_type() == DataType::BF16)
                .then_some(ErasedOperation::new(ReducePrecisionOperation::<ArrayType>::new(8, 7)))
        };
        let store_everything = policy("store_everything", |_: &ResidualCandidate<'_, ArrayIrType>| {
            Ok(ResidualDecision::SaveWith(NegationStorage))
        });
        let partition = program.partition(&[true, false]).unwrap().with_residual_policy(&store_everything).unwrap();
        assert_eq!(
            partition.with_rounded_residuals(&rounding, true).unwrap().to_string(),
            indoc! {"
                partition [
                    known_inputs=[0],
                    residual_inputs=[UnknownInput(1), ResidualEdge(0), ResidualEdge(1)],
                    outputs=[Unknown(0)],
                ]
                known={
                    lambda %0:bf16[] .
                    let %1:bf16[] = sin %0
                        %2:bf16[] = reduce_precision [exponent_bits=8, mantissa_bits=7] %1
                        %3:bf16[] = mul %2 %2
                        %4:bf16[] = neg %3
                        %5:bf16[] = neg %2
                    in (%4, %5)
                }
                residual={
                    lambda %0:bf16[], %1:bf16[], %2:bf16[] .
                    let %3:bf16[] = neg %1
                        %4:bf16[] = mul %3 %0
                        %5:bf16[] = neg %2
                        %6:bf16[] = mul %5 %0
                        %7:bf16[] = add %4 %6
                    in (%7)
                }"},
        );
    }

    #[test]
    fn test_residual_policy_reference_place_residuals() {
        // Each demanded condition output preserves its own nested cuts: the second stays saved, while the first
        // recomputes only its sine and tag. A saved sibling does not force the first output across the boundary.
        let program = condition_program(true);
        let placed = save_names(&["second"], &[])
            .place_residuals(program.partition(&[true, true, false]).unwrap(), None)
            .unwrap();
        assert_eq!(
            placed.to_string(),
            indoc! {"
                partition [
                    known_inputs=[0, 1],
                    residual_inputs=[UnknownInput(2), ResidualEdge(0), ResidualEdge(1), ResidualEdge(2)],
                    outputs=[Unknown(0), Unknown(1)],
                ]
                known={
                    lambda %0:bool[], %1:f64[] .
                    let %2:f64[] = condition %0 %1 [
                        true={
                            lambda %0:f64[] .
                            let %1:f64[] = cos %0
                                %2:f64[] = tag [key=second] %1
                            in (%2)
                        },
                        false={
                            lambda %0:f64[] .
                            let %1:f64[] = cos %0
                                %2:f64[] = tag [key=second] %1
                            in (%2)
                        },
                    ]
                    in (%2, %0, %1)
                }
                residual={
                    lambda %0:f64[], %1:f64[], %2:bool[], %3:f64[] .
                    let %4:f64[] = condition %2 %3 [
                        true={
                            lambda %0:f64[] .
                            let %1:f64[] = sin %0
                                %2:f64[] = tag [key=first] %1
                            in (%2)
                        },
                        false={
                            lambda %0:f64[] .
                            let %1:f64[] = sin %0
                                %2:f64[] = tag [key=first] %1
                            in (%2)
                        },
                    ]
                        %5:f64[] = mul %4 %0
                        %6:f64[] = mul %1 %0
                    in (%5, %6)
                }"},
        );
        let inputs = vec![
            ArrayIrValue::Array(Array::scalar(true).unwrap()),
            ArrayIrValue::Array(Array::scalar(0.5f64).unwrap()),
            ArrayIrValue::Array(Array::scalar(2.0f64).unwrap()),
        ];
        assert_eq!(run(&placed, &inputs), program.interpret(inputs.clone()).unwrap());

        // A policy that recomputes every value that the branches compute replays the condition as a whole instead.
        let placed = save_nothing().place_residuals(program.partition(&[true, true, false]).unwrap(), None).unwrap();
        assert_eq!(
            placed.to_string(),
            indoc! {"
                partition [
                    known_inputs=[0, 1],
                    residual_inputs=[UnknownInput(2), ResidualEdge(0), ResidualEdge(1)],
                    outputs=[Unknown(0), Unknown(1)],
                ]
                known={
                    lambda %0:bool[], %1:f64[] .
                    in (%0, %1)
                }
                residual={
                    lambda %0:f64[], %1:bool[], %2:f64[] .
                    let %3:f64[], %4:f64[] = condition %1 %2 [
                        true={
                            lambda %0:f64[] .
                            let %1:f64[] = sin %0
                                %2:f64[] = tag [key=first] %1
                                %3:f64[] = cos %0
                                %4:f64[] = tag [key=second] %3
                            in (%2, %4)
                        },
                        false={
                            lambda %0:f64[] .
                            let %1:f64[] = sin %0
                                %2:f64[] = tag [key=first] %1
                                %3:f64[] = cos %0
                                %4:f64[] = tag [key=second] %3
                            in (%2, %4)
                        },
                    ]
                        %5:f64[] = mul %3 %0
                        %6:f64[] = mul %4 %0
                    in (%5, %6)
                }"},
        );
        assert_eq!(run(&placed, &inputs), program.interpret(inputs.clone()).unwrap());
    }

    #[test]
    fn test_residual_policy_reference_place_residuals_propagates_rejection() {
        let program = condition_program(true).with_outputs(&[0]).unwrap();
        let reject =
            policy("reject_first", |candidate| {
                if candidate.producers().iter().any(|producer| {
                    producer.payload::<TagOperation<ArrayType>>().is_some_and(|tag| tag.key() == "first")
                }) {
                    Err(ResidualRejection::new("the first result cannot cross this boundary"))
                } else {
                    Ok(ResidualDecision::<NoStorage>::Recompute)
                }
            });
        let expected = ResidualPolicyError::Rejected {
            policy: "reject_first".to_owned(),
            rejection: ResidualRejection::new("the first result cannot cross this boundary"),
        };
        assert_eq!(
            program.partition(&[true, true, false]).unwrap().with_residual_policy(&reject).map(|_| ()),
            Err(expected.clone()),
        );
        assert_eq!(
            program
                .partition_with_residual_policy(&[true, true, false], &reject)
                .map(|_| ())
                .map_err(ResidualPolicyError::from),
            Err(expected),
        );
    }

    #[test]
    fn test_residual_policy_reference_place_residuals_rejects_after_saved_result() {
        let program = condition_program(true);
        let reject =
            policy("save_first_reject_second", |candidate| {
                if candidate.producers().iter().any(|producer| {
                    producer.payload::<TagOperation<ArrayType>>().is_some_and(|tag| tag.key() == "second")
                }) {
                    Err(ResidualRejection::new("the second result cannot cross this boundary"))
                } else {
                    Ok(ResidualDecision::<NoStorage>::Save)
                }
            });
        assert_eq!(
            program
                .partition_with_residual_policy(&[true, true, false], &reject)
                .map(|_| ())
                .map_err(ResidualPolicyError::from),
            Err(ResidualPolicyError::Rejected {
                policy: "save_first_reject_second".to_owned(),
                rejection: ResidualRejection::new("the second result cannot cross this boundary"),
            }),
        );
    }

    #[test]
    fn test_residual_policy_reference_place_residuals_rejects_after_saved_interior_producer() {
        // The demanded condition output is recomputed, so replay checks the live producers inside the branches. The
        // sine is saved and the later cosine is rejected: a saved interior producer must not hide that rejection.
        let mut branch = ProgramBuilder::<TestValue, TestOperation>::new();
        let input = branch.add_input(scalar_type());
        let sine = add(&mut branch, SinOperation::<ArrayType>::new().into(), vec![input]);
        let cosine = add(&mut branch, CosOperation::<ArrayType>::new().into(), vec![sine]);
        let terminal = add(&mut branch, TagOperation::<ArrayType>::new("terminal").into(), vec![cosine]);
        let branch = build(branch, vec![terminal]);
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let input = builder.add_input(scalar_type());
        let unknown = builder.add_input(scalar_type());
        let branch = builder.import_program(branch);
        let output = builder
            .add_instruction(ConditionOperation::new(), vec![branch, branch], vec![predicate, input], None)
            .unwrap()[0];
        let output = add(&mut builder, MulOperation::<ArrayType>::new().into(), vec![output, unknown]);
        let program = build(builder, vec![output]);
        let policy = policy("save_sine_reject_cosine", |candidate| {
            if candidate.producers().iter().any(|producer| producer.payload::<CosOperation<ArrayType>>().is_some()) {
                Err(ResidualRejection::new("the cosine cannot cross this boundary"))
            } else if candidate
                .producers()
                .iter()
                .any(|producer| producer.payload::<SinOperation<ArrayType>>().is_some())
            {
                Ok(ResidualDecision::<NoStorage>::Save)
            } else {
                Ok(ResidualDecision::Recompute)
            }
        });
        assert_eq!(
            program
                .partition_with_residual_policy(&[true, true, false], &policy)
                .map(|_| ())
                .map_err(ResidualPolicyError::from),
            Err(ResidualPolicyError::Rejected {
                policy: "save_sine_reject_cosine".to_owned(),
                rejection: ResidualRejection::new("the cosine cannot cross this boundary"),
            }),
        );
    }

    #[test]
    fn test_residual_policy_reference_place_residuals_ignores_unused_result() {
        // The second branch output is primal work only. Recomputing the first output must neither classify it nor
        // retain its cosine and tag in the replayed branches.
        let program = condition_program(true).with_outputs(&[0]).unwrap();
        let reject =
            policy("reject_unused_second", |candidate| {
                if candidate.producers().iter().any(|producer| {
                    producer.payload::<TagOperation<ArrayType>>().is_some_and(|tag| tag.key() == "second")
                }) {
                    Err(ResidualRejection::new("the second result cannot cross this boundary"))
                } else {
                    Ok(ResidualDecision::<NoStorage>::Recompute)
                }
            });
        let placed = program.partition_with_residual_policy(&[true, true, false], &reject).unwrap();
        assert_eq!(
            placed.to_string(),
            indoc! {"
                partition [
                    known_inputs=[0, 1],
                    residual_inputs=[UnknownInput(2), ResidualEdge(0), ResidualEdge(1)],
                    outputs=[Unknown(0)],
                ]
                known={
                    lambda %0:bool[], %1:f64[] .
                    in (%0, %1)
                }
                residual={
                    lambda %0:f64[], %1:bool[], %2:f64[] .
                    let %3:f64[] = condition %1 %2 [
                        true={
                            lambda %0:f64[] .
                            let %1:f64[] = sin %0
                                %2:f64[] = tag [key=first] %1
                            in (%2)
                        },
                        false={
                            lambda %0:f64[] .
                            let %1:f64[] = sin %0
                                %2:f64[] = tag [key=first] %1
                            in (%2)
                        },
                    ]
                        %4:f64[] = mul %3 %0
                    in (%4)
                }"},
        );
        let inputs = vec![
            ArrayIrValue::Array(Array::scalar(true).unwrap()),
            ArrayIrValue::Array(Array::scalar(0.5f64).unwrap()),
            ArrayIrValue::Array(Array::scalar(2.0f64).unwrap()),
        ];
        assert_eq!(run(&placed, &inputs), program.interpret(inputs.clone()).unwrap());
    }

    #[test]
    fn test_residual_policy_reference_place_residuals_ignores_unused_nested_result() {
        // The `linear_call` inside the branches keeps its unused tagged output, because its boundary cannot be pruned.
        // Only the sine-scaled input feeds the demanded output, so replay must not classify the unused cosine-scaled
        // output and reject it.
        let program = linear_call_condition_program();
        let placed = program
            .partition_with_residual_policy(&[true, true, false], &reject_name("reject_unused", "unused"))
            .unwrap();
        check_linear_call_condition_partition(&placed);
        let inputs = vec![
            ArrayIrValue::Array(Array::scalar(true).unwrap()),
            ArrayIrValue::Array(Array::scalar(0.5f64).unwrap()),
            ArrayIrValue::Array(Array::scalar(2.0f64).unwrap()),
        ];
        assert_eq!(run(&placed, &inputs), program.interpret(inputs.clone()).unwrap());
    }

    #[test]
    fn test_residual_policy_reference_place_residuals_ignores_dormant_derivative_regions() {
        // Replaying the branches executes the forward region of their `linear_call` but never its transpose rule, so
        // the rejected producer in that rule region is not residual demand.
        let program = linear_call_condition_program();
        let placed = program
            .partition_with_residual_policy(&[true, true, false], &reject_name("reject_rule", "rule"))
            .unwrap();
        check_linear_call_condition_partition(&placed);
        let inputs = vec![
            ArrayIrValue::Array(Array::scalar(false).unwrap()),
            ArrayIrValue::Array(Array::scalar(0.5f64).unwrap()),
            ArrayIrValue::Array(Array::scalar(2.0f64).unwrap()),
        ];
        assert_eq!(run(&placed, &inputs), program.interpret(inputs.clone()).unwrap());
    }

    #[test]
    fn test_residual_policy_reference_place_residuals_ignores_dormant_rules_of_opaque_regions() {
        // A nested region carrier that declares no output provenance conservatively demands every output of its
        // computation regions, but still never its dormant rule regions, so the rejected `add` in the rule region of
        // the nested carrier is not classified.
        /// Region-machinery fixture that optionally declares that its outputs are those of its first region.
        #[derive(Clone)]
        struct Carrier {
            /// Fixture with the ordinary region contract.
            operation: TestRegionOperation,

            /// Whether each output is the corresponding output of the first region.
            provenance: bool,
        }

        impl Operation for Carrier {
            type Type = ArrayType;

            fn name(&self) -> &'static str {
                self.operation.name()
            }

            fn region_slots(&self) -> &'static [RegionSlot] {
                self.operation.region_slots()
            }

            fn infer_output_types(
                &self,
                input_types: &[ArrayType],
                region_interfaces: &[RegionInterface<ArrayType>],
            ) -> Result<Vec<ArrayType>, TypeError> {
                self.operation.infer_output_types(input_types, region_interfaces)
            }

            fn input_region_provenance(&self, _region_index: usize, input_index: usize) -> InputRegionProvenance {
                InputRegionProvenance::Input { index: input_index }
            }

            fn output_region_provenance(&self, output_index: usize) -> Vec<OutputRegionProvenance> {
                if self.provenance {
                    vec![OutputRegionProvenance { region_index: 0, output_index }]
                } else {
                    Vec::new()
                }
            }

            fn region_data_flow(&self) -> RegionDataFlow<'_> {
                if self.provenance { RegionDataFlow::Provenance } else { RegionDataFlow::Opaque }
            }
        }

        impl OperationPayloadProjection for Carrier {
            fn project_payload(&self, _payload: TypeId) -> Option<&dyn Any> {
                None
            }

            fn from_payload(payload: ErasedOperation) -> Result<Self, ErasedOperation> {
                Err(payload)
            }
        }

        let scalar_type = ArrayType::scalar(DataType::F64);
        let region = |output: fn(&mut ProgramBuilder<Array, Carrier>, AtomId) -> AtomId| {
            let mut builder = ProgramBuilder::<Array, Carrier>::new();
            let input = builder.add_input(scalar_type.clone());
            let output = output(&mut builder, input);
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
        };
        let computation = region(|_, input| input);
        let rule = region(|builder, input| {
            let operation = Carrier { operation: TestRegionOperation::Add, provenance: false };
            builder.add_instruction(operation, Vec::new(), vec![input, input], None).unwrap()[0]
        });
        let mut body = ProgramBuilder::<Array, Carrier>::new();
        let input = body.add_input(scalar_type.clone());
        let computation = body.import_program(computation);
        let rule = body.import_program(rule);
        let slots = const { &[RegionSlot::computation("body"), RegionSlot::rule("rule")] };
        let nested = Carrier { operation: TestRegionOperation::WithRegions(slots), provenance: false };
        let output = body.add_instruction(nested, vec![computation, rule], vec![input], None).unwrap()[0];
        let body = body.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        let mut known = ProgramBuilder::<Array, Carrier>::new();
        let input = known.add_input(scalar_type.clone());
        let body = known.import_program(body);
        let slots = const { &[RegionSlot::computation("body")] };
        let carrier = Carrier { operation: TestRegionOperation::WithRegions(slots), provenance: true };
        let output = known.add_instruction(carrier, vec![body], vec![input], None).unwrap()[0];
        let known = known.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        let mut residual = ProgramBuilder::<Array, Carrier>::new();
        residual.add_input(scalar_type.clone());
        let edge = residual.add_input(scalar_type);
        let residual = residual
            .build::<Vec<Array>, Vec<Array>>(vec![edge], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let partition = PartitionedProgram::from_parts(
            known,
            residual,
            2,
            vec![0],
            vec![ResidualInputSource::UnknownInput(1), ResidualInputSource::ResidualEdge(0)],
            vec![PartialEvaluationOutput::Unknown(0)],
        )
        .unwrap();
        let policy = ResidualPolicyReference::new(TestPolicy {
            name: "reject_add",
            classify: |candidate: &ResidualCandidate<'_, ArrayType>| {
                if candidate.producers().iter().any(|producer| producer.name() == "add") {
                    Err(ResidualRejection::new("the `add` result cannot cross this boundary"))
                } else {
                    Ok(ResidualDecision::<NoStorage>::Recompute)
                }
            },
        });

        // The carrier is replayed by the residual program, so the known program computes nothing.
        let placed = policy.place_residuals(partition, None).unwrap();
        assert_eq!(
            placed.to_string(),
            indoc! {"
                partition [
                    known_inputs=[0],
                    residual_inputs=[UnknownInput(1), ResidualEdge(0)],
                    outputs=[Unknown(0)],
                ]
                known={
                    lambda %0:f64[] .
                    in (%0)
                }
                residual={
                    lambda %0:f64[], %1:f64[] .
                    let %2:f64[] = with_regions %1 [
                        body={
                            lambda %0:f64[] .
                            let %1:f64[] = with_regions %0 [
                                body={
                                    lambda %0:f64[] .
                                    in (%0)
                                },
                                rule={
                                    lambda %0:f64[] .
                                    let %1:f64[] = add %0 %0
                                    in (%1)
                                },
                            ]
                            in (%1)
                        },
                    ]
                    in (%2)
                }"},
        );
    }

    #[test]
    fn test_residual_policy_reference_place_residuals_memoizes_shared_regions() {
        // Every condition attaches the same child twice. Source size grows with depth, while enumerating attachment
        // paths would classify the one sine 2^depth times. This exercises the replay guard, not just provenance.
        let depth = 12;
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let input = builder.add_input(scalar_type());
        let sine = add(&mut builder, SinOperation::<ArrayType>::new().into(), vec![input]);
        let mut body = build(builder, vec![sine]);
        for _ in 0..depth {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
            let input = builder.add_input(scalar_type());
            let child = builder.import_program(body);
            let output = builder
                .add_instruction(ConditionOperation::new(), vec![child, child], vec![predicate, predicate, input], None)
                .unwrap()[0];
            body = build(builder, vec![output]);
        }
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let input = builder.add_input(scalar_type());
        let unknown = builder.add_input(scalar_type());
        let output = builder.splice_program(&body, &[predicate, input]).unwrap()[0];
        let output = add(&mut builder, MulOperation::<ArrayType>::new().into(), vec![output, unknown]);
        let program = build(builder, vec![output]);
        assert_eq!(program.entry_region_ref().instructions_in_closure().count(), depth + 2);

        let classifications = Arc::new(AtomicUsize::new(0));
        let counter = classifications.clone();
        let policy = policy("count_sines", move |candidate| {
            if candidate.producers().iter().any(|producer| producer.payload::<SinOperation<ArrayType>>().is_some()) {
                counter.fetch_add(1, Ordering::Relaxed);
            }
            Ok(ResidualDecision::<NoStorage>::Recompute)
        });
        let placed = program.partition_with_residual_policy(&[true, true, false], &policy).unwrap();
        let classifications = classifications.load(Ordering::Relaxed);
        // One outer candidate and the two known-side branch origins each classify the shared sine once.
        assert_eq!(classifications, 3);
        let inputs = vec![
            ArrayIrValue::Array(Array::scalar(true).unwrap()),
            ArrayIrValue::Array(Array::scalar(0.5f64).unwrap()),
            ArrayIrValue::Array(Array::scalar(2f64).unwrap()),
        ];
        assert_eq!(run(&placed, &inputs), vec![ArrayIrValue::Array(Array::scalar(2f64 * 0.5f64.sin()).unwrap())]);
    }

    #[test]
    fn test_residual_policy_reference_place_residuals_distinguishes_shared_region_output_demands() {
        // Two `linear_call`s, whose coupled boundaries stay intact, share a forward body that computes
        // `(r, u) ↦ (sin(r) · u, tag(cos(r) · u, "second"))` with `r = u`, but consume different outputs. Their
        // distinct inputs prevent common-subexpression elimination from merging the calls. Check both orders so caching
        // only by region cannot reuse the non-rejecting output's result for the rejected output in either traversal
        // order.
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let residual = builder.add_input(scalar_type());
        let linear = builder.add_input(scalar_type());
        let sine = add(&mut builder, SinOperation::<ArrayType>::new().into(), vec![residual]);
        let cosine = add(&mut builder, CosOperation::<ArrayType>::new().into(), vec![residual]);
        let first = add(&mut builder, MulOperation::<ArrayType>::new().into(), vec![sine, linear]);
        let second = add(&mut builder, MulOperation::<ArrayType>::new().into(), vec![cosine, linear]);
        let second = add(&mut builder, TagOperation::<ArrayType>::new("second").into(), vec![second]);
        let body = build(builder, vec![first, second]);
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let residual = builder.add_input(scalar_type());
        let first = builder.add_input(scalar_type());
        let second = builder.add_input(scalar_type());
        let sine = add(&mut builder, SinOperation::<ArrayType>::new().into(), vec![residual]);
        let cosine = add(&mut builder, CosOperation::<ArrayType>::new().into(), vec![residual]);
        let first = add(&mut builder, MulOperation::<ArrayType>::new().into(), vec![sine, first]);
        let second = add(&mut builder, MulOperation::<ArrayType>::new().into(), vec![cosine, second]);
        let cotangent = add(&mut builder, AddOperation::<ArrayType>::new().into(), vec![first, second]);
        let transpose = build(builder, vec![cotangent]);
        for selected_outputs in [[0, 1], [1, 0]] {
            let mut branch = ProgramBuilder::<TestValue, TestOperation>::new();
            let inputs = [branch.add_input(scalar_type()), branch.add_input(scalar_type())];
            let body = branch.import_program(body.clone());
            let transpose = branch.import_program(transpose.clone());
            let mut outputs = Vec::new();
            for (input, output_index) in inputs.into_iter().zip(selected_outputs) {
                let call = ArrayOperation::<Array>::LinearCall(LinearCallOperation::new(1));
                outputs.push(
                    branch.add_instruction(call, vec![body, transpose], vec![input, input], None).unwrap()
                        [output_index],
                );
            }
            let sum = add(&mut branch, AddOperation::<ArrayType>::new().into(), outputs);
            let branch = build(branch, vec![sum]);
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
            let first = builder.add_input(scalar_type());
            let second = builder.add_input(scalar_type());
            let unknown = builder.add_input(scalar_type());
            let branch = builder.import_program(branch);
            let output = builder
                .add_instruction(ConditionOperation::new(), vec![branch, branch], vec![predicate, first, second], None)
                .unwrap()[0];
            let output = add(&mut builder, MulOperation::<ArrayType>::new().into(), vec![output, unknown]);
            let program = build(builder, vec![output]);
            assert_eq!(
                program
                    .partition_with_residual_policy(
                        &[true, true, true, false],
                        &reject_name("reject_second", "second"),
                    )
                    .map(|_| ())
                    .map_err(ResidualPolicyError::from),
                Err(ResidualPolicyError::Rejected {
                    policy: "reject_second".to_owned(),
                    rejection: ResidualRejection::new("the `second` result cannot cross this boundary"),
                }),
            );
        }
    }

    #[test]
    fn test_residual_policy_reference_place_residuals_replays_atomic_operations() {
        // A known composite call computes `tag(sin(w), "inner")` inside a known condition, whose output scales an
        // unknown input. The policy classifies the call itself rather than the tag inside it, so neither a policy that
        // saves the tag nor one that rejects it applies, and the residual program replays the whole call from `w`,
        // which also keeps the condition from being split around the tag.
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let input = builder.add_input(scalar_type());
        let sine = add(&mut builder, SinOperation::<ArrayType>::new().into(), vec![input]);
        let tagged = add(&mut builder, TagOperation::<ArrayType>::new("inner").into(), vec![sine]);
        let body = build(builder, vec![tagged]);
        let mut branch = ProgramBuilder::<TestValue, TestOperation>::new();
        let input = branch.add_input(scalar_type());
        let body = branch.import_program(body);
        let call = TestOperation::Composite(CompositeOperation::from_rule_regions(CompositeJvpRule::Derived, false));
        let output = branch.add_instruction(call, vec![body], vec![input], None).unwrap()[0];
        let branch = build(branch, vec![output]);
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let input = builder.add_input(scalar_type());
        let unknown = builder.add_input(scalar_type());
        let branch = builder.import_program(branch);
        let output = builder
            .add_instruction(ConditionOperation::new(), vec![branch, branch], vec![predicate, input], None)
            .unwrap()[0];
        let output = add(&mut builder, MulOperation::<ArrayType>::new().into(), vec![output, unknown]);
        let program = build(builder, vec![output]);
        let inputs = vec![
            ArrayIrValue::Array(Array::scalar(true).unwrap()),
            ArrayIrValue::Array(Array::scalar(0.5f64).unwrap()),
            ArrayIrValue::Array(Array::scalar(2f64).unwrap()),
        ];
        for policy in [save_names(&["inner"], &[]), reject_name("reject_inner", "inner")] {
            let placed = program.partition_with_residual_policy(&[true, true, false], &policy).unwrap();
            assert_eq!(
                placed.to_string(),
                indoc! {"
                    partition [
                        known_inputs=[0, 1],
                        residual_inputs=[UnknownInput(2), ResidualEdge(0), ResidualEdge(1)],
                        outputs=[Unknown(0)],
                    ]
                    known={
                        lambda %0:bool[], %1:f64[] .
                        in (%0, %1)
                    }
                    residual={
                        lambda %0:f64[], %1:bool[], %2:f64[] .
                        let %3:f64[] = condition %1 %2 [
                            true={
                                lambda %0:f64[] .
                                let %1:f64[] = composite [jvp_from_primal=true] %0 [
                                    primal={
                                        lambda %0:f64[] .
                                        let %1:f64[] = sin %0
                                            %2:f64[] = tag [key=inner] %1
                                        in (%2)
                                    },
                                ]
                                in (%1)
                            },
                            false={
                                lambda %0:f64[] .
                                let %1:f64[] = composite [jvp_from_primal=true] %0 [
                                    primal={
                                        lambda %0:f64[] .
                                        let %1:f64[] = sin %0
                                            %2:f64[] = tag [key=inner] %1
                                        in (%2)
                                    },
                                ]
                                in (%1)
                            },
                        ]
                            %4:f64[] = mul %3 %0
                        in (%4)
                    }"},
            );
            assert_eq!(run(&placed, &inputs), program.interpret(inputs.clone()).unwrap());
        }
    }

    #[test]
    fn test_residual_policy_reference_place_residuals_rejects_scan_feedback() {
        // The ordinary attachment is visited first and can replay its selected input without executing a dot. The
        // same body attached to a scan needs both carry updates before it can produce that input's later values.
        // Caching only by region and output would reuse the ordinary attachment's successful replay decision.
        let reject = policy("reject_dots", |candidate| {
            if candidate.producers().iter().any(|producer| producer.payload::<DotOperation>().is_some()) {
                Err(ResidualRejection::new("the feedback dot cannot be replayed"))
            } else {
                Ok(ResidualDecision::<NoStorage>::Recompute)
            }
        });
        assert_eq!(
            reject.place_residuals(feedback_partition(3, &[0, 1]), None).map(|_| ()),
            Err(ProgramError::from(ResidualPolicyError::Rejected {
                policy: "reject_dots".to_owned(),
                rejection: ResidualRejection::new("the feedback dot cannot be replayed"),
            })),
        );
    }

    #[test]
    fn test_residual_policy_reference_place_residuals_ignores_unused_scan_result() {
        let reject =
            policy("reject_unused", |candidate| {
                if candidate.producers().iter().any(|producer| {
                    producer.payload::<TagOperation<ArrayType>>().is_some_and(|tag| tag.key() == "unused")
                }) {
                    Err(ResidualRejection::new("the unused result is not residual demand"))
                } else {
                    Ok(ResidualDecision::<NoStorage>::Recompute)
                }
            });
        let placed = reject.place_residuals(feedback_partition(3, &[1]), None).unwrap();
        assert_eq!(
            placed.to_string(),
            indoc! {"
                partition [
                    known_inputs=[0, 1],
                    residual_inputs=[UnknownInput(2), ResidualEdge(0)],
                    outputs=[Unknown(0)],
                ]
                known={
                    lambda %0:bool[], %1:f64[] .
                    in (%1)
                }
                residual={
                    lambda %0:f64[], %1:f64[] .
                    let %2:f64[] = tag [key=initial] %1
                        %3:f64[], %4:f64[], %5:f64[3] = scan [carry_count=2, iteration_count=3, reverse=false] %2 %1 [
                            body={
                                lambda %0:i64[], %1:f64[], %2:f64[] .
                                let %3:f64[] = dot [
                                    dimensions=(lhs_contracting=[], rhs_contracting=[], lhs_batching=[], rhs_batching=[]),
                                ] %1 %1
                                    %4:f64[] = sin %3
                                in (%2, %4, %1)
                            },
                        ]
                        %6:f64[3] = mul %5 %0
                    in (%6)
                }"},
        );
        let inputs = vec![
            ArrayIrValue::Array(Array::scalar(true).unwrap()),
            ArrayIrValue::Array(Array::scalar(2f64).unwrap()),
            ArrayIrValue::Array(Array::scalar(3f64).unwrap()),
        ];
        assert_eq!(
            run(&placed, &inputs),
            vec![ArrayIrValue::Array(Array::vector(vec![6f64, 6.0, 3.0 * 4f64.sin()]).unwrap())],
        );
    }

    #[test]
    fn test_residual_policy_reference_place_residuals_ignores_empty_scan_body() {
        let reject = policy("reject_dots", |candidate| {
            if candidate.producers().iter().any(|producer| producer.payload::<DotOperation>().is_some()) {
                Err(ResidualRejection::new("the unexecuted dot is not residual demand"))
            } else {
                Ok(ResidualDecision::<NoStorage>::Recompute)
            }
        });
        let placed = reject.place_residuals(feedback_partition(0, &[2]), None).unwrap();
        assert_eq!(
            placed.to_string(),
            indoc! {"
                partition [
                    known_inputs=[0, 1],
                    residual_inputs=[UnknownInput(2), ResidualEdge(0)],
                    outputs=[Unknown(0)],
                ]
                known={
                    lambda %0:bool[], %1:f64[] .
                    in (%1)
                }
                residual={
                    lambda %0:f64[], %1:f64[] .
                    let %2:f64[] = tag [key=initial] %1
                        %3:f64[], %4:f64[] = scan [carry_count=2, iteration_count=0, reverse=false] %2 %1 [
                            body={
                                lambda %0:i64[], %1:f64[], %2:f64[] .
                                let %3:f64[] = dot [
                                    dimensions=(lhs_contracting=[], rhs_contracting=[], lhs_batching=[], rhs_batching=[]),
                                ] %1 %1
                                    %4:f64[] = sin %3
                                in (%2, %4)
                            },
                        ]
                        %5:f64[] = mul %3 %0
                    in (%5)
                }"},
        );
        let inputs = vec![
            ArrayIrValue::Array(Array::scalar(true).unwrap()),
            ArrayIrValue::Array(Array::scalar(2f64).unwrap()),
            ArrayIrValue::Array(Array::scalar(3f64).unwrap()),
        ];
        assert_eq!(run(&placed, &inputs), vec![ArrayIrValue::Array(Array::scalar(6f64).unwrap())]);
    }

    #[test]
    fn test_residual_policy_reference_place_residuals_ignores_nested_empty_scan_body() {
        let reject = policy("reject_dots", |candidate| {
            if candidate.producers().iter().any(|producer| producer.payload::<DotOperation>().is_some()) {
                Err(ResidualRejection::new("the executed dot cannot be replayed"))
            } else {
                Ok(ResidualDecision::<NoStorage>::Recompute)
            }
        });
        let program = nested_scan_program(0, false);
        let placed = reject.place_residuals(program.partition(&[true, true, false]).unwrap(), None).unwrap();
        assert_eq!(
            placed.to_string(),
            indoc! {"
                partition [
                    known_inputs=[0, 1],
                    residual_inputs=[UnknownInput(2), ResidualEdge(0), ResidualEdge(1)],
                    outputs=[Unknown(0)],
                ]
                known={
                    lambda %0:bool[], %1:f64[] .
                    in (%0, %1)
                }
                residual={
                    lambda %0:f64[], %1:bool[], %2:f64[] .
                    let %3:f64[] = condition %1 %2 [
                        true={
                            lambda %0:f64[] .
                            let %1:f64[] = scan [carry_count=1, iteration_count=0, reverse=false] %0 [
                                body={
                                    lambda %0:i64[], %1:f64[] .
                                    let %2:f64[] = dot [
                                        dimensions=(lhs_contracting=[], rhs_contracting=[], lhs_batching=[], rhs_batching=[]),
                                    ] %1 %1
                                    in (%2)
                                },
                            ]
                                %2:f64[] = sin %1
                            in (%2)
                        },
                        false={
                            lambda %0:f64[] .
                            let %1:f64[] = scan [carry_count=1, iteration_count=0, reverse=false] %0 [
                                body={
                                    lambda %0:i64[], %1:f64[] .
                                    let %2:f64[] = dot [
                                        dimensions=(lhs_contracting=[], rhs_contracting=[], lhs_batching=[], rhs_batching=[]),
                                    ] %1 %1
                                    in (%2)
                                },
                            ]
                                %2:f64[] = sin %1
                            in (%2)
                        },
                    ]
                        %4:f64[] = mul %3 %0
                    in (%4)
                }"},
        );
        for predicate in [true, false] {
            let inputs = vec![
                ArrayIrValue::Array(Array::scalar(predicate).unwrap()),
                ArrayIrValue::Array(Array::scalar(0.5f64).unwrap()),
                ArrayIrValue::Array(Array::scalar(2f64).unwrap()),
            ];
            assert_eq!(run(&placed, &inputs), vec![ArrayIrValue::Array(Array::scalar(2f64 * 0.5f64.sin()).unwrap())]);
            assert_eq!(run(&placed, &inputs), program.interpret(inputs).unwrap());
        }

        // Executed attachments still demand the dot, including an ordinary call of the empty scan's shared body.
        let expected = Err(ProgramError::from(ResidualPolicyError::Rejected {
            policy: "reject_dots".to_owned(),
            rejection: ResidualRejection::new("the executed dot cannot be replayed"),
        }));
        assert_eq!(
            reject
                .place_residuals(nested_scan_program(1, false).partition(&[true, true, false]).unwrap(), None)
                .map(|_| ()),
            expected,
        );
        assert_eq!(
            reject
                .place_residuals(nested_scan_program(0, true).partition(&[true, true, false]).unwrap(), None)
                .map(|_| ()),
            expected,
        );
    }

    #[test]
    fn test_residual_policy_reference_place_residuals_rejects_while_condition() {
        let mut condition = ProgramBuilder::<TestValue, TestOperation>::new();
        let input = condition.add_input(scalar_type());
        let product = add(
            &mut condition,
            DotOperation::new(DotDimensionNumbers::new(vec![], vec![], vec![], vec![])).into(),
            vec![input, input],
        );
        let predicate =
            add(&mut condition, CompareOperation::new(ComparisonDirection::Equal).into(), vec![product, product]);
        let condition = build(condition, vec![predicate]);
        let mut body = ProgramBuilder::<TestValue, TestOperation>::new();
        let input = body.add_input(scalar_type());
        let output = add(&mut body, SinOperation::<ArrayType>::new().into(), vec![input]);
        let body = build(body, vec![output]);
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let input = builder.add_input(scalar_type());
        let unknown = builder.add_input(scalar_type());
        let condition = builder.import_program(condition);
        let body = builder.import_program(body);
        let output = builder
            .add_instruction(
                WhileOperation::<ArrayIrType>::new().with_iteration_bound(2).unwrap(),
                vec![condition, body],
                vec![input],
                None,
            )
            .unwrap()[0];
        let output = add(&mut builder, MulOperation::<ArrayType>::new().into(), vec![output, unknown]);
        let program = build(builder, vec![output]);
        let reject = policy("reject_dots", |candidate| {
            if candidate.producers().iter().any(|producer| producer.payload::<DotOperation>().is_some()) {
                Err(ResidualRejection::new("the predicate dot cannot be replayed"))
            } else {
                Ok(ResidualDecision::<NoStorage>::Recompute)
            }
        });
        assert_eq!(
            reject.place_residuals(program.partition(&[true, false]).unwrap(), None).map(|_| ()),
            Err(ProgramError::from(ResidualPolicyError::Rejected {
                policy: "reject_dots".to_owned(),
                rejection: ResidualRejection::new("the predicate dot cannot be replayed"),
            })),
        );
    }

    #[test]
    fn test_residual_policy_reference_plan_recomputation() {
        // The known program computes `d₁ = dot(h, x)`, which is already saved, and `e = sin(d₁) · dot(x, x)`, which
        // residual work additionally recomputes. The saved `d₁` is available without consulting the policy, so even a
        // policy that recomputes everything reads it, while the policy decides how to obtain the second dot product.
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let h = builder.add_input(vector_type());
        let x = builder.add_input(vector_type());
        let first = add(&mut builder, dot(), vec![h, x]);
        let sine = add(&mut builder, SinOperation::<ArrayType>::new().into(), vec![first]);
        let second = add(&mut builder, dot(), vec![x, x]);
        let product = add(&mut builder, MulOperation::<ArrayType>::new().into(), vec![sine, second]);
        let program = build(builder, vec![product, first]);

        // Saving dot products makes the second dot product an input of the recomputation.
        let recomputation = ResidualPlacement::<TestValue, TestOperation>::plan_recomputation(
            &save_dots(None),
            &program,
            &[first],
            &[product],
            None,
        )
        .unwrap();
        assert_eq!(recomputation.inputs, vec![first, second]);
        assert_eq!(
            recomputation.program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = sin %0
                    %3:f64[] = mul %2 %1
                in (%3)"},
        );

        // Recomputing everything replays the second dot product from `x`, but still reads the saved `d₁`.
        let recomputation = ResidualPlacement::<TestValue, TestOperation>::plan_recomputation(
            &save_nothing(),
            &program,
            &[first],
            &[product],
            None,
        )
        .unwrap();
        assert_eq!(recomputation.inputs, vec![x, first]);
        assert_eq!(
            recomputation.program.to_string(),
            indoc! {"
                lambda %0:f64[3], %1:f64[] .
                let %2:f64[] = sin %1
                    %3:f64[] = dot [
                        dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                    ] %0 %0
                    %4:f64[] = mul %2 %3
                in (%4)"},
        );
        let inputs = vec![
            ArrayIrValue::Array(Array::vector(vec![1.0f64, 2.0, 3.0]).unwrap()),
            ArrayIrValue::Array(Array::vector(vec![0.5f64, -1.0, 2.0]).unwrap()),
        ];
        let outputs = program.interpret(inputs.clone()).unwrap();
        assert_eq!(
            recomputation.program.interpret(vec![inputs[1].clone(), outputs[1].clone()]),
            Ok(vec![outputs[0].clone()]),
        );

        // A newly saved value is stored by the known program and restored once by the recomputation. The already
        // saved first dot product keeps its representation, so adding a stored input does not store existing edges.
        let recomputation = ResidualPlacement::<TestValue, TestOperation>::plan_recomputation(
            &save_dots(Some(NegationStorage)),
            &program,
            &[first],
            &[product],
            None,
        )
        .unwrap();
        let known = recomputation.known_program.unwrap();
        assert_eq!(
            format!("{known}\n{}", recomputation.program),
            indoc! {"
                lambda %0:f64[3], %1:f64[3] .
                let %2:f64[] = dot [
                    dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                ] %0 %1
                    %3:f64[] = sin %2
                    %4:f64[] = dot [
                        dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                    ] %1 %1
                    %5:f64[] = neg %4
                    %6:f64[] = mul %3 %4
                in (%6, %2, %5)
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = sin %0
                    %3:f64[] = neg %1
                    %4:f64[] = mul %2 %3
                in (%4)"},
        );
        assert_eq!(recomputation.inputs, vec![known.output_ids()[1], known.output_ids()[2]]);
        let known_outputs = known.interpret(inputs).unwrap();
        assert_eq!(
            recomputation.program.interpret(vec![known_outputs[1].clone(), known_outputs[2].clone()]),
            Ok(vec![known_outputs[0].clone()]),
        );
    }

    #[test]
    fn test_residual_policy_reference_plan_recomputation_splits_region_instructions() {
        // The known program computes `s = condition(p, x)`, whose branches return `sin(dot(x, x))`. Recomputing `s`
        // would replay the dot product that the policy saves, so without splits the recomputation receives `s` itself.
        // With splits, the condition is split on residual demand: the known program returns the dot product of the
        // branch that runs after its original outputs, and the recomputation replays the condition over `p` from it.
        let mut branch = ProgramBuilder::<TestValue, TestOperation>::new();
        let x = branch.add_input(vector_type());
        let product = add(&mut branch, dot(), vec![x, x]);
        let sine = add(&mut branch, SinOperation::<ArrayType>::new().into(), vec![product]);
        let branch = build(branch, vec![sine]);
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let p = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let x = builder.add_input(vector_type());
        let branches = vec![builder.import_program(branch.clone()), builder.import_program(branch)];
        let s = builder.add_instruction(ConditionOperation::new(), branches, vec![p, x], None).unwrap()[0];
        let program = build(builder, vec![s]);

        let policy = save_dots(None);
        let recomputation =
            ResidualPlacement::<TestValue, TestOperation>::plan_recomputation(&policy, &program, &[], &[s], None)
                .unwrap();
        assert!(recomputation.known_program.is_none());
        assert_eq!(recomputation.inputs, vec![s]);

        let placement: Rc<dyn ResidualPlacement<TestValue, TestOperation>> = Rc::new(policy.clone());
        let split = |region: RegionRef<'_, TestValue, TestOperation>, index: usize, demanded_outputs: &[bool]| {
            region.split_on_residual_demand(index, demanded_outputs, placement.clone())
        };
        let recomputation = ResidualPlacement::<TestValue, TestOperation>::plan_recomputation(
            &policy,
            &program,
            &[],
            &[s],
            Some(&split),
        )
        .unwrap();
        let known_program = recomputation.known_program.unwrap();
        assert_eq!(
            format!("{known_program}\n{}", recomputation.program),
            indoc! {"
                lambda %0:bool[], %1:f64[3] .
                let %2:f64[], %3:f64[] = condition %0 %1 [
                    true={
                        lambda %0:f64[3] .
                        let %1:f64[] = dot [
                            dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                        ] %0 %0
                            %2:f64[] = sin %1
                        in (%2, %1)
                    },
                    false={
                        lambda %0:f64[3] .
                        let %1:f64[] = dot [
                            dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                        ] %0 %0
                            %2:f64[] = sin %1
                        in (%2, %1)
                    },
                ]
                in (%2, %3)
                lambda %0:bool[], %1:f64[] .
                let %2:f64[] = condition %0 %1 [
                    true={
                        lambda %0:f64[] .
                        let %1:f64[] = sin %0
                        in (%1)
                    },
                    false={
                        lambda %0:f64[] .
                        let %1:f64[] = sin %0
                        in (%1)
                    },
                ]
                in (%2)"},
        );
        assert_eq!(recomputation.inputs, vec![known_program.input_ids()[0], known_program.output_ids()[1]]);

        // Running the known program and then the recomputation reproduces `s` for both predicates.
        let x = ArrayIrValue::Array(Array::vector(vec![1.0f64, 2.0, 3.0]).unwrap());
        for predicate in [true, false] {
            let inputs = vec![ArrayIrValue::Array(Array::scalar(predicate).unwrap()), x.clone()];
            let known_outputs = known_program.interpret(inputs.clone()).unwrap();
            let recomputation_inputs = recomputation
                .inputs
                .iter()
                .map(|input| {
                    let position = known_program.output_ids().iter().position(|output| output == input);
                    match position {
                        Some(position) => known_outputs[position].clone(),
                        None => inputs[known_program.input_ids().iter().position(|i| i == input).unwrap()].clone(),
                    }
                })
                .collect();
            assert_eq!(recomputation.program.interpret(recomputation_inputs), program.interpret(inputs));
        }
    }

    #[test]
    fn test_reusable_store() {
        /// Two-step encoding whose operations must be recognized at their respective positions.
        #[derive(Debug)]
        struct TaggedNegationStorage;

        impl ResidualStorage<ArrayIrType> for TaggedNegationStorage {
            fn name(&self) -> String {
                "tagged_negation".to_owned()
            }

            fn store_payloads(&self, _: &ArrayIrType) -> Result<Vec<ErasedOperation>, ResidualPolicyError> {
                Ok(vec![
                    ErasedOperation::new(NegOperation::<ArrayType>::new()),
                    ErasedOperation::new(TagOperation::<ArrayType>::new("encoded")),
                ])
            }

            fn restore_payloads(
                &self,
                _: &ArrayIrType,
                _: &ArrayIrType,
            ) -> Result<Vec<ErasedOperation>, ResidualPolicyError> {
                Ok(vec![ErasedOperation::new(NegOperation::<ArrayType>::new())])
            }

            fn matches_store_operation(
                &self,
                _: &ArrayIrType,
                index: usize,
                operation: &dyn OperationPayloadProjection,
            ) -> bool {
                match index {
                    0 => operation.project_payload(TypeId::of::<NegOperation<ArrayType>>()).is_some(),
                    1 => operation
                        .project_payload(TypeId::of::<TagOperation<ArrayType>>())
                        .and_then(|payload| payload.downcast_ref::<TagOperation<ArrayType>>())
                        .is_some_and(|tag| tag.key() == "encoded"),
                    _ => false,
                }
            }
        }

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let x = builder.add_input(scalar_type());
        let other = builder.add_input(scalar_type());
        let value = add(&mut builder, SinOperation::<ArrayType>::new().into(), vec![x]);
        let stored = add(&mut builder, NegOperation::<ArrayType>::new().into(), vec![value]);
        let twice = add(&mut builder, NegOperation::<ArrayType>::new().into(), vec![stored]);
        let encoded = add(&mut builder, TagOperation::<ArrayType>::new("encoded").into(), vec![stored]);
        let tagged = add(&mut builder, TagOperation::<ArrayType>::new("encoded").into(), vec![value]);
        let reversed = add(&mut builder, NegOperation::<ArrayType>::new().into(), vec![tagged]);
        let other_label = add(&mut builder, TagOperation::<ArrayType>::new("other").into(), vec![stored]);
        let program = build(builder, vec![value, stored, twice, encoded, reversed, other_label]);
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = sin %0
                    %3:f64[] = neg %2
                    %4:f64[] = neg %3
                    %5:f64[] = tag [key=encoded] %3
                    %6:f64[] = tag [key=encoded] %2
                    %7:f64[] = neg %6
                    %8:f64[] = tag [key=other] %3
                in (%2, %3, %4, %5, %7, %8)"},
        );
        let producers = program.instruction_by_output();
        assert_eq!(reusable_store(&program, value, 1, &NegationStorage, &producers), Some(stored));
        assert_eq!(reusable_store(&program, other, 1, &NegationStorage, &producers), None);

        // Equal types and a shared source do not equate a logical value, a one-operation encoding, and a two-operation
        // encoding. Reuse requires the complete requested chain and exactly the same source atom.
        let logical = program.clone().with_outputs(&[0]).unwrap();
        assert_eq!(reusable_store(&logical, value, 1, &NegationStorage, &logical.instruction_by_output()), None);
        let doubled = program.clone().with_outputs(&[2]).unwrap();
        assert_eq!(reusable_store(&doubled, value, 1, &NegationStorage, &doubled.instruction_by_output()), None);
        assert_eq!(reusable_store(&program, value, 2, &TaggedNegationStorage, &producers), Some(encoded));
        let reversed = program.clone().with_outputs(&[4]).unwrap();
        assert_eq!(
            reusable_store(&reversed, value, 2, &TaggedNegationStorage, &reversed.instruction_by_output()),
            None,
        );
        let other_label = program.clone().with_outputs(&[5]).unwrap();
        assert_eq!(
            reusable_store(&other_label, value, 2, &TaggedNegationStorage, &other_label.instruction_by_output()),
            None,
        );

        // The default storage query declines reuse even when its payload has an identical encoding.
        assert_eq!(reusable_store(&program, value, 1, &UnrecognizedNegationStorage, &producers), None);
    }

    #[test]
    fn test_rounded_residual_placement_place_residuals() {
        // With `y` unknown, `(p, x, y) ↦ condition(p, x) · y` over `bf16`, whose branches return
        // `sin(tag(x · x, "t"))`, saves `t`. Recomputing the output of the known condition would replay `t`, so the
        // condition is split on residual demand: the known condition returns `t` from the branch that runs, and the
        // residual condition recomputes the sine from it. The placement of the partition of each branch rounds `t`
        // right after its producer, because the known branch also computes the sine from `t` when its residuals are
        // placed (the enclosing placement then prunes that unused known output), so the residual condition receives
        // the rounded value.
        let bf16_type = ArrayIrType::from(ArrayType::scalar(DataType::BF16));
        let mut branch = ProgramBuilder::<TestValue, TestOperation>::new();
        let x = branch.add_input(bf16_type.clone());
        let square = add(&mut branch, MulOperation::<ArrayType>::new().into(), vec![x, x]);
        let t = add(&mut branch, TagOperation::<ArrayType>::new("t").into(), vec![square]);
        let sine = add(&mut branch, SinOperation::<ArrayType>::new().into(), vec![t]);
        let branch = build(branch, vec![sine]);
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let p = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let x = builder.add_input(bf16_type.clone());
        let y = builder.add_input(bf16_type);
        let branches = vec![builder.import_program(branch.clone()), builder.import_program(branch)];
        let s = builder.add_instruction(ConditionOperation::new(), branches, vec![p, x], None).unwrap()[0];
        let output = add(&mut builder, MulOperation::<ArrayType>::new().into(), vec![s, y]);
        let program = build(builder, vec![output]);

        let placement: Rc<dyn ResidualPlacement<TestValue, TestOperation>> =
            Rc::new(RoundedResidualPlacement::new(save_names(&["t"], &[]), Bfloat16Rounding));
        let (partition, _) = program
            .entry_region_ref()
            .partition_with_configuration(&[true, true, false], true, false, None, Some(placement))
            .unwrap();
        assert_eq!(
            partition.to_string(),
            indoc! {"
                partition [
                    known_inputs=[0, 1],
                    residual_inputs=[UnknownInput(2), ResidualEdge(0), ResidualEdge(1)],
                    outputs=[Unknown(0)],
                ]
                known={
                    lambda %0:bool[], %1:bf16[] .
                    let %2:bf16[] = condition %0 %1 [
                        true={
                            lambda %0:bf16[] .
                            let %1:bf16[] = mul %0 %0
                                %2:bf16[] = tag [key=t] %1
                                %3:bf16[] = reduce_precision [exponent_bits=8, mantissa_bits=7] %2
                            in (%3)
                        },
                        false={
                            lambda %0:bf16[] .
                            let %1:bf16[] = mul %0 %0
                                %2:bf16[] = tag [key=t] %1
                                %3:bf16[] = reduce_precision [exponent_bits=8, mantissa_bits=7] %2
                            in (%3)
                        },
                    ]
                    in (%0, %2)
                }
                residual={
                    lambda %0:bf16[], %1:bool[], %2:bf16[] .
                    let %3:bf16[] = condition %1 %2 [
                        true={
                            lambda %0:bf16[] .
                            let %1:bf16[] = sin %0
                            in (%1)
                        },
                        false={
                            lambda %0:bf16[] .
                            let %1:bf16[] = sin %0
                            in (%1)
                        },
                    ]
                        %4:bf16[] = mul %3 %0
                    in (%4)
                }"},
        );
    }

    #[test]
    fn test_rounded_residual_placement_plan_recomputation() {
        // Recomputing `s = sin(t)` for `t = tag(x · x, "t")` over `bf16` receives `t`, which the policy saves. The
        // known program also consumes `t`, so it rounds `t` right after its producer, and the recomputation receives
        // the rounded `t`.
        let bf16_type = ArrayIrType::from(ArrayType::scalar(DataType::BF16));
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let x = builder.add_input(bf16_type);
        let square = add(&mut builder, MulOperation::<ArrayType>::new().into(), vec![x, x]);
        let t = add(&mut builder, TagOperation::<ArrayType>::new("t").into(), vec![square]);
        let s = add(&mut builder, SinOperation::<ArrayType>::new().into(), vec![t]);
        let program = build(builder, vec![s]);

        let placement = RoundedResidualPlacement::new(save_names(&["t"], &[]), Bfloat16Rounding);
        let recomputation =
            ResidualPlacement::<TestValue, TestOperation>::plan_recomputation(&placement, &program, &[], &[s], None)
                .unwrap();
        let known_program = recomputation.known_program.unwrap();
        assert_eq!(
            format!("{known_program}\n{}", recomputation.program),
            indoc! {"
                lambda %0:bf16[] .
                let %1:bf16[] = mul %0 %0
                    %2:bf16[] = tag [key=t] %1
                    %3:bf16[] = reduce_precision [exponent_bits=8, mantissa_bits=7] %2
                    %4:bf16[] = sin %3
                in (%4)
                lambda %0:bf16[] .
                let %1:bf16[] = sin %0
                in (%1)"},
        );
        assert_eq!(recomputation.inputs, vec![AtomId::new(3)]);

        // Without rounding, the recomputation receives `t` itself.
        let policy = save_names(&["t"], &[]);
        let recomputation =
            ResidualPlacement::<TestValue, TestOperation>::plan_recomputation(&policy, &program, &[], &[s], None)
                .unwrap();
        assert!(recomputation.known_program.is_none());
        assert_eq!(recomputation.inputs, vec![t]);

        // A storage's encoded value is not the value to round. Round the logical value before both its known sine
        // consumer and its negation store, then restore it inside the recomputation before evaluating the sine.
        let placement = RoundedResidualPlacement::new(save_names(&[], &["t"]), Bfloat16Rounding);
        let recomputation =
            ResidualPlacement::<TestValue, TestOperation>::plan_recomputation(&placement, &program, &[], &[s], None)
                .unwrap();
        let known = recomputation.known_program.unwrap();
        assert_eq!(
            format!("{known}\n{}", recomputation.program),
            indoc! {"
                lambda %0:bf16[] .
                let %1:bf16[] = mul %0 %0
                    %2:bf16[] = tag [key=t] %1
                    %3:bf16[] = reduce_precision [exponent_bits=8, mantissa_bits=7] %2
                    %4:bf16[] = neg %3
                    %5:bf16[] = sin %3
                in (%5, %4)
                lambda %0:bf16[] .
                let %1:bf16[] = neg %0
                    %2:bf16[] = sin %1
                in (%2)"},
        );
        assert_eq!(recomputation.inputs, vec![known.output_ids()[1]]);

        // Extending that placement from its existing stored input forwards the same representation, without another
        // negation store or another rounding operation.
        let extended = ResidualPlacement::<TestValue, TestOperation>::plan_recomputation(
            &placement,
            &known,
            &recomputation.inputs,
            &[known.output_ids()[1]],
            None,
        )
        .unwrap();
        assert!(extended.known_program.is_none());
        assert_eq!(extended.inputs, recomputation.inputs);
        assert_eq!(extended.program.to_string(), "lambda %0:bf16[] .\nin (%0)");
    }

    #[test]
    fn test_rounded_residual_placement_plan_recomputation_through_shape_changing_storage() {
        // A scalar is saved as a negated length-one vector. Both operations of the store chain must be skipped when
        // locating the logical value to round, and the recomputation receives the vector before restoring the scalar.
        #[derive(Debug)]
        struct VectorNegationStorage;

        impl ResidualStorage<ArrayIrType> for VectorNegationStorage {
            fn name(&self) -> String {
                "vector_negation".to_owned()
            }

            fn store_payloads(&self, _: &ArrayIrType) -> Result<Vec<ErasedOperation>, ResidualPolicyError> {
                Ok(vec![
                    ErasedOperation::new(NegOperation::<ArrayType>::new()),
                    ErasedOperation::new(ReshapeOperation::new(
                        ArrayType::new_static(DataType::BF16, [1]).shape().clone(),
                    )),
                ])
            }

            fn restore_payloads(
                &self,
                _: &ArrayIrType,
                _: &ArrayIrType,
            ) -> Result<Vec<ErasedOperation>, ResidualPolicyError> {
                Ok(vec![
                    ErasedOperation::new(ReshapeOperation::new(ArrayType::scalar(DataType::BF16).shape().clone())),
                    ErasedOperation::new(NegOperation::<ArrayType>::new()),
                ])
            }
        }

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let x = builder.add_input(ArrayType::scalar(DataType::BF16).into());
        let square = add(&mut builder, MulOperation::<ArrayType>::new().into(), vec![x, x]);
        let t = add(&mut builder, TagOperation::<ArrayType>::new("t").into(), vec![square]);
        let s = add(&mut builder, SinOperation::<ArrayType>::new().into(), vec![t]);
        let program = build(builder, vec![s]);
        let policy = policy("store_tagged", |candidate| {
            Ok(
                if candidate.producers().iter().any(|producer| producer.payload::<TagOperation<ArrayType>>().is_some())
                {
                    ResidualDecision::SaveWith(VectorNegationStorage)
                } else {
                    ResidualDecision::Recompute
                },
            )
        });
        let placement = RoundedResidualPlacement::new(policy, Bfloat16Rounding);
        let recomputation =
            ResidualPlacement::<TestValue, TestOperation>::plan_recomputation(&placement, &program, &[], &[s], None)
                .unwrap();
        let known = recomputation.known_program.unwrap();
        assert_eq!(
            format!("{known}\n{}", recomputation.program),
            indoc! {"
                lambda %0:bf16[] .
                let %1:bf16[] = mul %0 %0
                    %2:bf16[] = tag [key=t] %1
                    %3:bf16[] = reduce_precision [exponent_bits=8, mantissa_bits=7] %2
                    %4:bf16[] = neg %3
                    %5:bf16[1] = reshape [shape=[1]] %4
                    %6:bf16[] = sin %3
                in (%6, %5)
                lambda %0:bf16[1] .
                let %1:bf16[] = reshape [shape=[]] %0
                    %2:bf16[] = neg %1
                    %3:bf16[] = sin %2
                in (%3)"},
        );
        assert_eq!(recomputation.inputs, vec![known.output_ids()[1]]);
    }

    #[test]
    fn test_residual_replay_analysis_allows_replay() {
        let mut first = ProgramBuilder::<TestValue, TestOperation>::new();
        let input = first.add_input(scalar_type());
        let output = add(&mut first, TagOperation::<ArrayType>::new("first").into(), vec![input]);
        let first = build(first, vec![output]);
        let mut second = ProgramBuilder::<TestValue, TestOperation>::new();
        let input = second.add_input(scalar_type());
        let output = add(&mut second, TagOperation::<ArrayType>::new("second").into(), vec![input]);
        let second = build(second, vec![output]);

        let mut body = ProgramBuilder::<TestValue, TestOperation>::new();
        let predicate = body.add_input(ArrayType::scalar(DataType::Boolean).into());
        let input = body.add_input(scalar_type());
        let first = body.import_program(first);
        let second = body.import_program(second);
        let output = body
            .add_instruction(ConditionOperation::new(), vec![first, second], vec![predicate, input], None)
            .unwrap()[0];
        let body = build(body, vec![output]);
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let input = builder.add_input(scalar_type());
        let body = builder.import_program(body);
        let output = builder
            .add_instruction(
                RematerializeOperation::<ArrayIrType>::new(ResidualPolicyReference::new(NothingSavable)),
                vec![body],
                vec![predicate, input],
                None,
            )
            .unwrap()[0];
        let program = build(builder, vec![output]);
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:bool[], %1:f64[] .
                let %2:f64[] = rematerialize %0 %1 [
                    body={
                        lambda %0:bool[], %1:f64[] .
                        let %2:f64[] = condition %0 %1 [
                            true={
                                lambda %0:f64[] .
                                let %1:f64[] = tag [key=first] %0
                                in (%1)
                            },
                            false={
                                lambda %0:f64[] .
                                let %1:f64[] = tag [key=second] %0
                                in (%1)
                            },
                        ]
                        in (%2)
                    },
                ]
                in (%2)"},
        );
        let save_nothing = save_nothing();
        let mut replay = ResidualReplayAnalysis::new(&program, &save_nothing);
        assert_eq!(replay.allows_replay(0, 0), Ok(true));

        // The root has one computation region; rejection order therefore comes from its nested branch dependencies.
        // Both descendants reject, and a LIFO traversal must still visit the first declared branch before the second.
        let reject = policy("reject_tags", |candidate| {
            if let Some(tag) = candidate.producers()[0].payload::<TagOperation<ArrayType>>() {
                Err(ResidualRejection::new(format!("the `{}` branch cannot be replayed", tag.key())))
            } else {
                Ok(ResidualDecision::<NoStorage>::Recompute)
            }
        });
        let mut replay = ResidualReplayAnalysis::new(&program, &reject);
        assert_eq!(
            replay.allows_replay(0, 0),
            Err(ResidualPolicyError::Rejected {
                policy: "reject_tags".to_owned(),
                rejection: ResidualRejection::new("the `first` branch cannot be replayed"),
            }),
        );
    }

    #[test]
    fn test_residual_replay_analysis_allows_replay_settles_wide_calls_with_bounds() {
        // The body returns values that alternate between dot products, which `save_dots` saves, and sines, which it
        // recomputes, like the known callee of a `jit_call` split that returns the residuals of the calls that it
        // contains. Traversing the body once per output would take time quadratic in its boundary, so the replay
        // bounds must settle every answer without traversing any region.
        let mut body = ProgramBuilder::<TestValue, TestOperation>::new();
        let input = body.add_input(vector_type());
        let outputs = (0..64)
            .map(|index| match index % 2 {
                0 => add(&mut body, dot(), vec![input, input]),
                _ => add(&mut body, SinOperation::<ArrayType>::new().into(), vec![input]),
            })
            .collect::<Vec<_>>();
        let body = build(body, outputs);
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let input = builder.add_input(vector_type());
        let body = builder.import_program(body);
        let rematerialize = RematerializeOperation::<ArrayIrType>::new(ResidualPolicyReference::new(NothingSavable));
        let outputs = builder.add_instruction(rematerialize, vec![body], vec![input], None).unwrap().to_vec();
        let program = build(builder, outputs);
        let classifications = Arc::new(AtomicUsize::new(0));
        let count = classifications.clone();
        let policy = policy("counted_save_dots", move |candidate| {
            let index = count.fetch_add(1, Ordering::Relaxed);
            let dot = candidate.producers()[0].payload::<DotOperation>().is_some();
            assert_eq!(dot, index % 2 == 0);
            Ok(if dot { ResidualDecision::<NoStorage>::Save } else { ResidualDecision::Recompute })
        });
        let mut replay = ResidualReplayAnalysis::new(&program, &policy);
        assert!(replay.bounds.is_empty());
        for output_index in 0..64 {
            assert_eq!(replay.allows_replay(0, output_index), Ok(output_index % 2 == 1));
        }
        assert!(replay.resolved.is_empty());
        assert_eq!(replay.bounds.iter().flatten().count(), 1);
        assert_eq!(classifications.load(Ordering::Relaxed), 64);
    }

    #[test]
    fn test_residual_replay_analysis_allows_replay_keeps_feedback_demands_distinct() {
        let partition = feedback_partition(3, &[0, 1]);
        let program = partition.known_program();
        let classifications = Arc::new(AtomicUsize::new(0));
        let count = classifications.clone();
        let policy = policy("counted_save_dots", move |candidate| {
            let index = count.fetch_add(1, Ordering::Relaxed);
            assert_eq!(candidate.producers()[0].name(), ["dot", "sin", "tag"][index]);
            Ok(if candidate.producers()[0].payload::<DotOperation>().is_some() {
                ResidualDecision::<NoStorage>::Save
            } else {
                ResidualDecision::Recompute
            })
        });
        let mut replay = ResidualReplayAnalysis::new(program, &policy);
        // The ordinary attachment forwards an input. The scan's history needs the shared body's feedback updates.
        assert_eq!(replay.allows_replay(1, 2), Ok(true));
        assert_eq!(classifications.load(Ordering::Relaxed), 3);
        assert_eq!(replay.allows_replay(2, 2), Ok(false));
        assert_eq!(classifications.load(Ordering::Relaxed), 3);
        assert_eq!(replay.allows_replay(1, 2), Ok(true));
        assert_eq!(replay.bounds.iter().flatten().count(), 1);
        assert_eq!(classifications.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn test_residual_replay_analysis_allows_replay_retains_deferred_inputs() {
        /// Region fixture exposing deferred work and positional call provenance to replay analysis.
        #[derive(Clone)]
        struct ReplayOperation(TestRegionOperation);

        impl Operation for ReplayOperation {
            type Type = ArrayType;

            fn name(&self) -> &'static str {
                self.0.name()
            }

            fn region_slots(&self) -> &'static [RegionSlot] {
                self.0.region_slots()
            }

            fn infer_output_types(
                &self,
                input_types: &[ArrayType],
                region_interfaces: &[RegionInterface<ArrayType>],
            ) -> Result<Vec<ArrayType>, TypeError> {
                self.0.infer_output_types(input_types, region_interfaces)
            }

            fn effects(&self) -> Cow<'_, Effects> {
                self.0.effects()
            }

            fn input_region_provenance(&self, _region_index: usize, input_index: usize) -> InputRegionProvenance {
                InputRegionProvenance::Input { index: input_index }
            }

            fn output_region_provenance(&self, output_index: usize) -> Vec<OutputRegionProvenance> {
                vec![OutputRegionProvenance { region_index: 0, output_index }]
            }

            fn region_data_flow(&self) -> RegionDataFlow<'_> {
                if self.region_slots().is_empty() { RegionDataFlow::Opaque } else { RegionDataFlow::Provenance }
            }
        }

        impl OperationPayloadProjection for ReplayOperation {
            fn project_payload(&self, _payload: TypeId) -> Option<&dyn Any> {
                None
            }

            fn from_payload(payload: ErasedOperation) -> Result<Self, ErasedOperation> {
                Err(payload)
            }
        }

        let scalar = ArrayType::scalar(DataType::F64);
        let mut body = ProgramBuilder::<Array, ReplayOperation>::new();
        let input = body.add_input(scalar.clone());
        let saved = body
            .add_instruction(ReplayOperation(TestRegionOperation::Add), Vec::new(), vec![input, input], None)
            .unwrap()[0];
        body.add_instruction(ReplayOperation(TestRegionOperation::Deferred), Vec::new(), vec![saved], None)
            .unwrap();
        let body = body.build::<Vec<Array>, Vec<Array>>(vec![input], vec![Placeholder], vec![Placeholder]).unwrap();
        let mut builder = ProgramBuilder::<Array, ReplayOperation>::new();
        let input = builder.add_input(scalar);
        let body = builder.import_program(body);
        let output = builder
            .add_instruction(ReplayOperation(TestRegionOperation::Call), vec![body], vec![input], None)
            .unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = array_identity %0 [
                    body={
                        lambda %0:f64[] .
                        let %1:f64[] = add %0 %0
                            %2:f64[] = deferred %1
                        in (%0)
                    },
                ]
                in (%1)
            "}
            .trim_end(),
        );
        let classifications = Arc::new(AtomicUsize::new(0));
        let count = classifications.clone();
        let policy = ResidualPolicyReference::new(TestPolicy {
            name: "save_add",
            classify: move |candidate: &ResidualCandidate<'_, ArrayType>| {
                let index = count.fetch_add(1, Ordering::Relaxed);
                assert_eq!(candidate.producers()[0].name(), ["add", "deferred", "add"][index]);
                Ok(if candidate.producers()[0].name() == "add" {
                    ResidualDecision::<NoStorage>::Save
                } else {
                    ResidualDecision::Recompute
                })
            },
        });
        let mut replay = ResidualReplayAnalysis::new(&program, &policy);
        // Returning the input still retains the deferred instruction and its saved input. Conservative bounds must
        // defer to exact liveness, which classifies the add but never the deferred instruction's unused output.
        assert_eq!(replay.demand_bounds(body, &[true]).answer(), None);
        assert_eq!(classifications.load(Ordering::Relaxed), 2);
        assert_eq!(replay.allows_replay(0, 0), Ok(false));
        assert_eq!(classifications.load(Ordering::Relaxed), 3);
        assert_eq!(replay.allows_replay(0, 0), Ok(false));
        assert_eq!(classifications.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn test_residual_replay_analysis_allows_replay_traverses_unsettled_answers() {
        // The body computes `d = x · x`, which `save_dots` saves, and `sin(x)`, and calls a nested rematerialization
        // with `d` and `x` that returns `cos(x)` and `d`. The bounds settle the sine and `d`, but they assume that the
        // nested call keeps all of its inputs, so only the exact traversal sees that replaying `cos(x)` prunes `d`.
        let mut inner = ProgramBuilder::<TestValue, TestOperation>::new();
        let product = inner.add_input(scalar_type());
        let input = inner.add_input(vector_type());
        let cosine = add(&mut inner, CosOperation::<ArrayType>::new().into(), vec![input]);
        let inner = build(inner, vec![cosine, product]);
        let mut body = ProgramBuilder::<TestValue, TestOperation>::new();
        let input = body.add_input(vector_type());
        let product = add(&mut body, dot(), vec![input, input]);
        let sine = add(&mut body, SinOperation::<ArrayType>::new().into(), vec![input]);
        let inner = body.import_program(inner);
        let rematerialize = RematerializeOperation::<ArrayIrType>::new(ResidualPolicyReference::new(NothingSavable));
        let nested = body
            .add_instruction(rematerialize.clone(), vec![inner], vec![product, input], None)
            .unwrap()
            .to_vec();
        let body = build(body, vec![sine, product, nested[0], nested[1]]);
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let input = builder.add_input(vector_type());
        let body = builder.import_program(body);
        let outputs = builder.add_instruction(rematerialize, vec![body], vec![input], None).unwrap().to_vec();
        let program = build(builder, outputs);

        // Every answer, settled or not, is the answer of the exact traversal.
        let policy = save_dots(None);
        let instruction = &program.instructions()[0];
        let regions = program.region_data_flow_boundaries(instruction).unwrap();
        let boundary = RegionDataFlowBoundary { input_count: 1, output_count: 4, regions: &regions };
        for (output_index, expected, settled) in
            [(0, true, true), (1, false, true), (2, true, false), (3, false, false)]
        {
            let mut replay = ResidualReplayAnalysis::new(&program, &policy);
            assert_eq!(replay.allows_replay(0, output_index), Ok(expected));
            assert_eq!(replay.resolved.is_empty(), settled);
            let mut outputs = vec![false; 4];
            outputs[output_index] = true;
            let mut replay = ResidualReplayAnalysis::new(&program, &policy);
            let roots = replay.demands(program.entry(), instruction, &outputs, boundary).unwrap();
            assert_eq!(replay.traverse(&roots), Ok(expected));
        }
    }

    #[test]
    fn test_residual_replay_analysis_demand_bounds_missing_region() {
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let input = builder.add_input(scalar_type());
        let program = build(builder, vec![input]);
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                in (%0)
            "}
            .trim_end(),
        );
        let policy = save_nothing();
        let mut replay = ResidualReplayAnalysis::new(&program, &policy);
        assert_eq!(replay.demand_bounds(RegionId::new(program.regions().len()), &[true]), ReplayBounds::UNKNOWN);
        assert!(replay.bounds.is_empty());
        assert_eq!(replay.demand_bounds(program.entry(), &[true]), ReplayBounds::CLEAN);
        assert_eq!(replay.bounds.iter().flatten().count(), 1);
        assert_eq!(replay.demand_bounds(RegionId::new(usize::MAX), &[true]), ReplayBounds::UNKNOWN);
        assert_eq!(replay.bounds.iter().flatten().count(), 1);
    }

    #[test]
    fn test_residual_provenance_analysis_resolve_shares_immutable_summaries() {
        let mut first = ProgramBuilder::<TestValue, TestOperation>::new();
        let input = first.add_input(scalar_type());
        let first_output = add(&mut first, TagOperation::<ArrayType>::new("first").into(), vec![input]);
        let first = build(first, vec![first_output]);
        let mut second = ProgramBuilder::<TestValue, TestOperation>::new();
        let input = second.add_input(scalar_type());
        let second_output = add(&mut second, TagOperation::<ArrayType>::new("second").into(), vec![input]);
        let second = build(second, vec![second_output]);
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let input = builder.add_input(scalar_type());
        let constant = builder.add_constant(ArrayIrValue::Array(Array::scalar(0.0f64).unwrap()));
        let first = builder.import_program(first);
        let second = builder.import_program(second);
        let output = builder
            .add_instruction(ConditionOperation::new(), vec![first, second], vec![predicate, input], None)
            .unwrap()[0];
        let program = build(builder, vec![output]);
        let mut provenance = ResidualProvenanceAnalysis::new(&program, None);
        assert!(provenance.resolved.iter().all(Vec::is_empty));
        assert_eq!(provenance.resolve(ValueId::new(program.entry(), constant)), Ok(ResidualProvenanceLeaves::Empty),);
        assert_eq!(
            provenance.resolve(ValueId::new(program.entry(), input)),
            Ok(ResidualProvenanceLeaves::One(ResidualProvenanceLeaf::Input(1))),
        );
        assert_eq!(provenance.resolved[program.entry().index()].len(), program.atoms().len());
        assert!(provenance.resolved[first.index()].is_empty());
        assert!(provenance.resolved[second.index()].is_empty());
        let initial = provenance.resolve(ValueId::new(program.entry(), output)).unwrap();
        let entry_slots = provenance.resolved[program.entry().index()].as_ptr();
        let repeated = provenance.resolve(ValueId::new(program.entry(), output)).unwrap();
        assert_eq!(provenance.resolved[program.entry().index()].as_ptr(), entry_slots);
        assert_eq!(
            initial.as_slice(),
            &[
                ResidualProvenanceLeaf::Producer(ValueId::new(first, first_output)),
                ResidualProvenanceLeaf::Producer(ValueId::new(second, second_output)),
            ],
        );
        assert!(matches!(
            (&initial, &repeated),
            (ResidualProvenanceLeaves::Many(first), ResidualProvenanceLeaves::Many(second))
                if Arc::ptr_eq(first, second),
        ));
        assert_eq!(
            provenance.resolved[first.index()][first_output.index()],
            Some(ResidualProvenanceLeaves::One(ResidualProvenanceLeaf::Producer(ValueId::new(first, first_output)))),
        );
        assert_eq!(
            provenance.resolved[second.index()][second_output.index()],
            Some(ResidualProvenanceLeaves::One(ResidualProvenanceLeaf::Producer(ValueId::new(second, second_output)))),
        );
    }

    #[test]
    fn test_residual_provenance_analysis_resolve_forwards_immutable_summaries() {
        let mut first = ProgramBuilder::<TestValue, TestOperation>::new();
        let input = first.add_input(scalar_type());
        let output = add(&mut first, TagOperation::<ArrayType>::new("first").into(), vec![input]);
        let first = build(first, vec![output]);
        let mut second = ProgramBuilder::<TestValue, TestOperation>::new();
        let input = second.add_input(scalar_type());
        let output = add(&mut second, TagOperation::<ArrayType>::new("second").into(), vec![input]);
        let second = build(second, vec![output]);
        let mut body = ProgramBuilder::<TestValue, TestOperation>::new();
        let predicate = body.add_input(ArrayType::scalar(DataType::Boolean).into());
        let input = body.add_input(scalar_type());
        let first = body.import_program(first);
        let second = body.import_program(second);
        let output = body
            .add_instruction(ConditionOperation::new(), vec![first, second], vec![predicate, input], None)
            .unwrap()[0];
        let body = build(body, vec![output]);
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let input = builder.add_input(scalar_type());
        let body = builder.import_program(body);
        let output = builder
            .add_instruction(
                RematerializeOperation::<ArrayIrType>::new(ResidualPolicyReference::new(NothingSavable)),
                vec![body],
                vec![predicate, input],
                None,
            )
            .unwrap()[0];
        let program = build(builder, vec![output]);
        let body_output = program.region(body).unwrap().output_ids()[0];
        let mut provenance = ResidualProvenanceAnalysis::new(&program, None);
        let nested = provenance.resolve(ValueId::new(body, body_output)).unwrap();
        let forwarded = provenance.resolve(ValueId::new(program.entry(), output)).unwrap();
        assert!(matches!(
            (&nested, &forwarded),
            (ResidualProvenanceLeaves::Many(nested), ResidualProvenanceLeaves::Many(forwarded))
                if Arc::ptr_eq(nested, forwarded),
        ));
        assert_eq!(forwarded.as_slice(), nested.as_slice());
    }

    #[test]
    fn test_residual_provenance_analysis_resolve_memoizes_shared_diamonds() {
        // Every condition reaches the same symbolic input through two paths, and forty conditions form a shared
        // diamond at each of two differently tagged callers. Expansion paths grow exponentially; distinct values do
        // not. Tags also ensure that the fixture cannot bypass provenance resolution by forwarding an original input.
        let mut branch = ProgramBuilder::<TestValue, TestOperation>::new();
        let input = branch.add_input(scalar_type());
        let branch = build(branch, vec![input]);
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let input = builder.add_input(scalar_type());
        let unknown = builder.add_input(scalar_type());
        let first = add(&mut builder, TagOperation::<ArrayType>::new("first").into(), vec![input]);
        let second = add(&mut builder, TagOperation::<ArrayType>::new("second").into(), vec![input]);
        let branch = builder.import_program(branch);
        let mut values = [first, second];
        for _ in 0..40 {
            for value in &mut values {
                *value = builder
                    .add_instruction(ConditionOperation::new(), vec![branch, branch], vec![predicate, *value], None)
                    .unwrap()[0];
            }
        }
        let outputs =
            values.map(|value| add(&mut builder, MulOperation::<ArrayType>::new().into(), vec![value, unknown]));
        let program = build(builder, outputs.to_vec());
        let mut provenance = ResidualProvenanceAnalysis::new(&program, None);
        for (value, producer) in values.into_iter().zip([first, second]) {
            assert_eq!(
                provenance
                    .resolve(ValueId::new(program.entry(), value))
                    .map(|leaves| leaves.as_slice().to_vec())
                    .unwrap(),
                vec![ResidualProvenanceLeaf::Producer(ValueId::new(program.entry(), producer))],
            );
        }
        assert_eq!(provenance.resolved.iter().flatten().filter(|leaves| leaves.is_some()).count(), 83);
        let resolved = provenance.resolved.clone();
        let candidate = provenance.candidate(ValueId::new(program.entry(), values[0]), scalar_type()).unwrap().unwrap();
        assert_eq!(candidate.producers().len(), 1);
        assert_eq!(candidate.producers()[0].payload::<TagOperation<ArrayType>>().unwrap().key(), "first");
        assert_eq!(provenance.resolved, resolved);

        let placed = program
            .partition(&[true, true, false])
            .unwrap()
            .with_residual_policy(&save_names(&["first"], &[]))
            .unwrap();
        let inputs = vec![
            ArrayIrValue::Array(Array::scalar(false).unwrap()),
            ArrayIrValue::Array(Array::scalar(0.5f64).unwrap()),
            ArrayIrValue::Array(Array::scalar(2.0f64).unwrap()),
        ];
        assert_eq!(run(&placed, &inputs), program.interpret(inputs.clone()).unwrap());
        assert_eq!(placed.known_program().output_ids().len(), 3);
    }

    #[test]
    fn test_residual_provenance_analysis_resolve_memoizes_operation_queries() {
        /// Canonical region fixture observing semantic queries without instrumenting the production resolver.
        #[derive(Clone)]
        struct CountedOperation {
            /// Operation whose region and type contracts the fixture reuses.
            operation: TestRegionOperation,

            /// Shared count of queries across the program's distinct operation applications.
            queries: Arc<AtomicUsize>,
        }

        impl Operation for CountedOperation {
            type Type = ArrayType;

            fn name(&self) -> &'static str {
                self.operation.name()
            }

            fn region_slots(&self) -> &'static [RegionSlot] {
                self.operation.region_slots()
            }

            fn infer_output_types(
                &self,
                input_types: &[ArrayType],
                region_interfaces: &[RegionInterface<ArrayType>],
            ) -> Result<Vec<ArrayType>, TypeError> {
                self.operation.infer_output_types(input_types, region_interfaces)
            }

            fn input_region_provenance(&self, _region_index: usize, input_index: usize) -> InputRegionProvenance {
                InputRegionProvenance::Input { index: input_index }
            }

            fn output_region_provenance(&self, output_index: usize) -> Vec<OutputRegionProvenance> {
                (0..self.region_slots().len())
                    .map(|region_index| OutputRegionProvenance { region_index, output_index })
                    .collect()
            }

            fn region_data_flow(&self) -> RegionDataFlow<'_> {
                let queries = self.queries.fetch_add(1, Ordering::Relaxed) + 1;
                // Bound work deterministically: a cache regression must fail before unfolding the diamond's paths.
                assert!(queries <= 82, "shared provenance must not repeat operation queries");
                if self.region_slots().is_empty() { RegionDataFlow::Opaque } else { RegionDataFlow::Provenance }
            }
        }

        impl OperationPayloadProjection for CountedOperation {
            fn project_payload(&self, _payload: TypeId) -> Option<&dyn Any> {
                None
            }

            fn from_payload(payload: ErasedOperation) -> Result<Self, ErasedOperation> {
                Err(payload)
            }
        }

        let scalar_type = ArrayType::scalar(DataType::F64);
        let mut branch = ProgramBuilder::<Array, CountedOperation>::new();
        let input = branch.add_input(scalar_type.clone());
        let branch = branch.build::<Vec<Array>, Vec<Array>>(vec![input], vec![Placeholder], vec![Placeholder]).unwrap();
        let queries = Arc::new(AtomicUsize::new(0));
        let leaf = CountedOperation { operation: TestRegionOperation::Add, queries: queries.clone() };
        let carrier = CountedOperation {
            operation: TestRegionOperation::WithRegions(
                const { &[RegionSlot::computation("first"), RegionSlot::computation("second")] },
            ),
            queries: queries.clone(),
        };
        let mut builder = ProgramBuilder::<Array, CountedOperation>::new();
        let first_input = builder.add_input(scalar_type.clone());
        let second_input = builder.add_input(scalar_type.clone());
        let first = builder.add_instruction(leaf.clone(), Vec::new(), vec![first_input, first_input], None).unwrap()[0];
        let second = builder.add_instruction(leaf, Vec::new(), vec![second_input, second_input], None).unwrap()[0];
        let branch = builder.import_program(branch);
        let mut values = [first, second];
        for _ in 0..40 {
            for value in &mut values {
                *value = builder.add_instruction(carrier.clone(), vec![branch, branch], vec![*value], None).unwrap()[0];
            }
        }
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(values.to_vec(), vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        assert_eq!(program.entry_region_ref().instructions_in_closure().count(), 82);
        queries.store(0, Ordering::Relaxed);
        let mut provenance = ResidualProvenanceAnalysis::new(&program, None);
        assert_eq!(
            provenance
                .resolve(ValueId::new(program.entry(), values[0]))
                .map(|leaves| leaves.as_slice().to_vec()),
            Ok(vec![ResidualProvenanceLeaf::Producer(ValueId::new(program.entry(), first))]),
        );
        assert_eq!(
            provenance
                .resolve(ValueId::new(program.entry(), values[1]))
                .map(|leaves| leaves.as_slice().to_vec()),
            Ok(vec![ResidualProvenanceLeaf::Producer(ValueId::new(program.entry(), second))]),
        );
        assert_eq!(queries.load(Ordering::Relaxed), 82);
        assert_eq!(provenance.resolved.iter().flatten().filter(|leaves| leaves.is_some()).count(), 83);

        // Repeating a resolution and asking for its candidate must use the already cached symbolic summary.
        assert_eq!(
            provenance
                .resolve(ValueId::new(program.entry(), values[0]))
                .map(|leaves| leaves.as_slice().to_vec()),
            Ok(vec![ResidualProvenanceLeaf::Producer(ValueId::new(program.entry(), first))]),
        );
        let candidate = provenance.candidate(ValueId::new(program.entry(), values[0]), scalar_type).unwrap().unwrap();
        assert_eq!(candidate.producers().len(), 1);
        assert_eq!(candidate.producers()[0].name(), "add");
        assert_eq!(queries.load(Ordering::Relaxed), 82);
    }

    #[test]
    fn test_residual_provenance_analysis_resolve_preserves_alternative_order() {
        // The first branch reaches its tagged producer through a forwarded input, while the second reaches its
        // producer immediately. Traversal must complete each declared alternative before starting its sibling.
        let mut first = ProgramBuilder::<TestValue, TestOperation>::new();
        let input = first.add_input(scalar_type());
        let first = build(first, vec![input]);
        let mut second = ProgramBuilder::<TestValue, TestOperation>::new();
        let input = second.add_input(scalar_type());
        let sine = add(&mut second, SinOperation::<ArrayType>::new().into(), vec![input]);
        let second = build(second, vec![sine]);
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let input = builder.add_input(scalar_type());
        let tagged = add(&mut builder, TagOperation::<ArrayType>::new("first").into(), vec![input]);
        let first = builder.import_program(first);
        let second = builder.import_program(second);
        let output = builder
            .add_instruction(ConditionOperation::new(), vec![first, second], vec![predicate, tagged], None)
            .unwrap()[0];
        let program = build(builder, vec![output]);
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:bool[], %1:f64[] .
                let %2:f64[] = tag [key=first] %1
                    %3:f64[] = condition %0 %2 [
                        true={
                            lambda %0:f64[] .
                            in (%0)
                        },
                        false={
                            lambda %0:f64[] .
                            let %1:f64[] = sin %0
                            in (%1)
                        },
                    ]
                in (%3)"},
        );
        let mut provenance = ResidualProvenanceAnalysis::new(&program, None);
        assert_eq!(
            provenance.resolve(ValueId::new(program.entry(), output)).map(|leaves| leaves.as_slice().to_vec()),
            Ok(vec![
                ResidualProvenanceLeaf::Producer(ValueId::new(program.entry(), tagged)),
                ResidualProvenanceLeaf::Producer(ValueId::new(second, sine)),
            ]),
        );
    }

    #[test]
    fn test_residual_provenance_analysis_resolve_looks_through_rounding() {
        // A value that a rounding operation produces resolves to the producer of the value that it rounds when the
        // analysis looks through the operations of that rounding, and to the rounding operation otherwise.
        let bf16_type = ArrayIrType::from(ArrayType::scalar(DataType::BF16));
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let input = builder.add_input(bf16_type);
        let tagged = add(&mut builder, TagOperation::<ArrayType>::new("t").into(), vec![input]);
        let rounded = add(&mut builder, ReducePrecisionOperation::<ArrayType>::new(8, 7).into(), vec![tagged]);
        let program = build(builder, vec![rounded]);
        let value = ValueId::new(program.entry(), rounded);
        let mut provenance = ResidualProvenanceAnalysis::new(&program, Some(&Bfloat16Rounding));
        assert_eq!(
            provenance.resolve(value).map(|leaves| leaves.as_slice().to_vec()),
            Ok(vec![ResidualProvenanceLeaf::Producer(ValueId::new(program.entry(), tagged))]),
        );
        let mut provenance = ResidualProvenanceAnalysis::new(&program, None);
        assert_eq!(
            provenance.resolve(value).map(|leaves| leaves.as_slice().to_vec()),
            Ok(vec![ResidualProvenanceLeaf::Producer(value)])
        );
    }

    #[test]
    fn test_residual_provenance_analysis_resolve_loop_feedback() {
        // The first carry reaches the dot only through the second carry's back edge. The same body attached to a
        // condition remains an ordinary call, and a zero-trip scan returns its own initial carry without feedback.
        let mut body = ProgramBuilder::<TestValue, TestOperation>::new();
        body.add_input(ArrayType::scalar(DataType::I64).into());
        let first = body.add_input(scalar_type());
        let second = body.add_input(scalar_type());
        let dot = add(
            &mut body,
            DotOperation::new(DotDimensionNumbers::new(vec![], vec![], vec![], vec![])).into(),
            vec![first, first],
        );
        let body = build(body, vec![second, dot, first]);
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let input = builder.add_input(scalar_type());
        let first = add(&mut builder, TagOperation::<ArrayType>::new("first").into(), vec![input]);
        let second = add(&mut builder, TagOperation::<ArrayType>::new("second").into(), vec![input]);
        let index = builder.add_constant(ArrayIrValue::Array(Array::scalar(0i64).unwrap()));
        let body = builder.import_program(body);
        let scan = builder
            .add_instruction(ScanOperation::<ArrayIrType>::new(2, 2), vec![body], vec![first, second], None)
            .unwrap()
            .to_vec();
        let empty = builder
            .add_instruction(ScanOperation::<ArrayIrType>::new(2, 0), vec![body], vec![first, second], None)
            .unwrap()
            .to_vec();
        let ordinary = builder
            .add_instruction(ConditionOperation::new(), vec![body, body], vec![predicate, index, first, second], None)
            .unwrap()
            .to_vec();
        let program = build(builder, [scan.clone(), empty.clone(), ordinary.clone()].concat());
        let mut provenance = ResidualProvenanceAnalysis::new(&program, None);
        let first = ResidualProvenanceLeaf::Producer(ValueId::new(program.entry(), first));
        let second = ResidualProvenanceLeaf::Producer(ValueId::new(program.entry(), second));
        let dot = ResidualProvenanceLeaf::Producer(ValueId::new(body, dot));
        assert_eq!(
            provenance
                .resolve(ValueId::new(program.entry(), scan[0]))
                .map(|leaves| leaves.as_slice().to_vec())
                .unwrap(),
            vec![first, second, dot]
        );
        assert_eq!(
            provenance
                .resolve(ValueId::new(program.entry(), scan[1]))
                .map(|leaves| leaves.as_slice().to_vec())
                .unwrap(),
            vec![second, dot]
        );
        assert_eq!(
            provenance
                .resolve(ValueId::new(program.entry(), scan[2]))
                .map(|leaves| leaves.as_slice().to_vec())
                .unwrap(),
            vec![first, second, dot]
        );
        assert_eq!(
            provenance
                .resolve(ValueId::new(program.entry(), empty[0]))
                .map(|leaves| leaves.as_slice().to_vec())
                .unwrap(),
            vec![first]
        );
        assert_eq!(
            provenance
                .resolve(ValueId::new(program.entry(), empty[1]))
                .map(|leaves| leaves.as_slice().to_vec())
                .unwrap(),
            vec![second]
        );
        assert_eq!(
            provenance
                .resolve(ValueId::new(program.entry(), empty[2]))
                .map(|leaves| leaves.as_slice().to_vec())
                .unwrap(),
            vec![]
        );
        assert_eq!(
            provenance
                .resolve(ValueId::new(program.entry(), ordinary[0]))
                .map(|leaves| leaves.as_slice().to_vec())
                .unwrap(),
            vec![second]
        );
        assert_eq!(
            provenance
                .resolve(ValueId::new(program.entry(), ordinary[2]))
                .map(|leaves| leaves.as_slice().to_vec())
                .unwrap(),
            vec![first]
        );
    }

    #[test]
    fn test_residual_provenance_analysis_resolve_while_feedback() {
        let mut condition = ProgramBuilder::<TestValue, TestOperation>::new();
        condition.add_input(scalar_type());
        condition.add_input(scalar_type());
        let predicate = condition.add_constant(ArrayIrValue::Array(Array::scalar(false).unwrap()));
        let condition = build(condition, vec![predicate]);
        let mut body = ProgramBuilder::<TestValue, TestOperation>::new();
        let first = body.add_input(scalar_type());
        let second = body.add_input(scalar_type());
        let dot = add(
            &mut body,
            DotOperation::new(DotDimensionNumbers::new(vec![], vec![], vec![], vec![])).into(),
            vec![first, first],
        );
        let body = build(body, vec![second, dot]);
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let input = builder.add_input(scalar_type());
        let first = add(&mut builder, TagOperation::<ArrayType>::new("first").into(), vec![input]);
        let second = add(&mut builder, TagOperation::<ArrayType>::new("second").into(), vec![input]);
        let condition = builder.import_program(condition);
        let body = builder.import_program(body);
        let outputs = builder
            .add_instruction(
                WhileOperation::<ArrayIrType>::new().with_iteration_bound(2).unwrap(),
                vec![condition, body],
                vec![first, second],
                None,
            )
            .unwrap()
            .to_vec();
        let program = build(builder, outputs.clone());
        let mut provenance = ResidualProvenanceAnalysis::new(&program, None);
        assert_eq!(
            provenance
                .resolve(ValueId::new(program.entry(), outputs[0]))
                .map(|leaves| leaves.as_slice().to_vec())
                .unwrap(),
            vec![
                ResidualProvenanceLeaf::Producer(ValueId::new(program.entry(), first)),
                ResidualProvenanceLeaf::Producer(ValueId::new(program.entry(), second)),
                ResidualProvenanceLeaf::Producer(ValueId::new(body, dot)),
            ],
        );
        // The condition happens to be false, illustrating why the initial carry remains in the conservative union.
        let input = ArrayIrValue::Array(Array::scalar(3f64).unwrap());
        assert_eq!(program.interpret(vec![input.clone()]).unwrap(), vec![input.clone(), input]);
    }

    #[test]
    fn test_residual_provenance_analysis_resolve_rejects_malformed_provenance() {
        /// Region-machinery fixture with deliberately invalid provenance, leaving slot and type validation intact.
        #[derive(Clone)]
        struct MalformedProvenance {
            /// Fixture whose ordinary region contract is valid.
            operation: TestRegionOperation,

            /// Declared source of the body input.
            input: InputRegionProvenance,

            /// Declared source of the instruction output.
            output: OutputRegionProvenance,
        }

        impl Operation for MalformedProvenance {
            type Type = ArrayType;

            fn name(&self) -> &'static str {
                "test.provenance"
            }

            fn region_slots(&self) -> &'static [RegionSlot] {
                self.operation.region_slots()
            }

            fn infer_output_types(
                &self,
                input_types: &[ArrayType],
                region_interfaces: &[RegionInterface<ArrayType>],
            ) -> Result<Vec<ArrayType>, TypeError> {
                self.operation.infer_output_types(input_types, region_interfaces)
            }

            fn input_region_provenance(&self, _region_index: usize, _input_index: usize) -> InputRegionProvenance {
                self.input
            }

            fn region_data_flow(&self) -> RegionDataFlow<'_> {
                RegionDataFlow::Provenance
            }

            fn output_region_provenance(&self, _output_index: usize) -> Vec<OutputRegionProvenance> {
                vec![self.output]
            }
        }

        impl OperationPayloadProjection for MalformedProvenance {
            fn project_payload(&self, _payload: TypeId) -> Option<&dyn Any> {
                None
            }

            fn from_payload(payload: ErasedOperation) -> Result<Self, ErasedOperation> {
                Err(payload)
            }
        }

        for (input, output, message) in [
            (
                InputRegionProvenance::Input { index: 0 },
                OutputRegionProvenance { region_index: 1, output_index: 0 },
                "operation `test.provenance` declares region 1 but has 1 attached regions",
            ),
            (
                InputRegionProvenance::Input { index: 0 },
                OutputRegionProvenance { region_index: 0, output_index: 1 },
                "operation `test.provenance` declares output 1 of region 0 with 1 outputs",
            ),
            (
                InputRegionProvenance::Input { index: 1 },
                OutputRegionProvenance { region_index: 0, output_index: 0 },
                "operation `test.provenance` declares instruction input 1 but has 1 inputs",
            ),
        ] {
            let mut body = ProgramBuilder::<Array, MalformedProvenance>::new();
            let value = body.add_input(ArrayType::scalar(DataType::F64));
            let body = body.build::<Vec<Array>, Vec<Array>>(vec![value], vec![Placeholder], vec![Placeholder]).unwrap();
            let mut known = ProgramBuilder::<Array, MalformedProvenance>::new();
            let value = known.add_input(ArrayType::scalar(DataType::F64));
            let body = known.import_program(body);
            let output = known
                .add_instruction(
                    MalformedProvenance {
                        operation: TestRegionOperation::WithRegions(const { &[RegionSlot::computation("body")] }),
                        input,
                        output,
                    },
                    vec![body],
                    vec![value],
                    None,
                )
                .unwrap()[0];
            let known =
                known.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
            let mut provenance = ResidualProvenanceAnalysis::new(&known, None);
            assert_eq!(
                provenance
                    .candidate(ValueId::new(known.entry(), output), ArrayType::scalar(DataType::F64))
                    .map(|_| ()),
                Err(ProgramError::MalformedProgram(message.to_owned())),
            );
        }
    }

    #[test]
    fn test_residual_provenance_analysis_resolve_rejects_invalid_values() {
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let input = builder.add_input(scalar_type());
        let program = build(builder, vec![input]);
        let mut provenance = ResidualProvenanceAnalysis::new(&program, None);
        let missing = AtomId::new(1);
        assert_eq!(
            provenance.resolve(ValueId::new(program.entry(), missing)).map(|leaves| leaves.as_slice().to_vec()),
            Err(ProgramError::UnboundAtomId { id: missing }),
        );
        assert_eq!(
            provenance.resolve(ValueId::new(RegionId::new(1), input)).map(|leaves| leaves.as_slice().to_vec()),
            Err(ProgramError::MalformedProgram("region ^1 is out of range".to_owned())),
        );
        assert!(provenance.resolved.iter().all(Vec::is_empty));
    }
}
