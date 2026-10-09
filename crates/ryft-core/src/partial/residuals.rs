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

use std::any::{Any, TypeId};
use std::borrow::Cow;
use std::collections::hash_map::Entry;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use thiserror::Error;

use crate::parameters::Placeholder;
use crate::partial::partitions::PartitionedProgram;
use crate::partial::values::ResidualInputSource;
use crate::programs::{
    Atom, AtomId, AttachedRegionLiveness, ErasedOperation, Instruction, Operation, OperationPayloadProjection, Program,
    ProgramBuilder, ProgramError, RegionDataFlowBoundary, RegionDataFlowSource, RegionDataFlowSources, RegionId,
    RegionPruningAnalysis, Type, Typed, Value, ValueId,
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
/// described without naming an operation family. Each transformation is a chain of payload operations (e.g.,
/// [`TransferToMemoryOperation`](crate::TransferToMemoryOperation)s) that [`PartitionedProgram::with_residual_policy`]
/// stages into the family of the partitioned programs through [`OperationPayloadProjection::from_payload`]. The store
/// chain applies to the residual in the known program, and the restore chain reproduces it in the residual program
/// before its first use. Every staged operation must be a pure unary operation with a single result, which leaves
/// program simplification free to schedule it anywhere that its data dependencies allow, and the restore chain must
/// reproduce the type of the residual up to identity renaming.
pub trait ResidualStorage<T: Type>: 'static + Send + Sync + Debug {
    /// Returns the name of this storage, which is used in diagnostics.
    fn name(&self) -> String;

    /// Returns the payload operations that store a residual of type `residual_type`, in application order.
    fn store_payloads(&self, residual_type: &T) -> Result<Vec<ErasedOperation>, ResidualPolicyError>;

    /// Returns the payload operations that restore a residual of type `residual_type` from its stored form of type
    /// `stored_type`, in application order.
    fn restore_payloads(&self, stored_type: &T, residual_type: &T)
    -> Result<Vec<ErasedOperation>, ResidualPolicyError>;
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
/// and so it has a native instantiation in every type universe in which both of them have one. Other composite policies
/// register their instantiations explicitly through [`with`](Self::with), which requires the composite policy to
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
/// combine registries; another composite policy that needs instantiations derived from erased child registries
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
        self.with_residual_policy_and_region_replay(policy, true)
    }

    /// Places the residuals of this partition like [`with_residual_policy`](Self::with_residual_policy). When
    /// `replay_region_operations` is `false`, a requested recomputation of a region-carrying output preserves the
    /// existing nested cuts unless the policy recomputes every live producer needed for that output. Unused sibling
    /// outputs and dormant derivative regions do not contribute demand. Requested saves, storage, and errors are
    /// handled directly before checking whether replay would discard a nested cut.
    fn with_residual_policy_and_region_replay(
        self,
        policy: &ResidualPolicyReference<V::Type>,
        replay_region_operations: bool,
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

        // Local references that escape through an output of the known program (including handle edges) are shared with
        // work outside it, so their lifecycles are not tracked and none of their accesses is ever replayed.
        let region = known_program.entry_region_ref();
        let analysis = region.reference_analysis(0).map_err(ProgramError::from)?;
        let escaping = analysis.output_roots().iter().flatten().copied().collect::<HashSet<_>>();
        let lifecycles = analysis.local_lifecycles(region, |root| !escaping.contains(&root))?;
        let atom_count = known_program.atoms().len();
        let instruction_by_output = known_program.instruction_by_output();
        let mut is_input = vec![false; atom_count];
        known_program.input_ids().iter().for_each(|input| is_input[input.index()] = true);

        // State predecessors always precede their consumer, so one forward pass determines replayability of every
        // lifecycle prefix. Demanded outputs can then share the result without repeatedly walking those prefixes.
        let replayable = lifecycles.replayable_instructions();

        // Replaying the known half of a nested split must preserve its interior policy decisions. Operations determine
        // the demanded outputs of the regions that they execute, which the analysis uses to cache work per region.
        let mut replay_analysis = ResidualReplayAnalysis::new(&known_program, policy);

        // Plans one demanded known atom, applying the rules listed in the documentation of `with_residual_policy` in
        // order: constants and known inputs never consult the policy, reference handles are either original edges or
        // replayed, values whose producers cannot be replayed are saved, and the policy classifies everything else.
        // The provenance of the known program provides the candidates that the policy classifies, and memoizes the
        // provenance of every visited value across calls.
        let mut provenance = ResidualProvenanceAnalysis::new(&known_program);
        let mut plan = |atom: AtomId| -> Result<ResidualPlan<V::Type>, ResidualPolicyError> {
            let atom_type = match &known_program.atoms()[atom.index()] {
                Atom::Constant(_) => return Ok(ResidualPlan::Constant),
                Atom::Variable(r#type) => r#type.clone(),
            };

            if is_input[atom.index()] {
                return Ok(ResidualPlan::Edge(None));
            }

            // The `unwrap` is safe because every variable that is not an input is produced by an instruction.
            let index = instruction_by_output[atom.index()].unwrap();
            if atom_type.is_reference() {
                // Residual work uses local reference state only through replayed lifecycles, and never through new
                // handle edges. The handle edges of the original partition stay edges.
                return if edges.contains(&atom) {
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
                    // cannot hide its rejection or its requested storage. Only recomputation can overwrite the
                    // existing interior cuts of a nested split and therefore needs this extra validation.
                    let instruction = &known_program.instructions()[index];
                    let output_index = instruction.outputs().iter().position(|output| *output == atom).unwrap();
                    if !replay_region_operations
                        && !instruction.regions().is_empty()
                        && !replay_analysis.allows_replay(index, output_index)?
                    {
                        ResidualPlan::Edge(None)
                    } else {
                        ResidualPlan::Recompute
                    }
                }
            })
        };

        // Discover the plans of all demanded known atoms before emitting anything, so that emission sees the final
        // decisions. Atoms are planned depth-first in seed order, and each one only once. Recomputing an atom marks its
        // producer for replay, and replaying a producer demands its inputs and replays its state predecessors in turn.
        // The loop therefore alternates between draining the pending atoms and marking the next pending instruction for
        // replay, until neither remains.
        let mut plans = (0..atom_count).map(|_| None).collect::<Vec<Option<ResidualPlan<V::Type>>>>();
        let mut replay_instructions = BTreeSet::new();
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
                    // The `unwrap` is safe because only atoms produced by instructions are ever recomputed.
                    ResidualPlan::Recompute => pending_instructions.push(instruction_by_output[atom.index()].unwrap()),
                    ResidualPlan::Edge(_) | ResidualPlan::Constant => {}
                }
                plans[atom.index()] = Some(atom_plan);
            }

            let Some(index) = pending_instructions.pop() else {
                break;
            };

            if replay_instructions.insert(index) {
                pending_atoms.extend(known_program.instructions()[index].inputs().iter().rev().copied());
                pending_instructions.extend(lifecycles.state_predecessors(index));
            }
        }

        // The edges of the resulting partition are the saved atoms, in the order of the residual program inputs that
        // consume them: the original edges that remain edges, at their relative positions among the inputs of the
        // original residual program, followed by the new edges in known-program order.
        let is_edge = |atom: &AtomId| matches!(plans[atom.index()], Some(ResidualPlan::Edge(_)));
        let mut new_edges = Vec::new();
        let original_edges = residual_inputs.iter().filter_map(|input| match input {
            ResidualInputSource::ResidualEdge(edge) => Some(edges[*edge]),
            _ => None,
        });
        for atom in original_edges.chain((0..atom_count).map(AtomId::new)) {
            if is_edge(&atom) && !new_edges.contains(&atom) {
                new_edges.push(atom);
            }
        }

        // Emit the resulting known program, which copies the known program with its known outputs followed by the new
        // edges as outputs. The store operations of stored edges are staged right after their producers, so that the
        // edge outputs carry the stored representations.
        let mut builder = ProgramBuilder::<V, O>::new();
        let mut atoms = vec![None; atom_count];
        for input in known_program.input_ids() {
            atoms[input.index()] = Some(builder.add_input(known_program.atoms()[input.index()].r#type().into_owned()));
        }

        let mut stored_atoms = HashMap::new();
        let mut remapping = HashMap::new();
        for instruction in known_program.instructions() {
            let inputs = instruction
                .inputs()
                .iter()
                .map(|input| copy_atom(&known_program, *input, &mut atoms, &mut builder))
                .collect::<Result<Vec<_>, _>>()?;
            let regions = instruction
                .regions()
                .iter()
                .map(|region| {
                    Ok(builder.import_region_with_remapping(known_program.region_ref(*region)?, &mut remapping))
                })
                .collect::<Result<Vec<_>, ProgramError>>()?;
            let outputs = builder
                .add_instruction(
                    instruction.operation().clone(),
                    regions,
                    inputs,
                    Some(instruction.provenance().clone()),
                )?
                .to_vec();
            for (source, output) in instruction.outputs().iter().zip(outputs) {
                atoms[source.index()] = Some(output);
                if let Some(ResidualPlan::Edge(Some(storage))) = &plans[source.index()] {
                    let residual_type = known_program.atoms()[source.index()].r#type().into_owned();
                    let payloads = storage.store_payloads(&residual_type)?;
                    stored_atoms.insert(*source, stage_storage(&mut builder, output, payloads, &**storage)?);
                }
            }
        }

        let mut known_outputs = known_program.output_ids()[..known_output_count]
            .iter()
            .map(|output| copy_atom(&known_program, *output, &mut atoms, &mut builder))
            .collect::<Result<Vec<_>, _>>()?;

        for edge in &new_edges {
            known_outputs.push(match stored_atoms.get(edge) {
                Some(stored_atom) => *stored_atom,
                None => copy_atom(&known_program, *edge, &mut atoms, &mut builder)?,
            });
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
        let mut add_edge_input = |edge: AtomId, builder: &mut ProgramBuilder<V, O>, inputs: &mut Vec<_>| {
            if let Some(position) = new_edges.iter().position(|new_edge| *new_edge == edge)
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
                    add_edge_input(edges[*edge], &mut builder, &mut new_residual_inputs)
                }
                ResidualInputSource::KnownInput(_) | ResidualInputSource::KnownOutput(_) => {
                    unreachable!("forwarded residual inputs are rejected before placement")
                }
            }
        }

        for edge in &new_edges {
            add_edge_input(*edge, &mut builder, &mut new_residual_inputs);
        }

        // Resolves the atom of the residual program that provides the demanded known atom `atom`, memoized in
        // `known_atoms`: its edge input (through the restore operations of its storage, which are staged on first use
        // and must reproduce the residual type), its replayed value, or a re-created constant.
        let resolve_known_atom = |atom: AtomId,
                                  known_atoms: &mut [Option<AtomId>],
                                  builder: &mut ProgramBuilder<V, O>|
         -> Result<AtomId, ResidualPolicyError> {
            if let Some(resolved) = known_atoms[atom.index()] {
                return Ok(resolved);
            }
            let resolved = match &plans[atom.index()] {
                Some(ResidualPlan::Edge(storage)) => {
                    let input = edge_inputs[&atom];
                    match storage {
                        None => input,
                        Some(storage) => {
                            let stored_type = builder.atoms()[input.index()].r#type().into_owned();
                            let residual_type = known_program.atoms()[atom.index()].r#type().into_owned();
                            let payloads = storage.restore_payloads(&stored_type, &residual_type)?;
                            let restored = stage_storage(builder, input, payloads, &**storage)?;
                            let restored_type = builder.atoms()[restored.index()].r#type().into_owned();
                            let reproduces = |left: &V::Type, right: &V::Type| {
                                V::Type::derive_identity_renaming(
                                    std::slice::from_ref(left),
                                    std::slice::from_ref(right),
                                )
                                .is_ok()
                            };
                            if !reproduces(&residual_type, &restored_type)
                                || !reproduces(&restored_type, &residual_type)
                            {
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
                Some(ResidualPlan::Recompute) => {
                    return Err(ProgramError::MalformedProgram(format!(
                        "known atom {atom} is used before the residual program recomputes it",
                    ))
                    .into());
                }
            };
            known_atoms[atom.index()] = Some(resolved);
            Ok(resolved)
        };

        // Replay the instructions of the known program that are marked for replay, in program order, so that every
        // replayed state access observes the same local reference state as in the known program.
        let mut known_atoms = vec![None; atom_count];
        let mut remapping = HashMap::new();
        for index in &replay_instructions {
            let instruction = &known_program.instructions()[*index];
            let inputs = instruction
                .inputs()
                .iter()
                .map(|input| resolve_known_atom(*input, &mut known_atoms, &mut builder))
                .collect::<Result<Vec<_>, _>>()?;
            let regions = instruction
                .regions()
                .iter()
                .map(|region| {
                    Ok(builder.import_region_with_remapping(known_program.region_ref(*region)?, &mut remapping))
                })
                .collect::<Result<Vec<_>, ProgramError>>()?;
            let outputs = builder
                .add_instruction(
                    instruction.operation().clone(),
                    regions,
                    inputs,
                    Some(instruction.provenance().clone()),
                )?
                .to_vec();
            for (source, output) in instruction.outputs().iter().zip(outputs) {
                // Saved outputs keep resolving to their edges even though their producer is replayed.
                if !matches!(plans[source.index()], Some(ResidualPlan::Edge(_))) {
                    known_atoms[source.index()] = Some(output);
                }
            }
        }

        // Edge inputs of the original residual program resolve to the known atoms that they received. Edges that the
        // original residual program does not read were never demanded, so they have no plan and are dropped.
        for (input, atom) in residual_inputs.iter().zip(residual_program.input_ids()) {
            if let ResidualInputSource::ResidualEdge(edge) = input
                && plans[edges[*edge].index()].is_some()
            {
                residual_atoms[atom.index()] = Some(resolve_known_atom(edges[*edge], &mut known_atoms, &mut builder)?);
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
            let regions = instruction
                .regions()
                .iter()
                .map(|region| {
                    Ok(builder.import_region_with_remapping(residual_program.region_ref(*region)?, &mut remapping))
                })
                .collect::<Result<Vec<_>, ProgramError>>()?;
            let outputs = builder
                .add_instruction(
                    instruction.operation().clone(),
                    regions,
                    inputs,
                    Some(instruction.provenance().clone()),
                )?
                .to_vec();
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

        let metadata = metadata.with_residual_inputs(new_residual_inputs);
        Ok(Self::from_programs_and_metadata(new_known_program, new_residual_program, metadata))
    }

    /// Returns this partition with the residual edges that the known program also consumes itself rounded right after
    /// their producers by the operations that `rounding` returns for their types, so that the known work that consumes
    /// such a residual and the residual work that receives it observe the same value. Backends may compute inexact
    /// values at a higher precision than their types (e.g., XLA computes `bf16` values in `f32` under its default
    /// `--xla_allow_excess_precision` setting), and only the edge is materialized at its declared type, so without the
    /// rounding the known consumers of a residual and its residual consumers could observe different values (refer to
    /// [JAX PR #22244](https://github.com/jax-ml/jax/pull/22244), whose `remat_partial_eval` applies the same
    /// rounding for more information). An edge that a chain of store operations of a [`ResidualStorage`] produces
    /// (i.e., of operations for which `is_storage` returns `true`) is rounded at the value that the chain stores,
    /// before it is stored, if the known program also consumes that value elsewhere. Edges that only the residual
    /// program consumes, constant edges, and edges whose types `rounding` returns no operation for are left unchanged.
    /// Each rounding operation is constructed in the operation family of this partition through
    /// [`OperationPayloadProjection::from_payload`]. Rounding requires a partition whose residual edges have not been
    /// [forwarded](Self::forward_residuals) yet, because a forwarded known input or output would reach the residual
    /// program without passing through the edge that this function rounds.
    ///
    /// # Parameters
    ///
    ///   - `rounding`: Function that returns a type-preserving rounding operation for a residual's type, or `None`
    ///     when that type needs no rounding under the caller's policy.
    ///   - `is_storage`: Predicate identifying store operations through which to trace a residual back to its stored
    ///     value. Only instructions with one input and one output are traversed. Selected operations must preserve
    ///     that value apart from its storage placement, so rounding the input also rounds the stored residual.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError::InvalidArgument`] when this partition has forwarded residual inputs and another
    /// [`ProgramError`] when the operation family cannot hold a rounding operation or when rebuilding the known
    /// program fails.
    pub(crate) fn with_rounded_residuals<R: Fn(&V::Type) -> Option<ErasedOperation>, S: Fn(&O) -> bool>(
        self,
        rounding: R,
        is_storage: S,
    ) -> Result<Self, ProgramError> {
        if self.has_forwarded_residual_inputs() {
            return Err(ProgramError::InvalidArgument {
                message: "cannot round the residuals of a partition whose residual inputs are already forwarded"
                    .to_string(),
            });
        }

        let known_output_count = self.outputs().iter().filter(|output| output.is_known()).count();
        let (known_program, residual_program, metadata) = self.into_programs_and_metadata();
        let mut use_counts = vec![0usize; known_program.atoms().len()];
        known_program
            .instructions()
            .iter()
            .flat_map(|instruction| instruction.inputs())
            .for_each(|input| use_counts[input.index()] += 1);
        let instruction_by_output = known_program.instruction_by_output();
        let mut rounded = HashMap::new();
        for edge in &known_program.output_ids()[known_output_count..] {
            // Walk back through the store operations that produce the edge, if any, to the value that the known program
            // consumes outside of that chain (which consumes each intermediate value of the chain exactly once).
            let mut atom = *edge;
            let mut chain_uses = 0;
            let value = loop {
                if !matches!(known_program.atoms()[atom.index()], Atom::Variable(_)) {
                    break None;
                }
                if use_counts[atom.index()] > chain_uses {
                    break Some(atom);
                }
                match instruction_by_output[atom.index()].map(|index| &known_program.instructions()[index]) {
                    Some(instruction)
                        if is_storage(instruction.operation())
                            && instruction.inputs().len() == 1
                            && instruction.outputs().len() == 1 =>
                    {
                        atom = instruction.inputs()[0];
                        chain_uses = 1;
                    }
                    _ => break None,
                }
            };
            if let Some(value) = value
                && !rounded.contains_key(&value)
                && let Some(payload) = rounding(&known_program.atoms()[value.index()].r#type())
            {
                rounded.insert(value, payload);
            }
        }
        if rounded.is_empty() {
            return Ok(Self::from_programs_and_metadata(known_program, residual_program, metadata));
        }

        // Copy the known program, staging each rounding right after the producer of its edge (or at the beginning for
        // edges that are known inputs) and redirecting every later use of the edge, including the edge output itself,
        // to the rounded value.
        let mut builder = ProgramBuilder::<V, O>::new();
        let mut atoms = vec![None; known_program.atoms().len()];
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
        for input in known_program.input_ids() {
            let copy = builder.add_input(known_program.atoms()[input.index()].r#type().into_owned());
            atoms[input.index()] = Some(round(&mut builder, *input, copy)?);
        }

        let mut remapping = HashMap::new();
        for instruction in known_program.instructions() {
            let inputs = instruction
                .inputs()
                .iter()
                .map(|input| copy_atom(&known_program, *input, &mut atoms, &mut builder))
                .collect::<Result<Vec<_>, _>>()?;
            let regions = instruction
                .regions()
                .iter()
                .map(|region| {
                    Ok(builder.import_region_with_remapping(known_program.region_ref(*region)?, &mut remapping))
                })
                .collect::<Result<Vec<_>, ProgramError>>()?;
            let outputs = builder
                .add_instruction(
                    instruction.operation().clone(),
                    regions,
                    inputs,
                    Some(instruction.provenance().clone()),
                )?
                .to_vec();
            for (source, output) in instruction.outputs().iter().zip(outputs) {
                atoms[source.index()] = Some(round(&mut builder, *source, output)?);
            }
        }

        let outputs = known_program
            .output_ids()
            .iter()
            .map(|output| copy_atom(&known_program, *output, &mut atoms, &mut builder))
            .collect::<Result<Vec<_>, _>>()?;
        let input_count = known_program.input_ids().len();
        let output_count = outputs.len();
        let known_program = builder.build::<Vec<V>, Vec<V>>(
            outputs,
            vec![Placeholder; input_count],
            vec![Placeholder; output_count],
        )?;
        Ok(Self::from_programs_and_metadata(known_program, residual_program, metadata))
    }
}

/// Placement of the residuals of the partitions of programs over values of type `V`
/// and operations of family `O` according to one residual policy. It is type-erased so that a
/// [`PartialEvaluationContext`](crate::PartialEvaluationContext) can carry a policy into the partitions that the
/// split rules of region-carrying operations construct, without requiring every operation family to support residual
/// placement. [`ResidualPolicyReference`]s implement it for every family that does.
pub(crate) trait ResidualPlacement<V: Value, O: Operation<Type = V::Type>> {
    /// Returns the policy that places this adapter's residuals, preserving its definition identity.
    fn policy(&self) -> &ResidualPolicyReference<V::Type>;

    /// Returns `partition` with its residuals placed according to the policy (refer to the documentation of
    /// [`PartitionedProgram::with_residual_policy`] for more information on that). Replaying a demanded output of a
    /// region-carrying operation preserves the saved values that its split rule already placed inside its body. Only
    /// producers needed for that output participate. Unused sibling outputs and dormant derivative regions do not
    /// contribute residual demand.
    fn place_residuals(&self, partition: PartitionedProgram<V, O>) -> Result<PartitionedProgram<V, O>, ProgramError>;
}

impl<V: Value<Type: 'static>, O: Operation<Type = V::Type> + OperationPayloadProjection> ResidualPlacement<V, O>
    for ResidualPolicyReference<V::Type>
{
    #[inline]
    fn policy(&self) -> &ResidualPolicyReference<V::Type> {
        self
    }

    #[inline]
    fn place_residuals(&self, partition: PartitionedProgram<V, O>) -> Result<PartitionedProgram<V, O>, ProgramError> {
        Ok(partition.with_residual_policy_and_region_replay(self, false)?)
    }
}

/// Analysis that decides, for one residual policy, whether replaying an output of a region-carrying instruction of the
/// known program of a nested split preserves the residual decisions that the split already made inside the regions of
/// that instruction. Replay recomputes all the work that the regions execute for that output, so it agrees with the
/// policy only when the policy recomputes every value that this work produces, which
/// [`allows_replay`](Self::allows_replay) checks by classifying each of those values.
///
/// The executed work comes from the [`RegionDataFlow`](crate::RegionDataFlow) of each region-carrying operation. For
/// each region that an instruction executes, it determines the _demanded outputs_ of that region: the outputs that the
/// instruction needs from the region, together with every output that the operation's own semantics need for them
/// (e.g., the carries that later iterations of a loop read). The answer for a region therefore depends only on the
/// region, its demanded outputs, and the policy. The policy is fixed for the lifetime of the analysis, so answers are
/// cached by region and demanded outputs, and a region that several instructions share is classified once for each
/// distinct set of demanded outputs.
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
}

impl<'o, V: Value<Type: 'static>, O: Operation<Type = V::Type> + OperationPayloadProjection>
    ResidualReplayAnalysis<'o, V, O>
{
    /// Creates a replay analysis of `program` under `policy`, with empty caches.
    #[inline]
    fn new(program: &'o Program<V, O, Vec<V>, Vec<V>>, policy: &'o ResidualPolicyReference<V::Type>) -> Self {
        Self { program, policy, analysis: RegionPruningAnalysis::new(program.regions()), resolved: HashMap::new() }
    }

    /// Returns whether replaying output `output_index` of the instruction at position `instruction_index` in the
    /// entry region of the program recomputes only values that the policy also recomputes. Returns `false` without
    /// classifying any value when the operation of the instruction does not declare which of its regions produce
    /// that output.
    ///
    /// # Errors
    ///
    /// Returns the errors of classifying values with the policy and of the region data flow and liveness queries.
    fn allows_replay(&mut self, instruction_index: usize, output_index: usize) -> Result<bool, ResidualPolicyError> {
        let program = self.program;
        let policy = self.policy;
        let instruction = &program.instructions()[instruction_index];
        let data_flow = instruction.operation().region_data_flow();
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

        let mut outputs = vec![false; instruction.outputs().len()];
        outputs[output_index] = true;
        let roots = self.demands(instruction, &outputs, boundary)?;
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

                        let used_outputs =
                            instruction.outputs().iter().map(|output| live.atoms()[output.index()]).collect::<Vec<_>>();
                        if instruction.regions().is_empty() {
                            let atom_type = |atom: &AtomId| source.atoms()[atom.index()].r#type().into_owned();
                            let input_types = instruction.inputs().iter().map(atom_type).collect::<Vec<_>>();
                            let output_types = instruction.outputs().iter().map(atom_type).collect::<Vec<_>>();
                            for (output_index, used) in used_outputs.iter().enumerate() {
                                if !used {
                                    continue;
                                }

                                let producer = ResidualProducer::new(
                                    instruction.operation(),
                                    output_index,
                                    input_types.clone(),
                                    output_types.clone(),
                                );

                                let candidate =
                                    ResidualCandidate::new(vec![producer], output_types[output_index].clone());

                                // Saved siblings must not hide another demanded output's rejection.
                                recomputes &= matches!(policy.classify(&candidate)?, ResidualDecision::Recompute);
                            }
                        } else {
                            let regions = program.region_data_flow_boundaries(instruction)?;
                            let boundary = RegionDataFlowBoundary {
                                input_count: instruction.inputs().len(),
                                output_count: instruction.outputs().len(),
                                regions: &regions,
                            };
                            dependencies.extend(self.demands(instruction, &used_outputs, boundary)?);
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
        instruction: &Instruction<O>,
        used_outputs: &[bool],
        boundary: RegionDataFlowBoundary<'_>,
    ) -> Result<Vec<(RegionId, Vec<bool>)>, ProgramError> {
        let operation = instruction.operation();
        let mut regions = AttachedRegionLiveness::new(&mut self.analysis, instruction.regions());
        let demands =
            operation.region_data_flow().execution_demands(operation, used_outputs, boundary, &mut regions)?;
        Ok(instruction
            .regions()
            .iter()
            .zip(demands)
            .filter_map(|(&region, outputs)| outputs.map(|outputs| (region, outputs)))
            .collect())
    }
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
/// is itself kept as a possible producer.
///
/// Every visited value is summarized symbolically, in terms of its containing region's inputs, and each summary is
/// computed once. Region summaries are instantiated separately at every call site. A region that several operations
/// invoke (e.g., with differently tagged inputs) therefore resolves to each call site's own input producers.
struct ResidualProvenanceAnalysis<'o, V: Value, O: Operation<Type = V::Type>> {
    /// Program whose values are resolved.
    program: &'o Program<V, O, Vec<V>, Vec<V>>,

    /// Producer and input positions, indexed once for each region that resolution visits.
    positions: HashMap<RegionId, ResidualAtomPositions>,

    /// Symbolic provenance of each resolved value, independent of the callers of its containing region.
    resolved: HashMap<ValueId, Vec<ResidualProvenanceLeaf>>,
}

impl<'o, V: Value<Type: 'static>, O: Operation<Type = V::Type> + OperationPayloadProjection>
    ResidualProvenanceAnalysis<'o, V, O>
{
    /// Creates a new [`ResidualProvenanceAnalysis`] of `program`, with empty caches.
    #[inline]
    fn new(program: &'o Program<V, O, Vec<V>, Vec<V>>) -> Self {
        Self { program, positions: HashMap::new(), resolved: HashMap::new() }
    }

    /// Returns the [`ResidualCandidate`] that describes `value` to a [`ResidualPolicy`], which classifies it to decide
    /// whether residual work receives `value` as a saved residual or recomputes it. The candidate lists every operation
    /// output that may have produced `value`, in the order of its [`resolve`](Self::resolve)d provenance. Operation
    /// outputs that only forward an output of an attached region are looked through, so that the candidate lists the
    /// operations inside those regions that may actually produce the value (e.g., one producer for each branch of a
    /// `condition`).
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
        for leaf in self.resolve(value)? {
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
        if let Entry::Vacant(entry) = self.positions.entry(region) {
            let region = self.program.region(region)?;
            let mut producers = vec![None; region.atoms().len()];
            let mut inputs = vec![None; region.atoms().len()];
            for (instruction_index, instruction) in region.instructions().iter().enumerate() {
                for (output_index, output) in instruction.outputs().iter().enumerate() {
                    producers[output.index()] = Some((instruction_index, output_index));
                }
            }
            for (input_index, input) in region.input_ids().iter().enumerate() {
                inputs[input.index()] = Some(input_index);
            }
            entry.insert(ResidualAtomPositions { producers, inputs });
        }
        Ok(&self.positions[&region])
    }

    /// Returns the symbolic provenance of `value`, in semantic order and without duplicates. Every value is resolved
    /// once, including caller values that several branches share. Region inputs remain symbolic until their caller
    /// instantiates them, so caching never confuses callers with differently tagged inputs.
    fn resolve(&mut self, value: ValueId) -> Result<Vec<ResidualProvenanceLeaf>, ProgramError> {
        if let Some(leaves) = self.resolved.get(&value) {
            return Ok(leaves.clone());
        }
        let leaves = self.resolve_uncached(value)?;
        self.resolved.insert(value, leaves.clone());
        Ok(leaves)
    }

    /// Computes the symbolic provenance of `value` for [`resolve`](Self::resolve), which caches it.
    fn resolve_uncached(&mut self, value: ValueId) -> Result<Vec<ResidualProvenanceLeaf>, ProgramError> {
        let program = self.program;
        let positions = self.positions(value.region())?;
        let producer = positions
            .producers
            .get(value.atom().index())
            .ok_or(ProgramError::UnboundAtomId { id: value.atom() })?;

        // Inputs stay symbolic and constants have no provenance. The caller of this region substitutes each input
        // through its own input bindings, without changing the cached region-local summary.
        let Some((instruction_index, output_index)) = *producer else {
            return Ok(positions.inputs[value.atom().index()].map(ResidualProvenanceLeaf::Input).into_iter().collect());
        };
        let instruction = &program.region(value.region())?.instructions()[instruction_index];

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
            return Ok(vec![ResidualProvenanceLeaf::Producer(value)]);
        };

        let mut pending = sources.into_iter().rev().map(ResidualProvenanceSource::Boundary).collect::<Vec<_>>();
        let mut expanded_inputs = HashSet::new();
        let mut expanded_outputs = HashSet::new();
        let mut leaves = Vec::new();
        let mut seen = HashSet::new();
        while let Some(source) = pending.pop() {
            let replacements = match source {
                ResidualProvenanceSource::Producer(producer) => vec![ResidualProvenanceLeaf::Producer(producer)],
                ResidualProvenanceSource::Boundary(RegionDataFlowSource::InstructionInput(index)) => {
                    self.resolve(ValueId::new(value.region(), instruction.inputs()[index]))?
                }
                ResidualProvenanceSource::Boundary(RegionDataFlowSource::RegionInput { region_index, input_index }) => {
                    if !expanded_inputs.insert((region_index, input_index)) {
                        continue;
                    }
                    match data_flow.input_sources(operation, region_index, input_index, boundary)? {
                        RegionDataFlowSources::Unknown => vec![ResidualProvenanceLeaf::Producer(value)],
                        RegionDataFlowSources::Known(sources) => {
                            // Complete each declared alternative before the next. Region input/output positions
                            // terminate cycles without changing the region-local symbolic summaries.
                            pending.extend(sources.into_iter().rev().map(ResidualProvenanceSource::Boundary));
                            Vec::new()
                        }
                    }
                }
                ResidualProvenanceSource::Boundary(RegionDataFlowSource::RegionOutput(origin)) => {
                    if !expanded_outputs.insert((origin.region_index, origin.output_index)) {
                        continue;
                    }
                    let region = instruction.regions()[origin.region_index];
                    let output = program.region(region)?.output_ids()[origin.output_index];
                    for leaf in self.resolve(ValueId::new(region, output))?.into_iter().rev() {
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
                    Vec::new()
                }
            };

            for leaf in replacements {
                if seen.insert(leaf) {
                    leaves.push(leaf);
                }
            }
        }

        Ok(leaves)
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

/// Plan of [`PartitionedProgram::with_residual_policy`] for one known atom that residual work demands.
enum ResidualPlan<T: Type> {
    /// The atom is an edge of the resulting partition, stored through the provided storage, if any.
    Edge(Option<ErasedResidualStorage<T>>),

    /// The atom is recomputed in the residual program by replaying its producer.
    Recompute,

    /// The atom is a constant, which the residual program re-creates.
    Constant,
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
    use std::sync::atomic::AtomicUsize;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayType, DataType, DimensionBounds,
        DimensionType, Memory,
    };
    use crate::differentiation::{MemoryTransferStorage, NothingSavable};
    use crate::operations::{
        AddOperation, CompareOperation, ComparisonDirection, ConditionOperation, CosOperation, CustomFunctionJvpRule,
        CustomFunctionOperation, DimensionSizeOperation, DotDimensionNumbers, DotOperation, ExpOperation,
        LinearCallOperation, MulOperation, NegOperation, ReducePrecisionOperation, ReferenceFreezeOperation,
        ReferenceNewOperation, ReferenceReadOperation, ReferenceWriteOperation, RematerializeOperation, ScanOperation,
        SinOperation, TagOperation, TransferToMemoryOperation, WhileOperation,
    };
    use crate::partial::values::PartialEvaluationOutput;
    use crate::programs::{
        InputRegionProvenance, OutputRegionProvenance, ReferenceType, RegionDataFlow, RegionInterface, RegionSlot,
        TypeError,
    };
    use crate::tests::TestRegionOperation;

    use super::*;

    type TestValue = ArrayIrValue<Array>;
    type TestOperation = ArrayIrOperation<Array>;
    type TestProgram = Program<TestValue, TestOperation, Vec<TestValue>, Vec<TestValue>>;
    type TestPartition = PartitionedProgram<TestValue, TestOperation>;

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
        /// Returns one native policy over the composite test universe with a fixed decision.
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
            partition.with_rounded_residuals(rounding, |_| false).unwrap().to_string(),
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
        let is_storage =
            |operation: &TestOperation| operation.projected_payload::<TransferToMemoryOperation>().is_some();
        assert_eq!(
            partition.with_rounded_residuals(rounding, is_storage).unwrap().to_string(),
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
        assert_eq!(partition.with_rounded_residuals(|_| None, |_| false).unwrap().to_string(), expected);

        // Forwarding would feed `x` to the residual program without passing through the edge that rounds it, so
        // rounding must precede forwarding.
        let forwarded = program.partition(&[true, false]).unwrap().forward_residuals().unwrap();
        assert!(forwarded.has_forwarded_residual_inputs());
        assert_eq!(
            forwarded.with_rounded_residuals(rounding, |_| false).map(|partition| partition.to_string()),
            Err(ProgramError::InvalidArgument {
                message: "cannot round the residuals of a partition whose residual inputs are already forwarded"
                    .to_string(),
            }),
        );
    }

    #[test]
    fn test_residual_policy_reference_place_residuals() {
        // Each demanded condition output preserves its own nested cuts: the second stays saved, while the first
        // recomputes only its sine and tag. A saved sibling does not force the first output across the boundary.
        let program = condition_program(true);
        let placed = save_names(&["second"], &[])
            .place_residuals(program.partition(&[true, true, false]).unwrap())
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
        let placed = save_nothing().place_residuals(program.partition(&[true, true, false]).unwrap()).unwrap();
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
        let placed = policy.place_residuals(partition).unwrap();
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
        // Two calls share a body whose coupled boundary stays intact, but consume different outputs. Their distinct
        // inputs prevent common-subexpression elimination from merging the calls. Check both orders so caching only
        // by region cannot reuse the non-rejecting output's result for the rejected output in either traversal order.
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let input = builder.add_input(scalar_type());
        let first = add(&mut builder, SinOperation::<ArrayType>::new().into(), vec![input]);
        let cosine = add(&mut builder, CosOperation::<ArrayType>::new().into(), vec![input]);
        let second = add(&mut builder, TagOperation::<ArrayType>::new("second").into(), vec![cosine]);
        let body = build(builder, vec![first, second]);
        for selected_outputs in [[0, 1], [1, 0]] {
            let mut branch = ProgramBuilder::<TestValue, TestOperation>::new();
            let inputs = [branch.add_input(scalar_type()), branch.add_input(scalar_type())];
            let body = branch.import_program(body.clone());
            let mut outputs = Vec::new();
            for (input, output_index) in inputs.into_iter().zip(selected_outputs) {
                let call = TestOperation::CustomFunction(CustomFunctionOperation::from_rule_regions(
                    CustomFunctionJvpRule::Primal,
                    false,
                ));
                outputs.push(branch.add_instruction(call, vec![body], vec![input], None).unwrap()[output_index]);
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
            reject.place_residuals(feedback_partition(3, &[0, 1])).map(|_| ()),
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
        let placed = reject.place_residuals(feedback_partition(3, &[1])).unwrap();
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
                        %3:f64[], %4:f64[], %5:f64[3] = scan [carry_count=2, length=3, reverse=false] %2 %1 [
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
        let placed = reject.place_residuals(feedback_partition(0, &[2])).unwrap();
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
                        %3:f64[], %4:f64[] = scan [carry_count=2, length=0, reverse=false] %2 %1 [
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
        let placed = reject.place_residuals(program.partition(&[true, true, false]).unwrap()).unwrap();
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
                            let %1:f64[] = scan [carry_count=1, length=0, reverse=false] %0 [
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
                            let %1:f64[] = scan [carry_count=1, length=0, reverse=false] %0 [
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
                .place_residuals(nested_scan_program(1, false).partition(&[true, true, false]).unwrap())
                .map(|_| ()),
            expected,
        );
        assert_eq!(
            reject
                .place_residuals(nested_scan_program(0, true).partition(&[true, true, false]).unwrap())
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
            reject.place_residuals(program.partition(&[true, false]).unwrap()).map(|_| ()),
            Err(ProgramError::from(ResidualPolicyError::Rejected {
                policy: "reject_dots".to_owned(),
                rejection: ResidualRejection::new("the predicate dot cannot be replayed"),
            })),
        );
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
        let mut provenance = ResidualProvenanceAnalysis::new(&program);
        for (value, producer) in values.into_iter().zip([first, second]) {
            assert_eq!(
                provenance.resolve(ValueId::new(program.entry(), value)).unwrap(),
                vec![ResidualProvenanceLeaf::Producer(ValueId::new(program.entry(), producer))],
            );
        }
        assert_eq!(provenance.resolved.len(), 83);
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
        let mut provenance = ResidualProvenanceAnalysis::new(&program);
        assert_eq!(
            provenance.resolve(ValueId::new(program.entry(), values[0])),
            Ok(vec![ResidualProvenanceLeaf::Producer(ValueId::new(program.entry(), first))]),
        );
        assert_eq!(
            provenance.resolve(ValueId::new(program.entry(), values[1])),
            Ok(vec![ResidualProvenanceLeaf::Producer(ValueId::new(program.entry(), second))]),
        );
        assert_eq!(queries.load(Ordering::Relaxed), 82);
        assert_eq!(provenance.resolved.len(), 83);

        // Repeating a resolution and asking for its candidate must use the already cached symbolic summary.
        assert_eq!(
            provenance.resolve(ValueId::new(program.entry(), values[0])),
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
        let mut provenance = ResidualProvenanceAnalysis::new(&program);
        assert_eq!(
            provenance.resolve(ValueId::new(program.entry(), output)),
            Ok(vec![
                ResidualProvenanceLeaf::Producer(ValueId::new(program.entry(), tagged)),
                ResidualProvenanceLeaf::Producer(ValueId::new(second, sine)),
            ]),
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
        let mut provenance = ResidualProvenanceAnalysis::new(&program);
        let first = ResidualProvenanceLeaf::Producer(ValueId::new(program.entry(), first));
        let second = ResidualProvenanceLeaf::Producer(ValueId::new(program.entry(), second));
        let dot = ResidualProvenanceLeaf::Producer(ValueId::new(body, dot));
        assert_eq!(provenance.resolve(ValueId::new(program.entry(), scan[0])).unwrap(), vec![first, second, dot]);
        assert_eq!(provenance.resolve(ValueId::new(program.entry(), scan[1])).unwrap(), vec![second, dot]);
        assert_eq!(provenance.resolve(ValueId::new(program.entry(), scan[2])).unwrap(), vec![first, second, dot]);
        assert_eq!(provenance.resolve(ValueId::new(program.entry(), empty[0])).unwrap(), vec![first]);
        assert_eq!(provenance.resolve(ValueId::new(program.entry(), empty[1])).unwrap(), vec![second]);
        assert_eq!(provenance.resolve(ValueId::new(program.entry(), empty[2])).unwrap(), vec![]);
        assert_eq!(provenance.resolve(ValueId::new(program.entry(), ordinary[0])).unwrap(), vec![second]);
        assert_eq!(provenance.resolve(ValueId::new(program.entry(), ordinary[2])).unwrap(), vec![first]);
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
        let mut provenance = ResidualProvenanceAnalysis::new(&program);
        assert_eq!(
            provenance.resolve(ValueId::new(program.entry(), outputs[0])).unwrap(),
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
            let mut provenance = ResidualProvenanceAnalysis::new(&known);
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
        let mut provenance = ResidualProvenanceAnalysis::new(&program);
        let missing = AtomId::new(1);
        assert_eq!(
            provenance.resolve(ValueId::new(program.entry(), missing)),
            Err(ProgramError::UnboundAtomId { id: missing }),
        );
    }
}
