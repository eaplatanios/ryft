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
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use thiserror::Error;

use crate::parameters::Placeholder;
use crate::partial::partitions::PartitionedProgram;
use crate::partial::values::PartialEvaluationInput;
use crate::programs::{
    Atom, AtomId, ErasedOperation, InputRegionProvenance, Operation, OperationPayloadProjection, Program,
    ProgramBuilder, ProgramError, RegionId, Type, Typed, Value, ValueId,
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
/// example, has one producer per branch). Policies return one decision for the complete candidate.
pub struct ResidualCandidate<'o, T: Type> {
    /// Operation outputs that may have produced the residual, in semantic order.
    producers: Vec<ResidualProducer<'o, T>>,

    /// Type of the residual.
    residual_type: T,
}

impl<'o, T: Type> ResidualCandidate<'o, T> {
    /// Creates a new [`ResidualCandidate`] with the provided producers, in semantic order, and residual type.
    #[inline]
    pub fn new(producers: Vec<ResidualProducer<'o, T>>, residual_type: T) -> Self {
        Self { producers, residual_type }
    }

    /// Returns the operation outputs that may have produced this residual, in semantic order.
    #[inline]
    pub fn producers(&self) -> &[ResidualProducer<'o, T>] {
        self.producers.as_slice()
    }

    /// Returns the type of this residual.
    #[inline]
    pub fn residual_type(&self) -> &T {
        &self.residual_type
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

/// Type-erased [`ResidualStorage`], which [`ResidualPolicyReference::classify`] returns.
pub type ErasedResidualStorage<T> = Arc<dyn ResidualStorage<T>>;

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
/// public. Its implementations are [`NativeResidualPolicy`], which wraps any [`ResidualPolicy`], and
/// [`LiftedResidualPolicy`].
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
        let residual_type = project(candidate.residual_type());
        if residual_type.is_none() && unprojectable.is_none() {
            unprojectable = Some(("the residual".to_owned(), candidate.residual_type().to_string()));
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

/// Instantiations of one [`ResidualPolicy`] in other type universes, which [`ResidualPolicy::native_instantiations`]
/// returns and [`ResidualPolicyReference::lift`] prefers over projecting candidate types. For example, a policy that is
/// generic over its type universe returns `NativeResidualPolicies::default().with::<ArrayType, _>(
/// self.clone()).with::<ArrayIrType, _>(self.clone())`.
///
/// An instantiation is _native_ to its universe because it is implemented for that universe and so classifies its
/// candidates in the universe's own types. This is the alternative to lifting a policy by projection, where the types
/// of each candidate are projected into the universe of the policy before the policy classifies them. Projection can
/// only handle candidates whose types project (e.g., when lifting from [`ArrayType`](crate::ArrayType) into
/// [`ArrayIrType`](crate::ArrayIrType), candidates that produce dimensions or references have no `ArrayType`
/// representation, and so a policy that saves everything could not classify them unless it has a native
/// `ArrayIrType` instantiation, which simply saves them).
#[derive(Clone, Default)]
pub struct NativeResidualPolicies {
    /// Instantiations keyed by the [`TypeId`] of their universe `U`, each holding an
    /// `Arc<dyn ErasedResidualPolicy<U>>`.
    entries: Vec<(TypeId, Arc<dyn Any + Send + Sync>)>,
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

    /// Returns the instantiation that is registered for type universe `U`, if any.
    fn get<U: 'static + Type>(&self) -> Option<Arc<dyn ErasedResidualPolicy<U>>> {
        self.entries
            .iter()
            .find(|(universe, _)| *universe == TypeId::of::<U>())
            .and_then(|(_, policy)| policy.downcast_ref::<Arc<dyn ErasedResidualPolicy<U>>>().cloned())
    }
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
    ///     Outputs of operations that forward region inputs are saved too, because replaying them would re-execute the
    ///     complete operation for a value that it merely forwards.
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
    /// gains only complete local reference lifecycles that nothing outside it can observe. A policy that saves every
    /// value therefore reproduces this partition, apart from dropping edges that the residual program does not read.
    ///
    /// # Errors
    ///
    /// Returns the errors of classifying candidates with `policy` (refer to [`ResidualPolicyReference::classify`]),
    /// [`ResidualPolicyError::UnsupportedStorage`] or [`ResidualPolicyError::InvalidStorage`] when a storage cannot be
    /// staged, and [`ResidualPolicyError::Program`] when rebuilding the programs fails (e.g., when residual work would
    /// require a local reference handle as a new edge).
    #[inline]
    pub fn with_residual_policy(self, policy: &ResidualPolicyReference<V::Type>) -> Result<Self, ResidualPolicyError> {
        self.with_residual_policy_and_region_replay(policy, true)
    }

    /// Places the residuals of this partition like [`with_residual_policy`](Self::with_residual_policy). When
    /// `replay_region_operations` is `false`, outputs of region-carrying operations are saved rather than replayed,
    /// because the split rules of those operations already placed the residuals of their bodies with the same policy.
    fn with_residual_policy_and_region_replay(
        self,
        policy: &ResidualPolicyReference<V::Type>,
        replay_region_operations: bool,
    ) -> Result<Self, ResidualPolicyError> {
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
                PartialEvaluationInput::Known(edge) if read[atom.index()] => Some(edges[*edge]),
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

        // An instruction of the known program can be replayed in the residual program only if it and its transitive
        // state predecessors (i.e., the earlier accesses to the local reference state that it observes) are all
        // recomputable. The answer is memoized per instruction, because many demanded atoms share predecessors.
        let mut replayable = vec![None; known_program.instructions().len()];
        let mut is_replayable = |index: usize| {
            if let Some(replayable) = replayable[index] {
                return replayable;
            }
            let mut pending = vec![index];
            let mut visited = HashSet::new();
            let mut is_replayable = true;
            while let Some(index) = pending.pop() {
                if !visited.insert(index) {
                    continue;
                }
                if !lifecycles.is_recomputable(index) {
                    is_replayable = false;
                    break;
                }
                pending.extend(lifecycles.state_predecessors(index));
            }
            replayable[index] = Some(is_replayable);
            is_replayable
        };

        // Plans one demanded known atom, applying the rules listed in the documentation of `with_residual_policy` in
        // order: constants and known inputs never consult the policy, reference handles are either original edges or
        // replayed, values whose producers cannot be replayed are saved, and the policy classifies everything else.
        // The provenance of the known program provides the candidates that the policy classifies, and memoizes the
        // provenance of region outputs across calls.
        let mut provenance = ResidualProvenance { program: &known_program, summaries: HashMap::new() };
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
                } else if is_replayable(index) {
                    Ok(ResidualPlan::Recompute)
                } else {
                    Err(ProgramError::MalformedProgram(format!(
                        "residual work requires the reference produced by operation `{}` as a new edge",
                        known_program.instructions()[index].operation().name(),
                    ))
                    .into())
                };
            }

            if !is_replayable(index) {
                return Ok(ResidualPlan::Edge(None));
            }

            // Replaying a region-carrying producer re-executes it as a whole, which would undo the per-iteration and
            // per-branch decisions of a split rule that already placed the residuals of its body with the same policy.
            if !replay_region_operations && !known_program.instructions()[index].regions().is_empty() {
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
                ResidualDecision::Recompute => ResidualPlan::Recompute,
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
                if matches!(atom_plan, ResidualPlan::Recompute) {
                    // The `unwrap` is safe because only atoms produced by instructions are ever recomputed.
                    pending_instructions.push(instruction_by_output[atom.index()].unwrap());
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
            PartialEvaluationInput::Known(edge) => Some(edges[*edge]),
            PartialEvaluationInput::Unknown(_) => None,
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
        let input_count = known_program.input_ids().len();
        let output_count = known_outputs.len();
        let new_known_program = builder
            .build::<Vec<V>, Vec<V>>(known_outputs, vec![Placeholder; input_count], vec![Placeholder; output_count])?
            .without_unobserved_local_references(&replayed_allocations)?
            .into_simplified()?;

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
                inputs.push(PartialEvaluationInput::Known(position));
            }
        };

        for (input, atom) in residual_inputs.iter().zip(residual_program.input_ids()) {
            match input {
                PartialEvaluationInput::Unknown(index) => {
                    let r#type = residual_program.atoms()[atom.index()].r#type().into_owned();
                    residual_atoms[atom.index()] = Some(builder.add_input(r#type));
                    new_residual_inputs.push(PartialEvaluationInput::Unknown(*index));
                }
                PartialEvaluationInput::Known(edge) => {
                    add_edge_input(edges[*edge], &mut builder, &mut new_residual_inputs)
                }
            }
        }

        for edge in &new_edges {
            add_edge_input(*edge, &mut builder, &mut new_residual_inputs);
        }

        // Resolves the atom of the residual program that provides the demanded known atom `atom`, memoized in
        // `known_atoms`: its edge input (through the restore operations of its storage, which are staged on
        // first use and must reproduce the residual type), its replayed value, or a re-created constant.
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
            if let PartialEvaluationInput::Known(edge) = input
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
            .into_simplified()?;

        let metadata = metadata.with_residual_inputs(new_residual_inputs);
        Ok(Self::from_programs_and_metadata(new_known_program, new_residual_program, metadata))
    }
}

// TODO(eaplatanios): Review from here onwards.

/// Placement of the residuals of the partitions of programs over values of type `V`
/// and operations of family `O` according to one residual policy. It is type-erased so that a
/// [`PartialEvaluationContext`](crate::PartialEvaluationContext) can carry a policy into the partitions that the
/// split rules of region-carrying operations construct, without requiring every operation family to support residual
/// placement. [`ResidualPolicyReference`]s implement it for every family that does.
pub(crate) trait ResidualPlacement<V: Value, O: Operation<Type = V::Type>> {
    /// Returns `partition` with its residuals placed according to the policy (refer to the documentation of
    /// [`PartitionedProgram::with_residual_policy`] for more information on that). Outputs of region-carrying
    /// operations in the known program are saved rather than replayed, because the split rules of those operations
    /// already placed the residuals of their bodies with the same policy when the partition was constructed.
    fn place_residuals(&self, partition: PartitionedProgram<V, O>) -> Result<PartitionedProgram<V, O>, ProgramError>;
}

impl<V: Value<Type: 'static>, O: Operation<Type = V::Type> + OperationPayloadProjection> ResidualPlacement<V, O>
    for ResidualPolicyReference<V::Type>
{
    #[inline]
    fn place_residuals(&self, partition: PartitionedProgram<V, O>) -> Result<PartitionedProgram<V, O>, ProgramError> {
        Ok(partition.with_residual_policy_and_region_replay(self, false)?)
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
}

// TODO(eaplatanios): Review from here onwards.

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

/// Stages the chain of storage operations that `payloads` describe on atom `atom` of `builder` and returns the atom of
/// its final result. Each payload is constructed in the operation family of `builder` and must be a pure unary
/// operation with a single result.
fn stage_storage<V: Value, O: Operation<Type = V::Type> + OperationPayloadProjection>(
    builder: &mut ProgramBuilder<V, O>,
    atom: AtomId,
    payloads: Vec<ErasedOperation>,
    storage: &dyn ResidualStorage<V::Type>,
) -> Result<AtomId, ResidualPolicyError>
where
    V::Type: 'static,
{
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
                message: format!("its operation `{name}` has {} results instead of one", outputs.len()),
            }),
        }
    })
}

/// Leaf of the symbolic provenance of a value, as resolved by [`ResidualProvenance`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
enum ProvenanceLeaf {
    /// Output of an instruction that produces it itself, rather than forwarding an output of an attached region.
    Producer(ValueId),

    /// Input at the provided position of the region that contains the value, which each call site of that region
    /// resolves through its own operands.
    Input(usize),
}

/// Resolution of the operation outputs that may have produced the values of a [`Program`], which looks through the
/// outputs of operations that forward the outputs of their attached regions (refer to
/// [`Operation::output_region_provenance`]) and through the inputs of those regions back to the operands that supply
/// them (refer to [`Operation::input_region_provenance`]).
///
/// Region outputs are summarized symbolically, in terms of the inputs of their regions, and each summary is computed
/// once and instantiated at every call site. A region that several operations invoke (e.g., with differently tagged
/// operands) therefore resolves to the producers of each call site's own operands.
struct ResidualProvenance<'p, V: Value, O: Operation<Type = V::Type>> {
    /// Program whose values are resolved.
    program: &'p Program<V, O, Vec<V>, Vec<V>>,

    /// Symbolic provenance of each region output that was summarized so far, keyed by region and output index.
    summaries: HashMap<(RegionId, usize), Vec<ProvenanceLeaf>>,
}

impl<'p, V: Value, O: Operation<Type = V::Type> + OperationPayloadProjection> ResidualProvenance<'p, V, O> {
    /// Returns the candidate for `value` with residual type `residual_type`, or [`None`] when every provenance path
    /// of `value` ends at an input or constant of the program.
    fn candidate(
        &mut self,
        value: ValueId,
        residual_type: V::Type,
    ) -> Result<Option<ResidualCandidate<'p, V::Type>>, ProgramError> {
        let program = self.program;
        let producers = self
            .resolve(value)?
            .into_iter()
            .filter_map(|leaf| match leaf {
                ProvenanceLeaf::Producer(value) => Some(value),
                ProvenanceLeaf::Input(_) => None,
            })
            .map(|value| {
                // The `unwrap`s are safe because producer leaves are always outputs of instructions.
                let instruction = program.instruction(program.producer(value)?.unwrap())?;
                let output_index = instruction.outputs().iter().position(|output| *output == value.atom()).unwrap();
                let region = program.region(value.region())?;
                let atom_type = |atom: &AtomId| region.atoms()[atom.index()].r#type().into_owned();
                Ok(ResidualProducer::new(
                    instruction.operation(),
                    output_index,
                    instruction.inputs().iter().map(atom_type).collect(),
                    instruction.outputs().iter().map(atom_type).collect(),
                ))
            })
            .collect::<Result<Vec<_>, ProgramError>>()?;
        Ok((!producers.is_empty()).then(|| ResidualCandidate::new(producers, residual_type)))
    }

    /// Returns the symbolic provenance of `value`, in semantic order and without duplicates.
    fn resolve(&mut self, value: ValueId) -> Result<Vec<ProvenanceLeaf>, ProgramError> {
        let program = self.program;
        let Some(instruction_id) = program.producer(value)? else {
            let region = program.region(value.region())?;
            let input = region.input_ids().iter().position(|input| *input == value.atom());
            return Ok(input.map(ProvenanceLeaf::Input).into_iter().collect());
        };
        let instruction = program.instruction(instruction_id)?;
        // The `unwrap` is safe because `value` is an output of the instruction that produces it.
        let output_index = instruction.outputs().iter().position(|output| *output == value.atom()).unwrap();
        let origins = instruction.operation().output_region_provenance(output_index);
        if origins.is_empty() {
            return Ok(vec![ProvenanceLeaf::Producer(value)]);
        }

        let mut leaves = Vec::new();
        for origin in origins {
            let region = *instruction.regions().get(origin.region_index).ok_or_else(|| {
                ProgramError::MalformedProgram(format!(
                    "operation `{}` declares provenance from its region {} but its instruction has {} regions",
                    instruction.operation().name(),
                    origin.region_index,
                    instruction.regions().len(),
                ))
            })?;
            for leaf in self.summary(region, origin.output_index)? {
                let leaves_of_leaf = match leaf {
                    ProvenanceLeaf::Producer(_) => vec![leaf],
                    ProvenanceLeaf::Input(input_index) => {
                        match instruction.operation().input_region_provenance(origin.region_index, input_index) {
                            InputRegionProvenance::Input { index } => {
                                let operand = *instruction.inputs().get(index).ok_or_else(|| {
                                    ProgramError::MalformedProgram(format!(
                                        "operation `{}` declares its operand {index} as the source of an input of its \
                                         region {} but its instruction has {} operands",
                                        instruction.operation().name(),
                                        origin.region_index,
                                        instruction.inputs().len(),
                                    ))
                                })?;
                                self.resolve(ValueId::new(value.region(), operand))?
                            }
                            InputRegionProvenance::None | InputRegionProvenance::Local => Vec::new(),
                        }
                    }
                };
                for leaf in leaves_of_leaf {
                    if !leaves.contains(&leaf) {
                        leaves.push(leaf);
                    }
                }
            }
        }
        Ok(leaves)
    }

    /// Returns the symbolic provenance of output `output_index` of `region`, which is computed once per region output.
    fn summary(&mut self, region: RegionId, output_index: usize) -> Result<Vec<ProvenanceLeaf>, ProgramError> {
        if let Some(summary) = self.summaries.get(&(region, output_index)) {
            return Ok(summary.clone());
        }
        let output =
            *self.program.region(region)?.output_ids().get(output_index).ok_or_else(|| {
                ProgramError::MalformedProgram(format!("region {region} has no output {output_index}"))
            })?;
        let summary = self.resolve(ValueId::new(region, output))?;
        self.summaries.insert((region, output_index), summary.clone());
        Ok(summary)
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayType, DataType, DimensionBounds,
        DimensionType, Memory,
    };
    use crate::operations::{
        ConditionOperation, CosOperation, DimensionSizeOperation, DotDimensionNumbers, DotOperation, ExpOperation,
        MulOperation, NegOperation, ReferenceFreezeOperation, ReferenceNewOperation, ReferenceReadOperation,
        ReferenceWriteOperation, SinOperation, TagOperation, TransferToMemoryOperation,
    };
    use crate::partial::values::PartialEvaluationOutput;
    use crate::programs::ReferenceType;

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

    /// Returns a dot product of two vectors.
    fn dot() -> ArrayOperation<Array> {
        DotOperation::new(DotDimensionNumbers::new(vec![0], vec![0], vec![], vec![])).into()
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

    /// Renders the known program, the residual program, and the residual inputs of `partition`.
    fn render(partition: &TestPartition) -> String {
        format!("{}\n{}\n{:?}", partition.known_program(), partition.residual_program(), partition.residual_inputs())
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
            .map(|input| match input {
                PartialEvaluationInput::Unknown(index) => inputs[*index].clone(),
                PartialEvaluationInput::Known(edge) => known_outputs[known_output_count + edge].clone(),
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

    #[test]
    fn test_residual_policy_error() {
        let rejection = ResidualRejection::new("never save `x`");
        let error = ResidualPolicyError::Rejected { policy: "names".to_owned(), rejection };
        assert_eq!(error.to_string(), "residual policy `names` rejected a residual: never save `x`");
        assert_eq!(ResidualPolicyError::from(ProgramError::from(error.clone())), error);

        // Program errors round trip through their dedicated variant.
        let program_error = ProgramError::MalformedProgram("broken".to_owned());
        assert_eq!(
            ResidualPolicyError::from(program_error.clone()),
            ResidualPolicyError::Program(program_error.clone())
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

        // Payloads are recognized through the projected array member of the operation family.
        assert!(producer.payload::<DotOperation>().is_some());
        assert!(producer.payload::<SinOperation<ArrayType>>().is_none());
    }

    #[test]
    fn test_residual_candidate() {
        let operation = TestOperation::from(dot());
        let producer = ResidualProducer::new(&operation, 0, vec![vector_type(), vector_type()], vec![scalar_type()]);
        let candidate = ResidualCandidate::new(vec![producer], scalar_type());
        assert_eq!(candidate.producers().len(), 1);
        assert_eq!(candidate.producers()[0].name(), "dot");
        assert_eq!(candidate.residual_type(), &scalar_type());
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
    fn test_residual_policy_reference() {
        let dots = save_dots(None);
        assert_eq!(dots.name(), "save_dots");
        assert_eq!(
            format!("{dots:?}"),
            format!("ResidualPolicyReference {{ name: \"save_dots\", id: {} }}", dots.id())
        );

        // References compare and hash by definition: clones are equal, while separately registered equal policies and
        // the opt-ins, which change what a policy decides, are distinct definitions.
        let hash = |reference: &ResidualPolicyReference<ArrayIrType>| {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            reference.hash(&mut hasher);
            hasher.finish()
        };
        assert_eq!(dots.clone(), dots);
        assert_eq!(hash(&dots.clone()), hash(&dots));
        assert_ne!(save_dots(None), dots);
        let native = dots.clone().with_native_instantiation::<ArrayType, _>(TestPolicy {
            name: "save_dots",
            classify: |_: &ResidualCandidate<'_, ArrayType>| Ok(ResidualDecision::<NoStorage>::Save),
        });
        assert_ne!(native.id(), dots.id());
        let fallback = dots.clone().with_projection_fallback::<ArrayType, _>(|_| Ok(ResidualDecision::Save));
        assert_ne!(fallback.id(), dots.id());
        assert_ne!(fallback.id(), native.id());
    }

    #[test]
    fn test_residual_policy_reference_classify() {
        let operation = TestOperation::from(dot());
        let producer = ResidualProducer::new(&operation, 0, vec![vector_type(), vector_type()], vec![scalar_type()]);
        let candidate = ResidualCandidate::new(vec![producer], scalar_type());
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
                rejection: ResidualRejection::new("never")
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
        let array_candidate = |operation| {
            ResidualCandidate::new(
                vec![ResidualProducer::new(operation, 0, vec![scalar_type()], vec![scalar_type()])],
                scalar_type(),
            )
        };
        let dimension_type =
            ArrayIrType::Dimension(DimensionType::new("n", DimensionBounds::non_negative(None).unwrap()));
        let dimension_size = TestOperation::DimensionSize(
            DimensionSizeOperation::new(&ArrayType::new_static(DataType::F64, [3]), 0).unwrap(),
        );
        let dimension_candidate = ResidualCandidate::new(
            vec![ResidualProducer::new(&dimension_size, 0, vec![vector_type()], vec![dimension_type.clone()])],
            dimension_type.clone(),
        );

        // Lifting by projection keeps the identity and the decisions of the source policy for projectable candidates,
        // including its storage, whose residual types are projected.
        let lifted = source.lift::<ArrayIrType>();
        assert_eq!(lifted.id(), source.id());
        assert_eq!(lifted.name(), "store_dots");
        let Ok(ResidualDecision::SaveWith(storage)) = lifted.classify(&array_candidate(&dot)) else {
            panic!("expected a stored residual");
        };
        assert_eq!(storage.name(), "negation");
        assert_eq!(storage.store_payloads(&scalar_type()).unwrap().len(), 1);
        assert_eq!(
            storage.store_payloads(&dimension_type).map(|_| ()),
            Err(ResidualPolicyError::UnsupportedStorage {
                storage: "negation".to_owned(),
                residual_type: dimension_type.to_string(),
                message: "the type does not project into the type universe of the storage".to_owned(),
            }),
        );
        assert!(matches!(lifted.classify(&array_candidate(&sine)), Ok(ResidualDecision::Recompute)));

        // Candidates that do not project are unsupported unless the policy opts into classifying them.
        assert_eq!(
            lifted.classify(&dimension_candidate).map(|_| ()),
            Err(ResidualPolicyError::UnsupportedProjection {
                policy: "store_dots".to_owned(),
                position: "the output 0 of producer `dimension_size`".to_owned(),
                residual_type: dimension_type.to_string(),
            }),
        );
        let fallback = source
            .clone()
            .with_projection_fallback::<ArrayIrType, _>(
                |candidate: &ProjectionFallbackCandidate<'_, '_, ArrayIrType>| {
                    assert_eq!(candidate.producer_projectable(), &[false]);
                    assert!(!candidate.residual_projectable());
                    assert_eq!(candidate.candidate().producers()[0].name(), "dimension_size");
                    Ok(ResidualDecision::Save)
                },
            )
            .lift::<ArrayIrType>();
        assert!(matches!(fallback.classify(&dimension_candidate), Ok(ResidualDecision::Save)));
        assert!(matches!(fallback.classify(&array_candidate(&dot)), Ok(ResidualDecision::SaveWith(_))));

        // A native instantiation classifies every candidate of its universe.
        let native = source.with_native_instantiation::<ArrayIrType, _>(TestPolicy {
            name: "store_dots",
            classify: |candidate: &ResidualCandidate<'_, ArrayIrType>| {
                Ok(match candidate.residual_type() {
                    ArrayIrType::Dimension(_) => ResidualDecision::<NoStorage>::Save,
                    _ => ResidualDecision::Recompute,
                })
            },
        });
        let lifted = native.lift::<ArrayIrType>();
        assert_eq!(lifted.id(), native.id());
        assert!(matches!(lifted.classify(&dimension_candidate), Ok(ResidualDecision::Save)));
        assert!(matches!(lifted.classify(&array_candidate(&dot)), Ok(ResidualDecision::Recompute)));
    }

    #[test]
    fn test_partitioned_program_with_residual_policy() {
        let program = sin_dot_program();
        let partition = program.partition(&[true, false]).unwrap();
        let expected = program.interpret(sin_dot_inputs()).unwrap();

        // Saving everything reproduces the partition, which saves the cosine.
        let planned = program.partition(&[true, false]).unwrap().with_residual_policy(&save_everything()).unwrap();
        assert_eq!(render(&planned), render(&partition));
        assert_eq!(
            render(&planned),
            indoc! {"
                lambda %0:f64[3] .
                let %1:f64[] = dot [
                    dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                ] %0 %0
                    %2:f64[] = sin %1
                    %3:f64[] = cos %1
                in (%2, %3)
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = mul %1 %0
                in (%2)
                [Unknown(1), Known(0)]"},
        );

        // Saving only dot products saves the dot product and recomputes its cosine in the residual program.
        let planned = partition.with_residual_policy(&save_dots(None)).unwrap();
        assert_eq!(
            render(&planned),
            indoc! {"
                lambda %0:f64[3] .
                let %1:f64[] = dot [
                    dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                ] %0 %0
                    %2:f64[] = sin %1
                in (%2, %1)
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = cos %1
                    %3:f64[] = mul %2 %0
                in (%3)
                [Unknown(1), Known(0)]"},
        );
        assert_eq!(run(&planned, &sin_dot_inputs()), expected);

        // Saving nothing recomputes the dot product too, which saves the known input that it needs.
        let planned = program.partition(&[true, false]).unwrap().with_residual_policy(&save_nothing()).unwrap();
        assert_eq!(
            render(&planned),
            indoc! {"
                lambda %0:f64[3] .
                let %1:f64[] = dot [
                    dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                ] %0 %0
                    %2:f64[] = sin %1
                in (%2, %0)
                lambda %0:f64[], %1:f64[3] .
                let %2:f64[] = dot [
                    dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                ] %1 %1
                    %3:f64[] = cos %2
                    %4:f64[] = mul %3 %0
                in (%4)
                [Unknown(1), Known(0)]"},
        );
        assert_eq!(run(&planned, &sin_dot_inputs()), expected);

        // Rejections of the policy fail planning.
        let rejecting = policy("rejecting", |_| Err::<ResidualDecision<NoStorage>, _>(ResidualRejection::new("never")));
        assert_eq!(
            program.partition(&[true, false]).unwrap().with_residual_policy(&rejecting).map(|_| ()).unwrap_err(),
            ResidualPolicyError::Rejected {
                policy: "rejecting".to_owned(),
                rejection: ResidualRejection::new("never")
            },
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
        let planned = program.partition(&[true, false]).unwrap().with_residual_policy(&save_everything()).unwrap();
        assert_eq!(
            render(&planned),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = exp %0
                in (%1, %1)
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = mul %1 %0
                in (%2)
                [Unknown(1), Known(0)]"},
        );
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

    #[test]
    fn test_partitioned_program_with_residual_policy_decides_per_producer_output() {
        let inputs = vec![
            ArrayIrValue::Array(Array::scalar(true).unwrap()),
            ArrayIrValue::Array(Array::scalar(0.5f64).unwrap()),
            ArrayIrValue::Array(Array::scalar(2.0f64).unwrap()),
        ];
        let plan = |first_consumed_first: bool, policy: ResidualPolicyReference<ArrayIrType>| {
            let program = condition_program(first_consumed_first);
            let planned = program.partition(&[true, true, false]).unwrap().with_residual_policy(&policy).unwrap();
            assert_eq!(run(&planned, &inputs), program.interpret(inputs.clone()).unwrap());
            render(&planned)
        };

        // Saving one output of the condition and recomputing the other replays the complete condition, whose saved
        // output still resolves to its edge in either demand order. Storing one output instead stages its storage, and
        // saving or storing both outputs replays nothing.
        let renderings = [
            plan(true, save_names(&["first"], &[])),
            plan(false, save_names(&["first"], &[])),
            plan(true, save_names(&[], &["first"])),
            plan(true, save_names(&["first"], &["second"])),
            plan(true, save_nothing()),
        ];
        assert_eq!(
            renderings.join("\n\n"),
            indoc! {"
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
                in (%2, %0, %1)
                lambda %0:f64[], %1:f64[], %2:bool[], %3:f64[] .
                let %4:f64[] = mul %1 %0
                    %5:f64[], %6:f64[] = condition %2 %3 [
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
                    %7:f64[] = mul %6 %0
                in (%4, %7)
                [Unknown(2), Known(0), Known(1), Known(2)]

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
                in (%2, %0, %1)
                lambda %0:f64[], %1:f64[], %2:bool[], %3:f64[] .
                let %4:f64[], %5:f64[] = condition %2 %3 [
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
                    %6:f64[] = mul %5 %0
                    %7:f64[] = mul %1 %0
                in (%6, %7)
                [Unknown(2), Known(0), Known(1), Known(2)]

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
                    %4:f64[] = neg %2
                in (%4, %0, %1)
                lambda %0:f64[], %1:f64[], %2:bool[], %3:f64[] .
                let %4:f64[] = neg %1
                    %5:f64[] = mul %4 %0
                    %6:f64[], %7:f64[] = condition %2 %3 [
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
                    %8:f64[] = mul %7 %0
                in (%5, %8)
                [Unknown(2), Known(0), Known(1), Known(2)]

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
                lambda %0:f64[], %1:f64[], %2:f64[] .
                let %3:f64[] = mul %1 %0
                    %4:f64[] = neg %2
                    %5:f64[] = mul %4 %0
                in (%3, %5)
                [Unknown(2), Known(0), Known(1)]

                lambda %0:bool[], %1:f64[] .
                in (%0, %1)
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
                [Unknown(2), Known(0), Known(1)]"},
        );
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
        let planned = program.partition(&[true, false]).unwrap().with_residual_policy(&save_nothing()).unwrap();
        assert_eq!(
            render(&planned),
            indoc! {"
                lambda %0:ref<f64[]> .
                let %1:f64[] = reference_read %0
                    %2:f64[] = cos %1
                    () = reference_write %0 %2
                    %3:f64[] = reference_read %0
                in (%1, %3)
                lambda %0:f64[], %1:f64[], %2:f64[] .
                let %3:f64[] = sin %1
                    %4:f64[] = mul %3 %0
                    %5:f64[] = sin %2
                    %6:f64[] = mul %5 %0
                in (%4, %6)
                [Unknown(1), Known(0), Known(1)]"},
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
            render(&partition),
            indoc! {"
                lambda %0:f64[] .
                let %1:ref<f64[]> = reference_new %0
                    %2:f64[] = reference_read %1
                    %3:f64[] = sin %0
                    () = reference_write %1 %3
                    %4:f64[] = reference_read %1
                    %5:f64[] = reference_freeze %1
                in (%2, %4, %5)
                lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[] .
                let %4:f64[] = mul %1 %0
                    %5:f64[] = mul %2 %0
                    %6:f64[] = mul %3 %0
                in (%4, %5, %6)
                [Unknown(1), Known(0), Known(1), Known(2)]"},
        );
        let planned = partition.with_residual_policy(&save_nothing()).unwrap();
        assert_eq!(
            render(&planned),
            indoc! {"
                lambda %0:f64[] .
                in (%0)
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
                [Unknown(1), Known(0)]"},
        );
        let inputs = vec![
            ArrayIrValue::Array(Array::scalar(0.5f64).unwrap()),
            ArrayIrValue::Array(Array::scalar(2.0f64).unwrap()),
        ];
        assert_eq!(run(&planned, &inputs), program.interpret(inputs.clone()).unwrap());
    }

    #[test]
    fn test_partitioned_program_with_residual_policy_stages_storage() {
        // The known program stores the dot product and the residual program restores it before the cosine uses it.
        let program = sin_dot_program();
        let planned = program
            .partition(&[true, false])
            .unwrap()
            .with_residual_policy(&save_dots(Some(NegationStorage)))
            .unwrap();
        assert_eq!(
            render(&planned),
            indoc! {"
                lambda %0:f64[3] .
                let %1:f64[] = dot [
                    dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                ] %0 %0
                    %2:f64[] = sin %1
                    %3:f64[] = neg %1
                in (%2, %3)
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = neg %1
                    %3:f64[] = cos %2
                    %4:f64[] = mul %3 %0
                in (%4)
                [Unknown(1), Known(0)]"},
        );
        assert_eq!(run(&planned, &sin_dot_inputs()), program.interpret(sin_dot_inputs()).unwrap());

        // Storage that does not reproduce the residual type or whose payloads the operation family cannot hold fails.
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
                .map(|_| ())
                .unwrap_err(),
            ResidualPolicyError::InvalidStorage {
                storage: "forgetful".to_owned(),
                message: "its restore operations produce `f64[]@Host[Pinned]` instead of the residual type `f64[]`"
                    .to_owned()
            },
        );
        assert_eq!(
            program
                .partition(&[true, false])
                .unwrap()
                .with_residual_policy(&store_dots(UnsupportedStorage))
                .map(|_| ())
                .unwrap_err(),
            ResidualPolicyError::UnsupportedStorage {
                storage: "unsupported".to_owned(),
                residual_type: "f64[]".to_owned(),
                message: format!(
                    "the operation family of the program cannot hold its payload `{}`",
                    std::any::type_name::<TagOperation<ArrayIrType>>(),
                ),
            },
        );
    }

    #[test]
    fn test_partitioned_program_with_residual_policy_resolves_provenance_through_shared_regions() {
        // Three levels of conditions share their branch regions and forward their operands: the innermost region
        // returns its input, and each enclosing region returns the output of a condition over the next region. Two
        // top-level conditions invoke the shared regions with differently tagged operands, so the provenance of each
        // resolves to the producer of its own operand.
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
        let planned = program
            .partition(&[true, true, false])
            .unwrap()
            .with_residual_policy(&save_names(&["u"], &[]))
            .unwrap();
        assert_eq!(
            render(&planned),
            indoc! {"
                lambda %0:bool[], %1:f64[] .
                let %2:f64[] = sin %1
                    %3:f64[] = tag [key=u] %2
                    %4:f64[] = condition %0 %0 %3 [
                        true={
                            lambda %0:bool[], %1:f64[] .
                            let %2:f64[] = condition %0 %0 %1 [
                                true=^1={
                                    lambda %0:bool[], %1:f64[] .
                                    let %2:f64[] = condition %0 %0 %1 [
                                        true=^0={
                                            lambda %0:bool[], %1:f64[] .
                                            in (%1)
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
                                    let %2:f64[] = condition %0 %0 %1 [
                                        true=^3={
                                            lambda %0:bool[], %1:f64[] .
                                            in (%1)
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
                                    let %2:f64[] = condition %0 %0 %1 [
                                        true=^0={
                                            lambda %0:bool[], %1:f64[] .
                                            in (%1)
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
                                    let %2:f64[] = condition %0 %0 %1 [
                                        true=^3={
                                            lambda %0:bool[], %1:f64[] .
                                            in (%1)
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
                [Unknown(2), Known(0), Known(1), Known(2)]"},
        );
        let inputs = vec![
            ArrayIrValue::Array(Array::scalar(false).unwrap()),
            ArrayIrValue::Array(Array::scalar(0.5f64).unwrap()),
            ArrayIrValue::Array(Array::scalar(2.0f64).unwrap()),
        ];
        assert_eq!(run(&planned, &inputs), program.interpret(inputs.clone()).unwrap());
    }

    #[test]
    fn test_partitioned_program_with_residual_policy_reproduces_partitions_that_save_everything() {
        let mut programs = vec![(sin_dot_program(), vec![true, false])];
        programs.extend(
            [true, false]
                .map(|first_consumed_first| (condition_program(first_consumed_first), vec![true, true, false])),
        );
        for (program, input_known) in programs {
            let partition = program.partition(&input_known).unwrap();
            let rendering = render(&partition);
            let planned = partition.with_residual_policy(&save_everything()).unwrap();
            assert_eq!(render(&planned), rendering);
        }
    }
}
