//! Staged calls of flat programs. A [`CallOperation`] calls a callee program that is attached to its instruction as a
//! region, and it carries the boundary of the function whose program that callee is (i.e., the sharding overrides of
//! the inputs and outputs of the function and the devices that the function names), together with what only the call
//! knows about where its inputs are placed. This module owns the backend-neutral contract of a call, which is its
//! payload and boundary, its type rule, the analysis of the placement requests of its callee (refer to
//! [`RegionPlacementRequests`]), and its rules for reference discharge, interpretation, partial evaluation, batching,
//! differentiation, and transposition. Every rule keeps a call boundary on the calls that it derives instead of
//! inlining the callee, so a transformed call still runs its callee as one function.
//!
//! A backend lowers calls and applies them on its own runtime. It reads the boundary through the functions of
//! [`CallOperation`] (e.g., [`CallOperation::declared_input_sharding`]), so that every entry point of the backend
//! agrees with these rules on where a call places its inputs. The interpretation of a call in the eager contexts
//! of this crate (i.e., [`EagerContext`] and the [`LockstepDomain`] of the `shard_map` emulation) is the reference
//! semantics of a call, which runs the callee region on the inputs that the type rule of the call accepts and returns
//! the outputs of the callee. A backend with a device runtime implements the interpretation for its own context instead
//! (e.g., by compiling the callee and executing it).

use std::any::TypeId;
use std::borrow::Cow;
use std::collections::BTreeSet;
use std::convert::Infallible;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::sync::Arc;

use crate::arrays::{
    Array, ArrayBatch, ArrayBatchingPolicy, ArrayIrBatch, ArrayIrBatchingPolicy, ArrayIrType, ArrayIrValue,
    ArrayMemberType, ArrayType, DeviceMesh, LogicalMesh, MeshAxisType, Sharding, ShardingDimension, UnreducedKind,
};
use crate::batching::{
    BatchAxis, BatchableOperation, BatchedProgram, BatchingContext, BatchingDriver, BatchingError,
    ProgramBatchingOutputAxesPolicy,
};
use crate::contexts::{Context, Domain, EagerContext, StagingContext};
use crate::differentiation::{
    CotangentAccumulator, DifferentiableOperation, DifferentiableType, DifferentiationContext, DifferentiationDriver,
    DifferentiationDual, DifferentiationError, DifferentiationPolicy, ResidualZeroProvider, TangentActivity,
    TransposableOperation, TransposedProgramCotangent, TranspositionContext, TranspositionDriver,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver, MemberInterpretableOperation};
use crate::macros::check_count;
use crate::operations::arithmetic::AddOperation;
use crate::operations::assertions::GuardOperation;
use crate::operations::collectives::parallel_reduce::ParallelReduceOperation;
use crate::operations::collectives::parallel_vary::ParallelVaryOperation;
use crate::operations::constants::zero::Zero;
use crate::operations::control_flow::scan::ScanOperation;
use crate::operations::manipulation::conversions::CastOperation;
use crate::operations::reductions::ReductionKind;
use crate::operations::references::ReferenceNewOperation;
use crate::operations::sharding::constrain_sharding::ConstrainShardingOperation;
use crate::operations::sharding::place::{PlaceOperation, PlacementRequest};
use crate::operations::sharding::reshard::{Reshard, ReshardOperation};
use crate::operations::sharding::shard_map::{LockstepDomain, LockstepOperation, LockstepValue};
use crate::operations::sharding::validate_preserved_mesh_axes;
use crate::partial::{
    PartialEvaluationContext, PartialEvaluationDriver, PartialEvaluationOutput, PartialEvaluationValue, PartialValue,
    PartiallyEvaluatableOperation, PartitionedProgram, ResidualDemandSplit, ResidualInputSource,
    ResidualPolicyReference,
};
use crate::programs::{
    AttachedRegions, InputRegionProvenance, Instruction, MaybeZero, Operation, OperationBoundaryPruning,
    OperationFormatter, OperationPayloadProjection, OperationProjection, OperationProvider, OutputRegionProvenance,
    Program, ProgramError, ReferenceDischargeContext, ReferenceDischargeDriver, ReferenceDischargePolicy,
    ReferenceDischargeValue, ReferenceDischargeableOperation, Region, RegionDataFlow, RegionInterface, RegionLiveness,
    RegionRef, RegionRole, RegionSlot, Transform, TransformArtifact, Type, TypeError, TypeIdentityRenaming,
    TypeRefinements, Typed, TypedProgram, Value, ValueProjection,
};
use crate::tracing::{Tracer, TracingContext};

/// Canonical operation name for [`CallOperation`].
pub const CALL_OPERATION_NAME: &str = "call";

/// Staged call to a flat program. The callee program is not part of this payload. It is a shared callee root [`Region`]
/// attached to the [`Instruction`] applying the operation (the single `["callee"]` slot), interned by [`Arc`] identity
/// when the call is staged through the [`AttachedRegions`] passed to [`Context::bind`], so repeated calls staged from
/// one function handle share one callee root and remain identity-comparable for the deduplication of callees by a
/// backend that lowers them. The `T` parameter fixes the callee boundary's type universe, allowing the reusable
/// homogeneous-array batching form and the executable array IR form to remain distinct payload types with one
/// [`Operation`] contract each. The retained `capture_count` names the callee's exact leading lifted-capture input
/// prefix. It participates in operation equality and callee deduplication identity and scopes reference capture
/// resolution during analysis.
///
/// [`Operation::region_data_flow`] declares ordinary callee data flow, in which callee inputs receive instruction
/// inputs positionally and instruction outputs originate from the corresponding callee outputs. Capture-prefix
/// retention remains an independent boundary-pruning requirement.
///
/// Declare callee-local result identities through [`with_callee_type_signature`](Self::with_callee_type_signature)
/// before binding the call. The operation allocates caller-owned result names once, so repeated inference retains
/// them while separate calls receive independent local result identities.
///
/// # Calls of Compiled Functions
///
/// A call stages the callee's source in its caller. When an input or output has a sharding override, or the body
/// requests placement through a `place` whose target names devices, a `constrain_sharding`, or a nested call that names
/// devices, the call uses the callee's mesh. Otherwise, including when its body only moves a value to another memory,
/// the backend that runs the call decides where it runs. An eager application of the call by a backend reads the mesh
/// of its function (refer to [`Self::function_mesh`]), and a nested call follows its caller. Lifted captures follow
/// those devices, even when the original array is committed.
///
/// A call therefore carries the boundary of its callee as that function was compiled, which is the sharding override
/// of each of its inputs and outputs, if any, together with the device mesh that the function names, if any. Neither
/// can be read from the callee body where the call is used (e.g., a replicated output override over auto mesh axes
/// stages nothing, and a callee inlined into a caller on other devices no longer shows where it was compiled to run).
/// A call of a `jit` dispatcher additionally records that it keeps each input without an override placed as it is
/// (refer to [`Self::with_kept_input_placements`]). Each entry point honors the boundary as follows:
///
///   - A backend that lowers or applies a call places the inputs and the outputs that have overrides as an execution
///     of the function places its arguments and results, and every other input at its effective boundary sharding
///     (refer to [`Self::boundary_input_sharding`]). The backend states how it does so on its own implementations.
///   - The type rule of the call accepts every input whose type refines the callee input type up to the placement
///     that its sharding describes, so every context that binds a call accepts the same inputs (refer to
///     [`Self::validate_input_types`]). It rejects an input whose unreduced or reduced axes differ from those of the
///     callee input type, which an execution would reshard where it can, because a call does not reshard its inputs.
///     An input of a `shard_map` body may vary along manual mesh axes that the callee input type does not declare,
///     and every output of the call then varies along them too.
///   - Program transforms keep the overrides of every input and output that the derived calls keep, together with
///     the device mesh and the kept input placements of the call (e.g., the known and residual calls of a partial
///     evaluation split, or a batched call). A call that a transform creates (e.g., the primal and tangent calls
///     of a linearization, the known and residual calls of a split, or a transposed call) also keeps the effective
///     boundary of every input of the original call that it receives, and places no input that the transform adds
///     (e.g., a residual), because its declared input types are only traced types. Partial evaluation treats a call
///     as forwarding one of its inputs only where the placement decision of every application returns that input
///     unchanged.
///
/// The boundary takes part in equality and hashing, but a call of a function that names no devices and that is not
/// a `jit` dispatcher hashes and renders exactly as it did before boundaries existed. A call that a transform derives
/// always renders the boundary that it keeps, so that it never renders like an ordinary call, which places its inputs
/// differently. A backend that deduplicates callees by their structure still compares the calls that they contain by
/// equality.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CallOperation<T: Type> {
    /// Number of leading callee inputs that form its lifted lexical capture prefix.
    capture_count: usize,

    /// Caller-owned result identities and the formal input scope used to instantiate forwarded names.
    output_types: Option<CallTypeBindings<T>>,

    /// Sharding override of each callee input, by position, or [`None`] when no callee input has one.
    input_shardings: Option<Arc<[Option<Sharding>]>>,

    /// Sharding override of each callee output, by position, or [`None`] when no callee output has one.
    output_shardings: Option<Arc<[Option<Sharding>]>>,

    /// Device mesh that the callee was compiled for, on which its sharding overrides are placed, when the callee names
    /// devices (refer to [`Self::with_boundary`]), or [`None`] when it does not.
    mesh: Option<DeviceMesh>,

    /// Whether the call leaves each input without a sharding override placed as it is, as the executables of a `jit`
    /// dispatcher are specialized for. Refer to [`Self::with_kept_input_placements`] for more information.
    keeps_input_placements: bool,

    /// Effective boundary sharding of each callee input without a sharding override, by position, for a call that a
    /// program transform derives from another call, or [`None`] for a call of a compiled function, whose declared input
    /// types define it. Refer to [`Self::boundary_input_sharding`] for more information.
    derived_input_shardings: Option<Arc<[Option<Sharding>]>>,

    /// Device mesh that the called function was compiled for (refer to [`Self::with_function_mesh`]), or [`None`] for
    /// a call that a program transform creates. Only the placement decision of an eager application by a backend reads
    /// it, so it takes part in equality but is neither hashed nor rendered, and staged programs and their lowering do
    /// not use it to select the execution mesh. It is separate from [`Self::mesh`], which type inference and transforms
    /// read as the devices that the call names. A transform keeps it on a call that replaces the whole call in the same
    /// context (i.e., the fused JVP call and a batched call), on the calls of a linearization (i.e., the primal call,
    /// the tangent call, and the transposed call that a pullback derives from them), which an eager VJP applies on its
    /// eager context, and on a call that partial evaluation folds into an eager known side or places into the residual
    /// program of an eager known side (e.g., the tangent call of an eager linearization, or the residual call of a
    /// split there), which an eager backend applies as well. The tangent call, and so the transposed call, keep it only
    /// for an eager linearization. The known call of an eager split keeps a named function mesh, whose devices every
    /// application must use. Calls staged into known and residual programs under a staging context follow the enclosing
    /// caller.
    function_mesh: Option<DeviceMesh>,

    /// Type universe of the callee boundary.
    marker: PhantomData<fn() -> T>,
}

impl<T: Type> CallOperation<T> {
    /// Creates a new [`CallOperation`]. The flat callee program is supplied as a shared region attachment to
    /// [`Context::bind`], and `capture_count` identifies its exact leading lifted-capture prefix.
    #[inline]
    pub fn new(capture_count: usize) -> Self {
        Self {
            capture_count,
            output_types: None,
            input_shardings: None,
            output_shardings: None,
            mesh: None,
            keeps_input_placements: false,
            derived_input_shardings: None,
            function_mesh: None,
            marker: PhantomData,
        }
    }

    /// Returns this [`CallOperation`] with the boundary of its callee, made of `mesh`, the device mesh that the
    /// callee was compiled for when it names devices, and of the sharding override of each callee input and output,
    /// by position, where [`None`] marks a position without one. A callee names devices when it has a sharding override
    /// or its program requests a placement that names devices (refer to [`RegionPlacementRequests::has_requests`]),
    /// so `mesh` must be present whenever an override is. A missing override is implicit past the end of either list.
    /// A list without overrides is stored as absent.
    pub fn with_boundary(
        mut self,
        mesh: Option<&DeviceMesh>,
        input_shardings: Vec<Option<Sharding>>,
        output_shardings: Vec<Option<Sharding>>,
    ) -> Self {
        let normalize = |shardings: Vec<Option<Sharding>>| {
            shardings.iter().any(Option::is_some).then(|| Arc::<[Option<Sharding>]>::from(shardings))
        };

        self.input_shardings = normalize(input_shardings);
        self.output_shardings = normalize(output_shardings);

        debug_assert!(mesh.is_some() || !self.has_boundary_shardings(), "sharding overrides require a mesh");

        self.mesh = mesh.cloned();
        self
    }

    /// Returns this [`CallOperation`] marked as a call of a Just-In-Time (JIT) compilation dispatcher, whose
    /// executables are specialized for the placement of each of their inputs without a sharding override, including
    /// over the auto mesh axes that input types do not record. Such a call leaves those inputs placed as they are, so
    /// that an eager call of a dispatcher places them as an execution of the dispatcher does. A call that is not marked
    /// is a call of a function compiled ahead of time for its input types, whose executable places such an input as its
    /// type records, and an eager call places it the same way.
    #[inline]
    pub fn with_kept_input_placements(mut self) -> Self {
        self.keeps_input_placements = true;
        self
    }

    /// Returns this call without the device mesh that it records, for a call that names no devices.
    /// Refer to [`Self::names_devices`] for more information.
    #[inline]
    pub fn without_mesh(&self) -> Self {
        Self { mesh: None, ..self.clone() }
    }

    /// Returns this call with `mesh`, the device mesh that the called function was compiled for, which decides where
    /// an eager application of the call by a backend runs (refer to [`Self::function_mesh`]). A call nested in another
    /// compiled function runs where its caller runs whatever this mesh is.
    #[inline]
    pub fn with_function_mesh(mut self, mesh: &DeviceMesh) -> Self {
        self.function_mesh = Some(mesh.clone());
        self
    }

    /// Returns this call without the mesh of its function, for a call that a transform places into a program,
    /// where it follows its caller.
    #[inline]
    pub fn without_function_mesh(&self) -> Self {
        Self { function_mesh: None, ..self.clone() }
    }

    /// Returns this call as a call that a program transform derived from another call, whose callee input at each
    /// position without a sharding override has the effective boundary sharding at that position of `shardings`, where
    /// [`None`] marks an input that the call leaves where it is (refer to [`Self::derived_input_shardings`]). Such a
    /// call places the inputs that it receives from the original call as that call does, and leaves every other input
    /// where it is, instead of placing it as the declared input types of its callee, which are only traced types, say.
    #[inline]
    pub fn with_derived_input_shardings(mut self, shardings: Vec<Option<Sharding>>) -> Self {
        self.derived_input_shardings = Some(Arc::from(shardings));
        self
    }

    /// Instantiates a sealed callee's result names, using its cached signature to recognize identity-free boundaries
    /// without cloning their input and output types. An identity-bearing callee follows the ordinary signature path.
    fn with_callee_region<V: Value<Type = T>, O: Operation<Type = T>>(
        mut self,
        callee: RegionRef<'_, V, O>,
    ) -> Result<Self, TypeError> {
        self.validate_capture_prefix(callee.input_ids().len())?;
        if callee.type_identity_signature().identities().is_empty() {
            // A closed signature without identities proves that `CallTypeBindings::fresh` would return `None` before
            // attempting any renaming. Preserve that exact payload, including when replacing an older signature.
            self.output_types = None;
            return Ok(self);
        }
        self.with_callee_type_signature(&callee.input_types(), &callee.output_types())
    }

    /// Returns this call instantiated for one call site of a callee with the exact signature `input_types` and
    /// `output_types`. Every callee-local result identity is allocated freshly and owned by the call instruction, and
    /// every input identity continues to name the forwarded quantity. Any earlier instantiation is replaced, so a call
    /// that is built once for a callee can be instantiated again for each of its call sites. This is necessary because
    /// sharing a callee root does not make the runtime results of separate calls the same quantity, and a region may
    /// define each identity only once. Call this after constructing the callee and before binding a call with local
    /// results. Allocation happens here, so repeated inference allocates no identities and returns the same result
    /// types.
    ///
    /// # Errors
    ///
    /// Returns a type error for an oversized capture prefix or a type that cannot rename its local identities.
    #[inline]
    pub fn with_callee_type_signature(mut self, input_types: &[T], output_types: &[T]) -> Result<Self, TypeError> {
        self.validate_capture_prefix(input_types.len())?;
        self.output_types = CallTypeBindings::fresh(input_types, output_types)?;
        Ok(self)
    }

    /// Retains caller-owned result identities after a callee boundary rewrite, including reference discharge.
    #[inline]
    fn with_caller_output_types(
        mut self,
        input_types: &[T],
        callee_outputs: &[T],
        caller_output_types: Vec<T>,
    ) -> Result<Self, TypeError> {
        self.output_types = CallTypeBindings::with_outputs(input_types, callee_outputs, caller_output_types)?;
        Ok(self)
    }

    /// Returns `derived` with the boundary shardings of this call at the positions that `input_indices` and
    /// `output_indices` select. An index of [`None`] gives the derived position no override. The derived call inherits
    /// this call's mesh, if any, because it executes part of the same callee on the devices that the callee names. It
    /// also inherits whether this call keeps the placement of its inputs. The derived call retains the function mesh
    /// of `derived` (refer to [`Self::function_mesh`]). A caller can copy a function mesh to a transformed call
    /// explicitly.
    ///
    /// `boundary` is the effective boundary sharding of each input of this call without an override (refer to
    /// `effective_input_boundary`), which the derived call receives as its own at the selected positions, so that a
    /// call that a program transform creates places the inputs of the original call as that call does, and leaves the
    /// inputs that the transform adds (e.g., residuals, tangents of inactive inputs, or cotangent references) where
    /// they are, as the declared types of its callee, which are only traced types, say nothing about where an execution
    /// of the original function places them. [`None`] keeps the kind of this call, for a derived call whose callee
    /// keeps the declared input types of the callee of this call (i.e., a pruned call).
    fn with_derived_boundary_shardings<
        U: Type,
        I: IntoIterator<Item = Option<usize>>,
        O: IntoIterator<Item = Option<usize>>,
    >(
        &self,
        derived: CallOperation<U>,
        boundary: Option<&[Option<Sharding>]>,
        input_indices: I,
        output_indices: O,
    ) -> CallOperation<U> {
        let input_indices = input_indices.into_iter().collect::<Vec<_>>();
        let boundary = boundary.or(self.derived_input_shardings.as_deref());
        let derived_input_shardings = boundary.map(|boundary| {
            input_indices.iter().map(|index| index.and_then(|index| boundary.get(index)?.clone())).collect()
        });
        let derived =
            CallOperation { keeps_input_placements: self.keeps_input_placements, derived_input_shardings, ..derived };
        let Some(mesh) = &self.mesh else {
            return derived;
        };
        let input_shardings = input_indices
            .into_iter()
            .map(|index| index.and_then(|index| self.input_sharding(index)).cloned())
            .collect();
        let output_shardings = output_indices
            .into_iter()
            .map(|index| index.and_then(|index| self.output_sharding(index)).cloned())
            .collect();
        derived.with_boundary(Some(mesh), input_shardings, output_shardings)
    }

    /// Returns `known_call` and `residual_call`, the calls of the two sides of a partial evaluation split of this call,
    /// with the boundary shardings of the inputs and outputs of this call that they receive and return. Residual edges
    /// have none.
    ///
    /// # Parameters
    ///
    ///   - `boundary`: Effective boundary sharding of each input of this call (refer to `effective_input_boundary`).
    ///   - `known_input_indices`: Input of this call that each known callee input receives.
    ///   - `residual_inputs`: Source of each residual callee input.
    ///   - `outputs`: Outputs of the split, each as the index of the output of this call that it computes together with
    ///     its position among the outputs of the known or the residual callee. A split on residual demand computes
    ///     each demanded output twice, once in the known callee and once more in the residual callee.
    ///   - `known_output_count`: Number of outputs of the known callee, including its residual edges.
    ///   - `residual_output_count`: Number of outputs of the residual callee.
    ///   - `eager`: Whether the known program's parent is eager.
    #[allow(clippy::too_many_arguments)]
    fn with_split_boundary_shardings<O: IntoIterator<Item = (usize, PartialEvaluationOutput<usize>)>>(
        &self,
        known_call: Self,
        residual_call: Self,
        boundary: &[Option<Sharding>],
        known_input_indices: &[usize],
        residual_inputs: &[ResidualInputSource],
        outputs: O,
        known_output_count: usize,
        residual_output_count: usize,
        eager: bool,
    ) -> (Self, Self) {
        let mut known_outputs = vec![None; known_output_count];
        let mut residual_outputs = vec![None; residual_output_count];
        for (index, output) in outputs {
            let position = match output {
                PartialEvaluationOutput::Known(position) => known_outputs.get_mut(position),
                PartialEvaluationOutput::Unknown(position) => residual_outputs.get_mut(position),
            };
            if let Some(position) = position {
                *position = Some(index);
            }
        }

        let residual_input_indices = residual_inputs.iter().map(|source| match *source {
            ResidualInputSource::UnknownInput(index) | ResidualInputSource::KnownInput(index) => Some(index),
            ResidualInputSource::KnownOutput(_) | ResidualInputSource::ResidualEdge(_) => None,
        });

        let mut known_call = self.with_derived_boundary_shardings(
            known_call,
            Some(boundary),
            known_input_indices.iter().copied().map(Some),
            known_outputs,
        );

        let mut residual_call = self.with_derived_boundary_shardings(
            residual_call,
            Some(boundary),
            residual_input_indices,
            residual_outputs,
        );

        if eager && let Some(mesh) = &self.function_mesh {
            // A named function fixes the devices independently of the inputs available to each side of the split.
            // The residual side also keeps an anonymous function mesh for later application (e.g., a pullback).
            if mesh.logical_mesh().is_named() {
                known_call.function_mesh = Some(mesh.clone());
            }
            residual_call.function_mesh = Some(mesh.clone());
        }

        (known_call, residual_call)
    }

    /// Returns the exact number of leading callee inputs that form the lifted-capture prefix.
    #[inline]
    pub fn capture_count(&self) -> usize {
        self.capture_count
    }

    /// Returns the sharding override of callee input `index`, if the callee has one.
    #[inline]
    pub fn input_sharding(&self, index: usize) -> Option<&Sharding> {
        self.input_shardings.as_deref().and_then(|shardings| shardings.get(index)?.as_ref())
    }

    /// Returns the sharding override of callee output `index`, if the callee has one.
    #[inline]
    pub fn output_sharding(&self, index: usize) -> Option<&Sharding> {
        self.output_shardings.as_deref().and_then(|shardings| shardings.get(index)?.as_ref())
    }

    /// Returns whether the callee has any boundary sharding override. [`Self::with_boundary`] stores lists without
    /// overrides as absent.
    #[inline]
    fn has_boundary_shardings(&self) -> bool {
        self.input_shardings.is_some() || self.output_shardings.is_some()
    }

    /// Returns every sharding override of the boundary of the callee, inputs first.
    #[inline]
    pub fn boundary_shardings(&self) -> impl Iterator<Item = &Sharding> {
        let input_shardings = self.input_shardings.iter().flat_map(|shardings| shardings.iter());
        let output_shardings = self.output_shardings.iter().flat_map(|shardings| shardings.iter());
        input_shardings.chain(output_shardings).flatten()
    }

    /// Returns the device mesh that the callee was compiled for, on which its sharding overrides are placed, or
    /// [`None`] when the callee names no devices (refer to [`Self::with_boundary`]).
    #[inline]
    pub fn mesh(&self) -> Option<&DeviceMesh> {
        self.mesh.as_ref()
    }

    /// Returns the recorded mesh (refer to [`Self::mesh`]) when the call has an input or output sharding override or
    /// its callee requests a placement that names devices (i.e., any request except one that only moves a value to
    /// another memory). Otherwise it returns [`None`]. This function only asks `callee_names_devices` when the call
    /// records a mesh and has no overrides. Staging records the mesh from the whole program. If pruning drops the
    /// last request that names it, this function no longer reports it.
    /// The analyses of a program (i.e., [`RegionPlacementRequests::requests`] and
    /// [`RegionPlacementRequests::names_devices_predicate`]) answer `callee_names_devices` with facts they derive in
    /// the same pass over the program, so that they never analyze a callee again for each call that encloses it.
    #[inline]
    pub fn names_devices(&self, callee_names_devices: impl FnOnce() -> bool) -> Option<&DeviceMesh> {
        let mesh = self.mesh.as_ref()?;
        (self.has_boundary_shardings() || callee_names_devices()).then_some(mesh)
    }

    /// Returns whether this call leaves each input without a sharding override placed as it is.
    /// Rrefer to [`Self::with_kept_input_placements`] for more information.
    #[inline]
    pub fn keeps_input_placements(&self) -> bool {
        self.keeps_input_placements
    }

    /// Returns the effective boundary sharding of each callee input without a sharding override, by position, for a
    /// call that a program transform derived from another call, or [`None`] for a call of a compiled function, whose
    /// declared input types define that boundary (refer to [`CallOperation::boundary_input_sharding`]). A backend
    /// reads it to leave the inputs of a derived call without a boundary sharding where they are, because the
    /// declared input types of its callee are only traced types.
    #[inline]
    pub fn derived_input_shardings(&self) -> Option<&[Option<Sharding>]> {
        self.derived_input_shardings.as_deref()
    }

    /// Returns the derived boundary sharding of callee input `index`.
    #[inline]
    pub fn derived_input_sharding(&self, index: usize) -> Option<&Sharding> {
        self.derived_input_shardings.as_deref().and_then(|shardings| shardings.get(index)?.as_ref())
    }

    /// Returns the device mesh that the called function was compiled for (refer to [`Self::with_function_mesh`]), or
    /// [`None`] for a call that a program transform creates. Only the placement decision of an eager application by a
    /// backend reads it. A call nested in another compiled function runs where its caller runs whatever this mesh is.
    #[inline]
    pub fn function_mesh(&self) -> Option<&DeviceMesh> {
        self.function_mesh.as_ref()
    }

    /// Checks that the lexical capture namespace is a valid prefix of this callee's actual boundary.
    fn validate_capture_prefix(&self, input_count: usize) -> Result<(), TypeError> {
        if self.capture_count > input_count {
            return Err(TypeError::invalid(format!(
                "`{}` lifted capture prefix {} exceeds its {} callee inputs",
                CALL_OPERATION_NAME, self.capture_count, input_count,
            )));
        }
        Ok(())
    }

    /// Validates exact local-result instantiation and returns the callee-to-caller result correspondence.
    fn output_renaming(
        &self,
        inputs: &[T],
        callee_outputs: &[T],
    ) -> Result<TypeIdentityRenaming<T::Identity>, TypeError> {
        if let Some(types) = &self.output_types {
            return types.renaming(inputs, callee_outputs);
        }

        if callee_outputs
            .iter()
            .flat_map(Type::identities)
            .any(|(_, identity)| !inputs.iter().flat_map(Type::identities).any(|(_, input)| input == identity))
        {
            return Err(TypeError::invalid(format!(
                "`{CALL_OPERATION_NAME}` with callee-local result identities requires `with_callee_type_signature`",
            )));
        }

        Ok(TypeIdentityRenaming::new())
    }
}

impl<T: ArrayMemberType> CallOperation<T> {
    /// Validates that this call accepts inputs of types `input_types` for a callee whose input types are
    /// `callee_input_types`, in count and position, by the type rule of the call. A call accepts an input of exactly
    /// the callee input type and every input whose type refines it up to the placement that its sharding describes,
    /// with the same unreduced and reduced axes and with manual variation along at most more manual mesh axes of one
    /// mesh, where a callee input type with a sharding must define every mesh axis that carries that state as the input
    /// type does. This is the only owner of that rule, so that every context that binds a call (e.g., the eager
    /// application of a backend, staging, and the transforms of a call) accepts the same inputs. An execution
    /// of the callee by a backend accepts a reduced argument as well, because it reshards its arguments.
    ///
    /// Returns the logical mesh and the manual mesh axes along which the inputs vary beyond their callee input types,
    /// if any (refer to [`CallOperation::input_variation`]), which is the union over the inputs of the axes of one
    /// mesh. Every array output of the call varies along those axes too, so every context that binds a call infers
    /// the same outputs.
    ///
    /// # Errors
    ///
    /// Returns a [`TypeError`] that names the first input that the call rejects.
    pub fn validate_input_types(
        &self,
        callee_input_types: &[T],
        input_types: &[T],
    ) -> Result<Option<(LogicalMesh, BTreeSet<String>)>, TypeError> {
        if callee_input_types.len() != input_types.len() {
            return Err(TypeError::invalid(format!(
                "`{}` expected {} input(s) but got {}",
                CALL_OPERATION_NAME,
                callee_input_types.len(),
                input_types.len(),
            )));
        }

        let mut variation = None::<(LogicalMesh, BTreeSet<String>)>;
        for (index, (expected, actual)) in callee_input_types.iter().zip(input_types).enumerate() {
            if expected == actual {
                continue;
            }

            let accepted = match (expected.as_array_type(), actual.as_array_type()) {
                (Ok(expected), Ok(actual)) => {
                    CallOperation::input_variation(expected, actual).map(|axes| (actual, axes))
                }
                _ => None,
            };

            let Some((actual, axes)) = accepted else {
                return Err(TypeError::invalid(format!(
                    "`{CALL_OPERATION_NAME}` input #{index} expected {expected} but got {actual}",
                )));
            };

            let Some(mesh) = actual.sharding().map(Sharding::mesh).filter(|_| !axes.is_empty()) else {
                continue;
            };

            match &mut variation {
                None => variation = Some((mesh.clone(), axes)),
                Some((variation_mesh, variation_axes)) if *variation_mesh == *mesh => variation_axes.extend(axes),
                Some((variation_mesh, _)) => {
                    return Err(TypeError::invalid(format!(
                        "`{CALL_OPERATION_NAME}` input #{index} varies along manual mesh axes of mesh `{mesh}`, \
                         but an earlier input varies along those of mesh `{variation_mesh}`",
                    )));
                }
            }
        }

        Ok(variation)
    }

    /// Returns the effective boundary sharding of callee input `index`, an input of type `actual` for a callee input of
    /// type `declared`, which is where an execution of the callee places that input. This is the sharding override of
    /// the input when the callee has one, which replaces its declared sharding. Otherwise, for a call of a function
    /// compiled for its input types, it is where the executable of that function places the input, which is the
    /// sharding of the declared input type, including over the auto mesh axes that the type does not record (e.g.,
    /// a declared replicated input stays replicated over them), and for a declared input type without a sharding,
    /// replicated over the mesh of the call, which is the mesh that the callee names, if any, and otherwise the mesh
    /// of `actual`, on whose devices the call runs. A call that a program transform derives from another call instead
    /// has the effective boundary of the original call for the inputs that it receives from it and none for the inputs
    /// that the transform adds, because the declared types of its callee are only traced types.
    ///
    /// There is none for an input of the lifted capture prefix, for an input that a call of a Just-In-Time (JIT)
    /// compilation dispatcher keeps placed as it is, for an input that varies along manual mesh axes that the declared
    /// type does not declare, which the declared sharding does not describe, for a declared input type without a
    /// sharding when the mesh of the call is unknown (i.e., `actual` has no sharding), and for a placement over a
    /// single device, which places nothing that the devices of the call do not. Every entry point of a backend that
    /// places the inputs of a call (e.g., its eager application, its lowering, and the result placements of its
    /// compiled programs) consults this function, so that they agree on where a call places its inputs. Only an eager
    /// application also gives an argument on one device the logical mesh of a declared placement over a single device.
    #[inline]
    pub fn boundary_input_sharding<'s>(
        &'s self,
        index: usize,
        actual: &ArrayType,
        declared: &'s T,
    ) -> Option<Cow<'s, Sharding>> {
        self.boundary_input_sharding_with(index, actual, declared, false)
    }

    /// Returns the effective boundary sharding of each callee input without a sharding override, for a call of a callee
    /// whose declared input types are `declared` with inputs of types `actual`, which a call that a program transform
    /// derives from this call receives.
    fn effective_boundary_input_shardings<'t>(
        &self,
        declared: &[T],
        actual: impl IntoIterator<Item = Cow<'t, T>>,
    ) -> Vec<Option<Sharding>>
    where
        T: 't,
    {
        declared
            .iter()
            .zip(actual)
            .enumerate()
            .map(|(index, (declared, actual))| match actual.as_array_type() {
                Ok(actual) if self.input_sharding(index).is_none() => {
                    self.boundary_input_sharding(index, actual, declared).map(Cow::into_owned)
                }
                _ => None,
            })
            .collect()
    }

    /// Returns the effective boundary sharding of callee input `index`, which, with `includes_single_device`,
    /// includes a placement over a single device.
    fn boundary_input_sharding_with<'s>(
        &'s self,
        index: usize,
        actual: &ArrayType,
        declared: &'s T,
        includes_single_device: bool,
    ) -> Option<Cow<'s, Sharding>> {
        if index < self.capture_count {
            return None;
        }

        if let Some(sharding) = self.input_sharding(index) {
            return Some(Cow::Borrowed(sharding));
        }

        if self.keeps_input_placements {
            return None;
        }

        if self.derived_input_shardings.is_some() {
            return self.derived_input_sharding(index).map(Cow::Borrowed);
        }

        let Ok(declared) = declared.as_array_type() else { return None };
        if !CallOperation::input_variation(declared, actual).is_some_and(|axes| axes.is_empty()) {
            return None;
        }

        let sharding = match declared.sharding() {
            Some(sharding) => Cow::Borrowed(sharding),
            None => {
                let mesh = match &self.mesh {
                    Some(mesh) => mesh.logical_mesh(),
                    None => actual.sharding()?.mesh(),
                };
                Cow::Owned(CallOperation::declared_input_sharding(declared, mesh))
            }
        };

        (includes_single_device || sharding.mesh().device_count() > 1).then_some(sharding)
    }
}

impl CallOperation<ArrayIrType> {
    /// Returns the sharding with which an executable that is compiled for a mesh whose logical mesh is `mesh` places
    /// a public argument of declared type `input_type` that has no input sharding override and no runtime placement
    /// (e.g., of a Just-In-Time compilation dispatcher), which is the sharding of that type, and otherwise replicated
    /// over `mesh`. A call of a function compiled for its input types places its inputs by this same rule (i.e., using
    /// [`Self::boundary_input_sharding`]), so that a direct execution, an eager call, and a nested call place an input
    /// alike. This function takes no call, because a backend also applies the rule where it executes or compiles a
    /// function directly.
    #[inline]
    pub fn declared_input_sharding(input_type: &ArrayType, mesh: &LogicalMesh) -> Sharding {
        input_type
            .sharding()
            .cloned()
            .unwrap_or_else(|| Sharding::replicated(mesh.clone(), input_type.shape().rank()))
    }

    /// Returns the sharding to which an eager application of this call places callee input `index`, which receives
    /// an argument of type `actual`, on the devices of that argument before the callee runs. This is the effective
    /// boundary sharding of the input (i.e., [`Self::boundary_input_sharding`]), and, for an argument on one device,
    /// also a placement over a single device, which moves no data but may name another logical mesh than `actual` does
    /// (e.g., `q` for an argument over `p`). A direct execution of the callee by a backend gives every argument the
    /// mesh and the sharding with which its executable receives it, so the application gives the argument that logical
    /// mesh too, and both entry points accept the same arguments and return the same results. Partial evaluation
    /// forwards an input only if eager application would preserve its sharding.
    #[inline]
    pub fn eager_boundary_input_sharding<'s>(
        &'s self,
        index: usize,
        actual: &ArrayType,
        declared: &'s ArrayIrType,
    ) -> Option<Cow<'s, Sharding>> {
        let includes_single_device = actual.sharding().is_some_and(|sharding| sharding.mesh().device_count() == 1);
        self.boundary_input_sharding_with(index, actual, declared, includes_single_device)
    }

    /// Returns `input_type`, the declared type of an input of a compiled function, refined by `boundary`,
    /// the placement with which the executable of the function receives it (e.g., the input override or
    /// [`Self::declared_input_sharding`]), when `input_type` records no sharding, and [`None`] when it records one or
    /// when `boundary` states more than such a type can (e.g., a pending reduction, which a type without a sharding
    /// never describes). The refined type records the placement of `boundary` without its auto mesh axes, which types
    /// never record. A transform that inserts a dimension into each item of an input (e.g., batching, which places the
    /// new dimension as its batch specification requests, refer to [`ArrayType::batched`]) starts from this type, so
    /// that the placement of the new dimension is kept for an item type that records none. Every entry point that
    /// batches a compiled function (i.e., the batching of a compiled function by a backend and the batching rule
    /// of a call of one) refines its item types by this rule. This function takes no call for the same reason as
    /// [`Self::declared_input_sharding`].
    #[inline]
    pub fn boundary_input_type(input_type: &ArrayType, boundary: &Sharding) -> Option<ArrayType> {
        if input_type.sharding().is_some() {
            return None;
        }
        let boundary_type = input_type.clone().with_sharding(boundary.without_auto_axes()).ok()?;
        input_type.is_refined_by(&boundary_type).then_some(boundary_type)
    }

    /// Returns the effective boundary sharding of callee input `index` (i.e., [`Self::boundary_input_sharding`]) when
    /// the call can move an input of type `actual` to it, which is when it is a sharding override, when it differs from
    /// the sharding of `actual`, or when its mesh has auto axes, along which `actual` cannot show where the input is
    /// placed. A backend that lowers a call constrains exactly these inputs, and partial evaluation forwards an input
    /// through a call only when it is not one of them.
    #[inline]
    pub fn boundary_input_sharding_constraint<'s>(
        &'s self,
        index: usize,
        actual: &ArrayType,
        declared: &'s ArrayIrType,
    ) -> Option<Cow<'s, Sharding>> {
        let sharding = self.boundary_input_sharding(index, actual, declared)?;
        let moves = self.input_sharding(index).is_some()
            || actual.sharding() != Some(sharding.as_ref())
            || sharding.mesh().has_untracked_auto_axes();
        moves.then_some(sharding)
    }

    /// Returns whether this call requests the placement of a value explicitly, which is when the call names devices
    /// (i.e., [`Self::names_devices`]) or its callee requests a placement anywhere in the regions that the backend
    /// runs, including one that only moves a value to another memory (i.e., [`RegionPlacementRequests::has_requests`]).
    /// A backend commits every result of a call that requests a placement, so the placement decision of an eager
    /// application by a backend and the forwarding of partial evaluation read it from here. `callee_requests` returns
    /// the [`RegionPlacementRequests`] of the callee for the target platform that the call runs on, or for no target
    /// platform, and it is called only when the answer depends on the callee. The summary of a allee depends on the
    /// callee and the platform alone, so it is derived once for each callee and platform and retained with the callee,
    /// and repeated applications only look it up.
    #[inline]
    pub fn has_placement_requests(&self, callee_requests: impl Fn() -> RegionPlacementRequests) -> bool {
        self.names_devices(|| callee_requests().has_requests(false)).is_some() || callee_requests().has_requests(true)
    }

    /// Returns the manual mesh axes along which an input of type `actual` varies beyond `declared`, the type of the
    /// callee input that receives it, when a call accepts that input, or [`None`] when it rejects it. The type of the
    /// input must refine the declared type up to the placement that their shardings describe, which a call may change
    /// because it does not change the values. What a sharding states about the values must agree, because a call,
    /// unlike an execution of its callee by a backend, does not reshard its inputs. These are the unreduced and reduced
    /// axes, with the kind of reduction, where a declared type without a sharding describes a fully reduced value. The
    /// input may vary along more manual mesh axes than the declared type, which a value of a `shard_map` body does
    /// where the callee declares none, as a JAX `jit` function called on such a value is retraced at its varying type,
    /// and the outputs of the call then vary along those axes too. When both types have shardings, the declared
    /// sharding's mesh must define every axis that carries manual variation or reduction state as the input mesh
    /// defines it. This preserves the meaning of the axis across the call boundary. Other axes may differ.
    fn input_variation(declared: &ArrayType, actual: &ArrayType) -> Option<BTreeSet<String>> {
        fn components(
            sharding: Option<&Sharding>,
        ) -> (&BTreeSet<String>, Option<UnreducedKind>, &BTreeSet<String>, &BTreeSet<String>) {
            let no_axes = const { &BTreeSet::new() };
            match sharding {
                Some(sharding) => (
                    sharding.unreduced_axes(),
                    (!sharding.unreduced_axes().is_empty()).then(|| sharding.unreduced_kind()),
                    sharding.reduced_axes(),
                    sharding.varying_manual_axes(),
                ),
                None => (no_axes, None, no_axes, no_axes),
            }
        }

        let (declared_unreduced, declared_kind, declared_reduced, declared_variation) = components(declared.sharding());
        let (actual_unreduced, actual_kind, actual_reduced, actual_variation) = components(actual.sharding());
        if (declared_unreduced, declared_kind, declared_reduced) != (actual_unreduced, actual_kind, actual_reduced)
            || !declared_variation.is_subset(actual_variation)
        {
            return None;
        }

        if let (Some(declared_sharding), Some(actual_sharding)) = (declared.sharding(), actual.sharding()) {
            validate_preserved_mesh_axes(
                CALL_OPERATION_NAME,
                actual_sharding.mesh(),
                actual_variation.iter().chain(actual_unreduced).chain(actual_reduced),
                declared_sharding.mesh(),
            )
            .ok()?;
        }

        let declared_without_sharding = declared.clone().with_sharding(None).ok()?;
        let actual_without_sharding = actual.clone().with_sharding(None).ok()?;
        declared_without_sharding
            .is_refined_by(&actual_without_sharding)
            .then(|| actual_variation.difference(declared_variation).cloned().collect())
    }

    /// Interprets this call in `context` on `inputs` by the reference semantics of a call. The type rule of the
    /// call validates the complete application first,  because direct eager binding does not pass through program
    /// construction, so that interpretation accepts exactly the inputs that staging accepts. Each array input is then
    /// placed where the eager application of a backend places it, as far as the callee region accepts that placement
    /// and the type of a value records it, and the callee region runs on the placed inputs:
    ///
    ///   - An input of a callee input type that declares a sharding enters the callee at that sharding, because the
    ///     callee region validates its inputs against its declared input types and accepts no other placement. This is
    ///     the boundary sharding of the input unless the call has a sharding override or a derived sharding for it that
    ///     differs from the declared one, whose placement is therefore not observable inside the callee. An input of
    ///     the lifted capture prefix, an input that the call keeps placed as it is, and an input that varies along
    ///     manual mesh axes beyond its callee input type have no boundary sharding and enter at the declared sharding
    ///     too. The target of the placement is the declared sharding without its manual reduction state, which
    ///     `reshard` carries over from the input together with its manual variation, so that an input that varies
    ///     beyond its callee input type keeps that variation.
    ///   - An input of a callee input type without a sharding is placed at its boundary sharding, which is its sharding
    ///     override, its derived sharding, or the sharding that replicates it over the mesh of the call, including the
    ///     logical mesh of a placement over a single device, as far as its type records that sharding. Placement along
    ///     auto mesh axes is runtime state that neither types nor the values of the reference backend record (refer to
    ///     [`Placed`](crate::Placed)), so a backend applies it while this rule, as the reference rules of `place` and
    ///     `constrain_sharding` do, keeps only the part that types record. A boundarysharding that states more than a
    ///     placement (e.g., a pending reduction) or that `reshard` rejects (e.g., one that places a dimension along a
    ///     manual mesh axis) is not applied at all. An input without a boundary sharding (i.e., of the lifted capture
    ///     prefix, kept placed as it is, varying beyond its callee input type, or without a sharding when the call
    ///     names no mesh) keeps its placement.
    ///
    /// Each placed input that varies along manual mesh axes beyond its callee input type is then passed to `narrow`
    /// with those axes. Returns the output types that the type rule infers together with the outputs of the callee.
    ///
    /// # Parameters
    ///
    ///   - `reshard`: Reshards a value of `context` to a [`Sharding`] (refer to [`Reshard`]).
    ///   - `narrow`: Returns a value of `context` that does not vary along the provided manual mesh axes, which a
    ///     context whose values hold the values of every device of a manual region can do without changing any value
    ///     (e.g., the [`LockstepDomain`] of the `shard_map` emulation). Other contexts return the value unchanged.
    fn interpret_with<
        C: Domain<Type = ArrayIrType>,
        D: InterpretationDriver<C>,
        ReshardFn: Fn(C::Value, &Sharding) -> Result<C::Value, ProgramError>,
        NarrowFn: Fn(C::Value, &BTreeSet<String>) -> Result<C::Value, ProgramError>,
    >(
        &self,
        context: &C,
        driver: &D,
        inputs: &[C::Value],
        reshard: ReshardFn,
        narrow: NarrowFn,
    ) -> Result<(Vec<ArrayIrType>, Vec<C::Value>), ProgramError> {
        let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        let region_interfaces = driver.region_interfaces();
        let output_types = self.infer_output_types(&input_types, &region_interfaces)?;
        let inputs = inputs
            .iter()
            .zip(&input_types)
            .zip(region_interfaces[0].input_types())
            .enumerate()
            .map(|(index, ((input, actual), declared))| {
                let placement = match (actual, declared) {
                    (ArrayIrType::Array(actual), ArrayIrType::Array(declared_type)) => match declared_type.sharding() {
                        Some(sharding) => Some(Cow::Borrowed(sharding)),
                        None => self
                            .eager_boundary_input_sharding(index, actual, declared)
                            .and_then(|boundary| {
                                CallOperation::boundary_input_type(declared_type, &boundary)?.sharding().cloned()
                            })
                            .filter(|sharding| {
                                let reshard = ReshardOperation::new(sharding.clone());
                                reshard.infer_output_types(&[actual.clone()], &[]).is_ok()
                            })
                            .map(Cow::Owned),
                    }
                    .filter(|sharding| actual.sharding() != Some(sharding.as_ref())),
                    _ => None,
                };

                let input = match placement {
                    Some(sharding) => reshard(input.clone(), &sharding.without_manual_reduction_axes())?,
                    None => input.clone(),
                };

                let variation = match (actual, declared) {
                    (ArrayIrType::Array(actual), ArrayIrType::Array(declared)) => {
                        Self::input_variation(declared, actual)
                    }
                    _ => None,
                };

                match variation.filter(|axes| !axes.is_empty()) {
                    Some(axes) => narrow(input, &axes),
                    None => Ok(input),
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok((output_types, driver.interpret_region(context, 0, inputs)?))
    }

    /// Folds or residualizes this call as a whole (refer to [`PartialEvaluationContext::fold_or_residualize`]). The
    /// callee is bound as a shared program (refer to [`shared_call_callee`]), so that repeated calls of one callee keep
    /// sharing one region in the known-side context and in the residual program.
    fn fold_or_residualize<V, C, D>(
        &self,
        context: &PartialEvaluationContext<C>,
        driver: &D,
        inputs: &[PartialEvaluationValue<C::Value>],
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError>
    where
        V: 'static + Value<Type = ArrayIrType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
        C: Context<
                Type = ArrayIrType,
                Constant = V,
                Operation: 'static
                               + From<CallOperation<ArrayIrType>>
                               + From<GuardOperation<ArrayIrType>>
                               + OperationPayloadProjection,
            >,
        D: PartialEvaluationDriver<C>,
    {
        // A call that folds into an eager known side (e.g., the primal call of an eager linearization) is applied
        // by the eager context and keeps the mesh of its function, and so does a call that an eager known side
        // residualizes (e.g., the tangent call of an eager linearization), whose residual program an eager context
        // applies too (e.g., the pullback of an eager VJP, which transposes it). A call placed into a known or a
        // residual program of a staged known side follows its caller.
        let callee = shared_call_callee(driver.region(0)?)?;
        let operation = if context.parent().is_eager() { self.clone() } else { self.without_function_mesh() };
        context.fold_or_residualize(
            C::Operation::from(operation),
            AttachedRegions::Callees(std::slice::from_ref(&callee)),
            inputs,
            None,
        )
    }

    /// Returns `outputs`, the outputs of this call of `callee` with `inputs`, with every output that `callee` returns
    /// by forwarding one of its inputs replaced by that call input, like the input forwarding of JAX's `pjit` staging
    /// rule, wherever the call provably returns that input unchanged. For an all-known call that partial evaluation
    /// keeps whole under a residual policy, this lets the residual planner reuse the forwarded input (e.g., a primal
    /// input that the primal call of a linearized callee returns as a residual).
    ///
    /// The input types must prove that forwarding preserves devices, placement, commitment, and admission checks.
    /// The proof requires all of the following conditions:
    ///
    ///   - The call leaves the function mesh and explicit devices unspecified, so a top-level application can follow
    ///     its argument.
    ///   - The callee makes no placement requests. Any request, including a memory-only transfer, commits every output.
    ///   - The forwarded input is the only public array argument. A committed sibling can select the devices, move an
    ///     uncommitted input there, and commit every result. Two committed arguments on different devices are rejected.
    ///     These decisions depend on runtime facts that the argument types omit.
    ///   - The input is outside the lifted capture prefix. The call moves its captures to where it runs and returns
    ///     them uncommitted.
    ///   - The input type records sharding on a mesh without auto axes. An eager call rejects pending reductions its
    ///     callee cannot receive. The type rule checks pending reductions on explicit and manual axes during staging,
    ///     and pending reductions on auto axes require a runtime check. An input with no recorded sharding can live
    ///     on any mesh.
    ///   - The boundary leaves the input in place and the eager application preserves its logical mesh on a single
    ///     device.
    ///
    /// The call rejects an argument from another client. A forwarded argument from another client can be returned,
    /// as by any program that returns an argument directly, and every consumer that executes rejects it.
    ///
    /// An eager known side folds the call on concrete arguments before forwarding any outputs, so admission checks have
    /// already run. A staged known side can later be compiled, in which case the outer executable admits the arguments,
    /// or interpreted eagerly (e.g., by [`Program::interpret_in_context`](crate::Program)), in which case each
    /// surviving call checks its arguments and decides placement at runtime. The same forwarding conditions
    /// apply in both cases.
    fn resolve_forwarded_outputs<V, O, W>(
        &self,
        callee: RegionRef<'_, V, O>,
        inputs: &[PartialEvaluationValue<W>],
        outputs: Vec<PartialEvaluationValue<W>>,
    ) -> Vec<PartialEvaluationValue<W>>
    where
        V: 'static + Value<Type = ArrayIrType>,
        O: 'static + Operation<Type = ArrayIrType> + OperationPayloadProjection,
        W: Value<Type = ArrayIrType>,
    {
        let input_types = inputs.iter().map(PartialEvaluationValue::r#type).collect::<Vec<_>>();
        let public_array_count = input_types
            .iter()
            .skip(self.capture_count)
            .filter(|r#type| matches!(r#type.as_ref(), ArrayIrType::Array(_)))
            .count();
        if public_array_count != 1
            || self.function_mesh.is_some()
            || self.has_placement_requests(|| RegionPlacementRequests::new(callee))
        {
            return outputs;
        }

        let declared = callee.input_types();
        let unchanged = input_types
            .iter()
            .enumerate()
            .map(|(position, r#type)| match r#type.as_ref() {
                ArrayIrType::Array(actual) if position >= self.capture_count => {
                    let records_reduction_state = actual.sharding().is_some_and(|sharding| {
                        sharding.mesh().axes().iter().all(|axis| axis.r#type() != MeshAxisType::Auto)
                    });
                    records_reduction_state
                        && self.boundary_input_sharding_constraint(position, actual, &declared[position]).is_none()
                        && self
                            .eager_boundary_input_sharding(position, actual, &declared[position])
                            .is_none_or(|sharding| actual.sharding() == Some(sharding.as_ref()))
                }
                _ => false,
            })
            .collect::<Vec<_>>();

        if !unchanged.contains(&true) {
            return outputs;
        }

        outputs
            .into_iter()
            .zip(callee.output_ids())
            .map(|(output, atom)| match callee.input_ids().iter().position(|input| input == atom) {
                Some(position) if unchanged[position] && *input_types[position] == *output.r#type() => {
                    inputs[position].clone()
                }
                _ => output,
            })
            .collect()
    }

    /// Splits this call into a known call, which binds a known callee in the known-side context, and a residual call,
    /// which is emitted into the residual program, and returns the outputs of the call in output order.
    ///
    ///   - A split by [`Knownness`](CallSplitKind::Knownness) splits a call whose inputs are partially known. The known
    ///     callee computes the known outputs together with the residual edges that the residual callee needs, which the
    ///     residual policy of `context`, if any, places. When the known callee contains no instructions, the split
    ///     hoists no work, and the call is kept whole.
    ///   - A split on [`ResidualDemand`](CallSplitKind::ResidualDemand) splits an all-known call (refer to
    ///     [`PartiallyEvaluatableOperation::split_on_residual_demand`]). The known callee computes every output of the
    ///     callee together with the interior values that the residual policy of `context` saves, and the residual
    ///     callee replays the demanded outputs from those values.
    ///
    /// The residual callee reads the call inputs and the known outputs that its residual edges would merely repeat
    /// directly, so that the known callee returns each value once and a residual planner plans those inputs like any
    /// other demanded value. The two callees are retained by the region transform cache of the callee and bound as
    /// shared regions. Every call of one callee, including the calls nested in the callees of an enclosing split,
    /// therefore shares one known callee and one residual callee for each kind of split and each policy, so the
    /// programs grow with the number of calls rather than with the size of their callees, and each callee is
    /// partitioned once rather than once per call. Calls whose partial evaluations round the saved values share a split
    /// of their own, whose known callee rounds the values that it saves right after their producers, before its own
    /// known work consumes them. Mixed splits may contain effects, with the driver's repeated-residual mode included in
    /// the key. All-known splits exclude effects, and both kinds exclude references, captures, and nominal identities.
    ///
    /// `partition` partitions the callee for `kind`, which keys the retained split. Callers supply it because a split
    /// on residual demand partitions through [`PartialEvaluationContext::partition_residual_demand`], whose bound only
    /// the caller's [`PartiallyEvaluatableOperation::split_on_residual_demand`] states.
    fn split<V, C, D>(
        &self,
        context: &PartialEvaluationContext<C>,
        driver: &D,
        inputs: &[PartialEvaluationValue<C::Value>],
        kind: CallSplitKind,
        partition: &dyn Fn(RegionRef<'_, V, C::Operation>) -> Result<PartitionedProgram<V, C::Operation>, ProgramError>,
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError>
    where
        V: 'static + Value<Type = ArrayIrType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
        C: Context<
                Type = ArrayIrType,
                Constant = V,
                Operation: 'static
                               + From<CallOperation<ArrayIrType>>
                               + From<GuardOperation<ArrayIrType>>
                               + OperationPayloadProjection,
            >,
        D: PartialEvaluationDriver<C>,
    {
        /// [`Region`] [`Transform`] marker for the splits of the callee of a `call` that this function binds. Such a
        /// split partitions the callee in fresh traces, where every input is a fresh known or unknown input, so it
        /// depends on no value of a live context and is a structural function of the callee and of its
        /// [`CallSplitArguments`]. Every call of one callee with the same arguments therefore shares one split.
        struct CallSplitTransform;

        impl<V, O> Transform<Region<V, O>> for CallSplitTransform
        where
            V: Value<Type = ArrayIrType>,
            O: Operation<Type = ArrayIrType>,
        {
            type Arguments = CallSplitArguments;
            type Artifact = TransformArtifact<V, O, Arc<CallSplit>>;

            const DEFAULT_CACHE_CAPACITY: usize = 8;
        }

        /// Argument key for one retained [`CallSplitTransform`].
        #[derive(Clone, Debug, PartialEq, Eq, Hash)]
        struct CallSplitArguments {
            /// Kind of the split.
            kind: CallSplitKind,

            /// Residual policy that places the residuals of the split and of the splits nested within it, compared by
            /// definition identity, or [`None`] when partial evaluation has no residual policy.
            policy: Option<ResidualPolicyReference<ArrayIrType>>,

            /// Type of the rounding with which the placement of the residuals of the split also rounds the values that
            /// the known callee saves (refer to [`PartialEvaluationContext::residual_rounding_type`]), which stages
            /// rounding operations in the known callee, or [`None`] when the placement keeps those values unrounded.
            residual_rounding_type: Option<TypeId>,

            /// Whether effectful operations with known inputs may fold into the known callee (refer to
            /// [`PartialEvaluationContext::allow_effect_folding`]).
            allow_effect_folding: bool,

            /// Whether operations whose results must have storage of their own for every execution may fold into the
            /// known callee (refer to [`PartialEvaluationContext::allow_copy_folding`]), which program specialization
            /// disallows.
            allow_copy_folding: bool,

            /// Whether recursive partitioning analyzes repeated residual invocations, which changes effect placement
            /// (refer to [`PartialEvaluationDriver::partitions_for_repeated_residual_calls`]). Pure and residual-demand
            /// splits use `false` because their partitions do not depend on this mode.
            repeated_residual: bool,
        }

        /// Boundary wiring of a retained [`CallSplitTransform`] artifact, whose programs are the known callee and the
        /// residual callee, in that order. The known callee returns the known outputs of the original callee followed
        /// by the residual edges that the residual callee reads from it, and the residual callee returns the remaining
        /// outputs (i.e., the demanded outputs for a split on residual demand). Refer to [`PartitionedProgram`] for
        /// more information on this layout.
        #[derive(Clone, Debug, PartialEq)]
        struct CallSplit {
            /// Index of the call input feeding each input of the known callee, in order.
            known_input_indices: Vec<usize>,

            /// Source of each input of the residual callee, which is a call input, an output of the known callee, or a
            /// residual edge.
            residual_inputs: Vec<ResidualInputSource>,

            /// Source of each output of the call, in output order, which is a known output of the known callee or an
            /// output of the residual callee.
            outputs: Vec<PartialEvaluationOutput<usize>>,
        }

        let callee = driver.region(0)?;
        let keeps_trivial_splits_whole = matches!(kind, CallSplitKind::Knownness(_));

        // A split on residual demand reports every output of the callee as known, followed by each demanded output once
        // more as an output of the residual callee, in output order. Each of those trailing outputs is therefore the
        // call output that was demanded, whose boundary sharding the residual call keeps.
        let demanded_output_indices = match &kind {
            CallSplitKind::Knownness(_) => Vec::new(),
            CallSplitKind::ResidualDemand(demanded_outputs) => demanded_outputs
                .iter()
                .enumerate()
                .filter_map(|(index, demanded)| demanded.then_some(index))
                .collect(),
        };

        let arguments = CallSplitArguments {
            kind,
            policy: context.residual_policy().cloned(),
            residual_rounding_type: context.residual_rounding_type(),
            allow_effect_folding: context.allow_effect_folding(),
            allow_copy_folding: context.allow_copy_folding(),
            repeated_residual: keeps_trivial_splits_whole
                && !callee.effects().classes().is_empty()
                && driver.partitions_for_repeated_residual_calls(),
        };

        let artifact = callee.transform::<CallSplitTransform, _, ProgramError>(arguments, |callee, _| {
            // Partitioning stages both callees in fresh traces, so the partition depends only on the callee and the
            // arguments, including whether mixed splits use repeated-residual allocation and effect-ordering analysis.
            let (known_program, residual_program, known_input_indices, residual_inputs, outputs) =
                partition(callee)?.forward_residuals()?.into_parts();
            Ok(TransformArtifact::new(
                vec![Arc::new(known_program), Arc::new(residual_program)],
                Arc::new(CallSplit { known_input_indices, residual_inputs, outputs }),
            ))
        })?;

        let [known_callee, residual_callee] = artifact.programs() else {
            unreachable!("`{CALL_OPERATION_NAME}` splits retain a known callee and a residual callee");
        };

        if keeps_trivial_splits_whole && known_callee.instructions().is_empty() {
            return self.fold_or_residualize(context, driver, inputs);
        }

        let split = artifact.metadata();

        // The effective boundary of the call follows the types of its inputs before they are guarded.
        let boundary = self.effective_boundary_input_shardings(
            &callee.input_types(),
            inputs.iter().map(PartialEvaluationValue::r#type),
        );

        // Partitioning seeds both callees at the declared input types of the callee, so neither checks a relationship
        // that the entry of the callee checks between a known and an unknown input. Both calls therefore consume
        // guarded inputs.
        let inputs = context.guard_inputs(&callee.input_types(), inputs.to_vec(), false)?;

        // The known callee keeps the members of the lifted capture prefix that it uses as its leading inputs, in order.
        // The residual callee reads no capture prefix, and neither callee holds a capture constant.
        let known_capture_count =
            split.known_input_indices.iter().filter(|index| **index < self.capture_count()).count();
        let known_call = CallOperation::new(known_capture_count).with_callee_region(known_callee.entry_region_ref())?;
        let residual_call = CallOperation::new(0).with_callee_region(residual_callee.entry_region_ref())?;
        let (known_call, residual_call) = self.with_split_boundary_shardings(
            known_call,
            residual_call,
            &boundary,
            &split.known_input_indices,
            &split.residual_inputs,
            (0..callee.output_types().len()).chain(demanded_output_indices).zip(split.outputs.iter().copied()),
            known_callee.output_count(),
            residual_callee.output_count(),
            context.parent().is_eager(),
        );
        let known_inputs = split.known_input_indices.iter().map(|index| inputs[*index].clone()).collect::<Vec<_>>();
        let known_outputs = context.fold_or_residualize(
            C::Operation::from(known_call.clone()),
            AttachedRegions::Callees(std::slice::from_ref(known_callee)),
            &known_inputs,
            None,
        )?;
        let known_outputs = match context.residual_policy() {
            Some(_) if known_callee.effects().classes().is_empty() => {
                known_call.resolve_forwarded_outputs(known_callee.entry_region_ref(), &known_inputs, known_outputs)
            }
            _ => known_outputs,
        };

        // The residual edges trail the known outputs among the outputs of the known call.
        let known_output_count = split.outputs.iter().filter(|output| output.is_known()).count();
        let residual_inputs = split
            .residual_inputs
            .iter()
            .map(|source| match *source {
                ResidualInputSource::UnknownInput(index) | ResidualInputSource::KnownInput(index) => {
                    inputs[index].clone()
                }
                ResidualInputSource::KnownOutput(index) => known_outputs[index].clone(),
                ResidualInputSource::ResidualEdge(edge) => known_outputs[known_output_count + edge].clone(),
            })
            .collect::<Vec<_>>();
        let residual_outputs = context.residualize(
            C::Operation::from(residual_call),
            AttachedRegions::Callees(std::slice::from_ref(residual_callee)),
            &residual_inputs,
        )?;

        Ok(split
            .outputs
            .iter()
            .map(|output| match *output {
                PartialEvaluationOutput::Known(index) => known_outputs[index].clone(),
                PartialEvaluationOutput::Unknown(index) => residual_outputs[index].clone(),
            })
            .collect())
    }
}

impl<T: Type + Hash> Hash for CallOperation<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // The fields that existed before the boundary shardings hash as the derived implementation did, and an absent
        // boundary writes nothing, so that the keys of programs whose calls have none stay unchanged.
        self.capture_count.hash(state);
        self.marker.hash(state);
        self.output_types.hash(state);
        if let Some(shardings) = &self.input_shardings {
            shardings.hash(state);
        }
        if let Some(shardings) = &self.output_shardings {
            shardings.hash(state);
        }
        if let Some(mesh) = &self.mesh {
            mesh.hash(state);
        }
        if self.keeps_input_placements {
            self.keeps_input_placements.hash(state);
        }
        if let Some(shardings) = &self.derived_input_shardings {
            shardings.hash(state);
        }
    }
}

impl<T: ArrayMemberType + From<ArrayType>> Operation for CallOperation<T> {
    type Type = T;

    #[inline]
    fn name(&self) -> &'static str {
        CALL_OPERATION_NAME
    }

    #[inline]
    fn defined_type_identities(&self) -> Cow<'_, [T::Identity]> {
        self.output_types.as_ref().map_or(Cow::Borrowed(&[]), |types| Cow::Owned(types.definitions()))
    }

    #[inline]
    fn region_slots(&self) -> &'static [RegionSlot] {
        const { &[RegionSlot::computation("callee")] }
    }

    fn infer_region_input_types(
        &self,
        input_types: &[T],
        interfaces: &[RegionInterface<T>],
    ) -> Result<Vec<Option<Vec<T>>>, TypeError> {
        check_count!("region", interfaces, 1, TypeError);
        let declared = interfaces[0].input_types();
        if declared == input_types {
            return Ok(vec![None]);
        }

        // Inputs that instantiate identities of the callee rename them in the callee, whose inputs otherwise
        // keep their declared types, because the call accepts inputs whose types only refine them (refer to
        // `CallOperation::validate_input_types`), as an execution of the callee accepts its arguments.
        match T::derive_identity_renaming(declared, input_types) {
            Ok(renaming) if !renaming.is_identity() => {
                let renamed =
                    declared.iter().map(|r#type| r#type.rename_identities(&renaming)).collect::<Result<Vec<_>, _>>()?;
                self.validate_input_types(&renamed, input_types)?;
                Ok(vec![Some(renamed)])
            }
            _ => {
                self.validate_input_types(declared, input_types)?;
                Ok(vec![None])
            }
        }
    }

    fn infer_output_types(
        &self,
        input_types: &[T],
        region_interfaces: &[RegionInterface<T>],
    ) -> Result<Vec<T>, TypeError> {
        if region_interfaces.len() != 1 {
            return Err(TypeError::invalid(format!(
                "`{}` expects 1 attached callee region but got {}",
                CALL_OPERATION_NAME,
                region_interfaces.len(),
            )));
        }

        let callee_interface = &region_interfaces[0];
        self.validate_capture_prefix(callee_interface.input_types().len())?;
        let variation = self.validate_input_types(callee_interface.input_types(), input_types)?;

        self.output_renaming(callee_interface.input_types(), callee_interface.output_types())?;
        let output_types = if let Some(types) = &self.output_types {
            types.instantiated(callee_interface.input_types())?
        } else {
            callee_interface.output_types().to_vec()
        };
        if callee_interface.input_types() == input_types {
            return Ok(output_types);
        }

        // Inputs that refine the callee input types (e.g., with a static extent for a dynamic one) refine the outputs
        // that name the same identities, which the identities of the inputs no longer define. A sharding that the
        // callee input type does not have establishes no such fact, except that the outputs vary along the manual
        // mesh axes along which the inputs vary beyond the callee input types.
        let output_types = match T::Refinements::new(callee_interface.input_types(), input_types) {
            Ok(refinements) => output_types
                .iter()
                .map(|r#type| refinements.refine_declared(r#type, &[]))
                .collect::<Result<_, _>>()?,
            Err(_) => output_types,
        };

        let Some((mesh, axes)) = variation else {
            return Ok(output_types);
        };

        output_types
            .into_iter()
            .enumerate()
            .map(|(index, r#type)| {
                let Ok(array_type) = r#type.as_array_type() else {
                    return Ok(r#type);
                };
                let mut sharding = match array_type.sharding() {
                    None => Sharding::replicated(mesh.clone(), array_type.rank()),
                    Some(sharding) if sharding.mesh() == &mesh => sharding.clone(),
                    Some(_) => {
                        return Err(TypeError::invalid(format!(
                            "`{CALL_OPERATION_NAME}` output #{index} has type {array_type}, which cannot vary along \
                             the manual mesh axes {axes:?} of mesh `{mesh}` along which its inputs vary",
                        )));
                    }
                };
                sharding.extend_varying_manual_axes(axes.iter().cloned())?;
                Ok(T::from(array_type.clone().with_sharding(sharding)?))
            })
            .collect()
    }

    #[inline]
    fn input_region_provenance(&self, region_index: usize, input_index: usize) -> InputRegionProvenance {
        if region_index == 0 {
            InputRegionProvenance::Input { index: input_index }
        } else {
            InputRegionProvenance::None
        }
    }

    #[inline]
    fn output_region_provenance(&self, output_index: usize) -> Vec<OutputRegionProvenance> {
        vec![OutputRegionProvenance { region_index: 0, output_index }]
    }

    #[inline]
    fn region_data_flow(&self) -> RegionDataFlow<'_> {
        RegionDataFlow::Provenance
    }

    fn prune_boundary(
        &self,
        input_types: &[Self::Type],
        used_outputs: &[bool],
        regions: &mut dyn RegionLiveness,
    ) -> Result<Option<OperationBoundaryPruning<Self>>, ProgramError> {
        // Inputs map onto the callee inputs one for one, so the inputs that the callee does not use are dropped
        // together with its unused outputs, as in JAX's `dce_jaxpr_pjit_rule`. The lifted capture prefix is always
        // kept, because the captures in the callee refer to it by absolute position.
        let mut kept_inputs = regions.used_region_inputs(0, used_outputs)?;
        kept_inputs.iter_mut().take(self.capture_count).for_each(|kept| *kept = true);
        check_count!("input", kept_inputs, input_types.len(), ProgramError);

        let mut operation = self.clone();
        operation.output_types =
            self.output_types.as_ref().map(|types| types.pruned(&kept_inputs, used_outputs)).transpose()?;

        // The boundary shardings follow the kept callee inputs and outputs.
        let kept_input_indices = kept_inputs.iter().enumerate().filter_map(|(index, kept)| kept.then_some(Some(index)));
        let kept_output_indices =
            used_outputs.iter().enumerate().filter_map(|(index, used)| used.then_some(Some(index)));
        let operation = self.with_derived_boundary_shardings(
            CallOperation { input_shardings: None, output_shardings: None, mesh: None, ..operation },
            None,
            kept_input_indices,
            kept_output_indices,
        );

        Ok(Some(OperationBoundaryPruning { operation, kept_inputs, kept_outputs: used_outputs.to_vec() }))
    }

    #[inline]
    fn region_capture_input_count(&self, region_index: usize) -> Option<usize> {
        (region_index == 0).then_some(self.capture_count)
    }

    #[inline]
    fn rename_type_identities(&self, renaming: &TypeIdentityRenaming<T::Identity>) -> Result<Self, TypeError> {
        let mut renamed = self.clone();
        renamed.output_types = self.output_types.as_ref().map(|types| types.renamed(renaming)).transpose()?;
        Ok(renamed)
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        let operation = OperationFormatter::new(formatter, indentation, CALL_OPERATION_NAME)?;
        let derived_input_shardings = self.derived_input_shardings.as_deref();
        if self.capture_count == 0
            && self.output_types.is_none()
            && self.mesh.is_none()
            && !self.keeps_input_placements
            && derived_input_shardings.is_none()
        {
            Ok(())
        } else {
            // Boundary shardings and the devices that the callee names render only when present, so that calls of
            // callees that name no devices render as before. A position without an override renders as `_`. The
            // effective boundary of a call that a transform derived always renders, even when it places no input,
            // because such a call leaves its inputs where they are while an ordinary call places them as its callee
            // input types record, so the two lower differently and must not render alike.
            let render_shardings = |shardings: &[Option<Sharding>]| {
                shardings
                    .iter()
                    .map(|sharding| sharding.as_ref().map_or_else(|| "_".to_string(), Sharding::to_string))
                    .collect::<Vec<_>>()
            };
            operation.bracketed(|operation| {
                if self.capture_count != 0 {
                    operation.field("capture_count", self.capture_count)?;
                }
                if let Some(bindings) = &self.output_types {
                    operation.list("output_types", bindings.outputs.as_ref())?;
                }
                if let Some(shardings) = &self.input_shardings {
                    operation.list("input_shardings", render_shardings(shardings))?;
                }
                if let Some(shardings) = &self.output_shardings {
                    operation.list("output_shardings", render_shardings(shardings))?;
                }
                if let Some(shardings) = derived_input_shardings {
                    operation.list("derived_input_shardings", render_shardings(shardings))?;
                }
                if let Some(mesh) = &self.mesh {
                    operation.list("devices", mesh.devices())?;
                }
                if self.keeps_input_placements {
                    operation.field("keeps_input_placements", true)?;
                }
                Ok(())
            })
        }
    }
}

impl<T, C, P> ReferenceDischargeableOperation<C, P> for CallOperation<T>
where
    T: ArrayMemberType + From<ArrayType> + From<P::Referent>,
    C: Context<Type = T, Operation: From<CallOperation<T>>>,
    P: ReferenceDischargePolicy<C>,
{
    fn discharge_references<D: ReferenceDischargeDriver<C, P>>(
        &self,
        context: &ReferenceDischargeContext<C, P>,
        driver: &D,
        inputs: &[ReferenceDischargeValue<C, P>],
    ) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError> {
        // A `call` operation forwards its inputs onto its callee's inputs one for one and reports the callee's outputs
        // as its own, which is the positionally forwarding shape the shared structured rewrite serves with no leading
        // inputs. The callee's lifted capture prefix is unaffected, because threaded state is appended after every
        // declared input, so the prefix keeps its positions. Instantiated result types belong to the source callee
        // signature, which discharge changes by turning discharged references into arrays and appending state inputs
        // and outputs. The rebuilt call therefore keeps the same caller-owned result identities but derives its result
        // metadata from the rebuilt callee signature.
        let result_renaming = match &self.output_types {
            Some(_) => {
                let callee = driver.region(0)?;
                Some(self.output_renaming(&callee.input_types(), &callee.output_types())?)
            }
            None => None,
        };

        context.discharge_positional_region_operation(self, driver, inputs, 0, |_, regions| {
            let Some(result_renaming) = result_renaming else { return Ok(self.clone()) };
            check_count!("region", regions, 1, TypeError);
            let input_types = regions[0].input_types();
            let callee_outputs = regions[0].output_types();
            let caller_output_types = callee_outputs
                .iter()
                .map(|r#type| r#type.rename_identities(&result_renaming))
                .collect::<Result<Vec<_>, _>>()?;
            let operation = Self { output_types: None, ..self.clone() };
            Ok(operation.with_caller_output_types(&input_types, &callee_outputs, caller_output_types)?)
        })
    }
}

impl<V: Value<Type = ArrayIrType>, O: Operation<Type = ArrayIrType>> InterpretableOperation<EagerContext<V, O>>
    for CallOperation<ArrayIrType>
where
    V: Reshard,
{
    #[inline]
    fn interpret<D: InterpretationDriver<EagerContext<V, O>>>(
        &self,
        context: &EagerContext<V, O>,
        driver: &D,
        inputs: &[V],
    ) -> Result<Vec<V>, ProgramError> {
        // Reference interpretation of a call in an eager context of this crate, which accepts exactly the inputs
        // that staging accepts and returns outputs of the types that the type rule of the call infers. Each input is
        // resharded to where the call places it (i.e., the sharding that its callee input type declares, or its
        // boundary sharding when that type declares none), which changes no value of an `Array`, so that the callee
        // receives the input at the placement that a backend gives it.
        self.interpret_with(context, driver, inputs, |input, sharding| input.reshard(sharding), |input, _| Ok(input))
            .map(|(_, outputs)| outputs)
    }
}

impl<O: Operation<Type = ArrayIrType>> InterpretableOperation<LockstepDomain<O>> for CallOperation<ArrayIrType> {
    fn interpret<D: InterpretationDriver<LockstepDomain<O>>>(
        &self,
        context: &LockstepDomain<O>,
        driver: &D,
        inputs: &[LockstepValue<O>],
    ) -> Result<Vec<LockstepValue<O>>, ProgramError> {
        // The `shard_map` emulation runs a call once for all emulated devices when its callee communicates across them
        // or when an input varies along manual mesh axes beyond its callee input type, by the reference interpretation
        // of a call, which reshards the value of every device of an input whose placement differs from its callee input
        // type, and whose region replay runs the callee in lockstep. That replay validates neither the entry nor the
        // outputs of the callee, so only the validation of the type rule of the call applies there. An input that
        // varies beyond its callee input type enters the callee without that extra variation, which changes no value of
        // any device, so that the callee runs at the types that it was traced at, as a backend runs it, because manual
        // variation is not materialized inside a manual computation. The operations of the callee that have regions of
        // their own (e.g., a `condition` or a `while` loop) were typed for those inputs and would reject the wider
        // variation. An output that the callee computes from invariant inputs alone, or from inputs that entered it
        // without their extra variation, still varies along the manual mesh axes along which inputs vary beyond their
        // callee input types, as the type rule of the call states, so that every later operation receives the output
        // at its staged type.
        let (output_types, outputs) = self.interpret_with(
            context,
            driver,
            inputs,
            |input: LockstepValue<O>, sharding: &Sharding| {
                let values =
                    input.into_values().into_iter().map(|value| value.reshard(sharding)).collect::<Result<_, _>>();
                LockstepValue::new(values?, context)
            },
            |input: LockstepValue<O>, axes: &BTreeSet<String>| {
                let values = input
                    .into_values()
                    .into_iter()
                    .map(|value| match value {
                        ArrayIrValue::Array(array) => {
                            let r#type = array.r#type().into_owned();
                            let Some(mut sharding) = r#type.sharding().cloned() else {
                                return Ok(ArrayIrValue::Array(array));
                            };
                            let variation =
                                sharding.varying_manual_axes().difference(axes).cloned().collect::<Vec<_>>();
                            sharding.set_varying_manual_axes(variation).map_err(TypeError::from)?;
                            let r#type = r#type.with_sharding(sharding).map_err(TypeError::from)?;
                            Array::from_logical_bytes(r#type, &array.logical_bytes()).map(ArrayIrValue::Array)
                        }
                        value => Ok(value),
                    })
                    .collect::<Result<Vec<_>, ProgramError>>()?;
                LockstepValue::new(values, context)
            },
        )?;
        check_count!("output", outputs, output_types.len(), ProgramError);

        // The callee computed its outputs at the types that it was traced at, while the type rule of the call makes
        // every output vary along the manual mesh axes along which inputs vary beyond their callee input types. Each
        // output is therefore lifted to its staged type by adding the manual mesh axes that it lacks.
        outputs
            .into_iter()
            .zip(&output_types)
            .map(|(output, r#type)| -> Result<LockstepValue<O>, ProgramError> {
                let actual_type = output.r#type().into_owned();
                let (ArrayIrType::Array(r#type), ArrayIrType::Array(actual_type)) = (r#type, &actual_type) else {
                    return Ok(output);
                };

                let Some(sharding) = r#type.sharding() else { return Ok(output) };
                let actual_sharding = actual_type.sharding();
                let missing_axes = sharding
                    .varying_manual_axes()
                    .iter()
                    .filter(|axis| !actual_sharding.is_some_and(|actual| actual.varying_manual_axes().contains(*axis)))
                    .cloned()
                    .collect::<Vec<_>>();
                if missing_axes.is_empty() {
                    return Ok(output);
                }

                // An output without a sharding first takes the replicated sharding over the mesh of its staged type,
                // as an output type without one does in the type rule. Only the type of the value of each device
                // changes, and its bytes are kept.
                let mut output = match actual_sharding {
                    Some(_) => output,
                    None => {
                        let replicated = Sharding::replicated(sharding.mesh().clone(), actual_type.rank());
                        let placed = actual_type.clone().with_sharding(replicated).map_err(TypeError::from)?;
                        let values = output
                            .into_values()
                            .into_iter()
                            .map(|value| match value {
                                ArrayIrValue::Array(array) => {
                                    Array::from_logical_bytes(placed.clone(), &array.logical_bytes())
                                        .map(ArrayIrValue::Array)
                                }
                                value => Ok(value),
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        LockstepValue::new(values, context)?
                    }
                };

                // Each missing axis is added by `parallel_vary`, which changes no value of any device.
                for axis in missing_axes {
                    let mut varied =
                        ParallelVaryOperation::new(axis).interpret_in_parent(context, driver, &[output])?;
                    check_count!("output", varied, 1, ProgramError);
                    output = varied.remove(0);
                }
                Ok(output)
            })
            .collect()
    }
}

impl<O: Operation<Type = ArrayIrType>> LockstepOperation<O> for CallOperation<ArrayIrType> {
    #[inline]
    fn interpret_in_lockstep<D: InterpretationDriver<LockstepDomain<O>>>(
        &self,
        context: &LockstepDomain<O>,
        driver: &D,
        inputs: &[LockstepValue<O>],
    ) -> Option<Result<Vec<LockstepValue<O>>, ProgramError>> {
        Some(self.interpret(context, driver, inputs))
    }
}

impl<V, C> PartiallyEvaluatableOperation<C> for CallOperation<ArrayIrType>
where
    V: 'static + Value<Type = ArrayIrType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C: Context<
            Type = ArrayIrType,
            Constant = V,
            Operation: 'static
                           + From<CallOperation<ArrayIrType>>
                           + From<GuardOperation<ArrayIrType>>
                           + OperationPayloadProjection,
        >,
{
    fn partially_evaluate<D: PartialEvaluationDriver<C>>(
        &self,
        context: &PartialEvaluationContext<C>,
        driver: &D,
        inputs: &[PartialEvaluationValue<C::Value>],
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
        // This implementation splits the callee against the caller's knownness while preserving the `call` boundary on
        // both sides. The split fires whenever the knownness of the call inputs is mixed (i.e., some inputs are known
        // and some are not). All-known and all-unknown calls preserve the original boundary, and their callee is bound
        // as a shared program (i.e., an interned callee, or a copy retained by the region transform cache of the
        // callee), so that repeated calls of one callee keep sharing one region on either side. Under a residual
        // policy, the outputs of a pure all-known call that only forward call inputs resolve to those inputs, and a
        // residual planner that demands an output of such a call whose callee saves interior values asks this operation
        // to split the call on residual demand (i.e., `PartiallyEvaluatableOperation::split_on_residual_demand`).
        // That split binds a known call that also returns those values and a replay call that recomputes the demanded
        // outputs from them. Whether the mixed call's known inputs are concrete values of an eager known-side context
        // or tracers into a live outer trace does not matter, as in JAX's call partial evaluation rules, which always
        // bind the known call. The known side is bound through the default fold-or-residualize policy, which applies it
        // under an eager context (e.g., the eager context of a backend, which compiles and executes it) and stages it
        // into the outer program under a staging context. The `shard_map` rule follows the same policy. When a split
        // fires, the callee is split through the shared `PartitionedProgram` machinery. The known side is bound into
        // the enclosing known-side context as a `call` over the known call inputs that it uses, and the residual side
        // is emitted as the residual `call` over the surviving unknown call inputs plus the known-side call's
        // residual-edge outputs. Residual edges that would merely repeat a known call input or a known call output
        // are forwarded, so the residual call consumes that input or output directly and the known-side call does not
        // return the same value twice. For a pure callee (i.e., one without effects, references, captures, or nominal
        // type identities), both kinds of split are retained by the region transform cache of the callee and bound as
        // shared regions, so repeated and nested calls of one callee reuse one split instead of partitioning and
        // copying the callee at every call site. A callee with references or captures keeps its boundary, and a callee
        // with nominal type identities is split per call when its inputs are mixed. Effectful mixed calls also share
        // their splits, keyed by the driver's repeated-residual mode because that mode determines effect placement.
        // Effectful and identity-bearing all-known calls remain whole on residual demand.

        let all_known = inputs.iter().all(PartialEvaluationValue::is_known);
        let callee = driver.region(0)?;
        if all_known || inputs.iter().all(PartialEvaluationValue::is_unknown) {
            // An all-known call folds as a whole, unless its callee holds work that must not fold under this context
            // (e.g., a copy that every execution of a specialized program must make), in which case the default rule
            // keeps the whole call residual.
            let outputs = self.fold_or_residualize(context, driver, inputs)?;
            if all_known && context.residual_policy().is_some() && !keeps_call_boundary(callee, inputs) {
                return Ok(self.resolve_forwarded_outputs(callee, inputs, outputs));
            }
            return Ok(outputs);
        }

        // The split calls have separate boundaries. Preserve the original call's staged result types when its caller
        // supplied staged declarations.
        let output_types = if inputs.iter().any(|input| input.declared_type().is_some()) {
            let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
            Some(self.infer_output_types(&input_types, std::slice::from_ref(callee.interface()))?)
        } else {
            None
        };

        let declare_outputs = |outputs| {
            if let Some(output_types) = &output_types {
                PartialEvaluationValue::with_declared_output_types(outputs, output_types)
            } else {
                Ok(outputs)
            }
        };

        // Retain mixed splits without references, captures, or nominal identities. Unlike all-known replay,
        // partitioning a mixed effectful call preserves its effects and shares only the structural split.
        if can_cache_call_split(callee, inputs) {
            let input_known = inputs.iter().map(PartialEvaluationValue::is_known).collect::<Vec<_>>();
            let partition =
                |callee: RegionRef<'_, V, C::Operation>| driver.partition_program(context, callee, &input_known);
            let kind = CallSplitKind::Knownness(input_known.clone());
            return self.split(context, driver, inputs, kind, &partition).and_then(declare_outputs);
        }

        // Split the callee of any other mixed call per call through the shared online boundary machinery, bind the
        // known side into the enclosing known-side context wrapped in a fresh `call` over the original known call
        // inputs, emit the residual side as the residual `call`, and reassemble the original output order. These
        // splits are not retained because the identity-bearing boundary needs call-local type identities.
        //
        // A reference input, a reference output, or a reference access anywhere in the callee closure keeps the call
        // whole. Splitting would spread the accesses to one root across two compiled calls whose known side runs
        // first, and a known reference-typed feeder of the residual side would then mean that both sides access the
        // same root, the same reordering hazard that keeps a `scan` body whole. The default rule preserves effect order
        // by placing the whole call on one side instead. An allocation that nothing accesses is dead intrinsic state
        // that neither side retains, so it does not pin the boundary, but an allocation that escapes through a
        // reference output is live state whose later accesses are unknown to the callee, so it does.
        let accesses_references = inputs.iter().any(|input| input.r#type().is_reference())
            || callee.output_types().iter().any(Type::is_reference)
            || callee
                .instructions_in_closure()
                .any(|(_, instruction)| instruction.operation().effects().has_reference_accesses());
        if accesses_references {
            return self.fold_or_residualize(context, driver, inputs);
        }

        // Partitioning does not remap the absolute indices of capture constants retained in attached regions, so a
        // split boundary would leave them naming a compacted or absent capture prefix, which fails lowering at best
        // and silently aliases an unrelated leading input at worst. Preserve the original call boundary for any callee
        // whose closure still holds a capture constant.
        if callee.contains_atom_in_closure(|atom| {
            atom.as_constant().is_some_and(|constant| constant.capture_index().is_some())
        }) {
            return self.fold_or_residualize(context, driver, inputs);
        }

        let input_known = inputs.iter().map(PartialEvaluationValue::is_known).collect::<Vec<bool>>();
        let partition = driver.partition_program(context, callee, input_known.as_slice())?;

        // A trivial partition, whose known program contains no instructions, hoists no work because its known side
        // can only forward known inputs as residual edges. Keep the original boundary and let the default materialize
        // those knowns directly as residual feeders.
        if partition.known_program().instructions().is_empty() {
            return self.fold_or_residualize(context, driver, inputs);
        }

        // Feed the residual call directly from the known call inputs and known call outputs that residual edges would
        // merely repeat (i.e., JAX's `in_fwd` and `out_fwd`), so that the known-side call returns each value once.
        let partition = partition.forward_residuals()?;

        // Known inputs keep callee source order, so the known-side callee's leading inputs are exactly the known
        // members of the original lifted-capture prefix that it still uses. The residual callee's inputs are call
        // inputs, known call outputs, and residual edges, which never form a capture prefix. The guard above already
        // preserved the boundary of any callee retaining attached region capture constants, so neither derived callee
        // can hold one.
        let known_capture_count =
            partition.known_input_indices().iter().filter(|index| **index < self.capture_count()).count();
        let known_call = CallOperation::new(known_capture_count).with_callee_type_signature(
            &partition.known_program().input_types(),
            &partition.known_program().output_types(),
        )?;
        let residual_call = CallOperation::new(0).with_callee_type_signature(
            &partition.residual_program().input_types(),
            &partition.residual_program().output_types(),
        )?;
        let boundary = self.effective_boundary_input_shardings(
            &callee.input_types(),
            inputs.iter().map(PartialEvaluationValue::r#type),
        );
        let (known_call, residual_call) = self.with_split_boundary_shardings(
            known_call,
            residual_call,
            &boundary,
            partition.known_input_indices(),
            partition.residual_inputs(),
            partition.outputs().iter().copied().enumerate(),
            partition.known_program().output_count(),
            partition.residual_program().output_count(),
            context.parent().is_eager(),
        );
        let outputs = context.inline_partitioned_program(
            partition,
            Some(callee),
            inputs,
            |known_program| (C::Operation::from(known_call), vec![known_program]),
            |residual_program| (C::Operation::from(residual_call), vec![residual_program]),
        )?;
        declare_outputs(outputs)
    }

    fn split_on_residual_demand<D: PartialEvaluationDriver<C>>(
        &self,
        context: &PartialEvaluationContext<C>,
        driver: &D,
        inputs: &[PartialEvaluationValue<C::Value>],
        demanded_outputs: &[bool],
    ) -> Result<Option<ResidualDemandSplit<C::Value>>, ProgramError>
    where
        C::Operation: PartiallyEvaluatableOperation<TracingContext<C::Constant, C::Operation>>,
    {
        let callee = driver.region(0)?;
        if !inputs.iter().all(PartialEvaluationValue::is_known)
            || context.residual_policy().is_none()
            || keeps_call_boundary(callee, inputs)
        {
            return Ok(None);
        }

        let kind = CallSplitKind::ResidualDemand(demanded_outputs.to_vec());
        let mut outputs = self.split(context, driver, inputs, kind, &|callee: RegionRef<'_, V, C::Operation>| {
            context.partition_residual_demand(callee, demanded_outputs)
        })?;
        let replayed_outputs = outputs.split_off(callee.output_types().len());
        Ok(Some(ResidualDemandSplit::new(outputs, replayed_outputs)?))
    }
}

impl<C> BatchableOperation<C, ArrayIrBatchingPolicy> for CallOperation<ArrayIrType>
where
    C: Context<Type = ArrayIrType>,
    C::Operation: From<Self>,
    TracingContext<C::Constant, C::Operation>:
        StagingContext<Type = ArrayIrType, Constant = C::Constant, Operation = C::Operation>,
{
    fn batch<D: BatchingDriver<C, ArrayIrBatchingPolicy>>(
        &self,
        context: &BatchingContext<C, ArrayIrBatchingPolicy>,
        driver: &D,
        inputs: &[ArrayIrBatch<C::Value>],
    ) -> Result<Vec<ArrayIrBatch<C::Value>>, BatchingError> {
        // The callee region is rebatched over the mapped input axes through `BatchingDriver::batch_program`, and
        // the batched call is bound through `context.parent()` with the batched callee attached again. An eager parent
        // (e.g., the eager context of a backend, which compiles and executes it) applies the batched call immediately,
        // a staging parent stages it into the enclosing program, and a differentiation parent dispatches it through its
        // own `call` Jacobian-Vector Product (JVP) rule, which is what serves `vmap` nested inside `gradient` and
        // `linearize` closures.
        let axes = inputs.iter().map(ArrayIrBatch::batch_axis).collect::<Vec<_>>();

        // Only an all-replicated call at an anonymous level can bind its original callee unchanged. A named level must
        // rebatch the callee even then, because the callee may read that level by name (e.g., through `axis_index`),
        // and binding it unchanged would let that read escape the level that binds it.
        if axes.iter().all(BatchAxis::is_replicated) && context.axis_name().is_none() {
            return Ok(context
                .parent()
                .bind(
                    self.clone(),
                    vec![driver.region(0)?.to_program()],
                    &inputs.iter().map(|input| input.value().clone()).collect::<Vec<_>>(),
                )?
                .into_iter()
                .map(ArrayIrBatch::replicated)
                .collect::<Vec<_>>());
        }

        // The canonical array IR transform threads one leading dimension through the callee as its first input and
        // first output. Treat the input as an additional explicit capture, so that the original capture order remains,
        // and drop the bookkeeping output from the callee itself rather than after the call. The output only forwards
        // the extent input, and an eager parent compiles the call as a standalone program, whose results must be
        // arrays.
        let original = driver.region(0)?;
        let boundary = self.effective_boundary_input_shardings(
            &original.input_types(),
            inputs.iter().map(|input| Cow::Owned(input.unbatched_type())),
        );
        let refined = with_boundary_input_types(original, &boundary)?;
        let original = refined.as_ref().map_or(original, Program::entry_region_ref);
        let (callee, axes) = driver
            .batch_program(context, original, &axes, ProgramBatchingOutputAxesPolicy::Natural)?
            .into_parts();
        let kept_outputs = (1..callee.output_count()).collect::<Vec<_>>();
        let callee = callee.with_outputs(&kept_outputs)?;
        let mut values = vec![context.axis_extent().clone()];
        values.extend(inputs.iter().map(|input| input.value().clone()));
        let operation = Self::new(self.capture_count() + 1).with_callee_region(callee.entry_region_ref())?;
        let array_type = |r#type: &ArrayIrType| match r#type {
            ArrayIrType::Array(r#type) => Some(r#type.clone()),
            _ => None,
        };

        // The batched callee is created by this transform, so its declared input types are only traced types, and
        // the batched call is a derived call in every case, which places each input at the effective boundary of the
        // original call for its items (refer to `with_derived_boundary_shardings`). The boundary shardings gain the
        // batch dimension of each batched value (refer to `batched_boundary_sharding`), and the leading extent input
        // has none.
        let derived_input_shardings = Some(
            std::iter::once(Ok(None))
                .chain(inputs.iter().zip(&boundary).map(|(input, sharding)| {
                    let packed = array_type(&input.value().r#type());
                    batched_boundary_sharding(sharding.as_ref(), &input.batch_axis(), packed.as_ref())
                }))
                .collect::<Result<_, _>>()?,
        );

        let operation = Self {
            keeps_input_placements: self.keeps_input_placements,
            derived_input_shardings,
            function_mesh: self.function_mesh.clone(),
            ..operation
        };
        let operation = if let Some(mesh) = self.mesh() {
            let input_shardings = std::iter::once(Ok(None))
                .chain(inputs.iter().enumerate().map(|(index, input)| {
                    let packed = array_type(&input.value().r#type());
                    batched_boundary_sharding(self.input_sharding(index), &input.batch_axis(), packed.as_ref())
                }))
                .collect::<Result<_, _>>()?;
            let output_types = callee.output_types();
            let output_shardings = axes
                .iter()
                .enumerate()
                .map(|(index, axis)| {
                    let packed = output_types.get(index).and_then(array_type);
                    batched_boundary_sharding(self.output_sharding(index), axis, packed.as_ref())
                })
                .collect::<Result<_, _>>()?;
            operation.with_boundary(Some(mesh), input_shardings, output_shardings)
        } else {
            operation
        };

        let outputs = context.parent().bind(operation, vec![callee], &values)?;
        check_count!("output", outputs, axes.len(), ProgramError);
        outputs
            .into_iter()
            .zip(axes)
            .map(|(value, axis)| ArrayIrBatch::new(value, axis))
            .collect::<Result<Vec<_>, _>>()
    }
}

impl<C> BatchableOperation<C, ArrayBatchingPolicy> for CallOperation<ArrayType>
where
    C: Context<Type = ArrayType>,
    C::Operation: From<CallOperation<ArrayType>>,
    TracingContext<C::Constant, C::Operation>:
        StagingContext<Type = ArrayType, Constant = C::Constant, Operation = C::Operation>,
{
    fn batch<D: BatchingDriver<C, ArrayBatchingPolicy>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy>,
        driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        // Rebatch the callee when inputs are mapped or when it can address the batching level by name. Only an
        // all-replicated call at an anonymous level can bind its original callee unchanged. Unlike the forward mode and
        // transpose rules below, this rebatching is deliberately not served from the callee region's retained transform
        // cache. The batched program depends on live `BatchingContext` state (i.e., the batch extent, the axis name,
        // and the surrounding nesting), so caching it against the callee region and its input axes alone would risk
        // exactly the unsound key reuse that the cache is designed to avoid. Caching it needs a complete key and a
        // measured benefit, through a batching policy hook returning `Option<key>` where `None` keeps a policy
        // uncached, so that soundness stays opt-in per policy.
        let physical_inputs = inputs.iter().map(|input| input.value().clone()).collect::<Vec<_>>();
        let boundary = self.effective_boundary_input_shardings(
            &driver.region(0)?.input_types(),
            inputs.iter().map(|input| Cow::Owned(input.unbatched_type())),
        );

        let batch_callee = ArrayBatch::common_batch_size(inputs)?.is_some() || context.axis_name().is_some();
        let (batched_callee, output_axes) = if batch_callee {
            let input_batch_axes = inputs
                .iter()
                .map(|input| BatchAxis::from_optional_position(input.batch_axis_position()))
                .collect::<Vec<_>>();
            let original = driver.region(0)?;
            let refined = with_boundary_input_types(original, &boundary)?;
            let (batched_callee, output_axes) = driver
                .batch_program(
                    context,
                    refined.as_ref().map_or(original, Program::entry_region_ref),
                    input_batch_axes.as_slice(),
                    ProgramBatchingOutputAxesPolicy::Natural,
                )?
                .into_parts();

            // The core program batching pass returns canonical nonnegative physical positions.
            let output_axes = output_axes
                .iter()
                .map(|axis| axis.axis().map(|axis| usize::try_from(axis.value()).unwrap()))
                .collect::<Vec<_>>();
            (batched_callee.into_simplified()?, output_axes)
        } else {
            let callee = driver.region(0)?;
            let output_axes = vec![None; callee.output_types().len()];
            (callee.to_program(), output_axes)
        };

        let operation = Self::new(self.capture_count()).with_callee_region(batched_callee.entry_region_ref())?;

        // The batched call is a derived call in every case, as in the batching rule of the array IR form,
        // and its boundary shardings gain the batch dimension of each mapped value at its batch axis.
        let derived_input_shardings = Some(
            inputs
                .iter()
                .zip(&boundary)
                .map(|(input, sharding)| {
                    let axis = BatchAxis::from_optional_position(input.batch_axis_position());
                    let packed = input.value().r#type();
                    batched_boundary_sharding(sharding.as_ref(), &axis, Some(packed.as_ref()))
                })
                .collect::<Result<_, _>>()?,
        );

        let operation = Self {
            keeps_input_placements: self.keeps_input_placements,
            derived_input_shardings,
            function_mesh: self.function_mesh.clone(),
            ..operation
        };
        let operation = if let Some(mesh) = self.mesh() {
            let input_shardings = inputs
                .iter()
                .enumerate()
                .map(|(index, input)| {
                    let axis = BatchAxis::from_optional_position(input.batch_axis_position());
                    let packed = input.value().r#type();
                    batched_boundary_sharding(self.input_sharding(index), &axis, Some(packed.as_ref()))
                })
                .collect::<Result<_, _>>()?;
            let output_types = batched_callee.output_types();
            let output_shardings = output_axes
                .iter()
                .enumerate()
                .map(|(index, axis)| {
                    let axis = BatchAxis::from_optional_position(*axis);
                    batched_boundary_sharding(self.output_sharding(index), &axis, output_types.get(index))
                })
                .collect::<Result<_, _>>()?;
            operation.with_boundary(Some(mesh), input_shardings, output_shardings)
        } else {
            operation
        };

        let outputs = context.parent().bind(
            operation,
            AttachedRegions::Callees(&[Arc::new(batched_callee)]),
            &physical_inputs,
        )?;
        outputs
            .into_iter()
            .zip(output_axes)
            .map(|(output, axis)| ArrayBatch::new(output, BatchAxis::from_optional_position(axis)))
            .collect::<Result<Vec<_>, _>>()
    }
}

impl<C> DifferentiableOperation<C> for CallOperation<ArrayIrType>
where
    C: Context<
            Type = ArrayIrType,
            Operation: From<CallOperation<ArrayIrType>>
                           + From<CastOperation<ArrayIrType>>
                           + OperationProjection<ArrayType, Projected: From<ReshardOperation>>,
        > + Zero<C::Value>,
{
    fn jvp<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        // This is a capture-free forward mode differentiation rule (i.e., Jacobian-Vector Product, or JVP, rule) for
        // `CallOperation`. When the primal and tangent contexts are shared (e.g., for `jvp` and for the fused program
        // that a `rematerialize` rule partitions), the rule binds one fused `call` over the callee's forward mode
        // program, projected onto the primal outputs and the live output tangents. An enclosing partial evaluation then
        // splits that call with its own residual policy. Otherwise, it binds a primal `call` and a tangent `call` as
        // ordinary operations of the family of the active context. A staging context stages both calls over its shared
        // builder, while an eager context (e.g., the eager context of a backend, which compiles and executes them)
        // applies them immediately.
        //
        // Both forms realize the identity `jvp(jit(f)) = jit(jvp f)`. Rather than capturing the primal inputs as
        // residual factors and staging a linear `call`, the separate form keeps the compilation boundary and threads
        // every residual as a plain primal input edge between two `call`s, so no symbolic capture is ever introduced.
        // The enclosing partial evaluation split then discovers the residual input edges structurally, exactly as it
        // does for the condition and rematerialize rules.
        //
        // For separate contexts, the rule linearizes the callee program capture-free through `Program::linearize` under
        // the activity mask of the inputs, giving a primal sub-program `inputs -> [outputs..., residuals...]` and a
        // tangent sub-program `[active_input_tangents..., residuals...] -> [output_tangents...]` together with the
        // residual count. An input with a live tangent is active, while the capture prefix, an input with a
        // structurally zero tangent, and a plumbing reference input that receives no tangent reference are not.
        // Tangents for zero differential spaces are omitted from both compact boundaries, and the tangent sub-program
        // is projected onto the output tangents that depend on an active input tangent (refer to
        // `Program::output_dependence` and `Program::with_outputs`), since every other output tangent is a structural
        // zero.  When no output tangent is live and the projected tangent sub-program has no observable effects, the
        // rule binds only the source call and pairs every output with a structurally zero tangent. Otherwise, it takes
        // the following steps:
        //
        //   1. Wraps the primal sub-program in a fresh `call` and stages it over the primal inputs, recovering the
        //      primal outputs followed by the residual values, which are program variables that the staged primal
        //      call produces.
        //   2. Wraps the projected tangent sub-program in a fresh `call` and stages it over the active input tangents
        //      followed by those residual values, recovering the live output tangents.
        //   3. Pairs each primal output tracer with its live tangent output tracer, or with a structural zero for an
        //      output without a live tangent, into a `DifferentiationDual`.
        //
        // The callee program is materialized from the instruction's callee region in the constant universe of the
        // context, so the split halves ride the fresh primal and tangent calls as shared callee regions. Preserving
        // both `call` boundaries keeps the callee body out of the caller's program, so forward mode over a `call`
        // operation stays compiled rather than inlined.
        let callee = driver.region(0)?;
        let input_types = callee.input_types();
        let output_types = callee.output_types();
        let output_count = output_types.len();
        check_count!("input", inputs, input_types.len(), ProgramError);

        // Linearize the callee through the instruction-scoped driver under the call's activity mask. The
        // callee's leading lifted-capture inputs are inactive, numeric and reference alike, because a capture is
        // non-differentiated plumbing that receives no tangent input. The remaining inputs take the activity of their
        // duals. An input with a live tangent is active, while an input with a structurally zero tangent is inactive
        // like a missing one (i.e., JAX's `which_nz`), so it gets no tangent callee slot instead of being materialized
        // as a zero input. A plumbing reference input, whose structurally zero tangent is a missing tangent reference,
        // is therefore inactive too. The primal sub-program produces `[outputs..., residuals...]` and the tangent
        // sub-program consumes `[active(input_tangents)..., residuals...]`. The residual count is the number of
        // trailing outputs of the primal sub-program beyond the original callee outputs.
        let capture_count = self.capture_count();
        let activity = inputs
            .iter()
            .enumerate()
            .map(|(index, input)| index >= capture_count && input.is_tangent_active() && !input.tangent().is_zero())
            .collect::<Vec<_>>();
        let input_indices = activity
            .iter()
            .enumerate()
            .filter_map(|(index, &active)| active.then_some(index))
            .collect::<Vec<_>>();

        // A shared primal/tangent destination binds one fused call over the callee's forward mode program, projected
        // onto its primal outputs and live output tangents, as JAX's `_pjit_jvp` does. An enclosing partial evaluation
        // then splits that call with its own residual policy, instead of reusing a linearization boundary chosen here.
        if std::ptr::eq(context.primal(), context.tangent()) {
            self.validate_capture_prefix(input_types.len())?;
            let activity = TangentActivity::new(driver, callee, &activity)?;
            let output_activity = activity.output_tangent_liveness();
            let live_tangent_slots = activity.live_tangent_slots(output_activity);
            let program = activity.jvp_program();
            let mut fused_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
            if live_tangent_slots.is_empty() && !program.effects().is_retained_when_unused() {
                let callee = shared_call_callee(callee)?;
                return context
                    .primal()
                    .bind(C::Operation::from(self.clone()), AttachedRegions::Callees(&[callee]), &fused_inputs)?
                    .into_iter()
                    .map(DifferentiationDual::new_with_zero_tangent)
                    .collect();
            }

            let outputs = (0..output_count).chain(live_tangent_slots.iter().map(|&slot| output_count + slot));
            let program = if outputs.clone().eq(0..program.output_count()) {
                program.clone()
            } else {
                Arc::new(program.as_ref().clone().with_outputs(&outputs.collect::<Vec<_>>())?)
            };

            fused_inputs.extend(
                activity.input_indices().iter().map(|&index| inputs[index].tangent().as_value().unwrap().clone()),
            );

            // The primal inputs come first, so the lifted-capture prefix of the source call carries over unchanged.
            // Each tangent has the boundary sharding of its primal, as JAX's `_pjit_jvp` gives each tangent the
            // sharding of its primal.
            let live_outputs = (0..output_count).filter(|&index| output_activity[index]);
            let boundary = self
                .effective_boundary_input_shardings(&input_types, inputs.iter().map(|input| input.primal().r#type()));

            // The fused call replaces the whole call in the same context, so it keeps the mesh of the function and an
            // eager JVP runs where the call runs.
            let operation = C::Operation::from(CallOperation {
                function_mesh: self.function_mesh.clone(),
                ..self.with_derived_boundary_shardings(
                    CallOperation::new(capture_count).with_callee_region(program.entry_region_ref())?,
                    Some(&boundary),
                    (0..input_types.len()).chain(activity.input_indices().iter().copied()).map(Some),
                    (0..output_count).chain(live_outputs).map(Some),
                )
            });

            let mut outputs = context.primal().bind(operation, AttachedRegions::Callees(&[program]), &fused_inputs)?;
            check_count!("output", outputs, output_count + live_tangent_slots.len(), ProgramError);

            let mut tangents = outputs.split_off(output_count).into_iter();
            return outputs
                .into_iter()
                .zip(output_activity)
                .map(|(primal, &live)| match live {
                    true => DifferentiationDual::new(primal, tangents.next().unwrap()),
                    false => DifferentiationDual::new_with_zero_tangent(primal),
                })
                .collect::<Result<Vec<_>, _>>();
        }

        let linearization = driver.linearize_program(callee, &input_indices, Some(&[]))?;

        // An output tangent that depends on no active input tangent (e.g., the tangent of a constant output or of
        // an output computed only from inactive inputs) is a structural zero, as under JAX's `which_nz_out`, so the
        // tangent callee is projected onto the live output tangents, and every other output is paired with a
        // structurally zero tangent. When no output tangent is live and the tangent callee has no observable effects,
        // no tangent call is bound, and the source callee is bound as the primal call, since nothing would consume its
        // residuals.
        //
        // The tangent program's leading inputs are compact active tangents, not the original primal input indices,
        // and its trailing residual inputs are held independent during dependence analysis. Linearization has
        // validated both boundaries, so subtracting the residual count cannot underflow.
        let tangent_program = linearization.tangent();
        let tangent_input_count = tangent_program.input_count() - linearization.residual_count();
        let tangent_slots = linearization.tangent_output_slots().to_vec();
        let mut dependence =
            tangent_program.output_dependence(&(0..tangent_input_count).collect::<Vec<_>>())?.into_iter();

        // Map compact tangent output dependence back to the original callee's output order. Outputs without tangent
        // slots consume no dependence entry, and trailing primal residual outputs are outside this boundary entirely.
        let output_activity = tangent_slots[..output_count]
            .iter()
            .map(|&slot| slot && dependence.next().unwrap())
            .collect::<Vec<_>>();
        let live_tangent_slots = (0..output_count)
            .filter(|&index| tangent_slots[index])
            .enumerate()
            .filter_map(|(slot, index)| output_activity[index].then_some(slot))
            .collect::<Vec<_>>();

        // Output projection retains the complete tangent/residual input boundary and observable effects. Keep the
        // shared callee handle when every slot is live, preserving the `Arc` identity used for callee interning.
        let tangent_program = if live_tangent_slots.len() == tangent_program.output_count() {
            tangent_program.clone()
        } else {
            Arc::new(tangent_program.as_ref().clone().with_outputs(&live_tangent_slots)?)
        };
        let primal_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        if !output_activity.contains(&true) && !tangent_program.effects().is_retained_when_unused() {
            let callee = shared_call_callee(callee)?;
            return context
                .primal()
                .bind(C::Operation::from(self.clone()), AttachedRegions::Callees(&[callee]), &primal_inputs)?
                .into_iter()
                .map(DifferentiationDual::new_with_zero_tangent)
                .collect();
        }
        let (primal_program, ..) = linearization.into_parts();

        // Wrap the primal sub-program in a fresh `call` and bind it over the primal inputs, recovering the primal
        // outputs followed by the residual values. The shared sub-program handles are attached directly, so repeated
        // binds of one derived callee intern by `Arc` identity instead of copying it again. The primal sub-program
        // preserves the callee's inputs in order, so the original lifted-capture prefix survives verbatim and its
        // length carries over. The primal call has the boundary shardings of this call, and its trailing residuals
        // have none.
        //
        // The primal and the tangent call carry the mesh of the function, and so does the transposed call that a
        // pullback derives from them, so an eager linearization (e.g., of an eager VJP) runs each of them where the
        // call runs. An uncommitted argument enters an eager linearization without the placement that it happens to
        // have (refer to `Value::relocatable_type`), so the tangent and the cotangent of such an argument may live
        // on the devices of the function.
        let boundary =
            self.effective_boundary_input_shardings(&input_types, primal_inputs.iter().map(|input| input.r#type()));
        let primal_call = C::Operation::from(CallOperation {
            function_mesh: self.function_mesh.clone(),
            ..self.with_derived_boundary_shardings(
                CallOperation::new(self.capture_count()).with_callee_region(primal_program.entry_region_ref())?,
                Some(&boundary),
                (0..input_types.len()).map(Some),
                (0..output_count).map(Some),
            )
        });
        let mut primal_call_outputs =
            context.primal().bind(primal_call, AttachedRegions::Callees(&[primal_program]), &primal_inputs)?;
        if primal_call_outputs.len() < output_count {
            return Err(ProgramError::MalformedProgram(format!(
                "`{}` primal program produced {} outputs which is fewer than its {} primal output(s)",
                CALL_OPERATION_NAME,
                primal_call_outputs.len(),
                output_count,
            ))
            .into());
        }
        let residuals = primal_call_outputs.split_off(output_count);
        let primal_outputs = primal_call_outputs;

        // Wrap the tangent sub-program in a fresh `call` and bind it over only the active input tangents followed by
        // the residual values. Structurally zero tangents are inactive, so every active input tangent is a live value
        // and materializing it only unwraps it, while inactive inputs (i.e., the capture prefix, structurally zero
        // tangents, inputs of zero differential spaces, and plumbing references) have no compact callee boundary slot.
        let mut tangent_inputs = inputs
            .iter()
            .zip(&activity)
            .filter(|(_, active)| **active)
            .map(|(input, _)| input.tangent().clone().materialize(context.tangent()))
            .collect::<Result<Vec<_>, _>>()?;
        tangent_inputs.extend(
            residuals.into_iter().map(|value| context.primal_to_tangent(value)).collect::<Result<Vec<_>, _>>()?,
        );

        // The tangent callee's inputs are live input tangents followed by residuals and never a lifted-capture prefix.
        // Attached region capture constants cannot appear in it because fresh-root region traces reject bodies that
        // register captures with `ProgramError::DiscardedCaptures`, and a stale reference that slipped in anyway would
        // fail lowering loudly against the tangent callee's empty capture prefix instead of aliasing an unrelated
        // value. Each active input tangent and each live output tangent has the boundary sharding of its primal,
        // and the trailing residual inputs have none.
        let active_inputs = activity.iter().enumerate().filter_map(|(index, active)| active.then_some(Some(index)));
        let live_outputs =
            output_activity.iter().enumerate().filter_map(|(index, active)| active.then_some(Some(index)));

        // Only the tangent call of an eager linearization keeps the mesh of the function, because only an eager context
        // applies it and the transposed call that a pullback derives from it, so that transposition may read that mesh
        // as where its cotangents live (refer to `refined_derived_value`). The tangent call of a staged linearization
        // is nested in a program and follows its caller.
        let function_mesh = self.function_mesh.clone().filter(|_| context.primal().is_eager());
        let tangent_call = C::Operation::from(CallOperation {
            function_mesh,
            ..self.with_derived_boundary_shardings(
                CallOperation::new(0).with_callee_region(tangent_program.entry_region_ref())?,
                Some(&boundary),
                active_inputs,
                live_outputs,
            )
        });
        let tangent_outputs =
            context
                .tangent()
                .bind(tangent_call, AttachedRegions::Callees(&[tangent_program]), &tangent_inputs)?;
        let tangent_output_count = output_activity.iter().filter(|&&active| active).count();
        check_count!("output", tangent_outputs, tangent_output_count, ProgramError);

        let mut tangent_outputs = tangent_outputs.into_iter();
        primal_outputs
            .into_iter()
            .zip(output_activity)
            .map(|(primal, active)| {
                if !active {
                    DifferentiationDual::new_with_zero_tangent(primal)
                } else {
                    let tangent = tangent_outputs.next().unwrap();
                    let refined = primal.r#type().tangent()?;
                    let tangent = refined_derived_value(context.tangent(), self.mesh(), refined, tangent)?;
                    DifferentiationDual::new(primal, tangent)
                }
            })
            .collect::<Result<Vec<_>, _>>()
    }
}

impl<V, O> TransposableOperation<V, O> for CallOperation<ArrayIrType>
where
    V: Value<Type = ArrayIrType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    O: Operation<Type = ArrayIrType>
        + From<CallOperation<ArrayIrType>>
        + From<CastOperation<ArrayIrType>>
        + OperationProjection<ArrayType, Projected: From<ReshardOperation> + From<ParallelReduceOperation>>
        + OperationProvider<ArrayIrType, AddOperation<ArrayIrType>, Operation = O>
        + OperationProvider<ArrayIrType, ReferenceNewOperation<ArrayType, ArrayIrType>, Operation = O>
        + ResidualZeroProvider<ArrayIrType, Operation = O>,
    TracingContext<V, O>: StagingContext<Type = ArrayIrType, Constant = V, Operation = O>,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TranspositionContext<V, O>,
        driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
        accumulators: &[CotangentAccumulator],
    ) -> Result<(), DifferentiationError> {
        // This is the transpose rule for a tangent `CallOperation`, which keeps its compilation boundary. The driver
        // transposes the attached callee under the linearity and cotangent destination masks of the inputs and the
        // structural zeros of its output cotangents, so instantiating this implementation introduces no recursive
        // `TransposableOperation` obligation on the operation family `O`. Known inputs supply residual or captured
        // values and may appear anywhere in the input boundary. The transposed call consumes the ordinary output
        // cotangents that are not structural zeros, live cotangent references, and known inputs, in that order.
        // A structural zero output cotangent seeds nothing, so it has no input in the transposed call and is not
        // materialized at the call boundary, although a dense backward rule inside the callee may still materialize
        // the zero that it consumes. Reference outputs forward input roots and therefore have no separate output
        // cotangent slot. Linear value inputs receive call results, reference inputs accumulate into their cotangent
        // destinations, and known or ignored inputs receive structural zeros.
        check_count!("output", outputs, driver.region(0)?.output_types().len(), ProgramError);
        check_count!("accumulator", accumulators, inputs.len(), DifferentiationError);

        // The callee is transposed with the destination kinds of the inputs, so the cotangent reference of a live
        // reference input (i.e., one with a `Reference` destination), which the context resolves and allocates on
        // first use, is an input of the transposed call, which accumulates into it in place and returns it by
        // identity, while a dead reference input (i.e., one with an `Ignore` destination) has no slot in the
        // transposed callee at all.
        let cotangents = context.cotangent_destinations(driver, inputs, accumulators)?;

        // A `call` operation with no live output cotangents and no live reference input is a zero linear map, so every
        // input cotangent is zero. A live reference input keeps the call live, because its accumulated state cotangent
        // flows through the transposed callee even when no ordinary output cotangent does. A callee with deferred work
        // or observable rule effects also keeps it live.
        check_count!("input", cotangents.kinds(), inputs.len(), ProgramError);
        if outputs.iter().all(MaybeZero::is_zero)
            && !cotangents.has_reference_state_destinations()
            && !driver.region(0)?.must_transpose()
        {
            for (accumulator, input) in accumulators.iter().zip(inputs) {
                accumulator.accumulate(context, MaybeZero::Zero(input.r#type().cotangent()?))?;
            }
            return Ok(());
        }

        // Each input maps to one callee input, independently linear (i.e., an input tangent) or known (i.e., a residual
        // value or the tangent of a captured constant). The linear inputs need not lead, because a captured compiled
        // function threads its captured prefix as known leading inputs. The dispatch guarantees that a `Known` input
        // carries its pullback value, so each known tracer is read directly in callee input order.
        let input_linearity = inputs.iter().map(PartialValue::is_unknown).collect::<Vec<_>>();
        let callee = driver.region(0)?;
        check_count!("input", input_linearity, callee.input_types().len(), ProgramError);

        // Transpose the callee with respect to its linear inputs and with its structural zero output cotangents
        // flagged, through the region's retained transform cache so that a callee shared by several outer programs is
        // transposed once per linearity and seed mask. A live linear reference input is transposed with a `Reference`
        // destination and a dead one with an `Ignore` destination, so the transposed callee maps
        // `[live_non_reference_output_cotangents..., cotangent_references..., known_input_values...]` to
        // `[linear_input_cotangents...]`, in callee input order, where a live reference input's cotangent is its
        // cotangent reference itself and a dead reference input has no cotangent slot. The flags follow the callee's
        // complete output order, and flags of reference outputs are normalized away because those outputs never own
        // a cotangent slot.
        let (input_indices, destination_kinds): (Vec<_>, Vec<_>) = input_linearity
            .iter()
            .zip(cotangents.kinds())
            .enumerate()
            .filter_map(|(index, (linear, kind))| linear.then_some((index, *kind)))
            .unzip();
        let output_types = callee.output_types();
        check_count!("output", outputs, output_types.len(), ProgramError);
        let zero_output_cotangents = outputs.iter().map(MaybeZero::is_zero).collect::<Vec<_>>();
        let transposed_callee = driver.transpose_program_with_zero_seeds(
            callee,
            &input_indices,
            &[],
            &zero_output_cotangents,
            &destination_kinds,
        )?;

        // Stage a fresh `call` over the transposed callee on `[live_outputs..., cotangent_references...,
        // known_input_values...]`. Its outputs are the cotangents of the linear inputs. A reference output forwards
        // a callee input root whose state cotangent lives in the cotangent reference of that input, so it owns no
        // cotangent slot.
        let mut transposed_inputs = Vec::with_capacity(output_types.len() + inputs.len());
        transposed_inputs.extend(
            outputs
                .iter()
                .zip(output_types.iter())
                .filter(|(_, output_type)| !output_type.is_reference())
                .filter_map(|(cotangent, _)| cotangent.as_value())
                .cloned(),
        );
        transposed_inputs.extend(cotangents.references().iter().cloned());
        transposed_inputs.extend(inputs.iter().filter_map(PartialValue::as_known).cloned());

        // The transposed callee's inputs are live output cotangents, cotangent references, and known input values, and
        // never a lifted-capture prefix. Attached region capture constants cannot appear in it because region bodies
        // are traced through fresh-root contexts. Each output cotangent has the cotangent sharding of the boundary
        // sharding of its output and each known input value the boundary sharding of its input, and the cotangent
        // references have none, as JAX's `_pjit_transpose` gives the cotangents the cotangent shardings of their
        // primals through `to_ct_spec`. Each returned linear input cotangent has the cotangent sharding of its input,
        // whichever other inputs have a structurally zero or absent cotangent, which returns nothing.
        let mut transposed_call =
            CallOperation::new(0).with_callee_region(transposed_callee.program().entry_region_ref())?;
        transposed_call.keeps_input_placements = self.keeps_input_placements();
        transposed_call.function_mesh = self.function_mesh.clone();

        // The known input values keep the effective boundary of the call, which, in the tangent call of a
        // linearization, places the inputs of the original call that it receives and leaves its residuals
        // where they are.
        let boundary =
            self.effective_boundary_input_shardings(&callee.input_types(), inputs.iter().map(PartialValue::r#type));

        // Seed inputs enter on the tangent result's cotangent sharding when the call already knows its devices. Eager
        // admission checks device commitments before applying these boundary shardings. The callee can then transpose
        // any placement-only result cast using the seed's admitted type.
        let function_mesh = self
            .function_mesh
            .as_ref()
            .filter(|function| function.logical_mesh().is_named())
            .map(|function| function.logical_mesh());
        let seed_shardings = outputs
            .iter()
            .zip(&output_types)
            .filter(|(cotangent, output_type)| !output_type.is_reference() && cotangent.as_value().is_some())
            .map(|(_, output_type)| {
                let ArrayIrType::Array(output_type) = output_type else { return Ok(None) };
                output_type
                    .sharding()
                    .filter(|sharding| Some(sharding.mesh()) == function_mesh)
                    .map(Sharding::cotangent)
                    .transpose()
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(TypeError::from)?;
        transposed_call.derived_input_shardings = Some(
            seed_shardings
                .into_iter()
                .chain(std::iter::repeat_n(None, cotangents.references().len()))
                .chain(
                    inputs
                        .iter()
                        .zip(boundary)
                        .filter(|(input, _)| input.as_known().is_some())
                        .map(|(_, sharding)| sharding),
                )
                .collect(),
        );

        if let Some(mesh) = self.mesh() {
            // An override places a primal value, and the cotangent of that value has the dual reduction state, so each
            // cotangent that crosses the boundary (i.e., an output cotangent that the transposed call receives and an
            // input cotangent that it returns) is placed at the cotangent sharding of the override of its primal, as
            // the cotangent types of those primals record it, and each known primal value is placed at the override
            // itself. An input cotangent that the transposed call returns is the cotangent of the argument as the
            // callee holds it once it has entered, so it carries no reduction state over auto mesh axes, which no
            // type records.
            let cotangent_sharding = |sharding: Option<&Sharding>| sharding.map(Sharding::cotangent).transpose();
            let output_cotangent_shardings = outputs
                .iter()
                .zip(output_types.iter())
                .enumerate()
                .filter(|(_, (cotangent, output_type))| !output_type.is_reference() && cotangent.as_value().is_some())
                .map(|(index, _)| cotangent_sharding(self.output_sharding(index)))
                .collect::<Result<Vec<_>, _>>()
                .map_err(TypeError::from)?;
            let reference_shardings = cotangents.references().iter().map(|_| None);
            let known_input_shardings = inputs
                .iter()
                .enumerate()
                .filter(|(_, input)| input.as_known().is_some())
                .map(|(index, _)| self.input_sharding(index).cloned());
            let input_shardings = output_cotangent_shardings
                .into_iter()
                .chain(reference_shardings)
                .chain(known_input_shardings)
                .collect();
            let output_shardings = input_indices
                .iter()
                .zip(transposed_callee.cotangents())
                .filter(|(_, cotangent)| **cotangent == TransposedProgramCotangent::Output)
                .map(|(&index, _)| cotangent_sharding(self.input_sharding(index)))
                .collect::<Result<Vec<_>, _>>()
                .map_err(TypeError::from)?
                .into_iter()
                .map(|sharding| sharding.map(|sharding| sharding.without_auto_reduction_axes()))
                .collect::<Vec<_>>();
            transposed_call = transposed_call.with_boundary(Some(mesh), input_shardings, output_shardings);
        }

        let transposed_call = O::from(transposed_call);
        let input_cotangents = context.bind(
            transposed_call,
            AttachedRegions::Callees(&[transposed_callee.program().clone()]),
            transposed_inputs.as_slice(),
        )?;
        let cotangent_types = input_indices
            .iter()
            .map(|&index| inputs[index].r#type().cotangent().map_err(ProgramError::from))
            .collect::<Result<Vec<_>, _>>()?;
        let mut input_cotangents =
            transposed_callee.assemble_cotangents(&cotangent_types, input_cotangents)?.into_iter();

        // Reassemble selected cotangents from the call's compact output boundary. Known and ignored inputs carry
        // structural zeros. Reference-state outputs only thread the caller-owned destination, so they do not contribute
        // a value cotangent to the enclosing context.
        let contributions = inputs
            .iter()
            .map(|input| {
                let cotangent = if input.is_unknown() { input_cotangents.next().unwrap() } else { None };
                match cotangent {
                    Some(MaybeZero::Value(mut cotangent)) if !input.r#type().is_reference() => {
                        let refined = input.r#type().cotangent().map_err(ProgramError::from)?;

                        // A call whose inputs vary along manual mesh axes beyond the callee input types runs its callee
                        // as if every input varied along all of them, as a JAX `jit` function is retraced with `pvary`
                        // inserted on the inputs that vary along fewer axes. Each shard along such an axis then holds
                        // one contribution to the cotangent of an input that does not vary along it, so the cotangent
                        // is summed over the manual mesh axes along which it varies beyond its refined type, with the
                        // `parallel_reduce` over a mesh that the transposition of `parallel_vary` stages.
                        let extra_variation = match (&refined, cotangent.r#type().as_ref()) {
                            (ArrayIrType::Array(refined), ArrayIrType::Array(actual)) => {
                                actual.sharding().map(|sharding| {
                                    let declared = refined.sharding().map(Sharding::varying_manual_axes);
                                    let axes = sharding
                                        .varying_manual_axes()
                                        .iter()
                                        .filter(|axis| !declared.is_some_and(|declared| declared.contains(*axis)))
                                        .cloned()
                                        .collect::<Vec<_>>();
                                    (sharding.mesh().clone(), axes)
                                })
                            }
                            _ => None,
                        };
                        if let Some((mesh, axes)) = extra_variation {
                            for axis in axes {
                                let reduction =
                                    ParallelReduceOperation::new(ReductionKind::Sum, axis).with_mesh(mesh.clone());
                                let reduction = <<O as OperationProjection<ArrayType>>::Projected>::from(reduction);
                                cotangent = context.bind(reduction, Vec::new(), &[cotangent])?.remove(0);
                            }
                        }

                        let cotangent = refined_derived_value(&**context, self.mesh(), refined, cotangent);
                        return Ok(MaybeZero::Value(cotangent?));
                    }
                    Some(cotangent) if !input.r#type().is_reference() => return Ok(cotangent),
                    _ => {}
                }
                input.r#type().cotangent().map(MaybeZero::Zero).map_err(ProgramError::from)
            })
            .collect::<Result<Vec<_>, ProgramError>>()?;

        check_count!("input", contributions, accumulators.len(), ProgramError);
        for (accumulator, contribution) in accumulators.iter().zip(contributions) {
            accumulator.accumulate(context, contribution)?;
        }

        Ok(())
    }
}

/// Placement request of one instruction of a program, as [`RegionPlacementRequests::requests`] returns it.
#[derive(Clone, Debug, PartialEq)]
pub enum RequestedPlacement {
    /// A request that names the devices of this mesh (i.e., a [`place`](crate::PlaceOperation) with a device target or
    /// a resolved target, or a [`call`](crate::CallOperation) whose callee names devices). A device target names no
    /// logical mesh, so it is resolved as the type rule of `place` resolves it (i.e., over the logical mesh of its
    /// input when the type of that input tracks a mesh of one device, and over the canonical mesh of one device
    /// otherwise).
    Devices(DeviceMesh),

    /// A [`place`](crate::PlaceOperation) with an in-place target, which only moves a value to another memory and names
    /// no devices.
    Memory,

    /// A [`constrain_sharding`](crate::ConstrainShardingOperation), which names a sharding over a logical mesh but no
    /// devices.
    Sharding,
}

/// Facts about the placements that the program of one region requests, every output of that region being used, which
/// [`Self::new`] and [`Self::for_target`] derive once for each region and target platform and retain with the region.
/// Every analysis of the placement requests of a region reads it, which includes the facts that a call reads of its
/// callee (refer to [`CallOperation::names_devices`] and [`CallOperation::has_placement_requests`]). The program that
/// it analyzes is the one that a backend runs for the region (i.e., the region pruned to the instructions that its
/// outputs and its observable effects depend on, refer to [`Program::into_pruned`]), so a request that nothing uses is
/// not among its requests. The regions of an instruction that the backend never runs are not analyzed either. These
/// include scan bodies with zero iterations on every target and regions that a backend omits for a given target
/// platform (refer to [`Self::for_target`]).
#[derive(Clone, Debug, PartialEq)]
pub struct RegionPlacementRequests {
    /// Placement requests of the program, in the order of its regions and of the instructions in each region, with
    /// the name of the operation that makes each request. Refer to [`Self::requests`] for more information.
    requests: Arc<[(RequestedPlacement, &'static str)]>,
}

impl RegionPlacementRequests {
    /// Returns the [`RegionPlacementRequests`] of the program of `region` without a target platform. Only scan bodies
    /// with zero iterations are omitted. Partial evaluation reads this summary, as does a backend that compiles a
    /// program before it knows its target platform. Refer to [`Self::for_target`] for how the summary is derived
    /// and retained.
    #[inline]
    pub fn new<V, O>(region: RegionRef<'_, V, O>) -> Self
    where
        V: 'static + Value<Type = ArrayIrType>,
        O: 'static + Operation<Type = ArrayIrType> + OperationPayloadProjection,
    {
        Self::derive(region, None, &|_, _| false)
    }

    /// Returns the [`RegionPlacementRequests`] of the program of `region` that a backend runs for `target_platform`,
    /// whose lowering omits the regions of each instruction for which `omits_regions` returns `true` (e.g., the body
    /// of a named composite call that the backend implements for that platform). `omits_regions` must depend on the
    /// instruction and `target_platform` alone, because the summary depends on `region` and `target_platform` alone,
    /// so it is derived once for each region and platform and retained with the region, and every later analysis of the
    /// same region and platform (e.g., of the callee of each eager application of a call, or of a call that partial
    /// evaluation forwards) only looks it up. Its derivation prunes the closure of `region` once and walks the regions
    /// of the pruned program once, deriving what each call of that program reads of its callee in the same pass, so it
    /// never analyzes a nested region on its own, and the cost of a region is linear in the size of its closure however
    /// deeply its calls nest. A region without a placement request pays one scan of its computation regions.
    #[inline]
    pub fn for_target<V, O>(
        region: RegionRef<'_, V, O>,
        target_platform: &str,
        omits_regions: impl Fn(RegionRef<'_, V, O>, &Instruction<O>) -> bool,
    ) -> Self
    where
        V: 'static + Value<Type = ArrayIrType>,
        O: 'static + Operation<Type = ArrayIrType> + OperationPayloadProjection,
    {
        Self::derive(region, Some(target_platform), &omits_regions)
    }

    /// Returns the placement requests of the program, in the order of its regions and of the instructions in each
    /// region, together with the name of the operation that makes each request. A request that no output of the region
    /// and no observable effect depends on is not among them, whether its instruction is dead in its own region (e.g.,
    /// a branch of a `condition` that drops the result of a `place`) or only computes an output of an instruction that
    /// carries regions that nothing uses (e.g., an output of a `condition` or of a `call` that the program drops).
    /// Nor is a request in the regions of an instruction that the backend never runs. Refer to
    /// [`RegionPlacementRequests`] for more information.
    #[inline]
    pub fn requests(&self) -> &[(RequestedPlacement, &'static str)] {
        &self.requests
    }

    /// Returns whether the program requests the placement of a value explicitly (i.e., in [`Self::requests`]), counting
    /// the requests that only move a value to another memory when `includes_memory_requests` is set. This is the one
    /// test of whether a region requests a placement, which the commitment of the results of a compiled function by
    /// a backend, whether a compiled function names the devices of its mesh, the placement decision of an eager call,
    /// and partial evaluation all read.
    #[inline]
    pub fn has_requests(&self, includes_memory_requests: bool) -> bool {
        self.requests.iter().any(|(request, _)| match request {
            RequestedPlacement::Memory => includes_memory_requests,
            RequestedPlacement::Devices(_) | RequestedPlacement::Sharding => true,
        })
    }

    /// Returns a function that answers, for each region of the program whose entry region is `entry`, whether that
    /// region requests a placement that names devices (i.e., any request except one that only moves a value to another
    /// memory, refer to [`Self::has_requests`]) in the regions that the backend runs, which is what a call that records
    /// a mesh reads of its callee (refer to [`CallOperation::names_devices`]). The program must be the one that the
    /// backend runs (i.e., it is already pruned, refer to [`Program::into_pruned`]), because each region is analyzed as
    /// the program holds it. The regions of the instructions for which `omits_regions` returns `true` are not analyzed,
    /// and neither is the body of a scan with zero iterations (refer to [`ScanOperation::has_zero_iterations`]). The
    /// function analyzes each region at most once and reads its fact for every call of it and for every region that
    /// encloses it, so that answering for every call of a program with nested calls costs one pass over the program,
    /// whereas the summary of each callee (refer to [`Self::for_target`]) would prune and walk the closure of each
    /// callee again.
    #[inline]
    pub fn names_devices_predicate<'o, V, O>(
        entry: RegionRef<'_, V, O>,
        omits_regions: &'o dyn Fn(RegionRef<'_, V, O>, &Instruction<O>) -> bool,
    ) -> impl FnMut(RegionRef<'_, V, O>) -> bool + use<'o, V, O>
    where
        V: Value<Type = ArrayIrType>,
        O: 'static + Operation<Type = ArrayIrType> + OperationPayloadProjection,
    {
        let mut analysis = DeviceNamingAnalysis::new(entry.arena().len(), omits_regions);
        move |region| analysis.names_devices(region)
    }

    /// Returns the summary of `region` for `target_platform`, whose regions that the backend omits are those for which
    /// `omits_regions` returns `true`, retained with `region` for that platform.
    fn derive<V, O>(
        region: RegionRef<'_, V, O>,
        target_platform: Option<&str>,
        omits_regions: &dyn Fn(RegionRef<'_, V, O>, &Instruction<O>) -> bool,
    ) -> Self
    where
        V: 'static + Value<Type = ArrayIrType>,
        O: 'static + Operation<Type = ArrayIrType> + OperationPayloadProjection,
    {
        /// [`Transform`] marker under which [`RegionPlacementRequests::for_target`] retains the summary of a region
        /// with it, for the target platform that its arguments name ([`None`] for no target), held as a [`String`]
        /// because the region-owned cache only retains owned arguments. Two summaries are retained, because an eager
        /// call reads the requests of its callee for the platform of its backend while partial evaluation and
        /// compilation read them for no target, and one summary would be derived again at every alternation.
        struct RegionPlacementAnalysis;

        impl<V: Value, O: Operation<Type = V::Type>> Transform<Region<V, O>> for RegionPlacementAnalysis {
            type Arguments = Option<String>;
            type Artifact = TransformArtifact<V, O, RegionPlacementRequests>;

            const DEFAULT_CACHE_CAPACITY: usize = 2;
        }

        let arguments = target_platform.map(str::to_owned);
        let Ok(artifact) = region.transform::<RegionPlacementAnalysis, _, Infallible>(arguments, |region, _| {
            // A request of the program that the backend runs is also a request of the program as it is, so a region
            // without any skips pruning. The same analysis classifies both, so that they never disagree on which
            // instructions request a placement.
            let mut naming = DeviceNamingAnalysis::new(region.arena().len(), omits_regions);
            let has_requests = region.computation_regions().any(|region| {
                region.instructions().iter().any(|instruction| naming.request(region, instruction).is_some())
            });
            let summary = if !has_requests {
                Self { requests: Arc::from([]) }
            } else {
                let program = region.to_program();
                let pruned = program.clone().into_pruned().and_then(|program| program.into_simplified());

                // Inspect the original program if pruning or simplification fails so the analysis stays conservative.
                let entry = pruned.as_ref().unwrap_or(&program).entry_region_ref();
                let mut naming = DeviceNamingAnalysis::new(entry.arena().len(), omits_regions);
                let mut requests = Vec::new();
                let mut visited = vec![false; entry.arena().len()];
                let mut pending = vec![entry.id()];

                // Follow the order of `RegionRef::computation_regions`, skipping regions the backend omits.
                while let Some(id) = pending.pop() {
                    if std::mem::replace(&mut visited[id.index()], true) {
                        continue;
                    }

                    let Ok(region) = entry.with_id(id) else { continue };
                    for instruction in region.instructions() {
                        if let Some(request) = naming.request(region, instruction) {
                            requests.push((request, instruction.operation().name()));
                        }

                        if !naming.skips_regions(region, instruction) {
                            pending.extend(instruction.regions().iter().copied().enumerate().filter_map(
                                |(index, id)| {
                                    (instruction.operation().region_role(index) == Some(RegionRole::Computation))
                                        .then_some(id)
                                },
                            ));
                        }
                    }
                }

                Self { requests: Arc::from(requests) }
            };

            Ok(TransformArtifact::new(Vec::new(), summary))
        });

        artifact.metadata().clone()
    }
}

/// Formal callee input types and caller result types that bind [`TypeIdentity`](crate::TypeIdentity)s across one
/// sealed call boundary. The bindings are encoded in these two type signatures, and the stored metadata contains no
/// runtime values.
///
/// An output identity present in the callee inputs is forwarded from the corresponding caller input. An identity
/// defined inside the callee receives one fresh caller identity shared by every corresponding result occurrence.
/// Separate calls receive independent local identities, because their results can have different extents even when
/// their input shapes agree. [`Self::fresh`] allocates those identities once during call construction, before
/// interpretation, so repeated type inference returns the same identities. Calls whose results only forward input
/// identities retain no instance of this metadata.
///
/// The formal input types allow [`Self::instantiated`] to specialize forwarded dimensions to actual inputs without
/// replacing the call's local result identities. When a transformation already owns the caller result names,
/// [`Self::with_outputs`] preserves them and validates their correspondence to the callee's output types.
/// [`Self::renaming`] recovers that correspondence from the current callee output types to the stored caller types.
///
/// For example, suppose a callee accepts a dimension with identity `n` and returns three dimensions with identities
/// `n`, `m`, and `m`, where `m` is defined inside the callee. The first result forwards `n`, and the other two share
/// one fresh caller identity. Another call receives a different identity for `m`, even though its diagnostic name can
/// remain `m`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct CallTypeBindings<T: Type> {
    /// Formal input namespace used when instantiating forwarded result identities.
    inputs: Arc<[T]>,

    /// Caller result types, with callee-local identities already freshly bound.
    outputs: Arc<[T]>,
}

impl<T: Type> CallTypeBindings<T> {
    /// Freshens callee-local result identities, returning no metadata when all result identities are forwarded.
    ///
    /// # Errors
    ///
    /// Returns a type error when a result type cannot preserve its structure under identity renaming.
    fn fresh(inputs: &[T], outputs: &[T]) -> Result<Option<Self>, TypeError> {
        if !outputs
            .iter()
            .flat_map(Type::identities)
            .any(|(_, identity)| !inputs.iter().flat_map(Type::identities).any(|(_, input)| input == identity))
        {
            return Ok(None);
        }

        let mut unavailable = inputs
            .iter()
            .chain(outputs)
            .flat_map(Type::identities)
            .map(|(_, identity)| identity.clone())
            .collect::<Vec<_>>();

        let mut renaming = TypeIdentityRenaming::new();
        for (_, identity) in outputs.iter().flat_map(Type::identities) {
            if renaming.target(identity).is_none()
                && !inputs.iter().flat_map(Type::identities).any(|(_, input)| input == identity)
            {
                unavailable.push(renaming.insert_fresh(identity.clone(), &unavailable)?);
            }
        }

        let outputs =
            outputs.iter().map(|r#type| r#type.rename_identities(&renaming)).collect::<Result<Vec<_>, _>>()?;
        Ok(Some(Self { inputs: inputs.into(), outputs: outputs.into() }))
    }

    /// Records caller result types whose local identities have already been chosen, validating their exact
    /// correspondence to the callee's output types. Unlike [`Self::fresh`], this function allocates no identities.
    /// Forwarded identities must stay unchanged, and callee-local identities must have fresh caller counterparts.
    /// Returns no metadata when both output signatures agree and only forward input identities.
    ///
    /// # Parameters
    ///
    ///   - `inputs`: Callee input types. Both output signatures use this input scope for forwarded identities.
    ///   - `callee_outputs`: Output types of the attached callee, using its local result identities.
    ///   - `outputs`: Result types retained by the caller, using the existing caller-owned result identities.
    ///
    /// # Errors
    ///
    /// Returns a type error for changed result structure, captured local names, or renamed forwarded identities.
    fn with_outputs(inputs: &[T], callee_outputs: &[T], outputs: Vec<T>) -> Result<Option<Self>, TypeError> {
        if callee_outputs == outputs {
            if callee_outputs
                .iter()
                .flat_map(Type::identities)
                .any(|(_, identity)| !inputs.iter().flat_map(Type::identities).any(|(_, input)| input == identity))
            {
                return Err(TypeError::invalid(format!(
                    "`{CALL_OPERATION_NAME}` local result declarations must use fresh identities",
                )));
            }
            return Ok(None);
        }
        let signature = Self { inputs: inputs.into(), outputs: outputs.into() };
        signature.renaming(inputs, callee_outputs)?;
        Ok(Some(signature))
    }

    /// Returns the caller-owned identities defined by this call, once each in result occurrence order.
    fn definitions(&self) -> Vec<T::Identity> {
        let mut definitions = Vec::new();
        for (_, identity) in self.outputs.iter().flat_map(Type::identities) {
            if !self.inputs.iter().flat_map(Type::identities).any(|(_, input)| input == identity)
                && !definitions.contains(identity)
            {
                definitions.push(identity.clone());
            }
        }
        definitions
    }

    /// Returns caller result types with forwarded names and ordinary shape refinements instantiated at `inputs`.
    /// Local result identities remain symbolic even when the input facts determine an exact extent.
    ///
    /// # Errors
    ///
    /// Returns the input-scope correspondence or type-renaming error when the call signature is incompatible.
    fn instantiated(&self, inputs: &[T]) -> Result<Vec<T>, TypeError> {
        if self.inputs.as_ref() == inputs {
            return Ok(self.outputs.to_vec());
        }
        let renaming = T::derive_identity_renaming(self.inputs.as_ref(), inputs)?;
        let refinements = T::Refinements::from_local_declarations(self.inputs.iter(), inputs)?;
        let local = self.definitions();
        self.outputs
            .iter()
            .map(|r#type| refinements.refine_declared(r#type, &local)?.rename_identities(&renaming))
            .collect()
    }

    /// Returns the validated renaming from the current callee output types to this call's instantiated result types.
    /// The source types are supplied in `callee_outputs`, and the target types come from [`Self::instantiated`]. For
    /// the type-level example, this maps the callee's local `m` to the caller's fresh identity while preserving
    /// forwarded `n`. Caller result types alone cannot recover the callee's original local identities.
    ///
    /// # Parameters
    ///
    ///   - `inputs`: Input types of the current callee instantiation, used to instantiate forwarded result identities.
    ///   - `callee_outputs`: Output types of that callee instantiation, before renaming its local identities into the
    ///     caller's scope.
    ///
    /// # Errors
    ///
    /// Returns a type error when local names are not fresh, forwarded names change, or result structure differs.
    fn renaming(&self, inputs: &[T], callee_outputs: &[T]) -> Result<TypeIdentityRenaming<T::Identity>, TypeError> {
        let outputs = self.instantiated(inputs)?;
        let renaming = T::derive_identity_renaming(callee_outputs, &outputs)?;
        for (_, identity) in callee_outputs.iter().flat_map(Type::identities) {
            let target = renaming.target(identity).unwrap_or(identity);
            let forwarded = inputs.iter().flat_map(Type::identities).any(|(_, input)| input == identity);
            if forwarded && target != identity {
                return Err(TypeError::invalid(format!(
                    "`{CALL_OPERATION_NAME}` result declarations must preserve forwarded input identities",
                )));
            }
            if !forwarded
                && (target == identity
                    || inputs
                        .iter()
                        .chain(callee_outputs)
                        .flat_map(Type::identities)
                        .any(|(_, original)| original == target))
            {
                return Err(TypeError::invalid(format!(
                    "`{CALL_OPERATION_NAME}` local result declarations must use fresh identities",
                )));
            }
        }
        Ok(renaming)
    }

    /// Renames caller-owned definitions while retaining the callee's formal input namespace.
    ///
    /// # Errors
    ///
    /// Returns an identity conflict or type-renaming error.
    fn renamed(&self, renaming: &TypeIdentityRenaming<T::Identity>) -> Result<Self, TypeError> {
        let mut owned = TypeIdentityRenaming::new();
        for (source, target) in renaming.replacements() {
            if !self.inputs.iter().flat_map(Type::identities).any(|(_, input)| input == source) {
                owned.insert(source.clone(), target.clone())?;
            }
        }
        let outputs =
            self.outputs.iter().map(|r#type| r#type.rename_identities(&owned)).collect::<Result<Vec<_>, _>>()?;
        Ok(Self { inputs: self.inputs.clone(), outputs: outputs.into() })
    }

    /// Retains the indicated input and result positions after ordinary call-boundary pruning.
    ///
    /// # Errors
    ///
    /// Returns a type error when a positional mask has the wrong arity.
    fn pruned(&self, inputs: &[bool], outputs: &[bool]) -> Result<Self, TypeError> {
        if inputs.len() != self.inputs.len() || outputs.len() != self.outputs.len() {
            return Err(TypeError::invalid(format!(
                "`{CALL_OPERATION_NAME}` result declaration pruning masks have incompatible arity",
            )));
        }
        Ok(Self {
            inputs: self
                .inputs
                .iter()
                .zip(inputs)
                .filter_map(|(r#type, kept)| kept.then_some(r#type.clone()))
                .collect(),
            outputs: self
                .outputs
                .iter()
                .zip(outputs)
                .filter_map(|(r#type, kept)| kept.then_some(r#type.clone()))
                .collect(),
        })
    }
}

/// Whether each region of one program requests a placement that names devices (i.e., any request except one that only
/// moves a value to another memory) in the regions that the backend runs, which is what a call that records a mesh
/// reads of its callee. Each region of the program is analyzed at most once, and its fact is read by every call of
/// it and by every region that encloses it, so a pass over a program with nested calls is linear in the size of the
/// program. The program must be the one that the backend runs, because a callee is analyzed as the program holds it.
/// The regions of an instruction that the backend omits are not analyzed.
struct DeviceNamingAnalysis<'o, V: Value<Type = ArrayIrType>, O: Operation<Type = ArrayIrType>> {
    /// Whether the region at each index of the arena of the program names devices, or [`None`] if it has not been
    /// analyzed yet.
    names_devices: Vec<Option<bool>>,

    /// Returns whether the backend omits the regions of an instruction for the target platform of the analysis.
    omits_regions: &'o dyn Fn(RegionRef<'_, V, O>, &Instruction<O>) -> bool,
}

impl<'o, V, O> DeviceNamingAnalysis<'o, V, O>
where
    V: Value<Type = ArrayIrType>,
    O: 'static + Operation<Type = ArrayIrType> + OperationPayloadProjection,
{
    /// Creates an analysis of a program whose arena holds `region_count` regions, whose backend omits the regions of
    /// the instructions for which `omits_regions` returns `true`.
    fn new(region_count: usize, omits_regions: &'o dyn Fn(RegionRef<'_, V, O>, &Instruction<O>) -> bool) -> Self {
        Self { names_devices: vec![None; region_count], omits_regions }
    }

    /// Returns whether the backend never runs the regions of `instruction` of `region`. This holds on every target for
    /// the body of a scan with zero iterations (i.e., [`ScanOperation::has_zero_iterations`]) and for the regions for
    /// which the `omits_regions` of this analysis returns `true`.
    fn skips_regions(&self, region: RegionRef<'_, V, O>, instruction: &Instruction<O>) -> bool {
        instruction
            .operation()
            .projected_payload::<ScanOperation<ArrayIrType>>()
            .is_some_and(ScanOperation::has_zero_iterations)
            || (self.omits_regions)(region, instruction)
    }

    /// Returns whether `region` requests a placement that names devices in the regions that the backend runs for it.
    fn names_devices(&mut self, region: RegionRef<'_, V, O>) -> bool {
        if let Some(names_devices) = self.names_devices[region.id().index()] {
            return names_devices;
        }

        let mut names_devices = false;
        for instruction in region.instructions() {
            let request = self.request(region, instruction);
            names_devices |= matches!(request, Some(RequestedPlacement::Devices(_) | RequestedPlacement::Sharding));

            if self.skips_regions(region, instruction) {
                continue;
            }

            for (index, &nested) in instruction.regions().iter().enumerate() {
                if instruction.operation().region_role(index) == Some(RegionRole::Computation)
                    && let Ok(nested) = region.with_id(nested)
                {
                    names_devices |= self.names_devices(nested);
                }
            }
        }

        self.names_devices[region.id().index()] = Some(names_devices);
        names_devices
    }

    /// Returns the placement request that `instruction` of `region` makes explicitly, if any, which is the one
    /// classification of the placement requests of a program. A `place` with a device target is resolved as
    /// [`RequestedPlacement::Devices`] documents, and a call that records a mesh requests a placement only when
    /// it names devices, which a call that pruning left without the request that made it name devices does not.
    fn request(&mut self, region: RegionRef<'_, V, O>, instruction: &Instruction<O>) -> Option<RequestedPlacement> {
        let operation = instruction.operation();
        if let Some(place) = operation.projected_payload::<PlaceOperation>() {
            return match place.target()? {
                PlacementRequest::InPlace { .. } => Some(RequestedPlacement::Memory),
                PlacementRequest::Device { device, .. } => {
                    let input_mesh = instruction.inputs().first().and_then(|input| {
                        let input_type = region.atoms()[input.index()].r#type();
                        let ArrayIrType::Array(array_type) = input_type.as_ref() else { return None };
                        array_type.sharding().map(|sharding| sharding.mesh().clone())
                    });
                    let logical_mesh =
                        input_mesh.filter(|mesh| mesh.device_count() == 1).unwrap_or_else(LogicalMesh::single_device);

                    // A logical mesh of one device always forms a mesh with one device.
                    Some(RequestedPlacement::Devices(DeviceMesh::new(logical_mesh, vec![*device]).unwrap()))
                }
                PlacementRequest::Resolved(placement) => Some(RequestedPlacement::Devices(placement.mesh().clone())),
            };
        }

        if operation.projected_payload::<ConstrainShardingOperation>().is_some() {
            return Some(RequestedPlacement::Sharding);
        }

        let call = operation.projected_payload::<CallOperation<ArrayIrType>>()?;
        call.mesh()?;
        let callee = region.with_id(instruction.regions()[0]).ok()?;
        let mesh = call.names_devices(|| self.names_devices(callee))?;
        Some(RequestedPlacement::Devices(mesh.clone()))
    }
}

/// Kind of a split of the callee of a `call` that [`CallOperation::split`] retains.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum CallSplitKind {
    /// Split of a call whose inputs are partially known, marked in input order.
    Knownness(Vec<bool>),

    /// Split of an all-known call on residual demand for the outputs of the callee that the residual planner demands,
    /// marked in output order.
    ResidualDemand(Vec<bool>),
}

// TODO(eaplatanios): Review from here onwards.

/// Returns the callee of a `call` bound over `callee` as a shared program, which is the program that `callee` was
/// interned from, if any, and otherwise a copy of `callee` that the region transform cache of `callee` retains.
fn shared_call_callee<V: Value<Type = ArrayIrType>, O: Operation<Type = ArrayIrType>>(
    callee: RegionRef<'_, V, O>,
) -> Result<Arc<Program<V, O, Vec<V>, Vec<V>>>, ProgramError> {
    /// [`Region`] [`Transform`] marker for the shared copy of the callee of a `call` that this function retains for a
    /// callee region that is not interned as a shared region (e.g., the entry region of a shared callee program that a
    /// rule binds directly, such as the primal or tangent callee of a linearized call). Binding that copy instead of a
    /// fresh copy of the region lets every call of one callee share one region.
    struct CallCalleeTransform;

    impl<V, O> Transform<Region<V, O>> for CallCalleeTransform
    where
        V: Value<Type = ArrayIrType>,
        O: Operation<Type = ArrayIrType>,
    {
        type Arguments = ();
        type Artifact = TransformArtifact<V, O, ()>;

        const DEFAULT_CACHE_CAPACITY: usize = 1;
    }

    if let Some(callee) = callee.interned_callee() {
        return Ok(callee.clone());
    }

    let artifact = callee.transform::<CallCalleeTransform, _, ProgramError>((), |callee, _| {
        Ok(TransformArtifact::new(vec![Arc::new(callee.to_program())], ()))
    })?;
    Ok(artifact.programs()[0].clone())
}

/// Returns whether an all-known call of `callee` with `inputs` keeps its boundary under a residual policy, which is the
/// case when `callee` has effects, references, nominal type identities, or capture constants. Splitting such a call on
/// residual demand, or forwarding its outputs, would move effects, reference accesses, nominally scoped types, or
/// captures that name its capture prefix across the boundary that validates them.
fn keeps_call_boundary<V: Value<Type = ArrayIrType>, O: Operation<Type = ArrayIrType>, W: Value>(
    callee: RegionRef<'_, V, O>,
    inputs: &[PartialEvaluationValue<W>],
) -> bool {
    !callee.effects().classes().is_empty() || !can_cache_call_split(callee, inputs)
}

/// Returns whether a mixed call can retain its structural partition. Effects are permitted because its cache key
/// includes their partition configuration, whereas references, captures, and nominal identities require the original
/// boundary or call-local identities. All-known replay additionally excludes effects (refer to
/// [`keeps_call_boundary`]).
fn can_cache_call_split<V: Value<Type = ArrayIrType>, O: Operation<Type = ArrayIrType>, W: Value>(
    callee: RegionRef<'_, V, O>,
    inputs: &[PartialEvaluationValue<W>],
) -> bool {
    !inputs
        .iter()
        .any(|input| input.r#type().is_reference() || input.r#type().identities().next().is_some())
        && !callee.contains_atom_type_in_closure(|r#type| r#type.is_reference() || r#type.identities().next().is_some())
        && !callee.contains_atom_in_closure(|atom| {
            atom.as_constant().is_some_and(|constant| constant.capture_index().is_some())
        })
}

/// Returns `callee`, the callee of a call whose effective boundary is `boundary` (refer to
/// `CallOperation::effective_input_boundary`), with each input type that declares no sharding refined to the
/// sharding of its boundary, as far as a type records it, or [`None`] when no input type changes. The batching rules
/// of a call batch this callee rather than the original one, so that type inference records the placement that the
/// batch dimension of each mapped input has at the boundary of the batched call on the values that the batched callee
/// computes, as it does for a callee whose input type declares that sharding (e.g., the batch split along an explicit
/// mesh axis of a callee compiled for an item type without a sharding stays on its outputs instead of being gathered).
/// The batched callee is a derived program whose input types are only traced types, so refining them changes no
/// boundary of the original call. A sharding over auto mesh axes refines a type only as far as types record it (i.e.,
/// with those axes replicated), and a sharding that does not refine the type (e.g., one with a pending reduction) is
/// not applied.
fn with_boundary_input_types<T, V, O>(
    callee: RegionRef<'_, V, O>,
    boundary: &[Option<Sharding>],
) -> Result<Option<Program<V, O, Vec<V>, Vec<V>>>, ProgramError>
where
    T: ArrayMemberType + From<ArrayType>,
    V: Value<Type = T>,
    O: Operation<Type = T>,
    TracingContext<V, O>: StagingContext<Type = T, Constant = V, Operation = O>,
{
    let declared = callee.input_types();
    let mut refined = None::<Vec<T>>;
    for (index, (declared_type, sharding)) in declared.iter().zip(boundary).enumerate() {
        let (Ok(array_type), Some(sharding)) = (declared_type.as_array_type(), sharding) else { continue };
        // A boundary that states more than a placement (e.g., a pending reduction) does not refine a type without a
        // sharding, which describes a reduced value, and is left to the boundary of the batched call.
        let Some(boundary_type) = CallOperation::boundary_input_type(array_type, sharding) else { continue };
        refined.get_or_insert_with(|| declared.to_vec())[index] = T::from(boundary_type);
    }
    refined.map(|refined| callee.to_program().specialize(&refined)).transpose()
}

/// Returns `sharding`, the boundary sharding of each item of a value that batching maps along `axis`, with the batch
/// dimension inserted at that axis, `sharding` itself for an unmapped value, and [`None`] without a sharding or when
/// the batch dimension cannot be inserted. The per-item override states nothing about the batch dimension, because an
/// explicit placement under batching applies to each batch item, so the batch dimension keeps the placement that
/// `packed`, the type of the batched value, records for it (e.g., split along a mesh axis by the batch specification),
/// and is replicated when that type records none (e.g., over auto mesh axes, which types never record).
///
/// # Errors
///
/// Returns the error of [`Sharding::batched`] for a batch placement that conflicts with `sharding` (e.g., a batch split
/// along a mesh axis along which `sharding` already splits each item), which no placement of the batched value can
/// satisfy, as the batching of a compiled function by a backend rejects it.
fn batched_boundary_sharding(
    sharding: Option<&Sharding>,
    axis: &BatchAxis,
    packed: Option<&ArrayType>,
) -> Result<Option<Sharding>, BatchingError> {
    let Some(sharding) = sharding else {
        return Ok(None);
    };
    let Some(axis) = axis.axis() else {
        return Ok(Some(sharding.clone()));
    };
    let rank = sharding.dimensions().len() as i128 + 1;
    let position = if axis.value() < 0 { axis.value() + rank } else { axis.value() };
    let Some(position) = usize::try_from(position).ok().filter(|&position| (position as i128) < rank) else {
        return Ok(None);
    };
    let batch_dimension = packed
        .and_then(ArrayType::sharding)
        .filter(|packed| packed.rank() as i128 == rank && packed.mesh().has_same_topology(sharding.mesh()))
        .map_or(ShardingDimension::Replicated, |packed| packed.dimensions()[position].clone());
    sharding.batched(position, batch_dimension).map(Some)
}

/// Returns `value`, a tangent or cotangent that a derived call of a callee returns at a type of the callee boundary, at
/// `refined`, the type that the value has in the caller (e.g., the tangent type of a primal output, or the cotangent
/// type of a primal input). A call accepts inputs whose types refine the callee input types (refer to
/// `CallOperation::validate_input_types`), so a primal that a partial evaluation computes from such inputs (e.g.,
/// an eager array, whose type records its sharding) can refine the type of the callee boundary, while the derived call
/// that computes the tangent or the cotangent of that primal returns it at the type of the callee boundary. The value
/// has the shape and placement of the primal, so an unchecked cast in `context` states its refined type. A value that a
/// call whose callee names devices on `mesh` returns lives on those devices, so it is cast only to a type over a mesh
/// of the same topology. A value whose refined type names a different mesh is returned unchanged.
///
/// A value whose type differs from `refined` in its sharding only in dimensions that one of them leaves unconstrained
/// (i.e., `{?}`) and the other replicates is cast to `refined` as
/// well (e.g., the cotangent of an input that a call takes through an unconstrained input override). This is an
/// unchecked cast rather than a `reshard`, because the value is already placed as `refined` states. A runtime array
/// stores an unconstrained dimension replicated, so its buffers are those of `refined`. Inside a compiled program, the
/// cast emits no sharding constraint, because an unconstrained and a replicated dimension do not conflict over explicit
/// mesh axes and types record no placement over auto mesh axes, and the entry boundary of the program replicates an
/// unconstrained dimension of any value that it exchanges.
///
/// A value whose type tracks a sharding, where `refined` tracks none, is cast to `refined` as well when a call accepts
/// an input of type `refined` for a callee input of the type of the value, as the cotangent of an input that a staged
/// caller of untracked type passes to a callee with an input sharding override is. This mirrors the forward direction,
/// where the call takes that input as it is (refer to `CallOperation::input_variation`) and its lowering constrains the
/// input to the override, so the derivative returns to the type of the caller input. The cast only stops tracking the
/// placement of the value, which a type without a sharding states nothing about, and keeps its values and buffers.
///
/// A value whose type differs from `refined` only in the placement that its sharding tracks over a mesh of the same
/// topology (e.g., the cotangent of a replicated argument of a callee that declares that input split, which the call
/// places at its effective boundary, refer to [`CallOperation::boundary_input_sharding`]) is resharded to `refined`,
/// so that it combines with the other cotangents of that argument, as JAX returns the cotangent of a `jit` argument
/// with the sharding of that argument.
fn refined_derived_value<C>(
    context: &C,
    mesh: Option<&DeviceMesh>,
    refined: ArrayIrType,
    value: C::Value,
) -> Result<C::Value, ProgramError>
where
    C: Context<
            Type = ArrayIrType,
            Operation: From<CastOperation<ArrayIrType>>
                           + OperationProjection<ArrayType, Projected: From<ReshardOperation>>,
        >,
{
    let declared = value.r#type().into_owned();
    let moves_devices = mesh.is_some_and(|mesh| match &refined {
        ArrayIrType::Array(refined) => {
            refined.sharding().is_some_and(|sharding| !sharding.mesh().has_same_topology(mesh.logical_mesh()))
        }
        _ => false,
    });

    // Two distinct shardings place an array alike when they have the same logical mesh, rank, reduction state, and
    // manual variation, and each of their dimensions is either equal in both or unconstrained (i.e., `{?}`) in one and
    // replicated in the other, because a runtime array stores an unconstrained dimension replicated. The final
    // comparison establishes everything but the dimensions, by giving one sharding the dimensions of the other.
    let shardings = match (&declared, &refined) {
        (ArrayIrType::Array(declared), ArrayIrType::Array(refined)) => declared.sharding().zip(refined.sharding()),
        _ => None,
    };
    let replicates_unconstrained_dimensions = shardings.is_some_and(|(declared, refined)| {
        declared != refined
            && declared.rank() == refined.rank()
            && declared.dimensions().iter().zip(refined.dimensions()).all(|dimensions| match dimensions {
                (ShardingDimension::Unconstrained, ShardingDimension::Replicated)
                | (ShardingDimension::Replicated, ShardingDimension::Unconstrained) => true,
                (declared, refined) => declared == refined,
            })
            && declared.with_dimensions(refined.dimensions().to_vec()).is_ok_and(|declared| &declared == refined)
    });

    // A value whose sharding differs from the refined one only in the placement that it tracks, over a mesh of the
    // same topology, is resharded. Shardings that place the value alike need no reshard and are cast below.
    if let (ArrayIrType::Array(declared), ArrayIrType::Array(refined)) = (&declared, &refined)
        && let (Some(declared_sharding), Some(refined_sharding)) = (declared.sharding(), refined.sharding())
        && !moves_devices
        && declared_sharding != refined_sharding
        && !replicates_unconstrained_dimensions
        && declared_sharding.mesh().has_same_topology(refined_sharding.mesh())
        && declared_sharding.unreduced_axes() == refined_sharding.unreduced_axes()
        && declared_sharding.reduced_axes() == refined_sharding.reduced_axes()
        && declared_sharding.varying_manual_axes() == refined_sharding.varying_manual_axes()
        && declared.clone().with_sharding(refined_sharding.clone()).is_ok_and(|declared| &declared == refined)
    {
        let reshard = ReshardOperation::new(refined_sharding.clone());
        let reshard = <<C::Operation as OperationProjection<ArrayType>>::Projected>::from(reshard);
        return Ok(context.bind(reshard, Vec::new(), &[value])?.remove(0));
    }

    if moves_devices || refined.is_interchangeable_with(&declared) {
        return Ok(value);
    }

    // The value is cast when the refined type refines its type as it is, and in the two cases below, which the
    // refinement of types alone does not cover.
    let casts = declared.is_refined_by(&refined)
        || match (&declared, &refined) {
            (ArrayIrType::Array(declared), ArrayIrType::Array(refined)) => {
                // The refined type refines the type of the value once its sharding is replaced by that of the value,
                // where the two shardings place the value alike.
                let is_refined_up_to_unconstrained_dimensions = replicates_unconstrained_dimensions && {
                    let resharded = refined.clone().with_sharding(declared.sharding().cloned());
                    declared.is_refined_by(&resharded.map_err(TypeError::from)?)
                };

                // The refined type tracks no sharding and is a type that a call accepts for a callee input of the type
                // of the value without varying along more manual mesh axes, which is when the type of the value has the
                // shape that the refined type refines and a sharding without reduction state and without manual
                // variation. A sharding that partitions a dimension along manual mesh axes is excluded, because a cast
                // cannot change placement over manual mesh axes, which inside a manual region also determines the shape
                // of the local value.
                let is_untracked_refinement = refined.sharding().is_none()
                    && declared.sharding().is_some_and(|sharding| {
                        sharding.dimensions().iter().all(|dimension| dimension.manual_axes(sharding.mesh()).is_empty())
                    })
                    && CallOperation::input_variation(declared, refined).is_some();

                is_refined_up_to_unconstrained_dimensions || is_untracked_refinement
            }
            _ => false,
        };
    if casts {
        let cast = CastOperation::unchecked(refined, vec![declared])?;
        return Ok(context.bind(cast, Vec::new(), &[value])?.remove(0));
    }

    Ok(value)
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::hash::DefaultHasher;
    use std::sync::Mutex;

    use indoc::{formatdoc, indoc};
    use pretty_assertions::assert_eq;
    use ryft_macros::Operation;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrValue, ArrayOperation, ArrayReference, DataType, Device, Dimension,
        DimensionBounds, DimensionType, DimensionValue, DimensionVariable, Memory, MeshAxis, Shape,
    };
    use crate::batching::RecursiveBatchingDriver;
    use crate::captures::CaptureReference;
    use crate::contexts::EagerContext;
    use crate::differentiation::{
        CotangentDestinationKind, DotsSavable, EverythingSavable, Linearization, NothingSavable,
        RematerializationPolicyFn, SaveOnlyTheseNames, TransposedProgram,
    };
    use crate::operations::arithmetic::{MulOperation, NegOperation};
    use crate::operations::assertions::AssertOperation;
    use crate::operations::collectives::axis_index::AxisIndexOperation;
    use crate::operations::collectives::parallel_vary::ParallelVaryOperation;
    use crate::operations::comparisons::{CompareOperation, ComparisonDirection};
    use crate::operations::composites::operations::composite::{CompositeJvpRule, CompositeOperation};
    use crate::operations::composites::operations::composite_transpose::CompositeTransposeOperation;
    use crate::operations::composites::rules::CustomRuleTracer;
    use crate::operations::composites::rules::definitions::{CompositeKey, CustomRuleDefinition, NamedComposite};
    use crate::operations::composites::rules::registrations::CustomRuleRegistration;
    use crate::operations::constants::iota::IotaOperation;
    use crate::operations::constants::one_like::OneLikeOperation;
    use crate::operations::constants::zero::ZeroOperation;
    use crate::operations::constants::zero_like::ZeroLikeOperation;
    use crate::operations::control_flow::condition::ConditionOperation;
    use crate::operations::control_flow::r#while::WhileOperation;
    use crate::operations::debugging::PrintOperation;
    use crate::operations::differentiation::linear_call::LinearCallOperation;
    use crate::operations::differentiation::rematerialize::RematerializeOperation;
    use crate::operations::dimensions::dimension_from_scalar::DimensionFromScalarOperation;
    use crate::operations::dimensions::dimension_to_scalar::DimensionToScalarOperation;
    use crate::operations::dot::{DotDimensionNumbers, DotOperation};
    use crate::operations::manipulation::broadcasting::{BroadcastOperation, DynamicBroadcastOperation};
    use crate::operations::manipulation::conversions::ConvertElementTypeOperation;
    use crate::operations::manipulation::transposition::TransposeOperation;
    use crate::operations::references::{
        ReferenceAddUpdateOperation, ReferenceFreezeOperation, ReferenceReadOperation, ReferenceSwapOperation,
    };
    use crate::operations::sharding::place::{PLACE_OPERATION_NAME, Placed};
    use crate::operations::sharding::shard_map::{ShardMapBoundary, ShardMapOperation};
    use crate::operations::tagging::TagOperation;
    use crate::operations::trigonometric::{CosOperation, SinOperation};
    use crate::parameters::Placeholder;
    use crate::partial::{
        NoStorage, PartialEvaluation, PartialEvaluationInput, ResidualCandidate, ResidualDecision, ResidualPolicy,
        ResidualPolicyError, ResidualRejection,
    };
    use crate::programs::{
        AtomId, EffectClass, EffectClasses, ProgramBuilder, ReferenceSource, ReferenceType, RegionDriver, TypeIdentity,
    };

    use super::*;

    /// Value universe of the programs that these tests build.
    type TestValue = ArrayIrValue<Array>;

    /// Operation family of the programs that these tests build.
    type TestOperation = ArrayIrOperation<Array>;

    /// Flat program over [`TestValue`]s and [`TestOperation`]s.
    type TestProgram = Program<TestValue, TestOperation, Vec<TestValue>, Vec<TestValue>>;

    /// Constant universe of the programs whose callees reach values through [`CaptureReference`]s.
    type TestCapture = CaptureReference<ArrayIrType>;

    /// Builds a program over inputs of `input_types` whose outputs are the atoms that `build` adds to it.
    fn build_program(
        input_types: Vec<ArrayIrType>,
        build: impl FnOnce(&mut ProgramBuilder<TestValue, TestOperation>, &[AtomId]) -> Vec<AtomId>,
    ) -> TestProgram {
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let inputs = input_types.into_iter().map(|r#type| builder.add_input(r#type)).collect::<Vec<_>>();
        let outputs = build(&mut builder, &inputs);
        let output_count = outputs.len();
        builder.build(outputs, vec![Placeholder; inputs.len()], vec![Placeholder; output_count]).unwrap()
    }

    /// Returns the static `f32` vector type of `length` elements.
    fn f32_vector_type(length: usize) -> ArrayType {
        ArrayType::new_static(DataType::F32, [length])
    }

    /// Returns the static `f64` vector type of `length` elements.
    fn f64_vector_type(length: usize) -> ArrayType {
        ArrayType::new_static(DataType::F64, [length])
    }

    /// Returns the array of type `r#type` that holds `elements`.
    fn f32_value(r#type: ArrayType, elements: &[f32]) -> TestValue {
        ArrayIrValue::Array(Array::from_elements(r#type, elements).unwrap())
    }

    /// Returns the array of type `r#type` that holds `elements`.
    fn f64_value(r#type: ArrayType, elements: &[f64]) -> TestValue {
        ArrayIrValue::Array(Array::from_elements(r#type, elements).unwrap())
    }

    /// Adds a call of the shared `callee`, without a lifted capture prefix, on `inputs` to `builder` and returns its
    /// first output. Calls of one [`Arc`] share one interned callee region.
    fn add_call(
        builder: &mut ProgramBuilder<TestValue, TestOperation>,
        callee: &Arc<TestProgram>,
        inputs: Vec<AtomId>,
    ) -> AtomId {
        let callee_region = builder.intern_callee(callee, None).unwrap();
        builder
            .add_instruction(TestOperation::Call(CallOperation::new(0)), vec![callee_region], inputs, None)
            .unwrap()[0]
    }

    /// Returns the hash of `operation` under the [`DefaultHasher`].
    fn hash(operation: &CallOperation<ArrayIrType>) -> u64 {
        let mut hasher = DefaultHasher::new();
        operation.hash(&mut hasher);
        hasher.finish()
    }

    /// Returns the logical mesh of one manual axis `x` of 2 devices, together with the local type `f32[1]` of the shard
    /// of an `f32[2]` input split along `x`, which varies along `x`, of the `shard_map` that
    /// [`lockstep_call_program`] builds.
    fn lockstep_call_mesh_and_input_type() -> (LogicalMesh, ArrayIrType) {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let varying = Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["x"]).unwrap();
        (mesh, ArrayIrType::Array(f32_vector_type(1).with_sharding(varying).unwrap()))
    }

    /// Returns `x ↦ x * x` at type `r#type`.
    fn lockstep_call_square(r#type: ArrayIrType) -> TestProgram {
        build_program(vec![r#type], |builder, inputs| {
            let mul = ArrayOperation::from(MulOperation::new());
            builder.add_instruction(mul, Vec::new(), vec![inputs[0], inputs[0]], None).unwrap().to_vec()
        })
    }

    /// Builds the body of a `shard_map` over the manual axis `x` of `mesh` that applies `build` to the shard `a`, of
    /// type `a_local_type`, of its one `f32[2]` input split along `x`, and sums every output of `build` over `x`,
    /// together with the program that applies that `shard_map` to an `f32[2]` input.
    fn lockstep_shard_map_program(
        mesh: &LogicalMesh,
        a_local_type: &ArrayIrType,
        build: impl FnOnce(&mut ProgramBuilder<TestValue, TestOperation>, AtomId) -> Vec<AtomId>,
    ) -> (TestProgram, TestProgram) {
        let body = build_program(vec![a_local_type.clone()], |builder, inputs| {
            build(builder, inputs[0])
                .into_iter()
                .map(|output| {
                    let sum = ParallelReduceOperation::new(ReductionKind::Sum, "x".to_string()).with_mesh(mesh.clone());
                    builder
                        .add_instruction(ArrayOperation::ParallelReduce(sum), Vec::new(), vec![output], None)
                        .unwrap()[0]
                })
                .collect()
        });
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated = Sharding::replicated(mesh.clone(), 1);
        let output_count = body.output_types().len();
        let boundary =
            ShardMapBoundary::new(mesh.clone(), vec![sharded], vec![replicated; output_count], Vec::new()).unwrap();
        let input_types = vec![ArrayIrType::Array(f32_vector_type(2))];
        let operation = ShardMapOperation::from_program(&body, input_types.clone(), boundary).unwrap();
        let program = build_program(input_types, |builder, inputs| {
            let body = builder.import_program(body.clone());
            builder.add_instruction(operation, vec![body], inputs.to_vec(), None).unwrap().to_vec()
        });
        (body, program)
    }

    /// Builds the body of a `shard_map` over the manual axis `x` of `mesh` that calls `callee` on the shard `a`, of
    /// type `a_local_type`, of its one `f32[2]` input split along `x`, and sums every output of the call over `x`,
    /// together with the program that applies that `shard_map` to an `f32[2]` input (refer to
    /// [`lockstep_shard_map_program`]). When `passes_axis_index` is set, the first input of the call is the index of
    /// each device along `x`, and every other input of the call is `a`.
    fn lockstep_call_program(
        mesh: &LogicalMesh,
        a_local_type: &ArrayIrType,
        callee: TestProgram,
        passes_axis_index: bool,
    ) -> (TestProgram, TestProgram) {
        lockstep_shard_map_program(mesh, a_local_type, |builder, a| {
            let mut call_inputs = vec![a; callee.input_types().len()];
            if passes_axis_index {
                let index = AxisIndexOperation::new("x".to_string()).with_mesh(mesh.clone());
                call_inputs[0] =
                    builder.add_instruction(ArrayOperation::AxisIndex(index), Vec::new(), Vec::new(), None).unwrap()[0];
            }
            let callee = builder.import_program(callee);
            let call = TestOperation::Call(CallOperation::new(0));
            builder.add_instruction(call, vec![callee], call_inputs, None).unwrap().to_vec()
        })
    }

    /// Builds the `shard_map` programs of [`lockstep_call_program`] for a callee traced at `callee_input_types` whose
    /// instructions `build` adds over its inputs, followed by the programs of [`lockstep_shard_map_program`] whose body
    /// applies `build` to `a` directly, once per entry of `callee_input_types`. `build` receives its inputs together
    /// with their types, at which it traces the regions of the operations that it adds.
    fn lockstep_nested_programs(
        mesh: &LogicalMesh,
        a_local_type: &ArrayIrType,
        callee_input_types: Vec<ArrayIrType>,
        build: impl Fn(&mut ProgramBuilder<TestValue, TestOperation>, &[AtomId], &[ArrayIrType]) -> Vec<AtomId>,
    ) -> [(TestProgram, TestProgram); 2] {
        let input_count = callee_input_types.len();
        let callee =
            build_program(callee_input_types.clone(), |builder, inputs| build(builder, inputs, &callee_input_types));
        let called = lockstep_call_program(mesh, a_local_type, callee, false);
        let direct = lockstep_shard_map_program(mesh, a_local_type, |builder, a| {
            build(builder, &vec![a; input_count], &vec![a_local_type.clone(); input_count])
        });
        [called, direct]
    }

    /// Returns `x ↦ x + x` at type `r#type`, which is linear and its own transpose.
    fn lockstep_call_double(r#type: ArrayIrType) -> TestProgram {
        build_program(vec![r#type], |builder, inputs| {
            let add = ArrayOperation::from(AddOperation::new());
            builder.add_instruction(add, Vec::new(), vec![inputs[0], inputs[0]], None).unwrap().to_vec()
        })
    }

    /// Returns whether `r#type` varies along the manual axis `x`.
    fn lockstep_call_varies_along_x(r#type: &ArrayIrType) -> bool {
        matches!(
            r#type,
            ArrayIrType::Array(r#type) if r#type.sharding().is_some_and(|sharding| {
                sharding.varying_manual_axes().contains("x")
            }),
        )
    }

    /// Returns `(b, a) ↦ a * Σ b` over `b` of type `b_type` and `a` of type `a_type`, where `Σ` sums over the manual
    /// axis `x` of `mesh`. The sum does not vary along `x`, so it first varies along `x` by `parallel_vary` when `a`
    /// does. The function is linear in `a` and its own transpose, with `b` as its residual.
    fn lockstep_call_scale_by_sum(mesh: &LogicalMesh, b_type: ArrayIrType, a_type: ArrayIrType) -> TestProgram {
        let varies = lockstep_call_varies_along_x(&a_type);
        build_program(vec![b_type, a_type], |builder, inputs| {
            let sum = ParallelReduceOperation::new(ReductionKind::Sum, "x".to_string()).with_mesh(mesh.clone());
            let mut sum = builder
                .add_instruction(ArrayOperation::ParallelReduce(sum), Vec::new(), vec![inputs[0]], None)
                .unwrap()[0];
            if varies {
                let vary = ArrayOperation::ParallelVary(ParallelVaryOperation::new("x".to_string()));
                sum = builder.add_instruction(vary, Vec::new(), vec![sum], None).unwrap()[0];
            }
            let mul = ArrayOperation::from(MulOperation::new());
            builder.add_instruction(mul, Vec::new(), vec![inputs[1], sum], None).unwrap().to_vec()
        })
    }

    /// Builds a mixed-call fixture with one print of `effect_class` in each partition, i.e., `known = print(k * k)` and
    /// `residual = print(x * known)`. Its two scalar inputs are the known coefficient and the unknown value.
    fn effectful_mixed_callee(effect_class: EffectClass) -> Arc<TestProgram> {
        let scalar_type: ArrayIrType = ArrayType::scalar(DataType::F32).into();
        Arc::new(build_program(vec![scalar_type.clone(), scalar_type], |builder, inputs| {
            let (known, unknown) = (inputs[0], inputs[1]);
            let squared = builder
                .add_instruction(ArrayOperation::from(MulOperation::new()), Vec::new(), vec![known, known], None)
                .unwrap()[0];
            let print = PrintOperation::<ArrayType>::new("known").with_effect_class(effect_class);
            let coefficient =
                builder.add_instruction(ArrayOperation::from(print), Vec::new(), vec![squared], None).unwrap()[0];
            let product = builder
                .add_instruction(
                    ArrayOperation::from(MulOperation::new()),
                    Vec::new(),
                    vec![unknown, coefficient],
                    None,
                )
                .unwrap()[0];
            let print = PrintOperation::<ArrayType>::new("residual").with_effect_class(effect_class);
            builder
                .add_instruction(ArrayOperation::from(print), Vec::new(), vec![product], None)
                .unwrap()
                .to_vec()
        }))
    }

    /// Driver that exposes one source callee and returns a predetermined transpose for it.
    struct TestTranspositionDriver {
        /// Source callee exposed to the `call` transpose rule.
        source: TestProgram,

        /// Predetermined transposed callee returned by the recursive request.
        transposed: TestProgram,
    }

    impl RegionDriver<TestValue, TestOperation> for TestTranspositionDriver {
        fn regions<'r>(&'r self) -> impl Iterator<Item = RegionRef<'r, TestValue, TestOperation>>
        where
            TestValue: 'r,
            TestOperation: 'r,
        {
            std::iter::once(self.source.entry_region_ref())
        }
    }

    impl TranspositionDriver<TestValue, TestOperation> for TestTranspositionDriver {
        fn transpose_program(
            &self,
            region: RegionRef<'_, TestValue, TestOperation>,
            input_indices: &[usize],
            _: &[bool],
            destination_kinds: &[CotangentDestinationKind],
        ) -> Result<TransposedProgram<TestValue, TestOperation>, DifferentiationError> {
            // The predetermined program must have the boundary of the requested region's transposition, with one input
            // per non-reference output, per `Reference` destination, and per known input. The default zero-seed
            // request projects the region onto its unflagged outputs before calling this function, so this also pins
            // the structural zero seed mask that the rule requests.
            let input_types = region.input_types();
            let output_count = region.output_types().iter().filter(|r#type| !r#type.is_reference()).count();
            let reference_count =
                destination_kinds.iter().filter(|&&kind| kind == CotangentDestinationKind::Reference).count();
            assert_eq!(
                self.transposed.input_types().len(),
                output_count + reference_count + input_types.len() - input_indices.len(),
            );
            let cotangents = input_indices
                .iter()
                .enumerate()
                .map(|(index, &input_index)| {
                    match destination_kinds.get(index).copied().unwrap_or(CotangentDestinationKind::Return) {
                        CotangentDestinationKind::Return => TransposedProgramCotangent::Output,
                        CotangentDestinationKind::Reference if input_types[input_index].is_reference() => {
                            TransposedProgramCotangent::Output
                        }
                        CotangentDestinationKind::Reference | CotangentDestinationKind::Ignore => {
                            TransposedProgramCotangent::Absent
                        }
                    }
                })
                .collect();
            Ok(TransposedProgram::new(Arc::new(self.transposed.clone()), cotangents)?)
        }
    }

    /// Returns the `f32[2, 2]` matrix type of the residual demand workloads.
    fn residual_demand_matrix_type() -> ArrayIrType {
        ArrayIrType::from(ArrayType::new_static(DataType::F32, [2, 2]))
    }

    /// Stages `sin(a · b) ⊙ cos(a · b) + a` for `a = left` and `b = right` into `builder` and returns its output.
    fn add_residual_demand_leaf(
        builder: &mut ProgramBuilder<TestValue, TestOperation>,
        left: AtomId,
        right: AtomId,
    ) -> AtomId {
        let dot = DotOperation::new(DotDimensionNumbers::new(vec![1], vec![0], vec![], vec![]));
        let product =
            builder.add_instruction(ArrayOperation::from(dot), Vec::new(), vec![left, right], None).unwrap()[0];
        let sine = builder
            .add_instruction(ArrayOperation::from(SinOperation::new()), Vec::new(), vec![product], None)
            .unwrap()[0];
        let cosine = builder
            .add_instruction(ArrayOperation::from(CosOperation::new()), Vec::new(), vec![product], None)
            .unwrap()[0];
        let scaled = builder
            .add_instruction(ArrayOperation::from(MulOperation::new()), Vec::new(), vec![sine, cosine], None)
            .unwrap()[0];
        builder
            .add_instruction(ArrayOperation::from(AddOperation::new()), Vec::new(), vec![scaled, left], None)
            .unwrap()[0]
    }

    /// Returns `(a, b) ↦ sin(a · b) ⊙ cos(a · b) + a`, the shared callee of the residual demand workloads, whose matrix
    /// product `DotsSavable` saves while it recomputes the rest of the callee.
    fn residual_demand_leaf_callee() -> Arc<TestProgram> {
        Arc::new(build_program(vec![residual_demand_matrix_type(); 2], |builder, inputs| {
            vec![add_residual_demand_leaf(builder, inputs[0], inputs[1])]
        }))
    }

    /// Returns the nested diamond `g_k(a, b) = g_{k-1}(a, b) + g_{k-1}(sin a, b)` of `depth` levels over the shared
    /// residual demand callee `g_0`, where each level calls the previous level twice through one shared callee.
    fn residual_demand_diamond_callee(depth: usize) -> Arc<TestProgram> {
        let mut callee = residual_demand_leaf_callee();
        for _ in 0..depth {
            callee = Arc::new(build_program(vec![residual_demand_matrix_type(); 2], |builder, inputs| {
                let (left, right) = (inputs[0], inputs[1]);
                let first = add_call(builder, &callee, vec![left, right]);
                let shifted = builder
                    .add_instruction(ArrayOperation::from(SinOperation::new()), Vec::new(), vec![left], None)
                    .unwrap()[0];
                let second = add_call(builder, &callee, vec![shifted, right]);
                builder
                    .add_instruction(ArrayOperation::from(AddOperation::new()), Vec::new(), vec![first, second], None)
                    .unwrap()
                    .to_vec()
            }));
        }
        callee
    }

    /// Stages the nested diamond `g_k(a, b) = g_{k-1}(a, b) + g_{k-1}(sin a, b)` of `depth` levels over
    /// `g_0(a, b) = sin(a · b) ⊙ cos(a · b) + a` into `builder` with every level inlined, and returns its output. This
    /// is the reference without calls of [`residual_demand_diamond_callee`].
    fn add_inlined_residual_demand_diamond(
        builder: &mut ProgramBuilder<TestValue, TestOperation>,
        depth: usize,
        left: AtomId,
        right: AtomId,
    ) -> AtomId {
        if depth == 0 {
            return add_residual_demand_leaf(builder, left, right);
        }
        let first = add_inlined_residual_demand_diamond(builder, depth - 1, left, right);
        let shifted = builder
            .add_instruction(ArrayOperation::from(SinOperation::new()), Vec::new(), vec![left], None)
            .unwrap()[0];
        let second = add_inlined_residual_demand_diamond(builder, depth - 1, shifted, right);
        builder
            .add_instruction(ArrayOperation::from(AddOperation::new()), Vec::new(), vec![first, second], None)
            .unwrap()[0]
    }

    /// Returns `(x, w, b) ↦ x ⊙ y`, where `body` stages the calls that compute `y` from `w` and `b`, and, when `policy`
    /// is provided, wraps that function in a rematerialization under `policy`. Differentiating only `x` makes `y` the
    /// only value that the tangent needs, so every call stays all-known, while differentiating `w` as well makes every
    /// call a differentiated call whose tangent call mixes unknown tangents with known residuals.
    fn residual_demand_program(
        policy: Option<ResidualPolicyReference<ArrayIrType>>,
        body: impl Fn(&mut ProgramBuilder<TestValue, TestOperation>, AtomId, AtomId) -> AtomId,
    ) -> TestProgram {
        let program = build_program(vec![residual_demand_matrix_type(); 3], |builder, inputs| {
            let called = body(builder, inputs[1], inputs[2]);
            builder
                .add_instruction(ArrayOperation::from(MulOperation::new()), Vec::new(), vec![inputs[0], called], None)
                .unwrap()
                .to_vec()
        });
        let Some(policy) = policy else {
            return program;
        };
        build_program(vec![residual_demand_matrix_type(); 3], |builder, inputs| {
            let body = builder.import_program(program);
            let rematerialize = TestOperation::Rematerialize(RematerializeOperation::new(policy));
            builder.add_instruction(rematerialize, vec![body], inputs.to_vec(), None).unwrap().to_vec()
        })
    }

    /// Returns the residual count of the linearization of `program` with respect to the inputs that `input_indices`
    /// selects, followed by the numbers of instructions in the complete region closures of its primal and tangent
    /// programs, which count every shared callee once.
    fn residual_demand_statistics(program: &TestProgram, input_indices: &[usize]) -> [usize; 3] {
        let linearization = program.linearize_with_respect_to(input_indices).unwrap();
        let closure_size = |program: &TestProgram| program.entry_region_ref().instructions_in_closure().count();
        [linearization.residual_count(), closure_size(linearization.primal()), closure_size(linearization.tangent())]
    }

    /// Asserts that the linearizations of `actual` and `expected` with respect to the inputs that `input_indices`
    /// selects (at most two) compute the same primal output, tangent, and cotangents on fixed inputs under the
    /// reference interpretation. Without a `relative_tolerance`, the values must be exactly equal. With one, the values
    /// must have equal types and each element must lie within `relative_tolerance · max(|e|, 1)` of its expected
    /// element `e`, which programs that add the same contributions in another order need (e.g., a program whose
    /// shared callees accumulate cotangents in another order than its inlined counterpart).
    fn assert_residual_demand_linearizations_match(
        actual: &TestProgram,
        expected: &TestProgram,
        input_indices: &[usize],
        relative_tolerance: Option<f32>,
    ) {
        let matrix = |elements: &[f32]| f32_value(ArrayType::new_static(DataType::F32, [2, 2]), elements);
        let execute = |program: &TestProgram| {
            let linearization = program.linearize_with_respect_to(input_indices).unwrap();
            let inputs =
                vec![matrix(&[0.5, -1.0, 0.25, 2.0]), matrix(&[0.3, 0.1, -0.2, 0.4]), matrix(&[0.7, -0.6, 0.2, 0.9])];
            let mut primal = linearization.primal().interpret(inputs).unwrap();
            let value = primal.remove(0);
            let input_tangents = [matrix(&[1.0, 0.5, -0.5, 0.25]), matrix(&[-0.25, 0.75, 0.5, -1.0])];
            let tangent_inputs =
                input_tangents[..input_indices.len()].iter().cloned().chain(primal.clone()).collect::<Vec<_>>();
            let tangent = linearization.tangent().interpret(tangent_inputs).unwrap();
            let pullback_inputs = std::iter::once(matrix(&[-1.5, 0.75, 2.0, 1.0])).chain(primal).collect();
            let pullback = linearization.pullback().unwrap().into_program().interpret(pullback_inputs).unwrap();
            [vec![value], tangent, pullback].concat()
        };
        let (actual, expected) = (execute(actual), execute(expected));
        let Some(relative_tolerance) = relative_tolerance else {
            assert_eq!(actual, expected);
            return;
        };
        let types = |values: &[TestValue]| values.iter().map(|value| value.r#type().into_owned()).collect::<Vec<_>>();
        assert_eq!(types(&actual), types(&expected));
        let elements = |values: &[TestValue]| {
            values
                .iter()
                .flat_map(|value| match value {
                    ArrayIrValue::Array(array) => array.elements::<f32>().unwrap(),
                    _ => panic!("expected an array value"),
                })
                .collect::<Vec<_>>()
        };
        let (actual, expected) = (elements(&actual), elements(&expected));
        assert!(
            actual
                .iter()
                .zip(&expected)
                .all(|(actual, expected)| (actual - expected).abs() <= relative_tolerance * expected.abs().max(1.0)),
            "{actual:?} != {expected:?}",
        );
    }

    #[test]
    fn test_call_operation_with_boundary() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let device_mesh = DeviceMesh::new(mesh.clone(), vec![Device::new(0, 0), Device::new(1, 0)]).unwrap();
        let reversed_mesh = DeviceMesh::new(mesh.clone(), vec![Device::new(1, 0), Device::new(0, 0)]).unwrap();
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated = Sharding::replicated(mesh, 1);

        // A call of a callee that names no devices, including one whose lists hold no override, equals the plain call
        // and hashes only its capture count, its type marker, and its absent result bindings.
        let plain = CallOperation::<ArrayIrType>::new(1);
        // A field added to the operation fails to compile here until the equality and hashing assertions of this test
        // and of `test_call_operation_with_derived_input_shardings` cover it.
        let CallOperation {
            capture_count: _,
            marker: _,
            output_types: _,
            input_shardings: _,
            output_shardings: _,
            mesh: _,
            keeps_input_placements: _,
            derived_input_shardings: _,
            function_mesh: _,
        } = &plain;
        let mut expected = DefaultHasher::new();
        1usize.hash(&mut expected);
        PhantomData::<fn() -> ArrayIrType>.hash(&mut expected);
        None::<CallTypeBindings<ArrayIrType>>.hash(&mut expected);
        assert_eq!(hash(&plain), expected.finish());
        let empty = CallOperation::<ArrayIrType>::new(1).with_boundary(None, vec![None, None], vec![None]);
        assert_eq!(empty, plain);
        assert_eq!(empty.mesh(), None);
        assert_eq!(hash(&empty), hash(&plain));

        // A callee that names devices without overrides (e.g., because its body places a value) keeps its mesh, and a
        // call of a `jit` dispatcher records that it keeps the placements of its inputs. Both take part in equality and
        // hashing.
        let pinned =
            CallOperation::<ArrayIrType>::new(1).with_boundary(Some(&device_mesh), vec![None, None], vec![None]);
        assert_ne!(pinned, plain);
        assert_ne!(hash(&pinned), hash(&plain));
        assert_eq!(pinned.mesh(), Some(&device_mesh));
        assert_eq!(pinned.boundary_shardings().count(), 0);
        let dispatched = CallOperation::<ArrayIrType>::new(1).with_kept_input_placements();
        assert_ne!(dispatched, plain);
        assert_ne!(hash(&dispatched), hash(&plain));
        assert!(dispatched.keeps_input_placements());
        assert!(!plain.keeps_input_placements());

        // Overrides and the devices of their mesh take part in equality and hashing, and render only when present.
        let placed = CallOperation::<ArrayIrType>::new(1).with_boundary(
            Some(&device_mesh),
            vec![None, Some(sharded.clone())],
            vec![Some(replicated.clone())],
        );
        let moved = CallOperation::<ArrayIrType>::new(1).with_boundary(
            Some(&reversed_mesh),
            vec![None, Some(sharded.clone())],
            vec![Some(replicated.clone())],
        );
        assert_ne!(placed, plain);
        assert_ne!(hash(&placed), hash(&plain));
        assert_ne!(placed, moved);
        assert_ne!(hash(&placed), hash(&moved));
        assert_eq!(placed.mesh(), Some(&device_mesh));
        assert_eq!(placed.input_sharding(0), None);
        assert_eq!(placed.input_sharding(1), Some(&sharded));
        assert_eq!(placed.input_sharding(2), None);
        assert_eq!(placed.output_sharding(0), Some(&replicated));
        assert_eq!(placed.boundary_shardings().collect::<Vec<_>>(), vec![&sharded, &replicated]);
        let input_type: ArrayIrType = ArrayType::new_static(DataType::F32, [4]).into();
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        builder.add_input(input_type.clone());
        let input = builder.add_input(input_type.clone());
        let sine = builder
            .add_instruction(ArrayOperation::from(SinOperation::new()), Vec::new(), vec![input], None)
            .unwrap()[0];
        let callee = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![sine], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let render = |operation: CallOperation<ArrayIrType>| {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let callee_region = builder.import_region(callee.entry_region_ref());
            let inputs = vec![builder.add_input(input_type.clone()), builder.add_input(input_type.clone())];
            let output =
                builder.add_instruction(TestOperation::Call(operation), vec![callee_region], inputs, None).unwrap()[0];
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
                .to_string()
        };
        assert_eq!(
            render(plain),
            indoc! {"
                lambda %0:f32[4], %1:f32[4] .
                let %2:f32[4] = call [capture_count=1] %0 %1 [
                    callee={
                        lambda %0:f32[4], %1:f32[4] .
                        let %2:f32[4] = sin %1
                        in (%2)
                    },
                ]
                in (%2)"},
        );
        assert_eq!(
            render(placed),
            indoc! {"
                lambda %0:f32[4], %1:f32[4] .
                let %2:f32[4] = call [
                    capture_count=1,
                    input_shardings=[_, {mesh<['x'=2:auto]>, [{'x'}]}],
                    output_shardings=[{mesh<['x'=2:auto]>, [{}]}],
                    devices=[0, 1],
                ] %0 %1 [
                    callee={
                        lambda %0:f32[4], %1:f32[4] .
                        let %2:f32[4] = sin %1
                        in (%2)
                    },
                ]
                in (%2)"},
        );
        // Each device renders its process when it is not process zero, so two calls whose meshes differ only in the
        // process of a device render differently, as they compare unequal.
        let remote_mesh =
            DeviceMesh::new(device_mesh.logical_mesh().clone(), vec![Device::new(0, 0), Device::new(1, 1)]).unwrap();
        let remote =
            CallOperation::<ArrayIrType>::new(1).with_boundary(Some(&remote_mesh), vec![None, None], vec![None]);
        assert_ne!(remote, pinned);
        assert_eq!(
            render(remote),
            indoc! {"
                lambda %0:f32[4], %1:f32[4] .
                let %2:f32[4] = call [capture_count=1, devices=[0, 1@1]] %0 %1 [
                    callee={
                        lambda %0:f32[4], %1:f32[4] .
                        let %2:f32[4] = sin %1
                        in (%2)
                    },
                ]
                in (%2)"},
        );
        assert_eq!(
            render(pinned.with_kept_input_placements()),
            indoc! {"
                lambda %0:f32[4], %1:f32[4] .
                let %2:f32[4] = call [capture_count=1, devices=[0, 1], keeps_input_placements=true] %0 %1 [
                    callee={
                        lambda %0:f32[4], %1:f32[4] .
                        let %2:f32[4] = sin %1
                        in (%2)
                    },
                ]
                in (%2)"},
        );
    }

    #[test]
    fn test_call_operation_with_derived_input_shardings() {
        // A call that a transform derived keeps its boundary for the inputs that have one, and renders it.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let replicated = Sharding::replicated(mesh, 1);
        let plain = CallOperation::<ArrayIrType>::new(0);
        let derived = plain.clone().with_derived_input_shardings(vec![None, Some(replicated.clone())]);
        assert_eq!(plain.derived_input_shardings(), None);
        assert_eq!(derived.derived_input_shardings(), Some(&[None, Some(replicated)][..]));
        assert_ne!(derived, plain);
        assert_ne!(hash(&derived), hash(&plain));

        // The function mesh takes part in equality but is neither hashed nor rendered.
        assert_eq!(plain.function_mesh(), None);
        let device_mesh = DeviceMesh::new(LogicalMesh::single_device(), vec![Device::new(0, 0)]).unwrap();
        let meshed = plain.clone().with_function_mesh(&device_mesh);
        assert_eq!(meshed.function_mesh(), Some(&device_mesh));
        assert_ne!(meshed, plain);
        assert_eq!(hash(&meshed), hash(&plain));
        let render = |operation: &CallOperation<ArrayIrType>| {
            std::fmt::from_fn(|formatter| operation.render(formatter, 0)).to_string()
        };
        assert_eq!(render(&meshed), render(&plain));
    }

    #[test]
    fn test_call_operation_with_callee_type_signature() {
        let bounds = DimensionBounds::non_negative(Some(8)).unwrap();
        let input = DimensionType::new("input", bounds);
        let local = DimensionType::new("local", bounds);
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![input.dimension().clone()]));
        let output_type =
            ArrayType::new(DataType::F32, Shape::new(vec![input.dimension().clone(), local.dimension().clone()]));
        let output_types = vec![output_type.clone(), output_type];
        let operation = CallOperation::<ArrayType>::new(0)
            .with_callee_type_signature(std::slice::from_ref(&input_type), &output_types)
            .unwrap();
        let other = CallOperation::<ArrayType>::new(0)
            .with_callee_type_signature(std::slice::from_ref(&input_type), &output_types)
            .unwrap();
        let interface = RegionInterface::new(vec![input_type.clone()], output_types.clone(), EffectClasses::NONE);
        let outputs = operation
            .infer_output_types(std::slice::from_ref(&input_type), std::slice::from_ref(&interface))
            .unwrap();
        assert_eq!(outputs[0], outputs[1]);
        assert_ne!(outputs, output_types);
        assert_eq!(
            operation
                .infer_output_types(std::slice::from_ref(&input_type), std::slice::from_ref(&interface))
                .unwrap(),
            outputs
        );
        assert_ne!(
            other
                .infer_output_types(std::slice::from_ref(&input_type), std::slice::from_ref(&interface))
                .unwrap(),
            outputs
        );
        let actual = ArrayType::new_static(DataType::F32, [2]);
        let callee_outputs = output_types
            .iter()
            .map(|output| {
                ArrayType::new(DataType::F32, Shape::new(vec![2usize.into(), output.shape().dimensions()[1].clone()]))
            })
            .collect::<Vec<_>>();
        let actual_interface = RegionInterface::new(vec![actual.clone()], callee_outputs, EffectClasses::NONE);
        let instantiated = operation.infer_output_types(&[actual], &[actual_interface]).unwrap();
        assert_eq!(instantiated[0].shape().dimensions()[0], 2usize.into());
        assert_eq!(instantiated[0].shape().dimensions()[1], outputs[0].shape().dimensions()[1]);
        assert_eq!(instantiated[0], instantiated[1]);
        assert_eq!(operation.defined_type_identities().len(), 1);
        let retained = CallOperation::<ArrayType>::new(0)
            .with_caller_output_types(std::slice::from_ref(&input_type), &output_types, outputs.clone())
            .unwrap();
        assert_eq!(retained.infer_output_types(&[input_type], &[interface]).unwrap(), outputs);
    }

    #[test]
    fn test_call_operation_with_callee_region() {
        let input_type: ArrayIrType = ArrayType::scalar(DataType::F32).into();
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let input = builder.add_input(input_type.clone());
        let sine = builder
            .add_instruction(ArrayOperation::from(SinOperation::new()), Vec::new(), vec![input], None)
            .unwrap()[0];
        let callee = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![sine], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let operation = CallOperation::new(1).with_callee_region(callee.entry_region_ref()).unwrap();
        assert_eq!(
            operation,
            CallOperation::new(1)
                .with_callee_type_signature(&callee.input_types(), &callee.output_types())
                .unwrap(),
        );
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let callee_region = builder.import_region(callee.entry_region_ref());
        let input = builder.add_input(input_type);
        let output = builder
            .add_instruction(TestOperation::Call(operation), vec![callee_region], vec![input], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:f32[] = call [capture_count=1] %0 [
                    callee={
                        lambda %0:f32[] .
                        let %1:f32[] = sin %0
                        in (%1)
                    },
                ]
                in (%1)"},
        );
        assert_eq!(
            CallOperation::new(2).with_callee_region(callee.entry_region_ref()),
            Err(TypeError::invalid(format!(
                "`{CALL_OPERATION_NAME}` lifted capture prefix 2 exceeds its 1 callee inputs",
            ))),
        );

        // Generated callees can introduce local identities even when the source that was transformed had none.
        // The sealed signature records those definitions, so each new call still receives a fresh caller result.
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::I64).into());
        let local = DimensionType::new("local", DimensionBounds::non_negative(Some(8)).unwrap());
        let output = builder
            .add_instruction(DimensionFromScalarOperation::new(local), Vec::new(), vec![input], None)
            .unwrap()[0];
        let identity_callee = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let first = CallOperation::new(0).with_callee_region(identity_callee.entry_region_ref()).unwrap();
        let second = CallOperation::new(0).with_callee_region(identity_callee.entry_region_ref()).unwrap();
        let original = CallOperation::new(0)
            .with_callee_type_signature(&identity_callee.input_types(), &identity_callee.output_types())
            .unwrap();
        assert_eq!(first.defined_type_identities().len(), 1);
        assert_ne!(first.defined_type_identities(), second.defined_type_identities());
        assert_ne!(
            first.defined_type_identities().as_ref(),
            identity_callee.type_identity_signature().internal_identities()
        );
        let interface = identity_callee.entry_region_ref().interface();
        let first_outputs =
            first.infer_output_types(&identity_callee.input_types(), std::slice::from_ref(interface)).unwrap();
        let repeated_outputs =
            first.infer_output_types(&identity_callee.input_types(), std::slice::from_ref(interface)).unwrap();
        assert_eq!(first_outputs, repeated_outputs);
        assert_ne!(first_outputs, identity_callee.output_types().into_owned());
        assert_eq!(
            first.infer_output_types(&identity_callee.input_types(), std::slice::from_ref(interface)).unwrap()[0]
                .to_string(),
            original
                .infer_output_types(&identity_callee.input_types(), std::slice::from_ref(identity_callee.interface()))
                .unwrap()[0]
                .to_string(),
        );
        // Replacing a previous identity-bearing declaration with an identity-free callee clears that declaration,
        // exactly as the public signature constructor does.
        assert_eq!(
            first.with_callee_region(callee.entry_region_ref()).unwrap(),
            CallOperation::new(0)
                .with_callee_type_signature(&callee.input_types(), &callee.output_types())
                .unwrap(),
        );
    }

    #[test]
    fn test_call_operation_names_devices() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Auto).unwrap()]).unwrap();
        let device_mesh = DeviceMesh::new(mesh.clone(), vec![Device::new(0, 0)]).unwrap();
        let replicated = Sharding::replicated(mesh, 0);

        // Lists of overrides that are empty or hold none state no placement at the boundary, whether they belong to the
        // inputs or to the outputs, so such a call names its devices only when its callee does. One override at either
        // boundary states a placement, and the call names its devices whatever its callee requests.
        let call = |inputs: Vec<Option<Sharding>>, outputs: Vec<Option<Sharding>>| {
            CallOperation::<ArrayIrType>::new(0).with_boundary(Some(&device_mesh), inputs, outputs)
        };
        let rows = [
            ("empty lists", call(vec![], vec![])),
            ("no input override", call(vec![None, None], vec![])),
            ("no output override", call(vec![], vec![None])),
            ("one input override", call(vec![None, Some(replicated.clone()), None], vec![None])),
            ("one output override", call(vec![None], vec![Some(replicated.clone())])),
        ];
        let observed = rows
            .iter()
            .map(|(name, call)| {
                (*name, call.has_boundary_shardings(), call.names_devices(|| false), call.names_devices(|| true))
            })
            .collect::<Vec<_>>();
        assert_eq!(
            observed,
            vec![
                ("empty lists", false, None, Some(&device_mesh)),
                ("no input override", false, None, Some(&device_mesh)),
                ("no output override", false, None, Some(&device_mesh)),
                ("one input override", true, Some(&device_mesh), Some(&device_mesh)),
                ("one output override", true, Some(&device_mesh), Some(&device_mesh)),
            ],
        );

        // A call that records no mesh names no devices and never asks about its callee.
        assert_eq!(CallOperation::<ArrayIrType>::new(0).names_devices(|| unreachable!()), None);
    }

    #[test]
    fn test_call_operation_validate_input_types() {
        // A call accepts an input that refines the callee input type up to its placement, and rejects another shape.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let placed = f32_vector_type(4).with_sharding(Sharding::replicated(mesh, 1)).unwrap();
        let call = CallOperation::<ArrayType>::new(0);
        assert_eq!(call.validate_input_types(&[f32_vector_type(4)], &[placed]), Ok(None));
        assert_eq!(
            call.validate_input_types(&[f32_vector_type(4)], &[f32_vector_type(3)]),
            Err(TypeError::invalid(format!("`{CALL_OPERATION_NAME}` input #0 expected f32[4] but got f32[3]"))),
        );
        assert_eq!(
            call.validate_input_types(&[f32_vector_type(4)], &[]),
            Err(TypeError::invalid(format!("`{CALL_OPERATION_NAME}` expected 1 input(s) but got 0"))),
        );
    }

    #[test]
    fn test_call_operation_validate_input_types_rejects_redefined_state_axes() {
        let typed = |mesh: &LogicalMesh, sharding: fn(Sharding) -> Sharding| {
            f32_vector_type(2).with_sharding(sharding(Sharding::replicated(mesh.clone(), 1))).unwrap()
        };
        let unreduced_x = |sharding: Sharding| sharding.with_unreduced_axes(["x"]).unwrap();
        let varying_x = |sharding: Sharding| sharding.with_varying_manual_axes(["x"]).unwrap();
        let varying_x_y = |sharding: Sharding| sharding.with_varying_manual_axes(["x", "y"]).unwrap();
        let call = CallOperation::<ArrayType>::new(0);

        // A callee input type with a sharding must define every mesh axis that carries the unreduced, reduced, or
        // varying state of the input as the mesh of the input does, because an axis of the same name with another
        // size or type is another axis.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let larger_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap();
        let explicit_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        for (declared, actual) in [
            (typed(&larger_mesh, unreduced_x), typed(&mesh, unreduced_x)),
            (typed(&explicit_mesh, unreduced_x), typed(&mesh, unreduced_x)),
            (typed(&larger_mesh, varying_x), typed(&mesh, varying_x)),
        ] {
            assert_eq!(
                call.validate_input_types(std::slice::from_ref(&declared), std::slice::from_ref(&actual)),
                Err(TypeError::invalid(format!(
                    "`{CALL_OPERATION_NAME}` input #0 expected {declared} but got {actual}",
                ))),
            );
        }

        // Axes that carry no state of the input may differ.
        let wide_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let other_wide_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 4, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        assert_eq!(
            call.validate_input_types(&[typed(&other_wide_mesh, varying_x)], &[typed(&wide_mesh, varying_x)]),
            Ok(None),
        );

        // Manual mesh axes along which the input varies beyond the callee input type carry state into the call too, so
        // the mesh of the callee input type must define each of them as well, with the same size and type.
        let manual_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let other_manual_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 4, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        assert_eq!(
            call.validate_input_types(&[typed(&manual_mesh, varying_x)], &[typed(&manual_mesh, varying_x_y)]),
            Ok(Some((manual_mesh.clone(), BTreeSet::from(["y".to_string()])))),
        );
        for declared in [typed(&other_manual_mesh, varying_x), typed(&mesh, varying_x)] {
            let actual = typed(&manual_mesh, varying_x_y);
            assert_eq!(
                call.validate_input_types(std::slice::from_ref(&declared), std::slice::from_ref(&actual)),
                Err(TypeError::invalid(format!(
                    "`{CALL_OPERATION_NAME}` input #0 expected {declared} but got {actual}",
                ))),
            );
        }
    }

    #[test]
    fn test_call_operation_declared_input_sharding() {
        // An executable places an argument of a declared type without a sharding replicated over its own mesh, and an
        // argument of a declared type with a sharding at that sharding, whatever its own mesh is.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let other_mesh = LogicalMesh::new(vec![MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let split = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        assert_eq!(
            CallOperation::declared_input_sharding(&f32_vector_type(4), &mesh),
            Sharding::replicated(mesh.clone(), 1),
        );
        assert_eq!(
            CallOperation::declared_input_sharding(
                &f32_vector_type(4).with_sharding(split.clone()).unwrap(),
                &other_mesh,
            ),
            split,
        );
    }

    #[test]
    fn test_call_operation_boundary_input_type() {
        // A declared type without a sharding takes the placement of its boundary without the auto mesh axes, which
        // types never record.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("a", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap();
        let split = |axes: &[&str]| Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(axes.iter().copied())]);
        let item_type = f32_vector_type(4);
        let split_type = item_type.clone().with_sharding(split(&["x"]).unwrap()).unwrap();
        let replicated_type = item_type.clone().with_sharding(Sharding::replicated(mesh.clone(), 1)).unwrap();
        assert_eq!(CallOperation::boundary_input_type(&item_type, &split(&["x"]).unwrap()), Some(split_type.clone()));
        assert_eq!(
            CallOperation::boundary_input_type(&item_type, &split(&["x", "a"]).unwrap()),
            Some(split_type.clone()),
        );
        assert_eq!(CallOperation::boundary_input_type(&item_type, &split(&["a"]).unwrap()), Some(replicated_type));

        // A declared type that records a sharding keeps it, and a boundary with a pending reduction states more than
        // a type without a sharding can.
        assert_eq!(CallOperation::boundary_input_type(&split_type, &split(&["x"]).unwrap()), None);
        let unreduced = Sharding::replicated(mesh.clone(), 1).with_unreduced_axes(["x"]).unwrap();
        assert_eq!(CallOperation::boundary_input_type(&item_type, &unreduced), None);
    }

    #[test]
    fn test_call_operation_infer_output_types() {
        let dimension_type =
            ArrayIrType::Dimension(DimensionType::new("size", DimensionBounds::positive(Some(9)).unwrap()));
        let array_type = ArrayIrType::Array(f64_vector_type(4));
        let interface = RegionInterface::new(
            vec![dimension_type.clone()],
            vec![array_type.clone(), dimension_type.clone()],
            EffectClasses::NONE,
        );
        let operation = CallOperation::new(0);

        assert_eq!(operation.name(), CALL_OPERATION_NAME);
        assert_eq!(
            operation.infer_output_types(&[], &[]),
            Err(TypeError::invalid(format!("`{CALL_OPERATION_NAME}` expects 1 attached callee region but got 0"))),
        );
        assert_eq!(
            operation
                .infer_output_types(std::slice::from_ref(&dimension_type), std::slice::from_ref(&interface))
                .unwrap(),
            vec![array_type.clone(), dimension_type.clone()],
        );

        // The call inputs must match the callee boundary, which must cover the lifted capture prefix.
        assert_eq!(
            CallOperation::new(2)
                .infer_output_types(std::slice::from_ref(&dimension_type), std::slice::from_ref(&interface)),
            Err(TypeError::invalid(format!(
                "`{CALL_OPERATION_NAME}` lifted capture prefix 2 exceeds its 1 callee inputs",
            ))),
        );
        assert_eq!(
            operation.infer_output_types(&[], std::slice::from_ref(&interface)),
            Err(TypeError::invalid(format!("`{CALL_OPERATION_NAME}` expected 1 input(s) but got 0"))),
        );
        assert_eq!(
            operation.infer_output_types(std::slice::from_ref(&array_type), std::slice::from_ref(&interface)),
            Err(TypeError::invalid(format!(
                "`{CALL_OPERATION_NAME}` input #0 expected {dimension_type} but got {array_type}",
            ))),
        );

        // A callee that defines its own result identities needs caller-owned result names.
        let local_interface = RegionInterface::new(Vec::new(), vec![dimension_type], EffectClasses::NONE);
        assert_eq!(
            operation.infer_output_types(&[], &[local_interface]),
            Err(TypeError::invalid(format!(
                "`{CALL_OPERATION_NAME}` with callee-local result identities requires `with_callee_type_signature`",
            ))),
        );
    }

    #[test]
    fn test_call_operation_input_region_provenance() {
        // A call forwards its inputs to the callee positionally, so input provenance preserves indices for the single
        // callee region and is absent for any other region.
        let operation = CallOperation::<ArrayIrType>::new(2);
        assert_eq!(operation.input_region_provenance(0, 3), InputRegionProvenance::Input { index: 3 });
        assert_eq!(operation.input_region_provenance(1, 3), InputRegionProvenance::None);
    }

    #[test]
    fn test_call_operation_output_region_provenance() {
        // A call returns the outputs of its callee positionally, so each output originates from the callee output with
        // the same index.
        let operation = CallOperation::<ArrayIrType>::new(2);
        assert_eq!(
            operation.output_region_provenance(2),
            vec![OutputRegionProvenance { region_index: 0, output_index: 2 }],
        );
    }

    #[test]
    fn test_call_operation_output_region_provenance_classifies_residual_candidates() {
        // `call` reports positional output-region provenance, so a residual produced by a computed callee output
        // is classified through the callee to that output's own producer instead of to the opaque `call`. The
        // program squares its known input `x` inside the callee and multiplies the result of the call by its unknown
        // input `y` outside of it, so the residual program needs the result of the call.
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F64));
        let callee = build_program(vec![scalar_type.clone()], |builder, inputs| {
            builder
                .add_instruction(
                    ArrayOperation::from(MulOperation::new()),
                    Vec::new(),
                    vec![inputs[0], inputs[0]],
                    None,
                )
                .unwrap()
                .to_vec()
        });
        let program = build_program(vec![scalar_type; 2], |builder, inputs| {
            let callee = builder.import_program(callee);
            let call = TestOperation::Call(CallOperation::new(0));
            let called = builder.add_instruction(call, vec![callee], vec![inputs[0]], None).unwrap()[0];
            builder
                .add_instruction(ArrayOperation::from(MulOperation::new()), Vec::new(), vec![called, inputs[1]], None)
                .unwrap()
                .to_vec()
        });
        let names = Arc::new(Mutex::new(Vec::new()));
        let recorded = names.clone();
        let policy = RematerializationPolicyFn::new::<ArrayIrType>(move |candidate| {
            let names = candidate.producers().iter().map(|producer| producer.name().to_string());
            recorded.lock().unwrap().extend(names);
            Ok::<_, ResidualRejection>(ResidualDecision::<NoStorage>::Save)
        });
        program
            .partition(&[true, false])
            .unwrap()
            .with_residual_policy(&ResidualPolicyReference::new(policy))
            .unwrap();
        assert_eq!(names.lock().unwrap().clone(), vec!["mul".to_string()]);
    }

    #[test]
    fn test_call_operation_region_data_flow() {
        let operation = CallOperation::<ArrayIrType>::new(2);
        assert!(matches!(operation.region_data_flow(), RegionDataFlow::Provenance));
        let dispatched = TestOperation::Call(operation);
        assert!(matches!(dispatched.region_data_flow(), RegionDataFlow::Provenance));
    }

    #[test]
    fn test_call_operation_prune_boundary() {
        // The callee maps its lifted capture `c` and its inputs `(x, y)` to `(cos(x), c * y)`. Using only the first
        // output drops the input `y` and the second output, while the capture prefix is kept although nothing uses it.
        let r#type = ArrayIrType::Array(f64_vector_type(4));
        let callee = build_program(vec![r#type.clone(); 3], |builder, inputs| {
            let cosine = builder
                .add_instruction(ArrayOperation::from(CosOperation::new()), Vec::new(), vec![inputs[1]], None)
                .unwrap()[0];
            let product = builder
                .add_instruction(
                    ArrayOperation::from(MulOperation::new()),
                    Vec::new(),
                    vec![inputs[0], inputs[2]],
                    None,
                )
                .unwrap()[0];
            vec![cosine, product]
        });
        let program = build_program(vec![r#type; 3], |builder, inputs| {
            let callee = builder.import_region(callee.entry_region_ref());
            let call = TestOperation::Call(CallOperation::new(1));
            vec![builder.add_instruction(call, vec![callee], inputs.to_vec(), None).unwrap()[0]]
        });
        assert_eq!(
            program.into_pruned().unwrap().to_string(),
            indoc! {"
                lambda %0:f64[4], %1:f64[4], %2:f64[4] .
                let %3:f64[4] = call [capture_count=1] %0 %1 [
                    callee={
                        lambda %0:f64[4], %1:f64[4] .
                        let %2:f64[4] = cos %1
                        in (%2)
                    },
                ]
                in (%3)"},
        );
    }

    #[test]
    fn test_call_operation_prune_boundary_keeps_boundary_shardings_of_kept_positions() {
        // The callee maps `(x, y, z)` to `(cos(x), x * z, sin(z))` and never uses `y`, and its caller drops the second
        // output. Pruning drops the overrides at those positions, which differ from the boundary at the positions that
        // follow them (i.e., no override for `z` and a replicated one for `sin(z)`), so a misindexed override would
        // render.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let device_mesh = DeviceMesh::new(mesh.clone(), vec![Device::new(0, 0), Device::new(1, 0)]).unwrap();
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated = Sharding::replicated(mesh, 1);
        let r#type = ArrayIrType::Array(f64_vector_type(4));
        let callee = build_program(vec![r#type.clone(); 3], |builder, inputs| {
            let cosine = builder
                .add_instruction(ArrayOperation::from(CosOperation::new()), Vec::new(), vec![inputs[0]], None)
                .unwrap()[0];
            let product = builder
                .add_instruction(
                    ArrayOperation::from(MulOperation::new()),
                    Vec::new(),
                    vec![inputs[0], inputs[2]],
                    None,
                )
                .unwrap()[0];
            let sine = builder
                .add_instruction(ArrayOperation::from(SinOperation::new()), Vec::new(), vec![inputs[2]], None)
                .unwrap()[0];
            vec![cosine, product, sine]
        });
        let program = build_program(vec![r#type; 3], |builder, inputs| {
            let callee = builder.import_region(callee.entry_region_ref());
            let call = TestOperation::Call(CallOperation::new(0).with_boundary(
                Some(&device_mesh),
                vec![Some(replicated.clone()), Some(sharded.clone()), None],
                vec![None, Some(sharded.clone()), Some(replicated.clone())],
            ));
            let outputs = builder.add_instruction(call, vec![callee], inputs.to_vec(), None).unwrap();
            vec![outputs[0], outputs[2]]
        });
        assert_eq!(
            program.into_pruned().unwrap().to_string(),
            indoc! {"
                lambda %0:f64[4], %1:f64[4], %2:f64[4] .
                let %3:f64[4], %4:f64[4] = call [
                    input_shardings=[{mesh<['x'=2:auto]>, [{}]}, _],
                    output_shardings=[_, {mesh<['x'=2:auto]>, [{}]}],
                    devices=[0, 1],
                ] %0 %2 [
                    callee={
                        lambda %0:f64[4], %1:f64[4] .
                        let %2:f64[4] = cos %0
                            %3:f64[4] = sin %1
                        in (%2, %3)
                    },
                ]
                in (%3, %4)"},
        );
    }

    #[test]
    fn test_call_operation_region_capture_input_count() {
        // The lifted capture prefix of a call is the capture prefix of its single callee region, and no other region
        // has one.
        let operation = CallOperation::<ArrayIrType>::new(2);
        assert_eq!(operation.region_capture_input_count(0), Some(2));
        assert_eq!(operation.region_capture_input_count(1), None);
    }

    #[test]
    fn test_call_operation_discharge_references() {
        // `call` forwards its inputs onto its callee one for one and reports the callee's outputs as its own, so
        // the shared positional rewrite threads state through it. The callee receives the entering state in place of
        // its reference input and publishes the final state after its declared outputs, while the call site keeps its
        // own input and output contract. The callee mutates the root and returns only the previous snapshot, so the
        // final state is exactly what the call site would otherwise have lost.
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));
        let callee = build_program(vec![reference_type, scalar_type.clone()], |builder, inputs| {
            builder
                .add_instruction(ReferenceSwapOperation::new(), Vec::new(), inputs.to_vec(), None)
                .unwrap()
                .to_vec()
        });
        let source = build_program(vec![scalar_type; 2], |builder, inputs| {
            let callee = builder.import_program(callee);
            let root =
                builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![inputs[0]], None).unwrap()[0];
            let call = TestOperation::Call(CallOperation::new(0));
            let previous = builder.add_instruction(call, vec![callee], vec![root, inputs[1]], None).unwrap()[0];
            let frozen =
                builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![root], None).unwrap()[0];
            vec![previous, frozen]
        });

        // The callee's local root is allocated, mutated, and consumed inside the call, so widening the boundary
        // leaves no external state behind and the rewritten call is the entry region's only instruction.
        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(discharged.external_reference_bindings(), &[]);
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[] .
                let %2:f32[], %3:f32[] = call %0 %1 [
                    callee={
                        lambda %0:f32[], %1:f32[] .
                        in (%0, %1)
                    },
                ]
                in (%2, %3)"},
        );
    }

    #[test]
    fn test_call_operation_discharge_references_threads_nested_callee_state_into_the_caller() {
        // Two nested call levels each swap a distinct reference. Discharge must lift both states to the caller's
        // boundary in public-input order, mark both as mutated with their own final-state output slot, and leave no
        // reference-typed atom anywhere in the resulting program.
        let reference_type = ArrayIrType::Reference(ReferenceType::new(f64_vector_type(4)));
        let array_type = ArrayIrType::Array(f64_vector_type(4));
        let inner_callee = build_program(vec![reference_type.clone(), array_type.clone()], |builder, inputs| {
            builder
                .add_instruction(ReferenceSwapOperation::new(), Vec::new(), inputs.to_vec(), None)
                .unwrap()
                .to_vec()
        });
        let boundary_types = vec![reference_type, array_type];
        let boundary_types = [boundary_types.clone(), boundary_types].concat();
        let callee = build_program(boundary_types.clone(), |builder, inputs| {
            let inner_callee = builder.import_region(inner_callee.entry_region_ref());
            let call = TestOperation::Call(CallOperation::new(0));
            let first_snapshot =
                builder.add_instruction(call.clone(), vec![inner_callee], inputs[..2].to_vec(), None).unwrap()[0];
            let second_snapshot =
                builder.add_instruction(call, vec![inner_callee], inputs[2..].to_vec(), None).unwrap()[0];
            vec![first_snapshot, second_snapshot]
        });
        let program = build_program(boundary_types, |builder, inputs| {
            let callee = builder.import_region(callee.entry_region_ref());
            let call = TestOperation::Call(CallOperation::new(0));
            builder.add_instruction(call, vec![callee], inputs.to_vec(), None).unwrap().to_vec()
        });

        let discharged = program.discharge_references(0).unwrap();
        assert_eq!(discharged.output_count(), 2);
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f64[4], %1:f64[4], %2:f64[4], %3:f64[4] .
                let %4:f64[4], %5:f64[4], %6:f64[4], %7:f64[4] = call %0 %1 %2 %3 [
                    callee={
                        lambda %0:f64[4], %1:f64[4], %2:f64[4], %3:f64[4] .
                        let %4:f64[4], %5:f64[4] = call %0 %1 [
                            callee={
                                lambda %0:f64[4], %1:f64[4] .
                                in (%0, %1)
                            },
                        ]
                            %6:f64[4], %7:f64[4] = call %2 %3 [
                                callee={
                                    lambda %0:f64[4], %1:f64[4] .
                                    in (%0, %1)
                                },
                            ]
                        in (%4, %6, %5, %7)
                    },
                ]
                in (%4, %5, %6, %7)"},
        );
        assert_eq!(discharged.external_reference_bindings().len(), 2);
        assert_eq!(discharged.external_reference_bindings()[0].source(), ReferenceSource::Input { index: 0 });
        assert!(discharged.external_reference_bindings()[0].is_mutated());
        assert_eq!(discharged.external_reference_bindings()[0].output_index(), Some(2));
        assert_eq!(discharged.external_reference_bindings()[1].source(), ReferenceSource::Input { index: 2 });
        assert!(discharged.external_reference_bindings()[1].is_mutated());
        assert_eq!(discharged.external_reference_bindings()[1].output_index(), Some(3));
    }

    #[test]
    fn test_call_operation_discharge_references_resolves_nested_callee_lexical_capture_scope() {
        // A branch region inside the callee reaches its reference through a capture constant, which must resolve
        // against the callee's own capture scope rather than the entry program's lifted prefix. The accessed root is
        // the caller's second public input, and only that mutated state gains a final-state output.
        let reference_type = ArrayIrType::Reference(ReferenceType::new(f64_vector_type(4)));
        let array_type = ArrayIrType::Array(f64_vector_type(4));
        let predicate_type = ArrayIrType::Array(ArrayType::scalar(DataType::Boolean));
        let branch = {
            let mut builder = ProgramBuilder::<TestCapture, TestOperation>::new();
            let replacement = builder.add_input(array_type.clone());
            let reference = builder.add_constant(CaptureReference::new(0, reference_type.clone()));
            let snapshot = builder
                .add_instruction(ReferenceSwapOperation::new(), Vec::new(), vec![reference, replacement], None)
                .unwrap()[0];
            builder
                .build::<Vec<TestCapture>, Vec<TestCapture>>(vec![snapshot], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let callee = {
            let mut builder = ProgramBuilder::<TestCapture, TestOperation>::new();
            let branch = builder.import_region(branch.entry_region_ref());
            builder.add_input(reference_type.clone());
            let replacement = builder.add_input(array_type.clone());
            let predicate = builder.add_input(predicate_type.clone());
            let snapshot = builder
                .add_instruction(ConditionOperation::new(), vec![branch, branch], vec![predicate, replacement], None)
                .unwrap()[0];
            builder
                .build::<Vec<TestCapture>, Vec<TestCapture>>(vec![snapshot], vec![Placeholder; 3], vec![Placeholder])
                .unwrap()
        };

        let mut builder = ProgramBuilder::<TestCapture, TestOperation>::new();
        let callee = builder.import_region(callee.entry_region_ref());
        builder.add_input(reference_type.clone());
        let reference = builder.add_input(reference_type);
        let replacement = builder.add_input(array_type);
        let predicate = builder.add_input(predicate_type);
        let snapshot = builder
            .add_instruction(
                TestOperation::Call(CallOperation::new(1)),
                vec![callee],
                vec![reference, replacement, predicate],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestCapture>, Vec<TestCapture>>(vec![snapshot], vec![Placeholder; 4], vec![Placeholder])
            .unwrap();

        let discharged = program.discharge_references_in_capture_lifted_program(0).unwrap();
        assert_eq!(discharged.output_count(), 1);
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f64[4], %1:f64[4], %2:f64[4], %3:bool[] .
                let %4:f64[4], %5:f64[4] = call [capture_count=1] %1 %2 %3 [
                    callee={
                        lambda %0:f64[4], %1:f64[4], %2:bool[] .
                        let %3:f64[4], %4:f64[4] = condition %2 %1 %0 [
                            true={
                                lambda %0:f64[4], %1:f64[4] .
                                in (%1, %0)
                            },
                            false={
                                lambda %0:f64[4], %1:f64[4] .
                                in (%1, %0)
                            },
                        ]
                        in (%3, %4)
                    },
                ]
                in (%4, %5)"},
        );
        assert_eq!(discharged.external_reference_bindings().len(), 2);
        assert_eq!(discharged.external_reference_bindings()[0].source(), ReferenceSource::Input { index: 0 });
        assert!(!discharged.external_reference_bindings()[0].is_mutated());
        assert_eq!(discharged.external_reference_bindings()[0].output_index(), None);
        assert_eq!(discharged.external_reference_bindings()[1].source(), ReferenceSource::Input { index: 1 });
        assert!(discharged.external_reference_bindings()[1].is_mutated());
        assert_eq!(discharged.external_reference_bindings()[1].output_index(), Some(1));
    }

    #[test]
    fn test_call_operation_discharge_references_preserves_forwarded_root_identity() {
        let reference_type = ArrayIrType::Reference(ReferenceType::new(f64_vector_type(4)));
        let callee = build_program(vec![reference_type.clone()], |_, inputs| inputs.to_vec());
        let program = build_program(vec![reference_type], |builder, inputs| {
            let callee = builder.import_region(callee.entry_region_ref());
            let call = TestOperation::Call(CallOperation::new(0));
            let forwarded = builder.add_instruction(call, vec![callee], inputs.to_vec(), None).unwrap()[0];
            builder
                .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![forwarded], None)
                .unwrap()
                .to_vec()
        });

        // The callee only forwards its reference input, so discharge must keep the forwarded value bound to the same
        // root as the caller's public reference input instead of synthesizing a second state slot for it. Reading
        // through the forwarded result is not a mutation, so no final-state output is appended.
        let discharged = program.discharge_references(0).unwrap();
        assert_eq!(discharged.output_count(), 1);
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f64[4] .
                let %1:f64[4] = call %0 [
                    callee={
                        lambda %0:f64[4] .
                        in (%0)
                    },
                ]
                in (%1)"},
        );
        assert_eq!(discharged.external_reference_bindings().len(), 1);
        assert_eq!(discharged.external_reference_bindings()[0].source(), ReferenceSource::Input { index: 0 });
        assert!(!discharged.external_reference_bindings()[0].is_mutated());
        assert_eq!(discharged.external_reference_bindings()[0].output_index(), None);
    }

    #[test]
    fn test_call_operation_discharge_references_instantiates_callee_local_results() {
        // The callee mutates a reference and returns a dimension that it defines itself, so the call instantiates a
        // fresh result identity. Discharge widens the callee with the reference state, which must not leave the
        // call's result metadata pinned to the narrower pre-discharge callee signature.
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Auto).unwrap()]).unwrap();
        let value_type = ArrayType::scalar(DataType::F32).with_sharding(Sharding::replicated(logical_mesh.clone(), 0));
        let value_type = value_type.unwrap();
        let extent_type = ArrayType::scalar(DataType::I64).with_sharding(Sharding::replicated(logical_mesh, 0));
        let extent_type = extent_type.unwrap();
        let rows = DimensionType::from(DimensionVariable::new("rows", DimensionBounds::non_negative(Some(8)).unwrap()));

        let mut callee_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = callee_builder.add_input(ArrayIrType::Reference(ReferenceType::new(value_type.clone())));
        let update = callee_builder.add_input(value_type.clone().into());
        let extent = callee_builder.add_input(extent_type.clone().into());
        let previous = callee_builder
            .add_instruction(
                TestOperation::ReferenceSwap(ReferenceSwapOperation::new()),
                Vec::new(),
                vec![reference, update],
                None,
            )
            .unwrap()[0];
        let dimension = callee_builder
            .add_instruction(DimensionFromScalarOperation::new(rows.clone()), Vec::new(), vec![extent], None)
            .unwrap()[0];
        let callee = callee_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![previous, dimension],
                vec![Placeholder; 3],
                vec![Placeholder; 2],
            )
            .unwrap();
        let operation = CallOperation::new(0)
            .with_callee_type_signature(&callee.input_types(), &callee.output_types())
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let callee = builder.import_program(callee);
        let initial = builder.add_input(value_type.clone().into());
        let update = builder.add_input(value_type.clone().into());
        let extent = builder.add_input(extent_type.clone().into());
        let root = builder
            .add_instruction(TestOperation::ReferenceNew(ReferenceNewOperation::new()), Vec::new(), vec![initial], None)
            .unwrap()[0];
        let outputs = builder
            .add_instruction(TestOperation::Call(operation), vec![callee], vec![root, update, extent], None)
            .unwrap()
            .to_vec();
        let frozen = builder
            .add_instruction(
                TestOperation::ReferenceFreeze(ReferenceFreezeOperation::new()),
                Vec::new(),
                vec![root],
                None,
            )
            .unwrap()[0];
        let extent = builder
            .add_instruction(
                TestOperation::DimensionToScalar(DimensionToScalarOperation),
                Vec::new(),
                vec![outputs[1]],
                None,
            )
            .unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![outputs[0], extent, frozen],
                vec![Placeholder; 3],
                vec![Placeholder; 3],
            )
            .unwrap();

        let program = source.discharge_references(0).unwrap().into_program_without_external_references().unwrap();
        assert_eq!(
            program.to_string(),
            formatdoc! {"
                lambda %0:{value_type}, %1:{value_type}, %2:{extent_type} .
                let %3:{value_type}, %4:dimension<rows ∈ [0, 8)>, %5:{value_type} = call [
                    output_types=[{value_type}, dimension<rows ∈ [0, 8)>, {value_type}],
                ] %0 %1 %2 [
                    callee={{
                        lambda %0:{value_type}, %1:{value_type}, %2:{extent_type} .
                        let %3:dimension<rows ∈ [0, 8)> = dimension_from_scalar [bounds=[0, 8)] %2
                        in (%0, %3, %1)
                    }},
                ]
                    %6:i64[] = dimension_to_scalar %4
                in (%3, %6, %5)"},
        );
        // The call owns a fresh result identity, distinct from the callee-local identity it instantiates.
        let call = &program.entry_region_ref().instructions()[0];
        assert_ne!(program.atoms()[call.outputs()[1].index()].r#type().as_ref(), &ArrayIrType::Dimension(rows));

        // The discharged program returns the previous reference value, the extent, and the final reference value.
        assert_eq!(
            program.interpret(vec![
                f32_value(value_type.clone(), &[3.0]),
                f32_value(value_type.clone(), &[4.0]),
                ArrayIrValue::Array(Array::from_elements(extent_type, &[5i64]).unwrap()),
            ]),
            Ok(vec![
                f32_value(value_type.clone(), &[3.0]),
                ArrayIrValue::Array(Array::from_elements(ArrayType::scalar(DataType::I64), &[5i64]).unwrap()),
                f32_value(value_type, &[4.0]),
            ]),
        );
    }

    #[test]
    fn test_call_operation_interpret() {
        // An eager call runs its callee on its inputs and returns the outputs of the callee.
        let vector_type = ArrayIrType::Array(f32_vector_type(2));
        let callee = build_program(vec![vector_type.clone()], |builder, inputs| {
            builder
                .add_instruction(
                    ArrayOperation::from(MulOperation::new()),
                    Vec::new(),
                    vec![inputs[0], inputs[0]],
                    None,
                )
                .unwrap()
                .to_vec()
        });
        let context = EagerContext::<TestValue, TestOperation>::new();
        let call = TestOperation::Call(CallOperation::new(0));
        assert_eq!(
            context.bind(call.clone(), vec![callee.clone()], &[f32_value(f32_vector_type(2), &[2.0, 3.0])]),
            Ok(vec![f32_value(f32_vector_type(2), &[4.0, 9.0])]),
        );

        // The type rule of the call validates the inputs of a direct application before the callee runs.
        assert_eq!(
            context.bind(call, vec![callee], &[f32_value(f32_vector_type(3), &[2.0, 3.0, 4.0])]),
            Err(ProgramError::from(TypeError::invalid(format!(
                "`{CALL_OPERATION_NAME}` input #0 expected f32[2] but got f32[3]",
            )))),
        );
    }

    #[test]
    fn test_call_operation_interpret_places_inputs_at_declared_input_shardings() {
        // A call accepts an input whose placement differs from the sharding that its callee input type declares and
        // places it at that sharding, so that the callee receives the input `x`, which is split along `x`, replicated,
        // and the output of the call has the replicated type that the type rule of the call infers.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let split = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let split_type = f32_vector_type(4).with_sharding(split).unwrap();
        let replicated_type = f32_vector_type(4).with_sharding(Sharding::replicated(mesh, 1)).unwrap();
        let callee = build_program(vec![ArrayIrType::Array(replicated_type.clone())], |builder, inputs| {
            builder
                .add_instruction(
                    ArrayOperation::from(MulOperation::new()),
                    Vec::new(),
                    vec![inputs[0], inputs[0]],
                    None,
                )
                .unwrap()
                .to_vec()
        });
        let program = build_program(vec![ArrayIrType::Array(split_type.clone())], |builder, inputs| {
            let callee = builder.import_program(callee);
            let call = TestOperation::Call(CallOperation::new(0));
            builder.add_instruction(call, vec![callee], inputs.to_vec(), None).unwrap().to_vec()
        });
        assert_eq!(
            program.to_string(),
            formatdoc! {"
                lambda %0:{split_type} .
                let %1:{replicated_type} = call %0 [
                    callee={{
                        lambda %0:{replicated_type} .
                        let %1:{replicated_type} = mul %0 %0
                        in (%1)
                    }},
                ]
                in (%1)"},
        );
        assert_eq!(
            program.interpret(vec![f32_value(split_type, &[1.0, 2.0, 3.0, 4.0])]),
            Ok(vec![f32_value(replicated_type, &[1.0, 4.0, 9.0, 16.0])]),
        );
    }

    #[test]
    fn test_call_operation_interpret_replicates_inputs_of_callee_input_types_without_shardings() {
        // A callee input type without a sharding receives its input replicated over the mesh of the call, which is the
        // mesh of the input when the call names none, as the eager application of a backend places it. The callee
        // therefore receives the input `x`, which is split along `x`, replicated, and its output is replicated too.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let split = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let split_type = f32_vector_type(4).with_sharding(split).unwrap();
        let replicated_type = f32_vector_type(4).with_sharding(Sharding::replicated(mesh, 1)).unwrap();
        let callee = build_program(vec![ArrayIrType::Array(f32_vector_type(4))], |builder, inputs| {
            builder
                .add_instruction(
                    ArrayOperation::from(MulOperation::new()),
                    Vec::new(),
                    vec![inputs[0], inputs[0]],
                    None,
                )
                .unwrap()
                .to_vec()
        });
        let program = build_program(vec![ArrayIrType::Array(split_type.clone())], |builder, inputs| {
            let callee = builder.import_program(callee.clone());
            let call = TestOperation::Call(CallOperation::new(0));
            builder.add_instruction(call, vec![callee], inputs.to_vec(), None).unwrap().to_vec()
        });
        assert_eq!(
            program.to_string(),
            formatdoc! {"
                lambda %0:{split_type} .
                let %1:f32[4] = call %0 [
                    callee={{
                        lambda %0:f32[4] .
                        let %1:f32[4] = mul %0 %0
                        in (%1)
                    }},
                ]
                in (%1)"},
        );
        assert_eq!(
            program.interpret(vec![f32_value(split_type, &[1.0, 2.0, 3.0, 4.0])]),
            Ok(vec![f32_value(replicated_type, &[1.0, 4.0, 9.0, 16.0])]),
        );

        // A placement over a single device is placed too, so the input `y`, which is split along the axis `y` of size
        // one, is replicated over its mesh.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("y", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let split = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["y"])]).unwrap();
        let split_type = f32_vector_type(4).with_sharding(split).unwrap();
        let replicated_type = f32_vector_type(4).with_sharding(Sharding::replicated(mesh, 1)).unwrap();
        let program = build_program(vec![ArrayIrType::Array(split_type.clone())], |builder, inputs| {
            let callee = builder.import_program(callee);
            let call = TestOperation::Call(CallOperation::new(0));
            builder.add_instruction(call, vec![callee], inputs.to_vec(), None).unwrap().to_vec()
        });
        assert_eq!(
            program.interpret(vec![f32_value(split_type, &[1.0, 2.0, 3.0, 4.0])]),
            Ok(vec![f32_value(replicated_type, &[1.0, 4.0, 9.0, 16.0])]),
        );
    }

    #[test]
    fn test_call_operation_interpret_places_inputs_at_boundary_shardings_that_callees_accept() {
        // The input `a` of a callee input type without a sharding is placed at its sharding override, which splits it
        // along `x`, so the callee receives `a` split. The input `b` of a callee input type that declares the
        // replicated sharding enters the callee at that sharding, which is the only one that the callee accepts, so its
        // override, which splits it along `x`, is not observable inside the callee.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let device_mesh = DeviceMesh::new(mesh.clone(), vec![Device::new(0, 0), Device::new(1, 0)]).unwrap();
        let split = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let split_type = f32_vector_type(4).with_sharding(split.clone()).unwrap();
        let replicated_type = f32_vector_type(4).with_sharding(Sharding::replicated(mesh, 1)).unwrap();
        let callee_input_types =
            vec![ArrayIrType::Array(f32_vector_type(4)), ArrayIrType::Array(replicated_type.clone())];
        let callee = build_program(callee_input_types, |builder, inputs| {
            inputs
                .iter()
                .map(|input| {
                    let mul = ArrayOperation::from(MulOperation::new());
                    builder.add_instruction(mul, Vec::new(), vec![*input, *input], None).unwrap()[0]
                })
                .collect()
        });
        let input_types =
            vec![ArrayIrType::Array(replicated_type.clone()), ArrayIrType::Array(replicated_type.clone())];
        let call = |operation: CallOperation<ArrayIrType>| {
            build_program(input_types.clone(), |builder, inputs| {
                let callee = builder.import_program(callee.clone());
                let call = TestOperation::Call(operation);
                builder.add_instruction(call, vec![callee], inputs.to_vec(), None).unwrap().to_vec()
            })
        };
        let overrides = "[{mesh<['x'=2:explicit]>, [{'x'}]}, {mesh<['x'=2:explicit]>, [{'x'}]}]";
        let overridden = call(CallOperation::new(0).with_boundary(
            Some(&device_mesh),
            vec![Some(split.clone()), Some(split.clone())],
            Vec::new(),
        ));
        assert_eq!(
            overridden.to_string(),
            formatdoc! {"
                lambda %0:{replicated_type}, %1:{replicated_type} .
                let %2:f32[4], %3:{replicated_type} = call [
                    input_shardings={overrides},
                    devices=[0, 1],
                ] %0 %1 [
                    callee={{
                        lambda %0:f32[4], %1:{replicated_type} .
                        let %2:f32[4] = mul %0 %0
                            %3:{replicated_type} = mul %1 %1
                        in (%2, %3)
                    }},
                ]
                in (%2, %3)"},
        );
        let inputs = vec![
            f32_value(replicated_type.clone(), &[1.0, 2.0, 3.0, 4.0]),
            f32_value(replicated_type.clone(), &[5.0, 6.0, 7.0, 8.0]),
        ];
        assert_eq!(
            overridden.interpret(inputs.clone()),
            Ok(vec![
                f32_value(split_type.clone(), &[1.0, 4.0, 9.0, 16.0]),
                f32_value(replicated_type.clone(), &[25.0, 36.0, 49.0, 64.0]),
            ]),
        );

        // A call that a program transform derives from another call places the inputs that it receives from the
        // original call at their derived shardings by the same rule.
        let derived = call(CallOperation::new(0).with_derived_input_shardings(vec![Some(split.clone()), Some(split)]));
        assert_eq!(
            derived.to_string(),
            formatdoc! {"
                lambda %0:{replicated_type}, %1:{replicated_type} .
                let %2:f32[4], %3:{replicated_type} = call [
                    derived_input_shardings={overrides},
                ] %0 %1 [
                    callee={{
                        lambda %0:f32[4], %1:{replicated_type} .
                        let %2:f32[4] = mul %0 %0
                            %3:{replicated_type} = mul %1 %1
                        in (%2, %3)
                    }},
                ]
                in (%2, %3)"},
        );
        assert_eq!(
            derived.interpret(inputs),
            Ok(vec![
                f32_value(split_type, &[1.0, 4.0, 9.0, 16.0]),
                f32_value(replicated_type, &[25.0, 36.0, 49.0, 64.0]),
            ]),
        );
    }

    #[test]
    fn test_call_operation_interpret_keeps_no_placement_along_auto_axes() {
        // Array types do not record placement along auto mesh axes, and reference arrays live on no devices, so they
        // report no placement. The overrides split both inputs along the auto axis `x`, which a backend applies at
        // runtime, while the reference results keep the replicated types that the type rule infers and report no
        // placement, whether the callee input type declares no sharding (`a`) or a sharding over the auto mesh (`b`).
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let device_mesh = DeviceMesh::new(mesh.clone(), vec![Device::new(0, 0), Device::new(1, 0)]).unwrap();
        let split = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated_type = f32_vector_type(4).with_sharding(Sharding::replicated(mesh, 1)).unwrap();
        let callee_input_types =
            vec![ArrayIrType::Array(f32_vector_type(4)), ArrayIrType::Array(replicated_type.clone())];
        let callee = build_program(callee_input_types, |builder, inputs| {
            inputs
                .iter()
                .map(|input| {
                    let mul = ArrayOperation::from(MulOperation::new());
                    builder.add_instruction(mul, Vec::new(), vec![*input, *input], None).unwrap()[0]
                })
                .collect()
        });
        let input_types =
            vec![ArrayIrType::Array(replicated_type.clone()), ArrayIrType::Array(replicated_type.clone())];
        let program = build_program(input_types, |builder, inputs| {
            let callee = builder.import_program(callee);
            let call = TestOperation::Call(CallOperation::new(0).with_boundary(
                Some(&device_mesh),
                vec![Some(split.clone()), Some(split)],
                Vec::new(),
            ));
            builder.add_instruction(call, vec![callee], inputs.to_vec(), None).unwrap().to_vec()
        });
        let overrides = "[{mesh<['x'=2:auto]>, [{'x'}]}, {mesh<['x'=2:auto]>, [{'x'}]}]";
        assert_eq!(
            program.to_string(),
            formatdoc! {"
                lambda %0:{replicated_type}, %1:{replicated_type} .
                let %2:f32[4], %3:{replicated_type} = call [
                    input_shardings={overrides},
                    devices=[0, 1],
                ] %0 %1 [
                    callee={{
                        lambda %0:f32[4], %1:{replicated_type} .
                        let %2:f32[4] = mul %0 %0
                            %3:{replicated_type} = mul %1 %1
                        in (%2, %3)
                    }},
                ]
                in (%2, %3)"},
        );
        let outputs = program
            .interpret(vec![
                f32_value(replicated_type.clone(), &[1.0, 2.0, 3.0, 4.0]),
                f32_value(replicated_type.clone(), &[5.0, 6.0, 7.0, 8.0]),
            ])
            .unwrap();
        assert_eq!(
            outputs,
            vec![
                f32_value(replicated_type.clone(), &[1.0, 4.0, 9.0, 16.0]),
                f32_value(replicated_type, &[25.0, 36.0, 49.0, 64.0]),
            ],
        );
        assert_eq!(outputs.iter().map(Placed::placement).collect::<Vec<_>>(), vec![None, None]);
    }

    #[test]
    fn test_call_operation_interpret_in_lockstep() {
        // Inside the body of a `shard_map`, a call whose callee sums over the manual axis `x` runs its callee once for
        // both devices in lockstep, which replaces the values `1` and `2` of the two devices by their sum.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let boundary =
            ShardMapBoundary::new(mesh.clone(), vec![sharded.clone()], vec![sharded.clone()], Vec::new()).unwrap();
        let global_type = f32_vector_type(2);
        let local_type = ArrayIrType::Array(
            boundary.local_input_type(0, &global_type.clone().with_sharding(sharded.clone()).unwrap()).unwrap(),
        );
        let callee = build_program(vec![local_type.clone()], |builder, inputs| {
            let sum = ParallelReduceOperation::new(ReductionKind::Sum, "x".to_string()).with_mesh(mesh.clone());
            let sum = builder
                .add_instruction(ArrayOperation::ParallelReduce(sum), Vec::new(), vec![inputs[0]], None)
                .unwrap()[0];
            let vary = ArrayOperation::ParallelVary(ParallelVaryOperation::new("x".to_string()));
            builder.add_instruction(vary, Vec::new(), vec![sum], None).unwrap().to_vec()
        });
        let body = build_program(vec![local_type], |builder, inputs| {
            let callee = builder.import_program(callee);
            let call = TestOperation::Call(CallOperation::new(0));
            builder.add_instruction(call, vec![callee], inputs.to_vec(), None).unwrap().to_vec()
        });
        let input_types = vec![ArrayIrType::Array(global_type.clone())];
        let operation = ShardMapOperation::from_program(&body, input_types.clone(), boundary).unwrap();
        let program = build_program(input_types, |builder, inputs| {
            let body = builder.import_program(body);
            builder.add_instruction(operation, vec![body], inputs.to_vec(), None).unwrap().to_vec()
        });
        assert_eq!(
            program.interpret(vec![f32_value(global_type.clone(), &[1.0, 2.0])]),
            Ok(vec![f32_value(global_type.with_sharding(sharded).unwrap(), &[3.0, 3.0])]),
        );
    }

    #[test]
    fn test_call_operation_interpret_in_lockstep_varies_outputs_along_extra_input_variation() {
        // Inside the body of a `shard_map`, input `b` varies along the manual axis `x` beyond its callee input type, so
        // both outputs of the call vary along `x` by the type rule of the call, although the callee computes them from
        // the sum of `a` over `x` and from a constant, which are invariant. The sums over `x` that follow therefore
        // count the value of each of the two devices, which are `3` and `0`.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated = Sharding::replicated(mesh.clone(), 1);
        let boundary = ShardMapBoundary::new(
            mesh.clone(),
            vec![sharded.clone(), sharded.clone()],
            vec![replicated.clone(), replicated.clone()],
            Vec::new(),
        )
        .unwrap();
        let a_type = f32_vector_type(2);
        let b_type = f32_vector_type(4);
        let local_type = |index, r#type: &ArrayType| {
            let global_type = r#type.clone().with_sharding(sharded.clone()).unwrap();
            ArrayIrType::Array(boundary.local_input_type(index, &global_type).unwrap())
        };
        let (a_local_type, b_local_type) = (local_type(0, &a_type), local_type(1, &b_type));
        let parallel_sum = || {
            ArrayOperation::ParallelReduce(
                ParallelReduceOperation::new(ReductionKind::Sum, "x".to_string()).with_mesh(mesh.clone()),
            )
        };
        let callee =
            build_program(vec![a_local_type.clone(), ArrayIrType::Array(f32_vector_type(2))], |builder, inputs| {
                let sum = builder.add_instruction(parallel_sum(), Vec::new(), vec![inputs[0]], None).unwrap()[0];
                let zero = ArrayOperation::from(ZeroOperation::new(f32_vector_type(1)));
                let zero = builder.add_instruction(zero, Vec::new(), Vec::new(), None).unwrap()[0];
                vec![sum, zero]
            });
        let body = build_program(vec![a_local_type.clone(), b_local_type.clone()], |builder, inputs| {
            let callee = builder.import_program(callee);
            let call = TestOperation::Call(CallOperation::new(0));
            let outputs = builder.add_instruction(call, vec![callee], inputs.to_vec(), None).unwrap().to_vec();
            outputs
                .into_iter()
                .map(|output| builder.add_instruction(parallel_sum(), Vec::new(), vec![output], None).unwrap()[0])
                .collect()
        });
        let output_type = f32_vector_type(1).with_sharding(replicated).unwrap();
        let sum = r#"parallel_reduce [kind=sum, axis_name="x", mesh=['x'=2:manual]]"#;
        assert_eq!(
            body.to_string(),
            formatdoc! {"
                lambda %0:{a_local_type}, %1:{b_local_type} .
                let %2:{a_local_type}, %3:{a_local_type} = call %0 %1 [
                    callee={{
                        lambda %0:{a_local_type}, %1:f32[2] .
                        let %2:{output_type} = {sum} %0
                            %3:f32[1] = zero [type=f32[1]]
                        in (%2, %3)
                    }},
                ]
                    %4:{output_type} = {sum} %2
                    %5:{output_type} = {sum} %3
                in (%4, %5)"},
        );
        let input_types = vec![ArrayIrType::Array(a_type.clone()), ArrayIrType::Array(b_type.clone())];
        let operation = ShardMapOperation::from_program(&body, input_types.clone(), boundary).unwrap();
        let program = build_program(input_types, |builder, inputs| {
            let body = builder.import_program(body);
            builder.add_instruction(operation, vec![body], inputs.to_vec(), None).unwrap().to_vec()
        });
        assert_eq!(
            program.interpret(vec![f32_value(a_type, &[1.0, 2.0]), f32_value(b_type, &[1.0, 2.0, 3.0, 4.0])]),
            Ok(vec![f32_value(output_type.clone(), &[6.0]), f32_value(output_type, &[0.0])]),
        );
    }

    #[test]
    fn test_call_operation_interpret_in_lockstep_varies_outputs_of_callees_without_collectives() {
        // Inside the body of a `shard_map`, input `a` varies along the manual axis `x` beyond its callee input type, so
        // the output of the call varies along `x` by the type rule of the call, although the callee contains no
        // collective and computes it from the invariant input `b` alone. The emulation therefore runs the call once for
        // both devices, which places `a` at the replicated sharding that its callee input type declares and keeps its
        // variation, and the sum over `x` that follows counts the value `b * b` of each of the two devices.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated = Sharding::replicated(mesh.clone(), 1);
        let boundary = ShardMapBoundary::new(
            mesh.clone(),
            vec![sharded.clone(), replicated.clone()],
            vec![replicated.clone()],
            Vec::new(),
        )
        .unwrap();
        let global_type = f32_vector_type(2);
        let a_local_type = ArrayIrType::Array(
            boundary.local_input_type(0, &global_type.clone().with_sharding(sharded).unwrap()).unwrap(),
        );
        let b_local_type = ArrayIrType::Array(
            boundary
                .local_input_type(1, &global_type.clone().with_sharding(replicated.clone()).unwrap())
                .unwrap(),
        );
        let a_declared_type = ArrayIrType::Array(f32_vector_type(1).with_sharding(replicated.clone()).unwrap());
        let callee =
            build_program(vec![a_declared_type.clone(), ArrayIrType::Array(f32_vector_type(2))], |builder, inputs| {
                builder
                    .add_instruction(
                        ArrayOperation::from(MulOperation::new()),
                        Vec::new(),
                        vec![inputs[1], inputs[1]],
                        None,
                    )
                    .unwrap()
                    .to_vec()
            });
        let body = build_program(vec![a_local_type.clone(), b_local_type.clone()], |builder, inputs| {
            let callee = builder.import_program(callee);
            let call = TestOperation::Call(CallOperation::new(0));
            let squared = builder.add_instruction(call, vec![callee], inputs.to_vec(), None).unwrap()[0];
            let sum = ParallelReduceOperation::new(ReductionKind::Sum, "x".to_string()).with_mesh(mesh.clone());
            builder
                .add_instruction(ArrayOperation::ParallelReduce(sum), Vec::new(), vec![squared], None)
                .unwrap()
                .to_vec()
        });
        let varying_type = f32_vector_type(2)
            .with_sharding(Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        let output_type = global_type.clone().with_sharding(replicated).unwrap();
        assert_eq!(
            body.to_string(),
            formatdoc! {"
                lambda %0:{a_local_type}, %1:{b_local_type} .
                let %2:{varying_type} = call %0 %1 [
                    callee={{
                        lambda %0:{a_declared_type}, %1:f32[2] .
                        let %2:f32[2] = mul %1 %1
                        in (%2)
                    }},
                ]
                    %3:{output_type} = parallel_reduce [kind=sum, axis_name=\"x\", mesh=['x'=2:manual]] %2
                in (%3)"},
        );
        let input_types = vec![ArrayIrType::Array(global_type.clone()), ArrayIrType::Array(global_type.clone())];
        let operation = ShardMapOperation::from_program(&body, input_types.clone(), boundary).unwrap();
        let program = build_program(input_types, |builder, inputs| {
            let body = builder.import_program(body);
            builder.add_instruction(operation, vec![body], inputs.to_vec(), None).unwrap().to_vec()
        });
        assert_eq!(
            program.interpret(vec![f32_value(global_type.clone(), &[1.0, 2.0]), f32_value(global_type, &[3.0, 4.0])]),
            Ok(vec![f32_value(output_type, &[18.0, 32.0])]),
        );
    }

    #[test]
    fn test_call_operation_interpret_in_lockstep_runs_nested_conditions_at_callee_input_types() {
        // Inside the body of a `shard_map`, input `a` varies along the manual axis `x` beyond its first callee input
        // type, and the callee, traced at that plain type, selects with a `condition` whose branches declare it too.
        // The emulation runs the callee at the types that it was traced at, as a backend does, so the `condition`
        // accepts `a` and squares it on both devices, and the sum over `x` that follows is `1 + 4`. The second callee
        // input declares the variation of `a`, so the callee also sums `a` over `x`, which runs the call in lockstep for
        // that reason too, and the second output then counts that sum `3` once per device.
        let (mesh, a_local_type) = lockstep_call_mesh_and_input_type();
        let plain_type = ArrayIrType::Array(f32_vector_type(1));
        let callee = build_program(vec![plain_type.clone(), a_local_type.clone()], |builder, inputs| {
            let zero = ArrayOperation::from(ZeroOperation::new(ArrayType::scalar(DataType::F32)));
            let zero = builder.add_instruction(zero, Vec::new(), Vec::new(), None).unwrap()[0];
            let compare = ArrayOperation::Compare(CompareOperation::<ArrayType>::new(ComparisonDirection::Equal));
            let predicate = builder.add_instruction(compare, Vec::new(), vec![zero, zero], None).unwrap()[0];
            let true_branch = builder.import_program(lockstep_call_square(plain_type.clone()));
            let false_branch =
                builder.import_program(build_program(vec![plain_type.clone()], |_, inputs| inputs.to_vec()));
            let condition = ConditionOperation::<ArrayIrType>::new();
            let selected = builder
                .add_instruction(condition, vec![true_branch, false_branch], vec![predicate, inputs[0]], None)
                .unwrap()[0];
            let sum = ParallelReduceOperation::new(ReductionKind::Sum, "x".to_string()).with_mesh(mesh.clone());
            let sum = builder
                .add_instruction(ArrayOperation::ParallelReduce(sum), Vec::new(), vec![inputs[1]], None)
                .unwrap()[0];
            vec![selected, sum]
        });
        let (body, program) = lockstep_call_program(&mesh, &a_local_type, callee, false);
        let varying = &a_local_type;
        let invariant = "f32[1][sharding={mesh<['x'=2:manual]>, [{}]}]";
        let sum = r#"parallel_reduce [kind=sum, axis_name="x", mesh=['x'=2:manual]]"#;
        assert_eq!(
            body.to_string(),
            formatdoc! {"
                lambda %0:{varying} .
                let %1:{varying}, %2:{varying} = call %0 %0 [
                    callee={{
                        lambda %0:f32[1], %1:{varying} .
                        let %2:f32[] = zero [type=f32[]]
                            %3:bool[] = compare [direction=Equal] %2 %2
                            %4:f32[1] = condition %3 %0 [
                                true={{
                                    lambda %0:f32[1] .
                                    let %1:f32[1] = mul %0 %0
                                    in (%1)
                                }},
                                false={{
                                    lambda %0:f32[1] .
                                    in (%0)
                                }},
                            ]
                            %5:{invariant} = {sum} %1
                        in (%4, %5)
                    }},
                ]
                    %3:{invariant} = {sum} %1
                    %4:{invariant} = {sum} %2
                in (%3, %4)"},
        );
        let output_type = f32_vector_type(1).with_sharding(Sharding::replicated(mesh, 1)).unwrap();
        assert_eq!(
            program.interpret(vec![f32_value(f32_vector_type(2), &[1.0, 2.0])]),
            Ok(vec![f32_value(output_type.clone(), &[5.0]), f32_value(output_type, &[6.0])]),
        );
    }

    #[test]
    fn test_call_operation_interpret_in_lockstep_selects_nested_condition_branches_per_device() {
        // The callee selects with a `condition` on whether its first input is `0`, which is the index of each device
        // along `x` and varies along `x` beyond its plain callee input type, as `a` does. The branches use no
        // collective, so each device takes its own branch, as on a backend: device 0 squares its value `1` and device 1
        // keeps its value `2`, and the sum over `x` that follows is `1 + 2`.
        let (mesh, a_local_type) = lockstep_call_mesh_and_input_type();
        let plain_type = ArrayIrType::Array(f32_vector_type(1));
        let index_type = ArrayIrType::Array(ArrayType::scalar(DataType::U64));
        let callee = build_program(vec![index_type, plain_type.clone()], |builder, inputs| {
            let zero = ArrayOperation::from(ZeroLikeOperation::<ArrayType>::new());
            let zero = builder.add_instruction(zero, Vec::new(), vec![inputs[0]], None).unwrap()[0];
            let compare = ArrayOperation::Compare(CompareOperation::<ArrayType>::new(ComparisonDirection::Equal));
            let predicate = builder.add_instruction(compare, Vec::new(), vec![inputs[0], zero], None).unwrap()[0];
            let true_branch = builder.import_program(lockstep_call_square(plain_type.clone()));
            let false_branch =
                builder.import_program(build_program(vec![plain_type.clone()], |_, inputs| inputs.to_vec()));
            let condition = ConditionOperation::<ArrayIrType>::new();
            builder
                .add_instruction(condition, vec![true_branch, false_branch], vec![predicate, inputs[1]], None)
                .unwrap()
                .to_vec()
        });
        let (body, program) = lockstep_call_program(&mesh, &a_local_type, callee, true);
        let varying = &a_local_type;
        let invariant = "f32[1][sharding={mesh<['x'=2:manual]>, [{}]}]";
        let sum = r#"parallel_reduce [kind=sum, axis_name="x", mesh=['x'=2:manual]]"#;
        let index = "u64[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}]";
        let axis_index = r#"axis_index [axis_name="x", mesh=['x'=2:manual]]"#;
        assert_eq!(
            body.to_string(),
            formatdoc! {"
                lambda %0:{varying} .
                let %1:{index} = {axis_index}
                    %2:{varying} = call %1 %0 [
                        callee={{
                            lambda %0:u64[], %1:f32[1] .
                            let %2:u64[] = zero_like %0
                                %3:bool[] = compare [direction=Equal] %0 %2
                                %4:f32[1] = condition %3 %1 [
                                    true={{
                                        lambda %0:f32[1] .
                                        let %1:f32[1] = mul %0 %0
                                        in (%1)
                                    }},
                                    false={{
                                        lambda %0:f32[1] .
                                        in (%0)
                                    }},
                                ]
                            in (%4)
                        }},
                    ]
                    %3:{invariant} = {sum} %2
                in (%3)"},
        );
        let output_type = f32_vector_type(1).with_sharding(Sharding::replicated(mesh, 1)).unwrap();
        assert_eq!(
            program.interpret(vec![f32_value(f32_vector_type(2), &[1.0, 2.0])]),
            Ok(vec![f32_value(output_type, &[3.0])]),
        );
    }

    #[test]
    fn test_call_operation_interpret_in_lockstep_rejects_disagreeing_predicates_of_nested_collective_conditions() {
        // As in `test_call_operation_interpret_in_lockstep_selects_nested_condition_branches_per_device`, the devices
        // disagree on the predicate of the `condition` of the callee, but its `true` branch sums over `x`, so the
        // branches must run in lockstep, which requires every device to take the same path. The emulation rejects
        // this, as its documentation states, while a backend would run a collective on device 0 alone.
        let (mesh, a_local_type) = lockstep_call_mesh_and_input_type();
        let declared_type =
            ArrayIrType::Array(f32_vector_type(1).with_sharding(Sharding::replicated(mesh.clone(), 1)).unwrap());
        let index_type = ArrayIrType::Array(ArrayType::scalar(DataType::U64));
        let true_branch = build_program(vec![declared_type.clone()], |builder, inputs| {
            let vary = ArrayOperation::ParallelVary(ParallelVaryOperation::new("x".to_string()));
            let varying = builder.add_instruction(vary, Vec::new(), vec![inputs[0]], None).unwrap()[0];
            let sum = ParallelReduceOperation::new(ReductionKind::Sum, "x".to_string()).with_mesh(mesh.clone());
            builder
                .add_instruction(ArrayOperation::ParallelReduce(sum), Vec::new(), vec![varying], None)
                .unwrap()
                .to_vec()
        });
        let callee = build_program(vec![index_type, declared_type.clone()], |builder, inputs| {
            let zero = ArrayOperation::from(ZeroLikeOperation::<ArrayType>::new());
            let zero = builder.add_instruction(zero, Vec::new(), vec![inputs[0]], None).unwrap()[0];
            let compare = ArrayOperation::Compare(CompareOperation::<ArrayType>::new(ComparisonDirection::Equal));
            let predicate = builder.add_instruction(compare, Vec::new(), vec![inputs[0], zero], None).unwrap()[0];
            let true_branch = builder.import_program(true_branch);
            let false_branch =
                builder.import_program(build_program(vec![declared_type.clone()], |_, inputs| inputs.to_vec()));
            let condition = ConditionOperation::<ArrayIrType>::new();
            builder
                .add_instruction(condition, vec![true_branch, false_branch], vec![predicate, inputs[1]], None)
                .unwrap()
                .to_vec()
        });
        let (body, program) = lockstep_call_program(&mesh, &a_local_type, callee, true);
        let varying = &a_local_type;
        let invariant = "f32[1][sharding={mesh<['x'=2:manual]>, [{}]}]";
        let sum = r#"parallel_reduce [kind=sum, axis_name="x", mesh=['x'=2:manual]]"#;
        let index = "u64[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}]";
        let axis_index = r#"axis_index [axis_name="x", mesh=['x'=2:manual]]"#;
        let vary = r#"parallel_vary [axis_name="x"]"#;
        assert_eq!(
            body.to_string(),
            formatdoc! {"
                lambda %0:{varying} .
                let %1:{index} = {axis_index}
                    %2:{varying} = call %1 %0 [
                        callee={{
                            lambda %0:u64[], %1:{invariant} .
                            let %2:u64[] = zero_like %0
                                %3:bool[] = compare [direction=Equal] %0 %2
                                %4:{invariant} = condition %3 %1 [
                                    true={{
                                        lambda %0:{invariant} .
                                        let %1:{varying} = {vary} %0
                                            %2:{invariant} = {sum} %1
                                        in (%2)
                                    }},
                                    false={{
                                        lambda %0:{invariant} .
                                        in (%0)
                                    }},
                                ]
                            in (%4)
                        }},
                    ]
                    %3:{invariant} = {sum} %2
                in (%3)"},
        );
        assert!(matches!(
            program.interpret(vec![f32_value(f32_vector_type(2), &[1.0, 2.0])]),
            Err(ProgramError::UnsupportedOperation { message }) if message == "the devices of a `shard_map` disagree \
                on the predicate of a `condition` whose regions use collectives over the manual axes, but lockstep \
                emulation requires every device to take the same path",
        ));
    }

    #[test]
    fn test_call_operation_interpret_in_lockstep_runs_nested_while_loops_at_callee_input_types() {
        // The callee, traced at a plain input type, squares its input once in a `while` loop whose state declares that
        // type. The loop accepts `a`, which varies along `x` beyond that type, because the callee runs at the types
        // that it was traced at, and the sum over `x` that follows is `1 + 4`.
        let (mesh, a_local_type) = lockstep_call_mesh_and_input_type();
        let plain_type = ArrayIrType::Array(f32_vector_type(1));
        let counter_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let condition = build_program(vec![counter_type.clone(), plain_type.clone()], |builder, inputs| {
            let one = ArrayOperation::OneLike(OneLikeOperation::new());
            let one = builder.add_instruction(one, Vec::new(), vec![inputs[0]], None).unwrap()[0];
            let compare = ArrayOperation::Compare(CompareOperation::<ArrayType>::new(ComparisonDirection::LessThan));
            builder.add_instruction(compare, Vec::new(), vec![inputs[0], one], None).unwrap().to_vec()
        });
        let body = build_program(vec![counter_type, plain_type.clone()], |builder, inputs| {
            let one = ArrayOperation::OneLike(OneLikeOperation::new());
            let one = builder.add_instruction(one, Vec::new(), vec![inputs[0]], None).unwrap()[0];
            let add = ArrayOperation::from(AddOperation::new());
            let counter = builder.add_instruction(add, Vec::new(), vec![inputs[0], one], None).unwrap()[0];
            let mul = ArrayOperation::from(MulOperation::new());
            let squared = builder.add_instruction(mul, Vec::new(), vec![inputs[1], inputs[1]], None).unwrap()[0];
            vec![counter, squared]
        });
        let callee = build_program(vec![plain_type], |builder, inputs| {
            let zero = ArrayOperation::from(ZeroOperation::new(ArrayType::scalar(DataType::F32)));
            let zero = builder.add_instruction(zero, Vec::new(), Vec::new(), None).unwrap()[0];
            let regions = vec![builder.import_program(condition), builder.import_program(body)];
            vec![builder.add_instruction(WhileOperation::new(), regions, vec![zero, inputs[0]], None).unwrap()[1]]
        });
        let (body, program) = lockstep_call_program(&mesh, &a_local_type, callee, false);
        let varying = &a_local_type;
        let invariant = "f32[1][sharding={mesh<['x'=2:manual]>, [{}]}]";
        let sum = r#"parallel_reduce [kind=sum, axis_name="x", mesh=['x'=2:manual]]"#;
        assert_eq!(
            body.to_string(),
            formatdoc! {"
                lambda %0:{varying} .
                let %1:{varying} = call %0 [
                    callee={{
                        lambda %0:f32[1] .
                        let %1:f32[] = zero [type=f32[]]
                            %2:f32[], %3:f32[1] = while %1 %0 [
                                condition={{
                                    lambda %0:f32[], %1:f32[1] .
                                    let %2:f32[] = one_like %0
                                        %3:bool[] = compare [direction=LessThan] %0 %2
                                    in (%3)
                                }},
                                body={{
                                    lambda %0:f32[], %1:f32[1] .
                                    let %2:f32[] = one_like %0
                                        %3:f32[] = add %0 %2
                                        %4:f32[1] = mul %1 %1
                                    in (%3, %4)
                                }},
                            ]
                        in (%3)
                    }},
                ]
                    %2:{invariant} = {sum} %1
                in (%2)"},
        );
        let output_type = f32_vector_type(1).with_sharding(Sharding::replicated(mesh, 1)).unwrap();
        assert_eq!(
            program.interpret(vec![f32_value(f32_vector_type(2), &[1.0, 2.0])]),
            Ok(vec![f32_value(output_type, &[5.0])]),
        );
    }

    #[test]
    fn test_call_operation_interpret_in_lockstep_runs_nested_scans_at_callee_input_types() {
        // The callee, traced at a plain input type, doubles its input twice in a `scan` whose carry declares that type.
        // The scan accepts `a`, which varies along `x` beyond that type, because the callee runs at the types that it
        // was traced at, and the sum over `x` that follows is `4 + 8`.
        let (mesh, a_local_type) = lockstep_call_mesh_and_input_type();
        let plain_type = ArrayIrType::Array(f32_vector_type(1));
        let body =
            build_program(vec![ArrayType::scalar(DataType::I64).into(), plain_type.clone()], |builder, inputs| {
                let add = ArrayOperation::from(AddOperation::new());
                builder.add_instruction(add, Vec::new(), vec![inputs[1], inputs[1]], None).unwrap().to_vec()
            });
        let callee = build_program(vec![plain_type], |builder, inputs| {
            let body = builder.import_program(body);
            let scan = ScanOperation::<ArrayIrType>::new(1, 2usize);
            builder.add_instruction(scan, vec![body], inputs.to_vec(), None).unwrap().to_vec()
        });
        let (body, program) = lockstep_call_program(&mesh, &a_local_type, callee, false);
        let varying = &a_local_type;
        let invariant = "f32[1][sharding={mesh<['x'=2:manual]>, [{}]}]";
        let sum = r#"parallel_reduce [kind=sum, axis_name="x", mesh=['x'=2:manual]]"#;
        assert_eq!(
            body.to_string(),
            formatdoc! {"
                lambda %0:{varying} .
                let %1:{varying} = call %0 [
                    callee={{
                        lambda %0:f32[1] .
                        let %1:f32[1] = scan [carry_count=1, iteration_count=2, reverse=false] %0 [
                            body={{
                                lambda %0:i64[], %1:f32[1] .
                                let %2:f32[1] = add %1 %1
                                in (%2)
                            }},
                        ]
                        in (%1)
                    }},
                ]
                    %2:{invariant} = {sum} %1
                in (%2)"},
        );
        let output_type = f32_vector_type(1).with_sharding(Sharding::replicated(mesh, 1)).unwrap();
        assert_eq!(
            program.interpret(vec![f32_value(f32_vector_type(2), &[1.0, 2.0])]),
            Ok(vec![f32_value(output_type, &[12.0])]),
        );
    }

    #[test]
    fn test_call_operation_interpret_in_lockstep_runs_nested_composites_at_callee_input_types() {
        // The callee, traced at a plain input type, doubles its input in a `composite` whose primal region declares
        // that type. `a` enters the callee without its variation along `x`, so the `composite` accepts it and binds
        // once per device, because its region uses no collective, and the sum over `x` that follows is `2 + 4`. The
        // same `composite` staged directly in the body, at the type of `a`, computes the same sum.
        let (mesh, a_local_type) = lockstep_call_mesh_and_input_type();
        let plain_type = ArrayIrType::Array(f32_vector_type(1));
        let [(call_body, call_program), (direct_body, direct_program)] =
            lockstep_nested_programs(&mesh, &a_local_type, vec![plain_type], |builder, inputs, input_types| {
                let primal = builder.import_program(lockstep_call_double(input_types[0].clone()));
                let composite =
                    CompositeOperation::<TestValue, TestOperation>::from_rule_regions(CompositeJvpRule::Derived, false);
                builder.add_instruction(composite, vec![primal], inputs.to_vec(), None).unwrap().to_vec()
            });
        let varying = &a_local_type;
        let invariant = "f32[1][sharding={mesh<['x'=2:manual]>, [{}]}]";
        let sum = r#"parallel_reduce [kind=sum, axis_name="x", mesh=['x'=2:manual]]"#;
        assert_eq!(
            call_body.to_string(),
            formatdoc! {"
                lambda %0:{varying} .
                let %1:{varying} = call %0 [
                    callee={{
                        lambda %0:f32[1] .
                        let %1:f32[1] = composite [jvp_from_primal=true] %0 [
                            primal={{
                                lambda %0:f32[1] .
                                let %1:f32[1] = add %0 %0
                                in (%1)
                            }},
                        ]
                        in (%1)
                    }},
                ]
                    %2:{invariant} = {sum} %1
                in (%2)"},
        );
        assert_eq!(
            direct_body.to_string(),
            formatdoc! {"
                lambda %0:{varying} .
                let %1:{varying} = composite [jvp_from_primal=true] %0 [
                    primal={{
                        lambda %0:{varying} .
                        let %1:{varying} = add %0 %0
                        in (%1)
                    }},
                ]
                    %2:{invariant} = {sum} %1
                in (%2)"},
        );
        let output_type = f32_vector_type(1).with_sharding(Sharding::replicated(mesh, 1)).unwrap();
        let input = f32_value(f32_vector_type(2), &[1.0, 2.0]);
        assert_eq!(call_program.interpret(vec![input.clone()]), Ok(vec![f32_value(output_type.clone(), &[6.0])]));
        assert_eq!(direct_program.interpret(vec![input]), Ok(vec![f32_value(output_type, &[6.0])]));
    }

    #[test]
    fn test_call_operation_interpret_in_lockstep_runs_nested_collective_composites_at_callee_input_types() {
        // The callee scales its second input, of an invariant type, by the sum over `x` of its first input, which
        // declares the variation of `a`, in a `composite` whose primal region computes that sum, so the `composite`
        // runs in lockstep by its rule over all devices. `a` enters the second input without its variation along `x`,
        // so the `composite` accepts it and scales `1` and `2` by `1 + 2`, and the call varies the result along `x`,
        // so that the sum over `x` that follows is `3 + 6`. The same `composite` staged directly in the body, at the
        // type of `a`, whose region varies the sum along `x` to scale `a` by it, computes the same sum.
        let (mesh, a_local_type) = lockstep_call_mesh_and_input_type();
        let invariant_type =
            ArrayIrType::Array(f32_vector_type(1).with_sharding(Sharding::replicated(mesh.clone(), 1)).unwrap());
        let [(call_body, call_program), (direct_body, direct_program)] = lockstep_nested_programs(
            &mesh,
            &a_local_type,
            vec![a_local_type.clone(), invariant_type],
            |builder, inputs, input_types| {
                let primal = lockstep_call_scale_by_sum(&mesh, input_types[0].clone(), input_types[1].clone());
                let primal = builder.import_program(primal);
                let composite =
                    CompositeOperation::<TestValue, TestOperation>::from_rule_regions(CompositeJvpRule::Derived, false);
                builder.add_instruction(composite, vec![primal], inputs.to_vec(), None).unwrap().to_vec()
            },
        );
        let varying = &a_local_type;
        let invariant = "f32[1][sharding={mesh<['x'=2:manual]>, [{}]}]";
        let sum = r#"parallel_reduce [kind=sum, axis_name="x", mesh=['x'=2:manual]]"#;
        let vary = r#"parallel_vary [axis_name="x"]"#;
        assert_eq!(
            call_body.to_string(),
            formatdoc! {"
                lambda %0:{varying} .
                let %1:{varying} = call %0 %0 [
                    callee={{
                        lambda %0:{varying}, %1:{invariant} .
                        let %2:{invariant} = composite [jvp_from_primal=true] %0 %1 [
                            primal={{
                                lambda %0:{varying}, %1:{invariant} .
                                let %2:{invariant} = {sum} %0
                                    %3:{invariant} = mul %1 %2
                                in (%3)
                            }},
                        ]
                        in (%2)
                    }},
                ]
                    %2:{invariant} = {sum} %1
                in (%2)"},
        );
        assert_eq!(
            direct_body.to_string(),
            formatdoc! {"
                lambda %0:{varying} .
                let %1:{varying} = composite [jvp_from_primal=true] %0 %0 [
                    primal={{
                        lambda %0:{varying}, %1:{varying} .
                        let %2:{invariant} = {sum} %0
                            %3:{varying} = {vary} %2
                            %4:{varying} = mul %1 %3
                        in (%4)
                    }},
                ]
                    %2:{invariant} = {sum} %1
                in (%2)"},
        );
        let output_type = f32_vector_type(1).with_sharding(Sharding::replicated(mesh, 1)).unwrap();
        let input = f32_value(f32_vector_type(2), &[1.0, 2.0]);
        assert_eq!(call_program.interpret(vec![input.clone()]), Ok(vec![f32_value(output_type.clone(), &[9.0])]));
        assert_eq!(direct_program.interpret(vec![input]), Ok(vec![f32_value(output_type, &[9.0])]));
    }

    #[test]
    fn test_call_operation_interpret_in_lockstep_runs_nested_named_composites_at_callee_input_types() {
        // As in `test_call_operation_interpret_in_lockstep_runs_nested_composites_at_callee_input_types`, but the
        // `composite` is a call of a named composite, whose primal region is the body that the definition traces for
        // the input types of the call. The callee shares the body traced at the plain type, the direct form shares the
        // body traced at the type of `a`, and both compute the sum `2 + 4`.
        let (mesh, a_local_type) = lockstep_call_mesh_and_input_type();
        let plain_type = ArrayIrType::Array(f32_vector_type(1));
        let registration = CustomRuleRegistration::new(
            CustomRuleDefinition::<TestValue, TestOperation>::new("double").with_named_composite(NamedComposite::new(
                CompositeKey::new("ryft.tests.lockstep_double", 1),
                |input_types: &[ArrayIrType]| Ok(input_types.to_vec()),
                |inputs: &[CustomRuleTracer<TestValue, TestOperation>]| {
                    let add = ArrayOperation::from(AddOperation::new());
                    inputs[0].context().bind(add, Vec::new(), &[inputs[0].clone(), inputs[0].clone()])
                },
            )),
        );
        let [(call_body, call_program), (direct_body, direct_program)] =
            lockstep_nested_programs(&mesh, &a_local_type, vec![plain_type], |builder, inputs, input_types| {
                let body = registration.reference().body(input_types).unwrap();
                let primal = builder.intern_callee(&body, None).unwrap();
                let composite = TestOperation::Composite(CompositeOperation::new(registration.reference()));
                builder.add_instruction(composite, vec![primal], inputs.to_vec(), None).unwrap().to_vec()
            });
        let varying = &a_local_type;
        let invariant = "f32[1][sharding={mesh<['x'=2:manual]>, [{}]}]";
        let sum = r#"parallel_reduce [kind=sum, axis_name="x", mesh=['x'=2:manual]]"#;
        let key = r#"key="ryft.tests.lockstep_double.v1""#;
        assert_eq!(
            call_body.to_string(),
            formatdoc! {"
                lambda %0:{varying} .
                let %1:{varying} = call %0 [
                    callee={{
                        lambda %0:f32[1] .
                        let %1:f32[1] = composite [{key}] %0 [
                            primal={{
                                lambda %0:f32[1] .
                                let %1:f32[1] = add %0 %0
                                in (%1)
                            }},
                        ]
                        in (%1)
                    }},
                ]
                    %2:{invariant} = {sum} %1
                in (%2)"},
        );
        assert_eq!(
            direct_body.to_string(),
            formatdoc! {"
                lambda %0:{varying} .
                let %1:{varying} = composite [{key}] %0 [
                    primal={{
                        lambda %0:{varying} .
                        let %1:{varying} = add %0 %0
                        in (%1)
                    }},
                ]
                    %2:{invariant} = {sum} %1
                in (%2)"},
        );
        let output_type = f32_vector_type(1).with_sharding(Sharding::replicated(mesh, 1)).unwrap();
        let input = f32_value(f32_vector_type(2), &[1.0, 2.0]);
        assert_eq!(call_program.interpret(vec![input.clone()]), Ok(vec![f32_value(output_type.clone(), &[6.0])]));
        assert_eq!(direct_program.interpret(vec![input]), Ok(vec![f32_value(output_type, &[6.0])]));
    }

    #[test]
    fn test_call_operation_interpret_in_lockstep_runs_nested_collective_named_composites_at_callee_input_types() {
        // As in `test_call_operation_interpret_in_lockstep_runs_nested_collective_composites_at_callee_input_types`,
        // but the `composite` is a call of a named composite whose body computes the scaled sum, and which varies the
        // sum along `x` when the scaled input varies along `x`. Its call in the callee runs in lockstep, and both forms
        // compute the sum `3 + 6`.
        let (mesh, a_local_type) = lockstep_call_mesh_and_input_type();
        let invariant_type =
            ArrayIrType::Array(f32_vector_type(1).with_sharding(Sharding::replicated(mesh.clone(), 1)).unwrap());
        let body_mesh = mesh.clone();
        let registration = CustomRuleRegistration::new(
            CustomRuleDefinition::<TestValue, TestOperation>::new("scale_by_sum").with_named_composite(
                NamedComposite::new(
                    CompositeKey::new("ryft.tests.lockstep_scale_by_sum", 1),
                    |input_types: &[ArrayIrType]| Ok(vec![input_types[1].clone()]),
                    move |inputs: &[CustomRuleTracer<TestValue, TestOperation>]| {
                        let context = inputs[0].context();
                        let sum = ParallelReduceOperation::new(ReductionKind::Sum, "x".to_string())
                            .with_mesh(body_mesh.clone());
                        let sum = ArrayOperation::ParallelReduce(sum);
                        let mut sum = context.bind(sum, Vec::new(), std::slice::from_ref(&inputs[0]))?.remove(0);
                        if lockstep_call_varies_along_x(&inputs[1].r#type()) {
                            let vary = ArrayOperation::ParallelVary(ParallelVaryOperation::new("x".to_string()));
                            sum = context.bind(vary, Vec::new(), &[sum])?.remove(0);
                        }
                        let mul = ArrayOperation::from(MulOperation::new());
                        context.bind(mul, Vec::new(), &[inputs[1].clone(), sum])
                    },
                ),
            ),
        );
        let [(call_body, call_program), (direct_body, direct_program)] = lockstep_nested_programs(
            &mesh,
            &a_local_type,
            vec![a_local_type.clone(), invariant_type],
            |builder, inputs, input_types| {
                let body = registration.reference().body(input_types).unwrap();
                let primal = builder.intern_callee(&body, None).unwrap();
                let composite = TestOperation::Composite(CompositeOperation::new(registration.reference()));
                builder.add_instruction(composite, vec![primal], inputs.to_vec(), None).unwrap().to_vec()
            },
        );
        let varying = &a_local_type;
        let invariant = "f32[1][sharding={mesh<['x'=2:manual]>, [{}]}]";
        let sum = r#"parallel_reduce [kind=sum, axis_name="x", mesh=['x'=2:manual]]"#;
        let vary = r#"parallel_vary [axis_name="x"]"#;
        let key = r#"key="ryft.tests.lockstep_scale_by_sum.v1""#;
        assert_eq!(
            call_body.to_string(),
            formatdoc! {"
                lambda %0:{varying} .
                let %1:{varying} = call %0 %0 [
                    callee={{
                        lambda %0:{varying}, %1:{invariant} .
                        let %2:{invariant} = composite [{key}] %0 %1 [
                            primal={{
                                lambda %0:{varying}, %1:{invariant} .
                                let %2:{invariant} = {sum} %0
                                    %3:{invariant} = mul %1 %2
                                in (%3)
                            }},
                        ]
                        in (%2)
                    }},
                ]
                    %2:{invariant} = {sum} %1
                in (%2)"},
        );
        assert_eq!(
            direct_body.to_string(),
            formatdoc! {"
                lambda %0:{varying} .
                let %1:{varying} = composite [{key}] %0 %0 [
                    primal={{
                        lambda %0:{varying}, %1:{varying} .
                        let %2:{invariant} = {sum} %0
                            %3:{varying} = {vary} %2
                            %4:{varying} = mul %1 %3
                        in (%4)
                    }},
                ]
                    %2:{invariant} = {sum} %1
                in (%2)"},
        );
        let output_type = f32_vector_type(1).with_sharding(Sharding::replicated(mesh, 1)).unwrap();
        let input = f32_value(f32_vector_type(2), &[1.0, 2.0]);
        assert_eq!(call_program.interpret(vec![input.clone()]), Ok(vec![f32_value(output_type.clone(), &[9.0])]));
        assert_eq!(direct_program.interpret(vec![input]), Ok(vec![f32_value(output_type, &[9.0])]));
    }

    #[test]
    fn test_call_operation_interpret_in_lockstep_runs_nested_linear_calls_at_callee_input_types() {
        // The callee, traced at a plain input type, doubles its input in a `linear_call` whose forward and transpose
        // regions declare that type. `a` enters the callee without its variation along `x`, so the `linear_call`
        // accepts it and binds once per device, because its regions use no collective, and the sum over `x` that
        // follows is `2 + 4`. The same `linear_call` staged directly in the body, at the type of `a`, computes the
        // same sum.
        let (mesh, a_local_type) = lockstep_call_mesh_and_input_type();
        let plain_type = ArrayIrType::Array(f32_vector_type(1));
        let [(call_body, call_program), (direct_body, direct_program)] =
            lockstep_nested_programs(&mesh, &a_local_type, vec![plain_type], |builder, inputs, input_types| {
                let forward = builder.import_program(lockstep_call_double(input_types[0].clone()));
                let transpose = builder.import_program(lockstep_call_double(input_types[0].clone()));
                let linear_call = LinearCallOperation::<ArrayIrType>::new(0);
                builder
                    .add_instruction(linear_call, vec![forward, transpose], inputs.to_vec(), None)
                    .unwrap()
                    .to_vec()
            });
        let varying = &a_local_type;
        let invariant = "f32[1][sharding={mesh<['x'=2:manual]>, [{}]}]";
        let sum = r#"parallel_reduce [kind=sum, axis_name="x", mesh=['x'=2:manual]]"#;
        assert_eq!(
            call_body.to_string(),
            formatdoc! {"
                lambda %0:{varying} .
                let %1:{varying} = call %0 [
                    callee={{
                        lambda %0:f32[1] .
                        let %1:f32[1] = linear_call [residual_count=0] %0 [
                            forward={{
                                lambda %0:f32[1] .
                                let %1:f32[1] = add %0 %0
                                in (%1)
                            }},
                            transpose={{
                                lambda %0:f32[1] .
                                let %1:f32[1] = add %0 %0
                                in (%1)
                            }},
                        ]
                        in (%1)
                    }},
                ]
                    %2:{invariant} = {sum} %1
                in (%2)"},
        );
        assert_eq!(
            direct_body.to_string(),
            formatdoc! {"
                lambda %0:{varying} .
                let %1:{varying} = linear_call [residual_count=0] %0 [
                    forward={{
                        lambda %0:{varying} .
                        let %1:{varying} = add %0 %0
                        in (%1)
                    }},
                    transpose={{
                        lambda %0:{varying} .
                        let %1:{varying} = add %0 %0
                        in (%1)
                    }},
                ]
                    %2:{invariant} = {sum} %1
                in (%2)"},
        );
        let output_type = f32_vector_type(1).with_sharding(Sharding::replicated(mesh, 1)).unwrap();
        let input = f32_value(f32_vector_type(2), &[1.0, 2.0]);
        assert_eq!(call_program.interpret(vec![input.clone()]), Ok(vec![f32_value(output_type.clone(), &[6.0])]));
        assert_eq!(direct_program.interpret(vec![input]), Ok(vec![f32_value(output_type, &[6.0])]));
    }

    #[test]
    fn test_call_operation_interpret_in_lockstep_runs_nested_collective_linear_calls_at_callee_input_types() {
        // The callee scales its second input, of an invariant type, by the sum over `x` of its first input, which
        // declares the variation of `a` and is the residual of a `linear_call` whose forward region computes that sum,
        // so the `linear_call` runs in lockstep by its rule over all devices. Its transpose region is the same linear
        // function. `a` enters the second input without its variation along `x`, so the `linear_call` accepts it and
        // scales `1` and `2` by `1 + 2`, and the call varies the result along `x`, so that the sum over `x` that
        // follows is `3 + 6`. The same `linear_call` staged directly in the body, at the type of `a`, whose regions
        // vary the sum along `x` to scale `a` by it, computes the same sum.
        let (mesh, a_local_type) = lockstep_call_mesh_and_input_type();
        let invariant_type =
            ArrayIrType::Array(f32_vector_type(1).with_sharding(Sharding::replicated(mesh.clone(), 1)).unwrap());
        let [(call_body, call_program), (direct_body, direct_program)] = lockstep_nested_programs(
            &mesh,
            &a_local_type,
            vec![a_local_type.clone(), invariant_type],
            |builder, inputs, input_types| {
                let forward = lockstep_call_scale_by_sum(&mesh, input_types[0].clone(), input_types[1].clone());
                let transpose = builder.import_program(forward.clone());
                let forward = builder.import_program(forward);
                let linear_call = LinearCallOperation::<ArrayIrType>::new(1);
                builder
                    .add_instruction(linear_call, vec![forward, transpose], inputs.to_vec(), None)
                    .unwrap()
                    .to_vec()
            },
        );
        let varying = &a_local_type;
        let invariant = "f32[1][sharding={mesh<['x'=2:manual]>, [{}]}]";
        let sum = r#"parallel_reduce [kind=sum, axis_name="x", mesh=['x'=2:manual]]"#;
        let vary = r#"parallel_vary [axis_name="x"]"#;
        assert_eq!(
            call_body.to_string(),
            formatdoc! {"
                lambda %0:{varying} .
                let %1:{varying} = call %0 %0 [
                    callee={{
                        lambda %0:{varying}, %1:{invariant} .
                        let %2:{invariant} = linear_call [residual_count=1] %0 %1 [
                            forward={{
                                lambda %0:{varying}, %1:{invariant} .
                                let %2:{invariant} = {sum} %0
                                    %3:{invariant} = mul %1 %2
                                in (%3)
                            }},
                            transpose={{
                                lambda %0:{varying}, %1:{invariant} .
                                let %2:{invariant} = {sum} %0
                                    %3:{invariant} = mul %1 %2
                                in (%3)
                            }},
                        ]
                        in (%2)
                    }},
                ]
                    %2:{invariant} = {sum} %1
                in (%2)"},
        );
        assert_eq!(
            direct_body.to_string(),
            formatdoc! {"
                lambda %0:{varying} .
                let %1:{varying} = linear_call [residual_count=1] %0 %0 [
                    forward={{
                        lambda %0:{varying}, %1:{varying} .
                        let %2:{invariant} = {sum} %0
                            %3:{varying} = {vary} %2
                            %4:{varying} = mul %1 %3
                        in (%4)
                    }},
                    transpose={{
                        lambda %0:{varying}, %1:{varying} .
                        let %2:{invariant} = {sum} %0
                            %3:{varying} = {vary} %2
                            %4:{varying} = mul %1 %3
                        in (%4)
                    }},
                ]
                    %2:{invariant} = {sum} %1
                in (%2)"},
        );
        let output_type = f32_vector_type(1).with_sharding(Sharding::replicated(mesh, 1)).unwrap();
        let input = f32_value(f32_vector_type(2), &[1.0, 2.0]);
        assert_eq!(call_program.interpret(vec![input.clone()]), Ok(vec![f32_value(output_type.clone(), &[9.0])]));
        assert_eq!(direct_program.interpret(vec![input]), Ok(vec![f32_value(output_type, &[9.0])]));
    }

    #[test]
    fn test_call_operation_interpret_in_lockstep_runs_nested_rematerializations_at_callee_input_types() {
        // The callee, traced at a plain input type, doubles its input in a `rematerialize` call whose body declares
        // that type. `a` enters the callee without its variation along `x`, so the `rematerialize` call accepts it
        // and binds once per device, because its body uses no collective, and the sum over `x` that follows is
        // `2 + 4`. The same `rematerialize` call staged directly in the body, at the type of `a`, computes the same
        // sum.
        let (mesh, a_local_type) = lockstep_call_mesh_and_input_type();
        let plain_type = ArrayIrType::Array(f32_vector_type(1));
        let [(call_body, call_program), (direct_body, direct_program)] =
            lockstep_nested_programs(&mesh, &a_local_type, vec![plain_type], |builder, inputs, input_types| {
                let body = builder.import_program(lockstep_call_double(input_types[0].clone()));
                let policy = ResidualPolicyReference::new(NothingSavable);
                let rematerialize = TestOperation::Rematerialize(RematerializeOperation::new(policy));
                builder.add_instruction(rematerialize, vec![body], inputs.to_vec(), None).unwrap().to_vec()
            });
        let varying = &a_local_type;
        let invariant = "f32[1][sharding={mesh<['x'=2:manual]>, [{}]}]";
        let sum = r#"parallel_reduce [kind=sum, axis_name="x", mesh=['x'=2:manual]]"#;
        assert_eq!(
            call_body.to_string(),
            formatdoc! {"
                lambda %0:{varying} .
                let %1:{varying} = call %0 [
                    callee={{
                        lambda %0:f32[1] .
                        let %1:f32[1] = rematerialize %0 [
                            body={{
                                lambda %0:f32[1] .
                                let %1:f32[1] = add %0 %0
                                in (%1)
                            }},
                        ]
                        in (%1)
                    }},
                ]
                    %2:{invariant} = {sum} %1
                in (%2)"},
        );
        assert_eq!(
            direct_body.to_string(),
            formatdoc! {"
                lambda %0:{varying} .
                let %1:{varying} = rematerialize %0 [
                    body={{
                        lambda %0:{varying} .
                        let %1:{varying} = add %0 %0
                        in (%1)
                    }},
                ]
                    %2:{invariant} = {sum} %1
                in (%2)"},
        );
        let output_type = f32_vector_type(1).with_sharding(Sharding::replicated(mesh, 1)).unwrap();
        let input = f32_value(f32_vector_type(2), &[1.0, 2.0]);
        assert_eq!(call_program.interpret(vec![input.clone()]), Ok(vec![f32_value(output_type.clone(), &[6.0])]));
        assert_eq!(direct_program.interpret(vec![input]), Ok(vec![f32_value(output_type, &[6.0])]));
    }

    #[test]
    fn test_call_operation_interpret_in_lockstep_runs_nested_collective_rematerializations_at_callee_input_types() {
        // The callee scales its second input, of an invariant type, by the sum over `x` of its first input, which
        // declares the variation of `a`, in a `rematerialize` call whose body computes that sum, so the call runs in
        // lockstep by its rule over all devices. `a` enters the second input without its variation along `x`, so the
        // `rematerialize` call accepts it and scales `1` and `2` by `1 + 2`, and the outer call varies the result
        // along `x`, so that the sum over `x` that follows is `3 + 6`. The same `rematerialize` call staged directly
        // in the body, at the type of `a`, whose body varies the sum along `x` to scale `a` by it, computes the same
        // sum.
        let (mesh, a_local_type) = lockstep_call_mesh_and_input_type();
        let invariant_type =
            ArrayIrType::Array(f32_vector_type(1).with_sharding(Sharding::replicated(mesh.clone(), 1)).unwrap());
        let [(call_body, call_program), (direct_body, direct_program)] = lockstep_nested_programs(
            &mesh,
            &a_local_type,
            vec![a_local_type.clone(), invariant_type],
            |builder, inputs, input_types| {
                let body = lockstep_call_scale_by_sum(&mesh, input_types[0].clone(), input_types[1].clone());
                let body = builder.import_program(body);
                let policy = ResidualPolicyReference::new(NothingSavable);
                let rematerialize = TestOperation::Rematerialize(RematerializeOperation::new(policy));
                builder.add_instruction(rematerialize, vec![body], inputs.to_vec(), None).unwrap().to_vec()
            },
        );
        let varying = &a_local_type;
        let invariant = "f32[1][sharding={mesh<['x'=2:manual]>, [{}]}]";
        let sum = r#"parallel_reduce [kind=sum, axis_name="x", mesh=['x'=2:manual]]"#;
        let vary = r#"parallel_vary [axis_name="x"]"#;
        assert_eq!(
            call_body.to_string(),
            formatdoc! {"
                lambda %0:{varying} .
                let %1:{varying} = call %0 %0 [
                    callee={{
                        lambda %0:{varying}, %1:{invariant} .
                        let %2:{invariant} = rematerialize %0 %1 [
                            body={{
                                lambda %0:{varying}, %1:{invariant} .
                                let %2:{invariant} = {sum} %0
                                    %3:{invariant} = mul %1 %2
                                in (%3)
                            }},
                        ]
                        in (%2)
                    }},
                ]
                    %2:{invariant} = {sum} %1
                in (%2)"},
        );
        assert_eq!(
            direct_body.to_string(),
            formatdoc! {"
                lambda %0:{varying} .
                let %1:{varying} = rematerialize %0 %0 [
                    body={{
                        lambda %0:{varying}, %1:{varying} .
                        let %2:{invariant} = {sum} %0
                            %3:{varying} = {vary} %2
                            %4:{varying} = mul %1 %3
                        in (%4)
                    }},
                ]
                    %2:{invariant} = {sum} %1
                in (%2)"},
        );
        let output_type = f32_vector_type(1).with_sharding(Sharding::replicated(mesh, 1)).unwrap();
        let input = f32_value(f32_vector_type(2), &[1.0, 2.0]);
        assert_eq!(call_program.interpret(vec![input.clone()]), Ok(vec![f32_value(output_type.clone(), &[9.0])]));
        assert_eq!(direct_program.interpret(vec![input]), Ok(vec![f32_value(output_type, &[9.0])]));
    }

    #[test]
    fn test_call_operation_interpret_in_lockstep_runs_collectives_of_callees_at_invariant_input_types() {
        // The callee, traced at the invariant type `f32[1]` over the mesh of `x`, sums its input over `x`, which the
        // type rule of `parallel_reduce` accepts only after `parallel_vary` makes the input varying. That transition
        // changes no value of any device, so the call of the callee on `a`, which varies along `x` beyond that type,
        // sums the values `1` and `2` of the two devices, as the callee traced at the varying type of `a` does, which
        // needs no transition. Both calls return the sum `3` on every device, typed as varying along `x` (the second
        // callee varies it explicitly), so the sum over `x` that follows is `6` for both.
        let (mesh, a_local_type) = lockstep_call_mesh_and_input_type();
        let invariant_type =
            ArrayIrType::Array(f32_vector_type(1).with_sharding(Sharding::replicated(mesh.clone(), 1)).unwrap());
        let parallel_sum = || {
            ArrayOperation::ParallelReduce(
                ParallelReduceOperation::new(ReductionKind::Sum, "x".to_string()).with_mesh(mesh.clone()),
            )
        };
        let parallel_vary = || ArrayOperation::ParallelVary(ParallelVaryOperation::new("x".to_string()));
        let invariant_callee = build_program(vec![invariant_type], |builder, inputs| {
            let varying = builder.add_instruction(parallel_vary(), Vec::new(), vec![inputs[0]], None).unwrap()[0];
            builder.add_instruction(parallel_sum(), Vec::new(), vec![varying], None).unwrap().to_vec()
        });
        let varying_callee = build_program(vec![a_local_type.clone()], |builder, inputs| {
            let sum = builder.add_instruction(parallel_sum(), Vec::new(), vec![inputs[0]], None).unwrap()[0];
            builder.add_instruction(parallel_vary(), Vec::new(), vec![sum], None).unwrap().to_vec()
        });
        let (invariant_body, invariant_program) = lockstep_call_program(&mesh, &a_local_type, invariant_callee, false);
        let (varying_body, varying_program) = lockstep_call_program(&mesh, &a_local_type, varying_callee, false);
        let varying = &a_local_type;
        let invariant = "f32[1][sharding={mesh<['x'=2:manual]>, [{}]}]";
        let sum = r#"parallel_reduce [kind=sum, axis_name="x", mesh=['x'=2:manual]]"#;
        let vary = r#"parallel_vary [axis_name="x"]"#;
        assert_eq!(
            invariant_body.to_string(),
            formatdoc! {"
                lambda %0:{varying} .
                let %1:{varying} = call %0 [
                    callee={{
                        lambda %0:{invariant} .
                        let %1:{varying} = {vary} %0
                            %2:{invariant} = {sum} %1
                        in (%2)
                    }},
                ]
                    %2:{invariant} = {sum} %1
                in (%2)"},
        );
        assert_eq!(
            varying_body.to_string(),
            formatdoc! {"
                lambda %0:{varying} .
                let %1:{varying} = call %0 [
                    callee={{
                        lambda %0:{varying} .
                        let %1:{invariant} = {sum} %0
                            %2:{varying} = {vary} %1
                        in (%2)
                    }},
                ]
                    %2:{invariant} = {sum} %1
                in (%2)"},
        );
        let output_type = f32_vector_type(1).with_sharding(Sharding::replicated(mesh, 1)).unwrap();
        assert_eq!(
            invariant_program.interpret(vec![f32_value(f32_vector_type(2), &[1.0, 2.0])]),
            Ok(vec![f32_value(output_type.clone(), &[6.0])]),
        );
        assert_eq!(
            varying_program.interpret(vec![f32_value(f32_vector_type(2), &[1.0, 2.0])]),
            Ok(vec![f32_value(output_type, &[6.0])]),
        );
    }

    #[test]
    fn test_call_operation_interpret_in_lockstep_rejects_diverging_mutations_of_replicated_references() {
        // The callee, traced at the invariant types of a reference replicated along `x` and of its referent, adds its
        // second input into the reference. The call accepts `a`, which varies along `x` beyond that type, as that
        // second input, so each device adds its own value into its own copy of the reference, and the copies diverge.
        // The emulation rejects the map when it writes the reference back, and reference discharge, which every
        // backend applies before lowering, rejects it as well, because the call types the final state of the
        // reference as varying along `x` while the replicated reference requires it to be invariant.
        let (mesh, a_local_type) = lockstep_call_mesh_and_input_type();
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated = Sharding::replicated(mesh.clone(), 1);
        let boundary =
            ShardMapBoundary::new(mesh.clone(), vec![replicated, sharded.clone()], vec![sharded], Vec::new()).unwrap();
        let referent_type = boundary.local_input_type(0, &f32_vector_type(1)).unwrap();
        let reference_type = ArrayIrType::Reference(ReferenceType::new(referent_type.clone()));
        let callee =
            build_program(vec![reference_type.clone(), ArrayIrType::Array(referent_type)], |builder, inputs| {
                builder
                    .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![inputs[0], inputs[1]], None)
                    .unwrap();
                builder
                    .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![inputs[0]], None)
                    .unwrap()
                    .to_vec()
            });
        let body = build_program(vec![reference_type, a_local_type.clone()], |builder, inputs| {
            let callee = builder.import_program(callee);
            let call = TestOperation::Call(CallOperation::new(0));
            builder.add_instruction(call, vec![callee], inputs.to_vec(), None).unwrap().to_vec()
        });
        let varying = &a_local_type;
        let invariant = "f32[1][sharding={mesh<['x'=2:manual]>, [{}]}]";
        assert_eq!(
            body.to_string(),
            formatdoc! {"
                lambda %0:ref<{invariant}>, %1:{varying} .
                let %2:{varying} = call %0 %1 [
                    callee={{
                        lambda %0:ref<{invariant}>, %1:{invariant} .
                        let () = reference_add_update %0 %1
                            %2:{invariant} = reference_read %0
                        in (%2)
                    }},
                ]
                in (%2)"},
        );
        let input_types = vec![
            ArrayIrType::Reference(ReferenceType::new(f32_vector_type(1))),
            ArrayIrType::Array(f32_vector_type(2)),
        ];
        let operation = ShardMapOperation::from_program(&body, input_types.clone(), boundary).unwrap();
        let program = build_program(input_types, |builder, inputs| {
            let body = builder.import_program(body);
            builder.add_instruction(operation, vec![body], inputs.to_vec(), None).unwrap().to_vec()
        });
        let initial_state = Array::from_elements(f32_vector_type(1), &[0.0f32]).unwrap();
        let reference = ArrayReference::new(initial_state.clone());
        assert_eq!(
            program
                .interpret(vec![TestValue::Reference(reference.clone()), f32_value(f32_vector_type(2), &[1.0, 2.0])]),
            Err(ProgramError::MalformedProgram(
                "`shard_map` reference input #0 is replicated along manual axis `x`, but its devices along that axis \
                 finished with different states"
                    .to_string(),
            )),
        );
        assert_eq!(reference.read(), Ok(initial_state));
        assert!(matches!(
            program.discharge_references(0),
            Err(ProgramError::MalformedProgram(message)) if message == format!(
                "reference discharge state has type `{varying}` but allocation `ref<{invariant}>` requires \
                 `{invariant}`",
            ),
        ));
    }

    #[test]
    fn test_call_operation_partially_evaluate() {
        // Partial evaluation of a mixed `call` against a live outer trace wraps the known half of the callee again as a
        // known-side `call`, which it stages into the outer program over the symbolic known input. The unknown half
        // stays behind a residual `call`, and the known→unknown residual edge flows from the outputs of the known-side
        // call into the inputs of the residual call.
        let r#type = ArrayIrType::Array(f64_vector_type(4));

        // Callee `f(a, x) = (a + a, x * x, (a + a) * x)` over a known `a` and an unknown `x`. Capture constants are
        // absent, because a capture-bearing callee preserves its boundary instead of splitting, which
        // `test_call_operation_partially_evaluate_preserves_boundary_for_capture_bearing_callees` pins.
        let callee = build_program(vec![r#type.clone(); 2], |builder, inputs| {
            let (known_input, runtime_input) = (inputs[0], inputs[1]);
            let shifted = builder
                .add_instruction(
                    ArrayOperation::from(AddOperation::new()),
                    Vec::new(),
                    vec![known_input, known_input],
                    None,
                )
                .unwrap()[0];
            let scaled = builder
                .add_instruction(
                    ArrayOperation::from(MulOperation::new()),
                    Vec::new(),
                    vec![runtime_input, runtime_input],
                    None,
                )
                .unwrap()[0];
            let product = builder
                .add_instruction(
                    ArrayOperation::from(MulOperation::new()),
                    Vec::new(),
                    vec![shifted, runtime_input],
                    None,
                )
                .unwrap()[0];
            vec![shifted, scaled, product]
        });

        // Enclosing program staging one call to the callee over `[a, x]`.
        let program = build_program(vec![r#type.clone(); 2], |builder, inputs| {
            let callee = builder.intern_callee(&Arc::new(callee), None).unwrap();
            let call = TestOperation::Call(CallOperation::new(0));
            builder.add_instruction(call, vec![callee], inputs.to_vec(), None).unwrap().to_vec()
        });

        let outer = TracingContext::<TestValue, TestOperation>::new();
        let known = outer.input(r#type.clone());
        let evaluation = program
            .partially_evaluate_in_context(&outer, &[PartialValue::Known(known), PartialValue::Unknown(r#type.clone())])
            .unwrap();

        // The known half lands in the outer program as one known-side `call` over the symbolic known input, which
        // produces the fully known callee output once, and the unknown half stays behind one residual `call` over
        // the unknown input and the residual edge fed from that output.
        let known_program = outer
            .builder()
            .borrow()
            .clone()
            .build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder], Vec::new())
            .unwrap();
        assert_eq!(
            known_program.to_string(),
            indoc! {"
                lambda %0:f64[4] .
                let %1:f64[4] = call [derived_input_shardings=[_]] %0 [
                    callee={
                        lambda %0:f64[4] .
                        let %1:f64[4] = add %0 %0
                        in (%1)
                    },
                ]
                in ()"},
        );
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:f64[4], %1:f64[4] .
                let %2:f64[4], %3:f64[4] = call [derived_input_shardings=[_, _]] %0 %1 [
                    callee={
                        lambda %0:f64[4], %1:f64[4] .
                        let %2:f64[4] = mul %0 %0
                            %3:f64[4] = mul %1 %0
                        in (%2, %3)
                    },
                ]
                in (%2, %3)"},
        );

        // In the boundary descriptors, the unknown enclosing input feeds the residual call, the residual edge is a
        // known feeder naming the known-side call's staged output, and the outputs reassemble in original order.
        let known_output = known_program.instructions()[0].outputs()[0];
        assert!(matches!(
            evaluation.inputs(),
            [PartialEvaluationInput::Unknown(1), PartialEvaluationInput::Known(value)]
                if value.atom_id().ok() == Some(known_output),
        ));
        assert!(matches!(
            evaluation.outputs(),
            [
                PartialEvaluationOutput::Known(value),
                PartialEvaluationOutput::Unknown(0),
                PartialEvaluationOutput::Unknown(1),
            ] if value.atom_id().ok() == Some(known_output),
        ));
    }

    #[test]
    fn test_call_operation_partially_evaluate_forwards_known_output_residuals() {
        // When the callee `f(a, x) = (sin(a), sin(a) * x)` is partitioned with `a` known and `x` unknown, the residual
        // call needs `sin(a)`, which is also a known output, so the known call returns it once and the residual call is
        // fed from that output (JAX's `out_fwd`). The outer partition repeats the known call's output as its own
        // residual edge, because `Program::partition` keeps one edge per residual feeder.
        let r#type = ArrayIrType::Array(f64_vector_type(4));
        let callee = build_program(vec![r#type.clone(); 2], |builder, inputs| {
            let sine = builder
                .add_instruction(ArrayOperation::from(SinOperation::new()), Vec::new(), vec![inputs[0]], None)
                .unwrap()[0];
            let product = builder
                .add_instruction(ArrayOperation::from(MulOperation::new()), Vec::new(), vec![sine, inputs[1]], None)
                .unwrap()[0];
            vec![sine, product]
        });
        let program = build_program(vec![r#type; 2], |builder, inputs| {
            let callee = builder.intern_callee(&Arc::new(callee), None).unwrap();
            let call = TestOperation::Call(CallOperation::new(0));
            builder.add_instruction(call, vec![callee], inputs.to_vec(), None).unwrap().to_vec()
        });
        assert_eq!(
            program.partition(&[true, false]).unwrap().to_string(),
            indoc! {"
                partition [
                    known_inputs=[0],
                    residual_inputs=[UnknownInput(1), ResidualEdge(0)],
                    outputs=[Known(0), Unknown(0)],
                ]
                known={
                    lambda %0:f64[4] .
                    let %1:f64[4] = call [derived_input_shardings=[_]] %0 [
                        callee={
                            lambda %0:f64[4] .
                            let %1:f64[4] = sin %0
                            in (%1)
                        },
                    ]
                    in (%1, %1)
                }
                residual={
                    lambda %0:f64[4], %1:f64[4] .
                    let %2:f64[4] = call [derived_input_shardings=[_, _]] %0 %1 [
                        callee={
                            lambda %0:f64[4], %1:f64[4] .
                            let %2:f64[4] = mul %1 %0
                            in (%2)
                        },
                    ]
                    in (%2)
                }"},
        );
    }

    #[test]
    fn test_call_operation_partially_evaluate_splits_concrete_known_inputs() {
        // Partial evaluation splits a mixed `call` whose known inputs are concrete arrays exactly as it splits one
        // whose known inputs are symbolic. The known-side call runs immediately, so the output `a + a` is known and
        // concrete, and the residual call keeps only `(a + a) * x`, fed by that known output (JAX's `out_fwd`).
        let array_type = ArrayType::new_static(DataType::F64, [4]);
        let r#type = ArrayIrType::Array(array_type.clone());

        // Callee `f(a, x) = (a + a, (a + a) * x)` over a known `a` and an unknown `x`.
        let callee = Arc::new(build_program(vec![r#type.clone(); 2], |builder, inputs| {
            let doubled = builder
                .add_instruction(
                    ArrayOperation::from(AddOperation::new()),
                    Vec::new(),
                    vec![inputs[0], inputs[0]],
                    None,
                )
                .unwrap()[0];
            let product = builder
                .add_instruction(ArrayOperation::from(MulOperation::new()), Vec::new(), vec![doubled, inputs[1]], None)
                .unwrap()[0];
            vec![doubled, product]
        }));
        let program = build_program(vec![r#type.clone(); 2], |builder, inputs| {
            let callee = builder.intern_callee(&callee, None).unwrap();
            let call = TestOperation::Call(CallOperation::new(0));
            builder.add_instruction(call, vec![callee], inputs.to_vec(), None).unwrap().to_vec()
        });
        let array = |values: &[f64]| f64_value(array_type.clone(), values);
        let context = EagerContext::<TestValue, TestOperation>::new();
        let evaluation = program
            .partially_evaluate_in_context(
                &context,
                &[PartialValue::Known(array(&[1.0, 2.0, 3.0, 4.0])), PartialValue::Unknown(r#type)],
            )
            .unwrap();
        let [PartialEvaluationInput::Unknown(1), PartialEvaluationInput::Known(residual)] = evaluation.inputs() else {
            panic!("expected the unknown input followed by the forwarded known output");
        };
        let [PartialEvaluationOutput::Known(doubled), PartialEvaluationOutput::Unknown(0)] = evaluation.outputs()
        else {
            panic!("expected one known output followed by one unknown output");
        };
        assert_eq!(doubled, &array(&[2.0, 4.0, 6.0, 8.0]));
        assert_eq!(residual, &array(&[2.0, 4.0, 6.0, 8.0]));
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:f64[4], %1:f64[4] .
                let %2:f64[4] = call [derived_input_shardings=[_, _]] %0 %1 [
                    callee={
                        lambda %0:f64[4], %1:f64[4] .
                        let %2:f64[4] = mul %1 %0
                        in (%2)
                    },
                ]
                in (%2)"},
        );
        assert_eq!(
            evaluation.interpret(&context, &[array(&[5.0, 6.0, 7.0, 8.0])]),
            Ok(vec![array(&[2.0, 4.0, 6.0, 8.0]), array(&[10.0, 24.0, 42.0, 64.0])]),
        );
    }

    #[test]
    fn test_call_operation_partially_evaluate_partitions_a_callee_with_a_dead_allocation() {
        let r#type = ArrayIrType::Array(f64_vector_type(4));
        let callee = build_program(vec![r#type.clone(); 2], |builder, inputs| {
            let (known_input, runtime_input) = (inputs[0], inputs[1]);
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![known_input], None).unwrap();
            let doubled = builder
                .add_instruction(
                    ArrayOperation::from(AddOperation::new()),
                    Vec::new(),
                    vec![known_input, known_input],
                    None,
                )
                .unwrap()[0];
            let product = builder
                .add_instruction(
                    ArrayOperation::from(MulOperation::new()),
                    Vec::new(),
                    vec![known_input, runtime_input],
                    None,
                )
                .unwrap()[0];
            vec![doubled, product]
        });
        let program = build_program(vec![r#type.clone(); 2], |builder, inputs| {
            let callee = builder.intern_callee(&Arc::new(callee), None).unwrap();
            let call = TestOperation::Call(CallOperation::new(0));
            builder.add_instruction(call, vec![callee], inputs.to_vec(), None).unwrap().to_vec()
        });

        // The callee partitions like any other. Its pure prefix over the known input folds into the known side, which
        // is staged into the outer program, while the runtime-dependent product stays in the residual callee, and the
        // allocation, which nothing observes, is a dead intrinsic state operation that neither side retains.
        let outer = TracingContext::<TestValue, TestOperation>::new();
        let known = outer.input(r#type.clone());
        let evaluation = program
            .partially_evaluate_in_context(&outer, &[PartialValue::Known(known), PartialValue::Unknown(r#type)])
            .unwrap();
        let known_program = outer
            .builder()
            .borrow()
            .clone()
            .build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder], Vec::new())
            .unwrap();
        assert_eq!(
            known_program.to_string(),
            indoc! {"
                lambda %0:f64[4] .
                let %1:f64[4] = call [derived_input_shardings=[_]] %0 [
                    callee={
                        lambda %0:f64[4] .
                        let %1:f64[4] = add %0 %0
                        in (%1)
                    },
                ]
                in ()"},
        );
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:f64[4], %1:f64[4] .
                let %2:f64[4] = call [derived_input_shardings=[_, _]] %0 %1 [
                    callee={
                        lambda %0:f64[4], %1:f64[4] .
                        let %2:f64[4] = mul %1 %0
                        in (%2)
                    },
                ]
                in (%2)"},
        );
    }

    #[test]
    fn test_call_operation_partially_evaluate_keeps_reference_bearing_callees_whole() {
        // The callee `f(r, x) = { a = read(r); r += x; a * read(r) }` reads the reference before and after mutating it.
        // Splitting it against a known `r` would hoist the first read into the outer program and leave the mutation
        // and the second read behind the residual call, so the call stays whole instead.
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));
        let callee = build_program(vec![reference_type.clone(), scalar_type.clone()], |builder, inputs| {
            let (reference, update) = (inputs[0], inputs[1]);
            let before =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
            builder
                .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
                .unwrap();
            let after =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
            builder
                .add_instruction(ArrayOperation::from(MulOperation::new()), Vec::new(), vec![before, after], None)
                .unwrap()
                .to_vec()
        });
        let program = build_program(vec![reference_type.clone(), scalar_type.clone()], |builder, inputs| {
            let callee = builder.import_region(callee.entry_region_ref());
            let call = TestOperation::Call(CallOperation::new(0));
            builder.add_instruction(call, vec![callee], inputs.to_vec(), None).unwrap().to_vec()
        });

        let outer = TracingContext::<TestValue, TestOperation>::new();
        let known = outer.input(reference_type);
        let evaluation = program
            .partially_evaluate_in_context(&outer, &[PartialValue::Known(known), PartialValue::Unknown(scalar_type)])
            .unwrap();
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:f32[], %1:ref<f32[]> .
                let %2:f32[] = call %1 %0 [
                    callee={
                        lambda %0:ref<f32[]>, %1:f32[] .
                        let %2:f32[] = reference_read %0
                            () = reference_add_update %0 %1
                            %3:f32[] = reference_read %0
                            %4:f32[] = mul %2 %3
                        in (%4)
                    },
                ]
                in (%2)"},
        );
        assert!(outer.builder().borrow().instructions().is_empty());
    }

    #[test]
    fn test_call_operation_partially_evaluate_keeps_escaping_allocation_callees_whole() {
        // The callee `f(k, x) = (reference_new(k), k * x)` allocates a reference from the known input and returns it
        // beside a runtime-dependent product. The allocation is computable from known inputs alone, but it escapes
        // through a reference output whose later accesses the callee cannot see, so it is live state rather than dead
        // intrinsic state, and the call stays whole instead of hoisting the allocation into the known side.
        let r#type = ArrayIrType::Array(f64_vector_type(4));
        let callee = build_program(vec![r#type.clone(); 2], |builder, inputs| {
            let reference =
                builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![inputs[0]], None).unwrap()[0];
            let product = builder
                .add_instruction(
                    ArrayOperation::from(MulOperation::new()),
                    Vec::new(),
                    vec![inputs[0], inputs[1]],
                    None,
                )
                .unwrap()[0];
            vec![reference, product]
        });
        let program = build_program(vec![r#type.clone(); 2], |builder, inputs| {
            let callee = builder.intern_callee(&Arc::new(callee), None).unwrap();
            let call = TestOperation::Call(CallOperation::new(0));
            builder.add_instruction(call, vec![callee], inputs.to_vec(), None).unwrap().to_vec()
        });
        let outer = TracingContext::<TestValue, TestOperation>::new();
        let known = outer.input(r#type.clone());
        let evaluation = program
            .partially_evaluate_in_context(&outer, &[PartialValue::Known(known), PartialValue::Unknown(r#type)])
            .unwrap();
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:f64[4], %1:f64[4] .
                let %2:ref<f64[4]>, %3:f64[4] = call %1 %0 [
                    callee={
                        lambda %0:f64[4], %1:f64[4] .
                        let %2:ref<f64[4]> = reference_new %0
                            %3:f64[4] = mul %0 %1
                        in (%2, %3)
                    },
                ]
                in (%2, %3)"},
        );
        assert!(outer.builder().borrow().instructions().is_empty());
    }

    #[test]
    fn test_call_operation_partially_evaluate_preserves_boundary_for_capture_bearing_callees() {
        // Partitioning does not remap the absolute indices of retained capture constants, so a mixed call whose
        // callee still holds one must keep its original boundary instead of splitting into derived callees whose
        // compacted or absent capture prefixes those indices would misname.
        /// Operation family of the array operations over captured constants, whose constants name captures.
        #[derive(Clone, Debug, Operation)]
        #[ryft(type(ArrayIrType), constant(TestCapture), members(ArrayType), dispatch(partial_evaluation))]
        enum CaptureOperation {
            #[ryft(projected(ArrayType), partial_evaluation = ArrayOperation::partially_evaluate_member)]
            Array(ArrayOperation<CaptureReference<ArrayType>>),
            Call(CallOperation<ArrayIrType>),
            Guard(GuardOperation<ArrayIrType>),
        }

        let r#type = ArrayIrType::Array(f64_vector_type(4));
        let callee = {
            let mut builder = ProgramBuilder::<TestCapture, CaptureOperation>::new();
            let known_input = builder.add_input(r#type.clone());
            let runtime_input = builder.add_input(r#type.clone());
            let captured = builder.add_constant(CaptureReference::new(0, r#type.clone()));
            let shifted = builder
                .add_instruction(
                    CaptureOperation::Array(ArrayOperation::from(AddOperation::new())),
                    Vec::new(),
                    vec![known_input, captured],
                    None,
                )
                .unwrap()[0];
            let product = builder
                .add_instruction(
                    CaptureOperation::Array(ArrayOperation::from(MulOperation::new())),
                    Vec::new(),
                    vec![shifted, runtime_input],
                    None,
                )
                .unwrap()[0];
            builder
                .build::<Vec<TestCapture>, Vec<TestCapture>>(
                    vec![shifted, product],
                    vec![Placeholder; 2],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        let mut builder = ProgramBuilder::<TestCapture, CaptureOperation>::new();
        let known_input = builder.add_input(r#type.clone());
        let runtime_input = builder.add_input(r#type.clone());
        let callee_region = builder.intern_callee(&Arc::new(callee), None).unwrap();
        let outputs = builder
            .add_instruction(
                CaptureOperation::Call(CallOperation::new(0)),
                vec![callee_region],
                vec![known_input, runtime_input],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<TestCapture>, Vec<TestCapture>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();

        let outer = TracingContext::<TestCapture, CaptureOperation>::new();
        let known = outer.input(r#type.clone());
        let evaluation = program
            .partially_evaluate_in_context(&outer, &[PartialValue::Known(known), PartialValue::Unknown(r#type)])
            .unwrap();

        // No known-side call is hoisted into the outer program, and the residual program retains the original call
        // with its callee intact, including its capture constant.
        assert!(outer.builder().borrow().instructions().is_empty());
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:f64[4], %1:f64[4] .
                let %2:f64[4], %3:f64[4] = call %1 %0 [
                    callee={
                        lambda %0:f64[4], %1:f64[4] .
                        let %2:f64[4] = const capture#0:f64[4]
                            %3:f64[4] = add %0 %2
                            %4:f64[4] = mul %3 %1
                        in (%3, %4)
                    },
                ]
                in (%2, %3)"},
        );
    }

    #[test]
    fn test_call_operation_partially_evaluate_places_callee_residuals_with_the_context_policy() {
        struct SaveDots;

        impl ResidualPolicy<ArrayIrType> for SaveDots {
            type Storage = NoStorage;

            fn name(&self) -> &str {
                "save_dots"
            }

            fn classify(
                &self,
                candidate: &ResidualCandidate<'_, ArrayIrType>,
            ) -> Result<ResidualDecision<NoStorage>, ResidualRejection> {
                Ok(match candidate.producers().iter().any(|producer| producer.payload::<DotOperation>().is_some()) {
                    true => ResidualDecision::Save,
                    false => ResidualDecision::Recompute,
                })
            }
        }

        // The callee computes `cos(dot(x, x)) * t`. Partitioning the caller with `x` known splits the call, and the
        // policy then plans the callee partition, in which the known call returns the dot product instead of its
        // cosine and the residual call recomputes the cosine.
        let callee = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let x = builder.add_input(ArrayIrType::Array(f64_vector_type(4)));
            let t = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::F64)));
            let dot = ArrayOperation::<Array>::from(DotOperation::new(DotDimensionNumbers::new(
                vec![0],
                vec![0],
                vec![],
                vec![],
            )));
            let product = builder.add_instruction(dot, Vec::new(), vec![x, x], None).unwrap()[0];
            let cosine = ArrayOperation::<Array>::from(CosOperation::<ArrayType>::new());
            let cosine = builder.add_instruction(cosine, Vec::new(), vec![product], None).unwrap()[0];
            let tangent = ArrayOperation::<Array>::from(MulOperation::<ArrayType>::new());
            let tangent = builder.add_instruction(tangent, Vec::new(), vec![cosine, t], None).unwrap()[0];
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![tangent], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let x = builder.add_input(ArrayIrType::Array(f64_vector_type(4)));
        let t = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::F64)));
        let callee_region = builder.intern_callee(&Arc::new(callee), None).unwrap();
        let output = builder
            .add_instruction(TestOperation::Call(CallOperation::new(0)), vec![callee_region], vec![x, t], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let partition = program
            .partition_with_residual_policy(&[true, false], &ResidualPolicyReference::new(SaveDots))
            .unwrap();
        assert_eq!(
            partition.to_string(),
            indoc! {"
                partition [
                    known_inputs=[0],
                    residual_inputs=[UnknownInput(1), ResidualEdge(0)],
                    outputs=[Unknown(0)],
                ]
                known={
                    lambda %0:f64[4] .
                    let %1:f64[] = call [derived_input_shardings=[_]] %0 [
                        callee={
                            lambda %0:f64[4] .
                            let %1:f64[] = dot [
                                dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                            ] %0 %0
                            in (%1)
                        },
                    ]
                    in (%1)
                }
                residual={
                    lambda %0:f64[], %1:f64[] .
                    let %2:f64[] = call [derived_input_shardings=[_, _]] %0 %1 [
                        callee={
                            lambda %0:f64[], %1:f64[] .
                            let %2:f64[] = cos %1
                                %3:f64[] = mul %2 %0
                            in (%3)
                        },
                    ]
                    in (%2)
                }"},
        );
    }

    #[test]
    fn test_call_operation_partially_evaluate_retains_splits_of_mixed_calls() {
        // Callee `g(k, x) = x ⊙ sin(k · k) + k` over a known `k` and an unknown `x`, whose known part computes
        // `sin(k · k)` and whose residual part computes the rest. The repeated workload chains `h ↦ g(k, h)` from `x`,
        // and the nested workload is the diamond `g_d(k, x) = g_{d-1}(k, x) + g_{d-1}(k, sin x)` with `g_0 = g`.
        let matrix_type = residual_demand_matrix_type();
        let callee = Arc::new(build_program(vec![matrix_type.clone(); 2], |builder, inputs| {
            let (known, unknown) = (inputs[0], inputs[1]);
            let dot = DotOperation::new(DotDimensionNumbers::new(vec![1], vec![0], vec![], vec![]));
            let product =
                builder.add_instruction(ArrayOperation::from(dot), Vec::new(), vec![known, known], None).unwrap()[0];
            let sine = builder
                .add_instruction(ArrayOperation::from(SinOperation::new()), Vec::new(), vec![product], None)
                .unwrap()[0];
            let scaled = builder
                .add_instruction(ArrayOperation::from(MulOperation::new()), Vec::new(), vec![unknown, sine], None)
                .unwrap()[0];
            builder
                .add_instruction(ArrayOperation::from(AddOperation::new()), Vec::new(), vec![scaled, known], None)
                .unwrap()
                .to_vec()
        }));
        let program = |body: &dyn Fn(&mut ProgramBuilder<TestValue, TestOperation>, AtomId, AtomId) -> AtomId| {
            build_program(vec![matrix_type.clone(); 2], |builder, inputs| vec![body(builder, inputs[0], inputs[1])])
        };
        let repeated = |count: usize| {
            program(&|builder, known, unknown| {
                (0..count).fold(unknown, |hidden, _| add_call(builder, &callee, vec![known, hidden]))
            })
        };
        let nested = |depth: usize| {
            let mut nested_callee = callee.clone();
            for _ in 0..depth {
                nested_callee = Arc::new(build_program(vec![matrix_type.clone(); 2], |builder, inputs| {
                    let (known, unknown) = (inputs[0], inputs[1]);
                    let first = add_call(builder, &nested_callee, vec![known, unknown]);
                    let shifted = builder
                        .add_instruction(ArrayOperation::from(SinOperation::new()), Vec::new(), vec![unknown], None)
                        .unwrap()[0];
                    let second = add_call(builder, &nested_callee, vec![known, shifted]);
                    builder
                        .add_instruction(
                            ArrayOperation::from(AddOperation::new()),
                            Vec::new(),
                            vec![first, second],
                            None,
                        )
                        .unwrap()
                        .to_vec()
                }));
            }
            program(&|builder, known, unknown| add_call(builder, &nested_callee, vec![known, unknown]))
        };
        let closure_sizes = |program: &TestProgram| {
            let partition = program.partition(&[true, false]).unwrap();
            let closure_size = |program: &TestProgram| program.entry_region_ref().instructions_in_closure().count();
            [closure_size(partition.known_program()), closure_size(partition.residual_program())]
        };

        // Every call shares one known callee and one residual callee, so both programs grow by one call per call site.
        // Each nesting level adds its two known calls to the known program and its two residual calls, the shifted
        // input, and the sum to the residual program, rather than copying the callees of the level below for each
        // call, which would make both programs grow exponentially with the nesting depth. The outer partition returns
        // `k` as a residual edge of its own, because `Program::partition` keeps one edge per residual feeder.
        assert_eq!([1, 2, 4, 8].map(|count| closure_sizes(&repeated(count))), [[3, 3], [4, 4], [6, 6], [10, 10]]);
        assert_eq!([1, 2, 3, 4].map(|depth| closure_sizes(&nested(depth))), [[5, 7], [7, 11], [9, 15], [11, 19]]);
        let partition = repeated(2).partition(&[true, false]).unwrap();
        assert_eq!(
            format!("{}\n{}", partition.known_program(), partition.residual_program()),
            indoc! {"
                lambda %0:f32[2, 2] .
                let %1:f32[2, 2] = call [derived_input_shardings=[_]] %0 [
                    callee=^0={
                        lambda %0:f32[2, 2] .
                        let %1:f32[2, 2] = dot [
                            dimensions=(lhs_contracting=[1], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                        ] %0 %0
                            %2:f32[2, 2] = sin %1
                        in (%2)
                    },
                ]
                    %2:f32[2, 2] = call [derived_input_shardings=[_]] %0 [
                        callee=^0,
                    ]
                in (%1, %0, %2)
                lambda %0:f32[2, 2], %1:f32[2, 2], %2:f32[2, 2], %3:f32[2, 2] .
                let %4:f32[2, 2] = call [derived_input_shardings=[_, _, _]] %0 %1 %2 [
                    callee=^0={
                        lambda %0:f32[2, 2], %1:f32[2, 2], %2:f32[2, 2] .
                        let %3:f32[2, 2] = mul %0 %1
                            %4:f32[2, 2] = add %3 %2
                        in (%4)
                    },
                ]
                    %5:f32[2, 2] = call [derived_input_shardings=[_, _, _]] %4 %3 %2 [
                        callee=^0,
                    ]
                in (%5)"},
        );

        // The splits are retained by the region transform cache of the shared callee, so partitioning the same
        // workload again binds the very same known callee instead of partitioning the callee again.
        let first_callee = |program: &TestProgram| {
            let region = program.instructions()[0].regions()[0];
            program.region_ref(region).unwrap().interned_callee().unwrap().clone()
        };
        let other_partition = repeated(4).partition(&[true, false]).unwrap();
        assert!(Arc::ptr_eq(&first_callee(partition.known_program()), &first_callee(other_partition.known_program())));

        // Interpreting the known program and then the residual program reproduces the outputs of the original program.
        let matrix = |elements: &[f32]| f32_value(ArrayType::new_static(DataType::F32, [2, 2]), elements);
        let inputs = vec![matrix(&[0.5, -1.0, 0.25, 2.0]), matrix(&[0.3, 0.1, -0.2, 0.4])];
        for program in [repeated(4), nested(3)] {
            let partition = program.partition(&[true, false]).unwrap();
            let known_inputs = partition.known_input_indices().iter().map(|index| inputs[*index].clone()).collect();
            let known_outputs = partition.known_program().interpret(known_inputs).unwrap();
            let known_output_count = partition.outputs().iter().filter(|output| output.is_known()).count();
            let residual_inputs = partition
                .residual_inputs()
                .iter()
                .map(|source| match *source {
                    ResidualInputSource::UnknownInput(index) | ResidualInputSource::KnownInput(index) => {
                        inputs[index].clone()
                    }
                    ResidualInputSource::KnownOutput(index) => known_outputs[index].clone(),
                    ResidualInputSource::ResidualEdge(edge) => known_outputs[known_output_count + edge].clone(),
                })
                .collect();
            assert_eq!(partition.outputs(), &[PartialEvaluationOutput::Unknown(0)]);
            assert_eq!(partition.residual_program().interpret(residual_inputs), program.interpret(inputs.clone()));
        }
    }

    #[test]
    fn test_call_operation_partially_evaluate_retains_splits_of_effectful_mixed_calls() {
        let callee = effectful_mixed_callee(EffectClass::UnorderedIo);
        let program = |callee: &Arc<TestProgram>, count: usize| {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let scalar_type: ArrayIrType = ArrayType::scalar(DataType::F32).into();
            let known = builder.add_input(scalar_type.clone());
            let unknown = builder.add_input(scalar_type);
            let output = (0..count).fold(unknown, |value, _| add_call(&mut builder, callee, vec![known, value]));
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let statistics = |program: &TestProgram| {
            let partition = program.partition(&[true, false]).unwrap();
            let count = |program: &TestProgram| {
                let region = program.entry_region_ref();
                [region.instructions_in_closure().count(), region.region_ids_in_closure().len()]
            };
            [count(partition.known_program()), count(partition.residual_program())]
        };

        // These counts precede lowering and count each shared callee once. Unordered effects let every known call
        // remain known, so sharing changes representation without changing the number or placement of runtime prints.
        assert_eq!(
            [1, 2, 4, 16].map(|count| statistics(&program(&callee, count))),
            [[[3, 2], [3, 2]], [[4, 2], [4, 2]], [[6, 2], [6, 2]], [[18, 2], [18, 2]]],
        );
        let partition = program(&callee, 1).partition(&[true, false]).unwrap();
        let other = program(&callee, 16).partition(&[true, false]).unwrap();
        let first_callee = |program: &TestProgram| {
            program
                .region_ref(program.instructions()[0].regions()[0])
                .unwrap()
                .interned_callee()
                .unwrap()
                .clone()
        };
        assert_eq!(
            partition.to_string(),
            indoc! {"
                partition [
                    known_inputs=[0],
                    residual_inputs=[UnknownInput(1), ResidualEdge(0)],
                    outputs=[Unknown(0)],
                ]
                known={
                    lambda %0:f32[] .
                    let %1:f32[] = call [derived_input_shardings=[_]] %0 [
                        callee={
                            lambda %0:f32[] .
                            let %1:f32[] = mul %0 %0
                                %2:f32[] = print [label=known, effect_class=unordered_io] %1
                            in (%2)
                        },
                    ]
                    in (%1)
                }
                residual={
                    lambda %0:f32[], %1:f32[] .
                    let %2:f32[] = call [derived_input_shardings=[_, _]] %0 %1 [
                        callee={
                            lambda %0:f32[], %1:f32[] .
                            let %2:f32[] = mul %0 %1
                                %3:f32[] = print [label=residual, effect_class=unordered_io] %2
                            in (%3)
                        },
                    ]
                    in (%2)
                }"},
        );
        assert!(Arc::ptr_eq(&first_callee(partition.known_program()), &first_callee(other.known_program())));
        assert!(Arc::ptr_eq(&first_callee(partition.residual_program()), &first_callee(other.residual_program())));

        let mut nested = callee;
        let mut counts = Vec::new();
        for _ in 0..6 {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let scalar_type: ArrayIrType = ArrayType::scalar(DataType::F32).into();
            let known = builder.add_input(scalar_type.clone());
            let unknown = builder.add_input(scalar_type);
            let first = add_call(&mut builder, &nested, vec![known, unknown]);
            let second = add_call(&mut builder, &nested, vec![known, unknown]);
            let sum = builder
                .add_instruction(ArrayOperation::from(AddOperation::new()), Vec::new(), vec![first, second], None)
                .unwrap()[0];
            nested = Arc::new(
                builder
                    .build::<Vec<TestValue>, Vec<TestValue>>(vec![sum], vec![Placeholder; 2], vec![Placeholder])
                    .unwrap(),
            );
            counts.push(statistics(&program(&nested, 1)));
        }
        assert_eq!(
            counts,
            vec![
                [[5, 3], [6, 3]],
                [[7, 4], [9, 4]],
                [[9, 5], [12, 5]],
                [[11, 6], [15, 6]],
                [[13, 7], [18, 7]],
                [[15, 8], [21, 8]],
            ],
        );
    }

    #[test]
    fn test_call_operation_partially_evaluate_cached_effectful_splits_preserve_order() {
        // With a policy that saves nothing, a prefix print of the unknown input keeps the ordered known print of the
        // callee in the residual program, which binds the known callee that the split cache retained for the call
        // without that prefix.
        let callee = effectful_mixed_callee(EffectClass::OrderedIo);
        let program = |prefix: bool| {
            let scalar_type: ArrayIrType = ArrayType::scalar(DataType::F32).into();
            build_program(vec![scalar_type.clone(), scalar_type], |builder, inputs| {
                if prefix {
                    let print = ArrayOperation::from(PrintOperation::<ArrayType>::new("before"));
                    builder.add_instruction(print, Vec::new(), vec![inputs[1]], None).unwrap();
                }
                vec![add_call(builder, &callee, vec![inputs[0], inputs[1]])]
            })
        };
        let policy = ResidualPolicyReference::new(NothingSavable);
        let warm = program(false).partition_with_residual_policy(&[true, false], &policy).unwrap();
        let partition = program(true).partition_with_residual_policy(&[true, false], &policy).unwrap();
        assert_eq!(
            partition.to_string(),
            indoc! {"
                partition [
                    known_inputs=[0],
                    residual_inputs=[UnknownInput(1), ResidualEdge(0)],
                    outputs=[Unknown(0)],
                ]
                known={
                    lambda %0:f32[] .
                    in (%0)
                }
                residual={
                    lambda %0:f32[], %1:f32[] .
                    let %2:f32[] = print [label=before] %0
                        %3:f32[] = call [derived_input_shardings=[_]] %1 [
                            callee={
                                lambda %0:f32[] .
                                let %1:f32[] = mul %0 %0
                                    %2:f32[] = print [label=known] %1
                                in (%2)
                            },
                        ]
                        %4:f32[] = call [derived_input_shardings=[_, _]] %0 %3 [
                            callee={
                                lambda %0:f32[], %1:f32[] .
                                let %2:f32[] = mul %0 %1
                                    %3:f32[] = print [label=residual] %2
                                in (%3)
                            },
                        ]
                    in (%4)
                }"},
        );
        let known_callee =
            warm.known_program().region_ref(warm.known_program().instructions()[0].regions()[0]).unwrap();
        let cached_callee = partition
            .residual_program()
            .region_ref(partition.residual_program().instructions()[1].regions()[0])
            .unwrap();
        assert!(Arc::ptr_eq(known_callee.interned_callee().unwrap(), cached_callee.interned_callee().unwrap()));
    }

    #[test]
    fn test_call_operation_partially_evaluate_cached_effectful_splits_keep_outputless_residuals() {
        // The callee returns the square of its known input and prints its unknown input, so the residual program keeps
        // that print in a residual call without outputs, also when the split comes from the cache.
        let scalar_type: ArrayIrType = ArrayType::scalar(DataType::F32).into();
        let callee = Arc::new(build_program(vec![scalar_type.clone(); 2], |builder, inputs| {
            let squared = builder
                .add_instruction(
                    ArrayOperation::from(MulOperation::new()),
                    Vec::new(),
                    vec![inputs[0], inputs[0]],
                    None,
                )
                .unwrap()[0];
            let print = ArrayOperation::from(PrintOperation::<ArrayType>::new("residual"));
            builder.add_instruction(print, Vec::new(), vec![inputs[1]], None).unwrap();
            vec![squared]
        }));
        let program =
            build_program(vec![scalar_type; 2], |builder, inputs| vec![add_call(builder, &callee, inputs.to_vec())]);
        let warm = program.partition(&[true, false]).unwrap();
        let partition = program.partition(&[true, false]).unwrap();
        assert_eq!(partition.outputs(), &[PartialEvaluationOutput::Known(0)]);
        assert_eq!(
            partition.residual_program().to_string(),
            indoc! {"
                lambda %0:f32[] .
                let () = call [derived_input_shardings=[_]] %0 [
                    callee={
                        lambda %0:f32[] .
                        let %1:f32[] = print [label=residual] %0
                        in ()
                    },
                ]
                in ()"},
        );
        let scalar = |value: f32| f32_value(ArrayType::scalar(DataType::F32), &[value]);
        assert_eq!(partition.known_program().interpret(vec![scalar(2.0)]), Ok(vec![scalar(4.0)]));
        for unknown in [3.0, 5.0] {
            assert_eq!(partition.residual_program().interpret(vec![scalar(unknown)]), Ok(Vec::new()));
        }
        assert_eq!(warm.residual_program().to_string(), partition.residual_program().to_string());
    }

    #[test]
    fn test_call_operation_partially_evaluate_keeps_forced_copies_under_specialization() {
        // Specializing a program never folds a copy that every execution must make (i.e., a `place` with `may_alias`
        // disabled), so an all-known call whose callee makes one stays residual, and two executions of the specialized
        // program return arrays with distinct storage. An all-known call without such a copy still folds.
        let array_type = ArrayType::new_static(DataType::F64, [4]);
        let r#type = ArrayIrType::Array(array_type.clone());
        let array = |values: &[f64]| f64_value(array_type.clone(), values);
        let storage = |value: &TestValue| match value {
            ArrayIrValue::Array(array) => array.shared_storage_bytes().clone(),
            _ => panic!("expected an array value"),
        };

        // Program `(a, x) -> call(a, x)` whose callee returns `copy(a)` or `-a`.
        let program = |copy: bool| {
            let callee = Arc::new(build_program(vec![r#type.clone(); 2], |builder, inputs| {
                let operation = if copy {
                    ArrayOperation::from(PlaceOperation::new(None, None).with_may_alias(false))
                } else {
                    ArrayOperation::from(NegOperation::new())
                };
                builder.add_instruction(operation, Vec::new(), vec![inputs[0]], None).unwrap().to_vec()
            }));
            build_program(vec![r#type.clone(); 2], |builder, inputs| {
                let callee = builder.intern_callee(&callee, None).unwrap();
                let call = TestOperation::Call(CallOperation::new(0));
                builder.add_instruction(call, vec![callee], inputs.to_vec(), None).unwrap().to_vec()
            })
        };
        let context = EagerContext::<TestValue, TestOperation>::new();
        let known = array(&[1.0, 2.0, 3.0, 4.0]);

        // An all-known call whose callee forces a copy stays residual as a whole.
        let evaluation = program(true)
            .partially_evaluate_in_context(
                &context,
                &[PartialValue::Known(known.clone()), PartialValue::Known(known.clone())],
            )
            .unwrap();
        assert_eq!(evaluation.outputs(), &[PartialEvaluationOutput::Unknown(0)]);
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:f64[4], %1:f64[4] .
                let %2:f64[4] = call %0 %1 [
                    callee={
                        lambda %0:f64[4], %1:f64[4] .
                        let %2:f64[4] = place [may_alias=false] %0
                        in (%2)
                    },
                ]
                in (%2)"},
        );
        let first = evaluation.interpret(&context, &[]).unwrap();
        let second = evaluation.interpret(&context, &[]).unwrap();
        assert_eq!(first, vec![known.clone()]);
        assert_eq!(second, vec![known.clone()]);
        assert!(!Arc::ptr_eq(&storage(&first[0]), &storage(&second[0])));
        assert!(!Arc::ptr_eq(&storage(&first[0]), &storage(&known)));

        // An all-known call without a forced copy folds.
        let evaluation = program(false)
            .partially_evaluate_in_context(&context, &[PartialValue::Known(known.clone()), PartialValue::Known(known)])
            .unwrap();
        assert_eq!(evaluation.outputs(), &[PartialEvaluationOutput::Known(array(&[-1.0, -2.0, -3.0, -4.0]))]);
    }

    #[test]
    fn test_call_operation_partially_evaluate_does_not_reuse_splits_that_fold_forced_copies() {
        // The split of a mixed call is cached per callee and partial evaluation configuration, so a split that a
        // partition which folds copies computed is not reused by a specialization, which must not fold them.
        let array_type = ArrayType::new_static(DataType::F64, [4]);
        let r#type = ArrayIrType::Array(array_type.clone());
        let array = |values: &[f64]| f64_value(array_type.clone(), values);
        let storage = |value: &TestValue| match value {
            ArrayIrValue::Array(array) => array.shared_storage_bytes().clone(),
            _ => panic!("expected an array value"),
        };

        // Program `(a, x) -> call(a, x)` whose callee returns `(copy(a), -x)`.
        let callee = Arc::new(build_program(vec![r#type.clone(); 2], |builder, inputs| {
            let place = ArrayOperation::from(PlaceOperation::new(None, None).with_may_alias(false));
            let copy = builder.add_instruction(place, Vec::new(), vec![inputs[0]], None).unwrap()[0];
            let negated = builder
                .add_instruction(ArrayOperation::from(NegOperation::new()), Vec::new(), vec![inputs[1]], None)
                .unwrap()[0];
            vec![copy, negated]
        }));
        let mixed = build_program(vec![r#type.clone(); 2], |builder, inputs| {
            let callee = builder.intern_callee(&callee, None).unwrap();
            let call = TestOperation::Call(CallOperation::new(0));
            builder.add_instruction(call, vec![callee], inputs.to_vec(), None).unwrap().to_vec()
        });
        let context = EagerContext::<TestValue, TestOperation>::new();
        let known = array(&[1.0, 2.0, 3.0, 4.0]);
        let unknown = array(&[5.0, 6.0, 7.0, 8.0]);

        // A mixed call whose split was cached by a partition that folds copies does not reuse that split.
        mixed.partition(&[true, false]).unwrap();
        let evaluation = mixed
            .partially_evaluate_in_context(
                &context,
                &[PartialValue::Known(known.clone()), PartialValue::Unknown(r#type)],
            )
            .unwrap();
        assert_eq!(evaluation.outputs(), &[PartialEvaluationOutput::Unknown(0), PartialEvaluationOutput::Unknown(1)]);
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:f64[4], %1:f64[4] .
                let %2:f64[4], %3:f64[4] = call %1 %0 [
                    callee={
                        lambda %0:f64[4], %1:f64[4] .
                        let %2:f64[4] = place [may_alias=false] %0
                            %3:f64[4] = neg %1
                        in (%2, %3)
                    },
                ]
                in (%2, %3)"},
        );
        let first = evaluation.interpret(&context, std::slice::from_ref(&unknown)).unwrap();
        let second = evaluation.interpret(&context, std::slice::from_ref(&unknown)).unwrap();
        assert_eq!(first, vec![known.clone(), array(&[-5.0, -6.0, -7.0, -8.0])]);
        assert!(!Arc::ptr_eq(&storage(&first[0]), &storage(&second[0])));
        assert!(!Arc::ptr_eq(&storage(&first[0]), &storage(&known)));
    }

    #[test]
    fn test_call_operation_partially_evaluate_preserves_guarded_known_callees() {
        let policy = ResidualPolicyReference::new(NothingSavable);
        let check_boundary = |callee: TestProgram, capture_count: usize| {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let inputs = callee
                .input_types()
                .into_owned()
                .into_iter()
                .map(|r#type| builder.add_input(r#type))
                .collect::<Vec<_>>();
            let input_count = inputs.len();
            let callee = builder.import_program(callee);
            let outputs = builder
                .add_instruction(TestOperation::Call(CallOperation::new(capture_count)), vec![callee], inputs, None)
                .unwrap()
                .to_vec();
            let output_count = outputs.len();
            let program = builder
                .build::<Vec<TestValue>, Vec<TestValue>>(
                    outputs,
                    vec![Placeholder; input_count],
                    vec![Placeholder; output_count],
                )
                .unwrap();
            let partition = program.partition_with_residual_policy(&vec![true; input_count], &policy).unwrap();
            assert_eq!(partition.known_program().to_string(), program.to_string());
            assert_eq!(
                partition.residual_program().to_string(),
                indoc! {"
                    lambda  .
                    in ()"},
            );
        };

        // Each callee returns an array input whose type is replicated over an explicit mesh, which is the only public
        // array argument of its call, so forwarding would replace that output by the input if the guard did not keep
        // the boundary of the call.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();

        // Nominal dimensions remain scoped to the call, even when the callee is otherwise pure.
        let dimension = DimensionVariable::new("length", DimensionBounds::new(1, Some(8)).unwrap());
        let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(dimension)]))
            .with_sharding(Sharding::replicated(mesh.clone(), 1))
            .unwrap();
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let input = builder.add_input(dynamic_type.into());
        let output = builder
            .add_instruction(ArrayOperation::from(SinOperation::new()), Vec::new(), vec![input], None)
            .unwrap()[0];
        check_boundary(
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![output, input], vec![Placeholder], vec![Placeholder; 2])
                .unwrap(),
            0,
        );

        // An ordered assertion must keep the same owning call and execution order.
        let predicate_type = ArrayIrType::from(
            ArrayType::scalar(DataType::Boolean).with_sharding(Sharding::replicated(mesh.clone(), 0)).unwrap(),
        );
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let predicate = builder.add_input(predicate_type);
        builder
            .add_instruction(
                ArrayOperation::from(AssertOperation::<ArrayType>::new("required predicate")),
                Vec::new(),
                vec![predicate],
                None,
            )
            .unwrap();
        check_boundary(
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![predicate], vec![Placeholder], vec![Placeholder])
                .unwrap(),
            0,
        );

        // Reference handles keep their existing boundary semantics too.
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = builder.add_input(ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32))));
        let value = builder
            .add_input(ArrayType::scalar(DataType::F32).with_sharding(Sharding::replicated(mesh, 0)).unwrap().into());
        check_boundary(
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(
                    vec![reference, value],
                    vec![Placeholder; 2],
                    vec![Placeholder; 2],
                )
                .unwrap(),
            0,
        );

        // A known value without a staged parent atom must not trigger per-operation replay. The same guard keeps
        // concrete eager calls whole. Using a literal in a live trace makes the retained call directly observable.
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let callee = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let outer = TracingContext::<TestValue, TestOperation>::new();
        let context = PartialEvaluationContext::new(outer.clone()).with_residual_policy(&policy);
        let literal = Array::from_elements(ArrayType::scalar(DataType::Boolean), &[true]).unwrap();
        let input = context.lift(ArrayIrValue::Array(literal)).unwrap();
        context.bind(TestOperation::Call(CallOperation::new(0)), vec![callee], &[input]).unwrap();
        let known_program = outer
            .builder()
            .borrow()
            .clone()
            .build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), Vec::new(), Vec::new())
            .unwrap();
        assert_eq!(
            known_program.to_string(),
            indoc! {"
                lambda  .
                let %0:bool[] = const true
                    %1:bool[] = call %0 [
                        callee={
                            lambda %0:bool[] .
                            in (%0)
                        },
                    ]
                in ()"},
        );
    }

    #[test]
    fn test_call_operation_partially_evaluate_forwards_unchanged_inputs() {
        // The callee `f(a) = (a, sin(a))` returns its only input, whose type is replicated over an explicit mesh, and
        // its call places nothing. The call therefore returns `a` unchanged whatever its argument is, so partial
        // evaluation under a residual policy resolves the first output of an all-known call to the call input, and the
        // known call no longer returns it.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let r#type: ArrayIrType = ArrayType::new_static(DataType::F32, [4])
            .with_sharding(Sharding::replicated(mesh, 1))
            .unwrap()
            .into();
        let callee = Arc::new(build_program(vec![r#type.clone()], |builder, inputs| {
            let sine = builder
                .add_instruction(ArrayOperation::from(SinOperation::new()), Vec::new(), vec![inputs[0]], None)
                .unwrap()[0];
            vec![inputs[0], sine]
        }));
        let program = build_program(vec![r#type], |builder, inputs| {
            let callee = builder.intern_callee(&callee, None).unwrap();
            let call = TestOperation::Call(CallOperation::new(0));
            builder.add_instruction(call, vec![callee], inputs.to_vec(), None).unwrap().to_vec()
        });
        let policy = ResidualPolicyReference::new(NothingSavable);
        assert_eq!(
            program.partition_with_residual_policy(&[true], &policy).unwrap().to_string(),
            indoc! {"
                partition [known_inputs=[0], residual_inputs=[], outputs=[Known(0), Known(1)]]
                known={
                    lambda %0:f32[4][sharding={mesh<['x'=2:explicit]>, [{}]}] .
                    let %1:f32[4][sharding={mesh<['x'=2:explicit]>, [{}]}] = call %0 [
                        callee={
                            lambda %0:f32[4][sharding={mesh<['x'=2:explicit]>, [{}]}] .
                            let %1:f32[4][sharding={mesh<['x'=2:explicit]>, [{}]}] = sin %0
                            in (%1)
                        },
                    ]
                    in (%0, %1)
                }
                residual={
                    lambda  .
                    in ()
                }"},
        );
    }

    #[test]
    fn test_call_operation_partially_evaluate_does_not_forward_inputs_across_call_boundaries() {
        // The callee `f(a, x) = (a, sin(a) * x)` returns its first input, which the boundary of the compiled function
        // that it comes from replicates on its own devices. Logical types do not record those devices, so neither an
        // all-known call of it nor the known call of its split may be replaced by its input under a residual policy.
        // The inputs are typed over an explicit mesh, so a known call whose only argument is `a` would otherwise be
        // forwarded, which is the known call of the split under a policy that saves `sin(a)`. The all-known call also
        // has `x` as a second public argument, and under a policy that saves nothing the known callee of the split is
        // empty, so the whole call stays residual.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let device_mesh = DeviceMesh::new(mesh.clone(), vec![Device::new(0, 0), Device::new(1, 0)]).unwrap();
        let replicated = Sharding::replicated(mesh, 1);
        let r#type: ArrayIrType =
            ArrayType::new_static(DataType::F32, [4]).with_sharding(replicated.clone()).unwrap().into();
        let callee = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let a = builder.add_input(r#type.clone());
            let x = builder.add_input(r#type.clone());
            let sine = builder
                .add_instruction(ArrayOperation::from(SinOperation::new()), Vec::new(), vec![a], None)
                .unwrap()[0];
            let product = builder
                .add_instruction(ArrayOperation::from(MulOperation::new()), Vec::new(), vec![sine, x], None)
                .unwrap()[0];
            Arc::new(
                builder
                    .build::<Vec<TestValue>, Vec<TestValue>>(
                        vec![a, product],
                        vec![Placeholder; 2],
                        vec![Placeholder; 2],
                    )
                    .unwrap(),
            )
        };
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let a = builder.add_input(r#type.clone());
        let x = builder.add_input(r#type.clone());
        let callee_region = builder.intern_callee(&callee, None).unwrap();
        let call = CallOperation::new(0).with_boundary(Some(&device_mesh), Vec::new(), vec![Some(replicated), None]);
        let outputs = builder
            .add_instruction(TestOperation::Call(call), vec![callee_region], vec![a, x], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        let policy = ResidualPolicyReference::new(NothingSavable);
        assert_eq!(
            program.partition_with_residual_policy(&[true, true], &policy).unwrap().to_string(),
            indoc! {"
                partition [known_inputs=[0, 1], residual_inputs=[], outputs=[Known(0), Known(1)]]
                known={
                    lambda %0:f32[4][sharding={mesh<['x'=2:explicit]>, [{}]}], %1:f32[4][sharding={mesh<['x'=2:explicit]>, [{}]}] .
                    let %2:f32[4][sharding={mesh<['x'=2:explicit]>, [{}]}], %3:f32[4][sharding={mesh<['x'=2:explicit]>, [{}]}] = call [output_shardings=[{mesh<['x'=2:explicit]>, [{}]}, _], devices=[0, 1]] %0 %1 [
                        callee={
                            lambda %0:f32[4][sharding={mesh<['x'=2:explicit]>, [{}]}], %1:f32[4][sharding={mesh<['x'=2:explicit]>, [{}]}] .
                            let %2:f32[4][sharding={mesh<['x'=2:explicit]>, [{}]}] = sin %0
                                %3:f32[4][sharding={mesh<['x'=2:explicit]>, [{}]}] = mul %2 %1
                            in (%0, %3)
                        },
                    ]
                    in (%2, %3)
                }
                residual={
                    lambda  .
                    in ()
                }"},
        );
        assert_eq!(
            program.partition_with_residual_policy(&[true, false], &policy).unwrap().to_string(),
            indoc! {"
                partition [
                    known_inputs=[0],
                    residual_inputs=[UnknownInput(1), ResidualEdge(0)],
                    outputs=[Unknown(0), Unknown(1)],
                ]
                known={
                    lambda %0:f32[4][sharding={mesh<['x'=2:explicit]>, [{}]}] .
                    in (%0)
                }
                residual={
                    lambda %0:f32[4][sharding={mesh<['x'=2:explicit]>, [{}]}], %1:f32[4][sharding={mesh<['x'=2:explicit]>, [{}]}] .
                    let %2:f32[4][sharding={mesh<['x'=2:explicit]>, [{}]}], %3:f32[4][sharding={mesh<['x'=2:explicit]>, [{}]}] = call [output_shardings=[{mesh<['x'=2:explicit]>, [{}]}, _], devices=[0, 1]] %1 %0 [
                        callee={
                            lambda %0:f32[4][sharding={mesh<['x'=2:explicit]>, [{}]}], %1:f32[4][sharding={mesh<['x'=2:explicit]>, [{}]}] .
                            let %2:f32[4][sharding={mesh<['x'=2:explicit]>, [{}]}] = sin %0
                                %3:f32[4][sharding={mesh<['x'=2:explicit]>, [{}]}] = mul %2 %1
                            in (%0, %3)
                        },
                    ]
                    in (%2, %3)
                }"},
        );
        let policy = ResidualPolicyReference::new(EverythingSavable);
        assert_eq!(
            program.partition_with_residual_policy(&[true, false], &policy).unwrap().to_string(),
            indoc! {"
                partition [
                    known_inputs=[0],
                    residual_inputs=[UnknownInput(1), ResidualEdge(0)],
                    outputs=[Known(0), Unknown(0)],
                ]
                known={
                    lambda %0:f32[4][sharding={mesh<['x'=2:explicit]>, [{}]}] .
                    let %1:f32[4][sharding={mesh<['x'=2:explicit]>, [{}]}], %2:f32[4][sharding={mesh<['x'=2:explicit]>, [{}]}] = call [
                        output_shardings=[{mesh<['x'=2:explicit]>, [{}]}, _],
                        derived_input_shardings=[{mesh<['x'=2:explicit]>, [{}]}],
                        devices=[0, 1],
                    ] %0 [
                        callee={
                            lambda %0:f32[4][sharding={mesh<['x'=2:explicit]>, [{}]}] .
                            let %1:f32[4][sharding={mesh<['x'=2:explicit]>, [{}]}] = sin %0
                            in (%0, %1)
                        },
                    ]
                    in (%1, %2)
                }
                residual={
                    lambda %0:f32[4][sharding={mesh<['x'=2:explicit]>, [{}]}], %1:f32[4][sharding={mesh<['x'=2:explicit]>, [{}]}] .
                    let %2:f32[4][sharding={mesh<['x'=2:explicit]>, [{}]}] = call [derived_input_shardings=[{mesh<['x'=2:explicit]>, [{}]}, _], devices=[0, 1]] %0 %1 [
                        callee={
                            lambda %0:f32[4][sharding={mesh<['x'=2:explicit]>, [{}]}], %1:f32[4][sharding={mesh<['x'=2:explicit]>, [{}]}] .
                            let %2:f32[4][sharding={mesh<['x'=2:explicit]>, [{}]}] = mul %1 %0
                            in (%2)
                        },
                    ]
                    in (%2)
                }"},
        );
    }

    #[test]
    fn test_call_operation_partially_evaluate_keeps_calls_with_several_public_arguments_under_a_staged_known_side() {
        // The callee `g(w, b) = (w, w ⊙ b)` returns its first argument and has two public array arguments, and its call
        // is one that a transform derived, whose boundary places neither argument (e.g., the primal call of a
        // linearized callee). Partitioning a caller with both inputs known stages the known side, and partial
        // evaluation keeps `w` as an output of the call instead of forwarding the argument, because a staged known
        // program may later be interpreted eagerly, where a committed `b` selects the devices of the call, moves `w`
        // there, and commits it (refer to `CallOperation::resolve_forwarded_outputs`). The known program therefore
        // returns the output of the call, which a residual planner then plans as a value of its own. Both arguments
        // are replicated over an explicit mesh, so that their number alone keeps `w` from being forwarded.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let matrix_type: ArrayIrType = ArrayType::new_static(DataType::F32, [2, 2])
            .with_sharding(Sharding::replicated(mesh, 2))
            .unwrap()
            .into();
        let callee = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let left = builder.add_input(matrix_type.clone());
            let right = builder.add_input(matrix_type.clone());
            let output = builder
                .add_instruction(ArrayOperation::from(MulOperation::new()), Vec::new(), vec![left, right], None)
                .unwrap()[0];
            Arc::new(
                builder
                    .build::<Vec<TestValue>, Vec<TestValue>>(
                        vec![left, output],
                        vec![Placeholder; 2],
                        vec![Placeholder; 2],
                    )
                    .unwrap(),
            )
        };
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let w = builder.add_input(matrix_type.clone());
        let b = builder.add_input(matrix_type);
        let callee_region = builder.intern_callee(&callee, None).unwrap();
        let derived = CallOperation::<ArrayIrType>::new(0).with_derived_input_shardings(vec![None, None]);
        let outputs = builder
            .add_instruction(TestOperation::Call(derived), vec![callee_region], vec![w, b], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        let partition = program
            .partition_with_residual_policy(&[true, true], &ResidualPolicyReference::new(NothingSavable))
            .unwrap();
        assert_eq!(
            partition.to_string(),
            indoc! {"
                partition [known_inputs=[0, 1], residual_inputs=[], outputs=[Known(0), Known(1)]]
                known={
                    lambda %0:f32[2, 2][sharding={mesh<['x'=2:explicit]>, [{}, {}]}], %1:f32[2, 2][sharding={mesh<['x'=2:explicit]>, [{}, {}]}] .
                    let %2:f32[2, 2][sharding={mesh<['x'=2:explicit]>, [{}, {}]}], %3:f32[2, 2][sharding={mesh<['x'=2:explicit]>, [{}, {}]}] = call [derived_input_shardings=[_, _]] %0 %1 [
                        callee={
                            lambda %0:f32[2, 2][sharding={mesh<['x'=2:explicit]>, [{}, {}]}], %1:f32[2, 2][sharding={mesh<['x'=2:explicit]>, [{}, {}]}] .
                            let %2:f32[2, 2][sharding={mesh<['x'=2:explicit]>, [{}, {}]}] = mul %0 %1
                            in (%0, %2)
                        },
                    ]
                    in (%2, %3)
                }
                residual={
                    lambda  .
                    in ()
                }"},
        );
    }

    #[test]
    fn test_call_operation_split_on_residual_demand() {
        let callee = residual_demand_leaf_callee();
        let repeated = |count: usize, policy: Option<ResidualPolicyReference<ArrayIrType>>| {
            residual_demand_program(policy, |builder, weights, bias| {
                (0..count).fold(weights, |hidden, _| add_call(builder, &callee, vec![hidden, bias]))
            })
        };

        // Every call is split into a known call that also returns its matrix product and a replay call that recomputes
        // its output from that product. All calls share one known callee and one replay callee, so the residual count
        // and the programs grow by one per call (i.e., the saved product and the call itself) rather than by the size
        // of the inlined callee, which would make them grow by five and four instructions per call.
        let policy = ResidualPolicyReference::new(DotsSavable);
        assert_eq!(
            [1, 2, 4, 8].map(|count| residual_demand_statistics(&repeated(count, Some(policy.clone())), &[0])),
            [[2, 7, 7], [3, 8, 8], [5, 10, 10], [9, 14, 14]],
        );
        let linearization = repeated(2, Some(policy.clone())).linearize_with_respect_to(&[0]).unwrap();
        assert_eq!(
            format!("{}\n{}", linearization.primal(), linearization.tangent()),
            indoc! {"
                lambda %0:f32[2, 2], %1:f32[2, 2], %2:f32[2, 2] .
                let %3:f32[2, 2], %4:f32[2, 2] = call [derived_input_shardings=[_, _]] %1 %2 [
                    callee=^0={
                        lambda %0:f32[2, 2], %1:f32[2, 2] .
                        let %2:f32[2, 2] = dot [
                            dimensions=(lhs_contracting=[1], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                        ] %0 %1
                            %3:f32[2, 2] = sin %2
                            %4:f32[2, 2] = cos %2
                            %5:f32[2, 2] = mul %3 %4
                            %6:f32[2, 2] = add %5 %0
                        in (%6, %2)
                    },
                ]
                    %5:f32[2, 2], %6:f32[2, 2] = call [derived_input_shardings=[_, _]] %3 %2 [
                        callee=^0,
                    ]
                    %7:f32[2, 2] = mul %0 %5
                in (%7, %1, %4, %6)
                lambda %0:f32[2, 2], %1:f32[2, 2], %2:f32[2, 2], %3:f32[2, 2] .
                let %4:f32[2, 2] = rematerialize [policy=\"dots_savable\", differentiated=true] %0 %1 %2 %3 [
                    body={
                        lambda %0:f32[2, 2], %1:f32[2, 2], %2:f32[2, 2], %3:f32[2, 2] .
                        let %4:f32[2, 2] = call [derived_input_shardings=[_, _]] %1 %2 [
                            callee=^0={
                                lambda %0:f32[2, 2], %1:f32[2, 2] .
                                let %2:f32[2, 2] = sin %1
                                    %3:f32[2, 2] = cos %1
                                    %4:f32[2, 2] = mul %2 %3
                                    %5:f32[2, 2] = add %4 %0
                                in (%5)
                            },
                        ]
                            %5:f32[2, 2] = call [derived_input_shardings=[_, _]] %4 %3 [
                                callee=^0,
                            ]
                            %6:f32[2, 2] = mul %5 %0
                        in (%6)
                    },
                ]
                in (%4)"},
        );

        // Rematerialization changes neither the primal value nor the derivatives.
        assert_residual_demand_linearizations_match(&repeated(4, Some(policy)), &repeated(4, None), &[0], None);
    }

    #[test]
    fn test_call_operation_split_on_residual_demand_keeps_output_overrides_of_replayed_outputs() {
        // The replay call recomputes the demanded output of the call, so it keeps the sharding override of that output,
        // as the known call does for the output that it still returns.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let device_mesh = DeviceMesh::new(mesh.clone(), vec![Device::new(0, 0), Device::new(1, 0)]).unwrap();
        let output_sharding = Sharding::replicated(mesh, 2);
        let callee = residual_demand_leaf_callee();
        let policy = ResidualPolicyReference::new(DotsSavable);
        let program = residual_demand_program(Some(policy), |builder, weights, bias| {
            let callee = builder.intern_callee(&callee, None).unwrap();
            let call = CallOperation::new(0).with_boundary(
                Some(&device_mesh),
                Vec::new(),
                vec![Some(output_sharding.clone())],
            );
            builder.add_instruction(TestOperation::Call(call), vec![callee], vec![weights, bias], None).unwrap()[0]
        });
        let linearization = program.linearize_with_respect_to(&[0]).unwrap();
        assert_eq!(
            format!("{}\n{}", linearization.primal(), linearization.tangent()),
            indoc! {"
                lambda %0:f32[2, 2], %1:f32[2, 2], %2:f32[2, 2] .
                let %3:f32[2, 2], %4:f32[2, 2] = call [
                    output_shardings=[{mesh<['x'=2:auto]>, [{}, {}]}, _],
                    derived_input_shardings=[{mesh<['x'=2:auto]>, [{}, {}]}, {mesh<['x'=2:auto]>, [{}, {}]}],
                    devices=[0, 1],
                ] %1 %2 [
                    callee={
                        lambda %0:f32[2, 2], %1:f32[2, 2] .
                        let %2:f32[2, 2] = dot [
                            dimensions=(lhs_contracting=[1], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                        ] %0 %1
                            %3:f32[2, 2] = sin %2
                            %4:f32[2, 2] = cos %2
                            %5:f32[2, 2] = mul %3 %4
                            %6:f32[2, 2] = add %5 %0
                        in (%6, %2)
                    },
                ]
                    %5:f32[2, 2] = mul %0 %3
                in (%5, %1, %4)
                lambda %0:f32[2, 2], %1:f32[2, 2], %2:f32[2, 2] .
                let %3:f32[2, 2] = rematerialize [policy=\"dots_savable\", differentiated=true] %0 %1 %2 [
                    body={
                        lambda %0:f32[2, 2], %1:f32[2, 2], %2:f32[2, 2] .
                        let %3:f32[2, 2] = call [
                            output_shardings=[{mesh<['x'=2:auto]>, [{}, {}]}],
                            derived_input_shardings=[{mesh<['x'=2:auto]>, [{}, {}]}, _],
                            devices=[0, 1],
                        ] %1 %2 [
                            callee={
                                lambda %0:f32[2, 2], %1:f32[2, 2] .
                                let %2:f32[2, 2] = sin %1
                                    %3:f32[2, 2] = cos %1
                                    %4:f32[2, 2] = mul %2 %3
                                    %5:f32[2, 2] = add %4 %0
                                in (%5)
                            },
                        ]
                            %4:f32[2, 2] = mul %3 %0
                        in (%4)
                    },
                ]
                in (%3)"},
        );
    }

    #[test]
    fn test_call_operation_split_on_residual_demand_splits_nested_all_known_calls() {
        let nested = |depth: usize, policy: Option<ResidualPolicyReference<ArrayIrType>>| {
            let callee = residual_demand_diamond_callee(depth);
            residual_demand_program(policy, |builder, weights, bias| add_call(builder, &callee, vec![weights, bias]))
        };

        // The split of each level contains the shared splits of the level below it, so the programs grow by four
        // instructions per level (i.e., one known or replay call, the second call that shares its callee, and the
        // shifted input and sum of the level) while the saved matrix products double, as in JAX. Inlining the calls
        // instead would double the programs with every level.
        let policy = ResidualPolicyReference::new(DotsSavable);
        assert_eq!(
            [1, 2, 3, 4].map(|depth| residual_demand_statistics(&nested(depth, Some(policy.clone())), &[0])),
            [[3, 11, 11], [5, 15, 15], [9, 19, 19], [17, 23, 23]],
        );
        let linearization = nested(1, Some(policy.clone())).linearize_with_respect_to(&[0]).unwrap();
        assert_eq!(
            format!("{}\n{}", linearization.primal(), linearization.tangent()),
            indoc! {"
                lambda %0:f32[2, 2], %1:f32[2, 2], %2:f32[2, 2] .
                let %3:f32[2, 2], %4:f32[2, 2], %5:f32[2, 2] = call [derived_input_shardings=[_, _]] %1 %2 [
                    callee={
                        lambda %0:f32[2, 2], %1:f32[2, 2] .
                        let %2:f32[2, 2], %3:f32[2, 2] = call [derived_input_shardings=[_, _]] %0 %1 [
                            callee=^0={
                                lambda %0:f32[2, 2], %1:f32[2, 2] .
                                let %2:f32[2, 2] = dot [
                                    dimensions=(lhs_contracting=[1], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                                ] %0 %1
                                    %3:f32[2, 2] = sin %2
                                    %4:f32[2, 2] = cos %2
                                    %5:f32[2, 2] = mul %3 %4
                                    %6:f32[2, 2] = add %5 %0
                                in (%6, %2)
                            },
                        ]
                            %4:f32[2, 2] = sin %0
                            %5:f32[2, 2], %6:f32[2, 2] = call [derived_input_shardings=[_, _]] %4 %1 [
                                callee=^0,
                            ]
                            %7:f32[2, 2] = add %2 %5
                        in (%7, %3, %6)
                    },
                ]
                    %6:f32[2, 2] = mul %0 %3
                in (%6, %1, %4, %5)
                lambda %0:f32[2, 2], %1:f32[2, 2], %2:f32[2, 2], %3:f32[2, 2] .
                let %4:f32[2, 2] = rematerialize [policy=\"dots_savable\", differentiated=true] %0 %1 %2 %3 [
                    body={
                        lambda %0:f32[2, 2], %1:f32[2, 2], %2:f32[2, 2], %3:f32[2, 2] .
                        let %4:f32[2, 2] = call [derived_input_shardings=[_, _, _]] %1 %2 %3 [
                            callee={
                                lambda %0:f32[2, 2], %1:f32[2, 2], %2:f32[2, 2] .
                                let %3:f32[2, 2] = call [derived_input_shardings=[_, _]] %0 %1 [
                                    callee=^0={
                                        lambda %0:f32[2, 2], %1:f32[2, 2] .
                                        let %2:f32[2, 2] = sin %1
                                            %3:f32[2, 2] = cos %1
                                            %4:f32[2, 2] = mul %2 %3
                                            %5:f32[2, 2] = add %4 %0
                                        in (%5)
                                    },
                                ]
                                    %4:f32[2, 2] = sin %0
                                    %5:f32[2, 2] = call [derived_input_shardings=[_, _]] %4 %2 [
                                        callee=^0,
                                    ]
                                    %6:f32[2, 2] = add %3 %5
                                in (%6)
                            },
                        ]
                            %5:f32[2, 2] = mul %4 %0
                        in (%5)
                    },
                ]
                in (%4)"},
        );

        assert_residual_demand_linearizations_match(&nested(3, Some(policy)), &nested(3, None), &[0], None);
    }

    #[test]
    fn test_call_operation_split_on_residual_demand_replays_all_known_calls_without_saved_interior_values() {
        let callee = residual_demand_leaf_callee();
        let repeated = |count: usize, policy: Option<ResidualPolicyReference<ArrayIrType>>| {
            residual_demand_program(policy, |builder, weights, bias| {
                (0..count).fold(weights, |hidden, _| add_call(builder, &callee, vec![hidden, bias]))
            })
        };

        // A policy that saves nothing splits no call. The tangent replays every call whole from the saved weights and
        // bias, which it saves no matter how many calls there are, and the calls keep sharing one callee.
        let policy = ResidualPolicyReference::new(NothingSavable);
        assert_eq!(
            [1, 2, 4, 8].map(|count| residual_demand_statistics(&repeated(count, Some(policy.clone())), &[0])),
            [[2, 7, 8], [2, 8, 9], [2, 10, 11], [2, 14, 15]],
        );
        let linearization = repeated(2, Some(policy.clone())).linearize_with_respect_to(&[0]).unwrap();
        assert_eq!(
            format!("{}\n{}", linearization.primal(), linearization.tangent()),
            indoc! {"
                lambda %0:f32[2, 2], %1:f32[2, 2], %2:f32[2, 2] .
                let %3:f32[2, 2] = call %1 %2 [
                    callee=^0={
                        lambda %0:f32[2, 2], %1:f32[2, 2] .
                        let %2:f32[2, 2] = dot [
                            dimensions=(lhs_contracting=[1], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                        ] %0 %1
                            %3:f32[2, 2] = sin %2
                            %4:f32[2, 2] = cos %2
                            %5:f32[2, 2] = mul %3 %4
                            %6:f32[2, 2] = add %5 %0
                        in (%6)
                    },
                ]
                    %4:f32[2, 2] = call %3 %2 [
                        callee=^0,
                    ]
                    %5:f32[2, 2] = mul %0 %4
                in (%5, %1, %2)
                lambda %0:f32[2, 2], %1:f32[2, 2], %2:f32[2, 2] .
                let %3:f32[2, 2] = rematerialize [differentiated=true] %0 %1 %2 [
                    body={
                        lambda %0:f32[2, 2], %1:f32[2, 2], %2:f32[2, 2] .
                        let %3:f32[2, 2] = call %1 %2 [
                            callee=^0={
                                lambda %0:f32[2, 2], %1:f32[2, 2] .
                                let %2:f32[2, 2] = dot [
                                    dimensions=(lhs_contracting=[1], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                                ] %0 %1
                                    %3:f32[2, 2] = sin %2
                                    %4:f32[2, 2] = cos %2
                                    %5:f32[2, 2] = mul %3 %4
                                    %6:f32[2, 2] = add %5 %0
                                in (%6)
                            },
                        ]
                            %4:f32[2, 2] = call %3 %2 [
                                callee=^0,
                            ]
                            %5:f32[2, 2] = mul %4 %0
                        in (%5)
                    },
                ]
                in (%3)"},
        );

        assert_residual_demand_linearizations_match(&repeated(4, Some(policy)), &repeated(4, None), &[0], None);
    }

    #[test]
    fn test_call_operation_split_on_residual_demand_places_all_known_callee_intermediates() {
        // Only the expanded activation is needed by the tangent. Its all-known call must expose the compact tagged
        // scale to the policy, saving 4 KiB instead of 256 KiB, so the call is split on residual demand into a known
        // call that also returns the scale and a replay call that recomputes only the broadcast.
        let compact_type = ArrayType::new_static(DataType::F32, [1024]);
        let expanded_type = ArrayType::new_static(DataType::F32, [64, 1024]);
        let callee = build_program(vec![compact_type.clone().into()], |builder, inputs| {
            let scale = builder
                .add_instruction(ArrayOperation::from(SinOperation::new()), Vec::new(), vec![inputs[0]], None)
                .unwrap()[0];
            let scale = builder
                .add_instruction(
                    ArrayOperation::from(TagOperation::<ArrayType>::new("scale")),
                    Vec::new(),
                    vec![scale],
                    None,
                )
                .unwrap()[0];
            let broadcast = BroadcastOperation::new(expanded_type.clone(), vec![1]);
            builder
                .add_instruction(ArrayOperation::from(broadcast), Vec::new(), vec![scale], None)
                .unwrap()
                .to_vec()
        });
        let program =
            build_program(vec![expanded_type.clone().into(), compact_type.clone().into()], |builder, inputs| {
                let callee = builder.import_program(callee);
                let call = TestOperation::Call(CallOperation::new(0));
                let scale = builder.add_instruction(call, vec![callee], vec![inputs[1]], None).unwrap()[0];
                builder
                    .add_instruction(
                        ArrayOperation::from(MulOperation::new()),
                        Vec::new(),
                        vec![inputs[0], scale],
                        None,
                    )
                    .unwrap()
                    .to_vec()
            });

        // Ordinary partial evaluation preserves the boundary of the original compiled call and the expanded residual.
        let ordinary = program.partition(&[false, true]).unwrap();
        assert_eq!(ordinary.known_program().output_types(), vec![ArrayIrType::from(expanded_type.clone())]);
        assert_eq!(
            ordinary.to_string(),
            indoc! {"
                partition [
                    known_inputs=[1],
                    residual_inputs=[UnknownInput(0), ResidualEdge(0)],
                    outputs=[Unknown(0)],
                ]
                known={
                    lambda %0:f32[1024] .
                    let %1:f32[64, 1024] = call %0 [
                        callee={
                            lambda %0:f32[1024] .
                            let %1:f32[1024] = sin %0
                                %2:f32[1024] = tag [key=scale] %1
                                %3:f32[64, 1024] = broadcast [output_type=f32[64, 1024], output_axes=[1]] %2
                            in (%3)
                        },
                    ]
                    in (%1)
                }
                residual={
                    lambda %0:f32[64, 1024], %1:f32[64, 1024] .
                    let %2:f32[64, 1024] = mul %0 %1
                    in (%2)
                }"},
        );
        assert_eq!(expanded_type.element_count().unwrap().unwrap() * size_of::<f32>(), 262_144);

        let policy = ResidualPolicyReference::new(SaveOnlyTheseNames::new(["scale"]));
        let placed = program.partition_with_residual_policy(&[false, true], &policy).unwrap();
        assert_eq!(placed.known_program().output_types(), vec![ArrayIrType::from(compact_type.clone())]);
        assert_eq!(compact_type.element_count().unwrap().unwrap() * size_of::<f32>(), 4096);
        assert_eq!(
            placed.to_string(),
            indoc! {"
                partition [
                    known_inputs=[1],
                    residual_inputs=[UnknownInput(0), ResidualEdge(0)],
                    outputs=[Unknown(0)],
                ]
                known={
                    lambda %0:f32[1024] .
                    let %1:f32[1024] = call [derived_input_shardings=[_]] %0 [
                        callee={
                            lambda %0:f32[1024] .
                            let %1:f32[1024] = sin %0
                                %2:f32[1024] = tag [key=scale] %1
                            in (%2)
                        },
                    ]
                    in (%1)
                }
                residual={
                    lambda %0:f32[64, 1024], %1:f32[1024] .
                    let %2:f32[64, 1024] = call [derived_input_shardings=[_]] %1 [
                        callee={
                            lambda %0:f32[1024] .
                            let %1:f32[64, 1024] = broadcast [output_type=f32[64, 1024], output_axes=[1]] %0
                            in (%1)
                        },
                    ]
                        %3:f32[64, 1024] = mul %0 %2
                    in (%3)
                }"},
        );

        // The same cut survives the rematerialization wrapper's linearization, where the original output remains a
        // known primal result and the expanded scale is needed only by the derivative with respect to the activation.
        let plain_linearization = program.linearize_with_respect_to(&[0]).unwrap();
        let program =
            build_program(vec![expanded_type.clone().into(), compact_type.clone().into()], |builder, inputs| {
                let body = builder.import_program(program);
                let rematerialize = TestOperation::Rematerialize(RematerializeOperation::new(policy));
                builder.add_instruction(rematerialize, vec![body], inputs.to_vec(), None).unwrap().to_vec()
            });
        let linearization = program.linearize_with_respect_to(&[0]).unwrap();
        assert_eq!(linearization.residual_count(), 1);
        assert_eq!(
            linearization.primal().output_types(),
            vec![ArrayIrType::from(expanded_type.clone()), ArrayIrType::from(compact_type.clone())],
        );
        assert_eq!(
            format!("{}\n{}", linearization.primal(), linearization.tangent()),
            indoc! {"
                lambda %0:f32[64, 1024], %1:f32[1024] .
                let %2:f32[64, 1024], %3:f32[1024] = call [derived_input_shardings=[_]] %1 [
                    callee={
                        lambda %0:f32[1024] .
                        let %1:f32[1024] = sin %0
                            %2:f32[1024] = tag [key=scale] %1
                            %3:f32[64, 1024] = broadcast [output_type=f32[64, 1024], output_axes=[1]] %2
                        in (%3, %2)
                    },
                ]
                    %4:f32[64, 1024] = mul %0 %2
                in (%4, %3)
                lambda %0:f32[64, 1024], %1:f32[1024] .
                let %2:f32[64, 1024] = rematerialize [policy=\"save_only_these_names\", differentiated=true] %0 %1 [
                    body={
                        lambda %0:f32[64, 1024], %1:f32[1024] .
                        let %2:f32[64, 1024] = call [derived_input_shardings=[_]] %1 [
                            callee={
                                lambda %0:f32[1024] .
                                let %1:f32[64, 1024] = broadcast [output_type=f32[64, 1024], output_axes=[1]] %0
                                in (%1)
                            },
                        ]
                            %3:f32[64, 1024] = mul %2 %0
                        in (%3)
                    },
                ]
                in (%2)"},
        );

        // The rematerialized linearization computes exactly the primal output, tangent, and cotangent of the plain one.
        let inputs = vec![
            f32_value(expanded_type.clone(), &[2.0; 64 * 1024]),
            f32_value(compact_type, &(0..1024).map(|index| 0.1 + index as f32 * 1e-4).collect::<Vec<_>>()),
        ];
        let direction = f32_value(expanded_type.clone(), &[0.75; 64 * 1024]);
        let cotangent = f32_value(expanded_type, &[1.5; 64 * 1024]);
        let evaluate = |linearization: &Linearization<TestValue, TestOperation>| {
            let mut primal = linearization.primal().interpret(inputs.clone()).unwrap();
            let value = primal.remove(0);
            let tangent_inputs = std::iter::once(direction.clone()).chain(primal.clone()).collect();
            let tangent = linearization.tangent().interpret(tangent_inputs).unwrap();
            let pullback_inputs = std::iter::once(cotangent.clone()).chain(primal).collect();
            let pullback = linearization.pullback().unwrap().into_program().interpret(pullback_inputs).unwrap();
            vec![value, tangent[0].clone(), pullback[0].clone()]
        };
        let actual = evaluate(&linearization);
        assert_eq!(actual, evaluate(&plain_linearization));
        for value in &actual {
            let ArrayIrValue::Array(array) = value else {
                panic!("expected an array value");
            };
            assert!(array.elements::<f32>().unwrap().iter().all(|value| value.is_finite() && *value != 0.0));
        }
    }

    #[test]
    fn test_call_operation_split_on_residual_demand_rounds_saved_values() {
        // The function `(x, s) ↦ w · s` calls `w = f32(t)` for `t = tag(bf16(x) · bf16(x), "t")`. Rematerializing it
        // under a policy that saves `t` and differentiating it with respect to `s` splits the all-known call on
        // residual demand into a known call that saves `t` and computes `w` from it and a replay call that recomputes
        // `w` from the saved `t`. The known callee rounds `t` right after its producer, before computing `w` from it,
        // so that the output and its derivative with respect to `s` use the same `bf16` value of `t` even on backends
        // that compute the `bf16` product at a higher precision.
        let scalar_type = ArrayIrType::from(ArrayType::scalar(DataType::F32));
        let callee = Arc::new(build_program(vec![scalar_type.clone()], |builder, inputs| {
            let to_bf16 = ConvertElementTypeOperation::<ArrayType>::new(DataType::BF16, false);
            let x =
                builder.add_instruction(ArrayOperation::from(to_bf16), Vec::new(), vec![inputs[0]], None).unwrap()[0];
            let t = builder
                .add_instruction(ArrayOperation::from(MulOperation::new()), Vec::new(), vec![x, x], None)
                .unwrap()[0];
            let tag = ArrayOperation::from(TagOperation::<ArrayType>::new("t"));
            let t = builder.add_instruction(tag, Vec::new(), vec![t], None).unwrap()[0];
            let to_f32 = ConvertElementTypeOperation::<ArrayType>::new(DataType::F32, false);
            builder.add_instruction(ArrayOperation::from(to_f32), Vec::new(), vec![t], None).unwrap().to_vec()
        }));
        let body = |returns_w: bool| {
            build_program(vec![scalar_type.clone(); 2], |builder, inputs| {
                let w = add_call(builder, &callee, vec![inputs[0]]);
                let y = builder
                    .add_instruction(ArrayOperation::from(MulOperation::new()), Vec::new(), vec![w, inputs[1]], None)
                    .unwrap()[0];
                if returns_w { vec![w, y] } else { vec![y] }
            })
        };
        let policy = ResidualPolicyReference::new(SaveOnlyTheseNames::new(["t"]));

        // Placing the residuals of `(x, s) ↦ (w, w · s)` with `x` known, with the same policy, and without rounding,
        // splits the call in the same way, but its known callee consumes the saved `t` unrounded. The
        // rematerialization below must not reuse that split of the shared callee.
        let partition = body(true).partition_with_residual_policy(&[true, false], &policy).unwrap();
        assert_eq!(
            partition.known_program().to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:f32[], %2:bf16[] = call [derived_input_shardings=[_]] %0 [
                    callee={
                        lambda %0:f32[] .
                        let %1:bf16[] = convert_element_type [data_type=bf16] %0
                            %2:bf16[] = mul %1 %1
                            %3:bf16[] = tag [key=t] %2
                            %4:f32[] = convert_element_type [data_type=f32] %3
                        in (%4, %3)
                    },
                ]
                in (%1, %2)"},
        );

        let program = build_program(vec![scalar_type.clone(); 2], |builder, inputs| {
            let body = builder.import_program(body(false));
            let rematerialize = TestOperation::Rematerialize(RematerializeOperation::new(policy));
            builder.add_instruction(rematerialize, vec![body], inputs.to_vec(), None).unwrap().to_vec()
        });
        let linearization = program.linearize_with_respect_to(&[1]).unwrap();
        assert_eq!(
            format!("{}\n{}", linearization.primal(), linearization.tangent()),
            indoc! {"
                lambda %0:f32[], %1:f32[] .
                let %2:f32[], %3:bf16[] = call [derived_input_shardings=[_]] %0 [
                    callee={
                        lambda %0:f32[] .
                        let %1:bf16[] = convert_element_type [data_type=bf16] %0
                            %2:bf16[] = mul %1 %1
                            %3:bf16[] = tag [key=t] %2
                            %4:bf16[] = reduce_precision [exponent_bits=8, mantissa_bits=7] %3
                            %5:f32[] = convert_element_type [data_type=f32] %4
                        in (%5, %4)
                    },
                ]
                    %4:f32[] = mul %2 %1
                in (%4, %3)
                lambda %0:f32[], %1:bf16[] .
                let %2:f32[] = rematerialize [policy=\"save_only_these_names\", differentiated=true] %0 %1 [
                    body={
                        lambda %0:f32[], %1:bf16[] .
                        let %2:f32[] = call [derived_input_shardings=[_]] %1 [
                            callee={
                                lambda %0:bf16[] .
                                let %1:f32[] = convert_element_type [data_type=f32] %0
                                in (%1)
                            },
                        ]
                            %3:f32[] = mul %2 %0
                        in (%3)
                    },
                ]
                in (%2)"},
        );
    }

    #[test]
    fn test_call_operation_split_on_residual_demand_keeps_effectful_all_known_calls_whole() {
        let scalar_type: ArrayIrType = ArrayType::scalar(DataType::F32).into();
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let input = builder.add_input(scalar_type.clone());
        builder
            .add_instruction(
                ArrayOperation::from(PrintOperation::<ArrayType>::new("known")),
                Vec::new(),
                vec![input],
                None,
            )
            .unwrap();
        let callee = Arc::new(
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![input], vec![Placeholder], vec![Placeholder])
                .unwrap(),
        );
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let known = builder.add_input(scalar_type.clone());
        let unknown = builder.add_input(scalar_type);
        let forwarded = add_call(&mut builder, &callee, vec![known]);
        let output = builder
            .add_instruction(ArrayOperation::from(MulOperation::new()), Vec::new(), vec![forwarded, unknown], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let partition = program
            .partition_with_residual_policy(&[true, false], &ResidualPolicyReference::new(NothingSavable))
            .unwrap();
        assert_eq!(
            partition.to_string(),
            indoc! {"
                partition [
                    known_inputs=[0],
                    residual_inputs=[UnknownInput(1), ResidualEdge(0)],
                    outputs=[Unknown(0)],
                ]
                known={
                    lambda %0:f32[] .
                    let %1:f32[] = call %0 [
                        callee={
                            lambda %0:f32[] .
                            let %1:f32[] = print [label=known] %0
                            in (%0)
                        },
                    ]
                    in (%1)
                }
                residual={
                    lambda %0:f32[], %1:f32[] .
                    let %2:f32[] = mul %1 %0
                    in (%2)
                }"},
        );
    }

    #[test]
    fn test_call_operation_split_on_residual_demand_saves_guarded_known_callee_outputs() {
        // The callee computes a matrix product that `DotsSavable` saves, so replaying its call would recompute a saved
        // value. An ordered assertion keeps the boundary of the call, so the residual planner saves the demanded output
        // of the call instead of splitting it.
        let matrix_type = residual_demand_matrix_type();
        let predicate_type = ArrayIrType::from(ArrayType::scalar(DataType::Boolean));
        let dot = ArrayOperation::<Array>::from(DotOperation::new(DotDimensionNumbers::new(
            vec![1],
            vec![0],
            vec![],
            vec![],
        )));
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let predicate = builder.add_input(predicate_type);
        let input = builder.add_input(matrix_type.clone());
        builder
            .add_instruction(
                ArrayOperation::from(AssertOperation::<ArrayType>::new("required predicate")),
                Vec::new(),
                vec![predicate],
                None,
            )
            .unwrap();
        let product = builder.add_instruction(dot, Vec::new(), vec![input, input], None).unwrap()[0];
        let output = builder
            .add_instruction(ArrayOperation::from(SinOperation::new()), Vec::new(), vec![product], None)
            .unwrap()[0];
        let callee = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let inputs = callee
            .input_types()
            .into_owned()
            .into_iter()
            .map(|r#type| builder.add_input(r#type))
            .collect::<Vec<_>>();
        let tangent = builder.add_input(matrix_type);
        let callee = builder.import_program(callee);
        let called = builder
            .add_instruction(TestOperation::Call(CallOperation::new(0)), vec![callee], inputs, None)
            .unwrap()[0];
        let output = builder
            .add_instruction(ArrayOperation::from(MulOperation::new()), Vec::new(), vec![called, tangent], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        let partition = program
            .partition_with_residual_policy(&[true, true, false], &ResidualPolicyReference::new(DotsSavable))
            .unwrap();
        assert_eq!(
            partition.to_string(),
            indoc! {"
                partition [
                    known_inputs=[0, 1],
                    residual_inputs=[UnknownInput(2), ResidualEdge(0)],
                    outputs=[Unknown(0)],
                ]
                known={
                    lambda %0:bool[], %1:f32[2, 2] .
                    let %2:f32[2, 2] = call %0 %1 [
                        callee={
                            lambda %0:bool[], %1:f32[2, 2] .
                            let () = assert [message=\"required predicate\", labels=[]] %0
                                %2:f32[2, 2] = dot [
                                    dimensions=(lhs_contracting=[1], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                                ] %1 %1
                                %3:f32[2, 2] = sin %2
                            in (%3)
                        },
                    ]
                    in (%2)
                }
                residual={
                    lambda %0:f32[2, 2], %1:f32[2, 2] .
                    let %2:f32[2, 2] = mul %1 %0
                    in (%2)
                }"},
        );
    }

    #[test]
    fn test_call_operation_split_on_residual_demand_ignores_undemanded_known_outputs() {
        let scalar_type = ArrayIrType::from(ArrayType::scalar(DataType::F32));
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let input = builder.add_input(scalar_type.clone());
        let sine = builder
            .add_instruction(ArrayOperation::from(SinOperation::new()), Vec::new(), vec![input], None)
            .unwrap()[0];
        let needed = builder
            .add_instruction(
                ArrayOperation::from(TagOperation::<ArrayType>::new("needed")),
                Vec::new(),
                vec![sine],
                None,
            )
            .unwrap()[0];
        let cosine = builder
            .add_instruction(ArrayOperation::from(CosOperation::new()), Vec::new(), vec![input], None)
            .unwrap()[0];
        let unused = builder
            .add_instruction(
                ArrayOperation::from(TagOperation::<ArrayType>::new("unused")),
                Vec::new(),
                vec![cosine],
                None,
            )
            .unwrap()[0];
        let callee = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![needed, unused], vec![Placeholder], vec![Placeholder; 2])
            .unwrap();
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let known = builder.add_input(scalar_type.clone());
        let unknown = builder.add_input(scalar_type.clone());
        let callee = builder.import_program(callee);
        let called = builder
            .add_instruction(TestOperation::Call(CallOperation::new(0)), vec![callee], vec![known], None)
            .unwrap()
            .to_vec();
        let first = builder
            .add_instruction(ArrayOperation::from(MulOperation::new()), Vec::new(), vec![called[0], unknown], None)
            .unwrap()[0];
        let second = builder
            .add_instruction(ArrayOperation::from(MulOperation::new()), Vec::new(), vec![called[1], unknown], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![first, second, called[1]],
                vec![Placeholder; 2],
                vec![Placeholder; 3],
            )
            .unwrap();
        let policy =
            ResidualPolicyReference::new(RematerializationPolicyFn::new::<ArrayIrType>(|candidate| {
                if candidate.producers().iter().any(|producer| {
                    producer.payload::<TagOperation<ArrayType>>().is_some_and(|tag| tag.key() == "unused")
                }) {
                    Err(ResidualRejection::new("the unused result must not cross the residual boundary"))
                } else {
                    Ok(ResidualDecision::<NoStorage>::Save)
                }
            }));

        // The second call output remains a primal result. Its presence must not classify it as a residual. The call
        // keeps its boundary, and the policy saves its first output, which it classifies through the callee.
        let primal_only = program.clone().with_outputs(&[0, 2]).unwrap();
        let placed = primal_only.partition_with_residual_policy(&[true, false], &policy).unwrap();
        assert_eq!(placed.known_program().output_types(), vec![scalar_type.clone(), scalar_type]);
        assert_eq!(
            placed.to_string(),
            indoc! {"
                partition [
                    known_inputs=[0],
                    residual_inputs=[UnknownInput(1), ResidualEdge(0)],
                    outputs=[Unknown(0), Known(0)],
                ]
                known={
                    lambda %0:f32[] .
                    let %1:f32[], %2:f32[] = call %0 [
                        callee={
                            lambda %0:f32[] .
                            let %1:f32[] = sin %0
                                %2:f32[] = tag [key=needed] %1
                                %3:f32[] = cos %0
                                %4:f32[] = tag [key=unused] %3
                            in (%2, %4)
                        },
                    ]
                    in (%2, %1)
                }
                residual={
                    lambda %0:f32[], %1:f32[] .
                    let %2:f32[] = mul %1 %0
                    in (%2)
                }"},
        );

        // Requiring the same result in residual work must propagate the original rejection instead.
        assert_eq!(
            program
                .with_outputs(&[0, 1])
                .unwrap()
                .partition_with_residual_policy(&[true, false], &policy)
                .map(|_| ())
                .map_err(ResidualPolicyError::from),
            Err(ResidualPolicyError::Rejected {
                policy: policy.name().to_owned(),
                rejection: ResidualRejection::new("the unused result must not cross the residual boundary"),
            }),
        );
    }

    #[test]
    fn test_call_operation_batch() {
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let callee = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let callees = [Arc::new(callee)];
        let regions = AttachedRegions::Callees(&callees);
        let driver = RecursiveBatchingDriver::new(&regions);
        let vector = ArrayIrType::Array(ArrayType::new_static(DataType::F32, [3]));
        let (_, program) = TracingContext::<TestValue, TestOperation>::trace(
            |inputs: Vec<Tracer<TracingContext<TestValue, TestOperation>>>| {
                let parent = inputs[0].context();
                let extent = parent.constant(ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap()));
                let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(parent.clone(), extent);
                let outputs = CallOperation::<ArrayIrType>::new(0).batch(
                    &context,
                    &driver,
                    &[ArrayIrBatch::new(inputs[0].clone(), Some(0))?],
                )?;
                Ok::<_, ProgramError>(outputs.into_iter().map(ArrayIrBatch::into_value).collect::<Vec<_>>())
            },
            vec![vector],
        )
        .unwrap();

        // The batched callee receives the extent as its leading input, and it does not return the extent, because that
        // bookkeeping output would only forward the input and an eager parent cannot return it from a compiled call.
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[3] .
                let %1:dimension<3> = const 3
                    %2:f32[3] = call [capture_count=1, derived_input_shardings=[_, _]] %1 %0 [
                        callee={
                            lambda %0:dimension<3>, %1:f32[3] .
                            in (%1)
                        },
                    ]
                in (%2)"},
        );
    }

    #[test]
    fn test_call_operation_batch_places_inputs_at_the_boundary_of_the_original_call() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let item = ArrayType::new_static(DataType::F32, [4]);
        let batch_split =
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::Replicated])
                .unwrap();
        let batch_type =
            ArrayIrType::Array(ArrayType::new_static(DataType::F32, [2, 4]).with_sharding(batch_split).unwrap());
        let callee = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let input = builder.add_input(item.into());
            let output = builder
                .add_instruction(ArrayOperation::from(SinOperation::new()), Vec::new(), vec![input], None)
                .unwrap()[0];
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let callees = [Arc::new(callee)];
        let regions = AttachedRegions::Callees(&callees);
        let driver = RecursiveBatchingDriver::new(&regions);
        let batched = |call: CallOperation<ArrayIrType>| {
            let (_, program) = TracingContext::<TestValue, TestOperation>::trace(
                |inputs: Vec<Tracer<TracingContext<TestValue, TestOperation>>>| {
                    let parent = inputs[0].context();
                    let extent = parent.constant(ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()));
                    let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(parent.clone(), extent)
                        .with_axis_sharding(ShardingDimension::sharded(["x"]));
                    let outputs = call.batch(&context, &driver, &[ArrayIrBatch::new(inputs[0].clone(), Some(0))?])?;
                    Ok::<_, ProgramError>(outputs.into_iter().map(ArrayIrBatch::into_value).collect::<Vec<_>>())
                },
                vec![batch_type.clone()],
            )
            .unwrap();
            program.to_string()
        };

        // The batched callee is created by batching, so the batched call is a derived call, which places each item
        // where the original call places it, with the batch dimension placed as the batched value is, rather than as
        // the traced types of the batched callee say. The callee declares an item type without a sharding, which the
        // original call places replicated over the mesh where it runs, so the batch stays split. The batched callee is
        // batched from the callee with that boundary as its input type, so its result records the split too, as it
        // would for a callee that declares its input replicated. A derived original call keeps its own boundary, and a
        // call of a `jit` dispatcher places nothing, so the types of their batched callees record no placement. The
        // leading extent input has no boundary.
        let derived = |shardings| CallOperation::<ArrayIrType>::new(0).with_derived_input_shardings(shardings);
        let split = indoc! {"
            lambda %0:f32[2, 4][sharding={mesh<['x'=2:explicit]>, [{'x'}, {}]}] .
            let %1:dimension<2> = const 2
                %2:f32[2, 4][sharding={mesh<['x'=2:explicit]>, [{'x'}, {}]}] = call [
                    capture_count=1,
                    derived_input_shardings=[_, {mesh<['x'=2:explicit]>, [{'x'}, {}]}],
                ] %1 %0 [
                    callee={
                        lambda %0:dimension<2>, %1:f32[2, 4][sharding={mesh<['x'=2:explicit]>, [{'x'}, {}]}] .
                        let %2:f32[2, 4][sharding={mesh<['x'=2:explicit]>, [{'x'}, {}]}] = sin %1
                        in (%2)
                    },
                ]
            in (%2)"};
        assert_eq!(batched(CallOperation::new(0)), split);
        assert_eq!(batched(derived(vec![Some(Sharding::replicated(mesh, 1))])), split);
        assert_eq!(
            batched(derived(vec![None])),
            indoc! {"
                lambda %0:f32[2, 4][sharding={mesh<['x'=2:explicit]>, [{'x'}, {}]}] .
                let %1:dimension<2> = const 2
                    %2:f32[2, 4] = call [capture_count=1, derived_input_shardings=[_, _]] %1 %0 [
                        callee={
                            lambda %0:dimension<2>, %1:f32[2, 4] .
                            let %2:f32[2, 4] = sin %1
                            in (%2)
                        },
                    ]
                in (%2)"},
        );
        assert_eq!(
            batched(CallOperation::new(0).with_kept_input_placements()),
            indoc! {"
                lambda %0:f32[2, 4][sharding={mesh<['x'=2:explicit]>, [{'x'}, {}]}] .
                let %1:dimension<2> = const 2
                    %2:f32[2, 4] = call [capture_count=1, derived_input_shardings=[_, _], keeps_input_placements=true] %1 %0 [
                        callee={
                            lambda %0:dimension<2>, %1:f32[2, 4] .
                            let %2:f32[2, 4] = sin %1
                            in (%2)
                        },
                    ]
                in (%2)"},
        );
    }

    #[test]
    fn test_call_operation_batch_homogeneous_named_axis() {
        /// Array-only family for exercising the homogeneous `call` batching rule without array IR carriers.
        #[derive(Clone, Debug, Operation)]
        #[ryft(type(ArrayType), constant(Array), dispatch(batching))]
        enum NamedAxisCallOperation {
            AxisIndex(AxisIndexOperation),
            Iota(IotaOperation<ArrayType>),
            Broadcast(BroadcastOperation),
            Transpose(TransposeOperation),
            Zero(ZeroOperation<ArrayType>),
            Call(CallOperation<ArrayType>),
        }

        for replicated in [false, true] {
            let mut builder = ProgramBuilder::<Array, NamedAxisCallOperation>::new();
            let input = replicated.then(|| builder.add_input(ArrayType::scalar(DataType::F32)));
            let index =
                builder.add_instruction(AxisIndexOperation::new("items".to_owned()), vec![], vec![], None).unwrap()[0];
            let outputs = std::iter::once(index).chain(input).collect::<Vec<_>>();
            let callee = builder
                .build::<Vec<Array>, Vec<Array>>(
                    outputs,
                    vec![Placeholder; usize::from(replicated)],
                    vec![Placeholder; 1 + usize::from(replicated)],
                )
                .unwrap();
            let callees = [Arc::new(callee)];
            let regions = AttachedRegions::Callees(&callees);
            let driver = RecursiveBatchingDriver::new(&regions);
            let (_, program) = TracingContext::<Array, NamedAxisCallOperation>::trace_with_context_and_named_axes(
                |parent, inputs: Vec<Tracer<TracingContext<Array, NamedAxisCallOperation>>>| {
                    let context = BatchingContext::<_, ArrayBatchingPolicy>::new(parent.clone(), 3)
                        .with_axis_name("items".to_owned());
                    let inputs = inputs.into_iter().map(ArrayBatch::replicated).collect::<Vec<_>>();
                    let outputs = CallOperation::<ArrayType>::new(0).batch(&context, &driver, &inputs)?;
                    assert_eq!(outputs[0].batch_axis_position(), Some(0));
                    if replicated {
                        assert_eq!(outputs[1].batch_axis_position(), None);
                    }
                    Ok::<_, ProgramError>(outputs.into_iter().map(ArrayBatch::into_value).collect::<Vec<_>>())
                },
                vec![ArrayType::scalar(DataType::F32); usize::from(replicated)],
                vec![],
            )
            .unwrap();
            let expected = if replicated {
                indoc! {"
                    lambda %0:f32[] .
                    let %1:u64[3], %2:f32[] = call [derived_input_shardings=[_]] %0 [
                        callee={
                            lambda %0:f32[] .
                            let %1:u64[3] = iota [type=u64[3], dimension=0]
                            in (%1, %0)
                        },
                    ]
                    in (%1, %2)"}
            } else {
                indoc! {"
                    lambda  .
                    let %0:u64[3] = call [derived_input_shardings=[]] [
                        callee={
                            lambda  .
                            let %0:u64[3] = iota [type=u64[3], dimension=0]
                            in (%0)
                        },
                    ]
                    in (%0)"}
            };
            assert_eq!(program.to_string(), expected);
            assert_eq!(program.output_types()[0], ArrayType::new_static(DataType::U64, [3]));
        }
    }

    #[test]
    fn test_call_operation_jvp() {
        // Linearizing a program that calls `f = |x| x.sin()` (i.e., `Program::linearize`, which dispatches the `jvp`
        // rule of `CallOperation`) realizes `jvp(jit(f)) = jit(jvp f)`, because the split keeps both call boundaries
        // instead of inlining the callee. The primal half is a single `call` that produces the primal output plus
        // one residual, and the tangent half is a single `call` that consumes that residual. Interpreting the two
        // halves gives `(sin(x), cos(x) * dx)`.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Auto).unwrap()]).unwrap();
        let input_type = ArrayType::scalar(DataType::F32).with_sharding(Sharding::replicated(mesh, 0)).unwrap();
        let callee = Arc::new(build_program(vec![input_type.clone().into()], |builder, inputs| {
            builder
                .add_instruction(ArrayOperation::from(SinOperation::new()), Vec::new(), vec![inputs[0]], None)
                .unwrap()
                .to_vec()
        }));
        let program = build_program(vec![input_type.clone().into()], |builder, inputs| {
            vec![add_call(builder, &callee, inputs.to_vec())]
        });
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[][sharding={mesh<['x'=1:auto]>, []}] .
                let %1:f32[][sharding={mesh<['x'=1:auto]>, []}] = call %0 [
                    callee={
                        lambda %0:f32[][sharding={mesh<['x'=1:auto]>, []}] .
                        let %1:f32[][sharding={mesh<['x'=1:auto]>, []}] = sin %0
                        in (%1)
                    },
                ]
                in (%1)"},
        );

        // The pushforward of `sin` saves `cos(x)` as its only residual, and both halves keep the callee behind a
        // `call` boundary rather than inlining `sin` and `cos`.
        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.residual_count(), 1);
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:f32[][sharding={mesh<['x'=1:auto]>, []}] .
                let %1:f32[][sharding={mesh<['x'=1:auto]>, []}], %2:f32[][sharding={mesh<['x'=1:auto]>, []}] = call [derived_input_shardings=[_]] %0 [
                    callee={
                        lambda %0:f32[][sharding={mesh<['x'=1:auto]>, []}] .
                        let %1:f32[][sharding={mesh<['x'=1:auto]>, []}] = sin %0
                            %2:f32[][sharding={mesh<['x'=1:auto]>, []}] = cos %0
                        in (%1, %2)
                    },
                ]
                in (%1, %2)"},
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f32[][sharding={mesh<['x'=1:auto]>, []}], %1:f32[][sharding={mesh<['x'=1:auto]>, []}] .
                let %2:f32[][sharding={mesh<['x'=1:auto]>, []}] = call [derived_input_shardings=[_]] %0 %1 [
                    callee={
                        lambda %0:f32[][sharding={mesh<['x'=1:auto]>, []}], %1:f32[][sharding={mesh<['x'=1:auto]>, []}] .
                        let %2:f32[][sharding={mesh<['x'=1:auto]>, []}] = mul %1 %0
                        in (%2)
                    },
                ]
                in (%2)"},
        );
        assert_eq!(linearization.primal().output_types().len(), 1 + linearization.residual_count());
        assert_eq!(linearization.tangent().input_types().len(), 1 + linearization.residual_count());
        let scalar = |value: f32| f32_value(input_type.clone(), &[value]);
        for (primal, tangent) in [(0.0f32, 1.0f32), (0.25, 2.0), (0.5, -0.5), (1.0, 0.7)] {
            let outputs = linearization.primal().interpret(vec![scalar(primal)]).unwrap();
            assert_eq!(outputs, vec![scalar(primal.sin()), scalar(primal.cos())]);
            assert_eq!(
                linearization.tangent().interpret(vec![scalar(tangent), outputs[1].clone()]),
                Ok(vec![scalar(primal.cos() * tangent)]),
            );
        }
    }

    #[test]
    fn test_call_operation_jvp_omits_zero_space_boundary_tangents() {
        let dimension_type =
            ArrayIrType::Dimension(DimensionType::new("size", DimensionBounds::positive(Some(9)).unwrap()));
        let array_type = ArrayIrType::Array(ArrayType::scalar(DataType::F64));

        let mut callee_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let dimension = callee_builder.add_input(dimension_type.clone());
        let array = callee_builder.add_input(array_type.clone());
        let callee = callee_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![dimension, array], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let dimension = builder.add_input(dimension_type);
        let array = builder.add_input(array_type);
        let callee = builder.import_region(callee.entry_region_ref());
        let outputs = builder
            .add_instruction(TestOperation::Call(CallOperation::new(0)), vec![callee], vec![dimension, array], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();

        let jvp = program.jvp().unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:dimension<size ∈ [1, 9)>, %1:f64[], %2:f64[] .
                let %3:dimension<size ∈ [1, 9)>, %4:f64[], %5:f64[] = call [derived_input_shardings=[_, _, _]] %0 %1 %2 [
                    callee={
                        lambda %0:dimension<size ∈ [1, 9)>, %1:f64[], %2:f64[] .
                        in (%0, %1, %2)
                    },
                ]
                in (%3, %4, %5)"},
        );
    }

    #[test]
    fn test_call_operation_jvp_keeps_capture_prefix_inactive() {
        // The callee's leading input is its lifted capture (`capture_count = 1`). The call's activity mask starts
        // inactive over that prefix, numeric captures included, so the fused callee consumes the primal inputs,
        // capture prefix first, followed only by the live input tangent.
        let r#type = ArrayIrType::Array(f64_vector_type(4));
        let callee = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let captured = builder.add_input(r#type.clone());
            let input = builder.add_input(r#type.clone());
            let product = builder
                .add_instruction(ArrayOperation::from(MulOperation::new()), Vec::new(), vec![captured, input], None)
                .unwrap()[0];
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![product], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let callee = builder.import_region(callee.entry_region_ref());
        let captured = builder.add_input(r#type.clone());
        let input = builder.add_input(r#type);
        let output = builder
            .add_instruction(TestOperation::Call(CallOperation::new(1)), vec![callee], vec![captured, input], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        assert_eq!(
            program.jvp().unwrap().to_string(),
            indoc! {"
                lambda %0:f64[4], %1:f64[4], %2:f64[4], %3:f64[4] .
                let %4:f64[4], %5:f64[4] = call [capture_count=1, derived_input_shardings=[_, _, _]] %0 %1 %3 [
                    callee={
                        lambda %0:f64[4], %1:f64[4], %2:f64[4] .
                        let %3:f64[4] = mul %0 %1
                            %4:f64[4] = mul %0 %2
                        in (%3, %4)
                    },
                ]
                in (%4, %5)"},
        );
    }

    #[test]
    fn test_call_operation_jvp_threads_plumbing_and_active_references() {
        // The callee `f(captured, r, x) = { r += x; read(r) * read(captured) }` behind `call[capture_count=1]`
        // reads a captured plumbing reference and mutates an active reference input.
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));
        let callee = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let captured = builder.add_input(reference_type.clone());
            let reference = builder.add_input(reference_type.clone());
            let update = builder.add_input(scalar_type.clone());
            builder
                .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
                .unwrap();
            let state =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
            let scale =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![captured], None).unwrap()[0];
            let output = builder
                .add_instruction(ArrayOperation::from(MulOperation::new()), Vec::new(), vec![state, scale], None)
                .unwrap()[0];
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder; 3], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let callee = builder.import_region(callee.entry_region_ref());
        let captured = builder.add_input(reference_type.clone());
        let reference = builder.add_input(reference_type.clone());
        let update = builder.add_input(scalar_type.clone());
        let output = builder
            .add_instruction(
                TestOperation::Call(CallOperation::new(1)),
                vec![callee],
                vec![captured, reference, update],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();

        // The plumbing capture is inactive and the active reference receives a tangent reference input, so the fused
        // program and its one fused call consume `[captured, r, x, ṫr, ẋ]`, keeping the capture prefix verbatim.
        let jvp = program.entry_region_ref().jvp(&[1, 2]).unwrap();
        assert_eq!(
            jvp.input_types(),
            vec![
                reference_type.clone(),
                reference_type.clone(),
                scalar_type.clone(),
                reference_type.clone(),
                scalar_type.clone(),
            ],
        );
        assert_eq!(jvp.output_types(), vec![scalar_type.clone(), scalar_type.clone()]);
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:ref<f32[]>, %1:ref<f32[]>, %2:f32[], %3:ref<f32[]>, %4:f32[] .
                let %5:f32[], %6:f32[] = call [capture_count=1, derived_input_shardings=[_, _, _, _, _]] %0 %1 %2 %3 %4 [
                    callee={
                        lambda %0:ref<f32[]>, %1:ref<f32[]>, %2:f32[], %3:ref<f32[]>, %4:f32[] .
                        let () = reference_add_update %1 %2
                            () = reference_add_update %3 %4
                            %5:f32[] = reference_read %1
                            %6:f32[] = reference_read %3
                            %7:f32[] = reference_read %0
                            %8:f32[] = mul %5 %7
                            %9:f32[] = mul %7 %6
                        in (%8, %9)
                    },
                ]
                in (%5, %6)"},
        );
    }

    #[test]
    fn test_call_operation_jvp_omits_inactive_reference_output_tangents() {
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let mut callee_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = callee_builder.add_input(reference_type.clone());
        let value = callee_builder.add_input(scalar_type.clone());
        let doubled = callee_builder
            .add_instruction(ArrayOperation::from(AddOperation::new()), Vec::new(), vec![value, value], None)
            .unwrap()[0];
        let callee = callee_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![reference, doubled],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = builder.add_input(reference_type.clone());
        let value = builder.add_input(scalar_type.clone());
        let callee = builder.import_region(callee.entry_region_ref());
        let outputs = builder
            .add_instruction(TestOperation::Call(CallOperation::new(1)), vec![callee], vec![reference, value], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        let jvp = program.entry_region_ref().jvp(&[1]).unwrap();
        assert_eq!(jvp.input_types(), vec![reference_type.clone(), scalar_type.clone(), scalar_type.clone()]);
        assert_eq!(jvp.output_types(), vec![reference_type, scalar_type.clone(), scalar_type]);
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:ref<f32[]>, %1:f32[], %2:f32[] .
                let %3:ref<f32[]>, %4:f32[], %5:f32[] = call [capture_count=1, derived_input_shardings=[_, _, _]] %0 %1 %2 [
                    callee={
                        lambda %0:ref<f32[]>, %1:f32[], %2:f32[] .
                        let %3:f32[] = add %1 %1
                            %4:f32[] = add %2 %2
                        in (%0, %3, %4)
                    },
                ]
                in (%3, %4, %5)"},
        );
    }

    #[test]
    fn test_call_operation_jvp_treats_structurally_zero_tangents_as_inactive() {
        // When `f(x, y) = (sin(x), cos(y))` is differentiated with respect to `x` only, the structurally zero tangent
        // of `y` gets no fused callee slot, and the tangent of `cos(y)`, which then depends on no tangent input, is a
        // structural zero that the fused call does not return.
        let r#type = ArrayIrType::Array(f64_vector_type(4));
        let callee = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let x = builder.add_input(r#type.clone());
            let y = builder.add_input(r#type.clone());
            let sine = builder
                .add_instruction(ArrayOperation::from(SinOperation::new()), Vec::new(), vec![x], None)
                .unwrap()[0];
            let cosine = builder
                .add_instruction(ArrayOperation::from(CosOperation::new()), Vec::new(), vec![y], None)
                .unwrap()[0];
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![sine, cosine], vec![Placeholder; 2], vec![Placeholder; 2])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let callee = builder.import_region(callee.entry_region_ref());
        let inputs = (0..2).map(|_| builder.add_input(r#type.clone())).collect::<Vec<_>>();
        let outputs = builder
            .add_instruction(TestOperation::Call(CallOperation::new(0)), vec![callee], inputs, None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        assert_eq!(
            program.entry_region_ref().jvp(&[0]).unwrap().to_string(),
            indoc! {"
                lambda %0:f64[4], %1:f64[4], %2:f64[4] .
                let %3:f64[4], %4:f64[4], %5:f64[4] = call [derived_input_shardings=[_, _, _]] %0 %1 %2 [
                    callee={
                        lambda %0:f64[4], %1:f64[4], %2:f64[4] .
                        let %3:f64[4] = sin %0
                            %4:f64[4] = cos %1
                            %5:f64[4] = cos %0
                            %6:f64[4] = mul %5 %2
                        in (%3, %4, %6)
                    },
                ]
                    %6:f64[4] = zero [type=f64[4]]
                in (%3, %4, %5, %6)"},
        );
    }

    #[test]
    fn test_call_operation_jvp_omits_independent_output_tangents() {
        // In `f(x) = (sin(x), 0)`, the tangent of the constant output depends on no tangent input, so it is a
        // structural zero that the fused call does not return.
        let r#type = ArrayIrType::Array(f64_vector_type(4));
        let callee = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let x = builder.add_input(r#type.clone());
            let sine = builder
                .add_instruction(ArrayOperation::from(SinOperation::new()), Vec::new(), vec![x], None)
                .unwrap()[0];
            let zero = ArrayOperation::from(ZeroOperation::new(f64_vector_type(4)));
            let zero = builder.add_instruction(zero, Vec::new(), Vec::new(), None).unwrap()[0];
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![sine, zero], vec![Placeholder], vec![Placeholder; 2])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let callee = builder.import_region(callee.entry_region_ref());
        let input = builder.add_input(r#type);
        let outputs = builder
            .add_instruction(TestOperation::Call(CallOperation::new(0)), vec![callee], vec![input], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(outputs, vec![Placeholder], vec![Placeholder; 2])
            .unwrap();
        assert_eq!(
            program.jvp().unwrap().to_string(),
            indoc! {"
                lambda %0:f64[4], %1:f64[4] .
                let %2:f64[4], %3:f64[4], %4:f64[4] = call [derived_input_shardings=[_, _]] %0 %1 [
                    callee={
                        lambda %0:f64[4], %1:f64[4] .
                        let %2:f64[4] = sin %0
                            %3:f64[4] = zero [type=f64[4]]
                            %4:f64[4] = cos %0
                            %5:f64[4] = mul %4 %1
                        in (%2, %3, %5)
                    },
                ]
                    %5:f64[4] = zero [type=f64[4]]
                in (%2, %3, %4, %5)"},
        );
    }

    #[test]
    fn test_call_operation_jvp_preserves_live_outputs_around_inactive_tangent() {
        // `f(x) = (sin(x), 0, x)` has an inactive middle tangent slot. Projecting that slot must leave both live
        // tangents aligned with their original primal outputs rather than pairing the final tangent with the zero.
        let r#type = ArrayIrType::Array(f64_vector_type(4));
        let callee = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let input = builder.add_input(r#type.clone());
            let sine = builder
                .add_instruction(ArrayOperation::from(SinOperation::new()), Vec::new(), vec![input], None)
                .unwrap()[0];
            let zero = ArrayOperation::from(ZeroOperation::new(f64_vector_type(4)));
            let zero = builder.add_instruction(zero, Vec::new(), Vec::new(), None).unwrap()[0];
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(
                    vec![sine, zero, input],
                    vec![Placeholder],
                    vec![Placeholder; 3],
                )
                .unwrap()
        };
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let callee = builder.import_region(callee.entry_region_ref());
        let input = builder.add_input(r#type.clone());
        let outputs = builder
            .add_instruction(TestOperation::Call(CallOperation::new(0)), vec![callee], vec![input], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(outputs, vec![Placeholder], vec![Placeholder; 3])
            .unwrap();

        let jvp = program.jvp().unwrap();
        assert_eq!(jvp.output_types(), vec![r#type; 6]);
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:f64[4], %1:f64[4] .
                let %2:f64[4], %3:f64[4], %4:f64[4], %5:f64[4], %6:f64[4] = call [derived_input_shardings=[_, _]] %0 %1 [
                    callee={
                        lambda %0:f64[4], %1:f64[4] .
                        let %2:f64[4] = sin %0
                            %3:f64[4] = zero [type=f64[4]]
                            %4:f64[4] = cos %0
                            %5:f64[4] = mul %4 %1
                        in (%2, %3, %0, %5, %1)
                    },
                ]
                    %7:f64[4] = zero [type=f64[4]]
                in (%2, %3, %4, %5, %7, %6)"},
        );
    }

    #[test]
    fn test_call_operation_jvp_skips_tangent_calls_without_live_output_tangents() {
        // When `f(x, y) = sin(x)` is differentiated with respect to `y` only, no output tangent depends on the tangent
        // of `y` and the tangent callee has no effects, so only the source call is bound and the output tangent is a
        // structural zero.
        let r#type = ArrayIrType::Array(f64_vector_type(4));
        let callee = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let x = builder.add_input(r#type.clone());
            builder.add_input(r#type.clone());
            let sine = builder
                .add_instruction(ArrayOperation::from(SinOperation::new()), Vec::new(), vec![x], None)
                .unwrap()[0];
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![sine], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let callee = builder.import_region(callee.entry_region_ref());
        let inputs = (0..2).map(|_| builder.add_input(r#type.clone())).collect::<Vec<_>>();
        let output = builder
            .add_instruction(TestOperation::Call(CallOperation::new(0)), vec![callee], inputs, None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.entry_region_ref().jvp(&[1]).unwrap().to_string(),
            indoc! {"
                lambda %0:f64[4], %1:f64[4], %2:f64[4] .
                let %3:f64[4] = call %0 %1 [
                    callee={
                        lambda %0:f64[4], %1:f64[4] .
                        let %2:f64[4] = sin %0
                        in (%2)
                    },
                ]
                    %4:f64[4] = zero [type=f64[4]]
                in (%3, %4)"},
        );
    }

    #[test]
    fn test_call_operation_jvp_preserves_local_result_shape_relationships() {
        // `f(x, size) = broadcast(x, [dimension_from_scalar(size)])` is linear in `x` and returns an array whose extent
        // the callee defines itself. The JVP rule binds one fused call whose callee broadcasts the primal and the
        // tangent over the same extent, so the call declares one local extent that the output tangent shares with the
        // primal output.
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Auto).unwrap()]).unwrap();
        let sharding = Sharding::replicated(logical_mesh, 0);
        let value_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32).with_sharding(sharding.clone()).unwrap());
        let size_type = ArrayIrType::Array(ArrayType::scalar(DataType::I64).with_sharding(sharding).unwrap());
        let rows = DimensionType::from(DimensionVariable::new("rows", DimensionBounds::non_negative(Some(8)).unwrap()));
        let callee = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let value = builder.add_input(value_type.clone());
            let size = builder.add_input(size_type.clone());
            let dimension = builder
                .add_instruction(DimensionFromScalarOperation::new(rows), Vec::new(), vec![size], None)
                .unwrap()[0];
            let broadcast = builder
                .add_instruction(DynamicBroadcastOperation::new(Vec::new()), Vec::new(), vec![value, dimension], None)
                .unwrap()[0];
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![broadcast], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let operation = CallOperation::new(0)
            .with_callee_type_signature(&callee.input_types(), &callee.output_types())
            .unwrap();
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let callee = builder.import_region(callee.entry_region_ref());
        let inputs = vec![builder.add_input(value_type), builder.add_input(size_type)];
        let output = builder.add_instruction(TestOperation::Call(operation), vec![callee], inputs, None).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let jvp = program.jvp().unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:f32[][sharding={mesh<['x'=1:auto]>, []}], %1:i64[][sharding={mesh<['x'=1:auto]>, []}], %2:f32[][sharding={mesh<['x'=1:auto]>, []}] .
                let %3:f32[rows][sharding={mesh<['x'=1:auto]>, [{}]}], %4:f32[rows][sharding={mesh<['x'=1:auto]>, [{}]}] = call [
                    output_types=[f32[rows][sharding={mesh<['x'=1:auto]>, [{}]}], f32[rows][sharding={mesh<['x'=1:auto]>, [{}]}]],
                    derived_input_shardings=[_, _, _],
                ] %0 %1 %2 [
                    callee={
                        lambda %0:f32[][sharding={mesh<['x'=1:auto]>, []}], %1:i64[][sharding={mesh<['x'=1:auto]>, []}], %2:f32[][sharding={mesh<['x'=1:auto]>, []}] .
                        let %3:dimension<rows ∈ [0, 8)> = dimension_from_scalar [bounds=[0, 8)] %1
                            %4:f32[rows][sharding={mesh<['x'=1:auto]>, [{}]}] = broadcast [output_axes=[]] %0 %3
                            %5:f32[rows][sharding={mesh<['x'=1:auto]>, [{}]}] = broadcast [output_axes=[]] %2 %3
                        in (%4, %5)
                    },
                ]
                in (%3, %4)"},
        );
        let outputs = jvp.output_types().into_owned();
        assert_eq!(outputs[1], outputs[0].tangent().unwrap());

        // A linearization binds a primal call and a tangent call instead, so the output tangent shares the extent of
        // the primal output only because that extent reaches the tangent call as a residual. The primal call returns
        // the residual under an identity of its own, at which the callee of the tangent call is instantiated.
        let linearization = program.linearize().unwrap();
        assert_eq!(
            format!("{}\n{}", linearization.primal(), linearization.tangent()),
            indoc! {"
                lambda %0:f32[][sharding={mesh<['x'=1:auto]>, []}], %1:i64[][sharding={mesh<['x'=1:auto]>, []}] .
                let %2:f32[rows][sharding={mesh<['x'=1:auto]>, [{}]}], %3:dimension<rows ∈ [0, 8)> = call [
                    output_types=[f32[rows][sharding={mesh<['x'=1:auto]>, [{}]}], dimension<rows ∈ [0, 8)>],
                    derived_input_shardings=[_, _],
                ] %0 %1 [
                    callee={
                        lambda %0:f32[][sharding={mesh<['x'=1:auto]>, []}], %1:i64[][sharding={mesh<['x'=1:auto]>, []}] .
                        let %2:dimension<rows ∈ [0, 8)> = dimension_from_scalar [bounds=[0, 8)] %1
                            %3:f32[rows][sharding={mesh<['x'=1:auto]>, [{}]}] = broadcast [output_axes=[]] %0 %2
                        in (%3, %2)
                    },
                ]
                in (%2, %3)
                lambda %0:f32[][sharding={mesh<['x'=1:auto]>, []}], %1:dimension<rows ∈ [0, 8)> .
                let %2:f32[rows][sharding={mesh<['x'=1:auto]>, [{}]}] = call [derived_input_shardings=[_]] %0 %1 [
                    callee={
                        lambda %0:f32[][sharding={mesh<['x'=1:auto]>, []}], %1:dimension<rows ∈ [0, 8)> .
                        let %2:f32[rows][sharding={mesh<['x'=1:auto]>, [{}]}] = broadcast [output_axes=[]] %0 %1
                        in (%2)
                    },
                ]
                in (%2)"},
        );
    }

    #[test]
    fn test_call_operation_jvp_shares_callees_of_repeated_calls() {
        let callee = residual_demand_leaf_callee();
        let repeated = |count: usize, policy: Option<ResidualPolicyReference<ArrayIrType>>| {
            residual_demand_program(policy, |builder, weights, bias| {
                (0..count).fold(weights, |hidden, _| add_call(builder, &callee, vec![hidden, bias]))
            })
        };
        let inlined = |count: usize| {
            residual_demand_program(None, |builder, weights, bias| {
                (0..count).fold(weights, |hidden, _| add_residual_demand_leaf(builder, hidden, bias))
            })
        };

        // Differentiating `w` as well as `x` differentiates every call. Linearization binds one primal call and one
        // tangent call of the shared linearized callee per call site, and the tangent call mixes unknown tangents with
        // known residuals. Every call shares one primal callee and one tangent callee, so the programs grow by one call
        // per call site, while the residuals grow with the number of calls as in JAX.
        assert_eq!(
            [1, 2, 4, 8].map(|count| residual_demand_statistics(&repeated(count, None), &[0, 1])),
            [[7, 9, 12], [12, 10, 13], [22, 12, 15], [42, 16, 19]],
        );
        let policy = ResidualPolicyReference::new(DotsSavable);
        assert_eq!(
            [1, 2, 4, 8].map(|count| residual_demand_statistics(&repeated(count, Some(policy.clone())), &[0, 1])),
            [[4, 7, 22], [5, 8, 24], [7, 10, 28], [11, 14, 36]],
        );
        let linearization = repeated(2, None).linearize_with_respect_to(&[0, 1]).unwrap();
        assert_eq!(
            format!("{}\n{}", linearization.primal(), linearization.tangent()),
            indoc! {"
                lambda %0:f32[2, 2], %1:f32[2, 2], %2:f32[2, 2] .
                let %3:f32[2, 2], %4:f32[2, 2], %5:f32[2, 2], %6:f32[2, 2], %7:f32[2, 2], %8:f32[2, 2] = call [derived_input_shardings=[_, _]] %1 %2 [
                    callee=^0={
                        lambda %0:f32[2, 2], %1:f32[2, 2] .
                        let %2:f32[2, 2] = dot [
                            dimensions=(lhs_contracting=[1], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                        ] %0 %1
                            %3:f32[2, 2] = sin %2
                            %4:f32[2, 2] = cos %2
                            %5:f32[2, 2] = mul %3 %4
                            %6:f32[2, 2] = add %5 %0
                            %7:f32[2, 2] = cos %2
                            %8:f32[2, 2] = sin %2
                        in (%6, %1, %7, %8, %4, %3)
                    },
                ]
                    %9:f32[2, 2], %10:f32[2, 2], %11:f32[2, 2], %12:f32[2, 2], %13:f32[2, 2], %14:f32[2, 2] = call [derived_input_shardings=[_, _]] %3 %2 [
                        callee=^0,
                    ]
                    %15:f32[2, 2] = mul %0 %9
                in (%15, %4, %5, %6, %7, %8, %10, %11, %12, %13, %14, %9, %0)
                lambda %0:f32[2, 2], %1:f32[2, 2], %2:f32[2, 2], %3:f32[2, 2], %4:f32[2, 2], %5:f32[2, 2], %6:f32[2, 2], %7:f32[2, 2], %8:f32[2, 2], %9:f32[2, 2], %10:f32[2, 2], %11:f32[2, 2], %12:f32[2, 2], %13:f32[2, 2] .
                let %14:f32[2, 2] = mul %12 %0
                    %15:f32[2, 2] = call [derived_input_shardings=[_]] %1 %2 %3 %4 %5 %6 [
                        callee=^0={
                            lambda %0:f32[2, 2], %1:f32[2, 2], %2:f32[2, 2], %3:f32[2, 2], %4:f32[2, 2], %5:f32[2, 2] .
                            let %6:f32[2, 2] = dot [
                                dimensions=(lhs_contracting=[1], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                            ] %0 %1
                                %7:f32[2, 2] = mul %2 %6
                                %8:f32[2, 2] = mul %4 %7
                                %9:f32[2, 2] = mul %3 %6
                                %10:f32[2, 2] = neg %9
                                %11:f32[2, 2] = mul %5 %10
                                %12:f32[2, 2] = add %8 %11
                                %13:f32[2, 2] = add %12 %0
                            in (%13)
                        },
                    ]
                    %16:f32[2, 2] = call [derived_input_shardings=[_]] %15 %7 %8 %9 %10 %11 [
                        callee=^0,
                    ]
                    %17:f32[2, 2] = mul %13 %16
                    %18:f32[2, 2] = add %14 %17
                in (%18)"},
        );

        // Neither sharing nor rematerialization changes the primal value or the derivatives, which match those of the
        // same function with every call inlined.
        assert_residual_demand_linearizations_match(&repeated(4, None), &inlined(4), &[0, 1], None);
        assert_residual_demand_linearizations_match(&repeated(4, Some(policy)), &inlined(4), &[0, 1], None);
    }

    #[test]
    fn test_call_operation_jvp_shares_callees_of_nested_calls() {
        let nested = |depth: usize, policy: Option<ResidualPolicyReference<ArrayIrType>>| {
            let callee = residual_demand_diamond_callee(depth);
            residual_demand_program(policy, |builder, weights, bias| add_call(builder, &callee, vec![weights, bias]))
        };
        let inlined = |depth: usize| {
            residual_demand_program(None, |builder, weights, bias| {
                add_inlined_residual_demand_diamond(builder, depth, weights, bias)
            })
        };

        // The primal and tangent callees of each level call the shared callees of the level below them twice, so the
        // programs grow by a few instructions per level, while the residuals double with every level as in JAX.
        assert_eq!(
            [1, 2, 3, 4].map(|depth| residual_demand_statistics(&nested(depth, None), &[0, 1])),
            [[13, 14, 16], [25, 19, 20], [49, 24, 24], [97, 29, 28]],
        );
        let policy = ResidualPolicyReference::new(DotsSavable);
        assert_eq!(
            [1, 2, 3, 4].map(|depth| residual_demand_statistics(&nested(depth, Some(policy.clone())), &[0, 1])),
            [[5, 11, 31], [7, 15, 41], [11, 19, 51], [19, 23, 61]],
        );

        // Neither sharing nor rematerialization changes the primal value or the derivatives, which match those of the
        // same function with every level inlined. The shared callees accumulate the cotangents in another order than
        // the inlined levels, so those values agree up to rounding.
        assert_residual_demand_linearizations_match(&nested(3, None), &inlined(3), &[0, 1], Some(1e-6));
        assert_residual_demand_linearizations_match(&nested(3, Some(policy)), &inlined(3), &[0, 1], Some(1e-6));
    }

    #[test]
    fn test_call_operation_transpose() {
        // The callee `x ↦ (x, x)` returns its input twice. With only the second output live, the first output's seed
        // is a structural zero, so the transposed call consumes the second seed alone, which is `x̄`. With both outputs
        // live, the transposed callee receives both seeds and adds them.
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F64));
        let callee = build_program(vec![scalar_type.clone()], |_, inputs| vec![inputs[0], inputs[0]]);
        let program = build_program(vec![scalar_type], |builder, inputs| {
            let callee = builder.import_region(callee.entry_region_ref());
            let call = TestOperation::Call(CallOperation::new(0));
            builder.add_instruction(call, vec![callee], inputs.to_vec(), None).unwrap().to_vec()
        });
        let scalar = |value: f64| f64_value(ArrayType::scalar(DataType::F64), &[value]);

        // Projecting onto the second output keeps the two-output call, so its first seed reaches the call's transpose
        // rule as a structural zero rather than being pruned away beforehand.
        let projected = program.clone().with_outputs(&[1]).unwrap();
        assert_eq!(
            projected.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[], %2:f64[] = call %0 [
                    callee={
                        lambda %0:f64[] .
                        in (%0, %0)
                    },
                ]
                in (%2)
            "}
            .trim_end(),
        );
        let transposed = projected.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = call [derived_input_shardings=[_]] %0 [
                    callee={
                        lambda %0:f64[] .
                        in (%0)
                    },
                ]
                in (%1)
            "}
            .trim_end(),
        );
        assert_eq!(transposed.interpret(vec![scalar(2.0)]), Ok(vec![scalar(2.0)]));

        let transposed = program.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = call [derived_input_shardings=[_, _]] %0 %1 [
                    callee={
                        lambda %0:f64[], %1:f64[] .
                        let %2:f64[] = add %0 %1
                        in (%2)
                    },
                ]
                in (%2)
            "}
            .trim_end(),
        );
        assert_eq!(transposed.interpret(vec![scalar(2.0), scalar(5.0)]), Ok(vec![scalar(7.0)]));
    }

    #[test]
    fn test_call_operation_transpose_ignores_dead_reference_inputs() {
        // The linear callee `ẏ = { ṫr += ẋ; ẋ }` only stores into its reference input and forwards `x`. Under an
        // `Ignore` destination for the reference no later instruction accumulated into its root and the callee never
        // reads it, so the callee is transposed with an `Ignore` destination as well. The transposed call consumes
        // `[ȳ]` only, returns `x̄` only, and neither it nor its callee stages a cotangent reference.
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));
        let callee = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let reference = builder.add_input(reference_type.clone());
            let update = builder.add_input(scalar_type.clone());
            builder
                .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
                .unwrap();
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![update], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let callee_region = builder.import_region(callee.entry_region_ref());
        let reference = builder.add_input(reference_type.clone());
        let update = builder.add_input(scalar_type.clone());
        let output = builder
            .add_instruction(
                TestOperation::Call(CallOperation::new(0)),
                vec![callee_region],
                vec![reference, update],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let destination_kinds = [CotangentDestinationKind::Ignore, CotangentDestinationKind::Return];
        let transposed = program.transpose_with_respect_to(&[0, 1], &destination_kinds).unwrap();
        assert_eq!(transposed.input_types(), vec![scalar_type.clone()]);
        assert_eq!(transposed.output_types(), vec![scalar_type.clone()]);
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:f32[] = call [derived_input_shardings=[_]] %0 [
                    callee={
                        lambda %0:f32[] .
                        in (%0)
                    },
                ]
                in (%1)"},
        );
    }

    #[test]
    fn test_call_operation_transpose_threads_cotangent_references() {
        // The linear callee `ẏ = { ṫr += ẋ; read(ṫr) }` accumulates into its reference input and reads it back.
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));
        let callee = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let reference = builder.add_input(reference_type.clone());
            let update = builder.add_input(scalar_type.clone());
            builder
                .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
                .unwrap();
            let output =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let callee_region = builder.import_region(callee.entry_region_ref());
        let reference = builder.add_input(reference_type.clone());
        let update = builder.add_input(scalar_type.clone());
        let output = builder
            .add_instruction(
                TestOperation::Call(CallOperation::new(0)),
                vec![callee_region],
                vec![reference, update],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        // Transposed under a `Reference` destination for the reference input, the pullback consumes `[ȳ, r̄]` and
        // returns the cotangent reference by identity ahead of `x̄`. The cotangent of the reference input lives in that
        // destination, which the transposed call accumulates into in place, so the pullback forwards its own
        // destination input rather than the call's forwarded reference output, while `x̄` is the call's second output.
        // The transposed callee is exactly the transposition of the inlined callee under the same destinations.
        let destination_kinds = [CotangentDestinationKind::Reference, CotangentDestinationKind::Return];
        let transposed = program.transpose_with_respect_to(&[0, 1], &destination_kinds).unwrap();
        assert_eq!(transposed.input_types(), vec![scalar_type.clone(), reference_type.clone()]);
        assert_eq!(transposed.output_types(), vec![reference_type.clone(), scalar_type]);
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f32[], %1:ref<f32[]> .
                let %2:ref<f32[]>, %3:f32[] = call [derived_input_shardings=[_, _]] %0 %1 [
                    callee={
                        lambda %0:f32[], %1:ref<f32[]> .
                        let () = reference_add_update %1 %0
                            %2:f32[] = reference_read %1
                        in (%1, %2)
                    },
                ]
                in (%1, %3)"},
        );
        let call = &transposed.instructions()[0];
        let transposed_callee = transposed.region_ref(call.regions()[0]).unwrap().to_program();
        let inlined = callee.transpose_with_respect_to(&[0, 1], &destination_kinds).unwrap();
        assert_eq!(transposed_callee.to_string(), inlined.to_string());
    }

    #[test]
    fn test_call_operation_transpose_omits_dead_seeds_beside_cotangent_references() {
        // The callee `ẏ = { ṫr += ẋ; (read(ṫr), ẋ) }` accumulates into its reference input, reads it back, and
        // forwards `x`. Only the read is live, so the forwarded output's seed is a structural zero. Under a `Reference`
        // destination the transposed call consumes `[ȳ₀, r̄]` with no input for the dead seed, accumulates into `r̄` in
        // place, and returns `x̄` read back from it.
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F64));
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F64)));
        let callee = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let reference = builder.add_input(reference_type.clone());
            let update = builder.add_input(scalar_type.clone());
            builder
                .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
                .unwrap();
            let read =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![read, update], vec![Placeholder; 2], vec![Placeholder; 2])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let callee = builder.import_region(callee.entry_region_ref());
        let reference = builder.add_input(reference_type.clone());
        let update = builder.add_input(scalar_type.clone());
        let outputs = builder
            .add_instruction(TestOperation::Call(CallOperation::new(0)), vec![callee], vec![reference, update], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![outputs[0]], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let destination_kinds = [CotangentDestinationKind::Reference, CotangentDestinationKind::Return];
        let transposed = program.transpose_with_respect_to(&[0, 1], &destination_kinds).unwrap();
        assert_eq!(transposed.input_types(), vec![scalar_type.clone(), reference_type.clone()]);
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f64[], %1:ref<f64[]> .
                let %2:ref<f64[]>, %3:f64[] = call [derived_input_shardings=[_, _]] %0 %1 [
                    callee={
                        lambda %0:f64[], %1:ref<f64[]> .
                        let () = reference_add_update %1 %0
                            %2:f64[] = reference_read %1
                        in (%1, %2)
                    },
                ]
                in (%1, %3)
            "}
            .trim_end(),
        );

        // With both outputs dead, every seed is a structural zero, but the live reference destination keeps the call
        // live, so the transposed call consumes `r̄` alone and returns `x̄` read back from it.
        let transposed =
            program.with_outputs(&[]).unwrap().transpose_with_respect_to(&[0, 1], &destination_kinds).unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:ref<f64[]> .
                let %1:ref<f64[]>, %2:f64[] = call [derived_input_shardings=[_]] %0 [
                    callee={
                        lambda %0:ref<f64[]> .
                        let %1:f64[] = reference_read %0
                        in (%0, %1)
                    },
                ]
                in (%0, %2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_call_operation_transpose_omits_disconnected_symbolic_cotangent() {
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent)]));
        let scalar_type = ArrayType::scalar(DataType::F32);
        let mut callee_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        callee_builder.add_input(dynamic_type.clone().into());
        let live = callee_builder.add_input(scalar_type.clone().into());
        let callee = callee_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![live], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let disconnected = builder.add_input(dynamic_type.into());
        let live = builder.add_input(scalar_type.into());
        let callee = builder.import_region(callee.entry_region_ref());
        let outputs = builder
            .add_instruction(TestOperation::Call(CallOperation::new(0)), vec![callee], vec![disconnected, live], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(outputs, vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.residual_count(), 0);
        let pullback = linearization.pullback().unwrap();
        assert_eq!(pullback.cotangents(), &[TransposedProgramCotangent::Zero, TransposedProgramCotangent::Output]);
        assert_eq!(
            pullback.program().to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:f32[] = call [derived_input_shardings=[_]] %0 [
                    callee={
                        lambda %0:f32[] .
                        in (%0)
                    },
                ]
                in (%1)"},
        );
    }

    #[test]
    fn test_call_operation_transpose_omits_dead_dynamic_seeds() {
        // The callee `(x, z) ↦ (x, z)` returns both of its dynamically shaped inputs, and only `z` is live. The dead
        // output's seed is a structural zero of type `f64[dead]`, whose extent no input of the call defines. The
        // transposed call has no input for that seed, so it needs no extent for it. It consumes the live seed alone,
        // `x̄` stays a structural zero, and `z̄` keeps the `live` identity of the caller's cotangent type.
        let dead = DimensionVariable::new("dead", DimensionBounds::new(1, Some(5)).unwrap());
        let live = DimensionVariable::new("live", DimensionBounds::new(1, Some(5)).unwrap());
        let dead_type = ArrayIrType::Array(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(dead)])));
        let live_type = ArrayIrType::Array(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(live)])));
        let callee = build_program(vec![dead_type.clone(), live_type.clone()], |_, inputs| inputs.to_vec());
        let program = build_program(vec![dead_type, live_type.clone()], |builder, inputs| {
            let callee = builder.import_region(callee.entry_region_ref());
            let call = TestOperation::Call(CallOperation::new(0));
            vec![builder.add_instruction(call, vec![callee], inputs.to_vec(), None).unwrap()[1]]
        });
        let transposed = program.entry_region_ref().transpose(&[0, 1], &[], &[]).unwrap();
        assert_eq!(transposed.cotangents(), &[TransposedProgramCotangent::Zero, TransposedProgramCotangent::Output]);
        let pullback = transposed.into_program();
        assert_eq!(pullback.output_types(), vec![live_type]);
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f64[live] .
                let %1:f64[live] = call [derived_input_shardings=[_]] %0 [
                    callee={
                        lambda %0:f64[live] .
                        in (%0)
                    },
                ]
                in (%1)
            "}
            .trim_end(),
        );
        let seed = f64_value(ArrayType::new_static(DataType::F64, [3]), &[1.0, 2.0, 3.0]);
        assert_eq!(pullback.interpret(vec![seed.clone()]), Ok(vec![seed]));
    }

    #[test]
    fn test_call_operation_transpose_runs_backward_effects_without_live_seeds() {
        // The callee maps its input through a reverse mode carrier whose backward rule prints its seed and returns a
        // zero. With the call's output dead, its only seed is a structural zero, yet the backward effect must still
        // run. The transposed call has no inputs, and its callee prints a materialized zero seed.
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F64));
        let backward = build_program(vec![scalar_type.clone()], |builder, inputs| {
            let print = ArrayOperation::from(PrintOperation::<ArrayType>::new("backward"));
            builder.add_instruction(print, Vec::new(), vec![inputs[0]], None).unwrap();
            let zero = TestOperation::Zero(ZeroOperation::new(ArrayType::scalar(DataType::F64)));
            builder.add_instruction(zero, Vec::new(), Vec::new(), None).unwrap().to_vec()
        });
        let callee = build_program(vec![scalar_type.clone()], |builder, inputs| {
            let backward = builder.import_region(backward.entry_region_ref());
            let carrier = CompositeTransposeOperation::<TestValue, TestOperation>::from_backward_region(
                0,
                vec![scalar_type.clone()],
                vec![scalar_type.clone()],
            );
            builder.add_instruction(carrier, vec![backward], inputs.to_vec(), None).unwrap().to_vec()
        });
        let program = build_program(vec![scalar_type], |builder, inputs| {
            let callee = builder.import_region(callee.entry_region_ref());
            let call = TestOperation::Call(CallOperation::new(0));
            builder.add_instruction(call, vec![callee], inputs.to_vec(), None).unwrap();
            Vec::new()
        });
        let transposed = program.entry_region_ref().transpose(&[0], &[], &[]).unwrap();
        assert_eq!(transposed.cotangents(), &[TransposedProgramCotangent::Zero]);
        let pullback = transposed.into_program();
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda  .
                let () = call [derived_input_shardings=[]] [
                    callee={
                        lambda  .
                        let %0:f64[] = zero [type=f64[]]
                            %1:f64[] = print [label=backward] %0
                            %2:f64[] = zero [type=f64[]]
                        in ()
                    },
                ]
                in ()"},
        );
        assert_eq!(pullback.interpret(Vec::new()), Ok(Vec::new()));
    }

    #[test]
    fn test_call_operation_transpose_keeps_dead_seeds_structural_in_nested_scans() {
        // The callee runs the scan `(c, x) ↦ (c + x, c)` over three items and returns its final carry and its stacked
        // outputs, of which only the final carry is live. The dead stack's seed reaches the scan inside the callee as a
        // structural zero, so neither the transposed call nor its reversed scan receives a zero stack.
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F64));
        let stack_type = ArrayIrType::Array(ArrayType::new_static(DataType::F64, [3]));
        let body = build_program(
            vec![ArrayType::scalar(DataType::I64).into(), scalar_type.clone(), scalar_type.clone()],
            |builder, inputs| {
                let next = builder
                    .add_instruction(
                        ArrayOperation::from(AddOperation::new()),
                        Vec::new(),
                        vec![inputs[1], inputs[2]],
                        None,
                    )
                    .unwrap()[0];
                vec![next, inputs[1]]
            },
        );
        let callee = build_program(vec![scalar_type.clone(), stack_type.clone()], |builder, inputs| {
            let body = builder.import_region(body.entry_region_ref());
            let scan = ScanOperation::<ArrayIrType>::new(1, 3);
            builder.add_instruction(scan, vec![body], inputs.to_vec(), None).unwrap().to_vec()
        });
        let program = build_program(vec![scalar_type, stack_type], |builder, inputs| {
            let callee = builder.import_region(callee.entry_region_ref());
            let call = TestOperation::Call(CallOperation::new(0));
            vec![builder.add_instruction(call, vec![callee], inputs.to_vec(), None).unwrap()[0]]
        });
        let transposed = program.transpose_with_respect_to(&[0, 1], &[]).unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[], %2:f64[3] = call [derived_input_shardings=[_]] %0 [
                    callee={
                        lambda %0:f64[] .
                        let %1:f64[], %2:f64[3] = scan [carry_count=1, iteration_count=3, reverse=true] %0 [
                            body={
                                lambda %0:i64[], %1:f64[] .
                                in (%1, %1)
                            },
                        ]
                        in (%1, %2)
                    },
                ]
                in (%1, %2)
            "}
            .trim_end(),
        );
        assert_eq!(
            transposed.interpret(vec![f64_value(ArrayType::scalar(DataType::F64), &[2.0])]),
            Ok(vec![
                f64_value(ArrayType::scalar(DataType::F64), &[2.0]),
                f64_value(ArrayType::new_static(DataType::F64, [3]), &[2.0, 2.0, 2.0]),
            ]),
        );
    }

    #[test]
    fn test_call_operation_transpose_keeps_dead_seeds_structural_in_nested_conditions() {
        // The callee selects on `c > 0` between the branches `x ↦ (x, x + x)` and `x ↦ (x + x, x)` and only the first
        // output is live. The dead output's seed reaches the condition inside the callee as a structural zero, so each
        // transposed branch consumes the live seed alone, giving `x̄ = ȳ₀` when `c > 0` and `x̄ = 2ȳ₀` otherwise.
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F64));
        let branch = |doubled_output: usize| {
            build_program(vec![scalar_type.clone()], |builder, inputs| {
                let doubled = builder
                    .add_instruction(
                        ArrayOperation::from(AddOperation::new()),
                        Vec::new(),
                        vec![inputs[0], inputs[0]],
                        None,
                    )
                    .unwrap()[0];
                if doubled_output == 0 { vec![doubled, inputs[0]] } else { vec![inputs[0], doubled] }
            })
        };
        let callee = build_program(vec![scalar_type.clone(); 2], |builder, inputs| {
            let true_branch = builder.import_region(branch(1).entry_region_ref());
            let false_branch = builder.import_region(branch(0).entry_region_ref());
            let zero = builder
                .add_instruction(
                    ArrayOperation::from(ZeroLikeOperation::<ArrayType>::new()),
                    Vec::new(),
                    vec![inputs[0]],
                    None,
                )
                .unwrap()[0];
            let compare = ArrayOperation::Compare(CompareOperation::<ArrayType>::new(ComparisonDirection::GreaterThan));
            let predicate = builder.add_instruction(compare, Vec::new(), vec![inputs[0], zero], None).unwrap()[0];
            let condition = ConditionOperation::<ArrayIrType>::new();
            builder
                .add_instruction(condition, vec![true_branch, false_branch], vec![predicate, inputs[1]], None)
                .unwrap()
                .to_vec()
        });
        let program = build_program(vec![scalar_type.clone(); 2], |builder, inputs| {
            let callee = builder.import_region(callee.entry_region_ref());
            let call = TestOperation::Call(CallOperation::new(0));
            vec![builder.add_instruction(call, vec![callee], inputs.to_vec(), None).unwrap()[0]]
        });
        let transposed = program.transpose_with_respect_to(&[1], &[]).unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = call [derived_input_shardings=[_, _]] %0 %1 [
                    callee={
                        lambda %0:f64[], %1:f64[] .
                        let %2:f64[] = zero_like %1
                            %3:bool[] = compare [direction=GreaterThan] %1 %2
                            %4:f64[] = condition %3 %0 [
                                true={
                                    lambda %0:f64[] .
                                    in (%0)
                                },
                                false={
                                    lambda %0:f64[] .
                                    let %1:f64[] = add %0 %0
                                    in (%1)
                                },
                            ]
                        in (%4)
                    },
                ]
                in (%2)
            "}
            .trim_end(),
        );
        let scalar = |value: f64| f64_value(ArrayType::scalar(DataType::F64), &[value]);
        for (selector, expected) in [(1.0, 3.0), (-1.0, 6.0)] {
            assert_eq!(transposed.interpret(vec![scalar(3.0), scalar(selector)]), Ok(vec![scalar(expected)]));
        }
    }

    #[test]
    fn test_call_operation_transpose_keeps_dead_seeds_structural_across_rematerializations() {
        // A rematerialization inside a `call` callee and a `call` operation inside a rematerialization both map
        // `x` to `(x, x + x)`, and only `x + x` is live. In both nestings the dead output's seed stays a structural
        // zero across both boundaries. The outer transposed region consumes the live seed alone, the inner one receives
        // it alone, and `x̄ = ȳ₁ + ȳ₁`.
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F64));
        let doubling = build_program(vec![scalar_type.clone()], |builder, inputs| {
            let doubled = builder
                .add_instruction(
                    ArrayOperation::from(AddOperation::new()),
                    Vec::new(),
                    vec![inputs[0], inputs[0]],
                    None,
                )
                .unwrap()[0];
            vec![inputs[0], doubled]
        });
        let rematerialize =
            || TestOperation::from(RematerializeOperation::new(ResidualPolicyReference::new(NothingSavable)));
        let nest = |outer: TestOperation, inner: TestOperation| {
            let inner_program = build_program(vec![scalar_type.clone()], |builder, inputs| {
                let region = builder.import_region(doubling.entry_region_ref());
                builder.add_instruction(inner, vec![region], inputs.to_vec(), None).unwrap().to_vec()
            });
            build_program(vec![scalar_type.clone()], |builder, inputs| {
                let region = builder.import_region(inner_program.entry_region_ref());
                vec![builder.add_instruction(outer, vec![region], inputs.to_vec(), None).unwrap()[1]]
            })
        };
        let scalar = |value: f64| f64_value(ArrayType::scalar(DataType::F64), &[value]);

        let program = nest(TestOperation::Call(CallOperation::new(0)), rematerialize());
        let transposed = program.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = call [derived_input_shardings=[_]] %0 [
                    callee={
                        lambda %0:f64[] .
                        let %1:f64[] = rematerialize %0 [
                            body={
                                lambda %0:f64[] .
                                let %1:f64[] = add %0 %0
                                in (%1)
                            },
                        ]
                        in (%1)
                    },
                ]
                in (%1)
            "}
            .trim_end(),
        );
        assert_eq!(transposed.interpret(vec![scalar(2.0)]), Ok(vec![scalar(4.0)]));

        let program = nest(rematerialize(), TestOperation::Call(CallOperation::new(0)));
        let transposed = program.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = rematerialize %0 [
                    body={
                        lambda %0:f64[] .
                        let %1:f64[] = call [derived_input_shardings=[_]] %0 [
                            callee={
                                lambda %0:f64[] .
                                let %1:f64[] = add %0 %0
                                in (%1)
                            },
                        ]
                        in (%1)
                    },
                ]
                in (%1)
            "}
            .trim_end(),
        );
        assert_eq!(transposed.interpret(vec![scalar(2.0)]), Ok(vec![scalar(4.0)]));
    }

    #[test]
    fn test_call_operation_transpose_keeps_the_input_overrides_of_returned_cotangents() {
        // The cotangent of the first input of a call keeps the split of its input override, as JAX keeps the input
        // sharding of a `jit` function on the cotangent of that input, whether the second input has a cotangent or a
        // structurally zero one, which the transposed callee does not return.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let device_mesh = DeviceMesh::new(mesh.clone(), vec![Device::new(0, 0), Device::new(1, 0)]).unwrap();
        let split = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated = Sharding::replicated(mesh, 1);
        let vector_type = ArrayIrType::Array(f32_vector_type(4));
        let callee = Arc::new(build_program(vec![vector_type.clone(); 2], |_, inputs| inputs.to_vec()));
        let seed = f32_value(f32_vector_type(4), &[1.0; 4]);
        let zero = f32_value(f32_vector_type(4), &[0.0; 4]);
        let mut renderings = Vec::new();
        for sums_both_outputs in [false, true] {
            let program = build_program(vec![vector_type.clone(); 2], |builder, inputs| {
                let callee = builder.intern_callee(&callee, None).unwrap();
                let call = CallOperation::new(0).with_boundary(
                    Some(&device_mesh),
                    vec![Some(split.clone()), Some(replicated.clone())],
                    Vec::new(),
                );
                let outputs = builder
                    .add_instruction(TestOperation::Call(call), vec![callee], inputs.to_vec(), None)
                    .unwrap()
                    .to_vec();
                if sums_both_outputs {
                    builder
                        .add_instruction(ArrayOperation::from(AddOperation::new()), Vec::new(), outputs, None)
                        .unwrap()
                        .to_vec()
                } else {
                    vec![outputs[0]]
                }
            });
            let transposed = program.transpose_with_respect_to(&[0, 1], &[]).unwrap();
            renderings.push(transposed.to_string());
            let second = if sums_both_outputs { seed.clone() } else { zero.clone() };
            assert_eq!(transposed.interpret(vec![seed.clone()]), Ok(vec![seed.clone(), second]));
        }
        assert_eq!(
            renderings.join("\n"),
            indoc! {"
                lambda %0:f32[4] .
                let %1:f32[4] = call [
                    output_shardings=[{mesh<['x'=2:auto]>, [{'x'}]}],
                    derived_input_shardings=[_],
                    devices=[0, 1],
                ] %0 [
                    callee={
                        lambda %0:f32[4] .
                        in (%0)
                    },
                ]
                    %2:f32[4] = zero [type=f32[4]]
                in (%1, %2)
                lambda %0:f32[4] .
                let %1:f32[4], %2:f32[4] = call [
                    output_shardings=[{mesh<['x'=2:auto]>, [{'x'}]}, {mesh<['x'=2:auto]>, [{}]}],
                    derived_input_shardings=[_, _],
                    devices=[0, 1],
                ] %0 %0 [
                    callee={
                        lambda %0:f32[4], %1:f32[4] .
                        in (%0, %1)
                    },
                ]
                in (%1, %2)"},
        );
    }

    #[test]
    fn test_call_operation_transpose_places_cotangents_at_the_cotangent_shardings_of_overrides() {
        // A call of a function that moves its scalar input to device memory has the same pullback when its boundary is
        // stated as an input override, an output override, or both, as when it is inferred, for every reduction state
        // of its argument over explicit axes, which its type records. The cotangent of a value with a pending sum has a
        // completed one and the reverse (refer to `Sharding::cotangent`), so the transposed call places each cotangent
        // that crosses the boundary at the cotangent sharding of the override of its primal. A pending maximum defines
        // no cotangent, so its pullback is rejected alike with and without overrides.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let device_mesh = DeviceMesh::new(mesh.clone(), vec![Device::new(0, 0), Device::new(1, 0)]).unwrap();
        let replicated = Sharding::replicated(mesh, 0);
        let pending = replicated.clone().with_unreduced_axes(["x"]).unwrap();
        let states = [
            replicated.clone(),
            pending.clone(),
            replicated.with_reduced_axes(["x"]).unwrap(),
            pending.clone().with_unreduced_kind(UnreducedKind::Max).unwrap(),
        ];
        let pullback = |sharding: &Sharding, input_override: bool, output_override: bool| {
            let r#type = ArrayIrType::Array(ArrayType::scalar(DataType::F32).with_sharding(sharding.clone()).unwrap());
            let callee = Arc::new(build_program(vec![r#type.clone()], |builder, inputs| {
                let target = PlacementRequest::InPlace { memory: Some(Memory::Device), layout: None };
                let place = ArrayOperation::from(PlaceOperation::new(target, None));
                builder.add_instruction(place, Vec::new(), inputs.to_vec(), None).unwrap().to_vec()
            }));
            let program = build_program(vec![r#type], |builder, inputs| {
                let callee = builder.intern_callee(&callee, None).unwrap();
                let call = CallOperation::new(0).with_boundary(
                    Some(&device_mesh),
                    vec![input_override.then(|| sharding.clone())],
                    vec![output_override.then(|| sharding.clone())],
                );
                let call = TestOperation::Call(call);
                builder.add_instruction(call, vec![callee], inputs.to_vec(), None).unwrap().to_vec()
            });
            program.transpose_with_respect_to(&[0], &[]).map_err(|error| error.to_string())
        };
        // Each row shows the outcome of the inferred boundary, so that a row cannot agree merely by failing alike.
        let describe = |outcome: &Result<Vec<ArrayIrType>, String>| match outcome {
            Ok(types) => format!("returns {}", types.iter().map(ToString::to_string).collect::<Vec<_>>().join(", ")),
            Err(error) => format!("fails with `{error}`"),
        };
        let mut outcomes = Vec::new();
        for sharding in &states {
            let reference = pullback(sharding, false, false).map(|pullback| pullback.output_types().into_owned());
            for (input_override, output_override) in [(true, false), (false, true), (true, true)] {
                let derivative = pullback(sharding, input_override, output_override)
                    .map(|pullback| pullback.output_types().into_owned());
                let outcome = match derivative == reference {
                    true => format!("agrees and {}", describe(&reference)),
                    false => format!("{} instead of {}", describe(&derivative), describe(&reference)),
                };
                outcomes.push(format!("{sharding}, {input_override}, {output_override}: {outcome}"));
            }
        }
        assert_eq!(
            outcomes.join("\n"),
            indoc! {"
                {mesh<['x'=2:explicit]>, []}, true, false: agrees and returns f32[][sharding={mesh<['x'=2:explicit]>, []}]
                {mesh<['x'=2:explicit]>, []}, false, true: agrees and returns f32[][sharding={mesh<['x'=2:explicit]>, []}]
                {mesh<['x'=2:explicit]>, []}, true, true: agrees and returns f32[][sharding={mesh<['x'=2:explicit]>, []}]
                {mesh<['x'=2:explicit]>, [], unreduced={'x'}}, true, false: agrees and returns f32[][sharding={mesh<['x'=2:explicit]>, [], reduced={'x'}}]
                {mesh<['x'=2:explicit]>, [], unreduced={'x'}}, false, true: agrees and returns f32[][sharding={mesh<['x'=2:explicit]>, [], reduced={'x'}}]
                {mesh<['x'=2:explicit]>, [], unreduced={'x'}}, true, true: agrees and returns f32[][sharding={mesh<['x'=2:explicit]>, [], reduced={'x'}}]
                {mesh<['x'=2:explicit]>, [], reduced={'x'}}, true, false: agrees and returns f32[][sharding={mesh<['x'=2:explicit]>, [], unreduced={'x'}}]
                {mesh<['x'=2:explicit]>, [], reduced={'x'}}, false, true: agrees and returns f32[][sharding={mesh<['x'=2:explicit]>, [], unreduced={'x'}}]
                {mesh<['x'=2:explicit]>, [], reduced={'x'}}, true, true: agrees and returns f32[][sharding={mesh<['x'=2:explicit]>, [], unreduced={'x'}}]
                {mesh<['x'=2:explicit]>, [], unreduced={'x'}, unreduced_kind=max}, true, false: agrees and fails with `type f32[][sharding={mesh<['x'=2:explicit]>, [], unreduced={'x'}, unreduced_kind=max}] does not define a cotangent representation`
                {mesh<['x'=2:explicit]>, [], unreduced={'x'}, unreduced_kind=max}, false, true: agrees and fails with `type f32[][sharding={mesh<['x'=2:explicit]>, [], unreduced={'x'}, unreduced_kind=max}] does not define a cotangent representation`
                {mesh<['x'=2:explicit]>, [], unreduced={'x'}, unreduced_kind=max}, true, true: agrees and fails with `type f32[][sharding={mesh<['x'=2:explicit]>, [], unreduced={'x'}, unreduced_kind=max}] does not define a cotangent representation`"},
        );
        // The transposed call of a call whose overrides hold a pending sum places the cotangents at a completed one.
        assert_eq!(
            pullback(&pending, true, true).unwrap().to_string(),
            indoc! {"
                lambda %0:f32[][sharding={mesh<['x'=2:explicit]>, [], reduced={'x'}}] .
                let %1:f32[][sharding={mesh<['x'=2:explicit]>, [], reduced={'x'}}] = call [
                    input_shardings=[{mesh<['x'=2:explicit]>, [], reduced={'x'}}],
                    output_shardings=[{mesh<['x'=2:explicit]>, [], reduced={'x'}}],
                    derived_input_shardings=[_],
                    devices=[0, 1],
                ] %0 [
                    callee={
                        lambda %0:f32[][sharding={mesh<['x'=2:explicit]>, [], reduced={'x'}}] .
                        let %1:f32[][sharding={mesh<['x'=2:explicit]>, [], reduced={'x'}}] = place [target=in_place(memory=Device), may_alias=false] %0
                        in (%1)
                    },
                ]
                in (%1)"},
        );
    }

    #[test]
    fn test_call_operation_transpose_omits_zero_space_seeds() {
        // The predicate output's cotangent is a zero space that arrives as a structural zero, so the transposed callee
        // has no input for it and the transposed call consumes the value output's seed alone.
        let value_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]));
        let predicate_type = ArrayType::new(DataType::Boolean, Shape::new(vec![Dimension::Static(4)]));
        let value_program_type = ArrayIrType::Array(value_type.clone());
        let source = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let value = builder.add_input(value_program_type.clone());
            let predicate = Array::from_elements(predicate_type.clone(), &[true, false, true, false]).unwrap();
            let predicate = builder.add_constant(ArrayIrValue::Array(predicate));
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(
                    vec![value, predicate],
                    vec![Placeholder],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        let transposed = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let value_cotangent = builder.add_input(value_program_type.clone());
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![value_cotangent], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let driver = TestTranspositionDriver { source, transposed };
        let mut context = TranspositionContext::new(TracingContext::<TestValue, TestOperation>::new());
        let value_cotangent = context.input(value_program_type.clone());
        let accumulators = [context.cotangent_accumulator(value_program_type.cotangent().unwrap(), true)];
        CallOperation::<ArrayIrType>::new(0)
            .transpose(
                &mut context,
                &driver,
                &[PartialValue::Unknown(value_program_type.clone())],
                &[
                    MaybeZero::Value(value_cotangent.clone()),
                    MaybeZero::Zero(ArrayIrType::Array(predicate_type.cotangent().unwrap())),
                ],
                &accumulators,
            )
            .unwrap();
        let contributions = context.take_cotangents(&accumulators).unwrap();

        let [MaybeZero::Value(contribution)] = &contributions[..] else {
            panic!("expected one cotangent contribution");
        };
        assert_eq!(contribution.r#type().as_ref(), &value_program_type);
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![contribution.atom_id().unwrap()],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[4] .
                let %1:f32[4] = call [derived_input_shardings=[_]] %0 [
                    callee={
                        lambda %0:f32[4] .
                        in (%0)
                    },
                ]
                in (%1)"},
        );
    }

    #[test]
    fn test_call_operation_transpose_uses_cotangent_descriptors_for_zero_cotangents() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let tangent_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(
                Sharding::new(mesh, vec![ShardingDimension::replicated()])
                    .unwrap()
                    .with_unreduced_axes(["x"])
                    .unwrap(),
            )
            .unwrap();
        let expected = ArrayIrType::Array(tangent_type.cotangent().unwrap());
        let tangent_type = ArrayIrType::Array(tangent_type);
        // Zero cotangents skip a pure callee only after its effects have been inspected. Supply the real boundary of
        // the identity region rather than an empty driver that cannot answer that query.
        let mut source_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let input = source_builder.add_input(tangent_type.clone());
        let source = source_builder.build(vec![input], vec![Placeholder], vec![Placeholder]).unwrap();
        let mut transpose_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let input = transpose_builder.add_input(expected.clone());
        let transposed = transpose_builder.build(vec![input], vec![Placeholder], vec![Placeholder]).unwrap();
        let driver = TestTranspositionDriver { source, transposed };
        let mut context = TranspositionContext::new(TracingContext::<TestValue, TestOperation>::new());
        let accumulators = [context.cotangent_accumulator(expected.clone(), true)];
        CallOperation::<ArrayIrType>::new(0)
            .transpose(
                &mut context,
                &driver,
                &[PartialValue::Unknown(tangent_type.clone())],
                &[MaybeZero::Zero(tangent_type.clone())],
                &accumulators,
            )
            .unwrap();
        let cotangents = context.take_cotangents(&accumulators).unwrap();
        assert!(matches!(&cotangents[..], [MaybeZero::Zero(actual)] if actual == &expected));

        let known = context.input(tangent_type.clone());
        let accumulators = [context.cotangent_accumulator(expected.clone(), true)];
        CallOperation::<ArrayIrType>::new(0)
            .transpose(
                &mut context,
                &driver,
                &[PartialValue::Known(known)],
                &[MaybeZero::Zero(tangent_type)],
                &accumulators,
            )
            .unwrap();
        let cotangents = context.take_cotangents(&accumulators).unwrap();
        assert!(matches!(&cotangents[..], [MaybeZero::Zero(actual)] if actual == &expected));
    }

    #[test]
    fn test_region_placement_requests_new() {
        let vector_type = ArrayIrType::Array(f32_vector_type(4));
        let device = Device::new(1, 0);
        let device_mesh = DeviceMesh::new(LogicalMesh::single_device(), vec![device]).unwrap();
        let place = |target: PlacementRequest| ArrayOperation::from(PlaceOperation::new(target, None));
        let memory = PlacementRequest::InPlace { memory: Some(Memory::Device), layout: None };

        // A target that only moves a value to another memory names no devices, and a device target names the canonical
        // mesh of that device.
        let program = build_program(vec![vector_type.clone()], |builder, inputs| {
            let moved = builder.add_instruction(place(memory.clone()), Vec::new(), vec![inputs[0]], None).unwrap()[0];
            let target = PlacementRequest::from(device);
            builder.add_instruction(place(target), Vec::new(), vec![moved], None).unwrap().to_vec()
        });
        let summary = RegionPlacementRequests::new(program.entry_region_ref());
        assert_eq!(
            summary.requests(),
            &[
                (RequestedPlacement::Memory, PLACE_OPERATION_NAME),
                (RequestedPlacement::Devices(device_mesh.clone()), PLACE_OPERATION_NAME),
            ],
        );
        assert!(summary.has_requests(false));
        let program = build_program(vec![vector_type.clone()], |builder, inputs| {
            builder.add_instruction(place(memory.clone()), Vec::new(), vec![inputs[0]], None).unwrap().to_vec()
        });
        let summary = RegionPlacementRequests::new(program.entry_region_ref());
        assert!(summary.has_requests(true));
        assert!(!summary.has_requests(false));

        // A request that no output uses is not a request of the program.
        let program = build_program(vec![vector_type.clone()], |builder, inputs| {
            builder
                .add_instruction(place(PlacementRequest::from(device)), Vec::new(), vec![inputs[0]], None)
                .unwrap();
            inputs.to_vec()
        });
        assert_eq!(RegionPlacementRequests::new(program.entry_region_ref()).requests(), &[]);

        // A call that records a mesh names its devices when its callee requests a placement that names devices, and
        // names none when its callee only moves a value to another memory.
        let call_program = |target: PlacementRequest| {
            let callee = build_program(vec![vector_type.clone()], |builder, inputs| {
                builder.add_instruction(place(target), Vec::new(), vec![inputs[0]], None).unwrap().to_vec()
            });
            build_program(vec![vector_type.clone()], |builder, inputs| {
                let callee = builder.import_program(callee);
                let call = CallOperation::new(0).with_boundary(Some(&device_mesh), Vec::new(), Vec::new());
                builder
                    .add_instruction(TestOperation::Call(call), vec![callee], inputs.to_vec(), None)
                    .unwrap()
                    .to_vec()
            })
        };
        assert_eq!(
            RegionPlacementRequests::new(call_program(PlacementRequest::from(device)).entry_region_ref()).requests(),
            &[
                (RequestedPlacement::Devices(device_mesh.clone()), CALL_OPERATION_NAME),
                (RequestedPlacement::Devices(device_mesh.clone()), PLACE_OPERATION_NAME),
            ],
        );
        assert_eq!(
            RegionPlacementRequests::new(call_program(memory).entry_region_ref()).requests(),
            &[(RequestedPlacement::Memory, PLACE_OPERATION_NAME)],
        );

        // The body of a scan with zero iterations never runs, so its requests do not count toward the program.
        let body =
            build_program(vec![ArrayType::scalar(DataType::I64).into(), vector_type.clone()], |builder, inputs| {
                builder
                    .add_instruction(place(PlacementRequest::from(device)), Vec::new(), vec![inputs[1]], None)
                    .unwrap()
                    .to_vec()
            });
        let scan_program = |length: usize| {
            build_program(vec![vector_type.clone()], |builder, inputs| {
                let body = builder.import_program(body.clone());
                let scan = ScanOperation::<ArrayIrType>::new(1, length);
                builder.add_instruction(scan, vec![body], inputs.to_vec(), None).unwrap().to_vec()
            })
        };
        assert_eq!(RegionPlacementRequests::new(scan_program(0).entry_region_ref()).requests(), &[]);
        assert_eq!(
            RegionPlacementRequests::new(scan_program(1).entry_region_ref()).requests(),
            &[(RequestedPlacement::Devices(device_mesh), PLACE_OPERATION_NAME)],
        );
    }

    #[test]
    fn test_region_placement_requests_for_target() {
        // A request in the regions that a backend omits for a target platform is not a request of the program for that
        // platform, and the summary of each platform is retained separately.
        let vector_type = ArrayIrType::Array(f32_vector_type(4));
        let device = Device::new(1, 0);
        let device_mesh = DeviceMesh::new(LogicalMesh::single_device(), vec![device]).unwrap();
        let body =
            build_program(vec![ArrayType::scalar(DataType::I64).into(), vector_type.clone()], |builder, inputs| {
                let place = ArrayOperation::from(PlaceOperation::new(PlacementRequest::from(device), None));
                builder.add_instruction(place, Vec::new(), vec![inputs[1]], None).unwrap().to_vec()
            });
        let program = build_program(vec![vector_type], |builder, inputs| {
            let body = builder.import_program(body);
            let scan = ScanOperation::<ArrayIrType>::new(1, 1);
            builder.add_instruction(scan, vec![body], inputs.to_vec(), None).unwrap().to_vec()
        });
        let omits_scans = |_: RegionRef<'_, TestValue, TestOperation>, instruction: &Instruction<TestOperation>| {
            instruction.operation().projected_payload::<ScanOperation<ArrayIrType>>().is_some()
        };
        let placed = [(RequestedPlacement::Devices(device_mesh), PLACE_OPERATION_NAME)];
        let entry = program.entry_region_ref();
        assert_eq!(RegionPlacementRequests::for_target(entry, "omitting", omits_scans).requests(), &[]);
        assert_eq!(RegionPlacementRequests::for_target(entry, "keeping", |_, _| false).requests(), &placed);
        assert_eq!(RegionPlacementRequests::new(entry).requests(), &placed);
        assert_eq!(RegionPlacementRequests::for_target(entry, "omitting", omits_scans).requests(), &[]);
    }

    #[test]
    fn test_call_type_bindings_fresh() {
        let bounds = DimensionBounds::new(0, Some(4)).unwrap();
        let input = ArrayIrType::from(DimensionType::new("input", bounds));
        let local = ArrayIrType::from(DimensionType::new("local", bounds));
        let inputs = vec![input.clone()];
        let outputs = vec![input.clone(), local.clone(), local.clone()];
        assert_eq!(CallTypeBindings::fresh(&inputs, &inputs).unwrap(), None);

        // One local family is exported once even when several results share it. Separate calls allocate separate
        // families, while forwarded input names stay shared.
        let first = CallTypeBindings::fresh(&inputs, &outputs).unwrap().unwrap();
        let second = CallTypeBindings::fresh(&inputs, &outputs).unwrap().unwrap();
        assert_eq!(first.outputs[0], input);
        assert_eq!(first.outputs[1], first.outputs[2]);
        assert_ne!(first.outputs[1], local);
        assert_ne!(first.outputs[1], second.outputs[1]);
        assert_eq!(first.definitions().len(), 1);
        assert_eq!(first.instantiated(&inputs).unwrap(), first.outputs.as_ref());
        let renaming = first.renaming(&inputs, &outputs).unwrap();
        assert_eq!(local.rename_identities(&renaming).unwrap(), first.outputs[1]);
        assert_eq!(input.rename_identities(&renaming).unwrap(), input);

        // Instantiating a forwarded input in another namespace must not rename the call's local result.
        let actual = ArrayIrType::from(DimensionType::new("actual", bounds));
        let instantiated = first.instantiated(std::slice::from_ref(&actual)).unwrap();
        assert_eq!(instantiated[0], actual);
        assert_eq!(&instantiated[1..], &first.outputs[1..]);
        let retained = first.pruned(&[true], &[true, false, true]).unwrap();
        assert_eq!(retained.outputs.as_ref(), &[input.clone(), first.outputs[2].clone()]);
        assert_eq!(retained.definitions(), first.definitions());
        assert_eq!(
            first.pruned(&[], &[true; 3]),
            Err(TypeError::invalid(format!(
                "`{CALL_OPERATION_NAME}` result declaration pruning masks have incompatible arity",
            ))),
        );

        // A declaration cannot export the callee's private name verbatim or rename a forwarded input.
        assert_eq!(
            CallTypeBindings::with_outputs(&inputs, &outputs, outputs.clone()),
            Err(TypeError::invalid(format!(
                "`{CALL_OPERATION_NAME}` local result declarations must use fresh identities",
            ))),
        );
        let other = ArrayIrType::from(DimensionType::new("other", bounds));
        assert_eq!(
            CallTypeBindings::with_outputs(&inputs, &inputs, vec![other]),
            Err(TypeError::invalid(format!(
                "`{CALL_OPERATION_NAME}` result declarations must preserve forwarded input identities",
            ))),
        );
        assert_eq!(CallTypeBindings::with_outputs(&inputs, &outputs, first.outputs.to_vec()), Ok(Some(first.clone())));
    }

    #[test]
    fn test_call_type_bindings_instantiated() {
        let bounds = DimensionBounds::non_negative(Some(8)).unwrap();
        let input = DimensionType::new("input", bounds);
        let local = DimensionType::new("local", bounds);
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![input.dimension().clone()]));
        let output_type =
            ArrayType::new(DataType::F32, Shape::new(vec![input.dimension().clone(), local.dimension().clone()]));
        let outputs = [output_type.clone(), output_type];
        let first = CallTypeBindings::fresh(std::slice::from_ref(&input_type), &outputs).unwrap().unwrap();
        let second = CallTypeBindings::fresh(std::slice::from_ref(&input_type), &outputs).unwrap().unwrap();
        assert_ne!(first.outputs, second.outputs);
        assert_eq!(first.outputs[0], first.outputs[1]);
        let actual_input = ArrayType::new_static(DataType::F32, [2]);
        let actual = first.instantiated(std::slice::from_ref(&actual_input)).unwrap();
        assert_eq!(actual[0].shape().dimensions()[0], 2usize.into());
        assert_eq!(actual[0].shape().dimensions()[1], first.outputs[0].shape().dimensions()[1]);
        assert_eq!(actual[0], actual[1]);
        let callee_outputs = outputs
            .iter()
            .map(|output| {
                ArrayType::new(DataType::F32, Shape::new(vec![2usize.into(), output.shape().dimensions()[1].clone()]))
            })
            .collect::<Vec<_>>();
        let renaming = first.renaming(std::slice::from_ref(&actual_input), &callee_outputs).unwrap();
        assert_eq!(callee_outputs[0].rename_identities(&renaming).unwrap(), actual[0]);
        assert_eq!(first.definitions().len(), 1);
        let retained =
            CallTypeBindings::with_outputs(std::slice::from_ref(&input_type), &outputs, first.outputs.to_vec())
                .unwrap()
                .unwrap();
        assert_eq!(retained.instantiated(&[actual_input]).unwrap(), actual);
        let mut renamed = TypeIdentityRenaming::new();
        let local = first.definitions()[0].fresh();
        renamed.insert(first.definitions()[0].clone(), local.clone()).unwrap();
        let rebound = first.renamed(&renamed).unwrap();
        assert_eq!(rebound.definitions(), vec![local]);
        assert_ne!(rebound.outputs, first.outputs);
    }

    #[test]
    fn test_call_operation_split() {
        /// Supplies only the source callee and partition mode, while the test provides the partition closure directly.
        struct SplitCacheDriver {
            /// Callee shared across both partition modes.
            callee: Arc<TestProgram>,

            /// Mode to include in the retained split's key.
            repeated_residual: bool,
        }

        impl RegionDriver<TestValue, TestOperation> for SplitCacheDriver {
            fn regions<'r>(&'r self) -> impl Iterator<Item = RegionRef<'r, TestValue, TestOperation>>
            where
                TestValue: 'r,
                TestOperation: 'r,
            {
                std::iter::once(self.callee.entry_region_ref())
            }
        }

        impl PartialEvaluationDriver<TracingContext<TestValue, TestOperation>> for SplitCacheDriver {
            fn partially_evaluate_region(
                &self,
                _: &PartialEvaluationContext<TracingContext<TestValue, TestOperation>>,
                _: usize,
                _: Vec<PartialEvaluationValue<Tracer<TracingContext<TestValue, TestOperation>>>>,
            ) -> Result<Vec<PartialEvaluationValue<Tracer<TracingContext<TestValue, TestOperation>>>>, ProgramError>
            {
                unreachable!("the split cache fixture does not recursively evaluate regions")
            }

            fn partially_evaluate_program(
                &self,
                _: &PartialEvaluationContext<TracingContext<TestValue, TestOperation>>,
                _: RegionRef<'_, TestValue, TestOperation>,
                _: &[PartialValue<Tracer<TracingContext<TestValue, TestOperation>>>],
            ) -> Result<PartialEvaluation<TracingContext<TestValue, TestOperation>>, ProgramError> {
                unreachable!("the split cache fixture does not recursively evaluate programs")
            }

            fn partition_program(
                &self,
                _: &PartialEvaluationContext<TracingContext<TestValue, TestOperation>>,
                _: RegionRef<'_, TestValue, TestOperation>,
                _: &[bool],
            ) -> Result<PartitionedProgram<TestValue, TestOperation>, ProgramError> {
                unreachable!("the split cache fixture supplies the partition closure directly")
            }

            fn partitions_for_repeated_residual_calls(&self) -> bool {
                self.repeated_residual
            }
        }

        // Mixed call `g(k, x) = print(x * print(k * k))`, whose known coefficient and unknown value are its inputs.
        let scalar_type: ArrayIrType = ArrayType::scalar(DataType::F32).into();
        let callee = effectful_mixed_callee(EffectClass::UnorderedIo);
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let known = builder.add_input(scalar_type.clone());
        let squared = builder
            .add_instruction(ArrayOperation::from(MulOperation::new()), Vec::new(), vec![known, known], None)
            .unwrap()[0];
        let ordinary_known = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![squared], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let unknown = builder.add_input(scalar_type.clone());
        let coefficient = builder.add_input(scalar_type.clone());
        let printed = builder
            .add_instruction(
                ArrayOperation::from(
                    PrintOperation::<ArrayType>::new("known").with_effect_class(EffectClass::UnorderedIo),
                ),
                Vec::new(),
                vec![coefficient],
                None,
            )
            .unwrap()[0];
        let product = builder
            .add_instruction(ArrayOperation::from(MulOperation::new()), Vec::new(), vec![unknown, printed], None)
            .unwrap()[0];
        let output = builder
            .add_instruction(
                ArrayOperation::from(
                    PrintOperation::<ArrayType>::new("residual").with_effect_class(EffectClass::UnorderedIo),
                ),
                Vec::new(),
                vec![product],
                None,
            )
            .unwrap()[0];
        let ordinary_residual = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let partitions = Cell::new(0);
        let mut renderings = Vec::new();
        for repeated_residual in [false, false, true, true] {
            let driver = SplitCacheDriver { callee: callee.clone(), repeated_residual };
            let outer = TracingContext::<TestValue, TestOperation>::new();
            let known = outer.input(scalar_type.clone());
            let context = PartialEvaluationContext::new(outer).with_allow_effect_folding(false);
            let inputs = [context.known_value(known), context.unknown_input(scalar_type.clone(), 1)];
            let partition = |callee: RegionRef<'_, TestValue, TestOperation>| {
                partitions.set(partitions.get() + 1);
                if repeated_residual {
                    // Partitioning for repeated calls permits known effects in its fresh analysis despite the caller's
                    // effect folding restriction. Emission still defers those effects through the active context.
                    callee.partition(&[true, false])
                } else {
                    // Ordinary specialization inherits that restriction, so only the pure square remains known.
                    PartitionedProgram::from_parts(
                        ordinary_known.clone(),
                        ordinary_residual.clone(),
                        2,
                        vec![0],
                        vec![ResidualInputSource::UnknownInput(1), ResidualInputSource::ResidualEdge(0)],
                        vec![PartialEvaluationOutput::Unknown(0)],
                    )
                }
            };
            let outputs = CallOperation::<ArrayIrType>::new(0)
                .split(&context, &driver, &inputs, CallSplitKind::Knownness(vec![true, false]), &partition)
                .unwrap();
            let known_program = context
                .parent()
                .builder()
                .borrow()
                .clone()
                .build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder], Vec::new())
                .unwrap();
            drop(inputs);
            let evaluation = context.into_evaluation(outputs).unwrap();
            renderings.push(format!("{known_program}\n{}", evaluation.program()));
        }

        // Ordinary specialization stages the pure square on the known side, while partitioning for repeated calls
        // defers the known print, so the whole split lands in the residual program. Repeating a mode reuses its split.
        let ordinary = indoc! {"
            lambda %0:f32[] .
            let %1:f32[] = call [derived_input_shardings=[_]] %0 [
                callee={
                    lambda %0:f32[] .
                    let %1:f32[] = mul %0 %0
                    in (%1)
                },
            ]
            in ()
            lambda %0:f32[], %1:f32[] .
            let %2:f32[] = call [derived_input_shardings=[_, _]] %0 %1 [
                callee={
                    lambda %0:f32[], %1:f32[] .
                    let %2:f32[] = print [label=known, effect_class=unordered_io] %1
                        %3:f32[] = mul %0 %2
                        %4:f32[] = print [label=residual, effect_class=unordered_io] %3
                    in (%4)
                },
            ]
            in (%2)"};
        let repeated = indoc! {"
            lambda %0:f32[] .
            in ()
            lambda %0:f32[], %1:f32[] .
            let %2:f32[] = call [derived_input_shardings=[_]] %1 [
                callee={
                    lambda %0:f32[] .
                    let %1:f32[] = mul %0 %0
                        %2:f32[] = print [label=known, effect_class=unordered_io] %1
                    in (%2)
                },
            ]
                %3:f32[] = call [derived_input_shardings=[_, _]] %0 %2 [
                    callee={
                        lambda %0:f32[], %1:f32[] .
                        let %2:f32[] = mul %0 %1
                            %3:f32[] = print [label=residual, effect_class=unordered_io] %2
                        in (%3)
                    },
                ]
            in (%3)"};
        assert_eq!(renderings, vec![ordinary, ordinary, repeated, repeated]);
        // Debug builds re-derive every cache hit to validate deterministic derivation.
        assert_eq!(partitions.get(), if cfg!(debug_assertions) { 4 } else { 2 });
    }

    #[test]
    fn test_refined_derived_value() {
        // Each case stages the value of a derived call at its `declared` type and states the `refined` type that it has
        // in the caller, over a mesh with an explicit axis `x` and a manual axis `y`.
        let axes = |axis_type: MeshAxisType| {
            vec![MeshAxis::new("x", 2, axis_type).unwrap(), MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap()]
        };
        let mesh = LogicalMesh::new(axes(MeshAxisType::Explicit)).unwrap();
        let typed = |mesh: &LogicalMesh, dimensions: Vec<ShardingDimension>| {
            let r#type = ArrayType::new_static(DataType::F32, vec![4; dimensions.len()]);
            r#type.with_sharding(Sharding::new(mesh.clone(), dimensions).unwrap()).unwrap()
        };
        let with_sharding = |r#type: &ArrayType, sharding: Sharding| r#type.clone().with_sharding(sharding).unwrap();
        let stage = |declared: &ArrayType, refined: &ArrayType, call_mesh: Option<&DeviceMesh>| {
            let context = TracingContext::<TestValue, TestOperation>::new();
            let value = context.input(ArrayIrType::Array(declared.clone()));
            let value = refined_derived_value(&context, call_mesh, ArrayIrType::Array(refined.clone()), value).unwrap();
            context
                .builder()
                .borrow()
                .clone()
                .build::<Vec<TestValue>, Vec<TestValue>>(
                    vec![value.atom_id().unwrap()],
                    vec![Placeholder],
                    vec![Placeholder],
                )
                .unwrap()
                .to_string()
        };
        let (replicated, unconstrained, split) =
            (ShardingDimension::Replicated, ShardingDimension::Unconstrained, ShardingDimension::sharded(["x"]));
        let untracked = f32_vector_type(4);
        let replicated_vector = typed(&mesh, vec![replicated.clone()]);
        let sharding = replicated_vector.sharding().unwrap().clone();

        // A value whose type tracks a sharding without reduction state, manual variation, or a split over manual axes
        // is cast to a type of the same shape that tracks none.
        let untracked_cases = [
            replicated_vector.clone(),
            typed(&mesh, vec![unconstrained.clone()]),
            typed(&mesh, vec![split.clone()]),
            with_sharding(&untracked, sharding.clone().with_unreduced_axes(["y"]).unwrap()),
            with_sharding(&untracked, sharding.clone().with_reduced_axes(["y"]).unwrap()),
            with_sharding(&untracked, sharding.clone().with_varying_manual_axes(["y"]).unwrap()),
            typed(&mesh, vec![ShardingDimension::sharded(["y"])]),
        ];
        assert_eq!(
            untracked_cases
                .iter()
                .map(|declared| stage(declared, &untracked, None))
                .collect::<Vec<_>>()
                .join("\n"),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=2:explicit, 'y'=2:manual]>, [{}]}] .
                let %1:f32[4] = cast [
                    type=f32[4],
                    inputs=[f32[4][sharding={mesh<['x'=2:explicit, 'y'=2:manual]>, [{}]}]],
                ] %0
                in (%1)
                lambda %0:f32[4][sharding={mesh<['x'=2:explicit, 'y'=2:manual]>, [{?}]}] .
                let %1:f32[4] = cast [
                    type=f32[4],
                    inputs=[f32[4][sharding={mesh<['x'=2:explicit, 'y'=2:manual]>, [{?}]}]],
                ] %0
                in (%1)
                lambda %0:f32[4][sharding={mesh<['x'=2:explicit, 'y'=2:manual]>, [{'x'}]}] .
                let %1:f32[4] = cast [
                    type=f32[4],
                    inputs=[f32[4][sharding={mesh<['x'=2:explicit, 'y'=2:manual]>, [{'x'}]}]],
                ] %0
                in (%1)
                lambda %0:f32[4][sharding={mesh<['x'=2:explicit, 'y'=2:manual]>, [{}], unreduced={'y'}}] .
                in (%0)
                lambda %0:f32[4][sharding={mesh<['x'=2:explicit, 'y'=2:manual]>, [{}], reduced={'y'}}] .
                in (%0)
                lambda %0:f32[4][sharding={mesh<['x'=2:explicit, 'y'=2:manual]>, [{}], varying_manual={'y'}}] .
                in (%0)
                lambda %0:f32[4][sharding={mesh<['x'=2:explicit, 'y'=2:manual]>, [{'y'}]}] .
                in (%0)"},
        );

        // Equal types, types that track no sharding, and types of other shapes leave the value as it is.
        assert_eq!(
            [
                stage(&replicated_vector, &replicated_vector, None),
                stage(&untracked, &untracked, None),
                stage(&replicated_vector, &f32_vector_type(3), None),
            ]
            .join("\n"),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=2:explicit, 'y'=2:manual]>, [{}]}] .
                in (%0)
                lambda %0:f32[4] .
                in (%0)
                lambda %0:f32[4][sharding={mesh<['x'=2:explicit, 'y'=2:manual]>, [{}]}] .
                in (%0)"},
        );

        // Shardings that differ only in dimensions that one leaves unconstrained and the other replicates place a
        // value alike, so it is cast, in either direction. Any other difference in placement over a mesh of the same
        // topology is a reshard, and a difference in reduction state or manual variation leaves the value as it is.
        let open = typed(&mesh, vec![unconstrained.clone(), split.clone()]);
        let closed = typed(&mesh, vec![replicated.clone(), split.clone()]);
        let auto_mesh = LogicalMesh::new(axes(MeshAxisType::Auto)).unwrap();
        let replicated_matrix = typed(&mesh, vec![replicated.clone(), replicated.clone()]);
        let matrix_sharding = replicated_matrix.sharding().unwrap().clone();
        assert_eq!(
            [
                stage(&open, &closed, None),
                stage(&closed, &open, None),
                stage(&typed(&mesh, vec![unconstrained.clone(), replicated.clone()]), &closed, None),
                stage(&replicated_matrix, &closed, None),
                stage(&open, &typed(&auto_mesh, vec![replicated.clone(), replicated.clone()]), None),
                stage(
                    &typed(&mesh, vec![unconstrained.clone(), replicated.clone()]),
                    &with_sharding(&replicated_matrix, matrix_sharding.clone().with_unreduced_axes(["y"]).unwrap()),
                    None,
                ),
                stage(
                    &typed(&mesh, vec![unconstrained.clone(), replicated.clone()]),
                    &with_sharding(&replicated_matrix, matrix_sharding.with_varying_manual_axes(["y"]).unwrap()),
                    None,
                ),
            ]
            .join("\n"),
            indoc! {"
                lambda %0:f32[4, 4][sharding={mesh<['x'=2:explicit, 'y'=2:manual]>, [{?}, {'x'}]}] .
                let %1:f32[4, 4][sharding={mesh<['x'=2:explicit, 'y'=2:manual]>, [{}, {'x'}]}] = cast [
                    type=f32[4, 4][sharding={mesh<['x'=2:explicit, 'y'=2:manual]>, [{}, {'x'}]}],
                    inputs=[f32[4, 4][sharding={mesh<['x'=2:explicit, 'y'=2:manual]>, [{?}, {'x'}]}]],
                ] %0
                in (%1)
                lambda %0:f32[4, 4][sharding={mesh<['x'=2:explicit, 'y'=2:manual]>, [{}, {'x'}]}] .
                let %1:f32[4, 4][sharding={mesh<['x'=2:explicit, 'y'=2:manual]>, [{?}, {'x'}]}] = cast [
                    type=f32[4, 4][sharding={mesh<['x'=2:explicit, 'y'=2:manual]>, [{?}, {'x'}]}],
                    inputs=[f32[4, 4][sharding={mesh<['x'=2:explicit, 'y'=2:manual]>, [{}, {'x'}]}]],
                ] %0
                in (%1)
                lambda %0:f32[4, 4][sharding={mesh<['x'=2:explicit, 'y'=2:manual]>, [{?}, {}]}] .
                let %1:f32[4, 4][sharding={mesh<['x'=2:explicit, 'y'=2:manual]>, [{}, {'x'}]}] = reshard [sharding={mesh<['x'=2:explicit, 'y'=2:manual]>, [{}, {'x'}]}] %0
                in (%1)
                lambda %0:f32[4, 4][sharding={mesh<['x'=2:explicit, 'y'=2:manual]>, [{}, {}]}] .
                let %1:f32[4, 4][sharding={mesh<['x'=2:explicit, 'y'=2:manual]>, [{}, {'x'}]}] = reshard [sharding={mesh<['x'=2:explicit, 'y'=2:manual]>, [{}, {'x'}]}] %0
                in (%1)
                lambda %0:f32[4, 4][sharding={mesh<['x'=2:explicit, 'y'=2:manual]>, [{?}, {'x'}]}] .
                let %1:f32[4, 4][sharding={mesh<['x'=2:auto, 'y'=2:manual]>, [{}, {}]}] = reshard [sharding={mesh<['x'=2:auto, 'y'=2:manual]>, [{}, {}]}] %0
                in (%1)
                lambda %0:f32[4, 4][sharding={mesh<['x'=2:explicit, 'y'=2:manual]>, [{?}, {}]}] .
                in (%0)
                lambda %0:f32[4, 4][sharding={mesh<['x'=2:explicit, 'y'=2:manual]>, [{?}, {}]}] .
                in (%0)"},
        );

        // A value whose type tracks no sharding is cast to a refined type that tracks one, unless the call names
        // devices over a mesh of another topology, on which the value lives.
        let other_mesh = LogicalMesh::new(vec![MeshAxis::new("z", 4, MeshAxisType::Explicit).unwrap()]).unwrap();
        let devices = (0..4).map(|id| Device::new(id, 0)).collect::<Vec<_>>();
        let call_mesh = DeviceMesh::new(other_mesh, devices).unwrap();
        assert_eq!(
            stage(&untracked, &replicated_vector, None),
            indoc! {"
                lambda %0:f32[4] .
                let %1:f32[4][sharding={mesh<['x'=2:explicit, 'y'=2:manual]>, [{}]}] = cast [
                    type=f32[4][sharding={mesh<['x'=2:explicit, 'y'=2:manual]>, [{}]}],
                    inputs=[f32[4]],
                ] %0
                in (%1)"},
        );
        assert_eq!(
            stage(&untracked, &replicated_vector, Some(&call_mesh)),
            indoc! {"
                lambda %0:f32[4] .
                in (%0)"},
        );
    }
}
