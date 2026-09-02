//! Contains machinery for representing and working with typed, structured, and effect-aware programs.
//!
//! A [`Program`] is Ryft's backend-neutral dataflow IR. It owns a flat arena of [`Region`]s that consists of the public
//! entry computation plus any nested computations referenced by its instructions, where each region stores typed
//! atoms, operation instructions, a flat boundary, and enough metadata for interpretation, transformation,
//! simplification, lowering, and compilation. Programs are immutable after construction. [`ProgramBuilder`]
//! owns the mutable construction phase, sealing every non-entry region before instructions can attach it.
//!
//! # Program Lifecycle
//!
//! Program construction is the only mutable phase. Every replay, transformation, analysis, and compilation step starts
//! from an immutable program whose region graph and structured boundaries have already been validated. Refer to
//! [`Program`] for a rendered diagram of this lifecycle.
//!
//! # Entry Points
//!
//! Most code obtains programs through [`trace`](crate::trace) or a transform rather than by manual construction, and
//! replays them with [`Program::interpret`] (eagerly) or [`Program::interpret_in_context`] (through a chosen staging
//! or transform context). Batching, differentiation, and partial evaluation add program-level functions in their own
//! modules, and compilation opens captures, flattens boundaries, and hands the program to a backend
//! [`CompilationDomain`](crate::CompilationDomain).
//!
//! Direct [`ProgramBuilder`] use is appropriate for operation and transform infrastructure. Tracer operations call
//! [`ProgramBuilder::add_instruction`], which infers output types, allocates variable atoms, validates arity, and
//! records the instruction, and [`ProgramBuilder::build`] validates the requested boundaries and freezes the result.
//! Keep [`AtomId`]s from one builder isolated from every other builder, use the checked instruction path, and
//! propagate the builder's first stored error rather than continuing with invalid IDs.
//!
//! [`Program::to_flat_program`] converts structured boundaries to vectors without changing the dataflow. Use it at
//! internal compiler or nested-program boundaries, and preserve the structured form in user-facing APIs.
//!
//! # Core Data Model
//!
//! [`Value`] is the common contract for leaf values that can inhabit programs or flow through Ryft contexts. Every
//! value has one associated type through [`Typed`], plus separate dispatch and execution domains used by capabilities
//! and transforms.
//!
//! [`Atom`] is either a stored constant or a typed variable, and [`AtomId`] is its stable index in the containing
//! [`Region`]'s atom table. An [`Instruction`] owns one [`Operation`], lists the input and output atom IDs of that
//! application, and carries the [`RegionId`]s of its attached nested regions, in the operation-defined order.
//! Operations define their own type inference and effect classes and the program supplies graph structure and order.
//!
//! [`Program`] combines the region arena with typed, structured input and output boundaries on its entry region.
//! The boundary types are [`Parameterized`](crate::Parameterized) containers whose leaves correspond positionally to
//! [`Program::input_ids`] and [`Program::output_ids`], so compiler and transform kernels can operate on the flat IDs
//! while callers retain tuples, vectors, maps, or derived product types. [`InstructionId`] and [`ValueId`] locate
//! instructions and values across [`Region`]s.
//!
//! # Regions, Sharing, and Sealing
//!
//! The canonical region graph and operation-application vocabulary lives in the [`regions`] module. This module owns
//! the surrounding program arena and its construction, validation, transformation, and rendering machinery.
//!
//! Every nested computation (e.g., a control-flow branch or body, a custom-derivative program, a rematerialization
//! program, a JIT-ed callee, etc.) is a [`Region`] in the owning [`Program`]'s one canonical arena, referenced from its
//! instructions through [`Instruction::regions`]. There is exactly one instruction edge kind: sharing is expressed by
//! repeating a [`RegionId`], not by a parallel node table or by operation payloads owning programs. The
//! [`ProgramBuilder`] offers three import policies for nested computations:
//!
//!   - [`ProgramBuilder::import_region`] copies a borrowed [`RegionRef`]'s complete region closure into the arena,
//!     preserving any sharing internal to the imported closure.
//!   - [`ProgramBuilder::import_program`] splices an owned [`Program`]'s arena in directly without cloning, for owned
//!     bodies whose builder would otherwise clone them away.
//!   - [`ProgramBuilder::intern_callee`] interns a shared [`Rc`](std::rc::Rc)-held [`Program`] by pointer identity
//!     (i.e., importing the same `Rc` twice yields the same root [`RegionId`], which is how repeated JIT-compiled calls
//!     to one compiled callee share one region and how lowering deduplication can count occurrences per root).
//!
//! Only *sealed* regions are attachable. [`ProgramBuilder::add_instruction`] validates the attached region list
//! against the operation's declared [`Operation::region_slots`] slots, and every non-entry region enters the arena
//! as a complete, immutable program with an explicit boundary (i.e., an explicit [`RegionInterface`]). A region never
//! references atoms of another region directly; values cross region boundaries only through the boundary inputs and
//! outputs, and cross-program constants only through captures (see [`captures`](crate::captures) for the capture-scope
//! model; captures are registered in the trace that owns the instruction, and nested traces reach the root table
//! through their parent chain).
//!
//! [`RegionRef`] borrows any sealed arena region for inspection or replay without cloning it.
//! [`RegionRef::to_program`] materializes that borrowed region back into a standalone flat [`Program`], copying its
//! reachable subtree. Locators such as [`InstructionId`], [`ValueId`], and [`RegionId`] are scoped to the program they
//! were derived from. Materialization and rebuilds renumber arenas, and locators never cross [`Program`] boundaries.
//!
//! # Structural Identity and Transform Reuse
//!
//! A sealed region may retain context-free program artifacts already derived from its complete reachable contents.
//! Cloning a program, importing a faithful region closure, or rebasing region identifiers while preserving graph
//! topology shares those artifacts. Rewriting a region or attaching it to a different descendant closure invalidates
//! them. This ownership rule keeps reuse local to the programs that carry the region and prevents a derived program
//! from being served for a different nested graph.
//!
//! Only context-free structural results belong in that retained state. Invocation-specific residual values, active
//! contexts, backend buffers, and other runtime data remain outside the program and its regions. External crates
//! can define a [`Transform`] and request its typed artifact through [`RegionRef::transform`]. Raw cache
//! adoption and invalidation remain internal construction invariants.
//!
//! # Effects, Liveness, and Simplification
//!
//! [`Program::effects`] unions the effects declared by its operations. Instruction order is semantically relevant
//! for ordered effects even when the dataflow graph contains no dependency between them.
//!
//! [`Program::live_sets`] computes the atoms and instructions required by selected roots. [`Program::simplified`]
//! removes dead pure work while retaining effectful instructions as roots. [`Program::filtered`] projects a program
//! to selected boundaries and accepts explicit keep-alive atoms for work that must survive the projection. These APIs
//! preserve the invariants checked by normal program construction rather than treating effects as ordinary unused
//! values.
//!
//! [`Program::statistics`] is the structural-inspection entry point. It reports backend-neutral per-region counts,
//! operation histograms, dependency depths, and attached-region edges for the stored program structure (including
//! dormant rule regions and dead instructions) without applying any simplification. Refer to [`ProgramStatistics`]
//! for more information.
//!
//! # Extending Programs
//!
//! New primitive behavior normally means adding an operation payload implementing [`Operation`] and including it in the
//! appropriate closed operation family. Keep type inference, rendering, effects, and operation-specific transform rules
//! with that payload, and never teach [`Program`] about individual operation variants.

use std::fmt::Debug;
use std::sync::Arc;

use thiserror::Error;

use crate::errors::CustomError;
use crate::parameters::ParameterError;

pub mod atoms;
pub mod builders;
pub mod effects;
pub mod identities;
pub mod instructions;
pub mod operations;
pub mod programs;
pub mod provenance;
pub mod references;
pub mod regions;
pub mod statistics;
pub mod transforms;
pub mod types;
pub mod values;

pub use atoms::{Atom, AtomId, MaybeZero};
pub use builders::ProgramBuilder;
pub use effects::{Effect, EffectOccurrence, Effects};
pub use identities::{NoIdentity, TypeIdentity, TypeIdentityPosition, TypeIdentityRenaming, TypeIdentitySignature};
pub use instructions::{Instruction, InstructionId};
pub use operations::{
    MemberOperation, Operation, OperationFormatter, OperationProjection, OperationProvider,
    infer_projected_operation_output_types, infer_projected_operation_region_input_types,
};
pub use programs::{FlatProgram, Program, ProgramLiveSets, ProgramRenderingMode};
pub use provenance::{Provenance, ProvenanceScope, ProvenanceState};
pub use references::{
    ExternalReferenceBinding, PartialReferenceDischargeResult, PreparedReferenceReplacement,
    REFERENCE_ADD_UPDATE_OPERATION_NAME, REFERENCE_FREEZE_OPERATION_NAME, REFERENCE_NEW_OPERATION_NAME,
    REFERENCE_READ_OPERATION_NAME, REFERENCE_SWAP_OPERATION_NAME, REFERENCE_WRITE_OPERATION_NAME,
    ReadyOrPendingReferenceGuard, ReadyReferenceGuard, RecursiveReferenceDischargeDriver, Reference,
    ReferenceAccessMode, ReferenceAccumulationPolicy, ReferenceAddUpdate, ReferenceAddUpdateOperation,
    ReferenceAliasKind, ReferenceCompletion, ReferenceCompletionBackend, ReferenceDischargeAllocationId,
    ReferenceDischargeContext, ReferenceDischargeDriver, ReferenceDischargePolicy, ReferenceDischargeReference,
    ReferenceDischargeRegionBoundary, ReferenceDischargeRegionResult, ReferenceDischargeRegionSummary,
    ReferenceDischargeResult, ReferenceDischargeTarget, ReferenceDischargeValue, ReferenceDischargeableOperation,
    ReferenceDischargeableType, ReferenceError, ReferenceFreeze, ReferenceFreezeOperation, ReferenceGeneration,
    ReferenceId, ReferenceInput, ReferenceNew, ReferenceNewOperation, ReferenceObservation,
    ReferenceOperationSemantics, ReferenceOutput, ReferenceRead, ReferenceReadOperation, ReferenceRegionStateInsertion,
    ReferenceReplacementPreparation, ReferenceReplacementTransaction, ReferenceSource, ReferenceStateWidening,
    ReferenceSwap, ReferenceSwapOperation, ReferenceType, ReferenceTypeRefinements, ReferenceWrite,
    ReferenceWriteOperation, TakenReferenceGuard, ValidatedPendingReplacementTransaction,
    discharge_positional_region_operation, discharge_reference_free_operation,
};
pub use regions::{
    BindingRegionDriver, CalleeRegionDriver, DestinationRegionMapping, EagerInterpretationValidation,
    EmptyRegionDriver, OutputRegionProvenance, Region, RegionArena, RegionArenaIterator, RegionDriver, RegionId,
    RegionInterface, RegionRef, RegionReplayMappings, RegionRole, RegionSlot, RegionWithMetadata, ReplayRegionDriver,
};
pub use statistics::{AttachedRegionStatistics, ProgramStatistics, RegionStatistics};
pub use transforms::{Transform, TransformArtifact, TransformCache};
pub use types::{Type, TypeError, TypeRefinements, Typed};
pub use values::{Concretizable, ParameterProjection, ProjectedValue, Value, ValueId, ValueProjection};

/// Represents errors related to [`Program`]s in `ryft-core`.
#[derive(Clone, Debug, Error, PartialEq, Eq, Hash)]
pub enum ProgramError {
    #[error("values used in the same operation must share the same program builder")]
    MismatchedProgramBuilders,

    #[error("{message}")]
    InvalidArgument { message: String },

    #[error("invalid number of inputs; expected {expected} but got {actual}")]
    InvalidInputCount { expected: usize, actual: usize },

    #[error("invalid number of outputs; expected {expected} but got {actual}")]
    InvalidOutputCount { expected: usize, actual: usize },

    #[error("unbound atom ID: {id}")]
    UnboundAtomId { id: AtomId },

    #[error("encountered malformed program: {0}")]
    MalformedProgram(String),

    #[error("encountered program builder that escaped its scope")]
    EscapedProgramBuilder,

    #[error(
        "trace registered {count} runtime capture(s) through a trace entry point that discards its capture table; \
         capture-owning traces must construct their `TracingContext` directly and pair the traced program with that \
         context's capture table"
    )]
    DiscardedCaptures { count: usize },

    #[error("encountered poisoned value where a live value was required")]
    PoisonedValue,

    #[error("{message}")]
    Concretization { message: String },

    #[error("{message}")]
    UnsupportedOperation { message: String },

    #[error(transparent)]
    Parameter(#[from] ParameterError),

    #[error(transparent)]
    Type(#[from] TypeError),

    #[error("{0}")]
    Custom(Arc<dyn CustomError>),
}

impl ProgramError {
    /// Wraps an operation- or transform-specific error in a [`Custom`](ProgramError::Custom) variant. The concrete
    /// error can later be recovered using [`ProgramError::downcast_custom`].
    #[inline]
    pub fn custom(error: impl CustomError) -> Self {
        ProgramError::Custom(Arc::new(error))
    }

    /// Returns the wrapped custom error downcast to `T` when this is a [`Custom`](ProgramError::Custom) variant holding
    /// a `T`, and [`None`] otherwise.
    #[inline]
    pub fn downcast_custom<T: CustomError>(&self) -> Option<&T> {
        match self {
            // Deref through the `Arc` to the `dyn CustomError`, upcast to `&dyn std::error::Error`, and then use the
            // standard error downcast. Going through the `Arc` directly would downcast the `Arc` instead of the error.
            ProgramError::Custom(custom) => (&**custom as &dyn std::error::Error).downcast_ref::<T>(),
            _ => None,
        }
    }
}
