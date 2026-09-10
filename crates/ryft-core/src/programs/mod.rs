//! Typed programs, their construction and interpretation, and the shared infrastructure for program transforms.
//!
//! A [`Program`] is Ryft's backend-neutral intermediate representation: it records which operations run, which values
//! they use and produce, and which nested computations they invoke. The program graph is immutable after construction,
//! so interpretation, analysis, transformation, and compilation can inspect the same representation. Operation-specific
//! behavior belongs to [`Operation`] implementations and their interpretation and transform rules.
//!
//! # Values, Instructions, and Boundaries
//!
//! A [`Value`] is a leaf value that can flow through a program or a [`Context`](crate::Context), with a type
//! provided by [`Typed`]. Within a program, an [`Atom`] represents either a stored constant or a typed variable.
//! An [`Instruction`] records one operation application: its operation, input and output [`AtomId`]s, attached
//! [`RegionId`]s, and source provenance. Operations supply type inference and effect declarations; instructions
//! supply the values and nested computations for a particular application.
//!
//! Each [`Region`] has its own atom table, instruction sequence, and ordered input and output lists. A program owns
//! an arena of regions and identifies one as its entry computation. Its public inputs and outputs retain their
//! [`Parameterized`](crate::Parameterized) structure, such as tuples, vectors, or derived product types. Their leaves
//! correspond positionally to [`Program::input_ids`] and [`Program::output_ids`]. [`Program::to_flat_program`] exposes
//! those boundaries as vectors without changing the computation, allowing transforms to work with flat lists while
//! user-facing APIs preserve the original structure.
//!
//! # Constructing and Running Programs
//!
//! Most callers obtain a program through [`trace`](crate::trace) or a transform. [`Program::interpret`] runs it
//! eagerly; [`Program::interpret_in_context`] replays it through a chosen context, which can execute operations
//! or stage further work. [`Batching`](crate::batching), [`differentiation`](crate::differentiation), and
//! [`partial evaluation`](crate::partial) provide their own program transformations. Compilation passes programs
//! to a backend [`CompilationDomain`](crate::CompilationDomain).
//!
//! [`ProgramBuilder`] supports direct construction for operation rules and transform infrastructure.
//! [`ProgramBuilder::add_instruction`] validates an application, infers its output types, and creates its output atoms.
//! [`ProgramBuilder::build`] validates the requested entry boundary and produces the immutable program. Atom IDs belong
//! to their builder; they must not be reused in another builder. Construction code must also propagate the builder's
//! stored error instead of treating IDs produced after an error as valid. See [`Program`] for the program lifecycle.
//!
//! # Nested Computations and Sharing
//!
//! Branches, loop bodies, callees, and custom derivative rules all use regions in the same program arena. Instructions
//! attach them in the order declared by [`Operation::region_slots`]. Each [`RegionSlot`] declares an attached region's
//! name and role. [`RegionRole::Computation`] regions execute as part of the operation, while [`RegionRole::Rule`]
//! regions supply transform implementations and remain dormant during ordinary interpretation.
//!
//! An attached region is sealed: its body and [`RegionInterface`] are complete before an instruction can use it.
//! Regions cannot refer directly to one another's atoms; values pass through explicit region boundaries. Captures are
//! handled through the owning trace's capture scope, as described in [`captures`](crate::captures). Operation rules
//! navigate attached regions through [`RegionDriver`] implementations rather than storing nested programs in operation
//! payloads.
//!
//! Repeated attachments of the same region ID share one stored computation. Builders support three ways to import one:
//!
//!   - [`ProgramBuilder::import_region`] copies a borrowed region and every region reachable through its attachments,
//!     preserving sharing within that graph.
//!   - [`ProgramBuilder::import_program`] moves an owned program's arena into the builder without cloning its regions.
//!   - [`ProgramBuilder::intern_callee`] reuses an imported [`Arc`]-held program. Callee identity and,
//!     when supplied, its exact input-type instantiation determine reuse; structural equality alone does not.
//!
//! [`RegionRef`] borrows a region together with access to its program arena. [`RegionRef::to_program`] copies its
//! reachable graph into a standalone program. [`RegionId`], [`InstructionId`], and [`ValueId`] identify locations
//! within one program, and [`AtomId`] identifies an atom within one region. Imports and rebuilds can renumber these
//! locations; IDs from the source must not be used to address the rebuilt program. The [`regions`] module explains
//! region access and interfaces in more detail.
//!
//! # Effects and Program Inspection
//!
//! [`Effects`] describes an operation's intrinsic effects, including reference accesses and aliases. An
//! [`EffectsSummary`] also accounts for attached computation regions. [`Program::effects`] summarizes the entry
//! computation recursively, excluding dormant rule regions. These declarations preserve dependencies that are not
//! expressed by immutable value inputs, such as a read following a write to the same reference. The [`references`]
//! module provides the reference model, validation, and conversion of mutable state into explicit value dataflow.
//!
//! [`Program::live_sets`] computes dependencies of the program outputs; [`Program::live_sets_for_atoms`] accepts other
//! roots. [`Program::simplified`] removes unused work while retaining instructions with observable consequences even
//! when their results are unused, in their original relative order. For example, an unused reference allocation and its
//! dead aliases can disappear, but a reference write remains observable. [`Program::filtered`] restricts the entry
//! boundary and accepts explicit keep-alive atoms for computations needed beyond the selected outputs.
//!
//! [`Program::statistics`] describes the stored graph, including dead instructions and dormant rule regions, without
//! simplifying it. [`ProgramStatistics`] provides counts, operation histograms, dependency depths, and attachment
//! edges.
//!
//! # Transform Reuse and Extensibility
//!
//! A sealed region can retain context-free artifacts derived from its complete reachable graph. Faithful clones and
//! imports share those artifacts even when region IDs change. Rewriting the computation or changing its descendants
//! invalidates them. Invocation-specific residual values, active contexts, and backend buffers remain outside this
//! retained state. Downstream crates can define a [`Transform`] and obtain its typed result through
//! [`RegionRef::transform`]; cache adoption and invalidation are handled by program construction.
//!
//! New operations define their type inference, effects, rendering, and transform rules alongside the operation type.
//! The program representation supplies their common structure without needing cases for individual operation variants.

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
pub use builders::{ProgramBuilder, ProgramBuilderId};
pub use effects::{
    EffectClass, EffectClassOccurrence, EffectClasses, Effects, EffectsSummary, ReferenceAccessMode, ReferenceAlias,
    ReferenceAliasKind, ReferenceEffect,
};
pub use identities::{NoIdentity, TypeIdentity, TypeIdentityPosition, TypeIdentityRenaming, TypeIdentitySignature};
pub use instructions::{Instruction, InstructionId};
pub use operations::{
    MemberOperation, Operation, OperationFormatter, OperationProjection, OperationProvider,
    infer_projected_operation_output_types, infer_projected_operation_region_input_types,
};
pub use programs::{FlatProgram, Program, ProgramLiveSets, ProgramRenderingMode};
pub use provenance::{Provenance, ProvenanceScope, ProvenanceState};
pub use references::{
    BatchableReferenceView, ExternalReferenceBinding, NoReferenceViewBinding, PartialReferenceDischargeResult,
    PreparedReferenceReplacement, ReadyOrPendingReferenceGuard, ReadyReferenceGuard, RecursiveReferenceDischargeDriver,
    Reference, ReferenceAccess, ReferenceAccumulationPolicy, ReferenceAliasEdge, ReferenceAnalysis,
    ReferenceAnalysisError, ReferenceBoundary, ReferenceBoundaryError, ReferenceBoundaryPosition, ReferenceCompletion,
    ReferenceCompletionBackend, ReferenceDischargeAllocationId, ReferenceDischargeBoundaryWidening,
    ReferenceDischargeContext, ReferenceDischargeDriver, ReferenceDischargePolicy, ReferenceDischargeReference,
    ReferenceDischargeRegionBoundary, ReferenceDischargeRegionBoundaryInsertion, ReferenceDischargeRegionInput,
    ReferenceDischargeRegionOutput, ReferenceDischargeRegionResult, ReferenceDischargeRegionSummary,
    ReferenceDischargeResult, ReferenceDischargeTarget, ReferenceDischargeValue, ReferenceDischargeableOperation,
    ReferenceDischargeableType, ReferenceError, ReferenceGeneration, ReferenceId, ReferenceIdentity,
    ReferenceObservation, ReferenceRegionInputBinding, ReferenceReplacementPreparation,
    ReferenceReplacementTransaction, ReferenceRoot, ReferenceSource, ReferenceTransitiveAccess, ReferenceType,
    ReferenceTypeRefinements, ReferenceView, ReferenceViewAnalysis, ReferenceViewAnalysisError, ReferenceViewOperation,
    ReferenceViewOverlap, ReferenceViewPath, ReferenceViewStep, ReferenceViewValidationError, TakenReferenceGuard,
    ValidatedPendingReplacementTransaction, discharge_local_reference_operation, discharge_positional_region_operation,
    discharge_reference_free_operation, validate_reference_boundary,
};
pub use regions::{
    BindingRegionDriver, CalleeRegionDriver, DestinationRegionMapping, EmptyRegionDriver, OutputRegionProvenance,
    Region, RegionArena, RegionArenaIterator, RegionDriver, RegionId, RegionInterface, RegionRef, RegionReplayMappings,
    RegionRole, RegionSlot, RegionWithMetadata, ReplayRegionDriver,
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

    #[error(transparent)]
    Reference(#[from] ReferenceError),

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

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn test_program_error_from_reference_error() {
        let error = ProgramError::from(ReferenceError::Frozen);
        assert_eq!(error, ProgramError::Reference(ReferenceError::Frozen));
        assert_eq!(error.to_string(), "reference is frozen");
        assert_eq!(format!("{error:?}"), "Reference(Frozen)");
    }
}
