//! Mutable references, their identities and views, and analysis and discharge of references in programs.
//!
//! A reference names mutable storage. Reading it produces an immutable value. Writing it changes the value seen by
//! subsequent reads through any alias of that storage. References therefore need more than ordinary value dataflow:
//! programs must preserve access order, track aliases, and prevent access after consumption. This module supplies
//! those contracts without assuming that the stored value is an array. Array reference values and indexing and slicing
//! operations are defined in [`arrays`](crate::arrays).
//!
//! # Types, Handles, and Identities
//!
//! [`ReferenceType`] describes a reference's referent type, written `ref<T>`. The referent is the value exposed through
//! the reference. Its type describes what can be read or written. Type information does not identify a particular
//! allocation or contain its runtime state.
//!
//! [`Reference`] is an eager handle to one allocation. Cloning a handle shares its synchronized state rather than
//! copying the stored value. A read returns an immutable snapshot. A consuming [`Reference::freeze`] returns the final
//! value and invalidates every handle to the allocation. [`ReferenceId`] identifies that runtime allocation, while
//! [`ReferenceIdentity`] also represents references in a staging context that do not yet have runtime storage.
//!
//! A program value of reference type is also a handle. Several values may name the same allocation,
//! so their distinct [`ValueId`](crate::ValueId)s do not establish that their state is independent.
//! [`ReferenceAlias`](crate::ReferenceAlias) declares that an operation output aliases one of its inputs. An alias
//! preserves that allocation, either exposing its complete value or forming a view of part of it, as distinguished by
//! [`ReferenceAliasKind`](crate::ReferenceAliasKind). Each handle resolves to one allocation; aliases do not combine
//! multiple independent allocations.
//!
//! A _local_ allocation is created inside the program. An *external* allocation enters through an input or capture and
//! belongs to the caller. [`ReferenceSource`] records this distinction during discharge. It determines whether state
//! can disappear with a local implementation detail or must be returned to the caller after execution.
//!
//! # Reference Operations
//!
//! The generic primitives and their value capabilities live in
//! [`operations::references`](crate::operations::references):
//!
//!   - [`ReferenceNew`](crate::ReferenceNew) allocates a reference initialized with a value.
//!   - [`ReferenceRead`](crate::ReferenceRead) reads its current value without consuming the reference.
//!   - [`ReferenceWrite`](crate::ReferenceWrite) replaces the value.
//!   - [`ReferenceSwap`](crate::ReferenceSwap) replaces the value and returns the previous value.
//!   - [`ReferenceAddUpdate`](crate::ReferenceAddUpdate) adds a contribution to the stored value.
//!   - [`ReferenceFreeze`](crate::ReferenceFreeze) consumes the reference and returns its final value.
//!
//! Each operation owns its type inference, effects, interpretation, and transform rules. Reference accesses and
//! lifetime changes are declared through [`ReferenceEffect`](crate::ReferenceEffect), allowing program analyses to
//! preserve state dependencies even when an instruction's outputs are unused. Numeric operations act on values read
//! from references, rather than implicitly reading reference operands.
//!
//! # Views and Program Analysis
//!
//! A view is an alias that exposes part of an allocation's value. For example, an array reference slice can expose
//! elements `2..5`. Writing through it updates those elements in the original allocation. The view and the complete
//! reference share one lifetime. Consuming through a narrowing view is rejected because consumption invalidates the
//! whole allocation, including handles that expose other elements.
//!
//! [`ReferenceAnalysis`] identifies allocations as [`ReferenceRoot`]s and records aliases, accesses, and reference
//! relationships across region boundaries. It validates reference lifetimes and provides transitive access summaries
//! for operations with nested computations. Consumers request it through
//! [`RegionRef::reference_analysis`](crate::RegionRef::reference_analysis), supplying the capture information required
//! by that boundary. It is an explicitly requested analysis, not a validation pass automatically run on every program.
//!
//! [`ReferenceView`] represents one view step, such as an array index or slice. [`ReferenceViewOperation`]
//! associates views with operation outputs and defines how to validate and reapply them to transformed references.
//! [`ReferenceViewAnalysis`] combines those steps into a [`ReferenceViewPath`] for each reference value, alongside the
//! structural reference analysis. This lets transforms reconstruct a view without matching array operation variants.
//!
//! A static slice stores its bounds in the view. A dynamic index instead depends on an instruction input:
//! [`ReferenceView::symbols`] lists that input's position, and analysis binds it to the corresponding
//! [`ValueId`](crate::ValueId) in a [`ReferenceViewStep`]. A path's binding type depends on its consumer: analysis
//! uses program [`ValueId`](crate::ValueId)s, discharge can use transformed values, and eager static paths use
//! [`NoReferenceViewBinding`]. [`ReferenceViewOverlap`] distinguishes identical views, provably disjoint views, and
//! views that may overlap. Disjoint views still share an allocation and its lifetime; the overlap result alone does
//! not establish independent mutable state.
//!
//! Attached regions receive complete reference handles and construct their views with instructions inside the region.
//! Any required index is an ordinary region input or computed value. For example, a
//! [`ScanOperation`](crate::ScanOperation) body receives an explicit iteration index and can use it to index a
//! reference passed to the body. The generic view analysis needs no scan-specific symbol or implicit boundary view.
//!
//! # Discharging Mutable State
//!
//! Reference discharge rewrites mutable state into explicit immutable values. For example, a write followed by a read
//! of a local reference becomes a direct use of the written value. Branches and loops thread the current state through
//! their inputs and outputs. [`ReferenceDischargeReference`] is the transform's temporary handle for an allocation and
//! its policy-defined view metadata; it is distinct from an eager reference.
//!
//! [`Program::discharge_references`](crate::Program::discharge_references) returns a [`ReferenceDischargeResult`] whose
//! program contains no surviving reference types or operations, including in attached regions. Local allocations are
//! removed. External references become value inputs carrying their initial state and, when mutated, hidden outputs
//! carrying their final state. [`ExternalReferenceBinding`] records how a backend must connect those values to the
//! caller's references. Discharge rewrites the program; it neither executes it nor publishes changes to eager state.
//! Its result documentation describes the resulting program boundary.
//!
//! [`Program::partially_discharge_references`](crate::Program::partially_discharge_references) instead rewrites
//! selected allocations and returns a [`PartialReferenceDischargeResult`]. Other references remain available to later
//! transforms or backends that support mutable state. A program may otherwise return a local reference to a caller that
//! keeps it as a reference, but full discharge rejects such escaping allocations: returning a value is not equivalent
//! to returning a mutable handle, and external-state bindings do not reconstruct newly allocated reference outputs.
//!
//! # Validation and Transform Contracts
//!
//! Different layers enforce the constraints for which they have enough information:
//!
//!   - [`ProgramBuilder::add_instruction`](crate::ProgramBuilder::add_instruction) tracks declared aliases and lifetime
//!     changes in the region being built, rejecting accesses after consumption and consumption through a view.
//!   - Eager references validate their runtime state, including frozen or poisoned allocations and conflicting
//!     accesses. [`ReferenceGeneration`] and [`ReferenceCompletion`] support updates whose backend work completes
//!     asynchronously. [`ReferenceObservation`] provides a coherent observation of that state.
//!   - Reference analysis and discharge inspect program structure, region boundaries, and the state threaded
//!     by a rewrite. Full discharge additionally checks that the result is reference-free.
//!   - [`ReferenceBoundary`] checks identities supplied by a [`Context`](crate::Context), while
//!     [`validate_reference_boundary`] checks concrete values. Callers define diagnostic positions and the aliasing
//!     restrictions of their transform. Types alone cannot reveal that two runtime inputs name the same allocation,
//!     and raw program interpretation does not perform these transform-specific boundary checks.
//!
//! [`Differentiation`](crate::differentiation), [`batching`](crate::batching), and
//! [`partial evaluation`](crate::partial) operate on reference programs directly; full discharge is not a prerequisite.
//! Those modules own their policies for tangent and cotangent state, batched references, and effects across split
//! computations. The reference layer supplies identities, access analysis, and view reconstruction without defining
//! transform-specific argument roles.
//!
//! # Extending the Reference Model
//!
//! A downstream value family defines its views through [`ReferenceView`] and [`ReferenceViewOperation`].
//! Batching is optional and requires the separate [`BatchableReferenceView`] capability. Discharge uses a
//! [`ReferenceDischargePolicy`], selected through [`ReferenceDischargeableType`], to read and replace values
//! through that family's reference handles; additive updates also use [`ReferenceAccumulationPolicy`].
//!
//! Operations participate through [`ReferenceDischargeableOperation`]. Structured operations own their branch or loop
//! boundary rewrites, while [`ReferenceDischargeContext`] and [`ReferenceDischargeDriver`] provide state tracking,
//! region access, and rebuilding. This keeps array indexing and operation-specific control flow out of the generic
//! reference model and lets downstream families and operations supply their own rules.

use thiserror::Error;

/// Error produced while accessing the shared state of an eager [`Reference`] allocation, validating reference views,
/// or analyzing references in a program with [`ReferenceAnalysis`] or [`ReferenceViewAnalysis`].
#[derive(Clone, Debug, PartialEq, Eq, Hash, Error)]
pub enum ReferenceError {
    /// A reference allocation attempted to store another reference as its immediate referent.
    #[error("reference referent type `{referent_type}` must not itself be a reference")]
    NestedReferent {
        /// Rejected immediate referent type.
        referent_type: String,
    },

    /// A replacement or update result did not preserve the allocation's exact declared referent type.
    #[error("reference value type `{actual}` must exactly match declared referent type `{expected}`")]
    ReferentTypeMismatch {
        /// Exact declared referent type.
        expected: String,

        /// Actual replacement or update-result type.
        actual: String,
    },

    /// A handle-local metadata mapping could not reconstruct a value crossing the shared-state boundary.
    #[error("reference value reconstruction failed: {message}")]
    ValueReconstruction {
        /// Underlying value-family reconstruction diagnostic.
        message: String,
    },

    /// The reference and its complete alias family were invalidated by a consuming freeze.
    #[error("reference is frozen")]
    Frozen,

    /// The shared reference state exhausted its monotonically increasing mutation generation space.
    #[error("reference mutation generation is exhausted")]
    GenerationExhausted,

    /// The reference allocation's synchronization primitive was poisoned by a panic during an earlier access.
    #[error("reference state mutex is poisoned")]
    Poisoned,

    /// The shared reference state was invalidated after a stateful backend invocation crossed its irreversible
    /// execution boundary.
    #[error("reference state is poisoned: {reason}")]
    ExecutionPoisoned {
        /// Backend-owned reason the state can no longer be used safely.
        reason: String,
    },

    /// A program violates the reference rules checked by [`ReferenceAnalysis`]. Note that the underlying
    /// [`ReferenceAnalysisError`] is boxed in order to not unnecessarily increase the size of [`ReferenceError`].
    #[error(transparent)]
    Analysis(#[from] Box<ReferenceAnalysisError>),

    /// A reference view description is invalid for its source or declared output type.
    #[error(transparent)]
    ViewValidation(#[from] ReferenceViewValidationError),

    /// Reference view analysis rejects a program's view descriptions or paths. Note that the underlying
    /// [`ReferenceViewAnalysisError`] is boxed in order to not unnecessarily increase the size of [`ReferenceError`].
    #[error(transparent)]
    ViewAnalysis(#[from] Box<ReferenceViewAnalysisError>),
}

impl From<ReferenceAnalysisError> for ReferenceError {
    #[inline]
    fn from(error: ReferenceAnalysisError) -> Self {
        Self::Analysis(Box::new(error))
    }
}

impl From<ReferenceViewAnalysisError> for ReferenceError {
    #[inline]
    fn from(error: ReferenceViewAnalysisError) -> Self {
        Self::ViewAnalysis(Box::new(error))
    }
}

mod analysis;
mod discharge;
mod types;
mod values;
mod views;

pub use analysis::{
    ReferenceAccess, ReferenceAliasEdge, ReferenceAnalysis, ReferenceAnalysisError, ReferenceRegionInputBinding,
    ReferenceRoot, ReferenceTransitiveAccess,
};
pub use discharge::{
    ExternalReferenceBinding, PartialReferenceDischargeResult, RecursiveReferenceDischargeDriver,
    ReferenceAccumulationPolicy, ReferenceDischargeAllocationId, ReferenceDischargeBoundaryWidening,
    ReferenceDischargeContext, ReferenceDischargeDriver, ReferenceDischargePolicy, ReferenceDischargeReference,
    ReferenceDischargeRegionBoundary, ReferenceDischargeRegionBoundaryInsertion, ReferenceDischargeRegionInput,
    ReferenceDischargeRegionOutput, ReferenceDischargeRegionResult, ReferenceDischargeRegionSummary,
    ReferenceDischargeResult, ReferenceDischargeTarget, ReferenceDischargeValue, ReferenceDischargeableOperation,
    ReferenceDischargeableType, ReferenceSource, discharge_local_reference_operation,
    discharge_positional_region_operation, discharge_reference_free_operation,
};
pub use types::{ReferenceType, ReferenceTypeRefinements};
pub use values::{
    PreparedReferenceReplacement, ReadyOrPendingReferenceGuard, ReadyReferenceGuard, Reference, ReferenceBoundary,
    ReferenceBoundaryError, ReferenceBoundaryPosition, ReferenceCompletion, ReferenceCompletionBackend,
    ReferenceGeneration, ReferenceId, ReferenceIdentity, ReferenceObservation, ReferenceReplacementPreparation,
    ReferenceReplacementTransaction, TakenReferenceGuard, ValidatedPendingReplacementTransaction,
    validate_reference_boundary,
};
pub use views::{
    BatchableReferenceView, NoReferenceViewBinding, ReferenceView, ReferenceViewAnalysis, ReferenceViewAnalysisError,
    ReferenceViewOperation, ReferenceViewOverlap, ReferenceViewPath, ReferenceViewStep, ReferenceViewValidationError,
};

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::programs::instructions::InstructionId;
    use crate::programs::regions::RegionId;

    use super::*;

    #[test]
    fn test_reference_error() {
        let cases = [
            (
                ReferenceError::NestedReferent { referent_type: "ref<f32[2]>".to_string() },
                "reference referent type `ref<f32[2]>` must not itself be a reference",
            ),
            (
                ReferenceError::ReferentTypeMismatch { expected: "f32[2]".to_string(), actual: "f32[3]".to_string() },
                "reference value type `f32[3]` must exactly match declared referent type `f32[2]`",
            ),
            (
                ReferenceError::ValueReconstruction { message: "unbound identity".to_string() },
                "reference value reconstruction failed: unbound identity",
            ),
            (ReferenceError::Frozen, "reference is frozen"),
            (ReferenceError::GenerationExhausted, "reference mutation generation is exhausted"),
            (ReferenceError::Poisoned, "reference state mutex is poisoned"),
            (
                ReferenceError::ExecutionPoisoned { reason: "submission failed".to_string() },
                "reference state is poisoned: submission failed",
            ),
        ];
        for (error, expected) in cases {
            assert_eq!(error.to_string(), expected);
        }
        let error = ReferenceError::Frozen;
        assert_eq!(format!("{error:?}"), "Frozen");
        assert!(std::error::Error::source(&error).is_none());
    }

    #[test]
    fn test_reference_error_from_reference_analysis_error() {
        let analysis_error = ReferenceAnalysisError::UnresolvedReference {
            operation: "reference_read",
            instruction: InstructionId::new(RegionId::new(0), 1),
            input_index: 0,
        };
        let error = ReferenceError::from(analysis_error.clone());
        assert_eq!(error, ReferenceError::Analysis(Box::new(analysis_error.clone())));
        assert_eq!(
            error.to_string(),
            "operation `reference_read` at ^0[1] uses input 0 as a reference but it resolves to no reference root",
        );
        assert_eq!(format!("{error:?}"), format!("Analysis({analysis_error:?})"));
    }

    #[test]
    fn test_reference_error_from_reference_view_validation_error() {
        let validation_error =
            ReferenceViewValidationError::TypeMismatch { expected: "f32[3]".to_string(), actual: "f32[2]".to_string() };
        let error = ReferenceError::from(validation_error.clone());
        assert_eq!(error, ReferenceError::ViewValidation(validation_error));
        assert_eq!(
            error.to_string(),
            "view declares referent type `f32[2]` but derives referent type `f32[3]` from its source",
        );
    }

    #[test]
    fn test_reference_error_from_reference_view_analysis_error() {
        let analysis_error = ReferenceViewAnalysisError::MissingView {
            operation: "view",
            instruction: InstructionId::new(RegionId::new(0), 1),
            output_index: 0,
        };
        let error = ReferenceError::from(analysis_error.clone());
        assert_eq!(error, ReferenceError::ViewAnalysis(Box::new(analysis_error)));
        assert_eq!(
            error.to_string(),
            "operation `view` at ^0[1] declares a reference view at output 0 but describes no view",
        );
    }
}
