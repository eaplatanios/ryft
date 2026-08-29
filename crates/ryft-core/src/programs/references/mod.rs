//! Generic reference types, operations, eager runtime state, and staged discharge.
//!
//! References are Ryft's second-class mutable-state values. A reference may be created, aliased, read, replaced,
//! updated, and consumed inside a program, but it is not ordinary immutable data: numeric operations cannot consume
//! it directly, a local reference cannot escape as a public output, and an external reference denotes state owned by
//! the caller. Reference operations carry ordered-state effects so that optimization and transformation machinery
//! cannot reorder or duplicate them as if they were pure computations.
//!
//! This module owns the value-family-independent reference language. It does not assume that a referent is an array
//! or that an alias is an array view. Array-specific view geometry, eager view traversal, and the array discharge
//! policy live in [`crate::arrays`].
//!
//! # Core Model
//!
//! Three similarly named types serve different stages of the system:
//!
//! - [`ReferenceType<T>`](ReferenceType) is structural program metadata. It says that a value refers to a `T`, but
//!   contains no eager state and no process-local resource identity.
//! - [`Reference<V>`] is the eager runtime handle. Its clones share the synchronized state of one reference
//!   allocation, so mutation through one handle is visible through every alias. A read returns an immutable snapshot,
//!   and a consuming [`freeze`](Reference::freeze) invalidates the complete alias family.
//! - [`ReferenceDischargeReference`] is a temporary handle used only while a program is being discharged. It names a
//!   reference allocation in the transform's environment and carries the policy-owned alias metadata for that handle.
//!
//! A reference family has one canonical allocation and any number of aliases. An alias preserves the allocation while
//! possibly selecting a narrower view. Every access resolves through that allocation, and consumption invalidates the
//! complete family.
//!
//! # Allocations, Aliases, and Views
//!
//! The reference terms used throughout this module and its consumers are defined relative to one concept:
//!
//! - A **reference allocation** is the canonical mutable storage cell that a reference family denotes. Only
//!   [`reference_new`](ReferenceNewOperation) mints one. Eagerly, the allocation is the reference allocation whose
//!   synchronized state every [`Reference`] clone shares; in operation semantics, it is the identity that
//!   [`ReferenceOutput::Allocation`] introduces and [`ReferenceOutput::Alias`] preserves; during discharge, it is the
//!   unit of state threading, named by a [`ReferenceAllocationHandle`].
//! - The **referent** is the structural type of the value a handle exposes, written `ref<T>` as [`ReferenceType`].
//!   The allocation has its own referent — the type of the complete stored value — and a view's handle-local referent may
//!   be narrower.
//! - A **handle** is one name for an allocation: a program value of reference type, or an eager [`Reference`] clone.
//! - An **alias** is a handle derived from another handle. It always denotes the same allocation, either identically or
//!   through operation-owned view metadata ([`ReferenceAliasKind`]).
//! - A **view** is a narrowing alias, such as the result of
//!   [`reference_slice`](crate::arrays::ReferenceSliceOperation) or
//!   [`reference_index`](crate::arrays::ReferenceIndexOperation): it selects part of the allocation's value while every
//!   access through it still resolves to the allocation.
//! - The **alias family** is the complete set of handles denoting one allocation. Mutation through any member is visible
//!   through every other member.
//! - A **complete-value handle** exposes the allocation's complete stored value with no narrowing view. State that
//!   crosses a structured-region or discharge boundary is always represented by a complete-value handle; views are
//!   re-derived from that handle inside the region that needs them.
//!
//! Every handle resolves to exactly one allocation: multi-source aliases (e.g., a hypothetical `select_reference(a, b)`)
//! are structurally unrepresentable rather than merely rejected, so analyses reason about state per allocation.
//! Access-mode summaries, discharge state threading, and race validation are per-allocation facts, and consumption
//! ([`reference_freeze`](ReferenceFreezeOperation)) is a complete-value lifetime event that invalidates the complete
//! family, which is why consuming through a narrowing view is rejected. Allocations also split by provenance: a *local*
//! allocation is created inside the program and disappears entirely after discharge, while an *external* allocation
//! denotes caller-owned state entering through an input or capture ([`ReferenceSource`]) and is what a
//! [`ExternalReferenceBinding`] describes to the backend.
//!
//! # Module Structure
//!
//! The implementation is split by responsibility:
//!
//! - `types.rs` defines the structural [`ReferenceType`] and its cross-occurrence refinements.
//! - `values.rs` defines the eager [`Reference`] value, coherent backend [`ReferenceObservation`]s, backend-neutral
//!   completion dependencies, and the synchronized state machine for each reference allocation, including identity,
//!   generations, guards, read leases, pending completion, and terminal poisoning.
//! - `semantics.rs` defines the operation-local [`ReferenceOperationSemantics`] descriptor, access modes, and
//!   allocation/alias classifications.
//! - `operations/` defines the six generic primitives in separate modules together with their value-level
//!   capabilities: allocation ([`ReferenceNew`]), immutable reads ([`ReferenceRead`]), write-only replacement
//!   ([`ReferenceWrite`]), swapping ([`ReferenceSwap`]), ordered additive updates ([`ReferenceAddUpdate`]), and
//!   consuming finalization ([`ReferenceFreeze`]). Each primitive module also owns its type inference, effects, eager
//!   interpretation, discharge rule, and unit tests.
//! - `discharge/` implements [`ReferenceDischarge`]: an interpreter-style transform that replaces selected mutable
//!   allocations with explicitly threaded immutable values. Its policy, context, driver, and operation-rule contracts
//!   keep the transform open to non-array value families and to third-party operations.
//!
//! Structured operations own their reference boundary rewrites. For example, condition, while, and scan operations
//! decide how immutable state is added to their branch or loop boundaries; the discharge driver supplies isolated
//! region rebuilding, allocation summaries, and validation rather than choosing the rewrite for them. The one rewrite
//! the replay path owns itself is preserved-access replay: a region-free, access-only application over exclusively
//! preserved references is replayed verbatim before any rule runs.
//!
//! # Eager and Staged State
//!
//! Eager code acts directly through a [`Reference`] handle. Staged programs instead use the six reference operations.
//! Before a staged program reaches a backend that accepts only ordinary immutable values, discharge rewrites those
//! operations into explicit state dataflow. A local allocation disappears entirely after that rewrite. An external
//! allocation becomes an ordinary state input and, when mutated, a hidden final-state output described by a
//! [`ExternalReferenceBinding`]; the backend's stateful invocation surface snapshots and publishes those values through
//! the caller's reference. [`ReferenceDischarge`] exposes the value-level entry point for this rewrite; the discharge
//! module documentation contains a concrete before-and-after example.
//!
//! [`PartialReferenceDischargeResult`] supports the kernel use case in which selected implementation-owned allocations
//! become immutable state while other references deliberately remain in the program. A full
//! [`ReferenceDischargeResult`] additionally proves that no reference type or reference operation survives anywhere
//! in the rewritten region closure.
//!
//! # Lifetime Enforcement
//!
//! Reference validity is enforced at the earliest layer that has enough information:
//!
//! 1. [`ProgramBuilder::add_instruction`](crate::ProgramBuilder::add_instruction) tracks aliases within the region
//!    under construction and rejects an access after consumption or consumption through a narrowing view.
//! 2. The eager [`Reference`] rejects frozen, poisoned, conflicting, and stale-generation accesses while preserving
//!    atomic replacement semantics across its alias family.
//! 3. Discharge validates the complete rewrite it observes, including use after consumption, unbound allocations,
//!    invalid structured-region threading, escaping local allocations, and surviving references in a claimed full result.
//!
//! These checks are complementary. Construction sees the source call, the eager reference sees runtime aliases and
//! concurrency, and discharge sees the state-threading transformation and complete attached-region closure.

// TODO(eaplatanios): Review this module.

use thiserror::Error;

/// Error produced while accessing the shared state of an eager [`Reference`] allocation.
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
}

mod discharge;
mod operations;
mod semantics;
mod types;
mod values;

pub use discharge::{
    ExternalReferenceBinding, PartialReferenceDischargeResult, RecursiveReferenceDischargeDriver,
    ReferenceAccumulationPolicy, ReferenceAllocationHandle, ReferenceDischarge, ReferenceDischargeContext,
    ReferenceDischargeDriver, ReferenceDischargePolicy, ReferenceDischargeReference,
    ReferenceDischargeRegionDestination, ReferenceDischargeResult, ReferenceDischargeTarget, ReferenceDischargeValue,
    ReferenceDischargeableOperation, ReferenceRegionDischargeBoundary, ReferenceRegionDischargeFork,
    ReferenceRegionStateInsertion, ReferenceRegionSummary, ReferenceSource, ReferenceStateWidening,
    discharge_positional_region_operation, discharge_preserved_access, discharge_reference_free_operation,
};
pub use operations::{
    REFERENCE_ADD_UPDATE_OPERATION_NAME, REFERENCE_FREEZE_OPERATION_NAME, REFERENCE_NEW_OPERATION_NAME,
    REFERENCE_READ_OPERATION_NAME, REFERENCE_SWAP_OPERATION_NAME, REFERENCE_WRITE_OPERATION_NAME, ReferenceAddUpdate,
    ReferenceAddUpdateOperation, ReferenceFreeze, ReferenceFreezeOperation, ReferenceNew, ReferenceNewOperation,
    ReferenceRead, ReferenceReadOperation, ReferenceSwap, ReferenceSwapOperation, ReferenceWrite,
    ReferenceWriteOperation,
};
pub use semantics::{
    ReferenceAccessMode, ReferenceAliasKind, ReferenceInput, ReferenceOperationSemantics, ReferenceOutput,
};
pub use types::{ReferenceType, ReferenceTypeRefinements};
pub use values::{
    PreparedReferenceReplacement, ReadyOrPendingReferenceGuard, ReadyReferenceGuard, Reference, ReferenceCompletion,
    ReferenceCompletionBackend, ReferenceGeneration, ReferenceId, ReferenceObservation,
    ReferenceReplacementPreparation, ReferenceReplacementTransaction, TakenReferenceGuard,
    ValidatedPendingReplacementTransaction,
};

#[cfg(test)]
mod tests {
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
}
