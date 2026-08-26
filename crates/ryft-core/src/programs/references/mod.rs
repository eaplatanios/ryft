//! Generic reference types, operations, eager holders, and staged discharge.
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
//!   contains no eager holder and no process-local resource identity.
//! - [`Reference<V>`] is the eager runtime handle. Its clones share one synchronized holder, so mutation through one
//!   handle is visible through every alias. A read returns an immutable snapshot, and a consuming
//!   [`freeze`](Reference::freeze) invalidates the complete alias family.
//! - [`ReferenceDischargeReference`] is a temporary handle used only while a program is being discharged. It names a
//!   root in the transform's environment and carries the policy-owned alias metadata for that particular handle.
//!
//! A reference family has one canonical root and any number of aliases. An alias preserves the root while possibly
//! selecting a narrower view. Every access resolves through that root, and consumption invalidates the whole family.
//!
//! # Roots, Aliases, and Views
//!
//! The reference vocabulary used throughout this module and its consumers is defined relative to one concept:
//!
//! - A **root** is the canonical mutable storage cell that a reference family denotes. Only
//!   [`reference_new`](ReferenceNewOperation) mints one. Eagerly, the root is the shared synchronized holder behind
//!   every [`Reference`] clone; in operation semantics, it is the identity that [`ReferenceOutput::Root`] introduces
//!   and [`ReferenceOutput::Alias`] preserves; during discharge, it is the unit of state threading, named by a
//!   [`ReferenceRootHandle`].
//! - The **referent** is the structural type of the value a handle exposes, written `ref<T>` as [`ReferenceType`].
//!   The root has its own referent — the type of the complete stored value — and a view's handle-local referent may
//!   be narrower.
//! - A **handle** is one name for a root: a program value of reference type, or an eager [`Reference`] clone.
//! - An **alias** is a handle derived from another handle. It always denotes the same root, either identically or
//!   through operation-owned view metadata ([`ReferenceAliasKind`]).
//! - A **view** is a narrowing alias, such as the result of
//!   [`reference_slice`](crate::arrays::ReferenceSliceOperation) or
//!   [`reference_index`](crate::arrays::ReferenceIndexOperation): it selects part of the root's value while every
//!   access through it still resolves to the root.
//! - The **alias family** is the complete set of handles denoting one root. Mutation through any member is visible
//!   through every other member.
//! - A **whole-root handle** exposes the root's complete stored value with no narrowing view. State that crosses a
//!   structured-region or discharge boundary is always whole-root; views are re-derived from the root inside the
//!   region that needs them.
//!
//! Every handle resolves to exactly one root: multi-source aliases (e.g., a hypothetical `select_reference(a, b)`)
//! are structurally unrepresentable rather than merely rejected, so analyses reason about state per root. Access-mode
//! summaries, discharge state threading, and race validation are per-root facts, and consumption
//! ([`reference_freeze`](ReferenceFreezeOperation)) is a whole-root lifetime event that invalidates the complete
//! family, which is why consuming through a narrowing view is rejected. Roots also split by provenance: a *local*
//! root is allocated inside the program and disappears entirely after discharge, while an *external* root denotes
//! caller-owned state entering through an input or capture ([`ReferenceSource`]) and is what a
//! [`ReferenceStateBinding`] describes to the backend.
//!
//! # Module Structure
//!
//! The implementation is split by responsibility:
//!
//! - `types.rs` defines the structural [`ReferenceType`] and its cross-occurrence refinements.
//! - `values.rs` defines the eager [`Reference`] value and its holder-facing operations.
//! - `semantics.rs` defines the operation-local [`ReferenceOperationSemantics`] descriptor, access modes, and
//!   root/alias classifications.
//! - `operations.rs` defines the six generic primitives and their value-level capabilities: allocation
//!   ([`ReferenceNew`]), immutable reads ([`ReferenceRead`]), write-only replacement ([`ReferenceWrite`]), swapping
//!   ([`ReferenceSwap`]), ordered additive updates ([`ReferenceAddUpdate`]), and consuming finalization
//!   ([`ReferenceFreeze`]). It also owns their type inference, effects, eager interpretation, and discharge rules.
//! - `runtime.rs` implements the synchronized holder state machine and hidden backend interface. It uses generations,
//!   completion dependencies, read leases, reservations, pending installation, and terminal poisoning to coordinate
//!   external state across synchronous and asynchronous execution.
//! - `discharge.rs` implements [`ReferenceDischarge`]: an interpreter-style transform that replaces selected mutable
//!   roots with explicitly threaded immutable values. Its policy, context, driver, and operation-rule contracts keep
//!   the transform open to non-array value families and to third-party operations.
//!
//! Structured operations own their reference boundary rewrites. For example, condition, while, and scan operations
//! decide how immutable state is added to their branch or loop boundaries; the discharge driver supplies isolated
//! region rebuilding, root summaries, and validation rather than choosing the rewrite for them.
//!
//! # Eager and Staged State
//!
//! Eager code acts directly on a [`Reference`] holder. Staged programs instead use the six reference operations.
//! Before a staged program reaches a backend that accepts only ordinary immutable values, discharge rewrites those
//! operations into explicit state dataflow. A local root disappears entirely after that rewrite. An external root
//! becomes an ordinary state input and, when mutated, a hidden final-state output described by a
//! [`ReferenceStateBinding`]; the backend's stateful invocation surface snapshots and publishes those values through
//! the caller's holder. Refer to the `discharge.rs` module documentation for a concrete before-and-after example.
//!
//! [`PartialReferenceDischargeResult`] supports the kernel use case in which selected implementation-owned roots
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
//! 2. The eager [`Reference`] holder rejects frozen, poisoned, conflicting, and stale-generation accesses while
//!    preserving atomic replacement semantics.
//! 3. Discharge validates the complete rewrite it observes, including use after consumption, unbound roots, invalid
//!    structured-region threading, escaping local allocations, and surviving references in a claimed full result.
//!
//! These checks are complementary. Construction sees the source call, the eager holder sees runtime aliases and
//! concurrency, and discharge sees the state-threading transformation and complete attached-region closure.

// TODO(eaplatanios): Review this module.

mod discharge;
mod operations;
mod runtime;
mod semantics;
mod types;
mod values;

pub use discharge::{
    PartialReferenceDischargeResult, RecursiveReferenceDischargeDriver, ReferenceAccumulationPolicy,
    ReferenceDischarge, ReferenceDischargeContext, ReferenceDischargeDriver, ReferenceDischargePayload,
    ReferenceDischargePolicy, ReferenceDischargeReference, ReferenceDischargeRegionDestination,
    ReferenceDischargeResult, ReferenceDischargeSite, ReferenceDischargeTracer, ReferenceDischargeValue,
    ReferenceDischargeableOperation, ReferenceRegionDischargeBoundary, ReferenceRegionDischargeFork,
    ReferenceRegionDischargeInput, ReferenceRegionSummary, ReferenceRootHandle, ReferenceRootState, ReferenceSource,
    ReferenceStateBinding, discharge_positional_region_operation, discharge_preserved_access,
    discharge_reference_free_operation,
};
pub use operations::{
    REFERENCE_ADD_UPDATE_OPERATION_NAME, REFERENCE_FREEZE_OPERATION_NAME, REFERENCE_NEW_OPERATION_NAME,
    REFERENCE_READ_OPERATION_NAME, REFERENCE_SWAP_OPERATION_NAME, REFERENCE_WRITE_OPERATION_NAME, ReferenceAddUpdate,
    ReferenceAddUpdateOperation, ReferenceFreeze, ReferenceFreezeOperation, ReferenceNew, ReferenceNewOperation,
    ReferenceRead, ReferenceReadOperation, ReferenceSwap, ReferenceSwapOperation, ReferenceWrite,
    ReferenceWriteOperation,
};
pub use runtime::{
    PendingReferenceReservation, PendingReferenceReservations, PreparedReferenceValue, ReferenceCompletion,
    ReferenceCompletionBackend, ReferenceCompletionCallback, ReferenceCompletionResult, ReferenceError,
    ReferenceGeneration, ReferenceGuard, ReferenceId,
};
pub use semantics::{
    ReferenceAccessMode, ReferenceAliasKind, ReferenceInput, ReferenceOperationSemantics, ReferenceOutput,
};
pub use types::{ReferenceType, ReferenceTypeRefinements};
pub use values::Reference;
