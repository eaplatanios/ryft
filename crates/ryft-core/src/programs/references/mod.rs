//! Generic reference types, operations, eager runtime state, and staged discharge.
//!
//! References are Ryft's second-class mutable-state values. A reference may be created, aliased, read, replaced,
//! updated, and consumed inside a program, but it is not immutable value data: numeric operations cannot consume it
//! directly, a local reference escapes as a public output only to a caller that keeps it as a reference (discharge and
//! backend lowering reject it), and an external reference denotes state owned by the caller. Reference operations carry
//! ordered-state effects so that optimization and transformation machinery cannot reorder or duplicate them as if they
//! were pure computations, while the transforms themselves operate on references directly (refer to the "Transforms"
//! section below).
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
//!   [`reference_new`](crate::operations::ReferenceNewOperation) mints one. Eagerly, the allocation is the reference
//!   allocation whose synchronized state every [`Reference`] clone shares; in operation effect declarations, it is the
//!   identity that [`ReferenceEffect::Allocate`](crate::ReferenceEffect::Allocate) introduces and
//!   [`ReferenceAlias`](crate::ReferenceAlias) preserves; during discharge, it is the unit of state threading, named by
//!   a [`ReferenceDischargeAllocationId`].
//! - The **referent** is the structural type of the value a handle exposes, written `ref<T>` as [`ReferenceType`].
//!   The allocation has its own referent — the type of the complete stored value — and a view's handle-local referent may
//!   be narrower.
//! - A **handle** is one name for an allocation: a program value of reference type, or an eager [`Reference`] clone.
//! - An **alias** is a reference created from another reference. It always denotes the same allocation, either
//!   identically or through operation-owned view metadata ([`ReferenceAliasKind`](crate::ReferenceAliasKind)).

//! - A **view** is a narrowing alias, such as the result of
//!   [`reference_slice`](crate::arrays::ReferenceSliceOperation) or
//!   [`reference_index`](crate::arrays::ReferenceIndexOperation): it selects part of the allocation's value while every
//!   access through it still resolves to the allocation.
//! - The **alias family** is the complete set of handles denoting one allocation. Mutation through any member is visible
//!   through every other member.
//! - A **complete-value handle** exposes the allocation's complete stored value with no narrowing view. State that
//!   crosses a structured-region or discharge boundary is always represented by a complete-value handle; views are
//!   created from that reference inside the region that needs them.
//!
//! Every handle resolves to exactly one allocation: multi-source aliases (e.g., a hypothetical `select_reference(a,
//! b)`) are structurally unrepresentable rather than merely rejected, so analyses reason about state per allocation.
//! Access-mode summaries, discharge state threading, and race validation are per-allocation facts, and consumption
//! ([`reference_freeze`](crate::operations::ReferenceFreezeOperation)) is a complete-value lifetime event that
//! invalidates the complete family, which is why consuming through a narrowing view is rejected. Allocations also split
//! by provenance: a *local* allocation is created inside the program and disappears entirely after discharge, while an
//! *external* allocation denotes caller-owned state entering through an input or capture ([`ReferenceSource`]) and is
//! what a [`ExternalReferenceBinding`] describes to the backend.
//!
//! # Module Structure
//!
//! The implementation is split by responsibility:
//!
//! - `types.rs` defines the structural [`ReferenceType`] and its cross-occurrence refinements.
//! - `values.rs` defines the eager [`Reference`] value, coherent backend [`ReferenceObservation`]s, backend-neutral
//!   completion dependencies, and the synchronized state machine for each reference allocation, including identity,
//!   generations, guards, read leases, pending completion, and terminal poisoning.
//! - `analysis.rs` defines the generic program-level [`ReferenceAnalysis`]: canonical [`ReferenceRoot`]s, alias
//!   edges, accesses, capture scopes, region boundaries (complete handles, plus the views an operation creates for its
//!   own region inputs), lifetime validation, and per-instruction transitive access summaries. It is kernel-owned
//!   validation infrastructure invoked explicitly by its consumers rather than a standing lint on every program.
//! - `views.rs` defines the value-family-generic view contract [`ReferenceViewOperation`] (owned per-edge view
//!   descriptions with static or symbolic coordinates, their type-level validation, their reapplication to a
//!   transformed reference, their batching, and their overlap query) and the retained [`ReferenceViewAnalysis`]
//!   overlay that composes those descriptions into one [`ReferenceViewPath`] per reference-typed value of a closure.
//! - [`crate::operations::references`] defines the six generic primitives in separate modules together with their
//!   value-level capabilities: allocation ([`ReferenceNew`](crate::operations::ReferenceNew)), immutable reads
//!   ([`ReferenceRead`](crate::operations::ReferenceRead)), write-only replacement
//!   ([`ReferenceWrite`](crate::operations::ReferenceWrite)), swapping ([`ReferenceSwap`](crate::operations::ReferenceSwap)),
//!   ordered additive updates ([`ReferenceAddUpdate`](crate::operations::ReferenceAddUpdate)), and consuming finalization
//!   ([`ReferenceFreeze`](crate::operations::ReferenceFreeze)). Each primitive module also owns its type inference,
//!   effects, eager interpretation, discharge rule, and unit tests.
//! - `discharge.rs` implements an interpreter-style transform that replaces selected mutable allocations with
//!   explicitly threaded immutable values. Its policy, context, driver, and operation-rule contracts keep the transform
//!   open to non-array value families and to third-party operations.
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
//! Before a staged program reaches a backend that accepts only immutable values, discharge rewrites those
//! operations into explicit state dataflow. A local allocation disappears entirely after that rewrite. An external
//! allocation becomes a state value input and, when mutated, a hidden final-state output described by a
//! [`ExternalReferenceBinding`]; the backend's stateful invocation surface snapshots and publishes those values through
//! the caller's reference. [`Program::discharge_references`](crate::Program::discharge_references) exposes the generic
//! program-level entry point for this rewrite; the discharge module documentation contains a concrete before-and-after
//! example.
//!
//! [`PartialReferenceDischargeResult`] supports the kernel use case in which selected implementation-owned allocations
//! become immutable state while other references deliberately remain in the program. A full
//! [`ReferenceDischargeResult`] additionally proves that no reference type or reference operation survives anywhere
//! in the rewritten region closure.
//!
//! # Transforms
//!
//! Transforms operate directly on references, using cached [`ReferenceAnalysis`] for roots, aliases, and access modes:
//!
//! - Forward mode pairs active primal references with caller-supplied tangent references. Linearization executes primal
//!   accesses and stages tangent accesses, using a dedicated partial-evaluation mode that preserves each root's order.
//! - Reverse mode accumulates state cotangents into references selected through
//!   [`CotangentDestination`](crate::CotangentDestination).
//! - Batching gives a reference a batch axis by giving its referent one.
//! - Ordinary partial evaluation preserves global ordered-effect and failure order, threading live references into the
//!   residual program. Rematerialization recomputes local lifecycles and saves external reads.
//! - Custom derivatives, `jit_call`, and `shard_map` thread references positionally under the contracts below.
//!
//! Reference types are never zero-space: their tangents and cotangents are references over the referent's tangent and
//! cotangent types. [`MaybeZero::Zero`](crate::MaybeZero::Zero) marks an inactive reference tangent or an unallocated
//! cotangent inside transforms; generic zero materialization rejects it. Internal region JVP and linearization
//! boundaries omit inactive reference tangent outputs while preserving the primal handle's identity across calls and
//! control flow. The lifetime-enforcement section below describes validation of runtime and staged boundary identities.
//!
//! # Reference Semantics and JAX Comparison
//!
//! The following contracts describe supported reference behavior and explicit restrictions. Comparisons identify
//! specific differences or equivalent capabilities; they do not imply complete transform parity.
//!
//! - **Reference-typed carries and outputs with positional identity.** A structured operation may return a reference
//!   when it states which input root it forwards, and a program may return an escaping local allocation to a caller
//!   that keeps it as a reference. JAX restricts reference outputs from
//!   [jitted functions and higher-order bodies](https://docs.jax.dev/en/latest/array_refs.html#restrictions).
//!   Forward mode forwards the tangent reference by identity, reverse mode shares the input's accumulator, batching
//!   carries the root's axis, and backend lowering rejects escaping allocations because the stateful ABI has no result
//!   reconstruction protocol.
//! - **Partial discharge preserving internal allocations.**
//!   [`Program::partially_discharge_references`](crate::Program::partially_discharge_references) rewrites only the
//!   selected allocations and leaves the others as live references. A kernel pipeline can normalize its own state
//!   while preserving references that a later kernel lowering consumes.
//! - **`condition` with an unbatched predicate mutating references under `vmap`.** When the predicate is replicated,
//!   only the selected branch runs, so its reference mutations execute unmasked; a batched predicate rejects any
//!   transitive reference access or local allocation in either branch, because effectful state cannot be masked per
//!   batch item. JAX's current
//!   [`cond` batching rule](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/control_flow/conditionals.py) rejects
//!   branch reference effects even with an unbatched predicate; its batched-predicate path uses selection.
//! - **Mutating `while` conditions rotated with shared writes allowed.** Discharge rotates a `while` whose condition
//!   mutates references into do-while form and admits a root written by both the condition and the body (body first,
//!   then condition). The current
//!   [JAX state-discharge rule](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/control_flow/loops.py) rejects
//!   writes to the same reference in both regions.
//! - **Explicit cotangent destinations.** [`CotangentDestination`](crate::CotangentDestination) selects returned
//!   values, accumulation into caller-owned references, or discarded gradients. This is comparable to
//!   [JAX's `VJP.with_refs`](https://docs.jax.dev/en/latest/301/refs.html#gradient-refs-for-value-arguments), which
//!   supports reference gradient destinations for both reference and array arguments; reference gradients are not a
//!   divergence by themselves.
//! - **Reference ownership under `shard_map`.** Ryft makes reference inputs read-only along replicated manual axes;
//!   writes require an owned shard. JAX also supports explicitly passed reference operands: its
//!   [reference-argument test](https://github.com/jax-ml/jax/blob/main/tests/shard_map_test.py#L4238-L4249) updates a
//!   sharded reference, and its
//!   [discharge rule](https://github.com/jax-ml/jax/blob/main/jax/_src/shard_map.py#L2002-L2031)
//!   returns updated reference values under the corresponding input specs. JAX's documented restriction concerns
//!   [functions that close over references](https://docs.jax.dev/en/latest/array_refs.html#restrictions).
//!   Ryft's replicated-read-only rule is a separate ownership policy; support for reference operands and support for
//!   captured references must be compared separately.
//! - **Custom-derivative reference outputs rejected outright.** `custom_jvp` and `custom_vjp` accept references only
//!   in their leading non-differentiated segment and never as outputs, even where a forwarded output would be
//!   well-defined, because neither derivative interface can name the output's tangent or cotangent reference.
//! - **Scanning over a reference stack.** A `scan` accepts a reference-typed stacked operand and presents its body
//!   the per-iteration slice of the referent as a reference view created at the region boundary, which forward mode,
//!   transposition, and batching restate with the scan and which discharge rewrites into the scan's own stacked operand
//!   and stacked output. Presenting the slice as a view lets state be indexed per iteration without dynamic indexing
//!   and discharges to the scan's own stacking rather than to a gather and scatter per iteration.
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
//! 4. Transform boundaries reject the same allocation at two positions, a reference both captured and passed, and
//!    values that misreport their identity. [`validate_reference_boundary`] checks concrete values by [`ReferenceId`].
//!    [`ReferenceBoundary`] resolves runtime and staged identities through the context, accepting caller-defined
//!    positions and validation order. Each transform owns its argument roles and aliasing policy, including whether
//!    later arguments must avoid retained allocations. Program interpretation performs no such boundary check.
//!
//! These checks are complementary. Construction sees the source call but only atoms and declared effects, so it
//! cannot detect two positions bound to the same runtime allocation; the eager reference sees runtime aliases and
//! concurrency; discharge sees the state-threading transformation and complete attached-region closure; and the
//! boundary validator sees the live values a transform is about to bind.

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

mod analysis;
mod discharge;
mod types;
mod values;
mod views;

pub use analysis::{
    ReferenceAccess, ReferenceAliasEdge, ReferenceAliasOrigin, ReferenceAnalysis, ReferenceAnalysisError,
    ReferenceRegionInputBinding, ReferenceRoot, ReferenceTransitiveAccess,
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
    NoBinding, ReferenceView, ReferenceViewAnalysis, ReferenceViewAnalysisError, ReferenceViewOperation,
    ReferenceViewPath, ReferenceViewStep, ReferenceViewValidationError, ViewOverlap, ViewSymbol, ViewSymbolBinding,
    batch_reference_view_operation,
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
