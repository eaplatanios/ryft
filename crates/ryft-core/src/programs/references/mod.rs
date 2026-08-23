//! Generic reference primitives, semantics, discharge, and eager runtime holders.
//!
//! This module owns value-family-independent reference concepts. Array view geometry and immutable array operation
//! replay remain in [`crate::arrays`].
//!
//! # Correspondence with JAX
//!
//! The vocabulary is JAX's, with the same division of labor. A [`ReferenceType`] is a `Ref`; three of the five
//! primitives in [`operations`] are `jax._src.state`'s `get`, `swap`, and `addupdate`, and the other two —
//! allocation and `freeze` — are JAX's mutable-array core surface rather than its state module;
//! [`ReferenceAccessMode`] mirrors `ReadEffect`/`WriteEffect`/`AccumEffect` plus consumption as a lifetime event; and
//! [`ReferenceDischarge`] is `discharge_state`, rewriting reference state into explicit immutable state threaded
//! through the program. Partial discharge follows `should_discharge`, which Pallas uses to discharge pipeline state
//! while keeping kernel references live, and [`ReferenceDischargeSite`] is the checked selection vocabulary that
//! names what to discharge.
//!
//! Two things are deliberately ahead of that correspondence. Read-only pruning keeps a root out of a structured
//! operation's widened boundary when no closure writes it, where `discharge_state` returns a final value for every
//! discharged reference. And the eager holders have typed runtime failure semantics — generations, read leases,
//! poisoning — where JAX's reference model is staged. Two further properties are load-bearing here without being
//! deltas: reference accesses carry ordered-state effects, so the ordering machinery sees them; and eager and staged
//! reference semantics share one view traversal, so they are held to each other by construction.
//!
//! # The prevention ladder
//!
//! Reference misuse is caught at three rungs, each reporting against what it can actually see. The rungs are not
//! redundant: each one exists because the rung above it cannot observe the program the rung below receives.
//!
//! 1. **Construction time**, within the region being built. `ProgramBuilder::add_instruction` maintains the alias
//!    family of every reference atom in the region under construction and rejects an access to a consumed family, or
//!    a consumption of a handle whose view narrows its root, at the append that performs it — for a traced program,
//!    that is the staging call. Constant lifting ([`Context::lift`](crate::Context::lift)) likewise rejects a value
//!    family that forbids constant storage at the lift. This rung sees the *call*, which is the most useful thing to
//!    name, and it is keyed by the atom, so every clone of one tracer shares it. Its horizon is one region: a builder
//!    constructs exactly one, so a consumption performed inside an attached region is invisible here and is left to
//!    the rungs below. `ProgramBuilder::add_instruction_unchecked` bypasses the rung for rebuilds of
//!    already-accepted programs and for tests that need a malformed program the rungs below reject.
//! 2. **The eager runtime**, while handles are used directly. [`Reference`] invalidates a complete alias family on
//!    [`freeze`](Reference::freeze) and reports every later access as [`ReferenceError::Frozen`], with generations
//!    and read leases covering concurrent and asynchronous misuse. This rung sees runtime identity, which no static
//!    rung has. The eager interpretation boundary validation belongs to this rung: interpretation rejects a program
//!    whose entry expects an external reference input, because generic interpretation has no holder binding table.
//! 3. **Discharge**, while the rewrite runs. Its root environment reports what it observes — an access to a
//!    consumed root, a derived view crossing a structured boundary, an escaping region-local allocation, a mutation
//!    the widening did not predict — against the environment root it reached rather than against a source
//!    coordinate, because its root handles are interpreter identities; the structured-boundary rejections
//!    additionally name the operation whose rule raised them. This rung sees the interpreter's own state, which the
//!    static rungs approximate.
//!
//! There is deliberately no standing whole-program lint between the second and third rungs. A static analysis that
//! resolved roots, aliases, and capture scopes over a complete region closure existed and was removed once its only
//! production consumers reduced to facts derivable at the entry boundary; a program built outside tracing meets its
//! reference rules when it is discharged, which every staged consumer requires anyway. Kernel-style validation of
//! preserved reference bodies is planned to reintroduce a closure analysis as part of `plan-pallas.md`.
//!
//! Rung 1 is the one worth seeing, because it is the rung a user meets while writing the program:
//!
//! ```
//! use ryft_core::{
//!     Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayType, DataType, FreezeReference, NewReference,
//!     ProgramError, ReferenceRead, Trace, Tracer, TracingContext,
//! };
//!
//! type Destination = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;
//!
//! // Freezing consumes the handle it is given, so a second handle has to be cloned deliberately — and that clone
//! // names the same staged atom, which is what the trace notices.
//! let error = Destination::trace(
//!     |input: Tracer<Destination>| {
//!         let reference = input.new_reference()?;
//!         let alias = reference.clone();
//!         reference.freeze()?;
//!         alias.read()
//!     },
//!     ArrayIrType::Array(ArrayType::scalar(DataType::F32)),
//! )
//! .unwrap_err();
//!
//! assert_eq!(
//!     error,
//!     ProgramError::MalformedProgram(
//!         "`reference_read` reads a reference whose alias family `freeze_reference` already consumed".to_string(),
//!     ),
//! );
//! ```

// TODO(eaplatanios): Review this whole module, its submodules, and all of the documentation and tests.

mod discharge;
mod operations;
mod runtime;
mod semantics;

pub use discharge::{
    PartialReferenceDischargeResult, RecursiveReferenceDischargeDriver, ReferenceAccumulationPolicy,
    ReferenceDischarge, ReferenceDischargeContext, ReferenceDischargeDriver, ReferenceDischargePolicy,
    ReferenceDischargeReference, ReferenceDischargeRegionDestination, ReferenceDischargeResult, ReferenceDischargeSite,
    ReferenceDischargeTracer, ReferenceDischargeValue, ReferenceDischargeableOperation,
    ReferenceRegionDischargeBoundary, ReferenceRegionDischargeFork, ReferenceRegionDischargeInput,
    ReferenceRegionSummary, ReferenceRootHandle, ReferenceRootState, ReferenceStateBinding,
    discharge_positional_region_operation, discharge_preserved_access, discharge_reference_free_operation,
};
pub use operations::{
    FREEZE_REFERENCE_OPERATION_NAME, FreezeReference, FreezeReferenceOperation, NEW_REFERENCE_OPERATION_NAME,
    NewReference, NewReferenceOperation, REFERENCE_ADD_UPDATE_OPERATION_NAME, REFERENCE_READ_OPERATION_NAME,
    REFERENCE_SWAP_OPERATION_NAME, ReferenceAddUpdate, ReferenceAddUpdateOperation, ReferenceRead,
    ReferenceReadOperation, ReferenceSwap, ReferenceSwapOperation,
};
pub use runtime::{
    PreparedReferenceValue, Reference, ReferenceCompletion, ReferenceCompletionBackend, ReferenceCompletionCallback,
    ReferenceCompletionResult, ReferenceError, ReferenceGeneration, ReferenceGuard, ReferenceId,
};
pub(crate) use semantics::ReferenceLifetimes;
pub use semantics::{
    ReferenceAccessMode, ReferenceAliasKind, ReferenceInputAccess, ReferenceOperationSemantics,
    ReferenceOutputSemantics, ReferenceSource, ReferenceType, ReferenceTypeRefinements,
};
