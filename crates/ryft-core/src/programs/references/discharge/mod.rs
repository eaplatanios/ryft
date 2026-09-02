//! Contains machinery related to _reference discharge_, which is the process of rewriting mutable reference state into
//! explicit immutable dataflow. Specifically, a program containing reference operations is not purely functional
//! Single Static Assignment (SSA) dataflow. A read operation depends on the latest write operation to the same
//! allocation even though that dependency is represented by a reference handle rather than by an SSA operand carrying
//! the current value. Many transforms and backends require the dependency to be explicit. Reference discharge makes it
//! explicit by replaying the program while replacing each selected allocation with an immutable state value that is
//! threaded from one access to the next.
//!
//! # Example
//!
//! Consider this schematic program with one local reference:
//!
//! ```text
//! %reference = reference_new(%initial)
//! %before = reference_swap(%reference, %replacement)
//! %after = reference_read(%reference)
//! %final = reference_freeze(%reference)
//! return %before, %after, %final
//! ```
//!
//! Discharge removes the mutable allocation and exposes the same dependencies as immutable values:
//!
//! ```text
//! %state0 = %initial
//! %before = %state0
//! %state1 = %replacement
//! %after = %state1
//! %final = %state1
//! return %before, %after, %final
//! ```
//!
//! The exact rewritten program may simplify aliases such as `%state0`, but the semantic relationship is the same:
//! replacement returns the previous state and produces a successor state, while every subsequent access consumes that
//! successor. The reference is an implementation detail of the source program, and no reference survives in the full
//! result.
//!
//! An external reference follows the same rewrite but its state crosses the program boundary. Its reference-typed input
//! becomes a value input carrying the entering referent. If the program mutates the reference, its final state is
//! appended after the public outputs as a hidden output. [`ExternalReferenceBinding`] records which capture or public
//! argument owns that state and which hidden output must replace the caller's reference value. Its discharged input
//! position follows from that logical source and the result's capture count. A read-only external reference has no
//! hidden output.
//!
//! Discharge rewrites a program. It does not itself lock eager reference state or execute a backend. The stateful
//! compilation surface uses the result's binding metadata together with the runtime reference protocol after
//! compilation.
//!
//! # Using Reference Discharge
//!
//! Callers normally invoke [`Program::discharge_references`](crate::Program::discharge_references) to eliminate every
//! reference, or [`Program::partially_discharge_references`](crate::Program::partially_discharge_references) when
//! selected allocations should become explicit state while
//! other allocations remain references. The full entry point returns a [`ReferenceDischargeResult`], whose program is
//! proven reference-free. The partial entry point returns a [`PartialReferenceDischargeResult`], which describes only
//! the discharged references and can be converted into a full result after proving that no references remain. A
//! caller that needs a bare program with no caller-owned reference bindings converts a full result through
//! [`ReferenceDischargeResult::into_program_without_external_references`].
//!
//! A reference universe participates by implementing [`ReferenceDischargePolicy`], selecting that policy through
//! [`ReferenceDischargeableType`], and, when supported, implementing [`ReferenceAccumulationPolicy`]. Each operation
//! implements [`ReferenceDischargeableOperation`] to rewrite its own reference effects. Region-free operations can
//! delegate to [`discharge_reference_free_operation`]; structured operations use [`ReferenceDischargeDriver`],
//! [`ReferenceDischargeRegionSummary`], and
//! [`ReferenceDischargeRegionBoundary`] to rebuild attached regions with the necessary state positions.
//!
//! # Full and Partial Discharge
//!
//! [`ReferenceDischargeResult`] is the full-discharge contract. Its program is proven reference-free across the
//! complete attached-region closure, its public outputs form a prefix of its complete outputs, and the remaining
//! suffix contains exactly the final states of mutated external allocations in canonical boundary order.
//!
//! [`PartialReferenceDischargeResult`] permits selected allocations to become immutable state while unselected references
//! and their operations remain in the program. Callers select external references or internal allocations through
//! [`ReferenceDischargeTarget`]. This is useful when normalizing a pipeline's internal state while deliberately
//! preserving references that a kernel will lower to target memory operations. Conversion from a partial result to a
//! full result performs a closure-wide proof that no reference type or reference operation remains.
//!
//! # Interpreter Architecture
//!
//! Discharge follows Ryft's context-and-per-operation-rule transform architecture:
//!
//! - [`ReferenceDischargePolicy`] names everything that varies by reference universe: the referent type family, the
//!   handle's composed alias metadata, type lifting and projection, and the mechanics of reading and replacing a
//!   selected value. [`ReferenceAccumulationPolicy`] adds ordered accumulation only for universes that support it.
//! - [`ReferenceDischargeValue`] is the context-free carrier flowing between rules. It contains either a destination
//!   value or an opaque [`ReferenceDischargeReference`] handle.
//! - [`ReferenceDischargeContext`] owns the live allocation environment. Each allocation is either `Discharged`, with a
//!   current immutable state and mutation bit, or `Preserved`, with the exact destination reference value that survived.
//! - [`ReferenceDischargeableOperation`] is the rule implemented by each operation. Reference primitives rewrite
//!   their own *discharged* accesses, structured operations own their boundary widening, and reference-free
//!   operations replay unchanged through the parent context. Accesses to preserved references never reach a rule: the
//!   replay path itself replays every region-free, access-only application over exclusively preserved references
//!   verbatim.
//! - [`ReferenceDischargeDriver`] exposes the current source instruction and attached regions. It can replay a region
//!   against the live environment or rebuild one against an isolated environment and return a sealed
//!   [`ReferenceDischargeRegionResult`]. [`ReferenceDischargeRegionSummary`] supplies the transitive access facts a
//!   structured rule needs before choosing its state boundary.
//!
//! The driver provides shared mechanics but never chooses how an operation is rewritten. This keeps the system open:
//! a third-party primitive or structured operation participates by implementing its own rule, while a non-array
//! value family supplies its own policy without changing this interpreter.
//!
//! # Allocation Identities and Boundaries
//!
//! [`ReferenceDischargeTarget`] names a source program location used before replay to select an external reference or
//! a locally allocated reference. [`ReferenceDischargeAllocationId`] is different: it is a temporary identity minted
//! inside one live discharge environment. IDs from isolated region rebuilds cannot address parent allocations, and
//! [`ReferenceDischargeRegionResult`] carries a sealed program and context-free summaries rather than temporary values.
//!
//! Structured rules use [`ReferenceDischargeRegionBoundary`] to describe their declared inputs plus the discharged
//! allocations that must enter and leave a rebuilt region. Read-only allocations are pruned where the operation's boundary permits
//! it; loop-shaped operations retain the symmetry their fixed-point contracts require. Every rebuilt region is
//! validated against the allocations and mutations its summary predicted before the parent environment accepts its outputs.
//!
//! # End-to-End Flow
//!
//! 1. The program entry point validates the selected targets and binds each external reference as either discharged
//!    state or a preserved reference in a new [`ReferenceDischargeContext`].
//! 2. The driver replays each instruction. An access involving only preserved references is replayed unchanged; every
//!    other application dispatches to its [`ReferenceDischargeableOperation`] rule.
//! 3. A rule reads or updates the allocation environment. A structured rule may first summarize an attached region, widen
//!    its boundary, and rebuild it in an isolated context before merging the resulting state into its caller.
//! 4. The transform reconstructs the program's public outputs, appends one hidden final-state output for each mutated
//!    external allocation, validates the complete boundary, and returns the appropriate result envelope.
//!
//! # Glossary
//!
//! - An **external source** is the capture or public input through which caller-owned reference state enters, named by
//!   [`ReferenceSource`].
//! - A **discharge target** is a stable source program location used to select an external reference or internal
//!   allocation for partial discharge, represented by [`ReferenceDischargeTarget`].
//! - An **allocation identity** is the temporary [`ReferenceDischargeAllocationId`] by which one running interpreter
//!   identifies a reference allocation. It does not identify a source program location and cannot cross between
//!   isolated environments.
//! - A **discharged reference** is represented by its current immutable state. A **preserved reference** remains
//!   represented by a reference value and its reference operations are replayed rather than rewritten.
//! - A **carrier** is a [`ReferenceDischargeValue`]: either an destination value or a temporary reference
//!   handle passed between operation rules.
//! - A summary allocation is **reached** when a region's closure accesses, returns, or otherwise rematerializes it. It is
//!   **accessed** only when the closure performs a semantic reference access on it. Reached allocations may need to cross a
//!   rebuilt boundary even when they are read nowhere inside the region.
//! - A widening's **threaded** allocations cross a structured boundary, its **entering** allocations require added inputs because
//!   no declared input already carries them, and its **published** allocations require added outputs because the region may
//!   mutate them and no declared output already publishes their state.

// TODO(eaplatanios): Review this module.

mod transform;

pub use transform::{
    ExternalReferenceBinding, PartialReferenceDischargeResult, RecursiveReferenceDischargeDriver,
    ReferenceAccumulationPolicy, ReferenceDischargeAllocationId, ReferenceDischargeBoundaryWidening,
    ReferenceDischargeContext, ReferenceDischargeDriver, ReferenceDischargePolicy, ReferenceDischargeReference,
    ReferenceDischargeRegionBoundary, ReferenceDischargeRegionResult, ReferenceDischargeRegionStateInsertion,
    ReferenceDischargeRegionSummary, ReferenceDischargeResult, ReferenceDischargeTarget, ReferenceDischargeValue,
    ReferenceDischargeableOperation, ReferenceDischargeableType, ReferenceSource,
    discharge_positional_region_operation, discharge_reference_free_operation,
};
