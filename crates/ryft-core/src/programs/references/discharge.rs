//! Reference discharge: rewriting mutable reference state into explicit immutable dataflow.
//!
//! A program containing reference operations is not ordinary functional Single Static Assignment (SSA) dataflow. A
//! read depends on the latest write to the same root even though that dependency is represented by a reference handle
//! rather than by an SSA operand carrying the current value. Many transforms and backends require the dependency to
//! be explicit. Reference discharge makes it explicit by replaying the program while replacing each selected root
//! with an immutable state value that is threaded from one access to the next.
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
//! Discharge removes the mutable root and exposes the same dependencies as immutable values:
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
//! replacement returns the previous state and produces a successor state, while every subsequent access consumes
//! that successor. The reference is an implementation detail of the source program and no reference survives in the
//! full result.
//!
//! An external reference follows the same rewrite but its state crosses the program boundary. Its reference-typed
//! input becomes an ordinary input carrying the entering referent. If the program mutates the root, its final state is
//! appended after the public outputs as a hidden output. [`ReferenceStateBinding`] records which capture or public
//! argument owns that state and which hidden output must be installed back into the caller's reference. Its discharged
//! input position follows from that logical source and the result's capture count. A read-only external root has no
//! hidden output.
//!
//! Discharge rewrites a program; it does not itself lock eager reference state or execute a backend. The stateful
//! compilation surface uses the result's binding metadata together with the runtime reference protocol after
//! compilation.
//!
//! # Full and Partial Discharge
//!
//! [`ReferenceDischargeResult`] is the full-discharge contract. Its payload is proven reference-free across the
//! complete attached-region closure, its public outputs form a prefix of its complete outputs, and the remaining
//! suffix contains exactly the final states of mutated external roots in canonical boundary order.
//!
//! [`PartialReferenceDischargeResult`] permits selected roots to become immutable state while unselected roots and
//! their operations remain in the payload. Callers select entry roots or allocation sites through
//! [`ReferenceDischargeSite`]. This is useful when normalizing a pipeline's internal state while deliberately
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
//! - [`ReferenceDischargeValue`] is the context-free carrier flowing between rules. It contains either an ordinary
//!   destination value or an opaque [`ReferenceDischargeReference`] handle. [`ReferenceDischargeTracer`] stamps that
//!   carrier with the active context so it can participate in normal Ryft interpretation.
//! - [`ReferenceDischargeContext`] owns the live root environment. Each root is either `Discharged`, with a current
//!   immutable state and mutation bit, or `Preserved`, with the exact destination reference value that survived.
//! - [`ReferenceDischargeableOperation`] is the rule implemented by each operation. Reference primitives rewrite
//!   their own accesses, structured operations own their boundary widening, and reference-free operations replay
//!   unchanged through the parent context.
//! - [`ReferenceDischargeDriver`] exposes the current source instruction and attached regions. It can replay a region
//!   against the live environment or rebuild one against an isolated environment and return a sealed
//!   [`ReferenceRegionDischargeFork`]. [`ReferenceRegionSummary`] supplies the transitive access facts a structured
//!   rule needs before choosing its state boundary.
//!
//! The driver provides shared mechanics but never chooses how an operation is rewritten. This keeps the system open:
//! a third-party primitive or structured operation participates by implementing its own rule, while a non-array
//! value family supplies its own policy without changing this interpreter.
//!
//! # Root Identities and Boundaries
//!
//! [`ReferenceDischargeSite`] is a source-program coordinate used before replay to select an external root or an
//! allocation. [`ReferenceRootHandle`] is different: it is a temporary identity minted inside one live discharge
//! environment. Handles from isolated region forks cannot address parent roots, and fork results carry sealed
//! programs and context-free summaries rather than child-context values.
//!
//! Structured rules use [`ReferenceRegionDischargeBoundary`] to describe their declared inputs plus the discharged
//! roots that must enter and leave a rebuilt region. Read-only roots are pruned where the operation's boundary permits
//! it; loop-shaped operations retain the symmetry their fixed-point contracts require. Every rebuilt region is
//! validated against the roots and mutations its summary predicted before the parent environment accepts its outputs.

use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt::{Debug, Display};
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::captures::CaptureConstant;
use crate::contexts::{Context, Domain, StagingContext, ValueResolution};
use crate::macros::check_count;
use crate::parameters::{Parameter, Parameterized, Placeholder};
use crate::programs::ProgramError;
use crate::programs::atoms::{Atom, AtomId};
use crate::programs::instructions::{Instruction, InstructionId};
use crate::programs::operations::Operation;
use crate::programs::programs::Program;
use crate::programs::provenance::{Provenance, ProvenanceScope};
use crate::programs::regions::{
    BindingRegionDriver, EmptyRegionDriver, RegionDriver, RegionId, RegionRef, RegionReplayMappings, ReplayRegionDriver,
};
use crate::programs::types::{Type, Typed};
use crate::programs::values::{Concretizable, Value};
use crate::tracing::TracingContext;

use super::semantics::{ReferenceAccessMode, ReferenceOutput};
use super::types::ReferenceType;

/// Shared payload and logical external-state metadata of a reference discharge result.
#[derive(Debug)]
struct ReferenceDischargeEnvelope<P> {
    /// Program payload whose public outputs form a prefix of its complete outputs.
    program: P,

    /// Number of leading program inputs originating in the source program's capture table.
    capture_count: usize,

    /// Number of public output leaves before hidden final-state outputs.
    public_output_count: usize,

    /// Discharged external reference binding recipes in canonical entry-boundary order.
    external_states: Vec<ReferenceStateBinding>,
}

impl<P: ReferenceDischargePayload> ReferenceDischargeEnvelope<P> {
    /// Creates an envelope after validating its discharged boundary layout.
    fn new(
        program: P,
        capture_count: usize,
        public_output_count: usize,
        external_states: Vec<ReferenceStateBinding>,
    ) -> Result<Self, ProgramError> {
        validate_discharged_boundary(
            capture_count,
            program.input_count(),
            program.output_count(),
            public_output_count,
            external_states.as_slice(),
        )?;
        Ok(Self { program, capture_count, public_output_count, external_states })
    }
}

/// Reference-free program payload and logical external-state metadata produced by reference discharge.
#[derive(Debug)]
pub struct ReferenceDischargeResult<P> {
    /// Shared checked result envelope.
    envelope: ReferenceDischargeEnvelope<P>,
}

/// Provider-owned proof boundary for payload families returned by full reference discharge.
///
/// Implementations report their own entry-boundary arities and must inspect every value and operation in the payload's
/// complete attached-region closure, returning success only when no reference-typed value or nonempty reference
/// semantics remains. These are provider contracts for downstream payload families. Rust's coherence rules prevent a
/// downstream crate from replacing the checked implementation for [`Program`].
pub trait ReferenceDischargePayload {
    /// Returns the number of inputs in this payload's entry boundary.
    fn input_count(&self) -> usize;

    /// Returns the number of outputs in this payload's entry boundary.
    fn output_count(&self) -> usize;

    /// Validates that this complete payload is reference-free.
    fn validate_reference_free(&self) -> Result<(), ProgramError>;
}

impl<P: ReferenceDischargePayload> ReferenceDischargeResult<P> {
    /// Creates a full discharge result after invoking the payload provider's reference-freedom proof.
    ///
    /// [`ReferenceDischargePayload::validate_reference_free`] proves the payload property, and this constructor then
    /// validates the discharged boundary layout. For [`Program`] payloads the implementation performs the same
    /// closure-wide scan as [`PartialReferenceDischargeResult::try_into_full`], so this entry point cannot bypass it.
    ///
    /// # Parameters
    ///
    ///   - `program`: Discharged program payload validated through its provider implementation.
    ///   - `capture_count`: Number of leading inputs originating in the source program's capture table.
    ///   - `public_output_count`: Number of public outputs preceding hidden final-state outputs.
    ///   - `external_states`: Logical external-state bindings in canonical entry-boundary order.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when the payload retains references or when the counts and bindings
    /// are not arithmetically consistent with one canonical discharged boundary: strict canonical source ordering,
    /// in-range flat positions, and exact hidden-suffix coverage. The constructor cannot prove *semantic* identity —
    /// that each named source is the state the payload actually threads at that position remains the provider's
    /// obligation, exactly as with any positional ABI.
    pub fn from_provider_payload(
        program: P,
        capture_count: usize,
        public_output_count: usize,
        external_states: Vec<ReferenceStateBinding>,
    ) -> Result<Self, ProgramError> {
        program.validate_reference_free()?;
        Ok(Self {
            envelope: ReferenceDischargeEnvelope::new(program, capture_count, public_output_count, external_states)?,
        })
    }
}

impl<P> ReferenceDischargeResult<P> {
    /// Returns the reference-free program payload.
    #[inline]
    pub const fn program(&self) -> &P {
        &self.envelope.program
    }

    /// Returns the number of leading inputs originating in the source program's capture table.
    #[inline]
    pub const fn capture_count(&self) -> usize {
        self.envelope.capture_count
    }

    /// Returns the number of public outputs at the front of the program payload's output boundary.
    #[inline]
    pub const fn public_output_count(&self) -> usize {
        self.envelope.public_output_count
    }

    /// Returns external reference binding recipes in canonical entry-boundary order.
    #[inline]
    pub fn external_states(&self) -> &[ReferenceStateBinding] {
        self.envelope.external_states.as_slice()
    }

    /// Consumes this result and returns its payload, capture count, public-output prefix, and external-state bindings.
    #[inline]
    pub fn into_parts(self) -> (P, usize, usize, Vec<ReferenceStateBinding>) {
        let ReferenceDischargeEnvelope { program, capture_count, public_output_count, external_states } = self.envelope;
        (program, capture_count, public_output_count, external_states)
    }
}

/// Program payload produced by *partial* reference discharge, in which only the caller-selected reference sites became
/// explicit immutable state and every unselected root survives as a well-typed reference value.
///
/// The discharged part of the boundary obeys exactly the invariants of [`ReferenceDischargeResult`]: discharged
/// external roots are reported as [`ReferenceStateBinding`]s in canonical entry-boundary order, and the mutated
/// subset of those bindings tiles the hidden output suffix that follows the public outputs. Discharged local
/// allocations leave no binding, because no caller owns their state. Preserved roots contribute neither bindings
/// nor hidden outputs; they simply remain reference-typed values inside the payload, and their accesses replay
/// verbatim.
///
/// There is deliberately no blanket conversion into [`ReferenceDischargeResult`]: "every site was selected" is a
/// statement about the request, not a proof about the produced payload, and a malformed provider could satisfy it
/// while still emitting references. [`try_into_full`](Self::try_into_full) therefore exists only for
/// [`Program`] payloads, where the reference-freedom proof can actually be carried out. Providers of other payload
/// families encode their equivalent proof through [`ReferenceDischargePayload`].
#[derive(Debug)]
pub struct PartialReferenceDischargeResult<P> {
    /// Shared checked result envelope.
    envelope: ReferenceDischargeEnvelope<P>,
}

impl<P: ReferenceDischargePayload> PartialReferenceDischargeResult<P> {
    /// Creates a checked partial reference discharge result.
    ///
    /// The external-state bindings describe the *discharged* roots only and must satisfy the same canonical boundary
    /// invariants as [`ReferenceDischargeResult::from_provider_payload`]: they must name valid discharged inputs in
    /// canonical source order, and their final-state output indices, omitting read-only bindings, must exactly cover
    /// the hidden output suffix in binding order.
    ///
    /// # Parameters
    ///
    ///   - `program`: Mixed discharged program payload.
    ///   - `capture_count`: Number of leading inputs originating in the source program's capture table.
    ///   - `public_output_count`: Number of public outputs preceding hidden final-state outputs.
    ///   - `external_states`: Logical external-state bindings for the discharged roots, in canonical entry-boundary
    ///     order.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when the counts and bindings do not describe one canonical
    /// discharged boundary.
    pub fn new(
        program: P,
        capture_count: usize,
        public_output_count: usize,
        external_states: Vec<ReferenceStateBinding>,
    ) -> Result<Self, ProgramError> {
        Ok(Self {
            envelope: ReferenceDischargeEnvelope::new(program, capture_count, public_output_count, external_states)?,
        })
    }

    /// Returns the mixed program payload.
    #[inline]
    pub const fn program(&self) -> &P {
        &self.envelope.program
    }

    /// Returns the number of leading inputs originating in the source program's capture table.
    #[inline]
    pub const fn capture_count(&self) -> usize {
        self.envelope.capture_count
    }

    /// Returns the number of public outputs at the front of the program payload's output boundary.
    #[inline]
    pub const fn public_output_count(&self) -> usize {
        self.envelope.public_output_count
    }

    /// Returns the binding recipes of the discharged external reference roots, in canonical entry-boundary order.
    /// Preserved roots are deliberately absent: they were never turned into state and so have nothing to bind.
    #[inline]
    pub fn external_states(&self) -> &[ReferenceStateBinding] {
        self.envelope.external_states.as_slice()
    }

    /// Consumes this result and returns its payload, capture count, public-output prefix, and external-state bindings.
    #[inline]
    pub fn into_parts(self) -> (P, usize, usize, Vec<ReferenceStateBinding>) {
        let ReferenceDischargeEnvelope { program, capture_count, public_output_count, external_states } = self.envelope;
        (program, capture_count, public_output_count, external_states)
    }
}

impl<V, O, Input, Output> PartialReferenceDischargeResult<Program<V, O, Input, Output>>
where
    V: Value,
    O: Operation<Type = V::Type>,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
{
    /// Proves that this partial result is in fact reference-free and converts it into a
    /// [`ReferenceDischargeResult`].
    ///
    /// The proof inspects the complete attached region closure of the payload, dormant transformation rule regions
    /// included, and requires that no atom carries a reference type and that no operation declares nonempty
    /// [`ReferenceOperationSemantics`](crate::ReferenceOperationSemantics). Because every boundary position and every
    /// stored constant is itself an atom, the first check covers input types, output types, and constants alike.
    ///
    /// The proof is deliberately reference-specific rather than a general state-purification check. Reference
    /// discharge normalizes references and nothing else, so an unrelated ordered-state operation contributed by a
    /// third-party backend passes through untouched, and the consumers that care about ordered state keep their own
    /// gates.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when the payload still contains a reference-typed atom or an
    /// operation with nonempty reference semantics.
    pub fn try_into_full(self) -> Result<ReferenceDischargeResult<Program<V, O, Input, Output>>, ProgramError> {
        self.envelope.program.validate_reference_free()?;
        Ok(ReferenceDischargeResult { envelope: self.envelope })
    }
}

impl<V, O, Input, Output> ReferenceDischargePayload for Program<V, O, Input, Output>
where
    V: Value,
    O: Operation<Type = V::Type>,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
{
    #[inline]
    fn input_count(&self) -> usize {
        self.entry_region_ref().input_ids().len()
    }

    #[inline]
    fn output_count(&self) -> usize {
        self.entry_region_ref().output_ids().len()
    }

    fn validate_reference_free(&self) -> Result<(), ProgramError> {
        let entry = self.entry_region_ref();
        if entry.contains_atom_type_in_closure(Type::is_reference) {
            return Err(ProgramError::MalformedProgram(
                "reference discharge payload still contains a reference-typed value and cannot form a full discharge"
                    .to_string(),
            ));
        }
        // The closure traversal visits regions in an unspecified order, so the reported occurrence is the smallest
        // coordinate rather than the first one encountered, keeping the diagnostic reproducible.
        if let Some((instruction_id, instruction)) = entry
            .instructions_in_closure()
            .filter(|(_, instruction)| !instruction.operation().reference_semantics().is_empty())
            .min_by_key(|(instruction_id, _)| *instruction_id)
        {
            return Err(ProgramError::MalformedProgram(format!(
                "reference discharge payload retains reference operation `{}` at `{instruction_id}` and cannot form \
                 a full discharge",
                instruction.operation().name(),
            )));
        }
        Ok(())
    }
}

/// Validates that a discharged program boundary and its external-state bindings describe one canonical discharged
/// shape, shared by the full and partial result envelopes.
///
/// # Parameters
///
///   - `capture_count`: Number of leading inputs originating in the source program's capture table.
///   - `total_input_count`: Number of inputs in the discharged payload.
///   - `total_output_count`: Number of outputs in the discharged payload.
///   - `public_output_count`: Number of public outputs preceding hidden final-state outputs.
///   - `external_states`: External-state bindings in canonical entry-boundary order.
fn validate_discharged_boundary(
    capture_count: usize,
    total_input_count: usize,
    total_output_count: usize,
    public_output_count: usize,
    external_states: &[ReferenceStateBinding],
) -> Result<(), ProgramError> {
    if capture_count > total_input_count {
        return Err(ProgramError::MalformedProgram(format!(
            "reference discharge reports {capture_count} captures but discharged input count is {total_input_count}",
        )));
    }
    if public_output_count > total_output_count {
        return Err(ProgramError::MalformedProgram(format!(
            "reference discharge reports {public_output_count} public outputs but discharged output count is \
             {total_output_count}",
        )));
    }
    for state in external_states {
        let input_index = state.source().flat_input_index(capture_count)?;
        if input_index >= total_input_count {
            return Err(ProgramError::MalformedProgram(format!(
                "reference discharge state for `{}` names input {input_index} but discharged input count is \
                 {total_input_count}",
                state.source(),
            )));
        }
    }
    for adjacent_states in external_states.windows(2) {
        let previous_source = adjacent_states[0].source();
        let source = adjacent_states[1].source();
        if source <= previous_source {
            return Err(ProgramError::MalformedProgram(format!(
                "reference discharge state source `{source}` does not follow source `{previous_source}` in canonical \
                 boundary order",
            )));
        }
    }
    let mut expected_output_index = public_output_count;
    for state in external_states.iter().filter(|state| state.is_mutated()) {
        let output_index = state.final_state_output_index().unwrap();
        if output_index != expected_output_index {
            return Err(ProgramError::MalformedProgram(format!(
                "reference discharge final-state output {output_index} for `{}` does not match expected hidden output \
                 {expected_output_index}",
                state.source(),
            )));
        }
        expected_output_index = expected_output_index.checked_add(1).ok_or_else(|| {
            ProgramError::MalformedProgram("reference discharge hidden output index overflows `usize`".to_string())
        })?;
    }
    if expected_output_index != total_output_count {
        return Err(ProgramError::MalformedProgram(format!(
            "reference discharge final states end at output {expected_output_index} but discharged output count is \
             {total_output_count}",
        )));
    }
    Ok(())
}

/// Program-level capability for normalizing references into explicit immutable state.
///
/// An implementation names its universe's [`ReferenceDischargePolicy`] and otherwise only forwards to the interpreter
/// entry point [`Program::discharge_references_with_policy`]. Generic transforms reach discharge through
/// [`discharge_local_references`](Self::discharge_local_references) and therefore neither name a policy nor inspect
/// family-specific alias metadata.
pub trait ReferenceDischarge: Sized {
    /// Reference-free program payload produced by this implementation.
    type DischargedProgram: ReferenceDischargePayload;

    /// Discharges every reference and returns the reference-free program plus its logical external-state bindings.
    ///
    /// # Parameters
    ///
    ///   - `capture_count`: Number of leading flat inputs that originated in the source program's capture table.
    fn discharge_references(
        self,
        capture_count: usize,
    ) -> Result<ReferenceDischargeResult<Self::DischargedProgram>, ProgramError>;

    /// Discharges local references for `transform`, rejecting every caller-owned external root.
    ///
    /// The full-discharge implementation must prove that the result contains neither reference types nor unresolved
    /// ordered reference state. The checked result envelope ensures that an external-state-free result has no hidden
    /// output suffix, so this default returns the same program family with an unchanged public boundary.
    ///
    /// # Parameters
    ///
    ///   - `capture_count`: Number of leading flat inputs that originated in the source program's capture table.
    ///   - `transform`: Name used in diagnostics when caller-owned state prevents the transform.
    fn discharge_local_references(
        self,
        capture_count: usize,
        transform: &'static str,
    ) -> Result<Self::DischargedProgram, ProgramError> {
        let discharged = self.discharge_references(capture_count)?;
        if let Some(state) = discharged.external_states().first() {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "{transform} supports only local references, but the program uses external `{}`",
                    state.source(),
                ),
            });
        }
        let (program, _, _, _) = discharged.into_parts();
        Ok(program)
    }
}

/// Invocation source of one external reference root.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReferenceSource {
    /// Capture lifted into the entry boundary before input arguments.
    Capture {
        /// Zero-based capture position in the lifted capture prefix.
        index: usize,
    },

    /// Reference input argument after the lifted capture prefix.
    Input {
        /// Zero-based input position, excluding lifted captures.
        index: usize,
    },
}

impl ReferenceSource {
    /// Returns the entry-boundary source of one flat input position, splitting the boundary at the lifted capture
    /// prefix.
    ///
    /// # Parameters
    ///
    ///   - `input_index`: Flat entry input position.
    ///   - `capture_count`: Number of leading flat inputs that originated in the program's capture table.
    #[inline]
    pub const fn from_input_index(input_index: usize, capture_count: usize) -> Self {
        if input_index < capture_count {
            Self::Capture { index: input_index }
        } else {
            Self::Input { index: input_index - capture_count }
        }
    }

    /// Resolves this logical source to its flat discharged entry-boundary position.
    ///
    /// # Parameters
    ///
    ///   - `capture_count`: Number of leading flat inputs originating in the source program's capture table.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when a capture lies outside the capture prefix or when adding the
    /// prefix to an input position overflows `usize`.
    pub fn flat_input_index(self, capture_count: usize) -> Result<usize, ProgramError> {
        match self {
            Self::Capture { index } if index < capture_count => Ok(index),
            Self::Capture { index } => Err(ProgramError::MalformedProgram(format!(
                "reference source capture {index} lies outside the capture prefix of length {capture_count}",
            ))),
            Self::Input { index } => capture_count.checked_add(index).ok_or_else(|| {
                ProgramError::MalformedProgram(format!(
                    "reference source input {index} overflows the flat boundary after {capture_count} captures",
                ))
            }),
        }
    }
}

impl Display for ReferenceSource {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Capture { index } => write!(formatter, "capture {index}"),
            Self::Input { index } => write!(formatter, "input {index}"),
        }
    }
}

/// Logical binding recipe for one external reference root in a discharged program.
///
/// The [`serde::Serialize`] implementation exposes the canonical in-memory shape for diagnostics and snapshots and is
/// deliberately distinct from the stable XLA persistence schema, which keeps its own versioned representation
/// (including a redundant validated flat-input coordinate) independent of this type's evolution.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub struct ReferenceStateBinding {
    /// Capture or input argument supplying the eager reference handle.
    source: ReferenceSource,

    /// Hidden output position containing final state, or [`None`] for a read-only root.
    final_state_output_index: Option<usize>,
}

impl ReferenceStateBinding {
    /// Creates one logical external-state binding.
    ///
    /// # Parameters
    ///
    ///   - `source`: Capture or input supplying the eager reference handle.
    ///   - `final_state_output_index`: Hidden output containing the final state, or [`None`] for a read-only root.
    pub const fn new(source: ReferenceSource, final_state_output_index: Option<usize>) -> Self {
        Self { source, final_state_output_index }
    }

    /// Returns the capture or input argument supplying the eager reference handle.
    pub const fn source(&self) -> ReferenceSource {
        self.source
    }

    /// Returns whether this external state is mutated.
    pub const fn is_mutated(&self) -> bool {
        self.final_state_output_index.is_some()
    }

    /// Returns the hidden final-state output position, if any.
    pub const fn final_state_output_index(&self) -> Option<usize> {
        self.final_state_output_index
    }
}

/// One caller-selectable reference site for partial reference discharge.
///
/// Selection needs an identity that exists in the *source* program, before any replay begins, which is why it is a
/// vocabulary of its own rather than a reuse of the environment's [`ReferenceRootHandle`]s. In particular, a nested
/// region's formal reference input is invocation-parameterized — the region may be invoked from several call sites —
/// so it names no single caller-owned reference and is deliberately not selectable. Sites resolve internally to
/// roots once discharge starts.
///
/// Sites are arena-relative in exactly the sense that every other reference artifact is: their coordinates are
/// meaningful only against the program they were enumerated from. [`Program::validate_reference_discharge_sites`]
/// rejects every kind mismatch, and the arena-relativity contract carries the rest, because a coordinate taken from
/// a different arena that happens to name a valid allocation here is indistinguishable in principle.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum ReferenceDischargeSite {
    /// Entry-boundary root supplied by the caller as a lifted capture or a public reference argument.
    External(ReferenceSource),

    /// Interior allocation site, identified by the allocating instruction and the output position that defines the
    /// fresh root.
    Allocation {
        /// Allocating instruction.
        instruction: InstructionId,

        /// Output position defining the fresh root.
        output_index: usize,
    },
}

// Sites exist to be named in diagnostics, so the rendering backticks the arena coordinate it embeds. That keeps every
// message that interpolates a whole site consistent with the reference-site diagnostics, which backtick coordinates.
impl Display for ReferenceDischargeSite {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::External(source) => write!(formatter, "external {source}"),
            Self::Allocation { instruction, output_index } => {
                write!(formatter, "allocation at `{instruction}` output {output_index}")
            }
        }
    }
}

/// Reference sites one discharge normalizes into immutable state, with every unselected root preserved.
///
/// Selecting everything is deliberately a state of its own rather than a set listing every site. A program's sites are
/// enumerated from its own arena while a selection is caller-supplied, so full discharge — which is exactly the
/// everything-selected case of the one rewrite — must be expressible without naming anything, and an allocation that
/// no site *can* name, such as one bound directly rather than replayed, must still be discharged by it.
#[derive(Clone, Debug)]
struct ReferenceDischargeSelection {
    /// Selected sites, or [`None`] when every site is selected.
    sites: Option<Rc<BTreeSet<ReferenceDischargeSite>>>,
}

impl ReferenceDischargeSelection {
    /// Returns the selection full discharge runs under, which selects every site.
    #[inline]
    const fn everything() -> Self {
        Self { sites: None }
    }

    /// Returns the selection naming exactly `sites`, which preserves every root they do not name.
    #[inline]
    fn from_sites(sites: &[ReferenceDischargeSite]) -> Self {
        Self { sites: Some(Rc::new(sites.iter().copied().collect())) }
    }

    /// Returns whether `site` is selected for discharge.
    #[inline]
    fn selects(&self, site: ReferenceDischargeSite) -> bool {
        self.sites.as_ref().is_none_or(|sites| sites.contains(&site))
    }
}

impl<V, O, Input, Output> Program<V, O, Input, Output>
where
    V: Value,
    O: Operation<Type = V::Type>,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
{
    /// Returns every [`ReferenceDischargeSite`] this program exposes to partial reference discharge, in canonical
    /// order: the entry-boundary externals in boundary order, followed by the interior allocations ordered by their
    /// arena coordinates.
    ///
    /// This is a deliberately lightweight query. It reads only the entry boundary types and the generic
    /// [`Operation::reference_semantics`] hook over the attached region closure, so it does not run the discharge
    /// rewrite or construct its environments, and callers can enumerate selectable sites without paying for either.
    /// Allocations inside nested regions are included, because an allocation is a concrete program-local root
    /// wherever it occurs.
    ///
    /// One class of enumerated site is inert: an allocation inside a closure that no operation ever replays, such as
    /// the dormant derivative rule region of a `custom_jvp`. Discharge rejects such a program outright, whichever way
    /// the site is selected, because how a reference boundary widens there has no defined meaning. The enumeration
    /// reports the site anyway rather than second-guessing the region roles, so that it stays a structural query.
    ///
    /// # Parameters
    ///
    ///   - `capture_count`: Number of leading flat inputs that originated in the program's capture table, used to
    ///     split the entry boundary into [`ReferenceSource::Capture`] and [`ReferenceSource::Input`]
    ///     positions.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when `capture_count` exceeds the program's input count.
    pub fn reference_discharge_sites(&self, capture_count: usize) -> Result<Vec<ReferenceDischargeSite>, ProgramError> {
        let entry = self.entry_region_ref();
        let input_ids = entry.input_ids();
        if capture_count > input_ids.len() {
            return Err(ProgramError::MalformedProgram(format!(
                "reference discharge site enumeration requests {capture_count} captures but the program has {} inputs",
                input_ids.len(),
            )));
        }
        let mut sites = input_ids
            .iter()
            .enumerate()
            .filter(|(_, input)| entry.atoms()[input.index()].r#type().is_reference())
            .map(|(input_index, _)| {
                ReferenceDischargeSite::External(ReferenceSource::from_input_index(input_index, capture_count))
            })
            .collect::<Vec<_>>();
        let mut allocations = entry
            .instructions_in_closure()
            .flat_map(|(instruction_id, instruction)| {
                instruction
                    .operation()
                    .reference_semantics()
                    .root_output_indices()
                    .collect::<Vec<_>>()
                    .into_iter()
                    .map(move |output_index| ReferenceDischargeSite::Allocation {
                        instruction: instruction_id,
                        output_index,
                    })
            })
            .collect::<Vec<_>>();

        // Closure traversal visits regions in an unspecified order, so allocation coordinates are sorted to make the
        // enumeration reproducible for callers that persist or compare selections.
        allocations.sort_unstable();
        sites.append(&mut allocations);
        Ok(sites)
    }

    /// Validates a caller-provided partial reference discharge selection against this program.
    ///
    /// Every named site must exist in this program, must name a reference-typed entry position or a genuine
    /// reference-allocating output, and must appear at most once. Duplication is checked across the complete selection
    /// first, because a repeated site is ambiguous whatever it names.
    ///
    /// # Parameters
    ///
    ///   - `capture_count`: Number of leading flat inputs that originated in the program's capture table.
    ///   - `sites`: Sites selected for discharge, in caller-chosen order.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] naming the offending site when a site is duplicated, names an
    /// out-of-range or non-reference entry position, names an instruction that this program does not contain, names
    /// an operation that allocates no reference root, or names an output position of an allocating operation that is
    /// not itself an allocation.
    pub fn validate_reference_discharge_sites(
        &self,
        capture_count: usize,
        sites: &[ReferenceDischargeSite],
    ) -> Result<(), ProgramError> {
        let entry = self.entry_region_ref();
        let input_ids = entry.input_ids();
        if capture_count > input_ids.len() {
            return Err(ProgramError::MalformedProgram(format!(
                "reference discharge selection requests {capture_count} captures but the program has {} inputs",
                input_ids.len(),
            )));
        }
        let mut seen = HashSet::with_capacity(sites.len());
        for site in sites {
            if !seen.insert(*site) {
                return Err(ProgramError::MalformedProgram(format!(
                    "reference discharge selection names {site} more than once",
                )));
            }
        }

        // Only the named instructions are resolved, so validating a small selection does not pay for the reference
        // semantics of every instruction in the closure.
        let instructions = entry.instructions_in_closure().collect::<HashMap<_, _>>();
        for site in sites {
            let invalid_site = || {
                ProgramError::MalformedProgram(format!(
                    "reference discharge selection names {site}, which is not a selectable site in this program",
                ))
            };
            match site {
                ReferenceDischargeSite::External(source) => {
                    let input_index = source.flat_input_index(capture_count).map_err(|_| invalid_site())?;
                    let input = input_ids.get(input_index).ok_or_else(invalid_site)?;
                    if !entry.atoms()[input.index()].r#type().is_reference() {
                        return Err(invalid_site());
                    }
                }
                ReferenceDischargeSite::Allocation { instruction, output_index } => {
                    let instruction = instructions.get(instruction).ok_or_else(invalid_site)?;
                    let operation = instruction.operation();
                    let output_indices = operation.reference_semantics().root_output_indices().collect::<Vec<_>>();
                    if !output_indices.contains(output_index) {
                        return Err(invalid_site());
                    }
                }
            }
        }
        Ok(())
    }
}

impl<V: Value, O: Operation<Type = V::Type>> Program<V, O, Vec<V>, Vec<V>> {
    /// Discharges every reference this program touches by interpreting it in a [`ReferenceDischargeContext`] over a
    /// fresh trace of its own universe, returning the reference-free program together with its logical external-state
    /// bindings.
    ///
    /// This is the entry point production discharge runs through, and it owns the whole reference language: the
    /// primitives rewrite themselves, and a region-carrying operation either discharges through its own
    /// [`ReferenceDischargeableOperation`] rule — widening its boundaries with the state its regions touch — or, when
    /// nothing in its attached closure touches a reference, replays those regions verbatim. What survives as a
    /// rejection is a region closure that does touch a reference behind an operation with no rule of its own, and a
    /// reference that reaches a region neither through a boundary nor through a capture scope. A universe whose
    /// programs name their caller's references through capture constants uses
    /// [the capture-aware entry point](Self::discharge_references_with_lifted_captures_and_policy) instead, which is
    /// the same rewrite under a populated capture scope.
    ///
    /// Each source input keeps its position. A reference-typed input becomes an ordinary input carrying the referent's
    /// lifted type, which is the entering immutable state of the root that input denotes; every other input is
    /// replayed unchanged. The public outputs are exactly the source outputs, in order, and the final state of each
    /// *mutated* external root is appended after them as a hidden output in entry-boundary order. A root that the
    /// program only reads contributes no hidden output, so a read-only program keeps its original boundary exactly.
    ///
    /// The replay runs through [`ReferenceDischargeDriver::discharge_region`] rather than through
    /// [`interpret_in_context`](Self::interpret_in_context), because that is the path that threads each instruction's
    /// source coordinate into the rules, which is what makes an entry-region allocation identifiable.
    ///
    /// The rewritten payload is proven reference-free rather than assumed to be: the replay assembles a
    /// [`PartialReferenceDischargeResult`] and converts it through
    /// [`try_into_full`](PartialReferenceDischargeResult::try_into_full), so a rule that returned a reference-touching
    /// operation is reported here instead of surviving into a result whose contract says it cannot exist.
    ///
    /// # Parameters
    ///
    ///   - `capture_count`: Number of leading flat inputs that originated in the source program's capture table, used
    ///     to split the entry boundary into [`ReferenceSource::Capture`] and [`ReferenceSource::Input`]
    ///     positions.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when `capture_count` exceeds the program's input count, when an
    /// output still denotes a reference, when the program consumes an external root, whose state belongs to the
    /// caller, or when the rewritten payload fails the reference-freedom proof. Rule-level failures, including a
    /// use-after-consume and an access to an unbound root, propagate from the replay itself.
    pub fn discharge_references_with_policy<P>(
        self,
        capture_count: usize,
    ) -> Result<ReferenceDischargeResult<Self>, ProgramError>
    where
        P: ReferenceDischargePolicy<TracingContext<V, O>>,
        O: ReferenceDischargeableOperation<TracingContext<V, O>, P>,
    {
        self.discharge_references_with_capture_seam::<P>(
            capture_count,
            |_| None,
            ReferenceDischargeSelection::everything(),
        )?
        .try_into_full()
    }

    /// Discharges the references the caller *selected* and preserves every other one, returning the mixed program
    /// together with the logical external-state bindings of the roots that became state.
    ///
    /// This is the same rewrite [`discharge_references_with_policy`](Self::discharge_references_with_policy) performs;
    /// full discharge is exactly its everything-selected case, and the two share one body. A selected root threads as
    /// immutable state in every respect described there. An unselected root instead *survives*: it keeps its
    /// reference-typed boundary position or its allocating instruction, every access to it replays verbatim as the
    /// reference operation the source performed, and a view derived from it replays its view operation too, so the
    /// rewritten program still denotes the same coordinates. Preserved roots contribute no state input, no hidden
    /// final-state output, and no [`ReferenceStateBinding`]: the payload's own boundary is where the caller sees them.
    ///
    /// This is what a kernel pipeline needs — normalize the pipeline's own state into explicit carries while the
    /// references a kernel body addresses stay references — and it is the reason the result envelope is
    /// [`PartialReferenceDischargeResult`], which proves nothing about reference freedom. A caller that expects the
    /// selection to have covered everything asks for the proof explicitly through
    /// [`try_into_full`](PartialReferenceDischargeResult::try_into_full).
    ///
    /// A preserved root crosses a structured operation's region boundary the same way it crosses anything else: as the
    /// reference it already is, at its own declared operand position, exactly as the source passed it. It occupies no
    /// state carry, publishes no successor, and widens nothing, so a condition, loop, scan, or call can thread
    /// discharged state and surviving references side by side. What a preserved root cannot become is *added* state
    /// that a rule synthesizes onto a rebuilt boundary, which is reported by name.
    ///
    /// Where a structured operation *declares* a reference-typed output — a loop carry, say — the rewritten operation
    /// still produces one, and it is deliberately left unused: the caller keeps the handle it already holds, because
    /// both denote the same root and one destination value per root is enough. A later full discharge of the same
    /// payload collapses that position into an ordinary state carry.
    ///
    /// A *capture-lifted* program has no partial form: this entry point recognizes no capture constant, so a
    /// reference-typed one is rejected where it is lifted, and
    /// [the capture-aware entry point](Self::discharge_references_with_lifted_captures_and_policy) remains
    /// full-discharge-only.
    ///
    /// # Parameters
    ///
    ///   - `capture_count`: Number of leading flat inputs that originated in the source program's capture table, used
    ///     to split the entry boundary into [`ReferenceSource::Capture`] and [`ReferenceSource::Input`]
    ///     positions.
    ///   - `sites`: Reference sites to discharge, enumerated from this same program through
    ///     [`reference_discharge_sites`](Self::reference_discharge_sites). Every other root is preserved.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when `sites` does not validate against this program (see
    /// [`validate_reference_discharge_sites`](Self::validate_reference_discharge_sites)), when a rule synthesizes a
    /// preserved root onto a rebuilt region's added state positions, and otherwise for every reason
    /// [`discharge_references_with_policy`](Self::discharge_references_with_policy) documents — with one deliberate
    /// exception. Consuming a *discharged* external root is still rejected, because a
    /// [`ReferenceStateBinding`] cannot express a caller-owned reference that no longer denotes live state; consuming
    /// a *preserved* one is accepted, because the payload retains the consuming operation and the caller passes its
    /// reference handle to that operation directly.
    pub fn partially_discharge_references_with_policy<P>(
        self,
        capture_count: usize,
        sites: &[ReferenceDischargeSite],
    ) -> Result<PartialReferenceDischargeResult<Self>, ProgramError>
    where
        P: ReferenceDischargePolicy<TracingContext<V, O>>,
        O: ReferenceDischargeableOperation<TracingContext<V, O>, P>,
    {
        self.validate_reference_discharge_sites(capture_count, sites)?;
        self.discharge_references_with_capture_seam::<P>(
            capture_count,
            |_| None,
            ReferenceDischargeSelection::from_sites(sites),
        )
    }

    /// Discharges every reference a *capture-lifted* program touches, resolving the capture-scoped reference
    /// constants its attached regions name their caller's references through.
    ///
    /// A capture-lifted program is one whose captures have been turned into a leading input prefix by
    /// [`ClosedProgram::to_program_with_lifted_captures`](crate::ClosedProgram::to_program_with_lifted_captures).
    /// Lifting rewrites the entry boundary, but an attached region keeps naming the same captures through
    /// [`CaptureReference`](crate::CaptureReference) constants, and those constants denote the very roots the lifted
    /// prefix binds. This entry point therefore differs from
    /// [`discharge_references_with_policy`](Self::discharge_references_with_policy) in exactly one respect: it seeds
    /// the entry capture scope from that prefix, so a reference-typed capture constant resolves to the
    /// root its position already binds instead of being rejected as belonging to no root.
    ///
    /// Everything else — the boundary rewrite, the hidden final-state outputs, and the reference-freedom proof — is
    /// identical, and a program with no capture-scoped reference constant discharges to exactly the same result
    /// through either entry point.
    ///
    /// # Parameters
    ///
    ///   - `capture_count`: Length of the lifted capture prefix, which is both the split between
    ///     [`ReferenceSource::Capture`] and [`ReferenceSource::Input`] positions and the capture scope's own
    ///     length.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`discharge_references_with_policy`](Self::discharge_references_with_policy), and
    /// additionally reports a capture constant whose declared reference type disagrees with the root its position
    /// binds.
    pub fn discharge_references_with_lifted_captures_and_policy<P>(
        self,
        capture_count: usize,
    ) -> Result<ReferenceDischargeResult<Self>, ProgramError>
    where
        V: CaptureConstant,
        P: ReferenceDischargePolicy<TracingContext<V, O>>,
        O: ReferenceDischargeableOperation<TracingContext<V, O>, P>,
    {
        self.discharge_references_with_capture_seam::<P>(
            capture_count,
            CaptureConstant::capture_index,
            ReferenceDischargeSelection::everything(),
        )?
        .try_into_full()
    }

    /// Discharges the selected references of this program, recognizing capture constants through `capture_index`.
    ///
    /// This is the shared body of the program-level entry points, and the partial rewrite is the general one: full
    /// discharge is the everything-selected case, which is why the body always assembles the partial envelope and
    /// leaves the reference-freedom proof to the caller that promised it.
    ///
    /// The capture seam is a parameter rather than a bound because the interpreter deliberately serves constant
    /// families that are not capture-bearing at all, and it is a function pointer rather than a closure because the
    /// only two seams that ever reach it are "nothing is a capture" and [`CaptureConstant::capture_index`].
    ///
    /// # Parameters
    ///
    ///   - `capture_count`: Number of leading flat inputs that originated in the source program's capture table.
    ///   - `capture_index`: Seam reporting the capture position a stored constant names.
    ///   - `selection`: Reference sites to discharge; every root the selection omits is preserved.
    ///
    /// # Errors
    ///
    /// Returns the errors the public entry points document, which is every error the replay can raise.
    fn discharge_references_with_capture_seam<P>(
        self,
        capture_count: usize,
        capture_index: fn(&V) -> Option<usize>,
        selection: ReferenceDischargeSelection,
    ) -> Result<PartialReferenceDischargeResult<Self>, ProgramError>
    where
        P: ReferenceDischargePolicy<TracingContext<V, O>>,
        O: ReferenceDischargeableOperation<TracingContext<V, O>, P>,
    {
        let input_types = self.input_types();
        let input_count = input_types.len();
        if capture_count > input_count {
            return Err(ProgramError::MalformedProgram(format!(
                "reference discharge requests {capture_count} captures but the program has {input_count} inputs",
            )));
        }
        let public_output_count = self.output_count();

        // A program that touches no reference anywhere is already its own discharge, so it is returned untouched
        // rather than replayed into a fresh trace. This is not only cheaper on the two transform adapters that
        // discharge unconditionally: re-tracing would also renumber its atoms, drop its dead constants, and abandon
        // the region transform cache its regions carry, all for a rewrite that has nothing to rewrite.
        let entry = self.entry_region_ref();
        if !region_closure_touches_references(entry) {
            return PartialReferenceDischargeResult::new(self, capture_count, public_output_count, Vec::new());
        }

        // The block scopes the destination context, the discharge context, and every carrier, because recovering the
        // traced program below requires unique ownership of the shared builder and therefore that every other handle
        // to it has been released.
        let (builder, output_ids, external_states) = {
            let destination = TracingContext::<V, O>::new();
            let builder = destination.builder().clone();
            let context =
                ReferenceDischargeContext::<TracingContext<V, O>, P>::new_selecting(destination.clone(), selection);
            let mut inputs = Vec::with_capacity(input_count);
            let mut discharged_roots = Vec::new();
            let mut capture_roots = vec![None; capture_count];
            for (input_index, input_type) in input_types.into_iter().enumerate() {
                let Some(reference_type) = P::project_reference_type(&input_type) else {
                    inputs.push(ReferenceDischargeValue::Ordinary(destination.input(input_type)));
                    continue;
                };
                let source = ReferenceSource::from_input_index(input_index, capture_count);
                let selected = context.selects_external(source);
                let carrier = if selected {
                    let state = destination.input(P::lift_referent_type(reference_type.referent().clone()));
                    context.allocate_discharged(reference_type, state)?
                } else {
                    // An unselected external root keeps its reference-typed boundary position exactly as the source
                    // declared it, so the caller still supplies a holder and every access to it replays verbatim.
                    context.bind_preserved(reference_type, destination.input(input_type))?
                };
                let root = carrier.expect_reference("an entry-boundary reference root")?.root();
                if selected {
                    discharged_roots.push((source, root));
                }
                if input_index < capture_count {
                    capture_roots[input_index] = Some(root);
                }
                inputs.push(carrier);
            }

            // The capture scope can only be installed once the prefix has minted its roots, and it is what lets a
            // nested region resolve the caller references it names through capture constants rather than through its
            // own boundary.
            let context = context.with_captures(ReferenceCaptureScope::new(capture_index, capture_roots));

            let regions = [self];
            let driver = RecursiveReferenceDischargeDriver::new(&regions, None);
            let outputs = driver.discharge_region(&context, 0, inputs)?;
            let mut output_ids = outputs
                .iter()
                .enumerate()
                .map(|(output_index, output)| match output {
                    ReferenceDischargeValue::Ordinary(value) => value.atom_id(),

                    // A preserved root survives in the rewritten program, so returning one returns its destination
                    // reference value. A discharged root has no such value, because it became state. Returning a root
                    // is a use of it like any other, so its liveness is resolved against the environment rather than
                    // taken from the handle, which is what reports a root the program already consumed.
                    ReferenceDischargeValue::Reference(reference) => {
                        context.validate_live_root(reference.root())?;
                        reference
                            .preserved()
                            .ok_or_else(|| {
                                ProgramError::MalformedProgram(format!(
                                    "reference discharge expected an ordinary value for output {output_index} but \
                                     received {reference}",
                                ))
                            })?
                            .atom_id()
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;

            // A mutated external root publishes its final state as a hidden output; a read-only one publishes nothing,
            // which is what keeps a read-only program's boundary identical to its source boundary. A preserved
            // external root binds nothing at all: it never became state, so there is no state for a caller to supply
            // or to write back.
            let mut external_states = Vec::with_capacity(discharged_roots.len());
            for (source, root) in discharged_roots {
                if context.validate_live_root(root).is_err() {
                    return Err(ProgramError::MalformedProgram(format!(
                        "reference discharge consumed external {source}, whose holder belongs to the caller",
                    )));
                }
                let final_state_output_index = if context.is_mutated(root)? {
                    output_ids.push(context.discharged_state(root)?.atom_id()?);
                    Some(output_ids.len() - 1)
                } else {
                    None
                };
                external_states.push(ReferenceStateBinding::new(source, final_state_output_index));
            }
            (builder, output_ids, external_states)
        };

        let output_count = output_ids.len();
        let builder = Rc::try_unwrap(builder).map_err(|_| ProgramError::EscapedProgramBuilder)?.into_inner();
        let program = builder.build(output_ids, vec![Placeholder; input_count], vec![Placeholder; output_count])?;
        PartialReferenceDischargeResult::new(program, capture_count, public_output_count, external_states)
    }
}

/// Policy naming the types and alias mechanics that one reference universe threads through reference discharge.
///
/// Discharge rewrites a program that mutates references into one that threads immutable state. Everything in that
/// rewrite that varies from one reference universe to another lives here, so the primitive rules, the context, the
/// driver, and the rule trait all name exactly one policy parameter instead of a loose collection of generics, and a
/// non-array universe is a first-class instantiation rather than an afterthought.
///
/// Implementors are zero-sized markers whose single generic implementation covers every destination [`Context`] of
/// their type system, following the [`BatchingPolicy`](crate::BatchingPolicy) precedent. This is deliberately not a
/// destination-context capability: a capability implemented by contexts would need a coherence-foreclosing blanket
/// implementation to achieve the same coverage.
///
/// `C` is the *destination* universe that discharge writes into, and is bounded by [`Domain`] rather than
/// [`Context`] so that naming this policy never obliges a caller to prove an active binding contract. The alias
/// application functions therefore reach their work through value-level capabilities on [`Domain::Value`], which is
/// what lets one implementation serve an eager destination and a staging destination alike: a staged
/// [`Tracer`](crate::Tracer) implements those capabilities by recording instructions. The destination context is
/// still passed to each of them, because a universe whose alias mechanics need context-owned state, or need to bind
/// an operation rather than call a value capability, narrows `C` to [`Context`] on its own implementation and uses
/// it.
///
/// This trait carries only what every reference universe can serve: allocation, reading, write-only replacement, and
/// swapping. Ordered accumulation is a separate contract, [`ReferenceAccumulationPolicy`], because its availability
/// and destination requirements genuinely vary. Splitting it keeps capability requirements at per-access granularity
/// rather than collapsing them into one implementation-level union: a universe that cannot accumulate implements
/// only this trait and still discharges every program that reads and replaces, while a program containing
/// `reference_add_update` fails to discharge for it at compile time, scoped to exactly that operation.
/// Each implementation additionally states whatever its own alias mechanics need on its `impl` block.
///
/// An implementation should leave [`Domain::Value`] generic and constrain it by the capabilities it uses, rather than
/// pinning it to one concrete value type, because a pinned policy serves exactly one destination. Pinning also
/// interacts badly with restating a capability bound: a bound on a concrete type is one Rust rejects outright unless
/// that type really does implement the capability, and a concrete backend value family cannot satisfy such a bound by
/// implementing the capability directly either, because the value-level arithmetic sugar is a blanket implementation
/// whose disjointness a downstream crate cannot prove.
pub trait ReferenceDischargePolicy<C: Domain>: Copy + Clone + Debug {
    /// Referent type system of this universe's references. A discharged root's immutable state is a
    /// [`Domain::Value`] whose type is this universe's lift of the referent.
    type Referent: Type;

    /// Composed alias metadata carried by one flowing reference handle. This is the complete view chain from the root
    /// to the handle, so a handle's view has exactly one source of truth during discharge. A reference family with no
    /// views uses a unit alias, whose composition and application are trivially the identity.
    type Alias: Clone + Debug + Parameter;

    /// Returns the identity alias of an unviewed root with the provided referent type, which is the alias that
    /// allocation and entry-boundary binding install on a fresh root handle.
    ///
    /// This is infallible by design. Validating a referent type is type inference's job, and deriving the identity
    /// alias of an already-valid referent is total.
    fn root_alias(referent: &Self::Referent) -> Self::Alias;

    /// Lifts a reference type into the destination type universe. This is the direction that types a reference-typed
    /// boundary position or a preserved handle in the destination program.
    fn lift_reference_type(r#type: ReferenceType<Self::Referent>) -> C::Type;

    /// Lifts a referent type into the destination type universe. A discharged root's immutable state is an ordinary
    /// destination value of exactly this type, so this is the direction that types an entry-boundary position whose
    /// reference became state, and the direction a rule uses to describe that state to the destination.
    fn lift_referent_type(referent: Self::Referent) -> C::Type;

    /// Projects a destination type back onto the reference type it denotes, or returns [`None`] when it denotes an
    /// ordinary value. Together with [`lift_reference_type`](Self::lift_reference_type) this is the conversion seam
    /// that access rules use to type-check their operands, so a non-reference operand is a classification outcome
    /// here and becomes the calling rule's own diagnostic rather than an error raised by the policy.
    fn project_reference_type(r#type: &C::Type) -> Option<ReferenceType<Self::Referent>>;

    /// Applies `alias` to one immutable state value and returns the selected value.
    ///
    /// # Parameters
    ///
    ///   - `context`: Destination context that owns `current` and any work this application performs.
    ///   - `current`: Current immutable state of the whole root.
    ///   - `alias`: Composed view chain selecting the coordinates to read.
    fn read(context: &C, current: &C::Value, alias: &Self::Alias) -> Result<C::Value, ProgramError>;

    /// Replaces the coordinates that `alias` selects and returns the successor state of the whole root without
    /// observing the previous selection.
    ///
    /// # Parameters
    ///
    ///   - `context`: Destination context that owns `current` and any work this application performs.
    ///   - `current`: Current immutable state of the whole root. A view implementation may consult it only to
    ///     preserve coordinates outside the selected logical handle.
    ///   - `replacement`: Value written into the selected coordinates.
    ///   - `alias`: Composed view chain selecting the coordinates to replace.
    fn write(
        context: &C,
        current: &C::Value,
        replacement: C::Value,
        alias: &Self::Alias,
    ) -> Result<C::Value, ProgramError>;

    /// Replaces the coordinates that `alias` selects and returns the previous selection followed by the successor
    /// state of the whole root.
    ///
    /// # Parameters
    ///
    ///   - `context`: Destination context that owns `current` and any work this application performs.
    ///   - `current`: Current immutable state of the whole root.
    ///   - `replacement`: Value written into the selected coordinates.
    ///   - `alias`: Composed view chain selecting the coordinates to replace.
    fn replace(
        context: &C,
        current: &C::Value,
        replacement: C::Value,
        alias: &Self::Alias,
    ) -> Result<(C::Value, C::Value), ProgramError>;
}

/// Ordered accumulation contract of a reference universe whose references support additive updates.
///
/// Accumulation is the one access mode a reference universe may be unable to serve, and the destination requirement
/// it implies is universe-specific: one universe adds through a value-level capability, another lifts an addition
/// operation into its destination operation family, and a third has no addition at all. A single requirement stated
/// on [`ReferenceDischargePolicy`] could serve none of them, so accumulation lives here instead, and each
/// implementation states its own destination requirement on its `impl` block.
///
/// The two ways accumulation can be unavailable stay distinct. A universe that cannot accumulate does not implement
/// this trait, so a program containing `reference_add_update` fails to discharge for it at compile time while reads
/// and replacements keep working through the base policy. A universe whose destination could add but whose references
/// forbid accumulation implements this trait with an explicit [`ProgramError::UnsupportedOperation`] rejection
/// instead. Closed operation-enum dispatch reintroduces the requirement for any enum whose members include an
/// accumulating operation, exactly as ordinary interpretation already does.
pub trait ReferenceAccumulationPolicy<C: Domain>: ReferenceDischargePolicy<C> {
    /// Accumulates `update` into the coordinates that `alias` selects and returns the successor state of the whole
    /// root.
    ///
    /// # Parameters
    ///
    ///   - `context`: Destination context that owns `current` and any work this application performs.
    ///   - `current`: Current immutable state of the whole root.
    ///   - `update`: Value added into the selected coordinates.
    ///   - `alias`: Composed view chain selecting the coordinates to accumulate into.
    fn accumulate(
        context: &C,
        current: &C::Value,
        update: C::Value,
        alias: &Self::Alias,
    ) -> Result<C::Value, ProgramError>;
}

/// Identity of one reference root inside a running reference discharge.
///
/// Handles are minted by [`ReferenceDischargeContext`] as roots enter its environment, so they are interpreter
/// identities rather than source-program coordinates: they exist only for the duration of one discharge and are
/// meaningful only against the environment that produced them. Pre-transform identity for caller-facing selection is
/// [`ReferenceDischargeSite`] instead.
///
/// Each handle records which environment minted it, so a handle from an unrelated discharge is reported rather than
/// silently addressing whichever root happens to occupy the same position. That is also what isolates a structured
/// rule's region fork: the fork mints its own environment, so a caller handle cannot address a fork root and a fork
/// handle cannot address a caller root. The one table relating the two lives inside
/// [`ReferenceDischargeDriver::discharge_region_program`], which reports its results in caller terms.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ReferenceRootHandle {
    /// Environment that minted this handle.
    environment: ReferenceDischargeEnvironmentId,

    /// Position of the root in that environment.
    index: usize,
}

impl Display for ReferenceRootHandle {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "reference root {}:{}", self.environment.0, self.index)
    }
}

/// Identity of one reference discharge root environment, shared by every clone of the context that owns it and
/// distinct for every environment a structured rule's region fork mints.
///
/// This is private because no caller ever names it: it exists to make [`ReferenceRootHandle`] addressable only in
/// the environment that minted it, and a handle is obtained from the context rather than constructed.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct ReferenceDischargeEnvironmentId(usize);

impl ReferenceDischargeEnvironmentId {
    /// Returns a fresh environment identity, distinct from every identity handed out so far in this process.
    fn next() -> Self {
        static NEXT_ENVIRONMENT_ID: AtomicUsize = AtomicUsize::new(0);
        Self(NEXT_ENVIRONMENT_ID.fetch_add(1, Ordering::Relaxed))
    }
}

/// Environment entry describing what one live reference root became during reference discharge.
#[derive(Debug)]
enum ReferenceRootState<V> {
    /// Root selected for discharge, which threads through the destination program as immutable state.
    Discharged {
        /// Current immutable state of the whole root.
        current: V,

        /// Whether any ordered write or accumulation has been applied to this root. Read-only roots are pruned from
        /// hidden outputs and from structured-operation widening, so this is the fact that pruning consults.
        mutated: bool,
    },

    /// Root not selected for discharge, which survives in the destination program as a reference value. This is the
    /// root's own destination reference value and is what boundary threading uses; a handle derived from it through
    /// a view carries its own exact destination value instead.
    Preserved {
        /// Destination reference-typed value denoting the root.
        reference: V,
    },
}

/// Reference roots the capture prefix of one discharge scope binds.
///
/// A capture-lifted program names its caller's references through constants rather than through its own boundary: the
/// entry boundary carries the lifted capture prefix, and an attached region inside that program names the very same
/// references through capture constants. Resolving one is therefore a property of the scope a region discharges
/// under, not of any rule, so the scope rides on [`ReferenceDischargeContext`] beside the root environment and is
/// recomputed at every region boundary — inherited by default, and replaced by a fresh prefix wherever an operation
/// declares one through [`Operation::region_capture_input_count`].
///
/// Recognizing a capture is a *constant-family* question, and the interpreter deliberately serves families that are
/// not capture-bearing at all, so the seam is a function pointer supplied by the entry point that knows the family
/// rather than a [`CaptureConstant`] bound on the whole architecture. The [`Default`] scope recognizes nothing and
/// binds nothing, which is exactly the behavior of a program that has no captures.
struct ReferenceCaptureScope<Constant> {
    /// Capture position a constant names, or [`None`] when it is an ordinary constant of its family.
    capture_index: fn(&Constant) -> Option<usize>,

    /// Root each capture position binds, or [`None`] when that position carries an ordinary value rather than a
    /// reference. A capture position past the end of this list binds nothing.
    roots: Rc<[Option<ReferenceRootHandle>]>,
}

impl<Constant> ReferenceCaptureScope<Constant> {
    /// Creates a capture scope.
    ///
    /// # Parameters
    ///
    ///   - `capture_index`: Seam reporting the capture position a constant of this family names.
    ///   - `roots`: Root each capture position binds, in capture order.
    #[inline]
    fn new(capture_index: fn(&Constant) -> Option<usize>, roots: Vec<Option<ReferenceRootHandle>>) -> Self {
        Self { capture_index, roots: roots.into() }
    }

    /// Returns the root each capture position binds, in capture order.
    #[inline]
    fn roots(&self) -> &[Option<ReferenceRootHandle>] {
        self.roots.as_ref()
    }

    /// Returns the root one constant denotes, or [`None`] when the constant names no capture position or that
    /// position binds no root. A constant this scope cannot resolve is an ordinary constant of its family, and a
    /// reference-typed one that no scope resolves is rejected where it is lifted.
    #[inline]
    fn resolve(&self, constant: &Constant) -> Option<ReferenceRootHandle> {
        (self.capture_index)(constant).and_then(|index| self.roots.get(index).copied().flatten())
    }

    /// Returns this scope's seam over a different set of bound roots, which is how a nested region's scope and a
    /// region fork's remapped scope are built without restating the constant family's recognition rule.
    #[inline]
    fn with_roots(&self, roots: Vec<Option<ReferenceRootHandle>>) -> Self {
        Self { capture_index: self.capture_index, roots: roots.into() }
    }
}

impl<Constant> Default for ReferenceCaptureScope<Constant> {
    #[inline]
    fn default() -> Self {
        Self { capture_index: |_| None, roots: Rc::from([]) }
    }
}

impl<Constant> Clone for ReferenceCaptureScope<Constant> {
    #[inline]
    fn clone(&self) -> Self {
        Self { capture_index: self.capture_index, roots: Rc::clone(&self.roots) }
    }
}

impl<Constant> Debug for ReferenceCaptureScope<Constant> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("ReferenceCaptureScope").field("roots", &self.roots).finish_non_exhaustive()
    }
}

/// Handle to one live reference root flowing through reference discharge.
///
/// The fields are private and only [`ReferenceDischargeContext`] constructs them, so a rule can read a handle but
/// cannot fabricate a root, an alias, a derived type, or a preserved destination value. That keeps root identity and
/// view composition checked even though the rule trait is open to third-party operations.
pub struct ReferenceDischargeReference<C: Domain, P: ReferenceDischargePolicy<C>> {
    /// Identity of the root this handle denotes.
    root: ReferenceRootHandle,

    /// Whether this handle denotes the complete root rather than any derived view of it.
    denotes_whole_root: bool,

    /// Composed policy-owned view chain from the root to this handle.
    alias: P::Alias,

    /// Reference type this exact handle exposes, which differs from the root's type under a composed view.
    r#type: ReferenceType<P::Referent>,

    /// Exact destination reference value this handle denotes when its root remains a reference in the rewritten
    /// program, and [`None`] when the root became explicit immutable state.
    ///
    /// A preserved handle must consume this value rather than re-deriving its view chain per access, because
    /// re-deriving would duplicate and reorder the replayed view operations in the destination program.
    preserved: Option<C::Value>,
}

impl<C: Domain, P: ReferenceDischargePolicy<C>> ReferenceDischargeReference<C, P> {
    /// Returns the identity of the root this handle denotes.
    #[inline]
    pub const fn root(&self) -> ReferenceRootHandle {
        self.root
    }

    /// Returns whether this handle denotes the complete root rather than a derived view.
    #[inline]
    const fn denotes_whole_root(&self) -> bool {
        self.denotes_whole_root
    }

    /// Returns the composed view chain from the root to this handle.
    #[inline]
    pub const fn alias(&self) -> &P::Alias {
        &self.alias
    }

    /// Returns the reference type this exact handle exposes.
    #[inline]
    pub const fn r#type(&self) -> &ReferenceType<P::Referent> {
        &self.r#type
    }

    /// Returns the exact destination reference value of a preserved handle, or [`None`] when the root was
    /// discharged.
    #[inline]
    pub const fn preserved(&self) -> Option<&C::Value> {
        self.preserved.as_ref()
    }
}

impl<C: Domain, P: ReferenceDischargePolicy<C>> Clone for ReferenceDischargeReference<C, P> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            root: self.root,
            denotes_whole_root: self.denotes_whole_root,
            alias: self.alias.clone(),
            r#type: self.r#type.clone(),
            preserved: self.preserved.clone(),
        }
    }
}

impl<C: Domain, P: ReferenceDischargePolicy<C>> Debug for ReferenceDischargeReference<C, P> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ReferenceDischargeReference")
            .field("root", &self.root)
            .field("denotes_whole_root", &self.denotes_whole_root)
            .field("alias", &self.alias)
            .field("type", &self.r#type)
            .field("preserved", &self.preserved)
            .finish()
    }
}

impl<C: Domain, P: ReferenceDischargePolicy<C>> Display for ReferenceDischargeReference<C, P> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{} {}", self.root, self.r#type)
    }
}

impl<C: Domain, P: ReferenceDischargePolicy<C>> PartialEq for ReferenceDischargeReference<C, P>
where
    C::Value: PartialEq,
    P::Alias: PartialEq,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.root == other.root
            && self.denotes_whole_root == other.denotes_whole_root
            && self.alias == other.alias
            && self.r#type == other.r#type
            && self.preserved == other.preserved
    }
}

/// Context-free carrier flowing through reference discharge.
///
/// This is the payload of a [`ReferenceDischargeTracer`], separated from it because the rule trait passes and
/// returns carriers while the context stamp is the transform's own bookkeeping. It is public because the rule trait
/// names it, and because enum variant fields are always as public as their enum, the reference payload is the opaque
/// [`ReferenceDischargeReference`] rather than inline fields.
pub enum ReferenceDischargeValue<C: Domain, P: ReferenceDischargePolicy<C>> {
    /// Ordinary destination value, carrying no reference and replayed as-is.
    Ordinary(C::Value),

    /// Handle to one live reference root.
    Reference(ReferenceDischargeReference<C, P>),
}

impl<C: Domain, P: ReferenceDischargePolicy<C>> ReferenceDischargeValue<C, P> {
    /// Returns the ordinary destination value this carrier holds, or an error naming `expectation` when it holds a
    /// reference handle instead.
    ///
    /// # Parameters
    ///
    ///   - `expectation`: Description of the operand the caller expected, used in the diagnostic.
    pub fn expect_ordinary(&self, expectation: &str) -> Result<&C::Value, ProgramError> {
        match self {
            Self::Ordinary(value) => Ok(value),
            Self::Reference(reference) => Err(ProgramError::MalformedProgram(format!(
                "reference discharge expected {expectation} but received {reference}",
            ))),
        }
    }

    /// Returns the reference handle this carrier holds, or an error naming `expectation` when it holds an ordinary
    /// value instead.
    ///
    /// # Parameters
    ///
    ///   - `expectation`: Description of the operand the caller expected, used in the diagnostic.
    pub fn expect_reference(&self, expectation: &str) -> Result<&ReferenceDischargeReference<C, P>, ProgramError> {
        match self {
            Self::Reference(reference) => Ok(reference),
            Self::Ordinary(_) => Err(ProgramError::MalformedProgram(format!(
                "reference discharge expected {expectation} but received an ordinary value",
            ))),
        }
    }
}

impl<C: Domain, P: ReferenceDischargePolicy<C>> Clone for ReferenceDischargeValue<C, P> {
    #[inline]
    fn clone(&self) -> Self {
        match self {
            Self::Ordinary(value) => Self::Ordinary(value.clone()),
            Self::Reference(reference) => Self::Reference(reference.clone()),
        }
    }
}

impl<C: Domain, P: ReferenceDischargePolicy<C>> Debug for ReferenceDischargeValue<C, P> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ordinary(value) => formatter.debug_tuple("Ordinary").field(value).finish(),
            Self::Reference(reference) => formatter.debug_tuple("Reference").field(reference).finish(),
        }
    }
}

impl<C: Domain, P: ReferenceDischargePolicy<C>> Display for ReferenceDischargeValue<C, P> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ordinary(value) => Display::fmt(value, formatter),
            Self::Reference(reference) => Display::fmt(reference, formatter),
        }
    }
}

impl<C: Domain, P: ReferenceDischargePolicy<C>> PartialEq for ReferenceDischargeValue<C, P>
where
    C::Value: PartialEq,
    P::Alias: PartialEq,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Ordinary(value), Self::Ordinary(other)) => value == other,
            (Self::Reference(reference), Self::Reference(other)) => reference == other,
            _ => false,
        }
    }
}

impl<C: Domain, P: ReferenceDischargePolicy<C>> Typed for ReferenceDischargeValue<C, P> {
    type Type = C::Type;

    #[inline]
    fn r#type(&self) -> Cow<'_, C::Type> {
        match self {
            Self::Ordinary(value) => value.r#type(),
            Self::Reference(reference) => Cow::Owned(P::lift_reference_type(reference.r#type().clone())),
        }
    }
}

/// Complete environment record of one live root: the reference type the whole root exposes, and what discharge turned
/// that root into.
///
/// The reference type is recorded because a root's identity outlives every handle that denotes it. A structured rule
/// threading an inherited root through a rebuilt region boundary holds only that root's handle, never a handle it
/// could read a type off, so the environment is where the whole-root type has to live.
struct ReferenceRootEntry<T: Type, V> {
    /// Reference type of the whole root, whose referent types the immutable state a discharged root threads.
    r#type: ReferenceType<T>,

    /// What discharge turned this root into.
    state: ReferenceRootState<V>,
}

/// Live root environment of one reference discharge, shared by every clone of its context.
struct ReferenceDischargeEnvironment<T: Type, V> {
    /// Identity that every handle minted from this environment records.
    id: ReferenceDischargeEnvironmentId,

    /// State of every root minted so far, indexed by [`ReferenceRootHandle`]. A consumed root keeps its slot and
    /// becomes [`None`], so a use-after-consume is reported against the exact root rather than as an unknown handle.
    roots: Vec<Option<ReferenceRootEntry<T, V>>>,
}

impl<T: Type, V> ReferenceDischargeEnvironment<T, V> {
    /// Returns the state slot that `root` names, or an error when the handle belongs to another environment or names
    /// a position this environment never minted.
    fn slot(&self, root: ReferenceRootHandle) -> Result<&Option<ReferenceRootEntry<T, V>>, ProgramError> {
        if root.environment != self.id {
            return Err(ProgramError::MalformedProgram(format!(
                "reference discharge accessed {root}, which belongs to an environment other than the active `{}`",
                self.id.0,
            )));
        }
        self.roots
            .get(root.index)
            .ok_or_else(|| ProgramError::MalformedProgram(format!("reference discharge accessed never-bound {root}")))
    }
}

/// Active [`Context`] of one reference discharge, owning the live root environment that its flowing values refer to.
///
/// Discharge is a single program-to-program interpretation, so it runs through
/// [`interpret_in_context`](Program::interpret_in_context) like batching and differentiation: operations bound
/// through this context dispatch to their [`ReferenceDischargeableOperation`] rules, and those rules bind the
/// rewritten, reference-free work through [`parent`](Self::parent).
///
/// Its state lives here rather than in the flowing values because a reference is an identity, not a payload: several
/// handles can denote the same root through different views, and every one of them must observe the same current
/// state. Clones therefore share one environment, exactly as every other stateful context in Ryft shares one active
/// builder. A structured rule that must rebuild an attached region instead runs it against an isolated environment
/// through [`ReferenceDischargeDriver::discharge_region_program`], which commits nothing here.
pub struct ReferenceDischargeContext<C: Domain, P: ReferenceDischargePolicy<C>> {
    /// Destination context that owns the discharged values and executes or stages the rewritten work.
    parent: C,

    /// Live root environment shared by every clone of this context.
    environment: Rc<RefCell<ReferenceDischargeEnvironment<P::Referent, C::Value>>>,

    /// Roots the capture prefix of the scope this context discharges binds. A region that inherits its parent's
    /// capture prefix discharges under the same scope; a region fork rebuilds the scope in its own root terms.
    captures: ReferenceCaptureScope<C::Constant>,

    /// Reference sites this discharge normalizes into immutable state. Every root the selection omits is preserved,
    /// and the selection is shared unchanged by every clone and by every region fork, because a site names a source
    /// coordinate that means the same thing wherever the replay reaches it.
    selection: ReferenceDischargeSelection,
}

impl<C: Domain, P: ReferenceDischargePolicy<C>> ReferenceDischargeContext<C, P> {
    /// Creates a discharge context with an empty root environment and an empty capture scope over the provided
    /// destination context, discharging every reference it reaches.
    ///
    /// The capture scope is populated afterwards rather than here, because the roots it binds are minted by this very
    /// context as its boundary is threaded. Partial discharge is requested through
    /// [`Program::partially_discharge_references_with_policy`] rather than by constructing a context, so that a
    /// selection is always validated against the program whose coordinates it names.
    #[inline]
    pub fn new(parent: C) -> Self {
        Self::new_selecting(parent, ReferenceDischargeSelection::everything())
    }

    /// Creates a discharge context with an empty root environment and an empty capture scope over the provided
    /// destination context, discharging exactly the references `selection` names.
    #[inline]
    fn new_selecting(parent: C, selection: ReferenceDischargeSelection) -> Self {
        Self {
            parent,
            environment: Rc::new(RefCell::new(ReferenceDischargeEnvironment {
                id: ReferenceDischargeEnvironmentId::next(),
                roots: Vec::new(),
            })),
            captures: ReferenceCaptureScope::default(),
            selection,
        }
    }

    /// Returns whether the allocation an operation application performs was selected for discharge, which is what an
    /// allocation rule asks before deciding between a discharged root and one that survives in the destination.
    ///
    /// An application that did not come from a replayed instruction has no source coordinate and is always
    /// discharged: no [`ReferenceDischargeSite`] can name it, so declining it would make direct
    /// [`Context::bind`] unusable rather than expressing a caller's choice.
    ///
    /// This is the only selection question a rule ever asks, which is why it is the only one exposed. Whether an
    /// *entry-boundary* root was selected is decided once, by the program-level entry point that threads the boundary,
    /// and no rule is in a position to ask it.
    ///
    /// # Parameters
    ///
    ///   - `instruction`: Replay position of the application, from [`ReferenceDischargeDriver::instruction`].
    ///   - `output_index`: Output position at which the application defines the fresh root.
    #[inline]
    pub fn selects_allocation(&self, instruction: Option<InstructionId>, output_index: usize) -> bool {
        instruction.is_none_or(|instruction| {
            self.selection.selects(ReferenceDischargeSite::Allocation { instruction, output_index })
        })
    }

    /// Returns whether one entry-boundary root was selected for discharge.
    #[inline]
    fn selects_external(&self, source: ReferenceSource) -> bool {
        self.selection.selects(ReferenceDischargeSite::External(source))
    }

    /// Returns the destination context that owns the discharged values.
    #[inline]
    pub const fn parent(&self) -> &C {
        &self.parent
    }

    /// Returns the capture scope this context discharges under.
    #[inline]
    const fn captures(&self) -> &ReferenceCaptureScope<C::Constant> {
        &self.captures
    }

    /// Returns this context discharging under a different capture scope, sharing its live root environment.
    ///
    /// A region fork reaches its own scope this way, because the roots that scope binds are minted by the fork itself
    /// and therefore exist only once its boundary has been threaded.
    #[inline]
    fn with_captures(&self, captures: ReferenceCaptureScope<C::Constant>) -> Self
    where
        C: Clone,
    {
        Self {
            parent: self.parent.clone(),
            environment: Rc::clone(&self.environment),
            captures,
            selection: self.selection.clone(),
        }
    }

    /// Binds a fresh root that threads as immutable state and returns the unviewed handle denoting it.
    ///
    /// # Parameters
    ///
    ///   - `r#type`: Reference type of the fresh root, normally derived from the allocating operation's inferred
    ///     output type through [`ReferenceDischargePolicy::project_reference_type`].
    ///   - `initial`: Destination value that becomes the root's initial immutable state.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when `initial` does not carry the lifted referent type of `r#type`.
    pub fn allocate_discharged(
        &self,
        r#type: ReferenceType<P::Referent>,
        initial: C::Value,
    ) -> Result<ReferenceDischargeValue<C, P>, ProgramError> {
        validate_discharged_value_type::<C, P>(&initial, &r#type)?;
        Ok(self.bind_root_value(r#type, ReferenceRootState::Discharged { current: initial, mutated: false }, None))
    }

    /// Binds a root that survives in the destination program and returns the unviewed handle denoting it.
    ///
    /// # Parameters
    ///
    ///   - `r#type`: Reference type of the root.
    ///   - `reference`: Destination reference-typed value denoting the root.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when `reference` does not carry the reference type `r#type`.
    pub fn bind_preserved(
        &self,
        r#type: ReferenceType<P::Referent>,
        reference: C::Value,
    ) -> Result<ReferenceDischargeValue<C, P>, ProgramError> {
        validate_preserved_value::<C, P>(&reference, &r#type)?;
        Ok(self.bind_root_value(
            r#type,
            ReferenceRootState::Preserved { reference: reference.clone() },
            Some(reference),
        ))
    }

    /// Returns a handle that composes `alias` onto `reference`, denoting the same root through a derived view.
    ///
    /// The composed alias is the authoritative view chain for the derived handle, so callers pass the complete chain
    /// rather than a single step. A derived handle on a preserved root must carry the destination value produced by
    /// replaying the view operation, so that later accesses consume that exact value instead of re-deriving the
    /// chain.
    ///
    /// # Parameters
    ///
    ///   - `reference`: Handle the view is composed onto.
    ///   - `alias`: Complete composed view chain of the derived handle.
    ///   - `r#type`: Reference type the derived handle exposes.
    ///   - `preserved`: Destination reference value of the derived handle, required when the root is preserved and
    ///     rejected when it was discharged.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when the root is no longer live, when `preserved` disagrees with
    /// the root's state, or when `preserved` does not carry the reference type `r#type`.
    pub fn derive(
        &self,
        reference: &ReferenceDischargeReference<C, P>,
        alias: P::Alias,
        r#type: ReferenceType<P::Referent>,
        preserved: Option<C::Value>,
    ) -> Result<ReferenceDischargeValue<C, P>, ProgramError> {
        let root = reference.root();
        match (self.root_is_discharged(root)?, &preserved) {
            (false, Some(preserved)) => {
                validate_preserved_value::<C, P>(preserved, &r#type)?;
            }
            (true, None) => {}
            (false, None) => {
                return Err(ProgramError::MalformedProgram(format!(
                    "reference discharge derived a handle from {root} without a destination reference value, but \
                     that root is preserved",
                )));
            }
            (true, Some(_)) => {
                return Err(ProgramError::MalformedProgram(format!(
                    "reference discharge derived a handle from {root} with a destination reference value, but that \
                     root is discharged",
                )));
            }
        }
        Ok(ReferenceDischargeValue::Reference(ReferenceDischargeReference {
            root,
            denotes_whole_root: false,
            alias,
            r#type,
            preserved,
        }))
    }

    /// Returns the reference type of one live root as a whole, which is the type a handle denoting the complete root
    /// exposes and whose referent types the root's immutable state.
    ///
    /// A structured rule threading an inherited root through a rebuilt region boundary knows only that root's
    /// identity, so this is how it recovers the type the boundary position must carry.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when the root was consumed or was never bound in this context.
    pub fn root_reference_type(&self, root: ReferenceRootHandle) -> Result<ReferenceType<P::Referent>, ProgramError> {
        self.with_root_entry(root, |entry| entry.r#type.clone())
    }

    /// Returns the current immutable state of one discharged root.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when the root is not live or was preserved rather than discharged.
    pub fn discharged_state(&self, root: ReferenceRootHandle) -> Result<C::Value, ProgramError> {
        self.with_root_entry(root, |entry| match &entry.state {
            ReferenceRootState::Discharged { current, .. } => Ok(current.clone()),
            ReferenceRootState::Preserved { .. } => Err(ProgramError::MalformedProgram(format!(
                "reference discharge requested the discharged state of preserved {root}",
            ))),
        })?
    }

    /// Installs the immutable state of one discharged root and records that the root was mutated.
    ///
    /// A primitive access rule uses this for the write it just performed. A structured rule merges a boundary's
    /// returned state through [`merge_discharged_state`](Self::merge_discharged_state) instead, because a symmetric
    /// boundary returns a successor state for roots it never wrote and the mutation flag decides whether a root
    /// publishes a hidden final-state output.
    ///
    /// # Parameters
    ///
    ///   - `root`: Discharged root whose state is being installed.
    ///   - `current`: Successor immutable state of the whole root.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when the root is not live or was preserved rather than discharged.
    pub fn set_discharged_state(&self, root: ReferenceRootHandle, current: C::Value) -> Result<(), ProgramError> {
        self.install_discharged_state(root, current, true)
    }

    /// Installs the state one boundary carried back out for a discharged root, recording a mutation only when that
    /// boundary's closure actually wrote it.
    ///
    /// A loop-shaped boundary is symmetric: it returns a successor state for every root it carries, including roots
    /// its closure only read. The value that comes back for such a root equals the one that entered, so re-threading
    /// it keeps the destination consistent — but recording it as a write would not, because the mutation flag is what
    /// decides whether an external root publishes a hidden final-state output and therefore whether its caller updates
    /// the shared reference state. A read-only loop must leave its caller's reference state unchanged.
    ///
    /// # Parameters
    ///
    ///   - `root`: Discharged root whose state is being merged.
    ///   - `current`: State the boundary carried back out, which for an unwritten root equals the entering state.
    ///   - `mutated`: Whether the boundary's closure wrote or accumulated into the root.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when the root is not live or was preserved rather than discharged.
    pub fn merge_discharged_state(
        &self,
        root: ReferenceRootHandle,
        current: C::Value,
        mutated: bool,
    ) -> Result<(), ProgramError> {
        self.install_discharged_state(root, current, mutated)
    }

    /// Returns whether any ordered write or accumulation has been applied to one discharged root.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when the root is not live or was preserved rather than discharged.
    pub fn is_mutated(&self, root: ReferenceRootHandle) -> Result<bool, ProgramError> {
        self.with_root_entry(root, |entry| match &entry.state {
            ReferenceRootState::Discharged { mutated, .. } => Ok(*mutated),
            ReferenceRootState::Preserved { .. } => Err(ProgramError::MalformedProgram(format!(
                "reference discharge queried mutation of preserved {root}",
            ))),
        })?
    }

    /// Returns every root that is still live in this context's environment, in binding order.
    pub fn live_roots(&self) -> Vec<ReferenceRootHandle> {
        let environment = self.environment.borrow();
        environment
            .roots
            .iter()
            .enumerate()
            .filter(|(_, state)| state.is_some())
            .map(|(index, _)| ReferenceRootHandle { environment: environment.id, index })
            .collect()
    }

    /// Reads the coordinates that `reference` selects from its root's current state.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when the root is not live, and propagates the policy's error when
    /// the alias cannot be applied. Reading a preserved root through this function is rejected, because a preserved
    /// access must replay verbatim in the destination instead.
    pub fn read(&self, reference: &ReferenceDischargeReference<C, P>) -> Result<C::Value, ProgramError> {
        let current = self.discharged_state(reference.root())?;
        P::read(&self.parent, &current, reference.alias())
    }

    /// Replaces the coordinates that `reference` selects without observing their previous contents.
    ///
    /// # Parameters
    ///
    ///   - `reference`: Handle selecting the coordinates to replace.
    ///   - `replacement`: Value written into the selected coordinates.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when the root is not live or was preserved, and propagates the
    /// policy's error when the write cannot be applied.
    pub fn write(
        &self,
        reference: &ReferenceDischargeReference<C, P>,
        replacement: C::Value,
    ) -> Result<(), ProgramError> {
        let root = reference.root();
        let current = self.discharged_state(root)?;
        let successor = P::write(&self.parent, &current, replacement, reference.alias())?;
        self.set_discharged_state(root, successor)
    }

    /// Replaces the coordinates that `reference` selects and returns their previous contents.
    ///
    /// # Parameters
    ///
    ///   - `reference`: Handle selecting the coordinates to replace.
    ///   - `replacement`: Value written into the selected coordinates.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when the root is not live or was preserved, and propagates the
    /// policy's error when the alias cannot be applied.
    pub fn replace(
        &self,
        reference: &ReferenceDischargeReference<C, P>,
        replacement: C::Value,
    ) -> Result<C::Value, ProgramError> {
        let root = reference.root();
        let current = self.discharged_state(root)?;
        let (previous, successor) = P::replace(&self.parent, &current, replacement, reference.alias())?;
        self.set_discharged_state(root, successor)?;
        Ok(previous)
    }

    /// Accumulates `update` into the coordinates that `reference` selects.
    ///
    /// # Parameters
    ///
    ///   - `reference`: Handle selecting the coordinates to accumulate into.
    ///   - `update`: Value added into the selected coordinates.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when the root is not live or was preserved, and propagates the
    /// policy's error when the alias cannot be applied or the universe forbids accumulation.
    pub fn accumulate(
        &self,
        reference: &ReferenceDischargeReference<C, P>,
        update: C::Value,
    ) -> Result<(), ProgramError>
    where
        P: ReferenceAccumulationPolicy<C>,
    {
        let root = reference.root();
        let current = self.discharged_state(root)?;
        let successor = P::accumulate(&self.parent, &current, update, reference.alias())?;
        self.set_discharged_state(root, successor)
    }

    /// Yields the current whole-root state of `reference`'s root and unbinds the root, so that every later access to
    /// it is reported as a use-after-consume.
    ///
    /// Consumption is a whole-root event and always yields the whole root's state, so the handle's alias is
    /// deliberately not applied. A derived handle therefore cannot name a consumption, even when its referent type
    /// happens to equal the root's. The invariant is enforced at the state transition where it is relied upon.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when the root is not live, was preserved rather than discharged, or
    /// is named through a derived handle rather than the original whole-root handle.
    pub fn consume(&self, reference: &ReferenceDischargeReference<C, P>) -> Result<C::Value, ProgramError> {
        let root = reference.root();
        let current_type = self.with_root_entry(root, |entry| match &entry.state {
            ReferenceRootState::Discharged { current, .. } => Ok(current.r#type().into_owned()),
            ReferenceRootState::Preserved { .. } => Err(ProgramError::MalformedProgram(format!(
                "reference discharge requested the discharged state of preserved {root}",
            ))),
        })??;
        if !reference.denotes_whole_root() {
            return Err(ProgramError::MalformedProgram(format!(
                "reference discharge cannot consume {root} through the derived view `{}`; consumption yields the \
                 whole root, whose referent is `{}`",
                reference.r#type(),
                current_type,
            )));
        }
        let mut environment = self.environment.borrow_mut();
        // The inspection above proved that this handle belongs to this environment and names a live discharged root.
        let entry = environment.roots[root.index].take().unwrap();
        let ReferenceRootState::Discharged { current, .. } = entry.state else { unreachable!() };
        Ok(current)
    }

    /// Unbinds one preserved root, so that every later access to it is reported as a use-after-consume.
    ///
    /// This is [`consume`](Self::consume)'s counterpart for a root that survives in the destination. It yields no
    /// value, because the consuming operation was replayed verbatim and its own result is what the destination
    /// produced; all that remains is to stop the discharge environment from handing the root out again. Consumption is
    /// still a whole-root event, so a derived handle cannot name one even when its referent type equals the root's.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when the root is not live, was discharged rather than preserved, or
    /// is named through a derived handle rather than the original whole-root handle.
    pub fn unbind_preserved(&self, reference: &ReferenceDischargeReference<C, P>) -> Result<(), ProgramError> {
        let root = reference.root();
        let whole = self.root_reference_type(root)?;
        if self.root_is_discharged(root)? {
            return Err(ProgramError::MalformedProgram(format!(
                "reference discharge unbound discharged {root} as a preserved root",
            )));
        }
        if !reference.denotes_whole_root() {
            return Err(ProgramError::MalformedProgram(format!(
                "reference discharge cannot consume {root} through the derived view `{}`; consumption yields the \
                 whole root, whose reference type is `{}`",
                reference.r#type(),
                whole,
            )));
        }

        // `root_reference_type` already proved that this handle belongs to this environment and names a live root.
        self.environment.borrow_mut().roots[root.index] = None;
        Ok(())
    }

    /// Summarizes the transitive reference accesses of one region closure, in the terms of the caller roots its
    /// boundary names.
    ///
    /// A structured rule calls this before it can size its state boundary: which roots a region closure touches, and
    /// which of them it mutates, is exactly what decides how wide the rewritten operation must be. The summary is
    /// derived from generic hooks alone — operation-local [`Operation::reference_semantics`], the region-provenance
    /// hooks, reference-output identity, and recursive summaries of nested regions — so a third-party structured
    /// operation needs no companion declaration surface to be summarized.
    ///
    /// The region's own capture scope is derived here rather than supplied, because whether a region establishes a
    /// fresh capture prefix is stated by [`Operation::region_capture_input_count`] and is therefore knowledge the
    /// summary can read off the operation itself. A rule never has to reason about captures.
    ///
    /// # Parameters
    ///
    ///   - `operation`: Operation the region is attached to.
    ///   - `region_index`: Position of the region among that operation's attached regions.
    ///   - `region`: Region whose closure is summarized.
    ///   - `inputs`: Caller root denoted by each of the region's declared inputs, in boundary order, with [`None`]
    ///     wherever the position carries an ordinary value.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when `inputs` does not describe the region's boundary, when the
    /// operation declares a capture prefix longer than the region's boundary, when a reference-typed nested boundary
    /// position declares no provenance the summary could follow, when the closure reaches a reference that entered
    /// neither through its boundary nor through its capture scope, or when the closure consumes a caller root, which
    /// no state boundary can express. It also returns this error when `operation` does not permit one of the exact
    /// access modes the closure performs through `region_index`.
    pub fn region_summary<O: Operation>(
        &self,
        operation: &O,
        region_index: usize,
        region: RegionRef<'_, C::Constant, C::Operation>,
        inputs: &[Option<ReferenceRootHandle>],
    ) -> Result<ReferenceRegionSummary, ProgramError> {
        let captures = nested_capture_scope(
            operation.region_capture_input_count(region_index),
            inputs,
            self.captures(),
            region.id(),
        )?;
        let mut summary = ReferenceRegionSummary::default();
        summary.output_roots = summarize_region_closure(region, inputs, &captures, &mut summary)?;
        validate_region_accesses(operation, region_index, &summary)?;
        Ok(summary)
    }

    /// Returns the whole root one operand of a structured operation denotes, or [`None`] when the operand is an
    /// ordinary value.
    ///
    /// A derived view is rejected rather than resolved to its root. A state boundary carries whole-root values, so
    /// only a handle with whole-root provenance may cross it; the view has to be re-derived from the root inside the
    /// region instead.
    ///
    /// A *preserved* root is resolved like any other. It crosses the boundary as the reference it already is, at its
    /// own declared operand position, so it needs no state carry at all — which is exactly what
    /// [`threaded_state_roots`](Self::threaded_state_roots) filters it out of.
    ///
    /// # Parameters
    ///
    ///   - `operand`: Carrier being classified.
    ///   - `operation`: Name of the operation being rewritten, used in the diagnostic.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when the operand denotes a derived view of its root, or when its
    /// root is no longer live.
    pub fn operand_root(
        &self,
        operand: &ReferenceDischargeValue<C, P>,
        operation: &str,
    ) -> Result<Option<ReferenceRootHandle>, ProgramError> {
        let ReferenceDischargeValue::Reference(reference) = operand else {
            return Ok(None);
        };
        let root = reference.root();
        let whole = self.root_reference_type(root)?;
        if !reference.denotes_whole_root() {
            return Err(ProgramError::MalformedProgram(format!(
                "operation `{operation}` passes the derived view `{}` of {root} across a region boundary, which \
                 carries the whole root `{}`; derive the view inside the region instead",
                reference.r#type(),
                whole,
            )));
        }
        Ok(Some(root))
    }

    /// Returns the roots one region closure needs threaded through the rewritten boundary as immutable state, in
    /// canonical root order, and validates that every one of them is still live.
    ///
    /// A closure needs a root threaded whenever its replay must be able to resolve that root — because it accesses
    /// it, returns it, or merely rematerializes a capture constant that denotes it — so the set is the summary's
    /// [`reached`](ReferenceRegionSummary::reached) roots, a strict superset of the accessed and returned roots, with
    /// the *preserved* roots removed. A preserved root survives in the destination as an
    /// ordinary reference and crosses at its own declared operand position, exactly as the source passed it, so it
    /// needs no state carry, publishes no successor, and widens nothing. This is the one place that distinction is
    /// drawn, which is what keeps the four structured rewrites stating one thing.
    ///
    /// # Parameters
    ///
    ///   - `summary`: Summary of the closures the rewritten operation attaches, in caller-root terms.
    ///   - `operation`: Name of the operation being rewritten, used in the diagnostic.
    ///
    /// # Errors
    ///
    /// Reports the first root the closures reach that has no live state, propagating the environment's own reason —
    /// consumed, never bound, or belonging to another environment — because that reason is what a caller needs and
    /// this check is in no position to restate it.
    pub fn threaded_state_roots(
        &self,
        summary: &ReferenceRegionSummary,
        operation: &str,
    ) -> Result<BTreeSet<ReferenceRootHandle>, ProgramError> {
        let mut threaded = BTreeSet::new();
        for root in summary.reached() {
            if self.root_is_discharged(root).map_err(|error| {
                ProgramError::MalformedProgram(format!("operation `{operation}` reaches {root}: {error}"))
            })? {
                threaded.insert(root);
            }
        }
        Ok(threaded)
    }

    /// Returns the destination value one operand of a structured operation contributes to the rewritten application:
    /// the current immutable state of a discharged root, the destination reference of a preserved one, or the
    /// operand's own ordinary value.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when the operand's root is not live.
    pub fn operand_value(&self, operand: &ReferenceDischargeValue<C, P>) -> Result<C::Value, ProgramError> {
        let reference = match operand {
            ReferenceDischargeValue::Ordinary(value) => return Ok(value.clone()),
            ReferenceDischargeValue::Reference(reference) => reference,
        };
        if self.root_is_discharged(reference.root())? {
            self.discharged_state(reference.root())
        } else {
            // `bind_preserved` and `derive` maintain the invariant that every handle on a preserved root carries the
            // exact destination reference value it denotes.
            reference.preserved().cloned().ok_or_else(|| {
                ProgramError::MalformedProgram(format!(
                    "reference discharge cannot pass preserved {}, whose handle carries no destination reference \
                     value",
                    reference.root(),
                ))
            })
        }
    }

    /// Returns the unviewed handle denoting one live root of this environment.
    ///
    /// This mints no root: it re-exposes one the environment already holds, which is what resolving a capture-scoped
    /// reference constant needs. A preserved root's handle carries the root's own destination reference value,
    /// exactly as [`bind_preserved`](Self::bind_preserved) produced it.
    fn root_handle(&self, root: ReferenceRootHandle) -> Result<ReferenceDischargeValue<C, P>, ProgramError> {
        let (r#type, preserved) = self.with_root_entry(root, |entry| {
            let preserved = match &entry.state {
                ReferenceRootState::Discharged { .. } => None,
                ReferenceRootState::Preserved { reference } => Some(reference.clone()),
            };
            (entry.r#type.clone(), preserved)
        })?;
        let alias = P::root_alias(r#type.referent());
        Ok(ReferenceDischargeValue::Reference(ReferenceDischargeReference {
            root,
            denotes_whole_root: true,
            alias,
            r#type,
            preserved,
        }))
    }

    /// Applies `use_entry` to one live root while holding the environment's immutable borrow.
    ///
    /// Callers must clone only the fields they need and must not invoke policy or destination operations from the
    /// callback, keeping the [`RefCell`] borrow local to this query.
    fn with_root_entry<R>(
        &self,
        root: ReferenceRootHandle,
        use_entry: impl FnOnce(&ReferenceRootEntry<P::Referent, C::Value>) -> R,
    ) -> Result<R, ProgramError> {
        let environment = self.environment.borrow();
        let entry = environment
            .slot(root)?
            .as_ref()
            .ok_or_else(|| ProgramError::MalformedProgram(format!("reference discharge accessed consumed {root}")))?;
        Ok(use_entry(entry))
    }

    /// Returns whether one live root is discharged rather than preserved.
    fn root_is_discharged(&self, root: ReferenceRootHandle) -> Result<bool, ProgramError> {
        self.with_root_entry(root, |entry| matches!(entry.state, ReferenceRootState::Discharged { .. }))
    }

    /// Validates that `root` belongs to this environment and remains live.
    fn validate_live_root(&self, root: ReferenceRootHandle) -> Result<(), ProgramError> {
        self.with_root_entry(root, |_| ())
    }

    /// Validates that `current` carries the lifted referent type of `root` without mutating the environment.
    fn validate_discharged_state_type(
        &self,
        root: ReferenceRootHandle,
        current: &C::Value,
    ) -> Result<(), ProgramError> {
        let r#type = self.root_reference_type(root)?;
        validate_discharged_value_type::<C, P>(current, &r#type)
    }

    /// Appends one root record to the environment and returns the handle that denotes it.
    fn bind_root(
        &self,
        r#type: ReferenceType<P::Referent>,
        state: ReferenceRootState<C::Value>,
    ) -> ReferenceRootHandle {
        let mut environment = self.environment.borrow_mut();
        environment.roots.push(Some(ReferenceRootEntry { r#type, state }));
        ReferenceRootHandle { environment: environment.id, index: environment.roots.len() - 1 }
    }

    /// Binds one fresh whole-root carrier from an already validated environment state.
    fn bind_root_value(
        &self,
        r#type: ReferenceType<P::Referent>,
        state: ReferenceRootState<C::Value>,
        preserved: Option<C::Value>,
    ) -> ReferenceDischargeValue<C, P> {
        let alias = P::root_alias(r#type.referent());
        let root = self.bind_root(r#type.clone(), state);
        ReferenceDischargeValue::Reference(ReferenceDischargeReference {
            root,
            denotes_whole_root: true,
            alias,
            r#type,
            preserved,
        })
    }

    /// Installs one live discharged root's successor state and merges its mutation fact.
    fn install_discharged_state(
        &self,
        root: ReferenceRootHandle,
        current: C::Value,
        mutated: bool,
    ) -> Result<(), ProgramError> {
        self.validate_discharged_state_type(root, &current)?;
        let mut environment = self.environment.borrow_mut();
        environment.slot(root)?;
        match environment.roots[root.index].as_mut().map(|entry| &mut entry.state) {
            Some(ReferenceRootState::Discharged { current: state, mutated: previous_mutated }) => {
                *state = current;
                *previous_mutated |= mutated;
                Ok(())
            }
            Some(ReferenceRootState::Preserved { .. }) => Err(ProgramError::MalformedProgram(format!(
                "reference discharge installed state into preserved {root}",
            ))),
            None => Err(ProgramError::MalformedProgram(format!("reference discharge accessed consumed {root}"))),
        }
    }
}

impl<C: Clone + Domain, P: ReferenceDischargePolicy<C>> Clone for ReferenceDischargeContext<C, P> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            parent: self.parent.clone(),
            environment: Rc::clone(&self.environment),
            captures: self.captures.clone(),
            selection: self.selection.clone(),
        }
    }
}

impl<C: Domain, P: ReferenceDischargePolicy<C>> Debug for ReferenceDischargeContext<C, P> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let live_root_count = self.environment.borrow().roots.iter().filter(|root| root.is_some()).count();
        formatter
            .debug_struct("ReferenceDischargeContext")
            .field("live_roots", &live_root_count)
            .finish_non_exhaustive()
    }
}

/// Destination one structured reference discharge rule rebuilds a nested [`Region`](crate::Region) into: a fresh trace
/// of the same program universe, which seals into a program the rule attaches to its rewritten operation.
///
/// It is deliberately a fresh root trace rather than a nested trace of the live destination. A rebuilt region is a
/// self-contained artifact whose complete interface is its own boundary, so it must not close over any value of the
/// destination it will be attached in. Being a root trace is also what makes the type a fixed point of its own
/// construction — the destination of a destination is that same destination — which is what keeps the obligation that
/// this universe's operations discharge into it finite.
pub type ReferenceDischargeRegionDestination<C> = TracingContext<<C as Domain>::Constant, <C as Domain>::Operation>;

/// Boundary a structured reference discharge rule requests for one rebuilt region.
///
/// The rule owns the mapping from its own operands onto a region's declared inputs, because that mapping is part of
/// what the operation *is*. It therefore describes the declared input boundary itself, in region order, and names
/// separately the threaded state the rebuilt region gains: which roots enter, which roots it must publish, and where
/// each group is inserted.
#[derive(Clone, Debug, PartialEq)]
pub struct ReferenceRegionDischargeBoundary {
    /// Root entering at each declared source-region input position, or [`None`] for an ordinary value.
    declared_input_roots: Vec<Option<ReferenceRootHandle>>,

    /// Length of the region's own leading capture prefix, from [`Operation::region_capture_input_count`], or [`None`]
    /// when the region inherits the capture scope of the region its operation is applied in.
    capture_input_count: Option<usize>,

    /// Roots whose entering carrier the rebuilt region receives as added inputs, in canonical root order.
    /// Discharged roots enter as immutable state; preserved roots enter as their destination reference value.
    added_input_roots: Vec<ReferenceRootHandle>,

    /// Position in the source region's input boundary at which the added state inputs are inserted.
    state_input_insertion: usize,

    /// Roots whose final state the rebuilt region publishes as added outputs, in canonical root order.
    added_state_output_roots: Vec<ReferenceRootHandle>,

    /// Position in the source region's output boundary at which the added state outputs are inserted.
    state_output_insertion: usize,
}

impl ReferenceRegionDischargeBoundary {
    /// Creates a rebuilt-region boundary request.
    ///
    /// Added state is described separately from the declared positions because only the declared positions are
    /// replayed: an added input exists in the rebuilt region's destination boundary and in the caller's operand list,
    /// but the source region's body never named it and therefore cannot consume it.
    ///
    /// The region's capture prefix is read off the operation rather than supplied, so that a rule cannot state one
    /// prefix here and let [`ReferenceDischargeContext::region_summary`] derive a different one from the same hook.
    /// A rule therefore never reasons about captures at all.
    ///
    /// # Parameters
    ///
    ///   - `operation`: Operation the region is attached to, whose
    ///     [`region_capture_input_count`](Operation::region_capture_input_count) states the region's own leading
    ///     capture prefix.
    ///   - `region_index`: Position of the region among that operation's attached regions.
    ///   - `declared_input_roots`: Root entering at each declared boundary position, or [`None`] for an ordinary value.
    ///     Reference positions must come from [`ReferenceDischargeContext::operand_root`], which validates that each
    ///     operand carries the whole root rather than a derived view. Its length must equal the source region's input
    ///     count, because every declared position is rebuilt.
    ///   - `added_input_roots`: Roots whose entering state or preserved reference the rebuilt region receives as added
    ///     inputs.
    ///   - `state_input_insertion`: Position in the source input boundary receiving those added inputs.
    ///   - `added_state_output_roots`: Roots whose final state the rebuilt region publishes as added outputs.
    ///   - `state_output_insertion`: Position in the source output boundary receiving those added outputs.
    #[inline]
    pub fn new<O: Operation>(
        operation: &O,
        region_index: usize,
        declared_input_roots: Vec<Option<ReferenceRootHandle>>,
        added_input_roots: Vec<ReferenceRootHandle>,
        state_input_insertion: usize,
        added_state_output_roots: Vec<ReferenceRootHandle>,
        state_output_insertion: usize,
    ) -> Self {
        Self {
            declared_input_roots,
            capture_input_count: operation.region_capture_input_count(region_index),
            added_input_roots,
            state_input_insertion,
            added_state_output_roots,
            state_output_insertion,
        }
    }

    /// Returns the root entering at each declared boundary position, or [`None`] for an ordinary value.
    #[inline]
    pub fn declared_input_roots(&self) -> &[Option<ReferenceRootHandle>] {
        self.declared_input_roots.as_slice()
    }

    /// Returns the region's own leading capture prefix, or [`None`] when it inherits its caller's capture scope.
    #[inline]
    pub const fn capture_input_count(&self) -> Option<usize> {
        self.capture_input_count
    }

    /// Returns the roots whose entering state or preserved reference the rebuilt region receives as added inputs.
    #[inline]
    pub fn added_input_roots(&self) -> &[ReferenceRootHandle] {
        self.added_input_roots.as_slice()
    }

    /// Returns the source input position at which the added state inputs are inserted.
    #[inline]
    pub const fn state_input_insertion(&self) -> usize {
        self.state_input_insertion
    }

    /// Returns the roots whose final state the rebuilt region publishes as added outputs.
    #[inline]
    pub fn added_state_output_roots(&self) -> &[ReferenceRootHandle] {
        self.added_state_output_roots.as_slice()
    }

    /// Returns the source output position at which the added state outputs are inserted.
    #[inline]
    pub const fn state_output_insertion(&self) -> usize {
        self.state_output_insertion
    }
}

/// Sealed result of discharging one attached region against an isolated environment.
///
/// This is the transactional artifact of a structured rule's region fork, and it deliberately carries no values of
/// any kind. A discharge tracer produced inside the fork would keep addressing the fork's own abandoned environment,
/// and even a plain destination value is not detached under a staging destination, because it is itself a tracer
/// stamped with the fork's builder. Excluding both structurally is what makes the isolation a type-level fact rather
/// than a convention: the owning rule binds the rebuilt operation in its *own* context and merges the final states
/// from the outputs that binding produced.
#[derive(Debug)]
pub struct ReferenceRegionDischargeFork<V: Value, O: Operation<Type = V::Type>> {
    /// Rebuilt, discharged region program.
    program: Program<V, O, Vec<V>, Vec<V>>,

    /// Root each *declared* region output denotes, or [`None`] for an ordinary output, in region-boundary order.
    output_roots: Vec<Option<ReferenceRootHandle>>,

    /// Threaded roots the region's closure mutated, in canonical root order.
    mutated_roots: Vec<ReferenceRootHandle>,
}

impl<V: Value, O: Operation<Type = V::Type>> ReferenceRegionDischargeFork<V, O> {
    /// Returns the root each declared region output denotes, or [`None`] where the output is an ordinary value.
    #[inline]
    pub fn output_roots(&self) -> &[Option<ReferenceRootHandle>] {
        self.output_roots.as_slice()
    }

    /// Consumes this fork and returns the rebuilt region program.
    #[inline]
    pub fn into_program(self) -> Program<V, O, Vec<V>, Vec<V>> {
        self.program
    }

    /// Validates that this region's declared outputs denote exactly the roots the widening that sized its boundary
    /// expected them to.
    ///
    /// The widening reads the declared output roots from a *static* summary, and the boundary it sizes depends on
    /// them: a root a region already returns publishes its final state at that output's own position and must not be
    /// published a second time. This is where that prediction is held to what the replay actually produced, so a rule
    /// whose operation disagrees with its own generic hooks is reported instead of silently losing an update. It also
    /// makes the several regions of one operation agree with each other, because they are all checked against the one
    /// summary that sized their shared boundary.
    ///
    /// # Parameters
    ///
    ///   - `expected`: Root each declared output was predicted to denote, in region-boundary order.
    ///   - `operation`: Name of the operation being rewritten, used in the diagnostic.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when this region's declared outputs denote different roots.
    pub fn validate_predicted_output_roots(
        &self,
        expected: &[Option<ReferenceRootHandle>],
        operation: &str,
    ) -> Result<(), ProgramError> {
        if self.output_roots != expected {
            return Err(ProgramError::MalformedProgram(format!(
                "operation `{operation}` attaches a region whose outputs do not denote the references its state \
                 widening expected",
            )));
        }
        Ok(())
    }

    /// Validates that this region mutated no root the widening that sized its boundary did not publish.
    ///
    /// The boundary was sized from a summary computed before the region ran, so this is where the summary and the
    /// replay are held to each other. A mismatch means one of the generic hooks the summary follows under-reports what
    /// its operation does, and reporting it here is what keeps that from surfacing later as a lost update.
    ///
    /// # Parameters
    ///
    ///   - `published`: Roots whose final state the widening decided this region publishes.
    ///   - `operation`: Name of the operation being rewritten, used in the diagnostic.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] naming the first root this region mutated that `published` does not
    /// contain.
    pub fn validate_predicted_mutations(
        &self,
        published: &[ReferenceRootHandle],
        operation: &str,
    ) -> Result<(), ProgramError> {
        for root in &self.mutated_roots {
            if !published.contains(root) {
                return Err(ProgramError::MalformedProgram(format!(
                    "operation `{operation}` mutated {root} in an attached region that its state widening did not \
                     predict",
                )));
            }
        }
        Ok(())
    }
}

/// Exact non-consuming access modes recorded for one caller root.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
struct ReferenceAccessModes {
    /// Whether the root is read.
    read: bool,

    /// Whether the root is written without observing its selected previous state.
    write: bool,

    /// Whether the root is read and replaced.
    read_write: bool,

    /// Whether the root receives an ordered additive update.
    accumulate: bool,
}

impl ReferenceAccessModes {
    /// Records one non-consuming mode.
    fn insert(&mut self, mode: ReferenceAccessMode) {
        match mode {
            ReferenceAccessMode::Read => self.read = true,
            ReferenceAccessMode::Write => self.write = true,
            ReferenceAccessMode::ReadWrite => self.read_write = true,
            ReferenceAccessMode::Accumulate => self.accumulate = true,
            ReferenceAccessMode::Consume => unreachable!("consuming accesses are rejected before summary insertion"),
        }
    }

    /// Returns the recorded modes in semantic order.
    fn iter(self) -> impl Iterator<Item = ReferenceAccessMode> {
        [
            self.read.then_some(ReferenceAccessMode::Read),
            self.write.then_some(ReferenceAccessMode::Write),
            self.read_write.then_some(ReferenceAccessMode::ReadWrite),
            self.accumulate.then_some(ReferenceAccessMode::Accumulate),
        ]
        .into_iter()
        .flatten()
    }
}

/// Transitive reference-access summary of one region closure, expressed in the caller roots its boundary names.
///
/// This is the analysis a structured rule needs *before* it can size its state boundary, and it is computed entirely
/// from generic hooks: operation-local [`Operation::reference_semantics`], the input- and output-region provenance
/// hooks, reference-output identity, and recursive summaries of nested regions. Roots allocated inside the closure are
/// deliberately absent: they belong to no caller and cross no boundary.
///
/// The summary separates *reachability* from *semantic access*. [`reached`](Self::reached) holds every caller root
/// the closure's replay must be able to resolve — including a capture constant that is only rematerialized and passed
/// along — and is what sizes the state boundary through
/// [`threaded_state_roots`](ReferenceDischargeContext::threaded_state_roots). [`accessed`](Self::accessed) and
/// [`access_modes`](Self::access_modes) hold only the roots the closure semantically accesses, which is what region
/// access policies validate. Reading `accessed` to size a boundary under-threads merely-forwarded captures.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReferenceRegionSummary {
    /// Every caller root the closure must be able to resolve while replaying, whether or not it is semantically
    /// accessed.
    reached: BTreeSet<ReferenceRootHandle>,

    /// Every caller root the closure accesses, mapped to its exact non-consuming access modes.
    accesses: BTreeMap<ReferenceRootHandle, ReferenceAccessModes>,

    /// Caller root each *declared* region output denotes, or [`None`] where the output is an ordinary value.
    output_roots: Vec<Option<ReferenceRootHandle>>,
}

impl ReferenceRegionSummary {
    /// Returns every caller root the closure must be able to resolve, in canonical root order.
    #[inline]
    fn reached(&self) -> impl Iterator<Item = ReferenceRootHandle> + '_ {
        self.reached.iter().copied()
    }

    /// Returns every caller root the closure accesses, in canonical root order.
    #[inline]
    pub fn accessed(&self) -> impl Iterator<Item = ReferenceRootHandle> + '_ {
        self.accesses.keys().copied()
    }

    /// Returns the exact access modes recorded for `root`, in semantic order.
    pub fn access_modes(&self, root: ReferenceRootHandle) -> impl Iterator<Item = ReferenceAccessMode> {
        self.accesses.get(&root).copied().unwrap_or_default().iter()
    }

    /// Returns whether `mode` is among the closure's recorded access modes for `root`.
    #[inline]
    pub fn has_access(&self, root: ReferenceRootHandle, mode: ReferenceAccessMode) -> bool {
        self.access_modes(root).any(|recorded| recorded == mode)
    }

    /// Returns the caller root each declared region output denotes, or [`None`] where the output is an ordinary
    /// value.
    ///
    /// A region that returns a root already publishes that root's final state at its own output position, so a rule
    /// that widens the boundary must not publish it a second time.
    #[inline]
    pub fn output_roots(&self) -> &[Option<ReferenceRootHandle>] {
        self.output_roots.as_slice()
    }

    /// Returns whether any statically reachable path through the closure writes or accumulates into `root`. A root the
    /// closure only reads is *not* mutated, which is the fact read-only pruning consults.
    ///
    /// This classification is intentionally conservative across structured control flow: a write in either branch or
    /// in a loop body marks the root as mutated even when one execution takes the other branch or performs zero
    /// iterations. Discharge therefore threads and publishes a hidden final state for every such root; at runtime that
    /// state is simply unchanged when the mutating path does not execute.
    #[inline]
    pub fn is_mutated(&self, root: ReferenceRootHandle) -> bool {
        self.access_modes(root).any(|mode| {
            matches!(
                mode,
                ReferenceAccessMode::Write | ReferenceAccessMode::ReadWrite | ReferenceAccessMode::Accumulate,
            )
        })
    }

    /// Returns the summary of the two closures taken together, which is what an operation with several attached
    /// regions threads through one shared *state* boundary.
    ///
    /// The reached roots and the accesses are both merged, so a root that only one nested closure returns or
    /// rematerializes stays reachable — and therefore threaded — at the merged level. Declared output roots belong to
    /// one region's own boundary rather than to the shared state, so the merged summary keeps the receiver's; an
    /// operation whose regions must agree on them, such as a condition, has that agreement checked against the
    /// rebuilt regions themselves.
    pub fn merged(mut self, other: &Self) -> Self {
        self.absorb(other);
        self
    }

    /// Merges another closure's reached roots and accesses into this summary in place, leaving the declared output
    /// roots alone.
    fn absorb(&mut self, other: &Self) {
        self.reached.extend(other.reached.iter().copied());
        for (root, modes) in &other.accesses {
            let entry = self.accesses.entry(*root).or_default();
            for mode in modes.iter() {
                entry.insert(mode);
            }
        }
    }

    /// Records one access, or rejects a consuming access to a caller root.
    ///
    /// # Parameters
    ///
    ///   - `root`: Caller root being accessed.
    ///   - `mode`: Semantic mode of the access.
    ///   - `operation`: Name of the accessing operation, used in the consumption diagnostic.
    fn record(
        &mut self,
        root: ReferenceRootHandle,
        mode: ReferenceAccessMode,
        operation: &str,
    ) -> Result<(), ProgramError> {
        // A consumed root has no successor, so no symmetric boundary and no final-state output can describe what
        // happened to it, and a root that survives as a reference fares no better: whether a region consumed it can
        // depend on which branch ran, which the caller's environment cannot represent.
        if mode.is_consuming() {
            return Err(ProgramError::MalformedProgram(format!(
                "reference discharge cannot pass {root} into a region that consumes it through `{operation}`",
            )));
        }
        self.reached.insert(root);
        self.accesses.entry(root).or_default().insert(mode);
        Ok(())
    }
}

/// Provides one [`Operation`] application with its replay position and with recursive discharge of the
/// [`Region`](crate::Region)s attached to it.
///
/// [`RegionDriver`] supplies the structural region access, and this trait adds the three services that discharge
/// rules need on top of it. Region-free applications expose a region count of zero through the same contract.
pub trait ReferenceDischargeDriver<C: Domain, P: ReferenceDischargePolicy<C>>:
    RegionDriver<C::Constant, C::Operation>
{
    /// Returns the source coordinate of the operation application being discharged, or [`None`] when the application
    /// did not come from a replayed instruction.
    ///
    /// An allocation rule needs its own site to decide whether the caller selected it for discharge, so replaying a
    /// region through [`discharge_region`](Self::discharge_region) must supply the coordinate of every instruction it
    /// replays. Returning [`None`] declares the allocation unnameable by any [`ReferenceDischargeSite`] and therefore
    /// *always discharged*, silently ignoring the caller's partial selection. This method is deliberately required
    /// rather than defaulted: a replaying driver that forgot to forward its coordinate would otherwise disable
    /// partial discharge for its regions without any diagnostic.
    fn instruction(&self) -> Option<InstructionId>;

    /// Discharges the region at `index` over the provided carriers by re-entering the active discharge transform,
    /// binding the region's rewritten work directly into the destination program.
    ///
    /// The region is inlined under the *caller's* capture scope, which is correct for every region that inherits one.
    /// A region declaring its own leading capture prefix has to be rebuilt instead, through
    /// [`discharge_region_program`](Self::discharge_region_program), which establishes that prefix as the rebuilt
    /// region's own scope.
    ///
    /// # Parameters
    ///
    ///   - `context`: Active discharge context whose environment the replayed region observes and mutates.
    ///   - `index`: Position of the attached region in operation-defined order.
    ///   - `inputs`: Carriers supplied to the region's boundary, in boundary order.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when this application has no region at `index` or when `inputs` does
    /// not describe the region's boundary, and propagates every failure the replayed rules raise.
    fn discharge_region(
        &self,
        context: &ReferenceDischargeContext<C, P>,
        index: usize,
        inputs: Vec<ReferenceDischargeValue<C, P>>,
    ) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError>;

    /// Discharges the region at `index` against an *isolated* environment over a fresh destination of the same
    /// universe, and returns the sealed [`ReferenceRegionDischargeFork`] describing what that rebuilt region became.
    ///
    /// This is the transactional fork every structured rule builds on, and it is what
    /// [`discharge_region`](Self::discharge_region) is deliberately not: that service inlines a region's rewritten
    /// work into the live destination, which is right for an operation whose region is invoked in place and wrong for
    /// one whose region must survive as a region. The fork's environment contains exactly the roots `boundary` names,
    /// each entering as an ordinary value at its boundary position, so a region cannot reach a root its caller did not
    /// thread, and nothing it does can reach the caller's environment. The owning rule binds the rebuilt operation in
    /// its own context and merges the final states from the outputs of that binding.
    ///
    /// # Parameters
    ///
    ///   - `context`: Active discharge context supplying the entering state, or the surviving reference, of every
    ///     root the boundary names.
    ///   - `index`: Position of the attached region in operation-defined order.
    ///   - `boundary`: Complete requested boundary of the rebuilt region.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when this application has no region at `index`, when `boundary` does
    /// not describe that region's declared boundary, when a root is threaded twice, when the region publishes a root
    /// its caller did not thread or publishes one through a derived view, and propagates every failure the rebuilt
    /// region's own rules raise.
    fn discharge_region_program(
        &self,
        context: &ReferenceDischargeContext<C, P>,
        index: usize,
        boundary: &ReferenceRegionDischargeBoundary,
    ) -> Result<ReferenceRegionDischargeFork<C::Constant, C::Operation>, ProgramError>;
}

impl<C: Domain, P: ReferenceDischargePolicy<C>> ReferenceDischargeDriver<C, P> for EmptyRegionDriver {
    // A region-free application replays no instruction, so its allocations carry no selectable coordinate.
    #[inline]
    fn instruction(&self) -> Option<InstructionId> {
        None
    }

    #[inline]
    fn discharge_region(
        &self,
        _context: &ReferenceDischargeContext<C, P>,
        _index: usize,
        _inputs: Vec<ReferenceDischargeValue<C, P>>,
    ) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError> {
        Err(ProgramError::MalformedProgram("empty region driver cannot discharge a region".to_string()))
    }

    #[inline]
    fn discharge_region_program(
        &self,
        _context: &ReferenceDischargeContext<C, P>,
        _index: usize,
        _boundary: &ReferenceRegionDischargeBoundary,
    ) -> Result<ReferenceRegionDischargeFork<C::Constant, C::Operation>, ProgramError> {
        Err(ProgramError::MalformedProgram("empty region driver cannot rebuild a region".to_string()))
    }
}

/// [`ReferenceDischargeDriver`] scoped to one [`Operation`] application. It borrows the application's complete region
/// driver, which preserves the operation-defined ordering of owned regions, borrowed regions, and shared callees
/// without materializing a combined region collection.
pub struct RecursiveReferenceDischargeDriver<'r, D> {
    /// Application-scoped [`RegionDriver`].
    driver: &'r D,

    /// Source coordinate of the application, or [`None`] for a direct bind.
    instruction: Option<InstructionId>,
}

impl<'r, D> RecursiveReferenceDischargeDriver<'r, D> {
    /// Creates a new [`RecursiveReferenceDischargeDriver`].
    ///
    /// # Parameters
    ///
    ///   - `driver`: Application-scoped [`RegionDriver`] exposing the attached regions.
    ///   - `instruction`: Source coordinate of the application, or [`None`] for a direct bind.
    #[inline]
    pub const fn new(driver: &'r D, instruction: Option<InstructionId>) -> Self {
        Self { driver, instruction }
    }
}

impl<V: Value, O: Operation<Type = V::Type>, D: RegionDriver<V, O>> RegionDriver<V, O>
    for RecursiveReferenceDischargeDriver<'_, D>
{
    #[inline]
    fn regions<'r>(&'r self) -> impl Iterator<Item = RegionRef<'r, V, O>>
    where
        V: 'r,
        O: 'r,
    {
        self.driver.regions()
    }
}

// Recursive discharge replays the attached region one instruction at a time against the live environment, so a root
// created outside the region stays the same root inside it and the region's own allocations are ordinary new roots.
// Constants lift into the destination through the parent, exactly as they do at the top level.
//
// The nested obligation is the one this crate's other structural transforms already carry: rebuilding a region needs
// this universe's operations to discharge into a fresh trace of the same universe as well as into the live
// destination. The requested reference type of a threaded root crosses that boundary, so the two policy
// instantiations must agree on their referent type system. Both obligations are stated here rather than on the
// per-operation rules on purpose. A rule that stated them would make the enum dispatcher's obligation graph circular,
// because the dispatcher's own predicate for a structured payload would then demand that the whole enum discharge
// into the destination whose dischargeability is what the graph is trying to establish.
impl<C, P, D> ReferenceDischargeDriver<C, P> for RecursiveReferenceDischargeDriver<'_, D>
where
    C: Context<
        Operation: ReferenceDischargeableOperation<C, P>
                       + ReferenceDischargeableOperation<ReferenceDischargeRegionDestination<C>, P>,
    >,
    P: ReferenceDischargePolicy<C>
        + ReferenceDischargePolicy<
            ReferenceDischargeRegionDestination<C>,
            Referent = <P as ReferenceDischargePolicy<C>>::Referent,
        >,
    D: RegionDriver<C::Constant, C::Operation>,
{
    #[inline]
    fn instruction(&self) -> Option<InstructionId> {
        self.instruction
    }

    #[inline]
    fn discharge_region(
        &self,
        context: &ReferenceDischargeContext<C, P>,
        index: usize,
        inputs: Vec<ReferenceDischargeValue<C, P>>,
    ) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError> {
        discharge_region_instructions(context, self.region(index)?, inputs)
    }

    #[inline]
    fn discharge_region_program(
        &self,
        context: &ReferenceDischargeContext<C, P>,
        index: usize,
        boundary: &ReferenceRegionDischargeBoundary,
    ) -> Result<ReferenceRegionDischargeFork<C::Constant, C::Operation>, ProgramError> {
        rebuild_discharged_region(context, self.region(index)?, boundary)
    }
}

/// Replays one region's instructions against the live environment of `context`, binding their rewritten work through
/// the destination that context already owns.
///
/// This is the inlining replay both [`ReferenceDischargeDriver::discharge_region`] and the region fork use: the fork
/// differs only in which context it hands over, not in how a region's instructions are discharged.
///
/// # Parameters
///
///   - `context`: Active discharge context whose environment the replay observes and mutates.
///   - `region`: Source region being replayed.
///   - `inputs`: Carriers supplied to the region's boundary, in boundary order.
fn discharge_region_instructions<C, P>(
    context: &ReferenceDischargeContext<C, P>,
    region: RegionRef<'_, C::Constant, C::Operation>,
    inputs: Vec<ReferenceDischargeValue<C, P>>,
) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError>
where
    C: Context<
        Operation: ReferenceDischargeableOperation<C, P>
                       + ReferenceDischargeableOperation<ReferenceDischargeRegionDestination<C>, P>,
    >,
    P: ReferenceDischargePolicy<C>
        + ReferenceDischargePolicy<
            ReferenceDischargeRegionDestination<C>,
            Referent = <P as ReferenceDischargePolicy<C>>::Referent,
        >,
{
    let mappings = RegionReplayMappings::new();
    let mut instruction_index = 0;
    region.interpret_with(
        inputs,
        |_, constant| lift_constant::<C, P>(context, constant.clone()),
        |instruction, instruction_inputs| {
            let regions = ReplayRegionDriver::new(region, instruction.regions(), &mappings)?;
            let position = InstructionId::new(region.id(), instruction_index);
            instruction_index += 1;
            let driver = RecursiveReferenceDischargeDriver::new(&regions, Some(position));
            // Run the discharge rule inside the source instruction's recorded origin so every staged instruction
            // records where it came from. Rules stage their rewritten work through the destination parent, which is
            // where the provenance state lives.
            context.parent().invoke_with_provenance_origin(instruction.provenance().clone(), || {
                instruction.operation().discharge_references(context, &driver, instruction_inputs)
            })
        },
    )
}

/// Discharges `region` against an isolated environment over a fresh destination and seals the result.
///
/// The fork's environment is built rather than copied: it holds exactly the roots `boundary` names, each entering as
/// an ordinary destination value at its own boundary position. Because the fork mints its own environment identity,
/// a handle from the caller cannot address a fork root and a handle from the fork cannot address a caller root — a
/// leak in either direction is reported instead of silently aliasing.
///
/// # Parameters
///
///   - `context`: Active discharge context supplying the entering state, or the surviving reference, of every root
///     the boundary names.
///   - `region`: Source region being rebuilt.
///   - `boundary`: Complete requested boundary of the rebuilt region.
fn rebuild_discharged_region<C, P>(
    context: &ReferenceDischargeContext<C, P>,
    region: RegionRef<'_, C::Constant, C::Operation>,
    boundary: &ReferenceRegionDischargeBoundary,
) -> Result<ReferenceRegionDischargeFork<C::Constant, C::Operation>, ProgramError>
where
    C: Context<Operation: ReferenceDischargeableOperation<ReferenceDischargeRegionDestination<C>, P>>,
    P: ReferenceDischargePolicy<C>
        + ReferenceDischargePolicy<
            ReferenceDischargeRegionDestination<C>,
            Referent = <P as ReferenceDischargePolicy<C>>::Referent,
        >,
{
    type Destination<C> = ReferenceDischargeRegionDestination<C>;

    check_count!("input", boundary.declared_input_roots(), region.input_ids().len(), ProgramError);
    let source_input_types = region.input_types();
    let source_input_count = source_input_types.len();
    let source_output_count = region.output_ids().len();
    if boundary.state_input_insertion() > source_input_count {
        return Err(ProgramError::MalformedProgram(format!(
            "reference discharge inserts region state inputs at {} but region `{}` declares {source_input_count} \
             inputs",
            boundary.state_input_insertion(),
            region.id(),
        )));
    }
    if boundary.state_output_insertion() > source_output_count {
        return Err(ProgramError::MalformedProgram(format!(
            "reference discharge inserts region state outputs at {} but region `{}` declares {source_output_count} \
             outputs",
            boundary.state_output_insertion(),
            region.id(),
        )));
    }

    // Added state may not land inside the region's own capture prefix: the rebuilt region keeps the prefix length its
    // operation declares, so a state input placed before the end of it would silently renumber the captures the
    // rebound operation still names.
    let capture_input_count = boundary.capture_input_count().unwrap_or(0);
    if boundary.state_input_insertion() < capture_input_count {
        return Err(ProgramError::MalformedProgram(format!(
            "reference discharge inserts region state inputs at {} but region `{}` declares a capture prefix of \
             {capture_input_count}",
            boundary.state_input_insertion(),
            region.id(),
        )));
    }

    // Every carrier, the fork's context, and the destination itself stay inside this block, because recovering the
    // rebuilt program below requires unique ownership of the destination's builder.
    let destination = Destination::<C>::new();
    let builder = destination.builder().clone();
    let (output_ids, output_roots, mutated_roots) = {
        // The fork inherits its caller's selection, because a site names a source coordinate that means the same thing
        // wherever the replay reaches it: an unselected allocation inside a rebuilt region survives there exactly as
        // it would have in the caller's own body.
        let fork = ReferenceDischargeContext::<Destination<C>, P>::new_selecting(
            destination.clone(),
            context.selection.clone(),
        );

        let mut declared_roots = BTreeSet::new();
        declared_roots.extend(boundary.declared_input_roots().iter().copied().flatten());
        let mut added_roots = BTreeSet::new();
        for root in boundary.added_input_roots() {
            if declared_roots.contains(root) || !added_roots.insert(*root) {
                return Err(ProgramError::MalformedProgram(format!(
                    "reference discharge adds {root} to region `{}` more than once",
                    region.id(),
                )));
            }
        }

        // Caller and fork roots live in different environments, so explicit directional maps are the only
        // correspondence between them. Repeated declared positions may intentionally alias one caller root, while
        // synthesized state positions were already proven unique (and disjoint from the declared roots) above. The
        // caller-to-fork map is ordered because the mutation-reconciliation loop below iterates it fallibly, and
        // diagnostics must not depend on hash order.
        let mut caller_to_fork = BTreeMap::<ReferenceRootHandle, ReferenceRootHandle>::new();
        let mut fork_to_caller = HashMap::<ReferenceRootHandle, ReferenceRootHandle>::new();
        let mut thread =
            |root: ReferenceRootHandle| -> Result<ReferenceDischargeValue<Destination<C>, P>, ProgramError> {
                let r#type = context.root_reference_type(root)?;
                let discharged = context.root_is_discharged(root)?;
                let input_type = if discharged {
                    <P as ReferenceDischargePolicy<Destination<C>>>::lift_referent_type(r#type.referent().clone())
                } else {
                    <P as ReferenceDischargePolicy<Destination<C>>>::lift_reference_type(r#type.clone())
                };
                let input = destination.input(input_type);
                if let Some(forked) = caller_to_fork.get(&root).copied() {
                    return fork.root_handle(forked);
                }
                let carrier = if discharged {
                    fork.allocate_discharged(r#type, input)?
                } else {
                    fork.bind_preserved(r#type, input)?
                };
                let forked = carrier.expect_reference("a threaded region root")?.root();
                caller_to_fork.insert(root, forked);
                fork_to_caller.insert(forked, root);
                Ok(carrier)
            };

        // Only the declared positions are replayed. An added input occupies a destination boundary position and a
        // caller operand position, but the source region's body never named it and so cannot consume it. A preserved
        // root occupies an added position only when an inherited capture is returned without a declared operand.
        let mut declared = Vec::with_capacity(source_input_count);
        for position in 0..=source_input_count {
            if position == boundary.state_input_insertion() {
                for root in boundary.added_input_roots() {
                    thread(*root)?;
                }
            }
            let Some(root) = boundary.declared_input_roots().get(position) else {
                continue;
            };
            let source_type = &source_input_types[position];
            declared.push(match root {
                None => {
                    if <P as ReferenceDischargePolicy<C>>::project_reference_type(source_type).is_some() {
                        return Err(ProgramError::MalformedProgram(format!(
                            "reference discharge declares reference input {position} of region `{}` without a root",
                            region.id(),
                        )));
                    }
                    ReferenceDischargeValue::Ordinary(destination.input(source_type.clone()))
                }
                Some(root) => {
                    let Some(source_reference_type) =
                        <P as ReferenceDischargePolicy<C>>::project_reference_type(source_type)
                    else {
                        return Err(ProgramError::MalformedProgram(format!(
                            "reference discharge assigns {root} to ordinary input {position} of region `{}`",
                            region.id(),
                        )));
                    };
                    let root_type = context.root_reference_type(*root)?;
                    if root_type != source_reference_type {
                        return Err(ProgramError::MalformedProgram(format!(
                            "reference discharge assigns {root} of type `{root_type}` to input {position} of region \
                             `{}` with reference type `{source_reference_type}`",
                            region.id(),
                        )));
                    }
                    thread(*root)?
                }
            });
        }

        // The rebuilt region discharges under a scope naming only fork roots, so the isolation the fork mints holds
        // for capture-scoped references too: a region declaring its own capture prefix reads that prefix off its
        // threaded declared inputs, and every other region inherits the caller's scope mapped onto the fork roots
        // standing for its caller roots. A caller root the boundary did not thread binds nothing. Discharged capture
        // accesses and outputs enter as state, while a preserved capture-scoped output enters as its destination
        // reference, so both states mint fork roots before the inherited scope is installed.
        let inherited = context.captures().with_roots(
            context
                .captures()
                .roots()
                .iter()
                .map(|root| root.and_then(|caller| caller_to_fork.get(&caller).copied()))
                .collect(),
        );
        let fork_declared_roots = declared
            .iter()
            .map(|input| match input {
                ReferenceDischargeValue::Ordinary(_) => None,
                ReferenceDischargeValue::Reference(reference) => Some(reference.root()),
            })
            .collect::<Vec<_>>();
        let fork = fork.with_captures(nested_capture_scope(
            boundary.capture_input_count(),
            fork_declared_roots.as_slice(),
            &inherited,
            region.id(),
        )?);

        let outputs = discharge_region_instructions(&fork, region, declared)?;
        check_count!("output", outputs, source_output_count, ProgramError);

        let mut output_ids = Vec::with_capacity(source_output_count + boundary.added_state_output_roots().len());
        let mut output_roots = Vec::with_capacity(source_output_count);
        for position in 0..=source_output_count {
            if position == boundary.state_output_insertion() {
                for root in boundary.added_state_output_roots() {
                    let forked = caller_to_fork.get(root).copied().ok_or_else(|| {
                        ProgramError::MalformedProgram(format!(
                            "reference discharge publishes {root} from region `{}` without threading it in",
                            region.id(),
                        ))
                    })?;
                    output_ids.push(fork.discharged_state(forked)?.atom_id()?);
                }
            }
            let Some(output) = outputs.get(position) else {
                continue;
            };
            match output {
                ReferenceDischargeValue::Ordinary(value) => {
                    output_roots.push(None);
                    output_ids.push(value.atom_id()?);
                }
                ReferenceDischargeValue::Reference(reference) => {
                    // A reference-typed region output publishes its root at that exact position — a discharged root's
                    // current state, a preserved root's own reference — and the owning rule maps it back onto the
                    // caller root through `output_roots`. A root the caller did not thread has nowhere to be
                    // published, which is how a region-local allocation is stopped from escaping through the boundary.
                    let caller = fork_to_caller.get(&reference.root()).copied().ok_or_else(|| {
                        ProgramError::MalformedProgram(format!(
                            "reference discharge cannot publish {} from region `{}`, whose caller did not thread \
                             that root",
                            reference.root(),
                            region.id(),
                        ))
                    })?;

                    // The published value denotes the whole root, so only a handle with whole-root provenance may
                    // cross. A view returned from a region has to be re-derived by whoever needs it, exactly as one
                    // passed into a region does.
                    let whole = fork.root_reference_type(reference.root())?;
                    if !reference.denotes_whole_root() {
                        return Err(ProgramError::MalformedProgram(format!(
                            "reference discharge cannot publish the derived view `{}` of {caller} from region `{}`, \
                             whose boundary carries the whole root `{whole}`",
                            reference.r#type(),
                            region.id(),
                        )));
                    }
                    output_roots.push(Some(caller));
                    output_ids.push(match reference.preserved() {
                        Some(value) => value.atom_id()?,
                        None => fork.discharged_state(reference.root())?.atom_id()?,
                    });
                }
            }
        }

        // Only threaded *state* can have been mutated. A preserved root's writes replayed into the rebuilt region as
        // the operations the source performed, so there is no successor state for the caller to merge.
        let mut mutated_roots = BTreeSet::new();
        for (caller, forked) in &caller_to_fork {
            if fork.root_is_discharged(*forked)? && fork.is_mutated(*forked)? {
                mutated_roots.insert(*caller);
            }
        }
        let mutated_roots = mutated_roots.into_iter().collect::<Vec<_>>();
        (output_ids, output_roots, mutated_roots)
    };
    drop(destination);

    let input_count = source_input_count + boundary.added_input_roots().len();
    let output_count = output_ids.len();
    let builder = Rc::try_unwrap(builder).map_err(|_| ProgramError::EscapedProgramBuilder)?.into_inner();
    let program = builder.build(output_ids, vec![Placeholder; input_count], vec![Placeholder; output_count])?;
    Ok(ReferenceRegionDischargeFork { program, output_roots, mutated_roots })
}

/// Returns the capture scope one attached region discharges under.
///
/// A region whose operation declares a leading capture prefix establishes a fresh scope over the roots that prefix
/// binds, exactly as a called program's captures shadow its caller's; every other region inherits the scope of the
/// region it is attached in. This is the interpreter's counterpart of the scope propagation the standalone reference
/// analysis performs over the whole arena, computed one boundary at a time because that is where the interpreter
/// already resolves roots.
///
/// # Parameters
///
///   - `capture_input_count`: Length of the region's own leading capture prefix, from
///     [`Operation::region_capture_input_count`], or [`None`] when the region inherits its parent's scope.
///   - `inputs`: Root each declared region input binds, in boundary order.
///   - `inherited`: Capture scope of the region this one is attached in.
///   - `region`: Identity of the region, used in the diagnostic.
///
/// # Errors
///
/// Returns [`ProgramError::MalformedProgram`] when the declared capture prefix is longer than the region's boundary.
fn nested_capture_scope<Constant>(
    capture_input_count: Option<usize>,
    inputs: &[Option<ReferenceRootHandle>],
    inherited: &ReferenceCaptureScope<Constant>,
    region: RegionId,
) -> Result<ReferenceCaptureScope<Constant>, ProgramError> {
    let Some(count) = capture_input_count else {
        return Ok(inherited.clone());
    };
    if count > inputs.len() {
        return Err(ProgramError::MalformedProgram(format!(
            "reference discharge cannot establish a capture prefix of {count} for region `{region}`, which declares \
             {} inputs",
            inputs.len(),
        )));
    }
    Ok(inherited.with_roots(inputs[..count].to_vec()))
}

/// Accumulates the transitive reference accesses of one region closure into `summary` and returns the caller root
/// each of the region's declared outputs denotes.
///
/// The traversal maps each reference-typed atom of the region onto the caller root it denotes, or onto [`None`] when
/// the root was allocated inside the closure and therefore crosses no boundary. Nested regions are entered through
/// [`Operation::input_region_provenance`], and a structured operation's reference-typed output is resolved either by
/// [`Operation::reference_output_identity_input`], which states outright which operand's root it preserves, or by
/// [`Operation::output_region_provenance`], which names the region output it forwards.
///
/// A reference-typed *constant* is resolved through `captures` and seeded exactly like a boundary position, because a
/// capture-lifted program names its caller's references that way. That is what lets a structured rule discover that
/// its closure reaches a root its operands never named, and therefore what makes synthesized state carries reachable.
///
/// # Parameters
///
///   - `region`: Region whose closure is summarized.
///   - `inputs`: Caller root denoted by each declared region input, in boundary order.
///   - `captures`: Capture scope this region discharges under.
///   - `summary`: Summary being accumulated.
///
/// # Errors
///
/// Returns [`ProgramError::MalformedProgram`] when `inputs` does not describe the region's boundary, when the closure
/// reaches a reference that entered neither through the boundary nor through the capture scope, when a nested
/// reference-typed boundary position declares no provenance to follow, when an operation's own contract forbids the
/// access mode its closure performs, or when the closure consumes a caller root.
fn summarize_region_closure<V: Value, O: Operation<Type = V::Type>>(
    region: RegionRef<'_, V, O>,
    inputs: &[Option<ReferenceRootHandle>],
    captures: &ReferenceCaptureScope<V>,
    summary: &mut ReferenceRegionSummary,
) -> Result<Vec<Option<ReferenceRootHandle>>, ProgramError> {
    check_count!("input", inputs, region.input_ids().len(), ProgramError);
    let is_reference = |atom: AtomId| region.atoms()[atom.index()].r#type().is_reference();
    let mut roots = HashMap::<AtomId, Option<ReferenceRootHandle>>::new();
    for (input, root) in region.input_ids().iter().copied().zip(inputs) {
        if is_reference(input) {
            roots.insert(input, *root);
        }
    }

    // A capture-scoped constant is seeded exactly like a boundary position. Materializing one makes its root reachable
    // during replay but is not itself a semantic reference read; actual accesses are recorded from operation semantics
    // below.
    let materialized_atoms = region
        .instructions()
        .iter()
        .flat_map(|instruction| instruction.inputs().iter().copied())
        .chain(region.output_ids().iter().copied())
        .collect::<HashSet<_>>();
    for (atom_index, atom) in region.atoms().iter().enumerate() {
        let atom_id = AtomId::new(atom_index);
        if let Atom::Constant(constant) = atom
            && constant.r#type().is_reference()
            && let Some(root) = captures.resolve(constant)
        {
            roots.insert(atom_id, Some(root));
            if materialized_atoms.contains(&atom_id) {
                summary.reached.insert(root);
            }
        }
    }
    let operand = |instruction: &Instruction<O>, index: usize, role: &str| {
        instruction.inputs().get(index).copied().ok_or_else(|| {
            ProgramError::MalformedProgram(format!(
                "operation `{}` names {role} operand {index} but the application has {} operands",
                instruction.operation().name(),
                instruction.inputs().len(),
            ))
        })
    };

    // A reference-typed atom the traversal never bound denotes a reference that entered this region neither through
    // its boundary nor through its capture scope. The environment has no root for it, so the summary reports it here
    // rather than dropping the access and letting the replay fail later for a reason that no longer names the
    // operation that performed it.
    let resolve =
        |roots: &HashMap<AtomId, Option<ReferenceRootHandle>>, atom: AtomId, operation: &str| match roots.get(&atom) {
            Some(root) => Ok(*root),
            None if is_reference(atom) => Err(ProgramError::MalformedProgram(format!(
                "operation `{operation}` reaches a reference that entered region `{}` neither through its boundary \
                 nor through its capture scope",
                region.id(),
            ))),
            None => Ok(None),
        };
    for instruction in region.instructions() {
        let operation = instruction.operation();
        let semantics = operation.reference_semantics();
        for access in semantics.inputs() {
            let accessed = operand(instruction, access.input_index(), "an accessed")?;
            if let Some(root) = resolve(&roots, accessed, operation.name())? {
                summary.record(root, access.mode(), operation.name())?;
            }
        }
        for output in semantics.outputs() {
            let defined = instruction.outputs().get(output.output_index()).copied().ok_or_else(|| {
                ProgramError::MalformedProgram(format!(
                    "operation `{}` classifies output {} but the application has {} outputs",
                    operation.name(),
                    output.output_index(),
                    instruction.outputs().len(),
                ))
            })?;
            let root = match output {
                ReferenceOutput::Root { .. } => None,
                ReferenceOutput::Alias { input_index, .. } => {
                    resolve(&roots, operand(instruction, *input_index, "an aliased")?, operation.name())?
                }
            };
            roots.insert(defined, root);
        }
        let mut attached_output_roots = Vec::with_capacity(instruction.regions().len());
        for (region_index, attached) in instruction.regions().iter().copied().enumerate() {
            let attached = region.with_id(attached)?;
            let nested = attached
                .input_ids()
                .iter()
                .copied()
                .enumerate()
                .map(|(input_index, input)| {
                    if !attached.atoms()[input.index()].r#type().is_reference() {
                        return Ok(None);
                    }
                    let Some(operand_index) = operation.input_region_provenance(region_index, input_index) else {
                        return Err(ProgramError::MalformedProgram(format!(
                            "operation `{}` passes a reference into region {region_index} input {input_index} \
                             without declaring which operand supplies it",
                            operation.name(),
                        )));
                    };
                    resolve(&roots, operand(instruction, operand_index, "a region")?, operation.name())
                })
                .collect::<Result<Vec<_>, _>>()?;

            // The nested closure is summarized on its own first, so that an operation restricting what its regions may
            // do to an entering root is held to that restriction here, where the offending region is still named,
            // rather than only indirectly when a rebuilt region contradicts the widening it was given.
            let nested_captures = nested_capture_scope(
                operation.region_capture_input_count(region_index),
                nested.as_slice(),
                captures,
                attached.id(),
            )?;
            let mut nested_summary = ReferenceRegionSummary::default();
            let nested_outputs =
                summarize_region_closure(attached, nested.as_slice(), &nested_captures, &mut nested_summary)?;
            validate_region_accesses(operation, region_index, &nested_summary)?;
            summary.absorb(&nested_summary);
            attached_output_roots.push(nested_outputs);
        }

        // A reference-typed output of a region-carrying operation preserves a root rather than classifying one, so it
        // resolves through the generic hooks that state where it came from: an explicit operand identity when the
        // operation declares one, and otherwise the region output it forwards.
        for (output_index, output) in instruction.outputs().iter().copied().enumerate() {
            if !is_reference(output) || roots.contains_key(&output) {
                continue;
            }
            let preserved = match operation.reference_output_identity_input(output_index) {
                Some(input_index) => {
                    resolve(&roots, operand(instruction, input_index, "a preserved")?, operation.name())?
                }
                None => forwarded_output_root(operation, output_index, attached_output_roots.as_slice())?,
            };
            roots.insert(output, preserved);
        }
    }
    let output_roots = region
        .output_ids()
        .iter()
        .copied()
        .map(|output| if is_reference(output) { roots.get(&output).copied().flatten() } else { None })
        .collect::<Vec<_>>();
    summary.reached.extend(output_roots.iter().copied().flatten());
    Ok(output_roots)
}

/// Validates every exact access mode in `summary` against one attached-region policy.
fn validate_region_accesses<O: Operation>(
    operation: &O,
    region_index: usize,
    summary: &ReferenceRegionSummary,
) -> Result<(), ProgramError> {
    for root in summary.accessed() {
        for mode in summary.access_modes(root) {
            if !operation.allows_reference_access_through_region_input(region_index, mode) {
                return Err(ProgramError::MalformedProgram(format!(
                    "operation `{}` does not allow region {region_index} to access {root} with mode `{mode}`",
                    operation.name(),
                )));
            }
        }
    }
    Ok(())
}

/// Returns the caller root one region-carrying operation's reference-typed output forwards out of its attached
/// regions, requiring every region that contributes to that output to agree on it.
///
/// # Parameters
///
///   - `operation`: Operation producing the output.
///   - `output_index`: Output position being resolved.
///   - `attached_output_roots`: Root each attached region's declared outputs denote, in region order.
fn forwarded_output_root<O: Operation>(
    operation: &O,
    output_index: usize,
    attached_output_roots: &[Vec<Option<ReferenceRootHandle>>],
) -> Result<Option<ReferenceRootHandle>, ProgramError> {
    let provenance = operation.output_region_provenance(output_index);
    if provenance.is_empty() {
        return Err(ProgramError::MalformedProgram(format!(
            "operation `{}` produces a reference at output {output_index} without declaring which operand root it \
             preserves or which region output it forwards",
            operation.name()
        )));
    }
    let mut forwarded = None;
    for (position, origin) in provenance.iter().enumerate() {
        let root = attached_output_roots
            .get(origin.region_index)
            .and_then(|roots| roots.get(origin.output_index).copied())
            .ok_or_else(|| {
                ProgramError::MalformedProgram(format!(
                    "operation `{}` forwards output {output_index} from region {} output {}, which it does not \
                     attach",
                    operation.name(),
                    origin.region_index,
                    origin.output_index,
                ))
            })?;
        if position == 0 {
            forwarded = root;
        } else if forwarded != root {
            return Err(ProgramError::MalformedProgram(format!(
                "operation `{}` forwards output {output_index} from regions that return different reference roots",
                operation.name(),
            )));
        }
    }
    Ok(forwarded)
}

/// Lifts one stored program constant into a discharge carrier.
///
/// A reference-typed constant resolves through the active [`ReferenceCaptureScope`], which is how a capture-lifted
/// program's nested regions name their caller's references: the constant denotes the root that capture position
/// already binds, so it yields that root's whole-root handle rather than a second root of its own.
///
/// A reference-typed constant that no scope resolves is rejected rather than lifted. Reference discharge threads roots
/// through the environment it owns, and such a reference belongs to no root: it never entered through an input, a
/// capture binding, or an allocation, so nothing in the environment describes it. Wrapping it as an ordinary value
/// instead would let it survive into the destination and silently break the reference-freedom guarantee of
/// [`ReferenceDischargeResult`].
///
/// # Parameters
///
///   - `context`: Active discharge context, supplying both the capture scope and the destination that lifts the
///     constant.
///   - `constant`: Stored program constant being lifted.
///
/// # Errors
///
/// Returns [`ProgramError::MalformedProgram`] when a reference-typed constant resolves to no root, or resolves to a
/// root whose reference type is not the one the constant declares, and propagates the destination's own lift error.
fn lift_constant<C: Context, P: ReferenceDischargePolicy<C>>(
    context: &ReferenceDischargeContext<C, P>,
    constant: C::Constant,
) -> Result<ReferenceDischargeValue<C, P>, ProgramError> {
    if let Some(r#type) = P::project_reference_type(constant.r#type().as_ref()) {
        let Some(root) = context.captures().resolve(&constant) else {
            return Err(ProgramError::MalformedProgram(format!(
                "reference discharge cannot lift a constant of reference type `{type}`; a reference enters a program \
                 through an input, a capture binding, or an allocation",
            )));
        };

        // A capture constant names the whole root its position binds, so a narrower declared type would silently
        // widen to the root's own value where the constant is used.
        let bound = context.root_reference_type(root)?;
        if r#type != bound {
            return Err(ProgramError::MalformedProgram(format!(
                "reference discharge resolved a capture constant of reference type `{type}` to {root}, which carries \
                 the reference type `{bound}`",
            )));
        }
        return context.root_handle(root);
    }
    Ok(ReferenceDischargeValue::Ordinary(context.parent().lift(constant)?))
}

/// Replays one reference-free operation application verbatim over its rewritten operands.
///
/// This is the shared rule body for every operation that touches no reference: it is the discharge counterpart of the
/// ordinary interpretation path, and it is where the conversion seam from the payload into the destination's operation
/// family is spent. Because the operation is replayed rather than reinterpreted, the destination decides what
/// replaying means, so an eager destination executes it and a staging destination records it.
///
/// The precondition is *reference freedom*, not effect purity in the [`Effects`](crate::Effects) sense. An operation
/// with ordered or other effects replays here unchanged, because replaying it reproduces those effects in the
/// destination exactly as the source performed them; only a reference makes the rewrite the operation's own business.
///
/// A region-carrying application replays verbatim only when nothing in its attached closure touches a reference: its
/// regions are copied into the destination as they stand, which is exactly right for an operation whose regions
/// contain no state to thread. As soon as a reference does appear anywhere in that closure — or as an operand — the
/// application is rejected, because how a reference boundary widens is knowledge that belongs to the operation, and
/// such an operation must implement its own [`ReferenceDischargeableOperation`] rule.
///
/// # Parameters
///
///   - `operation`: Operation application being replayed.
///   - `context`: Active discharge context whose [`parent`](ReferenceDischargeContext::parent) binds the replay.
///   - `driver`: Application-scoped driver supplying any attached regions.
///   - `inputs`: Carriers supplied as this application's operands, in operation-defined order.
///
/// # Errors
///
/// Returns [`ProgramError::UnsupportedOperation`] when a region-carrying application touches reference state, returns
/// [`ProgramError::MalformedProgram`] when a region-free application receives a live reference handle, and propagates
/// the destination's error from the replay itself.
pub fn discharge_reference_free_operation<C, P, O, D>(
    operation: &O,
    context: &ReferenceDischargeContext<C, P>,
    driver: &D,
    inputs: &[ReferenceDischargeValue<C, P>],
) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError>
where
    C: Context<Operation: From<O>>,
    P: ReferenceDischargePolicy<C>,
    O: Clone + Operation<Type = C::Type>,
    D: ReferenceDischargeDriver<C, P>,
{
    if driver.region_count() != 0 {
        let touches_references = inputs.iter().any(|input| matches!(input, ReferenceDischargeValue::Reference(_)))
            || driver.regions().any(region_closure_touches_references);
        if touches_references {
            return Err(ProgramError::UnsupportedOperation {
                message: format!("`{}` carries reference state but has no reference discharge rule", operation.name()),
            });
        }
    }
    let regions = driver.regions().map(RegionRef::to_program).collect::<Vec<_>>();
    let values = inputs
        .iter()
        .enumerate()
        .map(|(input_index, input)| {
            input
                .expect_ordinary(&format!("an ordinary operand {input_index} of `{}`", operation.name()))
                .cloned()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let outputs = context.parent().bind(operation.clone(), regions, values.as_slice())?;
    Ok(outputs.into_iter().map(ReferenceDischargeValue::Ordinary).collect())
}

/// Returns whether `region` or any attached descendant contains a reference type or reference operation.
fn region_closure_touches_references<V: Value, O: Operation<Type = V::Type>>(region: RegionRef<'_, V, O>) -> bool {
    region.contains_atom_type_in_closure(Type::is_reference)
        || region
            .instructions_in_closure()
            .any(|(_, instruction)| !instruction.operation().reference_semantics().is_empty())
}

/// Replays one access to a *preserved* root verbatim into the destination.
///
/// A preserved root survives partial reference discharge as an ordinary reference of the destination universe, so the
/// honest rewrite of an access to it is no rewrite at all: the operation is bound again, over the exact destination
/// reference value each handle denotes, and its results are the destination's own. This is the shared rule body every
/// access primitive reaches once it finds that its operand's root was not selected for discharge, which is why it
/// takes no driver: an access carries no regions.
///
/// Each reference operand's *liveness* is checked against the environment rather than assumed from the handle, while
/// the destination value it contributes comes from the handle itself, which is the only place a derived view's exact
/// value lives. Replaying reproduces the source's own operation, which the destination is free to reject later, but a
/// use-after-consume is discharge's own invariant and belongs at the access that violates it.
///
/// A reference-typed result is rejected rather than wrapped. The environment would have no root for it, so it could
/// later cross a boundary or reach an access as an untracked value; an operation that derives a reference owns that
/// bookkeeping and must state it in its own rule, as the view primitives do.
///
/// # Parameters
///
///   - `operation`: Access being replayed.
///   - `context`: Active discharge context whose [`parent`](ReferenceDischargeContext::parent) binds the replay.
///   - `inputs`: Carriers supplied as this application's operands, in operation-defined order.
///
/// # Errors
///
/// Returns [`ProgramError::MalformedProgram`] when an operand's root is no longer live, when an operand denotes a
/// discharged root, which has no destination reference value, or when a result is reference-typed, and propagates the
/// destination's error from the replay itself.
pub fn discharge_preserved_access<C, P, O>(
    operation: &O,
    context: &ReferenceDischargeContext<C, P>,
    inputs: &[ReferenceDischargeValue<C, P>],
) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError>
where
    C: Context<Operation: From<O>>,
    P: ReferenceDischargePolicy<C>,
    O: Clone + Operation<Type = C::Type>,
{
    let values = inputs
        .iter()
        .map(|input| match input {
            ReferenceDischargeValue::Ordinary(value) => Ok(value.clone()),
            ReferenceDischargeValue::Reference(reference) => {
                if context.root_is_discharged(reference.root())? {
                    Err(ProgramError::MalformedProgram(format!(
                        "reference discharge cannot replay `{}` over discharged {}, which has no destination reference \
                         value",
                        operation.name(),
                        reference.root(),
                    )))
                } else {
                    // `bind_preserved` and `derive` maintain the invariant that every handle on a preserved root
                    // carries the exact destination reference value it denotes, so this reports a handle that reached
                    // here some other way rather than panicking inside a fallible rewrite.
                    reference.preserved().cloned().ok_or_else(|| {
                        ProgramError::MalformedProgram(format!(
                            "reference discharge cannot replay `{}` over preserved {}, whose handle carries no \
                             destination reference value",
                            operation.name(),
                            reference.root(),
                        ))
                    })
                }
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let outputs = context.parent().bind(operation.clone(), Vec::new(), values.as_slice())?;
    outputs
        .into_iter()
        .enumerate()
        .map(|(output_index, output)| {
            if let Some(r#type) = P::project_reference_type(output.r#type().as_ref()) {
                return Err(ProgramError::MalformedProgram(format!(
                    "reference discharge replayed `{}` over a preserved root, but its output {output_index} is the \
                     reference `{type}`; an operation that derives a reference owns that root and needs a reference \
                     discharge rule of its own",
                    operation.name(),
                )));
            }
            Ok(ReferenceDischargeValue::Ordinary(output))
        })
        .collect()
}

/// Rewrites one *positionally forwarding* region-carrying application so that the references its region closures
/// touch become explicit immutable state.
///
/// This is the shared rule body for the two structured shapes whose regions all mirror the operand list after a
/// constant leading offset and whose results are each region's own outputs: a condition, whose branches follow its
/// predicate, and a positional call, whose single callee follows nothing. Both widen the same way, so both reach it:
///
///   - the roots every region closure touches are threaded in as operands appended after the declared ones, unless
///     they are already reference operands, in which case they thread at their own position;
///   - only the roots some closure *mutates* are published back, as outputs appended after the declared ones. A root
///     the closures merely read needs no successor state, and pruning it is what keeps a read-only branch's boundary
///     identical to its source boundary;
///   - every attached region receives the identical state positions, so a rebuilt condition's branches keep agreeing
///     with each other. Only the capture prefix is read per region, because how many of a region's leading inputs are
///     its own captures is the operation's own per-region declaration.
///
/// # Parameters
///
///   - `operation`: Operation application being rewritten. It is replayed unchanged, because threading state past a
///     positional boundary changes only the boundary.
///   - `context`: Active discharge context owning the root environment.
///   - `driver`: Application-scoped driver supplying the attached regions.
///   - `inputs`: Carriers supplied as this application's operands, in operation-defined order.
///   - `leading_operand_count`: Number of leading operands that parameterize the operation itself rather than being
///     forwarded to its regions, which is one for a condition's predicate and zero for a positional call.
///
/// # Errors
///
/// Returns [`ProgramError::MalformedProgram`] when the application has fewer operands than
/// `leading_operand_count`, when a leading operand is a live reference handle, when an attached region's boundary does
/// not forward the remaining operands positionally, when a reference operand names a derived view rather than a whole
/// root, when a region closure reaches a root that never entered the boundary or consumes one, when a region returns a
/// root its caller never threaded, when the attached regions disagree on which outputs denote references, or when a
/// region mutates a root the widening did not predict.
pub fn discharge_positional_region_operation<C, P, O, D>(
    operation: &O,
    context: &ReferenceDischargeContext<C, P>,
    driver: &D,
    inputs: &[ReferenceDischargeValue<C, P>],
    leading_operand_count: usize,
) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError>
where
    C: Context<Operation: From<O>>,
    P: ReferenceDischargePolicy<C>,
    O: Clone + Operation<Type = C::Type>,
    D: ReferenceDischargeDriver<C, P>,
{
    let name = operation.name();
    if inputs.len() < leading_operand_count {
        return Err(ProgramError::MalformedProgram(format!(
            "operation `{name}` forwards its operands after {leading_operand_count} leading operands but the \
             application has {} operands",
            inputs.len(),
        )));
    }
    let (leading, forwarded) = inputs.split_at(leading_operand_count);
    for (index, input) in leading.iter().enumerate() {
        input.expect_ordinary(&format!("an ordinary leading operand {index} of `{name}`"))?;
    }
    let forwarded_roots =
        forwarded.iter().map(|operand| context.operand_root(operand, name)).collect::<Result<Vec<_>, _>>()?;

    // Every region forwards the same operands, so one summary of all of them decides one shared boundary. It is seeded
    // from the first region rather than from an empty summary, because merging keeps the receiver's declared output
    // roots and an empty summary declares none.
    let region_count = driver.region_count();
    let mut summary: Option<ReferenceRegionSummary> = None;
    for index in 0..region_count {
        let region = driver.region(index)?;
        check_count!("input", region.input_ids(), forwarded.len(), ProgramError);
        let region_summary = context.region_summary(operation, index, region, forwarded_roots.as_slice())?;
        summary = Some(match summary {
            Some(summary) => summary.merged(&region_summary),
            None => region_summary,
        });
    }
    let summary = summary.ok_or_else(|| {
        ProgramError::MalformedProgram(format!("operation `{name}` forwards its operands but attaches no regions"))
    })?;

    // A region that returns a discharged root already publishes its final state at that output position, so only a
    // mutated state root absent from the declared outputs needs an appended output. The added input set also includes
    // a returned preserved root absent from the operands: its inherited capture must be rebound in the rebuilt region
    // even though it contributes no state.
    let represented = summary.output_roots().iter().copied().flatten().collect::<BTreeSet<_>>();
    let threaded = context.threaded_state_roots(&summary, name)?;
    let operand_roots = forwarded_roots.iter().copied().flatten().collect::<BTreeSet<_>>();
    let entering = threaded
        .union(&represented)
        .filter(|root| !operand_roots.contains(root))
        .copied()
        .collect::<Vec<_>>();
    let leaving = threaded
        .difference(&represented)
        .copied()
        .filter(|root| summary.is_mutated(*root))
        .collect::<Vec<_>>();

    // Every mutated root is published, whether through an appended output or through a declared reference output, and
    // that complete set is what the rebuilt regions are held to.
    let published = threaded.iter().copied().filter(|root| summary.is_mutated(*root)).collect::<Vec<_>>();

    let source_output_count = driver.region(0)?.output_ids().len();
    let declared_input_roots = forwarded_roots.clone();
    let mut regions = Vec::with_capacity(region_count);
    for index in 0..region_count {
        // Every region receives the same state positions, so a rebuilt condition's branches keep agreeing with each
        // other. Only the capture prefix is read per region, because it is the operation's own per-region declaration.
        let boundary = ReferenceRegionDischargeBoundary::new(
            operation,
            index,
            declared_input_roots.clone(),
            entering.clone(),
            forwarded.len(),
            leaving.clone(),
            source_output_count,
        );
        let fork = driver.discharge_region_program(context, index, &boundary)?;
        fork.validate_predicted_mutations(published.as_slice(), name)?;
        fork.validate_predicted_output_roots(summary.output_roots(), name)?;
        regions.push(fork.into_program());
    }
    let output_roots = summary.output_roots();

    let mut operands = Vec::with_capacity(inputs.len() + entering.len());
    for input in inputs {
        operands.push(context.operand_value(input)?);
    }
    for root in &entering {
        let carrier = context.root_handle(*root)?;
        operands.push(context.operand_value(&carrier)?);
    }
    let outputs = context.parent().bind(operation.clone(), regions, operands.as_slice())?;
    check_count!("output", outputs, source_output_count + leaving.len(), ProgramError);

    // A declared output that denotes a reference is reported as the handle the caller already holds rather than as a
    // value. For a discharged root that output carried its final state, which is merged back; for a preserved root it
    // carried the reference itself, and there is nothing to merge. Appended outputs publish the remaining final
    // states.
    let mut results = Vec::with_capacity(source_output_count);
    for (position, output) in outputs.into_iter().enumerate() {
        if position >= source_output_count {
            context.set_discharged_state(leaving[position - source_output_count], output)?;
            continue;
        }
        match output_roots[position] {
            Some(root) => {
                if threaded.contains(&root) {
                    context.merge_discharged_state(root, output, summary.is_mutated(root))?;
                }
                let forwarded = forwarded_roots
                    .iter()
                    .position(|candidate| *candidate == Some(root))
                    .and_then(|position| forwarded.get(position).cloned());
                results.push(match forwarded {
                    Some(forwarded) => forwarded,
                    None => context.root_handle(root)?,
                });
            }
            None => results.push(ReferenceDischargeValue::Ordinary(output)),
        }
    }
    Ok(results)
}

/// Validates that one destination value denotes a reference of exactly `r#type`.
///
/// # Parameters
///
///   - `value`: Destination value offered as a preserved root's reference.
///   - `r#type`: Reference type the handle that will carry `value` exposes.
fn validate_preserved_value<C: Domain, P: ReferenceDischargePolicy<C>>(
    value: &C::Value,
    r#type: &ReferenceType<P::Referent>,
) -> Result<(), ProgramError> {
    match P::project_reference_type(value.r#type().as_ref()) {
        Some(actual) if &actual == r#type => Ok(()),
        Some(actual) => Err(ProgramError::MalformedProgram(format!(
            "reference discharge preserved a root as `{actual}` but its handle exposes `{type}`",
        ))),
        None => Err(ProgramError::MalformedProgram(format!(
            "reference discharge preserved a root as `{}`, which is not a reference type",
            value.r#type(),
        ))),
    }
}

/// Validates that one immutable discharged state carries the lifted referent type of `r#type`.
fn validate_discharged_value_type<C: Domain, P: ReferenceDischargePolicy<C>>(
    value: &C::Value,
    r#type: &ReferenceType<P::Referent>,
) -> Result<(), ProgramError> {
    let expected = P::lift_referent_type(r#type.referent().clone());
    if value.r#type().as_ref() != &expected {
        return Err(ProgramError::MalformedProgram(format!(
            "reference discharge state has type `{}` but root `{}` requires `{expected}`",
            value.r#type(),
            r#type,
        )));
    }
    Ok(())
}

/// Represents [`Operation`]s that can be discharged (i.e., rewritten so that the references they touch become
/// explicit immutable state).
///
/// The trait is parameterized by the destination [`Domain`] `C` that owns the rewritten values and by the
/// [`ReferenceDischargePolicy`] `P` naming the reference universe being discharged. Every rule receives the active
/// [`ReferenceDischargeContext`], which owns the root environment, plus a [`ReferenceDischargeDriver`] exposing the
/// application's replay position and attached regions.
///
/// Reference primitives implement their own rewrites: an allocation binds a fresh root, an access acts on the root's
/// current state through the policy's alias mechanics, and a freeze yields the current state and unbinds the root.
/// Structured operations implement their own boundary widening, because widening is a property of what the operation
/// does with its regions and therefore belongs to the operation. Everything else replays as-is over rewritten
/// operands. The system is consequently open over primitives: a third-party operation family participates by
/// implementing this trait, with no companion declaration surface beyond the generic
/// [`Operation::reference_semantics`] and region-provenance hooks it already implements.
///
/// `C` is bounded by [`Domain`] rather than [`Context`] for the same reason
/// [`InterpretableOperation`](crate::InterpretableOperation) is: the destination context's own binding contract is
/// established in terms of its operation family's rules, so reaching [`Context`] through this trait would make that
/// obligation recursive. Implementations bound `C` by the value and conversion capabilities their rewrite actually
/// uses, and higher-order rules request nested work through their driver rather than carrying a bound stating that
/// their own operation family is dischargeable, which is what keeps an operation enum's bound graph finite.
///
/// The super-trait is a plain [`Operation`] rather than `Operation<Type = C::Type>`, with the equality required per
/// function instead, matching [`BatchableOperation`](crate::BatchableOperation): the current trait solver cannot
/// discharge that projection equality at implementation heads whose context type is built from `Self`.
pub trait ReferenceDischargeableOperation<C: Domain, P: ReferenceDischargePolicy<C>>: Operation {
    /// Rewrites this operation application so that the references it touches become explicit immutable state, and
    /// returns the carriers its outputs produce.
    ///
    /// # Parameters
    ///
    ///   - `context`: Active discharge context owning the root environment, through whose
    ///     [`parent`](ReferenceDischargeContext::parent) the rewritten work is bound.
    ///   - `driver`: Application-scoped driver exposing the replay position and any attached regions.
    ///   - `inputs`: Carriers supplied as this application's operands, in operation-defined order.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError`] when this application cannot be rewritten — because an operand is of the wrong kind,
    /// because the references its regions touch cannot be threaded through its boundary, or because the destination
    /// rejected the rewritten work.
    fn discharge_references<D: ReferenceDischargeDriver<C, P>>(
        &self,
        context: &ReferenceDischargeContext<C, P>,
        driver: &D,
        inputs: &[ReferenceDischargeValue<C, P>],
    ) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError>
    where
        Self: Operation<Type = C::Type>;
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::rc::Rc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::contexts::EagerContext;
    use crate::interpretation::{InterpretableOperation, InterpretationDriver};
    use crate::macros::check_count;
    use crate::operations::Add;
    use crate::parameters::Placeholder;
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::effects::{Effect, Effects};
    use crate::programs::identities::NoIdentity;
    use crate::programs::regions::{OutputRegionProvenance, RegionInterface, RegionSlot};
    use crate::programs::types::TypeError;

    use crate::captures::CaptureReference;

    use super::super::semantics::{ReferenceAliasKind, ReferenceInput, ReferenceOperationSemantics, ReferenceOutput};
    use super::*;

    /// Minimal generic type universe for the boundary tests below: opaque indexed values plus references over them.
    #[derive(Clone, Debug, PartialEq)]
    enum TestType {
        Value(u8),
        Reference(Box<ReferenceType<TestType>>),
    }

    impl Display for TestType {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::Value(index) => write!(formatter, "value<{index}>"),
                Self::Reference(reference) => Display::fmt(reference, formatter),
            }
        }
    }

    impl Parameter for TestType {}

    impl Type for TestType {
        type Identity = NoIdentity;
        type Refinements = ();

        fn is_compatible_with(&self, other: &Self) -> bool {
            self == other
        }

        fn is_refined_by(&self, other: &Self) -> bool {
            self == other
        }

        fn is_scalar(&self) -> bool {
            false
        }

        fn is_complex(&self) -> bool {
            false
        }

        fn is_reference(&self) -> bool {
            matches!(self, Self::Reference(_))
        }
    }

    /// Constant payload of the minimal universe. The boundary tests never materialize concrete values, so the capture
    /// reference stand-in is all they need.
    type TestValue = CaptureReference<TestType>;

    /// Returns a reference type over the opaque value type with the given index.
    fn reference_type(index: u8) -> TestType {
        TestType::Reference(Box::new(ReferenceType::new(TestType::Value(index))))
    }

    /// Minimal generic operation universe for the boundary tests below: allocation, read, and consumption of a
    /// reference, plus one positional call-like region operation.
    #[derive(Copy, Clone, Debug)]
    enum TestOperation {
        NewRoot,
        Read,
        Consume,
        Call,
    }

    impl Display for TestOperation {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str(self.name())
        }
    }

    impl Operation for TestOperation {
        type Type = TestType;

        fn name(&self) -> &'static str {
            match self {
                Self::NewRoot => "test.new_root",
                Self::Read => "test.read",
                Self::Consume => "test.consume",
                Self::Call => "test.call",
            }
        }

        fn region_slots(&self) -> &'static [RegionSlot] {
            match self {
                Self::Call => const { &[RegionSlot::computation("callee")] },
                _ => &[],
            }
        }

        fn infer_output_types(
            &self,
            input_types: &[TestType],
            region_interfaces: &[RegionInterface<TestType>],
        ) -> Result<Vec<TestType>, TypeError> {
            match self {
                Self::NewRoot => Ok(vec![TestType::Reference(Box::new(ReferenceType::new(input_types[0].clone())))]),
                Self::Read | Self::Consume => match input_types.first() {
                    Some(TestType::Reference(reference)) => Ok(vec![reference.referent().clone()]),
                    _ => Err(TypeError::invalid("test operation expected a reference input")),
                },
                Self::Call => Ok(region_interfaces[0].output_types().to_vec()),
            }
        }

        fn input_region_provenance(&self, _region_index: usize, input_index: usize) -> Option<usize> {
            matches!(self, Self::Call).then_some(input_index)
        }

        fn output_region_provenance(&self, output_index: usize) -> Vec<OutputRegionProvenance> {
            match self {
                Self::Call => vec![OutputRegionProvenance { region_index: 0, output_index }],
                _ => Vec::new(),
            }
        }

        fn allows_reference_access_through_region_input(
            &self,
            _region_index: usize,
            mode: ReferenceAccessMode,
        ) -> bool {
            matches!(self, Self::Call) && mode != ReferenceAccessMode::Consume
        }

        fn effects(&self) -> Effects {
            match self {
                Self::Call => Effects::PURE,
                _ => Effects::single(Effect::OrderedState),
            }
        }

        fn reference_semantics(&self) -> Cow<'_, ReferenceOperationSemantics> {
            let semantics = match self {
                Self::NewRoot => {
                    ReferenceOperationSemantics::new(Vec::new(), vec![ReferenceOutput::Root { output_index: 0 }])
                }
                Self::Read => ReferenceOperationSemantics::new(
                    vec![ReferenceInput::new(0, ReferenceAccessMode::Read)],
                    Vec::new(),
                ),
                Self::Consume => ReferenceOperationSemantics::new(
                    vec![ReferenceInput::new(0, ReferenceAccessMode::Consume)],
                    Vec::new(),
                ),
                Self::Call => ReferenceOperationSemantics::default(),
            };
            Cow::Owned(semantics)
        }
    }

    // The fixtures below are the non-array prototype universe shared by the interpreter tests: a deliberately small
    // reference universe whose referents are fixed-length integer lists and whose views are contiguous sub-ranges. It
    // is the standing proof that the discharge architecture has not silently become array-shaped, because nothing in
    // it mentions arrays and its alias mechanics are real rather than trivial.

    thread_local! {
        static OBSERVED_ALLOCATION_POSITIONS: RefCell<Vec<Option<InstructionId>>> =
            const { RefCell::new(Vec::new()) };
    }

    /// Destination universe of the prototype programs.
    type ListDestination = EagerContext<ListIrValue, ListOperation>;

    /// Discharge context over the prototype destination universe.
    type ListDischargeContext = ReferenceDischargeContext<ListDestination, ListReferenceDischarge>;

    /// Carrier flowing through prototype discharge.
    type ListDischargeValue = ReferenceDischargeValue<ListDestination, ListReferenceDischarge>;

    /// Referent type of the prototype universe: a list of integers with a fixed length.
    #[derive(Clone, Debug, PartialEq)]
    struct ListType {
        length: usize,
    }

    impl Display for ListType {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(formatter, "list<{}>", self.length)
        }
    }

    impl Parameter for ListType {}

    impl Type for ListType {
        type Identity = NoIdentity;
        type Refinements = ();

        fn is_compatible_with(&self, other: &Self) -> bool {
            self == other
        }

        fn is_refined_by(&self, other: &Self) -> bool {
            self == other
        }

        fn is_scalar(&self) -> bool {
            self.length == 1
        }

        fn is_complex(&self) -> bool {
            false
        }
    }

    /// Type universe of the prototype programs, pairing ordinary lists with references to them.
    #[derive(Clone, Debug, PartialEq)]
    enum ListIrType {
        List(ListType),
        Reference(ReferenceType<ListType>),
    }

    impl Display for ListIrType {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::List(r#type) => Display::fmt(r#type, formatter),
                Self::Reference(r#type) => Display::fmt(r#type, formatter),
            }
        }
    }

    impl Parameter for ListIrType {}

    impl Type for ListIrType {
        type Identity = NoIdentity;
        type Refinements = ();

        fn is_compatible_with(&self, other: &Self) -> bool {
            self == other
        }

        fn is_refined_by(&self, other: &Self) -> bool {
            self == other
        }

        fn is_scalar(&self) -> bool {
            matches!(self, Self::List(r#type) if r#type.is_scalar())
        }

        fn is_complex(&self) -> bool {
            false
        }

        fn is_reference(&self) -> bool {
            matches!(self, Self::Reference(_))
        }
    }

    /// Value universe of the prototype programs.
    #[derive(Clone, Debug, PartialEq)]
    enum ListIrValue {
        /// Concrete list payload.
        List(Vec<i64>),

        /// Reference value surviving in a destination program. The prototype universe has no runtime holder behind
        /// it, which is what makes it a good test of the machinery that must never look inside one.
        Reference(ReferenceType<ListType>),
    }

    impl ListIrValue {
        /// Returns the list payload of this value, or an error when it is a reference.
        fn list(&self) -> Result<&[i64], ProgramError> {
            match self {
                Self::List(elements) => Ok(elements.as_slice()),
                Self::Reference(r#type) => {
                    Err(ProgramError::MalformedProgram(format!("expected a list but got `{}`", r#type)))
                }
            }
        }
    }

    impl Display for ListIrValue {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::List(elements) => write!(formatter, "{elements:?}"),
                Self::Reference(r#type) => Display::fmt(r#type, formatter),
            }
        }
    }

    impl Parameter for ListIrValue {}

    impl Typed for ListIrValue {
        type Type = ListIrType;

        fn r#type(&self) -> Cow<'_, ListIrType> {
            match self {
                Self::List(elements) => Cow::Owned(ListIrType::List(ListType { length: elements.len() })),
                Self::Reference(r#type) => Cow::Owned(ListIrType::Reference(r#type.clone())),
            }
        }
    }

    impl Value for ListIrValue {
        type DispatchDomain = EagerContext<Self>;
        type ExecutionDomain = EagerContext<Self>;

        fn dispatch_domain(&self) -> Self::DispatchDomain {
            EagerContext::new()
        }

        fn execution_domain(&self) -> Self::ExecutionDomain {
            EagerContext::new()
        }
    }

    impl Add for ListIrValue {
        fn add(&self, rhs: &Self) -> Result<Self, ProgramError> {
            let (lhs, rhs) = (self.list()?, rhs.list()?);
            if lhs.len() != rhs.len() {
                return Err(ProgramError::MalformedProgram(format!(
                    "cannot add lists of lengths {} and {}",
                    lhs.len(),
                    rhs.len(),
                )));
            }
            Ok(Self::List(lhs.iter().zip(rhs).map(|(lhs, rhs)| lhs + rhs).collect()))
        }
    }

    impl Concretizable<bool> for ListIrValue {
        fn concretize(&self) -> Result<bool, ProgramError> {
            match self.list()? {
                [element] => Ok(*element != 0),
                elements => Err(ProgramError::Concretization {
                    message: format!("cannot extract a concrete boolean from a list of length {}", elements.len()),
                }),
            }
        }
    }

    /// View chain of the prototype universe: one contiguous sub-range of the root list.
    #[derive(Copy, Clone, Debug, PartialEq)]
    struct ListAlias {
        offset: usize,
        length: usize,
    }

    impl Parameter for ListAlias {}

    // Binds one single-result prototype operation into a destination. Routing the alias mechanics through the
    // operation family is what keeps this universe's policy independent of whether its destination executes the work
    // or stages it, which one policy implementation must be for a rebuilt region to be traced at all.
    fn bind_list<C: Context<Type = ListIrType, Operation: From<ListOperation>>>(
        context: &C,
        operation: ListOperation,
        inputs: &[C::Value],
    ) -> Result<C::Value, ProgramError> {
        let mut outputs = context.bind(operation, Vec::new(), inputs)?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }

    /// Reference discharge policy of the prototype universe.
    #[derive(Copy, Clone, Debug)]
    struct ListReferenceDischarge;

    // The policy leaves the destination value generic and reaches its alias mechanics through the operation family,
    // which is what one implementation must do to serve both the eager destination this universe's own tests use and
    // the fresh staging destination a rebuilt region is traced into.
    impl<C: Context<Type = ListIrType, Operation: From<ListOperation>>> ReferenceDischargePolicy<C>
        for ListReferenceDischarge
    {
        type Referent = ListType;
        type Alias = ListAlias;

        fn root_alias(referent: &ListType) -> ListAlias {
            ListAlias { offset: 0, length: referent.length }
        }

        fn lift_reference_type(r#type: ReferenceType<ListType>) -> ListIrType {
            ListIrType::Reference(r#type)
        }

        fn lift_referent_type(referent: ListType) -> ListIrType {
            ListIrType::List(referent)
        }

        fn project_reference_type(r#type: &ListIrType) -> Option<ReferenceType<ListType>> {
            match r#type {
                ListIrType::Reference(reference) => Some(reference.clone()),
                ListIrType::List(_) => None,
            }
        }

        fn read(context: &C, current: &C::Value, alias: &ListAlias) -> Result<C::Value, ProgramError> {
            bind_list(context, ListOperation::Select { offset: alias.offset, length: alias.length }, &[current.clone()])
        }

        fn write(
            context: &C,
            current: &C::Value,
            replacement: C::Value,
            alias: &ListAlias,
        ) -> Result<C::Value, ProgramError> {
            bind_list(context, ListOperation::Splice { offset: alias.offset }, &[current.clone(), replacement])
        }

        fn replace(
            context: &C,
            current: &C::Value,
            replacement: C::Value,
            alias: &ListAlias,
        ) -> Result<(C::Value, C::Value), ProgramError> {
            let previous = Self::read(context, current, alias)?;
            let successor =
                bind_list(context, ListOperation::Splice { offset: alias.offset }, &[current.clone(), replacement])?;
            Ok((previous, successor))
        }
    }

    // The prototype universe accumulates by lifting its own addition into the destination, which is the shape a
    // universe whose values carry no arithmetic capability of their own uses.
    impl<C: Context<Type = ListIrType, Operation: From<ListOperation>>> ReferenceAccumulationPolicy<C>
        for ListReferenceDischarge
    {
        fn accumulate(
            context: &C,
            current: &C::Value,
            update: C::Value,
            alias: &ListAlias,
        ) -> Result<C::Value, ProgramError> {
            let selected = Self::read(context, current, alias)?;
            let accumulated = bind_list(context, ListOperation::Add, &[selected, update])?;
            bind_list(context, ListOperation::Splice { offset: alias.offset }, &[current.clone(), accumulated])
        }
    }

    /// Operation family of the prototype universe.
    #[derive(Copy, Clone, Debug)]
    enum ListOperation {
        Add,
        Select { offset: usize, length: usize },
        Splice { offset: usize },
        ReferenceNew,
        Slice { offset: usize, length: usize },
        Read,
        Write,
        Swap,
        AddUpdate,
        Freeze,
        UnreportedFreeze,
        Call,
    }

    impl Display for ListOperation {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str(self.name())
        }
    }

    impl Operation for ListOperation {
        type Type = ListIrType;

        fn name(&self) -> &'static str {
            match self {
                Self::Add => "list.add",
                Self::Select { .. } => "list.select",
                Self::Splice { .. } => "list.splice",
                Self::ReferenceNew => "list.reference_new",
                Self::Slice { .. } => "list.slice",
                Self::Read => "list.read",
                Self::Write => "list.write",
                Self::Swap => "list.swap",
                Self::AddUpdate => "list.add_update",
                Self::Freeze => "list.freeze",
                Self::UnreportedFreeze => "test.unreported_freeze",
                Self::Call => "list.call",
            }
        }

        fn region_slots(&self) -> &'static [RegionSlot] {
            match self {
                Self::Call => const { &[RegionSlot::computation("callee")] },
                _ => &[],
            }
        }

        fn input_region_provenance(&self, region_index: usize, input_index: usize) -> Option<usize> {
            (matches!(self, Self::Call) && region_index == 0).then_some(input_index)
        }

        fn output_region_provenance(&self, output_index: usize) -> Vec<OutputRegionProvenance> {
            match self {
                Self::Call => vec![OutputRegionProvenance { region_index: 0, output_index }],
                _ => Vec::new(),
            }
        }

        fn infer_output_types(
            &self,
            input_types: &[ListIrType],
            region_interfaces: &[RegionInterface<ListIrType>],
        ) -> Result<Vec<ListIrType>, TypeError> {
            let referent = |index: usize| match input_types.get(index) {
                Some(ListIrType::Reference(reference)) => Ok(reference.referent().clone()),
                _ => Err(TypeError::invalid(format!("`{}` expects a reference operand", self.name()))),
            };
            match self {
                Self::Add => {
                    check_count!("input", input_types, 2, TypeError);
                    Ok(vec![input_types[0].clone()])
                }
                Self::Select { offset, length } => {
                    check_count!("input", input_types, 1, TypeError);
                    let ListIrType::List(source) = &input_types[0] else {
                        return Err(TypeError::invalid("`list.select` expects a list operand"));
                    };
                    if offset + length > source.length {
                        return Err(TypeError::invalid(format!(
                            "selection [{offset}, {}) does not fit `{source}`",
                            offset + length,
                        )));
                    }
                    Ok(vec![ListIrType::List(ListType { length: *length })])
                }
                Self::Splice { offset } => {
                    check_count!("input", input_types, 2, TypeError);
                    let (ListIrType::List(target), ListIrType::List(update)) = (&input_types[0], &input_types[1])
                    else {
                        return Err(TypeError::invalid("`list.splice` expects two list operands"));
                    };
                    if offset + update.length > target.length {
                        return Err(TypeError::invalid(format!(
                            "splice [{offset}, {}) does not fit `{target}`",
                            offset + update.length,
                        )));
                    }
                    Ok(vec![input_types[0].clone()])
                }
                Self::ReferenceNew => {
                    check_count!("input", input_types, 1, TypeError);
                    let ListIrType::List(referent) = &input_types[0] else {
                        return Err(TypeError::invalid("`list.reference_new` expects a list operand"));
                    };
                    Ok(vec![ListIrType::Reference(ReferenceType::new(referent.clone()))])
                }
                Self::Slice { offset, length } => {
                    check_count!("input", input_types, 1, TypeError);
                    let referent = referent(0)?;
                    if offset + length > referent.length {
                        return Err(TypeError::invalid(format!(
                            "view [{offset}, {}) does not fit `{referent}`",
                            offset + length,
                        )));
                    }
                    Ok(vec![ListIrType::Reference(ReferenceType::new(ListType { length: *length }))])
                }
                Self::Read | Self::Freeze | Self::UnreportedFreeze => {
                    check_count!("input", input_types, 1, TypeError);
                    Ok(vec![ListIrType::List(referent(0)?)])
                }
                Self::Write => {
                    check_count!("input", input_types, 2, TypeError);
                    referent(0)?;
                    Ok(Vec::new())
                }
                Self::Swap => {
                    check_count!("input", input_types, 2, TypeError);
                    Ok(vec![ListIrType::List(referent(0)?)])
                }
                Self::AddUpdate => {
                    check_count!("input", input_types, 2, TypeError);
                    referent(0)?;
                    Ok(Vec::new())
                }
                Self::Call => {
                    check_count!("region", region_interfaces, 1, TypeError);
                    check_count!("input", input_types, region_interfaces[0].input_types().len(), TypeError);
                    Ok(region_interfaces[0].output_types().to_vec())
                }
            }
        }

        fn reference_semantics(&self) -> Cow<'_, ReferenceOperationSemantics> {
            let semantics = match self {
                Self::ReferenceNew => {
                    ReferenceOperationSemantics::new(Vec::new(), vec![ReferenceOutput::Root { output_index: 0 }])
                }
                Self::Slice { .. } => ReferenceOperationSemantics::new(
                    Vec::new(),
                    vec![ReferenceOutput::Alias { output_index: 0, input_index: 0, kind: ReferenceAliasKind::View }],
                ),
                Self::Read => ReferenceOperationSemantics::new(
                    vec![ReferenceInput::new(0, ReferenceAccessMode::Read)],
                    Vec::new(),
                ),
                Self::Write => ReferenceOperationSemantics::new(
                    vec![ReferenceInput::new(0, ReferenceAccessMode::Write)],
                    Vec::new(),
                ),
                Self::Swap => ReferenceOperationSemantics::new(
                    vec![ReferenceInput::new(0, ReferenceAccessMode::ReadWrite)],
                    Vec::new(),
                ),
                Self::AddUpdate => ReferenceOperationSemantics::new(
                    vec![ReferenceInput::new(0, ReferenceAccessMode::Accumulate)],
                    Vec::new(),
                ),
                Self::Freeze => ReferenceOperationSemantics::new(
                    vec![ReferenceInput::new(0, ReferenceAccessMode::Consume)],
                    Vec::new(),
                ),
                Self::Add | Self::Select { .. } | Self::Splice { .. } | Self::UnreportedFreeze | Self::Call => {
                    return Cow::Borrowed(ReferenceOperationSemantics::empty());
                }
            };
            Cow::Owned(semantics)
        }

        fn effects(&self) -> Effects {
            match self {
                Self::Add | Self::Select { .. } | Self::Splice { .. } | Self::Slice { .. } | Self::Call => {
                    Effects::PURE
                }
                _ => Effects::single(Effect::OrderedState),
            }
        }
    }

    /// Region-policy stand-in that permits exactly one non-consuming reference access mode.
    #[derive(Copy, Clone, Debug)]
    struct SingleModeRegionOperation(ReferenceAccessMode);

    impl Display for SingleModeRegionOperation {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str(self.name())
        }
    }

    impl Operation for SingleModeRegionOperation {
        type Type = ListIrType;

        fn name(&self) -> &'static str {
            "test.single_mode_region"
        }

        fn infer_output_types(
            &self,
            _input_types: &[ListIrType],
            _region_interfaces: &[RegionInterface<ListIrType>],
        ) -> Result<Vec<ListIrType>, TypeError> {
            Ok(Vec::new())
        }

        fn allows_reference_access_through_region_input(&self, region_index: usize, mode: ReferenceAccessMode) -> bool {
            region_index == 0 && mode == self.0
        }
    }

    impl<C: Domain<Type = ListIrType, Value = ListIrValue>> InterpretableOperation<C> for ListOperation {
        fn interpret<D: InterpretationDriver<C>>(
            &self,
            context: &C,
            driver: &D,
            inputs: &[ListIrValue],
        ) -> Result<Vec<ListIrValue>, ProgramError> {
            match self {
                Self::Call => driver.interpret_region(context, 0, inputs.to_vec()),
                Self::Add => {
                    check_count!("input", inputs, 2, ProgramError);
                    Ok(vec![inputs[0].add(&inputs[1])?])
                }
                Self::Select { offset, length } => {
                    check_count!("input", inputs, 1, ProgramError);
                    let elements = inputs[0].list()?;
                    let selected = elements.get(*offset..offset + length).ok_or_else(|| {
                        ProgramError::MalformedProgram(format!(
                            "selection [{offset}, {}) does not fit a list of length {}",
                            offset + length,
                            elements.len(),
                        ))
                    })?;
                    Ok(vec![ListIrValue::List(selected.to_vec())])
                }
                Self::Splice { offset } => {
                    check_count!("input", inputs, 2, ProgramError);
                    let mut spliced = inputs[0].list()?.to_vec();
                    let update = inputs[1].list()?;
                    let length = spliced.len();
                    let range = spliced.get_mut(*offset..offset + update.len()).ok_or_else(|| {
                        ProgramError::MalformedProgram(format!(
                            "splice [{offset}, {}) does not fit a list of length {length}",
                            offset + update.len(),
                        ))
                    })?;
                    range.clone_from_slice(update);
                    Ok(vec![ListIrValue::List(spliced)])
                }
                _ => Err(ProgramError::UnsupportedOperation {
                    message: format!("`{}` must be discharged before interpretation", self.name()),
                }),
            }
        }
    }

    // One implementation covers every prototype operation, which is why the accumulating rule's
    // `ReferenceAccumulationPolicy` requirement appears as an implementation-level bound here: closed operation-enum
    // dispatch reintroduces the union that the policy split otherwise keeps separate.
    impl<C> ReferenceDischargeableOperation<C, ListReferenceDischarge> for ListOperation
    where
        C: Context<Type = ListIrType, Operation: From<ListOperation>>,
    {
        fn discharge_references<D: ReferenceDischargeDriver<C, ListReferenceDischarge>>(
            &self,
            context: &ReferenceDischargeContext<C, ListReferenceDischarge>,
            driver: &D,
            inputs: &[ReferenceDischargeValue<C, ListReferenceDischarge>],
        ) -> Result<Vec<ReferenceDischargeValue<C, ListReferenceDischarge>>, ProgramError> {
            match self {
                Self::Add | Self::Select { .. } | Self::Splice { .. } => {
                    discharge_reference_free_operation(self, context, driver, inputs)
                }
                Self::ReferenceNew => {
                    check_count!("input", inputs, 1, ProgramError);
                    OBSERVED_ALLOCATION_POSITIONS.with_borrow_mut(|positions| positions.push(driver.instruction()));
                    let initial = inputs[0].expect_ordinary("an initial state")?.clone();
                    let initial_type = initial.r#type().into_owned();
                    let output_type = self.infer_output_types(std::slice::from_ref(&initial_type), &[])?.remove(0);
                    let r#type =
                        <ListReferenceDischarge as ReferenceDischargePolicy<C>>::project_reference_type(&output_type)
                            .ok_or_else(|| {
                            ProgramError::MalformedProgram(
                                "`list.reference_new` produced a non-reference type".to_string(),
                            )
                        })?;
                    if context.selects_allocation(driver.instruction(), 0) {
                        return Ok(vec![context.allocate_discharged(r#type, initial)?]);
                    }
                    let mut outputs = context.parent().bind(*self, Vec::new(), std::slice::from_ref(&initial))?;
                    check_count!("output", outputs, 1, ProgramError);
                    Ok(vec![context.bind_preserved(r#type, outputs.remove(0))?])
                }
                Self::Slice { offset, length } => {
                    check_count!("input", inputs, 1, ProgramError);
                    let reference = inputs[0].expect_reference("a reference to view")?;
                    let alias = reference.alias();
                    if offset + length > alias.length {
                        return Err(ProgramError::MalformedProgram(format!(
                            "view [{offset}, {}) does not fit `{}`",
                            offset + length,
                            reference.r#type(),
                        )));
                    }
                    let composed = ListAlias { offset: alias.offset + offset, length: *length };
                    let r#type = ReferenceType::new(ListType { length: *length });
                    let preserved = match reference.preserved() {
                        None => None,
                        Some(value) => {
                            let mut outputs = context.parent().bind(*self, Vec::new(), std::slice::from_ref(value))?;
                            check_count!("output", outputs, 1, ProgramError);
                            Some(outputs.remove(0))
                        }
                    };
                    Ok(vec![context.derive(reference, composed, r#type, preserved)?])
                }
                Self::Read => {
                    check_count!("input", inputs, 1, ProgramError);
                    let reference = inputs[0].expect_reference("a reference to read")?;
                    if reference.preserved().is_some() {
                        return discharge_preserved_access(self, context, inputs);
                    }
                    Ok(vec![ReferenceDischargeValue::Ordinary(context.read(reference)?)])
                }
                Self::Write => {
                    check_count!("input", inputs, 2, ProgramError);
                    let reference = inputs[0].expect_reference("a reference to write")?;
                    let replacement = inputs[1].expect_ordinary("a replacement value")?.clone();
                    if reference.preserved().is_some() {
                        return discharge_preserved_access(self, context, inputs);
                    }
                    context.write(reference, replacement)?;
                    Ok(Vec::new())
                }
                Self::Swap => {
                    check_count!("input", inputs, 2, ProgramError);
                    let reference = inputs[0].expect_reference("a reference to replace")?;
                    let replacement = inputs[1].expect_ordinary("a replacement value")?.clone();
                    if reference.preserved().is_some() {
                        return discharge_preserved_access(self, context, inputs);
                    }
                    Ok(vec![ReferenceDischargeValue::Ordinary(context.replace(reference, replacement)?)])
                }
                Self::AddUpdate => {
                    check_count!("input", inputs, 2, ProgramError);
                    let reference = inputs[0].expect_reference("a reference to accumulate into")?;
                    let update = inputs[1].expect_ordinary("an update value")?.clone();
                    if reference.preserved().is_some() {
                        return discharge_preserved_access(self, context, inputs);
                    }
                    context.accumulate(reference, update)?;
                    Ok(Vec::new())
                }
                Self::Freeze => {
                    check_count!("input", inputs, 1, ProgramError);
                    let reference = inputs[0].expect_reference("a reference to freeze")?;
                    if reference.preserved().is_some() {
                        let outputs = discharge_preserved_access(self, context, inputs)?;
                        context.unbind_preserved(reference)?;
                        return Ok(outputs);
                    }
                    Ok(vec![ReferenceDischargeValue::Ordinary(context.consume(reference)?)])
                }
                Self::UnreportedFreeze => {
                    check_count!("input", inputs, 1, ProgramError);
                    let reference = inputs[0].expect_reference("a reference to freeze")?;
                    Ok(vec![ReferenceDischargeValue::Ordinary(context.consume(reference)?)])
                }
                Self::Call => discharge_positional_region_operation(self, context, driver, inputs, 0),
            }
        }
    }

    #[derive(Copy, Clone)]
    enum TestDischargeMode {
        Local,
        External,
        Malformed,
    }

    struct TestDischargeProvider {
        calls: Rc<Cell<usize>>,
        mode: TestDischargeMode,
    }

    #[derive(Debug, PartialEq, Eq)]
    struct TestPayload<T> {
        value: T,
        input_count: usize,
        output_count: usize,
    }

    impl<T> TestPayload<T> {
        fn new(value: T, input_count: usize, output_count: usize) -> Self {
            Self { value, input_count, output_count }
        }
    }

    impl<T> ReferenceDischargePayload for TestPayload<T> {
        fn input_count(&self) -> usize {
            self.input_count
        }

        fn output_count(&self) -> usize {
            self.output_count
        }

        fn validate_reference_free(&self) -> Result<(), ProgramError> {
            Ok(())
        }
    }

    impl ReferenceDischarge for TestDischargeProvider {
        type DischargedProgram = TestPayload<usize>;

        fn discharge_references(
            self,
            _capture_count: usize,
        ) -> Result<ReferenceDischargeResult<Self::DischargedProgram>, ProgramError> {
            self.calls.set(self.calls.get() + 1);
            match self.mode {
                TestDischargeMode::Local => {
                    ReferenceDischargeResult::from_provider_payload(TestPayload::new(7, 0, 0), 0, 0, Vec::new())
                }
                TestDischargeMode::External => ReferenceDischargeResult::from_provider_payload(
                    TestPayload::new(7, 1, 0),
                    0,
                    0,
                    vec![ReferenceStateBinding::new(ReferenceSource::Input { index: 0 }, None)],
                ),
                TestDischargeMode::Malformed => {
                    ReferenceDischargeResult::from_provider_payload(TestPayload::new(7, 0, 1), 0, 0, Vec::new())
                }
            }
        }
    }

    // Capture seam for the prototype universe, which has no capture constants of its own: a reference-typed constant
    // names the capture position given by its referent length. The seam is the only universe-specific part of capture
    // resolution, so supplying one here exercises every branch of it without inventing a second constant family.
    fn list_capture_position(constant: &ListIrValue) -> Option<usize> {
        match constant {
            ListIrValue::Reference(r#type) => Some(r#type.referent().length),
            ListIrValue::List(_) => None,
        }
    }

    #[test]
    fn test_reference_discharge_result_validates_boundaries() {
        let bindings = vec![
            ReferenceStateBinding::new(ReferenceSource::Capture { index: 0 }, Some(1)),
            ReferenceStateBinding::new(ReferenceSource::Input { index: 0 }, Some(2)),
        ];
        let result =
            ReferenceDischargeResult::from_provider_payload(TestPayload::new("program", 2, 3), 1, 1, bindings.clone())
                .unwrap();
        assert_eq!(result.program().value, "program");
        assert_eq!(result.capture_count(), 1);
        assert_eq!(result.public_output_count(), 1);
        assert_eq!(result.external_states(), bindings);

        assert_eq!(ReferenceSource::Capture { index: 0 }.flat_input_index(1), Ok(0));
        assert_eq!(ReferenceSource::Input { index: 2 }.flat_input_index(1), Ok(3));
        assert_eq!(
            ReferenceSource::Capture { index: 1 }.flat_input_index(1),
            Err(ProgramError::MalformedProgram(
                "reference source capture 1 lies outside the capture prefix of length 1".to_string(),
            )),
        );
        assert_eq!(
            ReferenceSource::Input { index: usize::MAX }.flat_input_index(1),
            Err(ProgramError::MalformedProgram(format!(
                "reference source input {} overflows the flat boundary after 1 captures",
                usize::MAX,
            ))),
        );

        assert_eq!(
            ReferenceDischargeResult::from_provider_payload(TestPayload::new((), 0, 0), 1, 0, Vec::new()).unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge reports 1 captures but discharged input count is 0".to_string(),
            ),
        );

        assert_eq!(
            ReferenceDischargeResult::from_provider_payload(TestPayload::new((), 0, 1), 0, 2, Vec::new()).unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge reports 2 public outputs but discharged output count is 1".to_string(),
            ),
        );
        assert_eq!(
            ReferenceDischargeResult::from_provider_payload(
                TestPayload::new((), 0, 0),
                0,
                0,
                vec![ReferenceStateBinding::new(ReferenceSource::Input { index: 0 }, None)],
            )
            .unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge state for `input 0` names input 0 but discharged input count is 0".to_string(),
            ),
        );
        assert_eq!(
            ReferenceDischargeResult::from_provider_payload(TestPayload::new((), 1, 1), 0, 0, Vec::new()).unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge final states end at output 0 but discharged output count is 1".to_string(),
            ),
        );
        assert_eq!(
            ReferenceDischargeResult::from_provider_payload(TestPayload::new((), 0, usize::MAX), 0, 0, Vec::new(),)
                .unwrap_err(),
            ProgramError::MalformedProgram(format!(
                "reference discharge final states end at output 0 but discharged output count is {}",
                usize::MAX,
            )),
        );
        for sources in [
            [ReferenceSource::Capture { index: 0 }, ReferenceSource::Capture { index: 0 }],
            [ReferenceSource::Input { index: 0 }, ReferenceSource::Capture { index: 0 }],
        ] {
            let bindings = sources
                .into_iter()
                .enumerate()
                .map(|(_, source)| ReferenceStateBinding::new(source, None))
                .collect();
            assert_eq!(
                ReferenceDischargeResult::from_provider_payload(TestPayload::new((), 2, 0), 1, 0, bindings)
                    .unwrap_err(),
                ProgramError::MalformedProgram(format!(
                    "reference discharge state source `{}` does not follow source `{}` in canonical boundary order",
                    sources[1], sources[0],
                )),
            );
        }
    }

    #[test]
    fn test_reference_discharge_local_references_calls_full_discharge_once_and_preserves_failure_precedence() {
        let calls = Rc::new(Cell::new(0));
        let local = TestDischargeProvider { calls: calls.clone(), mode: TestDischargeMode::Local };
        assert_eq!(local.discharge_local_references(0, "test transform"), Ok(TestPayload::new(7, 0, 0)));
        assert_eq!(calls.get(), 1);

        let external = TestDischargeProvider { calls: calls.clone(), mode: TestDischargeMode::External };
        assert_eq!(
            external.discharge_local_references(0, "test transform"),
            Err(ProgramError::UnsupportedOperation {
                message: "test transform supports only local references, but the program uses external \
                          `input 0`"
                    .to_string(),
            }),
        );
        assert_eq!(calls.get(), 2);

        let malformed = TestDischargeProvider { calls: calls.clone(), mode: TestDischargeMode::Malformed };
        assert_eq!(
            malformed.discharge_local_references(0, "test transform"),
            Err(ProgramError::MalformedProgram(
                "reference discharge final states end at output 0 but discharged output count is 1".to_string(),
            )),
        );
        assert_eq!(calls.get(), 3);
    }

    #[test]
    fn test_partial_reference_discharge_result_reports_only_discharged_bindings() {
        // The partial envelope keeps the canonical discharged boundary of the full envelope, so its accessors report
        // the discharged bindings and the public-output prefix that precedes their hidden final-state suffix.
        let bindings = vec![
            ReferenceStateBinding::new(ReferenceSource::Capture { index: 0 }, Some(1)),
            ReferenceStateBinding::new(ReferenceSource::Input { index: 0 }, None),
        ];
        let result =
            PartialReferenceDischargeResult::new(TestPayload::new("program", 2, 2), 1, 1, bindings.clone()).unwrap();
        assert_eq!(result.program().value, "program");
        assert_eq!(result.capture_count(), 1);
        assert_eq!(result.public_output_count(), 1);
        assert_eq!(result.external_states(), bindings);
        assert_eq!(result.into_parts(), (TestPayload::new("program", 2, 2), 1, 1, bindings));

        // The shared boundary validation applies to the partial envelope exactly as it does to the full one.
        assert_eq!(
            PartialReferenceDischargeResult::new(TestPayload::new((), 0, 1), 0, 2, Vec::new()).unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge reports 2 public outputs but discharged output count is 1".to_string(),
            ),
        );
        assert_eq!(
            PartialReferenceDischargeResult::new(TestPayload::new((), 1, 1), 0, 0, Vec::new()).unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge final states end at output 0 but discharged output count is 1".to_string(),
            ),
        );
    }

    #[test]
    fn test_partial_reference_discharge_result_try_into_full_proves_reference_freedom() {
        // Operation family that separates the two facts the reference-freedom proof must distinguish: an unrelated
        // ordered-state operation that discharge never touches, and a retained reference operation that it must
        // reject even though its boundary types are ordinary.
        #[derive(Copy, Clone, Debug)]
        enum ProofOperation {
            OrderedIo,
            RetainedReference,
        }

        impl Display for ProofOperation {
            fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str(self.name())
            }
        }

        impl Operation for ProofOperation {
            type Type = TestType;

            fn name(&self) -> &'static str {
                match self {
                    Self::OrderedIo => "test.ordered_io",
                    Self::RetainedReference => "test.retained_reference",
                }
            }

            fn infer_output_types(
                &self,
                input_types: &[TestType],
                _region_interfaces: &[RegionInterface<TestType>],
            ) -> Result<Vec<TestType>, TypeError> {
                Ok(input_types.to_vec())
            }

            fn reference_semantics(&self) -> Cow<'_, ReferenceOperationSemantics> {
                match self {
                    Self::OrderedIo => Cow::Borrowed(ReferenceOperationSemantics::empty()),
                    Self::RetainedReference => Cow::Owned(ReferenceOperationSemantics::new(
                        vec![ReferenceInput::new(0, ReferenceAccessMode::Read)],
                        Vec::new(),
                    )),
                }
            }

            fn effects(&self) -> Effects {
                Effects::single(Effect::OrderedIo)
            }
        }

        let program = |operations: &[ProofOperation], input_type: TestType| {
            let mut builder = ProgramBuilder::<TestValue, ProofOperation>::new();
            let mut value = builder.add_input(input_type);
            for operation in operations {
                value = builder.add_instruction(*operation, Vec::new(), vec![value], None).unwrap()[0];
            }
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![value], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let partial = |program| PartialReferenceDischargeResult::new(program, 0, 1, Vec::new()).unwrap();

        // Discharge normalizes references and nothing else, so an unrelated ordered-state operation is proof-neutral
        // and its program converts into the reference-free envelope unchanged.
        let discharged = partial(program(&[ProofOperation::OrderedIo], TestType::Value(0))).try_into_full().unwrap();
        assert_eq!(discharged.public_output_count(), 1);
        assert!(discharged.program().effects().contains(Effect::OrderedIo));

        // A surviving reference-typed value is disqualifying wherever it appears, including on the boundary.
        assert_eq!(
            ReferenceDischargeResult::from_provider_payload(
                program(&[ProofOperation::OrderedIo], reference_type(0)),
                0,
                1,
                Vec::new(),
            )
            .unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge payload still contains a reference-typed value and cannot form a full discharge"
                    .to_string(),
            ),
        );
        assert_eq!(
            partial(program(&[ProofOperation::OrderedIo], reference_type(0))).try_into_full().unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge payload still contains a reference-typed value and cannot form a full discharge"
                    .to_string(),
            ),
        );

        // A retained reference operation is disqualifying even when every value in the program is ordinary.
        assert_eq!(
            partial(program(&[ProofOperation::OrderedIo, ProofOperation::RetainedReference], TestType::Value(0)))
                .try_into_full()
                .unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge payload retains reference operation `test.retained_reference` at `^0[1]` and \
                 cannot form a full discharge"
                    .to_string(),
            ),
        );
    }

    #[test]
    fn test_reference_discharge_sites_enumerate_externals_before_allocations() {
        // A callee region that allocates its own local root, so that enumeration is exercised across the complete
        // attached region closure rather than the entry region alone.
        let mut callee_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let initial = callee_builder.add_input(TestType::Value(0));
        let root = callee_builder.add_instruction(TestOperation::NewRoot, Vec::new(), vec![initial], None).unwrap()[0];
        let frozen = callee_builder.add_instruction(TestOperation::Consume, Vec::new(), vec![root], None).unwrap()[0];
        let callee = callee_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![frozen], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let captured = builder.add_input(reference_type(0));
        let public = builder.add_input(reference_type(0));
        let initial = builder.add_input(TestType::Value(0));
        let callee = builder.import_program(callee);
        let local = builder.add_instruction(TestOperation::NewRoot, Vec::new(), vec![initial], None).unwrap()[0];
        builder.add_instruction(TestOperation::Read, Vec::new(), vec![captured], None).unwrap();
        builder.add_instruction(TestOperation::Read, Vec::new(), vec![public], None).unwrap();
        let called = builder.add_instruction(TestOperation::Call, vec![callee], vec![initial], None).unwrap()[0];

        // The same callee region is attached twice, so its interior allocation must be enumerated once rather than
        // once per invocation.
        let repeated = builder.add_instruction(TestOperation::Call, vec![callee], vec![initial], None).unwrap()[0];
        let frozen = builder.add_instruction(TestOperation::Consume, Vec::new(), vec![local], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![called, repeated, frozen],
                vec![Placeholder; 3],
                vec![Placeholder; 3],
            )
            .unwrap();

        // Externals come first in entry-boundary order, split at the capture prefix, and the allocations follow in
        // arena-coordinate order, including the one inside the shared callee region.
        assert_eq!(
            program.reference_discharge_sites(1),
            Ok(vec![
                ReferenceDischargeSite::External(ReferenceSource::Capture { index: 0 }),
                ReferenceDischargeSite::External(ReferenceSource::Input { index: 0 }),
                ReferenceDischargeSite::Allocation { instruction: InstructionId::new(callee, 0), output_index: 0 },
                ReferenceDischargeSite::Allocation {
                    instruction: InstructionId::new(program.entry(), 0),
                    output_index: 0,
                },
            ]),
        );

        // The capture prefix is the only thing that moves when it changes, and an oversized prefix is rejected.
        assert_eq!(
            program.reference_discharge_sites(0).unwrap()[..2],
            [
                ReferenceDischargeSite::External(ReferenceSource::Input { index: 0 }),
                ReferenceDischargeSite::External(ReferenceSource::Input { index: 1 }),
            ],
        );
        assert_eq!(
            program.reference_discharge_sites(4),
            Err(ProgramError::MalformedProgram(
                "reference discharge site enumeration requests 4 captures but the program has 3 inputs".to_string(),
            )),
        );

        // Every enumerated site validates, and sites render with their kind so diagnostics stay unambiguous.
        let sites = program.reference_discharge_sites(1).unwrap();
        assert_eq!(program.validate_reference_discharge_sites(1, sites.as_slice()), Ok(()));
        assert_eq!(sites[0].to_string(), "external capture 0");
        assert_eq!(sites[3].to_string(), "allocation at `^1[0]` output 0");
    }

    #[test]
    fn test_reference_discharge_site_validation_rejects_malformed_selections() {
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let public = builder.add_input(reference_type(0));
        let initial = builder.add_input(TestType::Value(0));
        let root = builder.add_instruction(TestOperation::NewRoot, Vec::new(), vec![initial], None).unwrap()[0];
        let read = builder.add_instruction(TestOperation::Read, Vec::new(), vec![public], None).unwrap()[0];
        let frozen = builder.add_instruction(TestOperation::Consume, Vec::new(), vec![root], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![read, frozen], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        let entry = program.entry();
        let allocation =
            ReferenceDischargeSite::Allocation { instruction: InstructionId::new(entry, 0), output_index: 0 };
        let reject = |sites: &[ReferenceDischargeSite]| {
            let ProgramError::MalformedProgram(message) =
                program.validate_reference_discharge_sites(0, sites).unwrap_err()
            else {
                panic!("reference discharge site validation must report a malformed program");
            };
            message
        };

        // A repeated site is rejected before any kind check, because a duplicate selection is ambiguous whatever it
        // names.
        assert_eq!(
            reject(&[allocation, allocation]),
            "reference discharge selection names allocation at `^0[0]` output 0 more than once",
        );

        // Every invalid site uses one deterministic diagnostic; validity is defined by the canonical enumerated set.
        assert_eq!(
            reject(&[ReferenceDischargeSite::External(ReferenceSource::Capture { index: 0 })]),
            "reference discharge selection names external capture 0, which is not a selectable site in this program",
        );
        assert_eq!(
            reject(&[ReferenceDischargeSite::External(ReferenceSource::Input { index: 2 })]),
            "reference discharge selection names external input 2, which is not a selectable site in this program",
        );
        assert_eq!(
            reject(&[ReferenceDischargeSite::External(ReferenceSource::Input { index: 1 })]),
            "reference discharge selection names external input 1, which is not a selectable site in this program",
        );
        assert_eq!(
            reject(&[ReferenceDischargeSite::External(ReferenceSource::Input { index: usize::MAX })]),
            format!(
                "reference discharge selection names external input {}, which is not a selectable site in this \
                 program",
                usize::MAX,
            ),
        );

        assert_eq!(
            reject(&[ReferenceDischargeSite::Allocation {
                instruction: InstructionId::new(entry, 7),
                output_index: 0,
            }]),
            "reference discharge selection names allocation at `^0[7]` output 0, which is not a selectable site in \
             this program",
        );
        assert_eq!(
            reject(&[ReferenceDischargeSite::Allocation {
                instruction: InstructionId::new(entry, 1),
                output_index: 0,
            }]),
            "reference discharge selection names allocation at `^0[1]` output 0, which is not a selectable site in \
             this program",
        );
        assert_eq!(
            reject(&[ReferenceDischargeSite::Allocation {
                instruction: InstructionId::new(entry, 0),
                output_index: 1,
            }]),
            "reference discharge selection names allocation at `^0[0]` output 1, which is not a selectable site in \
             this program",
        );
    }

    #[test]
    fn test_reference_discharge_value_reports_operand_kind_mismatches() {
        // A rule that receives the wrong carrier kind gets a diagnostic naming what it expected, which is what keeps
        // an open set of third-party rules diagnosable without each of them inventing its own message.
        let context = ListDischargeContext::new(ListDestination::new());
        let reference_type = ReferenceType::new(ListType { length: 1 });
        let allocated = context.allocate_discharged(reference_type, ListIrValue::List(vec![1])).unwrap();
        let root = allocated.expect_reference("the allocated root").unwrap().root();
        let ordinary: ListDischargeValue = ReferenceDischargeValue::Ordinary(ListIrValue::List(vec![1]));

        assert_eq!(ordinary.expect_ordinary("an update value"), Ok(&ListIrValue::List(vec![1])));
        assert_eq!(
            allocated.expect_ordinary("an update value"),
            Err(ProgramError::MalformedProgram(format!(
                "reference discharge expected an update value but received {root} ref<list<1>>",
            ))),
        );
        assert_eq!(
            ordinary.expect_reference("a reference to read"),
            Err(ProgramError::MalformedProgram(
                "reference discharge expected a reference to read but received an ordinary value".to_string(),
            )),
        );
    }

    #[test]
    fn test_reference_discharge_context_binds_through_the_operation_rule() {
        // Binding dispatches to the operation's own discharge rule, which is the mechanism that gives every
        // operation-backed value capability its unwrapping on an ordinary carrier and its rejection on a reference
        // handle without any bespoke delegation on the tracer.
        let context = ListDischargeContext::new(ListDestination::new());
        let lhs = context.lift(ListIrValue::List(vec![1, 2])).unwrap();
        let rhs = context.lift(ListIrValue::List(vec![10, 20])).unwrap();
        let sum = context.bind(ListOperation::Add, Vec::new(), &[lhs.clone(), rhs.clone()]).unwrap();
        assert_eq!(
            sum.into_iter().map(ReferenceDischargeTracer::into_value).collect::<Vec<_>>(),
            vec![ReferenceDischargeValue::Ordinary(ListIrValue::List(vec![11, 22]))],
        );

        let reference_type = ReferenceType::new(ListType { length: 2 });
        let allocated = context.allocate_discharged(reference_type, ListIrValue::List(vec![1, 2])).unwrap();
        let root = allocated.expect_reference("the allocated root").unwrap().root();
        let reference = ReferenceDischargeTracer::new(context.clone(), allocated);
        assert_eq!(
            context.bind(ListOperation::Add, Vec::new(), &[reference, rhs]),
            Err(ProgramError::MalformedProgram(format!(
                "reference discharge expected an ordinary operand 0 of `list.add` but received {root} ref<list<2>>",
            ))),
        );

        // A direct bind has no source instruction, so an allocation rule that consults its replay position sees
        // `None` and treats the allocation as unconditionally discharged.
        OBSERVED_ALLOCATION_POSITIONS.with_borrow_mut(Vec::clear);
        context.bind(ListOperation::ReferenceNew, Vec::new(), &[lhs]).unwrap();
        assert_eq!(OBSERVED_ALLOCATION_POSITIONS.with_borrow(Vec::clone), vec![None]);
    }

    #[test]
    fn test_reference_discharge_rejects_reference_typed_constants() {
        // A reference stored as a program constant belongs to no root, so lifting it as an ordinary value would let it
        // survive into the destination and break the reference-freedom guarantee. Both lifting paths reject it.
        let reference_type = ReferenceType::new(ListType { length: 2 });
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let stored = builder.add_constant(ListIrValue::Reference(reference_type.clone()));
        let program = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![stored], Vec::new(), vec![Placeholder])
            .unwrap();

        let context = ListDischargeContext::new(ListDestination::new());
        let rejection = ProgramError::MalformedProgram(
            "reference discharge cannot lift a constant of reference type `ref<list<2>>`; a reference enters a \
             program through an input, a capture binding, or an allocation"
                .to_string(),
        );
        assert_eq!(context.lift(ListIrValue::Reference(reference_type)).err(), Some(rejection.clone()));
        let regions = [program];
        let driver = RecursiveReferenceDischargeDriver::new(&regions, None);
        assert_eq!(driver.discharge_region(&context, 0, Vec::new()), Err(rejection));
    }

    #[test]
    fn test_reference_capture_scope() {
        // A scope binds one root per capture position. Positions carrying an ordinary value, positions past the end of
        // the scope, and constants that name no capture position at all all resolve to nothing, which is what leaves
        // an unresolvable reference-typed constant to the rejection at the lift site.
        let context = ListDischargeContext::new(ListDestination::new());
        let allocated = context
            .allocate_discharged(ReferenceType::new(ListType { length: 2 }), ListIrValue::List(vec![1, 2]))
            .unwrap();
        let root = allocated.expect_reference("the captured root").unwrap().root();

        let empty = ReferenceCaptureScope::<ListIrValue>::default();
        assert_eq!(empty.roots(), &[]);
        assert_eq!(empty.resolve(&ListIrValue::Reference(ReferenceType::new(ListType { length: 2 }))), None);

        let scope = ReferenceCaptureScope::new(list_capture_position, vec![None, None, Some(root)]);
        assert_eq!(scope.roots(), &[None, None, Some(root)]);
        assert_eq!(scope.resolve(&ListIrValue::Reference(ReferenceType::new(ListType { length: 2 }))), Some(root));
        assert_eq!(scope.resolve(&ListIrValue::Reference(ReferenceType::new(ListType { length: 1 }))), None);
        assert_eq!(scope.resolve(&ListIrValue::Reference(ReferenceType::new(ListType { length: 9 }))), None);
        assert_eq!(scope.resolve(&ListIrValue::List(vec![1, 2])), None);

        // Rebinding keeps the seam, which is how a nested region's scope and a fork's remapped scope are built.
        let rebound = scope.with_roots(vec![Some(root)]);
        assert_eq!(rebound.roots(), &[Some(root)]);
        assert_eq!(rebound.resolve(&ListIrValue::Reference(ReferenceType::new(ListType { length: 0 }))), Some(root));
    }

    #[test]
    fn test_reference_discharge_lifts_a_capture_scoped_constant_as_its_bound_root() {
        // A capture-lifted program names its caller's references through constants, and such a constant denotes the
        // root that capture position already binds rather than a second root of its own.
        let pair = ReferenceType::new(ListType { length: 2 });
        let triple = ReferenceType::new(ListType { length: 3 });
        let context = ListDischargeContext::new(ListDestination::new());
        let allocated = context.allocate_discharged(pair.clone(), ListIrValue::List(vec![1, 2])).unwrap();
        let root = allocated.expect_reference("the captured root").unwrap().root();
        let scoped =
            context.with_captures(ReferenceCaptureScope::new(list_capture_position, vec![None, None, Some(root)]));

        let lifted = scoped.lift(ListIrValue::Reference(pair.clone())).unwrap();
        let reference = lifted.value().expect_reference("the resolved capture").unwrap();
        assert_eq!(reference.root(), root);
        assert_eq!(reference.r#type(), &pair);
        assert_eq!(scoped.live_roots(), vec![root]);

        // An ordinary constant is unaffected by the scope and lifts through the destination as usual.
        let ordinary = scoped.lift(ListIrValue::List(vec![3, 4])).unwrap();
        assert_eq!(ordinary.value(), &ReferenceDischargeValue::Ordinary(ListIrValue::List(vec![3, 4])));

        // A capture position the scope does not bind keeps the ordinary reference-constant rejection.
        assert_eq!(
            scoped.lift(ListIrValue::Reference(triple.clone())).err(),
            Some(ProgramError::MalformedProgram(
                "reference discharge cannot lift a constant of reference type `ref<list<3>>`; a reference enters a \
                 program through an input, a capture binding, or an allocation"
                    .to_string(),
            )),
        );

        // A capture constant names the whole root its position binds, so a declared type the bound root does not
        // carry is reported rather than silently widened where the constant is used.
        let allocated = context.allocate_discharged(triple, ListIrValue::List(vec![1, 2, 3])).unwrap();
        let wider = allocated.expect_reference("the mismatched root").unwrap().root();
        let mismatched = scoped.with_captures(scoped.captures().with_roots(vec![None, None, Some(wider)]));
        assert_eq!(
            mismatched.lift(ListIrValue::Reference(pair)).err(),
            Some(ProgramError::MalformedProgram(format!(
                "reference discharge resolved a capture constant of reference type `ref<list<2>>` to {wider}, which \
                 carries the reference type `ref<list<3>>`",
            ))),
        );
    }

    #[test]
    fn test_reference_discharge_region_program_rebinds_the_capture_scope_in_fork_roots() {
        // A region whose closure reaches a caller root through a capture constant declares no boundary position for
        // it, so the rule threads it as added state. The fork rebinds the caller's scope onto the fork root standing
        // for that caller root, which is what lets the rebuilt body resolve the very same constant against its own
        // isolated environment.
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let captured = builder.add_constant(ListIrValue::Reference(ReferenceType::new(ListType { length: 2 })));
        let snapshot = builder.add_instruction(ListOperation::Read, Vec::new(), vec![captured], None).unwrap()[0];
        let program = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![snapshot], Vec::new(), vec![Placeholder])
            .unwrap();

        let context = ListDischargeContext::new(ListDestination::new());
        let allocated = context
            .allocate_discharged(ReferenceType::new(ListType { length: 2 }), ListIrValue::List(vec![1, 2]))
            .unwrap();
        let root = allocated.expect_reference("the captured root").unwrap().root();
        let context =
            context.with_captures(ReferenceCaptureScope::new(list_capture_position, vec![None, None, Some(root)]));

        // The summary reports the capture-scoped access in caller-root terms, which is what sizes the boundary.
        let summary = context.region_summary(&ListOperation::Call, 0, program.entry_region_ref(), &[]).unwrap();
        assert_eq!(summary.accessed().collect::<Vec<_>>(), vec![root]);
        assert!(!summary.is_mutated(root));
        assert_eq!(summary.output_roots(), &[None]);

        let regions = [program];
        let driver = RecursiveReferenceDischargeDriver::new(&regions, None);
        let boundary =
            ReferenceRegionDischargeBoundary::new(&ListOperation::Call, 0, Vec::new(), vec![root], 0, Vec::new(), 0);
        let fork = driver.discharge_region_program(&context, 0, &boundary).unwrap();
        assert_eq!(
            fork.program.to_string(),
            indoc! {"
                lambda %0:list<2> .
                let %1:list<2> = list.select %0
                in (%1)"},
        );
        assert_eq!(fork.output_roots(), &[None]);
        assert_eq!(fork.mutated_roots, []);

        // The caller environment is untouched: the fork read its own threaded copy of the state.
        assert_eq!(context.discharged_state(root), Ok(ListIrValue::List(vec![1, 2])));
        assert_eq!(context.is_mutated(root), Ok(false));
    }

    #[test]
    fn test_discharge_reference_free_operation_replays_reference_free_applications() {
        // The shared reference-free replay rule spends the conversion seam from the payload into the destination's
        // family, so an eager destination executes the replay and a staging destination would record it.
        let context = ListDischargeContext::new(ListDestination::new());
        let inputs = vec![
            ReferenceDischargeValue::Ordinary(ListIrValue::List(vec![1, 2])),
            ReferenceDischargeValue::Ordinary(ListIrValue::List(vec![10, 20])),
        ];
        assert_eq!(
            discharge_reference_free_operation(&ListOperation::Add, &context, &EmptyRegionDriver, inputs.as_slice()),
            Ok(vec![ReferenceDischargeValue::Ordinary(ListIrValue::List(vec![11, 22]))]),
        );

        // A region-carrying application whose closure touches a reference is rejected rather than replayed, because
        // how a reference boundary widens is knowledge that belongs to the operation. A reference-free closure instead
        // replays verbatim, which is what lets an operation whose regions hold no state keep this shared rule; here
        // the operation's own contract rejects the attachment, because `list.add` declares no region slots at all.
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let input = builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 1 })));
        let read = builder.add_instruction(ListOperation::Read, Vec::new(), vec![input], None).unwrap()[0];
        let stateful = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![read], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let regions = [stateful];
        let driver = RecursiveReferenceDischargeDriver::new(&regions, None);
        assert_eq!(
            discharge_reference_free_operation(&ListOperation::Add, &context, &driver, inputs.as_slice()),
            Err(ProgramError::UnsupportedOperation {
                message: "`list.add` carries reference state but has no reference discharge rule".to_string(),
            }),
        );

        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let input = builder.add_input(ListIrType::List(ListType { length: 1 }));
        let reference_free = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let regions = [reference_free];
        let driver = RecursiveReferenceDischargeDriver::new(&regions, None);
        assert_eq!(
            discharge_reference_free_operation(&ListOperation::Add, &context, &driver, inputs.as_slice()),
            Err(ProgramError::MalformedProgram(
                "operation `list.add` declares no region slots but 1 regions were attached".to_string(),
            )),
        );

        // An operation that does declare a region slot replays that region into the destination as it stands, which is
        // the whole rewrite for a region-carrying operation whose closure holds no state to thread.
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let callee_input = builder.add_input(ListIrType::List(ListType { length: 2 }));
        let doubled = builder
            .add_instruction(ListOperation::Add, Vec::new(), vec![callee_input, callee_input], None)
            .unwrap()[0];
        let callee = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![doubled], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let regions = [callee];
        let driver = RecursiveReferenceDischargeDriver::new(&regions, None);
        assert_eq!(
            discharge_reference_free_operation(&ListOperation::Call, &context, &driver, &inputs[..1]),
            Ok(vec![ReferenceDischargeValue::Ordinary(ListIrValue::List(vec![2, 4]))]),
        );
    }

    #[test]
    fn test_reference_discharge_drivers_report_their_replay_position() {
        let context = ListDischargeContext::new(ListDestination::new());
        let position = InstructionId::new(RegionId::new(0), 3);

        // A driver built without a source instruction reports none, and one built for a replayed instruction reports
        // exactly that coordinate.
        assert_eq!(
            ReferenceDischargeDriver::<ListDestination, ListReferenceDischarge>::instruction(&EmptyRegionDriver),
            None,
        );
        let driver = RecursiveReferenceDischargeDriver::new(&EmptyRegionDriver, Some(position));
        assert_eq!(
            ReferenceDischargeDriver::<ListDestination, ListReferenceDischarge>::instruction(&driver),
            Some(position),
        );

        // Neither driver can serve a nested region, because neither has one.
        let no_regions = ProgramError::MalformedProgram("empty region driver cannot discharge a region".to_string());
        assert_eq!(EmptyRegionDriver.discharge_region(&context, 0, Vec::new()), Err(no_regions));
        assert_eq!(
            driver.discharge_region(&context, 0, Vec::new()),
            Err(ProgramError::MalformedProgram("region index 0 is out of range".to_string())),
        );
    }

    #[test]
    fn test_list_reference_discharge_policy_applies_composed_aliases() {
        let destination = ListDestination::new();
        let referent = ListType { length: 4 };

        // The identity alias of a root covers its complete referent, and the lift/project pair round-trips a
        // reference type through the destination universe while classifying an ordinary type as not a reference.
        let root_alias = <ListReferenceDischarge as ReferenceDischargePolicy<ListDestination>>::root_alias(&referent);
        assert_eq!(root_alias, ListAlias { offset: 0, length: 4 });
        let reference_type = ReferenceType::new(referent.clone());
        let lifted = <ListReferenceDischarge as ReferenceDischargePolicy<ListDestination>>::lift_reference_type(
            reference_type.clone(),
        );
        assert_eq!(lifted, ListIrType::Reference(reference_type.clone()));
        assert_eq!(
            <ListReferenceDischarge as ReferenceDischargePolicy<ListDestination>>::project_reference_type(&lifted),
            Some(reference_type),
        );
        assert_eq!(
            <ListReferenceDischarge as ReferenceDischargePolicy<ListDestination>>::project_reference_type(
                &ListIrType::List(referent),
            ),
            None,
        );

        // A composed alias selects only its own coordinates on every access, and replacement and accumulation return
        // the successor state of the whole root rather than of the selection.
        let current = ListIrValue::List(vec![1, 2, 3, 4]);
        let view = ListAlias { offset: 1, length: 2 };
        assert_eq!(ListReferenceDischarge::read(&destination, &current, &view), Ok(ListIrValue::List(vec![2, 3])));
        assert_eq!(
            ListReferenceDischarge::replace(&destination, &current, ListIrValue::List(vec![20, 30]), &view),
            Ok((ListIrValue::List(vec![2, 3]), ListIrValue::List(vec![1, 20, 30, 4]))),
        );
        assert_eq!(
            ListReferenceDischarge::accumulate(&destination, &current, ListIrValue::List(vec![20, 30]), &view),
            Ok(ListIrValue::List(vec![1, 22, 33, 4])),
        );

        // The policy reports the universe's own failures instead of silently widening a selection.
        assert_eq!(
            ListReferenceDischarge::read(&destination, &current, &ListAlias { offset: 3, length: 2 }),
            Err(ProgramError::MalformedProgram("selection [3, 5) does not fit a list of length 4".to_string())),
        );
        assert_eq!(
            ListReferenceDischarge::replace(&destination, &current, ListIrValue::List(vec![20]), &view),
            Ok((ListIrValue::List(vec![2, 3]), ListIrValue::List(vec![1, 20, 3, 4]))),
        );
    }

    #[test]
    fn test_reference_discharge_context_threads_discharged_root_state() {
        let context = ListDischargeContext::new(ListDestination::new());
        let reference_type = ReferenceType::new(ListType { length: 4 });
        let allocated =
            context.allocate_discharged(reference_type.clone(), ListIrValue::List(vec![1, 2, 3, 4])).unwrap();
        let reference = allocated.expect_reference("the allocated root").unwrap().clone();
        let root = reference.root();

        // A fresh root starts unmutated, exposes its identity alias and reference type, and carries no destination
        // reference value because it was discharged rather than preserved.
        assert_eq!(context.live_roots(), vec![root]);
        assert_eq!(context.is_mutated(root), Ok(false));
        assert_eq!(reference.alias(), &ListAlias { offset: 0, length: 4 });
        assert_eq!(reference.r#type(), &reference_type);
        assert_eq!(reference.preserved(), None);
        assert_eq!(context.read(&reference), Ok(ListIrValue::List(vec![1, 2, 3, 4])));

        // A derived handle narrows the view without touching the root's identity, and its accesses act only on the
        // coordinates it selects.
        let view = context
            .derive(&reference, ListAlias { offset: 1, length: 2 }, ReferenceType::new(ListType { length: 2 }), None)
            .unwrap();
        let view = view.expect_reference("the derived view").unwrap().clone();
        assert_eq!(view.root(), root);
        assert_eq!(context.read(&view), Ok(ListIrValue::List(vec![2, 3])));
        assert_eq!(context.replace(&view, ListIrValue::List(vec![20, 30])), Ok(ListIrValue::List(vec![2, 3])));
        assert_eq!(context.read(&reference), Ok(ListIrValue::List(vec![1, 20, 30, 4])));
        assert_eq!(context.is_mutated(root), Ok(true));
        assert_eq!(context.accumulate(&view, ListIrValue::List(vec![1, 1])), Ok(()));
        assert_eq!(context.read(&reference), Ok(ListIrValue::List(vec![1, 21, 31, 4])));

        // Consumption is a whole-root event. Provenance, not type equality, distinguishes the root handle from a
        // derived view: a policy may derive a view whose referent happens to have the root's exact type.
        let same_type_view = context
            .derive(&reference, ListAlias { offset: 0, length: 4 }, reference_type.clone(), None)
            .unwrap();
        let same_type_view = same_type_view.expect_reference("the same-type derived view").unwrap();
        assert_eq!(
            context
                .operand_root(&ReferenceDischargeValue::Reference(same_type_view.clone()), ListOperation::Call.name(),),
            Err(ProgramError::MalformedProgram(format!(
                "operation `list.call` passes the derived view `ref<list<4>>` of {root} across a region boundary, \
                 which carries the whole root `ref<list<4>>`; derive the view inside the region instead",
            ))),
        );
        assert_eq!(
            context.consume(same_type_view),
            Err(ProgramError::MalformedProgram(format!(
                "reference discharge cannot consume {root} through the derived view `ref<list<4>>`; consumption \
                 yields the whole root, whose referent is `list<4>`",
            ))),
        );

        // A narrower derived view is rejected by the same provenance check rather than silently yielding the whole
        // root's value under the view's type.
        assert_eq!(
            context.consume(&view),
            Err(ProgramError::MalformedProgram(format!(
                "reference discharge cannot consume {root} through the derived view `ref<list<2>>`; consumption \
                 yields the whole root, whose referent is `list<4>`",
            ))),
        );

        // Through the root handle it yields the complete state and unbinds the root, so every later access through
        // any handle of that root is reported against the exact root.
        assert_eq!(context.consume(&reference), Ok(ListIrValue::List(vec![1, 21, 31, 4])));
        assert_eq!(context.live_roots(), Vec::new());
        let consumed = ProgramError::MalformedProgram(format!("reference discharge accessed consumed {root}"));
        assert_eq!(context.read(&reference), Err(consumed.clone()));
        assert_eq!(context.root_reference_type(root), Err(consumed.clone()));
        assert_eq!(context.set_discharged_state(root, ListIrValue::List(vec![0; 4])), Err(consumed));

        // A handle minted by an unrelated discharge is reported instead of silently addressing whichever root
        // occupies the same position here.
        let other = ListDischargeContext::new(ListDestination::new());
        let foreign = other.allocate_discharged(reference_type, ListIrValue::List(vec![0; 4])).unwrap();
        let foreign = foreign.expect_reference("the unrelated root").unwrap().root();
        let prefix =
            format!("reference discharge accessed {foreign}, which belongs to an environment other than the active");
        assert!(matches!(
            context.root_reference_type(foreign),
            Err(ProgramError::MalformedProgram(message)) if message.starts_with(&prefix),
        ));
    }

    #[test]
    fn test_reference_discharge_context_validates_root_state_types_before_mutation() {
        let context = ListDischargeContext::new(ListDestination::new());
        let reference_type = ReferenceType::new(ListType { length: 2 });
        let wrong_state = ListIrValue::List(vec![1]);
        let error = ProgramError::MalformedProgram(
            "reference discharge state has type `list<1>` but root `ref<list<2>>` requires `list<2>`".to_string(),
        );

        // A malformed allocation is rejected before a root is inserted into the environment.
        assert_eq!(context.allocate_discharged(reference_type.clone(), wrong_state.clone()), Err(error.clone()));
        assert_eq!(context.live_roots(), Vec::new());

        let allocated = context.allocate_discharged(reference_type, ListIrValue::List(vec![1, 2])).unwrap();
        let reference = allocated.expect_reference("the allocated root").unwrap();
        let root = reference.root();

        // Both state-installation paths validate before taking the mutable environment borrow, so failure preserves
        // the prior state and mutation bit.
        assert_eq!(context.set_discharged_state(root, wrong_state.clone()), Err(error.clone()));
        assert_eq!(context.read(reference), Ok(ListIrValue::List(vec![1, 2])));
        assert_eq!(context.is_mutated(root), Ok(false));
        assert_eq!(context.merge_discharged_state(root, wrong_state, true), Err(error));
        assert_eq!(context.read(reference), Ok(ListIrValue::List(vec![1, 2])));
        assert_eq!(context.is_mutated(root), Ok(false));
    }

    #[test]
    fn test_reference_discharge_context_clones_share_root_state() {
        let context = ListDischargeContext::new(ListDestination::new());
        let reference_type = ReferenceType::new(ListType { length: 2 });
        let allocated = context.allocate_discharged(reference_type, ListIrValue::List(vec![1, 2])).unwrap();
        let reference = allocated.expect_reference("the allocated root").unwrap().clone();

        // A clone shares the environment rather than copying it, which is the contract every stateful Ryft context
        // follows: several handles can denote one root, and every one of them must observe the same current state.
        // Isolation is therefore never implicit — a structured rule that must not commit rebuilds its region against
        // an environment of its own through `discharge_region_program`.
        let clone = context.clone();
        clone.accumulate(&reference, ListIrValue::List(vec![10, 10])).unwrap();
        assert_eq!(context.read(&reference), Ok(ListIrValue::List(vec![11, 12])));
        context.accumulate(&reference, ListIrValue::List(vec![1, 1])).unwrap();
        assert_eq!(clone.read(&reference), Ok(ListIrValue::List(vec![12, 13])));
    }

    #[test]
    fn test_reference_discharge_context_binds_preserved_roots() {
        let context = ListDischargeContext::new(ListDestination::new());
        let reference_type = ReferenceType::new(ListType { length: 2 });
        let destination_reference = ListIrValue::Reference(reference_type.clone());
        let bound = context.bind_preserved(reference_type.clone(), destination_reference.clone()).unwrap();
        let reference = bound.expect_reference("the preserved root").unwrap().clone();
        let root = reference.root();

        // A preserved root keeps its destination reference value on the handle, so a later access can replay
        // verbatim instead of re-deriving the handle.
        assert_eq!(context.root_reference_type(root), Ok(reference_type));
        assert_eq!(reference.preserved(), Some(&destination_reference));
        assert_eq!(
            context.operand_value(&ReferenceDischargeValue::Reference(reference.clone())),
            Ok(destination_reference.clone()),
        );

        // Every discharged-state service rejects a preserved root by name rather than silently treating it as state.
        assert_eq!(
            context.read(&reference),
            Err(ProgramError::MalformedProgram(format!(
                "reference discharge requested the discharged state of preserved {root}",
            ))),
        );
        assert_eq!(
            context.is_mutated(root),
            Err(ProgramError::MalformedProgram(format!("reference discharge queried mutation of preserved {root}"))),
        );
        assert_eq!(
            context.set_discharged_state(root, ListIrValue::List(vec![0, 0])),
            Err(ProgramError::MalformedProgram(format!("reference discharge installed state into preserved {root}"))),
        );

        // Deriving a handle must agree with the root's state about whether a destination reference value exists.
        let view_type = ReferenceType::new(ListType { length: 1 });
        let view_alias = ListAlias { offset: 0, length: 1 };
        assert_eq!(
            context.derive(&reference, view_alias, view_type.clone(), None),
            Err(ProgramError::MalformedProgram(format!(
                "reference discharge derived a handle from {root} without a destination reference value, but that \
                 root is preserved",
            ))),
        );
        let view = context
            .derive(&reference, view_alias, view_type.clone(), Some(ListIrValue::Reference(view_type.clone())))
            .unwrap();
        let view = view.expect_reference("the derived view").unwrap();
        assert_eq!(view.root(), root);
        assert_eq!(view.preserved(), Some(&ListIrValue::Reference(view_type)));
    }

    #[test]
    fn test_reference_discharge_tracer_reports_its_type_and_delegates_concretization() {
        let context = ListDischargeContext::new(ListDestination::new());
        let ordinary = context.lift(ListIrValue::List(vec![1])).unwrap();
        let reference_type = ReferenceType::new(ListType { length: 2 });
        let allocated = context.allocate_discharged(reference_type.clone(), ListIrValue::List(vec![1, 2])).unwrap();
        let reference = ReferenceDischargeTracer::new(context.clone(), allocated);

        // An ordinary carrier reports the wrapped destination value's type, while a reference handle reports its own
        // reference type lifted into the destination universe.
        assert_eq!(ordinary.r#type().into_owned(), ListIrType::List(ListType { length: 1 }));
        assert_eq!(reference.r#type().into_owned(), ListIrType::Reference(reference_type));
        assert_eq!(ordinary.to_string(), "[1]");
        let root = reference.value().expect_reference("the allocated root").unwrap().root();
        assert_eq!(reference.to_string(), format!("{root} ref<list<2>>"));
        assert_eq!(
            format!("{reference:?}"),
            format!(
                "ReferenceDischargeTracer {{ value: Reference(ReferenceDischargeReference {{ root: {root:?}, \
                 denotes_whole_root: true, alias: ListAlias {{ offset: 0, length: 2 }}, type: ReferenceType {{ \
                 referent: ListType {{ length: 2 }} }}, preserved: None }}), .. }}",
            ),
        );

        // Equality compares carriers and ignores the context stamp, mirroring the batching tracer.
        assert_eq!(
            ordinary,
            ReferenceDischargeTracer::new(
                context.clone(),
                ReferenceDischargeValue::Ordinary(ListIrValue::List(vec![1]))
            ),
        );
        assert_ne!(ordinary, reference);

        // Concretization delegates on an ordinary carrier and is rejected on a live reference handle, because a
        // handle is state bookkeeping rather than a value with observable contents.
        assert_eq!(Concretizable::<bool>::concretize(&ordinary), Ok(true));
        assert_eq!(
            Concretizable::<bool>::concretize(&reference),
            Err(ProgramError::Concretization {
                message: format!("cannot extract a concrete value from {root} ref<list<2>>"),
            }),
        );

        // The tracer dispatches through the context that owns its root environment, and reports the destination's
        // execution mode rather than inventing one of its own.
        assert!(ordinary.dispatch_domain().is_eager());
        assert!(ordinary.execution_domain().is_eager());
        assert_eq!(context.resolve(&ordinary), ValueResolution::Constant(ListIrValue::List(vec![1])));
        assert_eq!(context.resolve(&reference), ValueResolution::Opaque);
    }

    #[test]
    fn test_reference_discharge_rules_thread_state_through_a_replayed_program() {
        // The program allocates one local root, narrows it to a composed view, accumulates into that view, replaces
        // it, adds the replaced and current selections, and finally freezes the whole root.
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let initial = builder.add_input(ListIrType::List(ListType { length: 4 }));
        let root = builder.add_instruction(ListOperation::ReferenceNew, Vec::new(), vec![initial], None).unwrap()[0];
        let view = builder
            .add_instruction(ListOperation::Slice { offset: 1, length: 2 }, Vec::new(), vec![root], None)
            .unwrap()[0];
        let update = builder.add_constant(ListIrValue::List(vec![10, 20]));
        builder.add_instruction(ListOperation::AddUpdate, Vec::new(), vec![view, update], None).unwrap();
        let replacement = builder.add_constant(ListIrValue::List(vec![7, 8]));
        builder.add_instruction(ListOperation::Write, Vec::new(), vec![view, replacement], None).unwrap();
        let replaced =
            builder.add_instruction(ListOperation::Swap, Vec::new(), vec![view, replacement], None).unwrap()[0];
        let snapshot = builder.add_instruction(ListOperation::Read, Vec::new(), vec![view], None).unwrap()[0];
        let total = builder.add_instruction(ListOperation::Add, Vec::new(), vec![replaced, snapshot], None).unwrap()[0];
        let frozen = builder.add_instruction(ListOperation::Freeze, Vec::new(), vec![root], None).unwrap()[0];
        let program = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![total, frozen], vec![Placeholder], vec![Placeholder; 2])
            .unwrap();

        // Replaying the program in the discharge context rewrites every reference primitive into ordinary state
        // threading, so the outputs are the values an eager reference execution would have produced.
        OBSERVED_ALLOCATION_POSITIONS.with_borrow_mut(Vec::clear);
        let context = ListDischargeContext::new(ListDestination::new());
        let input = context.lift(ListIrValue::List(vec![1, 2, 3, 4])).unwrap();
        let outputs = program.interpret_in_context(&context, vec![input]).unwrap();
        assert_eq!(
            outputs.into_iter().map(ReferenceDischargeTracer::into_value).collect::<Vec<_>>(),
            vec![
                ReferenceDischargeValue::Ordinary(ListIrValue::List(vec![14, 16])),
                ReferenceDischargeValue::Ordinary(ListIrValue::List(vec![1, 7, 8, 4])),
            ],
        );

        // Every root the program created is gone once its `freeze` consumed it, so nothing leaks into the context.
        assert_eq!(context.live_roots(), Vec::new());

        // Replay through `interpret_in_context` binds each instruction without its source coordinate, which is why
        // an allocation rule that consults its replay position sees `None` on this path.
        assert_eq!(OBSERVED_ALLOCATION_POSITIONS.with_borrow(Vec::clone), vec![None]);
    }

    #[test]
    fn test_reference_region_summary_unions_exact_access_modes() {
        let context = ListDischargeContext::new(ListDestination::new());
        let allocated = context
            .allocate_discharged(ReferenceType::new(ListType { length: 2 }), ListIrValue::List(vec![1, 2]))
            .unwrap();
        let root = allocated.expect_reference("the caller root").unwrap().root();
        let mut left = ReferenceRegionSummary::default();
        left.record(root, ReferenceAccessMode::Read, "list.read").unwrap();
        left.record(root, ReferenceAccessMode::ReadWrite, "list.swap").unwrap();
        left.output_roots = vec![Some(root)];
        let mut right = ReferenceRegionSummary::default();
        right.record(root, ReferenceAccessMode::Write, "list.write").unwrap();
        right.record(root, ReferenceAccessMode::Accumulate, "list.add_update").unwrap();
        right.output_roots = vec![None];

        let merged = left.merged(&right);
        assert_eq!(
            merged.access_modes(root).collect::<Vec<_>>(),
            vec![
                ReferenceAccessMode::Read,
                ReferenceAccessMode::Write,
                ReferenceAccessMode::ReadWrite,
                ReferenceAccessMode::Accumulate,
            ],
        );
        assert!(merged.is_mutated(root));
        assert_eq!(merged.output_roots(), [Some(root)]);
    }

    #[test]
    fn test_reference_region_summary_validates_each_exact_access_mode() {
        let context = ListDischargeContext::new(ListDestination::new());
        let allocated = context
            .allocate_discharged(ReferenceType::new(ListType { length: 2 }), ListIrValue::List(vec![1, 2]))
            .unwrap();
        let root = allocated.expect_reference("the caller root").unwrap().root();
        let modes = [
            ReferenceAccessMode::Read,
            ReferenceAccessMode::Write,
            ReferenceAccessMode::ReadWrite,
            ReferenceAccessMode::Accumulate,
        ];

        for accessed in modes {
            let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
            let reference = builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
            let replacement = builder.add_input(ListIrType::List(ListType { length: 2 }));
            match accessed {
                ReferenceAccessMode::Read => {
                    builder.add_instruction(ListOperation::Read, Vec::new(), vec![reference], None).unwrap();
                }
                ReferenceAccessMode::Write => {
                    builder
                        .add_instruction(ListOperation::Write, Vec::new(), vec![reference, replacement], None)
                        .unwrap();
                }
                ReferenceAccessMode::ReadWrite => {
                    builder
                        .add_instruction(ListOperation::Swap, Vec::new(), vec![reference, replacement], None)
                        .unwrap();
                }
                ReferenceAccessMode::Accumulate => {
                    builder
                        .add_instruction(ListOperation::AddUpdate, Vec::new(), vec![reference, replacement], None)
                        .unwrap();
                }
                ReferenceAccessMode::Consume => unreachable!(),
            }
            let region = builder
                .build::<Vec<ListIrValue>, Vec<ListIrValue>>(Vec::new(), vec![Placeholder; 2], Vec::new())
                .unwrap();

            for allowed in modes {
                let result = context.region_summary(
                    &SingleModeRegionOperation(allowed),
                    0,
                    region.entry_region_ref(),
                    &[Some(root), None],
                );
                if allowed == accessed {
                    let summary = result.unwrap();
                    assert_eq!(summary.access_modes(root).collect::<Vec<_>>(), vec![accessed]);
                    assert_eq!(
                        summary.is_mutated(root),
                        matches!(
                            accessed,
                            ReferenceAccessMode::Write
                                | ReferenceAccessMode::ReadWrite
                                | ReferenceAccessMode::Accumulate,
                        ),
                    );
                } else {
                    assert_eq!(
                        result,
                        Err(ProgramError::MalformedProgram(format!(
                            "operation `test.single_mode_region` does not allow region 0 to access {root} with mode \
                             `{accessed}`",
                        ))),
                    );
                }
            }
        }

        // A nested call's swap remains `ReadWrite` at the outer policy boundary; permitting `Write` cannot admit it
        // through a lossy generic mutation fact.
        let mut callee_builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let reference = callee_builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let replacement = callee_builder.add_input(ListIrType::List(ListType { length: 2 }));
        callee_builder
            .add_instruction(ListOperation::Swap, Vec::new(), vec![reference, replacement], None)
            .unwrap();
        let callee = callee_builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(Vec::new(), vec![Placeholder; 2], Vec::new())
            .unwrap();
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let reference = builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let replacement = builder.add_input(ListIrType::List(ListType { length: 2 }));
        let callee = builder.import_program(callee);
        builder
            .add_instruction(ListOperation::Call, vec![callee], vec![reference, replacement], None)
            .unwrap();
        let region = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(Vec::new(), vec![Placeholder; 2], Vec::new())
            .unwrap();
        assert_eq!(
            context.region_summary(
                &SingleModeRegionOperation(ReferenceAccessMode::Write),
                0,
                region.entry_region_ref(),
                &[Some(root), None],
            ),
            Err(ProgramError::MalformedProgram(format!(
                "operation `test.single_mode_region` does not allow region 0 to access {root} with mode `read/write`",
            ))),
        );
    }

    #[test]
    fn test_reference_region_summary_reports_transitive_accesses_and_output_roots() {
        // A callee that replaces the state of the reference it receives, so the outer region's access to that root is
        // transitive rather than local.
        let mut callee_builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let callee_reference =
            callee_builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let replacement = callee_builder.add_input(ListIrType::List(ListType { length: 2 }));
        let previous = callee_builder
            .add_instruction(ListOperation::Swap, Vec::new(), vec![callee_reference, replacement], None)
            .unwrap()[0];
        let callee = callee_builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![previous], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        // The outer region reads the caller's root directly, replaces it through the callee, and separately allocates,
        // reads, and returns a root of its own.
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let reference = builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let replacement = builder.add_input(ListIrType::List(ListType { length: 2 }));
        let callee = builder.import_program(callee);
        let snapshot = builder.add_instruction(ListOperation::Read, Vec::new(), vec![reference], None).unwrap()[0];
        let local = builder.add_instruction(ListOperation::ReferenceNew, Vec::new(), vec![snapshot], None).unwrap()[0];
        let local_snapshot = builder.add_instruction(ListOperation::Read, Vec::new(), vec![local], None).unwrap()[0];
        let previous = builder
            .add_instruction(ListOperation::Call, vec![callee], vec![reference, replacement], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(
                vec![reference, local, snapshot, local_snapshot, previous],
                vec![Placeholder; 2],
                vec![Placeholder; 5],
            )
            .unwrap();

        let context = ListDischargeContext::new(ListDestination::new());
        let allocated = context
            .allocate_discharged(ReferenceType::new(ListType { length: 2 }), ListIrValue::List(vec![1, 2]))
            .unwrap();
        let root = allocated.expect_reference("the caller root").unwrap().root();
        let summary = context
            .region_summary(&ListOperation::Call, 0, program.entry_region_ref(), &[Some(root), None])
            .unwrap();

        // The caller root is reported as mutated because the nested callee replaces it, while the region's own
        // allocation crosses no boundary and is therefore absent from the summary entirely.
        assert_eq!(summary.accessed().collect::<Vec<_>>(), vec![root]);
        assert_eq!(
            summary.access_modes(root).collect::<Vec<_>>(),
            vec![ReferenceAccessMode::Read, ReferenceAccessMode::ReadWrite],
        );
        assert!(summary.has_access(root, ReferenceAccessMode::Read));
        assert!(summary.has_access(root, ReferenceAccessMode::ReadWrite));
        assert!(!summary.has_access(root, ReferenceAccessMode::Write));
        assert!(summary.is_mutated(root));

        // A declared output resolves to the caller root it denotes: the first output returns the root itself, the
        // second returns a region-local allocation, and the remaining three are ordinary values.
        assert_eq!(summary.output_roots(), &[Some(root), None, None, None, None]);
    }

    #[test]
    fn test_reference_region_summary_rejects_a_closure_that_consumes_a_caller_root() {
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let reference = builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let frozen = builder.add_instruction(ListOperation::Freeze, Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![frozen], vec![Placeholder], vec![Placeholder])
            .unwrap();

        // A consumed root has no successor state, so no state boundary can describe what became of it. The summary
        // rejects that outright rather than letting the caller keep threading state that is no longer live.
        let context = ListDischargeContext::new(ListDestination::new());
        let allocated = context
            .allocate_discharged(ReferenceType::new(ListType { length: 2 }), ListIrValue::List(vec![1, 2]))
            .unwrap();
        let root = allocated.expect_reference("the caller root").unwrap().root();
        assert_eq!(
            context.region_summary(&ListOperation::Call, 0, program.entry_region_ref(), &[Some(root)]),
            Err(ProgramError::MalformedProgram(format!(
                "reference discharge cannot pass {root} into a region that consumes it through `list.freeze`",
            ))),
        );
    }

    #[test]
    fn test_reference_discharge_preserves_aliasing_between_repeated_declared_region_roots() {
        // Both declared callee inputs denote one caller root. A write through the first must therefore be visible to
        // a read through the second even though the rebuilt boundary retains both declared positions.
        let mut callee_builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let written = callee_builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let observed = callee_builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let replacement = callee_builder.add_constant(ListIrValue::List(vec![7, 8]));
        callee_builder
            .add_instruction(ListOperation::Write, Vec::new(), vec![written, replacement], None)
            .unwrap();
        let snapshot =
            callee_builder.add_instruction(ListOperation::Read, Vec::new(), vec![observed], None).unwrap()[0];
        let callee = callee_builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![snapshot], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let reference = builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let callee = builder.import_program(callee);
        let snapshot = builder
            .add_instruction(ListOperation::Call, vec![callee], vec![reference, reference], None)
            .unwrap()[0];
        let source = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![snapshot], vec![Placeholder], vec![Placeholder])
            .unwrap();

        // Full discharge turns the shared root into state. The public snapshot and hidden final-state output both
        // observe the write, proving that the duplicate boundary position did not mint an independent fork root.
        let discharged = source.clone().discharge_references_with_policy::<ListReferenceDischarge>(0).unwrap();
        assert_eq!(
            discharged.program().interpret(vec![ListIrValue::List(vec![1, 2])]),
            Ok(vec![ListIrValue::List(vec![7, 8]), ListIrValue::List(vec![7, 8])]),
        );

        // Partial discharge preserves the same alias as a reference. Both declared positions remain present in the
        // callee boundary, but its second input is unused and both accesses replay through the first canonical value.
        let preserved = source.partially_discharge_references_with_policy::<ListReferenceDischarge>(0, &[]).unwrap();
        assert_eq!(
            preserved.program().to_string(),
            indoc! {"
                lambda %0:ref<list<2>> .
                let %1:list<2> = list.call %0 %0 [
                    callee={
                        lambda %0:ref<list<2>>, %1:ref<list<2>> .
                        let %2:list<2> = const [7, 8]
                            list.write %0 %2
                            %3:list<2> = list.read %0
                        in (%3)
                    },
                ]
                in (%1)"},
        );
    }

    #[test]
    fn test_reference_discharge_threads_a_capture_scoped_root_a_nested_region_only_receives() {
        // A closure can reach a capture-scoped root without ever accessing it, by passing the constant into a nested
        // region that ignores it. The replay still materializes the constant, because something consumes it, so the
        // root has to be threaded even though no reference access records it. In particular, materializing the
        // capture must not invent a semantic read that the enclosing operation's region policy could reject.
        let mut callee_builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let ignored = callee_builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let forwarded = callee_builder.add_input(ListIrType::List(ListType { length: 2 }));
        let callee = callee_builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![forwarded], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        assert!(callee.entry_region_ref().input_ids().contains(&ignored));

        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let callee = builder.import_program(callee);
        let captured = builder.add_constant(ListIrValue::Reference(ReferenceType::new(ListType { length: 2 })));
        let value = builder.add_input(ListIrType::List(ListType { length: 2 }));
        let forwarded =
            builder.add_instruction(ListOperation::Call, vec![callee], vec![captured, value], None).unwrap()[0];
        let program = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![forwarded], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let context = ListDischargeContext::new(ListDestination::new());
        let allocated = context
            .allocate_discharged(ReferenceType::new(ListType { length: 2 }), ListIrValue::List(vec![1, 2]))
            .unwrap();
        let root = allocated.expect_reference("the captured root").unwrap().root();
        let context =
            context.with_captures(ReferenceCaptureScope::new(list_capture_position, vec![None, None, Some(root)]));

        // The enclosing policy accepts writes only. Capture reachability still sizes the boundary, while the exact
        // access summary remains empty because neither closure semantically accesses the root.
        let summary = context
            .region_summary(
                &SingleModeRegionOperation(ReferenceAccessMode::Write),
                0,
                program.entry_region_ref(),
                &[None],
            )
            .unwrap();
        assert_eq!(summary.accessed().collect::<Vec<_>>(), Vec::<ReferenceRootHandle>::new());
        assert_eq!(summary.access_modes(root).collect::<Vec<_>>(), Vec::<ReferenceAccessMode>::new());
        assert!(!summary.is_mutated(root));
        assert_eq!(context.threaded_state_roots(&summary, "test.single_mode_region"), Ok(BTreeSet::from([root])),);

        // The rebuilt region therefore receives the root's entering state and hands it to its own callee.
        let regions = [program];
        let driver = RecursiveReferenceDischargeDriver::new(&regions, None);
        let boundary =
            ReferenceRegionDischargeBoundary::new(&ListOperation::Call, 0, vec![None], vec![root], 1, Vec::new(), 0);
        let fork = driver.discharge_region_program(&context, 0, &boundary).unwrap();
        assert_eq!(
            fork.program.to_string(),
            indoc! {"
                lambda %0:list<2>, %1:list<2> .
                let %2:list<2> = list.call %1 %0 [
                    callee={
                        lambda %0:list<2>, %1:list<2> .
                        in (%1)
                    },
                ]
                in (%2)"},
        );
        assert_eq!(fork.mutated_roots, []);
    }

    #[test]
    fn test_positional_region_discharge_recovers_a_returned_capture_scoped_root() {
        // This root reaches the region through its inherited capture scope, not through any forwarded operand. The
        // declared result must therefore be recovered from the context rather than from the empty operand list.
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let captured = builder.add_constant(ListIrValue::Reference(ReferenceType::new(ListType { length: 2 })));
        let program = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![captured], Vec::new(), vec![Placeholder])
            .unwrap();

        let context = ListDischargeContext::new(ListDestination::new());
        let allocated = context
            .allocate_discharged(ReferenceType::new(ListType { length: 2 }), ListIrValue::List(vec![1, 2]))
            .unwrap();
        let root = allocated.expect_reference("the capture-scoped root").unwrap().root();
        let context =
            context.with_captures(ReferenceCaptureScope::new(list_capture_position, vec![None, None, Some(root)]));
        let regions = [program];
        let driver = RecursiveReferenceDischargeDriver::new(&regions, None);

        let results = discharge_positional_region_operation(&ListOperation::Call, &context, &driver, &[], 0).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].expect_reference("the returned capture-scoped root").unwrap().root(), root);
        assert_eq!(
            context.read(results[0].expect_reference("the returned capture-scoped root").unwrap()),
            Ok(ListIrValue::List(vec![1, 2]),)
        );
        assert_eq!(context.is_mutated(root), Ok(false));
    }

    #[test]
    fn test_positional_region_discharge_recovers_a_returned_preserved_capture_scoped_root() {
        let reference_type = ReferenceType::new(ListType { length: 2 });
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let captured = builder.add_constant(ListIrValue::Reference(reference_type.clone()));
        let program = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![captured], Vec::new(), vec![Placeholder])
            .unwrap();

        let context = ListDischargeContext::new(ListDestination::new());
        let destination_reference = ListIrValue::Reference(reference_type.clone());
        let preserved = context.bind_preserved(reference_type, destination_reference.clone()).unwrap();
        let root = preserved.expect_reference("the preserved capture-scoped root").unwrap().root();
        let context =
            context.with_captures(ReferenceCaptureScope::new(list_capture_position, vec![None, None, Some(root)]));
        let regions = [program];
        let driver = RecursiveReferenceDischargeDriver::new(&regions, None);

        let results = discharge_positional_region_operation(&ListOperation::Call, &context, &driver, &[], 0).unwrap();
        assert_eq!(results.len(), 1);
        let returned = results[0].expect_reference("the returned preserved capture-scoped root").unwrap();
        assert_eq!(returned.root(), root);
        assert_eq!(returned.preserved(), Some(&destination_reference));
        assert_eq!(context.operand_value(&results[0]), Ok(destination_reference));
    }

    #[test]
    fn test_region_discharge_rejects_a_same_type_derived_root_output() {
        let reference_type = ReferenceType::new(ListType { length: 2 });
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let reference = builder.add_input(ListIrType::Reference(reference_type.clone()));
        let view = builder
            .add_instruction(ListOperation::Slice { offset: 0, length: 2 }, Vec::new(), vec![reference], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![view], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let context = ListDischargeContext::new(ListDestination::new());
        let allocated = context.allocate_discharged(reference_type, ListIrValue::List(vec![1, 2])).unwrap();
        let root = allocated.expect_reference("the caller root").unwrap().root();
        let regions = [program];
        let driver = RecursiveReferenceDischargeDriver::new(&regions, None);
        let boundary = ReferenceRegionDischargeBoundary::new(
            &ListOperation::Call,
            0,
            vec![Some(root)],
            Vec::new(),
            1,
            Vec::new(),
            1,
        );

        assert_eq!(
            driver.discharge_region_program(&context, 0, &boundary).unwrap_err(),
            ProgramError::MalformedProgram(format!(
                "reference discharge cannot publish the derived view `ref<list<2>>` of {root} from region `{}`, \
                 whose boundary carries the whole root `ref<list<2>>`",
                regions[0].entry_region_ref().id(),
            )),
        );
    }

    #[test]
    fn test_reference_region_discharge_fork_holds_the_replay_to_the_widening_that_sized_it() {
        // The boundary is sized from a summary computed before the region ran, so both validators exist to catch an
        // operation whose generic hooks disagree with what its closure actually does. Here the fork is produced
        // honestly and then held to deliberately wrong predictions, which is the shape a lying third-party family
        // would present.
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let reference = builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let update = builder.add_input(ListIrType::List(ListType { length: 2 }));
        let previous =
            builder.add_instruction(ListOperation::Swap, Vec::new(), vec![reference, update], None).unwrap()[0];
        let program = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(
                vec![previous, reference],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();

        let context = ListDischargeContext::new(ListDestination::new());
        let allocated = context
            .allocate_discharged(ReferenceType::new(ListType { length: 2 }), ListIrValue::List(vec![1, 2]))
            .unwrap();
        let root = allocated.expect_reference("the caller root").unwrap().root();
        let regions = [program];
        let driver = RecursiveReferenceDischargeDriver::new(&regions, None);
        let boundary = ReferenceRegionDischargeBoundary::new(
            &ListOperation::Call,
            0,
            vec![Some(root), None],
            Vec::new(),
            2,
            Vec::new(),
            2,
        );
        let fork = driver.discharge_region_program(&context, 0, &boundary).unwrap();

        // The region writes its entering root, so a widening that published nothing lost that update.
        assert_eq!(
            fork.validate_predicted_mutations(&[], "list.call"),
            Err(ProgramError::MalformedProgram(format!(
                "operation `list.call` mutated {root} in an attached region that its state widening did not predict",
            ))),
        );
        assert_eq!(fork.validate_predicted_mutations(&[root], "list.call"), Ok(()));

        // The region returns that root at its second output, so a widening that predicted an ordinary value there
        // would have published the root's final state twice.
        assert_eq!(fork.output_roots(), &[None, Some(root)]);
        assert_eq!(
            fork.validate_predicted_output_roots(&[None, None], "list.call"),
            Err(ProgramError::MalformedProgram(
                "operation `list.call` attaches a region whose outputs do not denote the references its state \
                 widening expected"
                    .to_string(),
            )),
        );
        assert_eq!(fork.validate_predicted_output_roots(&[None, Some(root)], "list.call"), Ok(()));
    }

    #[test]
    fn test_reference_discharge_region_program_isolates_the_caller_environment() {
        // A region that accumulates into the root it receives and returns that root unchanged, which is the shape a
        // structured rule threads state through.
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let reference = builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let update = builder.add_constant(ListIrValue::List(vec![10, 10]));
        builder
            .add_instruction(ListOperation::AddUpdate, Vec::new(), vec![reference, update], None)
            .unwrap();
        let program = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![reference], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let destination = TracingContext::<ListIrValue, ListOperation>::new();
        let context = ReferenceDischargeContext::<_, ListReferenceDischarge>::new(destination.clone());
        let state = destination.input(ListIrType::List(ListType { length: 2 }));
        let allocated = context.allocate_discharged(ReferenceType::new(ListType { length: 2 }), state).unwrap();
        let root = allocated.expect_reference("the caller root").unwrap().root();
        let regions = [program];
        let driver = RecursiveReferenceDischargeDriver::new(&regions, None);
        let boundary = ReferenceRegionDischargeBoundary::new(
            &ListOperation::Call,
            0,
            vec![Some(root)],
            Vec::new(),
            0,
            Vec::new(),
            1,
        );
        let fork = driver.discharge_region_program(&context, 0, &boundary).unwrap();

        // The rebuilt region reports what it did in the caller's own terms, and the caller's environment is untouched:
        // the root is still unmutated and still holds the state it entered with.
        assert_eq!(fork.mutated_roots, [root]);
        assert_eq!(fork.output_roots(), &[Some(root)]);
        assert!(!context.is_mutated(root).unwrap());
        assert_eq!(context.discharged_state(root).unwrap().atom_id().unwrap(), AtomId::new(0));
        assert_eq!(
            fork.program.to_string(),
            indoc! {"
                lambda %0:list<2> .
                let %1:list<2> = const [10, 10]
                    %2:list<2> = list.select %0
                    %3:list<2> = list.add %2 %1
                    %4:list<2> = list.splice %0 %3
                in (%4)"},
        );

        // A replay that fails leaves the caller's environment exactly as it was and yields no values at all, because
        // the fork's result type carries none. The checked append rejects a read of a consumed family at
        // construction, so the failing program is assembled through the unchecked rebuild hatch.
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let reference = builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        builder.add_instruction(ListOperation::Freeze, Vec::new(), vec![reference], None).unwrap();
        let failing = builder.add_variable(ListIrType::List(ListType { length: 2 }));
        builder.add_instruction_unchecked(Instruction::new(
            ListOperation::Read,
            vec![reference],
            vec![failing],
            Vec::new(),
        ));
        let program = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![failing], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let regions = [program];
        let driver = RecursiveReferenceDischargeDriver::new(&regions, None);
        assert!(matches!(
            driver.discharge_region_program(&context, 0, &boundary),
            Err(ProgramError::MalformedProgram(message))
                if message.starts_with("reference discharge accessed consumed reference root "),
        ));
        assert!(!context.is_mutated(root).unwrap());
        assert_eq!(context.discharged_state(root).unwrap().atom_id().unwrap(), AtomId::new(0));
    }

    #[test]
    fn test_reference_discharge_region_program_rejects_duplicate_added_roots() {
        let builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let program = builder.build::<Vec<ListIrValue>, Vec<ListIrValue>>(Vec::new(), Vec::new(), Vec::new()).unwrap();

        let context = ListDischargeContext::new(ListDestination::new());
        let allocated = context
            .allocate_discharged(ReferenceType::new(ListType { length: 2 }), ListIrValue::List(vec![1, 2]))
            .unwrap();
        let root = allocated.expect_reference("the added root").unwrap().root();
        let regions = [program];
        let driver = RecursiveReferenceDischargeDriver::new(&regions, None);
        let boundary = ReferenceRegionDischargeBoundary::new(
            &ListOperation::Call,
            0,
            Vec::new(),
            vec![root, root],
            0,
            Vec::new(),
            0,
        );

        assert_eq!(
            driver.discharge_region_program(&context, 0, &boundary).unwrap_err(),
            ProgramError::MalformedProgram(format!(
                "reference discharge adds {root} to region `{}` more than once",
                regions[0].entry_region_ref().id(),
            )),
        );
    }

    #[test]
    fn test_reference_discharge_region_program_rejects_an_added_root_duplicating_a_declared_root() {
        // A repeated *declared* position deliberately aliases one caller root, but a synthesized state position must
        // never restate a root the boundary already declares: the rebuilt region would carry two boundary positions
        // for one state with no rule deciding which successor wins.
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let program = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(Vec::new(), vec![Placeholder], Vec::new())
            .unwrap();

        let context = ListDischargeContext::new(ListDestination::new());
        let allocated = context
            .allocate_discharged(ReferenceType::new(ListType { length: 2 }), ListIrValue::List(vec![1, 2]))
            .unwrap();
        let root = allocated.expect_reference("the declared root").unwrap().root();
        let regions = [program];
        let driver = RecursiveReferenceDischargeDriver::new(&regions, None);
        let boundary = ReferenceRegionDischargeBoundary::new(
            &ListOperation::Call,
            0,
            vec![Some(root)],
            vec![root],
            1,
            Vec::new(),
            0,
        );

        assert_eq!(
            driver.discharge_region_program(&context, 0, &boundary).unwrap_err(),
            ProgramError::MalformedProgram(format!(
                "reference discharge adds {root} to region `{}` more than once",
                regions[0].entry_region_ref().id(),
            )),
        );
    }

    #[test]
    fn test_reference_discharge_region_program_propagates_a_consumed_fork_root() {
        // This operation deliberately violates the generic contract: its summary claims no reference access, while
        // its discharge rule consumes the root. Fork sealing must report that consumed root instead of silently
        // omitting it from the mutation report.
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let reference = builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let frozen =
            builder.add_instruction(ListOperation::UnreportedFreeze, Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![frozen], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let context = ListDischargeContext::new(ListDestination::new());
        let allocated = context
            .allocate_discharged(ReferenceType::new(ListType { length: 2 }), ListIrValue::List(vec![1, 2]))
            .unwrap();
        let root = allocated.expect_reference("the caller root").unwrap().root();
        let regions = [program];
        let driver = RecursiveReferenceDischargeDriver::new(&regions, None);
        let boundary = ReferenceRegionDischargeBoundary::new(
            &ListOperation::Call,
            0,
            vec![Some(root)],
            Vec::new(),
            1,
            Vec::new(),
            1,
        );

        assert!(matches!(
            driver.discharge_region_program(&context, 0, &boundary),
            Err(ProgramError::MalformedProgram(message))
                if message.contains("reference discharge accessed consumed reference root"),
        ));
    }

    #[test]
    fn test_reference_discharge_region_program_inserts_added_state_at_its_boundary_position() {
        // Added state is what a region closure reaches without receiving it as a declared operand. No source construct
        // the interpreter currently accepts produces one — a reference reaches a region only through its boundary,
        // because a reference-typed constant is rejected outright — so the mechanics are exercised here directly,
        // against the boundary request a rule would make once capture-scoped references resolve.
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let update = builder.add_input(ListIrType::List(ListType { length: 2 }));
        let reference = builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        builder
            .add_instruction(ListOperation::AddUpdate, Vec::new(), vec![reference, update], None)
            .unwrap();
        let snapshot = builder.add_instruction(ListOperation::Read, Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![snapshot], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let destination = TracingContext::<ListIrValue, ListOperation>::new();
        let context = ReferenceDischargeContext::<_, ListReferenceDischarge>::new(destination.clone());
        let reference_type = ReferenceType::new(ListType { length: 2 });
        let accessed = context
            .allocate_discharged(reference_type.clone(), destination.input(ListIrType::List(ListType { length: 2 })))
            .unwrap();
        let accessed = accessed.expect_reference("the accessed root").unwrap().root();
        let carried = context
            .allocate_discharged(reference_type, destination.input(ListIrType::List(ListType { length: 2 })))
            .unwrap();
        let carried = carried.expect_reference("the carried root").unwrap().root();

        // The added input goes between the two declared inputs and the added output goes before the declared output,
        // which is the insertion arithmetic a scan's carry prefix depends on.
        let regions = [program];
        let driver = RecursiveReferenceDischargeDriver::new(&regions, None);
        let boundary = ReferenceRegionDischargeBoundary::new(
            &ListOperation::Call,
            0,
            vec![None, Some(accessed)],
            vec![carried],
            1,
            vec![carried],
            0,
        );
        let fork = driver.discharge_region_program(&context, 0, &boundary).unwrap();
        assert_eq!(
            fork.program.to_string(),
            indoc! {"
                lambda %0:list<2>, %1:list<2>, %2:list<2> .
                let %3:list<2> = list.select %2
                    %4:list<2> = list.add %3 %0
                    %5:list<2> = list.splice %2 %4
                    %6:list<2> = list.select %5
                in (%1, %6)"},
        );

        // Only the root the closure actually reached is reported as mutated; the carried one passes through, which is
        // why a symmetric boundary can thread it without claiming the region wrote it.
        assert_eq!(fork.mutated_roots, [accessed]);
        assert_eq!(fork.output_roots(), &[None]);
    }

    #[test]
    fn test_reference_discharge_call_rule_threads_state_through_a_non_array_callee() {
        // The whole structured rewrite is universe-generic, so the prototype universe exercises it end to end: a
        // callee mutates the root it receives and returns only the previous snapshot, and discharge widens the call
        // with the final state the caller needs afterwards.
        let mut callee_builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let callee_reference =
            callee_builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let update = callee_builder.add_input(ListIrType::List(ListType { length: 2 }));
        let previous = callee_builder
            .add_instruction(ListOperation::Swap, Vec::new(), vec![callee_reference, update], None)
            .unwrap()[0];
        let callee = callee_builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![previous], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let initial = builder.add_input(ListIrType::List(ListType { length: 2 }));
        let update = builder.add_input(ListIrType::List(ListType { length: 2 }));
        let callee = builder.import_program(callee);
        let root = builder.add_instruction(ListOperation::ReferenceNew, Vec::new(), vec![initial], None).unwrap()[0];
        let previous = builder.add_instruction(ListOperation::Call, vec![callee], vec![root, update], None).unwrap()[0];
        let frozen = builder.add_instruction(ListOperation::Freeze, Vec::new(), vec![root], None).unwrap()[0];
        let source = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(
                vec![previous, frozen],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();

        let discharged = source.discharge_references_with_policy::<ListReferenceDischarge>(0).unwrap();
        assert_eq!(discharged.public_output_count(), 2);
        assert_eq!(discharged.external_states(), &[]);
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:list<2>, %1:list<2> .
                let %2:list<2>, %3:list<2> = list.call %0 %1 [
                    callee={
                        lambda %0:list<2>, %1:list<2> .
                        let %2:list<2> = list.select %0
                            %3:list<2> = list.splice %0 %1
                        in (%2, %3)
                    },
                ]
                in (%2, %3)"},
        );
        assert_eq!(
            discharged.program().interpret(vec![ListIrValue::List(vec![1, 2]), ListIrValue::List(vec![7, 8])]),
            Ok(vec![ListIrValue::List(vec![1, 2]), ListIrValue::List(vec![7, 8])]),
        );
    }

    #[test]
    fn test_partial_reference_discharge_preserves_unselected_external_roots() {
        // The kernel-pipeline shape, in a universe that mentions no arrays: one caller-owned root is selected and
        // becomes threaded state, while the other survives as a reference the rewritten program still accesses
        // through the very operations the source used.
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let pipeline = builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let kernel = builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let update = builder.add_input(ListIrType::List(ListType { length: 2 }));
        let observed = builder.add_instruction(ListOperation::Read, Vec::new(), vec![kernel], None).unwrap()[0];
        builder.add_instruction(ListOperation::AddUpdate, Vec::new(), vec![pipeline, update], None).unwrap();
        builder.add_instruction(ListOperation::Swap, Vec::new(), vec![kernel, observed], None).unwrap();
        let source = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![observed], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();

        let sites = source.reference_discharge_sites(0).unwrap();
        assert_eq!(
            sites,
            vec![
                ReferenceDischargeSite::External(ReferenceSource::Input { index: 0 }),
                ReferenceDischargeSite::External(ReferenceSource::Input { index: 1 }),
            ],
        );
        let discharged = source.partially_discharge_references_with_policy::<ListReferenceDischarge>(0, &sites[..1]);
        let discharged = discharged.unwrap();

        // The selected root became an ordinary state input at its own boundary position and publishes its final state
        // as a hidden output; the preserved root kept its reference type, so it binds nothing at all.
        assert_eq!(discharged.public_output_count(), 1);
        assert_eq!(
            discharged.external_states(),
            &[ReferenceStateBinding::new(ReferenceSource::Input { index: 0 }, Some(1))],
        );
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:list<2>, %1:ref<list<2>>, %2:list<2> .
                let %3:list<2> = list.read %1
                    %4:list<2> = list.select %0
                    %5:list<2> = list.add %4 %2
                    %6:list<2> = list.splice %0 %5
                    %7:list<2> = list.swap %1 %3
                in (%3, %6)"},
        );

        // The result deliberately proves nothing about reference freedom, and asking for the proof reports the
        // surviving reference rather than converting.
        assert_eq!(
            discharged.try_into_full().unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge payload still contains a reference-typed value and cannot form a full discharge"
                    .to_string(),
            ),
        );
    }

    #[test]
    fn test_partial_reference_discharge_preserves_an_unselected_allocation_site() {
        // An interior allocation is selectable in its own right, so a program can normalize its pipeline state while
        // the root a kernel body addresses is allocated, viewed, accessed, and consumed as a reference throughout.
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let initial = builder.add_input(ListIrType::List(ListType { length: 4 }));
        let update = builder.add_input(ListIrType::List(ListType { length: 2 }));
        let root = builder.add_instruction(ListOperation::ReferenceNew, Vec::new(), vec![initial], None).unwrap()[0];
        let view = builder
            .add_instruction(ListOperation::Slice { offset: 1, length: 2 }, Vec::new(), vec![root], None)
            .unwrap()[0];
        builder.add_instruction(ListOperation::AddUpdate, Vec::new(), vec![view, update], None).unwrap();
        let frozen = builder.add_instruction(ListOperation::Freeze, Vec::new(), vec![root], None).unwrap()[0];
        let source = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![frozen], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        // Selecting nothing preserves the allocation, so the whole reference language survives: the view operation is
        // replayed too, and the derived handle consumes the reference that replay produced rather than re-deriving
        // the chain at the access.
        let discharged = source.clone().partially_discharge_references_with_policy::<ListReferenceDischarge>(0, &[]);
        let discharged = discharged.unwrap();
        assert_eq!(discharged.public_output_count(), 1);
        assert_eq!(discharged.external_states(), &[]);
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:list<4>, %1:list<2> .
                let %2:ref<list<4>> = list.reference_new %0
                    %3:ref<list<2>> = list.slice %2
                    list.add_update %3 %1
                    %4:list<4> = list.freeze %2
                in (%4)"},
        );

        // Selecting the allocation instead discharges it, which is the everything-selected case and therefore has to
        // agree with full discharge exactly.
        let sites = source.reference_discharge_sites(0).unwrap();
        assert_eq!(
            sites,
            vec![ReferenceDischargeSite::Allocation {
                instruction: InstructionId::new(source.entry_region_ref().id(), 0),
                output_index: 0,
            }],
        );
        let selected = source
            .clone()
            .partially_discharge_references_with_policy::<ListReferenceDischarge>(0, sites.as_slice());
        let selected = selected.unwrap().try_into_full().unwrap();
        let full = source.discharge_references_with_policy::<ListReferenceDischarge>(0).unwrap();
        assert_eq!(selected.program().to_string(), full.program().to_string());
    }

    #[test]
    fn test_partial_reference_discharge_lets_a_program_consume_a_preserved_external_root() {
        // Full discharge rejects a program that consumes a caller-owned root, because a `ReferenceStateBinding`
        // cannot describe a holder that no longer exists. A preserved root has no binding to describe: the payload
        // keeps the consuming operation, and the caller hands its holder to that operation directly, so partial
        // discharge accepts what full discharge cannot express.
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let external = builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let frozen = builder.add_instruction(ListOperation::Freeze, Vec::new(), vec![external], None).unwrap()[0];
        let source = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![frozen], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let preserved =
            source.clone().partially_discharge_references_with_policy::<ListReferenceDischarge>(0, &[]).unwrap();
        assert_eq!(preserved.external_states(), &[]);
        assert_eq!(
            preserved.program().to_string(),
            indoc! {"
                lambda %0:ref<list<2>> .
                let %1:list<2> = list.freeze %0
                in (%1)"},
        );
        assert_eq!(
            source.discharge_references_with_policy::<ListReferenceDischarge>(0).unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge consumed external input 0, whose holder belongs to the caller".to_string(),
            ),
        );

        // Returning the root afterwards is a use of it like any other, so the consumed root is reported at the output
        // that names it rather than published as a stale reference.
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let external = builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let frozen = builder.add_instruction(ListOperation::Freeze, Vec::new(), vec![external], None).unwrap()[0];
        let source = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(
                vec![frozen, external],
                vec![Placeholder],
                vec![Placeholder; 2],
            )
            .unwrap();

        // A root rendering embeds the identity of the environment that minted it, which is process-global, so the
        // assertion pins everything around it.
        let error = source.partially_discharge_references_with_policy::<ListReferenceDischarge>(0, &[]).unwrap_err();
        let ProgramError::MalformedProgram(message) = &error else {
            panic!("expected a malformed-program rejection but got {error:?}");
        };
        assert!(message.starts_with("reference discharge accessed consumed reference root "), "{message}");
        assert!(message.ends_with(":0"), "{message}");
    }

    #[test]
    fn test_partial_reference_discharge_threads_a_preserved_root_beside_discharged_state() {
        // A structured boundary carries both kinds of root at once: a discharged carry crosses as immutable state and
        // is widened with a published successor, while a preserved carry crosses as the reference it already is, at
        // its own declared operand position, and widens nothing at all.
        let mut callee_builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let callee_state = callee_builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let callee_kernel = callee_builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let observed =
            callee_builder.add_instruction(ListOperation::Read, Vec::new(), vec![callee_kernel], None).unwrap()[0];
        callee_builder
            .add_instruction(ListOperation::AddUpdate, Vec::new(), vec![callee_state, observed], None)
            .unwrap();
        let callee = callee_builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![observed], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let pipeline = builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let kernel = builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let callee = builder.import_program(callee);
        let observed =
            builder.add_instruction(ListOperation::Call, vec![callee], vec![pipeline, kernel], None).unwrap()[0];
        let source = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![observed], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let sites = source.reference_discharge_sites(0).unwrap();
        let discharged =
            source.partially_discharge_references_with_policy::<ListReferenceDischarge>(0, &sites[..1]).unwrap();

        // The selected root's entering state occupies its own operand position and its successor is appended as a
        // published output; the preserved root's operand position still carries a reference, and the rebuilt callee
        // performs the read on it exactly as the source did.
        assert_eq!(discharged.public_output_count(), 1);
        assert_eq!(
            discharged.external_states(),
            &[ReferenceStateBinding::new(ReferenceSource::Input { index: 0 }, Some(1))],
        );
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:list<2>, %1:ref<list<2>> .
                let %2:list<2>, %3:list<2> = list.call %0 %1 [
                    callee={
                        lambda %0:list<2>, %1:ref<list<2>> .
                        let %2:list<2> = list.read %1
                            %3:list<2> = list.select %0
                            %4:list<2> = list.add %3 %2
                            %5:list<2> = list.splice %0 %4
                        in (%2, %5)
                    },
                ]
                in (%2, %3)"},
        );
    }

    #[test]
    fn test_partial_reference_discharge_validates_its_selection_against_the_program() {
        // The selection is checked before anything is replayed, so a site this program does not expose is reported
        // against the program rather than surfacing later as a root that never appeared.
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let external = builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let observed = builder.add_instruction(ListOperation::Read, Vec::new(), vec![external], None).unwrap()[0];
        let source = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![observed], vec![Placeholder], vec![Placeholder])
            .unwrap();

        assert_eq!(
            source
                .partially_discharge_references_with_policy::<ListReferenceDischarge>(
                    0,
                    &[ReferenceDischargeSite::External(ReferenceSource::Input { index: 3 })],
                )
                .unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge selection names external input 3, which is not a selectable site in this program"
                    .to_string(),
            ),
        );
    }

    #[test]
    fn test_discharge_preserved_access_replays_one_access_verbatim_into_the_destination() {
        // The shared preserved replay consumes each handle's own destination reference value and binds the source's
        // own operation over it, which is what makes an access to a surviving root no rewrite at all. It runs against
        // a staging destination, because the eager destination of this universe declines to execute a reference
        // primitive and recording is what production discharge does anyway.
        let referent = ListType { length: 2 };
        let staging = TracingContext::<ListIrValue, ListOperation>::new();
        let builder = staging.builder().clone();
        let outputs = {
            let context = ReferenceDischargeContext::<_, ListReferenceDischarge>::new(staging.clone());
            let preserved = context
                .bind_preserved(
                    ReferenceType::new(referent.clone()),
                    staging.input(ListIrType::Reference(ReferenceType::new(referent.clone()))),
                )
                .unwrap();
            let outputs =
                discharge_preserved_access(&ListOperation::Read, &context, std::slice::from_ref(&preserved)).unwrap();
            assert_eq!(outputs.len(), 1);
            vec![outputs[0].expect_ordinary("the replayed read result").unwrap().atom_id().unwrap()]
        };
        drop(staging);
        let program = Rc::try_unwrap(builder)
            .unwrap()
            .into_inner()
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(outputs, vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:ref<list<2>> .
                let %1:list<2> = list.read %0
                in (%1)"},
        );

        // A replayed access that produces a reference would leave the environment without a root for it, so the
        // operation owning that root has to state its own rule instead.
        let staging = TracingContext::<ListIrValue, ListOperation>::new();
        let staged = ReferenceDischargeContext::<_, ListReferenceDischarge>::new(staging.clone());
        let initial = ReferenceDischargeValue::Ordinary(staging.input(ListIrType::List(referent.clone())));
        assert_eq!(
            discharge_preserved_access(&ListOperation::ReferenceNew, &staged, std::slice::from_ref(&initial)),
            Err(ProgramError::MalformedProgram(
                "reference discharge replayed `list.reference_new` over a preserved root, but its output 0 is the \
                 reference `ref<list<2>>`; an operation that derives a reference owns that root and needs a reference \
                 discharge rule of its own"
                    .to_string(),
            )),
        );

        // A discharged root has no destination reference value at all, so it cannot be replayed over.
        let context = ListDischargeContext::new(ListDestination::new());
        let discharged =
            context.allocate_discharged(ReferenceType::new(referent), ListIrValue::List(vec![1, 2])).unwrap();
        let discharged_root = discharged.expect_reference("the discharged root").unwrap().root();
        assert_eq!(
            discharge_preserved_access(&ListOperation::Read, &context, std::slice::from_ref(&discharged)),
            Err(ProgramError::MalformedProgram(format!(
                "reference discharge cannot replay `list.read` over discharged {discharged_root}, which has no \
                 destination reference value",
            ))),
        );
    }

    #[test]
    fn test_reference_discharge_context_unbinds_preserved_roots() {
        // Consuming a preserved root yields no state — the replayed operation already produced the destination's own
        // result — but it must still stop the environment from handing the root out again, and only a handle denoting
        // the whole root can name a consumption.
        let context = ListDischargeContext::new(ListDestination::new());
        let referent = ListType { length: 2 };
        let preserved = context
            .bind_preserved(
                ReferenceType::new(referent.clone()),
                ListIrValue::Reference(ReferenceType::new(referent.clone())),
            )
            .unwrap();
        let reference = preserved.expect_reference("the preserved root").unwrap().clone();
        let discharged =
            context.allocate_discharged(ReferenceType::new(referent), ListIrValue::List(vec![1, 2])).unwrap();
        let discharged_root = discharged.expect_reference("the discharged root").unwrap().root();

        let same_type_view = context
            .derive(
                &reference,
                ListAlias { offset: 0, length: 2 },
                reference.r#type().clone(),
                Some(ListIrValue::Reference(reference.r#type().clone())),
            )
            .unwrap();
        let same_type_view = same_type_view.expect_reference("the same-type preserved view").unwrap();
        assert_eq!(
            context.unbind_preserved(same_type_view),
            Err(ProgramError::MalformedProgram(format!(
                "reference discharge cannot consume {} through the derived view `ref<list<2>>`; consumption yields \
                 the whole root, whose reference type is `ref<list<2>>`",
                reference.root(),
            ))),
        );

        let view = context
            .derive(
                &reference,
                ListAlias { offset: 0, length: 1 },
                ReferenceType::new(ListType { length: 1 }),
                Some(ListIrValue::Reference(ReferenceType::new(ListType { length: 1 }))),
            )
            .unwrap();
        let view = view.expect_reference("the derived preserved view").unwrap().clone();
        assert_eq!(
            context.unbind_preserved(&view),
            Err(ProgramError::MalformedProgram(format!(
                "reference discharge cannot consume {} through the derived view `ref<list<1>>`; consumption yields \
                 the whole root, whose reference type is `ref<list<2>>`",
                reference.root(),
            ))),
        );
        assert_eq!(context.unbind_preserved(&reference), Ok(()));
        assert_eq!(context.live_roots(), vec![discharged_root]);
        assert_eq!(
            context.unbind_preserved(&reference),
            Err(ProgramError::MalformedProgram(format!("reference discharge accessed consumed {}", reference.root()))),
        );

        // A discharged root is not unbound through the preserved path, which is what keeps the two states from being
        // confused by a rule that dispatched on the wrong one.
        let discharged = discharged.expect_reference("the discharged root").unwrap();
        assert_eq!(
            context.unbind_preserved(discharged),
            Err(ProgramError::MalformedProgram(format!(
                "reference discharge unbound discharged {discharged_root} as a preserved root",
            ))),
        );
    }
}
