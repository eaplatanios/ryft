use std::borrow::Cow;
use std::cell::{Ref, RefCell};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::{Debug, Display};
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::captures::{CaptureConstant, ClosedProgram};
use crate::contexts::{Context, Domain, StagingContext};
use crate::macros::check_count;
use crate::parameters::{Parameterized, Placeholder};
use crate::programs::ProgramError;
use crate::programs::instructions::InstructionId;
use crate::programs::operations::Operation;
use crate::programs::programs::Program;
use crate::programs::references::discharge::interpreter::{
    ReferenceDischargeRegionSummary, ReferenceDischargeStateWidening, summarize_region_closure,
    validate_region_accesses,
};
use crate::programs::references::types::ReferenceType;
use crate::programs::regions::{
    EmptyRegionDriver, RegionDriver, RegionId, RegionRef, RegionReplayMappings, ReplayRegionDriver,
};
use crate::programs::types::{Type, Typed};
use crate::programs::values::Value;
use crate::tracing::TracingContext;

/// A caller-selectable [`Reference`](crate::Reference) target for partial reference discharge. A target
/// needs an identity that exists in the _source_ [`Program`], before any replay begins, so it cannot reuse the
/// environment's [`ReferenceDischargeAllocationId`]s. In particular, a nested region's formal reference input is
/// invocation-parameterized (the region may be invoked from several call sites) and so it names no single caller-owned
/// reference and is deliberately not selectable. Targets resolve internally to allocations once discharge starts.
///
/// Targets are arena-relative in exactly the sense that every other reference artifact is: their instruction and
/// boundary identifiers are meaningful only in the program arena from which they were enumerated. Target validation
/// rejects every kind mismatch, and the arena-relativity contract carries the rest because a target from a different
/// arena that happens to name a valid allocation here is indistinguishable in principle.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ReferenceDischargeTarget {
    /// Entry-boundary allocation supplied by the caller as a lifted capture or a public reference argument.
    External(ReferenceSource),

    /// Interior allocation target, identified by the allocating [`Instruction`](crate::Instruction) and the output
    /// position that defines the fresh allocation.
    Internal {
        /// Allocating [`Instruction`](crate::Instruction).
        instruction: InstructionId,

        /// Output position defining the fresh allocation.
        output_index: usize,
    },
}

impl Display for ReferenceDischargeTarget {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::External(source) => write!(formatter, "external {source}"),
            Self::Internal { instruction, output_index } => {
                write!(formatter, "internal allocation at `{instruction}` output {output_index}")
            }
        }
    }
}

/// Selection of reference allocations to discharge (i.e., a collection of [`ReferenceDischargeTarget`]s). A partial
/// discharge stores the set of [`ReferenceDischargeTarget`]s selected by the caller and preserves every other
/// allocation. A full discharge instead uses a distinct "everything" state because it must also discharge allocations
/// that callers cannot name with a target, such as allocations bound directly during replay.
#[derive(Clone, Debug)]
pub struct ReferenceDischargeTargets {
    /// Selected targets, or [`None`] when every target is selected.
    targets: Option<Rc<BTreeSet<ReferenceDischargeTarget>>>,
}

impl ReferenceDischargeTargets {
    /// Returns the [`ReferenceDischargeTargets`] that full discharge runs under, which select every reference.
    pub const fn everything() -> Self {
        Self { targets: None }
    }

    /// Validates `targets` against `program` and returns a selection that discharges exactly those targets. Every
    /// target must name a reference-typed entry position or a reference allocation defined by an instruction in
    /// `program`, and each target may appear only once. Every reference not named by the resulting selection is
    /// preserved.
    ///
    /// # Parameters
    ///
    ///   - `program`: Program whose reference allocations the targets select.
    ///   - `capture_count`: Number of leading flat inputs that originated in the program's capture table.
    ///   - `targets`: Targets selected for discharge, in caller-chosen order.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] naming the offending target when a target is duplicated, names an
    /// out-of-range or non-reference entry position, names an instruction that the program does not contain, names an
    /// operation that defines no reference allocation, or names an output position that is not an allocation. It also
    /// returns an error when `capture_count` exceeds the program's input count.
    pub fn from_targets<V: Value, O: Operation<Type = V::Type>, Input: Parameterized<V>, Output: Parameterized<V>>(
        program: &Program<V, O, Input, Output>,
        capture_count: usize,
        targets: &[ReferenceDischargeTarget],
    ) -> Result<Self, ProgramError> {
        let entry = program.entry_region_ref();
        let input_ids = entry.input_ids();
        if capture_count > input_ids.len() {
            return Err(ProgramError::MalformedProgram(format!(
                "reference discharge target validation requests {} captures but the program has {} inputs",
                capture_count,
                input_ids.len(),
            )));
        }

        let mut selected = BTreeSet::new();
        for target in targets {
            if !selected.insert(*target) {
                return Err(ProgramError::MalformedProgram(format!(
                    "reference discharge targets contain {target} more than once",
                )));
            }
        }

        // Only the named instructions are resolved, so validating a small target set does not pay for the reference
        // semantics of every instruction in the closure.
        let instructions = entry.instructions_in_closure().collect::<HashMap<_, _>>();
        for target in targets {
            let invalid_target = || {
                ProgramError::MalformedProgram(format!(
                    "reference discharge targets include {target} which is not selectable in this program",
                ))
            };
            match target {
                ReferenceDischargeTarget::External(source) => {
                    let input_index = source.flat_input_index(capture_count).map_err(|_| invalid_target())?;
                    let input = input_ids.get(input_index).ok_or_else(invalid_target)?;
                    if !entry.atoms()[input.index()].r#type().is_reference() {
                        return Err(invalid_target());
                    }
                }
                ReferenceDischargeTarget::Internal { instruction, output_index } => {
                    let instruction = instructions.get(instruction).ok_or_else(invalid_target)?;
                    let output_indices =
                        instruction.operation().reference_semantics().allocation_output_indices().collect::<Vec<_>>();
                    if !output_indices.contains(output_index) {
                        return Err(invalid_target());
                    }
                }
            }
        }

        Ok(Self { targets: Some(Rc::new(selected)) })
    }

    /// Returns `true` if `target` is selected for discharge by this [`ReferenceDischargeTargets`] set.
    #[inline]
    pub fn selects(&self, target: ReferenceDischargeTarget) -> bool {
        self.targets.as_ref().is_none_or(|targets| targets.contains(&target))
    }
}

impl<V: Value, O: Operation<Type = V::Type>, Input: Parameterized<V>, Output: Parameterized<V>>
    Program<V, O, Input, Output>
{
    /// Returns every [`ReferenceDischargeTarget`] that this [`Program`] exposes to partial reference discharge, in a
    /// canonical ordering with the entry-boundary externals in boundary order, followed by the interior allocations
    /// ordered by instruction and output position. This is a deliberately lightweight query. It reads only the entry
    /// boundary types and the generic [`Operation::reference_semantics`] over the attached [`Region`](crate::Region)
    /// closure, so it does not run the discharge rewrite or construct its environments, and callers can enumerate
    /// selectable targets without paying for either. Allocations inside nested regions are included because every
    /// allocating [`Instruction`](crate::Instruction) defines a concrete local reference wherever it occurs.
    ///
    /// One class of enumerated targets is inert: an allocation inside a closure that no operation ever replays, such
    /// as the dormant derivative rule region of a [`CustomJvpOperation`](crate::CustomJvpOperation). Discharge rejects
    /// such a program outright, whichever way the target is selected, because how a reference boundary widens there has
    /// no defined meaning. The enumeration reports the target anyway rather than second-guessing the region roles, so
    /// that it stays a structural query.
    ///
    /// # Parameters
    ///
    ///   - `capture_count`: Number of leading flat inputs that originated in the program's capture table, used to
    ///     split the entry boundary into [`ReferenceSource::Capture`] and [`ReferenceSource::Input`] positions.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when `capture_count` exceeds the program's input count.
    pub fn reference_discharge_targets(
        &self,
        capture_count: usize,
    ) -> Result<Vec<ReferenceDischargeTarget>, ProgramError> {
        let entry = self.entry_region_ref();
        let input_ids = entry.input_ids();
        if capture_count > input_ids.len() {
            return Err(ProgramError::MalformedProgram(format!(
                "reference discharge target enumeration requests {} captures but the program has {} inputs",
                capture_count,
                input_ids.len(),
            )));
        }

        let mut targets = input_ids
            .iter()
            .enumerate()
            .filter(|(_, input)| entry.atoms()[input.index()].r#type().is_reference())
            .map(|(input_index, _)| {
                ReferenceDischargeTarget::External(ReferenceSource::from_flat_input_index(input_index, capture_count))
            })
            .collect::<Vec<_>>();

        let mut allocations = entry
            .instructions_in_closure()
            .flat_map(|(instruction_id, instruction)| {
                instruction
                    .operation()
                    .reference_semantics()
                    .allocation_output_indices()
                    .collect::<Vec<_>>()
                    .into_iter()
                    .map(move |output_index| ReferenceDischargeTarget::Internal {
                        instruction: instruction_id,
                        output_index,
                    })
            })
            .collect::<Vec<_>>();

        // Closure traversal visits regions in an unspecified order, so internal targets are sorted by instruction and
        // output position to make the enumeration reproducible for callers that persist or compare target sets.
        allocations.sort_unstable();
        targets.append(&mut allocations);
        Ok(targets)
    }
}

/// [`Reference`](crate::Reference)-free [`Program`] and external-reference bindings produced by the reference
/// discharge transform. A full result is a [`PartialReferenceDischargeResult`] whose complete attached region closure
/// has been proven to contain neither reference-typed atoms nor operations with reference semantics. The [`TryFrom`]
/// implementation performs that proof and otherwise wraps the partial result unchanged. The proof examines every
/// attached region, including dormant rule regions. It rejects the conversion with [`ProgramError::MalformedProgram`]
/// if any reference-typed atom or operation with reference semantics remains; unrelated ordered-state operations do not
/// prevent conversion.
#[derive(Debug)]
pub struct ReferenceDischargeResult<V: Value, O: Operation<Type = V::Type>> {
    /// Underlying [`PartialReferenceDischargeResult`].
    partial: PartialReferenceDischargeResult<V, O>,
}

impl<V: Value, O: Operation<Type = V::Type>> ReferenceDischargeResult<V, O> {
    /// Returns the underlying [`Reference`](crate::Reference)-free [`Program`].
    pub const fn program(&self) -> &Program<V, O, Vec<V>, Vec<V>> {
        self.partial.program()
    }

    /// Returns the number of leading [`Program`] inputs lifted from the source program's capture table. The discharged
    /// program has one flat input boundary in `[captures..., inputs...]` order. This count is the split point between
    /// those two groups; it counts every lifted capture, not only captures that contain references or appear in
    /// [`Self::external_reference_bindings`]. [`ReferenceSource::Capture`] indices are relative to the first group,
    /// while [`ReferenceSource::Input`] indices are relative to the second.
    ///
    /// For example, a count of `2` gives the following boundary:
    ///
    /// ```text
    /// [capture 0, capture 1 | input 0, input 1, ...]
    ///                       ^ capture_count
    /// ```
    pub const fn capture_count(&self) -> usize {
        self.partial.capture_count()
    }

    /// Returns the number of public outputs at the front of the [`Program`]'s complete output boundary. Public outputs
    /// occupy output indices `[0, output_count)`. Any remaining outputs form a hidden suffix containing the final
    /// values of mutated external references, in [`Self::external_reference_bindings`] order after read-only bindings
    /// are omitted. A read-only binding has no hidden output.
    ///
    /// For example, an `output_count` of `2` with one mutated external reference gives:
    ///
    /// ```text
    /// [output 0, output 1 | final external state]
    ///                     ^ output_count
    /// ```
    pub const fn output_count(&self) -> usize {
        self.partial.output_count()
    }

    /// Returns the bindings between caller-owned references and the discharged [`Program`] boundary. Bindings appear
    /// in canonical entry-boundary order: captures first, then public inputs, with each group ordered by its logical
    /// index. Each binding's [`ExternalReferenceBinding::source`] identifies the program input that receives the
    /// reference's initial value. A mutated binding also identifies the hidden output containing its final value.
    /// A read-only binding has no output index. Local allocations never appear because no caller owns their state.
    ///
    /// For example, the following metadata describes a read-only captured reference and a mutated reference supplied
    /// as public input 1:
    ///
    /// ```text
    /// capture_count = 1
    /// inputs         = [capture 0 | input 0, input 1]
    /// output_count   = 2
    /// outputs        = [output 0, output 1 | final input 1]
    /// bindings       = [capture 0 -> None, input 1 -> Some(2)]
    /// ```
    ///
    /// An empty slice means that executing the program requires no caller-owned reference state.
    #[inline]
    pub fn external_reference_bindings(&self) -> &[ExternalReferenceBinding] {
        self.partial.external_reference_bindings()
    }

    /// Consumes this [`ReferenceDischargeResult`] and returns its [`Program`] when execution requires no caller-owned
    /// reference state. A full discharge is reference-free, but that alone does not make its metadata discardable. Even
    /// a read-only external reference needs a binding so the caller can provide its initial value. A mutated external
    /// reference additionally needs a binding for its hidden final-state output. This conversion therefore accepts only
    /// an empty [`Self::external_reference_bindings`] slice.
    ///
    /// An empty binding slice also guarantees that there are no hidden external final-state outputs, so
    /// [`Self::output_count`] equals the returned program's complete output count. [`Self::capture_count`] may still
    /// be nonzero because non-reference captures require no binding. The returned program is both reference-free and
    /// independent of caller-owned references.
    ///
    /// Conceptually, a program with only non-reference captures or local reference allocations can be returned
    /// directly, while either of the following external dependencies is rejected:
    ///
    /// ```text
    /// read-only external: input initial state
    /// mutated external:   input initial state -> hidden final state
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::UnsupportedOperation`] identifying the first caller-owned reference when this result
    /// contains any external binding.
    pub fn into_program_without_external_references(self) -> Result<Program<V, O, Vec<V>, Vec<V>>, ProgramError> {
        if let Some(binding) = self.external_reference_bindings().first() {
            return Err(ProgramError::UnsupportedOperation {
                message: format!("reference discharge cannot discard the binding for external `{}`", binding.source()),
            });
        }
        let (program, _, _, _) = self.into_parts();
        Ok(program)
    }

    /// Consumes this [`ReferenceDischargeResult`] and returns its underlying [`Program`], capture count, public output
    /// count, and external reference bindings, in that order.
    #[inline]
    pub fn into_parts(self) -> (Program<V, O, Vec<V>, Vec<V>>, usize, usize, Vec<ExternalReferenceBinding>) {
        self.partial.into_parts()
    }
}

impl<V: Value, O: Operation<Type = V::Type>> TryFrom<PartialReferenceDischargeResult<V, O>>
    for ReferenceDischargeResult<V, O>
{
    type Error = ProgramError;

    fn try_from(partial: PartialReferenceDischargeResult<V, O>) -> Result<Self, Self::Error> {
        let entry = partial.program.entry_region_ref();
        if entry.contains_atom_type_in_closure(Type::is_reference) {
            return Err(ProgramError::MalformedProgram(
                "reference discharge program still contains a reference-typed value and cannot form a full discharge"
                    .to_string(),
            ));
        }

        // The closure traversal visits regions in an unspecified order, so the reported occurrence is the earliest
        // instruction position rather than the first one encountered, keeping the diagnostic reproducible.
        if let Some((instruction_id, instruction)) = entry
            .instructions_in_closure()
            .filter(|(_, instruction)| !instruction.operation().reference_semantics().is_empty())
            .min_by_key(|(instruction_id, _)| *instruction_id)
        {
            return Err(ProgramError::MalformedProgram(format!(
                "reference discharge program retains reference operation `{}` at `{}` and cannot form a full discharge",
                instruction.operation().name(),
                instruction_id,
            )));
        }

        Ok(Self { partial })
    }
}

/// [`Program`] produced by _partial_ reference discharge, in which only the caller-selected reference targets became
/// explicit immutable state and every unselected allocation survives as a well-typed reference value. The discharged
/// part of the boundary obeys exactly the invariants of [`ReferenceDischargeResult`]: discharged external allocations
/// are reported as [`ExternalReferenceBinding`]s in canonical entry-boundary order, and the mutated subset of those
/// bindings tiles the hidden output suffix that follows the public outputs. Discharged local allocations leave no
/// binding, because no caller owns their state. Preserved references contribute neither bindings nor hidden outputs;
/// they simply remain reference-typed values inside the program, and their accesses replay verbatim.
#[derive(Debug)]
pub struct PartialReferenceDischargeResult<V: Value, O: Operation<Type = V::Type>> {
    /// Refer to [`Self::program`].
    program: Program<V, O, Vec<V>, Vec<V>>,

    /// Refer to [`Self::capture_count`].
    capture_count: usize,

    /// Refer to [`Self::output_count`].
    output_count: usize,

    /// Refer to [`Self::external_reference_bindings`].
    external_reference_bindings: Vec<ExternalReferenceBinding>,
}

impl<V: Value, O: Operation<Type = V::Type>> PartialReferenceDischargeResult<V, O> {
    /// Creates a new [`PartialReferenceDischargeResult`]. The provided external reference bindings
    /// describe the discharged allocations only and must satisfy the same canonical boundary invariants as
    /// [`ReferenceDischargeResult`] (i.e., they must name valid discharged inputs in canonical source order, and their
    /// final state output indices, omitting read-only bindings, must exactly cover the hidden output suffix in binding
    /// order).
    ///
    /// # Parameters
    ///
    ///   - `program`: Partially discharged [`Program`].
    ///   - `capture_count`: Number of leading inputs originating in the source program's capture table.
    ///   - `output_count`: Number of public outputs preceding hidden final-state outputs.
    ///   - `external_reference_bindings`: Logical bindings for the discharged external references, in canonical
    ///     entry-boundary order.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when the counts and bindings do not describe one canonical
    /// discharged boundary.
    pub fn new(
        program: Program<V, O, Vec<V>, Vec<V>>,
        capture_count: usize,
        output_count: usize,
        external_reference_bindings: Vec<ExternalReferenceBinding>,
    ) -> Result<Self, ProgramError> {
        let total_input_count = program.input_count();
        let total_output_count = program.output_count();
        if capture_count > total_input_count {
            return Err(ProgramError::MalformedProgram(format!(
                "reference discharge reports {capture_count} captures but discharged input count is \
                 {total_input_count}",
            )));
        }
        if output_count > total_output_count {
            return Err(ProgramError::MalformedProgram(format!(
                "reference discharge reports {output_count} public outputs but discharged output count is \
                 {total_output_count}",
            )));
        }
        for binding in &external_reference_bindings {
            let input_index = binding.source().flat_input_index(capture_count)?;
            if input_index >= total_input_count {
                return Err(ProgramError::MalformedProgram(format!(
                    "reference discharge state for `{}` names input {} but discharged input count is {}",
                    binding.source(),
                    input_index,
                    total_input_count,
                )));
            }
        }
        for adjacent_bindings in external_reference_bindings.windows(2) {
            let previous_source = adjacent_bindings[0].source();
            let source = adjacent_bindings[1].source();
            if source <= previous_source {
                return Err(ProgramError::MalformedProgram(format!(
                    "reference discharge state source `{source}` does not follow source `{previous_source}` in \
                     canonical boundary order",
                )));
            }
        }
        let mut expected_output_index = output_count;
        for binding in external_reference_bindings.iter().filter(|binding| binding.is_mutated()) {
            let output_index = binding.output_index().unwrap();
            if output_index != expected_output_index {
                return Err(ProgramError::MalformedProgram(format!(
                    "reference discharge final-state output {} for `{}` does not match expected hidden output {}",
                    output_index,
                    binding.source(),
                    expected_output_index,
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
        Ok(Self { program, capture_count, output_count, external_reference_bindings })
    }

    /// Returns the underlying partially discharged [`Program`]. Unlike [`ReferenceDischargeResult::program`], this
    /// program may still contain reference-typed values and operations for allocations that the caller did not select
    /// for discharge.
    pub const fn program(&self) -> &Program<V, O, Vec<V>, Vec<V>> {
        &self.program
    }

    /// Returns the number of leading [`Program`] inputs lifted from the source program's capture table. The boundary
    /// uses `[captures..., inputs...]` order, and this count is the split point between the two groups. It counts all
    /// lifted captures, including non-reference captures and preserved reference captures that do not appear in
    /// [`Self::external_reference_bindings`]. For example, a count of `1` gives `[capture 0 | input 0, input 1, ...]`.
    pub const fn capture_count(&self) -> usize {
        self.capture_count
    }

    /// Returns the number of public outputs at the front of the [`Program`]'s complete output boundary. Public outputs
    /// occupy `[0, output_count)`. Hidden final state outputs for mutated, discharged external references follow in
    /// [`Self::external_reference_bindings`] order after read-only bindings are omitted. Preserved references remain
    /// reference-typed values and add no hidden output. For example, an `output_count` of `1` gives
    /// `[output 0 | hidden final states...]`.
    pub const fn output_count(&self) -> usize {
        self.output_count
    }

    /// Returns the bindings for external references selected and successfully discharged by the partial reference
    /// discharge transform. Bindings use the same canonical source ordering and input/output interpretation as
    /// [`ReferenceDischargeResult::external_reference_bindings`]. The difference is completeness: an external reference
    /// omitted from this slice may still survive as a reference-typed value because it was not selected. Local
    /// allocations never appear, and preserved external references contribute neither a binding nor a hidden output.
    /// For example, with inputs `[capture 0 | input 0]`, selecting only `input 0` produces a binding for
    /// [`ReferenceSource::Input`] index `0`; an unselected reference in `capture 0` remains in the program
    /// and does not produce a binding.
    #[inline]
    pub fn external_reference_bindings(&self) -> &[ExternalReferenceBinding] {
        self.external_reference_bindings.as_slice()
    }

    /// Consumes this [`PartialReferenceDischargeResult`] and returns its underlying [`Program`], capture count,
    /// public output count, and external reference bindings, in that order.
    #[inline]
    pub fn into_parts(self) -> (Program<V, O, Vec<V>, Vec<V>>, usize, usize, Vec<ExternalReferenceBinding>) {
        let Self { program, capture_count, output_count, external_reference_bindings } = self;
        (program, capture_count, output_count, external_reference_bindings)
    }
}

/// Metadata connecting one caller-owned [`Reference`] to its explicit inputs and outputs after reference discharge.
/// Reference discharge turns implicit access to a reference into value flow that a reference-free backend can execute.
/// For example, consider a source function that takes parameter state by reference, updates it, and returns a public
/// result:
///
/// ```text
/// train_step(parameters: Reference<Array>, batch: Array) -> Array
/// ```
///
/// Its discharged boundary has the following conceptual shape:
///
/// ```text
/// train_step(parameters: Array, batch: Array) -> (result: Array, updated_parameters: Array)
/// ```
///
/// [`Self::source`] identifies where the caller supplied the original reference. Before execution, the stateful
/// invocation layer reads the reference's current value into that discharged input. The `updated_parameters` value
/// is a synthetic writeback output (i.e., it is part of the complete discharged boundary, but the invocation layer
/// consumes it instead of returning it as part of the source function's public result). It "installs" that value back
/// into the caller's reference after execution.
///
/// In this example there is one public output, so [`Self::output_index`] is `Some(1)`: the absolute index of
/// `updated_parameters` in the complete output list `[result, updated_parameters]`. A reference that the function only
/// reads needs no writeback output and therefore has an output index of [`None`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub struct ExternalReferenceBinding {
    /// Capture or public input through which the caller supplies the reference.
    source: ReferenceSource,

    /// Absolute complete-output index containing the final reference value, or [`None`] for a read-only reference.
    output_index: Option<usize>,
}

impl ExternalReferenceBinding {
    /// Creates a new [`ExternalReferenceBinding`].
    ///
    /// # Parameters
    ///
    ///   - `source`: Capture or public input through which the caller supplies the reference.
    ///   - `output_index`: Absolute complete-output index containing the final reference value,
    ///     or [`None`] when the program only reads the reference.
    pub const fn new(source: ReferenceSource, output_index: Option<usize>) -> Self {
        Self { source, output_index }
    }

    /// Returns the capture or public input through which the caller supplies the reference.
    pub const fn source(&self) -> ReferenceSource {
        self.source
    }

    /// Returns whether the corresponding [`Program`] may mutate this external reference.
    pub const fn is_mutated(&self) -> bool {
        self.output_index.is_some()
    }

    /// Returns the absolute complete-output index containing the final reference value, if one must be written back.
    pub const fn output_index(&self) -> Option<usize> {
        self.output_index
    }
}

/// Source of a [`Reference`] that is external to a [`Program`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReferenceSource {
    /// Reference to a capture lifted into the entry boundary before input arguments.
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
    /// Returns the logical source occupying one position in a [`Program`]'s flat entry input boundary. Capture lifting
    /// forms one flat input list in canonical `[captures..., inputs...]` order: the first `capture_count` positions
    /// correspond to the source program's capture table, and every remaining position corresponds to a public input.
    /// This function classifies `flat_input_index` relative to that split and expresses public input positions without
    /// the leading capture prefix.
    ///
    /// This function cannot validate that `flat_input_index` is within the complete input boundary because it receives
    /// only the capture-prefix length. Callers enumerating a program boundary must thus supply one of that program's
    /// valid input positions.
    ///
    /// # Parameters
    ///
    ///   - `flat_input_index`: Zero-based position in the complete flat `[captures..., inputs...]` entry boundary.
    ///   - `capture_count`: Number of leading boundary positions originating in the source program's capture table.
    pub const fn from_flat_input_index(flat_input_index: usize, capture_count: usize) -> Self {
        if flat_input_index < capture_count {
            Self::Capture { index: flat_input_index }
        } else {
            Self::Input { index: flat_input_index - capture_count }
        }
    }

    /// Returns this logical source's position in the [`Program`]'s flat entry input boundary. Capture lifting forms one
    /// flat input list in canonical `[captures..., inputs...]` order. A capture's logical index is therefore already
    /// its flat position, while a public input's logical index is offset by `capture_count`, the length of the leading
    /// capture prefix.
    ///
    /// # Parameters
    ///
    ///   - `capture_count`: Number of leading boundary positions originating in the source program's capture table.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when a capture index lies outside the leading capture prefix or when
    /// offsetting a public input index by that prefix overflows `usize`.
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
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Capture { index } => write!(formatter, "capture {index}"),
            Self::Input { index } => write!(formatter, "input {index}"),
        }
    }
}

/// [`ReferenceDischargeContext`]-free carrier flowing through the reference discharge transform.
/// [`ReferenceDischargeableOperation`] implementations receive and return such carrier; the context that owns the
/// allocation environment travels separately as an explicit argument rather than being stamped onto every value.
pub enum ReferenceDischargeValue<C: Domain, P: ReferenceDischargePolicy<C>> {
    /// Destination value carrying no reference allocation handle and replayed as-is.
    Value(C::Value),

    /// Handle to a live reference allocation.
    Reference(ReferenceDischargeReference<C, P>),
}

impl<C: Domain, P: ReferenceDischargePolicy<C>> ReferenceDischargeValue<C, P> {
    /// Tries to borrow the value that this [`ReferenceDischargeValue`] holds, returning an error naming `expectation`
    /// when it holds a reference instead of a value.
    ///
    /// # Parameters
    ///
    ///   - `expectation`: Description of the operand the caller expected, used in the diagnostic.
    #[inline]
    pub fn try_as_value(&self, expectation: &str) -> Result<&C::Value, ProgramError> {
        match self {
            Self::Value(value) => Ok(value),
            Self::Reference(reference) => Err(ProgramError::MalformedProgram(format!(
                "reference discharge expected {expectation} but received {reference}",
            ))),
        }
    }

    /// Tries to borrow the reference that this [`ReferenceDischargeValue`] holds, returning an error naming
    /// `expectation` when it holds a value instead of a reference.
    ///
    /// # Parameters
    ///
    ///   - `expectation`: Description of the operand the caller expected, used in the diagnostic.
    #[inline]
    pub fn try_as_reference(&self, expectation: &str) -> Result<&ReferenceDischargeReference<C, P>, ProgramError> {
        match self {
            Self::Reference(reference) => Ok(reference),
            Self::Value(_) => Err(ProgramError::MalformedProgram(format!(
                "reference discharge expected {expectation} but received a value",
            ))),
        }
    }
}

impl<C: Domain, P: ReferenceDischargePolicy<C>> Clone for ReferenceDischargeValue<C, P> {
    #[inline]
    fn clone(&self) -> Self {
        match self {
            Self::Value(value) => Self::Value(value.clone()),
            Self::Reference(reference) => Self::Reference(reference.clone()),
        }
    }
}

impl<C: Domain, P: ReferenceDischargePolicy<C>> Debug for ReferenceDischargeValue<C, P> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Value(value) => formatter.debug_tuple("Value").field(value).finish(),
            Self::Reference(reference) => formatter.debug_tuple("Reference").field(reference).finish(),
        }
    }
}

impl<C: Domain, P: ReferenceDischargePolicy<C>> Display for ReferenceDischargeValue<C, P> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Value(value) => Display::fmt(value, formatter),
            Self::Reference(reference) => Display::fmt(reference, formatter),
        }
    }
}

impl<C: Domain<Value: PartialEq>, P: ReferenceDischargePolicy<C, Alias: PartialEq>> PartialEq
    for ReferenceDischargeValue<C, P>
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Value(value), Self::Value(other)) => value == other,
            (Self::Reference(reference), Self::Reference(other)) => reference == other,
            _ => false,
        }
    }
}

impl<C: Domain<Type: From<ReferenceType<P::Referent>>>, P: ReferenceDischargePolicy<C>> Typed
    for ReferenceDischargeValue<C, P>
{
    type Type = C::Type;

    #[inline]
    fn r#type(&self) -> Cow<'_, C::Type> {
        match self {
            Self::Value(value) => value.r#type(),
            Self::Reference(reference) => Cow::Owned(C::Type::from(reference.r#type().clone())),
        }
    }
}

/// Reference value tracked while reference discharge rewrites a [`Program`]. Each [`ReferenceDischargeReference`]
/// identifies one allocation and denotes either its complete stored value or a view created by
/// [`ReferenceDischargeContext::alias_reference`]. Its alias describes how the [`ReferenceDischargePolicy`] accesses
/// the portion selected by this reference, and its reference type describes that selected portion. When partial
/// discharge preserves the allocation, this value also retains the exact destination reference that it denotes.
///
/// Note that only [`ReferenceDischargeContext`] constructs instances of type while also doing any necessary validation.
pub struct ReferenceDischargeReference<C: Domain, P: ReferenceDischargePolicy<C>> {
    /// Refer to the documentation of [`Self::allocation_id`].
    allocation_id: ReferenceDischargeAllocationId,

    /// Refer to the documentation of [`Self::r#type`].
    r#type: ReferenceType<P::Referent>,

    /// Refer to the documentation of [`Self::is_view`].
    is_view: bool,

    /// Refer to the documentation of [`Self::alias`].
    alias: P::Alias,

    /// Refer to the documentation of [`Self::binding`].
    binding: ReferenceDischargeBinding<C::Value>,
}

impl<C: Domain, P: ReferenceDischargePolicy<C>> ReferenceDischargeReference<C, P> {
    /// Returns the [`ReferenceDischargeAllocationId`] of the allocation that this [`ReferenceDischargeReference`]
    /// denotes.
    pub const fn allocation_id(&self) -> ReferenceDischargeAllocationId {
        self.allocation_id
    }

    /// Returns the [`ReferenceType`] exposed by this [`ReferenceDischargeReference`]. Note that a view can expose a
    /// different type from the type of the allocation's complete stored value.
    pub const fn r#type(&self) -> &ReferenceType<P::Referent> {
        &self.r#type
    }

    /// Returns whether this [`ReferenceDischargeReference`] is a view created by
    /// [`ReferenceDischargeContext::alias_reference`]. Consumption and region boundaries reject views because they
    /// operate on the allocation's complete stored value. This function returns `true` even when the view's reference
    /// type equals the allocation's reference type.
    pub const fn is_view(&self) -> bool {
        self.is_view
    }

    /// Returns the alias that a [`ReferenceDischargePolicy`] can use to access the portion selected by this
    /// [`ReferenceDischargeReference`]. The alias always applies directly to the allocation's complete stored value.
    /// When this reference is created from another view, the alias therefore describes the portion selected by the new
    /// reference relative to the complete stored value, rather than relative only to the input view.
    pub const fn alias(&self) -> &P::Alias {
        &self.alias
    }

    /// Returns the exact destination reference value that this [`ReferenceDischargeReference`] denotes when its
    /// allocation is preserved, or [`None`] when the allocation is discharged. For a view of a preserved allocation,
    /// this is the result of replaying the view operation in the destination [`Program`].
    pub const fn preserved(&self) -> Option<&C::Value> {
        match &self.binding {
            ReferenceDischargeBinding::Discharged => None,
            ReferenceDischargeBinding::Preserved { reference } => Some(reference),
        }
    }

    /// Returns the [`ReferenceDischargeBinding`] that specifies whether the allocation of this
    /// [`ReferenceDischargeReference`] was discharged into explicit state or preserved as a reference in the
    /// destination [`Program`], including the exact destination reference that this value denoted when it was
    /// preserved.
    // TODO(eaplatanios): Make private once the `discharge` module review and cleanup is completed.
    pub(super) const fn binding(&self) -> &ReferenceDischargeBinding<C::Value> {
        &self.binding
    }
}

impl<C: Domain, P: ReferenceDischargePolicy<C>> Clone for ReferenceDischargeReference<C, P> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            allocation_id: self.allocation_id,
            r#type: self.r#type.clone(),
            is_view: self.is_view,
            alias: self.alias.clone(),
            binding: self.binding.clone(),
        }
    }
}

impl<C: Domain, P: ReferenceDischargePolicy<C>> Debug for ReferenceDischargeReference<C, P> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ReferenceDischargeReference")
            .field("allocation_id", &self.allocation_id)
            .field("type", &self.r#type)
            .field("is_view", &self.is_view)
            .field("alias", &self.alias)
            .field("binding", &self.binding)
            .finish()
    }
}

impl<C: Domain, P: ReferenceDischargePolicy<C>> Display for ReferenceDischargeReference<C, P> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{} {}", self.allocation_id, self.r#type)
    }
}

impl<C: Domain<Value: PartialEq>, P: ReferenceDischargePolicy<C, Alias: PartialEq>> PartialEq
    for ReferenceDischargeReference<C, P>
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.allocation_id == other.allocation_id
            && self.r#type == other.r#type
            && self.is_view == other.is_view
            && self.alias == other.alias
            && self.binding == other.binding
    }
}

/// Identity of a reference allocation inside an ongoing reference discharge transform.
/// [`ReferenceDischargeAllocationId`]s are minted by [`ReferenceDischargeContext`] as allocations enter its
/// environment, so they are temporary discharge identities rather than source [`Program`] locations. They exist only
/// for the duration of one discharge transform and are meaningful only against the environment that produced them.
/// Pre-transform identity for caller-facing targets is represented using [`ReferenceDischargeTarget`] instead.
///
/// Each [`ReferenceDischargeAllocationId`] records which [`ReferenceDischargeEnvironment`] minted it, so that an ID
/// from an unrelated discharge is reported rather than silently addressing whichever allocation happens to occupy the
/// same position. That is also what isolates a structured rule's rebuilt [`Region`](crate::Region): rebuilding mints a
/// temporary environment, so a caller ID cannot address a temporary allocation and a temporary ID cannot address a
/// caller allocation. The one table relating the two environments lives inside
/// [`ReferenceDischargeDriver::rebuild_region`], which reports its results in caller terms.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ReferenceDischargeAllocationId {
    /// ID of the [`ReferenceDischargeEnvironment`] that minted this [`ReferenceDischargeAllocationId`].
    environment: ReferenceDischargeEnvironmentId,

    /// Position of the allocation in that [`ReferenceDischargeEnvironment`].
    index: usize,
}

impl Display for ReferenceDischargeAllocationId {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "reference allocation {}:{}", self.environment.0, self.index)
    }
}

/// Representation of a [`ReferenceDischargeReference`] in the destination program. An allocation is represented either
/// as explicit immutable state or as a preserved reference, and that choice does not change after the allocation is
/// bound. The context records the same choice in the handle and its environment entry, so the two cannot disagree and
/// accesses do not need to check them against one another.
// TODO(eaplatanios): Make private once the `discharge` module review and cleanup is completed.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum ReferenceDischargeBinding<V> {
    /// The allocation is represented as explicit immutable state, so accesses through this handle rewrite into read
    /// and write operations against the environment.
    Discharged,

    /// The allocation remains a reference in the destination program. A preserved reference must consume this value
    /// rather than replaying its view chain per access, because doing so would duplicate and reorder the view
    /// operations in the destination program.
    Preserved {
        /// Exact destination reference value this handle denotes.
        reference: V,
    },
}

/// Boundary that a structured reference discharge rule requests for one rebuilt [`Region`](crate::Region) through
/// [`ReferenceDischargeDriver::rebuild_region`]. The rule owns the mapping from its operands onto the region's declared
/// inputs, because that mapping is part of what the operation is. The boundary therefore states the allocation entering
/// at each declared input position, in region order, and separately names the reference-related positions the rebuilt
/// region gains: the allocations that enter as added inputs, the allocations it publishes as added outputs, and the
/// position at which each group is inserted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReferenceDischargeRegionBoundary {
    /// Refer to the documentation of [`Self::declared_input_allocations`].
    declared_input_allocations: Vec<Option<ReferenceDischargeAllocationId>>,

    /// Refer to the documentation of [`Self::capture_input_count`].
    capture_input_count: Option<usize>,

    /// Refer to the documentation of [`Self::added_inputs`].
    added_inputs: ReferenceDischargeRegionStateInsertion,

    /// Refer to the documentation of [`Self::added_outputs`].
    added_outputs: ReferenceDischargeRegionStateInsertion,
}

impl ReferenceDischargeRegionBoundary {
    /// Creates a [`ReferenceDischargeRegionBoundary`] for one rebuilt region. Added positions are described separately
    /// from the declared positions because only the declared positions are replayed. An added input occupies a position
    /// in the rebuilt region's boundary and in the caller's operand list, but the source region's body never named it
    /// and so cannot consume it. The region's capture prefix is read from `operation` rather than supplied by the rule,
    /// so that the prefix used here always agrees with the one that [`ReferenceDischargeContext::region_summary`]
    /// computes from the same operation. A rule therefore never reasons about captures.
    ///
    /// # Parameters
    ///
    ///   - `operation`: [`Operation`] the region is attached to, whose [`Operation::region_capture_input_count`]
    ///     declares the [`Region`](crate::Region)'s own leading capture prefix.
    ///   - `region_index`: Position of the [`Region`](crate::Region) among that [`Operation`]'s attached regions.
    ///   - `declared_input_allocations`: Allocation entering at each declared input position, or [`None`] for a value
    ///     input. Reference positions must come from [`ReferenceDischargeContext::operand_allocation`], which validates
    ///     that each operand carries the complete stored value rather than a view. The length must equal the source
    ///     region's input count, because every declared position is rebuilt.
    ///   - `added_inputs`: Allocations the rebuilt region receives as added inputs, together with the source input
    ///     position at which they are inserted.
    ///   - `added_outputs`: Allocations the rebuilt region publishes as added outputs, together with the source output
    ///     position at which they are inserted.
    #[inline]
    pub fn new<O: Operation>(
        operation: &O,
        region_index: usize,
        declared_input_allocations: Vec<Option<ReferenceDischargeAllocationId>>,
        added_inputs: ReferenceDischargeRegionStateInsertion,
        added_outputs: ReferenceDischargeRegionStateInsertion,
    ) -> Self {
        Self {
            declared_input_allocations,
            capture_input_count: operation.region_capture_input_count(region_index),
            added_inputs,
            added_outputs,
        }
    }

    /// Creates a [`ReferenceDischargeRegionBoundary`] whose added inputs and added outputs are the same allocations
    /// inserted at the same position. This is the loop-carry shape that `while` and `scan` operation bodies thread.
    #[inline]
    pub fn symmetric<O: Operation>(
        operation: &O,
        region_index: usize,
        declared_input_allocations: Vec<Option<ReferenceDischargeAllocationId>>,
        state: ReferenceDischargeRegionStateInsertion,
    ) -> Self {
        Self::new(operation, region_index, declared_input_allocations, state.clone(), state)
    }

    /// Returns the [`ReferenceDischargeAllocationId`] of the allocation entering at each declared input position of
    /// the source region, or [`None`] for a value input, in [`Region`](crate::Region) input order.
    #[inline]
    pub fn declared_input_allocations(&self) -> &[Option<ReferenceDischargeAllocationId>] {
        self.declared_input_allocations.as_slice()
    }

    /// Returns the length of the [`Region`](crate::Region)'s own leading capture prefix, from
    /// [`Operation::region_capture_input_count`], or [`None`] when the region inherits the capture scope of the region
    /// in which its operation is applied.
    pub const fn capture_input_count(&self) -> Option<usize> {
        self.capture_input_count
    }

    /// Returns the allocations the rebuilt [`Region`](crate::Region) receives as added inputs, together with the
    /// position in the source region's input boundary at which they are inserted. A discharged allocation enters as
    /// immutable state and a preserved allocation enters as its destination reference.
    pub const fn added_inputs(&self) -> &ReferenceDischargeRegionStateInsertion {
        &self.added_inputs
    }

    /// Returns the allocations the rebuilt [`Region`](crate::Region) publishes as added outputs, together with the
    /// position in the source region's output boundary at which they are inserted. A discharged allocation publishes
    /// its final state and a preserved allocation publishes its destination reference.
    pub const fn added_outputs(&self) -> &ReferenceDischargeRegionStateInsertion {
        &self.added_outputs
    }
}

/// One group of reference-related positions that a rebuilt [`Region`](crate::Region) gains: the allocations crossing
/// at those positions and the position in the source region's boundary at which the group is inserted. A discharged
/// allocation crosses as immutable state and a preserved allocation crosses as its destination reference.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReferenceDischargeRegionStateInsertion {
    /// Refer to the documentation of [`Self::allocations`].
    allocations: Vec<ReferenceDischargeAllocationId>,

    /// Refer to the documentation of [`Self::position`].
    position: usize,
}

impl ReferenceDischargeRegionStateInsertion {
    /// Creates a [`ReferenceDischargeRegionStateInsertion`] that inserts `allocations` at `position`.
    #[inline]
    pub fn new(allocations: Vec<ReferenceDischargeAllocationId>, position: usize) -> Self {
        Self { allocations, position }
    }

    /// Returns the [`ReferenceDischargeAllocationId`]s of the allocations crossing at this
    /// [`ReferenceDischargeRegionStateInsertion`]'s positions, in canonical allocation order. A discharged
    /// allocation crosses as immutable state and a preserved allocation crosses as its destination reference.
    #[inline]
    pub fn allocations(&self) -> &[ReferenceDischargeAllocationId] {
        self.allocations.as_slice()
    }

    /// Returns the position in the source [`Region`](crate::Region)'s boundary at which this
    /// [`ReferenceDischargeRegionStateInsertion`] is inserted.
    pub const fn position(&self) -> usize {
        self.position
    }
}

/// Result of rebuilding one attached [`Region`](crate::Region) against an isolated reference environment through
/// [`ReferenceDischargeDriver::rebuild_region`]. This result carries the rebuilt [`Program`] and allocation facts
/// stated in the caller's terms, but no values of any kind. A reference produced during the isolated rebuild would keep
/// addressing the abandoned temporary environment, and a destination value produced under a staging destination is a
/// tracer stamped with the temporary builder. Excluding both makes the isolation a type-level fact rather than a
/// convention: the owning rule binds the rebuilt operation in its own context and merges final states from the outputs
/// that binding produces.
#[derive(Debug)]
pub struct ReferenceDischargeRegionResult<V: Value, O: Operation<Type = V::Type>> {
    /// Refer to the documentation of [`Self::program`].
    program: Program<V, O, Vec<V>, Vec<V>>,

    /// Refer to the documentation of [`Self::output_allocations`].
    output_allocations: Vec<Option<ReferenceDischargeAllocationId>>,

    /// Refer to the documentation of [`Self::mutated_allocations`].
    mutated_allocations: Vec<ReferenceDischargeAllocationId>,
}

impl<V: Value, O: Operation<Type = V::Type>> ReferenceDischargeRegionResult<V, O> {
    /// Returns the rebuilt [`Region`](crate::Region) [`Program`] with its reference effects discharged. Its input
    /// boundary is the source region's declared inputs with the boundary's added inputs inserted, and its output
    /// boundary is the source region's declared outputs with the boundary's added outputs inserted.
    #[inline]
    pub const fn program(&self) -> &Program<V, O, Vec<V>, Vec<V>> {
        &self.program
    }

    /// Returns the [`ReferenceDischargeAllocationId`] of the caller allocation each declared [`Region`](crate::Region)
    /// output denotes, or [`None`] for a value output, in region output order.
    #[inline]
    pub fn output_allocations(&self) -> &[Option<ReferenceDischargeAllocationId>] {
        self.output_allocations.as_slice()
    }

    /// Returns the [`ReferenceDischargeAllocationId`] of the caller allocations threaded as state that the
    /// [`Region`](crate::Region) mutated, in canonical allocation order. A preserved reference never appears here,
    /// because its writes replay into the rebuilt region as the operations the source performed and leave no successor
    /// state for the caller to merge.
    #[inline]
    pub fn mutated_allocations(&self) -> &[ReferenceDischargeAllocationId] {
        self.mutated_allocations.as_slice()
    }

    /// Consumes this [`ReferenceDischargeRegionResult`] and returns the rebuilt [`Region`](crate::Region) [`Program`].
    #[inline]
    pub fn into_program(self) -> Program<V, O, Vec<V>, Vec<V>> {
        self.program
    }

    /// Validates that the declared outputs of this [`Region`](crate::Region) denote exactly
    /// the allocations that `expected` predicted. A structured rule sizes its boundary from a
    /// [`ReferenceDischargeRegionSummary`](crate::ReferenceDischargeRegionSummary) computed before the region is
    /// rebuilt, and that boundary depends on the declared output allocations: an allocation the region already returns
    /// publishes its final state at that output position and must not be published a second time. This function holds
    /// that prediction to what the rebuild actually produced, so an operation whose rule disagrees with its own summary
    /// is reported instead of silently losing an update. Checking every region of one operation against the same
    /// summary also keeps those regions in agreement with each other.
    ///
    /// # Parameters
    ///
    ///   - `expected`: Allocation each declared output was predicted to denote, in region output order.
    ///   - `operation`: Name of the operation being rewritten, used in the diagnostic.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when the declared outputs of this region denote different
    /// allocations.
    #[inline]
    pub fn validate_predicted_output_allocations(
        &self,
        expected: &[Option<ReferenceDischargeAllocationId>],
        operation: &str,
    ) -> Result<(), ProgramError> {
        if self.output_allocations != expected {
            return Err(ProgramError::MalformedProgram(format!(
                "operation `{operation}` attaches a region whose outputs do not denote the references its state \
                 widening expected",
            )));
        }
        Ok(())
    }

    /// Validates that this [`Region`](crate::Region) mutated only allocations contained in `published`. A structured
    /// rule decides which final states its rebuilt regions publish from a summary computed before the region is
    /// rebuilt. This function holds that decision to the mutations the rebuild actually performed, so a summary that
    /// under-reports what its operation does is reported here instead of surfacing later as a lost update.
    ///
    /// # Parameters
    ///
    ///   - `published`: Allocations whose final state the rule publishes from this region.
    ///   - `operation`: Name of the operation being rewritten, used in the diagnostic.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] naming the first allocation this region mutated that `published`
    /// does not contain.
    #[inline]
    pub fn validate_predicted_mutations(
        &self,
        published: &[ReferenceDischargeAllocationId],
        operation: &str,
    ) -> Result<(), ProgramError> {
        for allocation in &self.mutated_allocations {
            if !published.contains(allocation) {
                return Err(ProgramError::MalformedProgram(format!(
                    "operation `{operation}` mutated {allocation} in an attached region that its state widening \
                     did not predict",
                )));
            }
        }
        Ok(())
    }
}

/// [`Type`] capability selecting the canonical [`ReferenceDischargePolicy`] used by reference discharge entry points.
/// This is a discharge-owned extension of [`Type`] rather than part of the core type contract. It lets generic
/// [`Program`] functions select the reference policy of each program universe without requiring callers to name that
/// policy or relying on overlapping implementations distinguished only by the program's type family.
pub trait ReferenceDischargeableType: Type {
    /// Canonical [`ReferenceDischargePolicy`] of this type universe.
    type Policy: Copy + Clone + Debug;
}

/// Trait that defines how a [`ReferenceType`] family reads and updates immutable state during reference discharge.
/// Reference discharge replaces mutable references with explicitly threaded immutable values. This policy supplies
/// the family-specific pieces of that rewrite like the referent type, the metadata describing which part of a complete
/// value a reference denotes, and the functions that apply that metadata. Discharge paths that cross a type boundary
/// require the destination universe to embed referent and reference types through [`From`] conversions and recognize
/// reference types through borrowed [`TryFrom`] conversions. Those are the same canonical conversions reference
/// operation type inference uses; the policy defines no parallel conversion seam.
///
/// `C` is the destination [`Domain`] into which discharge writes. Implementations should normally remain generic over
/// compatible destination domains so that the same policy can serve eager and tracing contexts. Ordered accumulation is
/// intentionally separate in [`ReferenceAccumulationPolicy`] because not every reference family supports it.
pub trait ReferenceDischargePolicy<C: Domain> {
    /// Referent [`Type`] family of this reference family.
    type Referent: Type;

    /// Metadata describing which part of a complete stored value a [`Reference`](crate::Reference) denotes.
    /// A reference family with no views can use a unit-like alias whose application is the identity function.
    type Alias: Clone + Debug;

    /// Returns the storage alias for a complete value with the provided referent [`Type`]. Allocation and
    /// entry-boundary binding assign this alias to each new complete-value handle. This is infallible by design.
    /// Validating a referent type is type inference's job, and constructing the identity alias of an already-valid
    /// referent is total.
    fn storage_alias(referent: &Self::Referent) -> Self::Alias;

    /// Returns the value that a [`Reference`](crate::Reference) with `alias` reads from `current`. If `alias` describes
    /// a view into the stored value (e.g., a slice of an array), this function returns only that view. Otherwise, it
    /// returns the complete stored value.
    ///
    /// # Parameters
    ///
    ///   - `context`: Destination context in which any values or operations needed for the read are created.
    ///   - `current`: Complete value currently stored for the reference.
    ///   - `alias`: Describes whether the reference reads all of `current` or a view into it.
    fn read(context: &C, current: &C::Value, alias: &Self::Alias) -> Result<C::Value, ProgramError>;

    /// Returns the complete value that should be stored after a reference with `alias` writes `replacement`. If `alias`
    /// describes a view into the stored value (e.g., a slice of an array), this function replaces only that view and
    /// preserves the rest of `current`.
    ///
    /// # Parameters
    ///
    ///   - `context`: Destination context in which any values or operations needed for the write are created.
    ///   - `current`: Complete value currently stored for the reference.
    ///   - `replacement`: New value to write through the reference.
    ///   - `alias`: Describes whether the reference writes all of `current` or a view into it.
    fn write(
        context: &C,
        current: &C::Value,
        replacement: C::Value,
        alias: &Self::Alias,
    ) -> Result<C::Value, ProgramError>;

    /// Replaces the value of a reference with `replacement` and returns `(previous, updated)`. `previous` is the value
    /// read through the reference before the swap. `updated` is the complete value that should be stored afterward. If
    /// `alias` describes a view into the stored value (e.g., a slice of an array), `updated` preserves the parts of
    /// `current` outside that view.
    ///
    /// The default implementation calls [`Self::read`] followed by [`Self::write`]. Policies may override it when they
    /// can compute both results more efficiently together.
    ///
    /// # Parameters
    ///
    ///   - `context`: Destination context in which any values or operations needed for the swap are created.
    ///   - `current`: Complete value currently stored for the reference.
    ///   - `replacement`: New value to swap into the reference.
    ///   - `alias`: Describes whether the reference swaps all of `current` or a view into it.
    fn swap(
        context: &C,
        current: &C::Value,
        replacement: C::Value,
        alias: &Self::Alias,
    ) -> Result<(C::Value, C::Value), ProgramError> {
        let previous = Self::read(context, current, alias)?;
        let successor = Self::write(context, current, replacement, alias)?;
        Ok((previous, successor))
    }
}

/// Trait that defines additive updates for [`ReferenceType`] families that support them. This contract is separate
/// because accumulation is optional and its destination requirements are family-specific. A family without this
/// implementation can still discharge read, write, and swap operations. However, attempting to discharge a
/// `reference_add_update` operation fails at compile time. An implementation may instead reject selected updates
/// with [`ProgramError::UnsupportedOperation`] when support depends on the particular reference.
pub trait ReferenceAccumulationPolicy<C: Domain>: ReferenceDischargePolicy<C> {
    /// Returns the complete value that should be stored after adding `update` to a reference with `alias`. If `alias`
    /// describes a view into the stored value (e.g., a slice of an array), this function updates only that view and
    /// preserves the rest of `current`.
    ///
    /// # Parameters
    ///
    ///   - `context`: Destination context in which any values or operations needed for the update are created.
    ///   - `current`: Complete value currently stored for the reference.
    ///   - `update`: Value to add through the reference.
    ///   - `alias`: Describes whether the reference updates all of `current` or a view into it.
    fn accumulate(
        context: &C,
        current: &C::Value,
        update: C::Value,
        alias: &Self::Alias,
    ) -> Result<C::Value, ProgramError>;
}

/// Provides one [`Operation`] application with its replay position and with recursive reference discharge for the
/// [`Region`](crate::Region)s attached to it. [`RegionDriver`] supplies the structural region access, and this trait
/// adds the three functions that discharge rules need on top of it. Region-free applications expose a region count of
/// zero through the same contract.
pub trait ReferenceDischargeDriver<C: Domain, P: ReferenceDischargePolicy<C>>:
    RegionDriver<C::Constant, C::Operation>
{
    /// Returns the source [`Program`] location of the [`Operation`] application that is being discharged, or [`None`]
    /// when the application did not come from a replayed [`Instruction`](crate::Instruction). An allocation rule needs
    /// its own target to decide whether the caller selected it for discharge, so replaying a region through
    /// [`inline_region`](Self::inline_region) must supply the source program location of every instruction it replays.
    /// Returning [`None`] declares the allocation unnameable by any [`ReferenceDischargeTarget`] and therefore _always
    /// discharged_, silently ignoring the caller's partial discharge targets.
    fn source_instruction_id(&self) -> Option<InstructionId>;

    /// Discharges the attached region at `index` directly through `context` and returns its outputs. The discharged
    /// instructions are added to the destination program that `context` already owns, and every reference allocation
    /// the region reads or modifies belongs to the same environment as the surrounding operation.
    ///
    /// Use this function when the surrounding operation executes the region during the current rewrite and does not
    /// need to keep that region attached to the rewritten operation. The region uses the surrounding operation's
    /// capture scope, and so this function is valid only when the region inherits that scope. If the rewritten
    /// operation must retain the region, or if the region declares its own capture prefix, use
    /// [`rebuild_region`](Self::rebuild_region) instead.
    ///
    /// # Parameters
    ///
    ///   - `context`: Active [`ReferenceDischargeContext`] whose destination [`Program`] and reference allocations
    ///     receive the [`Region`](crate::Region)'s rewritten instructions and state changes.
    ///   - `index`: Position of the attached [`Region`](crate::Region) in [`Operation`]-defined order.
    ///   - `inputs`: Values supplied to the [`Region`](crate::Region)'s inputs, in region input order.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when this application has no region at `index` or when `inputs` does
    /// not match the region's input boundary. It also propagates errors raised while discharging the region's
    /// instructions.
    fn inline_region(
        &self,
        context: &ReferenceDischargeContext<C, P>,
        index: usize,
        inputs: Vec<ReferenceDischargeValue<C, P>>,
    ) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError>;

    /// Discharges the attached region at `index` into a new destination program and returns that program together with
    /// the reference-allocation information in [`ReferenceDischargeRegionResult`]. The new program uses a separate
    /// reference environment containing only the allocations named by `boundary`, so rebuilding the region does not
    /// modify the allocations in `context`.
    ///
    /// Use this function when the rewritten operation must keep the region attached. The operation's discharge rule
    /// supplies a [`ReferenceDischargeRegionBoundary`] describing how the caller's values and reference allocations
    /// enter and leave the rebuilt region. After rebuilding, the rule validates the returned result, attaches its
    /// program to the rewritten operation, and updates the caller's allocations from that operation's outputs.
    ///
    /// This function and [`inline_region`](Self::inline_region) apply the same reference discharge rules to the
    /// region's instructions. The difference is where those instructions are written and where their reference
    /// allocations live: `inline_region` writes into `context` and updates its allocations directly, whereas this
    /// function returns a separate program and leaves `context` unchanged.
    ///
    /// # Parameters
    ///
    ///   - `context`: Active [`ReferenceDischargeContext`] supplying the current value or preserved reference for every
    ///     allocation named by `boundary`.
    ///   - `index`: Position of the attached [`Region`](crate::Region) in [`Operation`]-defined order.
    ///   - `boundary`: Mapping between the source region's declared inputs and the reference-related inputs and outputs
    ///     required by the rebuilt region.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when this application has no region at `index`, when `boundary` does
    /// not match the region's declared boundary, when an allocation appears more than once where it must be unique, or
    /// when the region returns an allocation that `boundary` did not provide or returns one through a view. It also
    /// propagates errors raised while discharging the region's instructions.
    fn rebuild_region(
        &self,
        context: &ReferenceDischargeContext<C, P>,
        index: usize,
        boundary: &ReferenceDischargeRegionBoundary,
    ) -> Result<ReferenceDischargeRegionResult<C::Constant, C::Operation>, ProgramError>;
}

impl<C: Domain, P: ReferenceDischargePolicy<C>> ReferenceDischargeDriver<C, P> for EmptyRegionDriver {
    #[inline]
    fn source_instruction_id(&self) -> Option<InstructionId> {
        // A region-free application replays no instruction, and so its allocations have no selectable
        // source program location.
        None
    }

    #[inline]
    fn inline_region(
        &self,
        _context: &ReferenceDischargeContext<C, P>,
        _index: usize,
        _inputs: Vec<ReferenceDischargeValue<C, P>>,
    ) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError> {
        Err(ProgramError::MalformedProgram("empty region driver cannot discharge a region".to_string()))
    }

    #[inline]
    fn rebuild_region(
        &self,
        _context: &ReferenceDischargeContext<C, P>,
        _index: usize,
        _boundary: &ReferenceDischargeRegionBoundary,
    ) -> Result<ReferenceDischargeRegionResult<C::Constant, C::Operation>, ProgramError> {
        Err(ProgramError::MalformedProgram("empty region driver cannot rebuild a region".to_string()))
    }
}

/// [`ReferenceDischargeDriver`] scoped to one [`Operation`] application. It borrows the application's complete
/// [`RegionDriver`], which preserves the operation-defined ordering of owned regions, borrowed regions, and shared
/// callees without materializing a combined region collection.
pub struct RecursiveReferenceDischargeDriver<'r, D> {
    /// Application-scoped [`RegionDriver`].
    driver: &'r D,

    /// Source [`Program`] location of the [`Operation`] application, or [`None`] for an application
    /// that replays no [`Instruction`](crate::Instruction).
    source_instruction_id: Option<InstructionId>,
}

impl<'r, D> RecursiveReferenceDischargeDriver<'r, D> {
    /// Creates a new [`RecursiveReferenceDischargeDriver`].
    pub const fn new(driver: &'r D, source_instruction_id: Option<InstructionId>) -> Self {
        Self { driver, source_instruction_id }
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

// Recursive discharge replays the attached region one instruction at a time against the live environment, so an
// allocation created outside the region stays the same allocation inside it and the region's own allocations remain
// distinct. Constants lift into the destination through the parent, exactly as they do at the top level. The nested
// obligation is the one this crate's other structural transforms already carry: rebuilding a region needs this
// universe's operations to discharge into a fresh trace of the same universe as well as into the live destination.
// The requested reference type of a threaded allocation crosses that boundary, so the two policy instantiations must
// agree on their referent type system. Both obligations are stated here rather than on the per-operation rules on
// purpose. A rule that stated them would make the enum dispatcher's obligation graph circular, because the dispatcher's
// own predicate for a structured payload would then demand that the whole enum discharge into the destination whose
// "dischargeability" is what the graph is trying to establish.
impl<
    C: Context<
            Type: From<<P as ReferenceDischargePolicy<C>>::Referent>
                      + From<ReferenceType<<P as ReferenceDischargePolicy<C>>::Referent>>,
            Operation: ReferenceDischargeableOperation<C, P>
                           + ReferenceDischargeableOperation<TracingContext<C::Constant, C::Operation>, P>,
        >,
    P: ReferenceDischargePolicy<C>
        + ReferenceDischargePolicy<
            TracingContext<C::Constant, C::Operation>,
            Referent = <P as ReferenceDischargePolicy<C>>::Referent,
        >,
    D: RegionDriver<C::Constant, C::Operation>,
> ReferenceDischargeDriver<C, P> for RecursiveReferenceDischargeDriver<'_, D>
where
    for<'t> &'t ReferenceType<<P as ReferenceDischargePolicy<C>>::Referent>: TryFrom<&'t C::Type>,
{
    #[inline]
    fn source_instruction_id(&self) -> Option<InstructionId> {
        self.source_instruction_id
    }

    #[inline]
    fn inline_region(
        &self,
        context: &ReferenceDischargeContext<C, P>,
        index: usize,
        inputs: Vec<ReferenceDischargeValue<C, P>>,
    ) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError> {
        context.inline_region(self.region(index)?, inputs)
    }

    fn rebuild_region(
        &self,
        context: &ReferenceDischargeContext<C, P>,
        index: usize,
        boundary: &ReferenceDischargeRegionBoundary,
    ) -> Result<ReferenceDischargeRegionResult<C::Constant, C::Operation>, ProgramError> {
        // Rebuild the source region in a fresh trace with a fresh allocation environment. The caller and rebuilt region
        // can communicate only through the boundary described below: neither side can accidentally retain a handle or
        // value belonging to the other environment.
        let region = self.region(index)?;
        let added_inputs = boundary.added_inputs();
        let added_outputs = boundary.added_outputs();
        check_count!("input", boundary.declared_input_allocations(), region.input_ids().len(), ProgramError);
        let source_input_types = region.input_types();
        let source_input_count = source_input_types.len();
        let source_output_count = region.output_ids().len();
        if added_inputs.position() > source_input_count {
            return Err(ProgramError::MalformedProgram(format!(
                "reference discharge inserts region state inputs at {} but region `{}` declares {} inputs",
                added_inputs.position(),
                region.id(),
                source_input_count,
            )));
        }
        if added_outputs.position() > source_output_count {
            return Err(ProgramError::MalformedProgram(format!(
                "reference discharge inserts region state outputs at {} but region `{}` declares {} outputs",
                added_outputs.position(),
                region.id(),
                source_output_count,
            )));
        }

        // Added state may not land inside the region's own capture prefix. The rebuilt region keeps the prefix length
        // its operation declares, so a state input placed before the end of it would silently renumber the captures the
        // rebound operation still names.
        let capture_input_count = boundary.capture_input_count().unwrap_or(0);
        if added_inputs.position() < capture_input_count {
            return Err(ProgramError::MalformedProgram(format!(
                "reference discharge inserts region state inputs at {} but region `{}` declares a capture prefix of {}",
                added_inputs.position(),
                region.id(),
                capture_input_count,
            )));
        }

        // Every carrier, the region context, and the destination itself stay inside this block, because recovering the
        // rebuilt program below requires unique ownership of the destination's builder.
        let destination = TracingContext::<C::Constant, C::Operation>::new();
        let builder = destination.builder().clone();
        let (output_ids, output_allocations, mutated_allocations) = {
            // The region context inherits the caller's targets because a target names the same source program location
            // wherever the replay reaches it: an unselected allocation inside a rebuilt region survives there exactly
            // as it would have in the caller's own body.
            let region_context =
                ReferenceDischargeContext::<TracingContext<C::Constant, C::Operation>, P>::new_with_targets(
                    destination.clone(),
                    context.targets().clone(),
                );

            let mut declared_allocations = BTreeSet::new();
            declared_allocations.extend(boundary.declared_input_allocations().iter().copied().flatten());
            let mut added_allocations = BTreeSet::new();
            for allocation in added_inputs.allocations() {
                if declared_allocations.contains(allocation) || !added_allocations.insert(*allocation) {
                    return Err(ProgramError::MalformedProgram(format!(
                        "reference discharge adds {} to region `{}` more than once",
                        allocation,
                        region.id(),
                    )));
                }
            }

            // Caller and rebuilt-region allocations live in different environments, so explicit directional maps are
            // the only correspondence between them. Repeated declared positions may intentionally alias one caller
            // allocation, while synthesized state positions were already proven unique (and disjoint from the declared
            // allocations) above. The caller-to-region map is ordered because the mutation-reconciliation loop below
            // iterates it fallibly, and diagnostics must not depend on hash order.
            let mut caller_to_region_allocations =
                BTreeMap::<ReferenceDischargeAllocationId, ReferenceDischargeAllocationId>::new();
            let mut region_to_caller_allocations =
                HashMap::<ReferenceDischargeAllocationId, ReferenceDischargeAllocationId>::new();
            let mut thread = |allocation: ReferenceDischargeAllocationId| -> Result<
                ReferenceDischargeValue<TracingContext<C::Constant, C::Operation>, P>,
                ProgramError,
            > {
                let r#type = context.allocation_entry(allocation)?.r#type().into_owned();
                let discharged = context.is_allocation_discharged(allocation)?;
                let input_type =
                    if discharged { C::Type::from(r#type.referent().clone()) } else { C::Type::from(r#type.clone()) };
                let input = destination.input(input_type);
                if let Some(region_allocation) = caller_to_region_allocations.get(&allocation).copied() {
                    return region_context.allocation_reference(region_allocation);
                }
                let carrier = if discharged {
                    region_context.bind_discharged(r#type, input)?
                } else {
                    region_context.bind_preserved(r#type, input)?
                };
                let region_allocation = carrier.try_as_reference("a threaded region allocation")?.allocation_id();
                caller_to_region_allocations.insert(allocation, region_allocation);
                region_to_caller_allocations.insert(region_allocation, allocation);
                Ok(carrier)
            };

            // Only the declared positions are replayed. An added input occupies a destination boundary position
            // and a caller operand position, but the source region's body never named it and so cannot consume it.
            // A preserved allocation occupies an added position only when an inherited capture is returned without
            // a declared operand.
            let mut declared = Vec::with_capacity(source_input_count);
            for (position, (source_type, allocation)) in
                source_input_types.iter().zip(boundary.declared_input_allocations()).enumerate()
            {
                if position == added_inputs.position() {
                    for allocation in added_inputs.allocations() {
                        thread(*allocation)?;
                    }
                }
                declared.push(match allocation {
                    None => {
                        if <&ReferenceType<<P as ReferenceDischargePolicy<C>>::Referent>>::try_from(source_type).is_ok()
                        {
                            return Err(ProgramError::MalformedProgram(format!(
                                "reference discharge declares reference input {} of region `{}` without an allocation",
                                position,
                                region.id(),
                            )));
                        }
                        ReferenceDischargeValue::Value(destination.input(source_type.clone()))
                    }
                    Some(allocation) => {
                        let Ok(source_reference_type) =
                            <&ReferenceType<<P as ReferenceDischargePolicy<C>>::Referent>>::try_from(source_type)
                        else {
                            return Err(ProgramError::MalformedProgram(format!(
                                "reference discharge assigns {} to value input {} of region `{}`",
                                allocation,
                                position,
                                region.id(),
                            )));
                        };
                        let allocation_type = context.allocation_entry(*allocation)?.r#type().into_owned();
                        if &allocation_type != source_reference_type {
                            return Err(ProgramError::MalformedProgram(format!(
                                "reference discharge assigns {} of type `{}` to input {} of region `{}` \
                                 with reference type `{}`",
                                allocation,
                                allocation_type,
                                position,
                                region.id(),
                                source_reference_type,
                            )));
                        }
                        thread(*allocation)?
                    }
                });
            }

            if added_inputs.position() == source_input_count {
                for allocation in added_inputs.allocations() {
                    thread(*allocation)?;
                }
            }

            // The rebuilt region discharges under a scope naming only its own allocations, so its isolated environment
            // also covers capture-scoped references. A region declaring its own capture prefix reads that prefix from
            // its threaded declared inputs, while every other region inherits the caller's scope mapped onto the region
            // allocations corresponding to its caller allocations. A caller allocation the boundary did not thread
            // binds nothing. Discharged capture accesses and outputs enter as state, while a preserved capture-scoped
            // output enters as its destination reference, so both representations bind region allocations before the
            // inherited scope is established.
            let inherited = context.captures().with_allocations(
                context
                    .captures()
                    .allocations()
                    .iter()
                    .map(|allocation| allocation.and_then(|caller| caller_to_region_allocations.get(&caller).copied()))
                    .collect(),
            );
            let declared_region_allocations = declared
                .iter()
                .map(|input| match input {
                    ReferenceDischargeValue::Value(_) => None,
                    ReferenceDischargeValue::Reference(reference) => Some(reference.allocation_id()),
                })
                .collect::<Vec<_>>();
            let region_context = region_context.with_captures(inherited.nested_scope(
                boundary.capture_input_count(),
                declared_region_allocations.as_slice(),
                region.id(),
            )?);

            let outputs = region_context.inline_region(region, declared)?;
            check_count!("output", outputs, source_output_count, ProgramError);
            let mut output_ids = Vec::with_capacity(source_output_count + added_outputs.allocations().len());
            let mut output_allocations = Vec::with_capacity(source_output_count);
            for position in 0..=source_output_count {
                if position == added_outputs.position() {
                    for allocation in added_outputs.allocations() {
                        let region_allocation =
                            caller_to_region_allocations.get(allocation).copied().ok_or_else(|| {
                                ProgramError::MalformedProgram(format!(
                                    "reference discharge publishes {} from region `{}` without threading it in",
                                    allocation,
                                    region.id(),
                                ))
                            })?;
                        output_ids.push(region_context.allocation_value(region_allocation)?.atom_id()?);
                    }
                }
                let Some(output) = outputs.get(position) else {
                    continue;
                };
                match output {
                    ReferenceDischargeValue::Value(value) => {
                        output_allocations.push(None);
                        output_ids.push(value.atom_id()?);
                    }
                    ReferenceDischargeValue::Reference(reference) => {
                        // A reference-typed region output publishes its allocation at that exact position (i.e., a
                        // discharged reference's current state and a preserved reference's own reference), and the
                        // owning rule maps it back onto the caller allocation through `output_allocations`. An
                        // allocation the caller did not thread has nowhere to be published, which is how a region-local
                        // allocation is stopped from escaping through the boundary.
                        let caller =
                            region_to_caller_allocations.get(&reference.allocation_id()).copied().ok_or_else(|| {
                            ProgramError::MalformedProgram(format!(
                                "reference discharge cannot publish {} from region `{}`, whose caller did not thread \
                                 that allocation",
                                reference.allocation_id(),
                                region.id(),
                            ))
                        })?;

                        // The boundary publishes the complete stored value, so a view cannot cross it. Whoever needs
                        // the view must create it inside the region, just as for a view passed into a region.
                        let whole = region_context.allocation_entry(reference.allocation_id())?.r#type().into_owned();
                        if reference.is_view() {
                            return Err(ProgramError::MalformedProgram(format!(
                                "reference discharge cannot publish the view `{}` of {} from region `{}`, \
                                 whose boundary carries the complete stored value `{}`",
                                reference.r#type(),
                                caller,
                                region.id(),
                                whole,
                            )));
                        }
                        output_allocations.push(Some(caller));
                        output_ids.push(match reference.preserved() {
                            Some(value) => value.atom_id()?,
                            None => region_context.discharged_state(reference.allocation_id())?.atom_id()?,
                        });
                    }
                }
            }

            // Only threaded *state* can have been mutated. A preserved reference's writes replayed into the rebuilt
            // region as the operations the source performed, so there is no successor state for the caller to merge.
            let mut mutated_allocations = BTreeSet::new();
            for (caller, region_allocation) in &caller_to_region_allocations {
                if region_context.is_allocation_discharged(*region_allocation)?
                    && region_context.is_mutated(*region_allocation)?
                {
                    mutated_allocations.insert(*caller);
                }
            }
            let mutated_allocations = mutated_allocations.into_iter().collect::<Vec<_>>();
            (output_ids, output_allocations, mutated_allocations)
        };
        drop(destination);

        let input_count = source_input_count + added_inputs.allocations().len();
        let output_count = output_ids.len();
        let builder = Rc::try_unwrap(builder).map_err(|_| ProgramError::EscapedProgramBuilder)?.into_inner();
        let program = builder.build(output_ids, vec![Placeholder; input_count], vec![Placeholder; output_count])?;
        Ok(ReferenceDischargeRegionResult { program, output_allocations, mutated_allocations })
    }
}

// TODO(eaplatanios): Restore the strict `Operation<Type = C::Type>` super-trait bound once the next-generation trait
//  solver stabilizes. The current solver cannot discharge this projection equality at implementation heads whose
//  context type is built from `Self` (E0284); the equality is enforced per function through `where` clauses instead.
/// Represents [`Operation`]s that can be discharged (i.e., rewritten so that the references they touch become explicit
/// immutable state).
///
/// The trait is parameterized by the destination [`Domain`] `C` that owns the rewritten values and by the
/// [`ReferenceDischargePolicy`] `P` naming the reference universe being discharged. Every rule receives the active
/// [`ReferenceDischargeContext`], which owns the allocation environment, plus a [`ReferenceDischargeDriver`] exposing
/// the application's replay position and attached regions.
///
/// Reference primitives implement their own rewrites (e.g., an allocation binds a fresh allocation, a read/write access
/// acts on the allocation's current state through the policy's alias mechanics, and a freeze yields the current state
/// and unbinds the allocation). Structured operations implement their own boundary widening, because widening is a
/// property of what the operation does with its regions and therefore belongs to the operation. Everything else replays
/// as-is over rewritten operands. The system is consequently open over primitives: a third-party operation family
/// participates by implementing this trait, with no companion declaration surface beyond the generic
/// [`Operation::reference_semantics`] and region-provenance hooks it already implements.
///
/// Access rules see only _discharged_ allocations. When partial discharge preserves an allocation, the dispatch path
/// replays every region-free, access-only application over it verbatim before rule dispatch, so an access rule never
/// needs a preserved branch of its own. The exceptions own their preserved handling because their outputs mint or
/// alias references (i.e., an allocation rule consults its replay position against the targets, and a view rule calls
/// [`ReferenceDischargeContext::alias_reference`], which replays the view over a preserved parent's destination value).
///
/// `C` is bounded by [`Domain`] rather than [`Context`] for the same reason as with
/// [`InterpretableOperation`](crate::InterpretableOperation): the destination context's own binding contract is
/// established in terms of its operation family's rules, so reaching [`Context`] through this trait would make that
/// obligation recursive. Implementations bound `C` by the value and conversion capabilities their rewrite actually
/// uses, and higher-order rules request nested work through their driver rather than carrying a bound stating that
/// their own operation family is dischargeable, which is what keeps an operation enum's bound graph finite.
///
/// The super-trait is a plain [`Operation`] rather than `Operation<Type = C::Type>` because the current trait solver
/// cannot discharge that projection equality at implementation heads whose reference discharge context is itself built
/// from `Self`. The equality is instead required per function through `where Self: Operation<Type = C::Type>`, so a
/// payload whose [`Operation::Type`] disagrees with `C::Type` cannot be batched in `C`: the requirement is restated by
/// the dispatcher's per-payload predicates and by the generic projected-discharge helpers, and any mismatched payload
/// is rejected with a type-mismatch error at its use site.
pub trait ReferenceDischargeableOperation<C: Domain, P: ReferenceDischargePolicy<C>>: Operation {
    /// Rewrites this [`Operation`] application so that the references it touches become explicit immutable state,
    /// and returns the carrier [`ReferenceDischargeValue`]s its outputs produce.
    ///
    /// # Parameters
    ///
    ///   - `context`: Active discharge context owning the allocation environment, through whose
    ///     [`ReferenceDischargeContext::parent`] the rewritten work is bound.
    ///   - `driver`: Application-scoped [`ReferenceDischargeDriver`] exposing the replay position and any attached
    ///     [`Region`](crate::Region)s.
    ///   - `inputs`: Carrier [`ReferenceDischargeValue`]s supplied as this application's operands,
    ///     in [`Operation`]-defined order.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError`] when this application cannot be rewritten because an operand is of the wrong kind,
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

/// Active state of a reference discharge transform. Reference discharge interprets a source [`Program`] into a
/// destination [`Program`], one [`Region`](crate::Region) at a time through a [`ReferenceDischargeDriver`]. Each
/// replayed [`Instruction`](crate::Instruction) dispatches to its [`ReferenceDischargeableOperation`] implementation
/// with this context, and that implementation emits destination work through [`parent`](Self::parent).
///
/// Each source reference allocation is bound into this context exactly once. [`bind_discharged`](Self::bind_discharged)
/// records an allocation as explicit immutable state, while [`bind_preserved`](Self::bind_preserved) records the
/// exact destination reference value when partial discharge leaves the allocation intact. The allocation remains in
/// that representation for the rest of the transform.
/// A [`ReferenceDischargeableOperation`] implementation uses [`alias_reference`](Self::alias_reference) to construct
/// another view of the same allocation, then either rewrites accesses through [`read`](Self::read),
/// [`write`](Self::write), [`swap`](Self::swap), and [`accumulate`](Self::accumulate), or replays
/// them against the preserved destination reference.
///
/// The allocation environment lives on the context rather than on flowing values because references carry identity:
/// several reference values can denote different views of the same allocation, and every one of them must observe the
/// same current state and liveness. Clones therefore share one environment. A structured rule that must rebuild an
/// attached region instead uses [`ReferenceDischargeDriver::rebuild_region`], which creates an isolated environment
/// and commits nothing here until the rule explicitly merges its outputs.
pub struct ReferenceDischargeContext<C: Domain, P: ReferenceDischargePolicy<C>> {
    /// Destination context that owns the discharged values and executes or stages the rewritten work.
    parent: C,

    /// [`ReferenceDischargeEnvironment`] shared by every clone of this context.
    environment: Rc<RefCell<ReferenceDischargeEnvironment<P::Referent, C::Value>>>,

    /// [`ReferenceDischargeCaptureScope`] that contains allocations the capture prefix of the scope this context
    /// discharges binds. A region that inherits its parent's capture prefix discharges under the same scope. A region
    /// rebuilt in isolation reconstructs the scope using allocations from its temporary environment.
    captures: ReferenceDischargeCaptureScope<C::Constant>,

    /// [`ReferenceDischargeTargets`] that the current reference discharge transform normalizes into immutable state.
    /// Every allocation they omit is preserved, and every clone and isolated region rebuild shares the targets
    /// unchanged because a target names the same source program location wherever the replay reaches it.
    targets: ReferenceDischargeTargets,
}

impl<C: Domain, P: ReferenceDischargePolicy<C>> ReferenceDischargeContext<C, P> {
    /// Creates a new [`ReferenceDischargeContext`] over `parent` that discharges every reference it reaches. The
    /// context starts with no live allocations or capture bindings. Capture bindings are populated while the program
    /// boundary is threaded because they refer to allocations minted by this context. To request partial discharge,
    /// use [`Program::partially_discharge_references`](crate::Program::partially_discharge_references) instead of
    /// constructing a context directly; that function validates the targets against the program in which they were
    /// identified.
    #[inline]
    pub fn new(parent: C) -> Self {
        Self::new_with_targets(parent, ReferenceDischargeTargets::everything())
    }

    /// Creates a new [`ReferenceDischargeContext`] with an empty [`ReferenceDischargeEnvironment`] and an empty
    /// [`ReferenceDischargeCaptureScope`] over the provided destination context, discharging exactly the references
    /// named by `targets`.
    #[inline]
    pub fn new_with_targets(parent: C, targets: ReferenceDischargeTargets) -> Self {
        Self {
            parent,
            environment: Rc::new(RefCell::new(ReferenceDischargeEnvironment {
                id: ReferenceDischargeEnvironmentId::fresh(),
                allocations: Vec::new(),
            })),
            captures: ReferenceDischargeCaptureScope::default(),
            targets,
        }
    }

    /// Returns this [`ReferenceDischargeContext`] discharging under a different [`ReferenceDischargeCaptureScope`],
    /// sharing its [`ReferenceDischargeEnvironment`]. An isolated region rebuild reaches its own scope this way because
    /// the temporary environment mints the allocations that scope binds only after its boundary is threaded.
    #[inline]
    pub fn with_captures(&self, captures: ReferenceDischargeCaptureScope<C::Constant>) -> Self
    where
        C: Clone,
    {
        Self {
            parent: self.parent.clone(),
            environment: Rc::clone(&self.environment),
            captures,
            targets: self.targets.clone(),
        }
    }

    /// Returns the destination context that owns the discharged values of this [`ReferenceDischargeContext`].
    pub const fn parent(&self) -> &C {
        &self.parent
    }

    /// Returns the [`ReferenceDischargeCaptureScope`] that this [`ReferenceDischargeContext`] discharges under.
    pub const fn captures(&self) -> &ReferenceDischargeCaptureScope<C::Constant> {
        &self.captures
    }

    /// Returns the [`ReferenceDischargeTargets`] that this [`ReferenceDischargeContext`] discharges.
    pub const fn targets(&self) -> &ReferenceDischargeTargets {
        &self.targets
    }

    /// Returns whether the allocation an [`Instruction`](crate::Instruction) performs was selected for discharge,
    /// which is what an allocation rule asks before deciding between a discharged reference and one that survives in
    /// the destination. An operation application that did not come from a replayed instruction (i.e., a region-free
    /// rule invocation through [`EmptyRegionDriver`](crate::programs::EmptyRegionDriver)) has no source program
    /// location and is always discharged as no [`ReferenceDischargeTarget`] can name it, and so declining it would
    /// express nothing about the caller's choice.
    ///
    /// This is the only target query a rule ever makes, which is why it is the only one exposed. Whether an entry
    /// boundary allocation was selected is decided once, by the program-level entry point that threads the boundary,
    /// and no rule is in a position to ask it.
    ///
    /// # Parameters
    ///
    ///   - `source_instruction_id`: Source program location of the application, from
    ///     [`ReferenceDischargeDriver::source_instruction_id`].
    ///   - `output_index`: Output position at which the application defines the fresh allocation.
    #[inline]
    pub fn selects_internal(&self, source_instruction_id: Option<InstructionId>, output_index: usize) -> bool {
        source_instruction_id.is_none_or(|instruction| {
            self.targets.selects(ReferenceDischargeTarget::Internal { instruction, output_index })
        })
    }

    /// Returns whether one entry boundary allocation was selected for discharge.
    #[inline]
    pub fn selects_external(&self, source: ReferenceSource) -> bool {
        self.targets.selects(ReferenceDischargeTarget::External(source))
    }

    /// Returns the complete current immutable state of one live discharged allocation. Operation rules normally use
    /// [`read`](Self::read) to observe the portion selected by a reference value. Structured operation implementations
    /// use this function when they must thread the allocation's complete state across a destination boundary.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when `allocation` is unknown, belongs to another environment, has
    /// already been consumed, or was preserved rather than discharged.
    #[inline]
    pub fn discharged_state(&self, allocation: ReferenceDischargeAllocationId) -> Result<C::Value, ProgramError> {
        match &self.allocation_entry(allocation)?.state {
            ReferenceDischargeAllocationState::Discharged { current, .. } => Ok(current.clone()),
            ReferenceDischargeAllocationState::Preserved { .. } => Err(ProgramError::MalformedProgram(format!(
                "reference discharge requested the discharged state of preserved {allocation}",
            ))),
        }
    }

    /// Sets the complete current state of one live discharged allocation. If `mutated` is `true`, this function also
    /// marks the allocation as mutated; passing `false` preserves its existing mutation status and never clears a prior
    /// mutation mark. Reference operation functions pass `true`. Structured boundary code passes its access summary
    /// because symmetric boundaries also return unchanged state for read-only allocations, which must not cause those
    /// allocations to publish hidden final-state outputs.
    ///
    /// # Parameters
    ///
    ///   - `allocation`: Live discharged allocation whose complete state is being set.
    ///   - `current`: New complete immutable state.
    ///   - `mutated`: Whether this transition should mark the allocation as mutated.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when `allocation` is not a live discharged allocation or `current`
    /// does not carry its referent type.
    pub fn set_discharged_state(
        &self,
        allocation: ReferenceDischargeAllocationId,
        current: C::Value,
        mutated: bool,
    ) -> Result<(), ProgramError>
    where
        C::Type: From<P::Referent>,
    {
        // Validate before taking the mutable environment borrow so a type error leaves both the current state and its
        // mutation bit unchanged.
        let r#type = self.allocation_entry(allocation)?.r#type().into_owned();
        let expected = C::Type::from(r#type.referent().clone());
        let actual = current.r#type();
        if actual.as_ref() != &expected {
            return Err(ProgramError::MalformedProgram(format!(
                "reference discharge state has type `{actual}` but allocation `{type}` requires `{expected}`",
            )));
        }
        let mut environment = self.environment.borrow_mut();
        environment.entry(allocation)?;
        match environment.allocations[allocation.index].as_mut().map(|entry| &mut entry.state) {
            Some(ReferenceDischargeAllocationState::Discharged { current: state, mutated: previous_mutated }) => {
                *state = current;
                *previous_mutated |= mutated;
                Ok(())
            }
            Some(ReferenceDischargeAllocationState::Preserved { .. }) => Err(ProgramError::MalformedProgram(format!(
                "reference discharge updated the state of preserved {allocation}",
            ))),
            None => Err(ProgramError::MalformedProgram(format!("reference discharge accessed consumed {allocation}"))),
        }
    }

    /// Returns whether one live discharged allocation has been mutated during this transform. A direct write, swap, or
    /// accumulation marks the allocation as mutated. Structured operation implementations use this fact to publish only
    /// final states that the source program could have changed; read-only state need not become a hidden output.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when `allocation` is unknown, belongs to another environment, has
    /// already been consumed, or was preserved rather than discharged.
    #[inline]
    pub fn is_mutated(&self, allocation: ReferenceDischargeAllocationId) -> Result<bool, ProgramError> {
        match &self.allocation_entry(allocation)?.state {
            ReferenceDischargeAllocationState::Discharged { mutated, .. } => Ok(*mutated),
            ReferenceDischargeAllocationState::Preserved { .. } => Err(ProgramError::MalformedProgram(format!(
                "reference discharge queried mutation of preserved {allocation}",
            ))),
        }
    }

    /// Returns the [`ReferenceDischargeAllocationId`] of every allocation still live in this context, in binding order.
    /// Consumption retains the allocation's environment slot but removes its entry, so consumed allocations are omitted
    /// while the relative order of all remaining allocations stays stable.
    #[inline]
    pub fn live_allocation_ids(&self) -> Vec<ReferenceDischargeAllocationId> {
        let environment = self.environment.borrow();
        environment
            .allocations
            .iter()
            .enumerate()
            .filter(|(_, state)| state.is_some())
            .map(|(index, _)| ReferenceDischargeAllocationId { environment: environment.id, index })
            .collect()
    }

    /// Returns whether one live allocation is discharged rather than preserved. This function is useful when structured
    /// operation code has only an allocation ID and must choose whether to thread immutable state or a destination
    /// reference value.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when `allocation` is unknown, belongs to another environment,
    /// or has already been consumed.
    #[inline]
    pub fn is_allocation_discharged(&self, allocation: ReferenceDischargeAllocationId) -> Result<bool, ProgramError> {
        Ok(matches!(&self.allocation_entry(allocation)?.state, ReferenceDischargeAllocationState::Discharged { .. }))
    }

    /// Returns an unviewed [`ReferenceDischargeValue`] denoting a live allocation already bound in this context. This
    /// function does not bind another allocation. Region threading and capture resolution use it to reconstruct the
    /// complete value reference associated with an allocation ID. A preserved allocation carries the exact destination
    /// reference originally supplied to [`bind_preserved`](Self::bind_preserved).
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when `allocation` is unknown, belongs to another environment,
    /// or has already been consumed.
    pub fn allocation_reference(
        &self,
        allocation: ReferenceDischargeAllocationId,
    ) -> Result<ReferenceDischargeValue<C, P>, ProgramError> {
        let (r#type, binding) = {
            let entry = self.allocation_entry(allocation)?;
            let binding = match &entry.state {
                ReferenceDischargeAllocationState::Discharged { .. } => ReferenceDischargeBinding::Discharged,
                ReferenceDischargeAllocationState::Preserved { reference } => {
                    ReferenceDischargeBinding::Preserved { reference: reference.clone() }
                }
            };
            (entry.r#type().into_owned(), binding)
        };
        let alias = P::storage_alias(r#type.referent());
        Ok(ReferenceDischargeValue::Reference(ReferenceDischargeReference {
            allocation_id: allocation,
            r#type,
            is_view: false,
            alias,
            binding,
        }))
    }

    /// Returns an immutable borrow of one live [`ReferenceDischargeAllocationEntry`]. The returned guard keeps the
    /// allocation environment immutably borrowed. Callers should copy or clone the fields they need and let the guard
    /// drop before invoking any operation that may borrow the environment mutably.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when `allocation` belongs to another environment, was never bound,
    /// or has already been consumed.
    pub(crate) fn allocation_entry(
        &self,
        allocation: ReferenceDischargeAllocationId,
    ) -> Result<Ref<'_, ReferenceDischargeAllocationEntry<P::Referent, C::Value>>, ProgramError> {
        let environment = self.environment.borrow();
        if environment.entry(allocation)?.is_none() {
            return Err(ProgramError::MalformedProgram(format!("reference discharge accessed consumed {allocation}")));
        }
        Ok(Ref::map(environment, |environment| {
            // The check above proved that this exact position exists and contains a live entry.
            environment.allocations[allocation.index].as_ref().unwrap()
        }))
    }

    /// Removes and returns one live [`ReferenceDischargeAllocationEntry`].
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when `allocation` belongs to another environment, was never bound,
    /// or has already been consumed.
    pub(crate) fn take_allocation_entry(
        &self,
        allocation: ReferenceDischargeAllocationId,
    ) -> Result<ReferenceDischargeAllocationEntry<P::Referent, C::Value>, ProgramError> {
        let mut environment = self.environment.borrow_mut();
        environment.entry(allocation)?;
        environment.allocations[allocation.index].take().ok_or_else(|| {
            ProgramError::MalformedProgram(format!("reference discharge accessed consumed {allocation}"))
        })
    }

    /// Binds an allocation selected for discharge and returns its unviewed reference value. The allocation is fresh to
    /// this context even when it represents a reference that already existed at the source program's entry boundary.
    /// Its `initial` value becomes the immutable state exposed by [`discharged_state`](Self::discharged_state),
    /// observed through [`read`](Self::read), and transformed through [`write`](Self::write), [`swap`](Self::swap),
    /// and [`accumulate`](Self::accumulate).
    ///
    /// # Parameters
    ///
    ///   - `r#type`: [`ReferenceType`] of the allocation.
    ///   - `initial`: Destination value that becomes the allocation's initial immutable state.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when `initial` does not carry the lifted referent type of `r#type`.
    pub fn bind_discharged(
        &self,
        r#type: ReferenceType<P::Referent>,
        initial: C::Value,
    ) -> Result<ReferenceDischargeValue<C, P>, ProgramError>
    where
        C::Type: From<P::Referent>,
    {
        let expected = C::Type::from(r#type.referent().clone());
        let actual = initial.r#type();
        if actual.as_ref() != &expected {
            return Err(ProgramError::MalformedProgram(format!(
                "reference discharge state has type `{actual}` but allocation `{type}` requires `{expected}`",
            )));
        }
        Ok(self.bind_allocation(
            r#type,
            ReferenceDischargeAllocationState::Discharged { current: initial, mutated: false },
            ReferenceDischargeBinding::Discharged,
        ))
    }

    /// Binds an allocation preserved by partial discharge and returns its unviewed reference value. The environment
    /// retains `reference` so structured boundaries can thread the allocation, while each alias retains the exact
    /// destination value produced when its view operation is replayed. A preserved allocation never becomes discharged
    /// later in this transform.
    ///
    /// # Parameters
    ///
    ///   - `r#type`: [`ReferenceType`] of the allocation.
    ///   - `reference`: Destination reference-typed value denoting the allocation.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when `reference` does not carry the reference type `r#type`.
    pub fn bind_preserved(
        &self,
        r#type: ReferenceType<P::Referent>,
        reference: C::Value,
    ) -> Result<ReferenceDischargeValue<C, P>, ProgramError>
    where
        for<'t> &'t ReferenceType<P::Referent>: TryFrom<&'t C::Type>,
    {
        let reference_type = reference.r#type();
        let actual = <&ReferenceType<P::Referent>>::try_from(reference_type.as_ref()).map_err(|_| {
            ProgramError::MalformedProgram(format!(
                "reference discharge preserved an allocation as `{reference_type}` which is not a reference type",
            ))
        })?;
        if actual != &r#type {
            return Err(ProgramError::MalformedProgram(format!(
                "reference discharge preserved an allocation as `{actual}` but its handle exposes `{type}`",
            )));
        }
        Ok(self.bind_allocation(
            r#type,
            ReferenceDischargeAllocationState::Preserved { reference: reference.clone() },
            ReferenceDischargeBinding::Preserved { reference },
        ))
    }

    /// Inserts one validated allocation entry and returns its unviewed reference value. [`Self::bind_discharged`]
    /// validates the initial immutable state, while [`Self::bind_preserved`] validates the destination reference value,
    /// before calling this shared environment primitive.
    fn bind_allocation(
        &self,
        r#type: ReferenceType<P::Referent>,
        state: ReferenceDischargeAllocationState<C::Value>,
        binding: ReferenceDischargeBinding<C::Value>,
    ) -> ReferenceDischargeValue<C, P> {
        let alias = P::storage_alias(r#type.referent());
        let allocation = {
            // The environment owns the complete type and state even after every handle to this allocation disappears.
            // Keep the mutable borrow scoped to inserting that record; the returned handle carries only its identity.
            let mut environment = self.environment.borrow_mut();
            environment
                .allocations
                .push(Some(ReferenceDischargeAllocationEntry { r#type: r#type.clone(), state }));
            ReferenceDischargeAllocationId { environment: environment.id, index: environment.allocations.len() - 1 }
        };
        ReferenceDischargeValue::Reference(ReferenceDischargeReference {
            allocation_id: allocation,
            r#type,
            is_view: false,
            alias,
            binding,
        })
    }

    /// Creates another [`ReferenceDischargeValue`] that aliases the same allocation with the provided composed view
    /// and exposed [`ReferenceType`]. This function creates another handle rather than binding a new allocation. The
    /// returned reference keeps the input reference's allocation identity, cannot denote the allocation's complete
    /// value, and carries `alias` as its authoritative complete view chain rather than merely its newest view step.
    ///
    /// For a discharged allocation, later accesses apply that chain to the allocation's immutable state and
    /// `replay_preserved_view_fn` is never called. For a preserved allocation, `replay_preserved_view_fn` must replay
    /// the source view operation against the parent reference's exact destination value and return that operation's
    /// single reference result. The returned destination value is retained on the alias so later accesses use it
    /// directly instead of replaying the view again. The function should perform only that replay because whether it
    /// is called depends on whether partial discharge preserved the allocation.
    ///
    /// # Parameters
    ///
    ///   - `reference`: Reference value being aliased.
    ///   - `alias`: Complete composed view chain of the alias.
    ///   - `r#type`: Reference type the alias exposes.
    ///   - `replay_preserved_view_fn`: Function that replays the source view operation against the parent destination
    ///     reference and returns its single reference result. It is called exactly once for a preserved allocation and
    ///     is never called for a discharged allocation.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when the allocation is no longer live or when the replayed destination
    /// value does not carry the reference type `r#type`, and propagates every `replay_preserved_view_fn` failure.
    pub fn alias_reference<F: FnOnce(&C::Value) -> Result<C::Value, ProgramError>>(
        &self,
        reference: &ReferenceDischargeReference<C, P>,
        alias: P::Alias,
        r#type: ReferenceType<P::Referent>,
        replay_preserved_view_fn: F,
    ) -> Result<ReferenceDischargeValue<C, P>, ProgramError>
    where
        for<'t> &'t ReferenceType<P::Referent>: TryFrom<&'t C::Type>,
    {
        let allocation = reference.allocation_id();

        // Handles can outlive the allocation they denote, so resolve the ID against the active environment before
        // creating another alias. This reports foreign, never-bound, and consumed allocations at the attempted use.
        self.allocation_entry(allocation)?;

        let binding = match &reference.binding {
            ReferenceDischargeBinding::Discharged => ReferenceDischargeBinding::Discharged,
            ReferenceDischargeBinding::Preserved { reference: parent } => {
                let replayed = replay_preserved_view_fn(parent)?;
                let replayed_type = replayed.r#type();
                let actual = <&ReferenceType<P::Referent>>::try_from(replayed_type.as_ref()).map_err(|_| {
                    ProgramError::MalformedProgram(format!(
                        "reference discharge preserved an allocation as `{replayed_type}` \
                         which is not a reference type",
                    ))
                })?;
                if actual != &r#type {
                    return Err(ProgramError::MalformedProgram(format!(
                        "reference discharge preserved an allocation as `{actual}` but its handle exposes `{type}`",
                    )));
                }
                ReferenceDischargeBinding::Preserved { reference: replayed }
            }
        };

        Ok(ReferenceDischargeValue::Reference(ReferenceDischargeReference {
            allocation_id: allocation,
            r#type,
            is_view: true,
            alias,
            binding,
        }))
    }

    /// Reads the portion that `reference` selects from its discharged allocation's current state. Reference operation
    /// rules call this function only for discharged references. An access to a preserved reference must instead replay
    /// the source operation against [`ReferenceDischargeReference::preserved`].
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when the allocation is not live, and propagates the policy's error
    /// when the alias cannot be applied. Reading a preserved reference through this function is rejected, because a
    /// preserved access must replay verbatim in the destination instead.
    pub fn read(&self, reference: &ReferenceDischargeReference<C, P>) -> Result<C::Value, ProgramError> {
        let current = self.discharged_state(reference.allocation_id())?;
        P::read(&self.parent, &current, reference.alias())
    }

    /// Replaces the portion that `reference` selects in a discharged allocation. The policy returns a complete
    /// successor state, which this function installs through [`set_discharged_state`](Self::set_discharged_state)
    /// and records as a mutation.
    ///
    /// # Parameters
    ///
    ///   - `reference`: Handle selecting the portion to replace.
    ///   - `replacement`: Value written into the selected portion.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when the allocation is not live or was preserved, and propagates the
    /// policy's error when the write cannot be applied.
    pub fn write(
        &self,
        reference: &ReferenceDischargeReference<C, P>,
        replacement: C::Value,
    ) -> Result<(), ProgramError>
    where
        C::Type: From<P::Referent>,
    {
        let allocation = reference.allocation_id();
        let current = self.discharged_state(allocation)?;
        let successor = P::write(&self.parent, &current, replacement, reference.alias())?;
        self.set_discharged_state(allocation, successor, true)
    }

    /// Replaces the portion that `reference` selects and returns its previous contents. Like [`write`](Self::write),
    /// this function installs the policy's complete successor state and records a mutation. Unlike `write`, it also
    /// returns the value selected before replacement.
    ///
    /// # Parameters
    ///
    ///   - `reference`: Handle selecting the portion to replace.
    ///   - `replacement`: Value written into the selected portion.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when the allocation is not live or was preserved, and propagates the
    /// policy's error when the alias cannot be applied.
    pub fn swap(
        &self,
        reference: &ReferenceDischargeReference<C, P>,
        replacement: C::Value,
    ) -> Result<C::Value, ProgramError>
    where
        C::Type: From<P::Referent>,
    {
        let allocation = reference.allocation_id();
        let current = self.discharged_state(allocation)?;
        let (previous, successor) = P::swap(&self.parent, &current, replacement, reference.alias())?;
        self.set_discharged_state(allocation, successor, true)?;
        Ok(previous)
    }

    /// Accumulates `update` into the portion that `reference` selects in a discharged allocation. The policy returns a
    /// complete successor state, which this function installs and records as a mutation.
    ///
    /// # Parameters
    ///
    ///   - `reference`: Handle selecting the portion to update.
    ///   - `update`: Value accumulated into the selected portion.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when the allocation is not live or was preserved, and propagates the
    /// policy's error when the alias cannot be applied or the universe forbids accumulation.
    pub fn accumulate(
        &self,
        reference: &ReferenceDischargeReference<C, P>,
        update: C::Value,
    ) -> Result<(), ProgramError>
    where
        C::Type: From<P::Referent>,
        P: ReferenceAccumulationPolicy<C>,
    {
        let allocation = reference.allocation_id();
        let current = self.discharged_state(allocation)?;
        let successor = P::accumulate(&self.parent, &current, update, reference.alias())?;
        self.set_discharged_state(allocation, successor, true)
    }

    /// Consumes a discharged allocation and returns its complete current immutable state. Consumption removes the
    /// allocation's live environment entry, so every later access reports a use-after-consume. It always yields the
    /// complete stored value and deliberately ignores aliases; only the unviewed reference value returned when the
    /// allocation was bound can therefore name the transition. For a preserved allocation, the replay path performs
    /// the destination operation first and then removes the corresponding live environment entry.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when the allocation is not live, was preserved rather than
    /// discharged, or is named through a view rather than the reference for its complete stored value.
    pub fn consume(&self, reference: &ReferenceDischargeReference<C, P>) -> Result<C::Value, ProgramError> {
        let allocation = reference.allocation_id();
        let referent_type = match &self.allocation_entry(allocation)?.state {
            ReferenceDischargeAllocationState::Discharged { current, .. } => Ok(current.r#type().into_owned()),
            ReferenceDischargeAllocationState::Preserved { .. } => Err(ProgramError::MalformedProgram(format!(
                "reference discharge requested the discharged state of preserved {allocation}",
            ))),
        }?;
        if reference.is_view() {
            return Err(ProgramError::MalformedProgram(format!(
                "reference discharge cannot consume {} through the view `{}`; consumption yields the \
                 complete stored value, whose referent is `{}`",
                allocation,
                reference.r#type(),
                referent_type,
            )));
        }
        let entry = self.take_allocation_entry(allocation)?;
        let ReferenceDischargeAllocationState::Discharged { current, .. } = entry.state else { unreachable!() };
        Ok(current)
    }

    /// Summarizes the transitive reference accesses of a [`Region`](crate::Region) closure, in the terms of the caller
    /// allocations its boundary names. A structured rule calls this before it can size its state boundary (i.e., which
    /// allocations a region closure touches, and which of them it mutates, is exactly what decides how wide the
    /// rewritten operation must be). The summary is computed entirely from generic hooks, namely operation-local
    /// [`Operation::reference_semantics`], the region-provenance hooks, reference-output identity, and recursive
    /// summaries of nested regions, so a third-party structured operation needs no companion declaration surface
    /// to be summarized.
    ///
    /// The region's own capture scope is computed here rather than supplied, because whether a region establishes a
    /// fresh capture prefix is stated by [`Operation::region_capture_input_count`] and is therefore knowledge the
    /// summary can read off the operation itself. A rule never has to reason about captures.
    ///
    /// # Parameters
    ///
    ///   - `operation`: [`Operation`] the [`Region`](crate::Region) is attached to.
    ///   - `region_index`: Position of the [`Region`](crate::Region) among that [`Operation`]'s attached regions.
    ///   - `region`: [`Region`](crate::Region) whose closure is summarized.
    ///   - `inputs`: Caller allocation denoted by each of the region's declared inputs, in boundary order, with
    ///     [`None`] wherever the position carries a value.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when `inputs` does not describe the region's boundary, when the
    /// operation declares a capture prefix longer than the region's boundary, when a reference-typed nested boundary
    /// position declares no provenance the summary could follow, when the closure reaches a reference that entered
    /// neither through its boundary nor through its capture scope, or when the closure consumes a caller allocation,
    /// which no state boundary can express. It also returns this error when `operation` does not permit one of the
    /// exact access modes the closure performs through `region_index`.
    pub fn region_summary<O: Operation>(
        &self,
        operation: &O,
        region_index: usize,
        region: RegionRef<'_, C::Constant, C::Operation>,
        inputs: &[Option<ReferenceDischargeAllocationId>],
    ) -> Result<ReferenceDischargeRegionSummary, ProgramError> {
        let captures =
            self.captures()
                .nested_scope(operation.region_capture_input_count(region_index), inputs, region.id())?;
        let mut summary = ReferenceDischargeRegionSummary::default();
        summarize_region_closure(region, inputs, &captures, &mut summary)?;
        validate_region_accesses(operation, region_index, &summary)?;
        Ok(summary)
    }

    // TODO(eaplatanios): Review this.
    /// Returns the allocation one input of a structured operation denotes, or [`None`] when the input is a value.
    ///
    /// A view is rejected rather than resolved to its allocation because a state boundary carries the allocation's
    /// complete stored value. The view must instead be created inside the region.
    ///
    /// A preserved allocation is resolved like any other. It crosses the boundary as the reference it already is, at
    /// its own declared input position, so it needs no state carry at all, which is exactly what
    /// [`state_widening`](Self::state_widening) leaves it out of.
    ///
    /// # Parameters
    ///
    ///   - `operand`: Carrier being classified.
    ///   - `operation`: Name of the operation being rewritten, used in the diagnostic.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when the input is a view or its allocation is no longer live.
    pub fn operand_allocation(
        &self,
        operand: &ReferenceDischargeValue<C, P>,
        operation: &str,
    ) -> Result<Option<ReferenceDischargeAllocationId>, ProgramError> {
        let ReferenceDischargeValue::Reference(reference) = operand else {
            return Ok(None);
        };
        let allocation = reference.allocation_id();
        let whole = self.allocation_entry(allocation)?.r#type().into_owned();
        if reference.is_view() {
            return Err(ProgramError::MalformedProgram(format!(
                "operation `{operation}` passes the view `{}` of {allocation} across a region boundary, which carries \
                 the complete stored value `{}`; create the view inside the region instead",
                reference.r#type(),
                whole,
            )));
        }
        Ok(Some(allocation))
    }

    // TODO(eaplatanios): Review this.
    /// Computes the symmetric widening facts one structured rule needs from a region summary: the discharged
    /// allocations threaded as state, every reached allocation gaining an added boundary position because no declared
    /// position already carries it, and the discharged subset whose successor states the rebuilt regions must publish.
    ///
    /// A closure needs an allocation threaded whenever its replay must be able to resolve that allocation, because it
    /// accesses it, returns it, or merely rematerializes a capture constant that denotes it. The threaded set is
    /// therefore the summary's reached allocations with the preserved allocations removed. A preserved reference
    /// survives in the destination as a reference value and crosses at its own declared input position, or at an
    /// added position as the destination reference it already denotes, exactly as the source passed it, so it needs no
    /// state carry, publishes no successor, and widens nothing. This is the one place that distinction is drawn, which
    /// is what keeps the structured rewrites stating one thing.
    ///
    /// # Parameters
    ///
    ///   - `summary`: Summary of the closures the rewritten operation attaches, in caller-allocation terms.
    ///   - `declared`: Allocations already crossing at declared boundary positions, which therefore need no added
    ///     position.
    ///
    /// # Errors
    ///
    /// Propagates the environment's error for the first reached allocation that is not live, which states whether the
    /// allocation was consumed, was never bound, or belongs to another environment.
    pub fn state_widening(
        &self,
        summary: &ReferenceDischargeRegionSummary,
        declared: &BTreeSet<ReferenceDischargeAllocationId>,
    ) -> Result<ReferenceDischargeStateWidening, ProgramError> {
        let mut threaded = BTreeSet::new();
        for allocation in summary.reached() {
            if self.is_allocation_discharged(allocation)? {
                threaded.insert(allocation);
            }
        }
        let entering = summary.reached().filter(|allocation| !declared.contains(allocation)).collect();
        let published = threaded.iter().copied().filter(|allocation| summary.is_mutated(*allocation)).collect();
        Ok(ReferenceDischargeStateWidening { threaded, entering, published })
    }

    // TODO(eaplatanios): Review this.
    /// Returns the destination value one input of a structured operation contributes to the rewritten application:
    /// the current immutable state of a discharged reference, the destination reference of a preserved one, or the
    /// input's own value.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when the input's allocation is not live.
    pub fn operand_value(&self, operand: &ReferenceDischargeValue<C, P>) -> Result<C::Value, ProgramError> {
        let reference = match operand {
            ReferenceDischargeValue::Reference(reference) => reference,
            ReferenceDischargeValue::Value(value) => return Ok(value.clone()),
        };
        match reference.binding() {
            ReferenceDischargeBinding::Discharged => self.discharged_state(reference.allocation_id()),
            ReferenceDischargeBinding::Preserved { reference: value } => {
                // A preserved handle keeps its destination value after consumption, so resolve its allocation before
                // returning that value to the rebuilt boundary.
                self.allocation_entry(reference.allocation_id())?;
                Ok(value.clone())
            }
        }
    }

    // TODO(eaplatanios): Review this.
    /// Returns the destination value one live allocation contributes to a rewritten boundary: its current immutable
    /// state when discharged, or its destination reference when preserved.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when `allocation` is not live in this environment.
    pub fn allocation_value(&self, allocation: ReferenceDischargeAllocationId) -> Result<C::Value, ProgramError> {
        self.operand_value(&self.allocation_reference(allocation)?)
    }

    // TODO(eaplatanios): Review this.
    /// Merges one boundary state output back into `allocation` with the summary's mutation fact, skipping allocations
    /// outside `threaded`: a carry that survives as a reference returned itself, so it has no successor state to merge.
    ///
    /// # Errors
    ///
    /// Propagates the underlying state replacement's liveness and type failures.
    pub fn merge_boundary_state(
        &self,
        summary: &ReferenceDischargeRegionSummary,
        threaded: &BTreeSet<ReferenceDischargeAllocationId>,
        allocation: ReferenceDischargeAllocationId,
        output: C::Value,
    ) -> Result<(), ProgramError>
    where
        C::Type: From<P::Referent>,
    {
        if threaded.contains(&allocation) {
            self.set_discharged_state(allocation, output, summary.is_mutated(allocation))?;
        }
        Ok(())
    }

    /// Lifts a stored [`Program`] constant into a value that can flow through reference discharge. A non-reference
    /// constant is lifted by the destination [`Context`] and wrapped as a [`ReferenceDischargeValue::Value`]. A
    /// reference-typed constant instead names an existing capture binding. This function resolves that binding through
    /// the active [`ReferenceDischargeCaptureScope`] and returns its [`ReferenceDischargeValue::Reference`]. It never
    /// creates another allocation for a captured reference. A reference-typed constant that the active capture scope
    /// does not resolve is rejected because no allocation in this context represents that reference. Allowing it to
    /// flow as an ordinary destination value would leave an untracked reference in the discharged program.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when a reference-typed constant does not resolve to an allocation or
    /// its declared reference type differs from the allocation's reference type. For a non-reference constant,
    /// propagates any error returned by the destination [`Context::lift`] function.
    pub fn lift(&self, constant: C::Constant) -> Result<ReferenceDischargeValue<C, P>, ProgramError>
    where
        C: Context,
        for<'t> &'t ReferenceType<P::Referent>: TryFrom<&'t C::Type>,
    {
        let constant_type = constant.r#type();
        if let Ok(r#type) = <&ReferenceType<P::Referent>>::try_from(constant_type.as_ref()) {
            let Some(allocation) = self.captures.resolve(&constant) else {
                return Err(ProgramError::MalformedProgram(format!(
                    "reference discharge cannot lift a constant of reference type `{type}`; a reference enters a \
                     program through an input, a capture binding, or an allocation",
                )));
            };

            // A capture constant names the complete stored value its position binds, so a narrower declared type would
            // silently widen to the allocation's own value where the constant is used.
            let bound = self.allocation_entry(allocation)?.r#type().into_owned();
            if r#type != &bound {
                return Err(ProgramError::MalformedProgram(format!(
                    "reference discharge resolved a capture constant of reference type `{type}` to {allocation}, \
                     which carries the reference type `{bound}`",
                )));
            }
            return self.allocation_reference(allocation);
        }
        Ok(ReferenceDischargeValue::Value(self.parent.lift(constant)?))
    }

    /// Discharges a source [`Region`](crate::Region) directly through this context and returns its outputs. The
    /// rewritten [`Instruction`](crate::Instruction)s are added to the destination [`Program`] this context owns,
    /// and their reference accesses observe and update this context's allocation environment.
    ///
    /// This is the shared region replay functionality used both by [`ReferenceDischargeDriver::inline_region`] and
    /// while rebuilding a region in an isolated context. Those paths differ in the context they use, but apply the
    /// same discharge rules to the source instructions.
    ///
    /// # Parameters
    ///
    ///   - `region`: Source region to discharge.
    ///   - `inputs`: Values supplied to the region's inputs, in region input order.
    ///
    /// # Errors
    ///
    /// Propagates errors raised while lifting the region's constants, validating preserved reference accesses, or
    /// discharging its instructions.
    fn inline_region(
        &self,
        region: RegionRef<'_, C::Constant, C::Operation>,
        inputs: Vec<ReferenceDischargeValue<C, P>>,
    ) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError>
    where
        C: Context<
                Type: From<<P as ReferenceDischargePolicy<C>>::Referent>
                          + From<ReferenceType<<P as ReferenceDischargePolicy<C>>::Referent>>,
                Operation: ReferenceDischargeableOperation<C, P>
                               + ReferenceDischargeableOperation<TracingContext<C::Constant, C::Operation>, P>,
            >,
        P: ReferenceDischargePolicy<C>
            + ReferenceDischargePolicy<
                TracingContext<C::Constant, C::Operation>,
                Referent = <P as ReferenceDischargePolicy<C>>::Referent,
            >,
        for<'t> &'t ReferenceType<<P as ReferenceDischargePolicy<C>>::Referent>: TryFrom<&'t C::Type>,
    {
        let mappings = RegionReplayMappings::new();
        let mut instruction_index = 0;
        region.interpret_with(
            inputs,
            |_, constant| self.lift(constant.clone()),
            |instruction, instruction_inputs| {
                let position = InstructionId::new(region.id(), instruction_index);
                instruction_index += 1;

                // Run the complete rewrite of one application (the preserved-access replay included) inside the source
                // instruction's recorded origin, so that every staged instruction records where it came from. Rules
                // stage their rewritten work through the destination parent, which is where the provenance state lives.
                self.parent().invoke_with_provenance_origin(instruction.provenance().clone(), || {
                    if instruction.regions().is_empty()
                        && let Some(outputs) =
                            self.replay_preserved_access(instruction.operation(), instruction_inputs)?
                    {
                        return Ok(outputs);
                    }

                    let regions = ReplayRegionDriver::new(region, instruction.regions(), &mappings)?;
                    let driver = RecursiveReferenceDischargeDriver::new(&regions, Some(position));
                    instruction.operation().discharge_references(self, &driver, instruction_inputs)
                })
            },
        )
    }

    /// Replays one [`Region`](crate::Region)-free [`Operation`] application that only accesses preserved references
    /// verbatim into the destination and returns its outputs, or returns [`None`] when the application is not such an
    /// access and must go through its own [`ReferenceDischargeableOperation`] implementation.
    ///
    /// A preserved reference survives partial reference discharge as a reference value of the destination universe, so
    /// the rewrite of an access to it is no rewrite at all: the operation is bound again over the destination reference
    /// value each handle denotes, and its results are the destination's own. This fast path runs before rule dispatch,
    /// which is why access rules only ever see discharged allocations. It applies only when the operation declares
    /// reference inputs but no reference outputs and every declared access is a preserved reference. Mixed preserved
    /// and discharged accesses are left to the operation's rule, which can reject or rewrite them with full knowledge
    /// of the operation's semantics, and an operation that produces a reference owns the bookkeeping for the resulting
    /// handle.
    ///
    /// Consumed allocations are validated before the replay and invalidated only after it succeeds, so a failed
    /// destination binding leaves every allocation live and a successful one makes every later alias of a consumed
    /// allocation report a use-after-consume.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when a consumed reference is a view rather than the reference for the
    /// complete stored value, when a reference operand's allocation is no longer live, when an operand outside the
    /// declared accesses denotes a discharged reference, or when a replayed output is reference-typed. It also
    /// propagates type inference errors and the destination's error from the replay itself.
    fn replay_preserved_access(
        &self,
        operation: &C::Operation,
        inputs: &[ReferenceDischargeValue<C, P>],
    ) -> Result<Option<Vec<ReferenceDischargeValue<C, P>>>, ProgramError>
    where
        C: Context<Type: From<ReferenceType<P::Referent>>>,
        for<'t> &'t ReferenceType<P::Referent>: TryFrom<&'t C::Type>,
    {
        let semantics = operation.reference_semantics();
        if semantics.inputs().is_empty() || !semantics.outputs().is_empty() {
            return Ok(None);
        }
        let mut consumed = Vec::new();
        for access in semantics.inputs() {
            let Some(ReferenceDischargeValue::Reference(reference)) = inputs.get(access.input_index()) else {
                return Ok(None);
            };
            if reference.preserved().is_none() {
                return Ok(None);
            }
            if access.mode().is_consuming() {
                consumed.push(reference);
            }
        }

        // Re-run inference over the carriers' current types before binding the unchanged operation. This preserves the
        // operation's own operand-relationship diagnostics instead of allowing a destination binding failure to obscure
        // them.
        let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        operation.infer_output_types(input_types.as_slice(), &[])?;

        // Validate every consumption before the replay. Consumption yields and invalidates the complete stored value,
        // so a view cannot name that transition even when it has the same reference type as the allocation. Rejecting
        // it after binding would leave a destination operation behind on an error path.
        for reference in &consumed {
            let allocation = reference.allocation_id();
            let complete_reference_type = self.allocation_entry(allocation)?.r#type().into_owned();
            if reference.is_view() {
                return Err(ProgramError::MalformedProgram(format!(
                    "reference discharge cannot consume {} through the view `{}`; consumption yields the complete \
                     stored value, whose reference type is `{}`",
                    allocation,
                    reference.r#type(),
                    complete_reference_type,
                )));
            }
        }

        // Each reference operand contributes the destination value its handle denotes, which is the only place a view's
        // exact value lives. Liveness is checked against the environment rather than assumed from the handle, because a
        // handle retains its destination value after its allocation is consumed.
        let values = inputs
            .iter()
            .map(|input| match input {
                ReferenceDischargeValue::Value(value) => Ok(value.clone()),
                ReferenceDischargeValue::Reference(reference) => match reference.preserved() {
                    Some(value) => {
                        self.allocation_entry(reference.allocation_id())?;
                        Ok(value.clone())
                    }
                    None => Err(ProgramError::MalformedProgram(format!(
                        "reference discharge cannot replay `{}` over discharged {}, which has no destination \
                         reference value",
                        operation.name(),
                        reference.allocation_id(),
                    ))),
                },
            })
            .collect::<Result<Vec<_>, _>>()?;
        let outputs = self.parent().bind(operation.clone(), Vec::new(), values.as_slice())?;

        // A reference-typed output is rejected rather than wrapped. The environment has no allocation for it, so it
        // could later cross a boundary or reach an access as an untracked value. An operation that produces a reference
        // owns that bookkeeping and must state it in its own rule.
        let outputs = outputs
            .into_iter()
            .enumerate()
            .map(|(output_index, output)| {
                let output_type = output.r#type();
                if let Ok(r#type) = <&ReferenceType<P::Referent>>::try_from(output_type.as_ref()) {
                    return Err(ProgramError::MalformedProgram(format!(
                        "reference discharge replayed `{}` over a preserved reference, but its output {} is \
                         the reference `{}`; an operation that produces a reference owns that allocation and needs \
                         a reference discharge rule of its own",
                        operation.name(),
                        output_index,
                        r#type,
                    )));
                }
                Ok(ReferenceDischargeValue::Value(output))
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Invalidate only after the replay succeeds.
        for reference in consumed {
            self.take_allocation_entry(reference.allocation_id())?;
        }
        Ok(Some(outputs))
    }
}

impl<C: Clone + Domain, P: ReferenceDischargePolicy<C>> Clone for ReferenceDischargeContext<C, P> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            parent: self.parent.clone(),
            environment: Rc::clone(&self.environment),
            captures: self.captures.clone(),
            targets: self.targets.clone(),
        }
    }
}

impl<C: Domain, P: ReferenceDischargePolicy<C>> Debug for ReferenceDischargeContext<C, P> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let live_allocation_count =
            self.environment.borrow().allocations.iter().filter(|allocation| allocation.is_some()).count();
        formatter
            .debug_struct("ReferenceDischargeContext")
            .field("live_allocation_count", &live_allocation_count)
            .finish_non_exhaustive()
    }
}

/// Process-local identity of a [`ReferenceDischargeEnvironment`], shared by every clone of the
/// [`ReferenceDischargeContext`] that owns it and distinct for every temporary environment created while rebuilding a
/// region. No caller names this identity directly. It makes [`ReferenceDischargeAllocationId`] addressable only in the
/// environment that minted it.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct ReferenceDischargeEnvironmentId(usize);

impl ReferenceDischargeEnvironmentId {
    /// Returns a fresh [`ReferenceDischargeEnvironmentId`], distinct from every [`ReferenceDischargeEnvironmentId`]
    /// handed out so far in this process.
    fn fresh() -> Self {
        static NEXT_ENVIRONMENT_ID: AtomicUsize = AtomicUsize::new(0);
        Self(NEXT_ENVIRONMENT_ID.fetch_add(1, Ordering::Relaxed))
    }
}

/// Live allocation environment of a reference discharge transform, shared by every clone of its
/// [`ReferenceDischargeContext`].
struct ReferenceDischargeEnvironment<T: Type, V> {
    /// Unique [`ReferenceDischargeEnvironmentId`] of this [`ReferenceDischargeEnvironment`] that every
    /// [`ReferenceDischargeAllocationId`] minted from this environment records.
    id: ReferenceDischargeEnvironmentId,

    /// State of every allocation minted so far, indexed by [`ReferenceDischargeAllocationId`]. A consumed allocation
    /// keeps its position and becomes [`None`], so that a use-after-consume is reported against the exact allocation
    /// rather than as an unknown ID.
    allocations: Vec<Option<ReferenceDischargeAllocationEntry<T, V>>>,
}

impl<T: Type, V> ReferenceDischargeEnvironment<T, V> {
    /// Returns the live [`ReferenceDischargeAllocationEntry`] that `allocation` names, or [`None`] if that allocation
    /// has been consumed.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when `allocation` belongs to another environment or names a position
    /// this environment never minted.
    fn entry(
        &self,
        allocation: ReferenceDischargeAllocationId,
    ) -> Result<Option<&ReferenceDischargeAllocationEntry<T, V>>, ProgramError> {
        if allocation.environment != self.id {
            return Err(ProgramError::MalformedProgram(format!(
                "reference discharge accessed {}, which belongs to an environment \
                 other than the active `{}` environment",
                allocation, self.id.0,
            )));
        }
        Ok(self
            .allocations
            .get(allocation.index)
            .ok_or_else(|| {
                ProgramError::MalformedProgram(format!("reference discharge accessed never-bound {allocation}"))
            })?
            .as_ref())
    }
}

/// Complete record of a live reference allocation during a reference discharge transform, which contains the reference
/// type of its complete stored value and what discharge turned that allocation into. The reference type is recorded
/// because an allocation's identity outlives every handle that denotes it. A structured rule threading an inherited
/// allocation through a rebuilt region boundary holds only that allocation's handle, never a handle it could read a
/// type off, so the environment is where the complete-value type has to live.
// TODO(eaplatanios): Make private once the `discharge` module review and cleanup is completed.
pub(crate) struct ReferenceDischargeAllocationEntry<T: Type, V> {
    /// [`ReferenceType`] of the allocation's complete stored value. When the allocation is discharged, its referent
    /// type is the type of the immutable state threaded through the rewritten program.
    r#type: ReferenceType<T>,

    /// Current representation of the allocation in the rewritten program (either immutable state for a discharged
    /// allocation or a destination reference value for a preserved allocation).
    state: ReferenceDischargeAllocationState<V>,
}

impl<T: Type, V> Typed for ReferenceDischargeAllocationEntry<T, V> {
    type Type = ReferenceType<T>;

    #[inline]
    fn r#type(&self) -> Cow<'_, Self::Type> {
        Cow::Borrowed(&self.r#type)
    }
}

/// [`ReferenceDischargeEnvironment`] entry describing what a reference allocation became during reference discharge.
#[derive(Debug)]
enum ReferenceDischargeAllocationState<V> {
    /// Allocation selected for discharge, which threads through the destination [`Program`] as an immutable state.
    Discharged {
        /// Current immutable state of the complete stored value.
        current: V,

        /// Whether any ordered write or accumulation operation has been applied to this reference allocation. Read-only
        /// allocations are pruned from hidden outputs and from structured operation widening, and this is the fact that
        /// pruning consults.
        mutated: bool,
    },

    /// Allocation not selected for discharge, which survives in the destination [`Program`] as a reference value. This
    /// is the allocation's own destination reference value and is what boundary threading uses; a view created from it
    /// carries its own exact destination value instead.
    Preserved {
        /// Destination reference-typed value denoting the allocation.
        reference: V,
    },
}

/// Reference allocations the capture prefix of a reference discharge scope binds. A capture-lifted [`Program`] names
/// its caller's references through constants rather than through its own boundary: the entry boundary carries the
/// lifted capture prefix, and an attached [`Region`](crate::Region) inside that program names the very same references
/// through capture constants. Resolving one is therefore a property of the scope a region discharges under, not of any
/// rule, and so the scope rides on [`ReferenceDischargeContext`] beside the allocation environment and is recomputed at
/// every region boundary (inherited by default, and replaced by a fresh prefix wherever an operation declares one
/// through [`Operation::region_capture_input_count`]).
///
/// Recognizing a capture is a _constant-family_ question, and the interpreter deliberately serves families that are
/// not capture-bearing at all, so the resolver is a function pointer supplied by the entry point that knows the family
/// rather than a [`CaptureConstant`] bound on the whole architecture. The [`Default`] scope recognizes nothing and
/// binds nothing, which is exactly the behavior of a program that has no captures.
pub struct ReferenceDischargeCaptureScope<Constant> {
    /// Function that returns the capture index/position that a constant names, or [`None`] when it is a non-reference
    /// constant of its family.
    capture_index_of: fn(&Constant) -> Option<usize>,

    /// List that contains the [`ReferenceDischargeAllocationId`] of the allocation that each capture position binds,
    /// or [`None`] when that position carries a value rather than a reference. A capture position past the end of this
    /// list binds nothing.
    allocations: Rc<[Option<ReferenceDischargeAllocationId>]>,
}

impl<Constant> ReferenceDischargeCaptureScope<Constant> {
    /// Creates a new [`ReferenceDischargeCaptureScope`].
    #[inline]
    pub fn new(
        capture_index_of: fn(&Constant) -> Option<usize>,
        allocations: Vec<Option<ReferenceDischargeAllocationId>>,
    ) -> Self {
        Self { capture_index_of, allocations: allocations.into() }
    }

    /// Returns a clone of this [`ReferenceDischargeCaptureScope`] with the same capture index resolver but over a
    /// different set of bound allocations. This is how a nested [`Region`](crate::Region)'s scope and an isolated
    /// rebuild's remapped [`ReferenceDischargeCaptureScope`] are built without restating the constant family's
    /// recognition rule.
    #[inline]
    pub fn with_allocations(&self, allocations: Vec<Option<ReferenceDischargeAllocationId>>) -> Self {
        Self { capture_index_of: self.capture_index_of, allocations: allocations.into() }
    }

    /// Returns the [`ReferenceDischargeCaptureScope`] to use when discharging a nested [`Region`](crate::Region). When
    /// the [`Operation`] that owns the region declares a capture prefix, the corresponding leading region inputs become
    /// the nested scope's capture bindings. When it does not declare a capture prefix, the region inherits this scope
    /// unchanged.
    ///
    /// # Parameters
    ///
    ///   - `capture_input_count`: Number of leading region inputs that provide capture bindings, as reported by
    ///     [`Operation::region_capture_input_count`], or [`None`] if the region inherits this scope.
    ///   - `inputs`: Reference allocation bound by each region input, in region input order. A [`None`] entry
    ///     identifies an input that does not bind a reference allocation.
    ///   - `region`: Region identity to include in an error diagnostic.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when the declared capture prefix is longer than the region's input
    /// boundary.
    pub fn nested_scope(
        &self,
        capture_input_count: Option<usize>,
        inputs: &[Option<ReferenceDischargeAllocationId>],
        region: RegionId,
    ) -> Result<Self, ProgramError> {
        let Some(count) = capture_input_count else {
            return Ok(self.clone());
        };
        if count > inputs.len() {
            return Err(ProgramError::MalformedProgram(format!(
                "reference discharge cannot establish a capture prefix of {} for region `{}`, which declares {} inputs",
                count,
                region,
                inputs.len(),
            )));
        }
        Ok(self.with_allocations(inputs[..count].to_vec()))
    }

    /// Returns the [`ReferenceDischargeAllocationId`] of the allocation each capture position binds, in capture order.
    #[inline]
    pub fn allocations(&self) -> &[Option<ReferenceDischargeAllocationId>] {
        self.allocations.as_ref()
    }

    /// Returns the [`ReferenceDischargeAllocationId`] of the allocation the provided constant names, or [`None`] when
    /// the constant names no capture position or that position binds no allocation. A constant this scope cannot
    /// resolve is a non-reference constant of its family, and a reference-typed one that no scope resolves is rejected
    /// where it is lifted.
    #[inline]
    pub fn resolve(&self, constant: &Constant) -> Option<ReferenceDischargeAllocationId> {
        (self.capture_index_of)(constant).and_then(|index| self.allocations.get(index).copied().flatten())
    }
}

impl<Constant> Default for ReferenceDischargeCaptureScope<Constant> {
    #[inline]
    fn default() -> Self {
        Self { capture_index_of: |_| None, allocations: Rc::from([]) }
    }
}

impl<Constant> Clone for ReferenceDischargeCaptureScope<Constant> {
    #[inline]
    fn clone(&self) -> Self {
        Self { capture_index_of: self.capture_index_of, allocations: Rc::clone(&self.allocations) }
    }
}

impl<Constant> Debug for ReferenceDischargeCaptureScope<Constant> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ReferenceDischargeCaptureScope")
            .field("allocations", &self.allocations)
            .finish_non_exhaustive()
    }
}

impl<V: Value, O: Operation<Type = V::Type>> Program<V, O, Vec<V>, Vec<V>> {
    /// Rewrites every [`Reference`](crate::Reference) in this [`Program`] as explicit immutable state and returns the
    /// resulting reference-free program together with bindings for its external references.
    ///
    /// A reference-typed input keeps its position but becomes a value input carrying the reference's initial state.
    /// Local reference allocations disappear. The source program's public outputs remain first and in the same order;
    /// the final state of each mutated external reference is appended as a hidden output in entry-boundary order.
    /// A read-only external reference adds no hidden output.
    ///
    /// Each operation defines how its reference effects are rewritten through [`ReferenceDischargeableOperation`].
    /// A structured operation must also thread through the reference state used by its attached
    /// [`Region`](crate::Region)s; a structured operation whose complete region closure is reference-free is replayed
    /// unchanged. The returned [`ReferenceDischargeResult`] proves that no reference type or reference operation
    /// remains anywhere in the rewritten region closure.
    ///
    /// Use [`discharge_references_in_capture_lifted_program`](Self::discharge_references_in_capture_lifted_program)
    /// instead when the program was produced by lifting a [`ClosedProgram`]'s captures and attached regions may refer
    /// to those captures through capture constants.
    ///
    /// # Parameters
    ///
    ///   - `capture_count`: Number of leading flat inputs that originated in the source program's capture table, used
    ///     to split the entry boundary into [`ReferenceSource::Capture`] and [`ReferenceSource::Input`] positions.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when `capture_count` exceeds the input count, when an external
    /// reference is consumed, when a reference reaches a region without a boundary or capture binding, when a
    /// structured operation has no rule for its reference-using regions, or when the rewritten program is not fully
    /// reference-free. Errors reported by individual operation rules, including use after consumption and access to
    /// an unbound reference, propagate unchanged.
    #[inline]
    pub fn discharge_references<P: ReferenceDischargePolicy<TracingContext<V, O>>>(
        self,
        capture_count: usize,
    ) -> Result<ReferenceDischargeResult<V, O>, ProgramError>
    where
        V::Type: From<P::Referent> + From<ReferenceType<P::Referent>> + ReferenceDischargeableType<Policy = P>,
        O: ReferenceDischargeableOperation<TracingContext<V, O>, P>,
        for<'t> &'t ReferenceType<P::Referent>: TryFrom<&'t V::Type>,
    {
        ReferenceDischargeResult::try_from(self.discharge_references_helper::<P>(
            capture_count,
            |_| None,
            ReferenceDischargeTargets::everything(),
        )?)
    }

    /// Rewrites every [`Reference`](crate::Reference) in a capture-lifted [`Program`] as explicit immutable state and
    /// returns a proven reference-free result.
    ///
    /// A capture-lifted program has the captures of a [`ClosedProgram`] represented by a leading input prefix. Attached
    /// [`Region`](crate::Region)s may still name those captures through capture constants. This function resolves each
    /// reference-typed capture constant to the external reference bound at the corresponding prefix position.
    ///
    /// Apart from that capture resolution, this function has the same boundary rewrite, hidden final-state outputs,
    /// and reference-freedom guarantee as [`discharge_references`](Self::discharge_references). The two functions
    /// produce the same result when no attached region uses a reference-typed capture constant.
    ///
    /// # Parameters
    ///
    ///   - `capture_count`: Number of inputs in the lifted capture prefix. It determines both the capture scope and the
    ///     split between [`ReferenceSource::Capture`] and [`ReferenceSource::Input`] positions.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`discharge_references`](Self::discharge_references). It also returns
    /// [`ProgramError::MalformedProgram`] when a capture constant's reference type disagrees with the external
    /// reference bound at its capture position.
    #[inline]
    pub fn discharge_references_in_capture_lifted_program<P: ReferenceDischargePolicy<TracingContext<V, O>>>(
        self,
        capture_count: usize,
    ) -> Result<ReferenceDischargeResult<V, O>, ProgramError>
    where
        V: CaptureConstant,
        V::Type: From<P::Referent> + From<ReferenceType<P::Referent>> + ReferenceDischargeableType<Policy = P>,
        O: ReferenceDischargeableOperation<TracingContext<V, O>, P>,
        for<'t> &'t ReferenceType<P::Referent>: TryFrom<&'t V::Type>,
    {
        ReferenceDischargeResult::try_from(self.discharge_references_helper::<P>(
            capture_count,
            CaptureConstant::capture_index,
            ReferenceDischargeTargets::everything(),
        )?)
    }

    /// Rewrites the selected/targeted [`Reference`](crate::Reference)s as explicit immutable state while preserving
    /// every unselected reference.
    ///
    /// The selected references follow the same rewrite as [`discharge_references`](Self::discharge_references). An
    /// unselected reference keeps its reference-typed boundary position or allocation operation, and its accesses and
    /// views are replayed unchanged. It contributes no [`ExternalReferenceBinding`] or hidden final-state
    /// output because it never becomes explicit state.
    ///
    /// Preserved references can cross structured-[`Region`](crate::Region) boundaries beside discharged state. A
    /// declared reference position remains a reference position; when an attached region reaches a preserved reference
    /// only through an inherited capture, the rewrite adds a reference-typed position to keep the rebuilt region
    /// self-contained.
    ///
    /// The returned [`PartialReferenceDischargeResult`] may therefore still contain reference types and operations.
    /// If the selected targets were expected to cover every reference, convert it through
    /// [`ReferenceDischargeResult::try_from`] to validate and obtain the full-discharge guarantee.
    ///
    /// A capture-lifted program instead uses [`Self::partially_discharge_references_in_capture_lifted_program`], which
    /// performs the same target selection under a populated capture scope.
    ///
    /// # Parameters
    ///
    ///   - `capture_count`: Number of leading flat inputs that originated in the source program's capture table,
    ///     used to split the entry boundary into [`ReferenceSource::Capture`] and [`ReferenceSource::Input`] positions.
    ///   - `targets`: Reference targets to discharge, enumerated from this same program through
    ///     [`reference_discharge_targets`](Self::reference_discharge_targets). Every other allocation is preserved.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when `targets` does not belong to this program or otherwise violates
    /// the target-selection contract. It also returns the applicable errors documented by
    /// [`discharge_references`](Self::discharge_references), except that consuming a preserved external reference is
    /// allowed because its consuming operation remains in the rewritten program. Consuming a discharged external
    /// reference remains invalid because its state is still owned by the caller.
    #[inline]
    pub fn partially_discharge_references<P: ReferenceDischargePolicy<TracingContext<V, O>>>(
        self,
        capture_count: usize,
        targets: &[ReferenceDischargeTarget],
    ) -> Result<PartialReferenceDischargeResult<V, O>, ProgramError>
    where
        V::Type: From<P::Referent> + From<ReferenceType<P::Referent>> + ReferenceDischargeableType<Policy = P>,
        O: ReferenceDischargeableOperation<TracingContext<V, O>, P>,
        for<'t> &'t ReferenceType<P::Referent>: TryFrom<&'t V::Type>,
    {
        let targets = ReferenceDischargeTargets::from_targets(&self, capture_count, targets)?;
        self.discharge_references_helper::<P>(capture_count, |_| None, targets)
    }

    /// Rewrites the selected/targeted [`Reference`](crate::Reference)s of a capture-lifted [`Program`] as explicit
    /// immutable state and preserves every unselected reference.
    ///
    /// This function combines the capture-constant resolution of
    /// [`discharge_references_in_capture_lifted_program`](Self::discharge_references_in_capture_lifted_program)
    /// with the selection and partial-result behavior of
    /// [`partially_discharge_references`](Self::partially_discharge_references). The lifted capture prefix
    /// establishes the capture scope; only references named by `targets` become immutable state.
    ///
    /// # Parameters
    ///
    ///   - `capture_count`: Number of inputs in the lifted capture prefix.
    ///   - `targets`: Reference targets to discharge, enumerated from this same capture-lifted program through
    ///     [`reference_discharge_targets`](Self::reference_discharge_targets). Every other allocation is preserved.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`partially_discharge_references`](Self::partially_discharge_references). It also
    /// returns [`ProgramError::MalformedProgram`] when a capture constant's reference type disagrees with the external
    /// reference bound at its capture position.
    #[inline]
    pub fn partially_discharge_references_in_capture_lifted_program<P: ReferenceDischargePolicy<TracingContext<V, O>>>(
        self,
        capture_count: usize,
        targets: &[ReferenceDischargeTarget],
    ) -> Result<PartialReferenceDischargeResult<V, O>, ProgramError>
    where
        V: CaptureConstant,
        V::Type: From<P::Referent> + From<ReferenceType<P::Referent>> + ReferenceDischargeableType<Policy = P>,
        O: ReferenceDischargeableOperation<TracingContext<V, O>, P>,
        for<'t> &'t ReferenceType<P::Referent>: TryFrom<&'t V::Type>,
    {
        let targets = ReferenceDischargeTargets::from_targets(&self, capture_count, targets)?;
        self.discharge_references_helper::<P>(capture_count, CaptureConstant::capture_index, targets)
    }

    /// Performs the shared partial-discharge rewrite for one validated target selection. `capture_index_of` resolves
    /// reference-typed capture constants when the input is capture-lifted; programs without lifted captures provide a
    /// resolver that matches no constant. The function always returns a [`PartialReferenceDischargeResult`].
    /// Full-discharge entry points select every reference and then validate the result through
    /// [`ReferenceDischargeResult::try_from`].
    ///
    /// # Parameters
    ///
    ///   - `capture_count`: Number of leading inputs that originated in the source program's capture table.
    ///   - `capture_index_of`: Function returning the capture position named by a stored constant.
    ///   - `targets`: Reference targets to discharge; every allocation they omit is preserved.
    ///
    /// # Errors
    ///
    /// Returns any validation or operation-rule error produced while rewriting the program.
    fn discharge_references_helper<P: ReferenceDischargePolicy<TracingContext<V, O>>>(
        self,
        capture_count: usize,
        capture_index_of: fn(&V) -> Option<usize>,
        targets: ReferenceDischargeTargets,
    ) -> Result<PartialReferenceDischargeResult<V, O>, ProgramError>
    where
        O: ReferenceDischargeableOperation<TracingContext<V, O>, P>,
        V::Type: From<P::Referent> + From<ReferenceType<P::Referent>>,
        for<'t> &'t ReferenceType<P::Referent>: TryFrom<&'t V::Type>,
    {
        let input_types = self.input_types();
        let input_count = input_types.len();
        if capture_count > input_count {
            return Err(ProgramError::MalformedProgram(format!(
                "reference discharge requests {capture_count} captures but the program has {input_count} inputs",
            )));
        }
        let output_count = self.output_count();

        // A program that touches no reference anywhere is already its own discharge, so it is returned untouched rather
        // than replayed into a fresh trace. This is not only cheaper on the two transform adapters that discharge
        // unconditionally: re-tracing would also renumber its atoms, drop its dead constants, and abandon the region
        // transform cache its regions carry, all for a rewrite that has nothing to rewrite.
        let entry = self.entry_region_ref();
        if !entry.contains_references_in_closure() {
            return PartialReferenceDischargeResult::new(self, capture_count, output_count, Vec::new());
        }

        // The block scopes the destination context, the discharge context, and every carrier, because recovering the
        // traced program below requires unique ownership of the shared builder and therefore that every other handle
        // to it has been released.
        let (builder, output_ids, external_reference_bindings) = {
            let destination = TracingContext::<V, O>::new();
            let builder = destination.builder().clone();
            let context = ReferenceDischargeContext::new_with_targets(destination.clone(), targets);
            let mut inputs = Vec::with_capacity(input_count);
            let mut discharged_allocations = Vec::new();
            let mut capture_allocations = vec![None; capture_count];
            for (input_index, input_type) in input_types.into_iter().enumerate() {
                let Ok(reference_type) = <&ReferenceType<P::Referent>>::try_from(&input_type) else {
                    inputs.push(ReferenceDischargeValue::Value(destination.input(input_type)));
                    continue;
                };
                let reference_type = reference_type.clone();
                let source = ReferenceSource::from_flat_input_index(input_index, capture_count);
                let selected = context.selects_external(source);
                let carrier = if selected {
                    let state = destination.input(V::Type::from(reference_type.referent().clone()));
                    context.bind_discharged(reference_type, state)?
                } else {
                    // An unselected external allocation keeps its reference-typed boundary position exactly as the
                    // source declared it, so the caller still supplies the reference, and every access to it replays
                    // verbatim.
                    context.bind_preserved(reference_type, destination.input(input_type))?
                };
                let allocation = carrier.try_as_reference("an entry-boundary reference allocation")?.allocation_id();
                if selected {
                    discharged_allocations.push((source, allocation));
                }
                if input_index < capture_count {
                    capture_allocations[input_index] = Some(allocation);
                }
                inputs.push(carrier);
            }

            // The capture scope can only be established once the prefix has minted its allocations, and it is what lets
            // a nested region resolve the caller references it names through capture constants rather than through its
            // own boundary.
            let context =
                context.with_captures(ReferenceDischargeCaptureScope::new(capture_index_of, capture_allocations));

            let regions = [self];
            let driver = RecursiveReferenceDischargeDriver::new(&regions, None);
            let outputs = driver.inline_region(&context, 0, inputs)?;
            let mut output_ids = outputs
                .iter()
                .enumerate()
                .map(|(output_index, output)| match output {
                    ReferenceDischargeValue::Value(value) => value.atom_id(),
                    ReferenceDischargeValue::Reference(reference) => {
                        // A preserved reference survives in the rewritten program, so returning one returns its
                        // destination reference value. A discharged reference has no such value, because it became
                        // state. Returning an allocation is a use of it like any other, so its liveness is resolved
                        // against the environment rather than taken from the handle, which is what reports an
                        // allocation the program already consumed.
                        context.allocation_entry(reference.allocation_id())?;
                        reference
                            .preserved()
                            .ok_or_else(|| {
                                ProgramError::MalformedProgram(format!(
                                    "reference discharge expected a value for output {output_index} but \
                                     received {reference}",
                                ))
                            })?
                            .atom_id()
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;

            // A mutated external allocation publishes its final state as a hidden output. A read-only one publishes
            // nothing, which is what keeps a read-only program's boundary identical to its source boundary. A preserved
            // external allocation binds nothing at all (i.e., because it never became state, so there is no state for a
            // caller to supply or to write back).
            let mut external_reference_bindings = Vec::with_capacity(discharged_allocations.len());
            for (source, allocation) in discharged_allocations {
                // External state remains caller-owned, so consuming its allocation during replay invalidates the
                // transform even when no later source operation tries to use it.
                if context.allocation_entry(allocation).is_err() {
                    return Err(ProgramError::MalformedProgram(format!(
                        "reference discharge consumed external {source}, whose state must remain owned by the caller",
                    )));
                }
                let output_index = if context.is_mutated(allocation)? {
                    output_ids.push(context.discharged_state(allocation)?.atom_id()?);
                    Some(output_ids.len() - 1)
                } else {
                    None
                };
                external_reference_bindings.push(ExternalReferenceBinding::new(source, output_index));
            }
            (builder, output_ids, external_reference_bindings)
        };

        let complete_output_count = output_ids.len();
        let builder = Rc::try_unwrap(builder).map_err(|_| ProgramError::EscapedProgramBuilder)?.into_inner();
        let program =
            builder.build(output_ids, vec![Placeholder; input_count], vec![Placeholder; complete_output_count])?;
        PartialReferenceDischargeResult::new(program, capture_count, output_count, external_reference_bindings)
    }
}

impl<
    Capture: Value,
    V: CaptureConstant<Type = Capture::Type>,
    O: Operation<Type = Capture::Type>,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
> ClosedProgram<Capture, V, O, Input, Output>
{
    /// Lifts this [`ClosedProgram`]'s captures into leading inputs and rewrites every [`Reference`](crate::Reference)
    /// as explicit immutable state.
    ///
    /// The returned [`ReferenceDischargeResult`] remains reference-free and records which leading inputs originated
    /// as captures rather than value inputs. The concrete capture values remain owned by this closed program; their
    /// mutable contents are not embedded in the rewritten program.
    ///
    /// # Errors
    ///
    /// Returns errors produced while lifting the captures or performing capture-aware reference discharge.
    #[inline]
    pub fn discharge_references<P: ReferenceDischargePolicy<TracingContext<V, O>>>(
        &self,
    ) -> Result<ReferenceDischargeResult<V, O>, ProgramError>
    where
        Capture::Type: From<P::Referent> + From<ReferenceType<P::Referent>> + ReferenceDischargeableType<Policy = P>,
        O: ReferenceDischargeableOperation<TracingContext<V, O>, P>,
        for<'t> &'t ReferenceType<P::Referent>: TryFrom<&'t Capture::Type>,
    {
        let capture_count = self.captures().len();
        let program = self.to_program_with_lifted_captures()?;
        program.discharge_references_in_capture_lifted_program::<P>(capture_count)
    }
}

/// Discharges one [`Operation`] application that touches no reference by replaying it verbatim over its rewritten
/// operands. This is the rule body that a [`ReferenceDischargeableOperation`] implementation delegates to for every
/// operation whose operands, outputs, and attached [`Region`](crate::Region)s are all reference-free, and it is the
/// discharge counterpart of ordinary interpretation. The destination decides what replaying means (e.g., an eager
/// destination executes the operation and a staging destination records it).
///
/// The precondition is reference freedom, not purity in the [`Effects`](crate::Effects) sense. An operation with
/// ordered or other effects replays here unchanged, because the replay reproduces those effects in the destination
/// exactly as the source performed them. Attached regions are copied into the destination as they stand, which is the
/// complete rewrite for regions that hold no state to thread. An application is rejected as soon as a reference appears
/// among its operands or anywhere inside an attached region's closure, because how a reference boundary widens is
/// knowledge that belongs to the operation, which must then implement its own rule. For the common case of a
/// region-carrying operation that forwards its operands to its regions positionally, that rule is
/// [`discharge_positional_region_operation`].
///
/// # Parameters
///
///   - `operation`: Operation application being replayed.
///   - `context`: Active discharge context whose [`ReferenceDischargeContext::parent`] binds the replay.
///   - `driver`: Application-scoped [`ReferenceDischargeDriver`] supplying any attached regions.
///   - `inputs`: Carriers supplied as this application's operands, in operation-defined order.
///
/// # Errors
///
/// Returns [`ProgramError::UnsupportedOperation`] when a region-carrying application touches a reference, returns
/// [`ProgramError::MalformedProgram`] when a region-free application receives a reference operand, and propagates the
/// destination's error from the replay itself.
pub fn discharge_reference_free_operation<
    O: Clone + Operation<Type = C::Type>,
    C: Context<Operation: From<O>>,
    P: ReferenceDischargePolicy<C>,
    D: ReferenceDischargeDriver<C, P>,
>(
    operation: &O,
    context: &ReferenceDischargeContext<C, P>,
    driver: &D,
    inputs: &[ReferenceDischargeValue<C, P>],
) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError> {
    if driver.region_count() != 0 {
        let touches_references = inputs.iter().any(|input| matches!(input, ReferenceDischargeValue::Reference(_)))
            || driver.regions().any(RegionRef::contains_references_in_closure);
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
            input.try_as_value(&format!("a value operand {} of `{}`", input_index, operation.name())).cloned()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let outputs = context.parent().bind(operation.clone(), regions, values.as_slice())?;
    Ok(outputs.into_iter().map(ReferenceDischargeValue::Value).collect())
}

/// Discharges one region-carrying [`Operation`] application whose [`Region`](crate::Region)s positionally forward its
/// operands, so that the references its region closures touch become explicit immutable state. This is the rule body
/// that a [`ReferenceDischargeableOperation`] implementation delegates to for structured operations whose attached
/// regions all mirror the operand list after a constant number of leading operands and whose outputs are each
/// region's own outputs, such as a condition, whose branches follow its predicate, and a call, whose callee follows
/// nothing. Loop-shaped operations such as `while` and `scan` carry their state symmetrically through a fixed point
/// and need a rule of their own built on [`ReferenceDischargeRegionBoundary::symmetric`]. When nothing the
/// application touches is a reference, use [`discharge_reference_free_operation`] instead.
///
/// All attached regions receive one shared boundary, which is widened as follows:
///
///   - every allocation that some region closure reaches enters as an operand appended after the declared ones, unless
///     a declared reference operand already carries it, in which case it enters at its own position;
///   - only the allocations that some closure mutates are published back, as outputs appended after the declared ones,
///     unless a declared reference output already publishes them. An allocation the closures merely read needs no
///     successor state, which keeps a read-only branch's boundary identical to its source boundary; and
///   - every region receives the identical state positions, so a rebuilt condition's branches keep agreeing with each
///     other. Only the capture prefix is read per region, because it is the operation's own per-region declaration.
///
/// # Parameters
///
///   - `operation`: Operation application being rewritten. It is bound unchanged over the widened operand list,
///     because threading state past a positional boundary changes only the boundary.
///   - `context`: Active [`ReferenceDischargeContext`] owning the allocation environment.
///   - `driver`: Application-scoped [`ReferenceDischargeDriver`] supplying the attached regions.
///   - `inputs`: Carrier [`ReferenceDischargeValue`]s supplied as this application's operands,
///     in operation-defined order.
///   - `leading_input_count`: Number of leading inputs/operands that parameterize the operation itself rather than
///     being forwarded to its regions, which is one for a condition's predicate and zero for a call.
///
/// # Errors
///
/// Returns [`ProgramError::MalformedProgram`] when the application has fewer operands than `leading_operand_count`,
/// when a leading operand is a reference, when an attached region's boundary does not forward the remaining operands
/// positionally, when a reference operand is a view rather than the reference for the complete stored value, when a
/// region closure reaches an allocation that never entered the boundary or consumes one, when a region returns an
/// allocation its caller never threaded, when the attached regions disagree on which outputs denote references, or
/// when a region mutates an allocation the widening did not predict.
pub fn discharge_positional_region_operation<C, P, O, D>(
    operation: &O,
    context: &ReferenceDischargeContext<C, P>,
    driver: &D,
    inputs: &[ReferenceDischargeValue<C, P>],
    leading_input_count: usize,
) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError>
where
    C: Context<Operation: From<O>>,
    C::Type: From<P::Referent>,
    P: ReferenceDischargePolicy<C>,
    O: Clone + Operation<Type = C::Type>,
    D: ReferenceDischargeDriver<C, P>,
{
    let name = operation.name();
    if inputs.len() < leading_input_count {
        return Err(ProgramError::MalformedProgram(format!(
            "operation `{}` forwards its inputs after {} leading inputs but the application has {} inputs",
            name,
            leading_input_count,
            inputs.len(),
        )));
    }
    let (leading, forwarded) = inputs.split_at(leading_input_count);
    for (index, input) in leading.iter().enumerate() {
        input.try_as_value(&format!("a value leading input {index} of `{name}`"))?;
    }
    let forwarded_allocations = forwarded
        .iter()
        .map(|operand| context.operand_allocation(operand, name))
        .collect::<Result<Vec<_>, _>>()?;

    // Every region forwards the same inputs, so one summary of all of them decides one shared boundary. It is seeded
    // from the first region rather than from an empty summary, because merging keeps the receiver's declared output
    // allocations and an empty summary declares none.
    let region_count = driver.region_count();
    let mut summary: Option<ReferenceDischargeRegionSummary> = None;
    for index in 0..region_count {
        let region = driver.region(index)?;
        check_count!("input", region.input_ids(), forwarded.len(), ProgramError);
        let region_summary = context.region_summary(operation, index, region, forwarded_allocations.as_slice())?;
        match &mut summary {
            Some(summary) => summary.merge(&region_summary),
            None => summary = Some(region_summary),
        }
    }
    let summary = summary.ok_or_else(|| {
        ProgramError::MalformedProgram(format!("operation `{name}` forwards its inputs but attaches no regions"))
    })?;

    // Every reached allocation absent from the forwarded inputs enters through an added input: a discharged capture
    // crosses as state and a preserved capture crosses as its destination reference, so the rebuilt region can bind
    // its inherited capture scope. A region that returns a discharged reference already publishes its final state at
    // that output position, so only a mutated state allocation absent from the declared outputs leaves through an
    // added output. The complete published set is what the rebuilt regions are held to.
    let forwarded_allocation_set = forwarded_allocations.iter().copied().flatten().collect::<BTreeSet<_>>();
    let widening = context.state_widening(&summary, &forwarded_allocation_set)?;
    let entering = widening.entering();
    let represented = summary.output_allocations().iter().copied().flatten().collect::<BTreeSet<_>>();
    let leaving = widening
        .published()
        .iter()
        .copied()
        .filter(|allocation| !represented.contains(allocation))
        .collect::<Vec<_>>();

    let source_output_count = driver.region(0)?.output_ids().len();
    let mut regions = Vec::with_capacity(region_count);
    for index in 0..region_count {
        // Every region receives the same state positions, so a rebuilt condition's branches keep agreeing with each
        // other. Only the capture prefix is read per region, because it is the operation's own per-region declaration.
        let boundary = ReferenceDischargeRegionBoundary::new(
            operation,
            index,
            forwarded_allocations.clone(),
            ReferenceDischargeRegionStateInsertion::new(entering.to_vec(), forwarded.len()),
            ReferenceDischargeRegionStateInsertion::new(leaving.clone(), source_output_count),
        );
        let result = driver.rebuild_region(context, index, &boundary)?;
        result.validate_predicted_mutations(widening.published(), name)?;
        result.validate_predicted_output_allocations(summary.output_allocations(), name)?;
        regions.push(result.into_program());
    }
    let output_allocations = summary.output_allocations();

    let mut operands = Vec::with_capacity(inputs.len() + entering.len());
    for input in inputs {
        operands.push(context.operand_value(input)?);
    }
    for allocation in entering {
        operands.push(context.allocation_value(*allocation)?);
    }
    let outputs = context.parent().bind(operation.clone(), regions, operands.as_slice())?;
    check_count!("output", outputs, source_output_count + leaving.len(), ProgramError);

    // A declared output that denotes a reference is reported as the handle the caller already holds rather than as a
    // value. For a discharged reference that output carried its final state, which is merged back. For a preserved
    // reference it carried the reference itself, and there is nothing to merge. Appended outputs publish the remaining
    // final states.
    let mut results = Vec::with_capacity(source_output_count);
    for (position, output) in outputs.into_iter().enumerate() {
        if position >= source_output_count {
            context.set_discharged_state(leaving[position - source_output_count], output, true)?;
            continue;
        }
        match output_allocations[position] {
            Some(allocation) => {
                context.merge_boundary_state(&summary, widening.threaded(), allocation, output)?;
                let forwarded = forwarded_allocations
                    .iter()
                    .position(|candidate| *candidate == Some(allocation))
                    .and_then(|position| forwarded.get(position).cloned());
                results.push(match forwarded {
                    Some(forwarded) => forwarded,
                    None => context.allocation_reference(allocation)?,
                });
            }
            None => results.push(ReferenceDischargeValue::Value(output)),
        }
    }
    Ok(results)
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::collections::{HashMap, HashSet};

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::captures::CaptureReference;
    use crate::parameters::Placeholder;
    use crate::programs::ProgramError;
    use crate::programs::atoms::AtomId;
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::effects::{Effect, Effects};
    use crate::programs::instructions::{Instruction, InstructionId};
    use crate::programs::operations::Operation;
    use crate::programs::programs::ProgramRenderingMode;
    use crate::programs::provenance::{Provenance, ProvenanceScope};
    use crate::programs::references::discharge::tests::*;
    use crate::programs::references::semantics::{ReferenceAccessMode, ReferenceInput, ReferenceOperationSemantics};
    use crate::programs::references::types::ReferenceType;
    use crate::programs::regions::{EmptyRegionDriver, RegionId, RegionInterface};
    use crate::programs::types::{TypeError, Typed};

    use super::*;

    /// Capture-constant family used by the capture-aware transform tests.
    type ListCapture = CaptureReference<ListIrType>;

    /// Closed list program used by the capture-aware transform tests.
    type ClosedListProgram = ClosedProgram<ListIrValue, ListCapture, ListOperation, Vec<ListCapture>, Vec<ListCapture>>;

    /// Builds a program exposing one external input and one internal allocation as selectable
    /// [`ReferenceDischargeTarget`]s.
    fn program_with_external_and_internal_reference_targets()
    -> Program<TestValue, TestOperation, Vec<TestValue>, Vec<TestValue>> {
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let public = builder.add_input(reference_type(0));
        let initial = builder.add_input(TestType::Value(0));
        let allocation =
            builder.add_instruction(TestOperation::NewAllocation, Vec::new(), vec![initial], None).unwrap()[0];
        let read = builder.add_instruction(TestOperation::Read, Vec::new(), vec![public], None).unwrap()[0];
        let frozen = builder.add_instruction(TestOperation::Consume, Vec::new(), vec![allocation], None).unwrap()[0];
        builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![read, frozen], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap()
    }

    /// Builds a closed program whose attached region reads a reference solely through a capture constant.
    fn closed_list_program_with_nested_reference_capture() -> ClosedListProgram {
        let reference_type = ReferenceType::new(ListType { length: 2 });
        let mut callee_builder = ProgramBuilder::<ListCapture, ListOperation>::new();
        let captured_reference =
            callee_builder.add_constant(ListCapture::new(0, ListIrType::Reference(reference_type.clone())));
        let observed = callee_builder
            .add_instruction(ListOperation::Read, Vec::new(), vec![captured_reference], None)
            .unwrap()[0];
        let callee = callee_builder
            .build::<Vec<ListCapture>, Vec<ListCapture>>(vec![observed], Vec::<Placeholder>::new(), vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<ListCapture, ListOperation>::new();
        let callee = builder.import_program(callee);
        let observed = builder.add_instruction(ListOperation::Call, vec![callee], Vec::new(), None).unwrap()[0];
        let source = builder
            .build::<Vec<ListCapture>, Vec<ListCapture>>(vec![observed], Vec::<Placeholder>::new(), vec![Placeholder])
            .unwrap();
        ClosedProgram::new(source, vec![ListIrValue::Reference(reference_type)]).unwrap()
    }

    #[test]
    fn test_reference_discharge_target_ordering_hashing_and_rendering() {
        let program = boundary_program(0, 0);
        let entry = program.entry();
        let external_capture = ReferenceDischargeTarget::External(ReferenceSource::Capture { index: 0 });
        let external_input = ReferenceDischargeTarget::External(ReferenceSource::Input { index: 0 });
        let first_internal =
            ReferenceDischargeTarget::Internal { instruction: InstructionId::new(entry, 0), output_index: 0 };
        let second_output =
            ReferenceDischargeTarget::Internal { instruction: InstructionId::new(entry, 0), output_index: 1 };
        let second_internal =
            ReferenceDischargeTarget::Internal { instruction: InstructionId::new(entry, 1), output_index: 0 };

        assert_eq!(external_capture, external_capture);
        assert_ne!(external_capture, external_input);
        assert!(external_capture < external_input);
        assert!(external_input < first_internal);
        assert!(first_internal < second_output);
        assert!(second_output < second_internal);

        let targets = HashSet::from([external_capture, first_internal]);
        assert!(targets.contains(&external_capture));
        assert!(targets.contains(&first_internal));
        assert!(!targets.contains(&external_input));
        assert_eq!(external_capture.to_string(), "external capture 0");
        assert_eq!(first_internal.to_string(), "internal allocation at `^0[0]` output 0");
        assert_eq!(format!("{external_capture:?}"), "External(Capture { index: 0 })");
    }

    #[test]
    fn test_reference_discharge_targets_select_everything_or_only_requested_targets() {
        let program = program_with_external_and_internal_reference_targets();
        let external = ReferenceDischargeTarget::External(ReferenceSource::Input { index: 0 });
        let internal =
            ReferenceDischargeTarget::Internal { instruction: InstructionId::new(program.entry(), 0), output_index: 0 };

        let everything = ReferenceDischargeTargets::everything();
        assert!(everything.selects(external));
        assert!(everything.selects(internal));

        let selected = ReferenceDischargeTargets::from_targets(&program, 0, &[external]).unwrap();
        assert_eq!(selected.targets.as_ref().unwrap().len(), 1);
        assert!(selected.selects(external));
        assert!(!selected.selects(internal));
        let cloned = selected.clone();
        assert!(cloned.selects(external));
        assert!(!cloned.selects(internal));

        let empty = ReferenceDischargeTargets::from_targets(&program, 0, &[]).unwrap();
        assert!(!empty.selects(external));
        assert!(!empty.selects(internal));
    }

    #[test]
    fn test_program_reference_discharge_targets_classifies_external_sources() {
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        builder.add_input(reference_type(0));
        builder.add_input(reference_type(1));
        let value = builder.add_input(TestType::Value(0));
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![value], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();

        assert_eq!(
            program.reference_discharge_targets(0),
            Ok(vec![
                ReferenceDischargeTarget::External(ReferenceSource::Input { index: 0 }),
                ReferenceDischargeTarget::External(ReferenceSource::Input { index: 1 }),
            ]),
        );
        assert_eq!(
            program.reference_discharge_targets(1),
            Ok(vec![
                ReferenceDischargeTarget::External(ReferenceSource::Capture { index: 0 }),
                ReferenceDischargeTarget::External(ReferenceSource::Input { index: 0 }),
            ]),
        );
        assert_eq!(
            program.reference_discharge_targets(2),
            Ok(vec![
                ReferenceDischargeTarget::External(ReferenceSource::Capture { index: 0 }),
                ReferenceDischargeTarget::External(ReferenceSource::Capture { index: 1 }),
            ]),
        );
        assert_eq!(program.reference_discharge_targets(3), program.reference_discharge_targets(2));
    }

    #[test]
    fn test_program_reference_discharge_targets_enumerates_nested_allocations_once_in_canonical_order() {
        // A callee region that allocates its own local allocation, so that enumeration is exercised across the complete
        // attached region closure rather than the entry region alone.
        let mut callee_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let initial = callee_builder.add_input(TestType::Value(0));
        let allocation = callee_builder
            .add_instruction(TestOperation::NewAllocation, Vec::new(), vec![initial], None)
            .unwrap()[0];
        let frozen =
            callee_builder.add_instruction(TestOperation::Consume, Vec::new(), vec![allocation], None).unwrap()[0];
        let callee = callee_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![frozen], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let initial = builder.add_input(TestType::Value(0));
        let callee = builder.import_program(callee);
        let local = builder.add_instruction(TestOperation::NewAllocation, Vec::new(), vec![initial], None).unwrap()[0];
        let called = builder.add_instruction(TestOperation::Call, vec![callee], vec![initial], None).unwrap()[0];

        // The same callee region is attached twice, so its interior allocation must be enumerated once rather than
        // once per invocation.
        let repeated = builder.add_instruction(TestOperation::Call, vec![callee], vec![initial], None).unwrap()[0];
        let frozen = builder.add_instruction(TestOperation::Consume, Vec::new(), vec![local], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![called, repeated, frozen],
                vec![Placeholder],
                vec![Placeholder; 3],
            )
            .unwrap();

        // Closure traversal order is unspecified, so internal targets must be sorted by instruction and output
        // position. Importing the same region at two call sites must not duplicate the allocation it contains.
        assert_eq!(
            program.reference_discharge_targets(0),
            Ok(vec![
                ReferenceDischargeTarget::Internal { instruction: InstructionId::new(callee, 0), output_index: 0 },
                ReferenceDischargeTarget::Internal {
                    instruction: InstructionId::new(program.entry(), 0),
                    output_index: 0,
                },
            ]),
        );
    }

    #[test]
    fn test_program_reference_discharge_targets_returns_an_empty_set_for_a_reference_free_program() {
        assert_eq!(boundary_program(2, 1).reference_discharge_targets(1), Ok(Vec::new()));
    }

    #[test]
    fn test_program_reference_discharge_targets_rejects_an_oversized_capture_prefix() {
        assert_eq!(
            boundary_program(3, 0).reference_discharge_targets(4),
            Err(ProgramError::MalformedProgram(
                "reference discharge target enumeration requests 4 captures but the program has 3 inputs".to_string(),
            )),
        );
    }

    #[test]
    fn test_reference_discharge_targets_construction_accepts_empty_and_reordered_valid_sets() {
        let program = program_with_external_and_internal_reference_targets();
        let external = ReferenceDischargeTarget::External(ReferenceSource::Input { index: 0 });
        let internal =
            ReferenceDischargeTarget::Internal { instruction: InstructionId::new(program.entry(), 0), output_index: 0 };

        let empty = ReferenceDischargeTargets::from_targets(&program, 0, &[]).unwrap();
        assert_eq!(empty.targets.as_ref().unwrap().len(), 0);

        let reordered = ReferenceDischargeTargets::from_targets(&program, 0, &[internal, external]).unwrap();
        assert_eq!(reordered.targets.as_ref().unwrap().len(), 2);
        assert!(reordered.selects(external));
        assert!(reordered.selects(internal));
    }

    #[test]
    fn test_reference_discharge_targets_construction_rejects_invalid_set_shape() {
        let program = program_with_external_and_internal_reference_targets();
        let entry = program.entry();
        let allocation =
            ReferenceDischargeTarget::Internal { instruction: InstructionId::new(entry, 0), output_index: 0 };

        assert_eq!(
            ReferenceDischargeTargets::from_targets(&program, 3, &[]).unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge target validation requests 3 captures but the program has 2 inputs".to_string(),
            ),
        );
        assert_eq!(
            ReferenceDischargeTargets::from_targets(&program, 0, &[allocation, allocation]).unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge targets contain internal allocation at `^0[0]` output 0 more than once"
                    .to_string(),
            ),
        );

        // Duplicate detection runs before kind validation because repetition is ambiguous regardless of what is named.
        let invalid = ReferenceDischargeTarget::External(ReferenceSource::Input { index: 7 });
        assert_eq!(
            ReferenceDischargeTargets::from_targets(&program, 0, &[invalid, invalid]).unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge targets contain external input 7 more than once".to_string(),
            ),
        );
    }

    #[test]
    fn test_reference_discharge_targets_construction_rejects_invalid_external_targets() {
        let program = program_with_external_and_internal_reference_targets();

        assert_eq!(
            ReferenceDischargeTargets::from_targets(
                &program,
                0,
                &[ReferenceDischargeTarget::External(ReferenceSource::Capture { index: 0 })],
            )
            .unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge targets include external capture 0 which is not selectable in this program"
                    .to_string(),
            ),
        );
        assert_eq!(
            ReferenceDischargeTargets::from_targets(
                &program,
                0,
                &[ReferenceDischargeTarget::External(ReferenceSource::Input { index: 2 })],
            )
            .unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge targets include external input 2 which is not selectable in this program"
                    .to_string(),
            ),
        );
        assert_eq!(
            ReferenceDischargeTargets::from_targets(
                &program,
                0,
                &[ReferenceDischargeTarget::External(ReferenceSource::Input { index: 1 })],
            )
            .unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge targets include external input 1 which is not selectable in this program"
                    .to_string(),
            ),
        );
        assert_eq!(
            ReferenceDischargeTargets::from_targets(
                &program,
                0,
                &[ReferenceDischargeTarget::External(ReferenceSource::Input { index: usize::MAX })],
            )
            .unwrap_err(),
            ProgramError::MalformedProgram(format!(
                "reference discharge targets include external input {} which is not selectable in this program",
                usize::MAX,
            )),
        );
    }

    #[test]
    fn test_reference_discharge_targets_construction_rejects_invalid_internal_targets() {
        let program = program_with_external_and_internal_reference_targets();
        let entry = program.entry();
        assert_eq!(
            ReferenceDischargeTargets::from_targets(
                &program,
                0,
                &[ReferenceDischargeTarget::Internal { instruction: InstructionId::new(entry, 7), output_index: 0 }],
            )
            .unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge targets include internal allocation at `^0[7]` output 0 which is not selectable \
                 in this program"
                    .to_string(),
            ),
        );
        assert_eq!(
            ReferenceDischargeTargets::from_targets(
                &program,
                0,
                &[ReferenceDischargeTarget::Internal { instruction: InstructionId::new(entry, 1), output_index: 0 }],
            )
            .unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge targets include internal allocation at `^0[1]` output 0 which is not selectable \
                 in this program"
                    .to_string(),
            ),
        );
        assert_eq!(
            ReferenceDischargeTargets::from_targets(
                &program,
                0,
                &[ReferenceDischargeTarget::Internal { instruction: InstructionId::new(entry, 0), output_index: 1 }],
            )
            .unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge targets include internal allocation at `^0[0]` output 1 which is not selectable \
                 in this program"
                    .to_string(),
            ),
        );
    }

    #[test]
    fn test_reference_discharge_result_accessors_and_into_parts() {
        let bindings = vec![
            ExternalReferenceBinding::new(ReferenceSource::Capture { index: 0 }, Some(1)),
            ExternalReferenceBinding::new(ReferenceSource::Input { index: 0 }, None),
        ];
        let result = ReferenceDischargeResult::try_from(
            PartialReferenceDischargeResult::new(boundary_program(2, 2), 1, 1, bindings.clone()).unwrap(),
        )
        .unwrap();
        assert_eq!(result.program().input_count(), 2);
        assert_eq!(result.program().output_count(), 2);
        assert_eq!(result.capture_count(), 1);
        assert_eq!(result.output_count(), 1);
        assert_eq!(result.external_reference_bindings(), bindings);

        let (program, capture_count, output_count, external_reference_bindings) = result.into_parts();
        assert_eq!(program.input_count(), 2);
        assert_eq!(program.output_count(), 2);
        assert_eq!(capture_count, 1);
        assert_eq!(output_count, 1);
        assert_eq!(external_reference_bindings, bindings);
    }

    #[test]
    fn test_reference_discharge_result_converts_only_without_external_references() {
        let local = ReferenceDischargeResult::try_from(
            PartialReferenceDischargeResult::new(boundary_program(1, 1), 0, 1, Vec::new()).unwrap(),
        )
        .unwrap();
        let program = local.into_program_without_external_references().unwrap();
        assert_eq!(program.input_count(), 1);
        assert_eq!(program.output_count(), 1);

        let external = ReferenceDischargeResult::try_from(
            PartialReferenceDischargeResult::new(
                boundary_program(1, 1),
                0,
                1,
                vec![ExternalReferenceBinding::new(ReferenceSource::Input { index: 0 }, None)],
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(
            external.into_program_without_external_references().unwrap_err(),
            ProgramError::UnsupportedOperation {
                message: "reference discharge cannot discard the binding for external `input 0`".to_string(),
            },
        );
    }

    #[test]
    fn test_partial_reference_discharge_result_accessors_and_into_parts() {
        let bindings = vec![
            ExternalReferenceBinding::new(ReferenceSource::Capture { index: 0 }, Some(1)),
            ExternalReferenceBinding::new(ReferenceSource::Input { index: 0 }, None),
        ];
        let result = PartialReferenceDischargeResult::new(boundary_program(2, 2), 1, 1, bindings.clone()).unwrap();
        assert_eq!(result.program().input_count(), 2);
        assert_eq!(result.program().output_count(), 2);
        assert_eq!(result.capture_count(), 1);
        assert_eq!(result.output_count(), 1);
        assert_eq!(result.external_reference_bindings(), bindings);

        let (program, capture_count, output_count, external_reference_bindings) = result.into_parts();
        assert_eq!(program.input_count(), 2);
        assert_eq!(program.output_count(), 2);
        assert_eq!(capture_count, 1);
        assert_eq!(output_count, 1);
        assert_eq!(external_reference_bindings, bindings);
    }

    #[test]
    fn test_partial_reference_discharge_result_new_rejects_invalid_boundary_counts() {
        assert_eq!(
            PartialReferenceDischargeResult::new(boundary_program(0, 0), 1, 0, Vec::new()).unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge reports 1 captures but discharged input count is 0".to_string(),
            ),
        );

        assert_eq!(
            PartialReferenceDischargeResult::new(boundary_program(0, 1), 0, 2, Vec::new()).unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge reports 2 public outputs but discharged output count is 1".to_string(),
            ),
        );
    }

    #[test]
    fn test_partial_reference_discharge_result_new_rejects_invalid_external_sources() {
        assert_eq!(
            PartialReferenceDischargeResult::new(
                boundary_program(2, 0),
                1,
                0,
                vec![ExternalReferenceBinding::new(ReferenceSource::Capture { index: 1 }, None)],
            )
            .unwrap_err(),
            ProgramError::MalformedProgram(
                "reference source capture 1 lies outside the capture prefix of length 1".to_string(),
            ),
        );
        assert_eq!(
            PartialReferenceDischargeResult::new(
                boundary_program(0, 0),
                0,
                0,
                vec![ExternalReferenceBinding::new(ReferenceSource::Input { index: 0 }, None)],
            )
            .unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge state for `input 0` names input 0 but discharged input count is 0".to_string(),
            ),
        );

        let duplicate = vec![
            ExternalReferenceBinding::new(ReferenceSource::Capture { index: 0 }, None),
            ExternalReferenceBinding::new(ReferenceSource::Capture { index: 0 }, None),
        ];
        assert_eq!(
            PartialReferenceDischargeResult::new(boundary_program(2, 0), 1, 0, duplicate).unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge state source `capture 0` does not follow source `capture 0` in canonical boundary \
                 order"
                    .to_string(),
            ),
        );

        let decreasing = vec![
            ExternalReferenceBinding::new(ReferenceSource::Input { index: 0 }, None),
            ExternalReferenceBinding::new(ReferenceSource::Capture { index: 0 }, None),
        ];
        assert_eq!(
            PartialReferenceDischargeResult::new(boundary_program(2, 0), 1, 0, decreasing).unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge state source `capture 0` does not follow source `input 0` in canonical boundary \
                 order"
                    .to_string(),
            ),
        );
    }

    #[test]
    fn test_partial_reference_discharge_result_new_rejects_invalid_hidden_outputs() {
        // A program output not covered by the public prefix or a mutated binding is an unaccounted hidden output.
        assert_eq!(
            PartialReferenceDischargeResult::new(boundary_program(1, 1), 0, 0, Vec::new()).unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge final states end at output 0 but discharged output count is 1".to_string(),
            ),
        );

        // A mutated binding cannot append a hidden output after the program's complete output boundary.
        assert_eq!(
            PartialReferenceDischargeResult::new(
                boundary_program(1, 1),
                0,
                1,
                vec![ExternalReferenceBinding::new(ReferenceSource::Input { index: 0 }, Some(1))],
            )
            .unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge final states end at output 2 but discharged output count is 1".to_string(),
            ),
        );

        // Mutated bindings tile the hidden suffix exactly in binding order; they cannot name a public output.
        assert_eq!(
            PartialReferenceDischargeResult::new(
                boundary_program(1, 2),
                0,
                1,
                vec![ExternalReferenceBinding::new(ReferenceSource::Input { index: 0 }, Some(0))],
            )
            .unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge final-state output 0 for `input 0` does not match expected hidden output 1"
                    .to_string(),
            ),
        );
    }

    #[test]
    fn test_reference_discharge_result_try_from_enforces_reference_freedom() {
        // Operation family that separates the two facts the reference-freedom proof must distinguish: an unrelated
        // ordered-state operation that discharge never touches, and a retained reference operation that it must reject
        // even though its boundary types contain no references.
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
        let discharged =
            ReferenceDischargeResult::try_from(partial(program(&[ProofOperation::OrderedIo], TestType::Value(0))))
                .unwrap();
        assert_eq!(discharged.output_count(), 1);
        assert!(discharged.program().effects().contains(Effect::OrderedIo));

        // A surviving reference-typed value is disqualifying wherever it appears, including on the boundary.
        assert_eq!(
            ReferenceDischargeResult::try_from(partial(program(&[ProofOperation::OrderedIo], reference_type(0),)))
                .unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge program still contains a reference-typed value and cannot form a full discharge"
                    .to_string(),
            ),
        );

        // A retained reference operation is disqualifying even when every value in the program is non-reference.
        assert_eq!(
            ReferenceDischargeResult::try_from(partial(program(
                &[ProofOperation::OrderedIo, ProofOperation::RetainedReference],
                TestType::Value(0),
            )))
            .unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge program retains reference operation `test.retained_reference` at `^0[1]` and \
                 cannot form a full discharge"
                    .to_string(),
            ),
        );
    }

    #[test]
    fn test_reference_discharge_result_try_from_checks_the_attached_region_closure() {
        // The entry boundary is reference-free, but its attached callee allocates and consumes a local reference.
        // Inspecting only the entry region would therefore accept this program incorrectly.
        let mut callee_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let initial = callee_builder.add_input(TestType::Value(0));
        let allocation = callee_builder
            .add_instruction(TestOperation::NewAllocation, Vec::new(), vec![initial], None)
            .unwrap()[0];
        let value =
            callee_builder.add_instruction(TestOperation::Consume, Vec::new(), vec![allocation], None).unwrap()[0];
        let callee = callee_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![value], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let initial = builder.add_input(TestType::Value(0));
        let callee = builder.import_program(callee);
        let value = builder.add_instruction(TestOperation::Call, vec![callee], vec![initial], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![value], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let partial = PartialReferenceDischargeResult::new(program, 0, 1, Vec::new()).unwrap();

        assert_eq!(
            ReferenceDischargeResult::try_from(partial).unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge program still contains a reference-typed value and cannot form a full discharge"
                    .to_string(),
            ),
        );
    }

    #[test]
    fn test_external_reference_binding_accessors_and_serialization() {
        let read_only = ExternalReferenceBinding::new(ReferenceSource::Capture { index: 0 }, None);
        let mutated = ExternalReferenceBinding::new(ReferenceSource::Input { index: 2 }, Some(3));

        assert_eq!(read_only.source(), ReferenceSource::Capture { index: 0 });
        assert!(!read_only.is_mutated());
        assert_eq!(read_only.output_index(), None);
        assert_eq!(mutated.source(), ReferenceSource::Input { index: 2 });
        assert!(mutated.is_mutated());
        assert_eq!(mutated.output_index(), Some(3));
        assert_eq!(read_only, read_only);
        assert_ne!(read_only, mutated);
        let bindings = HashMap::from([(read_only, "read-only"), (mutated, "mutated")]);
        assert_eq!(bindings.get(&read_only), Some(&"read-only"));
        assert_eq!(bindings.get(&mutated), Some(&"mutated"));
        assert_eq!(
            format!("{mutated:?}"),
            "ExternalReferenceBinding { source: Input { index: 2 }, output_index: Some(3) }",
        );
        assert_eq!(
            serde_json::to_string(&[read_only, mutated]).unwrap(),
            r#"[{"source":{"capture":{"index":0}},"output_index":null},{"source":{"input":{"index":2}},"output_index":3}]"#,
        );
    }

    #[test]
    fn test_reference_source_flat_input_index_round_trips() {
        for (flat_input_index, capture_count, source) in [
            (0, 0, ReferenceSource::Input { index: 0 }),
            (0, 2, ReferenceSource::Capture { index: 0 }),
            (1, 2, ReferenceSource::Capture { index: 1 }),
            (2, 2, ReferenceSource::Input { index: 0 }),
            (4, 2, ReferenceSource::Input { index: 2 }),
        ] {
            assert_eq!(ReferenceSource::from_flat_input_index(flat_input_index, capture_count), source);
            assert_eq!(source.flat_input_index(capture_count), Ok(flat_input_index));
        }
    }

    #[test]
    fn test_reference_source_flat_input_index_rejects_invalid_sources() {
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
    }

    #[test]
    fn test_reference_source_ordering_rendering_and_serialization() {
        let first_capture = ReferenceSource::Capture { index: 0 };
        let second_capture = ReferenceSource::Capture { index: 1 };
        let first_input = ReferenceSource::Input { index: 0 };
        let second_input = ReferenceSource::Input { index: 1 };

        assert!(first_capture < second_capture);
        assert!(second_capture < first_input);
        assert!(first_input < second_input);
        assert_eq!(first_capture.to_string(), "capture 0");
        assert_eq!(second_input.to_string(), "input 1");
        assert_eq!(format!("{first_capture:?}"), "Capture { index: 0 }");
        assert_eq!(format!("{second_input:?}"), "Input { index: 1 }");
        assert_eq!(
            serde_json::to_string(&[first_capture, second_input]).unwrap(),
            r#"[{"capture":{"index":0}},{"input":{"index":1}}]"#,
        );
    }

    #[test]
    fn test_reference_discharge_value_reports_operand_kind_mismatches() {
        // A rule that receives the wrong carrier kind gets a diagnostic naming what it expected, which is what keeps
        // an open set of third-party rules diagnosable without each of them inventing its own message.
        let context = ListDischargeContext::new(ListDestination::new());
        let reference_type = ReferenceType::new(ListType { length: 1 });
        let allocated = context.bind_discharged(reference_type, ListIrValue::List(vec![1])).unwrap();
        let allocation = allocated.try_as_reference("the allocated allocation").unwrap().allocation_id();
        let value: ListDischargeValue = ReferenceDischargeValue::Value(ListIrValue::List(vec![1]));

        assert_eq!(value.try_as_value("an update value"), Ok(&ListIrValue::List(vec![1])));
        assert_eq!(
            allocated.try_as_value("an update value"),
            Err(ProgramError::MalformedProgram(format!(
                "reference discharge expected an update value but received {allocation} ref<list<1>>",
            ))),
        );
        assert_eq!(
            value.try_as_reference("a reference to read"),
            Err(ProgramError::MalformedProgram(
                "reference discharge expected a reference to read but received a value".to_string(),
            )),
        );
    }

    #[test]
    fn test_reference_discharge_value_reports_its_type_and_display() {
        let context = ListDischargeContext::new(ListDestination::new());
        let value =
            ReferenceDischargeValue::<ListDestination, ListReferenceDischarge>::Value(ListIrValue::List(vec![1]));
        let reference_type = ReferenceType::new(ListType { length: 2 });
        let reference = context.bind_discharged(reference_type.clone(), ListIrValue::List(vec![1, 2])).unwrap();

        // A value carrier reports the wrapped destination value's type, while a reference handle reports its own
        // reference type lifted into the destination universe.
        assert_eq!(value.r#type().into_owned(), ListIrType::List(ListType { length: 1 }));
        assert_eq!(reference.r#type().into_owned(), ListIrType::Reference(reference_type));
        assert_eq!(value.to_string(), "[1]");
        let allocation = reference.try_as_reference("the allocated allocation").unwrap().allocation_id();
        assert_eq!(reference.to_string(), format!("{allocation} ref<list<2>>"));
        assert_eq!(
            format!("{reference:?}"),
            format!(
                "Reference(ReferenceDischargeReference {{ allocation_id: {allocation:?}, type: ReferenceType {{ \
                 referent: ListType {{ length: 2 }} }}, is_view: false, alias: ListAlias \
                 {{ offset: 0, length: 2 }}, \
                 binding: Discharged }})",
            ),
        );
    }

    #[test]
    fn test_reference_discharge_region_boundary() {
        let context = ListDischargeContext::new(ListDestination::new());
        let first = context
            .bind_discharged(ReferenceType::new(ListType { length: 1 }), ListIrValue::List(vec![1]))
            .unwrap();
        let first = first.try_as_reference("the first boundary allocation").unwrap().allocation_id();
        let second = context
            .bind_discharged(ReferenceType::new(ListType { length: 1 }), ListIrValue::List(vec![2]))
            .unwrap();
        let second = second.try_as_reference("the second boundary allocation").unwrap().allocation_id();

        let boundary = ReferenceDischargeRegionBoundary::new(
            &ListOperation::Call,
            0,
            vec![Some(first), None],
            ReferenceDischargeRegionStateInsertion::new(vec![second], 1),
            ReferenceDischargeRegionStateInsertion::new(vec![first, second], 2),
        );
        assert_eq!(boundary.declared_input_allocations(), &[Some(first), None]);
        assert_eq!(boundary.capture_input_count(), None);
        assert_eq!(boundary.added_inputs().allocations(), &[second]);
        assert_eq!(boundary.added_inputs().position(), 1);
        assert_eq!(boundary.added_outputs().allocations(), &[first, second]);
        assert_eq!(boundary.added_outputs().position(), 2);

        let symmetric = ReferenceDischargeRegionBoundary::symmetric(
            &ListOperation::Call,
            0,
            vec![Some(first)],
            ReferenceDischargeRegionStateInsertion::new(vec![second], 1),
        );
        assert_eq!(symmetric.added_inputs().allocations(), &[second]);
        assert_eq!(symmetric.added_inputs().position(), 1);
        assert_eq!(symmetric.added_outputs().allocations(), &[second]);
        assert_eq!(symmetric.added_outputs().position(), 1);
    }

    #[test]
    fn test_reference_discharge_region_result_holds_the_replay_to_the_widening_that_sized_it() {
        // The boundary is sized from a summary computed before the region ran, so both validators exist to catch an
        // operation whose generic hooks disagree with what its closure actually does. Here the result is produced
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
            .bind_discharged(ReferenceType::new(ListType { length: 2 }), ListIrValue::List(vec![1, 2]))
            .unwrap();
        let allocation = allocated.try_as_reference("the caller allocation").unwrap().allocation_id();
        let regions = [program];
        let driver = RecursiveReferenceDischargeDriver::new(&regions, None);
        let boundary = ReferenceDischargeRegionBoundary::new(
            &ListOperation::Call,
            0,
            vec![Some(allocation), None],
            ReferenceDischargeRegionStateInsertion::new(Vec::new(), 2),
            ReferenceDischargeRegionStateInsertion::new(Vec::new(), 2),
        );
        let result = driver.rebuild_region(&context, 0, &boundary).unwrap();

        // The region writes its entering allocation, so a widening that published nothing lost that update.
        assert_eq!(
            result.validate_predicted_mutations(&[], "list.call"),
            Err(ProgramError::MalformedProgram(format!(
                "operation `list.call` mutated {allocation} in an attached region that its state widening did not predict",
            ))),
        );
        assert_eq!(result.validate_predicted_mutations(&[allocation], "list.call"), Ok(()));

        // The region returns that allocation at its second output, so a widening that predicted a value there
        // would have published the allocation's final state twice.
        assert_eq!(result.output_allocations(), &[None, Some(allocation)]);
        assert_eq!(
            result.validate_predicted_output_allocations(&[None, None], "list.call"),
            Err(ProgramError::MalformedProgram(
                "operation `list.call` attaches a region whose outputs do not denote the references its state \
                 widening expected"
                    .to_string(),
            )),
        );
        assert_eq!(result.validate_predicted_output_allocations(&[None, Some(allocation)], "list.call"), Ok(()));
    }

    #[test]
    fn test_reference_discharge_policy_storage_alias_covers_the_complete_value() {
        assert_eq!(
            <ListReferenceDischarge as ReferenceDischargePolicy<ListDestination>>::storage_alias(&ListType {
                length: 4,
            }),
            ListAlias { offset: 0, length: 4 },
        );
    }

    #[test]
    fn test_reference_discharge_policy_read_returns_complete_values_and_views() {
        let destination = ListDestination::new();
        let current = ListIrValue::List(vec![1, 2, 3, 4]);

        // A storage alias reads the complete value, whereas a view alias reads only the part it describes.
        assert_eq!(
            ListReferenceDischarge::read(&destination, &current, &ListAlias { offset: 0, length: 4 }),
            Ok(ListIrValue::List(vec![1, 2, 3, 4])),
        );
        assert_eq!(
            ListReferenceDischarge::read(&destination, &current, &ListAlias { offset: 1, length: 2 }),
            Ok(ListIrValue::List(vec![2, 3])),
        );

        // An invalid view preserves the exact error produced by the reference family's selection operation.
        assert_eq!(
            ListReferenceDischarge::read(&destination, &current, &ListAlias { offset: 3, length: 2 }),
            Err(ProgramError::MalformedProgram("selection [3, 5) does not fit a list of length 4".to_string())),
        );
    }

    #[test]
    fn test_reference_discharge_policy_write_replaces_complete_values_and_views() {
        let destination = ListDestination::new();
        let current = ListIrValue::List(vec![1, 2, 3, 4]);

        assert_eq!(
            ListReferenceDischarge::write(
                &destination,
                &current,
                ListIrValue::List(vec![5, 6, 7, 8]),
                &ListAlias { offset: 0, length: 4 },
            ),
            Ok(ListIrValue::List(vec![5, 6, 7, 8])),
        );

        // Writing through a view replaces only that view and preserves the values around it.
        assert_eq!(
            ListReferenceDischarge::write(
                &destination,
                &current,
                ListIrValue::List(vec![20, 30]),
                &ListAlias { offset: 1, length: 2 },
            ),
            Ok(ListIrValue::List(vec![1, 20, 30, 4])),
        );
        assert_eq!(
            ListReferenceDischarge::write(
                &destination,
                &current,
                ListIrValue::List(vec![20, 30]),
                &ListAlias { offset: 3, length: 2 },
            ),
            Err(ProgramError::MalformedProgram("splice [3, 5) does not fit a list of length 4".to_string())),
        );
    }

    #[test]
    fn test_reference_discharge_policy_swap_returns_the_previous_view_and_complete_update() {
        let destination = ListDestination::new();
        let current = ListIrValue::List(vec![1, 2, 3, 4]);
        let view = ListAlias { offset: 1, length: 2 };

        // The default implementation returns the value read through the view first and the complete updated value
        // second.
        assert_eq!(
            ListReferenceDischarge::swap(&destination, &current, ListIrValue::List(vec![20, 30]), &view),
            Ok((ListIrValue::List(vec![2, 3]), ListIrValue::List(vec![1, 20, 30, 4]))),
        );

        // Because the default implementation reads before writing, a read failure is returned directly.
        assert_eq!(
            ListReferenceDischarge::swap(
                &destination,
                &current,
                ListIrValue::List(vec![20, 30]),
                &ListAlias { offset: 3, length: 2 },
            ),
            Err(ProgramError::MalformedProgram("selection [3, 5) does not fit a list of length 4".to_string())),
        );
    }

    #[test]
    fn test_reference_accumulation_policy_accumulate_updates_only_the_selected_view() {
        let destination = ListDestination::new();
        let current = ListIrValue::List(vec![1, 2, 3, 4]);
        let view = ListAlias { offset: 1, length: 2 };

        // Accumulation adds through the view and returns the complete value with everything outside the view intact.
        assert_eq!(
            ListReferenceDischarge::accumulate(&destination, &current, ListIrValue::List(vec![20, 30]), &view),
            Ok(ListIrValue::List(vec![1, 22, 33, 4])),
        );
        assert_eq!(
            ListReferenceDischarge::accumulate(&destination, &current, ListIrValue::List(vec![20]), &view),
            Err(ProgramError::MalformedProgram("cannot add lists of lengths 2 and 1".to_string())),
        );
    }

    #[test]
    fn test_reference_discharge_driver_empty_region_driver() {
        let context = ListDischargeContext::new(ListDestination::new());
        let boundary = ReferenceDischargeRegionBoundary::new(
            &ListOperation::Call,
            0,
            Vec::new(),
            ReferenceDischargeRegionStateInsertion::new(Vec::new(), 0),
            ReferenceDischargeRegionStateInsertion::new(Vec::new(), 0),
        );

        assert_eq!(
            ReferenceDischargeDriver::<ListDestination, ListReferenceDischarge>::source_instruction_id(
                &EmptyRegionDriver,
            ),
            None,
        );
        assert_eq!(
            EmptyRegionDriver.inline_region(&context, 0, Vec::new()),
            Err(ProgramError::MalformedProgram("empty region driver cannot discharge a region".to_string())),
        );
        assert_eq!(
            EmptyRegionDriver.rebuild_region(&context, 0, &boundary).unwrap_err(),
            ProgramError::MalformedProgram("empty region driver cannot rebuild a region".to_string()),
        );
    }

    #[test]
    fn test_recursive_reference_discharge_driver_new() {
        let regions = [ProgramBuilder::<ListIrValue, ListOperation>::new()
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(Vec::new(), Vec::new(), Vec::new())
            .unwrap()];
        let driver = RecursiveReferenceDischargeDriver::new(&regions, None);

        assert_eq!(driver.region_count(), 1);
    }

    #[test]
    fn test_recursive_reference_discharge_driver_source_instruction_id() {
        let source_instruction_id = InstructionId::new(RegionId::new(0), 3);
        let without_source = RecursiveReferenceDischargeDriver::new(&EmptyRegionDriver, None);
        let with_source = RecursiveReferenceDischargeDriver::new(&EmptyRegionDriver, Some(source_instruction_id));

        assert_eq!(
            ReferenceDischargeDriver::<ListDestination, ListReferenceDischarge>::source_instruction_id(&without_source,),
            None,
        );
        assert_eq!(
            ReferenceDischargeDriver::<ListDestination, ListReferenceDischarge>::source_instruction_id(&with_source),
            Some(source_instruction_id),
        );
    }

    #[test]
    fn test_recursive_reference_discharge_driver_inline_region() {
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let input = builder.add_input(ListIrType::List(ListType { length: 2 }));
        let program = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let context = ListDischargeContext::new(ListDestination::new());
        let input = ReferenceDischargeValue::Value(ListIrValue::List(vec![1, 2]));
        let regions = [program];
        let driver = RecursiveReferenceDischargeDriver::new(&regions, None);
        assert_eq!(driver.inline_region(&context, 0, vec![input.clone()]), Ok(vec![input]));

        // A reference stored as a program constant belongs to no allocation, so recursive replay rejects it rather
        // than allowing it to survive into the destination outside the allocation environment.
        let reference_type = ReferenceType::new(ListType { length: 2 });
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let stored = builder.add_constant(ListIrValue::Reference(reference_type));
        let program = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![stored], Vec::new(), vec![Placeholder])
            .unwrap();
        let regions = [program];
        let driver = RecursiveReferenceDischargeDriver::new(&regions, None);
        assert_eq!(
            driver.inline_region(&context, 0, Vec::new()),
            Err(ProgramError::MalformedProgram(
                "reference discharge cannot lift a constant of reference type `ref<list<2>>`; a reference enters a \
                 program through an input, a capture binding, or an allocation"
                    .to_string(),
            )),
        );
    }

    #[test]
    fn test_recursive_reference_discharge_driver_rebuild_region() {
        // A region that accumulates into the allocation it receives and returns that allocation unchanged,
        // which is the shape a structured rule threads state through.
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
        let allocated = context.bind_discharged(ReferenceType::new(ListType { length: 2 }), state).unwrap();
        let allocation = allocated.try_as_reference("the caller allocation").unwrap().allocation_id();
        let regions = [program];
        let driver = RecursiveReferenceDischargeDriver::new(&regions, None);
        let boundary = ReferenceDischargeRegionBoundary::new(
            &ListOperation::Call,
            0,
            vec![Some(allocation)],
            ReferenceDischargeRegionStateInsertion::new(Vec::new(), 0),
            ReferenceDischargeRegionStateInsertion::new(Vec::new(), 1),
        );
        let result = driver.rebuild_region(&context, 0, &boundary).unwrap();

        // The rebuilt region reports what it did in the caller's own terms, and the caller's environment is untouched:
        // the allocation is still unmutated and still holds the state it entered with.
        assert_eq!(result.mutated_allocations(), [allocation]);
        assert_eq!(result.output_allocations(), &[Some(allocation)]);
        assert!(!context.is_mutated(allocation).unwrap());
        assert_eq!(context.discharged_state(allocation).unwrap().atom_id().unwrap(), AtomId::new(0));
        assert_eq!(
            result.program().to_string(),
            indoc! {"
                lambda %0:list<2> .
                let %1:list<2> = const [10, 10]
                    %2:list<2> = list.select %0
                    %3:list<2> = list.add %2 %1
                    %4:list<2> = list.splice %0 %3
                in (%4)"},
        );

        // A replay that fails leaves the caller's environment exactly as it was and yields no values at all, because
        // the rebuilt region's result type carries none. The checked append rejects a read of a consumed family at
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
            driver.rebuild_region(&context, 0, &boundary),
            Err(ProgramError::MalformedProgram(message))
                if message.starts_with("reference discharge accessed consumed reference allocation "),
        ));
        assert!(!context.is_mutated(allocation).unwrap());
        assert_eq!(context.discharged_state(allocation).unwrap().atom_id().unwrap(), AtomId::new(0));
    }

    #[test]
    fn test_recursive_reference_discharge_driver_rebuild_region_rebinds_capture_scope() {
        // A region whose closure reaches a caller allocation through a capture constant declares no boundary position
        // for it, so the rule threads it as added state. The region context rebinds the caller's scope onto the region
        // allocation corresponding to that caller allocation, which lets the rebuilt body resolve the same constant
        // against its own isolated environment.
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let captured = builder.add_constant(ListIrValue::Reference(ReferenceType::new(ListType { length: 2 })));
        let snapshot = builder.add_instruction(ListOperation::Read, Vec::new(), vec![captured], None).unwrap()[0];
        let program = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![snapshot], Vec::new(), vec![Placeholder])
            .unwrap();

        let context = ListDischargeContext::new(ListDestination::new());
        let allocated = context
            .bind_discharged(ReferenceType::new(ListType { length: 2 }), ListIrValue::List(vec![1, 2]))
            .unwrap();
        let allocation = allocated.try_as_reference("the captured allocation").unwrap().allocation_id();
        let context = context.with_captures(ReferenceDischargeCaptureScope::new(
            list_capture_position,
            vec![None, None, Some(allocation)],
        ));

        // The summary reports the capture-scoped access in caller-allocation terms, which is what sizes the boundary.
        let summary = context.region_summary(&ListOperation::Call, 0, program.entry_region_ref(), &[]).unwrap();
        assert_eq!(summary.accessed().collect::<Vec<_>>(), vec![allocation]);
        assert!(!summary.is_mutated(allocation));
        assert_eq!(summary.output_allocations(), &[None]);

        let regions = [program];
        let driver = RecursiveReferenceDischargeDriver::new(&regions, None);
        let boundary = ReferenceDischargeRegionBoundary::new(
            &ListOperation::Call,
            0,
            Vec::new(),
            ReferenceDischargeRegionStateInsertion::new(vec![allocation], 0),
            ReferenceDischargeRegionStateInsertion::new(Vec::new(), 0),
        );
        let result = driver.rebuild_region(&context, 0, &boundary).unwrap();
        assert_eq!(
            result.program().to_string(),
            indoc! {"
                lambda %0:list<2> .
                let %1:list<2> = list.select %0
                in (%1)"},
        );
        assert_eq!(result.output_allocations(), &[None]);
        assert!(result.mutated_allocations().is_empty());

        // The caller environment is untouched: the rebuilt region read its own threaded copy of the state.
        assert_eq!(context.discharged_state(allocation), Ok(ListIrValue::List(vec![1, 2])));
        assert_eq!(context.is_mutated(allocation), Ok(false));
    }

    #[test]
    fn test_recursive_reference_discharge_driver_rebuild_region_rejects_same_type_view_output() {
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
        let allocated = context.bind_discharged(reference_type, ListIrValue::List(vec![1, 2])).unwrap();
        let allocation = allocated.try_as_reference("the caller allocation").unwrap().allocation_id();
        let regions = [program];
        let driver = RecursiveReferenceDischargeDriver::new(&regions, None);
        let boundary = ReferenceDischargeRegionBoundary::new(
            &ListOperation::Call,
            0,
            vec![Some(allocation)],
            ReferenceDischargeRegionStateInsertion::new(Vec::new(), 1),
            ReferenceDischargeRegionStateInsertion::new(Vec::new(), 1),
        );

        assert_eq!(
            driver.rebuild_region(&context, 0, &boundary).unwrap_err(),
            ProgramError::MalformedProgram(format!(
                "reference discharge cannot publish the view `ref<list<2>>` of {allocation} from region `{}`, \
                 whose boundary carries the complete stored value `ref<list<2>>`",
                regions[0].entry_region_ref().id(),
            )),
        );
    }

    #[test]
    fn test_recursive_reference_discharge_driver_rebuild_region_rejects_duplicate_added_allocations() {
        let builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let program = builder.build::<Vec<ListIrValue>, Vec<ListIrValue>>(Vec::new(), Vec::new(), Vec::new()).unwrap();

        let context = ListDischargeContext::new(ListDestination::new());
        let allocated = context
            .bind_discharged(ReferenceType::new(ListType { length: 2 }), ListIrValue::List(vec![1, 2]))
            .unwrap();
        let allocation = allocated.try_as_reference("the added allocation").unwrap().allocation_id();
        let regions = [program];
        let driver = RecursiveReferenceDischargeDriver::new(&regions, None);
        let boundary = ReferenceDischargeRegionBoundary::new(
            &ListOperation::Call,
            0,
            Vec::new(),
            ReferenceDischargeRegionStateInsertion::new(vec![allocation, allocation], 0),
            ReferenceDischargeRegionStateInsertion::new(Vec::new(), 0),
        );

        assert_eq!(
            driver.rebuild_region(&context, 0, &boundary).unwrap_err(),
            ProgramError::MalformedProgram(format!(
                "reference discharge adds {allocation} to region `{}` more than once",
                regions[0].entry_region_ref().id(),
            )),
        );
    }

    #[test]
    fn test_recursive_reference_discharge_driver_rebuild_region_rejects_added_declared_duplicate() {
        // A repeated _declared_ position deliberately aliases one caller allocation, but a synthesized state position
        // must never restate an allocation the boundary already declares: the rebuilt region would carry two boundary
        // positions for one state with no rule deciding which successor wins.
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let program = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(Vec::new(), vec![Placeholder], Vec::new())
            .unwrap();

        let context = ListDischargeContext::new(ListDestination::new());
        let allocated = context
            .bind_discharged(ReferenceType::new(ListType { length: 2 }), ListIrValue::List(vec![1, 2]))
            .unwrap();
        let allocation = allocated.try_as_reference("the declared allocation").unwrap().allocation_id();
        let regions = [program];
        let driver = RecursiveReferenceDischargeDriver::new(&regions, None);
        let boundary = ReferenceDischargeRegionBoundary::new(
            &ListOperation::Call,
            0,
            vec![Some(allocation)],
            ReferenceDischargeRegionStateInsertion::new(vec![allocation], 1),
            ReferenceDischargeRegionStateInsertion::new(Vec::new(), 0),
        );

        assert_eq!(
            driver.rebuild_region(&context, 0, &boundary).unwrap_err(),
            ProgramError::MalformedProgram(format!(
                "reference discharge adds {allocation} to region `{}` more than once",
                regions[0].entry_region_ref().id(),
            )),
        );
    }

    #[test]
    fn test_recursive_reference_discharge_driver_rebuild_region_propagates_consumed_allocation() {
        // This operation deliberately violates the generic contract: its summary claims no reference access, while its
        // discharge rule consumes the allocation. Rebuilt-region output validation must report that consumed allocation
        // instead of silently omitting it from the mutation report.
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let reference = builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let frozen =
            builder.add_instruction(ListOperation::UnreportedFreeze, Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![frozen], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let context = ListDischargeContext::new(ListDestination::new());
        let allocated = context
            .bind_discharged(ReferenceType::new(ListType { length: 2 }), ListIrValue::List(vec![1, 2]))
            .unwrap();
        let allocation = allocated.try_as_reference("the caller allocation").unwrap().allocation_id();
        let regions = [program];
        let driver = RecursiveReferenceDischargeDriver::new(&regions, None);
        let boundary = ReferenceDischargeRegionBoundary::new(
            &ListOperation::Call,
            0,
            vec![Some(allocation)],
            ReferenceDischargeRegionStateInsertion::new(Vec::new(), 1),
            ReferenceDischargeRegionStateInsertion::new(Vec::new(), 1),
        );

        assert!(matches!(
            driver.rebuild_region(&context, 0, &boundary),
            Err(ProgramError::MalformedProgram(message))
                if message.contains("reference discharge accessed consumed reference allocation"),
        ));
    }

    #[test]
    fn test_recursive_reference_discharge_driver_rebuild_region_inserts_added_state() {
        // Added state is what a region closure reaches without receiving it as a declared operand. No source construct
        // the interpreter currently accepts produces one (a reference reaches a region only through its boundary,
        // because a reference-typed constant is rejected outright) so the mechanics are exercised here directly,
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
            .bind_discharged(reference_type.clone(), destination.input(ListIrType::List(ListType { length: 2 })))
            .unwrap();
        let accessed = accessed.try_as_reference("the accessed allocation").unwrap().allocation_id();
        let carried = context
            .bind_discharged(reference_type, destination.input(ListIrType::List(ListType { length: 2 })))
            .unwrap();
        let carried = carried.try_as_reference("the carried allocation").unwrap().allocation_id();

        // The added input goes between the two declared inputs and the added output goes before the declared output,
        // which is the insertion arithmetic a scan's carry prefix depends on.
        let regions = [program];
        let driver = RecursiveReferenceDischargeDriver::new(&regions, None);
        let boundary = ReferenceDischargeRegionBoundary::new(
            &ListOperation::Call,
            0,
            vec![None, Some(accessed)],
            ReferenceDischargeRegionStateInsertion::new(vec![carried], 1),
            ReferenceDischargeRegionStateInsertion::new(vec![carried], 0),
        );
        let result = driver.rebuild_region(&context, 0, &boundary).unwrap();
        assert_eq!(
            result.program().to_string(),
            indoc! {"
                lambda %0:list<2>, %1:list<2>, %2:list<2> .
                let %3:list<2> = list.select %2
                    %4:list<2> = list.add %3 %0
                    %5:list<2> = list.splice %2 %4
                    %6:list<2> = list.select %5
                in (%1, %6)"},
        );

        // Only the allocation the closure actually reached is reported as mutated; the carried one passes through,
        // which is why a symmetric boundary can thread it without claiming the region wrote it.
        assert_eq!(result.mutated_allocations(), [accessed]);
        assert_eq!(result.output_allocations(), &[None]);
    }

    #[test]
    fn test_reference_dischargeable_operation_discharge_references() {
        // A region-free application has no source instruction, so an allocation rule sees `None` and treats the
        // allocation as unconditionally discharged.
        let context = ListDischargeContext::new(ListDestination::new());
        let input = ReferenceDischargeValue::Value(ListIrValue::List(vec![1, 2]));
        let driver = RecursiveReferenceDischargeDriver::new(&EmptyRegionDriver, None);

        OBSERVED_ALLOCATION_POSITIONS.with_borrow_mut(Vec::clear);
        ListOperation::ReferenceNew.discharge_references(&context, &driver, &[input]).unwrap();
        assert_eq!(OBSERVED_ALLOCATION_POSITIONS.with_borrow(Vec::clone), vec![None]);
    }

    #[test]
    fn test_reference_dischargeable_operation_discharge_references_threads_state_through_replayed_program() {
        // The program allocates one local reference, creates a composed view, accumulates into that view, replaces it,
        // adds the replaced and current selections, and finally freezes the complete stored value.
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let initial = builder.add_input(ListIrType::List(ListType { length: 4 }));
        let allocation =
            builder.add_instruction(ListOperation::ReferenceNew, Vec::new(), vec![initial], None).unwrap()[0];
        let view = builder
            .add_instruction(ListOperation::Slice { offset: 1, length: 2 }, Vec::new(), vec![allocation], None)
            .unwrap()[0];
        let update = builder.add_constant(ListIrValue::List(vec![10, 20]));
        builder.add_instruction(ListOperation::AddUpdate, Vec::new(), vec![view, update], None).unwrap();
        let replacement = builder.add_constant(ListIrValue::List(vec![7, 8]));
        builder.add_instruction(ListOperation::Write, Vec::new(), vec![view, replacement], None).unwrap();
        let replaced =
            builder.add_instruction(ListOperation::Swap, Vec::new(), vec![view, replacement], None).unwrap()[0];
        let snapshot = builder.add_instruction(ListOperation::Read, Vec::new(), vec![view], None).unwrap()[0];
        let total = builder.add_instruction(ListOperation::Add, Vec::new(), vec![replaced, snapshot], None).unwrap()[0];
        let frozen = builder.add_instruction(ListOperation::Freeze, Vec::new(), vec![allocation], None).unwrap()[0];
        let program = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![total, frozen], vec![Placeholder], vec![Placeholder; 2])
            .unwrap();

        // Replaying the program through the region driver rewrites every reference primitive into explicit state
        // threading, so the outputs are the values an eager reference execution would have produced.
        OBSERVED_ALLOCATION_POSITIONS.with_borrow_mut(Vec::clear);
        let context = ListDischargeContext::new(ListDestination::new());
        let input = ReferenceDischargeValue::Value(ListIrValue::List(vec![1, 2, 3, 4]));
        let regions = [program];
        let driver = RecursiveReferenceDischargeDriver::new(&regions, None);
        let outputs = driver.inline_region(&context, 0, vec![input]).unwrap();
        assert_eq!(
            outputs,
            vec![
                ReferenceDischargeValue::Value(ListIrValue::List(vec![14, 16])),
                ReferenceDischargeValue::Value(ListIrValue::List(vec![1, 7, 8, 4])),
            ],
        );

        // Every allocation the program created is gone once its `freeze` consumed it, so nothing leaks into the
        // context.
        assert_eq!(context.live_allocation_ids(), Vec::new());

        // Replaying through the driver supplies every instruction's source program location, which makes the
        // allocation selectable by a partial-discharge target.
        let observed = OBSERVED_ALLOCATION_POSITIONS.with_borrow(Vec::clone);
        assert_eq!(observed.len(), 1);
        assert!(observed[0].is_some());
    }

    #[test]
    fn test_reference_discharge_context_bind_discharged() {
        let context = ListDischargeContext::new(ListDestination::new());
        let reference_type = ReferenceType::new(ListType { length: 4 });
        let allocated = context.bind_discharged(reference_type.clone(), ListIrValue::List(vec![1, 2, 3, 4])).unwrap();
        let reference = allocated.try_as_reference("the allocated allocation").unwrap().clone();
        let allocation = reference.allocation_id();

        // A fresh allocation starts unmutated, exposes its identity alias and reference type, and carries no destination
        // reference value because it was discharged rather than preserved.
        assert_eq!(context.live_allocation_ids(), vec![allocation]);
        assert_eq!(context.is_mutated(allocation), Ok(false));
        assert_eq!(context.is_allocation_discharged(allocation), Ok(true));
        assert_eq!(context.allocation_reference(allocation), Ok(allocated.clone()));
        assert_eq!(reference.alias(), &ListAlias { offset: 0, length: 4 });
        assert_eq!(reference.r#type(), &reference_type);
        assert_eq!(reference.preserved(), None);
        assert_eq!(context.read(&reference), Ok(ListIrValue::List(vec![1, 2, 3, 4])));

        // A view keeps the allocation's identity, and its accesses act only on the portion it selects.
        let view = context
            .alias_reference(
                &reference,
                ListAlias { offset: 1, length: 2 },
                ReferenceType::new(ListType { length: 2 }),
                |_| unreachable!("the allocation is discharged"),
            )
            .unwrap();
        let view = view.try_as_reference("the view").unwrap().clone();
        assert_eq!(view.allocation_id(), allocation);
        assert_eq!(context.read(&view), Ok(ListIrValue::List(vec![2, 3])));
        assert_eq!(context.write(&view, ListIrValue::List(vec![10, 11])), Ok(()));
        assert_eq!(context.swap(&view, ListIrValue::List(vec![20, 30])), Ok(ListIrValue::List(vec![10, 11])));
        assert_eq!(context.read(&reference), Ok(ListIrValue::List(vec![1, 20, 30, 4])));
        assert_eq!(context.is_mutated(allocation), Ok(true));
        assert_eq!(context.accumulate(&view, ListIrValue::List(vec![1, 1])), Ok(()));
        assert_eq!(context.read(&reference), Ok(ListIrValue::List(vec![1, 21, 31, 4])));

        // A view remains distinct from the reference for the complete stored value even when both expose the same type.
        // Consumption and region boundaries therefore consult `is_view` rather than comparing reference types.
        let same_type_view = context
            .alias_reference(&reference, ListAlias { offset: 0, length: 4 }, reference_type.clone(), |_| {
                unreachable!("the allocation is discharged")
            })
            .unwrap();
        let same_type_view = same_type_view.try_as_reference("the same-type view").unwrap();
        assert_eq!(
            context.operand_allocation(
                &ReferenceDischargeValue::Reference(same_type_view.clone()),
                ListOperation::Call.name(),
            ),
            Err(ProgramError::MalformedProgram(format!(
                "operation `list.call` passes the view `ref<list<4>>` of {allocation} across a region boundary, which \
                 carries the complete stored value `ref<list<4>>`; create the view inside the region instead",
            ))),
        );
        assert_eq!(
            context.consume(same_type_view),
            Err(ProgramError::MalformedProgram(format!(
                "reference discharge cannot consume {allocation} through the view `ref<list<4>>`; consumption \
                 yields the complete stored value, whose referent is `list<4>`",
            ))),
        );

        // A narrower view is rejected rather than silently yielding the allocation's complete stored value under the
        // view's type.
        assert_eq!(
            context.consume(&view),
            Err(ProgramError::MalformedProgram(format!(
                "reference discharge cannot consume {allocation} through the view `ref<list<2>>`; consumption \
                 yields the complete stored value, whose referent is `list<4>`",
            ))),
        );

        // Through the complete-value handle it yields the complete state and unbinds the allocation, so every later
        // access through any handle of that allocation is reported against the exact allocation.
        assert_eq!(context.consume(&reference), Ok(ListIrValue::List(vec![1, 21, 31, 4])));
        assert_eq!(context.live_allocation_ids(), Vec::new());
        let consumed = ProgramError::MalformedProgram(format!("reference discharge accessed consumed {allocation}"));
        assert_eq!(context.read(&reference), Err(consumed.clone()));
        assert_eq!(context.set_discharged_state(allocation, ListIrValue::List(vec![0; 4]), true), Err(consumed),);

        // An allocation ID minted by an unrelated discharge is reported instead of silently addressing whichever
        // allocation occupies the same position here.
        let other = ListDischargeContext::new(ListDestination::new());
        let foreign = other.bind_discharged(reference_type, ListIrValue::List(vec![0; 4])).unwrap();
        let foreign = foreign.try_as_reference("the unrelated allocation").unwrap().allocation_id();
        let prefix =
            format!("reference discharge accessed {foreign}, which belongs to an environment other than the active");
        assert!(matches!(
            context.discharged_state(foreign),
            Err(ProgramError::MalformedProgram(message)) if message.starts_with(&prefix),
        ));
    }

    #[test]
    fn test_reference_discharge_context_bind_preserved() {
        let context = ListDischargeContext::new(ListDestination::new());
        let reference_type = ReferenceType::new(ListType { length: 2 });

        // Binding validates the destination value before inserting an allocation into the environment.
        assert_eq!(
            context.bind_preserved(reference_type.clone(), ListIrValue::List(vec![1, 2])),
            Err(ProgramError::MalformedProgram(
                "reference discharge preserved an allocation as `list<2>` which is not a reference type".to_string(),
            )),
        );
        assert_eq!(
            context.bind_preserved(
                reference_type.clone(),
                ListIrValue::Reference(ReferenceType::new(ListType { length: 1 })),
            ),
            Err(ProgramError::MalformedProgram(
                "reference discharge preserved an allocation as `ref<list<1>>` but its handle exposes `ref<list<2>>`"
                    .to_string(),
            )),
        );
        assert_eq!(context.live_allocation_ids(), Vec::new());

        let destination_reference = ListIrValue::Reference(reference_type.clone());
        let bound = context.bind_preserved(reference_type.clone(), destination_reference.clone()).unwrap();
        let reference = bound.try_as_reference("the preserved reference").unwrap().clone();
        let allocation = reference.allocation_id();

        // A preserved reference keeps its destination reference value, so a later access can replay verbatim without
        // reconstructing it.
        assert_eq!(reference.r#type(), &reference_type);
        assert_eq!(reference.preserved(), Some(&destination_reference));
        assert_eq!(context.is_allocation_discharged(allocation), Ok(false));
        assert_eq!(context.allocation_reference(allocation), Ok(bound.clone()));
        assert_eq!(
            context.operand_value(&ReferenceDischargeValue::Reference(reference.clone())),
            Ok(destination_reference.clone()),
        );

        // Every discharged-state service rejects a preserved reference by name rather than silently treating it as state.
        assert_eq!(
            context.read(&reference),
            Err(ProgramError::MalformedProgram(format!(
                "reference discharge requested the discharged state of preserved {allocation}",
            ))),
        );
        assert_eq!(
            context.is_mutated(allocation),
            Err(ProgramError::MalformedProgram(format!(
                "reference discharge queried mutation of preserved {allocation}"
            ))),
        );
        assert_eq!(
            context.set_discharged_state(allocation, ListIrValue::List(vec![0, 0]), true),
            Err(ProgramError::MalformedProgram(format!(
                "reference discharge updated the state of preserved {allocation}",
            ))),
        );

        // `alias_reference` hands the closure the preserved reference's exact destination value, so the returned view
        // cannot disagree with the allocation's preserved representation.
        let view_type = ReferenceType::new(ListType { length: 1 });
        let view_alias = ListAlias { offset: 0, length: 1 };
        let view = context
            .alias_reference(&reference, view_alias, view_type.clone(), |parent| {
                assert_eq!(parent, &ListIrValue::Reference(ReferenceType::new(ListType { length: 2 })));
                Ok(ListIrValue::Reference(view_type.clone()))
            })
            .unwrap();
        let view = view.try_as_reference("the view").unwrap();
        assert_eq!(view.allocation_id(), allocation);
        assert_eq!(view.preserved(), Some(&ListIrValue::Reference(view_type)));
    }

    #[test]
    fn test_reference_discharge_context_set_discharged_state() {
        let context = ListDischargeContext::new(ListDestination::new());
        let reference_type = ReferenceType::new(ListType { length: 2 });
        let wrong_state = ListIrValue::List(vec![1]);
        let error = ProgramError::MalformedProgram(
            "reference discharge state has type `list<1>` but allocation `ref<list<2>>` requires `list<2>`".to_string(),
        );

        // A malformed allocation is rejected before an allocation is inserted into the environment.
        assert_eq!(context.bind_discharged(reference_type.clone(), wrong_state.clone()), Err(error.clone()));
        assert_eq!(context.live_allocation_ids(), Vec::new());

        let allocated = context.bind_discharged(reference_type, ListIrValue::List(vec![1, 2])).unwrap();
        let reference = allocated.try_as_reference("the allocated allocation").unwrap();
        let allocation = reference.allocation_id();

        // Boundary reconciliation can install a successor without marking a read-only allocation as mutated.
        assert_eq!(context.set_discharged_state(allocation, ListIrValue::List(vec![3, 4]), false), Ok(()));
        assert_eq!(context.discharged_state(allocation), Ok(ListIrValue::List(vec![3, 4])));
        assert_eq!(context.is_mutated(allocation), Ok(false));

        // State updates validate before taking the mutable environment borrow, so failure preserves the prior state
        // and mutation bit.
        assert_eq!(context.set_discharged_state(allocation, wrong_state, true), Err(error));
        assert_eq!(context.read(reference), Ok(ListIrValue::List(vec![3, 4])));
        assert_eq!(context.is_mutated(allocation), Ok(false));
    }

    #[test]
    fn test_reference_discharge_context_lift() {
        // A capture-lifted program names its caller's references through constants, and such a constant denotes the
        // allocation that capture position already binds rather than a second allocation of its own.
        let pair = ReferenceType::new(ListType { length: 2 });
        let triple = ReferenceType::new(ListType { length: 3 });
        let context = ListDischargeContext::new(ListDestination::new());
        let allocated = context.bind_discharged(pair.clone(), ListIrValue::List(vec![1, 2])).unwrap();
        let allocation = allocated.try_as_reference("the captured allocation").unwrap().allocation_id();
        let scoped = context.with_captures(ReferenceDischargeCaptureScope::new(
            list_capture_position,
            vec![None, None, Some(allocation)],
        ));

        let lifted = scoped.lift(ListIrValue::Reference(pair.clone())).unwrap();
        let reference = lifted.try_as_reference("the resolved capture").unwrap();
        assert_eq!(reference.allocation_id(), allocation);
        assert_eq!(reference.r#type(), &pair);
        assert_eq!(scoped.live_allocation_ids(), vec![allocation]);

        // A non-reference constant is unaffected by the scope and lifts through the destination as usual.
        let value = scoped.lift(ListIrValue::List(vec![3, 4])).unwrap();
        assert_eq!(value, ReferenceDischargeValue::Value(ListIrValue::List(vec![3, 4])));

        // A capture position the scope does not bind keeps the unbound reference-constant rejection.
        assert_eq!(
            scoped.lift(ListIrValue::Reference(triple.clone())).err(),
            Some(ProgramError::MalformedProgram(
                "reference discharge cannot lift a constant of reference type `ref<list<3>>`; a reference enters a \
                 program through an input, a capture binding, or an allocation"
                    .to_string(),
            )),
        );

        // A capture constant names the complete stored value its position binds, so a declared type the bound
        // allocation does not carry is reported rather than silently widened where the constant is used.
        let allocated = context.bind_discharged(triple, ListIrValue::List(vec![1, 2, 3])).unwrap();
        let wider = allocated.try_as_reference("the mismatched allocation").unwrap().allocation_id();
        let mismatched = scoped.with_captures(scoped.captures().with_allocations(vec![None, None, Some(wider)]));
        assert_eq!(
            mismatched.lift(ListIrValue::Reference(pair)).err(),
            Some(ProgramError::MalformedProgram(format!(
                "reference discharge resolved a capture constant of reference type `ref<list<2>>` to {wider}, which \
                 carries the reference type `ref<list<3>>`",
            ))),
        );
    }

    #[test]
    fn test_reference_discharge_context_inline_region_consumes_a_preserved_allocation() {
        let reference_type = ReferenceType::new(ListType { length: 2 });
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let input = builder.add_input(ListIrType::Reference(reference_type.clone()));
        let frozen = builder.add_instruction(ListOperation::Freeze, Vec::new(), vec![input], None).unwrap()[0];
        let program = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![frozen], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let destination = TracingContext::<ListIrValue, ListOperation>::new();
        let context = ReferenceDischargeContext::<_, ListReferenceDischarge>::new(destination.clone());
        let destination_reference = destination.input(ListIrType::Reference(reference_type.clone()));
        let preserved = context.bind_preserved(reference_type.clone(), destination_reference).unwrap();
        let reference = preserved.try_as_reference("the preserved reference").unwrap();
        let allocation = reference.allocation_id();

        // A same-type view is not the reference for the allocation's complete stored value. Replaying the consuming
        // operation therefore leaves the allocation live and reports the invalid consumption at this seam.
        let same_type_view = context
            .alias_reference(reference, ListAlias { offset: 0, length: 2 }, reference_type, |value| Ok(value.clone()))
            .unwrap();
        assert_eq!(
            context.inline_region(program.entry_region_ref(), vec![same_type_view]),
            Err(ProgramError::MalformedProgram(format!(
                "reference discharge cannot consume {allocation} through the view `ref<list<2>>`; consumption \
                 yields the complete stored value, whose reference type is `ref<list<2>>`",
            ))),
        );
        assert_eq!(context.live_allocation_ids(), vec![allocation]);

        // The original reference does denote the complete value. Once its destination operation has been staged,
        // the allocation entry disappears so every later access is diagnosed as a use-after-consume.
        assert!(context.inline_region(program.entry_region_ref(), vec![preserved]).is_ok());
        assert!(context.live_allocation_ids().is_empty());
        assert_eq!(
            context.allocation_entry(allocation).err(),
            Some(ProgramError::MalformedProgram(format!("reference discharge accessed consumed {allocation}",))),
        );
    }

    #[test]
    fn test_reference_discharge_context_replay_preserved_access() {
        // An access whose every declared reference operand is preserved replays verbatim: the operation is bound again
        // over each handle's destination reference value. A staging destination is used because the eager destination
        // of this universe declines to execute a reference primitive, and recording is what production discharge does.
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
            let outputs = context
                .replay_preserved_access(&ListOperation::Read, std::slice::from_ref(&preserved))
                .unwrap()
                .unwrap();
            assert_eq!(outputs.len(), 1);
            vec![outputs[0].try_as_value("the replayed read result").unwrap().atom_id().unwrap()]
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

        // Applications that are not preserved-only accesses are left to their own rule: an operation without reference
        // inputs, an operation that produces a reference, and an access to a discharged allocation.
        let context = ListDischargeContext::new(ListDestination::new());
        let value = ReferenceDischargeValue::Value(ListIrValue::List(vec![1, 2]));
        assert_eq!(context.replay_preserved_access(&ListOperation::Add, &[value.clone(), value.clone()]), Ok(None));
        assert_eq!(
            context.replay_preserved_access(&ListOperation::ReferenceNew, std::slice::from_ref(&value)),
            Ok(None),
        );
        let discharged = context.bind_discharged(ReferenceType::new(referent), ListIrValue::List(vec![1, 2])).unwrap();
        assert_eq!(context.replay_preserved_access(&ListOperation::Read, std::slice::from_ref(&discharged)), Ok(None));
    }

    #[test]
    fn test_reference_discharge_context_clone() {
        let context = ListDischargeContext::new(ListDestination::new());
        let reference_type = ReferenceType::new(ListType { length: 2 });
        let allocated = context.bind_discharged(reference_type, ListIrValue::List(vec![1, 2])).unwrap();
        let reference = allocated.try_as_reference("the allocated allocation").unwrap().clone();

        // A clone shares the environment rather than copying it, which is the contract every stateful Ryft context
        // follows: several handles can denote one allocation, and every one of them must observe the same current
        // state. Isolation is therefore never implicit; a structured rule that must not commit rebuilds its region
        // against an environment of its own through `rebuild_region`.
        let clone = context.clone();
        clone.accumulate(&reference, ListIrValue::List(vec![10, 10])).unwrap();
        assert_eq!(context.read(&reference), Ok(ListIrValue::List(vec![11, 12])));
        context.accumulate(&reference, ListIrValue::List(vec![1, 1])).unwrap();
        assert_eq!(clone.read(&reference), Ok(ListIrValue::List(vec![12, 13])));
    }

    #[test]
    fn test_reference_discharge_capture_scope() {
        // A scope binds one allocation per capture position. Positions carrying a value, positions past the end of
        // the scope, and constants that name no capture position at all all resolve to nothing, which is what leaves
        // an unresolvable reference-typed constant to the rejection at the lift site.
        let context = ListDischargeContext::new(ListDestination::new());
        let allocated = context
            .bind_discharged(ReferenceType::new(ListType { length: 2 }), ListIrValue::List(vec![1, 2]))
            .unwrap();
        let allocation = allocated.try_as_reference("the captured allocation").unwrap().allocation_id();

        let empty = ReferenceDischargeCaptureScope::<ListIrValue>::default();
        assert_eq!(empty.allocations(), &[]);
        assert_eq!(empty.resolve(&ListIrValue::Reference(ReferenceType::new(ListType { length: 2 }))), None);

        let scope = ReferenceDischargeCaptureScope::new(list_capture_position, vec![None, None, Some(allocation)]);
        assert_eq!(scope.allocations(), &[None, None, Some(allocation)]);
        assert_eq!(
            scope.resolve(&ListIrValue::Reference(ReferenceType::new(ListType { length: 2 }))),
            Some(allocation)
        );
        assert_eq!(scope.resolve(&ListIrValue::Reference(ReferenceType::new(ListType { length: 1 }))), None);
        assert_eq!(scope.resolve(&ListIrValue::Reference(ReferenceType::new(ListType { length: 9 }))), None);
        assert_eq!(scope.resolve(&ListIrValue::List(vec![1, 2])), None);

        // Rebinding keeps the seam, which is how a nested region's scope and a rebuilt region's remapped scope are built.
        let rebound = scope.with_allocations(vec![Some(allocation)]);
        assert_eq!(rebound.allocations(), &[Some(allocation)]);
        assert_eq!(
            rebound.resolve(&ListIrValue::Reference(ReferenceType::new(ListType { length: 0 }))),
            Some(allocation)
        );
    }

    #[test]
    fn test_program_discharge_references() {
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let reference = builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let replacement = builder.add_input(ListIrType::List(ListType { length: 2 }));
        let previous = builder
            .add_instruction(ListOperation::Swap, Vec::new(), vec![reference, replacement], None)
            .unwrap()[0];
        let source = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![previous], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(discharged.capture_count(), 0);
        assert_eq!(discharged.output_count(), 1);
        assert_eq!(
            discharged.external_reference_bindings(),
            &[ExternalReferenceBinding::new(ReferenceSource::Input { index: 0 }, Some(1))],
        );
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:list<2>, %1:list<2> .
                let %2:list<2> = list.select %0
                    %3:list<2> = list.splice %0 %1
                in (%2, %3)"},
        );
    }

    #[test]
    fn test_program_discharge_references_rejects_consumed_external_allocations() {
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let external = builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let frozen = builder.add_instruction(ListOperation::Freeze, Vec::new(), vec![external], None).unwrap()[0];
        let source = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![frozen], vec![Placeholder], vec![Placeholder])
            .unwrap();

        assert_eq!(
            source.discharge_references(0).unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge consumed external input 0, whose state must remain owned by the caller"
                    .to_string(),
            ),
        );
    }

    #[test]
    fn test_program_discharge_references_in_capture_lifted_program() {
        // The attached region names the caller-owned reference only through a capture constant. Capture lifting moves
        // that reference into the entry input prefix while leaving the nested constant for the discharge transform to
        // resolve through its capture scope.
        let closed = closed_list_program_with_nested_reference_capture();
        let lifted = closed.to_program_with_lifted_captures().unwrap();

        let targets = lifted.reference_discharge_targets(1).unwrap();
        assert_eq!(targets, vec![ReferenceDischargeTarget::External(ReferenceSource::Capture { index: 0 })]);

        let discharged = lifted.discharge_references_in_capture_lifted_program::<ListReferenceDischarge>(1).unwrap();
        assert_eq!(discharged.capture_count(), 1);
        assert_eq!(discharged.output_count(), 1);
        assert_eq!(
            discharged.external_reference_bindings(),
            &[ExternalReferenceBinding::new(ReferenceSource::Capture { index: 0 }, None)],
        );
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:list<2> .
                let %1:list<2> = list.call %0 [
                    callee={
                        lambda %0:list<2> .
                        let %1:list<2> = list.select %0
                        in (%1)
                    },
                ]
                in (%1)"},
        );
    }

    #[test]
    fn test_program_partially_discharge_references() {
        // The kernel-pipeline shape, in a universe that mentions no arrays: one caller-owned allocation is selected
        // and becomes threaded state, while the other survives as a reference the rewritten program still accesses
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

        let targets = source.reference_discharge_targets(0).unwrap();
        assert_eq!(
            targets,
            vec![
                ReferenceDischargeTarget::External(ReferenceSource::Input { index: 0 }),
                ReferenceDischargeTarget::External(ReferenceSource::Input { index: 1 }),
            ],
        );
        let discharged = source.partially_discharge_references(0, &targets[..1]);
        let discharged = discharged.unwrap();

        // The selected allocation became a state value input at its own boundary position and publishes its final
        // state as a hidden output; the preserved reference kept its reference type, so it binds nothing at all.
        assert_eq!(discharged.output_count(), 1);
        assert_eq!(
            discharged.external_reference_bindings(),
            &[ExternalReferenceBinding::new(ReferenceSource::Input { index: 0 }, Some(1))],
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
            ReferenceDischargeResult::try_from(discharged).unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge program still contains a reference-typed value and cannot form a full discharge"
                    .to_string(),
            ),
        );
    }

    #[test]
    fn test_program_partially_discharge_references_preserves_an_unselected_internal_target() {
        // An interior allocation is selectable in its own right, so a program can normalize its pipeline state while
        // the allocation a kernel body addresses is allocated, viewed, accessed, and consumed as a reference
        // throughout.
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let initial = builder.add_input(ListIrType::List(ListType { length: 4 }));
        let update = builder.add_input(ListIrType::List(ListType { length: 2 }));
        let allocation =
            builder.add_instruction(ListOperation::ReferenceNew, Vec::new(), vec![initial], None).unwrap()[0];
        let view = builder
            .add_instruction(ListOperation::Slice { offset: 1, length: 2 }, Vec::new(), vec![allocation], None)
            .unwrap()[0];
        builder.add_instruction(ListOperation::AddUpdate, Vec::new(), vec![view, update], None).unwrap();
        let frozen = builder.add_instruction(ListOperation::Freeze, Vec::new(), vec![allocation], None).unwrap()[0];
        let source = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![frozen], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        // Selecting nothing preserves the allocation, so the whole reference language survives: the view operation is
        // replayed too, and the resulting view consumes the exact reference produced by that replay rather than
        // replaying the chain again at the access.
        let discharged = source.clone().partially_discharge_references(0, &[]);
        let discharged = discharged.unwrap();
        assert_eq!(discharged.output_count(), 1);
        assert_eq!(discharged.external_reference_bindings(), &[]);
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
        let targets = source.reference_discharge_targets(0).unwrap();
        assert_eq!(
            targets,
            vec![ReferenceDischargeTarget::Internal {
                instruction: InstructionId::new(source.entry_region_ref().id(), 0),
                output_index: 0,
            }],
        );
        let selected = source.clone().partially_discharge_references(0, targets.as_slice());
        let selected = ReferenceDischargeResult::try_from(selected.unwrap()).unwrap();
        let full = source.discharge_references(0).unwrap();
        assert_eq!(selected.program().to_string(), full.program().to_string());
    }

    #[test]
    fn test_program_partially_discharge_references_allows_consuming_a_preserved_external_allocation() {
        // A preserved reference has no external binding to describe: the program keeps the consuming operation, and
        // the caller passes its reference to that operation directly, so partial discharge preserves the source
        // program's consumption.
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let external = builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let frozen = builder.add_instruction(ListOperation::Freeze, Vec::new(), vec![external], None).unwrap()[0];
        let source = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![frozen], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let preserved = source.partially_discharge_references(0, &[]).unwrap();
        assert_eq!(preserved.external_reference_bindings(), &[]);
        assert_eq!(
            preserved.program().to_string(),
            indoc! {"
                lambda %0:ref<list<2>> .
                let %1:list<2> = list.freeze %0
                in (%1)"},
        );
        // Returning the allocation afterwards is a use of it like any other, so the consumed allocation is reported at
        // the output that names it rather than published as a stale reference.
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

        // An allocation rendering embeds the identity of the environment that minted it, which is process-global,
        // so the assertion pins everything around it.
        let error = source.partially_discharge_references(0, &[]).unwrap_err();
        let ProgramError::MalformedProgram(message) = &error else {
            panic!("expected a malformed-program rejection but got {error:?}");
        };
        assert!(message.starts_with("reference discharge accessed consumed reference allocation "), "{message}");
        assert!(message.ends_with(":0"), "{message}");
    }

    #[test]
    fn test_program_partially_discharge_references_threads_a_preserved_allocation_beside_discharged_state() {
        // A structured boundary carries both kinds of allocation at once: a discharged carry crosses as immutable
        // state and is widened with a published successor, while a preserved carry crosses as the reference it already
        // is, at its own declared operand position, and widens nothing at all.
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

        let targets = source.reference_discharge_targets(0).unwrap();
        let discharged = source.partially_discharge_references(0, &targets[..1]).unwrap();

        // The selected allocation's entering state occupies its own operand position and its successor is appended as
        // a published output; the preserved reference's operand position still carries a reference, and the rebuilt
        // callee performs the read on it exactly as the source did.
        assert_eq!(discharged.output_count(), 1);
        assert_eq!(
            discharged.external_reference_bindings(),
            &[ExternalReferenceBinding::new(ReferenceSource::Input { index: 0 }, Some(1))],
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
    fn test_program_partially_discharge_references_replays_preserved_accesses_inside_their_source_provenance() {
        // The dispatch path replays a preserved-allocation access itself, before any rule runs, and that replay must
        // still happen inside the source instruction's recorded origin. Provenance renders only under `WithProvenance`,
        // so no semantic rendering can catch an unwrapped replay dropping it.
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let reference = builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let update = builder.add_input(ListIrType::List(ListType { length: 2 }));
        let origin = |name: &str| Some(Provenance::scope(ProvenanceScope::new(name), Provenance::unknown()));
        builder
            .add_instruction(ListOperation::Write, Vec::new(), vec![reference, update], origin("probe_write"))
            .unwrap();
        let observed = builder
            .add_instruction(ListOperation::Read, Vec::new(), vec![reference], origin("probe_read"))
            .unwrap()[0];
        let source = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![observed], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let preserved = source.partially_discharge_references(0, &[]).unwrap();
        let rendered = std::fmt::from_fn(|formatter| {
            preserved.program().render(formatter, 0, ProgramRenderingMode::WithProvenance)
        })
        .to_string();
        assert!(rendered.contains("; probe_write"), "write provenance lost:\n{rendered}");
        assert!(rendered.contains("; probe_read"), "read provenance lost:\n{rendered}");
    }

    #[test]
    fn test_program_partially_discharge_references_validates_targets_against_the_program() {
        // The targets are checked before anything is replayed, so a target this program does not expose is reported
        // against the program rather than surfacing later as an allocation that never appeared.
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let external = builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let observed = builder.add_instruction(ListOperation::Read, Vec::new(), vec![external], None).unwrap()[0];
        let source = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![observed], vec![Placeholder], vec![Placeholder])
            .unwrap();

        assert_eq!(
            source
                .partially_discharge_references(
                    0,
                    &[ReferenceDischargeTarget::External(ReferenceSource::Input { index: 3 })],
                )
                .unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge targets include external input 3 which is not selectable in this program"
                    .to_string(),
            ),
        );
    }

    #[test]
    fn test_program_partially_discharge_references_in_capture_lifted_program() {
        // Selecting nothing preserves the capture as a reference and explicitly threads it into the nested region.
        let closed = closed_list_program_with_nested_reference_capture();
        let lifted = closed.to_program_with_lifted_captures().unwrap();
        let discharged = lifted
            .partially_discharge_references_in_capture_lifted_program::<ListReferenceDischarge>(1, &[])
            .unwrap();

        assert_eq!(discharged.capture_count(), 1);
        assert_eq!(discharged.output_count(), 1);
        assert_eq!(discharged.external_reference_bindings(), &[]);
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:ref<list<2>> .
                let %1:list<2> = list.call %0 [
                    callee={
                        lambda %0:ref<list<2>> .
                        let %1:list<2> = list.read %0
                        in (%1)
                    },
                ]
                in (%1)"},
        );
    }

    #[test]
    fn test_closed_program_discharge_references() {
        let closed = closed_list_program_with_nested_reference_capture();
        let discharged = closed.discharge_references::<ListReferenceDischarge>().unwrap();

        assert_eq!(discharged.capture_count(), 1);
        assert_eq!(discharged.output_count(), 1);
        assert_eq!(
            discharged.external_reference_bindings(),
            &[ExternalReferenceBinding::new(ReferenceSource::Capture { index: 0 }, None)],
        );
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:list<2> .
                let %1:list<2> = list.call %0 [
                    callee={
                        lambda %0:list<2> .
                        let %1:list<2> = list.select %0
                        in (%1)
                    },
                ]
                in (%1)"},
        );
    }

    #[test]
    fn test_discharge_reference_free_operation() {
        // Value operands replay verbatim through the destination, which executes the operation eagerly here.
        let context = ListDischargeContext::new(ListDestination::new());
        let inputs = vec![
            ReferenceDischargeValue::Value(ListIrValue::List(vec![1, 2])),
            ReferenceDischargeValue::Value(ListIrValue::List(vec![10, 20])),
        ];
        assert_eq!(
            discharge_reference_free_operation(&ListOperation::Add, &context, &EmptyRegionDriver, inputs.as_slice()),
            Ok(vec![ReferenceDischargeValue::Value(ListIrValue::List(vec![11, 22]))]),
        );

        // A reference operand is rejected, because an operation that receives a reference owns its own rule.
        let allocated = context
            .bind_discharged(ReferenceType::new(ListType { length: 2 }), ListIrValue::List(vec![1, 2]))
            .unwrap();
        let allocation = allocated.try_as_reference("the allocated allocation").unwrap().allocation_id();
        assert_eq!(
            discharge_reference_free_operation(
                &ListOperation::Add,
                &context,
                &EmptyRegionDriver,
                &[allocated, inputs[1].clone()],
            ),
            Err(ProgramError::MalformedProgram(format!(
                "reference discharge expected a value operand 0 of `list.add` but received {allocation} ref<list<2>>",
            ))),
        );
    }

    #[test]
    fn test_discharge_reference_free_operation_rejects_regions_that_touch_references() {
        // A region-carrying application whose closure touches a reference is rejected rather than replayed,
        // because how a reference boundary widens is knowledge that belongs to the operation.
        let context = ListDischargeContext::new(ListDestination::new());
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let input = builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 1 })));
        let read = builder.add_instruction(ListOperation::Read, Vec::new(), vec![input], None).unwrap()[0];
        let stateful = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![read], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let regions = [stateful];
        let driver = RecursiveReferenceDischargeDriver::new(&regions, None);
        let inputs = [
            ReferenceDischargeValue::Value(ListIrValue::List(vec![1, 2])),
            ReferenceDischargeValue::Value(ListIrValue::List(vec![10, 20])),
        ];
        assert_eq!(
            discharge_reference_free_operation(&ListOperation::Add, &context, &driver, &inputs),
            Err(ProgramError::UnsupportedOperation {
                message: "`list.add` carries reference state but has no reference discharge rule".to_string(),
            }),
        );
    }

    #[test]
    fn test_discharge_reference_free_operation_replays_reference_free_regions() {
        // An operation that declares a region slot replays its region into the destination as it stands, which is the
        // complete rewrite for a region-carrying operation whose closure holds no state to thread.
        let context = ListDischargeContext::new(ListDestination::new());
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
        let inputs = [
            ReferenceDischargeValue::Value(ListIrValue::List(vec![1, 2])),
            ReferenceDischargeValue::Value(ListIrValue::List(vec![10, 20])),
        ];
        assert_eq!(
            discharge_reference_free_operation(&ListOperation::Call, &context, &driver, &inputs[..1]),
            Ok(vec![ReferenceDischargeValue::Value(ListIrValue::List(vec![2, 4]))]),
        );

        // The operation's own contract still governs the replayed regions: `list.add` declares no region slots.
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let input = builder.add_input(ListIrType::List(ListType { length: 1 }));
        let reference_free = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let regions = [reference_free];
        let driver = RecursiveReferenceDischargeDriver::new(&regions, None);
        assert_eq!(
            discharge_reference_free_operation(&ListOperation::Add, &context, &driver, &inputs),
            Err(ProgramError::MalformedProgram(
                "operation `list.add` declares no region slots but 1 regions were attached".to_string(),
            )),
        );
    }

    #[test]
    fn test_discharge_positional_region_operation() {
        // The callee writes the forwarded reference and returns a snapshot of it. The forwarded allocation is
        // discharged, so the rewritten call publishes its final state through an appended output, which the rule merges
        // back into the caller's allocation while reporting only the declared snapshot output.
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let reference = builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let update = builder.add_input(ListIrType::List(ListType { length: 2 }));
        builder.add_instruction(ListOperation::Write, Vec::new(), vec![reference, update], None).unwrap();
        let snapshot = builder.add_instruction(ListOperation::Read, Vec::new(), vec![reference], None).unwrap()[0];
        let callee = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![snapshot], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let regions = [callee];
        let driver = RecursiveReferenceDischargeDriver::new(&regions, None);

        let context = ListDischargeContext::new(ListDestination::new());
        let allocated = context
            .bind_discharged(ReferenceType::new(ListType { length: 2 }), ListIrValue::List(vec![1, 2]))
            .unwrap();
        let forwarded = allocated.try_as_reference("the forwarded allocation").unwrap();
        let allocation = forwarded.allocation_id();
        let inputs = [allocated.clone(), ReferenceDischargeValue::Value(ListIrValue::List(vec![7, 8]))];
        assert_eq!(
            discharge_positional_region_operation(&ListOperation::Call, &context, &driver, &inputs, 0),
            Ok(vec![ReferenceDischargeValue::Value(ListIrValue::List(vec![7, 8]))]),
        );
        assert_eq!(context.read(forwarded), Ok(ListIrValue::List(vec![7, 8])));
        assert_eq!(context.is_mutated(allocation), Ok(true));

        // Fewer inputs than the leading count, or a reference among the leading inputs, are malformed.
        assert_eq!(
            discharge_positional_region_operation(&ListOperation::Call, &context, &driver, &inputs, 3),
            Err(ProgramError::MalformedProgram(
                "operation `list.call` forwards its inputs after 3 leading inputs but the application has 2 inputs"
                    .to_string(),
            )),
        );
        assert_eq!(
            discharge_positional_region_operation(&ListOperation::Call, &context, &driver, &inputs, 1),
            Err(ProgramError::MalformedProgram(format!(
                "reference discharge expected a value leading input 0 of `list.call` but received {allocation} \
                 ref<list<2>>",
            ))),
        );
    }

    #[test]
    fn test_discharge_positional_region_operation_recovers_a_returned_capture_scoped_allocation() {
        // This allocation reaches the region through its inherited capture scope, not through any forwarded operand.
        // The declared result must therefore be recovered from the context rather than from the empty operand list.
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let captured = builder.add_constant(ListIrValue::Reference(ReferenceType::new(ListType { length: 2 })));
        let program = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![captured], Vec::new(), vec![Placeholder])
            .unwrap();

        let context = ListDischargeContext::new(ListDestination::new());
        let allocated = context
            .bind_discharged(ReferenceType::new(ListType { length: 2 }), ListIrValue::List(vec![1, 2]))
            .unwrap();
        let allocation = allocated.try_as_reference("the capture-scoped allocation").unwrap().allocation_id();
        let context = context.with_captures(ReferenceDischargeCaptureScope::new(
            list_capture_position,
            vec![None, None, Some(allocation)],
        ));
        let regions = [program];
        let driver = RecursiveReferenceDischargeDriver::new(&regions, None);

        let results = discharge_positional_region_operation(&ListOperation::Call, &context, &driver, &[], 0).unwrap();
        assert_eq!(results.len(), 1);
        let returned = results[0].try_as_reference("the returned capture-scoped allocation").unwrap();
        assert_eq!(returned.allocation_id(), allocation);
        assert_eq!(context.read(returned), Ok(ListIrValue::List(vec![1, 2])));
        assert_eq!(context.is_mutated(allocation), Ok(false));
    }

    #[test]
    fn test_discharge_positional_region_operation_recovers_a_returned_preserved_capture_scoped_allocation() {
        // Same as above, but the capture-scoped allocation is preserved, so the returned handle denotes the destination
        // reference itself and there is no state to merge.
        let reference_type = ReferenceType::new(ListType { length: 2 });
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let captured = builder.add_constant(ListIrValue::Reference(reference_type.clone()));
        let program = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![captured], Vec::new(), vec![Placeholder])
            .unwrap();

        let context = ListDischargeContext::new(ListDestination::new());
        let destination_reference = ListIrValue::Reference(reference_type.clone());
        let preserved = context.bind_preserved(reference_type, destination_reference.clone()).unwrap();
        let allocation = preserved.try_as_reference("the preserved capture-scoped allocation").unwrap().allocation_id();
        let context = context.with_captures(ReferenceDischargeCaptureScope::new(
            list_capture_position,
            vec![None, None, Some(allocation)],
        ));
        let regions = [program];
        let driver = RecursiveReferenceDischargeDriver::new(&regions, None);

        let results = discharge_positional_region_operation(&ListOperation::Call, &context, &driver, &[], 0).unwrap();
        assert_eq!(results.len(), 1);
        let returned = results[0].try_as_reference("the returned preserved capture-scoped allocation").unwrap();
        assert_eq!(returned.allocation_id(), allocation);
        assert_eq!(returned.preserved(), Some(&destination_reference));
        assert_eq!(context.operand_value(&results[0]), Ok(destination_reference));
    }

    #[test]
    fn test_discharge_positional_region_operation_preserves_aliasing_between_repeated_declared_allocations() {
        // Both declared callee inputs denote one caller allocation. A write through the first must therefore be
        // visible to a read through the second even though the rebuilt boundary retains both declared positions.
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

        // Full discharge turns the shared allocation into state. The public snapshot and hidden final-state output
        // both observe the write, proving that the duplicate boundary position did not mint an independent region
        // allocation.
        let discharged = source.clone().discharge_references(0).unwrap();
        assert_eq!(
            discharged.program().interpret(vec![ListIrValue::List(vec![1, 2])]),
            Ok(vec![ListIrValue::List(vec![7, 8]), ListIrValue::List(vec![7, 8])]),
        );

        // Partial discharge preserves the same alias as a reference. Both declared positions remain present in the
        // callee boundary, but its second input is unused and both accesses replay through the first canonical value.
        let preserved = source.partially_discharge_references(0, &[]).unwrap();
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
    fn test_discharge_positional_region_operation_threads_state_through_a_callee() {
        // The whole structured rewrite is universe-generic, so the prototype universe exercises it end to end. A callee
        // mutates the allocation it receives and returns only the previous snapshot, and discharge widens the call
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
        let allocation =
            builder.add_instruction(ListOperation::ReferenceNew, Vec::new(), vec![initial], None).unwrap()[0];
        let previous =
            builder.add_instruction(ListOperation::Call, vec![callee], vec![allocation, update], None).unwrap()[0];
        let frozen = builder.add_instruction(ListOperation::Freeze, Vec::new(), vec![allocation], None).unwrap()[0];
        let source = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(
                vec![previous, frozen],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();

        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(discharged.output_count(), 2);
        assert_eq!(discharged.external_reference_bindings(), &[]);
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
}
