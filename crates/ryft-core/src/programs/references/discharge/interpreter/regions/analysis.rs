use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use crate::contexts::Domain;
use crate::macros::check_count;
use crate::programs::ProgramError;
use crate::programs::atoms::{Atom, AtomId};
use crate::programs::instructions::Instruction;
use crate::programs::operations::Operation;
use crate::programs::references::discharge::policies::ReferenceDischargePolicy;
use crate::programs::references::discharge::transform::{
    ReferenceDischargeAllocationId, ReferenceDischargeBinding, ReferenceDischargeCaptureScope,
    ReferenceDischargeContext, ReferenceDischargeValue,
};
use crate::programs::references::semantics::{ReferenceAccessMode, ReferenceOutput};
use crate::programs::regions::{RegionId, RegionRef};
use crate::programs::types::{Type, Typed};
use crate::programs::values::Value;

use super::boundaries::ReferenceStateWidening;

// TODO(eaplatanios): Review this module.

/// Exact non-consuming access modes recorded for one caller allocation.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
struct ReferenceAccessModes {
    /// Whether the allocation is read.
    read: bool,

    /// Whether the allocation is written without observing its selected previous state.
    write: bool,

    /// Whether the allocation is read and replaced.
    read_write: bool,

    /// Whether the allocation receives an ordered additive update.
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

/// Transitive reference-access summary of one region closure, expressed in the caller allocations its boundary names.
///
/// This is the analysis a structured rule needs *before* it can size its state boundary, and it is computed entirely
/// from generic hooks: operation-local [`Operation::reference_semantics`], the input- and output-region provenance
/// hooks, reference-output identity, and recursive summaries of nested regions. Allocations allocated inside the closure are
/// deliberately absent: they belong to no caller and cross no boundary.
///
/// The summary separates *reachability* from *semantic access*. The reached set holds every caller allocation
/// the closure's replay must be able to resolve — including a capture constant that is only rematerialized and passed
/// along — and is what sizes the state boundary through
/// [`threaded_state_allocations`](ReferenceDischargeContext::threaded_state_allocations). [`accessed`](Self::accessed) and
/// [`access_modes`](Self::access_modes) hold only the allocations the closure semantically accesses, which is what region
/// access policies validate. Reading `accessed` to size a boundary under-threads merely-forwarded captures.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReferenceRegionSummary {
    /// Every caller allocation the closure must be able to resolve while replaying, whether or not it is semantically
    /// accessed.
    reached: BTreeSet<ReferenceDischargeAllocationId>,

    /// Every caller allocation the closure accesses, mapped to its exact non-consuming access modes.
    accesses: BTreeMap<ReferenceDischargeAllocationId, ReferenceAccessModes>,

    /// Caller allocation each *declared* region output denotes, or [`None`] where the output is a value.
    pub(super) output_allocations: Vec<Option<ReferenceDischargeAllocationId>>,
}

impl ReferenceRegionSummary {
    /// Returns every caller allocation the closure must be able to resolve, in canonical allocation order.
    #[inline]
    pub(super) fn reached(&self) -> impl Iterator<Item = ReferenceDischargeAllocationId> + '_ {
        self.reached.iter().copied()
    }

    /// Returns every caller allocation the closure accesses, in canonical allocation order.
    #[inline]
    pub fn accessed(&self) -> impl Iterator<Item = ReferenceDischargeAllocationId> + '_ {
        self.accesses.keys().copied()
    }

    /// Returns the exact access modes recorded for `allocation`, in semantic order.
    pub fn access_modes(
        &self,
        allocation: ReferenceDischargeAllocationId,
    ) -> impl Iterator<Item = ReferenceAccessMode> {
        self.accesses.get(&allocation).copied().unwrap_or_default().iter()
    }

    /// Returns whether `mode` is among the closure's recorded access modes for `allocation`.
    #[inline]
    pub fn has_access(&self, allocation: ReferenceDischargeAllocationId, mode: ReferenceAccessMode) -> bool {
        self.access_modes(allocation).any(|recorded| recorded == mode)
    }

    /// Returns the caller allocation each declared region output denotes, or [`None`] where the output is a value.
    ///
    /// A region that returns an allocation already publishes that allocation's final state at its own output position, so a rule
    /// that widens the boundary must not publish it a second time.
    #[inline]
    pub fn output_allocations(&self) -> &[Option<ReferenceDischargeAllocationId>] {
        self.output_allocations.as_slice()
    }

    /// Returns whether any statically reachable path through the closure writes or accumulates into `allocation`. An allocation the
    /// closure only reads is *not* mutated, which is the fact read-only pruning consults.
    ///
    /// This classification is intentionally conservative across structured control flow: a write in either branch or
    /// in a loop body marks the allocation as mutated even when one execution takes the other branch or performs zero
    /// iterations. Discharge therefore threads and publishes a hidden final state for every such allocation; at runtime that
    /// state is simply unchanged when the mutating path does not execute.
    #[inline]
    pub fn is_mutated(&self, allocation: ReferenceDischargeAllocationId) -> bool {
        self.access_modes(allocation).any(|mode| {
            matches!(
                mode,
                ReferenceAccessMode::Write | ReferenceAccessMode::ReadWrite | ReferenceAccessMode::Accumulate,
            )
        })
    }

    /// Returns the summary of the two closures taken together, which is what an operation with several attached
    /// regions threads through one shared *state* boundary.
    ///
    /// The reached allocations and the accesses are both merged, so an allocation that only one nested closure returns or
    /// rematerializes stays reachable — and therefore threaded — at the merged level. Declared output allocations belong to
    /// one region's own boundary rather than to the shared state, so the merged summary keeps the receiver's; an
    /// operation whose regions must agree on them, such as a condition, has that agreement checked against the
    /// rebuilt regions themselves.
    pub fn merged(mut self, other: &Self) -> Self {
        self.absorb(other);
        self
    }

    /// Merges another closure's reached allocations and accesses into this summary in place, leaving the declared output
    /// allocations alone.
    fn absorb(&mut self, other: &Self) {
        self.reached.extend(other.reached.iter().copied());
        for (allocation, modes) in &other.accesses {
            let entry = self.accesses.entry(*allocation).or_default();
            for mode in modes.iter() {
                entry.insert(mode);
            }
        }
    }

    /// Records one access, or rejects a consuming access to a caller allocation.
    ///
    /// # Parameters
    ///
    ///   - `allocation`: Caller allocation being accessed.
    ///   - `mode`: Semantic mode of the access.
    ///   - `operation`: Name of the accessing operation, used in the consumption diagnostic.
    pub(super) fn record(
        &mut self,
        allocation: ReferenceDischargeAllocationId,
        mode: ReferenceAccessMode,
        operation: &str,
    ) -> Result<(), ProgramError> {
        // A consumed allocation has no successor, so no symmetric boundary and no final-state output can describe what
        // happened to it, and an allocation that survives as a reference fares no better: whether a region consumed it can
        // depend on which branch ran, which the caller's environment cannot represent.
        if mode.is_consuming() {
            return Err(ProgramError::MalformedProgram(format!(
                "reference discharge cannot pass {allocation} into a region that consumes it through `{operation}`",
            )));
        }
        self.reached.insert(allocation);
        self.accesses.entry(allocation).or_default().insert(mode);
        Ok(())
    }
}

/// Returns the capture scope one attached region discharges under.
///
/// A region whose operation declares a leading capture prefix establishes a fresh scope over the allocations that prefix
/// binds, exactly as a called program's captures shadow its caller's; every other region inherits the scope of the
/// region it is attached in. This is the interpreter's counterpart of the scope propagation the standalone reference
/// analysis performs over the whole arena, computed one boundary at a time because that is where the interpreter
/// already resolves allocations.
///
/// # Parameters
///
///   - `capture_input_count`: Length of the region's own leading capture prefix, from
///     [`Operation::region_capture_input_count`], or [`None`] when the region inherits its parent's scope.
///   - `inputs`: Allocation each declared region input binds, in boundary order.
///   - `inherited`: Capture scope of the region this one is attached in.
///   - `region`: Identity of the region, used in the diagnostic.
///
/// # Errors
///
/// Returns [`ProgramError::MalformedProgram`] when the declared capture prefix is longer than the region's boundary.
pub(super) fn nested_capture_scope<Constant>(
    capture_input_count: Option<usize>,
    inputs: &[Option<ReferenceDischargeAllocationId>],
    inherited: &ReferenceDischargeCaptureScope<Constant>,
    region: RegionId,
) -> Result<ReferenceDischargeCaptureScope<Constant>, ProgramError> {
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
    Ok(inherited.with_allocations(inputs[..count].to_vec()))
}

/// Accumulates the transitive reference accesses of one region closure into `summary` and returns the caller allocation
/// each of the region's declared outputs denotes.
///
/// The traversal maps each reference-typed atom of the region onto the caller allocation it denotes, or onto [`None`] when
/// the allocation was allocated inside the closure and therefore crosses no boundary. Nested regions are entered through
/// [`Operation::input_region_provenance`], and a structured operation's reference-typed output is resolved either by
/// [`Operation::reference_output_identity_input`], which states outright which operand's allocation it preserves, or by
/// [`Operation::output_region_provenance`], which names the region output it forwards.
///
/// A reference-typed *constant* is resolved through `captures` and seeded exactly like a boundary position, because a
/// capture-lifted program names its caller's references that way. That is what lets a structured rule discover that
/// its closure reaches an allocation its operands never named, and therefore what makes synthesized state carries reachable.
///
/// # Parameters
///
///   - `region`: Region whose closure is summarized.
///   - `inputs`: Caller allocation denoted by each declared region input, in boundary order.
///   - `captures`: Capture scope this region discharges under.
///   - `summary`: Summary being accumulated.
///
/// # Errors
///
/// Returns [`ProgramError::MalformedProgram`] when `inputs` does not describe the region's boundary, when the closure
/// reaches a reference that entered neither through the boundary nor through the capture scope, when a nested
/// reference-typed boundary position declares no provenance to follow, when an operation's own contract forbids the
/// access mode its closure performs, or when the closure consumes a caller allocation.
fn summarize_region_closure<V: Value, O: Operation<Type = V::Type>>(
    region: RegionRef<'_, V, O>,
    inputs: &[Option<ReferenceDischargeAllocationId>],
    captures: &ReferenceDischargeCaptureScope<V>,
    summary: &mut ReferenceRegionSummary,
) -> Result<Vec<Option<ReferenceDischargeAllocationId>>, ProgramError> {
    check_count!("input", inputs, region.input_ids().len(), ProgramError);
    let is_reference = |atom: AtomId| region.atoms()[atom.index()].r#type().is_reference();
    let mut allocations = HashMap::<AtomId, Option<ReferenceDischargeAllocationId>>::new();
    for (input, allocation) in region.input_ids().iter().copied().zip(inputs) {
        if is_reference(input) {
            allocations.insert(input, *allocation);
        }
    }

    // A capture-scoped constant is seeded exactly like a boundary position. Materializing one makes its allocation reachable
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
            && let Some(allocation) = captures.resolve(constant)
        {
            allocations.insert(atom_id, Some(allocation));
            if materialized_atoms.contains(&atom_id) {
                summary.reached.insert(allocation);
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
    // its boundary nor through its capture scope. The environment has no allocation for it, so the summary reports it here
    // rather than dropping the access and letting the replay fail later for a reason that no longer names the
    // operation that performed it.
    let resolve =
        |allocations: &HashMap<AtomId, Option<ReferenceDischargeAllocationId>>, atom: AtomId, operation: &str| {
            match allocations.get(&atom) {
                Some(allocation) => Ok(*allocation),
                None if is_reference(atom) => Err(ProgramError::MalformedProgram(format!(
                    "operation `{operation}` reaches a reference that entered region `{}` neither through its boundary \
                 nor through its capture scope",
                    region.id(),
                ))),
                None => Ok(None),
            }
        };
    for instruction in region.instructions() {
        let operation = instruction.operation();
        let semantics = operation.reference_semantics();
        for access in semantics.inputs() {
            let accessed = operand(instruction, access.input_index(), "an accessed")?;
            if let Some(allocation) = resolve(&allocations, accessed, operation.name())? {
                summary.record(allocation, access.mode(), operation.name())?;
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
            let allocation = match output {
                ReferenceOutput::Allocation { .. } => None,
                ReferenceOutput::Alias { input_index, .. } => {
                    resolve(&allocations, operand(instruction, *input_index, "an aliased")?, operation.name())?
                }
            };
            allocations.insert(defined, allocation);
        }
        let mut attached_output_allocations = Vec::with_capacity(instruction.regions().len());
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
                    resolve(&allocations, operand(instruction, operand_index, "a region")?, operation.name())
                })
                .collect::<Result<Vec<_>, _>>()?;

            // The nested closure is summarized on its own first, so that an operation restricting what its regions may
            // do to an entering allocation is held to that restriction here, where the offending region is still named,
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
            attached_output_allocations.push(nested_outputs);
        }

        // A reference-typed output of a region-carrying operation preserves an allocation rather than classifying one, so it
        // resolves through the generic hooks that state where it came from: an explicit operand identity when the
        // operation declares one, and otherwise the region output it forwards.
        for (output_index, output) in instruction.outputs().iter().copied().enumerate() {
            if !is_reference(output) || allocations.contains_key(&output) {
                continue;
            }
            let preserved = match operation.reference_output_identity_input(output_index) {
                Some(input_index) => {
                    resolve(&allocations, operand(instruction, input_index, "a preserved")?, operation.name())?
                }
                None => forwarded_output_allocation(operation, output_index, attached_output_allocations.as_slice())?,
            };
            allocations.insert(output, preserved);
        }
    }
    let output_allocations = region
        .output_ids()
        .iter()
        .copied()
        .map(|output| if is_reference(output) { allocations.get(&output).copied().flatten() } else { None })
        .collect::<Vec<_>>();
    summary.reached.extend(output_allocations.iter().copied().flatten());
    Ok(output_allocations)
}

/// Validates every exact access mode in `summary` against one attached-region policy.
fn validate_region_accesses<O: Operation>(
    operation: &O,
    region_index: usize,
    summary: &ReferenceRegionSummary,
) -> Result<(), ProgramError> {
    for allocation in summary.accessed() {
        for mode in summary.access_modes(allocation) {
            if !operation.allows_reference_access_through_region_input(region_index, mode) {
                return Err(ProgramError::MalformedProgram(format!(
                    "operation `{}` does not allow region {region_index} to access {allocation} with mode `{mode}`",
                    operation.name(),
                )));
            }
        }
    }
    Ok(())
}

/// Returns the caller allocation one region-carrying operation's reference-typed output forwards out of its attached
/// regions, requiring every region that contributes to that output to agree on it.
///
/// # Parameters
///
///   - `operation`: Operation producing the output.
///   - `output_index`: Output position being resolved.
///   - `attached_output_allocations`: Allocation each attached region's declared outputs denote, in region order.
fn forwarded_output_allocation<O: Operation>(
    operation: &O,
    output_index: usize,
    attached_output_allocations: &[Vec<Option<ReferenceDischargeAllocationId>>],
) -> Result<Option<ReferenceDischargeAllocationId>, ProgramError> {
    let provenance = operation.output_region_provenance(output_index);
    if provenance.is_empty() {
        return Err(ProgramError::MalformedProgram(format!(
            "operation `{}` produces a reference at output {output_index} without declaring which operand allocation it \
             preserves or which region output it forwards",
            operation.name()
        )));
    }
    let mut forwarded = None;
    for (position, origin) in provenance.iter().enumerate() {
        let allocation = attached_output_allocations
            .get(origin.region_index)
            .and_then(|allocations| allocations.get(origin.output_index).copied())
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
            forwarded = allocation;
        } else if forwarded != allocation {
            return Err(ProgramError::MalformedProgram(format!(
                "operation `{}` forwards output {output_index} from regions that return different reference allocations",
                operation.name(),
            )));
        }
    }
    Ok(forwarded)
}
impl<C: Domain, P: ReferenceDischargePolicy<C>> ReferenceDischargeContext<C, P> {
    /// Summarizes the transitive reference accesses of one region closure, in the terms of the caller allocations its
    /// boundary names.
    ///
    /// A structured rule calls this before it can size its state boundary: which allocations a region closure touches, and
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
    ///   - `inputs`: Caller allocation denoted by each of the region's declared inputs, in boundary order, with [`None`]
    ///     wherever the position carries a value.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when `inputs` does not describe the region's boundary, when the
    /// operation declares a capture prefix longer than the region's boundary, when a reference-typed nested boundary
    /// position declares no provenance the summary could follow, when the closure reaches a reference that entered
    /// neither through its boundary nor through its capture scope, or when the closure consumes a caller allocation, which
    /// no state boundary can express. It also returns this error when `operation` does not permit one of the exact
    /// access modes the closure performs through `region_index`.
    pub fn region_summary<O: Operation>(
        &self,
        operation: &O,
        region_index: usize,
        region: RegionRef<'_, C::Constant, C::Operation>,
        inputs: &[Option<ReferenceDischargeAllocationId>],
    ) -> Result<ReferenceRegionSummary, ProgramError> {
        let captures = nested_capture_scope(
            operation.region_capture_input_count(region_index),
            inputs,
            self.captures(),
            region.id(),
        )?;
        let mut summary = ReferenceRegionSummary::default();
        summary.output_allocations = summarize_region_closure(region, inputs, &captures, &mut summary)?;
        validate_region_accesses(operation, region_index, &summary)?;
        Ok(summary)
    }

    /// Returns the complete stored value one operand of a structured operation denotes, or [`None`] when the operand is an
    /// value.
    ///
    /// A derived view is rejected rather than resolved to its allocation. A state boundary carries complete-value values, so
    /// only a handle with complete-value provenance may cross it; the view has to be re-derived from the allocation inside the
    /// region instead.
    ///
    /// A *preserved* allocation is resolved like any other. It crosses the boundary as the reference it already is, at its
    /// own declared operand position, so it needs no state carry at all — which is exactly what
    /// [`threaded_state_allocations`](Self::threaded_state_allocations) filters it out of.
    ///
    /// # Parameters
    ///
    ///   - `operand`: Carrier being classified.
    ///   - `operation`: Name of the operation being rewritten, used in the diagnostic.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when the operand denotes a derived view of its allocation, or when its
    /// allocation is no longer live.
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
        if !reference.denotes_complete_value() {
            return Err(ProgramError::MalformedProgram(format!(
                "operation `{operation}` passes the derived view `{}` of {allocation} across a region boundary, which \
                 carries the complete stored value `{}`; derive the view inside the region instead",
                reference.r#type(),
                whole,
            )));
        }
        Ok(Some(allocation))
    }

    /// Returns the allocations one region closure needs threaded through the rewritten boundary as immutable state, in
    /// canonical allocation order, and validates that every one of them is still live.
    ///
    /// A closure needs an allocation threaded whenever its replay must be able to resolve that allocation — because it accesses
    /// it, returns it, or merely rematerializes a capture constant that denotes it — so the set is the summary's
    /// reached allocations, a strict superset of the accessed and returned allocations, with the *preserved* allocations removed. A
    /// preserved reference survives in the destination as a reference value and crosses at its own declared operand
    /// position, exactly as the source passed it, so it
    /// needs no state carry, publishes no successor, and widens nothing. This is the one place that distinction is
    /// drawn, which is what keeps the four structured rewrites stating one thing.
    ///
    /// # Parameters
    ///
    ///   - `summary`: Summary of the closures the rewritten operation attaches, in caller-allocation terms.
    ///   - `operation`: Name of the operation being rewritten, used in the diagnostic.
    ///
    /// # Errors
    ///
    /// Reports the first allocation the closures reach that has no live state, propagating the environment's own reason —
    /// consumed, never bound, or belonging to another environment — because that reason is what a caller needs and
    /// this check is in no position to restate it.
    pub fn threaded_state_allocations(
        &self,
        summary: &ReferenceRegionSummary,
        operation: &str,
    ) -> Result<BTreeSet<ReferenceDischargeAllocationId>, ProgramError> {
        let mut threaded = BTreeSet::new();
        for allocation in summary.reached() {
            if self.is_allocation_discharged(allocation).map_err(|error| {
                ProgramError::MalformedProgram(format!("operation `{operation}` reaches {allocation}: {error}"))
            })? {
                threaded.insert(allocation);
            }
        }
        Ok(threaded)
    }

    /// Computes the symmetric widening facts one structured rule needs from a region summary: the discharged
    /// allocations threaded as state, every reached allocation gaining an added boundary position because no declared
    /// position already carries it, and the discharged subset whose successor states the rebuilt regions must publish.
    /// An added preserved allocation crosses as the destination reference it already denotes rather than as state.
    ///
    /// # Parameters
    ///
    ///   - `summary`: Summary of the closures the rewritten operation attaches, in caller-allocation terms.
    ///   - `declared`: Allocations already crossing at declared boundary positions, which therefore need no added
    ///     position.
    ///   - `operation`: Name of the operation being rewritten, used in the diagnostic.
    ///
    /// # Errors
    ///
    /// Propagates [`threaded_state_allocations`](Self::threaded_state_allocations)'s liveness failures.
    pub fn state_widening(
        &self,
        summary: &ReferenceRegionSummary,
        declared: &BTreeSet<ReferenceDischargeAllocationId>,
        operation: &str,
    ) -> Result<ReferenceStateWidening, ProgramError> {
        let threaded = self.threaded_state_allocations(summary, operation)?;
        let entering = summary.reached().filter(|allocation| !declared.contains(allocation)).collect();
        let published = threaded.iter().copied().filter(|allocation| summary.is_mutated(*allocation)).collect();
        Ok(ReferenceStateWidening { threaded, entering, published })
    }

    /// Merges one boundary state output back into `allocation` with the summary's mutation fact, skipping allocations outside
    /// `threaded`: a carry that survives as a reference returned itself, so it has no successor state to merge.
    ///
    /// # Errors
    ///
    /// Propagates the underlying state replacement's liveness and type failures.
    pub fn merge_boundary_state(
        &self,
        summary: &ReferenceRegionSummary,
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

    /// Returns the destination value one operand of a structured operation contributes to the rewritten application:
    /// the current immutable state of a discharged reference, the destination reference of a preserved one, or the
    /// operand's own value.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when the operand's allocation is not live.
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

    /// Returns the destination value one live allocation contributes to a rewritten boundary: its current immutable
    /// state when discharged, or its destination reference when preserved.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when `allocation` is not live in this environment.
    pub fn allocation_value(&self, allocation: ReferenceDischargeAllocationId) -> Result<C::Value, ProgramError> {
        self.operand_value(&self.allocation_reference(allocation)?)
    }
}

#[cfg(test)]
mod tests {

    use std::collections::BTreeSet;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::parameters::Placeholder;
    use crate::programs::ProgramError;

    use crate::programs::builders::ProgramBuilder;

    use crate::programs::references::discharge::tests::*;
    use crate::programs::references::semantics::ReferenceAccessMode;
    use crate::programs::references::types::ReferenceType;

    use crate::programs::{
        RecursiveReferenceDischargeDriver, ReferenceDischargeDriver, ReferenceRegionDischargeBoundary,
        ReferenceRegionStateInsertion, ReferenceRegionSummary,
    };

    use super::*;
    use crate::programs::references::discharge::transform::ReferenceDischargeCaptureScope;

    #[test]
    fn test_reference_region_summary_unions_exact_access_modes() {
        let context = ListDischargeContext::new(ListDestination::new());
        let allocated = context
            .bind_discharged(ReferenceType::new(ListType { length: 2 }), ListIrValue::List(vec![1, 2]))
            .unwrap();
        let allocation = allocated.try_as_reference("the caller allocation").unwrap().allocation_id();
        let mut left = ReferenceRegionSummary::default();
        left.record(allocation, ReferenceAccessMode::Read, "list.read").unwrap();
        left.record(allocation, ReferenceAccessMode::ReadWrite, "list.swap").unwrap();
        left.output_allocations = vec![Some(allocation)];
        let mut right = ReferenceRegionSummary::default();
        right.record(allocation, ReferenceAccessMode::Write, "list.write").unwrap();
        right.record(allocation, ReferenceAccessMode::Accumulate, "list.add_update").unwrap();
        right.output_allocations = vec![None];

        let merged = left.merged(&right);
        assert_eq!(
            merged.access_modes(allocation).collect::<Vec<_>>(),
            vec![
                ReferenceAccessMode::Read,
                ReferenceAccessMode::Write,
                ReferenceAccessMode::ReadWrite,
                ReferenceAccessMode::Accumulate,
            ],
        );
        assert!(merged.is_mutated(allocation));
        assert_eq!(merged.output_allocations(), [Some(allocation)]);
    }

    #[test]
    fn test_reference_region_summary_validates_each_exact_access_mode() {
        let context = ListDischargeContext::new(ListDestination::new());
        let allocated = context
            .bind_discharged(ReferenceType::new(ListType { length: 2 }), ListIrValue::List(vec![1, 2]))
            .unwrap();
        let allocation = allocated.try_as_reference("the caller allocation").unwrap().allocation_id();
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
                    &[Some(allocation), None],
                );
                if allowed == accessed {
                    let summary = result.unwrap();
                    assert_eq!(summary.access_modes(allocation).collect::<Vec<_>>(), vec![accessed]);
                    assert_eq!(
                        summary.is_mutated(allocation),
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
                            "operation `test.single_mode_region` does not allow region 0 to access {allocation} with mode \
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
                &[Some(allocation), None],
            ),
            Err(ProgramError::MalformedProgram(format!(
                "operation `test.single_mode_region` does not allow region 0 to access {allocation} with mode `read/write`",
            ))),
        );
    }

    #[test]
    fn test_reference_region_summary_reports_transitive_accesses_and_output_allocations() {
        // A callee that replaces the state of the reference it receives, so the outer region's access to that allocation is
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

        // The outer region reads the caller's allocation directly, replaces it through the callee, and separately allocates,
        // reads, and returns an allocation of its own.
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
            .bind_discharged(ReferenceType::new(ListType { length: 2 }), ListIrValue::List(vec![1, 2]))
            .unwrap();
        let allocation = allocated.try_as_reference("the caller allocation").unwrap().allocation_id();
        let summary = context
            .region_summary(&ListOperation::Call, 0, program.entry_region_ref(), &[Some(allocation), None])
            .unwrap();

        // The caller allocation is reported as mutated because the nested callee replaces it, while the region's own
        // allocation crosses no boundary and is therefore absent from the summary entirely.
        assert_eq!(summary.accessed().collect::<Vec<_>>(), vec![allocation]);
        assert_eq!(
            summary.access_modes(allocation).collect::<Vec<_>>(),
            vec![ReferenceAccessMode::Read, ReferenceAccessMode::ReadWrite],
        );
        assert!(summary.has_access(allocation, ReferenceAccessMode::Read));
        assert!(summary.has_access(allocation, ReferenceAccessMode::ReadWrite));
        assert!(!summary.has_access(allocation, ReferenceAccessMode::Write));
        assert!(summary.is_mutated(allocation));

        // A declared output resolves to the caller allocation it denotes: the first output returns the allocation itself, the
        // second returns a region-local allocation, and the remaining three are values.
        assert_eq!(summary.output_allocations(), &[Some(allocation), None, None, None, None]);
    }

    #[test]
    fn test_reference_region_summary_rejects_a_closure_that_consumes_a_caller_allocation() {
        let mut builder = ProgramBuilder::<ListIrValue, ListOperation>::new();
        let reference = builder.add_input(ListIrType::Reference(ReferenceType::new(ListType { length: 2 })));
        let frozen = builder.add_instruction(ListOperation::Freeze, Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<ListIrValue>, Vec<ListIrValue>>(vec![frozen], vec![Placeholder], vec![Placeholder])
            .unwrap();

        // A consumed allocation has no successor state, so no state boundary can describe what became of it. The summary
        // rejects that outright rather than letting the caller keep threading state that is no longer live.
        let context = ListDischargeContext::new(ListDestination::new());
        let allocated = context
            .bind_discharged(ReferenceType::new(ListType { length: 2 }), ListIrValue::List(vec![1, 2]))
            .unwrap();
        let allocation = allocated.try_as_reference("the caller allocation").unwrap().allocation_id();
        assert_eq!(
            context.region_summary(&ListOperation::Call, 0, program.entry_region_ref(), &[Some(allocation)]),
            Err(ProgramError::MalformedProgram(format!(
                "reference discharge cannot pass {allocation} into a region that consumes it through `list.freeze`",
            ))),
        );
    }

    #[test]
    fn test_reference_discharge_threads_a_capture_scoped_allocation_a_nested_region_only_receives() {
        // A closure can reach a capture-scoped allocation without ever accessing it, by passing the constant into a nested
        // region that ignores it. The replay still materializes the constant, because something consumes it, so the
        // allocation has to be threaded even though no reference access records it. In particular, materializing the
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
            .bind_discharged(ReferenceType::new(ListType { length: 2 }), ListIrValue::List(vec![1, 2]))
            .unwrap();
        let allocation = allocated.try_as_reference("the captured allocation").unwrap().allocation_id();
        let context = context.with_captures(ReferenceDischargeCaptureScope::new(
            list_capture_position,
            vec![None, None, Some(allocation)],
        ));

        // The enclosing policy accepts writes only. Capture reachability still sizes the boundary, while the exact
        // access summary remains empty because neither closure semantically accesses the allocation.
        let summary = context
            .region_summary(
                &SingleModeRegionOperation(ReferenceAccessMode::Write),
                0,
                program.entry_region_ref(),
                &[None],
            )
            .unwrap();
        assert_eq!(summary.accessed().collect::<Vec<_>>(), Vec::<ReferenceDischargeAllocationId>::new());
        assert_eq!(summary.access_modes(allocation).collect::<Vec<_>>(), Vec::<ReferenceAccessMode>::new());
        assert!(!summary.is_mutated(allocation));
        assert_eq!(
            context.threaded_state_allocations(&summary, "test.single_mode_region"),
            Ok(BTreeSet::from([allocation])),
        );

        // The rebuilt region therefore receives the allocation's entering state and hands it to its own callee.
        let regions = [program.clone()];
        let driver = RecursiveReferenceDischargeDriver::new(&regions, None);
        let boundary = ReferenceRegionDischargeBoundary::new(
            &ListOperation::Call,
            0,
            vec![None],
            ReferenceRegionStateInsertion::new(vec![allocation], 1),
            ReferenceRegionStateInsertion::new(Vec::new(), 0),
        );
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
        assert_eq!(fork.mutated_allocations, []);

        // The same reached capture remains an entering boundary allocation when partial discharge preserves it. It
        // leaves the state-threaded and published sets empty because it crosses as its destination reference instead.
        let preserved_context = ListDischargeContext::new(ListDestination::new());
        let preserved = preserved_context
            .bind_preserved(
                ReferenceType::new(ListType { length: 2 }),
                ListIrValue::Reference(ReferenceType::new(ListType { length: 2 })),
            )
            .unwrap();
        let preserved_allocation =
            preserved.try_as_reference("the preserved captured allocation").unwrap().allocation_id();
        let preserved_context = preserved_context.with_captures(ReferenceDischargeCaptureScope::new(
            list_capture_position,
            vec![None, None, Some(preserved_allocation)],
        ));
        let preserved_summary = preserved_context
            .region_summary(
                &SingleModeRegionOperation(ReferenceAccessMode::Write),
                0,
                program.entry_region_ref(),
                &[None],
            )
            .unwrap();
        let widening = preserved_context
            .state_widening(&preserved_summary, &BTreeSet::new(), "test.single_mode_region")
            .unwrap();
        assert_eq!(widening.threaded(), &BTreeSet::new());
        assert_eq!(widening.entering(), &[preserved_allocation]);
        assert_eq!(widening.published(), &[]);
    }
}
