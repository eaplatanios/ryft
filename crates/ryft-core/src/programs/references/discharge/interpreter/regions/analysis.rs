use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use crate::contexts::Domain;
use crate::macros::check_count;
use crate::programs::ProgramError;
use crate::programs::atoms::{Atom, AtomId};
use crate::programs::instructions::Instruction;
use crate::programs::operations::Operation;
use crate::programs::references::semantics::{ReferenceAccessMode, ReferenceOutput};
use crate::programs::regions::{RegionId, RegionRef};
use crate::programs::types::{Type, Typed};
use crate::programs::values::Value;

use super::super::super::policies::ReferenceDischargePolicy;
use super::super::{
    ReferenceCaptureScope, ReferenceDischargeBinding, ReferenceDischargeContext, ReferenceDischargeValue,
    ReferenceRootHandle,
};
use super::boundaries::ReferenceStateWidening;

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
/// The summary separates *reachability* from *semantic access*. The reached set holds every caller root
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
    pub(super) output_roots: Vec<Option<ReferenceRootHandle>>,
}

impl ReferenceRegionSummary {
    /// Returns every caller root the closure must be able to resolve, in canonical root order.
    #[inline]
    pub(super) fn reached(&self) -> impl Iterator<Item = ReferenceRootHandle> + '_ {
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
    pub(super) fn record(
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
pub(super) fn nested_capture_scope<Constant>(
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
impl<C: Domain, P: ReferenceDischargePolicy<C>> ReferenceDischargeContext<C, P> {
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
    /// reached roots, a strict superset of the accessed and returned roots, with the *preserved* roots removed. A
    /// preserved root survives in the destination as an ordinary reference and crosses at its own declared operand
    /// position, exactly as the source passed it, so it
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

    /// Computes the symmetric widening facts one structured rule needs from a region summary: the threaded roots, the
    /// subset entering as added state because no declared position already carries them, and the subset whose
    /// successor states the rebuilt regions must publish.
    ///
    /// # Parameters
    ///
    ///   - `summary`: Summary of the closures the rewritten operation attaches, in caller-root terms.
    ///   - `declared`: Roots already crossing at declared boundary positions, which therefore need no added state.
    ///   - `operation`: Name of the operation being rewritten, used in the diagnostic.
    ///
    /// # Errors
    ///
    /// Propagates [`threaded_state_roots`](Self::threaded_state_roots)'s liveness failures.
    pub fn state_widening(
        &self,
        summary: &ReferenceRegionSummary,
        declared: &BTreeSet<ReferenceRootHandle>,
        operation: &str,
    ) -> Result<ReferenceStateWidening, ProgramError> {
        let threaded = self.threaded_state_roots(summary, operation)?;
        let entering = threaded.difference(declared).copied().collect();
        let published = threaded.iter().copied().filter(|root| summary.is_mutated(*root)).collect();
        Ok(ReferenceStateWidening { threaded, entering, published })
    }

    /// Merges one boundary state output back into `root` with the summary's mutation fact, skipping roots outside
    /// `threaded`: a carry that survives as a reference returned itself, so it has no successor state to merge.
    ///
    /// # Errors
    ///
    /// Propagates the underlying state replacement's liveness and type failures.
    pub fn merge_boundary_state(
        &self,
        summary: &ReferenceRegionSummary,
        threaded: &BTreeSet<ReferenceRootHandle>,
        root: ReferenceRootHandle,
        output: C::Value,
    ) -> Result<(), ProgramError> {
        if threaded.contains(&root) {
            self.merge_discharged_state(root, output, summary.is_mutated(root))?;
        }
        Ok(())
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
            ReferenceDischargeValue::Reference(reference) => reference,
            ReferenceDischargeValue::Ordinary(value) => return Ok(value.clone()),
        };
        match &reference.binding {
            ReferenceDischargeBinding::Discharged => self.discharged_state(reference.root()),
            ReferenceDischargeBinding::Preserved { reference: value } => {
                self.validate_live_root(reference.root())?;
                Ok(value.clone())
            }
        }
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

    use super::super::super::ReferenceCaptureScope;
    use super::*;

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
        let boundary = ReferenceRegionDischargeBoundary::new(
            &ListOperation::Call,
            0,
            vec![None],
            ReferenceRegionStateInsertion::new(vec![root], 1),
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
        assert_eq!(fork.mutated_roots, []);
    }
}
