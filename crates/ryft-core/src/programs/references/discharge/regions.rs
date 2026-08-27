//! Isolated region rebuilding for reference discharge: boundaries, forks, summaries, drivers, and the shared
//! positional rewrite. Everything here serves the structured half of the transform — how one attached region closure
//! is summarized, widened with threaded state, rebuilt against an isolated environment, validated against its
//! summary, and sealed for the rule that requested it.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::rc::Rc;

use crate::contexts::{Context, Domain, StagingContext};
use crate::macros::check_count;
use crate::parameters::Placeholder;
use crate::programs::ProgramError;
use crate::programs::atoms::{Atom, AtomId};
use crate::programs::instructions::{Instruction, InstructionId};
use crate::programs::operations::Operation;
use crate::programs::programs::Program;
use crate::programs::regions::{
    EmptyRegionDriver, RegionDriver, RegionId, RegionRef, RegionReplayMappings, ReplayRegionDriver,
};
use crate::programs::types::{Type, Typed};
use crate::programs::values::Value;
use crate::tracing::TracingContext;

use super::{
    ReferenceCaptureScope, ReferenceDischargeContext, ReferenceDischargePolicy, ReferenceDischargeValue,
    ReferenceDischargeableOperation, ReferenceRootHandle, replay_preserved_access,
};
use crate::programs::references::semantics::{ReferenceAccessMode, ReferenceOutput};

/// Destination one structured reference discharge rule rebuilds a nested [`Region`](crate::Region) into: a fresh trace
/// of the same program universe, which seals into a program the rule attaches to its rewritten operation.
///
/// It is deliberately a fresh root trace rather than a nested trace of the live destination. A rebuilt region is a
/// self-contained artifact whose complete interface is its own boundary, so it must not close over any value of the
/// destination it will be attached in. Being a root trace is also what makes the type a fixed point of its own
/// construction — the destination of a destination is that same destination — which is what keeps the obligation that
/// this universe's operations discharge into it finite.
pub type ReferenceDischargeRegionDestination<C> = TracingContext<<C as Domain>::Constant, <C as Domain>::Operation>;

/// One group of state positions a rebuilt region gains: the roots crossing there and the source-boundary position at
/// which they are inserted.
///
/// Grouping the roots with their insertion position keeps a boundary request's input and output groups from being
/// transposable: the two groups have the same shape, and passing them positionally compiled silently when swapped.
#[derive(Clone, Debug, PartialEq)]
pub struct ReferenceRegionStateInsertion {
    /// Roots crossing at this group's positions, in canonical root order.
    roots: Vec<ReferenceRootHandle>,

    /// Position in the source region's boundary at which the group is inserted.
    position: usize,
}

impl ReferenceRegionStateInsertion {
    /// Creates a state-position group inserting `roots` at `position`.
    #[inline]
    pub fn new(roots: Vec<ReferenceRootHandle>, position: usize) -> Self {
        Self { roots, position }
    }
}

/// Symmetric widening facts one structured rule derives from a region summary through
/// [`ReferenceDischargeContext::state_widening`].
///
/// The three sets state one algorithm every symmetric structured rewrite shares: the *threaded* roots cross the
/// rebuilt boundary as immutable state, the *entering* subset gains added positions because no declared position
/// already carries it, and the *published* subset is what the rebuilt regions must report as mutated.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReferenceStateWidening {
    /// Every discharged root the region closures reach, in canonical root order.
    pub(super) threaded: BTreeSet<ReferenceRootHandle>,

    /// Threaded roots gaining added state positions, in canonical root order.
    pub(super) entering: Vec<ReferenceRootHandle>,

    /// Threaded roots some closure mutates, in canonical root order.
    pub(super) published: Vec<ReferenceRootHandle>,
}

impl ReferenceStateWidening {
    /// Returns every discharged root the region closures reach, in canonical root order.
    #[inline]
    pub fn threaded(&self) -> &BTreeSet<ReferenceRootHandle> {
        &self.threaded
    }

    /// Returns the threaded roots gaining added state positions, in canonical root order.
    #[inline]
    pub fn entering(&self) -> &[ReferenceRootHandle] {
        self.entering.as_slice()
    }

    /// Returns the threaded roots some closure mutates, in canonical root order.
    #[inline]
    pub fn published(&self) -> &[ReferenceRootHandle] {
        self.published.as_slice()
    }
}

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
    ///   - `added_inputs`: Roots whose entering state or preserved reference the rebuilt region receives as added
    ///     inputs, grouped with the source input position receiving them.
    ///   - `added_state_outputs`: Roots whose final state the rebuilt region publishes as added outputs, grouped with
    ///     the source output position receiving them.
    #[inline]
    pub fn new<O: Operation>(
        operation: &O,
        region_index: usize,
        declared_input_roots: Vec<Option<ReferenceRootHandle>>,
        added_inputs: ReferenceRegionStateInsertion,
        added_state_outputs: ReferenceRegionStateInsertion,
    ) -> Self {
        Self {
            declared_input_roots,
            capture_input_count: operation.region_capture_input_count(region_index),
            added_input_roots: added_inputs.roots,
            state_input_insertion: added_inputs.position,
            added_state_output_roots: added_state_outputs.roots,
            state_output_insertion: added_state_outputs.position,
        }
    }

    /// Creates a rebuilt-region boundary request whose added inputs and added state outputs are the same roots at the
    /// same position, which is the symmetric loop-carry shape `while` bodies and `scan` bodies thread.
    #[inline]
    pub fn symmetric<O: Operation>(
        operation: &O,
        region_index: usize,
        declared_input_roots: Vec<Option<ReferenceRootHandle>>,
        state: ReferenceRegionStateInsertion,
    ) -> Self {
        Self::new(operation, region_index, declared_input_roots, state.clone(), state)
    }

    /// Returns the root entering at each declared boundary position, or [`None`] for an ordinary value.
    #[inline]
    fn declared_input_roots(&self) -> &[Option<ReferenceRootHandle>] {
        self.declared_input_roots.as_slice()
    }

    /// Returns the region's own leading capture prefix, or [`None`] when it inherits its caller's capture scope.
    #[inline]
    const fn capture_input_count(&self) -> Option<usize> {
        self.capture_input_count
    }

    /// Returns the roots whose entering state or preserved reference the rebuilt region receives as added inputs.
    #[inline]
    fn added_input_roots(&self) -> &[ReferenceRootHandle] {
        self.added_input_roots.as_slice()
    }

    /// Returns the source input position at which the added state inputs are inserted.
    #[inline]
    const fn state_input_insertion(&self) -> usize {
        self.state_input_insertion
    }

    /// Returns the roots whose final state the rebuilt region publishes as added outputs.
    #[inline]
    fn added_state_output_roots(&self) -> &[ReferenceRootHandle] {
        self.added_state_output_roots.as_slice()
    }

    /// Returns the source output position at which the added state outputs are inserted.
    #[inline]
    const fn state_output_insertion(&self) -> usize {
        self.state_output_insertion
    }
}

/// Sealed result of discharging one attached region against an isolated environment.
///
/// This is the transactional artifact of a structured rule's region fork, and it deliberately carries no values of
/// any kind. A reference handle produced inside the fork would keep addressing the fork's own abandoned environment,
/// and even a plain destination value is not detached under a staging destination, because it is itself a tracer
/// stamped with the fork's builder. Excluding both structurally is what makes the isolation a type-level fact rather
/// than a convention: the owning rule binds the rebuilt operation in its *own* context and merges the final states
/// from the outputs that binding produced.
#[derive(Debug)]
pub struct ReferenceRegionDischargeFork<V: Value, O: Operation<Type = V::Type>> {
    /// Rebuilt, discharged region program.
    pub(super) program: Program<V, O, Vec<V>, Vec<V>>,

    /// Root each *declared* region output denotes, or [`None`] for an ordinary output, in region-boundary order.
    output_roots: Vec<Option<ReferenceRootHandle>>,

    /// Threaded roots the region's closure mutated, in canonical root order.
    pub(super) mutated_roots: Vec<ReferenceRootHandle>,
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
    /// replays. Returning [`None`] declares the allocation unnameable by any
    /// [`ReferenceDischargeSite`](super::ReferenceDischargeSite) and therefore *always discharged*, silently ignoring
    /// the caller's partial selection. This method is deliberately required
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

    /// Source coordinate of the application, or [`None`] for an application that replays no instruction.
    instruction: Option<InstructionId>,
}

impl<'r, D> RecursiveReferenceDischargeDriver<'r, D> {
    /// Creates a new [`RecursiveReferenceDischargeDriver`].
    ///
    /// # Parameters
    ///
    ///   - `driver`: Application-scoped [`RegionDriver`] exposing the attached regions.
    ///   - `instruction`: Source coordinate of the application, or [`None`] for an application that replays no
    ///     instruction.
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
            let position = InstructionId::new(region.id(), instruction_index);
            instruction_index += 1;
            // Run the complete rewrite of one application — the preserved-access replay included — inside the source
            // instruction's recorded origin, so every staged instruction records where it came from. Rules stage
            // their rewritten work through the destination parent, which is where the provenance state lives.
            context.parent().invoke_with_provenance_origin(instruction.provenance().clone(), || {
                // Access-only applications over exclusively preserved roots replay verbatim here, so rules see only
                // discharged accesses; operations that attach regions or declare reference outputs keep their own
                // rules.
                if instruction.regions().is_empty()
                    && let Some(outputs) =
                        replay_preserved_access(instruction.operation(), context, instruction_inputs)?
                {
                    return Ok(outputs);
                }
                let regions = ReplayRegionDriver::new(region, instruction.regions(), &mappings)?;
                let driver = RecursiveReferenceDischargeDriver::new(&regions, Some(position));
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
pub(super) fn summarize_region_closure<V: Value, O: Operation<Type = V::Type>>(
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
pub(super) fn validate_region_accesses<O: Operation>(
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
pub(super) fn lift_constant<C: Context, P: ReferenceDischargePolicy<C>>(
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
            ReferenceRegionStateInsertion::new(entering.clone(), forwarded.len()),
            ReferenceRegionStateInsertion::new(leaving.clone(), source_output_count),
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
                context.merge_boundary_state(&summary, &threaded, root, output)?;
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
