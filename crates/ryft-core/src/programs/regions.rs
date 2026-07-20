//! Contains machinery for representing and working with [`Program`] _regions_.
//!
//! A [`Program`] owns one arena of sealed [`Region`]s. Each region has its own [`Atom`] table, [`Instruction`]
//! sequence, and ordered input/output boundary, while [`RegionId`]s identify roots within that arena. Instructions
//! attach nested computations by storing region IDs in [`Operation`]-defined order. Reusing one ID preserves sharing.
//! Equal computations built independently remain distinct regions.
//!
//! [`RegionRef`] borrows a root together with its complete source arena. It supports inspection without cloning and
//! can materialize the reachable region graph as an owned [`Program`] through [`RegionRef::to_program`]. Regions are
//! sealed before attachment to an [`Instruction`]/[`Program`], and so borrowed views remain immutable and recursively
//! derived metadata such as [`Effects`] cannot become stale.
//!
//! [`RegionInterface`] is the type-and-effect summary passed to [`Operation::infer_output_types`]. It deliberately
//! exposes a region boundary without exposing the region body. [`OutputRegionProvenance`] describes when an operation
//! output originates from an attached-region output (as opposed to it originating directly from that [`Instruction`]).
//!
//! Operation rules receive application-scoped structural access to attached [`Region`]s through [`RegionDriver`]s.
//! Binding applications obtain their complete ordered region sequence from [`BindingRegionDriver`], which can provide
//! owned programs, borrowed regions, and shared callees (e.g., a JIT-compiled function). Replay and transforms provide
//! borrowed views of the regions attached to the current instruction. [`EmptyRegionDriver`] supplies the same contract
//! for applications with no attached regions.

use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::Display;
use std::rc::{Rc, Weak};

use crate::parameters::Placeholder;
use crate::programs::ProgramError;
use crate::programs::atoms::{Atom, AtomId};
use crate::programs::builders::ProgramBuilder;
use crate::programs::effects::Effects;
use crate::programs::instructions::Instruction;
use crate::programs::operations::Operation;
use crate::programs::programs::Program;
use crate::programs::types::{Type, Typed};
use crate::programs::values::Value;

/// Unique identifier for a [`Region`] within a [`Program`]. [`RegionId`]s are stable indexes into a [`Program`]'s
/// region arena. Like [`AtomId`]s, they are meaningful only against the [`Program`] they were derived from.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RegionId {
    /// Zero-based index of the corresponding [`Region`] inside the owning [`Program`]'s region arena.
    index: usize,
}

impl RegionId {
    /// Creates a new [`RegionId`] from the provided zero-based region-arena index.
    #[inline]
    pub fn new(index: usize) -> Self {
        Self { index }
    }

    /// Returns the zero-based index of the corresponding [`Region`] inside the owning [`Program`]'s region arena.
    #[inline]
    pub fn index(self) -> usize {
        self.index
    }
}

impl Display for RegionId {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "^{}", self.index)
    }
}

/// Computation region inside a [`Program`]'s region arena. Every region owns its own [`Atom`] table, [`Instruction`]
/// sequence, and input/output boundary. The public program entry point and every nested computation are [`Region`]s in
/// the same arena. [`Instruction`]s reference them by [`RegionId`], and regions may be shared. Furthermore, regions are
/// _sealed_ meaning that a region referenced by an instruction can never change after that instruction is built.
#[derive(Clone, Debug)]
pub struct Region<V: Typed, O> {
    /// [`Atom`]s contained in this [`Region`], in the order in which they will be evaluated.
    pub(crate) atoms: Vec<Atom<V>>,

    /// [`AtomId`]s of the [`Atom`]s that correspond to the inputs of this [`Region`].
    pub(crate) input_ids: Vec<AtomId>,

    /// [`AtomId`]s of the [`Atom`]s that correspond to the outputs of this [`Region`].
    pub(crate) output_ids: Vec<AtomId>,

    /// Ordered sequence of [`Instruction`]s that make up the computational graph of this [`Region`].
    pub(crate) instructions: Vec<Instruction<O>>,
}

impl<V: Typed, O> Region<V, O> {
    /// Returns the [`Atom`]s contained in this [`Region`], in the order in which they will be evaluated.
    #[inline]
    pub fn atoms(&self) -> &[Atom<V>] {
        &self.atoms
    }

    /// Returns the [`AtomId`]s of the [`Atom`]s that correspond to the inputs of this [`Region`].
    #[inline]
    pub fn input_ids(&self) -> &[AtomId] {
        &self.input_ids
    }

    /// Returns the [`Type`]s of the inputs of this [`Region`], in boundary order.
    #[inline]
    pub fn input_types(&self) -> Vec<V::Type> {
        self.input_ids.iter().map(|input| self.atoms[input.index()].r#type().into_owned()).collect()
    }

    /// Returns the [`AtomId`]s of the [`Atom`]s that correspond to the outputs of this [`Region`].
    #[inline]
    pub fn output_ids(&self) -> &[AtomId] {
        &self.output_ids
    }

    /// Returns the [`Type`]s of the outputs of this [`Region`], in boundary order.
    #[inline]
    pub fn output_types(&self) -> Vec<V::Type> {
        self.output_ids.iter().map(|output| self.atoms[output.index()].r#type().into_owned()).collect()
    }

    /// Returns the ordered sequence of [`Instruction`]s that make up the computational graph of this [`Region`].
    #[inline]
    pub fn instructions(&self) -> &[Instruction<O>] {
        &self.instructions
    }

    /// Returns a vector that has the same length as this [`Region`]'s [`Atom`] table and contains, for every atom, the
    /// index of the [`Instruction`] that produces it. [`None`] is used for inputs and constants.
    pub(crate) fn instruction_by_output(&self) -> Vec<Option<usize>> {
        let mut instruction_by_output = vec![None; self.atoms.len()];
        for (instruction_index, instruction) in self.instructions.iter().enumerate() {
            for output in instruction.outputs().iter().copied() {
                if let Some(slot) = instruction_by_output.get_mut(output.index()) {
                    *slot = Some(instruction_index);
                }
            }
        }
        instruction_by_output
    }

    /// Computes the recursively derived [`Effects`] of every provided [`Region`], in [`RegionId`] order. Each region's
    /// effects are the union of its [`Instruction`]s' intrinsic [`Operation::effects`] and the effects of their
    /// attached regions. A single ascending pass suffices because instructions only reference previously sealed regions
    /// (i.e., regions with strictly smaller [`RegionId`]s), which is guaranteed by construction.
    pub(crate) fn effects(regions: &[Self]) -> Vec<Effects>
    where
        O: Operation<V::Type>,
    {
        let mut effects = Vec::<Effects>::with_capacity(regions.len());
        for region in regions {
            let mut region_effects = Effects::PURE;
            for instruction in &region.instructions {
                region_effects = region_effects.union(instruction.operation().effects());
                for nested_region in instruction.regions().iter().copied() {
                    region_effects = region_effects.union(effects[nested_region.index()]);
                }
            }
            effects.push(region_effects);
        }
        effects
    }
}

/// Borrowed view of a [`Region`]. [`RegionRef`] provides [`Program`]-like access to a rooted, nested computation
/// without cloning the owning [`Program`]'s region arena. The intended usage is to store [`RegionId`]s in long-lived
/// Intermediate Representation (IR) objects and recreate [`RegionRef`]s only while inspecting, replaying, importing,
/// or lowering a source arena. Calling [`RegionRef::to_program`] crosses the explicit ownership boundary as it clones
/// the selected region's complete reachable region closure into a detached [`Program`].
#[derive(Debug)]
pub struct RegionRef<'r, V: Typed, O> {
    /// Arena containing the referenced [`Region`] and its reachable descendants.
    regions: &'r [Region<V, O>],

    /// Identifier of the rooted [`Region`] within `regions`.
    id: RegionId,
}

impl<'r, V: Typed, O> RegionRef<'r, V, O> {
    /// Creates a new [`RegionRef`].
    pub fn new(regions: &'r [Region<V, O>], id: RegionId) -> Result<Self, ProgramError> {
        regions
            .get(id.index())
            .map(|_| Self { regions, id })
            .ok_or_else(|| ProgramError::MalformedProgram(format!("region {id} is out of range")))
    }

    /// Returns the [`RegionId`] identifying this rooted [`Region`] in its source arena.
    #[inline]
    pub fn id(self) -> RegionId {
        self.id
    }

    /// Returns the borrowed source arena that contains this [`RegionRef`]'s root and descendants.
    #[inline]
    pub fn regions(self) -> &'r [Region<V, O>] {
        self.regions
    }

    /// Returns the rooted [`Region`].
    #[inline]
    pub fn region(self) -> &'r Region<V, O> {
        &self.regions[self.id.index()]
    }

    /// Returns the [`Atom`]s contained in the rooted [`Region`].
    #[inline]
    pub fn atoms(self) -> &'r [Atom<V>] {
        self.region().atoms()
    }

    /// Returns the input [`AtomId`]s of the rooted [`Region`].
    #[inline]
    pub fn input_ids(self) -> &'r [AtomId] {
        self.region().input_ids()
    }

    /// Returns the input types of the rooted [`Region`], in boundary order.
    #[inline]
    pub fn input_types(self) -> Vec<V::Type> {
        self.region().input_types()
    }

    /// Returns the output [`AtomId`]s of the rooted [`Region`].
    #[inline]
    pub fn output_ids(self) -> &'r [AtomId] {
        self.region().output_ids()
    }

    /// Returns the output types of the rooted [`Region`], in boundary order.
    #[inline]
    pub fn output_types(self) -> Vec<V::Type> {
        self.region().output_types()
    }

    /// Returns the ordered [`Instruction`]s contained in the rooted [`Region`].
    #[inline]
    pub fn instructions(self) -> &'r [Instruction<O>] {
        self.region().instructions()
    }

    /// Returns the [`RegionInterface`] of the rooted [`Region`].
    #[inline]
    pub fn interface(self) -> RegionInterface<V::Type>
    where
        O: Operation<V::Type>,
    {
        RegionInterface::new(self.input_types(), self.output_types(), self.effects())
    }

    /// Returns the recursively derived [`Effects`] of the rooted [`Region`].
    #[inline]
    pub fn effects(self) -> Effects
    where
        O: Operation<V::Type>,
    {
        Region::effects(self.regions)[self.id.index()]
    }
}

impl<V: Value, O: Operation<V::Type>> RegionRef<'_, V, O> {
    /// Materializes this borrowed [`Region`] and its complete reachable region closure as a [`Program`]. Descendant
    /// sharing is preserved within the resulting [`Program`], meaning that if several [`Instruction`]s in the reachable
    /// closure point at the same source [`RegionId`], the resulting program contains one copied descendant and all
    /// copied instructions point at that one destination region. The selected root becomes the resulting program entry
    /// region with placeholder input/output structures matching its input/output boundary. This operation takes time
    /// and space proportional to the size of the complete reachable region graph.
    pub fn to_program(self) -> Program<V, O, Vec<V>, Vec<V>> {
        // Importing appends the root after all of its descendants. Removing that final region leaves the complete
        // descendant arena in place, and `build` assigns the promoted entry the same identifier the imported root had.
        let mut builder = ProgramBuilder::<V, O>::new();
        let root = builder.import_region(self);
        let Region { atoms, input_ids, output_ids, instructions } = builder.regions.pop().unwrap();
        debug_assert_eq!(root.index(), builder.regions.len());
        let input_count = input_ids.len();
        let output_count = output_ids.len();
        builder.atoms = atoms;
        builder.input_ids = input_ids;
        builder.instructions = instructions;
        builder.build(output_ids, vec![Placeholder; input_count], vec![Placeholder; output_count]).unwrap()
    }
}

impl<V: Typed, O> Copy for RegionRef<'_, V, O> {}

impl<V: Typed, O> Clone for RegionRef<'_, V, O> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

/// Read-only boundary summary of a sealed [`Region`], as seen by [`Operation`] type inference. A [`RegionInterface`]
/// preserves the exact [`Region::input_ids`] and [`Region::output_ids`] order and carries the region's recursively
/// derived [`Effects`], so that region-carrying operations can validate and consume the boundary contracts of their
/// attached regions (e.g., a condition operation checking that its branches agree, or a while operation rejecting an
/// effectful body when its predicate is batched) without ever seeing the region contents. [`ProgramBuilder`]s derive
/// [`RegionInterface`]s from their own region arenas immediately before invoking [`Operation::infer_output_types`] and
/// never store them. Final [`Program`] validation independently derives them again so that callers cannot inject stale
/// interface metadata into an [`Instruction`]. Note, though, that constructing a [`RegionInterface`] directly is still
/// allowed as passing synthetic interfaces to [`Operation::infer_output_types`] performs a pure hypothetical inference
/// and cannot mutate or create a [`Program`].
#[derive(Clone, Debug, PartialEq)]
pub struct RegionInterface<T: Type> {
    /// [`Type`]s derived from the [`Region`]'s input [`Atom`]s, in [`Region::input_ids`] order.
    input_types: Vec<T>,

    /// [`Type`]s derived from the [`Region`]'s output [`Atom`]s, in [`Region::output_ids`] order.
    output_types: Vec<T>,

    /// [`Effects`] of the [`Region`], derived recursively from its [`Instruction`]s and their attached regions.
    effects: Effects,
}

impl<T: Type> RegionInterface<T> {
    /// Creates a new [`RegionInterface`].
    #[inline]
    pub fn new(input_types: Vec<T>, output_types: Vec<T>, effects: Effects) -> Self {
        Self { input_types, output_types, effects }
    }

    /// Returns the [`Type`]s of the [`Region`]'s inputs, in [`Region::input_ids`] order.
    #[inline]
    pub fn input_types(&self) -> &[T] {
        &self.input_types
    }

    /// Returns the [`Type`]s of the [`Region`]'s outputs, in [`Region::output_ids`] order.
    #[inline]
    pub fn output_types(&self) -> &[T] {
        &self.output_types
    }

    /// Returns the [`Effects`] of the [`Region`], derived recursively from its [`Instruction`]s
    /// and their attached regions.
    #[inline]
    pub fn effects(&self) -> Effects {
        self.effects
    }
}

/// [`RegionDriver`]s provide structural access to the nested [`Region`]s of [`Operation`] applications.
/// A driver is application-scoped: during program replay it describes exactly one [`Instruction`], while a direct
/// [`Context::bind`](crate::Context::bind) invocation supplies the equivalent scope without requiring an existing
/// instruction. Transform-specific drivers extend this trait with the recursive work they can perform on those regions,
/// while this shared capability keeps region lookup independent of batching, differentiation, partial evaluation, or
/// transposition. Drivers **must not** combine regions from multiple operation applications.
pub trait RegionDriver<V: Value, O: Operation<V::Type>> {
    /// Returns an [`Iterator`] over borrowed views of every [`Region`] attached to the current operation application,
    /// in operation-defined order. Constructing and advancing this iterator cannot fail because the [`RegionDriver`]
    /// represents an already-validated application scope.
    fn regions<'r>(&'r self) -> impl Iterator<Item = RegionRef<'r, V, O>>
    where
        V: 'r,
        O: 'r;

    /// Returns the number of [`Region`]s attached to the current operation application.
    #[inline]
    fn region_count(&self) -> usize {
        self.regions().count()
    }

    /// Returns a borrowed view of the [`Region`] at `index`, or an error when the current operation application has no
    /// region at that index.
    #[inline]
    fn region(&self, index: usize) -> Result<RegionRef<'_, V, O>, ProgramError> {
        self.regions()
            .nth(index)
            .ok_or_else(|| ProgramError::MalformedProgram(format!("region index {index} is out of range")))
    }
}

impl<
    V: Value,
    O: Operation<V::Type>,
    R: AsRef<[Program<V, O, Vec<V>, Vec<V>>]> + IntoIterator<Item = Program<V, O, Vec<V>, Vec<V>>>,
> RegionDriver<V, O> for R
{
    #[inline]
    fn regions<'r>(&'r self) -> impl Iterator<Item = RegionRef<'r, V, O>>
    where
        V: 'r,
        O: 'r,
    {
        self.as_ref().iter().map(Program::entry_region_ref)
    }
}

/// Zero-region [`RegionDriver`] scoped to one operation application outside a staged [`Instruction`]. Normal program
/// and context replay always provide their concrete application driver, including for [`Operation`] applications with
/// no attached [`Region`]s, which is what this driver is intended for. Any attempted nested-region request will result
/// in a [`ProgramError`].
#[derive(Copy, Clone, Debug, Default)]
pub struct EmptyRegionDriver;

impl<V: Value, O: Operation<V::Type>> RegionDriver<V, O> for EmptyRegionDriver {
    #[inline]
    fn regions<'r>(&'r self) -> impl Iterator<Item = RegionRef<'r, V, O>>
    where
        V: 'r,
        O: 'r,
    {
        std::iter::empty()
    }
}

/// [`RegionDriver`] for the regions supplied to one [`Context::bind`](crate::Context::bind) [`Operation`] application.
/// In addition to providing application-scoped structural access through [`RegionDriver`], a binding region driver can
/// be consumed through [`import_into`](Self::import_into) to import its regions into a staging context's destination
/// [`ProgramBuilder`]. Implementations determine whether those regions are moved from owned [`Program`]s, borrowed from
/// a replayed program while preserving source-region sharing, or interned as shared callees.
///
/// Ordinary owned collections implement this trait when they support both slice-like borrowing and owned iteration.
/// Consequently, fixed-size arrays and [`Vec`]s remain valid direct binding arguments.
pub trait BindingRegionDriver<V: Value, O: Operation<V::Type>>: RegionDriver<V, O> + Sized {
    /// Imports these attached [`Region`]s into the provided [`ProgramBuilder`] in application order
    /// and returns their [`RegionId`]s in the same order.
    fn import_into(self, builder: &Rc<RefCell<ProgramBuilder<V, O>>>) -> Result<Vec<RegionId>, ProgramError>;
}

impl<
    V: Value,
    O: Operation<V::Type>,
    R: AsRef<[Program<V, O, Vec<V>, Vec<V>>]> + IntoIterator<Item = Program<V, O, Vec<V>, Vec<V>>>,
> BindingRegionDriver<V, O> for R
{
    #[inline]
    fn import_into(self, builder: &Rc<RefCell<ProgramBuilder<V, O>>>) -> Result<Vec<RegionId>, ProgramError> {
        let mut builder = builder.borrow_mut();
        Ok(self.into_iter().map(|region| builder.import_program(region)).collect())
    }
}

/// [`BindingRegionDriver`] for shared callee [`Program`]s attached to one [`Context::bind`](crate::Context::bind)
/// [`Operation`] application. Callees are exposed in the order provided at construction and are interned by [`Rc`]
/// identity when imported into a [`StagingContext`](crate::StagingContext), preserving sharing between repeated
/// references to the same program.
pub struct CalleeRegionDriver<'r, V: Value, O: Operation<V::Type>> {
    /// Shared callee [`Program`]s in [`Operation`]-defined region order.
    callees: &'r [Rc<Program<V, O, Vec<V>, Vec<V>>>],
}

impl<'r, V: Value, O: Operation<V::Type>> CalleeRegionDriver<'r, V, O> {
    /// Creates a new [`CalleeRegionDriver`].
    #[inline]
    pub fn new(callees: &'r [Rc<Program<V, O, Vec<V>, Vec<V>>>]) -> Self {
        Self { callees }
    }
}

impl<V: Value, O: Operation<V::Type>> RegionDriver<V, O> for CalleeRegionDriver<'_, V, O> {
    #[inline]
    fn regions<'r>(&'r self) -> impl Iterator<Item = RegionRef<'r, V, O>>
    where
        V: 'r,
        O: 'r,
    {
        self.callees.iter().map(|callee| callee.entry_region_ref())
    }
}

impl<V: Value, O: Operation<V::Type>> BindingRegionDriver<V, O> for CalleeRegionDriver<'_, V, O> {
    #[inline]
    fn import_into(self, builder: &Rc<RefCell<ProgramBuilder<V, O>>>) -> Result<Vec<RegionId>, ProgramError> {
        let mut builder = builder.borrow_mut();
        Ok(self.callees.iter().map(|callee| builder.intern_callee(callee)).collect())
    }
}

/// [`BindingRegionDriver`] for the borrowed [`Region`]s attached to one replayed [`Instruction`]. The roots remain
/// in their source region arena and are exposed in instruction order through [`RegionDriver`]. When a staging context
/// imports them, `mappings` preserves their source identities across every instruction in the surrounding replay.
/// Construction validates that every root belongs to `source`'s arena, which lets [`RegionDriver::regions`] remain
/// non-fallible without trusting callers to preserve that relationship.
pub(crate) struct ReplayRegionDriver<'r, V: Value, O: Operation<V::Type>> {
    /// Borrowed [`Region`] view used to access every root's shared source arena.
    source: RegionRef<'r, V, O>,

    /// Source [`RegionId`]s attached to the replayed instruction, in application order.
    roots: &'r [RegionId],

    /// Source-to-destination [`Region`] mappings shared by every [`Instruction`] driver in the surrounding replay.
    /// A [`Program`] is replayed one [`Instruction`] at a time, and so each instruction receives a distinct
    /// [`ReplayRegionDriver`]. Region identity, however, belongs to the complete source region arena. Two instructions
    /// can attach to the same source region, and two attached roots can share a descendant:
    ///
    /// ```text
    /// source instruction A ──┐
    ///                        ├──▶ region 3 ──▶ region 1
    /// source instruction B ──┘
    /// ```
    ///
    /// Importing each instruction with a fresh source-to-destination map would copy region 3 and region 1 once for A
    /// and again for B. The resulting program might contain equivalent region bodies, but it would no longer preserve
    /// the source graph's sharing. Keeping one [`RegionReplayMappings`] value for the complete source replay lets the
    /// second import reuse the destination identifiers established by the first:
    ///
    /// ```text
    /// source region 1 ──▶ destination region 0
    /// source region 3 ──▶ destination region 1
    /// ```
    ///
    /// The replay may feed different instructions to different destination [`ProgramBuilder`]s as composed contexts
    /// decide whether and where to stage them. Because a [`RegionId`] is meaningful only within its owning arena, each
    /// destination builder needs an independent source-to-destination map. [`RegionReplayMappings`] therefore stores
    /// one [`DestinationRegionMapping`] per live destination [`ProgramBuilder`]. The same source region might map to
    /// region 1 in one builder and region 7 in another without either mapping affecting the other.
    ///
    /// One [`RegionReplayMappings`] value must be scoped to exactly one source-arena replay. Its per-destination state
    /// is shared across that replay's instruction drivers, but must not be reused for a different source arena.
    mappings: &'r RegionReplayMappings<V, O>,
}

impl<'r, V: Value, O: Operation<V::Type>> ReplayRegionDriver<'r, V, O> {
    /// Creates a new [`ReplayRegionDriver`].
    #[inline]
    pub(crate) fn new(
        source: RegionRef<'r, V, O>,
        roots: &'r [RegionId],
        mappings: &'r RegionReplayMappings<V, O>,
    ) -> Result<Self, ProgramError> {
        for root in roots {
            RegionRef::new(source.regions(), *root)?;
        }
        Ok(Self { source, roots, mappings })
    }
}

impl<V: Value, O: Operation<V::Type>> RegionDriver<V, O> for ReplayRegionDriver<'_, V, O> {
    #[inline]
    fn regions<'r>(&'r self) -> impl Iterator<Item = RegionRef<'r, V, O>>
    where
        V: 'r,
        O: 'r,
    {
        self.roots.iter().map(|root| RegionRef::new(self.source.regions(), *root).unwrap())
    }
}

impl<V: Value, O: Operation<V::Type>> BindingRegionDriver<V, O> for ReplayRegionDriver<'_, V, O> {
    #[inline]
    fn import_into(self, builder: &Rc<RefCell<ProgramBuilder<V, O>>>) -> Result<Vec<RegionId>, ProgramError> {
        let builder_identity = Rc::downgrade(builder);
        let mut destinations = self.mappings.destinations.borrow_mut();
        destinations.retain(|mapping| mapping.builder.strong_count() > 0);
        let destination_index = destinations
            .iter()
            .position(|mapping| Weak::ptr_eq(&mapping.builder, &builder_identity))
            .unwrap_or_else(|| {
                destinations.push(DestinationRegionMapping { builder: builder_identity, remapping: HashMap::new() });
                destinations.len() - 1
            });
        let remapping = &mut destinations[destination_index].remapping;
        let mut builder = builder.borrow_mut();
        self.roots
            .iter()
            .map(|root| {
                let region = RegionRef::new(self.source.regions(), *root)?;
                Ok(builder.import_region_with_remapping(region, remapping))
            })
            .collect()
    }
}

/// Replay-scoped collection of source-to-destination [`RegionId`] mappings. One instance of this type is shared
/// by all [`ReplayRegionDriver`]s created while replaying one source [`Region`] arena. It maintains a separate
/// [`DestinationRegionMapping`] for every live destination [`ProgramBuilder`] so repeated roots and shared
/// descendants retain their identity across [`Instruction`] applications without mixing the unrelated identifier
/// spaces of different builders. Refer to [`ReplayRegionDriver::mappings`] for more information on how this is
/// used and why it is necessary.
pub(crate) struct RegionReplayMappings<V: Value, O: Operation<V::Type>> {
    /// Per-destination [`DestinationRegionMapping`]s accumulated during a replay.
    destinations: RefCell<Vec<DestinationRegionMapping<V, O>>>,
}

impl<V: Value, O: Operation<V::Type>> RegionReplayMappings<V, O> {
    /// Creates a new [`RegionReplayMappings`].
    #[inline]
    pub(crate) fn new() -> Self {
        Self { destinations: RefCell::new(Vec::new()) }
    }
}

/// Source-to-destination [`RegionId`] mapping for one live destination [`ProgramBuilder`] participating in a
/// [`Region`] replay. [`RegionReplayMappings`] owns one of these values per destination because [`RegionId`]s are local
/// to their owning arenas. Refer to [`ReplayRegionDriver::mappings`] for more information on how this is used and why
/// it is necessary.
pub(crate) struct DestinationRegionMapping<V: Value, O: Operation<V::Type>> {
    /// Weak identity of the destination [`ProgramBuilder`]. Weak ownership prevents replay bookkeeping from keeping
    /// a completed builder alive or interfering with trace finalization through `Rc::try_unwrap`.
    builder: Weak<RefCell<ProgramBuilder<V, O>>>,

    /// Source-to-destination [`RegionId`] remapping for the destination [`ProgramBuilder`].
    remapping: HashMap<RegionId, RegionId>,
}

/// Identifies one attached [`Region`] output that may produce an [`Operation`] output. Provenance is relative to an
/// [`Operation`] application: [`region_index`](Self::region_index) selects an entry from [`Instruction::regions`],
/// and [`output_index`](Self::output_index) selects an output of that attached region. Refer to
/// [`Operation::output_region_provenance`] for how operations provide this kind of provenance information.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct OutputRegionProvenance {
    /// Index of the attached [`Region`] in the [`Operation`]-defined region order.
    pub region_index: usize,

    /// Index of the output in the attached [`Region`]'s output boundary.
    pub output_index: usize,
}
