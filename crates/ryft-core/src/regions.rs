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

use crate::effects::Effects;
use crate::operations::Operation;
use crate::parameters::Placeholder;
use crate::programs::{Atom, AtomId, Instruction, Program, ProgramBuilder, ProgramError, Value};
use crate::types::{Type, Typed};

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
        let mut builder = ProgramBuilder::<V, O>::new();
        let mut remapping = HashMap::new();
        let mut region = self.region().clone();
        for instruction in &mut region.instructions {
            for attached in instruction.regions_mut() {
                *attached = builder.clone_region_closure_into_arena(self.regions, *attached, &mut remapping);
            }
        }
        let input_count = region.input_ids.len();
        let output_count = region.output_ids.len();
        builder.atoms = region.atoms;
        builder.input_ids = region.input_ids;
        builder.instructions = region.instructions;
        builder
            .build(region.output_ids, vec![Placeholder; input_count], vec![Placeholder; output_count])
            .unwrap()
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

// TODO(eaplatanios): Review from here onwards.

/// [`BindingRegionDriver`] for the borrowed regions attached to one replayed [`Instruction`]. The roots remain in their
/// source region arena and are exposed in instruction order through [`RegionDriver`]. When a staging context imports
/// them, `mappings` preserves their source identities across every instruction in the surrounding replay. Construction
/// validates that every root belongs to `source`'s arena, which lets [`RegionDriver::regions`] remain non-fallible
/// without trusting callers to preserve that relationship.
pub(crate) struct ReplayRegionDriver<'r, V: Value, O: Operation<V::Type>> {
    /// Borrowed view used to access every root's shared source arena.
    source: RegionRef<'r, V, O>,

    /// Source [`RegionId`]s attached to the replayed instruction, in application order.
    roots: &'r [RegionId],

    /// Replay-wide source-to-destination region mappings.
    mappings: &'r RegionReplayMappings<V, O>,
}

impl<'r, V: Value, O: Operation<V::Type>> ReplayRegionDriver<'r, V, O> {
    /// Creates a driver for one instruction in an active replay, returning an error if any root does not belong to
    /// `source`'s arena.
    ///
    /// # Parameters
    ///
    ///   - `source`: Any valid region reference in the source arena.
    ///   - `roots`: Region identifiers attached to the instruction, in application order.
    ///   - `mappings`: Replay-wide mappings shared by every instruction in this source replay.
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
        self.mappings.import_into(builder, self.source, self.roots)
    }
}

/// Replay-scoped source-to-destination [`RegionId`] mappings. A source identifier is meaningful only in the replayed
/// arena, while an imported identifier is meaningful only in one destination builder. One value is therefore shared
/// across the complete replay of one source arena and maintains a separate [`DestinationRegionMapping`] for every live
/// destination builder. Reusing those mappings prevents repeated roots or shared descendants from being copied into
/// distinct destination regions merely because they appear in different operation applications.
pub(crate) struct RegionReplayMappings<V: Value, O: Operation<V::Type>> {
    /// Per-destination mappings accumulated during this replay.
    destinations: RefCell<Vec<DestinationRegionMapping<V, O>>>,
}

impl<V: Value, O: Operation<V::Type>> RegionReplayMappings<V, O> {
    /// Creates empty mappings for a new source-arena replay.
    #[inline]
    pub(crate) fn new() -> Self {
        Self { destinations: RefCell::new(Vec::new()) }
    }

    /// Imports `roots` into `builder`, reusing the mapping previously established for that exact builder allocation.
    fn import_into(
        &self,
        builder: &Rc<RefCell<ProgramBuilder<V, O>>>,
        source: RegionRef<'_, V, O>,
        roots: &[RegionId],
    ) -> Result<Vec<RegionId>, ProgramError> {
        let builder_identity = Rc::downgrade(builder);
        let mut destinations = self.destinations.borrow_mut();
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
        roots
            .iter()
            .map(|root| {
                let region = RegionRef::new(source.regions(), *root)?;
                Ok(builder.import_region_with_remapping(region, remapping))
            })
            .collect()
    }
}

/// Source-to-destination [`RegionId`] mapping for one live destination builder participating in a region replay. This
/// is per-destination state because a transform stack may route applications from one source replay into several
/// builders, whose region identifiers are unrelated even when their imported source region is the same.
pub(crate) struct DestinationRegionMapping<V: Value, O: Operation<V::Type>> {
    /// Weak identity of the destination builder. Weak ownership prevents replay bookkeeping from keeping a completed
    /// builder alive or interfering with trace finalization through `Rc::try_unwrap`.
    builder: Weak<RefCell<ProgramBuilder<V, O>>>,

    /// Source-to-destination region identifier remapping for this builder.
    remapping: HashMap<RegionId, RegionId>,
}

/// Identifies one attached-region output that may produce an [`Operation`] output.
///
/// Provenance is relative to an operation application: [`region_index`](Self::region_index) selects an entry from
/// [`Instruction::regions`](crate::Instruction::regions), and [`output_index`](Self::output_index) selects an output
/// of that attached region. Refer to [`Operation::output_region_provenance`] for how operations declare these values.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct OutputRegionProvenance {
    /// Index of the attached region in the operation-defined region order.
    pub region_index: usize,

    /// Index of the output in the attached region's output boundary.
    pub output_index: usize,
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::backends::scalars::Scalar;
    use crate::parameters::Placeholder;
    use crate::programs::{Program, ProgramBuilder, ProgramError};
    use crate::tests::TestRegionOperation;
    use crate::types::DataType;

    use super::*;

    type TestProgram = Program<Scalar, TestRegionOperation, Vec<Scalar>, Vec<Scalar>>;

    fn identity_program(r#type: DataType) -> TestProgram {
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(r#type);
        builder.build(vec![input], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    fn program_with_nested_identity() -> TestProgram {
        let mut builder = ProgramBuilder::new();
        let nested = builder.import_program(identity_program(DataType::F64));
        let input = builder.add_input(DataType::F64);
        let output = builder
            .add_instruction(TestRegionOperation::WithRegions(&["nested"]), vec![input], vec![nested])
            .unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    fn program_with_shared_descendant() -> (TestProgram, RegionId, RegionId) {
        let root_template = program_with_nested_identity();
        let root_region = root_template.entry_region().clone();
        assert_eq!(root_region.instructions()[0].regions(), &[RegionId::new(0)]);

        let mut builder = ProgramBuilder::new();
        assert_eq!(builder.import_program(identity_program(DataType::F64)), RegionId::new(0));
        let first_root = RegionId::new(builder.regions.len());
        builder.regions.push(root_region.clone());
        let second_root = RegionId::new(builder.regions.len());
        builder.regions.push(root_region);

        let input = builder.add_input(DataType::F64);
        let output = builder
            .add_instruction(
                TestRegionOperation::WithRegions(&["first", "second"]),
                vec![input],
                vec![first_root, second_root],
            )
            .unwrap()[0];
        let program = builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        (program, first_root, second_root)
    }

    fn import_attachments<R: BindingRegionDriver<Scalar, TestRegionOperation>>(
        regions: R,
        builder: &Rc<RefCell<ProgramBuilder<Scalar, TestRegionOperation>>>,
    ) -> Vec<RegionId> {
        regions.import_into(builder).unwrap()
    }

    fn attached_input_types<R: RegionDriver<Scalar, TestRegionOperation>>(regions: &R) -> Vec<DataType> {
        regions.regions().map(|region| region.input_types()[0]).collect()
    }

    #[test]
    fn test_owned_region_attachments_support_empty_arrays_and_vectors() {
        let empty_builder = Rc::new(RefCell::new(ProgramBuilder::new()));
        assert!(import_attachments([], &empty_builder).is_empty());
        assert!(empty_builder.borrow().regions.is_empty());

        let array = [identity_program(DataType::F32), identity_program(DataType::F64)];
        assert_eq!(attached_input_types(&array), vec![DataType::F32, DataType::F64]);
        let array_builder = Rc::new(RefCell::new(ProgramBuilder::new()));
        assert_eq!(import_attachments(array, &array_builder), vec![RegionId::new(0), RegionId::new(1)]);
        assert_eq!(array_builder.borrow().regions.len(), 2);

        let regions = vec![identity_program(DataType::F64), identity_program(DataType::F32)];
        assert_eq!(attached_input_types(&regions), vec![DataType::F64, DataType::F32]);
        let vector_builder = Rc::new(RefCell::new(ProgramBuilder::new()));
        assert_eq!(import_attachments(regions, &vector_builder), vec![RegionId::new(0), RegionId::new(1)]);
        assert_eq!(vector_builder.borrow().regions.len(), 2);
    }

    #[test]
    fn test_callee_region_driver_interns_callees() {
        let callee = Rc::new(identity_program(DataType::F64));
        let callees = [Rc::clone(&callee), callee];
        let regions = CalleeRegionDriver::new(&callees);
        assert_eq!(attached_input_types(&regions), vec![DataType::F64, DataType::F64]);

        let builder = Rc::new(RefCell::new(ProgramBuilder::new()));
        assert_eq!(import_attachments(regions, &builder), vec![RegionId::new(0), RegionId::new(0)]);
        assert_eq!(builder.borrow().regions.len(), 1);
    }

    #[test]
    fn test_replay_region_driver_iterates_in_application_order() {
        let (source, first_root, second_root) = program_with_shared_descendant();
        let mappings = RegionReplayMappings::new();
        let roots = [second_root, first_root, second_root];
        let regions = ReplayRegionDriver::new(source.entry_region_ref(), &roots, &mappings).unwrap();

        assert_eq!(regions.regions().map(RegionRef::id).collect::<Vec<_>>(), roots);
    }

    #[test]
    fn test_replay_region_driver_rejects_out_of_range_roots() {
        let source = identity_program(DataType::F64);
        let mappings = RegionReplayMappings::new();
        let roots = [RegionId::new(42)];

        assert!(matches!(
            ReplayRegionDriver::new(source.entry_region_ref(), &roots, &mappings),
            Err(ProgramError::MalformedProgram(message)) if message == "region ^42 is out of range",
        ));
    }

    #[test]
    fn test_replay_import_preserves_duplicate_roots_and_shared_descendants() {
        let (source, first_root, second_root) = program_with_shared_descendant();
        let mappings = RegionReplayMappings::new();

        let duplicate_roots = [first_root, first_root];
        let duplicate_destination = Rc::new(RefCell::new(ProgramBuilder::new()));
        let imported = import_attachments(
            ReplayRegionDriver::new(source.entry_region_ref(), &duplicate_roots, &mappings).unwrap(),
            &duplicate_destination,
        );
        assert_eq!(imported[0], imported[1]);
        assert_eq!(duplicate_destination.borrow().regions.len(), 2);

        let roots = [first_root, second_root];
        let shared_destination = Rc::new(RefCell::new(ProgramBuilder::new()));
        let imported = import_attachments(
            ReplayRegionDriver::new(source.entry_region_ref(), &roots, &mappings).unwrap(),
            &shared_destination,
        );
        let shared_destination = shared_destination.borrow();
        assert_ne!(imported[0], imported[1]);
        assert_eq!(shared_destination.regions.len(), 3);
        assert_eq!(
            shared_destination.region_ref(imported[0]).unwrap().instructions()[0].regions()[0],
            shared_destination.region_ref(imported[1]).unwrap().instructions()[0].regions()[0],
        );
    }

    #[test]
    fn test_replay_import_preserves_sharing_across_applications() {
        let (source, first_root, second_root) = program_with_shared_descendant();
        let mappings = RegionReplayMappings::new();
        let destination = Rc::new(RefCell::new(ProgramBuilder::new()));

        let first_roots = [first_root];
        let first = import_attachments(
            ReplayRegionDriver::new(source.entry_region_ref(), &first_roots, &mappings).unwrap(),
            &destination,
        )[0];
        let second_roots = [second_root];
        let second = import_attachments(
            ReplayRegionDriver::new(source.entry_region_ref(), &second_roots, &mappings).unwrap(),
            &destination,
        )[0];
        let repeated = import_attachments(
            ReplayRegionDriver::new(source.entry_region_ref(), &first_roots, &mappings).unwrap(),
            &destination,
        )[0];

        let destination = destination.borrow();
        assert_eq!(first, repeated);
        assert_eq!(destination.regions.len(), 3);
        assert_eq!(
            destination.region_ref(first).unwrap().instructions()[0].regions()[0],
            destination.region_ref(second).unwrap().instructions()[0].regions()[0],
        );
    }

    #[test]
    fn test_replay_import_uses_independent_destination_mappings() {
        let (source, first_root, _) = program_with_shared_descendant();
        let mappings = RegionReplayMappings::new();
        let first_destination = Rc::new(RefCell::new(ProgramBuilder::new()));
        let second_destination = Rc::new(RefCell::new(ProgramBuilder::new()));
        second_destination.borrow_mut().import_program(identity_program(DataType::F32));
        let roots = [first_root];

        let first = import_attachments(
            ReplayRegionDriver::new(source.entry_region_ref(), &roots, &mappings).unwrap(),
            &first_destination,
        )[0];
        let second = import_attachments(
            ReplayRegionDriver::new(source.entry_region_ref(), &roots, &mappings).unwrap(),
            &second_destination,
        )[0];
        assert_ne!(first, second);
        assert_eq!(first_destination.borrow().regions.len(), 2);
        assert_eq!(second_destination.borrow().regions.len(), 3);
    }

    #[test]
    fn test_replay_mappings_do_not_retain_destinations() {
        let (source, first_root, _) = program_with_shared_descendant();
        let mappings = RegionReplayMappings::new();
        let roots = [first_root];
        let destination = Rc::new(RefCell::new(ProgramBuilder::new()));
        let weak_destination = Rc::downgrade(&destination);

        import_attachments(
            ReplayRegionDriver::new(source.entry_region_ref(), &roots, &mappings).unwrap(),
            &destination,
        );
        assert_eq!(Rc::strong_count(&destination), 1);
        assert_eq!(mappings.destinations.borrow().len(), 1);
        drop(destination);
        assert!(weak_destination.upgrade().is_none());

        let replacement = Rc::new(RefCell::new(ProgramBuilder::new()));
        import_attachments(
            ReplayRegionDriver::new(source.entry_region_ref(), &roots, &mappings).unwrap(),
            &replacement,
        );
        assert_eq!(mappings.destinations.borrow().len(), 1);
    }

    #[test]
    fn test_region_ref_and_to_program() {
        let mut builder = ProgramBuilder::<Scalar, TestRegionOperation>::new();
        let mut region_builder = ProgramBuilder::<Scalar, TestRegionOperation>::new();
        let region_input = region_builder.add_input(DataType::F64);
        let region_program = region_builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![region_input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let sealed = builder.import_region(region_program.entry_region_ref());
        let input = builder.add_input(DataType::F64);
        let first = builder
            .add_instruction(TestRegionOperation::WithRegions(&["body"]), vec![input], vec![sealed])
            .unwrap()[0];
        let second = builder
            .add_instruction(TestRegionOperation::WithRegions(&["body"]), vec![first], vec![sealed])
            .unwrap()[0];
        let program = builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![second], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let entry = program.entry_region_ref();
        assert_eq!(entry.id(), program.entry());
        assert_eq!(entry.input_ids(), program.input_ids());
        assert_eq!(entry.output_ids(), program.output_ids());
        assert_eq!(entry.instructions().len(), 2);
        assert_eq!(program.interface(), entry.interface());
        assert_eq!(entry.interface().output_types(), &[DataType::F64]);
        assert!(matches!(
            program.region_ref(RegionId::new(42)),
            Err(ProgramError::MalformedProgram(message)) if message == "region ^42 is out of range",
        ));

        let detached = entry.to_program();
        assert_eq!(detached.regions().len(), 2);
        assert_eq!(detached.instructions()[0].regions(), detached.instructions()[1].regions());
        assert_eq!(detached.input_types(), vec![DataType::F64]);
        assert_eq!(detached.output_types(), vec![DataType::F64]);
    }
}
