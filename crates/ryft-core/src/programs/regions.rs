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
use std::ops::Index;
use std::rc::{Rc, Weak};

use crate::parameters::Placeholder;
use crate::programs::ProgramError;
use crate::programs::atoms::{Atom, AtomId};
use crate::programs::builders::ProgramBuilder;
use crate::programs::effects::Effects;
use crate::programs::identities::{TypeIdentityPosition, TypeIdentityRenaming, TypeIdentitySignature};
use crate::programs::instructions::Instruction;
use crate::programs::operations::Operation;
use crate::programs::programs::Program;
use crate::programs::types::{Type, TypeError, Typed};
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

    /// Returns this [`Region`] after simultaneously renaming [`TypeIdentity`](crate::TypeIdentity)s in its [`Atom`]s,
    /// constants, and [`Operation`]s/[`Instruction`]s.
    pub fn rename_type_identities(
        &self,
        renaming: &TypeIdentityRenaming<<V::Type as Type>::Identity>,
    ) -> Result<Self, ProgramError>
    where
        V: Value,
        O: Operation<Type = V::Type>,
    {
        let atoms = self
            .atoms
            .iter()
            .map(|atom| match atom {
                Atom::Variable(r#type) => Ok(Atom::Variable(r#type.rename_identities(renaming)?)),
                Atom::Constant(value) => Ok(Atom::Constant(value.rename_type_identities(renaming)?)),
            })
            .collect::<Result<Vec<_>, ProgramError>>()?;
        let instructions = self
            .instructions
            .iter()
            .map(|instruction| {
                Ok(Instruction::new(
                    instruction.operation().rename_type_identities(renaming)?,
                    instruction.inputs().to_vec(),
                    instruction.outputs().to_vec(),
                    instruction.regions().to_vec(),
                ))
            })
            .collect::<Result<Vec<_>, ProgramError>>()?;
        Ok(Self { atoms, input_ids: self.input_ids.clone(), output_ids: self.output_ids.clone(), instructions })
    }

    /// Derives this [`Region`]'s closed [`TypeIdentitySignature`]. A definition-position constant establishes one
    /// immutable internal identity. A result definition whose [`TypeIdentity`](crate::TypeIdentity) occurs on an
    /// input/operand forwards that available identity, while any other result definition establishes one fresh internal
    /// identity. Every result reference must either forward an operand identity or refer to a definition-position
    /// occurrence on a sibling result.
    pub(crate) fn type_identity_signature(
        &self,
    ) -> Result<TypeIdentitySignature<<V::Type as Type>::Identity>, TypeError>
    where
        O: Operation<Type = V::Type>,
    {
        // Region inputs establish the identities that are available at entry. Preserve first-occurrence order and
        // record each identity once even when several input types refer to the same dynamic quantity. Note that the
        // complete ordered signature is also the dominance environment while walking the region. Every identity
        // appended after `input_count` is established internally, so one vector records availability and retains
        // the final input/internal partition without copying identities into parallel collections.
        let mut identities = Vec::new();
        for input in &self.input_ids {
            for (_, identity) in self.atoms[input.index()].r#type().identities() {
                if !identities.contains(identity) {
                    identities.push(identity.clone());
                }
            }
        }
        let input_count = identities.len();

        // A definition-position occurrence on a constant establishes one immutable Single Static Assignment (SSA) value
        // just like a definition-position instruction result. Process all such definitions before constant references
        // so the atom-table order does not impose an artificial dominance relationship among constants.
        self.atoms.iter().filter_map(Atom::as_constant).try_for_each(|value| {
            let r#type = value.r#type();
            r#type.identities().try_for_each(|(position, identity)| {
                if position != TypeIdentityPosition::Definition {
                    return Ok(());
                }
                if identities.contains(identity) {
                    return Err(TypeError::invalid(format!(
                        "constant type defines identity {identity} more than once in this region",
                    )));
                }
                identities.push(identity.clone());
                Ok(())
            })
        })?;

        // Reference-position constant metadata must refer to an identity supplied by the boundary or defined by a
        // constant. Constants cannot depend on instruction results because all constants exist before execution.
        self.atoms.iter().filter_map(Atom::as_constant).try_for_each(|value| {
            let r#type = value.r#type();
            r#type.identities().try_for_each(|(position, identity)| {
                if position != TypeIdentityPosition::Reference {
                    return Ok(());
                }
                if identities.contains(identity) {
                    Ok(())
                } else {
                    Err(TypeError::invalid(format!(
                        "constant type references identity {identity} which is not established by a region input",
                    )))
                }
            })
        })?;

        // Instructions are already in evaluation order, so processing them sequentially turns availability into a
        // simple dominance check: operands may use only identities established by the boundary or an earlier result.
        self.instructions.iter().try_for_each(|instruction| {
            // Collect the identities consumed by this instruction while rejecting any operand reference that appears
            // before its definition. Result occurrences matching these identities are forwarders, not new definitions.
            let mut operand_identities = Vec::new();
            instruction.inputs().iter().try_for_each(|input| {
                let r#type = self.atoms[input.index()].r#type();
                r#type.identities().try_for_each(|(_, identity)| {
                    if !identities.contains(identity) {
                        return Err(TypeError::invalid(format!(
                            "operation `{}` input type references identity {} before its definition",
                            instruction.operation().name(),
                            identity,
                        )));
                    }
                    if !operand_identities.contains(identity) {
                        operand_identities.push(identity.clone());
                    }
                    Ok(())
                })
            })?;

            // Process explicit definition-position occurrences first. A definition also present on an operand forwards
            // that identity. Every other definition establishes one fresh internal identity, which must not have been
            // established earlier or repeated by another result of the same instruction.
            let mut defined_identities = Vec::new();
            instruction.outputs().iter().try_for_each(|output| {
                let r#type = self.atoms[output.index()].r#type();
                r#type.identities().try_for_each(|(position, identity)| {
                    if position != TypeIdentityPosition::Definition || operand_identities.contains(identity) {
                        return Ok(());
                    }
                    if identities.contains(identity) || defined_identities.contains(identity) {
                        return Err(TypeError::invalid(format!(
                            "operation `{}` output defines identity {} more than once in this region",
                            instruction.operation().name(),
                            identity,
                        )));
                    }
                    identities.push(identity.clone());
                    defined_identities.push(identity.clone());
                    Ok(())
                })
            })?;

            // Validate reference-position result occurrences after all sibling definitions are known. A reference must
            // either be forwarded from an operand or refer to an identity defined by this instruction.
            instruction.outputs().iter().try_for_each(|output| {
                let r#type = self.atoms[output.index()].r#type();
                r#type.identities().try_for_each(|(position, identity)| {
                    if position != TypeIdentityPosition::Reference {
                        return Ok(());
                    }
                    if !operand_identities.contains(identity) && !defined_identities.contains(identity) {
                        return Err(TypeError::invalid(format!(
                            "operation `{}` output type references identity {} without consuming or defining it",
                            instruction.operation().name(),
                            identity,
                        )));
                    }
                    Ok(())
                })
            })
        })?;

        // The resulting signature transfers the dominance environment directly (caller-established identities occupy
        // its prefix and identities established within the region occupy its suffix).
        Ok(TypeIdentitySignature::new(identities, input_count))
    }
}

/// Sealed [`Region`] stored together with metadata derived from its immutable contents and already-sealed descendants.
/// Keeping the metadata in the same arena entry makes it impossible for a published [`RegionArena`] to pair a region
/// with stale [`Effects`] or with a [`TypeIdentitySignature`] derived from a different region.
#[derive(Clone, Debug)]
pub struct RegionWithMetadata<V: Typed, O> {
    /// Immutable computation [`Region`].
    region: Region<V, O>,

    /// Recursively derived observable [`Effects`] of `region`.
    effects: Effects,

    /// Structurally closed [`TypeIdentitySignature`] of `region`.
    type_identity_signature: TypeIdentitySignature<<V::Type as Type>::Identity>,
}

impl<V: Value, O: Operation<Type = V::Type>> RegionWithMetadata<V, O> {
    /// Creates a new [`RegionWithMetadata`] by _sealing_ the provided `region` after deriving its metadata against
    /// `sealed_regions`, which must contain every region it references.
    pub fn new(region: Region<V, O>, sealed_regions: &[Self]) -> Result<Self, ProgramError> {
        region
            .input_ids
            .iter()
            .chain(&region.output_ids)
            .chain(
                region
                    .instructions
                    .iter()
                    .flat_map(|instruction| instruction.inputs().iter().chain(instruction.outputs())),
            )
            .copied()
            .try_for_each(|id| region.atoms.get(id.index()).map(|_| ()).ok_or(ProgramError::UnboundAtomId { id }))?;
        let effects = region.instructions.iter().try_fold(Effects::PURE, |effects, instruction| {
            let region_slots = instruction.operation().region_slots();
            let declared_region_count = region_slots.len();
            if instruction.regions().len() != declared_region_count {
                return Err(ProgramError::MalformedProgram(format!(
                    "operation `{}` declares {} region slots but {} regions were attached",
                    instruction.operation().name(),
                    declared_region_count,
                    instruction.regions().len(),
                )));
            }
            instruction.regions().iter().copied().enumerate().try_fold(
                effects.union(instruction.operation().effects()),
                |effects, (region_index, nested_region)| {
                    let nested_effects = sealed_regions
                        .get(nested_region.index())
                        .map(|nested_region| nested_region.effects)
                        .ok_or_else(|| {
                            ProgramError::MalformedProgram(format!(
                                "instruction references region {nested_region} which has not been sealed yet",
                            ))
                        })?;

                    // Only computation regions may execute as part of the owning operation, so their effects are
                    // observable and must propagate outward. Rule regions are dormant transform definitions. Merely
                    // attaching one does not execute it and therefore must not make the owning computation effectful.
                    Ok(if region_slots[region_index].role == RegionRole::Computation {
                        effects.union(nested_effects)
                    } else {
                        effects
                    })
                },
            )
        })?;
        let type_identity_signature = region.type_identity_signature()?;
        Ok(Self { region, effects, type_identity_signature })
    }
}

/// Append-only arena of sealed [`Region`]s and their immutable derived metadata. [`RegionId`]s are stable indexes into
/// this arena. A region may reference only entries that precede it, allowing construction to validate and derive
/// recursive metadata like [`Effects`] and [`TypeIdentitySignature`]s in one ascending pass. Built [`Program`]s retain
/// this arena without exposing mutable access to either regions or metadata.
#[derive(Clone, Debug)]
pub struct RegionArena<V: Typed, O> {
    /// Sealed [`Region`]s and their derived metadata, in [`RegionId`] order.
    regions: Vec<RegionWithMetadata<V, O>>,
}

impl<V: Value, O: Operation<Type = V::Type>> RegionArena<V, O> {
    /// Creates an empty [`RegionArena`].
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new [`RegionArena`] by sealing the provided `regions` in order.
    #[inline]
    pub fn from_regions(regions: Vec<Region<V, O>>) -> Result<Self, ProgramError> {
        regions.into_iter().try_fold(Self::new(), |mut arena, region| {
            arena.push(region)?;
            Ok(arena)
        })
    }

    /// Returns the number of sealed [`Region`]s in this [`RegionArena`].
    #[inline]
    pub fn len(&self) -> usize {
        self.regions.len()
    }

    /// Returns whether this [`RegionArena`] contains no [`Region`]s.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.regions.is_empty()
    }

    /// Returns the [`Region`] identified by `id` in this [`RegionArena`], or [`None`] when it is out of range.
    #[inline]
    pub fn get(&self, id: RegionId) -> Option<&Region<V, O>> {
        self.regions.get(id.index()).map(|region| &region.region)
    }

    /// Returns an [`Iterator`] over the sealed [`Region`]s in this [`RegionArena`] in [`RegionId`] order.
    #[inline]
    pub fn iter(&self) -> RegionArenaIterator<'_, V, O> {
        self.into_iter()
    }

    /// Returns the [`Effects`] of the [`Region`] with the provided [`RegionId`] in this [`RegionArena`].
    #[inline]
    pub fn effects(&self, id: RegionId) -> Option<Effects> {
        self.regions.get(id.index()).map(|region| region.effects)
    }

    /// Returns the [`TypeIdentitySignature`] of the [`Region`] with the provided [`RegionId`] in this [`RegionArena`].
    #[inline]
    pub fn type_identity_signature(&self, id: RegionId) -> Option<&TypeIdentitySignature<<V::Type as Type>::Identity>> {
        self.regions.get(id.index()).map(|region| &region.type_identity_signature)
    }

    /// Consumes this arena and returns its [`Region`]s in [`RegionId`] order.
    #[inline]
    pub fn into_regions(self) -> Vec<Region<V, O>> {
        self.regions.into_iter().map(|region| region.region).collect()
    }

    /// Seals and appends the provided [`Region`] to this [`RegionArena`], returning its stable identifier.
    pub fn push(&mut self, region: Region<V, O>) -> Result<RegionId, ProgramError> {
        let id = RegionId::new(self.regions.len());
        self.regions.push(RegionWithMetadata::new(region, self.regions.as_slice())?);
        Ok(id)
    }

    /// Appends every sealed [`Region`] in `other` to this [`RegionArena`], rebasing its internal [`RegionId`]
    /// references by this arena's original length, and returns that offset. The derived metadata remains valid because
    /// rebasing preserves the complete source graph's topology and does not change any region boundary, [`Operation`],
    /// [`Effect`](crate::Effect), or [`TypeIdentity`](crate::TypeIdentity).
    pub fn append(&mut self, mut other: Self) -> usize {
        let offset = self.regions.len();
        for region in &mut other.regions {
            for instruction in &mut region.region.instructions {
                for attached in &mut instruction.regions {
                    *attached = RegionId::new(attached.index() + offset);
                }
            }
        }
        self.regions.append(&mut other.regions);
        offset
    }

    /// Removes and returns the last [`Region`] in this [`RegionArena`].
    #[inline]
    pub fn pop(&mut self) -> Option<Region<V, O>> {
        self.regions.pop().map(|region| region.region)
    }
}

impl<V: Typed, O> Default for RegionArena<V, O> {
    #[inline]
    fn default() -> Self {
        Self { regions: Vec::new() }
    }
}

impl<V: Value, O: Operation<Type = V::Type>> Index<usize> for RegionArena<V, O> {
    type Output = Region<V, O>;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.regions[index].region
    }
}

impl<'r, V: Typed, O> IntoIterator for &'r RegionArena<V, O> {
    type Item = &'r Region<V, O>;
    type IntoIter = RegionArenaIterator<'r, V, O>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        RegionArenaIterator { regions: self.regions.iter() }
    }
}

/// Borrowing [`Iterator`] over the [`Region`]s in a [`RegionArena`], in [`RegionId`] order.
pub struct RegionArenaIterator<'r, V: Typed, O> {
    /// [`Iterator`] over the [`RegionArena`]'s sealed [`Region`] entries.
    regions: std::slice::Iter<'r, RegionWithMetadata<V, O>>,
}

impl<'r, V: Typed, O> Iterator for RegionArenaIterator<'r, V, O> {
    type Item = &'r Region<V, O>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.regions.next().map(|region| &region.region)
    }
}

impl<V: Typed, O> ExactSizeIterator for RegionArenaIterator<'_, V, O> {
    #[inline]
    fn len(&self) -> usize {
        self.regions.len()
    }
}

/// Borrowed view of a [`Region`]. [`RegionRef`] provides [`Program`]-like access to a rooted, nested computation
/// without cloning the owning [`Program`]'s region arena. The intended usage is to store [`RegionId`]s in long-lived
/// Intermediate Representation (IR) objects and recreate [`RegionRef`]s only while inspecting, replaying, importing,
/// or lowering a source arena. Calling [`RegionRef::to_program`] crosses the explicit ownership boundary as it clones
/// the selected region's complete reachable region closure into a detached [`Program`].
#[derive(Debug)]
pub struct RegionRef<'r, V: Value, O: Operation<Type = V::Type>> {
    /// [`RegionArena`] containing the referenced [`Region`] and its reachable descendants.
    arena: &'r RegionArena<V, O>,

    /// Identifier of the rooted [`Region`] within `arena`.
    id: RegionId,
}

impl<'r, V: Value, O: Operation<Type = V::Type>> RegionRef<'r, V, O> {
    /// Creates a new [`RegionRef`] for the [`Region`] with the provided [`RegionId`] in the provided [`RegionArena`].
    #[inline]
    pub fn new(arena: &'r RegionArena<V, O>, id: RegionId) -> Result<Self, ProgramError> {
        arena
            .get(id)
            .map(|_| Self { arena, id })
            .ok_or_else(|| ProgramError::MalformedProgram(format!("region {id} is out of range")))
    }

    /// Returns this [`RegionRef`] with its root changed to [`RegionId`] in the same source [`RegionArena`].
    #[inline]
    pub fn with_id(mut self, id: RegionId) -> Result<Self, ProgramError> {
        if self.arena.get(id).is_none() {
            return Err(ProgramError::MalformedProgram(format!("region {id} is out of range")));
        }
        self.id = id;
        Ok(self)
    }

    /// Returns the [`RegionId`] identifying this rooted [`Region`] in its source arena.
    #[inline]
    pub fn id(self) -> RegionId {
        self.id
    }

    /// Returns the borrowed source arena that contains this [`RegionRef`]'s root and descendants.
    #[inline]
    pub fn arena(self) -> &'r RegionArena<V, O> {
        self.arena
    }

    /// Returns the rooted [`Region`].
    #[inline]
    pub fn region(self) -> &'r Region<V, O> {
        &self.arena[self.id.index()]
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

    /// Returns the observable effects of the [`Instruction`] at `instruction_index`, combining its operation's
    /// intrinsic effects with the recursively derived effects of attached computation regions. Dormant rule regions
    /// are excluded according to their declared [`RegionRole`].
    pub fn instruction_effects(self, instruction_index: usize) -> Result<Effects, ProgramError> {
        let instruction = self.instructions().get(instruction_index).ok_or_else(|| {
            ProgramError::MalformedProgram(format!(
                "instruction index {instruction_index} is out of range for region {}",
                self.id,
            ))
        })?;
        instruction.regions().iter().copied().enumerate().try_fold(
            instruction.operation().effects(),
            |effects, (region_index, attached)| {
                let attached_effects = self
                    .arena
                    .effects(attached)
                    .ok_or_else(|| ProgramError::MalformedProgram(format!("region {attached} is out of range")))?;

                // Include effects from regions the operation may execute during ordinary interpretation. Dormant rule
                // regions are inputs to later transforms rather than executed children of this instruction, so their
                // effects are intentionally excluded.
                Ok(if instruction.operation().region_role(region_index) == Some(RegionRole::Computation) {
                    effects.union(attached_effects)
                } else {
                    effects
                })
            },
        )
    }

    /// Returns the [`RegionInterface`] of the rooted [`Region`].
    #[inline]
    pub fn interface(self) -> RegionInterface<V::Type>
    where
        O: Operation<Type = V::Type>,
    {
        RegionInterface::new(self.input_types(), self.output_types(), self.effects())
    }

    /// Returns the recursively derived [`Effects`] of the rooted [`Region`].
    #[inline]
    pub fn effects(self) -> Effects {
        self.arena.effects(self.id).unwrap()
    }

    /// Returns this [`Region`]'s retained closed [`TypeIdentitySignature`].
    #[inline]
    pub fn type_identity_signature(self) -> &'r TypeIdentitySignature<<V::Type as Type>::Identity> {
        self.arena.type_identity_signature(self.id).unwrap()
    }

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

impl<V: Value, O: Operation<Type = V::Type>> Copy for RegionRef<'_, V, O> {}

impl<V: Value, O: Operation<Type = V::Type>> Clone for RegionRef<'_, V, O> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

/// Semantic role of a slot for a [`Region`] in an [`Operation`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum RegionRole {
    /// The [`Region`] may execute as part of ordinary interpretation, such as a control-flow bodies, callees, or primal
    /// computations. Its recursively derived [`Effects`] are consequently observable effects of the owning operation.
    Computation,

    /// The [`Region`] represents a dormant transformation rule, such as a custom derivative or rematerialization rule.
    /// It is consumed by a transform rather than ordinary interpretation and so its [`Effects`] do not belong to the
    /// owning computation.
    Rule,
}

/// Represents a slot for a [`Region`] in an [`Operation`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct RegionSlot {
    /// Stable [`Region`] name used in diagnostics, rendering, and analysis reports.
    pub name: &'static str,

    /// Semantic [`RegionRole`] of the attached [`Region`].
    pub role: RegionRole,
}

impl RegionSlot {
    /// Creates a [`RegionSlot`] for a [`Region`] that may execute during ordinary interpretation.
    #[inline]
    pub const fn computation(name: &'static str) -> Self {
        Self { name, role: RegionRole::Computation }
    }

    /// Creates a [`RegionSlot`] for a [`Region`] that represents a dormant transformation rule.
    #[inline]
    pub const fn rule(name: &'static str) -> Self {
        Self { name, role: RegionRole::Rule }
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
pub trait RegionDriver<V: Value, O: Operation<Type = V::Type>> {
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
    O: Operation<Type = V::Type>,
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

impl<V: Value, O: Operation<Type = V::Type>> RegionDriver<V, O> for EmptyRegionDriver {
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
pub trait BindingRegionDriver<V: Value, O: Operation<Type = V::Type>>: RegionDriver<V, O> + Sized {
    /// Imports these attached [`Region`]s into the provided [`ProgramBuilder`] in application order and returns their
    /// [`RegionId`]s in the same order. Each type in `input_types` corresponds to the corresponding attached [`Region`]
    /// at that same index and [`None`] preserves its declared input [`TypeIdentity`](crate::TypeIdentity)s, while
    /// [`Some`] instantiates them from the supplied input types.
    fn import_into(
        self,
        builder: &Rc<RefCell<ProgramBuilder<V, O>>>,
        input_types: &[Option<Vec<V::Type>>],
    ) -> Result<Vec<RegionId>, ProgramError>;
}

impl<
    V: Value,
    O: Operation<Type = V::Type>,
    R: AsRef<[Program<V, O, Vec<V>, Vec<V>>]> + IntoIterator<Item = Program<V, O, Vec<V>, Vec<V>>>,
> BindingRegionDriver<V, O> for R
{
    fn import_into(
        self,
        builder: &Rc<RefCell<ProgramBuilder<V, O>>>,
        input_types: &[Option<Vec<V::Type>>],
    ) -> Result<Vec<RegionId>, ProgramError> {
        let region_count = self.as_ref().len();
        if region_count != input_types.len() {
            return Err(ProgramError::MalformedProgram(format!(
                "region type-identity instantiation count {} does not match attached region count {}",
                input_types.len(),
                region_count,
            )));
        }
        let mut builder = builder.borrow_mut();
        self.into_iter()
            .zip(input_types)
            .map(|(program, input_types)| {
                Ok(builder.import_program(match input_types {
                    Some(input_types) => program.with_instantiated_type_identities(input_types)?.into_owned(),
                    None => program,
                }))
            })
            .collect()
    }
}

/// [`BindingRegionDriver`] for shared callee [`Program`]s attached to one [`Context::bind`](crate::Context::bind)
/// [`Operation`] application. Callees are exposed in the order provided at construction and are interned by [`Rc`]
/// identity when imported into a [`StagingContext`](crate::StagingContext), preserving sharing between repeated
/// references to the same program.
pub struct CalleeRegionDriver<'r, V: Value, O: Operation<Type = V::Type>> {
    /// Shared callee [`Program`]s in [`Operation`]-defined region order.
    callees: &'r [Rc<Program<V, O, Vec<V>, Vec<V>>>],
}

impl<'r, V: Value, O: Operation<Type = V::Type>> CalleeRegionDriver<'r, V, O> {
    /// Creates a new [`CalleeRegionDriver`].
    #[inline]
    pub fn new(callees: &'r [Rc<Program<V, O, Vec<V>, Vec<V>>>]) -> Self {
        Self { callees }
    }
}

impl<V: Value, O: Operation<Type = V::Type>> RegionDriver<V, O> for CalleeRegionDriver<'_, V, O> {
    #[inline]
    fn regions<'r>(&'r self) -> impl Iterator<Item = RegionRef<'r, V, O>>
    where
        V: 'r,
        O: 'r,
    {
        self.callees.iter().map(|callee| callee.entry_region_ref())
    }
}

impl<V: Value, O: Operation<Type = V::Type>> BindingRegionDriver<V, O> for CalleeRegionDriver<'_, V, O> {
    fn import_into(
        self,
        builder: &Rc<RefCell<ProgramBuilder<V, O>>>,
        input_types: &[Option<Vec<V::Type>>],
    ) -> Result<Vec<RegionId>, ProgramError> {
        if self.callees.len() != input_types.len() {
            return Err(ProgramError::MalformedProgram(format!(
                "region type-identity instantiation count {} does not match attached region count {}",
                input_types.len(),
                self.callees.len(),
            )));
        }
        let mut builder = builder.borrow_mut();
        self.callees
            .iter()
            .zip(input_types)
            .map(|(callee, input_types)| builder.intern_callee(callee, input_types.as_deref()))
            .collect()
    }
}

/// [`BindingRegionDriver`] for the borrowed [`Region`]s attached to one replayed [`Instruction`]. The roots remain
/// in their source region arena and are exposed in instruction order through [`RegionDriver`]. When a staging context
/// imports them, `mappings` preserves their source identities across every instruction in the surrounding replay.
/// Construction validates that every root belongs to `source`'s arena, which lets [`RegionDriver::regions`] remain
/// non-fallible without trusting callers to preserve that relationship.
pub struct ReplayRegionDriver<'r, V: Value, O: Operation<Type = V::Type>> {
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

impl<'r, V: Value, O: Operation<Type = V::Type>> ReplayRegionDriver<'r, V, O> {
    /// Creates a new [`ReplayRegionDriver`].
    #[inline]
    pub fn new(
        source: RegionRef<'r, V, O>,
        roots: &'r [RegionId],
        mappings: &'r RegionReplayMappings<V, O>,
    ) -> Result<Self, ProgramError> {
        for root in roots {
            source.with_id(*root)?;
        }
        Ok(Self { source, roots, mappings })
    }
}

impl<V: Value, O: Operation<Type = V::Type>> RegionDriver<V, O> for ReplayRegionDriver<'_, V, O> {
    #[inline]
    fn regions<'r>(&'r self) -> impl Iterator<Item = RegionRef<'r, V, O>>
    where
        V: 'r,
        O: 'r,
    {
        self.roots.iter().map(|root| self.source.with_id(*root).unwrap())
    }
}

impl<V: Value, O: Operation<Type = V::Type>> BindingRegionDriver<V, O> for ReplayRegionDriver<'_, V, O> {
    fn import_into(
        self,
        builder: &Rc<RefCell<ProgramBuilder<V, O>>>,
        input_types: &[Option<Vec<V::Type>>],
    ) -> Result<Vec<RegionId>, ProgramError> {
        if self.roots.len() != input_types.len() {
            return Err(ProgramError::MalformedProgram(format!(
                "region type-identity instantiation count {} does not match attached region count {}",
                input_types.len(),
                self.roots.len(),
            )));
        }
        let builder_identity = Rc::downgrade(builder);
        let mut destinations = self.mappings.destinations.borrow_mut();
        destinations.retain(|mapping| mapping.builder.strong_count() > 0);
        let destination_index = destinations
            .iter()
            .position(|mapping| Weak::ptr_eq(&mapping.builder, &builder_identity))
            .unwrap_or_else(|| {
                destinations.push(DestinationRegionMapping {
                    builder: builder_identity,
                    remapping: HashMap::new(),
                    instantiated_region_mappings: Vec::new(),
                });
                destinations.len() - 1
            });
        let destination = &mut destinations[destination_index];
        let mut builder = builder.borrow_mut();
        self.roots
            .iter()
            .zip(input_types)
            .map(|(root, input_types)| {
                let region = self.source.with_id(*root)?;
                let Some(input_types) = input_types else {
                    return Ok(builder.import_region_with_remapping(region, &mut destination.remapping));
                };
                if let Some(mapping) = destination
                    .instantiated_region_mappings
                    .iter()
                    .find(|mapping| &mapping.source_region == root && mapping.input_types == *input_types)
                {
                    return Ok(mapping.destination_region);
                }
                let renaming = V::Type::derive_identity_renaming(region.input_types().as_slice(), input_types)?;
                let imported = if renaming.is_identity() {
                    builder.import_region_with_remapping(region, &mut destination.remapping)
                } else {
                    builder.import_program(region.to_program().rename_type_identities(&renaming)?)
                };
                destination.instantiated_region_mappings.push(InstantiatedRegionMapping {
                    source_region: *root,
                    input_types: input_types.clone(),
                    destination_region: imported,
                });
                Ok(imported)
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
pub struct RegionReplayMappings<V: Value, O: Operation<Type = V::Type>> {
    /// Per-destination [`DestinationRegionMapping`]s accumulated during a replay.
    destinations: RefCell<Vec<DestinationRegionMapping<V, O>>>,
}

impl<V: Value, O: Operation<Type = V::Type>> RegionReplayMappings<V, O> {
    /// Creates a new [`RegionReplayMappings`].
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }
}

impl<V: Value, O: Operation<Type = V::Type>> Default for RegionReplayMappings<V, O> {
    #[inline]
    fn default() -> Self {
        Self { destinations: RefCell::new(Vec::new()) }
    }
}

/// Source-to-destination [`RegionId`] mapping for one live destination [`ProgramBuilder`] participating in a
/// [`Region`] replay. [`RegionReplayMappings`] owns one of these values per destination because [`RegionId`]s are local
/// to their owning arenas. Refer to [`ReplayRegionDriver::mappings`] for more information on how this is used and why
/// it is necessary.
pub struct DestinationRegionMapping<V: Value, O: Operation<Type = V::Type>> {
    /// Weak identity of the destination [`ProgramBuilder`]. Weak ownership prevents replay bookkeeping from keeping
    /// a completed builder alive or interfering with trace finalization through `Rc::try_unwrap`.
    pub builder: Weak<RefCell<ProgramBuilder<V, O>>>,

    /// Source-to-destination [`RegionId`] remapping for the destination [`ProgramBuilder`].
    pub remapping: HashMap<RegionId, RegionId>,

    /// Instantiated source-[`Region`] imports that cannot use `remapping` because one source [`RegionId`] may map
    /// to multiple destination regions under different [`TypeIdentity`](crate::TypeIdentity) instantiations.
    pub instantiated_region_mappings: Vec<InstantiatedRegionMapping<V::Type>>,
}

/// Cached import of one source [`Region`] instantiated for a particular input type signature. A source region can
/// be imported into the same destination [`ProgramBuilder`] under multiple [`TypeIdentity`](crate::TypeIdentity)
/// renamings, so the source [`RegionId`] alone cannot identify the resulting destination region. This mapping retains
/// the complete instantiation key and the corresponding imported root so repeated applications with the same live
/// input [`TypeIdentity`](crate::TypeIdentity)s can preserve region sharing.
pub struct InstantiatedRegionMapping<T: Type> {
    /// Root of the [`Region`] in the replay's source [`RegionArena`].
    pub source_region: RegionId,

    /// Complete actual input [`Type`] signature used to instantiate `source_region`. Exact types are required for
    /// cache reuse because the imported region retains these live [`TypeIdentity`](crate::TypeIdentity)s and attached
    /// [`Instruction`]s do not store a separate per-invocation renaming.
    pub input_types: Vec<T>,

    /// Root of the instantiated [`Region`] in the destination [`ProgramBuilder`]'s [`RegionArena`].
    pub destination_region: RegionId,
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

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::fmt::Display;
    use std::sync::Arc;

    use pretty_assertions::assert_eq;
    use ryft_macros::Parameter;

    use crate::arrays::{ArrayType, DataType, Dimension, DimensionBounds, DimensionVariable, Shape};
    use crate::backends::Array;
    use crate::contexts::EagerContext;
    use crate::parameters::{Parameter, Placeholder};
    use crate::programs::ProgramError;
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::effects::Effect;
    use crate::programs::identities::TypeIdentity;
    use crate::programs::programs::Program;
    use crate::tests::TestRegionOperation;

    use super::*;

    type TestProgram = Program<Array, TestRegionOperation, Vec<Array>, Vec<Array>>;

    /// Nominal identity used by the structural closure prototype.
    #[derive(Clone, Debug, Parameter)]
    struct StructuralIdentity {
        /// Diagnostic name.
        name: &'static str,

        /// Nominal identity token.
        token: Arc<()>,
    }

    impl StructuralIdentity {
        /// Creates a structural test identity.
        fn new(name: &'static str) -> Self {
            Self { name, token: Arc::new(()) }
        }
    }

    impl Display for StructuralIdentity {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str(self.name)
        }
    }

    impl PartialEq for StructuralIdentity {
        fn eq(&self, other: &Self) -> bool {
            Arc::ptr_eq(&self.token, &other.token)
        }
    }

    impl Eq for StructuralIdentity {}

    impl TypeIdentity for StructuralIdentity {
        fn fresh(&self) -> Self {
            Self::new(self.name)
        }
    }

    /// Test type that exposes definition and reference occurrences independently.
    #[derive(Clone, Debug, PartialEq, Eq, Parameter)]
    struct StructuralType {
        /// Identity definitions in positional order.
        definitions: Vec<StructuralIdentity>,

        /// Identity references in positional order.
        references: Vec<StructuralIdentity>,
    }

    impl StructuralType {
        /// Creates a type containing the provided identity occurrences.
        fn new(definitions: Vec<StructuralIdentity>, references: Vec<StructuralIdentity>) -> Self {
            Self { definitions, references }
        }

        /// Creates a type that defines one identity.
        fn definition(identity: StructuralIdentity) -> Self {
            Self::new(vec![identity], Vec::new())
        }

        /// Creates a type that references one identity.
        fn reference(identity: StructuralIdentity) -> Self {
            Self::new(Vec::new(), vec![identity])
        }
    }

    impl Display for StructuralType {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(
                formatter,
                "definitions={:?}, references={:?}",
                self.definitions.iter().map(ToString::to_string).collect::<Vec<_>>(),
                self.references.iter().map(ToString::to_string).collect::<Vec<_>>(),
            )
        }
    }

    impl Type for StructuralType {
        type Identity = StructuralIdentity;
        type Refinements = ();

        fn identities(&self) -> impl Iterator<Item = (TypeIdentityPosition, &Self::Identity)> {
            self.definitions
                .iter()
                .map(|identity| (TypeIdentityPosition::Definition, identity))
                .chain(self.references.iter().map(|identity| (TypeIdentityPosition::Reference, identity)))
        }

        fn rename_identities(&self, renaming: &TypeIdentityRenaming<Self::Identity>) -> Result<Self, TypeError> {
            Ok(Self {
                definitions: self.definitions.iter().map(|identity| renaming.rename(identity)).collect(),
                references: self.references.iter().map(|identity| renaming.rename(identity)).collect(),
            })
        }

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
    }

    impl Typed for StructuralType {
        type Type = Self;

        fn r#type(&self) -> Cow<'_, Self::Type> {
            Cow::Borrowed(self)
        }
    }

    impl Value for StructuralType {
        type DispatchDomain = EagerContext<Self>;
        type ExecutionDomain = EagerContext<Self>;

        fn dispatch_domain(&self) -> Self::DispatchDomain {
            EagerContext::new()
        }

        fn execution_domain(&self) -> Self::ExecutionDomain {
            EagerContext::new()
        }

        fn rename_type_identities(
            &self,
            renaming: &TypeIdentityRenaming<<Self::Type as Type>::Identity>,
        ) -> Result<Self, TypeError> {
            self.rename_identities(renaming)
        }
    }

    /// Minimal operation used by manually assembled structural closure fixtures.
    #[derive(Clone)]
    struct StructuralOperation;

    impl Operation for StructuralOperation {
        type Type = StructuralType;

        fn name(&self) -> &'static str {
            "structural"
        }

        fn infer_output_types(
            &self,
            input_types: &[StructuralType],
            _region_interfaces: &[RegionInterface<StructuralType>],
        ) -> Result<Vec<StructuralType>, TypeError> {
            Ok(input_types.to_vec())
        }
    }

    /// Builds one structurally valid-enough region arena for direct closure testing.
    fn structural_region(
        input_types: Vec<StructuralType>,
        loose_types: Vec<StructuralType>,
        applications: Vec<(Vec<usize>, Vec<StructuralType>)>,
    ) -> Region<StructuralType, StructuralOperation> {
        let input_count = input_types.len();
        let mut atoms = input_types.into_iter().chain(loose_types).map(Atom::Variable).collect::<Vec<_>>();
        let input_ids = (0..input_count).map(AtomId::new).collect();
        let mut instructions = Vec::with_capacity(applications.len());
        let mut output_ids = Vec::new();
        for (inputs, output_types) in applications {
            output_ids = output_types
                .into_iter()
                .map(|r#type| {
                    let output = AtomId::new(atoms.len());
                    atoms.push(Atom::Variable(r#type));
                    output
                })
                .collect();
            instructions.push(Instruction::new(
                StructuralOperation,
                inputs.into_iter().map(AtomId::new).collect(),
                output_ids.clone(),
                Vec::new(),
            ));
        }
        Region { atoms, input_ids, output_ids, instructions }
    }

    /// Test fixture containing two distinct root regions that share one descendant.
    struct SharedDescendantFixture {
        /// Program that owns the fixture's region graph.
        program: TestProgram,

        /// First root region in the shared-descendant graph.
        first_root: RegionId,

        /// Second root region in the shared-descendant graph.
        second_root: RegionId,
    }

    /// Builds an identity program with one input and one output of `r#type`.
    fn identity_program(r#type: ArrayType) -> TestProgram {
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(r#type);
        builder.build(vec![input], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Builds a program whose two instructions attach the same nested identity region.
    fn program_with_reused_region() -> TestProgram {
        let mut builder = ProgramBuilder::new();
        let region = builder.import_program(identity_program(ArrayType::scalar(DataType::F64)));
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let first = builder
            .add_instruction(
                TestRegionOperation::WithRegions(const { &[crate::RegionSlot::computation("body")] }),
                vec![region],
                vec![input],
            )
            .unwrap()[0];
        let second = builder
            .add_instruction(
                TestRegionOperation::WithRegions(const { &[crate::RegionSlot::computation("body")] }),
                vec![region],
                vec![first],
            )
            .unwrap()[0];
        builder.build(vec![second], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Builds a program whose two distinct root regions share one identity-region descendant.
    fn program_with_shared_descendant() -> SharedDescendantFixture {
        let mut root_builder = ProgramBuilder::new();
        let descendant = root_builder.import_program(identity_program(ArrayType::scalar(DataType::F64)));
        let input = root_builder.add_input(ArrayType::scalar(DataType::F64));
        let output = root_builder
            .add_instruction(
                TestRegionOperation::WithRegions(const { &[crate::RegionSlot::computation("nested")] }),
                vec![descendant],
                vec![input],
            )
            .unwrap()[0];
        let root_program: TestProgram = root_builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        let mut root_region = root_program.entry_region().clone();

        let mut builder = ProgramBuilder::new();
        let shared_descendant = builder.import_program(identity_program(ArrayType::scalar(DataType::F64)));
        root_region.instructions[0].regions[0] = shared_descendant;
        let first_root = RegionId::new(builder.regions.len());
        builder.regions.push(root_region.clone()).unwrap();
        let second_root = RegionId::new(builder.regions.len());
        builder.regions.push(root_region).unwrap();

        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder
            .add_instruction(
                TestRegionOperation::WithRegions(
                    const { &[crate::RegionSlot::computation("first"), crate::RegionSlot::computation("second")] },
                ),
                vec![first_root, second_root],
                vec![input],
            )
            .unwrap()[0];
        let program = builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        SharedDescendantFixture { program, first_root, second_root }
    }

    #[test]
    fn test_structural_identity_closure_classifies_forwarded_and_fresh_definitions() {
        let boundary = StructuralIdentity::new("boundary");
        let fresh = StructuralIdentity::new("fresh");

        // Repeated definition-position readers forward an identity because each result consumes that identity.
        let repeated_readers = structural_region(
            vec![StructuralType::reference(boundary.clone())],
            Vec::new(),
            vec![
                (vec![0], vec![StructuralType::definition(boundary.clone())]),
                (vec![1], vec![StructuralType::definition(boundary.clone())]),
            ],
        );
        let signature = repeated_readers.type_identity_signature().unwrap();
        assert_eq!(signature.input_identities(), &[boundary.clone()]);
        assert_eq!(signature.internal_identities(), &[]);

        // A result definition absent from the operands establishes one fresh internal identity.
        let arithmetic = structural_region(
            vec![StructuralType::reference(boundary.clone())],
            Vec::new(),
            vec![(vec![0], vec![StructuralType::definition(fresh.clone())])],
        );
        let signature = arithmetic.type_identity_signature().unwrap();
        assert_eq!(signature.input_identities(), &[boundary]);
        assert_eq!(signature.internal_identities(), &[fresh]);

        // A constant value is also a Single Static Assignment (SSA) definition and may therefore establish
        // its own fresh identity.
        let constant_identity = StructuralIdentity::new("constant");
        let constant: Region<StructuralType, StructuralOperation> = Region {
            atoms: vec![Atom::Constant(StructuralType::definition(constant_identity.clone()))],
            input_ids: Vec::new(),
            output_ids: vec![AtomId::new(0)],
            instructions: Vec::new(),
        };
        let signature = constant.type_identity_signature().unwrap();
        assert_eq!(signature.input_identities(), &[]);
        assert_eq!(signature.internal_identities(), &[constant_identity.clone()]);

        // Constant references may precede their definition in the atom table because constants have no execution
        // order and are all available before the first instruction.
        let constant_reference: Region<StructuralType, StructuralOperation> = Region {
            atoms: vec![
                Atom::Constant(StructuralType::reference(constant_identity.clone())),
                Atom::Constant(StructuralType::definition(constant_identity.clone())),
            ],
            input_ids: Vec::new(),
            output_ids: vec![AtomId::new(0)],
            instructions: Vec::new(),
        };
        let signature = constant_reference.type_identity_signature().unwrap();
        assert_eq!(signature.input_identities(), &[]);
        assert_eq!(signature.internal_identities(), &[constant_identity]);
    }

    #[test]
    fn test_structural_identity_closure_supports_shared_instruction_outputs() {
        let boundary = StructuralIdentity::new("boundary");
        let fresh = StructuralIdentity::new("fresh");
        let region = structural_region(
            vec![StructuralType::reference(boundary)],
            Vec::new(),
            vec![(vec![0], vec![StructuralType::definition(fresh.clone()), StructuralType::reference(fresh.clone())])],
        );
        let signature = region.type_identity_signature().unwrap();
        assert_eq!(signature.internal_identities(), &[fresh]);
    }

    #[test]
    fn test_structural_identity_closure_rejects_invalid_dominance_and_ownership() {
        let boundary = StructuralIdentity::new("boundary");
        let fresh = StructuralIdentity::new("fresh");

        let duplicate_definition = structural_region(
            vec![StructuralType::reference(boundary.clone())],
            Vec::new(),
            vec![(vec![0], vec![StructuralType::definition(fresh.clone()), StructuralType::definition(fresh.clone())])],
        );
        assert!(matches!(
            duplicate_definition.type_identity_signature(),
            Err(TypeError::Invalid { message })
                if message == "operation `structural` output defines identity fresh more than once in this region",
        ));

        let reference_before_definition = structural_region(
            vec![StructuralType::reference(boundary.clone())],
            vec![StructuralType::reference(fresh.clone())],
            vec![(vec![1], Vec::new())],
        );
        assert!(matches!(
            reference_before_definition.type_identity_signature(),
            Err(TypeError::Invalid { message })
                if message == "operation `structural` input type references identity fresh before its definition",
        ));

        let unrelated_reference = structural_region(
            vec![StructuralType::reference(boundary.clone())],
            Vec::new(),
            vec![(Vec::new(), vec![StructuralType::reference(boundary.clone())])],
        );
        assert!(matches!(
            unrelated_reference.type_identity_signature(),
            Err(TypeError::Invalid { message })
                if message
                    == "operation `structural` output type references identity boundary without consuming or defining it",
        ));

        let fresh_reference = structural_region(
            vec![StructuralType::reference(boundary.clone())],
            Vec::new(),
            vec![(Vec::new(), vec![StructuralType::reference(fresh.clone())])],
        );

        assert!(matches!(
            fresh_reference.type_identity_signature(),
            Err(TypeError::Invalid { message })
                if message
                    == "operation `structural` output type references identity fresh without consuming or defining it",
        ));

        let duplicate_constant_definition: Region<StructuralType, StructuralOperation> = Region {
            atoms: vec![
                Atom::Constant(StructuralType::definition(fresh.clone())),
                Atom::Constant(StructuralType::definition(fresh.clone())),
            ],
            input_ids: Vec::new(),
            output_ids: Vec::new(),
            instructions: Vec::new(),
        };
        assert!(matches!(
            duplicate_constant_definition.type_identity_signature(),
            Err(TypeError::Invalid { message })
                if message == "constant type defines identity fresh more than once in this region",
        ));

        let constant_reference: Region<StructuralType, StructuralOperation> = Region {
            atoms: vec![
                Atom::Variable(StructuralType::reference(boundary)),
                Atom::Constant(StructuralType::reference(fresh)),
            ],
            input_ids: vec![AtomId::new(0)],
            output_ids: Vec::new(),
            instructions: Vec::new(),
        };
        assert!(matches!(
            constant_reference.type_identity_signature(),
            Err(TypeError::Invalid { message })
                if message == "constant type references identity fresh which is not established by a region input",
        ));
    }

    #[test]
    fn test_region_ref() {
        let program = program_with_reused_region();
        let region = program.entry_region_ref();
        assert_eq!(region.id(), program.entry());
        assert_eq!(region.arena().len(), program.regions().len());
        assert_eq!(region.atoms().len(), program.atoms().len());
        assert_eq!(region.input_ids(), program.input_ids());
        assert_eq!(region.input_types(), vec![ArrayType::scalar(DataType::F64)]);
        assert_eq!(region.output_ids(), program.output_ids());
        assert_eq!(region.output_types(), vec![ArrayType::scalar(DataType::F64)]);
        assert_eq!(region.instructions().len(), 2);
        let interface = region.interface();
        assert_eq!(interface.input_types(), &[ArrayType::scalar(DataType::F64)]);
        assert_eq!(interface.output_types(), &[ArrayType::scalar(DataType::F64)]);
        assert_eq!(interface.effects(), Effects::PURE);
    }

    #[test]
    fn test_region_arena_retains_derived_metadata() {
        let mut body_builder = ProgramBuilder::<Array, TestRegionOperation>::new();
        let body_input = body_builder.add_input(ArrayType::scalar(DataType::F64));
        let body_output =
            body_builder.add_instruction(TestRegionOperation::Effectful, Vec::new(), vec![body_input]).unwrap()[0];
        let body = body_builder
            .build::<Vec<Array>, Vec<Array>>(vec![body_output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<Array, TestRegionOperation>::new();
        let body = builder.import_program(body);
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder
            .add_instruction(
                TestRegionOperation::WithRegions(const { &[crate::RegionSlot::computation("body")] }),
                vec![body],
                vec![input],
            )
            .unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();

        let arena = program.regions();
        assert_eq!(arena.len(), 2);
        assert_eq!(arena.iter().count(), 2);
        assert_eq!(arena[body.index()].input_types(), vec![ArrayType::scalar(DataType::F64)]);
        assert_eq!(program.region_ref(body).unwrap().effects(), Effects::single(Effect::OrderedIo));
        assert_eq!(program.entry_region_ref().effects(), Effects::single(Effect::OrderedIo));
        assert!(std::ptr::eq(
            program.entry_region_ref().type_identity_signature(),
            program.entry_region_ref().type_identity_signature(),
        ));
    }

    #[test]
    fn test_region_arena_rejects_unsealed_region_reference() {
        let atom = AtomId::new(0);
        let region: Region<Array, TestRegionOperation> = Region {
            atoms: vec![Atom::Variable(ArrayType::scalar(DataType::F64))],
            input_ids: vec![atom],
            output_ids: vec![atom],
            instructions: vec![Instruction::new(
                TestRegionOperation::WithRegions(const { &[crate::RegionSlot::computation("body")] }),
                vec![atom],
                vec![atom],
                vec![RegionId::new(0)],
            )],
        };
        assert!(matches!(
            RegionArena::from_regions(vec![region]),
            Err(ProgramError::MalformedProgram(message))
                if message == "instruction references region ^0 which has not been sealed yet",
        ));
    }

    #[test]
    fn test_region_ref_with_id() {
        let program = program_with_reused_region();
        let entry = program.entry_region_ref();
        let nested_id = entry.instructions()[0].regions()[0];
        let nested = entry.with_id(nested_id).unwrap();
        assert_eq!(nested.id(), nested_id);
        assert!(std::ptr::eq(nested.arena(), entry.arena()));
        assert_eq!(nested.input_types(), vec![ArrayType::scalar(DataType::F64)]);
        assert_eq!(nested.output_types(), vec![ArrayType::scalar(DataType::F64)]);
        assert!(matches!(
            entry.with_id(RegionId::new(42)),
            Err(ProgramError::MalformedProgram(message)) if message == "region ^42 is out of range",
        ));
    }

    #[test]
    fn test_region_ref_rejects_out_of_range_id() {
        let program = identity_program(ArrayType::scalar(DataType::F64));
        assert!(matches!(
            RegionRef::new(program.regions(), RegionId::new(42)),
            Err(ProgramError::MalformedProgram(message)) if message == "region ^42 is out of range",
        ));
    }

    #[test]
    fn test_region_ref_to_program() {
        let program = program_with_reused_region();
        let materialized = program.entry_region_ref().to_program();
        assert_eq!(materialized.regions().len(), 2);
        assert_eq!(materialized.instructions()[0].regions(), materialized.instructions()[1].regions());
        assert_eq!(materialized.input_types(), vec![ArrayType::scalar(DataType::F64)]);
        assert_eq!(materialized.output_types(), vec![ArrayType::scalar(DataType::F64)]);
    }

    #[test]
    fn test_binding_region_driver_for_owned_collections() {
        let empty_builder = Rc::new(RefCell::new(ProgramBuilder::new()));
        let empty_driver: [TestProgram; 0] = [];
        assert_eq!(empty_driver.import_into(&empty_builder, &[]), Ok(Vec::new()));
        assert!(empty_builder.borrow().regions.is_empty());
        let array_driver =
            [identity_program(ArrayType::scalar(DataType::F32)), identity_program(ArrayType::scalar(DataType::F64))];
        assert_eq!(
            array_driver.regions().map(|region| region.input_types()[0].clone()).collect::<Vec<_>>(),
            vec![ArrayType::scalar(DataType::F32), ArrayType::scalar(DataType::F64)],
        );
        let array_builder = Rc::new(RefCell::new(ProgramBuilder::new()));
        assert_eq!(
            array_driver.import_into(&array_builder, &[None, None]),
            Ok(vec![RegionId::new(0), RegionId::new(1)]),
        );
        assert_eq!(array_builder.borrow().regions.len(), 2);

        let vector_driver = vec![
            identity_program(ArrayType::scalar(DataType::F64)),
            identity_program(ArrayType::scalar(DataType::F32)),
        ];
        assert_eq!(
            vector_driver.regions().map(|region| region.input_types()[0].clone()).collect::<Vec<_>>(),
            vec![ArrayType::scalar(DataType::F64), ArrayType::scalar(DataType::F32)],
        );
        let vector_builder = Rc::new(RefCell::new(ProgramBuilder::new()));
        assert_eq!(
            vector_driver.import_into(&vector_builder, &[None, None]),
            Ok(vec![RegionId::new(0), RegionId::new(1)]),
        );
        assert_eq!(vector_builder.borrow().regions.len(), 2);
    }

    #[test]
    fn test_callee_region_driver() {
        let callee = Rc::new(identity_program(ArrayType::scalar(DataType::F64)));
        let callees = [Rc::clone(&callee), callee];
        let driver = CalleeRegionDriver::new(&callees);
        assert_eq!(
            driver.regions().map(|region| region.input_types()[0].clone()).collect::<Vec<_>>(),
            vec![ArrayType::scalar(DataType::F64), ArrayType::scalar(DataType::F64)],
        );
        let builder = Rc::new(RefCell::new(ProgramBuilder::new()));
        assert_eq!(driver.import_into(&builder, &[None, None]), Ok(vec![RegionId::new(0), RegionId::new(0)]));
        assert_eq!(builder.borrow().regions.len(), 1);
    }

    #[test]
    fn test_replay_region_driver() {
        let SharedDescendantFixture { program, first_root, second_root } = program_with_shared_descendant();
        let mappings = RegionReplayMappings::new();
        let roots = [second_root, first_root, second_root];
        let driver = ReplayRegionDriver::new(program.entry_region_ref(), &roots, &mappings).unwrap();
        assert_eq!(driver.regions().map(RegionRef::id).collect::<Vec<_>>(), roots);
    }

    #[test]
    fn test_replay_region_driver_rejects_out_of_range_root() {
        let program = identity_program(ArrayType::scalar(DataType::F64));
        let mappings = RegionReplayMappings::new();
        let roots = [RegionId::new(42)];
        assert!(matches!(
            ReplayRegionDriver::new(program.entry_region_ref(), &roots, &mappings),
            Err(ProgramError::MalformedProgram(message)) if message == "region ^42 is out of range",
        ));
    }

    #[test]
    fn test_replay_region_driver_import_preserves_duplicate_roots() {
        let SharedDescendantFixture { program, first_root, .. } = program_with_shared_descendant();
        let mappings = RegionReplayMappings::new();
        let roots = [first_root, first_root];
        let driver = ReplayRegionDriver::new(program.entry_region_ref(), &roots, &mappings).unwrap();
        let destination = Rc::new(RefCell::new(ProgramBuilder::new()));
        assert_eq!(driver.import_into(&destination, &[None, None]), Ok(vec![RegionId::new(1), RegionId::new(1)]),);
        assert_eq!(destination.borrow().regions.len(), 2);
    }

    #[test]
    fn test_replay_region_driver_instantiation_cache_preserves_live_type_identities() {
        #[derive(Clone)]
        struct ArrayIdentityOperation;

        impl Operation for ArrayIdentityOperation {
            type Type = ArrayType;

            fn name(&self) -> &'static str {
                "array_identity"
            }

            fn infer_output_types(
                &self,
                input_types: &[ArrayType],
                _region_interfaces: &[RegionInterface<ArrayType>],
            ) -> Result<Vec<ArrayType>, TypeError> {
                Ok(input_types.to_vec())
            }
        }

        let bounds = DimensionBounds::non_negative(Some(16)).unwrap();
        let array_type =
            |variable: DimensionVariable| ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(variable)]));
        let formal = DimensionVariable::new("formal", bounds);
        let mut source_builder = ProgramBuilder::<Array, ArrayIdentityOperation>::new();
        let input = source_builder.add_input(array_type(formal));
        let source = source_builder
            .build::<Vec<Array>, Vec<Array>>(vec![input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let roots = [source.entry()];
        let mappings = RegionReplayMappings::new();
        let destination = Rc::new(RefCell::new(ProgramBuilder::new()));

        let first_input_types = vec![array_type(DimensionVariable::new("first", bounds))];
        let first_driver = ReplayRegionDriver::new(source.entry_region_ref(), &roots, &mappings).unwrap();
        let first = first_driver.import_into(&destination, &[Some(first_input_types.clone())]).unwrap()[0];

        let second_input_types = vec![array_type(DimensionVariable::new("second", bounds))];
        let second_driver = ReplayRegionDriver::new(source.entry_region_ref(), &roots, &mappings).unwrap();
        let second = second_driver.import_into(&destination, &[Some(second_input_types.clone())]).unwrap()[0];
        let repeated_driver = ReplayRegionDriver::new(source.entry_region_ref(), &roots, &mappings).unwrap();
        let repeated = repeated_driver.import_into(&destination, &[Some(second_input_types.clone())]).unwrap()[0];

        assert_ne!(first, second);
        assert_eq!(second, repeated);
        let destination = destination.borrow();
        assert_eq!(destination.region_ref(first).unwrap().input_types(), first_input_types);
        assert_eq!(destination.region_ref(second).unwrap().input_types(), second_input_types);
    }

    #[test]
    fn test_replay_region_driver_import_preserves_shared_descendants() {
        let SharedDescendantFixture { program, first_root, second_root } = program_with_shared_descendant();
        let mappings = RegionReplayMappings::new();
        let roots = [first_root, second_root];
        let driver = ReplayRegionDriver::new(program.entry_region_ref(), &roots, &mappings).unwrap();
        let destination = Rc::new(RefCell::new(ProgramBuilder::new()));
        assert_eq!(driver.import_into(&destination, &[None, None]), Ok(vec![RegionId::new(1), RegionId::new(2)]),);
        let destination = destination.borrow();
        assert_eq!(destination.regions.len(), 3);
        assert_eq!(destination.region_ref(RegionId::new(1)).unwrap().instructions()[0].regions(), &[RegionId::new(0)]);
        assert_eq!(destination.region_ref(RegionId::new(2)).unwrap().instructions()[0].regions(), &[RegionId::new(0)]);
    }

    #[test]
    fn test_replay_region_driver_import_preserves_sharing_across_applications() {
        let SharedDescendantFixture { program, first_root, second_root } = program_with_shared_descendant();
        let mappings = RegionReplayMappings::new();
        let destination = Rc::new(RefCell::new(ProgramBuilder::new()));
        let first_roots = [first_root];
        let first_driver = ReplayRegionDriver::new(program.entry_region_ref(), &first_roots, &mappings).unwrap();
        assert_eq!(first_driver.import_into(&destination, &[None]), Ok(vec![RegionId::new(1)]));
        let second_roots = [second_root];
        let second_driver = ReplayRegionDriver::new(program.entry_region_ref(), &second_roots, &mappings).unwrap();
        assert_eq!(second_driver.import_into(&destination, &[None]), Ok(vec![RegionId::new(2)]));
        let repeated_driver = ReplayRegionDriver::new(program.entry_region_ref(), &first_roots, &mappings).unwrap();
        assert_eq!(repeated_driver.import_into(&destination, &[None]), Ok(vec![RegionId::new(1)]));
        let destination = destination.borrow();
        assert_eq!(destination.regions.len(), 3);
        assert_eq!(destination.region_ref(RegionId::new(1)).unwrap().instructions()[0].regions(), &[RegionId::new(0)]);
        assert_eq!(destination.region_ref(RegionId::new(2)).unwrap().instructions()[0].regions(), &[RegionId::new(0)]);
    }

    #[test]
    fn test_replay_region_driver_import_uses_destination_specific_mappings() {
        let SharedDescendantFixture { program, first_root, .. } = program_with_shared_descendant();
        let mappings = RegionReplayMappings::new();
        let first_destination = Rc::new(RefCell::new(ProgramBuilder::new()));
        let second_destination = Rc::new(RefCell::new(ProgramBuilder::new()));
        second_destination.borrow_mut().import_program(identity_program(ArrayType::scalar(DataType::F32)));
        let roots = [first_root];
        let first_driver = ReplayRegionDriver::new(program.entry_region_ref(), &roots, &mappings).unwrap();
        assert_eq!(first_driver.import_into(&first_destination, &[None]), Ok(vec![RegionId::new(1)]));
        let second_driver = ReplayRegionDriver::new(program.entry_region_ref(), &roots, &mappings).unwrap();
        assert_eq!(second_driver.import_into(&second_destination, &[None]), Ok(vec![RegionId::new(2)]));
        assert_eq!(first_destination.borrow().regions.len(), 2);
        assert_eq!(second_destination.borrow().regions.len(), 3);
    }

    #[test]
    fn test_region_replay_mappings_do_not_retain_destinations() {
        let SharedDescendantFixture { program, first_root, .. } = program_with_shared_descendant();
        let mappings = RegionReplayMappings::new();
        let roots = [first_root];
        let destination = Rc::new(RefCell::new(ProgramBuilder::new()));
        let weak_destination = Rc::downgrade(&destination);
        let driver = ReplayRegionDriver::new(program.entry_region_ref(), &roots, &mappings).unwrap();
        assert_eq!(driver.import_into(&destination, &[None]), Ok(vec![RegionId::new(1)]));
        assert_eq!(Rc::strong_count(&destination), 1);
        assert_eq!(mappings.destinations.borrow().len(), 1);
        drop(destination);
        assert!(weak_destination.upgrade().is_none());
        let replacement = Rc::new(RefCell::new(ProgramBuilder::new()));
        let replacement_driver = ReplayRegionDriver::new(program.entry_region_ref(), &roots, &mappings).unwrap();
        assert_eq!(replacement_driver.import_into(&replacement, &[None]), Ok(vec![RegionId::new(1)]));
        assert_eq!(mappings.destinations.borrow().len(), 1);
    }
}
