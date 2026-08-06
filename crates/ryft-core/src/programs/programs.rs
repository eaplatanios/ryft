use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;

use crate::contexts::Domain;
use crate::macros::check_count;
use crate::parameters::{Parameter, ParameterError, Parameterized, ParameterizedFamily, Placeholder};
use crate::programs::ProgramError;
use crate::programs::atoms::{Atom, AtomId};
use crate::programs::effects::Effects;
use crate::programs::identities::{TypeIdentityRenaming, TypeIdentitySignature};
use crate::programs::instructions::{Instruction, InstructionId};
use crate::programs::operations::Operation;
use crate::programs::regions::{Region, RegionArena, RegionId, RegionInterface, RegionRef};
use crate::programs::types::{Type, Typed};
use crate::programs::values::{Value, ValueId, ValueProjection};

/// [`Program`] that is produced by tracing and which can be interpreted or compiled and executed by a backend.
/// A program owns a flat arena of [`Region`]s. One region implements its public entry point, and every other region
/// is a nested computation referenced by one or more [`Instruction`]s (e.g., the branches of a condition, or the
/// shared program of a JIT call). Each region is a flat sequence of [`Instruction`]s over its own [`Atom`] table, and the
/// entry region's flat boundary is paired with [`Parameterized`] input and output types. This is the primary
/// intermediate representation (IR) used by the Ryft tracing and transformation system (e.g., to support things
/// like automatic differentiation and just-in-time compilation).
#[derive(Debug)]
pub struct Program<V: Typed + Parameter, O, Input: Parameterized<V>, Output: Parameterized<V>> {
    /// [`Parameter`] structure that can be used to map flat lists of inputs to structured `Input` values.
    pub(crate) input_structure: Input::ParameterStructure,

    /// [`Parameter`] structure that can be used to map flat lists of outputs to structured `Output` values.
    pub(crate) output_structure: Output::ParameterStructure,

    /// Validated [`Region`] arena containing the public entry computation and every nested computation.
    pub(crate) regions: RegionArena<V, O>,

    /// [`RegionId`] of the [`Region`] implementing this [`Program`]'s public entry point.
    pub(crate) entry: RegionId,

    /// [`PhantomData`] marker that ties this [`Program`] to its structured `Input` and `Output` types
    /// without making it own either value family.
    pub(crate) marker: PhantomData<(Input, Output)>,
}

impl<V: Value, O: Operation<Type = V::Type>, Input: Parameterized<V>, Output: Parameterized<V>>
    Program<V, O, Input, Output>
{
    /// Creates a new [`Program`] containing the provided [`Region`]s after validating them
    /// and their structural ordering.
    pub fn new(
        input_structure: Input::ParameterStructure,
        output_structure: Output::ParameterStructure,
        regions: Vec<Region<V, O>>,
        entry: RegionId,
    ) -> Result<Self, ProgramError> {
        // Seal regions in arena order first. Sealing validates atom and attached-region references while deriving the
        // immutable recursive metadata used by effect analysis and type-identity instantiation. Because a region may
        // reference only previously sealed regions, this also establishes the arena's descendant-before-parent order.
        let regions = RegionArena::from_regions(regions)?;

        // The entry identifier is supplied separately from the arena, so validate it before using the entry boundary
        // to establish the structured program signature.
        let Some(entry_region) = regions.get(entry) else {
            return Err(ProgramError::MalformedProgram(format!("entry region {entry} is out of range")));
        };

        // The parameter structures describe the public structured signature, while the entry region stores its flat
        // atom boundary. Their leaf counts must agree so flattening inputs and reconstructing outputs are total.
        check_count!("input", entry_region.input_ids(), input_structure.parameter_count(), ProgramError);
        check_count!("output", entry_region.output_ids(), output_structure.parameter_count(), ProgramError);

        // A complete program arena is rooted at its final sealed region. Every attached descendant must precede its
        // parent, and the entry has no enclosing parent. Keeping the entry final lets imports and transforms append or
        // remove complete rooted region closures without maintaining a separate arbitrary-root ordering.
        if entry.index() + 1 != regions.len() {
            return Err(ProgramError::MalformedProgram(format!(
                "entry region {entry} must be the final region in the arena",
            )));
        }

        // Walk the attached-region graph from the entry. Shared descendants may be encountered through several
        // instructions, so `reachable` both records the final closure and prevents traversing the same region twice.
        let mut reachable = vec![false; regions.len()];
        let mut pending = vec![entry];
        while let Some(current) = pending.pop() {
            if std::mem::replace(&mut reachable[current.index()], true) {
                continue;
            }
            for instruction in regions[current.index()].instructions() {
                pending.extend(instruction.regions().iter().copied());
            }
        }

        // Reject sealed but orphaned regions. Requiring the arena to equal the entry's complete reachable closure
        // prevents dead nested programs from being retained or unexpectedly copied by whole-program transforms and
        // imports.
        if let Some(unreachable) = reachable.iter().position(|is_reachable| !is_reachable) {
            return Err(ProgramError::MalformedProgram(format!(
                "region {} is not reachable from the program entry region",
                RegionId::new(unreachable),
            )));
        }

        Ok(Self { input_structure, output_structure, regions, entry, marker: PhantomData })
    }

    /// Returns the [`Atom`]s contained in this [`Program`]'s entry [`Region`],
    /// in the order in which they will be evaluated.
    #[inline]
    pub fn atoms(&self) -> &[Atom<V>] {
        self.entry_region_ref().atoms()
    }

    /// Returns the number of input [`Atom`]s (i.e., arguments) of this [`Program`].
    #[inline]
    pub fn input_count(&self) -> usize {
        self.entry_region_ref().input_ids().len()
    }

    /// Returns the [`AtomId`]s of the [`Atom`]s that correspond to the inputs (i.e., arguments) of this [`Program`].
    #[inline]
    pub fn input_ids(&self) -> &[AtomId] {
        self.entry_region_ref().input_ids()
    }

    /// Returns the [`Type`](crate::Type)s of the inputs of this [`Program`], in order.
    #[inline]
    pub fn input_types(&self) -> Vec<V::Type> {
        self.entry_region_ref().input_types()
    }

    /// Returns the [`Atom`]s that correspond to the inputs of this [`Program`].
    #[inline]
    pub fn inputs(&self) -> impl Iterator<Item = &Atom<V>> {
        let entry = self.entry_region();
        entry.input_ids.iter().map(|input_id| &entry.atoms[input_id.index()])
    }

    /// Returns the structured `Input` of this [`Program`] parameterized by the corresponding [`Atom`]s.
    #[inline]
    pub fn input(&self) -> Result<Input::To<Atom<V>>, ParameterError>
    where
        Input::Family: ParameterizedFamily<Atom<V>>,
    {
        Input::To::<Atom<V>>::from_parameters(self.input_structure.clone(), self.inputs().cloned())
    }

    /// Returns the number of output [`Atom`]s (i.e., return values) of this [`Program`].
    #[inline]
    pub fn output_count(&self) -> usize {
        self.entry_region_ref().output_ids().len()
    }

    /// Returns the [`AtomId`]s of the [`Atom`]s that correspond to the outputs (i.e., return values)
    /// of this [`Program`].
    #[inline]
    pub fn output_ids(&self) -> &[AtomId] {
        self.entry_region_ref().output_ids()
    }

    /// Returns the [`Type`](crate::Type)s of the outputs of this [`Program`], in order.
    #[inline]
    pub fn output_types(&self) -> Vec<V::Type> {
        self.entry_region_ref().output_types()
    }

    /// Returns the [`Atom`]s that correspond to the outputs of this [`Program`].
    #[inline]
    pub fn outputs(&self) -> impl Iterator<Item = &Atom<V>> {
        let entry = self.entry_region();
        entry.output_ids.iter().map(|output_id| &entry.atoms[output_id.index()])
    }

    /// Returns the structured `Output` of this [`Program`] parameterized by the corresponding [`Atom`]s.
    #[inline]
    pub fn output(&self) -> Result<Output::To<Atom<V>>, ParameterError>
    where
        Output::Family: ParameterizedFamily<Atom<V>>,
    {
        Output::To::<Atom<V>>::from_parameters(self.output_structure.clone(), self.outputs().cloned())
    }

    /// Returns the ordered sequence of [`Instruction`]s that make up the computational graph of this [`Program`]'s
    /// entry [`Region`].
    #[inline]
    pub fn instructions(&self) -> &[Instruction<O>] {
        self.entry_region_ref().instructions()
    }

    /// Returns the [`Parameter`] structure that can be used to map flat lists of inputs to structured `Input` values.
    #[inline]
    pub fn input_structure(&self) -> &Input::ParameterStructure {
        &self.input_structure
    }

    /// Returns the [`Parameter`] structure that can be used to map flat lists of outputs to structured `Output` values.
    #[inline]
    pub fn output_structure(&self) -> &Output::ParameterStructure {
        &self.output_structure
    }

    /// Returns this [`Program`]'s validated [`RegionArena`].
    #[inline]
    pub fn regions(&self) -> &RegionArena<V, O> {
        &self.regions
    }

    /// Returns the [`Region`] that corresponds to the provided [`RegionId`].
    #[inline]
    pub fn region(&self, id: RegionId) -> Result<&Region<V, O>, ProgramError> {
        self.regions
            .get(id)
            .ok_or_else(|| ProgramError::MalformedProgram(format!("region {id} is out of range")))
    }

    /// Returns a borrowed view of the [`Region`] that corresponds to the provided [`RegionId`].
    #[inline]
    pub fn region_ref(&self, id: RegionId) -> Result<RegionRef<'_, V, O>, ProgramError> {
        RegionRef::new(&self.regions, id)
    }

    /// Returns the [`RegionId`] of the [`Region`] implementing this [`Program`]'s public entry point.
    #[inline]
    pub fn entry(&self) -> RegionId {
        self.entry
    }

    /// Returns the entry [`Region`] of this [`Program`].
    #[inline]
    pub fn entry_region(&self) -> &Region<V, O> {
        &self.regions[self.entry.index()]
    }

    /// Returns a borrowed view of this [`Program`]'s entry [`Region`].
    #[inline]
    pub fn entry_region_ref(&self) -> RegionRef<'_, V, O> {
        RegionRef::new(&self.regions, self.entry).unwrap()
    }

    /// Returns the operation-inference [`RegionInterface`] of this [`Program`]'s entry [`Region`].
    #[inline]
    pub fn interface(&self) -> RegionInterface<V::Type> {
        self.entry_region_ref().interface()
    }

    /// Returns the [`InstructionId`] of the instruction producing the provided value, or [`None`] when the value is
    /// a region input or constant. Returns an error when the locator does not resolve against this [`Program`].
    pub fn producer(&self, value: ValueId) -> Result<Option<InstructionId>, ProgramError> {
        let region = self.region(value.region())?;
        if region.atoms.get(value.atom().index()).is_none() {
            return Err(ProgramError::UnboundAtomId { id: value.atom() });
        }
        Ok(region.instructions.iter().enumerate().find_map(|(index, instruction)| {
            instruction.outputs.contains(&value.atom()).then_some(InstructionId::new(value.region(), index))
        }))
    }

    /// Returns the [`Instruction`] at the provided [`InstructionId`].
    pub fn instruction(&self, id: InstructionId) -> Result<&Instruction<O>, ProgramError> {
        let region = self.region(id.region())?;
        region.instructions.get(id.index()).ok_or_else(|| {
            ProgramError::MalformedProgram(format!(
                "instruction index {} is out of range for region {}",
                id.index(),
                id.region(),
            ))
        })
    }

    /// Returns a vector that has the same length as the number of [`Atom`]s in this [`Program`] and for every atom, it
    /// contains the index of the [`Instruction`] that produces it. Note that input and constant atoms are not produced
    /// by an instruction and so the vector contains [`None`] for those atoms.
    #[inline]
    pub fn instruction_by_output(&self) -> Vec<Option<usize>> {
        self.entry_region().instruction_by_output()
    }

    /// Computes transitive liveness for the [`Atom`]s and [`Instruction`]s of this [`Program`] with respect to the
    /// [`Program`]'s outputs (i.e., it determines whether each atom or instruction contributes to at least one of the
    /// [`Program`]s outputs).
    ///
    /// Note that liveness here is computed in a conservative fashion where, when any output of an instruction is live,
    /// every input to that instruction is considered live as well. Refer to [`Self::live_sets_with`] if you want to
    /// compute liveness in a more fine-grained fashion.
    #[inline]
    pub fn live_sets(&self) -> ProgramLiveSets {
        self.live_sets_for_atoms(self.output_ids()).unwrap()
    }

    /// Computes transitive liveness for the [`Atom`]s and [`Instruction`]s of this [`Program`] with respect to the
    /// [`Program`]'s outputs (i.e., it determines whether each atom or instruction contributes to at least one of the
    /// [`Program`]s outputs), using a caller-provided operation-specific output-to-input liveness propagation function
    /// (i.e., `propagate_liveness`).
    ///
    /// This is to [`Self::live_sets`] what [`Self::live_sets_for_atoms_with`] is to [`Self::live_sets_for_atoms`].
    /// It computes liveness over the [`Program`]'s outputs like [`Self::live_sets`], but lets callers refine how each
    /// instruction propagates liveness from its outputs to its inputs like [`Self::live_sets_for_atoms_with`]. Refer
    /// to [`Self::live_sets_for_atoms_with`] for information on the `propagate_liveness` contract. Unlike
    /// [`Self::live_sets`], this function is fallible because `propagate_liveness` may fail.
    #[inline]
    pub fn live_sets_with<
        F: FnMut(&Program<V, O, Input, Output>, &Instruction<O>, &[bool], &mut Vec<bool>) -> Result<(), ProgramError>,
    >(
        &self,
        propagate_liveness: F,
    ) -> Result<ProgramLiveSets, ProgramError> {
        self.live_sets_for_atoms_with(self.output_ids(), propagate_liveness)
    }

    /// Computes transitive liveness for the [`Atom`]s and [`Instruction`]s of this [`Program`] with respect to the
    /// provided atom IDs (i.e., it determines whether each atom or instruction contributes to computing at least one
    /// of the atoms that correspond to the provided IDs).
    ///
    /// Note that liveness here is computed in a conservative fashion where, when any output of an instruction is live,
    /// every input to that instruction is considered live as well. Refer to [`Self::live_sets_for_atoms_with`] if you
    /// want to compute liveness in a more fine-grained fashion.
    #[inline]
    pub fn live_sets_for_atoms(&self, atom_ids: &[AtomId]) -> Result<ProgramLiveSets, ProgramError> {
        self.live_sets_for_atoms_with(atom_ids, |_, instruction, output_liveness, input_liveness| {
            let has_live_output = output_liveness.iter().copied().any(|is_live| is_live);
            input_liveness.resize(instruction.inputs().len(), has_live_output);
            Ok(())
        })
    }

    /// Computes transitive liveness for the [`Atom`]s and [`Instruction`]s of this [`Program`] with respect to the
    /// provided atom IDs (i.e., it determines whether each atom or instruction contributes to computing at least one
    /// of the atoms that correspond to the provided IDs), using a caller-provided operation-specific output-to-input
    /// liveness propagation function (i.e., `propagate_liveness`).
    ///
    /// The `propagate_liveness` function receives the source program, each live instruction, a boolean liveness flag
    /// per instruction output, and a cleared input liveness buffer. It must push to that buffer exactly one boolean
    /// value per instruction input. Conservative callers can mark all inputs live whenever any output is live, while
    /// primitive-aware callers can avoid marking inputs that are not needed for the selected outputs.
    pub fn live_sets_for_atoms_with<
        F: FnMut(&Program<V, O, Input, Output>, &Instruction<O>, &[bool], &mut Vec<bool>) -> Result<(), ProgramError>,
    >(
        &self,
        output_ids: &[AtomId],
        mut propagate_liveness: F,
    ) -> Result<ProgramLiveSets, ProgramError> {
        let entry = self.entry_region();
        let mut live_sets = ProgramLiveSets::new(vec![false; entry.atoms.len()], vec![false; entry.instructions.len()]);
        for output in output_ids.iter().copied() {
            let Some(slot) = live_sets.atoms.get_mut(output.index()) else {
                return Err(ProgramError::UnboundAtomId { id: output });
            };
            *slot = true;
        }
        let max_input_count =
            self.instructions().iter().map(|instruction| instruction.inputs().len()).max().unwrap_or(0);
        let max_output_count =
            self.instructions().iter().map(|instruction| instruction.outputs().len()).max().unwrap_or(0);
        let mut input_liveness = Vec::with_capacity(max_input_count);
        let mut output_liveness = Vec::with_capacity(max_output_count);
        for (instruction_index, instruction) in entry.instructions.iter().enumerate().rev() {
            output_liveness.clear();
            let mut has_live_output = false;
            for output in instruction.outputs.iter().copied() {
                let is_live =
                    live_sets.atoms.get(output.index()).copied().ok_or(ProgramError::UnboundAtomId { id: output })?;
                has_live_output |= is_live;
                output_liveness.push(is_live);
            }
            if !has_live_output {
                continue;
            }

            live_sets.instructions[instruction_index] = true;
            input_liveness.clear();
            propagate_liveness(self, instruction, output_liveness.as_slice(), &mut input_liveness)?;
            check_count!("input", input_liveness, instruction.inputs.len(), ProgramError);
            for (input, is_live) in instruction.inputs.iter().copied().zip(input_liveness.iter().copied()) {
                if is_live {
                    let Some(slot) = live_sets.atoms.get_mut(input.index()) else {
                        return Err(ProgramError::UnboundAtomId { id: input });
                    };
                    *slot = true;
                }
            }
        }

        Ok(live_sets)
    }

    /// Returns the [`Effect`](crate::Effect) classes reachable from this [`Program`]'s entry region, or
    /// [`Effects::PURE`] for programs with no instructions. Because attached regions live in the same arena,
    /// nested-computation effects are visible through higher-order boundaries. Attached
    /// [`RegionRole::Rule`](crate::RegionRole::Rule) regions are excluded because they are dormant during ordinary
    /// interpretation. The per-[`Instruction`] counterpart to this function is [`Self::instruction_effects`].
    #[inline]
    pub fn effects(&self) -> Effects {
        self.entry_region_ref().effects()
    }

    /// Returns the [`Effect`](crate::Effect) classes of the [`Instruction`] at the provided [`InstructionId`]. That is
    /// defined as the union of its [`Operation`]'s intrinsic [`Operation::effects`] and the recursively derived effects
    /// of attached [`RegionRole::Computation`](crate::RegionRole::Computation) regions. Consulting only the operation's
    /// intrinsic effects would be unsound for control-flow and call operations, while unconditionally including
    /// transform-only rule regions would incorrectly make the ordinary computation effectful.
    #[inline]
    pub fn instruction_effects(&self, id: InstructionId) -> Result<Effects, ProgramError> {
        self.region_ref(id.region())?.instruction_effects(id.index())
    }

    /// Returns this [`Program`]'s entry point's closed [`TypeIdentitySignature`].
    #[inline]
    pub fn type_identity_signature(&self) -> &TypeIdentitySignature<<V::Type as Type>::Identity> {
        self.regions.type_identity_signature(self.entry).unwrap()
    }

    /// Returns this [`Program`] with `renaming` applied to every [`Region`] [`Atom`], constant, and [`Operation`]
    /// payload inside this [`Program`].
    #[inline]
    pub fn rename_type_identities(
        &self,
        renaming: &TypeIdentityRenaming<<V::Type as Type>::Identity>,
    ) -> Result<Self, ProgramError> {
        Self::new(
            self.input_structure.clone(),
            self.output_structure.clone(),
            self.regions
                .iter()
                .map(|region| region.rename_type_identities(renaming))
                .collect::<Result<Vec<_>, _>>()?,
            self.entry,
        )
    }

    /// Returns this [`Program`] with its formal [`TypeIdentity`](crate::TypeIdentity)s instantiated for `input_types`.
    /// This method validates `input_types` against the declared program input types and derives the simultaneous
    /// [`TypeIdentityRenaming`] implied by that complete boundary. It applies a nonempty renaming throughout the
    /// program, rebuilding and structurally reclosing the complete [`RegionArena`]. Static refinements are validated
    /// but do not specialize the returned program's types. When no renaming is required, the result borrows this
    /// program.
    pub fn with_instantiated_type_identities<'p, 't>(
        &'p self,
        input_types: &'t [V::Type],
    ) -> Result<Cow<'p, Self>, ProgramError> {
        let renaming = V::Type::derive_identity_renaming(self.input_types().as_slice(), input_types)?;
        if renaming.is_identity() {
            return Ok(Cow::Borrowed(self));
        }
        Ok(Cow::Owned(self.rename_type_identities(&renaming)?))
    }

    /// Rebuilds this [`Program`] with each [`Operation`] mapped using the provided `map_fn`. The atom table,
    /// input/output atom identifiers, and parameter structures are preserved exactly. This is useful for transforms
    /// that keep the same value graph but need to change operation payloads. For example, a reusable residualized
    /// linear program may contain operations whose scale/dot factors are residual references rather than executable
    /// values. Before interpreting that program, the mapping closure can receive each linear operation, call the
    /// operation's factor-mapping hook, and replace each residual reference with the concrete residual value captured
    /// by the corresponding linearization run.
    pub fn map_operations<P: Operation<Type = V::Type>, F: FnMut(&O) -> Result<P, ProgramError>>(
        &self,
        mut map_fn: F,
    ) -> Result<Program<V, P, Input, Output>, ProgramError> {
        Program::new(
            self.input_structure.clone(),
            self.output_structure.clone(),
            self.regions
                .iter()
                .map(|region| {
                    Ok(Region {
                        atoms: region.atoms.clone(),
                        input_ids: region.input_ids.clone(),
                        output_ids: region.output_ids.clone(),
                        instructions: region
                            .instructions
                            .iter()
                            .map(|instruction| {
                                Ok(Instruction::new(
                                    map_fn(instruction.operation())?,
                                    instruction.inputs().to_vec(),
                                    instruction.outputs().to_vec(),
                                    instruction.regions().to_vec(),
                                ))
                            })
                            .collect::<Result<Vec<_>, ProgramError>>()?,
                    })
                })
                .collect::<Result<Vec<_>, ProgramError>>()?,
            self.entry,
        )
    }

    /// Returns a cloned view of this [`Program`] whose public input and output types are flat vectors. The atom table,
    /// input atom identifiers, output atom identifiers, and instruction sequence are preserved exactly. Only the
    /// `Input` and `Output` type parameters change to `Vec<V>`, with placeholder structures sized to the flat input and
    /// output arities. This is the canonical shape for standalone nested computations supplied positionally through the
    /// region driver passed to [`Context::bind`](crate::Context::bind), including both owned [`Region`]s and shared
    /// callees, without needing to preserve the caller's original [`Parameterized`] type.
    pub fn to_flat_program(&self) -> Program<V, O, Vec<V>, Vec<V>>
    where
        O: Clone,
    {
        Program {
            input_structure: vec![Placeholder; self.input_count()],
            output_structure: vec![Placeholder; self.output_count()],
            regions: self.regions.clone(),
            entry: self.entry,
            marker: PhantomData,
        }
    }

    /// Converts this [`Program`] into one whose public input and output types are flat vectors. This is the consuming
    /// counterpart of [`Program::to_flat_program`]. It preserves the atom table, input atom identifiers, output atom
    /// identifiers, and instruction sequence without cloning them, and only replaces the structured input and output
    /// metadata with [`Placeholder`] vector structures sized to the flat arities.
    pub fn into_flat_program(self) -> Program<V, O, Vec<V>, Vec<V>> {
        let input_structure = vec![Placeholder; self.input_count()];
        let output_structure = vec![Placeholder; self.output_count()];
        Program { input_structure, output_structure, regions: self.regions, entry: self.entry, marker: PhantomData }
    }

    /// Replaces this [`Program`]'s structured boundary metadata after validating that the new structures
    /// are compatible with this program's number of input and output values.
    #[inline]
    pub fn restructured<NewInput: Parameterized<V>, NewOutput: Parameterized<V>>(
        self,
        input_structure: NewInput::ParameterStructure,
        output_structure: NewOutput::ParameterStructure,
    ) -> Result<Program<V, O, NewInput, NewOutput>, ProgramError> {
        check_count!("input", self.input_ids(), input_structure.parameter_count(), ProgramError);
        check_count!("output", self.output_ids(), output_structure.parameter_count(), ProgramError);
        Ok(Program { input_structure, output_structure, regions: self.regions, entry: self.entry, marker: PhantomData })
    }

    /// Consumes this [`Program`] over projected values and operations and embeds it in its unprojected value and
    /// operation families. The conversion changes only the stored value, type, and operation representations. It
    /// preserves every [`AtomId`], [`RegionId`], instruction edge, attached-region edge, shared region, and public
    /// parameter structure. Constants are lifted through [`ValueProjection::from_projected`], variable types through
    /// `From<V::Type>`, and operations through `From<O>`. The mapped region arena is then rebuilt through
    /// [`Program::new`], which revalidates structural closure and derives effects and type-identity metadata
    /// from the converted graph exactly once.
    ///
    /// This is the canonical bridge for unprojecting an already-built member program into its containing program
    /// family. It does not interpret or replay any instruction and therefore cannot lose Single Static Assignment
    /// (SSA) identity or reconstruct dependencies outside the source graph.
    pub fn into_unprojected<UnprojectedValue, UnprojectedOperation>(
        self,
    ) -> Result<
        Program<UnprojectedValue, UnprojectedOperation, Input::To<UnprojectedValue>, Output::To<UnprojectedValue>>,
        ProgramError,
    >
    where
        UnprojectedValue: Value + ValueProjection<V::Type, Projected = V>,
        UnprojectedValue::Type: From<V::Type>,
        UnprojectedOperation: Operation<Type = UnprojectedValue::Type> + From<O>,
        Input::Family: ParameterizedFamily<UnprojectedValue>,
        Output::Family: ParameterizedFamily<UnprojectedValue>,
    {
        let Program { input_structure, output_structure, regions, entry, .. } = self;
        let regions = regions
            .into_regions()
            .into_iter()
            .map(|region| Region {
                atoms: region
                    .atoms
                    .into_iter()
                    .map(|atom| match atom {
                        Atom::Constant(value) => {
                            Atom::Constant(<UnprojectedValue as ValueProjection<V::Type>>::from_projected(value))
                        }
                        Atom::Variable(r#type) => Atom::Variable(r#type.into()),
                    })
                    .collect(),
                input_ids: region.input_ids,
                output_ids: region.output_ids,
                instructions: region
                    .instructions
                    .into_iter()
                    .map(|instruction| {
                        let (operation, inputs, outputs, regions) = instruction.into_parts();
                        Instruction::new(operation.into(), inputs, outputs, regions)
                    })
                    .collect(),
            })
            .collect();
        Program::new(input_structure, output_structure, regions, entry)
    }

    /// Returns a simplified version of this [`Program`] with dead constants and [`Instruction`]s that do not contribute
    /// to the [`Program`]'s output removed. [`Instruction`]s whose operations are not [`Effects::PURE`] are kept alive
    /// (together with the instructions producing their inputs) even when no program output consumes their results, in
    /// their original relative order, so that simplification never eliminates or reorders observable
    /// [`Effect`](crate::Effect)s.
    pub fn simplified(&self) -> Result<Self, ProgramError>
    where
        O: Clone,
    {
        // Simplify every region independently. A region's inputs and outputs are its boundary contract and always
        // survive, so per-region dead-code elimination only removes internal dead work. Retained instructions keep
        // their attached-region references, and regions that lose their last reference are dropped afterward by the
        // compaction step, which also rewrites the surviving references.
        let regions = self
            .regions
            .iter()
            .enumerate()
            .map(|(region_index, region)| {
                let instruction_by_output = region.instruction_by_output();
                let mut new_atoms = Vec::with_capacity(region.atoms.len());
                let mut new_input_ids = Vec::with_capacity(region.input_ids.len());
                let mut new_instructions = Vec::with_capacity(region.instructions.len());
                let mut atom_id_mapping = HashMap::with_capacity(region.atoms.len());
                for input_id in region.input_ids.iter().copied() {
                    let input =
                        region.atoms.get(input_id.index()).ok_or(ProgramError::UnboundAtomId { id: input_id })?;
                    let Atom::Variable(input_type) = input else {
                        return Err(ProgramError::MalformedProgram(
                            "program input atom was not a variable".to_string(),
                        ));
                    };
                    let new_input = AtomId::new(new_atoms.len());
                    new_atoms.push(Atom::Variable(input_type.clone()));
                    new_input_ids.push(new_input);
                    atom_id_mapping.insert(input_id, new_input);
                }

                // Make sure that effectful instructions and their transitive dependencies are processed in original
                // instruction order before the outputs, so that instructions with observable effects survive even
                // when dead and ordered effects keep their relative order.
                for (instruction_index, instruction) in region.instructions.iter().enumerate() {
                    if self
                        .instruction_effects(InstructionId::new(RegionId::new(region_index), instruction_index))?
                        .is_pure()
                    {
                        continue;
                    }
                    if instruction.outputs().is_empty() {
                        let inputs = instruction
                            .inputs()
                            .iter()
                            .copied()
                            .map(|input| {
                                clone_atom_subgraph_into_region(
                                    &mut atom_id_mapping,
                                    input,
                                    region,
                                    instruction_by_output.as_slice(),
                                    &mut new_atoms,
                                    &mut new_instructions,
                                )
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        new_instructions.push(Instruction::new(
                            instruction.operation().clone(),
                            inputs,
                            Vec::new(),
                            instruction.regions().to_vec(),
                        ));
                        continue;
                    }
                    for output_id in instruction.outputs().iter().copied() {
                        clone_atom_subgraph_into_region(
                            &mut atom_id_mapping,
                            output_id,
                            region,
                            instruction_by_output.as_slice(),
                            &mut new_atoms,
                            &mut new_instructions,
                        )?;
                    }
                }

                let output_ids = region
                    .output_ids
                    .iter()
                    .copied()
                    .map(|output| {
                        clone_atom_subgraph_into_region(
                            &mut atom_id_mapping,
                            output,
                            region,
                            instruction_by_output.as_slice(),
                            &mut new_atoms,
                            &mut new_instructions,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                Ok(Region { atoms: new_atoms, input_ids: new_input_ids, output_ids, instructions: new_instructions })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let (regions, entry) = compact_regions(regions, self.entry);
        Self::new(self.input_structure.clone(), self.output_structure.clone(), regions, entry)
    }

    /// Consumes this [`Program`] and returns a simplified version with dead constants and [`Instruction`]s that do not
    /// contribute to the [`Program`]'s output removed. Unlike [`Self::simplified`], this method moves live [`Atom`]s,
    /// [`Instruction`]s, and parameter structures into the returned [`Program`] instead of cloning them. This avoids
    /// copying constants and operations that are discarded during simplification. The behavior of [`Self::simplified`]
    /// around [`Effects`] applies here too. [`Instruction`]s whose operations are not [`Effects::PURE`] survive in
    /// their original relative order even when no program output consumes their outputs.
    pub fn into_simplified(self) -> Result<Self, ProgramError> {
        let expected_input_count = self.input_structure.parameter_count();
        check_count!("input", self.input_ids(), expected_input_count, ProgramError);

        let expected_output_count = self.output_structure.parameter_count();
        check_count!("output", self.output_ids(), expected_output_count, ProgramError);

        // Simplify every region independently, exactly like `Self::simplified` but moving live atoms and
        // instructions into the rebuilt regions instead of cloning them.
        let effectful_instructions = self
            .regions
            .iter()
            .enumerate()
            .map(|(region_index, region)| {
                region
                    .instructions
                    .iter()
                    .enumerate()
                    .filter(|(instruction_index, _)| {
                        !self
                            .instruction_effects(InstructionId::new(RegionId::new(region_index), *instruction_index))
                            .unwrap()
                            .is_pure()
                    })
                    .map(|(instruction_index, instruction)| (instruction_index, instruction.outputs().to_vec()))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let Self { regions, input_structure, output_structure, entry, .. } = self;
        let regions = regions
            .into_regions()
            .into_iter()
            .zip(effectful_instructions)
            .map(|(region, effectful_instructions)| {
                let instruction_by_output = region.instruction_by_output();
                let Region { atoms, input_ids, output_ids, instructions } = region;
                let mut atoms = atoms.into_iter().map(Some).collect::<Vec<_>>();
                let mut instructions = instructions.into_iter().map(Some).collect::<Vec<_>>();
                let mut new_atoms = Vec::with_capacity(atoms.len());
                let mut new_input_ids = Vec::with_capacity(input_ids.len());
                let mut new_instructions = Vec::with_capacity(instructions.len());
                let mut atom_id_mapping = HashMap::with_capacity(atoms.len());
                for input_id in input_ids {
                    let input = atoms
                        .get_mut(input_id.index())
                        .ok_or(ProgramError::UnboundAtomId { id: input_id })?
                        .take()
                        .ok_or(ProgramError::MalformedProgram("program input atom was already moved".to_string()))?;
                    let Atom::Variable(input_type) = input else {
                        return Err(ProgramError::MalformedProgram(
                            "program input atom was not a variable".to_string(),
                        ));
                    };
                    let new_input = AtomId::new(new_atoms.len());
                    new_atoms.push(Atom::Variable(input_type));
                    new_input_ids.push(new_input);
                    atom_id_mapping.insert(input_id, new_input);
                }

                // Make sure that effectful instructions and their transitive dependencies are processed in original
                // instruction order before the outputs, so that instructions with observable effects survive even
                // when dead and ordered effects keep their relative order.
                for (instruction_index, outputs) in effectful_instructions {
                    if outputs.is_empty() {
                        let instruction = instructions[instruction_index]
                            .take()
                            .ok_or(ProgramError::MalformedProgram("instruction was already moved".to_string()))?;
                        let inputs = instruction
                            .inputs()
                            .iter()
                            .copied()
                            .map(|input| {
                                move_atom_to_program(
                                    &mut atom_id_mapping,
                                    input,
                                    atoms.as_mut_slice(),
                                    instructions.as_mut_slice(),
                                    instruction_by_output.as_slice(),
                                    &mut new_atoms,
                                    &mut new_instructions,
                                )
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        new_instructions.push(Instruction::new(
                            instruction.operation,
                            inputs,
                            Vec::new(),
                            instruction.regions,
                        ));
                        continue;
                    }
                    for root in outputs {
                        move_atom_to_program(
                            &mut atom_id_mapping,
                            root,
                            atoms.as_mut_slice(),
                            instructions.as_mut_slice(),
                            instruction_by_output.as_slice(),
                            &mut new_atoms,
                            &mut new_instructions,
                        )?;
                    }
                }

                let output_ids = output_ids
                    .into_iter()
                    .map(|output| {
                        move_atom_to_program(
                            &mut atom_id_mapping,
                            output,
                            atoms.as_mut_slice(),
                            instructions.as_mut_slice(),
                            instruction_by_output.as_slice(),
                            &mut new_atoms,
                            &mut new_instructions,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                Ok(Region { atoms: new_atoms, input_ids: new_input_ids, output_ids, instructions: new_instructions })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let (regions, entry) = compact_regions(regions, entry);
        Self::new(input_structure, output_structure, regions, entry)
    }

    /// Rebuilds this [`Program`] as a flat subprogram over a chosen input/output boundary. The rebuilt program
    /// keeps only the [`Instruction`]s reachable from `outputs`, from the provided `keep_alive` atoms, or from
    /// observable effects, and lifts embedded constants directly into the result. Entries of `inputs` that are not
    /// reachable from any requested output, keep-alive atom, or effectful instruction are dropped. The returned index
    /// vector lists, in order, the positions of `inputs` that remain live and become the public inputs of the rebuilt
    /// program, so that callers can map rebuilt inputs back to the original boundary.
    ///
    /// Each [`Atom::Variable`] reachable from an output or keep-alive atom must either appear in `inputs` or be
    /// produced by an [`Instruction`] of this program. Reaching any other source variable (e.g., an original program
    /// input that was not selected) is reported as a [`ProgramError::MalformedProgram`]. Every entry of `inputs` must
    /// be an [`Atom::Variable`] and must appear at most once. [`Atom::Constant`]s are rebuilt automatically and need
    /// not be listed.
    ///
    /// This is the graph-projection primitive used by transforms that carve a subgraph out of an already-traced program
    /// over a known input boundary, such as separating a primal residual computation from a transposed cotangent
    /// application during shard-map transpose factorization.
    ///
    /// Refer to [`Self::into_filtered`] for a consuming variant that moves live atoms and instructions into the
    /// resulting program instead of cloning them.
    ///
    /// # Parameters
    ///
    ///   - `inputs`: [`AtomId`]s of the atoms eligible to become the rebuilt program's public inputs, in input order.
    ///   - `outputs`: [`AtomId`]s of the atoms to expose as the rebuilt program's outputs, in output order.
    ///   - `keep_alive`: [`AtomId`]s of atoms that must survive even if they are unreachable from `outputs`.
    pub fn filtered(
        &self,
        inputs: &[AtomId],
        outputs: &[AtomId],
        keep_alive: &[AtomId],
    ) -> Result<(Program<V, O, Vec<V>, Vec<V>>, Vec<usize>), ProgramError>
    where
        O: Clone,
    {
        let ProgramLivenessAnalysis { instruction_by_output, input_liveness, effectful_instruction_indices } =
            self.analyze_liveness(inputs, outputs, keep_alive)?;
        let entry_region = self.entry_region();
        let mut new_atoms = Vec::with_capacity(entry_region.atoms.len());
        let mut new_input_ids = Vec::new();
        let mut new_instructions = Vec::with_capacity(entry_region.instructions.len());
        let mut atom_id_mapping = HashMap::with_capacity(entry_region.atoms.len());
        let mut live_input_indices = Vec::new();

        for (position, id) in inputs.iter().copied().enumerate() {
            if !input_liveness[position] {
                continue;
            }
            let Atom::Variable(input_type) = &entry_region.atoms[id.index()] else {
                return Err(ProgramError::MalformedProgram(format!("filter input atom {id} is not a variable")));
            };
            let new_input = AtomId::new(new_atoms.len());
            new_atoms.push(Atom::Variable(input_type.clone()));
            new_input_ids.push(new_input);
            atom_id_mapping.insert(id, new_input);
            live_input_indices.push(position);
        }

        // Process effectful instructions first, in original order, so dead effects survive and ordered effects retain
        // their relative order. Instructions without outputs need explicit rebuilding because they cannot be reached
        // through the instruction-by-output map.
        for instruction_index in effectful_instruction_indices {
            let instruction = &entry_region.instructions()[instruction_index];
            if instruction.outputs().is_empty() {
                let effect_inputs = instruction
                    .inputs()
                    .iter()
                    .copied()
                    .map(|input| {
                        clone_atom_subgraph_into_region(
                            &mut atom_id_mapping,
                            input,
                            entry_region,
                            instruction_by_output.as_slice(),
                            &mut new_atoms,
                            &mut new_instructions,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                new_instructions.push(Instruction::new(
                    instruction.operation().clone(),
                    effect_inputs,
                    Vec::new(),
                    instruction.regions().to_vec(),
                ));
            } else {
                for root in instruction.outputs().iter().copied() {
                    clone_atom_subgraph_into_region(
                        &mut atom_id_mapping,
                        root,
                        entry_region,
                        instruction_by_output.as_slice(),
                        &mut new_atoms,
                        &mut new_instructions,
                    )?;
                }
            }
        }

        for root in keep_alive.iter().copied() {
            clone_atom_subgraph_into_region(
                &mut atom_id_mapping,
                root,
                entry_region,
                instruction_by_output.as_slice(),
                &mut new_atoms,
                &mut new_instructions,
            )?;
        }

        let output_ids = outputs
            .iter()
            .copied()
            .map(|id| {
                clone_atom_subgraph_into_region(
                    &mut atom_id_mapping,
                    id,
                    entry_region,
                    instruction_by_output.as_slice(),
                    &mut new_atoms,
                    &mut new_instructions,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Nested regions of retained instructions pass through unchanged (filtering is an entry-boundary projection);
        // regions that lost their last reference are dropped and the surviving references are rewritten.
        let mut regions = self.regions.iter().take(self.entry.index()).cloned().collect::<Vec<_>>();
        regions.push(Region { atoms: new_atoms, input_ids: new_input_ids, output_ids, instructions: new_instructions });
        let (regions, entry) = compact_regions(regions, self.entry);
        let program = Program::new(
            vec![Placeholder; live_input_indices.len()],
            vec![Placeholder; outputs.len()],
            regions,
            entry,
        )?;
        Ok((program, live_input_indices))
    }

    /// Consumes this [`Program`] and returns the same flat subprogram as [`Self::filtered`] over the chosen `inputs`
    /// and `outputs` boundary. Unlike [`Self::filtered`], this moves live [`Atom`]s and [`Instruction`]s into the
    /// returned program instead of cloning them, avoiding copies of the constants and operations that survive the
    /// projection. The boundary contract, keep-alive semantics, dead-input pruning, and returned live-input index
    /// vector are identical to [`Self::filtered`].
    pub fn into_filtered(
        self,
        inputs: &[AtomId],
        outputs: &[AtomId],
        keep_alive: &[AtomId],
    ) -> Result<(Program<V, O, Vec<V>, Vec<V>>, Vec<usize>), ProgramError> {
        let ProgramLivenessAnalysis { instruction_by_output, input_liveness, effectful_instruction_indices } =
            self.analyze_liveness(inputs, outputs, keep_alive)?;
        let entry = self.entry;
        let mut nested_regions = self.regions.into_regions();
        let Region { atoms, instructions, .. } = nested_regions.pop().unwrap();
        let mut atoms = atoms.into_iter().map(Some).collect::<Vec<_>>();
        let mut instructions = instructions.into_iter().map(Some).collect::<Vec<_>>();
        let mut new_atoms = Vec::with_capacity(atoms.len());
        let mut new_instructions = Vec::with_capacity(instructions.len());
        let mut new_input_ids = Vec::new();
        let mut atom_id_mapping = HashMap::with_capacity(atoms.len());
        let mut live_input_indices = Vec::new();

        for (position, id) in inputs.iter().copied().enumerate() {
            if !input_liveness[position] {
                continue;
            }
            let input = atoms
                .get_mut(id.index())
                .ok_or(ProgramError::UnboundAtomId { id })?
                .take()
                .ok_or(ProgramError::MalformedProgram(format!("filter input atom {id} was already moved")))?;
            let Atom::Variable(input_type) = input else {
                return Err(ProgramError::MalformedProgram(format!("filter input atom {id} is not a variable")));
            };
            let new_input = AtomId::new(new_atoms.len());
            new_atoms.push(Atom::Variable(input_type));
            new_input_ids.push(new_input);
            atom_id_mapping.insert(id, new_input);
            live_input_indices.push(position);
        }

        // Process effectful instructions first, in original order, so dead effects survive and ordered effects retain
        // their relative order. Instructions without outputs need explicit rebuilding because they cannot be reached
        // through the instruction-by-output map.
        for instruction_index in effectful_instruction_indices {
            if instructions[instruction_index].as_ref().unwrap().outputs().is_empty() {
                let instruction = instructions[instruction_index]
                    .take()
                    .ok_or(ProgramError::MalformedProgram("instruction was already moved".to_string()))?;
                let effect_inputs = instruction
                    .inputs()
                    .iter()
                    .copied()
                    .map(|input| {
                        move_atom_to_program(
                            &mut atom_id_mapping,
                            input,
                            atoms.as_mut_slice(),
                            instructions.as_mut_slice(),
                            instruction_by_output.as_slice(),
                            &mut new_atoms,
                            &mut new_instructions,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                new_instructions.push(Instruction::new(
                    instruction.operation,
                    effect_inputs,
                    Vec::new(),
                    instruction.regions,
                ));
            } else {
                // Moving one output may consume its producing instruction, so retain the small output-ID list
                // before mutating the instruction table.
                let effect_outputs = instructions[instruction_index].as_ref().unwrap().outputs().to_vec();
                for root in effect_outputs {
                    move_atom_to_program(
                        &mut atom_id_mapping,
                        root,
                        atoms.as_mut_slice(),
                        instructions.as_mut_slice(),
                        instruction_by_output.as_slice(),
                        &mut new_atoms,
                        &mut new_instructions,
                    )?;
                }
            }
        }

        for root in keep_alive.iter().copied() {
            move_atom_to_program(
                &mut atom_id_mapping,
                root,
                atoms.as_mut_slice(),
                instructions.as_mut_slice(),
                instruction_by_output.as_slice(),
                &mut new_atoms,
                &mut new_instructions,
            )?;
        }

        let output_ids = outputs
            .iter()
            .copied()
            .map(|id| {
                move_atom_to_program(
                    &mut atom_id_mapping,
                    id,
                    atoms.as_mut_slice(),
                    instructions.as_mut_slice(),
                    instruction_by_output.as_slice(),
                    &mut new_atoms,
                    &mut new_instructions,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        let input_structure = vec![Placeholder; new_input_ids.len()];
        let output_structure = vec![Placeholder; output_ids.len()];

        // Nested regions of retained instructions pass through unchanged (filtering is an entry-boundary projection).
        // Regions that lost their last reference are dropped, and the surviving references are rewritten.
        nested_regions.push(Region {
            atoms: new_atoms,
            input_ids: new_input_ids,
            output_ids,
            instructions: new_instructions,
        });

        let (regions, entry) = compact_regions(nested_regions, entry);
        Ok((
            Program::<V, O, Vec<V>, Vec<V>>::new(input_structure, output_structure, regions, entry)?,
            live_input_indices,
        ))
    }

    /// Analyzes entry-region liveness for a filtered program boundary. `inputs` must be a deduplicated collection of
    /// [`Atom::Variable`]s. Reverse reachability begins at `outputs`, the provided `keep_alive` atoms, and every
    /// effectful instruction, including instructions without outputs. Reaching a variable that is neither listed in
    /// `inputs` nor produced by an [`Instruction`] is reported as a [`ProgramError::MalformedProgram`].
    fn analyze_liveness(
        &self,
        inputs: &[AtomId],
        outputs: &[AtomId],
        keep_alive: &[AtomId],
    ) -> Result<ProgramLivenessAnalysis, ProgramError> {
        let mut input_position = vec![None; self.atoms().len()];
        for (position, id) in inputs.iter().copied().enumerate() {
            let atom = self.atoms().get(id.index()).ok_or(ProgramError::UnboundAtomId { id })?;
            if !atom.is_variable() {
                return Err(ProgramError::MalformedProgram(format!("filter input atom {id} is not a variable")));
            }
            let slot = &mut input_position[id.index()];
            if slot.is_some() {
                return Err(ProgramError::MalformedProgram(format!(
                    "filter input atom {id} was provided more than once",
                )));
            }
            *slot = Some(position);
        }

        let instruction_by_output = self.instruction_by_output();
        let effectful_instruction_indices = self
            .instructions()
            .iter()
            .enumerate()
            .filter(|(instruction_index, _)| {
                !self.instruction_effects(InstructionId::new(self.entry, *instruction_index)).unwrap().is_pure()
            })
            .map(|(instruction_index, _)| instruction_index)
            .collect::<Vec<_>>();
        let mut needed = vec![false; self.atoms().len()];
        let mut input_liveness = vec![false; inputs.len()];
        let mut stack = Vec::new();
        let effect_roots = effectful_instruction_indices.iter().flat_map(|instruction_index| {
            let instruction = &self.instructions()[*instruction_index];
            if instruction.outputs().is_empty() { instruction.inputs() } else { instruction.outputs() }
        });
        for output in outputs.iter().chain(keep_alive).chain(effect_roots).copied() {
            if output.index() >= self.atoms().len() {
                return Err(ProgramError::UnboundAtomId { id: output });
            }
            if !needed[output.index()] {
                needed[output.index()] = true;
                stack.push(output);
            }
        }

        while let Some(atom_id) = stack.pop() {
            if let Some(position) = input_position[atom_id.index()] {
                input_liveness[position] = true;
                continue;
            }
            match &self.atoms()[atom_id.index()] {
                Atom::Constant(_) => {}
                Atom::Variable(_) => {
                    let instruction_index = instruction_by_output.get(atom_id.index()).copied().flatten().ok_or(
                        ProgramError::MalformedProgram(format!(
                            "filter atom {atom_id} is not a selected input and has no producer",
                        )),
                    )?;
                    for input in self.instructions()[instruction_index].inputs.iter().copied() {
                        if !needed[input.index()] {
                            needed[input.index()] = true;
                            stack.push(input);
                        }
                    }
                }
            }
        }

        Ok(ProgramLivenessAnalysis { instruction_by_output, input_liveness, effectful_instruction_indices })
    }
}

impl<V: Value, O: Operation<Type = V::Type>, Input: Parameterized<V>, Output: Parameterized<V>>
    Program<V, O, Input, Output>
{
    /// Renders this [`Program`] with the provided indentation level that is useful for situations where [`Program`]s
    /// are nested within other programs like with control flow [`Operation`]s. [`Instruction`]s with attached
    /// [`Region`]s render a bracketed region section after their inputs, pairing each region with its declared
    /// name from [`Operation::region_slots`] (falling back to the region index for undeclared regions). A region
    /// referenced exactly once renders nested beneath its referencing instruction, while a region referenced multiple
    /// times renders its body exactly once (at its first reference, labeled with its [`RegionId`]), and every later
    /// reference renders as that identifier alone. [`RegionId`]s are arena indices and therefore deterministic
    /// [`Program`]-local names.
    pub fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        /// Renders one [`Region`] as a `lambda ... in (...)` block, recursively rendering the regions attached to its
        /// instructions according to `reference_counts` and `rendered`.
        fn render_region<V: Value, O: Operation<Type = V::Type>>(
            regions: &RegionArena<V, O>,
            id: RegionId,
            formatter: &mut std::fmt::Formatter<'_>,
            indentation: usize,
            reference_counts: &[usize],
            rendered: &mut [bool],
        ) -> std::fmt::Result {
            let region = &regions[id.index()];
            write!(formatter, "{:indentation$}", "")?;
            write!(formatter, "lambda ")?;
            region.input_ids.iter().enumerate().try_for_each(|(index, input_id)| {
                if index > 0 {
                    write!(formatter, ", {input_id}:{}", region.atoms[input_id.index()].r#type())
                } else {
                    write!(formatter, "{input_id}:{}", region.atoms[input_id.index()].r#type())
                }
            })?;
            writeln!(formatter, " .")?;
            let mut instructions_by_first_output = vec![None; region.atoms.len()];
            for (index, instruction) in region.instructions.iter().enumerate() {
                if let Some(output_id) = instruction.outputs.first() {
                    instructions_by_first_output[output_id.index()] = Some(index);
                }
            }
            let mut binding_count = 0usize;
            let mut is_input = vec![false; region.atoms.len()];
            for input_id in region.input_ids.iter().copied() {
                is_input[input_id.index()] = true;
            }
            for (atom_id, atom) in region.atoms.iter().enumerate() {
                match atom {
                    Atom::Constant(_) => {
                        write!(formatter, "{:indentation$}", "")?;
                        writeln!(
                            formatter,
                            "{} {}:{} = const",
                            if binding_count == 0 { "let" } else { "   " },
                            AtomId::new(atom_id),
                            region.atoms[atom_id].r#type()
                        )?;
                        binding_count += 1;
                    }
                    Atom::Variable(_) if is_input[atom_id] => {}
                    Atom::Variable(_) => {
                        if let Some(instruction_index) = instructions_by_first_output[atom_id] {
                            let instruction = &region.instructions[instruction_index];
                            let line_indentation = if binding_count == 0 { indentation } else { indentation + 4 };
                            write!(formatter, "{:indentation$}", "")?;
                            write!(formatter, "{} ", if binding_count == 0 { "let" } else { "   " })?;
                            instruction.outputs.iter().enumerate().try_for_each(|(index, output)| {
                                if index > 0 {
                                    write!(formatter, ", {output}:{}", region.atoms[output.index()].r#type())
                                } else {
                                    write!(formatter, "{output}:{}", region.atoms[output.index()].r#type())
                                }
                            })?;
                            write!(formatter, " = ")?;
                            instruction.operation.render(formatter, line_indentation)?;
                            instruction.inputs.iter().try_for_each(|input| write!(formatter, " {input}"))?;
                            if !instruction.regions.is_empty() {
                                let slots = instruction.operation.region_slots();
                                write!(formatter, " [")?;
                                for (slot, attached) in instruction.regions.iter().copied().enumerate() {
                                    writeln!(formatter)?;
                                    write!(formatter, "{:width$}", "", width = line_indentation + 4)?;
                                    match slots.get(slot) {
                                        Some(slot) => write!(formatter, "{}=", slot.name)?,
                                        None => write!(formatter, "{slot}=")?,
                                    }
                                    let is_shared = reference_counts[attached.index()] > 1;
                                    if is_shared && rendered[attached.index()] {
                                        write!(formatter, "{attached},")?;
                                        continue;
                                    }
                                    rendered[attached.index()] = true;
                                    if is_shared {
                                        write!(formatter, "{attached}=")?;
                                    }
                                    writeln!(formatter, "{{")?;
                                    render_region(
                                        regions,
                                        attached,
                                        formatter,
                                        line_indentation + 8,
                                        reference_counts,
                                        rendered,
                                    )?;
                                    writeln!(formatter)?;
                                    write!(formatter, "{:width$}", "", width = line_indentation + 4)?;
                                    write!(formatter, "}},")?;
                                }
                                writeln!(formatter)?;
                                write!(formatter, "{:width$}", "", width = line_indentation)?;
                                write!(formatter, "]")?;
                            }
                            writeln!(formatter)?;
                            binding_count += 1;
                        };
                    }
                }
            }
            write!(formatter, "{:indentation$}", "")?;
            write!(formatter, "in (")?;
            region.output_ids.iter().enumerate().try_for_each(|(index, output)| {
                if index > 0 { write!(formatter, ", {output}") } else { write!(formatter, "{output}") }
            })?;
            write!(formatter, ")")
        }

        let mut reference_counts = vec![0usize; self.regions.len()];
        for region in self.regions.iter() {
            for instruction in &region.instructions {
                for attached in instruction.regions().iter().copied() {
                    reference_counts[attached.index()] += 1;
                }
            }
        }

        let mut rendered = vec![false; self.regions.len()];
        render_region(
            &self.regions,
            self.entry,
            formatter,
            indentation,
            reference_counts.as_slice(),
            rendered.as_mut_slice(),
        )
    }
}

impl<V: Value, O: Clone, Input: Parameterized<V>, Output: Parameterized<V>> Clone for Program<V, O, Input, Output> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            input_structure: self.input_structure.clone(),
            output_structure: self.output_structure.clone(),
            regions: self.regions.clone(),
            entry: self.entry,
            marker: PhantomData,
        }
    }
}

impl<V: Value, O: Operation<Type = V::Type>, Input: Parameterized<V>, Output: Parameterized<V>> Display
    for Program<V, O, Input, Output>
{
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

/// _Flat_ [`Program`] (i.e., with flat `Vec`-valued inputs and outputs) over a [`Domain`]'s constant and operation
/// universe. This is the canonical shape for nested computations constructed standalone, including owned region
/// programs and shared callees composed into the region driver passed to [`Context::bind`](crate::Context::bind).
/// Borrowed replay exposes regions through [`BindingRegionDriver`](crate::BindingRegionDriver) without converting
/// them into this owned shape.
pub type FlatProgram<D> = Program<
    <D as Domain>::Constant,
    <D as Domain>::Operation,
    Vec<<D as Domain>::Constant>,
    Vec<<D as Domain>::Constant>,
>;

/// Liveness masks for a [`Program`]'s entry [`Region`]. The masks are indexed by entry-region [`Atom`] and
/// [`Instruction`] positions. Nested regions are not part of this analysis because their inputs and outputs are their
/// boundary contract (i.e., a referenced region is live exactly when a live instruction references it, which the
/// region-aware rebuild paths such as [`Program::simplified`] handle directly).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ProgramLiveSets {
    /// Contains a boolean value per atom in the [`Program`], indicating whether it contributes
    /// to at least one program output.
    atoms: Vec<bool>,

    /// Contains a boolean value per instruction in the [`Program`], indicating whether it contributes
    /// to at least one program output.
    instructions: Vec<bool>,
}

impl ProgramLiveSets {
    /// Creates new [`ProgramLiveSets`].
    #[inline]
    fn new(atoms: Vec<bool>, instructions: Vec<bool>) -> Self {
        Self { atoms, instructions }
    }

    /// Returns a slice that contains a boolean value per atom in the [`Program`], indicating whether it contributes
    /// to at least one program output.
    #[inline]
    pub fn atoms(&self) -> &[bool] {
        self.atoms.as_slice()
    }

    /// Returns a slice that contains a boolean value per instruction in the [`Program`], indicating whether it
    /// contributes to at least one program output.
    #[inline]
    pub fn instructions(&self) -> &[bool] {
        self.instructions.as_slice()
    }
}

/// Entry-[`Region`] liveness analysis result shared by the borrowing and consuming [`Program`] filter implementations.
struct ProgramLivenessAnalysis {
    /// Source [`Instruction`] index producing each entry-[`Region`] [`Atom`], or [`None`] for atoms without a producer.
    instruction_by_output: Vec<Option<usize>>,

    /// Liveness flag for each input [`Atom`], in caller-provided input order.
    input_liveness: Vec<bool>,

    /// Indices of effectful entry-[`Region`] [`Instruction`]s, retained in original [`Program`] order.
    effectful_instruction_indices: Vec<usize>,
}

/// Copies the [`Atom`] that corresponds to `atom_id` in `region` (and its transitive producers) into `new_atoms` and
/// `new_instructions`, memoizing the old-to-new [`AtomId`] mapping in `atom_id_mapping`. Atoms already present in the
/// mapping (e.g., rebuilt region inputs) are reused, [`Atom::Constant`]s are cloned directly, and [`Atom::Variable`]s
/// are reconstructed from their producing [`Instruction`], whose attached-region references are preserved verbatim
/// (unreferenced regions are dropped and identifiers rewritten by [`compact_regions`] afterward). A reachable variable
/// that is neither mapped nor produced by an instruction is reported as a [`ProgramError::MalformedProgram`].
fn clone_atom_subgraph_into_region<V: Value, O: Operation<Type = V::Type>>(
    atom_id_mapping: &mut HashMap<AtomId, AtomId>,
    atom_id: AtomId,
    region: &Region<V, O>,
    instruction_by_output: &[Option<usize>],
    new_atoms: &mut Vec<Atom<V>>,
    new_instructions: &mut Vec<Instruction<O>>,
) -> Result<AtomId, ProgramError> {
    if let Some(mapped_atom) = atom_id_mapping.get(&atom_id) {
        return Ok(*mapped_atom);
    }
    let atom = region.atoms.get(atom_id.index()).ok_or(ProgramError::UnboundAtomId { id: atom_id })?;
    let atom = match atom {
        Atom::Constant(value) => {
            let new_atom = AtomId::new(new_atoms.len());
            new_atoms.push(Atom::Constant(value.clone()));
            Ok(new_atom)
        }
        Atom::Variable(_) => {
            let instruction_index = instruction_by_output
                .get(atom_id.index())
                .copied()
                .flatten()
                .ok_or(ProgramError::MalformedProgram("variable atom has no owning instruction".to_string()))?;
            let instruction = &region.instructions[instruction_index];
            let inputs = instruction
                .inputs
                .iter()
                .copied()
                .map(|input| {
                    clone_atom_subgraph_into_region(
                        atom_id_mapping,
                        input,
                        region,
                        instruction_by_output,
                        new_atoms,
                        new_instructions,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mut outputs = Vec::with_capacity(instruction.outputs.len());
            for output in instruction.outputs.iter().copied() {
                let output_atom = region.atoms.get(output.index()).ok_or(ProgramError::UnboundAtomId { id: output })?;
                let Atom::Variable(output_type) = output_atom else {
                    return Err(ProgramError::MalformedProgram(
                        "instruction output atom was not a variable".to_string(),
                    ));
                };
                let new_output = AtomId::new(new_atoms.len());
                new_atoms.push(Atom::Variable(output_type.clone()));
                atom_id_mapping.insert(output, new_output);
                outputs.push(new_output);
            }
            new_instructions.push(Instruction::new(
                instruction.operation.clone(),
                inputs,
                outputs,
                instruction.regions.clone(),
            ));
            atom_id_mapping
                .get(&atom_id)
                .copied()
                .ok_or(ProgramError::MalformedProgram("remapped instruction output was missing".to_string()))
        }
    }?;
    atom_id_mapping.insert(atom_id, atom);
    Ok(atom)
}

/// Drops the [`Region`]s in `regions` that are not reachable from `entry` (following [`Instruction`] attached-region
/// references), compacts the surviving regions' identifiers while preserving their relative order, and rewrites every
/// surviving instruction's references accordingly. Returns the compacted arena together with the remapped entry
/// [`RegionId`]. Order preservation keeps the sealed-before-referenced invariant intact, so the compacted arena
/// remains valid for ascending-order recursive metadata derivation by [`RegionArena`].
fn compact_regions<V: Typed, O>(regions: Vec<Region<V, O>>, entry: RegionId) -> (Vec<Region<V, O>>, RegionId) {
    let mut reachable = vec![false; regions.len()];
    let mut pending = vec![entry];
    while let Some(current) = pending.pop() {
        if std::mem::replace(&mut reachable[current.index()], true) {
            continue;
        }
        for instruction in &regions[current.index()].instructions {
            pending.extend(instruction.regions().iter().copied());
        }
    }
    let mut remapping = vec![None; regions.len()];
    let mut kept = 0usize;
    for (index, is_reachable) in reachable.iter().copied().enumerate() {
        if is_reachable {
            remapping[index] = Some(RegionId::new(kept));
            kept += 1;
        }
    }
    let mut compacted = Vec::with_capacity(kept);
    for (index, mut region) in regions.into_iter().enumerate() {
        if !reachable[index] {
            continue;
        }
        for instruction in &mut region.instructions {
            for attached in &mut instruction.regions {
                *attached = remapping[attached.index()].unwrap();
            }
        }
        compacted.push(region);
    }
    (compacted, remapping[entry.index()].unwrap())
}

/// Moves the [`Atom`] that corresponds to `atom_id` (and its transitive producers) out of `atoms`/`instructions` into
/// `new_atoms`/`new_instructions`, memoizing the old-to-new [`AtomId`] mapping in `atom_id_mapping`. This is the
/// move-based counterpart of [`clone_atom_subgraph_into_region`]: it relocates owned [`Atom`]s and [`Instruction`]s
/// (including their attached-region references, verbatim) instead of cloning them, so each is taken from its slot at
/// most once. Atoms already present in the mapping are reused, and a reachable variable that is neither mapped nor
/// produced by an instruction is reported as a [`ProgramError::MalformedProgram`].
fn move_atom_to_program<V: Value, O: Operation<Type = V::Type>>(
    atom_id_mapping: &mut HashMap<AtomId, AtomId>,
    atom_id: AtomId,
    atoms: &mut [Option<Atom<V>>],
    instructions: &mut [Option<Instruction<O>>],
    instruction_by_output: &[Option<usize>],
    new_atoms: &mut Vec<Atom<V>>,
    new_instructions: &mut Vec<Instruction<O>>,
) -> Result<AtomId, ProgramError> {
    if let Some(mapped_atom) = atom_id_mapping.get(&atom_id) {
        return Ok(*mapped_atom);
    }
    let is_constant = match atoms.get(atom_id.index()) {
        Some(Some(Atom::Constant(_))) => true,
        Some(Some(Atom::Variable(_))) => false,
        Some(None) => {
            return Err(ProgramError::MalformedProgram(format!(
                "atom {atom_id} was already moved while rebuilding program",
            )));
        }
        None => return Err(ProgramError::UnboundAtomId { id: atom_id }),
    };
    if is_constant {
        let Some(Atom::Constant(value)) = atoms[atom_id.index()].take() else {
            unreachable!("constant atom kind was checked before moving the atom");
        };
        let new_atom = AtomId::new(new_atoms.len());
        new_atoms.push(Atom::Constant(value));
        atom_id_mapping.insert(atom_id, new_atom);
        return Ok(new_atom);
    }
    let instruction_index = instruction_by_output
        .get(atom_id.index())
        .copied()
        .flatten()
        .ok_or(ProgramError::MalformedProgram("variable atom has no owning instruction".to_string()))?;
    let instruction = instructions[instruction_index]
        .take()
        .ok_or(ProgramError::MalformedProgram("instruction was already moved".to_string()))?;
    let inputs = instruction
        .inputs
        .iter()
        .copied()
        .map(|input| {
            move_atom_to_program(
                atom_id_mapping,
                input,
                atoms,
                instructions,
                instruction_by_output,
                new_atoms,
                new_instructions,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut outputs = Vec::with_capacity(instruction.outputs.len());
    for output in instruction.outputs.iter().copied() {
        let output_atom = atoms
            .get_mut(output.index())
            .ok_or(ProgramError::UnboundAtomId { id: output })?
            .take()
            .ok_or(ProgramError::MalformedProgram("instruction output atom was already moved".to_string()))?;
        let Atom::Variable(output_type) = output_atom else {
            return Err(ProgramError::MalformedProgram("instruction output atom was not a variable".to_string()));
        };
        let new_output = AtomId::new(new_atoms.len());
        new_atoms.push(Atom::Variable(output_type));
        atom_id_mapping.insert(output, new_output);
        outputs.push(new_output);
    }
    new_instructions.push(Instruction::new(instruction.operation, inputs, outputs, instruction.regions));
    atom_id_mapping
        .get(&atom_id)
        .copied()
        .ok_or(ProgramError::MalformedProgram("remapped instruction output was missing".to_string()))
}
#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::cell::Cell;
    use std::fmt::Display;
    use std::rc::Rc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_macros::Parameter;

    use crate::backends::array_programs::{ArrayIrOperation, ArrayIrValue};
    use crate::backends::arrays::{Array, ArrayOperation};
    use crate::backends::scalars::{Scalar, ScalarOperation};
    use crate::macros::check_count;
    use crate::operations::compare::{CompareOperation, ComparisonDirection};
    use crate::operations::control_flow::{ConditionOperation, ScanOperation, WhileOperation};
    use crate::operations::debugging::PrintOperation;
    use crate::operations::math::{AddOperation, MulOperation, NegOperation};
    use crate::parameters::Placeholder;
    use crate::programs::ProgramBuilder;
    use crate::programs::effects::{Effect, Effects};
    use crate::programs::operations::OperationFormatter;
    use crate::programs::regions::RegionSlot;
    use crate::programs::types::TypeError;
    use crate::tests::TestRegionOperation;
    use crate::types::{ArrayIrType, ArrayType, DataType, Dimension, DimensionBounds, DimensionVariable, Shape};

    use super::*;

    #[derive(Clone, Debug)]
    struct LongMetadataOperation;

    impl LongMetadataOperation {
        const METADATA_VALUE: &str = concat!(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "aaaaaaaaaaaaaaaaaaaa",
        );
    }

    impl Operation for LongMetadataOperation {
        type Type = DataType;

        #[inline]
        fn name(&self) -> &'static str {
            "long_metadata"
        }

        fn infer_output_types(
            &self,
            input_types: &[DataType],
            _region_interfaces: &[RegionInterface<DataType>],
        ) -> Result<Vec<DataType>, TypeError> {
            check_count!("input", input_types, 1, TypeError);
            Ok(vec![input_types[0].clone()])
        }

        fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
            OperationFormatter::new(formatter, indentation, self.name())?
                .bracketed(|operation| operation.field("value", Self::METADATA_VALUE))
        }
    }

    /// Effectful test operation with no results, used to pin simplification's zero-output liveness behavior.
    #[derive(Clone, Debug)]
    struct ZeroOutputEffectOperation;

    impl Operation for ZeroOutputEffectOperation {
        type Type = DataType;

        fn name(&self) -> &'static str {
            "zero_output_effect"
        }

        fn infer_output_types(
            &self,
            input_types: &[DataType],
            _region_interfaces: &[RegionInterface<DataType>],
        ) -> Result<Vec<DataType>, TypeError> {
            check_count!("input", input_types, 1, TypeError);
            Ok(Vec::new())
        }

        fn effects(&self) -> Effects {
            Effects::single(Effect::OrderedIo)
        }
    }

    /// Pure test operation whose attached region is transform metadata rather than an executable computation.
    #[derive(Clone, Debug)]
    enum DormantRegionOperation {
        Dormant,
        Effectful,
    }

    impl Operation for DormantRegionOperation {
        type Type = DataType;

        fn name(&self) -> &'static str {
            match self {
                Self::Dormant => "dormant_region",
                Self::Effectful => "effectful",
            }
        }

        fn region_slots(&self) -> &'static [RegionSlot] {
            match self {
                Self::Dormant => const { &[RegionSlot::rule("rule")] },
                Self::Effectful => &[],
            }
        }

        fn infer_output_types(
            &self,
            input_types: &[DataType],
            _region_interfaces: &[RegionInterface<DataType>],
        ) -> Result<Vec<DataType>, TypeError> {
            check_count!("input", input_types, 1, TypeError);
            Ok(input_types.to_vec())
        }

        fn effects(&self) -> Effects {
            if matches!(self, Self::Effectful) { Effects::single(Effect::OrderedIo) } else { Effects::PURE }
        }
    }

    #[test]
    fn test_program() {
        // Test simple program with one argument.
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let i0 = builder.add_input(DataType::F64);
        let c0 = builder.add_constant(Scalar::from(3.0f64));
        let o0 = builder.add_instruction(AddOperation::new(), Vec::new(), vec![i0, c0]).unwrap()[0];
        let program = builder.build::<Scalar, Scalar>(vec![o0], Placeholder, Placeholder).unwrap();
        assert_eq!(program.input_types(), vec![DataType::F64]);
        assert_eq!(program.output_types(), vec![DataType::F64]);
        let input = program.input().unwrap();
        let output = program.output().unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = const
                    %2:f64 = add %0 %1
                in (%2)
            "}
            .trim_end(),
        );
        assert!(matches!(input, Atom::Variable(r#type) if r#type == DataType::F64));
        assert!(matches!(output, Atom::Variable(r#type) if r#type == DataType::F64));

        // Test simple program with two arguments.
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let i0 = builder.add_input(DataType::F64);
        let i1 = builder.add_input(DataType::F64);
        let v0 = builder.add_instruction(NegOperation::new(), Vec::new(), vec![i0]).unwrap()[0];
        let o0 = builder.add_instruction(AddOperation::new(), Vec::new(), vec![v0, i1]).unwrap()[0];
        let program = builder
            .build::<(Scalar, Scalar), Scalar>(vec![o0], (Placeholder, Placeholder), Placeholder)
            .unwrap();
        assert_eq!(program.input_types(), vec![DataType::F64, DataType::F64]);
        assert_eq!(program.output_types(), vec![DataType::F64]);
        let input = program.input().unwrap();
        let output = program.output().unwrap();
        assert_eq!(program.interpret((Scalar::from(2.0), Scalar::from(3.0))), Ok(Scalar::from(1.0)));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64, %1:f64 .
                let %2:f64 = neg %0
                    %3:f64 = add %2 %1
                in (%3)
            "}
            .trim_end(),
        );
        assert!(matches!(input.0, Atom::Variable(r#type) if r#type == DataType::F64));
        assert!(matches!(input.1, Atom::Variable(r#type) if r#type == DataType::F64));
        assert!(matches!(output, Atom::Variable(r#type) if r#type == DataType::F64));

        // Test a program that contains an operation with long metadata that should be rendered on multiple lines.
        let mut builder = ProgramBuilder::<Scalar, LongMetadataOperation>::new();
        let i0 = builder.add_input(DataType::F64);
        let o0 = builder.add_instruction(LongMetadataOperation, Vec::new(), vec![i0]).unwrap()[0];
        let program = builder.build::<Scalar, Scalar>(vec![o0], Placeholder, Placeholder).unwrap();
        let input = program.input().unwrap();
        let output = program.output().unwrap();
        assert_eq!(
            program.to_string(),
            format!(
                indoc! {"
                    lambda %0:f64 .
                    let %1:f64 = long_metadata [
                        value={metadata_value},
                    ] %0
                    in (%1)
                "},
                metadata_value = LongMetadataOperation::METADATA_VALUE,
            )
            .trim_end()
        );
        assert!(matches!(input, Atom::Variable(r#type) if r#type == DataType::F64));
        assert!(matches!(output, Atom::Variable(r#type) if r#type == DataType::F64));

        // Test a program with two outputs that are copies of the same value.
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let i0 = builder.add_input(DataType::F32);
        let o0 = builder.add_instruction(AddOperation::new(), Vec::new(), vec![i0, i0]).unwrap()[0];
        let program = builder
            .build::<Scalar, (Scalar, Scalar)>(vec![o0, o0], Placeholder, (Placeholder, Placeholder))
            .unwrap();
        let input = program.input().unwrap();
        let output = program.output().unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32 .
                let %1:f32 = add %0 %0
                in (%1, %1)
            "}
            .trim_end(),
        );
        assert!(matches!(input, Atom::Variable(r#type) if r#type == DataType::F32));
        assert!(matches!(output.0, Atom::Variable(r#type) if r#type == DataType::F32));
        assert!(matches!(output.1, Atom::Variable(r#type) if r#type == DataType::F32));

        // Test a case where we have an output atom with no parent instruction.
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        builder.add_input(DataType::F64);
        let o0 = builder.add_variable(DataType::F64);
        assert!(matches!(
            builder.build::<Scalar, Scalar>(vec![o0], Placeholder, Placeholder),
            Err(ProgramError::MalformedProgram(message)) if message == "variable atom has no owning instruction",
        ));

        // Test a case where we have an instruction input atom with no parent instruction.
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let i0 = builder.add_input(DataType::F64);
        let v0 = builder.add_variable(DataType::F64);
        let o0 = builder.add_instruction(AddOperation::new(), Vec::new(), vec![i0, v0]).unwrap()[0];
        assert!(matches!(
            builder.build::<Scalar, Scalar>(vec![o0], Placeholder, Placeholder),
            Err(ProgramError::MalformedProgram(message)) if message == "variable atom has no owning instruction",
        ));
    }

    #[test]
    fn test_program_instruction_by_output() {
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let input = builder.add_input(DataType::F64);
        let constant = builder.add_constant(Scalar::from(3.0f64));
        let scaled = builder.add_instruction(NegOperation::new(), Vec::new(), vec![input]).unwrap()[0];
        let output = builder.add_instruction(AddOperation::new(), Vec::new(), vec![scaled, constant]).unwrap()[0];
        let dead_output = builder.add_instruction(NegOperation::new(), Vec::new(), vec![input]).unwrap()[0];
        let program = builder.build::<Scalar, Scalar>(vec![output], Placeholder, Placeholder).unwrap();

        assert_eq!(
            program.instruction_by_output(),
            vec![
                None,    // `input`
                None,    // `constant`
                Some(0), // `scaled`
                Some(1), // `output`
                Some(2), // `dead_output`
            ],
        );
        assert_eq!(dead_output, AtomId::new(4));
    }

    #[test]
    fn test_program_live_sets() {
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let live_input = builder.add_input(DataType::F64);
        let dead_input = builder.add_input(DataType::F64);
        let live_constant = builder.add_constant(Scalar::from(3.0f64));
        let dead_constant = builder.add_constant(Scalar::from(5.0f64));
        let scaled = builder.add_instruction(NegOperation::new(), Vec::new(), vec![live_input]).unwrap()[0];
        let output = builder.add_instruction(AddOperation::new(), Vec::new(), vec![scaled, live_constant]).unwrap()[0];
        let dead_output =
            builder.add_instruction(AddOperation::new(), Vec::new(), vec![dead_input, dead_constant]).unwrap()[0];
        let program = builder
            .build::<(Scalar, Scalar), Scalar>(vec![output], (Placeholder, Placeholder), Placeholder)
            .unwrap();
        let live_sets = program.live_sets();
        assert_eq!(
            live_sets.atoms(),
            &[
                true,  // `live_input`
                false, // `dead_input`
                true,  // `live_constant`
                false, // `dead_constant`
                true,  // `scaled`
                true,  // `output`
                false, // `dead_output`
            ],
        );
        assert_eq!(
            live_sets.instructions(),
            &[
                true,  // `scaled`
                true,  // `output`
                false, // `dead_output`
            ],
        );
        assert_eq!(dead_output, AtomId::new(6));

        let live_sets = program
            .live_sets_with(|_, instruction, _, input_liveness| {
                input_liveness.resize(instruction.inputs().len(), false);
                if let Some(first_input_liveness) = input_liveness.first_mut() {
                    *first_input_liveness = true;
                }
                Ok(())
            })
            .unwrap();
        assert_eq!(
            live_sets.atoms(),
            &[
                true,  // `live_input`
                false, // `dead_input`
                false, // `live_constant` (dropped: only the first `add` input stays live)
                false, // `dead_constant`
                true,  // `scaled`
                true,  // `output`
                false, // `dead_output`
            ],
        );
        assert_eq!(live_sets.instructions(), &[true, true, false]);

        let live_sets = program.live_sets_for_atoms(&[scaled]).unwrap();
        assert_eq!(
            live_sets.atoms(),
            &[
                true,  // `live_input`
                false, // `dead_input`
                false, // `live_constant`
                false, // `dead_constant`
                true,  // `scaled`
                false, // `output`
                false, // `dead_output`
            ],
        );
        assert_eq!(
            live_sets.instructions(),
            &[
                true,  // `scaled`
                false, // `output`
                false, // `dead_output`
            ],
        );
        assert!(matches!(
            program.live_sets_for_atoms(&[AtomId::new(99)]),
            Err(ProgramError::UnboundAtomId { id }) if id == AtomId::new(99),
        ));

        let propagation_calls = Cell::new(0);
        let live_sets = program
            .live_sets_for_atoms_with(&[scaled], |source_program, instruction, output_liveness, input_liveness| {
                assert_eq!(source_program.input_ids(), &[live_input, dead_input]);
                assert_eq!(instruction.outputs(), &[scaled]);
                assert_eq!(output_liveness, &[true]);
                assert!(input_liveness.is_empty());
                propagation_calls.set(propagation_calls.get() + 1);
                input_liveness.resize(instruction.inputs().len(), true);
                Ok(())
            })
            .unwrap();
        assert_eq!(propagation_calls.get(), 1);
        assert_eq!(
            live_sets.atoms(),
            &[
                true,  // `live_input`
                false, // `dead_input`
                false, // `live_constant`
                false, // `dead_constant`
                true,  // `scaled`
                false, // `output`
                false, // `dead_output`
            ],
        );
        assert_eq!(live_sets.instructions(), &[true, false, false]);
    }

    #[test]
    fn test_program_map_operations() {
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let input = builder.add_input(DataType::F64);
        let constant = builder.add_constant(Scalar::from(3.0f64));
        let negated = builder.add_instruction(NegOperation::new(), Vec::new(), vec![input]).unwrap()[0];
        let combined = builder.add_instruction(AddOperation::new(), Vec::new(), vec![negated, constant]).unwrap()[0];
        let output = builder
            .add_instruction(CompareOperation::new(ComparisonDirection::LessThan), Vec::new(), vec![combined, constant])
            .unwrap()[0];
        let program = builder.build::<Scalar, Scalar>(vec![output], Placeholder, Placeholder).unwrap();

        // `map_operations` rebuilds the value graph while rewriting operations: the binary `add` is replaced by a
        // different operation (`mul`), the `compare` payload field is rewritten in place (its direction is flipped),
        // and the unary `neg` is forwarded unchanged. The atom table and rendered structure are preserved.
        let mapped = program
            .map_operations(|operation| {
                Ok::<_, ProgramError>(match operation {
                    ScalarOperation::Compare(operation) => {
                        assert_eq!(operation.direction(), ComparisonDirection::LessThan);
                        ScalarOperation::Compare(CompareOperation::new(ComparisonDirection::GreaterThan))
                    }
                    ScalarOperation::Add(_) => ScalarOperation::Mul(MulOperation::new()),
                    operation => operation.clone(),
                })
            })
            .unwrap();

        // Original: `(-input + 3) < 3`, so for `input = 2` this is `1 < 3 = true`.
        assert_eq!(program.interpret(Scalar::from(2.0f64)), Ok(Scalar::from(true)));
        // Mapped: `(-input * 3) > 3`, so for `input = 2` this is `-6 > 3 = false`.

        assert_eq!(mapped.interpret(Scalar::from(2.0f64)), Ok(Scalar::from(false)));

        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = const
                    %2:f64 = neg %0
                    %3:f64 = add %2 %1
                    %4:bool = compare [direction=LessThan] %3 %1
                in (%4)
            "}
            .trim_end(),
        );

        assert_eq!(
            mapped.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = const
                    %2:f64 = neg %0
                    %3:f64 = mul %2 %1
                    %4:bool = compare [direction=GreaterThan] %3 %1
                in (%4)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_program_to_flat_program_and_into_flat_program() {
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let i0 = builder.add_input(DataType::F64);
        let i1 = builder.add_input(DataType::F64);
        let v0 = builder.add_instruction(NegOperation::new(), Vec::new(), vec![i0]).unwrap()[0];
        let o0 = builder.add_instruction(AddOperation::new(), Vec::new(), vec![v0, i1]).unwrap()[0];
        let program = builder
            .build::<(Scalar, Scalar), Scalar>(vec![o0], (Placeholder, Placeholder), Placeholder)
            .unwrap();

        let flat_program = program.to_flat_program();
        assert_eq!(flat_program.input_structure(), &vec![Placeholder, Placeholder]);
        assert_eq!(flat_program.output_structure(), &vec![Placeholder]);
        assert_eq!(flat_program.interpret(vec![Scalar::from(2.0), Scalar::from(3.0)]), Ok(vec![Scalar::from(1.0)]));

        let flat_program = program.into_flat_program();
        assert_eq!(flat_program.input_structure(), &vec![Placeholder, Placeholder]);
        assert_eq!(flat_program.output_structure(), &vec![Placeholder]);
        assert_eq!(flat_program.interpret(vec![Scalar::from(2.0), Scalar::from(3.0)]), Ok(vec![Scalar::from(1.0)]));
    }

    #[test]
    fn test_program_construction_and_restructuring_validate_boundaries() {
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let input = builder.add_input(DataType::F64);
        let program = builder.build::<Scalar, Scalar>(vec![input], Placeholder, Placeholder).unwrap();
        let entry = program.entry();
        let regions = program.regions.clone().into_regions();

        assert!(matches!(
            Program::<Scalar, ScalarOperation<Scalar>, Vec<Scalar>, Vec<Scalar>>::new(
                Vec::new(),
                vec![Placeholder],
                regions.clone(),
                entry,
            ),
            Err(ProgramError::InvalidInputCount { actual: 1, expected: 0 }),
        ));
        assert!(matches!(
            program.to_flat_program().restructured::<Vec<Scalar>, Vec<Scalar>>(Vec::new(), vec![Placeholder]),
            Err(ProgramError::InvalidInputCount { actual: 1, expected: 0 }),
        ));

        let mut unreachable_regions = regions;
        unreachable_regions.push(unreachable_regions[0].clone());
        assert!(matches!(
            Program::<Scalar, ScalarOperation<Scalar>, Vec<Scalar>, Vec<Scalar>>::new(
                vec![Placeholder],
                vec![Placeholder],
                unreachable_regions.clone(),
                RegionId::new(0),
            ),
            Err(ProgramError::MalformedProgram(message))
                if message == "entry region ^0 must be the final region in the arena",
        ));
        assert!(matches!(
            Program::<Scalar, ScalarOperation<Scalar>, Vec<Scalar>, Vec<Scalar>>::new(
                vec![Placeholder],
                vec![Placeholder],
                unreachable_regions,
                RegionId::new(1),
            ),
            Err(ProgramError::MalformedProgram(message))
                if message == "region ^0 is not reachable from the program entry region",
        ));
    }

    #[test]
    fn test_program_into_unprojected() {
        let dynamic_dimension = DimensionVariable::new("elements", DimensionBounds::new(1, Some(8)).unwrap());
        let array_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(dynamic_dimension)]));

        // Build a scan body containing a while loop so unprojection must recursively promote all three control-flow
        // carriers. The scan capture additionally pins value lifting and capture-order preservation.
        let mut while_condition_builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let while_condition_input = while_condition_builder.add_input(ArrayType::scalar(DataType::F64));
        let false_predicate = while_condition_builder.add_constant(Array::scalar(false));
        let while_condition = while_condition_builder
            .build::<Vec<Array>, Vec<Array>>(vec![false_predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(while_condition.input_ids(), &[while_condition_input]);

        let mut while_body_builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let while_body_input = while_body_builder.add_input(ArrayType::scalar(DataType::F64));
        let while_body = while_body_builder
            .build::<Vec<Array>, Vec<Array>>(vec![while_body_input], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut scan_body_builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let scan_carry = scan_body_builder.add_input(ArrayType::scalar(DataType::F64));
        let scan_element = scan_body_builder.add_input(ArrayType::scalar(DataType::F64));
        let while_condition_region = scan_body_builder.import_program(while_condition);
        let while_body_region = scan_body_builder.import_program(while_body);
        let next_carry = scan_body_builder
            .add_instruction(
                WhileOperation::new().with_iteration_bound(1).unwrap(),
                vec![while_condition_region, while_body_region],
                vec![scan_carry],
            )
            .unwrap()[0];
        let scan_body = scan_body_builder
            .build::<Vec<Array>, Vec<Array>>(
                vec![next_carry, scan_element],
                vec![Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();

        // Build one branch and attach the same imported region to both condition slots. The branch's dynamic type
        // exercises identity preservation, while the otherwise-unused entry constant exercises value lifting.
        let mut branch_builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let branch_input = branch_builder.add_input(array_type.clone());
        let branch_output =
            branch_builder.add_instruction(NegOperation::new(), Vec::new(), vec![branch_input]).unwrap()[0];
        let scan_carry = branch_builder.add_constant(Array::scalar(1.0_f64));
        let scan_stack = branch_builder.add_constant(Array::vector(vec![2.0_f64, 3.0]));
        let scan_capture = Array::vector(vec![5.0_f64, 6.0]);
        let scan_body_region = branch_builder.import_program(scan_body);
        branch_builder
            .add_instruction(
                ScanOperation::<Array>::new(1, 2)
                    .with_reverse(true)
                    .with_unroll(2)
                    .unwrap()
                    .with_captures(vec![scan_capture.clone()]),
                vec![scan_body_region],
                vec![scan_carry, scan_stack],
            )
            .unwrap();
        let branch = branch_builder
            .build::<Vec<Array>, Vec<Array>>(vec![branch_output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean));
        let operand = builder.add_input(array_type.clone());
        let constant = builder.add_constant(Array::scalar(2.0_f64));
        let shared_branch = builder.import_program(branch);
        let output = builder
            .add_instruction(
                ConditionOperation::<Array>::new(),
                vec![shared_branch, shared_branch],
                vec![predicate, operand],
            )
            .unwrap()[0];
        let source = builder
            .build::<(Array, Array), Array>(vec![output], (Placeholder, Placeholder), Placeholder)
            .unwrap();
        let source_signature = source.type_identity_signature().clone();
        let source_effects = source.effects();
        let source_region_count = source.regions().len();
        let source_entry = source.entry();
        let source_structure = source
            .regions()
            .iter()
            .map(|region| {
                (
                    region.atoms().len(),
                    region.input_ids().to_vec(),
                    region.output_ids().to_vec(),
                    region
                        .instructions()
                        .iter()
                        .map(|instruction| {
                            (
                                instruction.inputs().to_vec(),
                                instruction.outputs().to_vec(),
                                instruction.regions().to_vec(),
                            )
                        })
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>();

        let composite: Program<
            ArrayIrValue<Array>,
            ArrayIrOperation<Array>,
            (ArrayIrValue<Array>, ArrayIrValue<Array>),
            ArrayIrValue<Array>,
        > = source.into_unprojected().unwrap();

        assert_eq!(
            composite.input_types(),
            vec![ArrayIrType::Array(ArrayType::scalar(DataType::Boolean)), ArrayIrType::Array(array_type.clone()),],
        );
        assert_eq!(composite.output_types(), vec![ArrayIrType::Array(array_type)]);
        assert_eq!(composite.input_structure(), &(Placeholder, Placeholder));
        assert_eq!(composite.output_structure(), &Placeholder);
        assert_eq!(composite.effects(), source_effects);
        assert_eq!(composite.type_identity_signature(), &source_signature);
        assert_eq!(composite.regions().len(), source_region_count);
        assert_eq!(composite.entry(), source_entry);
        for ((atom_count, input_ids, output_ids, instructions), region) in
            source_structure.iter().zip(composite.regions().iter())
        {
            assert_eq!(region.atoms().len(), *atom_count);
            assert_eq!(region.input_ids(), input_ids);
            assert_eq!(region.output_ids(), output_ids);
            assert_eq!(region.instructions().len(), instructions.len());
            for ((inputs, outputs, regions), instruction) in instructions.iter().zip(region.instructions()) {
                assert_eq!(instruction.inputs(), inputs);
                assert_eq!(instruction.outputs(), outputs);
                assert_eq!(instruction.regions(), regions);
            }
        }

        let entry = composite.entry_region();
        assert_eq!(entry.instructions().len(), 1);
        let branch_region = entry.instructions()[0].regions()[0];
        assert_eq!(entry.instructions()[0].regions(), &[branch_region, branch_region]);
        assert!(matches!(entry.instructions()[0].operation(), ArrayIrOperation::Condition(_),));
        assert!(matches!(
            &entry.atoms()[constant.index()],
            Atom::Constant(ArrayIrValue::Array(value)) if value == &Array::scalar(2.0_f64),
        ));
        let branch = composite.region(branch_region).unwrap();
        assert!(matches!(branch.instructions()[0].operation(), ArrayIrOperation::Array(ArrayOperation::Neg(_))));
        let scan_instruction = branch
            .instructions()
            .iter()
            .find(|instruction| matches!(instruction.operation(), ArrayIrOperation::Scan(_)))
            .unwrap();
        let ArrayIrOperation::Scan(scan) = scan_instruction.operation() else {
            unreachable!();
        };
        assert_eq!(scan.carry_count(), 1);
        assert_eq!(scan.length(), &Dimension::Static(2));
        assert!(scan.reverse());
        assert_eq!(scan.unroll(), 2);
        assert_eq!(scan.captures(), &[ArrayIrValue::Array(scan_capture)]);
        let scan_body = composite.region(scan_instruction.regions()[0]).unwrap();
        let while_instruction = scan_body
            .instructions()
            .iter()
            .find(|instruction| matches!(instruction.operation(), ArrayIrOperation::While(_)))
            .unwrap();
        assert_eq!(while_instruction.regions().len(), 2);
        assert!(matches!(while_instruction.operation(), ArrayIrOperation::While(_)));
    }

    #[test]
    fn test_program_simplified() {
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let i0 = builder.add_input(DataType::F64);
        let c0 = builder.add_constant(Scalar::from(2.0f64));
        let c1 = builder.add_constant(Scalar::from(3.0f64));
        let _ = builder.add_instruction(AddOperation::new(), Vec::new(), vec![i0, c0]).unwrap()[0];
        let v1 = builder.add_instruction(AddOperation::new(), Vec::new(), vec![i0, c1]).unwrap()[0];
        let program = builder
            .build::<Scalar, (Scalar, Scalar)>(vec![v1, v1], Placeholder, (Placeholder, Placeholder))
            .unwrap();
        let simplified = program.simplified().unwrap();

        assert_eq!(c0, AtomId::new(1));
        assert_eq!(simplified.interpret(Scalar::from(2.0f64)), Ok((Scalar::from(5.0f64), Scalar::from(5.0f64))));
        assert_eq!(
            simplified.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = const
                    %2:f64 = add %0 %1
                in (%2, %2)
            "}
            .trim_end(),
        );
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = const
                    %2:f64 = const
                    %3:f64 = add %0 %1
                    %4:f64 = add %0 %2
                in (%4, %4)
            "}
            .trim_end(),
        );

        // The pure program above reports no effects, and simplification removed its dead `add` as asserted. Effectful
        // instructions, in contrast, are kept alive by simplification even when they are dead code: nothing consumes
        // the print's output below, so only its effect keeps it in the simplified program.
        assert_eq!(program.effects(), Effects::PURE);
        let build = || {
            let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
            let input = builder.add_input(DataType::F64);
            let doubled = builder.add_instruction(AddOperation::new(), Vec::new(), vec![input, input]).unwrap()[0];
            let _printed = builder.add_instruction(PrintOperation::new("x"), Vec::new(), vec![input]).unwrap()[0];
            builder.build::<Scalar, Scalar>(vec![doubled], Placeholder, Placeholder).unwrap()
        };
        let effectful = build();
        assert_eq!(effectful.effects(), Effects::single(Effect::OrderedIo));
        let expected = indoc! {"
            lambda %0:f64 .
            let %1:f64 = print [label=x] %0
                %2:f64 = add %0 %0
            in (%2)
        "}
        .trim_end();
        assert_eq!(effectful.simplified().unwrap().to_string(), expected);
        assert_eq!(build().into_simplified().unwrap().to_string(), expected);

        // An effectful instruction with no outputs must itself be rooted: there is no result atom from which either
        // simplification implementation could otherwise discover it.
        let build_zero_output_effect = || {
            let mut builder = ProgramBuilder::<Scalar, ZeroOutputEffectOperation>::new();
            let input = builder.add_input(DataType::F64);
            assert!(builder.add_instruction(ZeroOutputEffectOperation, Vec::new(), vec![input]).unwrap().is_empty());
            builder.build::<Scalar, Vec<Scalar>>(Vec::new(), Placeholder, Vec::new()).unwrap()
        };
        let zero_output_effect = build_zero_output_effect();
        assert_eq!(zero_output_effect.simplified().unwrap().instructions().len(), 1);
        assert_eq!(build_zero_output_effect().into_simplified().unwrap().instructions().len(), 1);
    }

    #[test]
    fn test_program_into_simplified() {
        #[derive(Debug, Parameter)]
        struct CloneCountingValue {
            value: f64,
            clone_count: Rc<Cell<usize>>,
        }

        impl CloneCountingValue {
            fn new(value: f64, clone_count: Rc<Cell<usize>>) -> Self {
                Self { value, clone_count }
            }
        }

        impl Clone for CloneCountingValue {
            fn clone(&self) -> Self {
                self.clone_count.set(self.clone_count.get() + 1);
                Self { value: self.value, clone_count: Rc::clone(&self.clone_count) }
            }
        }

        impl Display for CloneCountingValue {
            fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(formatter, "{}", self.value)
            }
        }

        impl Typed for CloneCountingValue {
            type Type = DataType;

            fn r#type(&self) -> Cow<'_, DataType> {
                Cow::Owned(DataType::F64)
            }
        }

        impl Value for CloneCountingValue {
            type DispatchDomain = crate::EagerContext<Self>;
            type ExecutionDomain = crate::EagerContext<Self>;

            fn dispatch_domain(&self) -> crate::EagerContext<Self> {
                crate::EagerContext::new()
            }

            fn execution_domain(&self) -> crate::EagerContext<Self> {
                crate::EagerContext::new()
            }
        }

        let value_clone_count = Rc::new(Cell::new(0));
        let mut builder = ProgramBuilder::<_, ScalarOperation<CloneCountingValue>>::new();
        let i0 = builder.add_input(DataType::F64);
        let c0 = builder.add_constant(CloneCountingValue::new(2.0, Rc::clone(&value_clone_count)));
        let c1 = builder.add_constant(CloneCountingValue::new(3.0, Rc::clone(&value_clone_count)));
        let v0 = builder.add_instruction(AddOperation::new(), Vec::new(), vec![i0, c0]).unwrap()[0];
        let v1 = builder.add_instruction(AddOperation::new(), Vec::new(), vec![i0, c1]).unwrap()[0];
        let program = builder
            .build::<CloneCountingValue, (CloneCountingValue, CloneCountingValue)>(
                vec![v1, v1],
                Placeholder,
                (Placeholder, Placeholder),
            )
            .unwrap();

        assert_eq!(v0, AtomId::new(3));
        assert_eq!(v1, AtomId::new(4));
        assert_eq!(value_clone_count.get(), 0);
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = const
                    %2:f64 = const
                    %3:f64 = add %0 %1
                    %4:f64 = add %0 %2
                in (%4, %4)
            "}
            .trim_end(),
        );

        let simplified = program.into_simplified().unwrap();
        assert_eq!(value_clone_count.get(), 0);
        assert_eq!(simplified.input_ids(), vec![AtomId::new(0)]);
        assert_eq!(simplified.output_ids(), vec![AtomId::new(2), AtomId::new(2)]);
        assert_eq!(simplified.atoms().len(), 3);
        assert!(matches!(simplified.atoms().get(1), Some(Atom::Constant(value)) if value.value == 3.0));
        assert_eq!(simplified.instructions().len(), 1);
        assert_eq!(simplified.instructions()[0].inputs(), vec![AtomId::new(0), AtomId::new(1)]);
        assert_eq!(simplified.instructions()[0].outputs(), vec![AtomId::new(2)]);
        assert_eq!(
            simplified.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = const
                    %2:f64 = add %0 %1
                in (%2, %2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_program_filtered() {
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let i0 = builder.add_input(DataType::F64);
        let i1 = builder.add_input(DataType::F64);
        let c0 = builder.add_constant(Scalar::from(2.0f64));
        let v0 = builder.add_instruction(NegOperation::new(), Vec::new(), vec![i0]).unwrap()[0];
        let v1 = builder.add_instruction(AddOperation::new(), Vec::new(), vec![v0, c0]).unwrap()[0];
        let program = builder
            .build::<(Scalar, Scalar), Scalar>(vec![v1], (Placeholder, Placeholder), Placeholder)
            .unwrap();

        // Dead inputs are pruned and constants are lifted: `i1` is dead for `v1`, so it is dropped,
        // and `c0` is rebuilt into the projected program.
        let (pruned, pruned_live) = program.filtered(&[i0, i1], &[v1], &[]).unwrap();
        assert_eq!(pruned_live, vec![0]);
        assert_eq!(pruned.input_ids().len(), 1);
        assert_eq!(pruned.interpret(vec![Scalar::from(4.0)]), Ok(vec![Scalar::from(-2.0)]));

        // Selecting an intermediate atom (i.e., `v0`) as the output drops the downstream `add`
        // and the now-dead constant.
        let (intermediate, intermediate_live) = program.filtered(&[i0], &[v0], &[]).unwrap();
        assert_eq!(intermediate_live, vec![0]);
        assert_eq!(intermediate.instructions().len(), 1);
        assert_eq!(intermediate.interpret(vec![Scalar::from(5.0)]), Ok(vec![Scalar::from(-5.0)]));

        // Forwarding an input directly as an output yields an instruction-free program over only that input.
        let (forwarded, forwarded_live) = program.filtered(&[i0, i1], &[i0], &[]).unwrap();
        assert_eq!(forwarded_live, vec![0]);
        assert_eq!(forwarded.instructions().len(), 0);
        assert_eq!(forwarded.interpret(vec![Scalar::from(7.0)]), Ok(vec![Scalar::from(7.0)]));

        // Reaching a variable that is neither a selected input nor produced by an instruction is rejected:
        // `v1` depends on `i0`, which is omitted from the selected inputs here.
        assert!(matches!(program.filtered(&[i1], &[v1], &[]), Err(ProgramError::MalformedProgram(_))));

        // Providing the same input atom more than once is rejected.
        assert!(matches!(program.filtered(&[i0, i0], &[v1], &[]), Err(ProgramError::MalformedProgram(_))));

        // A keep-alive entry naming an otherwise-pruned atom retains its producing instruction chain without
        // widening the projection's outputs: projecting onto `v0` alone drops the downstream `add` and the constant,
        // while keeping `v1` alive pulls them back in.
        let (kept, kept_live) = program.filtered(&[i0], &[v0], &[v1]).unwrap();
        assert_eq!(kept_live, vec![0]);
        assert_eq!(kept.instructions().len(), 2);
        assert_eq!(kept.output_ids().len(), 1);
        assert_eq!(kept.interpret(vec![Scalar::from(5.0)]), Ok(vec![Scalar::from(-5.0)]));

        // A keep-alive entry naming a dead input pins it as a live public input instead of pruning it.
        let (pinned, pinned_live) = program.filtered(&[i0, i1], &[v1], &[i1]).unwrap();
        assert_eq!(pinned_live, vec![0, 1]);
        assert_eq!(pinned.input_ids().len(), 2);
        assert_eq!(pinned.interpret(vec![Scalar::from(4.0), Scalar::from(9.0)]), Ok(vec![Scalar::from(-2.0)]));

        // Observable effects are implicit roots even when neither their results nor their operands are explicitly
        // kept alive.
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let input = builder.add_input(DataType::F64);
        let output = builder.add_instruction(NegOperation::new(), Vec::new(), vec![input]).unwrap()[0];
        builder.add_instruction(PrintOperation::new("x"), Vec::new(), vec![input]).unwrap();
        let effectful = builder.build::<Scalar, Scalar>(vec![output], Placeholder, Placeholder).unwrap();
        let (filtered, live) = effectful.filtered(&[input], &[output], &[]).unwrap();
        assert_eq!(live, vec![0]);
        assert_eq!(filtered.instructions().len(), 2);
        assert_eq!(filtered.effects(), Effects::single(Effect::OrderedIo));

        // Zero-output effects are preserved explicitly because they have no result atom that can serve as a root.
        let mut builder = ProgramBuilder::<Scalar, ZeroOutputEffectOperation>::new();
        let input = builder.add_input(DataType::F64);
        builder.add_instruction(ZeroOutputEffectOperation, Vec::new(), vec![input]).unwrap();
        let effectful = builder.build::<Scalar, Vec<Scalar>>(Vec::new(), Placeholder, Vec::new()).unwrap();
        let (filtered, live) = effectful.filtered(&[input], &[], &[]).unwrap();
        assert_eq!(live, vec![0]);
        assert_eq!(filtered.instructions().len(), 1);
    }

    #[test]
    fn test_program_into_filtered() {
        // Build the same program twice, so that the consuming `into_filtered` can be compared
        // against the borrowing `filter`.
        let build = || {
            let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
            let i0 = builder.add_input(DataType::F64);
            let i1 = builder.add_input(DataType::F64);
            let c0 = builder.add_constant(Scalar::from(2.0f64));
            let v0 = builder.add_instruction(NegOperation::new(), Vec::new(), vec![i0]).unwrap()[0];
            let v1 = builder.add_instruction(AddOperation::new(), Vec::new(), vec![v0, c0]).unwrap()[0];
            let program = builder
                .build::<(Scalar, Scalar), Scalar>(vec![v1], (Placeholder, Placeholder), Placeholder)
                .unwrap();
            (program, i0, i1, v0, v1)
        };

        let (borrowed_program, b_i0, b_i1, _, b_v1) = build();
        let (borrowed, borrowed_live) = borrowed_program.filtered(&[b_i0, b_i1], &[b_v1], &[]).unwrap();
        let (owned_program, o_i0, o_i1, _, o_v1) = build();
        let (owned, owned_live) = owned_program.into_filtered(&[o_i0, o_i1], &[o_v1], &[]).unwrap();

        // The consuming variant drops the dead input, lifts the constant, and is identical to the borrowing `filter`.
        assert_eq!(owned_live, vec![0]);
        assert_eq!(owned_live, borrowed_live);
        assert_eq!(owned.input_ids().len(), 1);
        assert_eq!(owned.interpret(vec![Scalar::from(4.0)]), Ok(vec![Scalar::from(-2.0)]));
        assert_eq!(owned.to_string(), borrowed.to_string());

        // Keep-alive entries follow the same contract as the borrowing `filtered`: keeping `v1` alive moves its
        // otherwise-pruned `add` and constant into the projection onto `v0`, without widening the outputs.
        let (kept_program, k_i0, _, k_v0, k_v1) = build();
        let (kept, kept_live) = kept_program.into_filtered(&[k_i0], &[k_v0], &[k_v1]).unwrap();
        assert_eq!(kept_live, vec![0]);
        assert_eq!(kept.instructions().len(), 2);
        assert_eq!(kept.output_ids().len(), 1);
        assert_eq!(kept.interpret(vec![Scalar::from(4.0)]), Ok(vec![Scalar::from(-4.0)]));

        let mut builder = ProgramBuilder::<Scalar, ZeroOutputEffectOperation>::new();
        let input = builder.add_input(DataType::F64);
        builder.add_instruction(ZeroOutputEffectOperation, Vec::new(), vec![input]).unwrap();
        let effectful = builder.build::<Scalar, Vec<Scalar>>(Vec::new(), Placeholder, Vec::new()).unwrap();
        let (filtered, live) = effectful.into_filtered(&[input], &[], &[]).unwrap();
        assert_eq!(live, vec![0]);
        assert_eq!(filtered.instructions().len(), 1);
    }

    #[test]
    fn test_program_render_multi_region() {
        // A shared region renders its body once (labeled with its identifier, at its first reference) and later
        // references render as that identifier alone, while a singly referenced region renders nested inline.
        // Regions are labeled with the operation-declared names.
        let mut builder = ProgramBuilder::<Array, TestRegionOperation>::new();
        let mut region_builder = ProgramBuilder::<Array, TestRegionOperation>::new();
        let region_input = region_builder.add_input(ArrayType::scalar(DataType::F64));
        let region_program = region_builder
            .build::<Vec<Array>, Vec<Array>>(vec![region_input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let shared = builder.import_region(region_program.entry_region_ref());
        let inline = builder.import_region(region_program.entry_region_ref());
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let first = builder
            .add_instruction(
                TestRegionOperation::WithRegions(
                    const { &[RegionSlot::computation("first"), RegionSlot::computation("second")] },
                ),
                vec![shared, shared],
                vec![input],
            )
            .unwrap()[0];
        let second = builder
            .add_instruction(
                TestRegionOperation::WithRegions(const { &[RegionSlot::computation("body")] }),
                vec![inline],
                vec![first],
            )
            .unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![second], vec![Placeholder], vec![Placeholder]).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = with_regions %0 [
                    first=^0={
                        lambda %0:f64[] .
                        in (%0)
                    },
                    second=^0,
                ]
                    %2:f64[] = with_regions %1 [
                        body={
                            lambda %0:f64[] .
                            in (%0)
                        },
                    ]
                in (%2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_program_instruction_effects_include_attached_regions() {
        // An instruction whose operation is pure but whose attached region contains an effectful instruction reports
        // impure effects, while a sibling pure instruction stays pure.
        let mut builder = ProgramBuilder::<Array, TestRegionOperation>::new();
        let mut region_builder = ProgramBuilder::<Array, TestRegionOperation>::new();
        let region_input = region_builder.add_input(ArrayType::scalar(DataType::F64));
        let region_output = region_builder
            .add_instruction(TestRegionOperation::Effectful, Vec::new(), vec![region_input])
            .unwrap()[0];
        let region_program = region_builder
            .build::<Vec<Array>, Vec<Array>>(vec![region_output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let sealed = builder.import_region(region_program.entry_region_ref());
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let with_regions = builder
            .add_instruction(
                TestRegionOperation::WithRegions(const { &[RegionSlot::computation("body")] }),
                vec![sealed],
                vec![input],
            )
            .unwrap()[0];
        let output = builder.add_instruction(TestRegionOperation::Add, Vec::new(), vec![input, with_regions]).unwrap();
        let output = output[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        let entry = program.entry();
        assert_eq!(
            program.instruction_effects(InstructionId::new(entry, 0)).unwrap(),
            Effects::single(Effect::OrderedIo),
        );
        assert_eq!(program.instruction_effects(InstructionId::new(entry, 1)).unwrap(), Effects::PURE);
        assert_eq!(program.effects(), Effects::single(Effect::OrderedIo));

        // Effects in transform-only rule regions are dormant during ordinary execution and therefore do not make the
        // containing instruction or program effectful.
        let mut rule_builder = ProgramBuilder::<Scalar, DormantRegionOperation>::new();
        let rule_input = rule_builder.add_input(DataType::F64);
        let rule_output = rule_builder
            .add_instruction(DormantRegionOperation::Effectful, Vec::new(), vec![rule_input])
            .unwrap()[0];
        let rule_program = rule_builder.build::<Scalar, Scalar>(vec![rule_output], Placeholder, Placeholder).unwrap();
        let mut builder = ProgramBuilder::<Scalar, DormantRegionOperation>::new();
        let dormant = builder.import_region(rule_program.entry_region_ref());
        let input = builder.add_input(DataType::F64);
        let output = builder.add_instruction(DormantRegionOperation::Dormant, vec![dormant], vec![input]).unwrap()[0];
        let program = builder.build::<Scalar, Scalar>(vec![output], Placeholder, Placeholder).unwrap();
        assert_eq!(program.instruction_effects(InstructionId::new(program.entry(), 0)).unwrap(), Effects::PURE);
        assert_eq!(program.effects(), Effects::PURE);
    }

    #[test]
    fn test_program_simplified_multi_region() {
        // We use two sealed regions: a pure one (^0) referenced only by a dead instruction, and an effectful one (^1)
        // referenced by another dead instruction. Simplification drops the pure dead instruction together with its
        // region, keeps the effectful dead instruction alive (its attached region's effects are observable), and
        // compacts the surviving region identifiers (the effectful region moves from ^1 to ^0).
        let mut builder = ProgramBuilder::<Array, TestRegionOperation>::new();
        let mut pure_builder = ProgramBuilder::<Array, TestRegionOperation>::new();
        let pure_input = pure_builder.add_input(ArrayType::scalar(DataType::F64));
        let pure_program = pure_builder
            .build::<Vec<Array>, Vec<Array>>(vec![pure_input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let pure_region = builder.import_region(pure_program.entry_region_ref());
        let mut effectful_builder = ProgramBuilder::<Array, TestRegionOperation>::new();
        let effectful_input = effectful_builder.add_input(ArrayType::scalar(DataType::F64));
        let effectful_output = effectful_builder
            .add_instruction(TestRegionOperation::Effectful, Vec::new(), vec![effectful_input])
            .unwrap()[0];
        let effectful_program = effectful_builder
            .build::<Vec<Array>, Vec<Array>>(vec![effectful_output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let effectful_region = builder.import_region(effectful_program.entry_region_ref());
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        builder
            .add_instruction(
                TestRegionOperation::WithRegions(const { &[RegionSlot::computation("body")] }),
                vec![pure_region],
                vec![input],
            )
            .unwrap();
        builder
            .add_instruction(
                TestRegionOperation::WithRegions(const { &[RegionSlot::computation("body")] }),
                vec![effectful_region],
                vec![input],
            )
            .unwrap();
        let output = builder.add_instruction(TestRegionOperation::Add, Vec::new(), vec![input, input]).unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        assert_eq!(program.regions().len(), 3);
        let simplified = program.simplified().unwrap();
        assert_eq!(simplified.regions().len(), 2);
        assert_eq!(simplified.instructions().len(), 2);
        assert_eq!(
            simplified.instructions()[0].operation(),
            &TestRegionOperation::WithRegions(const { &[RegionSlot::computation("body")] }),
        );
        assert_eq!(simplified.instructions()[0].regions(), &[RegionId::new(0)]);
        assert_eq!(simplified.instructions()[1].operation(), &TestRegionOperation::Add);
        assert!(!simplified.region(RegionId::new(0)).unwrap().instructions()[0].operation().effects().is_pure());
        let simplified = program.into_simplified().unwrap();
        assert_eq!(simplified.regions().len(), 2);
        assert_eq!(simplified.instructions().len(), 2);
        assert_eq!(simplified.instructions()[0].regions(), &[RegionId::new(0)]);
    }
}
