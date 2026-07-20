use std::cell::RefCell;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::rc::Rc;

use crate::macros::check_count;
use crate::parameters::{Parameter, Parameterized};
use crate::programs::ProgramError;
use crate::programs::atoms::{Atom, AtomId};
use crate::programs::instructions::Instruction;
use crate::programs::operations::Operation;
use crate::programs::programs::Program;
use crate::programs::regions::{Region, RegionId, RegionInterface, RegionRef};
use crate::programs::types::Typed;
use crate::programs::values::Value;

/// Builder for [`Program`]s. It owns the entry [`Region`] under construction (i.e., its [`Atom`]s, input [`AtomId`]s,
/// and [`Instruction`]s), the previously added non-entry [`Region`]s together with their callee-interning state, and
/// an optional [`ProgramError`] that can be used to signal a failure during program construction. Non-entry regions
/// enter a builder only in sealed form: [`import_region`](Self::import_region) copies complete reachable closures
/// out of immutable regions, [`import_program`](Self::import_program) moves complete owned programs, and
/// [`intern_callee`](Self::intern_callee) reuses imports by [`Rc`] identity. A region can therefore never
/// change after an instruction attaches it.
#[derive(Clone, Debug, Default)]
pub struct ProgramBuilder<V: Typed + Parameter, O> {
    /// [`Atom`]s contained in the entry [`Region`] of the [`Program`] that is being built, in evaluation order.
    pub(crate) atoms: Vec<Atom<V>>,

    /// [`AtomId`]s of the [`Atom`]s that correspond to the inputs (i.e., arguments) of the [`Program`] being built.
    pub(crate) input_ids: Vec<AtomId>,

    /// Ordered sequence of [`Instruction`]s that make up the entry [`Region`] of the [`Program`] being built.
    pub(crate) instructions: Vec<Instruction<O>>,

    /// Sealed non-entry [`Region`]s of the [`Program`] being built, in [`RegionId`] order. Regions are appended
    /// to this list with [`Self::import_region`], [`Self::import_program`], and [`Self::intern_callee`], and
    /// [`Instruction`]s reference them by [`RegionId`].
    pub(crate) regions: Vec<Region<V, O>>,

    /// Callee-interning table mapping each imported callee source to its destination root, keyed by [`Rc`] identity
    /// (i.e., [`Rc::ptr_eq`]). Two imports of the same live source program reuse one callee root, while structurally
    /// equal but independently built programs remain distinct. Storing the [`Rc`] itself both provides the identity
    /// key and keeps the source alive, so a key can never be reused by a later allocation.
    pub(crate) callees: Vec<(Rc<Program<V, O, Vec<V>, Vec<V>>>, RegionId)>,

    /// Optional [`ProgramError`] encountered during program construction that will be propagated via [`Self::build`].
    pub(crate) error: Option<ProgramError>,
}

impl<V: Value, O: Operation<V::Type>> ProgramBuilder<V, O> {
    /// Creates a new [`ProgramBuilder`].
    #[inline]
    pub fn new() -> Self {
        Self {
            atoms: Vec::new(),
            input_ids: Vec::new(),
            instructions: Vec::new(),
            regions: Vec::new(),
            callees: Vec::new(),
            error: None,
        }
    }

    /// Returns the atoms currently owned by this builder.
    #[inline]
    pub fn atoms(&self) -> &[Atom<V>] {
        &self.atoms
    }

    /// Returns the input atom identifiers currently owned by this builder.
    #[inline]
    pub fn input_ids(&self) -> &[AtomId] {
        &self.input_ids
    }

    /// Returns the instructions currently owned by this builder.
    #[inline]
    pub fn instructions(&self) -> &[Instruction<O>] {
        &self.instructions
    }

    /// Returns the currently recorded construction error, if one exists.
    #[inline]
    pub fn error(&self) -> Option<&ProgramError> {
        self.error.as_ref()
    }

    /// Returns a borrowed view of the already sealed builder region that corresponds to the provided [`RegionId`].
    #[inline]
    pub fn region_ref(&self, id: RegionId) -> Result<RegionRef<'_, V, O>, ProgramError> {
        RegionRef::new(self.regions.as_slice(), id)
            .map_err(|_| ProgramError::MalformedProgram(format!("region {id} is not part of this builder")))
    }

    /// Adds an input [`Atom`] to the [`Program`] that is being built with the provided [`Type`](crate::Type).
    #[inline]
    pub fn add_input(&mut self, r#type: V::Type) -> AtomId {
        let id = self.add_variable(r#type);
        self.input_ids.push(id);
        id
    }

    /// Adds the provided value as an [`Atom::Constant`] to the [`Program`] that is being built.
    #[inline]
    pub fn add_constant(&mut self, value: V) -> AtomId {
        let id = AtomId::new(self.atoms.len());
        self.atoms.push(Atom::Constant(value));
        id
    }

    /// Adds an [`Atom::Variable`] to the [`Program`] that is being built with the provided [`Type`](crate::Type).
    #[inline]
    pub fn add_variable(&mut self, r#type: V::Type) -> AtomId {
        let id = AtomId::new(self.atoms.len());
        self.atoms.push(Atom::Variable(r#type));
        id
    }

    /// Adds an [`Instruction`] to the [`Program`] that is being built, that corresponds to an application of the
    /// provided [`Operation`] with the provided previously sealed regions attached in the operation-defined region
    /// order (region-free operations pass an empty `regions` list) to the provided input [`Atom`]s. The number of
    /// attached regions must match the operation's declared [`Operation::region_names`] slot count. Output types are
    /// inferred through [`Operation::infer_output_types`], with the attached regions' [`RegionInterface`]s derived
    /// from this builder's arena on the spot; interfaces are never stored, and final [`Self::build`] validation
    /// derives them again from the frozen arena.
    pub fn add_instruction<P: Into<O>>(
        &mut self,
        operation: P,
        regions: Vec<RegionId>,
        inputs: Vec<AtomId>,
    ) -> Result<&[AtomId], ProgramError> {
        let operation = operation.into();
        let region_names = operation.region_names();
        if regions.len() != region_names.len() {
            return Err(ProgramError::MalformedProgram(format!(
                "operation `{}` declares {} region slots but {} regions were attached",
                operation.name(),
                region_names.len(),
                regions.len(),
            )));
        }
        for region in regions.iter().copied() {
            if region.index() >= self.regions.len() {
                return Err(ProgramError::MalformedProgram(format!(
                    "instruction references region {region} which has not been sealed yet",
                )));
            }
        }
        let input_types = inputs
            .iter()
            .map(|input| {
                self.atoms
                    .get(input.index())
                    .map(|atom| atom.r#type().into_owned())
                    .ok_or(ProgramError::UnboundAtomId { id: *input })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let region_interfaces = if regions.is_empty() {
            Vec::new()
        } else {
            let effects = Region::effects(self.regions.as_slice());
            regions
                .iter()
                .map(|region_id| {
                    let region = &self.regions[region_id.index()];
                    RegionInterface::new(region.input_types(), region.output_types(), effects[region_id.index()])
                })
                .collect()
        };
        let output_types = operation.infer_output_types(input_types.as_slice(), region_interfaces.as_slice())?;
        let outputs = output_types.into_iter().map(|r#type| self.add_variable(r#type)).collect::<Vec<_>>();
        self.instructions.push(Instruction::new(operation, inputs, outputs, regions));
        Ok(self.instructions.last().unwrap().outputs.as_slice())
    }

    /// Adds an already-formed [`Instruction`] without inferring output types or allocating output atoms. Prefer
    /// [`add_instruction`](Self::add_instruction) for ordinary staging. This function is for callers that are
    /// rebuilding an existing [`Program`] and have already allocated the instruction outputs in this builder.
    /// The caller is responsible for ensuring that the instruction input and output IDs are bound in this builder
    /// and that the output atom types match the operation's inferred outputs.
    #[inline]
    pub fn add_instruction_unchecked(&mut self, instruction: Instruction<O>) {
        self.instructions.push(instruction);
    }

    /// Splices the provided [`Program`]'s [`Instruction`]s and live constants into this [`ProgramBuilder`], remapping
    /// its inputs to the caller-provided `inputs` and returning the builder atoms holding the program's outputs, in
    /// output order. This is a plain relocation and not a re-interpretation or partial evaluation. Every instruction
    /// and live constant of the provided program is rebuilt verbatim into this builder. It is, for example,
    /// the reconciliation primitive an unknown-predicate `condition` uses to graft each branch's residual program
    /// into the reconciled branch it emits during partial evaluation.
    #[inline]
    pub fn splice_program<Input: Parameterized<V>, Output: Parameterized<V>>(
        &mut self,
        program: &Program<V, O, Input, Output>,
        inputs: &[AtomId],
    ) -> Result<Vec<AtomId>, ProgramError>
    where
        O: Clone,
    {
        // The two closures below never run concurrently, but both need `&mut` access to this builder. A `RefCell` lets
        // each take a short-lived mutable borrow without the borrow checker conservatively rejecting the second one.
        // Regions referenced by the relocated instructions are imported through one call-scoped remapping so that a
        // source region referenced from several instructions becomes one destination region (sharing is preserved).
        let builder = RefCell::new(self);
        let mut region_remapping = HashMap::new();
        program.interpret_with::<AtomId, ProgramError, _, _>(
            inputs.to_vec(),
            |_, constant| Ok(builder.borrow_mut().add_constant(constant.clone())),
            |instruction, inputs| {
                let regions = instruction
                    .regions()
                    .iter()
                    .copied()
                    .map(|region| {
                        let region = program.region_ref(region)?;
                        Ok(builder.borrow_mut().import_region_with_remapping(region, &mut region_remapping))
                    })
                    .collect::<Result<Vec<_>, ProgramError>>()?;
                let operation = instruction.operation().clone();
                Ok(builder.borrow_mut().add_instruction(operation, regions, inputs.to_vec())?.to_vec())
            },
        )
    }

    /// Imports the provided borrowed rooted [`RegionRef`] as a fresh attachable [`Region`] root, copying its complete
    /// reachable closure and preserving sharing within that closure. Each call creates an independent import. Use
    /// [`Self::import_regions`] when importing several roots from the same source arena whose shared descendants must
    /// remain shared.
    #[inline]
    pub fn import_region(&mut self, region: RegionRef<'_, V, O>) -> RegionId
    where
        O: Clone,
    {
        self.import_region_with_remapping(region, &mut HashMap::new())
    }

    /// Imports several borrowed [`RegionRef`]s from a source arena as attachable [`Region`] roots, preserving shared
    /// roots and descendants across the complete batch. All provided [`RegionRef`]s must belong to the same source
    /// arena. An empty batch imports nothing.
    #[inline]
    pub fn import_regions(&mut self, regions: &[RegionRef<'_, V, O>]) -> Result<Vec<RegionId>, ProgramError>
    where
        O: Clone,
    {
        if let Some((first, remaining)) = regions.split_first()
            && remaining.iter().any(|region| !std::ptr::eq(first.regions(), region.regions()))
        {
            return Err(ProgramError::MalformedProgram(
                "all imported regions must belong to the same program".to_string(),
            ));
        }
        let mut remapping = HashMap::new();
        Ok(regions.iter().map(|region| self.import_region_with_remapping(*region, &mut remapping)).collect())
    }

    /// Imports one borrowed rooted [`RegionRef`] using an existing source-to-destination remapping, recursively copying
    /// its reachable closure into this [`ProgramBuilder`]'s arena in post-order (i.e., children before parents).
    /// Reusing one remapping preserves shared roots and descendants across incrementally discovered imports.
    /// Callers must scope `remapping` to one source arena and this destination builder. Public callers should use
    /// [`Self::import_region`] or [`Self::import_regions`] instead.
    pub(crate) fn import_region_with_remapping(
        &mut self,
        region: RegionRef<'_, V, O>,
        remapping: &mut HashMap<RegionId, RegionId>,
    ) -> RegionId
    where
        O: Clone,
    {
        if let Some(mapped) = remapping.get(&region.id()) {
            return *mapped;
        }
        let source_id = region.id();
        let source_regions = region.regions();
        let mut imported = region.region().clone();
        for instruction in &mut imported.instructions {
            for attached in &mut instruction.regions {
                let nested = RegionRef::new(source_regions, *attached).unwrap();
                *attached = self.import_region_with_remapping(nested, remapping);
            }
        }
        let id = RegionId::new(self.regions.len());
        self.regions.push(imported);
        remapping.insert(source_id, id);
        id
    }

    /// Imports the provided owned [`Program`] as an attachable region root by splicing its complete region arena into
    /// this builder's arena directly (i.e., without cloning it), remapping every region identifier by the arena offset.
    /// Sharing within the imported program is preserved. This is the owned-move counterpart of [`Self::import_region`]
    /// for callers that constructed the program themselves and would otherwise clone it away.
    pub fn import_program<Input: Parameterized<V>, Output: Parameterized<V>>(
        &mut self,
        program: Program<V, O, Input, Output>,
    ) -> RegionId {
        let offset = self.regions.len();
        let Program { mut regions, entry, .. } = program;
        for region in &mut regions {
            for instruction in &mut region.instructions {
                for attached in &mut instruction.regions {
                    *attached = RegionId::new(attached.index() + offset);
                }
            }
        }
        self.regions.extend(regions);
        RegionId::new(entry.index() + offset)
    }

    /// Imports `callee` if it has not previously been imported into this builder and otherwise returns the existing
    /// callee root [`RegionId`]. Callees are identified by [`Rc`] identity, not structural equality, so structurally
    /// equal but independently built programs remain distinct.
    pub fn intern_callee(&mut self, callee: &Rc<Program<V, O, Vec<V>, Vec<V>>>) -> RegionId
    where
        O: Clone,
    {
        if let Some((_, id)) = self.callees.iter().find(|(interned, _)| Rc::ptr_eq(interned, callee)) {
            return *id;
        }
        let id = self.import_region(callee.entry_region_ref());
        self.callees.push((callee.clone(), id));
        id
    }

    /// Finalizes this [`ProgramBuilder`] into a [`Program`] with the provided input and output structures.
    pub fn build<Input: Parameterized<V>, Output: Parameterized<V>>(
        self,
        output_ids: Vec<AtomId>,
        input_structure: Input::ParameterStructure,
        output_structure: Output::ParameterStructure,
    ) -> Result<Program<V, O, Input, Output>, ProgramError> {
        if let Some(error) = self.error {
            return Err(error);
        }

        let expected_input_count = input_structure.parameter_count();
        check_count!("input", self.input_ids, expected_input_count, ProgramError);

        let expected_output_count = output_structure.parameter_count();
        check_count!("output", output_ids, expected_output_count, ProgramError);

        // Check for entry-region well-formedness. Program inputs must be unique variable atoms. Every variable
        // an instruction consumes must be provided first (i.e., it is a program input or the output of an earlier
        // instruction, so that instruction order is a valid evaluation order). Every instruction output must be a
        // fresh variable with exactly one provider. Finally,every program output must be bound. Constants need no
        // provider and are usable anywhere.
        let mut input_atoms = vec![false; self.atoms.len()];
        let mut variable_has_provider = vec![false; self.atoms.len()];
        for input_id in self.input_ids.iter().copied() {
            let input = self.atoms.get(input_id.index()).ok_or(ProgramError::UnboundAtomId { id: input_id })?;
            let Atom::Variable(_) = input else {
                return Err(ProgramError::MalformedProgram("program input atom was not a variable".to_string()));
            };
            if input_atoms[input_id.index()] {
                return Err(ProgramError::MalformedProgram(format!(
                    "program input atom {input_id} appears more than once",
                )));
            }
            input_atoms[input_id.index()] = true;
            variable_has_provider[input_id.index()] = true;
        }
        for instruction in self.instructions.iter() {
            for input_id in instruction.inputs.iter().copied() {
                let input = self.atoms.get(input_id.index()).ok_or(ProgramError::UnboundAtomId { id: input_id })?;
                if input.is_variable() && !variable_has_provider[input_id.index()] {
                    return Err(ProgramError::MalformedProgram("variable atom has no owning instruction".to_string()));
                }
            }
            for output_id in instruction.outputs.iter().copied() {
                let output = self.atoms.get(output_id.index()).ok_or(ProgramError::UnboundAtomId { id: output_id })?;
                let Atom::Variable(_) = output else {
                    return Err(ProgramError::MalformedProgram(
                        "instruction output atom was not a variable".to_string(),
                    ));
                };
                if input_atoms[output_id.index()] {
                    return Err(ProgramError::MalformedProgram(format!(
                        "instruction output atom {output_id} is a program input",
                    )));
                }
                if variable_has_provider[output_id.index()] {
                    return Err(ProgramError::MalformedProgram(format!(
                        "instruction output atom {output_id} is produced by more than one instruction",
                    )));
                }
                variable_has_provider[output_id.index()] = true;
            }
        }
        for output_id in output_ids.iter().copied() {
            let output = self.atoms.get(output_id.index()).ok_or(ProgramError::UnboundAtomId { id: output_id })?;
            if output.is_variable() && !variable_has_provider[output_id.index()] {
                return Err(ProgramError::MalformedProgram("variable atom has no owning instruction".to_string()));
            }
        }

        // Entry instructions may only reference previously added regions (i.e., regions with identifiers strictly
        // below the entry's own, which is assigned last). Non-entry regions uphold the same property by construction,
        // because region imports copy them in post-order (i.e., children before parents). Every referenced region
        // identifier is therefore in range, and the region graph is acyclic, which is what allows the reachability walk
        // (and any future recursive derivation over regions, such as recursive effect inference) to recurse without
        // cycle tracking.
        let entry = RegionId::new(self.regions.len());
        for instruction in &self.instructions {
            for region in instruction.regions.iter().copied() {
                if region.index() >= entry.index() {
                    return Err(ProgramError::MalformedProgram(format!(
                        "instruction references region {region} which has not been sealed yet",
                    )));
                }
            }
        }

        // Check for well-formedness of the region graph. Every region must be reachable from the entry root. Sharing
        // is legal (i.e., several instructions may reference one region), so no ownership uniqueness is enforced.
        // Acyclicity holds by construction and by the per-region topological checks above, which only admit references
        // to previously sealed regions. The same checks keep the entry region unreferenced, since its identifier is
        // assigned last.
        let mut regions = self.regions;
        regions.push(Region {
            atoms: self.atoms,
            input_ids: self.input_ids,
            output_ids,
            instructions: self.instructions,
        });
        let mut reachable = vec![false; regions.len()];
        let mut pending = vec![entry];
        while let Some(current) = pending.pop() {
            if std::mem::replace(&mut reachable[current.index()], true) {
                continue;
            }
            for instruction in &regions[current.index()].instructions {
                pending.extend(instruction.regions.iter().copied());
            }
        }
        if let Some(unreachable) = reachable.iter().position(|is_reachable| !is_reachable) {
            return Err(ProgramError::MalformedProgram(format!(
                "region {} is not reachable from the program entry region",
                RegionId::new(unreachable),
            )));
        }

        // Every instruction's attached-region count must match its operation's declared slot count. The checked
        // instruction path already enforced this at insertion time for the entry region, but instructions can also
        // arrive through the unchecked path, and so the final validation re-checks the complete frozen arena.
        for region in &regions {
            for instruction in &region.instructions {
                let declared = instruction.operation().region_names().len();
                if instruction.regions.len() != declared {
                    return Err(ProgramError::MalformedProgram(format!(
                        "operation `{}` declares {} region slots but {} regions were attached",
                        instruction.operation().name(),
                        declared,
                        instruction.regions.len(),
                    )));
                }
            }
        }

        Ok(Program { input_structure, output_structure, regions, entry, marker: PhantomData })
    }
}
