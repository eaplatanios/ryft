use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::rc::Rc;

use crate::macros::check_count;
use crate::parameters::{Parameter, Parameterized};
use crate::programs::ProgramError;
use crate::programs::atoms::{Atom, AtomId};
use crate::programs::identities::TypeIdentityRenaming;
use crate::programs::instructions::Instruction;
use crate::programs::operations::Operation;
use crate::programs::programs::Program;
use crate::programs::regions::{Region, RegionArena, RegionId, RegionInterface, RegionRef};
use crate::programs::types::{Type, Typed};
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
    pub(crate) regions: RegionArena<V, O>,

    /// Callee-interning table mapping each imported callee source to its destination root, keyed by [`Rc`] identity
    /// (i.e., [`Rc::ptr_eq`]). Two imports of the same live source program reuse one callee root, while structurally
    /// equal but independently built programs remain distinct. Storing the [`Rc`] itself both provides the identity
    /// key and keeps the source alive, so a key can never be reused by a later allocation.
    pub(crate) callees: Vec<(Rc<Program<V, O, Vec<V>, Vec<V>>>, RegionId)>,

    /// [`TypeIdentity`](crate::TypeIdentity)-instantiated shared callees cached by source and caller input [`Type`]s.
    pub(crate) callee_instantiations: Vec<CalleeInstantiation<V, O>>,

    /// Optional [`ProgramError`] encountered during program construction that will be propagated via [`Self::build`].
    pub(crate) error: Option<ProgramError>,
}

impl<V: Value, O: Operation<Type = V::Type>> ProgramBuilder<V, O> {
    /// Creates a new [`ProgramBuilder`].
    #[inline]
    pub fn new() -> Self {
        Self {
            atoms: Vec::new(),
            input_ids: Vec::new(),
            instructions: Vec::new(),
            regions: RegionArena::new(),
            callees: Vec::new(),
            callee_instantiations: Vec::new(),
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
        RegionRef::new(&self.regions, id)
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
    /// attached regions must match the operation's declared [`Operation::region_slots`] count. Output types are
    /// inferred through [`Operation::infer_output_types`], with the attached regions' [`RegionInterface`]s derived
    /// from this builder's sealed arena on the spot (i.e., [`RegionInterface`]s are never stored).
    pub fn add_instruction<P: Into<O>>(
        &mut self,
        operation: P,
        regions: Vec<RegionId>,
        inputs: Vec<AtomId>,
    ) -> Result<&[AtomId], ProgramError> {
        let operation = operation.into();
        let region_slots = operation.region_slots();
        if regions.len() != region_slots.len() {
            return Err(ProgramError::MalformedProgram(format!(
                "operation `{}` declares {} region slots but {} regions were attached",
                operation.name(),
                region_slots.len(),
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
            regions
                .iter()
                .map(|region_id| {
                    let region = &self.regions[region_id.index()];
                    RegionInterface::new(
                        region.input_types(),
                        region.output_types(),
                        self.regions.effects(*region_id).unwrap(),
                    )
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
    /// output order. This is a structural relocation and not a re-interpretation or partial evaluation. Boundary
    /// identities are instantiated from the caller-provided input types, and [`TypeIdentity`](crate::TypeIdentity)s
    /// defined inside the source graph are replaced with fresh identities for each splice. Every fresh replacement
    /// is checked against the source graph, the destination builder, and replacements generated earlier in the same
    /// splice. Every instruction and live constant is otherwise rebuilt verbatim. This is, for example, the
    /// reconciliation primitive an unknown-predicate `condition` uses to graft each branch's residual [`Program`]
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
        let input_types = inputs
            .iter()
            .map(|input| {
                self.atoms
                    .get(input.index())
                    .map(|atom| atom.r#type().into_owned())
                    .ok_or(ProgramError::UnboundAtomId { id: *input })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut program = program.with_instantiated_type_identities(input_types.as_slice())?;

        // A fresh target must not alias any type identity already live in the destination entry or its sealed regions.
        let mut unavailable_type_identities = Vec::new();
        for atom in &self.atoms {
            let r#type = atom.r#type();
            unavailable_type_identities.extend(r#type.identities().map(|(_, identity)| identity.clone()));
        }
        for region_index in 0..self.regions.len() {
            let signature = self
                .regions
                .type_identity_signature(RegionId::new(region_index))
                .expect("iterating a region arena by index must produce valid region identifiers");
            unavailable_type_identities.extend(signature.identities().iter().cloned());
        }

        // Reserve every source type identity before generating targets, while retaining the internally defined subset
        // that this splice must rename.
        let mut internal_type_identities = Vec::new();
        for region_index in 0..program.regions().len() {
            let signature = program
                .regions()
                .type_identity_signature(RegionId::new(region_index))
                .expect("iterating a region arena by index must produce valid region identifiers");
            unavailable_type_identities.extend(signature.identities().iter().cloned());
            internal_type_identities.extend(signature.internal_identities().iter().cloned());
        }

        let mut renaming = TypeIdentityRenaming::new();
        for identity in internal_type_identities {
            if renaming.replacements().iter().any(|(source, _)| source == &identity) {
                continue;
            }
            let target = renaming.insert_fresh(identity, unavailable_type_identities.as_slice())?;
            unavailable_type_identities.push(target);
        }

        if !renaming.is_identity() {
            program = Cow::Owned(program.rename_type_identities(&renaming)?);
        }

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
            && remaining.iter().any(|region| !std::ptr::eq(first.arena(), region.arena()))
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
        let mut imported = region.region().clone();
        for instruction in &mut imported.instructions {
            for attached in &mut instruction.regions {
                let nested = region.with_id(*attached).unwrap();
                *attached = self.import_region_with_remapping(nested, remapping);
            }
        }
        let id = self.regions.push(imported).unwrap();
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
        let Program { regions, entry, .. } = program;
        let offset = self.regions.append(regions);
        RegionId::new(entry.index() + offset)
    }

    /// Interns `callee`, optionally after instantiating its formal [`TypeIdentity`](crate::TypeIdentity)s for
    /// `input_types`. Without `input_types`, callees are identified by [`Rc`] identity, not structural equality, so
    /// structurally equal but independently built [`Program`]s remain distinct. With `input_types`, repeated exact
    /// instantiations share one imported root. Semantically identical types carrying separately created live identities
    /// remain distinct because those identities are retained by the imported region's boundary and no per-attachment
    /// renaming is stored on an [`Instruction`]. An instantiation requiring no renaming reuses the uninstantiated
    /// callee root.
    pub fn intern_callee(
        &mut self,
        callee: &Rc<Program<V, O, Vec<V>, Vec<V>>>,
        input_types: Option<&[V::Type]>,
    ) -> Result<RegionId, ProgramError>
    where
        O: Clone,
    {
        // An exact repeated instantiation can immediately reuse its previously imported root. Keep this lookup
        // separate from the plain-callee cache below because the caller input types are part of this cache key.
        let cached_region = input_types.and_then(|input_types| {
            self.callee_instantiations.iter().find_map(|instantiation| {
                let same_callee = Rc::ptr_eq(&instantiation.callee, callee);
                let same_input_types = instantiation.input_types == input_types;
                (same_callee && same_input_types).then_some(instantiation.region)
            })
        });
        if let Some(region) = cached_region {
            return Ok(region);
        }

        // Deriving the renaming both validates the requested boundary and determines whether importing the callee
        // would actually change any live identity. Calls without `input_types` have no instantiation to derive.
        let renaming = input_types
            .map(|input_types| V::Type::derive_identity_renaming(callee.input_types().as_slice(), input_types))
            .transpose()?;

        let non_identity_renaming = renaming.as_ref().filter(|renaming| !renaming.is_identity());
        let region = if let Some(renaming) = non_identity_renaming {
            // A non-identity renaming is embedded in the imported region's types, so it requires its own root.
            self.import_program(callee.rename_type_identities(renaming)?)
        } else {
            // No renaming, or an identity renaming, can reuse the one plain root interned for this exact `Rc` callee.
            let existing_region = self.callees.iter().find_map(|(interned, region)| {
                let same_callee = Rc::ptr_eq(interned, callee);
                same_callee.then_some(*region)
            });
            if let Some(region) = existing_region {
                region
            } else {
                let region = self.import_region(callee.entry_region_ref());
                self.callees.push((callee.clone(), region));
                region
            }
        };

        if let Some(input_types) = input_types {
            // Record even identity-renaming requests so that their next exact call avoids deriving the renaming again.
            self.callee_instantiations.push(CalleeInstantiation {
                callee: callee.clone(),
                input_types: input_types.to_vec(),
                region,
            });
        }

        Ok(region)
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

        // The entry is sealed last. `RegionArena::push` verifies that every attached region already exists, preserving
        // the arena's topological ordering and acyclicity before it publishes the entry and its derived metadata.
        let entry = RegionId::new(self.regions.len());

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
        })?;
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

        Ok(Program { input_structure, output_structure, regions, entry, marker: PhantomData })
    }
}

/// [`ProgramBuilder`]-private cache record for one imported [`TypeIdentity`](crate::TypeIdentity) instantiation of a
/// shared callee. The source [`Rc`] supplies a stable identity key and keeps its allocation alive, `input_types`
/// identifies the exact instantiated boundary, and `region` points at the corresponding imported root.
#[derive(Clone, Debug)]
pub(crate) struct CalleeInstantiation<V: Typed + Parameter, O> {
    /// Shared source callee.
    pub(crate) callee: Rc<Program<V, O, Vec<V>, Vec<V>>>,

    /// Complete caller input [`Type`]s. An imported [`Region`] carries these exact live identities in its boundary
    /// types, so only another invocation with the same types can reuse it.
    pub(crate) input_types: Vec<V::Type>,

    /// [`RegionId`] of the imported root [`Region`].
    pub(crate) region: RegionId,
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use pretty_assertions::assert_eq;

    use crate::arrays::{ArrayOperation, ArrayType, DataType, Dimension, DimensionBounds, DimensionVariable, Shape};
    use crate::backends::Array;
    use crate::operations::{AddOperation, NegOperation};
    use crate::parameters::Placeholder;
    use crate::programs::instructions::InstructionId;
    use crate::programs::regions::RegionSlot;
    use crate::programs::types::TypeError;
    use crate::programs::values::ValueId;
    use crate::tests::TestRegionOperation;

    use super::*;

    #[test]
    fn test_program_builder() {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let i0 = builder.add_input(ArrayType::scalar(DataType::F64));
        let i1 = builder.add_input(ArrayType::scalar(DataType::F64));
        let c0 = builder.add_constant(Array::scalar(2.0f64));
        let v0 = builder.add_instruction(NegOperation::new(), Vec::new(), vec![i0]).unwrap()[0];
        let v1 = builder.add_instruction(AddOperation::new(), Vec::new(), vec![v0, i1]).unwrap()[0];
        assert_eq!(builder.input_ids, vec![i0, i1]);
        assert!(matches!(
            builder.atoms.get(i0.index()),
            Some(Atom::Variable(r#type)) if *r#type == ArrayType::scalar(DataType::F64)
        ));
        assert!(matches!(
            builder.atoms.get(i1.index()),
            Some(Atom::Variable(r#type)) if *r#type == ArrayType::scalar(DataType::F64)
        ));
        assert!(matches!(
            builder.atoms.get(c0.index()),
            Some(Atom::Constant(value)) if *value == Array::scalar(2.0)
        ));
        assert!(matches!(
            builder.atoms.get(v0.index()),
            Some(Atom::Variable(r#type)) if *r#type == ArrayType::scalar(DataType::F64)
        ));
        assert!(matches!(
            builder.atoms.get(v1.index()),
            Some(Atom::Variable(r#type)) if *r#type == ArrayType::scalar(DataType::F64)
        ));
        assert_eq!(builder.instructions.len(), 2);
        assert_eq!(builder.instructions[0].inputs, vec![i0]);
        assert_eq!(builder.instructions[0].outputs, vec![v0]);
        assert_eq!(builder.instructions[1].inputs, vec![v0, i1]);
        assert_eq!(builder.instructions[1].outputs, vec![v1]);

        let program =
            builder.build::<(Array, Array), Array>(vec![v1], (Placeholder, Placeholder), Placeholder).unwrap();
        assert_eq!(program.input_ids(), vec![i0, i1]);
        assert_eq!(program.output_ids(), vec![v1]);
        assert_eq!(program.instructions().len(), 2);
        assert_eq!(program.interpret((Array::scalar(2.0f64), Array::scalar(38.0f64))), Ok(Array::scalar(36.0f64)));

        // `splice_program` appends the program's reachable instructions into a fresh builder, remapping its inputs to
        // the provided builder atoms and returning the builder atoms for its outputs. The program's `2.0` constant is
        // dead (i.e., no instruction consumes it), and so only the two reachable instructions are rebuilt.
        let mut outer = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let a0 = outer.add_input(ArrayType::scalar(DataType::F64));
        let a1 = outer.add_input(ArrayType::scalar(DataType::F64));
        let outputs = outer.splice_program(&program, &[a0, a1]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outer.instructions.len(), 2);
        let outer_program =
            outer.build::<(Array, Array), Array>(outputs, (Placeholder, Placeholder), Placeholder).unwrap();
        assert_eq!(
            outer_program.interpret((Array::scalar(2.0f64), Array::scalar(38.0f64))),
            Ok(Array::scalar(36.0f64))
        );
    }

    #[test]
    fn test_program_builder_rejects_unbound_instruction_inputs() {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let v0 = builder.add_instruction(AddOperation::new(), Vec::new(), vec![AtomId::new(42), AtomId::new(99)]);
        assert!(matches!(v0, Err(ProgramError::UnboundAtomId { id }) if id == AtomId::new(42)));
    }

    #[test]
    fn test_program_builder_build_returns_error() {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        builder.error = Some(ProgramError::InvalidInputCount { expected: 1, actual: 0 });
        assert!(matches!(
            builder.build::<Array, Array>(Vec::new(), Placeholder, Placeholder),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        ));
    }

    #[test]
    fn test_program_builder_build_rejects_invalid_input_count() {
        let builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        assert!(matches!(
            builder.build::<Array, ()>(Vec::new(), Placeholder, ()),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        ));
    }

    #[test]
    fn test_program_builder_build_rejects_invalid_output_count() {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        builder.add_input(ArrayType::scalar(DataType::F64));
        assert!(matches!(
            builder.build::<Array, Array>(Vec::new(), Placeholder, Placeholder),
            Err(ProgramError::InvalidOutputCount { expected: 1, actual: 0 }),
        ));
    }

    #[test]
    fn test_program_builder_build_rejects_malformed_atom_providers() {
        let mut duplicate_input_builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = duplicate_input_builder.add_input(ArrayType::scalar(DataType::F64));
        duplicate_input_builder.input_ids.push(input);
        assert!(matches!(
            duplicate_input_builder.build::<Vec<Array>, Vec<Array>>(
                vec![input],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            ),
            Err(ProgramError::MalformedProgram(message))
                if message == format!("program input atom {input} appears more than once")
        ));

        let mut input_output_overlap_builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = input_output_overlap_builder.add_input(ArrayType::scalar(DataType::F64));
        input_output_overlap_builder.add_instruction_unchecked(Instruction::new(
            ArrayOperation::Neg(NegOperation::new()),
            vec![input],
            vec![input],
            Vec::new(),
        ));
        assert!(matches!(
            input_output_overlap_builder.build::<Vec<Array>, Vec<Array>>(
                vec![input],
                vec![Placeholder],
                vec![Placeholder],
            ),
            Err(ProgramError::MalformedProgram(message))
                if message == format!("instruction output atom {input} is a program input")
        ));

        let mut duplicate_output_builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = duplicate_output_builder.add_input(ArrayType::scalar(DataType::F64));
        let output = duplicate_output_builder.add_variable(ArrayType::scalar(DataType::F64));
        duplicate_output_builder.add_instruction_unchecked(Instruction::new(
            ArrayOperation::Neg(NegOperation::new()),
            vec![input],
            vec![output],
            Vec::new(),
        ));
        duplicate_output_builder.add_instruction_unchecked(Instruction::new(
            ArrayOperation::Neg(NegOperation::new()),
            vec![input],
            vec![output],
            Vec::new(),
        ));
        assert!(matches!(
            duplicate_output_builder.build::<Vec<Array>, Vec<Array>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            ),
            Err(ProgramError::MalformedProgram(message))
                if message == format!("instruction output atom {output} is produced by more than one instruction")
        ));
    }

    #[test]
    fn test_program_builder_import_region_and_intern_callee() {
        // A source program with one sealed region attached to its entry instruction.
        let mut source_builder = ProgramBuilder::<Array, TestRegionOperation>::new();
        let mut region_builder = ProgramBuilder::<Array, TestRegionOperation>::new();
        let region_input = region_builder.add_input(ArrayType::scalar(DataType::F64));
        let region_program = region_builder
            .build::<Vec<Array>, Vec<Array>>(vec![region_input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let sealed = source_builder.import_region(region_program.entry_region_ref());
        let input = source_builder.add_input(ArrayType::scalar(DataType::F64));
        let output = source_builder
            .add_instruction(
                TestRegionOperation::WithRegions(const { &[RegionSlot::computation("body")] }),
                vec![sealed],
                vec![input],
            )
            .unwrap()[0];
        let source = source_builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        // Fresh borrowed imports copy the complete closure independently: two imports produce two subtrees.
        let mut destination = ProgramBuilder::<Array, TestRegionOperation>::new();
        let first = destination.import_region(source.entry_region_ref());
        let second = destination.import_region(source.entry_region_ref());
        assert_ne!(first, second);
        let imported = destination.regions[first.index()].clone();
        assert_eq!(imported.instructions()[0].regions().len(), 1);
        assert_ne!(imported.instructions()[0].regions()[0], first);

        // Callee imports intern by live `Rc` identity: one shared root per live source, while structurally equal
        // but independently built programs remain distinct.
        let flat = Rc::new(source.to_flat_program());
        let equal_but_distinct = Rc::new(flat.as_ref().clone());
        let mut destination = ProgramBuilder::<Array, TestRegionOperation>::new();
        let first = destination.intern_callee(&flat, None).unwrap();
        let second = destination.intern_callee(&flat, None).unwrap();
        let third = destination.intern_callee(&equal_but_distinct, None).unwrap();
        assert_eq!(first, second);
        assert_ne!(first, third);
    }

    #[test]
    fn test_program_builder_type_identity_instantiation_cache_preserves_live_identities() {
        #[derive(Clone)]
        struct ArrayIdentityOperation;

        impl Operation for ArrayIdentityOperation {
            type Type = ArrayType;

            fn name(&self) -> &'static str {
                "array_identity"
            }

            fn region_slots(&self) -> &'static [RegionSlot] {
                const { &[RegionSlot::computation("body")] }
            }

            fn infer_output_types(
                &self,
                input_types: &[ArrayType],
                region_interfaces: &[RegionInterface<ArrayType>],
            ) -> Result<Vec<ArrayType>, TypeError> {
                let [region_interface] = region_interfaces else {
                    return Err(TypeError::invalid(format!(
                        "array identity expects 1 attached region but got {}",
                        region_interfaces.len(),
                    )));
                };
                if region_interface.input_types() != input_types {
                    return Err(TypeError::invalid("array identity region input types do not match its operand types"));
                }
                Ok(region_interface.output_types().to_vec())
            }
        }

        let bounds = DimensionBounds::non_negative(Some(16)).unwrap();
        let formal_first = DimensionVariable::new("formal_first", bounds);
        let formal_second = DimensionVariable::new("formal_second", bounds);
        let array_type =
            |variable: DimensionVariable| ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(variable)]));
        let mut callee_builder = ProgramBuilder::<ArrayType, ArrayIdentityOperation>::new();
        let first_input = callee_builder.add_input(array_type(formal_first.clone()));
        let second_input = callee_builder.add_input(array_type(formal_second.clone()));
        let callee = Rc::new(
            callee_builder
                .build::<Vec<ArrayType>, Vec<ArrayType>>(
                    vec![first_input, second_input],
                    vec![Placeholder, Placeholder],
                    vec![Placeholder, Placeholder],
                )
                .unwrap(),
        );

        // Repeated exact instantiations share, but otherwise-identical callers with separately created identities
        // remain distinct because each imported region retains those live identities in its boundary.
        let mut destination = ProgramBuilder::<ArrayType, ArrayIdentityOperation>::new();
        let caller_a = DimensionVariable::new("caller_a", bounds);
        let caller_b = DimensionVariable::new("caller_b", bounds);
        let caller_c = DimensionVariable::new("caller_c", bounds);
        let caller_d = DimensionVariable::new("caller_d", bounds);
        let first = destination.intern_callee(&callee, Some(&[array_type(caller_a), array_type(caller_b)])).unwrap();
        let second_input_types = [array_type(caller_c), array_type(caller_d)];
        let second = destination.intern_callee(&callee, Some(&second_input_types)).unwrap();
        let repeated = destination.intern_callee(&callee, Some(&second_input_types)).unwrap();
        assert_ne!(first, second);
        assert_eq!(second, repeated);
        assert_eq!(destination.region_ref(second).unwrap().input_types(), second_input_types);
        let inputs = second_input_types
            .iter()
            .cloned()
            .map(|input_type| destination.add_input(input_type))
            .collect::<Vec<_>>();
        let outputs = destination.add_instruction(ArrayIdentityOperation, vec![second], inputs).unwrap().to_vec();
        assert_eq!(
            outputs
                .iter()
                .map(|output| destination.atoms()[output.index()].r#type().into_owned())
                .collect::<Vec<_>>(),
            second_input_types,
        );

        // A type-identity instantiation reuses the plain callee, while a permutation of overlapping identities
        // does not.
        let mut destination = ProgramBuilder::<ArrayType, ArrayIdentityOperation>::new();
        let plain = destination.intern_callee(&callee, None).unwrap();
        let direct = destination
            .intern_callee(&callee, Some(&[array_type(formal_first.clone()), array_type(formal_second.clone())]))
            .unwrap();
        let permuted = destination
            .intern_callee(&callee, Some(&[array_type(formal_second), array_type(formal_first)]))
            .unwrap();
        assert_eq!(plain, direct);
        assert_ne!(direct, permuted);
    }

    #[test]
    fn test_program_builder_build_multi_region_program() {
        let mut builder = ProgramBuilder::<Array, TestRegionOperation>::new();
        let mut region_builder = ProgramBuilder::<Array, TestRegionOperation>::new();
        let region_input = region_builder.add_input(ArrayType::scalar(DataType::F64));
        let doubled = region_builder
            .add_instruction(TestRegionOperation::Add, Vec::new(), vec![region_input, region_input])
            .unwrap()[0];
        let region_program = region_builder
            .build::<Vec<Array>, Vec<Array>>(vec![doubled], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let sealed = builder.import_region(region_program.entry_region_ref());
        assert_eq!(sealed, RegionId::new(0));

        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder
            .add_instruction(
                TestRegionOperation::WithRegions(const { &[RegionSlot::computation("body")] }),
                vec![sealed],
                vec![input],
            )
            .unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();

        // The regions arena holds the sealed region plus the entry region, and producers resolve per region.
        assert_eq!(program.regions().len(), 2);
        assert_eq!(program.entry(), RegionId::new(1));
        assert_eq!(program.region(sealed).unwrap().input_ids(), &[region_input]);
        assert!(matches!(
            program.region(RegionId::new(7)),
            Err(ProgramError::MalformedProgram(message)) if message == "region ^7 is out of range",
        ));
        let instruction = &program.instructions()[0];
        assert_eq!(instruction.regions(), &[sealed]);
        assert_eq!(
            program.producer(ValueId::new(program.entry(), output)).unwrap(),
            Some(InstructionId::new(program.entry(), 0)),
        );
        assert_eq!(program.producer(ValueId::new(program.entry(), input)).unwrap(), None);
        assert_eq!(program.producer(ValueId::new(sealed, doubled)).unwrap(), Some(InstructionId::new(sealed, 0)),);

        // Instruction locators resolve against the complete region arena.
        let instruction = program.instruction(InstructionId::new(program.entry(), 0)).unwrap();
        assert_eq!(instruction.regions(), &[sealed]);
        assert!(program.instruction(InstructionId::new(program.entry(), 9)).is_err());

        // The multi-region program clones, maps, and reports effects across every region.
        let cloned = program.clone();
        assert_eq!(cloned.regions().len(), 2);
        let mapped = program.map_operations(|operation| Ok(operation.clone())).unwrap();
        assert_eq!(mapped.regions().len(), 2);
        assert_eq!(mapped.instructions()[0].regions(), &[sealed]);
        assert!(program.effects().is_pure());

        // The region-aware rebuild paths preserve regions. Simplification keeps the live region-carrying instruction
        // and its region, filtering projects the entry boundary while passing regions through, and relocation imports
        // the referenced regions into the destination builder.
        let simplified = program.simplified().unwrap();
        assert_eq!(simplified.regions().len(), 2);
        assert_eq!(simplified.instructions()[0].regions(), &[sealed]);
        let (filtered, live_inputs) = program.filtered(&[input], program.output_ids(), &[]).unwrap();
        assert_eq!(filtered.regions().len(), 2);
        assert_eq!(live_inputs, vec![0]);
        let mut relocation_builder = ProgramBuilder::<Array, TestRegionOperation>::new();
        let relocation_input = relocation_builder.add_input(ArrayType::scalar(DataType::F64));
        let relocated_outputs =
            relocation_builder.splice_program(&program.to_flat_program(), &[relocation_input]).unwrap();
        let relocated = relocation_builder
            .build::<Vec<Array>, Vec<Array>>(relocated_outputs, vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(relocated.regions().len(), 2);
        let simplified = program.into_simplified().unwrap();
        assert_eq!(simplified.regions().len(), 2);
    }

    #[test]
    fn test_program_builder_region_ref_and_import_region() {
        let mut source_builder = ProgramBuilder::<Array, TestRegionOperation>::new();
        let mut region_builder = ProgramBuilder::<Array, TestRegionOperation>::new();
        let region_input = region_builder.add_input(ArrayType::scalar(DataType::F64));
        let region_program = region_builder
            .build::<Vec<Array>, Vec<Array>>(vec![region_input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let sealed = source_builder.import_region(region_program.entry_region_ref());
        let sealed_ref = source_builder.region_ref(sealed).unwrap();
        assert_eq!(sealed_ref.id(), sealed);
        assert_eq!(sealed_ref.input_types(), vec![ArrayType::scalar(DataType::F64)]);
        assert!(matches!(
            source_builder.region_ref(RegionId::new(7)),
            Err(ProgramError::MalformedProgram(message)) if message == "region ^7 is not part of this builder",
        ));
        let input = source_builder.add_input(ArrayType::scalar(DataType::F64));
        let first = source_builder
            .add_instruction(
                TestRegionOperation::WithRegions(const { &[RegionSlot::computation("body")] }),
                vec![sealed],
                vec![input],
            )
            .unwrap()[0];
        let second = source_builder
            .add_instruction(
                TestRegionOperation::WithRegions(const { &[RegionSlot::computation("body")] }),
                vec![sealed],
                vec![first],
            )
            .unwrap()[0];
        let source = source_builder
            .build::<Vec<Array>, Vec<Array>>(vec![second], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut destination = ProgramBuilder::<Array, TestRegionOperation>::new();
        let imported = destination.import_region(source.entry_region_ref());
        let imported_region = destination.region_ref(imported).unwrap().region();
        assert_eq!(imported_region.instructions()[0].regions(), imported_region.instructions()[1].regions());
        assert_ne!(imported_region.instructions()[0].regions()[0], imported);
    }

    #[test]
    fn test_program_builder_import_regions_preserves_sharing() {
        let mut leaf_builder = ProgramBuilder::<Array, TestRegionOperation>::new();
        let leaf_input = leaf_builder.add_input(ArrayType::scalar(DataType::F64));
        let leaf = leaf_builder
            .build::<Vec<Array>, Vec<Array>>(vec![leaf_input], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut root_builder = ProgramBuilder::<Array, TestRegionOperation>::new();
        let nested = root_builder.import_region(leaf.entry_region_ref());
        let root_input = root_builder.add_input(ArrayType::scalar(DataType::F64));
        let root_output = root_builder
            .add_instruction(
                TestRegionOperation::WithRegions(const { &[RegionSlot::computation("body")] }),
                vec![nested],
                vec![root_input],
            )
            .unwrap()[0];
        let root = root_builder
            .build::<Vec<Array>, Vec<Array>>(vec![root_output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        // Construct two distinct roots in one source arena that both reference the same previously sealed leaf.
        let mut source_builder = ProgramBuilder::<Array, TestRegionOperation>::new();
        let shared_leaf = source_builder.import_region(leaf.entry_region_ref());
        let first_root = RegionId::new(source_builder.regions.len());
        source_builder.regions.push(root.entry_region().clone()).unwrap();
        let second_root = RegionId::new(source_builder.regions.len());
        source_builder.regions.push(root.entry_region().clone()).unwrap();
        let source_input = source_builder.add_input(ArrayType::scalar(DataType::F64));
        let source_output = source_builder
            .add_instruction(
                TestRegionOperation::WithRegions(
                    const { &[RegionSlot::computation("first"), RegionSlot::computation("second")] },
                ),
                vec![first_root, second_root],
                vec![source_input],
            )
            .unwrap()[0];
        let source = source_builder
            .build::<Vec<Array>, Vec<Array>>(vec![source_output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(source.region(first_root).unwrap().instructions()[0].regions(), &[shared_leaf]);
        assert_eq!(source.region(second_root).unwrap().instructions()[0].regions(), &[shared_leaf]);

        let roots = [source.region_ref(first_root).unwrap(), source.region_ref(second_root).unwrap()];
        let mut destination = ProgramBuilder::<Array, TestRegionOperation>::new();
        let imported = destination.import_regions(&roots).unwrap();
        assert_ne!(imported[0], imported[1]);
        assert_eq!(destination.regions.len(), 3);
        assert_eq!(
            destination.regions[imported[0].index()].instructions()[0].regions(),
            destination.regions[imported[1].index()].instructions()[0].regions(),
        );

        let mut duplicate_destination = ProgramBuilder::<Array, TestRegionOperation>::new();
        let duplicate_roots = [source.region_ref(first_root).unwrap(), source.region_ref(first_root).unwrap()];
        let imported = duplicate_destination.import_regions(&duplicate_roots).unwrap();
        assert_eq!(imported[0], imported[1]);
        assert_eq!(duplicate_destination.regions.len(), 2);

        let mut other_builder = ProgramBuilder::<Array, TestRegionOperation>::new();
        let other_input = other_builder.add_input(ArrayType::scalar(DataType::F64));
        let other = other_builder
            .build::<Vec<Array>, Vec<Array>>(vec![other_input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut mixed_destination = ProgramBuilder::<Array, TestRegionOperation>::new();
        assert!(matches!(
            mixed_destination.import_regions(&[source.entry_region_ref(), other.entry_region_ref()]),
            Err(ProgramError::MalformedProgram(message))
                if message == "all imported regions must belong to the same program",
        ));
        assert!(mixed_destination.regions.is_empty());
    }

    #[test]
    fn test_program_builder_build_shares_region_across_instructions() {
        // Sharing is legal: several instructions (and several slots of one instruction) may reference one region.
        let mut builder = ProgramBuilder::<Array, TestRegionOperation>::new();
        let mut region_builder = ProgramBuilder::<Array, TestRegionOperation>::new();
        let region_input = region_builder.add_input(ArrayType::scalar(DataType::F64));
        let region_program = region_builder
            .build::<Vec<Array>, Vec<Array>>(vec![region_input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let sealed = builder.import_region(region_program.entry_region_ref());
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let first = builder
            .add_instruction(
                TestRegionOperation::WithRegions(
                    const { &[RegionSlot::computation("first"), RegionSlot::computation("second")] },
                ),
                vec![sealed, sealed],
                vec![input],
            )
            .unwrap()[0];
        let second = builder
            .add_instruction(
                TestRegionOperation::WithRegions(const { &[RegionSlot::computation("body")] }),
                vec![sealed],
                vec![first],
            )
            .unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![second], vec![Placeholder], vec![Placeholder]).unwrap();
        assert_eq!(program.regions().len(), 2);
        assert_eq!(program.instructions()[0].regions(), &[sealed, sealed]);
        assert_eq!(program.instructions()[1].regions(), &[sealed]);
    }

    #[test]
    fn test_program_builder_add_instruction_derives_region_interfaces() {
        // The region-carrying operation's output types are its first region interface's output types, so an entry
        // input type that differs from the region output type pins that the builder derived and delivered the
        // interface (rather than the inference falling back to the operation inputs).
        let mut builder = ProgramBuilder::<Array, TestRegionOperation>::new();
        let mut region_builder = ProgramBuilder::<Array, TestRegionOperation>::new();
        let region_input = region_builder.add_input(ArrayType::scalar(DataType::I64));
        let region_program = region_builder
            .build::<Vec<Array>, Vec<Array>>(vec![region_input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let sealed = builder.import_region(region_program.entry_region_ref());
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder
            .add_instruction(
                TestRegionOperation::WithRegions(const { &[RegionSlot::computation("body")] }),
                vec![sealed],
                vec![input],
            )
            .unwrap()[0];
        assert_eq!(builder.atoms()[output.index()].r#type().into_owned(), ArrayType::scalar(DataType::I64));
    }

    #[test]
    fn test_program_builder_splice_program_preserves_region_sharing() {
        let mut leaf_builder = ProgramBuilder::<Array, TestRegionOperation>::new();
        let leaf_input = leaf_builder.add_input(ArrayType::scalar(DataType::F64));
        let leaf = leaf_builder
            .build::<Vec<Array>, Vec<Array>>(vec![leaf_input], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut root_builder = ProgramBuilder::<Array, TestRegionOperation>::new();
        let nested = root_builder.import_region(leaf.entry_region_ref());
        let root_input = root_builder.add_input(ArrayType::scalar(DataType::F64));
        let root_output = root_builder
            .add_instruction(
                TestRegionOperation::WithRegions(const { &[RegionSlot::computation("body")] }),
                vec![nested],
                vec![root_input],
            )
            .unwrap()[0];
        let root = root_builder
            .build::<Vec<Array>, Vec<Array>>(vec![root_output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        // Two distinct attached roots share one nested leaf, and both entry instructions reuse those same roots.
        // Splicing must preserve both levels of sharing through one source-to-destination remapping.
        let mut source_builder = ProgramBuilder::<Array, TestRegionOperation>::new();
        let shared_leaf = source_builder.import_region(leaf.entry_region_ref());
        let first_root = RegionId::new(source_builder.regions.len());
        source_builder.regions.push(root.entry_region().clone()).unwrap();
        let second_root = RegionId::new(source_builder.regions.len());
        source_builder.regions.push(root.entry_region().clone()).unwrap();
        let source_input = source_builder.add_input(ArrayType::scalar(DataType::F64));
        let first_output = source_builder
            .add_instruction(
                TestRegionOperation::WithRegions(
                    const { &[RegionSlot::computation("first"), RegionSlot::computation("second")] },
                ),
                vec![first_root, second_root],
                vec![source_input],
            )
            .unwrap()[0];
        let source_output = source_builder
            .add_instruction(
                TestRegionOperation::WithRegions(
                    const { &[RegionSlot::computation("first"), RegionSlot::computation("second")] },
                ),
                vec![first_root, second_root],
                vec![first_output],
            )
            .unwrap()[0];
        let source = source_builder
            .build::<Vec<Array>, Vec<Array>>(vec![source_output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(source.region(first_root).unwrap().instructions()[0].regions(), &[shared_leaf]);
        assert_eq!(source.region(second_root).unwrap().instructions()[0].regions(), &[shared_leaf]);

        let mut destination = ProgramBuilder::<Array, TestRegionOperation>::new();
        let destination_input = destination.add_input(ArrayType::scalar(DataType::F64));
        let outputs = destination.splice_program(&source.to_flat_program(), &[destination_input]).unwrap();
        let relocated =
            destination.build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder], vec![Placeholder]).unwrap();
        assert_eq!(relocated.regions().len(), 4);
        let relocated_instructions = relocated.instructions();
        assert_eq!(relocated_instructions[0].regions(), relocated_instructions[1].regions());
        assert_ne!(relocated_instructions[0].regions()[0], relocated_instructions[0].regions()[1]);
        let first_nested_regions =
            relocated.region(relocated_instructions[0].regions()[0]).unwrap().instructions()[0].regions();
        let second_nested_regions =
            relocated.region(relocated_instructions[0].regions()[1]).unwrap().instructions()[0].regions();
        assert_eq!(first_nested_regions, second_nested_regions);
    }

    #[test]
    fn test_program_builder_build_rejects_malformed_regions() {
        // Instruction regions must reference previously sealed regions (which keeps the graph acyclic by
        // construction). The checked instruction path rejects at insertion time and the unchecked path at build time.
        let mut builder = ProgramBuilder::<Array, TestRegionOperation>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        assert!(matches!(
            builder.add_instruction(
                TestRegionOperation::WithRegions(const { &[RegionSlot::computation("body")] }),
                vec![RegionId::new(3)],
                vec![input],
            ),
            Err(ProgramError::MalformedProgram(message))
                if message == "instruction references region ^3 which has not been sealed yet",
        ));
        let output = builder.add_variable(ArrayType::scalar(DataType::F64));
        builder.add_instruction_unchecked(Instruction::new(
            TestRegionOperation::WithRegions(const { &[RegionSlot::computation("body")] }),
            vec![input],
            vec![output],
            vec![RegionId::new(3)],
        ));
        assert!(matches!(
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]),
            Err(ProgramError::MalformedProgram(message))
                if message == "instruction references region ^3 which has not been sealed yet",
        ));

        // The attached-region count must match the operation's declared slot count. The checked instruction path
        // rejects at insertion time and the unchecked path at build time.
        let mut builder = ProgramBuilder::<Array, TestRegionOperation>::new();
        let mut region_builder = ProgramBuilder::<Array, TestRegionOperation>::new();
        let region_input = region_builder.add_input(ArrayType::scalar(DataType::F64));
        let region_program = region_builder
            .build::<Vec<Array>, Vec<Array>>(vec![region_input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let sealed = builder.import_region(region_program.entry_region_ref());
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        assert!(matches!(
            builder.add_instruction(TestRegionOperation::Add, vec![sealed], vec![input, input]),
            Err(ProgramError::MalformedProgram(message))
                if message == "operation `add` declares 0 region slots but 1 regions were attached",
        ));
        let output = builder.add_variable(ArrayType::scalar(DataType::F64));
        builder.add_instruction_unchecked(Instruction::new(
            TestRegionOperation::Add,
            vec![input, input],
            vec![output],
            vec![sealed],
        ));
        assert!(matches!(
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]),
            Err(ProgramError::MalformedProgram(message))
                if message == "operation `add` declares 0 region slots but 1 regions were attached",
        ));

        // Every sealed region must be reachable from the entry root.
        let mut builder = ProgramBuilder::<Array, TestRegionOperation>::new();
        let mut region_builder = ProgramBuilder::<Array, TestRegionOperation>::new();
        let region_input = region_builder.add_input(ArrayType::scalar(DataType::F64));
        let region_program = region_builder
            .build::<Vec<Array>, Vec<Array>>(vec![region_input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        builder.import_region(region_program.entry_region_ref());
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        assert!(matches!(
            builder.build::<Vec<Array>, Vec<Array>>(vec![input], vec![Placeholder], vec![Placeholder]),
            Err(ProgramError::MalformedProgram(message))
                if message == "region ^0 is not reachable from the program entry region",
        ));
    }
}
