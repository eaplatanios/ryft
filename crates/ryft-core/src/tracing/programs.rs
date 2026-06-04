use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;

use half::{bf16, f16};

use ryft_macros::Parameter;

use crate::macros::check_count;
use crate::operations::{InterpretableOperation, Operation};
use crate::parameters::{Parameter, ParameterError, Parameterized, ParameterizedFamily, Placeholder};
use crate::tracing::TracingError;
use crate::types::{DataType, Type, Typed};

/// Identifies value carriers in [`Program`]s. [`Value`] is a subtrait of [`Traceable`] implemented by leaf carriers
/// that behave as values for interpretation or transform dispatch (i.e., concrete data such as arrays, and opaque
/// references to concrete data held in a side environment). The sole purpose of this marker is to give Rust's coherence
/// checker a way to tell two blanket implementations apart. Each composable transform (e.g., for just-in-time
/// compilation or automatic differentiation) provides:
///
///   1. an implementation for `V: Value<T>` that evaluates the transform on concrete data, and
///   2. an implementation for context-backed [`Tracer`](crate::tracing::Tracer) values that stages the transform into
///      the enclosing traced [`Program`].
///
/// Because [`Tracer`](crate::tracing::Tracer) implements [`Traceable`] but not [`Value`], these two implementations
/// never overlap.
pub trait Value<T: Type>: Traceable<T> {}

impl Value<DataType> for bool {}
impl Value<DataType> for i8 {}
impl Value<DataType> for i16 {}
impl Value<DataType> for i32 {}
impl Value<DataType> for i64 {}
impl Value<DataType> for u8 {}
impl Value<DataType> for u16 {}
impl Value<DataType> for u32 {}
impl Value<DataType> for u64 {}
impl Value<DataType> for bf16 {}
impl Value<DataType> for f16 {}
impl Value<DataType> for f32 {}
impl Value<DataType> for f64 {}

/// Represents leaf values that can participate in traced [`Program`]s. [`Traceable`] is implemented by every type that
/// can appear as a leaf in a staged [`Program`]: both concrete data types such as `f32`, `f64`, and backend arrays, and
/// tracing wrappers such as [`Tracer`](crate::Tracer). It ties each leaf to a type descriptor `T` via [`Typed`] and
/// requires [`Debug`] and [`Display`] so that diagnostics, constants, and [`Operation`] metadata can render their
/// carried values directly.
pub trait Traceable<T: Type>: Clone + Debug + Display + Parameter + Typed<T> {}

impl Traceable<DataType> for bool {}
impl Traceable<DataType> for i8 {}
impl Traceable<DataType> for i16 {}
impl Traceable<DataType> for i32 {}
impl Traceable<DataType> for i64 {}
impl Traceable<DataType> for u8 {}
impl Traceable<DataType> for u16 {}
impl Traceable<DataType> for u32 {}
impl Traceable<DataType> for u64 {}
impl Traceable<DataType> for bf16 {}
impl Traceable<DataType> for f16 {}
impl Traceable<DataType> for f32 {}
impl Traceable<DataType> for f64 {}

/// [`Atom`]s represent nodes in [`Program`]s that represent either concrete values or variables of specific [`Type`]s.
#[derive(Clone, Debug, Parameter)]
pub enum Atom<T: Type, V: Typed<T>> {
    /// Literal constant value that appears in a [`Program`].
    Constant(V),

    /// Non-constant variable of a specific [`Type`] that appears in a [`Program`].
    Variable(T),
}

impl<T: Type, V: Typed<T>> Atom<T, V> {
    /// Returns `true` if this [`Atom`] is an [`Atom::Constant`].
    #[inline]
    pub fn is_constant(&self) -> bool {
        matches!(self, Self::Constant(_))
    }

    /// Returns `true` if this [`Atom`] is an [`Atom::Variable`].
    #[inline]
    pub fn is_variable(&self) -> bool {
        matches!(self, Self::Variable(_))
    }

    /// Returns the underlying constant value if this atom is an [`Atom::Constant`] and [`None`] otherwise.
    #[inline]
    pub fn as_constant(&self) -> Option<&V> {
        match self {
            Self::Constant(value) => Some(value),
            Self::Variable(_) => None,
        }
    }
}

impl<T: Type, V: Typed<T>> Typed<T> for Atom<T, V> {
    fn r#type(&self) -> Cow<'_, T> {
        match self {
            Self::Constant(value) => value.r#type(),
            Self::Variable(r#type) => Cow::Borrowed(r#type),
        }
    }
}

/// Unique identifier for an [`Atom`] within a [`Program`]. [`AtomId`]s are stable indexes into a [`Program`]'s atom
/// table. [`Instruction`]s refer to their inputs and outputs by these IDs, which keeps the intermediate representation
/// compact and easy to clone.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Parameter)]
pub struct AtomId {
    /// Zero-based index of the corresponding [`Atom`] inside the owning [`Program`]'s atom table.
    index: usize,
}

impl AtomId {
    /// Creates a new [`AtomId`] from the provided zero-based atom-table index.
    #[inline]
    pub fn new(index: usize) -> Self {
        Self { index }
    }

    /// Returns the zero-based index of the corresponding [`Atom`] inside the owning [`Program`]'s atom table.
    #[inline]
    pub fn index(self) -> usize {
        self.index
    }
}

impl Display for AtomId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "%{}", self.index)
    }
}

/// [`Instruction`]s represent applications of [`Operation`]s to input values in [`Program`]s. [`Program`]s execute
/// [`Instruction`]s in sequential order, and higher-order [`Operation`]s can carry nested programs for control flow
/// or other structured evaluation boundaries.
#[derive(Clone, Debug)]
pub struct Instruction<O> {
    /// [`Operation`] applied by this [`Instruction`].
    operation: O,

    /// [`AtomId`]s of the input [`Atom`]s consumed by this [`Instruction`].
    inputs: Vec<AtomId>,

    /// [`AtomId`]s of the output [`Atom`]s produced by this [`Instruction`].
    outputs: Vec<AtomId>,
}

impl<O> Instruction<O> {
    /// Creates a new [`Instruction`].
    #[inline]
    pub fn new(operation: O, inputs: Vec<AtomId>, outputs: Vec<AtomId>) -> Self {
        Self { operation, inputs, outputs }
    }

    /// Returns the [`Operation`] applied by this [`Instruction`].
    #[inline]
    pub fn operation(&self) -> &O {
        &self.operation
    }

    /// Returns the [`AtomId`]s of the input [`Atom`]s consumed by this [`Instruction`].
    #[inline]
    pub fn inputs(&self) -> &[AtomId] {
        self.inputs.as_slice()
    }

    /// Returns the [`AtomId`]s of the output [`Atom`]s produced by this [`Instruction`].
    #[inline]
    pub fn outputs(&self) -> &[AtomId] {
        self.outputs.as_slice()
    }

    /// Consumes this [`Instruction`] and returns its [`Operation`], input [`AtomId`]s, and output [`AtomId`]s.
    #[inline]
    pub fn into_parts(self) -> (O, Vec<AtomId>, Vec<AtomId>) {
        (self.operation, self.inputs, self.outputs)
    }
}

/// [`Program`] that is produced by tracing and which can be interpreted or compiled and executed by a backend. It
/// consists of a sequence of [`Instruction`]s paired with [`Parameterized`] input and output types. This is the primary
/// intermediate representation (IR) used by the Ryft tracing and transformation system (e.g., to support things like
/// automatic differentiation and just-in-time compilation).
#[derive(Debug)]
pub struct Program<T: Type, V: Typed<T> + Parameter, O, Input: Parameterized<V>, Output: Parameterized<V>> {
    /// [`Atom`]s contained in this [`Program`], in the order in which they will be evaluated.
    pub(crate) atoms: Vec<Atom<T, V>>,

    /// [`AtomId`]s of the [`Atom`]s that correspond to the inputs (i.e., arguments) of this [`Program`].
    pub(crate) input_ids: Vec<AtomId>,

    /// [`AtomId`]s of the [`Atom`]s that correspond to the outputs (i.e., return values) of this [`Program`].
    pub(crate) output_ids: Vec<AtomId>,

    /// Ordered sequence of [`Instruction`]s that make up the computational graph of this [`Program`].
    pub(crate) instructions: Vec<Instruction<O>>,

    /// [`Parameter`] structure that can be used to map flat lists of inputs to structured `Input` values.
    pub(crate) input_structure: Input::ParameterStructure,

    /// [`Parameter`] structure that can be used to map flat lists of outputs to structured `Output` values.
    pub(crate) output_structure: Output::ParameterStructure,

    /// [`PhantomData`] marker that ties this [`Program`] to its structured `Input` and `Output` types
    /// without making it own either value family.
    pub(crate) marker: PhantomData<(Input, Output)>,
}

impl<T: Type, V: Traceable<T>, O: Operation<T>, Input: Parameterized<V>, Output: Parameterized<V>>
    Program<T, V, O, Input, Output>
{
    /// Returns the [`Atom`]s contained in this [`Program`], in the order in which they will be evaluated.
    #[inline]
    pub fn atoms(&self) -> &[Atom<T, V>] {
        &self.atoms
    }

    /// Returns the [`AtomId`]s of the [`Atom`]s that correspond to the inputs (i.e., arguments) of this [`Program`].
    #[inline]
    pub fn input_ids(&self) -> &[AtomId] {
        &self.input_ids
    }

    /// Returns the [`Atom`]s that correspond to the inputs of this [`Program`].
    #[inline]
    pub fn inputs(&self) -> impl Iterator<Item = &Atom<T, V>> {
        self.input_ids.iter().map(|input_id| &self.atoms[input_id.index])
    }

    /// Returns the structured `Input` of this [`Program`] parameterized by the corresponding [`Atom`]s.
    #[inline]
    pub fn input(&self) -> Result<Input::To<Atom<T, V>>, ParameterError>
    where
        Input::Family: ParameterizedFamily<Atom<T, V>>,
    {
        Input::To::<Atom<T, V>>::from_parameters(self.input_structure.clone(), self.inputs().cloned())
    }

    /// Returns the [`AtomId`]s of the [`Atom`]s that correspond to the outputs (i.e., return values)
    /// of this [`Program`].
    #[inline]
    pub fn output_ids(&self) -> &[AtomId] {
        &self.output_ids
    }

    /// Returns the [`Atom`]s that correspond to the outputs of this [`Program`].
    #[inline]
    pub fn outputs(&self) -> impl Iterator<Item = &Atom<T, V>> {
        self.output_ids.iter().map(|output_id| &self.atoms[output_id.index])
    }

    /// Returns the structured `Output` of this [`Program`] parameterized by the corresponding [`Atom`]s.
    #[inline]
    pub fn output(&self) -> Result<Output::To<Atom<T, V>>, ParameterError>
    where
        Output::Family: ParameterizedFamily<Atom<T, V>>,
    {
        Output::To::<Atom<T, V>>::from_parameters(self.output_structure.clone(), self.outputs().cloned())
    }

    /// Returns the ordered sequence of [`Instruction`]s that make up the computational graph of this [`Program`].
    #[inline]
    pub fn instructions(&self) -> &[Instruction<O>] {
        &self.instructions
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

    /// Computes transitive liveness for the [`Atom`]s and [`Instruction`]s of this [`Program`] (i.e., determines
    /// whether each atom or instruction contributes to at least one of the [`Program`]s outputs).
    pub fn live_sets(&self) -> ProgramLiveSets {
        fn mark_live<T: Type, V: Traceable<T>, O: Operation<T>, Input: Parameterized<V>, Output: Parameterized<V>>(
            program: &Program<T, V, O, Input, Output>,
            atom_id: AtomId,
            live_sets: &mut ProgramLiveSets,
            instruction_by_output: &[Option<usize>],
        ) {
            if live_sets.atoms[atom_id.index()] {
                return;
            }

            live_sets.atoms[atom_id.index()] = true;
            if let Some(instruction_index) = instruction_by_output[atom_id.index()] {
                if live_sets.instructions[instruction_index] {
                    return;
                }
                live_sets.instructions[instruction_index] = true;
                for input in program.instructions[instruction_index].inputs.iter().copied() {
                    mark_live(program, input, live_sets, instruction_by_output);
                }
            }
        }

        let instruction_by_output = self.instruction_by_output();
        let mut live_sets = ProgramLiveSets::new(vec![false; self.atoms.len()], vec![false; self.instructions.len()]);
        for output in self.output_ids.iter().copied() {
            mark_live(self, output, &mut live_sets, instruction_by_output.as_slice());
        }

        live_sets
    }

    /// Returns a cloned view of this [`Program`] whose public input and output types are flat vectors. The atom table,
    /// input atom identifiers, output atom identifiers, and instruction sequence are preserved exactly. Only the
    /// `Input` and `Output` type parameters change to `Vec<V>`, with placeholder structures sized to the flat input and
    /// output arities. This is useful for higher-order operations that store nested [`Program`]s as operation payloads
    /// and replay them positionally, without needing to preserve the caller's original [`Parameterized`] type.
    pub fn to_flat_program(&self) -> Program<T, V, O, Vec<V>, Vec<V>>
    where
        O: Clone,
    {
        Program {
            atoms: self.atoms.clone(),
            input_ids: self.input_ids.clone(),
            output_ids: self.output_ids.clone(),
            instructions: self.instructions.clone(),
            input_structure: vec![Placeholder; self.input_ids.len()],
            output_structure: vec![Placeholder; self.output_ids.len()],
            marker: PhantomData,
        }
    }

    /// Converts this [`Program`] into one whose public input and output types are flat vectors. This is the consuming
    /// counterpart of [`Program::to_flat_program`]. It preserves the atom table, input atom identifiers, output atom
    /// identifiers, and instruction sequence without cloning them, and only replaces the structured input and output
    /// metadata with [`Placeholder`] vector structures sized to the flat arities.
    pub fn into_flat_program(self) -> Program<T, V, O, Vec<V>, Vec<V>> {
        let input_structure = vec![Placeholder; self.input_ids.len()];
        let output_structure = vec![Placeholder; self.output_ids.len()];
        Program {
            atoms: self.atoms,
            input_ids: self.input_ids,
            output_ids: self.output_ids,
            instructions: self.instructions,
            input_structure,
            output_structure,
            marker: PhantomData,
        }
    }

    // TODO(eaplatanios): Review this.
    /// Rebuilds this [`Program`] with each operation mapped through `map_operation`.
    ///
    /// The atom table, input/output atom identifiers, and parameter structures are preserved exactly. This is useful
    /// for transforms that keep the same value graph but need to change operation payloads, such as replacing
    /// residual references in a reusable linear program with the concrete residual values captured by a particular
    /// linearization run.
    pub fn try_map_operations<P, MapOperationFn>(
        &self,
        mut map_operation: MapOperationFn,
    ) -> Result<Program<T, V, P, Input, Output>, TracingError>
    where
        P: Operation<T>,
        MapOperationFn: FnMut(&O) -> Result<P, TracingError>,
    {
        Ok(Program {
            atoms: self.atoms.clone(),
            input_ids: self.input_ids.clone(),
            output_ids: self.output_ids.clone(),
            instructions: self
                .instructions
                .iter()
                .map(|instruction| {
                    Ok(Instruction::new(
                        map_operation(instruction.operation())?,
                        instruction.inputs().to_vec(),
                        instruction.outputs().to_vec(),
                    ))
                })
                .collect::<Result<Vec<_>, TracingError>>()?,
            input_structure: self.input_structure.clone(),
            output_structure: self.output_structure.clone(),
            marker: PhantomData,
        })
    }

    // TODO(eaplatanios): Review this.
    /// Computes dense owning-instruction indices for every atom produced by an instruction.
    ///
    /// Input and constant atoms have no owning instruction and therefore map to `None`.
    fn instruction_by_output(&self) -> Vec<Option<usize>> {
        let mut instruction_by_output = vec![None; self.atoms.len()];
        for (instruction_index, instruction) in self.instructions.iter().enumerate() {
            for output in instruction.outputs.iter().copied() {
                if let Some(slot) = instruction_by_output.get_mut(output.index()) {
                    *slot = Some(instruction_index);
                }
            }
        }
        instruction_by_output
    }

    // TODO(eaplatanios): Review this.
    /// Propagates a boolean dependency color from selected inputs through the program.
    ///
    /// The `input_depends` callback is invoked once for each public input atom, in input order. An instruction output
    /// is colored when any of the instruction's inputs are colored.
    pub fn dependency_mask_from_inputs<InputDepends>(&self, mut input_depends: InputDepends) -> Vec<bool>
    where
        InputDepends: FnMut(usize, AtomId) -> bool,
    {
        let mut depends = vec![false; self.atoms.len()];
        for (input_index, atom_id) in self.input_ids.iter().copied().enumerate() {
            depends[atom_id.index()] = input_depends(input_index, atom_id);
        }
        for instruction in self.instructions.iter() {
            let instruction_depends = instruction.inputs.iter().copied().any(|input| depends[input.index()]);
            for output in instruction.outputs.iter().copied() {
                depends[output.index()] = instruction_depends;
            }
        }
        depends
    }

    // TODO(eaplatanios): Review this.
    /// Rebuilds a flat program after replacing selected source atoms with public inputs.
    ///
    /// `replacement_input_atoms` become the public inputs of the returned program, in the provided order. Other
    /// requested outputs are rebuilt by recursively copying their producer instructions. `allow_unreplaced_variable`
    /// decides whether a non-replacement variable atom may be rebuilt from its producer, while
    /// `missing_producer_error` supplies the diagnostic for an allowed variable atom that has no producer.
    fn rebuild_flat_with_replacements<AllowVariableFn, MissingProducerFn>(
        &self,
        replacement_input_atoms: &[AtomId],
        output_atoms: &[AtomId],
        mut allow_unreplaced_variable: AllowVariableFn,
        mut missing_producer_error: MissingProducerFn,
    ) -> Result<Program<T, V, O, Vec<V>, Vec<V>>, TracingError>
    where
        O: Clone,
        AllowVariableFn: FnMut(AtomId) -> Result<(), TracingError>,
        MissingProducerFn: FnMut(AtomId) -> TracingError,
    {
        fn remap_atom<
            T: Type,
            V: Traceable<T>,
            O: Clone + Operation<T>,
            Input: Parameterized<V>,
            Output: Parameterized<V>,
            AllowVariableFn,
            MissingProducerFn,
        >(
            atom_id: AtomId,
            program: &Program<T, V, O, Input, Output>,
            builder: &mut ProgramBuilder<T, V, O>,
            atom_mapping: &mut HashMap<AtomId, AtomId>,
            replacements: &HashMap<AtomId, AtomId>,
            instruction_by_output: &[Option<usize>],
            allow_unreplaced_variable: &mut AllowVariableFn,
            missing_producer_error: &mut MissingProducerFn,
        ) -> Result<AtomId, TracingError>
        where
            AllowVariableFn: FnMut(AtomId) -> Result<(), TracingError>,
            MissingProducerFn: FnMut(AtomId) -> TracingError,
        {
            if let Some(mapped_atom) = atom_mapping.get(&atom_id) {
                return Ok(*mapped_atom);
            }
            if let Some(mapped_input) = replacements.get(&atom_id) {
                atom_mapping.insert(atom_id, *mapped_input);
                return Ok(*mapped_input);
            }

            let atom = program.atoms.get(atom_id.index()).ok_or(TracingError::UnboundAtomId { id: atom_id })?;
            let mapped_atom = match atom {
                Atom::Constant(value) => builder.add_constant(value.clone()),
                Atom::Variable(_) => {
                    allow_unreplaced_variable(atom_id)?;
                    let instruction_index =
                        instruction_by_output[atom_id.index()].ok_or_else(|| missing_producer_error(atom_id))?;
                    let instruction = &program.instructions[instruction_index];
                    let inputs = instruction
                        .inputs
                        .iter()
                        .copied()
                        .map(|input| {
                            remap_atom(
                                input,
                                program,
                                builder,
                                atom_mapping,
                                replacements,
                                instruction_by_output,
                                allow_unreplaced_variable,
                                missing_producer_error,
                            )
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let outputs = instruction
                        .outputs
                        .iter()
                        .map(|output| builder.add_variable(program.atoms[output.index()].r#type().into_owned()))
                        .collect::<Vec<_>>();
                    builder.add_instruction_unchecked(Instruction::new(
                        instruction.operation.clone(),
                        inputs,
                        outputs.clone(),
                    ));
                    for (old_output, new_output) in instruction.outputs.iter().copied().zip(outputs.iter().copied()) {
                        atom_mapping.insert(old_output, new_output);
                    }
                    atom_mapping
                        .get(&atom_id)
                        .copied()
                        .ok_or_else(|| TracingError::MalformedProgram(format!("remapped atom {atom_id} was missing")))?
                }
            };
            atom_mapping.insert(atom_id, mapped_atom);
            Ok(mapped_atom)
        }

        let instruction_by_output = self.instruction_by_output();
        let mut builder = ProgramBuilder::new();
        let mut replacements = HashMap::with_capacity(replacement_input_atoms.len());
        for atom_id in replacement_input_atoms.iter().copied() {
            let r#type = self
                .atoms
                .get(atom_id.index())
                .ok_or(TracingError::UnboundAtomId { id: atom_id })?
                .r#type()
                .into_owned();
            replacements.insert(atom_id, builder.add_input(r#type));
        }

        let mut atom_mapping = replacements.clone();
        let outputs = output_atoms
            .iter()
            .copied()
            .map(|output| {
                remap_atom(
                    output,
                    self,
                    &mut builder,
                    &mut atom_mapping,
                    &replacements,
                    instruction_by_output.as_slice(),
                    &mut allow_unreplaced_variable,
                    &mut missing_producer_error,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        builder.build(outputs, vec![Placeholder; replacement_input_atoms.len()], vec![Placeholder; output_atoms.len()])
    }

    // TODO(eaplatanios): Review this.
    /// Rebuilds a flat program over selected input and output atoms.
    ///
    /// `kept_input_atoms` become the public inputs of the returned program, in the provided order. `output_atoms`
    /// become the public outputs. All transitive producer instructions needed by `output_atoms` are copied.
    pub fn project_flat(
        &self,
        kept_input_atoms: &[AtomId],
        output_atoms: &[AtomId],
    ) -> Result<Program<T, V, O, Vec<V>, Vec<V>>, TracingError>
    where
        O: Clone,
    {
        self.rebuild_flat_with_replacements(
            kept_input_atoms,
            output_atoms,
            |_| Ok(()),
            |atom_id| TracingError::MalformedProgram(format!("projected atom {atom_id} has no producer")),
        )
    }

    // TODO(eaplatanios): Review this.
    /// Rebuilds a flat apply-stage program by replacing selected dependencies with residual inputs.
    ///
    /// `dynamic_input_atoms` and `residual_atoms` become the public inputs of the returned program, in that order.
    /// Atoms not in either set must be marked as depending on the dynamic side by `depends_on_dynamic_input`; otherwise
    /// they would require a residual that was not supplied.
    pub fn factorized_apply_flat(
        &self,
        dynamic_input_atoms: &[AtomId],
        residual_atoms: &[AtomId],
        depends_on_dynamic_input: &[bool],
        output_atoms: &[AtomId],
    ) -> Result<Program<T, V, O, Vec<V>, Vec<V>>, TracingError>
    where
        O: Clone,
    {
        if depends_on_dynamic_input.len() != self.atoms.len() {
            return Err(TracingError::MalformedProgram(format!(
                "expected {} dependency entries but got {}",
                self.atoms.len(),
                depends_on_dynamic_input.len(),
            )));
        }
        let replacement_atoms = dynamic_input_atoms.iter().chain(residual_atoms.iter()).copied().collect::<Vec<_>>();
        self.rebuild_flat_with_replacements(
            replacement_atoms.as_slice(),
            output_atoms,
            |atom_id| {
                if depends_on_dynamic_input[atom_id.index()] {
                    Ok(())
                } else {
                    Err(TracingError::MalformedProgram(format!(
                        "factorized apply atom {atom_id} needs a residual input"
                    )))
                }
            },
            |atom_id| TracingError::MalformedProgram(format!("factorized apply atom {atom_id} has no producer")),
        )
    }

    /// Returns a simplified version of this [`Program`] with dead constants and [`Instruction`]s that do not contribute
    /// to the [`Program`]'s output removed.
    pub fn simplified(&self) -> Result<Self, TracingError>
    where
        O: Clone,
    {
        /// Adds the [`Atom`] that corresponds to the provided `atom_id` to the provided `builder`, along with its
        /// transitive producers, memoizing the old-to-new [`AtomId`] mapping in `atom_id_mapping`.
        fn add_atom_to_program_builder<
            T: Type,
            V: Traceable<T>,
            O: Clone + Operation<T>,
            Input: Parameterized<V>,
            Output: Parameterized<V>,
        >(
            program_builder: &mut ProgramBuilder<T, V, O>,
            atom_id_mapping: &mut HashMap<AtomId, AtomId>,
            atom_id: AtomId,
            program: &Program<T, V, O, Input, Output>,
            parent_instructions: &[Option<usize>],
        ) -> Result<AtomId, TracingError> {
            if let Some(mapped_atom) = atom_id_mapping.get(&atom_id) {
                return Ok(*mapped_atom);
            }
            let atom = program.atoms.get(atom_id.index).ok_or(TracingError::UnboundAtomId { id: atom_id })?;
            let atom =
                match atom {
                    Atom::Constant(value) => Ok(program_builder.add_constant(value.clone())),
                    Atom::Variable(_) => {
                        let instruction_index = parent_instructions.get(atom_id.index).copied().flatten().ok_or(
                            TracingError::MalformedProgram("variable atom has no owning instruction".to_string()),
                        )?;
                        let instruction = &program.instructions[instruction_index];
                        let inputs = instruction
                            .inputs
                            .iter()
                            .copied()
                            .map(|input| {
                                add_atom_to_program_builder(
                                    program_builder,
                                    atom_id_mapping,
                                    input,
                                    program,
                                    parent_instructions,
                                )
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        let outputs = program_builder.add_instruction(instruction.operation.clone(), inputs)?;
                        check_count!("output", outputs, instruction.outputs.len(), TracingError);
                        instruction.outputs.iter().copied().zip(outputs.iter().copied()).for_each(|(old, new)| {
                            atom_id_mapping.insert(old, new);
                        });
                        atom_id_mapping.get(&atom_id).copied().ok_or(TracingError::MalformedProgram(
                            "remapped instruction output was missing".to_string(),
                        ))
                    }
                }?;
            atom_id_mapping.insert(atom_id, atom);
            Ok(atom)
        }

        let mut parent_instructions = vec![None; self.atoms.len()];
        for (instruction_index, instruction) in self.instructions.iter().enumerate() {
            for output in instruction.outputs.iter().copied() {
                let parent_instruction =
                    parent_instructions.get_mut(output.index).ok_or(TracingError::UnboundAtomId { id: output })?;
                *parent_instruction = Some(instruction_index);
            }
        }

        let mut program_builder = ProgramBuilder::new();
        let mut atom_id_mapping = HashMap::with_capacity(self.atoms.len());
        for input_id in self.input_ids.iter().copied() {
            let input = self.atoms.get(input_id.index).ok_or(TracingError::UnboundAtomId { id: input_id })?;
            let Atom::Variable(input_type) = input else {
                return Err(TracingError::MalformedProgram("program input atom was not a variable".to_string()));
            };
            atom_id_mapping.insert(input_id, program_builder.add_input(input_type.clone()));
        }

        let output_ids = self
            .output_ids
            .iter()
            .copied()
            .map(|output| {
                add_atom_to_program_builder(
                    &mut program_builder,
                    &mut atom_id_mapping,
                    output,
                    self,
                    parent_instructions.as_slice(),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        program_builder.build(output_ids, self.input_structure.clone(), self.output_structure.clone())
    }

    /// Consumes this [`Program`] and returns a simplified version with dead constants and [`Instruction`]s that do not
    /// contribute to the [`Program`]'s output removed. Unlike [`Self::simplified`], this method moves live [`Atom`]s,
    /// [`Instruction`]s, and parameter structures into the returned [`Program`] instead of cloning them. This avoids
    /// copying constants and operations that are discarded during simplification.
    pub fn into_simplified(self) -> Result<Self, TracingError> {
        /// Adds the [`Atom`] that corresponds to the provided `atom_id` to the simplified [`Program`] vectors, along
        /// with its transitive producers, memoizing the old-to-new [`AtomId`] mapping in `atom_id_mapping`.
        fn add_atom_to_simplified_program<T: Type, V: Traceable<T>, O: Operation<T>>(
            atoms: &mut [Option<Atom<T, V>>],
            instructions: &mut [Option<Instruction<O>>],
            parent_instructions: &[Option<usize>],
            atom_id_mapping: &mut HashMap<AtomId, AtomId>,
            new_atoms: &mut Vec<Atom<T, V>>,
            new_instructions: &mut Vec<Instruction<O>>,
            atom_id: AtomId,
        ) -> Result<AtomId, TracingError> {
            if let Some(mapped_atom) = atom_id_mapping.get(&atom_id) {
                return Ok(*mapped_atom);
            }
            let is_constant = match atoms.get(atom_id.index) {
                Some(Some(Atom::Constant(_))) => true,
                Some(Some(Atom::Variable(_))) => false,
                Some(None) => {
                    return Err(TracingError::MalformedProgram(format!(
                        "atom {atom_id} was already moved while simplifying program",
                    )));
                }
                None => return Err(TracingError::UnboundAtomId { id: atom_id }),
            };
            if is_constant {
                let Some(Atom::Constant(value)) = atoms[atom_id.index].take() else {
                    unreachable!("constant atom kind was checked before moving the atom");
                };
                let new_atom = AtomId { index: new_atoms.len() };
                new_atoms.push(Atom::Constant(value));
                atom_id_mapping.insert(atom_id, new_atom);
                return Ok(new_atom);
            }
            let instruction_index = parent_instructions
                .get(atom_id.index)
                .copied()
                .flatten()
                .ok_or(TracingError::MalformedProgram("variable atom has no owning instruction".to_string()))?;
            let instruction = instructions[instruction_index]
                .take()
                .ok_or(TracingError::MalformedProgram("instruction was already moved".to_string()))?;
            let inputs = instruction
                .inputs
                .iter()
                .copied()
                .map(|input| {
                    add_atom_to_simplified_program(
                        atoms,
                        instructions,
                        parent_instructions,
                        atom_id_mapping,
                        new_atoms,
                        new_instructions,
                        input,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mut outputs = Vec::with_capacity(instruction.outputs.len());
            for output in instruction.outputs.iter().copied() {
                let output_atom =
                    atoms.get_mut(output.index).ok_or(TracingError::UnboundAtomId { id: output })?.take().ok_or(
                        TracingError::MalformedProgram("instruction output atom was already moved".to_string()),
                    )?;
                let Atom::Variable(output_type) = output_atom else {
                    return Err(TracingError::MalformedProgram(
                        "instruction output atom was not a variable".to_string(),
                    ));
                };
                let new_output = AtomId { index: new_atoms.len() };
                new_atoms.push(Atom::Variable(output_type));
                atom_id_mapping.insert(output, new_output);
                outputs.push(new_output);
            }
            new_instructions.push(Instruction { operation: instruction.operation, inputs, outputs });
            atom_id_mapping
                .get(&atom_id)
                .copied()
                .ok_or(TracingError::MalformedProgram("remapped instruction output was missing".to_string()))
        }

        let Program { atoms, input_ids, output_ids, instructions, input_structure, output_structure, marker: _ } = self;

        let expected_input_count = input_structure.parameter_count();
        check_count!("input", input_ids, expected_input_count, TracingError);

        let expected_output_count = output_structure.parameter_count();
        check_count!("output", output_ids, expected_output_count, TracingError);

        let mut parent_instructions = vec![None; atoms.len()];
        for (instruction_index, instruction) in instructions.iter().enumerate() {
            for output in instruction.outputs.iter().copied() {
                let parent_instruction =
                    parent_instructions.get_mut(output.index).ok_or(TracingError::UnboundAtomId { id: output })?;
                *parent_instruction = Some(instruction_index);
            }
        }

        let mut atoms = atoms.into_iter().map(Some).collect::<Vec<_>>();
        let mut instructions = instructions.into_iter().map(Some).collect::<Vec<_>>();
        let mut new_atoms = Vec::with_capacity(atoms.len());
        let mut new_input_ids = Vec::with_capacity(input_ids.len());
        let mut new_instructions = Vec::with_capacity(instructions.len());
        let mut atom_id_mapping = HashMap::with_capacity(atoms.len());
        for input_id in input_ids {
            let input = atoms
                .get_mut(input_id.index)
                .ok_or(TracingError::UnboundAtomId { id: input_id })?
                .take()
                .ok_or(TracingError::MalformedProgram("program input atom was already moved".to_string()))?;
            let Atom::Variable(input_type) = input else {
                return Err(TracingError::MalformedProgram("program input atom was not a variable".to_string()));
            };
            let new_input = AtomId { index: new_atoms.len() };
            new_atoms.push(Atom::Variable(input_type));
            new_input_ids.push(new_input);
            atom_id_mapping.insert(input_id, new_input);
        }

        let output_ids = output_ids
            .into_iter()
            .map(|output| {
                add_atom_to_simplified_program(
                    atoms.as_mut_slice(),
                    instructions.as_mut_slice(),
                    parent_instructions.as_slice(),
                    &mut atom_id_mapping,
                    &mut new_atoms,
                    &mut new_instructions,
                    output,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            atoms: new_atoms,
            input_ids: new_input_ids,
            output_ids,
            instructions: new_instructions,
            input_structure,
            output_structure,
            marker: PhantomData,
        })
    }

    /// Interprets/executes this [`Program`] with the provided input. This is the main replay entry point for staged
    /// [`Program`]s. It checks that the provided input value matches the program's expected input structure, evaluates
    /// the [`Instruction`]s in order, and finally builds a structured output value from the computed output values.
    pub fn interpret(&self, input: Input) -> Result<Output, TracingError>
    where
        O: InterpretableOperation<T, V>,
        Input::ParameterStructure: Debug + PartialEq,
    {
        // Validate that the caller supplied an input with the expected parameter structure.
        let input_structure = input.parameter_structure();
        if input_structure != self.input_structure {
            return Err(ParameterError::MismatchedParameterStructures {
                left_structure: format!("{:?}", self.input_structure),
                right_structure: format!("{input_structure:?}"),
            }
            .into());
        }

        // Flatten the structured input, replay using ordinary interpretation, and reshape the flat outputs
        // back into the structured `Output` form expected by this program.
        let inputs = input.into_parameters().collect::<Vec<_>>();
        let outputs = self.interpret_with(
            inputs,
            |_, constant| Ok(constant.clone()),
            |instruction, inputs| instruction.operation.interpret(inputs),
        )?;
        Ok(Output::from_parameters(self.output_structure.clone(), outputs)?)
    }

    /// Interprets/executes this [`Program`]'s [`Instruction`]s using the caller-supplied value semantics. Transforms
    /// can specialize this interpretation function by choosing a runtime value type `Value`, a constant-lifting
    /// closure, and an instruction-interpretation closure. Inputs and outputs are flat [`Vec`]s aligned with the
    /// program's [`Self::input_ids`] and [`Self::output_ids`]; structured-input/output handling stays at the call
    /// site so that callers can use any parameter family of their choice.
    ///
    /// # Parameters
    ///
    ///   - `inputs`: Flat input values aligned with [`Self::input_ids`].
    ///   - `lift_constant`: Closure that lifts an [`Atom::Constant`]'s carried `V` into the runtime leaf type `Value`.
    ///     This closure receives the constant's [`AtomId`] for callers that surface diagnostics or maintain parallel
    ///     atom tables and is invoked at most once per live constant atom, in atom-index order.
    ///   - `interpret_instruction`: Closure that interprets one [`Instruction`]'s [`Operation`] to its already-lifted
    ///     inputs and returns the instruction's outputs. The full [`Instruction`] is provided so that the closure can
    ///     inspect the operation's expected output [`Atom`] IDs when needed (e.g., to look up output [`Type`]s).
    pub fn interpret_with<Value, Error, LiftConstantFn, InterpretInstructionFn>(
        &self,
        inputs: Vec<Value>,
        mut lift_constant: LiftConstantFn,
        mut interpret_instruction: InterpretInstructionFn,
    ) -> Result<Vec<Value>, Error>
    where
        Value: Clone,
        Error: From<TracingError>,
        LiftConstantFn: FnMut(AtomId, &V) -> Result<Value, Error>,
        InterpretInstructionFn: FnMut(&Instruction<O>, &[Value]) -> Result<Vec<Value>, Error>,
    {
        check_count!("input", inputs, self.input_ids.len(), TracingError);

        // Count every future consumer of each atom, including final program outputs. These counts let us move each
        // value out on its last use and clone it only when a later consumer still needs it.
        let mut remaining_uses = vec![0usize; self.atoms.len()];
        for instruction in self.instructions.iter() {
            for input_id in instruction.inputs.iter().copied() {
                let Some(remaining_uses) = remaining_uses.get_mut(input_id.index) else {
                    return Err(TracingError::UnboundAtomId { id: input_id }.into());
                };
                *remaining_uses += 1;
            }
        }
        for output_id in self.output_ids.iter().copied() {
            let Some(remaining_uses) = remaining_uses.get_mut(output_id.index) else {
                return Err(TracingError::UnboundAtomId { id: output_id }.into());
            };
            *remaining_uses += 1;
        }

        // Store concrete input values in a sparse value table indexed by [`AtomId`].
        let mut values = vec![None; self.atoms.len()];
        for (input_id, input) in self.input_ids.iter().copied().zip(inputs) {
            let Some(slot) = values.get_mut(input_id.index) else {
                return Err(TracingError::UnboundAtomId { id: input_id }.into());
            };
            *slot = Some(input);
        }

        // Materialize literal constants that are live. Dead constants can remain unset because no instruction or
        // program output will read them.
        for (atom_index, atom) in self.atoms.iter().enumerate() {
            if remaining_uses[atom_index] == 0 {
                continue;
            }
            if let Atom::Constant(value) = atom {
                values[atom_index] = Some(lift_constant(AtomId { index: atom_index }, value)?);
            }
        }

        // Replay instructions in program order, reusing one scratch input buffer to avoid per-instruction allocation.
        let max_input_count = self.instructions.iter().map(|instruction| instruction.inputs.len()).max().unwrap_or(0);
        let mut instruction_inputs = Vec::with_capacity(max_input_count);
        for instruction in self.instructions.iter() {
            instruction_inputs.clear();
            for input_id in instruction.inputs.iter().copied() {
                // Consume the appropriate input value for the current instruction. If this is the last consumer,
                // move the value out of the table. Otherwise, clone it so later consumers can still read it.
                let remaining_uses = remaining_uses.get_mut(input_id.index).unwrap();
                debug_assert!(*remaining_uses > 0);
                *remaining_uses -= 1;
                let value = values.get_mut(input_id.index).unwrap();
                let value = if *remaining_uses == 0 { value.take().unwrap() } else { value.as_ref().unwrap().clone() };
                instruction_inputs.push(value);
            }

            // Apply the operation using the supplied dispatcher and ensure it produces the expected number of outputs.
            let outputs = interpret_instruction(instruction, instruction_inputs.as_slice())?;
            check_count!("output", outputs, instruction.outputs.len(), TracingError);

            for (output_id, output) in instruction.outputs.iter().copied().zip(outputs) {
                let Some(value) = values.get_mut(output_id.index) else {
                    return Err(TracingError::UnboundAtomId { id: output_id }.into());
                };

                // Keep only outputs with a future consumer. Dead instruction results do not need to occupy the table.
                if remaining_uses[output_id.index] != 0 {
                    *value = Some(output);
                }
            }
        }

        // Gather the program outputs using the same last-use transfer logic that we used for the instruction inputs.
        let mut outputs = Vec::with_capacity(self.output_ids.len());
        for output_id in self.output_ids.iter().copied() {
            let remaining_uses = remaining_uses.get_mut(output_id.index).unwrap();
            debug_assert!(*remaining_uses > 0);
            *remaining_uses -= 1;
            let value = values.get_mut(output_id.index).unwrap();
            let value = if *remaining_uses == 0 { value.take().unwrap() } else { value.as_ref().unwrap().clone() };
            outputs.push(value);
        }

        Ok(outputs)
    }
}

impl<T: Type, V: Traceable<T>, O: Operation<T>, Input: Parameterized<V>, Output: Parameterized<V>>
    Program<T, V, O, Input, Output>
{
    /// Renders this [`Program`] with the provided indentation level that is useful for situations where [`Program`]s
    /// are nested within other programs like with control flow [`Operation`]s.
    pub fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        write!(formatter, "{:indentation$}", "")?;
        write!(formatter, "lambda ")?;
        self.input_ids.iter().enumerate().try_for_each(|(index, input_id)| {
            if index > 0 {
                write!(formatter, ", {input_id}:{}", self.atoms[input_id.index].r#type())
            } else {
                write!(formatter, "{input_id}:{}", self.atoms[input_id.index].r#type())
            }
        })?;
        writeln!(formatter, " .")?;
        let mut instructions_by_first_output = vec![None; self.atoms.len()];
        for (index, instruction) in self.instructions.iter().enumerate() {
            if let Some(output_id) = instruction.outputs.first() {
                instructions_by_first_output[output_id.index] = Some(index);
            }
        }
        let mut binding_count = 0usize;
        let mut is_input = vec![false; self.atoms.len()];
        for input_id in self.input_ids.iter().copied() {
            is_input[input_id.index] = true;
        }
        for (atom_id, atom) in self.atoms.iter().enumerate() {
            match atom {
                Atom::Constant(_) => {
                    write!(formatter, "{:indentation$}", "")?;
                    writeln!(
                        formatter,
                        "{} {}:{} = const",
                        if binding_count == 0 { "let" } else { "   " },
                        AtomId { index: atom_id },
                        self.atoms[atom_id].r#type()
                    )?;
                    binding_count += 1;
                }
                Atom::Variable(_) if is_input[atom_id] => {}
                Atom::Variable(_) => {
                    if let Some(instruction_index) = instructions_by_first_output[atom_id] {
                        let instruction = &self.instructions[instruction_index];
                        write!(formatter, "{:indentation$}", "")?;
                        write!(formatter, "{} ", if binding_count == 0 { "let" } else { "   " })?;
                        instruction.outputs.iter().enumerate().try_for_each(|(index, output)| {
                            if index > 0 {
                                write!(formatter, ", {output}:{}", self.atoms[output.index].r#type())
                            } else {
                                write!(formatter, "{output}:{}", self.atoms[output.index].r#type())
                            }
                        })?;
                        write!(formatter, " = ")?;
                        instruction
                            .operation
                            .render(formatter, if binding_count == 0 { indentation } else { indentation + 4 })?;
                        instruction.inputs.iter().try_for_each(|input| write!(formatter, " {input}"))?;
                        writeln!(formatter)?;
                        binding_count += 1;
                    };
                }
            }
        }
        write!(formatter, "{:indentation$}", "")?;
        write!(formatter, "in (")?;
        self.output_ids.iter().enumerate().try_for_each(|(index, output)| {
            if index > 0 { write!(formatter, ", {output}") } else { write!(formatter, "{output}") }
        })?;
        write!(formatter, ")")
    }
}

impl<T: Type, V: Traceable<T>, O: Clone, Input: Parameterized<V>, Output: Parameterized<V>> Clone
    for Program<T, V, O, Input, Output>
{
    fn clone(&self) -> Self {
        Self {
            atoms: self.atoms.clone(),
            input_ids: self.input_ids.clone(),
            output_ids: self.output_ids.clone(),
            instructions: self.instructions.clone(),
            input_structure: self.input_structure.clone(),
            output_structure: self.output_structure.clone(),
            marker: PhantomData,
        }
    }
}

impl<T: Type, V: Traceable<T>, O: Operation<T>, Input: Parameterized<V>, Output: Parameterized<V>> Display
    for Program<T, V, O, Input, Output>
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

/// Liveness masks for a [`Program`].
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

/// Builder for [`Program`]s that carries for the most part the same information as the [`Program`] that is being built,
/// but also carries an optional [`TracingError`] that can be used to signal a failure during program construction.
#[derive(Clone, Debug)]
pub struct ProgramBuilder<T: Type, V: Typed<T> + Parameter, O: Operation<T>> {
    /// [`Atom`]s contained in the [`Program`] that is being built, in the order in which they will be evaluated.
    pub(crate) atoms: Vec<Atom<T, V>>,

    /// [`AtomId`]s of the [`Atom`]s that correspond to the inputs (i.e., arguments) of the [`Program`] being built.
    pub(crate) input_ids: Vec<AtomId>,

    /// Ordered sequence of [`Instruction`]s that make up the computational graph of the [`Program`] being built.
    pub(crate) instructions: Vec<Instruction<O>>,

    /// Optional [`TracingError`] encountered during program construction that will be propagated via [`Self::build`].
    pub(crate) error: Option<TracingError>,
}

impl<T: Type, V: Traceable<T>, O: Operation<T>> ProgramBuilder<T, V, O> {
    /// Creates a new [`ProgramBuilder`].
    #[inline]
    pub fn new() -> Self {
        Self { atoms: Vec::new(), input_ids: Vec::new(), instructions: Vec::new(), error: None }
    }

    /// Returns the atoms currently owned by this builder.
    #[inline]
    pub fn atoms(&self) -> &[Atom<T, V>] {
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
    pub fn error(&self) -> Option<&TracingError> {
        self.error.as_ref()
    }

    /// Adds an input [`Atom`] to the [`Program`] that is being built with the provided [`Type`].
    #[inline]
    pub fn add_input(&mut self, r#type: T) -> AtomId {
        let id = self.add_variable(r#type);
        self.input_ids.push(id);
        id
    }

    /// Adds the provided value as an [`Atom::Constant`] to the [`Program`] that is being built.
    #[inline]
    pub fn add_constant(&mut self, value: V) -> AtomId {
        let id = AtomId { index: self.atoms.len() };
        self.atoms.push(Atom::Constant(value));
        id
    }

    /// Adds an [`Atom::Variable`] to the [`Program`] that is being built with the provided [`Type`].
    #[inline]
    pub fn add_variable(&mut self, r#type: T) -> AtomId {
        let id = AtomId { index: self.atoms.len() };
        self.atoms.push(Atom::Variable(r#type));
        id
    }

    /// Adds an [`Instruction`] to the [`Program`] that is being built, that corresponds to an application of the
    /// provided [`Operation`] to the provided input [`Atom`]s.
    #[inline]
    pub fn add_instruction(&mut self, operation: O, inputs: Vec<AtomId>) -> Result<&[AtomId], TracingError> {
        let input_types = inputs
            .iter()
            .map(|input| {
                self.atoms
                    .get(input.index)
                    .map(|atom| atom.r#type().into_owned())
                    .ok_or(TracingError::UnboundAtomId { id: *input })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let output_types = operation.infer_output_types(input_types.as_slice())?;
        let outputs = output_types.into_iter().map(|r#type| self.add_variable(r#type)).collect::<Vec<_>>();
        self.instructions.push(Instruction { operation, inputs, outputs });
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

    /// Finalizes this [`ProgramBuilder`] into a [`Program`] with the provided input and output structures.
    #[inline]
    pub fn build<Input: Parameterized<V>, Output: Parameterized<V>>(
        self,
        output_ids: Vec<AtomId>,
        input_structure: Input::ParameterStructure,
        output_structure: Output::ParameterStructure,
    ) -> Result<Program<T, V, O, Input, Output>, TracingError> {
        if let Some(error) = self.error {
            return Err(error);
        }

        let expected_input_count = input_structure.parameter_count();
        check_count!("input", self.input_ids, expected_input_count, TracingError);

        let expected_output_count = output_structure.parameter_count();
        check_count!("output", output_ids, expected_output_count, TracingError);

        // Verify that variable dependencies are either inputs or previous instruction outputs.
        let mut variable_has_provider = vec![false; self.atoms.len()];
        for input_id in self.input_ids.iter().copied() {
            let input = self.atoms.get(input_id.index).ok_or(TracingError::UnboundAtomId { id: input_id })?;
            let Atom::Variable(_) = input else {
                return Err(TracingError::MalformedProgram("program input atom was not a variable".to_string()));
            };
            variable_has_provider[input_id.index] = true;
        }
        for instruction in self.instructions.iter() {
            for input_id in instruction.inputs.iter().copied() {
                let input = self.atoms.get(input_id.index).ok_or(TracingError::UnboundAtomId { id: input_id })?;
                if input.is_variable() && !variable_has_provider[input_id.index] {
                    return Err(TracingError::MalformedProgram("variable atom has no owning instruction".to_string()));
                }
            }
            for output_id in instruction.outputs.iter().copied() {
                let output = self.atoms.get(output_id.index).ok_or(TracingError::UnboundAtomId { id: output_id })?;
                let Atom::Variable(_) = output else {
                    return Err(TracingError::MalformedProgram(
                        "instruction output atom was not a variable".to_string(),
                    ));
                };
                variable_has_provider[output_id.index] = true;
            }
        }
        for output_id in output_ids.iter().copied() {
            let output = self.atoms.get(output_id.index).ok_or(TracingError::UnboundAtomId { id: output_id })?;
            if output.is_variable() && !variable_has_provider[output_id.index] {
                return Err(TracingError::MalformedProgram("variable atom has no owning instruction".to_string()));
            }
        }

        Ok(Program {
            atoms: self.atoms,
            input_ids: self.input_ids,
            instructions: self.instructions,
            output_ids,
            input_structure,
            output_structure,
            marker: PhantomData,
        })
    }
}

impl<T: Type, V: Traceable<T>, O: Operation<T>> Default for ProgramBuilder<T, V, O> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::cell::Cell;
    use std::fmt::Display;
    use std::rc::Rc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::macros::check_count;
    use crate::operations::OperationFormatter;
    use crate::operations::scalars::ScalarOperation;
    use crate::parameters::{ParameterError, Parameterized, Placeholder};
    use crate::tracing::TracingError;
    use crate::types::{DataType, TypeError};

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

    impl Operation<DataType> for LongMetadataOperation {
        #[inline]
        fn name(&self) -> &'static str {
            "long_metadata"
        }

        fn infer_output_types(&self, input_types: &[DataType]) -> Result<Vec<DataType>, TypeError> {
            check_count!("input", input_types, 1, TypeError);
            Ok(vec![input_types[0].clone()])
        }

        fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
            OperationFormatter::new(formatter, indentation, self.name())?
                .bracketed(|operation| operation.field("value", Self::METADATA_VALUE))
        }
    }

    #[test]
    fn test_atom() {
        let constant = Atom::<DataType, f64>::Constant(3.0);
        let variable = Atom::<DataType, f64>::Variable(DataType::F64);

        assert!(constant.is_constant());
        assert!(!constant.is_variable());
        assert_eq!(constant.as_constant(), Some(&3.0));
        assert_eq!(constant.r#type().into_owned(), DataType::F64);

        assert!(variable.is_variable());
        assert_eq!(variable.as_constant(), None);
        assert_eq!(variable.r#type().into_owned(), DataType::F64);
    }

    #[test]
    fn test_atom_id() {
        assert_eq!(AtomId { index: 42 }.to_string(), "%42");
    }

    #[test]
    fn test_program() {
        // Test simple program with one argument.
        let mut builder = ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new();
        let i0 = builder.add_input(DataType::F64);
        let c0 = builder.add_constant(3.0f64);
        let o0 = builder.add_instruction(ScalarOperation::Add, vec![i0, c0]).unwrap()[0];
        let program = builder.build::<f64, f64>(vec![o0], Placeholder, Placeholder).unwrap();
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
        let mut builder = ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new();
        let i0 = builder.add_input(DataType::F64);
        let i1 = builder.add_input(DataType::F64);
        let v0 = builder.add_instruction(ScalarOperation::Scale { factor: 2.0 }, vec![i0]).unwrap()[0];
        let o0 = builder.add_instruction(ScalarOperation::Add, vec![v0, i1]).unwrap()[0];
        let program = builder.build::<(f64, f64), f64>(vec![o0], (Placeholder, Placeholder), Placeholder).unwrap();
        let input = program.input().unwrap();
        let output = program.output().unwrap();
        assert_eq!(program.interpret((2.0, 3.0)), Ok(7.0));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64, %1:f64 .
                let %2:f64 = scale [factor=2] %0
                    %3:f64 = add %2 %1
                in (%3)
            "}
            .trim_end(),
        );
        assert!(matches!(input.0, Atom::Variable(r#type) if r#type == DataType::F64));
        assert!(matches!(input.1, Atom::Variable(r#type) if r#type == DataType::F64));
        assert!(matches!(output, Atom::Variable(r#type) if r#type == DataType::F64));

        // Test a program that contains an operation with long metadata that should be rendered on multiple lines.
        let mut builder = ProgramBuilder::<DataType, f64, LongMetadataOperation>::new();
        let i0 = builder.add_input(DataType::F64);
        let o0 = builder.add_instruction(LongMetadataOperation, vec![i0]).unwrap()[0];
        let program = builder.build::<f64, f64>(vec![o0], Placeholder, Placeholder).unwrap();
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
        let mut builder = ProgramBuilder::<DataType, f32, ScalarOperation<f32>>::new();
        let i0 = builder.add_input(DataType::F32);
        let o0 = builder.add_instruction(ScalarOperation::Add, vec![i0, i0]).unwrap()[0];
        let program = builder.build::<f32, (f32, f32)>(vec![o0, o0], Placeholder, (Placeholder, Placeholder)).unwrap();
        let input = program.input().unwrap();
        let output = program.output().unwrap();
        assert_eq!(program.interpret(2.0f32), Ok((4.0f32, 4.0f32)));
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
        let mut builder = ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new();
        builder.add_input(DataType::F64);
        let o0 = builder.add_variable(DataType::F64);
        assert!(matches!(
            builder.build::<f64, f64>(vec![o0], Placeholder, Placeholder),
            Err(TracingError::MalformedProgram(message)) if message == "variable atom has no owning instruction",
        ));

        // Test a case where we have an instruction input atom with no parent instruction.
        let mut builder = ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new();
        let i0 = builder.add_input(DataType::F64);
        let v0 = builder.add_variable(DataType::F64);
        let o0 = builder.add_instruction(ScalarOperation::Add, vec![i0, v0]).unwrap()[0];
        assert!(matches!(
            builder.build::<f64, f64>(vec![o0], Placeholder, Placeholder),
            Err(TracingError::MalformedProgram(message)) if message == "variable atom has no owning instruction",
        ));
    }

    #[test]
    fn test_program_interpret_lifts_live_constants_once() {
        let mut builder = ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new();
        let i0 = builder.add_input(DataType::F64);
        let c0 = builder.add_constant(7.0f64);
        let c1 = builder.add_constant(3.0f64);
        let o0 = builder.add_instruction(ScalarOperation::Add, vec![i0, c1]).unwrap()[0];
        let program = builder.build::<f64, f64>(vec![o0], Placeholder, Placeholder).unwrap();
        let mut lifted_constants = Vec::new();
        assert_eq!(
            program.interpret_with(
                vec![2.0f64],
                |atom_id, value| {
                    lifted_constants.push((atom_id, *value));
                    Ok(*value)
                },
                |instruction, inputs| instruction.operation.interpret(inputs),
            ),
            Ok(vec![5.0f64]),
        );
        assert_eq!(lifted_constants, vec![(c1, 3.0f64)]);
        assert_eq!(c0, AtomId { index: 1 });
    }

    #[test]
    fn test_program_interpret_with_mismatched_parameter_structures() {
        let mut builder = ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new();
        let i0 = builder.add_input(DataType::F64);
        let program = builder.build::<Vec<f64>, f64>(vec![i0], vec![Placeholder], Placeholder).unwrap();
        assert!(matches!(
            program.interpret(vec![1.0f64, 2.0f64]),
            Err(TracingError::Parameter(ParameterError::MismatchedParameterStructures {
                left_structure,
                right_structure,
            })) if left_structure == format!("{:?}", vec![Placeholder])
                && right_structure == format!("{:?}", vec![1.0f64, 2.0f64].parameter_structure())
        ));
    }

    #[test]
    fn test_program_interpret_with_wrong_number_of_operation_inputs() {
        let mut builder = ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new();
        let i0 = builder.add_input(DataType::F64);
        let program = builder.build::<f64, f64>(vec![i0], Placeholder, Placeholder).unwrap();
        assert!(matches!(
            program.interpret_with(
                Vec::<f64>::new(),
                |_, value| Ok(*value),
                |instruction, inputs| instruction.operation.interpret(inputs),
            ),
            Err(TracingError::InvalidInputCount { expected: 1, got: 0 }),
        ));
    }

    #[test]
    fn test_program_interpret_with_wrong_number_of_operation_outputs() {
        let mut builder = ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new();
        let i0 = builder.add_input(DataType::F64);
        let o0 = builder.add_instruction(ScalarOperation::Scale { factor: 2.0 }, vec![i0]).unwrap()[0];
        let program = builder.build::<f64, f64>(vec![o0], Placeholder, Placeholder).unwrap();
        assert!(matches!(
            program.interpret_with(
                vec![2.0f64],
                |_, value| Ok(*value),
                |_, _| Ok::<Vec<f64>, TracingError>(Vec::new()),
            ),
            Err(TracingError::InvalidOutputCount { expected: 1, got: 0 }),
        ));
    }

    #[test]
    fn test_program_live_sets() {
        let mut builder = ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new();
        let live_input = builder.add_input(DataType::F64);
        let dead_input = builder.add_input(DataType::F64);
        let live_constant = builder.add_constant(3.0f64);
        let dead_constant = builder.add_constant(5.0f64);
        let scaled = builder.add_instruction(ScalarOperation::Scale { factor: 2.0 }, vec![live_input]).unwrap()[0];
        let output = builder.add_instruction(ScalarOperation::Add, vec![scaled, live_constant]).unwrap()[0];
        let _dead_output = builder.add_instruction(ScalarOperation::Add, vec![dead_input, dead_constant]).unwrap()[0];
        let program = builder.build::<(f64, f64), f64>(vec![output], (Placeholder, Placeholder), Placeholder).unwrap();
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
                false, // `_dead_output`
            ],
        );
        assert_eq!(
            live_sets.instructions(),
            &[
                true,  // `scaled`
                true,  // `output`
                false, // `_dead_output`
            ],
        );
    }

    #[test]
    fn test_program_to_flat_program_and_into_flat_program() {
        let mut builder = ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new();
        let i0 = builder.add_input(DataType::F64);
        let i1 = builder.add_input(DataType::F64);
        let v0 = builder.add_instruction(ScalarOperation::Scale { factor: 2.0 }, vec![i0]).unwrap()[0];
        let o0 = builder.add_instruction(ScalarOperation::Add, vec![v0, i1]).unwrap()[0];
        let program = builder.build::<(f64, f64), f64>(vec![o0], (Placeholder, Placeholder), Placeholder).unwrap();

        let flat_program = program.to_flat_program();
        assert_eq!(flat_program.input_structure(), &vec![Placeholder, Placeholder]);
        assert_eq!(flat_program.output_structure(), &vec![Placeholder]);
        assert_eq!(flat_program.interpret(vec![2.0, 3.0]), Ok(vec![7.0]));

        let flat_program = program.into_flat_program();
        assert_eq!(flat_program.input_structure(), &vec![Placeholder, Placeholder]);
        assert_eq!(flat_program.output_structure(), &vec![Placeholder]);
        assert_eq!(flat_program.interpret(vec![2.0, 3.0]), Ok(vec![7.0]));
    }

    #[test]
    fn test_program_simplified() {
        let mut builder = ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new();
        let i0 = builder.add_input(DataType::F64);
        let c0 = builder.add_constant(2.0f64);
        let c1 = builder.add_constant(3.0f64);
        let _ = builder.add_instruction(ScalarOperation::Add, vec![i0, c0]).unwrap()[0];
        let v1 = builder.add_instruction(ScalarOperation::Add, vec![i0, c1]).unwrap()[0];
        let program = builder.build::<f64, (f64, f64)>(vec![v1, v1], Placeholder, (Placeholder, Placeholder)).unwrap();
        let simplified = program.simplified().unwrap();

        assert_eq!(c0, AtomId { index: 1 });
        assert_eq!(simplified.interpret(2.0f64), Ok((5.0f64, 5.0f64)));
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

        impl Typed<DataType> for CloneCountingValue {
            fn r#type(&self) -> Cow<'_, DataType> {
                Cow::Owned(DataType::F64)
            }
        }

        impl Traceable<DataType> for CloneCountingValue {}

        let value_clone_count = Rc::new(Cell::new(0));
        let mut builder = ProgramBuilder::<_, _, ScalarOperation<CloneCountingValue>>::new();
        let i0 = builder.add_input(DataType::F64);
        let c0 = builder.add_constant(CloneCountingValue::new(2.0, Rc::clone(&value_clone_count)));
        let c1 = builder.add_constant(CloneCountingValue::new(3.0, Rc::clone(&value_clone_count)));
        let v0 = builder.add_instruction(ScalarOperation::Add, vec![i0, c0]).unwrap()[0];
        let v1 = builder.add_instruction(ScalarOperation::Add, vec![i0, c1]).unwrap()[0];
        let program = builder
            .build::<CloneCountingValue, (CloneCountingValue, CloneCountingValue)>(
                vec![v1, v1],
                Placeholder,
                (Placeholder, Placeholder),
            )
            .unwrap();

        assert_eq!(v0, AtomId { index: 3 });
        assert_eq!(v1, AtomId { index: 4 });
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
        assert_eq!(simplified.input_ids, vec![AtomId { index: 0 }]);
        assert_eq!(simplified.output_ids, vec![AtomId { index: 2 }, AtomId { index: 2 }]);
        assert_eq!(simplified.atoms.len(), 3);
        assert!(matches!(simplified.atoms.get(1), Some(Atom::Constant(value)) if value.value == 3.0));
        assert_eq!(simplified.instructions.len(), 1);
        assert_eq!(simplified.instructions[0].inputs, vec![AtomId { index: 0 }, AtomId { index: 1 }]);
        assert_eq!(simplified.instructions[0].outputs, vec![AtomId { index: 2 }]);
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
    fn test_program_builder() {
        let mut builder = ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new();
        let i0 = builder.add_input(DataType::F64);
        let i1 = builder.add_input(DataType::F64);
        let c0 = builder.add_constant(2.0f64);
        let v0 = builder.add_instruction(ScalarOperation::Scale { factor: 2.0 }, vec![i0]).unwrap()[0];
        let v1 = builder.add_instruction(ScalarOperation::Add, vec![v0, i1]).unwrap()[0];
        assert_eq!(builder.input_ids, vec![i0, i1]);
        assert!(matches!(
            builder.atoms.get(i0.index),
            Some(Atom::Variable(r#type)) if *r#type == DataType::F64
        ));
        assert!(matches!(
            builder.atoms.get(i1.index),
            Some(Atom::Variable(r#type)) if *r#type == DataType::F64
        ));
        assert!(matches!(builder.atoms.get(c0.index), Some(Atom::Constant(value)) if *value == 2.0));
        assert!(matches!(
            builder.atoms.get(v0.index),
            Some(Atom::Variable(r#type)) if *r#type == DataType::F64
        ));
        assert!(matches!(
            builder.atoms.get(v1.index),
            Some(Atom::Variable(r#type)) if *r#type == DataType::F64
        ));
        assert_eq!(builder.instructions.len(), 2);
        assert_eq!(builder.instructions[0].inputs, vec![i0]);
        assert_eq!(builder.instructions[0].outputs, vec![v0]);
        assert_eq!(builder.instructions[1].inputs, vec![v0, i1]);
        assert_eq!(builder.instructions[1].outputs, vec![v1]);

        let program = builder.build::<(f64, f64), f64>(vec![v1], (Placeholder, Placeholder), Placeholder).unwrap();
        assert_eq!(program.input_ids, vec![i0, i1]);
        assert_eq!(program.output_ids, vec![v1]);
        assert_eq!(program.instructions.len(), 2);
        assert_eq!(program.interpret((2.0f64, 38.0f64)), Ok(42.0f64));
    }

    #[test]
    fn test_program_builder_rejects_unbound_instruction_inputs() {
        let mut builder = ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new();
        let v0 = builder.add_instruction(ScalarOperation::Add, vec![AtomId { index: 42 }, AtomId { index: 99 }]);
        assert!(matches!(v0, Err(TracingError::UnboundAtomId { id }) if id == AtomId { index: 42 }));
    }

    #[test]
    fn test_program_builder_build_returns_error() {
        let mut builder = ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new();
        builder.error = Some(TracingError::InvalidInputCount { expected: 1, got: 0 });
        assert!(matches!(
            builder.build::<f64, f64>(Vec::new(), Placeholder, Placeholder),
            Err(TracingError::InvalidInputCount { expected: 1, got: 0 }),
        ));
    }

    #[test]
    fn test_program_builder_build_rejects_invalid_input_count() {
        let builder = ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new();
        assert!(matches!(
            builder.build::<f64, ()>(Vec::new(), Placeholder, ()),
            Err(TracingError::InvalidInputCount { expected: 1, got: 0 }),
        ));
    }

    #[test]
    fn test_program_builder_build_rejects_invalid_output_count() {
        let mut builder = ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new();
        builder.add_input(DataType::F64);
        assert!(matches!(
            builder.build::<f64, f64>(Vec::new(), Placeholder, Placeholder),
            Err(TracingError::InvalidOutputCount { expected: 1, got: 0 }),
        ));
    }
}
