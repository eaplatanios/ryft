use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;

use ryft_macros::Parameter;

use crate::operations::{InterpretableOperation, Operation};
use crate::parameters::{Parameter, ParameterError, Parameterized, ParameterizedFamily};
use crate::tracing::TracingError;
use crate::types::{Type, Typed};

/// Identifies values in [`Program`]s. [`Value`] is a subtrait of [`Traceable`] implemented by types that carry real
/// data, such as arrays. The sole purpose of this marker is to give Rust's coherence checker a way to tell two blanket
/// implementations apart. Each composable transform (e.g., for just-in-time compilation or automatic differentiation)
/// provides:
///
///   1. an implementation for `V: Value<T>` that evaluates the transform on concrete data, and
///   2. an implementation for `Tracer<V>` that stages the transform into the enclosing traced [`Program`].
///
/// Because `Tracer<V>` implements [`Traceable`] but not [`Value`], these two implementations never overlap.
pub trait Value<T: Type>: Traceable<T> {}

/// Represents leaf values that can participate in traced [`Program`]s. [`Traceable`] is implemented by every type that
/// can appear as a leaf in a staged [`Program`]: both concrete data types such as `f32`, `f64`, and backend arrays, and
/// tracing wrappers such as [`Tracer`](crate::Tracer). It ties each leaf to a type descriptor `T` via [`Typed`] and
/// requires [`Debug`] and [`Display`] so that diagnostics, constants, and [`Operation`] metadata can render their
/// carried values directly.
pub trait Traceable<T: Type>: Clone + Debug + Display + Parameter + Typed<T> {}

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
    pub index: usize,
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
    pub operation: O,

    /// [`AtomId`]s of the input [`Atom`]s consumed by this [`Instruction`].
    pub inputs: Vec<AtomId>,

    /// [`AtomId`]s of the output [`Atom`]s produced by this [`Instruction`].
    pub outputs: Vec<AtomId>,
}

/// [`Program`] that is produced by tracing and which can be interpreted or compiled and executed by a backend. It
/// consists of a sequence of [`Instruction`]s paired with [`Parameterized`] input and output types. This is the primary
/// intermediate representation (IR) used by the Ryft tracing and transformation system (e.g., to support things like
/// automatic differentiation and just-in-time compilation).
#[derive(Debug)]
pub struct Program<T: Type, V: Typed<T> + Parameter, O, Input: Parameterized<V>, Output: Parameterized<V>> {
    /// [`Atom`]s contained in this [`Program`], in the order in which they will be evaluated.
    pub atoms: Vec<Atom<T, V>>,

    /// [`AtomId`]s of the [`Atom`]s that correspond to the inputs (i.e., arguments) of this [`Program`].
    pub input_ids: Vec<AtomId>,

    /// [`AtomId`]s of the [`Atom`]s that correspond to the outputs (i.e., return values) of this [`Program`].
    pub output_ids: Vec<AtomId>,

    /// Ordered sequence of [`Instruction`]s that make up the computational graph of this [`Program`].
    pub instructions: Vec<Instruction<O>>,

    /// [`Parameter`] structure that can be used to map flat lists of inputs to structured `Input` values.
    pub input_structure: Input::ParameterStructure,

    /// [`Parameter`] structure that can be used to map flat lists of outputs to structured `Output` values.
    pub output_structure: Output::ParameterStructure,

    /// [`PhantomData`] marker that ties this [`Program`] to its structured `Input` and `Output` types.
    pub marker: PhantomData<fn(Input) -> Output>,
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

impl<T: Type, V: Traceable<T>, O: Operation<T>, Input: Parameterized<V>, Output: Parameterized<V>>
    Program<T, V, O, Input, Output>
{
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
                        if outputs.len() != instruction.outputs.len() {
                            return Err(TracingError::InvalidOutputCount {
                                expected: instruction.outputs.len(),
                                got: outputs.len(),
                            });
                        }
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
        if input_ids.len() != expected_input_count {
            return Err(TracingError::InvalidInputCount { expected: expected_input_count, got: input_ids.len() });
        }

        let expected_output_count = output_structure.parameter_count();
        if output_ids.len() != expected_output_count {
            return Err(TracingError::InvalidOutputCount { expected: expected_output_count, got: output_ids.len() });
        }

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
    ///     THis closure receives the constant's [`AtomId`] for callers that surface diagnostics or maintain parallel
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
        if inputs.len() != self.input_ids.len() {
            return Err(TracingError::InvalidInputCount { expected: self.input_ids.len(), got: inputs.len() }.into());
        }

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
            if outputs.len() != instruction.outputs.len() {
                return Err(TracingError::InvalidOutputCount {
                    expected: instruction.outputs.len(),
                    got: outputs.len(),
                }
                .into());
            }

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

impl<T: Type, V: Traceable<T>, O: Clone + Operation<T>, Input: Parameterized<V>, Output: Parameterized<V>>
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

impl<T: Type, V: Traceable<T>, O: Clone + Operation<T>, Input: Parameterized<V>, Output: Parameterized<V>> Display
    for Program<T, V, O, Input, Output>
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

/// Builder for [`Program`]s that carries for the most part the same information as the [`Program`] that is being built,
/// but also carries an optional [`TracingError`] that can be used to signal a failure during program construction.
#[derive(Clone, Debug)]
pub struct ProgramBuilder<T: Type, V: Typed<T> + Parameter, O: Operation<T>> {
    /// [`Atom`]s contained in the [`Program`] that is being built, in the order in which they will be evaluated.
    pub atoms: Vec<Atom<T, V>>,

    /// [`AtomId`]s of the [`Atom`]s that correspond to the inputs (i.e., arguments) of the [`Program`] being built.
    pub input_ids: Vec<AtomId>,

    /// Ordered sequence of [`Instruction`]s that make up the computational graph of the [`Program`] being built.
    pub instructions: Vec<Instruction<O>>,

    /// Optional [`TracingError`] encountered during program construction that will be propagated via [`Self::build`].
    pub error: Option<TracingError>,
}

impl<T: Type, V: Traceable<T>, O: Operation<T>> ProgramBuilder<T, V, O> {
    /// Creates a new [`ProgramBuilder`].
    #[inline]
    pub fn new() -> Self {
        Self { atoms: Vec::new(), input_ids: Vec::new(), instructions: Vec::new(), error: None }
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
        if self.input_ids.len() != expected_input_count {
            return Err(TracingError::InvalidInputCount { expected: expected_input_count, got: self.input_ids.len() });
        }

        let expected_output_count = output_structure.parameter_count();
        if output_ids.len() != expected_output_count {
            return Err(TracingError::InvalidOutputCount { expected: expected_output_count, got: output_ids.len() });
        }

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

impl<T: Type, V: Traceable<T>, O: Clone + Operation<T>> Default for ProgramBuilder<T, V, O> {
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

    use crate::operations::OperationFormatter;
    use crate::parameters::{ParameterError, Parameterized, Placeholder};
    use crate::tracing::TracingError;
    use crate::tracing_v2::ScalarOperation;
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
            if input_types.len() != 1 {
                return Err(TypeError { message: format!("expected 1 input type but got {}", input_types.len()) });
            }
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
