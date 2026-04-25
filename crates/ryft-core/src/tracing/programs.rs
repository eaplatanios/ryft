use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;

use ryft_macros::Parameter;

use crate::parameters::{Parameter, ParameterError, Parameterized, ParameterizedFamily};
use crate::tracing::TracingError;
use crate::types::{Type, TypeError, Typed};

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
/// tracing wrappers such as [`Tracer`](crate::Tracer). It ties each leaf to a type descriptor `T` via [`Typed`], but
/// does not imply any other requirements for the underlying values.
pub trait Traceable<T: Type>: Clone + Parameter + Typed<T> {}

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
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Parameter)]
pub struct AtomId {
    /// Zero-based index of the corresponding [`Atom`] inside the owning [`Program`]'s atom table.
    pub index: usize,
}

impl Display for AtomId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "%{}", self.index)
    }
}

/// [`Operation`] that can appear in [`Program`]s. [`Operation`] invocations are represented as [`Instruction`]s in
/// [`Program`]s. This trait represents the high-level operation interface that only requires operations to be able to
/// provide their name and to infer their output [`Type`]s given their input [`Type`]s.
pub trait Operation<T: Type>: Debug + Display {
    /// Returns the name of this [`Operation`] that is used in diagnostics and when rendering [`Program`]s as strings.
    fn name(&self) -> &'static str;

    /// Infers the output [`Type`]s of this [`Operation`] from the provided input [`Type`]s without executing it.
    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError>;
}

/// [`InterpretableOperation`]s are [`Operation`]s that can be interpreted (i.e., executed) given concrete input values.
pub trait InterpretableOperation<T: Type, V: Typed<T>>: Operation<T> {
    /// Interprets this [`Operation`] given the provided input values and returns the resulting output values.
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError>;
}

/// [`Instruction`]s represent applications of [`Operation`]s to input values in [`Program`]s. [`Program`]s are
/// purely dataflow-based, and so [`Instruction`]s execute in sequential order with no control-flow nodes.
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
pub struct Program<
    T: Type,
    V: Typed<T> + Parameter,
    O: Clone + Operation<T>,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
> {
    /// [`Atom`]s contained in this [`Program`], in the order in which they will be computed/evaluated.
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

impl<
    T: Type,
    V: Traceable<T>,
    O: Clone + Operation<T>,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Output: Parameterized<V, ParameterStructure: Clone>,
> Clone for Program<T, V, O, Input, Output>
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

impl<T: Type, V: Traceable<T>, O: Clone + Operation<T>, Input: Parameterized<V>, Output: Parameterized<V>>
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
        Input::ParameterStructure: Clone,
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
        Output::ParameterStructure: Clone,
        Output::Family: ParameterizedFamily<Atom<T, V>>,
    {
        Output::To::<Atom<T, V>>::from_parameters(self.output_structure.clone(), self.outputs().cloned())
    }

    /// Returns a simplified version of this [`Program`] with dead constants and [`Instruction`]s that do not contribute
    /// to the [`Program`]'s output removed.
    pub fn simplified(&self) -> Result<Self, TracingError>
    where
        Input::ParameterStructure: Clone,
        Output::ParameterStructure: Clone,
    {
        /// Marks the [`Atom`] that corresponds to the provided `atom_id`, its producing [`Instruction`], and all
        /// transitive input [`Atom`]s as _live_ in the provided `atom_is_live` array.
        fn mark_atom_as_live<
            T: Type,
            V: Traceable<T>,
            O: Clone + Operation<T>,
            Input: Parameterized<V>,
            Output: Parameterized<V>,
        >(
            program: &Program<T, V, O, Input, Output>,
            atom_id: AtomId,
            atom_is_live: &mut [bool],
            parent_instructions: &[Option<usize>],
        ) {
            if !atom_is_live[atom_id.index] {
                atom_is_live[atom_id.index] = true;
                if let Some(instruction_index) = parent_instructions[atom_id.index] {
                    for input in program.instructions[instruction_index].inputs.iter().copied() {
                        mark_atom_as_live(program, input, atom_is_live, parent_instructions);
                    }
                }
            }
        }

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
            let atom = match atom {
                Atom::Constant(value) => program_builder.add_constant(value.clone()),
                Atom::Variable(_) => {
                    let instruction_index = parent_instructions[atom_id.index]
                        .ok_or(TracingError::MalformedProgram("variable atom has no owning instruction".to_string()))?;
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
                    *atom_id_mapping
                        .get(&atom_id)
                        .expect("remapped instruction outputs should populate the atom mapping")
                }
            };
            atom_id_mapping.insert(atom_id, atom);
            Ok(atom)
        }

        let mut parent_instructions = vec![None; self.atoms.len()];
        for (instruction_index, instruction) in self.instructions.iter().enumerate() {
            for output in instruction.outputs.iter().copied() {
                parent_instructions[output.index] = Some(instruction_index);
            }
        }

        let mut atom_is_live = vec![false; self.atoms.len()];
        for output in self.output_ids.iter().copied() {
            mark_atom_as_live(self, output, atom_is_live.as_mut_slice(), parent_instructions.as_slice());
        }

        let mut program_builder = ProgramBuilder::<T, V, O>::new();
        let mut atom_id_mapping = HashMap::new();
        for input_id in self.input_ids.iter().copied() {
            let input = self.atoms.get(input_id.index).ok_or(TracingError::UnboundAtomId { id: input_id })?;
            let Atom::Variable(r#type) = input else {
                return Err(TracingError::MalformedProgram("program input atom was not a variable".to_string()));
            };
            atom_id_mapping.insert(input_id, program_builder.add_input(r#type.clone()));
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

        Ok(program_builder.build(output_ids, self.input_structure.clone(), self.output_structure.clone()))
    }

    /// Interprets/executes this [`Program`] with the provided input. This is the main replay entry point for staged
    /// [`Program`]s. It checks that the provided input value matches the program's expected input structure, evaluates
    /// the [`Instruction`]s in order, and finally builds a structured output value from the computed output values.
    pub fn interpret(&self, input: Input) -> Result<Output, TracingError>
    where
        O: InterpretableOperation<T, V>,
        Input::ParameterStructure: Debug + PartialEq,
        Output::ParameterStructure: Clone,
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

        // Flatten the structured input so that we can get the input values aligned to the input atoms of the program.
        let inputs = input.into_parameters().collect::<Vec<_>>();
        if inputs.len() != self.input_ids.len() {
            return Err(TracingError::InvalidInputCount { expected: self.input_ids.len(), got: inputs.len() });
        }

        // Count every future consumer of each atom, including final program outputs. These counts let us move each
        // value out on its last use and clone it only when a later consumer still needs it.
        let mut remaining_uses = vec![0usize; self.atoms.len()];
        for instruction in self.instructions.iter() {
            for input_id in instruction.inputs.iter().copied() {
                let Some(remaining_uses) = remaining_uses.get_mut(input_id.index) else {
                    return Err(TracingError::UnboundAtomId { id: input_id });
                };
                *remaining_uses += 1;
            }
        }
        for output_id in self.output_ids.iter().copied() {
            let Some(remaining_uses) = remaining_uses.get_mut(output_id.index) else {
                return Err(TracingError::UnboundAtomId { id: output_id });
            };
            *remaining_uses += 1;
        }

        // Store concrete input values in a sparse value table indexed by [`AtomId`].
        let mut values = vec![None; self.atoms.len()];
        for (input_id, input) in self.input_ids.iter().copied().zip(inputs) {
            let Some(slot) = values.get_mut(input_id.index) else {
                return Err(TracingError::UnboundAtomId { id: input_id });
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
                values[atom_index] = Some(value.clone());
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

            // Interpret the operation using concrete values and ensure it produces the expected number of outputs.
            let outputs = instruction.operation.interpret(instruction_inputs.as_slice())?;
            if outputs.len() != instruction.outputs.len() {
                return Err(TracingError::InvalidOutputCount {
                    expected: instruction.outputs.len(),
                    got: outputs.len(),
                });
            }

            for (output_id, output) in instruction.outputs.iter().copied().zip(outputs) {
                let Some(value) = values.get_mut(output_id.index) else {
                    return Err(TracingError::UnboundAtomId { id: output_id });
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

        // Reshape the flat output values into the structured output type that this program is expected to produce.
        Ok(Output::from_parameters(self.output_structure.clone(), outputs)?)
    }
}

impl<
    T: Type + Display,
    V: Traceable<T>,
    O: Clone + Display + Operation<T>,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
> Display for Program<T, V, O, Input, Output>
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            formatter,
            "lambda {} .",
            self.input_ids
                .iter()
                .map(|input| format!("{input}:{}", self.atoms[input.index].r#type()))
                .collect::<Vec<_>>()
                .join(", "),
        )?;
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
                        writeln!(
                            formatter,
                            "{} {} = {}{}",
                            if binding_count == 0 { "let" } else { "   " },
                            instruction
                                .outputs
                                .iter()
                                .map(|output| format!("{output}:{}", self.atoms[output.index].r#type()))
                                .collect::<Vec<_>>()
                                .join(", "),
                            instruction.operation,
                            instruction.inputs.iter().map(|input| format!(" {input}")).collect::<Vec<_>>().join(""),
                        )?;
                        binding_count += 1;
                    };
                }
            }
        }
        write!(
            formatter,
            "in ({})",
            self.output_ids.iter().map(|output| output.to_string()).collect::<Vec<_>>().join(", "),
        )
    }
}

/// Builder for staged programs.
///
/// [`ProgramBuilder`] is the mutable workhorse used by the tracing entry points and by the
/// linearization helpers. It mirrors the final [`Program`] IR closely: variable atoms retain only
/// their abstract types, while concrete values are kept only for literal constants.
///
/// During traced execution the builder also carries the first staging failure encountered in that
/// tracing scope. This lets infallible operator syntax like `x + y` poison the shared trace and
/// stop recording new instructions even though the surrounding closure cannot immediately return
/// `Result`.
#[derive(Clone, Debug)]
pub struct ProgramBuilder<T: Type, V: Typed<T>, O: Clone + Operation<T>> {
    /// Atom table accumulated so far, including inputs, constants, and derived outputs.
    pub atoms: Vec<Atom<T, V>>,

    /// Input atom ids in parameter order.
    pub input_ids: Vec<AtomId>,

    /// Instructions recorded so far in execution order.
    pub instructions: Vec<Instruction<O>>,

    /// First staging failure recorded while this builder was used for traced execution.
    pub error: Option<TracingError>,
}

impl<T: Type, V: Traceable<T>, O: Clone + Operation<T>> ProgramBuilder<T, V, O> {
    /// Creates an empty builder.
    ///
    /// Fresh builders contain no atoms or instructions and are typically owned by one tracing
    /// scope.
    #[inline]
    pub fn new() -> Self {
        Self { atoms: Vec::new(), input_ids: Vec::new(), instructions: Vec::new(), error: None }
    }

    /// Adds a new input atom retaining only its abstract type.
    ///
    /// Intended for program transforms that rebuild structure without needing intermediate values
    /// (for example [`Program::simplified`]). Callers that later need a representative value for
    /// this atom should synthesize it from the retained input type through an
    /// [`Engine`](crate::tracing_v2::Engine).
    #[inline]
    pub fn add_input(&mut self, r#type: T) -> AtomId {
        let id = self.add_variable(r#type);
        self.input_ids.push(id);
        id
    }

    /// Adds a constant atom to the program.
    ///
    /// Constants are retained verbatim in the final [`Program`] so later replay and lowering can
    /// recover the literal value.
    #[inline]
    pub fn add_constant(&mut self, value: V) -> AtomId {
        let id = AtomId { index: self.atoms.len() };
        self.atoms.push(Atom::Constant(value));
        id
    }

    /// Adds a non-constant variable atom retaining only its abstract type.
    ///
    /// This is the common helper used by input staging and by instruction staging paths that need
    /// to materialize fresh variable outputs in the atom table.
    #[inline]
    pub fn add_variable(&mut self, r#type: T) -> AtomId {
        let id = AtomId { index: self.atoms.len() };
        self.atoms.push(Atom::Variable(r#type));
        id
    }

    /// Adds a staged instruction using abstract evaluation.
    ///
    /// This validates the input atoms through [`Operation::infer_output_types`] and stages one
    /// variable-output instruction.
    pub fn add_instruction(&mut self, operation: O, inputs: Vec<AtomId>) -> Result<Vec<AtomId>, TracingError> {
        let input_abstracts = inputs
            .iter()
            .map(|input| {
                self.atoms
                    .get(input.index)
                    .map(|atom| atom.r#type().into_owned())
                    .ok_or(TracingError::UnboundAtomId { id: *input })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let output_abstracts = operation.infer_output_types(input_abstracts.as_slice())?;

        let outputs = output_abstracts.into_iter().map(|r#type| self.add_variable(r#type)).collect::<Vec<_>>();
        self.instructions.push(Instruction { operation, inputs, outputs: outputs.clone() });
        Ok(outputs)
    }

    /// Finalizes the builder into a program with the given input/output structures.
    pub fn build<Input, Output>(
        self,
        outputs: Vec<AtomId>,
        input_structure: Input::ParameterStructure,
        output_structure: Output::ParameterStructure,
    ) -> Program<T, V, O, Input, Output>
    where
        Input: Parameterized<V>,
        Output: Parameterized<V>,
    {
        Program {
            atoms: self.atoms,
            input_ids: self.input_ids,
            instructions: self.instructions,
            output_ids: outputs,
            input_structure,
            output_structure,
            marker: PhantomData,
        }
    }
}

impl<T: Type, V: Traceable<T>, O: Clone + Operation<T>> Default for ProgramBuilder<T, V, O> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;

    use crate::{
        parameters::{ParameterError, Parameterized, Placeholder},
        tracing::TracingError,
        tracing_v2::{PrimitiveOperation, test_support},
        types::{ArrayType, DataType, Typed},
    };

    use super::*;

    #[test]
    fn atom_id_display_uses_percent_prefix() {
        assert_eq!(AtomId { index: 42 }.to_string(), "%42");
    }

    #[test]
    fn atom_as_constant_returns_only_literal_values() {
        let constant = Atom::<ArrayType, f64>::Constant(3.0);
        let variable = Atom::<ArrayType, f64>::Variable(ArrayType::scalar(DataType::F64));

        assert!(constant.is_constant());
        assert!(!constant.is_variable());
        assert_eq!(constant.as_constant(), Some(&3.0));

        assert!(!variable.is_constant());
        assert!(variable.is_variable());
        assert_eq!(variable.as_constant(), None);
    }

    #[test]
    fn program_builder_tracks_atom_kinds_and_executes() {
        let mut builder = ProgramBuilder::<ArrayType, f64, PrimitiveOperation<f64>>::new();
        let x = builder.add_input(2.0f64.r#type().into_owned());
        let y = builder.add_input(3.0f64.r#type().into_owned());
        let two = builder.add_constant(2.0f64);
        let scaled_x = builder.add_instruction(PrimitiveOperation::Scale { factor: 2.0 }, vec![x]).unwrap()[0];
        let sum = builder.add_instruction(PrimitiveOperation::Add, vec![scaled_x, y]).unwrap()[0];
        let program = builder.build::<(f64, f64), f64>(vec![sum], (Placeholder, Placeholder), Placeholder);

        assert!(matches!(program.atoms.get(x.index), Some(Atom::Variable(_))));
        assert!(matches!(program.atoms.get(two.index), Some(Atom::Constant(_))));
        assert_eq!(program.interpret((2.0, 3.0)).unwrap(), 7.0);
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = const
                    %3:f64[] = scale %0
                    %4:f64[] = add %3 %1
                in (%4)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn program_interpret_preserves_duplicate_outputs() {
        let mut builder = ProgramBuilder::<ArrayType, f64, PrimitiveOperation<f64>>::new();
        let x = builder.add_input(1.0f64.r#type().into_owned());
        let doubled = builder.add_instruction(PrimitiveOperation::Add, vec![x, x]).unwrap()[0];
        let program = builder.build::<f64, (f64, f64)>(vec![doubled, doubled], Placeholder, (Placeholder, Placeholder));

        assert_eq!(program.interpret(2.0f64).unwrap(), (4.0f64, 4.0f64));
    }

    #[test]
    fn program_reconstructs_structured_input_and_output_atoms() {
        let mut builder = ProgramBuilder::<ArrayType, f64, PrimitiveOperation<f64>>::new();
        let x = builder.add_input(1.0f64.r#type().into_owned());
        let y = builder.add_input(2.0f64.r#type().into_owned());
        let sum = builder.add_instruction(PrimitiveOperation::Add, vec![x, y]).unwrap()[0];
        let program = builder.build::<(f64, f64), (f64, f64)>(
            vec![x, sum],
            (Placeholder, Placeholder),
            (Placeholder, Placeholder),
        );

        let input = program.input().unwrap();
        assert!(matches!(input.0, Atom::Variable(r#type) if r#type == ArrayType::scalar(DataType::F64)));
        assert!(matches!(input.1, Atom::Variable(r#type) if r#type == ArrayType::scalar(DataType::F64)));

        let output = program.output().unwrap();
        assert!(matches!(output.0, Atom::Variable(r#type) if r#type == ArrayType::scalar(DataType::F64)));
        assert!(matches!(output.1, Atom::Variable(r#type) if r#type == ArrayType::scalar(DataType::F64)));
    }

    #[test]
    fn program_interpret_rejects_mismatched_parameter_structures() {
        let mut builder = ProgramBuilder::<ArrayType, f64, PrimitiveOperation<f64>>::new();
        let x = builder.add_input(1.0f64.r#type().into_owned());
        let program = builder.build::<Vec<f64>, f64>(vec![x], vec![Placeholder], Placeholder);

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
    fn program_display_uses_typed_jaxpr_like_rendering() {
        let mut builder = ProgramBuilder::<ArrayType, f64, PrimitiveOperation<f64>>::new();
        let x = builder.add_input(1.0f64.r#type().into_owned());
        let three = builder.add_constant(3.0f64);
        let sum = builder.add_instruction(PrimitiveOperation::Add, vec![x, three]).unwrap()[0];
        let program = builder.build::<f64, f64>(vec![sum], Placeholder, Placeholder);

        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = const
                    %2:f64[] = add %0 %1
                in (%2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn program_builder_rejects_unbound_inputs() {
        let mut builder = ProgramBuilder::<ArrayType, f64, PrimitiveOperation<f64>>::new();
        let result = builder.add_instruction(PrimitiveOperation::Add, vec![AtomId { index: 42 }, AtomId { index: 99 }]);
        assert!(matches!(
            result,
            Err(TracingError::UnboundAtomId { id }) if id == AtomId { index: 42 }
        ));
        test_support::assert_reference_program_rendering();
    }

    #[test]
    fn program_builder_tracks_only_types_for_variable_atoms() {
        let mut builder = ProgramBuilder::<ArrayType, f64, PrimitiveOperation<f64>>::new();
        let x = builder.add_input(2.0f64.r#type().into_owned());
        let three = builder.add_constant(3.0f64);
        let sum = builder.add_instruction(PrimitiveOperation::Add, vec![x, three]).unwrap()[0];

        assert!(
            matches!(builder.atoms.get(x.index), Some(Atom::Variable(r#type)) if *r#type == ArrayType::scalar(DataType::F64))
        );
        assert!(matches!(builder.atoms.get(three.index), Some(Atom::Constant(value)) if *value == 3.0));
        assert!(matches!(builder.atoms.get(sum.index), Some(Atom::Variable(_))));

        let program = builder.build::<f64, f64>(vec![sum], Placeholder, Placeholder);
        assert!(
            matches!(program.atoms.get(x.index), Some(Atom::Variable(r#type)) if *r#type == ArrayType::scalar(DataType::F64))
        );
        assert!(matches!(program.atoms.get(three.index), Some(Atom::Constant(value)) if *value == 3.0));
        assert!(matches!(program.atoms.get(sum.index), Some(Atom::Variable(_))));
        assert_eq!(program.interpret(4.0).unwrap(), 7.0);
    }
}
