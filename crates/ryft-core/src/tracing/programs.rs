use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;

use ryft_macros::Parameter;

use crate::parameters::{Parameter, ParameterError, Parameterized};
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
#[derive(Clone, Debug)]
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
        write!(formatter, "{}", self.index)
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

/// Executable staged program over an open operation set.
///
/// [`Program`] is the persistent artifact produced by tracing. It stores the finalized atom table,
/// the ordered list of instructions, and the structured input/output metadata needed to turn flat leaf
/// evaluation back into user-facing structured values. Both ordinary JIT traces and higher-order
/// transforms exchange programs in this form. The operation carrier remains generic so the same IR
/// can represent ordinary programs plus tangent and cotangent programs over backend-specific
/// operation carriers, but every carrier must implement the shape-level [`Operation`] interface for
/// the program's type domain.
pub struct Program<
    T: Type,
    V: Typed<T> + Parameter,
    O: Clone + Operation<T>,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
> {
    /// Final atom table of the staged program.
    pub atoms: Vec<Atom<T, V>>,

    /// Input atom ids in the same order as the flattened input parameters.
    pub input_ids: Vec<AtomId>,

    /// Output atom ids in the same order as the flattened output parameters.
    pub output_ids: Vec<AtomId>,

    /// Ordered instructions that replay the staged computation.
    pub instructions: Vec<Instruction<O>>,

    /// Structured input shape used to rebuild typed inputs from flat leaves.
    pub input_structure: Input::ParameterStructure,

    /// Structured output shape used to rebuild typed outputs from flat leaves.
    pub output_structure: Output::ParameterStructure,

    /// Phantom marker that ties the program to its structured input and output [`Parameterized`] families.
    pub marker: PhantomData<fn(Input) -> Output>,
}

impl<
    O: Clone + Operation<T>,
    T: Type,
    V: Traceable<T>,
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

impl<O: Clone + Operation<T>, T: Type, V: Traceable<T>, Input: Parameterized<V>, Output: Parameterized<V>>
    Program<T, V, O, Input, Output>
{
    /// Returns the program input atoms in parameter order.
    #[inline]
    pub fn inputs(&self) -> impl Iterator<Item = &Atom<T, V>> {
        self.input_ids.iter().map(|input_id| &self.atoms[input_id.index])
    }

    /// Returns the program output atoms in parameter order.
    #[inline]
    pub fn outputs(&self) -> impl Iterator<Item = &Atom<T, V>> {
        self.output_ids.iter().map(|output_id| &self.atoms[output_id.index])
    }

    /// Eliminates dead constants and instructions that do not contribute to the program outputs.
    pub fn simplified(&self) -> Result<Self, TracingError>
    where
        Input::ParameterStructure: Clone,
        Output::ParameterStructure: Clone,
    {
        fn mark_live<
            O: Clone + Operation<T>,
            T: Type,
            V: Traceable<T>,
            Input: Parameterized<V>,
            Output: Parameterized<V>,
        >(
            program: &Program<T, V, O, Input, Output>,
            atom_id: AtomId,
            live_atoms: &mut [bool],
            live_instructions: &mut [bool],
            instruction_by_output: &[Option<usize>],
        ) {
            if live_atoms[atom_id.index] {
                return;
            }
            live_atoms[atom_id.index] = true;
            if let Some(instruction_index) = instruction_by_output[atom_id.index] {
                if live_instructions[instruction_index] {
                    return;
                }
                live_instructions[instruction_index] = true;
                let instruction = &program.instructions[instruction_index];
                for input in instruction.inputs.iter().copied() {
                    mark_live(program, input, live_atoms, live_instructions, instruction_by_output);
                }
            }
        }

        fn remap_atom<
            O: Clone + Operation<T>,
            T: Type,
            V: Traceable<T>,
            Input: Parameterized<V>,
            Output: Parameterized<V>,
        >(
            atom_id: AtomId,
            program: &Program<T, V, O, Input, Output>,
            builder: &mut ProgramBuilder<T, V, O>,
            atom_mapping: &mut HashMap<AtomId, AtomId>,
            live_instructions: &[bool],
            instruction_by_output: &[Option<usize>],
        ) -> Result<AtomId, TracingError> {
            if let Some(mapped_atom) = atom_mapping.get(&atom_id) {
                return Ok(*mapped_atom);
            }

            let atom = program.atoms.get(atom_id.index).ok_or(TracingError::UnboundAtomId { id: atom_id })?;
            let mapped_atom = match atom {
                Atom::Constant(value) => builder.add_constant(value.clone()),
                Atom::Variable(_) => {
                    let instruction_index = instruction_by_output[atom_id.index]
                        .ok_or(TracingError::MalformedProgram("variable atom had no owning instruction".to_string()))?;
                    assert!(
                        live_instructions[instruction_index],
                        "attempted to remap a dead variable atom during structural program cleanup"
                    );
                    let instruction = &program.instructions[instruction_index];
                    let remapped_inputs = instruction
                        .inputs
                        .iter()
                        .copied()
                        .map(|input| {
                            remap_atom(input, program, builder, atom_mapping, live_instructions, instruction_by_output)
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let remapped_outputs = builder.add_instruction(instruction.operation.clone(), remapped_inputs)?;
                    if remapped_outputs.len() != instruction.outputs.len() {
                        return Err(TracingError::InvalidOutputCount {
                            expected: instruction.outputs.len(),
                            got: remapped_outputs.len(),
                        });
                    }
                    for (old_output, new_output) in
                        instruction.outputs.iter().copied().zip(remapped_outputs.iter().copied())
                    {
                        atom_mapping.insert(old_output, new_output);
                    }
                    *atom_mapping.get(&atom_id).expect("remapped instruction outputs should populate the atom mapping")
                }
            };
            atom_mapping.entry(atom_id).or_insert(mapped_atom);
            Ok(mapped_atom)
        }

        let mut instruction_by_output = vec![None; self.atoms.len()];
        for (instruction_index, instruction) in self.instructions.iter().enumerate() {
            for output in instruction.outputs.iter().copied() {
                instruction_by_output[output.index] = Some(instruction_index);
            }
        }

        let mut live_atoms = vec![false; self.atoms.len()];
        let mut live_instructions = vec![false; self.instructions.len()];
        for output in self.output_ids.iter().copied() {
            mark_live(
                self,
                output,
                live_atoms.as_mut_slice(),
                live_instructions.as_mut_slice(),
                instruction_by_output.as_slice(),
            );
        }

        let mut builder = ProgramBuilder::<T, V, O>::new();
        let mut atom_mapping = HashMap::new();
        for input_atom in self.input_ids.iter().copied() {
            let input = self.atoms.get(input_atom.index).ok_or(TracingError::UnboundAtomId { id: input_atom })?;
            let Atom::Variable(r#type) = input else {
                return Err(TracingError::MalformedProgram("program input atom was not a variable".to_string()));
            };
            let mapped = builder.add_input(r#type.clone());
            atom_mapping.insert(input_atom, mapped);
        }

        let outputs = self
            .output_ids
            .iter()
            .copied()
            .map(|output| {
                remap_atom(
                    output,
                    self,
                    &mut builder,
                    &mut atom_mapping,
                    live_instructions.as_slice(),
                    instruction_by_output.as_slice(),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(builder.build::<Input, Output>(outputs, self.input_structure.clone(), self.output_structure.clone()))
    }
}

impl<
    T: Type,
    V: Traceable<T>,
    O: Clone + InterpretableOperation<T, V>,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
> Program<T, V, O, Input, Output>
{
    /// Interprets the staged program on concrete input values.
    ///
    /// This is the user-facing replay entry point for staged programs. It checks that the incoming
    /// structured value matches the program's expected parameter structure, evaluates the flat IR,
    /// and then rebuilds the structured output.
    pub fn interpret(&self, input: Input) -> Result<Output, TracingError>
    where
        Input::ParameterStructure: Debug + PartialEq,
        Output::ParameterStructure: Clone,
    {
        let input_structure = input.parameter_structure();
        if input_structure != self.input_structure {
            return Err(ParameterError::MismatchedParameterStructures {
                left_structure: format!("{:?}", self.input_structure),
                right_structure: format!("{input_structure:?}"),
            }
            .into());
        }

        let input_values = input.into_parameters().collect::<Vec<_>>();
        if input_values.len() != self.input_ids.len() {
            return Err(TracingError::InvalidInputCount { expected: self.input_ids.len(), got: input_values.len() });
        }

        let mut remaining_uses = vec![0usize; self.atoms.len()];
        for instruction in self.instructions.iter() {
            for input in instruction.inputs.iter().copied() {
                let Some(use_count) = remaining_uses.get_mut(input.index) else {
                    return Err(TracingError::UnboundAtomId { id: input });
                };
                *use_count += 1;
            }
        }
        for output in self.output_ids.iter().copied() {
            let Some(use_count) = remaining_uses.get_mut(output.index) else {
                return Err(TracingError::UnboundAtomId { id: output });
            };
            *use_count += 1;
        }

        let mut values = vec![None; self.atoms.len()];
        for (atom, value) in self.input_ids.iter().copied().zip(input_values) {
            let Some(slot) = values.get_mut(atom.index) else {
                return Err(TracingError::UnboundAtomId { id: atom });
            };
            *slot = Some(value);
        }

        for (atom_index, atom) in self.atoms.iter().enumerate() {
            if remaining_uses[atom_index] == 0 {
                continue;
            }
            if let Atom::Constant(value) = atom {
                values[atom_index] = Some(value.clone());
            }
        }

        let max_input_count = self.instructions.iter().map(|instruction| instruction.inputs.len()).max().unwrap_or(0);
        let mut instruction_inputs = Vec::with_capacity(max_input_count);
        for instruction in self.instructions.iter() {
            instruction_inputs.clear();
            for input in instruction.inputs.iter().copied() {
                let Some(use_count) = remaining_uses.get_mut(input.index) else {
                    return Err(TracingError::UnboundAtomId { id: input });
                };
                if *use_count == 0 {
                    return Err(TracingError::MalformedProgram(
                        "instruction consumed an already-exhausted atom".to_string(),
                    ));
                }
                *use_count -= 1;

                let Some(slot) = values.get_mut(input.index) else {
                    return Err(TracingError::UnboundAtomId { id: input });
                };
                let value = if *use_count == 0 {
                    slot.take().ok_or(TracingError::UnboundAtomId { id: input })?
                } else {
                    slot.as_ref().ok_or(TracingError::UnboundAtomId { id: input })?.clone()
                };
                instruction_inputs.push(value);
            }

            let outputs = instruction.operation.interpret(instruction_inputs.as_slice())?;
            if outputs.len() != instruction.outputs.len() {
                return Err(TracingError::InvalidOutputCount {
                    expected: instruction.outputs.len(),
                    got: outputs.len(),
                });
            }

            for (atom, value) in instruction.outputs.iter().copied().zip(outputs) {
                let Some(slot) = values.get_mut(atom.index) else {
                    return Err(TracingError::UnboundAtomId { id: atom });
                };
                if remaining_uses[atom.index] != 0 {
                    *slot = Some(value);
                }
            }
        }

        let mut outputs = Vec::with_capacity(self.output_ids.len());
        for output in self.output_ids.iter().copied() {
            let Some(use_count) = remaining_uses.get_mut(output.index) else {
                return Err(TracingError::UnboundAtomId { id: output });
            };
            if *use_count == 0 {
                return Err(TracingError::MalformedProgram(
                    "program output consumed an already-exhausted atom".to_string(),
                ));
            }
            *use_count -= 1;

            let Some(slot) = values.get_mut(output.index) else {
                return Err(TracingError::UnboundAtomId { id: output });
            };
            let value = if *use_count == 0 {
                slot.take().ok_or(TracingError::UnboundAtomId { id: output })?
            } else {
                slot.as_ref().ok_or(TracingError::UnboundAtomId { id: output })?.clone()
            };
            outputs.push(value);
        }

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
        let format_atom = |id: AtomId| format!("%{id}");
        let format_typed_atom = |id: AtomId| format!("%{id}:{}", self.atoms[id.index].r#type());

        let inputs = self.input_ids.iter().map(|input| format_typed_atom(*input)).collect::<Vec<_>>().join(", ");
        writeln!(formatter, "lambda {inputs} .")?;

        let mut instruction_by_first_output = vec![None; self.atoms.len()];
        for (index, instruction) in self.instructions.iter().enumerate() {
            if let Some(first_output) = instruction.outputs.first() {
                instruction_by_first_output[first_output.index] = Some(index);
            }
        }

        let mut binding_count = 0usize;
        let mut input_atom_flags = vec![false; self.atoms.len()];
        for input_atom in self.input_ids.iter().copied() {
            input_atom_flags[input_atom.index] = true;
        }
        for (atom_id, atom) in self.atoms.iter().enumerate() {
            match atom {
                Atom::Constant(_) => {
                    let prefix = if binding_count == 0 { "let" } else { "   " };
                    writeln!(formatter, "{prefix} {} = const", format_typed_atom(AtomId { index: atom_id }))?;
                    binding_count += 1;
                }
                Atom::Variable(_) if input_atom_flags[atom_id] => {}
                Atom::Variable(_) => {
                    let Some(instruction_index) = instruction_by_first_output[atom_id] else {
                        continue;
                    };
                    let instruction = &self.instructions[instruction_index];
                    let outputs = instruction
                        .outputs
                        .iter()
                        .map(|output| format_typed_atom(*output))
                        .collect::<Vec<_>>()
                        .join(", ");
                    let inputs =
                        instruction.inputs.iter().map(|input| format_atom(*input)).collect::<Vec<_>>().join(" ");
                    let prefix = if binding_count == 0 { "let" } else { "   " };
                    if inputs.is_empty() {
                        writeln!(formatter, "{prefix} {outputs} = {}", instruction.operation)?;
                    } else {
                        writeln!(formatter, "{prefix} {outputs} = {} {inputs}", instruction.operation)?;
                    }
                    binding_count += 1;
                }
            }
        }

        let outputs = self.output_ids.iter().map(|output| format_atom(*output)).collect::<Vec<_>>().join(", ");
        write!(formatter, "in ({outputs})")
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
