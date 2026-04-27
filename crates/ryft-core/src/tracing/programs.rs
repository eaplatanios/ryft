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
/// tracing wrappers such as [`Tracer`](crate::Tracer). It ties each leaf to a type descriptor `T` via [`Typed`] and
/// requires [`Display`] so that constants and [`Operation`] metadata can render their carried values directly.
pub trait Traceable<T: Type>: Clone + Display + Parameter + Typed<T> {}

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

/// Maximum length for the contents of a bracketed section in an [`OperationFormatter`] that should be rendered inline.
/// If the length exceeds this value, then the section contents will be rendered over multiple lines.
const MAX_INLINE_OPERATION_SECTION_CONTENTS_LENGTH: usize = 80;

/// Helper for rendering [`Operation`]s that supports proper bracketing and indentation for operation metadata.
/// [`OperationFormatter`] centralizes the indentation and bracket layout used by higher-order or metadata-carrying
/// operations. The operation name is written immediately by [`OperationFormatter::new`], while
/// [`OperationFormatter::bracketed`] owns the bracketed metadata delimiters. Scalar fields are buffered so that short
/// metadata can render inline when no nested program fields are present, while nested program fields force multiline
/// rendering.
pub struct OperationFormatter<'f, 'a> {
    /// [`Formatter`](std::fmt::Formatter) receiving the rendered text.
    formatter: &'f mut std::fmt::Formatter<'a>,

    /// Indentation of the rendered [`Instruction`] line that owns the [`Operation`] that is being rendered.
    indentation: usize,

    /// Buffered scalar field name-value pairs that may be rendered inline if no nested [`Program`] fields are present.
    fields: Vec<(String, String)>,

    /// Boolean indicating whether this [`Operation`] being rendered has been forced to use multiple lines.
    is_multiline: bool,
}

impl<'f, 'a> OperationFormatter<'f, 'a> {
    /// Creates a new [`OperationFormatter`] and writes the provided [`Operation`] name.
    #[inline]
    pub fn new(
        formatter: &'f mut std::fmt::Formatter<'a>,
        indentation: usize,
        name: &'static str,
    ) -> Result<Self, std::fmt::Error> {
        write!(formatter, "{name}")?;
        Ok(Self { formatter, indentation, fields: Vec::new(), is_multiline: false })
    }

    /// Renders the provided field name-value pair.
    #[inline]
    pub fn field(&mut self, name: &str, value: impl Display) -> std::fmt::Result {
        if self.is_multiline {
            write!(self.formatter, "\n{:indentation$}{name}={value},", "", indentation = self.indentation + 4)
        } else {
            self.fields.push((name.to_string(), value.to_string()));
            Ok(())
        }
    }

    /// Renders the provided nested field name-[`Program`] pair. This must be used for [`Program`]-valued fields.
    #[inline]
    pub fn program<
        T: Type,
        V: Traceable<T>,
        O: Clone + Operation<T>,
        Input: Parameterized<V>,
        Output: Parameterized<V>,
    >(
        &mut self,
        name: &str,
        program: &Program<T, V, O, Input, Output>,
    ) -> std::fmt::Result {
        self.is_multiline = true;
        for (name, value) in std::mem::take(&mut self.fields) {
            write!(self.formatter, "\n{:indentation$}{name}={value},", "", indentation = self.indentation + 4)?;
        }
        writeln!(self.formatter)?;
        write!(self.formatter, "{:indentation$}", "", indentation = self.indentation + 4)?;
        writeln!(self.formatter, "{name}={{")?;
        program.render(self.formatter, self.indentation + 8)?;
        writeln!(self.formatter)?;
        write!(self.formatter, "{:indentation$}", "", indentation = self.indentation + 4)?;
        write!(self.formatter, "}},")
    }

    /// Renders a bracketed section (using square brackets) using the provided closure for rendering its contents.
    #[inline]
    pub fn bracketed(mut self, render_contents: impl FnOnce(&mut Self) -> std::fmt::Result) -> std::fmt::Result {
        write!(self.formatter, " [")?;
        render_contents(&mut self)?;
        let inline_contents_length = self
            .fields
            .iter()
            .enumerate()
            .map(|(index, (name, value))| name.len() + 1 + value.len() + if index == 0 { 0 } else { 2 })
            .sum::<usize>();
        if self.is_multiline || inline_contents_length > MAX_INLINE_OPERATION_SECTION_CONTENTS_LENGTH {
            self.is_multiline = true;
            for (name, value) in std::mem::take(&mut self.fields) {
                write!(self.formatter, "\n{:indentation$}{name}={value},", "", indentation = self.indentation + 4)?;
            }
            writeln!(self.formatter)?;
            write!(self.formatter, "{:indentation$}", "", indentation = self.indentation)?;
        } else {
            for (index, (name, value)) in self.fields.iter().enumerate() {
                if index > 0 {
                    write!(self.formatter, ", {name}={value}")?;
                } else {
                    write!(self.formatter, "{name}={value}")?;
                }
            }
        }
        write!(self.formatter, "]")
    }
}

/// [`Operation`] that can appear in [`Program`]s. [`Operation`] invocations are represented as [`Instruction`]s in
/// [`Program`]s. This trait represents the high-level operation interface that only requires operations to be able to
/// provide their name and to infer their output [`Type`]s given their input [`Type`]s.
pub trait Operation<T: Type>: Debug {
    /// Returns the name of this [`Operation`] that is used in diagnostics and when rendering [`Program`]s as strings.
    fn name(&self) -> &'static str;

    /// Infers the output [`Type`]s of this [`Operation`] from the provided input [`Type`]s without executing it.
    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError>;

    /// Renders this [`Operation`] as part of an [`Instruction`]. The default implementation simply renders
    /// [`Operation::name`]. Operations carrying semantic metadata or nested [`Program`]s should override this
    /// function and use [`OperationFormatter`] for consistent bracketed and indented formatting.
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        let _ = indentation;
        formatter.write_str(self.name())
    }
}

/// [`InterpretableOperation`]s are [`Operation`]s that can be interpreted (i.e., executed) given concrete input values.
pub trait InterpretableOperation<T: Type, V: Typed<T>>: Operation<T> {
    /// Interprets this [`Operation`] given the provided input values and returns the resulting output values.
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError>;
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
pub struct Program<
    T: Type,
    V: Typed<T> + Parameter,
    O: Clone + Operation<T>,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
> {
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
    marker: PhantomData<fn(Input) -> Output>,
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
        fn add_atom_to_simplified_program<T: Type, V: Traceable<T>, O: Clone + Operation<T>>(
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
            let atom = atoms.get_mut(atom_id.index).ok_or(TracingError::UnboundAtomId { id: atom_id })?.take();
            match atom {
                None => Err(TracingError::MalformedProgram(format!(
                    "atom {atom_id} was already moved while simplifying program",
                ))),
                Some(Atom::Constant(value)) => {
                    let new_atom = AtomId { index: new_atoms.len() };
                    new_atoms.push(Atom::Constant(value));
                    atom_id_mapping.insert(atom_id, new_atom);
                    Ok(new_atom)
                }
                Some(Atom::Variable(_)) => {
                    let instruction_index =
                        parent_instructions.get(atom_id.index).copied().flatten().ok_or(
                            TracingError::MalformedProgram("variable atom has no owning instruction".to_string()),
                        )?;
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
                        let output_atom = atoms
                            .get_mut(output.index)
                            .ok_or(TracingError::UnboundAtomId { id: output })?
                            .take()
                            .ok_or(TracingError::MalformedProgram(
                                "instruction output atom was already moved".to_string(),
                            ))?;
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
            }
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
pub struct ProgramBuilder<T: Type, V: Typed<T> + Parameter, O: Clone + Operation<T>> {
    /// [`Atom`]s contained in the [`Program`] that is being built, in the order in which they will be evaluated.
    pub atoms: Vec<Atom<T, V>>,

    /// [`AtomId`]s of the [`Atom`]s that correspond to the inputs (i.e., arguments) of the [`Program`] being built.
    pub input_ids: Vec<AtomId>,

    /// Ordered sequence of [`Instruction`]s that make up the computational graph of the [`Program`] being built.
    pub instructions: Vec<Instruction<O>>,

    /// Optional [`TracingError`] encountered during program construction that will be propagated via [`Self::build`].
    pub error: Option<TracingError>,
}

impl<T: Type, V: Traceable<T>, O: Clone + Operation<T>> ProgramBuilder<T, V, O> {
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
    use std::fmt::{Debug, Display};
    use std::rc::Rc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::parameters::{ParameterError, Parameterized, Placeholder};
    use crate::tracing::TracingError;
    use crate::tracing_v2::PrimitiveOperation;
    use crate::types::{ArrayType, DataType, TypeError};

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

    impl Operation<ArrayType> for LongMetadataOperation {
        fn name(&self) -> &'static str {
            "long_metadata"
        }

        fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
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
        let constant = Atom::<ArrayType, f64>::Constant(3.0);
        let variable = Atom::<ArrayType, f64>::Variable(ArrayType::scalar(DataType::F64));

        assert!(constant.is_constant());
        assert!(!constant.is_variable());
        assert_eq!(constant.as_constant(), Some(&3.0));
        assert_eq!(constant.r#type().into_owned(), ArrayType::scalar(DataType::F64));

        assert!(variable.is_variable());
        assert_eq!(variable.as_constant(), None);
        assert_eq!(variable.r#type().into_owned(), ArrayType::scalar(DataType::F64));
    }

    #[test]
    fn test_atom_id() {
        assert_eq!(AtomId { index: 42 }.to_string(), "%42");
    }

    #[test]
    fn test_program() {
        // Test simple program with one argument.
        let mut builder = ProgramBuilder::<ArrayType, f64, PrimitiveOperation<f64>>::new();
        let i0 = builder.add_input(ArrayType::scalar(DataType::F64));
        let c0 = builder.add_constant(3.0f64);
        let o0 = builder.add_instruction(PrimitiveOperation::Add, vec![i0, c0]).unwrap()[0];
        let program = builder.build::<f64, f64>(vec![o0], Placeholder, Placeholder).unwrap();
        let input = program.input().unwrap();
        let output = program.output().unwrap();
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
        assert!(matches!(input, Atom::Variable(r#type) if r#type == ArrayType::scalar(DataType::F64)));
        assert!(matches!(output, Atom::Variable(r#type) if r#type == ArrayType::scalar(DataType::F64)));

        // Test simple program with two arguments.
        let mut builder = ProgramBuilder::<ArrayType, f64, PrimitiveOperation<f64>>::new();
        let i0 = builder.add_input(ArrayType::scalar(DataType::F64));
        let i1 = builder.add_input(ArrayType::scalar(DataType::F64));
        let v0 = builder.add_instruction(PrimitiveOperation::Scale { factor: 2.0 }, vec![i0]).unwrap()[0];
        let o0 = builder.add_instruction(PrimitiveOperation::Add, vec![v0, i1]).unwrap()[0];
        let program = builder.build::<(f64, f64), f64>(vec![o0], (Placeholder, Placeholder), Placeholder).unwrap();
        let input = program.input().unwrap();
        let output = program.output().unwrap();
        assert_eq!(program.interpret((2.0, 3.0)), Ok(7.0));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = scale [factor=2] %0
                    %3:f64[] = add %2 %1
                in (%3)
            "}
            .trim_end(),
        );
        assert!(matches!(input.0, Atom::Variable(r#type) if r#type == ArrayType::scalar(DataType::F64)));
        assert!(matches!(input.1, Atom::Variable(r#type) if r#type == ArrayType::scalar(DataType::F64)));
        assert!(matches!(output, Atom::Variable(r#type) if r#type == ArrayType::scalar(DataType::F64)));

        // Test a program that contains an operation with long metadata that should be rendered on multiple lines.
        let mut builder = ProgramBuilder::<ArrayType, f64, LongMetadataOperation>::new();
        let i0 = builder.add_input(ArrayType::scalar(DataType::F64));
        let o0 = builder.add_instruction(LongMetadataOperation, vec![i0]).unwrap()[0];
        let program = builder.build::<f64, f64>(vec![o0], Placeholder, Placeholder).unwrap();
        let input = program.input().unwrap();
        let output = program.output().unwrap();
        assert_eq!(
            program.to_string(),
            format!(
                indoc! {"
                    lambda %0:f64[] .
                    let %1:f64[] = long_metadata [
                        value={metadata_value},
                    ] %0
                    in (%1)
                "},
                metadata_value = LongMetadataOperation::METADATA_VALUE,
            )
            .trim_end()
        );
        assert!(matches!(input, Atom::Variable(r#type) if r#type == ArrayType::scalar(DataType::F64)));
        assert!(matches!(output, Atom::Variable(r#type) if r#type == ArrayType::scalar(DataType::F64)));

        // Test a program with two outputs that are copies of the same value.
        let mut builder = ProgramBuilder::<ArrayType, f32, PrimitiveOperation<f32>>::new();
        let i0 = builder.add_input(ArrayType::scalar(DataType::F32));
        let o0 = builder.add_instruction(PrimitiveOperation::Add, vec![i0, i0]).unwrap()[0];
        let program = builder.build::<f32, (f32, f32)>(vec![o0, o0], Placeholder, (Placeholder, Placeholder)).unwrap();
        let input = program.input().unwrap();
        let output = program.output().unwrap();
        assert_eq!(program.interpret(2.0f32), Ok((4.0f32, 4.0f32)));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:f32[] = add %0 %0
                in (%1, %1)
            "}
            .trim_end(),
        );
        assert!(matches!(input, Atom::Variable(r#type) if r#type == ArrayType::scalar(DataType::F32)));
        assert!(matches!(output.0, Atom::Variable(r#type) if r#type == ArrayType::scalar(DataType::F32)));
        assert!(matches!(output.1, Atom::Variable(r#type) if r#type == ArrayType::scalar(DataType::F32)));
    }

    #[test]
    fn test_program_interpret_lifts_live_constants_once() {
        let mut builder = ProgramBuilder::<ArrayType, f64, PrimitiveOperation<f64>>::new();
        let i0 = builder.add_input(ArrayType::scalar(DataType::F64));
        let c0 = builder.add_constant(7.0f64);
        let c1 = builder.add_constant(3.0f64);
        let o0 = builder.add_instruction(PrimitiveOperation::Add, vec![i0, c1]).unwrap()[0];
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
        let mut builder = ProgramBuilder::<ArrayType, f64, PrimitiveOperation<f64>>::new();
        let i0 = builder.add_input(ArrayType::scalar(DataType::F64));
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
        let mut builder = ProgramBuilder::<ArrayType, f64, PrimitiveOperation<f64>>::new();
        let i0 = builder.add_input(ArrayType::scalar(DataType::F64));
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
        let mut builder = ProgramBuilder::<ArrayType, f64, PrimitiveOperation<f64>>::new();
        let i0 = builder.add_input(ArrayType::scalar(DataType::F64));
        let o0 = builder.add_instruction(PrimitiveOperation::Scale { factor: 2.0 }, vec![i0]).unwrap()[0];
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
        let mut builder = ProgramBuilder::<ArrayType, f64, PrimitiveOperation<f64>>::new();
        let i0 = builder.add_input(ArrayType::scalar(DataType::F64));
        let c0 = builder.add_constant(2.0f64);
        let c1 = builder.add_constant(3.0f64);
        let _ = builder.add_instruction(PrimitiveOperation::Add, vec![i0, c0]).unwrap()[0];
        let v1 = builder.add_instruction(PrimitiveOperation::Add, vec![i0, c1]).unwrap()[0];
        let program = builder.build::<f64, (f64, f64)>(vec![v1, v1], Placeholder, (Placeholder, Placeholder)).unwrap();
        let simplified = program.simplified().unwrap();

        assert_eq!(c0, AtomId { index: 1 });
        assert_eq!(simplified.interpret(2.0f64), Ok((5.0f64, 5.0f64)));
        assert_eq!(
            simplified.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = const
                    %2:f64[] = add %0 %1
                in (%2, %2)
            "}
            .trim_end(),
        );
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = const
                    %2:f64[] = const
                    %3:f64[] = add %0 %1
                    %4:f64[] = add %0 %2
                in (%4, %4)
            "}
            .trim_end(),
        );
    }

    // TODO(eaplatanios): Review this.
    #[test]
    fn test_program_into_simplified() {
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

        impl crate::parameters::Parameter for CloneCountingValue {}

        impl crate::types::Typed<ArrayType> for CloneCountingValue {
            fn r#type(&self) -> Cow<'_, ArrayType> {
                Cow::Owned(ArrayType::scalar(DataType::F64))
            }
        }

        impl Traceable<ArrayType> for CloneCountingValue {}

        struct CloneCountingOperation {
            clone_count: Rc<Cell<usize>>,
        }

        impl CloneCountingOperation {
            fn new(clone_count: Rc<Cell<usize>>) -> Self {
                Self { clone_count }
            }
        }

        impl Clone for CloneCountingOperation {
            fn clone(&self) -> Self {
                self.clone_count.set(self.clone_count.get() + 1);
                Self { clone_count: Rc::clone(&self.clone_count) }
            }
        }

        impl Debug for CloneCountingOperation {
            fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.debug_struct("CloneCountingOperation").finish_non_exhaustive()
            }
        }

        impl Operation<ArrayType> for CloneCountingOperation {
            fn name(&self) -> &'static str {
                "clone_counting"
            }

            fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
                if input_types.len() != 2 {
                    return Err(TypeError { message: format!("expected 2 input types but got {}", input_types.len()) });
                }
                Ok(vec![input_types[0].clone()])
            }
        }

        let value_clone_count = Rc::new(Cell::new(0));
        let operation_clone_count = Rc::new(Cell::new(0));
        let mut builder = ProgramBuilder::<ArrayType, CloneCountingValue, CloneCountingOperation>::new();
        let i0 = builder.add_input(ArrayType::scalar(DataType::F64));
        let c0 = builder.add_constant(CloneCountingValue::new(2.0, Rc::clone(&value_clone_count)));
        let c1 = builder.add_constant(CloneCountingValue::new(3.0, Rc::clone(&value_clone_count)));
        let v0 = builder
            .add_instruction(CloneCountingOperation::new(Rc::clone(&operation_clone_count)), vec![i0, c0])
            .unwrap()[0];
        let v1 = builder
            .add_instruction(CloneCountingOperation::new(Rc::clone(&operation_clone_count)), vec![i0, c1])
            .unwrap()[0];
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
        assert_eq!(operation_clone_count.get(), 0);

        let simplified = program.into_simplified().unwrap();

        assert_eq!(value_clone_count.get(), 0);
        assert_eq!(operation_clone_count.get(), 0);
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
                lambda %0:f64[] .
                let %1:f64[] = const
                    %2:f64[] = clone_counting %0 %1
                in (%2, %2)
            "}
            .trim_end(),
        );
    }

    // TODO(eaplatanios): Review this.
    #[test]
    fn test_program_simplified_rejects_variable_output_without_parent_instruction() {
        let mut builder = ProgramBuilder::<ArrayType, f64, PrimitiveOperation<f64>>::new();
        builder.add_input(ArrayType::scalar(DataType::F64));
        let orphaned_variable = builder.add_variable(ArrayType::scalar(DataType::F64));
        let program = builder.build::<f64, f64>(vec![orphaned_variable], Placeholder, Placeholder).unwrap();

        assert!(matches!(
            program.simplified(),
            Err(TracingError::MalformedProgram(message)) if message == "variable atom has no owning instruction",
        ));
    }

    // TODO(eaplatanios): Review this.
    #[test]
    fn test_program_builder() {
        let mut builder = ProgramBuilder::<ArrayType, f64, PrimitiveOperation<f64>>::new();
        let left = builder.add_input(ArrayType::scalar(DataType::F64));
        let right = builder.add_input(ArrayType::scalar(DataType::F64));
        let constant = builder.add_constant(2.0f64);
        let scaled_left = builder.add_instruction(PrimitiveOperation::Scale { factor: 2.0 }, vec![left]).unwrap()[0];
        let sum = builder.add_instruction(PrimitiveOperation::Add, vec![scaled_left, right]).unwrap()[0];

        assert_eq!(builder.input_ids, vec![left, right]);
        assert!(matches!(
            builder.atoms.get(left.index),
            Some(Atom::Variable(r#type)) if *r#type == ArrayType::scalar(DataType::F64)
        ));
        assert!(matches!(
            builder.atoms.get(right.index),
            Some(Atom::Variable(r#type)) if *r#type == ArrayType::scalar(DataType::F64)
        ));
        assert!(matches!(builder.atoms.get(constant.index), Some(Atom::Constant(value)) if *value == 2.0));
        assert!(matches!(
            builder.atoms.get(scaled_left.index),
            Some(Atom::Variable(r#type)) if *r#type == ArrayType::scalar(DataType::F64)
        ));
        assert!(matches!(
            builder.atoms.get(sum.index),
            Some(Atom::Variable(r#type)) if *r#type == ArrayType::scalar(DataType::F64)
        ));
        assert_eq!(builder.instructions.len(), 2);
        assert_eq!(builder.instructions[0].inputs, vec![left]);
        assert_eq!(builder.instructions[0].outputs, vec![scaled_left]);
        assert_eq!(builder.instructions[1].inputs, vec![scaled_left, right]);
        assert_eq!(builder.instructions[1].outputs, vec![sum]);
    }

    // TODO(eaplatanios): Review this.
    #[test]
    fn test_program_builder_rejects_unbound_instruction_inputs() {
        let mut builder = ProgramBuilder::<ArrayType, f64, PrimitiveOperation<f64>>::new();
        let result = builder.add_instruction(PrimitiveOperation::Add, vec![AtomId { index: 42 }, AtomId { index: 99 }]);

        assert!(matches!(
            result,
            Err(TracingError::UnboundAtomId { id }) if id == AtomId { index: 42 }
        ));
    }

    // TODO(eaplatanios): Review this.
    #[test]
    fn test_program_builder_build_returns_stored_error() {
        let mut builder = ProgramBuilder::<ArrayType, f64, PrimitiveOperation<f64>>::new();
        builder.error = Some(TracingError::InvalidInputCount { expected: 1, got: 0 });

        assert!(matches!(
            builder.build::<f64, f64>(Vec::new(), Placeholder, Placeholder),
            Err(TracingError::InvalidInputCount { expected: 1, got: 0 }),
        ));
    }

    // TODO(eaplatanios): Review this.
    #[test]
    fn test_program_builder_build_rejects_input_structure_count_mismatch() {
        let builder = ProgramBuilder::<ArrayType, f64, PrimitiveOperation<f64>>::new();

        assert!(matches!(
            builder.build::<f64, ()>(Vec::new(), Placeholder, ()),
            Err(TracingError::InvalidInputCount { expected: 1, got: 0 }),
        ));
    }

    // TODO(eaplatanios): Review this.
    #[test]
    fn test_program_builder_build_rejects_output_structure_count_mismatch() {
        let mut builder = ProgramBuilder::<ArrayType, f64, PrimitiveOperation<f64>>::new();
        builder.add_input(ArrayType::scalar(DataType::F64));

        assert!(matches!(
            builder.build::<f64, f64>(Vec::new(), Placeholder, Placeholder),
            Err(TracingError::InvalidOutputCount { expected: 1, got: 0 }),
        ));
    }

    // TODO(eaplatanios): Review this.
    #[test]
    fn test_program_builder_build_succeeds_with_matching_parameter_counts() {
        let mut builder = ProgramBuilder::<ArrayType, f64, PrimitiveOperation<f64>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let program = builder.build::<f64, f64>(vec![input], Placeholder, Placeholder).unwrap();

        assert_eq!(program.input_ids, vec![input]);
        assert_eq!(program.output_ids, vec![input]);
        assert_eq!(program.instructions.len(), 0);
        assert_eq!(program.interpret(4.0f64), Ok(4.0f64));
    }
}
