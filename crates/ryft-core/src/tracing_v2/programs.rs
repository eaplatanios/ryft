//! Shared staged-program representation and default op-carrier aliases used by the tracing
//! transforms.
//!
//! This module owns the IR that all of `tracing_v2` is built on top of. Whether we are capturing a
//! user closure with [`trace`](crate::tracing_v2::trace), replaying a staged program with
//! [`Program::interpret`], or linearizing a primal program into a [`LinearProgram`](crate::tracing_v2::LinearProgram),
//! we keep coming back to the same core pieces:
//!
//! - [`Atom`] values describe the leaves of the staged program.
//! - [`Instruction`] values connect atoms through concrete operation objects.
//! - [`ProgramBuilder`] incrementally stages those instructions while optionally retaining eager
//!   exemplars for simplification and validation.
//! - [`Program`] is the persistent, executable artifact shared between JIT tracing, linearization,
//!   transposition, and backend lowering.
//!
//! The generic parameters stay intentionally open so that the same IR can represent both ordinary
//! programs and tangent/cotangent programs over backend-specific op carriers.

use std::{borrow::Cow, collections::HashMap, fmt::Display, marker::PhantomData};

use ryft_macros::Parameter;

use crate::{
    parameters::{Parameter, Parameterized},
    tracing_v2::{InterpretableOp, Op, Traceable, TracingError},
    types::{Type, Typed},
};

/// Staged atom carrying abstract metadata.
///
/// The variant encodes whether the atom is a retained literal constant or an ordinary program
/// variable. Input-vs-derived provenance for variable atoms lives in the owning [`Program`]'s
/// [`Program::input_ids`] list and instruction outputs rather than in the atom enum itself.
#[derive(Clone, Debug)]
pub enum Atom<T: Type, V: Typed<T>> {
    /// Literal constant folded or supplied at trace time. Constants retain their value so the
    /// interpreter and MLIR lowering can emit them.
    Constant(V),

    /// Non-constant program variable carrying only its abstract type. Any builder-time exemplar
    /// lives in the owning [`ProgramBuilder`]'s side table and is discarded when the program is
    /// finalized.
    Variable(T),
}

impl<T: Type, V: Typed<T>> Typed<T> for Atom<T, V> {
    fn r#type(&self) -> Cow<'_, T> {
        match self {
            Self::Constant(value) => value.r#type(),
            Self::Variable(r#type) => Cow::Borrowed(r#type),
        }
    }
}

/// Identifier for an atom within a staged program.
///
/// Atom identifiers are stable indexes into a program's atom table. Instructions refer to their
/// inputs and outputs by these ids, which keeps the staged IR compact and easy to clone.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash, Parameter)]
pub struct AtomId {
    /// Zero-based index of this atom inside the owning [`Program`]'s atom table.
    pub index: usize,
}

impl Display for AtomId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.index)
    }
}

/// Single instruction in a staged program.
///
/// An [`Instruction`] is the IR-level record of one primitive application. It names the operation
/// object to apply, the input atoms it consumes, and the output atoms it defines. Programs are
/// purely dataflow-based, so instructions execute in list order with no control-flow nodes.
#[derive(Clone, Debug)]
pub struct Instruction<O> {
    /// Operation applied by this instruction.
    pub operation: O,

    /// Input atoms consumed by the instruction.
    pub inputs: Vec<AtomId>,

    /// Output atoms produced by the instruction.
    pub outputs: Vec<AtomId>,
}

/// Executable staged program over an open operation set.
///
/// [`Program`] is the persistent artifact produced by tracing. It stores the finalized atom table,
/// the ordered list of instructions, and the structured input/output metadata needed to turn flat leaf
/// evaluation back into user-facing structured values. Both ordinary JIT traces and higher-order
/// transforms exchange programs in this form.
pub struct Program<T: Type, V: Typed<T> + Parameter, O, Input: Parameterized<V>, Output: Parameterized<V>> {
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
    O: Clone,
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

impl<O: Clone, T: Type, V: Traceable<T>, Input: Parameterized<V>, Output: Parameterized<V>>
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

    /// Interprets the staged program on concrete input values.
    ///
    /// This is the user-facing replay entry point for staged programs. It checks that the incoming
    /// structured value matches the program's expected parameter structure, evaluates the flat IR,
    /// and then rebuilds the structured output.
    pub fn interpret(&self, input: Input) -> Result<Output, TracingError>
    where
        O: InterpretableOp<T, V>,
        Input::ParameterStructure: PartialEq,
        Output::ParameterStructure: Clone,
    {
        if input.parameter_structure() != self.input_structure {
            return Err(TracingError::MismatchedParameterStructure);
        }

        let input_values = input.into_parameters().collect::<Vec<_>>();
        if input_values.len() != self.input_ids.len() {
            return Err(TracingError::InvalidInputCount { expected: self.input_ids.len(), got: input_values.len() });
        }

        let mut values = vec![None; self.atoms.len()];
        for (atom, value) in self.input_ids.iter().copied().zip(input_values) {
            values[atom.index] = Some(value);
        }

        for (atom_index, atom) in self.atoms.iter().enumerate() {
            if let Atom::Constant(value) = atom {
                values[atom_index] = Some(value.clone());
            }
        }

        for instruction in &self.instructions {
            let inputs = instruction
                .inputs
                .iter()
                .map(|input| values[input.index].clone().ok_or(TracingError::UnboundAtomId { id: *input }))
                .collect::<Result<Vec<_>, _>>()?;
            let outputs = instruction.operation.interpret(inputs.as_slice())?;
            if outputs.len() != instruction.outputs.len() {
                return Err(TracingError::InvalidOutputCount {
                    expected: instruction.outputs.len(),
                    got: outputs.len(),
                });
            }

            for (atom, value) in instruction.outputs.iter().copied().zip(outputs) {
                values[atom.index] = Some(value);
            }
        }

        let values = values
            .into_iter()
            .enumerate()
            .map(|(atom_index, value)| value.ok_or(TracingError::UnboundAtomId { id: AtomId { index: atom_index } }))
            .collect::<Result<Vec<_>, _>>()?;
        let outputs = self.output_ids.iter().map(|output| values[output.index].clone()).collect::<Vec<_>>();
        Ok(Output::from_parameters(self.output_structure.clone(), outputs)?)
    }

    /// Eliminates dead constants and instructions that do not contribute to the program outputs.
    pub fn simplify(&self) -> Result<Self, TracingError>
    where
        O: Op<T>,
        Input::ParameterStructure: Clone,
        Output::ParameterStructure: Clone,
    {
        fn mark_live<O: Clone, T: Type, V: Traceable<T>, Input: Parameterized<V>, Output: Parameterized<V>>(
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

        fn remap_atom<O, T, V, Input, Output>(
            atom_id: AtomId,
            program: &Program<T, V, O, Input, Output>,
            builder: &mut ProgramBuilder<O, T, V>,
            atom_mapping: &mut HashMap<AtomId, AtomId>,
            live_instructions: &[bool],
            instruction_by_output: &[Option<usize>],
        ) -> Result<AtomId, TracingError>
        where
            O: Clone + Op<T>,
            T: Type,
            V: Traceable<T>,
            Input: Parameterized<V>,
            Output: Parameterized<V>,
        {
            if let Some(mapped_atom) = atom_mapping.get(&atom_id) {
                return Ok(*mapped_atom);
            }

            let atom = program.atoms.get(atom_id.index).ok_or(TracingError::UnboundAtomId { id: atom_id })?;
            let mapped_atom = match atom {
                Atom::Constant(value) => builder.add_constant(value.clone()),
                Atom::Variable(_) => {
                    let instruction_index = instruction_by_output[atom_id.index]
                        .ok_or(TracingError::InternalInvariantViolation("variable atom had no owning instruction"))?;
                    if !live_instructions[instruction_index] {
                        return Err(TracingError::InternalInvariantViolation(
                            "attempted to remap a dead variable atom during program simplification",
                        ));
                    }
                    let instruction = &program.instructions[instruction_index];
                    let remapped_inputs = instruction
                        .inputs
                        .iter()
                        .copied()
                        .map(|input| {
                            remap_atom(input, program, builder, atom_mapping, live_instructions, instruction_by_output)
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let output_abstracts = instruction
                        .outputs
                        .iter()
                        .map(|output| program.atoms[output.index].r#type().into_owned())
                        .collect::<Vec<_>>();
                    let remapped_outputs = builder.add_instruction_prevalidated(
                        instruction.operation.clone(),
                        remapped_inputs,
                        output_abstracts,
                    );
                    for (old_output, new_output) in
                        instruction.outputs.iter().copied().zip(remapped_outputs.iter().copied())
                    {
                        atom_mapping.insert(old_output, new_output);
                    }
                    *atom_mapping
                        .get(&atom_id)
                        .ok_or(TracingError::InternalInvariantViolation("failed to record remapped program outputs"))?
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

        let mut builder = ProgramBuilder::<O, T, V>::new();
        let mut atom_mapping = HashMap::new();
        for input_atom in self.input_ids.iter().copied() {
            let input = self.atoms.get(input_atom.index).ok_or(TracingError::UnboundAtomId { id: input_atom })?;
            let Atom::Variable(r#type) = input else {
                return Err(TracingError::InternalInvariantViolation(
                    "staged program input atom did not retain a type",
                ));
            };
            let mapped = builder.add_input_abstract(r#type.clone());
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

impl<O: Clone + Display, T: Type + Display, V: Traceable<T>, Input: Parameterized<V>, Output: Parameterized<V>> Display
    for Program<T, V, O, Input, Output>
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
/// linearization helpers. It is deliberately more stateful than [`Program`]: while staging, it can
/// retain eager exemplar values for variable atoms so that primitive applications can be
/// validated immediately, constants can be folded, and algebraic identities can be removed before
/// they ever enter the final IR.
///
/// The builder keeps one entry in [`Self::intermediates`] for every atom: `Some` for
/// non-constant atoms whose value has been eagerly computed during staging, `None` otherwise.
/// Those intermediates are an implementation detail of tracing and are discarded when the builder
/// is finalized via [`Self::build`].
///
/// During traced execution the builder also carries the first staging failure encountered in that
/// tracing scope. This lets infallible operator syntax like `x + y` poison the shared trace and
/// stop recording new instructions even though the surrounding closure cannot immediately return
/// `Result`.
#[derive(Clone, Debug)]
pub struct ProgramBuilder<O, T: Type, V: Typed<T>> {
    /// Atom table accumulated so far, including inputs, constants, and derived outputs.
    atoms: Vec<Atom<T, V>>,

    /// Optional eager exemplars retained for non-constant atoms while staging.
    intermediates: Vec<Option<V>>,

    /// Input atom ids in parameter order.
    input_ids: Vec<AtomId>,

    /// Instructions recorded so far in execution order.
    instructions: Vec<Instruction<O>>,

    /// First staging failure recorded while this builder was used for traced execution.
    error: Option<TracingError>,
}

impl<O: Clone, T: Type, V: Traceable<T>> ProgramBuilder<O, T, V> {
    /// Creates an empty builder.
    ///
    /// Fresh builders contain no atoms, instructions, or retained exemplars and are typically owned
    /// by one tracing scope.
    #[inline]
    pub fn new() -> Self {
        Self {
            atoms: Vec::new(),
            intermediates: Vec::new(),
            input_ids: Vec::new(),
            instructions: Vec::new(),
            error: None,
        }
    }

    /// Returns the atom with the provided identifier.
    #[inline]
    pub fn atom(&self, id: AtomId) -> Option<&Atom<T, V>> {
        self.atoms.get(id.index)
    }

    /// Returns the concrete value associated with the provided atom, if one is available.
    ///
    /// For [`Atom::Constant`] this returns the retained value. For [`Atom::Variable`] this returns
    /// the eagerly computed exemplar stored in the builder's
    /// side table, or `None` if none is available.
    #[inline]
    pub(crate) fn stored_value(&self, id: AtomId) -> Option<&V> {
        match self.atoms.get(id.index)? {
            Atom::Constant(value) => Some(value),
            Atom::Variable(_) => self.intermediates.get(id.index).and_then(Option::as_ref),
        }
    }

    /// Adds a new input atom retaining only its abstract type, without recording any exemplar in
    /// the builder's side table.
    ///
    /// Intended for program transforms that rebuild structure without needing intermediate values
    /// (for example [`Program::simplify`]). Callers that later need a representative value for this
    /// atom should synthesize it from the retained input type through an
    /// [`Engine`](crate::tracing_v2::Engine).
    #[inline]
    pub fn add_input_abstract(&mut self, abstract_value: T) -> AtomId {
        let id = AtomId { index: self.atoms.len() };
        self.atoms.push(Atom::Variable(abstract_value));
        self.intermediates.push(None);
        self.input_ids.push(id);
        id
    }

    /// Adds a new input atom using the abstract type and value of `example`.
    #[inline]
    pub fn add_input(&mut self, example: &V) -> AtomId {
        let abstract_value = <V as Typed<T>>::r#type(example).into_owned();
        self.add_input_with_example(abstract_value, example.clone())
    }

    /// Adds a new input atom with the supplied abstract type and a caller-supplied exemplar value.
    #[inline]
    fn add_input_with_example(&mut self, abstract_value: T, example_value: V) -> AtomId {
        let id = AtomId { index: self.atoms.len() };
        self.atoms.push(Atom::Variable(abstract_value));
        self.intermediates.push(Some(example_value));
        self.input_ids.push(id);
        id
    }

    /// Adds a constant atom to the program.
    ///
    /// Constants are retained verbatim in the final [`Program`] so later replay, lowering, and
    /// simplification passes can recover the literal value.
    #[inline]
    pub fn add_constant(&mut self, value: V) -> AtomId {
        let id = AtomId { index: self.atoms.len() };
        self.atoms.push(Atom::Constant(value));
        self.intermediates.push(None);
        id
    }

    /// Adds a staged instruction without running abstract or concrete evaluation.
    ///
    /// This is intended for linear program construction where the output types are already known.
    pub fn add_instruction_prevalidated(
        &mut self,
        operation: O,
        inputs: Vec<AtomId>,
        output_abstracts: Vec<T>,
    ) -> Vec<AtomId> {
        let outputs = output_abstracts
            .into_iter()
            .map(|r#type| {
                let id = AtomId { index: self.atoms.len() };
                self.atoms.push(Atom::Variable(r#type));
                self.intermediates.push(None);
                id
            })
            .collect::<Vec<_>>();
        self.instructions.push(Instruction { operation, inputs, outputs: outputs.clone() });
        outputs
    }

    /// Returns the number of instructions added so far.
    #[inline]
    pub fn instruction_count(&self) -> usize {
        self.instructions.len()
    }

    /// Returns `true` when traced execution has already recorded a staging failure on this builder.
    #[inline]
    pub(crate) fn has_error(&self) -> bool {
        self.error.is_some()
    }

    /// Records the first staging failure encountered by traced execution on this builder.
    #[inline]
    pub(crate) fn record_error_if_absent(&mut self, error: TracingError) {
        if self.error.is_none() {
            self.error = Some(error);
        }
    }

    /// Removes and returns the first staging failure recorded on this builder, if any.
    #[inline]
    pub(crate) fn take_error(&mut self) -> Option<TracingError> {
        self.error.take()
    }

    /// Adds a staged instruction using pre-computed output values, performing abstract-eval validation,
    /// algebraic identity elimination, and constant folding.
    ///
    /// Unlike [`add_instruction`](Self::add_instruction), this method does not call [`InterpretableOp::eval`].
    /// the caller supplies the concrete output values directly. Use this when the caller has already
    /// computed the outputs (e.g., inside [`Tracer`](crate::tracing_v2::Tracer) staging
    /// methods).
    pub fn add_instruction_with_output_values(
        &mut self,
        operation: O,
        inputs: Vec<AtomId>,
        output_values: Vec<V>,
    ) -> Result<Vec<AtomId>, TracingError>
    where
        O: Op<T>,
    {
        let input_abstracts = inputs
            .iter()
            .map(|input| {
                self.atom(*input)
                    .map(|atom| atom.r#type().into_owned())
                    .ok_or(TracingError::UnboundAtomId { id: *input })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let output_abstracts = operation.abstract_eval(input_abstracts.as_slice())?;

        // Algebraic identity elimination: eliminate trivial ops like scale-by-1, add-by-0, mul-by-1.
        let is_zero = |id: AtomId| matches!(self.atom(id), Some(Atom::Constant(value)) if value.is_zero());
        let is_one = |id: AtomId| matches!(self.atom(id), Some(Atom::Constant(value)) if value.is_one());
        if let Some(simplified) = operation.try_simplify(&inputs, &is_zero, &is_one) {
            return Ok(simplified);
        }

        let all_constant = inputs.iter().all(|input| matches!(self.atom(*input), Some(Atom::Constant(_))));

        let outputs = output_abstracts
            .into_iter()
            .zip(output_values)
            .map(|(r#type, output_value)| {
                let id = self.atoms.len();
                if all_constant {
                    self.atoms.push(Atom::Constant(output_value));
                    self.intermediates.push(None);
                } else {
                    self.atoms.push(Atom::Variable(r#type));
                    self.intermediates.push(Some(output_value));
                }
                AtomId { index: id }
            })
            .collect::<Vec<_>>();

        if !all_constant {
            self.instructions.push(Instruction { operation, inputs, outputs: outputs.clone() });
        }
        Ok(outputs)
    }

    /// Adds a staged instruction using only abstract evaluation.
    ///
    /// This is the staging path used by type-directed tracing and any traced replay that does not
    /// have representative concrete values available for the participating atoms.
    pub fn add_instruction_abstract(&mut self, operation: O, inputs: Vec<AtomId>) -> Result<Vec<AtomId>, TracingError>
    where
        O: Op<T>,
    {
        let input_abstracts = inputs
            .iter()
            .map(|input| {
                self.atom(*input)
                    .map(|atom| atom.r#type().into_owned())
                    .ok_or(TracingError::UnboundAtomId { id: *input })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let output_abstracts = operation.abstract_eval(input_abstracts.as_slice())?;

        let is_zero = |id: AtomId| matches!(self.atom(id), Some(Atom::Constant(value)) if value.is_zero());
        let is_one = |id: AtomId| matches!(self.atom(id), Some(Atom::Constant(value)) if value.is_one());
        if let Some(simplified) = operation.try_simplify(&inputs, &is_zero, &is_one) {
            return Ok(simplified);
        }

        Ok(self.add_instruction_prevalidated(operation, inputs, output_abstracts))
    }

    /// Adds a staged instruction, validating its inputs through abstract evaluation first.
    ///
    /// When every input atom is an [`Atom::Constant`], the operation is folded at program-construction
    /// time: `abstract_eval` and `eval` are still executed for validation, but the output atoms are
    /// recorded as constants and no instruction is added to the program.
    pub fn add_instruction(&mut self, operation: O, inputs: Vec<AtomId>) -> Result<Vec<AtomId>, TracingError>
    where
        O: InterpretableOp<T, V>,
    {
        let input_examples = inputs
            .iter()
            .map(|input| self.stored_value(*input).cloned().ok_or(TracingError::UnboundAtomId { id: *input }))
            .collect::<Result<Vec<_>, _>>()?;
        let output_values = operation.interpret(input_examples.as_slice())?;
        self.add_instruction_with_output_values(operation, inputs, output_values)
    }

    /// Finalizes the builder into a program with the given input/output structures. The builder's
    /// intermediate values are discarded; the resulting program retains only the atoms, instructions,
    /// and input/output structure.
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

impl<O: Clone, T: Type, V: Traceable<T>> Default for ProgramBuilder<O, T, V> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use std::ops::{Add, Mul, Neg};

    use indoc::indoc;
    use ryft_macros::Parameter;

    use crate::{
        parameters::{Parameter, Placeholder},
        tracing_v2::{Cos, MatrixOps, OneLike, PrimitiveOp, Sin, TracingError, Value, ZeroLike, test_support},
        types::{ArrayType, DataType, Shape, Typed},
    };

    use super::*;

    #[test]
    fn program_builder_tracks_atom_kinds_and_executes() {
        let mut builder = ProgramBuilder::<PrimitiveOp<ArrayType, f64>, ArrayType, f64>::new();
        let x = builder.add_input(&2.0f64);
        let y = builder.add_input(&3.0f64);
        let two = builder.add_constant(2.0f64);
        let scaled_x = builder.add_instruction(PrimitiveOp::Scale { factor: 2.0 }, vec![x]).unwrap()[0];
        let sum = builder.add_instruction(PrimitiveOp::Add, vec![scaled_x, y]).unwrap()[0];
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
    fn program_display_uses_typed_jaxpr_like_rendering() {
        let mut builder = ProgramBuilder::<PrimitiveOp<ArrayType, f64>, ArrayType, f64>::new();
        let x = builder.add_input(&1.0f64);
        let three = builder.add_constant(3.0f64);
        let sum = builder.add_instruction(PrimitiveOp::Add, vec![x, three]).unwrap()[0];
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
        let mut builder = ProgramBuilder::<PrimitiveOp<ArrayType, f64>, ArrayType, f64>::new();
        let result = builder.add_instruction(PrimitiveOp::Add, vec![AtomId { index: 42 }, AtomId { index: 99 }]);
        assert!(matches!(
            result,
            Err(TracingError::UnboundAtomId { id }) if id == AtomId { index: 42 }
        ));
        test_support::assert_reference_program_rendering();
    }

    #[test]
    fn test_constant_folding_eliminates_instructions() {
        let mut builder = ProgramBuilder::<PrimitiveOp<ArrayType, f64>, ArrayType, f64>::new();
        let a = builder.add_constant(2.0f64);
        let b = builder.add_constant(3.0f64);

        // Adding two constants should fold: no instruction, output is constant.
        let folded = builder.add_instruction(PrimitiveOp::Add, vec![a, b]).unwrap();
        assert_eq!(folded.len(), 1);
        assert!(matches!(builder.atom(folded[0]).unwrap(), Atom::Constant(_)));
        assert_eq!(builder.instruction_count(), 0);

        // Introduce a non-constant input and combine with the folded constant.
        let x = builder.add_input(&10.0f64);
        let result = builder.add_instruction(PrimitiveOp::Mul, vec![folded[0], x]).unwrap();
        assert_eq!(result.len(), 1);
        assert!(matches!(builder.atom(result[0]).unwrap(), Atom::Variable(_)));
        assert_eq!(builder.instruction_count(), 1);

        // Build the program and verify only the non-folded instruction survived.
        let program = builder.build::<f64, f64>(vec![result[0]], Placeholder, Placeholder);
        assert_eq!(program.instructions.len(), 1);
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %3:f64[] .
                let %0:f64[] = const
                    %1:f64[] = const
                    %2:f64[] = const
                    %4:f64[] = mul %2 %3
                in (%4)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_constant_folding_program_call_produces_correct_results() {
        let mut builder = ProgramBuilder::<PrimitiveOp<ArrayType, f64>, ArrayType, f64>::new();
        let a = builder.add_constant(2.0f64);
        let b = builder.add_constant(3.0f64);
        let folded_sum = builder.add_instruction(PrimitiveOp::Add, vec![a, b]).unwrap()[0];

        let x = builder.add_input(&10.0f64);
        let product = builder.add_instruction(PrimitiveOp::Mul, vec![folded_sum, x]).unwrap()[0];
        let program = builder.build::<f64, f64>(vec![product], Placeholder, Placeholder);

        // folded_sum = 2.0 + 3.0 = 5.0, product = 5.0 * input
        assert_eq!(program.interpret(10.0).unwrap(), 50.0);
        assert_eq!(program.interpret(0.5).unwrap(), 2.5);
        assert_eq!(program.interpret(0.0).unwrap(), 0.0);
    }

    #[test]
    fn built_program_drops_derived_stored_values_but_remains_executable() {
        let mut builder = ProgramBuilder::<PrimitiveOp<ArrayType, f64>, ArrayType, f64>::new();
        let x = builder.add_input(&2.0f64);
        let three = builder.add_constant(3.0f64);
        let sum = builder.add_instruction(PrimitiveOp::Add, vec![x, three]).unwrap()[0];

        assert!(
            matches!(builder.atom(x).unwrap(), Atom::Variable(r#type) if *r#type == ArrayType::scalar(DataType::F64))
        );
        assert!(matches!(builder.atom(three).unwrap(), Atom::Constant(value) if *value == 3.0));
        assert!(matches!(builder.atom(sum).unwrap(), Atom::Variable(_)));
        assert_eq!(builder.stored_value(x), Some(&2.0));
        assert_eq!(builder.stored_value(sum), Some(&5.0));

        let program = builder.build::<f64, f64>(vec![sum], Placeholder, Placeholder);
        assert!(
            matches!(program.atoms.get(x.index), Some(Atom::Variable(r#type)) if *r#type == ArrayType::scalar(DataType::F64))
        );
        assert!(matches!(program.atoms.get(three.index), Some(Atom::Constant(value)) if *value == 3.0));
        assert!(matches!(program.atoms.get(sum.index), Some(Atom::Variable(_))));
        assert_eq!(program.interpret(4.0).unwrap(), 7.0);
    }

    #[test]
    fn custom_identity_values_participate_in_algebraic_simplification() {
        #[derive(Clone, Debug, PartialEq, Parameter)]
        struct TestIdentityValue {
            r#type: ArrayType,
            value: f64,
        }

        impl TestIdentityValue {
            fn scalar(value: f64) -> Self {
                Self { r#type: ArrayType::scalar(DataType::F64), value }
            }
        }

        impl Typed<ArrayType> for TestIdentityValue {
            fn r#type(&self) -> std::borrow::Cow<'_, ArrayType> {
                std::borrow::Cow::Borrowed(&self.r#type)
            }
        }

        impl Traceable<ArrayType> for TestIdentityValue {
            fn is_zero(&self) -> bool {
                self.value == 0.0
            }

            fn is_one(&self) -> bool {
                self.value == 1.0
            }
        }

        impl Value<ArrayType> for TestIdentityValue {}

        impl Add for TestIdentityValue {
            type Output = Self;

            fn add(self, rhs: Self) -> Self::Output {
                Self { r#type: self.r#type, value: self.value + rhs.value }
            }
        }

        impl Mul for TestIdentityValue {
            type Output = Self;

            fn mul(self, rhs: Self) -> Self::Output {
                Self { r#type: self.r#type, value: self.value * rhs.value }
            }
        }

        impl Neg for TestIdentityValue {
            type Output = Self;

            fn neg(self) -> Self::Output {
                Self { r#type: self.r#type, value: -self.value }
            }
        }

        impl Sin for TestIdentityValue {
            fn sin(self) -> Self {
                self
            }
        }

        impl Cos for TestIdentityValue {
            fn cos(self) -> Self {
                self
            }
        }

        impl ZeroLike for TestIdentityValue {
            fn zero_like(&self) -> Self {
                Self::scalar(0.0)
            }
        }

        impl OneLike for TestIdentityValue {
            fn one_like(&self) -> Self {
                Self::scalar(1.0)
            }
        }

        impl MatrixOps for TestIdentityValue {
            fn matmul(self, rhs: Self) -> Self {
                Self { r#type: self.r#type, value: self.value * rhs.value }
            }

            fn transpose_matrix(self) -> Self {
                self
            }
        }

        impl crate::tracing_v2::operations::reshape::ReshapeOps for TestIdentityValue {
            fn reshape(self, target_shape: Shape) -> Result<Self, TracingError> {
                Ok(Self { r#type: ArrayType::new(DataType::F64, target_shape, None, None).unwrap(), value: self.value })
            }
        }

        let mut builder =
            ProgramBuilder::<PrimitiveOp<ArrayType, TestIdentityValue>, ArrayType, TestIdentityValue>::new();
        let x = builder.add_input(&TestIdentityValue::scalar(5.0));
        let zero = builder.add_constant(TestIdentityValue::scalar(0.0));

        let simplified_add = builder.add_instruction(PrimitiveOp::Add, vec![x, zero]).unwrap();
        assert_eq!(simplified_add, vec![x]);
        assert_eq!(builder.instruction_count(), 0);

        let simplified_scale = builder
            .add_instruction(PrimitiveOp::Scale { factor: TestIdentityValue::scalar(1.0) }, vec![x])
            .unwrap();
        assert_eq!(simplified_scale, vec![x]);
        assert_eq!(builder.instruction_count(), 0);
    }
}
