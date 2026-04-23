use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;

use ryft_macros::Parameter;

use crate::parameters::{Parameter, ParameterError, Parameterized};
use crate::tracing::TracingError;
use crate::types::{ArrayType, Type, TypeError, Typed};

/// Marker trait that identifies concrete, non-tracer leaves.
///
/// [`Value`] is a subtrait of [`Traceable`] implemented by types that carry real data, such as
/// scalars, dense arrays, and backend-backed tensors. Tracing wrappers such as
/// [`Tracer`](crate::tracing_v2::Tracer) must not implement this trait.
///
/// The sole purpose of this marker is to give Rust's coherence checker a way to tell two blanket
/// impls apart. Each composable transform such as `jvp`, `grad`, and `vmap` provides:
///
/// 1. an impl for `V: Value<T>` that evaluates the transform on concrete data, and
/// 2. an impl for `Tracer<V>` that stages the transform into the enclosing traced program.
///
/// Because `Tracer<V>` implements [`Traceable`] but not [`Value`], the two impls never overlap.
pub trait Value<T: Type>: Traceable<T> {}

/// Base trait for any leaf type that can participate in traced computations.
///
/// [`Traceable`] is implemented by every type that can appear as a leaf in a staged program: both
/// concrete data types such as `f32`, `f64`, and backend arrays, and tracing wrappers such as
/// [`Tracer`](crate::tracing_v2::Tracer). It ties each leaf to a type descriptor `T` via
/// [`Typed`], while deliberately not implying eager numeric operations such as
/// [`Sin`](crate::tracing_v2::Sin) or differentiation-specific capabilities such as
/// [`ZeroLike`](crate::tracing_v2::operations::constants::ZeroLike). Those requirements live on
/// the primitive operations and transforms that actually need them.
///
/// The type parameter `T` determines the abstract metadata used to describe leaf shapes and element
/// types. The primary instantiation is [`ArrayType`](crate::types::ArrayType), used throughout the
/// core tracing infrastructure.
///
/// Concrete leaves that support exact algebraic identity detection should override
/// [`Traceable::is_zero`] and [`Traceable::is_one`]. The default implementations return `false`,
/// which keeps purely abstract or traced leaves valid while opting them out of
/// constant-identity simplification.
///
/// [`Traceable`] itself does not require `'static`. Borrowed leaf wrappers such as
/// [`Tracer`](crate::tracing_v2::Tracer) are therefore free to model real engine borrows
/// explicitly. Individual APIs that store traceable values behind [`Any`](std::any::Any), inside
/// long-lived registries, or in staged artifacts that intentionally escape the current scope should
/// add `'static` at those specific seams instead of imposing it on every traceable leaf globally.
///
/// # Implementing [`Traceable`] for new leaf types
///
/// Most concrete runtime values should still own their data:
///
/// - Small [`Copy`] scalars (`f32`, `i32`, `half::bf16`, ...) can implement [`Traceable`] directly,
///   as the built-in scalar impls illustrate.
/// - Heavier payloads (array buffers, tensors, device allocations) should typically wrap the
///   underlying handle in [`Arc`](std::sync::Arc) (or [`Rc`](std::rc::Rc) for single-threaded
///   cases) so the leaf stays cheaply cloneable.
///
/// Borrowing leaf types is still valid when the borrow is semantically tied to the surrounding
/// tracing scope; they simply cannot be stored in APIs that later add a `'static` requirement.
///
/// See also [`Value`], the marker subtrait that distinguishes concrete leaves from tracing
/// wrappers.
pub trait Traceable<T: Type>: Clone + Parameter + Typed<T> {
    /// Returns `true` if every element of this value is exactly zero.
    ///
    /// The program builder calls this on constant atoms during [`Operation::try_simplify`] to detect and
    /// eliminate algebraic identities at staging time, for example folding `x + 0` into `x` or
    /// `x * 0` into `0` without emitting the operation into the staged program.
    ///
    /// The default returns `false`, which is always safe: it simply opts the value out of
    /// identity-based simplification. Concrete leaf types that can inspect their contents should
    /// override this to return an accurate answer. Tracing wrappers like
    /// [`Tracer`](crate::tracing_v2::Tracer) cannot meaningfully inspect their contents at staging
    /// time and therefore keep the default.
    #[inline]
    fn is_zero(&self) -> bool {
        false
    }

    /// Returns `true` if every element of this value is exactly one.
    ///
    /// This is the multiplicative-identity counterpart of [`Traceable::is_zero`]. The program
    /// builder uses it during [`Operation::try_simplify`]
    /// to fold operations like `x * 1` into `x` or `scale(x, 1)` into `x`.
    ///
    /// The same defaulting rationale applies: `false` is always safe, and only concrete leaf types
    /// that can inspect their contents should override this.
    #[inline]
    fn is_one(&self) -> bool {
        false
    }
}

/// Staged atom carrying abstract metadata.
///
/// The variant encodes whether the atom is a retained literal constant or an ordinary program
/// variable. Input-vs-derived provenance for variable atoms lives in the owning [`Program`]'s
/// [`Program::input_ids`] list and instruction outputs rather than in the atom enum itself.
#[derive(Clone, Debug)]
pub enum Atom<T: Type, V: Typed<T>> {
    /// Literal constant value that appears in a [`Program`].
    Constant(V),

    /// Non-constant program variable carrying only its abstract type.
    Variable(T),
}

impl<T: Type, V: Typed<T>> Atom<T, V> {
    /// Returns `true` if this [`Atom`] is a [`Atom::Constant`].
    #[inline]
    pub fn is_constant(&self) -> bool {
        matches!(self, Self::Constant(_))
    }

    /// Returns `true` if this [`Atom`] is a [`Atom::Variable`].
    #[inline]
    pub fn is_variable(&self) -> bool {
        matches!(self, Self::Variable(_))
    }

    /// Returns the underlying constant value if this atom is a [`Atom::Constant`] and [`None`] otherwise.
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

/// Shape-level operation interface for staged programs.
///
/// This trait covers the metadata surface needed for program construction, display, simplification,
/// and backend lowering. Concrete execution is provided by the separate
/// [`InterpretableOperation`] trait. Staged-program differentiation rules are split between
/// [`crate::tracing_v2::LinearOperation`] (transpose/replay) and
/// [`crate::tracing_v2::DifferentiableOperation`] (forward-mode JVP).
///
/// The type parameter `T` determines which abstract type descriptor is used for shape-level
/// reasoning. The default is [`ArrayType`], which covers the entire core tracing infrastructure.
/// Future instantiations with different type descriptors can reuse the same trait without
/// modifying existing implementations.
pub trait Operation<T: Type = ArrayType>: Debug + Display {
    /// Returns the stable primitive name used in diagnostics and pretty-printing.
    fn name(&self) -> &'static str;

    /// Computes output types from input types without executing the operation.
    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError>;

    /// Returns simplified output atoms if this operation is a trivial algebraic identity.
    ///
    /// Called during program construction to eliminate no-op operations like `x + 0`, `x * 1`, or
    /// `scale(x, 1)`. The callbacks check whether an input atom is a constant zero or one. Returns
    /// [`None`] if no simplification applies.
    fn try_simplify(
        &self,
        _inputs: &[AtomId],
        _is_zero_constant: &dyn Fn(AtomId) -> bool,
        _is_one_constant: &dyn Fn(AtomId) -> bool,
    ) -> Option<Vec<AtomId>> {
        None
    }
}

/// Concrete execution capability for staged operations.
///
/// Separated from [`Operation`] so that program construction, display, and simplification can work
/// without value-type bounds. Only code paths that actually execute operations, such as program
/// replay and JIT example propagation, require this trait.
pub trait InterpretableOperation<T: Type, V: Typed<T>>: Operation<T> {
    /// Executes the operation on concrete values.
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError>;
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
/// transforms exchange programs in this form. The generic parameters intentionally stay open so the
/// same IR can represent ordinary programs plus tangent and cotangent programs over backend-specific
/// operation carriers.
pub struct Program<T: Type, V: Typed<T> + Parameter, O: Operation<T>, Input: Parameterized<V>, Output: Parameterized<V>>
{
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
                        "attempted to remap a dead variable atom during program simplification"
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
    /// Executes a *flat* [`Program`] directly given its structural parts. This helper is shared by
    /// [`Program::interpret`] and by higher-order program payloads that store flattened body programs without wrapping
    /// them in full [`Program`] values. It tracks the number of remaining uses for each atom so that replay can move
    /// values into their final consumer instead of cloning them unconditionally at every edge.
    pub(crate) fn interpret_from_parts(
        atoms: &[Atom<T, V>],
        input_ids: &[AtomId],
        output_ids: &[AtomId],
        instructions: &[Instruction<O>],
        input_values: Vec<V>,
    ) -> Result<Vec<V>, TracingError> {
        if input_values.len() != input_ids.len() {
            return Err(TracingError::InvalidInputCount { expected: input_ids.len(), got: input_values.len() });
        }

        let mut remaining_uses = vec![0usize; atoms.len()];
        for instruction in instructions {
            for input in instruction.inputs.iter().copied() {
                let Some(use_count) = remaining_uses.get_mut(input.index) else {
                    return Err(TracingError::UnboundAtomId { id: input });
                };
                *use_count += 1;
            }
        }
        for output in output_ids.iter().copied() {
            let Some(use_count) = remaining_uses.get_mut(output.index) else {
                return Err(TracingError::UnboundAtomId { id: output });
            };
            *use_count += 1;
        }

        let mut values = vec![None; atoms.len()];
        for (atom, value) in input_ids.iter().copied().zip(input_values) {
            let Some(slot) = values.get_mut(atom.index) else {
                return Err(TracingError::UnboundAtomId { id: atom });
            };
            *slot = Some(value);
        }

        for (atom_index, atom) in atoms.iter().enumerate() {
            if remaining_uses[atom_index] == 0 {
                continue;
            }
            if let Atom::Constant(value) = atom {
                values[atom_index] = Some(value.clone());
            }
        }

        let max_input_count = instructions.iter().map(|instruction| instruction.inputs.len()).max().unwrap_or(0);
        let mut instruction_inputs = Vec::with_capacity(max_input_count);
        for instruction in instructions {
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

        let mut outputs = Vec::with_capacity(output_ids.len());
        for output in output_ids.iter().copied() {
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
        Ok(outputs)
    }

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

        let outputs = Self::interpret_from_parts(
            self.atoms.as_slice(),
            self.input_ids.as_slice(),
            self.output_ids.as_slice(),
            self.instructions.as_slice(),
            input.into_parameters().collect::<Vec<_>>(),
        )?;
        Ok(Output::from_parameters(self.output_structure.clone(), outputs)?)
    }

    /// Folds any instruction whose inputs are all currently-known constants.
    ///
    /// This pass preserves the surrounding program structure. It removes the instructions that it
    /// successfully folds and rewrites their output atoms to [`Atom::Constant`], but it does not
    /// perform dead-code elimination or liveness-based cleanup afterward.
    pub fn with_folded_constants(&self) -> Result<Self, TracingError>
    where
        Input::ParameterStructure: Clone,
        Output::ParameterStructure: Clone,
    {
        let mut atoms = self.atoms.clone();
        let mut instructions = Vec::with_capacity(self.instructions.len());

        for instruction in self.instructions.iter() {
            let mut input_constants = Vec::with_capacity(instruction.inputs.len());
            let mut all_inputs_constant = true;
            for input in instruction.inputs.iter().copied() {
                let atom = atoms.get(input.index).ok_or(TracingError::UnboundAtomId { id: input })?;
                let Some(value) = atom.as_constant() else {
                    all_inputs_constant = false;
                    break;
                };
                input_constants.push(value.clone());
            }

            if !all_inputs_constant {
                instructions.push(instruction.clone());
                continue;
            }

            let output_constants = instruction.operation.interpret(input_constants.as_slice())?;
            if output_constants.len() != instruction.outputs.len() {
                return Err(TracingError::InvalidOutputCount {
                    expected: instruction.outputs.len(),
                    got: output_constants.len(),
                });
            }

            for (output_atom, output_value) in instruction.outputs.iter().copied().zip(output_constants.into_iter()) {
                let atom = atoms.get_mut(output_atom.index).ok_or(TracingError::UnboundAtomId { id: output_atom })?;
                *atom = Atom::Constant(output_value);
            }
        }

        Ok(Self {
            atoms,
            input_ids: self.input_ids.clone(),
            output_ids: self.output_ids.clone(),
            instructions,
            input_structure: self.input_structure.clone(),
            output_structure: self.output_structure.clone(),
            marker: PhantomData,
        })
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
pub struct ProgramBuilder<T: Type, V: Typed<T>, O: Operation<T>> {
    /// Atom table accumulated so far, including inputs, constants, and derived outputs.
    pub atoms: Vec<Atom<T, V>>,

    /// Input atom ids in parameter order.
    pub input_ids: Vec<AtomId>,

    /// Instructions recorded so far in execution order.
    pub instructions: Vec<Instruction<O>>,

    /// First staging failure recorded while this builder was used for traced execution.
    pub error: Option<TracingError>,
}

impl<T: Type, V: Traceable<T>, O: Operation<T>> ProgramBuilder<T, V, O> {
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
    /// (for example [`Program::simplified`] or [`Program::with_folded_constants`]). Callers that later need a representative value for this
    /// atom should synthesize it from the retained input type through an
    /// [`Engine`](crate::tracing_v2::Engine).
    #[inline]
    pub fn add_input(&mut self, r#type: T) -> AtomId {
        let id = self.add_variable(r#type);
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

    /// Adds a staged instruction using abstract evaluation and local algebraic simplification.
    ///
    /// This validates the input atoms through [`Operation::infer_output_types`], applies any local
    /// `try_simplify` rewrite exposed by the operation, and otherwise stages one variable-output
    /// instruction.
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

        let is_zero = |id: AtomId| matches!(self.atoms.get(id.index), Some(Atom::Constant(value)) if value.is_zero());
        let is_one = |id: AtomId| matches!(self.atoms.get(id.index), Some(Atom::Constant(value)) if value.is_one());
        if let Some(simplified) = operation.try_simplify(&inputs, &is_zero, &is_one) {
            return Ok(simplified);
        }

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

impl<T: Type, V: Traceable<T>, O: Operation<T>> Default for ProgramBuilder<T, V, O> {
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
        parameters::{Parameter, ParameterError, Parameterized, Placeholder},
        tracing::{TracingError, Value},
        tracing_v2::{
            Cos, MatrixOps, PrimitiveOperation, Sin,
            operations::constants::{OneLike, ZeroLike},
            test_support,
        },
        types::{ArrayType, DataType, Shape, Typed},
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
        let mut builder = ProgramBuilder::<ArrayType, f64, PrimitiveOperation<ArrayType, f64>>::new();
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
        let mut builder = ProgramBuilder::<ArrayType, f64, PrimitiveOperation<ArrayType, f64>>::new();
        let x = builder.add_input(1.0f64.r#type().into_owned());
        let doubled = builder.add_instruction(PrimitiveOperation::Add, vec![x, x]).unwrap()[0];
        let program = builder.build::<f64, (f64, f64)>(vec![doubled, doubled], Placeholder, (Placeholder, Placeholder));

        assert_eq!(program.interpret(2.0f64).unwrap(), (4.0f64, 4.0f64));
    }

    #[test]
    fn program_interpret_rejects_mismatched_parameter_structures() {
        let mut builder = ProgramBuilder::<ArrayType, f64, PrimitiveOperation<ArrayType, f64>>::new();
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
        let mut builder = ProgramBuilder::<ArrayType, f64, PrimitiveOperation<ArrayType, f64>>::new();
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
        let mut builder = ProgramBuilder::<ArrayType, f64, PrimitiveOperation<ArrayType, f64>>::new();
        let result = builder.add_instruction(PrimitiveOperation::Add, vec![AtomId { index: 42 }, AtomId { index: 99 }]);
        assert!(matches!(
            result,
            Err(TracingError::UnboundAtomId { id }) if id == AtomId { index: 42 }
        ));
        test_support::assert_reference_program_rendering();
    }

    #[test]
    fn test_constant_folding_eliminates_instructions() {
        let mut builder = ProgramBuilder::<ArrayType, f64, PrimitiveOperation<ArrayType, f64>>::new();
        let a = builder.add_constant(2.0f64);
        let b = builder.add_constant(3.0f64);

        // Builder staging stays symbolic even for constant-only instructions.
        let folded = builder.add_instruction(PrimitiveOperation::Add, vec![a, b]).unwrap();
        assert_eq!(folded.len(), 1);
        assert!(matches!(builder.atoms.get(folded[0].index), Some(Atom::Variable(_))));
        assert_eq!(builder.instructions.len(), 1);

        // Introduce a non-constant input and combine with the symbolic sum.
        let x = builder.add_input(10.0f64.r#type().into_owned());
        let result = builder.add_instruction(PrimitiveOperation::Mul, vec![folded[0], x]).unwrap();
        assert_eq!(result.len(), 1);
        assert!(matches!(builder.atoms.get(result[0].index), Some(Atom::Variable(_))));
        assert_eq!(builder.instructions.len(), 2);

        // `with_folded_constants` folds the constant-only sum but leaves dead constants in place.
        let program = builder.build::<f64, f64>(vec![result[0]], Placeholder, Placeholder);
        assert_eq!(program.instructions.len(), 2);
        let program = program.with_folded_constants().unwrap();
        assert_eq!(program.instructions.len(), 1);
        assert_eq!(program.atoms.len(), 5);
        let program = program.simplified().unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = const
                    %2:f64[] = mul %1 %0
                in (%2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_constant_folding_program_call_produces_correct_results() {
        let mut builder = ProgramBuilder::<ArrayType, f64, PrimitiveOperation<ArrayType, f64>>::new();
        let a = builder.add_constant(2.0f64);
        let b = builder.add_constant(3.0f64);
        let folded_sum = builder.add_instruction(PrimitiveOperation::Add, vec![a, b]).unwrap()[0];

        let x = builder.add_input(10.0f64.r#type().into_owned());
        let product = builder.add_instruction(PrimitiveOperation::Mul, vec![folded_sum, x]).unwrap()[0];
        let program =
            builder.build::<f64, f64>(vec![product], Placeholder, Placeholder).with_folded_constants().unwrap();

        // folded_sum = 2.0 + 3.0 = 5.0, product = 5.0 * input
        assert_eq!(program.interpret(10.0).unwrap(), 50.0);
        assert_eq!(program.interpret(0.5).unwrap(), 2.5);
        assert_eq!(program.interpret(0.0).unwrap(), 0.0);
    }

    #[test]
    fn program_builder_tracks_only_types_for_variable_atoms() {
        let mut builder = ProgramBuilder::<ArrayType, f64, PrimitiveOperation<ArrayType, f64>>::new();
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
            ProgramBuilder::<ArrayType, TestIdentityValue, PrimitiveOperation<ArrayType, TestIdentityValue>>::new();
        let x = builder.add_input(TestIdentityValue::scalar(5.0).r#type().into_owned());
        let zero = builder.add_constant(TestIdentityValue::scalar(0.0));

        let simplified_add = builder.add_instruction(PrimitiveOperation::Add, vec![x, zero]).unwrap();
        assert_eq!(simplified_add, vec![x]);
        assert_eq!(builder.instructions.len(), 0);

        let simplified_scale = builder
            .add_instruction(PrimitiveOperation::Scale { factor: TestIdentityValue::scalar(1.0) }, vec![x])
            .unwrap();
        assert_eq!(simplified_scale, vec![x]);
        assert_eq!(builder.instructions.len(), 0);
    }
}
