use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use std::sync::Arc;

use thiserror::Error;

use ryft_macros::Parameter;

use crate::contexts::Context;
use crate::errors::CustomError;
use crate::macros::check_count;
use crate::operations::{InterpretableOperation, Operation};
use crate::parameters::{Parameter, ParameterError, Parameterized, ParameterizedFamily, Placeholder};
use crate::types::{Type, TypeError, Typed};

/// Represents errors related to [`Program`]s in `ryft-core`.
#[derive(Clone, Debug, Error, PartialEq, Eq, Hash)]
pub enum ProgramError {
    #[error("values used in the same operation must share the same program builder")]
    MismatchedProgramBuilders,

    #[error("invalid number of inputs; expected {expected} but got {actual}")]
    InvalidInputCount { expected: usize, actual: usize },

    #[error("invalid number of outputs; expected {expected} but got {actual}")]
    InvalidOutputCount { expected: usize, actual: usize },

    #[error("unbound atom ID: {id}")]
    UnboundAtomId { id: AtomId },

    #[error("encountered malformed program: {0}")]
    MalformedProgram(String),

    #[error("encountered program builder that escaped its scope")]
    EscapedProgramBuilder,

    #[error("encountered poisoned value where a live value was required")]
    PoisonedValue,

    #[error("{message}")]
    Concretization { message: String },

    #[error("{message}")]
    UnsupportedOperation { message: String },

    #[error(transparent)]
    Parameter(#[from] ParameterError),

    #[error(transparent)]
    Type(#[from] TypeError),

    #[error("{0}")]
    Custom(Arc<dyn CustomError>),
}

impl ProgramError {
    /// Wraps an operation- or transform-specific error in a [`Custom`](ProgramError::Custom) variant. The concrete
    /// error can later be recovered using [`ProgramError::downcast_custom`].
    #[inline]
    pub fn custom(error: impl CustomError) -> Self {
        ProgramError::Custom(Arc::new(error))
    }

    /// Returns the wrapped custom error downcast to `T` when this is a [`Custom`](ProgramError::Custom) variant holding
    /// a `T`, and [`None`] otherwise.
    #[inline]
    pub fn downcast_custom<T: CustomError>(&self) -> Option<&T> {
        match self {
            // Deref through the `Arc` to the `dyn CustomError`, upcast to `&dyn std::error::Error`, and then use the
            // standard error downcast. Going through the `Arc` directly would downcast the `Arc` instead of the error.
            ProgramError::Custom(custom) => (&**custom as &dyn std::error::Error).downcast_ref::<T>(),
            _ => None,
        }
    }
}

/// Represents leaf values that can participate in traced [`Program`]s. [`Value`] is implemented by every type that
/// can appear as a leaf in a staged [`Program`]: both concrete data types such as `f32`, `f64`, and backend arrays, and
/// tracing wrappers such as [`Tracer`](crate::Tracer). It ties each leaf to a type descriptor `T` via [`Typed`] and
/// requires [`Debug`] and [`Display`] so that diagnostics, constants, and [`Operation`] metadata can render their
/// carried values directly.
pub trait Value<T: Type>: Clone + Debug + Display + Parameter + Typed<T> + Sized {
    /// Context required to interpret an [`Operation`] into values of this type. Concrete (i.e., **eager**) values use
    /// [`EagerContext`](crate::EagerContext) to make the absence of backend-owned runtime state explicit. On the other
    /// hand, [`Tracer`](crate::Tracer) values use the [`StagingContext`](crate::StagingContext) that owns them so that
    /// nullary operations (e.g., [`ZeroOperation`](crate::ZeroOperation)) can stage themselves into the surrounding
    /// trace instead of failing for lack of an operand from which to recover that context. Wrapper value types delegate
    /// to their payload's context when they carry one.
    type InterpretationContext: Context<Type = T>;

    /// Recovers the [`InterpretationContext`](Self::InterpretationContext) carried by this value, or `None` when this
    /// value does not have a payload (e.g., when it represents a symbolic zero value) and therefore carries no context
    /// to recover. Concrete (i.e., **eager**) values return a zero-sized [`EagerContext`](crate::EagerContext). On the
    /// other hand, [`Tracer`](crate::Tracer) values return clones of the [`StagingContext`](crate::StagingContext) that
    /// owns them, and wrapper value types delegate to their payload's context, returning `None` for payload-less
    /// variants.
    ///
    /// This function lets generic helpers that interpret operand-bearing [`Operation`]s recover the context from one of
    /// their input values instead of requiring callers to thread one in. Because a single operand can sometimes be a
    /// symbolic zero, callers scan their operands for the first one that yields `Some`.
    fn interpretation_context(&self) -> Option<Self::InterpretationContext>;
}

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

impl<T: Type, V: Value<T>, O: Operation<T>, Input: Parameterized<V>, Output: Parameterized<V>>
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

    /// Returns the [`Type`]s of the inputs of this [`Program`], in order.
    pub fn input_types(&self) -> Vec<T> {
        self.inputs().map(|input| input.r#type().into_owned()).collect()
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

    /// Returns the [`Type`]s of the outputs of this [`Program`], in order.
    pub fn output_types(&self) -> Vec<T> {
        self.outputs().map(|output| output.r#type().into_owned()).collect()
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

    /// Returns a boolean mask that has the same length as the number of [`Atom`]s in this [`Program`] and contains the
    /// value `true` for atoms that are inputs of the program, and `false` for other atoms.
    pub fn inputs_mask(&self) -> Vec<bool> {
        let mut inputs_mask = vec![false; self.atoms.len()];
        for input in self.input_ids.iter().copied() {
            if let Some(slot) = inputs_mask.get_mut(input.index()) {
                *slot = true;
            }
        }
        inputs_mask
    }

    /// Returns a vector that has the same length as the number of [`Atom`]s in this [`Program`] and for every atom, it
    /// contains the index of the [`Instruction`] that produces it. Note that input and constant atoms are not produced
    /// by an instruction and so the vector contains [`None`] for those atoms.
    pub fn instruction_by_output(&self) -> Vec<Option<usize>> {
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

    /// Computes transitive liveness for the [`Atom`]s and [`Instruction`]s of this [`Program`] with respect to the
    /// [`Program`]'s outputs (i.e., it determines whether each atom or instruction contributes to at least one of the
    /// [`Program`]s outputs).
    ///
    /// Note that liveness here is computed in a conservative fashion where, when any output of an instruction is live,
    /// every input to that instruction is considered live as well. Refer to [`Self::live_sets_with`] if you want to
    /// compute liveness in a more fine-grained fashion.
    #[inline]
    pub fn live_sets(&self) -> ProgramLiveSets {
        self.live_sets_for_atoms(self.output_ids.as_slice()).unwrap()
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
        F: FnMut(&Program<T, V, O, Input, Output>, &Instruction<O>, &[bool], &mut Vec<bool>) -> Result<(), ProgramError>,
    >(
        &self,
        propagate_liveness: F,
    ) -> Result<ProgramLiveSets, ProgramError> {
        self.live_sets_for_atoms_with(self.output_ids.as_slice(), propagate_liveness)
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
        F: FnMut(&Program<T, V, O, Input, Output>, &Instruction<O>, &[bool], &mut Vec<bool>) -> Result<(), ProgramError>,
    >(
        &self,
        output_ids: &[AtomId],
        mut propagate_liveness: F,
    ) -> Result<ProgramLiveSets, ProgramError> {
        let mut live_sets = ProgramLiveSets::new(vec![false; self.atoms.len()], vec![false; self.instructions.len()]);
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
        for (instruction_index, instruction) in self.instructions.iter().enumerate().rev() {
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

    /// Rebuilds this [`Program`] with each [`Operation`] mapped using the provided `map_fn`. The atom table,
    /// input/output atom identifiers, and parameter structures are preserved exactly. This is useful for transforms
    /// that keep the same value graph but need to change operation payloads. For example, a reusable residualized
    /// linear program may contain operations whose scale/dot factors are residual references rather than executable
    /// values. Before interpreting that program, the mapping closure can receive each linear operation, call the
    /// operation's factor-mapping hook, and replace each residual reference with the concrete residual value captured
    /// by the corresponding linearization run.
    pub fn map_operations<P: Operation<T>, F: FnMut(&O) -> Result<P, ProgramError>>(
        &self,
        mut map_fn: F,
    ) -> Result<Program<T, V, P, Input, Output>, ProgramError> {
        Ok(Program {
            atoms: self.atoms.clone(),
            input_ids: self.input_ids.clone(),
            output_ids: self.output_ids.clone(),
            instructions: self
                .instructions
                .iter()
                .map(|instruction| {
                    Ok(Instruction::new(
                        map_fn(instruction.operation())?,
                        instruction.inputs().to_vec(),
                        instruction.outputs().to_vec(),
                    ))
                })
                .collect::<Result<Vec<_>, ProgramError>>()?,
            input_structure: self.input_structure.clone(),
            output_structure: self.output_structure.clone(),
            marker: PhantomData,
        })
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

    /// Returns a simplified version of this [`Program`] with dead constants and [`Instruction`]s that do not contribute
    /// to the [`Program`]'s output removed.
    pub fn simplified(&self) -> Result<Self, ProgramError>
    where
        O: Clone,
    {
        let instruction_by_output = self.instruction_by_output();
        let mut program_builder = ProgramBuilder::new();
        let mut atom_id_mapping = HashMap::with_capacity(self.atoms.len());
        for input_id in self.input_ids.iter().copied() {
            let input = self.atoms.get(input_id.index).ok_or(ProgramError::UnboundAtomId { id: input_id })?;
            let Atom::Variable(input_type) = input else {
                return Err(ProgramError::MalformedProgram("program input atom was not a variable".to_string()));
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
                    instruction_by_output.as_slice(),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        program_builder.build(output_ids, self.input_structure.clone(), self.output_structure.clone())
    }

    /// Consumes this [`Program`] and returns a simplified version with dead constants and [`Instruction`]s that do not
    /// contribute to the [`Program`]'s output removed. Unlike [`Self::simplified`], this method moves live [`Atom`]s,
    /// [`Instruction`]s, and parameter structures into the returned [`Program`] instead of cloning them. This avoids
    /// copying constants and operations that are discarded during simplification.
    pub fn into_simplified(self) -> Result<Self, ProgramError> {
        let instruction_by_output = self.instruction_by_output();
        let Program { atoms, input_ids, output_ids, instructions, input_structure, output_structure, marker: _ } = self;

        let expected_input_count = input_structure.parameter_count();
        check_count!("input", input_ids, expected_input_count, ProgramError);

        let expected_output_count = output_structure.parameter_count();
        check_count!("output", output_ids, expected_output_count, ProgramError);

        let mut atoms = atoms.into_iter().map(Some).collect::<Vec<_>>();
        let mut instructions = instructions.into_iter().map(Some).collect::<Vec<_>>();
        let mut new_atoms = Vec::with_capacity(atoms.len());
        let mut new_input_ids = Vec::with_capacity(input_ids.len());
        let mut new_instructions = Vec::with_capacity(instructions.len());
        let mut atom_id_mapping = HashMap::with_capacity(atoms.len());
        for input_id in input_ids {
            let input = atoms
                .get_mut(input_id.index)
                .ok_or(ProgramError::UnboundAtomId { id: input_id })?
                .take()
                .ok_or(ProgramError::MalformedProgram("program input atom was already moved".to_string()))?;
            let Atom::Variable(input_type) = input else {
                return Err(ProgramError::MalformedProgram("program input atom was not a variable".to_string()));
            };
            let new_input = AtomId { index: new_atoms.len() };
            new_atoms.push(Atom::Variable(input_type));
            new_input_ids.push(new_input);
            atom_id_mapping.insert(input_id, new_input);
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

    /// Rebuilds this [`Program`] as a flat subprogram over a chosen input/output boundary. The rebuilt program
    /// keeps only the [`Instruction`]s reachable from `outputs` and lifts embedded constants directly into the result.
    /// Entries of `inputs` that are not reachable from any requested output are dropped. The returned index vector
    /// lists, in order, the positions of `inputs` that remain live and become the public inputs of the rebuilt
    /// program, so that callers can map rebuilt inputs back to the original boundary.
    ///
    /// Each [`Atom::Variable`] reachable from an output must either appear in `inputs` or be produced by an
    /// [`Instruction`] of this program; reaching any other source variable (e.g., an original program input that
    /// was not selected) is reported as a [`ProgramError::MalformedProgram`]. Every entry of `inputs` must be an
    /// [`Atom::Variable`] and must appear at most once. [`Atom::Constant`]s are rebuilt automatically and need not
    /// be listed.
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
    pub fn filtered(
        &self,
        inputs: &[AtomId],
        outputs: &[AtomId],
    ) -> Result<(Program<T, V, O, Vec<V>, Vec<V>>, Vec<usize>), ProgramError>
    where
        O: Clone,
    {
        let (instruction_by_output, input_liveness) = self.compute_live_inputs(inputs, outputs)?;
        let mut program_builder = ProgramBuilder::new();
        let mut atom_id_mapping = HashMap::with_capacity(self.atoms.len());
        let mut live_input_indices = Vec::new();

        for (position, id) in inputs.iter().copied().enumerate() {
            if !input_liveness[position] {
                continue;
            }
            let Atom::Variable(input_type) = &self.atoms[id.index()] else {
                return Err(ProgramError::MalformedProgram(format!("filter input atom {id} is not a variable")));
            };
            atom_id_mapping.insert(id, program_builder.add_input(input_type.clone()));
            live_input_indices.push(position);
        }

        let output_ids = outputs
            .iter()
            .copied()
            .map(|id| {
                add_atom_to_program_builder(
                    &mut program_builder,
                    &mut atom_id_mapping,
                    id,
                    self,
                    instruction_by_output.as_slice(),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        let program = program_builder.build::<Vec<V>, Vec<V>>(
            output_ids,
            vec![Placeholder; live_input_indices.len()],
            vec![Placeholder; outputs.len()],
        )?;

        Ok((program, live_input_indices))
    }

    /// Consumes this [`Program`] and returns the same flat subprogram as [`Self::filtered`] over the chosen `inputs`
    /// and `outputs` boundary. Unlike [`Self::filtered`], this moves live [`Atom`]s and [`Instruction`]s into the
    /// returned program instead of cloning them, avoiding copies of the constants and operations that survive the
    /// projection. The boundary contract, dead-input pruning, and returned live-input index vector are identical
    /// to [`Self::filtered`].
    pub fn into_filtered(
        self,
        inputs: &[AtomId],
        outputs: &[AtomId],
    ) -> Result<(Program<T, V, O, Vec<V>, Vec<V>>, Vec<usize>), ProgramError> {
        let (instruction_by_output, input_liveness) = self.compute_live_inputs(inputs, outputs)?;
        let Program { atoms, instructions, .. } = self;
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
            let new_input = AtomId { index: new_atoms.len() };
            new_atoms.push(Atom::Variable(input_type));
            new_input_ids.push(new_input);
            atom_id_mapping.insert(id, new_input);
            live_input_indices.push(position);
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

        Ok((
            Program {
                input_structure: vec![Placeholder; new_input_ids.len()],
                output_structure: vec![Placeholder; output_ids.len()],
                atoms: new_atoms,
                input_ids: new_input_ids,
                output_ids,
                instructions: new_instructions,
                marker: PhantomData,
            },
            live_input_indices,
        ))
    }

    /// Validates `inputs` as a deduplicated set of [`Atom::Variable`]s and determines, by reverse reachability
    /// from `outputs`, which of them are live (reachable from a requested output). Returns this program's
    /// instruction-by-output map together with one liveness flag per `inputs` entry. Reaching any variable that is
    /// neither listed in `inputs` nor produced by an [`Instruction`] is reported as a
    /// [`ProgramError::MalformedProgram`].
    fn compute_live_inputs(
        &self,
        inputs: &[AtomId],
        outputs: &[AtomId],
    ) -> Result<(Vec<Option<usize>>, Vec<bool>), ProgramError> {
        let mut input_position = vec![None; self.atoms.len()];
        for (position, id) in inputs.iter().copied().enumerate() {
            let atom = self.atoms.get(id.index()).ok_or(ProgramError::UnboundAtomId { id })?;
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
        let mut needed = vec![false; self.atoms.len()];
        let mut input_liveness = vec![false; inputs.len()];
        let mut stack = Vec::new();
        for output in outputs.iter().copied() {
            if output.index() >= self.atoms.len() {
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
            match &self.atoms[atom_id.index()] {
                Atom::Constant(_) => {}
                Atom::Variable(_) => {
                    let instruction_index = instruction_by_output.get(atom_id.index()).copied().flatten().ok_or(
                        ProgramError::MalformedProgram(format!(
                            "filter atom {atom_id} is not a selected input and has no producer",
                        )),
                    )?;
                    for input in self.instructions[instruction_index].inputs.iter().copied() {
                        if !needed[input.index()] {
                            needed[input.index()] = true;
                            stack.push(input);
                        }
                    }
                }
            }
        }

        Ok((instruction_by_output, input_liveness))
    }

    /// Interprets/executes this [`Program`] with the provided input. This is the main replay entry point for staged
    /// [`Program`]s. It checks that the provided input value matches the program's expected input structure and type,
    /// evaluates the [`Instruction`]s in order, and finally builds a structured output value from the computed output
    /// values.
    #[inline]
    pub fn interpret(&self, input: Input) -> Result<Output, ProgramError>
    where
        O: InterpretableOperation<T, V>,
        <V as Value<T>>::InterpretationContext: Default,
        Input::ParameterStructure: Debug + PartialEq,
    {
        self.interpret_in_context(&<V as Value<T>>::InterpretationContext::default(), input)
    }

    /// Interprets/executes this [`Program`] with the provided input, within the supplied
    /// [`InterpretationContext`](Value::InterpretationContext). This is the context-taking core
    /// behind [`Self::interpret`]. [`Self::interpret`] supports eager value types by building an
    /// [`EagerContext`](crate::EagerContext), while callers that interpret using [`Tracer`](crate::Tracer)s need to
    /// supply the surrounding [`StagingContext`](crate::StagingContext) using this function so that nullary operations
    /// can stage themselves into it. Nested program interpretation (e.g., control flow branches, custom derivative
    /// programs, etc.) routes through here so that a single replay path handles both eager and traced values.
    pub fn interpret_in_context(
        &self,
        context: &<V as Value<T>>::InterpretationContext,
        input: Input,
    ) -> Result<Output, ProgramError>
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

        // Flatten the structured input and validate each input value's type.
        let inputs = input.into_parameters().collect::<Vec<_>>();
        for (input, input_id) in inputs.iter().zip(self.input_ids.iter()) {
            let Some(declared) = self.atoms.get(input_id.index()) else {
                return Err(ProgramError::UnboundAtomId { id: *input_id });
            };
            let declared = declared.r#type();
            let actual = input.r#type();
            if !declared.is_refined_by(actual.as_ref()) {
                return Err(TypeError {
                    message: format!(
                        "encountered input type {actual} which is incompatible with the program's \
                        declared type {declared}",
                    ),
                }
                .into());
            }
        }

        // Replay using ordinary interpretation and reshape the flat outputs back into the expected
        // structured `Output` form expected by this program.
        let outputs = self.interpret_with(
            inputs,
            |_, constant| Ok(constant.clone()),
            |instruction, inputs| instruction.operation.interpret(context, inputs),
        )?;

        Ok(Output::from_parameters(self.output_structure.clone(), outputs)?)
    }

    /// Interprets/executes this [`Program`]'s [`Instruction`]s using the caller-supplied value and error semantics.
    /// Transforms and backends specialize this interpretation by choosing a runtime value type `V`, an error type `E`,
    /// a constant-lifting closure `lift_fn`, and an instruction-interpretation closure `interpret_fn`. Inputs and
    /// outputs are flat [`Vec`]s aligned with the program's [`Self::input_ids`] and [`Self::output_ids`]. Structured
    /// input/output handling stays at the call site so that callers can use any parameter family of their choice.
    ///
    /// The `E` type parameter mirrors `V`: a [`Program`] is not tied to a single interpretation, and each
    /// interpretation has its own natural error type. Eager execution interprets instructions into concrete values and
    /// fails with [`ProgramError`], while a backend that lowers each instruction into a compiler IR interprets them
    /// into IR value handles and fails with that backend's own error (e.g., the XLA backend lowers a program into MLIR
    /// values, failing with an MLIR or sharding lowering error). The `E: From<ProgramError>` bound lets one signature
    /// serve both: callers choose the error their closures fail with, and this function's own structural errors (e.g.,
    /// [`ProgramError::UnboundAtomId`] or an input/output count mismatch) fold into that type. A backend error could
    /// instead be boxed into [`ProgramError::Custom`] and recovered by downcasting, but because this function is
    /// already generic over `V`, carrying the matching `E` keeps each interpreter's error statically typed rather than
    /// erasing it to a runtime downcast.
    ///
    /// # Parameters
    ///
    ///   - `inputs`: Flat input values aligned with [`Self::input_ids`].
    ///   - `lift_fn`: Closure that lifts an [`Atom::Constant`]'s carried `V` into the runtime leaf type `Value`. This
    ///     closure receives the constant's [`AtomId`] for callers that surface diagnostics or maintain parallel atom
    ///     tables and is invoked at most once per live constant atom, in atom-index order.
    ///   - `interpret_fn`: Closure that interprets one [`Instruction`]'s [`Operation`] to its already-lifted inputs and
    ///     returns the instruction's outputs. The full [`Instruction`] is provided so that the closure can inspect the
    ///     operation's expected output [`Atom`] IDs when needed (e.g., to look up output [`Type`]s).
    pub fn interpret_with<
        Value: Clone,
        Error: From<ProgramError>,
        LiftFn: FnMut(AtomId, &V) -> Result<Value, Error>,
        InterpretFn: FnMut(&Instruction<O>, &[Value]) -> Result<Vec<Value>, Error>,
    >(
        &self,
        inputs: Vec<Value>,
        mut lift_fn: LiftFn,
        mut interpret_fn: InterpretFn,
    ) -> Result<Vec<Value>, Error> {
        check_count!("input", inputs, self.input_ids.len(), ProgramError);

        // Count every future consumer of each atom, including final program outputs. These counts let us move each
        // value out on its last use and clone it only when a later consumer still needs it.
        let mut remaining_uses = vec![0usize; self.atoms.len()];
        for instruction in self.instructions.iter() {
            for input_id in instruction.inputs.iter().copied() {
                let Some(remaining_uses) = remaining_uses.get_mut(input_id.index) else {
                    return Err(ProgramError::UnboundAtomId { id: input_id }.into());
                };
                *remaining_uses += 1;
            }
        }
        for output_id in self.output_ids.iter().copied() {
            let Some(remaining_uses) = remaining_uses.get_mut(output_id.index) else {
                return Err(ProgramError::UnboundAtomId { id: output_id }.into());
            };
            *remaining_uses += 1;
        }

        // Store concrete input values in a sparse value table indexed by [`AtomId`].
        let mut values = vec![None; self.atoms.len()];
        for (input_id, input) in self.input_ids.iter().copied().zip(inputs) {
            let Some(slot) = values.get_mut(input_id.index) else {
                return Err(ProgramError::UnboundAtomId { id: input_id }.into());
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
                values[atom_index] = Some(lift_fn(AtomId { index: atom_index }, value)?);
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
            let outputs = interpret_fn(instruction, instruction_inputs.as_slice())?;
            check_count!("output", outputs, instruction.outputs.len(), ProgramError);

            for (output_id, output) in instruction.outputs.iter().copied().zip(outputs) {
                let Some(value) = values.get_mut(output_id.index) else {
                    return Err(ProgramError::UnboundAtomId { id: output_id }.into());
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

impl<T: Type, V: Value<T>, O: Operation<T>, Input: Parameterized<V>, Output: Parameterized<V>>
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

impl<T: Type, V: Value<T>, O: Clone, Input: Parameterized<V>, Output: Parameterized<V>> Clone
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

impl<T: Type, V: Value<T>, O: Operation<T>, Input: Parameterized<V>, Output: Parameterized<V>> Display
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
/// but also carries an optional [`ProgramError`] that can be used to signal a failure during program construction.
#[derive(Clone, Debug)]
pub struct ProgramBuilder<T: Type, V: Typed<T> + Parameter, O: Operation<T>> {
    /// [`Atom`]s contained in the [`Program`] that is being built, in the order in which they will be evaluated.
    pub(crate) atoms: Vec<Atom<T, V>>,

    /// [`AtomId`]s of the [`Atom`]s that correspond to the inputs (i.e., arguments) of the [`Program`] being built.
    pub(crate) input_ids: Vec<AtomId>,

    /// Ordered sequence of [`Instruction`]s that make up the computational graph of the [`Program`] being built.
    pub(crate) instructions: Vec<Instruction<O>>,

    /// Optional [`ProgramError`] encountered during program construction that will be propagated via [`Self::build`].
    pub(crate) error: Option<ProgramError>,
}

impl<T: Type, V: Value<T>, O: Operation<T>> ProgramBuilder<T, V, O> {
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
    pub fn error(&self) -> Option<&ProgramError> {
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
    pub fn add_instruction<P: Into<O>>(
        &mut self,
        operation: P,
        inputs: Vec<AtomId>,
    ) -> Result<&[AtomId], ProgramError> {
        let operation = operation.into();
        let input_types = inputs
            .iter()
            .map(|input| {
                self.atoms
                    .get(input.index)
                    .map(|atom| atom.r#type().into_owned())
                    .ok_or(ProgramError::UnboundAtomId { id: *input })
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
    ) -> Result<Program<T, V, O, Input, Output>, ProgramError> {
        if let Some(error) = self.error {
            return Err(error);
        }

        let expected_input_count = input_structure.parameter_count();
        check_count!("input", self.input_ids, expected_input_count, ProgramError);

        let expected_output_count = output_structure.parameter_count();
        check_count!("output", output_ids, expected_output_count, ProgramError);

        // Verify that variable dependencies are either inputs or previous instruction outputs.
        let mut input_atoms = vec![false; self.atoms.len()];
        let mut variable_has_provider = vec![false; self.atoms.len()];
        for input_id in self.input_ids.iter().copied() {
            let input = self.atoms.get(input_id.index).ok_or(ProgramError::UnboundAtomId { id: input_id })?;
            let Atom::Variable(_) = input else {
                return Err(ProgramError::MalformedProgram("program input atom was not a variable".to_string()).into());
            };
            if input_atoms[input_id.index] {
                return Err(ProgramError::MalformedProgram(format!(
                    "program input atom {input_id} appears more than once",
                )));
            }
            input_atoms[input_id.index] = true;
            variable_has_provider[input_id.index] = true;
        }
        for instruction in self.instructions.iter() {
            for input_id in instruction.inputs.iter().copied() {
                let input = self.atoms.get(input_id.index).ok_or(ProgramError::UnboundAtomId { id: input_id })?;
                if input.is_variable() && !variable_has_provider[input_id.index] {
                    return Err(ProgramError::MalformedProgram("variable atom has no owning instruction".to_string()));
                }
            }
            for output_id in instruction.outputs.iter().copied() {
                let output = self.atoms.get(output_id.index).ok_or(ProgramError::UnboundAtomId { id: output_id })?;
                let Atom::Variable(_) = output else {
                    return Err(ProgramError::MalformedProgram(
                        "instruction output atom was not a variable".to_string(),
                    ));
                };
                if input_atoms[output_id.index] {
                    return Err(ProgramError::MalformedProgram(format!(
                        "instruction output atom {output_id} is a program input",
                    )));
                }
                if variable_has_provider[output_id.index] {
                    return Err(ProgramError::MalformedProgram(format!(
                        "instruction output atom {output_id} is produced by more than one instruction",
                    )));
                }
                variable_has_provider[output_id.index] = true;
            }
        }
        for output_id in output_ids.iter().copied() {
            let output = self.atoms.get(output_id.index).ok_or(ProgramError::UnboundAtomId { id: output_id })?;
            if output.is_variable() && !variable_has_provider[output_id.index] {
                return Err(ProgramError::MalformedProgram("variable atom has no owning instruction".to_string()));
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

impl<T: Type, V: Value<T>, O: Operation<T>> Default for ProgramBuilder<T, V, O> {
    fn default() -> Self {
        Self::new()
    }
}

/// Adds the [`Atom`] that corresponds to `atom_id` in `program` to the provided [`ProgramBuilder`], recursively adding
/// its transitive producers first and memoizing the old-to-new [`AtomId`] mapping in `atom_id_mapping`. Atoms already
/// present in the mapping (e.g., rebuilt program inputs) are reused, [`Atom::Constant`]s are rebuilt directly, and
/// [`Atom::Variable`]s are reconstructed from their producing [`Instruction`]. A reachable variable that is neither
/// mapped nor produced by an instruction is reported as a [`ProgramError::MalformedProgram`].
fn add_atom_to_program_builder<
    T: Type,
    V: Value<T>,
    O: Clone + Operation<T>,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
>(
    program_builder: &mut ProgramBuilder<T, V, O>,
    atom_id_mapping: &mut HashMap<AtomId, AtomId>,
    atom_id: AtomId,
    program: &Program<T, V, O, Input, Output>,
    instruction_by_output: &[Option<usize>],
) -> Result<AtomId, ProgramError> {
    if let Some(mapped_atom) = atom_id_mapping.get(&atom_id) {
        return Ok(*mapped_atom);
    }
    let atom = program.atoms.get(atom_id.index).ok_or(ProgramError::UnboundAtomId { id: atom_id })?;
    let atom = match atom {
        Atom::Constant(value) => Ok(program_builder.add_constant(value.clone())),
        Atom::Variable(_) => {
            let instruction_index = instruction_by_output
                .get(atom_id.index)
                .copied()
                .flatten()
                .ok_or(ProgramError::MalformedProgram("variable atom has no owning instruction".to_string()))?;
            let instruction = &program.instructions[instruction_index];
            let inputs = instruction
                .inputs
                .iter()
                .copied()
                .map(|input| {
                    add_atom_to_program_builder(program_builder, atom_id_mapping, input, program, instruction_by_output)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let outputs = program_builder.add_instruction(instruction.operation.clone(), inputs)?;
            check_count!("output", outputs, instruction.outputs.len(), ProgramError);
            instruction.outputs.iter().copied().zip(outputs.iter().copied()).for_each(|(old, new)| {
                atom_id_mapping.insert(old, new);
            });
            atom_id_mapping
                .get(&atom_id)
                .copied()
                .ok_or(ProgramError::MalformedProgram("remapped instruction output was missing".to_string()))
        }
    }?;
    atom_id_mapping.insert(atom_id, atom);
    Ok(atom)
}

/// Moves the [`Atom`] that corresponds to `atom_id` (and its transitive producers) out of `atoms`/`instructions` into
/// `new_atoms`/`new_instructions`, memoizing the old-to-new [`AtomId`] mapping in `atom_id_mapping`. This is the
/// move-based counterpart of [`add_atom_to_program_builder`]: it relocates owned [`Atom`]s and [`Instruction`]s instead
/// of cloning them, so each is taken from its slot at most once. Atoms already present in the mapping are reused, and a
/// reachable variable that is neither mapped nor produced by an instruction is reported as a
/// [`ProgramError::MalformedProgram`].
fn move_atom_to_program<T: Type, V: Value<T>, O: Operation<T>>(
    atom_id_mapping: &mut HashMap<AtomId, AtomId>,
    atom_id: AtomId,
    atoms: &mut [Option<Atom<T, V>>],
    instructions: &mut [Option<Instruction<O>>],
    instruction_by_output: &[Option<usize>],
    new_atoms: &mut Vec<Atom<T, V>>,
    new_instructions: &mut Vec<Instruction<O>>,
) -> Result<AtomId, ProgramError> {
    if let Some(mapped_atom) = atom_id_mapping.get(&atom_id) {
        return Ok(*mapped_atom);
    }
    let is_constant = match atoms.get(atom_id.index) {
        Some(Some(Atom::Constant(_))) => true,
        Some(Some(Atom::Variable(_))) => false,
        Some(None) => {
            return Err(ProgramError::MalformedProgram(format!(
                "atom {atom_id} was already moved while rebuilding program",
            )));
        }
        None => return Err(ProgramError::UnboundAtomId { id: atom_id }.into()),
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
    let instruction_index = instruction_by_output
        .get(atom_id.index)
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
            .get_mut(output.index)
            .ok_or(ProgramError::UnboundAtomId { id: output })?
            .take()
            .ok_or(ProgramError::MalformedProgram("instruction output atom was already moved".to_string()))?;
        let Atom::Variable(output_type) = output_atom else {
            return Err(ProgramError::MalformedProgram("instruction output atom was not a variable".to_string()));
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

    use crate::contexts::EagerContext;
    use crate::macros::check_count;
    use crate::operations::OperationFormatter;
    use crate::operations::arithmetic::{AddOperation, MulOperation, ScaleOperation};
    use crate::operations::scalars::ScalarOperation;
    use crate::parameters::{ParameterError, Parameterized, Placeholder};
    use crate::tests::TestArray;
    use crate::types::{ArrayType, DataType, Shape, Size, TypeError};

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
        let o0 = builder.add_instruction(AddOperation, vec![i0, c0]).unwrap()[0];
        let program = builder.build::<f64, f64>(vec![o0], Placeholder, Placeholder).unwrap();
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
        let mut builder = ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new();
        let i0 = builder.add_input(DataType::F64);
        let i1 = builder.add_input(DataType::F64);
        let v0 = builder.add_instruction(ScaleOperation::new(2.0), vec![i0]).unwrap()[0];
        let o0 = builder.add_instruction(AddOperation, vec![v0, i1]).unwrap()[0];
        let program = builder.build::<(f64, f64), f64>(vec![o0], (Placeholder, Placeholder), Placeholder).unwrap();
        assert_eq!(program.input_types(), vec![DataType::F64, DataType::F64]);
        assert_eq!(program.output_types(), vec![DataType::F64]);
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
        let o0 = builder.add_instruction(AddOperation, vec![i0, i0]).unwrap()[0];
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
            Err(ProgramError::MalformedProgram(message)) if message == "variable atom has no owning instruction",
        ));

        // Test a case where we have an instruction input atom with no parent instruction.
        let mut builder = ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new();
        let i0 = builder.add_input(DataType::F64);
        let v0 = builder.add_variable(DataType::F64);
        let o0 = builder.add_instruction(AddOperation, vec![i0, v0]).unwrap()[0];
        assert!(matches!(
            builder.build::<f64, f64>(vec![o0], Placeholder, Placeholder),
            Err(ProgramError::MalformedProgram(message)) if message == "variable atom has no owning instruction",
        ));
    }

    #[test]
    fn test_program_interpret_lifts_live_constants_once() {
        let mut builder = ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new();
        let i0 = builder.add_input(DataType::F64);
        let c0 = builder.add_constant(7.0f64);
        let c1 = builder.add_constant(3.0f64);
        let o0 = builder.add_instruction(AddOperation, vec![i0, c1]).unwrap()[0];
        let program = builder.build::<f64, f64>(vec![o0], Placeholder, Placeholder).unwrap();
        let mut lifted_constants = Vec::new();
        assert_eq!(
            program.interpret_with(
                vec![2.0f64],
                |atom_id, value| {
                    lifted_constants.push((atom_id, *value));
                    Ok(*value)
                },
                |instruction, inputs| instruction.operation.interpret(&EagerContext::new(), inputs),
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
            Err(ProgramError::Parameter(ParameterError::MismatchedParameterStructures {
                left_structure,
                right_structure,
            })) if left_structure == format!("{:?}", vec![Placeholder])
                && right_structure == format!("{:?}", vec![1.0f64, 2.0f64].parameter_structure())
        ));
    }

    #[test]
    fn test_program_interpret_input_type_checking() {
        // A statically typed program input rejects values whose concrete types do not match it exactly.
        let mut builder = ProgramBuilder::<ArrayType, TestArray, AddOperation>::new();
        let i0 = builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2)])));
        let o0 = builder.add_instruction(AddOperation, vec![i0, i0]).unwrap()[0];
        let program = builder.build::<TestArray, TestArray>(vec![o0], Placeholder, Placeholder).unwrap();
        assert_eq!(program.interpret(TestArray::vector(vec![1.0, 2.0])).unwrap().values, vec![2.0, 4.0]);
        assert!(matches!(
            program.interpret(TestArray::vector(vec![1.0, 2.0, 3.0])),
            Err(ProgramError::Type(TypeError { message })) if message
                == "encountered input type f64[3] which is incompatible with the program's declared type f64[2]",
        ));

        // An unbounded dynamically sized program input accepts concrete values of any size, so one staged program
        // replays at several concrete sizes. Rank mismatches are still rejected.
        let mut builder = ProgramBuilder::<ArrayType, TestArray, AddOperation>::new();
        let i0 = builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Size::Dynamic(None)])));
        let o0 = builder.add_instruction(AddOperation, vec![i0, i0]).unwrap()[0];
        let program = builder.build::<TestArray, TestArray>(vec![o0], Placeholder, Placeholder).unwrap();
        assert_eq!(program.interpret(TestArray::vector(vec![1.0, 2.0])).unwrap().values, vec![2.0, 4.0]);
        assert_eq!(program.interpret(TestArray::vector(vec![1.0, 2.0, 3.0])).unwrap().values, vec![2.0, 4.0, 6.0]);
        assert!(matches!(
            program.interpret(TestArray::scalar(1.0)),
            Err(ProgramError::Type(TypeError { message })) if message
                == "encountered input type f64[] which is incompatible with the program's declared type f64[*]",
        ));

        // A bounded dynamically sized program input enforces its exclusive upper bound on concrete sizes.
        let mut builder = ProgramBuilder::<ArrayType, TestArray, AddOperation>::new();
        let i0 = builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Size::Dynamic(Some(3))])));
        let o0 = builder.add_instruction(AddOperation, vec![i0, i0]).unwrap()[0];
        let program = builder.build::<TestArray, TestArray>(vec![o0], Placeholder, Placeholder).unwrap();
        assert_eq!(program.interpret(TestArray::vector(vec![1.0, 2.0])).unwrap().values, vec![2.0, 4.0]);
        assert!(matches!(
            program.interpret(TestArray::vector(vec![1.0, 2.0, 3.0])),
            Err(ProgramError::Type(TypeError { message })) if message
                == "encountered input type f64[3] which is incompatible with the program's declared type f64[<3]",
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
                |instruction, inputs| instruction.operation.interpret(&EagerContext::new(), inputs),
            ),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        ));
    }

    #[test]
    fn test_program_interpret_with_wrong_number_of_operation_outputs() {
        let mut builder = ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new();
        let i0 = builder.add_input(DataType::F64);
        let o0 = builder.add_instruction(ScaleOperation::new(2.0), vec![i0]).unwrap()[0];
        let program = builder.build::<f64, f64>(vec![o0], Placeholder, Placeholder).unwrap();
        assert!(matches!(
            program.interpret_with(
                vec![2.0f64],
                |_, value| Ok(*value),
                |_, _| Ok::<Vec<f64>, ProgramError>(Vec::new()),
            ),
            Err(ProgramError::InvalidOutputCount { expected: 1, actual: 0 }),
        ));
    }

    #[test]
    fn test_program_inputs_mask() {
        let mut builder = ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new();
        let first_input = builder.add_input(DataType::F64);
        let constant = builder.add_constant(3.0f64);
        let second_input = builder.add_input(DataType::F64);
        let scaled = builder.add_instruction(ScaleOperation::new(2.0), vec![first_input]).unwrap()[0];
        let output = builder.add_instruction(AddOperation, vec![scaled, second_input]).unwrap()[0];
        let program =
            builder.build::<Vec<f64>, f64>(vec![output], vec![Placeholder, Placeholder], Placeholder).unwrap();
        assert_eq!(
            program.inputs_mask(),
            vec![
                true,  // `first_input`
                false, // `constant`
                true,  // `second_input`
                false, // `scaled`
                false, // `output`
            ],
        );
        assert_eq!(constant, AtomId { index: 1 });
    }

    #[test]
    fn test_program_instruction_by_output() {
        let mut builder = ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new();
        let input = builder.add_input(DataType::F64);
        let constant = builder.add_constant(3.0f64);
        let scaled = builder.add_instruction(ScaleOperation::new(2.0), vec![input]).unwrap()[0];
        let output = builder.add_instruction(AddOperation, vec![scaled, constant]).unwrap()[0];
        let dead_output = builder.add_instruction(ScaleOperation::new(4.0), vec![input]).unwrap()[0];
        let program = builder.build::<f64, f64>(vec![output], Placeholder, Placeholder).unwrap();

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
        assert_eq!(dead_output, AtomId { index: 4 });
    }

    #[test]
    fn test_program_live_sets() {
        let mut builder = ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new();
        let live_input = builder.add_input(DataType::F64);
        let dead_input = builder.add_input(DataType::F64);
        let live_constant = builder.add_constant(3.0f64);
        let dead_constant = builder.add_constant(5.0f64);
        let scaled = builder.add_instruction(ScaleOperation::new(2.0), vec![live_input]).unwrap()[0];
        let output = builder.add_instruction(AddOperation, vec![scaled, live_constant]).unwrap()[0];
        let dead_output = builder.add_instruction(AddOperation, vec![dead_input, dead_constant]).unwrap()[0];
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
        assert_eq!(dead_output, AtomId { index: 6 });

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
        let mut builder = ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new();
        let input = builder.add_input(DataType::F64);
        let constant = builder.add_constant(3.0f64);
        let scaled = builder.add_instruction(ScaleOperation::new(2.0), vec![input]).unwrap()[0];
        let output = builder.add_instruction(AddOperation, vec![scaled, constant]).unwrap()[0];
        let program = builder.build::<f64, f64>(vec![output], Placeholder, Placeholder).unwrap();
        let mapped = program
            .map_operations(|operation| {
                Ok::<_, ProgramError>(match operation {
                    ScalarOperation::Scale(operation) => {
                        ScalarOperation::Scale(ScaleOperation::new(operation.factor() + 2.0))
                    }
                    ScalarOperation::Add(_) => ScalarOperation::Mul(MulOperation),
                    operation => operation.clone(),
                })
            })
            .unwrap();
        assert_eq!(program.interpret(2.0f64), Ok(7.0f64));
        assert_eq!(mapped.interpret(2.0f64), Ok(24.0f64));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = const
                    %2:f64 = scale [factor=2] %0
                    %3:f64 = add %2 %1
                in (%3)
            "}
            .trim_end(),
        );
        assert_eq!(
            mapped.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = const
                    %2:f64 = scale [factor=4] %0
                    %3:f64 = mul %2 %1
                in (%3)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_program_to_flat_program_and_into_flat_program() {
        let mut builder = ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new();
        let i0 = builder.add_input(DataType::F64);
        let i1 = builder.add_input(DataType::F64);
        let v0 = builder.add_instruction(ScaleOperation::new(2.0), vec![i0]).unwrap()[0];
        let o0 = builder.add_instruction(AddOperation, vec![v0, i1]).unwrap()[0];
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
        let _ = builder.add_instruction(AddOperation, vec![i0, c0]).unwrap()[0];
        let v1 = builder.add_instruction(AddOperation, vec![i0, c1]).unwrap()[0];
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

        impl Value<DataType> for CloneCountingValue {
            type InterpretationContext = EagerContext<DataType, Self>;

            #[inline]
            fn interpretation_context(&self) -> Option<Self::InterpretationContext> {
                Some(EagerContext::new())
            }
        }

        impl BooleanLike for CloneCountingValue {
            fn as_boolean(&self) -> Self {
                Self::new(if self.value != 0.0 { 1.0 } else { 0.0 }, Rc::clone(&self.clone_count))
            }

            fn boolean(&self) -> Result<bool, ProgramError> {
                Ok(self.value != 0.0)
            }
        }

        let value_clone_count = Rc::new(Cell::new(0));
        let mut builder = ProgramBuilder::<_, _, ScalarOperation<CloneCountingValue>>::new();
        let i0 = builder.add_input(DataType::F64);
        let c0 = builder.add_constant(CloneCountingValue::new(2.0, Rc::clone(&value_clone_count)));
        let c1 = builder.add_constant(CloneCountingValue::new(3.0, Rc::clone(&value_clone_count)));
        let v0 = builder.add_instruction(AddOperation, vec![i0, c0]).unwrap()[0];
        let v1 = builder.add_instruction(AddOperation, vec![i0, c1]).unwrap()[0];
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
    fn test_program_filtered() {
        let mut builder = ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new();
        let i0 = builder.add_input(DataType::F64);
        let i1 = builder.add_input(DataType::F64);
        let c0 = builder.add_constant(2.0f64);
        let v0 = builder.add_instruction(ScaleOperation::new(3.0), vec![i0]).unwrap()[0];
        let v1 = builder.add_instruction(AddOperation, vec![v0, c0]).unwrap()[0];
        let program = builder.build::<(f64, f64), f64>(vec![v1], (Placeholder, Placeholder), Placeholder).unwrap();

        // Dead inputs are pruned and constants are lifted: `i1` is dead for `v1`, so it is dropped,
        // and `c0` is rebuilt into the projected program.
        let (pruned, pruned_live) = program.filtered(&[i0, i1], &[v1]).unwrap();
        assert_eq!(pruned_live, vec![0]);
        assert_eq!(pruned.input_ids().len(), 1);
        assert_eq!(pruned.interpret(vec![4.0]), Ok(vec![14.0]));

        // Selecting an intermediate atom (i.e., `v0`) as the output drops the downstream `add`
        // and the now-dead constant.
        let (intermediate, intermediate_live) = program.filtered(&[i0], &[v0]).unwrap();
        assert_eq!(intermediate_live, vec![0]);
        assert_eq!(intermediate.instructions().len(), 1);
        assert_eq!(intermediate.interpret(vec![5.0]), Ok(vec![15.0]));

        // Forwarding an input directly as an output yields an instruction-free program over only that input.
        let (forwarded, forwarded_live) = program.filtered(&[i0, i1], &[i0]).unwrap();
        assert_eq!(forwarded_live, vec![0]);
        assert_eq!(forwarded.instructions().len(), 0);
        assert_eq!(forwarded.interpret(vec![7.0]), Ok(vec![7.0]));

        // Reaching a variable that is neither a selected input nor produced by an instruction is rejected:
        // `v1` depends on `i0`, which is omitted from the selected inputs here.
        assert!(matches!(program.filtered(&[i1], &[v1]), Err(ProgramError::MalformedProgram(_))));

        // Providing the same input atom more than once is rejected.
        assert!(matches!(program.filtered(&[i0, i0], &[v1]), Err(ProgramError::MalformedProgram(_))));
    }

    #[test]
    fn test_program_into_filtered() {
        // Build the same program twice, so that the consuming `into_filtered` can be compared
        // against the borrowing `filter`.
        let build = || {
            let mut builder = ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new();
            let i0 = builder.add_input(DataType::F64);
            let i1 = builder.add_input(DataType::F64);
            let c0 = builder.add_constant(2.0f64);
            let v0 = builder.add_instruction(ScaleOperation::new(3.0), vec![i0]).unwrap()[0];
            let v1 = builder.add_instruction(AddOperation, vec![v0, c0]).unwrap()[0];
            let program = builder.build::<(f64, f64), f64>(vec![v1], (Placeholder, Placeholder), Placeholder).unwrap();
            (program, i0, i1, v1)
        };

        let (borrowed_program, b_i0, b_i1, b_v1) = build();
        let (borrowed, borrowed_live) = borrowed_program.filtered(&[b_i0, b_i1], &[b_v1]).unwrap();
        let (owned_program, o_i0, o_i1, o_v1) = build();
        let (owned, owned_live) = owned_program.into_filtered(&[o_i0, o_i1], &[o_v1]).unwrap();

        // The consuming variant drops the dead input, lifts the constant, and is identical to the borrowing `filter`.
        assert_eq!(owned_live, vec![0]);
        assert_eq!(owned_live, borrowed_live);
        assert_eq!(owned.input_ids().len(), 1);
        assert_eq!(owned.interpret(vec![4.0]), Ok(vec![14.0]));
        assert_eq!(owned.to_string(), borrowed.to_string());
    }

    #[test]
    fn test_program_builder() {
        let mut builder = ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new();
        let i0 = builder.add_input(DataType::F64);
        let i1 = builder.add_input(DataType::F64);
        let c0 = builder.add_constant(2.0f64);
        let v0 = builder.add_instruction(ScaleOperation::new(2.0), vec![i0]).unwrap()[0];
        let v1 = builder.add_instruction(AddOperation, vec![v0, i1]).unwrap()[0];
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
        let v0 = builder.add_instruction(AddOperation, vec![AtomId { index: 42 }, AtomId { index: 99 }]);
        assert!(matches!(v0, Err(ProgramError::UnboundAtomId { id }) if id == AtomId { index: 42 }));
    }

    #[test]
    fn test_program_builder_build_returns_error() {
        let mut builder = ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new();
        builder.error = Some(ProgramError::InvalidInputCount { expected: 1, actual: 0 });
        assert!(matches!(
            builder.build::<f64, f64>(Vec::new(), Placeholder, Placeholder),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        ));
    }

    #[test]
    fn test_program_builder_build_rejects_invalid_input_count() {
        let builder = ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new();
        assert!(matches!(
            builder.build::<f64, ()>(Vec::new(), Placeholder, ()),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        ));
    }

    #[test]
    fn test_program_builder_build_rejects_invalid_output_count() {
        let mut builder = ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new();
        builder.add_input(DataType::F64);
        assert!(matches!(
            builder.build::<f64, f64>(Vec::new(), Placeholder, Placeholder),
            Err(ProgramError::InvalidOutputCount { expected: 1, actual: 0 }),
        ));
    }

    #[test]
    fn test_program_builder_build_rejects_malformed_atom_providers() {
        let mut duplicate_input_builder = ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new();
        let input = duplicate_input_builder.add_input(DataType::F64);
        duplicate_input_builder.input_ids.push(input);
        assert!(matches!(
            duplicate_input_builder.build::<Vec<f64>, Vec<f64>>(
                vec![input],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            ),
            Err(ProgramError::MalformedProgram(message))
                if message == format!("program input atom {input} appears more than once")
        ));

        let mut input_output_overlap_builder = ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new();
        let input = input_output_overlap_builder.add_input(DataType::F64);
        input_output_overlap_builder.add_instruction_unchecked(Instruction::new(
            ScalarOperation::Scale(ScaleOperation::new(2.0)),
            vec![input],
            vec![input],
        ));
        assert!(matches!(
            input_output_overlap_builder.build::<Vec<f64>, Vec<f64>>(
                vec![input],
                vec![Placeholder],
                vec![Placeholder],
            ),
            Err(ProgramError::MalformedProgram(message))
                if message == format!("instruction output atom {input} is a program input")
        ));

        let mut duplicate_output_builder = ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new();
        let input = duplicate_output_builder.add_input(DataType::F64);
        let output = duplicate_output_builder.add_variable(DataType::F64);
        duplicate_output_builder.add_instruction_unchecked(Instruction::new(
            ScalarOperation::Scale(ScaleOperation::new(2.0)),
            vec![input],
            vec![output],
        ));
        duplicate_output_builder.add_instruction_unchecked(Instruction::new(
            ScalarOperation::Scale(ScaleOperation::new(3.0)),
            vec![input],
            vec![output],
        ));
        assert!(matches!(
            duplicate_output_builder.build::<Vec<f64>, Vec<f64>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            ),
            Err(ProgramError::MalformedProgram(message))
                if message == format!("instruction output atom {output} is produced by more than one instruction")
        ));
    }
}
