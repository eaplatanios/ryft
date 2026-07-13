//! Contains machinery for representing and working with typed, structured, and effect-aware programs.
//!
//! A [`Program`] is Ryft's backend-neutral dataflow IR. It owns a flat arena of [`Region`]s that consists of the public
//! entry computation plus any nested computations referenced by its instructions, where each region stores typed
//! atoms, operation instructions, a flat boundary, and enough metadata for interpretation, transformation,
//! simplification, lowering, and compilation. Programs are immutable after construction. [`ProgramBuilder`]
//! owns the mutable construction phase, sealing every non-entry region before instructions can attach it.
//!
//! ```text
//! ┌─────────────────────────────┐
//! │ Abstract Inputs + Constants │
//! └──────────────┬──────────────┘
//!                │ add atoms and record instructions
//!                ▼
//!       ┌─────────────────┐
//!       │ Program Builder │
//!       └────────┬────────┘
//!                │ build structured boundaries
//!                ▼
//!           ┌─────────┐
//!           │ Program │
//!           └────┬────┘
//!                ├── interpret through a context
//!                ├── batch, differentiate, or partially evaluate
//!                ├── simplify, filter, or inspect liveness and effects
//!                └── lower and compile through a backend
//! ```
//!
//! # Entry Points
//!
//! Most code obtains programs through [`trace`](crate::trace) or a transform rather than by manual construction, and
//! replays them with [`Program::interpret`] (eagerly) or [`Program::interpret_in_context`] (through a chosen staging
//! or transform context). Batching, differentiation, and partial evaluation add program-level functions in their own
//! modules, and compilation opens captures, flattens boundaries, and hands the program to a backend
//! [`CompilationDomain`](crate::CompilationDomain).
//!
//! Direct [`ProgramBuilder`] use is appropriate for operation and transform infrastructure. Tracer operations call
//! [`ProgramBuilder::add_instruction`], which infers output types, allocates variable atoms, validates arity, and
//! records the instruction, and [`ProgramBuilder::build`] validates the requested boundaries and freezes the result.
//! Keep [`AtomId`]s from one builder isolated from every other builder, use the checked instruction path, and
//! propagate the builder's first stored error rather than continuing with invalid IDs.
//!
//! [`Program::to_flat_program`] converts structured boundaries to vectors without changing the dataflow. Use it at
//! internal compiler or nested-program boundaries, and preserve the structured form in user-facing APIs.
//!
//! # Core Data Model
//!
//! [`Value`] is the common contract for leaf values that can inhabit programs or flow through Ryft contexts. Every
//! value has one associated type descriptor through [`Typed`], plus separate dispatch and execution domains used by
//! capabilities and transforms.
//!
//! [`Atom`] is either a stored constant or a typed variable, and [`AtomId`] is its stable index in the containing
//! [`Region`]'s atom table. An [`Instruction`] owns one [`Operation`], lists the input and output atom IDs of that
//! application, and carries the [`RegionId`]s of its attached nested regions, in the operation-defined order.
//! Operations define their own type inference and effect classes and the program supplies graph structure and order.
//!
//! [`Program`] combines the region arena with typed, structured input and output boundaries on its entry region. The
//! boundary types are [`Parameterized`] containers whose leaves correspond positionally to [`Program::input_ids`] and
//! [`Program::output_ids`], so compiler and transform kernels can operate on the flat IDs while callers retain tuples,
//! vectors, maps, or derived product types. [`InstructionId`] and [`ValueId`] locate instructions and values across
//! regions, and [`InstructionRef`] resolves an instruction's attached regions against the arena.
//!
//! # Effects, Liveness, and Simplification
//!
//! [`Program::effects`] unions the effects declared by its operations. Instruction order is semantically relevant
//! for ordered effects even when the dataflow graph contains no dependency between them.
//!
//! [`Program::live_sets`] computes the atoms and instructions required by selected roots. [`Program::simplified`]
//! removes dead pure work while retaining effectful instructions as roots. [`Program::filtered`] projects a program
//! to selected boundaries and accepts explicit keep-alive atoms for work that must survive the projection. These APIs
//! preserve the invariants checked by normal program construction rather than treating effects as ordinary unused
//! values.
//!
//! # Extending Programs
//!
//! New primitive behavior normally means adding an operation payload implementing [`Operation`] and including it in the
//! appropriate closed operation family. Keep type inference, rendering, effects, and operation-specific transform rules
//! with that payload, and never teach [`Program`] about individual operation variants.

use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Arc;

use thiserror::Error;

use ryft_macros::Parameter;

use crate::contexts::{Context, Domain};
use crate::effects::Effects;
use crate::errors::CustomError;
use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::constants::Zero;
use crate::parameters::{Parameter, ParameterError, Parameterized, ParameterizedFamily, Placeholder};
use crate::types::{TypeError, Typed};

/// Represents errors related to [`Program`]s in `ryft-core`.
#[derive(Clone, Debug, Error, PartialEq, Eq, Hash)]
pub enum ProgramError {
    #[error("values used in the same operation must share the same program builder")]
    MismatchedProgramBuilders,

    #[error("{message}")]
    InvalidArgument { message: String },

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
/// tracing wrappers such as [`Tracer`](crate::Tracer). It inherits its type descriptor from [`Typed`], so generic code
/// recovers the descriptor as `V::Type` and pinning sites write `V: Value<Type = ArrayType>`. It additionally requires
/// [`Debug`] and [`Display`] so that diagnostics, constants, and [`Operation`] metadata can render their carried
/// values directly.
pub trait Value: Clone + Debug + Display + Parameter + Typed + Sized {
    /// [`Domain`] that operations involving this [`Value`] *dispatch* through. Every value names two domains:
    /// capability function calls dispatch through the [`DispatchDomain`](Self::DispatchDomain), while transform work
    /// executes in the [`ExecutionDomain`](Self::ExecutionDomain). The two domains coincide for every transform and
    /// staged value (e.g., a staged [`Tracer`](crate::Tracer)'s trace, a [`BatchingTracer`](crate::BatchingTracer)'s
    /// batching level, etc.): dispatch and execution both happen in the live context such a value flows through.
    /// However, they become separate for concrete backend values (e.g., concrete arrays). In those cases, the
    /// [`DispatchDomain`](Self::DispatchDomain) is the constant-only [`EagerContext`](crate::EagerContext) such that
    /// capability calls dispatch to direct implementations instead of a context, while the
    /// [`ExecutionDomain`](Self::ExecutionDomain) names the backend's *rich*, operation-executing eager domain. Backend
    /// values whose rich domain requires state or defaults that cannot be derived from a value (e.g., a client handle)
    /// keep the constant-only domain here too, which simply means free transform entry points do not serve them and an
    /// explicit context must be used instead.
    ///
    /// Blanket capability implementations (e.g., the value-level arithmetic sugar) bind through this domain and use its
    /// operation universe as their coherence discriminator: the sugar applies when `V::DispatchDomain::Operation` can
    /// accept the operation being bound. A staged [`Tracer`](crate::Tracer)'s dispatch domain is its live trace, so the
    /// sugar records instructions there. A concrete backend value's dispatch domain is the constant-only
    /// [`EagerContext`](crate::EagerContext), whose [`ConstantOperation`](crate::ConstantOperation) universe accepts
    /// nothing. This is precisely what keeps the blanket implementations coherent with (i.e., disjoint from) the direct
    /// capability implementations that concrete values provide instead.
    type DispatchDomain: Domain<Type = Self::Type, Value = Self>;

    /// [`Domain`] that transform work involving this [`Value`] *executes* in. Refer to the documentation of
    /// [`DispatchDomain`](Self::DispatchDomain) for information on the two types of [`Domain`]s that each value
    /// provides.
    type ExecutionDomain: Domain<Type = Self::Type, Value = Self>;

    /// Returns the [`Domain`] that operations involving this [`Value`] *dispatch* through. Refer to the
    /// documentation of [`DispatchDomain`](Self::DispatchDomain) for more information.
    fn dispatch_domain(&self) -> Self::DispatchDomain;

    /// Returns the [`Domain`] that transform work involving this [`Value`] *executes* in. Refer to the
    /// documentation of [`ExecutionDomain`](Self::ExecutionDomain) for more information.
    fn execution_domain(&self) -> Self::ExecutionDomain;
}

/// Represents either a [`Typed`] value or a _structural zero_ that carries only its [`Type`](crate::Type).
/// [`MaybeZero`] is the symbolic-zero representation shared by transforms like forward-mode and reverse-mode
/// differentiation where it is the tangent type carried by [`DifferentiationTracer`](crate::DifferentiationTracer)s
/// and the cotangent type that transposition rules consume and produce. A [`MaybeZero::Zero`] means that no value
/// exists and nothing has been staged or computed for it. In the context of differentiation, it means that the
/// corresponding derivative is zero *by construction* (e.g., a disconnected input, a severed tangent, an unused output,
/// etc.), and is not a runtime value that happens to contain zeros. Differentiation rules branch on the variant to skip
/// work entirely. A rule that sees a zero tangent or cotangent emits no operations for it, and "zero-ness" propagates
/// transitively through rules without ever inspecting a program or materializing a buffer. A zero is _materialized_
/// into a real value only at the boundaries where one is structurally required (e.g., a nested sub-program operand,
/// a program output, or an eagerly returned tangent), which is also where its carried [`Type`](crate::Type)
/// is consumed.
#[derive(Clone, Debug)]
pub enum MaybeZero<V: Typed> {
    /// Structural zero of the carried [`Type`](crate::Type) (i.e., no value exists and nothing has been staged
    /// or computed for it).
    Zero(V::Type),

    /// Value that is not known to be structurally equal to zero.
    Value(V),
}

impl<V: Typed> MaybeZero<V> {
    /// Returns `true` if this is a [`MaybeZero::Zero`].
    #[inline]
    pub const fn is_zero(&self) -> bool {
        matches!(self, Self::Zero(_))
    }

    /// Returns the value stored in this [`MaybeZero`], if it is a [`MaybeZero::Value`], and [`None`] otherwise.
    #[inline]
    pub const fn as_value(&self) -> Option<&V> {
        match self {
            Self::Zero(_) => None,
            Self::Value(value) => Some(value),
        }
    }

    /// Maps the value stored in this [`MaybeZero`] using the provided function, leaving a [`MaybeZero::Zero`] and
    /// its carried [`Type`](crate::Type) unchanged. If this is [`MaybeZero::Zero`], then [`MaybeZero::Zero`] will be
    /// returned irrespective of what `function` is provided.
    #[inline]
    pub fn map<W: Typed<Type = V::Type>, F: FnOnce(V) -> W>(self, function: F) -> MaybeZero<W> {
        match self {
            Self::Zero(r#type) => MaybeZero::Zero(r#type),
            Self::Value(value) => MaybeZero::Value(function(value)),
        }
    }

    /// Returns the value inside this [`MaybeZero`], materializing a structural [`MaybeZero::Zero`] as a real typed
    /// zero value in the provided [`Context`] through its [`Zero`] capability (a staging context stages a typed
    /// [`ZeroOperation`](crate::ZeroOperation) instruction, while an eager context constructs a concrete zero value).
    #[inline]
    pub fn materialize<C: Context<Value = V> + Zero<V>>(self, context: &C) -> Result<V, ProgramError> {
        match self {
            Self::Value(value) => Ok(value),
            Self::Zero(r#type) => context.zero(&r#type),
        }
    }
}

impl<V: Typed> Typed for MaybeZero<V> {
    type Type = V::Type;

    #[inline]
    fn r#type(&self) -> Cow<'_, V::Type> {
        match self {
            Self::Zero(r#type) => Cow::Borrowed(r#type),
            Self::Value(value) => value.r#type(),
        }
    }
}

impl<V: Typed> From<V> for MaybeZero<V> {
    #[inline]
    fn from(value: V) -> Self {
        Self::Value(value)
    }
}

/// [`Atom`]s represent nodes in the [`Region`]s of [`Program`]s that represent either concrete values or variables
/// of specific [`Type`](crate::Type)s.
#[derive(Clone, Debug, Parameter)]
pub enum Atom<V: Typed> {
    /// Literal constant value that appears in a [`Program`].
    Constant(V),

    /// Non-constant variable of a specific [`Type`](crate::Type) that appears in a [`Program`].
    Variable(V::Type),
}

impl<V: Typed> Atom<V> {
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

impl<V: Typed> Typed for Atom<V> {
    type Type = V::Type;

    fn r#type(&self) -> Cow<'_, V::Type> {
        match self {
            Self::Constant(value) => value.r#type(),
            Self::Variable(r#type) => Cow::Borrowed(r#type),
        }
    }
}

/// Unique identifier for an [`Atom`] within one [`Region`] of a [`Program`]. [`AtomId`]s are stable indexes into the
/// containing [`Region`]'s atom table (every region owns its own table, so an [`AtomId`] is meaningful only together
/// with its region). [`Instruction`]s refer to their inputs and outputs by these IDs, which keeps the intermediate
/// representation compact and easy to clone.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Parameter)]
pub struct AtomId {
    /// Zero-based index of the corresponding [`Atom`] inside the containing [`Region`]'s atom table.
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
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "%{}", self.index)
    }
}

/// Unique identifier for a [`Region`] within a [`Program`]. [`RegionId`]s are stable indexes into a [`Program`]'s
/// region arena. Like [`AtomId`]s, they are meaningful only against the [`Program`] they were derived from.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RegionId {
    /// Zero-based index of the corresponding [`Region`] inside the owning [`Program`]'s region arena.
    index: usize,
}

impl RegionId {
    /// Creates a new [`RegionId`] from the provided zero-based region-arena index.
    #[inline]
    pub fn new(index: usize) -> Self {
        Self { index }
    }

    /// Returns the zero-based index of the corresponding [`Region`] inside the owning [`Program`]'s region arena.
    #[inline]
    pub fn index(self) -> usize {
        self.index
    }
}

impl Display for RegionId {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "^{}", self.index)
    }
}

/// Location of one [`Instruction`] in a multi-region [`Program`], identified by its containing [`Region`] and its
/// zero-based index within that region's instruction sequence.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct InstructionId {
    /// [`Region`] containing the instruction.
    region: RegionId,

    /// Zero-based instruction index within the containing [`Region`].
    index: usize,
}

impl InstructionId {
    /// Creates a new [`InstructionId`] from the provided containing region and instruction index.
    #[inline]
    pub fn new(region: RegionId, index: usize) -> Self {
        Self { region, index }
    }

    /// Returns the [`RegionId`] of the [`Region`] containing the instruction.
    #[inline]
    pub fn region(self) -> RegionId {
        self.region
    }

    /// Returns the zero-based instruction index within the containing [`Region`].
    #[inline]
    pub fn index(self) -> usize {
        self.index
    }
}

/// Location of one Single Static Assignment (SSA) value in a multi-region [`Program`], identified by its containing
/// [`Region`] and its region-local [`AtomId`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ValueId {
    /// [`Region`] containing the atom.
    region: RegionId,

    /// Region-local [`AtomId`] of the value.
    atom: AtomId,
}

impl ValueId {
    /// Creates a new [`ValueId`] from the provided containing region and region-local atom identifier.
    #[inline]
    pub fn new(region: RegionId, atom: AtomId) -> Self {
        Self { region, atom }
    }

    /// Returns the [`RegionId`] of the [`Region`] containing the atom.
    #[inline]
    pub fn region(self) -> RegionId {
        self.region
    }

    /// Returns the region-local [`AtomId`] of the value.
    #[inline]
    pub fn atom(self) -> AtomId {
        self.atom
    }
}

/// Computation region inside a [`Program`]'s region arena. Every region owns its own [`Atom`] table, [`Instruction`]
/// sequence, and input/output boundary. The public program entry point and every nested computation are [`Region`]s in
/// the same arena. [`Instruction`]s reference them by [`RegionId`], and one region may be shared. Regions are _sealed_
/// meaning that a [`ProgramBuilder`] creates them only by copying already-built (and therefore immutable and fully
/// validated) [`Program`]s via [`ProgramBuilder::add_region`] and [`ProgramBuilder::add_callee`], and so a region
/// referenced by an instruction can never change after that instruction is built.
#[derive(Clone, Debug)]
pub struct Region<V: Typed, O> {
    /// [`Atom`]s contained in this [`Region`], in the order in which they will be evaluated.
    pub(crate) atoms: Vec<Atom<V>>,

    /// [`AtomId`]s of the [`Atom`]s that correspond to the inputs of this [`Region`].
    pub(crate) input_ids: Vec<AtomId>,

    /// [`AtomId`]s of the [`Atom`]s that correspond to the outputs of this [`Region`].
    pub(crate) output_ids: Vec<AtomId>,

    /// Ordered sequence of [`Instruction`]s that make up the computational graph of this [`Region`].
    pub(crate) instructions: Vec<Instruction<O>>,
}

impl<V: Typed, O> Region<V, O> {
    /// Returns the [`Atom`]s contained in this [`Region`], in the order in which they will be evaluated.
    #[inline]
    pub fn atoms(&self) -> &[Atom<V>] {
        &self.atoms
    }

    /// Returns the [`AtomId`]s of the [`Atom`]s that correspond to the inputs of this [`Region`].
    #[inline]
    pub fn input_ids(&self) -> &[AtomId] {
        &self.input_ids
    }

    /// Returns the [`Type`](crate::Type)s of the inputs of this [`Region`], in boundary order.
    #[inline]
    pub fn input_types(&self) -> Vec<V::Type> {
        self.input_ids.iter().map(|input| self.atoms[input.index()].r#type().into_owned()).collect()
    }

    /// Returns the [`AtomId`]s of the [`Atom`]s that correspond to the outputs of this [`Region`].
    #[inline]
    pub fn output_ids(&self) -> &[AtomId] {
        &self.output_ids
    }

    /// Returns the [`Type`](crate::Type)s of the outputs of this [`Region`], in boundary order.
    #[inline]
    pub fn output_types(&self) -> Vec<V::Type> {
        self.output_ids.iter().map(|output| self.atoms[output.index()].r#type().into_owned()).collect()
    }

    /// Returns the ordered sequence of [`Instruction`]s that make up the computational graph of this [`Region`].
    #[inline]
    pub fn instructions(&self) -> &[Instruction<O>] {
        &self.instructions
    }
}

/// [`Instruction`]s represent applications of [`Operation`]s to input values in [`Program`]s. Each [`Region`] executes
/// its [`Instruction`]s in sequential order. Beyond its operation and its input and output [`Atom`]s, an instruction
/// carries the [`RegionId`]s of the nested computations attached to the application (e.g., the `true`/`false`
/// branches of a condition, a scan body, or the shared program of a JIT call), in the operation-defined order.
///
/// There is one region edge kind, and sharing is expressed directly in the graph. Several instructions may reference
/// the same [`RegionId`], and a region stays alive for as long as it is reachable from the entry region. What a slot
/// *means* (i.e., a branch-like computation that lowers inline versus a call-like computation that lowers and
/// compiles once as a shared function) is defined by the operation and not by the edge. For example, `if p { f(x) +
/// f(2 * x) } else { x }` with a JIT-compiled `f` is one condition instruction attaching a `true` and a `false` branch
/// [`Region`], where the `true` branch contains two call instructions that both reference the single region holding
/// `f`'s body (i.e., one shared region, three region edges, and the inline-versus-shared lowering decision carried by
/// the condition and call operations, respectively). Two structurally equal but independently created computations
/// remain distinct regions, because [`ProgramBuilder`] adds regions by *identity* (i.e.,
/// [`add_region`](ProgramBuilder::add_region) always copies and [`add_callee`](ProgramBuilder::add_callee)
/// interns by [`Rc`] identity), never by structure.
#[derive(Clone, Debug)]
pub struct Instruction<O> {
    /// [`Operation`] applied by this [`Instruction`].
    operation: O,

    /// [`AtomId`]s of the input [`Atom`]s consumed by this [`Instruction`].
    inputs: Vec<AtomId>,

    /// [`AtomId`]s of the output [`Atom`]s produced by this [`Instruction`].
    outputs: Vec<AtomId>,

    /// [`RegionId`]s of the nested computations attached to this [`Instruction`], in the operation-defined order.
    regions: Vec<RegionId>,
}

impl<O> Instruction<O> {
    /// Creates a new [`Instruction`].
    #[inline]
    pub fn new(operation: O, inputs: Vec<AtomId>, outputs: Vec<AtomId>, regions: Vec<RegionId>) -> Self {
        Self { operation, inputs, outputs, regions }
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

    /// Returns the [`RegionId`]s of the nested computations attached to this [`Instruction`],
    /// in the operation-defined order.
    #[inline]
    pub fn regions(&self) -> &[RegionId] {
        &self.regions
    }

    /// Consumes this [`Instruction`] and returns its [`Operation`], input [`AtomId`]s, output [`AtomId`]s,
    /// and attached region [`RegionId`]s.
    #[inline]
    pub fn into_parts(self) -> (O, Vec<AtomId>, Vec<AtomId>, Vec<RegionId>) {
        (self.operation, self.inputs, self.outputs, self.regions)
    }
}

/// Borrowed view of one [`Instruction`] resolved against its owning [`Program`]'s [`Region`] arena. Unlike a bare
/// [`Instruction`], this view can resolve the instruction's attached regions, so that analyses can traverse them
/// without holding the owning [`Program`] separately.
pub struct InstructionRef<'r, V: Typed, O> {
    /// Borrowed [`Region`] arena used to resolve this instruction and its attached regions.
    regions: &'r [Region<V, O>],

    /// [`InstructionId`] specifying the [`Instruction`]'s location in the owning [`Program`].
    id: InstructionId,
}

impl<'r, V: Typed, O> InstructionRef<'r, V, O> {
    /// Returns the [`InstructionId`] of this instruction.
    #[inline]
    pub fn id(&self) -> InstructionId {
        self.id
    }

    /// Returns the underlying [`Instruction`].
    #[inline]
    pub fn instruction(&self) -> &'r Instruction<O> {
        &self.regions[self.id.region().index()].instructions[self.id.index()]
    }

    // TODO(eaplatanios): Do we really need this function?
    //  And if not, should `InstructionRef` be replaced by simply `&Instruction`?
    /// Returns the [`Region`] attached at the provided index.
    #[inline]
    pub fn region(&self, index: usize) -> Result<&'r Region<V, O>, ProgramError> {
        let id = self
            .instruction()
            .regions()
            .get(index)
            .copied()
            .ok_or_else(|| ProgramError::MalformedProgram(format!("instruction has no region index {index}")))?;
        Ok(&self.regions[id.index()])
    }
}

/// [`Program`] that is produced by tracing and which can be interpreted or compiled and executed by a backend.
/// A program owns a flat arena of [`Region`]s. One region implements its public entry point, and every other region
/// is a nested computation referenced by one or more [`Instruction`]s (e.g., the branches of a condition, or the
/// shared program of a JIT call). Each region is a flat sequence of [`Instruction`]s over its own [`Atom`] table, and the
/// entry region's flat boundary is paired with [`Parameterized`] input and output types. This is the primary
/// intermediate representation (IR) used by the Ryft tracing and transformation system (e.g., to support things
/// like automatic differentiation and just-in-time compilation).
#[derive(Debug)]
pub struct Program<V: Typed + Parameter, O, Input: Parameterized<V>, Output: Parameterized<V>> {
    /// [`Parameter`] structure that can be used to map flat lists of inputs to structured `Input` values.
    pub(crate) input_structure: Input::ParameterStructure,

    /// [`Parameter`] structure that can be used to map flat lists of outputs to structured `Output` values.
    pub(crate) output_structure: Output::ParameterStructure,

    /// [`Region`] arena containing the public entry computation and every nested computation.
    pub(crate) regions: Vec<Region<V, O>>,

    /// [`RegionId`] of the [`Region`] implementing this [`Program`]'s public entry point.
    pub(crate) entry: RegionId,

    /// [`PhantomData`] marker that ties this [`Program`] to its structured `Input` and `Output` types
    /// without making it own either value family.
    pub(crate) marker: PhantomData<(Input, Output)>,
}

impl<V: Value, O: Operation<V::Type>, Input: Parameterized<V>, Output: Parameterized<V>> Program<V, O, Input, Output> {
    /// Returns the [`Atom`]s contained in this [`Program`]'s entry [`Region`],
    /// in the order in which they will be evaluated.
    #[inline]
    pub fn atoms(&self) -> &[Atom<V>] {
        &self.entry_region().atoms
    }

    /// Returns the number of input [`Atom`]s (i.e., arguments) of this [`Program`].
    #[inline]
    pub fn input_count(&self) -> usize {
        self.entry_region().input_ids.len()
    }

    /// Returns the [`AtomId`]s of the [`Atom`]s that correspond to the inputs (i.e., arguments) of this [`Program`].
    #[inline]
    pub fn input_ids(&self) -> &[AtomId] {
        &self.entry_region().input_ids
    }

    /// Returns the [`Type`](crate::Type)s of the inputs of this [`Program`], in order.
    #[inline]
    pub fn input_types(&self) -> Vec<V::Type> {
        self.inputs().map(|input| input.r#type().into_owned()).collect()
    }

    /// Returns the [`Atom`]s that correspond to the inputs of this [`Program`].
    #[inline]
    pub fn inputs(&self) -> impl Iterator<Item = &Atom<V>> {
        let entry = self.entry_region();
        entry.input_ids.iter().map(|input_id| &entry.atoms[input_id.index])
    }

    /// Returns the structured `Input` of this [`Program`] parameterized by the corresponding [`Atom`]s.
    #[inline]
    pub fn input(&self) -> Result<Input::To<Atom<V>>, ParameterError>
    where
        Input::Family: ParameterizedFamily<Atom<V>>,
    {
        Input::To::<Atom<V>>::from_parameters(self.input_structure.clone(), self.inputs().cloned())
    }

    /// Returns the number of output [`Atom`]s (i.e., return values) of this [`Program`].
    #[inline]
    pub fn output_count(&self) -> usize {
        self.entry_region().output_ids.len()
    }

    /// Returns the [`AtomId`]s of the [`Atom`]s that correspond to the outputs (i.e., return values)
    /// of this [`Program`].
    #[inline]
    pub fn output_ids(&self) -> &[AtomId] {
        &self.entry_region().output_ids
    }

    /// Returns the [`Type`](crate::Type)s of the outputs of this [`Program`], in order.
    #[inline]
    pub fn output_types(&self) -> Vec<V::Type> {
        self.outputs().map(|output| output.r#type().into_owned()).collect()
    }

    /// Returns the [`Atom`]s that correspond to the outputs of this [`Program`].
    #[inline]
    pub fn outputs(&self) -> impl Iterator<Item = &Atom<V>> {
        let entry = self.entry_region();
        entry.output_ids.iter().map(|output_id| &entry.atoms[output_id.index])
    }

    /// Returns the structured `Output` of this [`Program`] parameterized by the corresponding [`Atom`]s.
    #[inline]
    pub fn output(&self) -> Result<Output::To<Atom<V>>, ParameterError>
    where
        Output::Family: ParameterizedFamily<Atom<V>>,
    {
        Output::To::<Atom<V>>::from_parameters(self.output_structure.clone(), self.outputs().cloned())
    }

    /// Returns the ordered sequence of [`Instruction`]s that make up the computational graph of this [`Program`]'s
    /// entry [`Region`].
    #[inline]
    pub fn instructions(&self) -> &[Instruction<O>] {
        &self.entry_region().instructions
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

    /// Returns the [`Region`]s in this [`Program`].
    #[inline]
    pub fn regions(&self) -> &[Region<V, O>] {
        &self.regions
    }

    /// Returns the [`Region`] that corresponds to the provided [`RegionId`].
    #[inline]
    pub fn region(&self, id: RegionId) -> Result<&Region<V, O>, ProgramError> {
        self.regions
            .get(id.index())
            .ok_or_else(|| ProgramError::MalformedProgram(format!("region {id} is out of range")))
    }

    /// Returns the [`RegionId`] of the [`Region`] implementing this [`Program`]'s public entry point.
    #[inline]
    pub fn entry(&self) -> RegionId {
        self.entry
    }

    /// Returns the entry [`Region`] of this [`Program`].
    #[inline]
    pub fn entry_region(&self) -> &Region<V, O> {
        &self.regions[self.entry.index()]
    }

    // TODO(eaplatanios): Delete this once the first-class program regions plan implementation is complete.
    /// Returns a mutable reference to the entry [`Region`] of this [`Program`].
    #[inline]
    pub(crate) fn entry_region_mut(&mut self) -> &mut Region<V, O> {
        &mut self.regions[self.entry.index()]
    }

    /// Returns the [`InstructionId`] of the instruction producing the provided value, or [`None`] when the value is
    /// a region input or constant. Returns an error when the locator does not resolve against this [`Program`].
    pub fn producer(&self, value: ValueId) -> Result<Option<InstructionId>, ProgramError> {
        let region = self.region(value.region())?;
        if region.atoms.get(value.atom().index()).is_none() {
            return Err(ProgramError::UnboundAtomId { id: value.atom() });
        }
        Ok(region.instructions.iter().enumerate().find_map(|(index, instruction)| {
            instruction.outputs.contains(&value.atom()).then_some(InstructionId::new(value.region(), index))
        }))
    }

    /// Returns a borrowed [`InstructionRef`] view of the [`Instruction`] at the provided [`InstructionId`], which can
    /// resolve the instruction's attached regions against this [`Program`]'s region arena.
    pub fn instruction(&self, id: InstructionId) -> Result<InstructionRef<'_, V, O>, ProgramError> {
        let region = self.region(id.region())?;
        if region.instructions.get(id.index()).is_none() {
            return Err(ProgramError::MalformedProgram(format!(
                "instruction {} of region {} is out of range",
                id.index(),
                id.region(),
            )));
        }
        Ok(InstructionRef { regions: &self.regions, id })
    }

    /// Returns a boolean mask that has the same length as the number of [`Atom`]s in this [`Program`] and contains the
    /// value `true` for atoms that are inputs of the program, and `false` for other atoms.
    pub fn inputs_mask(&self) -> Vec<bool> {
        let entry = self.entry_region();
        let mut inputs_mask = vec![false; entry.atoms.len()];
        for input in entry.input_ids.iter().copied() {
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
        let entry = self.entry_region();
        let mut instruction_by_output = vec![None; entry.atoms.len()];
        for (instruction_index, instruction) in entry.instructions.iter().enumerate() {
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
        self.live_sets_for_atoms(self.output_ids()).unwrap()
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
        F: FnMut(&Program<V, O, Input, Output>, &Instruction<O>, &[bool], &mut Vec<bool>) -> Result<(), ProgramError>,
    >(
        &self,
        propagate_liveness: F,
    ) -> Result<ProgramLiveSets, ProgramError> {
        self.live_sets_for_atoms_with(self.output_ids(), propagate_liveness)
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
        F: FnMut(&Program<V, O, Input, Output>, &Instruction<O>, &[bool], &mut Vec<bool>) -> Result<(), ProgramError>,
    >(
        &self,
        output_ids: &[AtomId],
        mut propagate_liveness: F,
    ) -> Result<ProgramLiveSets, ProgramError> {
        let entry = self.entry_region();
        let mut live_sets = ProgramLiveSets::new(vec![false; entry.atoms.len()], vec![false; entry.instructions.len()]);
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
        for (instruction_index, instruction) in entry.instructions.iter().enumerate().rev() {
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

    /// Returns the [`Effect`](crate::Effect) classes of this [`Program`] which is the [union](Effects::union) of its
    /// [`Instruction`]s' [`Operation::effects`] sets, or [`Effects::PURE`] for [`Program`]s with no instructions.
    /// Operations with nested programs report the [`Effects`] returned by this function for their nested programs as
    /// their own [`Operation::effects`] set so that effects remain visible through higher-order boundaries.
    #[inline]
    pub fn effects(&self) -> Effects {
        self.regions
            .iter()
            .flat_map(|region| region.instructions.iter())
            .map(|instruction| instruction.operation().effects())
            .fold(Effects::PURE, Effects::union)
    }

    /// Rebuilds this [`Program`] with each [`Operation`] mapped using the provided `map_fn`. The atom table,
    /// input/output atom identifiers, and parameter structures are preserved exactly. This is useful for transforms
    /// that keep the same value graph but need to change operation payloads. For example, a reusable residualized
    /// linear program may contain operations whose scale/dot factors are residual references rather than executable
    /// values. Before interpreting that program, the mapping closure can receive each linear operation, call the
    /// operation's factor-mapping hook, and replace each residual reference with the concrete residual value captured
    /// by the corresponding linearization run.
    pub fn map_operations<P: Operation<V::Type>, F: FnMut(&O) -> Result<P, ProgramError>>(
        &self,
        mut map_fn: F,
    ) -> Result<Program<V, P, Input, Output>, ProgramError> {
        Ok(Program {
            input_structure: self.input_structure.clone(),
            output_structure: self.output_structure.clone(),
            regions: self
                .regions
                .iter()
                .map(|region| {
                    Ok(Region {
                        atoms: region.atoms.clone(),
                        input_ids: region.input_ids.clone(),
                        output_ids: region.output_ids.clone(),
                        instructions: region
                            .instructions
                            .iter()
                            .map(|instruction| {
                                Ok(Instruction::new(
                                    map_fn(instruction.operation())?,
                                    instruction.inputs().to_vec(),
                                    instruction.outputs().to_vec(),
                                    instruction.regions().to_vec(),
                                ))
                            })
                            .collect::<Result<Vec<_>, ProgramError>>()?,
                    })
                })
                .collect::<Result<Vec<_>, ProgramError>>()?,
            entry: self.entry,
            marker: PhantomData,
        })
    }

    /// Returns a cloned view of this [`Program`] whose public input and output types are flat vectors. The atom table,
    /// input atom identifiers, output atom identifiers, and instruction sequence are preserved exactly. Only the
    /// `Input` and `Output` type parameters change to `Vec<V>`, with placeholder structures sized to the flat input and
    /// output arities. This is useful for higher-order operations that store nested [`Program`]s as operation payloads
    /// and replay them positionally, without needing to preserve the caller's original [`Parameterized`] type.
    pub fn to_flat_program(&self) -> Program<V, O, Vec<V>, Vec<V>>
    where
        O: Clone,
    {
        Program {
            input_structure: vec![Placeholder; self.input_count()],
            output_structure: vec![Placeholder; self.output_count()],
            regions: self.regions.clone(),
            entry: self.entry,
            marker: PhantomData,
        }
    }

    /// Converts this [`Program`] into one whose public input and output types are flat vectors. This is the consuming
    /// counterpart of [`Program::to_flat_program`]. It preserves the atom table, input atom identifiers, output atom
    /// identifiers, and instruction sequence without cloning them, and only replaces the structured input and output
    /// metadata with [`Placeholder`] vector structures sized to the flat arities.
    pub fn into_flat_program(self) -> Program<V, O, Vec<V>, Vec<V>> {
        let input_structure = vec![Placeholder; self.input_count()];
        let output_structure = vec![Placeholder; self.output_count()];
        Program { input_structure, output_structure, regions: self.regions, entry: self.entry, marker: PhantomData }
    }

    /// Returns a simplified version of this [`Program`] with dead constants and [`Instruction`]s that do not contribute
    /// to the [`Program`]'s output removed. [`Instruction`]s whose operations are not [`Effects::PURE`] are kept alive
    /// (together with the instructions producing their inputs) even when no program output consumes their results, in
    /// their original relative order, so that simplification never eliminates or reorders observable
    /// [`Effect`](crate::Effect)s.
    pub fn simplified(&self) -> Result<Self, ProgramError>
    where
        O: Clone,
    {
        // TODO(eaplatanios): Fix this as part of phase 2.
        if self.regions.len() != 1 {
            return Err(ProgramError::MalformedProgram(
                "multi-region programs are not yet supported by this transformation".to_string(),
            ));
        }
        let instruction_by_output = self.instruction_by_output();
        let mut program_builder = ProgramBuilder::new();
        let mut atom_id_mapping = HashMap::with_capacity(self.atoms().len());
        for input_id in self.input_ids().to_vec() {
            let input = self.atoms().get(input_id.index).ok_or(ProgramError::UnboundAtomId { id: input_id })?;
            let Atom::Variable(input_type) = input else {
                return Err(ProgramError::MalformedProgram("program input atom was not a variable".to_string()));
            };
            atom_id_mapping.insert(input_id, program_builder.add_input(input_type.clone()));
        }

        // Make sure that effectful instructions and their transitive dependencies are processed in original instruction
        // order before the outputs, so that instructions with observable effects survive even when dead and ordered
        // effects keep their relative order.
        for instruction in self.instructions().iter() {
            if instruction.operation().effects().is_pure() {
                continue;
            }
            for output_id in instruction.outputs().iter().copied() {
                add_atom_to_program_builder(
                    &mut program_builder,
                    &mut atom_id_mapping,
                    output_id,
                    self,
                    instruction_by_output.as_slice(),
                )?;
            }
        }

        let output_ids = self
            .output_ids()
            .to_vec()
            .into_iter()
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
    /// copying constants and operations that are discarded during simplification. The behavior of [`Self::simplified`]
    /// around [`Effects`] applies here too. [`Instruction`]s whose operations are not [`Effects::PURE`] survive in
    /// their original relative order even when no program output consumes their outputs.
    pub fn into_simplified(self) -> Result<Self, ProgramError> {
        let instruction_by_output = self.instruction_by_output();
        let effectful_instruction_outputs = self
            .instructions()
            .iter()
            .filter(|instruction| !instruction.operation().effects().is_pure())
            .flat_map(|instruction| instruction.outputs().iter().copied())
            .collect::<Vec<_>>();
        // TODO(eaplatanios): Fix this as part of phase 2.
        if self.regions.len() != 1 {
            return Err(ProgramError::MalformedProgram(
                "multi-region programs are not yet supported by this transformation".to_string(),
            ));
        }
        let Self { regions, input_structure, output_structure, .. } = self;
        let Region { atoms, input_ids, output_ids, instructions } = regions.into_iter().next().unwrap();

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

        // Make sure that effectful instructions and their transitive dependencies are processed in original instruction
        // order before the outputs, so that instructions with observable effects survive even when dead and ordered
        // effects keep their relative order.
        for root in effectful_instruction_outputs {
            move_atom_to_program(
                &mut atom_id_mapping,
                root,
                atoms.as_mut_slice(),
                instructions.as_mut_slice(),
                instruction_by_output.as_slice(),
                &mut new_atoms,
                &mut new_instructions,
            )?;
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
            input_structure,
            output_structure,
            regions: vec![Region {
                atoms: new_atoms,
                input_ids: new_input_ids,
                output_ids,
                instructions: new_instructions,
            }],
            entry: RegionId::new(0),
            marker: PhantomData,
        })
    }

    /// Rebuilds this [`Program`] as a flat subprogram over a chosen input/output boundary. The rebuilt program
    /// keeps only the [`Instruction`]s reachable from `outputs` or from the provided `keep_alive` atoms and lifts
    /// embedded constants directly into the result. Entries of `inputs` that are not reachable from any requested
    /// output or keep-alive atom are dropped. The returned index vector lists, in order, the positions of `inputs`
    /// that remain live and become the public inputs of the rebuilt program, so that callers can map rebuilt inputs
    /// back to the original boundary.
    ///
    /// Each [`Atom::Variable`] reachable from an output or keep-alive atom must either appear in `inputs` or be
    /// produced by an [`Instruction`] of this program. Reaching any other source variable (e.g., an original program
    /// input that was not selected) is reported as a [`ProgramError::MalformedProgram`]. Every entry of `inputs` must
    /// be an [`Atom::Variable`] and must appear at most once. [`Atom::Constant`]s are rebuilt automatically and need
    /// not be listed.
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
    ///   - `keep_alive`: [`AtomId`]s of atoms that must survive even if they are unreachable from `outputs`.
    pub fn filtered(
        &self,
        inputs: &[AtomId],
        outputs: &[AtomId],
        keep_alive: &[AtomId],
    ) -> Result<(Program<V, O, Vec<V>, Vec<V>>, Vec<usize>), ProgramError>
    where
        O: Clone,
    {
        // TODO(eaplatanios): Fix this as part of phase 2.
        if self.regions.len() != 1 {
            return Err(ProgramError::MalformedProgram(
                "multi-region programs are not yet supported by this transformation".to_string(),
            ));
        }
        let (instruction_by_output, input_liveness) = self.compute_live_inputs(inputs, outputs, keep_alive)?;
        let mut program_builder = ProgramBuilder::new();
        let mut atom_id_mapping = HashMap::with_capacity(self.atoms().len());
        let mut live_input_indices = Vec::new();

        for (position, id) in inputs.iter().copied().enumerate() {
            if !input_liveness[position] {
                continue;
            }
            let Atom::Variable(input_type) = &self.atoms()[id.index()] else {
                return Err(ProgramError::MalformedProgram(format!("filter input atom {id} is not a variable")));
            };
            atom_id_mapping.insert(id, program_builder.add_input(input_type.clone()));
            live_input_indices.push(position);
        }

        // Make sure that the keep-alive-atom-producing instructions and their transitive dependencies are processed in
        // original instruction order before the outputs, so that instructions with observable effects survive even when
        // dead and ordered effects keep their relative order.
        for root in keep_alive.iter().copied() {
            add_atom_to_program_builder(
                &mut program_builder,
                &mut atom_id_mapping,
                root,
                self,
                instruction_by_output.as_slice(),
            )?;
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
    /// projection. The boundary contract, keep-alive semantics, dead-input pruning, and returned live-input index
    /// vector are identical to [`Self::filtered`].
    pub fn into_filtered(
        self,
        inputs: &[AtomId],
        outputs: &[AtomId],
        keep_alive: &[AtomId],
    ) -> Result<(Program<V, O, Vec<V>, Vec<V>>, Vec<usize>), ProgramError> {
        let (instruction_by_output, input_liveness) = self.compute_live_inputs(inputs, outputs, keep_alive)?;
        // TODO(eaplatanios): Fix this as part of phase 2.
        if self.regions.len() != 1 {
            return Err(ProgramError::MalformedProgram(
                "multi-region programs are not yet supported by this transformation".to_string(),
            ));
        }
        let Region { atoms, instructions, .. } = self.regions.into_iter().next().unwrap();
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

        // Make sure that the keep-alive-atom-producing instructions and their transitive dependencies are processed in
        // original instruction order before the outputs, so that instructions with observable effects survive even when
        // dead and ordered effects keep their relative order.
        for root in keep_alive.iter().copied() {
            move_atom_to_program(
                &mut atom_id_mapping,
                root,
                atoms.as_mut_slice(),
                instructions.as_mut_slice(),
                instruction_by_output.as_slice(),
                &mut new_atoms,
                &mut new_instructions,
            )?;
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

        let input_structure = vec![Placeholder; new_input_ids.len()];
        let output_structure = vec![Placeholder; output_ids.len()];
        Ok((
            Program {
                input_structure,
                output_structure,
                regions: vec![Region {
                    atoms: new_atoms,
                    input_ids: new_input_ids,
                    output_ids,
                    instructions: new_instructions,
                }],
                entry: RegionId::new(0),
                marker: PhantomData,
            },
            live_input_indices,
        ))
    }

    /// Validates `inputs` as a deduplicated set of [`Atom::Variable`]s and determines, by reverse reachability from
    /// `outputs` and the provided `keep_alive` atoms, which of them are live (i.e., reachable from a requested output
    /// or a keep-alive atom). Returns this program's instruction-by-output map together with one liveness flag per
    /// `inputs` entry. Reaching any variable that is neither listed in `inputs` nor produced by an [`Instruction`]
    /// is reported as a [`ProgramError::MalformedProgram`].
    fn compute_live_inputs(
        &self,
        inputs: &[AtomId],
        outputs: &[AtomId],
        keep_alive: &[AtomId],
    ) -> Result<(Vec<Option<usize>>, Vec<bool>), ProgramError> {
        let mut input_position = vec![None; self.atoms().len()];
        for (position, id) in inputs.iter().copied().enumerate() {
            let atom = self.atoms().get(id.index()).ok_or(ProgramError::UnboundAtomId { id })?;
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
        let mut needed = vec![false; self.atoms().len()];
        let mut input_liveness = vec![false; inputs.len()];
        let mut stack = Vec::new();
        for output in outputs.iter().copied().chain(keep_alive.iter().copied()) {
            if output.index() >= self.atoms().len() {
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
            match &self.atoms()[atom_id.index()] {
                Atom::Constant(_) => {}
                Atom::Variable(_) => {
                    let instruction_index = instruction_by_output.get(atom_id.index()).copied().flatten().ok_or(
                        ProgramError::MalformedProgram(format!(
                            "filter atom {atom_id} is not a selected input and has no producer",
                        )),
                    )?;
                    for input in self.instructions()[instruction_index].inputs.iter().copied() {
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
}

impl<V: Value, O: Operation<V::Type>, Input: Parameterized<V>, Output: Parameterized<V>> Program<V, O, Input, Output> {
    /// Renders this [`Program`] with the provided indentation level that is useful for situations where [`Program`]s
    /// are nested within other programs like with control flow [`Operation`]s.
    pub fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        let entry_region = self.entry_region();
        write!(formatter, "{:indentation$}", "")?;
        write!(formatter, "lambda ")?;
        entry_region.input_ids.iter().enumerate().try_for_each(|(index, input_id)| {
            if index > 0 {
                write!(formatter, ", {input_id}:{}", entry_region.atoms[input_id.index].r#type())
            } else {
                write!(formatter, "{input_id}:{}", entry_region.atoms[input_id.index].r#type())
            }
        })?;
        writeln!(formatter, " .")?;
        let mut instructions_by_first_output = vec![None; entry_region.atoms.len()];
        for (index, instruction) in entry_region.instructions.iter().enumerate() {
            if let Some(output_id) = instruction.outputs.first() {
                instructions_by_first_output[output_id.index] = Some(index);
            }
        }
        let mut binding_count = 0usize;
        let mut is_input = vec![false; entry_region.atoms.len()];
        for input_id in entry_region.input_ids.iter().copied() {
            is_input[input_id.index] = true;
        }
        for (atom_id, atom) in entry_region.atoms.iter().enumerate() {
            match atom {
                Atom::Constant(_) => {
                    write!(formatter, "{:indentation$}", "")?;
                    writeln!(
                        formatter,
                        "{} {}:{} = const",
                        if binding_count == 0 { "let" } else { "   " },
                        AtomId { index: atom_id },
                        entry_region.atoms[atom_id].r#type()
                    )?;
                    binding_count += 1;
                }
                Atom::Variable(_) if is_input[atom_id] => {}
                Atom::Variable(_) => {
                    if let Some(instruction_index) = instructions_by_first_output[atom_id] {
                        let instruction = &entry_region.instructions[instruction_index];
                        write!(formatter, "{:indentation$}", "")?;
                        write!(formatter, "{} ", if binding_count == 0 { "let" } else { "   " })?;
                        instruction.outputs.iter().enumerate().try_for_each(|(index, output)| {
                            if index > 0 {
                                write!(formatter, ", {output}:{}", entry_region.atoms[output.index].r#type())
                            } else {
                                write!(formatter, "{output}:{}", entry_region.atoms[output.index].r#type())
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
        entry_region.output_ids.iter().enumerate().try_for_each(|(index, output)| {
            if index > 0 { write!(formatter, ", {output}") } else { write!(formatter, "{output}") }
        })?;
        write!(formatter, ")")
    }
}

impl<V: Value, O: Clone, Input: Parameterized<V>, Output: Parameterized<V>> Clone for Program<V, O, Input, Output> {
    fn clone(&self) -> Self {
        Self {
            input_structure: self.input_structure.clone(),
            output_structure: self.output_structure.clone(),
            regions: self.regions.clone(),
            entry: self.entry,
            marker: PhantomData,
        }
    }
}

impl<V: Value, O: Operation<V::Type>, Input: Parameterized<V>, Output: Parameterized<V>> Display
    for Program<V, O, Input, Output>
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

/// _Flat_ [`Program`] (i.e., with flat `Vec`-valued inputs and outputs) over a [`Domain`]'s constant and operation
/// universe. This is the canonical shape for nested computations that are constructed standalone and are replayed
/// positionally, such as the `regions` and `callees` arguments of [`Context::bind`].
pub type FlatProgram<D> = Program<
    <D as Domain>::Constant,
    <D as Domain>::Operation,
    Vec<<D as Domain>::Constant>,
    Vec<<D as Domain>::Constant>,
>;

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

/// Builder for [`Program`]s. It owns the entry [`Region`] under construction (i.e., its [`Atom`]s, input [`AtomId`]s,
/// and [`Instruction`]s), the previously added non-entry [`Region`]s together with their callee-interning state, and
/// an optional [`ProgramError`] that can be used to signal a failure during program construction. Non-entry regions
/// enter a builder only in sealed form: [`add_region`](Self::add_region) and [`add_callee`](Self::add_callee) copy
/// complete reachable closures out of already-built (and therefore immutable and fully validated) [`Program`]s, so a
/// region can never change after an instruction attaches it.
#[derive(Clone, Debug, Default)]
pub struct ProgramBuilder<V: Typed + Parameter, O> {
    /// [`Atom`]s contained in the entry [`Region`] of the [`Program`] that is being built, in evaluation order.
    pub(crate) atoms: Vec<Atom<V>>,

    /// [`AtomId`]s of the [`Atom`]s that correspond to the inputs (i.e., arguments) of the [`Program`] being built.
    pub(crate) input_ids: Vec<AtomId>,

    /// Ordered sequence of [`Instruction`]s that make up the entry [`Region`] of the [`Program`] being built.
    pub(crate) instructions: Vec<Instruction<O>>,

    /// Sealed non-entry [`Region`]s of the [`Program`] being built, in [`RegionId`] order. Regions are appended to
    /// this list with [`Self::add_region`] and [`Self::add_callee`], and instructions reference them by [`RegionId`].
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
        let id = AtomId { index: self.atoms.len() };
        self.atoms.push(Atom::Constant(value));
        id
    }

    /// Adds an [`Atom::Variable`] to the [`Program`] that is being built with the provided [`Type`](crate::Type).
    #[inline]
    pub fn add_variable(&mut self, r#type: V::Type) -> AtomId {
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
        self.instructions.push(Instruction::new(operation, inputs, outputs, Vec::new()));
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

    /// Appends the provided [`Program`]'s [`Instruction`]s and constants to this [`ProgramBuilder`], remapping its
    /// inputs to the caller-provided `inputs` and returning the builder atoms holding the program's outputs, in output
    /// order. This is a plain relocation and not a re-interpretation or partial evaluation. Every instruction and every
    /// constant of the provided program is rebuilt verbatim into this builder. It is, for example, the reconciliation
    /// primitive an unknown-predicate `condition` uses to graft each branch's residual program into the reconciled
    /// branch it emits during partial evaluation.
    #[inline]
    pub fn add_program<Input: Parameterized<V>, Output: Parameterized<V>>(
        &mut self,
        program: &Program<V, O, Input, Output>,
        inputs: &[AtomId],
    ) -> Result<Vec<AtomId>, ProgramError>
    where
        O: Clone,
    {
        // TODO(eaplatanios): Fix this as part of phase 2.
        if program.regions.len() != 1 {
            return Err(ProgramError::MalformedProgram(
                "multi-region programs are not yet supported by this transformation".to_string(),
            ));
        }
        // The two closures below never run concurrently but both need `&mut` access to this builder. A `RefCell` lets
        // each take a short-lived mutable borrow without the borrow checker conservatively rejecting the second one.
        let builder = RefCell::new(self);
        program.interpret_with::<AtomId, ProgramError, _, _>(
            inputs.to_vec(),
            |_, constant| Ok(builder.borrow_mut().add_constant(constant.clone())),
            |instruction, inputs| {
                Ok(builder.borrow_mut().add_instruction(instruction.operation().clone(), inputs.to_vec())?.to_vec())
            },
        )
    }

    // TODO(eaplatanios): [regions] `add_region` and `add_callee` are reference-based (clone-on-add) for now,
    //  matching the detached-trace construction idiom. Once the operation-family migrations land and
    //  `Context::bind` hands the builder owned programs, consider owned-move variants that splice the
    //  source arena directly instead of cloning it. This is a Phase 2+ decision.
    /// Imports the provided [`Program`] as a lexically owned subtree, copying the complete reachable closure of its
    /// entry [`Region`] (i.e., lexical descendants, referenced callee roots, and their transitive closures) with all
    /// [`RegionId`]s remapped and sharing within the imported closure preserved.
    pub fn add_region<Input: Parameterized<V>, Output: Parameterized<V>>(
        &mut self,
        program: &Program<V, O, Input, Output>,
    ) -> RegionId
    where
        O: Clone,
    {
        /// Copies the [`Region`]s of `regions[root]` that are reachable from `root` into `builder` in post order
        /// (i.e., children before parents) so that every copied instruction references previously appended regions,
        /// memoizing the [`RegionId`] remapping in `remapping` so that sharing within one added program is preserved.
        fn copy_reachable_regions<V: Value, O: Clone + Operation<V::Type>>(
            builder: &mut ProgramBuilder<V, O>,
            regions: &[Region<V, O>],
            root: RegionId,
            remapping: &mut HashMap<RegionId, RegionId>,
        ) -> RegionId {
            if let Some(mapped) = remapping.get(&root) {
                return *mapped;
            }
            let mut region = regions[root.index()].clone();
            for instruction in &mut region.instructions {
                for region in &mut instruction.regions {
                    *region = copy_reachable_regions(builder, regions, *region, remapping);
                }
            }
            let id = RegionId::new(builder.regions.len());
            builder.regions.push(region);
            remapping.insert(root, id);
            id
        }

        copy_reachable_regions(self, &program.regions, program.entry, &mut HashMap::new())
    }

    // TODO(eaplatanios): [regions] `add_region` and `add_callee` are reference-based (clone-on-add) for now,
    //  matching the detached-trace construction idiom. Once the operation-family migrations land and
    //  `Context::bind` hands the builder owned programs, consider owned-move variants that splice the
    //  source arena directly instead of cloning it. This is a Phase 2+ decision.
    // TODO(eaplatanios): Explain why this is separate from `add_region` in its docstring and why it uses `Rc`.
    /// Imports the provided [`Program`] as a shareable callee root, interning by [`Rc`] identity within this builder.
    /// Two calls to this function with the same live [`Rc`] will reuse one callee root [`RegionId`], while structurally
    /// equal but independently built programs will remain distinct.
    pub fn add_callee(&mut self, program: &Rc<Program<V, O, Vec<V>, Vec<V>>>) -> RegionId
    where
        O: Clone,
    {
        if let Some((_, id)) = self.callees.iter().find(|(interned, _)| Rc::ptr_eq(interned, program)) {
            return *id;
        }
        let id = self.add_region(program.as_ref());
        self.callees.push((program.clone(), id));
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
            let input = self.atoms.get(input_id.index).ok_or(ProgramError::UnboundAtomId { id: input_id })?;
            let Atom::Variable(_) = input else {
                return Err(ProgramError::MalformedProgram("program input atom was not a variable".to_string()));
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

        // Entry instructions may only reference previously added regions (i.e., regions with identifiers strictly
        // below the entry's own, which is assigned last). Non-entry regions uphold the same property by construction,
        // because `add_region` copies them in post order (i.e., children before parents). Every referenced region
        // identifier is therefore in range and the region graph is acyclic, which is what allows the reachability walk
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

        Ok(Program { input_structure, output_structure, regions, entry, marker: PhantomData })
    }
}

/// Adds the [`Atom`] that corresponds to `atom_id` in `program` to the provided [`ProgramBuilder`], recursively adding
/// its transitive producers first and memoizing the old-to-new [`AtomId`] mapping in `atom_id_mapping`. Atoms already
/// present in the mapping (e.g., rebuilt program inputs) are reused, [`Atom::Constant`]s are rebuilt directly, and
/// [`Atom::Variable`]s are reconstructed from their producing [`Instruction`]. A reachable variable that is neither
/// mapped nor produced by an instruction is reported as a [`ProgramError::MalformedProgram`].
fn add_atom_to_program_builder<
    V: Value,
    O: Clone + Operation<V::Type>,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
>(
    program_builder: &mut ProgramBuilder<V, O>,
    atom_id_mapping: &mut HashMap<AtomId, AtomId>,
    atom_id: AtomId,
    program: &Program<V, O, Input, Output>,
    instruction_by_output: &[Option<usize>],
) -> Result<AtomId, ProgramError> {
    if let Some(mapped_atom) = atom_id_mapping.get(&atom_id) {
        return Ok(*mapped_atom);
    }
    let atom = program.atoms().get(atom_id.index).ok_or(ProgramError::UnboundAtomId { id: atom_id })?;
    let atom = match atom {
        Atom::Constant(value) => Ok(program_builder.add_constant(value.clone())),
        Atom::Variable(_) => {
            let instruction_index = instruction_by_output
                .get(atom_id.index)
                .copied()
                .flatten()
                .ok_or(ProgramError::MalformedProgram("variable atom has no owning instruction".to_string()))?;
            let instruction = &program.instructions()[instruction_index];
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
fn move_atom_to_program<V: Value, O: Operation<V::Type>>(
    atom_id_mapping: &mut HashMap<AtomId, AtomId>,
    atom_id: AtomId,
    atoms: &mut [Option<Atom<V>>],
    instructions: &mut [Option<Instruction<O>>],
    instruction_by_output: &[Option<usize>],
    new_atoms: &mut Vec<Atom<V>>,
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
        None => return Err(ProgramError::UnboundAtomId { id: atom_id }),
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
    new_instructions.push(Instruction::new(instruction.operation, inputs, outputs, Vec::new()));
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

    use crate::backends::scalars::{Scalar, ScalarOperation};
    use crate::effects::{Effect, Effects};
    use crate::macros::check_count;
    use crate::operations::OperationFormatter;
    use crate::operations::compare::{CompareOperation, ComparisonDirection};
    use crate::operations::debugging::PrintOperation;
    use crate::operations::math::{AddOperation, MulOperation, NegOperation};
    use crate::parameters::Placeholder;
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
        let constant = Atom::<Scalar>::Constant(Scalar::from(3.0));
        let variable = Atom::<Scalar>::Variable(DataType::F64);

        assert!(constant.is_constant());
        assert!(!constant.is_variable());
        assert_eq!(constant.as_constant(), Some(&Scalar::from(3.0)));
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
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let i0 = builder.add_input(DataType::F64);
        let c0 = builder.add_constant(Scalar::from(3.0f64));
        let o0 = builder.add_instruction(AddOperation, vec![i0, c0]).unwrap()[0];
        let program = builder.build::<Scalar, Scalar>(vec![o0], Placeholder, Placeholder).unwrap();
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
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let i0 = builder.add_input(DataType::F64);
        let i1 = builder.add_input(DataType::F64);
        let v0 = builder.add_instruction(NegOperation, vec![i0]).unwrap()[0];
        let o0 = builder.add_instruction(AddOperation, vec![v0, i1]).unwrap()[0];
        let program = builder
            .build::<(Scalar, Scalar), Scalar>(vec![o0], (Placeholder, Placeholder), Placeholder)
            .unwrap();
        assert_eq!(program.input_types(), vec![DataType::F64, DataType::F64]);
        assert_eq!(program.output_types(), vec![DataType::F64]);
        let input = program.input().unwrap();
        let output = program.output().unwrap();
        assert_eq!(program.interpret((Scalar::from(2.0), Scalar::from(3.0))), Ok(Scalar::from(1.0)));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64, %1:f64 .
                let %2:f64 = neg %0
                    %3:f64 = add %2 %1
                in (%3)
            "}
            .trim_end(),
        );
        assert!(matches!(input.0, Atom::Variable(r#type) if r#type == DataType::F64));
        assert!(matches!(input.1, Atom::Variable(r#type) if r#type == DataType::F64));
        assert!(matches!(output, Atom::Variable(r#type) if r#type == DataType::F64));

        // Test a program that contains an operation with long metadata that should be rendered on multiple lines.
        let mut builder = ProgramBuilder::<Scalar, LongMetadataOperation>::new();
        let i0 = builder.add_input(DataType::F64);
        let o0 = builder.add_instruction(LongMetadataOperation, vec![i0]).unwrap()[0];
        let program = builder.build::<Scalar, Scalar>(vec![o0], Placeholder, Placeholder).unwrap();
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
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let i0 = builder.add_input(DataType::F32);
        let o0 = builder.add_instruction(AddOperation, vec![i0, i0]).unwrap()[0];
        let program = builder
            .build::<Scalar, (Scalar, Scalar)>(vec![o0, o0], Placeholder, (Placeholder, Placeholder))
            .unwrap();
        let input = program.input().unwrap();
        let output = program.output().unwrap();
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
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        builder.add_input(DataType::F64);
        let o0 = builder.add_variable(DataType::F64);
        assert!(matches!(
            builder.build::<Scalar, Scalar>(vec![o0], Placeholder, Placeholder),
            Err(ProgramError::MalformedProgram(message)) if message == "variable atom has no owning instruction",
        ));

        // Test a case where we have an instruction input atom with no parent instruction.
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let i0 = builder.add_input(DataType::F64);
        let v0 = builder.add_variable(DataType::F64);
        let o0 = builder.add_instruction(AddOperation, vec![i0, v0]).unwrap()[0];
        assert!(matches!(
            builder.build::<Scalar, Scalar>(vec![o0], Placeholder, Placeholder),
            Err(ProgramError::MalformedProgram(message)) if message == "variable atom has no owning instruction",
        ));
    }

    #[test]
    fn test_program_inputs_mask() {
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let first_input = builder.add_input(DataType::F64);
        let constant = builder.add_constant(Scalar::from(3.0f64));
        let second_input = builder.add_input(DataType::F64);
        let scaled = builder.add_instruction(NegOperation, vec![first_input]).unwrap()[0];
        let output = builder.add_instruction(AddOperation, vec![scaled, second_input]).unwrap()[0];
        let program = builder
            .build::<Vec<Scalar>, Scalar>(vec![output], vec![Placeholder, Placeholder], Placeholder)
            .unwrap();
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
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let input = builder.add_input(DataType::F64);
        let constant = builder.add_constant(Scalar::from(3.0f64));
        let scaled = builder.add_instruction(NegOperation, vec![input]).unwrap()[0];
        let output = builder.add_instruction(AddOperation, vec![scaled, constant]).unwrap()[0];
        let dead_output = builder.add_instruction(NegOperation, vec![input]).unwrap()[0];
        let program = builder.build::<Scalar, Scalar>(vec![output], Placeholder, Placeholder).unwrap();

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
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let live_input = builder.add_input(DataType::F64);
        let dead_input = builder.add_input(DataType::F64);
        let live_constant = builder.add_constant(Scalar::from(3.0f64));
        let dead_constant = builder.add_constant(Scalar::from(5.0f64));
        let scaled = builder.add_instruction(NegOperation, vec![live_input]).unwrap()[0];
        let output = builder.add_instruction(AddOperation, vec![scaled, live_constant]).unwrap()[0];
        let dead_output = builder.add_instruction(AddOperation, vec![dead_input, dead_constant]).unwrap()[0];
        let program = builder
            .build::<(Scalar, Scalar), Scalar>(vec![output], (Placeholder, Placeholder), Placeholder)
            .unwrap();
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
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let input = builder.add_input(DataType::F64);
        let constant = builder.add_constant(Scalar::from(3.0f64));
        let negated = builder.add_instruction(NegOperation, vec![input]).unwrap()[0];
        let combined = builder.add_instruction(AddOperation, vec![negated, constant]).unwrap()[0];
        let output = builder
            .add_instruction(CompareOperation::new(ComparisonDirection::LessThan), vec![combined, constant])
            .unwrap()[0];
        let program = builder.build::<Scalar, Scalar>(vec![output], Placeholder, Placeholder).unwrap();

        // `map_operations` rebuilds the value graph while rewriting operations: the binary `add` is replaced by a
        // different operation (`mul`), the `compare` payload field is rewritten in place (its direction is flipped),
        // and the unary `neg` is forwarded unchanged. The atom table and rendered structure are preserved.
        let mapped = program
            .map_operations(|operation| {
                Ok::<_, ProgramError>(match operation {
                    ScalarOperation::Compare(operation) => {
                        assert_eq!(operation.direction(), ComparisonDirection::LessThan);
                        ScalarOperation::Compare(CompareOperation::new(ComparisonDirection::GreaterThan))
                    }
                    ScalarOperation::Add(_) => ScalarOperation::Mul(MulOperation),
                    operation => operation.clone(),
                })
            })
            .unwrap();

        // Original: `(-input + 3) < 3`, so for `input = 2` this is `1 < 3 = true`.
        assert_eq!(program.interpret(Scalar::from(2.0f64)), Ok(Scalar::from(true)));
        // Mapped: `(-input * 3) > 3`, so for `input = 2` this is `-6 > 3 = false`.

        assert_eq!(mapped.interpret(Scalar::from(2.0f64)), Ok(Scalar::from(false)));

        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = const
                    %2:f64 = neg %0
                    %3:f64 = add %2 %1
                    %4:bool = compare [direction=LessThan] %3 %1
                in (%4)
            "}
            .trim_end(),
        );

        assert_eq!(
            mapped.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = const
                    %2:f64 = neg %0
                    %3:f64 = mul %2 %1
                    %4:bool = compare [direction=GreaterThan] %3 %1
                in (%4)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_program_to_flat_program_and_into_flat_program() {
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let i0 = builder.add_input(DataType::F64);
        let i1 = builder.add_input(DataType::F64);
        let v0 = builder.add_instruction(NegOperation, vec![i0]).unwrap()[0];
        let o0 = builder.add_instruction(AddOperation, vec![v0, i1]).unwrap()[0];
        let program = builder
            .build::<(Scalar, Scalar), Scalar>(vec![o0], (Placeholder, Placeholder), Placeholder)
            .unwrap();

        let flat_program = program.to_flat_program();
        assert_eq!(flat_program.input_structure(), &vec![Placeholder, Placeholder]);
        assert_eq!(flat_program.output_structure(), &vec![Placeholder]);
        assert_eq!(flat_program.interpret(vec![Scalar::from(2.0), Scalar::from(3.0)]), Ok(vec![Scalar::from(1.0)]));

        let flat_program = program.into_flat_program();
        assert_eq!(flat_program.input_structure(), &vec![Placeholder, Placeholder]);
        assert_eq!(flat_program.output_structure(), &vec![Placeholder]);
        assert_eq!(flat_program.interpret(vec![Scalar::from(2.0), Scalar::from(3.0)]), Ok(vec![Scalar::from(1.0)]));
    }

    #[test]
    fn test_program_simplified() {
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let i0 = builder.add_input(DataType::F64);
        let c0 = builder.add_constant(Scalar::from(2.0f64));
        let c1 = builder.add_constant(Scalar::from(3.0f64));
        let _ = builder.add_instruction(AddOperation, vec![i0, c0]).unwrap()[0];
        let v1 = builder.add_instruction(AddOperation, vec![i0, c1]).unwrap()[0];
        let program = builder
            .build::<Scalar, (Scalar, Scalar)>(vec![v1, v1], Placeholder, (Placeholder, Placeholder))
            .unwrap();
        let simplified = program.simplified().unwrap();

        assert_eq!(c0, AtomId { index: 1 });
        assert_eq!(simplified.interpret(Scalar::from(2.0f64)), Ok((Scalar::from(5.0f64), Scalar::from(5.0f64))));
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

        // The pure program above reports no effects, and simplification removed its dead `add` as asserted. Effectful
        // instructions, in contrast, are kept alive by simplification even when they are dead code: nothing consumes
        // the print's output below, so only its effect keeps it in the simplified program.
        assert_eq!(program.effects(), Effects::PURE);
        let build = || {
            let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
            let input = builder.add_input(DataType::F64);
            let doubled = builder.add_instruction(AddOperation, vec![input, input]).unwrap()[0];
            let _printed = builder.add_instruction(PrintOperation::new("x"), vec![input]).unwrap()[0];
            builder.build::<Scalar, Scalar>(vec![doubled], Placeholder, Placeholder).unwrap()
        };
        let effectful = build();
        assert_eq!(effectful.effects(), Effects::single(Effect::OrderedIo));
        let expected = indoc! {"
            lambda %0:f64 .
            let %1:f64 = print [label=x] %0
                %2:f64 = add %0 %0
            in (%2)
        "}
        .trim_end();
        assert_eq!(effectful.simplified().unwrap().to_string(), expected);
        assert_eq!(build().into_simplified().unwrap().to_string(), expected);
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

        impl Typed for CloneCountingValue {
            type Type = DataType;

            fn r#type(&self) -> Cow<'_, DataType> {
                Cow::Owned(DataType::F64)
            }
        }

        impl Value for CloneCountingValue {
            type DispatchDomain = crate::EagerContext<Self>;
            type ExecutionDomain = crate::EagerContext<Self>;

            fn dispatch_domain(&self) -> crate::EagerContext<Self> {
                crate::EagerContext::new()
            }

            fn execution_domain(&self) -> crate::EagerContext<Self> {
                crate::EagerContext::new()
            }
        }

        let value_clone_count = Rc::new(Cell::new(0));
        let mut builder = ProgramBuilder::<_, ScalarOperation<CloneCountingValue>>::new();
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
        assert_eq!(simplified.input_ids(), vec![AtomId { index: 0 }]);
        assert_eq!(simplified.output_ids(), vec![AtomId { index: 2 }, AtomId { index: 2 }]);
        assert_eq!(simplified.atoms().len(), 3);
        assert!(matches!(simplified.atoms().get(1), Some(Atom::Constant(value)) if value.value == 3.0));
        assert_eq!(simplified.instructions().len(), 1);
        assert_eq!(simplified.instructions()[0].inputs(), vec![AtomId { index: 0 }, AtomId { index: 1 }]);
        assert_eq!(simplified.instructions()[0].outputs(), vec![AtomId { index: 2 }]);
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
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let i0 = builder.add_input(DataType::F64);
        let i1 = builder.add_input(DataType::F64);
        let c0 = builder.add_constant(Scalar::from(2.0f64));
        let v0 = builder.add_instruction(NegOperation, vec![i0]).unwrap()[0];
        let v1 = builder.add_instruction(AddOperation, vec![v0, c0]).unwrap()[0];
        let program = builder
            .build::<(Scalar, Scalar), Scalar>(vec![v1], (Placeholder, Placeholder), Placeholder)
            .unwrap();

        // Dead inputs are pruned and constants are lifted: `i1` is dead for `v1`, so it is dropped,
        // and `c0` is rebuilt into the projected program.
        let (pruned, pruned_live) = program.filtered(&[i0, i1], &[v1], &[]).unwrap();
        assert_eq!(pruned_live, vec![0]);
        assert_eq!(pruned.input_ids().len(), 1);
        assert_eq!(pruned.interpret(vec![Scalar::from(4.0)]), Ok(vec![Scalar::from(-2.0)]));

        // Selecting an intermediate atom (i.e., `v0`) as the output drops the downstream `add`
        // and the now-dead constant.
        let (intermediate, intermediate_live) = program.filtered(&[i0], &[v0], &[]).unwrap();
        assert_eq!(intermediate_live, vec![0]);
        assert_eq!(intermediate.instructions().len(), 1);
        assert_eq!(intermediate.interpret(vec![Scalar::from(5.0)]), Ok(vec![Scalar::from(-5.0)]));

        // Forwarding an input directly as an output yields an instruction-free program over only that input.
        let (forwarded, forwarded_live) = program.filtered(&[i0, i1], &[i0], &[]).unwrap();
        assert_eq!(forwarded_live, vec![0]);
        assert_eq!(forwarded.instructions().len(), 0);
        assert_eq!(forwarded.interpret(vec![Scalar::from(7.0)]), Ok(vec![Scalar::from(7.0)]));

        // Reaching a variable that is neither a selected input nor produced by an instruction is rejected:
        // `v1` depends on `i0`, which is omitted from the selected inputs here.
        assert!(matches!(program.filtered(&[i1], &[v1], &[]), Err(ProgramError::MalformedProgram(_))));

        // Providing the same input atom more than once is rejected.
        assert!(matches!(program.filtered(&[i0, i0], &[v1], &[]), Err(ProgramError::MalformedProgram(_))));

        // A keep-alive entry naming an otherwise-pruned atom retains its producing instruction chain without
        // widening the projection's outputs: projecting onto `v0` alone drops the downstream `add` and the constant,
        // while keeping `v1` alive pulls them back in.
        let (kept, kept_live) = program.filtered(&[i0], &[v0], &[v1]).unwrap();
        assert_eq!(kept_live, vec![0]);
        assert_eq!(kept.instructions().len(), 2);
        assert_eq!(kept.output_ids().len(), 1);
        assert_eq!(kept.interpret(vec![Scalar::from(5.0)]), Ok(vec![Scalar::from(-5.0)]));

        // A keep-alive entry naming a dead input pins it as a live public input instead of pruning it.
        let (pinned, pinned_live) = program.filtered(&[i0, i1], &[v1], &[i1]).unwrap();
        assert_eq!(pinned_live, vec![0, 1]);
        assert_eq!(pinned.input_ids().len(), 2);
        assert_eq!(pinned.interpret(vec![Scalar::from(4.0), Scalar::from(9.0)]), Ok(vec![Scalar::from(-2.0)]));
    }

    #[test]
    fn test_program_into_filtered() {
        // Build the same program twice, so that the consuming `into_filtered` can be compared
        // against the borrowing `filter`.
        let build = || {
            let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
            let i0 = builder.add_input(DataType::F64);
            let i1 = builder.add_input(DataType::F64);
            let c0 = builder.add_constant(Scalar::from(2.0f64));
            let v0 = builder.add_instruction(NegOperation, vec![i0]).unwrap()[0];
            let v1 = builder.add_instruction(AddOperation, vec![v0, c0]).unwrap()[0];
            let program = builder
                .build::<(Scalar, Scalar), Scalar>(vec![v1], (Placeholder, Placeholder), Placeholder)
                .unwrap();
            (program, i0, i1, v0, v1)
        };

        let (borrowed_program, b_i0, b_i1, _, b_v1) = build();
        let (borrowed, borrowed_live) = borrowed_program.filtered(&[b_i0, b_i1], &[b_v1], &[]).unwrap();
        let (owned_program, o_i0, o_i1, _, o_v1) = build();
        let (owned, owned_live) = owned_program.into_filtered(&[o_i0, o_i1], &[o_v1], &[]).unwrap();

        // The consuming variant drops the dead input, lifts the constant, and is identical to the borrowing `filter`.
        assert_eq!(owned_live, vec![0]);
        assert_eq!(owned_live, borrowed_live);
        assert_eq!(owned.input_ids().len(), 1);
        assert_eq!(owned.interpret(vec![Scalar::from(4.0)]), Ok(vec![Scalar::from(-2.0)]));
        assert_eq!(owned.to_string(), borrowed.to_string());

        // Keep-alive entries follow the same contract as the borrowing `filtered`: keeping `v1` alive moves its
        // otherwise-pruned `add` and constant into the projection onto `v0`, without widening the outputs.
        let (kept_program, k_i0, _, k_v0, k_v1) = build();
        let (kept, kept_live) = kept_program.into_filtered(&[k_i0], &[k_v0], &[k_v1]).unwrap();
        assert_eq!(kept_live, vec![0]);
        assert_eq!(kept.instructions().len(), 2);
        assert_eq!(kept.output_ids().len(), 1);
        assert_eq!(kept.interpret(vec![Scalar::from(4.0)]), Ok(vec![Scalar::from(-4.0)]));
    }

    #[test]
    fn test_program_builder() {
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let i0 = builder.add_input(DataType::F64);
        let i1 = builder.add_input(DataType::F64);
        let c0 = builder.add_constant(Scalar::from(2.0f64));
        let v0 = builder.add_instruction(NegOperation, vec![i0]).unwrap()[0];
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

        let program = builder
            .build::<(Scalar, Scalar), Scalar>(vec![v1], (Placeholder, Placeholder), Placeholder)
            .unwrap();
        assert_eq!(program.input_ids(), vec![i0, i1]);
        assert_eq!(program.output_ids(), vec![v1]);
        assert_eq!(program.instructions().len(), 2);
        assert_eq!(program.interpret((Scalar::from(2.0f64), Scalar::from(38.0f64))), Ok(Scalar::from(36.0f64)));

        // `add_program` appends the program's reachable instructions into a fresh builder, remapping its inputs to the
        // provided builder atoms and returning the builder atoms for its outputs. The program's `2.0` constant is dead
        // (i.e., no instruction consumes it), and so only the two reachable instructions are rebuilt.
        let mut outer = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let a0 = outer.add_input(DataType::F64);
        let a1 = outer.add_input(DataType::F64);
        let outputs = outer.add_program(&program, &[a0, a1]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outer.instructions.len(), 2);
        let outer_program =
            outer.build::<(Scalar, Scalar), Scalar>(outputs, (Placeholder, Placeholder), Placeholder).unwrap();
        assert_eq!(outer_program.interpret((Scalar::from(2.0f64), Scalar::from(38.0f64))), Ok(Scalar::from(36.0f64)));
    }

    #[test]
    fn test_program_builder_rejects_unbound_instruction_inputs() {
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let v0 = builder.add_instruction(AddOperation, vec![AtomId { index: 42 }, AtomId { index: 99 }]);
        assert!(matches!(v0, Err(ProgramError::UnboundAtomId { id }) if id == AtomId { index: 42 }));
    }

    #[test]
    fn test_program_builder_build_returns_error() {
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        builder.error = Some(ProgramError::InvalidInputCount { expected: 1, actual: 0 });
        assert!(matches!(
            builder.build::<Scalar, Scalar>(Vec::new(), Placeholder, Placeholder),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        ));
    }

    #[test]
    fn test_program_builder_build_rejects_invalid_input_count() {
        let builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        assert!(matches!(
            builder.build::<Scalar, ()>(Vec::new(), Placeholder, ()),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        ));
    }

    #[test]
    fn test_program_builder_build_rejects_invalid_output_count() {
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        builder.add_input(DataType::F64);
        assert!(matches!(
            builder.build::<Scalar, Scalar>(Vec::new(), Placeholder, Placeholder),
            Err(ProgramError::InvalidOutputCount { expected: 1, actual: 0 }),
        ));
    }

    #[test]
    fn test_program_builder_build_rejects_malformed_atom_providers() {
        let mut duplicate_input_builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let input = duplicate_input_builder.add_input(DataType::F64);
        duplicate_input_builder.input_ids.push(input);
        assert!(matches!(
            duplicate_input_builder.build::<Vec<Scalar>, Vec<Scalar>>(
                vec![input],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            ),
            Err(ProgramError::MalformedProgram(message))
                if message == format!("program input atom {input} appears more than once")
        ));

        let mut input_output_overlap_builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let input = input_output_overlap_builder.add_input(DataType::F64);
        input_output_overlap_builder.add_instruction_unchecked(Instruction::new(
            ScalarOperation::Neg(NegOperation),
            vec![input],
            vec![input],
            Vec::new(),
        ));
        assert!(matches!(
            input_output_overlap_builder.build::<Vec<Scalar>, Vec<Scalar>>(
                vec![input],
                vec![Placeholder],
                vec![Placeholder],
            ),
            Err(ProgramError::MalformedProgram(message))
                if message == format!("instruction output atom {input} is a program input")
        ));

        let mut duplicate_output_builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let input = duplicate_output_builder.add_input(DataType::F64);
        let output = duplicate_output_builder.add_variable(DataType::F64);
        duplicate_output_builder.add_instruction_unchecked(Instruction::new(
            ScalarOperation::Neg(NegOperation),
            vec![input],
            vec![output],
            Vec::new(),
        ));
        duplicate_output_builder.add_instruction_unchecked(Instruction::new(
            ScalarOperation::Neg(NegOperation),
            vec![input],
            vec![output],
            Vec::new(),
        ));
        assert!(matches!(
            duplicate_output_builder.build::<Vec<Scalar>, Vec<Scalar>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            ),
            Err(ProgramError::MalformedProgram(message))
                if message == format!("instruction output atom {output} is produced by more than one instruction")
        ));
    }

    // TODO(eaplatanios): Review this.
    #[test]
    fn test_program_builder_add_region_and_add_callee() {
        // A source program with one sealed region attached to its entry instruction.
        let mut source_builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let mut region_builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let region_input = region_builder.add_input(DataType::F64);
        let region_program = region_builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![region_input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let sealed = source_builder.add_region(&region_program);
        let input = source_builder.add_input(DataType::F64);
        let output = source_builder.add_variable(DataType::F64);
        source_builder.add_instruction_unchecked(Instruction::new(
            AddOperation.into(),
            vec![input, input],
            vec![output],
            vec![sealed],
        ));
        let source = source_builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        // Lexical imports copy the complete closure and never share: two imports produce two subtrees.
        let mut destination = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let first = destination.add_region(&source);
        let second = destination.add_region(&source);
        assert_ne!(first, second);
        let imported = destination.regions[first.index()].clone();
        assert_eq!(imported.instructions()[0].regions().len(), 1);
        assert_ne!(imported.instructions()[0].regions()[0], first);

        // Callee imports intern by live `Rc` identity: one shared root per live source, while structurally equal
        // but independently built programs remain distinct.
        let flat = Rc::new(source.to_flat_program());
        let equal_but_distinct = Rc::new(flat.as_ref().clone());
        let mut destination = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let first = destination.add_callee(&flat);
        let second = destination.add_callee(&flat);
        let third = destination.add_callee(&equal_but_distinct);
        assert_eq!(first, second);
        assert_ne!(first, third);
    }

    #[test]
    fn test_program_builder_build_multi_region_program() {
        // TODO(eaplatanios): The operation-declared region contract arrives in Phase 2, and so the region-carrying
        //  instruction is assembled through the unchecked path.
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let mut region_builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let region_input = region_builder.add_input(DataType::F64);
        let doubled = region_builder.add_instruction(AddOperation, vec![region_input, region_input]).unwrap()[0];
        let region_program = region_builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![doubled], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let sealed = builder.add_region(&region_program);
        assert_eq!(sealed, RegionId::new(0));

        let input = builder.add_input(DataType::F64);
        let output = builder.add_variable(DataType::F64);
        builder.add_instruction_unchecked(Instruction::new(
            AddOperation.into(),
            vec![input, input],
            vec![output],
            vec![sealed],
        ));
        let program = builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();

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

        // Borrowed instruction views resolve attached regions against the arena.
        let view = program.instruction(InstructionId::new(program.entry(), 0)).unwrap();
        assert_eq!(view.id(), InstructionId::new(program.entry(), 0));
        assert_eq!(view.instruction().regions(), &[sealed]);
        assert_eq!(view.region(0).unwrap().input_ids(), &[region_input]);
        assert!(program.instruction(InstructionId::new(program.entry(), 9)).is_err());

        // The multi-region program clones, maps, and reports effects across every region.
        let cloned = program.clone();
        assert_eq!(cloned.regions().len(), 2);
        let mapped = program.map_operations(|operation| Ok(operation.clone())).unwrap();
        assert_eq!(mapped.regions().len(), 2);
        assert_eq!(mapped.instructions()[0].regions(), &[sealed]);
        assert!(program.effects().is_pure());

        // TODO(eaplatanios): Fix this as part of later phase of the first-class program regions plan.
        // Legacy single-region rebuild paths reject multi-region programs instead of silently dropping regions.
        assert!(matches!(
            program.simplified(),
            Err(ProgramError::MalformedProgram(message))
                if message == "multi-region programs are not yet supported by this transformation",
        ));
        assert!(matches!(
            program.filtered(&[], program.output_ids(), &[]),
            Err(ProgramError::MalformedProgram(message))
                if message == "multi-region programs are not yet supported by this transformation",
        ));
        assert!(matches!(
            ProgramBuilder::new().add_program(&program.to_flat_program(), &[]),
            Err(ProgramError::MalformedProgram(message))
                if message == "multi-region programs are not yet supported by this transformation",
        ));
        assert!(matches!(
            program.into_simplified(),
            Err(ProgramError::MalformedProgram(message))
                if message == "multi-region programs are not yet supported by this transformation",
        ));
    }

    #[test]
    fn test_program_builder_build_shares_region_across_instructions() {
        // Sharing is legal: several instructions (and several slots of one instruction) may reference one region.
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let mut region_builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let region_input = region_builder.add_input(DataType::F64);
        let region_program = region_builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![region_input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let sealed = builder.add_region(&region_program);
        let input = builder.add_input(DataType::F64);
        let first = builder.add_variable(DataType::F64);
        let second = builder.add_variable(DataType::F64);
        builder.add_instruction_unchecked(Instruction::new(
            AddOperation.into(),
            vec![input, input],
            vec![first],
            vec![sealed, sealed],
        ));
        builder.add_instruction_unchecked(Instruction::new(
            AddOperation.into(),
            vec![first, input],
            vec![second],
            vec![sealed],
        ));
        let program = builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![second], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(program.regions().len(), 2);
        assert_eq!(program.instructions()[0].regions(), &[sealed, sealed]);
        assert_eq!(program.instructions()[1].regions(), &[sealed]);
    }

    #[test]
    fn test_program_builder_build_rejects_malformed_regions() {
        // Instruction regions must reference previously sealed regions (which keeps the graph acyclic by
        // construction).
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let input = builder.add_input(DataType::F64);
        let output = builder.add_variable(DataType::F64);
        builder.add_instruction_unchecked(Instruction::new(
            AddOperation.into(),
            vec![input, input],
            vec![output],
            vec![RegionId::new(3)],
        ));
        assert!(matches!(
            builder.build::<Vec<Scalar>, Vec<Scalar>>(vec![output], vec![Placeholder], vec![Placeholder]),
            Err(ProgramError::MalformedProgram(message))
                if message == "instruction references region ^3 which has not been sealed yet",
        ));

        // Every sealed region must be reachable from the entry root.
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let mut region_builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let region_input = region_builder.add_input(DataType::F64);
        let region_program = region_builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![region_input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        builder.add_region(&region_program);
        let input = builder.add_input(DataType::F64);
        assert!(matches!(
            builder.build::<Vec<Scalar>, Vec<Scalar>>(vec![input], vec![Placeholder], vec![Placeholder]),
            Err(ProgramError::MalformedProgram(message))
                if message == "region ^0 is not reachable from the program entry region",
        ));
    }
}
