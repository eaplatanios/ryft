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
//! [`Region`]s.
//!
//! # Regions, Sharing, and Sealing
//!
//! The canonical region graph and operation-application vocabulary lives in [`regions`](crate::regions); this module
//! owns the surrounding program arena and its construction, validation, transformation, and rendering machinery.
//!
//! Every nested computation (e.g., a control-flow branch or body, a custom-derivative program, a rematerialization
//! program, a JIT-ed callee, etc.) is a [`Region`] in the owning [`Program`]'s one canonical arena, referenced from its
//! instructions through [`Instruction::regions`]. There is exactly one instruction edge kind: sharing is expressed by
//! repeating a [`RegionId`], not by a parallel node table or by operation payloads owning programs. The
//! [`ProgramBuilder`] offers three import policies for nested computations:
//!
//!   - [`ProgramBuilder::import_region`] copies a borrowed [`RegionRef`]'s complete region closure into the arena,
//!     preserving any sharing internal to the imported closure.
//!   - [`ProgramBuilder::import_program`] splices an owned [`Program`]'s arena in directly without cloning, for owned
//!     bodies whose builder would otherwise clone them away.
//!   - [`ProgramBuilder::intern_callee`] interns a shared [`Rc`]-held [`Program`] by pointer identity (i.e., importing
//!     the same `Rc` twice yields the same root [`RegionId`], which is how repeated JIT-compiled calls to one compiled
//!     callee share one region and how lowering deduplication can count occurrences per root).
//!
//! Only *sealed* regions are attachable. [`ProgramBuilder::add_instruction`] validates the attached region list
//! against the operation's declared [`Operation::region_names`] slots, and every non-entry region enters the arena
//! as a complete, immutable program with an explicit boundary (i.e., an explicit [`RegionInterface`]). A region never
//! references atoms of another region directly; values cross region boundaries only through the boundary inputs and
//! outputs, and cross-program constants only through captures (see [`captures`](crate::captures) for the capture-scope
//! model; captures are registered in the trace that owns the instruction, and nested traces reach the root table
//! through their parent chain).
//!
//! [`RegionRef`] borrows any sealed arena region for inspection or replay without cloning it.
//! [`RegionRef::to_program`] materializes that borrowed region back into a standalone flat [`Program`], copying its
//! reachable subtree. Locators such as [`InstructionId`], [`ValueId`], and [`RegionId`] are scoped to the program they
//! were derived from. Materialization and rebuilds renumber arenas, and locators never cross [`Program`] boundaries.
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
use crate::regions::{Region, RegionId, RegionInterface, RegionRef};
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
/// [`MaybeZero`] is the symbolic zero representation shared by transforms like forward-mode and reverse-mode
/// differentiation, where it is the tangent type carried by [`DifferentiationTracer`](crate::DifferentiationTracer)s
/// and the cotangent type that transposition rules consume and produce. A [`MaybeZero::Zero`] means that no value
/// exists and nothing has been staged or computed for it. In the context of differentiation, it means that the
/// corresponding derivative is zero *by construction* (e.g., a disconnected input, a severed tangent, an unused output,
/// etc.), and is not a runtime value that happens to contain zeros. Differentiation rules branch on the variant to skip
/// work entirely. A rule that sees a zero tangent or cotangent emits no operations for it, and "zero-ness" propagates
/// transitively through rules without ever inspecting a program or materializing a buffer. A zero is _materialized_
/// into a real value only at boundaries where one is structurally required (e.g., a nested sub-program operand, a
/// program output, or an eagerly returned tangent), which is also where its carried [`Type`](crate::Type) is consumed.
#[derive(Clone, Debug)]
pub enum MaybeZero<V: Typed> {
    /// Structural zero of the carried [`Type`](crate::Type) (i.e., no value exists and nothing has been staged or
    /// computed for it).
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

/// [`Instruction`]s represent applications of [`Operation`]s to input values in [`Program`]s. Each [`Region`] executes
/// its [`Instruction`]s in sequential order. Beyond its operation and its input and output [`Atom`]s, an instruction
/// carries the [`RegionId`]s of the nested computations attached to the application (e.g., the `true`/`false` branches
/// of a condition, a scan body, or the shared program of a JIT call), in the operation-defined order. Note that there
/// is one [`Region`] edge kind, and sharing is expressed directly in the graph. Several [`Instruction`]s may reference
/// the same [`RegionId`], and a region stays alive for as long as it is reachable from the entry region. What a slot
/// *means* (i.e., a branch-like computation that lowers inline versus a call-like computation that lowers and compiles
/// once as a shared function) is defined by the operation and not by the edge. For example, `if p { f(x) + f(2 * x) }
/// else { x }` with a JIT-compiled `f` is one condition instruction attaching a `true` and a `false` branch [`Region`],
/// where the `true` branch contains two call instructions that both reference the single region holding `f`'s body
/// (i.e., one shared region, three region edges, and the inline-versus-shared lowering decision carried by the
/// condition and call operations, respectively). Two structurally equal but independently created computations
/// remain distinct regions, because [`ProgramBuilder`] imports regions by *identity* (i.e.,
/// [`import_region`](ProgramBuilder::import_region) always copies and
/// [`intern_callee`](ProgramBuilder::intern_callee) interns by [`Rc`] identity), never by structure.
#[derive(Clone, Debug)]
pub struct Instruction<O> {
    /// [`Operation`] applied by this [`Instruction`].
    pub(crate) operation: O,

    /// [`AtomId`]s of the input [`Atom`]s consumed by this [`Instruction`].
    pub(crate) inputs: Vec<AtomId>,

    /// [`AtomId`]s of the output [`Atom`]s produced by this [`Instruction`].
    pub(crate) outputs: Vec<AtomId>,

    /// [`RegionId`]s of the nested computations attached to this [`Instruction`], in the operation-defined order.
    pub(crate) regions: Vec<RegionId>,
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
        self.entry_region_ref().atoms()
    }

    /// Returns the number of input [`Atom`]s (i.e., arguments) of this [`Program`].
    #[inline]
    pub fn input_count(&self) -> usize {
        self.entry_region_ref().input_ids().len()
    }

    /// Returns the [`AtomId`]s of the [`Atom`]s that correspond to the inputs (i.e., arguments) of this [`Program`].
    #[inline]
    pub fn input_ids(&self) -> &[AtomId] {
        self.entry_region_ref().input_ids()
    }

    /// Returns the [`Type`](crate::Type)s of the inputs of this [`Program`], in order.
    #[inline]
    pub fn input_types(&self) -> Vec<V::Type> {
        self.entry_region_ref().input_types()
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
        self.entry_region_ref().output_ids().len()
    }

    /// Returns the [`AtomId`]s of the [`Atom`]s that correspond to the outputs (i.e., return values)
    /// of this [`Program`].
    #[inline]
    pub fn output_ids(&self) -> &[AtomId] {
        self.entry_region_ref().output_ids()
    }

    /// Returns the [`Type`](crate::Type)s of the outputs of this [`Program`], in order.
    #[inline]
    pub fn output_types(&self) -> Vec<V::Type> {
        self.entry_region_ref().output_types()
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
        self.entry_region_ref().instructions()
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

    /// Returns a borrowed view of the [`Region`] that corresponds to the provided [`RegionId`].
    #[inline]
    pub fn region_ref(&self, id: RegionId) -> Result<RegionRef<'_, V, O>, ProgramError> {
        RegionRef::new(self.regions.as_slice(), id)
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

    /// Returns a borrowed view of this [`Program`]'s entry [`Region`].
    #[inline]
    pub fn entry_region_ref(&self) -> RegionRef<'_, V, O> {
        RegionRef::new(self.regions.as_slice(), self.entry).unwrap()
    }

    /// Returns the operation-inference [`RegionInterface`] of this [`Program`]'s entry [`Region`].
    #[inline]
    pub fn interface(&self) -> RegionInterface<V::Type> {
        self.entry_region_ref().interface()
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

    /// Returns the [`Instruction`] at the provided [`InstructionId`].
    pub fn instruction(&self, id: InstructionId) -> Result<&Instruction<O>, ProgramError> {
        let region = self.region(id.region())?;
        region.instructions.get(id.index()).ok_or_else(|| {
            ProgramError::MalformedProgram(format!(
                "instruction index {} is out of range for region {}",
                id.index(),
                id.region(),
            ))
        })
    }

    /// Returns a vector that has the same length as the number of [`Atom`]s in this [`Program`] and for every atom, it
    /// contains the index of the [`Instruction`] that produces it. Note that input and constant atoms are not produced
    /// by an instruction and so the vector contains [`None`] for those atoms.
    #[inline]
    pub fn instruction_by_output(&self) -> Vec<Option<usize>> {
        self.entry_region().instruction_by_output()
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

    /// Returns the [`Effect`](crate::Effect) classes reachable from this [`Program`]'s entry region, or
    /// [`Effects::PURE`] for programs with no instructions. Because attached regions live in the same arena,
    /// nested-computation effects are visible through higher-order boundaries without any per-operation forwarding.
    /// The per-[`Instruction`] counterpart to this function is [`Self::instruction_effects`], which merges one
    /// instruction's own effects with its attached [`Region`]s' effects.
    #[inline]
    pub fn effects(&self) -> Effects {
        self.entry_region_ref().effects()
    }

    /// Returns the [`Effect`](crate::Effect) classes of the [`Instruction`] at the provided [`InstructionId`]. That is
    /// defined as the union of its [`Operation`]'s intrinsic [`Operation::effects`] and the recursively derived effects
    /// of its attached [`Region`]s (including regions attached to instructions inside those regions). Consulting only
    /// the operation's intrinsic effects would be unsound for region-carrying instructions because an effect inside an
    /// attached region is observable whenever the instruction executes that region.
    pub fn instruction_effects(&self, id: InstructionId) -> Result<Effects, ProgramError> {
        let instruction = self.instruction(id)?;
        let mut effects = instruction.operation().effects();
        if !instruction.regions().is_empty() {
            let instruction_effects = Region::effects(self.regions.as_slice());
            for attached in instruction.regions().iter().copied() {
                effects = effects.union(instruction_effects[attached.index()]);
            }
        }
        Ok(effects)
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
    /// output arities. This is the canonical shape for standalone nested computations supplied positionally through the
    /// `regions` argument of [`Context::bind`], including both owned [`Region`]s and shared callees, without needing to
    /// preserve the caller's original [`Parameterized`] type.
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
        // Simplify every region independently. A region's inputs and outputs are its boundary contract and always
        // survive, so per-region dead-code elimination only removes internal dead work. Retained instructions keep
        // their attached-region references, and regions that lose their last reference are dropped afterward by the
        // compaction step, which also rewrites the surviving references.
        let effects = Region::effects(self.regions.as_slice());
        let regions = self
            .regions
            .iter()
            .map(|region| {
                let instruction_by_output = region.instruction_by_output();
                let mut new_atoms = Vec::with_capacity(region.atoms.len());
                let mut new_input_ids = Vec::with_capacity(region.input_ids.len());
                let mut new_instructions = Vec::with_capacity(region.instructions.len());
                let mut atom_id_mapping = HashMap::with_capacity(region.atoms.len());
                for input_id in region.input_ids.iter().copied() {
                    let input = region.atoms.get(input_id.index).ok_or(ProgramError::UnboundAtomId { id: input_id })?;
                    let Atom::Variable(input_type) = input else {
                        return Err(ProgramError::MalformedProgram(
                            "program input atom was not a variable".to_string(),
                        ));
                    };
                    let new_input = AtomId { index: new_atoms.len() };
                    new_atoms.push(Atom::Variable(input_type.clone()));
                    new_input_ids.push(new_input);
                    atom_id_mapping.insert(input_id, new_input);
                }

                // Make sure that effectful instructions and their transitive dependencies are processed in original
                // instruction order before the outputs, so that instructions with observable effects survive even
                // when dead and ordered effects keep their relative order.
                for instruction in region.instructions.iter() {
                    let mut instruction_effects = instruction.operation().effects();
                    for attached in instruction.regions().iter().copied() {
                        instruction_effects = instruction_effects.union(effects[attached.index()]);
                    }
                    if instruction_effects.is_pure() {
                        continue;
                    }
                    if instruction.outputs().is_empty() {
                        let inputs = instruction
                            .inputs()
                            .iter()
                            .copied()
                            .map(|input| {
                                clone_atom_subgraph_into_region(
                                    &mut atom_id_mapping,
                                    input,
                                    region,
                                    instruction_by_output.as_slice(),
                                    &mut new_atoms,
                                    &mut new_instructions,
                                )
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        new_instructions.push(Instruction::new(
                            instruction.operation().clone(),
                            inputs,
                            Vec::new(),
                            instruction.regions().to_vec(),
                        ));
                        continue;
                    }
                    for output_id in instruction.outputs().iter().copied() {
                        clone_atom_subgraph_into_region(
                            &mut atom_id_mapping,
                            output_id,
                            region,
                            instruction_by_output.as_slice(),
                            &mut new_atoms,
                            &mut new_instructions,
                        )?;
                    }
                }

                let output_ids = region
                    .output_ids
                    .iter()
                    .copied()
                    .map(|output| {
                        clone_atom_subgraph_into_region(
                            &mut atom_id_mapping,
                            output,
                            region,
                            instruction_by_output.as_slice(),
                            &mut new_atoms,
                            &mut new_instructions,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                Ok(Region { atoms: new_atoms, input_ids: new_input_ids, output_ids, instructions: new_instructions })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let (regions, entry) = compact_regions(regions, self.entry);
        Ok(Self {
            input_structure: self.input_structure.clone(),
            output_structure: self.output_structure.clone(),
            regions,
            entry,
            marker: PhantomData,
        })
    }

    /// Consumes this [`Program`] and returns a simplified version with dead constants and [`Instruction`]s that do not
    /// contribute to the [`Program`]'s output removed. Unlike [`Self::simplified`], this method moves live [`Atom`]s,
    /// [`Instruction`]s, and parameter structures into the returned [`Program`] instead of cloning them. This avoids
    /// copying constants and operations that are discarded during simplification. The behavior of [`Self::simplified`]
    /// around [`Effects`] applies here too. [`Instruction`]s whose operations are not [`Effects::PURE`] survive in
    /// their original relative order even when no program output consumes their outputs.
    pub fn into_simplified(self) -> Result<Self, ProgramError> {
        let expected_input_count = self.input_structure.parameter_count();
        check_count!("input", self.input_ids(), expected_input_count, ProgramError);

        let expected_output_count = self.output_structure.parameter_count();
        check_count!("output", self.output_ids(), expected_output_count, ProgramError);

        // Simplify every region independently, exactly like `Self::simplified` but moving live atoms and
        // instructions into the rebuilt regions instead of cloning them.
        let arena_effects = Region::effects(self.regions.as_slice());
        let Self { regions, input_structure, output_structure, entry, .. } = self;
        let regions = regions
            .into_iter()
            .map(|region| {
                let instruction_by_output = region.instruction_by_output();
                let effectful_instructions = region
                    .instructions
                    .iter()
                    .enumerate()
                    .filter(|instruction| {
                        let mut effects = instruction.1.operation().effects();
                        for attached in instruction.1.regions().iter().copied() {
                            effects = effects.union(arena_effects[attached.index()]);
                        }
                        !effects.is_pure()
                    })
                    .map(|(index, instruction)| (index, instruction.outputs().to_vec()))
                    .collect::<Vec<_>>();
                let Region { atoms, input_ids, output_ids, instructions } = region;
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
                        return Err(ProgramError::MalformedProgram(
                            "program input atom was not a variable".to_string(),
                        ));
                    };
                    let new_input = AtomId { index: new_atoms.len() };
                    new_atoms.push(Atom::Variable(input_type));
                    new_input_ids.push(new_input);
                    atom_id_mapping.insert(input_id, new_input);
                }

                // Make sure that effectful instructions and their transitive dependencies are processed in original
                // instruction order before the outputs, so that instructions with observable effects survive even
                // when dead and ordered effects keep their relative order.
                for (instruction_index, outputs) in effectful_instructions {
                    if outputs.is_empty() {
                        let instruction = instructions[instruction_index]
                            .take()
                            .ok_or(ProgramError::MalformedProgram("instruction was already moved".to_string()))?;
                        let inputs = instruction
                            .inputs()
                            .iter()
                            .copied()
                            .map(|input| {
                                move_atom_to_program(
                                    &mut atom_id_mapping,
                                    input,
                                    atoms.as_mut_slice(),
                                    instructions.as_mut_slice(),
                                    instruction_by_output.as_slice(),
                                    &mut new_atoms,
                                    &mut new_instructions,
                                )
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        new_instructions.push(Instruction::new(
                            instruction.operation,
                            inputs,
                            Vec::new(),
                            instruction.regions,
                        ));
                        continue;
                    }
                    for root in outputs {
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

                Ok(Region { atoms: new_atoms, input_ids: new_input_ids, output_ids, instructions: new_instructions })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let (regions, entry) = compact_regions(regions, entry);
        Ok(Self { input_structure, output_structure, regions, entry, marker: PhantomData })
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
        let (instruction_by_output, input_liveness) = self.compute_live_inputs(inputs, outputs, keep_alive)?;
        let entry_region = self.entry_region();
        let mut new_atoms = Vec::with_capacity(entry_region.atoms.len());
        let mut new_input_ids = Vec::new();
        let mut new_instructions = Vec::with_capacity(entry_region.instructions.len());
        let mut atom_id_mapping = HashMap::with_capacity(entry_region.atoms.len());
        let mut live_input_indices = Vec::new();

        for (position, id) in inputs.iter().copied().enumerate() {
            if !input_liveness[position] {
                continue;
            }
            let Atom::Variable(input_type) = &entry_region.atoms[id.index()] else {
                return Err(ProgramError::MalformedProgram(format!("filter input atom {id} is not a variable")));
            };
            let new_input = AtomId { index: new_atoms.len() };
            new_atoms.push(Atom::Variable(input_type.clone()));
            new_input_ids.push(new_input);
            atom_id_mapping.insert(id, new_input);
            live_input_indices.push(position);
        }

        // Make sure that the keep-alive-atom-producing instructions and their transitive dependencies are processed in
        // original instruction order before the outputs, so that instructions with observable effects survive even when
        // dead and ordered effects keep their relative order.
        for root in keep_alive.iter().copied() {
            clone_atom_subgraph_into_region(
                &mut atom_id_mapping,
                root,
                entry_region,
                instruction_by_output.as_slice(),
                &mut new_atoms,
                &mut new_instructions,
            )?;
        }

        let output_ids = outputs
            .iter()
            .copied()
            .map(|id| {
                clone_atom_subgraph_into_region(
                    &mut atom_id_mapping,
                    id,
                    entry_region,
                    instruction_by_output.as_slice(),
                    &mut new_atoms,
                    &mut new_instructions,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Nested regions of retained instructions pass through unchanged (filtering is an entry-boundary projection);
        // regions that lost their last reference are dropped and the surviving references are rewritten.
        let mut regions = self.regions[..self.entry.index()].to_vec();
        regions.push(Region { atoms: new_atoms, input_ids: new_input_ids, output_ids, instructions: new_instructions });
        let (regions, entry) = compact_regions(regions, self.entry);
        let program = Program {
            input_structure: vec![Placeholder; live_input_indices.len()],
            output_structure: vec![Placeholder; outputs.len()],
            regions,
            entry,
            marker: PhantomData,
        };
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
        let entry = self.entry;
        let mut nested_regions = self.regions;
        let Region { atoms, instructions, .. } = nested_regions.pop().unwrap();
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

        // Nested regions of retained instructions pass through unchanged (filtering is an entry-boundary projection).
        // Regions that lost their last reference are dropped, and the surviving references are rewritten.
        nested_regions.push(Region {
            atoms: new_atoms,
            input_ids: new_input_ids,
            output_ids,
            instructions: new_instructions,
        });

        let (regions, entry) = compact_regions(nested_regions, entry);
        Ok((Program { input_structure, output_structure, regions, entry, marker: PhantomData }, live_input_indices))
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
    /// are nested within other programs like with control flow [`Operation`]s. [`Instruction`]s with attached
    /// [`Region`]s render a bracketed region section after their inputs, pairing each region with its declared
    /// name from [`Operation::region_names`] (falling back to the region index for undeclared regions). A region
    /// referenced exactly once renders nested beneath its referencing instruction, while a region referenced multiple
    /// times renders its body exactly once (at its first reference, labeled with its [`RegionId`]), and every later
    /// reference renders as that identifier alone. [`RegionId`]s are arena indices and therefore deterministic
    /// [`Program`]-local names.
    pub fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        /// Renders one [`Region`] as a `lambda ... in (...)` block, recursively rendering the regions attached to its
        /// instructions according to `reference_counts` and `rendered`.
        fn render_region<V: Value, O: Operation<V::Type>>(
            regions: &[Region<V, O>],
            id: RegionId,
            formatter: &mut std::fmt::Formatter<'_>,
            indentation: usize,
            reference_counts: &[usize],
            rendered: &mut [bool],
        ) -> std::fmt::Result {
            let region = &regions[id.index()];
            write!(formatter, "{:indentation$}", "")?;
            write!(formatter, "lambda ")?;
            region.input_ids.iter().enumerate().try_for_each(|(index, input_id)| {
                if index > 0 {
                    write!(formatter, ", {input_id}:{}", region.atoms[input_id.index].r#type())
                } else {
                    write!(formatter, "{input_id}:{}", region.atoms[input_id.index].r#type())
                }
            })?;
            writeln!(formatter, " .")?;
            let mut instructions_by_first_output = vec![None; region.atoms.len()];
            for (index, instruction) in region.instructions.iter().enumerate() {
                if let Some(output_id) = instruction.outputs.first() {
                    instructions_by_first_output[output_id.index] = Some(index);
                }
            }
            let mut binding_count = 0usize;
            let mut is_input = vec![false; region.atoms.len()];
            for input_id in region.input_ids.iter().copied() {
                is_input[input_id.index] = true;
            }
            for (atom_id, atom) in region.atoms.iter().enumerate() {
                match atom {
                    Atom::Constant(_) => {
                        write!(formatter, "{:indentation$}", "")?;
                        writeln!(
                            formatter,
                            "{} {}:{} = const",
                            if binding_count == 0 { "let" } else { "   " },
                            AtomId { index: atom_id },
                            region.atoms[atom_id].r#type()
                        )?;
                        binding_count += 1;
                    }
                    Atom::Variable(_) if is_input[atom_id] => {}
                    Atom::Variable(_) => {
                        if let Some(instruction_index) = instructions_by_first_output[atom_id] {
                            let instruction = &region.instructions[instruction_index];
                            let line_indentation = if binding_count == 0 { indentation } else { indentation + 4 };
                            write!(formatter, "{:indentation$}", "")?;
                            write!(formatter, "{} ", if binding_count == 0 { "let" } else { "   " })?;
                            instruction.outputs.iter().enumerate().try_for_each(|(index, output)| {
                                if index > 0 {
                                    write!(formatter, ", {output}:{}", region.atoms[output.index].r#type())
                                } else {
                                    write!(formatter, "{output}:{}", region.atoms[output.index].r#type())
                                }
                            })?;
                            write!(formatter, " = ")?;
                            instruction.operation.render(formatter, line_indentation)?;
                            instruction.inputs.iter().try_for_each(|input| write!(formatter, " {input}"))?;
                            if !instruction.regions.is_empty() {
                                let names = instruction.operation.region_names();
                                write!(formatter, " [")?;
                                for (slot, attached) in instruction.regions.iter().copied().enumerate() {
                                    writeln!(formatter)?;
                                    write!(formatter, "{:width$}", "", width = line_indentation + 4)?;
                                    match names.get(slot) {
                                        Some(name) => write!(formatter, "{name}=")?,
                                        None => write!(formatter, "{slot}=")?,
                                    }
                                    let is_shared = reference_counts[attached.index()] > 1;
                                    if is_shared && rendered[attached.index()] {
                                        write!(formatter, "{attached},")?;
                                        continue;
                                    }
                                    rendered[attached.index()] = true;
                                    if is_shared {
                                        write!(formatter, "{attached}=")?;
                                    }
                                    writeln!(formatter, "{{")?;
                                    render_region(
                                        regions,
                                        attached,
                                        formatter,
                                        line_indentation + 8,
                                        reference_counts,
                                        rendered,
                                    )?;
                                    writeln!(formatter)?;
                                    write!(formatter, "{:width$}", "", width = line_indentation + 4)?;
                                    write!(formatter, "}},")?;
                                }
                                writeln!(formatter)?;
                                write!(formatter, "{:width$}", "", width = line_indentation)?;
                                write!(formatter, "]")?;
                            }
                            writeln!(formatter)?;
                            binding_count += 1;
                        };
                    }
                }
            }
            write!(formatter, "{:indentation$}", "")?;
            write!(formatter, "in (")?;
            region.output_ids.iter().enumerate().try_for_each(|(index, output)| {
                if index > 0 { write!(formatter, ", {output}") } else { write!(formatter, "{output}") }
            })?;
            write!(formatter, ")")
        }

        let mut reference_counts = vec![0usize; self.regions.len()];
        for region in &self.regions {
            for instruction in &region.instructions {
                for attached in instruction.regions().iter().copied() {
                    reference_counts[attached.index()] += 1;
                }
            }
        }

        let mut rendered = vec![false; self.regions.len()];
        render_region(
            self.regions.as_slice(),
            self.entry,
            formatter,
            indentation,
            reference_counts.as_slice(),
            rendered.as_mut_slice(),
        )
    }
}

impl<V: Value, O: Clone, Input: Parameterized<V>, Output: Parameterized<V>> Clone for Program<V, O, Input, Output> {
    #[inline]
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
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

/// _Flat_ [`Program`] (i.e., with flat `Vec`-valued inputs and outputs) over a [`Domain`]'s constant and operation
/// universe. This is the canonical shape for nested computations constructed standalone, including owned region
/// attachments and shared callees composed into the `regions` argument of [`Context::bind`]. Borrowed replay exposes
/// regions through [`BindingRegionDriver`](crate::BindingRegionDriver) without converting them into this owned shape.
pub type FlatProgram<D> = Program<
    <D as Domain>::Constant,
    <D as Domain>::Operation,
    Vec<<D as Domain>::Constant>,
    Vec<<D as Domain>::Constant>,
>;

/// Liveness masks for a [`Program`]'s entry [`Region`]. The masks are indexed by entry-region [`Atom`] and
/// [`Instruction`] positions. Nested regions are not part of this analysis because their inputs and outputs are their
/// boundary contract (i.e., a referenced region is live exactly when a live instruction references it, which the
/// region-aware rebuild paths such as [`Program::simplified`] handle directly).
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

    /// Returns a borrowed view of the already sealed builder region that corresponds to the provided [`RegionId`].
    #[inline]
    pub fn region_ref(&self, id: RegionId) -> Result<RegionRef<'_, V, O>, ProgramError> {
        RegionRef::new(self.regions.as_slice(), id)
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
    /// provided [`Operation`] with the provided previously sealed regions attached in the operation-defined region
    /// order (region-free operations pass an empty `regions` list) to the provided input [`Atom`]s. The number of
    /// attached regions must match the operation's declared [`Operation::region_names`] slot count. Output types are
    /// inferred through [`Operation::infer_output_types`], with the attached regions' [`RegionInterface`]s derived
    /// from this builder's arena on the spot; interfaces are never stored, and final [`Self::build`] validation
    /// derives them again from the frozen arena.
    pub fn add_instruction<P: Into<O>>(
        &mut self,
        operation: P,
        regions: Vec<RegionId>,
        inputs: Vec<AtomId>,
    ) -> Result<&[AtomId], ProgramError> {
        let operation = operation.into();
        let region_names = operation.region_names();
        if regions.len() != region_names.len() {
            return Err(ProgramError::MalformedProgram(format!(
                "operation `{}` declares {} region slots but {} regions were attached",
                operation.name(),
                region_names.len(),
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
                    .get(input.index)
                    .map(|atom| atom.r#type().into_owned())
                    .ok_or(ProgramError::UnboundAtomId { id: *input })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let region_interfaces = if regions.is_empty() {
            Vec::new()
        } else {
            let effects = Region::effects(self.regions.as_slice());
            regions
                .iter()
                .map(|region_id| {
                    let region = &self.regions[region_id.index()];
                    RegionInterface::new(region.input_types(), region.output_types(), effects[region_id.index()])
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
    /// output order. This is a plain relocation and not a re-interpretation or partial evaluation. Every instruction
    /// and live constant of the provided program is rebuilt verbatim into this builder. It is, for example,
    /// the reconciliation primitive an unknown-predicate `condition` uses to graft each branch's residual program
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
            && remaining.iter().any(|region| !std::ptr::eq(first.regions(), region.regions()))
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
        let source_regions = region.regions();
        let mut imported = region.region().clone();
        for instruction in &mut imported.instructions {
            for attached in &mut instruction.regions {
                let nested = RegionRef::new(source_regions, *attached).unwrap();
                *attached = self.import_region_with_remapping(nested, remapping);
            }
        }
        let id = RegionId::new(self.regions.len());
        self.regions.push(imported);
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
        let offset = self.regions.len();
        let Program { mut regions, entry, .. } = program;
        for region in &mut regions {
            for instruction in &mut region.instructions {
                for attached in &mut instruction.regions {
                    *attached = RegionId::new(attached.index() + offset);
                }
            }
        }
        self.regions.extend(regions);
        RegionId::new(entry.index() + offset)
    }

    /// Imports `callee` if it has not previously been imported into this builder and otherwise returns the existing
    /// callee root [`RegionId`]. Callees are identified by [`Rc`] identity, not structural equality, so structurally
    /// equal but independently built programs remain distinct.
    pub fn intern_callee(&mut self, callee: &Rc<Program<V, O, Vec<V>, Vec<V>>>) -> RegionId
    where
        O: Clone,
    {
        if let Some((_, id)) = self.callees.iter().find(|(interned, _)| Rc::ptr_eq(interned, callee)) {
            return *id;
        }
        let id = self.import_region(callee.entry_region_ref());
        self.callees.push((callee.clone(), id));
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
        // because region imports copy them in post-order (i.e., children before parents). Every referenced region
        // identifier is therefore in range, and the region graph is acyclic, which is what allows the reachability walk
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

        // Every instruction's attached-region count must match its operation's declared slot count. The checked
        // instruction path already enforced this at insertion time for the entry region, but instructions can also
        // arrive through the unchecked path, and so the final validation re-checks the complete frozen arena.
        for region in &regions {
            for instruction in &region.instructions {
                let declared = instruction.operation().region_names().len();
                if instruction.regions.len() != declared {
                    return Err(ProgramError::MalformedProgram(format!(
                        "operation `{}` declares {} region slots but {} regions were attached",
                        instruction.operation().name(),
                        declared,
                        instruction.regions.len(),
                    )));
                }
            }
        }

        Ok(Program { input_structure, output_structure, regions, entry, marker: PhantomData })
    }
}

// TODO(eaplatanios): Review this.
/// Copies the [`Atom`] that corresponds to `atom_id` in `region` (and its transitive producers) into
/// `new_atoms`/`new_instructions`, memoizing the old-to-new [`AtomId`] mapping in `atom_id_mapping`. Atoms already
/// present in the mapping (e.g., rebuilt region inputs) are reused, [`Atom::Constant`]s are cloned directly, and
/// [`Atom::Variable`]s are reconstructed from their producing [`Instruction`], whose attached-region references are
/// preserved verbatim (unreferenced regions are dropped and identifiers rewritten by [`compact_regions`] afterwards).
/// A reachable variable that is neither mapped nor produced by an instruction is reported as a
/// [`ProgramError::MalformedProgram`].
fn clone_atom_subgraph_into_region<V: Value, O: Operation<V::Type>>(
    atom_id_mapping: &mut HashMap<AtomId, AtomId>,
    atom_id: AtomId,
    region: &Region<V, O>,
    instruction_by_output: &[Option<usize>],
    new_atoms: &mut Vec<Atom<V>>,
    new_instructions: &mut Vec<Instruction<O>>,
) -> Result<AtomId, ProgramError> {
    if let Some(mapped_atom) = atom_id_mapping.get(&atom_id) {
        return Ok(*mapped_atom);
    }
    let atom = region.atoms.get(atom_id.index).ok_or(ProgramError::UnboundAtomId { id: atom_id })?;
    let atom = match atom {
        Atom::Constant(value) => {
            let new_atom = AtomId { index: new_atoms.len() };
            new_atoms.push(Atom::Constant(value.clone()));
            Ok(new_atom)
        }
        Atom::Variable(_) => {
            let instruction_index = instruction_by_output
                .get(atom_id.index)
                .copied()
                .flatten()
                .ok_or(ProgramError::MalformedProgram("variable atom has no owning instruction".to_string()))?;
            let instruction = &region.instructions[instruction_index];
            let inputs = instruction
                .inputs
                .iter()
                .copied()
                .map(|input| {
                    clone_atom_subgraph_into_region(
                        atom_id_mapping,
                        input,
                        region,
                        instruction_by_output,
                        new_atoms,
                        new_instructions,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mut outputs = Vec::with_capacity(instruction.outputs.len());
            for output in instruction.outputs.iter().copied() {
                let output_atom = region.atoms.get(output.index).ok_or(ProgramError::UnboundAtomId { id: output })?;
                let Atom::Variable(output_type) = output_atom else {
                    return Err(ProgramError::MalformedProgram(
                        "instruction output atom was not a variable".to_string(),
                    ));
                };
                let new_output = AtomId { index: new_atoms.len() };
                new_atoms.push(Atom::Variable(output_type.clone()));
                atom_id_mapping.insert(output, new_output);
                outputs.push(new_output);
            }
            new_instructions.push(Instruction::new(
                instruction.operation.clone(),
                inputs,
                outputs,
                instruction.regions.clone(),
            ));
            atom_id_mapping
                .get(&atom_id)
                .copied()
                .ok_or(ProgramError::MalformedProgram("remapped instruction output was missing".to_string()))
        }
    }?;
    atom_id_mapping.insert(atom_id, atom);
    Ok(atom)
}

// TODO(eaplatanios): Review this.
/// Drops the [`Region`]s in `regions` that are not reachable from `entry` (following instruction attached-region
/// references), compacts the surviving regions' identifiers while preserving their relative order, and rewrites every
/// surviving instruction's references accordingly. Returns the compacted arena together with the remapped entry
/// identifier. Order preservation keeps the sealed-before-referenced invariant intact, so the compacted arena remains
/// valid for ascending-order recursive derivations such as [`Region::effects`].
fn compact_regions<V: Typed, O>(regions: Vec<Region<V, O>>, entry: RegionId) -> (Vec<Region<V, O>>, RegionId) {
    let mut reachable = vec![false; regions.len()];
    let mut pending = vec![entry];
    while let Some(current) = pending.pop() {
        if std::mem::replace(&mut reachable[current.index()], true) {
            continue;
        }
        for instruction in &regions[current.index()].instructions {
            pending.extend(instruction.regions().iter().copied());
        }
    }
    let mut remapping = vec![None; regions.len()];
    let mut kept = 0usize;
    for (index, is_reachable) in reachable.iter().copied().enumerate() {
        if is_reachable {
            remapping[index] = Some(RegionId::new(kept));
            kept += 1;
        }
    }
    let mut compacted = Vec::with_capacity(kept);
    for (index, mut region) in regions.into_iter().enumerate() {
        if !reachable[index] {
            continue;
        }
        for instruction in &mut region.instructions {
            for attached in &mut instruction.regions {
                *attached = remapping[attached.index()].unwrap();
            }
        }
        compacted.push(region);
    }
    (compacted, remapping[entry.index()].unwrap())
}

/// Moves the [`Atom`] that corresponds to `atom_id` (and its transitive producers) out of `atoms`/`instructions` into
/// `new_atoms`/`new_instructions`, memoizing the old-to-new [`AtomId`] mapping in `atom_id_mapping`. This is the
/// move-based counterpart of [`clone_atom_subgraph_into_region`]: it relocates owned [`Atom`]s and [`Instruction`]s
/// (including their attached-region references, verbatim) instead of cloning them, so each is taken from its slot at
/// most once. Atoms already present in the mapping are reused, and a reachable variable that is neither mapped nor
/// produced by an instruction is reported as a [`ProgramError::MalformedProgram`].
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
    new_instructions.push(Instruction::new(instruction.operation, inputs, outputs, instruction.regions));
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
    use crate::tests::TestRegionOperation;
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

        fn infer_output_types(
            &self,
            input_types: &[DataType],
            _region_interfaces: &[RegionInterface<DataType>],
        ) -> Result<Vec<DataType>, TypeError> {
            check_count!("input", input_types, 1, TypeError);
            Ok(vec![input_types[0].clone()])
        }

        fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
            OperationFormatter::new(formatter, indentation, self.name())?
                .bracketed(|operation| operation.field("value", Self::METADATA_VALUE))
        }
    }

    /// Effectful test operation with no results, used to pin simplification's zero-output liveness behavior.
    #[derive(Clone, Debug)]
    struct ZeroOutputEffectOperation;

    impl Operation<DataType> for ZeroOutputEffectOperation {
        fn name(&self) -> &'static str {
            "zero_output_effect"
        }

        fn infer_output_types(
            &self,
            input_types: &[DataType],
            _region_interfaces: &[RegionInterface<DataType>],
        ) -> Result<Vec<DataType>, TypeError> {
            check_count!("input", input_types, 1, TypeError);
            Ok(Vec::new())
        }

        fn effects(&self) -> Effects {
            Effects::single(Effect::OrderedIo)
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
        let o0 = builder.add_instruction(AddOperation, Vec::new(), vec![i0, c0]).unwrap()[0];
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
        let v0 = builder.add_instruction(NegOperation, Vec::new(), vec![i0]).unwrap()[0];
        let o0 = builder.add_instruction(AddOperation, Vec::new(), vec![v0, i1]).unwrap()[0];
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
        let o0 = builder.add_instruction(LongMetadataOperation, Vec::new(), vec![i0]).unwrap()[0];
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
        let o0 = builder.add_instruction(AddOperation, Vec::new(), vec![i0, i0]).unwrap()[0];
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
        let o0 = builder.add_instruction(AddOperation, Vec::new(), vec![i0, v0]).unwrap()[0];
        assert!(matches!(
            builder.build::<Scalar, Scalar>(vec![o0], Placeholder, Placeholder),
            Err(ProgramError::MalformedProgram(message)) if message == "variable atom has no owning instruction",
        ));
    }

    #[test]
    fn test_program_instruction_by_output() {
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let input = builder.add_input(DataType::F64);
        let constant = builder.add_constant(Scalar::from(3.0f64));
        let scaled = builder.add_instruction(NegOperation, Vec::new(), vec![input]).unwrap()[0];
        let output = builder.add_instruction(AddOperation, Vec::new(), vec![scaled, constant]).unwrap()[0];
        let dead_output = builder.add_instruction(NegOperation, Vec::new(), vec![input]).unwrap()[0];
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
        let scaled = builder.add_instruction(NegOperation, Vec::new(), vec![live_input]).unwrap()[0];
        let output = builder.add_instruction(AddOperation, Vec::new(), vec![scaled, live_constant]).unwrap()[0];
        let dead_output =
            builder.add_instruction(AddOperation, Vec::new(), vec![dead_input, dead_constant]).unwrap()[0];
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
        let negated = builder.add_instruction(NegOperation, Vec::new(), vec![input]).unwrap()[0];
        let combined = builder.add_instruction(AddOperation, Vec::new(), vec![negated, constant]).unwrap()[0];
        let output = builder
            .add_instruction(CompareOperation::new(ComparisonDirection::LessThan), Vec::new(), vec![combined, constant])
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
        let v0 = builder.add_instruction(NegOperation, Vec::new(), vec![i0]).unwrap()[0];
        let o0 = builder.add_instruction(AddOperation, Vec::new(), vec![v0, i1]).unwrap()[0];
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
        let _ = builder.add_instruction(AddOperation, Vec::new(), vec![i0, c0]).unwrap()[0];
        let v1 = builder.add_instruction(AddOperation, Vec::new(), vec![i0, c1]).unwrap()[0];
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
            let doubled = builder.add_instruction(AddOperation, Vec::new(), vec![input, input]).unwrap()[0];
            let _printed = builder.add_instruction(PrintOperation::new("x"), Vec::new(), vec![input]).unwrap()[0];
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

        // An effectful instruction with no outputs must itself be rooted: there is no result atom from which either
        // simplification implementation could otherwise discover it.
        let build_zero_output_effect = || {
            let mut builder = ProgramBuilder::<Scalar, ZeroOutputEffectOperation>::new();
            let input = builder.add_input(DataType::F64);
            assert!(builder.add_instruction(ZeroOutputEffectOperation, Vec::new(), vec![input]).unwrap().is_empty());
            builder.build::<Scalar, Vec<Scalar>>(Vec::new(), Placeholder, Vec::new()).unwrap()
        };
        let zero_output_effect = build_zero_output_effect();
        assert_eq!(zero_output_effect.simplified().unwrap().instructions().len(), 1);
        assert_eq!(build_zero_output_effect().into_simplified().unwrap().instructions().len(), 1);
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
        let v0 = builder.add_instruction(AddOperation, Vec::new(), vec![i0, c0]).unwrap()[0];
        let v1 = builder.add_instruction(AddOperation, Vec::new(), vec![i0, c1]).unwrap()[0];
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
        let v0 = builder.add_instruction(NegOperation, Vec::new(), vec![i0]).unwrap()[0];
        let v1 = builder.add_instruction(AddOperation, Vec::new(), vec![v0, c0]).unwrap()[0];
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
            let v0 = builder.add_instruction(NegOperation, Vec::new(), vec![i0]).unwrap()[0];
            let v1 = builder.add_instruction(AddOperation, Vec::new(), vec![v0, c0]).unwrap()[0];
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
        let v0 = builder.add_instruction(NegOperation, Vec::new(), vec![i0]).unwrap()[0];
        let v1 = builder.add_instruction(AddOperation, Vec::new(), vec![v0, i1]).unwrap()[0];
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

        // `splice_program` appends the program's reachable instructions into a fresh builder, remapping its inputs to
        // the provided builder atoms and returning the builder atoms for its outputs. The program's `2.0` constant is
        // dead (i.e., no instruction consumes it), and so only the two reachable instructions are rebuilt.
        let mut outer = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let a0 = outer.add_input(DataType::F64);
        let a1 = outer.add_input(DataType::F64);
        let outputs = outer.splice_program(&program, &[a0, a1]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outer.instructions.len(), 2);
        let outer_program =
            outer.build::<(Scalar, Scalar), Scalar>(outputs, (Placeholder, Placeholder), Placeholder).unwrap();
        assert_eq!(outer_program.interpret((Scalar::from(2.0f64), Scalar::from(38.0f64))), Ok(Scalar::from(36.0f64)));
    }

    #[test]
    fn test_program_builder_rejects_unbound_instruction_inputs() {
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let v0 = builder.add_instruction(AddOperation, Vec::new(), vec![AtomId { index: 42 }, AtomId { index: 99 }]);
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

    #[test]
    fn test_program_builder_import_region_and_intern_callee() {
        // A source program with one sealed region attached to its entry instruction.
        let mut source_builder = ProgramBuilder::<Scalar, TestRegionOperation>::new();
        let mut region_builder = ProgramBuilder::<Scalar, TestRegionOperation>::new();
        let region_input = region_builder.add_input(DataType::F64);
        let region_program = region_builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![region_input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let sealed = source_builder.import_region(region_program.entry_region_ref());
        let input = source_builder.add_input(DataType::F64);
        let output = source_builder
            .add_instruction(TestRegionOperation::WithRegions(&["body"]), vec![sealed], vec![input])
            .unwrap()[0];
        let source = source_builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        // Fresh borrowed imports copy the complete closure independently: two imports produce two subtrees.
        let mut destination = ProgramBuilder::<Scalar, TestRegionOperation>::new();
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
        let mut destination = ProgramBuilder::<Scalar, TestRegionOperation>::new();
        let first = destination.intern_callee(&flat);
        let second = destination.intern_callee(&flat);
        let third = destination.intern_callee(&equal_but_distinct);
        assert_eq!(first, second);
        assert_ne!(first, third);
    }

    #[test]
    fn test_program_builder_build_multi_region_program() {
        let mut builder = ProgramBuilder::<Scalar, TestRegionOperation>::new();
        let mut region_builder = ProgramBuilder::<Scalar, TestRegionOperation>::new();
        let region_input = region_builder.add_input(DataType::F64);
        let doubled = region_builder
            .add_instruction(TestRegionOperation::Add, Vec::new(), vec![region_input, region_input])
            .unwrap()[0];
        let region_program = region_builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![doubled], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let sealed = builder.import_region(region_program.entry_region_ref());
        assert_eq!(sealed, RegionId::new(0));

        let input = builder.add_input(DataType::F64);
        let output = builder
            .add_instruction(TestRegionOperation::WithRegions(&["body"]), vec![sealed], vec![input])
            .unwrap()[0];
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
        let mut relocation_builder = ProgramBuilder::<Scalar, TestRegionOperation>::new();
        let relocation_input = relocation_builder.add_input(DataType::F64);
        let relocated_outputs =
            relocation_builder.splice_program(&program.to_flat_program(), &[relocation_input]).unwrap();
        let relocated = relocation_builder
            .build::<Vec<Scalar>, Vec<Scalar>>(relocated_outputs, vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(relocated.regions().len(), 2);
        let simplified = program.into_simplified().unwrap();
        assert_eq!(simplified.regions().len(), 2);
    }

    #[test]
    fn test_program_builder_region_ref_and_import_region() {
        let mut source_builder = ProgramBuilder::<Scalar, TestRegionOperation>::new();
        let mut region_builder = ProgramBuilder::<Scalar, TestRegionOperation>::new();
        let region_input = region_builder.add_input(DataType::F64);
        let region_program = region_builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![region_input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let sealed = source_builder.import_region(region_program.entry_region_ref());
        let sealed_ref = source_builder.region_ref(sealed).unwrap();
        assert_eq!(sealed_ref.id(), sealed);
        assert_eq!(sealed_ref.input_types(), vec![DataType::F64]);
        assert!(matches!(
            source_builder.region_ref(RegionId::new(7)),
            Err(ProgramError::MalformedProgram(message)) if message == "region ^7 is not part of this builder",
        ));
        let input = source_builder.add_input(DataType::F64);
        let first = source_builder
            .add_instruction(TestRegionOperation::WithRegions(&["body"]), vec![sealed], vec![input])
            .unwrap()[0];
        let second = source_builder
            .add_instruction(TestRegionOperation::WithRegions(&["body"]), vec![sealed], vec![first])
            .unwrap()[0];
        let source = source_builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![second], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut destination = ProgramBuilder::<Scalar, TestRegionOperation>::new();
        let imported = destination.import_region(source.entry_region_ref());
        let imported_region = destination.region_ref(imported).unwrap().region();
        assert_eq!(imported_region.instructions()[0].regions(), imported_region.instructions()[1].regions());
        assert_ne!(imported_region.instructions()[0].regions()[0], imported);
    }

    #[test]
    fn test_program_builder_import_regions_preserves_sharing() {
        let mut leaf_builder = ProgramBuilder::<Scalar, TestRegionOperation>::new();
        let leaf_input = leaf_builder.add_input(DataType::F64);
        let leaf = leaf_builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![leaf_input], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut root_builder = ProgramBuilder::<Scalar, TestRegionOperation>::new();
        let nested = root_builder.import_region(leaf.entry_region_ref());
        let root_input = root_builder.add_input(DataType::F64);
        let root_output = root_builder
            .add_instruction(TestRegionOperation::WithRegions(&["body"]), vec![nested], vec![root_input])
            .unwrap()[0];
        let root = root_builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![root_output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        // Construct two distinct roots in one source arena that both reference the same previously sealed leaf.
        let mut source_builder = ProgramBuilder::<Scalar, TestRegionOperation>::new();
        let shared_leaf = source_builder.import_region(leaf.entry_region_ref());
        let first_root = RegionId::new(source_builder.regions.len());
        source_builder.regions.push(root.entry_region().clone());
        let second_root = RegionId::new(source_builder.regions.len());
        source_builder.regions.push(root.entry_region().clone());
        let source_input = source_builder.add_input(DataType::F64);
        let source_output = source_builder
            .add_instruction(
                TestRegionOperation::WithRegions(&["first", "second"]),
                vec![first_root, second_root],
                vec![source_input],
            )
            .unwrap()[0];
        let source = source_builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![source_output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(source.region(first_root).unwrap().instructions()[0].regions(), &[shared_leaf]);
        assert_eq!(source.region(second_root).unwrap().instructions()[0].regions(), &[shared_leaf]);

        let roots = [source.region_ref(first_root).unwrap(), source.region_ref(second_root).unwrap()];
        let mut destination = ProgramBuilder::<Scalar, TestRegionOperation>::new();
        let imported = destination.import_regions(&roots).unwrap();
        assert_ne!(imported[0], imported[1]);
        assert_eq!(destination.regions.len(), 3);
        assert_eq!(
            destination.regions[imported[0].index()].instructions()[0].regions(),
            destination.regions[imported[1].index()].instructions()[0].regions(),
        );

        let mut duplicate_destination = ProgramBuilder::<Scalar, TestRegionOperation>::new();
        let duplicate_roots = [source.region_ref(first_root).unwrap(), source.region_ref(first_root).unwrap()];
        let imported = duplicate_destination.import_regions(&duplicate_roots).unwrap();
        assert_eq!(imported[0], imported[1]);
        assert_eq!(duplicate_destination.regions.len(), 2);

        let mut other_builder = ProgramBuilder::<Scalar, TestRegionOperation>::new();
        let other_input = other_builder.add_input(DataType::F64);
        let other = other_builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![other_input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut mixed_destination = ProgramBuilder::<Scalar, TestRegionOperation>::new();
        assert!(matches!(
            mixed_destination.import_regions(&[source.entry_region_ref(), other.entry_region_ref()]),
            Err(ProgramError::MalformedProgram(message))
                if message == "one region import batch cannot combine roots from different source arenas",
        ));
        assert!(mixed_destination.regions.is_empty());
    }

    #[test]
    fn test_program_builder_build_shares_region_across_instructions() {
        // Sharing is legal: several instructions (and several slots of one instruction) may reference one region.
        let mut builder = ProgramBuilder::<Scalar, TestRegionOperation>::new();
        let mut region_builder = ProgramBuilder::<Scalar, TestRegionOperation>::new();
        let region_input = region_builder.add_input(DataType::F64);
        let region_program = region_builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![region_input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let sealed = builder.import_region(region_program.entry_region_ref());
        let input = builder.add_input(DataType::F64);
        let first = builder
            .add_instruction(TestRegionOperation::WithRegions(&["first", "second"]), vec![sealed, sealed], vec![input])
            .unwrap()[0];
        let second = builder
            .add_instruction(TestRegionOperation::WithRegions(&["body"]), vec![sealed], vec![first])
            .unwrap()[0];
        let program = builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![second], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(program.regions().len(), 2);
        assert_eq!(program.instructions()[0].regions(), &[sealed, sealed]);
        assert_eq!(program.instructions()[1].regions(), &[sealed]);
    }

    #[test]
    fn test_program_builder_add_instruction_derives_region_interfaces() {
        // The region-carrying operation's output types are its first region interface's output types, so an entry
        // input type that differs from the region output type pins that the builder derived and delivered the
        // interface (rather than the inference falling back to the operation inputs).
        let mut builder = ProgramBuilder::<Scalar, TestRegionOperation>::new();
        let mut region_builder = ProgramBuilder::<Scalar, TestRegionOperation>::new();
        let region_input = region_builder.add_input(DataType::I64);
        let region_program = region_builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![region_input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let sealed = builder.import_region(region_program.entry_region_ref());
        let input = builder.add_input(DataType::F64);
        let output = builder
            .add_instruction(TestRegionOperation::WithRegions(&["body"]), vec![sealed], vec![input])
            .unwrap()[0];
        assert_eq!(builder.atoms()[output.index()].r#type().into_owned(), DataType::I64);
    }

    #[test]
    fn test_program_render_multi_region() {
        // A shared region renders its body once (labeled with its identifier, at its first reference) and later
        // references render as that identifier alone, while a singly referenced region renders nested inline.
        // Regions are labeled with the operation-declared names.
        let mut builder = ProgramBuilder::<Scalar, TestRegionOperation>::new();
        let mut region_builder = ProgramBuilder::<Scalar, TestRegionOperation>::new();
        let region_input = region_builder.add_input(DataType::F64);
        let region_program = region_builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![region_input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let shared = builder.import_region(region_program.entry_region_ref());
        let inline = builder.import_region(region_program.entry_region_ref());
        let input = builder.add_input(DataType::F64);
        let first = builder
            .add_instruction(TestRegionOperation::WithRegions(&["first", "second"]), vec![shared, shared], vec![input])
            .unwrap()[0];
        let second = builder
            .add_instruction(TestRegionOperation::WithRegions(&["body"]), vec![inline], vec![first])
            .unwrap()[0];
        let program = builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![second], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = with_regions %0 [
                    first=^0={
                        lambda %0:f64 .
                        in (%0)
                    },
                    second=^0,
                ]
                    %2:f64 = with_regions %1 [
                        body={
                            lambda %0:f64 .
                            in (%0)
                        },
                    ]
                in (%2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_program_instruction_effects_include_attached_regions() {
        // An instruction whose operation is pure but whose attached region contains an effectful instruction reports
        // impure effects, while a sibling pure instruction stays pure.
        let mut builder = ProgramBuilder::<Scalar, TestRegionOperation>::new();
        let mut region_builder = ProgramBuilder::<Scalar, TestRegionOperation>::new();
        let region_input = region_builder.add_input(DataType::F64);
        let region_output = region_builder
            .add_instruction(TestRegionOperation::Effectful, Vec::new(), vec![region_input])
            .unwrap()[0];
        let region_program = region_builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![region_output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let sealed = builder.import_region(region_program.entry_region_ref());
        let input = builder.add_input(DataType::F64);
        let with_regions = builder
            .add_instruction(TestRegionOperation::WithRegions(&["body"]), vec![sealed], vec![input])
            .unwrap()[0];
        let output = builder.add_instruction(TestRegionOperation::Add, Vec::new(), vec![input, with_regions]).unwrap();
        let output = output[0];
        let program = builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let entry = program.entry();
        assert_eq!(
            program.instruction_effects(InstructionId::new(entry, 0)).unwrap(),
            Effects::single(Effect::OrderedIo),
        );
        assert_eq!(program.instruction_effects(InstructionId::new(entry, 1)).unwrap(), Effects::PURE);
        assert_eq!(program.effects(), Effects::single(Effect::OrderedIo));
    }

    #[test]
    fn test_program_simplified_multi_region() {
        // We use two sealed regions: a pure one (^0) referenced only by a dead instruction, and an effectful one (^1)
        // referenced by another dead instruction. Simplification drops the pure dead instruction together with its
        // region, keeps the effectful dead instruction alive (its attached region's effects are observable), and
        // compacts the surviving region identifiers (the effectful region moves from ^1 to ^0).
        let mut builder = ProgramBuilder::<Scalar, TestRegionOperation>::new();
        let mut pure_builder = ProgramBuilder::<Scalar, TestRegionOperation>::new();
        let pure_input = pure_builder.add_input(DataType::F64);
        let pure_program = pure_builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![pure_input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let pure_region = builder.import_region(pure_program.entry_region_ref());
        let mut effectful_builder = ProgramBuilder::<Scalar, TestRegionOperation>::new();
        let effectful_input = effectful_builder.add_input(DataType::F64);
        let effectful_output = effectful_builder
            .add_instruction(TestRegionOperation::Effectful, Vec::new(), vec![effectful_input])
            .unwrap()[0];
        let effectful_program = effectful_builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![effectful_output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let effectful_region = builder.import_region(effectful_program.entry_region_ref());
        let input = builder.add_input(DataType::F64);
        builder
            .add_instruction(TestRegionOperation::WithRegions(&["body"]), vec![pure_region], vec![input])
            .unwrap();
        builder
            .add_instruction(TestRegionOperation::WithRegions(&["body"]), vec![effectful_region], vec![input])
            .unwrap();
        let output = builder.add_instruction(TestRegionOperation::Add, Vec::new(), vec![input, input]).unwrap()[0];
        let program = builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(program.regions().len(), 3);
        let simplified = program.simplified().unwrap();
        assert_eq!(simplified.regions().len(), 2);
        assert_eq!(simplified.instructions().len(), 2);
        assert_eq!(simplified.instructions()[0].operation(), &TestRegionOperation::WithRegions(&["body"]));
        assert_eq!(simplified.instructions()[0].regions(), &[RegionId::new(0)]);
        assert_eq!(simplified.instructions()[1].operation(), &TestRegionOperation::Add);
        assert!(!simplified.region(RegionId::new(0)).unwrap().instructions()[0].operation().effects().is_pure());
        let simplified = program.into_simplified().unwrap();
        assert_eq!(simplified.regions().len(), 2);
        assert_eq!(simplified.instructions().len(), 2);
        assert_eq!(simplified.instructions()[0].regions(), &[RegionId::new(0)]);
    }

    #[test]
    fn test_program_builder_splice_program_preserves_region_sharing() {
        let mut leaf_builder = ProgramBuilder::<Scalar, TestRegionOperation>::new();
        let leaf_input = leaf_builder.add_input(DataType::F64);
        let leaf = leaf_builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![leaf_input], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut root_builder = ProgramBuilder::<Scalar, TestRegionOperation>::new();
        let nested = root_builder.import_region(leaf.entry_region_ref());
        let root_input = root_builder.add_input(DataType::F64);
        let root_output = root_builder
            .add_instruction(TestRegionOperation::WithRegions(&["body"]), vec![nested], vec![root_input])
            .unwrap()[0];
        let root = root_builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![root_output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        // Two distinct attached roots share one nested leaf, and both entry instructions reuse those same roots.
        // Splicing must preserve both levels of sharing through one source-to-destination remapping.
        let mut source_builder = ProgramBuilder::<Scalar, TestRegionOperation>::new();
        let shared_leaf = source_builder.import_region(leaf.entry_region_ref());
        let first_root = RegionId::new(source_builder.regions.len());
        source_builder.regions.push(root.entry_region().clone());
        let second_root = RegionId::new(source_builder.regions.len());
        source_builder.regions.push(root.entry_region().clone());
        let source_input = source_builder.add_input(DataType::F64);
        let first_output = source_builder
            .add_instruction(
                TestRegionOperation::WithRegions(&["first", "second"]),
                vec![first_root, second_root],
                vec![source_input],
            )
            .unwrap()[0];
        let source_output = source_builder
            .add_instruction(
                TestRegionOperation::WithRegions(&["first", "second"]),
                vec![first_root, second_root],
                vec![first_output],
            )
            .unwrap()[0];
        let source = source_builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![source_output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(source.region(first_root).unwrap().instructions()[0].regions(), &[shared_leaf]);
        assert_eq!(source.region(second_root).unwrap().instructions()[0].regions(), &[shared_leaf]);

        let mut destination = ProgramBuilder::<Scalar, TestRegionOperation>::new();
        let destination_input = destination.add_input(DataType::F64);
        let outputs = destination.splice_program(&source.to_flat_program(), &[destination_input]).unwrap();
        let relocated = destination
            .build::<Vec<Scalar>, Vec<Scalar>>(outputs, vec![Placeholder], vec![Placeholder])
            .unwrap();
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
        let mut builder = ProgramBuilder::<Scalar, TestRegionOperation>::new();
        let input = builder.add_input(DataType::F64);
        assert!(matches!(
            builder.add_instruction(TestRegionOperation::WithRegions(&["body"]), vec![RegionId::new(3)], vec![input]),
            Err(ProgramError::MalformedProgram(message))
                if message == "instruction references region ^3 which has not been sealed yet",
        ));
        let output = builder.add_variable(DataType::F64);
        builder.add_instruction_unchecked(Instruction::new(
            TestRegionOperation::WithRegions(&["body"]),
            vec![input],
            vec![output],
            vec![RegionId::new(3)],
        ));
        assert!(matches!(
            builder.build::<Vec<Scalar>, Vec<Scalar>>(vec![output], vec![Placeholder], vec![Placeholder]),
            Err(ProgramError::MalformedProgram(message))
                if message == "instruction references region ^3 which has not been sealed yet",
        ));

        // The attached-region count must match the operation's declared slot count. The checked instruction path
        // rejects at insertion time and the unchecked path at build time.
        let mut builder = ProgramBuilder::<Scalar, TestRegionOperation>::new();
        let mut region_builder = ProgramBuilder::<Scalar, TestRegionOperation>::new();
        let region_input = region_builder.add_input(DataType::F64);
        let region_program = region_builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![region_input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let sealed = builder.import_region(region_program.entry_region_ref());
        let input = builder.add_input(DataType::F64);
        assert!(matches!(
            builder.add_instruction(TestRegionOperation::Add, vec![sealed], vec![input, input]),
            Err(ProgramError::MalformedProgram(message))
                if message == "operation `add` declares 0 region slots but 1 regions were attached",
        ));
        let output = builder.add_variable(DataType::F64);
        builder.add_instruction_unchecked(Instruction::new(
            TestRegionOperation::Add,
            vec![input, input],
            vec![output],
            vec![sealed],
        ));
        assert!(matches!(
            builder.build::<Vec<Scalar>, Vec<Scalar>>(vec![output], vec![Placeholder], vec![Placeholder]),
            Err(ProgramError::MalformedProgram(message))
                if message == "operation `add` declares 0 region slots but 1 regions were attached",
        ));

        // Every sealed region must be reachable from the entry root.
        let mut builder = ProgramBuilder::<Scalar, TestRegionOperation>::new();
        let mut region_builder = ProgramBuilder::<Scalar, TestRegionOperation>::new();
        let region_input = region_builder.add_input(DataType::F64);
        let region_program = region_builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![region_input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        builder.import_region(region_program.entry_region_ref());
        let input = builder.add_input(DataType::F64);
        assert!(matches!(
            builder.build::<Vec<Scalar>, Vec<Scalar>>(vec![input], vec![Placeholder], vec![Placeholder]),
            Err(ProgramError::MalformedProgram(message))
                if message == "region ^0 is not reachable from the program entry region",
        ));
    }
}
