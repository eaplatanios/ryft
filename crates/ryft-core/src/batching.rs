//! Contains machinery for _batching_ (i.e., _vectorizing_) computations.
//!
//! Batching turns a function written for individual array values into one that processes a whole batch. It does not
//! mechanically add an outer dimension to every intermediate value. Instead, each flowing value carries the packed
//! position of its batch axis, and each operation decides how to propagate, align, introduce, or remove that axis.
//! Replicated values carry no batch axis and are broadcast only when an operation needs alignment.
//!
//! ```text
//!  ┌───────────────────────────────────┐
//!  │ Values + Input Axis Specification │
//!  └─────────────────┬─────────────────┘
//!                    │ wrap each leaf in the policy-selected batch carrier
//!                    ▼
//!          ┌───────────────────┐
//!          │ Batching Tracers  │
//!          └─────────┬─────────┘
//!                    │ bind each operation through its batching rule
//!                    ▼
//! ┌─────────────────────────────────────┐
//! │ Values + Output Axis Specification  │
//! └─────────────────────────────────────┘
//! ```
//!
//! # Entry Points
//!
//! Use the free [`batch`] function for ordinary value-level vectorization. It recovers the input values' execution
//! context, wraps leaves according to the requested input axes, runs the closure through a [`BatchingContext`], and
//! returns values aligned to the requested output axes. The [`Batch`] context capability exposes the same transform
//! when generic code already owns a context. For an already traced program, use [`Program::batched`]. Program-level
//! batching rewrites the program while retaining explicit input and output axis metadata, which is useful for
//! compilation and higher-order operation rules.
//!
//! [`BatchAxis`] describes a leaf as either batched at a signed axis or replicated, and a
//! [`BatchAxisSpecification`] supplies the batch-axis extent and optional name. Negative axes are normalized against
//! each array's rank, and every batched input must agree on the batch extent.
//!
//! # Core Abstractions
//!
//! The array specialization [`ArrayBatch`](crate::ArrayBatch) pairs an underlying array value with its batch-axis
//! position. Its alignment helpers can broadcast a replicated value, move an existing axis, or align several operands
//! to a common position. The wrapper's type is the unbatched per-item type (the packed value retains its batch
//! dimension).
//!
//! [`BatchingContext`] wraps a parent [`Context`] and records the active axis extent and optional axis name.
//! [`BatchingTracer`] is the value flowing through a batched closure. It carries the representation selected by a
//! [`BatchingPolicy`] and delegates each bind to the bound operation's [`BatchableOperation`] rule. Because the parent
//! may be eager or staging, the same rule can execute concrete values or build a transformed program.
//!
//! [`ElementwiseOperation`]s infer a common batch size, align operands to a common batch-axis position, bind the underlying
//! operation once, and propagate that axis to every result. Shape-changing, reducing, control-flow, and higher-order
//! operations provide explicit rules because their output axes cannot be inferred from elementwise semantics.
//!
//! # Nested Programs
//!
//! Nested flat programs batch structurally through [`Program::batched`], requested by rules through their active
//! [`BatchingDriver`]. [`ProgramBatchingOutputAxesPolicy`] controls whether nested results keep their inferred axes or
//! are forced to explicit positions, and [`InterpretableBatchableOperation`] connects batching rules to eager
//! interpretation when an operation needs both capabilities.
//!
//! # Extending Batching
//!
//! Implement [`BatchableOperation`] for each primitive operation payload that can appear under batching. A rule
//! receives batched inputs and the active batching context and returns batched outputs. Prefer the shared elementwise
//! implementation for genuinely elementwise operations, and write a dedicated rule when dimensions are reordered,
//! reduced, created, indexed, or controlled by nested programs.
//!
//! For a region-carrying operation, keep the recursive batching logic with that operation and request nested work
//! through its active [`BatchingDriver`] (per-item replay via [`BatchingDriver::batch_region`] or structural program
//! batching via [`BatchingDriver::batch_program`]). Preserve named-axis and sharding semantics when moving or
//! introducing the batch axis, and return [`BatchingError`] for invalid axis contracts rather than panicking.

use std::borrow::Cow;
use std::fmt::{Debug, Display};
use std::rc::Rc;

use thiserror::Error;

use ryft_macros::Parameter;

use crate::arrays::{ArrayType, DimensionType, ShardingDimension};
use crate::axes::{Axis, AxisError};
use crate::contexts::{Context, Domain, ProjectedContext, StagingContext, ValueResolution};
use crate::interpretation::InterpretableOperation;
use crate::macros::check_count;
use crate::parameters::{Parameter, ParameterError, Parameterized, ParameterizedFamily, Placeholder};
use crate::programs::{
    BindingRegionDriver, EmptyRegionDriver, Operation, OperationProjection, Program, ProgramError, RegionDriver,
    RegionRef, Type, TypeError, Typed, Value, ValueProjection,
};
use crate::tracing::{Tracer, TracingContext};

/// Represents batching-related errors.
///
/// [`BatchingError`] and [`ProgramError`] deliberately form a conversion cycle in which each type can
/// carry the other. Batching rules get executed by binding operations (i.e., via [`Context::bind`] and
/// [`StagingContext::stage_operation`]), which can result in [`ProgramError`]s. So, [`BatchingError`]s travel
/// up a trace, type-erased, inside [`ProgramError::Custom`] payloads. In the other direction, the public batching
/// transform entry point is typed to [`BatchingError`], and a batching trace can also fail for reasons that are not
/// batching-related. Those program errors surface through the [`BatchingError::Program`] variant. The paired [`From`]
/// implementations keep this cycle normalized instead of letting the two types nest: converting to [`ProgramError`]
/// unwraps a [`BatchingError::Program`] back into the program error that it carries and wraps every other variant in
/// [`ProgramError::Custom`], while converting to [`BatchingError`] unwraps a [`ProgramError::Custom`] payload holding
/// a [`BatchingError`] and wraps every other program error in [`BatchingError::Program`]. Round trips therefore never
/// nest one error type inside the other, and `?` re-types errors correctly at both boundaries. Outside of these
/// conversions, a [`BatchingError`] carried by a [`ProgramError`] can be recovered using
/// [`ProgramError::downcast_custom`].
#[derive(Error, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BatchingError {
    #[error("encountered an empty batch")]
    EmptyBatch,

    #[error("mismatched batch sizes across batched leaves; expected size {expected} but got {actual}")]
    MismatchedBatchSizes { expected: usize, actual: usize },

    #[error("{message}")]
    MisalignedBatchAxes { message: String },

    #[error("batch axis {axis} of array type {type} has dynamic size")]
    DynamicBatchAxis { r#type: Box<ArrayType>, axis: Axis },

    #[error("batch axis {axis} is out of bounds for array type {type}")]
    BatchAxisOutOfBounds { r#type: Box<ArrayType>, axis: Axis },

    #[error("dimension type {type} cannot carry mapped batch axis {axis}")]
    MappedDimension { r#type: Box<DimensionType>, axis: BatchAxis },

    #[error("{message}")]
    InvalidBatchMetadata { message: String },

    #[error("{message}")]
    UnsupportedOperation { message: String },

    #[error("mismatched batch output axes; expected {expected} but got {actual}")]
    MismatchedOutputAxes { expected: BatchAxis, actual: BatchAxis },

    #[error(transparent)]
    Parameter(#[from] ParameterError),

    #[error(transparent)]
    Type(#[from] TypeError),

    #[error(transparent)]
    Axis(#[from] AxisError),

    #[error(transparent)]
    Program(ProgramError),
}

impl From<ProgramError> for BatchingError {
    #[inline]
    fn from(error: ProgramError) -> Self {
        if let Some(batching) = error.downcast_custom::<BatchingError>() {
            batching.clone()
        } else {
            BatchingError::Program(error)
        }
    }
}

impl From<BatchingError> for ProgramError {
    #[inline]
    fn from(error: BatchingError) -> Self {
        match error {
            BatchingError::Program(error) => error,
            error => ProgramError::custom(error),
        }
    }
}

/// A batched value's mapped batch axis. [`BatchAxis::new`]`(k)` means that the value's batch dimension sits at packed
/// axis `k`. [`BatchAxis::replicated`] (the [`Default`]) means that the value is *replicated* (i.e., it carries no
/// dimension for the batch and is interpreted as the same value for every batch item). For example, a traced constant
/// in `batch(|x| x + 1)` is replicated, while `x` carries the mapped input axis. Runtime control flow predicates may
/// also be replicated, because a single predicate may select one branch for the whole batch while a batch-varying
/// predicate would need a dedicated batching rule. Note that replication is not limited to rank-0 (i.e., scalar)
/// values. Any shaped constant or input is replicated when none of its dimensions indexes the batch.
///
/// This is the batch axis carried by an [`ArrayBatch`](crate::ArrayBatch) and, during the batching transform, by the
/// [`Tracer`] metadata. Carrying it on the value itself lets the per-operation batching rules route the mapped batch
/// axis straight from the value in hand.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash, Parameter)]
pub struct BatchAxis(Option<Axis>);

impl BatchAxis {
    /// Creates a mapped [`BatchAxis`] at position `axis`.
    #[inline]
    pub fn new<A: Into<Axis>>(axis: A) -> Self {
        Self(Some(axis.into()))
    }

    /// Creates a mapped [`BatchAxis`] from an already-normalized position.
    #[inline]
    pub fn from_position(position: usize) -> Self {
        Self::new(position)
    }

    /// Creates a replicated or mapped [`BatchAxis`] from an already-normalized optional position.
    #[inline]
    pub fn from_optional_position(position: Option<usize>) -> Self {
        position.map(Self::from_position).unwrap_or_default()
    }

    /// Creates a replicated [`BatchAxis`] (i.e., the batched value is shared unchanged across every batch item).
    /// This is equivalent to [`BatchAxis::default`].
    #[inline]
    pub fn replicated() -> Self {
        Self(None)
    }

    /// Returns the mapped batch axis position, or `None` if this [`BatchAxis`] is replicated.
    #[inline]
    pub fn axis(&self) -> Option<Axis> {
        self.0
    }

    /// Returns `true` if this [`BatchAxis`] is replicated (i.e., if it carries no mapped batch axis).
    #[inline]
    pub fn is_replicated(&self) -> bool {
        self.0.is_none()
    }
}

impl Display for BatchAxis {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            Some(axis) => write!(formatter, "axis {axis}"),
            None => write!(formatter, "replicated"),
        }
    }
}

impl From<Option<isize>> for BatchAxis {
    #[inline]
    fn from(axis: Option<isize>) -> Self {
        Self(axis.map(Axis::from))
    }
}

impl<A: Into<Axis>> From<A> for BatchAxis {
    #[inline]
    fn from(axis: A) -> Self {
        Self(Some(axis.into()))
    }
}

/// Specification of a batch axis introduced by the batching transform that contains an optional explicit extent and
/// an optional axis name that can be referenced by operations that support named axes. The extent is normally
/// inferred from the inputs that are being batched. An explicit extent can be provided to either pin it or to drive
/// a broadcasted batching transform whose extent would otherwise be unobservable. Its representation is selected by
/// the active [`BatchingPolicy`] (e.g., homogeneous arrays use a host `usize`, while array IR programs use a
/// first-class dimension value). The axis name makes the batch axis addressable by name from collective operations
/// inside the batched function body. The default `usize` form converts from `None`, `Some(extent)`, and `extent`
/// directly. For example:
///
/// ```ignore
/// domain.batch(f, input, input_axes, output_axes, None)?;                                     // Inferred, anonymous.
/// domain.batch(f, input, input_axes, output_axes, 8)?;                                        // Explicit, anonymous.
/// domain.batch(f, input, input_axes, output_axes, BatchAxisSpecification::named("devices"))?; // Inferred, named.
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchAxisSpecification<E = usize> {
    /// Explicit batch extent, or `None` to infer it from the inputs that are being batched.
    extent: Option<E>,

    /// Name that operations (e.g., collectives) can use to refer to it, or `None` for an anonymous axis.
    name: Option<String>,
}

impl<E> Default for BatchAxisSpecification<E> {
    #[inline]
    fn default() -> Self {
        Self { extent: None, name: None }
    }
}

impl<E> BatchAxisSpecification<E> {
    /// Creates a named [`BatchAxisSpecification`] with an explicit batch extent.
    #[inline]
    pub fn new<N: Into<String>>(extent: E, name: N) -> Self {
        Self { extent: Some(extent), name: Some(name.into()) }
    }

    /// Creates an anonymous [`BatchAxisSpecification`] with an explicit batch extent.
    #[inline]
    pub fn with_extent(extent: E) -> Self {
        Self { extent: Some(extent), name: None }
    }

    /// Creates a named [`BatchAxisSpecification`] whose batch extent is inferred from the mapped inputs.
    #[inline]
    pub fn named<N: Into<String>>(name: N) -> Self {
        Self { extent: None, name: Some(name.into()) }
    }

    /// Returns the explicit batch extent, or `None` when it is to be inferred from mapped inputs.
    #[inline]
    pub fn extent(&self) -> Option<&E> {
        self.extent.as_ref()
    }

    /// Returns the name of this [`BatchAxisSpecification`] that operations (e.g., collectives) can use to refer
    /// to it, or `None` for an anonymous axis.
    #[inline]
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
}

impl<E> From<Option<E>> for BatchAxisSpecification<E> {
    #[inline]
    fn from(extent: Option<E>) -> Self {
        Self { extent, name: None }
    }
}

impl From<usize> for BatchAxisSpecification {
    #[inline]
    fn from(extent: usize) -> Self {
        Self::with_extent(extent)
    }
}

/// Result produced by batching a nested [`Program`]. Every [`BatchingPolicy`] selects one implementation through
/// [`BatchingPolicy::BatchedProgram`]. Implementations retain their complete policy-specific program boundary while
/// exposing the two pieces of information needed by universe-neutral batching machinery: one [`BatchAxis`] per semantic
/// source-program output and the transformed program itself. Any bookkeeping inputs or outputs added by a policy are
/// retained in the returned program and excluded from [`Self::output_axes`]. Concrete implementations remain
/// responsible for validating their stronger boundary invariants.
pub trait BatchedProgram<V: Value, O: Operation<Type = V::Type>> {
    /// Returns the mapped axes of the source [`Region`](crate::Region)'s semantic outputs.
    fn output_axes(&self) -> &[BatchAxis];

    /// Consumes this [`BatchedProgram`] and returns its underlying transformed [`Program`] and semantic output axes
    /// without altering its [`BatchingPolicy`]-specific boundary.
    fn into_parts(self) -> (Program<V, O, Vec<V>, Vec<V>>, Vec<BatchAxis>);
}

/// [`BatchedProgram`] whose underlying transformed [`Program`] carries exactly the source [`Region`](crate::Region)'s
/// inputs and outputs, without any [`BatchingPolicy`]-owned bookkeeping boundary values. This is the reusable default
/// carrier for batching policies that preserve region boundaries. Policies that widen transformed boundaries select
/// their own [`BatchedProgram`] implementation so consumers cannot silently mistake a bookkeeping input or output for
/// one belonging to the source region.
pub struct BoundaryPreservingBatchedProgram<V: Typed + Parameter, O> {
    /// Structurally transformed [`Program`].
    program: Program<V, O, Vec<V>, Vec<V>>,

    /// Mapped axes of the source [`Region`](crate::Region)'s outputs.
    output_axes: Vec<BatchAxis>,
}

impl<V: Value, O: Operation<Type = V::Type>> BoundaryPreservingBatchedProgram<V, O> {
    /// Creates a new [`BoundaryPreservingBatchedProgram`] that carries exactly the source
    /// [`Region`](crate::Region)'s inputs and outputs.
    #[inline]
    pub fn new(program: Program<V, O, Vec<V>, Vec<V>>, output_axes: Vec<BatchAxis>) -> Result<Self, ProgramError> {
        check_count!("output", output_axes, program.output_count(), ProgramError);
        Ok(Self { program, output_axes })
    }

    /// Creates a new [`BoundaryPreservingBatchedProgram`] from the parts of a structurally batched program
    /// whose boundary may still carry a [`BatchingPolicy`]'s widening.
    ///
    /// This constructor performs the two boundary adjustments that turn a widened batched program back into an
    /// ordinary [`Region`](crate::Region) program, both by replaying the program and rebuilding its output
    /// boundary:
    ///
    ///   - It drops the leading `widening_output_count` outputs, which carry transform bookkeeping (e.g., the
    ///     composite policy's forwarded mapped extent) rather than source-region results.
    ///   - It reconciles each remaining output with the batch axis its consumer requires: an output that is mapped
    ///     while its required axis is replicated is collapsed along its mapped axis by `collapse_fn`. Every other
    ///     combination passes through unchanged.
    ///
    /// Unlike [`Self::new`], the provided parts deliberately do *not* need to satisfy this type's boundary
    /// invariant: `output_axes` excludes the bookkeeping outputs that `program` still carries, so the parts only
    /// become a valid [`BoundaryPreservingBatchedProgram`] once the replay has dropped those outputs. The constructed
    /// result always satisfies the invariant, and its output axes report each collapsed output as replicated
    /// (collapsing is what made it so). When neither adjustment applies (no bookkeeping outputs and no required axes),
    /// the parts are rewrapped directly, without a replay.
    ///
    /// This is the shared implementation behind every [`BatchingPolicy::adapt_batched_program`] implementation:
    /// each policy calls it with its own widening output count so that the replay machinery exists exactly once.
    ///
    /// # Parameters
    ///
    ///   - `program`: Structurally batched program, which may still carry `widening_output_count` leading
    ///     bookkeeping outputs beyond the source region's own outputs.
    ///   - `output_axes`: Mapped batch axis of each source-region output of `program`, in order, *excluding* the
    ///     bookkeeping outputs (i.e., one entry per output that the constructed result keeps).
    ///   - `required_output_axes`: Batch axis that the consumer requires for each kept output, or [`None`] to
    ///     keep every output's natural axis. Only mapped-to-replicated mismatches are resolved (via `collapse_fn`),
    ///     because they are the one mismatch no structural axis movement can fix; batch the program with
    ///     [`ProgramBatchingOutputAxesPolicy::AlignEachTo`] so that every structurally movable axis already
    ///     matches its requirement.
    ///   - `widening_output_count`: Number of leading `program` outputs that belong to the policy's boundary
    ///     widening and must be dropped.
    ///   - `collapse_fn`: Collapses one mapped output along the provided axis, within the replay's own
    ///     [`TracingContext`]. A collapsed output arrives as the per-item family `{y₀, …, yₙ₋₁}` packed along the
    ///     mapped axis while its consumer requires the one value that is correct in the replicated position, so this
    ///     function is the consumer's chosen left inverse to replication along the batch axis. Which inverse is correct
    ///     depends on what the output *means*, which is why callers own this step. For example, the
    ///     [`LinearCallOperation`](crate::LinearCallOperation) transpose consumer collapses cotangents by summation
    ///     (i.e., `ȳ = Σᵢ ȳᵢ`, because replicating a primal across the batch is a broadcast and the transpose of a
    ///     broadcast is a summation), while a hypothetical consumer of replicated primal values would instead select
    ///     any one item.
    pub fn from_widened_boundary<
        CollapseFn: Fn(
            &TracingContext<V, O>,
            Tracer<TracingContext<V, O>>,
            Axis,
        ) -> Result<Tracer<TracingContext<V, O>>, BatchingError>,
    >(
        program: Program<V, O, Vec<V>, Vec<V>>,
        output_axes: Vec<BatchAxis>,
        required_output_axes: Option<&[BatchAxis]>,
        widening_output_count: usize,
        collapse_fn: CollapseFn,
    ) -> Result<Self, BatchingError> {
        if let Some(required_output_axes) = required_output_axes {
            check_count!("output", output_axes, required_output_axes.len(), ProgramError);
        } else if widening_output_count == 0 {
            return Ok(Self::new(program, output_axes)?);
        }

        // Replay the program over fresh tracer inputs so that its output boundary can be rebuilt. The block scopes the
        // tracing context and every traced value because recovering the rebuilt program below requires unique ownership
        // of the shared builder, and so all other handles to it must have been released by then.
        let (builder, output_ids) = {
            let context = TracingContext::<V, O>::new();
            let builder = context.builder().clone();
            let inputs = program.input_types().into_iter().map(|r#type| context.input(r#type)).collect::<Vec<_>>();
            let mut outputs = program.interpret_in_context(&context, inputs)?;
            check_count!("output", outputs, output_axes.len() + widening_output_count, ProgramError);

            // The leading widening outputs carry transform bookkeeping rather than source-region results, so the
            // rebuilt boundary omits them.
            outputs.drain(..widening_output_count);

            // Collapse each output that is mapped while its consumer requires it replicated. This is the one
            // mismatch no structural axis movement can resolve, so it must be reduced away here.
            if let Some(required_output_axes) = required_output_axes {
                outputs = outputs
                    .into_iter()
                    .zip(output_axes.iter().zip(required_output_axes))
                    .map(|(output, (actual, required))| match (actual.axis(), required.axis()) {
                        (Some(axis), None) => collapse_fn(&context, output, axis),
                        _ => Ok(output),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
            }

            (builder, outputs.iter().map(Tracer::atom_id).collect::<Result<Vec<_>, _>>()?)
        };

        // With the context and tracers gone, the builder handle is unique and the traced instructions can be sealed
        // into the rebuilt program behind the adjusted output boundary.
        let input_count = program.input_count();
        let output_count = output_ids.len();
        let builder = Rc::try_unwrap(builder).map_err(|_| ProgramError::EscapedProgramBuilder)?.into_inner();
        let program = builder.build(output_ids, vec![Placeholder; input_count], vec![Placeholder; output_count])?;

        // A collapsed output is replicated in the rebuilt program, so the reported axes must say so
        // instead of repeating the pre-collapse mapped axis.
        let output_axes = match required_output_axes {
            Some(required_output_axes) => output_axes
                .into_iter()
                .zip(required_output_axes)
                .map(|(actual, required)| match (actual.axis(), required.axis()) {
                    (Some(_), None) => BatchAxis::replicated(),
                    _ => actual,
                })
                .collect(),
            None => output_axes,
        };

        Ok(Self::new(program, output_axes)?)
    }
}

impl<V: Value, O: Operation<Type = V::Type>> BatchedProgram<V, O> for BoundaryPreservingBatchedProgram<V, O> {
    #[inline]
    fn output_axes(&self) -> &[BatchAxis] {
        self.output_axes.as_slice()
    }

    #[inline]
    fn into_parts(self) -> (Program<V, O, Vec<V>, Vec<V>>, Vec<BatchAxis>) {
        (self.program, self.output_axes)
    }
}

/// [`Type`] capability selecting the canonical [`BatchingEntrypointPolicy`] used by the public batching transform.
/// This is a batching-owned extension of [`Type`] and not part of the core type contract. It lets one blanket [`Batch`]
/// implementation select the canonical policy for each program universe without adding batching machinery to [`Type`]
/// or [`Context`] and without relying on overlapping blanket implementations distinguished only by [`Domain::Type`].
pub trait BatchableType: Type {
    /// Canonical [`BatchingEntrypointPolicy`] selecting this type universe's batch carrier, extent representation,
    /// and public entrypoint behavior.
    type Policy: Copy + Clone + Debug;
}

/// Transform-owned policy selecting the batch carrier and mapped-axis extent representation used by a
/// [`BatchingContext`]. One generic batching frame (i.e., [`BatchingContext`], [`BatchingTracer`], [`BatchingDriver`],
/// and the [`BatchableOperation`] rule contract) serves every [`Program`] universe, and this policy names the three
/// places where those universes genuinely differ:
///
///   - **The Batch Carrier (i.e., [`Self::Batch`]):** A batched value maintains a split between its unbatched per-item
///     type and the parent-owned packed value that stores the whole batch. How that split is represented is specific
///     to each value kind: an ordinary array gains a mapped axis in its packed shape (i.e.,
///     [`ArrayBatch`](crate::ArrayBatch)), while a first-class dimension has no mapped representation at all (as a
///     per-item extent would be a ragged shape), and so a composite carrier must keep dimension members replicated and
///     reject mapped non-array members.
///   - **The Mapped-Axis Extent (i.e., [`Self::Extent`]):** Homogeneous array batching uses a static `usize`, while a
///     composite universe may carry an ordinary parent-owned first-class dimension value so that a dynamic batch extent
///     remains a Single Static Assignment (SSA) value flowing through operand edges rather than being treated as static
///     transform metadata.
///   - **The Structurally Batched Program (i.e., [`Self::BatchedProgram`]):** Programs over homogeneous arrays preserve
///     source region's boundary exactly and produce [`BoundaryPreservingBatchedProgram`]s, while a composite universe
///     may thread bookkeeping values such as its first-class mapped extent through standalone nested programs. Every
///     selected carrier implements [`BatchedProgram`], while [`Self::boundary_operands`] and
///     [`Self::adapt_batched_program`] let consumers complete or shed policy-specific widening.
///
/// The policy is deliberately limited to carrier selection, construction, access, and invariant enforcement.
/// Array-specific alignment and broadcasting are represented as functions on [`ArrayBatch`](crate::ArrayBatch) (a
/// composite policy may project an array member into that carrier to reuse an existing array rule). Batching nested
/// [`Region`](crate::Region) support is provided separately by the [`RecursiveBatchingPolicy`] trait, allowing a
/// batching carrier to support region-free operations without also supporting recursive transformation of nested
/// programs.
///
/// Carrier metadata that is not represented by [`Type`] is guarded at the three boundaries where it could otherwise
/// escape the policy's sight. Every rule application is checked by [`Self::validate_operation_outputs`] against the
/// rule's [`Self::Evidence`], member projection transports it through [`BatchingPolicyProjection::project_batch`] and
/// [`BatchingPolicyProjection::lift_batch`], and an opaque batched-region rebuild recovers it through
/// [`RecursiveBatchingPolicy::restore_batch`].
pub trait BatchingPolicy<C: Context>: Copy + Clone + Debug {
    /// Batch-carrying representation for values owned by `C`.
    type Batch: Clone + Debug + Display + Parameter;

    /// Representation of the mapped-axis extent.
    type Extent: Clone + Debug;

    /// Operation-local validation evidence produced by one [`BatchableOperation`] rule application and consumed by
    /// [`Self::validate_operation_outputs`]. Evidence records what a rule did that its outputs alone cannot show, such
    /// as that the rule deliberately consumed carrier metadata instead of dropping it. It is created by the rule, read
    /// once at the validation boundary, and dropped there, so it must never outlive that rule invocation. Policies
    /// whose carriers hold no metadata beyond [`Type`] use `()`. The [`Default`] bound is what lets the overwhelming
    /// majority of rules return a plain [`Vec`] of carriers converted through [`BatchedOutputs`]'s [`From`]
    /// implementation, without naming evidence they do not produce.
    type Evidence: Default;

    /// Result of structurally batching a nested [`Program`], including any policy-owned bookkeeping widening of the
    /// program boundary. Refer to the documentation of [`Self::adapt_batched_program`] for how consumers shed that
    /// widening when they need an ordinary [`Region`](crate::Region) boundary.
    type BatchedProgram: BatchedProgram<C::Constant, C::Operation>;

    /// Wraps a parent-owned packed value with the requested mapped axis, validating and normalizing that axis.
    fn batch(value: C::Value, batch_axis: BatchAxis) -> Result<Self::Batch, BatchingError>;

    /// Wraps a parent-owned packed value as replicated across the batch.
    fn replicated(value: C::Value) -> Self::Batch;

    /// Returns the parent-owned packed value stored in `batch`.
    fn value(batch: &Self::Batch) -> &C::Value;

    /// Returns the [`BatchAxis`] carried by `batch`.
    fn batch_axis(batch: &Self::Batch) -> BatchAxis;

    /// Returns the per-item type exposed by `batch` after removing its mapped batch axis, if any.
    fn unbatched_type(batch: &Self::Batch) -> Cow<'_, C::Type>;

    /// Validates the batch carriers produced by one operation's batching rule against the carriers that rule consumed.
    /// The batching transform calls this hook after every rule, including rules replayed directly inside a nested
    /// region, and before exposing their outputs to subsequent operations. The default accepts every transition.
    /// A policy whose carrier owns semantic metadata that is absent from [`Type`] overrides this method to reject
    /// transitions that would silently drop that metadata, using `evidence` to recognize the rules that consumed it
    /// deliberately.
    ///
    /// This is a carrier-invariant boundary and not an additional batching rule. The outputs are borrowed immutably,
    /// so an implementation can accept or reject a transition but cannot rewrite one. The evidence is owned by the
    /// transform for the duration of this call alone and is dropped immediately afterward, so one rule's evidence can
    /// never be observed by the operations that follow it.
    ///
    /// The division of labor is deliberate. This hook states the policy-owned conservation law once and applies it
    /// uniformly to the open set of [`BatchableOperation`] rules, precisely because the rules cannot police that law
    /// themselves: the defect class is a rule that is oblivious to the carrier metadata, and an oblivious rule does
    /// not know it should check anything. Every fact specific to one operation must therefore arrive as
    /// [`Self::Evidence`] supplied by the rule. If an implementation of this method ever needs to branch on
    /// `operation_name`, that is the signal that the missing knowledge belongs in the evidence type rather than here.
    ///
    /// # Parameters
    ///
    ///   - `operation_name`: Canonical name of the [`Operation`] whose batching rule produced `outputs` that is used
    ///     only for diagnostic purposes (e.g., error reporting).
    ///   - `inputs`: Batch carriers supplied to that rule.
    ///   - `outputs`: Batch carriers produced by that rule.
    ///   - `evidence`: Operation-local [`Self::Evidence`] that the rule returned alongside `outputs`.
    #[inline]
    fn validate_operation_outputs(
        _operation_name: &'static str,
        _inputs: &[Self::Batch],
        _outputs: &[Self::Batch],
        _evidence: &Self::Evidence,
    ) -> Result<(), BatchingError> {
        Ok(())
    }

    /// Returns the parent-owned bookkeeping operand values that this policy's structurally batched programs require
    /// prepended to their source operands when they are rebound as ordinary [`Region`](crate::Region)s of an operation
    /// that does not thread this policy's batching state itself.
    ///
    /// A structurally batched program's boundary is not always the source region's boundary. A batched nested program
    /// is sealed and cannot reference values owned by the parent program, so a policy whose batching state is itself a
    /// parent-owned value must widen the batched boundary with leading inputs that reintroduce that state. This
    /// function supplies the parent-owned value for each such input, in boundary order, so consumers can complete
    /// an adapted program's operands without knowing which policy produced it:
    ///
    ///   - Homogeneous array policies return no values (the default), because their mapped-axis extent
    ///     is static transform metadata and their batched programs carry exactly the source boundary.
    ///   - [`ArrayIrBatching`](crate::ArrayIrBatching) returns its first-class mapped-extent dimension value,
    ///     because every dynamic batch dimension inserted into one of its batched programs references the
    ///     [`DimensionVariable`](crate::DimensionVariable) defined by that program's leading extent input.
    ///
    /// Refer to the documentation of [`Self::adapt_batched_program`] for the output-side counterpart of this contract,
    /// along with a complete boundary example.
    #[inline]
    fn boundary_operands(_axis_extent: &Self::Extent) -> Vec<C::Value> {
        Vec::new()
    }

    /// Adapts one structurally batched program to an ordinary [`Region`](crate::Region) boundary by removing this
    /// policy's bookkeeping outputs and reducing each mapped output whose requested target axis is replicated.
    ///
    /// Structural batching widens some policies' program boundaries with bookkeeping state (refer to the documentation
    /// of [`Self::boundary_operands`] for more information on that bookkeeping state). Extent-threading higher-order
    /// operations such as `condition`, `while`, and `scan` consume that widened boundary as-is, but an operation whose
    /// attached regions are plain programs must adapt it first. For example, batched program with support for dynamic
    /// dimensions and shapes may have the boundary:
    ///
    /// ```text
    /// [extent, inputs...] ↦ [extent, outputs...]
    /// ```
    ///
    /// The two extent slots play different roles, which is why adaptation removes one but not the other. The leading
    /// *input* is load-bearing for the program itself: it defines the [`DimensionVariable`](crate::DimensionVariable)
    /// that every inserted dynamic batch dimension's type references, and a sealed program's types must be grounded by
    /// a value in its own scope, so removing it would leave the program referencing an undefined identity. The leading
    /// *output* carries no information of its own: it merely relays the extent so an enclosing extent-threading
    /// operation can chain it through its own sealed regions (e.g., a batched while body must return the extent it
    /// consumed so the next iteration's boundary can be fed). A consumer that does not thread extents has no use for
    /// the relay, so adapting the program drops the forwarded output, producing `[extent, inputs...] ↦ [outputs...]`,
    /// while [`Self::boundary_operands`] supplies the value for the kept input (e.g., a batched linear call consumes
    /// the extent as one more leading residual). Homogeneous array policies adapt without any boundary change.
    ///
    /// When `required_output_axes` is provided, the adaptation additionally reconciles each remaining output with
    /// the batch axis its consumer requires (i.e., an output that is mapped while its required axis is replicated is
    /// passed to `reduce`, which must collapse it along the mapped axis). This is, for example, how a batched transpose
    /// program returns one shared cotangent for a replicated linear input (i.e., replicating a value across batch items
    /// is semantically a broadcast, the transpose of a broadcast is a summation, and so the per-item cotangents are
    /// summed).
    ///
    /// The result is always a [`BoundaryPreservingBatchedProgram`], so programs that need neither adjustment are
    /// rewrapped unchanged, without a replay.
    ///
    /// # Parameters
    ///
    ///   - `program`: Structurally batched program to adapt.
    ///   - `required_output_axes`: Batch axes that the consumer requires for each non-bookkeeping output, or [`None`]
    ///     when the outputs' natural axes are already correct. Only mapped-to-replicated mismatches are resolved here,
    ///     because no structural alignment can fix them; batch the program with
    ///     [`ProgramBatchingOutputAxesPolicy::AlignEachTo`] so that every structurally
    ///     movable axis already matches its requirement.
    ///   - `collapse_fn`: Function that collapses one mapped output along the provided axis within the replayed
    ///     program's own [`TracingContext`], turning the per-item family `{y₀, …, yₙ₋₁}` packed along that axis into
    ///     the one value that is correct in the replicated position (i.e., the consumer's chosen left inverse to
    ///     replication along the batch axis). Callers own this step because which inverse is correct depends on what
    ///     the output means (e.g., summation for cotangents), and not on boundary bookkeeping. Refer to the
    ///     documentation of [`BoundaryPreservingBatchedProgram::from_widened_boundary`] for more information.
    fn adapt_batched_program<
        CollapseFn: Fn(
            &TracingContext<C::Constant, C::Operation>,
            Tracer<TracingContext<C::Constant, C::Operation>>,
            Axis,
        ) -> Result<Tracer<TracingContext<C::Constant, C::Operation>>, BatchingError>,
    >(
        program: Self::BatchedProgram,
        required_output_axes: Option<&[BatchAxis]>,
        collapse_fn: CollapseFn,
    ) -> Result<BoundaryPreservingBatchedProgram<C::Constant, C::Operation>, BatchingError>;
}

/// Selects the [`BatchingPolicy`] used when an outer policy `Self` projects one member type `T` from composite
/// [`Context`] `C`. A [`BatchingPolicy`] determines one context's batch carrier, mapped-extent representation, and
/// structurally batched program boundary. It cannot by itself determine how each member type of a composite context
/// should represent those concepts. For example, an array member needs an [`ArrayBatch`](crate::ArrayBatch) carrier
/// that may hold a mapped axis, whereas a first-class dimension member must remain replicated because a different
/// dimension per batch item would require a ragged value model. Both projected policies must nevertheless preserve
/// the outer policy's exact mapped-extent and validation-evidence representations. This type-indexed relation records
/// that choice independently for every supported `(C, T)` pair.
///
/// Note that this trait carries no runtime state. Implementing it only establishes the associated
/// [`Projected`](Self::Projected) policy used by [`batch_projected_operation`]. Unsupported member projections simply
/// omit an implementation, while composite backends with several independently batchable member kinds can select a
/// different policy for each kind.
pub trait BatchingPolicyProjection<C: Context, T: Type>: BatchingPolicy<C>
where
    ProjectedContext<C, T>: Context,
{
    /// [`BatchingPolicy`] used while applying the projected member operation's batching rule. Its extent and evidence
    /// representations must be identical to the outer policy's, so projection never specializes or reconstructs the
    /// mapped extent, and a member rule's validation evidence crosses the projection boundary unchanged on its way to
    /// the outer policy's [`BatchingPolicy::validate_operation_outputs`].
    type Projected: BatchingPolicy<ProjectedContext<C, T>, Extent = Self::Extent, Evidence = Self::Evidence>;

    /// Converts one outer carrier into the member policy's carrier, at the boundary where a projected member-family
    /// rule runs. The default preserves exactly the canonical pair, the packed value and the mapped [`BatchAxis`],
    /// which is the complete conversion for a carrier that holds nothing else. A policy whose carrier owns additional
    /// semantic metadata (e.g., bounded ragged axes and the logical unbatched type) must override this method so the
    /// member rule observes that metadata, projecting any values embedded in it (e.g., per-item extent vectors) into
    /// the member universe as well.
    ///
    /// Silently dropping carrier metadata here is equivalent to a rule dropping it, except that it also escapes
    /// detection. The outer policy's [`BatchingPolicy::validate_operation_outputs`] sees only what survives this
    /// conversion, so a narrowed input carrier makes the conservation check vacuous. A projection that cannot represent
    /// some metadata in the member carrier must therefore fail with an exact diagnostic naming what it cannot carry,
    /// rather than narrow the carrier and continue.
    ///
    /// [`Self::lift_batch`] is the inverse boundary crossing.
    #[inline]
    fn project_batch(
        batch: &Self::Batch,
    ) -> Result<<Self::Projected as BatchingPolicy<ProjectedContext<C, T>>>::Batch, BatchingError>
    where
        C::Value: ValueProjection<T, Projected: Value<Type = T>>,
        ProjectedContext<C, T>: Context<Value = <C::Value as ValueProjection<T>>::Projected>,
    {
        Self::Projected::batch(C::Value::into_projected(Self::value(batch).clone())?, Self::batch_axis(batch))
    }

    /// Converts one member-policy carrier back into the outer carrier, the inverse boundary crossing of
    /// [`Self::project_batch`]. _Lift_ names the member-to-composite direction here, matching the member-lifting
    /// [`From`] conversions between the two value families. The default restores the canonical pair, the packed value
    /// and the mapped [`BatchAxis`]. A policy that overrode [`Self::project_batch`] must restore everything that
    /// conversion carried down and map any values embedded in that metadata back into the outer universe.
    ///
    /// Projecting, applying the member rule, and lifting must present the member rule's outputs to the outer policy's
    /// [`BatchingPolicy::validate_operation_outputs`] exactly as a native (i.e., unprojected) rule's outputs would be
    /// presented. That round trip is what lets the conservation law judge member rules and native rules by one
    /// standard instead of exempting whatever crossed a projection.
    #[inline]
    fn lift_batch(
        batch: &<Self::Projected as BatchingPolicy<ProjectedContext<C, T>>>::Batch,
    ) -> Result<Self::Batch, BatchingError>
    where
        C::Value: ValueProjection<T, Projected: Value<Type = T>>,
        ProjectedContext<C, T>: Context<Value = <C::Value as ValueProjection<T>>::Projected>,
    {
        Self::batch(C::Value::from_projected(Self::Projected::value(batch).clone()), Self::Projected::batch_axis(batch))
    }
}

/// Policy capability for recursively applying batching to nested [`Program`] [`Region`](crate::Region)s. This is
/// separate from [`BatchingPolicy`] because a carrier can be useful for region-free batching before its program
/// universe supports structural region rewriting. The active [`BatchingContext`] and [`RecursiveBatchingDriver`]
/// are neutral with respect to the underlying value kind (each policy owns the mechanics required to replay its
/// own [`Operation`] universe).
pub trait RecursiveBatchingPolicy<C: Context>: BatchingPolicy<C> {
    /// Replays `region` through the provided [`BatchingContext`].
    fn batch_region(
        context: &BatchingContext<C, Self>,
        region: RegionRef<'_, C::Constant, C::Operation>,
        inputs: Vec<Self::Batch>,
    ) -> Result<Vec<Self::Batch>, BatchingError>;

    /// Structurally batches `region` and returns the resulting transformed [`BatchedProgram`].
    fn batch_program(
        context: &BatchingContext<C, Self>,
        region: RegionRef<'_, C::Constant, C::Operation>,
        input_axes: &[BatchAxis],
        output_axes_policy: ProgramBatchingOutputAxesPolicy,
    ) -> Result<Self::BatchedProgram, BatchingError>;

    /// Restores the batch carrier for a value returned across an opaque batched-region boundary, which erases carrier
    /// metadata because transform state cannot ride on program values. The carrier is rebuilt from the two sources that
    /// survive the crossing (i.e., the region's declared per-item output type and the original input carriers) and
    /// never from the packed physical representation.
    ///
    /// Ordinary carriers are completely determined by the packed parent-context value and mapped axis, so the default
    /// delegates to [`BatchingPolicy::batch`]. A recursive policy whose carrier owns transform-only metadata that is
    /// not represented by [`Type`] must override this method to rebuild that metadata from the surviving sources, and
    /// must reject the boundary with an exact diagnostic when the required metadata cannot be recovered soundly.
    ///
    /// [`RecursiveBatchingDriver`] delegates its corresponding [`BatchingDriver::restore_batch`] request to this
    /// method. Region-carrying operation rules use that driver method, rather than calling [`BatchingPolicy::batch`]
    /// directly, when wrapping outputs of a rebuilt call whose attached batched region is opaque at the call boundary.
    /// Rules that replay a region directly into batch carriers need no restoration because no carrier metadata is
    /// erased at that boundary.
    ///
    /// # Parameters
    ///
    ///   - `value`: Packed parent-context result produced by the rebuilt region-carrying operation.
    ///   - `batch_axis`: Mapped axis inferred while structurally batching the source region.
    ///   - `r#type`: Per-item output type declared by the unbatched source region.
    ///   - `inputs`: Original batch carriers supplied to the region-carrying operation, from which
    ///     an overriding policy may recover transform-only metadata referenced by `r#type`.
    #[inline]
    fn restore_batch(
        value: C::Value,
        batch_axis: BatchAxis,
        r#type: &C::Type,
        inputs: &[Self::Batch],
    ) -> Result<Self::Batch, BatchingError> {
        let _ = r#type;
        let _ = inputs;
        Self::batch(value, batch_axis)
    }
}

/// Policy capability for invoking the public batching transform on flat parent values. [`Batch::batch`] owns
/// [`Parameterized`] broadcasting, tracer construction, closure invocation, and output structure reconstruction once
/// for every program universe. This capability owns the universe-specific boundary mechanics: selecting and validating
/// the mapped extent, packing and normalizing inputs, and materializing each requested output axis.
pub trait BatchingEntrypointPolicy<C: Context>: BatchingPolicy<C> {
    /// Prepares the provided input values for the batching transform and constructs a [`BatchingContext`] for them.
    fn prepare_inputs(
        context: &C,
        inputs: Vec<C::Value>,
        input_batch_axes: Vec<BatchAxis>,
        batch_axis: BatchAxisSpecification<Self::Extent>,
    ) -> Result<(BatchingContext<C, Self>, Vec<Self::Batch>), BatchingError>;

    /// Materializes `output` with the provided output [`BatchAxis`] and returns its parent value.
    fn materialize_output(
        context: &BatchingContext<C, Self>,
        output: Self::Batch,
        output_batch_axis: BatchAxis,
    ) -> Result<C::Value, BatchingError>;
}

/// Provides [`Instruction`](crate::Instruction)-scoped access to the nested [`Region`](crate::Region)s attached to
/// an [`Operation`] being batched. Every [`BatchableOperation`] application receives a [`BatchingDriver`]. The driver
/// keeps any attached regions borrowed and re-enters batching with the durable [`BatchingContext`] supplied by the
/// operation rule. [`RegionDriver`] provides its structural region access, while this trait adds batching-specific
/// recursion. Region-free applications expose a region count of zero through the same contract.
pub trait BatchingDriver<C: Context, P: BatchingPolicy<C>>: RegionDriver<C::Constant, C::Operation> {
    /// Batches the region at `index` over the provided batched values by re-entering the active batching transform.
    fn batch_region(
        &self,
        context: &BatchingContext<C, P>,
        index: usize,
        inputs: Vec<P::Batch>,
    ) -> Result<Vec<P::Batch>, BatchingError>;

    /// Batches `region` structurally at the provided input batch axes and output-axes policy, returning the rewritten
    /// standalone program and its inferred output batch axes.
    fn batch_program(
        &self,
        context: &BatchingContext<C, P>,
        region: RegionRef<'_, C::Constant, C::Operation>,
        input_axes: &[BatchAxis],
        output_axes_policy: ProgramBatchingOutputAxesPolicy,
    ) -> Result<P::BatchedProgram, BatchingError>;

    /// Restores a batch carrier for an output returned by a rebuilt region-carrying operation. Recursive drivers
    /// delegate to [`RecursiveBatchingPolicy::restore_batch`] so a policy can recover carrier-only metadata erased by
    /// an opaque region boundary. A driver that cannot recursively batch regions must still implement this method,
    /// normally by wrapping the packed value through [`BatchingPolicy::batch`]; its region operations will fail before
    /// such an output can be produced.
    ///
    /// # Parameters
    ///
    ///   - `value`: Packed parent-context result produced by the rebuilt region-carrying operation.
    ///   - `batch_axis`: Mapped axis inferred while batching the source region.
    ///   - `r#type`: Per-item output type declared by the unbatched source region.
    ///   - `inputs`: Original batch carriers supplied to the region-carrying operation.
    fn restore_batch(
        &self,
        value: C::Value,
        batch_axis: BatchAxis,
        r#type: &C::Type,
        inputs: &[P::Batch],
    ) -> Result<P::Batch, BatchingError>;
}

impl<C: Context, P: BatchingPolicy<C>> BatchingDriver<C, P> for EmptyRegionDriver {
    #[inline]
    fn batch_region(
        &self,
        _context: &BatchingContext<C, P>,
        _index: usize,
        _inputs: Vec<P::Batch>,
    ) -> Result<Vec<P::Batch>, BatchingError> {
        Err(ProgramError::MalformedProgram("empty region driver cannot batch a region".to_string()).into())
    }

    #[inline]
    fn batch_program(
        &self,
        _context: &BatchingContext<C, P>,
        _region: RegionRef<'_, C::Constant, C::Operation>,
        _input_axes: &[BatchAxis],
        _output_axes_policy: ProgramBatchingOutputAxesPolicy,
    ) -> Result<P::BatchedProgram, BatchingError> {
        Err(ProgramError::MalformedProgram("empty region driver cannot batch a program".to_string()).into())
    }

    #[inline]
    fn restore_batch(
        &self,
        value: C::Value,
        batch_axis: BatchAxis,
        _type: &C::Type,
        _inputs: &[P::Batch],
    ) -> Result<P::Batch, BatchingError> {
        P::batch(value, batch_axis)
    }
}

/// [`BatchingDriver`] scoped to one [`Operation`] application. It borrows the application's complete region driver,
/// which preserves the operation-defined ordering of owned regions, borrowed regions, and shared callees without
/// materializing a combined region collection. Recursive requests re-enter the active batching transform or batch a
/// selected region structurally.
pub struct RecursiveBatchingDriver<'r, D> {
    /// Application-scoped [`RegionDriver`].
    driver: &'r D,
}

impl<'r, D> RecursiveBatchingDriver<'r, D> {
    /// Creates a new [`RecursiveBatchingDriver`] over the provided application-scoped [`RegionDriver`].
    #[inline]
    pub fn new(driver: &'r D) -> Self {
        Self { driver }
    }

    /// Returns the underlying application-scoped [`RegionDriver`].
    #[inline]
    pub fn driver(&self) -> &'r D {
        self.driver
    }
}

impl<T: Type, V: Value<Type = T>, O: Operation<Type = T>, D: RegionDriver<V, O>> RegionDriver<V, O>
    for RecursiveBatchingDriver<'_, D>
{
    #[inline]
    fn regions<'r>(&'r self) -> impl Iterator<Item = RegionRef<'r, V, O>>
    where
        V: 'r,
        O: 'r,
    {
        self.driver().regions()
    }
}

impl<C: Context, P: RecursiveBatchingPolicy<C>, D: RegionDriver<C::Constant, C::Operation>> BatchingDriver<C, P>
    for RecursiveBatchingDriver<'_, D>
{
    #[inline]
    fn batch_region(
        &self,
        context: &BatchingContext<C, P>,
        index: usize,
        inputs: Vec<P::Batch>,
    ) -> Result<Vec<P::Batch>, BatchingError> {
        P::batch_region(context, self.region(index)?, inputs)
    }

    #[inline]
    fn batch_program(
        &self,
        context: &BatchingContext<C, P>,
        region: RegionRef<'_, C::Constant, C::Operation>,
        input_axes: &[BatchAxis],
        output_axes_policy: ProgramBatchingOutputAxesPolicy,
    ) -> Result<P::BatchedProgram, BatchingError> {
        P::batch_program(context, region, input_axes, output_axes_policy)
    }

    #[inline]
    fn restore_batch(
        &self,
        value: C::Value,
        batch_axis: BatchAxis,
        r#type: &C::Type,
        inputs: &[P::Batch],
    ) -> Result<P::Batch, BatchingError> {
        P::restore_batch(value, batch_axis, r#type, inputs)
    }
}

/// Represents the result of a [`BatchableOperation`] rule application which consists of the batch
/// carriers it produced together with the operation-local [`BatchingPolicy::Evidence`] it produced for
/// [`BatchingPolicy::validate_operation_outputs`]. A rule that has nothing to attest to converts its carriers directly
/// (i.e., `Ok(batches.into())`), which pairs them with [`Default`] evidence, so only the rules that genuinely make a
/// claim about what they did mention evidence at all. Because the evidence travels in the rule's return value rather
/// than on the carriers, the transform reads it at the validation boundary and drops it there, and it therefore cannot
/// reach the next operation.
pub struct BatchedOutputs<C: Context, P: BatchingPolicy<C>> {
    /// Batch carriers produced by the rule, in output order.
    batches: Vec<P::Batch>,

    /// Operation-local validation evidence produced by the rule.
    evidence: P::Evidence,
}

impl<C: Context, P: BatchingPolicy<C>> BatchedOutputs<C, P> {
    /// Creates a new [`BatchedOutputs`] instance.
    #[inline]
    pub fn new(batches: Vec<P::Batch>, evidence: P::Evidence) -> Self {
        Self { batches, evidence }
    }

    /// Consumes this [`BatchedOutputs`] instance and returns the underlying batch carriers along with their
    /// validation evidence.
    #[inline]
    pub fn into_parts(self) -> (Vec<P::Batch>, P::Evidence) {
        (self.batches, self.evidence)
    }
}

impl<C: Context, P: BatchingPolicy<C, Evidence: Debug>> Debug for BatchedOutputs<C, P> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("BatchedOutputs")
            .field("batches", &self.batches)
            .field("evidence", &self.evidence)
            .finish()
    }
}

impl<C: Context, P: BatchingPolicy<C, Batch: PartialEq, Evidence: PartialEq>> PartialEq for BatchedOutputs<C, P> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.batches == other.batches && self.evidence == other.evidence
    }
}

impl<C: Context, P: BatchingPolicy<C>> From<Vec<P::Batch>> for BatchedOutputs<C, P> {
    #[inline]
    fn from(batches: Vec<P::Batch>) -> Self {
        Self::new(batches, P::Evidence::default())
    }
}

// TODO(eaplatanios): Restore the strict `Operation<Type = C::Type>` super-trait bound once the next-generation trait
//  solver stabilizes. The current solver cannot discharge this projection equality at implementation heads whose
//  context type is built from `Self` (E0284); the equality is enforced per method through `where` clauses instead.
/// Represents [`Operation`]s that can be batched (i.e., vectorized). The trait is parameterized by the parent
/// [`Context`] `C` that owns the packed values flowing through the batching transform, and every rule receives
/// the active durable [`BatchingContext`] plus a [`BatchingDriver`]. Ordinary rules lift their operation to its
/// batch-carrying inputs and then execute or stage the lifted work through `context.parent()` (typically via
/// [`InterpretableBatchableOperation::interpret_with_batch_axes`]), so an eager parent interprets it immediately while
/// a staging parent stages it into the enclosing trace. Rules whose semantics depend on the active transform frame
/// (e.g., named-axis collectives) inspect [`BatchingContext::axis_name`] and [`BatchingContext::axis_extent`] directly,
/// and recursive higher-order rules replay their nested programs through the same contract. Consequently, invoking a
/// batching rule always requires an active [`BatchingContext`].
///
/// # Deriving Batchable Operation Enums
///
/// `#[derive(Operation)]` generates a [`BatchableOperation`] dispatcher when the enum specifies
/// `#[ryft(dispatch(batching))]`. It follows the operation derivation's enum-shape and type-inference rules
/// and generates:
///
///   - A dispatcher at `BatchableOperation<C, ArrayBatching>`, where [`ArrayBatching`](crate::ArrayBatching) is the
///     array-domain policy and `C` is the parent [`Context`], that forwards the active [`BatchingContext`] to every
///     variant's own rule. One dispatcher covers eager and staging parents alike, because the parent/active distinction
///     lives in each rule's body rather than in dispatch.
///   - Per-variant `Payload: BatchableOperation<C, ArrayBatching>` predicates that transport each rule's own
///     capability requirements to the use site. Nested programs batch structurally through [`Program::batched`],
///     requested by higher-order rules through their active [`BatchingDriver`], whose concrete implementation
///     establishes the finite program-level bounds at its construction site.
///
/// The super-trait is a plain [`Operation`] rather than `Operation<Type = C::Type>` because the current trait solver
/// cannot discharge that projection equality at implementation heads whose batching context is itself built from
/// `Self`. The equality is instead required per method through `where Self: Operation<Type = C::Type>`, so a payload
/// whose [`Operation::Type`] disagrees with `C::Type` cannot be batched in `C`: the requirement is restated by the
/// derived dispatcher's per-payload predicates and by the generic projected-batching helpers, and any mismatched
/// payload is rejected with a type-mismatch error at its use site.
pub trait BatchableOperation<C: Context, P: BatchingPolicy<C>>: Operation {
    /// Applies this operation to packed batched inputs, returning batched outputs with the resulting batch axes.
    /// `context` borrows the durable [`BatchingContext`] for the transform level being applied. `driver` exposes the
    /// current operation application's regions and has a region count of zero for region-free applications. Packed
    /// values in `inputs` and the returned outputs are owned by `context.parent()`. Note that the resulting
    /// [`BatchedOutputs`] instance contains the output carriers along with the validation [`BatchingPolicy::Evidence`]
    /// that [`BatchingPolicy::validate_operation_outputs`] consumes. A rule that makes no claim about what it did with
    /// its inputs' carrier metadata simply converts its carriers (i.e., `Ok(batches.into())`).
    ///
    /// # Contract
    ///
    ///   - **Axis Alignment:** If two or more inputs carry a mapped axis (i.e., `batch_axis.is_some()`), elementwise
    ///     operations require them to agree on the axis position. When they disagree, this function returns
    ///     [`BatchingError::MisalignedBatchAxes`] with an error message that names the misaligned axes and suggests the
    ///     user repositions one of them with [`Transpose`] (i.e., the N-D axis permutation primitive) before invoking
    ///     the operation. Operations with explicit axis arguments (e.g., `Dot`, `Transpose`, `Reshape`, etc.) rewrite
    ///     those arguments to thread the mapped axis through correctly.
    ///   - **Output Axes:** For elementwise operations, the output
    ///     [`ArrayBatch::batch_axis`](crate::ArrayBatch::batch_axis) matches the common input batch axis.
    ///     For operations with explicit axis arguments, the output axis follows from the lifted axis arguments.
    ///   - **Zero Propagation:** Linear batching rules preserve zero tangent payloads through their operation-specific
    ///     semantics. Canonical staged zeros are handled before batching reaches concrete value-level interpretation.
    ///   - **Parent-Owned Work:** Ordinary rules must execute or stage lifted work through `context.parent()` and
    ///     return parent-owned values; only rules keyed on the active frame's axis metadata inspect the
    ///     [`BatchingContext`] itself.
    ///
    /// Note that in order to be able to provide [`BatchableOperation`] implementations for operation families that
    /// select the generated batching dispatcher, it is a common convention for operations that can be part of such
    /// operation families to implement this trait even if they do not support batching and to have this  function
    /// simply return a [`BatchingError::UnsupportedOperation`] error.
    fn batch<D: BatchingDriver<C, P>>(
        &self,
        context: &BatchingContext<C, P>,
        driver: &D,
        inputs: &[P::Batch],
    ) -> Result<BatchedOutputs<C, P>, BatchingError>
    where
        Self: Operation<Type = C::Type>;
}

/// Batching rule for a member [`Operation`] whose instruction has a mixed signature in a parent operation universe.
/// If this operation has native type `T` while the enclosing operation family uses type `U`, an ordinary
/// [`BatchableOperation`] implementation can describe only the homogeneous `T -> T` contract because its
/// [`batch`](BatchableOperation::batch) method requires `Self::Type = C::Type`. This trait instead receives the parent
/// [`BatchingContext`] and its `U`-typed batches, allowing the rule to account for operands or results belonging to
/// other members of `U`.
///
/// Shape-changing collectives are the motivating example. Their native array operation consumes one array, whereas
/// their array-or-dimension instruction consumes that array followed by first-class result extents. Operation-family
/// dispatchers should use this trait only when the parent signature differs from the operation's native
/// [`Operation::Type`].
pub trait MemberBatchableOperation<C: Context, P: BatchingPolicy<C>>: Operation {
    /// Applies this member operation's batching rule in its enclosing parent operation universe.
    ///
    /// # Parameters
    ///
    ///   - `context`: Active parent [`BatchingContext`] through which the rule stages mixed operations.
    ///   - `driver`: Instruction-scoped [`BatchingDriver`] exposing any attached regions.
    ///   - `inputs`: Parent-universe batches in the mixed instruction's operand order.
    fn batch_in_parent<D: BatchingDriver<C, P>>(
        &self,
        context: &BatchingContext<C, P>,
        driver: &D,
        inputs: &[P::Batch],
    ) -> Result<BatchedOutputs<C, P>, BatchingError>;
}

/// Policy for choosing a batched [`Program`]'s output axes. Program batching always replays the program over packed
/// values whose mapped batch axes are specified by the caller. This policy controls how the replayed output tracers are
/// packaged into the resulting program.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProgramBatchingOutputAxesPolicy {
    /// Keep the output axes naturally produced by the per-operation batching rules.
    /// Replicated outputs remain replicated.
    Natural,

    /// Align/normalize every output to the specified mapped axis, moving already-batched outputs with [`Transpose`]
    /// and broadcasting replicated outputs across the batch.
    AlignAllTo(Axis),

    /// Align each output `i` to the mapped axis of the `i`-th entry, with one entry per program output. A *mapped*
    /// entry forces the output to carry its batch axis at that position, moving an already-batched output with
    /// [`Transpose`] and broadcasting a replicated output across the batch, while a *replicated* entry keeps that
    /// output's natural axis (it is a lower bound, not an equality constraint).
    AlignEachTo(Vec<BatchAxis>),
}

/// Capability to interpret an [`Operation`] on [`BatchingPolicy`]-selected batches and repackage its outputs as batches
/// carrying specified axes. This is the shared application path the per-operation [`BatchableOperation`] rules use once
/// they have lifted an operation to its batch-carrying inputs. It centralizes the packed-value ownership invariant of
/// the batching transform: the lifted operation is always interpreted against `context.parent()`, which owns the packed
/// values, and never against the [`BatchingContext`] itself. Every [`InterpretableOperation`] over the parent's values
/// gets it for free through the blanket implementation below, so an operation earns it directly from its interpretation
/// rule.
pub trait InterpretableBatchableOperation<C: Context, P: BatchingPolicy<C>> {
    /// Interprets this operation on the *unpacked* values of batched `inputs` against `context.parent()` and repackages
    /// each output as a [`BatchingPolicy`]-selected batch carrying the corresponding `output_batch_axes` entry.
    ///
    /// # Parameters
    ///
    ///   - `context`: Active [`BatchingContext`] whose parent interprets the lifted operation,
    ///     as in [`InterpretableOperation::interpret`].
    ///   - `inputs`: Batch-carrying inputs the lifted operation is interpreted over.
    ///   - `output_batch_axes`: [`BatchAxis`] to attach to each produced output. This slice must have one entry
    ///     per output that this [`Operation`] produces on these inputs.
    fn interpret_with_batch_axes(
        &self,
        context: &BatchingContext<C, P>,
        inputs: &[P::Batch],
        output_batch_axes: &[BatchAxis],
    ) -> Result<Vec<P::Batch>, BatchingError>;
}

impl<C: Context, O: InterpretableOperation<C>, P: BatchingPolicy<C>> InterpretableBatchableOperation<C, P> for O {
    fn interpret_with_batch_axes(
        &self,
        context: &BatchingContext<C, P>,
        inputs: &[P::Batch],
        output_batch_axes: &[BatchAxis],
    ) -> Result<Vec<P::Batch>, BatchingError> {
        // Every `InterpretableOperation` over the parent context's values is an `InterpretableBatchableOperation`.
        // The batched interpretation unpacks the input values, interprets the operation once through `interpret`
        // against the parent context (which owns those packed values), and repackages each output with its
        // requested `BatchAxis`.
        if inputs.is_empty() {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
        }
        let input_values = inputs.iter().map(P::value).cloned().collect::<Vec<_>>();
        let output_values = self.interpret(&context.parent().clone(), &EmptyRegionDriver, input_values.as_slice())?;
        check_count!("output", output_values, output_batch_axes.len(), ProgramError);
        output_values
            .into_iter()
            .zip(output_batch_axes.iter().copied())
            .map(|(value, axis)| P::batch(value, axis))
            .collect()
    }
}

/// [`Context`] used to batch a computation by introducing exactly one mapped batch axis. [`BatchingContext`] is the
/// active context for one batching level. Its [`BatchingPolicy`] selects both the batch-carrying representation and the
/// extent representation, allowing ordinary homogeneous arrays to use a static `usize` while composite programs retain
/// a first-class dimension value. [`Operation`]s bound through this context are lifted through their
/// [`BatchableOperation`] implementations into the parent context, so nested transforms compose without
/// making an active transform pretend to be a backend domain.
#[derive(Debug, Clone)]
pub struct BatchingContext<C: Context, P: BatchingPolicy<C>> {
    /// [`Context`] that this [`BatchingContext`] is nested into.
    parent: C,

    /// Extent of the new batch axis.
    axis_extent: P::Extent,

    /// Optional name for the new batch axis that enables [`Operation`]s (e.g., collective operations)
    /// to address this axis by name.
    axis_name: Option<String>,

    /// Sharding placement of the transform-owned mapped dimension.
    axis_sharding: ShardingDimension,
}

impl<C: Context, P: BatchingPolicy<C>> BatchingContext<C, P> {
    /// Creates a new [`BatchingPolicy`]-explicit [`BatchingContext`] with an unnamed and
    /// [`ShardingDimension::Replicated`] mapped axis.
    ///
    /// # Parameters
    ///
    ///   - `parent`: [`Context`] into which batched operations are lifted.
    ///   - `axis_extent`: Extent of the transform-owned mapped dimension.
    #[inline]
    pub fn with_policy(parent: C, axis_extent: P::Extent) -> Self {
        Self { parent, axis_extent, axis_name: None, axis_sharding: ShardingDimension::Replicated }
    }

    /// Sets the optional name through which [`Operation`]s such as collectives can address the mapped axis/dimension.
    #[inline]
    pub fn with_axis_name<A: Into<Option<String>>>(mut self, axis_name: A) -> Self {
        self.axis_name = axis_name.into();
        self
    }

    /// Sets the sharding placement of the transform-owned mapped axis/dimension.
    #[inline]
    pub fn with_axis_sharding(mut self, axis_sharding: ShardingDimension) -> Self {
        self.axis_sharding = axis_sharding;
        self
    }

    /// Returns the [`Context`] that this [`BatchingContext`] is nested into.
    #[inline]
    pub fn parent(&self) -> &C {
        &self.parent
    }

    /// Returns the extent of the new batch axis.
    #[inline]
    pub fn axis_extent(&self) -> &P::Extent {
        &self.axis_extent
    }

    /// Returns the optional name for the new batch axis that enables [`Operation`]s (e.g., collective operations)
    /// to address this axis by name.
    #[inline]
    pub fn axis_name(&self) -> Option<&str> {
        self.axis_name.as_deref()
    }

    /// Returns the sharding placement of the transform-owned mapped dimension.
    #[inline]
    pub fn axis_sharding(&self) -> &ShardingDimension {
        &self.axis_sharding
    }
}

impl<C: Context<Type: BatchableType<Policy: BatchingPolicy<C>>>>
    BatchingContext<C, <C::Type as BatchableType>::Policy>
{
    /// Creates a [`BatchingContext`] using the canonical policy selected by the provided context's [`BatchableType`].
    #[inline]
    pub fn new(parent: C, axis_extent: <<C::Type as BatchableType>::Policy as BatchingPolicy<C>>::Extent) -> Self {
        Self::with_policy(parent, axis_extent)
    }
}

impl<C: Context, P: BatchingPolicy<C>> BatchingContext<C, P> {
    /// Aligns the logical outputs of `batched_program` to `target_output_axes` while preserving the complete program
    /// boundary selected by `P::BatchedProgram`, including any [`BatchingPolicy`]-owned bookkeeping inputs or outputs.
    /// The existing transformed [`Program`] is returned when its output axes already match. Otherwise, `region` is
    /// batched again with its live outputs aligned to the target axes.
    ///
    /// [`Region`](crate::Region)-carrying operation rules use this after discovering and semantically reconciling
    /// natural output axes across related regions. This method only performs the mechanical output-boundary alignment.
    /// It deliberately does not decide which axes correspond or which one should win. A mapped target axis moves a
    /// naturally mapped output or broadcasts a naturally replicated output. Callers must never use this method to
    /// collapse a mapped output to replicated, because that requires [`Operation`]-specific semantics such as cotangent
    /// summation.
    ///
    /// # Parameters
    ///
    ///   - `driver`: [`BatchingDriver`] that structurally transforms `region` if alignment is required.
    ///   - `region`: Source [`Region`](crate::Region) from which `batched_program` was produced.
    ///   - `input_axes`: Batch axes used to produce `batched_program` and to perform any aligned replay.
    ///   - `batched_program`: Batched [`Program`] with its policy-specific boundary and natural output axes.
    ///   - `target_output_axes`: Semantically reconciled output axes targeted by the region's consumer.
    pub(crate) fn align_batched_program_outputs<D: BatchingDriver<C, P>>(
        &self,
        driver: &D,
        region: RegionRef<'_, C::Constant, C::Operation>,
        input_axes: &[BatchAxis],
        batched_program: P::BatchedProgram,
        target_output_axes: &[BatchAxis],
    ) -> Result<Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>, BatchingError> {
        let aligned_program =
            self.align_batched_program(driver, region, input_axes, batched_program, target_output_axes)?;
        Ok(aligned_program.into_parts().0)
    }

    /// Aligns the logical outputs in the same way as [`Self::align_batched_program_outputs`], then uses
    /// [`BatchingPolicy::adapt_batched_program`] to convert the policy-specific boundary into an ordinary attached
    /// [`Region`](crate::Region) boundary. Adaptation preserves bookkeeping inputs required by the region itself but
    /// removes bookkeeping outputs used only by operations that thread batching state through their regions.
    ///
    /// Alignment happens before adaptation because the reconciled `target_output_axes` describe the source region's
    /// outputs, which is exactly the axis vector both steps are stated over (a policy's bookkeeping outputs never
    /// appear in [`BatchedProgram::output_axes`]). Callers must reintroduce the shed inputs by prepending
    /// [`BatchingPolicy::boundary_operands`] to the operands of the operation they attach the result to.
    ///
    /// Like [`Self::align_batched_program_outputs`], this performs no mapped-to-replicated collapse. Alignment resolves
    /// every structurally movable mismatch, and collapsing a genuinely mapped output requires [`Operation`]-specific
    /// semantics such as cotangent summation. Callers that need a collapse must batch and adapt the region themselves
    /// with their own collapse function.
    ///
    /// # Parameters
    ///
    ///   - `driver`: [`BatchingDriver`] that structurally transforms `region` if alignment is required.
    ///   - `region`: Source [`Region`](crate::Region) from which `batched_program` was produced.
    ///   - `input_axes`: Batch axes used to produce `batched_program` and to perform any aligned replay.
    ///   - `batched_program`: Batched [`Program`] with its policy-specific boundary and natural output axes.
    ///   - `target_output_axes`: Semantically reconciled output axes targeted by the region's consumer.
    pub(crate) fn align_and_adapt_batched_program_outputs<D: BatchingDriver<C, P>>(
        &self,
        driver: &D,
        region: RegionRef<'_, C::Constant, C::Operation>,
        input_axes: &[BatchAxis],
        batched_program: P::BatchedProgram,
        target_output_axes: &[BatchAxis],
    ) -> Result<Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>, BatchingError> {
        let aligned_program =
            self.align_batched_program(driver, region, input_axes, batched_program, target_output_axes)?;
        let adapted_program = P::adapt_batched_program(aligned_program, None, |_, _, axis| {
            Err(BatchingError::MisalignedBatchAxes {
                message: format!("cannot collapse a batched region output mapped along {axis} without a consumer rule"),
            })
        })?;
        Ok(adapted_program.into_parts().0)
    }

    /// Returns `batched_program` when its output axes already equal `target_output_axes`, or structurally batches
    /// `region` again while aligning its live outputs to those axes. This is the shared alignment step behind
    /// [`Self::align_batched_program_outputs`] and [`Self::align_and_adapt_batched_program_outputs`], which differ
    /// only in whether they adapt the policy-specific program boundary afterwards.
    fn align_batched_program<D: BatchingDriver<C, P>>(
        &self,
        driver: &D,
        region: RegionRef<'_, C::Constant, C::Operation>,
        input_axes: &[BatchAxis],
        batched_program: P::BatchedProgram,
        target_output_axes: &[BatchAxis],
    ) -> Result<P::BatchedProgram, BatchingError> {
        if batched_program.output_axes() == target_output_axes {
            return Ok(batched_program);
        }
        let aligned_program = driver.batch_program(
            self,
            region,
            input_axes,
            ProgramBatchingOutputAxesPolicy::AlignEachTo(target_output_axes.to_vec()),
        )?;
        if aligned_program.output_axes() != target_output_axes {
            let actual_output_axes =
                aligned_program.output_axes().iter().map(ToString::to_string).collect::<Vec<_>>().join(", ");
            let target_output_axes = target_output_axes.iter().map(ToString::to_string).collect::<Vec<_>>().join(", ");
            return Err(BatchingError::MisalignedBatchAxes {
                message: format!(
                    "batched region output axes [{actual_output_axes}] do not match the target output axes \
                     [{target_output_axes}]",
                ),
            });
        }
        Ok(aligned_program)
    }
}

impl<C: Context<Operation: BatchableOperation<C, P>>, P: RecursiveBatchingPolicy<C>> Domain for BatchingContext<C, P> {
    type Type = C::Type;
    type Value = BatchingTracer<C, P>;
    type Constant = C::Constant;
    type Operation = C::Operation;
}

impl<C: Context<Operation: BatchableOperation<C, P>>, P: RecursiveBatchingPolicy<C>> Context for BatchingContext<C, P> {
    #[inline]
    fn lift(&self, constant: C::Constant) -> Result<BatchingTracer<C, P>, ProgramError> {
        // Lifts a constant by lifting it in the parent context and replicating it across the batch.
        Ok(BatchingTracer::new(self.clone(), P::replicated(self.parent().lift(constant)?)))
    }

    #[inline]
    fn bind<O: Into<Self::Operation>, D: BindingRegionDriver<Self::Constant, Self::Operation>>(
        &self,
        operation: O,
        driver: D,
        inputs: &[BatchingTracer<C, P>],
    ) -> Result<Vec<BatchingTracer<C, P>>, ProgramError> {
        // Binding routes the operation through its `BatchableOperation` implementation against the batch-carrying
        // inputs. The implementation dispatches primitive work through the parent context, executing eagerly under an
        // eager parent or staging into an enclosing trace under a staging parent, and axis-referencing work (e.g.,
        // collectives) through this batching context, and so multi-operation lowering (e.g., a batch-varying
        // `Instruction` becoming two branches plus a per-item select instruction) emerges automatically.
        let operation = operation.into();
        let input_batches = inputs.iter().map(|input| input.batch().clone()).collect::<Vec<_>>();
        let driver = RecursiveBatchingDriver::new(&driver);

        // The rule's evidence is scoped to this one application. It is read by the validation boundary below and
        // dropped at the end of this statement group, so it can never be observed by the next operation.
        let (output_batches, evidence) = operation.batch(self, &driver, input_batches.as_slice())?.into_parts();
        P::validate_operation_outputs(
            operation.name(),
            input_batches.as_slice(),
            output_batches.as_slice(),
            &evidence,
        )?;
        Ok(output_batches.into_iter().map(|batch| BatchingTracer::new(self.clone(), batch)).collect())
    }

    #[inline]
    fn is_eager(&self) -> bool {
        self.parent().is_eager()
    }

    #[inline]
    fn resolve(&self, value: &BatchingTracer<C, P>) -> ValueResolution<C::Constant> {
        self.parent().resolve(P::value(value.batch()))
    }
}

/// Batch-carrying value flowing through a [`BatchingContext`]. The function being batched operates on
/// [`BatchingTracer`]s directly. Each operation dispatches through the stamped context via [`Context::bind`], which
/// applies the operation's [`BatchableOperation`] implementation against the parent context. An eager parent owns
/// concrete packed values, while a staging parent owns [`Tracer`]s in the enclosing trace. The [`Typed`] view is always
/// the policy-provided unbatched per-item type. The policy-selected inner carrier retains the packed value and any
/// mapped-axis metadata.
#[derive(Clone, Parameter)]
pub struct BatchingTracer<C: Context, P: BatchingPolicy<C>> {
    /// [`BatchingContext`] this value flows through, used to dispatch operations that involve it.
    context: BatchingContext<C, P>,

    /// [`BatchingPolicy`]-selected batch that corresponds to the batched underlying value.
    batch: P::Batch,
}

impl<C: Context, P: BatchingPolicy<C>> BatchingTracer<C, P> {
    /// Creates a new [`BatchingTracer`].
    #[inline]
    pub fn new(context: BatchingContext<C, P>, batch: P::Batch) -> Self {
        Self { context, batch }
    }

    /// Returns the [`BatchingContext`] this [`BatchingTracer`] flows through.
    #[inline]
    pub fn context(&self) -> &BatchingContext<C, P> {
        &self.context
    }

    /// Returns the [`BatchingPolicy`]-selected batch that corresponds to the batched underlying value.
    #[inline]
    pub fn batch(&self) -> &P::Batch {
        &self.batch
    }

    /// Consumes this [`BatchingTracer`] and returns the underlying [`BatchingPolicy`]-selected batch.
    #[inline]
    pub fn into_batch(self) -> P::Batch {
        self.batch
    }

    /// Returns the extent of the mapped axis introduced by this [`BatchingTracer`]'s active batching context.
    #[inline]
    pub fn batch_extent(&self) -> &P::Extent {
        self.context.axis_extent()
    }
}

impl<C: Context, P: BatchingPolicy<C, Batch: PartialEq>> PartialEq for BatchingTracer<C, P> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        // A batch-carrying value compares by its packed value (through that value's own `PartialEq`, which is
        // identity-shaped for tracer-valued parents) and its batch axis, ignoring the stamped context. Consumers such
        // as the scan/while loop-invariance fixed points of partial evaluation compare flowing values across replay
        // rounds to detect passthrough, and a batched value passes through exactly when its packed value does on the
        // same axis.
        self.batch == other.batch
    }
}

impl<C: Context, P: BatchingPolicy<C>> Debug for BatchingTracer<C, P> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("BatchingTracer").field("batch", &self.batch).finish()
    }
}

impl<C: Context, P: BatchingPolicy<C>> Display for BatchingTracer<C, P> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.batch)
    }
}

impl<C: Context, P: BatchingPolicy<C>> Typed for BatchingTracer<C, P> {
    type Type = C::Type;

    #[inline]
    fn r#type(&self) -> Cow<'_, C::Type> {
        P::unbatched_type(&self.batch)
    }
}

impl<C: Context<Operation: BatchableOperation<C, P>>, P: RecursiveBatchingPolicy<C>> Value for BatchingTracer<C, P> {
    type DispatchDomain = BatchingContext<C, P>;
    type ExecutionDomain = BatchingContext<C, P>;

    #[inline]
    fn dispatch_domain(&self) -> BatchingContext<C, P> {
        self.context().clone()
    }

    #[inline]
    fn execution_domain(&self) -> BatchingContext<C, P> {
        self.context().clone()
    }
}

/// Extension trait that exposes a [`BatchingEntrypointPolicy`]-selected batching transform as a function on a
/// [`Context`]. Refer to [`batch`] for the transform semantics. This trait also serves call sites that must name the
/// context explicitly, most notably empty input structures from which the free function cannot recover an execution
/// domain. [`Self::Policy`] selects the batch carrier, mapped-extent representation, and boundary mechanics for this
/// context's program universe.
pub trait Batch: Context {
    /// Canonical [`BatchingEntrypointPolicy`] for this [`Context`]'s [`Program`] universe.
    type Policy: BatchingEntrypointPolicy<Self>;

    /// Batches `function` over the mapped axes of `input`, with this [`Context`] executing (or staging) the batched
    /// operations. Refer to the documentation of the [`batch`] function for information on the batching transform and
    /// its arguments. Unlike that function, this method also serves inputs with no leaf values (i.e., that are empty),
    /// provided that `batch_axis` supplies an explicit batch extent.
    fn batch<
        F: FnOnce(I::To<BatchingTracer<Self, Self::Policy>>) -> Result<O, ProgramError>,
        I: Parameterized<
                Self::Value,
                Family: ParameterizedFamily<BatchAxis> + ParameterizedFamily<BatchingTracer<Self, Self::Policy>>,
            >,
        O: Parameterized<
                BatchingTracer<Self, Self::Policy>,
                Family: ParameterizedFamily<BatchAxis> + ParameterizedFamily<Self::Value>,
            >,
        InputBatchAxes: Parameterized<BatchAxis>,
        OutputBatchAxes: Parameterized<BatchAxis>,
        Specification: Into<BatchAxisSpecification<<Self::Policy as BatchingPolicy<Self>>::Extent>>,
    >(
        &self,
        function: F,
        input: I,
        input_batch_axes: InputBatchAxes,
        output_batch_axes: OutputBatchAxes,
        batch_axis: Specification,
    ) -> Result<O::To<Self::Value>, BatchingError> {
        let input_structure = input.parameter_structure();
        let inputs = input.into_parameters().collect::<Vec<_>>();

        // Broadcast the caller's `input_batch_axes` into the input parameter structure. A single `BatchAxis` leaf fills
        // every input leaf, a matching structure gives one axis per leaf, and a smaller compatible structure broadcasts
        // based on its prefixes. A structure that cannot fill the input surfaces as a `ParameterError`.
        let input_batch_axes = input_batch_axes
            .broadcast_to_parameter_structure::<I::To<BatchAxis>>(input_structure.clone())?
            .into_parameters()
            .collect::<Vec<_>>();

        // The active policy validates and packs flat inputs, selects the extent representation, normalizes sharding
        // placement, and returns the configured context. Parameter structure remains entirely outside that boundary.
        let (context, inputs) = Self::Policy::prepare_inputs(self, inputs, input_batch_axes, batch_axis.into())?;
        let inputs = inputs.into_iter().map(|batch| BatchingTracer::new(context.clone(), batch)).collect::<Vec<_>>();
        let input = I::To::<BatchingTracer<Self, Self::Policy>>::from_parameters(input_structure, inputs)?;
        let output = function(input)?;

        // Broadcast the caller's `output_batch_axes` into the output parameter structure, mirroring the
        // `input_batch_axes` handling above. A single `BatchAxis` leaf applies to every output, and a matching
        // structure gives one axis per leaf.
        let output_structure = output.parameter_structure();
        let output_batch_axis_values = output_batch_axes
            .broadcast_to_parameter_structure::<O::To<BatchAxis>>(output_structure.clone())?
            .into_parameters()
            .collect::<Vec<_>>();

        // The active policy materializes each requested output axis. Unwrapping to parent values preserves any
        // enclosing batching level's metadata, so nested transforms require no side table.
        let parent_outputs = output
            .into_parameters()
            .zip(output_batch_axis_values)
            .map(|(output, output_batch_axis)| {
                Self::Policy::materialize_output(&context, output.into_batch(), output_batch_axis)
            })
            .collect::<Result<Vec<_>, BatchingError>>()?;

        Ok(O::To::<Self::Value>::from_parameters(output_structure, parent_outputs)?)
    }
}

impl<C: Context<Type: BatchableType<Policy: BatchingEntrypointPolicy<C>>>> Batch for C {
    type Policy = <C::Type as BatchableType>::Policy;
}

/// Batches the provided `function` over the mapped axes of `input`, running it once over whole batches instead of once
/// per batch item. This is the batching (i.e., vectorization) transform and the analogue of
/// [JAX's `vmap`](https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html).
///
/// The transform recovers a [`Context`] from the input's leaf values through [`Value::ExecutionDomain`], wraps it in
/// a [`BatchingContext`], and invokes `function` using [`BatchingTracer`] values, so that every operation inside the
/// closure is lifted through its [`BatchableOperation`] implementation against the recovered context. This composes
/// uniformly across the whole stack: an eager backend context interprets each batched operation immediately, an active
/// [`StagingContext`] stages it into the enclosing trace, and a [`BatchingContext`] nests `batch` inside `batch` (each
/// level's [`BatchingTracer`] carries its own batch axis and so nested maps thread through with no side table).
/// Concretely, staged [`Tracer`]s recover their trace, [`BatchingTracer`]s recover their batching level, concrete
/// values recover the eager backend domain they name, etc. Inputs with *no leaf values* are the one case this function
/// cannot serve. With nothing to recover a context from, it returns [`BatchingError::EmptyBatch`] even when
/// `batch_axis` supplies an explicit batch extent. [`Batch::batch`] must be used in that case, with an explicit
/// context.
///
/// `input_batch_axes` selects the mapped axis of each input leaf and `output_batch_axes` the position of the mapped
/// axis in each output leaf. Both are [`Parameterized`] values over [`BatchAxis`] leaves that are broadcast into the
/// corresponding parameter structure via [`Parameterized::broadcast_to_parameter_structure`]: a single [`BatchAxis`]
/// applies to every leaf, a value whose structure matches gives one axis per leaf, and a smaller compatible structure
/// broadcasts based on its prefixes. On the input side, [`BatchAxis::new(k)`](BatchAxis::new) maps the leaf on axis
/// `k` of its packed type, while [`BatchAxis::replicated`] shares the leaf unchanged across the batch. On the output
/// side, [`BatchAxis::new(k)`](BatchAxis::new) requests the mapped axis at position `k`: a naturally mapped output is
/// transposed when needed, while a naturally replicated output is broadcast across the batch, matching JAX's `vmap`.
/// Negative axes are normalized against the packed input or requested output rank, so `-1` denotes the final axis.
/// [`BatchAxis::replicated`] requires the output to remain replicated; collapsing a genuinely mapped output instead
/// requires an explicit reduction inside `function`.
///
/// When at least one input is mapped, the batch extent is inferred from those inputs. The `batch_axis` argument accepts
/// the specification form selected by the execution context's [`Batch::Policy`]. It can supply an explicit extent
/// (either to pin the inferred extent or to drive a fully-replicated transform whose extent would otherwise be
/// unobservable) and an axis name that operations inside `function` like collectives can address.
///
/// # Parameters
///
///   - `function`: Function that represents the computation that needs to be batched/vectorized.
///   - `input`: Input (potentially structured) that should be batched/vectorized.
///   - `input_batch_axes`: [`BatchAxis`] selection for the input leaves, broadcast into the input's structure.
///   - `output_batch_axes`: [`BatchAxis`] selection for the output leaves, broadcast into the output's structure.
///   - `batch_axis`: [`BatchAxisSpecification`] to use carrying an optional explicit batch extent and an optional
///     batch axis name.
#[inline]
pub fn batch<
    V: Value<ExecutionDomain: Batch>,
    F: FnOnce(I::To<BatchingTracer<V::ExecutionDomain, <V::ExecutionDomain as Batch>::Policy>>) -> Result<O, ProgramError>,
    I: Parameterized<
            V,
            Family: ParameterizedFamily<BatchAxis>
                        + ParameterizedFamily<BatchingTracer<V::ExecutionDomain, <V::ExecutionDomain as Batch>::Policy>>,
        >,
    O: Parameterized<
            BatchingTracer<V::ExecutionDomain, <V::ExecutionDomain as Batch>::Policy>,
            Family: ParameterizedFamily<BatchAxis> + ParameterizedFamily<V>,
        >,
    InputBatchAxes: Parameterized<BatchAxis>,
    OutputBatchAxes: Parameterized<BatchAxis>,
    Specification: Into<BatchAxisSpecification<<<V::ExecutionDomain as Batch>::Policy as BatchingPolicy<V::ExecutionDomain>>::Extent>>,
>(
    function: F,
    input: I,
    input_batch_axes: InputBatchAxes,
    output_batch_axes: OutputBatchAxes,
    batch_axis: Specification,
) -> Result<O::To<V>, BatchingError> {
    let Some(context) = input.parameters().next().map(Value::execution_domain) else {
        return Err(BatchingError::EmptyBatch);
    };
    context.batch(function, input, input_batch_axes, output_batch_axes, batch_axis)
}

/// Applies a member [`Operation`]'s batching rule through a projected view of a composite [`BatchingContext`]. Use
/// this function from a composite operation dispatcher when the operation is [`Region`](crate::Region)-free and every
/// operand and result belongs to the same projectable member type `T`. It converts the packed input values to the
/// member value family, preserves the outer batch axes and mapped extent, runs the member's existing batching rule,
/// and converts the results back to the composite value family, relaying the member rule's validation evidence
/// unchanged. [`BatchingPolicyProjection`] selects the member policy that represents that same extent and evidence for
/// the specific projected type `T`. This keeps homogeneous member rules independent of the enclosing composite type.
///
/// Operations with mixed member types or attached regions require an explicit composite batching rule instead. A
/// member operation that declares [`RegionSlot`](crate::RegionSlot)s is rejected with an exact diagnostic naming it,
/// because projection reaches the member rule with no region access: the attached regions are programs in the
/// _composite_ universe, and no projected driver can present them in the member universe.
///
/// # Parameters
///
///   - `context`: Active composite [`BatchingContext`] whose mapped extent, axis metadata, and parent context are
///     preserved while the member rule runs.
///   - `operation`: Region-free operation expressed in the projected member operation family.
///   - `inputs`: Packed composite batches corresponding to the operation's operands.
pub fn batch_projected_operation<
    T: Type,
    O: Operation<Type = T> + BatchableOperation<ProjectedContext<C, T>, P::Projected>,
    P: BatchingPolicyProjection<C, T>,
    C: Context<
            Value: ValueProjection<T, Projected: Value<Type = T>>,
            Constant: ValueProjection<T, Projected: Value<Type = T>>,
            Operation: OperationProjection<T>,
        >,
>(
    context: &BatchingContext<C, P>,
    operation: &O,
    inputs: &[P::Batch],
) -> Result<BatchedOutputs<C, P>, BatchingError> {
    if !operation.region_slots().is_empty() {
        return Err(BatchingError::UnsupportedOperation {
            message: format!(
                "projected operation `{}` carries regions and cannot be batched through its member family; batch it \
                 through a composite carrier for that operation instead",
                operation.name(),
            ),
        });
    }
    let projected_context = BatchingContext::<_, P::Projected>::with_policy(
        ProjectedContext::new(context.parent().clone()),
        context.axis_extent().clone(),
    )
    .with_axis_name(context.axis_name().map(str::to_string))
    .with_axis_sharding(context.axis_sharding().clone());
    let inputs = inputs.iter().map(P::project_batch).collect::<Result<Vec<_>, BatchingError>>()?;

    // `BatchingPolicyProjection` pins the member policy's evidence to the outer policy's, so the member rule's
    // evidence reaches the composite validation boundary exactly as the member rule stated it.
    let (outputs, evidence) = operation.batch(&projected_context, &EmptyRegionDriver, inputs.as_slice())?.into_parts();
    let outputs = outputs.iter().map(P::lift_batch).collect::<Result<Vec<_>, BatchingError>>()?;
    Ok(BatchedOutputs::new(outputs, evidence))
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayBatch, ArrayBatching, ArrayIrBatching, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation,
        ArrayType, DataType, Dimension, DimensionBounds, DimensionType, DimensionVariable, Shape, ShardingDimension,
        StaticArrayBatchingPolicy,
    };
    use crate::contexts::EagerContext;
    use crate::contexts::tests::{
        ProjectedMemberOperation, ProjectedMemberType, ProjectedMemberValue, ProjectedProgramOperation,
        ProjectedProgramType, ProjectedProgramValue,
    };
    use crate::operations::{AddOperation, NegOperation, Reduce, ReductionKind};
    use crate::parameters::Placeholder;
    use crate::programs::{ProgramBuilder, Typed};
    use crate::tracing::Trace;

    use super::*;

    /// Batch carrier shared by the composite and projected-member batching fixtures.
    #[derive(Clone, Debug, PartialEq)]
    struct ProjectedBatch<V: Value> {
        /// Packed value.
        value: V,

        /// Mapped axis.
        batch_axis: BatchAxis,
    }

    impl<V: Value> Display for ProjectedBatch<V> {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(formatter, "batch[{}, {}]", self.value, self.batch_axis)
        }
    }

    impl<V: Value> Parameter for ProjectedBatch<V> {}

    impl<V: Value> Typed for ProjectedBatch<V> {
        type Type = V::Type;

        fn r#type(&self) -> Cow<'_, Self::Type> {
            self.value.r#type()
        }
    }

    /// Batching policy proving that the generic frame does not encode the fixture's member kinds.
    #[derive(Copy, Clone, Debug)]
    struct ProjectedProgramBatching;

    type ProjectedProgramContext = EagerContext<ProjectedProgramValue, ProjectedProgramOperation>;

    impl<C> BatchingPolicy<C> for ProjectedProgramBatching
    where
        C: Context<
                Type = ProjectedProgramType,
                Value = ProjectedProgramValue,
                Constant = ProjectedProgramValue,
                Operation = ProjectedProgramOperation,
            >,
    {
        type Batch = ProjectedBatch<C::Value>;
        type Extent = usize;
        type Evidence = ();
        type BatchedProgram = BoundaryPreservingBatchedProgram<C::Constant, C::Operation>;

        fn batch(value: C::Value, batch_axis: BatchAxis) -> Result<Self::Batch, BatchingError> {
            if !batch_axis.is_replicated() && !matches!(value, ProjectedProgramValue::First(_)) {
                return Err(BatchingError::UnsupportedOperation {
                    message: format!("{} values must remain replicated under batching", value.r#type()),
                });
            }
            Ok(ProjectedBatch { value, batch_axis })
        }

        fn replicated(value: C::Value) -> Self::Batch {
            ProjectedBatch { value, batch_axis: BatchAxis::replicated() }
        }

        fn value(batch: &Self::Batch) -> &C::Value {
            &batch.value
        }

        fn batch_axis(batch: &Self::Batch) -> BatchAxis {
            batch.batch_axis
        }

        fn unbatched_type(batch: &Self::Batch) -> Cow<'_, C::Type> {
            batch.r#type()
        }

        fn adapt_batched_program<
            CollapseFn: Fn(
                &TracingContext<C::Constant, C::Operation>,
                Tracer<TracingContext<C::Constant, C::Operation>>,
                Axis,
            ) -> Result<Tracer<TracingContext<C::Constant, C::Operation>>, BatchingError>,
        >(
            program: Self::BatchedProgram,
            required_output_axes: Option<&[BatchAxis]>,
            collapse_fn: CollapseFn,
        ) -> Result<BoundaryPreservingBatchedProgram<C::Constant, C::Operation>, BatchingError> {
            let (program, output_axes) = program.into_parts();
            BoundaryPreservingBatchedProgram::from_widened_boundary(
                program,
                output_axes,
                required_output_axes,
                0,
                collapse_fn,
            )
        }
    }

    impl BatchingPolicyProjection<ProjectedProgramContext, ProjectedMemberType<2>> for ProjectedProgramBatching {
        type Projected = ProjectedMemberBatching<2>;
    }

    // The rule is context-generic so that the per-method `Self: Operation<Type = C::Type>` requirement of
    // `BatchableOperation::batch` is discharged from the context bound instead of by normalizing an eager
    // context that is itself built from this operation family.
    impl<C> BatchableOperation<C, ProjectedProgramBatching> for ProjectedProgramOperation
    where
        C: Context<
                Type = ProjectedProgramType,
                Value = ProjectedProgramValue,
                Constant = ProjectedProgramValue,
                Operation = ProjectedProgramOperation,
            >,
    {
        fn batch<D: BatchingDriver<C, ProjectedProgramBatching>>(
            &self,
            _context: &BatchingContext<C, ProjectedProgramBatching>,
            _driver: &D,
            inputs: &[ProjectedBatch<C::Value>],
        ) -> Result<BatchedOutputs<C, ProjectedProgramBatching>, BatchingError>
        where
            Self: Operation<Type = C::Type>,
        {
            if inputs.len() != 1 {
                return Err(ProgramError::InvalidInputCount { expected: 1, actual: inputs.len() }.into());
            }
            Ok(inputs.to_vec().into())
        }
    }

    /// Projected-member policy used to prove that projected batching does not depend on array carriers.
    #[derive(Copy, Clone, Debug)]
    struct ProjectedMemberBatching<const MEMBER: u8>;

    impl<const MEMBER: u8, C: Context<Type = ProjectedMemberType<MEMBER>>> BatchingPolicy<C>
        for ProjectedMemberBatching<MEMBER>
    {
        type Batch = ProjectedBatch<C::Value>;
        type Extent = usize;
        type Evidence = ();
        type BatchedProgram = BoundaryPreservingBatchedProgram<C::Constant, C::Operation>;

        fn batch(value: C::Value, batch_axis: BatchAxis) -> Result<Self::Batch, BatchingError> {
            Ok(ProjectedBatch { value, batch_axis })
        }

        fn replicated(value: C::Value) -> Self::Batch {
            ProjectedBatch { value, batch_axis: BatchAxis::replicated() }
        }

        fn value(batch: &Self::Batch) -> &C::Value {
            &batch.value
        }

        fn batch_axis(batch: &Self::Batch) -> BatchAxis {
            batch.batch_axis
        }

        fn unbatched_type(batch: &Self::Batch) -> Cow<'_, C::Type> {
            batch.r#type()
        }

        fn adapt_batched_program<
            CollapseFn: Fn(
                &TracingContext<C::Constant, C::Operation>,
                Tracer<TracingContext<C::Constant, C::Operation>>,
                Axis,
            ) -> Result<Tracer<TracingContext<C::Constant, C::Operation>>, BatchingError>,
        >(
            program: Self::BatchedProgram,
            required_output_axes: Option<&[BatchAxis]>,
            collapse_fn: CollapseFn,
        ) -> Result<BoundaryPreservingBatchedProgram<C::Constant, C::Operation>, BatchingError> {
            let (program, output_axes) = program.into_parts();
            BoundaryPreservingBatchedProgram::from_widened_boundary(
                program,
                output_axes,
                required_output_axes,
                0,
                collapse_fn,
            )
        }
    }

    impl<const MEMBER: u8, C: Context<Type = ProjectedMemberType<MEMBER>>>
        BatchableOperation<C, ProjectedMemberBatching<MEMBER>> for ProjectedMemberOperation<MEMBER>
    {
        fn batch<D: BatchingDriver<C, ProjectedMemberBatching<MEMBER>>>(
            &self,
            _context: &BatchingContext<C, ProjectedMemberBatching<MEMBER>>,
            _driver: &D,
            inputs: &[ProjectedBatch<C::Value>],
        ) -> Result<BatchedOutputs<C, ProjectedMemberBatching<MEMBER>>, BatchingError> {
            check_count!("input", inputs, 1, ProgramError);
            Ok(inputs.to_vec().into())
        }
    }

    #[test]
    fn test_batching_error_conversions_normalize_round_trips() {
        // A batching error that crossed into the kernel as a custom payload converts back to itself, and a
        // `BatchingError::Program` converts back to the program error it carries, so round trips never nest.
        let batching = BatchingError::MismatchedBatchSizes { expected: 4, actual: 5 };
        let program = ProgramError::from(batching.clone());
        assert!(matches!(
            program.downcast_custom::<BatchingError>(),
            Some(BatchingError::MismatchedBatchSizes { expected: 4, actual: 5 }),
        ));
        assert_eq!(BatchingError::from(program), batching);

        let program = ProgramError::EscapedProgramBuilder;
        let batching = BatchingError::from(program.clone());
        assert_eq!(batching, BatchingError::Program(ProgramError::EscapedProgramBuilder));
        assert_eq!(ProgramError::from(batching), program);
    }

    #[test]
    fn test_batch_axis() {
        assert_eq!(BatchAxis::default(), BatchAxis::replicated());
        assert!(BatchAxis::replicated().is_replicated());
        assert!(!BatchAxis::new(2).is_replicated());
        assert_eq!(BatchAxis::replicated().axis(), None);
        assert_eq!(BatchAxis::new(2).axis(), Some(Axis::from(2)));
        assert_eq!(BatchAxis::from(None), BatchAxis::replicated());
        assert_eq!(BatchAxis::from(Some(3)), BatchAxis::new(3));
        assert_eq!(BatchAxis::from(3), BatchAxis::new(3));
        assert_ne!(BatchAxis::new(0), BatchAxis::new(1));
        assert_eq!(format!("{:?}", BatchAxis::new(1)), "BatchAxis(Some(Axis(1)))");
    }

    #[test]
    fn test_batch_axis_specification() {
        assert_eq!(BatchAxisSpecification::from(None::<usize>), BatchAxisSpecification::default());
        assert_eq!(BatchAxisSpecification::from(Some(4)), BatchAxisSpecification::with_extent(4));
        assert_eq!(BatchAxisSpecification::from(4), BatchAxisSpecification::with_extent(4));
        assert_eq!(BatchAxisSpecification::new(4, "i"), BatchAxisSpecification::new(4, "i").clone());
        assert_ne!(BatchAxisSpecification::with_extent(4), BatchAxisSpecification::with_extent(5));
        assert_ne!(BatchAxisSpecification::<usize>::named("i"), BatchAxisSpecification::<usize>::named("j"),);
        assert_ne!(BatchAxisSpecification::<usize>::named("i"), BatchAxisSpecification::default());
        assert_eq!(
            format!("{:?}", BatchAxisSpecification::<usize>::named("i")),
            "BatchAxisSpecification { extent: None, name: Some(\"i\") }",
        );
    }

    #[test]
    fn test_boundary_preserving_batched_program() {
        // Construction validates that the output axes cover exactly the program's outputs.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F32));
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![input], vec![Placeholder], vec![Placeholder]).unwrap();
        assert_eq!(
            BoundaryPreservingBatchedProgram::new(program, Vec::new()).map(|_| ()),
            Err(ProgramError::InvalidOutputCount { expected: 1, actual: 0 }),
        );

        // A well-formed result preserves its program and output axes through `into_parts`.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F32));
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![input], vec![Placeholder], vec![Placeholder]).unwrap();
        let (program, output_axes) =
            BoundaryPreservingBatchedProgram::new(program, vec![BatchAxis::replicated()]).unwrap().into_parts();
        assert_eq!(program.output_ids(), &[input]);
        assert_eq!(output_axes, vec![BatchAxis::replicated()]);
    }

    #[test]
    fn test_boundary_preserving_batched_program_from_widened_boundary() {
        let packed_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)]));
        let sum = |context: &TracingContext<Array, ArrayOperation<Array>>,
                   output: Tracer<TracingContext<Array, ArrayOperation<Array>>>,
                   axis: Axis| {
            let _ = context;
            Ok(output.reduce(&[axis.normalize(output.r#type().rank()).unwrap()], ReductionKind::Sum))
        };

        // Parts that need no adjustment (no widening outputs and no required axes) are rewrapped without a replay.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(packed_type.clone());
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![input], vec![Placeholder], vec![Placeholder]).unwrap();
        let rendered = program.to_string();
        let (program, output_axes) = BoundaryPreservingBatchedProgram::from_widened_boundary(
            program,
            vec![BatchAxis::new(0)],
            None,
            0,
            |_, _, _| unreachable!("rewrapping must not collapse any output"),
        )
        .unwrap()
        .into_parts();
        assert_eq!(program.to_string(), rendered);
        assert_eq!(output_axes, vec![BatchAxis::new(0)]);

        // Widening outputs are bookkeeping rather than source-region results, so the rebuilt boundary drops them
        // while the corresponding inputs stay (they ground the program's own computation).
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let widening = builder.add_input(ArrayType::scalar(DataType::F64));
        let source = builder.add_input(packed_type.clone());
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![widening, source], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        let (program, output_axes) = BoundaryPreservingBatchedProgram::from_widened_boundary(
            program,
            vec![BatchAxis::new(0)],
            None,
            1,
            |_, _, _| unreachable!("dropping a widening output must not collapse any source output"),
        )
        .unwrap()
        .into_parts();
        assert_eq!(program.input_count(), 2);
        assert_eq!(
            program.interpret(vec![Array::scalar(7.0), Array::vector(vec![2.0, 3.0])]),
            Ok(vec![Array::vector(vec![2.0, 3.0])]),
        );
        assert_eq!(output_axes, vec![BatchAxis::new(0)]);

        // A mapped output whose required axis is replicated is collapsed by `collapse_fn` and reported as replicated,
        // while a mapped output whose required axis is mapped passes through untouched.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(packed_type.clone());
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![input, input], vec![Placeholder], vec![Placeholder; 2])
            .unwrap();
        let (program, output_axes) = BoundaryPreservingBatchedProgram::from_widened_boundary(
            program,
            vec![BatchAxis::new(0), BatchAxis::new(0)],
            Some(&[BatchAxis::replicated(), BatchAxis::new(0)]),
            0,
            sum,
        )
        .unwrap()
        .into_parts();
        assert_eq!(
            program.interpret(vec![Array::vector(vec![2.0, 3.0])]),
            Ok(vec![Array::scalar(5.0), Array::vector(vec![2.0, 3.0])]),
        );
        assert_eq!(output_axes, vec![BatchAxis::replicated(), BatchAxis::new(0)]);

        // The required axes must cover exactly the non-widening outputs.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(packed_type);
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![input], vec![Placeholder], vec![Placeholder]).unwrap();
        assert!(matches!(
            BoundaryPreservingBatchedProgram::from_widened_boundary(
                program,
                vec![BatchAxis::new(0)],
                Some(&[BatchAxis::replicated(), BatchAxis::new(0)]),
                0,
                sum,
            )
            .map(|_| ()),
            Err(BatchingError::Program(ProgramError::InvalidOutputCount { expected: 2, actual: 1 })),
        ));
    }

    #[test]
    fn test_batching_policy_is_member_kind_agnostic() {
        let context = BatchingContext::<_, ProjectedProgramBatching>::with_policy(ProjectedProgramContext::new(), 5);
        let tracer = BatchingTracer::new(
            context.clone(),
            <ProjectedProgramBatching as BatchingPolicy<ProjectedProgramContext>>::replicated(
                ProjectedProgramValue::Third(ProjectedMemberValue::<2>(7)),
            ),
        );
        assert_eq!(tracer.r#type(), Cow::Owned(ProjectedProgramType::Third(ProjectedMemberType::<2>)));
        assert_eq!(tracer.batch_extent(), &5);

        let operation = ProjectedProgramOperation::from(ProjectedMemberOperation::<2>);
        let outputs = operation.batch(&context, &EmptyRegionDriver, &[tracer.into_batch()]).unwrap().into_parts().0;
        assert_eq!(
            outputs,
            vec![ProjectedBatch {
                value: ProjectedProgramValue::Third(ProjectedMemberValue::<2>(7)),
                batch_axis: BatchAxis::replicated(),
            }],
        );

        assert!(matches!(
            <ProjectedProgramBatching as BatchingPolicy<ProjectedProgramContext>>::batch(
                ProjectedProgramValue::Third(ProjectedMemberValue::<2>(7)),
                BatchAxis::new(0),
            ),
            Err(BatchingError::UnsupportedOperation { message })
                if message == "member_2 values must remain replicated under batching",
        ));
    }

    #[test]
    fn test_elementwise_operation_interpret_with_batch_axes_packages_outputs_and_validates_count() {
        // `interpret_with_batch_axes` interprets the operation on the unpacked input values and repackages each
        // output as an `ArrayBatch` carrying the requested output batch axis. Here two batched length-3 inputs are
        // added elementwise, yielding a single batched sum mapped on axis 0.
        let vector_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));
        let left = ArrayBatch::new(vector_type.clone(), Array::vector(vec![1.0, 2.0, 3.0]), Some(0)).unwrap();
        let right = ArrayBatch::new(vector_type.clone(), Array::vector(vec![10.0, 20.0, 30.0]), Some(0)).unwrap();
        let context = BatchingContext::<_, ArrayBatching>::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 3);
        let outputs = AddOperation::new()
            .interpret_with_batch_axes(&context, &[left, right], &[BatchAxis::new(0)])
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(*outputs[0].r#type(), vector_type);
        assert_eq!(outputs[0].value(), &Array::vector(vec![11.0, 22.0, 33.0]));

        // An `output_batch_axes` length that disagrees with the number of produced outputs is rejected.
        let left = ArrayBatch::new(vector_type.clone(), Array::vector(vec![1.0, 2.0, 3.0]), Some(0)).unwrap();
        let right = ArrayBatch::new(vector_type, Array::vector(vec![10.0, 20.0, 30.0]), Some(0)).unwrap();
        assert!(matches!(
            AddOperation::new().interpret_with_batch_axes(&context, &[left, right], &[]),
            Err(BatchingError::Program(_)),
        ));
    }

    #[test]
    fn test_batching_context() {
        let context = BatchingContext::<_, ArrayBatching>::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 4);
        assert!(context.parent().is_eager());
        assert_eq!(context.axis_extent(), &4);
        assert_eq!(context.axis_name(), None);
        assert_eq!(context.axis_sharding(), &ShardingDimension::Replicated);

        let context = context.with_axis_name("items".to_string()).with_axis_sharding(ShardingDimension::sharded(["x"]));
        assert_eq!(context.axis_name(), Some("items"));
        assert_eq!(context.axis_sharding(), &ShardingDimension::sharded(["x"]));
    }

    #[test]
    fn test_batching_context_align_batched_program_outputs() {
        let vector_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));
        let (_, source_program) =
            EagerContext::<Array, ArrayOperation<Array>>::trace(|inputs: Vec<_>| Ok(inputs), vec![vector_type])
                .unwrap();
        let input_axes = [BatchAxis::new(1)];
        let context = BatchingContext::<_, ArrayBatching>::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 2);

        // A program whose natural output axes already match the required axes is returned without asking the driver
        // to replay it. Using the empty driver pins that fast path because any replay through it would fail.
        let batched_program = source_program
            .entry_region_ref()
            .batched(2, ShardingDimension::Replicated, input_axes.as_slice(), ProgramBatchingOutputAxesPolicy::Natural)
            .unwrap();
        let unchanged_program = context
            .align_batched_program_outputs(
                &EmptyRegionDriver,
                source_program.entry_region_ref(),
                input_axes.as_slice(),
                batched_program,
                &[BatchAxis::new(1)],
            )
            .unwrap();
        let axis_one_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3), Dimension::Static(2)]));
        assert_eq!(unchanged_program.input_types(), std::slice::from_ref(&axis_one_type));
        assert_eq!(unchanged_program.output_types(), std::slice::from_ref(&axis_one_type));
        assert!(unchanged_program.instructions().is_empty());

        // Requiring axis 0 instead causes one aligned replay of the source region. The input retains its axis-1 packed
        // layout, while the live output is transposed to the requested axis-0 layout.
        let regions = vec![source_program];
        let driver = RecursiveBatchingDriver::new(&regions);
        let region = driver.region(0).unwrap();
        let batched_program = region
            .batched(2, ShardingDimension::Replicated, input_axes.as_slice(), ProgramBatchingOutputAxesPolicy::Natural)
            .unwrap();
        let aligned_program = context
            .align_batched_program_outputs(
                &driver,
                region,
                input_axes.as_slice(),
                batched_program,
                &[BatchAxis::new(0)],
            )
            .unwrap();
        let axis_zero_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        assert_eq!(aligned_program.input_types(), std::slice::from_ref(&axis_one_type));
        assert_eq!(aligned_program.output_types(), std::slice::from_ref(&axis_zero_type));
        assert_eq!(aligned_program.instructions().len(), 1);
        assert!(matches!(aligned_program.instructions()[0].operation(), ArrayOperation::Transpose(_)));
        assert_eq!(
            aligned_program.interpret(vec![Array::matrix(3, 2, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0])]),
            Ok(vec![Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])]),
        );
    }

    #[test]
    fn test_batching_context_align_and_adapt_batched_program_outputs() {
        type TraceContext = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

        let vector_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));
        let mut source_builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let source_input = source_builder.add_input(vector_type.clone().into());
        let source_program = source_builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![source_input],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        let trace = TraceContext::new();
        let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(8)).unwrap());
        let extent_type = DimensionType::new(batch.clone());
        let extent = trace.input(extent_type.clone().into());
        let context = BatchingContext::<_, ArrayIrBatching>::new(trace, extent);
        let input_axes = [BatchAxis::new(0)];
        let batched_program = <ArrayIrBatching as RecursiveBatchingPolicy<TraceContext>>::batch_program(
            &context,
            source_program.entry_region_ref(),
            input_axes.as_slice(),
            ProgramBatchingOutputAxesPolicy::Natural,
        )
        .unwrap();

        // The target axis already matches, so the empty driver proves that the fast path is used. Adaptation retains
        // the leading extent input that grounds the dynamic batch dimension but removes its bookkeeping output.
        let adapted_program = context
            .align_and_adapt_batched_program_outputs(
                &EmptyRegionDriver,
                source_program.entry_region_ref(),
                input_axes.as_slice(),
                batched_program,
                input_axes.as_slice(),
            )
            .unwrap();
        let packed_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(batch), Dimension::Static(3)]));
        assert_eq!(
            adapted_program.input_types(),
            &[ArrayIrType::Dimension(extent_type), ArrayIrType::Array(packed_type.clone())],
        );
        assert_eq!(adapted_program.output_types(), &[ArrayIrType::Array(packed_type)]);
        assert_eq!(adapted_program.output_ids(), &adapted_program.input_ids()[1..]);
        assert!(adapted_program.instructions().is_empty());
    }

    /// Batching policy pinning the [`BatchedOutputs`] evidence lifecycle. Its carrier, extent, and batched-program
    /// boundary are the ordinary array ones, but its validator accepts an operation only when that operation's own
    /// rule attested to a claim naming it. Comparing the claim's subject with the operation being validated is what
    /// makes both a missing claim and a claim leaking from a previous operation observable; it is not a per-operation
    /// branch in the conservation law itself.
    #[derive(Copy, Clone, Debug)]
    struct EvidenceBatching;

    impl<C: Context<Type = ArrayType>> BatchingPolicy<C> for EvidenceBatching {
        type Batch = ArrayBatch<C::Value>;
        type Extent = usize;
        type Evidence = Vec<&'static str>;
        type BatchedProgram = BoundaryPreservingBatchedProgram<C::Constant, C::Operation>;

        fn batch(value: C::Value, batch_axis: BatchAxis) -> Result<Self::Batch, BatchingError> {
            <StaticArrayBatchingPolicy as BatchingPolicy<C>>::batch(value, batch_axis)
        }

        fn replicated(value: C::Value) -> Self::Batch {
            <StaticArrayBatchingPolicy as BatchingPolicy<C>>::replicated(value)
        }

        fn value(batch: &Self::Batch) -> &C::Value {
            batch.value()
        }

        fn batch_axis(batch: &Self::Batch) -> BatchAxis {
            batch.batch_axis()
        }

        fn unbatched_type(batch: &Self::Batch) -> Cow<'_, C::Type> {
            Cow::Owned(batch.unbatched_type())
        }

        fn validate_operation_outputs(
            operation_name: &'static str,
            _inputs: &[Self::Batch],
            _outputs: &[Self::Batch],
            evidence: &Self::Evidence,
        ) -> Result<(), BatchingError> {
            if evidence.contains(&operation_name) {
                Ok(())
            } else {
                Err(BatchingError::UnsupportedOperation {
                    message: format!("operation '{operation_name}' supplied no batching evidence"),
                })
            }
        }

        fn adapt_batched_program<
            CollapseFn: Fn(
                &TracingContext<C::Constant, C::Operation>,
                Tracer<TracingContext<C::Constant, C::Operation>>,
                Axis,
            ) -> Result<Tracer<TracingContext<C::Constant, C::Operation>>, BatchingError>,
        >(
            program: Self::BatchedProgram,
            required_output_axes: Option<&[BatchAxis]>,
            collapse_fn: CollapseFn,
        ) -> Result<BoundaryPreservingBatchedProgram<C::Constant, C::Operation>, BatchingError> {
            <StaticArrayBatchingPolicy as BatchingPolicy<C>>::adapt_batched_program(
                program,
                required_output_axes,
                collapse_fn,
            )
        }
    }

    impl<C: Context<Type = ArrayType>> RecursiveBatchingPolicy<C> for EvidenceBatching {
        fn batch_region(
            _context: &BatchingContext<C, Self>,
            _region: RegionRef<'_, C::Constant, C::Operation>,
            _inputs: Vec<Self::Batch>,
        ) -> Result<Vec<Self::Batch>, BatchingError> {
            Err(BatchingError::UnsupportedOperation { message: "the evidence fixture has no regions".to_string() })
        }

        fn batch_program(
            _context: &BatchingContext<C, Self>,
            _region: RegionRef<'_, C::Constant, C::Operation>,
            _input_axes: &[BatchAxis],
            _output_axes_policy: ProgramBatchingOutputAxesPolicy,
        ) -> Result<Self::BatchedProgram, BatchingError> {
            Err(BatchingError::UnsupportedOperation { message: "the evidence fixture has no regions".to_string() })
        }
    }

    /// Identity rule attesting to `add` alone, so one context exercises both the attested and the silent transition.
    impl<C: Context<Type = ArrayType>> BatchableOperation<C, EvidenceBatching> for ArrayOperation<C::Value> {
        fn batch<D: BatchingDriver<C, EvidenceBatching>>(
            &self,
            _context: &BatchingContext<C, EvidenceBatching>,
            _driver: &D,
            inputs: &[ArrayBatch<C::Value>],
        ) -> Result<BatchedOutputs<C, EvidenceBatching>, BatchingError>
        where
            Self: Operation<Type = C::Type>,
        {
            let evidence = match self.name() {
                name @ "add" => vec![name],
                _ => Vec::new(),
            };
            Ok(BatchedOutputs::new(inputs.to_vec(), evidence))
        }
    }

    #[test]
    fn test_operation_evidence_reaches_validation_and_does_not_outlive_its_rule() {
        type Parent = EagerContext<Array, ArrayOperation<Array>>;
        let context = BatchingContext::<Parent, EvidenceBatching>::with_policy(Parent::new(), 2);
        let input = BatchingTracer::new(context.clone(), ArrayBatch::replicated(Array::scalar(1.0_f32)));

        // An attesting rule's evidence reaches the validation boundary, which accepts the transition.
        let outputs = context.bind(AddOperation::new(), Vec::new(), &[input.clone(), input.clone()]).unwrap();
        assert_eq!(outputs.len(), 2);

        // That evidence dies with its rule: the very next operation is validated against its own (absent) claim
        // rather than against the accepted one that immediately preceded it.
        let silent = context.bind(NegOperation::new(), Vec::new(), std::slice::from_ref(&input)).unwrap_err();
        assert_eq!(
            BatchingError::from(silent),
            BatchingError::UnsupportedOperation {
                message: "operation 'neg' supplied no batching evidence".to_string(),
            },
        );
    }

    #[test]
    fn test_batch_projected_operation() {
        // The third fixture member is intentionally unrelated to arrays. Successfully applying its identity batching
        // rule proves that the adapter depends only on the projection and policy contracts, while preserving the
        // composite policy's packed value and replicated-axis representation.
        let context = BatchingContext::<_, ProjectedProgramBatching>::with_policy(ProjectedProgramContext::new(), 5);
        let input = <ProjectedProgramBatching as BatchingPolicy<ProjectedProgramContext>>::replicated(
            ProjectedProgramValue::Third(ProjectedMemberValue::<2>(7)),
        );
        let (outputs, evidence) =
            batch_projected_operation(&context, &ProjectedMemberOperation::<2>, &[input]).unwrap().into_parts();
        assert_eq!(
            outputs,
            vec![ProjectedBatch {
                value: ProjectedProgramValue::Third(ProjectedMemberValue::<2>(7)),
                batch_axis: BatchAxis::replicated(),
            }],
        );
        assert_eq!(evidence, ());
    }
}
