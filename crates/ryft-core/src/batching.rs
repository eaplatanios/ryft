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
//! [`ArrayBatch`] pairs an underlying array value with its batch-axis position. Its alignment helpers can broadcast a
//! replicated value, move an existing axis, or align several operands to a common position. The wrapper's type is the
//! unbatched per-item type (the packed value retains its batch dimension).
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
use std::marker::PhantomData;
use std::rc::Rc;

use thiserror::Error;

use ryft_macros::Parameter;

use crate::axes::{Axis, AxisError};
use crate::broadcasting::Broadcastable;
use crate::contexts::{Context, Domain, EagerContext, ProjectedContext, StagingContext, ValueResolution};
use crate::interpretation::InterpretableOperation;
use crate::macros::{check_builders, check_count};
use crate::operations::ElementwiseOperation;
use crate::operations::manipulation::{LegacyBroadcast, LegacyBroadcastOperation, Transpose, TransposeOperation};
use crate::parameters::{Parameter, ParameterError, Parameterized, ParameterizedFamily, Placeholder};
use crate::programs::ProgramError;
use crate::programs::operations::{Operation, OperationProjection};
use crate::programs::programs::Program;
use crate::programs::regions::{
    BindingRegionDriver, EmptyRegionDriver, RegionDriver, RegionRef, RegionReplayMappings, ReplayRegionDriver,
};
use crate::programs::types::{Type, TypeError, Typed};
use crate::programs::values::{Value, ValueProjection};
use crate::sharding::{MeshAxisType, Sharding, ShardingDimension, ShardingError};
use crate::tracing::{Tracer, TracingContext};
use crate::types::{ArrayType, Dimension, DimensionType, Shape};

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
/// This is the batch axis carried by an [`ArrayBatch`] and, during the batching transform, by the [`Tracer`] metadata.
/// Carrying it on the value itself lets the per-operation batching rules route the mapped batch axis straight from the
/// value in hand.
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
/// the active [`BatchingPolicy`] (e.g., homogeneous arrays use a host `usize`, while composite array programs use a
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

/// Result of structurally batching a nested [`Region`](crate::Region) whose transformed program carries exactly the
/// source region's inputs and outputs. [`BatchingPolicy`]s whose transformed programs carry bookkeeping inputs or
/// outputs beyond the source region's own boundary select their own result type through
/// [`BatchingPolicy::BatchedProgram`] instead, so consumers statically acknowledge that widened boundary
/// through the result type's own API rather than through a shared accessor that could silently drop it.
pub struct BatchedProgram<V: Typed + Parameter, O> {
    /// Structurally transformed [`Program`].
    program: Program<V, O, Vec<V>, Vec<V>>,

    /// Mapped axes of the source [`Region`](crate::Region)'s outputs.
    output_axes: Vec<BatchAxis>,
}

impl<V: Value, O: Operation<Type = V::Type>> BatchedProgram<V, O> {
    /// Creates a new [`BatchedProgram`] that carries exactly the source [`Region`](crate::Region)'s inputs and outputs.
    #[inline]
    pub fn new(program: Program<V, O, Vec<V>, Vec<V>>, output_axes: Vec<BatchAxis>) -> Result<Self, ProgramError> {
        check_count!("output", output_axes, program.output_count(), ProgramError);
        Ok(Self { program, output_axes })
    }

    /// Creates a new plain source-boundary [`BatchedProgram`] from the parts of a structurally batched program
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
    /// become a valid [`BatchedProgram`] once the replay has dropped those outputs. The constructed result always
    /// satisfies the invariant, and its output axes report each collapsed output as replicated (collapsing is what
    /// made it so). When neither adjustment applies (no bookkeeping outputs and no required axes), the parts are
    /// rewrapped directly, without a replay.
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

    /// Returns the mapped axes of the source [`Region`](crate::Region)'s outputs.
    #[inline]
    pub fn output_axes(&self) -> &[BatchAxis] {
        self.output_axes.as_slice()
    }

    /// Consumes this [`BatchedProgram`] and returns its transformed underlying program and output axes.
    pub fn into_parts(self) -> (Program<V, O, Vec<V>, Vec<V>>, Vec<BatchAxis>) {
        (self.program, self.output_axes)
    }
}

/// Value with [`ArrayType`] type that represents a _packed_ batch of arrays. [`ArrayBatch`] is the batching
/// representation for Ryft's batching/vectorization transform over arrays. It pairs a packed array value with a
/// [`BatchAxis`] that marks which of its dimensions indexes the batch items. A value is either *batched* (i.e., its
/// packed type carries the batch dimension) or *replicated*, meaning that it is shared unchanged across every batch
/// item.
#[derive(Clone, Debug, PartialEq, Parameter)]
pub struct ArrayBatch<V> {
    /// Packed array type of `value`. When the value is batched this type includes the mapped batch dimension at
    /// `batch_axis`. The unbatched (i.e., per-item) [`ArrayType`] is recovered by removing that dimension and can be
    /// obtained using [`Self::unbatched_type`].
    r#type: ArrayType,

    /// Refer to the documentation of [`value`](Self::value) for more information.
    value: V,

    /// Refer to the documentation of [`batch_axis`](Self::batch_axis) for more information.
    batch_axis: BatchAxis,
}

impl<V: Value<Type = ArrayType>> ArrayBatch<V> {
    /// Creates a new [`ArrayBatch`].
    #[inline]
    pub fn new<A: Into<BatchAxis>>(r#type: ArrayType, value: V, batch_axis: A) -> Result<Self, BatchingError> {
        let (batch_axis, _) = r#type.normalize_batch_axis(batch_axis.into())?;
        Ok(Self { r#type, value, batch_axis })
    }

    /// Creates a new [`ArrayBatch`] that replicates the provided value across the batch.
    #[inline]
    pub fn replicated(value: V) -> Self {
        Self { r#type: value.r#type().into_owned(), value, batch_axis: BatchAxis::replicated() }
    }

    /// Returns the [`BatchAxis`] marking which dimension of [`value`](Self::value) indexes the batch items.
    #[inline]
    pub fn batch_axis(&self) -> BatchAxis {
        self.batch_axis
    }

    /// Returns the canonical nonnegative position of this [`ArrayBatch`]'s mapped [`BatchAxis`]. [`ArrayBatch::new`]
    /// normalizes signed declarations before storing them, and so internal batching rules can use this index directly.
    #[inline]
    pub fn batch_axis_position(&self) -> Option<usize> {
        self.batch_axis.axis().map(|axis| axis.normalize(self.r#type.rank()).unwrap())
    }

    /// Returns the packed array value.
    #[inline]
    pub fn value(&self) -> &V {
        &self.value
    }

    /// Consumes `self` and returns the packed array value.
    #[inline]
    pub fn into_value(self) -> V {
        self.value
    }

    /// Returns the batch size of this [`ArrayBatch`] (i.e., the number of items that are batched together),
    /// or `None` if it is replicated (i.e., shared as-is across the whole batch).
    #[inline]
    pub fn batch_size(&self) -> Result<Option<usize>, BatchingError> {
        let Some(axis) = self.batch_axis.axis() else {
            return Ok(None);
        };
        let size = self
            .r#type()
            .into_owned()
            .dimension(axis)
            .value()
            .ok_or_else(|| BatchingError::DynamicBatchAxis { r#type: Box::new(self.r#type.clone()), axis })?;
        Ok(Some(size))
    }

    /// Returns the [`ArrayType`] of each item in the batch (i.e., with the batch axis removed, if any).
    #[inline]
    pub fn unbatched_type(&self) -> ArrayType {
        self.r#type.unbatched_type(self.batch_axis).unwrap()
    }

    /// Computes and validates the common batch size across `inputs`, returning `None` when no input is batched.
    /// Returns [`BatchingError::MismatchedBatchSizes`] when two batched inputs disagree on their batch size and
    /// [`BatchingError::DynamicBatchAxis`] when any batched input's mapped axis has a non-static size.
    pub fn common_batch_size(inputs: &[Self]) -> Result<Option<usize>, BatchingError> {
        inputs.iter().try_fold(None, |common_size, input| match (common_size, input.batch_size()?) {
            (Some(common_size), Some(size)) if common_size != size => {
                Err(BatchingError::MismatchedBatchSizes { expected: common_size, actual: size })
            }
            (None, Some(size)) => Ok(Some(size)),
            (common_size, _) => Ok(common_size),
        })
    }

    /// Returns a copy of this _replicated_ [`ArrayBatch`] broadcast to gain a batch axis of size `axis_size` at `axis`.
    /// This is the analogue of JAX's `batching.broadcast` and is the canonical building block for mixed
    /// batched/replicated primitive rules (e.g., the batching rule of the `dot` operation) and for lifting replicated
    /// residuals during linearization. It returns an error if called on an already-batched value, since such callers
    /// are expected to dispatch the replicated case explicitly (the blanket elementwise [`BatchableOperation`]
    /// implementation instead broadcasts replicated inputs to the full common batched shape).
    ///
    /// # Parameters
    ///
    ///   - `axis`: Possibly-negative position of the inserted batch axis in the output, normalized against the
    ///     batched output rank (e.g., `-1` denotes the final output axis).
    ///   - `axis_size`: Dimension of the inserted batch axis.
    ///   - `axis_sharding`: Sharding placement assigned to the inserted batch axis when the value carries
    ///     sharding metadata.
    pub fn broadcast<A: Into<Axis>>(
        &self,
        axis: A,
        axis_size: usize,
        axis_sharding: ShardingDimension,
    ) -> Result<Self, BatchingError>
    where
        V: LegacyBroadcast,
    {
        if !self.batch_axis().is_replicated() {
            return Err(BatchingError::MisalignedBatchAxes {
                message: "'ArrayBatch::broadcast' expects a replicated operand but received a batched value"
                    .to_string(),
            });
        }

        // The insertion position is normalized against the batched output rank (i.e., the per-item rank plus the
        // inserted batch dimension).
        let axis = axis.into();
        let per_item_type = self.unbatched_type();
        let output_rank = per_item_type.rank() + 1;
        let position = axis
            .normalize(output_rank)
            .map_err(|_| BatchingError::BatchAxisOutOfBounds { r#type: Box::new(self.r#type.clone()), axis })?;

        let mut batched_type = per_item_type.with_inserted_dimension(position, Dimension::Static(axis_size))?;
        if let Some(sharding) = per_item_type.sharding() {
            batched_type.sharding = Some(
                sharding
                    .with_inserted_dimension(position, axis_sharding)
                    .map_err(|error| BatchingError::MisalignedBatchAxes { message: error.to_string() })?,
            );
        }

        let output_axes = (0..per_item_type.rank())
            .map(|dimension| if dimension < position { dimension } else { dimension + 1 })
            .collect::<Vec<_>>();

        let broadcasted = self.value().clone().legacy_broadcast(batched_type.clone(), output_axes.as_slice())?;
        ArrayBatch::new(batched_type, broadcasted, axis)
    }

    /// Returns a copy of this [`ArrayBatch`] with its mapped batch axis moved to `axis`, staging a transpose on the
    /// packed value via [`Transpose::move_axis`] to realign it. This is the *move* half of JAX's `matchaxis` (i.e., a
    /// `moveaxis` on the batch dimension). The full move-or-broadcast behavior lives on
    /// [`match_axis`](Self::match_axis). It brings inputs that map their batch axis at different positions onto a
    /// common axis before an elementwise operation is applied. A replicated value (i.e., one with no mapped axis),
    /// or one already mapped at `axis`, is returned unchanged.
    ///
    /// # Parameters
    ///
    ///   - `axis`: Possibly-negative position the mapped batch axis should occupy in the returned [`ArrayBatch`],
    ///     normalized against the value's unchanged rank (e.g., `-1` denotes the final axis).
    pub fn move_axis<A: Into<Axis>>(&self, axis: A) -> Result<Self, BatchingError>
    where
        V: Transpose,
    {
        let Some(current_axis) = self.batch_axis_position() else {
            return Ok(self.clone());
        };
        // The target is normalized against the value's unchanged rank.
        let axis = axis.into();
        let position = axis
            .normalize(self.r#type.rank())
            .map_err(|_| BatchingError::BatchAxisOutOfBounds { r#type: Box::new(self.r#type.clone()), axis })?;
        if current_axis == position {
            return Ok(self.clone());
        }
        let permuted_value = self.value().clone().move_axis(current_axis, position)?;
        let permuted_type = permuted_value.r#type().into_owned();
        ArrayBatch::new(permuted_type, permuted_value, axis)
    }

    /// Returns a copy of this [`ArrayBatch`] with a batch axis of size `axis_size` materialized at `axis`. An
    /// already-batched value has its mapped axis realigned to `axis` via [`Self::move_axis`], while a replicated
    /// value is broadcast to gain a batch axis there via [`Self::broadcast`]. This is the analogue of JAX's
    /// `batching.matchaxis`, used by rules whose inputs must agree on one packed batch axis (e.g., `pad`,
    /// `concatenate`, etc.).
    ///
    /// # Parameters
    ///
    ///   - `axis`: Possibly-negative position the batch axis should occupy in the output, normalized against the
    ///     batched output rank (e.g., `-1` denotes the final output axis).
    ///   - `axis_size`: Dimension of the batch axis.
    ///   - `axis_sharding`: Sharding placement assigned to the batch axis if a replicated value must be broadcast.
    #[inline]
    pub fn match_axis<A: Into<Axis>>(
        &self,
        axis: A,
        axis_size: usize,
        axis_sharding: ShardingDimension,
    ) -> Result<Self, BatchingError>
    where
        V: LegacyBroadcast + Transpose,
    {
        let axis = axis.into();
        if self.batch_axis().is_replicated() {
            self.broadcast(axis, axis_size, axis_sharding)
        } else {
            self.move_axis(axis)
        }
    }

    /// Returns a copy of this [`ArrayBatch`] aligned to the provided output `batch_axis`. A mapped value is moved to
    /// a different requested position, while a replicated value is broadcast across `axis_size` when the declaration
    /// requests a mapped result. This matches JAX's mapped-output instantiation. A mapped value cannot be collapsed
    /// into a replicated declaration without an explicit reduction and that direction returns
    /// [`BatchingError::MismatchedOutputAxes`]. Signed declarations are normalized against the resulting packed
    /// rank. [`Batch::batch`] uses this function to realize the caller's declared `output_batch_axes`.
    ///
    /// # Parameters
    ///
    ///   - `batch_axis`: Requested mapped axis, or a replicated declaration if the output must remain replicated.
    ///   - `axis_size`: Dimension of the batch axis.
    ///   - `axis_sharding`: Sharding placement assigned to the batch axis if a replicated output must be broadcast.
    #[inline]
    pub fn align_axis(
        &self,
        batch_axis: BatchAxis,
        axis_size: usize,
        axis_sharding: ShardingDimension,
    ) -> Result<Self, BatchingError>
    where
        V: LegacyBroadcast + Transpose,
    {
        // Signed declaration normalization is owned by the delegates: `move_axis` normalizes against the (unchanged)
        // packed rank and `broadcast` against the batched output rank gaining the batch dimension.
        match (self.batch_axis.axis(), batch_axis.axis()) {
            (None, None) => Ok(self.clone()),
            (Some(_), Some(axis)) => self.move_axis(axis),
            (None, Some(axis)) => self.broadcast(axis, axis_size, axis_sharding),
            (Some(_), None) => {
                Err(BatchingError::MismatchedOutputAxes { expected: batch_axis, actual: self.batch_axis() })
            }
        }
    }

    /// Returns the [`ShardingDimension`] to place on the batch axis that batching introduces in each output, derived
    /// from how the provided batched inputs shard their own batch axis. Every input that is actually mapped on a batch
    /// axis and carries a [`Sharding`] contributes the [`ShardingDimension`] of that axis. Replicated inputs and inputs
    /// without a sharding contribute nothing. Replicated and unconstrained mapped dimensions can be normalized to a
    /// concrete sharded placement contributed by another mapped input. Distinct concrete sharded placements remain
    /// ambiguous and return [`BatchingError::MisalignedBatchAxes`]. When nothing pins the axis the result is
    /// [`ShardingDimension::Replicated`], leaving the new batch dimension replicated. The dimensions are read from the
    /// inputs as given, before batching realigns or broadcasts them, so a replicated input that later gains a singleton
    /// batch axis does not spuriously disagree with a genuinely batched input.
    #[inline]
    pub fn sharding_for_inputs(inputs: &[Self]) -> Result<ShardingDimension, ProgramError> {
        batch_axis_sharding(inputs.iter().map(|input| (&input.r#type, input.batch_axis_position())))
    }
}

impl<V: Display> Display for ArrayBatch<V> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.batch_axis.axis() {
            Some(axis) => write!(formatter, "batch[{}, axis={axis}]({})", self.r#type, self.value),
            None => write!(formatter, "batch[{}, replicated]({})", self.r#type, self.value),
        }
    }
}

impl<V: Typed<Type = ArrayType>> Typed for ArrayBatch<V> {
    type Type = ArrayType;

    #[inline]
    fn r#type(&self) -> Cow<'_, ArrayType> {
        Cow::Borrowed(&self.r#type)
    }
}

impl<V: Value<Type = ArrayType>> Value for ArrayBatch<V> {
    type DispatchDomain = EagerContext<Self>;
    type ExecutionDomain = EagerContext<Self>;

    #[inline]
    fn dispatch_domain(&self) -> EagerContext<Self> {
        EagerContext::new()
    }

    #[inline]
    fn execution_domain(&self) -> EagerContext<Self> {
        EagerContext::new()
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

impl BatchableType for ArrayType {
    type Policy = ArrayBatching;
}

/// Transform-owned policy selecting the batch carrier and mapped-axis extent representation used by a
/// [`BatchingContext`]. One generic batching frame (i.e., [`BatchingContext`], [`BatchingTracer`], [`BatchingDriver`],
/// and the [`BatchableOperation`] rule contract) serves every [`Program`] universe, and this policy names the three
/// places where those universes genuinely differ:
///
///   - **The Batch Carrier (i.e., [`Self::Batch`]):** A batched value maintains a split between its unbatched per-item
///     type and the parent-owned packed value that stores the whole batch. How that split is represented is specific
///     to each value kind: an ordinary array gains a mapped axis in its packed shape (i.e., [`ArrayBatch`]), while a
///     first-class dimension has no mapped representation at all (as a per-item extent would be a ragged shape), and so
///     a composite carrier must keep dimension members replicated and reject mapped non-array members.
///   - **The Mapped-Axis Extent (i.e., [`Self::Extent`]):** Homogeneous array batching uses a static `usize`, while a
///     composite universe may carry an ordinary parent-owned first-class dimension value so that a dynamic batch extent
///     remains a Single Static Assignment (SSA) value flowing through operand edges rather than being treated as static
///     transform metadata.
///   - **The Structurally Batched Program (i.e., [`Self::BatchedProgram`]):** Homogeneous array programs preserve the
///     source region's boundary exactly and produce ordinary [`BatchedProgram`]s, while a composite universe threads
///     bookkeeping values such as its first-class mapped extent through standalone nested programs and selects
///     a result type whose API makes that widened boundary explicit. [`Self::boundary_operands`] and
///     [`Self::adapt_batched_program`] let consumers complete or shed that widening without knowing
///     which policy produced the program.
///
/// The policy is deliberately limited to carrier selection, construction, and access. Array-specific alignment and
/// broadcasting are represented as functions on [`ArrayBatch`] (a composite policy may project an array member into
/// that carrier to reuse an existing array rule), and recursion into nested regions is the separate
/// [`RecursiveBatchingPolicy`] capability so that a carrier can exist before its universe supports structural region
/// rewriting.
pub trait BatchingPolicy<C: Context>: Copy + Clone + Debug {
    /// Batch-carrying representation for values owned by `C`.
    type Batch: Clone + Debug + Display + Parameter;

    /// Representation of the mapped-axis extent.
    type Extent: Clone + Debug;

    /// Result of structurally batching a nested [`Program`], including any policy-owned bookkeeping widening of the
    /// program boundary. Refer to the documentation of [`Self::adapt_batched_program`] for how consumers shed that
    /// widening when they need an ordinary [`Region`](crate::Region) boundary.
    type BatchedProgram;

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
    ///   - Homogeneous array policies return no values (the default), because their mapped-axis extent is static
    ///     transform metadata and their batched programs carry exactly the source boundary.
    ///   - [`ArrayProgramBatching`](crate::backends::ArrayProgramBatching) returns its first-class mapped-extent
    ///     dimension value, because every dynamic batch dimension inserted into one of its batched programs references
    ///     the [`DimensionVariable`](crate::DimensionVariable) defined by that program's leading extent input.
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
    /// *input* is load-bearing for the program itself: it defines the
    /// [`DimensionVariable`](crate::DimensionVariable) that every inserted dynamic batch dimension's type references,
    /// and a sealed program's types must be grounded by a value in its own scope, so removing it would leave the
    /// program referencing an undefined identity. The leading *output* carries no information of its own: it merely
    /// relays the extent so an enclosing extent-threading operation can chain it through its own sealed regions
    /// (e.g., a batched while body must return the extent it consumed so the next iteration's boundary can be fed).
    /// A consumer that does not thread extents has no use for the relay, so adapting the program drops the forwarded
    /// output, producing `[extent, inputs...] ↦ [outputs...]`, while [`Self::boundary_operands`] supplies the value
    /// for the kept input (e.g., a batched linear call consumes the extent as one more leading residual).
    /// Homogeneous array policies adapt without any boundary change.
    ///
    /// When `required_output_axes` is provided, the adaptation additionally reconciles each remaining output with
    /// the batch axis its consumer requires (i.e., an output that is mapped while its required axis is replicated is
    /// passed to `reduce`, which must collapse it along the mapped axis). This is, for example, how a batched transpose
    /// program returns one shared cotangent for a replicated linear input (i.e., replicating a value across batch items
    /// is semantically a broadcast, the transpose of a broadcast is a summation, and so the per-item cotangents are
    /// summed).
    ///
    /// The result is always a plain source-boundary [`BatchedProgram`], so programs that need neither adjustment are
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
    ///     documentation of [`BatchedProgram::from_widened_boundary`] for more information.
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
    ) -> Result<BatchedProgram<C::Constant, C::Operation>, BatchingError>;
}

/// Selects the [`BatchingPolicy`] used when an outer policy `Self` projects one member type `T` from composite
/// [`Context`] `C`. A [`BatchingPolicy`] determines one context's batch carrier, mapped-extent representation, and
/// structurally batched program boundary. It cannot by itself determine how each member type of a composite context
/// should represent those concepts. For example, an array member needs an [`ArrayBatch`] carrier that may hold a mapped
/// axis, whereas a first-class dimension member must remain replicated because a different dimension per batch item
/// would require a ragged value model. Both projected policies must nevertheless preserve the outer policy's exact
/// mapped-extent representation. This type-indexed relation records that choice independently for every supported
/// `(C, T)` pair.
///
/// Note that this trait carries no runtime state. Implementing it only establishes the associated
/// [`Projected`](Self::Projected) policy used by [`batch_projected_operation`]. Unsupported member projections simply
/// omit an implementation, while composite backends with several independently batchable member kinds can select a
/// different policy for each kind.
pub trait BatchingPolicyProjection<C: Context, T: Type>: BatchingPolicy<C>
where
    ProjectedContext<C, T>: Context,
{
    /// [`BatchingPolicy`] used while applying the projected member operation's batching rule. Its extent representation
    /// must be identical to the outer policy's so projection never specializes or reconstructs the mapped extent.
    type Projected: BatchingPolicy<ProjectedContext<C, T>, Extent = Self::Extent>;
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

/// Source of an output dimension of an elementwise batching broadcast operation. The shared elementwise batching
/// algorithm owns all broadcast geometry. For every batched output axis, it identifies the source of that dimension
/// and passes it to [`ArrayBatchingPolicy::broadcast_input`]. [`StaticArrayBatchingPolicy`] ignores these sources
/// because its output metadata already carries every extent, while a dimension-valued policy materializes each source
/// as a first-class value without re-deriving any geometry.
#[derive(Clone, Debug)]
pub enum DimensionSource<V> {
    /// Dimension with the provided static extent.
    Static(usize),

    /// Dynamic per-item dimension readable from axis `axis` of the broadcast-compatible source value `source`,
    /// which is a clone of the corresponding operand's parent-owned value.
    Value {
        /// Parent-owned value whose axis carries this dimension.
        source: V,

        /// Axis of `source`'s packed shape that carries this dimension.
        axis: usize,
    },

    /// The mapped batch axis itself, provided by the transform's extent.
    BatchExtent,
}

/// Array-specific capability that lets one homogeneous-array batching rule set serve multiple extent representations.
/// Ordinary array batching and projected array batching inside a composite array/dimension program share everything
/// that makes an array rule what it is (e.g., the [`ArrayBatch`] carrier, the shared elementwise alignment algorithm,
/// and every per-operation [`BatchableOperation`] implementation). They differ in exactly two ways:
///
///   - **Extent Representation:** Ordinary batching knows its mapped-axis extent as one static host `usize`. Projected
///     batching's extent is an ordinary parent-owned first-class dimension value, so a dynamic batch extent remains a
///     Single Static Assignment (SSA) value flowing through operand edges rather than static transform metadata.
///   - **Replicated-Array Materialization:** Aligning a replicated array with mapped inputs requires broadcasting it
///     across the mapped axis. Ordinary batching can use broadcasting operation with static output metadata, while
///     projected batching must stage the mixed broadcast that consumes explicit dimension operands.
///
/// An [`ArrayBatchingPolicy`] is a type-level selector packaging those two differences, so each policy owns the
/// complete translation from a homogeneous rule's extent and broadcast requests to its universe's operations. Every
/// policy also implements [`BatchingPolicy`] with `Batch = ArrayBatch<C::Value>` and `BatchedProgram =
/// BatchedProgram<C::Constant, C::Operation>`: homogeneous rules bind structurally batched branch programs directly,
/// so an array-authority policy is direct-boundary by definition and the supertrait binding lets every rule rely on
/// that without restating it. The shared rules are then written once against the nominal [`ArrayBatching<P>`] family
/// rather than as a `P: ArrayBatchingPolicy` blanket, because Rust coherence cannot use the *absence* of a trait
/// implementation to prove such a blanket disjoint from the genuinely mixed composite-operation rules registered
/// for other policies.
///
/// Keeping this capability on the batching transform, rather than on [`ProjectedContext`], [`ArrayBatch`], or [`Type`],
/// means that neither the carrier nor the type contract needs to know anything about dynamic-shape state that only
/// batching needs.
pub trait ArrayBatchingPolicy<C: Context<Type = ArrayType>>:
    BatchingPolicy<C, Batch = ArrayBatch<C::Value>, BatchedProgram = BatchedProgram<C::Constant, C::Operation>>
{
    /// Returns the mapped-axis [`Dimension`] to insert when building a batched [`ArrayType`].
    /// [`StaticArrayBatchingPolicy`] derives an exact [`Dimension::Static`] from the context's mapped-axis extent,
    /// while a dimension-valued policy returns the possibly dynamic dimension described by its first-class extent
    /// value's type. Shape computations can therefore construct batched types without forcing the extent to
    /// be statically known.
    fn axis_dimension(context: &BatchingContext<C, ArrayBatching<Self>>) -> Result<Dimension, BatchingError>;

    /// Returns the statically known mapped-axis size. This is the exact-size projection of [`Self::axis_dimension`].
    /// It succeeds when that dimension is static and returns a [`BatchingError::UnsupportedOperation`] when the mapped
    /// extent is genuinely dynamic. Rules that only move or broadcast arrays must use [`Self::match_axis`] and
    /// [`Self::broadcast_input`] instead, so that they keep working with dynamic mapped extents.
    #[inline]
    fn axis_size(context: &BatchingContext<C, ArrayBatching<Self>>) -> Result<usize, BatchingError> {
        Self::axis_dimension(context)?.value().ok_or_else(|| BatchingError::UnsupportedOperation {
            message: "this batching rule requires a statically known mapped-axis extent".to_string(),
        })
    }

    /// Aligns `batch` so that its mapped axis sits at position `axis`. A mapped batch moves its existing axis. A
    /// replicated batch is materialized across the mapped extent by inserting the axis at `axis`.
    /// [`StaticArrayBatchingPolicy`] broadcasts using the context's exact extent, while a dimension-valued policy
    /// stages the mixed broadcast whose inserted axis is grounded by the transform's first-class extent value.
    fn match_axis(
        context: &BatchingContext<C, ArrayBatching<Self>>,
        batch: &ArrayBatch<C::Value>,
        axis: Axis,
    ) -> Result<ArrayBatch<C::Value>, BatchingError>;

    /// Materializes one input/operand at `r#type`, broadcasting its per-item dimensions to the common target
    /// shape and inserting the mapped batch axis. The shared elementwise algorithm computes the complete broadcast
    /// geometry, including one [`DimensionSource`] per batched output axis, and delegates only the materialization
    /// itself, which is the policy-specific step. [`StaticArrayBatchingPolicy`] broadcasts with static output metadata
    /// and ignores the sources, while a dimension-valued policy spends each source mechanically (i.e., exact constants
    /// for static dimensions, `dimension_size` reads of the provided source values for dynamic per-item dimensions,
    /// and the transform's extent value for the mapped axis itself).
    ///
    /// # Parameters
    ///
    ///   - `context`: Active [`BatchingContext`] for the transform level being applied.
    ///   - `input`: Input/operand batch to materialize.
    ///   - `r#type`: Complete batched type of the materialized result (i.e., the common per-item target type
    ///     with the mapped axis inserted at `batch_axis`).
    ///   - `output_axes`: Mapping from each of `input`'s axes to its output axis.
    ///   - `batch_axis`: Position of the mapped batch axis in `r#type`.
    ///   - `dimension_sources`: Source of each batched output dimension, in axis order.
    fn broadcast_input(
        context: &BatchingContext<C, ArrayBatching<Self>>,
        input: &ArrayBatch<C::Value>,
        r#type: ArrayType,
        output_axes: Vec<usize>,
        batch_axis: Axis,
        dimension_sources: Vec<DimensionSource<C::Value>>,
    ) -> Result<ArrayBatch<C::Value>, BatchingError>;
}

/// [`ArrayBatchingPolicy`] for ordinary homogeneous array batching with a static host extent. This is the default
/// policy of [`ArrayBatching`] and preserves the established public array batching behavior. The mapped-axis
/// extent is one `usize` fixed when the transform is constructed, [`Self::axis_dimension`] is always an exact static
/// dimension, and replicated arrays are materialized with a broadcasting operation using static output metadata.
/// [`Self::axis_size`] always succeeds, so every batching rule (including host-side item enumeration) is
/// available under this policy.
#[derive(Copy, Clone, Debug, Default)]
pub struct StaticArrayBatchingPolicy;

impl<C: Context<Type = ArrayType>> BatchingPolicy<C> for StaticArrayBatchingPolicy {
    type Batch = ArrayBatch<C::Value>;
    type Extent = usize;
    type BatchedProgram = BatchedProgram<C::Constant, C::Operation>;

    #[inline]
    fn batch(value: C::Value, batch_axis: BatchAxis) -> Result<Self::Batch, BatchingError> {
        ArrayBatch::new(value.r#type().into_owned(), value, batch_axis)
    }

    #[inline]
    fn replicated(value: C::Value) -> Self::Batch {
        ArrayBatch::replicated(value)
    }

    #[inline]
    fn value(batch: &Self::Batch) -> &C::Value {
        batch.value()
    }

    #[inline]
    fn batch_axis(batch: &Self::Batch) -> BatchAxis {
        batch.batch_axis()
    }

    #[inline]
    fn unbatched_type(batch: &Self::Batch) -> Cow<'_, C::Type> {
        Cow::Owned(batch.unbatched_type())
    }

    #[inline]
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
    ) -> Result<BatchedProgram<C::Constant, C::Operation>, BatchingError> {
        let (program, output_axes) = program.into_parts();
        BatchedProgram::from_widened_boundary(program, output_axes, required_output_axes, 0, collapse_fn)
    }
}

impl<C: Context<Type = ArrayType, Value: LegacyBroadcast + Transpose>> ArrayBatchingPolicy<C>
    for StaticArrayBatchingPolicy
{
    #[inline]
    fn axis_dimension(context: &BatchingContext<C, ArrayBatching<Self>>) -> Result<Dimension, BatchingError> {
        Ok(Dimension::Static(*context.axis_extent()))
    }

    #[inline]
    fn match_axis(
        context: &BatchingContext<C, ArrayBatching<Self>>,
        batch: &ArrayBatch<C::Value>,
        axis: Axis,
    ) -> Result<ArrayBatch<C::Value>, BatchingError> {
        batch.match_axis(axis, *context.axis_extent(), context.axis_sharding().clone())
    }

    #[inline]
    fn broadcast_input(
        _context: &BatchingContext<C, ArrayBatching<Self>>,
        input: &ArrayBatch<C::Value>,
        r#type: ArrayType,
        output_axes: Vec<usize>,
        batch_axis: Axis,
        _dimension_sources: Vec<DimensionSource<C::Value>>,
    ) -> Result<ArrayBatch<C::Value>, BatchingError> {
        let broadcasted = input.value().clone().legacy_broadcast(r#type.clone(), output_axes.as_slice())?;
        ArrayBatch::new(r#type, broadcasted, batch_axis)
    }
}

/// Homogeneous-array [`BatchingPolicy`] parameterized by its [`ArrayBatchingPolicy`]. The default
/// [`StaticArrayBatchingPolicy`] preserves the ordinary public array batching API. Composite programs use a private
/// dynamic policy whose extent is a parent-owned first-class dimension value. Keeping both policies under this
/// nominal policy family lets every homogeneous array operation share one generated dispatcher without making those
/// implementations overlap genuinely mixed composite-operation rules. The wrapper delegates its batch carrier and
/// extent representation to `P`. Array-specific alignment is supplied by `P`'s [`ArrayBatchingPolicy`] implementation.
pub struct ArrayBatching<P = StaticArrayBatchingPolicy>(PhantomData<fn() -> P>);

impl<P> Copy for ArrayBatching<P> {}

impl<P> Clone for ArrayBatching<P> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<P> Debug for ArrayBatching<P> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("ArrayBatching")
    }
}

impl<P> Default for ArrayBatching<P> {
    #[inline]
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<C: Context<Type = ArrayType>, P: BatchingPolicy<C, Batch = ArrayBatch<C::Value>>> BatchingPolicy<C>
    for ArrayBatching<P>
{
    type Batch = P::Batch;
    type Extent = P::Extent;
    type BatchedProgram = P::BatchedProgram;

    #[inline]
    fn batch(value: C::Value, batch_axis: BatchAxis) -> Result<Self::Batch, BatchingError> {
        P::batch(value, batch_axis)
    }

    #[inline]
    fn replicated(value: C::Value) -> Self::Batch {
        P::replicated(value)
    }

    #[inline]
    fn value(batch: &Self::Batch) -> &C::Value {
        P::value(batch)
    }

    #[inline]
    fn batch_axis(batch: &Self::Batch) -> BatchAxis {
        P::batch_axis(batch)
    }

    #[inline]
    fn unbatched_type(batch: &Self::Batch) -> Cow<'_, C::Type> {
        P::unbatched_type(batch)
    }

    #[inline]
    fn boundary_operands(axis_extent: &Self::Extent) -> Vec<C::Value> {
        P::boundary_operands(axis_extent)
    }

    #[inline]
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
    ) -> Result<BatchedProgram<C::Constant, C::Operation>, BatchingError> {
        P::adapt_batched_program(program, required_output_axes, collapse_fn)
    }
}

impl<C: Context<Type = ArrayType, Value: LegacyBroadcast + Transpose>> BatchingEntrypointPolicy<C> for ArrayBatching {
    fn prepare_inputs(
        context: &C,
        inputs: Vec<C::Value>,
        input_batch_axes: Vec<BatchAxis>,
        batch_axis: BatchAxisSpecification<Self::Extent>,
    ) -> Result<(BatchingContext<C, Self>, Vec<Self::Batch>), BatchingError> {
        // Validate before zipping so a malformed flat axis declaration cannot silently drop unmatched inputs or axes.
        if inputs.len() != input_batch_axes.len() {
            return Err(
                ProgramError::InvalidInputCount { expected: inputs.len(), actual: input_batch_axes.len() }.into()
            );
        }

        // With no input leaves there is no mapped dimension from which to infer the batch extent.
        // An explicit extent still permits a valid input-free batching transform.
        if inputs.is_empty() && batch_axis.extent().is_none() {
            return Err(BatchingError::EmptyBatch);
        }

        // Pair each parent-owned packed value with its declared axis. `ArrayBatch::new` normalizes signed axes,
        // validates them against the packed rank, and derives the unbatched per-item type.
        let inputs = inputs
            .into_iter()
            .zip(input_batch_axes)
            .map(|(input, input_batch_axis)| ArrayBatch::new(input.r#type().into_owned(), input, input_batch_axis))
            .collect::<Result<Vec<_>, _>>()?;

        // Reconcile the caller's optional extent with the common extent inferred from mapped inputs. Either source can
        // establish the extent, but when both are available they must agree exactly.
        let explicit_extent = batch_axis.extent().copied();
        let batch_extent = match (explicit_extent, ArrayBatch::common_batch_size(&inputs)?) {
            (Some(explicit_extent), Some(inferred_extent)) if explicit_extent != inferred_extent => {
                return Err(BatchingError::MismatchedBatchSizes { expected: explicit_extent, actual: inferred_extent });
            }
            (explicit_extent, inferred_extent) => {
                explicit_extent.or(inferred_extent).ok_or(BatchingError::EmptyBatch)?
            }
        };

        // Select one placement for the batch axis from the original mapped inputs before any normalization.
        // This prevents a subsequently broadcast replicated input from appearing to constrain the placement.
        let axis_sharding = ArrayBatch::sharding_for_inputs(inputs.as_slice())?;

        // The batching context carries the reconciled batch extent, optional dynamic-scope name, and common sharding
        // placement that every operation rule at this batching level observes.
        let batching_context = BatchingContext::new(context.clone(), batch_extent)
            .with_axis_name(batch_axis.name().map(String::from))
            .with_axis_sharding(axis_sharding);

        // Materialize the common batch-axis placement on each mapped input whose packed sharding differs.
        // Replicated inputs have no mapped axis and therefore pass through unchanged.
        let inputs = inputs
            .into_iter()
            .map(|batch| {
                let normalized_type = batch
                    .batch_axis_position()
                    .map(|position| {
                        normalized_batch_axis_type(&batch.r#type, position, batching_context.axis_sharding())
                    })
                    .transpose()?
                    .flatten();
                let batch = if let Some(r#type) = normalized_type {
                    // A rank-preserving broadcast with identity output axes changes only the requested packed
                    // sharding placement. Rewrapping the result retains the original batch-axis declaration.
                    let output_axes = (0..batch.r#type().rank()).collect::<Vec<_>>();
                    let value = batch.value.clone().legacy_broadcast(r#type.clone(), output_axes.as_slice())?;
                    ArrayBatch::new(r#type, value, batch.batch_axis)?
                } else {
                    batch
                };
                Ok(batch)
            })
            .collect::<Result<Vec<_>, BatchingError>>()?;

        // Return the configured transform context together with parent-owned inputs carrying normalized batch metadata.
        Ok((batching_context, inputs))
    }

    #[inline]
    fn materialize_output(
        context: &BatchingContext<C, Self>,
        output: Self::Batch,
        output_batch_axis: BatchAxis,
    ) -> Result<C::Value, BatchingError> {
        Ok(output
            .align_axis(output_batch_axis, *context.axis_extent(), context.axis_sharding().clone())?
            .into_value())
    }
}

impl<C: Context<Type = ArrayType>> RecursiveBatchingPolicy<C> for ArrayBatching
where
    C::Operation: BatchableOperation<C, ArrayBatching>
        + BatchableOperation<TracingContext<C::Constant, C::Operation>, ArrayBatching>
        + From<TransposeOperation>
        + From<LegacyBroadcastOperation>,
{
    #[inline]
    fn batch_region(
        context: &BatchingContext<C, ArrayBatching>,
        region: RegionRef<'_, C::Constant, C::Operation>,
        inputs: Vec<ArrayBatch<C::Value>>,
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        let region_mappings = RegionReplayMappings::new();
        region.interpret_with(
            inputs,
            |_, constant| Ok(ArrayBatch::replicated(context.parent().lift(constant.clone())?)),
            |instruction, instruction_inputs| {
                let regions = ReplayRegionDriver::new(region, instruction.regions(), &region_mappings)?;
                instruction.operation().batch(context, &RecursiveBatchingDriver::new(&regions), instruction_inputs)
            },
        )
    }

    #[inline]
    fn batch_program(
        context: &BatchingContext<C, ArrayBatching>,
        region: RegionRef<'_, C::Constant, C::Operation>,
        input_axes: &[BatchAxis],
        output_axes_policy: ProgramBatchingOutputAxesPolicy,
    ) -> Result<Self::BatchedProgram, BatchingError> {
        region.batched(*context.axis_extent(), context.axis_sharding().clone(), input_axes, output_axes_policy)
    }
}

impl ArrayType {
    /// Normalizes and validates the provided [`BatchAxis`] against this packed [`ArrayType`],
    /// returning its canonical [`BatchAxis`] and position.
    pub fn normalize_batch_axis(&self, batch_axis: BatchAxis) -> Result<(BatchAxis, Option<usize>), BatchingError> {
        // A possibly-negative mapped axis is normalized against this type's rank, following Python/JAX indexing:
        // valid axes lie in `[-rank, rank)`, with `-1` denoting the final axis.
        Ok(match batch_axis.axis() {
            Some(axis) => match axis.normalize(self.rank()) {
                Ok(position) => (BatchAxis::from_position(position), Some(position)),
                Err(_) => {
                    return Err(BatchingError::BatchAxisOutOfBounds { r#type: Box::new(self.clone()), axis });
                }
            },
            None => (BatchAxis::replicated(), None),
        })
    }

    /// Returns the unbatched per-item [`ArrayType`] obtained by removing `batch_axis` from this packed [`ArrayType`].
    /// A replicated axis leaves the type unchanged. Possibly-negative mapped axes are normalized against the packed
    /// rank. When the removed dimension carries sharding, only manual mesh axes remain visible as varying manual axes
    /// in the per-item type because all other placement belongs to the transform-owned batch dimension.
    #[inline]
    pub fn unbatched_type<A: Into<BatchAxis>>(&self, batch_axis: A) -> Result<Self, BatchingError> {
        self.unbatched_type_and_axis(batch_axis.into()).map(|(r#type, _)| r#type)
    }

    /// Returns the unbatched per-item [`ArrayType`] together with the normalized [`BatchAxis`]. This internal form lets
    /// composite batch carriers validate and retain canonical axis metadata without cloning or otherwise projecting
    /// their array payloads.
    pub(crate) fn unbatched_type_and_axis(&self, batch_axis: BatchAxis) -> Result<(Self, BatchAxis), BatchingError> {
        let (batch_axis, axis) = self.normalize_batch_axis(batch_axis)?;
        let Some(axis) = axis else {
            return Ok((self.clone(), batch_axis));
        };

        // This is a transform-level view, not an ordinary rank-changing array operation. In particular, explicit
        // placement on the transform-owned mapped dimension describes how the batch is distributed; it is not part
        // of each batch item's placement. `ArrayType::without_dimension` correctly rejects dropping such placement
        // information for regular array dimensions, and so it cannot be used for these batching-specific semantics.
        let mut dimensions = self.shape().dimensions().to_vec();
        dimensions.remove(axis);
        let sharding = self
            .sharding()
            .map(|sharding| -> Result<Sharding, ShardingError> {
                let mut sharding_dimensions = sharding.dimensions().to_vec();
                let removed_dimension = sharding_dimensions.remove(axis);
                let mut projected_sharding = sharding.with_dimensions(sharding_dimensions)?;

                // Manual axes remain semantically visible after their ranked dimension is removed because values may
                // still vary across those axes. All other mesh axes are intentionally omitted (they placed the batch
                // dimension itself, which is outside the unbatched per-item type).
                if let ShardingDimension::Sharded(axis_names) = removed_dimension {
                    projected_sharding.extend_varying_manual_axes(
                        axis_names
                            .into_iter()
                            .filter(|name| sharding.mesh().axis_type(name) == Some(MeshAxisType::Manual)),
                    )?;
                }

                Ok(projected_sharding)
            })
            .transpose()
            .map_err(|error| BatchingError::MisalignedBatchAxes { message: error.to_string() })?;

        Ok((
            Self {
                data_type: self.data_type(),
                shape: Shape::new(dimensions),
                layout: None,
                sharding,
                memory: self.memory(),
            },
            batch_axis,
        ))
    }
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
///   - A dispatcher at `BatchableOperation<C, ArrayBatching>`, generic over the parent [`Context`] `C`, that
///     forwards the active [`BatchingContext`] to every variant's own rule. One dispatcher covers eager and staging
///     parents alike, because the parent/active distinction lives in each rule's body rather than in dispatch.
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
    /// values in `inputs` and the returned outputs are owned by `context.parent()`.
    ///
    /// # Contract
    ///
    ///   - **Axis Alignment:** If two or more inputs carry a mapped axis (i.e., `batch_axis.is_some()`), elementwise
    ///     operations require them to agree on the axis position. When they disagree, this function returns
    ///     [`BatchingError::MisalignedBatchAxes`] with an error message that names the misaligned axes and suggests the
    ///     user repositions one of them with [`Transpose`] (i.e., the N-D axis permutation primitive) before invoking
    ///     the operation. Operations with explicit axis arguments (e.g., `Dot`, `Transpose`, `Reshape`, etc.) rewrite
    ///     those arguments to thread the mapped axis through correctly.
    ///   - **Output Axes:** For elementwise operations, the output [`ArrayBatch::batch_axis`] matches the common input
    ///     batch axis. For operations with explicit axis arguments, the output axis follows from the lifted axis
    ///     arguments.
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
    ) -> Result<Vec<P::Batch>, BatchingError>
    where
        Self: Operation<Type = C::Type>;
}

// Blanket `BatchableOperation` implementation for any `ElementwiseOperation`, so per-operation `BatchableOperation`
// implementations do not have to be written for elementwise primitives (e.g., `ZeroLike`, `OneLike`, `Add`, `Sub`,
// `Mul`, `Div`, `Neg`, `Sin`, `Cos`, `Select`, etc.). Operations with non-trivial axis arithmetic (e.g., `Dot`,
// `Transpose`, `Reshape`, etc.) keep their explicit implementations. Coherence is preserved because none of those
// types implement `ElementwiseOperation`. The rule follows JAX's `defbroadcasting` policy where every input is
// broadcast to the common batched shape before the operation is applied, so the value-level primitive only ever
// sees inputs that agree on shape. When no input is mapped there is no batch axis to thread, and so the inputs are
// interpreted as given and every output is replicated. Otherwise, mapped inputs retain the first mapped input's
// position when that position is valid for every mapped operand. For mixed ranks where it is not, they are temporarily
// realigned to leading axis `0`. Every input is then broadcast to the common per-item shape with the batch axis
// inserted there. Operands whose per-item shapes are not broadcast-compatible are left at their batch-axis-inserted
// shapes so the operation surfaces its own shape error.
impl<
    C: Context<Type = ArrayType, Value: Transpose>,
    O: ElementwiseOperation + InterpretableOperation<C>,
    M: ArrayBatchingPolicy<C>,
> BatchableOperation<C, ArrayBatching<M>> for O
{
    #[inline]
    fn batch<D: BatchingDriver<C, ArrayBatching<M>>>(
        &self,
        context: &BatchingContext<C, ArrayBatching<M>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        // No input carries the batch axis. Interpret the inputs as given and report every output replicated.
        // Any per-item shape broadcasting between replicated inputs is the operation's own concern.
        let Some(output_batch_axis_position) = inputs.iter().find_map(ArrayBatch::batch_axis_position) else {
            let packed_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
            let output_count = Operation::infer_output_types(self, packed_types.as_slice(), &[])?.len();
            return self.interpret_with_batch_axes(context, inputs, &vec![BatchAxis::replicated(); output_count]);
        };

        // Preserve the first mapped input's natural position when every mapped input can represent it. Otherwise, use
        // a leading internal axis, which is valid regardless of differences in unbatched per-item rank, and restore the
        // natural output position after interpretation.
        let batch_axis_position = if inputs
            .iter()
            .all(|input| input.batch_axis().is_replicated() || output_batch_axis_position < input.r#type().rank())
        {
            output_batch_axis_position
        } else {
            0
        };
        let batch_axis = Axis::from(batch_axis_position);

        // Preserve placement removed with each mapped dimension, then align every mapped input onto the common
        // batch-axis position before broadcasting per-item shapes.
        let axis_sharding = ArrayBatch::sharding_for_inputs(inputs)?;
        let unbatched_types = inputs
            .iter()
            .map(|input| -> Result<_, BatchingError> {
                let mut unbatched_type = input.unbatched_type();
                if let (Some(sharding), ShardingDimension::Sharded(axis_names)) =
                    (unbatched_type.sharding.as_mut(), &axis_sharding)
                {
                    let varying_manual_axes = axis_names
                        .iter()
                        .filter(|name| sharding.mesh().axis_type(name.as_str()) == Some(MeshAxisType::Manual))
                        .cloned()
                        .collect::<Vec<_>>();
                    sharding
                        .extend_varying_manual_axes(varying_manual_axes)
                        .map_err(|error| BatchingError::MisalignedBatchAxes { message: error.to_string() })?;
                }
                Ok(unbatched_type)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let inputs = inputs.iter().map(|input| input.move_axis(batch_axis)).collect::<Result<Vec<_>, _>>()?;

        // Broadcast every operand to the common unbatched per-item shape when one exists. Otherwise, retain
        // each input's own per-item shape and let the operation's inference report the incompatibility.
        let common_unbatched_type = Broadcastable::broadcasted(unbatched_types.as_slice()).ok();
        let axis_dimension = M::axis_dimension(context)?;
        let broadcasted_inputs = inputs
            .iter()
            .zip(unbatched_types.iter())
            .map(|(input, unbatched_type)| -> Result<ArrayBatch<C::Value>, BatchingError> {
                let mut target_type = common_unbatched_type.as_ref().unwrap_or(unbatched_type).clone();
                target_type.data_type = unbatched_type.data_type();
                let mut batched_type =
                    target_type.with_inserted_dimension(batch_axis_position, axis_dimension.clone())?;
                if let Some(sharding) = target_type.sharding() {
                    batched_type.sharding = Some(
                        sharding
                            .with_inserted_dimension(batch_axis_position, axis_sharding.clone())
                            .map_err(|error| BatchingError::MisalignedBatchAxes { message: error.to_string() })?,
                    );
                }
                if batched_type == *input.r#type() {
                    return Ok(input.clone());
                }

                // Right-align this input's per-item dimensions in the target while preserving the mapped dimension
                // at the common batch-axis position.
                let target_unbatched_rank = batched_type.rank() - 1;
                let is_mapped = !input.batch_axis().is_replicated();
                let output_axes = (0..input.r#type().rank())
                    .map(|dimension| {
                        if is_mapped && dimension == batch_axis_position {
                            return batch_axis_position;
                        }
                        let per_item_index =
                            if is_mapped && dimension > batch_axis_position { dimension - 1 } else { dimension };
                        let position = (target_unbatched_rank - unbatched_type.rank()) + per_item_index;
                        if position < batch_axis_position { position } else { position + 1 }
                    })
                    .collect();

                // Resolve the source of every batched output dimension. The mapped axis comes from the transform's
                // extent, static target dimensions carry their extents directly, and each dynamic target dimension is
                // read from the broadcast-compatible source axis that supplied it (which exists by construction of the
                // broadcasted target shape).
                let dimension_sources = batched_type
                    .shape()
                    .dimensions()
                    .iter()
                    .enumerate()
                    .map(|(axis, _)| -> Result<DimensionSource<C::Value>, BatchingError> {
                        if axis == batch_axis_position {
                            return Ok(DimensionSource::BatchExtent);
                        }
                        let target_axis = if axis < batch_axis_position { axis } else { axis - 1 };
                        let target_dimension = &target_type.shape().dimensions()[target_axis];
                        if let Dimension::Static(extent) = target_dimension {
                            return Ok(DimensionSource::Static(*extent));
                        }
                        inputs
                            .iter()
                            .zip(unbatched_types.iter())
                            .find_map(|(source, source_type)| {
                                let rank_offset = target_type.rank() - source_type.rank();
                                let source_axis = target_axis.checked_sub(rank_offset)?;
                                if source_axis >= source_type.rank()
                                    || source_type.shape().dimensions()[source_axis] != *target_dimension
                                {
                                    return None;
                                }
                                let packed_axis =
                                    if source.batch_axis().is_replicated() || source_axis < batch_axis_position {
                                        source_axis
                                    } else {
                                        source_axis + 1
                                    };
                                Some(DimensionSource::Value { source: source.value().clone(), axis: packed_axis })
                            })
                            .ok_or_else(|| {
                                TypeError::invalid(format!(
                                    "cannot locate a source for elementwise output dimension \
                                     {target_dimension}",
                                ))
                                .into()
                            })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                M::broadcast_input(context, input, batched_type, output_axes, batch_axis, dimension_sources)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let input_types = broadcasted_inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        let output_count = Operation::infer_output_types(self, input_types.as_slice(), &[])?.len();
        let output_batch_axes = vec![BatchAxis::new(batch_axis); output_count];
        self.interpret_with_batch_axes(context, &broadcasted_inputs, &output_batch_axes)?
            .into_iter()
            .map(|output| output.move_axis(output_batch_axis_position))
            .collect()
    }
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
        let batching_driver = RecursiveBatchingDriver::new(&driver);
        let output_batches = operation.batch(self, &batching_driver, input_batches.as_slice())?;
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

impl<
    V: Value<Type = ArrayType>,
    O: Operation<Type = ArrayType>
        + BatchableOperation<TracingContext<V, O>, ArrayBatching>
        + From<TransposeOperation>
        + From<LegacyBroadcastOperation>,
> RegionRef<'_, V, O>
{
    /// Structurally batches this borrowed homogeneous-array [`Region`](crate::Region) so that the resulting program
    /// operates over inputs batched along the specified [`BatchAxis`]s. Staged higher-order [`BatchableOperation`]
    /// implementations use this function to batch captured programs *without* concretizing any batch-item values, so
    /// that batched control-flow and custom-derivative structure can be staged back into the enclosing trace. This
    /// function replays the region through an [`ArrayBatching`] [`BatchingContext`] over a fresh
    /// [`TracingContext`], lifts every instruction through its [`BatchableOperation`] rule, and extracts the resulting
    /// staged program together with the requested [`ProgramBatchingOutputAxesPolicy`].
    ///
    /// This method is intentionally specific to [`ArrayType`]. It constructs mapped batched input types by inserting
    /// static [`Dimension`]s, rewrites array sharding metadata, and materializes output axes through [`ArrayBatch`].
    /// Composite program families with first-class dynamic dimension extent values instead own structural recursion
    /// through [`RecursiveBatchingPolicy::batch_program`], where the [`RecursiveBatchingPolicy`] can define how extent
    /// Single Static Assignment (SSA) values cross a rewritten region boundary.
    ///
    /// Inputs whose `input_batch_axes[i]` is mapped at position `k` consume the original unbatched input [`ArrayType`]
    /// with a mapped batch axis of size `axis_size` inserted at `k`, while replicated inputs enter at their original
    /// unbatched types. [`ProgramBatchingOutputAxesPolicy::Natural`] keeps the mapped axes produced by the
    /// [`BatchableOperation`] implementations (e.g., during the discovery pass of staged control-flow batching).
    /// [`ProgramBatchingOutputAxesPolicy::AlignEachTo`] instantiates each output at a requested axis while the outputs
    /// are still live tracers, which is how staged `condition` branches agree on one output layout and the staged
    /// `while` fix-point keeps body outputs on the loop-invariant state axes.
    /// [`ProgramBatchingOutputAxesPolicy::AlignAllTo`] imposes one canonical output axis, which is what custom
    /// derivative re-wrapping needs so that independently batched primal/JVP/forward/backward programs have mutually
    /// consistent signatures.
    ///
    /// # Parameters
    ///
    ///   - `axis_size`: Dimension of the new batch axis.
    ///   - `axis_sharding`: Sharding placement assigned to every newly materialized batch axis.
    ///   - `input_batch_axes`: [`BatchAxis`] for each input (i.e., argument) of this [`Program`].
    ///   - `output_axes_policy`: [`ProgramBatchingOutputAxesPolicy`] for packaging the batched program outputs.
    pub fn batched(
        &self,
        axis_size: usize,
        axis_sharding: ShardingDimension,
        input_batch_axes: &[BatchAxis],
        output_axes_policy: ProgramBatchingOutputAxesPolicy,
    ) -> Result<BatchedProgram<V, O>, BatchingError> {
        let input_count = self.input_ids().len();
        check_count!("input", input_batch_axes, input_count, ProgramError);

        let parent_context = TracingContext::<V, O>::new();
        let builder = parent_context.builder().clone();

        // Keep every tracer and context that holds a clone of `builder` inside the following scope so that recovering
        // the builder later on (below) is a real ownership check. In particular, `context` (which owns a clone of the
        // builder through its parent trace) must be created *inside* this scope so it is dropped before the builder is
        // recovered below; leaving it in the enclosing scope leaks a builder clone past the recovery.
        let (output_atom_ids, output_axes) = {
            let batching_context =
                BatchingContext::new(parent_context, axis_size).with_axis_sharding(axis_sharding.clone());
            let inputs = self
                .input_types()
                .iter()
                .zip(input_batch_axes.iter())
                .map(|(unbatched_type, batch_axis)| {
                    let batched_type = match batch_axis.axis() {
                        Some(axis) => {
                            // A possibly-negative mapped axis is normalized against the packed input rank (i.e., with
                            // the inserted batch dimension counted). Valid axes lie in `[-rank, rank)`, with `-1`
                            // denoting the final axis.
                            let batched_rank = unbatched_type.rank() + 1;
                            let position = axis.normalize(batched_rank).map_err(|_| {
                                BatchingError::BatchAxisOutOfBounds { r#type: Box::new(unbatched_type.clone()), axis }
                            })?;
                            let mut batched_type =
                                unbatched_type.with_inserted_dimension(position, Dimension::Static(axis_size))?;
                            if let Some(sharding) = unbatched_type.sharding() {
                                batched_type.sharding =
                                    Some(sharding.with_inserted_dimension(position, axis_sharding.clone()).map_err(
                                        |error| BatchingError::MisalignedBatchAxes { message: error.to_string() },
                                    )?);
                            }
                            batched_type
                        }
                        None => unbatched_type.clone(),
                    };
                    let input = builder.borrow_mut().add_input(batched_type.clone());
                    let value = batching_context.parent().tracer(input, Some(batched_type.clone()));
                    Ok(ArrayBatch::new(batched_type, value, *batch_axis)?)
                })
                .collect::<Result<Vec<_>, ProgramError>>()?;

            // Replay this program by binding each instruction's `BatchableOperation` rule against the batching context,
            // threading the batch-carrying inputs through. Constants lift in the parent trace and replicate across the
            // batch. This only requires this program's own operation family to be batchable, so staged higher-order
            // batching rules can batch a captured sub-program without concretizing any batch-item values.
            let region_mappings = RegionReplayMappings::new();
            let outputs = self.interpret_with(
                inputs,
                |_, constant| Ok(ArrayBatch::replicated(batching_context.parent().lift(constant.clone())?)),
                |instruction, instruction_inputs| {
                    let regions = ReplayRegionDriver::new(*self, instruction.regions(), &region_mappings)?;
                    instruction.operation().batch(
                        &batching_context,
                        &RecursiveBatchingDriver::new(&regions),
                        instruction_inputs,
                    )
                },
            )?;

            // Resolve `output_axes_policy` into one optional alignment declaration per output. The outer `None` keeps
            // the natural axis, while `Some(mapped)` forces the output to carry its batch axis at that signed position.
            // A replicated `AlignEachTo` entry is a lower bound rather than an equality constraint, mirroring JAX's
            // `instantiate` behavior.
            let output_target_batch_axes = match &output_axes_policy {
                ProgramBatchingOutputAxesPolicy::Natural => vec![None; outputs.len()],
                ProgramBatchingOutputAxesPolicy::AlignAllTo(axis) => vec![Some(BatchAxis::new(*axis)); outputs.len()],
                ProgramBatchingOutputAxesPolicy::AlignEachTo(axes) => {
                    check_count!("output", outputs, axes.len(), ProgramError);
                    axes.iter().map(|target| (!target.is_replicated()).then_some(*target)).collect::<Vec<_>>()
                }
            };
            let mut output_atom_ids = Vec::with_capacity(outputs.len());
            let mut output_axes = Vec::with_capacity(outputs.len());
            for (output, output_target_batch_axis) in outputs.into_iter().zip(output_target_batch_axes) {
                // The batched outputs must belong to this batched trace. A foreign tracer's atom ID would silently
                // alias whichever atom shares its index in this builder, and so we perform a check here to avoid that.
                check_builders!(&builder, output.value().builder())?;

                // Untargeted outputs keep the natural axis produced by their batching rules.
                let Some(target_batch_axis) = output_target_batch_axis else {
                    output_atom_ids.push(output.value().atom_id()?);
                    output_axes.push(output.batch_axis());
                    continue;
                };

                // Move naturally mapped outputs or broadcast naturally replicated outputs while they are still live
                // tracers, then report the normalized position stored by `ArrayBatch`.
                let output =
                    output.align_axis(target_batch_axis, axis_size, batching_context.axis_sharding().clone())?;
                output_axes.push(output.batch_axis());
                output_atom_ids.push(output.into_value().atom_id()?);
            }

            Ok::<_, ProgramError>((output_atom_ids, output_axes))
        }?;

        let output_count = output_atom_ids.len();
        let builder = Rc::try_unwrap(builder).map_err(|_| ProgramError::EscapedProgramBuilder)?.into_inner();
        let program = builder
            .build(output_atom_ids, vec![Placeholder; input_count], vec![Placeholder; output_count])?
            .into_simplified()?;
        Ok(BatchedProgram::new(program, output_axes)?)
    }
}

impl<
    V: Value<Type = ArrayType>,
    O: Operation<Type = ArrayType>
        + BatchableOperation<TracingContext<V, O>, ArrayBatching>
        + From<TransposeOperation>
        + From<LegacyBroadcastOperation>,
> Program<V, O, Vec<V>, Vec<V>>
{
    /// Structurally batches this homogeneous-array [`Program`] over the provided input axes. Refer to
    /// [`RegionRef::batched`] for the complete transformation semantics and for the reason that this helper
    /// is intentionally specific to [`ArrayType`].
    ///
    /// # Parameters
    ///
    ///   - `axis_size`: Dimension of the new batch axis.
    ///   - `axis_sharding`: Sharding placement assigned to every newly materialized batch axis.
    ///   - `input_batch_axes`: [`BatchAxis`] for each program input.
    ///   - `output_axes_policy`: Policy controlling how the transformed program packages its output batch axes.
    #[inline]
    pub fn batched(
        &self,
        axis_size: usize,
        axis_sharding: ShardingDimension,
        input_batch_axes: &[BatchAxis],
        output_axes_policy: ProgramBatchingOutputAxesPolicy,
    ) -> Result<BatchedProgram<V, O>, BatchingError> {
        self.entry_region_ref().batched(axis_size, axis_sharding, input_batch_axes, output_axes_policy)
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

/// Derives the common [`ShardingDimension`] of mapped axes from packed array types and normalized axis positions.
/// This representation-neutral helper lets homogeneous and composite batching share the same placement join without
/// projecting or cloning array payloads.
pub(crate) fn batch_axis_sharding<T: std::borrow::Borrow<ArrayType>, I: IntoIterator<Item = (T, Option<usize>)>>(
    inputs: I,
) -> Result<ShardingDimension, ProgramError> {
    let (_, dimension) = inputs
        .into_iter()
        .filter_map(|(r#type, position)| {
            // Only mapped inputs with explicit sharding metadata constrain the new batch axis. Clone just the cheap
            // mesh handle and the mapped dimension placement so the fold does not borrow from an iterator-owned type.
            let position = position?;
            let r#type = r#type.borrow();
            let sharding = r#type.sharding()?;
            Some((sharding.mesh().clone(), sharding.dimensions()[position].clone()))
        })
        .try_fold((None, None), |(mesh, dimension), (current_mesh, current_dimension)| -> Result<_, ProgramError> {
            // Carry the common mesh and the joined mapped-axis placement together so this remains a single-pass operation
            // over potentially one-shot iterators.

            // `ShardingDimension` identifies mesh axes by name, so placements are comparable only when every
            // contributing sharding belongs to the same logical mesh.
            let mesh = match mesh {
                Some(mesh) if mesh != current_mesh => {
                    return Err(BatchingError::MisalignedBatchAxes {
                        message: format!("mismatched batch axis sharding meshes: {mesh:?} vs {current_mesh:?}"),
                    }
                    .into());
                }
                Some(mesh) => Some(mesh),
                None => Some(current_mesh),
            };

            // Join placements from least to most specific. Unconstrained and replicated placements yield to a concrete
            // sharded placement, equal placements agree, and distinct concrete shardings are ambiguous. These rules
            // make the result independent of input order.
            let dimension = match dimension {
                Some(folded_dimension) if folded_dimension == current_dimension => Some(folded_dimension),
                Some(ShardingDimension::Unconstrained) => Some(current_dimension),
                Some(folded_dimension) if current_dimension == ShardingDimension::Unconstrained => {
                    Some(folded_dimension)
                }
                Some(ShardingDimension::Replicated) => Some(current_dimension),
                Some(folded_dimension) if current_dimension == ShardingDimension::Replicated => Some(folded_dimension),
                Some(folded_dimension) => {
                    return Err(BatchingError::MisalignedBatchAxes {
                        message: format!("mismatched batch axis sharding: {folded_dimension} vs {current_dimension}"),
                    }
                    .into());
                }
                None => Some(current_dimension),
            };
            Ok((mesh, dimension))
        })?;

    // With no mapped, explicitly sharded contributor, introducing a replicated batch axis is the neutral choice.
    Ok(dimension.unwrap_or(ShardingDimension::Replicated))
}

/// Returns the [`ArrayType`] required to place `position` on `axis_sharding`,
/// or `None` when no normalization is needed.
pub(crate) fn normalized_batch_axis_type(
    r#type: &ArrayType,
    position: usize,
    axis_sharding: &ShardingDimension,
) -> Result<Option<ArrayType>, BatchingError> {
    // An unsharded type has no placement metadata to normalize.
    let Some(sharding) = r#type.sharding() else {
        return Ok(None);
    };

    // Preserve the original type when the mapped axis already has the common placement selected for this batch.
    if sharding.dimensions().get(position) == Some(axis_sharding) {
        return Ok(None);
    }

    // The batch carrier has already normalized and validated `position` against the array's rank. Replace only
    // that dimension's placement, leaving every non-batch dimension unchanged.
    let mut dimensions = sharding.dimensions().to_vec();
    dimensions[position] = axis_sharding.clone();

    // A value sharded along a manual mesh axis varies across that axis inside a manual region. Preserve the variation
    // facts already known for the input and add any manual axes introduced by the replacement placement.
    let mut varying_manual_axes = sharding.varying_manual_axes().clone();
    if let ShardingDimension::Sharded(axis_names) = axis_sharding {
        varying_manual_axes.extend(
            axis_names
                .iter()
                .filter(|name| sharding.mesh().axis_type(name) == Some(MeshAxisType::Manual))
                .cloned(),
        );
    }

    // `Sharding::new` validates the new per-dimension placement and starts with empty auxiliary axis state. Reapply the
    // input's reduction state and the updated manual-variation state to construct the complete normalized sharding.
    let sharding = Sharding::new(sharding.mesh().clone(), dimensions)
        .and_then(|normalized| normalized.with_unreduced_axes(sharding.unreduced_axes().clone()))
        .and_then(|normalized| normalized.with_reduced_axes(sharding.reduced_axes().clone()))
        .and_then(|normalized| normalized.with_varying_manual_axes(varying_manual_axes))
        .map_err(|error| BatchingError::MisalignedBatchAxes { message: error.to_string() })?;

    // Install the validated sharding on a cloned packed type. Returning `Some` tells the caller that it must
    // materialize this placement change rather than reuse the input value unchanged.
    r#type
        .clone()
        .with_sharding(sharding)
        .map(Some)
        .map_err(|error| BatchingError::MisalignedBatchAxes { message: error.to_string() })
}

/// Applies a member [`Operation`]'s batching rule through a projected view of a composite [`BatchingContext`]. Use
/// this function from a composite operation dispatcher when the operation is [`Region`](crate::Region)-free and every
/// operand and result belongs to the same projectable member type `T`. It converts the packed input values to the
/// member value family, preserves the outer batch axes and mapped extent, runs the member's existing batching rule,
/// and converts the results back to the composite value family. [`BatchingPolicyProjection`] selects the member policy
/// that represents that same extent for the specific projected type `T`. This keeps homogeneous member rules
/// independent of the enclosing composite type.
///
/// Operations with mixed member types or attached regions require an explicit composite batching rule instead.
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
) -> Result<Vec<P::Batch>, BatchingError> {
    let projected_context = BatchingContext::<_, P::Projected>::with_policy(
        ProjectedContext::new(context.parent().clone()),
        context.axis_extent().clone(),
    )
    .with_axis_name(context.axis_name().map(str::to_string))
    .with_axis_sharding(context.axis_sharding().clone());
    let inputs = inputs
        .iter()
        .map(|input| P::Projected::batch(C::Value::into_projected(P::value(input).clone())?, P::batch_axis(input)))
        .collect::<Result<Vec<_>, BatchingError>>()?;
    operation
        .batch(&projected_context, &EmptyRegionDriver, inputs.as_slice())?
        .into_iter()
        .map(|output| {
            let batch_axis = P::Projected::batch_axis(&output);
            let value = C::Value::from_projected(P::Projected::value(&output).clone());
            P::batch(value, batch_axis)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use approx::assert_abs_diff_eq;
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::backends::arrays::{Array, ArrayOperation};
    use crate::backends::scalars::Scalar;
    use crate::contexts::EagerContext;
    use crate::contexts::tests::{
        ProjectedMemberOperation, ProjectedMemberType, ProjectedMemberValue, ProjectedProgramOperation,
        ProjectedProgramType, ProjectedProgramValue,
    };
    use crate::differentiation::forward::{ForwardModeDifferentiate, LinearizationTracer};
    use crate::differentiation::reverse::ReverseModeDifferentiate;
    use crate::operations::constants::OneLike;
    use crate::operations::math::{AddOperation, NegOperation, Reduce, ReductionKind};
    use crate::parameters::Placeholder;
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::types::Typed;
    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use crate::tracing::{DomainTracingContext, Trace};
    use crate::types::{ArrayType, DataType, Dimension, DimensionBounds, DimensionVariable, Shape};

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
        type BatchedProgram = BatchedProgram<C::Constant, C::Operation>;

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
        ) -> Result<BatchedProgram<C::Constant, C::Operation>, BatchingError> {
            let (program, output_axes) = program.into_parts();
            BatchedProgram::from_widened_boundary(program, output_axes, required_output_axes, 0, collapse_fn)
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
        ) -> Result<Vec<ProjectedBatch<C::Value>>, BatchingError>
        where
            Self: Operation<Type = C::Type>,
        {
            if inputs.len() != 1 {
                return Err(ProgramError::InvalidInputCount { expected: 1, actual: inputs.len() }.into());
            }
            Ok(inputs.to_vec())
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
        type BatchedProgram = BatchedProgram<C::Constant, C::Operation>;

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
        ) -> Result<BatchedProgram<C::Constant, C::Operation>, BatchingError> {
            let (program, output_axes) = program.into_parts();
            BatchedProgram::from_widened_boundary(program, output_axes, required_output_axes, 0, collapse_fn)
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
        ) -> Result<Vec<ProjectedBatch<C::Value>>, BatchingError> {
            check_count!("input", inputs, 1, ProgramError);
            Ok(inputs.to_vec())
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
    fn test_batched_program() {
        // Construction validates that the output axes cover exactly the program's outputs.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F32));
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![input], vec![Placeholder], vec![Placeholder]).unwrap();
        assert_eq!(
            BatchedProgram::new(program, Vec::new()).map(|_| ()),
            Err(ProgramError::InvalidOutputCount { expected: 1, actual: 0 }),
        );

        // A well-formed result preserves its program and output axes through `into_parts`.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F32));
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![input], vec![Placeholder], vec![Placeholder]).unwrap();
        let (program, output_axes) = BatchedProgram::new(program, vec![BatchAxis::replicated()]).unwrap().into_parts();
        assert_eq!(program.output_ids(), &[input]);
        assert_eq!(output_axes, vec![BatchAxis::replicated()]);
    }

    #[test]
    fn test_batched_program_from_widened_boundary() {
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
        let (program, output_axes) =
            BatchedProgram::from_widened_boundary(program, vec![BatchAxis::new(0)], None, 0, |_, _, _| {
                unreachable!("rewrapping must not collapse any output")
            })
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
        let (program, output_axes) =
            BatchedProgram::from_widened_boundary(program, vec![BatchAxis::new(0)], None, 1, |_, _, _| {
                unreachable!("dropping a widening output must not collapse any source output")
            })
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
        let (program, output_axes) = BatchedProgram::from_widened_boundary(
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
            BatchedProgram::from_widened_boundary(
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
    fn test_array_batch() {
        let matrix = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let matrix_type = matrix.r#type().into_owned();

        // `new` builds a batched value when the mapped axis is in bounds, and the accessors report the packed value,
        // its packed type, the batch size read off the mapped axis, and the per-item type with that axis removed.
        let batched = ArrayBatch::new(matrix_type.clone(), matrix.clone(), Some(0)).unwrap();
        assert_eq!(batched.batch_axis(), BatchAxis::new(0));
        assert_eq!(batched.value(), &matrix);
        assert_eq!(*batched.r#type(), matrix_type);
        assert_eq!(batched.batch_size(), Ok(Some(2)));
        assert_eq!(batched.unbatched_type(), ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)])));
        assert_eq!(batched.to_string(), "batch[f64[2, 3], axis=0]([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])");
        assert_eq!(batched.into_value(), matrix);

        // A different mapped axis reads the batch size and per-item type from that axis instead.
        let batched_axis_one = ArrayBatch::new(matrix_type.clone(), matrix.clone(), Some(1)).unwrap();
        assert_eq!(batched_axis_one.batch_size(), Ok(Some(3)));
        assert_eq!(
            batched_axis_one.unbatched_type(),
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)])),
        );

        // Negative axes follow Python/JAX indexing and are normalized once at construction.
        // `-1` denotes the final axis and the stored metadata is the canonical nonnegative position.
        let batched_axis_negative_one =
            ArrayBatch::new(matrix_type.clone(), matrix.clone(), BatchAxis::new(-1)).unwrap();
        assert_eq!(batched_axis_negative_one.batch_axis(), BatchAxis::new(1));
        assert_eq!(batched_axis_negative_one.batch_size(), Ok(Some(3)));

        // `new` rejects an out-of-bounds mapped axis.
        assert_eq!(
            ArrayBatch::new(matrix_type.clone(), matrix, Some(2)),
            Err(BatchingError::BatchAxisOutOfBounds { r#type: Box::new(matrix_type), axis: Axis::from(2) }),
        );

        let matrix = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let matrix_type = matrix.r#type().into_owned();
        assert_eq!(
            ArrayBatch::new(matrix_type.clone(), matrix, BatchAxis::new(-3)),
            Err(BatchingError::BatchAxisOutOfBounds { r#type: Box::new(matrix_type), axis: Axis::from(-3) }),
        );

        // `replicated` shares the value unchanged across the batch: no mapped axis, no batch size, and the per-item
        // type is the whole packed type.
        let vector = Array::vector(vec![1.0, 2.0, 3.0]);
        let vector_type = vector.r#type().into_owned();
        let replicated = ArrayBatch::replicated(vector.clone());
        assert_eq!(replicated.batch_axis(), BatchAxis::replicated());
        assert_eq!(*replicated.r#type(), vector_type);
        assert_eq!(replicated.batch_size(), Ok(None));
        assert_eq!(replicated.unbatched_type(), vector_type);
        assert_eq!(replicated.to_string(), "batch[f64[3], replicated]([1.0, 2.0, 3.0])");
        assert_eq!(replicated.into_value(), vector);
    }

    #[test]
    fn test_array_batch_common_batch_size() {
        let matrix = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let matrix_type = matrix.r#type().into_owned();
        let vector = Array::vector(vec![7.0, 8.0]);

        // All-replicated inputs pin no batch size.
        let replicated = ArrayBatch::replicated(matrix.clone());
        assert_eq!(ArrayBatch::common_batch_size(&[replicated.clone()]), Ok(None));

        // A single batched input pins its own batch size, and a replicated input alongside it is ignored.
        let batched_axis_zero = ArrayBatch::new(matrix_type.clone(), matrix.clone(), Some(0)).unwrap();
        assert_eq!(ArrayBatch::common_batch_size(&[batched_axis_zero.clone()]), Ok(Some(2)));
        assert_eq!(ArrayBatch::common_batch_size(&[replicated, batched_axis_zero.clone()]), Ok(Some(2)));

        // Two batched inputs that agree on their batch size share it, even across different mapped axes.
        let batched_vector = ArrayBatch::new(vector.r#type().into_owned(), vector, Some(0)).unwrap();
        assert_eq!(ArrayBatch::common_batch_size(&[batched_axis_zero.clone(), batched_vector]), Ok(Some(2)));

        // Two batched inputs that disagree on their batch size are rejected.
        let batched_axis_one = ArrayBatch::new(matrix_type, matrix, Some(1)).unwrap();
        assert_eq!(
            ArrayBatch::common_batch_size(&[batched_axis_zero, batched_axis_one]),
            Err(BatchingError::MismatchedBatchSizes { expected: 2, actual: 3 }),
        );
    }

    #[test]
    fn test_array_batch_broadcast() {
        // Broadcasting a replicated vector to gain a leading batch axis of size 2 replicates it across the batch.
        let replicated = ArrayBatch::replicated(Array::vector(vec![1.0, 2.0, 3.0]));
        let broadcasted = replicated.broadcast(0, 2, ShardingDimension::Replicated).unwrap();
        assert_eq!(broadcasted.batch_axis(), BatchAxis::new(0));
        assert_eq!(
            *broadcasted.r#type(),
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)])),
        );
        assert_eq!(broadcasted.value(), &Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]));
        assert_eq!(broadcasted.unbatched_type(), ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)])),);

        // Broadcasting to a trailing batch axis keeps the per-item dimensions before it in place.
        // The input's own axis stays at position 0 and the size-2 batch axis is inserted at position 1.
        let broadcasted = replicated.broadcast(1, 2, ShardingDimension::Replicated).unwrap();
        assert_eq!(broadcasted.batch_axis(), BatchAxis::new(1));
        assert_eq!(
            *broadcasted.r#type(),
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3), Dimension::Static(2)])),
        );
        assert_eq!(broadcasted.value(), &Array::matrix(3, 2, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]));

        // Broadcasting rejects an already-batched value.
        let vector_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)]));
        let batched = ArrayBatch::new(vector_type, Array::vector(vec![1.0, 2.0]), Some(0)).unwrap();
        assert!(matches!(
            batched.broadcast(0, 2, ShardingDimension::Replicated),
            Err(BatchingError::MisalignedBatchAxes { .. }),
        ));
    }

    #[test]
    fn test_array_batch_move_axis() {
        let matrix = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let matrix_type = matrix.r#type().into_owned();
        let batched = ArrayBatch::new(matrix_type.clone(), matrix, Some(0)).unwrap();

        // Moving to the axis the value already maps returns it unchanged.
        assert_eq!(batched.move_axis(0).unwrap(), batched);

        // Moving the mapped batch axis from 0 to 1 transposes the packed value (from [2, 3] to [3, 2]) and records the
        // new mapped axis, while the per-item type ([3]) and the batch size (2) are preserved.
        let moved = batched.move_axis(1).unwrap();
        assert_eq!(moved.batch_axis(), BatchAxis::new(1));
        assert_eq!(
            *moved.r#type(),
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3), Dimension::Static(2)])),
        );
        assert_eq!(moved.value(), &Array::matrix(3, 2, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]));
        assert_eq!(moved.unbatched_type(), ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)])));
        assert_eq!(moved.batch_size(), Ok(Some(2)));

        // A replicated value has no mapped axis, so moving to any axis is a no-op.
        let replicated = ArrayBatch::replicated(Array::vector(vec![1.0, 2.0, 3.0]));
        assert_eq!(replicated.move_axis(1).unwrap(), replicated);
    }

    #[test]
    fn test_array_batch_match_axis() {
        // `match_axis` on a batched value moves its mapped axis to the target (like `move_axis`). [2, 3] mapped at 0
        // becomes [3, 2] mapped at 1, and the `axis_size` argument is unused for an already-batched value.
        let batched = ArrayBatch::new(
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)])),
            Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            Some(0),
        )
        .unwrap();
        let matched = batched.match_axis(1, 2, ShardingDimension::Replicated).unwrap();
        assert_eq!(matched.batch_axis(), BatchAxis::new(1));
        assert_eq!(
            *matched.r#type(),
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3), Dimension::Static(2)])),
        );
        assert_eq!(matched.value(), &Array::matrix(3, 2, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]));

        // `match_axis` on a replicated value broadcasts it to gain a batch axis there (like `broadcast`).
        let replicated = ArrayBatch::replicated(Array::vector(vec![1.0, 2.0, 3.0]));
        let matched = replicated.match_axis(0, 2, ShardingDimension::Replicated).unwrap();
        assert_eq!(matched.batch_axis(), BatchAxis::new(0));
        assert_eq!(matched.value(), &Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]));
    }

    #[test]
    fn test_array_batch_align_axis() {
        let batched = ArrayBatch::new(
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)])),
            Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            Some(0),
        )
        .unwrap();

        // Aligning a batched value to the axis it already maps returns it unchanged.
        assert_eq!(batched.align_axis(BatchAxis::new(0), 2, ShardingDimension::Replicated).unwrap(), batched);

        // Aligning to a different mapped position stages a transpose (like `move_axis`). [2, 3] mapped at 0 becomes
        // [3, 2] mapped at 1. The equivalent negative declaration is normalized against the batched output rank.
        let aligned = batched.align_axis(BatchAxis::new(-1), 2, ShardingDimension::Replicated).unwrap();
        assert_eq!(aligned.batch_axis(), BatchAxis::new(1));
        assert_eq!(
            *aligned.r#type(),
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3), Dimension::Static(2)])),
        );
        assert_eq!(aligned.value(), &Array::matrix(3, 2, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]));

        // Aligning a replicated value to a replicated declaration returns it unchanged.
        let replicated = ArrayBatch::replicated(Array::vector(vec![1.0, 2.0, 3.0]));
        assert_eq!(
            replicated.align_axis(BatchAxis::replicated(), 2, ShardingDimension::Replicated).unwrap(),
            replicated,
        );

        // A mapped output declaration instantiates a naturally replicated value by broadcasting it across the batch.
        let aligned = replicated.align_axis(BatchAxis::new(-1), 2, ShardingDimension::Replicated).unwrap();
        assert_eq!(aligned.batch_axis(), BatchAxis::new(1));
        assert_eq!(aligned.value(), &Array::matrix(3, 2, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]));

        // The reverse presence change remains invalid: collapsing a mapped output requires an explicit reduction.
        assert_eq!(
            batched.align_axis(BatchAxis::replicated(), 2, ShardingDimension::Replicated),
            Err(BatchingError::MismatchedOutputAxes { expected: BatchAxis::replicated(), actual: BatchAxis::new(0) }),
        );
    }

    #[test]
    fn test_array_batch_sharding_for_inputs() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let sharded_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]))
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                    .unwrap(),
            )
            .unwrap();

        // A batched input whose mapped axis is sharded contributes that `ShardingDimension`.
        let batched = {
            let value = sharded_type.clone();
            ArrayBatch::new(value.r#type().into_owned(), value, Some(0))
        }
        .unwrap();
        assert_eq!(
            ArrayBatch::sharding_for_inputs(std::slice::from_ref(&batched)).unwrap(),
            ShardingDimension::sharded(["x"])
        );

        // A replicated batch input (no mapped axis) contributes no mapped-axis sharding, so the derived batch
        // dimension defaults to replicated sharding.
        let replicated = ArrayBatch::replicated(sharded_type);
        assert_eq!(
            ArrayBatch::sharding_for_inputs(std::slice::from_ref(&replicated)).unwrap(),
            ShardingDimension::replicated()
        );

        // A replicated mapped dimension is explicitly normalized to the concrete sharded placement contributed
        // by another mapped input.
        let replicated_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]))
                .with_sharding(Sharding::replicated(mesh.clone(), 2))
                .unwrap();
        let other = {
            let value = replicated_type;
            ArrayBatch::new(value.r#type().into_owned(), value, Some(0))
        }
        .unwrap();
        assert_eq!(
            ArrayBatch::sharding_for_inputs(&[batched.clone(), other.clone()]).unwrap(),
            ShardingDimension::sharded(["x"]),
        );
        assert_eq!(
            ArrayBatch::sharding_for_inputs(&[other, batched.clone()]).unwrap(),
            ShardingDimension::sharded(["x"]),
        );

        // Two distinct concrete placements remain ambiguous and are rejected.
        let differently_sharded_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]))
                .with_sharding(
                    Sharding::new(mesh, vec![ShardingDimension::sharded(["y"]), ShardingDimension::replicated()])
                        .unwrap(),
                )
                .unwrap();
        let differently_sharded =
            ArrayBatch::new(differently_sharded_type.clone(), differently_sharded_type, BatchAxis::new(0)).unwrap();
        let error = ArrayBatch::sharding_for_inputs(&[batched, differently_sharded]).unwrap_err();
        assert!(matches!(
            error.downcast_custom::<BatchingError>(),
            Some(BatchingError::MisalignedBatchAxes { message })
                if message.contains("mismatched batch axis sharding"),
        ));
    }

    #[test]
    fn test_array_batch_preserves_explicit_mapped_axis_sharding() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 2);
        for (batch_axis, dimensions) in [vec![2, 3, 4], vec![3, 2, 4], vec![3, 4, 2]].into_iter().enumerate() {
            let mut sharding_dimensions = vec![ShardingDimension::replicated(); 3];
            sharding_dimensions[batch_axis] = ShardingDimension::sharded(["x"]);
            let batched_type =
                ArrayType::new(DataType::F64, Shape::new(dimensions.into_iter().map(Dimension::Static).collect()))
                    .with_sharding(Sharding::new(mesh.clone(), sharding_dimensions).unwrap())
                    .unwrap();
            let batch = ArrayBatch::new(
                batched_type.clone(),
                Array::from_f64s(batched_type.clone(), (0..24).map(f64::from).collect()),
                BatchAxis::from_position(batch_axis),
            )
            .unwrap();
            let mut unbatched_dimensions = batched_type.shape().dimensions().to_vec();
            unbatched_dimensions.remove(batch_axis);
            assert_eq!(
                batch.unbatched_type(),
                ArrayType::new(DataType::F64, Shape::new(unbatched_dimensions))
                    .with_sharding(Sharding::replicated(mesh.clone(), 2))
                    .unwrap(),
            );
            let outputs = AddOperation::new().batch(&context, &EmptyRegionDriver, &[batch.clone(), batch]).unwrap();
            assert_eq!(outputs[0].r#type(), Cow::Borrowed(&batched_type));
            assert_eq!(outputs[0].batch_axis(), BatchAxis::from_position(batch_axis));
        }
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
        let outputs = operation.batch(&context, &EmptyRegionDriver, &[tracer.into_batch()]).unwrap();
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
    fn test_array_type_normalize_batch_axis() {
        let r#type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        assert_eq!(r#type.normalize_batch_axis(BatchAxis::replicated()), Ok((BatchAxis::replicated(), None)));
        assert_eq!(r#type.normalize_batch_axis(BatchAxis::new(0)), Ok((BatchAxis::new(0), Some(0))));
        assert_eq!(r#type.normalize_batch_axis(BatchAxis::new(-1)), Ok((BatchAxis::new(1), Some(1))));
        assert_eq!(
            r#type.normalize_batch_axis(BatchAxis::new(2)),
            Err(BatchingError::BatchAxisOutOfBounds { r#type: Box::new(r#type.clone()), axis: Axis::from(2) }),
        );
        assert_eq!(
            r#type.normalize_batch_axis(BatchAxis::new(-3)),
            Err(BatchingError::BatchAxisOutOfBounds { r#type: Box::new(r#type), axis: Axis::from(-3) }),
        );
    }

    #[test]
    fn test_array_type_unbatched_type() {
        let r#type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        assert_eq!(r#type.unbatched_type(BatchAxis::replicated()), Ok(r#type.clone()));
        assert_eq!(r#type.unbatched_type(0), Ok(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]))));
        assert_eq!(
            r#type.unbatched_type(BatchAxis::new(-1)),
            Ok(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)]))),
        );
    }

    #[test]
    fn test_elementwise_operation_batch() {
        // The blanket `BatchableOperation` for elementwise operations lifts `interpret` over the mapped batch axis.
        // It realigns every mapped operand onto the common axis, broadcasts replicated operands across the batch,
        // and reports each output on that common axis.
        let vector_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));
        let matrix_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let make_batch = |r#type: &ArrayType, values: Vec<f64>, axis: Option<isize>| {
            ArrayBatch::new(r#type.clone(), Array::from_f64s(r#type.clone(), values), axis).unwrap()
        };

        // Batching rules always receive the active `BatchingContext`, with the underlying work running
        // through its parent context.
        let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 2);

        // Two operands mapped on the same axis add per item, and the output stays mapped on that axis.
        let left = make_batch(&matrix_type, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Some(0));
        let right = make_batch(&matrix_type, vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], Some(0));
        let outputs = AddOperation::new().batch(&context, &EmptyRegionDriver, &[left.clone(), right]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value(), &Array::matrix(2, 3, vec![11.0, 22.0, 33.0, 44.0, 55.0, 66.0]));

        // A replicated operand is broadcast across the mapped operand's batch before adding.
        let replicated = make_batch(&vector_type, vec![10.0, 20.0, 30.0], None);
        let outputs = AddOperation::new().batch(&context, &EmptyRegionDriver, &[left.clone(), replicated]).unwrap();
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value(), &Array::matrix(2, 3, vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]));

        // Operands mapped on different axes are realigned onto the first mapped operand's axis before adding.
        let transposed_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3), Dimension::Static(2)]));
        let right_axis_one = make_batch(&transposed_type, vec![10.0, 40.0, 20.0, 50.0, 30.0, 60.0], Some(1));
        let outputs = AddOperation::new().batch(&context, &EmptyRegionDriver, &[left, right_axis_one]).unwrap();
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value(), &Array::matrix(2, 3, vec![11.0, 22.0, 33.0, 44.0, 55.0, 66.0]));

        // Packed mapped-axis positions are canonicalized independently of operand rank. The rank-3 left operand
        // maps its trailing axis while the rank-1 right operand maps its only axis; their unbatched per-item shapes are
        // `[3, 4]` and scalar, respectively. The output is restored to the first mapped input's trailing axis.
        let left_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Static(3), Dimension::Static(4), Dimension::Static(2)]),
        );
        let right_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)]));
        let left = make_batch(&left_type, (1..=24).map(f64::from).collect(), Some(2));
        let right = make_batch(&right_type, vec![10.0, 20.0], Some(0));
        let outputs = AddOperation::new().batch(&context, &EmptyRegionDriver, &[left, right]).unwrap();
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(2));
        assert_eq!(
            outputs[0].value(),
            &Array::from_f64s(
                left_type,
                vec![
                    11.0, 22.0, 13.0, 24.0, 15.0, 26.0, 17.0, 28.0, 19.0, 30.0, 21.0, 32.0, 23.0, 34.0, 25.0, 36.0,
                    27.0, 38.0, 29.0, 40.0, 31.0, 42.0, 33.0, 44.0,
                ],
            ),
        );

        // With no operand mapped, the operands are interpreted as given and the output is replicated.
        let left_replicated = make_batch(&vector_type, vec![1.0, 2.0, 3.0], None);
        let right_replicated = make_batch(&vector_type, vec![10.0, 20.0, 30.0], None);
        let outputs = AddOperation::new()
            .batch(&context, &EmptyRegionDriver, &[left_replicated, right_replicated])
            .unwrap();
        assert_eq!(outputs[0].batch_axis(), BatchAxis::replicated());
        assert_eq!(outputs[0].value(), &Array::vector(vec![11.0, 22.0, 33.0]));

        // Unary elementwise operations use the same blanket rule and preserve the mapped input axis.
        let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 3);
        let input = make_batch(&vector_type, vec![1.0, 2.0, 3.0], Some(0));
        let outputs = NegOperation::new().batch(&context, &EmptyRegionDriver, &[input]).unwrap();
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value(), &Array::vector(vec![-1.0, -2.0, -3.0]));
    }

    #[test]
    fn test_elementwise_operation_batch_normalizes_replicated_and_sharded_mapped_axes() {
        for axis_type in [MeshAxisType::Explicit, MeshAxisType::Manual] {
            let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, axis_type).unwrap()]).unwrap();
            let make_type = |axis_sharding| {
                let varying_manual_axes = (axis_type == MeshAxisType::Manual
                    && matches!(axis_sharding, ShardingDimension::Sharded(_)))
                .then_some("x");
                let sharding = Sharding::new(mesh.clone(), vec![axis_sharding, ShardingDimension::replicated()])
                    .unwrap()
                    .with_varying_manual_axes(varying_manual_axes)
                    .unwrap();
                ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]))
                    .with_sharding(sharding)
                    .unwrap()
            };
            let sharded_type = make_type(ShardingDimension::sharded(["x"]));
            let replicated_type = make_type(ShardingDimension::replicated());
            let sharded = ArrayBatch::new(
                sharded_type.clone(),
                Array::from_f64s(sharded_type.clone(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                BatchAxis::new(0),
            )
            .unwrap();
            let replicated = ArrayBatch::new(
                replicated_type.clone(),
                Array::from_f64s(replicated_type, vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
                BatchAxis::new(0),
            )
            .unwrap();
            let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 2)
                .with_axis_sharding(ShardingDimension::sharded(["x"]));
            let outputs = AddOperation::new().batch(&context, &EmptyRegionDriver, &[sharded, replicated]).unwrap();
            assert_eq!(outputs.len(), 1);
            assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
            assert_eq!(outputs[0].r#type(), Cow::Borrowed(&sharded_type));
            assert_eq!(outputs[0].value().values(), &[11.0, 22.0, 33.0, 44.0, 55.0, 66.0]);
        }
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
    fn test_region_batched_preserves_mapped_axis_sharding() {
        for axis_type in [MeshAxisType::Explicit, MeshAxisType::Manual] {
            let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, axis_type).unwrap()]).unwrap();
            let unbatched_sharding = if axis_type == MeshAxisType::Manual {
                Sharding::new(mesh.clone(), vec![ShardingDimension::replicated()])
                    .unwrap()
                    .with_varying_manual_axes(["x"])
                    .unwrap()
            } else {
                Sharding::replicated(mesh.clone(), 1)
            };
            let unbatched_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]))
                .with_sharding(unbatched_sharding)
                .unwrap();
            let (_, program) =
                EagerContext::<Array, ArrayOperation<Array>>::trace(|inputs: Vec<_>| Ok(inputs), vec![unbatched_type])
                    .unwrap();

            let (batched, output_axes) = program
                .entry_region_ref()
                .batched(
                    2,
                    ShardingDimension::sharded(["x"]),
                    &[BatchAxis::new(1)],
                    ProgramBatchingOutputAxesPolicy::Natural,
                )
                .unwrap()
                .into_parts();
            let expected_sharding =
                Sharding::new(mesh, vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])])
                    .unwrap()
                    .with_varying_manual_axes((axis_type == MeshAxisType::Manual).then_some("x"))
                    .unwrap();
            let expected_type =
                ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3), Dimension::Static(2)]))
                    .with_sharding(expected_sharding)
                    .unwrap();

            assert_eq!(batched.input_types(), &[expected_type.clone()]);
            assert_eq!(batched.output_types(), &[expected_type]);
            assert_eq!(output_axes, vec![BatchAxis::new(1)]);
        }
    }

    #[test]
    fn test_program_batched_transforms_input_and_output_axes() {
        // Trace a per-item squaring function into a flat program over per-item vector types.
        let vector_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |inputs: Vec<_>| Ok(vec![inputs[0].clone() * inputs[0].clone()]),
            vec![vector_type.clone()],
        )
        .unwrap();

        // A mapped input at axis 0 turns the program into one over `[2, 3]`-shaped packed inputs that squares each
        // row, with the output naturally mapped on the same axis.
        let (batched, output_axes) = program
            .batched(2, ShardingDimension::Replicated, &[BatchAxis::new(0)], ProgramBatchingOutputAxesPolicy::Natural)
            .unwrap()
            .into_parts();
        assert_eq!(output_axes, vec![BatchAxis::new(0)]);
        let input = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let outputs = batched.interpret(vec![input]).unwrap();
        assert_eq!(outputs, vec![Array::matrix(2, 3, vec![1.0, 4.0, 9.0, 16.0, 25.0, 36.0])]);

        // A negative input axis is normalized against the packed input rank. Mapping the final axis consumes a
        // `[3, 2]` packed value and preserves that canonical axis through the elementwise body.
        let (batched, output_axes) = program
            .batched(2, ShardingDimension::Replicated, &[BatchAxis::new(-1)], ProgramBatchingOutputAxesPolicy::Natural)
            .unwrap()
            .into_parts();
        assert_eq!(output_axes, vec![BatchAxis::new(1)]);
        let input = Array::matrix(3, 2, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        let outputs = batched.interpret(vec![input]).unwrap();
        assert_eq!(outputs, vec![Array::matrix(3, 2, vec![1.0, 16.0, 4.0, 25.0, 9.0, 36.0])]);

        // A replicated input keeps its unbatched `[3]` type, and `AlignAllTo(0)` broadcasts the naturally replicated
        // output across the batch so the batched program still produces one `[2, 3]` output per item.
        let (batched, output_axes) = program
            .batched(
                2,
                ShardingDimension::Replicated,
                &[BatchAxis::replicated()],
                ProgramBatchingOutputAxesPolicy::AlignAllTo(Axis::from(0)),
            )
            .unwrap()
            .into_parts();
        assert_eq!(output_axes, vec![BatchAxis::new(0)]);
        let outputs = batched.interpret(vec![Array::vector(vec![1.0, 2.0, 3.0])]).unwrap();
        assert_eq!(outputs, vec![Array::matrix(2, 3, vec![1.0, 4.0, 9.0, 1.0, 4.0, 9.0])]);

        // Signed output policies normalize after accounting for the inserted batch dimension. `-1` places the
        // instantiated batch axis last.
        let (batched, output_axes) = program
            .batched(
                2,
                ShardingDimension::Replicated,
                &[BatchAxis::replicated()],
                ProgramBatchingOutputAxesPolicy::AlignAllTo(Axis::from(-1)),
            )
            .unwrap()
            .into_parts();
        assert_eq!(output_axes, vec![BatchAxis::new(1)]);
        let outputs = batched.interpret(vec![Array::vector(vec![1.0, 2.0, 3.0])]).unwrap();
        assert_eq!(outputs, vec![Array::matrix(3, 2, vec![1.0, 1.0, 4.0, 4.0, 9.0, 9.0])]);

        // A mismatched `input_batch_axes` count is rejected.
        assert!(
            program
                .batched(2, ShardingDimension::Replicated, &[], ProgramBatchingOutputAxesPolicy::Natural)
                .is_err(),
        );
    }

    #[test]
    fn test_batch_entry_points_and_axis_contracts() {
        // `Batch::batch` on an explicit context maps the closure over the mapped input axis: each item of the
        // length-3 batch is squared, and the output carries its mapped axis back at the requested position.
        let output: Array = EagerContext::<Array, ArrayOperation<Array>>::new()
            .batch(
                |x| Ok(x.clone() * x),
                Array::vector(vec![1.0, 2.0, 3.0]),
                BatchAxis::new(0),
                BatchAxis::new(0),
                None,
            )
            .unwrap();
        assert_eq!(output, Array::vector(vec![1.0, 4.0, 9.0]));

        // A mapped output declaration broadcasts a naturally replicated result across the explicit batch.
        // The signed `-1` declaration places that new batch dimension last.
        let output: Array = EagerContext::<Array, ArrayOperation<Array>>::new()
            .batch(
                |x| Ok(x.clone() * x),
                Array::vector(vec![1.0, 2.0, 3.0]),
                BatchAxis::replicated(),
                BatchAxis::new(-1),
                2,
            )
            .unwrap();
        assert_eq!(output, Array::matrix(3, 2, vec![1.0, 1.0, 4.0, 4.0, 9.0, 9.0]));

        // The free `batch` serves top-level concrete values through their `Value::ExecutionDomain` declarations: a
        // plain `Array` input recovers the test backend's rich eager domain, mirroring how JAX's `vmap` falls back
        // to the default eager interpreter for concrete arrays.
        let output: Array = batch(
            |x| Ok(x.clone() * x),
            Array::vector(vec![1.0, 2.0, 3.0]),
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        )
        .unwrap();
        assert_eq!(output, Array::vector(vec![1.0, 4.0, 9.0]));

        // Under an active trace, the free `batch` recovers the staging context from its tracer input instead, so
        // `batch` composes inside traced code without threading a context. The traced function squares each row of
        // its `[2, 3]` input by batching a per-item squaring closure over axis 0.
        let matrix_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |inputs: Vec<_>| {
                let mapped =
                    batch(|x| Ok(x.clone() * x), inputs[0].clone(), BatchAxis::new(0), BatchAxis::new(0), None)?;
                Ok(vec![mapped])
            },
            vec![matrix_type],
        )
        .unwrap();
        let input = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let outputs = program.interpret(vec![input]).unwrap();
        assert_eq!(outputs, vec![Array::matrix(2, 3, vec![1.0, 4.0, 9.0, 16.0, 25.0, 36.0])]);

        // Nested inside an eager `batch`, the inner free `batch` recovers the outer `BatchingContext` from its
        // `BatchingTracer` input, so that `batch` nests inside `batch`: the outer level maps rows and the inner level
        // maps items within each row.
        let output: Array = EagerContext::<Array, ArrayOperation<Array>>::new()
            .batch(
                |row| Ok(batch(|item| Ok(item.clone() * item), row, BatchAxis::new(0), BatchAxis::new(0), None)?),
                Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                BatchAxis::new(0),
                BatchAxis::new(0),
                None,
            )
            .unwrap();
        assert_eq!(output, Array::matrix(2, 3, vec![1.0, 4.0, 9.0, 16.0, 25.0, 36.0]));

        // A replicated input with an explicit batch size runs the closure on the shared value and returns
        // a replicated output.
        let output: Array = EagerContext::<Array, ArrayOperation<Array>>::new()
            .batch(
                |x| Ok(x.clone() * x),
                Array::vector(vec![1.0, 2.0, 3.0]),
                BatchAxis::replicated(),
                BatchAxis::replicated(),
                2,
            )
            .unwrap();
        assert_eq!(output, Array::vector(vec![1.0, 4.0, 9.0]));

        // Declaring a mapped output as replicated is rejected: collapsing a mapped axis requires an explicit
        // reduction inside the batched function.
        let error = EagerContext::<Array, ArrayOperation<Array>>::new()
            .batch(
                |x| Ok(x.clone() * x),
                Array::vector(vec![1.0, 2.0, 3.0]),
                BatchAxis::new(0),
                BatchAxis::replicated(),
                None,
            )
            .unwrap_err();
        assert!(matches!(error, BatchingError::MismatchedOutputAxes { .. }));

        // With no mapped input and no explicit batch size, the batch size is unobservable.
        let error = EagerContext::<Array, ArrayOperation<Array>>::new()
            .batch(
                |x| Ok(x.clone() * x),
                Array::vector(vec![1.0, 2.0, 3.0]),
                BatchAxis::replicated(),
                BatchAxis::replicated(),
                None,
            )
            .unwrap_err();
        assert_eq!(error, BatchingError::EmptyBatch);

        // With no leaf value to recover a context from, the free `batch` reports an empty batch even when an
        // explicit batch size is provided.
        let error = batch(
            |x: Vec<BatchingTracer<EagerContext<Array, ArrayOperation<Array>>, ArrayBatching>>| Ok(x),
            Vec::<Array>::new(),
            BatchAxis::replicated(),
            BatchAxis::replicated(),
            2,
        )
        .unwrap_err();
        assert_eq!(error, BatchingError::EmptyBatch);
    }

    #[test]
    fn test_batch_normalizes_mapped_input_sharding_before_tracing() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharded_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)]))
            .with_sharding(Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        let replicated_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)]))
            .with_sharding(Sharding::replicated(mesh, 1))
            .unwrap();
        let sharded = Array::from_f64s(sharded_type.clone(), vec![1.0, 2.0]);
        let replicated = Array::from_f64s(replicated_type, vec![3.0, 4.0]);
        let output = EagerContext::<Array, ArrayOperation<Array>>::new()
            .batch(
                |(_sharded, normalized)| Ok(normalized),
                (sharded, replicated),
                BatchAxis::new(0),
                BatchAxis::new(0),
                None,
            )
            .unwrap();
        assert_eq!(output.r#type(), Cow::Borrowed(&sharded_type));
        assert_eq!(output.values(), &[3.0, 4.0]);
    }

    #[test]
    fn test_value_and_gradient_flow_through_batch_staged_broadcast() {
        // The scalar input is replicated inside the batch, so the elementwise batching rule stages a broadcasting
        // operation on the differentiated value; the gradient must flow back through the broadcast's transpose rule
        // (a sum-reduction over the batch axis).
        let (value, gradient) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .value_and_gradient(
                |x| {
                    let context = x.context().clone();
                    let y = context.lift(Array::vector(vec![1.0, 2.0, 3.0, 4.0])).unwrap();
                    let mapped: LinearizationTracer<EagerContext<Array, ArrayOperation<Array>>> = Batch::batch(
                        &context,
                        |(item, shift)| Ok(item * shift),
                        (y, x),
                        (BatchAxis::new(0), BatchAxis::replicated()),
                        BatchAxis::new(0),
                        None,
                    )
                    .unwrap();
                    mapped.reduce(&[0], ReductionKind::Sum)
                },
                Array::scalar(2.0),
            )
            .unwrap();
        assert_abs_diff_eq!(value.to_f64s()[0], 20.0, epsilon = 1e-9);
        assert_eq!(gradient.to_f64s(), vec![10.0]);
    }

    #[test]
    fn test_batch_composes_with_context_jvp() {
        let output: (Array, Array) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .batch(
                |x| {
                    let context = x.context().clone();
                    ForwardModeDifferentiate::jvp(&context, |y| Ok(y.clone() * y), x.clone(), x.one_like())
                        .map_err(ProgramError::from)
                },
                Array::vector(vec![2.0, 3.0]),
                BatchAxis::new(0),
                (BatchAxis::new(0), BatchAxis::new(0)),
                None,
            )
            .unwrap();

        assert_eq!(output.0.to_f64s(), vec![4.0, 9.0]);
        assert_eq!(output.1.to_f64s(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_batch_composes_with_context_value_and_gradient() {
        let output: (Array, Array) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .batch(
                |x| {
                    let context = x.context().clone();
                    Ok(context
                        .value_and_gradient(|y| y.clone() * y, x)
                        .expect("scalar value_and_gradient should succeed"))
                },
                Array::vector(vec![2.0, 3.0]),
                BatchAxis::new(0),
                (BatchAxis::new(0), BatchAxis::new(0)),
                None,
            )
            .unwrap();
        assert_eq!(output.0.to_f64s(), vec![4.0, 9.0]);
        assert_eq!(output.1.to_f64s(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_context_batch_composes_inside_jvp() {
        let (primal, tangent): (Array, Array) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .jvp(
                |x| {
                    let context = x.context().clone();
                    Ok(Batch::batch(
                        &context,
                        |item| Ok(item.clone() * item),
                        x,
                        BatchAxis::new(0),
                        BatchAxis::new(0),
                        None,
                    )?)
                },
                Array::vector(vec![2.0, 3.0]),
                Array::vector(vec![1.0, 1.0]),
            )
            .unwrap();
        assert_eq!(primal.to_f64s(), vec![4.0, 9.0]);
        assert_eq!(tangent.to_f64s(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_context_batch_composes_inside_value_and_gradient() {
        let (value, gradient): (Array, Array) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .value_and_gradient(
                |x| {
                    let context = x.context().clone();
                    let mapped: LinearizationTracer<EagerContext<Array, ArrayOperation<Array>>> = Batch::batch(
                        &context,
                        |item| Ok(item.clone() * item),
                        x,
                        BatchAxis::new(0),
                        BatchAxis::new(0),
                        None,
                    )
                    .unwrap();
                    mapped.reduce(&[0], ReductionKind::Sum)
                },
                Array::vector(vec![2.0, 3.0]),
            )
            .unwrap();
        assert_eq!(value.to_f64s(), vec![13.0]);
        assert_eq!(gradient.to_f64s(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_batch_broadcasts_replicated_input_along_mapped_axis() {
        // x is a [4]-vector mapped on axis 0 (batch items), y is a replicated scalar that should be
        // added to every batch item. The output should be element-wise `x + y` over the 4 batch items.
        let x = Array::vector(vec![1.0, 2.0, 3.0, 4.0]);
        let y = Array::scalar(10.0);
        let output: Array = EagerContext::<Array, ArrayOperation<Array>>::new()
            .batch(
                |(left, right)| Ok(left + right),
                (x, y),
                (BatchAxis::new(0), BatchAxis::replicated()),
                BatchAxis::new(0),
                None,
            )
            .unwrap();
        assert_eq!(output.to_f64s(), vec![11.0, 12.0, 13.0, 14.0]);
    }

    #[test]
    fn test_batch_validates_explicit_axis_size() {
        // An explicit axis size that agrees with the mapped input is accepted and flows through the computation.
        let x = Array::vector(vec![1.0, 2.0, 3.0, 4.0]);
        let output: Array = EagerContext::<Array, ArrayOperation<Array>>::new()
            .batch(|x| Ok(x.clone() + x), x.clone(), BatchAxis::new(0), BatchAxis::new(0), 4)
            .unwrap();
        assert_eq!(output.to_f64s(), vec![2.0, 4.0, 6.0, 8.0]);

        // A different explicit size conflicts with the same mapped input and is rejected.
        let result: Result<Array, BatchingError> = EagerContext::<Array, ArrayOperation<Array>>::new().batch(
            |x| Ok(x.clone() + x),
            x,
            BatchAxis::new(0),
            BatchAxis::new(0),
            5,
        );
        assert!(matches!(result, Err(BatchingError::MismatchedBatchSizes { expected: 5, actual: 4 })));
    }

    #[test]
    fn test_batch_rejects_dynamic_batch_axis() {
        // A mapped input whose batch dimension is `Dimension::Dynamic` cannot be batched since `batch`
        // has no way of determining the batch size.
        let dynamic_input = Array::with_unchecked_type(
            ArrayType::new(
                DataType::F64,
                Shape::new(vec![Dimension::Dynamic(DimensionVariable::new("dynamic", DimensionBounds::unbounded()))]),
            ),
            vec![Scalar::F64(1.0), Scalar::F64(2.0), Scalar::F64(3.0)],
        );
        let result: Result<Array, BatchingError> = EagerContext::<Array, ArrayOperation<Array>>::new().batch(
            |x| Ok(x.clone() + x),
            dynamic_input,
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        );
        assert!(matches!(result, Err(BatchingError::DynamicBatchAxis { axis, .. }) if axis == Axis::from(0)));
    }

    #[test]
    fn test_batch_repositions_mapped_output_axis() {
        // Outer batch over axis 0 of a [3, 4] matrix: each batch item returns its row unchanged. Requesting output
        // batch axis 1 forces a transpose that moves the mapped axis to the end of the rank-2 output.
        let x_data: Vec<f64> = (0..12).map(|value| value as f64).collect();
        let x = Array::matrix(3, 4, x_data.clone());
        let output: Array = EagerContext::<Array, ArrayOperation<Array>>::new()
            .batch(|row| Ok(row), x, BatchAxis::new(0), BatchAxis::new(1), None)
            .unwrap();
        assert_eq!(
            output.r#type().into_owned(),
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(4), Dimension::Static(3)])),
        );
        // Transpose of [3, 4]: output[i, j] = x[j, i]. Row-major flat indexing:
        // x[j, i] = x_data[j*4 + i]; output[i, j] = output_values[i*3 + j].
        for j in 0..3 {
            for i in 0..4 {
                assert_eq!(output.to_f64s()[i * 3 + j], x_data[j * 4 + i]);
            }
        }
    }

    #[test]
    fn test_nested_batch_with_mixed_input_axes_propagates_broadcast() {
        // Outer batch over axis 0 of `x: [3, 4]` exposes a rank-1 row to the closure; inside, a
        // second inner batch maps that row's batch axis 0 while broadcasting a captured `bias`
        // scalar to every inner batch item. The combined output is x + bias broadcasted.
        let x_data: Vec<f64> = (0..12).map(|value| value as f64).collect();
        let x = Array::matrix(3, 4, x_data.clone());
        let bias = Array::scalar(0.5);

        let output: Array = EagerContext::<Array, ArrayOperation<Array>>::new()
            .batch(
                |(row, bias_inner)| {
                    let context = row.context().clone();
                    Ok(Batch::batch(
                        &context,
                        |(scalar, bias_inner)| Ok(scalar + bias_inner),
                        (row, bias_inner),
                        (BatchAxis::new(0), BatchAxis::replicated()),
                        BatchAxis::new(0),
                        None,
                    )?)
                },
                (x, bias),
                (BatchAxis::new(0), BatchAxis::replicated()),
                BatchAxis::new(0),
                None,
            )
            .unwrap();

        assert_eq!(
            output.r#type().into_owned(),
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3), Dimension::Static(4)])),
        );
        let expected: Vec<f64> = x_data.iter().map(|value| value + 0.5).collect();
        for (actual, expected) in output.to_f64s().iter().zip(expected.iter()) {
            assert_abs_diff_eq!(*actual, *expected, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_batch_broadcasts_single_axis_specification_to_every_leaf() {
        // A single `BatchAxis` specification broadcasts into the whole input and output parameter structures, so both
        // leaves of the pair are mapped on axis 0 without spelling out either structure.
        let x = Array::vector(vec![1.0, 3.0]);
        let y = Array::vector(vec![2.0, 4.0]);
        let output: (Array, Array) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .batch(
                |(left, right)| Ok((left.clone() + right.clone(), left * right)),
                (x, y),
                BatchAxis::new(0),
                BatchAxis::new(0),
                None,
            )
            .unwrap();
        assert_eq!(output.0.to_f64s(), vec![3.0, 7.0]);
        assert_eq!(output.1.to_f64s(), vec![2.0, 12.0]);
    }

    #[test]
    fn test_batch_broadcasts_mapped_inputs_with_mixed_per_item_ranks() {
        // x is mapped with per-item shape [3]; y is mapped with a per-item scalar shape. The
        // elementwise rule broadcasts y's per-item scalar across the common per-item shape, so
        // each batch item computes `row + shift` with its own shift.
        let x = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let y = Array::vector(vec![10.0, 20.0]);
        let output: Array = EagerContext::<Array, ArrayOperation<Array>>::new()
            .batch(|(row, shift)| Ok(row + shift), (x, y), BatchAxis::new(0), BatchAxis::new(0), None)
            .unwrap();
        assert_eq!(
            output.r#type().into_owned(),
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)])),
        );
        assert_eq!(output.to_f64s(), vec![11.0, 12.0, 13.0, 24.0, 25.0, 26.0]);
    }

    #[test]
    fn test_batch_broadcasts_scalar_replicated_operand_to_full_shape() {
        // A replicated scalar constant added to a mapped [3, 4] input: the elementwise rule materializes a broadcasting
        // operation to the full common batched shape so the staged add receives shape-congruent operands. This is
        // required for backends such as XLA whose elementwise lowering (e.g., `stablehlo.add`) has no implicit
        // broadcasting.
        let parent = DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::new();
        let builder = parent.builder().clone();
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3), Dimension::Static(4)]));
        let input_atom = builder.borrow_mut().add_input(input_type);
        let input_tracer = parent.tracer(input_atom, None);
        let output = Batch::batch(
            &parent,
            |x| {
                let bias = x.context().lift(Array::scalar(1.0))?;
                Ok(x + bias)
            },
            input_tracer,
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        )
        .unwrap();
        let output_atom = output.atom_id().unwrap();
        let program =
            builder.borrow().clone().build::<Array, Array>(vec![output_atom], Placeholder, Placeholder).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[3, 4] .
                let %1:f64[] = const
                    %2:f64[3, 4] = broadcast [output_type=f64[3, 4], output_axes=[]] %1
                    %3:f64[3, 4] = add %0 %2
                in (%3)
            "}
            .trim_end(),
        );
        let input = Array::matrix(3, 4, (0..12).map(|value| value as f64).collect());
        let output = program.interpret(input).unwrap();
        assert_eq!(output.to_f64s(), (0..12).map(|value| value as f64 + 1.0).collect::<Vec<_>>());
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
        let outputs = batch_projected_operation(&context, &ProjectedMemberOperation::<2>, &[input]).unwrap();
        assert_eq!(
            outputs,
            vec![ProjectedBatch {
                value: ProjectedProgramValue::Third(ProjectedMemberValue::<2>(7)),
                batch_axis: BatchAxis::replicated(),
            }],
        );
    }
}
