//! Contains machinery for _batching_ (i.e., _vectorizing_) computations.
//!
//! Batching turns a function written for individual array values into one that processes a logical batch. It does not
//! mechanically add an outer dimension to every intermediate value. Instead, each flowing value carries the physical
//! position of its logical batch axis, and each operation decides how to propagate, align, introduce, or remove that
//! axis. Replicated values carry no batch axis and are broadcast only when an operation needs alignment.
//!
//! ```text
//!  ┌───────────────────────────────────┐
//!  │ Values + Input Axis Specification │
//!  └─────────────────┬─────────────────┘
//!                    │ wrap each leaf as an array batch
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
//! [`BatchAxis`] describes a leaf as either batched at a signed physical axis or replicated, and a
//! [`BatchAxisSpecification`] supplies the logical axis size and optional name. Negative axes are normalized against
//! each array's rank, and every batched input must agree on the logical size.
//!
//! # Core Abstractions
//!
//! [`ArrayBatch`] pairs an underlying array value with its logical batch-axis position. Its alignment helpers can
//! broadcast a replicated value, move an existing axis, or align several operands to a common position. The wrapper's
//! type is the logical unbatched type (the underlying value retains the physical batch dimension).
//!
//! [`BatchingContext`] wraps a parent [`Context`] and records the active axis size and optional axis name.
//! [`BatchingTracer`] is the value flowing through a batched closure. It carries an [`ArrayBatch`] and delegates each
//! bind to the bound operation's [`BatchableOperation`] rule. Because the parent may be eager or staging, the same rule
//! can execute concrete arrays or build a transformed program.
//!
//! [`ElementwiseOperation`]s infer a common batch size, align operands to a common physical axis, bind the underlying
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
//! introducing the logical axis, and return [`BatchingError`] for invalid axis contracts rather than panicking.

use std::borrow::Cow;
use std::fmt::{Debug, Display};
use std::rc::Rc;

use thiserror::Error;

use ryft_macros::Parameter;

use crate::axes::{Axis, AxisError};
use crate::broadcasting::Broadcastable;
use crate::contexts::{Context, Domain, EagerContext, StagingContext, ValueResolution};
use crate::interpretation::InterpretableOperation;
use crate::macros::{check_builders, check_count};
use crate::operations::ElementwiseOperation;
use crate::operations::manipulation::{Broadcast, BroadcastOperation, Transpose, TransposeOperation};
use crate::parameters::{Parameter, ParameterError, Parameterized, ParameterizedFamily, Placeholder};
use crate::programs::ProgramError;
use crate::programs::operations::Operation;
use crate::programs::programs::Program;
use crate::programs::regions::{
    BindingRegionDriver, EmptyRegionDriver, RegionDriver, RegionRef, RegionReplayMappings, ReplayRegionDriver,
};
use crate::programs::types::{TypeError, Typed};
use crate::programs::values::Value;
use crate::sharding::{MeshAxisType, Sharding, ShardingDimension};
use crate::tracing::TracingContext;
use crate::types::{ArrayType, Dimension, Shape};

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

/// A batched value's mapped batch axis. [`BatchAxis::new`]`(k)` means that the value's batch dimension sits at physical
/// axis `k`. [`BatchAxis::replicated`] (the [`Default`]) means that the value is *replicated* (i.e., it carries no
/// physical dimension for the batch and is interpreted as the same value for every batch item). For example, a traced
/// constant in `batch(|x| x + 1)` is replicated, while `x` carries the mapped input axis. Runtime control flow
/// predicates may also be replicated, because a single predicate may select one branch for the whole batch while a
/// batch-varying predicate would need a dedicated batching rule. Note that replication is not limited to rank-0 (i.e.,
/// scalar) values. Any shaped constant or input is replicated when none of its physical dimensions indexes the batch.
///
/// This is the batch axis carried by an [`ArrayBatch`] and, during the batching transform, by the
/// [`Tracer`](crate::Tracer) metadata. Carrying it on the value itself lets the per-operation batching rules
/// route the mapped batch axis straight from the value in hand.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash, Parameter)]
pub struct BatchAxis(Option<Axis>);

impl BatchAxis {
    /// Creates a mapped [`BatchAxis`] at physical position `axis`.
    #[inline]
    pub fn new<A: Into<Axis>>(axis: A) -> Self {
        Self(Some(axis.into()))
    }

    /// Creates a mapped [`BatchAxis`] from an already-normalized physical position.
    #[inline]
    pub fn from_position(position: usize) -> Self {
        Self::new(position)
    }

    /// Creates a replicated or mapped [`BatchAxis`] from an already-normalized optional physical position.
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

/// Specification of a batch axis introduced by the batching transform that contains an optional explicit batch size
/// and an optional axis name that can be referenced by operations that support named axes. The batch size is normally
/// inferred from the inputs that are being batched. An explicit size can be provided to either pin it or to drive a
/// broadcasted batching transform whose batch size would otherwise be unobservable. The axis name makes the batch axis
/// addressable by name from collective operations inside the batched function body. [`BatchAxisSpecification`] converts
/// from the plain size forms, so call sites that do not need a name can pass `None`, `Some(size)`, or `size` directly.
/// For example:
///
/// ```ignore
/// domain.batch(f, input, input_axes, output_axes, None)?;                                     // Inferred, anonymous.
/// domain.batch(f, input, input_axes, output_axes, 8)?;                                        // Explicit, anonymous.
/// domain.batch(f, input, input_axes, output_axes, BatchAxisSpecification::named("devices"))?; // Inferred, named.
/// ```
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BatchAxisSpecification {
    /// Explicit batch size, or `None` to infer it from the inputs that are being batched.
    size: Option<usize>,

    /// Name that operations (e.g., collectives) can use to refer to it, or `None` for an anonymous axis.
    name: Option<String>,
}

impl BatchAxisSpecification {
    /// Creates a named [`BatchAxisSpecification`] with an explicit batch size.
    #[inline]
    pub fn new<N: Into<String>>(size: usize, name: N) -> Self {
        Self { size: Some(size), name: Some(name.into()) }
    }

    /// Creates a [`BatchAxisSpecification`] with an explicit batch size.
    #[inline]
    pub fn sized(size: usize) -> Self {
        Self { size: Some(size), name: None }
    }

    /// Creates a named [`BatchAxisSpecification`] whose batch size is inferred from the inputs that are being batched.
    #[inline]
    pub fn named<N: Into<String>>(name: N) -> Self {
        Self { size: None, name: Some(name.into()) }
    }

    /// Returns the explicit batch size of this [`BatchAxisSpecification`], or `None` when it is to be inferred
    /// from the inputs that are being batched.
    #[inline]
    pub fn size(&self) -> Option<usize> {
        self.size
    }

    /// Returns the name of this [`BatchAxisSpecification`] that operations (e.g., collectives) can use to refer
    /// to it, or `None` for an anonymous axis.
    #[inline]
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
}

impl From<Option<usize>> for BatchAxisSpecification {
    #[inline]
    fn from(size: Option<usize>) -> Self {
        Self { size, name: None }
    }
}

impl From<usize> for BatchAxisSpecification {
    #[inline]
    fn from(size: usize) -> Self {
        Self::sized(size)
    }
}

/// Value with [`ArrayType`] type that represents a _packed_ batch of arrays. [`ArrayBatch`] is the batching
/// representation for Ryft's batching/vectorization transform. It pairs a physical array value with a [`BatchAxis`]
/// that marks which of its dimensions indexes the batch items. A value is either *batched* (i.e., its physical type
/// carries the batch dimension) or *replicated*, meaning that it is shared unchanged across every batch item.
#[derive(Clone, Debug, PartialEq, Parameter)]
pub struct ArrayBatch<V> {
    /// Physical array type of `value`. When the value is batched this type includes the mapped batch dimension at
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
        // A possibly-negative mapped axis is normalized against the physical rank, following Python/JAX indexing:
        // valid axes lie in `[-rank, rank)`, with `-1` denoting the final axis.
        let batch_axis = match batch_axis.into().axis() {
            Some(axis) => match axis.normalize(r#type.rank()) {
                Ok(position) => BatchAxis::from_position(position),
                Err(_) => return Err(BatchingError::BatchAxisOutOfBounds { r#type: Box::new(r#type), axis }),
            },
            None => BatchAxis::replicated(),
        };
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

    /// Returns the canonical nonnegative physical position of this [`ArrayBatch`]'s mapped [`BatchAxis`].
    /// [`ArrayBatch::new`] normalizes signed declarations before storing them, and so internal batching rules
    /// can use this index directly.
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
    pub fn unbatched_type(&self) -> ArrayType {
        let Some(axis) = self.batch_axis_position() else {
            return self.r#type.clone();
        };

        // This is a logical transform view, not an ordinary rank-changing array operation. In particular, explicit
        // placement on the transform-owned mapped dimension describes how the batch is distributed; it is not part
        // of each batch item's placement. `ArrayType::without_dimension` correctly rejects dropping such placement
        // information for regular array dimensions, and so it cannot be used for these batching-specific semantics.
        let mut dimensions = self.r#type.shape().dimensions().to_vec();
        dimensions.remove(axis);
        let sharding = self.r#type.sharding().map(|sharding| {
            let mut sharding_dimensions = sharding.dimensions().to_vec();
            let removed_dimension = sharding_dimensions.remove(axis);
            let mut varying_manual_axes = sharding.varying_manual_axes().clone();

            // Manual axes remain semantically visible after their ranked dimension is removed because values may
            // still vary across those axes. All other mesh axes are intentionally omitted (they placed the batch
            // dimension itself, which is outside the logical per-item type).
            if let ShardingDimension::Sharded(axis_names) = removed_dimension {
                varying_manual_axes.extend(
                    axis_names.into_iter().filter(|name| sharding.mesh().axis_type(name) == Some(MeshAxisType::Manual)),
                );
            }

            Sharding {
                mesh: sharding.mesh().clone(),
                dimensions: sharding_dimensions,
                unreduced_axes: sharding.unreduced_axes().clone(),
                reduced_axes: sharding.reduced_axes().clone(),
                varying_manual_axes,
            }
        });

        ArrayType {
            data_type: self.r#type.data_type(),
            shape: Shape::new(dimensions),
            layout: None,
            sharding,
            memory: self.r#type.memory(),
        }
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
    ///     physical output rank (e.g., `-1` denotes the final output axis).
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
        V: Broadcast,
    {
        if !self.batch_axis().is_replicated() {
            return Err(BatchingError::MisalignedBatchAxes {
                message: "'ArrayBatch::broadcast' expects a replicated operand but received a batched value"
                    .to_string(),
            });
        }
        // The insertion position is normalized against the physical output rank (i.e., the per-item rank plus the
        // inserted batch dimension).
        let axis = axis.into();
        let per_item_type = self.unbatched_type();
        let output_rank = per_item_type.rank() + 1;
        let position = axis
            .normalize(output_rank)
            .map_err(|_| BatchingError::BatchAxisOutOfBounds { r#type: Box::new(self.r#type.clone()), axis })?;
        let mut physical_type = per_item_type.with_inserted_dimension(position, Dimension::Static(axis_size))?;
        if let Some(sharding) = per_item_type.sharding() {
            physical_type.sharding = Some(
                sharding
                    .with_inserted_dimension(position, axis_sharding)
                    .map_err(|error| BatchingError::MisalignedBatchAxes { message: error.to_string() })?,
            );
        }
        let output_axes = (0..per_item_type.rank())
            .map(|dimension| if dimension < position { dimension } else { dimension + 1 })
            .collect::<Vec<_>>();
        let broadcasted = self.value().clone().broadcast(physical_type.clone(), output_axes.as_slice())?;
        ArrayBatch::new(physical_type, broadcasted, axis)
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
    ///     normalized against the (unchanged) physical rank (e.g., `-1` denotes the final axis).
    pub fn move_axis<A: Into<Axis>>(&self, axis: A) -> Result<Self, BatchingError>
    where
        V: Transpose,
    {
        let Some(current_axis) = self.batch_axis_position() else {
            return Ok(self.clone());
        };
        // The target is normalized against the unchanged physical rank.
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
    /// `batching.matchaxis`, used by rules whose inputs must agree on one physical batch axis (e.g., `pad`,
    /// `concatenate`, etc.).
    ///
    /// # Parameters
    ///
    ///   - `axis`: Possibly-negative position the batch axis should occupy in the output, normalized against the
    ///     physical output rank (e.g., `-1` denotes the final output axis).
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
        V: Broadcast + Transpose,
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
    /// [`BatchingError::MismatchedOutputAxes`]. Signed declarations are normalized against the resulting physical
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
        V: Broadcast + Transpose,
    {
        // Signed declaration normalization is owned by the delegates: `move_axis` normalizes against the (unchanged)
        // physical rank and `broadcast` against the physical output rank gaining the batch dimension.
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
    pub fn sharding_for_inputs(inputs: &[Self]) -> Result<ShardingDimension, ProgramError> {
        // A `ShardingDimension` identifies mesh axes only by name, so equal dimensions are meaningful only when their
        // owning shardings use the same mesh. Establish that common mesh before combining any dimension placements.
        // Replicated batches and values without sharding metadata do not constrain it.
        inputs
            .iter()
            .filter_map(|input| {
                input.batch_axis_position()?;
                Some(input.r#type.sharding()?.mesh())
            })
            .try_fold(None, |mesh, current_mesh| -> Result<_, ProgramError> {
                match mesh {
                    Some(mesh) if mesh != current_mesh => Err(BatchingError::MisalignedBatchAxes {
                        message: format!("mismatched batch axis sharding meshes: {mesh:?} vs {current_mesh:?}"),
                    }
                    .into()),
                    Some(mesh) => Ok(Some(mesh)),
                    None => Ok(Some(current_mesh)),
                }
            })?;

        // Join the mapped-axis placements from least to most specific. An unconstrained dimension accepts any concrete
        // decision, and a replicated dimension can be normalized to a sharded placement. Matching sharded placements
        // agree, but two different sharded placements are ambiguous because neither input provides a principled choice
        // of target placement. Keeping this join order-independent is important because input ordering is semantic only
        // to the operation, not to batch-placement selection.
        let dimension = inputs
            .iter()
            .filter_map(|input| Some(input.r#type().sharding()?.dimensions()[input.batch_axis_position()?].clone()))
            .try_fold(
                None,
                |folded_dimension, current_dimension| -> Result<Option<ShardingDimension>, ProgramError> {
                    match folded_dimension {
                        Some(folded_dimension) if folded_dimension == current_dimension => Ok(Some(folded_dimension)),
                        Some(ShardingDimension::Unconstrained) => Ok(Some(current_dimension)),
                        Some(folded_dimension) if current_dimension == ShardingDimension::Unconstrained => {
                            Ok(Some(folded_dimension))
                        }
                        Some(ShardingDimension::Replicated) => Ok(Some(current_dimension)),
                        Some(folded_dimension) if current_dimension == ShardingDimension::Replicated => {
                            Ok(Some(folded_dimension))
                        }
                        Some(folded_dimension) => Err(BatchingError::MisalignedBatchAxes {
                            message: format!(
                                "mismatched batch axis sharding: {folded_dimension} vs {current_dimension}"
                            ),
                        }
                        .into()),
                        None => Ok(Some(current_dimension)),
                    }
                },
            )?;
        Ok(dimension.unwrap_or(ShardingDimension::Replicated))
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

/// Provides [`Instruction`](crate::Instruction)-scoped access to the nested [`Region`](crate::Region)s attached to
/// an [`Operation`] being batched. Every [`BatchableOperation`] application receives a [`BatchingDriver`]. The driver
/// keeps any attached regions borrowed and re-enters batching with the durable [`BatchingContext`] supplied by the
/// operation rule. [`RegionDriver`] provides its structural region access, while this trait adds batching-specific
/// recursion. Region-free applications expose a region count of zero through the same contract.
pub trait BatchingDriver<C: Context<Type = ArrayType>>: RegionDriver<C::Constant, C::Operation> {
    /// Batches the region at `index` over the provided batched values by re-entering the active batching transform.
    fn batch_region(
        &self,
        context: &BatchingContext<C>,
        index: usize,
        inputs: Vec<ArrayBatch<C::Value>>,
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError>;

    /// Batches `region` structurally at the provided input batch axes and output-axes policy, returning the rewritten
    /// standalone program and its inferred output batch axes.
    fn batch_program(
        &self,
        context: &BatchingContext<C>,
        region: RegionRef<'_, C::Constant, C::Operation>,
        input_axes: &[BatchAxis],
        output_axes_policy: ProgramBatchingOutputAxesPolicy,
    ) -> Result<(Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>, Vec<BatchAxis>), BatchingError>;
}

impl<C: Context<Type = ArrayType>> BatchingDriver<C> for EmptyRegionDriver {
    #[inline]
    fn batch_region(
        &self,
        _context: &BatchingContext<C>,
        _index: usize,
        _inputs: Vec<ArrayBatch<C::Value>>,
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        Err(ProgramError::MalformedProgram("empty region driver cannot batch a region".to_string()).into())
    }

    #[inline]
    fn batch_program(
        &self,
        _context: &BatchingContext<C>,
        _region: RegionRef<'_, C::Constant, C::Operation>,
        _input_axes: &[BatchAxis],
        _output_axes_policy: ProgramBatchingOutputAxesPolicy,
    ) -> Result<(Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>, Vec<BatchAxis>), BatchingError>
    {
        Err(ProgramError::MalformedProgram("empty region driver cannot batch a program".to_string()).into())
    }
}

/// [`BatchingDriver`] scoped to one [`Operation`] application. It borrows the application's complete region driver,
/// which preserves the operation-defined ordering of owned regions, borrowed regions, and shared callees without
/// materializing a combined region collection. Recursive requests re-enter the active batching transform or batch a
/// selected region structurally.
struct RecursiveBatchingDriver<'r, D> {
    /// Application-scoped [`RegionDriver`], in [`Operation`]-defined order.
    driver: &'r D,
}

impl<V: Value<Type = ArrayType>, O: Operation<ArrayType>, D: RegionDriver<V, O>> RegionDriver<V, O>
    for RecursiveBatchingDriver<'_, D>
{
    #[inline]
    fn regions<'r>(&'r self) -> impl Iterator<Item = RegionRef<'r, V, O>>
    where
        V: 'r,
        O: 'r,
    {
        self.driver.regions()
    }
}

impl<C: Context<Type = ArrayType>, D: RegionDriver<C::Constant, C::Operation>> BatchingDriver<C>
    for RecursiveBatchingDriver<'_, D>
where
    C::Operation: BatchableOperation<C>
        + BatchableOperation<TracingContext<C::Constant, C::Operation>>
        + From<TransposeOperation>
        + From<BroadcastOperation>,
{
    #[inline]
    fn batch_region(
        &self,
        context: &BatchingContext<C>,
        index: usize,
        inputs: Vec<ArrayBatch<C::Value>>,
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        context.batch_region(self.region(index)?, inputs)
    }

    #[inline]
    fn batch_program(
        &self,
        context: &BatchingContext<C>,
        region: RegionRef<'_, C::Constant, C::Operation>,
        input_axes: &[BatchAxis],
        output_axes_policy: ProgramBatchingOutputAxesPolicy,
    ) -> Result<(Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>, Vec<BatchAxis>), BatchingError>
    {
        region.batched(context.axis_size(), context.axis_sharding().clone(), input_axes, output_axes_policy)
    }
}

/// Represents [`Operation`]s that can be batched (i.e., vectorized).
///
/// The trait is parameterized by the parent [`Context`] `C` that owns the physical values flowing through the batching
/// transform, and every rule receives the active durable [`BatchingContext`] plus a [`BatchingDriver`]. Ordinary rules
/// lift their operation to its batch-carrying inputs and then execute or stage the lifted work through
/// `context.parent()` (typically via [`InterpretableBatchableOperation::interpret_with_batch_axes`]), so an eager
/// parent interprets it immediately while a staging parent stages it into the enclosing trace. Rules whose semantics
/// depend on the active transform frame (e.g., named-axis collectives) inspect [`BatchingContext::axis_name`] and
/// [`BatchingContext::axis_size`] directly, and recursive higher-order rules replay their nested programs through the
/// same contract. Consequently, invoking a batching rule always requires an active [`BatchingContext`].
///
/// # Deriving Batchable Operation Enums
///
/// `#[derive(Operation)]` generates a [`BatchableOperation`] dispatcher when the enum specifies
/// `#[ryft(dispatch(batching))]`. It follows the operation derivation's enum-shape and type-inference rules
/// and generates:
///
///   - A dispatcher at `BatchableOperation<C>`, generic over the parent [`Context`] `C`, that forwards the active
///     [`BatchingContext`] to every variant's own rule. One dispatcher covers eager and staging parents alike, because
///     the parent/active distinction lives in each rule's body rather than in dispatch.
///   - Per-variant `Payload: BatchableOperation<C>` predicates that transport each rule's own capability requirements
///     to the use site. Nested programs batch structurally through [`Program::batched`], requested by higher-order
///     rules through their active [`BatchingDriver`], whose concrete implementation establishes the finite
///     program-level bounds at its construction site.
pub trait BatchableOperation<C: Context<Type = ArrayType>>: Operation<ArrayType> {
    /// Applies this operation to packed batched inputs, returning batched outputs with the resulting batch axes.
    /// `context` borrows the durable [`BatchingContext`] for the transform level being applied. `driver` exposes the
    /// current operation application's regions and has a region count of zero for region-free applications. Packed
    /// physical values in `inputs` and the returned outputs are owned by `context.parent()`.
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
    ///   - **Parent-Owned Physical Work:** Ordinary rules must execute or stage lifted physical work through
    ///     `context.parent()` and return parent-owned values; only rules keyed on the active frame's axis metadata
    ///     inspect the [`BatchingContext`] itself.
    ///
    /// Note that in order to be able to provide [`BatchableOperation`] implementations for operation families that
    /// select the generated batching dispatcher, it is a common convention for operations that can be part of such
    /// operation families to implement this trait even if they do not support batching and to have this  function
    /// simply return a [`BatchingError::UnsupportedOperation`] error.
    fn batch<D: BatchingDriver<C>>(
        &self,
        context: &BatchingContext<C>,
        driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError>;
}

// Blanket `BatchableOperation` implementation for any `ElementwiseOperation`, so per-operation `BatchableOperation`
// implementations do not have to be written for elementwise primitives (e.g., `ZeroLike`, `OneLike`, `Add`, `Sub`,
// `Mul`, `Div`, `Neg`, `Sin`, `Cos`, `Select`, etc.). Operations with non-trivial axis arithmetic (e.g., `Dot`,
// `Transpose`, `Reshape`, etc.) keep their explicit implementations. Coherence is preserved because none of those
// types implement `ElementwiseOperation`. The rule follows JAX's `defbroadcasting` policy where every input is
// broadcast to the common batched physical shape before the operation is applied, so the value-level primitive only
// ever sees inputs that agree on shape. When no input is mapped there is no batch axis to thread, and so the inputs
// are interpreted as given and every output is replicated. Otherwise, mapped inputs retain the first mapped input's
// physical position when that position is valid for every mapped operand. For mixed ranks where it is not, they are
// temporarily realigned to leading physical axis `0`. Every input is then broadcast to the common per-item shape with
// the batch axis inserted there. Operands whose per-item shapes are not broadcast-compatible are left at their
// batch-axis-inserted shapes so the operation surfaces its own shape error.
impl<C: Context<Type = ArrayType, Value: Broadcast + Transpose>, O: ElementwiseOperation + InterpretableOperation<C>>
    BatchableOperation<C> for O
{
    fn batch<D: BatchingDriver<C>>(
        &self,
        context: &BatchingContext<C>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        let input_axes = inputs.iter().map(ArrayBatch::batch_axis).collect::<Vec<_>>();

        // No input carries the batch axis. Interpret the inputs as given and report every output replicated.
        // Any per-item shape broadcasting between replicated inputs is the operation's own concern.
        let Some(output_batch_axis_position) = inputs.iter().find_map(ArrayBatch::batch_axis_position) else {
            let physical_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
            let output_count = Operation::infer_output_types(self, physical_types.as_slice(), &[])?.len();
            return self.interpret_with_batch_axes(context, inputs, &vec![BatchAxis::replicated(); output_count]);
        };

        // Preserve the first mapped input's natural position when every mapped input can represent it. Otherwise, use
        // a leading internal axis, which is valid regardless of differences in logical per-item rank, and restore the
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

        // Realign every mapped input's batch axis to the common position, then broadcast every input to the common
        // batched physical shape.
        let axis_size = ArrayBatch::common_batch_size(inputs)?.expect("a mapped input pins the batch size");
        let axis_sharding = ArrayBatch::sharding_for_inputs(inputs)?;
        let unbatched_types = inputs
            .iter()
            .map(|input| {
                let mut unbatched_type = input.unbatched_type();
                if let (Some(sharding), ShardingDimension::Sharded(axis_names)) =
                    (unbatched_type.sharding.as_mut(), &axis_sharding)
                {
                    let varying_manual_axes = axis_names
                        .iter()
                        .filter(|name| sharding.mesh().axis_type(name.as_str()) == Some(MeshAxisType::Manual))
                        .cloned()
                        .collect::<Vec<_>>();
                    sharding.varying_manual_axes.extend(varying_manual_axes);
                }
                unbatched_type
            })
            .collect::<Vec<_>>();
        let inputs = inputs.iter().map(|input| input.move_axis(batch_axis)).collect::<Result<Vec<_>, _>>()?;

        // The common per-item shape every input broadcasts to, or `None` when the per-item shapes are not
        // broadcast-compatible, in which case each input keeps its own per-item shape below and the operation
        // reports the incompatibility itself.
        let common_unbatched_type = Broadcastable::broadcasted(unbatched_types.as_slice()).ok();
        let broadcasted_inputs = inputs
            .iter()
            .zip(input_axes.iter())
            .zip(unbatched_types.iter())
            .map(|((input, input_axis), unbatched_type)| -> Result<ArrayBatch<C::Value>, BatchingError> {
                // The target physical type is the common per-item shape (falling back to this input's own when
                // incompatible) carrying this input's own data type (e.g. a Boolean select condition stays Boolean
                // against numeric branches), with the batch axis inserted at `batch_axis`.
                let mut target_type = common_unbatched_type.as_ref().unwrap_or(unbatched_type).clone();
                target_type.data_type = unbatched_type.data_type();
                let mut physical_type =
                    target_type.with_inserted_dimension(batch_axis_position, Dimension::Static(axis_size))?;
                if let Some(sharding) = target_type.sharding() {
                    physical_type.sharding = Some(
                        sharding
                            .with_inserted_dimension(batch_axis_position, axis_sharding.clone())
                            .map_err(|error| BatchingError::MisalignedBatchAxes { message: error.to_string() })?,
                    );
                }
                if physical_type == *input.r#type() {
                    return Ok(input.clone());
                }

                // Map each aligned dimension to its position in the target. The batch axis maps to itself, and the
                // per-item dimensions right-align within the target's per-item dimensions, skipping the inserted
                // batch axis.
                let target_unbatched_rank = physical_type.rank() - 1;
                let is_mapped = !input_axis.is_replicated();
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
                    .collect::<Vec<_>>();
                let broadcasted = input.value().clone().broadcast(physical_type.clone(), output_axes.as_slice())?;
                ArrayBatch::new(physical_type, broadcasted, batch_axis)
            })
            .collect::<Result<Vec<_>, _>>()?;

        // The lifted operation is the original applied to the batch-carrying inputs. Every output takes the common
        // batch axis. Broadcast-incompatible per-item shapes surface the operation's own shape error at this inference.
        let input_types = broadcasted_inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        let output_count = Operation::infer_output_types(self, input_types.as_slice(), &[])?.len();
        let output_batch_axes = vec![BatchAxis::new(batch_axis); output_count];
        self.interpret_with_batch_axes(context, &broadcasted_inputs, &output_batch_axes)?
            .into_iter()
            .map(|output| output.move_axis(output_batch_axis_position))
            .collect()
    }
}

/// Policy for choosing a batched [`Program`]'s output axes. Program batching always replays the program over physical
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

/// Capability to interpret an [`Operation`] on batch-carrying inputs and repackage its outputs as [`ArrayBatch`]es.
/// This is the shared application path the per-operation [`BatchableOperation`] rules use once they have lifted an
/// operation to its batch-carrying inputs. It centralizes the physical-value ownership invariant of the batching
/// transform: the lifted operation is always interpreted against `context.parent()`, which owns the packed physical
/// values, and never against the [`BatchingContext`] itself. Every [`InterpretableOperation`] over the parent's
/// values gets it for free through the blanket implementation below, so an operation earns it directly from its
/// interpretation rule.
pub trait InterpretableBatchableOperation<C: Context<Type = ArrayType>> {
    /// Interprets this operation on the *unpacked* values of batched `inputs` against `context.parent()` and
    /// repackages each output as an [`ArrayBatch`] carrying the corresponding `output_batch_axes` entry.
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
        context: &BatchingContext<C>,
        inputs: &[ArrayBatch<C::Value>],
        output_batch_axes: &[BatchAxis],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError>;
}

impl<C: Context<Type = ArrayType>, O: InterpretableOperation<C>> InterpretableBatchableOperation<C> for O {
    fn interpret_with_batch_axes(
        &self,
        context: &BatchingContext<C>,
        inputs: &[ArrayBatch<C::Value>],
        output_batch_axes: &[BatchAxis],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        // Every `InterpretableOperation` over the parent context's values is an `InterpretableBatchableOperation`.
        // The batched interpretation unpacks the input values, interprets the operation once through `interpret`
        // against the parent context (which owns those physical values), and repackages each output with its
        // requested `BatchAxis`.
        if inputs.is_empty() {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
        }
        let input_values = inputs.iter().map(|input| input.value().clone()).collect::<Vec<_>>();
        let output_values = self.interpret(&context.parent().clone(), &EmptyRegionDriver, input_values.as_slice())?;
        check_count!("output", output_values, output_batch_axes.len(), ProgramError);
        output_values
            .into_iter()
            .zip(output_batch_axes.iter().copied())
            .map(|(value, axis)| ArrayBatch::new(value.r#type().into_owned(), value, axis))
            .collect()
    }
}

/// [`Context`] that is used for batching a computation by introducing exactly one batch dimension at a specified axis.
/// [`BatchingContext`] is the active context for one level of batching. It runs the function being batched against
/// logical per-item [`ArrayType`]s while leaving the runtime value type of the staged program equal to the parent
/// context's value type. [`Operation`]s staged through this context are lifted through their [`BatchableOperation`]
/// implementations at bind time. The lifted operation is then staged into the parent context, so nested transforms
/// compose by wrapping contexts rather than by making each active transform pretend to be a backend domain.
#[derive(Debug, Clone)]
pub struct BatchingContext<C> {
    /// [`Context`] that this [`BatchingContext`] is nested into.
    parent: C,

    /// Dimension of the new batch axis.
    axis_size: usize,

    /// Optional name for the new batch axis that enables [`Operation`]s (e.g., collective operations)
    /// to address this axis by name.
    axis_name: Option<String>,

    /// Sharding placement of the transform-owned mapped dimension.
    axis_sharding: ShardingDimension,
}

impl<C> BatchingContext<C> {
    /// Creates a new [`BatchingContext`] with an unnamed and [`ShardingDimension::Replicated`] mapped axis.
    ///
    /// # Parameters
    ///
    ///   - `parent`: [`Context`] into which batched operations are lifted.
    ///   - `axis_size`: Dimension of the transform-owned mapped dimension.
    #[inline]
    pub fn new(parent: C, axis_size: usize) -> Self {
        Self { parent, axis_size, axis_name: None, axis_sharding: ShardingDimension::Replicated }
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

    /// Returns the size of the new batch axis.
    #[inline]
    pub fn axis_size(&self) -> usize {
        self.axis_size
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

    /// Replays `region` [`Instruction`](crate::Instruction) by [`Instruction`](crate::Instruction) through this
    /// [`BatchingContext`], dispatching every instruction's [`BatchableOperation`] rule with application-scoped access
    /// to its attached regions, and returns the region output batches. This is the batching counterpart of
    /// [`Program::interpret_in_context`].
    #[inline]
    pub(crate) fn batch_region(
        &self,
        region: RegionRef<'_, C::Constant, C::Operation>,
        inputs: Vec<ArrayBatch<C::Value>>,
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError>
    where
        C: Context<Type = ArrayType>,
        C::Operation: BatchableOperation<C>
            + BatchableOperation<TracingContext<C::Constant, C::Operation>>
            + From<TransposeOperation>
            + From<BroadcastOperation>,
    {
        let region_mappings = RegionReplayMappings::new();
        region.interpret_with(
            inputs,
            |_, constant| Ok(ArrayBatch::replicated(self.parent().lift(constant.clone())?)),
            |instruction, instruction_inputs| {
                let regions = ReplayRegionDriver::new(region, instruction.regions(), &region_mappings)?;
                instruction
                    .operation()
                    .batch(self, &RecursiveBatchingDriver { driver: &regions }, instruction_inputs)
            },
        )
    }
}

impl<C: Context<Type = ArrayType>> Domain for BatchingContext<C> {
    type Type = ArrayType;
    type Value = BatchingTracer<C>;
    type Constant = C::Constant;
    type Operation = C::Operation;
}

impl<C: Context<Type = ArrayType>> Context for BatchingContext<C>
where
    C::Operation: BatchableOperation<C>
        + BatchableOperation<TracingContext<C::Constant, C::Operation>>
        + From<TransposeOperation>
        + From<BroadcastOperation>,
{
    #[inline]
    fn lift(&self, constant: C::Constant) -> Result<BatchingTracer<C>, ProgramError> {
        // Lifts a constant by lifting it in the parent context and replicating it across the batch.
        Ok(BatchingTracer::new(self.clone(), ArrayBatch::replicated(self.parent().lift(constant)?)))
    }

    #[inline]
    fn bind<P: Into<Self::Operation>, D: BindingRegionDriver<Self::Constant, Self::Operation>>(
        &self,
        operation: P,
        driver: D,
        inputs: &[BatchingTracer<C>],
    ) -> Result<Vec<BatchingTracer<C>>, ProgramError> {
        // Binding routes the operation through its `BatchableOperation` implementation against the batch-carrying
        // inputs. The implementation dispatches primitive work through the parent context, executing eagerly under an
        // eager parent or staging into an enclosing trace under a staging parent, and axis-referencing work (e.g.,
        // collectives) through this batching context, and so multi-operation lowering (e.g., a batch-varying
        // `Instruction` becoming two branches plus a per-item select instruction) emerges automatically.
        let operation = operation.into();
        let input_batches = inputs.iter().map(|input| input.batch().clone()).collect::<Vec<_>>();
        let batching_driver = RecursiveBatchingDriver { driver: &driver };
        let output_batches = operation.batch(self, &batching_driver, input_batches.as_slice())?;
        Ok(output_batches.into_iter().map(|batch| BatchingTracer::new(self.clone(), batch)).collect())
    }

    #[inline]
    fn is_eager(&self) -> bool {
        self.parent().is_eager()
    }

    #[inline]
    fn resolve(&self, value: &BatchingTracer<C>) -> ValueResolution<C::Constant> {
        self.parent().resolve(value.batch().value())
    }
}

/// Batch-carrying value flowing through a [`BatchingContext`]. The function being batched operates on
/// [`BatchingTracer`]s directly. Each operation dispatches through the stamped context via [`Context::bind`],
/// which applies the operation's [`BatchableOperation`] implementation against the parent context. Over an eager
/// parent context the packed value is concrete and so the closure sees real per-item values (e.g., it can branch
/// on a replicated value, print it, etc.). Over a staging parent context it is a [`Tracer`](crate::Tracer) staged into
/// the enclosing trace. Its [`Typed`] view is the *logical* per-item [`ArrayType`] (i.e., with the batch axis removed),
/// while the inner [`ArrayBatch`] carries the *physical* type and [`BatchAxis`].
#[derive(Clone, Parameter)]
pub struct BatchingTracer<C: Context> {
    /// [`BatchingContext`] this value flows through, used to dispatch operations that involve it.
    context: BatchingContext<C>,

    /// [`ArrayBatch`] that corresponds to the batched underlying values.
    batch: ArrayBatch<C::Value>,
}

impl<C: Context<Type = ArrayType>> BatchingTracer<C> {
    /// Creates a new [`BatchingTracer`].
    #[inline]
    pub fn new(context: BatchingContext<C>, batch: ArrayBatch<C::Value>) -> Self {
        Self { context, batch }
    }

    /// Returns the [`BatchingContext`] this value flows through.
    #[inline]
    pub fn context(&self) -> &BatchingContext<C> {
        &self.context
    }

    /// Returns the [`ArrayBatch`] that corresponds to the batched underlying values.
    #[inline]
    pub fn batch(&self) -> &ArrayBatch<C::Value> {
        &self.batch
    }
    /// Returns this [`BatchingTracer`]'s mapped [`BatchAxis`].
    #[inline]
    pub fn batch_axis(&self) -> BatchAxis {
        self.batch.batch_axis()
    }

    /// Consumes this value and returns the underlying [`ArrayBatch`].
    #[inline]
    pub fn into_batch(self) -> ArrayBatch<C::Value> {
        self.batch
    }
}

impl<C: Context<Type = ArrayType, Value: PartialEq>> PartialEq for BatchingTracer<C> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        // A batch-carrying value compares by its packed value (through that value's own `PartialEq`, which is
        // identity-shaped for tracer-valued parents) and its batch axis, ignoring the stamped context. Consumers such
        // as the scan/while loop-invariance fixed points of partial evaluation compare flowing values across replay
        // rounds to detect passthrough, and a batched value passes through exactly when its packed value does on the
        // same axis.
        self.batch.batch_axis() == other.batch.batch_axis() && self.batch.value() == other.batch.value()
    }
}

impl<C: Context> Debug for BatchingTracer<C> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("BatchingTracer").field("batch", &self.batch).finish()
    }
}

impl<C: Context<Type = ArrayType>> Display for BatchingTracer<C> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.batch)
    }
}

impl<C: Context<Type = ArrayType>> Typed for BatchingTracer<C> {
    type Type = ArrayType;

    #[inline]
    fn r#type(&self) -> Cow<'_, ArrayType> {
        Cow::Owned(self.batch.unbatched_type())
    }
}

impl<C: Context<Type = ArrayType>> Value for BatchingTracer<C> {
    type DispatchDomain = BatchingContext<C>;
    type ExecutionDomain = BatchingContext<C>;

    #[inline]
    fn dispatch_domain(&self) -> BatchingContext<C> {
        self.context().clone()
    }

    #[inline]
    fn execution_domain(&self) -> BatchingContext<C> {
        self.context().clone()
    }
}

impl<
    V: Value<Type = ArrayType>,
    O: BatchableOperation<TracingContext<V, O>> + From<TransposeOperation> + From<BroadcastOperation>,
> RegionRef<'_, V, O>
{
    /// Batches this borrowed [`Region`](crate::Region) so that the resulting program operates over batched inputs
    /// along the specified [`BatchAxis`]s. Staged higher-order [`BatchableOperation`] implementations use this function
    /// to batch captured programs *without* concretizing any batch-item values, so that batched control-flow and
    /// custom-derivative structure can be staged back into the enclosing trace. This function works by replying this
    /// program through a [`BatchingContext`] over a fresh [`TracingContext`], lifting every instruction through its
    /// [`BatchableOperation`] rule, and the resulting staged program is extracted together with the requested
    /// [`ProgramBatchingOutputAxesPolicy`].
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
    ) -> Result<(Program<V, O, Vec<V>, Vec<V>>, Vec<BatchAxis>), BatchingError> {
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
                            // A possibly-negative mapped axis is normalized against the physical input rank (i.e.,
                            // with the inserted batch dimension counted). Valid axes lie in `[-rank, rank)`, with `-1`
                            // denoting the final axis.
                            let physical_rank = unbatched_type.rank() + 1;
                            let position = axis.normalize(physical_rank).map_err(|_| {
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
                        &RecursiveBatchingDriver { driver: &regions },
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
                // tracers, then report the normalized physical position stored by `ArrayBatch`.
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
        Ok((program, output_axes))
    }
}

impl<
    V: Value<Type = ArrayType>,
    O: BatchableOperation<TracingContext<V, O>> + From<TransposeOperation> + From<BroadcastOperation>,
> Program<V, O, Vec<V>, Vec<V>>
{
    /// Batches this [`Program`] over the provided input axes. Refer to [`RegionRef::batched`] for the complete
    /// transformation semantics.
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
    ) -> Result<(Self, Vec<BatchAxis>), BatchingError> {
        self.entry_region_ref().batched(axis_size, axis_sharding, input_batch_axes, output_axes_policy)
    }
}

/// Extension trait that exposes the batching transform as a method on any [`Context`] over [`ArrayType`]. Refer to the
/// documentation of the [`batch`] function for information on what the batching transform does and how to use it. This
/// trait serves the call sites that must name the [`Context`] explicitly (most notably inputs with no values to recover
/// a context from).
pub trait Batch: Context<Type = ArrayType, Value: Broadcast + Transpose> {
    /// Batches `function` over the mapped axes of `input`, with this [`Context`] executing (or staging) the batched
    /// operations. Refer to the documentation of the [`batch`] function for information on the batching transform and
    /// its arguments. Unlike that function, this method also serves inputs with no leaf values (i.e., that are empty),
    /// provided that `batch_axis` supplies an explicit batch size.
    fn batch<
        F: FnOnce(I::To<BatchingTracer<Self>>) -> Result<O, ProgramError>,
        I: Parameterized<Self::Value, Family: ParameterizedFamily<BatchAxis> + ParameterizedFamily<BatchingTracer<Self>>>,
        O: Parameterized<BatchingTracer<Self>, Family: ParameterizedFamily<BatchAxis> + ParameterizedFamily<Self::Value>>,
        InputBatchAxes: Parameterized<BatchAxis>,
        OutputBatchAxes: Parameterized<BatchAxis>,
        Specification: Into<BatchAxisSpecification>,
    >(
        &self,
        function: F,
        input: I,
        input_batch_axes: InputBatchAxes,
        output_batch_axes: OutputBatchAxes,
        batch_axis: Specification,
    ) -> Result<O::To<Self::Value>, BatchingError> {
        let batch_axis = batch_axis.into();
        let input_structure = input.parameter_structure();
        let inputs = input.into_parameters().collect::<Vec<_>>();
        if inputs.is_empty() && batch_axis.size().is_none() {
            return Err(BatchingError::EmptyBatch);
        }

        // Broadcast the caller's `input_batch_axes` into the input parameter structure. A single `BatchAxis` leaf fills
        // every input leaf, a matching structure gives one axis per leaf, and a smaller compatible structure broadcasts
        // based on its prefixes. A structure that cannot fill the input surfaces as a `ParameterError`.
        let input_batch_axes = input_batch_axes
            .broadcast_to_parameter_structure::<I::To<BatchAxis>>(input_structure.clone())?
            .into_parameters()
            .collect::<Vec<_>>();

        // Pack each input parent value with its mapped batch axis at its physical type (`ArrayBatch::new` validates
        // that each mapped axis is in bounds). A value already produced by an enclosing `batch` keeps that level's
        // axis (its own `BatchingTracer` carries it), and so nested maps thread through with no side table. Fresh
        // inputs simply flow the receiver's own value representation.
        let inputs = inputs
            .into_iter()
            .zip(input_batch_axes)
            .map(|(input, input_batch_axis)| ArrayBatch::new(input.r#type().into_owned(), input, input_batch_axis))
            .collect::<Result<Vec<_>, _>>()?;

        // The batch size is the explicit `batch_axis` size when one is provided and the common size of the mapped
        // inputs otherwise. The two must agree when both are present, and at least one of them must pin the size.
        let batch_size = match (batch_axis.size(), ArrayBatch::common_batch_size(&inputs)?) {
            (Some(explicit_size), Some(common_size)) if explicit_size != common_size => {
                return Err(BatchingError::MismatchedBatchSizes { expected: explicit_size, actual: common_size });
            }
            (explicit_size, common_size) => explicit_size.or(common_size).ok_or(BatchingError::EmptyBatch)?,
        };

        // Create a `BatchingContext`, construct the batched function input, and invoke the function with it. Binds
        // inside the closure fold through the receiver directly and so, an eager context interprets each immediately,
        // while a staging context stages it into the enclosing trace, whose own drain surfaces any deferred error.
        let axis_sharding = ArrayBatch::sharding_for_inputs(inputs.as_slice())?;
        let context = BatchingContext::new(self.clone(), batch_size)
            .with_axis_name(batch_axis.name().map(String::from))
            .with_axis_sharding(axis_sharding);
        let inputs = inputs
            .into_iter()
            .map(|batch| {
                // Every mapped input with sharding metadata must use the common batch placement selected above. A
                // replicated batch or an input without sharding metadata imposes no physical placement to normalize.
                let normalization = batch.batch_axis_position().and_then(|position| {
                    let sharding = batch.r#type.sharding()?;
                    (sharding.dimensions().get(position) != Some(context.axis_sharding()))
                        .then(|| (position, sharding.clone()))
                });
                let batch = if let Some((position, sharding)) = normalization {
                    let axis_sharding = context.axis_sharding().clone();
                    let mut dimensions = sharding.dimensions().to_vec();
                    dimensions[position] = axis_sharding.clone();
                    let mut varying_manual_axes = sharding.varying_manual_axes().clone();

                    // A manual mesh axis remains semantically visible while it carries the mapped dimension,
                    // so record it in the rank-independent varying set as well as in the dimension placement.
                    if let ShardingDimension::Sharded(axis_names) = axis_sharding {
                        varying_manual_axes.extend(
                            axis_names
                                .into_iter()
                                .filter(|name| sharding.mesh().axis_type(name) == Some(MeshAxisType::Manual)),
                        );
                    }

                    let normalized_sharding = Sharding::new(sharding.mesh().clone(), dimensions)
                        .and_then(|normalized| normalized.with_unreduced_axes(sharding.unreduced_axes().clone()))
                        .and_then(|normalized| normalized.with_reduced_axes(sharding.reduced_axes().clone()))
                        .and_then(|normalized| normalized.with_varying_manual_axes(varying_manual_axes))
                        .map_err(|error| BatchingError::MisalignedBatchAxes { message: error.to_string() })?;

                    let normalized_type = batch
                        .r#type()
                        .into_owned()
                        .clone()
                        .with_sharding(normalized_sharding)
                        .map_err(|error| BatchingError::MisalignedBatchAxes { message: error.to_string() })?;

                    // Use an identity-shaped value broadcast instead of changing only `ArrayBatch`'s stored type. The
                    // value capability can therefore realize or stage the placement transition for the active backend.
                    let output_axes = (0..batch.r#type().rank()).collect::<Vec<_>>();
                    let value = batch.value.clone().broadcast(normalized_type.clone(), output_axes.as_slice())?;
                    ArrayBatch::new(normalized_type, value, batch.batch_axis)?
                } else {
                    batch
                };
                Ok(BatchingTracer::new(context.clone(), batch))
            })
            .collect::<Result<Vec<_>, BatchingError>>()?;
        let input = I::To::<BatchingTracer<Self>>::from_parameters(input_structure, inputs)?;
        let output = function(input)?;

        // Broadcast the caller's `output_batch_axes` into the output parameter structure, mirroring the
        // `input_batch_axes` handling above. A single `BatchAxis` leaf applies to every output, and a matching
        // structure gives one axis per leaf.
        let output_structure = output.parameter_structure();
        let output_batch_axis_values = output_batch_axes
            .broadcast_to_parameter_structure::<O::To<BatchAxis>>(output_structure.clone())?
            .into_parameters()
            .collect::<Vec<_>>();

        // Realign each output's packed batch axis to the caller's `output_batch_axes` and unwrap the parent tracer,
        // which already carries any enclosing level's metadata, so nested `batch` calls thread through with no side
        // table. `ArrayBatch::align_axis` owns the boundary contract: mapped positions are normalized and moved,
        // replicated outputs are broadcast for mapped declarations, and mapped outputs cannot be collapsed into a
        // replicated declaration without an explicit reduction.
        let parent_outputs = output
            .into_parameters()
            .zip(output_batch_axis_values)
            .map(|(output, output_batch_axis)| {
                Ok(output
                    .into_batch()
                    .align_axis(output_batch_axis, batch_size, context.axis_sharding().clone())?
                    .into_value())
            })
            .collect::<Result<Vec<_>, BatchingError>>()?;

        Ok(O::To::<Self::Value>::from_parameters(output_structure, parent_outputs)?)
    }
}

impl<C: Context<Type = ArrayType, Value: Broadcast + Transpose>> Batch for C {}

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
/// Concretely, staged [`Tracer`](crate::Tracer)s recover their trace, [`BatchingTracer`]s recover their batching level,
/// concrete values recover the eager backend domain they name, etc. Inputs with *no leaf values* are the one case this
/// function cannot serve. With nothing to recover a context from, it returns [`BatchingError::EmptyBatch`] even when
/// `batch_axis` supplies an explicit batch size. [`Batch::batch`] must be used in that case, with an explicit context.
///
/// `input_batch_axes` selects the mapped axis of each input leaf and `output_batch_axes` the position of the mapped
/// axis in each output leaf. Both are [`Parameterized`] values over [`BatchAxis`] leaves that are broadcast into the
/// corresponding parameter structure via [`Parameterized::broadcast_to_parameter_structure`]: a single [`BatchAxis`]
/// applies to every leaf, a value whose structure matches gives one axis per leaf, and a smaller compatible structure
/// broadcasts based on its prefixes. On the input side, [`BatchAxis::new(k)`](BatchAxis::new) maps the leaf on axis
/// `k` of its physical type, while [`BatchAxis::replicated`] shares the leaf unchanged across the batch. On the output
/// side, [`BatchAxis::new(k)`](BatchAxis::new) requests the mapped axis at position `k`: a naturally mapped output is
/// transposed when needed, while a naturally replicated output is broadcast across the batch, matching JAX's `vmap`.
/// Negative axes are normalized against the physical input or requested output rank, so `-1` denotes the final axis.
/// [`BatchAxis::replicated`] requires the output to remain replicated; collapsing a genuinely mapped output instead
/// requires an explicit reduction inside `function`.
///
/// When at least one input is mapped, the batch size is inferred from those inputs. The `batch_axis` argument accepts
/// anything convertible to a [`BatchAxisSpecification`] and can supply an explicit batch size (either to pin the
/// inferred size or to drive a fully-replicated `batch` transform whose batch size would otherwise be unobservable)
/// as well as an axis name that operations inside `function` like collectives can address.
///
/// # Parameters
///
///   - `function`: Function that represents the computation that needs to be batched/vectorized.
///   - `input`: Input (potentially structured) that the ought to be batched/vectorized.
///   - `input_batch_axes`: [`BatchAxis`] selection for the input leaves, broadcast into the input's structure.
///   - `output_batch_axes`: [`BatchAxis`] selection for the output leaves, broadcast into the output's structure.
///   - `batch_axis`: [`BatchAxisSpecification`] to use carrying an optional explicit batch size and an optional
///     batch axis name.
#[inline]
pub fn batch<
    V: Value<Type = ArrayType, ExecutionDomain: Context> + Broadcast + Transpose,
    F: FnOnce(I::To<BatchingTracer<V::ExecutionDomain>>) -> Result<O, ProgramError>,
    I: Parameterized<V, Family: ParameterizedFamily<BatchAxis> + ParameterizedFamily<BatchingTracer<V::ExecutionDomain>>>,
    O: Parameterized<BatchingTracer<V::ExecutionDomain>, Family: ParameterizedFamily<BatchAxis> + ParameterizedFamily<V>>,
    InputBatchAxes: Parameterized<BatchAxis>,
    OutputBatchAxes: Parameterized<BatchAxis>,
    Specification: Into<BatchAxisSpecification>,
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

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::backends::arrays::{Array, ArrayOperation};
    use crate::backends::scalars::Scalar;
    use crate::contexts::EagerContext;
    use crate::differentiation::forward::{ForwardModeDifferentiate, LinearizationTracer};
    use crate::differentiation::reverse::ReverseModeDifferentiate;
    use crate::operations::constants::OneLike;
    use crate::operations::math::{AddOperation, NegOperation, Reduce, ReductionKind};
    use crate::parameters::Placeholder;
    use crate::programs::types::Typed;
    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use crate::tracing::{DomainTracingContext, Trace};
    use crate::types::{ArrayType, DataType, Dimension, Shape};

    use super::*;

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
        assert_eq!(BatchAxisSpecification::from(None), BatchAxisSpecification::default());
        assert_eq!(BatchAxisSpecification::from(Some(4)), BatchAxisSpecification::sized(4));
        assert_eq!(BatchAxisSpecification::from(4), BatchAxisSpecification::sized(4));
        assert_eq!(BatchAxisSpecification::new(4, "i"), BatchAxisSpecification::new(4, "i").clone());
        assert_ne!(BatchAxisSpecification::sized(4), BatchAxisSpecification::sized(5));
        assert_ne!(BatchAxisSpecification::named("i"), BatchAxisSpecification::named("j"));
        assert_ne!(BatchAxisSpecification::named("i"), BatchAxisSpecification::default());
        assert_eq!(
            format!("{:?}", BatchAxisSpecification::named("i")),
            "BatchAxisSpecification { size: None, name: Some(\"i\") }",
        );
    }

    #[test]
    fn test_array_batch() {
        let matrix = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let matrix_type = matrix.r#type().into_owned();

        // `new` builds a batched value when the mapped axis is in bounds, and the accessors report the packed value,
        // its physical type, the batch size read off the mapped axis, and the per-item type with that axis removed.
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
        // `-1` denotes the final physical axis and the stored metadata is the canonical nonnegative position.
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
        // type is the whole physical type.
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
        // [3, 2] mapped at 1. The equivalent negative declaration is normalized against the physical output rank.
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
            let physical_type =
                ArrayType::new(DataType::F64, Shape::new(dimensions.into_iter().map(Dimension::Static).collect()))
                    .with_sharding(Sharding::new(mesh.clone(), sharding_dimensions).unwrap())
                    .unwrap();
            let batch = ArrayBatch::new(
                physical_type.clone(),
                Array::from_f64s(physical_type.clone(), (0..24).map(f64::from).collect()),
                BatchAxis::from_position(batch_axis),
            )
            .unwrap();
            let mut logical_dimensions = physical_type.shape().dimensions().to_vec();
            logical_dimensions.remove(batch_axis);
            assert_eq!(
                batch.unbatched_type(),
                ArrayType::new(DataType::F64, Shape::new(logical_dimensions))
                    .with_sharding(Sharding::replicated(mesh.clone(), 2))
                    .unwrap(),
            );
            let outputs = AddOperation.batch(&context, &EmptyRegionDriver, &[batch.clone(), batch]).unwrap();
            assert_eq!(outputs[0].r#type(), Cow::Borrowed(&physical_type));
            assert_eq!(outputs[0].batch_axis(), BatchAxis::from_position(batch_axis));
        }
    }

    #[test]
    fn test_batching_context() {
        let context = BatchingContext::new("parent", 4);
        assert_eq!(context.parent(), &"parent");
        assert_eq!(context.axis_size(), 4);
        assert_eq!(context.axis_name(), None);
        assert_eq!(context.axis_sharding(), &ShardingDimension::Replicated);

        let context = context.with_axis_name("items".to_string()).with_axis_sharding(ShardingDimension::sharded(["x"]));
        assert_eq!(context.axis_name(), Some("items"));
        assert_eq!(context.axis_sharding(), &ShardingDimension::sharded(["x"]));
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

        // Batching rules always receive the active `BatchingContext`, with the physical work running
        // through its parent context.
        let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 2);

        // Two operands mapped on the same axis add per item, and the output stays mapped on that axis.
        let left = make_batch(&matrix_type, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Some(0));
        let right = make_batch(&matrix_type, vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], Some(0));
        let outputs = AddOperation.batch(&context, &EmptyRegionDriver, &[left.clone(), right]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value(), &Array::matrix(2, 3, vec![11.0, 22.0, 33.0, 44.0, 55.0, 66.0]));

        // A replicated operand is broadcast across the mapped operand's batch before adding.
        let replicated = make_batch(&vector_type, vec![10.0, 20.0, 30.0], None);
        let outputs = AddOperation.batch(&context, &EmptyRegionDriver, &[left.clone(), replicated]).unwrap();
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value(), &Array::matrix(2, 3, vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]));

        // Operands mapped on different axes are realigned onto the first mapped operand's axis before adding.
        let transposed_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3), Dimension::Static(2)]));
        let right_axis_one = make_batch(&transposed_type, vec![10.0, 40.0, 20.0, 50.0, 30.0, 60.0], Some(1));
        let outputs = AddOperation.batch(&context, &EmptyRegionDriver, &[left, right_axis_one]).unwrap();
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value(), &Array::matrix(2, 3, vec![11.0, 22.0, 33.0, 44.0, 55.0, 66.0]));

        // Physical mapped-axis positions are canonicalized independently of operand rank. The rank-3 left operand
        // maps its trailing axis while the rank-1 right operand maps its only axis; their logical per-item shapes are
        // `[3, 4]` and scalar, respectively. The output is restored to the first mapped input's trailing axis.
        let left_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Static(3), Dimension::Static(4), Dimension::Static(2)]),
        );
        let right_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)]));
        let left = make_batch(&left_type, (1..=24).map(f64::from).collect(), Some(2));
        let right = make_batch(&right_type, vec![10.0, 20.0], Some(0));
        let outputs = AddOperation.batch(&context, &EmptyRegionDriver, &[left, right]).unwrap();
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
        let outputs = AddOperation.batch(&context, &EmptyRegionDriver, &[left_replicated, right_replicated]).unwrap();
        assert_eq!(outputs[0].batch_axis(), BatchAxis::replicated());
        assert_eq!(outputs[0].value(), &Array::vector(vec![11.0, 22.0, 33.0]));

        // Unary elementwise operations use the same blanket rule and preserve the mapped input axis.
        let input = make_batch(&vector_type, vec![1.0, 2.0, 3.0], Some(0));
        let outputs = NegOperation.batch(&context, &EmptyRegionDriver, &[input]).unwrap();
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
            let outputs = AddOperation.batch(&context, &EmptyRegionDriver, &[sharded, replicated]).unwrap();
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

        let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 3);
        let outputs = AddOperation.interpret_with_batch_axes(&context, &[left, right], &[BatchAxis::new(0)]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(*outputs[0].r#type(), vector_type);
        assert_eq!(outputs[0].value(), &Array::vector(vec![11.0, 22.0, 33.0]));

        // An `output_batch_axes` length that disagrees with the number of produced outputs is rejected.
        let left = ArrayBatch::new(vector_type.clone(), Array::vector(vec![1.0, 2.0, 3.0]), Some(0)).unwrap();
        let right = ArrayBatch::new(vector_type, Array::vector(vec![10.0, 20.0, 30.0]), Some(0)).unwrap();
        assert!(matches!(
            AddOperation.interpret_with_batch_axes(&context, &[left, right], &[]),
            Err(BatchingError::Program(_)),
        ));
    }

    #[test]
    fn test_region_batching_preserves_mapped_axis_sharding() {
        for axis_type in [MeshAxisType::Explicit, MeshAxisType::Manual] {
            let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, axis_type).unwrap()]).unwrap();
            let logical_sharding = if axis_type == MeshAxisType::Manual {
                Sharding::new(mesh.clone(), vec![ShardingDimension::replicated()])
                    .unwrap()
                    .with_varying_manual_axes(["x"])
                    .unwrap()
            } else {
                Sharding::replicated(mesh.clone(), 1)
            };
            let logical_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]))
                .with_sharding(logical_sharding)
                .unwrap();
            let (_, program) =
                EagerContext::<Array, ArrayOperation<Array>>::trace(|inputs: Vec<_>| Ok(inputs), vec![logical_type])
                    .unwrap();

            let (batched, output_axes) = program
                .entry_region_ref()
                .batched(
                    2,
                    ShardingDimension::sharded(["x"]),
                    &[BatchAxis::new(1)],
                    ProgramBatchingOutputAxesPolicy::Natural,
                )
                .unwrap();
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
        // Trace a per-item squaring function into a flat program over per-item (logical) vector types.
        let vector_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |inputs: Vec<_>| Ok(vec![inputs[0].clone() * inputs[0].clone()]),
            vec![vector_type.clone()],
        )
        .unwrap();

        // A mapped input at axis 0 turns the program into one over `[2, 3]`-shaped physical inputs that squares each
        // row, with the output naturally mapped on the same axis.
        let (batched, output_axes) = program
            .batched(2, ShardingDimension::Replicated, &[BatchAxis::new(0)], ProgramBatchingOutputAxesPolicy::Natural)
            .unwrap();
        assert_eq!(output_axes, vec![BatchAxis::new(0)]);
        let input = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let outputs = batched.interpret(vec![input]).unwrap();
        assert_eq!(outputs, vec![Array::matrix(2, 3, vec![1.0, 4.0, 9.0, 16.0, 25.0, 36.0])]);

        // A negative input axis is normalized against the physical input rank. Mapping the final axis consumes a
        // `[3, 2]` physical value and preserves that canonical axis through the elementwise body.
        let (batched, output_axes) = program
            .batched(2, ShardingDimension::Replicated, &[BatchAxis::new(-1)], ProgramBatchingOutputAxesPolicy::Natural)
            .unwrap();
        assert_eq!(output_axes, vec![BatchAxis::new(1)]);
        let input = Array::matrix(3, 2, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        let outputs = batched.interpret(vec![input]).unwrap();
        assert_eq!(outputs, vec![Array::matrix(3, 2, vec![1.0, 16.0, 4.0, 25.0, 9.0, 36.0])]);

        // A replicated input keeps its logical `[3]` type, and `AlignAllTo(0)` broadcasts the naturally replicated
        // output across the batch so the batched program still produces one `[2, 3]` output per item.
        let (batched, output_axes) = program
            .batched(
                2,
                ShardingDimension::Replicated,
                &[BatchAxis::replicated()],
                ProgramBatchingOutputAxesPolicy::AlignAllTo(Axis::from(0)),
            )
            .unwrap();
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
            .unwrap();
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
            |x: Vec<BatchingTracer<EagerContext<Array, ArrayOperation<Array>>>>| Ok(x),
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
        // The scalar input is replicated inside the batch, so the elementwise batching rule stages a `Broadcast` on
        // the differentiated value; the gradient must flow back through the broadcast's transpose rule (a sum-reduction
        // over the batch axis).
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
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(None)])),
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
        // A replicated scalar constant added to a mapped [3, 4] input: the elementwise rule materializes a
        // `BroadcastOperation` to the full common batched shape so the staged add receives shape-congruent operands.
        // This is required for backends such as XLA whose elementwise lowerings (e.g., `stablehlo.add`) have no
        // implicit broadcasting.
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
}
