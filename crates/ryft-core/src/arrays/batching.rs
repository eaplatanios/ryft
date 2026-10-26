//! Contains batching representations, policies, and other machinery to support the batching transform for the arrays
//! domain. The top-level [`batching`](crate::batching) module owns the universe-neutral transform protocol, contexts,
//! drivers, and entry points. This module supplies that protocol's concrete implementations for [`ArrayType`] and
//! [`ArrayIrType`]. Arrays use the ordinary [`ArrayBatch`] representation, while first-class dimensions are shared
//! shape values and therefore remain replicated across the batch, and mixed operations explicitly state how they cross
//! that boundary. A reference is batched by batching its referent (i.e., `ref<T>` batched at axis `k` is carried as
//! `ref<T with the batch extent inserted at k>`, and that axis is fixed for the reference's whole lifetime because
//! it is the axis of the shared storage rather than that of any one value). A reference therefore never moves or
//! broadcasts to a requested axis; a batched reference enters the transform as a batched input, while captured
//! references arrive replicated and stay unbatched.
//!
//! Every batch item must have the same rectangular shape. Mapped data cannot produce first-class dimensions
//! or per-item array extents. Variable-length workloads can use padded arrays and explicit sizes with
//! [`ragged_dot`](crate::RaggedDot) or [`ParallelRaggedAllToAllOperation`](crate::ParallelRaggedAllToAllOperation).

use std::borrow::Cow;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Arc;

use ryft_macros::Parameter;

use crate::arrays::broadcasting::Broadcastable;
use crate::arrays::dimensions::DimensionValue;
use crate::arrays::sharding::{Sharding, ShardingDimension, ShardingError};
use crate::arrays::types::{ArrayIrType, ArrayType, DataType, Dimension, DimensionType, Shape};
use crate::axes::Axis;
use crate::batching::{
    BatchAxis, BatchAxisSpecification, BatchableOperation, BatchableType, BatchedProgram, BatchingContext,
    BatchingDriver, BatchingEntrypointPolicy, BatchingError, BatchingLevel, BatchingLevelExtent, BatchingPolicy,
    BatchingPolicyProjection, BatchingTracer, BoundaryPreservingBatchedProgram, DiagonalBatchingPolicy,
    InterpretableBatchableOperation, ProgramBatchingOutputAxesPolicy, RecursiveBatchingDriver, RecursiveBatchingPolicy,
    ReplicatedBatchingPolicyProjection,
};
use crate::contexts::{Context, ContextProjection, EagerContext, ProjectedContext, StagingContext, ValueResolution};
use crate::interpretation::InterpretableOperation;
use crate::macros::{check_builders, check_count};
use crate::operations::{
    Assert, Broadcast, BroadcastOperation, Compare, ComparisonDirection, ConstantOperation, DimensionConstant,
    DimensionSize, DimensionSizeOperation, DynamicBroadcast, DynamicBroadcastOperation, DynamicIota,
    ElementwiseOperation, Gather, GatherDimensionNumbers, GatherOptions, GuardOperation, Iota, Transpose,
    TransposeOperation,
};
use crate::parameters::{Parameter, Placeholder};
use crate::programs::{
    AttachedRegions, Operation, OperationProjection, OperationProvider, Program, ProgramError, ProjectedValue,
    ReferenceType, Region, RegionRef, RegionReplayMappings, ReplayRegionDriver, Transform, TransformArtifact, Type,
    TypeError, Typed, Value, ValueDirectDispatch, ValueProjection,
};
use crate::tracing::{Tracer, TracingContext};

/// Source of an output dimension of an elementwise batching broadcast operation. The shared elementwise batching
/// algorithm computes every broadcast shape. For every batched output axis, it identifies the source of that dimension
/// and passes it to [`ArrayExtentBatchingPolicy::broadcast_input`]. [`StaticArrayExtentBatchingPolicy`] ignores these
/// sources because its output metadata already carries every extent, while a dimension-valued policy materializes each
/// source as a first-class value without re-deriving any extents.
#[derive(Clone, Debug)]
pub enum DimensionSource<V> {
    /// Dimension with the provided static extent.
    Static(usize),

    /// Dynamic per-item dimension readable from axis `axis` of the broadcast-compatible source value `source`, which is
    /// a clone of the corresponding input's parent-owned value.
    Value {
        /// Parent-owned value whose axis carries this dimension.
        source: V,

        /// Axis of `source`'s packed shape that carries this dimension.
        axis: usize,
    },

    /// The mapped batch axis itself, provided by the transform's extent.
    BatchExtent,
}

/// Value with [`ArrayType`] type that represents a _packed_ batch of arrays. [`ArrayBatch`] is the batching
/// representation for Ryft's batching/vectorization transform over arrays. It pairs a packed array value with a
/// [`BatchAxis`] that marks which of its dimensions indexes the batch items. A value is either _batched_ (i.e., its
/// packed type carries the batch dimension) or _replicated_, meaning that it is shared unchanged across every batch
/// item.
#[derive(Clone, Debug, PartialEq, Parameter)]
pub struct ArrayBatch<V> {
    /// Refer to the documentation of [`value`](Self::value) for more information.
    value: V,

    /// Refer to the documentation of [`batch_axis`](Self::batch_axis) for more information.
    batch_axis: BatchAxis,
}

impl<V: Value<Type = ArrayType>> ArrayBatch<V> {
    /// Creates a new dense [`ArrayBatch`] over the provided packed array value. This constructor interprets an
    /// existing dimension of an already-packed value as the batch dimension; it moves no data and adds no axis. Use
    /// [`Self::replicated`] for a per-item value shared unchanged across the batch, and [`Self::broadcast`] to
    /// materialize a packed batch dimension onto a replicated carrier.
    ///
    /// # Parameters
    ///
    ///   - `value`: Packed array value, whose own [`ArrayType`] is the batch's packed type.
    ///   - `batch_axis`: Possibly-negative [`BatchAxis`], normalized against the packed rank before it is stored.
    #[inline]
    pub fn new<A: Into<BatchAxis>>(value: V, batch_axis: A) -> Result<Self, BatchingError> {
        let batch_axis = {
            let r#type = value.r#type();
            let (batch_axis, _) = r#type.normalize_batch_axis(batch_axis.into())?;
            r#type.unbatched(batch_axis)?;
            batch_axis
        };
        Ok(Self { value, batch_axis })
    }

    /// Creates a new [`ArrayBatch`] that replicates the provided value across the batch.
    #[inline]
    pub fn replicated(value: V) -> Self {
        Self { value, batch_axis: BatchAxis::replicated() }
    }

    /// Returns the [`BatchAxis`] marking which dimension of [`value`](Self::value) indexes the batch items.
    #[inline]
    pub fn batch_axis(&self) -> BatchAxis {
        self.batch_axis
    }

    /// Returns the canonical non-negative position of this [`ArrayBatch`]'s mapped [`BatchAxis`]. [`ArrayBatch::new`]
    /// normalizes signed declarations before storing them, and so internal batching rules can use this index directly.
    #[inline]
    pub fn batch_axis_position(&self) -> Option<usize> {
        self.batch_axis.axis().map(|axis| axis.normalize(self.r#type().rank()).unwrap())
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
        let r#type = self.r#type();
        let size = r#type
            .dimension(axis)
            .value()
            .ok_or_else(|| BatchingError::DynamicBatchAxis { r#type: Box::new(r#type.clone().into_owned()), axis })?;
        Ok(Some(size))
    }

    /// Returns the per-item [`ArrayType`] of this batch, with the mapped batch dimension removed.
    #[inline]
    pub fn unbatched_type(&self) -> ArrayType {
        self.r#type().unbatched(self.batch_axis).unwrap()
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

    /// Returns an [`ArrayBatch`] by broadcasting this replicated value to gain a batch axis of size `axis_size` at
    /// `axis`. This is the analogue of JAX's `batching.broadcast` and is the canonical building block for mixed
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
        V: Broadcast,
    {
        if !self.batch_axis().is_replicated() {
            return Err(BatchingError::MisalignedBatchAxes {
                message: "`ArrayBatch::broadcast` expects a replicated input but received a batched value".to_string(),
            });
        }

        // The insertion position is normalized against the batched output rank (i.e., the per-item rank plus the
        // inserted batch dimension).
        let axis = axis.into();
        let packed_type = self.r#type().into_owned();
        let output_rank = packed_type.rank() + 1;
        let position = axis
            .normalize(output_rank)
            .map_err(|_| BatchingError::BatchAxisOutOfBounds { r#type: Box::new(self.r#type().into_owned()), axis })?;
        let batched_type = packed_type.batched(position, Dimension::Static(axis_size), axis_sharding)?;
        let output_axes = (0..packed_type.rank())
            .map(|dimension| if dimension < position { dimension } else { dimension + 1 })
            .collect::<Vec<_>>();
        let broadcasted = self.value().broadcast(batched_type, output_axes.as_slice())?;
        ArrayBatch::new(broadcasted, axis)
    }

    /// Returns an [`ArrayBatch`] with its mapped batch axis moved to `axis`, staging a transpose on the packed value
    /// via [`Transpose::move_axis`] to realign it. This is the *move* half of JAX's `matchaxis` (i.e., a `moveaxis` on
    /// the batch dimension). The full move-or-broadcast behavior lives on [`match_axis`](Self::match_axis). It brings
    /// inputs that map their batch axis at different positions onto a common axis before an elementwise operation is
    /// applied. A replicated value (i.e., one with no mapped axis), or one already mapped at `axis`, is returned
    /// unchanged.
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
        let rank = self.r#type().rank();
        let position = axis
            .normalize(rank)
            .map_err(|_| BatchingError::BatchAxisOutOfBounds { r#type: Box::new(self.r#type().into_owned()), axis })?;
        if current_axis == position {
            return Ok(self.clone());
        }

        // Moving one axis keeps every other axis in its relative order, exactly as `Transpose::move_axis` does.
        let mut permutation = (0..rank).filter(|axis| *axis != current_axis).collect::<Vec<_>>();
        permutation.insert(position, current_axis);
        let permuted_value = self.value().transpose(permutation)?;
        Self::new(permuted_value, axis)
    }

    /// Returns an [`ArrayBatch`] with a batch axis of size `axis_size` materialized at `axis`. An already-batched value
    /// has its mapped axis realigned to `axis` via [`Self::move_axis`], while a replicated value is broadcast to gain a
    /// batch axis there via [`Self::broadcast`]. This is the analogue of JAX's `batching.matchaxis`, used by rules
    /// whose inputs must agree on one packed batch axis (e.g., `pad`, `concatenate`, etc.).
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
        V: Broadcast + Transpose,
    {
        if self.batch_axis().is_replicated() {
            self.broadcast(axis, axis_size, axis_sharding)
        } else {
            self.move_axis(axis)
        }
    }

    /// Returns an [`ArrayBatch`] aligned to the provided output `batch_axis`. A mapped value is moved to a
    /// different requested position, while a replicated value is broadcast across `axis_size` when the declaration
    /// requests a mapped result. This matches JAX's mapped-output instantiation. A mapped value cannot be collapsed
    /// into a replicated declaration without an explicit reduction and that direction returns
    /// [`BatchingError::MismatchedOutputAxes`]. Signed declarations are normalized against the resulting packed
    /// rank. [`Batch::batch`](crate::Batch::batch) uses this function to realize the caller's declared
    /// `output_batch_axes`.
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
        batch_axis_sharding(inputs.iter().map(|input| (input.r#type(), input.batch_axis_position())))
    }

    /// Returns the packed form of `declared` using this [`ArrayBatch`]'s mapped axis and the provided batch extent and
    /// [`ShardingDimension`]. A replicated array batch keeps `declared` unchanged. Every homogeneous array extent
    /// policy uses this derivation; the policy supplies its extent separately.
    ///
    /// Callers validate the resulting packed declaration against the supplied values to enforce its type relationships.
    ///
    /// # Parameters
    ///
    ///   - `declared`: Per-item declaration to lift using this [`ArrayBatch`]'s representation.
    ///   - `extent`: Extent of the mapped batch dimension.
    ///   - `axis_sharding`: Sharding of the mapped batch dimension.
    ///
    /// # Errors
    ///
    /// Returns a [`BatchingError`] for an invalid mapped-axis position or invalid resulting sharding.
    fn batched_declaration(
        &self,
        declared: &ArrayType,
        extent: &Dimension,
        axis_sharding: &ShardingDimension,
    ) -> Result<ArrayType, BatchingError> {
        match self.batch_axis().axis() {
            None => Ok(declared.clone()),
            Some(axis) => {
                let axis = axis.normalize(declared.rank() + 1).map_err(ProgramError::from)?;
                declared.batched_declaration(self.value().r#type().as_ref(), axis, extent, axis_sharding)
            }
        }
    }
}

impl<V: Display + Typed<Type = ArrayType>> Display for ArrayBatch<V> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.batch_axis.axis() {
            Some(axis) => write!(formatter, "batch[{}, axis={axis}]({})", self.r#type(), self.value),
            None => write!(formatter, "batch[{}, replicated]({})", self.r#type(), self.value),
        }
    }
}

impl<V: Typed<Type = ArrayType>> Typed for ArrayBatch<V> {
    type Type = ArrayType;

    #[inline]
    fn r#type(&self) -> Cow<'_, ArrayType> {
        // The packed type of a batch is exactly the type of the packed value that it carries.
        self.value.r#type()
    }
}

impl<V: Value<Type = ArrayType>> Value for ArrayBatch<V> {
    type Dispatch = ValueDirectDispatch;
    type Domain = EagerContext<Self>;

    #[inline]
    fn domain(&self) -> EagerContext<Self> {
        EagerContext::new()
    }
}

/// Packed array, reference, or replicated first-class dimension used by array IR batching. Arrays and references
/// carry their mapped axis in the array or referent shape. A dimension is shared unchanged across the batch, so
/// assigning it a mapped axis is rejected by [`Self::new`].
#[derive(Clone, Debug, PartialEq, Parameter)]
pub struct ArrayIrBatch<V: Value<Type = ArrayIrType>> {
    /// Packed value whose type determines the carrier member.
    value: V,

    /// Normalized mapped axis, or a replicated declaration.
    batch_axis: BatchAxis,
}

impl<V: Value<Type = ArrayIrType>> ArrayIrBatch<V> {
    /// Creates a new [`ArrayIrBatch`] carrier for an already-packed value, normalizing its mapped axis against
    /// the array or referent rank.
    ///
    /// # Parameters
    ///
    ///   - `value`: Packed value, whose own [`ArrayIrType`] determines the member kind.
    ///   - `batch_axis`: Possibly-negative [`BatchAxis`], normalized against the rank the member kind batches over
    ///     before it is stored.
    ///
    /// # Errors
    ///
    /// Returns a [`BatchingError`] for an invalid array axis, projected sharding, or a mapped first-class dimension.
    pub fn new<A: Into<BatchAxis>>(value: V, batch_axis: A) -> Result<Self, BatchingError> {
        let batch_axis = batch_axis.into();
        let batch_axis = match value.r#type().as_ref() {
            ArrayIrType::Array(r#type) => {
                let (batch_axis, _) = r#type.normalize_batch_axis(batch_axis)?;
                r#type.unbatched(batch_axis)?;
                batch_axis
            }
            ArrayIrType::Reference(r#type) => {
                let (batch_axis, _) = r#type.referent().normalize_batch_axis(batch_axis)?;
                r#type.referent().unbatched(batch_axis)?;
                batch_axis
            }
            ArrayIrType::Dimension(r#type) if !batch_axis.is_replicated() => {
                return Err(BatchingError::MappedDimension { r#type: Box::new(r#type.clone()), axis: batch_axis });
            }
            ArrayIrType::Dimension(_) => batch_axis,
        };
        Ok(Self { value, batch_axis })
    }

    /// Creates a carrier that shares `value` unchanged across every batch item.
    #[inline]
    pub fn replicated(value: V) -> Self {
        Self { value, batch_axis: BatchAxis::replicated() }
    }

    /// Returns the [`BatchAxis`] marking which dimension of [`value`](Self::value) indexes the batch items, or a
    /// replicated axis for a value shared unchanged across the batch. The rank the axis is normalized against depends
    /// on the member kind (for an array member it is an axis of the packed array, and for a reference member an axis
    /// of the packed referent, fixed by the input or allocation that produced the reference for the reference's whole
    /// lifetime). A first-class dimension always carries a replicated axis.
    #[inline]
    pub fn batch_axis(&self) -> BatchAxis {
        self.batch_axis
    }

    /// Returns the non-negative mapped-axis position in the array or referent, or `None` for a replicated value.
    pub fn batch_axis_position(&self) -> Option<usize> {
        let value_type = self.value.r#type();
        let rank = match value_type.as_ref() {
            ArrayIrType::Array(r#type) => r#type.rank(),
            ArrayIrType::Reference(r#type) => r#type.referent().rank(),
            ArrayIrType::Dimension(_) => return None,
        };
        self.batch_axis.axis().map(|axis| axis.normalize(rank).unwrap())
    }

    /// Returns the packed value.
    #[inline]
    pub fn value(&self) -> &V {
        &self.value
    }

    /// Consumes this carrier and returns the packed value.
    #[inline]
    pub fn into_value(self) -> V {
        self.value
    }

    /// Returns the per-item type by removing the mapped axis from an array or reference referent.
    pub fn unbatched_type(&self) -> ArrayIrType {
        match self.value.r#type().as_ref() {
            ArrayIrType::Array(r#type) => r#type.unbatched(self.batch_axis).unwrap().into(),
            ArrayIrType::Reference(r#type) => {
                ReferenceType::new(r#type.referent().unbatched(self.batch_axis).unwrap()).into()
            }
            ArrayIrType::Dimension(r#type) => r#type.clone().into(),
        }
    }

    /// Validates that this carrier holds a replicated dimension, as required for an ordinary shape operand.
    ///
    /// # Errors
    ///
    /// Returns a type error for an array or reference, and a [`BatchingError::MappedDimension`] for a dimension that
    /// carries a mapped batch axis.
    pub fn validate_replicated_dimension(&self) -> Result<(), BatchingError> {
        let value_type = self.value.r#type();
        let r#type = <&DimensionType>::try_from(value_type.as_ref())?;
        if self.batch_axis.is_replicated() {
            Ok(())
        } else {
            Err(BatchingError::MappedDimension { r#type: Box::new(r#type.clone()), axis: self.batch_axis })
        }
    }
}

impl<V: Value<Type = ArrayIrType>> Typed for ArrayIrBatch<V> {
    type Type = ArrayIrType;

    #[inline]
    fn r#type(&self) -> Cow<'_, ArrayIrType> {
        // The type of a batch carrier is its logical per-item type.
        Cow::Owned(self.unbatched_type())
    }
}

impl<V: Value<Type = ArrayIrType>> Display for ArrayIrBatch<V> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.batch_axis.axis() {
            Some(axis) => write!(formatter, "batch[{}, axis={axis}]({})", self.r#type(), self.value),
            None => write!(formatter, "batch[{}, replicated]({})", self.r#type(), self.value),
        }
    }
}

/// Result of batching an array IR [`Region`], whose boundary explicitly threads its mapped extent.
///
/// Array IR regions are retraced as standalone programs and therefore cannot capture the parent program's first-class
/// mapped extent Single Static Assignment (SSA) value. Their boundary instead gains one additional leading dimension
/// input and output where the input defines the identity referenced by every inserted dynamic batch dimension, and the
/// output forwards that same atom so that enclosing higher-order operations can carry it through the sealed region. For
/// example, batching the following region:
///
/// ```text
/// lambda %0:f32[3] .
/// let %1:f32[3] = neg %0
/// in (%1)
/// ```
///
/// with its input mapped at axis `0` over a dynamic extent `batch ∈ [1, 9)` produces the following program:
///
/// ```text
/// lambda %0:dimension<batch ∈ [1, 9)>, %1:f32[batch, 3] .
/// let %2:f32[batch, 3] = neg %1
/// in (%0, %2)
/// ```
///
/// where `%0` is the threaded extent. The batched input type `f32[batch, 3]` names it, and the leading output forwards
/// it unchanged. A static extent such as `dimension<2>` uses the identical boundary and merely renders the batched
/// input as `f32[2, 3]`. [`Self::output_axes`] excludes the bookkeeping output and reports one mapped axis `0` for the
/// single semantic output `%2`, so the program returned by [`BatchedProgram::into_parts`] has one more input and output
/// than that metadata describes. Consumers that instead need an ordinary [`Region`] boundary shed the widening through
/// [`BatchingPolicy::adapt_batched_program`] and complete the adapted program's inputs with
/// [`BatchingPolicy::boundary_inputs`].
pub struct ThreadedExtentBatchedProgram<V: Typed<Type = ArrayIrType> + Parameter, O> {
    /// Structurally transformed program, including its leading bookkeeping input and output.
    program: Program<V, O, Vec<V>, Vec<V>>,

    /// Mapped axes of the source region's outputs. The bookkeeping-only threaded extent is excluded.
    output_axes: Vec<BatchAxis>,
}

impl<V: Value<Type = ArrayIrType>, O: Operation<Type = ArrayIrType>> ThreadedExtentBatchedProgram<V, O> {
    /// Creates a new [`ThreadedExtentBatchedProgram`] with one leading mapped extent input and a forwarded output.
    pub(crate) fn new(
        program: Program<V, O, Vec<V>, Vec<V>>,
        output_axes: Vec<BatchAxis>,
    ) -> Result<Self, ProgramError> {
        if program.input_count() == 0 || program.output_count() == 0 {
            return Err(ProgramError::MalformedProgram(
                "a structurally batched program with a threaded extent must have a leading input and output"
                    .to_string(),
            ));
        }
        check_count!("output", output_axes, program.output_count() - 1, ProgramError);
        if !matches!(program.inputs().next().unwrap().r#type().as_ref(), ArrayIrType::Dimension(_)) {
            return Err(ProgramError::MalformedProgram(
                "a structurally batched program's leading threaded extent input must be a dimension".to_string(),
            ));
        }
        if !matches!(program.outputs().next().unwrap().r#type().as_ref(), ArrayIrType::Dimension(_)) {
            return Err(ProgramError::MalformedProgram(
                "a structurally batched program's leading threaded extent output must be a dimension".to_string(),
            ));
        }
        if program.output_ids()[0] != program.input_ids()[0] {
            return Err(ProgramError::MalformedProgram(
                "a structurally batched program's leading threaded extent output must forward its leading input"
                    .to_string(),
            ));
        }
        Ok(Self { program, output_axes })
    }
}

impl<V: Value<Type = ArrayIrType>, O: Operation<Type = ArrayIrType>> BatchedProgram<V, O>
    for ThreadedExtentBatchedProgram<V, O>
{
    #[inline]
    fn output_axes(&self) -> &[BatchAxis] {
        self.output_axes.as_slice()
    }

    #[inline]
    fn into_parts(self) -> (Program<V, O, Vec<V>, Vec<V>>, Vec<BatchAxis>) {
        (self.program, self.output_axes)
    }
}

/// Array-specific capability that lets one homogeneous-array batching rule set serve multiple extent representations.
/// Ordinary array batching and projected array batching inside an array IR program share everything that makes an
/// array rule what it is (e.g., the [`ArrayBatch`] carrier, the shared elementwise alignment algorithm, and every
/// per-operation [`BatchableOperation`] implementation). They differ in exactly two ways:
///
///   - **Extent Representation:** Ordinary batching knows its mapped-axis extent as one static host `usize`. Projected
///     batching's extent is an ordinary parent-owned first-class dimension value, so a dynamic batch extent remains a
///     Single Static Assignment (SSA) value flowing through instruction input edges rather than static transform
///     metadata.
///   - **Replicated-Array Materialization:** Aligning a replicated array with mapped inputs requires broadcasting it
///     across the mapped axis. Ordinary batching can use broadcasting operation with static output metadata, while
///     projected batching must stage the mixed broadcast that consumes explicit dimension inputs.
///
/// An [`ArrayExtentBatchingPolicy`] is a type-level selector packaging those two differences, so each policy owns the
/// complete translation from a homogeneous rule's extent and broadcast requests to its universe's operations. Every
/// policy also implements [`BatchingPolicy`] with `Batch = ArrayBatch<C::Value>` and `BatchedProgram =
/// BoundaryPreservingBatchedProgram<C::Constant, C::Operation>`: homogeneous rules bind structurally batched branch
/// programs directly, so an array batching policy preserves the source boundary by definition and the supertrait
/// binding lets every rule rely on that without restating it. The shared rules are written once against the nominal
/// [`ArrayBatchingPolicy<P>`] family rather than as a `P: ArrayExtentBatchingPolicy` blanket, because Rust coherence
/// cannot use the _absence_ of a trait implementation to prove such a blanket disjoint from the genuinely mixed
/// array IR operation rules registered for other policies.
///
/// Keeping this capability on the batching transform, rather than on [`ProjectedContext`], [`ArrayBatch`], or [`Type`],
/// means that neither the carrier nor the type contract needs to know anything about dynamic-shape state that only
/// batching needs.
pub trait ArrayExtentBatchingPolicy<C: Context<Type = ArrayType>>:
    BatchingPolicy<
        C,
        Batch = ArrayBatch<C::Value>,
        BatchedProgram = BoundaryPreservingBatchedProgram<C::Constant, C::Operation>,
    >
{
    /// Returns the mapped-axis [`Dimension`] to insert when building a batched [`ArrayType`].
    /// [`StaticArrayExtentBatchingPolicy`] derives an exact [`Dimension::Static`] from the context's mapped-axis
    /// extent, while a dimension-valued policy returns the possibly dynamic dimension described by its first-class
    /// extent value's type. Shape computations can therefore construct batched types without forcing the extent to be
    /// statically known.
    fn axis_dimension(context: &BatchingContext<C, ArrayBatchingPolicy<Self>>) -> Result<Dimension, BatchingError>;

    /// Returns the statically known mapped-axis size. This is the exact-size projection of [`Self::axis_dimension`].
    /// It succeeds when that dimension is static and returns a [`BatchingError::UnsupportedOperation`] when the mapped
    /// extent is genuinely dynamic. Rules that only move or broadcast arrays must use [`Self::match_axis`] and
    /// [`Self::broadcast_input`] instead, so that they keep working with dynamic mapped extents.
    #[inline]
    fn axis_size(context: &BatchingContext<C, ArrayBatchingPolicy<Self>>) -> Result<usize, BatchingError> {
        Self::axis_dimension(context)?.value().ok_or_else(|| BatchingError::UnsupportedOperation {
            message: "this batching rule requires a statically known mapped-axis extent".to_string(),
        })
    }

    /// Aligns `batch` so that its mapped axis sits at position `axis`. A mapped batch moves its existing axis.
    /// A replicated batch is materialized across the mapped extent by inserting the axis at `axis`.
    /// [`StaticArrayExtentBatchingPolicy`] broadcasts using the context's exact extent, while a dimension-valued policy
    /// stages the mixed broadcast whose inserted axis is grounded by the transform's first-class extent value.
    fn match_axis(
        context: &BatchingContext<C, ArrayBatchingPolicy<Self>>,
        batch: &ArrayBatch<C::Value>,
        axis: Axis,
    ) -> Result<ArrayBatch<C::Value>, BatchingError>;

    /// Materializes one instruction input at `r#type`, broadcasting its per-item dimensions to the common target shape
    /// and inserting the mapped batch axis. The shared elementwise algorithm computes the complete broadcast shape,
    /// including one [`DimensionSource`] per batched output axis, and delegates only the materialization itself, which
    /// is the policy-specific step. [`StaticArrayExtentBatchingPolicy`] broadcasts with static output metadata and
    /// ignores the sources, while a dimension-valued policy spends each source mechanically (i.e., exact constants for
    /// static dimensions, `dimension_size` reads of the provided source values for dynamic per-item dimensions, and the
    /// transform's extent value for the mapped axis itself).
    ///
    /// # Parameters
    ///
    ///   - `context`: Active [`BatchingContext`] for the transform level being applied.
    ///   - `input`: Instruction input batch to materialize.
    ///   - `r#type`: Complete batched type of the materialized result (i.e., the common per-item target type
    ///     with the mapped axis inserted at `batch_axis`).
    ///   - `output_axes`: Mapping from each of `input`'s axes to its output axis.
    ///   - `batch_axis`: Position of the mapped batch axis in `r#type`.
    ///   - `dimension_sources`: Source of each batched output dimension, in axis order.
    fn broadcast_input(
        context: &BatchingContext<C, ArrayBatchingPolicy<Self>>,
        input: &ArrayBatch<C::Value>,
        r#type: ArrayType,
        output_axes: Vec<usize>,
        batch_axis: Axis,
        dimension_sources: Vec<DimensionSource<C::Value>>,
    ) -> Result<ArrayBatch<C::Value>, BatchingError>;
}

/// [`ArrayExtentBatchingPolicy`] for ordinary homogeneous array batching with a static host extent. This is the default
/// policy of [`ArrayBatchingPolicy`] and preserves the established public array batching behavior. The mapped-axis
/// extent is one `usize` fixed when the transform is constructed, [`Self::axis_dimension`] is always an exact static
/// dimension, and replicated arrays are materialized with a broadcasting operation using static output metadata.
/// [`Self::axis_size`] always succeeds, so every batching rule (including host-side item enumeration) is
/// available under this policy.
#[derive(Copy, Clone, Debug, Default)]
pub struct StaticArrayExtentBatchingPolicy;

impl<C: Context<Type = ArrayType>> BatchingPolicy<C> for StaticArrayExtentBatchingPolicy {
    type Batch = ArrayBatch<C::Value>;
    type Extent = usize;
    type BatchedProgram = BoundaryPreservingBatchedProgram<C::Constant, C::Operation>;

    #[inline]
    fn batch(value: C::Value, batch_axis: BatchAxis) -> Result<Self::Batch, BatchingError> {
        ArrayBatch::new(value, batch_axis)
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
    fn batched_type(
        context: &BatchingContext<C, Self>,
        input: &Self::Batch,
        declared: &ArrayType,
    ) -> Result<ArrayType, BatchingError> {
        let extent = Dimension::Static(*context.axis_extent());
        input.batched_declaration(declared, &extent, context.axis_sharding())
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

impl<C: Context<Type = ArrayType>> ArrayExtentBatchingPolicy<C> for StaticArrayExtentBatchingPolicy
where
    C::Value: Broadcast + Transpose,
{
    #[inline]
    fn axis_dimension(context: &BatchingContext<C, ArrayBatchingPolicy<Self>>) -> Result<Dimension, BatchingError> {
        Ok(Dimension::Static(*context.axis_extent()))
    }

    #[inline]
    fn match_axis(
        context: &BatchingContext<C, ArrayBatchingPolicy<Self>>,
        batch: &ArrayBatch<C::Value>,
        axis: Axis,
    ) -> Result<ArrayBatch<C::Value>, BatchingError> {
        batch.match_axis(axis, *context.axis_extent(), context.axis_sharding().clone())
    }

    #[inline]
    fn broadcast_input(
        _context: &BatchingContext<C, ArrayBatchingPolicy<Self>>,
        input: &ArrayBatch<C::Value>,
        r#type: ArrayType,
        output_axes: Vec<usize>,
        batch_axis: Axis,
        _dimension_sources: Vec<DimensionSource<C::Value>>,
    ) -> Result<ArrayBatch<C::Value>, BatchingError> {
        let broadcasted = input.value().clone().broadcast(r#type, output_axes.as_slice())?;
        ArrayBatch::new(broadcasted, batch_axis)
    }
}

/// [`ArrayExtentBatchingPolicy`] used while a homogeneous array rule runs inside an array IR batching transform.
///
/// When array IR batching reaches an array member operation, it projects the operation and its batches into the
/// zero-state [`ProjectedContext`] over [`ArrayType`] and reuses the homogeneous rule unchanged: batches remain
/// ordinary [`ArrayBatch`]es, and so the rule cannot tell it is running inside an array IR program. What does change is
/// extent representation. The mapped-axis extent is the outer array IR context's first-class dimension value rather
/// than a static host `usize`, and so a dynamic batch extent remains an ordinary Single Static Assignment (SSA) value.
///
/// This [`ArrayExtentBatchingPolicy`] implementation is correspondingly the only place that translates a homogeneous
/// rule's extent and move-or-broadcast requests into mixed array IR operations: static per-item dimensions become exact
/// dimension constants, dynamic per-item dimensions become `dimension_size` reads of their broadcast-compatible source
/// axes, and the mapped axis itself is grounded by the extent value. [`ArrayExtentBatchingPolicy::axis_size`] succeeds
/// only when the extent value's type proves one exact extent, so rules that genuinely enumerate batch items fail with a
/// precise error at dynamic extents instead of silently specializing them.
#[derive(Copy, Clone, Debug, Default)]
pub struct DynamicArrayExtentBatchingPolicy;

impl<C: ContextProjection<ArrayType, Type = ArrayIrType>> BatchingPolicy<ProjectedContext<C, ArrayType>>
    for DynamicArrayExtentBatchingPolicy
{
    type Batch = ArrayBatch<<C::Value as ValueProjection<ArrayType>>::Projected>;
    type Extent = C::Value;
    type BatchedProgram = BoundaryPreservingBatchedProgram<
        <C::Constant as ValueProjection<ArrayType>>::Projected,
        <C::Operation as OperationProjection<ArrayType>>::Projected,
    >;

    #[inline]
    fn batch(
        value: <C::Value as ValueProjection<ArrayType>>::Projected,
        batch_axis: BatchAxis,
    ) -> Result<Self::Batch, BatchingError> {
        ArrayBatch::new(value, batch_axis)
    }

    #[inline]
    fn replicated(value: <C::Value as ValueProjection<ArrayType>>::Projected) -> Self::Batch {
        ArrayBatch::replicated(value)
    }

    #[inline]
    fn value(batch: &Self::Batch) -> &<C::Value as ValueProjection<ArrayType>>::Projected {
        batch.value()
    }

    #[inline]
    fn batch_axis(batch: &Self::Batch) -> BatchAxis {
        batch.batch_axis()
    }

    #[inline]
    fn batched_type(
        context: &BatchingContext<ProjectedContext<C, ArrayType>, Self>,
        input: &Self::Batch,
        declared: &ArrayType,
    ) -> Result<ArrayType, BatchingError> {
        // The mapped axis is described by the _type_ of the first-class extent value, as in `axis_dimension`.
        let extent = <&DimensionType>::try_from(context.axis_extent().r#type().as_ref())?.simplified_dimension();
        input.batched_declaration(declared, &extent, context.axis_sharding())
    }

    #[inline]
    fn unbatched_type(batch: &Self::Batch) -> Cow<'_, ArrayType> {
        Cow::Owned(batch.unbatched_type())
    }

    #[inline]
    fn adapt_batched_program<CollapseFn>(
        program: Self::BatchedProgram,
        required_output_axes: Option<&[BatchAxis]>,
        collapse_fn: CollapseFn,
    ) -> Result<
        BoundaryPreservingBatchedProgram<
            <C::Constant as ValueProjection<ArrayType>>::Projected,
            <C::Operation as OperationProjection<ArrayType>>::Projected,
        >,
        BatchingError,
    >
    where
        CollapseFn: Fn(
            &TracingContext<
                <C::Constant as ValueProjection<ArrayType>>::Projected,
                <C::Operation as OperationProjection<ArrayType>>::Projected,
            >,
            Tracer<
                TracingContext<
                    <C::Constant as ValueProjection<ArrayType>>::Projected,
                    <C::Operation as OperationProjection<ArrayType>>::Projected,
                >,
            >,
            Axis,
        ) -> Result<
            Tracer<
                TracingContext<
                    <C::Constant as ValueProjection<ArrayType>>::Projected,
                    <C::Operation as OperationProjection<ArrayType>>::Projected,
                >,
            >,
            BatchingError,
        >,
    {
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

impl<C: ContextProjection<ArrayType, Type = ArrayIrType>> ArrayExtentBatchingPolicy<ProjectedContext<C, ArrayType>>
    for DynamicArrayExtentBatchingPolicy
where
    C::Value: DimensionSize + DynamicBroadcast + ValueProjection<ArrayType>,
    <C::Value as ValueProjection<ArrayType>>::Projected: Transpose + Value<Type = ArrayType>,
    C::Operation:
        From<ConstantOperation<DimensionValue>> + From<DimensionSizeOperation> + OperationProjection<ArrayType>,
{
    #[inline]
    fn axis_dimension(
        context: &BatchingContext<
            ProjectedContext<C, ArrayType>,
            ArrayBatchingPolicy<DynamicArrayExtentBatchingPolicy>,
        >,
    ) -> Result<Dimension, BatchingError> {
        // The mapped axis is described by the _type_ of the first-class extent value (i.e., an exact dimension
        // when the extent is a constant, and the symbolic dimension it names otherwise).
        Ok(<&DimensionType>::try_from(context.axis_extent().r#type().as_ref())?.simplified_dimension())
    }

    fn match_axis(
        context: &BatchingContext<
            ProjectedContext<C, ArrayType>,
            ArrayBatchingPolicy<DynamicArrayExtentBatchingPolicy>,
        >,
        batch: &ArrayBatch<<C::Value as ValueProjection<ArrayType>>::Projected>,
        axis: Axis,
    ) -> Result<ArrayBatch<<C::Value as ValueProjection<ArrayType>>::Projected>, BatchingError> {
        // A mapped batch already carries the axis, so aligning it is a plain transpose that needs no extent at all.
        if !batch.batch_axis().is_replicated() {
            return batch.move_axis(axis);
        }

        // A replicated batch gains the mapped axis, so the request is normalized against the rank _after_ insertion.
        let array_type = batch.unbatched_type();
        let output_rank = array_type.rank() + 1;
        let position = axis
            .normalize(output_rank)
            .map_err(|_| BatchingError::BatchAxisOutOfBounds { r#type: Box::new(array_type.clone()), axis })?;

        // The broadcast is a mixed operation, so it is staged in the array IR parent (i.e., two levels up, past this
        // batching level and past the projected array view) on the lifted array IR value, and the result is projected
        // back into the array member afterward.
        let value = <C::Value as ValueProjection<ArrayType>>::from_projected(batch.value().clone());
        let value =
            broadcast_replicated_array(value, &array_type, position, context.axis_extent(), context.axis_sharding())?;
        ArrayBatch::new(
            <C::Value as ValueProjection<ArrayType>>::into_projected(value)?,
            BatchAxis::from_position(position),
        )
    }

    fn broadcast_input(
        context: &BatchingContext<
            ProjectedContext<C, ArrayType>,
            ArrayBatchingPolicy<DynamicArrayExtentBatchingPolicy>,
        >,
        input: &ArrayBatch<<C::Value as ValueProjection<ArrayType>>::Projected>,
        r#type: ArrayType,
        output_axes: Vec<usize>,
        batch_axis: Axis,
        dimension_sources: Vec<DimensionSource<<C::Value as ValueProjection<ArrayType>>::Projected>>,
    ) -> Result<ArrayBatch<<C::Value as ValueProjection<ArrayType>>::Projected>, BatchingError> {
        // Materialize every algorithm-provided output-dimension source in first-class form: exact constants for static
        // dimensions, `dimension_size` reads of the provided source values for dynamic per-item dimensions, and the
        // transform's extent value for the mapped axis.
        let outer_context = context.parent().parent();
        let output_dimensions = dimension_sources
            .into_iter()
            .map(|dimension_source| -> Result<C::Value, BatchingError> {
                match dimension_source {
                    DimensionSource::Static(extent) => Ok(outer_context.dimension_constant(extent)?),
                    DimensionSource::Value { source, axis } => {
                        let source = <C::Value as ValueProjection<ArrayType>>::from_projected(source);
                        Ok(source.dimension_size(axis)?)
                    }
                    DimensionSource::BatchExtent => Ok(context.axis_extent().clone()),
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        // The algorithm already decided where every input axis lands (i.e., `output_axes`) and which batch axis the
        // result carries, so this step only stages the mixed broadcast in the array IR parent.
        let value = <C::Value as ValueProjection<ArrayType>>::from_projected(input.value().clone());
        let value = value.dynamic_broadcast_with_output_sharding(
            &output_dimensions,
            &output_axes,
            r#type.sharding().cloned(),
        )?;
        ArrayBatch::new(<C::Value as ValueProjection<ArrayType>>::into_projected(value)?, batch_axis)
    }
}

/// Homogeneous-array [`BatchingPolicy`] parameterized by its [`ArrayExtentBatchingPolicy`]. The default
/// [`StaticArrayExtentBatchingPolicy`] preserves the ordinary public array batching API. Array IR programs use the
/// dynamic policy whose extent is a parent-owned first-class dimension value. Keeping both policies under this nominal
/// policy family lets every homogeneous array operation share one generated dispatcher without making those
/// implementations overlap genuinely mixed array IR operation rules. The wrapper delegates its batch carrier and
/// extent representation to `P`. Array-specific alignment is supplied by `P`'s [`ArrayExtentBatchingPolicy`]
/// implementation.
pub struct ArrayBatchingPolicy<P = StaticArrayExtentBatchingPolicy>(PhantomData<fn() -> P>);

impl<P> Copy for ArrayBatchingPolicy<P> {}

impl<P> Clone for ArrayBatchingPolicy<P> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<P> Debug for ArrayBatchingPolicy<P> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("ArrayBatchingPolicy")
    }
}

impl<P> Default for ArrayBatchingPolicy<P> {
    #[inline]
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<C: Context<Type = ArrayType>, P: BatchingPolicy<C, Batch = ArrayBatch<C::Value>>> BatchingPolicy<C>
    for ArrayBatchingPolicy<P>
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

    fn batched_type(
        context: &BatchingContext<C, Self>,
        input: &Self::Batch,
        declared: &ArrayType,
    ) -> Result<ArrayType, BatchingError> {
        // `P` owns the extent representation, so the declared type is batched under the same batching level viewed
        // through `P`. Deriving a type stages nothing, so the view needs no reference boundary of its own.
        let context = BatchingContext::<C, P>::with_policy(context.parent().clone(), context.axis_extent().clone())
            .with_axis_name(context.axis_name().map(str::to_owned))
            .with_axis_sharding(context.axis_sharding().clone());
        P::batched_type(&context, input, declared)
    }

    #[inline]
    fn unbatched_type(batch: &Self::Batch) -> Cow<'_, C::Type> {
        P::unbatched_type(batch)
    }

    #[inline]
    fn boundary_inputs(axis_extent: &Self::Extent) -> Vec<C::Value> {
        P::boundary_inputs(axis_extent)
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
    ) -> Result<BoundaryPreservingBatchedProgram<C::Constant, C::Operation>, BatchingError> {
        P::adapt_batched_program(program, required_output_axes, collapse_fn)
    }
}

impl<C: Context<Type = ArrayType>> RecursiveBatchingPolicy<C> for ArrayBatchingPolicy
where
    C::Value: Broadcast + Transpose,
    C::Operation: BatchableOperation<C, ArrayBatchingPolicy>
        + BatchableOperation<TracingContext<C::Constant, C::Operation>, ArrayBatchingPolicy>
        + From<TransposeOperation>
        + From<BroadcastOperation>,
    TracingContext<C::Constant, C::Operation>:
        StagingContext<Type = ArrayType, Constant = C::Constant, Operation = C::Operation>,
{
    #[inline]
    fn batching_level(context: &BatchingContext<C, ArrayBatchingPolicy>) -> BatchingLevel<ArrayType> {
        BatchingLevel::new(
            BatchingLevelExtent::Static(*context.axis_extent()),
            context.axis_name().map(str::to_string),
            context.axis_sharding().clone(),
        )
    }

    fn batch_program_at_level(
        level: &BatchingLevel<ArrayType>,
        region: RegionRef<'_, C::Constant, C::Operation>,
        input_axes: &[BatchAxis],
        output_axes_policy: ProgramBatchingOutputAxesPolicy,
    ) -> Result<Self::BatchedProgram, BatchingError> {
        let BatchingLevelExtent::Static(extent) = level.extent() else {
            return Err(BatchingError::UnsupportedOperation {
                message: "homogeneous array batching requires a static batch extent".to_string(),
            });
        };
        region.batched_with_axis_name(
            *extent,
            level.axis_name().map(str::to_string),
            level.axis_sharding().clone(),
            input_axes,
            output_axes_policy,
        )
    }

    #[inline]
    fn batch_region(
        context: &BatchingContext<C, ArrayBatchingPolicy>,
        region: RegionRef<'_, C::Constant, C::Operation>,
        inputs: Vec<ArrayBatch<C::Value>>,
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        let region_mappings = RegionReplayMappings::new();
        region.interpret_with(
            inputs,
            |_, constant| Ok(ArrayBatch::replicated(context.parent().lift(constant.clone())?)),
            |instruction, instruction_inputs| {
                // Run the batching rule inside the source instruction's recorded origin so that every staged
                // instruction records where it came from.
                let regions = ReplayRegionDriver::new(region, instruction.regions(), &region_mappings)?;
                let regions = AttachedRegions::from(regions);
                let outputs = context.invoke_with_provenance_origin(instruction.provenance().clone(), || {
                    instruction.operation().batch(context, &RecursiveBatchingDriver::new(&regions), instruction_inputs)
                })?;
                Ok(outputs)
            },
        )
    }

    #[inline]
    fn align_batch_axis(
        context: &BatchingContext<C, ArrayBatchingPolicy>,
        batch: ArrayBatch<C::Value>,
        axis: Axis,
    ) -> Result<ArrayBatch<C::Value>, BatchingError> {
        batch.match_axis(axis, *context.axis_extent(), context.axis_sharding().clone())
    }

    #[inline]
    fn static_batch_axis_extent(context: &BatchingContext<C, Self>) -> Option<usize> {
        // The homogeneous policy's extent is one host `usize` fixed when the transform was constructed.
        Some(*context.axis_extent())
    }
}

impl<C: Context<Type = ArrayType> + Iota<C::Value>, P: BatchingPolicy<C, Batch = ArrayBatch<C::Value>>>
    DiagonalBatchingPolicy<C> for ArrayBatchingPolicy<P>
where
    C::Value: Gather,
{
    fn diagonal(
        context: &C,
        value: &C::Value,
        kept_axis: usize,
        removed_axis: usize,
    ) -> Result<C::Value, BatchingError> {
        // Homogeneous array contexts construct indices from host extents only, so every extent must be static.
        let r#type = value.r#type();
        let (indices_type, iota_dimension, dimensions, slice_sizes) =
            diagonal_gather_layout(r#type.as_ref(), kept_axis, removed_axis)?;
        if indices_type.shape().dimensions().iter().any(|dimension| dimension.value().is_none()) {
            return Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "cannot take the diagonal of a value of type `{type}` with dynamic extents in a homogeneous array \
                     context",
                ),
            });
        }
        let indices = context.iota(&indices_type, iota_dimension)?;
        Ok(value.gather(&indices, &dimensions, &slice_sizes, &GatherOptions::new())?)
    }
}

impl<C: Context<Type = ArrayType>> BatchingEntrypointPolicy<C> for ArrayBatchingPolicy
where
    C::Value: Broadcast + Transpose,
{
    fn pack_inputs(
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
            .map(|(input, input_batch_axis)| ArrayBatch::new(input, input_batch_axis))
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
                        normalized_batch_axis_type(batch.r#type().as_ref(), position, batching_context.axis_sharding())
                    })
                    .transpose()?
                    .flatten();
                let batch = if let Some(r#type) = normalized_type {
                    // A rank-preserving broadcast with identity output axes changes only the requested packed
                    // sharding placement. Rewrapping the result retains the original batch-axis declaration.
                    let output_axes = (0..batch.r#type().rank()).collect::<Vec<_>>();
                    let value = batch.value().clone().broadcast(r#type.clone(), output_axes.as_slice())?;
                    ArrayBatch::new(value, batch.batch_axis())?
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

/// [`BatchingPolicy`] for [`Program`]s over [`ArrayIrValue`](crate::ArrayIrValue)s.
///
/// [`ArrayIrType::Array`] members may carry a mapped axis. [`ArrayIrType::Dimension`] members are shared shape values
/// and therefore remain replicated. [`ArrayIrType::Reference`]s carry the mapped axis of their referents, which is
/// fixed by the input or allocation that produced the reference and can never be moved or broadcast by a rule;
/// references have no [`BatchingPolicyProjection`], so the reference operations own their array IR batching rules
/// directly. The mapped axis extent is itself an ordinary parent-owned dimension value, so dynamic extents remain as
/// Single Static Assignment (SSA) values rather than becoming transform metadata.
///
/// A mapped reference supplies its extent when the corresponding referent dimension is static or has exact bounds.
/// Otherwise, an explicit extent or mapped array must carry the same dimension identity. Unrelated dimensions are
/// rejected even when another input supplies a usable extent; reading mutable contents is not used to guess extents.
#[derive(Copy, Clone, Debug, Default)]
pub struct ArrayIrBatchingPolicy;

impl<C: Context<Type = ArrayIrType>> BatchingPolicy<C> for ArrayIrBatchingPolicy {
    type Batch = ArrayIrBatch<C::Value>;
    type Extent = C::Value;
    type BatchedProgram = ThreadedExtentBatchedProgram<C::Constant, C::Operation>;

    #[inline]
    fn batch(value: C::Value, batch_axis: BatchAxis) -> Result<Self::Batch, BatchingError> {
        ArrayIrBatch::new(value, batch_axis)
    }

    #[inline]
    fn replicated(value: C::Value) -> Self::Batch {
        ArrayIrBatch::replicated(value)
    }

    #[inline]
    fn value(batch: &Self::Batch) -> &C::Value {
        batch.value()
    }

    #[inline]
    fn batch_axis(batch: &Self::Batch) -> BatchAxis {
        batch.batch_axis()
    }

    fn batched_type(
        context: &BatchingContext<C, Self>,
        input: &Self::Batch,
        declared: &ArrayIrType,
    ) -> Result<ArrayIrType, BatchingError> {
        let Some(axis) = input.batch_axis_position() else {
            return Ok(declared.clone());
        };

        let extent = match context.axis_extent().r#type().as_ref() {
            ArrayIrType::Dimension(dimension) => dimension.simplified_dimension(),
            _ => return Err(TypeError::invalid("batch extent must be a dimension").into()),
        };

        match (declared, input.value().r#type().as_ref()) {
            (ArrayIrType::Array(declared), ArrayIrType::Array(actual)) => {
                Ok(declared.batched_declaration(actual, axis, &extent, context.axis_sharding())?.into())
            }
            (ArrayIrType::Reference(declared), ArrayIrType::Reference(actual)) => {
                // A mapped reference carries its batch axis in its referent, whose shape the operation relates.
                Ok(ReferenceType::new(declared.referent().batched_declaration(
                    actual.referent(),
                    axis,
                    &extent,
                    context.axis_sharding(),
                )?)
                .into())
            }
            (declared, _) => Err(BatchingError::UnsupportedOperation {
                message: format!("cannot batch mapped member type `{declared}`"),
            }),
        }
    }

    #[inline]
    fn unbatched_type(batch: &Self::Batch) -> Cow<'_, C::Type> {
        batch.r#type()
    }

    #[inline]
    fn boundary_inputs(axis_extent: &Self::Extent) -> Vec<C::Value> {
        // The adapted program's leading input still defines the `DimensionVariable` referenced by every inserted
        // dynamic batch dimension, so the first-class mapped extent value must become its matching program input.
        vec![axis_extent.clone()]
    }

    #[inline]
    fn adapt_batched_program<CollapseFn>(
        program: Self::BatchedProgram,
        required_output_axes: Option<&[BatchAxis]>,
        collapse_fn: CollapseFn,
    ) -> Result<BoundaryPreservingBatchedProgram<C::Constant, C::Operation>, BatchingError>
    where
        CollapseFn: Fn(
            &TracingContext<C::Constant, C::Operation>,
            Tracer<TracingContext<C::Constant, C::Operation>>,
            Axis,
        ) -> Result<Tracer<TracingContext<C::Constant, C::Operation>>, BatchingError>,
    {
        // Drop the leading forwarded extent bookkeeping output that exists only for extent-threading consumers.
        let (program, output_axes) = program.into_parts();
        BoundaryPreservingBatchedProgram::from_widened_boundary(
            program,
            output_axes,
            required_output_axes,
            1,
            collapse_fn,
        )
    }
}

impl<C: Context<Type = ArrayIrType>> BatchingPolicyProjection<C, ArrayType> for ArrayIrBatchingPolicy
where
    C::Value: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Operation: OperationProjection<ArrayType>,
    ProjectedContext<C, ArrayType>: Context<
            Type = ArrayType,
            Value = <C::Value as ValueProjection<ArrayType>>::Projected,
            Constant = <C::Constant as ValueProjection<ArrayType>>::Projected,
            Operation = <C::Operation as OperationProjection<ArrayType>>::Projected,
        >,
{
    type Projected = ArrayBatchingPolicy<DynamicArrayExtentBatchingPolicy>;
}

// First-class dimensions are shared shape metadata. Their member operations bind once in the parent context;
// the array IR carrier constructor rejects a mapped axis before projection.
impl<C: ContextProjection<DimensionType, Type = ArrayIrType>> ReplicatedBatchingPolicyProjection<C, DimensionType>
    for ArrayIrBatchingPolicy
{
    fn project_replicated_batch(
        batch: &ArrayIrBatch<C::Value>,
    ) -> Result<<C::Value as ValueProjection<DimensionType>>::Projected, BatchingError> {
        let value = ValueProjection::<DimensionType>::into_projected(batch.value().clone())?;
        if !batch.batch_axis().is_replicated() {
            return Err(BatchingError::MappedDimension {
                r#type: Box::new(value.r#type().into_owned()),
                axis: batch.batch_axis(),
            });
        }
        Ok(value)
    }
}

impl<C: ContextProjection<ArrayType, Type = ArrayIrType>> RecursiveBatchingPolicy<C> for ArrayIrBatchingPolicy
where
    C::Value: DimensionSize + DynamicBroadcast + ValueProjection<ArrayType>,
    <C::Value as ValueProjection<ArrayType>>::Projected: Transpose + Value<Type = ArrayType>,
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>
        + ValueProjection<DimensionType, Projected = DimensionValue>,
    C::Operation: BatchableOperation<C, ArrayIrBatchingPolicy>
        + BatchableOperation<TracingContext<C::Constant, C::Operation>, ArrayIrBatchingPolicy>
        + From<DynamicBroadcastOperation>
        + From<ConstantOperation<DimensionValue>>
        + From<DimensionSizeOperation>
        + OperationProjection<ArrayType, Projected: From<TransposeOperation>>,
    TracingContext<C::Constant, C::Operation>:
        StagingContext<Type = ArrayIrType, Constant = C::Constant, Operation = C::Operation>,
{
    #[inline]
    fn batching_level(context: &BatchingContext<C, Self>) -> BatchingLevel<ArrayIrType> {
        BatchingLevel::new(
            BatchingLevelExtent::Dynamic(context.axis_extent().r#type().into_owned()),
            context.axis_name().map(str::to_string),
            context.axis_sharding().clone(),
        )
    }

    fn batch_program_at_level(
        level: &BatchingLevel<ArrayIrType>,
        region: RegionRef<'_, C::Constant, C::Operation>,
        input_axes: &[BatchAxis],
        output_axes_policy: ProgramBatchingOutputAxesPolicy,
    ) -> Result<Self::BatchedProgram, BatchingError> {
        // There is deliberately no `ArrayIrBatchingTransform` analogous to `ArrayBatchingTransform` here. Homogeneous
        // array batching is determined entirely by a sealed region plus static host metadata, including a `usize`
        // mapped extent, so that complete request can key a region-owned artifact directly. Array IR batching instead
        // receives its extent as a live `C::Value` (i.e., a single static assignment value owned by the caller's
        // current parent context). Such a value is neither stable cache-key material nor something a retained region
        // artifact may capture. This function makes the required template/runtime split: the level carries only the
        // live extent's _type_, which becomes a fresh leading input/output of the transformed program, while each
        // consumer supplies the current extent through `BatchingPolicy::boundary_inputs` and removes the bookkeeping
        // output through `BatchingPolicy::adapt_batched_program`. A future cache could therefore retain this widened,
        // context-neutral template. It would need its own marker and a complete normalized key containing the extent
        // type and identity contract, named-axis scope, sharding, input axes, and output policy. The live extent would
        // remain a per-call boundary input. The current measured cache consumer is the simpler homogeneous path, so
        // adding that separate specialization mechanism here would be premature rather than technically impossible.
        check_count!("input", input_axes, region.input_types().len(), ProgramError);
        let BatchingLevelExtent::Dynamic(extent_type) = level.extent() else {
            return Err(BatchingError::UnsupportedOperation {
                message: "array IR batching requires a first-class batch extent".to_string(),
            });
        };
        let extent_type = <&DimensionType>::try_from(extent_type)?.clone();
        let extent_dimension = extent_type.simplified_dimension();
        let parent_context = TracingContext::<C::Constant, C::Operation>::new();
        let builder = parent_context.builder().clone();

        // The fresh structural trace cannot refer to the parent trace's mapped-extent Single Static Assignment (SSA)
        // value directly. Give the transformed region one leading dimension input and carry that same atom out as its
        // leading output. Every inserted packed batch axis below references this input's type identity.
        let (output_atom_ids, output_axes) = {
            let extent = parent_context.input(extent_type.into());
            let extent_atom_id = extent.atom_id()?;
            let batching_context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(parent_context, extent)
                .with_axis_name(level.axis_name().map(str::to_string))
                .with_axis_sharding(level.axis_sharding().clone());

            let inputs = region
                .input_types()
                .iter()
                .zip(input_axes)
                .map(|(unbatched_type, batch_axis)| {
                    // Inserts the mapped extent (and its sharding placement) into one per-item array type. A reference
                    // is batched by batching its referent, so both the array and the reference arm share this step.
                    let insert_batch_dimension = |array_type: &ArrayType, axis: Axis| -> Result<_, BatchingError> {
                        let batched_rank = array_type.rank() + 1;
                        let position = axis.normalize(batched_rank).map_err(|_| {
                            BatchingError::BatchAxisOutOfBounds { r#type: Box::new(array_type.clone()), axis }
                        })?;
                        array_type.batched(position, extent_dimension.clone(), level.axis_sharding().clone())
                    };
                    let batched_type = match (unbatched_type, batch_axis.axis()) {
                        (ArrayIrType::Array(array_type), Some(axis)) => {
                            ArrayIrType::Array(insert_batch_dimension(array_type, axis)?)
                        }
                        (ArrayIrType::Reference(reference_type), Some(axis)) => {
                            ReferenceType::new(insert_batch_dimension(reference_type.referent(), axis)?).into()
                        }
                        _ => unbatched_type.clone(),
                    };
                    let value = batching_context.parent().input(batched_type);
                    ArrayIrBatch::new(value, *batch_axis)
                })
                .collect::<Result<Vec<_>, BatchingError>>()?;

            let outputs = <Self as RecursiveBatchingPolicy<TracingContext<C::Constant, C::Operation>>>::batch_region(
                &batching_context,
                region,
                inputs,
            )?;

            // Resolve the policy into one optional target axis per output. `None` keeps the axis the rule produced.
            let output_target_axes = match &output_axes_policy {
                ProgramBatchingOutputAxesPolicy::Natural => vec![None; outputs.len()],
                ProgramBatchingOutputAxesPolicy::AlignAllTo(axis) => {
                    vec![Some(BatchAxis::new(*axis)); outputs.len()]
                }
                ProgramBatchingOutputAxesPolicy::AlignEachTo(axes) => {
                    check_count!("output", axes, outputs.len(), ProgramError);
                    axes.iter().map(|axis| (!axis.is_replicated()).then_some(*axis)).collect()
                }
            };

            let mut output_atom_ids = Vec::with_capacity(outputs.len() + 1);
            let mut output_axes = Vec::with_capacity(outputs.len());
            output_atom_ids.push(extent_atom_id);
            for (output, target_axis) in outputs.into_iter().zip(output_target_axes) {
                let output = match target_axis {
                    Some(target_axis) => {
                        Self::align_batch_axis(&batching_context, output, target_axis.axis().unwrap())?
                    }
                    None => output,
                };
                check_builders!(&builder, output.value().builder())?;
                output_axes.push(output.batch_axis());
                output_atom_ids.push(output.into_value().atom_id()?);
            }

            Ok::<_, BatchingError>((output_atom_ids, output_axes))
        }?;

        // The batching context and every tracer it produced were dropped with the block above, so the builder must be
        // uniquely owned here. A remaining reference means that a tracer escaped the structural trace.
        let input_count = region.input_types().len() + 1;
        let output_count = output_atom_ids.len();
        let builder = Rc::try_unwrap(builder).map_err(|_| ProgramError::EscapedProgramBuilder)?.into_inner();
        let program = builder
            .build(output_atom_ids, vec![Placeholder; input_count], vec![Placeholder; output_count])?
            .into_simplified()?;
        Ok(ThreadedExtentBatchedProgram::new(program, output_axes)?)
    }

    fn batch_region(
        context: &BatchingContext<C, Self>,
        region: RegionRef<'_, C::Constant, C::Operation>,
        inputs: Vec<Self::Batch>,
    ) -> Result<Vec<Self::Batch>, BatchingError> {
        // Regions are replayed through the checked constant and operation boundaries.
        let region_mappings = RegionReplayMappings::new();
        region.interpret_with(
            inputs,
            |_, constant| {
                // Constants are lifted through the checked constructor. The infallible replicated wrapper would let
                // a reference-typed capture constant ride through structural batching unchanged.
                <Self as BatchingPolicy<C>>::batch(context.parent().lift(constant.clone())?, BatchAxis::replicated())
            },
            |instruction, instruction_inputs| {
                // Run the batching rule inside the source instruction's recorded origin so that every staged
                // instruction records where it came from.
                let regions = ReplayRegionDriver::new(region, instruction.regions(), &region_mappings)?;
                let regions = AttachedRegions::from(regions);
                let outputs = context.invoke_with_provenance_origin(instruction.provenance().clone(), || {
                    instruction.operation().batch(context, &RecursiveBatchingDriver::new(&regions), instruction_inputs)
                })?;
                Ok(outputs)
            },
        )
    }

    fn align_batch_axis(
        context: &BatchingContext<C, Self>,
        batch: Self::Batch,
        axis: Axis,
    ) -> Result<Self::Batch, BatchingError> {
        // Aligns one array IR batch to `axis`, moving an existing mapped axis or dynamically broadcasting a
        // replicated array with the context's first-class extent. A reference's batch axis is fixed by its referent,
        // so a reference batch is returned unchanged when it already carries `axis` and rejected otherwise; first-class
        // dimensions have no packed axis and are always rejected. The mapped move transposes through the value's
        // projected `Transpose` capability. Dynamic broadcasts are emitted through the parent context.
        //
        // A first-class dimension has no array axis to align.
        if let ArrayIrType::Dimension(r#type) = batch.unbatched_type() {
            return Err(BatchingError::MappedDimension { r#type: Box::new(r#type), axis: BatchAxis::from(axis) });
        }

        if let ArrayIrType::Reference(r#type) = batch.value.r#type().as_ref() {
            // A mapped reference already carrying the requested axis is an identity alignment. Anything else would have
            // to move or broadcast shared storage, which no reference can do. The request is normalized against the
            // referent's packed rank, so an out-of-bounds axis is reported as such rather than as an alignment failure.
            let position = axis.normalize(r#type.referent().rank()).map_err(|_| {
                BatchingError::BatchAxisOutOfBounds { r#type: Box::new(r#type.referent().clone()), axis }
            })?;

            if Some(position) == batch.batch_axis_position() {
                return Ok(batch);
            }

            return Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "a reference's batch axis is fixed by its referent, so a reference batch carrying {} cannot be \
                     aligned to axis {}; pass the reference as a batched input at that axis instead",
                    batch.batch_axis, axis,
                ),
            });
        }

        let Some(current_position) = batch.batch_axis_position() else {
            // A replicated array gains the mapped axis through one dynamic broadcast along the transform's own extent.
            let unbatched_type = batch.unbatched_type();
            let array_type = <&ArrayType>::try_from(&unbatched_type)?;
            let output_rank = array_type.rank() + 1;
            let position = axis
                .normalize(output_rank)
                .map_err(|_| BatchingError::BatchAxisOutOfBounds { r#type: Box::new(array_type.clone()), axis })?;
            let value = batch.into_value();
            let value = broadcast_replicated_array(
                value,
                array_type,
                position,
                context.axis_extent(),
                context.axis_sharding(),
            )?;
            return ArrayIrBatch::new(value, BatchAxis::from_position(position));
        };

        // A mapped array moves its axis with one transpose that keeps every other axis in its relative order.
        // The target is normalized against the packed value's unchanged rank.
        let packed_type = batch.value.r#type();
        let packed_type = <&ArrayType>::try_from(packed_type.as_ref())?;
        let rank = packed_type.rank();
        let position = axis
            .normalize(rank)
            .map_err(|_| BatchingError::BatchAxisOutOfBounds { r#type: Box::new(packed_type.clone()), axis })?;
        if current_position == position {
            return Ok(batch);
        }
        let mut permutation = (0..rank).filter(|axis| *axis != current_position).collect::<Vec<_>>();
        permutation.insert(position, current_position);
        let value = batch.into_value();
        let projected = <C::Value as ValueProjection<ArrayType>>::into_projected(value)?;
        let value = C::Value::from_projected(projected.transpose(permutation)?);
        ArrayIrBatch::new(value, BatchAxis::from_position(position))
    }

    fn static_batch_axis_extent(context: &BatchingContext<C, Self>) -> Option<usize> {
        // Exact dimension bounds establish the size even when the extent is a traced instruction. Otherwise, ask the
        // parent to resolve the first-class extent to a concrete dimension constant (e.g., under eager batching).
        // A symbolic staged or opaque extent without an exact type has no static size.
        if let ArrayIrType::Dimension(axis_type) = context.axis_extent().r#type().as_ref()
            && let Some(extent) = axis_type.extent()
        {
            return Some(extent);
        }
        match context.parent().resolve(context.axis_extent()) {
            ValueResolution::Constant(axis_extent) => {
                <C::Constant as ValueProjection<DimensionType>>::into_projected(axis_extent)
                    .ok()
                    .map(|axis_extent| axis_extent.extent())
            }
            ValueResolution::Staged(_) | ValueResolution::Opaque => None,
        }
    }
}

impl<C: Context<Type = ArrayIrType> + DynamicIota<C::Value>> DiagonalBatchingPolicy<C> for ArrayIrBatchingPolicy
where
    C::Value: DimensionSize + ValueProjection<ArrayType>,
    <C::Value as ValueProjection<ArrayType>>::Projected: Gather,
{
    fn diagonal(
        context: &C,
        value: &C::Value,
        kept_axis: usize,
        removed_axis: usize,
    ) -> Result<C::Value, BatchingError> {
        // Dynamic extents are read from `value` itself, so the indices share their dimension identities
        // with the paired input axes. Static extents consume no dimension values.
        let r#type = value.r#type();
        let r#type = <&ArrayType>::try_from(r#type.as_ref())?;
        let (indices_type, iota_dimension, dimensions, slice_sizes) =
            diagonal_gather_layout(r#type, kept_axis, removed_axis)?;
        let mut extents = Vec::new();
        for axis in (0..r#type.rank()).filter(|axis| *axis != removed_axis) {
            if r#type.dimension(axis).value().is_none() {
                extents.push(value.dimension_size(axis)?);
            }
        }
        let indices = context.dynamic_iota(&indices_type, iota_dimension, &extents)?;
        let value = <C::Value as ValueProjection<ArrayType>>::into_projected(value.clone())?;
        let indices = <C::Value as ValueProjection<ArrayType>>::into_projected(indices)?;
        let diagonal = value.gather(&indices, &dimensions, &slice_sizes, &GatherOptions::new())?;
        Ok(<C::Value as ValueProjection<ArrayType>>::from_projected(diagonal))
    }
}

impl<C: ContextProjection<ArrayType, Type = ArrayIrType>> BatchingEntrypointPolicy<C> for ArrayIrBatchingPolicy
where
    C::Value: DimensionSize + Assert + DynamicBroadcast + ValueProjection<ArrayType> + ValueProjection<DimensionType>,
    <C::Value as ValueProjection<ArrayType>>::Projected: Transpose + Value<Type = ArrayType>,
    <C::Value as ValueProjection<DimensionType>>::Projected: Compare<C::Value>,
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>
        + ValueProjection<DimensionType, Projected = DimensionValue>,
    C::Operation: BatchableOperation<C, ArrayIrBatchingPolicy>
        + BatchableOperation<TracingContext<C::Constant, C::Operation>, ArrayIrBatchingPolicy>
        + From<DynamicBroadcastOperation>
        + From<ConstantOperation<DimensionValue>>
        + From<DimensionSizeOperation>
        + OperationProjection<ArrayType, Projected: From<TransposeOperation>>,
    TracingContext<C::Constant, C::Operation>:
        StagingContext<Type = ArrayIrType, Constant = C::Constant, Operation = C::Operation>,
{
    fn pack_inputs(
        context: &C,
        inputs: Vec<C::Value>,
        input_batch_axes: Vec<BatchAxis>,
        batch_axis: BatchAxisSpecification<Self::Extent>,
    ) -> Result<(BatchingContext<C, Self>, Vec<Self::Batch>), BatchingError> {
        if inputs.len() != input_batch_axes.len() {
            return Err(
                ProgramError::InvalidInputCount { expected: inputs.len(), actual: input_batch_axes.len() }.into()
            );
        }

        let batches = inputs
            .into_iter()
            .zip(input_batch_axes)
            .map(|(input, input_batch_axis)| ArrayIrBatch::new(input, input_batch_axis))
            .collect::<Result<Vec<_>, _>>()?;

        // An explicitly supplied extent must be a first-class dimension value. It then acts as the reference every
        // mapped input's extent is checked against below; without one, the first mapped input's extent takes that
        // role.
        let mut axis_extent = batch_axis.extent().cloned();
        if let Some(axis_extent) = &axis_extent {
            let extent_type = axis_extent.r#type();
            <&DimensionType>::try_from(extent_type.as_ref())?;
        }

        // Mapped arrays are visited before mapped references so that the extent evidence a reference is validated
        // against is complete regardless of the order in which the caller listed the inputs.
        let is_reference =
            |batch: &&ArrayIrBatch<C::Value>| matches!(batch.value.r#type().as_ref(), ArrayIrType::Reference(_));
        let arrays_first =
            batches.iter().filter(|batch| !is_reference(batch)).chain(batches.iter().filter(is_reference));
        for batch in arrays_first {
            let Some(position) = batch.batch_axis_position() else {
                continue;
            };

            // A reference's extent is available without reading its contents only when its bounds are exact or a
            // dimension value or mapped array supplies the same symbolic dimension. An unrelated extent is not
            // evidence as accepting it would silently give different inputs different batch lengths.
            let input_extent = match batch.value.r#type().as_ref() {
                ArrayIrType::Reference(r#type) => match r#type.referent().shape().dimensions().get(position) {
                    Some(Dimension::Static(extent)) => context.dimension_constant(*extent)?,
                    Some(Dimension::Dynamic(variable)) if DimensionType::from(variable.clone()).extent().is_some() => {
                        context.dimension_constant(DimensionType::from(variable.clone()).extent().unwrap())?
                    }
                    Some(Dimension::Dynamic(variable))
                        if axis_extent.as_ref().is_some_and(|extent| {
                            matches!(extent.r#type().as_ref(), ArrayIrType::Dimension(r#type)
                                if r#type.variable() == Some(variable))
                        }) || batches.iter().any(|candidate| {
                            let Some(axis) = candidate.batch_axis_position() else { return false };
                            matches!(candidate.value.r#type().as_ref(), ArrayIrType::Array(r#type)
                                if r#type.shape().dimensions()[axis].variable() == Some(variable))
                        }) =>
                    {
                        // A symbolic referent extent whose identity is already carried by the explicit extent or
                        // by a mapped array needs no further evidence (equality is implied by the shared dimension
                        // identity).
                        continue;
                    }
                    _ => {
                        return Err(BatchingError::UnsupportedOperation {
                            message: format!(
                                "cannot infer the mapped extent from reference type {} whose batch axis {} has \
                                 a dynamic extent; pass an extent or a mapped array with the same dimension identity",
                                r#type, position,
                            ),
                        });
                    }
                },
                _ => {
                    // A mapped array's extent is read from its packed batch axis.
                    batch.value.dimension_size(position)?
                }
            };

            if let Some(axis_extent) = &axis_extent {
                // Dimension comparisons fold known relationships and retain an assertion when either extent is symbolic.
                <C::Value as ValueProjection<DimensionType>>::into_projected(axis_extent.clone())?
                    .compare(
                        &<C::Value as ValueProjection<DimensionType>>::into_projected(input_extent.clone())?,
                        ComparisonDirection::Equal,
                    )?
                    .assert(
                        "batch dimensions must agree",
                        &[("expected", axis_extent.clone()), ("actual", input_extent)],
                    )?;
            } else {
                axis_extent = Some(input_extent);
            }
        }

        // With neither an explicit extent nor a mapped input there is nothing that determines the batch size.
        let axis_extent = axis_extent.ok_or(BatchingError::EmptyBatch)?;

        // A mapped reference's referent is batched like an array, so its batch-axis placement participates in the
        // sharding join alongside the mapped arrays.
        let axis_sharding = batch_axis_sharding(batches.iter().filter_map(|batch| {
            let array_type = match batch.value.r#type() {
                Cow::Borrowed(ArrayIrType::Array(array_type)) => Cow::Borrowed(array_type),
                Cow::Owned(ArrayIrType::Array(array_type)) => Cow::Owned(array_type),
                Cow::Borrowed(ArrayIrType::Reference(r#type)) => Cow::Borrowed(r#type.referent()),
                Cow::Owned(ArrayIrType::Reference(r#type)) => Cow::Owned(r#type.referent().clone()),
                _ => return None,
            };
            Some((array_type, batch.batch_axis_position()))
        }))?;
        let batching_context = BatchingContext::new(context.clone(), axis_extent)
            .with_axis_name(batch_axis.name().map(String::from))
            .with_axis_sharding(axis_sharding);
        let batches = batches
            .into_iter()
            .enumerate()
            .map(|(index, batch)| -> Result<_, BatchingError> {
                // Only a mapped array has a batch-axis placement to normalize. A reference's referent placement is
                // part of the shared storage and cannot be rewritten by a broadcast, so it must already agree with the
                // placement selected for the batch axis.
                let Some(position) = batch.batch_axis_position() else {
                    return Ok(batch);
                };
                let value_type = batch.value.r#type();
                let array_type = match value_type.as_ref() {
                    ArrayIrType::Array(array_type) => array_type,
                    ArrayIrType::Reference(r#type) => {
                        let axis_sharding = batching_context.axis_sharding();
                        if normalized_batch_axis_type(r#type.referent(), position, axis_sharding)?.is_some() {
                            return Err(BatchingError::MisalignedBatchAxes {
                                message: format!(
                                    "mapped reference input {} of type {} does not place its batch axis \
                                     {} on {}; a reference's referent placement is fixed by its storage \
                                     and cannot be renormalized",
                                    index, r#type, position, axis_sharding,
                                ),
                            });
                        }
                        return Ok(batch);
                    }
                    ArrayIrType::Dimension(_) => return Ok(batch),
                };
                let Some(normalized_type) =
                    normalized_batch_axis_type(array_type, position, batching_context.axis_sharding())?
                else {
                    return Ok(batch);
                };

                // The mapped axis takes the transform's own extent, and every other axis is either an exact constant
                // or a `dimension_size` read of the input being renormalized.
                let output_dimensions = (0..array_type.rank())
                    .map(|axis| match axis == position {
                        true => Ok(batching_context.axis_extent().clone()),
                        false => batch.value.dimension_size(axis),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let output_axes = (0..array_type.rank()).collect::<Vec<_>>();
                let batch_axis = batch.batch_axis;
                let value = batch.value.dynamic_broadcast_with_output_sharding(
                    &output_dimensions,
                    &output_axes,
                    normalized_type.sharding().cloned(),
                )?;
                ArrayIrBatch::new(value, batch_axis)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok((batching_context, batches))
    }

    fn materialize_output(
        context: &BatchingContext<C, Self>,
        output: Self::Batch,
        output_batch_axis: BatchAxis,
    ) -> Result<C::Value, BatchingError> {
        if let ArrayIrType::Reference(r#type) = output.value().r#type().as_ref() {
            // A reference output carries the batch axis of the root it forwards or of its allocation. That axis is
            // fixed by the referent, so the request must already agree with it (a replicated request for a replicated
            // reference, or a mapped request naming the referent's own mapped axis).
            let aligned = match (output.batch_axis().axis(), output_batch_axis.axis()) {
                (None, None) => true,
                (Some(_), Some(_)) => {
                    r#type.referent().normalize_batch_axis(output_batch_axis)?.0 == output.batch_axis()
                }
                _ => false,
            };

            if aligned {
                return Ok(output.into_value());
            }

            return Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "a reference output carrying {} cannot be materialized with {} requested because \
                     a reference's batch axis is fixed by its referent",
                    output.batch_axis(),
                    output_batch_axis,
                ),
            });
        }

        // A mapped output cannot be presented as replicated, because dropping the axis would need a reduction the
        // caller did not ask for. A mapped request is satisfied by alignment, which moves an existing axis or
        // broadcasts a replicated output along the transform's extent.
        match (output.batch_axis.axis(), output_batch_axis.axis()) {
            (None, None) => Ok(output.into_value()),
            (Some(_), None) => {
                Err(BatchingError::MismatchedOutputAxes { expected: output_batch_axis, actual: output.batch_axis })
            }
            (_, Some(axis)) => {
                Ok(<Self as RecursiveBatchingPolicy<C>>::align_batch_axis(context, output, axis)?.into_value())
            }
        }
    }
}

impl<C: ContextProjection<ArrayType, Type = ArrayIrType>, T: Type> ValueProjection<T>
    for BatchingTracer<C, ArrayIrBatchingPolicy>
where
    C::Value: DimensionSize + DynamicBroadcast + ValueProjection<ArrayType>,
    <C::Value as ValueProjection<ArrayType>>::Projected: Transpose + Value<Type = ArrayType>,
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>
        + ValueProjection<DimensionType, Projected = DimensionValue>,
    C::Operation: BatchableOperation<C, ArrayIrBatchingPolicy>
        + BatchableOperation<TracingContext<C::Constant, C::Operation>, ArrayIrBatchingPolicy>
        + From<DynamicBroadcastOperation>
        + From<ConstantOperation<DimensionValue>>
        + From<DimensionSizeOperation>
        + OperationProjection<ArrayType, Projected: From<TransposeOperation>>,
    for<'t> &'t T: TryFrom<&'t ArrayIrType, Error = TypeError>,
    TracingContext<C::Constant, C::Operation>:
        StagingContext<Type = ArrayIrType, Constant = C::Constant, Operation = C::Operation>,
{
    type Projected = ProjectedValue<T, Self>;
    type ProjectedRef<'v>
        = ProjectedValue<T, &'v Self>
    where
        Self: 'v,
        T: 'v;

    #[inline]
    fn from_projected(value: Self::Projected) -> Self {
        value.into_value()
    }

    #[inline]
    fn projected<'v>(&'v self) -> Result<Self::ProjectedRef<'v>, TypeError>
    where
        T: 'v,
    {
        let unbatched_type = <ArrayIrBatchingPolicy as BatchingPolicy<C>>::unbatched_type(self.batch());
        Ok(ProjectedValue::new(self, <&T>::try_from(unbatched_type.as_ref())?.clone()))
    }

    #[inline]
    fn into_projected(self) -> Result<Self::Projected, TypeError> {
        let unbatched_type = <ArrayIrBatchingPolicy as BatchingPolicy<C>>::unbatched_type(self.batch());
        let r#type = <&T>::try_from(unbatched_type.as_ref())?.clone();
        Ok(ProjectedValue::new(self, r#type))
    }
}

// Blanket `BatchableOperation` implementation for any `ElementwiseOperation`, so per-operation `BatchableOperation`
// implementations do not have to be written for elementwise primitives (e.g., `ZeroLike`, `OneLike`, `Add`, `Sub`,
// `Mul`, `Div`, `Neg`, `Sin`, `Cos`, `Select`, etc.). Operations with non-trivial axis arithmetic (e.g., `Dot`,
// `Transpose`, `Reshape`, etc.) keep their explicit implementations. Coherence is preserved because none of those types
// implement `ElementwiseOperation`. The rule follows JAX's `defbroadcasting` policy where every input is broadcast to
// the common batched shape before the operation is applied, so the value-level primitive only ever sees inputs that
// agree on shape. When no input is mapped there is no batch axis to thread, and so the inputs are interpreted as given
// and every output is replicated. Otherwise, mapped inputs retain the first mapped input's position when that position
// is valid for every mapped input. For mixed ranks where it is not, they are temporarily realigned to leading axis `0`.
// Every input is then broadcast to the common per-item shape with the batch axis inserted there. Inputs whose per-item
// shapes are not broadcast-compatible are left at their batch-axis-inserted shapes so the operation surfaces its own
// shape error.
impl<C: Context<Type = ArrayType>, O: ElementwiseOperation + InterpretableOperation<C>, M: ArrayExtentBatchingPolicy<C>>
    BatchableOperation<C, ArrayBatchingPolicy<M>> for O
where
    C::Value: Transpose,
{
    #[inline]
    fn batch<D: BatchingDriver<C, ArrayBatchingPolicy<M>>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy<M>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        batch_elementwise(
            context,
            inputs,
            &|input_types| Ok(Operation::infer_output_types(self, input_types, &[])?),
            &|inputs, output_batch_axes| self.interpret_with_batch_axes(context, inputs, output_batch_axes),
        )
    }
}

/// Internal helper that batches an elementwise operation application following the policy documented on the blanket
/// [`BatchableOperation`] implementation above, which is its only caller. The operation enters only through
/// `infer_output_types` (its output type inference) and `interpret` (its interpretation over the aligned inputs with
/// the provided output batch axes), so this body is compiled once per context and extent policy rather than once per
/// elementwise operation type. Keep it free of the operation type: every closure and iterator adapter in it is
/// instantiated per distinct signature.
#[allow(clippy::type_complexity)]
fn batch_elementwise<C: Context<Type = ArrayType>, M: ArrayExtentBatchingPolicy<C>>(
    context: &BatchingContext<C, ArrayBatchingPolicy<M>>,
    inputs: &[ArrayBatch<C::Value>],
    infer_output_types: &dyn Fn(&[ArrayType]) -> Result<Vec<ArrayType>, BatchingError>,
    interpret: &dyn Fn(&[ArrayBatch<C::Value>], &[BatchAxis]) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError>,
) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError>
where
    C::Value: Transpose,
{
    let Some(output_batch_axis_position) = inputs.iter().find_map(ArrayBatch::batch_axis_position) else {
        let packed_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        let output_types = infer_output_types(packed_types.as_slice())?;
        return interpret(inputs, &vec![BatchAxis::replicated(); output_types.len()]);
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
            if let Some(sharding) = unbatched_type.sharding.as_mut() {
                let varying_manual_axes = axis_sharding.manual_axes(sharding.mesh());
                sharding
                    .extend_varying_manual_axes(varying_manual_axes)
                    .map_err(|error| BatchingError::MisalignedBatchAxes { message: error.to_string() })?;
            }
            Ok(unbatched_type)
        })
        .collect::<Result<Vec<_>, _>>()?;

    let inputs = inputs.iter().map(|input| input.move_axis(batch_axis)).collect::<Result<Vec<_>, _>>()?;

    // Broadcast every input to the common unbatched per-item shape when one exists. Otherwise, retain each input's
    // own per-item shape and let the operation's inference report the incompatibility.
    let common_unbatched_type = Broadcastable::broadcasted(unbatched_types.as_slice()).ok();
    let axis_dimension = M::axis_dimension(context)?;
    let broadcasted_inputs = inputs
        .iter()
        .zip(unbatched_types.iter())
        .map(|(input, unbatched_type)| -> Result<ArrayBatch<C::Value>, BatchingError> {
            let mut target_type = common_unbatched_type.as_ref().unwrap_or(unbatched_type).clone();
            target_type.data_type = unbatched_type.data_type();
            let batched_type =
                target_type.batched(batch_axis_position, axis_dimension.clone(), axis_sharding.clone())?;
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
                                "cannot locate a source for elementwise output dimension {target_dimension}",
                            ))
                            .into()
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            M::broadcast_input(context, input, batched_type, output_axes, batch_axis, dimension_sources)
        })
        .collect::<Result<Vec<_>, _>>()?;

    let input_types = broadcasted_inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
    let output_types = infer_output_types(input_types.as_slice())?;
    let output_batch_axes = vec![BatchAxis::new(batch_axis); output_types.len()];
    interpret(&broadcasted_inputs, &output_batch_axes)?
        .into_iter()
        .map(|output| output.move_axis(output_batch_axis_position))
        .collect::<Result<Vec<_>, _>>()
}

impl<
    V: Value<Type = ArrayType>,
    O: Operation<Type = ArrayType>
        + BatchableOperation<TracingContext<V, O>, ArrayBatchingPolicy>
        + From<TransposeOperation>
        + From<BroadcastOperation>,
> RegionRef<'_, V, O>
where
    TracingContext<V, O>: StagingContext<Type = ArrayType, Constant = V, Operation = O>,
{
    /// Structurally batches this borrowed homogeneous-array [`Region`] so that the resulting program operates over
    /// inputs batched along the specified [`BatchAxis`]s. Staged higher-order [`BatchableOperation`] implementations
    /// use this function to batch captured programs _without_ concretizing any batch-item values, so that batched
    /// control-flow and composite structure can be staged back into the enclosing trace. This function replays the
    /// region through an [`ArrayBatchingPolicy`] [`BatchingContext`] over a fresh [`TracingContext`], lifts every
    /// instruction through its [`BatchableOperation`] rule, and extracts the resulting staged program together with the
    /// requested [`ProgramBatchingOutputAxesPolicy`].
    ///
    /// This method is intentionally specific to [`ArrayType`]. It constructs mapped batched input types by inserting
    /// static [`Dimension`]s, rewrites array sharding metadata, and materializes output axes through [`ArrayBatch`].
    /// Heterogeneous program families with first-class dynamic dimension extent values instead own structural recursion
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
    /// [`ProgramBatchingOutputAxesPolicy::AlignAllTo`] imposes one canonical
    /// output axis when a consumer explicitly requires one common layout.
    ///
    /// Successful structural results are retained in this sealed region's [`Transform`] namespace. The complete key
    /// contains the static extent, named-axis scope, mapped-axis sharding, normalized input axes, and normalized output
    /// policy. Equal repeated requests therefore clone the retained program boundary instead of replaying every
    /// operation rule again; changed arguments derive an independent bounded specialization. Unlike the Jacobian-Vector
    /// Product (JVP), linearization, and transposition boundaries, which hand back the retained [`Arc`]-shared
    /// [`Program`] itself, [`BoundaryPreservingBatchedProgram`] owns its [`Program`], so a hit returns an owned deep
    /// clone of the retained program. Repeated batched attachments therefore never dedupe by [`Arc`] identity, and the
    /// cache removes replay cost only.
    ///
    /// Dynamic per-item dimensions are supported. The mapped axis is always _freshly inserted_ as [`Dimension::Static`]
    /// carrying the caller's `axis_size`, so program-level batching never reads a batch extent off an input type and
    /// consequently never raises [`BatchingError::DynamicBatchAxis`] the way the value-level
    /// [`Batch::batch`](crate::Batch::batch) entry point does, which must recover the batch size from a mapped value's
    /// own type. A dynamic dimension in an input's per-item type is therefore never the batch axis: it crosses the
    /// rewritten boundary unchanged, and where a [`BatchableOperation`] rule needs its runtime extent (such as the
    /// broadcast that aligns a replicated elementwise input against a mapped one) that extent is resolved through
    /// [`DimensionSource`] from the source axis that supplied it, or rejected with an exact type diagnostic when no
    /// source axis carries it.
    ///
    /// # Parameters
    ///
    ///   - `axis_size`: Dimension of the new batch axis.
    ///   - `axis_sharding`: Sharding placement assigned to every newly materialized batch axis.
    ///   - `input_batch_axes`: [`BatchAxis`] for each input (i.e., argument) of this [`Program`].
    ///   - `output_axes_policy`: [`ProgramBatchingOutputAxesPolicy`] for packaging the batched program outputs.
    #[inline]
    pub fn batched(
        &self,
        axis_size: usize,
        axis_sharding: ShardingDimension,
        input_batch_axes: &[BatchAxis],
        output_axes_policy: ProgramBatchingOutputAxesPolicy,
    ) -> Result<BoundaryPreservingBatchedProgram<V, O>, BatchingError> {
        self.batched_with_axis_name(axis_size, None, axis_sharding, input_batch_axes, output_axes_policy)
    }

    /// Structurally batches this region while making `axis_name` visible to named-axis operations in its body. Public
    /// program batching has no named-axis parameter and delegates with `None`. Recursive batching of an attached region
    /// uses this path to preserve the active [`BatchingContext`]'s axis environment.
    ///
    /// # Parameters
    ///
    ///   - `axis_size`: Dimension of the new batch axis.
    ///   - `axis_name`: Optional name exposed to named-axis operations while replaying this region.
    ///   - `axis_sharding`: Sharding placement assigned to every newly materialized batch axis.
    ///   - `input_batch_axes`: [`BatchAxis`] for each input (i.e., argument) of this [`Program`].
    ///   - `output_axes_policy`: [`ProgramBatchingOutputAxesPolicy`] for packaging the batched program outputs.
    pub(crate) fn batched_with_axis_name(
        &self,
        axis_size: usize,
        axis_name: Option<String>,
        axis_sharding: ShardingDimension,
        input_batch_axes: &[BatchAxis],
        output_axes_policy: ProgramBatchingOutputAxesPolicy,
    ) -> Result<BoundaryPreservingBatchedProgram<V, O>, BatchingError> {
        // Cache the semantic packed-axis positions, not the caller's rank-relative spelling. For one rank-two source
        // input, for example, `-1` and `2` both select the trailing axis after insertion and must share one transform
        // specialization. Validation must also precede `zip`. Otherwise, a short axis list would silently omit source
        // inputs from both the key and the derived boundary, while an overlong list would leave unchecked key material.
        check_count!("input", input_batch_axes, self.input_ids().len(), ProgramError);
        let input_batch_axes = self
            .input_types()
            .iter()
            .zip(input_batch_axes)
            .map(|(unbatched_type, batch_axis)| {
                batch_axis.axis().map_or(Ok(*batch_axis), |axis| {
                    let batched_rank = unbatched_type.rank() + 1;
                    axis.normalize(batched_rank).map(BatchAxis::from_position).map_err(|_| {
                        BatchingError::BatchAxisOutOfBounds { r#type: Box::new(unbatched_type.clone()), axis }
                    })
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Output policy is likewise part of the transform's semantic identity. Normalize `AlignEachTo` independently
        // against each packed output rank so equivalent positive and negative spellings hit the same entry, and reject
        // malformed arity or out-of-bounds axes before lookup. `AlignAllTo` is subtler: one signed axis is interpreted
        // separately for every output, so heterogeneous ranks may resolve it to different positions. It can be replaced
        // by one canonical non-negative axis only when every output resolves to that same position. Otherwise, retaining
        // the raw signed axis preserves the per-output semantics. Using the raw policy everywhere would remain correct,
        // but equivalent requests would occupy duplicate cache entries and repeat structural derivation.
        let output_axes_policy = match output_axes_policy {
            ProgramBatchingOutputAxesPolicy::Natural => ProgramBatchingOutputAxesPolicy::Natural,
            ProgramBatchingOutputAxesPolicy::AlignAllTo(axis) => {
                let positions = self
                    .output_types()
                    .iter()
                    .map(|unbatched_type| {
                        axis.normalize(unbatched_type.rank() + 1).map_err(|_| BatchingError::BatchAxisOutOfBounds {
                            r#type: Box::new(unbatched_type.clone()),
                            axis,
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                // One signed axis can only be canonicalized when it resolves to the same position for every output, so
                // outputs of differing ranks keep the caller's raw axis. Requests spelling one position two ways (e.g.
                // `-1` and `1`) then occupy two cache entries, which costs extra misses but never reuses an artifact
                // for a different alignment.
                if positions.windows(2).all(|positions| positions[0] == positions[1]) {
                    positions
                        .first()
                        .copied()
                        .map(Axis::from)
                        .map(ProgramBatchingOutputAxesPolicy::AlignAllTo)
                        .unwrap_or(ProgramBatchingOutputAxesPolicy::AlignAllTo(axis))
                } else {
                    ProgramBatchingOutputAxesPolicy::AlignAllTo(axis)
                }
            }
            ProgramBatchingOutputAxesPolicy::AlignEachTo(axes) => {
                check_count!("output", axes, self.output_types().len(), ProgramError);
                let axes = self
                    .output_types()
                    .iter()
                    .zip(axes)
                    .map(|(unbatched_type, batch_axis)| -> Result<BatchAxis, BatchingError> {
                        if let Some(axis) = batch_axis.axis() {
                            let position = axis.normalize(unbatched_type.rank() + 1).map_err(|_| {
                                BatchingError::BatchAxisOutOfBounds { r#type: Box::new(unbatched_type.clone()), axis }
                            })?;
                            Ok(BatchAxis::from_position(position))
                        } else {
                            Ok(batch_axis)
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                ProgramBatchingOutputAxesPolicy::AlignEachTo(axes)
            }
        };
        let arguments = ArrayBatchingTransformArguments {
            axis_size,
            axis_name,
            axis_sharding,
            input_batch_axes,
            output_axes_policy,
        };
        let artifact =
            (*self).transform::<ArrayBatchingTransform, _, BatchingError>(arguments, |region, arguments| {
                let ArrayBatchingTransformArguments {
                    axis_size,
                    axis_name,
                    axis_sharding,
                    input_batch_axes,
                    output_axes_policy,
                } = arguments;
                let input_count = region.input_ids().len();

                let parent_context = TracingContext::<V, O>::new();
                let builder = parent_context.builder().clone();

                // Keep every tracer and context that holds a clone of `builder` inside the following scope so that
                // recovering the builder later on (below) is a real ownership check. In particular, `context` (which
                // owns a clone of the builder through its parent trace) must be created *inside* this scope so it is
                // dropped before the builder is recovered below; leaving it in the enclosing scope leaks a builder
                // clone past the recovery.
                let (output_atom_ids, output_axes) = {
                    let batching_context = BatchingContext::new(parent_context, *axis_size)
                        .with_axis_name(axis_name.clone())
                        .with_axis_sharding(axis_sharding.clone());
                    let inputs = region
                        .input_types()
                        .iter()
                        .zip(input_batch_axes.iter())
                        .map(|(unbatched_type, batch_axis)| {
                            let batched_type = match batch_axis.axis() {
                                Some(axis) => {
                                    // The mapped axis is already normalized against the packed input rank (i.e., with
                                    // the inserted batch dimension counted), so reading its position back cannot fail.
                                    let position = usize::try_from(axis.value()).unwrap();
                                    unbatched_type.batched(
                                        position,
                                        Dimension::Static(*axis_size),
                                        axis_sharding.clone(),
                                    )?
                                }
                                None => unbatched_type.clone(),
                            };
                            let input = builder.borrow_mut().add_input(batched_type.clone());
                            let value = batching_context.parent().tracer(input, Some(batched_type.clone()));
                            Ok(ArrayBatch::new(value, *batch_axis)?)
                        })
                        .collect::<Result<Vec<_>, ProgramError>>()?;

                    // Replay this program by binding each instruction's `BatchableOperation` rule against the batching
                    // context, threading the batch-carrying inputs through. Constants lift in the parent trace and
                    // replicate across the batch. This only requires this program's own operation family to be
                    // batchable, so staged higher-order batching rules can batch a captured sub-program without
                    // concretizing any batch-item values.
                    let region_mappings = RegionReplayMappings::new();
                    let outputs = region.interpret_with(
                        inputs,
                        |_, constant| Ok(ArrayBatch::replicated(batching_context.parent().lift(constant.clone())?)),
                        |instruction, instruction_inputs| -> Result<_, BatchingError> {
                            // Run the batching rule inside the source instruction's recorded origin so that every
                            // staged instruction records where it came from.
                            let regions = ReplayRegionDriver::new(region, instruction.regions(), &region_mappings)?;
                            let regions = AttachedRegions::from(regions);
                            let outputs = batching_context.invoke_with_provenance_origin(
                                instruction.provenance().clone(),
                                || {
                                    instruction.operation().batch(
                                        &batching_context,
                                        &RecursiveBatchingDriver::new(&regions),
                                        instruction_inputs,
                                    )
                                },
                            )?;

                            Ok(outputs)
                        },
                    )?;

                    // Resolve `output_axes_policy` into one optional alignment declaration per output. The outer `None`
                    // keeps the natural axis, while `Some(mapped)` forces the output to carry its batch axis at that
                    // signed position. A replicated `AlignEachTo` entry is a lower bound rather than an equality
                    // constraint, mirroring JAX's `instantiate` behavior.
                    let output_target_batch_axes = match &output_axes_policy {
                        ProgramBatchingOutputAxesPolicy::Natural => vec![None; outputs.len()],
                        ProgramBatchingOutputAxesPolicy::AlignAllTo(axis) => {
                            vec![Some(BatchAxis::new(*axis)); outputs.len()]
                        }
                        ProgramBatchingOutputAxesPolicy::AlignEachTo(axes) => {
                            // This check is defense in depth and cannot fire today, because `batched_with_axis_name`
                            // already rejects a policy whose axis count differs from this region's output count, and
                            // replaying the region produces exactly one output per region output.
                            check_count!("output", axes, outputs.len(), ProgramError);
                            axes.iter().map(|target| (!target.is_replicated()).then_some(*target)).collect::<Vec<_>>()
                        }
                    };
                    let mut output_atom_ids = Vec::with_capacity(outputs.len());
                    let mut output_axes = Vec::with_capacity(outputs.len());
                    for (output, output_target_batch_axis) in outputs.into_iter().zip(output_target_batch_axes) {
                        // The batched outputs must belong to this batched trace. A foreign tracer's atom ID would
                        // silently alias whichever atom shares its index in this builder, and so we perform a check
                        // here to avoid that.
                        check_builders!(&builder, output.value().builder())?;

                        // Untargeted outputs keep the natural axis produced by their batching rules.
                        let Some(target_batch_axis) = output_target_batch_axis else {
                            output_atom_ids.push(output.value().atom_id()?);
                            output_axes.push(output.batch_axis());
                            continue;
                        };

                        // Move naturally mapped outputs or broadcast naturally replicated outputs while they are still
                        // live tracers, then report the normalized position stored by `ArrayBatch`.
                        let output = output.align_axis(
                            target_batch_axis,
                            *axis_size,
                            batching_context.axis_sharding().clone(),
                        )?;
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
                let (program, output_axes) = BoundaryPreservingBatchedProgram::new(program, output_axes)?.into_parts();
                Ok(TransformArtifact::new(vec![Arc::new(program)], output_axes))
            })?;
        let (mut programs, output_axes) = artifact.into_parts();
        assert_eq!(programs.len(), 1);
        let program = programs.pop().unwrap().as_ref().clone();
        Ok(BoundaryPreservingBatchedProgram::new(program, output_axes)?)
    }
}

impl<
    V: Value<Type = ArrayType>,
    O: Operation<Type = ArrayType>
        + BatchableOperation<TracingContext<V, O>, ArrayBatchingPolicy>
        + From<TransposeOperation>
        + From<BroadcastOperation>,
> Program<V, O, Vec<V>, Vec<V>>
where
    TracingContext<V, O>: StagingContext<Type = ArrayType, Constant = V, Operation = O>,
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
    ) -> Result<BoundaryPreservingBatchedProgram<V, O>, BatchingError> {
        self.entry_region_ref().batched(axis_size, axis_sharding, input_batch_axes, output_axes_policy)
    }
}

impl<
    V: Value<Type = ArrayIrType>
        + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>
        + ValueProjection<DimensionType, Projected = DimensionValue>,
    O: Operation<Type = ArrayIrType>
        + BatchableOperation<TracingContext<V, O>, ArrayIrBatchingPolicy>
        + From<ConstantOperation<DimensionValue>>
        + From<DimensionSizeOperation>
        + From<DynamicBroadcastOperation>
        + OperationProjection<
            ArrayType,
            Projected: OperationProvider<
                ArrayType,
                GuardOperation<ArrayType>,
                Operation = <O as OperationProjection<ArrayType>>::Projected,
            > + From<TransposeOperation>,
        >,
> Program<V, O, Vec<V>, Vec<V>>
where
    TracingContext<V, O>: StagingContext<Type = ArrayIrType, Constant = V, Operation = O>,
{
    /// Structurally batches this array IR [`Program`] over the provided input axes, threading the batch extent as a
    /// leading runtime dimension input instead of specializing on a static size.
    ///
    /// This is the array IR counterpart of the homogeneous [`ArrayType`] `Program::batched` entry point: the mapped
    /// extent is a first-class [`DimensionType`] value supplied at invocation time, so the transformed program carries
    /// [`ArrayIrBatchingPolicy`]'s leading mapped-extent input and forwarded output, while its
    /// [`BatchedProgram::output_axes`] continue to describe only the source program's public outputs. The new batch
    /// axis is unnamed, mirroring the homogeneous entry, whose public batching contract has no named-axis parameter.
    ///
    /// Reference inputs batch like any other input: a reference input declared mapped at axis `k` becomes `ref<T with
    /// the batch extent inserted at k>`, and every operation on it reads and writes per-item values at that axis, while
    /// a reference input declared replicated stays one shared holder for the whole batch and rejects any write of a
    /// batched value (pass the reference as a batched input instead). Local allocations always produce batched
    /// references, so a program that allocates, mutates, reads, and freezes its own state batches exactly like its
    /// discharged counterpart. A reference-typed output carries the batch axis of the root it forwards and cannot be
    /// realigned by `output_axes_policy`. Captured references are unbatched and arrive replicated. Discharging
    /// references through [`Program::discharge_references`] first remains valid and yields the same batched semantics
    /// over ordinary arrays.
    ///
    /// # Parameters
    ///
    ///   - `axis_extent_type`: Type of the leading runtime dimension input that defines the mapped batch extent.
    ///   - `axis_sharding`: Sharding placement assigned to every newly materialized batch axis.
    ///   - `input_batch_axes`: Batch-axis declaration for each source program input.
    ///   - `output_axes_policy`: Policy controlling the transformed public output axes.
    pub fn batched_with_threaded_extent(
        &self,
        axis_type: DimensionType,
        axis_sharding: ShardingDimension,
        input_batch_axes: &[BatchAxis],
        output_axes_policy: ProgramBatchingOutputAxesPolicy,
    ) -> Result<ThreadedExtentBatchedProgram<V, O>, BatchingError> {
        let context = TracingContext::<V, O>::new();
        let axis_extent = context.input(axis_type.into());
        let context =
            BatchingContext::<_, ArrayIrBatchingPolicy>::new(context, axis_extent).with_axis_sharding(axis_sharding);
        <ArrayIrBatchingPolicy as RecursiveBatchingPolicy<TracingContext<V, O>>>::batch_program(
            &context,
            self.entry_region_ref(),
            input_batch_axes,
            output_axes_policy,
        )
    }
}

impl<C: Context<Type = ArrayType>, P: ArrayExtentBatchingPolicy<C>> BatchingContext<C, ArrayBatchingPolicy<P>> {
    /// Binds `operation` unchanged into the [parent](Self::parent) context on the packed values of `inputs`, and
    /// returns one output per input, each carrying the batch axis of the input at the same position. This is the
    /// batching rule of an operation that leaves the mapped axis alone (i.e., one whose application to the packed
    /// values already computes every batch item's result independently, without any per-item axis parameter that the
    /// inserted batch axis would shift). Examples are manual-variation transitions, value tags, reductions over a
    /// manual mesh axis, and named-axis collectives that target an enclosing `batch` level, which this level must pass
    /// through untouched so that each of its batch items participates in the outer collective independently. The parent
    /// may itself be another [`BatchingContext`], whose own rule dispatch then handles the operation, or an ordinary
    /// context that interprets or stages it.
    ///
    /// The operation is bound rather than interpreted, so a staging parent records it even when its interpretation
    /// would be an identity (e.g., for value tags) or an error (e.g., for collectives).
    ///
    /// # Errors
    ///
    /// Returns a [`BatchingError`] if the parent fails to bind `operation`, if it does not produce exactly one output
    /// per input, or if an output cannot carry the batch axis of its input.
    pub fn forward_to_parent(
        &self,
        operation: C::Operation,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        let values = inputs.iter().map(|input| input.value().clone()).collect::<Vec<_>>();
        let outputs = self.parent().bind(operation, Vec::new(), values.as_slice())?;
        check_count!("output", outputs, inputs.len(), ProgramError);
        outputs
            .into_iter()
            .zip(inputs)
            .map(|(output, input)| ArrayBatch::new(output, input.batch_axis()))
            .collect()
    }
}

/// [`Region`] [`Transform`] marker for retained homogeneous array batched [`Program`]s.
struct ArrayBatchingTransform;

impl<V: Value<Type = ArrayType>, O: Operation<Type = ArrayType>> Transform<Region<V, O>> for ArrayBatchingTransform {
    type Arguments = ArrayBatchingTransformArguments;
    type Artifact = TransformArtifact<V, O, Vec<BatchAxis>>;

    const DEFAULT_CACHE_CAPACITY: usize = 8;
}

/// Argument key for one retained [`ArrayBatchingTransform`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ArrayBatchingTransformArguments {
    /// Static extent inserted for the mapped axis.
    axis_size: usize,

    /// Optional name visible to named-axis operations while replaying the region.
    axis_name: Option<String>,

    /// Sharding assigned to every inserted mapped axis.
    axis_sharding: ShardingDimension,

    /// Normalized mapped axis or replication state of every source input.
    input_batch_axes: Vec<BatchAxis>,

    /// Normalized output-axis packaging requirements.
    output_axes_policy: ProgramBatchingOutputAxesPolicy,
}

impl Sharding {
    /// Returns a per-item output [`Sharding`] with `axis_sharding` inserted as its leading physical dimension. The
    /// mesh and reduction state are preserved, and mesh axes named by the inserted dimension are removed from the
    /// varying-manual set: those axes are now represented by the explicit batch placement.
    ///
    /// For example, a per-item output replicated along its array dimensions but varying over manual mesh axis `x`
    /// becomes an output with a leading dimension sharded over `x`, without `x` remaining in its varying-manual set.
    /// Other varying-manual axes are preserved. The resulting sharding is revalidated, so unknown or already-used
    /// placement axes are rejected.
    ///
    /// This function lifts a requested output placement. Use [`Self::batched`] for body-side value shardings instead.
    /// That function records the manual-axis variation introduced by the batch placement rather than removing it.
    ///
    /// # Parameters
    ///
    ///   - `axis_sharding`: Placement of the new leading batch dimension. A replicated or unconstrained placement
    ///     leaves the varying-manual set unchanged.
    pub fn with_leading_batch_axis(&self, axis_sharding: ShardingDimension) -> Result<Self, BatchingError> {
        let mut dimensions = self.dimensions().to_vec();
        dimensions.insert(0, axis_sharding.clone());
        let mut varying_manual_axes = self.varying_manual_axes().clone();
        if let ShardingDimension::Sharded(axis_names) = axis_sharding {
            for axis_name in axis_names {
                varying_manual_axes.remove(&axis_name);
            }
        }
        Sharding::new(self.mesh().clone(), dimensions)
            .and_then(|sharding| sharding.with_unreduced_axes(self.unreduced_axes().clone()))
            .and_then(|sharding| sharding.with_reduced_axes(self.reduced_axes().clone()))
            .and_then(|sharding| sharding.with_varying_manual_axes(varying_manual_axes))
            .map_err(|error| BatchingError::MisalignedBatchAxes { message: error.to_string() })
    }

    /// Returns this [`Sharding`] with the mapped batch axis inserted at `position` on the placement `axis_sharding`.
    /// Besides gaining that placement, the result records that the value now varies along every
    /// [`MeshAxisType`](crate::MeshAxisType::Manual) axis the placement names: a batch axis placed on a manual axis
    /// spreads the batch items across that axis's shards, so the batched value differs from device to device along it.
    /// This is the body-side rule for values produced inside a batched computation.
    ///
    /// Batching rules construct body-side value shardings through this function, or through [`ArrayType::batched`],
    /// which uses it, rather than inserting the batch placement with [`Self::with_inserted_dimension`] directly, so
    /// that placement and variation cannot drift apart. To lift a requested per-item output placement instead, use
    /// [`Self::with_leading_batch_axis`], which removes the inserted axes from the varying-manual set.
    pub fn batched(&self, position: usize, axis_sharding: ShardingDimension) -> Result<Self, BatchingError> {
        let varying_manual_axes = axis_sharding.manual_axes(self.mesh());
        let mut batched = self
            .with_inserted_dimension(position, axis_sharding)
            .map_err(|error| BatchingError::MisalignedBatchAxes { message: error.to_string() })?;
        batched
            .extend_varying_manual_axes(varying_manual_axes)
            .map_err(|error| BatchingError::MisalignedBatchAxes { message: error.to_string() })?;
        Ok(batched)
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

    /// Returns the packed [`ArrayType`] obtained by inserting the mapped batch axis at `position` into this per-item
    /// [`ArrayType`]. Its shape gains `dimension` there and any sharding gains `axis_sharding` at the same position
    /// through [`Sharding::batched`], which also records the variation that placement introduces along manual mesh
    /// axes. This is the inverse of [`Self::unbatched`], and batching rules must construct every batched array type
    /// through it for the reason documented on [`Sharding::batched`].
    #[inline]
    pub fn batched(
        &self,
        position: usize,
        dimension: Dimension,
        axis_sharding: ShardingDimension,
    ) -> Result<Self, BatchingError> {
        let mut batched = self.with_inserted_dimension(position, dimension)?;
        if let Some(sharding) = self.sharding() {
            batched.sharding = Some(sharding.batched(position, axis_sharding)?);
        }
        Ok(batched)
    }

    /// Returns the unbatched per-item [`ArrayType`] obtained by removing `batch_axis` from this packed [`ArrayType`].
    /// A replicated axis leaves the type unchanged. Possibly-negative mapped axes are normalized against the packed
    /// rank. When the removed dimension carries sharding, only manual mesh axes remain visible as varying manual axes
    /// in the per-item type because all other placement belongs to the transform-owned batch dimension. This is the
    /// inverse of [`Self::batched`].
    ///
    /// # Errors
    ///
    /// Returns a [`BatchingError`] for an invalid batch axis or projected sharding.
    pub fn unbatched<A: Into<BatchAxis>>(&self, batch_axis: A) -> Result<Self, BatchingError> {
        let (_, axis) = self.normalize_batch_axis(batch_axis.into())?;
        let Some(axis) = axis else {
            return Ok(self.clone());
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
                projected_sharding.extend_varying_manual_axes(removed_dimension.manual_axes(sharding.mesh()))?;
                Ok(projected_sharding)
            })
            .transpose()
            .map_err(|error| BatchingError::MisalignedBatchAxes { message: error.to_string() })?;

        Ok(Self {
            data_type: self.data_type(),
            shape: Shape::new(dimensions),
            layout: None,
            sharding,
            memory: self.memory(),
        })
    }

    /// Returns this per-item [`ArrayType`] with a batch dimension inserted at `axis`, preserving the existing packed
    /// value's variation on the mapped manual mesh axes. Guards and casts use this derivation because they keep the
    /// value's physical batch axis and introduce no new variation. Every other constraint stays as declared, so
    /// lifting cannot hide an unrelated mismatch. Callers validate the packed value against the returned type.
    ///
    /// # Parameters
    ///
    ///   - `actual`: Type of the existing packed value, supplying its placement and mapped manual-axis variation.
    ///   - `axis`: Position at which to insert the mapped batch dimension.
    ///   - `extent`: Extent of the mapped batch dimension.
    ///   - `axis_sharding`: Sharding of the mapped batch dimension.
    ///
    /// # Errors
    ///
    /// Returns a [`BatchingError`] when the inserted axis or resulting sharding is invalid.
    fn batched_declaration(
        &self,
        actual: &Self,
        axis: usize,
        extent: &Dimension,
        axis_sharding: &ShardingDimension,
    ) -> Result<Self, BatchingError> {
        let mut batched = self.batched(axis, extent.clone(), axis_sharding.clone())?;
        if batched.sharding().is_none()
            && let Some(actual) = actual.sharding()
            && !axis_sharding.manual_axes(actual.mesh()).is_empty()
        {
            // Unspecified placement still needs a mesh to express batch-induced manual variation. Choose the supplied
            // placement, enforcing the batch axis's placement, without adopting unrelated variation or reduction state.
            let dimensions = actual
                .dimensions()
                .iter()
                .enumerate()
                .map(|(position, placement)| if position == axis { axis_sharding.clone() } else { placement.clone() })
                .collect();
            let sharding = Sharding::new(actual.mesh().clone(), dimensions).map_err(TypeError::from)?;
            batched = batched.with_sharding(sharding).map_err(TypeError::from)?;
        }

        if let (Some(sharding), Some(actual)) = (batched.sharding(), actual.sharding()) {
            let mut varying = sharding.varying_manual_axes().clone();
            for axis in axis_sharding.manual_axes(sharding.mesh()) {
                if actual.varying_manual_axes().contains(&axis) {
                    varying.insert(axis);
                } else {
                    varying.remove(&axis);
                }
            }
            let sharding = sharding.clone().with_varying_manual_axes(varying).map_err(TypeError::from)?;
            batched = batched.with_sharding(sharding).map_err(TypeError::from)?;
        }
        Ok(batched)
    }
}

impl BatchableType for ArrayType {
    type Policy = ArrayBatchingPolicy;
}

impl BatchableType for ArrayIrType {
    type Policy = ArrayIrBatchingPolicy;
}

/// Derives the common [`ShardingDimension`] of mapped axes from packed array types and normalized axis positions.
/// This representation-neutral helper lets homogeneous and array IR batching share the same placement join without
/// projecting or cloning array payloads.
fn batch_axis_sharding<T: std::borrow::Borrow<ArrayType>, I: IntoIterator<Item = (T, Option<usize>)>>(
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
            // Carry the common mesh and the joined mapped-axis placement together so this remains a single-pass
            // operation over potentially one-shot iterators.

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

/// Returns the paired gather that takes the diagonal of an array of type `type` along `kept_axis` and `removed_axis`
/// (refer to [`DiagonalBatchingPolicy::diagonal`]), as the type of its indices, the indices axis along which they
/// count, its dimension numbers, and its window sizes. The indices enumerate every input coordinate except those of
/// `removed_axis`, followed by a one-component index vector, and count along the coordinates of `kept_axis`. Pairing
/// every enumerated axis with the input axis that it enumerates (i.e., as gather batching dimensions) then reads, for
/// each input coordinate, the element whose `removed_axis` index equals its `kept_axis` index, with no window over a
/// possibly dynamic extent.
fn diagonal_gather_layout(
    r#type: &ArrayType,
    kept_axis: usize,
    removed_axis: usize,
) -> Result<(ArrayType, usize, GatherDimensionNumbers, Vec<usize>), BatchingError> {
    let rank = r#type.rank();
    if kept_axis >= rank || removed_axis >= rank || kept_axis == removed_axis {
        return Err(TypeError::invalid(format!(
            "cannot take the diagonal of a value of type `{type}` along axes {kept_axis} and {removed_axis}, \
             which must be two distinct axes of its rank {rank}",
        ))
        .into());
    }

    if r#type.dimension(kept_axis) != r#type.dimension(removed_axis) {
        return Err(TypeError::invalid(format!(
            "cannot take the diagonal of a value of type `{type}` along axes {kept_axis} and {removed_axis}, \
             whose extents differ",
        ))
        .into());
    }

    let paired_axes = (0..rank).filter(|axis| *axis != removed_axis).collect::<Vec<_>>();
    let mut indices_shape = paired_axes.iter().map(|axis| r#type.dimension(*axis)).collect::<Vec<_>>();
    indices_shape.push(Dimension::Static(1));
    let indices_type = ArrayType::new(DataType::I64, Shape::new(indices_shape));
    let iota_dimension = if removed_axis < kept_axis { kept_axis - 1 } else { kept_axis };
    let dimensions = GatherDimensionNumbers::new(Vec::new(), vec![removed_axis], vec![removed_axis])
        .with_batching_dimensions(paired_axes.iter().enumerate().map(|(index, axis)| (*axis, index)).collect());

    // Paired axes use a zero window when they may be empty, exactly as `DynamicGather::dynamic_gather_axis` does.
    let slice_sizes = (0..rank)
        .map(|axis| if axis == removed_axis { 1 } else { r#type.dimension(axis).bounds().lower().min(1) })
        .collect();
    Ok((indices_type, iota_dimension, dimensions, slice_sizes))
}

/// Returns the [`ArrayType`] required to place `position` on `axis_sharding`,
/// or `None` when no normalization is needed.
fn normalized_batch_axis_type(
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
    varying_manual_axes.extend(axis_sharding.manual_axes(sharding.mesh()));

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

/// Broadcasts the replicated array `value`, whose per-item type is `r#type`, so that it gains the mapped batch axis at
/// `position` with the transform's `axis_extent` and `axis_sharding`. The per-item shape survives unchanged, so each of
/// its axes is read through [`DimensionSize`], which folds statically known extents into literals, and the inserted
/// axis takes the transform's own extent.
///
/// # Parameters
///
///   - `value`: Replicated packed array value to broadcast.
///   - `r#type`: Per-item type of `value`, whose sharding (if any) gains the batch axis placement.
///   - `position`: Normalized position of the inserted mapped axis in the broadcast output.
///   - `axis_extent`: First-class extent of the transform's mapped axis.
///   - `axis_sharding`: Sharding placement of the transform's mapped axis.
fn broadcast_replicated_array<V: Clone + DimensionSize + DynamicBroadcast>(
    value: V,
    r#type: &ArrayType,
    position: usize,
    axis_extent: &V,
    axis_sharding: &ShardingDimension,
) -> Result<V, BatchingError> {
    // The output shape is the per-item shape with the mapped extent inserted at `position`. Every per-item axis is read
    // back from `value` (or folded to a constant when static) and keeps its relative order, shifting by one past the
    // inserted axis.
    let mut output_dimensions =
        (0..r#type.rank()).map(|axis| value.dimension_size(axis)).collect::<Result<Vec<_>, _>>()?;
    output_dimensions.insert(position, axis_extent.clone());

    let output_axes = (0..r#type.rank())
        .map(|input_axis| if input_axis < position { input_axis } else { input_axis + 1 })
        .collect::<Vec<_>>();

    // The sharding, when present, gains the transform's placement for the mapped axis at the same position.
    let output_sharding =
        r#type.sharding().map(|sharding| sharding.batched(position, axis_sharding.clone())).transpose()?;

    Ok(value.dynamic_broadcast_with_output_sharding(&output_dimensions, &output_axes, output_sharding)?)
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use indoc::indoc;
    use num_complex::Complex;
    use pretty_assertions::assert_eq;

    use crate::arrays::arrays::Array;
    use crate::arrays::dimensions::DimensionValue;
    use crate::arrays::ir::ArrayIrValue;
    use crate::arrays::operations::{ArrayIrOperation, ArrayOperation, DimensionOperation};
    use crate::arrays::references::ArrayReference;
    use crate::arrays::sharding::meshes::{LogicalMesh, MeshAxis, MeshAxisType};
    use crate::arrays::sharding::shardings::ShardingDimension;
    use crate::arrays::types::data::DataType;
    use crate::arrays::types::dimensions::{Dimension, DimensionBounds, DimensionError, DimensionVariable, Shape};
    use crate::axes::{NamedAxes, NamedAxis};
    use crate::batching::{
        Batch, BatchAxisSpecification, BatchingPolicy, BatchingTracer, DiagonalBatchingPolicy,
        InterpretableBatchableOperation, RecursiveBatchingPolicy, batch,
    };
    use crate::contexts::{EagerContext, StagingContext};
    use crate::differentiation::{Differentiate, ForwardModeDifferentiate, LinearizationTracer};
    use crate::operations::{
        AddOperation, CompareOperation, ConditionOperation, DimensionAddOperation, DimensionDivOperation,
        DimensionFromScalar, DimensionToScalar, DimensionToScalarOperation, DynamicReshapeOperation, NegOperation,
        OneLike, ParallelReduceOperation, Reduce, ReductionKind, ReferenceAddUpdateOperation, ReferenceFreezeOperation,
        ReferenceNewOperation, ReferenceReadOperation, ReshardOperation,
    };
    use crate::parameters::Placeholder;
    use crate::programs::{EmptyRegionDriver, ProgramBuilder, Provenance, ProvenanceScope, ReferenceType};
    use crate::specialization::SpecializationCacheStatistics;
    use crate::tests::TestArrayContext;
    use crate::tracing::{DomainTracingContext, Trace, TracingContext};

    use super::*;

    /// Eager array IR context shared by array, dimension, and reference batching tests.
    type ArrayIrEagerContext = EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

    /// Tracing array IR context shared by batching tests.
    type ArrayIrTraceContext = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

    #[test]
    fn test_array_batch_new() {
        let matrix = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let matrix_type = matrix.r#type().into_owned();

        // `new` builds a batched value when the mapped axis is in bounds, and the accessors report the packed value,
        // its packed type, the batch size read off the mapped axis, and the per-item type with that axis removed.
        let batched = ArrayBatch::new(matrix.clone(), Some(0)).unwrap();
        assert_eq!(batched.batch_axis(), BatchAxis::new(0));
        assert_eq!(batched.batch_axis_position(), Some(0));
        assert_eq!(batched.value(), &matrix);
        assert_eq!(*batched.r#type(), matrix_type);
        assert_eq!(batched.batch_size(), Ok(Some(2)));
        assert_eq!(batched.unbatched_type(), ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)])));
        assert_eq!(batched.to_string(), "batch[f64[2, 3], axis=0]([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])");
        assert_eq!(batched.into_value(), matrix);

        // A different mapped axis reads the batch size and per-item type from that axis instead.
        let batched_axis_one = ArrayBatch::new(matrix.clone(), Some(1)).unwrap();
        assert_eq!(batched_axis_one.batch_size(), Ok(Some(3)));
        assert_eq!(
            batched_axis_one.unbatched_type(),
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)])),
        );

        // Negative axes follow Python/JAX indexing and are normalized once at construction.
        // `-1` denotes the final axis and the stored metadata is the canonical non-negative position.
        let batched_axis_negative_one = ArrayBatch::new(matrix.clone(), BatchAxis::new(-1)).unwrap();
        assert_eq!(batched_axis_negative_one.batch_axis(), BatchAxis::new(1));
        assert_eq!(batched_axis_negative_one.batch_axis_position(), Some(1));
        assert_eq!(batched_axis_negative_one.batch_size(), Ok(Some(3)));

        // `new` rejects an out-of-bounds mapped axis in either direction.
        assert_eq!(
            ArrayBatch::new(matrix.clone(), Some(2)),
            Err(BatchingError::BatchAxisOutOfBounds { r#type: Box::new(matrix_type.clone()), axis: Axis::from(2) }),
        );
        assert_eq!(
            ArrayBatch::new(matrix, BatchAxis::new(-3)),
            Err(BatchingError::BatchAxisOutOfBounds { r#type: Box::new(matrix_type), axis: Axis::from(-3) }),
        );

        // `replicated` shares the value unchanged across the batch: no mapped axis, no batch size, and the per-item
        // type is the whole packed type.
        let vector = Array::vector(vec![1.0, 2.0, 3.0]).unwrap();
        let vector_type = vector.r#type().into_owned();
        let replicated = ArrayBatch::replicated(vector.clone());
        assert_eq!(replicated.batch_axis(), BatchAxis::replicated());
        assert_eq!(replicated.batch_axis_position(), None);
        assert_eq!(*replicated.r#type(), vector_type);
        assert_eq!(replicated.batch_size(), Ok(None));
        assert_eq!(replicated.unbatched_type(), vector_type);
        assert_eq!(replicated.to_string(), "batch[f64[3], replicated]([1.0, 2.0, 3.0])");

        // A batch is itself a `Value` whose domains are the eager context over batches, so batches can flow through
        // generic value code (e.g., as batching-context values) without a backend-owned context.
        assert!(replicated.domain().is_eager());
        assert!(replicated.domain().is_eager());
        assert_eq!(replicated.into_value(), vector);
    }

    #[test]
    fn test_array_batch_common_batch_size() {
        let matrix = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let vector = Array::vector(vec![7.0, 8.0]).unwrap();

        // All-replicated inputs pin no batch size.
        let replicated = ArrayBatch::replicated(matrix.clone());
        assert_eq!(ArrayBatch::common_batch_size(&[replicated.clone()]), Ok(None));

        // A single batched input pins its own batch size, and a replicated input alongside it is ignored.
        let batched_axis_zero = ArrayBatch::new(matrix.clone(), Some(0)).unwrap();
        assert_eq!(ArrayBatch::common_batch_size(std::slice::from_ref(&batched_axis_zero)), Ok(Some(2)));
        assert_eq!(ArrayBatch::common_batch_size(&[replicated, batched_axis_zero.clone()]), Ok(Some(2)));

        // Two batched inputs that agree on their batch size share it, even across different mapped axes.
        let batched_vector = ArrayBatch::new(vector, Some(0)).unwrap();
        assert_eq!(ArrayBatch::common_batch_size(&[batched_axis_zero.clone(), batched_vector]), Ok(Some(2)));

        // Two batched inputs that disagree on their batch size are rejected.
        let batched_axis_one = ArrayBatch::new(matrix, Some(1)).unwrap();
        assert_eq!(
            ArrayBatch::common_batch_size(&[batched_axis_zero, batched_axis_one]),
            Err(BatchingError::MismatchedBatchSizes { expected: 2, actual: 3 }),
        );
    }

    #[test]
    fn test_array_batch_broadcast() {
        // Broadcasting a replicated vector to gain a leading batch axis of size 2 replicates it across the batch.
        let replicated = ArrayBatch::replicated(Array::vector(vec![1.0, 2.0, 3.0]).unwrap());
        let broadcasted = replicated.broadcast(0, 2, ShardingDimension::Replicated).unwrap();
        assert_eq!(broadcasted.batch_axis(), BatchAxis::new(0));
        assert_eq!(
            *broadcasted.r#type(),
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)])),
        );
        assert_eq!(broadcasted.value(), &Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]).unwrap());
        assert_eq!(broadcasted.unbatched_type(), ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)])));

        // Broadcasting to a trailing batch axis keeps the per-item dimensions before it in place.
        // The input's own axis stays at position 0 and the size-2 batch axis is inserted at position 1.
        let broadcasted = replicated.broadcast(1, 2, ShardingDimension::Replicated).unwrap();
        assert_eq!(broadcasted.batch_axis(), BatchAxis::new(1));
        assert_eq!(
            *broadcasted.r#type(),
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3), Dimension::Static(2)])),
        );
        assert_eq!(broadcasted.value(), &Array::matrix(3, 2, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]).unwrap());

        // Broadcasting rejects an already-batched value.
        let batched = ArrayBatch::new(Array::vector(vec![1.0, 2.0]).unwrap(), Some(0)).unwrap();
        assert!(matches!(
            batched.broadcast(0, 2, ShardingDimension::Replicated),
            Err(BatchingError::MisalignedBatchAxes { .. }),
        ));
    }

    #[test]
    fn test_array_batch_broadcast_marks_manual_batch_axis_placement_varying() {
        // Broadcasting a replicated value onto a batch axis placed on a manual mesh axis distributes the batch items
        // across that axis's shards, so the result varies along it. A non-manual placement records no variation.
        for axis_type in [MeshAxisType::Explicit, MeshAxisType::Manual] {
            let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, axis_type).unwrap()]).unwrap();
            let replicated_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]))
                .with_sharding(Sharding::replicated(mesh.clone(), 1))
                .unwrap();
            let replicated =
                ArrayBatch::replicated(Array::from_elements::<f64>(replicated_type, &[1.0, 2.0, 3.0]).unwrap());
            let broadcasted = replicated.broadcast(0, 2, ShardingDimension::sharded(["x"])).unwrap();
            let expected_sharding =
                Sharding::new(mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                    .unwrap()
                    .with_varying_manual_axes((axis_type == MeshAxisType::Manual).then_some("x"))
                    .unwrap();
            assert_eq!(broadcasted.batch_axis(), BatchAxis::new(0));
            assert_eq!(broadcasted.value().r#type().sharding(), Some(&expected_sharding));
        }
    }

    #[test]
    fn test_array_batch_move_axis() {
        let matrix = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let batched = ArrayBatch::new(matrix, Some(0)).unwrap();

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
        assert_eq!(moved.value(), &Array::matrix(3, 2, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap());
        assert_eq!(moved.unbatched_type(), ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)])));
        assert_eq!(moved.batch_size(), Ok(Some(2)));

        // A replicated value has no mapped axis, so moving to any axis is a no-op.
        let replicated = ArrayBatch::replicated(Array::vector(vec![1.0, 2.0, 3.0]).unwrap());
        assert_eq!(replicated.move_axis(1).unwrap(), replicated);
    }

    #[test]
    fn test_array_batch_match_axis() {
        // `match_axis` on a batched value moves its mapped axis to the target (like `move_axis`). [2, 3] mapped at 0
        // becomes [3, 2] mapped at 1, and the `axis_size` argument is unused for an already-batched value.
        let batched =
            ArrayBatch::new(Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(), Some(0)).unwrap();
        let matched = batched.match_axis(1, 2, ShardingDimension::Replicated).unwrap();
        assert_eq!(matched.batch_axis(), BatchAxis::new(1));
        assert_eq!(
            *matched.r#type(),
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3), Dimension::Static(2)])),
        );
        assert_eq!(matched.value(), &Array::matrix(3, 2, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap());

        // `match_axis` on a replicated value broadcasts it to gain a batch axis there (like `broadcast`).
        let replicated = ArrayBatch::replicated(Array::vector(vec![1.0, 2.0, 3.0]).unwrap());
        let matched = replicated.match_axis(0, 2, ShardingDimension::Replicated).unwrap();
        assert_eq!(matched.batch_axis(), BatchAxis::new(0));
        assert_eq!(matched.value(), &Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]).unwrap());
    }

    #[test]
    fn test_array_batch_align_axis() {
        let batched =
            ArrayBatch::new(Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(), Some(0)).unwrap();

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
        assert_eq!(aligned.value(), &Array::matrix(3, 2, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap());

        // Aligning a replicated value to a replicated declaration returns it unchanged.
        let replicated = ArrayBatch::replicated(Array::vector(vec![1.0, 2.0, 3.0]).unwrap());
        assert_eq!(
            replicated.align_axis(BatchAxis::replicated(), 2, ShardingDimension::Replicated).unwrap(),
            replicated,
        );

        // A mapped output declaration instantiates a naturally replicated value by broadcasting it across the batch.
        let aligned = replicated.align_axis(BatchAxis::new(-1), 2, ShardingDimension::Replicated).unwrap();
        assert_eq!(aligned.batch_axis(), BatchAxis::new(1));
        assert_eq!(aligned.value(), &Array::matrix(3, 2, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]).unwrap());

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
        let batched = ArrayBatch::new(sharded_type.clone(), Some(0)).unwrap();
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
        let other = ArrayBatch::new(replicated_type, Some(0)).unwrap();
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
        let differently_sharded = ArrayBatch::new(differently_sharded_type, BatchAxis::new(0)).unwrap();
        let error = ArrayBatch::sharding_for_inputs(&[batched, differently_sharded]).unwrap_err();
        assert!(matches!(
            error.downcast_custom::<BatchingError>(),
            Some(BatchingError::MisalignedBatchAxes { message })
                if message == "mismatched batch axis sharding: {'x'} vs {'y'}",
        ));
    }

    #[test]
    fn test_array_ir_batch_new() -> Result<(), BatchingError> {
        // `new` derives the member kind from the value's type. An array member normalizes signed axes against the
        // packed rank and removes the mapped axis from its per-item type.
        let matrix = ArrayIrValue::Array(Array::matrix(2, 3, vec![0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap());
        let batch = ArrayIrBatch::new(matrix.clone(), BatchAxis::new(-2))?;
        assert_eq!(batch.batch_axis(), BatchAxis::new(0));
        assert_eq!(batch.batch_axis_position(), Some(0));
        assert_eq!(batch.value(), &matrix);
        assert_eq!(batch.unbatched_type(), ArrayIrType::Array(ArrayType::new_static(DataType::F32, [3])));
        assert_eq!(batch.into_value(), matrix.clone());
        assert_eq!(
            ArrayIrBatch::new(matrix, BatchAxis::new(2)),
            Err(BatchingError::BatchAxisOutOfBounds {
                r#type: Box::new(ArrayType::new_static(DataType::F32, [2, 3])),
                axis: Axis::from(2),
            }),
        );

        // A mapped sharded reference derives its per-item type by removing the batch dimension from the referent
        // together with that dimension's placement, exactly as a mapped array does, and normalizes signed axes against
        // the referent's rank.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let referent = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]))
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                    .unwrap(),
            )
            .unwrap();
        let trace = ArrayIrTraceContext::new();
        let reference = trace.input(ReferenceType::new(referent.clone()).into());
        let batch = ArrayIrBatch::new(reference.clone(), BatchAxis::new(-2))?;
        assert_eq!(batch.batch_axis(), BatchAxis::new(0));
        assert_eq!(batch.batch_axis_position(), Some(0));
        let per_item_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)]))
            .with_sharding(Sharding::new(mesh, vec![ShardingDimension::replicated()]).unwrap())
            .unwrap();
        assert_eq!(batch.unbatched_type(), ArrayIrType::Reference(ReferenceType::new(per_item_type)));
        assert_eq!(
            ArrayIrBatch::new(reference, BatchAxis::new(2)),
            Err(BatchingError::BatchAxisOutOfBounds { r#type: Box::new(referent), axis: Axis::from(2) }),
        );

        // A first-class dimension has no axes to map, so it only admits the replicated batch axis.
        // Eager dimension values are typed exactly, so diagnostics report the value's own exact type.
        let dimension_value =
            DimensionValue::new(DimensionType::new("n", DimensionBounds::new(1, Some(8)).unwrap()), 4).unwrap();
        let dimension_type = dimension_value.r#type().into_owned();
        let dimension = ArrayIrValue::<Array>::Dimension(dimension_value);
        let batch = ArrayIrBatch::new(dimension.clone(), BatchAxis::replicated())?;
        assert_eq!(batch.batch_axis_position(), None);
        assert_eq!(batch.unbatched_type(), ArrayIrType::Dimension(dimension_type.clone()));
        assert_eq!(
            ArrayIrBatch::new(dimension, BatchAxis::new(0)),
            Err(BatchingError::MappedDimension { r#type: Box::new(dimension_type), axis: BatchAxis::new(0) }),
        );
        Ok(())
    }

    #[test]
    fn test_array_ir_batch_replicated() {
        // `replicated` derives the member kind from the value's type, so every kind reports itself exactly as the
        // checked constructor would: a replicated axis and the packed type as the per-item type.
        let array = ArrayIrBatch::replicated(ArrayIrValue::<Array>::Array(Array::vector(vec![1.0_f32, 2.0]).unwrap()));
        assert_eq!(array.batch_axis(), BatchAxis::replicated());
        assert_eq!(array.batch_axis_position(), None);
        assert_eq!(array.unbatched_type(), ArrayIrType::Array(ArrayType::new_static(DataType::F32, [2])));

        let reference =
            ArrayIrBatch::replicated(ArrayIrValue::Reference(ArrayReference::new(Array::scalar(1.0_f32).unwrap())));
        assert_eq!(reference.batch_axis_position(), None);
        assert_eq!(
            reference.unbatched_type(),
            ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32))),
        );

        let dimension =
            ArrayIrBatch::replicated(ArrayIrValue::<Array>::Dimension(DimensionValue::constant(3).unwrap()));
        assert_eq!(dimension.batch_axis_position(), None);
        assert_eq!(dimension.unbatched_type(), dimension.value().r#type().into_owned());
    }

    #[test]
    fn test_array_ir_batch_validate_replicated_dimension() {
        // A replicated first-class dimension passes; a non-dimension member fails the dimension projection.
        let dimension_value =
            DimensionValue::new(DimensionType::new("n", DimensionBounds::new(1, Some(8)).unwrap()), 4).unwrap();
        let dimension = ArrayIrValue::<Array>::Dimension(dimension_value);
        assert_eq!(ArrayIrBatch::replicated(dimension).validate_replicated_dimension(), Ok(()));
        let array = ArrayIrBatch::replicated(ArrayIrValue::<Array>::Array(Array::vector(vec![1.0_f32, 2.0]).unwrap()));
        assert_eq!(
            array.validate_replicated_dimension(),
            Err(BatchingError::Type(TypeError::Invalid {
                message: "expected dimension type but got array type".to_string(),
            })),
        );
    }

    #[test]
    fn test_array_ir_batch_display() {
        // A carrier's `Typed` type is its logical per-item type, and `Display` renders that type, the batch axis, and
        // the packed value.
        let array =
            ArrayIrBatch::new(ArrayIrValue::Array(Array::matrix(2, 2, vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap()), 0)
                .unwrap();
        assert_eq!(array.r#type().into_owned(), ArrayIrType::Array(ArrayType::new_static(DataType::F32, [2])));
        assert_eq!(array.to_string(), "batch[f32[2], axis=0]([[1.0, 2.0], [3.0, 4.0]])");

        let reference =
            ArrayIrBatch::replicated(ArrayIrValue::Reference(ArrayReference::new(Array::scalar(1.0_f32).unwrap())));
        assert_eq!(
            reference.r#type().into_owned(),
            ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32))),
        );
        assert_eq!(reference.to_string(), "batch[ref<f32[]>, replicated](ref<f32[]>)");

        let dimension =
            ArrayIrBatch::replicated(ArrayIrValue::<Array>::Dimension(DimensionValue::constant(3).unwrap()));
        assert_eq!(dimension.r#type().into_owned(), dimension.value().r#type().into_owned());
        assert_eq!(dimension.to_string(), "batch[dimension<3>, replicated](3)");
    }

    #[test]
    fn test_threaded_extent_batched_program_new() -> Result<(), ProgramError> {
        type TestProgramBuilder = ProgramBuilder<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

        // A threaded boundary always contributes one leading bookkeeping input and output.
        let program = TestProgramBuilder::new().build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            Vec::new(),
            Vec::new(),
            Vec::new(),
        )?;
        let Err(ProgramError::MalformedProgram(message)) = ThreadedExtentBatchedProgram::new(program, Vec::new())
        else {
            panic!("threaded extent batching accepted a missing bookkeeping boundary");
        };
        assert_eq!(
            message,
            "a structurally batched program with a threaded extent must have a leading input and output",
        );

        // The leading bookkeeping input must be a first-class dimension rather than an arbitrary array IR member.
        let mut builder = TestProgramBuilder::new();
        let array = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let program = builder.build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            vec![array],
            vec![Placeholder],
            vec![Placeholder],
        )?;
        let Err(ProgramError::MalformedProgram(message)) = ThreadedExtentBatchedProgram::new(program, Vec::new())
        else {
            panic!("threaded extent batching accepted a non-dimension bookkeeping input");
        };
        assert_eq!(message, "a structurally batched program's leading threaded extent input must be a dimension",);

        // The leading bookkeeping output must also be a first-class dimension.
        let mut builder = TestProgramBuilder::new();
        builder.add_input(DimensionType::new("extent", DimensionBounds::new(0, Some(8))?).into());
        let array = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let program = builder.build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            vec![array],
            vec![Placeholder, Placeholder],
            vec![Placeholder],
        )?;
        let Err(ProgramError::MalformedProgram(message)) = ThreadedExtentBatchedProgram::new(program, Vec::new())
        else {
            panic!("threaded extent batching accepted a non-dimension bookkeeping output");
        };
        assert_eq!(message, "a structurally batched program's leading threaded extent output must be a dimension",);

        // A merely compatible dimension output is insufficient: the program must forward the exact input atom.
        let mut builder = TestProgramBuilder::new();
        builder.add_input(DimensionType::new("extent", DimensionBounds::new(0, Some(8))?).into());
        let other_extent =
            builder.add_input(DimensionType::new("other_extent", DimensionBounds::new(0, Some(8))?).into());
        let program = builder.build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            vec![other_extent],
            vec![Placeholder, Placeholder],
            vec![Placeholder],
        )?;
        let Err(ProgramError::MalformedProgram(message)) = ThreadedExtentBatchedProgram::new(program, Vec::new())
        else {
            panic!("threaded extent batching accepted a substituted bookkeeping output");
        };
        assert_eq!(
            message,
            "a structurally batched program's leading threaded extent output must forward its leading input",
        );

        // A well-formed threaded boundary preserves its program and excludes the bookkeeping output from its axes.
        let mut builder = TestProgramBuilder::new();
        let extent = builder.add_input(DimensionType::new("extent", DimensionBounds::new(0, Some(8))?).into());
        let program = builder.build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            vec![extent],
            vec![Placeholder],
            vec![Placeholder],
        )?;
        let (program, output_axes) = ThreadedExtentBatchedProgram::new(program, Vec::new())?.into_parts();
        assert_eq!(program.input_ids(), &[extent]);
        assert_eq!(program.output_ids(), &[extent]);
        assert!(output_axes.is_empty());

        Ok(())
    }

    #[test]
    fn test_static_array_extent_batching_policy() {
        // The static policy's `BatchingPolicy` surface is the `ArrayBatch` carrier: `batch` checks the mapped axis,
        // `replicated` records no axis, and the accessors read the carrier's value, axis, and per-item type.
        let matrix = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let batch = <StaticArrayExtentBatchingPolicy as BatchingPolicy<TestArrayContext>>::batch(
            matrix.clone(),
            BatchAxis::new(-1),
        )
        .unwrap();
        assert_eq!(batch, ArrayBatch::new(matrix.clone(), Some(1)).unwrap());
        assert_eq!(<StaticArrayExtentBatchingPolicy as BatchingPolicy<TestArrayContext>>::value(&batch), &matrix);
        assert_eq!(
            <StaticArrayExtentBatchingPolicy as BatchingPolicy<TestArrayContext>>::batch_axis(&batch),
            BatchAxis::new(1),
        );
        assert_eq!(
            <StaticArrayExtentBatchingPolicy as BatchingPolicy<TestArrayContext>>::unbatched_type(&batch).into_owned(),
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)])),
        );
        assert_eq!(
            <StaticArrayExtentBatchingPolicy as BatchingPolicy<TestArrayContext>>::batch(
                matrix.clone(),
                BatchAxis::new(2),
            ),
            Err(BatchingError::BatchAxisOutOfBounds {
                r#type: Box::new(matrix.r#type().into_owned()),
                axis: Axis::from(2),
            }),
        );
        let replicated =
            <StaticArrayExtentBatchingPolicy as BatchingPolicy<TestArrayContext>>::replicated(matrix.clone());
        assert_eq!(replicated, ArrayBatch::replicated(matrix.clone()));
        assert_eq!(
            <StaticArrayExtentBatchingPolicy as BatchingPolicy<TestArrayContext>>::batch_axis(&replicated),
            BatchAxis::replicated(),
        );
        assert_eq!(
            <StaticArrayExtentBatchingPolicy as BatchingPolicy<TestArrayContext>>::unbatched_type(&replicated)
                .into_owned(),
            matrix.r#type().into_owned(),
        );
    }

    #[test]
    fn test_static_array_extent_batching_policy_axis_dimension() {
        // The static policy's extent is one host `usize`, so the mapped axis is always an exact static dimension and
        // `axis_size` never fails.
        let context = BatchingContext::new(TestArrayContext::new(), 3);
        assert_eq!(StaticArrayExtentBatchingPolicy::axis_dimension(&context), Ok(Dimension::Static(3)));
        assert_eq!(StaticArrayExtentBatchingPolicy::axis_size(&context), Ok(3));
    }

    #[test]
    fn test_static_array_extent_batching_policy_match_axis() {
        // `match_axis` is `ArrayBatch::match_axis` fed with the context's extent and batch axis sharding: a mapped
        // batch is transposed onto the requested axis and a replicated batch is broadcast to gain it there.
        let context = BatchingContext::new(TestArrayContext::new(), 2);
        let batched =
            ArrayBatch::new(Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(), Some(0)).unwrap();
        let matched = StaticArrayExtentBatchingPolicy::match_axis(&context, &batched, Axis::from(1)).unwrap();
        assert_eq!(matched.batch_axis(), BatchAxis::new(1));
        assert_eq!(matched.value(), &Array::matrix(3, 2, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap());

        let replicated = ArrayBatch::replicated(Array::vector(vec![1.0, 2.0, 3.0]).unwrap());
        let matched = StaticArrayExtentBatchingPolicy::match_axis(&context, &replicated, Axis::from(0)).unwrap();
        assert_eq!(matched.batch_axis(), BatchAxis::new(0));
        assert_eq!(matched.value(), &Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]).unwrap());

        // The broadcast places the new batch axis with the context's batch axis sharding.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let context = context.with_axis_sharding(ShardingDimension::sharded(["x"]));
        let replicated_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]))
            .with_sharding(Sharding::replicated(mesh.clone(), 1))
            .unwrap();
        let replicated =
            ArrayBatch::replicated(Array::from_elements::<f64>(replicated_type, &[1.0, 2.0, 3.0]).unwrap());
        let matched = StaticArrayExtentBatchingPolicy::match_axis(&context, &replicated, Axis::from(0)).unwrap();
        assert_eq!(
            matched.value().r#type().sharding(),
            Some(
                &Sharding::new(mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()]).unwrap()
            ),
        );
    }

    #[test]
    fn test_static_array_extent_batching_policy_broadcast_input() {
        // The algorithm has already decided the input-to-output axis mapping and the result's batch axis, so the static
        // policy only broadcasts the packed value to the requested type. The dimension sources are not needed
        // because the type is fully static.
        let context = BatchingContext::new(TestArrayContext::new(), 2);
        let output_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let replicated = ArrayBatch::replicated(Array::vector(vec![1.0, 2.0, 3.0]).unwrap());
        let broadcasted = StaticArrayExtentBatchingPolicy::broadcast_input(
            &context,
            &replicated,
            output_type.clone(),
            vec![1],
            Axis::from(0),
            vec![DimensionSource::BatchExtent, DimensionSource::Static(3)],
        )
        .unwrap();
        assert_eq!(broadcasted.batch_axis(), BatchAxis::new(0));
        assert_eq!(broadcasted.value(), &Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]).unwrap());
    }

    #[test]
    fn test_dynamic_array_extent_batching_policy() -> Result<(), BatchingError> {
        // The dynamic policy carries projected array members of an array IR context in the same `ArrayBatch` carrier
        // as the static policy, so its `BatchingPolicy` surface is the same delegation over projected values.
        type Projected = ProjectedContext<ArrayIrTraceContext, ArrayType>;
        let trace = ArrayIrTraceContext::new();
        let matrix = trace.input(ArrayType::new_static(DataType::F32, [2, 3]).into()).into_projected()?;
        let batch =
            <DynamicArrayExtentBatchingPolicy as BatchingPolicy<Projected>>::batch(matrix.clone(), BatchAxis::new(-1))?;
        assert_eq!(<DynamicArrayExtentBatchingPolicy as BatchingPolicy<Projected>>::value(&batch), &matrix);
        assert_eq!(
            <DynamicArrayExtentBatchingPolicy as BatchingPolicy<Projected>>::batch_axis(&batch),
            BatchAxis::new(1),
        );
        assert_eq!(
            <DynamicArrayExtentBatchingPolicy as BatchingPolicy<Projected>>::unbatched_type(&batch).into_owned(),
            ArrayType::new_static(DataType::F32, [2]),
        );
        let replicated = <DynamicArrayExtentBatchingPolicy as BatchingPolicy<Projected>>::replicated(matrix.clone());
        assert_eq!(
            <DynamicArrayExtentBatchingPolicy as BatchingPolicy<Projected>>::batch_axis(&replicated),
            BatchAxis::replicated(),
        );
        assert_eq!(
            <DynamicArrayExtentBatchingPolicy as BatchingPolicy<Projected>>::unbatched_type(&replicated).into_owned(),
            ArrayType::new_static(DataType::F32, [2, 3]),
        );
        Ok(())
    }

    #[test]
    fn test_dynamic_array_extent_batching_policy_batched_type() -> Result<(), BatchingError> {
        // The inserted axis is described by the type of the first-class extent value, so a symbolic extent inserts the
        // dimension that it names rather than requiring a host size, while a replicated input keeps its declared type.
        type Projected = ProjectedContext<ArrayIrTraceContext, ArrayType>;
        let trace = ArrayIrTraceContext::new();
        let items = DimensionVariable::new("items", DimensionBounds::new(1, Some(9)).unwrap());
        let context = BatchingContext::<Projected, DynamicArrayExtentBatchingPolicy>::with_policy(
            ProjectedContext::new(trace.clone()),
            trace.input(DimensionType::from(items.clone()).into()),
        );
        let declared = ArrayType::new_static(DataType::F32, [2]);
        let expected = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(items), Dimension::Static(2)]));
        let mapped = ArrayBatch::new(trace.input(expected.clone().into()).into_projected()?, BatchAxis::new(0))?;
        assert_eq!(DynamicArrayExtentBatchingPolicy::batched_type(&context, &mapped, &declared), Ok(expected));
        let replicated = ArrayBatch::replicated(trace.input(declared.clone().into()).into_projected()?);
        assert_eq!(
            DynamicArrayExtentBatchingPolicy::batched_type(&context, &replicated, &declared),
            Ok(declared.clone()),
        );
        Ok(())
    }

    #[test]
    fn test_dynamic_array_extent_batching_policy_axis_dimension() -> Result<(), BatchingError> {
        // The mapped axis is described by the type of the first-class extent value: an exact dimension for a constant
        // extent, and the symbolic dimension it names otherwise, in which case there is no host `usize` to hand out.
        let trace = ArrayIrTraceContext::new();
        let context = BatchingContext::<_, ArrayBatchingPolicy<DynamicArrayExtentBatchingPolicy>>::with_policy(
            ProjectedContext::new(trace.clone()),
            trace.constant(ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap())),
        );
        assert_eq!(DynamicArrayExtentBatchingPolicy::axis_dimension(&context), Ok(Dimension::Static(2)));
        assert_eq!(DynamicArrayExtentBatchingPolicy::axis_size(&context), Ok(2));

        let items = DimensionVariable::new("items", DimensionBounds::new(1, Some(9)).unwrap());
        let context = BatchingContext::<_, ArrayBatchingPolicy<DynamicArrayExtentBatchingPolicy>>::with_policy(
            ProjectedContext::new(trace.clone()),
            trace.input(DimensionType::from(items.clone()).into()),
        );
        assert_eq!(DynamicArrayExtentBatchingPolicy::axis_dimension(&context), Ok(Dimension::Dynamic(items)));
        assert_eq!(
            DynamicArrayExtentBatchingPolicy::axis_size(&context),
            Err(BatchingError::UnsupportedOperation {
                message: "this batching rule requires a statically known mapped-axis extent".to_string(),
            }),
        );
        Ok(())
    }

    #[test]
    fn test_dynamic_array_extent_batching_policy_match_axis() -> Result<(), ProgramError> {
        let trace = ArrayIrTraceContext::new();
        let items = DimensionVariable::new("items", DimensionBounds::new(1, Some(9)).unwrap());
        let batch_extent = trace.input(DimensionType::from(items.clone()).into());
        let mapped = trace.input(
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3), Dimension::Dynamic(items.clone())]))
                .into(),
        );
        let replicated = trace.input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)])).into());
        let context = BatchingContext::<_, ArrayBatchingPolicy<DynamicArrayExtentBatchingPolicy>>::with_policy(
            ProjectedContext::new(trace.clone()),
            batch_extent,
        );

        // A mapped batch already carries the axis, so aligning it is a plain transpose that needs no extent at all.
        let batch = ArrayBatch::new(mapped.into_projected()?, BatchAxis::new(1))?;
        let moved = DynamicArrayExtentBatchingPolicy::match_axis(&context, &batch, Axis::from(0))?;
        assert_eq!(moved.batch_axis(), BatchAxis::new(0));
        assert_eq!(moved.unbatched_type(), ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)])),);
        let moved_id = moved.into_value().into_value().atom_id()?;

        // A replicated batch gains the mapped axis through a dynamic broadcast staged in the array IR parent, using
        // the first-class extent for the new axis, and the request is normalized against the rank after insertion.
        let batch = ArrayBatch::replicated(replicated.into_projected()?);
        let broadcasted = DynamicArrayExtentBatchingPolicy::match_axis(&context, &batch, Axis::from(-1))?;
        assert_eq!(broadcasted.batch_axis(), BatchAxis::new(1));
        assert_eq!(broadcasted.unbatched_type(), ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)])));
        let broadcasted_id = broadcasted.into_value().into_value().atom_id()?;
        assert_eq!(
            DynamicArrayExtentBatchingPolicy::match_axis(&context, &batch, Axis::from(2)),
            Err(BatchingError::BatchAxisOutOfBounds {
                r#type: Box::new(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)]))),
                axis: Axis::from(2),
            }),
        );

        let program = trace.builder().borrow().clone().build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            vec![moved_id, broadcasted_id],
            vec![Placeholder, Placeholder, Placeholder],
            vec![Placeholder, Placeholder],
        )?;
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:dimension<items ∈ [1, 9)>, %1:f32[3, items], %2:f32[3] .
                let %3:f32[items, 3] = transpose [permutation=[1, 0]] %1
                    %4:dimension<3> = constant [value=3]
                    %5:f32[3, items] = broadcast [output_axes=[0]] %2 %4 %0
                in (%3, %5)
            "}
            .trim_end(),
        );
        Ok(())
    }

    #[test]
    fn test_dynamic_array_extent_batching_policy_broadcast_input() -> Result<(), ProgramError> {
        // The algorithm has already decided the input-to-output axis mapping and the result's batch axis. The policy
        // materializes each output dimension source in first-class form (an exact constant for a static dimension, a
        // `dimension_size` read for a per-item dimension taken from another input, and the transform's own extent for
        // the mapped axis), then stages the mixed broadcast in the array IR parent.
        let trace = ArrayIrTraceContext::new();
        let items = DimensionVariable::new("items", DimensionBounds::new(1, Some(9)).unwrap());
        let width = DimensionVariable::new("width", DimensionBounds::new(1, Some(9)).unwrap());
        let batch_extent = trace.input(DimensionType::from(items.clone()).into());
        let replicated = trace.input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)])).into());
        let other = trace.input(
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3), Dimension::Dynamic(width.clone())]))
                .into(),
        );
        let context = BatchingContext::<_, ArrayBatchingPolicy<DynamicArrayExtentBatchingPolicy>>::with_policy(
            ProjectedContext::new(trace.clone()),
            batch_extent,
        );
        let input = ArrayBatch::replicated(replicated.into_projected()?);
        let output_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Dynamic(items), Dimension::Static(3), Dimension::Dynamic(width)]),
        );
        let output = DynamicArrayExtentBatchingPolicy::broadcast_input(
            &context,
            &input,
            output_type,
            vec![1],
            Axis::from(0),
            vec![
                DimensionSource::BatchExtent,
                DimensionSource::Static(3),
                DimensionSource::Value { source: other.into_projected()?, axis: 1 },
            ],
        )?;
        assert_eq!(output.batch_axis(), BatchAxis::new(0));
        let output_id = output.into_value().into_value().atom_id()?;

        let program = trace.builder().borrow().clone().build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            vec![output_id],
            vec![Placeholder, Placeholder, Placeholder],
            vec![Placeholder],
        )?;
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:dimension<items ∈ [1, 9)>, %1:f32[3], %2:f32[3, width] .
                let %3:dimension<3> = constant [value=3]
                    %4:dimension<width ∈ [1, 9)> = dimension_size [axis=1] %2
                    %5:f32[items, 3, width] = broadcast [output_axes=[1]] %1 %0 %3 %4
                in (%5)
            "}
            .trim_end(),
        );
        Ok(())
    }

    #[test]
    fn test_array_batching_policy() {
        // The wrapper is a zero-sized nominal marker that lends the homogeneous array rules a policy type of their own.
        // It renders without its extent-policy parameter, since that parameter only selects the extent discipline.
        assert_eq!(
            format!("{:?}", ArrayBatchingPolicy::<StaticArrayExtentBatchingPolicy>::default()),
            "ArrayBatchingPolicy"
        );
        assert_eq!(
            format!("{:?}", ArrayBatchingPolicy::<DynamicArrayExtentBatchingPolicy>::default()),
            "ArrayBatchingPolicy"
        );

        // Every `BatchingPolicy` function delegates to the extent policy, so the carrier surface is `ArrayBatch`'s.
        let matrix = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let batch =
            <ArrayBatchingPolicy as BatchingPolicy<TestArrayContext>>::batch(matrix.clone(), BatchAxis::new(-1))
                .unwrap();
        assert_eq!(batch, ArrayBatch::new(matrix.clone(), Some(1)).unwrap());
        assert_eq!(<ArrayBatchingPolicy as BatchingPolicy<TestArrayContext>>::value(&batch), &matrix);
        assert_eq!(<ArrayBatchingPolicy as BatchingPolicy<TestArrayContext>>::batch_axis(&batch), BatchAxis::new(1));
        assert_eq!(
            <ArrayBatchingPolicy as BatchingPolicy<TestArrayContext>>::unbatched_type(&batch).into_owned(),
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)])),
        );
        assert_eq!(
            <ArrayBatchingPolicy as BatchingPolicy<TestArrayContext>>::replicated(matrix.clone()),
            ArrayBatch::replicated(matrix.clone()),
        );

        // A static extent is a host `usize` and adds no boundary program inputs.
        assert_eq!(<ArrayBatchingPolicy as BatchingPolicy<TestArrayContext>>::boundary_inputs(&2), Vec::<Array>::new());
    }

    #[test]
    fn test_array_batching_policy_batched_type() {
        // The mapped axis is inserted at the input's batch-axis position with the context's extent, while a replicated
        // input keeps its declared type.
        let context =
            BatchingContext::<_, ArrayBatchingPolicy>::new(EagerContext::<Array, ConstantOperation<Array>>::new(), 3);
        let declared = ArrayType::new_static(DataType::F32, [2]);
        let input = ArrayBatch::new(Array::matrix(3, 2, vec![1f32; 6]).unwrap(), BatchAxis::new(0)).unwrap();
        let expected = ArrayType::new_static(DataType::F32, [3, 2]);
        assert_eq!(ArrayBatchingPolicy::batched_type(&context, &input, &declared), Ok(expected));
        let replicated = ArrayBatch::replicated(Array::vector(vec![1f32; 2]).unwrap());
        assert_eq!(ArrayBatchingPolicy::batched_type(&context, &replicated, &declared), Ok(declared.clone()));

        // Only the mapped manual axes take their variation from physical inputs. A mismatch on y must remain
        // visible to the caller's full type validation, while x can be invariant or varying in an existing packed value.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let item = declared.clone().with_sharding(Sharding::replicated(mesh, 1)).unwrap();
        let packed = item.batched(0, Dimension::Static(3), ShardingDimension::sharded(["x"])).unwrap();
        let manual = context.with_axis_sharding(ShardingDimension::sharded(["x"]));
        for varying in [vec![], vec!["x"], vec!["y"], vec!["x", "y"]] {
            let actual = packed
                .clone()
                .with_sharding(packed.sharding().unwrap().clone().with_varying_manual_axes(varying.clone()).unwrap())
                .unwrap();
            let expected = packed
                .clone()
                .with_sharding(
                    packed
                        .sharding()
                        .unwrap()
                        .clone()
                        .with_varying_manual_axes(varying.into_iter().filter(|axis| *axis == "x"))
                        .unwrap(),
                )
                .unwrap();
            let input = ArrayBatch::new(Array::from_elements(actual, &[1f32; 6]).unwrap(), BatchAxis::new(0)).unwrap();
            assert_eq!(ArrayBatchingPolicy::batched_type(&manual, &input, &item), Ok(expected.clone()));
            assert_eq!(ArrayBatchingPolicy::batched_type(&manual, &input, &declared), Ok(expected));
        }
        // Optional placement does not authorize inheriting a pending or completed reduction on an unrelated axis.
        for sharding in [
            packed.sharding().unwrap().clone().with_unreduced_axes(["y"]).unwrap(),
            packed.sharding().unwrap().clone().with_reduced_axes(["y"]).unwrap(),
        ] {
            let actual = packed.clone().with_sharding(sharding).unwrap();
            let input = ArrayBatch::new(Array::from_elements(actual, &[1f32; 6]).unwrap(), BatchAxis::new(0)).unwrap();
            assert_eq!(ArrayBatchingPolicy::batched_type(&manual, &input, &declared), Ok(packed.clone()));
        }
    }

    #[test]
    fn test_array_batching_policy_adapt_batched_program() -> Result<(), BatchingError> {
        type Parent = EagerContext<Array, ArrayOperation<Array>>;

        // Structurally batch `x ↦ (x + 1, x * x)` over a leading axis of size 2.
        let parent = DomainTracingContext::<Parent>::new();
        let builder = parent.builder().clone();
        let input_atom = builder
            .borrow_mut()
            .add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)])));
        let input = parent.tracer(input_atom, None);
        let shifted = input.clone() + parent.lift(Array::scalar(1.0).unwrap())?;
        let squared = input.clone() * input;
        let program = builder.borrow().clone().build::<Array, Vec<Array>>(
            vec![shifted.atom_id()?, squared.atom_id()?],
            Placeholder,
            vec![Placeholder, Placeholder],
        )?;
        let context = BatchingContext::new(Parent::new(), 2);
        let batched = <ArrayBatchingPolicy as RecursiveBatchingPolicy<Parent>>::batch_program(
            &context,
            program.entry_region_ref(),
            &[BatchAxis::new(0)],
            ProgramBatchingOutputAxesPolicy::Natural,
        )?;

        let rendered = batched.into_parts().0.to_string();

        // Without required output axes the parts are rewrapped as they are: the array policies widen no boundary, so
        // there are no bookkeeping outputs to drop.
        let batched = <ArrayBatchingPolicy as RecursiveBatchingPolicy<Parent>>::batch_program(
            &context,
            program.entry_region_ref(),
            &[BatchAxis::new(0)],
            ProgramBatchingOutputAxesPolicy::Natural,
        )?;
        let adapted =
            <ArrayBatchingPolicy as BatchingPolicy<Parent>>::adapt_batched_program(batched, None, |_, _, _| {
                panic!("no output needs collapsing")
            })?;
        assert_eq!(adapted.output_axes(), &[BatchAxis::new(0), BatchAxis::new(0)]);
        assert_eq!(adapted.into_parts().0.to_string(), rendered);

        // A mapped output whose consumer requires it replicated is collapsed through the caller's function inside the
        // replay (here by summation), and the reported axes say so; outputs whose requirement is met stay untouched.
        let batched = <ArrayBatchingPolicy as RecursiveBatchingPolicy<Parent>>::batch_program(
            &context,
            program.entry_region_ref(),
            &[BatchAxis::new(0)],
            ProgramBatchingOutputAxesPolicy::Natural,
        )?;
        let adapted = <ArrayBatchingPolicy as BatchingPolicy<Parent>>::adapt_batched_program(
            batched,
            Some(&[BatchAxis::replicated(), BatchAxis::new(0)]),
            |_, output, axis| Ok(output.reduce(&[axis.normalize(2)?], ReductionKind::Sum)?),
        )?;
        assert_eq!(adapted.output_axes(), &[BatchAxis::replicated(), BatchAxis::new(0)]);
        assert_eq!(
            adapted.into_parts().0.to_string(),
            indoc! {"
                lambda %0:f64[2, 3] .
                let %1:f64[] = const 1.0
                    %2:f64[2, 3] = broadcast [output_type=f64[2, 3], output_axes=[]] %1
                    %3:f64[2, 3] = add %0 %2
                    %4:f64[2, 3] = mul %0 %0
                    %5:f64[3] = reduce [kind=sum, axes=[0]] %3
                in (%5, %4)
            "}
            .trim_end(),
        );
        Ok(())
    }

    #[test]
    fn test_array_batching_policy_batching_level() {
        let context = BatchingContext::new(TestArrayContext::new(), 3)
            .with_axis_name("items".to_string())
            .with_axis_sharding(ShardingDimension::Replicated);
        assert_eq!(
            <ArrayBatchingPolicy as RecursiveBatchingPolicy<TestArrayContext>>::batching_level(&context),
            BatchingLevel::new(
                BatchingLevelExtent::Static(3),
                Some("items".to_string()),
                ShardingDimension::Replicated,
            ),
        );
    }

    #[test]
    fn test_array_batching_policy_batch_program_at_level() -> Result<(), BatchingError> {
        // Batching at a recorded level needs no live context and matches batching under the context that produced it.
        let (_, program) = TestArrayContext::trace(|x| Ok(x.clone() * x), ArrayType::scalar(DataType::F64))?;
        let context = BatchingContext::new(TestArrayContext::new(), 2);
        let level = <ArrayBatchingPolicy as RecursiveBatchingPolicy<TestArrayContext>>::batching_level(&context);
        let batched = <ArrayBatchingPolicy as RecursiveBatchingPolicy<TestArrayContext>>::batch_program_at_level(
            &level,
            program.entry_region_ref(),
            &[BatchAxis::new(0)],
            ProgramBatchingOutputAxesPolicy::Natural,
        )?;
        let expected = <ArrayBatchingPolicy as RecursiveBatchingPolicy<TestArrayContext>>::batch_program(
            &context,
            program.entry_region_ref(),
            &[BatchAxis::new(0)],
            ProgramBatchingOutputAxesPolicy::Natural,
        )?;
        assert_eq!(batched.output_axes(), expected.output_axes());
        assert_eq!(batched.into_parts().0.to_string(), expected.into_parts().0.to_string());

        // A first-class extent has no homogeneous representation.
        let dynamic = BatchingLevel::new(
            BatchingLevelExtent::Dynamic(ArrayType::scalar(DataType::I64)),
            None,
            ShardingDimension::Replicated,
        );
        assert!(matches!(
            <ArrayBatchingPolicy as RecursiveBatchingPolicy<TestArrayContext>>::batch_program_at_level(
                &dynamic,
                program.entry_region_ref(),
                &[BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            ),
            Err(BatchingError::UnsupportedOperation { message })
                if message == "homogeneous array batching requires a static batch extent",
        ));
        Ok(())
    }

    #[test]
    fn test_array_batching_policy_batch_program() -> Result<(), BatchingError> {
        // Structural batching of a region is `RegionRef::batched` under the context's extent, axis name, and batch axis
        // sharding, so the result is a boundary-preserving program over the packed types.
        let parent = DomainTracingContext::<TestArrayContext>::new();
        let builder = parent.builder().clone();
        let input_atom = builder
            .borrow_mut()
            .add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)])));
        let input = parent.tracer(input_atom, None);
        let shifted = input.clone() + parent.lift(Array::scalar(1.0).unwrap())?;
        let squared = input.clone() * input;
        let program = builder.borrow().clone().build::<Array, Vec<Array>>(
            vec![shifted.atom_id()?, squared.atom_id()?],
            Placeholder,
            vec![Placeholder, Placeholder],
        )?;
        let context = BatchingContext::new(TestArrayContext::new(), 2);
        let batched = <ArrayBatchingPolicy as RecursiveBatchingPolicy<TestArrayContext>>::batch_program(
            &context,
            program.entry_region_ref(),
            &[BatchAxis::new(0)],
            ProgramBatchingOutputAxesPolicy::Natural,
        )?;
        assert_eq!(batched.output_axes(), &[BatchAxis::new(0), BatchAxis::new(0)]);
        assert_eq!(
            batched.into_parts().0.to_string(),
            indoc! {"
                lambda %0:f64[2, 3] .
                let %1:f64[] = const 1.0
                    %2:f64[2, 3] = broadcast [output_type=f64[2, 3], output_axes=[]] %1
                    %3:f64[2, 3] = add %0 %2
                    %4:f64[2, 3] = mul %0 %0
                in (%3, %4)
            "}
            .trim_end(),
        );
        Ok(())
    }

    #[test]
    fn test_array_batching_policy_batch_region() -> Result<(), BatchingError> {
        // The homogeneous replay interprets the region over `ArrayBatch` carriers: constants are lifted into the parent
        // and replicated, and every instruction runs through its batching rule against the eager parent.
        let parent = DomainTracingContext::<TestArrayContext>::new();
        let builder = parent.builder().clone();
        let input_atom = builder
            .borrow_mut()
            .add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)])));
        let input = parent.tracer(input_atom, None);
        let shifted = input.clone() + parent.lift(Array::scalar(1.0).unwrap())?;
        let squared = input.clone() * input;
        let program = builder.borrow().clone().build::<Array, Vec<Array>>(
            vec![shifted.atom_id()?, squared.atom_id()?],
            Placeholder,
            vec![Placeholder, Placeholder],
        )?;
        let context = BatchingContext::new(TestArrayContext::new(), 2);
        let outputs = <ArrayBatchingPolicy as RecursiveBatchingPolicy<TestArrayContext>>::batch_region(
            &context,
            program.entry_region_ref(),
            vec![ArrayBatch::new(Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(), Some(0))?],
        )?;
        assert_eq!(
            outputs,
            vec![
                ArrayBatch::new(Array::matrix(2, 3, vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap(), Some(0))?,
                ArrayBatch::new(Array::matrix(2, 3, vec![1.0, 4.0, 9.0, 16.0, 25.0, 36.0]).unwrap(), Some(0))?,
            ],
        );
        Ok(())
    }

    #[test]
    fn test_array_batching_policy_align_batch_axis() -> Result<(), BatchingError> {
        // The homogeneous policy stages alignment through the parent's operation family: a replicated batch is
        // broadcast to gain the mapped axis at the requested position, a mapped batch moves its axis there, and a
        // batch that already carries the requested axis is returned unchanged.
        let context = BatchingContext::new(TestArrayContext::new(), 2);
        let replicated = ArrayBatch::replicated(Array::vector(vec![1.0, 2.0, 3.0]).unwrap());
        let broadcasted = <ArrayBatchingPolicy as RecursiveBatchingPolicy<TestArrayContext>>::align_batch_axis(
            &context,
            replicated,
            Axis::from(1),
        )?;
        assert_eq!(broadcasted.batch_axis(), BatchAxis::new(1));
        assert_eq!(broadcasted.value(), &Array::matrix(3, 2, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]).unwrap());
        let mapped = ArrayBatch::new(Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(), Some(0))?;
        let moved = <ArrayBatchingPolicy as RecursiveBatchingPolicy<TestArrayContext>>::align_batch_axis(
            &context,
            mapped.clone(),
            Axis::from(1),
        )?;
        assert_eq!(moved.batch_axis(), BatchAxis::new(1));
        assert_eq!(moved.value(), &Array::matrix(3, 2, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap());
        assert_eq!(
            <ArrayBatchingPolicy as RecursiveBatchingPolicy<TestArrayContext>>::align_batch_axis(
                &context,
                mapped.clone(),
                Axis::from(0),
            )?,
            mapped,
        );
        Ok(())
    }

    #[test]
    fn test_array_batching_policy_static_batch_axis_extent() {
        // The homogeneous policy's extent is one host `usize` fixed when the transform was constructed, so it is always
        // statically known (e.g., to named-axis queries).
        let context = BatchingContext::new(TestArrayContext::new(), 3);
        assert_eq!(ArrayBatchingPolicy::static_batch_axis_extent(&context), Some(3));
    }

    #[test]
    fn test_array_batching_policy_diagonal() -> Result<(), BatchingError> {
        type Context = EagerContext<Array, ArrayOperation<Array>>;
        let diagonal = |value: &Array, kept_axis: usize, removed_axis: usize| {
            <ArrayBatchingPolicy as DiagonalBatchingPolicy<Context>>::diagonal(
                &Context::new(),
                value,
                kept_axis,
                removed_axis,
            )
        };

        // The diagonal of a square matrix is the same along either axis order.
        let matrix = Array::matrix(3, 3, vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])?;
        assert_eq!(diagonal(&matrix, 0, 1), Ok(Array::vector(vec![1.0f64, 5.0, 9.0])?));
        assert_eq!(diagonal(&matrix, 1, 0), Ok(Array::vector(vec![1.0f64, 5.0, 9.0])?));

        // The other axes keep their positions, and the kept axis moves one position earlier when it follows the
        // removed axis: element `(i, j)` of the first result and element `(j, i)` of the second are element `(i, j, i)`
        // of the input, whose value is `7i + 3j`.
        let values = (0..18).map(f64::from).collect::<Vec<_>>();
        let cube = Array::from_elements(ArrayType::new_static(DataType::F64, [3, 2, 3]), &values)?;
        assert_eq!(diagonal(&cube, 0, 2), Ok(Array::matrix(3, 2, vec![0.0f64, 3.0, 7.0, 10.0, 14.0, 17.0])?));
        assert_eq!(diagonal(&cube, 2, 0), Ok(Array::matrix(2, 3, vec![0.0f64, 7.0, 14.0, 3.0, 10.0, 17.0])?));

        // Elements are copied exactly, including signed zeros, `NaN`s, and complex values.
        let special = diagonal(&Array::matrix(2, 2, vec![-0.0f64, 1.0, 2.0, f64::NAN])?, 0, 1)?.elements::<f64>()?;
        assert!(special[0] == 0.0 && special[0].is_sign_negative() && special[1].is_nan());
        let complex = Array::matrix(
            2,
            2,
            vec![Complex::new(1.0f64, -1.0), Complex::new(2.0, 0.0), Complex::new(3.0, 0.0), Complex::new(4.0, 4.0)],
        )?;
        assert_eq!(
            diagonal(&complex, 0, 1),
            Ok(Array::vector(vec![Complex::new(1.0f64, -1.0), Complex::new(4.0, 4.0)])?),
        );

        // Staging records one gather whose indices count along the kept axis.
        let (_, program) = DomainTracingContext::<Context>::trace(
            |value| {
                Ok(
                        <ArrayBatchingPolicy as DiagonalBatchingPolicy<
                            TracingContext<Array, ArrayOperation<Array>>,
                        >>::diagonal(value.context(), &value, 0, 1)?,
                    )
            },
            ArrayType::new_static(DataType::F64, [3, 3]),
        )
        .map_err(BatchingError::Program)?;
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[3, 3] .
                let %1:i64[3, 1] = iota [type=i64[3, 1], dimension=0]
                    %2:f64[3] = gather [
                        dimensions=(offset=[], collapsed_slice=[1], start_index_map=[1], batching=[(0, 0)]),
                        slice_sizes=[1, 1],
                    ] %0 %1
                in (%2)
            "}
            .trim_end(),
        );

        // The axes must be two distinct in-range axes with equal extents, and a homogeneous context can only take the
        // diagonal of static extents.
        let invalid_axes = "cannot take the diagonal of a value of type `f64[3, 3]` along axes {} and {}, \
                            which must be two distinct axes of its rank 2";
        assert_eq!(diagonal(&matrix, 0, 0), Err(TypeError::invalid(invalid_axes.replacen("{}", "0", 2)).into()),);
        assert_eq!(
            diagonal(&matrix, 0, 2),
            Err(TypeError::invalid(invalid_axes.replacen("{}", "0", 1).replacen("{}", "2", 1)).into()),
        );
        assert_eq!(
            diagonal(&Array::matrix(2, 3, vec![0.0f64; 6])?, 0, 1),
            Err(TypeError::invalid(
                "cannot take the diagonal of a value of type `f64[2, 3]` along axes 0 and 1, whose extents differ",
            )
            .into()),
        );
        let extent = DimensionVariable::new("n", DimensionBounds::new(1, Some(8)).unwrap());
        let dynamic = Dimension::Dynamic(extent);
        let result = DomainTracingContext::<Context>::trace(
            |value| {
                Ok(<ArrayBatchingPolicy as DiagonalBatchingPolicy<TracingContext<Array, ArrayOperation<Array>>>>::diagonal(
                    value.context(),
                    &value,
                    0,
                    1,
                )?)
            },
            ArrayType::new(DataType::F64, Shape::new(vec![dynamic.clone(), dynamic])),
        );
        assert_eq!(
            result.map(|_| ()),
            Err(ProgramError::from(BatchingError::UnsupportedOperation {
                message: "cannot take the diagonal of a value of type `f64[n, n]` with dynamic extents in a \
                          homogeneous array context"
                    .to_string(),
            })),
        );
        Ok(())
    }

    #[test]
    fn test_array_batching_policy_pack_inputs() -> Result<(), BatchingError> {
        let parent = TestArrayContext::new();
        let matrix = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let vector = Array::vector(vec![7.0, 8.0]).unwrap();

        // Each input is wrapped with its declared axis (normalized against its rank), the extent is inferred from the
        // mapped inputs, and the context records that extent with no axis name and a replicated batch axis placement.
        let (context, inputs) = <ArrayBatchingPolicy as BatchingEntrypointPolicy<TestArrayContext>>::pack_inputs(
            &parent,
            vec![matrix.clone(), vector.clone()],
            vec![BatchAxis::new(-2), BatchAxis::replicated()],
            BatchAxisSpecification::default(),
        )?;
        assert_eq!(*context.axis_extent(), 2);
        assert_eq!(context.axis_name(), None);
        assert_eq!(context.axis_sharding(), &ShardingDimension::replicated());
        assert_eq!(inputs, vec![ArrayBatch::new(matrix.clone(), Some(0))?, ArrayBatch::replicated(vector.clone())]);

        // An explicit extent must agree with the inferred one, and a name becomes the context's axis name.
        let (context, _) = <ArrayBatchingPolicy as BatchingEntrypointPolicy<TestArrayContext>>::pack_inputs(
            &parent,
            vec![matrix.clone()],
            vec![BatchAxis::new(0)],
            BatchAxisSpecification::new(2, "items"),
        )?;
        assert_eq!(*context.axis_extent(), 2);
        assert_eq!(context.axis_name(), Some("items"));
        assert_eq!(
            <ArrayBatchingPolicy as BatchingEntrypointPolicy<TestArrayContext>>::pack_inputs(
                &parent,
                vec![matrix.clone()],
                vec![BatchAxis::new(0)],
                BatchAxisSpecification::with_extent(5),
            )
            .err(),
            Some(BatchingError::MismatchedBatchSizes { expected: 5, actual: 2 }),
        );

        // An explicit extent alone still permits an input-free transform, while without it there is no mapped
        // dimension to infer the extent from.
        let (context, inputs) = <ArrayBatchingPolicy as BatchingEntrypointPolicy<TestArrayContext>>::pack_inputs(
            &parent,
            Vec::new(),
            Vec::new(),
            BatchAxisSpecification::with_extent(3),
        )?;
        assert_eq!(*context.axis_extent(), 3);
        assert_eq!(inputs, Vec::new());
        assert_eq!(
            <ArrayBatchingPolicy as BatchingEntrypointPolicy<TestArrayContext>>::pack_inputs(
                &parent,
                Vec::new(),
                Vec::new(),
                BatchAxisSpecification::default(),
            )
            .err(),
            Some(BatchingError::EmptyBatch),
        );
        assert_eq!(
            <ArrayBatchingPolicy as BatchingEntrypointPolicy<TestArrayContext>>::pack_inputs(
                &parent,
                vec![vector.clone()],
                vec![BatchAxis::replicated()],
                BatchAxisSpecification::default(),
            )
            .err(),
            Some(BatchingError::EmptyBatch),
        );

        // The flat axis declaration is validated against the inputs before zipping so that unmatched entries cannot be
        // silently dropped.
        assert_eq!(
            <ArrayBatchingPolicy as BatchingEntrypointPolicy<TestArrayContext>>::pack_inputs(
                &parent,
                vec![matrix.clone()],
                vec![BatchAxis::new(0), BatchAxis::new(0)],
                BatchAxisSpecification::default(),
            )
            .err(),
            Some(BatchingError::Program(ProgramError::InvalidInputCount { expected: 1, actual: 2 })),
        );

        // A mapped input whose batch dimension is dynamic yields no host extent to infer.
        let dynamic_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Dynamic(DimensionVariable::new("dynamic", DimensionBounds::unbounded()))]),
        );
        let dynamic = Array::new_unchecked(
            dynamic_type.clone(),
            Arc::new([1.0_f64, 2.0, 3.0].into_iter().flat_map(f64::to_le_bytes).collect()),
        );
        assert_eq!(
            <ArrayBatchingPolicy as BatchingEntrypointPolicy<TestArrayContext>>::pack_inputs(
                &parent,
                vec![dynamic],
                vec![BatchAxis::new(0)],
                BatchAxisSpecification::default(),
            )
            .err(),
            Some(BatchingError::DynamicBatchAxis { r#type: Box::new(dynamic_type), axis: Axis::from(0) }),
        );

        // The batch axis placement is chosen from the mapped inputs, and a mapped input whose packed sharding differs
        // from it is normalized onto that placement before tracing begins.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharded_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)]))
            .with_sharding(Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        let replicated_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)]))
            .with_sharding(Sharding::replicated(mesh, 1))
            .unwrap();
        let (context, inputs) = <ArrayBatchingPolicy as BatchingEntrypointPolicy<TestArrayContext>>::pack_inputs(
            &parent,
            vec![
                Array::from_elements::<f64>(sharded_type.clone(), &[1.0, 2.0]).unwrap(),
                Array::from_elements::<f64>(replicated_type, &[3.0, 4.0]).unwrap(),
            ],
            vec![BatchAxis::new(0), BatchAxis::new(0)],
            BatchAxisSpecification::default(),
        )?;
        assert_eq!(context.axis_sharding(), &ShardingDimension::sharded(["x"]));
        assert_eq!(
            inputs,
            vec![
                ArrayBatch::new(Array::from_elements::<f64>(sharded_type.clone(), &[1.0, 2.0]).unwrap(), Some(0))?,
                ArrayBatch::new(Array::from_elements::<f64>(sharded_type, &[3.0, 4.0]).unwrap(), Some(0))?,
            ],
        );
        Ok(())
    }

    #[test]
    fn test_array_batching_policy_materialize_output() -> Result<(), BatchingError> {
        type Parent = EagerContext<Array, ArrayOperation<Array>>;
        let context = BatchingContext::new(Parent::new(), 3);
        let matrix = Array::matrix(3, 4, (0..12).map(f64::from).collect()).unwrap();
        let rows = ArrayBatch::new(matrix.clone(), Some(0))?;

        // An output already on the requested axis is unwrapped as it is, and a different mapped position transposes the
        // packed value so that the mapped axis lands there.
        assert_eq!(
            <ArrayBatchingPolicy as BatchingEntrypointPolicy<Parent>>::materialize_output(
                &context,
                rows.clone(),
                BatchAxis::new(0),
            ),
            Ok(matrix),
        );
        assert_eq!(
            <ArrayBatchingPolicy as BatchingEntrypointPolicy<Parent>>::materialize_output(
                &context,
                rows.clone(),
                BatchAxis::new(1),
            ),
            Ok(Array::matrix(4, 3, vec![0.0, 4.0, 8.0, 1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0]).unwrap()),
        );

        // A replicated output declared mapped is broadcast across the batch, whereas collapsing a mapped output into a
        // replicated declaration requires an explicit reduction and is rejected.
        let replicated = ArrayBatch::replicated(Array::vector(vec![1.0, 2.0]).unwrap());
        assert_eq!(
            <ArrayBatchingPolicy as BatchingEntrypointPolicy<Parent>>::materialize_output(
                &context,
                replicated,
                BatchAxis::new(0),
            ),
            Ok(Array::matrix(3, 2, vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]).unwrap()),
        );
        assert_eq!(
            <ArrayBatchingPolicy as BatchingEntrypointPolicy<Parent>>::materialize_output(
                &context,
                rows.clone(),
                BatchAxis::replicated(),
            ),
            Err(BatchingError::MismatchedOutputAxes { expected: BatchAxis::replicated(), actual: BatchAxis::new(0) }),
        );

        Ok(())
    }

    #[test]
    fn test_array_ir_batching_policy_batch() {
        // The array IR policy's carrier is `ArrayIrBatch`, so `batch` derives the member kind from the value's type
        // and checks the axis against it: arrays and references map an axis of the (referent) array, while a
        // first-class dimension admits only the replicated axis.
        let matrix = ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap());
        let batch =
            <ArrayIrBatchingPolicy as BatchingPolicy<ArrayIrEagerContext>>::batch(matrix.clone(), BatchAxis::new(-2))
                .unwrap();
        assert_eq!(batch, ArrayIrBatch::new(matrix.clone(), BatchAxis::new(0)).unwrap());
        assert_eq!(<ArrayIrBatchingPolicy as BatchingPolicy<ArrayIrEagerContext>>::value(&batch), &matrix);
        assert_eq!(
            <ArrayIrBatchingPolicy as BatchingPolicy<ArrayIrEagerContext>>::batch_axis(&batch),
            BatchAxis::new(0),
        );
        assert_eq!(
            <ArrayIrBatchingPolicy as BatchingPolicy<ArrayIrEagerContext>>::unbatched_type(&batch).into_owned(),
            ArrayIrType::Array(ArrayType::new_static(DataType::F32, [3])),
        );

        // A captured reference is unbatched by definition, so batching one at the replicated axis produces a replicated
        // carrier whose per-item type is the reference type itself.
        let reference = ArrayIrValue::Reference(ArrayReference::new(Array::scalar(1.0_f32).unwrap()));
        let batch = <ArrayIrBatchingPolicy as BatchingPolicy<ArrayIrEagerContext>>::batch(
            reference.clone(),
            BatchAxis::replicated(),
        )
        .unwrap();
        assert_eq!(batch, ArrayIrBatch::replicated(reference.clone()));
        assert_eq!(
            <ArrayIrBatchingPolicy as BatchingPolicy<ArrayIrEagerContext>>::replicated(reference.clone()),
            ArrayIrBatch::replicated(reference),
        );
        assert_eq!(
            <ArrayIrBatchingPolicy as BatchingPolicy<ArrayIrEagerContext>>::unbatched_type(&batch).into_owned(),
            ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32))),
        );

        // Eager dimension values are typed exactly, so diagnostics report the value's own exact type.
        let dimension_value =
            DimensionValue::new(DimensionType::new("n", DimensionBounds::new(1, Some(8)).unwrap()), 2).unwrap();
        let dimension_type = dimension_value.r#type().into_owned();
        let dimension = ArrayIrValue::<Array>::Dimension(dimension_value);
        assert_eq!(
            <ArrayIrBatchingPolicy as BatchingPolicy<ArrayIrEagerContext>>::batch(dimension, BatchAxis::new(0)),
            Err(BatchingError::MappedDimension { r#type: Box::new(dimension_type), axis: BatchAxis::new(0) }),
        );
    }

    #[test]
    fn test_array_ir_batching_policy_batched_type() {
        // Mapped arrays gain the batch axis directly and mapped references gain it in their referents, while replicated
        // members, including first-class dimensions, keep their declared types.
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            EagerContext::<ArrayIrValue<Array>, ConstantOperation<ArrayIrValue<Array>>>::new(),
            ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap()),
        );
        let declared = ArrayType::new_static(DataType::F32, [2]);
        let value = Array::matrix(3, 2, vec![1f32; 6]).unwrap();
        let expected = ArrayType::new_static(DataType::F32, [3, 2]);
        let input = ArrayIrBatch::new(ArrayIrValue::Array(value.clone()), BatchAxis::new(0)).unwrap();
        assert_eq!(
            ArrayIrBatchingPolicy::batched_type(&context, &input, &declared.clone().into()),
            Ok(expected.clone().into()),
        );
        let replicated = ArrayIrBatch::replicated(ArrayIrValue::Array(Array::vector(vec![1f32; 2]).unwrap()));
        assert_eq!(
            ArrayIrBatchingPolicy::batched_type(&context, &replicated, &declared.clone().into()),
            Ok(declared.clone().into()),
        );
        let reference =
            ArrayIrBatch::new(ArrayIrValue::Reference(ArrayReference::new(value)), BatchAxis::new(0)).unwrap();
        assert_eq!(
            ArrayIrBatchingPolicy::batched_type(&context, &reference, &ReferenceType::new(declared).into()),
            Ok(ReferenceType::new(expected).into()),
        );
        let dimension = ArrayIrBatch::replicated(ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()));
        assert_eq!(
            ArrayIrBatchingPolicy::batched_type(&context, &dimension, dimension.value().r#type().as_ref()),
            Ok(dimension.value().r#type().into_owned()),
        );

        // Mapped declarations require a first-class extent, whose symbolic identity is retained without staging work.
        let invalid = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            *context.parent(),
            ArrayIrValue::Array(Array::scalar(3i32).unwrap()),
        );
        assert_eq!(
            ArrayIrBatchingPolicy::batched_type(&invalid, &input, &input.unbatched_type()),
            Err(TypeError::invalid("batch extent must be a dimension").into()),
        );
        let parent = TracingContext::<ArrayIrValue<Array>, ConstantOperation<ArrayIrValue<Array>>>::new();
        let dimension = DimensionType::new("batch", DimensionBounds::new(0, Some(5)).unwrap());
        let context =
            BatchingContext::<_, ArrayIrBatchingPolicy>::new(parent.clone(), parent.input(dimension.clone().into()));
        let declared = ArrayType::new_static(DataType::F32, [2]);
        let packed = ArrayType::new(DataType::F32, Shape::new(vec![dimension.simplified_dimension(), 2.into()]));
        let input = ArrayIrBatch::new(parent.input(packed.clone().into()), BatchAxis::new(0)).unwrap();
        assert_eq!(ArrayIrBatchingPolicy::batched_type(&context, &input, &declared.into()), Ok(packed.into()));
        assert!(parent.builder().borrow().instructions().is_empty());
    }

    #[test]
    fn test_array_ir_batching_policy_boundary_inputs() {
        // The adapted program's leading input still defines the dimension variable referenced by every inserted batch
        // dimension, so the first-class extent value becomes its matching program input.
        let extent = ArrayIrValue::<Array>::Dimension(DimensionValue::constant(2).unwrap());
        assert_eq!(
            <ArrayIrBatchingPolicy as BatchingPolicy<ArrayIrEagerContext>>::boundary_inputs(&extent),
            vec![extent],
        );
    }

    #[test]
    fn test_array_ir_batching_policy_adapt_batched_program() -> Result<(), ProgramError> {
        // Structurally batch `x ↦ -x` over a leading axis whose extent is the constant 2.
        let trace = ArrayIrTraceContext::new();
        let input = trace.input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)])).into());
        let negated = trace
            .bind(ArrayIrOperation::Array(ArrayOperation::Neg(NegOperation::new())), Vec::new(), &[input])?
            .remove(0);
        let program = trace.builder().borrow().clone().build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            vec![negated.atom_id()?],
            vec![Placeholder],
            vec![Placeholder],
        )?;
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            ArrayIrEagerContext::new(),
            ArrayIrValue::Dimension(DimensionValue::constant(2)?),
        );

        // Adapting drops the leading forwarded-extent bookkeeping output, which exists only for extent-threading
        // consumers, while keeping the leading extent input that defines the batch dimension.
        let batched = <ArrayIrBatchingPolicy as RecursiveBatchingPolicy<ArrayIrEagerContext>>::batch_program(
            &context,
            program.entry_region_ref(),
            &[BatchAxis::new(0)],
            ProgramBatchingOutputAxesPolicy::Natural,
        )?;
        let adapted = <ArrayIrBatchingPolicy as BatchingPolicy<ArrayIrEagerContext>>::adapt_batched_program(
            batched,
            None,
            |_, _, _| panic!("no output needs collapsing"),
        )?;
        assert_eq!(adapted.output_axes(), &[BatchAxis::new(0)]);
        assert_eq!(
            adapted.into_parts().0.to_string(),
            indoc! {"
                lambda %0:dimension<2>, %1:f32[2, 3] .
                let %2:f32[2, 3] = neg %1
                in (%2)
            "}
            .trim_end(),
        );

        // A mapped output whose consumer requires it replicated is collapsed through the caller's function (here by
        // summation), and the reported axes say so.
        let batched = <ArrayIrBatchingPolicy as RecursiveBatchingPolicy<ArrayIrEagerContext>>::batch_program(
            &context,
            program.entry_region_ref(),
            &[BatchAxis::new(0)],
            ProgramBatchingOutputAxesPolicy::Natural,
        )?;
        let adapted = <ArrayIrBatchingPolicy as BatchingPolicy<ArrayIrEagerContext>>::adapt_batched_program(
            batched,
            Some(&[BatchAxis::replicated()]),
            |_, output, axis| {
                let output = ValueProjection::<ArrayType>::into_projected(output)?;
                Ok(output.reduce(&[axis.normalize(2)?], ReductionKind::Sum)?.into_value())
            },
        )?;
        assert_eq!(adapted.output_axes(), &[BatchAxis::replicated()]);
        assert_eq!(
            adapted.into_parts().0.to_string(),
            indoc! {"
                lambda %0:dimension<2>, %1:f32[2, 3] .
                let %2:f32[2, 3] = neg %1
                    %3:f32[3] = reduce [kind=sum, axes=[0]] %2
                in (%3)
            "}
            .trim_end(),
        );
        Ok(())
    }

    #[test]
    fn test_array_ir_batching_policy_project_replicated_batch() {
        // A replicated dimension carrier projects onto its bare dimension member value.
        let dimension =
            DimensionValue::new(DimensionType::new("extent", DimensionBounds::new(0, Some(9)).unwrap()), 4).unwrap();
        let dimension_type = dimension.r#type().into_owned();
        assert_eq!(
            <ArrayIrBatchingPolicy as ReplicatedBatchingPolicyProjection<
                ArrayIrEagerContext,
                DimensionType,
            >>::project_replicated_batch(
                &ArrayIrBatch::replicated(ArrayIrValue::<Array>::Dimension(dimension.clone())),
            ),
            Ok(dimension.clone()),
        );

        // A malformed carrier that maps a dimension value reports the mapped dimension. It is built through the private
        // fields on purpose, since `ArrayIrBatch::new` would already reject it.
        let malformed =
            ArrayIrBatch { value: ArrayIrValue::<Array>::Dimension(dimension), batch_axis: BatchAxis::new(0) };
        assert_eq!(
            <ArrayIrBatchingPolicy as ReplicatedBatchingPolicyProjection<
                ArrayIrEagerContext,
                DimensionType,
            >>::project_replicated_batch(
                &malformed,
            ),
            Err(BatchingError::MappedDimension { r#type: Box::new(dimension_type.clone()), axis: BatchAxis::new(0) }),
        );
    }

    #[test]
    fn test_array_ir_batching_policy_batching_level() -> Result<(), ProgramError> {
        // The level retains the first-class extent's type, never its value.
        let items = DimensionVariable::new("items", DimensionBounds::new(1, Some(9))?);
        let parent = ArrayIrTraceContext::new();
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            parent.clone(),
            parent.input(DimensionType::from(items.clone()).into()),
        )
        .with_axis_name("items".to_string());
        assert_eq!(
            <ArrayIrBatchingPolicy as RecursiveBatchingPolicy<ArrayIrTraceContext>>::batching_level(&context),
            BatchingLevel::new(
                BatchingLevelExtent::Dynamic(DimensionType::from(items).into()),
                Some("items".to_string()),
                ShardingDimension::Replicated,
            ),
        );
        Ok(())
    }

    #[test]
    fn test_array_ir_batching_policy_batch_program_at_level() -> Result<(), ProgramError> {
        // Batching at a recorded level needs no live context and matches batching under the context that produced it.
        let trace = ArrayIrTraceContext::new();
        let input = trace.input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)])).into());
        let doubled = trace
            .bind(
                ArrayIrOperation::Array(ArrayOperation::Add(AddOperation::new())),
                Vec::new(),
                &[input.clone(), input],
            )?
            .remove(0);
        let program = trace.builder().borrow().clone().build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            vec![doubled.atom_id()?],
            vec![Placeholder],
            vec![Placeholder],
        )?;
        let items = DimensionVariable::new("items", DimensionBounds::new(1, Some(9))?);
        let parent = ArrayIrTraceContext::new();
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            parent.clone(),
            parent.input(DimensionType::from(items).into()),
        );
        let level = <ArrayIrBatchingPolicy as RecursiveBatchingPolicy<ArrayIrTraceContext>>::batching_level(&context);
        let batched = <ArrayIrBatchingPolicy as RecursiveBatchingPolicy<ArrayIrTraceContext>>::batch_program_at_level(
            &level,
            program.entry_region_ref(),
            &[BatchAxis::new(0)],
            ProgramBatchingOutputAxesPolicy::Natural,
        )?;
        let expected = <ArrayIrBatchingPolicy as RecursiveBatchingPolicy<ArrayIrTraceContext>>::batch_program(
            &context,
            program.entry_region_ref(),
            &[BatchAxis::new(0)],
            ProgramBatchingOutputAxesPolicy::Natural,
        )?;
        assert_eq!(batched.output_axes(), expected.output_axes());
        assert_eq!(batched.into_parts().0.to_string(), expected.into_parts().0.to_string());

        // A host extent has no first-class representation in this policy.
        let fixed = BatchingLevel::new(BatchingLevelExtent::Static(2), None, ShardingDimension::Replicated);
        assert!(matches!(
            <ArrayIrBatchingPolicy as RecursiveBatchingPolicy<ArrayIrTraceContext>>::batch_program_at_level(
                &fixed,
                program.entry_region_ref(),
                &[BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            ),
            Err(BatchingError::UnsupportedOperation { message })
                if message == "array IR batching requires a first-class batch extent",
        ));
        Ok(())
    }

    #[test]
    fn test_array_ir_batching_policy_batch_program() -> Result<(), ProgramError> {
        // Structural batching gives the transformed program one leading dimension input carrying the extent's type
        // and forwards that same atom as its leading output. Every inserted packed batch axis references this input's
        // type identity, so the batched program stays context-neutral and a consumer supplies the live extent later.
        let trace = ArrayIrTraceContext::new();
        let input = trace.input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)])).into());
        let replicated = trace.input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)])).into());
        let added = trace
            .bind(ArrayIrOperation::Array(ArrayOperation::Add(AddOperation::new())), Vec::new(), &[input, replicated])?
            .remove(0);
        let program = trace.builder().borrow().clone().build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            vec![added.atom_id()?],
            vec![Placeholder, Placeholder],
            vec![Placeholder],
        )?;
        let items = DimensionVariable::new("items", DimensionBounds::new(1, Some(9))?);
        let parent = ArrayIrTraceContext::new();
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            parent.clone(),
            parent.input(DimensionType::from(items).into()),
        );
        let batched = <ArrayIrBatchingPolicy as RecursiveBatchingPolicy<ArrayIrTraceContext>>::batch_program(
            &context,
            program.entry_region_ref(),
            &[BatchAxis::new(0), BatchAxis::replicated()],
            ProgramBatchingOutputAxesPolicy::Natural,
        )?;
        assert_eq!(batched.output_axes(), &[BatchAxis::new(0)]);
        assert_eq!(
            batched.into_parts().0.to_string(),
            indoc! {"
                lambda %0:dimension<items ∈ [1, 9)>, %1:f32[items, 3], %2:f32[3] .
                let %3:dimension<3> = constant [value=3]
                    %4:f32[items, 3] = broadcast [output_axes=[1]] %2 %0 %3
                    %5:f32[items, 3] = add %1 %4
                in (%0, %5)
            "}
            .trim_end(),
        );

        // An output axes policy realigns each output after the replay.
        let batched = <ArrayIrBatchingPolicy as RecursiveBatchingPolicy<ArrayIrTraceContext>>::batch_program(
            &context,
            program.entry_region_ref(),
            &[BatchAxis::new(0), BatchAxis::replicated()],
            ProgramBatchingOutputAxesPolicy::AlignAllTo(Axis::from(1)),
        )?;
        assert_eq!(batched.output_axes(), &[BatchAxis::new(1)]);
        assert_eq!(
            batched.into_parts().0.to_string(),
            indoc! {"
                lambda %0:dimension<items ∈ [1, 9)>, %1:f32[items, 3], %2:f32[3] .
                let %3:dimension<3> = constant [value=3]
                    %4:f32[items, 3] = broadcast [output_axes=[1]] %2 %0 %3
                    %5:f32[items, 3] = add %1 %4
                    %6:f32[3, items] = transpose [permutation=[1, 0]] %5
                in (%0, %6)
            "}
            .trim_end(),
        );

        // The input axes must cover every region input.
        assert_eq!(
            <ArrayIrBatchingPolicy as RecursiveBatchingPolicy<ArrayIrTraceContext>>::batch_program(
                &context,
                program.entry_region_ref(),
                &[BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .err(),
            Some(BatchingError::Program(ProgramError::InvalidInputCount { expected: 2, actual: 1 })),
        );

        Ok(())
    }

    #[test]
    fn test_array_ir_batching_policy_batch_region() -> Result<(), ProgramError> {
        // The array IR replay interprets a region over `ArrayIrBatch` carriers against the parent context: constants
        // are lifted through the checked constructor and replicated, and each instruction runs its batching rule.
        let trace = ArrayIrTraceContext::new();
        let input = trace.input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)])).into());
        let constant = trace.constant(ArrayIrValue::Array(Array::scalar(1.0_f32).unwrap()));
        let negated = trace
            .bind(ArrayIrOperation::Array(ArrayOperation::Neg(NegOperation::new())), Vec::new(), &[input])?
            .remove(0);
        let program = trace.builder().borrow().clone().build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            vec![negated.atom_id()?, constant.atom_id()?],
            vec![Placeholder],
            vec![Placeholder, Placeholder],
        )?;
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            ArrayIrEagerContext::new(),
            ArrayIrValue::Dimension(DimensionValue::constant(2)?),
        );
        let outputs = <ArrayIrBatchingPolicy as RecursiveBatchingPolicy<ArrayIrEagerContext>>::batch_region(
            &context,
            program.entry_region_ref(),
            vec![ArrayIrBatch::new(
                ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap()),
                BatchAxis::new(0),
            )?],
        )?;
        assert_eq!(
            outputs,
            vec![
                ArrayIrBatch::new(
                    ArrayIrValue::Array(Array::matrix(2, 3, vec![-1.0_f32, -2.0, -3.0, -4.0, -5.0, -6.0]).unwrap()),
                    BatchAxis::new(0),
                )?,
                ArrayIrBatch::replicated(ArrayIrValue::Array(Array::scalar(1.0_f32).unwrap())),
            ],
        );
        Ok(())
    }

    #[test]
    fn test_array_ir_batching_policy_align_batch_axis() -> Result<(), ProgramError> {
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            ArrayIrEagerContext::new(),
            ArrayIrValue::Dimension(DimensionValue::constant(2)?),
        );

        // Arrays align like under the homogeneous policy: a replicated array is broadcast along the context's extent to
        // gain the mapped axis, and a mapped array moves its axis to the requested position.
        let replicated = ArrayIrBatch::replicated(ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]).unwrap()));
        let broadcasted = ArrayIrBatchingPolicy::align_batch_axis(&context, replicated, Axis::from(0))?;
        assert_eq!(broadcasted.batch_axis(), BatchAxis::new(0));
        assert_eq!(
            broadcasted.value(),
            &ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0_f32, 2.0, 3.0, 1.0, 2.0, 3.0]).unwrap()),
        );
        let mapped = ArrayIrBatch::new(
            ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap()),
            BatchAxis::new(0),
        )?;
        let moved = ArrayIrBatchingPolicy::align_batch_axis(&context, mapped, Axis::from(1))?;
        assert_eq!(moved.batch_axis(), BatchAxis::new(1));
        assert_eq!(
            moved.value(),
            &ArrayIrValue::Array(Array::matrix(3, 2, vec![1.0_f32, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap())
        );

        // Placing the batch axis on a manual mesh axis distributes the batch items across that axis's shards, so the
        // broadcast result varies along it; a non-manual placement records no variation.
        for axis_type in [MeshAxisType::Explicit, MeshAxisType::Manual] {
            let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, axis_type).unwrap()]).unwrap();
            let replicated_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]))
                .with_sharding(Sharding::replicated(mesh.clone(), 1))
                .unwrap();
            let replicated = ArrayIrBatch::replicated(ArrayIrValue::Array(
                Array::from_elements::<f64>(replicated_type, &[1.0, 2.0, 3.0]).unwrap(),
            ));
            let sharded_context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
                ArrayIrEagerContext::new(),
                ArrayIrValue::Dimension(DimensionValue::constant(2)?),
            )
            .with_axis_sharding(ShardingDimension::sharded(["x"]));
            let aligned = ArrayIrBatchingPolicy::align_batch_axis(&sharded_context, replicated, Axis::from(0))?;
            let expected_sharding =
                Sharding::new(mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                    .unwrap()
                    .with_varying_manual_axes((axis_type == MeshAxisType::Manual).then_some("x"))
                    .unwrap();
            assert_eq!(aligned.batch_axis(), BatchAxis::new(0));
            let ArrayIrType::Array(packed_type) = aligned.value().r#type().into_owned() else {
                panic!("expected an array carrier");
            };
            assert_eq!(packed_type.sharding(), Some(&expected_sharding));
        }

        // A reference's batch axis is fixed by its referent: a request for the axis it already carries is an identity
        // alignment, and every other request is rejected instead of moving or broadcasting the shared storage.
        let reference = ArrayIrValue::Reference(ArrayReference::new(
            Array::matrix(2, 3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        ));
        let batch = ArrayIrBatch::new(reference.clone(), BatchAxis::new(0))?;
        let aligned = ArrayIrBatchingPolicy::align_batch_axis(&context, batch.clone(), Axis::from(0))?;
        assert_eq!(aligned.batch_axis(), BatchAxis::new(0));
        assert_eq!(aligned.value(), &reference);
        assert_eq!(
            ArrayIrBatchingPolicy::align_batch_axis(&context, batch.clone(), Axis::from(2)),
            Err(BatchingError::BatchAxisOutOfBounds {
                r#type: Box::new(ArrayType::new_static(DataType::F32, [2, 3])),
                axis: Axis::from(2),
            }),
        );
        assert_eq!(
            ArrayIrBatchingPolicy::align_batch_axis(&context, batch, Axis::from(1)),
            Err(BatchingError::UnsupportedOperation {
                message: "a reference's batch axis is fixed by its referent, so a reference batch carrying axis 0 \
                          cannot be aligned to axis 1; pass the reference as a batched input at that axis instead"
                    .to_string(),
            }),
        );
        assert_eq!(
            ArrayIrBatchingPolicy::align_batch_axis(&context, ArrayIrBatch::replicated(reference), Axis::from(0)),
            Err(BatchingError::UnsupportedOperation {
                message: "a reference's batch axis is fixed by its referent, so a reference batch carrying replicated \
                          cannot be aligned to axis 0; pass the reference as a batched input at that axis instead"
                    .to_string(),
            }),
        );

        // A first-class dimension has no packed axis to align, because dimensions are shared shape values.
        // Eager dimension values are typed exactly, so diagnostics report the value's own exact type.
        let dimension_value = DimensionValue::new(DimensionType::new("n", DimensionBounds::new(1, Some(8))?), 3)?;
        let dimension_type = dimension_value.r#type().into_owned();
        let dimension = ArrayIrValue::<Array>::Dimension(dimension_value);
        assert_eq!(
            ArrayIrBatchingPolicy::align_batch_axis(&context, ArrayIrBatch::replicated(dimension), Axis::from(0)),
            Err(BatchingError::MappedDimension { r#type: Box::new(dimension_type.clone()), axis: BatchAxis::new(0) }),
        );
        Ok(())
    }

    #[test]
    fn test_array_ir_batching_policy_static_batch_axis_extent() -> Result<(), ProgramError> {
        // Exact types and resolved dimension constants establish a static size; a symbolic extent remains dynamic.
        let eager = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            ArrayIrEagerContext::new(),
            ArrayIrValue::Dimension(DimensionValue::constant(2)?),
        );
        assert_eq!(ArrayIrBatchingPolicy::static_batch_axis_extent(&eager), Some(2));

        let trace = ArrayIrTraceContext::new();
        let constant = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            trace.clone(),
            trace.constant(ArrayIrValue::Dimension(DimensionValue::constant(3)?)),
        );
        assert_eq!(ArrayIrBatchingPolicy::static_batch_axis_extent(&constant), Some(3));
        let instruction = BatchingContext::<_, ArrayIrBatchingPolicy>::new(trace.clone(), trace.dimension_constant(4)?);
        assert_eq!(ArrayIrBatchingPolicy::static_batch_axis_extent(&instruction), Some(4));
        let exact = DimensionVariable::new("exact", DimensionBounds::new(5, Some(6))?);
        let exact_input = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            trace.clone(),
            trace.input(DimensionType::from(exact).into()),
        );
        assert_eq!(ArrayIrBatchingPolicy::static_batch_axis_extent(&exact_input), Some(5));
        let items = DimensionVariable::new("items", DimensionBounds::new(1, Some(9))?);
        let staged = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            trace.clone(),
            trace.input(DimensionType::from(items).into()),
        );
        assert_eq!(ArrayIrBatchingPolicy::static_batch_axis_extent(&staged), None);
        Ok(())
    }

    #[test]
    fn test_array_ir_batching_policy_diagonal() -> Result<(), ProgramError> {
        // Dynamic extents are read from the value, so one staged diagonal serves every extent, and static extents
        // consume no dimension values.
        let extent = DimensionVariable::new("n", DimensionBounds::new(1, Some(8)).unwrap());
        let dynamic = Dimension::Dynamic(extent);
        let r#type = ArrayType::new(DataType::F32, Shape::new(vec![dynamic.clone(), Dimension::Static(2), dynamic]));
        let (_, program) = DomainTracingContext::<ArrayIrEagerContext>::trace(
            |value| {
                Ok(<ArrayIrBatchingPolicy as DiagonalBatchingPolicy<ArrayIrTraceContext>>::diagonal(
                    value.context(),
                    &value,
                    2,
                    0,
                )?)
            },
            ArrayIrType::from(r#type),
        )?;
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[n, 2, n] .
                let %1:dimension<n ∈ [1, 8)> = dimension_size [axis=2] %0
                    %2:i64[2, n, 1] = iota [type=i64[2, n, 1], dimension=1] %1
                    %3:f32[2, n] = gather [
                        dimensions=(offset=[], collapsed_slice=[0], start_index_map=[0], batching=[(1, 0), (2, 1)]),
                        slice_sizes=[1, 1, 1],
                    ] %0 %2
                in (%3)
            "}
            .trim_end(),
        );
        let values = (0..18).map(|value| value as f32).collect::<Vec<_>>();
        let cube = Array::from_elements(ArrayType::new_static(DataType::F32, [3, 2, 3]), &values)?;
        assert_eq!(
            program.interpret(ArrayIrValue::Array(cube)),
            Ok(ArrayIrValue::Array(Array::matrix(2, 3, vec![0.0f32, 7.0, 14.0, 3.0, 10.0, 17.0])?)),
        );

        // Mismatched extents are rejected before anything is staged.
        assert_eq!(
            <ArrayIrBatchingPolicy as DiagonalBatchingPolicy<ArrayIrEagerContext>>::diagonal(
                &ArrayIrEagerContext::new(),
                &ArrayIrValue::Array(Array::matrix(2, 3, vec![0.0f32; 6])?),
                0,
                1,
            ),
            Err(TypeError::invalid(
                "cannot take the diagonal of a value of type `f32[2, 3]` along axes 0 and 1, whose extents differ",
            )
            .into()),
        );
        Ok(())
    }

    #[test]
    fn test_array_ir_batching_policy_pack_inputs() -> Result<(), BatchingError> {
        let pack_inputs = <ArrayIrBatchingPolicy as BatchingEntrypointPolicy<ArrayIrTraceContext>>::pack_inputs;

        // Each input is wrapped with its declared axis, and the extent is read from the first mapped array's packed
        // batch axis as a first-class `dimension_size` value, so a symbolic extent stays symbolic. Replicated inputs
        // and first-class dimensions pass through, and the context records the extent, the axis name, and the batch
        // axis placement joined over the mapped inputs.
        let items = DimensionVariable::new("items", DimensionBounds::new(1, Some(9)).unwrap());
        let trace = ArrayIrTraceContext::new();
        let mapped = trace.input(
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(items.clone()), Dimension::Static(3)]))
                .into(),
        );
        let replicated = trace.input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)])).into());
        let dimension = trace.input(DimensionValue::constant(4).unwrap().r#type().into_owned().into());
        let (context, batches) = pack_inputs(
            &trace,
            vec![mapped.clone(), replicated.clone(), dimension.clone()],
            vec![BatchAxis::new(-2), BatchAxis::replicated(), BatchAxis::replicated()],
            BatchAxisSpecification::named("items"),
        )?;
        assert_eq!(
            context.axis_extent().r#type().into_owned(),
            ArrayIrType::Dimension(DimensionType::from(items.clone()))
        );
        assert_eq!(context.axis_name(), Some("items"));
        assert_eq!(context.axis_sharding(), &ShardingDimension::replicated());
        assert_eq!(
            batches,
            vec![
                ArrayIrBatch::new(mapped.clone(), BatchAxis::new(0))?,
                ArrayIrBatch::replicated(replicated),
                ArrayIrBatch::replicated(dimension),
            ],
        );
        let program = trace.builder().borrow().clone().build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            vec![context.axis_extent().atom_id()?],
            vec![Placeholder, Placeholder, Placeholder],
            vec![Placeholder],
        )?;
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[items, 3], %1:f32[3], %2:dimension<4> .
                let %3:dimension<items ∈ [1, 9)> = dimension_size [axis=0] %0
                in (%3)
            "}
            .trim_end(),
        );

        // The flat axis declaration must match the inputs, an explicit extent must be a first-class dimension, and
        // without an explicit extent or a mapped input nothing determines the batch size. A first-class dimension
        // cannot itself be declared mapped.
        assert_eq!(
            pack_inputs(&trace, vec![mapped.clone()], vec![BatchAxis::new(0); 2], BatchAxisSpecification::default())
                .err(),
            Some(BatchingError::Program(ProgramError::InvalidInputCount { expected: 1, actual: 2 })),
        );
        assert_eq!(
            pack_inputs(&trace, Vec::new(), Vec::new(), BatchAxisSpecification::with_extent(mapped.clone())).err(),
            Some(BatchingError::Type(TypeError::Invalid {
                message: "expected dimension type but got array type".to_string(),
            })),
        );
        assert_eq!(
            pack_inputs(&trace, Vec::new(), Vec::new(), BatchAxisSpecification::default()).err(),
            Some(BatchingError::EmptyBatch),
        );
        let mapped_dimension = trace.constant(ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()));
        let mapped_dimension_type = <&DimensionType>::try_from(mapped_dimension.r#type().as_ref()).unwrap().clone();
        assert_eq!(
            pack_inputs(&trace, vec![mapped_dimension], vec![BatchAxis::new(0)], BatchAxisSpecification::default())
                .err(),
            Some(BatchingError::MappedDimension { r#type: Box::new(mapped_dimension_type), axis: BatchAxis::new(0) }),
        );

        // An explicit extent is the reference every mapped input's extent is checked against. Under eager batching the
        // requirement folds against the concrete extents and a mismatch is reported immediately.
        let eager = ArrayIrEagerContext::new();
        let matrix = ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap());
        let extent = ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap());
        let (context, _) = <ArrayIrBatchingPolicy as BatchingEntrypointPolicy<ArrayIrEagerContext>>::pack_inputs(
            &eager,
            vec![matrix.clone()],
            vec![BatchAxis::new(0)],
            BatchAxisSpecification::with_extent(extent.clone()),
        )?;
        assert_eq!(context.axis_extent(), &extent);
        let mismatched = <ArrayIrBatchingPolicy as BatchingEntrypointPolicy<ArrayIrEagerContext>>::pack_inputs(
            &eager,
            vec![matrix],
            vec![BatchAxis::new(0)],
            BatchAxisSpecification::with_extent(ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap())),
        )
        .unwrap_err();
        assert!(matches!(mismatched, BatchingError::Program(ProgramError::Custom(_))));
        assert_eq!(mismatched.to_string(), "assertion failed: batch dimensions must agree; expected=3, actual=2",);

        // A mapped reference whose referent has a dynamic extent at the batch axis cannot supply the mapped extent
        // itself, so a mapped array must expose that same dimension identity regardless of input order. Arrays are
        // visited before references so that the evidence is complete either way (the traced inputs are created in one
        // order, so the rendering is the same for both input orders).
        let dynamic = Dimension::Dynamic(DimensionVariable::new("n", DimensionBounds::unbounded()));
        let reference_type =
            ReferenceType::new(ArrayType::new(DataType::F32, Shape::new(vec![dynamic.clone(), Dimension::Static(3)])));
        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![dynamic, Dimension::Static(3)]));
        for reference_first in [true, false] {
            let trace = ArrayIrTraceContext::new();
            let reference = trace.input(reference_type.clone().into());
            let array = trace.input(array_type.clone().into());
            let inputs = if reference_first { vec![reference, array] } else { vec![array, reference] };
            let (context, batches) =
                pack_inputs(&trace, inputs, vec![BatchAxis::new(0); 2], BatchAxisSpecification::default())?;
            assert_eq!(context.axis_extent().r#type().to_string(), "dimension<n ∈ [0, ∞)>");
            assert_eq!(batches.iter().map(ArrayIrBatch::batch_axis).collect::<Vec<_>>(), vec![BatchAxis::new(0); 2]);
            let program =
                trace.builder().borrow().clone().build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                    vec![context.axis_extent().atom_id()?],
                    vec![Placeholder, Placeholder],
                    vec![Placeholder],
                )?;
            assert_eq!(
                program.to_string(),
                indoc! {"
                    lambda %0:ref<f32[n, 3]>, %1:f32[n, 3] .
                    let %2:dimension<n ∈ [0, ∞)> = dimension_size [axis=0] %1
                    in (%2)
                "}
                .trim_end(),
            );
        }

        // Without a source with the same dimension identity the extent is unknown: an unrelated mapped array or an
        // unrelated explicit extent is not evidence, because accepting it would silently give different inputs
        // different batch lengths. An explicit dimension with that identity supplies the evidence without reading the
        // reference.
        let trace = ArrayIrTraceContext::new();
        let reference = trace.input(reference_type.clone().into());
        let unrelated_array = trace.input(ArrayType::new_static(DataType::F32, [2, 3]).into());
        let expected = BatchingError::UnsupportedOperation {
            message: format!(
                "cannot infer the mapped extent from reference type {reference_type} whose batch axis 0 has a dynamic \
                 extent; pass an extent or a mapped array with the same dimension identity",
            ),
        };
        assert_eq!(
            pack_inputs(&trace, vec![reference.clone()], vec![BatchAxis::new(0)], BatchAxisSpecification::default())
                .err(),
            Some(expected.clone()),
        );
        assert_eq!(
            pack_inputs(
                &trace,
                vec![reference.clone(), unrelated_array],
                vec![BatchAxis::new(0); 2],
                BatchAxisSpecification::default(),
            )
            .err(),
            Some(expected.clone()),
        );
        assert_eq!(
            pack_inputs(
                &trace,
                vec![reference.clone()],
                vec![BatchAxis::new(0)],
                BatchAxisSpecification::with_extent(
                    trace.constant(ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap())),
                ),
            )
            .err(),
            Some(expected),
        );
        let variable = reference_type.referent().shape().dimensions()[0].variable().unwrap().clone();
        let extent = trace.input(DimensionType::from(variable).into());
        let (context, _) = pack_inputs(
            &trace,
            vec![reference],
            vec![BatchAxis::new(0)],
            BatchAxisSpecification::with_extent(extent.clone()),
        )?;
        assert_eq!(context.axis_extent(), &extent);

        // A referent whose dynamic batch dimension has exact bounds supplies its extent as a constant, and that extent
        // is then required to match every other mapped input. Tracing retains a known mismatch for execution.
        let trace = ArrayIrTraceContext::new();
        let exact = DimensionVariable::new("n", DimensionBounds::new(4, Some(5)).unwrap());
        let exact_reference_type = ReferenceType::new(ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Dynamic(exact), Dimension::Static(3)]),
        ));
        let reference = trace.input(exact_reference_type.into());
        let (context, _) =
            pack_inputs(&trace, vec![reference.clone()], vec![BatchAxis::new(0)], BatchAxisSpecification::default())?;
        assert_eq!(context.axis_extent().r#type().to_string(), "dimension<4>");
        let array = trace.input(ArrayType::new_static(DataType::F32, [2, 3]).into());
        pack_inputs(&trace, vec![reference, array], vec![BatchAxis::new(0); 2], BatchAxisSpecification::default())?;
        let program = trace.builder().borrow().clone().build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            Vec::new(),
            vec![Placeholder; 2],
            Vec::new(),
        )?;
        assert!(program.instructions().iter().any(|instruction| {
            matches!(instruction.operation(), ArrayIrOperation::Assert(operation)
                if operation.message() == "batch dimensions must agree")
        }));
        let mismatched = program
            .interpret(vec![
                ArrayIrValue::Reference(ArrayReference::new(Array::matrix(4, 3, vec![0.0_f32; 12]).unwrap())),
                ArrayIrValue::Array(Array::matrix(2, 3, vec![0.0_f32; 6]).unwrap()),
            ])
            .unwrap_err();
        assert!(matches!(mismatched, ProgramError::Custom(_)));
        assert_eq!(mismatched.to_string(), "assertion failed: batch dimensions must agree; expected=2, actual=4");

        // A mapped reference contributes its referent's batch-axis placement to the sharding join, so a mapped array
        // whose placement is only replicated is renormalized onto the reference's placement. Both mapped extents fold
        // to equal exact constants, so their comparison and assertion fold away, and the renormalizing broadcast
        // takes the transform's extent at the mapped axis while folding the statically known axis into an exact
        // constant rather than a `dimension_size` read.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let shape = Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]);
        let sharded_type = ArrayType::new(DataType::F32, shape.clone())
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                    .unwrap(),
            )
            .unwrap();
        let replicated_type =
            ArrayType::new(DataType::F32, shape).with_sharding(Sharding::replicated(mesh, 2)).unwrap();
        let trace = ArrayIrTraceContext::new();
        let reference = trace.input(ReferenceType::new(sharded_type.clone()).into());
        let array = trace.input(replicated_type.clone().into());
        let (context, batches) =
            pack_inputs(&trace, vec![reference, array], vec![BatchAxis::new(0); 2], BatchAxisSpecification::default())?;
        assert_eq!(context.axis_sharding(), &ShardingDimension::sharded(["x"]));
        assert_eq!(
            batches[0].value().r#type().as_ref(),
            &ArrayIrType::Reference(ReferenceType::new(sharded_type.clone())),
        );
        assert_eq!(batches[1].value().r#type().as_ref(), &ArrayIrType::Array(sharded_type.clone()));
        let program = trace.builder().borrow().clone().build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            vec![batches[1].value().atom_id()?],
            vec![Placeholder, Placeholder],
            vec![Placeholder],
        )?;
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:ref<f32[2, 3][sharding={mesh<['x'=2:explicit]>, [{'x'}, {}]}]>, \
                    %1:f32[2, 3][sharding={mesh<['x'=2:explicit]>, [{}, {}]}] .
                let %2:dimension<2> = constant [value=2]
                    %3:dimension<2> = constant [value=2]
                    %4:bool[] = const true
                    %5:dimension<3> = constant [value=3]
                    %6:f32[2, 3][sharding={mesh<['x'=2:explicit]>, [{'x'}, {}]}] = \
                        broadcast [output_axes=[0, 1], output_sharding={mesh<['x'=2:explicit]>, [{'x'}, {}]}] %1 %2 %5
                in (%6)
            "}
            .trim_end(),
        );

        // A referent placement that disagrees with the selected batch-axis placement cannot be renormalized, because
        // the reference's storage is shared, so the mismatch is rejected instead of being silently ignored.
        let trace = ArrayIrTraceContext::new();
        let array = trace.input(sharded_type.into());
        let reference = trace.input(ReferenceType::new(replicated_type.clone()).into());
        assert_eq!(
            pack_inputs(&trace, vec![array, reference], vec![BatchAxis::new(0); 2], BatchAxisSpecification::default())
                .err(),
            Some(BatchingError::MisalignedBatchAxes {
                message: format!(
                    "mapped reference input 1 of type {} does not place its batch axis 0 on {}; a reference's \
                     referent placement is fixed by its storage and cannot be renormalized",
                    ReferenceType::new(replicated_type),
                    ShardingDimension::sharded(["x"]),
                ),
            }),
        );
        Ok(())
    }

    #[test]
    fn test_array_ir_batching_policy_materialize_output() -> Result<(), ProgramError> {
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            ArrayIrEagerContext::new(),
            ArrayIrValue::Dimension(DimensionValue::constant(2)?),
        );

        // An array output is aligned to the requested axis: an existing mapped axis is moved, a replicated output is
        // broadcast along the transform's extent, and a mapped output cannot be presented as replicated because
        // dropping the axis would need a reduction the caller did not ask for.
        let matrix = ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap());
        let rows = ArrayIrBatch::new(matrix.clone(), BatchAxis::new(0))?;
        assert_eq!(ArrayIrBatchingPolicy::materialize_output(&context, rows.clone(), BatchAxis::new(0)), Ok(matrix));
        assert_eq!(
            ArrayIrBatchingPolicy::materialize_output(&context, rows.clone(), BatchAxis::new(1)),
            Ok(ArrayIrValue::Array(Array::matrix(3, 2, vec![1.0_f32, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap())),
        );
        assert_eq!(
            ArrayIrBatchingPolicy::materialize_output(
                &context,
                ArrayIrBatch::replicated(ArrayIrValue::Array(Array::vector(vec![10.0_f32, 20.0, 30.0]).unwrap())),
                BatchAxis::new(0),
            ),
            Ok(ArrayIrValue::Array(Array::matrix(2, 3, vec![10.0_f32, 20.0, 30.0, 10.0, 20.0, 30.0]).unwrap())),
        );
        let replicated = ArrayIrValue::Array(Array::vector(vec![10.0_f32, 20.0, 30.0]).unwrap());
        assert_eq!(
            ArrayIrBatchingPolicy::materialize_output(
                &context,
                ArrayIrBatch::replicated(replicated.clone()),
                BatchAxis::replicated(),
            ),
            Ok(replicated),
        );
        assert_eq!(
            ArrayIrBatchingPolicy::materialize_output(&context, rows.clone(), BatchAxis::replicated()),
            Err(BatchingError::MismatchedOutputAxes { expected: BatchAxis::replicated(), actual: BatchAxis::new(0) }),
        );

        // First-class dimensions leave the transform unchanged as replicated shape values.
        let dimension = ArrayIrValue::<Array>::Dimension(DimensionValue::constant(3)?);
        assert_eq!(
            ArrayIrBatchingPolicy::materialize_output(
                &context,
                ArrayIrBatch::replicated(dimension.clone()),
                BatchAxis::replicated(),
            ),
            Ok(dimension),
        );
        // A reference output is returned as-is when the request agrees with the axis its referent fixes: a replicated
        // request for a replicated reference, or a mapped request naming the reference's own axis (in any signed
        // spelling that normalizes to it).
        let replicated = ArrayIrValue::Reference(ArrayReference::new(Array::scalar(1.0_f32).unwrap()));
        assert_eq!(
            ArrayIrBatchingPolicy::materialize_output(
                &context,
                ArrayIrBatch::replicated(replicated.clone()),
                BatchAxis::replicated()
            ),
            Ok(replicated.clone()),
        );
        let mapped = ArrayIrValue::Reference(ArrayReference::new(
            Array::matrix(2, 3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        ));
        let batch = ArrayIrBatch::new(mapped.clone(), BatchAxis::new(0))?;
        assert_eq!(
            ArrayIrBatchingPolicy::materialize_output(&context, batch.clone(), BatchAxis::new(0)),
            Ok(mapped.clone())
        );
        assert_eq!(ArrayIrBatchingPolicy::materialize_output(&context, batch.clone(), BatchAxis::new(-2)), Ok(mapped));

        // Any other request would have to move or broadcast shared storage, which no reference can do.
        assert_eq!(
            ArrayIrBatchingPolicy::materialize_output(&context, batch.clone(), BatchAxis::new(1)),
            Err(BatchingError::UnsupportedOperation {
                message: "a reference output carrying axis 0 cannot be materialized with axis 1 requested because a \
                          reference's batch axis is fixed by its referent"
                    .to_string(),
            }),
        );
        assert_eq!(
            ArrayIrBatchingPolicy::materialize_output(&context, batch, BatchAxis::replicated()),
            Err(BatchingError::UnsupportedOperation {
                message: "a reference output carrying axis 0 cannot be materialized with replicated requested because \
                          a reference's batch axis is fixed by its referent"
                    .to_string(),
            }),
        );
        assert_eq!(
            ArrayIrBatchingPolicy::materialize_output(
                &context,
                ArrayIrBatch::replicated(replicated),
                BatchAxis::new(0)
            ),
            Err(BatchingError::UnsupportedOperation {
                message: "a reference output carrying replicated cannot be materialized with axis 0 requested because \
                          a reference's batch axis is fixed by its referent"
                    .to_string(),
            }),
        );
        Ok(())
    }

    #[test]
    fn test_array_ir_batching_policy_bind_dispatch() -> Result<(), ProgramError> {
        // Binding through an array IR batching context dispatches each operation to its batching rule with the
        // `ArrayIrBatch` carriers unpacked and repackaged around it, without projecting their member kind. A
        // homogeneous array rule runs unchanged, and first-class dimensions flow through replicated.
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            ArrayIrEagerContext::new(),
            ArrayIrValue::Dimension(DimensionValue::constant(2)?),
        );
        let input = ArrayIrBatch::new(
            ArrayIrValue::Array(Array::matrix(2, 2, vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap()),
            BatchAxis::new(0),
        )?;
        let [output] = ArrayIrOperation::Array(ArrayOperation::Neg(NegOperation::new()))
            .interpret_with_batch_axes(&context, std::slice::from_ref(&input), &[BatchAxis::new(0)])?
            .try_into()
            .unwrap();
        assert_eq!(output.batch_axis(), BatchAxis::new(0));
        assert_eq!(
            output.value(),
            &ArrayIrValue::Array(Array::matrix(2, 2, vec![-1.0_f32, -2.0, -3.0, -4.0]).unwrap())
        );
        let [output] = context
            .bind(
                ArrayIrOperation::Array(ArrayOperation::Neg(NegOperation::new())),
                Vec::new(),
                &[BatchingTracer::new(context.clone(), input)],
            )?
            .try_into()
            .unwrap();
        assert_eq!(output.batch().batch_axis(), BatchAxis::new(0));
        assert_eq!(
            output.batch().value(),
            &ArrayIrValue::Array(Array::matrix(2, 2, vec![-1.0_f32, -2.0, -3.0, -4.0]).unwrap())
        );

        let left = ArrayIrValue::Dimension(DimensionValue::constant(3)?);
        let right = ArrayIrValue::Dimension(DimensionValue::constant(4)?);
        let operation = DimensionAddOperation::new(
            <&DimensionType>::try_from(left.r#type().as_ref())?,
            <&DimensionType>::try_from(right.r#type().as_ref())?,
        )?;
        let inputs = [
            BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(left)),
            BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(right)),
        ];
        let [dimension] = context
            .bind(ArrayIrOperation::<Array>::Dimension(DimensionOperation::Add(operation)), Vec::new(), &inputs)?
            .try_into()
            .unwrap();
        assert_eq!(
            dimension.to_scalar()?.into_batch(),
            ArrayIrBatch::replicated(ArrayIrValue::Array(Array::scalar(7_i64).unwrap())),
        );

        // Under a trace with a symbolic extent, a lifted mixed operation keeps the mapped extent as an ordinary SSA
        // input, and generic homogeneous dispatch stages the alignment of its inputs explicitly: a replicated input
        // gains the mapped axis through a dynamic broadcast that consumes the extent, and differently positioned mapped
        // inputs are reconciled by one transpose.
        let trace = ArrayIrTraceContext::new();
        let batch_variable = DimensionVariable::new("batch", DimensionBounds::new(1, Some(9))?);
        let batch_extent = trace.input(DimensionType::from(batch_variable.clone()).into());
        let mapped = trace.input(
            ArrayType::new(
                DataType::F32,
                Shape::new(vec![Dimension::Dynamic(batch_variable.clone()), Dimension::Static(3)]),
            )
            .into(),
        );
        let transposed = trace.input(
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3), Dimension::Dynamic(batch_variable)]))
                .into(),
        );
        let replicated = trace.input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)])).into());
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(trace.clone(), batch_extent)
            .with_axis_name("items".to_string());
        assert_eq!(context.named_axis("items"), Some(NamedAxis::Batched { size: None }));
        let output_extent = trace.constant(ArrayIrValue::Dimension(DimensionValue::constant(3)?));
        let [reshaped] = context
            .bind(
                ArrayIrOperation::<Array>::from(DynamicReshapeOperation::new()),
                Vec::new(),
                &[
                    BatchingTracer::new(context.clone(), ArrayIrBatch::new(mapped.clone(), BatchAxis::new(0))?),
                    BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(output_extent)),
                ],
            )?
            .try_into()
            .unwrap();
        assert_eq!(reshaped.batch().batch_axis(), BatchAxis::new(0));
        let [added] = context
            .bind(
                ArrayIrOperation::Array(ArrayOperation::Add(AddOperation::new())),
                Vec::new(),
                &[
                    BatchingTracer::new(context.clone(), ArrayIrBatch::new(mapped.clone(), BatchAxis::new(0))?),
                    BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(replicated)),
                ],
            )?
            .try_into()
            .unwrap();
        assert_eq!(added.batch().batch_axis(), BatchAxis::new(0));
        assert_eq!(
            added.r#type().as_ref(),
            &ArrayIrType::Array(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)]))),
        );
        let [reconciled] = context
            .bind(
                ArrayIrOperation::Array(ArrayOperation::Add(AddOperation::new())),
                Vec::new(),
                &[
                    BatchingTracer::new(context.clone(), ArrayIrBatch::new(mapped, BatchAxis::new(0))?),
                    BatchingTracer::new(context.clone(), ArrayIrBatch::new(transposed, BatchAxis::new(1))?),
                ],
            )?
            .try_into()
            .unwrap();
        assert_eq!(reconciled.batch().batch_axis(), BatchAxis::new(0));
        let program = trace.builder().borrow().clone().build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            vec![
                reshaped.into_batch().into_value().atom_id()?,
                added.into_batch().into_value().atom_id()?,
                reconciled.into_batch().into_value().atom_id()?,
            ],
            vec![Placeholder, Placeholder, Placeholder, Placeholder],
            vec![Placeholder, Placeholder, Placeholder],
        )?;
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:dimension<batch ∈ [1, 9)>, %1:f32[batch, 3], %2:f32[3, batch], %3:f32[3] .
                let %4:dimension<3> = const 3
                    %5:f32[batch, 3] = reshape [requires_runtime_assertion=false] %1 %0 %4
                    %6:dimension<3> = constant [value=3]
                    %7:f32[batch, 3] = broadcast [output_axes=[1]] %3 %0 %6
                    %8:f32[batch, 3] = add %1 %7
                    %9:f32[batch, 3] = transpose [permutation=[1, 0]] %2
                    %10:f32[batch, 3] = add %1 %9
                in (%5, %8, %10)
            "}
            .trim_end(),
        );
        Ok(())
    }

    #[test]
    fn test_array_ir_batching_policy_bind_dispatch_replicated_dimension_operations() -> Result<(), ProgramError> {
        // Homogeneous dimension operations have no batching rules of their own. The array IR dispatcher binds each
        // one once in the parent context under the replicated dimension policy, so a zero-input constant produces a
        // replicated dimension, and checked arithmetic keeps its single ordered assertion and its runtime failure
        // rather than running once per batch item.
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            ArrayIrEagerContext::new(),
            ArrayIrValue::Dimension(DimensionValue::constant(2)?),
        );
        let [constant] = context
            .bind(
                ArrayIrOperation::<Array>::Dimension(DimensionOperation::Constant(ConstantOperation::new(
                    DimensionValue::constant(5)?,
                ))),
                Vec::new(),
                &[],
            )?
            .try_into()
            .unwrap();
        assert_eq!(constant.batch().batch_axis(), BatchAxis::replicated());
        assert!(matches!(constant.batch().value(), ArrayIrValue::Dimension(value) if value.extent() == 5));

        // Checked division by a divisor that may be zero succeeds for a nonzero divisor and preserves the underlying
        // failure for a zero divisor.
        let dividend = DimensionValue::constant(7)?;
        let maybe_zero = DimensionType::new("maybe_zero", DimensionBounds::new(0, Some(5))?);
        let operation = ArrayIrOperation::<Array>::Dimension(DimensionOperation::Div(DimensionDivOperation::new(
            dividend.r#type().as_ref(),
            &maybe_zero,
        )?));
        let [quotient] = context
            .bind(
                operation.clone(),
                Vec::new(),
                &[
                    BatchingTracer::new(
                        context.clone(),
                        ArrayIrBatch::replicated(ArrayIrValue::Dimension(dividend.clone())),
                    ),
                    BatchingTracer::new(
                        context.clone(),
                        ArrayIrBatch::replicated(ArrayIrValue::Dimension(DimensionValue::new(maybe_zero.clone(), 2)?)),
                    ),
                ],
            )?
            .try_into()
            .unwrap();
        assert_eq!(quotient.batch().batch_axis(), BatchAxis::replicated());
        assert!(matches!(quotient.batch().value(), ArrayIrValue::Dimension(value) if value.extent() == 3));
        let error = context
            .bind(
                operation,
                Vec::new(),
                &[
                    BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(ArrayIrValue::Dimension(dividend))),
                    BatchingTracer::new(
                        context.clone(),
                        ArrayIrBatch::replicated(ArrayIrValue::Dimension(DimensionValue::new(maybe_zero.clone(), 0)?)),
                    ),
                ],
            )
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<DimensionError>(),
            Some(&DimensionError::RequirementViolation {
                message: "`dimension_div` requires right > 0; got left=7, right=0".to_string(),
            }),
        );

        // Under a trace with a symbolic extent, the checked division is staged exactly once in the parent program,
        // with its replicated inputs and its ordered assertion intact, and no extent-dependent alignment.
        let trace = ArrayIrTraceContext::new();
        let batch_variable = DimensionVariable::new("batch", DimensionBounds::new(1, Some(9))?);
        let batch_extent = trace.input(DimensionType::from(batch_variable).into());
        let dividend_type = DimensionType::new("dividend", DimensionBounds::new(0, Some(9))?);
        let dividend = trace.input(dividend_type.clone().into());
        let divisor = trace.input(maybe_zero.clone().into());
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(trace.clone(), batch_extent);
        let [quotient] = context
            .bind(
                ArrayIrOperation::<Array>::Dimension(DimensionOperation::Div(DimensionDivOperation::new(
                    &dividend_type,
                    &maybe_zero,
                )?)),
                Vec::new(),
                &[
                    BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(dividend)),
                    BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(divisor)),
                ],
            )?
            .try_into()
            .unwrap();
        assert_eq!(quotient.batch().batch_axis(), BatchAxis::replicated());
        let program = trace.builder().borrow().clone().build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            vec![quotient.into_batch().into_value().atom_id()?],
            vec![Placeholder, Placeholder, Placeholder],
            vec![Placeholder],
        )?;
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda \
                    %0:dimension<batch ∈ [1, 9)>, \
                    %1:dimension<dividend ∈ [0, 9)>, \
                    %2:dimension<maybe_zero ∈ [0, 5)> .
                let %3:dimension<dividend / maybe_zero ∈ [0, 9)> = dimension_div [requires_runtime_assertion=true] %1 %2
                in (%3)
            "}
            .trim_end(),
        );
        Ok(())
    }

    #[test]
    fn test_batching_tracer_value_projection() -> Result<(), ProgramError> {
        // An array IR batching tracer projects onto a member type through its per-item type: the projected value wraps
        // the tracer itself (there is no tracer of a projected context), and lifting simply unwraps it. A carrier of
        // another member kind fails the projection.
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            ArrayIrEagerContext::new(),
            ArrayIrValue::Dimension(DimensionValue::constant(2)?),
        );
        let tracer = BatchingTracer::new(
            context.clone(),
            ArrayIrBatch::new(
                ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap()),
                BatchAxis::new(0),
            )?,
        );
        let projected = ValueProjection::<ArrayType>::projected(&tracer)?;
        assert_eq!(projected.r#type().into_owned(), ArrayType::new_static(DataType::F32, [3]));
        let projected = ValueProjection::<ArrayType>::into_projected(tracer.clone())?;
        assert_eq!(projected.r#type().into_owned(), ArrayType::new_static(DataType::F32, [3]));
        assert_eq!(
            <BatchingTracer<_, ArrayIrBatchingPolicy> as ValueProjection<ArrayType>>::from_projected(projected),
            tracer.clone()
        );
        assert_eq!(
            ValueProjection::<DimensionType>::into_projected(tracer).err(),
            Some(TypeError::Invalid { message: "expected dimension type but got array type".to_string() }),
        );

        let dimension = BatchingTracer::new(
            context,
            ArrayIrBatch::replicated(ArrayIrValue::<Array>::Dimension(DimensionValue::constant(3)?)),
        );
        let projected = ValueProjection::<DimensionType>::into_projected(dimension)?;
        assert_eq!(projected.r#type().extent(), Some(3));
        Ok(())
    }

    #[test]
    fn test_array_ir_batching_policy_bind_dispatch_rejects_mapped_dimension_members() {
        // Rules that consume a first-class dimension report the typed mapped-dimension diagnostic, even when a
        // malformed internal carrier bypasses the constructor's equivalent check. The carrier below is built through
        // the private fields on purpose, since `ArrayIrBatch::new` would already reject it.
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            ArrayIrEagerContext::new(),
            ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()),
        );
        let dimension_value =
            DimensionValue::new(DimensionType::new("extent", DimensionBounds::new(0, Some(9)).unwrap()), 4).unwrap();
        let dimension_type = dimension_value.r#type().into_owned();
        let dimension = ArrayIrValue::<Array>::Dimension(dimension_value);
        let mapped_dimension = ArrayIrBatch { value: dimension.clone(), batch_axis: BatchAxis::new(0) };
        let rejection =
            Err(BatchingError::MappedDimension { r#type: Box::new(dimension_type), axis: BatchAxis::new(0) });
        assert_eq!(
            ArrayIrOperation::<Array>::from(DimensionToScalarOperation).batch(
                &context,
                &EmptyRegionDriver,
                std::slice::from_ref(&mapped_dimension),
            ),
            rejection,
        );

        // A replicated dimension compares like ordinary shared data, while a mapped one is rejected.
        let comparison = ArrayIrOperation::<Array>::from(CompareOperation::new(ComparisonDirection::LessThan));
        let comparison_right = ArrayIrValue::Dimension(DimensionValue::constant(5).unwrap());
        assert_eq!(
            comparison.batch(
                &context,
                &EmptyRegionDriver,
                &[ArrayIrBatch::replicated(dimension), ArrayIrBatch::replicated(comparison_right.clone())],
            ),
            Ok(vec![ArrayIrBatch::replicated(ArrayIrValue::Array(Array::scalar(true).unwrap()))]),
        );
        assert_eq!(
            comparison.batch(
                &context,
                &EmptyRegionDriver,
                &[mapped_dimension, ArrayIrBatch::replicated(comparison_right)],
            ),
            rejection,
        );
    }

    #[test]
    fn test_elementwise_operation_batch() {
        // The blanket `BatchableOperation` for elementwise operations lifts `interpret` over the mapped batch axis. It
        // realigns every mapped input onto the common axis, broadcasts replicated inputs across the batch, and reports
        // each output on that common axis.
        let vector_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));
        let matrix_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));

        // Batching rules always receive the active `BatchingContext`, with the underlying work running
        // through its parent context.
        let context = BatchingContext::new(TestArrayContext::new(), 2);

        // Two inputs mapped on the same axis add per item, and the output stays mapped on that axis.
        let left = ArrayBatch::new(
            Array::from_elements::<f64>(matrix_type.clone(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
            Some(0),
        )
        .unwrap();
        let right = ArrayBatch::new(
            Array::from_elements::<f64>(matrix_type.clone(), &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]).unwrap(),
            Some(0),
        )
        .unwrap();
        let outputs = AddOperation::new().batch(&context, &EmptyRegionDriver, &[left.clone(), right]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value(), &Array::matrix(2, 3, vec![11.0, 22.0, 33.0, 44.0, 55.0, 66.0]).unwrap());

        // A replicated input is broadcast across the mapped input's batch before adding.
        let replicated =
            ArrayBatch::new(Array::from_elements::<f64>(vector_type.clone(), &[10.0, 20.0, 30.0]).unwrap(), None)
                .unwrap();
        let outputs = AddOperation::new().batch(&context, &EmptyRegionDriver, &[left.clone(), replicated]).unwrap();
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value(), &Array::matrix(2, 3, vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]).unwrap());

        // Inputs mapped on different axes are realigned onto the first mapped input's axis before adding.
        let transposed_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3), Dimension::Static(2)]));
        let right_axis_one = ArrayBatch::new(
            Array::from_elements::<f64>(transposed_type.clone(), &[10.0, 40.0, 20.0, 50.0, 30.0, 60.0]).unwrap(),
            Some(1),
        )
        .unwrap();
        let outputs = AddOperation::new().batch(&context, &EmptyRegionDriver, &[left, right_axis_one]).unwrap();
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value(), &Array::matrix(2, 3, vec![11.0, 22.0, 33.0, 44.0, 55.0, 66.0]).unwrap());

        // Packed mapped-axis positions are canonicalized independently of input rank. The rank-3 left input maps its
        // trailing axis while the rank-1 right input maps its only axis; their unbatched per-item shapes are `[3, 4]`
        // and scalar, respectively. The output is restored to the first mapped input's trailing axis.
        let left_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Static(3), Dimension::Static(4), Dimension::Static(2)]),
        );
        let right_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)]));
        let left = ArrayBatch::new(
            Array::from_elements::<f64>(left_type.clone(), &(1..=24).map(f64::from).collect::<Vec<_>>()).unwrap(),
            Some(2),
        )
        .unwrap();
        let right =
            ArrayBatch::new(Array::from_elements::<f64>(right_type.clone(), &[10.0, 20.0]).unwrap(), Some(0)).unwrap();
        let outputs = AddOperation::new().batch(&context, &EmptyRegionDriver, &[left, right]).unwrap();
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(2));
        assert_eq!(
            outputs[0].value(),
            &Array::from_elements::<f64>(
                left_type,
                &[
                    11.0, 22.0, 13.0, 24.0, 15.0, 26.0, 17.0, 28.0, 19.0, 30.0, 21.0, 32.0, 23.0, 34.0, 25.0, 36.0,
                    27.0, 38.0, 29.0, 40.0, 31.0, 42.0, 33.0, 44.0,
                ]
            )
            .unwrap(),
        );

        // A replicated per-item scalar is broadcast across the batch and against the mapped input's per-item shape, and
        // mapped inputs with different per-item ranks broadcast their per-item shapes item by item.
        let mapped = ArrayBatch::new(Array::vector(vec![1.0, 2.0]).unwrap(), Some(0)).unwrap();
        let scalar = ArrayBatch::replicated(Array::scalar(10.0).unwrap());
        let outputs = AddOperation::new().batch(&context, &EmptyRegionDriver, &[mapped, scalar]).unwrap();
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value(), &Array::vector(vec![11.0, 12.0]).unwrap());
        let rows = ArrayBatch::new(Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(), Some(0)).unwrap();
        let shifts = ArrayBatch::new(Array::vector(vec![10.0, 20.0]).unwrap(), Some(0)).unwrap();
        let outputs = AddOperation::new().batch(&context, &EmptyRegionDriver, &[rows, shifts]).unwrap();
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value(), &Array::matrix(2, 3, vec![11.0, 12.0, 13.0, 24.0, 25.0, 26.0]).unwrap());

        // With no input mapped, the inputs are interpreted as given and the output is replicated.
        let left_replicated =
            ArrayBatch::new(Array::from_elements::<f64>(vector_type.clone(), &[1.0, 2.0, 3.0]).unwrap(), None).unwrap();
        let right_replicated =
            ArrayBatch::new(Array::from_elements::<f64>(vector_type.clone(), &[10.0, 20.0, 30.0]).unwrap(), None)
                .unwrap();
        let outputs = AddOperation::new()
            .batch(&context, &EmptyRegionDriver, &[left_replicated, right_replicated])
            .unwrap();
        assert_eq!(outputs[0].batch_axis(), BatchAxis::replicated());
        assert_eq!(outputs[0].value(), &Array::vector(vec![11.0, 22.0, 33.0]).unwrap());

        // Unary elementwise operations use the same blanket rule and preserve the mapped input axis.
        let context = BatchingContext::new(TestArrayContext::new(), 3);
        let input =
            ArrayBatch::new(Array::from_elements::<f64>(vector_type.clone(), &[1.0, 2.0, 3.0]).unwrap(), Some(0))
                .unwrap();
        let outputs = NegOperation::new().batch(&context, &EmptyRegionDriver, &[input]).unwrap();
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value(), &Array::vector(vec![-1.0, -2.0, -3.0]).unwrap());
    }

    #[test]
    fn test_elementwise_operation_batch_normalizes_replicated_and_sharded_mapped_axes() {
        // Mapped inputs whose packed placements differ are normalized onto the context's batch axis placement before
        // the operation runs, so the output carries that placement rather than a mixture.
        for axis_type in [MeshAxisType::Explicit, MeshAxisType::Manual] {
            let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, axis_type).unwrap()]).unwrap();
            // The mapped axis of one input is placed on `x`, which makes that input vary along `x` when the axis is
            // manual, while the other input's mapped axis is replicated.
            let sharded_type =
                ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]))
                    .with_sharding(
                        Sharding::new(
                            mesh.clone(),
                            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
                        )
                        .unwrap()
                        .with_varying_manual_axes((axis_type == MeshAxisType::Manual).then_some("x"))
                        .unwrap(),
                    )
                    .unwrap();
            let replicated_type =
                ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]))
                    .with_sharding(Sharding::replicated(mesh.clone(), 2))
                    .unwrap();
            let sharded = ArrayBatch::new(
                Array::from_elements::<f64>(sharded_type.clone(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
                BatchAxis::new(0),
            )
            .unwrap();
            let replicated = ArrayBatch::new(
                Array::from_elements::<f64>(replicated_type, &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]).unwrap(),
                BatchAxis::new(0),
            )
            .unwrap();
            let context =
                BatchingContext::new(TestArrayContext::new(), 2).with_axis_sharding(ShardingDimension::sharded(["x"]));
            let outputs = AddOperation::new().batch(&context, &EmptyRegionDriver, &[sharded, replicated]).unwrap();
            assert_eq!(outputs.len(), 1);
            assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
            assert_eq!(outputs[0].r#type(), Cow::Borrowed(&sharded_type));
            assert_eq!(outputs[0].value().to_f64s(), vec![11.0, 22.0, 33.0, 44.0, 55.0, 66.0]);
        }
    }

    #[test]
    fn test_elementwise_operation_batch_preserves_explicit_mapped_axis_sharding() {
        // The per-item type of a mapped input drops the mapped dimension's placement, and the batched output restores
        // exactly that placement at the input's own mapped position, whichever position that is.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let context = BatchingContext::new(TestArrayContext::new(), 2);
        for (batch_axis, dimensions) in [vec![2, 3, 4], vec![3, 2, 4], vec![3, 4, 2]].into_iter().enumerate() {
            let mut sharding_dimensions = vec![ShardingDimension::replicated(); 3];
            sharding_dimensions[batch_axis] = ShardingDimension::sharded(["x"]);
            let batched_type =
                ArrayType::new(DataType::F64, Shape::new(dimensions.into_iter().map(Dimension::Static).collect()))
                    .with_sharding(Sharding::new(mesh.clone(), sharding_dimensions).unwrap())
                    .unwrap();
            let batch = ArrayBatch::new(
                Array::from_elements::<f64>(batched_type.clone(), &(0..24).map(f64::from).collect::<Vec<_>>()).unwrap(),
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
    fn test_elementwise_operation_batch_stages_broadcast_to_full_shape() {
        // A replicated scalar constant added to a mapped [3, 4] input: the elementwise rule materializes a broadcasting
        // operation to the full common batched shape so the staged add receives shape-congruent inputs. This is
        // required for backends such as XLA whose elementwise lowering (e.g., `stablehlo.add`) has no implicit
        // broadcasting.
        let parent = DomainTracingContext::<TestArrayContext>::new();
        let builder = parent.builder().clone();
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3), Dimension::Static(4)]));
        let input_atom = builder.borrow_mut().add_input(input_type);
        let input_tracer = parent.tracer(input_atom, None);
        let output = Batch::batch(
            &parent,
            |x| {
                let bias = x.context().lift(Array::scalar(1.0).unwrap())?;
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
                let %1:f64[] = const 1.0
                    %2:f64[3, 4] = broadcast [output_type=f64[3, 4], output_axes=[]] %1
                    %3:f64[3, 4] = add %0 %2
                in (%3)
            "}
            .trim_end(),
        );
        let input = Array::matrix(3, 4, (0..12).map(|value| value as f64).collect()).unwrap();
        let output = program.interpret(input).unwrap();
        assert_eq!(output.to_f64s(), (0..12).map(|value| value as f64 + 1.0).collect::<Vec<_>>());
    }

    #[test]
    fn test_region_ref_batched() {
        // Structural batching of a region inserts the mapped extent into every mapped boundary type with the requested
        // placement, records manual-axis variation for a manual placement, and reports each output's natural axis.
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
            let (_, program) = TestArrayContext::trace(|inputs: Vec<_>| Ok(inputs), vec![unbatched_type]).unwrap();

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
    fn test_region_ref_batched_cache_specializes_arguments() {
        // Repeated structural requests are served from the region's retained artifact, keyed by the complete request.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(NegOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        let region = program.entry_region_ref();

        let first = region
            .batched(2, ShardingDimension::Replicated, &[BatchAxis::new(0)], ProgramBatchingOutputAxesPolicy::Natural)
            .unwrap();
        let repeated = region
            .batched(2, ShardingDimension::Replicated, &[BatchAxis::new(0)], ProgramBatchingOutputAxesPolicy::Natural)
            .unwrap();
        let (first, first_axes) = first.into_parts();
        let (repeated, repeated_axes) = repeated.into_parts();
        assert_eq!(first.to_string(), repeated.to_string());
        assert_eq!(first_axes, repeated_axes);

        // Signed axes normalize before lookup, so `-1` and `0` share the scalar input's one valid packed position.
        let normalized = region
            .batched(2, ShardingDimension::Replicated, &[BatchAxis::new(-1)], ProgramBatchingOutputAxesPolicy::Natural)
            .unwrap();
        assert_eq!(normalized.output_axes(), &[BatchAxis::new(0)]);

        // Every other structural input separates entries: extent, named-axis scope, sharding, output policy,
        // and mapped-versus-replicated input state.
        let wider = region
            .batched(3, ShardingDimension::Replicated, &[BatchAxis::new(0)], ProgramBatchingOutputAxesPolicy::Natural)
            .unwrap();
        assert_eq!(
            wider.into_parts().0.input_types(),
            &[ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]))],
        );
        region
            .batched_with_axis_name(
                2,
                Some("items".to_string()),
                ShardingDimension::Replicated,
                &[BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap();
        region
            .batched_with_axis_name(
                2,
                Some("models".to_string()),
                ShardingDimension::Replicated,
                &[BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap();
        region
            .batched(
                2,
                ShardingDimension::Unconstrained,
                &[BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap();
        region
            .batched(
                2,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::AlignAllTo(Axis::from(0)),
            )
            .unwrap();
        let trailing = region
            .batched(
                2,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::AlignAllTo(Axis::from(-1)),
            )
            .unwrap();
        assert_eq!(trailing.output_axes(), &[BatchAxis::new(0)]);
        region
            .batched(
                2,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::AlignEachTo(vec![BatchAxis::new(0)]),
            )
            .unwrap();
        let each_trailing = region
            .batched(
                2,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::AlignEachTo(vec![BatchAxis::new(-1)]),
            )
            .unwrap();
        assert_eq!(each_trailing.output_axes(), &[BatchAxis::new(0)]);
        let replicated = region
            .batched(
                2,
                ShardingDimension::Replicated,
                &[BatchAxis::replicated()],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap();
        assert_eq!(replicated.output_axes(), &[BatchAxis::replicated()]);

        assert_eq!(
            region.transform_statistics::<ArrayBatchingTransform>(),
            Some(SpecializationCacheStatistics {
                hits: 4,
                misses: 8,
                productions: 8,
                abandoned_productions: 0,
                evictions: 0,
            }),
        );
    }

    #[test]
    fn test_region_ref_batched_cache_preserves_mixed_rank_align_all_semantics() {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let scalar = builder.add_input(ArrayType::scalar(DataType::F64));
        let vector = builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)])));
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![scalar, vector], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        let region = program.entry_region_ref();
        let input_axes = [BatchAxis::replicated(), BatchAxis::replicated()];

        // `-1` resolves separately against each packed output rank: position 0 for the scalar and position 1 for the
        // vector. It therefore cannot be canonicalized to one positive `AlignAllTo` position.
        let negative = region
            .batched(
                2,
                ShardingDimension::Replicated,
                &input_axes,
                ProgramBatchingOutputAxesPolicy::AlignAllTo(Axis::from(-1)),
            )
            .unwrap();
        let repeated = region
            .batched(
                2,
                ShardingDimension::Replicated,
                &input_axes,
                ProgramBatchingOutputAxesPolicy::AlignAllTo(Axis::from(-1)),
            )
            .unwrap();
        let leading = region
            .batched(
                2,
                ShardingDimension::Replicated,
                &input_axes,
                ProgramBatchingOutputAxesPolicy::AlignAllTo(Axis::from(0)),
            )
            .unwrap();

        assert_eq!(negative.output_axes(), &[BatchAxis::new(0), BatchAxis::new(1)]);
        assert_eq!(repeated.output_axes(), negative.output_axes());
        assert_eq!(leading.output_axes(), &[BatchAxis::new(0), BatchAxis::new(0)]);
        assert_eq!(
            region.transform_statistics::<ArrayBatchingTransform>(),
            Some(SpecializationCacheStatistics {
                hits: 1,
                misses: 2,
                productions: 2,
                abandoned_productions: 0,
                evictions: 0,
            }),
        );
    }

    #[test]
    fn test_region_ref_batched_with_axis_name_separates_named_axis_semantics() {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder
            .add_instruction(
                ParallelReduceOperation::new(ReductionKind::Sum, "items".to_string()),
                Vec::new(),
                vec![input],
                None,
            )
            .unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        let region = program.entry_region_ref();

        // The matching named level consumes the mapped axis, while a different level forwards the collective and
        // preserves that axis. These are semantically distinct cached programs despite sharing every other argument.
        let matching = region
            .batched_with_axis_name(
                4,
                Some("items".to_string()),
                ShardingDimension::Replicated,
                &[BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap();
        let forwarded = region
            .batched_with_axis_name(
                4,
                Some("models".to_string()),
                ShardingDimension::Replicated,
                &[BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap();
        let repeated = region
            .batched_with_axis_name(
                4,
                Some("items".to_string()),
                ShardingDimension::Replicated,
                &[BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap();
        let (matching, matching_axes) = matching.into_parts();
        let (forwarded, forwarded_axes) = forwarded.into_parts();
        let (repeated, repeated_axes) = repeated.into_parts();
        assert_eq!(matching_axes, vec![BatchAxis::replicated()]);
        assert_eq!(forwarded_axes, vec![BatchAxis::new(0)]);
        assert_ne!(matching.to_string(), forwarded.to_string());
        assert_eq!(matching.to_string(), repeated.to_string());
        assert_eq!(matching_axes, repeated_axes);

        assert_eq!(
            region.transform_statistics::<ArrayBatchingTransform>(),
            Some(SpecializationCacheStatistics {
                hits: 1,
                misses: 2,
                productions: 2,
                abandoned_productions: 0,
                evictions: 0,
            }),
        );
    }

    #[test]
    fn test_region_ref_batched_cache_artifacts_preserve_dynamic_identities() {
        let dimension =
            Dimension::Dynamic(DimensionVariable::new("items", DimensionBounds::non_negative(Some(16)).unwrap()));
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![dimension.clone()]));
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(input_type);
        let source =
            builder.build::<Vec<Array>, Vec<Array>>(vec![input], vec![Placeholder], vec![Placeholder]).unwrap();

        let (first, first_axes) = source
            .batched(2, ShardingDimension::Replicated, &[BatchAxis::new(0)], ProgramBatchingOutputAxesPolicy::Natural)
            .unwrap()
            .into_parts();
        let (second, second_axes) = source
            .batched(2, ShardingDimension::Replicated, &[BatchAxis::new(0)], ProgramBatchingOutputAxesPolicy::Natural)
            .unwrap()
            .into_parts();
        assert_eq!(first_axes, second_axes);
        assert_eq!(first.to_string(), second.to_string());

        // Attaching both cached copies as separate branches must not introduce any identity relationship that two
        // independent derivations would not have had. Structural batching inherits the source program's own dynamic
        // dimension identity instead of minting a fresh one per derivation, and importing splices region types
        // verbatim, so both branches present exactly that source identity. Comparing types rather than renderings is
        // what makes this an identity assertion: two separately created "items" variables render identically but are
        // never equal.
        let mut outer = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let first_region = outer.import_program(first);
        let second_region = outer.import_program(second);
        let first_branch_type = RegionRef::new(&outer.regions, first_region).unwrap().input_types().remove(0);
        let second_branch_type = RegionRef::new(&outer.regions, second_region).unwrap().input_types().remove(0);
        assert_eq!(first_branch_type, second_branch_type);
        assert_eq!(first_branch_type.shape().dimensions(), &[Dimension::Static(2), dimension]);

        // The condition therefore seals against one coherent shared interface.
        let predicate = outer.add_input(ArrayType::scalar(DataType::Boolean));
        let input = outer.add_input(
            source
                .batched(
                    2,
                    ShardingDimension::Replicated,
                    &[BatchAxis::new(0)],
                    ProgramBatchingOutputAxesPolicy::Natural,
                )
                .unwrap()
                .into_parts()
                .0
                .input_types()[0]
                .clone(),
        );
        let output = outer
            .add_instruction(ConditionOperation::new(), vec![first_region, second_region], vec![predicate, input], None)
            .unwrap()[0];
        let outer = outer
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        assert_eq!(outer.output_types(), &outer.input_types()[1..]);

        assert_eq!(
            source.entry_region_ref().transform_statistics::<ArrayBatchingTransform>(),
            Some(SpecializationCacheStatistics {
                hits: 2,
                misses: 1,
                productions: 1,
                abandoned_productions: 0,
                evictions: 0,
            }),
        );
    }

    #[test]
    fn test_program_batched() {
        // Trace a per-item squaring function into a flat program over per-item vector types.
        let vector_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));
        let (_, program) = TestArrayContext::trace(
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
        let input = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let outputs = batched.interpret(vec![input]).unwrap();
        assert_eq!(outputs, vec![Array::matrix(2, 3, vec![1.0, 4.0, 9.0, 16.0, 25.0, 36.0]).unwrap()]);

        // A negative input axis is normalized against the packed input rank. Mapping the final axis consumes a
        // `[3, 2]` packed value and preserves that canonical axis through the elementwise body.
        let (batched, output_axes) = program
            .batched(2, ShardingDimension::Replicated, &[BatchAxis::new(-1)], ProgramBatchingOutputAxesPolicy::Natural)
            .unwrap()
            .into_parts();
        assert_eq!(output_axes, vec![BatchAxis::new(1)]);
        let input = Array::matrix(3, 2, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap();
        let outputs = batched.interpret(vec![input]).unwrap();
        assert_eq!(outputs, vec![Array::matrix(3, 2, vec![1.0, 16.0, 4.0, 25.0, 9.0, 36.0]).unwrap()]);

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
        let outputs = batched.interpret(vec![Array::vector(vec![1.0, 2.0, 3.0]).unwrap()]).unwrap();
        assert_eq!(outputs, vec![Array::matrix(2, 3, vec![1.0, 4.0, 9.0, 1.0, 4.0, 9.0]).unwrap()]);

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
        let outputs = batched.interpret(vec![Array::vector(vec![1.0, 2.0, 3.0]).unwrap()]).unwrap();
        assert_eq!(outputs, vec![Array::matrix(3, 2, vec![1.0, 1.0, 4.0, 4.0, 9.0, 9.0]).unwrap()]);

        // A mismatched `input_batch_axes` count is rejected with the source program's input count as `expected` and
        // the caller-supplied axis count as `actual`.
        assert_eq!(
            program
                .batched(2, ShardingDimension::Replicated, &[], ProgramBatchingOutputAxesPolicy::Natural)
                .map(|_| ()),
            Err(BatchingError::from(ProgramError::InvalidInputCount { expected: 1, actual: 0 })),
        );

        // Repeating a request is served from the entry region's retained artifact: production counts every uncached
        // derivation, so the four distinct requests above produced four artifacts and the repetitions produce none.
        for _ in 0..3 {
            let batched = program
                .batched(
                    2,
                    ShardingDimension::Replicated,
                    &[BatchAxis::new(0)],
                    ProgramBatchingOutputAxesPolicy::Natural,
                )
                .unwrap();
            assert_eq!(batched.output_axes(), &[BatchAxis::new(0)]);
        }
        assert_eq!(
            program.entry_region_ref().transform_statistics::<ArrayBatchingTransform>(),
            Some(SpecializationCacheStatistics {
                hits: 3,
                misses: 4,
                productions: 4,
                abandoned_productions: 0,
                evictions: 0,
            }),
        );
    }

    #[test]
    fn test_program_batched_preserves_source_provenance() {
        // Batching rewrites each source instruction into the instructions its rule stages, so every instruction of the
        // batched program records the source instruction it came from. The mapped/replicated input pair forces the
        // elementwise rule down its broadcasting path, which makes the first rewrite one-to-many.
        let first = Provenance::scope(ProvenanceScope::new("a"), Provenance::unknown());
        let second = Provenance::scope(ProvenanceScope::new("b"), Provenance::unknown());
        let vector_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let mapped = builder.add_input(vector_type.clone());
        let replicated = builder.add_input(vector_type);
        let summed = builder
            .add_instruction(AddOperation::new(), Vec::new(), vec![mapped, replicated], Some(first.clone()))
            .unwrap()[0];
        let output = builder
            .add_instruction(NegOperation::new(), Vec::new(), vec![summed], Some(second.clone()))
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let (batched, output_axes) = program
            .batched(
                2,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0), BatchAxis::replicated()],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts();
        assert_eq!(output_axes, vec![BatchAxis::new(0)]);
        assert_eq!(
            batched
                .instructions()
                .iter()
                .map(|instruction| (instruction.operation().name(), instruction.provenance().clone()))
                .collect::<Vec<_>>(),
            vec![("broadcast", first.clone()), ("add", first), ("neg", second)],
        );
    }

    #[test]
    fn test_program_batched_rejects_mismatched_output_axes_count() {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(NegOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();

        // `AlignEachTo` must list exactly one axis per program output, and the rejection blames the caller-supplied
        // axes: the program's output count is `expected` and the policy length is `actual`, matching the sibling
        // input-axes check. Asserting the exact payload keeps those two roles from silently swapping.
        assert_eq!(
            program
                .batched(
                    2,
                    ShardingDimension::Replicated,
                    &[BatchAxis::new(0)],
                    ProgramBatchingOutputAxesPolicy::AlignEachTo(vec![BatchAxis::new(0), BatchAxis::new(0)]),
                )
                .map(|_| ()),
            Err(BatchingError::from(ProgramError::InvalidOutputCount { expected: 1, actual: 2 })),
        );
    }

    #[test]
    fn test_program_batched_carries_dynamic_per_item_dimensions() {
        // Contract: program batching inserts the mapped axis as `Dimension::Static(axis_size)` taken from the caller's
        // `axis_size`, so it never reads a batch extent off the input type the way value-level `Batch::batch` does and
        // therefore never raises `BatchingError::DynamicBatchAxis`. A dynamic per-item dimension is not the batch
        // axis, and it crosses the rewritten boundary unchanged.
        let dynamic = Dimension::Dynamic(DimensionVariable::new("n", DimensionBounds::unbounded()));
        let unbatched_type = ArrayType::new(DataType::F64, Shape::new(vec![dynamic.clone()]));
        let batched_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), dynamic]));
        let (_, program) = TestArrayContext::trace(
            |inputs: Vec<_>| Ok(vec![inputs[0].clone() + inputs[0].clone()]),
            vec![unbatched_type.clone()],
        )
        .unwrap();
        let (batched, output_axes) = program
            .batched(2, ShardingDimension::Replicated, &[BatchAxis::new(0)], ProgramBatchingOutputAxesPolicy::Natural)
            .unwrap()
            .into_parts();
        assert_eq!(batched.input_types(), &[batched_type.clone()]);
        assert_eq!(batched.output_types(), &[batched_type]);
        assert_eq!(output_axes, vec![BatchAxis::new(0)]);

        // Adding a replicated second input routes the elementwise rule through its broadcast path, where each dynamic
        // target dimension is resolved by `DimensionSource` from the source axis that supplied it instead of being
        // rejected. The staged homogeneous `broadcast` therefore retains the dynamic extent in its stored output type,
        // and the replicated input keeps its unbatched dynamically shaped type.
        let (_, program) = TestArrayContext::trace(
            |inputs: Vec<_>| Ok(vec![inputs[0].clone() + inputs[1].clone()]),
            vec![unbatched_type.clone(), unbatched_type],
        )
        .unwrap();
        let (batched, output_axes) = program
            .batched(
                2,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0), BatchAxis::replicated()],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts();
        assert_eq!(output_axes, vec![BatchAxis::new(0)]);
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:f64[2, n], %1:f64[n] .
                let %2:f64[2, n] = broadcast [output_type=f64[2, n], output_axes=[1]] %1
                    %3:f64[2, n] = add %0 %2
                in (%3)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_batch_over_arrays() {
        // `Batch::batch` on an explicit context maps the closure over the mapped input axis: each item of the
        // length-3 batch is squared, and the output carries its mapped axis back at the requested position.
        let output: Array = TestArrayContext::new()
            .batch(
                |x| Ok(x.clone() * x),
                Array::vector(vec![1.0, 2.0, 3.0]).unwrap(),
                BatchAxis::new(0),
                BatchAxis::new(0),
                None,
            )
            .unwrap();
        assert_eq!(output, Array::vector(vec![1.0, 4.0, 9.0]).unwrap());

        // A mapped output declaration broadcasts a naturally replicated result across the explicit batch.
        // The signed `-1` declaration places that new batch dimension last.
        let output: Array = TestArrayContext::new()
            .batch(
                |x| Ok(x.clone() * x),
                Array::vector(vec![1.0, 2.0, 3.0]).unwrap(),
                BatchAxis::replicated(),
                BatchAxis::new(-1),
                2,
            )
            .unwrap();
        assert_eq!(output, Array::matrix(3, 2, vec![1.0, 1.0, 4.0, 4.0, 9.0, 9.0]).unwrap());

        // The free `batch` serves top-level concrete values through their `Value::Domain` declarations: a
        // plain `Array` input recovers the test backend's rich eager domain, mirroring how JAX's `vmap` falls back
        // to the default eager interpreter for concrete arrays.
        let output: Array = batch(
            |x| Ok(x.clone() * x),
            Array::vector(vec![1.0, 2.0, 3.0]).unwrap(),
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        )
        .unwrap();
        assert_eq!(output, Array::vector(vec![1.0, 4.0, 9.0]).unwrap());

        // Under an active trace, the free `batch` recovers the staging context from its tracer input instead, so
        // `batch` composes inside traced code without threading a context. The traced function squares each row of
        // its `[2, 3]` input by batching a per-item squaring closure over axis 0.
        let matrix_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let (_, program) = TestArrayContext::trace(
            |inputs: Vec<_>| {
                let mapped =
                    batch(|x| Ok(x.clone() * x), inputs[0].clone(), BatchAxis::new(0), BatchAxis::new(0), None)?;
                Ok(vec![mapped])
            },
            vec![matrix_type],
        )
        .unwrap();
        let input = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let outputs = program.interpret(vec![input]).unwrap();
        assert_eq!(outputs, vec![Array::matrix(2, 3, vec![1.0, 4.0, 9.0, 16.0, 25.0, 36.0]).unwrap()]);

        // Nested inside an eager `batch`, the inner free `batch` recovers the outer `BatchingContext` from its
        // `BatchingTracer` input, so that `batch` nests inside `batch`: the outer level maps rows and the inner level
        // maps items within each row.
        let output: Array = TestArrayContext::new()
            .batch(
                |row| Ok(batch(|item| Ok(item.clone() * item), row, BatchAxis::new(0), BatchAxis::new(0), None)?),
                Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
                BatchAxis::new(0),
                BatchAxis::new(0),
                None,
            )
            .unwrap();
        assert_eq!(output, Array::matrix(2, 3, vec![1.0, 4.0, 9.0, 16.0, 25.0, 36.0]).unwrap());

        // The inner level may mix mapped and replicated inputs of its own: here the row is mapped while a captured bias
        // is replicated across the inner batch items.
        let output: Array = TestArrayContext::new()
            .batch(
                |(row, bias)| {
                    let context = row.context().clone();
                    Ok(Batch::batch(
                        &context,
                        |(item, bias)| Ok(item + bias),
                        (row, bias),
                        (BatchAxis::new(0), BatchAxis::replicated()),
                        BatchAxis::new(0),
                        None,
                    )?)
                },
                (Array::matrix(3, 4, (0..12).map(f64::from).collect()).unwrap(), Array::scalar(0.5).unwrap()),
                (BatchAxis::new(0), BatchAxis::replicated()),
                BatchAxis::new(0),
                None,
            )
            .unwrap();
        assert_eq!(output, Array::matrix(3, 4, (0..12).map(|value| f64::from(value) + 0.5).collect()).unwrap());

        // A replicated input with an explicit batch size runs the closure on the shared value and returns
        // a replicated output.
        let output: Array = TestArrayContext::new()
            .batch(
                |x| Ok(x.clone() * x),
                Array::vector(vec![1.0, 2.0, 3.0]).unwrap(),
                BatchAxis::replicated(),
                BatchAxis::replicated(),
                2,
            )
            .unwrap();
        assert_eq!(output, Array::vector(vec![1.0, 4.0, 9.0]).unwrap());

        // Declaring a mapped output as replicated is rejected: collapsing a mapped axis requires an explicit
        // reduction inside the batched function.
        let error = TestArrayContext::new()
            .batch(
                |x| Ok(x.clone() * x),
                Array::vector(vec![1.0, 2.0, 3.0]).unwrap(),
                BatchAxis::new(0),
                BatchAxis::replicated(),
                None,
            )
            .unwrap_err();
        assert_eq!(
            error,
            BatchingError::MismatchedOutputAxes { expected: BatchAxis::replicated(), actual: BatchAxis::new(0) }
        );
    }

    #[test]
    fn test_batch_composes_with_differentiation() {
        // Differentiation inside a batched function sees per-item tracers: forward mode produces the per-item value and
        // tangent, and reverse mode the per-item value and gradient, each mapped back onto the batch axis.
        let (value, tangent): (Array, Array) = TestArrayContext::new()
            .batch(
                |x| {
                    let context = x.context().clone();
                    ForwardModeDifferentiate::jvp(&context, |y, ()| Ok(y.clone() * y), x.clone(), x.one_like()?, ())
                        .map_err(ProgramError::from)
                },
                Array::vector(vec![2.0, 3.0]).unwrap(),
                BatchAxis::new(0),
                (BatchAxis::new(0), BatchAxis::new(0)),
                None,
            )
            .unwrap();
        assert_eq!(value, Array::vector(vec![4.0, 9.0]).unwrap());
        assert_eq!(tangent, Array::vector(vec![4.0, 6.0]).unwrap());

        let (value, gradient): (Array, Array) = TestArrayContext::new()
            .batch(
                |x| {
                    let context = x.context().clone();
                    Ok(context.differentiate_at(x).value_and_gradient(|y| y.clone() * y)?)
                },
                Array::vector(vec![2.0, 3.0]).unwrap(),
                BatchAxis::new(0),
                (BatchAxis::new(0), BatchAxis::new(0)),
                None,
            )
            .unwrap();
        assert_eq!(value, Array::vector(vec![4.0, 9.0]).unwrap());
        assert_eq!(gradient, Array::vector(vec![4.0, 6.0]).unwrap());
    }

    #[test]
    fn test_batch_composes_inside_differentiation() {
        // A batched function inside a differentiated one is differentiated through its batching rules. In forward mode
        // the tangent flows through the per-item squares.
        let (primal, tangent): (Array, Array) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .jvp(
                |x, ()| {
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
                Array::vector(vec![2.0, 3.0]).unwrap(),
                Array::vector(vec![1.0, 1.0]).unwrap(),
                (),
            )
            .unwrap();
        assert_eq!(primal, Array::vector(vec![4.0, 9.0]).unwrap());
        assert_eq!(tangent, Array::vector(vec![4.0, 6.0]).unwrap());

        // In reverse mode the gradient flows back through the batched squares into each item of the mapped input.
        let (value, gradient): (Array, Array) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .differentiate_at(Array::vector(vec![2.0, 3.0]).unwrap())
            .value_and_gradient(|x| {
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
                mapped.reduce(&[0], ReductionKind::Sum).unwrap()
            })
            .unwrap();
        assert_eq!(value, Array::scalar(13.0).unwrap());
        assert_eq!(gradient, Array::vector(vec![4.0, 6.0]).unwrap());

        // A differentiated scalar that is replicated inside the batch makes the elementwise rule stage a broadcast, so
        // the gradient flows back through the broadcast's transpose (a sum over the batch axis).
        let (value, gradient) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .differentiate_at(Array::scalar(2.0).unwrap())
            .value_and_gradient(|x| {
                let context = x.context().clone();
                let y = context.lift(Array::vector(vec![1.0, 2.0, 3.0, 4.0]).unwrap()).unwrap();
                let mapped: LinearizationTracer<EagerContext<Array, ArrayOperation<Array>>> = Batch::batch(
                    &context,
                    |(item, shift)| Ok(item * shift),
                    (y, x),
                    (BatchAxis::new(0), BatchAxis::replicated()),
                    BatchAxis::new(0),
                    None,
                )
                .unwrap();
                mapped.reduce(&[0], ReductionKind::Sum).unwrap()
            })
            .unwrap();
        assert_eq!(value, Array::scalar(20.0).unwrap());
        assert_eq!(gradient, Array::scalar(10.0).unwrap());
    }

    #[test]
    fn test_batch_over_array_ir_values() -> Result<(), ProgramError> {
        let matrix = ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap());

        // The free transform infers its first-class mapped extent from the packed array input and can move
        // the mapped output axis without exposing the policy at the call site.
        let moved: ArrayIrValue<Array> =
            batch(|row| Ok(row), matrix.clone(), BatchAxis::new(0), BatchAxis::new(1), None)?;
        assert_eq!(moved, ArrayIrValue::Array(Array::matrix(3, 2, vec![1.0_f32, 4.0, 2.0, 5.0, 3.0, 6.0],).unwrap()),);

        // A replicated array output is dynamically broadcast with the inferred extent input.
        let replicated = ArrayIrValue::Array(Array::vector(vec![10.0_f32, 20.0, 30.0]).unwrap());
        let broadcasted: ArrayIrValue<Array> = batch(
            |(_, replicated)| Ok(replicated),
            (matrix.clone(), replicated),
            (BatchAxis::new(0), BatchAxis::replicated()),
            BatchAxis::new(0),
            None,
        )?;
        assert_eq!(
            broadcasted,
            ArrayIrValue::Array(Array::matrix(2, 3, vec![10.0_f32, 20.0, 30.0, 10.0, 20.0, 30.0],).unwrap()),
        );

        // A named array IR specification reaches the policy-selected context, and an explicit first-class extent
        // drives mapped output materialization when every input is replicated.
        let named_extent = BatchAxisSpecification::new(ArrayIrValue::Dimension(DimensionValue::constant(2)?), "items");
        let explicitly_broadcasted: ArrayIrValue<Array> = batch(
            |replicated| {
                assert_eq!(replicated.context().axis_name(), Some("items"));
                Ok(replicated)
            },
            ArrayIrValue::Array(Array::vector(vec![7.0_f32, 8.0, 9.0]).unwrap()),
            BatchAxis::replicated(),
            BatchAxis::new(0),
            named_extent,
        )?;
        assert_eq!(
            explicitly_broadcasted,
            ArrayIrValue::Array(Array::matrix(2, 3, vec![7.0_f32, 8.0, 9.0, 7.0, 8.0, 9.0],).unwrap()),
        );

        // Exact zero extents use the same first-class extent path and produce an empty mapped dimension.
        let empty_extent = BatchAxisSpecification::with_extent(ArrayIrValue::Dimension(DimensionValue::constant(0)?));
        let empty: ArrayIrValue<Array> = batch(
            |replicated| Ok(replicated),
            ArrayIrValue::Array(Array::vector(vec![7.0_f32, 8.0, 9.0]).unwrap()),
            BatchAxis::replicated(),
            BatchAxis::new(0),
            empty_extent,
        )?;
        assert_eq!(
            empty.r#type().as_ref(),
            &ArrayIrType::Array(ArrayType::new(
                DataType::F32,
                Shape::new(vec![Dimension::Static(0), Dimension::Static(3)]),
            )),
        );

        // Dimension values remain shared shape values and can flow through the closure only as replicated outputs.
        let extent = ArrayIrValue::Dimension(DimensionValue::constant(3)?);
        let dimension: ArrayIrValue<Array> = batch(
            |(_, extent)| Ok(extent),
            (matrix.clone(), extent.clone()),
            (BatchAxis::new(0), BatchAxis::replicated()),
            BatchAxis::replicated(),
            None,
        )?;
        assert_eq!(dimension, extent);
        let mapped_dimension: Result<ArrayIrValue<Array>, BatchingError> = batch(
            |(_, extent)| Ok(extent),
            (matrix.clone(), extent),
            (BatchAxis::new(0), BatchAxis::replicated()),
            BatchAxis::new(0),
            None,
        );
        assert!(
            matches!(mapped_dimension, Err(BatchingError::MappedDimension { axis, .. }) if axis == BatchAxis::new(0))
        );

        // An explicit first-class extent is checked against every mapped input.
        let mismatched_extent =
            BatchAxisSpecification::with_extent(ArrayIrValue::Dimension(DimensionValue::constant(3)?));
        let mismatched: Result<ArrayIrValue<Array>, BatchingError> =
            batch(|row| Ok(row), matrix.clone(), BatchAxis::new(0), BatchAxis::new(0), mismatched_extent);
        let mismatched = mismatched.unwrap_err();
        assert!(matches!(mismatched, BatchingError::Program(ProgramError::Custom(_))));
        assert_eq!(mismatched.to_string(), "assertion failed: batch dimensions must agree; expected=3, actual=2",);

        // A first-class dimension itself cannot be declared mapped at the transform boundary.
        let mapped_input: Result<ArrayIrValue<Array>, BatchingError> = batch(
            |extent| Ok(extent),
            ArrayIrValue::Dimension(DimensionValue::constant(2)?),
            BatchAxis::new(0),
            BatchAxis::replicated(),
            None,
        );
        assert!(matches!(mapped_input, Err(BatchingError::MappedDimension { axis, .. }) if axis == BatchAxis::new(0)));

        // A dimension derived from mapped data would differ across batch items, so it cannot shape a broadcast. The
        // closure fails at the derivation without staging anything beyond the extent read by the transform entry.
        let trace = ArrayIrTraceContext::new();
        let extent_variable = DimensionVariable::new("extent", DimensionBounds::new(1, Some(4))?);
        let scalars = trace.input(ArrayType::new_static(DataType::I32, [2]).into());
        let fill = trace.input(ArrayType::scalar(DataType::F32).into());
        let mut staged_before_derivation = None;
        let broadcasted = Batch::batch(
            &trace,
            |(scalar, fill)| {
                staged_before_derivation = Some(trace.builder().borrow().instructions().len());
                let extent = scalar.to_dimension(extent_variable.clone())?;
                fill.dynamic_broadcast_to(std::slice::from_ref(&extent))
            },
            (scalars, fill),
            (BatchAxis::new(0), BatchAxis::replicated()),
            BatchAxis::new(0),
            None,
        );
        assert_eq!(
            broadcasted.map(|_| ()),
            Err(BatchingError::MappedDimension { r#type: Box::new(extent_variable.into()), axis: BatchAxis::new(0) }),
        );
        assert_eq!(Some(trace.builder().borrow().instructions().len()), staged_before_derivation);

        // Nested public batching selects the array IR policy again from the outer batching context.
        let nested: ArrayIrValue<Array> = batch(
            |row| batch(|item| Ok(item), row, BatchAxis::new(0), BatchAxis::new(0), None).map_err(Into::into),
            matrix.clone(),
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        )?;
        assert_eq!(nested, matrix);

        // Under staging, an inferred dynamic extent remains an explicit `dimension_size` result consumed by
        // the output broadcast rather than metadata reconstructed from the array type, while the replicated
        // input's statically known axis folds into an exact dimension constant instead of another read.
        let trace = ArrayIrTraceContext::new();
        let batch_variable = DimensionVariable::new("batch", DimensionBounds::new(1, Some(9))?);
        let mapped = trace.input(
            ArrayType::new(
                DataType::F32,
                Shape::new(vec![Dimension::Dynamic(batch_variable.clone()), Dimension::Static(3)]),
            )
            .into(),
        );
        let replicated = trace.input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)])).into());
        let staged = Batch::batch(
            &trace,
            |(_, replicated)| Ok(replicated),
            (mapped, replicated),
            (BatchAxis::new(0), BatchAxis::replicated()),
            BatchAxis::new(0),
            None,
        )?;
        assert_eq!(
            staged.r#type().as_ref(),
            &ArrayIrType::Array(ArrayType::new(
                DataType::F32,
                Shape::new(vec![Dimension::Dynamic(batch_variable), Dimension::Static(3)]),
            )),
        );
        let program = trace.builder().borrow().clone().build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            vec![staged.atom_id()?],
            vec![Placeholder, Placeholder],
            vec![Placeholder],
        )?;
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[batch, 3], %1:f32[3] .
                let %2:dimension<batch ∈ [1, 9)> = dimension_size [axis=0] %0
                    %3:dimension<3> = constant [value=3]
                    %4:f32[batch, 3] = broadcast [output_axes=[1]] %1 %2 %3
                in (%4)
            "}
            .trim_end(),
        );

        Ok(())
    }

    #[test]
    fn test_program_batched_with_threaded_extent() -> Result<(), ProgramError> {
        // Structurally batching an array IR program with a threaded extent gives the result one leading dimension input
        // of the requested extent type and forwards it as the leading output. The reported output axes exclude that
        // bookkeeping output, and every mapped boundary type carries the extent at its batch axis.
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let array = builder.add_input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)])).into());
        let shared_dimension_type = DimensionType::new("shared", DimensionBounds::new(0, Some(17))?);
        let dimension = builder.add_input(shared_dimension_type.clone().into());
        let negated = builder.add_instruction(
            ArrayIrOperation::Array(ArrayOperation::Neg(NegOperation::new())),
            Vec::new(),
            vec![array],
            None,
        )?[0];
        let program = builder.build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            vec![negated, dimension],
            vec![Placeholder, Placeholder],
            vec![Placeholder, Placeholder],
        )?;
        let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(9))?);
        let batched = program.batched_with_threaded_extent(
            DimensionType::from(batch.clone()),
            ShardingDimension::Replicated,
            &[BatchAxis::new(0), BatchAxis::replicated()],
            ProgramBatchingOutputAxesPolicy::Natural,
        )?;
        assert_eq!(batched.output_axes(), &[BatchAxis::new(0), BatchAxis::replicated()]);
        let (dynamic, _) = batched.into_parts();
        assert_eq!(
            dynamic.to_string(),
            indoc! {"
                lambda %0:dimension<batch ∈ [1, 9)>, %1:f32[batch, 3], %2:dimension<shared ∈ [0, 17)> .
                let %3:f32[batch, 3] = neg %1
                in (%0, %3, %2)
            "}
            .trim_end(),
        );

        // Structural callers may force each output axis independently. Alignment happens while the replayed values are
        // still live tracers, and the extent bookkeeping stays separate from the reported axis metadata.
        let forced = program.batched_with_threaded_extent(
            DimensionType::from(batch.clone()),
            ShardingDimension::Replicated,
            &[BatchAxis::new(0), BatchAxis::replicated()],
            ProgramBatchingOutputAxesPolicy::AlignEachTo(vec![BatchAxis::new(1), BatchAxis::replicated()]),
        )?;
        assert_eq!(forced.output_axes(), &[BatchAxis::new(1), BatchAxis::replicated()]);
        assert_eq!(
            forced.into_parts().0.output_types(),
            vec![
                ArrayIrType::Dimension(DimensionType::from(batch.clone())),
                ArrayIrType::Array(ArrayType::new(
                    DataType::F32,
                    Shape::new(vec![Dimension::Static(3), Dimension::Dynamic(batch.clone())]),
                )),
                ArrayIrType::Dimension(shared_dimension_type.clone()),
            ],
        );

        // A first-class dimension input cannot be declared mapped.
        assert_eq!(
            program
                .batched_with_threaded_extent(
                    DimensionType::from(batch),
                    ShardingDimension::Replicated,
                    &[BatchAxis::new(0), BatchAxis::new(0)],
                    ProgramBatchingOutputAxesPolicy::Natural,
                )
                .err(),
            Some(BatchingError::MappedDimension { r#type: Box::new(shared_dimension_type), axis: BatchAxis::new(0) }),
        );

        // An exact static extent uses the identical threaded boundary contract and instruction count. Only the boundary
        // types differ, so the structural IR neither grows with nor specializes on the mapped extent's runtime value.
        let extent_type = DimensionValue::constant(2)?.r#type().into_owned();
        let (r#static, _) = program
            .batched_with_threaded_extent(
                extent_type,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0), BatchAxis::replicated()],
                ProgramBatchingOutputAxesPolicy::Natural,
            )?
            .into_parts();
        assert_eq!(
            r#static.to_string(),
            indoc! {"
                lambda %0:dimension<2>, %1:f32[2, 3], %2:dimension<shared ∈ [0, 17)> .
                let %3:f32[2, 3] = neg %1
                in (%0, %3, %2)
            "}
            .trim_end(),
        );
        assert_eq!(r#static.instructions().len(), dynamic.instructions().len());
        Ok(())
    }

    #[test]
    fn test_program_batched_with_threaded_extent_threads_array_ir_region_boundary() -> Result<(), ProgramError> {
        // Array IR control flow batched under a replicated predicate keeps one structural instruction whose regions
        // are batched programs with the threaded extent as leading state, and the reported output axes exclude the
        // bookkeeping value. Branch alignment, per-item selection, and loop semantics are covered by the operations'
        // own tests; this test pins the boundary contract that this module threads through them.
        let unbatched_array_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)]));
        let shared_dimension_type = DimensionType::new("shared", DimensionBounds::new(0, Some(17))?);
        let mut branch_builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let branch_array = branch_builder.add_input(unbatched_array_type.clone().into());
        let branch_dimension = branch_builder.add_input(shared_dimension_type.clone().into());
        let branch_array = branch_builder.add_instruction(
            ArrayIrOperation::Array(ArrayOperation::Neg(NegOperation::new())),
            Vec::new(),
            vec![branch_array],
            None,
        )?[0];
        let branch = branch_builder.build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            vec![branch_array, branch_dimension],
            vec![Placeholder, Placeholder],
            vec![Placeholder, Placeholder],
        )?;

        let trace = ArrayIrTraceContext::new();
        let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(9))?);
        let batch_extent = trace.input(DimensionType::from(batch.clone()).into());
        let predicate = trace.input(ArrayType::scalar(DataType::Boolean).into());
        let batched_array_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(batch.clone()), Dimension::Static(3)]));
        let array = trace.input(batched_array_type.clone().into());
        let shared_dimension = trace.input(shared_dimension_type.clone().into());
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(trace.clone(), batch_extent);
        let outputs = context.bind(
            ArrayIrOperation::Condition(ConditionOperation::new()),
            vec![branch.clone(), branch],
            &[
                BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(predicate)),
                BatchingTracer::new(context.clone(), ArrayIrBatch::new(array, BatchAxis::new(0))?),
                BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(shared_dimension)),
            ],
        )?;
        assert_eq!(
            outputs.iter().map(|output| output.batch().batch_axis()).collect::<Vec<_>>(),
            vec![BatchAxis::new(0), BatchAxis::replicated()],
        );
        let output_ids =
            outputs.iter().map(|output| output.batch().value().atom_id()).collect::<Result<Vec<_>, _>>()?;
        let program = trace.builder().borrow().clone().build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            output_ids,
            vec![Placeholder; 4],
            vec![Placeholder; 2],
        )?;
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:dimension<batch ∈ [1, 9)>, %1:bool[], %2:f32[batch, 3], %3:dimension<shared ∈ [0, 17)> .
                let %4:f32[batch, 3], %5:dimension<shared ∈ [0, 17)> = condition %1 %0 %2 %3 [
                    true={
                        lambda %0:dimension<batch ∈ [1, 9)>, %1:f32[batch, 3], %2:dimension<shared ∈ [0, 17)> .
                        let %3:f32[batch, 3] = neg %1
                        in (%3, %2)
                    },
                    false={
                        lambda %0:dimension<batch ∈ [1, 9)>, %1:f32[batch, 3], %2:dimension<shared ∈ [0, 17)> .
                        let %3:f32[batch, 3] = neg %1
                        in (%3, %2)
                    },
                ]
                in (%4, %5)
            "}
            .trim_end(),
        );
        let shared_dimension_value = DimensionValue::new(shared_dimension_type.clone(), 4)?;
        assert_eq!(
            program.interpret(vec![
                ArrayIrValue::Dimension(DimensionValue::new(DimensionType::from(batch.clone()), 2)?),
                ArrayIrValue::Array(Array::scalar(true).unwrap()),
                ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap()),
                ArrayIrValue::Dimension(shared_dimension_value.clone()),
            ])?,
            vec![
                ArrayIrValue::Array(Array::matrix(2, 3, vec![-1.0_f32, -2.0, -3.0, -4.0, -5.0, -6.0]).unwrap()),
                ArrayIrValue::Dimension(shared_dimension_value),
            ],
        );

        // A second structural pass over the already-batched program introduces one new leading threaded extent and
        // recursively re-batches the attached regions. The source extent stays an ordinary replicated dimension input,
        // so nested batching never recovers either extent from array metadata.
        let outer_batch = DimensionVariable::new("outer_batch", DimensionBounds::new(1, Some(5))?);
        let nested = program.batched_with_threaded_extent(
            DimensionType::from(outer_batch.clone()),
            ShardingDimension::Replicated,
            &[BatchAxis::replicated(), BatchAxis::replicated(), BatchAxis::new(0), BatchAxis::replicated()],
            ProgramBatchingOutputAxesPolicy::Natural,
        )?;
        assert_eq!(nested.output_axes(), &[BatchAxis::new(0), BatchAxis::replicated()]);
        let (nested, _) = nested.into_parts();
        assert_eq!(
            nested.to_string(),
            indoc! {"
                lambda %0:dimension<outer_batch ∈ [1, 5)>, %1:dimension<batch ∈ [1, 9)>, %2:bool[], \
                    %3:f32[outer_batch, batch, 3], %4:dimension<shared ∈ [0, 17)> .
                let %5:f32[outer_batch, batch, 3], %6:dimension<shared ∈ [0, 17)> = condition %2 %0 %1 %3 %4 [
                    true={
                        lambda %0:dimension<outer_batch ∈ [1, 5)>, %1:dimension<batch ∈ [1, 9)>, \
                            %2:f32[outer_batch, batch, 3], %3:dimension<shared ∈ [0, 17)> .
                        let %4:f32[outer_batch, batch, 3] = neg %2
                        in (%4, %3)
                    },
                    false={
                        lambda %0:dimension<outer_batch ∈ [1, 5)>, %1:dimension<batch ∈ [1, 9)>, \
                            %2:f32[outer_batch, batch, 3], %3:dimension<shared ∈ [0, 17)> .
                        let %4:f32[outer_batch, batch, 3] = neg %2
                        in (%4, %3)
                    },
                ]
                in (%0, %5, %6)
            "}
            .trim_end(),
        );
        Ok(())
    }

    #[test]
    fn test_program_batched_with_threaded_extent_after_reference_discharge() {
        let scalar_type = ArrayType::scalar(DataType::F32);
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(scalar_type.into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, input], None)
            .unwrap();
        let snapshot =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let frozen =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![snapshot, frozen],
                vec![Placeholder],
                vec![Placeholder; 2],
            )
            .unwrap();

        // Discharge functionalizes the local root, and ordinary structural batching consumes the result: the batched
        // program is pure and reference-free while the mapped extent remains the policy's one bookkeeping output.
        // Batching the stateful program directly keeps its reference state instead of functionalizing it.
        //
        // The local allocation takes the mapped input's axis, the accumulate and read operate per item at that axis,
        // and freezing carries it out, so batching the stateful program directly agrees with batching its discharged
        // counterpart while keeping its reference state instead of functionalizing it.
        let axis_extent = DimensionValue::constant(2).unwrap();
        let extent_type = axis_extent.r#type().into_owned();
        let direct = program
            .batched_with_threaded_extent(
                extent_type.clone(),
                ShardingDimension::Replicated,
                &[BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap();
        let discharged = program
            .discharge_references(0)
            .unwrap()
            .into_program_without_external_references()
            .unwrap()
            .batched_with_threaded_extent(
                extent_type,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap();
        assert_eq!(direct.output_axes(), &[BatchAxis::new(0), BatchAxis::new(0)]);
        assert_eq!(discharged.output_axes(), direct.output_axes());
        let (direct, _) = direct.into_parts();
        let (discharged, _) = discharged.into_parts();
        assert!(direct.regions().iter().flat_map(|region| region.atoms()).any(|atom| atom.r#type().is_reference()));
        assert!(discharged.effects().classes().is_empty());
        assert!(
            discharged
                .regions()
                .iter()
                .flat_map(|region| region.atoms())
                .all(|atom| !atom.r#type().is_reference())
        );
        let inputs = vec![
            ArrayIrValue::Dimension(axis_extent.clone()),
            ArrayIrValue::Array(Array::vector(vec![3.0_f32, 4.0]).unwrap()),
        ];
        let expected = vec![
            ArrayIrValue::Dimension(axis_extent),
            ArrayIrValue::Array(Array::vector(vec![6.0_f32, 8.0]).unwrap()),
            ArrayIrValue::Array(Array::vector(vec![6.0_f32, 8.0]).unwrap()),
        ];
        assert_eq!(direct.interpret(inputs.clone()), Ok(expected.clone()));
        assert_eq!(discharged.interpret(inputs), Ok(expected));
    }

    #[test]
    fn test_program_batched_with_threaded_extent_reads_mapped_reference_input() {
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let reference = builder.add_input(ReferenceType::new(ArrayType::scalar(DataType::F32)).into());
        let output =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        // A reference input mapped at axis 0 becomes a reference over the batched referent, and reading it yields the
        // per-item values at that same axis without discharging the program first.
        let axis_extent = DimensionValue::constant(2).unwrap();
        let batched = program
            .batched_with_threaded_extent(
                axis_extent.r#type().into_owned(),
                ShardingDimension::Replicated,
                &[BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap();
        assert_eq!(batched.output_axes(), &[BatchAxis::new(0)]);
        let (batched, _) = batched.into_parts();
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:dimension<2>, %1:ref<f32[2]> .
                let %2:f32[2] = reference_read %1
                in (%0, %2)"},
        );

        // The eager entry point infers the mapped extent from the reference's statically shaped referent and reads the
        // per-item values through the same rule.
        let read: Result<ArrayIrValue<Array>, BatchingError> = batch(
            |reference| {
                Ok(reference
                    .context()
                    .bind(ReferenceReadOperation::new(), Vec::new(), std::slice::from_ref(&reference))?
                    .remove(0))
            },
            ArrayIrValue::Reference(ArrayReference::new(Array::vector(vec![3.0_f32, 4.0]).unwrap())),
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        );
        assert_eq!(read, Ok(ArrayIrValue::Array(Array::vector(vec![3.0_f32, 4.0]).unwrap())));
    }

    #[test]
    fn test_program_batched_with_threaded_extent_reference_output_carries_axis() {
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let reference = builder.add_input(ReferenceType::new(ArrayType::new_static(DataType::F32, [3])).into());
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![reference],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        // A reference-typed output carries the batch axis of the root it forwards: the batched referent gains the
        // extent at that axis and the output axis metadata reports it.
        let extent_type = DimensionValue::constant(2).unwrap().r#type().into_owned();
        let batched = program
            .batched_with_threaded_extent(
                extent_type.clone(),
                ShardingDimension::Replicated,
                &[BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap();
        assert_eq!(batched.output_axes(), &[BatchAxis::new(0)]);
        assert_eq!(
            batched.into_parts().0.output_types()[1],
            ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [2, 3]))),
        );

        // Requesting the axis the reference already carries is an identity alignment, while any other request is
        // rejected because a reference's batch axis is fixed by its referent.
        let aligned = program
            .batched_with_threaded_extent(
                extent_type.clone(),
                ShardingDimension::Replicated,
                &[BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::AlignAllTo(Axis::from(0)),
            )
            .unwrap();
        assert_eq!(aligned.output_axes(), &[BatchAxis::new(0)]);
        assert_eq!(
            program
                .batched_with_threaded_extent(
                    extent_type,
                    ShardingDimension::Replicated,
                    &[BatchAxis::new(0)],
                    ProgramBatchingOutputAxesPolicy::AlignAllTo(Axis::from(1)),
                )
                .err(),
            Some(BatchingError::UnsupportedOperation {
                message: "a reference's batch axis is fixed by its referent, so a reference batch carrying axis 0 \
                          cannot be aligned to axis 1; pass the reference as a batched input at that axis instead"
                    .to_string(),
            }),
        );
    }

    #[test]
    fn test_batching_context_forward_to_parent() {
        // Under an eager parent, forwarding applies the operation to the packed values, and each output keeps the batch
        // axis of the input at the same position.
        let context =
            BatchingContext::<_, ArrayBatchingPolicy>::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 2);
        let input =
            ArrayBatch::new(Array::matrix(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(), BatchAxis::new(1))
                .unwrap();
        assert_eq!(
            context.forward_to_parent(ArrayOperation::Neg(NegOperation::new()), std::slice::from_ref(&input)),
            Ok(vec![
                ArrayBatch::new(
                    Array::matrix(3, 2, vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]).unwrap(),
                    BatchAxis::new(1)
                )
                .unwrap(),
            ]),
        );

        // Under a staging parent, the operation is bound unchanged and keeps the mapped batch axis.
        let trace = TracingContext::<Array, ArrayOperation<Array>>::new();
        let input =
            ArrayBatch::new(trace.input(ArrayType::new_static(DataType::F32, [2, 3])), BatchAxis::new(0)).unwrap();
        let context = BatchingContext::<_, ArrayBatchingPolicy>::new(trace.clone(), 2);
        let outputs = context
            .forward_to_parent(ArrayOperation::Neg(NegOperation::new()), std::slice::from_ref(&input))
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), input.batch_axis());
        assert_eq!(outputs[0].value().r#type().as_ref(), &ArrayType::new_static(DataType::F32, [2, 3]));
        assert_eq!(trace.builder().borrow().instructions().len(), 1);
        assert!(matches!(trace.builder().borrow().instructions()[0].operation(), ArrayOperation::Neg(_)));
    }

    #[test]
    fn test_sharding_with_leading_batch_axis() {
        // Lifting an explicit per-item sharding moves a manual mapped axis out of the varying set and onto the new
        // physical batch dimension.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let per_item_sharding =
            Sharding::new(mesh.clone(), vec![ShardingDimension::replicated(), ShardingDimension::replicated()])
                .unwrap()
                .with_varying_manual_axes(["x"])
                .unwrap();
        assert_eq!(
            per_item_sharding.with_leading_batch_axis(ShardingDimension::sharded(["x"])),
            Ok(Sharding::new(
                mesh,
                vec![
                    ShardingDimension::sharded(["x"]),
                    ShardingDimension::replicated(),
                    ShardingDimension::replicated(),
                ],
            )
            .unwrap()),
        );
    }

    #[test]
    fn test_sharding_batched() {
        // The batch axis takes the requested placement. Placing it on a manual mesh axis distributes the batch items
        // across that axis's shards, so only that placement makes the value vary; an out-of-range position is rejected.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("data", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["data"])]).unwrap();
        for (axis_sharding, expected_varying) in [
            (ShardingDimension::sharded(["x"]), vec!["x".to_string()]),
            (ShardingDimension::replicated(), Vec::new()),
            (ShardingDimension::unconstrained(), Vec::new()),
        ] {
            let batched = sharding.batched(1, axis_sharding.clone()).unwrap();
            assert_eq!(batched.dimensions(), &[ShardingDimension::sharded(["data"]), axis_sharding]);
            assert_eq!(batched.varying_manual_axes().iter().cloned().collect::<Vec<_>>(), expected_varying);
        }
        assert!(matches!(
            sharding.batched(2, ShardingDimension::replicated()),
            Err(BatchingError::MisalignedBatchAxes { .. }),
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
    fn test_array_type_batched() {
        // The array-level form inserts the dimension and threads the sharding through the same rule as
        // `Sharding::batched`, so a manual placement of the batch axis records the variation on the batched type.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("data", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["data"])]).unwrap();
        let r#type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]))
            .with_sharding(sharding.clone())
            .unwrap();
        let batched = r#type.batched(0, Dimension::Static(2), ShardingDimension::sharded(["x"])).unwrap();
        assert_eq!(batched.shape(), &Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        assert_eq!(batched.sharding(), Some(&sharding.batched(0, ShardingDimension::sharded(["x"])).unwrap()));

        // An unsharded type only gains the dimension, and an out-of-range position is rejected.
        let unsharded = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));
        assert_eq!(
            unsharded.batched(1, Dimension::Static(2), ShardingDimension::replicated()),
            Ok(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3), Dimension::Static(2)]))),
        );
        assert_eq!(
            unsharded.batched(2, Dimension::Static(2), ShardingDimension::replicated()),
            Err(BatchingError::Type(TypeError::Invalid {
                message: "cannot insert dimension at index 2 for rank-1 array type".to_string(),
            })),
        );
    }

    #[test]
    fn test_array_type_unbatched() {
        // The per-item type removes the mapped axis (normalized against the packed rank); a replicated batch axis
        // leaves the type unchanged.
        let r#type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        assert_eq!(r#type.unbatched(BatchAxis::replicated()), Ok(r#type.clone()));
        assert_eq!(r#type.unbatched(0), Ok(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]))));
        assert_eq!(
            r#type.unbatched(BatchAxis::new(-1)),
            Ok(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)]))),
        );

        // Removing a batch axis placed on a manual mesh axis keeps the variation along that axis on the per-item type,
        // since the items still vary across the shards, while the placement of the removed dimension itself is gone.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("data", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["data"])]).unwrap();
        let r#type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]))
            .with_sharding(sharding.clone())
            .unwrap();
        let batched = r#type.batched(0, Dimension::Static(2), ShardingDimension::sharded(["x"])).unwrap();
        assert_eq!(
            batched.unbatched(0),
            Ok(r#type.with_sharding(sharding.with_varying_manual_axes(["x"]).unwrap()).unwrap()),
        );
    }

    #[test]
    fn test_batch_axis_sharding() {
        // Only mapped inputs with explicit sharding constrain the new batch axis. Placements join from least to most
        // specific (unconstrained and replicated yield to a concrete sharding, equal placements agree) independently of
        // input order, and with no contributor the batch axis is replicated.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let unsharded = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let sharded = unsharded
            .clone()
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                    .unwrap(),
            )
            .unwrap();
        let replicated = unsharded.clone().with_sharding(Sharding::replicated(mesh.clone(), 2)).unwrap();
        let unconstrained = unsharded
            .clone()
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::unconstrained(), ShardingDimension::replicated()])
                    .unwrap(),
            )
            .unwrap();
        let differently_sharded = unsharded
            .clone()
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["y"]), ShardingDimension::replicated()])
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(batch_axis_sharding::<&ArrayType, _>([]), Ok(ShardingDimension::replicated()));
        assert_eq!(batch_axis_sharding([(&unsharded, Some(0))]), Ok(ShardingDimension::replicated()));
        assert_eq!(batch_axis_sharding([(&sharded, None)]), Ok(ShardingDimension::replicated()));
        assert_eq!(batch_axis_sharding([(&sharded, Some(0))]), Ok(ShardingDimension::sharded(["x"])));
        assert_eq!(
            batch_axis_sharding([(&replicated, Some(0)), (&sharded, Some(0))]),
            Ok(ShardingDimension::sharded(["x"])),
        );
        assert_eq!(
            batch_axis_sharding([(&sharded, Some(0)), (&unconstrained, Some(0))]),
            Ok(ShardingDimension::sharded(["x"])),
        );
        assert_eq!(
            batch_axis_sharding([(&unconstrained, Some(0)), (&replicated, Some(0))]),
            Ok(ShardingDimension::replicated()),
        );

        // Two distinct concrete placements are ambiguous, and placements are comparable only within one mesh.
        assert!(matches!(
            batch_axis_sharding([(&sharded, Some(0)), (&differently_sharded, Some(0))]),
            Err(error) if error.to_string() == "mismatched batch axis sharding: {'x'} vs {'y'}",
        ));
        let other_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let other = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]))
            .with_sharding(
                Sharding::new(other_mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                    .unwrap(),
            )
            .unwrap();
        assert!(matches!(
            batch_axis_sharding([(&sharded, Some(0)), (&other, Some(0))]),
            Err(error) if error.to_string().starts_with("mismatched batch axis sharding meshes: "),
        ));
    }

    #[test]
    fn test_normalized_batch_axis_type() {
        // Normalization replaces only the mapped axis's placement with the batch axis placement, keeps the other
        // dimensions, and adds the manual axes the new placement introduces to the varying set.
        // It reports `None` when there is nothing to change: no sharding metadata, or a placement that already agrees.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let unsharded = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4), Dimension::Static(6)]));
        assert_eq!(normalized_batch_axis_type(&unsharded, 0, &ShardingDimension::sharded(["x"])), Ok(None));
        let sharding =
            Sharding::new(mesh.clone(), vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])])
                .unwrap();
        let input = unsharded.with_sharding(sharding.clone()).unwrap();
        assert_eq!(normalized_batch_axis_type(&input, 0, &ShardingDimension::replicated()), Ok(None));
        assert_eq!(
            normalized_batch_axis_type(&input, 0, &ShardingDimension::sharded(["m"])),
            Ok(Some(
                input
                    .clone()
                    .with_sharding(
                        Sharding::new(mesh, vec![ShardingDimension::sharded(["m"]), ShardingDimension::sharded(["x"])])
                            .unwrap()
                            .with_varying_manual_axes(["m"])
                            .unwrap(),
                    )
                    .unwrap(),
            )),
        );
    }

    #[test]
    fn test_normalized_batch_axis_type_is_not_a_reshard() {
        // Drift gate for the rejected proposal to stage batch-axis sharding normalization as a `reshard` instead of an
        // axis-identity broadcast. Reshard preserves input variation and rejects manual or auto placement targets;
        // it cannot describe the ownership introduced by materializing a globally sharded mapped dimension.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("a", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap();

        // Normalizing a replicated mapped axis onto a manual placement also makes the value vary along that manual
        // axis. Such a manual placement is not a legal explicit reshard target.
        let input = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(Sharding::new(mesh.clone(), vec![ShardingDimension::replicated()]).unwrap())
            .unwrap();
        let normalized = normalized_batch_axis_type(&input, 0, &ShardingDimension::sharded(["m"])).unwrap().unwrap();
        assert_eq!(
            normalized.sharding().unwrap().varying_manual_axes().iter().collect::<Vec<_>>(),
            vec!["m"],
            "normalization adds the manual axis introduced by the new placement",
        );
        assert_eq!(
            ReshardOperation::new(normalized.sharding().unwrap().clone())
                .infer_output_types(std::slice::from_ref(&input), &[]),
            Err(TypeError::invalid(
                "`reshard` cannot target manual mesh axes; use manual collectives for transitions over them",
            )),
        );

        // A non-batch dimension placed on an auto mesh axis passes through normalization untouched, but the resulting
        // sharding is not a legal reshard target at all, so the swap would turn a placement relabel into a hard error.
        let input = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4), Dimension::Static(6)]))
            .with_sharding(
                Sharding::new(mesh, vec![ShardingDimension::replicated(), ShardingDimension::sharded(["a"])]).unwrap(),
            )
            .unwrap();
        let normalized = normalized_batch_axis_type(&input, 0, &ShardingDimension::sharded(["m"])).unwrap().unwrap();
        assert_eq!(
            ReshardOperation::new(normalized.sharding().unwrap().clone())
                .infer_output_types(std::slice::from_ref(&input), &[]),
            Err(TypeError::invalid(
                "`reshard` cannot target auto mesh axes; use `constrain_sharding` to constrain placement over them"
                    .to_string(),
            )),
        );
    }

    #[test]
    fn test_broadcast_replicated_array() -> Result<(), ProgramError> {
        // The output shape is the per-item shape with the extent inserted at the requested position: static per-item
        // axes fold to exact constants, dynamic ones are read back from the value, the input-to-output axis mapping
        // shifts the axes past the insertion point, and the sharding gains the batch axis placement at that position.
        let trace = ArrayIrTraceContext::new();
        let items = DimensionVariable::new("items", DimensionBounds::new(1, Some(9))?);
        let width = DimensionVariable::new("width", DimensionBounds::new(1, Some(9))?);
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let r#type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3), Dimension::Dynamic(width)]))
            .with_sharding(Sharding::replicated(mesh.clone(), 2))
            .unwrap();
        let extent = trace.input(DimensionType::from(items).into());
        let value = trace.input(r#type.clone().into());
        let output = broadcast_replicated_array(value, &r#type, 1, &extent, &ShardingDimension::sharded(["x"]))?;
        let program = trace.builder().borrow().clone().build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            vec![output.atom_id()?],
            vec![Placeholder, Placeholder],
            vec![Placeholder],
        )?;
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:dimension<items ∈ [1, 9)>, %1:f32[3, width][sharding={mesh<['x'=2:explicit]>, [{}, {}]}] .
                let %2:dimension<3> = constant [value=3]
                    %3:dimension<width ∈ [1, 9)> = dimension_size [axis=1] %1
                    %4:f32[3, items, width][sharding={mesh<['x'=2:explicit]>, [{}, {'x'}, {}]}] = broadcast [\
                        output_axes=[0, 2], output_sharding={mesh<['x'=2:explicit]>, [{}, {'x'}, {}]}] %1 %2 %0 %3
                in (%4)
            "}
            .trim_end(),
        );
        Ok(())
    }
}
