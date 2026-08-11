//! Array-domain batching representations, policies, and mixed-array specialization machinery. The top-level
//! [`batching`](crate::batching) module owns the universe-neutral transform protocol, contexts, drivers, and entry
//! points. This module supplies that protocol's concrete implementations for [`ArrayType`] and [`ArrayIrType`]. Arrays
//! use the ordinary [`ArrayBatch`] representation, while first-class dimensions are shared shape values and therefore
//! remain replicated across the batch, and mixed operations explicitly state how they cross that boundary.

use std::borrow::Cow;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use std::rc::Rc;

use ryft_macros::Parameter;

use crate::arrays::broadcasting::Broadcastable;
use crate::arrays::dimensions::DimensionValue;
use crate::arrays::sharding::ShardingError;
use crate::arrays::sharding::meshes::MeshAxisType;
use crate::arrays::sharding::shardings::{Sharding, ShardingDimension};
use crate::arrays::types::arrays::ArrayType;
use crate::arrays::types::dimensions::{Dimension, DimensionType, DimensionVariable, Shape};
use crate::arrays::types::ir::ArrayIrType;
use crate::axes::{Axis, NamedAxes, NamedAxis};
use crate::batching::{
    BatchAxis, BatchAxisSpecification, BatchableOperation, BatchableType, BatchedOutputs, BatchedProgram,
    BatchingContext, BatchingDriver, BatchingEntrypointPolicy, BatchingError, BatchingPolicy, BatchingPolicyProjection,
    BatchingTracer, BoundaryPreservingBatchedProgram, InterpretableBatchableOperation, ProgramBatchingOutputAxesPolicy,
    RecursiveBatchingDriver, RecursiveBatchingPolicy,
};
use crate::contexts::{Context, EagerContext, ProjectedContext, StagingContext, ValueResolution};
use crate::interpretation::InterpretableOperation;
use crate::macros::{check_builders, check_count};
use crate::operations::{
    AndOperation, Broadcast, BroadcastOperation, CompareOperation, ComparisonDirection, ConstantOperation,
    DimensionRequirementOperation, DimensionSizeOperation, DynamicBroadcastOperation, ElementwiseOperation,
    IotaOperation, ReductionKind, SelectOperation, Transpose, TransposeOperation, ZeroLikeOperation,
};
use crate::parameters::{Parameter, Placeholder};
use crate::programs::{
    Operation, OperationProjection, Program, ProgramError, ProjectedValue, RegionRef, RegionReplayMappings,
    ReplayRegionDriver, Type, TypeError, Typed, Value, ValueProjection,
};
use crate::tracing::{Tracer, TracingContext};

/// Describes one bounded ragged axis of an [`ArrayBatch`]. `axis` identifies the physical packed axis whose storage
/// extent is the declared upper bound of `dimension`, while `extents` contains the actual per-item extents. The
/// [`DimensionVariable`] is the semantic identity shared by compatible ragged operands. This metadata exists only while
/// batching (i.e., it is not a [`Type`] variant and consumers such as reductions own the masks that distinguish live
/// elements from padding).
#[derive(Clone, Debug, PartialEq, Parameter)]
pub struct RaggedAxis<V> {
    /// Physical axis in the packed array value.
    axis: usize,

    /// Value containing the per-item extents of this [`RaggedAxis`].
    extents: V,

    /// Dynamic dimension represented by `extents`.
    dimension: DimensionVariable,

    /// Mapping from each axis of [`Self::extents`] into the packed data array.
    extent_axes: Vec<usize>,
}

impl<V> RaggedAxis<V> {
    /// Creates ragged-axis metadata claiming that packed axis `axis` is logically ragged: `extents` holds its true
    /// per-item logical extents as an ordinary integer array value, `dimension` is the dynamic [`DimensionVariable`]
    /// those extents instantiate, and `extent_axes` maps each axis of `extents` onto the packed batch axis it indexes.
    /// This claim is not verifiable at construction; refer to the documentation of [`ArrayBatch::with_ragged_axes`]
    /// for the invariants it carries.
    #[inline]
    pub fn new(axis: usize, extents: V, dimension: DimensionVariable, extent_axes: Vec<usize>) -> Self {
        Self { axis, extents, dimension, extent_axes }
    }

    /// Returns the physical packed-array axis.
    #[inline]
    pub fn axis(&self) -> usize {
        self.axis
    }

    /// Returns the per-item extents of this [`RaggedAxis`] as an ordinary integer array _value_ in the carrier's own
    /// value universe (a concrete array during eager batching, a [`Tracer`] while staging, and the outer carrier's
    /// value under nested batching), never a host-side list of sizes. Data-derived extents are device-born, so
    /// materializing them on the host would require a stream-stalling readback, and during tracing the sizes do not
    /// exist yet; only this staged value does. Rules use it directly (e.g., masking stages comparisons against it),
    /// and it is itself batchable data, whose axes [`extent_axes`](Self::extent_axes) maps into the packed array.
    #[inline]
    pub fn extents(&self) -> &V {
        &self.extents
    }

    /// Returns the dynamic dimension represented by this ragged axis.
    #[inline]
    pub fn dimension(&self) -> &DimensionVariable {
        &self.dimension
    }

    /// Returns the mapping from each axis of [`Self::extents`] into the packed data array. These axes index the batch
    /// segments whose lengths the extent array stores and make nested batching explicit without changing array types.
    #[inline]
    pub fn extent_axes(&self) -> &[usize] {
        self.extent_axes.as_slice()
    }

    /// Returns this [`RaggedAxis`] after applying the same single-axis move as its packed array. When the carrier
    /// relocates the packed axis at `source` to `destination` (i.e., removing it and reinserting it, as
    /// [`move_axis`](Transpose::move_axis) does during batch-axis realignment), every stored packed-axis position must
    /// be renumbered under that move. A position at `source` follows it to `destination`, positions between the two
    /// shift by one toward the vacated slot, and all other positions are unchanged. This applies both to
    /// [`axis`](Self::axis), where the ragged data lives, and to every entry of [`extent_axes`](Self::extent_axes),
    /// which are also packed-axis positions, keeping the metadata aligned with the moved data.
    fn moved(mut self, source: usize, destination: usize) -> Self {
        let move_axis = |axis: usize| {
            if axis == source {
                destination
            } else if source < destination && axis > source && axis <= destination {
                axis - 1
            } else if destination < source && axis >= destination && axis < source {
                axis + 1
            } else {
                axis
            }
        };
        self.axis = move_axis(self.axis);
        self.extent_axes.iter_mut().for_each(|axis| *axis = move_axis(*axis));
        self
    }

    /// Returns this [`RaggedAxis`] after its packed array has flowed through a broadcast. A broadcast relocates each
    /// operand axis `i` to `output_axes[i]` in its result, with newly inserted result axes appearing nowhere in that
    /// mapping, so every stored packed-axis position (i.e., [`axis`](Self::axis) and each entry of
    /// [`extent_axes`](Self::extent_axes)) is remapped through the same table. `output_axes` must be the broadcast's
    /// complete operand-to-result axis mapping, indexed by operand axis; its length equals the operand rank, so every
    /// stored position is covered by construction.
    fn broadcasted(mut self, output_axes: &[usize]) -> Self {
        self.axis = output_axes[self.axis];
        self.extent_axes.iter_mut().for_each(|axis| *axis = output_axes[*axis]);
        self
    }

    /// Returns this [`RaggedAxis`] after the provided packed axes have been removed by a reduction, or [`None`] when
    /// the ragged axis itself is among them as reducing the ragged axis consumes the raggedness (e.g., a masked sum
    /// over it produces a per-item scalar), so no metadata survives it. Surviving positions (i.e., [`axis`](Self::axis)
    /// and each entry of [`extent_axes`](Self::extent_axes)) shift down by the number of removed axes that preceded
    /// them. `reduced_axes` must not contain any extent axis as extent axes index batch items and the masked reduction
    /// rules never reduce a batch axis, so a violation would mean the caller discards the per-item extents while
    /// keeping metadata that references them.
    pub(crate) fn reduced(mut self, reduced_axes: &[usize]) -> Option<Self> {
        if reduced_axes.contains(&self.axis) {
            return None;
        }
        self.axis -= reduced_axes.iter().filter(|axis| **axis < self.axis).count();
        self.extent_axes.iter_mut().for_each(|extent_axis| {
            *extent_axis -= reduced_axes.iter().filter(|axis| **axis < *extent_axis).count();
        });
        Some(self)
    }

    /// Returns this [`RaggedAxis`] relocated onto the result axes of an operation that keeps only some of its operand's
    /// axes and reorders the survivors (e.g., a contraction such as `dot`), or [`None`] when any stored packed-axis
    /// position does not survive. `output_axes` maps each operand axis to the result axis it becomes, indexed by
    /// operand axis, with [`None`] marking an axis the operation consumes, which makes it the partial counterpart of
    /// [`broadcasted`](Self::broadcasted)'s total mapping. Both [`axis`](Self::axis) and every entry of
    /// [`extent_axes`](Self::extent_axes) are remapped through that table, and losing either kind of position means
    /// the result can no longer describe this ragged axis, so the whole relocation fails instead of silently dropping
    /// metadata that references a vanished axis.
    pub(crate) fn relocated(mut self, output_axes: &[Option<usize>]) -> Option<Self> {
        self.axis = output_axes[self.axis]?;
        for extent_axis in &mut self.extent_axes {
            *extent_axis = output_axes[*extent_axis]?;
        }
        Some(self)
    }
}

/// Value with [`ArrayType`] type that represents a _packed_ batch of arrays. [`ArrayBatch`] is the batching
/// representation for Ryft's batching/vectorization transform over arrays. It pairs a packed array value with a
/// [`BatchAxis`] that marks which of its dimensions indexes the batch items. A value is either *batched* (i.e., its
/// packed type carries the batch dimension) or *replicated*, meaning that it is shared unchanged across every batch
/// item.
#[derive(Clone, Debug, PartialEq, Parameter)]
pub struct ArrayBatch<V> {
    /// Refer to the documentation of [`value`](Self::value) for more information.
    value: V,

    /// Refer to the documentation of [`batch_axis`](Self::batch_axis) for more information.
    batch_axis: BatchAxis,

    /// Bounded ragged axes of this [`ArrayBatch`].
    ragged_axes: Vec<RaggedAxis<V>>,
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
        let (_, batch_axis) = value.r#type().unbatched_type_and_axis::<V>(batch_axis.into(), &[])?;
        Ok(Self { value, batch_axis, ragged_axes: Vec::new() })
    }

    /// Returns this [`ArrayBatch`] carrying the provided bounded ragged axes in place of any it already carried, so
    /// that its logical per-item type keeps a dynamic [`Dimension`] wherever the packed value stores a finite
    /// physical bound.
    ///
    /// Structural requirements are validated here. Every ragged axis must name an ordinary packed axis of the value
    /// (positions index the packed type), must not name the mapped batch axis, and the logical per-item type must
    /// derive cleanly. However, the semantic content of each [`RaggedAxis`] is a claim that this method cannot verify
    /// and that downstream consumers (e.g., masked reductions and boundary restoration) treat as truth: the
    /// [`extents`](RaggedAxis::extents) value must hold the axis's true per-item logical extents (each within the
    /// dynamic [`DimensionVariable`]'s declared bounds and never exceeding the packed physical bound) and
    /// [`extent_axes`](RaggedAxis::extent_axes) must map the extents array's axes onto the packed batch axes they
    /// index. Violating those invariants makes physical padding observable in results or masks live data away.
    pub fn with_ragged_axes(mut self, ragged_axes: Vec<RaggedAxis<V>>) -> Result<Self, BatchingError> {
        let value_type = self.value.r#type();
        let packed_type = value_type.as_ref();
        let batch_axis = self.batch_axis;
        let batch_axis_position = self.batch_axis_position();
        for ragged_axis in &ragged_axes {
            if ragged_axis.axis >= packed_type.rank() || Some(ragged_axis.axis) == batch_axis_position {
                return Err(BatchingError::InvalidBatchMetadata {
                    message: format!(
                        "ragged axis {} is invalid for packed array type {} with batch axis {}",
                        ragged_axis.axis, packed_type, batch_axis,
                    ),
                });
            }
        }

        // Deriving the per-item type once here is what lets `Self::unbatched_type` be infallible. The packed value
        // and the batch metadata are immutable afterward, so the derivation can never start failing later.
        packed_type.unbatched_type_and_axis(batch_axis, ragged_axes.as_slice())?;
        self.ragged_axes = ragged_axes;
        Ok(self)
    }

    /// Creates a new [`ArrayBatch`] that replicates the provided value across the batch.
    #[inline]
    pub fn replicated(value: V) -> Self {
        Self { value, batch_axis: BatchAxis::replicated(), ragged_axes: Vec::new() }
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
        self.batch_axis.axis().map(|axis| axis.normalize(self.r#type().rank()).unwrap())
    }

    /// Returns the bounded ragged axes of this [`ArrayBatch`].
    #[inline]
    pub fn ragged_axes(&self) -> &[RaggedAxis<V>] {
        self.ragged_axes.as_slice()
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

    /// Returns the logical per-item [`ArrayType`] of this batch (i.e., the packed type with the mapped batch dimension
    /// removed and the dynamic [`Dimension`] of every bounded ragged axis restored).
    #[inline]
    pub fn unbatched_type(&self) -> ArrayType {
        self.r#type().unbatched_type_and_axis(self.batch_axis, self.ragged_axes.as_slice()).unwrap().0
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
        V: Broadcast,
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
            .map_err(|_| BatchingError::BatchAxisOutOfBounds { r#type: Box::new(self.r#type().into_owned()), axis })?;

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

        let broadcasted = self.value().clone().broadcast(batched_type, output_axes.as_slice())?;
        ArrayBatch::new(broadcasted, axis)
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
            .normalize(self.r#type().rank())
            .map_err(|_| BatchingError::BatchAxisOutOfBounds { r#type: Box::new(self.r#type().into_owned()), axis })?;
        if current_axis == position {
            return Ok(self.clone());
        }
        let permuted_value = self.value().clone().move_axis(current_axis, position)?;
        let ragged_axes = self
            .ragged_axes
            .iter()
            .cloned()
            .map(|ragged_axis| ragged_axis.moved(current_axis, position))
            .collect();
        Self::new(permuted_value, axis)?.with_ragged_axes(ragged_axes)
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
        self.unbatched_type_and_axis::<()>(batch_axis.into(), &[]).map(|(r#type, _)| r#type)
    }

    /// Returns the unbatched per-item [`ArrayType`] together with the normalized [`BatchAxis`]. This derivation
    /// _defines_ what the logical type of a packed array batch is. It removes the mapped batch dimension, exactly as
    /// [`ArrayType::unbatched_type`] documents, and then restores the dynamic [`Dimension`] of every provided bounded
    /// ragged axis, because a packed ragged axis stores the finite physical bound shared by the whole batch while each
    /// item logically extends only as far as its own per-item extent. Ragged axis positions index the packed type. This
    /// internal form lets composite batch carriers validate and retain canonical axis metadata without cloning or
    /// otherwise projecting their array payloads.
    pub(crate) fn unbatched_type_and_axis<V>(
        &self,
        batch_axis: BatchAxis,
        ragged_axes: &[RaggedAxis<V>],
    ) -> Result<(Self, BatchAxis), BatchingError> {
        let (batch_axis, axis) = self.normalize_batch_axis(batch_axis)?;

        // The following closure represents the shared tail of the derivation. Both the replicated early return and the
        // mapped path below must replace each ragged axis's finite physical bound with its dynamic per-item dimension
        // after the mapped batch dimension (if any) has been removed, so that the returned per-item type is logical
        // rather than physical.
        let restore_ragged_dimensions = |mut unbatched_type: Self| {
            if !ragged_axes.is_empty() {
                let mut dimensions = unbatched_type.shape().dimensions().to_vec();
                for ragged_axis in ragged_axes {
                    // Ragged axis positions index the packed type, which additionally carries the mapped batch
                    // dimension.
                    let position = ragged_axis.axis - usize::from(axis.is_some_and(|axis| axis < ragged_axis.axis));
                    dimensions[position] = Dimension::Dynamic(ragged_axis.dimension.clone());
                }
                unbatched_type.shape = Shape::new(dimensions);
            }
            unbatched_type
        };

        let Some(axis) = axis else {
            return Ok((restore_ragged_dimensions(self.clone()), batch_axis));
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
            restore_ragged_dimensions(Self {
                data_type: self.data_type(),
                shape: Shape::new(dimensions),
                layout: None,
                sharding,
                memory: self.memory(),
            }),
            batch_axis,
        ))
    }
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
/// BoundaryPreservingBatchedProgram<C::Constant, C::Operation>`: homogeneous rules bind structurally batched branch
/// programs directly, so an array batching policy preserves the source boundary by definition and the supertrait
/// binding lets every rule rely on that without restating it. The supertrait also pins [`BatchingPolicy::Evidence`]
/// to the bounded ragged [`DimensionVariable`]s a rule consumed, so one shared array rule (e.g., the reduction rule
/// that reduces a ragged axis away) states that claim identically under every array policy. Currently, only
/// [`ArrayIrBatching`] reads the claim, because it is the only policy whose carriers can hold ragged axes in the
/// first place; the others always produce an empty claim. The shared rules are then written once against the nominal
/// [`ArrayBatching<P>`] family rather than as a `P: ArrayBatchingPolicy` blanket, because Rust coherence cannot use the
/// _absence_ of a trait implementation to prove such a blanket disjoint from the genuinely mixed composite operation
/// rules registered for other policies.
///
/// Keeping this capability on the batching transform, rather than on [`ProjectedContext`], [`ArrayBatch`], or [`Type`],
/// means that neither the carrier nor the type contract needs to know anything about dynamic-shape state that only
/// batching needs.
pub trait ArrayBatchingPolicy<C: Context<Type = ArrayType>>:
    BatchingPolicy<
        C,
        Batch = ArrayBatch<C::Value>,
        Evidence = Vec<DimensionVariable>,
        BatchedProgram = BoundaryPreservingBatchedProgram<C::Constant, C::Operation>,
    >
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

/// Ragged consumption surface of an [`ArrayBatchingPolicy`] which provides the per-operation-discipline hooks through
/// which batching rules consume bounded ragged axes instead of dropping them. Implementors must make padding along
/// every consumed ragged axis unobservable in the consuming operation's result, or reject inputs for which they cannot
/// do so safely. Policies whose carriers never contain ragged axes may return `input` unchanged. This capability is
/// separate from [`ArrayBatchingPolicy`] so unrelated array rules do not inherit the constructor, comparison, and
/// selection bounds needed only by ragged masking.
pub trait RaggedArrayBatchingPolicy<C: Context<Type = ArrayType>>: ArrayBatchingPolicy<C> {
    /// Replaces padding along the ragged axes in `reduced_axes` with `kind`'s reduction identity. This is the
    /// identity-masking consumption discipline, owned by reductions.
    fn mask_reduction_input(
        context: &BatchingContext<C, ArrayBatching<Self>>,
        input: &ArrayBatch<C::Value>,
        reduced_axes: &[usize],
        kind: ReductionKind,
    ) -> Result<ArrayBatch<C::Value>, BatchingError>;

    /// Replaces padding along the ragged axes in `contracted_axes` with zero. This is the zero-padding consumption
    /// discipline, owned by contractions (e.g., `dot`), and it is the companion of the identity-masking discipline
    /// that [`Self::mask_reduction_input`] owns for reductions. Zero is the contraction identity (a padded element
    /// enters a contraction only as a factor of a product that is then summed, so zeroing it removes its product from
    /// the sum). Consumers zero every ragged operand along its own contracted ragged axes, which is sufficient because
    /// zeroing either factor of a contracted pair already neutralizes that product, and it requires no agreement
    /// between the two operands about which of them is ragged.
    fn pad_contraction_input(
        context: &BatchingContext<C, ArrayBatching<Self>>,
        input: &ArrayBatch<C::Value>,
        contracted_axes: &[usize],
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
    type Evidence = Vec<DimensionVariable>;
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

impl<C: Context<Type = ArrayType, Value: Broadcast + Transpose>> ArrayBatchingPolicy<C> for StaticArrayBatchingPolicy {
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
        let broadcasted = input.value().clone().broadcast(r#type.clone(), output_axes.as_slice())?;
        let ragged_axes = input
            .ragged_axes()
            .iter()
            .cloned()
            .map(|ragged_axis| ragged_axis.broadcasted(output_axes.as_slice()))
            .collect();
        ArrayBatch::new(broadcasted, batch_axis)?.with_ragged_axes(ragged_axes)
    }
}

impl<C: Context<Type = ArrayType, Value: Broadcast + Transpose>> RaggedArrayBatchingPolicy<C>
    for StaticArrayBatchingPolicy
{
    #[inline]
    fn mask_reduction_input(
        _context: &BatchingContext<C, ArrayBatching<Self>>,
        input: &ArrayBatch<C::Value>,
        _reduced_axes: &[usize],
        _kind: ReductionKind,
    ) -> Result<ArrayBatch<C::Value>, BatchingError> {
        // Static array batching never creates bounded ragged axes. Its mapped extent is one host `usize` and every
        // alignment materializes a rectangular carrier, so masking is the identity. A ragged carrier can still reach
        // this hook because `ArrayBatch::with_ragged_axes` is public, and it is rejected rather than passed through,
        // because returning it unchanged would let the consuming reduction claim consumption evidence for padding it
        // never neutralized.
        if !input.ragged_axes().is_empty() {
            return Err(BatchingError::UnsupportedOperation {
                message: "static array batching cannot mask bounded ragged axes".to_string(),
            });
        }
        Ok(input.clone())
    }

    #[inline]
    fn pad_contraction_input(
        _context: &BatchingContext<C, ArrayBatching<Self>>,
        input: &ArrayBatch<C::Value>,
        _contracted_axes: &[usize],
    ) -> Result<ArrayBatch<C::Value>, BatchingError> {
        // Zeroing padding requires the same comparison and selection machinery that masking does, and static array
        // batching has neither the per-item extents nor a reason to (it never creates bounded ragged axes). A ragged
        // carrier reaching this hook through the public `ArrayBatch::with_ragged_axes` is therefore rejected rather
        // than passed through, because returning it unchanged would let the consuming contraction claim consumption
        // evidence for padding that still contributes to its result.
        if !input.ragged_axes().is_empty() {
            return Err(BatchingError::UnsupportedOperation {
                message: "static array batching cannot zero-pad bounded ragged axes".to_string(),
            });
        }
        Ok(input.clone())
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
    type Evidence = P::Evidence;
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
    fn validate_operation_outputs(
        operation_name: &'static str,
        inputs: &[Self::Batch],
        outputs: &[Self::Batch],
        evidence: &Self::Evidence,
    ) -> Result<(), BatchingError> {
        P::validate_operation_outputs(operation_name, inputs, outputs, evidence)
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

impl<C: Context<Type = ArrayType, Value: Broadcast + Transpose>> BatchingEntrypointPolicy<C> for ArrayBatching {
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
        if !output.ragged_axes().is_empty() {
            return Err(BatchingError::UnsupportedOperation {
                message: "a bounded ragged array cannot cross the batching transform output boundary".to_string(),
            });
        }
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
        + From<BroadcastOperation>,
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
                let (outputs, evidence) = instruction
                    .operation()
                    .batch(context, &RecursiveBatchingDriver::new(&regions), instruction_inputs)?
                    .into_parts();
                <Self as BatchingPolicy<C>>::validate_operation_outputs(
                    instruction.operation().name(),
                    instruction_inputs,
                    outputs.as_slice(),
                    &evidence,
                )?;
                Ok(outputs)
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
        region.batched_with_axis_name(
            *context.axis_extent(),
            context.axis_name().map(str::to_string),
            context.axis_sharding().clone(),
            input_axes,
            output_axes_policy,
        )
    }
}

impl BatchableType for ArrayType {
    type Policy = ArrayBatching;
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
    ) -> Result<BatchedOutputs<C, ArrayBatching<M>>, BatchingError> {
        // No input carries the batch axis. Interpret the inputs as given and report every output replicated.
        // Any per-item shape broadcasting between replicated inputs is the operation's own concern.
        let Some(output_batch_axis_position) = inputs.iter().find_map(ArrayBatch::batch_axis_position) else {
            let packed_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
            let output_count = Operation::infer_output_types(self, packed_types.as_slice(), &[])?.len();
            return Ok(self
                .interpret_with_batch_axes(context, inputs, &vec![BatchAxis::replicated(); output_count])?
                .into());
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

        // Ragged elementwise operands must agree wherever they describe the same physical axis. Replicated and fully
        // dense operands impose no raggedness of their own and can be combined with a ragged operand normally.
        let mut ragged_axes = Vec::<RaggedAxis<C::Value>>::new();
        for ragged_axis in broadcasted_inputs.iter().flat_map(|input| input.ragged_axes()) {
            if let Some(existing) = ragged_axes.iter().find(|existing| existing.axis() == ragged_axis.axis()) {
                if existing.dimension() != ragged_axis.dimension() {
                    return Err(BatchingError::InvalidBatchMetadata {
                        message: format!(
                            "elementwise operation `{}` received incompatible ragged dimensions {} and {} on packed \
                             axis {}",
                            self.name(),
                            existing.dimension(),
                            ragged_axis.dimension(),
                            ragged_axis.axis(),
                        ),
                    });
                }
            } else {
                ragged_axes.push(ragged_axis.clone());
            }
        }

        let output_batch_axes = vec![BatchAxis::new(batch_axis); output_count];
        Ok(self
            .interpret_with_batch_axes(context, &broadcasted_inputs, &output_batch_axes)?
            .into_iter()
            .map(|output| {
                ArrayBatch::new(output.value, output.batch_axis)?
                    .with_ragged_axes(ragged_axes.clone())?
                    .move_axis(output_batch_axis_position)
            })
            .collect::<Result<Vec<_>, _>>()?
            .into())
    }
}

impl<
    V: Value<Type = ArrayType>,
    O: Operation<Type = ArrayType>
        + BatchableOperation<TracingContext<V, O>, ArrayBatching>
        + From<TransposeOperation>
        + From<BroadcastOperation>,
> RegionRef<'_, V, O>
{
    /// Structurally batches this borrowed homogeneous-array [`Region`](crate::Region) so that the resulting program
    /// operates over inputs batched along the specified [`BatchAxis`]s. Staged higher-order [`BatchableOperation`]
    /// implementations use this function to batch captured programs *without* concretizing any batch-item values, so
    /// that batched control-flow and custom-derivative structure can be staged back into the enclosing trace. This
    /// function replays the region through an [`ArrayBatching`] [`BatchingContext`] over a fresh [`TracingContext`],
    /// lifts every instruction through its [`BatchableOperation`] rule, and extracts the resulting staged program
    /// together with the requested [`ProgramBatchingOutputAxesPolicy`].
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
    /// [`ProgramBatchingOutputAxesPolicy::AlignAllTo`] imposes one canonical
    /// output axis when a consumer explicitly requires one common layout.
    ///
    /// Dynamic per-item dimensions are supported. The mapped axis is always _freshly inserted_ as [`Dimension::Static`]
    /// carrying the caller's `axis_size`, so program-level batching never reads a batch extent off an input type and
    /// consequently never raises [`BatchingError::DynamicBatchAxis`] the way the value-level
    /// [`Batch::batch`](crate::Batch::batch) entry point does, which must recover the batch size from a mapped value's
    /// own type. A dynamic dimension in an input's per-item type is therefore never the batch axis: it crosses the
    /// rewritten boundary unchanged, and where a [`BatchableOperation`] rule needs its runtime extent (such as the
    /// broadcast that aligns a replicated elementwise operand against a mapped one) that extent is resolved through
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
        let input_count = self.input_ids().len();
        check_count!("input", input_batch_axes, input_count, ProgramError);

        let parent_context = TracingContext::<V, O>::new();
        let builder = parent_context.builder().clone();

        // Keep every tracer and context that holds a clone of `builder` inside the following scope so that recovering
        // the builder later on (below) is a real ownership check. In particular, `context` (which owns a clone of the
        // builder through its parent trace) must be created *inside* this scope so it is dropped before the builder is
        // recovered below; leaving it in the enclosing scope leaks a builder clone past the recovery.
        let (output_atom_ids, output_axes) = {
            let batching_context = BatchingContext::new(parent_context, axis_size)
                .with_axis_name(axis_name)
                .with_axis_sharding(axis_sharding.clone());
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
                    Ok(ArrayBatch::new(value, *batch_axis)?)
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
                |instruction, instruction_inputs| -> Result<_, BatchingError> {
                    let regions = ReplayRegionDriver::new(*self, instruction.regions(), &region_mappings)?;
                    let (outputs, evidence) = instruction
                        .operation()
                        .batch(&batching_context, &RecursiveBatchingDriver::new(&regions), instruction_inputs)?
                        .into_parts();
                    <ArrayBatching as BatchingPolicy<TracingContext<V, O>>>::validate_operation_outputs(
                        instruction.operation().name(),
                        instruction_inputs,
                        outputs.as_slice(),
                        &evidence,
                    )?;
                    Ok(outputs)
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
        Ok(BoundaryPreservingBatchedProgram::new(program, output_axes)?)
    }
}

impl<
    V: Value<Type = ArrayType>,
    O: Operation<Type = ArrayType>
        + BatchableOperation<TracingContext<V, O>, ArrayBatching>
        + From<TransposeOperation>
        + From<BroadcastOperation>,
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
    ) -> Result<BoundaryPreservingBatchedProgram<V, O>, BatchingError> {
        self.entry_region_ref().batched(axis_size, axis_sharding, input_batch_axes, output_axes_policy)
    }
}

// TODO(eaplatanios): Review from here onwards.

/// Kind-aware batched view of one array IR value.
#[derive(Clone, Debug, Parameter)]
pub struct ArrayIrBatch<V: Value<Type = ArrayIrType>> {
    /// Packed parent value.
    value: V,

    /// Mapped packed array axis, or replicated for array and dimension values shared across the batch.
    batch_axis: BatchAxis,

    /// Per-item [`DimensionType`] of a mapped first-class dimension, whose per-item extents are packed into `value` as
    /// ordinary integer array data. It is `None` for every other carrier kind, whose per-item type is instead derived
    /// from `value` by [`Self::unbatched_type`].
    mapped_dimension: Option<DimensionType>,

    /// Bounded ragged axes of an array member. A mapped dimension stores its per-item extents directly in `value` and
    /// therefore does not duplicate them here.
    ragged_axes: Vec<RaggedAxis<V>>,
}

impl<V: Value<Type = ArrayIrType>> ArrayIrBatch<V> {
    /// Creates a batch view and rejects mapped first-class dimensions.
    pub fn new(value: V, batch_axis: BatchAxis) -> Result<Self, BatchingError> {
        let batch_axis = {
            let value_type = value.r#type();
            match value_type.as_ref() {
                // Validating the per-item derivation once here is what lets `Self::unbatched_type` be infallible.
                ArrayIrType::Array(packed_type) => packed_type.normalize_batch_axis(batch_axis)?.0,
                ArrayIrType::Dimension(_) if batch_axis.is_replicated() => batch_axis,
                ArrayIrType::Dimension(r#type) => {
                    return Err(BatchingError::MappedDimension { r#type: Box::new(r#type.clone()), axis: batch_axis });
                }
            }
        };
        Ok(Self { value, batch_axis, mapped_dimension: None, ragged_axes: Vec::new() })
    }

    /// Creates a mapped first-class dimension whose per-item extents are packed as ordinary integer array data. This is
    /// the one carrier kind whose per-item type cannot be derived from its packed value, because the value is an
    /// integer extent array while the per-item type is a [`DimensionType`].
    pub(crate) fn mapped_dimension(
        value: V,
        batch_axis: BatchAxis,
        r#type: DimensionType,
    ) -> Result<Self, BatchingError> {
        let batch_axis = {
            let value_type = value.r#type();
            let array_type = <&ArrayType>::try_from(value_type.as_ref())?;
            if !array_type.data_type().is_integer() {
                return Err(TypeError::invalid(format!(
                    "ragged dimension extents must use an integer array type but got {array_type}",
                ))
                .into());
            }
            let (batch_axis, _) = array_type.normalize_batch_axis(batch_axis)?;
            if batch_axis.is_replicated() {
                return Err(BatchingError::InvalidBatchMetadata {
                    message: "a mapped dimension requires a mapped batch axis".to_string(),
                });
            }
            batch_axis
        };
        Ok(Self { value, batch_axis, mapped_dimension: Some(r#type), ragged_axes: Vec::new() })
    }

    /// Replaces this array batch's ragged-axis metadata. Each ragged axis must be an ordinary packed axis of the
    /// carried value and cannot be its mapped batch axis; a mapped first-class dimension member is rejected because it
    /// carries its per-item extents directly and has no ragged array axes. The per-item type follows the new metadata,
    /// because [`Self::unbatched_type`] derives it from the packed value and the ragged axes together. Refer to the
    /// documentation of [`ArrayBatch::with_ragged_axes`] for the semantic invariants each [`RaggedAxis`] claim
    /// carries; this method validates the same structural requirements and trusts the same claims.
    pub fn with_ragged_axes(mut self, ragged_axes: Vec<RaggedAxis<V>>) -> Result<Self, BatchingError> {
        let value_type = self.value.r#type();
        let packed_type = match (&self.mapped_dimension, value_type.as_ref()) {
            // A mapped first-class dimension carries its per-item extents directly and has no ragged array axes.
            (Some(r#type), _) => {
                return Err(BatchingError::MappedDimension { r#type: Box::new(r#type.clone()), axis: self.batch_axis });
            }
            (None, r#type) => <&ArrayType>::try_from(r#type)?,
        };
        let position = self.batch_axis_position();
        for ragged_axis in &ragged_axes {
            if ragged_axis.axis() >= packed_type.rank() || Some(ragged_axis.axis()) == position {
                return Err(BatchingError::InvalidBatchMetadata {
                    message: format!(
                        "ragged axis {} is invalid for packed array type {packed_type} with batch axis {}",
                        ragged_axis.axis(),
                        self.batch_axis,
                    ),
                });
            }
        }
        self.ragged_axes = ragged_axes;
        Ok(self)
    }

    /// Creates a replicated batch view.
    #[inline]
    pub fn replicated(value: V) -> Self {
        Self { value, batch_axis: BatchAxis::replicated(), mapped_dimension: None, ragged_axes: Vec::new() }
    }

    /// Returns the packed parent value.
    #[inline]
    pub fn value(&self) -> &V {
        &self.value
    }

    /// Consumes this batch and returns its packed parent value.
    #[inline]
    pub fn into_value(self) -> V {
        self.value
    }

    /// Returns the mapped packed array axis, or replicated.
    #[inline]
    pub fn batch_axis(&self) -> BatchAxis {
        self.batch_axis
    }

    /// Returns the canonical nonnegative mapped-axis position for an array member, or `None` for a replicated member.
    pub(crate) fn batch_axis_position(&self) -> Option<usize> {
        let value_type = self.value.r#type();
        let r#type = <&ArrayType>::try_from(value_type.as_ref()).ok()?;
        self.batch_axis.axis().map(|axis| axis.normalize(r#type.rank()).unwrap())
    }

    /// Returns the logical per-item [`ArrayIrType`] reported to the transformed program. A mapped first-class dimension
    /// reports the [`DimensionType`] it was created with, since its packed value only holds the per-item extents. Every
    /// other carrier derives its per-item type from the packed value, an array member by removing the mapped batch
    /// dimension and restoring the dynamic [`Dimension`] of every bounded ragged axis, and a replicated first-class
    /// dimension by reporting the packed [`DimensionType`] unchanged.
    pub fn unbatched_type(&self) -> ArrayIrType {
        if let Some(r#type) = &self.mapped_dimension {
            return r#type.clone().into();
        }
        // The constructors validate this derivation against the packed value, which cannot change afterwards.
        match self.value.r#type().as_ref() {
            ArrayIrType::Array(packed_type) => {
                packed_type.unbatched_type_and_axis(self.batch_axis, self.ragged_axes.as_slice()).unwrap().0.into()
            }
            ArrayIrType::Dimension(r#type) => r#type.clone().into(),
        }
    }

    /// Returns this array member's bounded ragged axes.
    #[inline]
    pub fn ragged_axes(&self) -> &[RaggedAxis<V>] {
        self.ragged_axes.as_slice()
    }

    /// Returns the mapped per-item dimension's packed extent array, or `None` for every other carrier kind.
    #[inline]
    pub(crate) fn mapped_dimension_extents(&self) -> Option<&V> {
        self.mapped_dimension.as_ref().map(|_| &self.value)
    }

    /// Validates that this batch contains a replicated first-class dimension.
    pub(crate) fn validate_replicated_dimension(&self) -> Result<(), BatchingError> {
        let r#type = match &self.mapped_dimension {
            Some(r#type) => r#type.clone(),
            None => {
                let value_type = self.value.r#type();
                <&DimensionType>::try_from(value_type.as_ref())?.clone()
            }
        };
        if self.mapped_dimension.is_none() && self.batch_axis.is_replicated() {
            Ok(())
        } else {
            Err(BatchingError::MappedDimension { r#type: Box::new(r#type), axis: self.batch_axis })
        }
    }
}

impl<V: Value<Type = ArrayIrType> + PartialEq> PartialEq for ArrayIrBatch<V> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
            && self.batch_axis == other.batch_axis
            && self.mapped_dimension == other.mapped_dimension
            && self.ragged_axes == other.ragged_axes
    }
}

impl<V: Value<Type = ArrayIrType>> Typed for ArrayIrBatch<V> {
    type Type = ArrayIrType;

    /// The type of a batch carrier is its logical per-item type, as returned by [`Self::unbatched_type`].
    #[inline]
    fn r#type(&self) -> Cow<'_, ArrayIrType> {
        Cow::Owned(self.unbatched_type())
    }
}

impl<V: Value<Type = ArrayIrType>> Display for ArrayIrBatch<V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "batch[{}, {}]({})", self.r#type(), self.batch_axis, self.value)
    }
}

/// Result of batching an array IR [`Region`](crate::Region), whose boundary explicitly
/// threads its mapped extent.
///
/// Composite regions are retraced as standalone programs and therefore cannot capture the parent program's
/// first-class mapped-extent SSA value. Their boundary has one additional leading dimension input and output:
/// the input defines the identity referenced by every inserted dynamic batch dimension, and the output forwards that
/// same atom so enclosing higher-order operations can carry it through the sealed region. Output-axis metadata
/// excludes this bookkeeping output, and [`BatchedProgram::into_parts`] documents the arity contract consumers must
/// uphold.
/// Consumers that instead need an ordinary [`Region`](crate::Region) boundary shed the widening through
/// [`BatchingPolicy::adapt_batched_program`] and complete the adapted program's operands with
/// [`BatchingPolicy::boundary_operands`].
pub struct ThreadedExtentBatchedProgram<V: Typed<Type = ArrayIrType> + Parameter, O> {
    /// Structurally transformed program, including its leading bookkeeping input and output.
    program: Program<V, O, Vec<V>, Vec<V>>,

    /// Mapped axes of the source region's outputs. The bookkeeping-only threaded extent is excluded.
    output_axes: Vec<BatchAxis>,
}

/// Batching policy for programs whose values may be arrays or first-class dimensions.
///
/// Array members may carry a mapped axis. Dimension members are shared shape values and therefore remain
/// replicated. The mapped-axis extent is itself an ordinary parent-owned dimension value, so dynamic extents remain
/// SSA data rather than transform metadata.
#[derive(Copy, Clone, Debug, Default)]
pub struct ArrayIrBatching;

impl BatchableType for ArrayIrType {
    type Policy = ArrayIrBatching;
}

impl<C: Context<Type = ArrayIrType>> BatchingPolicy<C> for ArrayIrBatching {
    type Batch = ArrayIrBatch<C::Value>;
    type Extent = C::Value;
    type Evidence = Vec<DimensionVariable>;
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

    #[inline]
    fn unbatched_type(batch: &Self::Batch) -> Cow<'_, C::Type> {
        batch.r#type()
    }

    /// Every bounded ragged dimension carried by an operand must survive the rule that consumed it, either as a ragged
    /// output axis, as a mapped output dimension holding its per-item extents, or as a dimension the rule's evidence
    /// claims it consumed deliberately. Anything else would silently forget per-item extents that no [`ArrayIrType`]
    /// records, so the operation is rejected while naming the exact dimension that was lost.
    fn validate_operation_outputs(
        operation_name: &'static str,
        inputs: &[Self::Batch],
        outputs: &[Self::Batch],
        evidence: &Self::Evidence,
    ) -> Result<(), BatchingError> {
        let preserves_dimension = |dimension: &DimensionVariable| {
            evidence.contains(dimension)
                || outputs.iter().any(|output| {
                    output.ragged_axes().iter().any(|axis| axis.dimension() == dimension)
                        || (output.mapped_dimension_extents().is_some()
                            && matches!(
                                output.unbatched_type(),
                                ArrayIrType::Dimension(r#type) if r#type.variable() == dimension
                            ))
                })
        };
        let lost_dimension = inputs
            .iter()
            .flat_map(ArrayIrBatch::ragged_axes)
            .map(RaggedAxis::dimension)
            .find(|dimension| !preserves_dimension(dimension));
        match lost_dimension {
            Some(dimension) => Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "operation `{operation_name}` neither preserves nor consumes bounded ragged dimension \
                     `{dimension}`",
                ),
            }),
            None => Ok(()),
        }
    }

    /// The adapted program's leading input still defines the [`DimensionVariable`](crate::arrays::DimensionVariable)
    /// referenced by every inserted dynamic batch dimension, so the first-class mapped-extent value must become its
    /// matching operand.
    #[inline]
    fn boundary_operands(axis_extent: &Self::Extent) -> Vec<C::Value> {
        vec![axis_extent.clone()]
    }

    /// Drops the leading forwarded-extent bookkeeping output that exists only for extent-threading consumers.
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

impl<V: Value<Type = ArrayIrType>, O: Operation<Type = ArrayIrType>> ThreadedExtentBatchedProgram<V, O> {
    /// Creates a batched array IR program with one leading mapped-extent input and forwarded output.
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
                "a structurally batched program's leading threaded-extent input must be a dimension".to_string(),
            ));
        }
        if !matches!(program.outputs().next().unwrap().r#type().as_ref(), ArrayIrType::Dimension(_)) {
            return Err(ProgramError::MalformedProgram(
                "a structurally batched program's leading threaded-extent output must be a dimension".to_string(),
            ));
        }
        if program.output_ids()[0] != program.input_ids()[0] {
            return Err(ProgramError::MalformedProgram(
                "a structurally batched program's leading threaded-extent output must forward its leading input"
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

/// [`ArrayBatchingPolicy`] used while a homogeneous array rule runs inside an array IR batching transform.
///
/// When composite batching reaches an array-member operation, it projects the operation and its batches into the
/// zero-state [`ProjectedContext`] over [`ArrayType`] and reuses the homogeneous rule unchanged: batches remain
/// ordinary [`ArrayBatch`]es, so the rule cannot tell it is running inside a composite program. What does change is
/// extent representation — the mapped-axis extent is the outer composite context's first-class dimension value rather
/// than a static host `usize`, so a dynamic batch extent stays an ordinary SSA operand edge.
///
/// This [`ArrayBatchingPolicy`] implementation is correspondingly the only place that translates a homogeneous rule's
/// extent and move-or-broadcast requests into mixed array IR operations: static per-item dimensions become exact
/// dimension constants, dynamic per-item dimensions become `dimension_size` reads of their broadcast-compatible
/// source axes, and the mapped axis itself is grounded by the extent value. [`ArrayBatchingPolicy::axis_size`]
/// succeeds only when the extent value's type proves one exact extent, so rules that genuinely enumerate batch items
/// fail with a precise error at dynamic extents instead of silently specializing them.
#[derive(Copy, Clone, Debug, Default)]
pub struct DynamicArrayBatchingPolicy;

impl<C: Context<Type = ArrayIrType>> BatchingPolicy<ProjectedContext<C, ArrayType>> for DynamicArrayBatchingPolicy
where
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Value: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Operation: OperationProjection<ArrayType>,
{
    type Batch = ArrayBatch<<C::Value as ValueProjection<ArrayType>>::Projected>;
    type Extent = C::Value;
    type Evidence = Vec<DimensionVariable>;
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

/// Batching policy used while a homogeneous first-class-dimension operation runs inside an array IR batching
/// transform. A dimension is shared shape metadata, so its projected value is itself the complete batch carrier:
/// replicated inputs pass through unchanged, while any mapped input is rejected because a different extent per batch
/// item would require a ragged array representation. The policy still carries the outer transform's first-class
/// mapped extent and ragged-dimension evidence representation so
/// [`batch_projected_operation`](crate::batch_projected_operation) can construct one uniform projected batching context
/// for every member kind without specializing either. A dimension rule never consumes a ragged dimension, so its
/// evidence is always empty.
#[derive(Copy, Clone, Debug, Default)]
pub struct ReplicatedDimensionBatchingPolicy;

impl<C: Context<Type = ArrayIrType>> BatchingPolicy<ProjectedContext<C, DimensionType>>
    for ReplicatedDimensionBatchingPolicy
where
    C::Constant: ValueProjection<DimensionType, Projected: Value<Type = DimensionType>>,
    C::Value: ValueProjection<DimensionType, Projected: Value<Type = DimensionType>>,
    C::Operation: OperationProjection<DimensionType>,
{
    type Batch = <C::Value as ValueProjection<DimensionType>>::Projected;
    type Extent = C::Value;
    type Evidence = Vec<DimensionVariable>;
    type BatchedProgram = BoundaryPreservingBatchedProgram<
        <C::Constant as ValueProjection<DimensionType>>::Projected,
        <C::Operation as OperationProjection<DimensionType>>::Projected,
    >;

    fn batch(
        value: <C::Value as ValueProjection<DimensionType>>::Projected,
        batch_axis: BatchAxis,
    ) -> Result<Self::Batch, BatchingError> {
        if !batch_axis.is_replicated() {
            return Err(BatchingError::MappedDimension {
                r#type: Box::new(value.r#type().into_owned()),
                axis: batch_axis,
            });
        }
        Ok(value)
    }

    #[inline]
    fn replicated(value: <C::Value as ValueProjection<DimensionType>>::Projected) -> Self::Batch {
        value
    }

    #[inline]
    fn value(batch: &Self::Batch) -> &<C::Value as ValueProjection<DimensionType>>::Projected {
        batch
    }

    #[inline]
    fn batch_axis(_batch: &Self::Batch) -> BatchAxis {
        BatchAxis::replicated()
    }

    #[inline]
    fn unbatched_type(batch: &Self::Batch) -> Cow<'_, DimensionType> {
        batch.r#type()
    }

    #[inline]
    fn adapt_batched_program<CollapseFn>(
        program: Self::BatchedProgram,
        required_output_axes: Option<&[BatchAxis]>,
        collapse_fn: CollapseFn,
    ) -> Result<
        BoundaryPreservingBatchedProgram<
            <C::Constant as ValueProjection<DimensionType>>::Projected,
            <C::Operation as OperationProjection<DimensionType>>::Projected,
        >,
        BatchingError,
    >
    where
        CollapseFn: Fn(
            &TracingContext<
                <C::Constant as ValueProjection<DimensionType>>::Projected,
                <C::Operation as OperationProjection<DimensionType>>::Projected,
            >,
            Tracer<
                TracingContext<
                    <C::Constant as ValueProjection<DimensionType>>::Projected,
                    <C::Operation as OperationProjection<DimensionType>>::Projected,
                >,
            >,
            Axis,
        ) -> Result<
            Tracer<
                TracingContext<
                    <C::Constant as ValueProjection<DimensionType>>::Projected,
                    <C::Operation as OperationProjection<DimensionType>>::Projected,
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

impl<C: Context<Type = ArrayIrType>> BatchingPolicyProjection<C, ArrayType> for ArrayIrBatching
where
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Value: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Operation: OperationProjection<ArrayType>,
    ProjectedContext<C, ArrayType>: Context<
            Type = ArrayType,
            Value = <C::Value as ValueProjection<ArrayType>>::Projected,
            Constant = <C::Constant as ValueProjection<ArrayType>>::Projected,
            Operation = <C::Operation as OperationProjection<ArrayType>>::Projected,
        >,
{
    type Projected = ArrayBatching<DynamicArrayBatchingPolicy>;

    fn project_batch(
        batch: &Self::Batch,
    ) -> Result<<Self::Projected as BatchingPolicy<ProjectedContext<C, ArrayType>>>::Batch, BatchingError> {
        let value = C::Value::into_projected(batch.value().clone())?;
        let ragged_axes = batch
            .ragged_axes()
            .iter()
            .cloned()
            .map(|ragged_axis| -> Result<_, BatchingError> {
                Ok(RaggedAxis::new(
                    ragged_axis.axis(),
                    C::Value::into_projected(ragged_axis.extents)?,
                    ragged_axis.dimension,
                    ragged_axis.extent_axes,
                ))
            })
            .collect::<Result<Vec<_>, _>>()?;
        ArrayBatch::new(value, batch.batch_axis())?.with_ragged_axes(ragged_axes)
    }

    fn lift_batch(
        batch: &<Self::Projected as BatchingPolicy<ProjectedContext<C, ArrayType>>>::Batch,
    ) -> Result<Self::Batch, BatchingError> {
        let ragged_axes = batch
            .ragged_axes()
            .iter()
            .cloned()
            .map(|ragged_axis| {
                RaggedAxis::new(
                    ragged_axis.axis,
                    C::Value::from_projected(ragged_axis.extents),
                    ragged_axis.dimension,
                    ragged_axis.extent_axes,
                )
            })
            .collect();
        ArrayIrBatch::new(C::Value::from_projected(batch.value().clone()), batch.batch_axis())?
            .with_ragged_axes(ragged_axes)
    }
}

impl<C: Context<Type = ArrayIrType>> BatchingPolicyProjection<C, DimensionType> for ArrayIrBatching
where
    C::Constant: ValueProjection<DimensionType, Projected: Value<Type = DimensionType>>,
    C::Value: ValueProjection<DimensionType, Projected: Value<Type = DimensionType>>,
    C::Operation: OperationProjection<DimensionType>,
{
    type Projected = ReplicatedDimensionBatchingPolicy;
}

impl<C, T> ValueProjection<T> for BatchingTracer<C, ArrayIrBatching>
where
    C: Context<Type = ArrayIrType, Operation: BatchableOperation<C, ArrayIrBatching>>,
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Operation: BatchableOperation<TracingContext<C::Constant, C::Operation>, ArrayIrBatching>
        + From<DynamicBroadcastOperation>
        + From<ConstantOperation<DimensionValue>>
        + From<DimensionSizeOperation>
        + OperationProjection<ArrayType>,
    <C::Operation as OperationProjection<ArrayType>>::Projected: From<TransposeOperation>,
    T: Type,
    for<'t> &'t T: TryFrom<&'t ArrayIrType, Error = TypeError>,
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
        Ok(ProjectedValue::new(self, <&T>::try_from(&self.batch().unbatched_type())?.clone()))
    }

    #[inline]
    fn into_projected(self) -> Result<Self::Projected, TypeError> {
        let r#type = <&T>::try_from(&self.batch().unbatched_type())?.clone();
        Ok(ProjectedValue::new(self, r#type))
    }
}

/// Reads one packed array axis as a first-class dimension value in `context`.
pub(crate) fn array_dimension<C: Context<Type = ArrayIrType, Operation: From<DimensionSizeOperation>>>(
    context: &C,
    value: &C::Value,
    axis: usize,
) -> Result<C::Value, BatchingError> {
    let value_type = value.r#type();
    let array_type = <&ArrayType>::try_from(value_type.as_ref())?;
    let operation = DimensionSizeOperation::new(array_type, axis)?;
    Ok(context.bind(operation, Vec::new(), std::slice::from_ref(value))?.remove(0))
}

/// Stages one exact first-class dimension constant carrying `extent` in `context`.
pub(crate) fn dimension_constant<C>(context: &C, extent: usize) -> Result<C::Value, BatchingError>
where
    C: Context<Type = ArrayIrType, Operation: From<ConstantOperation<DimensionValue>>>,
{
    let value = DimensionValue::constant(extent).map_err(ProgramError::from)?;
    let mut outputs = context.bind(ConstantOperation::new(value), Vec::new(), &[])?;
    check_count!("output", outputs, 1, ProgramError);
    Ok(outputs.remove(0))
}

/// Returns one packed array axis as a first-class dimension value, staging an exact constant when the axis extent is
/// statically known and reading the axis through [`array_dimension`] only when it is genuinely dynamic. Folding the
/// static axes keeps staged programs free of `dimension_size` reads whose results the type system already knows.
pub(crate) fn folded_array_dimension<C>(context: &C, value: &C::Value, axis: usize) -> Result<C::Value, BatchingError>
where
    C: Context<Type = ArrayIrType>,
    C::Operation: From<ConstantOperation<DimensionValue>> + From<DimensionSizeOperation>,
{
    let value_type = value.r#type();
    let array_type = <&ArrayType>::try_from(value_type.as_ref())?;
    match array_type.shape().dimensions().get(axis) {
        Some(Dimension::Static(extent)) => dimension_constant(context, *extent),
        _ => array_dimension(context, value, axis),
    }
}

/// Requires two composite dimension values to describe the same mapped extent.
pub(crate) fn require_equal_dimensions<C>(context: &C, left: &C::Value, right: &C::Value) -> Result<(), BatchingError>
where
    C: Context<Type = ArrayIrType>,
    C::Constant: ValueProjection<DimensionType, Projected: Value<Type = DimensionType>>,
    C::Value: ValueProjection<DimensionType, Projected: Value<Type = DimensionType>>,
    C::Operation: OperationProjection<DimensionType>,
    <C::Operation as OperationProjection<DimensionType>>::Projected: From<DimensionRequirementOperation>,
{
    let left = <C::Value as ValueProjection<DimensionType>>::into_projected(left.clone())?;
    let right = <C::Value as ValueProjection<DimensionType>>::into_projected(right.clone())?;
    let operation = DimensionRequirementOperation::equal(left.r#type().as_ref(), right.r#type().as_ref());
    ProjectedContext::<C, DimensionType>::new(context.clone()).bind(operation, Vec::new(), &[left, right])?;
    Ok(())
}

/// Binds one mixed dynamic broadcast against explicit first-class output dimensions.
pub(crate) fn broadcast_array<C>(
    context: &C,
    value: C::Value,
    output_dimensions: Vec<C::Value>,
    output_axes: Vec<usize>,
    output_sharding: Option<Sharding>,
) -> Result<C::Value, BatchingError>
where
    C: Context<Type = ArrayIrType, Operation: From<DynamicBroadcastOperation>>,
{
    let operation = DynamicBroadcastOperation::new(output_axes).with_output_sharding(output_sharding);
    let mut inputs = Vec::with_capacity(output_dimensions.len() + 1);
    inputs.push(value);
    inputs.extend(output_dimensions);
    Ok(context.bind(operation, Vec::new(), inputs.as_slice())?.remove(0))
}

impl<C> ArrayBatchingPolicy<ProjectedContext<C, ArrayType>> for DynamicArrayBatchingPolicy
where
    C: Context<
            Type = ArrayIrType,
            Operation: From<DynamicBroadcastOperation>
                           + From<ConstantOperation<DimensionValue>>
                           + From<DimensionSizeOperation>
                           + OperationProjection<ArrayType>,
        >,
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Value: ValueProjection<ArrayType, Projected: Transpose + Value<Type = ArrayType>>,
{
    fn axis_dimension(
        context: &BatchingContext<ProjectedContext<C, ArrayType>, ArrayBatching<DynamicArrayBatchingPolicy>>,
    ) -> Result<Dimension, BatchingError> {
        let extent_type = context.axis_extent().r#type();
        Ok(<&DimensionType>::try_from(extent_type.as_ref())?.to_dimension())
    }

    fn match_axis(
        context: &BatchingContext<ProjectedContext<C, ArrayType>, ArrayBatching<DynamicArrayBatchingPolicy>>,
        batch: &ArrayBatch<<C::Value as ValueProjection<ArrayType>>::Projected>,
        axis: Axis,
    ) -> Result<ArrayBatch<<C::Value as ValueProjection<ArrayType>>::Projected>, BatchingError> {
        if !batch.batch_axis().is_replicated() {
            return batch.move_axis(axis);
        }
        let array_type = batch.unbatched_type();
        let output_rank = array_type.rank() + 1;
        let position = axis
            .normalize(output_rank)
            .map_err(|_| BatchingError::BatchAxisOutOfBounds { r#type: Box::new(array_type.clone()), axis })?;
        let outer_context = context.parent().parent();
        let value = <C::Value as ValueProjection<ArrayType>>::from_projected(batch.value().clone());
        // The replicated per-item shape survives unchanged, so each of its axes contributes either an exact constant
        // or a `dimension_size` read, and the inserted mapped axis takes the transform's own extent.
        let mut output_dimensions = (0..array_type.rank())
            .map(|axis| folded_array_dimension(outer_context, &value, axis))
            .collect::<Result<Vec<_>, _>>()?;
        output_dimensions.insert(position, context.axis_extent().clone());
        let output_axes = (0..array_type.rank())
            .map(|input_axis| if input_axis < position { input_axis } else { input_axis + 1 })
            .collect::<Vec<_>>();
        let output_sharding = array_type
            .sharding()
            .map(|sharding| {
                sharding
                    .with_inserted_dimension(position, context.axis_sharding().clone())
                    .map_err(|error| BatchingError::MisalignedBatchAxes { message: error.to_string() })
            })
            .transpose()?;
        let value = broadcast_array(outer_context, value, output_dimensions, output_axes, output_sharding)?;
        ArrayBatch::new(
            <C::Value as ValueProjection<ArrayType>>::into_projected(value)?,
            BatchAxis::from_position(position),
        )
    }

    fn broadcast_input(
        context: &BatchingContext<ProjectedContext<C, ArrayType>, ArrayBatching<DynamicArrayBatchingPolicy>>,
        input: &ArrayBatch<<C::Value as ValueProjection<ArrayType>>::Projected>,
        r#type: ArrayType,
        output_axes: Vec<usize>,
        batch_axis: Axis,
        dimension_sources: Vec<DimensionSource<<C::Value as ValueProjection<ArrayType>>::Projected>>,
    ) -> Result<ArrayBatch<<C::Value as ValueProjection<ArrayType>>::Projected>, BatchingError> {
        // Materialize every algorithm-provided output-dimension source in first-class form: exact
        // constants for static dimensions, `dimension_size` reads of the provided source values for dynamic per-item
        // dimensions, and the transform's extent value for the mapped axis.
        let outer_context = context.parent().parent();
        let output_dimensions = dimension_sources
            .into_iter()
            .map(|dimension_source| -> Result<C::Value, BatchingError> {
                match dimension_source {
                    DimensionSource::Static(extent) => dimension_constant(outer_context, extent),
                    DimensionSource::Value { source, axis } => {
                        let source = <C::Value as ValueProjection<ArrayType>>::from_projected(source);
                        array_dimension(outer_context, &source, axis)
                    }
                    DimensionSource::BatchExtent => Ok(context.axis_extent().clone()),
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let value = <C::Value as ValueProjection<ArrayType>>::from_projected(input.value().clone());
        let value =
            broadcast_array(outer_context, value, output_dimensions, output_axes.clone(), r#type.sharding().cloned())?;
        let ragged_axes = input
            .ragged_axes()
            .iter()
            .cloned()
            .map(|ragged_axis| ragged_axis.broadcasted(output_axes.as_slice()))
            .collect();
        ArrayBatch::new(<C::Value as ValueProjection<ArrayType>>::into_projected(value)?, batch_axis)?
            .with_ragged_axes(ragged_axes)
    }
}

impl<C> RaggedArrayBatchingPolicy<ProjectedContext<C, ArrayType>> for DynamicArrayBatchingPolicy
where
    C: Context<
            Type = ArrayIrType,
            Operation: From<DynamicBroadcastOperation>
                           + From<ConstantOperation<DimensionValue>>
                           + From<DimensionSizeOperation>
                           + OperationProjection<ArrayType>,
        >,
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Value: ValueProjection<ArrayType, Projected: Transpose + Value<Type = ArrayType>>,
    <C::Operation as OperationProjection<ArrayType>>::Projected: From<AndOperation<ArrayType>>
        + From<CompareOperation<ArrayType>>
        + From<IotaOperation<ArrayType>>
        + From<SelectOperation<ArrayType>>
        + From<ZeroLikeOperation<ArrayType>>,
{
    fn mask_reduction_input(
        context: &BatchingContext<ProjectedContext<C, ArrayType>, ArrayBatching<DynamicArrayBatchingPolicy>>,
        input: &ArrayBatch<<C::Value as ValueProjection<ArrayType>>::Projected>,
        reduced_axes: &[usize],
        kind: ReductionKind,
    ) -> Result<ArrayBatch<<C::Value as ValueProjection<ArrayType>>::Projected>, BatchingError> {
        // Zero is the identity of a sum and nothing else, so a reduction that actually reduces a ragged axis under any
        // other kind would observe the padding this policy can only zero.
        if kind != ReductionKind::Sum
            && input.ragged_axes().iter().any(|ragged_axis| reduced_axes.contains(&ragged_axis.axis()))
        {
            return Err(BatchingError::UnsupportedOperation {
                message: format!("ragged reduction kind {kind} is not supported; use reduce_sum"),
            });
        }
        zero_ragged_padding(context, input, reduced_axes)
    }

    #[inline]
    fn pad_contraction_input(
        context: &BatchingContext<ProjectedContext<C, ArrayType>, ArrayBatching<DynamicArrayBatchingPolicy>>,
        input: &ArrayBatch<<C::Value as ValueProjection<ArrayType>>::Projected>,
        contracted_axes: &[usize],
    ) -> Result<ArrayBatch<<C::Value as ValueProjection<ArrayType>>::Projected>, BatchingError> {
        zero_ragged_padding(context, input, contracted_axes)
    }
}

/// Stages the zero-masking shared by the dynamic policy's ragged consumption disciplines: every ragged axis of `input`
/// named in `consumed_axes` contributes an `iota < extents` predicate over the packed shape, the predicates are
/// combined, and the padded elements are selected away in favor of a zero of the operand. Zero is both the identity of
/// the one reduction kind whose padding this policy neutralizes and the identity of a contraction, so
/// [`RaggedArrayBatchingPolicy::mask_reduction_input`] and [`RaggedArrayBatchingPolicy::pad_contraction_input`] differ
/// only in which axes they declare consumed and in the discipline-specific validation they perform first. Input with no
/// consumed ragged axis is returned unchanged, so nothing is staged for an operand this discipline does not touch.
fn zero_ragged_padding<C>(
    context: &BatchingContext<ProjectedContext<C, ArrayType>, ArrayBatching<DynamicArrayBatchingPolicy>>,
    input: &ArrayBatch<<C::Value as ValueProjection<ArrayType>>::Projected>,
    consumed_axes: &[usize],
) -> Result<ArrayBatch<<C::Value as ValueProjection<ArrayType>>::Projected>, BatchingError>
where
    C: Context<
            Type = ArrayIrType,
            Operation: From<DynamicBroadcastOperation>
                           + From<ConstantOperation<DimensionValue>>
                           + From<DimensionSizeOperation>
                           + OperationProjection<ArrayType>,
        >,
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Value: ValueProjection<ArrayType, Projected: Transpose + Value<Type = ArrayType>>,
    <C::Operation as OperationProjection<ArrayType>>::Projected: From<AndOperation<ArrayType>>
        + From<CompareOperation<ArrayType>>
        + From<IotaOperation<ArrayType>>
        + From<SelectOperation<ArrayType>>
        + From<ZeroLikeOperation<ArrayType>>,
{
    let consumed = input
        .ragged_axes()
        .iter()
        .filter(|ragged_axis| consumed_axes.contains(&ragged_axis.axis()))
        .collect::<Vec<_>>();
    if consumed.is_empty() {
        return Ok(input.clone());
    }

    let outer_context = context.parent().parent();
    let array_context = context.parent();
    let input_value = input.value().clone();
    let lifted_input_value = <C::Value as ValueProjection<ArrayType>>::from_projected(input_value.clone());
    let packed_type = input.r#type().into_owned();
    let output_dimensions = (0..packed_type.rank())
        .map(|axis| folded_array_dimension(outer_context, &lifted_input_value, axis))
        .collect::<Result<Vec<_>, _>>()?;
    let mut mask = None;
    for ragged_axis in consumed {
        let extent_value = ragged_axis.extents().clone();
        let extent_value_type = extent_value.r#type();
        let extent_type = extent_value_type.as_ref();
        let physical_extent = packed_type.shape()[ragged_axis.axis()].value().unwrap();
        let iota_type = ArrayType::new(extent_type.data_type(), Shape::new(vec![Dimension::Static(physical_extent)]));
        let mut iota = array_context.bind(IotaOperation::new(iota_type, 0)?, Vec::new(), &[])?;
        check_count!("output", iota, 1, ProgramError);
        let iota = broadcast_array(
            outer_context,
            <C::Value as ValueProjection<ArrayType>>::from_projected(iota.remove(0)),
            output_dimensions.clone(),
            vec![ragged_axis.axis()],
            None,
        )?;
        let broadcasted_extent = broadcast_array(
            outer_context,
            <C::Value as ValueProjection<ArrayType>>::from_projected(extent_value),
            output_dimensions.clone(),
            ragged_axis.extent_axes().to_vec(),
            None,
        )?;
        let mut current = array_context.bind(
            CompareOperation::<ArrayType>::new(ComparisonDirection::LessThan),
            Vec::new(),
            &[
                <C::Value as ValueProjection<ArrayType>>::into_projected(iota)?,
                <C::Value as ValueProjection<ArrayType>>::into_projected(broadcasted_extent)?,
            ],
        )?;
        check_count!("output", current, 1, ProgramError);
        mask = Some(match mask {
            None => current.remove(0),
            Some(mask) => {
                let mut combined =
                    array_context.bind(AndOperation::<ArrayType>::new(), Vec::new(), &[mask, current.remove(0)])?;
                check_count!("output", combined, 1, ProgramError);
                combined.remove(0)
            }
        });
    }

    let mut zero =
        array_context.bind(ZeroLikeOperation::<ArrayType>::new(), Vec::new(), std::slice::from_ref(&input_value))?;
    check_count!("output", zero, 1, ProgramError);
    let mut masked = array_context.bind(
        SelectOperation::<ArrayType>::new(),
        Vec::new(),
        &[mask.unwrap(), input_value, zero.remove(0)],
    )?;
    check_count!("output", masked, 1, ProgramError);
    ArrayBatch::new(masked.remove(0), input.batch_axis())?.with_ragged_axes(input.ragged_axes().to_vec())
}

/// Aligns one composite array batch to `axis`, moving an existing mapped axis or dynamically broadcasting a
/// replicated array with the context's first-class extent.
pub(crate) fn align_array_batch<C>(
    context: &BatchingContext<C, ArrayIrBatching>,
    batch: ArrayIrBatch<C::Value>,
    axis: Axis,
) -> Result<ArrayIrBatch<C::Value>, BatchingError>
where
    C: Context<
            Type = ArrayIrType,
            Operation: From<DynamicBroadcastOperation>
                           + From<ConstantOperation<DimensionValue>>
                           + From<DimensionSizeOperation>
                           + OperationProjection<ArrayType>,
        >,
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Value: ValueProjection<ArrayType, Projected: Transpose + Value<Type = ArrayType>>,
{
    // Only an array member has a packed axis to align, so both a mapped and a replicated first-class dimension are
    // rejected here, whether the kind is recorded in the carrier or visible in the packed value's own type.
    if let ArrayIrType::Dimension(r#type) = batch.unbatched_type() {
        return Err(BatchingError::MappedDimension { r#type: Box::new(r#type), axis: BatchAxis::from(axis) });
    }
    if let ArrayIrType::Dimension(r#type) = batch.value.r#type().as_ref() {
        return Err(BatchingError::MappedDimension { r#type: Box::new(r#type.clone()), axis: BatchAxis::from(axis) });
    }
    let ragged_axes = batch
        .ragged_axes
        .into_iter()
        .map(|ragged_axis| -> Result<_, BatchingError> {
            Ok(RaggedAxis::new(
                ragged_axis.axis,
                <C::Value as ValueProjection<ArrayType>>::into_projected(ragged_axis.extents)?,
                ragged_axis.dimension,
                ragged_axis.extent_axes,
            ))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let batch =
        ArrayBatch::new(<C::Value as ValueProjection<ArrayType>>::into_projected(batch.value)?, batch.batch_axis)?
            .with_ragged_axes(ragged_axes)?;
    let projected_context = BatchingContext::<_, ArrayBatching<DynamicArrayBatchingPolicy>>::with_policy(
        ProjectedContext::new(context.parent().clone()),
        context.axis_extent().clone(),
    )
    .with_axis_name(context.axis_name().map(str::to_string))
    .with_axis_sharding(context.axis_sharding().clone());
    let output = DynamicArrayBatchingPolicy::match_axis(&projected_context, &batch, axis)?;
    let batch_axis = output.batch_axis();
    let ragged_axes = output
        .ragged_axes()
        .iter()
        .cloned()
        .map(|ragged_axis| {
            RaggedAxis::new(
                ragged_axis.axis,
                C::Value::from_projected(ragged_axis.extents),
                ragged_axis.dimension,
                ragged_axis.extent_axes,
            )
        })
        .collect();
    ArrayIrBatch::new(<C::Value as ValueProjection<ArrayType>>::from_projected(output.into_value()), batch_axis)?
        .with_ragged_axes(ragged_axes)
}

impl<C> BatchingEntrypointPolicy<C> for ArrayIrBatching
where
    C: Context<
            Type = ArrayIrType,
            Operation: From<DynamicBroadcastOperation>
                           + From<ConstantOperation<DimensionValue>>
                           + From<DimensionSizeOperation>
                           + OperationProjection<ArrayType>
                           + OperationProjection<DimensionType>,
        >,
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>
        + ValueProjection<DimensionType, Projected: Value<Type = DimensionType>>,
    C::Value: ValueProjection<ArrayType, Projected: Transpose + Value<Type = ArrayType>>
        + ValueProjection<DimensionType, Projected: Value<Type = DimensionType>>,
    <C::Operation as OperationProjection<DimensionType>>::Projected: From<DimensionRequirementOperation>,
{
    fn prepare_inputs(
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

        let mut axis_extent = batch_axis.extent().cloned();
        if let Some(axis_extent) = &axis_extent {
            let extent_type = axis_extent.r#type();
            <&DimensionType>::try_from(extent_type.as_ref())?;
        }
        for batch in &batches {
            let Some(position) = batch.batch_axis_position() else {
                continue;
            };
            let input_extent = array_dimension(context, &batch.value, position)?;
            if let Some(axis_extent) = &axis_extent {
                require_equal_dimensions(context, axis_extent, &input_extent)?;
            } else {
                axis_extent = Some(input_extent);
            }
        }
        let axis_extent = axis_extent.ok_or(BatchingError::EmptyBatch)?;

        let axis_sharding = batch_axis_sharding(batches.iter().filter_map(|batch| {
            let array_type = match batch.value.r#type() {
                Cow::Borrowed(ArrayIrType::Array(array_type)) => Cow::Borrowed(array_type),
                Cow::Owned(ArrayIrType::Array(array_type)) => Cow::Owned(array_type),
                _ => return None,
            };
            Some((array_type, batch.batch_axis_position()))
        }))?;
        let batching_context = BatchingContext::new(context.clone(), axis_extent)
            .with_axis_name(batch_axis.name().map(String::from))
            .with_axis_sharding(axis_sharding);
        let batches = batches
            .into_iter()
            .map(|batch| -> Result<_, BatchingError> {
                let Some(position) = batch.batch_axis_position() else {
                    return Ok(batch);
                };
                let value_type = batch.value.r#type();
                let array_type = <&ArrayType>::try_from(value_type.as_ref())?;
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
                        false => folded_array_dimension(batching_context.parent(), &batch.value, axis),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let output_axes = (0..array_type.rank()).collect::<Vec<_>>();
                let batch_axis = batch.batch_axis;
                let value = broadcast_array(
                    batching_context.parent(),
                    batch.value,
                    output_dimensions,
                    output_axes,
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
        if !output.ragged_axes().is_empty() {
            return Err(BatchingError::UnsupportedOperation {
                message: "a bounded ragged array cannot cross the batching transform output boundary".to_string(),
            });
        }
        match (output.batch_axis.axis(), output_batch_axis.axis()) {
            (None, None) => Ok(output.into_value()),
            (Some(_), None) => {
                Err(BatchingError::MismatchedOutputAxes { expected: output_batch_axis, actual: output.batch_axis })
            }
            (_, Some(axis)) => Ok(align_array_batch(context, output, axis)?.into_value()),
        }
    }
}

impl<C> RecursiveBatchingPolicy<C> for ArrayIrBatching
where
    C: Context<Type = ArrayIrType>,
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Operation: BatchableOperation<C, ArrayIrBatching>
        + BatchableOperation<TracingContext<C::Constant, C::Operation>, ArrayIrBatching>
        + From<DynamicBroadcastOperation>
        + From<ConstantOperation<DimensionValue>>
        + From<DimensionSizeOperation>
        + OperationProjection<ArrayType>,
    <C::Operation as OperationProjection<ArrayType>>::Projected: From<TransposeOperation>,
{
    fn restore_batch(
        value: C::Value,
        batch_axis: BatchAxis,
        r#type: &C::Type,
        inputs: &[Self::Batch],
    ) -> Result<Self::Batch, BatchingError> {
        let output = ArrayIrBatch::new(value, batch_axis)?;
        let ArrayIrType::Array(logical_type) = r#type else {
            return Ok(output);
        };
        let packed_value_type = output.value.r#type();
        let packed_type = <&ArrayType>::try_from(packed_value_type.as_ref())?;
        let output_batch_axis = output.batch_axis_position();
        let mut ragged_axes = Vec::new();
        for (logical_axis, dimension) in logical_type.shape().dimensions().iter().enumerate() {
            let Dimension::Dynamic(variable) = dimension else {
                continue;
            };
            let source = inputs.iter().find_map(|input| {
                input
                    .ragged_axes()
                    .iter()
                    .find(|ragged_axis| ragged_axis.dimension() == variable)
                    .map(|ragged_axis| (input, ragged_axis))
            });
            let Some((source, ragged_axis)) = source else {
                let packed_axis =
                    logical_axis + usize::from(output_batch_axis.is_some_and(|batch_axis| batch_axis <= logical_axis));
                if packed_type.shape().dimensions().get(packed_axis) == Some(dimension) {
                    continue;
                }
                return Err(BatchingError::UnsupportedOperation {
                    message: format!(
                        "linear call output {logical_type} has bounded dynamic dimension {variable} but no input \
                         carries its per-item extents",
                    ),
                });
            };
            let source_batch_axis = source.batch_axis_position();
            let extent_axes = ragged_axis
                .extent_axes()
                .iter()
                .map(|extent_axis| {
                    if source_batch_axis == Some(*extent_axis) {
                        return output_batch_axis.ok_or_else(|| BatchingError::InvalidBatchMetadata {
                            message: format!(
                                "linear call output {logical_type} is replicated but its ragged dimension {variable} \
                                 varies along the mapped input axis",
                            ),
                        });
                    }
                    let logical_extent_axis =
                        extent_axis - usize::from(source_batch_axis.is_some_and(|axis| axis < *extent_axis));
                    Ok(logical_extent_axis
                        + usize::from(output_batch_axis.is_some_and(|axis| axis <= logical_extent_axis)))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let physical_axis = logical_axis + usize::from(output_batch_axis.is_some_and(|axis| axis <= logical_axis));
            ragged_axes.push(RaggedAxis::new(
                physical_axis,
                ragged_axis.extents().clone(),
                variable.clone(),
                extent_axes,
            ));
        }
        // The ragged axes recovered above are exactly the dynamic dimensions of the declared per-item type, so the
        // carrier's own derivation reproduces `logical_type` without threading it through the carrier.
        output.with_ragged_axes(ragged_axes)
    }

    fn batch_region(
        context: &BatchingContext<C, Self>,
        region: RegionRef<'_, C::Constant, C::Operation>,
        inputs: Vec<Self::Batch>,
    ) -> Result<Vec<Self::Batch>, BatchingError> {
        let region_mappings = RegionReplayMappings::new();
        region.interpret_with(
            inputs,
            |_, constant| Ok(<Self as BatchingPolicy<C>>::replicated(context.parent().lift(constant.clone())?)),
            |instruction, instruction_inputs| {
                let regions = ReplayRegionDriver::new(region, instruction.regions(), &region_mappings)?;
                let (outputs, evidence) = instruction
                    .operation()
                    .batch(context, &RecursiveBatchingDriver::new(&regions), instruction_inputs)?
                    .into_parts();
                <Self as BatchingPolicy<C>>::validate_operation_outputs(
                    instruction.operation().name(),
                    instruction_inputs,
                    outputs.as_slice(),
                    &evidence,
                )?;
                Ok(outputs)
            },
        )
    }

    fn batch_program(
        context: &BatchingContext<C, Self>,
        region: RegionRef<'_, C::Constant, C::Operation>,
        input_axes: &[BatchAxis],
        output_axes_policy: ProgramBatchingOutputAxesPolicy,
    ) -> Result<Self::BatchedProgram, BatchingError> {
        check_count!("input", input_axes, region.input_types().len(), ProgramError);
        let extent_type = context.axis_extent().r#type();
        let extent_type = <&DimensionType>::try_from(extent_type.as_ref())?.clone();
        let extent_dimension = extent_type.to_dimension();
        let parent_context = TracingContext::<C::Constant, C::Operation>::new();
        let builder = parent_context.builder().clone();

        // The fresh structural trace cannot refer to the parent trace's mapped-extent SSA value directly. Give the
        // transformed region one leading dimension input and carry that same atom out as its leading output. Every
        // inserted packed batch axis below references this input's type identity.
        let (output_atom_ids, output_axes) = {
            let extent = parent_context.input(extent_type.into());
            let extent_atom_id = extent.atom_id()?;
            let batching_context = BatchingContext::<_, ArrayIrBatching>::new(parent_context, extent)
                .with_axis_name(context.axis_name().map(str::to_string))
                .with_axis_sharding(context.axis_sharding().clone());
            let inputs = region
                .input_types()
                .iter()
                .zip(input_axes)
                .map(|(unbatched_type, batch_axis)| {
                    let batched_type = match (unbatched_type, batch_axis.axis()) {
                        (ArrayIrType::Array(array_type), Some(axis)) => {
                            let batched_rank = array_type.rank() + 1;
                            let position = axis.normalize(batched_rank).map_err(|_| {
                                BatchingError::BatchAxisOutOfBounds { r#type: Box::new(array_type.clone()), axis }
                            })?;
                            let mut batched_type =
                                array_type.with_inserted_dimension(position, extent_dimension.clone())?;
                            if let Some(sharding) = array_type.sharding() {
                                batched_type = batched_type
                                    .with_sharding(Some(
                                        sharding
                                            .with_inserted_dimension(position, context.axis_sharding().clone())
                                            .map_err(|error| BatchingError::MisalignedBatchAxes {
                                                message: error.to_string(),
                                            })?,
                                    ))
                                    .map_err(|error| BatchingError::MisalignedBatchAxes {
                                        message: error.to_string(),
                                    })?;
                            }
                            ArrayIrType::Array(batched_type)
                        }
                        _ => unbatched_type.clone(),
                    };
                    let value = batching_context.parent().input(batched_type);
                    ArrayIrBatch::new(value, *batch_axis)
                })
                .collect::<Result<Vec<_>, BatchingError>>()?;

            let region_mappings = RegionReplayMappings::new();
            let outputs = region.interpret_with(
                inputs,
                |_, constant| -> Result<_, BatchingError> {
                    Ok(ArrayIrBatch::replicated(batching_context.parent().lift(constant.clone())?))
                },
                |instruction, instruction_inputs| -> Result<_, BatchingError> {
                    let regions = ReplayRegionDriver::new(region, instruction.regions(), &region_mappings)?;
                    let (outputs, evidence) = instruction
                        .operation()
                        .batch(&batching_context, &RecursiveBatchingDriver::new(&regions), instruction_inputs)?
                        .into_parts();
                    <Self as BatchingPolicy<TracingContext<C::Constant, C::Operation>>>::validate_operation_outputs(
                        instruction.operation().name(),
                        instruction_inputs,
                        outputs.as_slice(),
                        &evidence,
                    )?;
                    Ok(outputs)
                },
            )?;

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
                    Some(target_axis) => align_array_batch(&batching_context, output, target_axis.axis().unwrap())?,
                    None => output,
                };
                check_builders!(&builder, output.value().builder())?;
                output_axes.push(output.batch_axis());
                output_atom_ids.push(output.into_value().atom_id()?);
            }
            Ok::<_, BatchingError>((output_atom_ids, output_axes))
        }?;

        let input_count = region.input_types().len() + 1;
        let output_count = output_atom_ids.len();
        let builder = Rc::try_unwrap(builder).map_err(|_| ProgramError::EscapedProgramBuilder)?.into_inner();
        let program = builder
            .build(output_atom_ids, vec![Placeholder; input_count], vec![Placeholder; output_count])?
            .into_simplified()?;
        Ok(ThreadedExtentBatchedProgram::new(program, output_axes)?)
    }
}

impl<C> NamedAxes for BatchingContext<C, ArrayIrBatching>
where
    C: NamedAxes<Type = ArrayIrType>,
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>
        + ValueProjection<DimensionType, Projected = DimensionValue>,
    C::Operation: BatchableOperation<C, ArrayIrBatching>
        + BatchableOperation<TracingContext<C::Constant, C::Operation>, ArrayIrBatching>
        + From<DynamicBroadcastOperation>
        + From<ConstantOperation<DimensionValue>>
        + From<DimensionSizeOperation>
        + OperationProjection<ArrayType>,
    <C::Operation as OperationProjection<ArrayType>>::Projected: From<TransposeOperation>,
{
    fn named_axis(&self, name: &str) -> Option<NamedAxis> {
        if self.axis_name() == Some(name) {
            let size = match self.parent().resolve(self.axis_extent()) {
                ValueResolution::Constant(axis_extent) => {
                    <C::Constant as ValueProjection<DimensionType>>::into_projected(axis_extent)
                        .ok()
                        .map(|axis_extent| axis_extent.extent())
                }
                ValueResolution::Staged(_) | ValueResolution::Opaque => None,
            };
            Some(NamedAxis::Batched { size })
        } else {
            self.parent().named_axis(name)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use approx::assert_abs_diff_eq;
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::arrays::Array;
    use crate::arrays::dimensions::DimensionValue;
    use crate::arrays::ir::ArrayIrValue;
    use crate::arrays::operations::{ArrayIrOperation, ArrayOperation, DimensionOperation};
    use crate::arrays::sharding::meshes::{LogicalMesh, MeshAxis};
    use crate::arrays::sharding::shardings::ShardingDimension;
    use crate::arrays::types::data::DataType;
    use crate::arrays::types::dimensions::{Dimension, DimensionBounds, DimensionVariable, Shape};
    use crate::batching::{
        Batch, BatchAxisSpecification, BatchingPolicy, BatchingTracer, InterpretableBatchableOperation,
        RecursiveBatchingPolicy, batch,
    };
    use crate::contexts::{EagerContext, StagingContext};
    use crate::differentiation::{
        ForwardModeDifferentiate, LinearCallOperation, LinearizationTracer, ReverseModeDifferentiate,
    };
    use crate::operations::collectives::{
        AllGatherOperation, AllGatherOutputVariance, AllToAllOperation, CollectiveOptions, PSumScatterOperation,
    };
    use crate::operations::random::{RandomAlgorithm, RngBitGeneratorOperation};
    use crate::operations::{
        AddOperation, CollectiveKind, CollectiveOperation, CompareOperation, ComparisonDirection, ConcatenateOperation,
        ConditionOperation, DimensionAddOperation, DimensionFromScalar, DimensionFromScalarOperation, DimensionSize,
        DimensionToScalar, DimensionToScalarOperation, DotDimensionNumbers, DotOperation, DynamicBroadcast,
        DynamicReshapeOperation, IotaOperation, NegOperation, OneLike, OneOperation, PadOperation, Reduce,
        ReduceOperation, ReductionKind, ReshardOperation, ScanOperation, SelectOperation, Slice, WhileOperation,
        ZeroOperation,
    };
    use crate::parameters::Placeholder;
    use crate::programs::{EmptyRegionDriver, ProgramBuilder};
    use crate::tracing::{DomainTracingContext, Trace, TracingContext};

    use super::*;

    #[test]
    fn test_array_batch() {
        let matrix = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let matrix_type = matrix.r#type().into_owned();

        // `new` builds a batched value when the mapped axis is in bounds, and the accessors report the packed value,
        // its packed type, the batch size read off the mapped axis, and the per-item type with that axis removed.
        let batched = ArrayBatch::new(matrix.clone(), Some(0)).unwrap();
        assert_eq!(batched.batch_axis(), BatchAxis::new(0));
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
        // `-1` denotes the final axis and the stored metadata is the canonical nonnegative position.
        let batched_axis_negative_one = ArrayBatch::new(matrix.clone(), BatchAxis::new(-1)).unwrap();
        assert_eq!(batched_axis_negative_one.batch_axis(), BatchAxis::new(1));
        assert_eq!(batched_axis_negative_one.batch_size(), Ok(Some(3)));

        // `new` rejects an out-of-bounds mapped axis.
        assert_eq!(
            ArrayBatch::new(matrix, Some(2)),
            Err(BatchingError::BatchAxisOutOfBounds { r#type: Box::new(matrix_type), axis: Axis::from(2) }),
        );

        let matrix = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let matrix_type = matrix.r#type().into_owned();
        assert_eq!(
            ArrayBatch::new(matrix, BatchAxis::new(-3)),
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
    fn test_array_batch_unbatched_type() {
        let packed_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let packed = Array::from_f64s(packed_type.clone(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // A dense batch reports the packed type with only its mapped axis removed.
        let dense = ArrayBatch::new(packed.clone(), Some(0)).unwrap();
        assert_eq!(dense.unbatched_type(), ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)])));

        // A replicated batch has no mapped axis to remove, so its per-item type is the whole packed type.
        assert_eq!(ArrayBatch::replicated(packed.clone()).unbatched_type(), packed_type);

        // A bounded ragged axis stores its finite physical bound in the packed type while each batch item extends only
        // to its own per-item extent, so the derivation restores the dynamic dimension at that axis.
        let variable = DimensionVariable::new("extent", DimensionBounds::new(0, Some(4)).unwrap());
        let extents = Array::vector(vec![1.0, 3.0]);
        let ragged_axis = RaggedAxis::new(1, extents, variable.clone(), vec![0]);
        let ragged = ArrayBatch::new(packed.clone(), Some(0))
            .unwrap()
            .with_ragged_axes(vec![ragged_axis.clone()])
            .unwrap();
        assert_eq!(ragged.r#type().into_owned(), packed_type);
        assert_eq!(
            ragged.unbatched_type(),
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(variable.clone())])),
        );

        // A ragged axis cannot coincide with the mapped batch axis, which carries the batch items themselves.
        let extents = Array::vector(vec![1.0, 3.0]);
        assert_eq!(
            ArrayBatch::new(packed, Some(0)).unwrap().with_ragged_axes(vec![RaggedAxis::new(
                0,
                extents,
                variable,
                vec![0]
            )]),
            Err(BatchingError::InvalidBatchMetadata {
                message: "ragged axis 0 is invalid for packed array type f64[2, 3] with batch axis axis 0".to_string(),
            }),
        );
    }

    #[test]
    fn test_array_batch_common_batch_size() {
        let matrix = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let vector = Array::vector(vec![7.0, 8.0]);

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
        let batched = ArrayBatch::new(Array::vector(vec![1.0, 2.0]), Some(0)).unwrap();
        assert!(matches!(
            batched.broadcast(0, 2, ShardingDimension::Replicated),
            Err(BatchingError::MisalignedBatchAxes { .. }),
        ));
    }

    #[test]
    fn test_array_batch_move_axis() {
        let matrix = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
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
        let batched = ArrayBatch::new(Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), Some(0)).unwrap();
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
        let batched = ArrayBatch::new(Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), Some(0)).unwrap();

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
            ArrayBatch::new(value, Some(0))
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
            ArrayBatch::new(value, Some(0))
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
        let differently_sharded = ArrayBatch::new(differently_sharded_type, BatchAxis::new(0)).unwrap();
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
            let outputs = AddOperation::new()
                .batch(&context, &EmptyRegionDriver, &[batch.clone(), batch])
                .unwrap()
                .into_parts()
                .0;
            assert_eq!(outputs[0].r#type(), Cow::Borrowed(&batched_type));
            assert_eq!(outputs[0].batch_axis(), BatchAxis::from_position(batch_axis));
        }
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
    fn test_static_array_batching_rejects_ragged_carriers() {
        let packed_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let packed = Array::from_f64s(packed_type, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let variable = DimensionVariable::new("extent", DimensionBounds::new(0, Some(4)).unwrap());
        let ragged = ArrayBatch::new(packed, Some(0))
            .unwrap()
            .with_ragged_axes(vec![RaggedAxis::new(1, Array::vector(vec![1.0, 3.0]), variable, vec![0])])
            .unwrap();
        let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 2);

        // Static array batching never creates ragged carriers and cannot neutralize padding, so it rejects a ragged
        // input instead of returning it unchanged and letting the reduction claim consumption evidence for it.
        assert_eq!(
            StaticArrayBatchingPolicy::mask_reduction_input(&context, &ragged, &[1], ReductionKind::Sum),
            Err(BatchingError::UnsupportedOperation {
                message: "static array batching cannot mask bounded ragged axes".to_string(),
            }),
        );

        // The zero-padding discipline of a contraction needs the same per-item extents and is rejected identically.
        assert_eq!(
            StaticArrayBatchingPolicy::pad_contraction_input(&context, &ragged, &[1]),
            Err(BatchingError::UnsupportedOperation {
                message: "static array batching cannot zero-pad bounded ragged axes".to_string(),
            }),
        );

        // A ragged carrier cannot leave the transform either, matching the composite policy's boundary guard.
        assert_eq!(
            ArrayBatching::materialize_output(&context, ragged, BatchAxis::from_position(0)),
            Err(BatchingError::UnsupportedOperation {
                message: "a bounded ragged array cannot cross the batching transform output boundary".to_string(),
            }),
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
            ArrayBatch::new(Array::from_f64s(r#type.clone(), values), axis).unwrap()
        };

        // Batching rules always receive the active `BatchingContext`, with the underlying work running
        // through its parent context.
        let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 2);

        // Two operands mapped on the same axis add per item, and the output stays mapped on that axis.
        let left = make_batch(&matrix_type, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Some(0));
        let right = make_batch(&matrix_type, vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], Some(0));
        let outputs = AddOperation::new()
            .batch(&context, &EmptyRegionDriver, &[left.clone(), right])
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value(), &Array::matrix(2, 3, vec![11.0, 22.0, 33.0, 44.0, 55.0, 66.0]));

        // A replicated operand is broadcast across the mapped operand's batch before adding.
        let replicated = make_batch(&vector_type, vec![10.0, 20.0, 30.0], None);
        let outputs = AddOperation::new()
            .batch(&context, &EmptyRegionDriver, &[left.clone(), replicated])
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value(), &Array::matrix(2, 3, vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]));

        // Operands mapped on different axes are realigned onto the first mapped operand's axis before adding.
        let transposed_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3), Dimension::Static(2)]));
        let right_axis_one = make_batch(&transposed_type, vec![10.0, 40.0, 20.0, 50.0, 30.0, 60.0], Some(1));
        let outputs = AddOperation::new()
            .batch(&context, &EmptyRegionDriver, &[left, right_axis_one])
            .unwrap()
            .into_parts()
            .0;
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
        let outputs = AddOperation::new().batch(&context, &EmptyRegionDriver, &[left, right]).unwrap().into_parts().0;
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
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(outputs[0].batch_axis(), BatchAxis::replicated());
        assert_eq!(outputs[0].value(), &Array::vector(vec![11.0, 22.0, 33.0]));

        // Unary elementwise operations use the same blanket rule and preserve the mapped input axis.
        let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 3);
        let input = make_batch(&vector_type, vec![1.0, 2.0, 3.0], Some(0));
        let outputs = NegOperation::new().batch(&context, &EmptyRegionDriver, &[input]).unwrap().into_parts().0;
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
                Array::from_f64s(sharded_type.clone(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                BatchAxis::new(0),
            )
            .unwrap();
            let replicated = ArrayBatch::new(
                Array::from_f64s(replicated_type, vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
                BatchAxis::new(0),
            )
            .unwrap();
            let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 2)
                .with_axis_sharding(ShardingDimension::sharded(["x"]));
            let outputs = AddOperation::new()
                .batch(&context, &EmptyRegionDriver, &[sharded, replicated])
                .unwrap()
                .into_parts()
                .0;
            assert_eq!(outputs.len(), 1);
            assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
            assert_eq!(outputs[0].r#type(), Cow::Borrowed(&sharded_type));
            assert_eq!(outputs[0].value().to_f64s(), vec![11.0, 22.0, 33.0, 44.0, 55.0, 66.0]);
        }
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
    fn test_program_batched_carries_dynamic_per_item_dimensions() {
        // Contract: program batching inserts the mapped axis as `Dimension::Static(axis_size)` taken from the caller's
        // `axis_size`, so it never reads a batch extent off the input type the way value-level `Batch::batch` does and
        // therefore never raises `BatchingError::DynamicBatchAxis`. A dynamic per-item dimension is not the batch
        // axis, and it crosses the rewritten boundary unchanged.
        let dynamic = Dimension::Dynamic(DimensionVariable::new("n", DimensionBounds::unbounded()));
        let unbatched_type = ArrayType::new(DataType::F64, Shape::new(vec![dynamic.clone()]));
        let batched_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), dynamic]));
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
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
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
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
        assert_eq!(output.to_f64s(), vec![3.0, 4.0]);
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
            [1.0f64, 2.0, 3.0].into_iter().flat_map(f64::to_le_bytes).collect(),
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
                let %1:f64[] = const 1.0
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

    // TODO(eaplatanios): Review from here onwards.

    #[test]
    fn test_threaded_extent_batched_program_validates_its_boundary() -> Result<(), ProgramError> {
        type TestProgramBuilder = ProgramBuilder<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

        // A threaded boundary always contributes one leading bookkeeping input and output.
        let program = TestProgramBuilder::new().build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            Vec::new(),
            Vec::new(),
            Vec::new(),
        )?;
        let Err(ProgramError::MalformedProgram(message)) = ThreadedExtentBatchedProgram::new(program, Vec::new())
        else {
            panic!("threaded-extent batching accepted a missing bookkeeping boundary");
        };
        assert_eq!(
            message,
            "a structurally batched program with a threaded extent must have a leading input and output",
        );

        // The leading bookkeeping input must be a first-class dimension rather than an arbitrary composite member.
        let mut builder = TestProgramBuilder::new();
        let array = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let program = builder.build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            vec![array],
            vec![Placeholder],
            vec![Placeholder],
        )?;
        let Err(ProgramError::MalformedProgram(message)) = ThreadedExtentBatchedProgram::new(program, Vec::new())
        else {
            panic!("threaded-extent batching accepted a non-dimension bookkeeping input");
        };
        assert_eq!(message, "a structurally batched program's leading threaded-extent input must be a dimension",);

        // The leading bookkeeping output must also be a first-class dimension.
        let mut builder = TestProgramBuilder::new();
        builder
            .add_input(DimensionType::new(DimensionVariable::new("extent", DimensionBounds::new(0, Some(8))?)).into());
        let array = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let program = builder.build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            vec![array],
            vec![Placeholder, Placeholder],
            vec![Placeholder],
        )?;
        let Err(ProgramError::MalformedProgram(message)) = ThreadedExtentBatchedProgram::new(program, Vec::new())
        else {
            panic!("threaded-extent batching accepted a non-dimension bookkeeping output");
        };
        assert_eq!(message, "a structurally batched program's leading threaded-extent output must be a dimension",);

        // A merely compatible dimension output is insufficient: the program must forward the exact input atom.
        let mut builder = TestProgramBuilder::new();
        builder
            .add_input(DimensionType::new(DimensionVariable::new("extent", DimensionBounds::new(0, Some(8))?)).into());
        let other_extent = builder.add_input(
            DimensionType::new(DimensionVariable::new("other_extent", DimensionBounds::new(0, Some(8))?)).into(),
        );
        let program = builder.build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            vec![other_extent],
            vec![Placeholder, Placeholder],
            vec![Placeholder],
        )?;
        let Err(ProgramError::MalformedProgram(message)) = ThreadedExtentBatchedProgram::new(program, Vec::new())
        else {
            panic!("threaded-extent batching accepted a substituted bookkeeping output");
        };
        assert_eq!(
            message,
            "a structurally batched program's leading threaded-extent output must forward its leading input",
        );

        // A well-formed threaded boundary preserves its program and excludes the bookkeeping output from its axes.
        let mut builder = TestProgramBuilder::new();
        let extent = builder
            .add_input(DimensionType::new(DimensionVariable::new("extent", DimensionBounds::new(0, Some(8))?)).into());
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
    fn test_array_ir_batch_entrypoints() -> Result<(), ProgramError> {
        let matrix = ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]));

        // The free transform infers its first-class mapped extent from the packed array input and can move
        // the mapped output axis without exposing the policy at the call site.
        let moved: ArrayIrValue<Array> =
            batch(|row| Ok(row), matrix.clone(), BatchAxis::new(0), BatchAxis::new(1), None)?;
        assert_eq!(moved, ArrayIrValue::Array(Array::matrix(3, 2, vec![1.0_f32, 4.0, 2.0, 5.0, 3.0, 6.0],)),);

        // A replicated array output is dynamically broadcast with the inferred extent operand.
        let replicated = ArrayIrValue::Array(Array::vector(vec![10.0_f32, 20.0, 30.0]));
        let broadcasted: ArrayIrValue<Array> = batch(
            |(_, replicated)| Ok(replicated),
            (matrix.clone(), replicated),
            (BatchAxis::new(0), BatchAxis::replicated()),
            BatchAxis::new(0),
            None,
        )?;
        assert_eq!(
            broadcasted,
            ArrayIrValue::Array(Array::matrix(2, 3, vec![10.0_f32, 20.0, 30.0, 10.0, 20.0, 30.0],)),
        );

        // A named composite specification reaches the policy-selected context, and an explicit first-class extent
        // drives mapped output materialization when every input is replicated.
        let named_extent = BatchAxisSpecification::new(ArrayIrValue::Dimension(DimensionValue::constant(2)?), "items");
        let explicitly_broadcasted: ArrayIrValue<Array> = batch(
            |replicated| {
                assert_eq!(replicated.context().axis_name(), Some("items"));
                Ok(replicated)
            },
            ArrayIrValue::Array(Array::vector(vec![7.0_f32, 8.0, 9.0])),
            BatchAxis::replicated(),
            BatchAxis::new(0),
            named_extent,
        )?;
        assert_eq!(
            explicitly_broadcasted,
            ArrayIrValue::Array(Array::matrix(2, 3, vec![7.0_f32, 8.0, 9.0, 7.0, 8.0, 9.0],)),
        );

        // Exact zero extents use the same first-class extent path and produce an empty mapped dimension.
        let empty_extent = BatchAxisSpecification::with_extent(ArrayIrValue::Dimension(DimensionValue::constant(0)?));
        let empty: ArrayIrValue<Array> = batch(
            |replicated| Ok(replicated),
            ArrayIrValue::Array(Array::vector(vec![7.0_f32, 8.0, 9.0])),
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
        assert!(mismatched.to_string().contains("observed 3=3, size(axis=0)=2"), "{mismatched:?}");

        // A first-class dimension itself cannot be declared mapped at the transform boundary.
        let mapped_input: Result<ArrayIrValue<Array>, BatchingError> = batch(
            |extent| Ok(extent),
            ArrayIrValue::Dimension(DimensionValue::constant(2)?),
            BatchAxis::new(0),
            BatchAxis::replicated(),
            None,
        );
        assert!(matches!(mapped_input, Err(BatchingError::MappedDimension { axis, .. }) if axis == BatchAxis::new(0)));

        // Nested public batching selects the composite policy again from the outer batching context.
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
        type TraceContext = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;
        let trace = TraceContext::new();
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
        let builder = trace.builder().borrow();
        assert_eq!(builder.instructions().len(), 3);
        assert!(matches!(builder.instructions()[0].operation(), ArrayIrOperation::DimensionSize(_),));
        assert!(matches!(
            builder.instructions()[1].operation(),
            ArrayIrOperation::Dimension(DimensionOperation::Constant(_)),
        ));
        assert!(matches!(builder.instructions()[2].operation(), ArrayIrOperation::Broadcast(_),));
        assert_eq!(builder.instructions()[2].inputs().len(), 3);
        assert_eq!(
            staged.r#type().as_ref(),
            &ArrayIrType::Array(ArrayType::new(
                DataType::F32,
                Shape::new(vec![Dimension::Dynamic(batch_variable), Dimension::Static(3)]),
            )),
        );
        drop(builder);

        Ok(())
    }

    #[test]
    fn test_array_ir_batch_folds_static_axes_when_normalizing_input_sharding() -> Result<(), ProgramError> {
        // Renormalizing a mapped input's batch-axis placement stages one dynamic broadcast whose output dimensions
        // are the transform's own extent at the mapped axis and exact constants at statically known axes, so no
        // `dimension_size` read is staged for an extent the type system already knows.
        type TraceContext = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;
        let trace = TraceContext::new();
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
        let sharded = trace.input(sharded_type.clone().into());
        let replicated = trace.input(replicated_type.into());
        let staged = Batch::batch(
            &trace,
            |(_sharded, normalized)| Ok(normalized),
            (sharded, replicated),
            (BatchAxis::new(0), BatchAxis::new(0)),
            BatchAxis::new(0),
            None,
        )?;
        let builder = trace.builder().borrow();
        let operations = builder.instructions().iter().map(|instruction| instruction.operation()).collect::<Vec<_>>();
        assert_eq!(operations.len(), 5);

        // Both mapped inputs still spend explicit extent reads for the mapped axis itself, which the ordered
        // requirement checks against each other.
        assert!(matches!(operations[0], ArrayIrOperation::DimensionSize(_)));
        assert!(matches!(operations[1], ArrayIrOperation::DimensionSize(_)));
        assert!(matches!(operations[2], ArrayIrOperation::Dimension(DimensionOperation::Requirement(_))));

        // Only the trailing static axis of the renormalized input costs an instruction, and it is an exact constant
        // rather than a read. The broadcast consumes the input, the mapped extent, and that constant.
        assert!(matches!(operations[3], ArrayIrOperation::Dimension(DimensionOperation::Constant(_))));
        assert!(matches!(operations[4], ArrayIrOperation::Broadcast(_)));
        assert_eq!(builder.instructions()[4].inputs().len(), 3);

        // Folding the static axis leaves the inferred normalized type exactly the sharded input type.
        assert_eq!(staged.r#type().as_ref(), &ArrayIrType::Array(sharded_type));
        drop(builder);

        Ok(())
    }

    #[test]
    fn test_batch_axis_sharding_normalization_is_not_a_reshard() {
        // Drift gate for the rejected proposal to stage batch-axis sharding normalization as a `reshard` instead of an
        // axis-identity broadcast. Both operations replace an array's placement, but `reshard` does not honor the two
        // parts of the normalized type that batching depends on: it takes its varying-manual axes from the operand
        // rather than from the requested sharding, and it rejects any requested sharding that references an auto mesh
        // axis. If either assertion below stops holding, the normalization can be restaged as a single `reshard` bind.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("a", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap();

        // Normalizing a replicated mapped axis onto a manual placement also makes the value vary along that manual
        // axis. A reshard to the very same target sharding drops that fact, because its inference substitutes the
        // operand's varying-manual axes for the target's.
        let input = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(Sharding::new(mesh.clone(), vec![ShardingDimension::replicated()]).unwrap())
            .unwrap();
        let normalized = normalized_batch_axis_type(&input, 0, &ShardingDimension::sharded(["m"])).unwrap().unwrap();
        assert_eq!(
            normalized.sharding().unwrap().varying_manual_axes().iter().collect::<Vec<_>>(),
            vec!["m"],
            "normalization adds the manual axis introduced by the new placement",
        );
        let resharded = ReshardOperation::new(normalized.sharding().unwrap().clone())
            .infer_output_types(std::slice::from_ref(&input), &[])
            .unwrap();
        assert!(resharded[0].sharding().unwrap().varying_manual_axes().is_empty());
        assert_ne!(resharded[0], normalized);

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
                "reshard cannot target auto mesh axes; use a sharding constraint to hint propagation over auto axes"
                    .to_string(),
            )),
        );
    }

    #[test]
    fn test_array_ir_batching_policy() -> Result<(), ProgramError> {
        type Parent = EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;
        type PolicyContext = BatchingContext<Parent, ArrayIrBatching>;

        let axis_extent = ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap());
        let context = PolicyContext::new(Parent::new(), axis_extent.clone()).with_axis_name("items".to_string());
        assert_eq!(context.axis_extent(), &axis_extent);
        assert_eq!(context.axis_name(), Some("items"));

        // The policy-generic interpretation helper unpacks and repackages composite batches without projecting their
        // member kind or depending on the homogeneous array carrier.
        let direct_input = ArrayIrBatch::new(
            ArrayIrValue::Array(Array::matrix(2, 2, vec![1.0_f32, 2.0, 3.0, 4.0])),
            BatchAxis::new(0),
        )
        .unwrap();
        let [direct_output] = ArrayIrOperation::Array(ArrayOperation::Neg(NegOperation::new()))
            .interpret_with_batch_axes(&context, &[direct_input], &[BatchAxis::new(0)])?
            .try_into()
            .unwrap();
        assert_eq!(direct_output.batch_axis(), BatchAxis::new(0));
        assert_eq!(direct_output.value(), &ArrayIrValue::Array(Array::matrix(2, 2, vec![-1.0_f32, -2.0, -3.0, -4.0])),);

        // The generic frame preserves the existing homogeneous array rule unchanged.
        let input = BatchingTracer::new(
            context.clone(),
            ArrayIrBatch::new(
                ArrayIrValue::Array(Array::matrix(2, 2, vec![1.0_f32, 2.0, 3.0, 4.0])),
                BatchAxis::new(0),
            )
            .unwrap(),
        );
        let [output] = context
            .bind(
                ArrayIrOperation::Array(ArrayOperation::Neg(NegOperation::new())),
                Vec::new(),
                std::slice::from_ref(&input),
            )?
            .try_into()
            .unwrap();
        assert_eq!(output.batch().batch_axis(), BatchAxis::new(0));
        assert_eq!(output.batch().value(), &ArrayIrValue::Array(Array::matrix(2, 2, vec![-1.0_f32, -2.0, -3.0, -4.0])),);

        // Dimension-only and mixed dimension/array boundaries remain replicated under the same generic frame.
        let left = ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap());
        let right = ArrayIrValue::Dimension(DimensionValue::constant(4).unwrap());
        let operation = DimensionAddOperation::new(
            <&DimensionType>::try_from(left.r#type().as_ref()).unwrap(),
            <&DimensionType>::try_from(right.r#type().as_ref()).unwrap(),
        )
        .unwrap();
        let inputs = [
            BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(left)),
            BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(right)),
        ];
        let [dimension] = context
            .bind(ArrayIrOperation::<Array>::Dimension(DimensionOperation::Add(operation)), Vec::new(), &inputs)?
            .try_into()
            .unwrap();
        let scalar = dimension.to_scalar().unwrap().into_batch();
        assert_eq!(scalar.batch_axis(), BatchAxis::replicated());
        assert_eq!(scalar.into_value(), ArrayIrValue::Array(Array::scalar(7_i64)));

        let mapped_dimension = <ArrayIrBatching as BatchingPolicy<Parent>>::batch(
            ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()),
            BatchAxis::new(0),
        );
        assert!(
            matches!(mapped_dimension, Err(BatchingError::MappedDimension { axis, .. }) if axis == BatchAxis::new(0))
        );

        // A staged dynamic mapped extent remains an ordinary SSA operand of the lifted reshape.
        type TraceContext = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;
        let trace = TraceContext::new();
        let batch_variable = DimensionVariable::new("batch", DimensionBounds::new(1, Some(9)).unwrap());
        let batch_type = DimensionType::new(batch_variable.clone());
        let batched_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(batch_variable), Dimension::Static(3)]));
        let batch_extent = trace.input(batch_type.clone().into());
        let input = trace.input(batched_type.clone().into());
        let output_extent = trace.constant(ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap()));
        let batch_extent_id = batch_extent.atom_id().unwrap();
        let input_id = input.atom_id().unwrap();
        let output_extent_id = output_extent.atom_id().unwrap();
        let context =
            BatchingContext::<_, ArrayIrBatching>::new(trace.clone(), batch_extent).with_axis_name("items".to_string());
        assert_eq!(context.named_axis("items"), Some(NamedAxis::Batched { size: None }));
        let inputs = [
            BatchingTracer::new(context.clone(), ArrayIrBatch::new(input, BatchAxis::new(0))?),
            BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(output_extent)),
        ];
        let [output] = context
            .bind(ArrayIrOperation::<Array>::from(DynamicReshapeOperation::new()), Vec::new(), &inputs)?
            .try_into()
            .unwrap();
        let output_id = output.into_batch().into_value().atom_id().unwrap();
        let builder = trace.builder().borrow();
        let [instruction] = builder.instructions() else {
            panic!("expected one lifted reshape instruction");
        };
        assert_eq!(instruction.inputs(), &[input_id, batch_extent_id, output_extent_id]);
        drop(builder);
        let program = trace.builder().borrow().clone().build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            vec![output_id],
            vec![Placeholder, Placeholder],
            vec![Placeholder],
        )?;
        let rendered = program.to_string();
        assert_eq!(
            rendered,
            indoc! {"
                lambda %0:dimension<batch ∈ [1, 9)>, %1:f32[batch, 3] .
                let %2:dimension<3> = const 3
                    %3:f32[batch, 3] = reshape %1 %0 %2
                in (%3)
            "}
            .trim_end(),
        );

        // Generic homogeneous dispatch keeps a dynamic mapped extent as an explicit broadcast operand when an
        // elementwise primitive aligns a replicated operand. The family dispatcher does not name `AddOperation`;
        // adding the primitive to `ArrayOperation` and giving it its ordinary homogeneous rule is sufficient.
        let elementwise_trace = TraceContext::new();
        let batch_variable = DimensionVariable::new("batch", DimensionBounds::new(1, Some(9)).unwrap());
        let batch_extent = elementwise_trace.input(DimensionType::new(batch_variable.clone()).into());
        let mapped = elementwise_trace.input(
            ArrayType::new(
                DataType::F32,
                Shape::new(vec![Dimension::Dynamic(batch_variable.clone()), Dimension::Static(3)]),
            )
            .into(),
        );
        let replicated =
            elementwise_trace.input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)])).into());
        let batch_extent_id = batch_extent.atom_id().unwrap();
        let replicated_id = replicated.atom_id().unwrap();
        let elementwise_context = BatchingContext::<_, ArrayIrBatching>::new(elementwise_trace.clone(), batch_extent);
        let inputs = [
            BatchingTracer::new(elementwise_context.clone(), ArrayIrBatch::new(mapped, BatchAxis::new(0))?),
            BatchingTracer::new(elementwise_context.clone(), ArrayIrBatch::replicated(replicated)),
        ];
        let [output] = elementwise_context
            .bind(ArrayIrOperation::Array(ArrayOperation::Add(AddOperation::new())), Vec::new(), &inputs)?
            .try_into()
            .unwrap();
        let elementwise_output_id = output.batch().value().atom_id().unwrap();
        assert_eq!(output.batch().batch_axis(), BatchAxis::new(0));
        assert_eq!(
            output.r#type().as_ref(),
            &ArrayIrType::Array(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)]),)),
        );
        let elementwise_builder = elementwise_trace.builder().borrow();
        let [dimension, broadcast, add] = elementwise_builder.instructions() else {
            panic!("expected one dimension constant, one dynamic broadcast, and one array add");
        };
        assert!(matches!(dimension.operation(), ArrayIrOperation::Dimension(DimensionOperation::Constant(_)),));
        assert!(matches!(broadcast.operation(), ArrayIrOperation::Broadcast(_)));
        assert_eq!(broadcast.inputs()[0], replicated_id);
        assert_eq!(broadcast.inputs()[1], batch_extent_id);
        assert_eq!(broadcast.inputs().len(), 3);
        assert!(matches!(add.operation(), ArrayIrOperation::Array(ArrayOperation::Add(_))));
        drop(elementwise_builder);
        let elementwise_program = elementwise_trace
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![elementwise_output_id],
                vec![Placeholder, Placeholder, Placeholder],
                vec![Placeholder],
            )?;
        let elementwise_rendered = elementwise_program.to_string();
        assert!(elementwise_rendered.contains("broadcast [output_axes=[1]] %2 %0 %3"), "{elementwise_rendered}",);
        let mut destination = ProgramBuilder::new();
        let imported = destination.import_region(elementwise_program.entry_region_ref());
        assert_eq!(destination.region_ref(imported)?.to_program().to_string(), elementwise_rendered);

        // The same policy owns recursive region replay, so composite instructions no longer receive a plain region
        // driver that cannot re-enter batching.
        let recursive_parent = TraceContext::new();
        let recursive_batch_extent = recursive_parent.input(batch_type.into());
        let recursive_input = recursive_parent.input(batched_type.into());
        let recursive_axis_extent =
            recursive_parent.constant(ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()));
        let recursive_context = BatchingContext::<_, ArrayIrBatching>::new(recursive_parent, recursive_axis_extent);
        let recursive_outputs = ArrayIrBatching::batch_region(
            &recursive_context,
            program.entry_region_ref(),
            vec![ArrayIrBatch::replicated(recursive_batch_extent), ArrayIrBatch::replicated(recursive_input)],
        )?;
        assert_eq!(recursive_outputs.len(), 1);
        assert_eq!(recursive_outputs[0].batch_axis(), BatchAxis::replicated());

        let mut destination = ProgramBuilder::new();
        let imported = destination.import_region(program.entry_region_ref());
        assert_eq!(destination.region_ref(imported)?.to_program().to_string(), rendered);

        Ok(())
    }

    #[test]
    fn test_dynamic_array_ir_elementwise_dispatch_and_alignment() -> Result<(), ProgramError> {
        type TraceContext = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

        // A unary primitive already carrying a non-leading mapped axis stages directly through the generic
        // homogeneous-family arm without any alignment operation.
        let trace = TraceContext::new();
        let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(9))?);
        let extent = trace.input(DimensionType::new(batch.clone()).into());
        let mapped = trace.input(
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3), Dimension::Dynamic(batch.clone())]))
                .into(),
        );
        let context = BatchingContext::<_, ArrayIrBatching>::new(trace.clone(), extent);
        let input = BatchingTracer::new(context.clone(), ArrayIrBatch::new(mapped, BatchAxis::new(1))?);
        let [output] = context
            .bind(
                ArrayIrOperation::Array(ArrayOperation::Neg(NegOperation::new())),
                Vec::new(),
                std::slice::from_ref(&input),
            )?
            .try_into()
            .unwrap();
        assert_eq!(output.batch().batch_axis(), BatchAxis::new(1));
        let builder = trace.builder().borrow();
        let [negate] = builder.instructions() else {
            panic!("expected one generic unary instruction");
        };
        assert!(matches!(negate.operation(), ArrayIrOperation::Array(ArrayOperation::Neg(_))));
        drop(builder);

        // Differently positioned mapped inputs are reconciled by one transpose before generic binary dispatch.
        let trace = TraceContext::new();
        let extent = trace.input(DimensionType::new(batch.clone()).into());
        let left = trace.input(
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(batch.clone()), Dimension::Static(3)]))
                .into(),
        );
        let right = trace.input(
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3), Dimension::Dynamic(batch.clone())]))
                .into(),
        );
        let context = BatchingContext::<_, ArrayIrBatching>::new(trace.clone(), extent);
        let inputs = [
            BatchingTracer::new(context.clone(), ArrayIrBatch::new(left, BatchAxis::new(0))?),
            BatchingTracer::new(context.clone(), ArrayIrBatch::new(right, BatchAxis::new(1))?),
        ];
        let [output] = context
            .bind(ArrayIrOperation::Array(ArrayOperation::Add(AddOperation::new())), Vec::new(), &inputs)?
            .try_into()
            .unwrap();
        assert_eq!(output.batch().batch_axis(), BatchAxis::new(0));
        let builder = trace.builder().borrow();
        let [transpose, add] = builder.instructions() else {
            panic!("expected one transpose followed by one generic binary instruction");
        };
        assert!(matches!(transpose.operation(), ArrayIrOperation::Array(ArrayOperation::Transpose(_)),));
        assert!(matches!(add.operation(), ArrayIrOperation::Array(ArrayOperation::Add(_))));
        drop(builder);

        // Comparison and selection use the same dispatcher and consume the dynamic extent only when a replicated
        // operand must gain the mapped axis.
        let trace = TraceContext::new();
        let extent = trace.input(DimensionType::new(batch.clone()).into());
        let mapped = trace.input(
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(batch.clone()), Dimension::Static(3)]))
                .into(),
        );
        let replicated = trace.input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)])).into());
        let context = BatchingContext::<_, ArrayIrBatching>::new(trace.clone(), extent.clone());
        let inputs = [
            BatchingTracer::new(context.clone(), ArrayIrBatch::new(mapped.clone(), BatchAxis::new(0))?),
            BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(replicated.clone())),
        ];
        let [predicate] = context
            .bind(
                ArrayIrOperation::Array(ArrayOperation::Compare(CompareOperation::new(
                    ComparisonDirection::GreaterThan,
                ))),
                Vec::new(),
                &inputs,
            )?
            .try_into()
            .unwrap();
        assert_eq!(predicate.batch().batch_axis(), BatchAxis::new(0));
        let false_value = BatchingTracer::new(context.clone(), ArrayIrBatch::new(mapped, BatchAxis::new(0))?);
        let true_value = BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(replicated));
        let [selected] = context
            .bind(
                ArrayIrOperation::Array(ArrayOperation::Select(SelectOperation::new())),
                Vec::new(),
                &[predicate, true_value, false_value],
            )?
            .try_into()
            .unwrap();
        assert_eq!(selected.batch().batch_axis(), BatchAxis::new(0));
        assert_eq!(
            trace
                .builder()
                .borrow()
                .instructions()
                .iter()
                .filter(|instruction| matches!(instruction.operation(), ArrayIrOperation::Broadcast(_)))
                .count(),
            2,
        );

        Ok(())
    }

    #[test]
    fn test_composite_condition_batching() -> Result<(), ProgramError> {
        type TraceContext = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

        let unbatched_array_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)]));
        let shared_dimension_type =
            DimensionType::new(DimensionVariable::new("shared", DimensionBounds::new(0, Some(17))?));
        let mut branch_builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let branch_array = branch_builder.add_input(unbatched_array_type.clone().into());
        let branch_dimension = branch_builder.add_input(shared_dimension_type.clone().into());
        let branch_array = branch_builder.add_instruction(
            ArrayIrOperation::Array(ArrayOperation::Neg(NegOperation::new())),
            Vec::new(),
            vec![branch_array],
        )?[0];
        let branch = branch_builder.build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            vec![branch_array, branch_dimension],
            vec![Placeholder, Placeholder],
            vec![Placeholder, Placeholder],
        )?;

        // A replicated predicate keeps one condition. Its transformed branches carry the mapped extent explicitly as
        // leading dimension state, while the reported output-axis metadata excludes that bookkeeping value.
        let trace = TraceContext::new();
        let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(9))?);
        let batch_extent = trace.input(DimensionType::new(batch.clone()).into());
        let predicate = trace.input(ArrayType::scalar(DataType::Boolean).into());
        let batched_array_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(batch.clone()), Dimension::Static(3)]));
        let array = trace.input(batched_array_type.clone().into());
        let shared_dimension = trace.input(shared_dimension_type.clone().into());
        let batch_extent_id = batch_extent.atom_id().unwrap();
        let predicate_id = predicate.atom_id().unwrap();
        let array_id = array.atom_id().unwrap();
        let shared_dimension_id = shared_dimension.atom_id().unwrap();
        let context = BatchingContext::<_, ArrayIrBatching>::new(trace.clone(), batch_extent);
        let outputs = context.bind(
            ArrayIrOperation::Condition(ConditionOperation::new()),
            vec![branch.clone(), branch.clone()],
            &[
                BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(predicate)),
                BatchingTracer::new(context.clone(), ArrayIrBatch::new(array, BatchAxis::new(0))?),
                BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(shared_dimension)),
            ],
        )?;
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].batch().batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[1].batch().batch_axis(), BatchAxis::replicated());

        let output_ids = outputs.iter().map(|output| output.batch().value().atom_id().unwrap()).collect::<Vec<_>>();
        let program = trace.builder().borrow().clone().build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            output_ids,
            vec![Placeholder, Placeholder, Placeholder, Placeholder],
            vec![Placeholder, Placeholder],
        )?;
        let [condition] = program.entry_region().instructions() else {
            panic!("expected exactly one structural condition instruction");
        };
        assert!(matches!(condition.operation(), ArrayIrOperation::Condition(_)));
        assert_eq!(condition.inputs(), &[predicate_id, batch_extent_id, array_id, shared_dimension_id]);
        assert_eq!(condition.regions().len(), 2);
        for region_id in condition.regions() {
            let region = program.region(*region_id)?;
            assert_eq!(
                region.input_types(),
                vec![
                    ArrayIrType::Dimension(DimensionType::new(batch.clone())),
                    ArrayIrType::Array(batched_array_type.clone()),
                    ArrayIrType::Dimension(shared_dimension_type.clone()),
                ],
            );
            assert_eq!(region.output_types(), region.input_types());
        }
        let rendered = program.to_string();
        let mut imported_builder = ProgramBuilder::new();
        let imported = imported_builder.import_region(program.entry_region_ref());
        assert_eq!(imported_builder.region_ref(imported)?.to_program().to_string(), rendered);
        let shared_dimension_value = DimensionValue::new(shared_dimension_type.clone(), 4)?;
        assert_eq!(
            program.interpret(vec![
                ArrayIrValue::Dimension(DimensionValue::new(DimensionType::new(batch.clone()), 2)?),
                ArrayIrValue::Array(Array::scalar(true)),
                ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])),
                ArrayIrValue::Dimension(shared_dimension_value.clone()),
            ])?,
            vec![
                ArrayIrValue::Array(Array::matrix(2, 3, vec![-1.0_f32, -2.0, -3.0, -4.0, -5.0, -6.0])),
                ArrayIrValue::Dimension(shared_dimension_value),
            ],
        );

        // Structural callers may force each output axis independently. Alignment happens while the replayed
        // values are still live tracers, and the leading extent bookkeeping boundary remains separate from the
        // reported axis metadata.
        let forced_trace = TraceContext::new();
        let forced_extent = forced_trace.input(DimensionType::new(batch.clone()).into());
        let forced_context = BatchingContext::<_, ArrayIrBatching>::new(forced_trace, forced_extent);
        let dynamic_natural = <ArrayIrBatching as RecursiveBatchingPolicy<TraceContext>>::batch_program(
            &forced_context,
            branch.entry_region_ref(),
            &[BatchAxis::new(0), BatchAxis::replicated()],
            ProgramBatchingOutputAxesPolicy::Natural,
        )?
        .into_parts()
        .0;
        let forced = <ArrayIrBatching as RecursiveBatchingPolicy<TraceContext>>::batch_program(
            &forced_context,
            branch.entry_region_ref(),
            &[BatchAxis::new(0), BatchAxis::replicated()],
            ProgramBatchingOutputAxesPolicy::AlignEachTo(vec![BatchAxis::new(1), BatchAxis::replicated()]),
        )?;
        assert_eq!(forced.output_axes(), &[BatchAxis::new(1), BatchAxis::replicated()]);
        let (forced, forced_axes) = forced.into_parts();
        assert_eq!(forced_axes, vec![BatchAxis::new(1), BatchAxis::replicated()]);
        assert_eq!(
            forced.output_types(),
            vec![
                ArrayIrType::Dimension(DimensionType::new(batch.clone())),
                ArrayIrType::Array(ArrayType::new(
                    DataType::F32,
                    Shape::new(vec![Dimension::Static(3), Dimension::Dynamic(batch.clone())]),
                )),
                ArrayIrType::Dimension(shared_dimension_type.clone()),
            ],
        );
        assert!(matches!(
            <ArrayIrBatching as RecursiveBatchingPolicy<TraceContext>>::batch_program(
                &forced_context,
                branch.entry_region_ref(),
                &[BatchAxis::new(0), BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            ),
            Err(BatchingError::MappedDimension { r#type, axis })
                if *r#type == shared_dimension_type && axis == BatchAxis::new(0),
        ));

        // Exact static extents use the identical threaded-extent boundary contract and instruction count. Only the
        // boundary types differ, so structural IR does not grow with or specialize on the mapped extent's runtime
        // value.
        let static_trace = TraceContext::new();
        let static_extent_type = DimensionValue::constant(2)?.r#type().into_owned();
        let static_extent = static_trace.input(static_extent_type.clone().into());
        let static_context = BatchingContext::<_, ArrayIrBatching>::new(static_trace, static_extent);
        let static_natural = <ArrayIrBatching as RecursiveBatchingPolicy<TraceContext>>::batch_program(
            &static_context,
            branch.entry_region_ref(),
            &[BatchAxis::new(0), BatchAxis::replicated()],
            ProgramBatchingOutputAxesPolicy::Natural,
        )?
        .into_parts()
        .0;
        assert_eq!(
            static_natural.input_types(),
            vec![
                ArrayIrType::Dimension(static_extent_type),
                ArrayIrType::Array(ArrayType::new(
                    DataType::F32,
                    Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]),
                )),
                ArrayIrType::Dimension(shared_dimension_type.clone()),
            ],
        );
        assert_eq!(static_natural.instructions().len(), dynamic_natural.instructions().len());

        // A second structural pass over the already-batched condition introduces one new leading threaded extent and
        // recursively re-batches the attached branches. The source extent stays an ordinary replicated dimension
        // operand, proving that nested batching does not recover either extent from array metadata.
        let nested_trace = TraceContext::new();
        let outer_batch = DimensionVariable::new("outer_batch", DimensionBounds::new(1, Some(5))?);
        let outer_extent = nested_trace.input(DimensionType::new(outer_batch.clone()).into());
        let nested_context = BatchingContext::<_, ArrayIrBatching>::new(nested_trace, outer_extent);
        let nested = <ArrayIrBatching as RecursiveBatchingPolicy<TraceContext>>::batch_program(
            &nested_context,
            program.entry_region_ref(),
            &[BatchAxis::replicated(), BatchAxis::replicated(), BatchAxis::new(0), BatchAxis::replicated()],
            ProgramBatchingOutputAxesPolicy::Natural,
        )?;
        let (nested, nested_axes) = nested.into_parts();
        assert_eq!(nested_axes, vec![BatchAxis::new(0), BatchAxis::replicated()]);
        assert_eq!(nested.input_types()[0], ArrayIrType::Dimension(DimensionType::new(outer_batch.clone())),);
        assert_eq!(nested.input_types()[1], ArrayIrType::Dimension(DimensionType::new(batch.clone())),);
        assert!(
            nested
                .instructions()
                .iter()
                .any(|instruction| matches!(instruction.operation(), ArrayIrOperation::Condition(_)),)
        );

        // A mapped predicate replays both pure branches and selects array results per item. Equal dimension results
        // remain replicated and are guarded by an explicit equality requirement rather than becoming ragged values.
        let trace = TraceContext::new();
        let batch_extent = trace.input(DimensionType::new(batch.clone()).into());
        let predicate =
            trace.input(ArrayType::new(DataType::Boolean, Shape::new(vec![Dimension::Dynamic(batch.clone())])).into());
        let array = trace.input(batched_array_type.into());
        let shared_dimension = trace.input(shared_dimension_type.clone().into());
        let context = BatchingContext::<_, ArrayIrBatching>::new(trace.clone(), batch_extent);
        let outputs = context.bind(
            ArrayIrOperation::Condition(ConditionOperation::new()),
            vec![branch.clone(), branch],
            &[
                BatchingTracer::new(context.clone(), ArrayIrBatch::new(predicate, BatchAxis::new(0))?),
                BatchingTracer::new(context.clone(), ArrayIrBatch::new(array, BatchAxis::new(0))?),
                BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(shared_dimension)),
            ],
        )?;
        assert_eq!(outputs[0].batch().batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[1].batch().batch_axis(), BatchAxis::replicated());
        let builder = trace.builder().borrow();
        assert!(
            builder
                .instructions()
                .iter()
                .all(|instruction| !matches!(instruction.operation(), ArrayIrOperation::Condition(_))),
        );
        assert!(
            builder.instructions().iter().any(|instruction| matches!(
                instruction.operation(),
                ArrayIrOperation::Array(ArrayOperation::Select(_))
            ),)
        );
        assert!(builder.instructions().iter().any(|instruction| matches!(
            instruction.operation(),
            ArrayIrOperation::Dimension(DimensionOperation::Requirement(_)),
        )));

        Ok(())
    }

    #[test]
    fn test_composite_while_batching() -> Result<(), ProgramError> {
        type Context = EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

        let dimension_type = DimensionType::new(DimensionVariable::new("shared", DimensionBounds::new(0, Some(17))?));
        let mut condition_builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let condition_predicate = condition_builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        condition_builder.add_input(ArrayType::scalar(DataType::F32).into());
        condition_builder.add_input(dimension_type.clone().into());
        let condition = condition_builder.build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            vec![condition_predicate],
            vec![Placeholder, Placeholder, Placeholder],
            vec![Placeholder],
        )?;

        let mut body_builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        body_builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let body_array = body_builder.add_input(ArrayType::scalar(DataType::F32).into());
        let body_dimension = body_builder.add_input(dimension_type.clone().into());
        let false_value = body_builder.add_constant(ArrayIrValue::Array(Array::scalar(false)));
        let negated = body_builder.add_instruction(
            ArrayIrOperation::Array(ArrayOperation::Neg(NegOperation::new())),
            Vec::new(),
            vec![body_array],
        )?[0];
        let body = body_builder.build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            vec![false_value, negated, body_dimension],
            vec![Placeholder, Placeholder, Placeholder],
            vec![Placeholder, Placeholder, Placeholder],
        )?;

        // A scalar predicate controls the whole batch. Array state stays mapped, the dimension carry stays
        // replicated, and the explicit mapped extent crosses both rewritten regions as leading state.
        let context = BatchingContext::<_, ArrayIrBatching>::new(
            Context::new(),
            ArrayIrValue::Dimension(DimensionValue::constant(2)?),
        );
        let outputs = context.bind(
            ArrayIrOperation::While(WhileOperation::new().with_iteration_bound(1)?),
            vec![condition.clone(), body.clone()],
            &[
                BatchingTracer::new(
                    context.clone(),
                    ArrayIrBatch::replicated(ArrayIrValue::Array(Array::scalar(true))),
                ),
                BatchingTracer::new(
                    context.clone(),
                    ArrayIrBatch::new(ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0])), BatchAxis::new(0))?,
                ),
                BatchingTracer::new(
                    context.clone(),
                    ArrayIrBatch::replicated(ArrayIrValue::Dimension(DimensionValue::new(dimension_type.clone(), 4)?)),
                ),
            ],
        )?;
        assert_eq!(outputs.len(), 3);
        assert_eq!(outputs[0].batch().batch_axis(), BatchAxis::replicated());
        assert_eq!(outputs[0].batch().value(), &ArrayIrValue::Array(Array::scalar(false)));
        assert_eq!(outputs[1].batch().batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[1].batch().value(), &ArrayIrValue::Array(Array::vector(vec![-1.0_f32, -2.0])),);
        assert_eq!(outputs[2].batch().batch_axis(), BatchAxis::replicated());
        assert_eq!(
            outputs[2].batch().value(),
            &ArrayIrValue::Dimension(DimensionValue::new(dimension_type.clone(), 4)?),
        );

        // A batch-varying predicate masks the array carries per item while the replicated dimension carry rides through
        // the loop as loop-invariant state. The single permitted iteration updates the active item only: item 0 takes
        // the body's `(false, -1.0)` candidate, while item 1 (whose predicate is already false) keeps its carried
        // `(false, 2.0)`, and the dimension stays replicated at its incoming extent.
        let outputs = context.bind(
            ArrayIrOperation::While(WhileOperation::new().with_iteration_bound(1)?),
            vec![condition.clone(), body.clone()],
            &[
                BatchingTracer::new(
                    context.clone(),
                    ArrayIrBatch::new(ArrayIrValue::Array(Array::vector(vec![true, false])), BatchAxis::new(0))?,
                ),
                BatchingTracer::new(
                    context.clone(),
                    ArrayIrBatch::new(ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0])), BatchAxis::new(0))?,
                ),
                BatchingTracer::new(
                    context.clone(),
                    ArrayIrBatch::replicated(ArrayIrValue::Dimension(DimensionValue::new(dimension_type.clone(), 4)?)),
                ),
            ],
        )?;
        assert_eq!(outputs.len(), 3);
        assert_eq!(outputs[0].batch().batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].batch().value(), &ArrayIrValue::Array(Array::vector(vec![false, false])));
        assert_eq!(outputs[1].batch().batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[1].batch().value(), &ArrayIrValue::Array(Array::vector(vec![-1.0_f32, 2.0])));
        assert_eq!(outputs[2].batch().batch_axis(), BatchAxis::replicated());
        assert_eq!(
            outputs[2].batch().value(),
            &ArrayIrValue::Dimension(DimensionValue::new(dimension_type.clone(), 4)?),
        );

        // Staging retains one direct composite while with explicit threaded extents in both regions. Rendering and
        // import preserve that boundary, and a second vmap recursively re-batches it without unrolling per item.
        type TraceContext = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;
        let trace = TraceContext::new();
        let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(9))?);
        let batch_extent = trace.input(DimensionType::new(batch.clone()).into());
        let predicate = trace.input(ArrayType::scalar(DataType::Boolean).into());
        let array =
            trace.input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(batch.clone())])).into());
        let dimension = trace.input(dimension_type.clone().into());
        let input_ids = [batch_extent.clone(), predicate.clone(), array.clone(), dimension.clone()]
            .map(|input| input.atom_id().unwrap());
        let context = BatchingContext::<_, ArrayIrBatching>::new(trace.clone(), batch_extent);
        let outputs = context.bind(
            ArrayIrOperation::While(WhileOperation::new().with_iteration_bound(1)?),
            vec![condition, body],
            &[
                BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(predicate)),
                BatchingTracer::new(context.clone(), ArrayIrBatch::new(array, BatchAxis::new(0))?),
                BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(dimension)),
            ],
        )?;
        let output_ids = outputs.iter().map(|output| output.batch().value().atom_id().unwrap()).collect::<Vec<_>>();
        let program = trace.builder().borrow().clone().build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            output_ids,
            vec![Placeholder; 4],
            vec![Placeholder; 3],
        )?;
        let [r#while] = program.entry_region().instructions() else {
            panic!("composite while batching should stage exactly one instruction");
        };
        assert!(matches!(r#while.operation(), ArrayIrOperation::While(_)));
        assert_eq!(r#while.inputs(), &[input_ids[0], input_ids[1], input_ids[2], input_ids[3]]);
        assert_eq!(r#while.regions().len(), 2);
        let rendered = program.to_string();
        let mut imported_builder = ProgramBuilder::new();
        let imported = imported_builder.import_region(program.entry_region_ref());
        assert_eq!(imported_builder.region_ref(imported)?.to_program().to_string(), rendered);

        let nested_trace = TraceContext::new();
        let outer = DimensionVariable::new("outer", DimensionBounds::new(1, Some(5))?);
        let outer_extent = nested_trace.input(DimensionType::new(outer.clone()).into());
        let nested_context = BatchingContext::<_, ArrayIrBatching>::new(nested_trace, outer_extent);
        let nested = <ArrayIrBatching as RecursiveBatchingPolicy<TraceContext>>::batch_program(
            &nested_context,
            program.entry_region_ref(),
            &[BatchAxis::replicated(), BatchAxis::replicated(), BatchAxis::new(0), BatchAxis::replicated()],
            ProgramBatchingOutputAxesPolicy::Natural,
        )?;
        assert_eq!(nested.output_axes(), &[BatchAxis::replicated(), BatchAxis::new(0), BatchAxis::replicated()],);
        let (nested, _) = nested.into_parts();
        assert_eq!(
            nested
                .instructions()
                .iter()
                .filter(|instruction| matches!(instruction.operation(), ArrayIrOperation::While(_)))
                .count(),
            1,
        );
        assert_eq!(nested.input_types()[0], ArrayIrType::Dimension(DimensionType::new(outer)));

        Ok(())
    }

    #[test]
    fn test_composite_scan_batching() -> Result<(), ProgramError> {
        type Context = EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

        let dimension_type = DimensionType::new(DimensionVariable::new("shared", DimensionBounds::new(0, Some(17))?));
        let mut body_builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let carry = body_builder.add_input(ArrayType::scalar(DataType::F32).into());
        let dimension = body_builder.add_input(dimension_type.clone().into());
        let item = body_builder.add_input(ArrayType::scalar(DataType::F32).into());
        let next_carry = body_builder.add_instruction(
            ArrayIrOperation::Array(ArrayOperation::Add(AddOperation::new())),
            Vec::new(),
            vec![carry, item],
        )?[0];
        let output = body_builder.add_instruction(
            ArrayIrOperation::Array(ArrayOperation::Neg(NegOperation::new())),
            Vec::new(),
            vec![item],
        )?[0];
        let body = body_builder.build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            vec![next_carry, dimension, output],
            vec![Placeholder, Placeholder, Placeholder],
            vec![Placeholder, Placeholder, Placeholder],
        )?;

        let context = BatchingContext::<_, ArrayIrBatching>::new(
            Context::new(),
            ArrayIrValue::Dimension(DimensionValue::constant(2)?),
        );
        let outputs = context.bind(
            ArrayIrOperation::Scan(ScanOperation::new(2, 3).with_reverse(true)),
            vec![body.clone()],
            &[
                BatchingTracer::new(
                    context.clone(),
                    ArrayIrBatch::new(ArrayIrValue::Array(Array::vector(vec![0.0_f32, 10.0])), BatchAxis::new(0))?,
                ),
                BatchingTracer::new(
                    context.clone(),
                    ArrayIrBatch::replicated(ArrayIrValue::Dimension(DimensionValue::new(dimension_type.clone(), 4)?)),
                ),
                BatchingTracer::new(
                    context.clone(),
                    ArrayIrBatch::new(
                        ArrayIrValue::Array(Array::matrix(3, 2, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])),
                        BatchAxis::new(1),
                    )?,
                ),
            ],
        )?;
        assert_eq!(outputs.len(), 3);
        assert_eq!(outputs[0].batch().batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].batch().value(), &ArrayIrValue::Array(Array::vector(vec![9.0_f32, 22.0])),);
        assert_eq!(outputs[1].batch().batch_axis(), BatchAxis::replicated());
        assert_eq!(
            outputs[1].batch().value(),
            &ArrayIrValue::Dimension(DimensionValue::new(dimension_type.clone(), 4)?),
        );
        assert_eq!(outputs[2].batch().batch_axis(), BatchAxis::new(1));
        assert_eq!(
            outputs[2].batch().value(),
            &ArrayIrValue::Array(Array::matrix(3, 2, vec![-1.0_f32, -2.0, -3.0, -4.0, -5.0, -6.0],)),
        );

        // A zero-length scan never probes its body, preserves both carries, and returns an empty mapped stack.
        let zero_outputs = context.bind(
            ArrayIrOperation::Scan(ScanOperation::new(2, 0)),
            vec![body],
            &[
                BatchingTracer::new(
                    context.clone(),
                    ArrayIrBatch::new(ArrayIrValue::Array(Array::vector(vec![0.0_f32, 10.0])), BatchAxis::new(0))?,
                ),
                BatchingTracer::new(
                    context.clone(),
                    ArrayIrBatch::replicated(ArrayIrValue::Dimension(DimensionValue::new(dimension_type, 4)?)),
                ),
                BatchingTracer::new(
                    context.clone(),
                    ArrayIrBatch::new(
                        ArrayIrValue::Array(Array::new(
                            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(0), Dimension::Static(2)])),
                            Vec::new(),
                        )?),
                        BatchAxis::new(1),
                    )?,
                ),
            ],
        )?;
        assert_eq!(zero_outputs[0].batch().value(), &ArrayIrValue::Array(Array::vector(vec![0.0_f32, 10.0])),);
        assert_eq!(zero_outputs[2].batch().batch_axis(), BatchAxis::new(1));
        assert_eq!(
            zero_outputs[2].batch().value().r#type().as_ref(),
            &ArrayIrType::Array(ArrayType::new(
                DataType::F32,
                Shape::new(vec![Dimension::Static(0), Dimension::Static(2)]),
            )),
        );

        Ok(())
    }

    #[test]
    fn test_composite_condition_batching_rejects_effectful_mapped_predicate() -> Result<(), ProgramError> {
        type TraceContext = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

        let unbatched_array_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)]));
        let left_dimension_type =
            DimensionType::new(DimensionVariable::new("left", DimensionBounds::new(0, Some(17))?));
        let right_dimension_type =
            DimensionType::new(DimensionVariable::new("right", DimensionBounds::new(0, Some(17))?));
        let mut branch_builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let array = branch_builder.add_input(unbatched_array_type.clone().into());
        let left = branch_builder.add_input(left_dimension_type.clone().into());
        let right = branch_builder.add_input(right_dimension_type.clone().into());
        assert!(
            branch_builder
                .add_instruction(
                    DimensionOperation::Requirement(DimensionRequirementOperation::equal(
                        &left_dimension_type,
                        &right_dimension_type,
                    )),
                    Vec::new(),
                    vec![left, right],
                )?
                .is_empty(),
        );
        let branch = branch_builder.build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            vec![array],
            vec![Placeholder, Placeholder, Placeholder],
            vec![Placeholder],
        )?;

        let trace = TraceContext::new();
        let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(9))?);
        let batch_extent = trace.input(DimensionType::new(batch.clone()).into());
        let predicate =
            trace.input(ArrayType::new(DataType::Boolean, Shape::new(vec![Dimension::Dynamic(batch.clone())])).into());
        let array = trace.input(
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(batch), Dimension::Static(3)])).into(),
        );
        let left = trace.input(left_dimension_type.into());
        let right = trace.input(right_dimension_type.into());
        let context = BatchingContext::<_, ArrayIrBatching>::new(trace, batch_extent);
        let error = context
            .bind(
                ArrayIrOperation::Condition(ConditionOperation::new()),
                vec![branch.clone(), branch],
                &[
                    BatchingTracer::new(context.clone(), ArrayIrBatch::new(predicate, BatchAxis::new(0))?),
                    BatchingTracer::new(context.clone(), ArrayIrBatch::new(array, BatchAxis::new(0))?),
                    BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(left)),
                    BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(right)),
                ],
            )
            .unwrap_err();
        assert!(matches!(
            error.downcast_custom::<BatchingError>(),
            Some(BatchingError::UnsupportedOperation { message })
                if message == "cannot batch a condition with a batch-varying predicate and effectful branches because \
                               observable effects cannot be selected per batch item",
        ));

        Ok(())
    }

    #[test]
    fn test_composite_condition_batching_rejects_varying_dimension_result() -> Result<(), ProgramError> {
        type TraceContext = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

        let mut true_builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let true_extent = true_builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(2)?));
        let true_branch = true_builder.build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            vec![true_extent],
            Vec::new(),
            vec![Placeholder],
        )?;
        let mut false_builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let false_extent = false_builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(3)?));
        let false_branch = false_builder.build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            vec![false_extent],
            Vec::new(),
            vec![Placeholder],
        )?;

        let trace = TraceContext::new();
        let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(9))?);
        let batch_extent = trace.input(DimensionType::new(batch.clone()).into());
        let predicate =
            trace.input(ArrayType::new(DataType::Boolean, Shape::new(vec![Dimension::Dynamic(batch)])).into());
        let context = BatchingContext::<_, ArrayIrBatching>::new(trace, batch_extent);
        let error = context
            .bind(
                ArrayIrOperation::Condition(ConditionOperation::new()),
                vec![true_branch, false_branch],
                &[BatchingTracer::new(context.clone(), ArrayIrBatch::new(predicate, BatchAxis::new(0))?)],
            )
            .unwrap_err();
        assert_eq!(error.to_string(), "2 == 3; observed 2=2, 3=3");

        Ok(())
    }

    #[test]
    fn test_dynamic_array_ir_shape_changing_alignment() -> Result<(), ProgramError> {
        type TraceContext = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

        // Concatenation broadcasts a replicated operand with the first-class mapped extent, then passes the explicit
        // concatenated result extent to the mixed operation.
        let trace = TraceContext::new();
        let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(9))?);
        let batch_extent = trace.input(DimensionType::new(batch.clone()).into());
        let mapped = trace.input(
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(batch.clone()), Dimension::Static(2)]))
                .into(),
        );
        let replicated = trace.input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)])).into());
        let result_extent = trace.constant(ArrayIrValue::Dimension(DimensionValue::constant(4)?));
        let batch_extent_id = batch_extent.atom_id().unwrap();
        let replicated_id = replicated.atom_id().unwrap();
        let result_extent_id = result_extent.atom_id().unwrap();
        let context = BatchingContext::<_, ArrayIrBatching>::new(trace.clone(), batch_extent);
        let inputs = [
            BatchingTracer::new(context.clone(), ArrayIrBatch::new(mapped, BatchAxis::new(0))?),
            BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(replicated)),
            BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(result_extent)),
        ];
        let operation = ConcatenateOperation::<ArrayIrType>::from_input_types(
            0,
            &inputs.iter().map(|input| input.batch().unbatched_type().clone()).collect::<Vec<_>>(),
        )?;
        let [output] = context.bind(ArrayIrOperation::from(operation), Vec::new(), &inputs)?.try_into().unwrap();
        assert_eq!(output.batch().batch_axis(), BatchAxis::new(0));
        let builder = trace.builder().borrow();
        let broadcast = builder
            .instructions()
            .iter()
            .find(|instruction| matches!(instruction.operation(), ArrayIrOperation::Broadcast(_)))
            .expect("expected dynamic operand alignment");
        assert!(matches!(broadcast.operation(), ArrayIrOperation::Broadcast(_)));
        assert_eq!(broadcast.inputs()[0], replicated_id);
        assert_eq!(broadcast.inputs()[1], batch_extent_id);
        let concatenate = builder.instructions().last().unwrap();
        assert!(matches!(concatenate.operation(), ArrayIrOperation::Concatenate(_)));
        assert_eq!(concatenate.inputs().last(), Some(&result_extent_id));
        drop(builder);

        // A mapped padding scalar forces the replicated operand through the same dynamic alignment path. Every pad
        // in the mask decomposition consumes the mapped extent explicitly.
        let trace = TraceContext::new();
        let batch_extent = trace.input(DimensionType::new(batch.clone()).into());
        let operand = trace.input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)])).into());
        let padding =
            trace.input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(batch.clone())])).into());
        let result_extent = trace.constant(ArrayIrValue::Dimension(DimensionValue::constant(3)?));
        let batch_extent_id = batch_extent.atom_id().unwrap();
        let operand_id = operand.atom_id().unwrap();
        let context = BatchingContext::<_, ArrayIrBatching>::new(trace.clone(), batch_extent);
        let inputs = [
            BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(operand)),
            BatchingTracer::new(context.clone(), ArrayIrBatch::new(padding, BatchAxis::new(0))?),
            BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(result_extent)),
        ];
        let [output] = context
            .bind(ArrayIrOperation::from(PadOperation::new(vec![1], vec![0], vec![0])?), Vec::new(), &inputs)?
            .try_into()
            .unwrap();
        assert_eq!(output.batch().batch_axis(), BatchAxis::new(0));
        let builder = trace.builder().borrow();
        let broadcast = builder
            .instructions()
            .iter()
            .find(|instruction| matches!(instruction.operation(), ArrayIrOperation::Broadcast(_)))
            .expect("expected dynamic operand alignment");
        assert_eq!(broadcast.inputs()[0], operand_id);
        assert_eq!(broadcast.inputs()[1], batch_extent_id);
        assert!(
            builder
                .instructions()
                .iter()
                .filter(|instruction| matches!(instruction.operation(), ArrayIrOperation::Pad(_)))
                .all(|instruction| instruction.inputs().contains(&batch_extent_id)),
        );
        drop(builder);

        // Matching-axis collective batching consumes a complete logical result shape. A replicated operand is
        // materialized along the mapped axis from those extents, dynamic unchanged axes keep their boundary-provided
        // identity, and the rule introduces no metadata read from the source array.
        let trace = TraceContext::new();
        let batch_extent = trace.input(DimensionType::new(batch.clone()).into());
        let sequence = DimensionVariable::new("sequence", DimensionBounds::new(1, Some(17))?);
        let width = DimensionVariable::new("width", DimensionBounds::new(1, Some(33))?);
        let gathered = DimensionVariable::new("gathered", DimensionBounds::new(1, Some(65))?);
        let input = trace.input(
            ArrayType::new(
                DataType::F32,
                Shape::new(vec![Dimension::Dynamic(sequence), Dimension::Dynamic(width.clone())]),
            )
            .into(),
        );
        let gathered_extent = trace.input(DimensionType::new(gathered).into());
        let width_extent = trace.input(DimensionType::new(width).into());
        let context =
            BatchingContext::<_, ArrayIrBatching>::new(trace.clone(), batch_extent).with_axis_name("items".to_string());
        let inputs = [
            BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(input)),
            BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(gathered_extent)),
            BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(width_extent)),
        ];
        let [output] = context
            .bind(
                ArrayIrOperation::AllGather(AllGatherOperation::new(
                    "items".to_string(),
                    4,
                    0,
                    CollectiveOptions::tiled(),
                    AllGatherOutputVariance::Varying,
                )),
                Vec::new(),
                &inputs,
            )?
            .try_into()
            .unwrap();
        assert!(output.batch().batch_axis().is_replicated());
        let builder = trace.builder().borrow();
        assert!(
            builder
                .instructions()
                .iter()
                .all(|instruction| !matches!(instruction.operation(), ArrayIrOperation::DimensionSize(_))),
        );
        assert!(
            builder
                .instructions()
                .iter()
                .all(|instruction| !matches!(instruction.operation(), ArrayIrOperation::AllGather(_))),
        );
        assert!(
            builder
                .instructions()
                .iter()
                .any(|instruction| matches!(instruction.operation(), ArrayIrOperation::Reshape(_))),
        );
        drop(builder);

        // Distinct-axis all-to-all derives its temporary pre-exchange shape from the supplied result extents and the
        // mapped extent using ordinary dimension arithmetic; it likewise never reads the source array shape.
        let trace = TraceContext::new();
        let batch_extent = trace.input(DimensionType::new(batch.clone()).into());
        let input_split = DimensionVariable::new("input_split", DimensionBounds::new(1, Some(65))?);
        let input_concat = DimensionVariable::new("input_concat", DimensionBounds::new(1, Some(65))?);
        let output_split = DimensionVariable::new("output_split", DimensionBounds::new(1, Some(65))?);
        let output_concat = DimensionVariable::new("output_concat", DimensionBounds::new(1, Some(129))?);
        let input = trace.input(
            ArrayType::new(
                DataType::F32,
                Shape::new(vec![
                    Dimension::Dynamic(batch.clone()),
                    Dimension::Dynamic(input_split),
                    Dimension::Dynamic(input_concat),
                ]),
            )
            .into(),
        );
        let output_split = trace.input(DimensionType::new(output_split).into());
        let output_concat = trace.input(DimensionType::new(output_concat).into());
        let context =
            BatchingContext::<_, ArrayIrBatching>::new(trace.clone(), batch_extent).with_axis_name("items".to_string());
        let outputs = context.bind(
            ArrayIrOperation::AllToAll(AllToAllOperation::new(
                "items".to_string(),
                4,
                0,
                1,
                CollectiveOptions::tiled(),
            )),
            Vec::new(),
            &[
                BatchingTracer::new(context.clone(), ArrayIrBatch::new(input, BatchAxis::new(0))?),
                BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(output_split)),
                BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(output_concat)),
            ],
        )?;
        assert_eq!(outputs[0].batch().batch_axis(), BatchAxis::new(0));
        let builder = trace.builder().borrow();
        assert!(
            builder
                .instructions()
                .iter()
                .all(|instruction| !matches!(instruction.operation(), ArrayIrOperation::DimensionSize(_))),
        );
        assert!(builder.instructions().iter().any(|instruction| matches!(
            instruction.operation(),
            ArrayIrOperation::Dimension(DimensionOperation::Mul(_)),
        )));
        assert!(builder.instructions().iter().any(|instruction| matches!(
            instruction.operation(),
            ArrayIrOperation::Dimension(DimensionOperation::DivFloor(_)),
        )));
        drop(builder);

        // A collective over a different named axis is forwarded as the same mixed operation. Only its physical axis
        // index and complete result shape are lifted around the current mapped axis.
        let trace = TraceContext::new();
        let batch_extent = trace.input(DimensionType::new(batch.clone()).into());
        let logical_extent = DimensionVariable::new("logical", DimensionBounds::new(1, Some(17))?);
        let result_extent = DimensionVariable::new("result", DimensionBounds::new(1, Some(33))?);
        let input = trace.input(
            ArrayType::new(
                DataType::F32,
                Shape::new(vec![Dimension::Dynamic(logical_extent), Dimension::Dynamic(batch), Dimension::Static(3)]),
            )
            .into(),
        );
        let result_extent = trace.input(DimensionType::new(result_extent).into());
        let width_extent = trace.constant(ArrayIrValue::Dimension(DimensionValue::constant(3)?));
        let batch_extent_id = batch_extent.atom_id().unwrap();
        let context =
            BatchingContext::<_, ArrayIrBatching>::new(trace.clone(), batch_extent).with_axis_name("outer".to_string());
        let outputs = context.bind(
            ArrayIrOperation::AllGather(AllGatherOperation::new(
                "inner".to_string(),
                2,
                0,
                CollectiveOptions::tiled(),
                AllGatherOutputVariance::Varying,
            )),
            Vec::new(),
            &[
                BatchingTracer::new(context.clone(), ArrayIrBatch::new(input, BatchAxis::new(1))?),
                BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(result_extent)),
                BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(width_extent)),
            ],
        )?;
        assert_eq!(outputs[0].batch().batch_axis(), BatchAxis::new(1));
        let builder = trace.builder().borrow();
        let collective = builder.instructions().last().unwrap();
        let ArrayIrOperation::AllGather(operation) = collective.operation() else {
            panic!("expected a forwarded all-gather");
        };
        assert_eq!(operation.concat_axis(), 0);
        assert_eq!(collective.inputs().len(), 4);
        assert_eq!(collective.inputs()[2], batch_extent_id);
        assert!(
            builder
                .instructions()
                .iter()
                .all(|instruction| !matches!(instruction.operation(), ArrayIrOperation::DimensionSize(_))),
        );

        Ok(())
    }

    #[test]
    fn test_executable_linear_call_batching_threads_the_mapped_extent() -> Result<(), ProgramError> {
        type TestProgram =
            Program<ArrayIrValue<Array>, ArrayIrOperation<Array>, Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>;

        let residual_type = DimensionType::new(DimensionVariable::new("residual", DimensionBounds::new(0, Some(9))?));
        let array_type = ArrayType::scalar(DataType::F64);
        let identity_region = || -> Result<TestProgram, ProgramError> {
            let mut builder = ProgramBuilder::new();
            builder.add_input(residual_type.clone().into());
            let linear = builder.add_input(array_type.clone().into());
            builder.build(vec![linear], vec![Placeholder; 2], vec![Placeholder])
        };
        let forward = identity_region()?;
        let transpose = identity_region()?;
        let residual = ArrayIrValue::Dimension(DimensionValue::new(residual_type, 3)?);
        let linear = ArrayIrValue::Array(Array::vector(vec![2.0_f64, 5.0]));

        // The dimension residual remains replicated while the linear input and output carry the inferred mapped
        // extent. Both attached regions are structurally batched with that extent threaded through their boundary.
        let output: ArrayIrValue<Array> = batch(
            |(residual, linear)| {
                let outputs = residual.context().bind(
                    ArrayIrOperation::LinearCall(LinearCallOperation::new(1)),
                    vec![forward, transpose],
                    &[residual.clone(), linear],
                )?;
                Ok(outputs.into_iter().next().unwrap())
            },
            (residual, linear.clone()),
            (BatchAxis::replicated(), BatchAxis::new(0)),
            BatchAxis::new(0),
            None,
        )?;
        assert_eq!(output, linear);

        Ok(())
    }

    #[test]
    fn test_array_ir_batching() {
        type Parent = EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;
        fn assert_batchable<C: Context<Type = ArrayIrType>, O: BatchableOperation<C, ArrayIrBatching>>() {}
        assert_batchable::<Parent, ArrayIrOperation<Array>>();

        let dimension_type =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::new(0, Some(9)).unwrap()));
        let dimension = ArrayIrValue::<Array>::Dimension(DimensionValue::new(dimension_type.clone(), 4).unwrap());
        assert_eq!(
            ArrayIrBatch::new(dimension.clone(), BatchAxis::new(0)),
            Err(BatchingError::MappedDimension { r#type: Box::new(dimension_type.clone()), axis: BatchAxis::new(0) }),
        );
        let negative_axis_batch = ArrayIrBatch::new(
            ArrayIrValue::Array(Array::matrix(2, 3, vec![0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0])),
            BatchAxis::new(-2),
        )
        .unwrap();
        assert_eq!(negative_axis_batch.batch_axis(), BatchAxis::new(0));
        assert_eq!(
            negative_axis_batch.unbatched_type(),
            ArrayIrType::Array(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)]),)),
        );

        let context = BatchingContext::<_, ArrayIrBatching>::new(
            Parent::new(),
            ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()),
        )
        .with_axis_name("items".to_string())
        .with_axis_sharding(ShardingDimension::Unconstrained);
        assert_eq!(context.axis_name(), Some("items"));
        assert_eq!(context.axis_sharding(), &ShardingDimension::Unconstrained);

        let extent_value = DimensionValue::constant(3).unwrap();
        let dynamic_zero = ArrayIrOperation::<Array>::from(ZeroOperation::new(ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Dynamic(extent_value.r#type().variable().clone())]),
        )));
        let extent = ArrayIrValue::Dimension(extent_value);
        let dynamic_zero_output = dynamic_zero
            .batch(&context, &EmptyRegionDriver, &[ArrayIrBatch::replicated(extent)])
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(dynamic_zero_output.len(), 1);
        assert_eq!(dynamic_zero_output[0].batch_axis(), BatchAxis::replicated());
        assert_eq!(dynamic_zero_output[0].value(), &ArrayIrValue::Array(Array::vector(vec![0.0_f32, 0.0, 0.0])),);

        let extent_value = DimensionValue::constant(3).unwrap();
        let dynamic_one = ArrayIrOperation::<Array>::from(OneOperation::new(ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Dynamic(extent_value.r#type().variable().clone())]),
        )));
        let dynamic_one_output = dynamic_one
            .batch(&context, &EmptyRegionDriver, &[ArrayIrBatch::replicated(ArrayIrValue::Dimension(extent_value))])
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(dynamic_one_output.len(), 1);
        assert_eq!(dynamic_one_output[0].batch_axis(), BatchAxis::replicated());
        assert_eq!(dynamic_one_output[0].value(), &ArrayIrValue::Array(Array::vector(vec![1.0_f32, 1.0, 1.0])),);

        let extent_value = DimensionValue::constant(3).unwrap();
        let dynamic_iota = ArrayIrOperation::<Array>::from(
            IotaOperation::new(
                ArrayType::new(
                    DataType::I32,
                    Shape::new(vec![Dimension::Dynamic(extent_value.r#type().variable().clone())]),
                ),
                0,
            )
            .unwrap(),
        );
        let dynamic_iota_output = dynamic_iota
            .batch(&context, &EmptyRegionDriver, &[ArrayIrBatch::replicated(ArrayIrValue::Dimension(extent_value))])
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(dynamic_iota_output.len(), 1);
        assert_eq!(dynamic_iota_output[0].batch_axis(), BatchAxis::replicated());
        assert_eq!(
            dynamic_iota_output[0].value(),
            &ArrayIrValue::Array(
                Array::from_elements(
                    ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Static(3)])),
                    &[0i32, 1, 2],
                )
                .unwrap(),
            ),
        );

        let mapped_type =
            DimensionType::new(DimensionVariable::new("mapped_extent", DimensionBounds::new(1, Some(5)).unwrap()));
        let mapped_extent = ArrayIrBatch {
            value: ArrayIrValue::Dimension(DimensionValue::new(mapped_type.clone(), 3).unwrap()),
            batch_axis: BatchAxis::new(0),
            mapped_dimension: None,
            ragged_axes: Vec::new(),
        };
        let mapped_extent_error = BatchingError::UnsupportedOperation {
            message: "member operand 0 of type dimension<mapped_extent ∈ [1, 5)> must be replicated but is mapped at \
                      axis 0"
                .to_string(),
        };
        assert_eq!(
            dynamic_zero.batch(&context, &EmptyRegionDriver, &[mapped_extent.clone()]),
            Err(mapped_extent_error.clone()),
        );
        assert_eq!(
            dynamic_one.batch(&context, &EmptyRegionDriver, &[mapped_extent.clone()]),
            Err(mapped_extent_error.clone()),
        );
        assert_eq!(dynamic_iota.batch(&context, &EmptyRegionDriver, &[mapped_extent]), Err(mapped_extent_error),);

        // The composite boundary forwards the mapped-axis name into homogeneous array rules, allowing the matching
        // collective to consume the mapped axis instead of incorrectly forwarding an unbound collective.
        let collective_input =
            ArrayIrBatch::new(ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0])), BatchAxis::new(0)).unwrap();
        let collective = ArrayIrOperation::<Array>::from(ArrayOperation::Collective(CollectiveOperation::new(
            "items".to_string(),
            CollectiveKind::PSum,
        )));
        let collective_output =
            collective.batch(&context, &EmptyRegionDriver, &[collective_input]).unwrap().into_parts().0;
        assert_eq!(collective_output.len(), 1);
        assert_eq!(collective_output[0].batch_axis(), BatchAxis::replicated());
        assert_eq!(collective_output[0].value(), &ArrayIrValue::Array(Array::scalar(3.0_f32)));

        let all_gather = ArrayIrOperation::<Array>::from(AllGatherOperation::new(
            "items".to_string(),
            2,
            0,
            CollectiveOptions::tiled(),
            AllGatherOutputVariance::Varying,
        ));
        let all_gather_input = ArrayIrBatch::new(
            ArrayIrValue::Array(Array::matrix(2, 2, vec![1.0_f32, 2.0, 3.0, 4.0])),
            BatchAxis::new(0),
        )
        .unwrap();
        let all_gather_extent = ArrayIrBatch::replicated(ArrayIrValue::Dimension(DimensionValue::constant(4).unwrap()));
        let all_gather_output = all_gather
            .batch(&context, &EmptyRegionDriver, &[all_gather_input, all_gather_extent])
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(all_gather_output.len(), 1);
        assert_eq!(all_gather_output[0].batch_axis(), BatchAxis::replicated());
        assert_eq!(all_gather_output[0].value(), &ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0, 4.0])),);

        let psum_scatter = ArrayIrOperation::<Array>::from(PSumScatterOperation::new(
            "items".to_string(),
            2,
            0,
            CollectiveOptions::tiled(),
        ));
        let psum_scatter_input = ArrayIrBatch::new(
            ArrayIrValue::Array(Array::matrix(2, 4, vec![1.0_f32, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0])),
            BatchAxis::new(0),
        )
        .unwrap();
        let psum_scatter_extent =
            ArrayIrBatch::replicated(ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()));
        let psum_scatter_output = psum_scatter
            .batch(&context, &EmptyRegionDriver, &[psum_scatter_input, psum_scatter_extent])
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(psum_scatter_output.len(), 1);
        assert_eq!(psum_scatter_output[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(
            psum_scatter_output[0].value(),
            &ArrayIrValue::Array(Array::matrix(2, 2, vec![11.0_f32, 22.0, 33.0, 44.0])),
        );

        let all_to_all = ArrayIrOperation::<Array>::from(AllToAllOperation::new(
            "items".to_string(),
            2,
            0,
            0,
            CollectiveOptions::tiled(),
        ));
        let all_to_all_input = ArrayIrBatch::new(
            ArrayIrValue::Array(Array::matrix(2, 4, vec![1.0_f32, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0])),
            BatchAxis::new(0),
        )
        .unwrap();
        let all_to_all_extent = ArrayIrBatch::replicated(ArrayIrValue::Dimension(DimensionValue::constant(4).unwrap()));
        let all_to_all_output = all_to_all
            .batch(&context, &EmptyRegionDriver, &[all_to_all_input, all_to_all_extent])
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(all_to_all_output.len(), 1);
        assert_eq!(all_to_all_output[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(
            all_to_all_output[0].value(),
            &ArrayIrValue::Array(Array::matrix(2, 4, vec![1.0_f32, 2.0, 10.0, 20.0, 3.0, 4.0, 30.0, 40.0])),
        );

        // Rank-changing collective modes use the same complete axis-ordered extent signature. All-gather consumes
        // the mapped axis into a new logical axis, while sum-scatter and all-to-all re-map their materialized result.
        let untiled_input = || {
            ArrayIrBatch::new(ArrayIrValue::Array(Array::matrix(2, 2, vec![1.0_f32, 2.0, 3.0, 4.0])), BatchAxis::new(0))
                .unwrap()
        };
        let extent_two = || ArrayIrBatch::replicated(ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()));
        let untiled_gather = ArrayIrOperation::<Array>::from(AllGatherOperation::new(
            "items".to_string(),
            2,
            1,
            CollectiveOptions::default(),
            AllGatherOutputVariance::Varying,
        ));
        let gathered = untiled_gather
            .batch(&context, &EmptyRegionDriver, &[untiled_input(), extent_two(), extent_two()])
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(gathered[0].batch_axis(), BatchAxis::replicated());
        assert_eq!(gathered[0].value(), &ArrayIrValue::Array(Array::matrix(2, 2, vec![1.0_f32, 3.0, 2.0, 4.0])),);

        let untiled_scatter = ArrayIrOperation::<Array>::from(PSumScatterOperation::new(
            "items".to_string(),
            2,
            0,
            CollectiveOptions::default(),
        ));
        let scattered = untiled_scatter.batch(&context, &EmptyRegionDriver, &[untiled_input()]).unwrap().into_parts().0;
        assert_eq!(scattered[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(scattered[0].value(), &ArrayIrValue::Array(Array::vector(vec![4.0_f32, 6.0])));

        let untiled_exchange = ArrayIrOperation::<Array>::from(AllToAllOperation::new(
            "items".to_string(),
            2,
            0,
            0,
            CollectiveOptions::default(),
        ));
        let exchanged = untiled_exchange
            .batch(&context, &EmptyRegionDriver, &[untiled_input(), extent_two()])
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(exchanged[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(exchanged[0].value(), &ArrayIrValue::Array(Array::matrix(2, 2, vec![1.0_f32, 3.0, 2.0, 4.0])),);

        // Every rule that consumes a first-class dimension preserves the same typed mapped-dimension diagnostic,
        // even if a malformed internal batch bypasses the public constructor's equivalent boundary check.
        let mapped_dimension = ArrayIrBatch {
            value: dimension.clone(),
            batch_axis: BatchAxis::new(0),
            mapped_dimension: None,
            ragged_axes: Vec::new(),
        };
        let dimension_to_scalar = ArrayIrOperation::<Array>::from(DimensionToScalarOperation);
        assert_eq!(
            dimension_to_scalar.batch(&context, &EmptyRegionDriver, std::slice::from_ref(&mapped_dimension)),
            Err(BatchingError::MappedDimension { r#type: Box::new(dimension_type.clone()), axis: BatchAxis::new(0) }),
        );
        let comparison = ArrayIrOperation::<Array>::from(CompareOperation::new(ComparisonDirection::LessThan));
        let comparison_right = ArrayIrValue::Dimension(DimensionValue::new(dimension_type.clone(), 5).unwrap());
        assert_eq!(
            comparison.batch(
                &context,
                &EmptyRegionDriver,
                &[ArrayIrBatch::replicated(dimension.clone()), ArrayIrBatch::replicated(comparison_right.clone()),],
            ),
            Ok(vec![ArrayIrBatch::replicated(ArrayIrValue::Array(Array::scalar(true)))].into()),
        );
        assert_eq!(
            comparison.batch(
                &context,
                &EmptyRegionDriver,
                &[mapped_dimension.clone(), ArrayIrBatch::replicated(comparison_right.clone())],
            ),
            Err(BatchingError::MappedDimension { r#type: Box::new(dimension_type.clone()), axis: BatchAxis::new(0) }),
        );
        let mapped_comparison_right = ArrayIrBatch {
            value: comparison_right,
            batch_axis: BatchAxis::new(0),
            mapped_dimension: None,
            ragged_axes: Vec::new(),
        };
        assert_eq!(
            comparison.batch(
                &context,
                &EmptyRegionDriver,
                &[ArrayIrBatch::replicated(dimension.clone()), mapped_comparison_right],
            ),
            Err(BatchingError::MappedDimension { r#type: Box::new(dimension_type.clone()), axis: BatchAxis::new(0) }),
        );
        let dimension_add = ArrayIrOperation::<Array>::from(DimensionOperation::Add(
            DimensionAddOperation::new(&dimension_type, &dimension_type).unwrap(),
        ));
        assert_eq!(
            dimension_add.batch(
                &context,
                &EmptyRegionDriver,
                &[mapped_dimension, ArrayIrBatch::replicated(dimension.clone())],
            ),
            Err(BatchingError::MappedDimension { r#type: Box::new(dimension_type.clone()), axis: BatchAxis::new(0) }),
        );

        let gateway_variable = DimensionVariable::new("gateway", DimensionBounds::new(0, Some(9)).unwrap());
        let gateway_operation =
            ArrayIrOperation::<Array>::from(DimensionFromScalarOperation::new(gateway_variable.clone()));
        let gateway_output = gateway_operation
            .batch(&context, &EmptyRegionDriver, &[ArrayIrBatch::replicated(ArrayIrValue::Array(Array::scalar(4_i32)))])
            .unwrap()
            .into_parts()
            .0;
        let [gateway_output] = gateway_output.as_slice() else {
            panic!("expected one dimension-from-scalar batching result");
        };
        assert_eq!(gateway_output.batch_axis(), BatchAxis::replicated());
        assert_eq!(
            gateway_output.value(),
            &ArrayIrValue::Dimension(DimensionValue::new(DimensionType::new(gateway_variable.clone()), 4).unwrap(),),
        );
        let mapped_gateway_input =
            ArrayIrBatch::new(ArrayIrValue::Array(Array::vector(vec![4_i32, 5_i32])), BatchAxis::new(0)).unwrap();
        assert_eq!(
            gateway_operation.batch(&context, &EmptyRegionDriver, &[mapped_gateway_input]),
            Ok(vec![
                ArrayIrBatch::mapped_dimension(
                    ArrayIrValue::Array(Array::vector(vec![4_i64, 5_i64])),
                    BatchAxis::new(0),
                    DimensionType::new(gateway_variable.clone()),
                )
                .unwrap(),
            ]
            .into()),
        );
        assert_eq!(
            gateway_operation.batch(&context, &EmptyRegionDriver, &[]),
            Err(BatchingError::from(ProgramError::InvalidInputCount { expected: 1, actual: 0 })),
        );

        let zero = ArrayIrOperation::<Array>::from(ZeroOperation::new(ArrayType::scalar(DataType::F32)));
        assert_eq!(
            zero.batch(
                &context,
                &EmptyRegionDriver,
                &[ArrayIrBatch::replicated(ArrayIrValue::Array(Array::scalar(1.0_f32)))],
            ),
            Err(BatchingError::from(ProgramError::InvalidInputCount { expected: 0, actual: 1 })),
        );

        let reshape = ArrayIrOperation::<Array>::from(DynamicReshapeOperation::new());
        let reshape_input = ArrayIrBatch::new(
            ArrayIrValue::Array(Array::matrix(2, 6, (0..12).map(|value| value as f32).collect())),
            BatchAxis::new(0),
        )
        .unwrap();
        let first_extent = ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap());
        let first_extent_type = first_extent.r#type().into_owned();
        let second_extent = ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap());
        let reshape_output = reshape
            .batch(
                &context,
                &EmptyRegionDriver,
                &[
                    reshape_input,
                    ArrayIrBatch::replicated(first_extent.clone()),
                    ArrayIrBatch::replicated(second_extent.clone()),
                ],
            )
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(reshape_output.len(), 1);
        assert_eq!(reshape_output[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(
            reshape_output[0].value(),
            &ArrayIrValue::Array(Array::from_f64s(
                ArrayType::new(
                    DataType::F32,
                    Shape::new(vec![Dimension::Static(2), Dimension::Static(2), Dimension::Static(3)]),
                ),
                (0..12).map(|value| value as f64).collect(),
            )),
        );
        assert_eq!(
            reshape.batch(
                &context,
                &EmptyRegionDriver,
                &[
                    ArrayIrBatch::replicated(ArrayIrValue::Array(Array::vector(vec![
                        0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0,
                    ]))),
                    ArrayIrBatch {
                        value: first_extent,
                        batch_axis: BatchAxis::new(0),
                        mapped_dimension: None,
                        ragged_axes: Vec::new(),
                    },
                    ArrayIrBatch::replicated(second_extent),
                ],
            ),
            Err(BatchingError::MappedDimension {
                r#type: Box::new(<&DimensionType>::try_from(&first_extent_type).unwrap().clone()),
                axis: BatchAxis::new(0),
            }),
        );

        let broadcast = ArrayIrOperation::<Array>::from(DynamicBroadcastOperation::new(vec![1]));
        let broadcast_input =
            ArrayIrBatch::new(ArrayIrValue::Array(Array::matrix(2, 1, vec![1.0_f32, 2.0])), BatchAxis::new(0)).unwrap();
        let broadcast_output = broadcast
            .batch(
                &context,
                &EmptyRegionDriver,
                &[
                    broadcast_input,
                    ArrayIrBatch::replicated(ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap())),
                    ArrayIrBatch::replicated(ArrayIrValue::Dimension(DimensionValue::constant(1).unwrap())),
                ],
            )
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(broadcast_output.len(), 1);
        assert_eq!(broadcast_output[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(
            broadcast_output[0].value(),
            &ArrayIrValue::Array(Array::from_f64s(
                ArrayType::new(
                    DataType::F32,
                    Shape::new(vec![Dimension::Static(2), Dimension::Static(3), Dimension::Static(1),]),
                ),
                vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
            )),
        );

        let mapped_broadcast_extent = ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap());
        let mapped_broadcast_extent_type = mapped_broadcast_extent.r#type().into_owned();
        assert_eq!(
            broadcast.batch(
                &context,
                &EmptyRegionDriver,
                &[
                    ArrayIrBatch::replicated(ArrayIrValue::Array(Array::vector(vec![1.0_f32]))),
                    ArrayIrBatch {
                        value: mapped_broadcast_extent,
                        batch_axis: BatchAxis::new(0),
                        mapped_dimension: None,
                        ragged_axes: Vec::new(),
                    },
                    ArrayIrBatch::replicated(ArrayIrValue::Dimension(DimensionValue::constant(1).unwrap(),)),
                ],
            ),
            Err(BatchingError::MappedDimension {
                r#type: Box::new(<&DimensionType>::try_from(&mapped_broadcast_extent_type).unwrap().clone()),
                axis: BatchAxis::new(0),
            }),
        );

        // A mapped padding value is decomposed into zero-padding, a padding-position mask, a broadcast of the
        // per-item scalar, and a select. Every shape-changing instruction in that decomposition receives the same
        // explicit output extents, including the inserted batch extent.
        let pad = ArrayIrOperation::<Array>::from(PadOperation::new(vec![1], vec![0], vec![0]).unwrap());
        let pad_output = pad
            .batch(
                &context,
                &EmptyRegionDriver,
                &[
                    ArrayIrBatch::new(
                        ArrayIrValue::Array(Array::matrix(2, 2, vec![1.0_f32, 2.0, 3.0, 4.0])),
                        BatchAxis::new(0),
                    )
                    .unwrap(),
                    ArrayIrBatch::new(ArrayIrValue::Array(Array::vector(vec![8.0_f32, 9.0])), BatchAxis::new(0))
                        .unwrap(),
                    ArrayIrBatch::replicated(ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap())),
                ],
            )
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(
            pad_output,
            vec![
                ArrayIrBatch::new(
                    ArrayIrValue::Array(Array::matrix(2, 3, vec![8.0_f32, 1.0, 2.0, 9.0, 3.0, 4.0],)),
                    BatchAxis::new(0),
                )
                .unwrap()
            ],
        );

        // Mapped RNG state batching is scan-based: each mapped state is advanced independently and the generated bits
        // retain the mapped axis as their leading axis.
        let states = Array::from_elements(
            ArrayType::new(DataType::U64, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)])),
            &[1u64, 0, 2, 0],
        )
        .unwrap();
        let state_batch = ArrayIrBatch::new(ArrayIrValue::Array(states.clone()), BatchAxis::new(0)).unwrap();
        let static_rng = ArrayIrOperation::<Array>::from(RngBitGeneratorOperation::new(
            RandomAlgorithm::ThreeFry,
            ArrayType::new(DataType::U32, Shape::new(vec![Dimension::Static(2)])),
        ));
        let static_outputs = static_rng
            .batch(&context, &EmptyRegionDriver, std::slice::from_ref(&state_batch))
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(static_outputs.len(), 2);
        assert_eq!(static_outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(static_outputs[1].batch_axis(), BatchAxis::new(0));
        assert_eq!(
            static_outputs[0].value().r#type().as_ref(),
            &ArrayIrType::Array(ArrayType::new(
                DataType::U64,
                Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]),
            )),
        );
        assert_eq!(
            static_outputs[1].value().r#type().as_ref(),
            &ArrayIrType::Array(ArrayType::new(
                DataType::U32,
                Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]),
            )),
        );

        let dynamic_rng_extent = DimensionVariable::new("rng_count", DimensionBounds::new(1, Some(5)).unwrap());
        let dynamic_rng = ArrayIrOperation::<Array>::from(RngBitGeneratorOperation::new(
            RandomAlgorithm::ThreeFry,
            ArrayType::new(DataType::U32, Shape::new(vec![Dimension::Dynamic(dynamic_rng_extent.clone())])),
        ));
        let dynamic_outputs = dynamic_rng
            .batch(
                &context,
                &EmptyRegionDriver,
                &[
                    state_batch,
                    ArrayIrBatch::replicated(ArrayIrValue::Dimension(
                        DimensionValue::new(DimensionType::new(dynamic_rng_extent), 2).unwrap(),
                    )),
                ],
            )
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(dynamic_outputs.len(), 2);
        assert_eq!(dynamic_outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(dynamic_outputs[1].batch_axis(), BatchAxis::new(0));
        assert_eq!(
            dynamic_outputs[1].value().r#type().as_ref(),
            &ArrayIrType::Array(ArrayType::new(
                DataType::U32,
                Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]),
            )),
        );
        assert_eq!(
            static_rng.batch(&context, &EmptyRegionDriver, &[ArrayIrBatch::replicated(ArrayIrValue::Array(states))],),
            Err(BatchingError::UnsupportedOperation {
                message: "'rng_bit_generator' cannot batch a replicated state because every batch item would see \
                          the same state; derive one state per batch item with `split_key` and map over the states \
                          explicitly"
                    .to_string(),
            }),
        );

        // Concatenate aligns mapped array operands before shifting the per-item concatenation axis around the common
        // packed batch axis. Its trailing extent remains a replicated shape value.
        let concatenate_extent = ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap());
        let concatenate = ArrayIrOperation::<Array>::from(
            ConcatenateOperation::<ArrayIrType>::from_input_types(
                0,
                &[
                    ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)])).into(),
                    ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(1)])).into(),
                    concatenate_extent.r#type().into_owned(),
                ],
            )
            .unwrap(),
        );
        let concatenate_output = concatenate
            .batch(
                &context,
                &EmptyRegionDriver,
                &[
                    ArrayIrBatch::new(
                        ArrayIrValue::Array(Array::matrix(2, 2, vec![1.0_f32, 3.0, 2.0, 4.0])),
                        BatchAxis::new(1),
                    )
                    .unwrap(),
                    ArrayIrBatch::new(ArrayIrValue::Array(Array::matrix(2, 1, vec![5.0_f32, 6.0])), BatchAxis::new(0))
                        .unwrap(),
                    ArrayIrBatch::replicated(concatenate_extent.clone()),
                ],
            )
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(
            concatenate_output,
            vec![
                ArrayIrBatch::new(
                    ArrayIrValue::Array(Array::matrix(3, 2, vec![1.0_f32, 3.0, 2.0, 4.0, 5.0, 6.0])),
                    BatchAxis::new(1),
                )
                .unwrap()
            ],
        );
        assert_eq!(
            concatenate
                .batch(
                    &context,
                    &EmptyRegionDriver,
                    &[
                        ArrayIrBatch::new(
                            ArrayIrValue::Array(Array::matrix(2, 2, vec![1.0_f32, 2.0, 3.0, 4.0])),
                            BatchAxis::new(0),
                        )
                        .unwrap(),
                        ArrayIrBatch::replicated(ArrayIrValue::Array(Array::vector(vec![5.0_f32]))),
                        ArrayIrBatch::replicated(concatenate_extent.clone()),
                    ],
                )
                .unwrap()
                .into_parts()
                .0,
            vec![
                ArrayIrBatch::new(
                    ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0_f32, 2.0, 5.0, 3.0, 4.0, 5.0],)),
                    BatchAxis::new(0),
                )
                .unwrap()
            ],
        );
        let concatenate_extent_type = concatenate_extent.r#type().into_owned();
        assert_eq!(
            concatenate.batch(
                &context,
                &EmptyRegionDriver,
                &[
                    ArrayIrBatch::replicated(ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0]))),
                    ArrayIrBatch::replicated(ArrayIrValue::Array(Array::vector(vec![3.0_f32]))),
                    ArrayIrBatch {
                        value: concatenate_extent,
                        batch_axis: BatchAxis::new(0),
                        mapped_dimension: None,
                        ragged_axes: Vec::new(),
                    },
                ],
            ),
            Err(BatchingError::MappedDimension {
                r#type: Box::new(<&DimensionType>::try_from(&concatenate_extent_type).unwrap().clone()),
                axis: BatchAxis::new(0),
            }),
        );

        let dimension = BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(dimension));
        let scalar = dimension.to_scalar().unwrap().into_batch();
        assert_eq!(scalar.batch_axis(), BatchAxis::replicated());
        assert_eq!(scalar.into_value(), ArrayIrValue::Array(Array::scalar(4_i64)));

        let scalar =
            BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(ArrayIrValue::Array(Array::scalar(4_i32))));
        let dimension = scalar.to_dimension(gateway_variable).unwrap().into_batch();
        assert_eq!(dimension.batch_axis(), BatchAxis::replicated());
        assert!(matches!(dimension.into_value(), ArrayIrValue::Dimension(value) if value.extent() == 4));

        let array = ArrayIrValue::Array(Array::matrix(2, 3, vec![0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0]));
        let array = ArrayIrBatch::new(array, BatchAxis::new(0)).unwrap();
        let array = BatchingTracer::new(context, array);
        let scalar = array.dimension_size(0).unwrap().to_scalar().unwrap().into_batch();
        assert_eq!(scalar.batch_axis(), BatchAxis::replicated());
        assert_eq!(scalar.into_value(), ArrayIrValue::Array(Array::scalar(3_i64)));

        type TraceContext = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;
        let trace = TraceContext::new();
        let input = trace.input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)])).into());
        let output = input.dimension_size(0).unwrap().to_scalar().unwrap();
        let program = trace
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output.atom_id().unwrap()],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let context = BatchingContext::<_, ArrayIrBatching>::new(
            Parent::new(),
            ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()),
        );
        let input = BatchingTracer::new(
            context.clone(),
            ArrayIrBatch::new(
                ArrayIrValue::Array(Array::matrix(2, 3, vec![0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0])),
                BatchAxis::new(0),
            )
            .unwrap(),
        );
        let outputs = program.interpret_in_context(&context, vec![input]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch().batch_axis(), BatchAxis::replicated());
        assert_eq!(outputs[0].batch().value(), &ArrayIrValue::Array(Array::scalar(3_i64)));
    }

    #[test]
    fn test_array_ir_ragged_batching() -> Result<(), ProgramError> {
        let variable = DimensionVariable::new("length", DimensionBounds::new(0, Some(4))?);

        // Each item broadcasts to its own checked length, the elementwise square preserves that ragged geometry, and
        // the reduction masks the padded suffix before summing it.
        let output: ArrayIrValue<Array> = batch(
            |(value, extent)| {
                let extent = extent.to_dimension(variable.clone())?;
                let repeated = value.dynamic_broadcast_to(&[extent])?;
                let extent = repeated.dimension_size(0)?;
                let repeated = value.dynamic_broadcast_to(&[extent])?;
                let repeated = ValueProjection::<ArrayType>::into_projected(repeated)?;
                let squared = repeated.clone() * repeated;
                let squared = squared.into_value();
                let mut squared = LinearCallOperation::stage(
                    squared.context(),
                    Vec::new(),
                    vec![squared.clone()],
                    |_, inputs| Ok(inputs.to_vec()),
                    |_, cotangents| Ok(cotangents.to_vec()),
                )?;
                let squared = ValueProjection::<ArrayType>::into_projected(squared.remove(0))?;
                Ok(squared.reduce(&[0], ReductionKind::Sum).into_value())
            },
            (
                ArrayIrValue::Array(Array::vector(vec![2.0_f32, 3.0])),
                ArrayIrValue::Array(Array::vector(vec![1_i32, 3])),
            ),
            (BatchAxis::new(0), BatchAxis::new(0)),
            BatchAxis::new(0),
            None,
        )?;
        assert_eq!(output, ArrayIrValue::Array(Array::vector(vec![4.0_f32, 27.0])));

        // Converting the mapped dimension back to scalar data exposes the checked per-item extent vector directly.
        let extents: ArrayIrValue<Array> = batch(
            |extent| Ok(extent.to_dimension(variable.clone())?.to_scalar()?),
            ArrayIrValue::Array(Array::vector(vec![1_i32, 3])),
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        )?;
        assert_eq!(extents, ArrayIrValue::Array(Array::vector(vec![1_i64, 3])));

        // The same carrier path remains symbolic when the mapped batch extent is read from a dynamic input axis.
        type TraceContext = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;
        let trace = TraceContext::new();
        let batch_variable = DimensionVariable::new("batch", DimensionBounds::new(1, Some(5))?);
        let values = trace
            .input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(batch_variable.clone())])).into());
        let extents =
            trace.input(ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Dynamic(batch_variable)])).into());
        let output = batch(
            |(value, extent)| {
                let extent = extent.to_dimension(variable.clone())?;
                let repeated = value.dynamic_broadcast_to(&[extent])?;
                let extent = repeated.dimension_size(0)?;
                let repeated = value.dynamic_broadcast_to(&[extent])?;
                let repeated = ValueProjection::<ArrayType>::into_projected(repeated)?;
                let squared = repeated.clone() * repeated;
                let squared = squared.into_value();
                let mut squared = LinearCallOperation::stage(
                    squared.context(),
                    Vec::new(),
                    vec![squared.clone()],
                    |_, inputs| Ok(inputs.to_vec()),
                    |_, cotangents| Ok(cotangents.to_vec()),
                )?;
                let squared = ValueProjection::<ArrayType>::into_projected(squared.remove(0))?;
                Ok(squared.reduce(&[0], ReductionKind::Sum).into_value())
            },
            (values, extents),
            (BatchAxis::new(0), BatchAxis::new(0)),
            BatchAxis::new(0),
            None,
        )?;
        let program = trace.builder().borrow().clone().build::<Vec<ArrayIrValue<Array>>, ArrayIrValue<Array>>(
            vec![output.atom_id()?],
            vec![Placeholder, Placeholder],
            Placeholder,
        )?;
        assert_eq!(
            program.interpret(vec![
                ArrayIrValue::Array(Array::vector(vec![2.0_f32, 3.0])),
                ArrayIrValue::Array(Array::vector(vec![1_i32, 3])),
            ])?,
            ArrayIrValue::Array(Array::vector(vec![4.0_f32, 27.0])),
        );

        // Applying structural batching twice realizes nested `vmap` through the existing recursive policy stack. The
        // inner transform owns and masks the ragged intermediate; the outer pass then batches that complete program
        // without introducing another context, tracer, or type universe.
        let trace = TraceContext::new();
        let value = trace.input(ArrayType::scalar(DataType::F32).into());
        let extent = trace.input(ArrayType::scalar(DataType::I32).into()).to_dimension(variable)?;
        let repeated = value.dynamic_broadcast_to(&[extent])?;
        let repeated = ValueProjection::<ArrayType>::into_projected(repeated)?;
        let squared = repeated.clone() * repeated;
        let output = squared.reduce(&[0], ReductionKind::Sum).into_value();
        let program = trace.builder().borrow().clone().build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            vec![output.atom_id()?],
            vec![Placeholder, Placeholder],
            vec![Placeholder],
        )?;
        type Parent = EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;
        let inner_context = BatchingContext::<_, ArrayIrBatching>::new(
            Parent::new(),
            ArrayIrValue::Dimension(DimensionValue::constant(2)?),
        );
        let inner = <ArrayIrBatching as RecursiveBatchingPolicy<Parent>>::batch_program(
            &inner_context,
            program.entry_region_ref(),
            &[BatchAxis::new(0), BatchAxis::new(0)],
            ProgramBatchingOutputAxesPolicy::Natural,
        )?;
        let (inner, _) = inner.into_parts();
        let outer_context = BatchingContext::<_, ArrayIrBatching>::new(
            Parent::new(),
            ArrayIrValue::Dimension(DimensionValue::constant(2)?),
        );
        let outer = <ArrayIrBatching as RecursiveBatchingPolicy<Parent>>::batch_program(
            &outer_context,
            inner.entry_region_ref(),
            &[BatchAxis::replicated(), BatchAxis::new(0), BatchAxis::new(0)],
            ProgramBatchingOutputAxesPolicy::Natural,
        )?;
        let (outer, _) = outer.into_parts();
        let outputs = outer.interpret(vec![
            ArrayIrValue::Dimension(DimensionValue::constant(2)?),
            ArrayIrValue::Dimension(DimensionValue::constant(2)?),
            ArrayIrValue::Array(Array::matrix(2, 2, vec![2.0_f32, 3.0, 4.0, 5.0])),
            ArrayIrValue::Array(Array::matrix(2, 2, vec![1_i32, 3, 2, 1])),
        ])?;
        assert_eq!(outputs.len(), 3);
        assert!(matches!(&outputs[0], ArrayIrValue::Dimension(value) if value.extent() == 2));
        assert!(matches!(&outputs[1], ArrayIrValue::Dimension(value) if value.extent() == 2));
        assert_eq!(outputs[2], ArrayIrValue::Array(Array::matrix(2, 2, vec![4.0_f32, 27.0, 32.0, 25.0])));

        Ok(())
    }

    #[test]
    fn test_array_ir_ragged_contraction_batching() -> Result<(), ProgramError> {
        let variable = DimensionVariable::new("length", DimensionBounds::new(0, Some(4))?);

        // Each item broadcasts to its own checked length and then contracts that ragged vector with itself. Zeroing
        // the padded elements of the contraction's operands removes their products from its sums, so each item's
        // result is the inner product over its live prefix, computed through the contraction discipline rather than
        // through a square followed by a masked reduction.
        let output: ArrayIrValue<Array> = batch(
            |(value, extent)| {
                let extent = extent.to_dimension(variable.clone())?;
                let repeated = value.dynamic_broadcast_to(&[extent])?;
                let repeated = ValueProjection::<ArrayType>::into_projected(repeated)?;
                let mut outputs = repeated.dispatch_domain().bind(
                    DotOperation::new(DotDimensionNumbers::inner_product()),
                    Vec::new(),
                    &[repeated.clone(), repeated],
                )?;
                Ok(outputs.remove(0).into_value())
            },
            (
                ArrayIrValue::Array(Array::vector(vec![2.0_f32, 3.0])),
                ArrayIrValue::Array(Array::vector(vec![1_i32, 3])),
            ),
            (BatchAxis::new(0), BatchAxis::new(0)),
            BatchAxis::new(0),
            None,
        )?;
        assert_eq!(output, ArrayIrValue::Array(Array::vector(vec![4.0_f32, 27.0])));

        // The same composition remains symbolic when the mapped batch extent itself is read from a dynamic input axis.
        // The contraction aligns its operands through the policy rather than through a host batch size, so the staged
        // dot simply carries the dynamic mapped dimension on its batching dimension.
        type TraceContext = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;
        let trace = TraceContext::new();
        let batch_variable = DimensionVariable::new("batch", DimensionBounds::new(1, Some(5))?);
        let values = trace
            .input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(batch_variable.clone())])).into());
        let extents =
            trace.input(ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Dynamic(batch_variable)])).into());
        let output = batch(
            |(value, extent)| {
                let extent = extent.to_dimension(variable.clone())?;
                let repeated = value.dynamic_broadcast_to(&[extent])?;
                let repeated = ValueProjection::<ArrayType>::into_projected(repeated)?;
                let mut outputs = repeated.dispatch_domain().bind(
                    DotOperation::new(DotDimensionNumbers::inner_product()),
                    Vec::new(),
                    &[repeated.clone(), repeated],
                )?;
                Ok(outputs.remove(0).into_value())
            },
            (values, extents),
            (BatchAxis::new(0), BatchAxis::new(0)),
            BatchAxis::new(0),
            None,
        )?;
        let program = trace.builder().borrow().clone().build::<Vec<ArrayIrValue<Array>>, ArrayIrValue<Array>>(
            vec![output.atom_id()?],
            vec![Placeholder, Placeholder],
            Placeholder,
        )?;
        assert_eq!(
            program.interpret(vec![
                ArrayIrValue::Array(Array::vector(vec![2.0_f32, 3.0])),
                ArrayIrValue::Array(Array::vector(vec![1_i32, 3])),
            ])?,
            ArrayIrValue::Array(Array::vector(vec![4.0_f32, 27.0])),
        );

        Ok(())
    }

    // TODO(eaplatanios): Move (?) to `ryft_core::arrays::operations::...` where `dot` for arrays is implemented.
    #[test]
    fn test_array_ir_dense_dot_batching_under_a_dynamic_mapped_extent() -> Result<(), ProgramError> {
        // A plain dense contraction is batched under a dynamic mapped extent as well. The mapped operand keeps the
        // dynamic batch dimension it already carries, and the replicated right-hand vector gains one through the
        // policy's dynamic broadcast, whose inserted axis is grounded by the transform's first-class extent value.
        type TraceContext = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;
        let trace = TraceContext::new();
        let batch_variable = DimensionVariable::new("batch", DimensionBounds::new(1, Some(5))?);
        let rows = trace.input(
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(batch_variable), Dimension::Static(3)]))
                .into(),
        );
        let vector = trace.input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)])).into());
        let output = batch(
            |(row, vector)| {
                let row = ValueProjection::<ArrayType>::into_projected(row)?;
                let vector = ValueProjection::<ArrayType>::into_projected(vector)?;
                let mut outputs = row.dispatch_domain().bind(
                    DotOperation::new(DotDimensionNumbers::inner_product()),
                    Vec::new(),
                    &[row, vector],
                )?;
                Ok(outputs.remove(0).into_value())
            },
            (rows, vector),
            (BatchAxis::new(0), BatchAxis::replicated()),
            BatchAxis::new(0),
            None,
        )?;
        let program = trace.builder().borrow().clone().build::<Vec<ArrayIrValue<Array>>, ArrayIrValue<Array>>(
            vec![output.atom_id()?],
            vec![Placeholder, Placeholder],
            Placeholder,
        )?;
        // The replicated operand is materialized by a dynamic broadcast against the mapped extent read off the mapped
        // operand's own axis, and the lifted dot then contracts the trailing axis under a dynamic batching dimension.
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[batch, 3], %1:f32[3] .
                let %2:dimension<batch ∈ [1, 5)> = dimension_size [axis=0] %0
                    %3:dimension<3> = constant [value=3]
                    %4:f32[batch, 3] = broadcast [output_axes=[1]] %1 %2 %3
                    %5:f32[batch] = dot [
                        dimensions=(lhs_contracting=[1], rhs_contracting=[1], lhs_batching=[0], rhs_batching=[0]),
                    ] %0 %4
                in (%5)
            "}
            .trim_end(),
        );
        assert_eq!(
            program.interpret(vec![
                ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])),
                ArrayIrValue::Array(Array::vector(vec![10.0_f32, 100.0, 1000.0])),
            ])?,
            ArrayIrValue::Array(Array::vector(vec![3210.0_f32, 6540.0])),
        );

        Ok(())
    }

    #[test]
    fn test_array_ir_ragged_batching_rejects_unsupported_consumers() -> Result<(), ProgramError> {
        let variable = DimensionVariable::new("length", DimensionBounds::new(0, Some(4))?);

        // Sum owns a padding identity and is supported above; reduction kinds without a defined ragged masking rule
        // fail explicitly instead of observing padded storage.
        let reduction: Result<ArrayIrValue<Array>, BatchingError> = batch(
            |(value, extent)| {
                let extent = extent.to_dimension(variable.clone())?;
                let repeated = value.dynamic_broadcast_to(&[extent])?;
                let repeated = ValueProjection::<ArrayType>::into_projected(repeated)?;
                let mut outputs = repeated.dispatch_domain().bind(
                    ReduceOperation::new(vec![0], ReductionKind::Max),
                    Vec::new(),
                    std::slice::from_ref(&repeated),
                )?;
                Ok(outputs.remove(0).into_value())
            },
            (
                ArrayIrValue::Array(Array::vector(vec![2.0_f32, 3.0])),
                ArrayIrValue::Array(Array::vector(vec![1_i32, 3])),
            ),
            (BatchAxis::new(0), BatchAxis::new(0)),
            BatchAxis::new(0),
            None,
        );
        assert_eq!(
            reduction,
            Err(BatchingError::UnsupportedOperation {
                message: "ragged reduction kind max is not supported; use reduce_sum".to_string(),
            }),
        );

        // Projected member rules that have no ragged contract fail at the shared projection boundary if they discard
        // the carrier metadata. A static slice of the bounded storage must not silently expose padding as live data.
        let slice: Result<ArrayIrValue<Array>, BatchingError> = batch(
            |(value, extent)| {
                let extent = extent.to_dimension(variable.clone())?;
                let repeated = value.dynamic_broadcast_to(&[extent])?;
                let repeated = ValueProjection::<ArrayType>::into_projected(repeated)?;
                Ok(repeated.slice(&[0], &[0], &[1])?.into_value())
            },
            (
                ArrayIrValue::Array(Array::vector(vec![2.0_f32, 3.0])),
                ArrayIrValue::Array(Array::vector(vec![1_i32, 3])),
            ),
            (BatchAxis::new(0), BatchAxis::new(0)),
            BatchAxis::new(0),
            None,
        );
        assert_eq!(
            slice,
            Err(BatchingError::UnsupportedOperation {
                message: "operation `slice` neither preserves nor consumes bounded ragged dimension `length`"
                    .to_string(),
            }),
        );

        // Mixed rules are subject to the same fail-closed carrier validation. Concatenating along the static axis
        // cannot silently turn the unrelated ragged axis's padded bound into a logical extent.
        let concatenation: Result<ArrayIrValue<Array>, BatchingError> = batch(
            |(value, extent)| {
                let extent = extent.to_dimension(variable.clone())?;
                let width = value
                    .context()
                    .lift(ArrayIrValue::Dimension(DimensionValue::constant(2).map_err(ProgramError::from)?))?;
                let repeated = value.dynamic_broadcast_to(&[extent, width])?;
                let result_extent = value
                    .context()
                    .lift(ArrayIrValue::Dimension(DimensionValue::constant(4).map_err(ProgramError::from)?))?;
                let operation = ConcatenateOperation::<ArrayIrType>::from_input_types(
                    1,
                    &[
                        repeated.r#type().into_owned(),
                        repeated.r#type().into_owned(),
                        result_extent.r#type().into_owned(),
                    ],
                )?;
                let mut outputs = repeated.context().bind(
                    operation,
                    Vec::new(),
                    &[repeated.clone(), repeated.clone(), result_extent],
                )?;
                Ok(outputs.remove(0))
            },
            (
                ArrayIrValue::Array(Array::matrix(2, 2, vec![2.0_f32, 3.0, 4.0, 5.0])),
                ArrayIrValue::Array(Array::vector(vec![1_i32, 3])),
            ),
            (BatchAxis::new(0), BatchAxis::new(0)),
            BatchAxis::new(0),
            None,
        );
        assert_eq!(
            concatenation,
            Err(BatchingError::UnsupportedOperation {
                message: "operation `concatenate` neither preserves nor consumes bounded ragged dimension `length`"
                    .to_string(),
            }),
        );

        // A ragged array must be consumed by a supported operation before leaving the transform. Returning its dense
        // packed storage would otherwise expose the declared bound as if every padded element were live.
        let output_boundary: Result<ArrayIrValue<Array>, BatchingError> = batch(
            |(value, extent)| {
                let extent = extent.to_dimension(variable.clone())?;
                value.dynamic_broadcast_to(&[extent])
            },
            (
                ArrayIrValue::Array(Array::vector(vec![2.0_f32, 3.0])),
                ArrayIrValue::Array(Array::vector(vec![1_i32, 3])),
            ),
            (BatchAxis::new(0), BatchAxis::new(0)),
            BatchAxis::new(0),
            None,
        );
        assert_eq!(
            output_boundary,
            Err(BatchingError::UnsupportedOperation {
                message: "a bounded ragged array cannot cross the batching transform output boundary".to_string(),
            }),
        );

        // Each mapped gateway item executes the same checked conversion as an unbatched scalar; one invalid item
        // fails the complete transform rather than entering the extent carrier.
        let gateway: Result<ArrayIrValue<Array>, BatchingError> = batch(
            |extent| Ok(extent.to_dimension(variable.clone())?.to_scalar()?),
            ArrayIrValue::Array(Array::vector(vec![1_i32, 4])),
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        );
        assert_eq!(
            gateway.unwrap_err().to_string(),
            "input dimension `length` = 4 is outside its declared bounds [0, 4)",
        );

        // A mapped extent carrier may flow between rules inside the transform, but a first-class dimension leaving it
        // must be replicated. There is no packed axis to align one per-item extent along, so the public boundary
        // reports the typed mapped-dimension diagnostic instead of materializing the extents as ordinary data.
        let mapped_dimension_output: Result<ArrayIrValue<Array>, BatchingError> = batch(
            |extent| extent.to_dimension(variable.clone()),
            ArrayIrValue::Array(Array::vector(vec![1_i32, 3])),
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        );
        assert_eq!(
            mapped_dimension_output,
            Err(BatchingError::MappedDimension {
                r#type: Box::new(DimensionType::new(variable.clone())),
                axis: BatchAxis::new(0),
            }),
        );

        // An opaque region may only recover a physically bounded logical axis from an input that carries that exact
        // per-item extent. A fresh logical identity therefore fails at the region boundary.
        type Parent = EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;
        let physical = ArrayIrValue::Array(Array::matrix(2, 3, vec![0.0_f32; 6]));
        let logical_type =
            ArrayIrType::Array(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(variable.clone())])));
        assert_eq!(
            <ArrayIrBatching as RecursiveBatchingPolicy<Parent>>::restore_batch(
                physical,
                BatchAxis::new(0),
                &logical_type,
                &[],
            ),
            Err(BatchingError::UnsupportedOperation {
                message: "linear call output f32[length] has bounded dynamic dimension length but no input carries \
                          its per-item extents"
                    .to_string(),
            }),
        );

        // Mixed shape operations reject the same unsupported carrier before inspecting their shape operands.
        let reshape: Result<ArrayIrValue<Array>, BatchingError> = batch(
            |(value, extent)| {
                let extent = extent.to_dimension(variable.clone())?;
                let repeated = value.dynamic_broadcast_to(std::slice::from_ref(&extent))?;
                let mut outputs =
                    repeated.context().bind(DynamicReshapeOperation::new(), Vec::new(), &[repeated.clone(), extent])?;
                Ok(outputs.remove(0))
            },
            (
                ArrayIrValue::Array(Array::vector(vec![2.0_f32, 3.0])),
                ArrayIrValue::Array(Array::vector(vec![1_i32, 3])),
            ),
            (BatchAxis::new(0), BatchAxis::new(0)),
            BatchAxis::new(0),
            None,
        );
        assert_eq!(
            reshape,
            Err(BatchingError::UnsupportedOperation {
                message: "dynamic reshape does not support bounded ragged array operands".to_string(),
            }),
        );

        // A mapped gateway result cannot become loop-carried trip-count state. Doing so would require independently
        // masked dimension values, which remain outside the supported ragged carrier surface.
        let dimension_type = DimensionType::new(variable.clone());
        let mut condition_builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        condition_builder.add_input(dimension_type.clone().into());
        let predicate = condition_builder.add_constant(ArrayIrValue::Array(Array::scalar(false)));
        let condition = condition_builder.build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            vec![predicate],
            vec![Placeholder],
            vec![Placeholder],
        )?;
        let mut body_builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let state = body_builder.add_input(dimension_type.clone().into());
        let body = body_builder.build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            vec![state],
            vec![Placeholder],
            vec![Placeholder],
        )?;
        let control_flow: Result<ArrayIrValue<Array>, BatchingError> = batch(
            |extent| {
                let extent = extent.to_dimension(variable.clone())?;
                let mut outputs = extent.context().bind(
                    WhileOperation::new(),
                    vec![condition.clone(), body.clone()],
                    std::slice::from_ref(&extent),
                )?;
                Ok(outputs.remove(0))
            },
            ArrayIrValue::Array(Array::vector(vec![1_i32, 3])),
            BatchAxis::new(0),
            BatchAxis::replicated(),
            None,
        );
        assert_eq!(
            control_flow,
            Err(BatchingError::MappedDimension { r#type: Box::new(dimension_type), axis: BatchAxis::new(0) }),
        );

        Ok(())
    }
}
