//! [`ArrayType`]d batch carriers, policies, and mapped-axis mechanics.
//!
//! This module owns the array-specific representation and policy used by the batching transform. Universe-neutral
//! batching contracts, contexts, drivers, and transform entry points remain in the parent
//! [`batching`](crate::batching) module.

use std::borrow::Cow;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use std::rc::Rc;

use ryft_macros::Parameter;

use crate::axes::Axis;
use crate::batching::{
    BatchAxis, BatchAxisSpecification, BatchableOperation, BatchableType, BatchedProgram, BatchingContext,
    BatchingDriver, BatchingEntrypointPolicy, BatchingError, BatchingPolicy, BoundaryPreservingBatchedProgram,
    InterpretableBatchableOperation, ProgramBatchingOutputAxesPolicy, RecursiveBatchingDriver, RecursiveBatchingPolicy,
};
use crate::broadcasting::Broadcastable;
use crate::contexts::{Context, EagerContext, StagingContext};
use crate::interpretation::InterpretableOperation;
use crate::macros::{check_builders, check_count};
use crate::operations::ElementwiseOperation;
use crate::operations::manipulation::{LegacyBroadcast, LegacyBroadcastOperation, Transpose, TransposeOperation};
use crate::parameters::{Parameter, Placeholder};
use crate::programs::ProgramError;
use crate::programs::operations::Operation;
use crate::programs::programs::Program;
use crate::programs::regions::{RegionRef, RegionReplayMappings, ReplayRegionDriver};
use crate::programs::types::{TypeError, Typed};
use crate::programs::values::Value;
use crate::sharding::{MeshAxisType, Sharding, ShardingDimension, ShardingError};
use crate::tracing::{Tracer, TracingContext};
use crate::types::{ArrayType, Dimension, Shape};

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
    /// are expected to dispatch the replicated case explicitly (the blanket elementwise
    /// [`BatchableOperation`](super::BatchableOperation)
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
    /// rank. [`Batch::batch`](super::Batch::batch) uses this function to realize the caller's declared
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
/// binding lets every rule rely on that without restating it. The shared rules are then written once against the
/// nominal [`ArrayBatching<P>`] family rather than as a `P: ArrayBatchingPolicy` blanket, because Rust coherence cannot
/// use the *absence* of a trait implementation to prove such a blanket disjoint from the genuinely mixed composite
/// operation rules registered for other policies.
///
/// Keeping this capability on the batching transform, rather than on [`ProjectedContext`], [`ArrayBatch`], or [`Type`],
/// means that neither the carrier nor the type contract needs to know anything about dynamic-shape state that only
/// batching needs.
pub trait ArrayBatchingPolicy<C: Context<Type = ArrayType>>:
    BatchingPolicy<
        C,
        Batch = ArrayBatch<C::Value>,
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
    type BatchedProgram = BoundaryPreservingBatchedProgram<C::Constant, C::Operation>;

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
    ) -> Result<BoundaryPreservingBatchedProgram<C::Constant, C::Operation>, BatchingError> {
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
                        normalized_batch_axis_type(batch.r#type().as_ref(), position, batching_context.axis_sharding())
                    })
                    .transpose()?
                    .flatten();
                let batch = if let Some(r#type) = normalized_type {
                    // A rank-preserving broadcast with identity output axes changes only the requested packed
                    // sharding placement. Rewrapping the result retains the original batch-axis declaration.
                    let output_axes = (0..batch.r#type().rank()).collect::<Vec<_>>();
                    let value = batch.value().clone().legacy_broadcast(r#type.clone(), output_axes.as_slice())?;
                    ArrayBatch::new(r#type, value, batch.batch_axis())?
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
    /// [`ProgramBatchingOutputAxesPolicy::AlignAllTo`] imposes one canonical
    /// output axis when a consumer explicitly requires one common layout.
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
        Ok(BoundaryPreservingBatchedProgram::new(program, output_axes)?)
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
    ) -> Result<BoundaryPreservingBatchedProgram<V, O>, BatchingError> {
        self.entry_region_ref().batched(axis_size, axis_sharding, input_batch_axes, output_axes_policy)
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use approx::assert_abs_diff_eq;
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::backends::arrays::{Array, ArrayOperation};
    use crate::batching::{Batch, BatchingTracer, batch};
    use crate::contexts::EagerContext;
    use crate::differentiation::forward::{ForwardModeDifferentiate, LinearizationTracer};
    use crate::differentiation::reverse::ReverseModeDifferentiate;
    use crate::operations::constants::OneLike;
    use crate::operations::math::{AddOperation, NegOperation, Reduce, ReductionKind};
    use crate::parameters::Placeholder;
    use crate::programs::regions::EmptyRegionDriver;
    use crate::sharding::{LogicalMesh, MeshAxis};
    use crate::tracing::{DomainTracingContext, Trace};
    use crate::types::{DataType, DimensionBounds, DimensionVariable};

    use super::*;

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
