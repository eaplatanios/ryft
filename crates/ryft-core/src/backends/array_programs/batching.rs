//! Batching support for programs that mix arrays with first-class dimensions.
//!
//! Arrays retain the ordinary [`ArrayBatch`] representation and existing array batching rules. First-class
//! dimensions are shared shape values and therefore remain replicated across the logical batch. Mixed operations
//! explicitly state how they cross that boundary.

use std::borrow::Cow;
use std::fmt::Display;

use ryft_macros::Parameter;

use crate::axes::{Axis, NamedAxes, NamedAxis};
use crate::backends::array_programs::ArrayProgramOperation;
use crate::backends::arrays::ArrayOperation;
use crate::backends::dimensions::{DimensionOperation, DimensionValue};
use crate::batching::{
    ArrayBatch, ArrayBatching, ArrayBatchingPolicy, BatchAxis, BatchAxisSpecification, BatchableOperation,
    BatchableType, BatchedProgram, BatchingContext, BatchingDriver, BatchingEntrypointPolicy, BatchingError,
    BatchingPolicy, BatchingTracer, DimensionSource, ProgramBatchingOutputAxesPolicy, RecursiveBatchingDriver,
    RecursiveBatchingPolicy, batch_axis_sharding, normalized_batch_axis_type,
};
use crate::contexts::{Context, ProjectedContext, ValueResolution};
use crate::macros::check_count;
use crate::operations::constants::{OneOperation, Zero, ZeroOperation};
use crate::operations::control_flow::SelectOperation;
use crate::operations::custom_call::CustomCallOperation;
use crate::operations::dimensions::{
    DimensionRequirementOperation, DimensionSizeOperation, DimensionToScalarOperation,
};
use crate::operations::manipulation::reshaping::lift_output_sharding_for_leading_batch_axis;
use crate::operations::manipulation::{
    BroadcastOperation, CONCATENATE_OPERATION_NAME, ConcatenateOperation, LegacyBroadcast, PadOperation, Reshape,
    ReshapeOperation, Transpose,
};
use crate::operations::math::Reduce;
use crate::operations::random::RngBitGeneratorOperation;
use crate::parameters::Parameter;
use crate::programs::ProgramError;
use crate::programs::operations::{Operation, OperationProjection};
use crate::programs::regions::{EmptyRegionDriver, RegionRef, RegionReplayMappings, ReplayRegionDriver};
use crate::programs::types::{Type, TypeError, Typed};
use crate::programs::values::{ProjectedValue, ProjectedValueRef, Value, ValueProjection};
use crate::sharding::Sharding;
use crate::types::{ArrayProgramType, ArrayType, DataType, Dimension, DimensionType};

/// Kind-aware batched view of one composite array-program value.
#[derive(Clone, Debug, Parameter)]
pub struct ArrayProgramBatch<V: Value<Type = ArrayProgramType>> {
    /// Physical parent value.
    value: V,

    /// Mapped physical array axis, or replicated for array and dimension values shared across the batch.
    batch_axis: BatchAxis,

    /// Logical per-item type reported to the transformed program.
    r#type: ArrayProgramType,
}

impl<V: Value<Type = ArrayProgramType>> ArrayProgramBatch<V> {
    /// Creates a batch view and rejects mapped first-class dimensions.
    pub fn new(value: V, batch_axis: BatchAxis) -> Result<Self, BatchingError> {
        let (r#type, batch_axis) = match value.r#type().as_ref() {
            ArrayProgramType::Array(r#type) => {
                let (r#type, batch_axis) = r#type.unbatched_type_and_axis(batch_axis)?;
                (ArrayProgramType::Array(r#type), batch_axis)
            }
            ArrayProgramType::Dimension(r#type) if batch_axis.is_replicated() => {
                (ArrayProgramType::Dimension(r#type.clone()), batch_axis)
            }
            ArrayProgramType::Dimension(r#type) => {
                return Err(BatchingError::MappedDimension { r#type: Box::new(r#type.clone()), axis: batch_axis });
            }
        };
        Ok(Self { value, batch_axis, r#type })
    }

    /// Creates a replicated batch view.
    #[inline]
    pub fn replicated(value: V) -> Self {
        let r#type = value.r#type().into_owned();
        Self { value, batch_axis: BatchAxis::replicated(), r#type }
    }

    /// Returns the physical parent value.
    #[inline]
    pub fn value(&self) -> &V {
        &self.value
    }

    /// Consumes this batch and returns its physical parent value.
    #[inline]
    pub fn into_value(self) -> V {
        self.value
    }

    /// Returns the mapped physical array axis, or replicated.
    #[inline]
    pub fn batch_axis(&self) -> BatchAxis {
        self.batch_axis
    }

    /// Returns the canonical nonnegative mapped-axis position for an array member, or `None` for a replicated member.
    fn batch_axis_position(&self) -> Option<usize> {
        let value_type = self.value.r#type();
        let r#type = <&ArrayType>::try_from(value_type.as_ref()).ok()?;
        self.batch_axis.axis().map(|axis| axis.normalize(r#type.rank()).unwrap())
    }

    /// Returns the logical per-item composite type.
    pub fn unbatched_type(&self) -> &ArrayProgramType {
        &self.r#type
    }

    /// Validates that this batch contains a replicated first-class dimension.
    fn validate_replicated_dimension(&self) -> Result<(), BatchingError> {
        let r#type = <&DimensionType>::try_from(&self.r#type)?;
        if self.batch_axis.is_replicated() {
            Ok(())
        } else {
            Err(BatchingError::MappedDimension { r#type: Box::new(r#type.clone()), axis: self.batch_axis })
        }
    }
}

impl<V: Value<Type = ArrayProgramType> + PartialEq> PartialEq for ArrayProgramBatch<V> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value && self.batch_axis == other.batch_axis
    }
}

impl<V: Value<Type = ArrayProgramType>> Typed for ArrayProgramBatch<V> {
    type Type = ArrayProgramType;

    #[inline]
    fn r#type(&self) -> Cow<'_, ArrayProgramType> {
        Cow::Borrowed(self.unbatched_type())
    }
}

impl<V: Value<Type = ArrayProgramType>> Display for ArrayProgramBatch<V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "batch[{}, {}]({})", self.r#type(), self.batch_axis, self.value)
    }
}

/// Batching policy for programs whose values may be arrays or first-class dimensions.
///
/// Array members may carry a mapped physical axis. Dimension members are shared shape values and therefore remain
/// replicated. The mapped-axis extent is itself an ordinary parent-owned dimension value, so dynamic extents remain
/// SSA data rather than transform metadata.
#[derive(Copy, Clone, Debug, Default)]
pub struct ArrayProgramBatching;

impl BatchableType for ArrayProgramType {
    type Policy = ArrayProgramBatching;
}

impl<C: Context<Type = ArrayProgramType>> BatchingPolicy<C> for ArrayProgramBatching {
    type Batch = ArrayProgramBatch<C::Value>;
    type Extent = C::Value;

    #[inline]
    fn batch(value: C::Value, batch_axis: BatchAxis) -> Result<Self::Batch, BatchingError> {
        ArrayProgramBatch::new(value, batch_axis)
    }

    #[inline]
    fn replicated(value: C::Value) -> Self::Batch {
        ArrayProgramBatch::replicated(value)
    }

    #[inline]
    fn value(batch: &Self::Batch) -> &C::Value {
        batch.value()
    }

    #[inline]
    fn unbatched_type(batch: &Self::Batch) -> Cow<'_, C::Type> {
        batch.r#type()
    }
}

/// [`ArrayBatchingPolicy`] used while a homogeneous array rule runs inside an array-program batching transform.
///
/// When composite batching reaches an array-member operation, it projects the operation and its batches into the
/// zero-state [`ProjectedContext`] over [`ArrayType`] and reuses the homogeneous rule unchanged: batches remain
/// ordinary [`ArrayBatch`]es, so the rule cannot tell it is running inside a composite program. What does change is
/// extent representation — the mapped-axis extent is the outer composite context's first-class dimension value rather
/// than a static host `usize`, so a dynamic batch extent stays an ordinary SSA operand edge.
///
/// This [`ArrayBatchingPolicy`] implementation is correspondingly the only place that translates a homogeneous rule's
/// extent and move-or-broadcast requests into mixed array-program operations: static per-item dimensions become exact
/// dimension constants, dynamic per-item dimensions become `dimension_size` reads of their broadcast-compatible
/// source axes, and the mapped axis itself is grounded by the extent value. [`ArrayBatchingPolicy::axis_size`]
/// succeeds only when the extent value's type proves one exact extent, so rules that genuinely enumerate batch items
/// fail with a precise error at dynamic extents instead of silently specializing them.
#[derive(Copy, Clone, Debug, Default)]
pub struct DynamicArrayBatchingPolicy;

impl<C: Context<Type = ArrayProgramType>> BatchingPolicy<ProjectedContext<C, ArrayType>> for DynamicArrayBatchingPolicy
where
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Value: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Operation: OperationProjection<ArrayType>,
{
    type Batch = ArrayBatch<<C::Value as ValueProjection<ArrayType>>::Projected>;
    type Extent = C::Value;

    #[inline]
    fn batch(
        value: <C::Value as ValueProjection<ArrayType>>::Projected,
        batch_axis: BatchAxis,
    ) -> Result<Self::Batch, BatchingError> {
        ArrayBatch::new(value.r#type().into_owned(), value, batch_axis)
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
    fn unbatched_type(batch: &Self::Batch) -> Cow<'_, ArrayType> {
        Cow::Owned(batch.unbatched_type())
    }
}

impl<C, T> ValueProjection<T> for BatchingTracer<C, ArrayProgramBatching>
where
    C: Context<Type = ArrayProgramType, Operation: BatchableOperation<C, ArrayProgramBatching>>,
    T: Type,
    for<'t> &'t T: TryFrom<&'t ArrayProgramType, Error = TypeError>,
{
    type Projected = ProjectedValue<T, Self>;
    type ProjectedRef<'v>
        = ProjectedValueRef<'v, T, Self>
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
        Ok(ProjectedValueRef::new(self, <&T>::try_from(self.batch().unbatched_type())?))
    }

    #[inline]
    fn into_projected(self) -> Result<Self::Projected, TypeError> {
        let r#type = <&T>::try_from(self.batch().unbatched_type())?.clone();
        Ok(ProjectedValue::new(self, r#type))
    }
}

/// Reads one physical array axis as a first-class dimension value in `context`.
fn array_dimension<C: Context<Type = ArrayProgramType, Operation: From<DimensionSizeOperation>>>(
    context: &C,
    value: &C::Value,
    axis: usize,
) -> Result<C::Value, BatchingError> {
    let value_type = value.r#type();
    let array_type = <&ArrayType>::try_from(value_type.as_ref())?;
    let operation = DimensionSizeOperation::new(array_type, axis)?;
    Ok(context.bind(operation, Vec::new(), std::slice::from_ref(value))?.remove(0))
}

/// Requires two composite dimension values to describe the same mapped extent.
fn require_equal_dimensions<C>(context: &C, left: &C::Value, right: &C::Value) -> Result<(), BatchingError>
where
    C: Context<Type = ArrayProgramType>,
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

/// Returns one first-class dimension operand for every physical array axis.
fn array_dimensions<C>(context: &C, value: &C::Value, rank: usize) -> Result<Vec<C::Value>, BatchingError>
where
    C: Context<Type = ArrayProgramType, Operation: From<DimensionSizeOperation>>,
{
    (0..rank).map(|axis| array_dimension(context, value, axis)).collect()
}

/// Binds one mixed dynamic broadcast against explicit first-class output dimensions.
fn broadcast_array<C>(
    context: &C,
    value: C::Value,
    output_dimensions: Vec<C::Value>,
    output_axes: Vec<usize>,
    output_sharding: Option<Sharding>,
) -> Result<C::Value, BatchingError>
where
    C: Context<Type = ArrayProgramType, Operation: From<BroadcastOperation>>,
{
    let operation = BroadcastOperation::new(output_axes).with_output_sharding(output_sharding);
    let mut inputs = Vec::with_capacity(output_dimensions.len() + 1);
    inputs.push(value);
    inputs.extend(output_dimensions);
    Ok(context.bind(operation, Vec::new(), inputs.as_slice())?.remove(0))
}

impl<C> ArrayBatchingPolicy<ProjectedContext<C, ArrayType>> for DynamicArrayBatchingPolicy
where
    C: Context<
            Type = ArrayProgramType,
            Operation: From<BroadcastOperation> + From<DimensionSizeOperation> + OperationProjection<ArrayType>,
        >,
    C::Constant: From<DimensionValue> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
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
        let mut output_dimensions = array_dimensions(outer_context, &value, array_type.rank())?;
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
        let array_type = <&ArrayType>::try_from(value.r#type().as_ref())?.clone();
        ArrayBatch::new(
            array_type,
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
                    DimensionSource::Static(extent) => {
                        Ok(outer_context.lift(DimensionValue::constant(extent).map_err(ProgramError::from)?.into())?)
                    }
                    DimensionSource::Value { source, axis } => {
                        let source = <C::Value as ValueProjection<ArrayType>>::from_projected(source);
                        array_dimension(outer_context, &source, axis)
                    }
                    DimensionSource::BatchExtent => Ok(context.axis_extent().clone()),
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let value = <C::Value as ValueProjection<ArrayType>>::from_projected(input.value().clone());
        let value = broadcast_array(outer_context, value, output_dimensions, output_axes, r#type.sharding().cloned())?;
        ArrayBatch::new(r#type, <C::Value as ValueProjection<ArrayType>>::into_projected(value)?, batch_axis)
    }
}

/// Aligns one composite array batch to `axis`, moving an existing mapped axis or dynamically broadcasting a
/// replicated array with the context's first-class extent.
fn align_array_batch<C>(
    context: &BatchingContext<C, ArrayProgramBatching>,
    batch: ArrayProgramBatch<C::Value>,
    axis: Axis,
) -> Result<ArrayProgramBatch<C::Value>, BatchingError>
where
    C: Context<
            Type = ArrayProgramType,
            Operation: From<BroadcastOperation> + From<DimensionSizeOperation> + OperationProjection<ArrayType>,
        >,
    C::Constant: From<DimensionValue> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Value: ValueProjection<ArrayType, Projected: Transpose + Value<Type = ArrayType>>,
{
    let value_type = batch.value.r#type();
    let array_type = match value_type.as_ref() {
        ArrayProgramType::Array(r#type) => r#type.clone(),
        ArrayProgramType::Dimension(r#type) => {
            return Err(BatchingError::MappedDimension {
                r#type: Box::new(r#type.clone()),
                axis: BatchAxis::from(axis),
            });
        }
    };
    let batch = ArrayBatch::new(
        array_type,
        <C::Value as ValueProjection<ArrayType>>::into_projected(batch.value)?,
        batch.batch_axis,
    )?;
    let projected_context = BatchingContext::<_, ArrayBatching<DynamicArrayBatchingPolicy>>::with_policy(
        ProjectedContext::new(context.parent().clone()),
        context.axis_extent().clone(),
    )
    .with_axis_name(context.axis_name().map(str::to_string))
    .with_axis_sharding(context.axis_sharding().clone());
    let output = DynamicArrayBatchingPolicy::match_axis(&projected_context, &batch, axis)?;
    let batch_axis = output.batch_axis();
    ArrayProgramBatch::new(<C::Value as ValueProjection<ArrayType>>::from_projected(output.into_value()), batch_axis)
}

/// Normalizes one mapped composite array input to the batching context's common physical sharding placement.
fn normalize_array_input<C>(
    context: &BatchingContext<C, ArrayProgramBatching>,
    batch: ArrayProgramBatch<C::Value>,
) -> Result<ArrayProgramBatch<C::Value>, BatchingError>
where
    C: Context<Type = ArrayProgramType, Operation: From<BroadcastOperation> + From<DimensionSizeOperation>>,
{
    let Some(position) = batch.batch_axis_position() else {
        return Ok(batch);
    };
    let value_type = batch.value.r#type();
    let array_type = <&ArrayType>::try_from(value_type.as_ref())?;
    let Some(normalized_type) = normalized_batch_axis_type(array_type, position, context.axis_sharding())? else {
        return Ok(batch);
    };

    let mut output_dimensions = array_dimensions(context.parent(), &batch.value, array_type.rank())?;
    output_dimensions[position] = context.axis_extent().clone();
    let output_axes = (0..array_type.rank()).collect::<Vec<_>>();
    let batch_axis = batch.batch_axis;
    let value = broadcast_array(
        context.parent(),
        batch.value,
        output_dimensions,
        output_axes,
        normalized_type.sharding().cloned(),
    )?;
    ArrayProgramBatch::new(value, batch_axis)
}

impl<C> BatchingEntrypointPolicy<C> for ArrayProgramBatching
where
    C: Context<
            Type = ArrayProgramType,
            Operation: From<BroadcastOperation>
                           + From<DimensionSizeOperation>
                           + OperationProjection<ArrayType>
                           + OperationProjection<DimensionType>,
        >,
    C::Constant: From<DimensionValue>
        + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>
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
            .map(|(input, input_batch_axis)| ArrayProgramBatch::new(input, input_batch_axis))
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
                Cow::Borrowed(ArrayProgramType::Array(array_type)) => Cow::Borrowed(array_type),
                Cow::Owned(ArrayProgramType::Array(array_type)) => Cow::Owned(array_type),
                _ => return None,
            };
            Some((array_type, batch.batch_axis_position()))
        }))?;
        let batching_context = BatchingContext::new(context.clone(), axis_extent)
            .with_axis_name(batch_axis.name().map(String::from))
            .with_axis_sharding(axis_sharding);
        let batches = batches
            .into_iter()
            .map(|batch| normalize_array_input(&batching_context, batch))
            .collect::<Result<Vec<_>, _>>()?;
        Ok((batching_context, batches))
    }

    fn materialize_output(
        context: &BatchingContext<C, Self>,
        output: Self::Batch,
        output_batch_axis: BatchAxis,
    ) -> Result<C::Value, BatchingError> {
        match (output.batch_axis.axis(), output_batch_axis.axis()) {
            (None, None) => Ok(output.into_value()),
            (Some(_), None) => {
                Err(BatchingError::MismatchedOutputAxes { expected: output_batch_axis, actual: output.batch_axis })
            }
            (_, Some(axis)) => Ok(align_array_batch(context, output, axis)?.into_value()),
        }
    }
}

impl<C> RecursiveBatchingPolicy<C> for ArrayProgramBatching
where
    C: Context<Type = ArrayProgramType>,
    C::Operation: BatchableOperation<C, ArrayProgramBatching>,
{
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
                instruction.operation().batch(context, &RecursiveBatchingDriver::new(&regions), instruction_inputs)
            },
        )
    }

    fn batch_program(
        _context: &BatchingContext<C, Self>,
        _region: RegionRef<'_, C::Constant, C::Operation>,
        _input_axes: &[BatchAxis],
        _output_axes_policy: ProgramBatchingOutputAxesPolicy,
    ) -> Result<(BatchedProgram<C>, Vec<BatchAxis>), BatchingError> {
        Err(BatchingError::UnsupportedOperation {
            message: "structural batching of composite regions has not been migrated to the batching policy"
                .to_string(),
        })
    }
}

impl<C> NamedAxes for BatchingContext<C, ArrayProgramBatching>
where
    C: NamedAxes<Type = ArrayProgramType>,
    C::Constant: ValueProjection<DimensionType, Projected = DimensionValue>,
    C::Operation: BatchableOperation<C, ArrayProgramBatching>,
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

/// Projects composite array batches, applies one homogeneous array batching rule, and lifts its outputs.
fn batch_projected_array_operation<C, O>(
    operation: &O,
    context: &BatchingContext<C, ArrayProgramBatching>,
    inputs: &[ArrayProgramBatch<C::Value>],
) -> Result<Vec<ArrayProgramBatch<C::Value>>, BatchingError>
where
    C: Context<
            Type = ArrayProgramType,
            Operation: From<BroadcastOperation> + From<DimensionSizeOperation> + OperationProjection<ArrayType>,
        >,
    C::Constant: From<DimensionValue> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Value: ValueProjection<ArrayType, Projected: Transpose + Value<Type = ArrayType>>,
    O: Operation<ArrayType>
        + BatchableOperation<ProjectedContext<C, ArrayType>, ArrayBatching<DynamicArrayBatchingPolicy>>,
{
    let inputs = inputs
        .iter()
        .map(|input| {
            let r#type = <&ArrayType>::try_from(input.value.r#type().as_ref())?.clone();
            ArrayBatch::new(
                r#type,
                <C::Value as ValueProjection<ArrayType>>::into_projected(input.value.clone())?,
                input.batch_axis,
            )
        })
        .collect::<Result<Vec<_>, BatchingError>>()?;
    let projected_context = BatchingContext::<_, ArrayBatching<DynamicArrayBatchingPolicy>>::with_policy(
        ProjectedContext::new(context.parent().clone()),
        context.axis_extent().clone(),
    )
    .with_axis_name(context.axis_name().map(str::to_string))
    .with_axis_sharding(context.axis_sharding().clone());
    operation
        .batch(&projected_context, &EmptyRegionDriver, inputs.as_slice())?
        .into_iter()
        .map(|output| {
            let batch_axis = output.batch_axis();
            ArrayProgramBatch::new(
                <C::Value as ValueProjection<ArrayType>>::from_projected(output.into_value()),
                batch_axis,
            )
        })
        .collect()
}

/// Applies one matching-axis homogeneous collective batching rule while retaining the composite operation's explicit
/// result extents as replicated shape values.
fn batch_explicit_shape_changing_collective<C, O>(
    operation: &O,
    axis_name: &str,
    context: &BatchingContext<C, ArrayProgramBatching>,
    inputs: &[ArrayProgramBatch<C::Value>],
) -> Result<Vec<ArrayProgramBatch<C::Value>>, BatchingError>
where
    C: Context<
            Type = ArrayProgramType,
            Operation: From<BroadcastOperation> + From<DimensionSizeOperation> + OperationProjection<ArrayType>,
        >,
    C::Constant: From<DimensionValue>
        + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>
        + ValueProjection<DimensionType, Projected = DimensionValue>,
    C::Value: ValueProjection<ArrayType, Projected: Transpose + Value<Type = ArrayType>>,
    O: Operation<ArrayType>
        + BatchableOperation<ProjectedContext<C, ArrayType>, ArrayBatching<DynamicArrayBatchingPolicy>>,
{
    let Some((array, extents)) = inputs.split_first() else {
        return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
    };
    for extent in extents {
        extent.validate_replicated_dimension()?;
    }
    if context.axis_name() != Some(axis_name) {
        return Err(BatchingError::UnsupportedOperation {
            message: format!(
                "'{}' forwarding through a different composite named axis requires the Phase 5 collective policy",
                operation.name(),
            ),
        });
    }
    batch_projected_array_operation(operation, context, std::slice::from_ref(array))
}

// TODO(eaplatanios): Move this to the module where `ConcatenateOperation` is defined.
impl<C: Context<Type = ArrayProgramType>> BatchableOperation<C, ArrayProgramBatching> for ConcatenateOperation
where
    C::Constant: From<DimensionValue> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Value: ValueProjection<ArrayType, Projected: LegacyBroadcast + Transpose + Value<Type = ArrayType>>,
    C::Operation: From<BroadcastOperation>
        + From<ConcatenateOperation>
        + From<DimensionSizeOperation>
        + OperationProjection<ArrayType>,
{
    fn batch<D: BatchingDriver<C, ArrayProgramBatching>>(
        &self,
        context: &BatchingContext<C, ArrayProgramBatching>,
        _driver: &D,
        inputs: &[ArrayProgramBatch<C::Value>],
    ) -> Result<Vec<ArrayProgramBatch<C::Value>>, BatchingError> {
        let Some((result_extent, inputs)) = inputs.split_last() else {
            return Err(TypeError::invalid(format!(
                "'{}' expects at least one array followed by its result extent",
                CONCATENATE_OPERATION_NAME,
            ))
            .into());
        };
        if inputs.is_empty() {
            return match result_extent.unbatched_type() {
                ArrayProgramType::Array(_) => Err(TypeError::invalid(format!(
                    "'{}' expects a trailing result-extent dimension",
                    CONCATENATE_OPERATION_NAME,
                ))
                .into()),
                ArrayProgramType::Dimension(_) => Err(TypeError::invalid(format!(
                    "'{}' expects at least one array before its result extent",
                    CONCATENATE_OPERATION_NAME,
                ))
                .into()),
            };
        }
        // A mapped extent would authorize a different output shape for each batch item, which requires a ragged
        // representation. Concatenate therefore accepts only one replicated result extent.
        result_extent.validate_replicated_dimension()?;

        let Some(batch_axis) = inputs.iter().find_map(ArrayProgramBatch::batch_axis_position) else {
            return Ok(context
                .parent()
                .bind(
                    self.clone(),
                    Vec::new(),
                    &inputs
                        .iter()
                        .chain(std::iter::once(result_extent))
                        .map(|input| input.value.clone())
                        .collect::<Vec<_>>(),
                )?
                .into_iter()
                .map(ArrayProgramBatch::replicated)
                .collect());
        };

        // Align every physical array on one mapped axis. Replicated operands gain that axis using the transform's
        // declared sharding, so each batch item concatenates the corresponding logical arrays.
        let aligned_inputs = inputs
            .iter()
            .cloned()
            .map(|input| align_array_batch(context, input, Axis::from(batch_axis)))
            .collect::<Result<Vec<_>, _>>()?;
        let lifted_axis = if batch_axis <= self.axis() { self.axis() + 1 } else { self.axis() };
        let first_type = aligned_inputs[0].value.r#type();
        let first_type = <&ArrayType>::try_from(first_type.as_ref())?;
        let lifted_operation = ConcatenateOperation::new(lifted_axis, first_type.rank())?;
        let mut lifted_inputs = aligned_inputs.into_iter().map(ArrayProgramBatch::into_value).collect::<Vec<_>>();
        lifted_inputs.push(result_extent.value.clone());
        context
            .parent()
            .bind(lifted_operation, Vec::new(), lifted_inputs.as_slice())?
            .into_iter()
            .map(|output| ArrayProgramBatch::new(output, BatchAxis::from_position(batch_axis)))
            .collect()
    }
}

impl<C: Context<Type = ArrayProgramType>> BatchableOperation<C, ArrayProgramBatching> for CustomCallOperation {
    fn batch<D: BatchingDriver<C, ArrayProgramBatching>>(
        &self,
        _context: &BatchingContext<C, ArrayProgramBatching>,
        _driver: &D,
        _inputs: &[ArrayProgramBatch<C::Value>],
    ) -> Result<Vec<ArrayProgramBatch<C::Value>>, BatchingError> {
        Err(BatchingError::UnsupportedOperation {
            message: format!(
                "custom call '{}' has no batching rule; invoke a kernel that understands the batch axis instead",
                self.target_name(),
            ),
        })
    }
}

impl<C: Context<Type = ArrayProgramType>> BatchableOperation<C, ArrayProgramBatching> for PadOperation
where
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>
        + ValueProjection<DimensionType, Projected = DimensionValue>
        + From<DimensionValue>,
    C::Value: ValueProjection<ArrayType, Projected: LegacyBroadcast + Transpose + Value<Type = ArrayType>>,
    C::Operation: From<BroadcastOperation>
        + From<DimensionSizeOperation>
        + From<OneOperation<ArrayType>>
        + From<PadOperation>
        + OperationProjection<ArrayType, Projected: From<SelectOperation> + From<ZeroOperation<ArrayType>>>,
{
    fn batch<D: BatchingDriver<C, ArrayProgramBatching>>(
        &self,
        context: &BatchingContext<C, ArrayProgramBatching>,
        _driver: &D,
        inputs: &[ArrayProgramBatch<C::Value>],
    ) -> Result<Vec<ArrayProgramBatch<C::Value>>, BatchingError> {
        if inputs.len() < 2 {
            return Err(ProgramError::InvalidInputCount { expected: 2, actual: inputs.len() }.into());
        }
        let (array_inputs, output_extents) = inputs.split_at(2);
        let [operand, padding_value] = array_inputs else {
            unreachable!();
        };
        <&ArrayType>::try_from(operand.unbatched_type())?;
        <&ArrayType>::try_from(padding_value.unbatched_type())?;
        for extent in output_extents {
            extent.validate_replicated_dimension()?;
        }
        let operand_type = <&ArrayType>::try_from(operand.value.r#type().as_ref())?.clone();
        let padding_value_type = <&ArrayType>::try_from(padding_value.value.r#type().as_ref())?.clone();
        let operand_batch = ArrayBatch::new(
            operand_type,
            <C::Value as ValueProjection<ArrayType>>::into_projected(operand.value.clone())?,
            operand.batch_axis,
        )?;
        let padding_value_batch = ArrayBatch::new(
            padding_value_type,
            <C::Value as ValueProjection<ArrayType>>::into_projected(padding_value.value.clone())?,
            padding_value.batch_axis,
        )?;
        let Some(batch_axis) = operand_batch
            .batch_axis_position()
            .or(Some(0).filter(|_| !padding_value_batch.batch_axis().is_replicated()))
        else {
            return Ok(context
                .parent()
                .bind(self.clone(), Vec::new(), &inputs.iter().map(|input| input.value.clone()).collect::<Vec<_>>())?
                .into_iter()
                .map(ArrayProgramBatch::replicated)
                .collect());
        };

        let operand_batch = align_array_batch(context, operand.clone(), Axis::from(batch_axis))?;
        let operand_batch = ArrayBatch::new(
            <&ArrayType>::try_from(operand_batch.value.r#type().as_ref())?.clone(),
            <C::Value as ValueProjection<ArrayType>>::into_projected(operand_batch.value)?,
            operand_batch.batch_axis,
        )?;
        let mut edge_padding_low = self.edge_padding_low().to_vec();
        edge_padding_low.insert(batch_axis, 0);
        let mut edge_padding_high = self.edge_padding_high().to_vec();
        edge_padding_high.insert(batch_axis, 0);
        let mut interior_padding = self.interior_padding().to_vec();
        interior_padding.insert(batch_axis, 0);
        let lifted_operation = PadOperation::new(edge_padding_low, edge_padding_high, interior_padding)?;
        let batch_extent = context.axis_extent().clone();
        let mut lifted_output_extents = Vec::with_capacity(output_extents.len() + 1);
        lifted_output_extents.extend(output_extents[..batch_axis].iter().map(|extent| extent.value.clone()));
        lifted_output_extents.push(batch_extent);
        lifted_output_extents.extend(output_extents[batch_axis..].iter().map(|extent| extent.value.clone()));

        if padding_value_batch.batch_axis().is_replicated() {
            let mut lifted_inputs = Vec::with_capacity(lifted_output_extents.len() + 2);
            lifted_inputs.push(<C::Value as ValueProjection<ArrayType>>::from_projected(operand_batch.into_value()));
            lifted_inputs.push(padding_value.value.clone());
            lifted_inputs.extend(lifted_output_extents);
            return context
                .parent()
                .bind(lifted_operation, Vec::new(), lifted_inputs.as_slice())?
                .into_iter()
                .map(|output| ArrayProgramBatch::new(output, BatchAxis::from_position(batch_axis)))
                .collect();
        }

        // A mapped scalar padding value cannot be passed directly to `pad`, whose padding operand is scalar. Pad the
        // aligned operand with zero, construct a Boolean mask for the original input positions, broadcast the mapped
        // padding values across the result, and select them only at padding positions. This is the same semantic
        // decomposition as the homogeneous array rule, but every shape-changing operation receives the canonical
        // first-class result extents.
        let array_context = ProjectedContext::<C, ArrayType>::new(context.parent().clone());
        let padding_scalar_type = padding_value_batch.unbatched_type();
        let zero_padding =
            <C::Value as ValueProjection<ArrayType>>::from_projected(array_context.zero(&padding_scalar_type)?);
        let operand = <C::Value as ValueProjection<ArrayType>>::from_projected(operand_batch.into_value());
        let mut padded_inputs = Vec::with_capacity(lifted_output_extents.len() + 2);
        padded_inputs.push(operand.clone());
        padded_inputs.push(zero_padding);
        padded_inputs.extend(lifted_output_extents.iter().cloned());
        let mut padded = context.parent().bind(lifted_operation.clone(), Vec::new(), padded_inputs.as_slice())?;
        check_count!("output", padded, 1, ProgramError);
        let padded = padded.remove(0);

        let operand_type = <&ArrayType>::try_from(operand.r#type().as_ref())?
            .clone()
            .with_data_type(DataType::Boolean)
            .with_layout(None);
        let mask_input_dimensions = operand_type
            .shape()
            .dimensions()
            .iter()
            .enumerate()
            .filter_map(|(axis, dimension)| {
                matches!(dimension, Dimension::Dynamic(_)).then(|| {
                    if axis == batch_axis {
                        Ok(context.axis_extent().clone())
                    } else {
                        array_dimension(context.parent(), &operand, axis)
                    }
                })
            })
            .collect::<Result<Vec<_>, BatchingError>>()?;
        let mut mask_input =
            context
                .parent()
                .bind(OneOperation::new(operand_type), Vec::new(), mask_input_dimensions.as_slice())?;
        check_count!("output", mask_input, 1, ProgramError);
        let mask_input = mask_input.remove(0);
        let mask_padding_type = padding_scalar_type.with_data_type(DataType::Boolean).with_layout(None);
        let mask_padding =
            <C::Value as ValueProjection<ArrayType>>::from_projected(array_context.zero(&mask_padding_type)?);
        let mut mask_inputs = Vec::with_capacity(lifted_output_extents.len() + 2);
        mask_inputs.push(mask_input);
        mask_inputs.push(mask_padding);
        mask_inputs.extend(lifted_output_extents.iter().cloned());
        let mut mask = context.parent().bind(lifted_operation, Vec::new(), mask_inputs.as_slice())?;
        check_count!("output", mask, 1, ProgramError);
        let mask = mask.remove(0);

        let mut broadcast_inputs = Vec::with_capacity(lifted_output_extents.len() + 1);
        broadcast_inputs.push(<C::Value as ValueProjection<ArrayType>>::from_projected(
            padding_value_batch.move_axis(0)?.into_value(),
        ));
        broadcast_inputs.extend(lifted_output_extents);
        let mut broadcasted_padding = context.parent().bind(
            BroadcastOperation::new(vec![batch_axis]),
            Vec::new(),
            broadcast_inputs.as_slice(),
        )?;
        check_count!("output", broadcasted_padding, 1, ProgramError);
        let broadcasted_padding = broadcasted_padding.remove(0);

        let mask = <C::Value as ValueProjection<ArrayType>>::into_projected(mask)?;
        let padded = <C::Value as ValueProjection<ArrayType>>::into_projected(padded)?;
        let broadcasted_padding = <C::Value as ValueProjection<ArrayType>>::into_projected(broadcasted_padding)?;
        let mut output = array_context.bind(SelectOperation, Vec::new(), &[mask, padded, broadcasted_padding])?;
        check_count!("output", output, 1, ProgramError);
        Ok(vec![ArrayProgramBatch::new(
            <C::Value as ValueProjection<ArrayType>>::from_projected(output.remove(0)),
            BatchAxis::from_position(batch_axis),
        )?])
    }
}

impl<C: Context<Type = ArrayProgramType>> BatchableOperation<C, ArrayProgramBatching> for RngBitGeneratorOperation {
    fn batch<D: BatchingDriver<C, ArrayProgramBatching>>(
        &self,
        _context: &BatchingContext<C, ArrayProgramBatching>,
        _driver: &D,
        inputs: &[ArrayProgramBatch<C::Value>],
    ) -> Result<Vec<ArrayProgramBatch<C::Value>>, BatchingError> {
        let Some((state, output_extents)) = inputs.split_first() else {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
        };
        for extent in output_extents {
            extent.validate_replicated_dimension()?;
        }
        if state.batch_axis().is_replicated() {
            return Err(BatchingError::UnsupportedOperation {
                message: "'rng_bit_generator' cannot batch a replicated state because every batch item would see \
                          the same state; derive one state per batch item with `split_key` and map over the states \
                          explicitly"
                    .to_string(),
            });
        }
        Err(BatchingError::UnsupportedOperation {
            message: "'rng_bit_generator' batching requires Phase 5 composite scan-region support".to_string(),
        })
    }
}

impl<A, C> BatchableOperation<C, ArrayProgramBatching> for ArrayProgramOperation<A>
where
    A: Value<Type = ArrayType>,
    C: Context<
            Type = ArrayProgramType,
            Constant: ValueProjection<ArrayType, Projected = A>
                          + ValueProjection<DimensionType, Projected = DimensionValue>
                          + From<DimensionValue>,
        >,
    C::Value: ValueProjection<ArrayType> + ValueProjection<DimensionType, Projected: Value<Type = DimensionType>>,
    <C::Value as ValueProjection<ArrayType>>::Projected:
        LegacyBroadcast + Reduce + Reshape + Transpose + Value<Type = ArrayType>,
    C::Operation: From<ArrayProgramOperation<A>>
        + From<BroadcastOperation>
        + From<ConcatenateOperation>
        + From<DimensionSizeOperation>
        + From<OneOperation<ArrayType>>
        + From<PadOperation>
        + OperationProjection<ArrayType, Projected = ArrayOperation<A>>
        + OperationProjection<DimensionType, Projected = DimensionOperation<DimensionValue>>,
    ArrayOperation<A>: BatchableOperation<ProjectedContext<C, ArrayType>, ArrayBatching<DynamicArrayBatchingPolicy>>,
{
    fn batch<D: BatchingDriver<C, ArrayProgramBatching>>(
        &self,
        context: &BatchingContext<C, ArrayProgramBatching>,
        driver: &D,
        inputs: &[ArrayProgramBatch<C::Value>],
    ) -> Result<Vec<ArrayProgramBatch<C::Value>>, BatchingError> {
        match self {
            Self::Zero(_) => {
                if !inputs.is_empty() {
                    return Err(ProgramError::InvalidInputCount { expected: 0, actual: inputs.len() }.into());
                }
                Ok(context
                    .parent()
                    .bind(self.clone(), Vec::new(), &[])?
                    .into_iter()
                    .map(ArrayProgramBatch::replicated)
                    .collect())
            }
            Self::DynamicZero(_) | Self::DynamicOne(_) | Self::DynamicIota(_) => {
                // Output extents are shared shape values. A mapped extent would request a different output shape
                // for each batch item, which requires a ragged representation that ordinary array batching lacks.
                for extent in inputs {
                    extent.validate_replicated_dimension()?;
                }
                Ok(context
                    .parent()
                    .bind(
                        self.clone(),
                        Vec::new(),
                        &inputs.iter().map(|input| input.value.clone()).collect::<Vec<_>>(),
                    )?
                    .into_iter()
                    .map(ArrayProgramBatch::replicated)
                    .collect())
            }
            Self::Array(operation) => batch_projected_array_operation(operation, context, inputs),
            Self::Dimension(operation) => {
                for input in inputs {
                    input.validate_replicated_dimension()?;
                }
                let inputs = inputs
                    .iter()
                    .map(|input| <C::Value as ValueProjection<DimensionType>>::into_projected(input.value.clone()))
                    .collect::<Result<Vec<_>, TypeError>>()?;
                Ok(ProjectedContext::<C, DimensionType>::new(context.parent().clone())
                    .bind(operation.clone(), Vec::new(), inputs.as_slice())?
                    .into_iter()
                    .map(<C::Value as ValueProjection<DimensionType>>::from_projected)
                    .map(ArrayProgramBatch::replicated)
                    .collect())
            }
            Self::Compare(_) => {
                let [left, right] = inputs else {
                    return Err(ProgramError::InvalidInputCount { expected: 2, actual: inputs.len() }.into());
                };
                // First-class dimensions describe one shared array shape, so mapping either operand would make the
                // comparison predicate vary per batch item without a corresponding ragged-shape model.
                left.validate_replicated_dimension()?;
                right.validate_replicated_dimension()?;
                // Comparing two replicated dimensions produces ordinary replicated Boolean array data.
                Ok(context
                    .parent()
                    .bind(self.clone(), Vec::new(), &[left.value.clone(), right.value.clone()])?
                    .into_iter()
                    .map(ArrayProgramBatch::replicated)
                    .collect())
            }
            Self::DimensionFromScalar(operation) => {
                let [input] = inputs else {
                    return Err(ProgramError::InvalidInputCount { expected: 1, actual: inputs.len() }.into());
                };
                <&ArrayType>::try_from(input.unbatched_type())?;
                if !input.batch_axis.is_replicated() {
                    return Err(BatchingError::MappedDimension {
                        r#type: Box::new(operation.result_type().clone()),
                        axis: input.batch_axis,
                    });
                }
                Ok(context
                    .parent()
                    .bind(self.clone(), Vec::new(), std::slice::from_ref(&input.value))?
                    .into_iter()
                    .map(ArrayProgramBatch::replicated)
                    .collect())
            }
            Self::DimensionToScalar(DimensionToScalarOperation) => {
                let [input] = inputs else {
                    return Err(ProgramError::InvalidInputCount { expected: 1, actual: inputs.len() }.into());
                };
                input.validate_replicated_dimension()?;
                Ok(context
                    .parent()
                    .bind(self.clone(), Vec::new(), std::slice::from_ref(&input.value))?
                    .into_iter()
                    .map(ArrayProgramBatch::replicated)
                    .collect())
            }
            Self::DimensionSize(operation) => {
                let [input] = inputs else {
                    return Err(ProgramError::InvalidInputCount { expected: 1, actual: inputs.len() }.into());
                };
                let input_type = input.value.r#type();
                let physical_type = <&ArrayType>::try_from(input_type.as_ref())?;
                let physical_axis = match input.batch_axis.axis() {
                    Some(batch_axis) => {
                        let batch_axis = batch_axis.normalize(physical_type.rank())?;
                        if operation.axis() < batch_axis { operation.axis() } else { operation.axis() + 1 }
                    }
                    None => operation.axis(),
                };
                let operation = DimensionSizeOperation::new(physical_type, physical_axis)?;
                Ok(context
                    .parent()
                    .bind(ArrayProgramOperation::<A>::from(operation), Vec::new(), std::slice::from_ref(&input.value))?
                    .into_iter()
                    .map(ArrayProgramBatch::replicated)
                    .collect())
            }
            Self::Concatenate(operation) => BatchableOperation::batch(operation, context, driver, inputs),
            Self::CustomCall(operation) => BatchableOperation::batch(operation, context, driver, inputs),
            Self::Pad(operation) => BatchableOperation::batch(operation, context, driver, inputs),
            Self::RngBitGenerator(operation) => BatchableOperation::batch(operation, context, driver, inputs),
            Self::AllGather(operation) => {
                batch_explicit_shape_changing_collective::<C, _>(operation, operation.axis_name(), context, inputs)
            }
            Self::PSumScatter(operation) => {
                batch_explicit_shape_changing_collective::<C, _>(operation, operation.axis_name(), context, inputs)
            }
            Self::AllToAll(operation) => {
                batch_explicit_shape_changing_collective::<C, _>(operation, operation.axis_name(), context, inputs)
            }
            Self::Reshape(operation) => {
                let Some((input, output_extents)) = inputs.split_first() else {
                    return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
                };
                <&ArrayType>::try_from(input.unbatched_type())?;
                for extent in output_extents {
                    extent.validate_replicated_dimension()?;
                }

                if input.batch_axis.is_replicated() {
                    return Ok(context
                        .parent()
                        .bind(
                            self.clone(),
                            Vec::new(),
                            &inputs.iter().map(|input| input.value.clone()).collect::<Vec<_>>(),
                        )?
                        .into_iter()
                        .map(ArrayProgramBatch::replicated)
                        .collect());
                }

                let physical_type = <&ArrayType>::try_from(input.value.r#type().as_ref())?.clone();
                let moved_input = ArrayBatch::new(
                    physical_type,
                    <C::Value as ValueProjection<ArrayType>>::into_projected(input.value.clone())?,
                    input.batch_axis,
                )?
                .move_axis(0)?;
                let moved_input = <C::Value as ValueProjection<ArrayType>>::from_projected(moved_input.into_value());
                let axis_extent = context.axis_extent().clone();

                let mut lifted_operation = ReshapeOperation::new();
                if let Some(dimensions) = operation.dimensions() {
                    let mut lifted_dimensions = Vec::with_capacity(dimensions.len() + 1);
                    lifted_dimensions.push(0);
                    lifted_dimensions.extend(dimensions.iter().map(|dimension| dimension + 1));
                    lifted_operation = lifted_operation.with_dimensions(lifted_dimensions);
                }
                if let Some(output_sharding) = operation.output_sharding() {
                    lifted_operation = lifted_operation.with_output_sharding(
                        lift_output_sharding_for_leading_batch_axis(output_sharding, context.axis_sharding().clone())?,
                    );
                }

                let mut lifted_inputs = Vec::with_capacity(inputs.len() + 1);
                lifted_inputs.push(moved_input);
                lifted_inputs.push(axis_extent);
                lifted_inputs.extend(output_extents.iter().map(|extent| extent.value.clone()));
                context
                    .parent()
                    .bind(ArrayProgramOperation::<A>::from(lifted_operation), Vec::new(), lifted_inputs.as_slice())?
                    .into_iter()
                    .map(|output| ArrayProgramBatch::new(output, BatchAxis::from_position(0)))
                    .collect()
            }
            Self::Broadcast(operation) => {
                let Some((input, output_extents)) = inputs.split_first() else {
                    return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
                };
                <&ArrayType>::try_from(input.unbatched_type())?;
                for extent in output_extents {
                    extent.validate_replicated_dimension()?;
                }

                if input.batch_axis.is_replicated() {
                    return Ok(context
                        .parent()
                        .bind(
                            self.clone(),
                            Vec::new(),
                            &inputs.iter().map(|input| input.value.clone()).collect::<Vec<_>>(),
                        )?
                        .into_iter()
                        .map(ArrayProgramBatch::replicated)
                        .collect());
                }

                // Canonicalize the physical mapped axis to the leading position, then represent that axis in both the
                // explicit output extents and the input-to-output mapping of the lifted broadcast.
                let physical_type = <&ArrayType>::try_from(input.value.r#type().as_ref())?.clone();
                let moved_input = ArrayBatch::new(
                    physical_type,
                    <C::Value as ValueProjection<ArrayType>>::into_projected(input.value.clone())?,
                    input.batch_axis,
                )?
                .move_axis(0)?;
                let moved_input = <C::Value as ValueProjection<ArrayType>>::from_projected(moved_input.into_value());
                let axis_extent = context.axis_extent().clone();

                let mut lifted_output_axes = Vec::with_capacity(operation.output_axes().len() + 1);
                lifted_output_axes.push(0);
                lifted_output_axes.extend(operation.output_axes().iter().map(|axis| axis + 1));
                let mut lifted_operation = BroadcastOperation::new(lifted_output_axes);
                if let Some(output_sharding) = operation.output_sharding() {
                    lifted_operation = lifted_operation.with_output_sharding(
                        lift_output_sharding_for_leading_batch_axis(output_sharding, context.axis_sharding().clone())?,
                    );
                }

                let mut lifted_inputs = Vec::with_capacity(inputs.len() + 1);
                lifted_inputs.push(moved_input);
                lifted_inputs.push(axis_extent);
                lifted_inputs.extend(output_extents.iter().map(|extent| extent.value.clone()));
                context
                    .parent()
                    .bind(ArrayProgramOperation::<A>::from(lifted_operation), Vec::new(), lifted_inputs.as_slice())?
                    .into_iter()
                    .map(|output| ArrayProgramBatch::new(output, BatchAxis::from_position(0)))
                    .collect()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::Scalar;
    use crate::backends::array_programs::ArrayProgramValue;
    use crate::backends::arrays::Array;
    use crate::backends::dimensions::DimensionValue;
    use crate::batching::{
        Batch, BatchAxisSpecification, BatchingPolicy, BatchingTracer, InterpretableBatchableOperation,
        RecursiveBatchingPolicy, batch,
    };
    use crate::contexts::{EagerContext, StagingContext};
    use crate::operations::collectives::{
        AllGatherOperation, AllGatherOutputVariance, AllToAllOperation, CollectiveOptions, PSumScatterOperation,
    };
    use crate::operations::compare::{CompareOperation, ComparisonDirection};
    use crate::operations::constants::{IotaOperation, OneOperation, ZeroOperation};
    use crate::operations::dimensions::{
        DimensionAddOperation, DimensionFromScalar, DimensionFromScalarOperation, DimensionSize, DimensionToScalar,
        DimensionToScalarOperation,
    };
    use crate::operations::math::{AddOperation, NegOperation};
    use crate::operations::random::{RandomAlgorithm, RngBitGeneratorOperation};
    use crate::operations::{CollectiveKind, CollectiveOperation};
    use crate::parameters::Placeholder;
    use crate::programs::ProgramBuilder;
    use crate::sharding::ShardingDimension;
    use crate::tracing::TracingContext;
    use crate::types::dimensions::{DimensionBounds, DimensionVariable};
    use crate::types::{DataType, Dimension, Shape};

    use super::*;

    #[test]
    fn test_array_program_batch_entrypoints() -> Result<(), ProgramError> {
        let matrix = ArrayProgramValue::Array(Array::matrix(2, 3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]));

        // The free transform infers its first-class mapped extent from the physical array input and can move
        // the mapped output axis without exposing the policy at the call site.
        let moved: ArrayProgramValue<Array> =
            batch(|row| Ok(row), matrix.clone(), BatchAxis::new(0), BatchAxis::new(1), None)?;
        assert_eq!(moved, ArrayProgramValue::Array(Array::matrix(3, 2, vec![1.0_f32, 4.0, 2.0, 5.0, 3.0, 6.0],)),);

        // A replicated array output is dynamically broadcast with the inferred extent operand.
        let replicated = ArrayProgramValue::Array(Array::vector(vec![10.0_f32, 20.0, 30.0]));
        let broadcasted: ArrayProgramValue<Array> = batch(
            |(_, replicated)| Ok(replicated),
            (matrix.clone(), replicated),
            (BatchAxis::new(0), BatchAxis::replicated()),
            BatchAxis::new(0),
            None,
        )?;
        assert_eq!(
            broadcasted,
            ArrayProgramValue::Array(Array::matrix(2, 3, vec![10.0_f32, 20.0, 30.0, 10.0, 20.0, 30.0],)),
        );

        // A named composite specification reaches the policy-selected context, and an explicit first-class extent
        // drives mapped output materialization when every input is replicated.
        let named_extent =
            BatchAxisSpecification::new(ArrayProgramValue::Dimension(DimensionValue::constant(2)?), "items");
        let explicitly_broadcasted: ArrayProgramValue<Array> = batch(
            |replicated| {
                assert_eq!(replicated.context().axis_name(), Some("items"));
                Ok(replicated)
            },
            ArrayProgramValue::Array(Array::vector(vec![7.0_f32, 8.0, 9.0])),
            BatchAxis::replicated(),
            BatchAxis::new(0),
            named_extent,
        )?;
        assert_eq!(
            explicitly_broadcasted,
            ArrayProgramValue::Array(Array::matrix(2, 3, vec![7.0_f32, 8.0, 9.0, 7.0, 8.0, 9.0],)),
        );

        // Exact zero extents use the same first-class extent path and produce an empty mapped dimension.
        let empty_extent =
            BatchAxisSpecification::with_extent(ArrayProgramValue::Dimension(DimensionValue::constant(0)?));
        let empty: ArrayProgramValue<Array> = batch(
            |replicated| Ok(replicated),
            ArrayProgramValue::Array(Array::vector(vec![7.0_f32, 8.0, 9.0])),
            BatchAxis::replicated(),
            BatchAxis::new(0),
            empty_extent,
        )?;
        assert_eq!(
            empty.r#type().as_ref(),
            &ArrayProgramType::Array(ArrayType::new(
                DataType::F32,
                Shape::new(vec![Dimension::Static(0), Dimension::Static(3)]),
            )),
        );

        // Dimension values remain shared shape values and can flow through the closure only as replicated outputs.
        let extent = ArrayProgramValue::Dimension(DimensionValue::constant(3)?);
        let dimension: ArrayProgramValue<Array> = batch(
            |(_, extent)| Ok(extent),
            (matrix.clone(), extent.clone()),
            (BatchAxis::new(0), BatchAxis::replicated()),
            BatchAxis::replicated(),
            None,
        )?;
        assert_eq!(dimension, extent);
        let mapped_dimension: Result<ArrayProgramValue<Array>, BatchingError> = batch(
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
            BatchAxisSpecification::with_extent(ArrayProgramValue::Dimension(DimensionValue::constant(3)?));
        let mismatched: Result<ArrayProgramValue<Array>, BatchingError> =
            batch(|row| Ok(row), matrix.clone(), BatchAxis::new(0), BatchAxis::new(0), mismatched_extent);
        let mismatched = mismatched.unwrap_err();
        assert!(mismatched.to_string().contains("observed 3=3, size(axis=0)=2"), "{mismatched:?}");

        // A first-class dimension itself cannot be declared mapped at the transform boundary.
        let mapped_input: Result<ArrayProgramValue<Array>, BatchingError> = batch(
            |extent| Ok(extent),
            ArrayProgramValue::Dimension(DimensionValue::constant(2)?),
            BatchAxis::new(0),
            BatchAxis::replicated(),
            None,
        );
        assert!(matches!(mapped_input, Err(BatchingError::MappedDimension { axis, .. }) if axis == BatchAxis::new(0)));

        // Nested public batching selects the composite policy again from the outer batching context.
        let nested: ArrayProgramValue<Array> = batch(
            |row| batch(|item| Ok(item), row, BatchAxis::new(0), BatchAxis::new(0), None).map_err(Into::into),
            matrix.clone(),
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        )?;
        assert_eq!(nested, matrix);

        // Under staging, an inferred dynamic extent remains an explicit `dimension_size` result consumed by
        // the output broadcast rather than metadata reconstructed from the array type.
        type TraceContext = TracingContext<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>;
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
        assert!(matches!(builder.instructions()[0].operation(), ArrayProgramOperation::DimensionSize(_),));
        assert!(matches!(builder.instructions()[1].operation(), ArrayProgramOperation::DimensionSize(_),));
        assert!(matches!(builder.instructions()[2].operation(), ArrayProgramOperation::Broadcast(_),));
        assert_eq!(builder.instructions()[2].inputs().len(), 3);
        assert_eq!(
            staged.r#type().as_ref(),
            &ArrayProgramType::Array(ArrayType::new(
                DataType::F32,
                Shape::new(vec![Dimension::Dynamic(batch_variable), Dimension::Static(3)]),
            )),
        );
        drop(builder);

        Ok(())
    }

    #[test]
    fn test_array_program_batching_policy() -> Result<(), ProgramError> {
        type Parent = EagerContext<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>;
        type PolicyContext = BatchingContext<Parent, ArrayProgramBatching>;

        let axis_extent = ArrayProgramValue::Dimension(DimensionValue::constant(2).unwrap());
        let context = PolicyContext::new(Parent::new(), axis_extent.clone()).with_axis_name("items".to_string());
        assert_eq!(context.axis_extent(), &axis_extent);
        assert_eq!(context.axis_name(), Some("items"));

        // The policy-generic interpretation helper unpacks and repackages composite batches without projecting their
        // member kind or depending on the homogeneous array carrier.
        let direct_input = ArrayProgramBatch::new(
            ArrayProgramValue::Array(Array::matrix(2, 2, vec![1.0_f32, 2.0, 3.0, 4.0])),
            BatchAxis::new(0),
        )
        .unwrap();
        let [direct_output] = ArrayProgramOperation::Array(ArrayOperation::Neg(NegOperation))
            .interpret_with_batch_axes(&context, &[direct_input], &[BatchAxis::new(0)])?
            .try_into()
            .unwrap();
        assert_eq!(direct_output.batch_axis(), BatchAxis::new(0));
        assert_eq!(
            direct_output.value(),
            &ArrayProgramValue::Array(Array::matrix(2, 2, vec![-1.0_f32, -2.0, -3.0, -4.0])),
        );

        // The generic frame preserves the existing homogeneous array rule unchanged.
        let input = BatchingTracer::new(
            context.clone(),
            ArrayProgramBatch::new(
                ArrayProgramValue::Array(Array::matrix(2, 2, vec![1.0_f32, 2.0, 3.0, 4.0])),
                BatchAxis::new(0),
            )
            .unwrap(),
        );
        let [output] = context
            .bind(
                ArrayProgramOperation::Array(ArrayOperation::Neg(NegOperation)),
                Vec::new(),
                std::slice::from_ref(&input),
            )?
            .try_into()
            .unwrap();
        assert_eq!(output.batch().batch_axis(), BatchAxis::new(0));
        assert_eq!(
            output.batch().value(),
            &ArrayProgramValue::Array(Array::matrix(2, 2, vec![-1.0_f32, -2.0, -3.0, -4.0])),
        );

        // Dimension-only and mixed dimension/array boundaries remain replicated under the same generic frame.
        let left = ArrayProgramValue::Dimension(DimensionValue::constant(3).unwrap());
        let right = ArrayProgramValue::Dimension(DimensionValue::constant(4).unwrap());
        let operation = DimensionAddOperation::new(
            <&DimensionType>::try_from(left.r#type().as_ref()).unwrap(),
            <&DimensionType>::try_from(right.r#type().as_ref()).unwrap(),
        )
        .unwrap();
        let inputs = [
            BatchingTracer::new(context.clone(), ArrayProgramBatch::replicated(left)),
            BatchingTracer::new(context.clone(), ArrayProgramBatch::replicated(right)),
        ];
        let [dimension] = context
            .bind(ArrayProgramOperation::<Array>::Dimension(DimensionOperation::Add(operation)), Vec::new(), &inputs)?
            .try_into()
            .unwrap();
        let scalar = dimension.to_scalar().unwrap().into_batch();
        assert_eq!(scalar.batch_axis(), BatchAxis::replicated());
        assert_eq!(scalar.into_value(), ArrayProgramValue::Array(Array::scalar(7_i64)));

        let mapped_dimension = <ArrayProgramBatching as BatchingPolicy<Parent>>::batch(
            ArrayProgramValue::Dimension(DimensionValue::constant(2).unwrap()),
            BatchAxis::new(0),
        );
        assert!(
            matches!(mapped_dimension, Err(BatchingError::MappedDimension { axis, .. }) if axis == BatchAxis::new(0))
        );

        // A staged dynamic mapped extent remains an ordinary SSA operand of the lifted reshape.
        type TraceContext = TracingContext<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>;
        let trace = TraceContext::new();
        let batch_variable = DimensionVariable::new("batch", DimensionBounds::new(1, Some(9)).unwrap());
        let batch_type = DimensionType::new(batch_variable.clone());
        let physical_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(batch_variable), Dimension::Static(3)]));
        let batch_extent = trace.input(batch_type.clone().into());
        let input = trace.input(physical_type.clone().into());
        let output_extent = trace.constant(ArrayProgramValue::Dimension(DimensionValue::constant(3).unwrap()));
        let batch_extent_id = batch_extent.atom_id().unwrap();
        let input_id = input.atom_id().unwrap();
        let output_extent_id = output_extent.atom_id().unwrap();
        let context = BatchingContext::<_, ArrayProgramBatching>::new(trace.clone(), batch_extent)
            .with_axis_name("items".to_string());
        assert_eq!(context.named_axis("items"), Some(NamedAxis::Batched { size: None }));
        let inputs = [
            BatchingTracer::new(context.clone(), ArrayProgramBatch::new(input, BatchAxis::new(0))?),
            BatchingTracer::new(context.clone(), ArrayProgramBatch::replicated(output_extent)),
        ];
        let [output] = context
            .bind(ArrayProgramOperation::<Array>::from(ReshapeOperation::new()), Vec::new(), &inputs)?
            .try_into()
            .unwrap();
        let output_id = output.into_batch().into_value().atom_id().unwrap();
        let builder = trace.builder().borrow();
        let [instruction] = builder.instructions() else {
            panic!("expected one lifted reshape instruction");
        };
        assert_eq!(instruction.inputs(), &[input_id, batch_extent_id, output_extent_id]);
        drop(builder);
        let program = trace
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
                vec![output_id],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )?;
        let rendered = program.to_string();
        assert_eq!(
            rendered,
            indoc! {"
                lambda %0:dimension<batch ∈ [1, 9)>, %1:f32[batch, 3] .
                let %2:dimension<3> = const
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
        let elementwise_context =
            BatchingContext::<_, ArrayProgramBatching>::new(elementwise_trace.clone(), batch_extent);
        let inputs = [
            BatchingTracer::new(elementwise_context.clone(), ArrayProgramBatch::new(mapped, BatchAxis::new(0))?),
            BatchingTracer::new(elementwise_context.clone(), ArrayProgramBatch::replicated(replicated)),
        ];
        let [output] = elementwise_context
            .bind(ArrayProgramOperation::Array(ArrayOperation::Add(AddOperation)), Vec::new(), &inputs)?
            .try_into()
            .unwrap();
        let elementwise_output_id = output.batch().value().atom_id().unwrap();
        assert_eq!(output.batch().batch_axis(), BatchAxis::new(0));
        assert_eq!(
            output.r#type().as_ref(),
            &ArrayProgramType::Array(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)]),)),
        );
        let elementwise_builder = elementwise_trace.builder().borrow();
        let [broadcast, add] = elementwise_builder.instructions() else {
            panic!("expected one dynamic broadcast followed by one array add");
        };
        assert!(matches!(broadcast.operation(), ArrayProgramOperation::Broadcast(_)));
        assert_eq!(broadcast.inputs()[0], replicated_id);
        assert_eq!(broadcast.inputs()[1], batch_extent_id);
        assert_eq!(broadcast.inputs().len(), 3);
        assert!(matches!(add.operation(), ArrayProgramOperation::Array(ArrayOperation::Add(_))));
        drop(elementwise_builder);
        let elementwise_program = elementwise_trace
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
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
        let recursive_input = recursive_parent.input(physical_type.into());
        let recursive_axis_extent =
            recursive_parent.constant(ArrayProgramValue::Dimension(DimensionValue::constant(2).unwrap()));
        let recursive_context =
            BatchingContext::<_, ArrayProgramBatching>::new(recursive_parent, recursive_axis_extent);
        let recursive_outputs = ArrayProgramBatching::batch_region(
            &recursive_context,
            program.entry_region_ref(),
            vec![ArrayProgramBatch::replicated(recursive_batch_extent), ArrayProgramBatch::replicated(recursive_input)],
        )?;
        assert_eq!(recursive_outputs.len(), 1);
        assert_eq!(recursive_outputs[0].batch_axis(), BatchAxis::replicated());

        let mut destination = ProgramBuilder::new();
        let imported = destination.import_region(program.entry_region_ref());
        assert_eq!(destination.region_ref(imported)?.to_program().to_string(), rendered);

        Ok(())
    }

    #[test]
    fn test_dynamic_array_program_elementwise_dispatch_and_alignment() -> Result<(), ProgramError> {
        type TraceContext = TracingContext<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>;

        // A unary primitive already carrying a non-leading mapped axis stages directly through the generic
        // homogeneous-family arm without any alignment operation.
        let trace = TraceContext::new();
        let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(9))?);
        let extent = trace.input(DimensionType::new(batch.clone()).into());
        let mapped = trace.input(
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3), Dimension::Dynamic(batch.clone())]))
                .into(),
        );
        let context = BatchingContext::<_, ArrayProgramBatching>::new(trace.clone(), extent);
        let input = BatchingTracer::new(context.clone(), ArrayProgramBatch::new(mapped, BatchAxis::new(1))?);
        let [output] = context
            .bind(
                ArrayProgramOperation::Array(ArrayOperation::Neg(NegOperation)),
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
        assert!(matches!(negate.operation(), ArrayProgramOperation::Array(ArrayOperation::Neg(_))));
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
        let context = BatchingContext::<_, ArrayProgramBatching>::new(trace.clone(), extent);
        let inputs = [
            BatchingTracer::new(context.clone(), ArrayProgramBatch::new(left, BatchAxis::new(0))?),
            BatchingTracer::new(context.clone(), ArrayProgramBatch::new(right, BatchAxis::new(1))?),
        ];
        let [output] = context
            .bind(ArrayProgramOperation::Array(ArrayOperation::Add(AddOperation)), Vec::new(), &inputs)?
            .try_into()
            .unwrap();
        assert_eq!(output.batch().batch_axis(), BatchAxis::new(0));
        let builder = trace.builder().borrow();
        let [transpose, add] = builder.instructions() else {
            panic!("expected one transpose followed by one generic binary instruction");
        };
        assert!(matches!(transpose.operation(), ArrayProgramOperation::Array(ArrayOperation::Transpose(_)),));
        assert!(matches!(add.operation(), ArrayProgramOperation::Array(ArrayOperation::Add(_))));
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
        let context = BatchingContext::<_, ArrayProgramBatching>::new(trace.clone(), extent.clone());
        let inputs = [
            BatchingTracer::new(context.clone(), ArrayProgramBatch::new(mapped.clone(), BatchAxis::new(0))?),
            BatchingTracer::new(context.clone(), ArrayProgramBatch::replicated(replicated.clone())),
        ];
        let [predicate] = context
            .bind(
                ArrayProgramOperation::Array(ArrayOperation::Compare(CompareOperation::new(
                    ComparisonDirection::GreaterThan,
                ))),
                Vec::new(),
                &inputs,
            )?
            .try_into()
            .unwrap();
        assert_eq!(predicate.batch().batch_axis(), BatchAxis::new(0));
        let false_value = BatchingTracer::new(context.clone(), ArrayProgramBatch::new(mapped, BatchAxis::new(0))?);
        let true_value = BatchingTracer::new(context.clone(), ArrayProgramBatch::replicated(replicated));
        let [selected] = context
            .bind(
                ArrayProgramOperation::Array(ArrayOperation::Select(SelectOperation)),
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
                .filter(|instruction| matches!(instruction.operation(), ArrayProgramOperation::Broadcast(_)))
                .count(),
            2,
        );

        Ok(())
    }

    #[test]
    fn test_dynamic_array_program_shape_changing_alignment() -> Result<(), ProgramError> {
        type TraceContext = TracingContext<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>;

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
        let result_extent = trace.constant(ArrayProgramValue::Dimension(DimensionValue::constant(4)?));
        let batch_extent_id = batch_extent.atom_id().unwrap();
        let replicated_id = replicated.atom_id().unwrap();
        let result_extent_id = result_extent.atom_id().unwrap();
        let context = BatchingContext::<_, ArrayProgramBatching>::new(trace.clone(), batch_extent);
        let inputs = [
            BatchingTracer::new(context.clone(), ArrayProgramBatch::new(mapped, BatchAxis::new(0))?),
            BatchingTracer::new(context.clone(), ArrayProgramBatch::replicated(replicated)),
            BatchingTracer::new(context.clone(), ArrayProgramBatch::replicated(result_extent)),
        ];
        let [output] = context
            .bind(ArrayProgramOperation::from(ConcatenateOperation::new(0, 1)?), Vec::new(), &inputs)?
            .try_into()
            .unwrap();
        assert_eq!(output.batch().batch_axis(), BatchAxis::new(0));
        let builder = trace.builder().borrow();
        let broadcast = builder
            .instructions()
            .iter()
            .find(|instruction| matches!(instruction.operation(), ArrayProgramOperation::Broadcast(_)))
            .expect("expected dynamic operand alignment");
        assert!(matches!(broadcast.operation(), ArrayProgramOperation::Broadcast(_)));
        assert_eq!(broadcast.inputs()[0], replicated_id);
        assert_eq!(broadcast.inputs()[1], batch_extent_id);
        let concatenate = builder.instructions().last().unwrap();
        assert!(matches!(concatenate.operation(), ArrayProgramOperation::Concatenate(_)));
        assert_eq!(concatenate.inputs().last(), Some(&result_extent_id));
        drop(builder);

        // A mapped padding scalar forces the replicated operand through the same dynamic alignment path. Every pad
        // in the mask decomposition consumes the mapped extent explicitly.
        let trace = TraceContext::new();
        let batch_extent = trace.input(DimensionType::new(batch.clone()).into());
        let operand = trace.input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)])).into());
        let padding =
            trace.input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(batch.clone())])).into());
        let result_extent = trace.constant(ArrayProgramValue::Dimension(DimensionValue::constant(3)?));
        let batch_extent_id = batch_extent.atom_id().unwrap();
        let operand_id = operand.atom_id().unwrap();
        let context = BatchingContext::<_, ArrayProgramBatching>::new(trace.clone(), batch_extent);
        let inputs = [
            BatchingTracer::new(context.clone(), ArrayProgramBatch::replicated(operand)),
            BatchingTracer::new(context.clone(), ArrayProgramBatch::new(padding, BatchAxis::new(0))?),
            BatchingTracer::new(context.clone(), ArrayProgramBatch::replicated(result_extent)),
        ];
        let [output] = context
            .bind(ArrayProgramOperation::from(PadOperation::new(vec![1], vec![0], vec![0])?), Vec::new(), &inputs)?
            .try_into()
            .unwrap();
        assert_eq!(output.batch().batch_axis(), BatchAxis::new(0));
        let builder = trace.builder().borrow();
        let broadcast = builder
            .instructions()
            .iter()
            .find(|instruction| matches!(instruction.operation(), ArrayProgramOperation::Broadcast(_)))
            .expect("expected dynamic operand alignment");
        assert_eq!(broadcast.inputs()[0], operand_id);
        assert_eq!(broadcast.inputs()[1], batch_extent_id);
        assert!(
            builder
                .instructions()
                .iter()
                .filter(|instruction| matches!(instruction.operation(), ArrayProgramOperation::Pad(_)))
                .all(|instruction| instruction.inputs().contains(&batch_extent_id)),
        );

        Ok(())
    }

    #[test]
    fn test_array_program_batching() {
        type Parent = EagerContext<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>;
        fn assert_batchable<C: Context<Type = ArrayProgramType>, O: BatchableOperation<C, ArrayProgramBatching>>() {}
        assert_batchable::<Parent, ArrayProgramOperation<Array>>();

        let dimension_type =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::new(0, Some(9)).unwrap()));
        let dimension = ArrayProgramValue::<Array>::Dimension(DimensionValue::new(dimension_type.clone(), 4).unwrap());
        assert_eq!(
            ArrayProgramBatch::new(dimension.clone(), BatchAxis::new(0)),
            Err(BatchingError::MappedDimension { r#type: Box::new(dimension_type.clone()), axis: BatchAxis::new(0) }),
        );
        let negative_axis_batch = ArrayProgramBatch::new(
            ArrayProgramValue::Array(Array::matrix(2, 3, vec![0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0])),
            BatchAxis::new(-2),
        )
        .unwrap();
        assert_eq!(negative_axis_batch.batch_axis(), BatchAxis::new(0));
        assert_eq!(
            negative_axis_batch.unbatched_type(),
            &ArrayProgramType::Array(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)]),)),
        );

        let context = BatchingContext::<_, ArrayProgramBatching>::new(
            Parent::new(),
            ArrayProgramValue::Dimension(DimensionValue::constant(2).unwrap()),
        )
        .with_axis_name("items".to_string())
        .with_axis_sharding(ShardingDimension::Unconstrained);
        assert_eq!(context.axis_name(), Some("items"));
        assert_eq!(context.axis_sharding(), &ShardingDimension::Unconstrained);

        let extent_value = DimensionValue::constant(3).unwrap();
        let dynamic_zero = ArrayProgramOperation::<Array>::from(ZeroOperation::new(ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Dynamic(extent_value.r#type().variable().clone())]),
        )));
        let extent = ArrayProgramValue::Dimension(extent_value);
        let dynamic_zero_output =
            dynamic_zero.batch(&context, &EmptyRegionDriver, &[ArrayProgramBatch::replicated(extent)]).unwrap();
        assert_eq!(dynamic_zero_output.len(), 1);
        assert_eq!(dynamic_zero_output[0].batch_axis(), BatchAxis::replicated());
        assert_eq!(dynamic_zero_output[0].value(), &ArrayProgramValue::Array(Array::vector(vec![0.0_f32, 0.0, 0.0])),);

        let extent_value = DimensionValue::constant(3).unwrap();
        let dynamic_one = ArrayProgramOperation::<Array>::from(OneOperation::new(ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Dynamic(extent_value.r#type().variable().clone())]),
        )));
        let dynamic_one_output = dynamic_one
            .batch(
                &context,
                &EmptyRegionDriver,
                &[ArrayProgramBatch::replicated(ArrayProgramValue::Dimension(extent_value))],
            )
            .unwrap();
        assert_eq!(dynamic_one_output.len(), 1);
        assert_eq!(dynamic_one_output[0].batch_axis(), BatchAxis::replicated());
        assert_eq!(dynamic_one_output[0].value(), &ArrayProgramValue::Array(Array::vector(vec![1.0_f32, 1.0, 1.0])),);

        let extent_value = DimensionValue::constant(3).unwrap();
        let dynamic_iota = ArrayProgramOperation::<Array>::from(
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
            .batch(
                &context,
                &EmptyRegionDriver,
                &[ArrayProgramBatch::replicated(ArrayProgramValue::Dimension(extent_value))],
            )
            .unwrap();
        assert_eq!(dynamic_iota_output.len(), 1);
        assert_eq!(dynamic_iota_output[0].batch_axis(), BatchAxis::replicated());
        assert_eq!(
            dynamic_iota_output[0].value(),
            &ArrayProgramValue::Array(
                Array::new(
                    ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Static(3)])),
                    vec![Scalar::I32(0), Scalar::I32(1), Scalar::I32(2)],
                )
                .unwrap(),
            ),
        );

        let mapped_type =
            DimensionType::new(DimensionVariable::new("mapped_extent", DimensionBounds::new(1, Some(5)).unwrap()));
        let mapped_extent = ArrayProgramBatch {
            value: ArrayProgramValue::Dimension(DimensionValue::new(mapped_type.clone(), 3).unwrap()),
            batch_axis: BatchAxis::new(0),
            r#type: mapped_type.clone().into(),
        };
        assert_eq!(
            dynamic_zero.batch(&context, &EmptyRegionDriver, &[mapped_extent.clone()]),
            Err(BatchingError::MappedDimension { r#type: Box::new(mapped_type.clone()), axis: BatchAxis::new(0) }),
        );
        assert_eq!(
            dynamic_one.batch(&context, &EmptyRegionDriver, &[mapped_extent.clone()]),
            Err(BatchingError::MappedDimension { r#type: Box::new(mapped_type.clone()), axis: BatchAxis::new(0) }),
        );
        assert_eq!(
            dynamic_iota.batch(&context, &EmptyRegionDriver, &[mapped_extent]),
            Err(BatchingError::MappedDimension { r#type: Box::new(mapped_type), axis: BatchAxis::new(0) }),
        );

        // The composite boundary forwards the mapped-axis name into homogeneous array rules, allowing the matching
        // collective to consume the mapped axis instead of incorrectly forwarding an unbound collective.
        let collective_input =
            ArrayProgramBatch::new(ArrayProgramValue::Array(Array::vector(vec![1.0_f32, 2.0])), BatchAxis::new(0))
                .unwrap();
        let collective = ArrayProgramOperation::<Array>::from(ArrayOperation::Collective(CollectiveOperation::new(
            "items".to_string(),
            CollectiveKind::PSum,
        )));
        let collective_output = collective.batch(&context, &EmptyRegionDriver, &[collective_input]).unwrap();
        assert_eq!(collective_output.len(), 1);
        assert_eq!(collective_output[0].batch_axis(), BatchAxis::replicated());
        assert_eq!(collective_output[0].value(), &ArrayProgramValue::Array(Array::scalar(3.0_f32)));

        let all_gather = ArrayProgramOperation::<Array>::from(AllGatherOperation::new(
            "items".to_string(),
            2,
            0,
            CollectiveOptions::tiled(),
            AllGatherOutputVariance::Varying,
        ));
        let all_gather_input = ArrayProgramBatch::new(
            ArrayProgramValue::Array(Array::matrix(2, 2, vec![1.0_f32, 2.0, 3.0, 4.0])),
            BatchAxis::new(0),
        )
        .unwrap();
        let all_gather_extent =
            ArrayProgramBatch::replicated(ArrayProgramValue::Dimension(DimensionValue::constant(4).unwrap()));
        let all_gather_output =
            all_gather.batch(&context, &EmptyRegionDriver, &[all_gather_input, all_gather_extent]).unwrap();
        assert_eq!(all_gather_output.len(), 1);
        assert_eq!(all_gather_output[0].batch_axis(), BatchAxis::replicated());
        assert_eq!(
            all_gather_output[0].value(),
            &ArrayProgramValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0, 4.0])),
        );

        let psum_scatter = ArrayProgramOperation::<Array>::from(PSumScatterOperation::new(
            "items".to_string(),
            2,
            0,
            CollectiveOptions::tiled(),
        ));
        let psum_scatter_input = ArrayProgramBatch::new(
            ArrayProgramValue::Array(Array::matrix(2, 4, vec![1.0_f32, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0])),
            BatchAxis::new(0),
        )
        .unwrap();
        let psum_scatter_extent =
            ArrayProgramBatch::replicated(ArrayProgramValue::Dimension(DimensionValue::constant(2).unwrap()));
        let psum_scatter_output = psum_scatter
            .batch(&context, &EmptyRegionDriver, &[psum_scatter_input, psum_scatter_extent])
            .unwrap();
        assert_eq!(psum_scatter_output.len(), 1);
        assert_eq!(psum_scatter_output[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(
            psum_scatter_output[0].value(),
            &ArrayProgramValue::Array(Array::matrix(2, 2, vec![11.0_f32, 22.0, 33.0, 44.0])),
        );

        let all_to_all = ArrayProgramOperation::<Array>::from(AllToAllOperation::new(
            "items".to_string(),
            2,
            0,
            0,
            CollectiveOptions::tiled(),
        ));
        let all_to_all_input = ArrayProgramBatch::new(
            ArrayProgramValue::Array(Array::matrix(2, 4, vec![1.0_f32, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0])),
            BatchAxis::new(0),
        )
        .unwrap();
        let all_to_all_output = all_to_all.batch(&context, &EmptyRegionDriver, &[all_to_all_input]).unwrap();
        assert_eq!(all_to_all_output.len(), 1);
        assert_eq!(all_to_all_output[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(
            all_to_all_output[0].value(),
            &ArrayProgramValue::Array(Array::matrix(2, 4, vec![1.0_f32, 2.0, 10.0, 20.0, 3.0, 4.0, 30.0, 40.0])),
        );

        // Every rule that consumes a first-class dimension preserves the same typed mapped-dimension diagnostic,
        // even if a malformed internal batch bypasses the public constructor's equivalent boundary check.
        let mapped_dimension = ArrayProgramBatch {
            value: dimension.clone(),
            batch_axis: BatchAxis::new(0),
            r#type: ArrayProgramType::Dimension(dimension_type.clone()),
        };
        let dimension_to_scalar = ArrayProgramOperation::<Array>::from(DimensionToScalarOperation);
        assert_eq!(
            dimension_to_scalar.batch(&context, &EmptyRegionDriver, std::slice::from_ref(&mapped_dimension)),
            Err(BatchingError::MappedDimension { r#type: Box::new(dimension_type.clone()), axis: BatchAxis::new(0) }),
        );
        let comparison = ArrayProgramOperation::<Array>::from(CompareOperation::new(ComparisonDirection::LessThan));
        let comparison_right = ArrayProgramValue::Dimension(DimensionValue::new(dimension_type.clone(), 5).unwrap());
        assert_eq!(
            comparison.batch(
                &context,
                &EmptyRegionDriver,
                &[
                    ArrayProgramBatch::replicated(dimension.clone()),
                    ArrayProgramBatch::replicated(comparison_right.clone()),
                ],
            ),
            Ok(vec![ArrayProgramBatch::replicated(ArrayProgramValue::Array(Array::scalar(true)))]),
        );
        assert_eq!(
            comparison.batch(
                &context,
                &EmptyRegionDriver,
                &[mapped_dimension.clone(), ArrayProgramBatch::replicated(comparison_right.clone())],
            ),
            Err(BatchingError::MappedDimension { r#type: Box::new(dimension_type.clone()), axis: BatchAxis::new(0) }),
        );
        let mapped_comparison_right = ArrayProgramBatch {
            value: comparison_right,
            batch_axis: BatchAxis::new(0),
            r#type: ArrayProgramType::Dimension(dimension_type.clone()),
        };
        assert_eq!(
            comparison.batch(
                &context,
                &EmptyRegionDriver,
                &[ArrayProgramBatch::replicated(dimension.clone()), mapped_comparison_right],
            ),
            Err(BatchingError::MappedDimension { r#type: Box::new(dimension_type.clone()), axis: BatchAxis::new(0) }),
        );
        let dimension_add = ArrayProgramOperation::<Array>::from(DimensionOperation::Add(
            DimensionAddOperation::new(&dimension_type, &dimension_type).unwrap(),
        ));
        assert_eq!(
            dimension_add.batch(
                &context,
                &EmptyRegionDriver,
                &[mapped_dimension, ArrayProgramBatch::replicated(dimension.clone())],
            ),
            Err(BatchingError::MappedDimension { r#type: Box::new(dimension_type.clone()), axis: BatchAxis::new(0) }),
        );

        let gateway_variable = DimensionVariable::new("gateway", DimensionBounds::new(0, Some(9)).unwrap());
        let gateway_operation =
            ArrayProgramOperation::<Array>::from(DimensionFromScalarOperation::new(gateway_variable.clone()));
        let gateway_output = gateway_operation
            .batch(
                &context,
                &EmptyRegionDriver,
                &[ArrayProgramBatch::replicated(ArrayProgramValue::Array(Array::scalar(4_i32)))],
            )
            .unwrap();
        let [gateway_output] = gateway_output.as_slice() else {
            panic!("expected one dimension-from-scalar batching result");
        };
        assert_eq!(gateway_output.batch_axis(), BatchAxis::replicated());
        assert_eq!(
            gateway_output.value(),
            &ArrayProgramValue::Dimension(
                DimensionValue::new(DimensionType::new(gateway_variable.clone()), 4).unwrap(),
            ),
        );
        let mapped_gateway_input =
            ArrayProgramBatch::new(ArrayProgramValue::Array(Array::vector(vec![4_i32, 5_i32])), BatchAxis::new(0))
                .unwrap();
        assert_eq!(
            gateway_operation.batch(&context, &EmptyRegionDriver, &[mapped_gateway_input]),
            Err(BatchingError::MappedDimension {
                r#type: Box::new(DimensionType::new(gateway_variable.clone())),
                axis: BatchAxis::new(0),
            }),
        );
        assert_eq!(
            gateway_operation.batch(&context, &EmptyRegionDriver, &[]),
            Err(BatchingError::from(ProgramError::InvalidInputCount { expected: 1, actual: 0 })),
        );

        let zero = ArrayProgramOperation::<Array>::from(ZeroOperation::new(ArrayProgramType::Array(
            ArrayType::scalar(DataType::F32),
        )));
        assert_eq!(
            zero.batch(
                &context,
                &EmptyRegionDriver,
                &[ArrayProgramBatch::replicated(ArrayProgramValue::Array(Array::scalar(1.0_f32)))],
            ),
            Err(BatchingError::from(ProgramError::InvalidInputCount { expected: 0, actual: 1 })),
        );

        let reshape = ArrayProgramOperation::<Array>::from(ReshapeOperation::new());
        let reshape_input = ArrayProgramBatch::new(
            ArrayProgramValue::Array(Array::matrix(2, 6, (0..12).map(|value| value as f32).collect())),
            BatchAxis::new(0),
        )
        .unwrap();
        let first_extent = ArrayProgramValue::Dimension(DimensionValue::constant(2).unwrap());
        let first_extent_type = first_extent.r#type().into_owned();
        let second_extent = ArrayProgramValue::Dimension(DimensionValue::constant(3).unwrap());
        let reshape_output = reshape
            .batch(
                &context,
                &EmptyRegionDriver,
                &[
                    reshape_input,
                    ArrayProgramBatch::replicated(first_extent.clone()),
                    ArrayProgramBatch::replicated(second_extent.clone()),
                ],
            )
            .unwrap();
        assert_eq!(reshape_output.len(), 1);
        assert_eq!(reshape_output[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(
            reshape_output[0].value(),
            &ArrayProgramValue::Array(Array::from_f64s(
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
                    ArrayProgramBatch::replicated(ArrayProgramValue::Array(Array::vector(vec![
                        0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0,
                    ]))),
                    ArrayProgramBatch {
                        value: first_extent,
                        batch_axis: BatchAxis::new(0),
                        r#type: first_extent_type.clone(),
                    },
                    ArrayProgramBatch::replicated(second_extent),
                ],
            ),
            Err(BatchingError::MappedDimension {
                r#type: Box::new(<&DimensionType>::try_from(&first_extent_type).unwrap().clone()),
                axis: BatchAxis::new(0),
            }),
        );

        let broadcast = ArrayProgramOperation::<Array>::from(BroadcastOperation::new(vec![1]));
        let broadcast_input = ArrayProgramBatch::new(
            ArrayProgramValue::Array(Array::matrix(2, 1, vec![1.0_f32, 2.0])),
            BatchAxis::new(0),
        )
        .unwrap();
        let broadcast_output = broadcast
            .batch(
                &context,
                &EmptyRegionDriver,
                &[
                    broadcast_input,
                    ArrayProgramBatch::replicated(ArrayProgramValue::Dimension(DimensionValue::constant(3).unwrap())),
                    ArrayProgramBatch::replicated(ArrayProgramValue::Dimension(DimensionValue::constant(1).unwrap())),
                ],
            )
            .unwrap();
        assert_eq!(broadcast_output.len(), 1);
        assert_eq!(broadcast_output[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(
            broadcast_output[0].value(),
            &ArrayProgramValue::Array(Array::from_f64s(
                ArrayType::new(
                    DataType::F32,
                    Shape::new(vec![Dimension::Static(2), Dimension::Static(3), Dimension::Static(1),]),
                ),
                vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
            )),
        );

        let mapped_broadcast_extent = ArrayProgramValue::Dimension(DimensionValue::constant(3).unwrap());
        let mapped_broadcast_extent_type = mapped_broadcast_extent.r#type().into_owned();
        assert_eq!(
            broadcast.batch(
                &context,
                &EmptyRegionDriver,
                &[
                    ArrayProgramBatch::replicated(ArrayProgramValue::Array(Array::vector(vec![1.0_f32]))),
                    ArrayProgramBatch {
                        value: mapped_broadcast_extent,
                        batch_axis: BatchAxis::new(0),
                        r#type: mapped_broadcast_extent_type.clone(),
                    },
                    ArrayProgramBatch::replicated(ArrayProgramValue::Dimension(DimensionValue::constant(1).unwrap(),)),
                ],
            ),
            Err(BatchingError::MappedDimension {
                r#type: Box::new(<&DimensionType>::try_from(&mapped_broadcast_extent_type).unwrap().clone()),
                axis: BatchAxis::new(0),
            }),
        );

        // A mapped padding value is decomposed into zero-padding, a padding-position mask, a broadcast of the
        // per-item scalar, and a select. Every shape-changing instruction in that decomposition receives the same
        // explicit output extents, including the inserted physical batch extent.
        let pad = ArrayProgramOperation::<Array>::from(PadOperation::new(vec![1], vec![0], vec![0]).unwrap());
        let pad_output = pad
            .batch(
                &context,
                &EmptyRegionDriver,
                &[
                    ArrayProgramBatch::new(
                        ArrayProgramValue::Array(Array::matrix(2, 2, vec![1.0_f32, 2.0, 3.0, 4.0])),
                        BatchAxis::new(0),
                    )
                    .unwrap(),
                    ArrayProgramBatch::new(
                        ArrayProgramValue::Array(Array::vector(vec![8.0_f32, 9.0])),
                        BatchAxis::new(0),
                    )
                    .unwrap(),
                    ArrayProgramBatch::replicated(ArrayProgramValue::Dimension(DimensionValue::constant(3).unwrap())),
                ],
            )
            .unwrap();
        assert_eq!(
            pad_output,
            vec![
                ArrayProgramBatch::new(
                    ArrayProgramValue::Array(Array::matrix(2, 3, vec![8.0_f32, 1.0, 2.0, 9.0, 3.0, 4.0],)),
                    BatchAxis::new(0),
                )
                .unwrap()
            ],
        );

        // Mapped RNG state batching is scan-based. Composite region-carrying operations deliberately arrive in
        // Phase 5, so the current boundary rejects mapped states without attempting to project a region through
        // `ProjectedContext`.
        let states = Array::new(
            ArrayType::new(DataType::U64, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)])),
            vec![Scalar::U64(1), Scalar::U64(0), Scalar::U64(2), Scalar::U64(0)],
        )
        .unwrap();
        let state_batch = ArrayProgramBatch::new(ArrayProgramValue::Array(states.clone()), BatchAxis::new(0)).unwrap();
        let static_rng = ArrayProgramOperation::<Array>::from(RngBitGeneratorOperation::new(
            RandomAlgorithm::ThreeFry,
            ArrayType::new(DataType::U32, Shape::new(vec![Dimension::Static(2)])),
        ));
        assert_eq!(
            static_rng.batch(&context, &EmptyRegionDriver, std::slice::from_ref(&state_batch)),
            Err(BatchingError::UnsupportedOperation {
                message: "'rng_bit_generator' batching requires Phase 5 composite scan-region support".to_string(),
            }),
        );

        let dynamic_rng_extent = DimensionVariable::new("rng_count", DimensionBounds::new(1, Some(5)).unwrap());
        let dynamic_rng = ArrayProgramOperation::<Array>::from(RngBitGeneratorOperation::new(
            RandomAlgorithm::ThreeFry,
            ArrayType::new(DataType::U32, Shape::new(vec![Dimension::Dynamic(dynamic_rng_extent.clone())])),
        ));
        assert_eq!(
            dynamic_rng.batch(
                &context,
                &EmptyRegionDriver,
                &[
                    state_batch,
                    ArrayProgramBatch::replicated(ArrayProgramValue::Dimension(
                        DimensionValue::new(DimensionType::new(dynamic_rng_extent), 2).unwrap(),
                    )),
                ],
            ),
            Err(BatchingError::UnsupportedOperation {
                message: "'rng_bit_generator' batching requires Phase 5 composite scan-region support".to_string(),
            }),
        );
        assert_eq!(
            static_rng.batch(
                &context,
                &EmptyRegionDriver,
                &[ArrayProgramBatch::replicated(ArrayProgramValue::Array(states))],
            ),
            Err(BatchingError::UnsupportedOperation {
                message: "'rng_bit_generator' cannot batch a replicated state because every batch item would see \
                          the same state; derive one state per batch item with `split_key` and map over the states \
                          explicitly"
                    .to_string(),
            }),
        );

        // Concatenate aligns mapped array operands before shifting the logical concatenation axis around the common
        // physical batch axis. Its trailing extent remains a replicated shape value.
        let concatenate = ArrayProgramOperation::<Array>::from(ConcatenateOperation::new(0, 1).unwrap());
        let concatenate_extent = ArrayProgramValue::Dimension(DimensionValue::constant(3).unwrap());
        let concatenate_output = concatenate
            .batch(
                &context,
                &EmptyRegionDriver,
                &[
                    ArrayProgramBatch::new(
                        ArrayProgramValue::Array(Array::matrix(2, 2, vec![1.0_f32, 3.0, 2.0, 4.0])),
                        BatchAxis::new(1),
                    )
                    .unwrap(),
                    ArrayProgramBatch::new(
                        ArrayProgramValue::Array(Array::matrix(2, 1, vec![5.0_f32, 6.0])),
                        BatchAxis::new(0),
                    )
                    .unwrap(),
                    ArrayProgramBatch::replicated(concatenate_extent.clone()),
                ],
            )
            .unwrap();
        assert_eq!(
            concatenate_output,
            vec![
                ArrayProgramBatch::new(
                    ArrayProgramValue::Array(Array::matrix(3, 2, vec![1.0_f32, 3.0, 2.0, 4.0, 5.0, 6.0])),
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
                        ArrayProgramBatch::new(
                            ArrayProgramValue::Array(Array::matrix(2, 2, vec![1.0_f32, 2.0, 3.0, 4.0])),
                            BatchAxis::new(0),
                        )
                        .unwrap(),
                        ArrayProgramBatch::replicated(ArrayProgramValue::Array(Array::vector(vec![5.0_f32]))),
                        ArrayProgramBatch::replicated(concatenate_extent.clone()),
                    ],
                )
                .unwrap(),
            vec![
                ArrayProgramBatch::new(
                    ArrayProgramValue::Array(Array::matrix(2, 3, vec![1.0_f32, 2.0, 5.0, 3.0, 4.0, 5.0],)),
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
                    ArrayProgramBatch::replicated(ArrayProgramValue::Array(Array::vector(vec![1.0_f32, 2.0]))),
                    ArrayProgramBatch::replicated(ArrayProgramValue::Array(Array::vector(vec![3.0_f32]))),
                    ArrayProgramBatch {
                        value: concatenate_extent,
                        batch_axis: BatchAxis::new(0),
                        r#type: concatenate_extent_type.clone(),
                    },
                ],
            ),
            Err(BatchingError::MappedDimension {
                r#type: Box::new(<&DimensionType>::try_from(&concatenate_extent_type).unwrap().clone()),
                axis: BatchAxis::new(0),
            }),
        );

        let dimension = BatchingTracer::new(context.clone(), ArrayProgramBatch::replicated(dimension));
        let scalar = dimension.to_scalar().unwrap().into_batch();
        assert_eq!(scalar.batch_axis(), BatchAxis::replicated());
        assert_eq!(scalar.into_value(), ArrayProgramValue::Array(Array::scalar(4_i64)));

        let scalar = BatchingTracer::new(
            context.clone(),
            ArrayProgramBatch::replicated(ArrayProgramValue::Array(Array::scalar(4_i32))),
        );
        let dimension = scalar.to_dimension(gateway_variable).unwrap().into_batch();
        assert_eq!(dimension.batch_axis(), BatchAxis::replicated());
        assert!(matches!(dimension.into_value(), ArrayProgramValue::Dimension(value) if value.extent() == 4));

        let array = ArrayProgramValue::Array(Array::matrix(2, 3, vec![0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0]));
        let array = ArrayProgramBatch::new(array, BatchAxis::new(0)).unwrap();
        let array = BatchingTracer::new(context, array);
        let scalar = array.dimension_size(0).unwrap().to_scalar().unwrap().into_batch();
        assert_eq!(scalar.batch_axis(), BatchAxis::replicated());
        assert_eq!(scalar.into_value(), ArrayProgramValue::Array(Array::scalar(3_i64)));

        type TraceContext = TracingContext<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>;
        let trace = TraceContext::new();
        let input = trace.input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)])).into());
        let output = input.dimension_size(0).unwrap().to_scalar().unwrap();
        let program = trace
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
                vec![output.atom_id().unwrap()],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let context = BatchingContext::<_, ArrayProgramBatching>::new(
            Parent::new(),
            ArrayProgramValue::Dimension(DimensionValue::constant(2).unwrap()),
        );
        let input = BatchingTracer::new(
            context.clone(),
            ArrayProgramBatch::new(
                ArrayProgramValue::Array(Array::matrix(2, 3, vec![0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0])),
                BatchAxis::new(0),
            )
            .unwrap(),
        );
        let outputs = program.interpret_in_context(&context, vec![input]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch().batch_axis(), BatchAxis::replicated());
        assert_eq!(outputs[0].batch().value(), &ArrayProgramValue::Array(Array::scalar(3_i64)));
    }
}
