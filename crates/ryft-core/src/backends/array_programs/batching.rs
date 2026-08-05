//! Batching support for programs that mix arrays with first-class dimensions.
//!
//! Arrays retain the ordinary [`ArrayBatch`] representation and existing array batching rules. First-class
//! dimensions are shared shape values and therefore remain replicated across the batch. Mixed operations
//! explicitly state how they cross that boundary.

use std::borrow::Cow;
use std::fmt::Display;
use std::rc::Rc;

use ryft_macros::Parameter;

use crate::axes::{Axis, NamedAxes, NamedAxis};
use crate::backends::dimensions::{DimensionOperation, DimensionValue};
use crate::batching::arrays::{batch_axis_sharding, normalized_batch_axis_type};
use crate::batching::{
    ArrayBatch, ArrayBatching, ArrayBatchingPolicy, BatchAxis, BatchAxisSpecification, BatchableOperation,
    BatchableType, BatchedProgram, BatchingContext, BatchingDriver, BatchingEntrypointPolicy, BatchingError,
    BatchingPolicy, BatchingPolicyProjection, BatchingTracer, BoundaryPreservingBatchedProgram, DimensionSource,
    MemberBatchableOperation, ProgramBatchingOutputAxesPolicy, RecursiveBatchingDriver, RecursiveBatchingPolicy,
};
use crate::contexts::{Context, ProjectedContext, StagingContext, ValueResolution};
use crate::macros::{check_builders, check_count};
use crate::operations::collectives::CollectiveBatchingPolicy;
use crate::operations::constants::{ConstantOperation, IotaOperation, OneOperation, ZeroOperation};
use crate::operations::dimensions::{DimensionRequirement, DimensionRequirementOperation, DimensionSizeOperation};
use crate::operations::manipulation::{BroadcastOperation, ReshapeOperation, Transpose, TransposeOperation};
use crate::operations::math::{Div, Mul};
use crate::parameters::{Parameter, Placeholder};
use crate::programs::operations::{Operation, OperationProjection};
use crate::programs::regions::{RegionRef, RegionReplayMappings, ReplayRegionDriver};
use crate::programs::types::{Type, TypeError, Typed};
use crate::programs::values::{ProjectedValue, Value, ValueProjection};
use crate::programs::{Program, ProgramError};
use crate::sharding::Sharding;
use crate::tracing::{Tracer, TracingContext};
use crate::types::{ArrayIrType, ArrayType, Dimension, DimensionType};

/// Kind-aware batched view of one array IR value.
#[derive(Clone, Debug, Parameter)]
pub struct ArrayIrBatch<V: Value<Type = ArrayIrType>> {
    /// Packed parent value.
    value: V,

    /// Mapped packed array axis, or replicated for array and dimension values shared across the batch.
    batch_axis: BatchAxis,

    /// Unbatched per-item type reported to the transformed program.
    r#type: ArrayIrType,
}

impl<V: Value<Type = ArrayIrType>> ArrayIrBatch<V> {
    /// Creates a batch view and rejects mapped first-class dimensions.
    pub fn new(value: V, batch_axis: BatchAxis) -> Result<Self, BatchingError> {
        let (r#type, batch_axis) = match value.r#type().as_ref() {
            ArrayIrType::Array(r#type) => {
                let (r#type, batch_axis) = r#type.unbatched_type_and_axis(batch_axis)?;
                (ArrayIrType::Array(r#type), batch_axis)
            }
            ArrayIrType::Dimension(r#type) if batch_axis.is_replicated() => {
                (ArrayIrType::Dimension(r#type.clone()), batch_axis)
            }
            ArrayIrType::Dimension(r#type) => {
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

    /// Returns the unbatched per-item array IR type.
    pub fn unbatched_type(&self) -> &ArrayIrType {
        &self.r#type
    }

    /// Validates that this batch contains a replicated first-class dimension.
    pub(crate) fn validate_replicated_dimension(&self) -> Result<(), BatchingError> {
        let r#type = <&DimensionType>::try_from(&self.r#type)?;
        if self.batch_axis.is_replicated() {
            Ok(())
        } else {
            Err(BatchingError::MappedDimension { r#type: Box::new(r#type.clone()), axis: self.batch_axis })
        }
    }
}

impl<V: Value<Type = ArrayIrType> + PartialEq> PartialEq for ArrayIrBatch<V> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value && self.batch_axis == other.batch_axis
    }
}

impl<V: Value<Type = ArrayIrType>> Typed for ArrayIrBatch<V> {
    type Type = ArrayIrType;

    #[inline]
    fn r#type(&self) -> Cow<'_, ArrayIrType> {
        Cow::Borrowed(self.unbatched_type())
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

    /// The adapted program's leading input still defines the [`DimensionVariable`](crate::DimensionVariable)
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
    type BatchedProgram = BoundaryPreservingBatchedProgram<
        <C::Constant as ValueProjection<ArrayType>>::Projected,
        <C::Operation as OperationProjection<ArrayType>>::Projected,
    >;

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
/// mapped extent so [`batch_projected_operation`] can construct one uniform projected batching context for every
/// member kind without specializing that extent.
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
{
    type Projected = ArrayBatching<DynamicArrayBatchingPolicy>;
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
        + From<BroadcastOperation>
        + From<DimensionOperation<DimensionValue>>
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
        Ok(ProjectedValue::new(self, <&T>::try_from(self.batch().unbatched_type())?.clone()))
    }

    #[inline]
    fn into_projected(self) -> Result<Self::Projected, TypeError> {
        let r#type = <&T>::try_from(self.batch().unbatched_type())?.clone();
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

/// Returns one first-class dimension operand for every packed array axis.
fn array_dimensions<C>(context: &C, value: &C::Value, rank: usize) -> Result<Vec<C::Value>, BatchingError>
where
    C: Context<Type = ArrayIrType, Operation: From<DimensionSizeOperation>>,
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
    C: Context<Type = ArrayIrType, Operation: From<BroadcastOperation>>,
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
            Type = ArrayIrType,
            Operation: From<BroadcastOperation>
                           + From<DimensionOperation<DimensionValue>>
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
                        let value = DimensionValue::constant(extent).map_err(ProgramError::from)?;
                        let mut outputs = outer_context.bind(
                            DimensionOperation::Constant(ConstantOperation::new(value)),
                            Vec::new(),
                            &[],
                        )?;
                        check_count!("output", outputs, 1, ProgramError);
                        Ok(outputs.remove(0))
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

impl<C> CollectiveBatchingPolicy<ProjectedContext<C, ArrayType>> for DynamicArrayBatchingPolicy
where
    C: Context<
            Type = ArrayIrType,
            Operation: From<BroadcastOperation>
                           + From<DimensionOperation<DimensionValue>>
                           + From<DimensionSizeOperation>
                           + From<ReshapeOperation>
                           + OperationProjection<ArrayType>,
        >,
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Value:
        ValueProjection<ArrayType, Projected: Transpose + Value<Type = ArrayType>> + ValueProjection<DimensionType>,
    <C::Value as ValueProjection<DimensionType>>::Projected:
        DimensionRequirement + Div + Mul + Value<Type = DimensionType>,
{
    type ShapeExtent = <C::Value as ValueProjection<DimensionType>>::Projected;

    fn collective_axis_extent(
        context: &BatchingContext<ProjectedContext<C, ArrayType>, ArrayBatching<Self>>,
        _operation_name: &str,
        _axis_name: &str,
        axis_size: usize,
    ) -> Result<Self::ShapeExtent, BatchingError> {
        let axis_extent = <C::Value as ValueProjection<DimensionType>>::into_projected(context.axis_extent().clone())?;
        let axis_size = Self::collective_extent_constant(context, axis_size)?;
        axis_extent.require_equal(&axis_size)?;
        Ok(axis_extent)
    }

    fn collective_extent_constant(
        context: &BatchingContext<ProjectedContext<C, ArrayType>, ArrayBatching<Self>>,
        extent: usize,
    ) -> Result<Self::ShapeExtent, BatchingError> {
        Ok(<C::Value as ValueProjection<DimensionType>>::into_projected(dimension_constant(
            context.parent().parent(),
            extent,
        )?)?)
    }

    fn require_divisible_collective_extents(
        left: &Self::ShapeExtent,
        right: &Self::ShapeExtent,
    ) -> Result<(), BatchingError> {
        left.require_divisible_by(right).map_err(Into::into)
    }

    fn match_collective_axis(
        context: &BatchingContext<ProjectedContext<C, ArrayType>, ArrayBatching<Self>>,
        batch: &ArrayBatch<<C::Value as ValueProjection<ArrayType>>::Projected>,
        input_extents: &[Self::ShapeExtent],
    ) -> Result<ArrayBatch<<C::Value as ValueProjection<ArrayType>>::Projected>, BatchingError> {
        if !batch.batch_axis().is_replicated() {
            return batch.move_axis(0);
        }

        let input_type = batch.unbatched_type();
        let input_extent_dimensions =
            input_extents.iter().map(|extent| extent.r#type().to_dimension()).collect::<Vec<_>>();
        let value = if input_type.shape().dimensions() == input_extent_dimensions {
            batch.value().clone()
        } else {
            Self::reshape_collective(context, batch.value().clone(), input_extents, input_type.sharding().cloned())?
        };
        let output_axes = (1..=input_type.rank()).collect::<Vec<_>>();
        let output_sharding = input_type
            .sharding()
            .map(|sharding| {
                sharding
                    .with_inserted_dimension(0, context.axis_sharding().clone())
                    .map_err(|error| BatchingError::MisalignedBatchAxes { message: error.to_string() })
            })
            .transpose()?;
        let mut output_extents = Vec::with_capacity(input_extents.len() + 1);
        output_extents.push(context.axis_extent().clone());
        output_extents
            .extend(input_extents.iter().cloned().map(<C::Value as ValueProjection<DimensionType>>::from_projected));
        let value = broadcast_array(
            context.parent().parent(),
            <C::Value as ValueProjection<ArrayType>>::from_projected(value),
            output_extents,
            output_axes,
            output_sharding,
        )?;
        let r#type = <&ArrayType>::try_from(value.r#type().as_ref())?.clone();
        ArrayBatch::new(
            r#type,
            <C::Value as ValueProjection<ArrayType>>::into_projected(value)?,
            BatchAxis::from_position(0),
        )
    }

    fn reshape_collective(
        context: &BatchingContext<ProjectedContext<C, ArrayType>, ArrayBatching<Self>>,
        value: <C::Value as ValueProjection<ArrayType>>::Projected,
        output_extents: &[Self::ShapeExtent],
        output_sharding: Option<Sharding>,
    ) -> Result<<C::Value as ValueProjection<ArrayType>>::Projected, BatchingError> {
        let operation = ReshapeOperation::new().with_output_sharding(output_sharding);
        let inputs = std::iter::once(<C::Value as ValueProjection<ArrayType>>::from_projected(value))
            .chain(output_extents.iter().cloned().map(<C::Value as ValueProjection<DimensionType>>::from_projected))
            .collect::<Vec<_>>();
        let mut outputs = context.parent().parent().bind(operation, Vec::new(), inputs.as_slice())?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(<C::Value as ValueProjection<ArrayType>>::into_projected(outputs.remove(0))?)
    }
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
            Operation: From<BroadcastOperation>
                           + From<DimensionOperation<DimensionValue>>
                           + From<DimensionSizeOperation>
                           + OperationProjection<ArrayType>,
        >,
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Value: ValueProjection<ArrayType, Projected: Transpose + Value<Type = ArrayType>>,
{
    let value_type = batch.value.r#type();
    let array_type = match value_type.as_ref() {
        ArrayIrType::Array(r#type) => r#type.clone(),
        ArrayIrType::Dimension(r#type) => {
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
    ArrayIrBatch::new(<C::Value as ValueProjection<ArrayType>>::from_projected(output.into_value()), batch_axis)
}

/// Normalizes one mapped composite array input to the batching context's common packed sharding placement.
fn normalize_array_input<C>(
    context: &BatchingContext<C, ArrayIrBatching>,
    batch: ArrayIrBatch<C::Value>,
) -> Result<ArrayIrBatch<C::Value>, BatchingError>
where
    C: Context<Type = ArrayIrType, Operation: From<BroadcastOperation> + From<DimensionSizeOperation>>,
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
    ArrayIrBatch::new(value, batch_axis)
}

impl<C> BatchingEntrypointPolicy<C> for ArrayIrBatching
where
    C: Context<
            Type = ArrayIrType,
            Operation: From<BroadcastOperation>
                           + From<DimensionOperation<DimensionValue>>
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

impl<C> RecursiveBatchingPolicy<C> for ArrayIrBatching
where
    C: Context<Type = ArrayIrType>,
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Operation: BatchableOperation<C, ArrayIrBatching>
        + BatchableOperation<TracingContext<C::Constant, C::Operation>, ArrayIrBatching>
        + From<BroadcastOperation>
        + From<DimensionOperation<DimensionValue>>
        + From<DimensionSizeOperation>
        + OperationProjection<ArrayType>,
    <C::Operation as OperationProjection<ArrayType>>::Projected: From<TransposeOperation>,
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
                |_, constant| Ok(ArrayIrBatch::replicated(batching_context.parent().lift(constant.clone())?)),
                |instruction, instruction_inputs| {
                    let regions = ReplayRegionDriver::new(region, instruction.regions(), &region_mappings)?;
                    instruction.operation().batch(
                        &batching_context,
                        &RecursiveBatchingDriver::new(&regions),
                        instruction_inputs,
                    )
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
        + From<BroadcastOperation>
        + From<DimensionOperation<DimensionValue>>
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

/// Materializes one exact first-class dimension in a composite context.
fn dimension_constant<C>(context: &C, extent: usize) -> Result<C::Value, BatchingError>
where
    C: Context<Type = ArrayIrType, Operation: From<DimensionOperation<DimensionValue>>>,
{
    let value = DimensionValue::constant(extent).map_err(ProgramError::from)?;
    let mut outputs = context.bind(DimensionOperation::Constant(ConstantOperation::new(value)), Vec::new(), &[])?;
    check_count!("output", outputs, 1, ProgramError);
    Ok(outputs.remove(0))
}

macro_rules! impl_dynamic_constructor_member_batching {
    // Implements the replicated-extent batching rule shared by the three dynamic array constructors. Batching is the
    // one transform the `#[ryft(mixed(structural))]` role does not generate for these payloads, because a
    // mixed signature cannot be projected into one member kind the way a projected structural member is.
    ($operation:ty) => {
        impl<C> MemberBatchableOperation<C, ArrayIrBatching> for $operation
        where
            C: Context<Type = ArrayIrType, Operation: From<$operation>>,
        {
            fn batch_in_parent<D: BatchingDriver<C, ArrayIrBatching>>(
                &self,
                context: &BatchingContext<C, ArrayIrBatching>,
                _driver: &D,
                inputs: &[ArrayIrBatch<C::Value>],
            ) -> Result<Vec<ArrayIrBatch<C::Value>>, BatchingError> {
                // Output extents are shared shape values. A mapped extent would request a different output shape for
                // each batch item, which requires a ragged representation that ordinary array batching lacks.
                for extent in inputs {
                    extent.validate_replicated_dimension()?;
                }
                Ok(context
                    .parent()
                    .bind(
                        self.clone(),
                        Vec::new(),
                        &inputs.iter().map(|input| input.value().clone()).collect::<Vec<_>>(),
                    )?
                    .into_iter()
                    .map(ArrayIrBatch::replicated)
                    .collect())
            }
        }
    };
}

impl_dynamic_constructor_member_batching!(ZeroOperation<ArrayType>);
impl_dynamic_constructor_member_batching!(OneOperation<ArrayType>);
impl_dynamic_constructor_member_batching!(IotaOperation<ArrayType>);

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::backends::array_programs::ArrayIrValue;
    use crate::backends::arrays::Array;
    use crate::backends::dimensions::DimensionValue;
    use crate::batching::{
        Batch, BatchAxisSpecification, BatchingPolicy, BatchingTracer, InterpretableBatchableOperation,
        RecursiveBatchingPolicy, batch,
    };
    use crate::contexts::{EagerContext, StagingContext};
    use crate::differentiation::LinearCallOperation;
    use crate::operations::collectives::{
        AllGatherOperation, AllGatherOutputVariance, AllToAllOperation, CollectiveOptions, PSumScatterOperation,
    };
    use crate::operations::compare::{CompareOperation, ComparisonDirection};
    use crate::operations::constants::{IotaOperation, OneOperation, ZeroOperation};
    use crate::operations::control_flow::{ConditionOperation, ScanOperation, SelectOperation, WhileOperation};
    use crate::operations::dimensions::{
        DimensionAddOperation, DimensionFromScalar, DimensionFromScalarOperation, DimensionSize, DimensionToScalar,
        DimensionToScalarOperation,
    };
    use crate::operations::manipulation::{ConcatenateOperation, PadOperation};
    use crate::operations::math::{AddOperation, NegOperation};
    use crate::operations::random::{RandomAlgorithm, RngBitGeneratorOperation};
    use crate::operations::{CollectiveKind, CollectiveOperation};
    use crate::parameters::Placeholder;
    use crate::programs::ProgramBuilder;
    use crate::programs::regions::EmptyRegionDriver;
    use crate::sharding::ShardingDimension;
    use crate::tracing::TracingContext;
    use crate::types::dimensions::{DimensionBounds, DimensionVariable};
    use crate::types::{DataType, Dimension, Shape};
    use crate::{ArrayIrOperation, ArrayOperation, Scalar};

    use super::*;

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
        // the output broadcast rather than metadata reconstructed from the array type.
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
        assert!(matches!(builder.instructions()[1].operation(), ArrayIrOperation::DimensionSize(_),));
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
            .bind(ArrayIrOperation::<Array>::from(ReshapeOperation::new()), Vec::new(), &inputs)?
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
        let static_extent_type = DimensionValue::constant(2)?.r#type().clone();
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
            &ArrayIrType::Array(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)]),)),
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
        let dynamic_zero_output =
            dynamic_zero.batch(&context, &EmptyRegionDriver, &[ArrayIrBatch::replicated(extent)]).unwrap();
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
            .unwrap();
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
            .unwrap();
        assert_eq!(dynamic_iota_output.len(), 1);
        assert_eq!(dynamic_iota_output[0].batch_axis(), BatchAxis::replicated());
        assert_eq!(
            dynamic_iota_output[0].value(),
            &ArrayIrValue::Array(
                Array::new(
                    ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Static(3)])),
                    vec![Scalar::I32(0), Scalar::I32(1), Scalar::I32(2)],
                )
                .unwrap(),
            ),
        );

        let mapped_type =
            DimensionType::new(DimensionVariable::new("mapped_extent", DimensionBounds::new(1, Some(5)).unwrap()));
        let mapped_extent = ArrayIrBatch {
            value: ArrayIrValue::Dimension(DimensionValue::new(mapped_type.clone(), 3).unwrap()),
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
            ArrayIrBatch::new(ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0])), BatchAxis::new(0)).unwrap();
        let collective = ArrayIrOperation::<Array>::from(ArrayOperation::Collective(CollectiveOperation::new(
            "items".to_string(),
            CollectiveKind::PSum,
        )));
        let collective_output = collective.batch(&context, &EmptyRegionDriver, &[collective_input]).unwrap();
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
        let all_gather_output =
            all_gather.batch(&context, &EmptyRegionDriver, &[all_gather_input, all_gather_extent]).unwrap();
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
            .unwrap();
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
        let all_to_all_output =
            all_to_all.batch(&context, &EmptyRegionDriver, &[all_to_all_input, all_to_all_extent]).unwrap();
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
            .unwrap();
        assert_eq!(gathered[0].batch_axis(), BatchAxis::replicated());
        assert_eq!(gathered[0].value(), &ArrayIrValue::Array(Array::matrix(2, 2, vec![1.0_f32, 3.0, 2.0, 4.0])),);

        let untiled_scatter = ArrayIrOperation::<Array>::from(PSumScatterOperation::new(
            "items".to_string(),
            2,
            0,
            CollectiveOptions::default(),
        ));
        let scattered = untiled_scatter.batch(&context, &EmptyRegionDriver, &[untiled_input()]).unwrap();
        assert_eq!(scattered[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(scattered[0].value(), &ArrayIrValue::Array(Array::vector(vec![4.0_f32, 6.0])));

        let untiled_exchange = ArrayIrOperation::<Array>::from(AllToAllOperation::new(
            "items".to_string(),
            2,
            0,
            0,
            CollectiveOptions::default(),
        ));
        let exchanged = untiled_exchange.batch(&context, &EmptyRegionDriver, &[untiled_input(), extent_two()]).unwrap();
        assert_eq!(exchanged[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(exchanged[0].value(), &ArrayIrValue::Array(Array::matrix(2, 2, vec![1.0_f32, 3.0, 2.0, 4.0])),);

        // Every rule that consumes a first-class dimension preserves the same typed mapped-dimension diagnostic,
        // even if a malformed internal batch bypasses the public constructor's equivalent boundary check.
        let mapped_dimension = ArrayIrBatch {
            value: dimension.clone(),
            batch_axis: BatchAxis::new(0),
            r#type: ArrayIrType::Dimension(dimension_type.clone()),
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
            Ok(vec![ArrayIrBatch::replicated(ArrayIrValue::Array(Array::scalar(true)))]),
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
            r#type: ArrayIrType::Dimension(dimension_type.clone()),
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
            .unwrap();
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
            Err(BatchingError::MappedDimension {
                r#type: Box::new(DimensionType::new(gateway_variable.clone())),
                axis: BatchAxis::new(0),
            }),
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

        let reshape = ArrayIrOperation::<Array>::from(ReshapeOperation::new());
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
            .unwrap();
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
                        r#type: first_extent_type.clone(),
                    },
                    ArrayIrBatch::replicated(second_extent),
                ],
            ),
            Err(BatchingError::MappedDimension {
                r#type: Box::new(<&DimensionType>::try_from(&first_extent_type).unwrap().clone()),
                axis: BatchAxis::new(0),
            }),
        );

        let broadcast = ArrayIrOperation::<Array>::from(BroadcastOperation::new(vec![1]));
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
            .unwrap();
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
                        r#type: mapped_broadcast_extent_type.clone(),
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
            .unwrap();
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
        let states = Array::new(
            ArrayType::new(DataType::U64, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)])),
            vec![Scalar::U64(1), Scalar::U64(0), Scalar::U64(2), Scalar::U64(0)],
        )
        .unwrap();
        let state_batch = ArrayIrBatch::new(ArrayIrValue::Array(states.clone()), BatchAxis::new(0)).unwrap();
        let static_rng = ArrayIrOperation::<Array>::from(RngBitGeneratorOperation::new(
            RandomAlgorithm::ThreeFry,
            ArrayType::new(DataType::U32, Shape::new(vec![Dimension::Static(2)])),
        ));
        let static_outputs =
            static_rng.batch(&context, &EmptyRegionDriver, std::slice::from_ref(&state_batch)).unwrap();
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
            .unwrap();
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
            .unwrap();
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
                .unwrap(),
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
                        r#type: concatenate_extent_type.clone(),
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
}
