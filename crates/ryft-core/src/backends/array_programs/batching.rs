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
use crate::backends::array_programs::{ArrayProgramOperation, ArrayProgramValue};
use crate::backends::arrays::ArrayOperation;
use crate::backends::dimensions::{DimensionOperation, DimensionValue};
use crate::batching::{
    ArrayBatch, ArrayBatching, ArrayBatchingPolicy, BatchAxis, BatchAxisSpecification, BatchableOperation,
    BatchableType, BatchedProgram, BatchingContext, BatchingDriver, BatchingEntrypointPolicy, BatchingError,
    BatchingPolicy, BatchingTracer, DimensionSource, ProgramBatchingOutputAxesPolicy, RecursiveBatchingDriver,
    RecursiveBatchingPolicy, batch_axis_sharding, normalized_batch_axis_type,
};
use crate::contexts::{Context, Domain, ProjectedContext, StagingContext, ValueResolution};
use crate::macros::{check_builders, check_count};
use crate::operations::constants::{ConstantOperation, OneOperation, Zero, ZeroOperation};
use crate::operations::control_flow::scan::scan_iteration_batch_axis;
use crate::operations::control_flow::{ConditionOperation, ScanOperation, Select, SelectOperation, WhileOperation};
use crate::operations::custom_call::CustomCallOperation;
use crate::operations::dimensions::{
    DimensionRequirementOperation, DimensionSizeOperation, DimensionToScalarOperation,
};
use crate::operations::manipulation::reshaping::lift_output_sharding_for_leading_batch_axis;
use crate::operations::manipulation::{
    BroadcastOperation, CONCATENATE_OPERATION_NAME, ConcatenateOperation, LegacyBroadcast, LegacyBroadcastOperation,
    PadOperation, Reshape, ReshapeOperation, Transpose, TransposeOperation,
};
use crate::operations::math::Reduce;
use crate::operations::random::RngBitGeneratorOperation;
use crate::parameters::{Parameter, Placeholder};
use crate::programs::operations::{Operation, OperationProjection};
use crate::programs::regions::{EmptyRegionDriver, RegionRef, RegionReplayMappings, ReplayRegionDriver};
use crate::programs::types::{Type, TypeError, Typed};
use crate::programs::values::{ProjectedValue, ProjectedValueRef, Value, ValueProjection};
use crate::programs::{Program, ProgramBuilder, ProgramError};
use crate::sharding::Sharding;
use crate::tracing::TracingContext;
use crate::types::{ArrayProgramType, ArrayType, DataType, Dimension, DimensionType};

/// Kind-aware batched view of one composite array-program value.
#[derive(Clone, Debug, Parameter)]
pub struct ArrayProgramBatch<V: Value<Type = ArrayProgramType>> {
    /// Packed parent value.
    value: V,

    /// Mapped packed array axis, or replicated for array and dimension values shared across the batch.
    batch_axis: BatchAxis,

    /// Unbatched per-item type reported to the transformed program.
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
    fn batch_axis_position(&self) -> Option<usize> {
        let value_type = self.value.r#type();
        let r#type = <&ArrayType>::try_from(value_type.as_ref()).ok()?;
        self.batch_axis.axis().map(|axis| axis.normalize(r#type.rank()).unwrap())
    }

    /// Returns the unbatched per-item composite type.
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

/// Result of structurally batching a composite array-program [`Region`](crate::Region), whose boundary explicitly
/// threads its mapped extent.
///
/// Composite regions are retraced as standalone programs and therefore cannot capture the parent program's
/// first-class mapped-extent SSA value. Their boundary has one additional leading dimension input and output:
/// the input defines the identity referenced by every inserted dynamic batch dimension, and the output forwards that
/// same atom so enclosing higher-order operations can carry it through the sealed region. Output-axis metadata
/// excludes this protocol output, and [`Self::into_parts`] documents the arity contract consumers must uphold.
pub struct ThreadedExtentBatchedProgram<V: Typed<Type = ArrayProgramType> + Parameter, O> {
    /// Structurally transformed program, including its leading protocol input and output.
    program: Program<V, O, Vec<V>, Vec<V>>,

    /// Mapped axes of the source region's outputs. The protocol-only threaded extent is excluded.
    output_axes: Vec<BatchAxis>,
}

/// Batching policy for programs whose values may be arrays or first-class dimensions.
///
/// Array members may carry a mapped axis. Dimension members are shared shape values and therefore remain
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
    type BatchedProgram = ThreadedExtentBatchedProgram<C::Constant, C::Operation>;

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

impl<V: Value<Type = ArrayProgramType>, O: Operation<ArrayProgramType>> ThreadedExtentBatchedProgram<V, O> {
    /// Creates a structurally batched array program with one leading mapped-extent input and forwarded output.
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

        if !matches!(program.inputs().next().unwrap().r#type().as_ref(), ArrayProgramType::Dimension(_)) {
            return Err(ProgramError::MalformedProgram(
                "a structurally batched program's leading threaded-extent input must be a dimension".to_string(),
            ));
        }
        if !matches!(program.outputs().next().unwrap().r#type().as_ref(), ArrayProgramType::Dimension(_)) {
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

    /// Returns the mapped axes of the source region's outputs, excluding the protocol-only threaded extent.
    #[inline]
    pub fn output_axes(&self) -> &[BatchAxis] {
        self.output_axes.as_slice()
    }

    /// Consumes this result and returns its transformed program and output axes.
    ///
    /// The returned program's boundary is `[extent, packed inputs...] -> [extent, packed outputs...]`:
    /// callers must supply the mapped-extent dimension value as the leading operand and drop the leading forwarded
    /// output, whose axis is deliberately absent from the returned output axes.
    pub fn into_parts(self) -> (Program<V, O, Vec<V>, Vec<V>>, Vec<BatchAxis>) {
        (self.program, self.output_axes)
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
    type BatchedProgram = BatchedProgram<
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
    fn unbatched_type(batch: &Self::Batch) -> Cow<'_, ArrayType> {
        Cow::Owned(batch.unbatched_type())
    }
}

impl<C, T> ValueProjection<T> for BatchingTracer<C, ArrayProgramBatching>
where
    C: Context<Type = ArrayProgramType, Operation: BatchableOperation<C, ArrayProgramBatching>>,
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Operation: BatchableOperation<TracingContext<C::Constant, C::Operation>, ArrayProgramBatching>
        + From<BroadcastOperation>
        + From<DimensionOperation<DimensionValue>>
        + From<DimensionSizeOperation>
        + OperationProjection<ArrayType>,
    <C::Operation as OperationProjection<ArrayType>>::Projected: From<TransposeOperation>,
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

/// Reads one packed array axis as a first-class dimension value in `context`.
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

/// Returns one first-class dimension operand for every packed array axis.
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

/// Normalizes one mapped composite array input to the batching context's common packed sharding placement.
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
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Operation: BatchableOperation<C, ArrayProgramBatching>
        + BatchableOperation<TracingContext<C::Constant, C::Operation>, ArrayProgramBatching>
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
            let batching_context = BatchingContext::<_, ArrayProgramBatching>::new(parent_context, extent)
                .with_axis_name(context.axis_name().map(str::to_string))
                .with_axis_sharding(context.axis_sharding().clone());
            let inputs = region
                .input_types()
                .iter()
                .zip(input_axes)
                .map(|(unbatched_type, batch_axis)| {
                    let batched_type = match (unbatched_type, batch_axis.axis()) {
                        (ArrayProgramType::Array(array_type), Some(axis)) => {
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
                            ArrayProgramType::Array(batched_type)
                        }
                        _ => unbatched_type.clone(),
                    };
                    let value = batching_context.parent().input(batched_type);
                    ArrayProgramBatch::new(value, *batch_axis)
                })
                .collect::<Result<Vec<_>, BatchingError>>()?;

            let region_mappings = RegionReplayMappings::new();
            let outputs = region.interpret_with(
                inputs,
                |_, constant| Ok(ArrayProgramBatch::replicated(batching_context.parent().lift(constant.clone())?)),
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

impl<C> NamedAxes for BatchingContext<C, ArrayProgramBatching>
where
    C: NamedAxes<Type = ArrayProgramType>,
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>
        + ValueProjection<DimensionType, Projected = DimensionValue>,
    C::Operation: BatchableOperation<C, ArrayProgramBatching>
        + BatchableOperation<TracingContext<C::Constant, C::Operation>, ArrayProgramBatching>
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

/// Projects composite array batches, applies one homogeneous array batching rule, and lifts its outputs.
fn batch_projected_array_operation<C, O>(
    operation: &O,
    context: &BatchingContext<C, ArrayProgramBatching>,
    inputs: &[ArrayProgramBatch<C::Value>],
) -> Result<Vec<ArrayProgramBatch<C::Value>>, BatchingError>
where
    C: Context<
            Type = ArrayProgramType,
            Operation: From<BroadcastOperation>
                           + From<DimensionOperation<DimensionValue>>
                           + From<DimensionSizeOperation>
                           + OperationProjection<ArrayType>,
        >,
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
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

/// Batches a condition whose regions live in the composite array-program universe.
///
/// A replicated predicate keeps one structural condition and gives each transformed branch the explicit leading
/// threaded mapped extent owned by [`ArrayProgramBatching::batch_program`]. A mapped predicate replays both pure branches
/// and selects their array outputs per item. Dimension outputs remain replicated: the rule requires both branch values
/// to be equal and returns that shared extent, so a genuinely predicate-varying dimension fails instead of being
/// misrepresented as one batch-wide shape value.
fn batch_condition<A, C, D>(
    context: &BatchingContext<C, ArrayProgramBatching>,
    driver: &D,
    inputs: &[ArrayProgramBatch<C::Value>],
) -> Result<Vec<ArrayProgramBatch<C::Value>>, BatchingError>
where
    A: Value<Type = ArrayType>,
    C: Context<
            Type = ArrayProgramType,
            Operation: From<BroadcastOperation>
                           + From<ConditionOperation<ArrayProgramValue<A>>>
                           + From<DimensionOperation<DimensionValue>>
                           + From<DimensionSizeOperation>
                           + OperationProjection<ArrayType>
                           + OperationProjection<DimensionType>,
        >,
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>
        + ValueProjection<DimensionType, Projected: Value<Type = DimensionType>>,
    C::Value: ValueProjection<ArrayType, Projected: LegacyBroadcast + Select + Transpose + Value<Type = ArrayType>>
        + ValueProjection<DimensionType, Projected: Value<Type = DimensionType>>,
    <C::Operation as OperationProjection<ArrayType>>::Projected:
        From<LegacyBroadcastOperation> + From<SelectOperation> + From<TransposeOperation>,
    <C::Operation as OperationProjection<DimensionType>>::Projected: From<DimensionRequirementOperation>,
    D: BatchingDriver<C, ArrayProgramBatching>,
{
    let Some((predicate, operands)) = inputs.split_first() else {
        return Err(BatchingError::UnsupportedOperation {
            message: "cannot batch a condition operation with no predicate input".to_string(),
        });
    };
    <&ArrayType>::try_from(predicate.unbatched_type())?;

    if predicate.batch_axis().is_replicated() {
        let operand_axes = operands.iter().map(ArrayProgramBatch::batch_axis).collect::<Vec<_>>();
        let true_region = driver.region(0)?;
        let false_region = driver.region(1)?;
        let true_axes = driver
            .batch_program(context, true_region, operand_axes.as_slice(), ProgramBatchingOutputAxesPolicy::Natural)?
            .output_axes()
            .to_vec();
        let false_axes = driver
            .batch_program(context, false_region, operand_axes.as_slice(), ProgramBatchingOutputAxesPolicy::Natural)?
            .output_axes()
            .to_vec();
        check_count!("output", false_axes, true_axes.len(), ProgramError);
        let output_axes = true_axes
            .iter()
            .zip(false_axes)
            .map(|(true_axis, false_axis)| if true_axis.is_replicated() { false_axis } else { *true_axis })
            .collect::<Vec<_>>();
        let (true_branch, _) = driver
            .batch_program(
                context,
                true_region,
                operand_axes.as_slice(),
                ProgramBatchingOutputAxesPolicy::AlignEachTo(output_axes.clone()),
            )?
            .into_parts();
        let (false_branch, _) = driver
            .batch_program(
                context,
                false_region,
                operand_axes.as_slice(),
                ProgramBatchingOutputAxesPolicy::AlignEachTo(output_axes.clone()),
            )?
            .into_parts();

        let mut packed_inputs = Vec::with_capacity(inputs.len() + 1);
        packed_inputs.push(predicate.value().clone());
        packed_inputs.push(context.axis_extent().clone());
        packed_inputs.extend(operands.iter().map(|operand| operand.value().clone()));
        let mut outputs = context.parent().bind(
            ConditionOperation::<ArrayProgramValue<A>>::new(),
            vec![true_branch, false_branch],
            packed_inputs.as_slice(),
        )?;
        check_count!("output", outputs, output_axes.len() + 1, ProgramError);
        outputs.remove(0);
        return outputs
            .into_iter()
            .zip(output_axes)
            .map(|(output, axis)| ArrayProgramBatch::new(output, axis))
            .collect();
    }

    let true_region = driver.region(0)?;
    let false_region = driver.region(1)?;
    if !true_region.effects().is_pure() || !false_region.effects().is_pure() {
        return Err(BatchingError::UnsupportedOperation {
            message: "cannot batch a condition with a batch-varying predicate and effectful branches because \
                      observable effects cannot be selected per batch item"
                .to_string(),
        });
    }
    let true_outputs = driver.batch_region(context, 0, operands.to_vec())?;
    let false_outputs = driver.batch_region(context, 1, operands.to_vec())?;
    check_count!("output", false_outputs, true_outputs.len(), ProgramError);
    true_outputs
        .into_iter()
        .zip(false_outputs)
        .map(|(true_output, false_output)| match true_output.unbatched_type() {
            ArrayProgramType::Array(_) => {
                <&ArrayType>::try_from(false_output.unbatched_type())?;
                let mut selected = batch_projected_array_operation(
                    &SelectOperation,
                    context,
                    &[predicate.clone(), true_output, false_output],
                )?;
                check_count!("output", selected, 1, ProgramError);
                Ok(selected.remove(0))
            }
            ArrayProgramType::Dimension(_) => {
                true_output.validate_replicated_dimension()?;
                false_output.validate_replicated_dimension()?;
                require_equal_dimensions(context.parent(), true_output.value(), false_output.value())?;
                Ok(true_output)
            }
        })
        .collect()
}

/// Removes the leading threaded-extent output while preserving the transformed region's complete input boundary.
///
/// A structurally batched condition program forwards the extent because that is the generic composite-region
/// protocol. A [`WhileOperation`] condition consumes the extent as loop state but must return only its predicate, so
/// the while rule uses this boundary projection before attaching the condition region.
fn without_threaded_extent_output<C: Context<Type = ArrayProgramType>>(
    program: Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>,
) -> Result<Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>, ProgramError> {
    if program.output_count() < 2 {
        return Err(ProgramError::MalformedProgram(
            "a structurally batched while condition must return its threaded extent and predicate".to_string(),
        ));
    }
    let mut builder = ProgramBuilder::<C::Constant, C::Operation>::new();
    let inputs = program.input_types().into_iter().map(|r#type| builder.add_input(r#type)).collect::<Vec<_>>();
    let outputs = builder.splice_program(&program, inputs.as_slice())?;
    let output_count = outputs.len() - 1;
    builder.build(outputs[1..].to_vec(), vec![Placeholder; inputs.len()], vec![Placeholder; output_count])
}

/// Batches a composite while loop while keeping first-class dimensions replicated loop state.
fn batch_while<C, D>(
    operation: &WhileOperation,
    context: &BatchingContext<C, ArrayProgramBatching>,
    driver: &D,
    inputs: &[ArrayProgramBatch<C::Value>],
) -> Result<Vec<ArrayProgramBatch<C::Value>>, BatchingError>
where
    C: Context<
            Type = ArrayProgramType,
            Operation: From<BroadcastOperation>
                           + From<DimensionOperation<DimensionValue>>
                           + From<DimensionSizeOperation>
                           + From<WhileOperation>
                           + OperationProjection<ArrayType>,
        >,
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Value: ValueProjection<ArrayType, Projected: Transpose + Value<Type = ArrayType>>,
    <C::Operation as OperationProjection<ArrayType>>::Projected: From<TransposeOperation>,
    D: BatchingDriver<C, ArrayProgramBatching>,
{
    let condition_region = driver.region(0)?;
    let body_region = driver.region(1)?;
    let state_count = inputs.len();

    // Canonicalize every mapped array carry to the leading axis. Dimension carries remain replicated.
    let mut state = inputs
        .iter()
        .cloned()
        .map(|input| match input.unbatched_type() {
            ArrayProgramType::Array(_) if !input.batch_axis().is_replicated() => {
                align_array_batch(context, input, Axis::from(0))
            }
            ArrayProgramType::Array(_) => Ok(input),
            ArrayProgramType::Dimension(_) => {
                input.validate_replicated_dimension()?;
                Ok(input)
            }
        })
        .collect::<Result<Vec<_>, BatchingError>>()?;
    let mut state_axes = state.iter().map(ArrayProgramBatch::batch_axis).collect::<Vec<_>>();

    // Iterate array carry axes to the same monotonic fixed point as the homogeneous rule. A dimension output can
    // never widen because structural composite batching rejects mapped dimensions at its boundary.
    let mut batched_body = None;
    for _ in 0..=state_count {
        let candidate = driver.batch_program(
            context,
            body_region,
            state_axes.as_slice(),
            ProgramBatchingOutputAxesPolicy::Natural,
        )?;
        check_count!("output", candidate.output_axes(), state_count, ProgramError);
        let mut widened = false;
        for (index, (state_axis, body_axis)) in state_axes.iter_mut().zip(candidate.output_axes().iter()).enumerate() {
            if state_axis.is_replicated() && !body_axis.is_replicated() {
                if matches!(inputs[index].unbatched_type(), ArrayProgramType::Dimension(_)) {
                    return Err(BatchingError::MappedDimension {
                        r#type: Box::new(<&DimensionType>::try_from(inputs[index].unbatched_type())?.clone()),
                        axis: *body_axis,
                    });
                }
                *state_axis = BatchAxis::new(0);
                widened = true;
            }
        }
        if !widened {
            batched_body = Some(
                driver
                    .batch_program(
                        context,
                        body_region,
                        state_axes.as_slice(),
                        ProgramBatchingOutputAxesPolicy::AlignEachTo(state_axes.clone()),
                    )?
                    .into_parts()
                    .0,
            );
            break;
        }
    }
    let Some(mut batched_body) = batched_body else {
        return Err(BatchingError::UnsupportedOperation {
            message: format!(
                "while loop batching failed to stabilize the loop state batch axes within {state_count} widening passes",
            ),
        });
    };

    let mut batched_condition = driver.batch_program(
        context,
        condition_region,
        state_axes.as_slice(),
        ProgramBatchingOutputAxesPolicy::Natural,
    )?;
    check_count!("output", batched_condition.output_axes(), 1, ProgramError);
    let batch_varying = !batched_condition.output_axes()[0].is_replicated();
    if batch_varying {
        if let Some(dimension) =
            inputs.iter().find(|input| matches!(input.unbatched_type(), ArrayProgramType::Dimension(_)))
        {
            return Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "cannot batch a while loop with a batch-varying predicate and first-class dimension state {} \
                     because one replicated dimension cannot represent per-item loop state",
                    dimension.unbatched_type(),
                ),
            });
        }

        // Per-item termination masks every array carry, so widen any still-replicated arrays and rebuild both regions
        // at the final invariant boundary. The predicate itself is forced to leading axis 0.
        state_axes.fill(BatchAxis::new(0));
        batched_body = driver
            .batch_program(
                context,
                body_region,
                state_axes.as_slice(),
                ProgramBatchingOutputAxesPolicy::AlignEachTo(state_axes.clone()),
            )?
            .into_parts()
            .0;
        batched_condition = driver.batch_program(
            context,
            condition_region,
            state_axes.as_slice(),
            ProgramBatchingOutputAxesPolicy::AlignEachTo(vec![BatchAxis::new(0)]),
        )?;
    }

    for (value, axis) in state.iter_mut().zip(state_axes.iter()) {
        if !axis.is_replicated() && value.batch_axis().is_replicated() {
            *value = align_array_batch(context, value.clone(), Axis::from(0))?;
        }
    }
    let (batched_condition, _) = batched_condition.into_parts();
    let batched_condition = without_threaded_extent_output::<C>(batched_condition)?;
    let mut packed_inputs = Vec::with_capacity(state.len() + 1);
    packed_inputs.push(context.axis_extent().clone());
    packed_inputs.extend(state.iter().map(|value| value.value().clone()));
    let mut outputs =
        context
            .parent()
            .bind(operation.clone(), vec![batched_condition, batched_body], packed_inputs.as_slice())?;
    check_count!("output", outputs, state_count + 1, ProgramError);
    outputs.remove(0);
    outputs
        .into_iter()
        .zip(state_axes)
        .map(|(output, axis)| ArrayProgramBatch::new(output, axis))
        .collect()
}

/// Batches a composite scan by carrying the mapped extent as its leading replicated state value.
fn batch_scan<A, C, D>(
    operation: &ScanOperation<ArrayProgramValue<A>>,
    context: &BatchingContext<C, ArrayProgramBatching>,
    driver: &D,
    inputs: &[ArrayProgramBatch<C::Value>],
) -> Result<Vec<ArrayProgramBatch<C::Value>>, BatchingError>
where
    A: Value<Type = ArrayType>,
    C: Context<
            Type = ArrayProgramType,
            Operation: From<BroadcastOperation>
                           + From<DimensionOperation<DimensionValue>>
                           + From<DimensionSizeOperation>
                           + From<ScanOperation<ArrayProgramValue<A>>>
                           + OperationProjection<ArrayType>,
        >,
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Value: ValueProjection<ArrayType, Projected: Transpose + Value<Type = ArrayType>>,
    <C::Operation as OperationProjection<ArrayType>>::Projected: From<TransposeOperation>,
    D: BatchingDriver<C, ArrayProgramBatching>,
{
    let body = driver.region(0)?;
    let (scan_inputs, runtime_length) = if operation.length().variable().is_some() {
        let Some((runtime_length, scan_inputs)) = inputs.split_last() else {
            return Err(ProgramError::InvalidInputCount { expected: body.input_types().len() + 1, actual: 0 }.into());
        };
        runtime_length.validate_replicated_dimension()?;
        (scan_inputs, Some(runtime_length))
    } else {
        (inputs, None)
    };
    check_count!("input", scan_inputs, body.input_types().len(), ProgramError);
    let carry_count = operation.carry_count();

    let mut carries = scan_inputs[..carry_count]
        .iter()
        .cloned()
        .map(|input| match input.unbatched_type() {
            ArrayProgramType::Array(_) if !input.batch_axis().is_replicated() => {
                align_array_batch(context, input, Axis::from(0))
            }
            ArrayProgramType::Array(_) => Ok(input),
            ArrayProgramType::Dimension(_) => {
                input.validate_replicated_dimension()?;
                Ok(input)
            }
        })
        .collect::<Result<Vec<_>, BatchingError>>()?;
    let stacks = scan_inputs[carry_count..]
        .iter()
        .cloned()
        .map(|input| {
            <&ArrayType>::try_from(input.unbatched_type())?;
            if input.batch_axis_position() == Some(0) {
                align_array_batch(context, input, Axis::from(1))
            } else {
                Ok(input)
            }
        })
        .collect::<Result<Vec<_>, BatchingError>>()?;
    let mut carry_axes = carries.iter().map(ArrayProgramBatch::batch_axis).collect::<Vec<_>>();
    let slice_axes = stacks.iter().map(|stack| scan_iteration_batch_axis(stack.batch_axis())).collect::<Vec<_>>();

    let mut y_axes = None;
    for _ in 0..=carry_count {
        let iteration_axes = carry_axes.iter().chain(slice_axes.iter()).copied().collect::<Vec<_>>();
        let candidate =
            driver.batch_program(context, body, iteration_axes.as_slice(), ProgramBatchingOutputAxesPolicy::Natural)?;
        check_count!("output", candidate.output_axes(), body.output_types().len(), ProgramError);
        let mut widened = false;
        for (index, (carry_axis, output_axis)) in carry_axes.iter_mut().zip(candidate.output_axes().iter()).enumerate()
        {
            if carry_axis.is_replicated() && !output_axis.is_replicated() {
                if matches!(scan_inputs[index].unbatched_type(), ArrayProgramType::Dimension(_)) {
                    return Err(BatchingError::MappedDimension {
                        r#type: Box::new(<&DimensionType>::try_from(scan_inputs[index].unbatched_type())?.clone()),
                        axis: *output_axis,
                    });
                }
                *carry_axis = BatchAxis::new(0);
                widened = true;
            }
        }
        if !widened {
            y_axes = Some(candidate.output_axes()[carry_count..].to_vec());
            break;
        }
    }
    let Some(y_axes) = y_axes else {
        return Err(BatchingError::UnsupportedOperation {
            message: format!(
                "scan batching failed to stabilize the carry batch axes within {carry_count} widening passes",
            ),
        });
    };

    let iteration_axes = carry_axes.iter().chain(slice_axes.iter()).copied().collect::<Vec<_>>();
    let target_axes = carry_axes.iter().chain(y_axes.iter()).copied().collect::<Vec<_>>();
    let (batched_body, _) = driver
        .batch_program(
            context,
            body,
            iteration_axes.as_slice(),
            ProgramBatchingOutputAxesPolicy::AlignEachTo(target_axes),
        )?
        .into_parts();
    for (carry, axis) in carries.iter_mut().zip(carry_axes.iter()) {
        if !axis.is_replicated() && carry.batch_axis().is_replicated() {
            *carry = align_array_batch(context, carry.clone(), Axis::from(0))?;
        }
    }

    let batched_scan = ScanOperation::<ArrayProgramValue<A>>::new(carry_count + 1, operation.length())
        .with_reverse(operation.reverse())
        .with_unroll(operation.unroll())?
        .with_captures(operation.captures().to_vec());
    let mut packed_inputs = Vec::with_capacity(inputs.len() + 1);
    packed_inputs.push(context.axis_extent().clone());
    packed_inputs.extend(carries.iter().map(|carry| carry.value().clone()));
    packed_inputs.extend(stacks.iter().map(|stack| stack.value().clone()));
    packed_inputs.extend(runtime_length.map(|runtime_length| runtime_length.value().clone()));
    let mut outputs = context.parent().bind(batched_scan, vec![batched_body], packed_inputs.as_slice())?;
    check_count!("output", outputs, 1 + carry_count + y_axes.len(), ProgramError);
    outputs.remove(0);
    let mut output_axes = carry_axes;
    output_axes.extend(y_axes.iter().map(|axis| match axis.axis() {
        Some(axis) => BatchAxis::new(axis.value() + 1),
        None => BatchAxis::replicated(),
    }));
    outputs
        .into_iter()
        .zip(output_axes)
        .map(|(output, axis)| ArrayProgramBatch::new(output, axis))
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
            Operation: From<BroadcastOperation>
                           + From<DimensionOperation<DimensionValue>>
                           + From<DimensionSizeOperation>
                           + OperationProjection<ArrayType>,
        >,
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>
        + ValueProjection<DimensionType, Projected: Value<Type = DimensionType>>,
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
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Value: ValueProjection<ArrayType, Projected: LegacyBroadcast + Transpose + Value<Type = ArrayType>>,
    C::Operation: From<BroadcastOperation>
        + From<ConcatenateOperation>
        + From<DimensionOperation<DimensionValue>>
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

        // Align every packed array on one mapped axis. Replicated operands gain that axis using the transform's
        // declared sharding, so each batch item concatenates the corresponding per-item arrays.
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
        + ValueProjection<DimensionType, Projected: Value<Type = DimensionType>>,
    C::Value: ValueProjection<ArrayType, Projected: LegacyBroadcast + Transpose + Value<Type = ArrayType>>,
    C::Operation: From<BroadcastOperation>
        + From<DimensionOperation<DimensionValue>>
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

/// Batches mapped RNG states through one composite scan whose replicated dimension carries define dynamic outputs.
fn batch_rng_bit_generator<A, C>(
    operation: &RngBitGeneratorOperation,
    context: &BatchingContext<C, ArrayProgramBatching>,
    inputs: &[ArrayProgramBatch<C::Value>],
) -> Result<Vec<ArrayProgramBatch<C::Value>>, BatchingError>
where
    A: Value<Type = ArrayType>,
    C: Context<
            Type = ArrayProgramType,
            Operation: From<BroadcastOperation>
                           + From<DimensionOperation<DimensionValue>>
                           + From<DimensionSizeOperation>
                           + From<RngBitGeneratorOperation>
                           + From<ScanOperation<ArrayProgramValue<A>>>
                           + OperationProjection<ArrayType>,
        >,
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Value: ValueProjection<ArrayType, Projected: Transpose + Value<Type = ArrayType>>,
    <C::Operation as OperationProjection<ArrayType>>::Projected: From<TransposeOperation>,
{
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
    let state = align_array_batch(context, state.clone(), Axis::from(0))?;
    let mut builder = ProgramBuilder::<C::Constant, C::Operation>::new();
    let extent_inputs = output_extents
        .iter()
        .map(|extent| builder.add_input(extent.unbatched_type().clone()))
        .collect::<Vec<_>>();
    let state_input = builder.add_input(ArrayProgramType::Array(operation.algorithm().state_type()));
    let operation_inputs = std::iter::once(state_input).chain(extent_inputs.iter().copied()).collect::<Vec<_>>();
    let random_outputs = builder.add_instruction(operation.clone(), Vec::new(), operation_inputs)?.to_vec();
    let body_outputs = extent_inputs.iter().copied().chain(random_outputs).collect::<Vec<_>>();
    let body = builder.build::<Vec<C::Constant>, Vec<C::Constant>>(
        body_outputs,
        vec![Placeholder; output_extents.len() + 1],
        vec![Placeholder; output_extents.len() + 2],
    )?;

    let extent_type = context.axis_extent().r#type();
    let length = <&DimensionType>::try_from(extent_type.as_ref())?.to_dimension();
    let scan = ScanOperation::<ArrayProgramValue<A>>::new(output_extents.len(), length.clone());
    let mut packed_inputs = output_extents.iter().map(|extent| extent.value().clone()).collect::<Vec<_>>();
    packed_inputs.push(state.into_value());
    if length.variable().is_some() {
        packed_inputs.push(context.axis_extent().clone());
    }
    let mut outputs = context.parent().bind(scan, vec![body], packed_inputs.as_slice())?;
    check_count!("output", outputs, output_extents.len() + 2, ProgramError);
    outputs.drain(..output_extents.len());
    let bits = outputs.remove(1);
    let advanced_states = outputs.remove(0);
    Ok(vec![
        ArrayProgramBatch::new(advanced_states, BatchAxis::new(0))?,
        ArrayProgramBatch::new(bits, BatchAxis::new(0))?,
    ])
}

impl<A, C> BatchableOperation<C, ArrayProgramBatching> for ArrayProgramOperation<A>
where
    A: Value<Type = ArrayType>,
    C: Context<
            Type = ArrayProgramType,
            Constant: ValueProjection<ArrayType, Projected = A>
                          + ValueProjection<DimensionType, Projected: Value<Type = DimensionType>>,
        >,
    C::Value: ValueProjection<ArrayType> + ValueProjection<DimensionType, Projected: Value<Type = DimensionType>>,
    <C::Value as ValueProjection<ArrayType>>::Projected:
        LegacyBroadcast + Reduce + Reshape + Select + Transpose + Value<Type = ArrayType>,
    C::Operation: From<ArrayProgramOperation<A>>
        + From<BroadcastOperation>
        + From<ConcatenateOperation>
        + From<ConditionOperation<ArrayProgramValue<A>>>
        + From<DimensionSizeOperation>
        + From<OneOperation<ArrayType>>
        + From<PadOperation>
        + From<RngBitGeneratorOperation>
        + From<ScanOperation<ArrayProgramValue<A>>>
        + From<WhileOperation>
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
            Self::Array(ArrayOperation::Condition(_)) | Self::Condition(_) => {
                batch_condition::<A, _, _>(context, driver, inputs)
            }
            Self::Array(ArrayOperation::While(operation)) | Self::While(operation) => {
                batch_while(operation, context, driver, inputs)
            }
            Self::Array(ArrayOperation::Scan(operation)) => {
                let operation = ScanOperation::<ArrayProgramValue<A>>::new(operation.carry_count(), operation.length())
                    .with_reverse(operation.reverse())
                    .with_unroll(operation.unroll())?
                    .with_captures(operation.captures().iter().cloned().map(ArrayProgramValue::Array).collect());
                batch_scan(&operation, context, driver, inputs)
            }
            Self::Scan(operation) => batch_scan(operation, context, driver, inputs),
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
                let batched_type = <&ArrayType>::try_from(input_type.as_ref())?;
                let packed_axis = match input.batch_axis.axis() {
                    Some(batch_axis) => {
                        let batch_axis = batch_axis.normalize(batched_type.rank())?;
                        if operation.axis() < batch_axis { operation.axis() } else { operation.axis() + 1 }
                    }
                    None => operation.axis(),
                };
                let operation = DimensionSizeOperation::new(batched_type, packed_axis)?;
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
            Self::RngBitGenerator(operation) => batch_rng_bit_generator::<A, _>(operation, context, inputs),
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

                let batched_type = <&ArrayType>::try_from(input.value.r#type().as_ref())?.clone();
                let moved_input = ArrayBatch::new(
                    batched_type,
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

                // Canonicalize the mapped axis to the leading position, then represent that axis in both the
                // explicit output extents and the input-to-output mapping of the lifted broadcast.
                let batched_type = <&ArrayType>::try_from(input.value.r#type().as_ref())?.clone();
                let moved_input = ArrayBatch::new(
                    batched_type,
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
    fn test_threaded_extent_batched_program_validates_its_boundary() -> Result<(), ProgramError> {
        type Context = TracingContext<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>;
        type TestProgramBuilder = ProgramBuilder<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>;

        // A threaded boundary always contributes one leading protocol input and output.
        let program = TestProgramBuilder::new().build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
            Vec::new(),
            Vec::new(),
            Vec::new(),
        )?;
        let Err(ProgramError::MalformedProgram(message)) = ThreadedExtentBatchedProgram::new(program, Vec::new())
        else {
            panic!("threaded-extent batching accepted a missing protocol boundary");
        };
        assert_eq!(
            message,
            "a structurally batched program with a threaded extent must have a leading input and output",
        );

        // The leading protocol input must be a first-class dimension rather than an arbitrary composite member.
        let mut builder = TestProgramBuilder::new();
        let array = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let program = builder.build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
            vec![array],
            vec![Placeholder],
            vec![Placeholder],
        )?;
        let Err(ProgramError::MalformedProgram(message)) = ThreadedExtentBatchedProgram::new(program, Vec::new())
        else {
            panic!("threaded-extent batching accepted a non-dimension protocol input");
        };
        assert_eq!(message, "a structurally batched program's leading threaded-extent input must be a dimension",);

        // The leading protocol output must also be a first-class dimension.
        let mut builder = TestProgramBuilder::new();
        builder
            .add_input(DimensionType::new(DimensionVariable::new("extent", DimensionBounds::new(0, Some(8))?)).into());
        let array = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let program = builder.build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
            vec![array],
            vec![Placeholder, Placeholder],
            vec![Placeholder],
        )?;
        let Err(ProgramError::MalformedProgram(message)) = ThreadedExtentBatchedProgram::new(program, Vec::new())
        else {
            panic!("threaded-extent batching accepted a non-dimension protocol output");
        };
        assert_eq!(message, "a structurally batched program's leading threaded-extent output must be a dimension",);

        // A merely compatible dimension output is insufficient: the program must forward the exact input atom.
        let mut builder = TestProgramBuilder::new();
        builder
            .add_input(DimensionType::new(DimensionVariable::new("extent", DimensionBounds::new(0, Some(8))?)).into());
        let other_extent = builder.add_input(
            DimensionType::new(DimensionVariable::new("other_extent", DimensionBounds::new(0, Some(8))?)).into(),
        );
        let program = builder.build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
            vec![other_extent],
            vec![Placeholder, Placeholder],
            vec![Placeholder],
        )?;
        let Err(ProgramError::MalformedProgram(message)) = ThreadedExtentBatchedProgram::new(program, Vec::new())
        else {
            panic!("threaded-extent batching accepted a substituted protocol output");
        };
        assert_eq!(
            message,
            "a structurally batched program's leading threaded-extent output must forward its leading input",
        );

        // A well-formed threaded boundary preserves its program and excludes the protocol output from its axes.
        let mut builder = TestProgramBuilder::new();
        let extent = builder
            .add_input(DimensionType::new(DimensionVariable::new("extent", DimensionBounds::new(0, Some(8))?)).into());
        let program = builder.build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
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
    fn test_array_program_batch_entrypoints() -> Result<(), ProgramError> {
        let matrix = ArrayProgramValue::Array(Array::matrix(2, 3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]));

        // The free transform infers its first-class mapped extent from the packed array input and can move
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
        let batched_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(batch_variable), Dimension::Static(3)]));
        let batch_extent = trace.input(batch_type.clone().into());
        let input = trace.input(batched_type.clone().into());
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
        let [dimension, broadcast, add] = elementwise_builder.instructions() else {
            panic!("expected one dimension constant, one dynamic broadcast, and one array add");
        };
        assert!(matches!(dimension.operation(), ArrayProgramOperation::Dimension(DimensionOperation::Constant(_)),));
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
        let recursive_input = recursive_parent.input(batched_type.into());
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
    fn test_composite_condition_batching() -> Result<(), ProgramError> {
        type TraceContext = TracingContext<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>;

        let unbatched_array_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)]));
        let shared_dimension_type =
            DimensionType::new(DimensionVariable::new("shared", DimensionBounds::new(0, Some(17))?));
        let mut branch_builder = ProgramBuilder::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
        let branch_array = branch_builder.add_input(unbatched_array_type.clone().into());
        let branch_dimension = branch_builder.add_input(shared_dimension_type.clone().into());
        let branch_array = branch_builder.add_instruction(
            ArrayProgramOperation::Array(ArrayOperation::Neg(NegOperation)),
            Vec::new(),
            vec![branch_array],
        )?[0];
        let branch = branch_builder.build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
            vec![branch_array, branch_dimension],
            vec![Placeholder, Placeholder],
            vec![Placeholder, Placeholder],
        )?;

        // A replicated predicate keeps one condition. Its transformed branches carry the mapped extent explicitly as
        // leading dimension state, while the reported output-axis metadata excludes that protocol value.
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
        let context = BatchingContext::<_, ArrayProgramBatching>::new(trace.clone(), batch_extent);
        let outputs = context.bind(
            ArrayProgramOperation::Condition(ConditionOperation::new()),
            vec![branch.clone(), branch.clone()],
            &[
                BatchingTracer::new(context.clone(), ArrayProgramBatch::replicated(predicate)),
                BatchingTracer::new(context.clone(), ArrayProgramBatch::new(array, BatchAxis::new(0))?),
                BatchingTracer::new(context.clone(), ArrayProgramBatch::replicated(shared_dimension)),
            ],
        )?;
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].batch().batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[1].batch().batch_axis(), BatchAxis::replicated());

        let output_ids = outputs.iter().map(|output| output.batch().value().atom_id().unwrap()).collect::<Vec<_>>();
        let program = trace
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
                output_ids,
                vec![Placeholder, Placeholder, Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )?;
        let [condition] = program.entry_region().instructions() else {
            panic!("expected exactly one structural condition instruction");
        };
        assert!(matches!(condition.operation(), ArrayProgramOperation::Condition(_)));
        assert_eq!(condition.inputs(), &[predicate_id, batch_extent_id, array_id, shared_dimension_id]);
        assert_eq!(condition.regions().len(), 2);
        for region_id in condition.regions() {
            let region = program.region(*region_id)?;
            assert_eq!(
                region.input_types(),
                vec![
                    ArrayProgramType::Dimension(DimensionType::new(batch.clone())),
                    ArrayProgramType::Array(batched_array_type.clone()),
                    ArrayProgramType::Dimension(shared_dimension_type.clone()),
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
                ArrayProgramValue::Dimension(DimensionValue::new(DimensionType::new(batch.clone()), 2)?),
                ArrayProgramValue::Array(Array::scalar(true)),
                ArrayProgramValue::Array(Array::matrix(2, 3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])),
                ArrayProgramValue::Dimension(shared_dimension_value.clone()),
            ])?,
            vec![
                ArrayProgramValue::Array(Array::matrix(2, 3, vec![-1.0_f32, -2.0, -3.0, -4.0, -5.0, -6.0])),
                ArrayProgramValue::Dimension(shared_dimension_value),
            ],
        );

        // Structural callers may force each output axis independently. Alignment happens while the replayed
        // values are still live tracers, and the leading extent protocol remains separate from the reported axis metadata.
        let forced_trace = TraceContext::new();
        let forced_extent = forced_trace.input(DimensionType::new(batch.clone()).into());
        let forced_context = BatchingContext::<_, ArrayProgramBatching>::new(forced_trace, forced_extent);
        let dynamic_natural = <ArrayProgramBatching as RecursiveBatchingPolicy<TraceContext>>::batch_program(
            &forced_context,
            branch.entry_region_ref(),
            &[BatchAxis::new(0), BatchAxis::replicated()],
            ProgramBatchingOutputAxesPolicy::Natural,
        )?
        .into_parts()
        .0;
        let forced = <ArrayProgramBatching as RecursiveBatchingPolicy<TraceContext>>::batch_program(
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
                ArrayProgramType::Dimension(DimensionType::new(batch.clone())),
                ArrayProgramType::Array(ArrayType::new(
                    DataType::F32,
                    Shape::new(vec![Dimension::Static(3), Dimension::Dynamic(batch.clone())]),
                )),
                ArrayProgramType::Dimension(shared_dimension_type.clone()),
            ],
        );
        assert!(matches!(
            <ArrayProgramBatching as RecursiveBatchingPolicy<TraceContext>>::batch_program(
                &forced_context,
                branch.entry_region_ref(),
                &[BatchAxis::new(0), BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            ),
            Err(BatchingError::MappedDimension { r#type, axis })
                if *r#type == shared_dimension_type && axis == BatchAxis::new(0),
        ));

        // Exact static extents use the identical threaded-extent protocol and instruction count. Only the boundary types
        // differ, so structural IR does not grow with or specialize on the mapped extent's runtime value.
        let static_trace = TraceContext::new();
        let static_extent_type = DimensionValue::constant(2)?.r#type().clone();
        let static_extent = static_trace.input(static_extent_type.clone().into());
        let static_context = BatchingContext::<_, ArrayProgramBatching>::new(static_trace, static_extent);
        let static_natural = <ArrayProgramBatching as RecursiveBatchingPolicy<TraceContext>>::batch_program(
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
                ArrayProgramType::Dimension(static_extent_type),
                ArrayProgramType::Array(ArrayType::new(
                    DataType::F32,
                    Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]),
                )),
                ArrayProgramType::Dimension(shared_dimension_type.clone()),
            ],
        );
        assert_eq!(static_natural.instructions().len(), dynamic_natural.instructions().len());

        // A second structural pass over the already-batched condition introduces one new leading threaded extent and
        // recursively re-batches the attached branches. The source extent stays an ordinary replicated dimension
        // operand, proving that nested batching does not recover either extent from array metadata.
        let nested_trace = TraceContext::new();
        let outer_batch = DimensionVariable::new("outer_batch", DimensionBounds::new(1, Some(5))?);
        let outer_extent = nested_trace.input(DimensionType::new(outer_batch.clone()).into());
        let nested_context = BatchingContext::<_, ArrayProgramBatching>::new(nested_trace, outer_extent);
        let nested = <ArrayProgramBatching as RecursiveBatchingPolicy<TraceContext>>::batch_program(
            &nested_context,
            program.entry_region_ref(),
            &[BatchAxis::replicated(), BatchAxis::replicated(), BatchAxis::new(0), BatchAxis::replicated()],
            ProgramBatchingOutputAxesPolicy::Natural,
        )?;
        let (nested, nested_axes) = nested.into_parts();
        assert_eq!(nested_axes, vec![BatchAxis::new(0), BatchAxis::replicated()]);
        assert_eq!(nested.input_types()[0], ArrayProgramType::Dimension(DimensionType::new(outer_batch.clone())),);
        assert_eq!(nested.input_types()[1], ArrayProgramType::Dimension(DimensionType::new(batch.clone())),);
        assert!(
            nested
                .instructions()
                .iter()
                .any(|instruction| matches!(instruction.operation(), ArrayProgramOperation::Condition(_)),)
        );

        // A mapped predicate replays both pure branches and selects array results per item. Equal dimension results
        // remain replicated and are guarded by an explicit equality requirement rather than becoming ragged values.
        let trace = TraceContext::new();
        let batch_extent = trace.input(DimensionType::new(batch.clone()).into());
        let predicate =
            trace.input(ArrayType::new(DataType::Boolean, Shape::new(vec![Dimension::Dynamic(batch.clone())])).into());
        let array = trace.input(batched_array_type.into());
        let shared_dimension = trace.input(shared_dimension_type.clone().into());
        let context = BatchingContext::<_, ArrayProgramBatching>::new(trace.clone(), batch_extent);
        let outputs = context.bind(
            ArrayProgramOperation::Condition(ConditionOperation::new()),
            vec![branch.clone(), branch],
            &[
                BatchingTracer::new(context.clone(), ArrayProgramBatch::new(predicate, BatchAxis::new(0))?),
                BatchingTracer::new(context.clone(), ArrayProgramBatch::new(array, BatchAxis::new(0))?),
                BatchingTracer::new(context.clone(), ArrayProgramBatch::replicated(shared_dimension)),
            ],
        )?;
        assert_eq!(outputs[0].batch().batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[1].batch().batch_axis(), BatchAxis::replicated());
        let builder = trace.builder().borrow();
        assert!(
            builder
                .instructions()
                .iter()
                .all(|instruction| !matches!(instruction.operation(), ArrayProgramOperation::Condition(_))),
        );
        assert!(builder.instructions().iter().any(|instruction| matches!(
            instruction.operation(),
            ArrayProgramOperation::Array(ArrayOperation::Select(_))
        ),));
        assert!(builder.instructions().iter().any(|instruction| matches!(
            instruction.operation(),
            ArrayProgramOperation::Dimension(DimensionOperation::Requirement(_)),
        )));

        Ok(())
    }

    #[test]
    fn test_composite_while_batching() -> Result<(), ProgramError> {
        type Context = EagerContext<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>;

        let dimension_type = DimensionType::new(DimensionVariable::new("shared", DimensionBounds::new(0, Some(17))?));
        let mut condition_builder = ProgramBuilder::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
        let condition_predicate = condition_builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        condition_builder.add_input(ArrayType::scalar(DataType::F32).into());
        condition_builder.add_input(dimension_type.clone().into());
        let condition = condition_builder.build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
            vec![condition_predicate],
            vec![Placeholder, Placeholder, Placeholder],
            vec![Placeholder],
        )?;

        let mut body_builder = ProgramBuilder::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
        body_builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let body_array = body_builder.add_input(ArrayType::scalar(DataType::F32).into());
        let body_dimension = body_builder.add_input(dimension_type.clone().into());
        let false_value = body_builder.add_constant(ArrayProgramValue::Array(Array::scalar(false)));
        let negated = body_builder.add_instruction(
            ArrayProgramOperation::Array(ArrayOperation::Neg(NegOperation)),
            Vec::new(),
            vec![body_array],
        )?[0];
        let body = body_builder.build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
            vec![false_value, negated, body_dimension],
            vec![Placeholder, Placeholder, Placeholder],
            vec![Placeholder, Placeholder, Placeholder],
        )?;

        // A scalar predicate controls the whole batch. Array state stays mapped, the dimension carry stays
        // replicated, and the explicit mapped extent crosses both rewritten regions as leading state.
        let context = BatchingContext::<_, ArrayProgramBatching>::new(
            Context::new(),
            ArrayProgramValue::Dimension(DimensionValue::constant(2)?),
        );
        let outputs = context.bind(
            ArrayProgramOperation::While(WhileOperation::new().with_iteration_bound(1)?),
            vec![condition.clone(), body.clone()],
            &[
                BatchingTracer::new(
                    context.clone(),
                    ArrayProgramBatch::replicated(ArrayProgramValue::Array(Array::scalar(true))),
                ),
                BatchingTracer::new(
                    context.clone(),
                    ArrayProgramBatch::new(
                        ArrayProgramValue::Array(Array::vector(vec![1.0_f32, 2.0])),
                        BatchAxis::new(0),
                    )?,
                ),
                BatchingTracer::new(
                    context.clone(),
                    ArrayProgramBatch::replicated(ArrayProgramValue::Dimension(DimensionValue::new(
                        dimension_type.clone(),
                        4,
                    )?)),
                ),
            ],
        )?;
        assert_eq!(outputs.len(), 3);
        assert_eq!(outputs[0].batch().batch_axis(), BatchAxis::replicated());
        assert_eq!(outputs[0].batch().value(), &ArrayProgramValue::Array(Array::scalar(false)));
        assert_eq!(outputs[1].batch().batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[1].batch().value(), &ArrayProgramValue::Array(Array::vector(vec![-1.0_f32, -2.0])),);
        assert_eq!(outputs[2].batch().batch_axis(), BatchAxis::replicated());
        assert_eq!(
            outputs[2].batch().value(),
            &ArrayProgramValue::Dimension(DimensionValue::new(dimension_type.clone(), 4)?),
        );

        // Once the predicate varies per item, first-class dimension state would need one independent value per item
        // and is therefore rejected instead of being silently treated as replicated.
        let error = context
            .bind(
                ArrayProgramOperation::While(WhileOperation::new().with_iteration_bound(1)?),
                vec![condition.clone(), body.clone()],
                &[
                    BatchingTracer::new(
                        context.clone(),
                        ArrayProgramBatch::new(
                            ArrayProgramValue::Array(Array::vector(vec![true, false])),
                            BatchAxis::new(0),
                        )?,
                    ),
                    BatchingTracer::new(
                        context.clone(),
                        ArrayProgramBatch::new(
                            ArrayProgramValue::Array(Array::vector(vec![1.0_f32, 2.0])),
                            BatchAxis::new(0),
                        )?,
                    ),
                    BatchingTracer::new(
                        context.clone(),
                        ArrayProgramBatch::replicated(ArrayProgramValue::Dimension(DimensionValue::new(
                            dimension_type.clone(),
                            4,
                        )?)),
                    ),
                ],
            )
            .unwrap_err();
        assert!(
            matches!(
                error.downcast_custom::<BatchingError>(),
                Some(BatchingError::UnsupportedOperation { message })
                    if message == "cannot batch a while loop with a batch-varying predicate and first-class dimension \
                                   state dimension<shared \u{2208} [0, 17)> because one replicated dimension cannot represent \
                                   per-item loop state",
            ),
            "{error:?}"
        );

        // Staging retains one direct composite while with explicit threaded extents in both regions. Rendering and
        // import preserve that boundary, and a second vmap recursively re-batches it without unrolling per item.
        type TraceContext = TracingContext<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>;
        let trace = TraceContext::new();
        let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(9))?);
        let batch_extent = trace.input(DimensionType::new(batch.clone()).into());
        let predicate = trace.input(ArrayType::scalar(DataType::Boolean).into());
        let array =
            trace.input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(batch.clone())])).into());
        let dimension = trace.input(dimension_type.clone().into());
        let input_ids = [batch_extent.clone(), predicate.clone(), array.clone(), dimension.clone()]
            .map(|input| input.atom_id().unwrap());
        let context = BatchingContext::<_, ArrayProgramBatching>::new(trace.clone(), batch_extent);
        let outputs = context.bind(
            ArrayProgramOperation::While(WhileOperation::new().with_iteration_bound(1)?),
            vec![condition, body],
            &[
                BatchingTracer::new(context.clone(), ArrayProgramBatch::replicated(predicate)),
                BatchingTracer::new(context.clone(), ArrayProgramBatch::new(array, BatchAxis::new(0))?),
                BatchingTracer::new(context.clone(), ArrayProgramBatch::replicated(dimension)),
            ],
        )?;
        let output_ids = outputs.iter().map(|output| output.batch().value().atom_id().unwrap()).collect::<Vec<_>>();
        let program = trace
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
                output_ids,
                vec![Placeholder; 4],
                vec![Placeholder; 3],
            )?;
        let [r#while] = program.entry_region().instructions() else {
            panic!("composite while batching should stage exactly one instruction");
        };
        assert!(matches!(r#while.operation(), ArrayProgramOperation::While(_)));
        assert_eq!(r#while.inputs(), &[input_ids[0], input_ids[1], input_ids[2], input_ids[3]]);
        assert_eq!(r#while.regions().len(), 2);
        let rendered = program.to_string();
        let mut imported_builder = ProgramBuilder::new();
        let imported = imported_builder.import_region(program.entry_region_ref());
        assert_eq!(imported_builder.region_ref(imported)?.to_program().to_string(), rendered);

        let nested_trace = TraceContext::new();
        let outer = DimensionVariable::new("outer", DimensionBounds::new(1, Some(5))?);
        let outer_extent = nested_trace.input(DimensionType::new(outer.clone()).into());
        let nested_context = BatchingContext::<_, ArrayProgramBatching>::new(nested_trace, outer_extent);
        let nested = <ArrayProgramBatching as RecursiveBatchingPolicy<TraceContext>>::batch_program(
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
                .filter(|instruction| matches!(instruction.operation(), ArrayProgramOperation::While(_)))
                .count(),
            1,
        );
        assert_eq!(nested.input_types()[0], ArrayProgramType::Dimension(DimensionType::new(outer)));

        Ok(())
    }

    #[test]
    fn test_composite_scan_batching() -> Result<(), ProgramError> {
        type Context = EagerContext<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>;

        let dimension_type = DimensionType::new(DimensionVariable::new("shared", DimensionBounds::new(0, Some(17))?));
        let mut body_builder = ProgramBuilder::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
        let carry = body_builder.add_input(ArrayType::scalar(DataType::F32).into());
        let dimension = body_builder.add_input(dimension_type.clone().into());
        let item = body_builder.add_input(ArrayType::scalar(DataType::F32).into());
        let next_carry = body_builder.add_instruction(
            ArrayProgramOperation::Array(ArrayOperation::Add(AddOperation)),
            Vec::new(),
            vec![carry, item],
        )?[0];
        let output = body_builder.add_instruction(
            ArrayProgramOperation::Array(ArrayOperation::Neg(NegOperation)),
            Vec::new(),
            vec![item],
        )?[0];
        let body = body_builder.build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
            vec![next_carry, dimension, output],
            vec![Placeholder, Placeholder, Placeholder],
            vec![Placeholder, Placeholder, Placeholder],
        )?;

        let context = BatchingContext::<_, ArrayProgramBatching>::new(
            Context::new(),
            ArrayProgramValue::Dimension(DimensionValue::constant(2)?),
        );
        let outputs = context.bind(
            ArrayProgramOperation::Scan(ScanOperation::new(2, 3).with_reverse(true)),
            vec![body.clone()],
            &[
                BatchingTracer::new(
                    context.clone(),
                    ArrayProgramBatch::new(
                        ArrayProgramValue::Array(Array::vector(vec![0.0_f32, 10.0])),
                        BatchAxis::new(0),
                    )?,
                ),
                BatchingTracer::new(
                    context.clone(),
                    ArrayProgramBatch::replicated(ArrayProgramValue::Dimension(DimensionValue::new(
                        dimension_type.clone(),
                        4,
                    )?)),
                ),
                BatchingTracer::new(
                    context.clone(),
                    ArrayProgramBatch::new(
                        ArrayProgramValue::Array(Array::matrix(3, 2, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])),
                        BatchAxis::new(1),
                    )?,
                ),
            ],
        )?;
        assert_eq!(outputs.len(), 3);
        assert_eq!(outputs[0].batch().batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].batch().value(), &ArrayProgramValue::Array(Array::vector(vec![9.0_f32, 22.0])),);
        assert_eq!(outputs[1].batch().batch_axis(), BatchAxis::replicated());
        assert_eq!(
            outputs[1].batch().value(),
            &ArrayProgramValue::Dimension(DimensionValue::new(dimension_type.clone(), 4)?),
        );
        assert_eq!(outputs[2].batch().batch_axis(), BatchAxis::new(1));
        assert_eq!(
            outputs[2].batch().value(),
            &ArrayProgramValue::Array(Array::matrix(3, 2, vec![-1.0_f32, -2.0, -3.0, -4.0, -5.0, -6.0],)),
        );

        // A zero-length scan never probes its body, preserves both carries, and returns an empty mapped stack.
        let zero_outputs = context.bind(
            ArrayProgramOperation::Scan(ScanOperation::new(2, 0)),
            vec![body],
            &[
                BatchingTracer::new(
                    context.clone(),
                    ArrayProgramBatch::new(
                        ArrayProgramValue::Array(Array::vector(vec![0.0_f32, 10.0])),
                        BatchAxis::new(0),
                    )?,
                ),
                BatchingTracer::new(
                    context.clone(),
                    ArrayProgramBatch::replicated(ArrayProgramValue::Dimension(DimensionValue::new(
                        dimension_type,
                        4,
                    )?)),
                ),
                BatchingTracer::new(
                    context.clone(),
                    ArrayProgramBatch::new(
                        ArrayProgramValue::Array(Array::new(
                            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(0), Dimension::Static(2)])),
                            Vec::new(),
                        )?),
                        BatchAxis::new(1),
                    )?,
                ),
            ],
        )?;
        assert_eq!(zero_outputs[0].batch().value(), &ArrayProgramValue::Array(Array::vector(vec![0.0_f32, 10.0])),);
        assert_eq!(zero_outputs[2].batch().batch_axis(), BatchAxis::new(1));
        assert_eq!(
            zero_outputs[2].batch().value().r#type().as_ref(),
            &ArrayProgramType::Array(ArrayType::new(
                DataType::F32,
                Shape::new(vec![Dimension::Static(0), Dimension::Static(2)]),
            )),
        );

        Ok(())
    }

    #[test]
    fn test_mapped_rng_batching_stages_one_dynamic_composite_scan() -> Result<(), ProgramError> {
        type Context = TracingContext<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>;

        let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(9))?);
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(7))?);
        let columns = DimensionVariable::new("columns", DimensionBounds::new(1, Some(11))?);
        let trace = Context::new();
        let batch_extent = trace.input(DimensionType::new(batch.clone()).into());
        let states = trace.input(
            ArrayType::new(DataType::U64, Shape::new(vec![Dimension::Dynamic(batch.clone()), Dimension::Static(2)]))
                .into(),
        );
        let row_extent = trace.input(DimensionType::new(rows.clone()).into());
        let column_extent = trace.input(DimensionType::new(columns.clone()).into());
        let input_ids = [batch_extent.clone(), states.clone(), row_extent.clone(), column_extent.clone()]
            .map(|input| input.atom_id().unwrap());
        let context = BatchingContext::<_, ArrayProgramBatching>::new(trace.clone(), batch_extent);
        let outputs = context.bind(
            ArrayProgramOperation::RngBitGenerator(RngBitGeneratorOperation::new(
                RandomAlgorithm::ThreeFry,
                ArrayType::new(
                    DataType::U32,
                    Shape::new(vec![Dimension::Dynamic(rows.clone()), Dimension::Dynamic(columns.clone())]),
                ),
            )),
            Vec::new(),
            &[
                BatchingTracer::new(context.clone(), ArrayProgramBatch::new(states, BatchAxis::new(0))?),
                BatchingTracer::new(context.clone(), ArrayProgramBatch::replicated(row_extent)),
                BatchingTracer::new(context.clone(), ArrayProgramBatch::replicated(column_extent)),
            ],
        )?;
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].batch().batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[1].batch().batch_axis(), BatchAxis::new(0));
        assert_eq!(
            outputs[1].batch().unbatched_type(),
            &ArrayProgramType::Array(ArrayType::new(
                DataType::U32,
                Shape::new(vec![Dimension::Dynamic(rows.clone()), Dimension::Dynamic(columns.clone())]),
            )),
        );

        let output_ids = outputs.iter().map(|output| output.batch().value().atom_id().unwrap()).collect::<Vec<_>>();
        let program = trace
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
                output_ids,
                vec![Placeholder; 4],
                vec![Placeholder; 2],
            )?;
        let [scan] = program.entry_region().instructions() else {
            panic!("mapped RNG batching should stage exactly one scan instruction");
        };
        let ArrayProgramOperation::Scan(scan_operation) = scan.operation() else {
            panic!("mapped RNG batching should stage the direct composite scan carrier");
        };
        assert_eq!(scan_operation.carry_count(), 2);
        assert_eq!(scan_operation.length(), &Dimension::Dynamic(batch.clone()));
        assert_eq!(scan.inputs(), &[input_ids[2], input_ids[3], input_ids[1], input_ids[0]]);
        assert_eq!(scan.regions().len(), 1);
        assert!(matches!(
            program.region(scan.regions()[0])?.instructions()[0].operation(),
            ArrayProgramOperation::RngBitGenerator(_),
        ));
        let rendered = program.to_string();
        let mut imported_builder = ProgramBuilder::new();
        let imported = imported_builder.import_region(program.entry_region_ref());
        assert_eq!(imported_builder.region_ref(imported)?.to_program().to_string(), rendered);

        // A second vmap structurally replays the already scan-decomposed RNG program. The inner runtime scan length
        // remains an explicit replicated dimension operand while the new mapped extent becomes its leading carry.
        let nested_trace = Context::new();
        let outer = DimensionVariable::new("outer", DimensionBounds::new(1, Some(5))?);
        let outer_extent = nested_trace.input(DimensionType::new(outer.clone()).into());
        let nested_context = BatchingContext::<_, ArrayProgramBatching>::new(nested_trace, outer_extent);
        let nested = <ArrayProgramBatching as RecursiveBatchingPolicy<Context>>::batch_program(
            &nested_context,
            program.entry_region_ref(),
            &[BatchAxis::replicated(), BatchAxis::new(0), BatchAxis::replicated(), BatchAxis::replicated()],
            ProgramBatchingOutputAxesPolicy::Natural,
        )?;
        assert_eq!(nested.output_axes(), &[BatchAxis::new(1), BatchAxis::new(1)]);
        let (nested, _) = nested.into_parts();
        assert_eq!(
            nested
                .instructions()
                .iter()
                .filter(|instruction| matches!(instruction.operation(), ArrayProgramOperation::Scan(_)))
                .count(),
            1,
        );
        assert_eq!(nested.input_types()[0], ArrayProgramType::Dimension(DimensionType::new(outer)));

        Ok(())
    }

    #[test]
    fn test_composite_condition_batching_rejects_effectful_mapped_predicate() -> Result<(), ProgramError> {
        type TraceContext = TracingContext<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>;

        let unbatched_array_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)]));
        let left_dimension_type =
            DimensionType::new(DimensionVariable::new("left", DimensionBounds::new(0, Some(17))?));
        let right_dimension_type =
            DimensionType::new(DimensionVariable::new("right", DimensionBounds::new(0, Some(17))?));
        let mut branch_builder = ProgramBuilder::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
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
        let branch = branch_builder.build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
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
        let context = BatchingContext::<_, ArrayProgramBatching>::new(trace, batch_extent);
        let error = context
            .bind(
                ArrayProgramOperation::Condition(ConditionOperation::new()),
                vec![branch.clone(), branch],
                &[
                    BatchingTracer::new(context.clone(), ArrayProgramBatch::new(predicate, BatchAxis::new(0))?),
                    BatchingTracer::new(context.clone(), ArrayProgramBatch::new(array, BatchAxis::new(0))?),
                    BatchingTracer::new(context.clone(), ArrayProgramBatch::replicated(left)),
                    BatchingTracer::new(context.clone(), ArrayProgramBatch::replicated(right)),
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
        type TraceContext = TracingContext<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>;

        let mut true_builder = ProgramBuilder::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
        let true_extent = true_builder.add_constant(ArrayProgramValue::Dimension(DimensionValue::constant(2)?));
        let true_branch = true_builder.build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
            vec![true_extent],
            Vec::new(),
            vec![Placeholder],
        )?;
        let mut false_builder = ProgramBuilder::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
        let false_extent = false_builder.add_constant(ArrayProgramValue::Dimension(DimensionValue::constant(3)?));
        let false_branch = false_builder.build::<Vec<ArrayProgramValue<Array>>, Vec<ArrayProgramValue<Array>>>(
            vec![false_extent],
            Vec::new(),
            vec![Placeholder],
        )?;

        let trace = TraceContext::new();
        let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(9))?);
        let batch_extent = trace.input(DimensionType::new(batch.clone()).into());
        let predicate =
            trace.input(ArrayType::new(DataType::Boolean, Shape::new(vec![Dimension::Dynamic(batch)])).into());
        let context = BatchingContext::<_, ArrayProgramBatching>::new(trace, batch_extent);
        let error = context
            .bind(
                ArrayProgramOperation::Condition(ConditionOperation::new()),
                vec![true_branch, false_branch],
                &[BatchingTracer::new(context.clone(), ArrayProgramBatch::new(predicate, BatchAxis::new(0))?)],
            )
            .unwrap_err();
        assert_eq!(error.to_string(), "2 == 3; observed 2=2, 3=3");

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
        // explicit output extents, including the inserted batch extent.
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

        // Mapped RNG state batching is scan-based: each mapped state is advanced independently and the generated bits
        // retain the mapped axis as their leading axis.
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
        let static_outputs =
            static_rng.batch(&context, &EmptyRegionDriver, std::slice::from_ref(&state_batch)).unwrap();
        assert_eq!(static_outputs.len(), 2);
        assert_eq!(static_outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(static_outputs[1].batch_axis(), BatchAxis::new(0));
        assert_eq!(
            static_outputs[0].value().r#type().as_ref(),
            &ArrayProgramType::Array(ArrayType::new(
                DataType::U64,
                Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]),
            )),
        );
        assert_eq!(
            static_outputs[1].value().r#type().as_ref(),
            &ArrayProgramType::Array(ArrayType::new(
                DataType::U32,
                Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]),
            )),
        );

        let dynamic_rng_extent = DimensionVariable::new("rng_count", DimensionBounds::new(1, Some(5)).unwrap());
        let dynamic_rng = ArrayProgramOperation::<Array>::from(RngBitGeneratorOperation::new(
            RandomAlgorithm::ThreeFry,
            ArrayType::new(DataType::U32, Shape::new(vec![Dimension::Dynamic(dynamic_rng_extent.clone())])),
        ));
        let dynamic_outputs = dynamic_rng
            .batch(
                &context,
                &EmptyRegionDriver,
                &[
                    state_batch,
                    ArrayProgramBatch::replicated(ArrayProgramValue::Dimension(
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
            &ArrayProgramType::Array(ArrayType::new(
                DataType::U32,
                Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]),
            )),
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

        // Concatenate aligns mapped array operands before shifting the per-item concatenation axis around the common
        // packed batch axis. Its trailing extent remains a replicated shape value.
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
