//! Batching support for programs that mix arrays with first-class dimensions.
//!
//! Arrays retain the ordinary [`ArrayBatch`] representation and existing array batching rules. First-class
//! dimensions are shared shape authority and therefore remain replicated across the logical batch. Mixed operations
//! explicitly state how they cross that boundary.

use std::borrow::Cow;
use std::fmt::{Debug, Display};

use ryft_macros::Parameter;

use crate::backends::array_programs::ArrayProgramOperation;
use crate::backends::arrays::ArrayOperation;
use crate::backends::dimensions::{DimensionOperation, DimensionValue};
use crate::batching::{ArrayBatch, BatchAxis, BatchableOperation, BatchingContext, BatchingError};
use crate::contexts::{Context, Domain, ProjectedContext, ValueResolution};
use crate::macros::check_count;
use crate::operations::constants::{One, OneOperation, Zero, ZeroOperation};
use crate::operations::control_flow::SelectOperation;
use crate::operations::custom_call::CustomCallOperation;
use crate::operations::dimensions::{DimensionSizeOperation, DimensionToScalarOperation};
use crate::operations::manipulation::reshaping::lift_output_sharding_for_leading_batch_axis;
use crate::operations::manipulation::{
    BroadcastOperation, CONCATENATE_OPERATION_NAME, ConcatenateOperation, LegacyBroadcast, PadOperation,
    ReshapeOperation, Transpose,
};
use crate::operations::random::RngBitGeneratorOperation;
use crate::parameters::Parameter;
use crate::programs::ProgramError;
use crate::programs::operations::{Operation, OperationProjection};
use crate::programs::regions::{BindingRegionDriver, EmptyRegionDriver, RegionDriver};
use crate::programs::types::{TypeError, Typed};
use crate::programs::values::{Value, ValueProjection};
use crate::sharding::ShardingDimension;
use crate::types::{ArrayProgramType, ArrayType, DataType, DimensionType};

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

/// Composite operation contract used by [`ArrayProgramBatchingContext`].
pub trait ArrayProgramBatchableOperation<C: Context<Type = ArrayProgramType>>: Operation<ArrayProgramType> {
    /// Applies this operation's batching rule. The current outer family is region-free, so its implementations do not
    /// yet use `driver`. Phase 5 assigns region-carrying composite operations dedicated rules that must use this driver
    /// to recurse into their attached regions rather than treating it as permanently unused.
    fn batch<D: RegionDriver<C::Constant, C::Operation>>(
        &self,
        _context: &ArrayProgramBatchingContext<C>,
        _driver: &D,
        inputs: &[ArrayProgramBatch<C::Value>],
    ) -> Result<Vec<ArrayProgramBatch<C::Value>>, BatchingError>;
}

/// Batching context for composite array programs.
#[derive(Clone)]
pub struct ArrayProgramBatchingContext<C: Context<Type = ArrayProgramType>> {
    /// Parent context that executes or stages physical work.
    parent: C,

    /// Size of the logical mapped axis.
    // TODO(eaplatanios): Phase 5 replaces this static metadata with a first-class dimension value when the mapped
    // extent is dynamic.
    axis_size: usize,

    /// Optional name through which collective operations address the mapped axis.
    axis_name: Option<String>,

    /// Sharding placement of the mapped axis.
    axis_sharding: ShardingDimension,
}

impl<C: Context<Type = ArrayProgramType>> ArrayProgramBatchingContext<C> {
    /// Creates a composite batching context.
    #[inline]
    pub fn new(parent: C, axis_size: usize) -> Self {
        Self { parent, axis_size, axis_name: None, axis_sharding: ShardingDimension::Replicated }
    }

    /// Sets the optional name through which collective operations address the mapped axis.
    #[inline]
    pub fn with_axis_name<N: Into<Option<String>>>(mut self, axis_name: N) -> Self {
        self.axis_name = axis_name.into();
        self
    }

    /// Sets the sharding placement of the mapped axis.
    #[inline]
    pub fn with_axis_sharding(mut self, axis_sharding: ShardingDimension) -> Self {
        self.axis_sharding = axis_sharding;
        self
    }

    /// Returns the parent context.
    #[inline]
    pub fn parent(&self) -> &C {
        &self.parent
    }

    /// Returns the mapped-axis size.
    #[inline]
    pub fn axis_size(&self) -> usize {
        self.axis_size
    }

    /// Returns the optional name through which collective operations address the mapped axis.
    #[inline]
    pub fn axis_name(&self) -> Option<&str> {
        self.axis_name.as_deref()
    }

    /// Returns the sharding placement of the mapped axis.
    #[inline]
    pub fn axis_sharding(&self) -> &ShardingDimension {
        &self.axis_sharding
    }
}

impl<C: Context<Type = ArrayProgramType>> Domain for ArrayProgramBatchingContext<C>
where
    C::Operation: ArrayProgramBatchableOperation<C>,
{
    type Type = ArrayProgramType;
    type Value = ArrayProgramBatchingTracer<C>;
    type Constant = C::Constant;
    type Operation = C::Operation;
}

impl<C: Context<Type = ArrayProgramType>> Context for ArrayProgramBatchingContext<C>
where
    C::Operation: ArrayProgramBatchableOperation<C>,
{
    fn lift(&self, constant: C::Constant) -> Result<Self::Value, ProgramError> {
        Ok(ArrayProgramBatchingTracer::new(self.clone(), ArrayProgramBatch::replicated(self.parent.lift(constant)?)))
    }

    fn bind<O: Into<Self::Operation>, D: BindingRegionDriver<Self::Constant, Self::Operation>>(
        &self,
        operation: O,
        driver: D,
        inputs: &[Self::Value],
    ) -> Result<Vec<Self::Value>, ProgramError> {
        let operation = operation.into();
        let inputs = inputs.iter().map(|input| input.batch.clone()).collect::<Vec<_>>();
        Ok(operation
            .batch(self, &driver, inputs.as_slice())?
            .into_iter()
            .map(|batch| ArrayProgramBatchingTracer::new(self.clone(), batch))
            .collect())
    }

    #[inline]
    fn is_eager(&self) -> bool {
        self.parent.is_eager()
    }

    #[inline]
    fn resolve(&self, value: &Self::Value) -> ValueResolution<Self::Constant> {
        self.parent.resolve(value.batch.value())
    }
}

/// Batch-carrying value flowing through [`ArrayProgramBatchingContext`].
#[derive(Clone, Parameter)]
pub struct ArrayProgramBatchingTracer<C: Context<Type = ArrayProgramType>> {
    /// Context through which operations dispatch.
    context: ArrayProgramBatchingContext<C>,

    /// Kind-aware physical batch.
    batch: ArrayProgramBatch<C::Value>,
}

impl<C: Context<Type = ArrayProgramType>> ArrayProgramBatchingTracer<C> {
    /// Creates a batch-carrying tracer.
    #[inline]
    pub fn new(context: ArrayProgramBatchingContext<C>, batch: ArrayProgramBatch<C::Value>) -> Self {
        Self { context, batch }
    }

    /// Returns the carried batch.
    #[inline]
    pub fn batch(&self) -> &ArrayProgramBatch<C::Value> {
        &self.batch
    }

    /// Consumes this tracer and returns its batch.
    #[inline]
    pub fn into_batch(self) -> ArrayProgramBatch<C::Value> {
        self.batch
    }
}

impl<C: Context<Type = ArrayProgramType, Value: PartialEq>> PartialEq for ArrayProgramBatchingTracer<C> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.batch == other.batch
    }
}

impl<C: Context<Type = ArrayProgramType>> Debug for ArrayProgramBatchingTracer<C> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("ArrayProgramBatchingTracer").field("batch", &self.batch).finish()
    }
}

impl<C: Context<Type = ArrayProgramType>> Display for ArrayProgramBatchingTracer<C> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.batch, formatter)
    }
}

impl<C: Context<Type = ArrayProgramType>> Typed for ArrayProgramBatchingTracer<C> {
    type Type = ArrayProgramType;

    #[inline]
    fn r#type(&self) -> Cow<'_, ArrayProgramType> {
        self.batch.r#type()
    }
}

impl<C: Context<Type = ArrayProgramType>> Value for ArrayProgramBatchingTracer<C>
where
    C::Operation: ArrayProgramBatchableOperation<C>,
{
    type DispatchDomain = ArrayProgramBatchingContext<C>;
    type ExecutionDomain = ArrayProgramBatchingContext<C>;

    #[inline]
    fn dispatch_domain(&self) -> Self::DispatchDomain {
        self.context.clone()
    }

    #[inline]
    fn execution_domain(&self) -> Self::ExecutionDomain {
        self.context.clone()
    }
}

// TODO(eaplatanios): Move this to the module where `ConcatenateOperation` is defined.
impl<C: Context<Type = ArrayProgramType>> ArrayProgramBatchableOperation<C> for ConcatenateOperation
where
    C::Value: ValueProjection<ArrayType, Projected: LegacyBroadcast + Transpose + Value<Type = ArrayType>>,
    C::Operation: From<ConcatenateOperation>,
{
    fn batch<D: RegionDriver<C::Constant, C::Operation>>(
        &self,
        context: &ArrayProgramBatchingContext<C>,
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

        let projected_inputs = inputs
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
        let Some(batch_axis) = projected_inputs.iter().find_map(ArrayBatch::batch_axis_position) else {
            return Ok(context
                .parent
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
        let axis_size = ArrayBatch::common_batch_size(&projected_inputs)?.expect("a mapped input pins the batch size");
        let aligned_inputs = projected_inputs
            .iter()
            .map(|input| input.match_axis(batch_axis, axis_size, context.axis_sharding.clone()))
            .collect::<Result<Vec<_>, _>>()?;
        let lifted_axis = if batch_axis <= self.axis() { self.axis() + 1 } else { self.axis() };
        let lifted_operation = ConcatenateOperation::new(lifted_axis, aligned_inputs[0].r#type().rank())?;
        let mut lifted_inputs = aligned_inputs
            .into_iter()
            .map(ArrayBatch::into_value)
            .map(<C::Value as ValueProjection<ArrayType>>::from_projected)
            .collect::<Vec<_>>();
        lifted_inputs.push(result_extent.value.clone());
        context
            .parent
            .bind(lifted_operation, Vec::new(), lifted_inputs.as_slice())?
            .into_iter()
            .map(|output| ArrayProgramBatch::new(output, BatchAxis::from_position(batch_axis)))
            .collect()
    }
}

impl<C: Context<Type = ArrayProgramType>> ArrayProgramBatchableOperation<C> for CustomCallOperation {
    fn batch<D: RegionDriver<C::Constant, C::Operation>>(
        &self,
        _context: &ArrayProgramBatchingContext<C>,
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

impl<C: Context<Type = ArrayProgramType>> ArrayProgramBatchableOperation<C> for PadOperation
where
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>
        + ValueProjection<DimensionType, Projected = DimensionValue>,
    C::Value: ValueProjection<ArrayType, Projected: LegacyBroadcast + Transpose + Value<Type = ArrayType>>,
    C::Operation: From<BroadcastOperation>
        + From<PadOperation>
        + OperationProjection<
            ArrayType,
            Projected: From<OneOperation<ArrayType>> + From<SelectOperation> + From<ZeroOperation<ArrayType>>,
        >,
{
    fn batch<D: RegionDriver<C::Constant, C::Operation>>(
        &self,
        context: &ArrayProgramBatchingContext<C>,
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
                .parent
                .bind(self.clone(), Vec::new(), &inputs.iter().map(|input| input.value.clone()).collect::<Vec<_>>())?
                .into_iter()
                .map(ArrayProgramBatch::replicated)
                .collect());
        };

        let operand_batch =
            operand_batch.match_axis(batch_axis, context.axis_size(), context.axis_sharding().clone())?;
        let mut edge_padding_low = self.edge_padding_low().to_vec();
        edge_padding_low.insert(batch_axis, 0);
        let mut edge_padding_high = self.edge_padding_high().to_vec();
        edge_padding_high.insert(batch_axis, 0);
        let mut interior_padding = self.interior_padding().to_vec();
        interior_padding.insert(batch_axis, 0);
        let lifted_operation = PadOperation::new(edge_padding_low, edge_padding_high, interior_padding)?;
        let batch_extent = DimensionValue::constant(context.axis_size()).map_err(ProgramError::from)?;
        let batch_extent = <C::Constant as ValueProjection<DimensionType>>::from_projected(batch_extent);
        let batch_extent = context.parent.lift(batch_extent)?;
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
                .parent
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
        let array_context = ProjectedContext::<C, ArrayType>::new(context.parent.clone());
        let padding_scalar_type = padding_value_batch.unbatched_type();
        let zero_padding =
            <C::Value as ValueProjection<ArrayType>>::from_projected(array_context.zero(&padding_scalar_type)?);
        let operand = <C::Value as ValueProjection<ArrayType>>::from_projected(operand_batch.into_value());
        let mut padded_inputs = Vec::with_capacity(lifted_output_extents.len() + 2);
        padded_inputs.push(operand.clone());
        padded_inputs.push(zero_padding);
        padded_inputs.extend(lifted_output_extents.iter().cloned());
        let mut padded = context.parent.bind(lifted_operation.clone(), Vec::new(), padded_inputs.as_slice())?;
        check_count!("output", padded, 1, ProgramError);
        let padded = padded.remove(0);

        let operand_type = <&ArrayType>::try_from(operand.r#type().as_ref())?
            .clone()
            .with_data_type(DataType::Boolean)
            .with_layout(None);
        let mask_input = <C::Value as ValueProjection<ArrayType>>::from_projected(array_context.one(&operand_type)?);
        let mask_padding_type = padding_scalar_type.with_data_type(DataType::Boolean).with_layout(None);
        let mask_padding =
            <C::Value as ValueProjection<ArrayType>>::from_projected(array_context.zero(&mask_padding_type)?);
        let mut mask_inputs = Vec::with_capacity(lifted_output_extents.len() + 2);
        mask_inputs.push(mask_input);
        mask_inputs.push(mask_padding);
        mask_inputs.extend(lifted_output_extents.iter().cloned());
        let mut mask = context.parent.bind(lifted_operation, Vec::new(), mask_inputs.as_slice())?;
        check_count!("output", mask, 1, ProgramError);
        let mask = mask.remove(0);

        let mut broadcast_inputs = Vec::with_capacity(lifted_output_extents.len() + 1);
        broadcast_inputs.push(<C::Value as ValueProjection<ArrayType>>::from_projected(
            padding_value_batch.move_axis(0)?.into_value(),
        ));
        broadcast_inputs.extend(lifted_output_extents);
        let mut broadcasted_padding =
            context
                .parent
                .bind(BroadcastOperation::new(vec![batch_axis]), Vec::new(), broadcast_inputs.as_slice())?;
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

impl<C: Context<Type = ArrayProgramType>> ArrayProgramBatchableOperation<C> for RngBitGeneratorOperation {
    fn batch<D: RegionDriver<C::Constant, C::Operation>>(
        &self,
        _context: &ArrayProgramBatchingContext<C>,
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

impl<A, C> ArrayProgramBatchableOperation<C> for ArrayProgramOperation<A>
where
    A: Value<Type = ArrayType>,
    C: Context<
            Type = ArrayProgramType,
            Constant: ValueProjection<ArrayType, Projected = A>
                          + ValueProjection<DimensionType, Projected = DimensionValue>,
        >,
    C::Value: ValueProjection<ArrayType, Projected: LegacyBroadcast + Transpose + Value<Type = ArrayType>>
        + ValueProjection<DimensionType, Projected: Value<Type = DimensionType>>,
    C::Operation: From<ArrayProgramOperation<A>>
        + From<BroadcastOperation>
        + From<ConcatenateOperation>
        + From<PadOperation>
        + OperationProjection<ArrayType, Projected = ArrayOperation<A>>
        + OperationProjection<DimensionType, Projected = DimensionOperation<DimensionValue>>,
    ArrayOperation<A>: BatchableOperation<ProjectedContext<C, ArrayType>>,
{
    fn batch<D: RegionDriver<C::Constant, C::Operation>>(
        &self,
        context: &ArrayProgramBatchingContext<C>,
        driver: &D,
        inputs: &[ArrayProgramBatch<C::Value>],
    ) -> Result<Vec<ArrayProgramBatch<C::Value>>, BatchingError> {
        match self {
            Self::Zero(_) => {
                if !inputs.is_empty() {
                    return Err(ProgramError::InvalidInputCount { expected: 0, actual: inputs.len() }.into());
                }
                Ok(context
                    .parent
                    .bind(self.clone(), Vec::new(), &[])?
                    .into_iter()
                    .map(ArrayProgramBatch::replicated)
                    .collect())
            }
            Self::DynamicZero(_) | Self::DynamicOne(_) | Self::DynamicIota(_) => {
                // Output extents are shared shape authority. A mapped extent would request a different output shape
                // for each batch item, which requires a ragged representation that ordinary array batching lacks.
                for extent in inputs {
                    extent.validate_replicated_dimension()?;
                }
                Ok(context
                    .parent
                    .bind(
                        self.clone(),
                        Vec::new(),
                        &inputs.iter().map(|input| input.value.clone()).collect::<Vec<_>>(),
                    )?
                    .into_iter()
                    .map(ArrayProgramBatch::replicated)
                    .collect())
            }
            Self::Array(operation) => {
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
                let projected_context =
                    BatchingContext::new(ProjectedContext::new(context.parent.clone()), context.axis_size)
                        .with_axis_name(context.axis_name.clone())
                        .with_axis_sharding(context.axis_sharding.clone());
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
            Self::Dimension(operation) => {
                for input in inputs {
                    input.validate_replicated_dimension()?;
                }
                let inputs = inputs
                    .iter()
                    .map(|input| <C::Value as ValueProjection<DimensionType>>::into_projected(input.value.clone()))
                    .collect::<Result<Vec<_>, TypeError>>()?;
                Ok(ProjectedContext::<C, DimensionType>::new(context.parent.clone())
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
                    .parent
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
                    .parent
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
                    .parent
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
                    .parent
                    .bind(ArrayProgramOperation::<A>::from(operation), Vec::new(), std::slice::from_ref(&input.value))?
                    .into_iter()
                    .map(ArrayProgramBatch::replicated)
                    .collect())
            }
            Self::Concatenate(operation) => ArrayProgramBatchableOperation::batch(operation, context, driver, inputs),
            Self::CustomCall(operation) => ArrayProgramBatchableOperation::batch(operation, context, driver, inputs),
            Self::Pad(operation) => ArrayProgramBatchableOperation::batch(operation, context, driver, inputs),
            Self::RngBitGenerator(operation) => {
                ArrayProgramBatchableOperation::batch(operation, context, driver, inputs)
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
                        .parent
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
                let axis_extent = DimensionValue::constant(context.axis_size()).map_err(ProgramError::from)?;
                let axis_extent = <C::Constant as ValueProjection<DimensionType>>::from_projected(axis_extent);
                let axis_extent = context.parent.lift(axis_extent)?;

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
                    .parent
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
                        .parent
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
                let axis_extent = DimensionValue::constant(context.axis_size()).map_err(ProgramError::from)?;
                let axis_extent = <C::Constant as ValueProjection<DimensionType>>::from_projected(axis_extent);
                let axis_extent = context.parent.lift(axis_extent)?;

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
                    .parent
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
    use pretty_assertions::assert_eq;

    use crate::Scalar;
    use crate::backends::array_programs::ArrayProgramValue;
    use crate::backends::arrays::Array;
    use crate::backends::dimensions::DimensionValue;
    use crate::contexts::{EagerContext, StagingContext};
    use crate::operations::compare::{CompareOperation, ComparisonDirection};
    use crate::operations::constants::{IotaOperation, OneOperation, ZeroOperation};
    use crate::operations::dimensions::{
        DimensionAddOperation, DimensionFromScalar, DimensionFromScalarOperation, DimensionSize, DimensionToScalar,
        DimensionToScalarOperation,
    };
    use crate::operations::random::{RandomAlgorithm, RngBitGeneratorOperation};
    use crate::operations::{CollectiveKind, CollectiveOperation};
    use crate::parameters::Placeholder;
    use crate::tracing::TracingContext;
    use crate::types::dimensions::{DimensionBounds, DimensionVariable};
    use crate::types::{DataType, Dimension, Shape};

    use super::*;

    #[test]
    fn test_array_program_batching() {
        type Parent = EagerContext<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>;
        fn assert_batchable<C: Context<Type = ArrayProgramType>, O: ArrayProgramBatchableOperation<C>>() {}
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

        let context = ArrayProgramBatchingContext::new(Parent::new(), 2)
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

        // Every rule that consumes a first-class dimension preserves the same typed mapped-authority diagnostic,
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
        // physical batch axis. Its trailing extent remains replicated shape authority.
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

        let dimension = ArrayProgramBatchingTracer::new(context.clone(), ArrayProgramBatch::replicated(dimension));
        let scalar = dimension.to_scalar().unwrap().into_batch();
        assert_eq!(scalar.batch_axis(), BatchAxis::replicated());
        assert_eq!(scalar.into_value(), ArrayProgramValue::Array(Array::scalar(4_i64)));

        let scalar = ArrayProgramBatchingTracer::new(
            context.clone(),
            ArrayProgramBatch::replicated(ArrayProgramValue::Array(Array::scalar(4_i32))),
        );
        let dimension = scalar.to_dimension(gateway_variable).unwrap().into_batch();
        assert_eq!(dimension.batch_axis(), BatchAxis::replicated());
        assert!(matches!(dimension.into_value(), ArrayProgramValue::Dimension(value) if value.extent() == 4));

        let array = ArrayProgramValue::Array(Array::matrix(2, 3, vec![0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0]));
        let array = ArrayProgramBatch::new(array, BatchAxis::new(0)).unwrap();
        let array = ArrayProgramBatchingTracer::new(context, array);
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
        let context = ArrayProgramBatchingContext::new(Parent::new(), 2);
        let input = ArrayProgramBatchingTracer::new(
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
