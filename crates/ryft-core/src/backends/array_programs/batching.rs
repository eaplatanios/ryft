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
use crate::operations::dimensions::{DimensionSizeOperation, DimensionToScalarOperation};
use crate::operations::manipulation::reshaping::lift_reshape_output_sharding;
use crate::operations::manipulation::{ReshapeOperation, Transpose};
use crate::parameters::Parameter;
use crate::programs::ProgramError;
use crate::programs::operations::{Operation, OperationProjection};
use crate::programs::regions::{BindingRegionDriver, EmptyRegionDriver, RegionDriver};
use crate::programs::types::{TypeError, Typed};
use crate::programs::values::{Value, ValueProjection};
use crate::sharding::ShardingDimension;
use crate::types::{ArrayProgramType, ArrayType, DimensionType};

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
        context: &ArrayProgramBatchingContext<C>,
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

impl<A, C> ArrayProgramBatchableOperation<C> for ArrayProgramOperation<A>
where
    A: Value<Type = ArrayType>,
    C: Context<
            Type = ArrayProgramType,
            Constant: ValueProjection<ArrayType, Projected = A>
                          + ValueProjection<DimensionType, Projected = DimensionValue>,
        >,
    C::Value: ValueProjection<ArrayType, Projected: Transpose + Value<Type = ArrayType>>
        + ValueProjection<DimensionType, Projected: Value<Type = DimensionType>>,
    C::Operation: From<ArrayProgramOperation<A>>
        + OperationProjection<ArrayType, Projected = ArrayOperation<A>>
        + OperationProjection<DimensionType, Projected = DimensionOperation<DimensionValue>>,
    ArrayOperation<A>: BatchableOperation<ProjectedContext<C, ArrayType>>,
{
    fn batch<D: RegionDriver<C::Constant, C::Operation>>(
        &self,
        context: &ArrayProgramBatchingContext<C>,
        _driver: &D,
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
                    lifted_operation = lifted_operation.with_output_sharding(lift_reshape_output_sharding(
                        output_sharding,
                        context.axis_sharding().clone(),
                    )?);
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

    use crate::backends::array_programs::ArrayProgramValue;
    use crate::backends::arrays::Array;
    use crate::backends::dimensions::DimensionValue;
    use crate::contexts::{EagerContext, StagingContext};
    use crate::operations::compare::{CompareOperation, ComparisonDirection};
    use crate::operations::constants::ZeroOperation;
    use crate::operations::dimensions::{
        DimensionAddOperation, DimensionFromScalar, DimensionFromScalarOperation, DimensionSize, DimensionToScalar,
        DimensionToScalarOperation,
    };
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
