//! Contains the named-axis [`ParallelSumScatterOperation`], which sums every participant's operand across a named
//! axis and scatters the result, together with its interpretation, partial-evaluation, batching, forward-mode
//! differentiation, and transposition rules.

// TODO(eaplatanios): Review this module.

use std::fmt::Display;

use crate::arrays::batching::DynamicArrayBatchingPolicy;
use crate::arrays::{
    ArrayBatch, ArrayBatching, ArrayIrBatch, ArrayIrBatching, ArrayIrType, ArrayType, Dimension, DimensionOperation,
    DimensionType, DimensionValue, DimensionVariable, Shape, Sharding,
};
use crate::axes::NamedAxes;
use crate::batching::{
    BatchAxis, BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError,
    MemberBatchableOperation,
};
use crate::contexts::{Context, Domain, ProjectedContext};
use crate::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationDriver, DifferentiationDual, DifferentiationError,
    LinearCallOperation, MemberDifferentiableOperation, TransposableOperation, TranspositionDriver,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver, MemberInterpretableOperation};
use crate::macros::check_count;
use crate::operations::constants::constant::ConstantOperation;
use crate::operations::dimensions::dimension_requirement::DimensionRequirement;
use crate::operations::dimensions::dimension_size::{DimensionSize, DimensionSizeOperation};
use crate::operations::manipulation::broadcasting::DynamicBroadcastOperation;
use crate::operations::manipulation::reshaping::{
    DynamicReshapeOperation, Reshape, lift_output_sharding_for_leading_batch_axis,
};
use crate::operations::manipulation::slicing::resized_output_sharding;
use crate::operations::manipulation::transposition::Transpose;
use crate::operations::math::div::Div;
use crate::operations::math::mul::Mul;
use crate::operations::math::reduce::{Reduce, ReductionKind};
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::{
    MaybeZero, MemberOperation, Operation, OperationFormatter, OperationProjection, ProgramError, ProjectedValue,
    RegionInterface, TypeError, TypeIdentityRenaming, Typed, Value, ValueProjection,
};
use crate::tracing::{Tracer, TracingContext};

use super::all_gather::{AllGatherOperation, AllGatherOutputVariance};
use super::{
    CollectiveBatchingPolicy, CollectiveMode, CollectiveOptions, collective_input_extents, divided_collective_extent,
    explicit_collective_inputs, forward_collective_to_parent, forward_explicit_collective,
    forward_shape_changing_collective, impl_shape_changing_collective_member_operation,
    infer_explicit_shape_changing_collective_output_type, interpret_degenerate_collective,
    jvp_shape_changing_collective_with_adjoint, reject_ragged_collective_inputs, require_collective_axis_extent,
    resolve_named_axis_size, shape_changing_collective, shape_changing_collective_dimensions,
    shape_changing_collective_output_type, transpose_shape_changing_collective, validate_collective_axis_size,
    validate_explicit_collective_output_extents,
};

/// Applies sum-scatter's reduction-state transition. Ordinary operands preserve their variance metadata. An operand
/// that is unreduced over the scattered manual axis is the cotangent of a reduced all-gather result; sum-scatter
/// consumes that pending reduction and returns a value varying over the manual axis.
fn parallel_sum_scatter_output_type(
    input_type: &ArrayType,
    mut output_type: ArrayType,
    operation: &ParallelSumScatterOperation,
) -> Result<ArrayType, TypeError> {
    if input_type.unreduced_axes().is_empty() {
        return Ok(output_type);
    }
    if input_type.unreduced_axes().len() != 1 || !input_type.unreduced_axes().contains(operation.axis_name()) {
        return Err(TypeError::invalid(format!(
            "`parallel_sum_scatter` only supports an unreduced operand over its own axis `{}`",
            operation.axis_name(),
        )));
    }
    let input_sharding = input_type.sharding().expect("unreduced axes require sharding metadata");
    let mut varying_axes = input_sharding.varying_manual_axes().clone();
    varying_axes.insert(operation.axis_name().to_string());
    let output_sharding = output_type.sharding().expect("shape projection preserves sharding").clone();
    output_type.sharding = Some(
        output_sharding
            .with_unreduced_axes(Vec::<String>::new())
            .and_then(|sharding| sharding.with_varying_manual_axes(varying_axes))
            .map_err(|error| TypeError::invalid(error.to_string()))?,
    );
    Ok(output_type)
}

/// Infers the composite sum-scatter contract.
pub(crate) fn infer_explicit_parallel_sum_scatter_output_types(
    operation: &ParallelSumScatterOperation,
    input_types: &[ArrayIrType],
) -> Result<Vec<ArrayIrType>, TypeError> {
    let effective_axis_size = operation.effective_axis_size()?;
    let Some(input_type) = input_types.first() else {
        return Err(TypeError::invalid("`parallel_sum_scatter` expects an array followed by its output extents"));
    };
    let input_type = <&ArrayType>::try_from(input_type)?;
    if operation.options.mode == CollectiveMode::Untiled {
        let Some(input_extent) = input_type.shape().dimensions().get(operation.scatter_axis) else {
            return Err(TypeError::invalid(format!(
                "`parallel_sum_scatter` scatter axis {} is out of bounds for rank {}",
                operation.scatter_axis,
                input_type.rank(),
            )));
        };
        if let Dimension::Static(input_extent) = input_extent
            && *input_extent != effective_axis_size
        {
            return Err(TypeError::invalid(format!(
                "`parallel_sum_scatter` untiled scatter axis {} size {input_extent} must equal group size \
                 {effective_axis_size}",
                operation.scatter_axis,
            )));
        }
        let base_output_type = input_type.without_dimension(operation.scatter_axis)?.0;
        let unchanged_input_axes = (0..base_output_type.rank())
            .map(|axis| if axis < operation.scatter_axis { Some(axis) } else { Some(axis + 1) })
            .collect::<Vec<_>>();
        let mut output_types = infer_explicit_shape_changing_collective_output_type(
            PARALLEL_SUM_SCATTER_OPERATION_NAME,
            input_types,
            base_output_type,
            unchanged_input_axes.as_slice(),
            |_, _| Ok(()),
        )?;
        let output_type = <&ArrayType>::try_from(&output_types.remove(0))?.clone();
        return Ok(vec![parallel_sum_scatter_output_type(input_type, output_type, operation)?.into()]);
    }
    if operation.scatter_axis >= input_type.rank() {
        return Err(TypeError::invalid(format!(
            "`parallel_sum_scatter` scatter axis {} is out of bounds for rank {}",
            operation.scatter_axis,
            input_type.rank(),
        )));
    }
    let mut dimensions = input_type.shape().dimensions().to_vec();
    dimensions[operation.scatter_axis] = Dimension::Static(0);
    let sharding = resized_output_sharding(input_type, dimensions.as_slice(), PARALLEL_SUM_SCATTER_OPERATION_NAME)?;
    let mut base_output_type =
        ArrayType::new(input_type.data_type(), Shape::new(dimensions)).with_memory(input_type.memory());
    base_output_type.sharding = sharding;
    let unchanged_input_axes = (0..input_type.rank())
        .map(|axis| (axis != operation.scatter_axis).then_some(axis))
        .collect::<Vec<_>>();
    let mut output_types = infer_explicit_shape_changing_collective_output_type(
        PARALLEL_SUM_SCATTER_OPERATION_NAME,
        input_types,
        base_output_type,
        unchanged_input_axes.as_slice(),
        |input_type, output_extents| {
            let rank = input_type.rank();
            let Some(input_extent) = input_type.shape().dimensions().get(operation.scatter_axis) else {
                return Err(TypeError::invalid(format!(
                    "`parallel_sum_scatter` scatter axis {} is out of bounds for rank {rank}",
                    operation.scatter_axis,
                )));
            };
            if let (Dimension::Static(input_extent), Dimension::Static(output_extent)) =
                (input_extent, &output_extents[operation.scatter_axis])
            {
                if *input_extent % effective_axis_size != 0 {
                    return Err(TypeError::invalid(format!(
                        "`parallel_sum_scatter` scatter axis {} size {input_extent} is not divisible by group size \
                         {effective_axis_size}",
                        operation.scatter_axis,
                    )));
                }
                let expected = *input_extent / effective_axis_size;
                if *output_extent != expected {
                    return Err(TypeError::invalid(format!(
                        "`parallel_sum_scatter` result extent must equal input axis {} extent {input_extent} divided \
                         by axis group size {effective_axis_size}; expected {expected} but got {output_extent}",
                        operation.scatter_axis,
                    )));
                }
            }
            Ok(())
        },
    )?;
    let output_type = <&ArrayType>::try_from(&output_types.remove(0))?.clone();
    Ok(vec![parallel_sum_scatter_output_type(input_type, output_type, operation)?.into()])
}

shape_changing_collective! {
    /// [`Operation`] that sums every participant's operand across the named axis and scatters the result: each
    /// participant receives its own chunk of the sum along `scatter_axis` — the analogue of
    /// [JAX's `psum_scatter`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.psum_scatter.html) with
    /// `tiled = True` and [StableHLO's `reduce_scatter`](https://openxla.org/stablehlo/spec#reduce_scatter) with a
    /// sum reduction. The output shrinks `scatter_axis` by the axis size (the dimension must be divisible by it).
    /// The collective is linear and its transpose is [`AllGatherOperation`] over the same axis and dimension. A
    /// matching `batch` level consumes the mapped batch axis by summing over it and re-mapping the chunks of
    /// `scatter_axis` onto it, so batch item `i` receives chunk `i` of the sum.
    operation = ParallelSumScatterOperation,
    name = PARALLEL_SUM_SCATTER_OPERATION_NAME = "parallel_sum_scatter",
    fields = {
        /// Axis of the operand along which the summed result is scattered across the participants.
        scatter_axis: usize,

        /// Shared rank and participant-group semantics.
        options: CollectiveOptions,
    },
    infer = |operation, input_type, dimensions| {
        let effective_axis_size = operation.effective_axis_size()?;
        let output_type = match operation.options.mode {
            CollectiveMode::Untiled => {
                let Some(dimension) = dimensions.get(operation.scatter_axis) else {
                    return Err(TypeError::invalid(format!(
                        "`parallel_sum_scatter` scatter axis {} is out of bounds for rank {}",
                        operation.scatter_axis,
                        dimensions.len(),
                    )));
                };
                if *dimension != effective_axis_size {
                    return Err(TypeError::invalid(format!(
                        "`parallel_sum_scatter` untiled scatter axis {} size {dimension} must equal group size \
                         {effective_axis_size}",
                        operation.scatter_axis,
                    )));
                }
                Ok::<_, TypeError>(input_type.without_dimension(operation.scatter_axis)?.0)
            }
            CollectiveMode::Tiled => {
                let mut output_dimensions = dimensions;
                let Some(dimension) = output_dimensions.get_mut(operation.scatter_axis) else {
                    return Err(TypeError::invalid(format!(
                        "`parallel_sum_scatter` scatter axis {} is out of bounds for rank {}",
                        operation.scatter_axis,
                        output_dimensions.len(),
                    )));
                };
                if *dimension % effective_axis_size != 0 {
                    return Err(TypeError::invalid(format!(
                        "`parallel_sum_scatter` scatter axis {} size {} is not divisible by group size {}",
                        operation.scatter_axis,
                        *dimension,
                        effective_axis_size,
                    )));
                }
                *dimension /= effective_axis_size;
                shape_changing_collective_output_type(
                    PARALLEL_SUM_SCATTER_OPERATION_NAME,
                    input_type,
                    output_dimensions,
                )
            }
        }?;
        parallel_sum_scatter_output_type(input_type, output_type, operation)
    },
}

impl ParallelSumScatterOperation {
    /// Returns the axis of the operand along which the summed result is scattered across the participants.
    #[inline]
    pub fn scatter_axis(&self) -> usize {
        self.scatter_axis
    }

    /// Returns the shared rank and participant-group semantics.
    #[inline]
    pub fn options(&self) -> &CollectiveOptions {
        &self.options
    }

    /// Returns the participant count used for result-shape arithmetic.
    #[inline]
    pub fn effective_axis_size(&self) -> Result<usize, TypeError> {
        self.options.effective_axis_size(PARALLEL_SUM_SCATTER_OPERATION_NAME, self.axis_size)
    }
}

impl_shape_changing_collective_member_operation!(
    ParallelSumScatterOperation,
    infer_explicit_parallel_sum_scatter_output_types
);

/// Stages a sum-scatter with first-class dynamic tiled extents and rank-changing untiled semantics.
pub trait ParallelSumScatter: Sized {
    /// Sums participants and consumes `scatter_axis`, whose extent must equal the effective participant count.
    #[inline]
    fn parallel_sum_scatter(&self, axis_name: &str, scatter_axis: usize) -> Result<Self, ProgramError> {
        self.parallel_sum_scatter_with_options(axis_name, scatter_axis, CollectiveOptions::default())
    }

    /// Sums participants and scatters equal chunks along the existing `scatter_axis`.
    #[inline]
    fn parallel_sum_scatter_tiled(&self, axis_name: &str, scatter_axis: usize) -> Result<Self, ProgramError> {
        self.parallel_sum_scatter_with_options(axis_name, scatter_axis, CollectiveOptions::new(CollectiveMode::Tiled))
    }

    /// Sums and scatters participants using explicit shape and grouping semantics.
    fn parallel_sum_scatter_with_options(
        &self,
        axis_name: &str,
        scatter_axis: usize,
        options: CollectiveOptions,
    ) -> Result<Self, ProgramError>;
}

impl<V> ParallelSumScatter for V
where
    V: Value<Type = ArrayIrType> + DimensionSize<V> + ValueProjection<DimensionType>,
    V::DispatchDomain: Context<Type = ArrayIrType> + NamedAxes,
    <V::DispatchDomain as Domain>::Constant: From<DimensionValue>,
    <V::DispatchDomain as Domain>::Operation: From<ParallelSumScatterOperation>,
    <V as ValueProjection<DimensionType>>::Projected: DimensionRequirement + Div,
{
    fn parallel_sum_scatter_with_options(
        &self,
        axis_name: &str,
        scatter_axis: usize,
        options: CollectiveOptions,
    ) -> Result<Self, ProgramError> {
        let context = self.dispatch_domain();
        let axis_size = resolve_named_axis_size(&context, axis_name)?;
        let effective_axis_size = options.effective_axis_size(PARALLEL_SUM_SCATTER_OPERATION_NAME, axis_size)?;
        let operation =
            ParallelSumScatterOperation::new(axis_name.to_string(), axis_size, scatter_axis, options.clone());
        let mut output_extents = collective_input_extents(&context, self)?;
        if scatter_axis >= output_extents.len() {
            return Err(TypeError::invalid(format!(
                "`parallel_sum_scatter` scatter axis {scatter_axis} is out of bounds for rank {}",
                output_extents.len(),
            ))
            .into());
        }
        match options.mode {
            CollectiveMode::Untiled => {
                require_collective_axis_extent(&context, &output_extents[scatter_axis], effective_axis_size)?;
                output_extents.remove(scatter_axis);
            }
            CollectiveMode::Tiled => {
                output_extents[scatter_axis] =
                    divided_collective_extent(&context, &output_extents[scatter_axis], effective_axis_size)?;
            }
        };
        let inputs = std::iter::once(self.clone()).chain(output_extents).collect::<Vec<_>>();
        Ok(context.bind(operation, Vec::new(), inputs.as_slice())?.remove(0))
    }
}

impl<V> ParallelSumScatter for ProjectedValue<ArrayType, V>
where
    V: ParallelSumScatter + ValueProjection<ArrayType, Projected = ProjectedValue<ArrayType, V>>,
{
    fn parallel_sum_scatter_with_options(
        &self,
        axis_name: &str,
        scatter_axis: usize,
        options: CollectiveOptions,
    ) -> Result<Self, ProgramError> {
        self.value()
            .parallel_sum_scatter_with_options(axis_name, scatter_axis, options)?
            .into_projected()
            .map_err(Into::into)
    }
}

// Mixed array IR JVP for sum-scatter. Explicit output extents are retained as ordinary residual values, and
// the transposed linear region applies varying all-gather to the output cotangent.
impl<C> MemberDifferentiableOperation<C> for ParallelSumScatterOperation
where
    C: Context<Type = ArrayIrType>,
    C::Operation: From<AllGatherOperation>
        + From<DimensionSizeOperation>
        + From<LinearCallOperation<ArrayIrType>>
        + From<ParallelSumScatterOperation>
        + From<ConstantOperation<DimensionValue>>
        + OperationProjection<DimensionType, Projected = DimensionOperation<DimensionValue>>,
{
    fn jvp_in_parent<D: DifferentiationDriver<C>>(
        &self,
        context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        jvp_shape_changing_collective_with_adjoint(
            self,
            AllGatherOperation::new(
                self.axis_name().to_string(),
                self.axis_size(),
                self.scatter_axis(),
                self.options().clone(),
                AllGatherOutputVariance::Varying,
            ),
            context,
            inputs,
        )
    }
}

/// Returns the physical scatter axis and mapped result axis for a forwarded sum-scatter.
fn forwarded_parallel_sum_scatter_axes(mode: CollectiveMode, scatter_axis: usize, batch_axis: usize) -> (usize, usize) {
    let physical_scatter_axis = scatter_axis + usize::from(scatter_axis >= batch_axis);
    let output_batch_axis = match mode {
        CollectiveMode::Tiled => batch_axis,
        CollectiveMode::Untiled if scatter_axis < batch_axis => batch_axis - 1,
        CollectiveMode::Untiled => batch_axis,
    };
    (physical_scatter_axis, output_batch_axis)
}

/// Applies the matching-axis sum-scatter batching semantics over the policy-selected extent representation.
fn batch_parallel_sum_scatter_matching_axis<C, P>(
    operation: &ParallelSumScatterOperation,
    context: &BatchingContext<C, ArrayBatching<P>>,
    input: &ArrayBatch<C::Value>,
    logical_input_rank: usize,
    output_extents: Vec<P::ShapeExtent>,
    output_sharding: Option<Sharding>,
) -> Result<ArrayBatch<C::Value>, BatchingError>
where
    C: Context<Type = ArrayType>,
    C::Value: Reduce + Transpose,
    P: CollectiveBatchingPolicy<C>,
{
    if operation.options.axis_index_groups.is_some() {
        return Err(BatchingError::UnsupportedOperation {
            message: "`parallel_sum_scatter` axis index groups are not supported when a batch transform binds the \
                      collective axis"
                .to_string(),
        });
    }
    if operation.scatter_axis >= logical_input_rank {
        return Err(BatchingError::UnsupportedOperation {
            message: format!(
                "`parallel_sum_scatter` scatter axis {} is out of bounds for rank {logical_input_rank}",
                operation.scatter_axis,
            ),
        });
    }

    let axis_extent = P::collective_axis_extent(
        context,
        PARALLEL_SUM_SCATTER_OPERATION_NAME,
        &operation.axis_name,
        operation.axis_size,
    )?;

    let mut input_extents = output_extents.clone();
    match operation.options.mode {
        CollectiveMode::Untiled => input_extents.insert(operation.scatter_axis, axis_extent.clone()),
        CollectiveMode::Tiled => {
            input_extents[operation.scatter_axis] = output_extents[operation.scatter_axis].mul(&axis_extent)?;
        }
    }
    let input = P::match_collective_axis(context, input, input_extents.as_slice())?;
    let summed = input.into_value().reduce(&[0], ReductionKind::Sum);
    let scattered = match operation.options.mode {
        CollectiveMode::Untiled => summed.move_axis(operation.scatter_axis, 0)?,
        CollectiveMode::Tiled => {
            let mut split_extents = output_extents.clone();
            split_extents.insert(operation.scatter_axis, axis_extent.clone());
            P::reshape_collective(context, summed, split_extents.as_slice(), None)?
                .move_axis(operation.scatter_axis, 0)?
        }
    };
    let mut physical_output_extents = Vec::with_capacity(output_extents.len() + 1);
    physical_output_extents.push(axis_extent);
    physical_output_extents.extend(output_extents);
    let physical_output_sharding = output_sharding
        .map(|sharding| lift_output_sharding_for_leading_batch_axis(&sharding, context.axis_sharding().clone()))
        .transpose()?;
    let output =
        P::reshape_collective(context, scattered, physical_output_extents.as_slice(), physical_output_sharding)?;
    ArrayBatch::new(output, BatchAxis::from_position(0))
}

// Batching rule for [`ParallelSumScatterOperation`]. A matching `batch` level consumes the mapped batch axis by
// summing over it and re-mapping the chunks of the per-item `scatter_axis` onto it: the sum's `scatter_axis` is split
// into `(b, d_s / b)` chunks and the new chunk axis becomes the output batch axis, so batch item `i` receives chunk
// `i` of the sum. A non-matching level forwards the collective untouched to the parent context via
// [`forward_collective_to_parent`].
impl<C, P: CollectiveBatchingPolicy<C>> BatchableOperation<C, ArrayBatching<P>> for ParallelSumScatterOperation
where
    C: Context<Type = ArrayType>,
    C::Operation: From<ParallelSumScatterOperation>,
    <C as Domain>::Value: Reduce + Transpose,
{
    fn batch<D: BatchingDriver<C, ArrayBatching<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatching<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<<C as Domain>::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatching<P>>, BatchingError> {
        reject_ragged_collective_inputs(self.name(), inputs)?;
        if context.axis_name() != Some(self.axis_name.as_str()) {
            let [input] = inputs else {
                return Err(ProgramError::InvalidInputCount { expected: 1, actual: inputs.len() }.into());
            };
            let Some(batch_axis) = input.batch_axis_position() else {
                return Ok(forward_collective_to_parent(context, C::Operation::from(self.clone()), inputs)?.into());
            };
            let (scatter_axis, output_batch_axis) =
                forwarded_parallel_sum_scatter_axes(self.options.mode, self.scatter_axis, batch_axis);
            let operation = Self::new(self.axis_name.clone(), self.axis_size, scatter_axis, self.options.clone());
            return Ok(forward_shape_changing_collective(
                context,
                C::Operation::from(operation),
                input,
                Some(output_batch_axis),
            )?
            .into());
        }
        let [input] = inputs else {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: inputs.len() }.into());
        };
        let input_type = input.unbatched_type();
        let mut output_types = self.infer_output_types(std::slice::from_ref(&input_type), &[])?;
        let output_type = output_types.remove(0);
        let output_extents = output_type
            .shape()
            .dimensions()
            .iter()
            .map(|dimension| P::collective_extent_from_dimension(context, dimension))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(vec![batch_parallel_sum_scatter_matching_axis::<C, P>(
            self,
            context,
            input,
            input_type.rank(),
            output_extents,
            output_type.sharding().cloned(),
        )?]
        .into())
    }
}

// Batching rule for explicit-extent [`ParallelSumScatterOperation`]. The explicit result extents remain the only
// source for dynamic reshape geometry while matching-axis array mechanics reuse the homogeneous collective kernel.
impl<C> MemberBatchableOperation<C, ArrayIrBatching> for ParallelSumScatterOperation
where
    C: Context<
            Type = ArrayIrType,
            Operation: From<ParallelSumScatterOperation>
                           + From<DynamicBroadcastOperation>
                           + From<ConstantOperation<DimensionValue>>
                           + From<DimensionSizeOperation>
                           + From<DynamicReshapeOperation>
                           + OperationProjection<ArrayType>,
        >,
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Value: ValueProjection<ArrayType, Projected: Reduce + Transpose + Value<Type = ArrayType>>
        + ValueProjection<DimensionType>,
    <C::Value as ValueProjection<DimensionType>>::Projected:
        DimensionRequirement + Div + Mul + Value<Type = DimensionType>,
{
    fn batch_in_parent<D: BatchingDriver<C, ArrayIrBatching>>(
        &self,
        context: &BatchingContext<C, ArrayIrBatching>,
        _driver: &D,
        inputs: &[ArrayIrBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayIrBatching>, BatchingError> {
        let (array, output_extents) = explicit_collective_inputs(inputs)?;
        if let Some(ragged_axis) = array.ragged_axes().first() {
            return Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "`{}` does not support bounded ragged dimension `{}` on operand 0",
                    self.name(),
                    ragged_axis.dimension(),
                ),
            });
        }
        validate_explicit_collective_output_extents(output_extents)?;
        let logical_input_types = inputs.iter().map(|input| input.unbatched_type().clone()).collect::<Vec<_>>();
        let mut logical_output_types =
            infer_explicit_parallel_sum_scatter_output_types(self, logical_input_types.as_slice())?;
        let logical_output_type = <&ArrayType>::try_from(&logical_output_types.remove(0))?.clone();

        if context.axis_name() != Some(self.axis_name()) {
            if array.batch_axis().is_replicated() {
                return Ok(forward_explicit_collective(self.clone(), context, array, output_extents, None)?.into());
            }
            let input_batch_axis = array.batch_axis_position().unwrap();
            let (physical_scatter_axis, output_batch_axis) =
                forwarded_parallel_sum_scatter_axes(self.options().mode(), self.scatter_axis(), input_batch_axis);
            let operation = Self::new(
                self.axis_name().to_string(),
                self.axis_size(),
                physical_scatter_axis,
                self.options().clone(),
            );
            return Ok(forward_explicit_collective(
                operation,
                context,
                array,
                output_extents,
                Some(output_batch_axis),
            )?
            .into());
        }

        let array = ArrayBatch::new(
            <C::Value as ValueProjection<ArrayType>>::into_projected(array.value().clone())?,
            array.batch_axis(),
        )?;
        let output_extents = output_extents
            .iter()
            .map(|extent| <C::Value as ValueProjection<DimensionType>>::into_projected(extent.value().clone()))
            .collect::<Result<Vec<_>, _>>()?;
        let projected_context = BatchingContext::<_, ArrayBatching<DynamicArrayBatchingPolicy>>::with_policy(
            ProjectedContext::new(context.parent().clone()),
            context.axis_extent().clone(),
        )
        .with_axis_name(context.axis_name().map(str::to_string))
        .with_axis_sharding(context.axis_sharding().clone());
        let output = batch_parallel_sum_scatter_matching_axis::<_, DynamicArrayBatchingPolicy>(
            self,
            &projected_context,
            &array,
            array.unbatched_type().rank(),
            output_extents,
            logical_output_type.sharding().cloned(),
        )?;
        let batch_axis = output.batch_axis();
        Ok(ArrayIrBatch::new(<C::Value as ValueProjection<ArrayType>>::from_projected(output.into_value()), batch_axis)
            .map(|output| vec![output])?
            .into())
    }
}

// Transpose rule for [`ParallelSumScatterOperation`]. A sum-scatter is the adjoint of a varying all-gather with the
// same mode, axis, and participant groups, so the operand cotangent is an [`AllGatherOperation`] of the output
// cotangent.
impl<V, O> TransposableOperation<V, O> for ParallelSumScatterOperation
where
    V: Value<Type = ArrayType>,
    O: Operation<Type = ArrayType> + From<AllGatherOperation>,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TracingContext<V, O>,
        _driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        transpose_shape_changing_collective(
            context,
            inputs,
            outputs,
            AllGatherOperation::new(
                self.axis_name.clone(),
                self.axis_size,
                self.scatter_axis,
                self.options.clone(),
                AllGatherOutputVariance::Varying,
            ),
        )
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrValue, DataType, DimensionBounds, DimensionType, DimensionVariable, RaggedAxis,
    };
    use crate::batching::{BatchAxis, BatchAxisSpecification, BatchingContext, batch};
    use crate::contexts::EagerContext;
    use crate::operations::collectives::tests::f32_vector;

    use super::*;

    #[test]
    fn test_parallel_sum_scatter_type_inference() {
        use crate::macros::check_operation_type_inference;

        check_operation_type_inference!(
            operation = ParallelSumScatterOperation::new("x".to_string(), 4, 0, CollectiveOptions::tiled()),
            cases = [
                {
                    input_types = [f32_vector(8)],
                    output_types = [f32_vector(2)],
                },
                {
                    input_types = [f32_vector(6)],
                    error = "`parallel_sum_scatter` scatter axis 0 size 6 is not divisible by group size 4",
                },
                {
                    input_types = [ArrayType::scalar(DataType::F32)],
                    error = "`parallel_sum_scatter` scatter axis 0 is out of bounds for rank 0",
                },
            ],
        );
    }

    #[test]
    fn test_parallel_sum_scatter_forwarded_axes_account_for_the_mapped_axis() {
        assert_eq!(forwarded_parallel_sum_scatter_axes(CollectiveMode::Tiled, 0, 0), (1, 0));
        assert_eq!(forwarded_parallel_sum_scatter_axes(CollectiveMode::Tiled, 0, 1), (0, 1));
        assert_eq!(forwarded_parallel_sum_scatter_axes(CollectiveMode::Untiled, 0, 1), (0, 0));
        assert_eq!(forwarded_parallel_sum_scatter_axes(CollectiveMode::Untiled, 1, 0), (2, 0));
    }

    #[test]
    fn test_array_ir_parallel_sum_scatter_rejects_ragged_input_before_mapped_extents() {
        let variable = DimensionVariable::new("length", DimensionBounds::new(0, Some(5)).unwrap());
        let extents = ArrayIrValue::Array(Array::vector(vec![2_i32, 4]));
        let input = ArrayIrBatch::new(ArrayIrValue::Array(Array::matrix(2, 4, vec![1.0_f32; 8])), BatchAxis::new(0))
            .unwrap()
            .with_ragged_axes(vec![RaggedAxis::new(1, extents.clone(), variable.clone(), vec![0])])
            .unwrap();
        let output_extent =
            ArrayIrBatch::mapped_dimension(extents, BatchAxis::new(0), DimensionType::new(variable)).unwrap();
        let context = BatchingContext::new(
            EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new(),
            ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()),
        )
        .with_axis_name("x".to_string());

        assert_eq!(
            ParallelSumScatterOperation::new("x".to_string(), 2, 0, CollectiveOptions::tiled()).batch_in_parent(
                &context,
                &crate::EmptyRegionDriver,
                &[input, output_extent],
            ),
            Err(BatchingError::UnsupportedOperation {
                message: "`parallel_sum_scatter` does not support bounded ragged dimension `length` on operand 0"
                    .to_string(),
            }),
        );
    }

    #[test]
    fn test_parallel_sum_scatter_over_batched_axis_sums_and_scatters() {
        use crate::batching::BatchingTracer;

        // The batch binds the axis `"x"` that the `parallel_sum_scatter` names, so the matching batching rule sums
        // over the mapped axis and re-maps the chunks of `scatter_axis` onto it. With items `[1, 2, 3, 4]` and
        // `[10, 20, 30, 40]` the sum is `[11, 22, 33, 44]`, so item 0 receives `[11, 22]` and item 1 receives
        // `[33, 44]`, matching the verified cross-device `shard_map` execution semantics of StableHLO's
        // `reduce_scatter`.
        let x = Array::matrix(2, 4, vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]);
        let output: ArrayIrValue<Array> = batch(
            |item: BatchingTracer<EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>, ArrayIrBatching>| {
                item.parallel_sum_scatter_tiled("x", 0)
            },
            ArrayIrValue::Array(x),
            BatchAxis::new(0),
            BatchAxis::new(0),
            BatchAxisSpecification::named("x"),
        )
        .unwrap();
        assert_eq!(
            output.r#type().into_owned(),
            ArrayIrType::Array(ArrayType::new(
                DataType::F64,
                Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]),
            )),
        );
        let ArrayIrValue::Array(output) = output else {
            panic!("`parallel_sum_scatter` must preserve the array member kind");
        };
        assert_eq!(output.to_f64s(), vec![11.0, 22.0, 33.0, 44.0]);
    }
}
