use crate::arrays::batching::DynamicArrayExtentBatchingPolicy;
use crate::arrays::{
    ArrayBatch, ArrayBatchingPolicy, ArrayIrBatch, ArrayIrBatchingPolicy, ArrayIrType, ArrayType, DataType, Dimension,
    DimensionOperation, DimensionType, DimensionValue, DimensionVariable, MeshAxisType, Shape, Sharding,
};
use crate::axes::{NamedAxes, NamedAxis};
use crate::batching::{
    BatchAxis, BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError,
    MemberBatchableOperation,
};
use crate::contexts::{Context, Domain, ProjectedContext};
use crate::differentiation::{
    DifferentiationContext, DifferentiationDriver, DifferentiationDual, DifferentiationError, DifferentiationPolicy,
    MemberDifferentiableOperation,
};
use crate::interpretation::{InterpretationDriver, MemberInterpretableOperation};
use crate::macros::check_count;
use crate::operations::arithmetic::{Div, Mul, Rem};
use crate::operations::assertions::Assert;
use crate::operations::comparisons::Compare;
use crate::operations::constants::constant::{ConstantOperation, DimensionConstant};
use crate::operations::differentiation::linear_call::LinearCallOperation;
use crate::operations::dimensions::dimension_max::DimensionMax;
use crate::operations::dimensions::dimension_size::{DimensionSize, DimensionSizeOperation};
use crate::operations::manipulation::broadcasting::{DynamicBroadcast, DynamicBroadcastOperation};
use crate::operations::manipulation::reshaping::{DynamicReshapeOperation, Reshape};
use crate::operations::manipulation::transposition::Transpose;
use crate::operations::reductions::{Reduce, ReductionKind};
use crate::programs::{
    MemberOperation, Operation, OperationProjection, ProgramError, ProjectedValue, RegionInterface, TypeError,
    TypeIdentityRenaming, Typed, Value, ValueProjection,
};

use super::all_gather::{AllGatherOperation, AllGatherOutputVariance};
use super::parallel_vary::ParallelVary;
use super::{
    CollectiveArrayExtentBatchingPolicy, CollectiveMode, CollectiveOptions, collective_input_extents,
    collective_output_extents, define_linear_collective_operation, explicit_collective_inputs,
    forward_explicit_collective, forward_shape_changing_collective, impl_differentiable_linear_collective_operation,
    impl_shape_changing_collective_member_operation, infer_explicit_shape_changing_collective_output_type,
    infer_linear_collective_operation_output_type, jvp_shape_changing_collective_with_adjoint, resolve_named_axis_size,
    validate_explicit_collective_output_extents,
};

/// Canonical operation name for [`ParallelSumScatterOperation`].
pub const PARALLEL_SUM_SCATTER_OPERATION_NAME: &str = "parallel_sum_scatter";

define_linear_collective_operation!(
    /// [`Operation`] that sums every participant's input across the named axis and scatters the sum along
    /// `scatter_axis`, so that every participant receives only its own chunk of the sum. This is the analogue of
    /// JAX's [`jax.lax.psum_scatter`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.psum_scatter.html) and
    /// of StableHLO's [`reduce_scatter`](https://openxla.org/stablehlo/spec#reduce_scatter) with a sum reduction.
    /// The [`CollectiveMode`] of its options selects the output shape over a group of `n` participants:
    ///
    ///   - [`CollectiveMode::Untiled`] requires `scatter_axis` to have extent `n` and removes it, so that participant
    ///     `i` receives row `i` of the sum.
    ///   - [`CollectiveMode::Tiled`] requires the extent of `scatter_axis` to be divisible by `n` and divides it by
    ///     `n`, so that participant `i` receives the `i`-th contiguous chunk of the sum.
    ///
    /// Inputs must be numeric (or structural zeros), and the sum accumulates in the input element type, without the
    /// `f32` accumulation that [`ReductionKind::Sum`] documents for narrower floating-point `reduce` inputs.
    /// Participant groups (refer to [`CollectiveOptions`]) restrict the sum and the scatter to each group.
    ///
    /// Over a manual mesh axis, every participant receives a different chunk, so the input must vary over the axis
    /// (refer to [`ParallelVary`]) and the output varies over it as well. An input that is instead unreduced over
    /// the operation's own axis (e.g., the cotangent of a reduced [`AllGatherOperation`] result) has its pending
    /// cross-device sum completed by the exchange, and its output varies over the axis too. The collective is linear,
    /// and its transpose is a varying [`AllGatherOperation`] with the same mode, axis, and participant groups. Outside
    /// any binder, the single participant of a degenerate axis keeps its value, with the size-one scatter axis removed
    /// in untiled mode.
    ///
    /// A matching `batch` level consumes the mapped batch axis by summing over it and mapping the scattered chunks back
    /// onto it, so that batch item `i` receives chunk `i` of the sum and a value that is the same for every item counts
    /// once per item. A matching level rejects participant groups, and every `batch` level rejects bounded ragged
    /// inputs.
    ParallelSumScatterOperation,
    PARALLEL_SUM_SCATTER_OPERATION_NAME,
    fields = {
        /// Axis of the input along which the summed result is scattered across the participants.
        scatter_axis: usize,

        /// Shared rank and participant-group semantics.
        options: CollectiveOptions,
    },
    infer_output_type = |operation, input_type, dimensions| {
        let effective_axis_size = operation.effective_axis_size()?;
        let output_type = match operation.options.mode {
            CollectiveMode::Untiled => {
                let Some(dimension) = dimensions.get(operation.scatter_axis) else {
                    return Err(TypeError::invalid(format!(
                        "`{}` scatter axis {} is out of bounds for rank {}",
                        PARALLEL_SUM_SCATTER_OPERATION_NAME,
                        operation.scatter_axis,
                        dimensions.len(),
                    )));
                };
                if *dimension != effective_axis_size {
                    return Err(TypeError::invalid(format!(
                        "`{}` untiled scatter axis {} size {} must equal group size {}",
                        PARALLEL_SUM_SCATTER_OPERATION_NAME,
                        operation.scatter_axis,
                        dimension,
                        effective_axis_size,
                    )));
                }
                input_type.without_dimension(operation.scatter_axis)?.0
            }
            CollectiveMode::Tiled => {
                let mut output_dimensions = dimensions;
                let Some(dimension) = output_dimensions.get_mut(operation.scatter_axis) else {
                    return Err(TypeError::invalid(format!(
                        "`{}` scatter axis {} is out of bounds for rank {}",
                        PARALLEL_SUM_SCATTER_OPERATION_NAME,
                        operation.scatter_axis,
                        output_dimensions.len(),
                    )));
                };
                if *dimension % effective_axis_size != 0 {
                    return Err(TypeError::invalid(format!(
                        "`{}` scatter axis {} size {} is not divisible by group size {}",
                        PARALLEL_SUM_SCATTER_OPERATION_NAME,
                        operation.scatter_axis,
                        *dimension,
                        effective_axis_size,
                    )));
                }
                *dimension /= effective_axis_size;
                infer_linear_collective_operation_output_type(
                    PARALLEL_SUM_SCATTER_OPERATION_NAME,
                    input_type,
                    output_dimensions,
                )?
            }
        };
        parallel_sum_scatter_output_type(input_type, output_type, operation)
    },
    interpret<C> where C::Value: Reshape {
        |operation, input| {
            // A single participant sums only its own value. Untiled mode removes the size-one scatter axis,
            // which a reshape to the inferred output type expresses, while tiled mode leaves the shape unchanged.
            match operation.options.mode {
                CollectiveMode::Tiled => Ok(input.clone()),
                CollectiveMode::Untiled => {
                    let output_type = operation.infer_output_types(&[input.r#type().into_owned()], &[])?.remove(0);
                    input.reshape_with_output_sharding(output_type.shape().clone(), output_type.sharding().cloned())
                }
            }
        }
    },
);

// TODO(eaplatanios): Review from here onwards.

impl ParallelSumScatterOperation {
    /// Returns the axis of the input along which the summed result is scattered across the participants.
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

// Batching rule for [`ParallelSumScatterOperation`]. A matching `batch` level consumes the mapped batch axis by summing
// over it and re-mapping the chunks of the per-item `scatter_axis` onto it: the sum's `scatter_axis` is split into
// `(b, d_s / b)` chunks and the new chunk axis becomes the output batch axis, so batch item `i` receives chunk `i` of
// the sum. A non-matching level forwards the collective to the parent context, unchanged for a replicated input
// (through `BatchingContext::forward_to_parent`) and with its array axes shifted past the batch axis for a mapped one.
impl<C, P: CollectiveArrayExtentBatchingPolicy<C>> BatchableOperation<C, ArrayBatchingPolicy<P>>
    for ParallelSumScatterOperation
where
    C: Context<Type = ArrayType>,
    C::Operation: From<ParallelSumScatterOperation>,
    <C as Domain>::Value: Reduce + Transpose,
{
    fn batch<D: BatchingDriver<C, ArrayBatchingPolicy<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<<C as Domain>::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatchingPolicy<P>>, BatchingError> {
        ArrayBatch::reject_ragged_inputs(self, inputs)?;
        if context.axis_name() != Some(self.axis_name.as_str()) {
            return forward_shape_changing_collective(context, self, inputs, |batch_axis| {
                let (scatter_axis, output_batch_axis) =
                    forwarded_parallel_sum_scatter_axes(self.options.mode, self.scatter_axis, batch_axis);
                let operation = Self::new(self.axis_name.clone(), self.axis_size, scatter_axis, self.options.clone());
                (operation, output_batch_axis)
            });
        }
        let [input] = inputs else {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: inputs.len() }.into());
        };
        let input_type = input.unbatched_type();
        let (output_type, output_extents) = collective_output_extents(context, self, &input_type)?;
        Ok(vec![batch_parallel_sum_scatter_matching_axis::<C, P>(
            self,
            context,
            input,
            output_extents,
            output_type.sharding().cloned(),
        )?]
        .into())
    }
}

// Transpose rule for [`ParallelSumScatterOperation`]. A sum-scatter is the adjoint of a varying all-gather with the
// same mode, axis, and participant groups, so the input cotangent is an [`AllGatherOperation`] of the output cotangent.
impl_differentiable_linear_collective_operation! {
    ParallelSumScatterOperation,
    transpose = |operation| -> AllGatherOperation {
        AllGatherOperation::new(
            operation.axis_name.clone(),
            operation.axis_size,
            operation.scatter_axis,
            operation.options.clone(),
            AllGatherOutputVariance::Varying,
        )
    },
}

impl_shape_changing_collective_member_operation!(
    ParallelSumScatterOperation,
    infer_explicit_parallel_sum_scatter_output_types
);

// Batching rule for explicit-extent [`ParallelSumScatterOperation`]. The explicit result extents remain the only
// source for dynamic reshape geometry while matching-axis array mechanics reuse the homogeneous collective kernel.
impl<C> MemberBatchableOperation<C, ArrayIrBatchingPolicy> for ParallelSumScatterOperation
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
    C::Value: Assert
        + DimensionSize
        + DynamicBroadcast
        + ValueProjection<ArrayType, Projected: Reduce + Transpose + Value<Type = ArrayType>>
        + ValueProjection<DimensionType>,
    <C::Value as ValueProjection<DimensionType>>::Projected:
        Compare<C::Value> + DimensionMax + Rem + Div + Mul + Value<Type = DimensionType>,
{
    fn batch_in_parent<D: BatchingDriver<C, ArrayIrBatchingPolicy>>(
        &self,
        context: &BatchingContext<C, ArrayIrBatchingPolicy>,
        _driver: &D,
        inputs: &[ArrayIrBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayIrBatchingPolicy>, BatchingError> {
        let (array, output_extents) = explicit_collective_inputs(inputs)?;
        ArrayIrBatch::reject_ragged_inputs(self, inputs)?;
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
        let projected_context =
            BatchingContext::<_, ArrayBatchingPolicy<DynamicArrayExtentBatchingPolicy>>::with_policy(
                ProjectedContext::new(context.parent().clone()),
                context.axis_extent().clone(),
            )
            .with_axis_name(context.axis_name().map(str::to_string))
            .with_axis_sharding(context.axis_sharding().clone());
        let output = batch_parallel_sum_scatter_matching_axis::<_, DynamicArrayExtentBatchingPolicy>(
            self,
            &projected_context,
            &array,
            output_extents,
            logical_output_type.sharding().cloned(),
        )?;
        let batch_axis = output.batch_axis();
        Ok(ArrayIrBatch::new(<C::Value as ValueProjection<ArrayType>>::from_projected(output.into_value()), batch_axis)
            .map(|output| vec![output])?
            .into())
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
    fn jvp_in_parent<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        jvp_shape_changing_collective_with_adjoint(self, self.adjoint()?, context, inputs)
    }
}

/// Represents the ability to sum values across the participants of a named axis and scatter the sum, so that every
/// participant receives only its own chunk, by staging a [`ParallelSumScatterOperation`]. This is the analogue of
/// [JAX's `psum_scatter`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.psum_scatter.html), whose default
/// `tiled = False` corresponds to [`ParallelSumScatter::parallel_sum_scatter`] and whose `tiled = True` corresponds to
/// [`ParallelSumScatter::parallel_sum_scatter_tiled`]. Refer to [`ParallelSumScatterOperation`] for the semantics and
/// transformation rules.
///
/// Over a manual mesh axis, an input that neither varies over the axis nor carries a pending cross-device sum over it
/// is first made varying through [`ParallelVary`], so that every device's copy is counted, as with
/// [`ParallelReduce::parallel_reduce`](super::ParallelReduce::parallel_reduce). The output extents are staged as
/// explicit extent values, and a runtime assertion checks every extent that is not statically known.
pub trait ParallelSumScatter: Sized {
    /// Returns the sum of this value across the participants of the named axis `axis_name`, scattered along
    /// `scatter_axis`. The extent of `scatter_axis` must equal the number of participants, and the axis is removed, so
    /// that participant `i` receives row `i` of the sum.
    ///
    /// # Parameters
    ///
    ///   - `axis_name`: Name of an axis bound by an enclosing `batch` level or manual region.
    ///   - `scatter_axis`: Axis of this value along which the sum is scattered.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`ParallelSumScatter::parallel_sum_scatter_with_options`].
    #[inline]
    fn parallel_sum_scatter(&self, axis_name: &str, scatter_axis: usize) -> Result<Self, ProgramError> {
        self.parallel_sum_scatter_with_options(axis_name, scatter_axis, CollectiveOptions::default())
    }

    /// Returns the sum of this value across the participants of the named axis `axis_name`, scattered in equal
    /// contiguous chunks along `scatter_axis`. The extent of `scatter_axis` must be divisible by the number of
    /// participants, and participant `i` receives chunk `i` of the sum.
    ///
    /// # Parameters
    ///
    ///   - `axis_name`: Name of an axis bound by an enclosing `batch` level or manual region.
    ///   - `scatter_axis`: Axis of this value along which the sum is scattered.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`ParallelSumScatter::parallel_sum_scatter_with_options`].
    #[inline]
    fn parallel_sum_scatter_tiled(&self, axis_name: &str, scatter_axis: usize) -> Result<Self, ProgramError> {
        self.parallel_sum_scatter_with_options(axis_name, scatter_axis, CollectiveOptions::new(CollectiveMode::Tiled))
    }

    /// Returns the sum of this value across the participants of the named axis `axis_name`, scattered along
    /// `scatter_axis` with the tiling mode and participant groups of `options`.
    ///
    /// # Parameters
    ///
    ///   - `axis_name`: Name of an axis bound by an enclosing `batch` level or manual region.
    ///   - `scatter_axis`: Axis of this value along which the sum is scattered.
    ///   - `options`: [`CollectiveMode`] and optional participant groups of the collective.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError::Axis`] error wrapping [`AxisError::UnboundAxisName`](crate::axes::AxisError) when no
    /// enclosing binder binds `axis_name`, and a [`ProgramError`] if `scatter_axis` is out of bounds, if its extent
    /// does not fit the tiling mode, if the participant groups are invalid, if this value is not numeric, or if it
    /// carries unreduced axes other than `axis_name`.
    fn parallel_sum_scatter_with_options(
        &self,
        axis_name: &str,
        scatter_axis: usize,
        options: CollectiveOptions,
    ) -> Result<Self, ProgramError>;
}

// A composite value binds a `ParallelSumScatterOperation` through its own context, followed by one explicit extent
// value per output axis, which also asserts at runtime that dynamic extents fit the tiling mode. Over a manual mesh
// axis, an input that neither varies over the axis nor is unreduced over it is first made varying through its array
// view, exactly as JAX's `psum_scatter` does, so that every device's copy is counted.
impl<V> ParallelSumScatter for V
where
    V: Value<Type = ArrayIrType>
        + Assert
        + DimensionSize<V>
        + ValueProjection<DimensionType>
        + ValueProjection<ArrayType, Projected = ProjectedValue<ArrayType, V>>,
    V::DispatchDomain: Context<Type = ArrayIrType> + NamedAxes,
    V::DispatchDomain: DimensionConstant,
    <V::DispatchDomain as Domain>::Operation: From<ParallelSumScatterOperation>,
    <V as ValueProjection<DimensionType>>::Projected: Value<Type = DimensionType> + Compare<V> + Rem + Div,
    ProjectedValue<ArrayType, V>: ParallelVary,
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
        let mut input = self.clone();
        if matches!(context.named_axis(axis_name), Some(NamedAxis::Mesh { .. })) {
            let array = ValueProjection::<ArrayType>::into_projected(self.clone())?;
            if !array.r#type().sharding().is_some_and(|sharding| {
                sharding.varying_manual_axes().contains(axis_name) || sharding.unreduced_axes().contains(axis_name)
            }) {
                input = <V as ValueProjection<ArrayType>>::from_projected(array.parallel_vary(axis_name)?);
            }
        }
        let operation =
            ParallelSumScatterOperation::new(axis_name.to_string(), axis_size, scatter_axis, options.clone());
        let mut output_extents = collective_input_extents(&input)?;
        if scatter_axis >= output_extents.len() {
            return Err(TypeError::invalid(format!(
                "`{PARALLEL_SUM_SCATTER_OPERATION_NAME}` scatter axis {scatter_axis} is out of bounds for rank {}",
                output_extents.len(),
            ))
            .into());
        }
        match options.mode {
            CollectiveMode::Untiled => {
                output_extents[scatter_axis].require_equal(&context, effective_axis_size)?;
                output_extents.remove(scatter_axis);
            }
            CollectiveMode::Tiled => {
                output_extents[scatter_axis] = output_extents[scatter_axis].divided(&context, effective_axis_size)?;
            }
        };
        let output_extents =
            output_extents.into_iter().map(|extent| extent.stage(&context)).collect::<Result<Vec<_>, _>>()?;
        let inputs = std::iter::once(input).chain(output_extents).collect::<Vec<_>>();
        let mut outputs = context.bind(operation, Vec::new(), inputs.as_slice())?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
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

/// Validates the element data type and the manual variation of a sum-scatter input, and applies the reduction-state
/// transition to the shape-only `output_type` shared by the static and explicit-extent inference paths. Ordinary inputs
/// preserve their variance metadata. An input that is unreduced over the scattered manual axis is the cotangent of a
/// reduced all-gather result, so the sum-scatter consumes that pending reduction and returns a value that varies over
/// the manual axis.
fn parallel_sum_scatter_output_type(
    input_type: &ArrayType,
    mut output_type: ArrayType,
    operation: &ParallelSumScatterOperation,
) -> Result<ArrayType, TypeError> {
    let data_type = input_type.data_type();
    if !data_type.is_numeric() && data_type != DataType::Zero {
        return Err(TypeError::invalid(format!(
            "`{PARALLEL_SUM_SCATTER_OPERATION_NAME}` requires numeric inputs but got `{data_type}`",
        )));
    }
    if input_type.unreduced_axes().is_empty() {
        // Every participant of a manual mesh axis receives a different chunk, so an input that is still invariant over
        // that axis would yield an output whose type wrongly claims that it is invariant.
        if let Some(sharding) = input_type.sharding()
            && sharding.mesh().axis_type(operation.axis_name()) == Some(MeshAxisType::Manual)
            && !sharding.varying_manual_axes().contains(operation.axis_name())
        {
            return Err(TypeError::invalid(format!(
                "`{PARALLEL_SUM_SCATTER_OPERATION_NAME}` input must vary over manual axis `{}`; pass an invariant \
                 value through `parallel_vary` first so that every copy is counted",
                operation.axis_name(),
            )));
        }
        return Ok(output_type);
    }
    if input_type.unreduced_axes().len() != 1 || !input_type.unreduced_axes().contains(operation.axis_name()) {
        return Err(TypeError::invalid(format!(
            "`{PARALLEL_SUM_SCATTER_OPERATION_NAME}` only supports an unreduced input over its own axis `{}`",
            operation.axis_name(),
        )));
    }
    // Unreduced axes require a sharding, and the shape-only output type preserves the input sharding.
    let input_sharding = input_type.sharding().unwrap();
    let mut varying_axes = input_sharding.varying_manual_axes().clone();
    varying_axes.insert(operation.axis_name().to_string());
    let output_sharding = output_type.sharding().unwrap().clone();
    output_type.sharding = Some(
        output_sharding
            .with_unreduced_axes(Vec::<String>::new())
            .and_then(|sharding| sharding.with_varying_manual_axes(varying_axes))
            .map_err(|error| TypeError::invalid(error.to_string()))?,
    );
    Ok(output_type)
}

/// Infers the output type of a sum-scatter in the composite array/dimension family, whose array input is followed by
/// one explicit extent per output axis. It applies the same contract as static type inference, checking the extents
/// that are statically known and leaving dynamic extents to the runtime assertions that the capability stages.
pub(crate) fn infer_explicit_parallel_sum_scatter_output_types(
    operation: &ParallelSumScatterOperation,
    input_types: &[ArrayIrType],
) -> Result<Vec<ArrayIrType>, TypeError> {
    let effective_axis_size = operation.effective_axis_size()?;
    let Some(input_type) = input_types.first() else {
        return Err(TypeError::invalid(format!(
            "`{PARALLEL_SUM_SCATTER_OPERATION_NAME}` expects an array followed by its output extents",
        )));
    };
    let input_type = <&ArrayType>::try_from(input_type)?;
    if operation.options.mode == CollectiveMode::Untiled {
        let Some(input_extent) = input_type.shape().dimensions().get(operation.scatter_axis) else {
            return Err(TypeError::invalid(format!(
                "`{PARALLEL_SUM_SCATTER_OPERATION_NAME}` scatter axis {} is out of bounds for rank {}",
                operation.scatter_axis,
                input_type.rank(),
            )));
        };
        if let Dimension::Static(input_extent) = input_extent
            && *input_extent != effective_axis_size
        {
            return Err(TypeError::invalid(format!(
                "`{PARALLEL_SUM_SCATTER_OPERATION_NAME}` untiled scatter axis {} size {input_extent} must equal \
                 group size {effective_axis_size}",
                operation.scatter_axis,
            )));
        }
        let base_output_type = input_type.without_dimension(operation.scatter_axis)?.0;
        let unchanged_input_axes = (0..base_output_type.rank())
            .map(|axis| if axis < operation.scatter_axis { Some(axis) } else { Some(axis + 1) })
            .collect::<Vec<_>>();
        let mut output_types = infer_explicit_shape_changing_collective_output_type(
            PARALLEL_SUM_SCATTER_OPERATION_NAME,
            true,
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
            "`{PARALLEL_SUM_SCATTER_OPERATION_NAME}` scatter axis {} is out of bounds for rank {}",
            operation.scatter_axis,
            input_type.rank(),
        )));
    }
    let mut dimensions = input_type.shape().dimensions().to_vec();
    dimensions[operation.scatter_axis] = Dimension::Static(0);
    let sharding = input_type.resized_sharding(dimensions.as_slice(), PARALLEL_SUM_SCATTER_OPERATION_NAME)?;
    let mut base_output_type =
        ArrayType::new(input_type.data_type(), Shape::new(dimensions)).with_memory(input_type.memory());
    base_output_type.sharding = sharding;
    let unchanged_input_axes = (0..input_type.rank())
        .map(|axis| (axis != operation.scatter_axis).then_some(axis))
        .collect::<Vec<_>>();
    let mut output_types = infer_explicit_shape_changing_collective_output_type(
        PARALLEL_SUM_SCATTER_OPERATION_NAME,
        true,
        input_types,
        base_output_type,
        unchanged_input_axes.as_slice(),
        |input_type, output_extents| {
            let rank = input_type.rank();
            let Some(input_extent) = input_type.shape().dimensions().get(operation.scatter_axis) else {
                return Err(TypeError::invalid(format!(
                    "`{PARALLEL_SUM_SCATTER_OPERATION_NAME}` scatter axis {} is out of bounds for rank {rank}",
                    operation.scatter_axis,
                )));
            };
            if let (Dimension::Static(input_extent), Dimension::Static(output_extent)) =
                (input_extent, &output_extents[operation.scatter_axis])
            {
                if *input_extent % effective_axis_size != 0 {
                    return Err(TypeError::invalid(format!(
                        "`{PARALLEL_SUM_SCATTER_OPERATION_NAME}` scatter axis {} size {input_extent} is not \
                         divisible by group size {effective_axis_size}",
                        operation.scatter_axis,
                    )));
                }
                let expected = *input_extent / effective_axis_size;
                if *output_extent != expected {
                    return Err(TypeError::invalid(format!(
                        "`{PARALLEL_SUM_SCATTER_OPERATION_NAME}` result extent must equal input axis {} extent \
                         {input_extent} divided by axis group size {effective_axis_size}; expected {expected} but got \
                         {output_extent}",
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
    context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
    input: &ArrayBatch<C::Value>,
    output_extents: Vec<P::ShapeExtent>,
    output_sharding: Option<Sharding>,
) -> Result<ArrayBatch<C::Value>, BatchingError>
where
    C: Context<Type = ArrayType>,
    C::Value: Reduce + Transpose,
    P: CollectiveArrayExtentBatchingPolicy<C>,
{
    // Both callers infer the output type first, so the scatter axis is known to be within the input rank here.
    if operation.options.axis_index_groups.is_some() {
        return Err(BatchingError::UnsupportedOperation {
            message: format!(
                "`{PARALLEL_SUM_SCATTER_OPERATION_NAME}` axis index groups are not supported when a batch transform \
                 binds the collective axis",
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
        CollectiveMode::Untiled => summed?.move_axis(operation.scatter_axis, 0)?,
        CollectiveMode::Tiled => {
            let mut split_extents = output_extents.clone();
            split_extents.insert(operation.scatter_axis, axis_extent.clone());
            P::reshape_collective(context, summed?, split_extents.as_slice(), None)?
                .move_axis(operation.scatter_axis, 0)?
        }
    };
    let mut physical_output_extents = Vec::with_capacity(output_extents.len() + 1);
    physical_output_extents.push(axis_extent);
    physical_output_extents.extend(output_extents);
    let physical_output_sharding = output_sharding
        .map(|sharding| sharding.with_leading_batch_axis(context.axis_sharding().clone()))
        .transpose()?;
    let output =
        P::reshape_collective(context, scattered, physical_output_extents.as_slice(), physical_output_sharding)?;
    ArrayBatch::new(output, BatchAxis::from_position(0))
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrValue, ArrayOperation, DimensionBounds, LogicalMesh, MeshAxis, RaggedAxis,
    };
    use crate::batching::{BatchAxisSpecification, BatchingTracer, batch};
    use crate::contexts::{EagerContext, StagingContext};
    use crate::differentiation::{TranspositionContext, transpose_mixed_operation};
    use crate::interpretation::InterpretableOperation;
    use crate::macros::check_operation_type_inference;
    use crate::operations::collectives::tests::f32_vector;
    use crate::parameters::Placeholder;
    use crate::partial::{
        PartialEvaluationContext, PartialEvaluationOutput, PartialEvaluationValue, PartialValue,
        PartiallyEvaluatableOperation,
    };
    use crate::programs::{EmptyRegionDriver, MaybeZero, Program, ProgramBuilder};
    use crate::tracing::TracingContext;

    use super::*;

    /// Returns the static `f32` matrix type with the provided number of rows and columns.
    fn f32_matrix(rows: usize, columns: usize) -> ArrayType {
        ArrayType::new_static(DataType::F32, [rows, columns])
    }

    /// Returns the type of the explicit extent value `extent` of the composite array/dimension family.
    fn extent_type(extent: usize) -> ArrayIrType {
        DimensionValue::constant(extent).unwrap().r#type().into_owned().into()
    }

    /// Creates a manual mesh whose axis `"x"` has two participants and whose axis `"y"` has one.
    fn manual_mesh() -> LogicalMesh {
        LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 1, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap()
    }

    /// Builds the single-instruction program that applies `operation` to one input of type `input_type`.
    fn parallel_sum_scatter_program(
        operation: ParallelSumScatterOperation,
        input_type: ArrayType,
    ) -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(input_type);
        let outputs = builder.add_instruction(operation, Vec::new(), vec![input], None).unwrap().to_vec();
        builder.build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Batches `operation` on `input` under an eager batching level of size `axis_size` that binds the axis `"x"`.
    fn batch_parallel_sum_scatter(
        operation: &ParallelSumScatterOperation,
        axis_size: usize,
        input: ArrayBatch<Array>,
    ) -> Result<Vec<ArrayBatch<Array>>, BatchingError> {
        let context = BatchingContext::<EagerContext<Array, ArrayOperation<Array>>, ArrayBatchingPolicy>::new(
            EagerContext::new(),
            axis_size,
        )
        .with_axis_name("x".to_string());
        Ok(operation.batch(&context, &EmptyRegionDriver, &[input])?.into_parts().0)
    }

    #[test]
    fn test_parallel_sum_scatter() {
        let operation = ParallelSumScatterOperation::new("x".to_string(), 4, 1, CollectiveOptions::tiled());
        assert_eq!(operation.name(), PARALLEL_SUM_SCATTER_OPERATION_NAME);
        assert_eq!(operation.axis_name(), "x");
        assert_eq!(operation.axis_size(), 4);
        assert_eq!(operation.scatter_axis(), 1);
        assert_eq!(operation.options(), &CollectiveOptions::tiled());
        assert_eq!(operation.effective_axis_size(), Ok(4));
        assert_eq!(
            operation.to_string(),
            "parallel_sum_scatter [axis_name=\"x\", axis_size=4, scatter_axis=1, options=Tiled]",
        );
        assert_eq!(operation, operation.clone());
        assert_ne!(operation, ParallelSumScatterOperation::new("x".to_string(), 4, 1, CollectiveOptions::default()));

        // Participant groups restrict the effective axis size to the size of one group.
        let grouped = ParallelSumScatterOperation::new(
            "x".to_string(),
            4,
            0,
            CollectiveOptions::tiled().with_axis_index_groups(vec![vec![0, 2], vec![3, 1]]),
        );
        assert_eq!(grouped.effective_axis_size(), Ok(2));
    }

    #[test]
    fn test_parallel_sum_scatter_type_inference() {
        // A tiled sum-scatter divides the scatter axis by the group size, while an untiled one requires the scatter
        // axis to have exactly that size and removes it.
        check_operation_type_inference!(
            operation = ParallelSumScatterOperation::new("x".to_string(), 4, 0, CollectiveOptions::tiled()),
            cases = [
                { input_types = [f32_vector(8)], output_types = [f32_vector(2)] },
                {
                    input_types = [ArrayType::new_static(DataType::Zero, [8])],
                    output_types = [ArrayType::new_static(DataType::Zero, [2])],
                },
                {
                    input_types = [f32_vector(6)],
                    error = "`parallel_sum_scatter` scatter axis 0 size 6 is not divisible by group size 4",
                },
                {
                    input_types = [ArrayType::scalar(DataType::F32)],
                    error = "`parallel_sum_scatter` scatter axis 0 is out of bounds for rank 0",
                },
                {
                    input_types = [ArrayType::new_static(DataType::Boolean, [8])],
                    error = "`parallel_sum_scatter` requires numeric inputs but got `bool`",
                },
            ],
        );
        check_operation_type_inference!(
            operation = ParallelSumScatterOperation::new("x".to_string(), 4, 1, CollectiveOptions::default()),
            cases = [
                { input_types = [f32_matrix(3, 4)], output_types = [f32_vector(3)] },
                {
                    input_types = [f32_matrix(4, 3)],
                    error = "`parallel_sum_scatter` untiled scatter axis 1 size 3 must equal group size 4",
                },
            ],
        );
        check_operation_type_inference!(
            operation = ParallelSumScatterOperation::new(
                "x".to_string(),
                4,
                0,
                CollectiveOptions::tiled().with_axis_index_groups(vec![vec![0, 2], vec![3, 1]]),
            ),
            cases = [{ input_types = [f32_vector(6)], output_types = [f32_vector(3)] }],
        );

        // Over a manual mesh axis, every participant receives a different chunk, so an invariant input is rejected and
        // a varying input keeps its variation. An input that is unreduced over the operation's own axis has its pending
        // sum completed and its output varies over the axis, while any other unreduced axis is rejected.
        let sharding = Sharding::replicated(manual_mesh(), 1);
        let with_sharding = |sharding: Sharding| f32_vector(4).with_sharding(sharding).unwrap();
        let output_with_sharding = |sharding: Sharding| f32_vector(2).with_sharding(sharding).unwrap();
        let varying = sharding.clone().with_varying_manual_axes(["x"]).unwrap();
        check_operation_type_inference!(
            operation = ParallelSumScatterOperation::new("x".to_string(), 2, 0, CollectiveOptions::tiled()),
            cases = [
                {
                    input_types = [with_sharding(varying.clone())],
                    output_types = [output_with_sharding(varying.clone())],
                },
                {
                    input_types = [with_sharding(sharding.clone().with_unreduced_axes(["x"]).unwrap())],
                    output_types = [output_with_sharding(varying)],
                },
                {
                    input_types = [with_sharding(sharding.clone())],
                    error = "`parallel_sum_scatter` input must vary over manual axis `x`; pass an invariant value \
                             through `parallel_vary` first so that every copy is counted",
                },
                {
                    input_types = [with_sharding(sharding.with_unreduced_axes(["y"]).unwrap())],
                    error = "`parallel_sum_scatter` only supports an unreduced input over its own axis `x`",
                },
            ],
        );

        // The composite family follows each array input with one explicit extent per output axis, checks every extent
        // that is statically known, and keeps dynamic extents.
        assert_eq!(
            infer_explicit_parallel_sum_scatter_output_types(
                &ParallelSumScatterOperation::new("x".to_string(), 4, 1, CollectiveOptions::default()),
                &[ArrayType::new_static(DataType::F32, [2, 4, 3]).into(), extent_type(2), extent_type(3)],
            ),
            Ok(vec![f32_matrix(2, 3).into()]),
        );
        assert_eq!(
            infer_explicit_parallel_sum_scatter_output_types(
                &ParallelSumScatterOperation::new("x".to_string(), 4, 1, CollectiveOptions::default()),
                &[f32_matrix(2, 5).into(), extent_type(2)],
            ),
            Err(TypeError::invalid("`parallel_sum_scatter` untiled scatter axis 1 size 5 must equal group size 4")),
        );
        assert_eq!(
            infer_explicit_parallel_sum_scatter_output_types(
                &ParallelSumScatterOperation::new(
                    "x".to_string(),
                    4,
                    0,
                    CollectiveOptions::tiled().with_axis_index_groups(vec![vec![0, 2], vec![3, 1]]),
                ),
                &[f32_vector(6).into(), extent_type(3)],
            ),
            Ok(vec![f32_vector(3).into()]),
        );
        let input_axis = DimensionVariable::new("input", DimensionBounds::new(1, Some(17)).unwrap());
        let output_axis = DimensionVariable::new("split", DimensionBounds::new(1, Some(9)).unwrap());
        assert_eq!(
            infer_explicit_parallel_sum_scatter_output_types(
                &ParallelSumScatterOperation::new("x".to_string(), 2, 0, CollectiveOptions::tiled()),
                &[
                    ArrayType::new(
                        DataType::F32,
                        Shape::new(vec![Dimension::Dynamic(input_axis), Dimension::Static(3)])
                    )
                    .into(),
                    ArrayIrType::Dimension(DimensionType::from(output_axis.clone())),
                    extent_type(3),
                ],
            ),
            Ok(vec![
                ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(output_axis), Dimension::Static(3)]))
                    .into(),
            ]),
        );
        assert_eq!(
            infer_explicit_parallel_sum_scatter_output_types(
                &ParallelSumScatterOperation::new("x".to_string(), 0, 0, CollectiveOptions::tiled()),
                &[f32_vector(3).into(), extent_type(3)],
            ),
            Err(TypeError::invalid("`parallel_sum_scatter` axis size must be greater than zero")),
        );
    }

    #[test]
    fn test_parallel_sum_scatter_interpretation() {
        // A single participant sums only its own value: tiled mode is the identity, while untiled mode removes the
        // size-one scatter axis. Any larger axis has no per-item semantics outside an enclosing binder.
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let input = Array::matrix(1, 3, vec![1.0f32, 2.0, 3.0]).unwrap();
        assert_eq!(
            ParallelSumScatterOperation::new("x".to_string(), 1, 0, CollectiveOptions::tiled()).interpret(
                &context,
                &EmptyRegionDriver,
                std::slice::from_ref(&input),
            ),
            Ok(vec![input.clone()]),
        );
        assert_eq!(
            ParallelSumScatterOperation::new("x".to_string(), 1, 0, CollectiveOptions::default()).interpret(
                &context,
                &EmptyRegionDriver,
                std::slice::from_ref(&input),
            ),
            Ok(vec![Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap()]),
        );
        assert_eq!(
            ParallelSumScatterOperation::new("x".to_string(), 2, 1, CollectiveOptions::default()).interpret(
                &context,
                &EmptyRegionDriver,
                std::slice::from_ref(&input),
            ),
            Err(ProgramError::UnsupportedOperation {
                message: "cannot interpret `parallel_sum_scatter` over axis `x` of size 2 without an enclosing binder"
                    .to_string(),
            }),
        );

        // The composite family interprets the array input with its explicit extents in the same way.
        let context = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = ArrayIrValue::Array(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap());
        let extent = ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap());
        assert_eq!(
            context.bind(
                ParallelSumScatterOperation::new("x".to_string(), 1, 0, CollectiveOptions::tiled()),
                Vec::new(),
                &[input.clone(), extent.clone()],
            ),
            Ok(vec![input.clone()]),
        );
        assert_eq!(
            context.bind(
                ParallelSumScatterOperation::new("x".to_string(), 0, 0, CollectiveOptions::tiled()),
                Vec::new(),
                &[input, extent],
            ),
            Err(ProgramError::Type(TypeError::invalid("`parallel_sum_scatter` axis size must be greater than zero"))),
        );
    }

    #[test]
    fn test_parallel_sum_scatter_partial_evaluation() {
        // A known input over a degenerate axis folds through interpretation, here removing the untiled scatter axis.
        let input = Array::matrix(1, 2, vec![1.0f32, 2.0]).unwrap();
        let program = parallel_sum_scatter_program(
            ParallelSumScatterOperation::new("x".to_string(), 1, 0, CollectiveOptions::default()),
            f32_matrix(1, 2),
        );
        let evaluation = program.partially_evaluate(&[PartialValue::Known(input.clone())]).unwrap();
        assert_eq!(evaluation.outputs(), &[PartialEvaluationOutput::Known(Array::vector(vec![1.0f32, 2.0]).unwrap())]);

        // A known input over a larger axis under an eager parent residualizes the operation, which has no per-item
        // value, so the residual program is the source program itself.
        let operation = ParallelSumScatterOperation::new("x".to_string(), 2, 1, CollectiveOptions::tiled());
        let program = parallel_sum_scatter_program(operation.clone(), f32_matrix(1, 2));
        let evaluation = program.partially_evaluate(&[PartialValue::Known(input)]).unwrap();
        assert_eq!(evaluation.program().to_string(), program.to_string());
        assert!(evaluation.outputs()[0].is_unknown());

        // A known input under a staging parent stays known, because the operation is staged into the parent trace.
        let trace = TracingContext::<Array, ArrayOperation<Array>>::new();
        let input = PartialEvaluationValue::known(trace.input(f32_matrix(1, 2)));
        let outputs = operation
            .partially_evaluate(&PartialEvaluationContext::new(trace), &EmptyRegionDriver, &[input])
            .unwrap();
        assert!(outputs[0].is_known());
        assert_eq!(outputs[0].r#type().as_ref(), &f32_matrix(1, 1));
    }

    #[test]
    fn test_parallel_sum_scatter_batching() {
        // A level that binds the axis sums over its mapped axis and maps the scattered chunks back onto it, so that
        // item `i` receives chunk `i` of the sum: a contiguous chunk in tiled mode and row `i` in untiled mode.
        let mapped = |value: Array| ArrayBatch::new(value, BatchAxis::new(0)).unwrap();
        let tiled = ParallelSumScatterOperation::new("x".to_string(), 2, 0, CollectiveOptions::tiled());
        let untiled = ParallelSumScatterOperation::new("x".to_string(), 2, 0, CollectiveOptions::default());
        assert_eq!(
            batch_parallel_sum_scatter(
                &tiled,
                2,
                mapped(Array::matrix(2, 4, vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]).unwrap()),
            ),
            Ok(vec![mapped(Array::matrix(2, 2, vec![11.0, 22.0, 33.0, 44.0]).unwrap())]),
        );
        assert_eq!(
            batch_parallel_sum_scatter(&untiled, 2, mapped(Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap())),
            Ok(vec![mapped(Array::vector(vec![4.0, 6.0]).unwrap())]),
        );

        // A replicated input holds the same value for every item, so every item's copy is counted in the sum.
        assert_eq!(
            batch_parallel_sum_scatter(&tiled, 2, ArrayBatch::replicated(Array::vector(vec![1.0, 2.0]).unwrap())),
            Ok(vec![mapped(Array::matrix(2, 1, vec![2.0, 4.0]).unwrap())]),
        );

        // A level that binds the axis rejects participant groups, and every level rejects bounded ragged inputs.
        let grouped = ParallelSumScatterOperation::new(
            "x".to_string(),
            2,
            0,
            CollectiveOptions::tiled().with_axis_index_groups(vec![vec![0], vec![1]]),
        );
        assert_eq!(
            batch_parallel_sum_scatter(&grouped, 2, mapped(Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap())),
            Err(BatchingError::UnsupportedOperation {
                message: "`parallel_sum_scatter` axis index groups are not supported when a batch transform binds the \
                          collective axis"
                    .to_string(),
            }),
        );
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(4)).unwrap());
        let ragged = mapped(Array::matrix(2, 4, vec![1.0; 8]).unwrap())
            .with_ragged_axes(vec![RaggedAxis::new(1, Array::vector(vec![2i32, 4]).unwrap(), length.clone(), vec![0])])
            .unwrap();
        assert_eq!(
            batch_parallel_sum_scatter(&tiled, 2, ragged),
            Err(BatchingError::UnsupportedOperation {
                message: "`parallel_sum_scatter` does not support bounded ragged dimension `length` on input 0"
                    .to_string(),
            }),
        );

        // A level that binds another axis forwards the sum-scatter to its parent, shifting the scatter axis past the
        // mapped axis and, in untiled mode, tracking the mapped axis across the removed scatter axis.
        for (operation, input_type, batch_axis, expected) in [
            (
                tiled.clone(),
                f32_matrix(3, 4),
                BatchAxis::new(0),
                indoc! {"
                    lambda %0:f32[3, 4] .
                    let %1:f32[3, 2] = parallel_sum_scatter [axis_name=\"x\", axis_size=2, scatter_axis=1, options=Tiled] %0
                    in (%1)"
                },
            ),
            (
                ParallelSumScatterOperation::new("x".to_string(), 2, 0, CollectiveOptions::default()),
                f32_matrix(2, 3),
                BatchAxis::new(1),
                indoc! {"
                    lambda %0:f32[2, 3] .
                    let %1:f32[3] = parallel_sum_scatter [axis_name=\"x\", axis_size=2, scatter_axis=0, options=Untiled] %0
                    in (%1)"
                },
            ),
        ] {
            let trace = TracingContext::<Array, ArrayOperation<Array>>::new();
            let input = ArrayBatch::new(trace.input(input_type), batch_axis).unwrap();
            let context =
                BatchingContext::<_, ArrayBatchingPolicy>::new(trace.clone(), 3).with_axis_name("y".to_string());
            let outputs = operation.batch(&context, &EmptyRegionDriver, &[input]).unwrap().into_parts().0;
            let program = trace
                .builder()
                .borrow()
                .clone()
                .build::<Vec<Array>, Vec<Array>>(
                    vec![outputs[0].value().atom_id().unwrap()],
                    vec![Placeholder],
                    vec![Placeholder],
                )
                .unwrap();
            assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
            assert_eq!(program.to_string(), expected);
        }

        // The composite family rejects bounded ragged inputs before it reads the mapped extents.
        let extents = ArrayIrValue::Array(Array::vector(vec![2i32, 4]).unwrap());
        let input =
            ArrayIrBatch::new(ArrayIrValue::Array(Array::matrix(2, 4, vec![1.0f32; 8]).unwrap()), BatchAxis::new(0))
                .unwrap()
                .with_ragged_axes(vec![RaggedAxis::new(1, extents.clone(), length.clone(), vec![0])])
                .unwrap();
        let output_extent =
            ArrayIrBatch::mapped_dimension(extents, BatchAxis::new(0), DimensionType::from(length)).unwrap();
        let context = BatchingContext::new(
            EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new(),
            ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()),
        )
        .with_axis_name("x".to_string());
        assert_eq!(
            tiled.batch_in_parent(&context, &EmptyRegionDriver, &[input, output_extent]),
            Err(BatchingError::UnsupportedOperation {
                message: "`parallel_sum_scatter` does not support bounded ragged dimension `length` on input 0"
                    .to_string(),
            }),
        );
    }

    #[test]
    fn test_parallel_sum_scatter_differentiation() {
        // The collective is linear, so the tangent rides the same sum-scatter as the primal.
        let operation = ParallelSumScatterOperation::new("x".to_string(), 2, 0, CollectiveOptions::tiled());
        assert_eq!(
            parallel_sum_scatter_program(operation, f32_vector(4)).jvp().unwrap().to_string(),
            indoc! {"
                lambda %0:f32[4], %1:f32[4] .
                let %2:f32[2] = parallel_sum_scatter [axis_name=\"x\", axis_size=2, scatter_axis=0, options=Tiled] %0
                    %3:f32[2] = parallel_sum_scatter [axis_name=\"x\", axis_size=2, scatter_axis=0, options=Tiled] %1
                in (%2, %3)"
            },
        );

        // The composite family linearizes a sum-scatter with an explicit extent into a linear call whose pullback, over
        // a single participant, returns the output cotangent unchanged.
        let variable = DimensionVariable::new("extent", DimensionBounds::new(1, Some(9)).unwrap());
        let dimension_type = DimensionType::from(variable.clone());
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let array =
            builder.add_input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(variable)])).into());
        let extent = builder.add_input(dimension_type.clone().into());
        let output = builder
            .add_instruction(
                ParallelSumScatterOperation::new("x".to_string(), 1, 0, CollectiveOptions::tiled()),
                Vec::new(),
                vec![array, extent],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.residual_count(), 1);
        let input = ArrayIrValue::Array(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap());
        let extent = ArrayIrValue::Dimension(DimensionValue::new(dimension_type, 3).unwrap());
        let mut primal_outputs = linearization.primal().interpret(vec![input, extent]).unwrap();
        let residuals = primal_outputs.split_off(1);
        let cotangent = ArrayIrValue::Array(Array::vector(vec![4.0f32, 5.0, 6.0]).unwrap());
        let mut pullback_inputs = vec![cotangent.clone()];
        pullback_inputs.extend(residuals);
        assert_eq!(linearization.pullback().unwrap().interpret(pullback_inputs), Ok(vec![cotangent]));
    }

    #[test]
    fn test_parallel_sum_scatter_transposition() {
        // A sum-scatter hands every participant its chunk of the summed cotangents, so its transpose gathers the output
        // cotangent into a varying all-gather, and transposing that recovers the sum-scatter.
        let program = parallel_sum_scatter_program(
            ParallelSumScatterOperation::new("x".to_string(), 2, 0, CollectiveOptions::tiled()),
            f32_vector(8),
        );
        let transposed = program.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f32[4] .
                let %1:f32[8] = all_gather [
                    axis_name=\"x\",
                    axis_size=2,
                    concat_axis=0,
                    options=Tiled,
                    output_variance=Varying,
                ] %0
                in (%1)"
            },
        );
        assert_eq!(transposed.transpose_with_respect_to(&[0], &[]).unwrap().to_string(), program.to_string());

        // The composite family transposes the array input through the same all-gather and gives the explicit extent a
        // structural-zero cotangent.
        let context = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let output_cotangent = context.input(f32_vector(3).into());
        let mut context = TranspositionContext::new(context);
        let inputs = [PartialValue::Unknown(f32_vector(3).into()), PartialValue::Unknown(extent_type(3))];
        let accumulators = context.cotangent_accumulators(&inputs, &[]).unwrap();
        transpose_mixed_operation(
            &mut context,
            &ParallelSumScatterOperation::new("x".to_string(), 1, 0, CollectiveOptions::tiled()),
            &inputs,
            &[MaybeZero::Value(output_cotangent)],
            &accumulators,
        )
        .unwrap();
        let cotangents = context.take_cotangents(&accumulators).unwrap();
        assert!(matches!(cotangents.as_slice(), [MaybeZero::Value(_), MaybeZero::Zero(_)]));
        assert!(matches!(
            context.builder().borrow().instructions()[0].operation(),
            ArrayIrOperation::Array(ArrayOperation::AllGather(_)),
        ));
    }

    #[test]
    fn test_parallel_sum_scatter_parallel_sum_scatter() {
        type CompositeContext = EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

        // A name that a `batch` level binds sums its batch items and scatters the chunks back onto them.
        let output = batch(
            |item: BatchingTracer<CompositeContext, ArrayIrBatchingPolicy>| item.parallel_sum_scatter_tiled("x", 0),
            ArrayIrValue::Array(Array::matrix(2, 4, vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]).unwrap()),
            BatchAxis::new(0),
            BatchAxis::new(0),
            BatchAxisSpecification::named("x"),
        );
        assert_eq!(output, Ok(ArrayIrValue::Array(Array::matrix(2, 2, vec![11.0, 22.0, 33.0, 44.0]).unwrap())));

        // A name that no enclosing binder binds fails fast instead of silently acting as identity.
        assert_eq!(
            batch(
                |item: BatchingTracer<CompositeContext, ArrayIrBatchingPolicy>| item.parallel_sum_scatter("y", 0),
                ArrayIrValue::Array(Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap()),
                BatchAxis::new(0),
                BatchAxis::new(0),
                BatchAxisSpecification::named("x"),
            ),
            Err::<ArrayIrValue<Array>, _>(BatchingError::Axis(crate::axes::AxisError::UnboundAxisName {
                name: "y".to_string(),
            })),
        );

        // Over a manual mesh axis, the capability stages the output extents and the sum-scatter for a varying value
        // directly, while an invariant value, and a value without a sharding, are first made varying, so that every
        // copy is counted. A dynamic scatter extent is checked by a staged runtime assertion.
        let mesh = manual_mesh();
        let sharding = Sharding::replicated(mesh.clone(), 1);
        let items = DimensionVariable::new("items", DimensionBounds::new(1, Some(5)).unwrap());
        for (input_type, expected) in [
            (
                f32_vector(4).with_sharding(sharding.clone().with_varying_manual_axes(["x"]).unwrap()).unwrap(),
                indoc! {"
                    lambda %0:f32[4][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}], varying_manual={'x'}}] .
                    let %1:dimension<4> = constant [value=4]
                        %2:dimension<2> = constant [value=2]
                        %3:dimension<0> = constant [value=0]
                        %4:dimension<1> = constant [value=1]
                        %5:bool[] = const true
                        %6:dimension<0> = dimension_rem %1 %2
                        %7:bool[] = const true
                        %8:dimension<2> = dimension_div %1 %2
                        %9:f32[2][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}], varying_manual={'x'}}] = parallel_sum_scatter [axis_name=\"x\", axis_size=2, scatter_axis=0, options=Tiled] %0 %8
                    in (%9)"
                },
            ),
            (
                f32_vector(4).with_sharding(sharding).unwrap(),
                indoc! {"
                    lambda %0:f32[4][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}]}] .
                    let %1:f32[4][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}], varying_manual={'x'}}] = parallel_vary [axis_name=\"x\"] %0
                        %2:dimension<4> = constant [value=4]
                        %3:dimension<2> = constant [value=2]
                        %4:dimension<0> = constant [value=0]
                        %5:dimension<1> = constant [value=1]
                        %6:bool[] = const true
                        %7:dimension<0> = dimension_rem %2 %3
                        %8:bool[] = const true
                        %9:dimension<2> = dimension_div %2 %3
                        %10:f32[2][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}], varying_manual={'x'}}] = parallel_sum_scatter [axis_name=\"x\", axis_size=2, scatter_axis=0, options=Tiled] %1 %9
                    in (%10)"
                },
            ),
            (
                ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(items)])),
                indoc! {"
                    lambda %0:f32[items] .
                    let %1:f32[items][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}]}] = broadcast [
                        output_type=f32[items][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}]}],
                        output_axes=[0],
                    ] %0
                        %2:f32[items][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}], varying_manual={'x'}}] = parallel_vary [axis_name=\"x\"] %1
                        %3:dimension<items ∈ [1, 5)> = dimension_size [axis=0] %2
                        %4:dimension<2> = constant [value=2]
                        %5:dimension<0> = constant [value=0]
                        %6:dimension<1> = constant [value=1]
                        %7:bool[] = const true
                        %8:dimension<items % 2 ∈ [0, 2)> = dimension_rem %3 %4
                        %9:bool[] = compare [direction=Equal] %8 %5
                        () = assert [
                            message=\"collective extent must be divisible by the participant count\",
                            labels=[\"extent\", \"divisor\"],
                        ] %9 %3 %4
                        %10:dimension<items / 2 ∈ [0, 3)> = dimension_div %3 %4
                        %11:f32[items / 2][sharding={mesh<['x'=2:manual, 'y'=1:manual]>, [{}], varying_manual={'x'}}] = parallel_sum_scatter [axis_name=\"x\", axis_size=2, scatter_axis=0, options=Tiled] %2 %10
                    in (%11)"
                },
            ),
        ] {
            let (_, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace_with_named_axes(
                |input| input.parallel_sum_scatter_tiled("x", 0),
                ArrayIrType::Array(input_type),
                vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh: mesh.clone() })],
            )
            .unwrap();
            println!("CAPTURE\n{program}\nEND");
            assert_eq!(program.to_string(), expected);
        }
    }
}
