//! Contains the named-axis [`AllGatherOperation`], which concatenates every participant's operand across a named
//! axis, together with its interpretation, partial-evaluation, batching, forward-mode differentiation, and
//! transposition rules.

// TODO(eaplatanios): Review this module.

use std::fmt::Display;

use crate::arrays::batching::DynamicArrayBatchingPolicy;
use crate::arrays::{
    ArrayBatch, ArrayBatching, ArrayIrBatch, ArrayIrBatching, ArrayIrType, ArrayType, Dimension, DimensionOperation,
    DimensionType, DimensionValue, DimensionVariable, LinearResiduals, RaggedAxis, Shape, Sharding,
};
use crate::axes::{AxisIndexOperation, NamedAxes};
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
use crate::operations::dimensions::dimension_from_scalar::DimensionFromScalarOperation;
use crate::operations::dimensions::dimension_mul::DimensionMulOperation;
use crate::operations::dimensions::dimension_requirement::DimensionRequirement;
use crate::operations::dimensions::dimension_size::{DimensionSize, DimensionSizeOperation};
use crate::operations::manipulation::broadcasting::DynamicBroadcastOperation;
use crate::operations::manipulation::reshaping::{DynamicReshapeOperation, Reshape};
use crate::operations::manipulation::slicing::{DynamicShapeSliceOperation, resized_output_sharding};
use crate::operations::manipulation::transposition::Transpose;
use crate::operations::math::div::Div;
use crate::operations::math::mul::Mul;
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::{
    MaybeZero, MemberOperation, Operation, OperationFormatter, OperationProjection, ProgramError, ProjectedValue,
    RegionInterface, TypeError, TypeIdentityRenaming, Typed, Value, ValueProjection,
};
use crate::tracing::{Tracer, TracingContext};

use super::parallel_sum_scatter::ParallelSumScatterOperation;
use super::{
    CollectiveBatchingPolicy, CollectiveMode, CollectiveOptions, collective_extent_constant, collective_input_extents,
    explicit_collective_inputs, forward_collective_to_parent, forward_explicit_collective,
    forward_shape_changing_collective, impl_shape_changing_collective_member_operation,
    infer_explicit_shape_changing_collective_output_type, interpret_degenerate_collective,
    jvp_shape_changing_collective_with_adjoint, multiplied_collective_extent, reject_ragged_collective_inputs,
    resolve_named_axis_size, shape_changing_collective, shape_changing_collective_dimensions,
    shape_changing_collective_output_type, transpose_shape_changing_collective, validate_collective_axis_size,
    validate_explicit_collective_output_extents,
};

/// Named-axis variance carried by an all-gather result.
///
/// This is an operation option rather than parallel type metadata. Type inference maps it onto the canonical
/// [`Sharding::varying_manual_axes`](crate::arrays::Sharding::varying_manual_axes) and
/// [`Sharding::reduced_axes`](crate::arrays::Sharding::reduced_axes) sets.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum AllGatherOutputVariance {
    /// The result continues to vary across the gathered manual mesh axis.
    #[default]
    Varying,

    /// The result is invariant across the gathered manual mesh axis.
    Invariant,

    /// The result records the gathered manual mesh axis as reduced.
    Reduced,
}

/// Applies an all-gather's named-axis variance transition to the canonical sharding metadata.
fn all_gather_output_type(
    input_type: &ArrayType,
    mut output_type: ArrayType,
    operation: &AllGatherOperation,
) -> Result<ArrayType, TypeError> {
    let Some(input_sharding) = input_type.sharding() else {
        if operation.output_variance == AllGatherOutputVariance::Reduced {
            return Err(TypeError::invalid(
                "`all_gather` with reduced output variance requires sharding metadata".to_string(),
            ));
        }
        return Ok(output_type);
    };
    if input_type.unreduced_axes().contains(operation.axis_name()) {
        return Err(TypeError::invalid(format!(
            "`all_gather` does not support an operand that is unreduced over axis `{}`",
            operation.axis_name(),
        )));
    }

    let mut varying_axes = input_sharding.varying_manual_axes().clone();
    let mut reduced_axes = input_sharding.reduced_axes().clone();
    match operation.output_variance {
        AllGatherOutputVariance::Varying => {
            if reduced_axes.contains(operation.axis_name()) {
                return Err(TypeError::invalid(format!(
                    "`all_gather` cannot make axis `{}` varying because the operand records it as reduced",
                    operation.axis_name(),
                )));
            }
            varying_axes.insert(operation.axis_name.clone());
        }
        AllGatherOutputVariance::Invariant => {
            if reduced_axes.contains(operation.axis_name()) {
                return Err(TypeError::invalid(format!(
                    "`all_gather` cannot make axis `{}` invariant because the operand records it as reduced",
                    operation.axis_name(),
                )));
            }
            varying_axes.remove(operation.axis_name());
        }
        AllGatherOutputVariance::Reduced => {
            if !varying_axes.remove(operation.axis_name()) {
                return Err(TypeError::invalid(format!(
                    "`all_gather` with reduced output variance requires an operand varying over axis `{}`",
                    operation.axis_name(),
                )));
            }
            if !reduced_axes.insert(operation.axis_name.clone()) {
                return Err(TypeError::invalid(format!(
                    "`all_gather` operand is already reduced over axis `{}`",
                    operation.axis_name(),
                )));
            }
        }
    }
    let output_sharding = output_type.sharding().expect("shape projection preserves sharding").clone();
    let output_sharding = output_sharding
        .with_varying_manual_axes(varying_axes)
        .and_then(|sharding| sharding.with_reduced_axes(reduced_axes))
        .map_err(|error| TypeError::invalid(error.to_string()))?;
    output_type.sharding = Some(output_sharding);
    Ok(output_type)
}

/// Infers the composite all-gather contract.
pub(crate) fn infer_explicit_all_gather_output_types(
    operation: &AllGatherOperation,
    input_types: &[ArrayIrType],
) -> Result<Vec<ArrayIrType>, TypeError> {
    let effective_axis_size = operation.effective_axis_size()?;
    let Some(input_type) = input_types.first() else {
        return Err(TypeError::invalid("`all_gather` expects an array followed by its output extents"));
    };
    let input_type = <&ArrayType>::try_from(input_type)?;
    let (base_output_type, unchanged_input_axes) = match operation.options.mode {
        CollectiveMode::Untiled => (
            input_type.with_inserted_dimension(operation.concat_axis, Dimension::Static(effective_axis_size))?,
            (0..=input_type.rank())
                .map(|axis| {
                    if axis == operation.concat_axis {
                        None
                    } else if axis < operation.concat_axis {
                        Some(axis)
                    } else {
                        Some(axis - 1)
                    }
                })
                .collect::<Vec<_>>(),
        ),
        CollectiveMode::Tiled => {
            if operation.concat_axis >= input_type.rank() {
                return Err(TypeError::invalid(format!(
                    "`all_gather` concat axis {} is out of bounds for rank {}",
                    operation.concat_axis,
                    input_type.rank(),
                )));
            }
            let mut dimensions = input_type.shape().dimensions().to_vec();
            dimensions[operation.concat_axis] = Dimension::Static(0);
            let sharding = resized_output_sharding(input_type, dimensions.as_slice(), ALL_GATHER_OPERATION_NAME)?;
            let mut output_type =
                ArrayType::new(input_type.data_type(), Shape::new(dimensions)).with_memory(input_type.memory());
            output_type.sharding = sharding;
            (output_type, (0..input_type.rank()).map(|axis| (axis != operation.concat_axis).then_some(axis)).collect())
        }
    };
    let mut output_types = infer_explicit_shape_changing_collective_output_type(
        ALL_GATHER_OPERATION_NAME,
        input_types,
        base_output_type,
        unchanged_input_axes.as_slice(),
        |input_type, output_extents| {
            match operation.options.mode {
                CollectiveMode::Untiled => {
                    let output_extent = &output_extents[operation.concat_axis];
                    if output_extent != &Dimension::Static(effective_axis_size) {
                        return Err(TypeError::invalid(format!(
                            "`all_gather` inserted output axis {} extent must equal axis group size \
                             {effective_axis_size} but got {output_extent}",
                            operation.concat_axis,
                        )));
                    }
                }
                CollectiveMode::Tiled => {
                    let input_extent = &input_type.shape().dimensions()[operation.concat_axis];
                    let output_extent = &output_extents[operation.concat_axis];
                    if let (Dimension::Static(input_extent), Dimension::Static(output_extent)) =
                        (input_extent, output_extent)
                    {
                        let expected = input_extent.checked_mul(effective_axis_size).ok_or_else(|| {
                            TypeError::invalid("`all_gather` result extent does not fit in usize".to_string())
                        })?;
                        if *output_extent != expected {
                            return Err(TypeError::invalid(format!(
                                "`all_gather` result extent must equal input axis {} extent {input_extent} multiplied \
                                 by axis group size {effective_axis_size}; expected {expected} but got {output_extent}",
                                operation.concat_axis,
                            )));
                        }
                    }
                }
            }
            Ok(())
        },
    )?;
    let input_type = <&ArrayType>::try_from(&input_types[0])?;
    let output_type = <&ArrayType>::try_from(&output_types.remove(0))?.clone();
    Ok(vec![all_gather_output_type(input_type, output_type, operation)?.into()])
}

shape_changing_collective! {
    /// [`Operation`] that concatenates every participant's operand along `concat_axis` across the named axis, so
    /// every participant receives the full concatenation — the analogue of
    /// [JAX's `all_gather`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.all_gather.html) with `tiled = True`
    /// and [StableHLO's `all_gather`](https://openxla.org/stablehlo/spec#all_gather). The output extends
    /// `concat_axis` by the axis size; all other dimensions are unchanged. The collective is linear and its
    /// transpose depends on the requested output variance: varying results use [`ParallelSumScatterOperation`],
    /// invariant results select the current participant's chunk locally, and reduced results use sum-scatter while
    /// consuming the cotangent's unreduced-axis state. A matching `batch` level consumes the mapped batch axis by
    /// merging it item-major into `concat_axis`, replicating the gathered value across the batch items.
    ///
    /// Untiled batching co-moves bounded ragged metadata with the gathered value: the named participant axis becomes
    /// an ordinary output axis and is added to each participant-varying extent array's `extent_axes` mapping. Tiled
    /// gathering of a ragged carrier is rejected because fusing the participant and concatenation axes can make live
    /// chunks non-prefix-shaped, which one [`RaggedAxis`] cannot represent faithfully.
    operation = AllGatherOperation,
    name = ALL_GATHER_OPERATION_NAME = "all_gather",
    fields = {
        /// Axis of the operand along which the participants' values are concatenated.
        concat_axis: usize,

        /// Shared rank and participant-group semantics.
        options: CollectiveOptions,

        /// Named-axis variance of the result.
        output_variance: AllGatherOutputVariance,
    },
    infer = |operation, input_type, dimensions| {
        let effective_axis_size = operation.effective_axis_size()?;
        let output_type = match operation.options.mode {
            CollectiveMode::Untiled => input_type
                .with_inserted_dimension(operation.concat_axis, Dimension::Static(effective_axis_size))?,
            CollectiveMode::Tiled => {
                let mut output_dimensions = dimensions;
                let Some(dimension) = output_dimensions.get_mut(operation.concat_axis) else {
                    return Err(TypeError::invalid(format!(
                        "`all_gather` concat axis {} is out of bounds for rank {}",
                        operation.concat_axis,
                        output_dimensions.len(),
                    )));
                };
                *dimension = dimension.checked_mul(effective_axis_size).ok_or_else(|| {
                    TypeError::invalid("`all_gather` result extent does not fit in usize".to_string())
                })?;
                shape_changing_collective_output_type(ALL_GATHER_OPERATION_NAME, input_type, output_dimensions)?
            }
        };
        all_gather_output_type(input_type, output_type, operation)
    },
}

impl AllGatherOperation {
    /// Returns the axis of the operand along which the participants' values are concatenated.
    #[inline]
    pub fn concat_axis(&self) -> usize {
        self.concat_axis
    }

    /// Returns the shared rank and participant-group semantics.
    #[inline]
    pub fn options(&self) -> &CollectiveOptions {
        &self.options
    }

    /// Returns the named-axis variance of the result.
    #[inline]
    pub fn output_variance(&self) -> AllGatherOutputVariance {
        self.output_variance
    }

    /// Returns the participant count used for result-shape arithmetic.
    #[inline]
    pub fn effective_axis_size(&self) -> Result<usize, TypeError> {
        if self.output_variance != AllGatherOutputVariance::Varying && self.options.axis_index_groups.is_some() {
            return Err(TypeError::invalid(
                "`all_gather` axis index groups are not supported with invariant or reduced output variance"
                    .to_string(),
            ));
        }
        self.options.effective_axis_size(ALL_GATHER_OPERATION_NAME, self.axis_size)
    }
}

impl_shape_changing_collective_member_operation!(AllGatherOperation, infer_explicit_all_gather_output_types);

/// Stages an all-gather with first-class dynamic tiled extents and rank-changing untiled semantics.
pub trait AllGather: Sized {
    /// Stacks participants along a new axis at `concat_axis`, producing an output that varies across `axis_name`.
    #[inline]
    fn all_gather(&self, axis_name: &str, concat_axis: usize) -> Result<Self, ProgramError> {
        self.all_gather_with_options(
            axis_name,
            concat_axis,
            CollectiveOptions::default(),
            AllGatherOutputVariance::Varying,
        )
    }

    /// Concatenates participants into the existing `concat_axis`, producing an output that varies across
    /// `axis_name`.
    #[inline]
    fn all_gather_tiled(&self, axis_name: &str, concat_axis: usize) -> Result<Self, ProgramError> {
        self.all_gather_with_options(
            axis_name,
            concat_axis,
            CollectiveOptions::new(CollectiveMode::Tiled),
            AllGatherOutputVariance::Varying,
        )
    }

    /// Gathers participants using explicit shape, grouping, and output-variance semantics.
    fn all_gather_with_options(
        &self,
        axis_name: &str,
        concat_axis: usize,
        options: CollectiveOptions,
        output_variance: AllGatherOutputVariance,
    ) -> Result<Self, ProgramError>;
}

impl<V> AllGather for V
where
    V: Value<Type = ArrayIrType> + DimensionSize<V> + ValueProjection<DimensionType>,
    V::DispatchDomain: Context<Type = ArrayIrType> + NamedAxes,
    <V::DispatchDomain as Domain>::Constant: From<DimensionValue>,
    <V::DispatchDomain as Domain>::Operation: From<AllGatherOperation>,
    <V as ValueProjection<DimensionType>>::Projected: Mul,
{
    fn all_gather_with_options(
        &self,
        axis_name: &str,
        concat_axis: usize,
        options: CollectiveOptions,
        output_variance: AllGatherOutputVariance,
    ) -> Result<Self, ProgramError> {
        let context = self.dispatch_domain();
        let axis_size = resolve_named_axis_size(&context, axis_name)?;
        let effective_axis_size = options.effective_axis_size(ALL_GATHER_OPERATION_NAME, axis_size)?;
        if output_variance != AllGatherOutputVariance::Varying && options.axis_index_groups.is_some() {
            return Err(TypeError::invalid(
                "`all_gather` axis index groups are not supported with invariant or reduced output variance"
                    .to_string(),
            )
            .into());
        }
        let operation =
            AllGatherOperation::new(axis_name.to_string(), axis_size, concat_axis, options.clone(), output_variance);
        let mut output_extents = collective_input_extents(&context, self)?;
        match options.mode {
            CollectiveMode::Untiled => {
                if concat_axis > output_extents.len() {
                    return Err(TypeError::invalid(format!(
                        "`all_gather` concat axis {concat_axis} is out of bounds for rank {}",
                        output_extents.len(),
                    ))
                    .into());
                }
                output_extents.insert(concat_axis, collective_extent_constant(&context, effective_axis_size)?);
            }
            CollectiveMode::Tiled => {
                let rank = output_extents.len();
                let Some(output_extent) = output_extents.get_mut(concat_axis) else {
                    return Err(TypeError::invalid(format!(
                        "`all_gather` concat axis {concat_axis} is out of bounds for rank {rank}",
                    ))
                    .into());
                };
                *output_extent = multiplied_collective_extent(&context, output_extent, effective_axis_size)?;
            }
        };
        let inputs = std::iter::once(self.clone()).chain(output_extents).collect::<Vec<_>>();
        Ok(context.bind(operation, Vec::new(), inputs.as_slice())?.remove(0))
    }
}

impl<V> AllGather for ProjectedValue<ArrayType, V>
where
    V: AllGather + ValueProjection<ArrayType, Projected = ProjectedValue<ArrayType, V>>,
{
    fn all_gather_with_options(
        &self,
        axis_name: &str,
        concat_axis: usize,
        options: CollectiveOptions,
        output_variance: AllGatherOutputVariance,
    ) -> Result<Self, ProgramError> {
        self.value()
            .all_gather_with_options(axis_name, concat_axis, options, output_variance)?
            .into_projected()
            .map_err(Into::into)
    }
}

/// Applies the mixed array IR JVP for invariant all-gather. Its transpose selects the current participant's
/// gathered chunk using the retained input geometry and reshapes an untiled size-one participant axis away.
fn jvp_invariant_all_gather<C>(
    operation: &AllGatherOperation,
    context: &C,
    inputs: &[DifferentiationDual<C::Value>],
) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError>
where
    C: Context<Type = ArrayIrType>,
    C::Operation: From<AllGatherOperation>
        + From<DimensionFromScalarOperation>
        + From<DimensionSizeOperation>
        + From<DynamicShapeSliceOperation>
        + From<LinearCallOperation<ArrayIrType>>
        + From<DynamicReshapeOperation>
        + From<ConstantOperation<DimensionValue>>
        + OperationProjection<ArrayType>
        + OperationProjection<DimensionType, Projected = DimensionOperation<DimensionValue>>,
    <C::Operation as OperationProjection<ArrayType>>::Projected: From<AxisIndexOperation>,
{
    let Some((array, output_extents)) = inputs.split_first() else {
        return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
    };
    let primal_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
    let primal = context.bind(operation.clone(), Vec::new(), primal_inputs.as_slice())?.remove(0);
    let tangent = match array.tangent() {
        MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().tangent()?),
        MaybeZero::Value(array_tangent) => {
            let mut residuals = LinearResiduals::new();
            let output_extents = residuals.retain_all(output_extents.iter().map(|extent| extent.primal().clone()));
            let input_shape = residuals.retain_shape(context, array.primal())?;
            let forward_operation = operation.clone();
            let forward_output_extents = output_extents.clone();
            let transpose_operation = operation.clone();
            let transpose_target_type = <&ArrayType>::try_from(array.primal().r#type().as_ref())?.cotangent()?;
            let tangent = LinearCallOperation::stage(
                context,
                residuals.into_values(),
                vec![array_tangent.clone()],
                move |residuals, linear_inputs| {
                    let mut collective_inputs = Vec::with_capacity(1 + forward_output_extents.len());
                    collective_inputs.push(linear_inputs[0].clone());
                    collective_inputs.extend(forward_output_extents.iter().map(|index| residuals[*index].clone()));
                    linear_inputs[0].dispatch_domain().bind(forward_operation, Vec::new(), collective_inputs.as_slice())
                },
                move |residuals, output_cotangents| {
                    let transpose_context = output_cotangents[0].dispatch_domain();
                    let input_dimensions = input_shape.dimensions(&transpose_context, residuals)?;
                    let output_cotangent_type = output_cotangents[0].r#type();
                    let output_cotangent_type = <&ArrayType>::try_from(output_cotangent_type.as_ref())?;
                    let output_rank = output_cotangent_type.rank();
                    let zero = transpose_context
                        .bind(
                            DimensionOperation::from(ConstantOperation::new(DimensionValue::constant(0)?)),
                            Vec::new(),
                            &[],
                        )?
                        .remove(0);
                    let chunk_extent = match transpose_operation.options().mode() {
                        CollectiveMode::Tiled => input_dimensions[transpose_operation.concat_axis()].clone(),
                        CollectiveMode::Untiled => transpose_context
                            .bind(
                                DimensionOperation::from(ConstantOperation::new(DimensionValue::constant(1)?)),
                                Vec::new(),
                                &[],
                            )?
                            .remove(0),
                    };
                    let start = if transpose_operation.axis_size() == 1 {
                        zero.clone()
                    } else {
                        let axis_index = transpose_context
                            .bind(
                                <C::Operation as OperationProjection<ArrayType>>::Projected::from(
                                    AxisIndexOperation::new(transpose_operation.axis_name().to_string()),
                                ),
                                Vec::new(),
                                &[],
                            )?
                            .remove(0);
                        let axis_index_variable = DimensionVariable::new(
                            format!("{}_index", transpose_operation.axis_name()),
                            crate::arrays::DimensionBounds::non_negative(Some(transpose_operation.axis_size()))?,
                        );
                        let axis_index = transpose_context
                            .bind(
                                DimensionFromScalarOperation::new(axis_index_variable),
                                Vec::new(),
                                std::slice::from_ref(&axis_index),
                            )?
                            .remove(0);
                        let axis_index_type = <&DimensionType>::try_from(axis_index.r#type().as_ref())?.clone();
                        let chunk_extent_type = <&DimensionType>::try_from(chunk_extent.r#type().as_ref())?.clone();
                        transpose_context
                            .bind(
                                DimensionOperation::Mul(DimensionMulOperation::new(
                                    &axis_index_type,
                                    &chunk_extent_type,
                                )?),
                                Vec::new(),
                                &[axis_index, chunk_extent.clone()],
                            )?
                            .remove(0)
                    };
                    let mut starts = vec![zero; output_rank];
                    starts[transpose_operation.concat_axis()] = start;
                    let mut slice_sizes = input_dimensions.clone();
                    if transpose_operation.options().mode() == CollectiveMode::Untiled {
                        slice_sizes.insert(transpose_operation.concat_axis(), chunk_extent);
                    }
                    let mut slice_inputs = Vec::with_capacity(1 + 2 * output_rank);
                    slice_inputs.push(output_cotangents[0].clone());
                    slice_inputs.extend(starts);
                    slice_inputs.extend(slice_sizes);
                    let selected = transpose_context
                        .bind(DynamicShapeSliceOperation::new(output_rank), Vec::new(), slice_inputs.as_slice())?
                        .remove(0);
                    let mut reshape_inputs = Vec::with_capacity(1 + input_dimensions.len());
                    reshape_inputs.push(selected);
                    reshape_inputs.extend(input_dimensions);
                    transpose_context.bind(
                        DynamicReshapeOperation::new().with_output_sharding(transpose_target_type.sharding().cloned()),
                        Vec::new(),
                        reshape_inputs.as_slice(),
                    )
                },
            )?
            .remove(0);
            MaybeZero::Value(tangent)
        }
    };
    Ok(vec![DifferentiationDual::new(primal, tangent)?])
}

// Mixed array IR JVP for all-gather. Explicit output extents are retained as ordinary residual values, and an
// invariant result uses participant-indexed slicing in its transposed linear region.
impl<C> MemberDifferentiableOperation<C> for AllGatherOperation
where
    C: Context<Type = ArrayIrType>,
    C::Operation: From<AllGatherOperation>
        + From<DimensionFromScalarOperation>
        + From<DimensionSizeOperation>
        + From<DynamicShapeSliceOperation>
        + From<LinearCallOperation<ArrayIrType>>
        + From<ParallelSumScatterOperation>
        + From<DynamicReshapeOperation>
        + From<ConstantOperation<DimensionValue>>
        + OperationProjection<ArrayType>
        + OperationProjection<DimensionType, Projected = DimensionOperation<DimensionValue>>,
    <C::Operation as OperationProjection<ArrayType>>::Projected: From<AxisIndexOperation>,
{
    fn jvp_in_parent<D: DifferentiationDriver<C>>(
        &self,
        context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        if self.output_variance() == AllGatherOutputVariance::Invariant {
            return jvp_invariant_all_gather(self, context, inputs);
        }
        jvp_shape_changing_collective_with_adjoint(
            self,
            ParallelSumScatterOperation::new(
                self.axis_name().to_string(),
                self.axis_size(),
                self.concat_axis(),
                self.options().clone(),
            ),
            context,
            inputs,
        )
    }
}

/// Returns the physical concat axis and mapped result axis for a forwarded all-gather.
fn forwarded_all_gather_axes(mode: CollectiveMode, concat_axis: usize, batch_axis: usize) -> (usize, usize) {
    match mode {
        CollectiveMode::Tiled => (concat_axis + usize::from(concat_axis >= batch_axis), batch_axis),
        CollectiveMode::Untiled if concat_axis <= batch_axis => (concat_axis, batch_axis + 1),
        CollectiveMode::Untiled => (concat_axis + 1, batch_axis),
    }
}

/// Applies the matching-axis all-gather batching semantics over the policy-selected extent representation.
fn batch_all_gather_matching_axis<C, P>(
    operation: &AllGatherOperation,
    context: &BatchingContext<C, ArrayBatching<P>>,
    input: &ArrayBatch<C::Value>,
    logical_input_rank: usize,
    output_extents: Vec<P::ShapeExtent>,
    output_sharding: Option<Sharding>,
) -> Result<ArrayBatch<C::Value>, BatchingError>
where
    C: Context<Type = ArrayType>,
    C::Value: Transpose,
    P: CollectiveBatchingPolicy<C>,
{
    if operation.options.axis_index_groups.is_some() {
        return Err(BatchingError::UnsupportedOperation {
            message: "`all_gather` axis index groups are not supported when a batch transform binds the collective \
                      axis"
                .to_string(),
        });
    }
    if operation.output_variance == AllGatherOutputVariance::Reduced {
        return Err(BatchingError::UnsupportedOperation {
            message: "`all_gather` with reduced output variance is not supported when a batch transform binds the \
                      collective axis"
                .to_string(),
        });
    }
    let axis_extent =
        P::collective_axis_extent(context, ALL_GATHER_OPERATION_NAME, &operation.axis_name, operation.axis_size)?;

    let axis_is_out_of_bounds = match operation.options.mode {
        CollectiveMode::Untiled => operation.concat_axis > logical_input_rank,
        CollectiveMode::Tiled => operation.concat_axis >= logical_input_rank,
    };
    if axis_is_out_of_bounds {
        return Err(BatchingError::UnsupportedOperation {
            message: format!(
                "`all_gather` concat axis {} is out of bounds for rank {logical_input_rank}",
                operation.concat_axis,
            ),
        });
    }

    let mut input_extents = output_extents.clone();
    match operation.options.mode {
        CollectiveMode::Untiled => {
            input_extents.remove(operation.concat_axis);
        }
        CollectiveMode::Tiled => {
            P::require_divisible_collective_extents(&output_extents[operation.concat_axis], &axis_extent)?;
            input_extents[operation.concat_axis] = output_extents[operation.concat_axis].div(&axis_extent)?;
        }
    }
    let input = P::match_collective_axis(context, input, input_extents.as_slice())?;
    let moved = input.into_value().move_axis(0, operation.concat_axis)?;
    let gathered = P::reshape_collective(context, moved, output_extents.as_slice(), output_sharding)?;
    Ok(ArrayBatch::replicated(gathered))
}

/// Relocates bounded-ragged metadata through a matching untiled all-gather.
fn gathered_ragged_axes<C, P>(
    operation: &AllGatherOperation,
    context: &BatchingContext<C, ArrayBatching<P>>,
    ragged_axes: Vec<RaggedAxis<C::Value>>,
    input_batch_axis: Option<usize>,
    input_rank: usize,
) -> Result<Vec<RaggedAxis<C::Value>>, BatchingError>
where
    C: Context<Type = ArrayType>,
    P: CollectiveBatchingPolicy<C>,
{
    if let Some(input_batch_axis) = input_batch_axis {
        return Ok(ragged_axes
            .into_iter()
            .map(|ragged_axis| ragged_axis.moved(input_batch_axis, 0).moved(0, operation.concat_axis))
            .collect());
    }

    let output_axes = (1..=input_rank).collect::<Vec<_>>();
    ragged_axes
        .into_iter()
        .map(|ragged_axis| {
            if !ragged_axis.extent_axes().is_empty() {
                return Err(BatchingError::UnsupportedOperation {
                    message: "untiled `all_gather` requires replicated ragged inputs to carry scalar extents"
                        .to_string(),
                });
            }
            let extents =
                P::match_axis(context, &ArrayBatch::replicated(ragged_axis.extents().clone()), 0.into())?.into_value();
            let ragged_axis = ragged_axis.broadcasted(output_axes.as_slice()).moved(0, operation.concat_axis);
            Ok(RaggedAxis::new(
                ragged_axis.axis(),
                extents,
                ragged_axis.dimension().clone(),
                vec![operation.concat_axis],
            ))
        })
        .collect()
}

// Batching rule for [`AllGatherOperation`]. A matching `batch` level consumes the mapped batch axis by
// materializing the gather: the batch axis is transposed to sit immediately before the per-item `concat_axis` and
// merged into it, laying the gathered chunks out item-major (item 0's chunk first), which matches the tiled
// StableHLO `all_gather` ordering. Every batch item sees the same gathered value, so the output is replicated. A
// non-matching level forwards the collective untouched to the parent context via [`forward_collective_to_parent`].
impl<C, P: CollectiveBatchingPolicy<C>> BatchableOperation<C, ArrayBatching<P>> for AllGatherOperation
where
    C: Context<Type = ArrayType>,
    C::Operation: From<AllGatherOperation>,
    <C as Domain>::Value: Transpose,
{
    fn batch<D: BatchingDriver<C, ArrayBatching<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatching<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<<C as Domain>::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatching<P>>, BatchingError> {
        if context.axis_name() != Some(self.axis_name.as_str()) {
            reject_ragged_collective_inputs(self.name(), inputs)?;
            let [input] = inputs else {
                return Err(ProgramError::InvalidInputCount { expected: 1, actual: inputs.len() }.into());
            };
            let Some(batch_axis) = input.batch_axis_position() else {
                return Ok(forward_collective_to_parent(context, C::Operation::from(self.clone()), inputs)?.into());
            };
            let (concat_axis, output_batch_axis) =
                forwarded_all_gather_axes(self.options.mode, self.concat_axis, batch_axis);
            let operation = Self::new(
                self.axis_name.clone(),
                self.axis_size,
                concat_axis,
                self.options.clone(),
                self.output_variance,
            );
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
        if self.options.mode == CollectiveMode::Tiled && !input.ragged_axes().is_empty() {
            return Err(BatchingError::UnsupportedOperation {
                message: "tiled `all_gather` cannot represent participant-specific bounded ragged extents after the \
                          participant and concatenation axes are fused"
                    .to_string(),
            });
        }
        let input_type = if input.ragged_axes().is_empty() {
            input.unbatched_type()
        } else {
            input.value().r#type().unbatched_type(input.batch_axis())?
        };
        let mut output_types = self.infer_output_types(std::slice::from_ref(&input_type), &[])?;
        let output_type = output_types.remove(0);
        let output_extents = output_type
            .shape()
            .dimensions()
            .iter()
            .map(|dimension| P::collective_extent_from_dimension(context, dimension))
            .collect::<Result<Vec<_>, _>>()?;
        let input_batch_axis = input.batch_axis_position();
        let ragged_axes = input.ragged_axes().to_vec();
        let mut output = batch_all_gather_matching_axis::<C, P>(
            self,
            context,
            input,
            input_type.rank(),
            output_extents,
            output_type.sharding().cloned(),
        )?;
        if !ragged_axes.is_empty() {
            let ragged_axes =
                gathered_ragged_axes::<C, P>(self, context, ragged_axes, input_batch_axis, input_type.rank())?;
            output = output.with_ragged_axes(ragged_axes)?;
        }
        Ok(vec![output].into())
    }
}

// Batching rule for explicit-extent [`AllGatherOperation`]. The logical result extents remain ordinary replicated
// dimension SSA operands; matching-axis batching delegates its array mechanics to the homogeneous collective kernel.
impl<C> MemberBatchableOperation<C, ArrayIrBatching> for AllGatherOperation
where
    C: Context<
            Type = ArrayIrType,
            Operation: From<AllGatherOperation>
                           + From<DynamicBroadcastOperation>
                           + From<ConstantOperation<DimensionValue>>
                           + From<DimensionSizeOperation>
                           + From<DynamicReshapeOperation>
                           + OperationProjection<ArrayType>,
        >,
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Value:
        ValueProjection<ArrayType, Projected: Transpose + Value<Type = ArrayType>> + ValueProjection<DimensionType>,
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
        let logical_input_types = inputs.iter().map(|input| input.unbatched_type().clone()).collect::<Vec<_>>();
        let mut logical_output_types = infer_explicit_all_gather_output_types(self, logical_input_types.as_slice())?;
        let logical_output_type = <&ArrayType>::try_from(&logical_output_types.remove(0))?.clone();

        if context.axis_name() != Some(self.axis_name()) {
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
            if array.batch_axis().is_replicated() {
                return Ok(forward_explicit_collective(self.clone(), context, array, output_extents, None)?.into());
            }
            let input_batch_axis = array.batch_axis_position().unwrap();
            let (physical_concat_axis, output_batch_axis) =
                forwarded_all_gather_axes(self.options().mode(), self.concat_axis(), input_batch_axis);
            let operation = Self::new(
                self.axis_name().to_string(),
                self.axis_size(),
                physical_concat_axis,
                self.options().clone(),
                self.output_variance(),
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

        if self.options().mode() == CollectiveMode::Tiled && !array.ragged_axes().is_empty() {
            return Err(BatchingError::UnsupportedOperation {
                message: "tiled `all_gather` cannot represent participant-specific bounded ragged extents after the \
                          participant and concatenation axes are fused"
                    .to_string(),
            });
        }

        let input_batch_axis = array.batch_axis_position();
        let ragged_axes = array
            .ragged_axes()
            .iter()
            .map(|ragged_axis| {
                let logical_axis =
                    ragged_axis.axis() - usize::from(input_batch_axis.is_some_and(|axis| axis < ragged_axis.axis()));
                let output_axis = logical_axis + usize::from(logical_axis >= self.concat_axis());
                let output_extent = &output_extents[output_axis];
                let output_extent_type = output_extent.unbatched_type();
                let output_extent_type = <&DimensionType>::try_from(&output_extent_type)?;
                if output_extent_type.variable() != ragged_axis.dimension() {
                    return Err(BatchingError::InvalidBatchMetadata {
                        message: format!(
                            "untiled `all_gather` output axis {output_axis} carries dimension `{}` instead of \
                             bounded ragged dimension `{}`",
                            output_extent_type.variable(),
                            ragged_axis.dimension(),
                        ),
                    });
                }
                let extents = if let Some(input_batch_axis) = input_batch_axis {
                    let Some(extents) = output_extent.mapped_dimension_extents() else {
                        return Err(BatchingError::InvalidBatchMetadata {
                            message: format!(
                                "untiled `all_gather` output axis {output_axis} must carry mapped extents for bounded \
                                 ragged dimension `{}`",
                                ragged_axis.dimension(),
                            ),
                        });
                    };
                    let expected_extent_axis = ragged_axis
                        .extent_axes()
                        .iter()
                        .position(|axis| *axis == input_batch_axis)
                        .map(BatchAxis::from_position)
                        .ok_or_else(|| BatchingError::InvalidBatchMetadata {
                            message: format!(
                                "bounded ragged dimension `{}` does not carry extents for the mapped input axis",
                                ragged_axis.dimension(),
                            ),
                        })?;
                    if output_extent.batch_axis() != expected_extent_axis {
                        return Err(BatchingError::InvalidBatchMetadata {
                            message: format!(
                                "untiled `all_gather` output axis {output_axis} maps bounded ragged extents on {} \
                                 instead of {expected_extent_axis}",
                                output_extent.batch_axis(),
                            ),
                        });
                    }
                    extents.clone()
                } else {
                    output_extent.validate_replicated_dimension()?;
                    if !ragged_axis.extent_axes().is_empty() {
                        return Err(BatchingError::InvalidBatchMetadata {
                            message: format!(
                                "replicated bounded ragged dimension `{}` must carry scalar extents",
                                ragged_axis.dimension(),
                            ),
                        });
                    }
                    ragged_axis.extents().clone()
                };
                Ok((
                    output_axis,
                    RaggedAxis::new(
                        ragged_axis.axis(),
                        <C::Value as ValueProjection<ArrayType>>::into_projected(extents)?,
                        ragged_axis.dimension().clone(),
                        ragged_axis.extent_axes().to_vec(),
                    ),
                ))
            })
            .collect::<Result<Vec<_>, BatchingError>>()?;
        let array = ArrayBatch::new(
            <C::Value as ValueProjection<ArrayType>>::into_projected(array.value().clone())?,
            array.batch_axis(),
        )?;
        let input_rank = array.unbatched_type().rank();
        let projected_context = BatchingContext::<_, ArrayBatching<DynamicArrayBatchingPolicy>>::with_policy(
            ProjectedContext::new(context.parent().clone()),
            context.axis_extent().clone(),
        )
        .with_axis_name(context.axis_name().map(str::to_string))
        .with_axis_sharding(context.axis_sharding().clone());
        let output_extents = output_extents
            .iter()
            .enumerate()
            .map(|(axis, extent)| {
                if ragged_axes.iter().any(|(ragged_output_axis, _)| *ragged_output_axis == axis) {
                    let extent_type = extent.unbatched_type();
                    let extent_type = <&DimensionType>::try_from(&extent_type)?;
                    let physical_extent =
                        extent_type.bounds().upper().and_then(|upper| upper.checked_sub(1)).ok_or_else(|| {
                            BatchingError::InvalidBatchMetadata {
                                message: format!(
                                    "bounded ragged dimension `{}` requires a finite, nonempty declared upper bound",
                                    extent_type.variable(),
                                ),
                            }
                        })?;
                    return DynamicArrayBatchingPolicy::collective_extent_constant(&projected_context, physical_extent);
                }
                if extent.mapped_dimension_extents().is_some() {
                    let extent_type = extent.unbatched_type();
                    let extent_type = <&DimensionType>::try_from(&extent_type)?;
                    return Err(BatchingError::InvalidBatchMetadata {
                        message: format!(
                            "untiled `all_gather` output axis {axis} has mapped dimension `{}` without a matching \
                             bounded ragged input axis",
                            extent_type.variable(),
                        ),
                    });
                }
                extent.validate_replicated_dimension()?;
                Ok(<C::Value as ValueProjection<DimensionType>>::into_projected(extent.value().clone())?)
            })
            .collect::<Result<Vec<_>, BatchingError>>()?;
        let ragged_axes = ragged_axes.into_iter().map(|(_, ragged_axis)| ragged_axis).collect::<Vec<_>>();
        let mut output = batch_all_gather_matching_axis::<_, DynamicArrayBatchingPolicy>(
            self,
            &projected_context,
            &array,
            input_rank,
            output_extents,
            logical_output_type.sharding().cloned(),
        )?;
        if !ragged_axes.is_empty() {
            let ragged_axes = gathered_ragged_axes::<_, DynamicArrayBatchingPolicy>(
                self,
                &projected_context,
                ragged_axes,
                input_batch_axis,
                input_rank,
            )?;
            output = output.with_ragged_axes(ragged_axes)?;
        }
        let ragged_axes = output
            .ragged_axes()
            .iter()
            .map(|ragged_axis| {
                RaggedAxis::new(
                    ragged_axis.axis(),
                    <C::Value as ValueProjection<ArrayType>>::from_projected(ragged_axis.extents().clone()),
                    ragged_axis.dimension().clone(),
                    ragged_axis.extent_axes().to_vec(),
                )
            })
            .collect();
        let output =
            ArrayIrBatch::replicated(<C::Value as ValueProjection<ArrayType>>::from_projected(output.into_value()))
                .with_ragged_axes(ragged_axes)?;
        Ok(vec![output].into())
    }
}

// Transpose rule for [`AllGatherOperation`]. A varying all-gather is the adjoint of a sum-scatter with the same
// mode, axis, and participant groups, so the operand cotangent is a [`ParallelSumScatterOperation`] of the output
// cotangent. Invariant and reduced variance require the residual-aware composite adjoints because their pullbacks
// depend on participant-indexed runtime geometry.
impl<V, O> TransposableOperation<V, O> for AllGatherOperation
where
    V: Value<Type = ArrayType>,
    O: Operation<Type = ArrayType> + From<ParallelSumScatterOperation>,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TracingContext<V, O>,
        _driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        if self.output_variance == AllGatherOutputVariance::Invariant {
            return Err(ProgramError::UnsupportedOperation {
                message: "direct transposition of invariant `all_gather` cannot represent the participant-indexed \
                          slice; linearize so that the current participant can select its gathered chunk"
                    .to_string(),
            }
            .into());
        }
        transpose_shape_changing_collective(
            context,
            inputs,
            outputs,
            ParallelSumScatterOperation::new(
                self.axis_name.clone(),
                self.axis_size,
                self.concat_axis,
                self.options.clone(),
            ),
        )
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrValue, ArrayOperation, DataType, DimensionBounds, LogicalMesh, MeshAxis,
        MeshAxisType, RaggedAxis, Sharding,
    };
    use crate::axes::AxisError;
    use crate::batching::{BatchAxis, BatchAxisSpecification, BatchingContext, batch};
    use crate::contexts::EagerContext;
    use crate::operations::collectives::parallel_sum_scatter::infer_explicit_parallel_sum_scatter_output_types;
    use crate::operations::collectives::tests::f32_vector;

    use super::*;

    #[test]
    fn test_all_gather_output_variance_updates_canonical_sharding_state() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let varying_sharding = Sharding::replicated(mesh, 1).with_varying_manual_axes(["x"]).unwrap();
        let input = f32_vector(3).with_sharding(varying_sharding).unwrap();

        let infer = |output_variance| {
            infer_explicit_all_gather_output_types(
                &AllGatherOperation::new("x".to_string(), 2, 0, CollectiveOptions::default(), output_variance),
                &[
                    ArrayIrType::Array(input.clone()),
                    DimensionValue::constant(2).unwrap().r#type().into_owned().into(),
                    DimensionValue::constant(3).unwrap().r#type().into_owned().into(),
                ],
            )
        };
        let varying = infer(AllGatherOutputVariance::Varying).unwrap();
        let varying = <&ArrayType>::try_from(&varying[0]).unwrap();
        assert_eq!(varying.sharding().unwrap().varying_manual_axes(), &["x".to_string()].into_iter().collect());
        assert!(varying.sharding().unwrap().reduced_axes().is_empty());

        let invariant = infer(AllGatherOutputVariance::Invariant).unwrap();
        let invariant = <&ArrayType>::try_from(&invariant[0]).unwrap();
        assert!(invariant.sharding().unwrap().varying_manual_axes().is_empty());
        assert!(invariant.sharding().unwrap().reduced_axes().is_empty());

        let reduced = infer(AllGatherOutputVariance::Reduced).unwrap();
        let reduced = <&ArrayType>::try_from(&reduced[0]).unwrap();
        assert!(reduced.sharding().unwrap().varying_manual_axes().is_empty());
        assert_eq!(reduced.sharding().unwrap().reduced_axes(), &["x".to_string()].into_iter().collect());

        // The cotangent of a reduced gather result is unreduced. Sum-scatter consumes exactly that marker and
        // restores the varying operand-cotangent state without a second reduce-scatter operation type.
        let reduced_cotangent = reduced.cotangent().unwrap();
        assert_eq!(
            infer_explicit_parallel_sum_scatter_output_types(
                &ParallelSumScatterOperation::new("x".to_string(), 2, 0, CollectiveOptions::default()),
                &[reduced_cotangent.into(), DimensionValue::constant(3).unwrap().r#type().into_owned().into(),],
            ),
            Ok(vec![input.cotangent().unwrap().into()]),
        );
    }

    #[test]
    fn test_all_gather_type_inference() {
        use crate::macros::check_operation_type_inference;

        let operation = AllGatherOperation::new(
            "x".to_string(),
            4,
            0,
            CollectiveOptions::tiled(),
            AllGatherOutputVariance::Varying,
        );
        assert_eq!(operation.axis_name(), "x");
        assert_eq!(operation.axis_size(), 4);
        assert_eq!(operation.concat_axis(), 0);
        assert_eq!(operation.name(), ALL_GATHER_OPERATION_NAME);
        assert_eq!(
            operation.to_string(),
            indoc::indoc! {r#"
                all_gather [
                    axis_name="x",
                    axis_size=4,
                    concat_axis=0,
                    options=Tiled,
                    output_variance=Varying,
                ]
            "#}
            .trim_end(),
        );
        check_operation_type_inference!(
            operation = operation,
            cases = [
                {
                    input_types = [f32_vector(2)],
                    output_types = [f32_vector(8)],
                },
                {
                    input_types = [ArrayType::scalar(DataType::F32)],
                    error = "`all_gather` concat axis 0 is out of bounds for rank 0",
                },
                {
                    input_types = [ArrayType::new(
                        DataType::F32,
                        Shape::new(vec![Dimension::Dynamic(
                            DimensionVariable::new("dynamic", DimensionBounds::unbounded()),
                        )]),
                    )],
                    error = "`all_gather` does not support dynamically shaped operands",
                },
            ],
        );
        check_operation_type_inference!(
            @reject @unreduced,
            operation = AllGatherOperation::new(
                "x".to_string(),
                4,
                0,
                CollectiveOptions::tiled(),
                AllGatherOutputVariance::Varying,
            ),
            input_types = [f32_vector(2)],
        );
    }

    #[test]
    fn test_all_gather_interpretation_requires_an_enclosing_binder() {
        use crate::interpretation::InterpretableOperation;

        // A single-participant axis is degenerate: the gather concatenates exactly one operand, so interpretation is
        // the identity.
        let outputs = AllGatherOperation::new(
            "x".to_string(),
            1,
            0,
            CollectiveOptions::tiled(),
            AllGatherOutputVariance::Varying,
        )
        .interpret(
            &EagerContext::<Array, ArrayOperation<Array>>::new(),
            &crate::EmptyRegionDriver,
            &[Array::vector(vec![1.0, 2.0])],
        )
        .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].to_f64s(), vec![1.0, 2.0]);

        // Any larger axis has no per-item semantics: the other participants do not exist outside an enclosing binder.
        let error = AllGatherOperation::new(
            "x".to_string(),
            2,
            0,
            CollectiveOptions::tiled(),
            AllGatherOutputVariance::Varying,
        )
        .interpret(
            &EagerContext::<Array, ArrayOperation<Array>>::new(),
            &crate::EmptyRegionDriver,
            &[Array::vector(vec![1.0, 2.0])],
        )
        .unwrap_err();
        assert!(matches!(
            error,
            ProgramError::UnsupportedOperation { message }
                if message == "cannot interpret `all_gather` over axis `x` of size 2 without an enclosing binder",
        ));
    }

    #[test]
    fn test_all_gather_over_unbound_axis_is_rejected() {
        use crate::batching::BatchingTracer;

        // The batch binds only the axis `"i"`, but the `all_gather` names `"x"`, which no enclosing transform binds.
        // Axis-size resolution fails fast at staging time with `AxisError::UnboundAxisName` rather than silently
        // acting as identity.
        let result: Result<ArrayIrValue<Array>, BatchingError> = batch(
            |item: BatchingTracer<EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>, ArrayIrBatching>| {
                item.all_gather_tiled("x", 0)
            },
            ArrayIrValue::Array(Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0])),
            BatchAxis::new(0),
            BatchAxis::replicated(),
            BatchAxisSpecification::named("i"),
        );
        assert_eq!(result.unwrap_err(), BatchingError::Axis(AxisError::UnboundAxisName { name: "x".to_string() }));
    }

    #[test]
    fn test_all_gather_forwarded_axes_account_for_the_mapped_axis() {
        assert_eq!(forwarded_all_gather_axes(CollectiveMode::Tiled, 0, 0), (1, 0));
        assert_eq!(forwarded_all_gather_axes(CollectiveMode::Tiled, 0, 1), (0, 1));
        assert_eq!(forwarded_all_gather_axes(CollectiveMode::Untiled, 0, 0), (0, 1));
        assert_eq!(forwarded_all_gather_axes(CollectiveMode::Untiled, 1, 0), (2, 0));
    }

    #[test]
    fn test_all_gather_over_batched_axis_materializes_the_gather() {
        use crate::batching::BatchingTracer;

        // The batch binds the axis `"x"` that the `all_gather` names, so the matching batching rule consumes the
        // mapped axis: every item receives the item-major concatenation of all items along `concat_axis`,
        // replicated across the batch. With items `[1, 2]` and `[3, 4]` the gathered value is `[1, 2, 3, 4]`,
        // matching the verified cross-device `shard_map` execution semantics of the tiled StableHLO `all_gather`.
        let output: ArrayIrValue<Array> = batch(
            |item: BatchingTracer<EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>, ArrayIrBatching>| {
                item.all_gather_tiled("x", 0)
            },
            ArrayIrValue::Array(Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0])),
            BatchAxis::new(0),
            BatchAxis::replicated(),
            BatchAxisSpecification::named("x"),
        )
        .unwrap();
        assert_eq!(
            output.r#type().into_owned(),
            ArrayIrType::Array(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(4)]))),
        );
        let ArrayIrValue::Array(output) = output else {
            panic!("`all_gather` must preserve the array member kind");
        };
        assert_eq!(output.to_f64s(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_untiled_all_gather_co_moves_ragged_extents_onto_the_participant_axis() {
        let variable = DimensionVariable::new("length", DimensionBounds::new(0, Some(3)).unwrap());
        let input = ArrayBatch::new(Array::matrix(2, 3, vec![1.0, 0.0, 0.0, 2.0, 3.0, 4.0]), BatchAxis::new(0))
            .unwrap()
            .with_ragged_axes(vec![RaggedAxis::new(1, Array::vector(vec![1_i32, 3]), variable.clone(), vec![0])])
            .unwrap();
        let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 2)
            .with_axis_name("x".to_string());
        let operation = AllGatherOperation::new(
            "x".to_string(),
            2,
            0,
            CollectiveOptions::default(),
            AllGatherOutputVariance::Varying,
        );

        let output = operation.batch(&context, &crate::EmptyRegionDriver, &[input]).unwrap().into_parts().0.remove(0);

        assert_eq!(output.batch_axis(), BatchAxis::replicated());
        assert_eq!(output.value().to_f64s(), vec![1.0, 0.0, 0.0, 2.0, 3.0, 4.0]);
        assert_eq!(output.ragged_axes(), &[RaggedAxis::new(1, Array::vector(vec![1_i32, 3]), variable, vec![0])],);
    }

    #[test]
    fn test_untiled_all_gather_materializes_replicated_ragged_extents() {
        let variable = DimensionVariable::new("length", DimensionBounds::new(0, Some(3)).unwrap());
        let input = ArrayBatch::replicated(Array::vector(vec![1.0_f32, 2.0, 0.0]))
            .with_ragged_axes(vec![RaggedAxis::new(0, Array::scalar(2_i32), variable.clone(), Vec::new())])
            .unwrap();
        let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 2)
            .with_axis_name("x".to_string());
        let operation = AllGatherOperation::new(
            "x".to_string(),
            2,
            0,
            CollectiveOptions::default(),
            AllGatherOutputVariance::Varying,
        );

        let output = operation.batch(&context, &crate::EmptyRegionDriver, &[input]).unwrap().into_parts().0.remove(0);

        assert_eq!(output.batch_axis(), BatchAxis::replicated());
        assert_eq!(output.value().to_f64s(), vec![1.0, 2.0, 0.0, 1.0, 2.0, 0.0]);
        assert_eq!(output.ragged_axes(), &[RaggedAxis::new(1, Array::vector(vec![2_i32, 2]), variable, vec![0])],);
    }

    #[test]
    fn test_tiled_all_gather_rejects_unrepresentable_ragged_chunks() {
        let variable = DimensionVariable::new("length", DimensionBounds::new(0, Some(3)).unwrap());
        let input = ArrayBatch::new(Array::matrix(2, 3, vec![1.0_f32; 6]), BatchAxis::new(0))
            .unwrap()
            .with_ragged_axes(vec![RaggedAxis::new(1, Array::vector(vec![1_i32, 3]), variable, vec![0])])
            .unwrap();
        let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 2)
            .with_axis_name("x".to_string());

        assert_eq!(
            AllGatherOperation::new(
                "x".to_string(),
                2,
                0,
                CollectiveOptions::tiled(),
                AllGatherOutputVariance::Varying,
            )
            .batch(&context, &crate::EmptyRegionDriver, &[input]),
            Err(BatchingError::UnsupportedOperation {
                message: "tiled `all_gather` cannot represent participant-specific bounded ragged extents after the \
                          participant and concatenation axes are fused"
                    .to_string(),
            }),
        );
    }

    #[test]
    fn test_array_ir_all_gather_preserves_untiled_ragged_metadata_and_rejects_tiled() {
        let variable = DimensionVariable::new("length", DimensionBounds::new(0, Some(4)).unwrap());
        let extents = ArrayIrValue::Array(Array::vector(vec![1_i32, 3]));
        let input = ArrayIrBatch::new(
            ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0_f32, 0.0, 0.0, 2.0, 3.0, 4.0])),
            BatchAxis::new(0),
        )
        .unwrap()
        .with_ragged_axes(vec![RaggedAxis::new(1, extents.clone(), variable.clone(), vec![0])])
        .unwrap();
        let extent =
            |value| ArrayIrBatch::replicated(ArrayIrValue::Dimension(DimensionValue::constant(value).unwrap()));
        let ragged_extent =
            ArrayIrBatch::mapped_dimension(extents.clone(), BatchAxis::new(0), DimensionType::new(variable.clone()))
                .unwrap();
        let context = BatchingContext::new(
            EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new(),
            ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()),
        )
        .with_axis_name("x".to_string());
        let untiled = AllGatherOperation::new(
            "x".to_string(),
            2,
            0,
            CollectiveOptions::default(),
            AllGatherOutputVariance::Varying,
        );
        let output = untiled
            .batch_in_parent(&context, &crate::EmptyRegionDriver, &[input.clone(), extent(2), ragged_extent])
            .unwrap()
            .into_parts()
            .0
            .remove(0);
        assert_eq!(output.ragged_axes(), &[RaggedAxis::new(1, extents, variable.clone(), vec![0])]);

        let tiled = AllGatherOperation::new(
            "x".to_string(),
            2,
            0,
            CollectiveOptions::tiled(),
            AllGatherOutputVariance::Varying,
        );
        assert_eq!(
            tiled.batch_in_parent(&context, &crate::EmptyRegionDriver, &[input, extent(6)]),
            Err(BatchingError::UnsupportedOperation {
                message: "tiled `all_gather` cannot represent participant-specific bounded ragged extents after the \
                          participant and concatenation axes are fused"
                    .to_string(),
            }),
        );
    }

    #[test]
    fn test_array_ir_untiled_all_gather_materializes_replicated_ragged_extents() {
        let variable = DimensionVariable::new("length", DimensionBounds::new(0, Some(4)).unwrap());
        let input = ArrayIrBatch::replicated(ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 0.0])))
            .with_ragged_axes(vec![RaggedAxis::new(
                0,
                ArrayIrValue::Array(Array::scalar(2_i32)),
                variable.clone(),
                Vec::new(),
            )])
            .unwrap();
        let extent =
            |value| ArrayIrBatch::replicated(ArrayIrValue::Dimension(DimensionValue::constant(value).unwrap()));
        let ragged_extent = ArrayIrBatch::replicated(ArrayIrValue::Dimension(
            DimensionValue::new(DimensionType::new(variable.clone()), 2).unwrap(),
        ));
        let context = BatchingContext::new(
            EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new(),
            ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()),
        )
        .with_axis_name("x".to_string());
        let output = AllGatherOperation::new(
            "x".to_string(),
            2,
            0,
            CollectiveOptions::default(),
            AllGatherOutputVariance::Varying,
        )
        .batch_in_parent(&context, &crate::EmptyRegionDriver, &[input, extent(2), ragged_extent])
        .unwrap()
        .into_parts()
        .0
        .remove(0);

        assert_eq!(output.value(), &ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0_f32, 2.0, 0.0, 1.0, 2.0, 0.0])),);
        assert_eq!(
            output.ragged_axes(),
            &[RaggedAxis::new(1, ArrayIrValue::Array(Array::vector(vec![2_i32, 2])), variable, vec![0],)],
        );
    }

    #[test]
    fn test_all_gather_of_replicated_input_concatenates_copies() {
        // A replicated operand at a matching level is first materialized as `axis_size` identical batch items, so
        // the gather degenerates to the item-major concatenation of that many copies of the shared value.
        let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 2)
            .with_axis_name("x".to_string());
        let outputs = AllGatherOperation::new(
            "x".to_string(),
            2,
            0,
            CollectiveOptions::tiled(),
            AllGatherOutputVariance::Varying,
        )
        .batch(&context, &crate::EmptyRegionDriver, &[ArrayBatch::replicated(Array::vector(vec![1.0, 2.0]))])
        .unwrap()
        .into_parts()
        .0;
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::replicated());
        assert_eq!(outputs[0].value().to_f64s(), vec![1.0, 2.0, 1.0, 2.0]);
    }

    #[test]
    fn test_all_gather_transposes_to_parallel_sum_scatter() {
        use crate::parameters::Placeholder;
        use crate::programs::ProgramBuilder;

        // A tiled all-gather is the adjoint of a sum-scatter over the same axis and dimension, so the pullback stages
        // a `parallel_sum_scatter` on the output cotangent with the gather's concat axis as its scatter axis.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(f32_vector(2));
        let output = builder
            .add_instruction(
                AllGatherOperation::new(
                    "x".to_string(),
                    2,
                    0,
                    CollectiveOptions::tiled(),
                    AllGatherOutputVariance::Varying,
                ),
                Vec::new(),
                vec![input],
                None,
            )
            .unwrap()[0];
        let program = builder.build::<Array, Array>(vec![output], Placeholder, Placeholder).unwrap();
        let pullback = program.transpose_with_respect_to(&[0]).unwrap();
        assert_eq!(
            pullback.to_string(),
            indoc::indoc! {r#"
                lambda %0:f32[4] .
                let %1:f32[2] = parallel_sum_scatter [axis_name="x", axis_size=2, scatter_axis=0, options=Tiled] %0
                in (%1)
            "#}
            .trim_end(),
        );

        let groups = vec![vec![0, 2], vec![3, 1]];
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(f32_vector(2));
        let output = builder
            .add_instruction(
                AllGatherOperation::new(
                    "x".to_string(),
                    4,
                    0,
                    CollectiveOptions::tiled().with_axis_index_groups(groups.clone()),
                    AllGatherOutputVariance::Varying,
                ),
                Vec::new(),
                vec![input],
                None,
            )
            .unwrap()[0];
        let program = builder.build::<Array, Array>(vec![output], Placeholder, Placeholder).unwrap();
        let pullback = program.transpose_with_respect_to(&[0]).unwrap();
        let ArrayOperation::ParallelSumScatter(adjoint) = pullback.instructions()[0].operation() else {
            panic!("expected grouped all-gather transpose to stage parallel-sum-scatter");
        };
        assert_eq!(adjoint.options().axis_index_groups(), Some(groups.as_slice()));

        // Reduced output variance swaps to an unreduced cotangent type. The same sum-scatter operation consumes that
        // state and returns the original varying operand cotangent.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let input_type = f32_vector(2)
            .with_sharding(Sharding::replicated(mesh, 1).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(input_type.clone());
        let output = builder
            .add_instruction(
                AllGatherOperation::new(
                    "x".to_string(),
                    2,
                    0,
                    CollectiveOptions::tiled(),
                    AllGatherOutputVariance::Reduced,
                ),
                Vec::new(),
                vec![input],
                None,
            )
            .unwrap()[0];
        let program = builder.build::<Array, Array>(vec![output], Placeholder, Placeholder).unwrap();
        let pullback = program.transpose_with_respect_to(&[0]).unwrap();
        assert!(matches!(pullback.instructions()[0].operation(), ArrayOperation::ParallelSumScatter(_)));
        assert_eq!(pullback.output_types(), vec![input_type.cotangent().unwrap()]);
    }
}
