//! Contains the named-axis [`AllToAllOperation`], which exchanges chunks between the participants along a named
//! axis, together with its interpretation, partial-evaluation, batching, forward-mode differentiation, and
//! transposition rules.

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
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::{
    MaybeZero, MemberOperation, Operation, OperationFormatter, OperationProjection, ProgramError, ProjectedValue,
    RegionInterface, TypeError, TypeIdentityRenaming, Typed, Value, ValueProjection,
};
use crate::tracing::{Tracer, TracingContext};

use super::{
    CollectiveBatchingPolicy, CollectiveMode, CollectiveOptions, collective_extent_constant, collective_input_extents,
    divided_collective_extent, explicit_collective_inputs, forward_collective_to_parent, forward_explicit_collective,
    forward_shape_changing_collective, impl_shape_changing_collective_member_operation,
    infer_explicit_shape_changing_collective_output_type, interpret_degenerate_collective,
    jvp_shape_changing_collective_with_adjoint, multiplied_collective_extent, require_collective_axis_divisible,
    require_collective_axis_extent, resolve_named_axis_size, shape_changing_collective,
    shape_changing_collective_dimensions, shape_changing_collective_output_type, transpose_shape_changing_collective,
    validate_collective_axis_size, validate_explicit_collective_output_extents,
};

/// Infers the composite all-to-all contract.
pub(crate) fn infer_explicit_all_to_all_output_types(
    operation: &AllToAllOperation,
    input_types: &[ArrayIrType],
) -> Result<Vec<ArrayIrType>, TypeError> {
    let effective_axis_size = operation.effective_axis_size()?;
    let Some(input_type) = input_types.first() else {
        return Err(TypeError::invalid("`all_to_all` expects an array followed by its output extents"));
    };
    let input_type = <&ArrayType>::try_from(input_type)?;
    if operation.options.mode == CollectiveMode::Untiled {
        let Some(input_extent) = input_type.shape().dimensions().get(operation.split_axis) else {
            return Err(TypeError::invalid(format!(
                "`all_to_all` split axis {} is out of bounds for rank {}",
                operation.split_axis,
                input_type.rank(),
            )));
        };
        if let Dimension::Static(input_extent) = input_extent
            && *input_extent != effective_axis_size
        {
            return Err(TypeError::invalid(format!(
                "`all_to_all` untiled split axis {} size {input_extent} must equal group size {effective_axis_size}",
                operation.split_axis,
            )));
        }
        let output_type = input_type
            .without_dimension(operation.split_axis)?
            .0
            .with_inserted_dimension(operation.concat_axis, Dimension::Static(effective_axis_size))?;
        let mut unchanged_input_axes =
            (0..input_type.rank()).filter(|axis| *axis != operation.split_axis).map(Some).collect::<Vec<_>>();
        unchanged_input_axes.insert(operation.concat_axis, None);
        return infer_explicit_shape_changing_collective_output_type(
            ALL_TO_ALL_OPERATION_NAME,
            input_types,
            output_type,
            unchanged_input_axes.as_slice(),
            |_, output_extents| {
                let output_extent = &output_extents[operation.concat_axis];
                if output_extent != &Dimension::Static(effective_axis_size) {
                    return Err(TypeError::invalid(format!(
                        "`all_to_all` inserted output axis {} extent must equal axis group size \
                         {effective_axis_size} but got {output_extent}",
                        operation.concat_axis,
                    )));
                }
                Ok(())
            },
        );
    }
    if operation.split_axis == operation.concat_axis {
        let Some(input_extent) = input_type.shape().dimensions().get(operation.split_axis) else {
            return Err(TypeError::invalid(format!(
                "`all_to_all` split axis {} is out of bounds for rank {}",
                operation.split_axis,
                input_type.rank(),
            )));
        };
        if let Dimension::Static(input_extent) = input_extent
            && *input_extent % effective_axis_size != 0
        {
            return Err(TypeError::invalid(format!(
                "`all_to_all` split axis {} size {input_extent} is not divisible by group size \
                 {effective_axis_size}",
                operation.split_axis,
            )));
        }
        return infer_explicit_shape_changing_collective_output_type(
            ALL_TO_ALL_OPERATION_NAME,
            input_types,
            input_type.clone(),
            &(0..input_type.rank()).map(Some).collect::<Vec<_>>(),
            |_, _| Ok(()),
        );
    }
    if operation.split_axis >= input_type.rank() || operation.concat_axis >= input_type.rank() {
        return Err(TypeError::invalid(format!(
            "`all_to_all` split axis {} or concat axis {} is out of bounds for rank {}",
            operation.split_axis,
            operation.concat_axis,
            input_type.rank(),
        )));
    }
    let mut dimensions = input_type.shape().dimensions().to_vec();
    dimensions[operation.split_axis] = Dimension::Static(0);
    dimensions[operation.concat_axis] = Dimension::Static(0);
    let sharding = resized_output_sharding(input_type, dimensions.as_slice(), ALL_TO_ALL_OPERATION_NAME)?;
    let mut base_output_type =
        ArrayType::new(input_type.data_type(), Shape::new(dimensions)).with_memory(input_type.memory());
    base_output_type.sharding = sharding;
    let unchanged_input_axes = (0..input_type.rank())
        .map(|axis| (axis != operation.split_axis && axis != operation.concat_axis).then_some(axis))
        .collect::<Vec<_>>();
    infer_explicit_shape_changing_collective_output_type(
        ALL_TO_ALL_OPERATION_NAME,
        input_types,
        base_output_type,
        unchanged_input_axes.as_slice(),
        |input_type, output_extents| {
            if let (Dimension::Static(input_extent), Dimension::Static(output_extent)) =
                (&input_type.shape().dimensions()[operation.split_axis], &output_extents[operation.split_axis])
            {
                if *input_extent % effective_axis_size != 0 {
                    return Err(TypeError::invalid(format!(
                        "`all_to_all` split axis {} size {input_extent} is not divisible by group size \
                         {effective_axis_size}",
                        operation.split_axis,
                    )));
                }
                let expected = *input_extent / effective_axis_size;
                if *output_extent != expected {
                    return Err(TypeError::invalid(format!(
                        "`all_to_all` split result extent must equal input axis {} extent {input_extent} divided by \
                         group size {effective_axis_size}; expected {expected} but got {output_extent}",
                        operation.split_axis,
                    )));
                }
            }
            if let (Dimension::Static(input_extent), Dimension::Static(output_extent)) =
                (&input_type.shape().dimensions()[operation.concat_axis], &output_extents[operation.concat_axis])
            {
                let expected = input_extent.checked_mul(effective_axis_size).ok_or_else(|| {
                    TypeError::invalid("`all_to_all` concatenation result extent does not fit in usize".to_string())
                })?;
                if *output_extent != expected {
                    return Err(TypeError::invalid(format!(
                        "`all_to_all` concat result extent must equal input axis {} extent {input_extent} multiplied \
                         by group size {effective_axis_size}; expected {expected} but got {output_extent}",
                        operation.concat_axis,
                    )));
                }
            }
            Ok(())
        },
    )
}

shape_changing_collective! {
    /// [`Operation`] that exchanges chunks between the participants along the named axis: every participant splits
    /// its operand into `axis_size` chunks along `split_axis` and receives the participants' chunks concatenated
    /// along `concat_axis` — the analogue of
    /// [JAX's `all_to_all`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.all_to_all.html) and
    /// [StableHLO's `all_to_all`](https://openxla.org/stablehlo/spec#all_to_all). The output shrinks `split_axis`
    /// by the axis size (the dimension must be divisible by it) and extends `concat_axis` by it. The collective is
    /// linear and its transpose is the exchange with the split and concatenation axes swapped. A matching `batch`
    /// level consumes the mapped batch axis with a reshape/transpose block exchange: batch item `i` receives every
    /// item's chunk `i` of `split_axis`, concatenated item-major along `concat_axis`.
    ///
    /// A bounded ragged operand is rejected even when its logical extents are available. One extent per item does not
    /// determine how each source partitions its live prefix among destinations; that requires the explicit input and
    /// output offsets and per-destination sizes of [`RaggedAllToAllOperation`](super::RaggedAllToAllOperation).
    operation = AllToAllOperation,
    name = ALL_TO_ALL_OPERATION_NAME = "all_to_all",
    fields = {
        /// Axis of the operand that is split into one chunk per participant.
        split_axis: usize,

        /// Axis of the output along which the received chunks are concatenated.
        concat_axis: usize,

        /// Shared rank and participant-group semantics.
        options: CollectiveOptions,
    },
    infer = |operation, input_type, dimensions| {
        let effective_axis_size = operation.effective_axis_size()?;
        let mut output_dimensions = dimensions;
        let rank = output_dimensions.len();
        if operation.split_axis >= rank || operation.concat_axis >= rank {
            return Err(TypeError::invalid(format!(
                    "`all_to_all` split axis {} or concat axis {} is out of bounds for rank {rank}",
                    operation.split_axis,
                    operation.concat_axis,
                )));
        }
        if operation.options.mode == CollectiveMode::Untiled {
            if output_dimensions[operation.split_axis] != effective_axis_size {
                return Err(TypeError::invalid(format!(
                    "`all_to_all` untiled split axis {} size {} must equal group size {}",
                    operation.split_axis,
                    output_dimensions[operation.split_axis],
                    effective_axis_size,
                )));
            }
            input_type
                .without_dimension(operation.split_axis)?
                .0
                .with_inserted_dimension(operation.concat_axis, Dimension::Static(effective_axis_size))
        } else {
            if output_dimensions[operation.split_axis] % effective_axis_size != 0 {
                return Err(TypeError::invalid(format!(
                    "`all_to_all` split axis {} size {} is not divisible by group size {}",
                    operation.split_axis,
                    output_dimensions[operation.split_axis],
                    effective_axis_size,
                )));
            }
            output_dimensions[operation.split_axis] /= effective_axis_size;
            output_dimensions[operation.concat_axis] = output_dimensions[operation.concat_axis]
                .checked_mul(effective_axis_size)
                .ok_or_else(|| {
                    TypeError::invalid("`all_to_all` concatenation result extent does not fit in usize".to_string())
                })?;
            shape_changing_collective_output_type(ALL_TO_ALL_OPERATION_NAME, input_type, output_dimensions)
        }
    },
}

impl AllToAllOperation {
    /// Returns the axis of the operand that is split into one chunk per participant.
    #[inline]
    pub fn split_axis(&self) -> usize {
        self.split_axis
    }

    /// Returns the axis of the output along which the received chunks are concatenated.
    #[inline]
    pub fn concat_axis(&self) -> usize {
        self.concat_axis
    }

    /// Returns the shared rank and participant-group semantics.
    #[inline]
    pub fn options(&self) -> &CollectiveOptions {
        &self.options
    }

    /// Returns the participant count used for result-shape arithmetic.
    #[inline]
    pub fn effective_axis_size(&self) -> Result<usize, TypeError> {
        self.options.effective_axis_size(ALL_TO_ALL_OPERATION_NAME, self.axis_size)
    }
}

impl_shape_changing_collective_member_operation!(AllToAllOperation, infer_explicit_all_to_all_output_types);

/// Stages an all-to-all with first-class dynamic tiled extents and rank-changing untiled semantics.
pub trait AllToAll: Sized {
    /// Maps `split_axis` onto the named axis and materializes that named axis at `concat_axis`.
    #[inline]
    fn all_to_all(&self, axis_name: &str, split_axis: usize, concat_axis: usize) -> Result<Self, ProgramError> {
        self.all_to_all_with_options(axis_name, split_axis, concat_axis, CollectiveOptions::default())
    }

    /// Exchanges equal chunks while preserving rank.
    #[inline]
    fn all_to_all_tiled(&self, axis_name: &str, split_axis: usize, concat_axis: usize) -> Result<Self, ProgramError> {
        self.all_to_all_with_options(axis_name, split_axis, concat_axis, CollectiveOptions::new(CollectiveMode::Tiled))
    }

    /// Exchanges participants using explicit shape and grouping semantics.
    fn all_to_all_with_options(
        &self,
        axis_name: &str,
        split_axis: usize,
        concat_axis: usize,
        options: CollectiveOptions,
    ) -> Result<Self, ProgramError>;
}

impl<V> AllToAll for V
where
    V: Value<Type = ArrayIrType> + DimensionSize<V> + ValueProjection<DimensionType>,
    V::DispatchDomain: Context<Type = ArrayIrType> + NamedAxes,
    <V::DispatchDomain as Domain>::Constant: From<DimensionValue>,
    <V::DispatchDomain as Domain>::Operation: From<AllToAllOperation>,
    <V as ValueProjection<DimensionType>>::Projected: DimensionRequirement + Div + Mul,
{
    fn all_to_all_with_options(
        &self,
        axis_name: &str,
        split_axis: usize,
        concat_axis: usize,
        options: CollectiveOptions,
    ) -> Result<Self, ProgramError> {
        let context = self.dispatch_domain();
        let axis_size = resolve_named_axis_size(&context, axis_name)?;
        let effective_axis_size = options.effective_axis_size(ALL_TO_ALL_OPERATION_NAME, axis_size)?;
        let operation =
            AllToAllOperation::new(axis_name.to_string(), axis_size, split_axis, concat_axis, options.clone());
        let mut output_extents = collective_input_extents(&context, self)?;
        let rank = output_extents.len();
        if split_axis >= rank || concat_axis >= rank {
            return Err(TypeError::invalid(format!(
                "`all_to_all` split axis {split_axis} or concat axis {concat_axis} is out of bounds for rank {rank}",
            ))
            .into());
        }
        match options.mode {
            CollectiveMode::Untiled => {
                require_collective_axis_extent(&context, &output_extents[split_axis], effective_axis_size)?;
                output_extents.remove(split_axis);
                output_extents.insert(concat_axis, collective_extent_constant(&context, effective_axis_size)?);
            }
            CollectiveMode::Tiled if split_axis == concat_axis => {
                require_collective_axis_divisible(&context, &output_extents[split_axis], effective_axis_size)?;
            }
            CollectiveMode::Tiled => {
                let split_extent =
                    divided_collective_extent(&context, &output_extents[split_axis], effective_axis_size)?;
                let concat_extent =
                    multiplied_collective_extent(&context, &output_extents[concat_axis], effective_axis_size)?;
                output_extents[split_axis] = split_extent;
                output_extents[concat_axis] = concat_extent;
            }
        };
        let inputs = std::iter::once(self.clone()).chain(output_extents).collect::<Vec<_>>();
        Ok(context.bind(operation, Vec::new(), inputs.as_slice())?.remove(0))
    }
}

impl<V> AllToAll for ProjectedValue<ArrayType, V>
where
    V: AllToAll + ValueProjection<ArrayType, Projected = ProjectedValue<ArrayType, V>>,
{
    fn all_to_all_with_options(
        &self,
        axis_name: &str,
        split_axis: usize,
        concat_axis: usize,
        options: CollectiveOptions,
    ) -> Result<Self, ProgramError> {
        self.value()
            .all_to_all_with_options(axis_name, split_axis, concat_axis, options)?
            .into_projected()
            .map_err(Into::into)
    }
}

/// Convenience untiled all-to-all that exchanges one ranked array axis with a named axis.
pub trait ParallelSwapAxes: AllToAll {
    /// Swaps `axis` with `axis_name` over the full named axis.
    #[inline]
    fn parallel_swap_axes(&self, axis_name: &str, axis: usize) -> Result<Self, ProgramError> {
        self.all_to_all(axis_name, axis, axis)
    }

    /// Swaps `axis` with `axis_name` within the provided ordered participant groups.
    #[inline]
    fn parallel_swap_axes_with_axis_index_groups(
        &self,
        axis_name: &str,
        axis: usize,
        axis_index_groups: Vec<Vec<usize>>,
    ) -> Result<Self, ProgramError> {
        self.all_to_all_with_options(
            axis_name,
            axis,
            axis,
            CollectiveOptions::default().with_axis_index_groups(axis_index_groups),
        )
    }
}

impl<V: AllToAll> ParallelSwapAxes for V {}

// Mixed array IR JVP for all-to-all. Explicit output extents are retained as ordinary residual values, and the
// transposed linear region swaps the split and concatenation axes.
impl<C> MemberDifferentiableOperation<C> for AllToAllOperation
where
    C: Context<Type = ArrayIrType>,
    C::Operation: From<AllToAllOperation>
        + From<DimensionSizeOperation>
        + From<LinearCallOperation<ArrayIrType>>
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
            AllToAllOperation::new(
                self.axis_name().to_string(),
                self.axis_size(),
                self.concat_axis(),
                self.split_axis(),
                self.options().clone(),
            ),
            context,
            inputs,
        )
    }
}

/// Returns the physical split/concat axes and mapped result axis for a forwarded all-to-all.
fn forwarded_all_to_all_axes(
    mode: CollectiveMode,
    split_axis: usize,
    concat_axis: usize,
    batch_axis: usize,
) -> (usize, usize, usize) {
    let physical_split_axis = split_axis + usize::from(split_axis >= batch_axis);
    let (physical_concat_axis, output_batch_axis) = match mode {
        CollectiveMode::Tiled => (concat_axis + usize::from(concat_axis >= batch_axis), batch_axis),
        CollectiveMode::Untiled => {
            let output_batch_axis = batch_axis - usize::from(split_axis < batch_axis);
            if concat_axis <= output_batch_axis {
                (concat_axis, output_batch_axis + 1)
            } else {
                (concat_axis + 1, output_batch_axis)
            }
        }
    };
    (physical_split_axis, physical_concat_axis, output_batch_axis)
}

/// Applies the matching-axis all-to-all batching semantics over the policy-selected extent representation.
fn batch_all_to_all_matching_axis<C, P>(
    operation: &AllToAllOperation,
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
            message: "`all_to_all` axis index groups are not supported when a batch transform binds the collective \
                      axis"
                .to_string(),
        });
    }
    if operation.split_axis >= logical_input_rank || operation.concat_axis >= logical_input_rank {
        return Err(BatchingError::UnsupportedOperation {
            message: format!(
                "`all_to_all` split axis {} or concat axis {} is out of bounds for rank {logical_input_rank}",
                operation.split_axis, operation.concat_axis,
            ),
        });
    }

    let axis_extent =
        P::collective_axis_extent(context, ALL_TO_ALL_OPERATION_NAME, &operation.axis_name, operation.axis_size)?;

    let (input_extents, chunk_extent) = match operation.options.mode {
        CollectiveMode::Untiled => {
            let mut input_extents = output_extents.clone();
            input_extents.remove(operation.concat_axis);
            input_extents.insert(operation.split_axis, axis_extent.clone());
            (input_extents, P::collective_extent_constant(context, 1)?)
        }
        CollectiveMode::Tiled if operation.split_axis == operation.concat_axis => {
            P::require_divisible_collective_extents(&output_extents[operation.split_axis], &axis_extent)?;
            (output_extents.clone(), output_extents[operation.split_axis].div(&axis_extent)?)
        }
        CollectiveMode::Tiled => {
            P::require_divisible_collective_extents(&output_extents[operation.concat_axis], &axis_extent)?;
            let mut input_extents = output_extents.clone();
            input_extents[operation.split_axis] = output_extents[operation.split_axis].mul(&axis_extent)?;
            input_extents[operation.concat_axis] = output_extents[operation.concat_axis].div(&axis_extent)?;
            (input_extents, output_extents[operation.split_axis].clone())
        }
    };
    let input = P::match_collective_axis(context, input, input_extents.as_slice())?;
    let mut split_extents = Vec::with_capacity(input_extents.len() + 2);
    split_extents.push(axis_extent.clone());
    split_extents.extend(input_extents.iter().cloned());
    split_extents[operation.split_axis + 1] = axis_extent.clone();
    split_extents.insert(operation.split_axis + 2, chunk_extent);
    let split = P::reshape_collective(context, input.into_value(), split_extents.as_slice(), None)?;
    let exchanged = split.swap_axes(0, operation.split_axis + 1)?;
    let received = match operation.options.mode {
        CollectiveMode::Untiled => {
            let mut squeezed_extents = Vec::with_capacity(input_extents.len() + 1);
            squeezed_extents.push(axis_extent.clone());
            squeezed_extents.extend(input_extents);
            P::reshape_collective(context, exchanged, squeezed_extents.as_slice(), None)?
                .move_axis(operation.split_axis + 1, operation.concat_axis + 1)?
        }
        CollectiveMode::Tiled => exchanged.move_axis(operation.split_axis + 1, operation.concat_axis + 1)?,
    };
    let mut physical_output_extents = Vec::with_capacity(output_extents.len() + 1);
    physical_output_extents.push(axis_extent);
    physical_output_extents.extend(output_extents);
    let physical_output_sharding = output_sharding
        .map(|sharding| lift_output_sharding_for_leading_batch_axis(&sharding, context.axis_sharding().clone()))
        .transpose()?;
    let output =
        P::reshape_collective(context, received, physical_output_extents.as_slice(), physical_output_sharding)?;
    ArrayBatch::new(output, BatchAxis::from_position(0))
}

// Batching rule for [`AllToAllOperation`]. A matching `batch` level consumes the mapped batch axis with a
// reshape/transpose block exchange: the per-item `split_axis` is split into `(b, d_p / b)` chunks, the chunk axis
// is swapped with the leading batch axis (so the batch axis indexes the *receiving* item), and the sender axis is
// then merged item-major into the per-item `concat_axis` — batch item `i` receives every item's chunk `i`,
// concatenated along `concat_axis`. A non-matching level forwards the collective untouched to the parent context
// via [`forward_collective_to_parent`].
impl<C, P: CollectiveBatchingPolicy<C>> BatchableOperation<C, ArrayBatching<P>> for AllToAllOperation
where
    C: Context<Type = ArrayType>,
    C::Operation: From<AllToAllOperation>,
    <C as Domain>::Value: Transpose,
{
    fn batch<D: BatchingDriver<C, ArrayBatching<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatching<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<<C as Domain>::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatching<P>>, BatchingError> {
        if let Some(ragged_axis) = inputs.iter().find_map(|input| input.ragged_axes().first()) {
            return Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "`all_to_all` cannot route bounded ragged dimension `{}` without explicit per-destination \
                     offsets and sizes; use `ragged_all_to_all`",
                    ragged_axis.dimension(),
                ),
            });
        }
        if context.axis_name() != Some(self.axis_name.as_str()) {
            let [input] = inputs else {
                return Err(ProgramError::InvalidInputCount { expected: 1, actual: inputs.len() }.into());
            };
            let Some(batch_axis) = input.batch_axis_position() else {
                return Ok(forward_collective_to_parent(context, C::Operation::from(self.clone()), inputs)?.into());
            };
            let (split_axis, concat_axis, output_batch_axis) =
                forwarded_all_to_all_axes(self.options.mode, self.split_axis, self.concat_axis, batch_axis);
            let operation =
                Self::new(self.axis_name.clone(), self.axis_size, split_axis, concat_axis, self.options.clone());
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
        Ok(vec![batch_all_to_all_matching_axis::<C, P>(
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

// Batching rule for explicit-extent [`AllToAllOperation`]. Dimension SSA supplies its temporary split and merge
// shapes directly, while matching-axis array mechanics reuse the homogeneous collective kernel.
impl<C> MemberBatchableOperation<C, ArrayIrBatching> for AllToAllOperation
where
    C: Context<
            Type = ArrayIrType,
            Operation: From<AllToAllOperation>
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
        if let Some(ragged_axis) = array.ragged_axes().first() {
            return Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "`all_to_all` cannot route bounded ragged dimension `{}` without explicit per-destination \
                     offsets and sizes; use `ragged_all_to_all`",
                    ragged_axis.dimension(),
                ),
            });
        }
        validate_explicit_collective_output_extents(output_extents)?;
        let logical_input_types = inputs.iter().map(|input| input.unbatched_type().clone()).collect::<Vec<_>>();
        let mut logical_output_types = infer_explicit_all_to_all_output_types(self, logical_input_types.as_slice())?;
        let logical_output_type = <&ArrayType>::try_from(&logical_output_types.remove(0))?.clone();

        if context.axis_name() != Some(self.axis_name()) {
            if array.batch_axis().is_replicated() {
                return Ok(forward_explicit_collective(self.clone(), context, array, output_extents, None)?.into());
            }
            let input_batch_axis = array.batch_axis_position().unwrap();
            let (physical_split_axis, physical_concat_axis, output_batch_axis) = forwarded_all_to_all_axes(
                self.options().mode(),
                self.split_axis(),
                self.concat_axis(),
                input_batch_axis,
            );
            let operation = Self::new(
                self.axis_name().to_string(),
                self.axis_size(),
                physical_split_axis,
                physical_concat_axis,
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
        let output = batch_all_to_all_matching_axis::<_, DynamicArrayBatchingPolicy>(
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

// Transpose rule for [`AllToAllOperation`]: the chunk exchange is its own adjoint with the split and concatenation
// axes swapped.
impl<V, O> TransposableOperation<V, O> for AllToAllOperation
where
    V: Value<Type = ArrayType>,
    O: Operation<Type = ArrayType> + From<AllToAllOperation>,
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
            AllToAllOperation::new(
                self.axis_name.clone(),
                self.axis_size,
                self.concat_axis,
                self.split_axis,
                self.options.clone(),
            ),
        )
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrValue, ArrayOperation, DataType, DimensionBounds, DimensionVariable, RaggedAxis,
    };
    use crate::axes::NamedAxis;
    use crate::batching::{BatchAxis, BatchAxisSpecification, BatchingContext, batch};
    use crate::contexts::EagerContext;

    use super::*;

    #[test]
    fn test_parallel_swap_axes_composes_untiled_all_to_all_in_the_composite_domain() {
        type TestContext = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let (_, program) = TestContext::trace_with_named_axes(
            |input| input.parallel_swap_axes("x", 0),
            ArrayIrType::Array(input_type),
            vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 2 })],
        )
        .unwrap();

        let all_to_all = program.instructions().last().unwrap();
        let ArrayIrOperation::AllToAll(operation) = all_to_all.operation() else {
            panic!("parallel_swap_axes must compose the canonical all-to-all operation");
        };
        assert_eq!(operation.split_axis(), 0);
        assert_eq!(operation.concat_axis(), 0);
        assert_eq!(operation.options(), &CollectiveOptions::default());
        assert_eq!(all_to_all.inputs().len(), 3);
        assert_eq!(all_to_all.inputs()[0], program.input_ids()[0]);
    }

    #[test]
    fn test_all_to_all_type_inference() {
        use crate::macros::check_operation_type_inference;

        let matrix = |rows: usize, columns: usize| {
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(rows), Dimension::Static(columns)]))
        };
        check_operation_type_inference!(
            operation = AllToAllOperation::new("x".to_string(), 4, 0, 1, CollectiveOptions::tiled()),
            cases = [
                {
                    input_types = [matrix(8, 3)],
                    output_types = [matrix(2, 12)],
                },
                {
                    input_types = [matrix(6, 3)],
                    error = "`all_to_all` split axis 0 size 6 is not divisible by group size 4",
                },
            ],
        );
    }

    #[test]
    fn test_all_to_all_forwarded_axes_account_for_the_mapped_axis() {
        assert_eq!(forwarded_all_to_all_axes(CollectiveMode::Tiled, 0, 1, 1), (0, 2, 1));
        assert_eq!(forwarded_all_to_all_axes(CollectiveMode::Tiled, 1, 0, 0), (2, 1, 0));
        assert_eq!(forwarded_all_to_all_axes(CollectiveMode::Untiled, 0, 0, 2), (0, 0, 2));
        assert_eq!(forwarded_all_to_all_axes(CollectiveMode::Untiled, 1, 1, 0), (2, 2, 0));
    }

    #[test]
    fn test_all_to_all_over_batched_axis_exchanges_chunks() {
        use crate::batching::BatchingTracer;

        // Block exchange with `split_axis == concat_axis == 0`: each item splits its vector into two chunks and
        // receives its own chunk index from every item, concatenated item-major. With items `[1, 2, 3, 4]` and
        // `[5, 6, 7, 8]`, item 0 receives `[1, 2, 5, 6]` and item 1 receives `[3, 4, 7, 8]`, matching the verified
        // cross-device `shard_map` execution semantics of StableHLO's `all_to_all`.
        let x = Array::matrix(2, 4, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let output: ArrayIrValue<Array> = batch(
            |item: BatchingTracer<EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>, ArrayIrBatching>| {
                item.all_to_all_tiled("x", 0, 0)
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
                Shape::new(vec![Dimension::Static(2), Dimension::Static(4)]),
            )),
        );
        let ArrayIrValue::Array(output) = output else {
            panic!("`all_to_all` must preserve the array member kind");
        };
        assert_eq!(output.to_f64s(), vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]);
    }

    #[test]
    fn test_all_to_all_rejects_ragged_operands_without_explicit_routing() {
        let variable = DimensionVariable::new("length", DimensionBounds::new(0, Some(4)).unwrap());
        let input = ArrayBatch::new(Array::matrix(2, 4, vec![1.0_f32; 8]), BatchAxis::new(0))
            .unwrap()
            .with_ragged_axes(vec![RaggedAxis::new(1, Array::vector(vec![2_i32, 4]), variable.clone(), vec![0])])
            .unwrap();
        let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 2)
            .with_axis_name("x".to_string());

        assert_eq!(
            AllToAllOperation::new("x".to_string(), 2, 0, 0, CollectiveOptions::tiled()).batch(
                &context,
                &crate::EmptyRegionDriver,
                &[input],
            ),
            Err(BatchingError::UnsupportedOperation {
                message: "`all_to_all` cannot route bounded ragged dimension `length` without explicit \
                          per-destination offsets and sizes; use `ragged_all_to_all`"
                    .to_string(),
            }),
        );

        let extents = ArrayIrValue::Array(Array::vector(vec![2_i32, 4]));
        let input = ArrayIrBatch::new(ArrayIrValue::Array(Array::matrix(2, 4, vec![1.0_f32; 8])), BatchAxis::new(0))
            .unwrap()
            .with_ragged_axes(vec![RaggedAxis::new(1, extents.clone(), variable.clone(), vec![0])])
            .unwrap();
        let context = BatchingContext::new(
            EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new(),
            ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()),
        )
        .with_axis_name("x".to_string());
        let output_extent =
            ArrayIrBatch::mapped_dimension(extents, BatchAxis::new(0), DimensionType::new(variable)).unwrap();
        assert_eq!(
            AllToAllOperation::new("x".to_string(), 2, 0, 0, CollectiveOptions::tiled()).batch_in_parent(
                &context,
                &crate::EmptyRegionDriver,
                &[input, output_extent],
            ),
            Err(BatchingError::UnsupportedOperation {
                message: "`all_to_all` cannot route bounded ragged dimension `length` without explicit \
                          per-destination offsets and sizes; use `ragged_all_to_all`"
                    .to_string(),
            }),
        );
    }

    #[test]
    fn test_all_to_all_over_batched_axis_with_distinct_axes_exchanges_chunks() {
        use crate::batching::BatchingTracer;

        // Distinct split and concatenation axes over per-item `[2, 2]` matrices: each item splits its rows across
        // the items and receives its own row index from every item, concatenated item-major along the columns. With
        // item 0 = `[[1, 2], [3, 4]]` and item 1 = `[[5, 6], [7, 8]]`, item 0 receives `[[1, 2, 5, 6]]` and item 1
        // receives `[[3, 4, 7, 8]]` (per-item shape `[1, 4]`).
        let x = Array::from_f64s(
            ArrayType::new(
                DataType::F64,
                Shape::new(vec![Dimension::Static(2), Dimension::Static(2), Dimension::Static(2)]),
            ),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        );
        let output: ArrayIrValue<Array> = batch(
            |item: BatchingTracer<EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>, ArrayIrBatching>| {
                item.all_to_all_tiled("x", 0, 1)
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
                Shape::new(vec![Dimension::Static(2), Dimension::Static(1), Dimension::Static(4)]),
            )),
        );
        let ArrayIrValue::Array(output) = output else {
            panic!("`all_to_all` must preserve the array member kind");
        };
        assert_eq!(output.to_f64s(), vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]);
    }
}
