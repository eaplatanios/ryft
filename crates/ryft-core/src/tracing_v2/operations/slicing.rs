use crate::batching::BatchingError;
use crate::contexts::StagingContext;
use crate::differentiation::{Cotangent, TransposableOperation};
use crate::macros::check_count;
use crate::operations::constants::{MaybeZeroOperation, ZeroLike, ZeroOperation};
use crate::operations::manipulation::{
    Broadcast, DynamicSlice, DynamicSliceOperation, DynamicUpdateSlice, DynamicUpdateSliceOperation,
    LinearDynamicSliceOperation, LinearDynamicUpdateSliceOperation, PadOperation, Reshape, Slice, SliceOperation,
    Transpose, UpdateSlice, UpdateSliceOperation,
};
use crate::operations::{InterpretableOperation, Operation};
use crate::programs::{ProgramError, Value};
use crate::tracing::AbstractTracingContext;
use crate::tracing_v2::batching::{
    ArrayBatch, BatchableOperation, align_batch_axis, apply_with_axes, batch_input_metadata,
};
use crate::tracing_v2::differentiation::{JvpTracer, LinearOperationOf, TangentContext};
use crate::tracing_v2::{CapturedFactor, DifferentiableOperation, DifferentiationContext};
use crate::types::{ArrayType, Shape, Size, TypeError, Typed};

use super::control_flow::stage_cotangent;

/// Transpose (vector-Jacobian product) for a [`SliceOperation`].
///
/// The forward map extracts a (possibly strided) block, so its pullback scatters the output cotangent back into the
/// positions the forward map read, with the strategy split on the strides:
///
///   - **Unit strides** read a contiguous block, so the pullback writes the cotangent into a zero array of the input
///     type at the same static offsets: `cotangent ↦ update_slice(zeros(input_type), cotangent, start_indices)`.
///   - **Non-unit strides** read every `strides[d]`-th element, so the pullback pads the cotangent with a zero
///     scalar at exactly the inverse geometry: `edge_padding_low[d] = start_indices[d]`,
///     `interior_padding[d] = strides[d] - 1`, and `edge_padding_high[d]` covers the rest of the input extent
///     (everything after the last element the forward slice covered). For example, slicing `[0..6)` with `start = 1`
///     and `stride = 2` reads positions `1`, `3`, and `5`, and the pullback pads the cotangent of length `3` with
///     `low = 1`, `interior = 1`, and `high = 0`, scattering its elements back to positions `1`, `3`, and `5` of a
///     zero-filled length-`6` array.
///
/// Symbolic-zero cotangents propagate unchanged.
impl<V: Value<ArrayType>, O> TransposableOperation<ArrayType, V, O> for SliceOperation
where
    O: Operation<ArrayType> + From<UpdateSliceOperation> + From<PadOperation> + From<ZeroOperation<ArrayType>>,
{
    fn transpose<'transpose>(
        &self,
        context: &mut AbstractTracingContext<'transpose, ArrayType, V, O>,
        input_types: &[&ArrayType],
        output_cotangents: &[Cotangent<'transpose, ArrayType, V, O>],
    ) -> Result<Vec<Cotangent<'transpose, ArrayType, V, O>>, ProgramError> {
        check_count!("input", input_types, 1, ProgramError);
        check_count!("output", output_cotangents, 1, ProgramError);
        match &output_cotangents[0] {
            Cotangent::Zero => Ok(vec![Cotangent::Zero]),
            Cotangent::Staged(cotangent) if self.strides().iter().all(|stride| *stride == 1) => {
                let zeros = stage_cotangent(context, &Cotangent::Zero, input_types[0]);
                let outputs = context.stage_operation(
                    UpdateSliceOperation::new(self.start_indices().to_vec()),
                    &[zeros, cotangent.clone()],
                )?;
                check_count!("output", outputs, 1, ProgramError);
                Ok(vec![Cotangent::Staged(outputs.into_iter().next().unwrap())])
            }
            Cotangent::Staged(cotangent) => {
                let input_type = input_types[0];
                let mut edge_padding_low = Vec::with_capacity(input_type.rank());
                let mut edge_padding_high = Vec::with_capacity(input_type.rank());
                let mut interior_padding = Vec::with_capacity(input_type.rank());
                for (axis, ((&start, &limit), &stride)) in
                    self.start_indices().iter().zip(self.limit_indices()).zip(self.strides()).enumerate()
                {
                    let dimension = input_type.dimension(axis as isize);
                    let Some(input_size) = dimension.value() else {
                        return Err(TypeError {
                            message: format!(
                                "slice transpose requires a static input shape but axis {axis} has size {dimension}",
                            ),
                        }
                        .into());
                    };
                    let output_size = (limit - start).div_ceil(stride);
                    // The forward slice covered positions `start + i * stride` for `i < output_size`; everything
                    // after the last covered position becomes high edge padding. An empty slice covered nothing, so
                    // the pullback is pure edge padding around zero interior elements.
                    let high = match output_size {
                        0 => input_size - start,
                        size => input_size - (start + (size - 1) * stride) - 1,
                    };
                    edge_padding_low.push(start);
                    edge_padding_high.push(high);
                    interior_padding.push(stride - 1);
                }
                let zero = stage_cotangent(context, &Cotangent::Zero, &ArrayType::scalar(input_type.data_type()));
                let outputs = context.stage_operation(
                    PadOperation::new(edge_padding_low, edge_padding_high, interior_padding)?,
                    &[cotangent.clone(), zero],
                )?;
                check_count!("output", outputs, 1, ProgramError);
                Ok(vec![Cotangent::Staged(outputs.into_iter().next().unwrap())])
            }
        }
    }
}

/// Transpose (vector-Jacobian product) for an [`UpdateSliceOperation`].
///
/// The forward map overwrites a block of the input with the update, so its pullback splits the output cotangent into
/// two contributions: the input cotangent is the cotangent with the update window zeroed
/// (`update_slice(cotangent, zeros(update_type), start_indices)`) and the update cotangent is the static slice of
/// the cotangent at the update window (`slice(cotangent, start_indices, start_indices + update_shape)`).
/// Symbolic-zero cotangents propagate unchanged.
impl<V: Value<ArrayType>, O> TransposableOperation<ArrayType, V, O> for UpdateSliceOperation
where
    O: Operation<ArrayType> + From<SliceOperation> + From<UpdateSliceOperation> + From<ZeroOperation<ArrayType>>,
{
    fn transpose<'transpose>(
        &self,
        context: &mut AbstractTracingContext<'transpose, ArrayType, V, O>,
        input_types: &[&ArrayType],
        output_cotangents: &[Cotangent<'transpose, ArrayType, V, O>],
    ) -> Result<Vec<Cotangent<'transpose, ArrayType, V, O>>, ProgramError> {
        check_count!("input", input_types, 2, ProgramError);
        check_count!("output", output_cotangents, 1, ProgramError);
        match &output_cotangents[0] {
            Cotangent::Zero => Ok(vec![Cotangent::Zero, Cotangent::Zero]),
            Cotangent::Staged(cotangent) => {
                let update_type = input_types[1];
                let update_sizes = static_update_sizes(UPDATE_SLICE_TRANSPOSE_CONTEXT, update_type)?;
                let zeros = stage_cotangent(context, &Cotangent::Zero, update_type);
                let input_cotangents = context.stage_operation(
                    UpdateSliceOperation::new(self.start_indices().to_vec()),
                    &[cotangent.clone(), zeros],
                )?;
                check_count!("output", input_cotangents, 1, ProgramError);
                let limit_indices: Vec<usize> =
                    self.start_indices().iter().zip(update_sizes.iter()).map(|(start, size)| start + size).collect();
                let update_cotangents = context.stage_operation(
                    SliceOperation::new(self.start_indices().to_vec(), limit_indices)
                        .with_strides(vec![1; self.start_indices().len()])?,
                    std::slice::from_ref(cotangent),
                )?;
                check_count!("output", update_cotangents, 1, ProgramError);
                Ok(vec![
                    Cotangent::Staged(input_cotangents.into_iter().next().unwrap()),
                    Cotangent::Staged(update_cotangents.into_iter().next().unwrap()),
                ])
            }
        }
    }
}

/// Operation-name prefix used by [`static_update_sizes`] errors raised from the update-slice transpose rule.
const UPDATE_SLICE_TRANSPOSE_CONTEXT: &str = "update_slice transpose";

/// Extracts the static dimensions of an update operand type, reporting a precise error when any dimension is
/// dynamic. The `context` parameter selects the reported rule name because this helper serves both the static and
/// the captured-index update-slice transpose rules.
pub(crate) fn static_update_sizes(context: &str, update_type: &ArrayType) -> Result<Vec<usize>, ProgramError> {
    update_type
        .shape()
        .dimensions()
        .iter()
        .enumerate()
        .map(|(axis, size)| {
            size.value().ok_or_else(|| {
                TypeError {
                    message: format!("{context} requires a static update shape but axis {axis} has size {size}"),
                }
                .into()
            })
        })
        .collect()
}

/// JVP rule for [`SliceOperation`]: slicing is a linear map, so the primal output is the slice of the input primal
/// and the tangent is the same slice of the input tangent.
impl<D> DifferentiableOperation<D> for SliceOperation
where
    D: DifferentiationContext<Type = ArrayType>,
    D::Value: Slice,
    D::Tangent: Slice,
    LinearOperationOf<D>: From<SliceOperation>,
{
    fn jvp<'jvp>(
        &self,
        _context: &mut TangentContext<'jvp, D>,
        inputs: &[JvpTracer<'jvp, D>],
    ) -> Result<Vec<JvpTracer<'jvp, D>>, ProgramError>
    where
        D: 'jvp,
    {
        check_count!("input", inputs, 1, ProgramError);
        let primal = inputs[0].primal().clone().slice(self.start_indices(), self.limit_indices(), self.strides())?;
        let tangent = inputs[0].tangent().clone().slice(self.start_indices(), self.limit_indices(), self.strides())?;
        Ok(vec![JvpTracer::new(primal, tangent)])
    }
}

/// JVP rule for [`UpdateSliceOperation`]: the operation is jointly linear in its input and update operands, so the
/// tangent is the update-slice of the two operand tangents at the same static offsets. When both operand tangents
/// are canonical staged zeros, the output tangent is a canonical staged zero of the output type and no linear
/// operation is staged.
impl<D> DifferentiableOperation<D> for UpdateSliceOperation
where
    D: DifferentiationContext<Type = ArrayType>,
    D::Value: UpdateSlice,
    LinearOperationOf<D>: From<UpdateSliceOperation> + From<ZeroOperation<ArrayType>>,
    LinearOperationOf<D>: MaybeZeroOperation<ArrayType>,
{
    fn jvp<'jvp>(
        &self,
        context: &mut TangentContext<'jvp, D>,
        inputs: &[JvpTracer<'jvp, D>],
    ) -> Result<Vec<JvpTracer<'jvp, D>>, ProgramError>
    where
        D: 'jvp,
    {
        check_count!("input", inputs, 2, ProgramError);
        let input = &inputs[0];
        let update = &inputs[1];
        let primal = input.primal().update_slice(update.primal(), self.start_indices())?;
        if context.is_zero(input.tangent())? && context.is_zero(update.tangent())? {
            let tangent_type = primal.r#type().into_owned();
            let mut tangent_outputs = context.stage_nullary_operation(ZeroOperation::new(tangent_type))?;
            check_count!("output", tangent_outputs, 1, ProgramError);
            return Ok(vec![JvpTracer::new(primal, tangent_outputs.remove(0))]);
        }
        let mut outputs = context.stage_operation(
            UpdateSliceOperation::new(self.start_indices().to_vec()),
            &[input.tangent(), update.tangent()],
        )?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(vec![JvpTracer::from_value(primal, outputs.remove(0))])
    }
}

/// JVP rule for [`DynamicSliceOperation`]: the operation is linear in its sliced operand, while the scalar integer
/// start indices have no tangent space (their tangents are structural zeros and are ignored). The primal output is
/// the dynamic slice of the input primal at the index primals, and the tangent is a captured-index dynamic slice of
/// the input tangent whose start indices are the index primals captured as residual factors (the
/// [`LinearDynamicSliceOperation`] form). When the input tangent is a canonical staged zero, the output tangent is a
/// canonical staged zero of the output type and no linear operation is staged.
impl<D> DifferentiableOperation<D> for DynamicSliceOperation
where
    D: DifferentiationContext<Type = ArrayType>,
    D::Value: DynamicSlice,
    LinearOperationOf<D>:
        From<LinearDynamicSliceOperation<CapturedFactor<ArrayType, D::Value>>> + From<ZeroOperation<ArrayType>>,
    LinearOperationOf<D>: MaybeZeroOperation<ArrayType>,
{
    fn jvp<'jvp>(
        &self,
        context: &mut TangentContext<'jvp, D>,
        inputs: &[JvpTracer<'jvp, D>],
    ) -> Result<Vec<JvpTracer<'jvp, D>>, ProgramError>
    where
        D: 'jvp,
    {
        let [input, start_indices @ ..] = inputs else {
            return Err(ProgramError::InvalidInputCount { expected: 1 + self.sizes().len(), actual: 0 });
        };
        let index_primals: Vec<D::Value> = start_indices.iter().map(|index| index.primal().clone()).collect();
        let primal = input.primal().dynamic_slice(&index_primals, self.sizes())?;
        if context.is_zero(input.tangent())? {
            let tangent_type = primal.r#type().into_owned();
            let mut tangent_outputs = context.stage_nullary_operation(ZeroOperation::new(tangent_type))?;
            check_count!("output", tangent_outputs, 1, ProgramError);
            return Ok(vec![JvpTracer::new(primal, tangent_outputs.remove(0))]);
        }
        let index_factors: Vec<_> = start_indices.iter().map(|index| index.factor(context)).collect();
        let mut outputs = context.stage_operation(
            LinearDynamicSliceOperation::new(index_factors, self.sizes().to_vec()),
            &[input.tangent()],
        )?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(vec![JvpTracer::from_value(primal, outputs.remove(0))])
    }
}

/// JVP rule for [`DynamicUpdateSliceOperation`]: the operation is jointly linear in its input and update operands,
/// while the scalar integer start indices have no tangent space (their tangents are structural zeros and are
/// ignored). The primal output is the dynamic update-slice of the operand primals at the index primals, and the
/// tangent is a captured-index dynamic update-slice of the operand tangents whose start indices are the index
/// primals captured as residual factors (the [`LinearDynamicUpdateSliceOperation`] form). When both operand tangents
/// are canonical staged zeros, the output tangent is a canonical staged zero of the output type and no linear
/// operation is staged.
impl<D> DifferentiableOperation<D> for DynamicUpdateSliceOperation
where
    D: DifferentiationContext<Type = ArrayType>,
    D::Value: DynamicUpdateSlice,
    LinearOperationOf<D>:
        From<LinearDynamicUpdateSliceOperation<CapturedFactor<ArrayType, D::Value>>> + From<ZeroOperation<ArrayType>>,
    LinearOperationOf<D>: MaybeZeroOperation<ArrayType>,
{
    fn jvp<'jvp>(
        &self,
        context: &mut TangentContext<'jvp, D>,
        inputs: &[JvpTracer<'jvp, D>],
    ) -> Result<Vec<JvpTracer<'jvp, D>>, ProgramError>
    where
        D: 'jvp,
    {
        let [input, update, start_indices @ ..] = inputs else {
            return Err(ProgramError::InvalidInputCount { expected: 2, actual: inputs.len() });
        };
        let index_primals: Vec<D::Value> = start_indices.iter().map(|index| index.primal().clone()).collect();
        let primal = input.primal().dynamic_update_slice(update.primal(), &index_primals)?;
        if context.is_zero(input.tangent())? && context.is_zero(update.tangent())? {
            let tangent_type = primal.r#type().into_owned();
            let mut tangent_outputs = context.stage_nullary_operation(ZeroOperation::new(tangent_type))?;
            check_count!("output", tangent_outputs, 1, ProgramError);
            return Ok(vec![JvpTracer::new(primal, tangent_outputs.remove(0))]);
        }
        let index_factors: Vec<_> = start_indices.iter().map(|index| index.factor(context)).collect();
        let mut outputs = context.stage_operation(
            LinearDynamicUpdateSliceOperation::new(index_factors),
            &[input.tangent(), update.tangent()],
        )?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(vec![JvpTracer::from_value(primal, outputs.remove(0))])
    }
}

/// Returns `batch` with a lane axis of size `axis_size` materialized at `axis`: already-batched values have their
/// lane axis realigned to `axis`, while lane-uniform values are broadcast to gain a replicated lane axis there. Used
/// by the update-slice and pad batching rules, whose operands must agree on one physical lane axis.
pub(crate) fn materialize_lane_axis<V: Value<ArrayType> + Broadcast + Transpose>(
    batch: &ArrayBatch<V>,
    axis: usize,
    axis_size: usize,
) -> Result<ArrayBatch<V>, ProgramError> {
    if batch.batch_axis().is_some() {
        return align_batch_axis(batch, axis);
    }
    let input_type = batch.r#type().into_owned();
    let mut dimensions = input_type.shape().dimensions().to_vec();
    dimensions.insert(axis, Size::Static(axis_size));
    let output_type = ArrayType::new(input_type.data_type(), Shape::new(dimensions));
    let output_axes: Vec<usize> = (0..input_type.rank())
        .map(|input_axis| if input_axis < axis { input_axis } else { input_axis + 1 })
        .collect();
    let value = batch.value().clone().broadcast(output_type.clone(), output_axes.as_slice())?;
    ArrayBatch::new(output_type, value, Some(axis))
}

/// Extracts lane `lane` of a per-lane expansion operand: batched operands (whose lane axis must already sit at the
/// leading physical axis) contribute slice `lane` with the lane axis dropped, while lane-uniform operands are used
/// whole. Batched operand types must be fully static so the lane slice bounds are provable; `operation_name` selects
/// the rule named in the error reported otherwise.
pub(crate) fn expansion_lane<V>(
    operation_name: &'static str,
    input: &ArrayBatch<V>,
    lane: usize,
) -> Result<V, ProgramError>
where
    V: Value<ArrayType> + Slice + Reshape,
{
    if input.batch_axis().is_none() {
        return Ok(input.value().clone());
    }
    let input_type = input.r#type().into_owned();
    let dimensions = input_type
        .shape()
        .dimensions()
        .iter()
        .map(|dimension| {
            dimension.value().ok_or_else(|| {
                TypeError {
                    message: format!(
                        "{operation_name} per-lane expansion requires static batched operand types but got \
                         {input_type}",
                    ),
                }
                .into()
            })
        })
        .collect::<Result<Vec<usize>, ProgramError>>()?;
    let mut start_indices = vec![0; dimensions.len()];
    start_indices[0] = lane;
    let mut limit_indices = dimensions.clone();
    limit_indices[0] = lane + 1;
    let unit_strides = vec![1; dimensions.len()];
    let lane_value =
        input
            .value()
            .clone()
            .slice(start_indices.as_slice(), limit_indices.as_slice(), unit_strides.as_slice())?;
    lane_value.reshape(Shape::new(dimensions[1..].iter().map(|&dimension| Size::Static(dimension)).collect()))
}

/// Stacks per-lane expansion results along a fresh leading lane axis of size `axis_size`: lane `0` seeds the stacked
/// accumulator by broadcast replication and later lanes overwrite their slices via [`UpdateSlice`] at static lane
/// offsets. `interpret_lane` produces the per-lane result; an empty lane axis is rejected with a precise error
/// naming `operation_name` because no lane can seed the accumulator.
pub(crate) fn stack_expansion_lanes<V, InterpretLaneFn>(
    operation_name: &'static str,
    axis_size: usize,
    mut interpret_lane: InterpretLaneFn,
) -> Result<ArrayBatch<V>, ProgramError>
where
    V: Value<ArrayType> + Broadcast + UpdateSlice + Reshape,
    InterpretLaneFn: FnMut(usize) -> Result<V, ProgramError>,
{
    let mut accumulator: Option<V> = None;
    for lane in 0..axis_size {
        let output_lane = interpret_lane(lane)?;
        let output_lane_type = output_lane.r#type().into_owned();
        accumulator = Some(match accumulator {
            None => {
                // Lane `0` seeds the stacked accumulator by replication; later lanes overwrite their slices.
                let mut stacked_dimensions = Vec::with_capacity(output_lane_type.rank() + 1);
                stacked_dimensions.push(Size::Static(axis_size));
                stacked_dimensions.extend(output_lane_type.shape().dimensions().iter().cloned());
                let stacked_type = ArrayType::new(output_lane_type.data_type(), Shape::new(stacked_dimensions));
                let output_axes: Vec<usize> = (1..=output_lane_type.rank()).collect();
                output_lane.broadcast(stacked_type, output_axes.as_slice())?
            }
            Some(accumulator) => {
                let mut expanded_dimensions = Vec::with_capacity(output_lane_type.rank() + 1);
                expanded_dimensions.push(Size::Static(1));
                expanded_dimensions.extend(output_lane_type.shape().dimensions().iter().cloned());
                let expanded = output_lane.reshape(Shape::new(expanded_dimensions))?;
                let mut write_indices = vec![0; output_lane_type.rank() + 1];
                write_indices[0] = lane;
                accumulator.update_slice(&expanded, write_indices.as_slice())?
            }
        });
    }
    let Some(accumulator) = accumulator else {
        return Err(BatchingError::UnsupportedOperation {
            message: format!("{operation_name} does not support per-lane expansion over an empty lane axis"),
        }
        .into());
    };
    let stacked_type = accumulator.r#type().into_owned();
    ArrayBatch::new(stacked_type, accumulator, Some(0))
}

/// Applies a single-output `operation` independently per lane and restacks the results along a fresh leading lane
/// axis: every input is realigned so any mapped lane axis sits at the leading physical axis, lane `lane` of each
/// batched input is extracted via [`expansion_lane`] (lane-uniform inputs are used whole), and the per-lane outputs
/// are stacked via [`stack_expansion_lanes`]. This is the shared fallback for batched operands that cannot ride
/// along structurally — lane-varying dynamic-slice start indices and lane-varying pad padding values — and it stages
/// `O(axis_size)` operations because everything goes through the value capability traits (which also makes it work
/// identically in eager and tracing contexts, since capabilities stage on tracers).
pub(crate) fn batch_by_lane_expansion<V, O>(
    context: &V::InterpretationContext,
    operation_name: &'static str,
    operation: &O,
    inputs: &[ArrayBatch<V>],
    axis_size: usize,
) -> Result<Vec<ArrayBatch<V>>, ProgramError>
where
    V: Value<ArrayType> + Broadcast + Transpose + Slice + UpdateSlice + Reshape,
    O: InterpretableOperation<ArrayType, V>,
{
    if inputs.is_empty() {
        return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 });
    }
    let aligned = inputs.iter().map(|input| align_batch_axis(input, 0)).collect::<Result<Vec<_>, _>>()?;
    let stacked = stack_expansion_lanes(operation_name, axis_size, |lane| {
        let lane_inputs = aligned
            .iter()
            .map(|input| expansion_lane(operation_name, input, lane))
            .collect::<Result<Vec<_>, _>>()?;
        let mut outputs = operation.interpret(context, lane_inputs.as_slice())?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    })?;
    Ok(vec![stacked])
}

/// Batching rule for [`SliceOperation`]: a batched operand keeps its lane axis by slicing it fully, so the lifted
/// operation inserts start index `0`, limit `axis_size`, and stride `1` at the lane axis position.
impl<V: Value<ArrayType>> BatchableOperation<V, V::InterpretationContext> for SliceOperation
where
    SliceOperation: InterpretableOperation<ArrayType, V>,
{
    fn batch(
        &self,
        context: &V::InterpretationContext,
        inputs: &[ArrayBatch<V>],
    ) -> Result<Vec<ArrayBatch<V>>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        let (_, input_axes, axis_size) = batch_input_metadata(inputs)?;
        match input_axes[0] {
            None => apply_with_axes(context, self, inputs, &[None]),
            Some(batch_axis) => {
                let mut start_indices = self.start_indices().to_vec();
                start_indices.insert(batch_axis, 0);
                let mut limit_indices = self.limit_indices().to_vec();
                limit_indices.insert(batch_axis, axis_size);
                let mut strides = self.strides().to_vec();
                strides.insert(batch_axis, 1);
                let lifted = SliceOperation::new(start_indices, limit_indices).with_strides(strides)?;
                apply_with_axes(context, &lifted, inputs, &[Some(batch_axis)])
            }
        }
    }
}

/// Batching rule for [`UpdateSliceOperation`]: the input and update operands are aligned on one physical lane axis
/// (lane-uniform operands are broadcast to gain it), and the lifted operation inserts start index `0` at that axis
/// so each lane updates its own block.
impl<V: Value<ArrayType> + Broadcast + Transpose> BatchableOperation<V, V::InterpretationContext>
    for UpdateSliceOperation
where
    UpdateSliceOperation: InterpretableOperation<ArrayType, V>,
{
    fn batch(
        &self,
        context: &V::InterpretationContext,
        inputs: &[ArrayBatch<V>],
    ) -> Result<Vec<ArrayBatch<V>>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        let (_, input_axes, axis_size) = batch_input_metadata(inputs)?;
        let Some(batch_axis) = input_axes.iter().copied().flatten().next() else {
            return apply_with_axes(context, self, inputs, &[None]);
        };
        let input = materialize_lane_axis(&inputs[0], batch_axis, axis_size)?;
        let update = materialize_lane_axis(&inputs[1], batch_axis, axis_size)?;
        let mut start_indices = self.start_indices().to_vec();
        start_indices.insert(batch_axis, 0);
        apply_with_axes(context, &UpdateSliceOperation::new(start_indices), &[input, update], &[Some(batch_axis)])
    }
}

/// Batching rule for [`DynamicSliceOperation`].
///
/// Lane-uniform start indices keep the structural fast path: a batched operand keeps its lane axis by slicing it
/// fully, so the lifted operation inserts size `axis_size` at the lane axis position and a zero start index for it,
/// derived from an existing index operand via [`ZeroLike`] so the inserted index carries the same scalar integer
/// type. Rank-0 operands have no index operands to donate a zero index, but a rank-0 dynamic slice is the identity
/// map, so the batched operand passes through unchanged.
///
/// Lane-varying (batched) start indices cannot ride along structurally — every lane needs its own slice origin while
/// the lifted operation reads one origin for all lanes — so the rule falls back to per-lane expansion via
/// [`batch_by_lane_expansion`]: each lane's operand (when batched; a lane-uniform operand is used whole) and start
/// indices are extracted, sliced dynamically per lane, and restacked along a fresh leading lane axis (the result's
/// lane axis is `0` even when the operand carried its lane axis elsewhere). The expansion stages `O(batch_size)`
/// operations — a gather-based rule is an explicit non-goal — and behaves identically in eager and tracing contexts
/// because it only goes through the value capability traits.
impl<V> BatchableOperation<V, V::InterpretationContext> for DynamicSliceOperation
where
    V: Value<ArrayType> + ZeroLike + Broadcast + Transpose + Slice + UpdateSlice + Reshape,
    DynamicSliceOperation: InterpretableOperation<ArrayType, V>,
{
    fn batch(
        &self,
        context: &V::InterpretationContext,
        inputs: &[ArrayBatch<V>],
    ) -> Result<Vec<ArrayBatch<V>>, ProgramError> {
        if inputs.is_empty() {
            return Err(ProgramError::InvalidInputCount { expected: 1 + self.sizes().len(), actual: 0 });
        }
        let (_, input_axes, axis_size) = batch_input_metadata(inputs)?;
        if input_axes[1..].iter().any(Option::is_some) {
            return batch_by_lane_expansion(
                context,
                crate::operations::manipulation::DYNAMIC_SLICE_OPERATION_NAME,
                self,
                inputs,
                axis_size,
            );
        }
        let Some(batch_axis) = input_axes[0] else {
            return apply_with_axes(context, self, inputs, &[None]);
        };
        if self.sizes().is_empty() {
            return Ok(vec![inputs[0].clone()]);
        }
        let mut sizes = self.sizes().to_vec();
        sizes.insert(batch_axis, axis_size);
        let zero_index = ArrayBatch::unbatched(inputs[1].value().clone().zero_like());
        let mut lifted_inputs = inputs.to_vec();
        lifted_inputs.insert(1 + batch_axis, zero_index);
        apply_with_axes(context, &DynamicSliceOperation::new(sizes), lifted_inputs.as_slice(), &[Some(batch_axis)])
    }
}

/// Batching rule for [`DynamicUpdateSliceOperation`].
///
/// Lane-uniform start indices keep the structural fast path: the input and update operands are aligned on one
/// physical lane axis (lane-uniform operands are broadcast to gain it), and the lifted operation inserts a zero
/// start index for that axis, derived from an existing index operand via [`ZeroLike`] so the inserted index carries
/// the same scalar integer type. Rank-0 operands have no index operands to donate a zero index, but a rank-0 dynamic
/// update-slice replaces the operand with the update entirely, so the update operand passes through unchanged.
///
/// Lane-varying (batched) start indices cannot ride along structurally — every lane needs its own update origin
/// while the lifted operation reads one origin for all lanes — so the rule falls back to per-lane expansion via
/// [`batch_by_lane_expansion`]: each lane's input, update, and start indices are extracted (lane-uniform operands
/// are used whole), updated per lane, and restacked along a fresh leading lane axis (the result's lane axis is `0`
/// even when the operands carried their lane axes elsewhere). The expansion stages `O(batch_size)` operations — a
/// scatter-based rule is an explicit non-goal — and behaves identically in eager and tracing contexts because it
/// only goes through the value capability traits.
impl<V> BatchableOperation<V, V::InterpretationContext> for DynamicUpdateSliceOperation
where
    V: Value<ArrayType> + ZeroLike + Broadcast + Transpose + Slice + UpdateSlice + Reshape,
    DynamicUpdateSliceOperation: InterpretableOperation<ArrayType, V>,
{
    fn batch(
        &self,
        context: &V::InterpretationContext,
        inputs: &[ArrayBatch<V>],
    ) -> Result<Vec<ArrayBatch<V>>, ProgramError> {
        if inputs.len() < 2 {
            return Err(ProgramError::InvalidInputCount { expected: 2, actual: inputs.len() });
        }
        let (_, input_axes, axis_size) = batch_input_metadata(inputs)?;
        if input_axes[2..].iter().any(Option::is_some) {
            return batch_by_lane_expansion(
                context,
                crate::operations::manipulation::DYNAMIC_UPDATE_SLICE_OPERATION_NAME,
                self,
                inputs,
                axis_size,
            );
        }
        let Some(batch_axis) = input_axes[..2].iter().copied().flatten().next() else {
            return apply_with_axes(context, self, inputs, &[None]);
        };
        if inputs.len() == 2 {
            return Ok(vec![inputs[1].clone()]);
        }
        let input = materialize_lane_axis(&inputs[0], batch_axis, axis_size)?;
        let update = materialize_lane_axis(&inputs[1], batch_axis, axis_size)?;
        let zero_index = ArrayBatch::unbatched(inputs[2].value().clone().zero_like());
        let mut lifted_inputs = vec![input, update];
        lifted_inputs.extend(inputs[2..].iter().cloned());
        lifted_inputs.insert(2 + batch_axis, zero_index);
        apply_with_axes(context, self, lifted_inputs.as_slice(), &[Some(batch_axis)])
    }
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;

    use pretty_assertions::assert_eq;

    use crate::ProvidesContext;
    use crate::operations::arithmetic::AddOperation;
    use crate::operations::compare::CompareOperation;
    use crate::operations::manipulation::{LinearDynamicSliceOperation, LinearDynamicUpdateSliceOperation};
    use crate::programs::AtomId;
    use crate::tests::{TestArray, TestArrayDomain};
    use crate::tracing_v2::operations::control_flow::{DefactorizedOperation, SupportsLinearWhile};
    use crate::tracing_v2::operations::reduce::{Reduce, ReductionKind};
    use crate::tracing_v2::test_util::assert_close;
    use crate::tracing_v2::{
        ArrayOperation, CapturedFactor, DifferentiableDomainExtension, LinearArrayOperation, LinearizationTracer,
        value_and_grad,
    };
    use crate::types::DataType;

    use super::*;

    /// Lifts a scalar `i32` index constant into the differentiation trace that `exemplar` belongs to.
    fn index_constant<C>(exemplar: &crate::tracing::Tracer<C>, value: f64) -> crate::tracing::Tracer<C>
    where
        C: crate::contexts::StagingContext<Constant = TestArray>,
    {
        exemplar.context().constant(TestArray::new(ArrayType::scalar(DataType::I32), vec![value]))
    }

    /// Returns a scalar integer-typed test array carrying `value` as its in-band payload.
    fn index(value: f64) -> TestArray {
        TestArray::new(ArrayType::scalar(DataType::I32), vec![value])
    }

    /// Concrete test [`DifferentiationContext`] that reports *no* primal concretization, forcing the staged dispatch
    /// of higher-order rules (the [`WhileOperation`](crate::operations::control_flow::WhileOperation) JVP rule in
    /// particular) even though its primal values are concrete. This mirrors the canonical staged-dispatch domain used
    /// by the `control_flow` tests and lets the defactorization test below exercise the staged fused-loop path that an
    /// eager domain would otherwise unroll.
    #[derive(Copy, Clone, Debug)]
    struct StagedDispatchTestArrayDomain;

    impl crate::Domain for StagedDispatchTestArrayDomain {
        type Type = ArrayType;
        type Value = TestArray;
        type Constant = TestArray;
        type Operation = ArrayOperation<TestArray>;
    }

    impl crate::contexts::Context for StagedDispatchTestArrayDomain {
        fn lift(&self, constant: TestArray) -> Result<TestArray, ProgramError> {
            Ok(constant)
        }

        fn bind<P: Into<Self::Operation>>(
            &self,
            operation: P,
            inputs: &[Self::Value],
        ) -> Result<Vec<Self::Value>, ProgramError> {
            let operation = operation.into();
            operation.interpret(&crate::EagerContext::new(), inputs)
        }
    }

    impl DifferentiationContext for StagedDispatchTestArrayDomain {
        type Tangent = TestArray;
        type LinearOperation<V: Value<ArrayType>, F: Value<ArrayType>> =
            LinearArrayOperation<V, TestArray, Infallible, F, ArrayOperation<TestArray>>;

        fn supports_primal_concretization(&self) -> bool {
            false
        }
    }

    impl ProvidesContext<<TestArray as Value<ArrayType>>::InterpretationContext> for StagedDispatchTestArrayDomain {
        fn context(&self) -> <TestArray as Value<ArrayType>>::InterpretationContext {
            crate::EagerContext::new()
        }
    }

    #[test]
    fn test_slice_value_and_grad_zero_pads_cotangent() {
        // f(x) = sum(slice(x, [1], [3])): the pullback writes the all-ones cotangent into a zero array at the slice
        // offsets, so the gradient is the indicator of the sliced window.
        let (value, gradient) = value_and_grad(
            &TestArrayDomain,
            |x| x.slice(&[1], &[3], &[1]).unwrap().reduce(&[0], ReductionKind::Sum),
            TestArray::vector(vec![1.0, 2.0, 3.0, 4.0]),
        )
        .unwrap();
        assert_close(value.values[0], 5.0);
        assert_eq!(gradient.values, vec![0.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn test_strided_slice_value_and_grad_pads_cotangent() {
        // f(x) = sum(slice(x, [1], [6], strides=[2]) * w) with w = [1, 2, 3]: the forward slice reads positions 1,
        // 3, and 5, so the pullback pads the weighted cotangent [1, 2, 3] with `low = 1`, `interior = 1`, and
        // `high = 0`, scattering it back to exactly those positions.
        let (value, gradient) = value_and_grad(
            &TestArrayDomain,
            |x| {
                let weights = x.context().constant(TestArray::vector(vec![1.0, 2.0, 3.0]));
                (x.slice(&[1], &[6], &[2]).unwrap() * weights).reduce(&[0], ReductionKind::Sum)
            },
            TestArray::vector(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        )
        .unwrap();
        // f = x[1] + 2 * x[3] + 3 * x[5] = 1 + 6 + 15.
        assert_close(value.values[0], 22.0);
        assert_eq!(gradient.values, vec![0.0, 1.0, 0.0, 2.0, 0.0, 3.0]);
    }

    #[test]
    fn test_slice_jacfwd_selects_input_coordinates() {
        // Forward mode through `f(x) = slice(x, [1], [3])` produces the 2x4 selection Jacobian.
        let jacobian = TestArrayDomain
            .jacfwd(|x| Ok(x.slice(&[1], &[3], &[1]).unwrap()), TestArray::vector(vec![1.0, 2.0, 3.0, 4.0]))
            .unwrap();
        let block = jacobian.rows().partials();
        assert_eq!(block.output_shape(), &[2]);
        assert_eq!(block.input_shape(), &[4]);
        assert_eq!(block.values(), &[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_update_slice_value_and_grad_splits_cotangent() {
        // f(x, u) = sum(update_slice(x, u, [1])): the input gradient is the cotangent with the update window zeroed
        // and the update gradient is the slice of the cotangent at the update window.
        let (value, (input_gradient, update_gradient)) = value_and_grad(
            &TestArrayDomain,
            |(x, update)| x.update_slice(&update, &[1]).unwrap().reduce(&[0], ReductionKind::Sum),
            (TestArray::vector(vec![1.0, 2.0, 3.0, 4.0]), TestArray::vector(vec![7.0, 8.0])),
        )
        .unwrap();
        assert_close(value.values[0], 20.0);
        assert_eq!(input_gradient.values, vec![1.0, 0.0, 0.0, 1.0]);
        assert_eq!(update_gradient.values, vec![1.0, 1.0]);
    }

    #[test]
    fn test_dynamic_slice_value_and_grad_scatters_at_captured_indices() {
        // f(x) = sum(dynamic_slice(x, [1], [2])): the integer start index is a constant of the trace (its tangent is
        // a structural zero that the JVP rule ignores), and the pullback scatters the all-ones cotangent into a zero
        // array at the captured index.
        let (value, gradient) = value_and_grad(
            &TestArrayDomain,
            |x| {
                let start = index_constant(&x, 1.0);
                x.dynamic_slice(&[start], &[2]).unwrap().reduce(&[0], ReductionKind::Sum)
            },
            TestArray::vector(vec![1.0, 2.0, 3.0, 4.0]),
        )
        .unwrap();
        assert_close(value.values[0], 5.0);
        assert_eq!(gradient.values, vec![0.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn test_dynamic_slice_jacfwd_selects_input_coordinates() {
        // Forward mode through `f(x) = dynamic_slice(x, [1], [2])` exercises the captured-index dynamic slice under
        // batched basis tangents (the direct batched JVP path).
        let jacobian = TestArrayDomain
            .jacfwd(
                |x| {
                    let start = index_constant(&x, 1.0);
                    Ok(x.dynamic_slice(&[start], &[2]).unwrap())
                },
                TestArray::vector(vec![1.0, 2.0, 3.0, 4.0]),
            )
            .unwrap();
        let block = jacobian.rows().partials();
        assert_eq!(block.output_shape(), &[2]);
        assert_eq!(block.input_shape(), &[4]);
        assert_eq!(block.values(), &[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_dynamic_update_slice_value_and_grad_splits_cotangent() {
        // f(x, u) = sum(dynamic_update_slice(x, u, [1])): the input gradient is the cotangent with the update window
        // zeroed and the update gradient is the dynamic slice of the cotangent at the captured index.
        let (value, (input_gradient, update_gradient)) = value_and_grad(
            &TestArrayDomain,
            |(x, update)| {
                let start = index_constant(&x, 1.0);
                x.dynamic_update_slice(&update, &[start]).unwrap().reduce(&[0], ReductionKind::Sum)
            },
            (TestArray::vector(vec![1.0, 2.0, 3.0, 4.0]), TestArray::vector(vec![7.0, 8.0])),
        )
        .unwrap();
        assert_close(value.values[0], 20.0);
        assert_eq!(input_gradient.values, vec![1.0, 0.0, 0.0, 1.0]);
        assert_eq!(update_gradient.values, vec![1.0, 1.0]);
    }

    #[test]
    fn test_slice_batching_lifts_lane_axis() {
        // A batched operand keeps its lane axis by slicing it fully.
        let input =
            ArrayBatch::mapped(TestArray::matrix(2, 4, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]), 0).unwrap();
        let outputs = SliceOperation::new(vec![1], vec![3]).batch(&crate::EagerContext::new(), &[input]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), Some(0));
        assert_eq!(outputs[0].value().values, vec![1.0, 2.0, 5.0, 6.0]);

        // Lane-uniform operands pass through the unlifted rule.
        let uniform = ArrayBatch::unbatched(TestArray::vector(vec![0.0, 1.0, 2.0, 3.0]));
        let outputs = SliceOperation::new(vec![1], vec![3]).batch(&crate::EagerContext::new(), &[uniform]).unwrap();
        assert_eq!(outputs[0].batch_axis(), None);
        assert_eq!(outputs[0].value().values, vec![1.0, 2.0]);

        // Strided slices keep their strides and gain a unit stride at the lane axis: each lane keeps the elements
        // at positions 0 and 2.
        let input =
            ArrayBatch::mapped(TestArray::matrix(2, 4, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]), 0).unwrap();
        let strided = SliceOperation::new(vec![0], vec![4]).with_strides(vec![2]).unwrap();
        let outputs = strided.batch(&crate::EagerContext::new(), &[input]).unwrap();
        assert_eq!(outputs[0].batch_axis(), Some(0));
        assert_eq!(outputs[0].value().values, vec![0.0, 2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_slice_batching_carries_lane_extended_sharding() {
        use crate::operations::manipulation::Slice;
        use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
        use crate::tracing::trace;
        use crate::tracing_v2::batching::BatchContext;

        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        // The full input is [2 (lane), 4]: the lane axis is replicated and the data axis is sharded over `x`.
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(2), Size::Static(4)]))
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])])
                    .unwrap(),
            )
            .unwrap();
        // Each lane slices its `x`-sharded [4] vector to [2] (2 is divisible by the `x` mesh-axis size, so the slice
        // keeps the sharding); batching restores the replicated lane axis, so the staged slice's output stays sharded.
        let (output_type, _program) = trace(
            &TestArrayDomain,
            |x| {
                let context = x.context().clone();
                Ok(BatchContext::batch(&context, |lane| lane.slice(&[0], &[2], &[1]), x, Some(0), Some(0), None)
                    .unwrap())
            },
            input_type,
        )
        .unwrap();
        assert_eq!(
            output_type.sharding().unwrap().dimensions(),
            &[ShardingDimension::Replicated, ShardingDimension::sharded(["x"])],
        );
    }

    #[test]
    fn test_update_slice_batching_materializes_uniform_operands() {
        // A lane-uniform update is broadcast to gain the lane axis so each lane writes the same block.
        let input =
            ArrayBatch::mapped(TestArray::matrix(2, 4, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]), 0).unwrap();
        let update = ArrayBatch::unbatched(TestArray::vector(vec![9.0, 9.0]));
        let outputs = UpdateSliceOperation::new(vec![1]).batch(&crate::EagerContext::new(), &[input, update]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), Some(0));
        assert_eq!(outputs[0].value().values, vec![0.0, 9.0, 9.0, 3.0, 4.0, 9.0, 9.0, 7.0]);

        // A lane-uniform input is broadcast to gain the lane axis when only the update is batched.
        let input = ArrayBatch::unbatched(TestArray::vector(vec![0.0, 1.0, 2.0, 3.0]));
        let update = ArrayBatch::mapped(TestArray::matrix(2, 2, vec![8.0, 8.0, 9.0, 9.0]), 0).unwrap();
        let outputs = UpdateSliceOperation::new(vec![1]).batch(&crate::EagerContext::new(), &[input, update]).unwrap();
        assert_eq!(outputs[0].batch_axis(), Some(0));
        assert_eq!(outputs[0].value().values, vec![0.0, 8.0, 8.0, 3.0, 0.0, 9.0, 9.0, 3.0]);
    }

    /// Returns a lane-varying scalar integer index batch carrying one start index per lane, mapped at axis `0`.
    fn lane_varying_indices(values: Vec<f64>) -> ArrayBatch<TestArray> {
        let length = values.len();
        ArrayBatch::mapped(
            TestArray::new(ArrayType::new(DataType::I32, Shape::new(vec![Size::Static(length)])), values),
            0,
        )
        .unwrap()
    }

    #[test]
    fn test_dynamic_slice_batching_lifts_lane_uniform_indices() {
        // Lane-uniform start indices lift the lane axis with a zero start index for it.
        let input =
            ArrayBatch::mapped(TestArray::matrix(2, 4, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]), 0).unwrap();
        let outputs = DynamicSliceOperation::new(vec![2])
            .batch(&crate::EagerContext::new(), &[input, ArrayBatch::unbatched(index(1.0))])
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), Some(0));
        assert_eq!(outputs[0].value().values, vec![1.0, 2.0, 5.0, 6.0]);
    }

    #[test]
    fn test_dynamic_slice_batching_expands_lane_varying_indices() {
        // Lane-varying start indices over a lane-uniform operand expand per lane: lane 0 reads `x[0..2]` and lane 1
        // reads `x[2..4]` of the shared operand, restacked along a fresh leading lane axis.
        let uniform = ArrayBatch::unbatched(TestArray::vector(vec![0.0, 1.0, 2.0, 3.0]));
        let outputs = DynamicSliceOperation::new(vec![2])
            .batch(&crate::EagerContext::new(), &[uniform, lane_varying_indices(vec![0.0, 2.0])])
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), Some(0));
        assert_eq!(outputs[0].r#type().shape().dimensions(), &[Size::Static(2), Size::Static(2)]);
        assert_eq!(outputs[0].value().values, vec![0.0, 1.0, 2.0, 3.0]);

        // A batched operand pairs lane `i` of the operand with lane `i` of the indices; lane 1's start index 3 is
        // clamped to 2 so the extracted block stays in bounds.
        let input =
            ArrayBatch::mapped(TestArray::matrix(2, 4, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]), 0).unwrap();
        let outputs = DynamicSliceOperation::new(vec![2])
            .batch(&crate::EagerContext::new(), &[input, lane_varying_indices(vec![1.0, 3.0])])
            .unwrap();
        assert_eq!(outputs[0].batch_axis(), Some(0));
        assert_eq!(outputs[0].value().values, vec![1.0, 2.0, 6.0, 7.0]);

        // An operand batched on a non-leading axis is realigned to the fresh leading lane axis first: the physical
        // `[4, 2]` operand carries per-lane vectors `[0, 1, 2, 3]` and `[4, 5, 6, 7]` along axis 1.
        let trailing =
            ArrayBatch::mapped(TestArray::matrix(4, 2, vec![0.0, 4.0, 1.0, 5.0, 2.0, 6.0, 3.0, 7.0]), 1).unwrap();
        let outputs = DynamicSliceOperation::new(vec![2])
            .batch(&crate::EagerContext::new(), &[trailing, lane_varying_indices(vec![1.0, 2.0])])
            .unwrap();
        assert_eq!(outputs[0].batch_axis(), Some(0));
        assert_eq!(outputs[0].value().values, vec![1.0, 2.0, 6.0, 7.0]);
    }

    #[test]
    fn test_dynamic_update_slice_batching_materializes_uniform_operands() {
        let input =
            ArrayBatch::mapped(TestArray::matrix(2, 4, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]), 0).unwrap();
        let update = ArrayBatch::unbatched(TestArray::vector(vec![9.0, 9.0]));
        let outputs = DynamicUpdateSliceOperation
            .batch(&crate::EagerContext::new(), &[input, update, ArrayBatch::unbatched(index(1.0))])
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), Some(0));
        assert_eq!(outputs[0].value().values, vec![0.0, 9.0, 9.0, 3.0, 4.0, 9.0, 9.0, 7.0]);
    }

    #[test]
    fn test_dynamic_update_slice_batching_expands_lane_varying_indices() {
        // A batched update with lane-varying start indices over a lane-uniform input expands per lane: lane 0
        // writes `[9, 9]` at offset 0 and lane 1 writes `[8, 8]` at offset 2 of the shared input.
        let uniform_input = ArrayBatch::unbatched(TestArray::vector(vec![0.0, 1.0, 2.0, 3.0]));
        let update = ArrayBatch::mapped(TestArray::matrix(2, 2, vec![9.0, 9.0, 8.0, 8.0]), 0).unwrap();
        let outputs = DynamicUpdateSliceOperation
            .batch(&crate::EagerContext::new(), &[uniform_input, update, lane_varying_indices(vec![0.0, 2.0])])
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), Some(0));
        assert_eq!(outputs[0].r#type().shape().dimensions(), &[Size::Static(2), Size::Static(4)]);
        assert_eq!(outputs[0].value().values, vec![9.0, 9.0, 2.0, 3.0, 0.0, 1.0, 8.0, 8.0]);

        // A batched input with a lane-uniform update writes the same block at each lane's own offset.
        let input =
            ArrayBatch::mapped(TestArray::matrix(2, 4, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]), 0).unwrap();
        let uniform_update = ArrayBatch::unbatched(TestArray::vector(vec![9.0, 9.0]));
        let outputs = DynamicUpdateSliceOperation
            .batch(&crate::EagerContext::new(), &[input, uniform_update, lane_varying_indices(vec![1.0, 0.0])])
            .unwrap();
        assert_eq!(outputs[0].batch_axis(), Some(0));
        assert_eq!(outputs[0].value().values, vec![0.0, 9.0, 9.0, 3.0, 9.0, 9.0, 6.0, 7.0]);
    }

    #[test]
    fn test_dynamic_slice_batching_expands_lane_varying_indices_under_tracing() {
        use crate::tracing_v2::batching::BatchContext;

        // vmap-under-tracing composition: each lane extracts a window of the differentiated vector at its own start
        // index, so the batching rule must stage the per-lane expansion (instead of rejecting the lane-varying
        // indices) and the staged slicing operations must transpose. With `starts = [1, 2]` over `x = [1, 2, 3, 4]`
        // the lanes read `[x1, x2]` and `[x2, x3]`, so `f(x) = sum(stack * w)` with `w = [[1, 2], [3, 4]]` is
        // `f = x1 + 2 * x2 + 3 * x2 + 4 * x3` and the gradient is `[0, 1, 5, 4]`.
        let (value, gradient) = value_and_grad(
            &TestArrayDomain,
            |x| {
                let context = x.context().clone();
                let starts = context.constant(TestArray::new(
                    ArrayType::new(DataType::I32, Shape::new(vec![Size::Static(2)])),
                    vec![1.0, 2.0],
                ));
                let stacked: LinearizationTracer<'_, TestArrayDomain> = BatchContext::batch(
                    &context,
                    |(lane, start)| lane.dynamic_slice(&[start], &[2]),
                    (x, starts),
                    (None, Some(0)),
                    Some(0),
                    None,
                )
                .unwrap();
                let weights = context.constant(TestArray::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]));
                (stacked * weights).reduce(&[0, 1], ReductionKind::Sum)
            },
            TestArray::vector(vec![1.0, 2.0, 3.0, 4.0]),
        )
        .unwrap();
        // f = 1 * 2 + 2 * 3 + 3 * 3 + 4 * 4 = 33.
        assert_close(value.values[0], 33.0);
        assert_eq!(gradient.values, vec![0.0, 1.0, 5.0, 4.0]);
    }

    #[test]
    fn test_while_jvp_defactorizes_loop_varying_dynamic_slice_indices() {
        use std::cell::RefCell;
        use std::rc::Rc;

        use crate::operations::compare::ComparisonDirection;
        use crate::operations::control_flow::WhileOperation;
        use crate::parameters::Placeholder;
        use crate::programs::ProgramBuilder;
        use crate::tracing_v2::{DifferentiableOperation, FactorParameterizedOperation};

        type TestArrayOperation = ArrayOperation<TestArray>;
        type TestLinearOperation = LinearArrayOperation<
            TestArray,
            TestArray,
            Infallible,
            CapturedFactor<ArrayType, TestArray>,
            ArrayOperation<TestArray>,
        >;

        let index_type = ArrayType::scalar(DataType::I32);
        let vector_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)]));

        // Condition: `i < 2` over the `(i, x)` loop state.
        let mut condition_builder = ProgramBuilder::<ArrayType, TestArray, TestArrayOperation>::new();
        let condition_counter = condition_builder.add_input(index_type.clone());
        condition_builder.add_input(vector_type.clone());
        let limit = condition_builder.add_constant(index(2.0));
        let predicate = condition_builder
            .add_instruction(CompareOperation::new(ComparisonDirection::LessThan), vec![condition_counter, limit])
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![predicate], vec![Placeholder, Placeholder], vec![Placeholder])
            .unwrap();

        // Body: `x[i] *= 2` through a dynamic slice and update at the loop-varying counter, then `i += 1`.
        let mut body_builder = ProgramBuilder::<ArrayType, TestArray, TestArrayOperation>::new();
        let body_counter = body_builder.add_input(index_type.clone());
        let body_vector = body_builder.add_input(vector_type.clone());
        let lane = body_builder
            .add_instruction(DynamicSliceOperation::new(vec![1]), vec![body_vector, body_counter])
            .unwrap()[0];
        let doubled = body_builder.add_instruction(AddOperation, vec![lane, lane]).unwrap()[0];
        let updated = body_builder
            .add_instruction(DynamicUpdateSliceOperation, vec![body_vector, doubled, body_counter])
            .unwrap()[0];
        let increment = body_builder.add_constant(index(1.0));
        let next_counter = body_builder.add_instruction(AddOperation, vec![body_counter, increment]).unwrap()[0];
        let body = body_builder
            .build::<Vec<TestArray>, Vec<TestArray>>(
                vec![next_counter, updated],
                vec![Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();
        let while_operation = WhileOperation::<ArrayType, TestArray, TestArrayOperation>::new(condition, body).unwrap();

        // Differentiate through the staged path: the body pushforward stages captured-index dynamic slicing
        // operations whose start indices reference the loop-varying counter residual, and the fused linear while
        // defactorizes those references into operand form against the recomputed counter. The integer counter state
        // carries a structural-zero tangent. The staged-dispatch domain reports no primal concretization, so the
        // `while` rule stages the fused loop here instead of unrolling it (the eager path a concretizing domain takes).
        let builder = Rc::new(RefCell::new(ProgramBuilder::<ArrayType, TestArray, TestLinearOperation>::new()));
        let mut context = TangentContext::new(&StagedDispatchTestArrayDomain, builder.clone());
        let vector_tangent = context.input(vector_type.clone());
        let mut counter_tangents = context.stage_nullary_operation(ZeroOperation::new(index_type)).unwrap();
        assert_eq!(counter_tangents.len(), 1);
        let counter = JvpTracer::new(index(0.0), counter_tangents.remove(0));
        let outputs = while_operation
            .jvp(
                &mut context,
                &[counter, JvpTracer::from_value(TestArray::vector(vec![3.0, 4.0, 5.0]), vector_tangent)],
            )
            .unwrap();

        // The primal loop doubles the first two lanes.
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].primal().values, vec![2.0]);
        assert_eq!(outputs[1].primal().values, vec![6.0, 8.0, 5.0]);

        // Replaying the staged pushforward doubles the first two tangent lanes through the defactorized linear
        // slicing operations recomputed inside the fused loop.
        let tangent_output = outputs[1].tangent().atom_id().unwrap();
        drop(outputs);
        drop(context);
        let builder = Rc::try_unwrap(builder).unwrap().into_inner();
        let tangent_program = builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![tangent_output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        // Every loop-varying residual reference must have been defactorized into operand form (loop-entry state
        // enters through closed constant residual injections), so unwrapping the factors yields a directly
        // interpretable pushforward.
        let tangent_program = tangent_program
            .map_operations(|operation| {
                operation.try_map_factors::<TestArray, _>(&mut |factor: &CapturedFactor<ArrayType, TestArray>| {
                    match factor {
                        CapturedFactor::Constant(value) => Ok(value.clone()),
                        CapturedFactor::Reference { .. } => Err(crate::programs::ProgramError::MalformedProgram(
                            "expected all loop-varying residual references to be closed into constants".to_string(),
                        )),
                    }
                })
            })
            .unwrap();
        assert_eq!(
            tangent_program.interpret(vec![TestArray::vector(vec![1.0, 1.0, 1.0])]),
            Ok(vec![TestArray::vector(vec![2.0, 2.0, 1.0])]),
        );
    }

    #[test]
    fn test_dynamic_slicing_defactorize_splices_residual_start_indices() {
        type TestLinearOperation = LinearArrayOperation<
            TestArray,
            TestArray,
            Infallible,
            CapturedFactor<ArrayType, TestArray>,
            ArrayOperation<TestArray>,
        >;

        // A loop-varying residual start index is rewritten into operand form: the residual atom is spliced into the
        // operand list and the operation becomes the recomputed primal dynamic slice.
        let operation = TestLinearOperation::DynamicSlice(LinearDynamicSliceOperation::new(
            vec![CapturedFactor::Reference { index: 1, r#type: ArrayType::scalar(DataType::I32) }],
            vec![2],
        ));
        let residual_atoms = vec![AtomId::new(7), AtomId::new(9)];
        match operation.defactorize(residual_atoms.as_slice(), vec![AtomId::new(3)]).unwrap() {
            DefactorizedOperation::Operation { operation, inputs } => {
                let TestLinearOperation::Recompute(recompute) = operation else {
                    panic!("expected a recomputed dynamic slice");
                };
                let ArrayOperation::DynamicSlice(operation) = recompute.operation() else {
                    panic!("expected a recomputed dynamic slice");
                };
                assert_eq!(operation.sizes(), [2]);
                assert_eq!(inputs, vec![AtomId::new(3), AtomId::new(9)]);
            }
            DefactorizedOperation::Forward { .. } => panic!("expected an operand-form defactorized operation"),
        }

        // The same rewrite applies to the captured-index dynamic update-slice.
        let operation = TestLinearOperation::DynamicUpdateSlice(LinearDynamicUpdateSliceOperation::new(vec![
            CapturedFactor::Reference { index: 0, r#type: ArrayType::scalar(DataType::I32) },
        ]));
        match operation.defactorize(residual_atoms.as_slice(), vec![AtomId::new(3), AtomId::new(4)]).unwrap() {
            DefactorizedOperation::Operation { operation, inputs } => {
                let TestLinearOperation::Recompute(recompute) = operation else {
                    panic!("expected a recomputed dynamic update slice");
                };
                assert!(matches!(recompute.operation(), ArrayOperation::DynamicUpdateSlice(_)));
                assert_eq!(inputs, vec![AtomId::new(3), AtomId::new(4), AtomId::new(7)]);
            }
            DefactorizedOperation::Forward { .. } => panic!("expected an operand-form defactorized operation"),
        }

        // Mixed constant/reference index lists are rejected precisely.
        let mixed = TestLinearOperation::DynamicUpdateSlice(LinearDynamicUpdateSliceOperation::new(vec![
            CapturedFactor::Reference { index: 0, r#type: ArrayType::scalar(DataType::I32) },
            CapturedFactor::Constant(index(0.0)),
        ]));
        assert!(matches!(
            mixed.defactorize(residual_atoms.as_slice(), vec![AtomId::new(3), AtomId::new(4)]),
            Err(ProgramError::UnsupportedOperation { message })
                if message == "jvp of a while loop whose body captures a mix of loop-varying and constant \
                    dynamic_update_slice start indices is not supported",
        ));

        // All-constant index lists pass through unchanged via the closed-factor catch-all.
        let constant = TestLinearOperation::DynamicSlice(LinearDynamicSliceOperation::new(
            vec![CapturedFactor::Constant(index(1.0))],
            vec![2],
        ));
        match constant.defactorize(residual_atoms.as_slice(), vec![AtomId::new(3)]).unwrap() {
            DefactorizedOperation::Operation { operation, inputs } => {
                assert!(matches!(operation, TestLinearOperation::DynamicSlice(_)));
                assert_eq!(inputs, vec![AtomId::new(3)]);
            }
            DefactorizedOperation::Forward { .. } => panic!("expected an operand-form defactorized operation"),
        }
    }
}
