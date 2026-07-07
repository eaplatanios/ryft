use crate::batching::ArrayBatch;
use crate::batching::BatchAxis;
use crate::batching::BatchableOperation;
use crate::batching::BatchingError;
use crate::batching::InterpretableBatchableOperation;
use crate::contexts::{Context, StagingContext};
use crate::differentiation::TransposableOperation;
use crate::interpretation::InterpretableOperation;
use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::constants::{Zero, ZeroLike, ZeroOperation};
use crate::operations::manipulation::{
    Broadcast, DynamicSlice, DynamicSliceOperation, DynamicUpdateSlice, DynamicUpdateSliceOperation, PadOperation,
    Reshape, Slice, SliceOperation, Transpose, UpdateSlice, UpdateSliceOperation,
};
use crate::partial::PartialValue;
use crate::programs::{MaybeZero, ProgramError, Value};
use crate::tracing::{Tracer, TracingContext};

use crate::differentiation::DifferentiationDual;
use crate::tracing_v2::differentiation::{DifferentiableOperation, materialize};
use crate::types::{ArrayType, Shape, Size, TypeError, Typed};

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
impl<V: Value<Type = ArrayType>, O> TransposableOperation<V, O> for SliceOperation
where
    O: Operation<ArrayType> + From<UpdateSliceOperation> + From<PadOperation> + From<ZeroOperation<ArrayType>>,
{
    fn transpose(
        &self,
        context: &mut TracingContext<V, O>,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        check_count!("output", outputs, 1, ProgramError);
        match &outputs[0] {
            MaybeZero::Zero(_) => Ok(vec![MaybeZero::Zero(inputs[0].r#type().into_owned())]),
            MaybeZero::Value(cotangent) if self.strides().iter().all(|stride| *stride == 1) => {
                let zeros = materialize(context, MaybeZero::Zero(inputs[0].r#type().into_owned()))?;
                let outputs = context.stage_operation(
                    UpdateSliceOperation::new(self.start_indices().to_vec()),
                    &[zeros, cotangent.clone()],
                )?;
                check_count!("output", outputs, 1, ProgramError);
                Ok(vec![MaybeZero::Value(outputs.into_iter().next().unwrap())])
            }
            MaybeZero::Value(cotangent) => {
                let input_type = inputs[0].r#type();
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
                                "'slice' transpose requires a static input shape but axis {axis} has size {dimension}",
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
                let zero = materialize(context, MaybeZero::Zero(ArrayType::scalar(input_type.data_type())))?;
                let outputs = context.stage_operation(
                    PadOperation::new(edge_padding_low, edge_padding_high, interior_padding)?,
                    &[cotangent.clone(), zero],
                )?;
                check_count!("output", outputs, 1, ProgramError);
                Ok(vec![MaybeZero::Value(outputs.into_iter().next().unwrap())])
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
impl<V: Value<Type = ArrayType>, O> TransposableOperation<V, O> for UpdateSliceOperation
where
    O: Operation<ArrayType> + From<SliceOperation> + From<UpdateSliceOperation> + From<ZeroOperation<ArrayType>>,
{
    fn transpose(
        &self,
        context: &mut TracingContext<V, O>,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        check_count!("output", outputs, 1, ProgramError);
        match &outputs[0] {
            MaybeZero::Zero(_) => Ok(vec![
                MaybeZero::Zero(inputs[0].r#type().into_owned()),
                MaybeZero::Zero(inputs[1].r#type().into_owned()),
            ]),
            MaybeZero::Value(cotangent) => {
                let update_type = inputs[1].r#type();
                let update_sizes = static_update_sizes(UPDATE_SLICE_TRANSPOSE_CONTEXT, &update_type)?;
                let zeros = materialize(context, MaybeZero::Zero(update_type.clone().into_owned()))?;
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
                    MaybeZero::Value(input_cotangents.into_iter().next().unwrap()),
                    MaybeZero::Value(update_cotangents.into_iter().next().unwrap()),
                ])
            }
        }
    }
}

/// Operation-name prefix used by [`static_update_sizes`] errors raised from the update-slice transpose rule.
const UPDATE_SLICE_TRANSPOSE_CONTEXT: &str = "'update_slice' transpose";

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

/// Forward-mode rule for [`DynamicSliceOperation`]: `dynamic_slice` is linear in the operand, and the scalar
/// start indices are non-differentiated primal operand edges, so the tangent slices the operand tangent at the same
/// primal start indices. A zero operand tangent yields a typed zero output tangent.
impl<C: Context<Type = ArrayType>> DifferentiableOperation<C> for DynamicSliceOperation
where
    C::Operation: Clone + From<DynamicSliceOperation>,
    C::Value: DynamicSlice,
{
    fn jvp(
        &self,
        _context: &C,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, ProgramError> {
        let (operand, start_indices) =
            inputs.split_first().ok_or(ProgramError::InvalidInputCount { expected: 1, actual: 0 })?;
        let primal_starts = start_indices.iter().map(|dual| dual.primal().clone()).collect::<Vec<_>>();
        let primal = operand.primal().dynamic_slice(&primal_starts, self.sizes())?;
        let tangent = match operand.tangent() {
            MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().into_owned()),
            MaybeZero::Value(tangent) => MaybeZero::Value(tangent.dynamic_slice(&primal_starts, self.sizes())?),
        };
        Ok(vec![DifferentiationDual::new(primal, tangent)])
    }
}

/// Forward-mode rule for [`DynamicUpdateSliceOperation`]: `dynamic_update_slice` is jointly linear in the operand
/// and the update, while the scalar start indices are non-differentiated primal operand edges, so the tangent updates
/// the operand tangent with the update tangent at the same primal start indices. A zero operand and update tangent
/// yields a typed zero output tangent.
impl<C: Context<Type = ArrayType> + Zero<C::Value>> DifferentiableOperation<C> for DynamicUpdateSliceOperation
where
    C::Operation: Clone + From<DynamicUpdateSliceOperation>,
    C::Value: DynamicUpdateSlice,
{
    fn jvp(
        &self,
        context: &C,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, ProgramError> {
        if inputs.len() < 2 {
            return Err(ProgramError::InvalidInputCount { expected: 2, actual: inputs.len() }.into());
        }
        let operand = &inputs[0];
        let update = &inputs[1];
        let primal_starts = inputs[2..].iter().map(|dual| dual.primal().clone()).collect::<Vec<_>>();
        let primal = operand.primal().dynamic_update_slice(update.primal(), &primal_starts)?;
        let tangent = if operand.tangent().is_zero() && update.tangent().is_zero() {
            MaybeZero::Zero(primal.r#type().into_owned())
        } else {
            let operand_tangent = materialize(context, operand.tangent().clone())?;
            let update_tangent = materialize(context, update.tangent().clone())?;
            MaybeZero::Value(operand_tangent.dynamic_update_slice(&update_tangent, &primal_starts)?)
        };
        Ok(vec![DifferentiationDual::new(primal, tangent)])
    }
}

/// Forward-mode rule for [`SliceOperation`]: slicing is a linear map, so the primal output is the slice of the
/// operand primal and the tangent is the same slice of the operand tangent. A zero operand tangent yields a typed zero
/// output tangent.
impl<C: Context<Type = ArrayType>> DifferentiableOperation<C> for SliceOperation
where
    C::Operation: Clone + From<SliceOperation>,
    C::Value: Slice,
{
    fn jvp(
        &self,
        _context: &C,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        let primal = inputs[0].primal().slice(self.start_indices(), self.limit_indices(), self.strides())?;
        let tangent = match inputs[0].tangent() {
            MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().into_owned()),
            MaybeZero::Value(tangent) => {
                MaybeZero::Value(tangent.slice(self.start_indices(), self.limit_indices(), self.strides())?)
            }
        };
        Ok(vec![DifferentiationDual::new(primal, tangent)])
    }
}

/// Forward-mode rule for [`UpdateSliceOperation`]: the operation is jointly linear in its operand and update, so
/// the tangent updates the operand tangent with the update tangent at the same static start indices. A zero operand and
/// update tangent yields a typed zero output tangent.
impl<C: Context<Type = ArrayType> + Zero<C::Value>> DifferentiableOperation<C> for UpdateSliceOperation
where
    C::Operation: Clone + From<UpdateSliceOperation>,
    C::Value: UpdateSlice,
{
    fn jvp(
        &self,
        context: &C,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        let operand = &inputs[0];
        let update = &inputs[1];
        let primal = operand.primal().update_slice(update.primal(), self.start_indices())?;
        let tangent = if operand.tangent().is_zero() && update.tangent().is_zero() {
            MaybeZero::Zero(primal.r#type().into_owned())
        } else {
            let operand_tangent = materialize(context, operand.tangent().clone())?;
            let update_tangent = materialize(context, update.tangent().clone())?;
            MaybeZero::Value(operand_tangent.update_slice(&update_tangent, self.start_indices())?)
        };
        Ok(vec![DifferentiationDual::new(primal, tangent)])
    }
}

/// Extracts item `item` of a per-item expansion operand: batched operands (whose batch axis must already sit at the
/// leading physical axis) contribute slice `item` with the batch axis dropped, while replicated operands are used
/// whole. Batched operand types must be fully static so the item slice bounds are provable; `operation_name` selects
/// the rule named in the error reported otherwise.
pub(crate) fn expansion_item<V>(
    operation_name: &'static str,
    input: &ArrayBatch<V>,
    item: usize,
) -> Result<V, ProgramError>
where
    V: Value<Type = ArrayType> + Slice + Reshape,
{
    if input.batch_axis().is_replicated() {
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
                        "'{operation_name}' per-item expansion requires static batched operand types but got \
                         {input_type}",
                    ),
                }
                .into()
            })
        })
        .collect::<Result<Vec<usize>, ProgramError>>()?;
    let mut start_indices = vec![0; dimensions.len()];
    start_indices[0] = item;
    let mut limit_indices = dimensions.clone();
    limit_indices[0] = item + 1;
    let unit_strides = vec![1; dimensions.len()];
    let item_value =
        input
            .value()
            .clone()
            .slice(start_indices.as_slice(), limit_indices.as_slice(), unit_strides.as_slice())?;
    item_value.reshape(Shape::new(dimensions[1..].iter().map(|&dimension| Size::Static(dimension)).collect()))
}

/// Stacks per-item expansion results along a fresh leading batch axis of size `axis_size`: item `0` seeds the stacked
/// accumulator by broadcast replication and later items overwrite their slices via [`UpdateSlice`] at static item
/// offsets. `interpret_item` produces the per-item result; an empty batch axis is rejected with a precise error
/// naming `operation_name` because no batch item can seed the accumulator.
pub(crate) fn stack_expansion_items<V, InterpretItemFn>(
    operation_name: &'static str,
    axis_size: usize,
    mut interpret_item: InterpretItemFn,
) -> Result<ArrayBatch<V>, BatchingError>
where
    V: Value<Type = ArrayType> + Broadcast + UpdateSlice + Reshape,
    InterpretItemFn: FnMut(usize) -> Result<V, ProgramError>,
{
    let mut accumulator: Option<V> = None;
    for item in 0..axis_size {
        let output_item = interpret_item(item)?;
        let output_item_type = output_item.r#type().into_owned();
        accumulator = Some(match accumulator {
            None => {
                // Item `0` seeds the stacked accumulator by replication; later items overwrite their slices.
                let mut stacked_dimensions = Vec::with_capacity(output_item_type.rank() + 1);
                stacked_dimensions.push(Size::Static(axis_size));
                stacked_dimensions.extend(output_item_type.shape().dimensions().iter().cloned());
                let stacked_type = ArrayType::new(output_item_type.data_type(), Shape::new(stacked_dimensions));
                let output_axes: Vec<usize> = (1..=output_item_type.rank()).collect();
                output_item.broadcast(stacked_type, output_axes.as_slice())?
            }
            Some(accumulator) => {
                let mut expanded_dimensions = Vec::with_capacity(output_item_type.rank() + 1);
                expanded_dimensions.push(Size::Static(1));
                expanded_dimensions.extend(output_item_type.shape().dimensions().iter().cloned());
                let expanded = output_item.reshape(Shape::new(expanded_dimensions))?;
                let mut write_indices = vec![0; output_item_type.rank() + 1];
                write_indices[0] = item;
                accumulator.update_slice(&expanded, write_indices.as_slice())?
            }
        });
    }
    let Some(accumulator) = accumulator else {
        return Err(BatchingError::UnsupportedOperation {
            message: format!("'{operation_name}' does not support per-item expansion over an empty batch axis"),
        }
        .into());
    };
    let stacked_type = accumulator.r#type().into_owned();
    ArrayBatch::new(stacked_type, accumulator, Some(0))
}

/// Applies a single-output `operation` independently per batch item and restacks the results along a fresh leading
/// batch axis: every input is realigned so any mapped batch axis sits at the leading physical axis, item `item` of each
/// batched input is extracted via [`expansion_item`] (replicated inputs are used whole), and the per-item outputs
/// are stacked via [`stack_expansion_items`]. This is the shared fallback for batched operands that cannot ride
/// along structurally — batch-varying dynamic-slice start indices and batch-varying pad padding values — and it stages
/// `O(axis_size)` operations because everything goes through the value capability traits (which also makes it work
/// identically in eager and tracing contexts, since capabilities stage on tracers).
pub(crate) fn batch_by_item_expansion<V, C, O>(
    context: &C,
    operation_name: &'static str,
    operation: &O,
    inputs: &[ArrayBatch<V>],
    axis_size: usize,
) -> Result<Vec<ArrayBatch<V>>, BatchingError>
where
    V: Value<Type = ArrayType> + Broadcast + Transpose + Slice + UpdateSlice + Reshape,
    O: InterpretableOperation<V, C>,
{
    if inputs.is_empty() {
        return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
    }
    let aligned = inputs.iter().map(|input| input.move_axis(0)).collect::<Result<Vec<_>, _>>()?;
    let stacked = stack_expansion_items(operation_name, axis_size, |item| {
        let item_inputs = aligned
            .iter()
            .map(|input| expansion_item(operation_name, input, item))
            .collect::<Result<Vec<_>, _>>()?;
        let mut outputs = operation.interpret(context, item_inputs.as_slice())?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    })?;
    Ok(vec![stacked])
}

/// Batching rule for [`SliceOperation`]: a batched operand keeps its batch axis by slicing it fully, so the lifted
/// operation inserts start index `0`, limit `axis_size`, and stride `1` at the batch axis position.
impl<V: Value<Type = ArrayType>, C> BatchableOperation<V, C> for SliceOperation
where
    SliceOperation: InterpretableOperation<V, C>,
{
    fn batch(&self, context: &C, inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, BatchingError> {
        check_count!("input", inputs, 1, ProgramError);
        let batch_axes: Vec<Option<usize>> = inputs.iter().map(|input| input.batch_axis().axis()).collect();
        match batch_axes[0] {
            None => self.interpret_with_batch_axes(context, inputs, &[BatchAxis::replicated()]),
            Some(batch_axis) => {
                let axis_size = ArrayBatch::common_batch_size(inputs)?.expect("a mapped input pins the batch size");
                let mut start_indices = self.start_indices().to_vec();
                start_indices.insert(batch_axis, 0);
                let mut limit_indices = self.limit_indices().to_vec();
                limit_indices.insert(batch_axis, axis_size);
                let mut strides = self.strides().to_vec();
                strides.insert(batch_axis, 1);
                let lifted = SliceOperation::new(start_indices, limit_indices).with_strides(strides)?;
                lifted.interpret_with_batch_axes(context, inputs, &[BatchAxis::new(batch_axis)])
            }
        }
    }
}

/// Batching rule for [`UpdateSliceOperation`]: the input and update operands are aligned on one physical batch axis
/// (replicated operands are broadcast to gain it), and the lifted operation inserts start index `0` at that axis
/// so each batch item updates its own block.
impl<V: Value<Type = ArrayType> + Broadcast + Transpose, C> BatchableOperation<V, C> for UpdateSliceOperation
where
    UpdateSliceOperation: InterpretableOperation<V, C>,
{
    fn batch(&self, context: &C, inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, BatchingError> {
        check_count!("input", inputs, 2, ProgramError);
        let batch_axes: Vec<Option<usize>> = inputs.iter().map(|input| input.batch_axis().axis()).collect();
        let Some(batch_axis) = batch_axes.iter().copied().flatten().next() else {
            return self.interpret_with_batch_axes(context, inputs, &[BatchAxis::replicated()]);
        };
        let axis_size = ArrayBatch::common_batch_size(inputs)?.expect("a mapped input pins the batch size");
        let input = inputs[0].match_axis(batch_axis, axis_size)?;
        let update = inputs[1].match_axis(batch_axis, axis_size)?;
        let mut start_indices = self.start_indices().to_vec();
        start_indices.insert(batch_axis, 0);
        UpdateSliceOperation::new(start_indices).interpret_with_batch_axes(
            context,
            &[input, update],
            &[BatchAxis::new(batch_axis)],
        )
    }
}

/// Batching rule for [`DynamicSliceOperation`].
///
/// Replicated start indices keep the structural fast path: a batched operand keeps its batch axis by slicing it
/// fully, so the lifted operation inserts size `axis_size` at the batch axis position and a zero start index for it,
/// derived from an existing index operand via [`ZeroLike`] so the inserted index carries the same scalar integer
/// type. Rank-0 operands have no index operands to donate a zero index, but a rank-0 dynamic slice is the identity
/// map, so the batched operand passes through unchanged.
///
/// Batch-varying (batched) start indices cannot ride along structurally — every batch item needs its own slice origin
/// while the lifted operation reads one origin for all batch items — so the rule falls back to per-item expansion via
/// `batch_by_item_expansion`: each batch item's operand (when batched; a replicated operand is used whole) and start
/// indices are extracted, sliced dynamically per item, and restacked along a fresh leading batch axis (the result's
/// batch axis is `0` even when the operand carried its batch axis elsewhere). The expansion stages `O(batch_size)`
/// operations — a gather-based rule is an explicit non-goal — and behaves identically in eager and tracing contexts
/// because it only goes through the value capability traits.
impl<V, C> BatchableOperation<V, C> for DynamicSliceOperation
where
    V: Value<Type = ArrayType> + ZeroLike + Broadcast + Transpose + Slice + UpdateSlice + Reshape,
    DynamicSliceOperation: InterpretableOperation<V, C>,
{
    fn batch(&self, context: &C, inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, BatchingError> {
        if inputs.is_empty() {
            return Err(ProgramError::InvalidInputCount { expected: 1 + self.sizes().len(), actual: 0 }.into());
        }
        let batch_axes: Vec<Option<usize>> = inputs.iter().map(|input| input.batch_axis().axis()).collect();
        let axis_size = ArrayBatch::common_batch_size(inputs)?;
        if batch_axes[1..].iter().any(Option::is_some) {
            return batch_by_item_expansion(
                context,
                crate::operations::manipulation::DYNAMIC_SLICE_OPERATION_NAME,
                self,
                inputs,
                axis_size.expect("a mapped input pins the batch size"),
            );
        }
        let Some(batch_axis) = batch_axes[0] else {
            return self.interpret_with_batch_axes(context, inputs, &[BatchAxis::replicated()]);
        };
        if self.sizes().is_empty() {
            return Ok(vec![inputs[0].clone()]);
        }
        let axis_size = axis_size.expect("a mapped input pins the batch size");
        let mut sizes = self.sizes().to_vec();
        sizes.insert(batch_axis, axis_size);
        let zero_index = ArrayBatch::replicated(inputs[1].value().clone().zero_like());
        let mut lifted_inputs = inputs.to_vec();
        lifted_inputs.insert(1 + batch_axis, zero_index);
        DynamicSliceOperation::new(sizes).interpret_with_batch_axes(
            context,
            lifted_inputs.as_slice(),
            &[BatchAxis::new(batch_axis)],
        )
    }
}

/// Batching rule for [`DynamicUpdateSliceOperation`].
///
/// Replicated start indices keep the structural fast path: the input and update operands are aligned on one
/// physical batch axis (replicated operands are broadcast to gain it), and the lifted operation inserts a zero
/// start index for that axis, derived from an existing index operand via [`ZeroLike`] so the inserted index carries
/// the same scalar integer type. Rank-0 operands have no index operands to donate a zero index, but a rank-0 dynamic
/// update-slice replaces the operand with the update entirely, so the update operand passes through unchanged.
///
/// Batch-varying (batched) start indices cannot ride along structurally — every batch item needs its own update origin
/// while the lifted operation reads one origin for all batch items — so the rule falls back to per-item expansion via
/// `batch_by_item_expansion`: each batch item's input, update, and start indices are extracted (replicated operands
/// are used whole), updated per item, and restacked along a fresh leading batch axis (the result's batch axis is `0`
/// even when the operands carried their batch axes elsewhere). The expansion stages `O(batch_size)` operations — a
/// scatter-based rule is an explicit non-goal — and behaves identically in eager and tracing contexts because it
/// only goes through the value capability traits.
impl<V, C> BatchableOperation<V, C> for DynamicUpdateSliceOperation
where
    V: Value<Type = ArrayType> + ZeroLike + Broadcast + Transpose + Slice + UpdateSlice + Reshape,
    DynamicUpdateSliceOperation: InterpretableOperation<V, C>,
{
    fn batch(&self, context: &C, inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, BatchingError> {
        if inputs.len() < 2 {
            return Err(ProgramError::InvalidInputCount { expected: 2, actual: inputs.len() }.into());
        }
        let batch_axes: Vec<Option<usize>> = inputs.iter().map(|input| input.batch_axis().axis()).collect();
        let axis_size = ArrayBatch::common_batch_size(inputs)?;
        if batch_axes[2..].iter().any(Option::is_some) {
            return batch_by_item_expansion(
                context,
                crate::operations::manipulation::DYNAMIC_UPDATE_SLICE_OPERATION_NAME,
                self,
                inputs,
                axis_size.expect("a mapped input pins the batch size"),
            );
        }
        let Some(batch_axis) = batch_axes[..2].iter().copied().flatten().next() else {
            return self.interpret_with_batch_axes(context, inputs, &[BatchAxis::replicated()]);
        };
        if inputs.len() == 2 {
            return Ok(vec![inputs[1].clone()]);
        }
        let axis_size = axis_size.expect("a mapped input pins the batch size");
        let input = inputs[0].match_axis(batch_axis, axis_size)?;
        let update = inputs[1].match_axis(batch_axis, axis_size)?;
        let zero_index = ArrayBatch::replicated(inputs[2].value().clone().zero_like());
        let mut lifted_inputs = vec![input, update];
        lifted_inputs.extend(inputs[2..].iter().cloned());
        lifted_inputs.insert(2 + batch_axis, zero_index);
        self.interpret_with_batch_axes(context, lifted_inputs.as_slice(), &[BatchAxis::new(batch_axis)])
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::contexts::EagerContext;
    use crate::tests::TestArray;
    use crate::tracing_v2::ArrayOperation;
    use crate::tracing_v2::operations::reduce::{Reduce, ReductionKind};
    use crate::tracing_v2::test_util::assert_close;
    use crate::tracing_v2::{DifferentiableDomainExtension, NestedTracer, value_and_grad};
    use crate::types::DataType;

    use super::*;
    use crate::batching::BatchAxis;

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

    #[test]
    fn test_slice_value_and_grad_zero_pads_cotangent() {
        // f(x) = sum(slice(x, [1], [3])): the pullback writes the all-ones cotangent into a zero array at the slice
        // offsets, so the gradient is the indicator of the sliced window.
        let (value, gradient) = value_and_grad(
            &EagerContext::<TestArray, ArrayOperation<TestArray>>::new(),
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
            &EagerContext::<TestArray, ArrayOperation<TestArray>>::new(),
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
        let jacobian = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
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
            &EagerContext::<TestArray, ArrayOperation<TestArray>>::new(),
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
            &EagerContext::<TestArray, ArrayOperation<TestArray>>::new(),
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
        let jacobian = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
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
            &EagerContext::<TestArray, ArrayOperation<TestArray>>::new(),
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
    fn test_slice_batching_lifts_batch_axis() {
        // A batched operand keeps its batch axis by slicing it fully.
        let input = {
            let value = TestArray::matrix(2, 4, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
            ArrayBatch::new(value.r#type().into_owned(), value, Some(0))
        }
        .unwrap();
        let outputs = SliceOperation::new(vec![1], vec![3])
            .batch(&crate::EagerContext::<TestArray>::new(), &[input])
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value().values, vec![1.0, 2.0, 5.0, 6.0]);

        // Replicated operands pass through the unlifted rule.
        let uniform = ArrayBatch::replicated(TestArray::vector(vec![0.0, 1.0, 2.0, 3.0]));
        let outputs = SliceOperation::new(vec![1], vec![3])
            .batch(&crate::EagerContext::<TestArray>::new(), &[uniform])
            .unwrap();
        assert_eq!(outputs[0].batch_axis(), BatchAxis::replicated());
        assert_eq!(outputs[0].value().values, vec![1.0, 2.0]);

        // Strided slices keep their strides and gain a unit stride at the batch axis: each batch item keeps the
        // elements at positions 0 and 2.
        let input = {
            let value = TestArray::matrix(2, 4, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
            ArrayBatch::new(value.r#type().into_owned(), value, Some(0))
        }
        .unwrap();
        let strided = SliceOperation::new(vec![0], vec![4]).with_strides(vec![2]).unwrap();
        let outputs = strided.batch(&crate::EagerContext::<TestArray>::new(), &[input]).unwrap();
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value().values, vec![0.0, 2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_slice_batching_carries_batch_extended_sharding() {
        use crate::batching::Batch;
        use crate::contexts::Domain;
        use crate::operations::manipulation::Slice;
        use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};

        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        // The full input is [2 (batch), 4]: the batch axis is replicated and the data axis is sharded over `x`.
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(2), Size::Static(4)]))
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])])
                    .unwrap(),
            )
            .unwrap();
        // Each batch item slices its `x`-sharded [4] vector to [2] (2 is divisible by the `x` mesh-axis size, so the
        // slice keeps the sharding); batching restores the replicated batch axis, so the staged slice's output stays
        // sharded.
        let (output_type, _program) = EagerContext::<TestArray, ArrayOperation<TestArray>>::trace(
            |x| {
                let context = x.context().clone();
                Ok(Batch::batch(
                    &context,
                    |item| item.slice(&[0], &[2], &[1]),
                    x,
                    BatchAxis::new(0),
                    BatchAxis::new(0),
                    None,
                )
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
        // A replicated update is broadcast to gain the batch axis so each batch item writes the same block.
        let input = {
            let value = TestArray::matrix(2, 4, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
            ArrayBatch::new(value.r#type().into_owned(), value, Some(0))
        }
        .unwrap();
        let update = ArrayBatch::replicated(TestArray::vector(vec![9.0, 9.0]));
        let outputs = UpdateSliceOperation::new(vec![1])
            .batch(&crate::EagerContext::<TestArray>::new(), &[input, update])
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value().values, vec![0.0, 9.0, 9.0, 3.0, 4.0, 9.0, 9.0, 7.0]);

        // A replicated input is broadcast to gain the batch axis when only the update is batched.
        let input = ArrayBatch::replicated(TestArray::vector(vec![0.0, 1.0, 2.0, 3.0]));
        let update = {
            let value = TestArray::matrix(2, 2, vec![8.0, 8.0, 9.0, 9.0]);
            ArrayBatch::new(value.r#type().into_owned(), value, Some(0))
        }
        .unwrap();
        let outputs = UpdateSliceOperation::new(vec![1])
            .batch(&crate::EagerContext::<TestArray>::new(), &[input, update])
            .unwrap();
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value().values, vec![0.0, 8.0, 8.0, 3.0, 0.0, 9.0, 9.0, 3.0]);
    }

    /// Returns a batch-varying scalar integer index batch carrying one start index per batch item, mapped at axis `0`.
    fn batch_varying_indices(values: Vec<f64>) -> ArrayBatch<TestArray> {
        let length = values.len();
        let value = TestArray::new(ArrayType::new(DataType::I32, Shape::new(vec![Size::Static(length)])), values);
        ArrayBatch::new(value.r#type().into_owned(), value, Some(0)).unwrap()
    }

    #[test]
    fn test_dynamic_slice_batching_lifts_replicated_indices() {
        // Replicated start indices lift the batch axis with a zero start index for it.
        let input = {
            let value = TestArray::matrix(2, 4, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
            ArrayBatch::new(value.r#type().into_owned(), value, Some(0))
        }
        .unwrap();
        let outputs = DynamicSliceOperation::new(vec![2])
            .batch(&crate::EagerContext::<TestArray>::new(), &[input, ArrayBatch::replicated(index(1.0))])
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value().values, vec![1.0, 2.0, 5.0, 6.0]);
    }

    #[test]
    fn test_dynamic_slice_batching_expands_batch_varying_indices() {
        // Batch-varying start indices over a replicated operand expand per item: item 0 reads `x[0..2]` and item 1
        // reads `x[2..4]` of the shared operand, restacked along a fresh leading batch axis.
        let uniform = ArrayBatch::replicated(TestArray::vector(vec![0.0, 1.0, 2.0, 3.0]));
        let outputs = DynamicSliceOperation::new(vec![2])
            .batch(&crate::EagerContext::<TestArray>::new(), &[uniform, batch_varying_indices(vec![0.0, 2.0])])
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].r#type().shape().dimensions(), &[Size::Static(2), Size::Static(2)]);
        assert_eq!(outputs[0].value().values, vec![0.0, 1.0, 2.0, 3.0]);

        // A batched operand pairs item `i` of the operand with item `i` of the indices; item 1's start index 3 is
        // clamped to 2 so the extracted block stays in bounds.
        let input = {
            let value = TestArray::matrix(2, 4, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
            ArrayBatch::new(value.r#type().into_owned(), value, Some(0))
        }
        .unwrap();
        let outputs = DynamicSliceOperation::new(vec![2])
            .batch(&crate::EagerContext::<TestArray>::new(), &[input, batch_varying_indices(vec![1.0, 3.0])])
            .unwrap();
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value().values, vec![1.0, 2.0, 6.0, 7.0]);

        // An operand batched on a non-leading axis is realigned to the fresh leading batch axis first: the physical
        // `[4, 2]` operand carries per-item vectors `[0, 1, 2, 3]` and `[4, 5, 6, 7]` along axis 1.
        let trailing = {
            let value = TestArray::matrix(4, 2, vec![0.0, 4.0, 1.0, 5.0, 2.0, 6.0, 3.0, 7.0]);
            ArrayBatch::new(value.r#type().into_owned(), value, Some(1))
        }
        .unwrap();
        let outputs = DynamicSliceOperation::new(vec![2])
            .batch(&crate::EagerContext::<TestArray>::new(), &[trailing, batch_varying_indices(vec![1.0, 2.0])])
            .unwrap();
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value().values, vec![1.0, 2.0, 6.0, 7.0]);
    }

    #[test]
    fn test_dynamic_update_slice_batching_materializes_uniform_operands() {
        let input = {
            let value = TestArray::matrix(2, 4, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
            ArrayBatch::new(value.r#type().into_owned(), value, Some(0))
        }
        .unwrap();
        let update = ArrayBatch::replicated(TestArray::vector(vec![9.0, 9.0]));
        let outputs = DynamicUpdateSliceOperation
            .batch(&crate::EagerContext::<TestArray>::new(), &[input, update, ArrayBatch::replicated(index(1.0))])
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value().values, vec![0.0, 9.0, 9.0, 3.0, 4.0, 9.0, 9.0, 7.0]);
    }

    #[test]
    fn test_dynamic_update_slice_batching_expands_batch_varying_indices() {
        // A batched update with batch-varying start indices over a replicated input expands per item: item 0
        // writes `[9, 9]` at offset 0 and item 1 writes `[8, 8]` at offset 2 of the shared input.
        let uniform_input = ArrayBatch::replicated(TestArray::vector(vec![0.0, 1.0, 2.0, 3.0]));
        let update = {
            let value = TestArray::matrix(2, 2, vec![9.0, 9.0, 8.0, 8.0]);
            ArrayBatch::new(value.r#type().into_owned(), value, Some(0))
        }
        .unwrap();
        let outputs = DynamicUpdateSliceOperation
            .batch(
                &crate::EagerContext::<TestArray>::new(),
                &[uniform_input, update, batch_varying_indices(vec![0.0, 2.0])],
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].r#type().shape().dimensions(), &[Size::Static(2), Size::Static(4)]);
        assert_eq!(outputs[0].value().values, vec![9.0, 9.0, 2.0, 3.0, 0.0, 1.0, 8.0, 8.0]);

        // A batched input with a replicated update writes the same block at each batch item's own offset.
        let input = {
            let value = TestArray::matrix(2, 4, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
            ArrayBatch::new(value.r#type().into_owned(), value, Some(0))
        }
        .unwrap();
        let uniform_update = ArrayBatch::replicated(TestArray::vector(vec![9.0, 9.0]));
        let outputs = DynamicUpdateSliceOperation
            .batch(
                &crate::EagerContext::<TestArray>::new(),
                &[input, uniform_update, batch_varying_indices(vec![1.0, 0.0])],
            )
            .unwrap();
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value().values, vec![0.0, 9.0, 9.0, 3.0, 9.0, 9.0, 6.0, 7.0]);
    }

    #[test]
    fn test_dynamic_slice_batching_expands_batch_varying_indices_under_tracing() {
        use crate::batching::Batch;

        // vmap-under-tracing composition: each batch item extracts a window of the differentiated vector at its own
        // start index, so the batching rule must stage the per-item expansion (instead of rejecting the batch-varying
        // indices) and the staged slicing operations must transpose. With `starts = [1, 2]` over `x = [1, 2, 3, 4]`
        // the batch items read `[x1, x2]` and `[x2, x3]`, so `f(x) = sum(stack * w)` with `w = [[1, 2], [3, 4]]` is
        // `f = x1 + 2 * x2 + 3 * x2 + 4 * x3` and the gradient is `[0, 1, 5, 4]`.
        let (value, gradient) = value_and_grad(
            &EagerContext::<TestArray, ArrayOperation<TestArray>>::new(),
            |x| {
                let context = x.context().clone();
                let starts = context.constant(TestArray::new(
                    ArrayType::new(DataType::I32, Shape::new(vec![Size::Static(2)])),
                    vec![1.0, 2.0],
                ));
                let stacked: NestedTracer<EagerContext<TestArray, ArrayOperation<TestArray>>> = Batch::batch(
                    &context,
                    |(item, start)| item.dynamic_slice(&[start], &[2]),
                    (x, starts),
                    (BatchAxis::replicated(), BatchAxis::new(0)),
                    BatchAxis::new(0),
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
}
