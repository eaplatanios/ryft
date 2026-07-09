use crate::batching::ArrayBatch;
use crate::batching::BatchAxis;
use crate::batching::BatchableOperation;
use crate::batching::BatchingError;
use crate::batching::InterpretableBatchableOperation;
use crate::contexts::{Context, StagingContext};
use crate::differentiation::{DifferentiableOperation, TransposableOperation};
use crate::interpretation::InterpretableOperation;
use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::arithmetic::SubOperation;
use crate::operations::constants::{Zero, ZeroOperation};
use crate::operations::manipulation::{
    Broadcast, Pad, PadOperation, Reshape, Slice, SliceOperation, Transpose, UpdateSlice,
};
use crate::partial::PartialValue;
use crate::programs::{MaybeZero, ProgramError, Value};
use crate::tracing::{Tracer, TracingContext};

use crate::differentiation::DifferentiationDual;
use crate::tracing_v2::operations::reduce::{ReduceOperation, ReductionKind};
use crate::types::{ArrayType, TypeError, Typed};

use super::slicing::batch_by_item_expansion;

/// Transpose (vector-Jacobian product) for a [`PadOperation`].
///
/// The forward map `(t, p) ↦ pad(t, p, low, high, interior)` writes input element `i` to output position
/// `low + i * (interior + 1)` along each axis and the padding value everywhere else, so its pullback splits the
/// output cotangent into two contributions:
///
///   - **Input cotangent**: the strided slice of the cotangent at the pad geometry — `start = low`,
///     `stride = interior + 1`, and `limit = low + (d - 1) * (interior + 1) + 1` for input dimension `d > 0`
///     (`limit = low` for `d == 0`, an empty slice) — which reads back exactly the positions the forward map wrote
///     input elements to. For example, padding `d = 3` elements with `low = 1`, `high = 2`, and `interior = 1`
///     produces an output of dimension `1 + (3 - 1) * 2 + 1 + 2 = 8` whose positions `1`, `3`, and `5` hold the
///     input elements, and the pullback slices the cotangent with `start = 1`, `limit = 6`, and `stride = 2`,
///     reading positions `1`, `3`, and `5`.
///   - **Padding-value cotangent**: the sum of the cotangent over every *padding* position, computed as the full
///     sum of the cotangent minus the sum of the strided-slice region (two staged full reductions and a
///     subtraction, which avoids materializing a mask). When the input has no elements (some dimension is `0`),
///     the sliced region is empty and its sum is a staged scalar zero.
///
/// Symbolic-zero cotangents propagate unchanged.
impl<V: Value<Type = ArrayType>, O> TransposableOperation<V, O> for PadOperation
where
    O: Operation<ArrayType>
        + From<SliceOperation>
        + From<ReduceOperation>
        + From<SubOperation>
        + From<ZeroOperation<ArrayType>>,
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
                let input_type = inputs[0].r#type();
                let rank = input_type.rank();
                let mut start_indices = Vec::with_capacity(rank);
                let mut limit_indices = Vec::with_capacity(rank);
                let mut strides = Vec::with_capacity(rank);
                let mut input_is_empty = false;
                for axis in 0..rank {
                    let dimension = input_type.dimension(axis as isize);
                    let Some(input_size) = dimension.value() else {
                        return Err(TypeError {
                            message: format!(
                                "'pad' transpose requires a static input shape but axis {axis} has size {dimension}",
                            ),
                        }
                        .into());
                    };
                    let low = self.edge_padding_low()[axis];
                    let stride = self.interior_padding()[axis] + 1;
                    let limit = match input_size {
                        0 => low,
                        size => low + (size - 1) * stride + 1,
                    };
                    input_is_empty |= input_size == 0;
                    start_indices.push(low);
                    limit_indices.push(limit);
                    strides.push(stride);
                }
                let input_cotangents = context.stage_operation(
                    SliceOperation::new(start_indices, limit_indices).with_strides(strides)?,
                    std::slice::from_ref(cotangent),
                )?;
                check_count!("output", input_cotangents, 1, ProgramError);
                let input_cotangent = input_cotangents.into_iter().next().unwrap();
                let all_axes: Vec<usize> = (0..cotangent.r#type().as_ref().rank()).collect();
                let total_sums = context.stage_operation(
                    ReduceOperation::new(all_axes.clone(), ReductionKind::Sum),
                    std::slice::from_ref(cotangent),
                )?;
                check_count!("output", total_sums, 1, ProgramError);
                let sliced_sum = if input_is_empty {
                    // The strided slice covered no positions, so its sum is a scalar zero of the padding value's
                    // type.
                    MaybeZero::Zero(inputs[1].r#type().into_owned()).materialize(context)?
                } else {
                    let sliced_sums = context.stage_operation(
                        ReduceOperation::new(all_axes, ReductionKind::Sum),
                        std::slice::from_ref(&input_cotangent),
                    )?;
                    check_count!("output", sliced_sums, 1, ProgramError);
                    sliced_sums.into_iter().next().unwrap()
                };
                let padding_value_cotangents = context
                    .stage_operation(O::from(SubOperation), &[total_sums.into_iter().next().unwrap(), sliced_sum])?;
                check_count!("output", padding_value_cotangents, 1, ProgramError);
                Ok(vec![
                    MaybeZero::Value(input_cotangent),
                    MaybeZero::Value(padding_value_cotangents.into_iter().next().unwrap()),
                ])
            }
        }
    }
}

/// Forward-mode rule for [`PadOperation`]: `pad` is linear in both the operand and the padding value, so the
/// tangent pads the operand tangent with the padding-value tangent using the same padding amounts.
impl<C: Context<Type = ArrayType> + Zero<C::Value>> DifferentiableOperation<C> for PadOperation
where
    C::Operation: Clone + From<PadOperation>,
    C::Value: Pad,
{
    fn jvp(
        &self,
        context: &C,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        let primal = inputs[0].primal().pad(
            inputs[1].primal(),
            self.edge_padding_low(),
            self.edge_padding_high(),
            self.interior_padding(),
        )?;
        // The pad needs both the operand and padding-value tangents as real values, so materialize the structurally
        // zero side (the shared all-zero fast path already handled the case where both are zero).
        let operand_tangent = inputs[0].tangent().clone().materialize(context)?;
        let padding_tangent = inputs[1].tangent().clone().materialize(context)?;
        let tangent = operand_tangent.pad(
            &padding_tangent,
            self.edge_padding_low(),
            self.edge_padding_high(),
            self.interior_padding(),
        )?;
        Ok(vec![DifferentiationDual::new(primal, tangent)])
    }
}

/// Batching rule for [`PadOperation`].
///
/// A batched input with a replicated padding value keeps its batch axis by padding it with zero amounts: the
/// lifted operation inserts `0` into all three padding vectors at the batch axis position. A batch-varying (batched)
/// padding value cannot ride along structurally — the lifted operation would need a rank-1 padding operand, which
/// the operation cannot represent — so the rule falls back to per-item expansion via `batch_by_item_expansion`:
/// the batch size is static, so each batch item's input and padding value are extracted, padded independently, and
/// restacked along a fresh leading batch axis (`O(batch)` staged operations, the same trade the batch-varying
/// dynamic-slice start-index rules make). This keeps the direct batched JVP path (dense forward Jacobians and
/// batched pullbacks) total even though the padding-value tangent is represented as a per-item batch there.
impl<V, C> BatchableOperation<V, C> for PadOperation
where
    V: Value<Type = ArrayType> + Broadcast + Transpose + Slice + UpdateSlice + Reshape,
    PadOperation: InterpretableOperation<V, C>,
{
    fn batch(&self, context: &C, inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, BatchingError> {
        check_count!("input", inputs, 2, ProgramError);
        let batch_axes: Vec<Option<usize>> = inputs.iter().map(|input| input.batch_axis().axis()).collect();
        let axis_size = ArrayBatch::common_batch_size(inputs)?;
        if batch_axes[1].is_none() {
            let Some(batch_axis) = batch_axes[0] else {
                return self.interpret_with_batch_axes(context, inputs, &[BatchAxis::replicated()]);
            };
            let mut edge_padding_low = self.edge_padding_low().to_vec();
            edge_padding_low.insert(batch_axis, 0);
            let mut edge_padding_high = self.edge_padding_high().to_vec();
            edge_padding_high.insert(batch_axis, 0);
            let mut interior_padding = self.interior_padding().to_vec();
            interior_padding.insert(batch_axis, 0);
            let lifted = PadOperation::new(edge_padding_low, edge_padding_high, interior_padding)?;
            return lifted.interpret_with_batch_axes(context, inputs, &[BatchAxis::new(batch_axis)]);
        }
        // Batch-varying padding value: pad each batch item independently and restack along a fresh leading batch axis.
        let axis_size = axis_size.expect("a mapped input pins the batch size");
        batch_by_item_expansion(context, crate::operations::manipulation::PAD_OPERATION_NAME, self, inputs, axis_size)
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
    use crate::tracing_v2::{DifferentiableDomainExtension, value_and_gradient};

    use super::*;
    use crate::batching::BatchAxis;

    #[test]
    fn test_pad_value_and_grad_splits_cotangent() {
        // f(x, p) = sum(pad(x, p, low=[1], high=[2], interior=[1]) * w) with w = [1..8]: the padded output is
        // [p, x0, p, x1, p, x2, p, p], so f = 2*x0 + 4*x1 + 6*x2 + (1 + 3 + 5 + 7 + 8)*p. The input gradient is the
        // strided slice of the weighted cotangent at the pad geometry (positions 1, 3, and 5 of w) and the
        // padding-value gradient is the sum over the padding positions, computed as sum(w) - sum(sliced w) =
        // 36 - 12 = 24.
        let (value, (input_gradient, padding_value_gradient)) = value_and_gradient(
            &EagerContext::<TestArray, ArrayOperation<TestArray>>::new(),
            |(x, padding_value)| {
                let weights =
                    x.context().lift(TestArray::vector(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])).unwrap();
                (x.pad(&padding_value, &[1], &[2], &[1]).unwrap() * weights).reduce(&[0], ReductionKind::Sum)
            },
            (TestArray::vector(vec![1.0, 2.0, 3.0]), TestArray::scalar(9.0)),
        )
        .unwrap();
        // f = 2 * 1 + 4 * 2 + 6 * 3 + 24 * 9 = 28 + 216.
        assert_close(value.values[0], 244.0);
        assert_eq!(input_gradient.values, vec![2.0, 4.0, 6.0]);
        assert_eq!(padding_value_gradient.values, vec![24.0]);
    }

    #[test]
    fn test_pad_jacfwd_scatters_input_coordinates() {
        // Forward mode through `f(x) = pad(x, 0, low=[1], high=[2], interior=[1])` produces the 8x3 scatter
        // Jacobian: output positions 1, 3, and 5 hold the input coordinates.
        let jacobian = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .jacfwd(
                |x| {
                    let padding_value = x.context().lift(TestArray::scalar(0.0))?;
                    x.pad(&padding_value, &[1], &[2], &[1])
                },
                TestArray::vector(vec![1.0, 2.0, 3.0]),
            )
            .unwrap();
        let block = jacobian.rows().partials();
        assert_eq!(block.output_shape(), &[8]);
        assert_eq!(block.input_shape(), &[3]);
        assert_eq!(
            block.values(),
            &[
                0.0, 0.0, 0.0, //
                1.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, //
                0.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, //
                0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, //
            ],
        );
    }

    #[test]
    fn test_pad_batching_lifts_batch_axis_with_zero_paddings() {
        // A batched input keeps its batch axis by padding it with zero amounts: each batch item pads independently.
        let input = {
            let value = TestArray::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
            ArrayBatch::new(value.r#type().into_owned(), value, Some(0))
        }
        .unwrap();
        let padding_value = ArrayBatch::replicated(TestArray::scalar(0.0));
        let operation = PadOperation::new(vec![1], vec![0], vec![0]).unwrap();
        let outputs =
            operation.batch(&crate::EagerContext::<TestArray>::new(), &[input.clone(), padding_value]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value().values, vec![0.0, 1.0, 2.0, 0.0, 3.0, 4.0]);

        // Replicated operands pass through the unlifted rule.
        let uniform = ArrayBatch::replicated(TestArray::vector(vec![1.0, 2.0]));
        let outputs = operation
            .batch(&crate::EagerContext::<TestArray>::new(), &[uniform, ArrayBatch::replicated(TestArray::scalar(0.0))])
            .unwrap();
        assert_eq!(outputs[0].batch_axis(), BatchAxis::replicated());
        assert_eq!(outputs[0].value().values, vec![0.0, 1.0, 2.0]);

        // Batch-varying padding values expand per item: item 0 pads with 8 and item 1 pads with 9.
        let batch_varying = {
            let value = TestArray::vector(vec![8.0, 9.0]);
            ArrayBatch::new(value.r#type().into_owned(), value, Some(0))
        }
        .unwrap();
        let outputs =
            operation.batch(&crate::EagerContext::<TestArray>::new(), &[input, batch_varying.clone()]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value().values, vec![8.0, 1.0, 2.0, 9.0, 3.0, 4.0]);

        // A replicated input is broadcast to gain the batch axis when only the padding value is batched.
        let uniform_input = ArrayBatch::replicated(TestArray::vector(vec![1.0, 2.0]));
        let outputs =
            operation.batch(&crate::EagerContext::<TestArray>::new(), &[uniform_input, batch_varying]).unwrap();
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value().values, vec![8.0, 1.0, 2.0, 9.0, 1.0, 2.0]);
    }
}
