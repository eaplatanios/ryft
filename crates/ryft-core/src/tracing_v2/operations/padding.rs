use crate::contexts::StagingContext;
use crate::differentiation::{Cotangent, TransposableOperation};
use crate::macros::check_count;
use crate::operations::arithmetic::SupportsSub;
use crate::operations::constants::SupportsZero;
use crate::operations::manipulation::{
    Broadcast, Pad, PadOperation, Reshape, Slice, SupportsPad, SupportsSlice, Transpose, UpdateSlice,
};
use crate::operations::{InterpretableOperation, Operation};
use crate::programs::{ProgramError, Value};
use crate::tracing::AbstractTracingContext;
use crate::tracing_v2::batching::{ArrayBatch, BatchableOperation, apply_with_axes, batch_input_metadata};
use crate::tracing_v2::differentiation::{JvpTracer, LinearOperationOf, TangentContext};
use crate::tracing_v2::operations::reduce::{ReductionKind, SupportsReduce};
use crate::tracing_v2::{DifferentiableOperation, DifferentiationContext};
use crate::types::{ArrayType, TypeError, Typed};

use super::control_flow::stage_cotangent;
use super::slicing::batch_by_lane_expansion;

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
impl<V: Value<ArrayType>, O> TransposableOperation<ArrayType, V, O> for PadOperation
where
    O: Operation<ArrayType>
        + SupportsSlice<ArrayType>
        + SupportsReduce<ArrayType>
        + SupportsSub<ArrayType>
        + SupportsZero<ArrayType>,
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
                let input_type = input_types[0];
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
                                "pad transpose requires a static input shape but axis {axis} has size {dimension}",
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
                    O::slice_operation(start_indices, limit_indices, strides),
                    std::slice::from_ref(cotangent),
                )?;
                check_count!("output", input_cotangents, 1, ProgramError);
                let input_cotangent = input_cotangents.into_iter().next().unwrap();
                let all_axes: Vec<usize> = (0..cotangent.r#type().as_ref().rank()).collect();
                let total_sums = context.stage_operation(
                    O::reduce_operation(all_axes.clone(), ReductionKind::Sum, None),
                    std::slice::from_ref(cotangent),
                )?;
                check_count!("output", total_sums, 1, ProgramError);
                let sliced_sum = if input_is_empty {
                    // The strided slice covered no positions, so its sum is a scalar zero of the padding value's
                    // type.
                    stage_cotangent(context, &Cotangent::Zero, input_types[1])
                } else {
                    let sliced_sums = context.stage_operation(
                        O::reduce_operation(all_axes, ReductionKind::Sum, None),
                        std::slice::from_ref(&input_cotangent),
                    )?;
                    check_count!("output", sliced_sums, 1, ProgramError);
                    sliced_sums.into_iter().next().unwrap()
                };
                let padding_value_cotangents = context
                    .stage_operation(O::sub_operation(), &[total_sums.into_iter().next().unwrap(), sliced_sum])?;
                check_count!("output", padding_value_cotangents, 1, ProgramError);
                Ok(vec![
                    Cotangent::Staged(input_cotangent),
                    Cotangent::Staged(padding_value_cotangents.into_iter().next().unwrap()),
                ])
            }
        }
    }
}

/// JVP rule for [`PadOperation`]: the operation is jointly linear in its input and padding-value operands, so the
/// tangent is the pad of the input tangent that uses the padding-value tangent as its padding value, at the same
/// padding geometry. When both operand tangents are symbolic
/// [`Tangent::Zero`](crate::differentiation::Tangent::Zero)s, the output tangent is a symbolic zero of the output
/// type and no linear operation is staged.
impl<D> DifferentiableOperation<D> for PadOperation
where
    D: DifferentiationContext<Type = ArrayType>,
    D::Value: Pad,
    LinearOperationOf<D>: SupportsPad<ArrayType>,
{
    fn jvp<'jvp>(
        &self,
        context: &mut TangentContext<'jvp, D>,
        inputs: &[JvpTracer<'jvp, D>],
    ) -> Result<Vec<JvpTracer<'jvp, D>>, ProgramError>
    where
        D: 'jvp,
        LinearOperationOf<D>: SupportsZero<ArrayType>,
    {
        check_count!("input", inputs, 2, ProgramError);
        let input = &inputs[0];
        let padding_value = &inputs[1];
        let primal = input.primal().pad(
            padding_value.primal(),
            self.edge_padding_low(),
            self.edge_padding_high(),
            self.interior_padding(),
        )?;
        if input.tangent().is_zero() && padding_value.tangent().is_zero() {
            let tangent_type = primal.r#type().into_owned();
            return Ok(vec![JvpTracer::from_zero_tangent(primal, tangent_type)]);
        }
        let input_tangent = context.materialize_tangent(input.tangent().clone())?;
        let padding_value_tangent = context.materialize_tangent(padding_value.tangent().clone())?;
        let mut outputs = context.stage_operation(
            LinearOperationOf::<D>::pad_operation(
                self.edge_padding_low().to_vec(),
                self.edge_padding_high().to_vec(),
                self.interior_padding().to_vec(),
            ),
            &[input_tangent, padding_value_tangent],
        )?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(vec![JvpTracer::from_value(primal, outputs.remove(0))])
    }
}

/// Batching rule for [`PadOperation`].
///
/// A batched input with a lane-uniform padding value keeps its lane axis by padding it with zero amounts: the
/// lifted operation inserts `0` into all three padding vectors at the lane axis position. A lane-varying (batched)
/// padding value cannot ride along structurally — the lifted operation would need a rank-1 padding operand, which
/// the operation cannot represent — so the rule falls back to per-lane expansion via [`batch_by_lane_expansion`]:
/// the batch size is static, so each lane's input and padding value are extracted, padded independently, and
/// restacked along a fresh leading lane axis (`O(batch)` staged operations, the same trade the lane-varying
/// dynamic-slice start-index rules make). This keeps the direct batched JVP path (dense forward Jacobians and
/// batched pullbacks) total even though the padding-value tangent is represented as a per-lane batch there.
impl<V, C> BatchableOperation<V, C> for PadOperation
where
    V: Value<ArrayType>
        + Broadcast
        + Transpose
        + Slice
        + UpdateSlice
        + Reshape,
    PadOperation: InterpretableOperation<ArrayType, V>,
{
    fn batch(&self, _context: &C, inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        let (_, input_axes, axis_size) = batch_input_metadata(inputs)?;
        if input_axes[1].is_none() {
            let Some(batch_axis) = input_axes[0] else {
                return apply_with_axes(self, inputs, &[None]);
            };
            let mut edge_padding_low = self.edge_padding_low().to_vec();
            edge_padding_low.insert(batch_axis, 0);
            let mut edge_padding_high = self.edge_padding_high().to_vec();
            edge_padding_high.insert(batch_axis, 0);
            let mut interior_padding = self.interior_padding().to_vec();
            interior_padding.insert(batch_axis, 0);
            let lifted = PadOperation::new(edge_padding_low, edge_padding_high, interior_padding)?;
            return apply_with_axes(&lifted, inputs, &[Some(batch_axis)]);
        }
        // Lane-varying padding value: pad each lane independently and restack along a fresh leading lane axis.
        batch_by_lane_expansion(crate::operations::manipulation::PAD_OPERATION_NAME, self, inputs, axis_size)
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::tests::{TestArray, TestArrayDomain};
    use crate::tracing_v2::operations::reduce::{Reduce, ReductionKind};
    use crate::tracing_v2::test_util::assert_close;
    use crate::tracing_v2::{DifferentiableDomainExtension, value_and_grad};

    use super::*;

    #[test]
    fn test_pad_value_and_grad_splits_cotangent() {
        // f(x, p) = sum(pad(x, p, low=[1], high=[2], interior=[1]) * w) with w = [1..8]: the padded output is
        // [p, x0, p, x1, p, x2, p, p], so f = 2*x0 + 4*x1 + 6*x2 + (1 + 3 + 5 + 7 + 8)*p. The input gradient is the
        // strided slice of the weighted cotangent at the pad geometry (positions 1, 3, and 5 of w) and the
        // padding-value gradient is the sum over the padding positions, computed as sum(w) - sum(sliced w) =
        // 36 - 12 = 24.
        let (value, (input_gradient, padding_value_gradient)) = value_and_grad(
            &TestArrayDomain,
            |(x, padding_value)| {
                let weights = x.context().constant(TestArray::vector(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]));
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
        let jacobian = TestArrayDomain
            .jacfwd(
                |x| {
                    let padding_value = x.context().constant(TestArray::scalar(0.0));
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
    fn test_pad_batching_lifts_lane_axis_with_zero_paddings() {
        // A batched input keeps its lane axis by padding it with zero amounts: each lane pads independently.
        let input = ArrayBatch::mapped(TestArray::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]), 0).unwrap();
        let padding_value = ArrayBatch::unbatched(TestArray::scalar(0.0));
        let operation = PadOperation::new(vec![1], vec![0], vec![0]).unwrap();
        let outputs = operation.batch(&(), &[input.clone(), padding_value]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), Some(0));
        assert_eq!(outputs[0].value().values, vec![0.0, 1.0, 2.0, 0.0, 3.0, 4.0]);

        // Lane-uniform operands pass through the unlifted rule.
        let uniform = ArrayBatch::unbatched(TestArray::vector(vec![1.0, 2.0]));
        let outputs = operation.batch(&(), &[uniform, ArrayBatch::unbatched(TestArray::scalar(0.0))]).unwrap();
        assert_eq!(outputs[0].batch_axis(), None);
        assert_eq!(outputs[0].value().values, vec![0.0, 1.0, 2.0]);

        // Lane-varying padding values expand per lane: lane 0 pads with 8 and lane 1 pads with 9.
        let lane_varying = ArrayBatch::mapped(TestArray::vector(vec![8.0, 9.0]), 0).unwrap();
        let outputs = operation.batch(&(), &[input, lane_varying.clone()]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), Some(0));
        assert_eq!(outputs[0].value().values, vec![8.0, 1.0, 2.0, 9.0, 3.0, 4.0]);

        // A lane-uniform input is broadcast to gain the lane axis when only the padding value is batched.
        let uniform_input = ArrayBatch::unbatched(TestArray::vector(vec![1.0, 2.0]));
        let outputs = operation.batch(&(), &[uniform_input, lane_varying]).unwrap();
        assert_eq!(outputs[0].batch_axis(), Some(0));
        assert_eq!(outputs[0].value().values, vec![8.0, 1.0, 2.0, 9.0, 1.0, 2.0]);
    }
}
