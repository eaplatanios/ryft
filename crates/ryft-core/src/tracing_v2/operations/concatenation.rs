use crate::contexts::StagingContext;
use crate::differentiation::{Cotangent, TransposableOperation};
use crate::macros::check_count;
use crate::operations::constants::{MaybeZeroOperation, ZeroOperation};
use crate::operations::manipulation::{Broadcast, Concatenate, ConcatenateOperation, SliceOperation, Transpose};
use crate::operations::{InterpretableOperation, Operation};
use crate::programs::{ProgramError, Value};
use crate::tracing::AbstractTracingContext;
use crate::tracing_v2::batching::{ArrayBatch, BatchableOperation, batch_input_metadata};
use crate::tracing_v2::differentiation::{JvpTracer, LinearOperationOf, TangentContext};
use crate::tracing_v2::{DifferentiableOperation, DifferentiationContext};
use crate::types::{ArrayType, Size, TypeError, Typed};

use super::slicing::materialize_lane_axis;

/// Transpose (vector-Jacobian product) for a [`ConcatenateOperation`].
///
/// The forward map `(t_0, ..., t_n) ↦ concatenate([t_0, ..., t_n], axis)` lays the operands end to end along `axis`,
/// so its pullback splits the output cotangent back into the per-operand pieces by slicing the cotangent at the
/// cumulative operand offsets along `axis`: operand `i` receives `slice(cotangent, start, limit, unit strides)` with
/// `start[axis]` and `limit[axis]` set to that operand's `[offset, offset + operand_axis_size)` window and the full
/// extent on every other axis. The operands must have a static size along `axis` so the offsets are known.
/// Symbolic-zero cotangents propagate unchanged to every operand.
impl<V: Value<ArrayType>, O> TransposableOperation<ArrayType, V, O> for ConcatenateOperation
where
    O: Operation<ArrayType> + From<SliceOperation>,
{
    fn transpose<'transpose>(
        &self,
        context: &mut AbstractTracingContext<'transpose, ArrayType, V, O>,
        input_types: &[&ArrayType],
        output_cotangents: &[Cotangent<'transpose, ArrayType, V, O>],
    ) -> Result<Vec<Cotangent<'transpose, ArrayType, V, O>>, ProgramError> {
        check_count!("output", output_cotangents, 1, ProgramError);
        if input_types.is_empty() {
            return Err(TypeError {
                message: "concatenate transpose expects at least one operand but got none".to_string(),
            }
            .into());
        }
        let axis = self.axis();
        match &output_cotangents[0] {
            Cotangent::Zero => Ok(vec![Cotangent::Zero; input_types.len()]),
            Cotangent::Staged(cotangent) => {
                let rank = input_types[0].rank();
                let mut offset = 0usize;
                let mut input_cotangents = Vec::with_capacity(input_types.len());
                for (index, input_type) in input_types.iter().enumerate() {
                    let dimension = input_type.dimension(axis as isize);
                    let Size::Static(operand_axis_size) = dimension else {
                        return Err(TypeError {
                            message: format!(
                                "concatenate transpose requires a static size along the concatenated axis {axis} \
                                but operand {index} has size {dimension}",
                            ),
                        }
                        .into());
                    };
                    let mut start_indices = vec![0usize; rank];
                    let mut limit_indices = input_type
                        .shape()
                        .dimensions()
                        .iter()
                        .enumerate()
                        .map(|(other_axis, size)| {
                            size.value().ok_or_else(|| {
                                TypeError {
                                    message: format!(
                                        "concatenate transpose requires a static operand shape but operand {index} \
                                        has size {size} on axis {other_axis}",
                                    ),
                                }
                                .into()
                            })
                        })
                        .collect::<Result<Vec<usize>, ProgramError>>()?;
                    start_indices[axis] = offset;
                    limit_indices[axis] = offset + operand_axis_size;
                    let strides = vec![1; rank];
                    let outputs = context.stage_operation(
                        SliceOperation::new(start_indices, limit_indices).with_strides(strides)?,
                        std::slice::from_ref(cotangent),
                    )?;
                    check_count!("output", outputs, 1, ProgramError);
                    input_cotangents.push(Cotangent::Staged(outputs.into_iter().next().unwrap()));
                    offset += operand_axis_size;
                }
                Ok(input_cotangents)
            }
        }
    }
}

/// JVP rule for [`ConcatenateOperation`]: concatenation is jointly linear in all of its operands, so the tangent is
/// the concatenation of the operand tangents along the same axis. When every operand tangent is a canonical staged
/// zero, the output tangent is a canonical staged zero of the output type and no linear operation is staged.
impl<D> DifferentiableOperation<D> for ConcatenateOperation
where
    D: DifferentiationContext<Type = ArrayType>,
    D::Value: Concatenate,
    LinearOperationOf<D>: From<ConcatenateOperation> + From<ZeroOperation<ArrayType>>,
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
        if inputs.is_empty() {
            return Err(
                TypeError { message: "concatenate expects at least one operand but got none".to_string() }.into()
            );
        }
        let primal_operands = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        let primal = Concatenate::concatenate(&primal_operands, self.axis())?;
        if inputs
            .iter()
            .try_fold(true, |all_zero, input| Ok::<_, ProgramError>(all_zero && context.is_zero(input.tangent())?))?
        {
            let tangent_type = primal.r#type().into_owned();
            let mut tangent_outputs = context.stage_nullary_operation(ZeroOperation::new(tangent_type))?;
            check_count!("output", tangent_outputs, 1, ProgramError);
            return Ok(vec![JvpTracer::new(primal, tangent_outputs.remove(0))]);
        }
        let operand_tangent_references = inputs.iter().map(JvpTracer::tangent).collect::<Vec<_>>();
        let mut outputs =
            context.stage_operation(ConcatenateOperation::new(self.axis()), operand_tangent_references.as_slice())?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(vec![JvpTracer::from_value(primal, outputs.remove(0))])
    }
}

/// Batching rule for [`ConcatenateOperation`].
///
/// All operands are aligned on one physical lane axis (lane-uniform operands are broadcast to gain it via
/// [`materialize_lane_axis`], so each lane concatenates its own per-lane operands), and the concatenated axis is
/// shifted past the inserted lane axis when the lane axis sits at or before it. When no operand is batched, the
/// operation passes through unchanged.
impl<V: Value<ArrayType> + Broadcast + Transpose> BatchableOperation<V, V::InterpretationContext>
    for ConcatenateOperation
where
    ConcatenateOperation: InterpretableOperation<ArrayType, V>,
{
    fn batch(
        &self,
        context: &V::InterpretationContext,
        inputs: &[ArrayBatch<V>],
    ) -> Result<Vec<ArrayBatch<V>>, ProgramError> {
        if inputs.is_empty() {
            return Err(
                TypeError { message: "concatenate expects at least one operand but got none".to_string() }.into()
            );
        }
        let (_, input_axes, axis_size) = batch_input_metadata(inputs)?;
        let Some(batch_axis) = input_axes.iter().copied().flatten().next() else {
            return crate::tracing_v2::batching::apply_with_axes(context, self, inputs, &[None]);
        };
        let materialized = inputs
            .iter()
            .map(|input| materialize_lane_axis(input, batch_axis, axis_size))
            .collect::<Result<Vec<_>, _>>()?;
        let lifted_axis = if batch_axis <= self.axis() { self.axis() + 1 } else { self.axis() };
        crate::tracing_v2::batching::apply_with_axes(
            context,
            &ConcatenateOperation::new(lifted_axis),
            materialized.as_slice(),
            &[Some(batch_axis)],
        )
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::operations::manipulation::Concatenate;
    use crate::tests::{TestArray, TestArrayDomain};
    use crate::tracing_v2::operations::reduce::{Reduce, ReductionKind};
    use crate::tracing_v2::test_util::assert_close;
    use crate::tracing_v2::{DifferentiableDomainExtension, value_and_grad};

    use super::*;

    #[test]
    fn test_concatenate_value_and_grad_routes_cotangent_per_operand() {
        // f(x, y) = sum(concatenate([x, y], 0) * w) with w = [1, 2, 3, 4, 5]: the joined output is [x0, x1, y0, y1,
        // y2], so f = x0 + 2*x1 + 3*y0 + 4*y1 + 5*y2. The pullback slices the weighted cotangent [1, 2, 3, 4, 5] into
        // the first two entries for x and the last three for y.
        let (value, (x_gradient, y_gradient)) = value_and_grad(
            &TestArrayDomain,
            |(x, y)| {
                let weights = x.context().constant(TestArray::vector(vec![1.0, 2.0, 3.0, 4.0, 5.0]));
                (Concatenate::concatenate(&[x, y], 0).unwrap() * weights).reduce(&[0], ReductionKind::Sum)
            },
            (TestArray::vector(vec![1.0, 2.0]), TestArray::vector(vec![3.0, 4.0, 5.0])),
        )
        .unwrap();
        // f = 1 + 4 + 9 + 16 + 25 = 55.
        assert_close(value.values[0], 55.0);
        assert_eq!(x_gradient.values, vec![1.0, 2.0]);
        assert_eq!(y_gradient.values, vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_concatenate_jacfwd_stacks_operand_coordinates() {
        // Forward mode through `f(x, y) = concatenate([x, y], 0)` over `x = [a, b]` and `y = [c]` produces one
        // selection Jacobian block per operand: `x` maps to the first two output rows and `y` to the last.
        let jacobian = TestArrayDomain
            .jacfwd(
                |(x, y)| Concatenate::concatenate(&[x, y], 0),
                (TestArray::vector(vec![1.0, 2.0]), TestArray::vector(vec![3.0])),
            )
            .unwrap();
        let (x_block, y_block) = jacobian.rows().partials();
        assert_eq!(x_block.output_shape(), &[3]);
        assert_eq!(x_block.input_shape(), &[2]);
        // d(output)/d(x): output rows 0 and 1 are x0 and x1; row 2 (from y) is unaffected by x.
        assert_eq!(x_block.values(), &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        assert_eq!(y_block.output_shape(), &[3]);
        assert_eq!(y_block.input_shape(), &[1]);
        // d(output)/d(y): only output row 2 (from y0) depends on y.
        assert_eq!(y_block.values(), &[0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_concatenate_batching_lifts_lane_axis() {
        // Two batched operands keep their lane axis at 0 and concatenate along the shifted axis 1: lane 0 joins
        // [0, 1] with [4, 5] and lane 1 joins [2, 3] with [6, 7].
        let first = ArrayBatch::mapped(TestArray::matrix(2, 2, vec![0.0, 1.0, 2.0, 3.0]), 0).unwrap();
        let second = ArrayBatch::mapped(TestArray::matrix(2, 2, vec![4.0, 5.0, 6.0, 7.0]), 0).unwrap();
        let outputs = ConcatenateOperation::new(0).batch(&crate::EagerContext::new(), &[first, second]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), Some(0));
        assert_eq!(outputs[0].r#type().shape().dimensions(), &[Size::Static(2), Size::Static(4)]);
        assert_eq!(outputs[0].value().values, vec![0.0, 1.0, 4.0, 5.0, 2.0, 3.0, 6.0, 7.0]);

        // A lane-uniform operand is broadcast to gain the lane axis so each lane concatenates the same copy.
        let batched = ArrayBatch::mapped(TestArray::matrix(2, 2, vec![0.0, 1.0, 2.0, 3.0]), 0).unwrap();
        let uniform = ArrayBatch::unbatched(TestArray::vector(vec![8.0, 9.0]));
        let outputs = ConcatenateOperation::new(0).batch(&crate::EagerContext::new(), &[batched, uniform]).unwrap();
        assert_eq!(outputs[0].batch_axis(), Some(0));
        assert_eq!(outputs[0].value().values, vec![0.0, 1.0, 8.0, 9.0, 2.0, 3.0, 8.0, 9.0]);

        // Lane-uniform operands pass through the unlifted rule.
        let left = ArrayBatch::unbatched(TestArray::vector(vec![1.0, 2.0]));
        let right = ArrayBatch::unbatched(TestArray::vector(vec![3.0]));
        let outputs = ConcatenateOperation::new(0).batch(&crate::EagerContext::new(), &[left, right]).unwrap();
        assert_eq!(outputs[0].batch_axis(), None);
        assert_eq!(outputs[0].value().values, vec![1.0, 2.0, 3.0]);
    }
}
