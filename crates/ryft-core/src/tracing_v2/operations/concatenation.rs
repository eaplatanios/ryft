use crate::batching::ArrayBatch;
use crate::batching::BatchAxis;
use crate::batching::BatchableOperation;
use crate::batching::BatchingContext;
use crate::batching::BatchingError;
use crate::batching::InterpretableBatchableOperation;
use crate::contexts::{Context, StagingContext};
use crate::differentiation::{DifferentiableOperation, DifferentiationError, TransposableOperation};
use crate::interpretation::InterpretableOperation;
use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::constants::Zero;
use crate::operations::manipulation::{Broadcast, Concatenate, ConcatenateOperation, SliceOperation, Transpose};
use crate::partial::PartialValue;
use crate::programs::{MaybeZero, ProgramError, Value};
use crate::tracing::{Tracer, TracingContext};

use crate::differentiation::DifferentiationDual;
use crate::types::{ArrayType, Size, TypeError, Typed};

/// Transpose (vector-Jacobian product) for a [`ConcatenateOperation`].
///
/// The forward map `(t_0, ..., t_n) ↦ concatenate([t_0, ..., t_n], axis)` lays the operands end to end along `axis`,
/// so its pullback splits the output cotangent back into the per-operand pieces by slicing the cotangent at the
/// cumulative operand offsets along `axis`: operand `i` receives `slice(cotangent, start, limit, unit strides)` with
/// `start[axis]` and `limit[axis]` set to that operand's `[offset, offset + operand_axis_size)` window and the full
/// extent on every other axis. The operands must have a static size along `axis` so the offsets are known.
/// Symbolic-zero cotangents propagate unchanged to every operand.
impl<V: Value<Type = ArrayType>, O> TransposableOperation<V, O> for ConcatenateOperation
where
    O: Operation<ArrayType> + From<SliceOperation>,
{
    fn transpose(
        &self,
        context: &mut TracingContext<V, O>,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        check_count!("output", outputs, 1, ProgramError);
        if inputs.is_empty() {
            return Err(TypeError {
                message: "'concatenate' transpose expects at least one operand but got none".to_string(),
            }
            .into());
        }
        let axis = self.axis();
        match &outputs[0] {
            MaybeZero::Zero(_) => Ok(inputs.iter().map(|input| MaybeZero::Zero(input.r#type().into_owned())).collect()),
            MaybeZero::Value(cotangent) => {
                let rank = inputs[0].r#type().rank();
                let mut offset = 0usize;
                let mut input_cotangents = Vec::with_capacity(inputs.len());
                for (index, input) in inputs.iter().enumerate() {
                    let input_type = input.r#type();
                    let dimension = input_type.dimension(axis as isize);
                    let Size::Static(operand_axis_size) = dimension else {
                        return Err(TypeError {
                            message: format!(
                                "'concatenate' transpose requires a static size along the concatenated axis {axis} \
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
                                        "'concatenate' transpose requires a static operand shape but operand {index} \
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
                    input_cotangents.push(MaybeZero::Value(outputs.into_iter().next().unwrap()));
                    offset += operand_axis_size;
                }
                Ok(input_cotangents)
            }
        }
    }
}

/// Forward-mode rule for [`ConcatenateOperation`]: `concatenate` is linear in every operand, so the tangent
/// concatenates the operand tangents along the same axis.
impl<C: Context<Type = ArrayType> + Zero<C::Value>> DifferentiableOperation<C> for ConcatenateOperation
where
    C::Operation: Clone + From<ConcatenateOperation>,
    C::Value: Concatenate,
{
    fn jvp(
        &self,
        context: &C,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        let primals = inputs.iter().map(|dual| dual.primal().clone()).collect::<Vec<_>>();
        // The concatenation needs every operand tangent as a real value, so materialize the structurally zero ones
        // (the shared all-zero fast path already handled the case where every operand tangent is zero).
        let tangents = inputs
            .iter()
            .map(|dual| dual.tangent().clone().materialize(context))
            .collect::<Result<Vec<_>, _>>()?;
        let primal = Concatenate::concatenate(&primals, self.axis())?;
        let tangent = Concatenate::concatenate(&tangents, self.axis())?;
        Ok(vec![DifferentiationDual::new(primal, tangent)])
    }
}

/// Batching rule for [`ConcatenateOperation`].
///
/// All operands are aligned on one physical batch axis (replicated operands are broadcast to gain it via
/// [`ArrayBatch::match_axis`](crate::batching::ArrayBatch::match_axis), so each batch item
/// concatenates its own operands), and the concatenated axis is
/// shifted past the inserted batch axis when the batch axis sits at or before it. When no operand is batched, the
/// operation passes through unchanged.
impl<C: Context<Type = ArrayType, Value: Broadcast + Transpose>> BatchableOperation<C> for ConcatenateOperation
where
    ConcatenateOperation: InterpretableOperation<C::Value, C>,
{
    fn batch(
        &self,
        context: &BatchingContext<C>,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        if inputs.is_empty() {
            return Err(
                TypeError { message: "'concatenate' expects at least one operand but got none".to_string() }.into()
            );
        }
        let batch_axes: Vec<Option<usize>> = inputs.iter().map(|input| input.batch_axis_position()).collect();
        let Some(batch_axis) = batch_axes.iter().copied().flatten().next() else {
            return self.interpret_with_batch_axes(context, inputs, &[BatchAxis::replicated()]);
        };
        let axis_size = ArrayBatch::common_batch_size(inputs)?.expect("a mapped input pins the batch size");
        let materialized = inputs
            .iter()
            .map(|input| input.match_axis(batch_axis as isize, axis_size))
            .collect::<Result<Vec<_>, _>>()?;
        let lifted_axis = if batch_axis <= self.axis() { self.axis() + 1 } else { self.axis() };
        ConcatenateOperation::new(lifted_axis).interpret_with_batch_axes(
            context,
            materialized.as_slice(),
            &[BatchAxis::from_position(batch_axis)],
        )
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use pretty_assertions::assert_eq;

    use crate::contexts::EagerContext;
    use crate::operations::manipulation::Concatenate;
    use crate::tests::TestArray;
    use crate::tracing_v2::ArrayOperation;
    use crate::tracing_v2::operations::reduce::{Reduce, ReductionKind};
    use crate::tracing_v2::{DifferentiableDomainExtension, ReverseModeDifferentiate};
    use crate::types::Typed;

    use super::*;
    use crate::batching::BatchAxis;

    #[test]
    fn test_concatenate_value_and_grad_routes_cotangent_per_operand() {
        // f(x, y) = sum(concatenate([x, y], 0) * w) with w = [1, 2, 3, 4, 5]: the joined output is [x0, x1, y0, y1,
        // y2], so f = x0 + 2*x1 + 3*y0 + 4*y1 + 5*y2. The pullback slices the weighted cotangent [1, 2, 3, 4, 5] into
        // the first two entries for x and the last three for y.
        let (value, (x_gradient, y_gradient)) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .value_and_gradient(
                |(x, y)| {
                    let weights = x.context().lift(TestArray::vector(vec![1.0, 2.0, 3.0, 4.0, 5.0])).unwrap();
                    (Concatenate::concatenate(&[x, y], 0).unwrap() * weights).reduce(&[0], ReductionKind::Sum)
                },
                (TestArray::vector(vec![1.0, 2.0]), TestArray::vector(vec![3.0, 4.0, 5.0])),
            )
            .unwrap();
        // f = 1 + 4 + 9 + 16 + 25 = 55.
        assert_abs_diff_eq!(value.values[0], 55.0, epsilon = 1e-9);
        assert_eq!(x_gradient.values, vec![1.0, 2.0]);
        assert_eq!(y_gradient.values, vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_concatenate_jacfwd_stacks_operand_coordinates() {
        // Forward mode through `f(x, y) = concatenate([x, y], 0)` over `x = [a, b]` and `y = [c]` produces one
        // selection Jacobian block per operand: `x` maps to the first two output rows and `y` to the last.
        let jacobian = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .jacfwd(
                |(x, y)| Concatenate::concatenate(&[x, y], 0),
                (TestArray::vector(vec![1.0, 2.0]), TestArray::vector(vec![3.0])),
            )
            .unwrap();
        let (x_block, y_block) = jacobian.rows().partials();
        assert_eq!(x_block.output_shape(), &[3]);
        assert_eq!(x_block.input_shape(), &[2]);
        // d(output)/d(x): output rows 0 and 1 are x0 and x1; row 2 (from y) is unaffected by x.
        assert_eq!(x_block.value().values(), &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        assert_eq!(y_block.output_shape(), &[3]);
        assert_eq!(y_block.input_shape(), &[1]);
        // d(output)/d(y): only output row 2 (from y0) depends on y.
        assert_eq!(y_block.value().values(), &[0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_concatenate_batching_lifts_batch_axis() {
        // Two batched operands keep their batch axis at 0 and concatenate along the shifted axis 1: batch item 0 joins
        // [0, 1] with [4, 5] and batch item 1 joins [2, 3] with [6, 7].
        let first = {
            let value = TestArray::matrix(2, 2, vec![0.0, 1.0, 2.0, 3.0]);
            ArrayBatch::new(value.r#type().into_owned(), value, Some(0))
        }
        .unwrap();
        let second = {
            let value = TestArray::matrix(2, 2, vec![4.0, 5.0, 6.0, 7.0]);
            ArrayBatch::new(value.r#type().into_owned(), value, Some(0))
        }
        .unwrap();
        let outputs = ConcatenateOperation::new(0)
            .batch(&BatchingContext::new(crate::EagerContext::<TestArray>::new(), 2, None), &[first, second])
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].r#type().shape().dimensions(), &[Size::Static(2), Size::Static(4)]);
        assert_eq!(outputs[0].value().values, vec![0.0, 1.0, 4.0, 5.0, 2.0, 3.0, 6.0, 7.0]);

        // A replicated operand is broadcast to gain the batch axis so each batch item concatenates the same copy.
        let batched = {
            let value = TestArray::matrix(2, 2, vec![0.0, 1.0, 2.0, 3.0]);
            ArrayBatch::new(value.r#type().into_owned(), value, Some(0))
        }
        .unwrap();
        let uniform = ArrayBatch::replicated(TestArray::vector(vec![8.0, 9.0]));
        let outputs = ConcatenateOperation::new(0)
            .batch(&BatchingContext::new(crate::EagerContext::<TestArray>::new(), 2, None), &[batched, uniform])
            .unwrap();
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value().values, vec![0.0, 1.0, 8.0, 9.0, 2.0, 3.0, 8.0, 9.0]);

        // Replicated operands pass through the unlifted rule.
        let left = ArrayBatch::replicated(TestArray::vector(vec![1.0, 2.0]));
        let right = ArrayBatch::replicated(TestArray::vector(vec![3.0]));
        let outputs = ConcatenateOperation::new(0)
            .batch(&BatchingContext::new(crate::EagerContext::<TestArray>::new(), 2, None), &[left, right])
            .unwrap();
        assert_eq!(outputs[0].batch_axis(), BatchAxis::replicated());
        assert_eq!(outputs[0].value().values, vec![1.0, 2.0, 3.0]);
    }
}
