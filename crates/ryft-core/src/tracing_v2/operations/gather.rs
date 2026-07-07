//! Differentiation rule for [`GatherOperation`](crate::operations::manipulation::GatherOperation). Gather is linear in
//! its data operand — the integer index operand has no tangent space — so its JVP gathers the operand tangent at the
//! same indices, captured as a residual factor, via the captured-index linear gather form
//! ([`LinearGatherOperation`](crate::operations::manipulation::LinearGatherOperation)). That linear form's transpose
//! is the gather/scatter-add duality.

use crate::batching::ArrayBatch;
use crate::batching::BatchAxis;
use crate::batching::BatchableOperation;
use crate::batching::BatchingError;
use crate::batching::InterpretableBatchableOperation;
use crate::contexts::Context;
use crate::interpretation::InterpretableOperation;
use crate::macros::check_count;
use crate::operations::manipulation::{
    Broadcast, GATHER_OPERATION_NAME, Gather, GatherOperation, Reshape, Slice, Transpose, UpdateSlice,
};
use crate::programs::{MaybeZero, ProgramError, Value};

use crate::differentiation::DifferentiationDual;
use crate::tracing_v2::differentiation::DifferentiableOperation;
use crate::tracing_v2::operations::slicing::batch_by_item_expansion;
use crate::types::{ArrayType, Typed};

/// Forward-mode rule for [`GatherOperation`]: `gather` is linear in the data operand, and the index operand is a
/// non-differentiated primal operand edge, so the tangent gathers the operand tangent at the same primal indices. A
/// zero operand tangent yields a typed zero output tangent.
impl<C: Context<Type = ArrayType>> DifferentiableOperation<C> for GatherOperation
where
    C::Operation: Clone + From<GatherOperation>,
    C::Value: Gather,
{
    fn jvp(
        &self,
        _context: &C,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        let indices = inputs[1].primal();
        let primal = inputs[0].primal().gather(indices, self)?;
        let tangent = match inputs[0].tangent() {
            MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().into_owned()),
            MaybeZero::Value(tangent) => MaybeZero::Value(tangent.gather(indices, self)?),
        };
        Ok(vec![DifferentiationDual::new(primal, tangent)])
    }
}

/// Batching rule for [`GatherOperation`]. A gather mixes window reads, collapsed axes, and index-driven offsets whose
/// axis bookkeeping does not compose cleanly with an extra mapped axis, so any batched operand, indices, or both is
/// handled by per-item expansion (`batch_by_item_expansion`): each batch item gathers independently and the results
/// restack along a fresh leading batch axis. This stages `O(axis_size)` gathers but is correct for every
/// dimension-number configuration; dimension-number lifting (one lifted gather, no expansion) is a performance
/// optimization left as a follow-up. When no input is mapped the gather applies once, unbatched.
impl<V, C> BatchableOperation<V, C> for GatherOperation
where
    V: Value<Type = ArrayType> + Broadcast + Transpose + Slice + UpdateSlice + Reshape,
    GatherOperation: InterpretableOperation<V, C>,
{
    fn batch(&self, context: &C, inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, BatchingError> {
        check_count!("input", inputs, 2, ProgramError);
        let Some(axis_size) = ArrayBatch::common_batch_size(inputs)? else {
            return self.interpret_with_batch_axes(context, inputs, &[BatchAxis::replicated()]);
        };
        batch_by_item_expansion(context, GATHER_OPERATION_NAME, self, inputs, axis_size)
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::contexts::EagerContext;
    use crate::contexts::StagingContext;
    use crate::operations::manipulation::{Gather, GatherDimensionNumbers, GatherOperation};
    use crate::tests::TestArray;
    use crate::tracing::Tracer;
    use crate::tracing_v2::ArrayOperation;
    use crate::tracing_v2::operations::reduce::{Reduce, ReductionKind};
    use crate::tracing_v2::test_util::assert_close;
    use crate::tracing_v2::{DifferentiableDomainExtension, value_and_grad};
    use crate::types::{ArrayType, DataType, Shape, Size};

    /// Lifts a constant integer index array into the differentiation trace that `exemplar` belongs to.
    fn index_array<C>(exemplar: &Tracer<C>, shape: Vec<usize>, values: Vec<f64>) -> Tracer<C>
    where
        C: StagingContext<Constant = TestArray>,
    {
        let r#type = ArrayType::new(DataType::I32, Shape::new(shape.into_iter().map(Size::Static).collect()));
        exemplar.context().constant(TestArray::new(r#type, values))
    }

    #[test]
    fn test_gather_value_and_grad_scatters_at_captured_indices() {
        // f(x) = sum(gather(x, [[0], [2]])) takes rows 0 and 2 of a 3x2 matrix; the integer indices are constants of
        // the trace, so the gather/scatter-add transpose duality pulls the all-ones cotangent back into a zero operand
        // at exactly those rows.
        let (value, gradient) = value_and_grad(
            &EagerContext::<TestArray, ArrayOperation<TestArray>>::new(),
            |x| {
                let indices = index_array(&x, vec![2, 1], vec![0.0, 2.0]);
                let operation =
                    GatherOperation::new(GatherDimensionNumbers::new(vec![1], vec![0], vec![0]), vec![1, 2]);
                x.gather(&indices, &operation).unwrap().reduce(&[0, 1], ReductionKind::Sum)
            },
            TestArray::matrix(3, 2, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        )
        .unwrap();
        assert_close(value.values[0], 10.0);
        assert_eq!(gradient.values, vec![1.0, 1.0, 0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn test_gather_jacfwd_selects_operand_coordinates() {
        // Forward mode through `f(x) = gather(x, [[0], [2]])` selects the operand coordinate feeding each output, so the
        // Jacobian is the row-selection indicator from the captured-index linear gather.
        let jacobian = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .jacfwd(
                |x| {
                    let indices = index_array(&x, vec![2, 1], vec![0.0, 2.0]);
                    let operation =
                        GatherOperation::new(GatherDimensionNumbers::new(vec![1], vec![0], vec![0]), vec![1, 2]);
                    Ok(x.gather(&indices, &operation).unwrap())
                },
                TestArray::matrix(3, 2, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
            )
            .unwrap();
        let block = jacobian.rows().partials();
        assert_eq!(block.output_shape(), &[2, 2]);
        assert_eq!(block.input_shape(), &[3, 2]);
        assert_eq!(
            block.values(),
            &[
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, 1.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, 0.0, 1.0, //
            ],
        );
    }
}
