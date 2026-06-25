//! Differentiation rule for [`GatherOperation`]. Gather is linear in its data operand — the integer index operand has
//! no tangent space — so its JVP gathers the operand tangent at the same indices, captured as a residual factor, via
//! the captured-index linear gather form ([`LinearGatherOperation`]). That linear form's transpose is the
//! gather/scatter-add duality, implemented on
//! [`LinearArrayOperation`](crate::tracing_v2::operations::primitive::LinearArrayOperation).

use crate::contexts::StagingContext;
use crate::macros::check_count;
use crate::operations::InterpretableOperation;
use crate::operations::constants::{MaybeZeroOperation, ZeroOperation};
use crate::operations::manipulation::{
    Broadcast, GATHER_OPERATION_NAME, Gather, GatherOperation, LinearGatherOperation, Reshape, Slice, Transpose,
    UpdateSlice,
};
use crate::programs::{ProgramError, Value};
use crate::tracing_v2::batching::{ArrayBatch, BatchableOperation, apply_with_axes, batch_input_metadata};
use crate::tracing_v2::differentiation::{JvpTracer, TangentContext};
use crate::tracing_v2::operations::slicing::batch_by_lane_expansion;
use crate::tracing_v2::{DifferentiableOperation, DifferentiationContext, ValueOrCapture};
use crate::types::{ArrayType, Typed};

/// JVP rule for [`GatherOperation`]: the primal output is the gather of the operand primal at the index primals, and
/// the tangent is a captured-index gather of the operand tangent whose indices are the index primals captured as a
/// residual factor (the [`LinearGatherOperation`] form). A symbolic-zero operand tangent yields a symbolic-zero output.
impl<D> DifferentiableOperation<D> for GatherOperation
where
    D: DifferentiationContext<Type = ArrayType>,
    D::Value: Gather,
    D::LinearOperation<D::Tangent, ValueOrCapture<D::Type, D::Value>>:
        From<LinearGatherOperation<ValueOrCapture<ArrayType, D::Value>>> + From<ZeroOperation<ArrayType>>,
    D::LinearOperation<D::Tangent, ValueOrCapture<D::Type, D::Value>>: MaybeZeroOperation<ArrayType>,
{
    fn jvp<'jvp>(
        &self,
        context: &mut TangentContext<'jvp, D>,
        inputs: &[JvpTracer<'jvp, D>],
    ) -> Result<Vec<JvpTracer<'jvp, D>>, ProgramError>
    where
        D: 'jvp,
    {
        let [operand, indices] = inputs else {
            return Err(ProgramError::InvalidInputCount { expected: 2, actual: inputs.len() });
        };
        let primal = operand.primal().gather(indices.primal(), self)?;
        if context.is_zero(operand.tangent())? {
            let tangent_type = primal.r#type().into_owned();
            let mut tangent_outputs = context.stage_nullary_operation(ZeroOperation::new(tangent_type))?;
            check_count!("output", tangent_outputs, 1, ProgramError);
            return Ok(vec![JvpTracer::new(primal, tangent_outputs.remove(0))]);
        }
        let indices_factor = indices.factor(context);
        let mut outputs =
            context.stage_operation(LinearGatherOperation::new(self.clone(), indices_factor), &[operand.tangent()])?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(vec![JvpTracer::from_value(primal, outputs.remove(0))])
    }
}

/// Batching rule for [`GatherOperation`]. A gather mixes window reads, collapsed axes, and index-driven offsets whose
/// axis bookkeeping does not compose cleanly with an extra mapped axis, so any batched operand, indices, or both is
/// handled by per-lane expansion ([`batch_by_lane_expansion`]): each lane gathers independently and the results restack
/// along a fresh leading lane axis. This stages `O(axis_size)` gathers but is correct for every dimension-number
/// configuration; dimension-number lifting (one lifted gather, no expansion) is a performance optimization left as a
/// follow-up. When no input is mapped the gather applies once, unbatched.
impl<V> BatchableOperation<V, V::InterpretationContext> for GatherOperation
where
    V: Value<ArrayType> + Broadcast + Transpose + Slice + UpdateSlice + Reshape,
    GatherOperation: InterpretableOperation<ArrayType, V>,
{
    fn batch(
        &self,
        context: &V::InterpretationContext,
        inputs: &[ArrayBatch<V>],
    ) -> Result<Vec<ArrayBatch<V>>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        let (_, input_axes, axis_size) = batch_input_metadata(inputs)?;
        if input_axes.iter().all(Option::is_none) {
            return apply_with_axes(context, self, inputs, &[None]);
        }
        batch_by_lane_expansion(context, GATHER_OPERATION_NAME, self, inputs, axis_size)
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::contexts::StagingContext;
    use crate::operations::manipulation::{Gather, GatherDimensionNumbers, GatherOperation};
    use crate::tests::{TestArray, TestArrayDomain};
    use crate::tracing::Tracer;
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
            &TestArrayDomain,
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
        let jacobian = TestArrayDomain
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
