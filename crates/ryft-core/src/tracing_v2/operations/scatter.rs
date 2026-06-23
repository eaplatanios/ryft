//! Differentiation rule for [`ScatterOperation`]. Scatter-add ([`ScatterReductionKind::Add`]) is jointly linear in its
//! operand and updates — the integer index operand has no tangent space — so its JVP scatter-adds the operand and
//! update tangents at the same indices, captured as a residual factor, via the captured-index linear scatter-add form
//! ([`LinearScatterAddOperation`]). That linear form's transpose is the scatter-add/gather duality, implemented on
//! [`LinearArrayOperation`](crate::tracing_v2::operations::primitive::LinearArrayOperation). The other combiner kinds
//! (`Overwrite`/`Mul`/`Min`/`Max`) are not linear; differentiating them is not yet implemented (their pullbacks need a
//! primal-domain mask and a `unique_indices` guarantee, matching JAX's restrictions).

use crate::contexts::StagingContext;
use crate::macros::check_count;
use crate::operations::InterpretableOperation;
use crate::operations::constants::{MaybeZeroOperation, ZeroOperation};
use crate::operations::manipulation::{
    Broadcast, LinearScatterAddOperation, Reshape, SCATTER_OPERATION_NAME, Scatter, ScatterOperation,
    ScatterReductionKind, Slice, Transpose, UpdateSlice,
};
use crate::programs::{ProgramError, Value};
use crate::tracing_v2::batching::{ArrayBatch, BatchableOperation, apply_with_axes, batch_input_metadata};
use crate::tracing_v2::differentiation::{JvpTracer, LinearOperationOf, TangentContext};
use crate::tracing_v2::operations::slicing::batch_by_lane_expansion;
use crate::tracing_v2::{DifferentiableOperation, DifferentiationContext, ValueOrCapture};
use crate::types::{ArrayType, Typed};

/// JVP rule for [`ScatterOperation`]. For the [`Add`](ScatterReductionKind::Add) combiner the operation is jointly
/// linear in its operand and updates, so the tangent is a captured-index scatter-add of the operand and update
/// tangents (the [`LinearScatterAddOperation`] form). When both operand tangents are symbolic zeros the output tangent
/// is a symbolic zero. The non-additive combiners are not linear: their JVP is only defined under `unique_indices`
/// with a primal-domain mask and is not yet implemented, so a non-zero tangent through them is rejected.
impl<D> DifferentiableOperation<D> for ScatterOperation
where
    D: DifferentiationContext<Type = ArrayType>,
    D::Value: Scatter,
    LinearOperationOf<D>:
        From<LinearScatterAddOperation<ValueOrCapture<ArrayType, D::Value>>> + From<ZeroOperation<ArrayType>>,
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
        let [operand, indices, updates] = inputs else {
            return Err(ProgramError::InvalidInputCount { expected: 3, actual: inputs.len() });
        };
        let primal = operand.primal().scatter(indices.primal(), updates.primal(), self)?;
        if context.is_zero(operand.tangent())? && context.is_zero(updates.tangent())? {
            let tangent_type = primal.r#type().into_owned();
            let mut tangent_outputs = context.stage_nullary_operation(ZeroOperation::new(tangent_type))?;
            check_count!("output", tangent_outputs, 1, ProgramError);
            return Ok(vec![JvpTracer::new(primal, tangent_outputs.remove(0))]);
        }
        if self.kind() != ScatterReductionKind::Add {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "differentiation of scatter with the {} combiner is not yet implemented (only scatter-add is \
                     linear)",
                    self.kind(),
                ),
            });
        }
        let indices_factor = indices.factor(context);
        let mut outputs = context.stage_operation(
            LinearScatterAddOperation::new(self.clone(), indices_factor),
            &[operand.tangent(), updates.tangent()],
        )?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(vec![JvpTracer::from_value(primal, outputs.remove(0))])
    }
}

/// Batching rule for [`ScatterOperation`]. As with gather, a scatter's window/inserted/index axis bookkeeping does not
/// compose cleanly with an extra mapped axis, so any batched operand, indices, or updates is handled by per-lane
/// expansion ([`batch_by_lane_expansion`]): each lane scatters independently and the results restack along a fresh
/// leading lane axis. This stages `O(axis_size)` scatters but is correct for every combiner and dimension-number
/// configuration; dimension-number lifting is a performance optimization left as a follow-up. When no input is mapped
/// the scatter applies once, unbatched.
impl<V> BatchableOperation<V, V::InterpretationContext> for ScatterOperation
where
    V: Value<ArrayType> + Broadcast + Transpose + Slice + UpdateSlice + Reshape,
    ScatterOperation: InterpretableOperation<ArrayType, V>,
{
    fn batch(
        &self,
        context: &V::InterpretationContext,
        inputs: &[ArrayBatch<V>],
    ) -> Result<Vec<ArrayBatch<V>>, ProgramError> {
        check_count!("input", inputs, 3, ProgramError);
        let (_, input_axes, axis_size) = batch_input_metadata(inputs)?;
        if input_axes.iter().all(Option::is_none) {
            return apply_with_axes(context, self, inputs, &[None]);
        }
        batch_by_lane_expansion(context, SCATTER_OPERATION_NAME, self, inputs, axis_size)
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::contexts::StagingContext;
    use crate::operations::manipulation::{Scatter, ScatterDimensionNumbers, ScatterOperation, ScatterReductionKind};
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
    fn test_scatter_add_value_and_grad_splits_cotangent() {
        // f(x, u) = sum(scatter_add(x, [[1], [3]], u)) adds the two scalar updates into operand positions 1 and 3.
        // Scatter-add is the identity in its operand (`∂output/∂operand = I`), so the operand gradient is the all-ones
        // cotangent unchanged, while the update gradient gathers that cotangent at the captured indices — the
        // scatter-add/gather transpose duality.
        let (value, (operand_gradient, update_gradient)) = value_and_grad(
            &TestArrayDomain,
            |(x, updates)| {
                let indices = index_array(&x, vec![2, 1], vec![1.0, 3.0]);
                let operation = ScatterOperation::new(
                    ScatterDimensionNumbers::new(vec![], vec![0], vec![0]),
                    ScatterReductionKind::Add,
                );
                x.scatter(&indices, &updates, &operation).unwrap().reduce(&[0], ReductionKind::Sum)
            },
            (TestArray::vector(vec![1.0, 2.0, 3.0, 4.0]), TestArray::vector(vec![10.0, 20.0])),
        )
        .unwrap();
        assert_close(value.values[0], 40.0);
        assert_eq!(operand_gradient.values, vec![1.0, 1.0, 1.0, 1.0]);
        assert_eq!(update_gradient.values, vec![1.0, 1.0]);
    }

    #[test]
    fn test_scatter_add_jacfwd_is_identity_in_operand() {
        // Forward mode through `f(x) = scatter_add(x, [[1], [3]], [10, 20])` exercises the captured-index scatter-add
        // under batched basis tangents (the per-lane batch rule). Scatter-add is the identity in its operand, so the
        // Jacobian with respect to `x` is the identity matrix.
        let jacobian = TestArrayDomain
            .jacfwd(
                |x| {
                    let indices = index_array(&x, vec![2, 1], vec![1.0, 3.0]);
                    let updates = x.context().constant(TestArray::vector(vec![10.0, 20.0]));
                    let operation = ScatterOperation::new(
                        ScatterDimensionNumbers::new(vec![], vec![0], vec![0]),
                        ScatterReductionKind::Add,
                    );
                    Ok(x.scatter(&indices, &updates, &operation).unwrap())
                },
                TestArray::vector(vec![1.0, 2.0, 3.0, 4.0]),
            )
            .unwrap();
        let block = jacobian.rows().partials();
        assert_eq!(block.output_shape(), &[4]);
        assert_eq!(block.input_shape(), &[4]);
        assert_eq!(
            block.values(),
            &[
                1.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0, //
                0.0, 0.0, 0.0, 1.0, //
            ],
        );
    }
}
