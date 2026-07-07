//! Differentiation rule for [`ScatterOperation`](crate::operations::manipulation::ScatterOperation). Scatter-add
//! ([`ScatterReductionKind::Add`](crate::operations::manipulation::ScatterReductionKind::Add)) is jointly linear in
//! its operand and updates — the integer index operand has no tangent space — so its JVP scatter-adds the operand and
//! update tangents at the same indices, captured as a residual factor, via the captured-index linear scatter-add form
//! ([`LinearScatterAddOperation`](crate::operations::manipulation::LinearScatterAddOperation)). That linear form's
//! transpose is the scatter-add/gather duality. The other combiner
//! kinds (`Overwrite`/`Mul`/`Min`/`Max`) are not linear; differentiating them is not yet implemented (their pullbacks
//! need a primal-domain mask and a `unique_indices` guarantee, matching JAX's restrictions).

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
use crate::operations::constants::{Zero, ZeroOperation};
use crate::operations::manipulation::{
    Broadcast, GatherDimensionNumbers, GatherOperation, Reshape, SCATTER_OPERATION_NAME, Scatter, ScatterOperation,
    ScatterReductionKind, Slice, Transpose, UpdateSlice,
};
use crate::partial::PartialValue;
use crate::programs::{MaybeZero, ProgramError, Value};
use crate::tracing::{Tracer, TracingContext};

use crate::tracing_v2::differentiation::{DifferentiableOperation, JvpTracer, materialize};
use crate::tracing_v2::operations::slicing::batch_by_item_expansion;
use crate::types::{ArrayType, TypeError, Typed};

/// Forward-mode rule for [`ScatterOperation`]. For the [`Add`](ScatterReductionKind::Add) combiner the operation
/// is jointly linear in its operand and updates, while the integer index operand is a non-differentiated primal operand
/// edge, so the tangent scatter-adds the operand and update tangents at the same primal indices. A zero operand and
/// update tangent yields a typed zero output tangent. The non-additive combiners are not linear and their tangent
/// is not synthesized, so a non-zero tangent through them is rejected with
/// [`UnsupportedOperation`](ProgramError::UnsupportedOperation).
impl<C: Context<Type = ArrayType> + Zero<C::Value>> DifferentiableOperation<C> for ScatterOperation
where
    C::Operation: Clone + From<ScatterOperation>,
    C::Value: Scatter,
{
    fn jvp(&self, context: &C, inputs: &[JvpTracer<C>]) -> Result<Vec<JvpTracer<C>>, ProgramError> {
        check_count!("input", inputs, 3, ProgramError);
        let operand = &inputs[0];
        let indices = inputs[1].primal();
        let updates = &inputs[2];
        let primal = operand.primal().scatter(indices, updates.primal(), self)?;
        let tangent = if operand.tangent().is_zero() && updates.tangent().is_zero() {
            MaybeZero::Zero(primal.r#type().into_owned())
        } else if self.kind() != ScatterReductionKind::Add {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "differentiation of scatter with the {} combiner is not yet implemented (only scatter-add is \
                     linear)",
                    self.kind(),
                ),
            });
        } else {
            // One of the two linear tangents may still be a structural zero; scatter-add needs both as real values,
            // so materialize the zero side before staging the tangent scatter.
            let operand_tangent = materialize(context, operand.tangent().clone())?;
            let updates_tangent = materialize(context, updates.tangent().clone())?;
            MaybeZero::Value(operand_tangent.scatter(indices, &updates_tangent, self)?)
        };
        Ok(vec![JvpTracer::new(primal, tangent)])
    }
}

/// Partition-aware transpose rule for the primal [`ScatterOperation`] with an [`Add`](ScatterReductionKind::Add)
/// combiner. The integer index operand (operand 1) has no tangent space, so in a valid pushforward it is the known
/// operand while the scattered operand (operand 0) and the updates (operand 2) are the linear ones. Scatter-add
/// accumulates into its operand (`output = operand + scattered(updates)`, so the operand Jacobian is the identity), so
/// the operand cotangent is the output cotangent unchanged; the update cotangent gathers the output cotangent at the
/// scattered windows via the dual gather built by mirroring the scatter geometry. This reproduces the captured-index
/// [`LinearScatterAddOperation`](crate::operations::manipulation::LinearScatterAddOperation) transpose rule, reading
/// the indices from the pullback through `operand_values` and staging a primal [`GatherOperation`] instead of folding
/// the indices into a captured factor. The indices receive a structural zero, and a zero output cotangent stays a
/// structural zero. Non-additive combiners are not linear and are
/// rejected.
impl<V: Value<Type = ArrayType>, O> TransposableOperation<V, O> for ScatterOperation
where
    O: Operation<ArrayType> + From<ZeroOperation<ArrayType>> + From<GatherOperation>,
{
    fn transpose(
        &self,
        context: &mut TracingContext<V, O>,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, ProgramError> {
        check_count!("input", inputs, 3, ProgramError);
        check_count!("output", outputs, 1, ProgramError);
        if self.kind() != ScatterReductionKind::Add {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "transposition of scatter with the {} combiner is not yet implemented (only scatter-add is linear)",
                    self.kind(),
                ),
            });
        }
        match &outputs[0] {
            MaybeZero::Zero(_) => Ok(inputs.iter().map(|input| MaybeZero::Zero(input.r#type().into_owned())).collect()),
            MaybeZero::Value(cotangent) => {
                // The indices are the known operand; the dispatch guarantees a `Known` operand carries its pullback
                // value, so read the tracer directly.
                let indices = inputs[1]
                    .as_known()
                    .expect("dispatch guarantees a known operand carries its pullback value")
                    .clone();
                // Build the dual gather by mirroring the scatter geometry: the slice sizes pair each operand window
                // axis with its update window extent, with size 1 at the inserted and batching axes.
                let dimensions = self.dimensions();
                let updates_type = inputs[2].r#type();
                let operand_rank = inputs[0].r#type().rank();
                let update_window_dimensions = dimensions.update_window_dimensions();
                let inserted_window_dimensions = dimensions.inserted_window_dimensions();
                let operand_batching_dimensions = dimensions.operand_batching_dimensions();
                let mut slice_sizes = Vec::with_capacity(operand_rank);
                let mut window_position = 0;
                for operand_axis in 0..operand_rank {
                    if inserted_window_dimensions.contains(&operand_axis)
                        || operand_batching_dimensions.contains(&operand_axis)
                    {
                        slice_sizes.push(1);
                    } else {
                        let update_axis = update_window_dimensions[window_position];
                        let extent = updates_type.dimension(update_axis as isize).value().ok_or_else(|| {
                            ProgramError::from(TypeError {
                                message: format!(
                                    "'{SCATTER_OPERATION_NAME}' transpose requires a static update shape but update axis \
                                     {update_axis} has a dynamic size",
                                ),
                            })
                        })?;
                        slice_sizes.push(extent);
                        window_position += 1;
                    }
                }
                let gather_dimensions = GatherDimensionNumbers::new(
                    update_window_dimensions.to_vec(),
                    inserted_window_dimensions.to_vec(),
                    dimensions.scatter_dimensions_to_operand_dimensions().to_vec(),
                )
                .with_batching_dimensions(
                    operand_batching_dimensions.to_vec(),
                    dimensions.scatter_indices_batching_dimensions().to_vec(),
                );
                let gather_operation = GatherOperation::new(gather_dimensions, slice_sizes)
                    .with_mode(self.mode())
                    .with_indices_are_sorted(self.indices_are_sorted())
                    .with_unique_indices(self.unique_indices())
                    .with_output_sharding(updates_type.sharding().cloned());
                let update_cotangents = context.stage_operation(gather_operation, &[cotangent.clone(), indices])?;
                check_count!("output", update_cotangents, 1, ProgramError);
                Ok(vec![
                    MaybeZero::Value(cotangent.clone()),
                    MaybeZero::Zero(inputs[1].r#type().into_owned()),
                    MaybeZero::Value(update_cotangents.into_iter().next().unwrap()),
                ])
            }
        }
    }
}

/// Batching rule for [`ScatterOperation`]. As with gather, a scatter's window/inserted/index axis bookkeeping does not
/// compose cleanly with an extra mapped axis, so any batched operand, indices, or updates is handled by per-item
/// expansion (`batch_by_item_expansion`): each batch item scatters independently and the results restack along a fresh
/// leading batch axis. This stages `O(axis_size)` scatters but is correct for every combiner and dimension-number
/// configuration; dimension-number lifting is a performance optimization left as a follow-up. When no input is mapped
/// the scatter applies once, unbatched.
impl<V, C> BatchableOperation<V, C> for ScatterOperation
where
    V: Value<Type = ArrayType> + Broadcast + Transpose + Slice + UpdateSlice + Reshape,
    ScatterOperation: InterpretableOperation<V, C>,
{
    fn batch(&self, context: &C, inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, BatchingError> {
        check_count!("input", inputs, 3, ProgramError);
        let Some(axis_size) = ArrayBatch::common_batch_size(inputs)? else {
            return self.interpret_with_batch_axes(context, inputs, &[BatchAxis::replicated()]);
        };
        batch_by_item_expansion(context, SCATTER_OPERATION_NAME, self, inputs, axis_size)
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::contexts::EagerContext;
    use crate::contexts::StagingContext;
    use crate::operations::manipulation::{Scatter, ScatterDimensionNumbers, ScatterOperation, ScatterReductionKind};
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
    fn test_scatter_add_value_and_grad_splits_cotangent() {
        // f(x, u) = sum(scatter_add(x, [[1], [3]], u)) adds the two scalar updates into operand positions 1 and 3.
        // Scatter-add is the identity in its operand (`∂output/∂operand = I`), so the operand gradient is the all-ones
        // cotangent unchanged, while the update gradient gathers that cotangent at the captured indices — the
        // scatter-add/gather transpose duality.
        let (value, (operand_gradient, update_gradient)) = value_and_grad(
            &EagerContext::<TestArray, ArrayOperation<TestArray>>::new(),
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
        // under batched basis tangents (the per-item batch rule). Scatter-add is the identity in its operand, so the
        // Jacobian with respect to `x` is the identity matrix.
        let jacobian = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
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
