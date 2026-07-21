use crate::backends::arrays::{Array, ArrayOperation};
use crate::operations::math::MulOperation;
use crate::parameters::Placeholder;
use crate::programs::ProgramBuilder;
use crate::types::{ArrayType, DataType};

/// Builds a single-input flat program that scales its scalar input by `factor`, multiplying the input by a captured
/// constant carrying `factor`.
pub(crate) fn scalar_scale_branch(
    factor: f64,
) -> crate::programs::Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
    let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
    let input = builder.add_input(ArrayType::scalar(DataType::F64));
    let factor = builder.add_constant(Array::scalar(factor));
    let output = builder.add_instruction(MulOperation, Vec::new(), vec![input, factor]).unwrap()[0];
    builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use approx::assert_abs_diff_eq;
    use num_complex::Complex as ComplexNumber;
    use pretty_assertions::assert_eq;

    use crate::backends::scalars::Scalar;
    use crate::batching::{ArrayBatch, BatchAxis, BatchableOperation, BatchingContext, BatchingError, BatchingTracer};
    use crate::contexts::{Context, EagerContext};
    use crate::differentiation::{DerivativeTransform, DifferentiationError, DifferentiationParameterRole};
    use crate::interpretation::InterpretableOperation;
    use crate::operations::compare::{CompareOperation, ComparisonDirection};
    use crate::operations::constants::{OneLike, OneLikeOperation, ZeroLike, ZeroLikeOperation};
    use crate::operations::control_flow::ConditionOperation;
    use crate::operations::debugging::PrintOperation;
    use crate::operations::math::{AddOperation, MulOperation, Sin, SubOperation};
    use crate::parameters::Placeholder;
    use crate::programs::ProgramBuilder;
    use crate::programs::types::Typed;
    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use crate::tracing_v2::{DenseDifferentiate, ForwardModeDifferentiate, ReverseModeDifferentiate, jacrev};
    use crate::types::{Shape, Size};

    use super::*;

    #[test]
    fn test_dot_batches_mixed_lhs_batched_rhs_replicated() {
        // LHS is mapped at axis 0 with per-item shape [3]; RHS is replicated with shape [3].
        // Per-item semantics: dot(lhs_row, rhs) over the shared K=3 dimension. The batching rule
        // should broadcast the RHS to gain a singleton batch axis at position 0, then thread the
        // batch axis through `lift_dot_dimensions`.
        use crate::operations::math::{DotDimensionNumbers, DotOperation};
        let lhs = {
            let value = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
            ArrayBatch::new(value.r#type().into_owned(), value, Some(0))
        }
        .unwrap();
        let rhs = ArrayBatch::replicated(Array::vector(vec![10.0, 100.0, 1000.0]));
        let dimensions = DotDimensionNumbers::new(vec![0], vec![0], vec![], vec![]);
        let outputs = DotOperation::new(dimensions)
            .batch(&BatchingContext::new(EagerContext::<Array>::new(), 2), &crate::EmptyRegionDriver, &[lhs, rhs])
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        // Batch item 0: 1*10 + 2*100 + 3*1000 = 3210; batch item 1: 4*10 + 5*100 + 6*1000 = 6540.
        assert_eq!(outputs[0].value().values(), &[3210.0, 6540.0]);
    }

    #[test]
    fn test_reduce_sum_jvp_linearizes_to_itself() {
        // Verify the linear reduce rule directly over a concrete tangent value.
        use crate::operations::math::{ReduceOperation, ReductionKind};
        let primal = Array::vector(vec![1.0, 2.0, 3.0, 4.0]);
        let tangent_value = Array::vector(vec![0.5, 0.5, 0.5, 0.5]);

        let operation = ReduceOperation::new(vec![0], ReductionKind::Sum);

        // Primal: reduce(x, [0], Sum) on `Array` directly.
        let primal_output = operation
            .interpret(&EagerContext::<Array>::new(), &crate::EmptyRegionDriver, std::slice::from_ref(&primal))
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        assert_eq!(primal_output.values(), &[10.0]);

        // Tangent: linearizes to itself (Sum is linear), so the tangent of the reduce is the
        // reduce of the tangent.
        let tangent_outputs = operation
            .interpret(&EagerContext::<Array>::new(), &crate::EmptyRegionDriver, std::slice::from_ref(&tangent_value))
            .unwrap();
        let tangent_output = tangent_outputs.into_iter().next().unwrap();
        assert_eq!(tangent_output.values(), &[2.0]);
    }

    #[test]
    fn test_batch_varying_while_terminates_items_independently() {
        // Build a batched while loop with a per-item termination predicate. Each batch item starts at a
        // different value and decrements by 1 until it reaches 0. Batch item 0 (initial 3.0) iterates
        // three times, batch item 1 (initial 1.0) iterates once, batch item 2 (initial 2.0) iterates twice;
        // inactive batch items retain their final state via per-item `Select` masking.
        use crate::operations::compare::ComparisonDirection;
        use crate::operations::control_flow::WhileOperation;
        use crate::programs::Program;
        type TestOp = ArrayOperation<Array>;

        let scalar_f64 = ArrayType::scalar(DataType::F64);

        // Condition program: state -> (state > 0). Returns a scalar Boolean.
        let mut condition_builder = ProgramBuilder::<Array, TestOp>::new();
        let cond_input = condition_builder.add_input(scalar_f64.clone());
        let cond_zero = condition_builder.add_instruction(ZeroLikeOperation, Vec::new(), vec![cond_input]).unwrap()[0];
        let cond_output = condition_builder
            .add_instruction(
                CompareOperation::new(ComparisonDirection::GreaterThan),
                Vec::new(),
                vec![cond_input, cond_zero],
            )
            .unwrap()[0];
        let condition: Program<Array, TestOp, Vec<Array>, Vec<Array>> = condition_builder
            .build::<Vec<Array>, Vec<Array>>(vec![cond_output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        // Body program: state -> state - 1.
        let mut body_builder = ProgramBuilder::<Array, TestOp>::new();
        let body_input = body_builder.add_input(scalar_f64);
        let body_one = body_builder.add_instruction(OneLikeOperation, Vec::new(), vec![body_input]).unwrap()[0];
        let body_output =
            body_builder.add_instruction(SubOperation, Vec::new(), vec![body_input, body_one]).unwrap()[0];
        let body: Program<Array, TestOp, Vec<Array>, Vec<Array>> = body_builder
            .build::<Vec<Array>, Vec<Array>>(vec![body_output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let while_op = WhileOperation::new();
        let context = BatchingContext::new(EagerContext::<Array, TestOp>::new(), 3);

        let initial_state = {
            let value = Array::vector(vec![3.0, 1.0, 2.0]);
            ArrayBatch::new(value.r#type().into_owned(), value, Some(0))
        }
        .unwrap();
        let inputs = [BatchingTracer::new(context.clone(), initial_state)];
        let outputs = context
            .bind(while_op, vec![condition.clone(), body.clone()], &inputs)
            .unwrap()
            .into_iter()
            .map(BatchingTracer::into_batch)
            .collect::<Vec<_>>();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        // Each batch item terminates when its value reaches 0; inactive batch items retain their last value.
        assert_eq!(outputs[0].value().values(), &[0.0, 0.0, 0.0]);

        // A semantic iteration bound truncates the batched loop too: every batch item performs at most two body
        // applications, so batch item 0 (initial 3.0) is cut off at 1.0 while the other batch items terminate through
        // their own predicates first.
        let bounded_while_op = while_op.with_iteration_bound(2).unwrap();
        let initial_state = {
            let value = Array::vector(vec![3.0, 1.0, 2.0]);
            ArrayBatch::new(value.r#type().into_owned(), value, Some(0))
        }
        .unwrap();
        let inputs = [BatchingTracer::new(context.clone(), initial_state)];
        let outputs = context
            .bind(bounded_while_op, vec![condition.clone(), body.clone()], &inputs)
            .unwrap()
            .into_iter()
            .map(BatchingTracer::into_batch)
            .collect::<Vec<_>>();
        assert_eq!(outputs[0].value().values(), &[1.0, 0.0, 0.0]);

        // The replicated batched loop respects the bound as well: an unbatched initial state of 5.0 stops at 3.0.
        let initial_state = ArrayBatch::replicated(Array::scalar(5.0));
        let inputs = [BatchingTracer::new(context.clone(), initial_state)];
        let outputs = context
            .bind(bounded_while_op, vec![condition, body], &inputs)
            .unwrap()
            .into_iter()
            .map(BatchingTracer::into_batch)
            .collect::<Vec<_>>();
        assert_eq!(outputs[0].value().values(), &[3.0]);
    }

    #[test]
    fn test_jacfwd_batches_basis_tangents() {
        let jacobian = EagerContext::<Array, ArrayOperation<Array>>::new()
            .jacfwd(|(x, y)| Ok((x.clone() * y.clone() + x.sin()?, x + y)), (Array::scalar(2.0), Array::scalar(3.0)))
            .unwrap();

        let blocks = jacobian.iter_blocks().collect::<Vec<_>>();
        let [block_00, block_01, block_10, block_11] = blocks.as_slice() else { unreachable!() };

        assert_eq!(block_00.output_type().static_shape().unwrap().as_slice(), &[] as &[usize]);
        assert_eq!(block_00.input_type().static_shape().unwrap().as_slice(), &[] as &[usize]);

        assert_abs_diff_eq!(block_00.value().values()[0], 3.0 + 2.0f64.cos(), epsilon = 1e-9);
        assert_abs_diff_eq!(block_01.value().values()[0], 2.0, epsilon = 1e-9);
        assert_abs_diff_eq!(block_10.value().values()[0], 1.0, epsilon = 1e-9);
        assert_abs_diff_eq!(block_11.value().values()[0], 1.0, epsilon = 1e-9);
    }

    #[test]
    fn test_jacrev_batches_basis_cotangents() {
        let jacobian =
            jacrev(|(x, y)| Ok((x.clone() * y.clone() + x.sin()?, x + y)), (Array::scalar(2.0), Array::scalar(3.0)))
                .unwrap();

        let blocks = jacobian.iter_blocks().collect::<Vec<_>>();
        let [block_00, block_01, block_10, block_11] = blocks.as_slice() else { unreachable!() };

        assert_abs_diff_eq!(block_00.value().values()[0], 3.0 + 2.0f64.cos(), epsilon = 1e-9);
        assert_abs_diff_eq!(block_01.value().values()[0], 2.0, epsilon = 1e-9);
        assert_abs_diff_eq!(block_10.value().values()[0], 1.0, epsilon = 1e-9);
        assert_abs_diff_eq!(block_11.value().values()[0], 1.0, epsilon = 1e-9);
    }

    #[test]
    fn test_dense_differentiation_uses_widened_f8_differential_values() {
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let input = Array::from_f64s(ArrayType::scalar(DataType::F8E8M0FNU), vec![2.0]);

        let forward = context.jacfwd(|value| value.sin(), input.clone()).unwrap();
        let forward_block = forward.iter_blocks().next().unwrap();
        assert_eq!(forward_block.value().r#type().as_ref(), &ArrayType::scalar(DataType::F32));
        // The derivative payload is honestly `f32`-encoded, so the comparison happens at `f32` precision.
        assert_abs_diff_eq!(forward_block.value().values()[0], 2.0f64.cos(), epsilon = 1e-6);

        let reverse = context.jacrev(|value| value.sin(), input.clone()).unwrap();
        let reverse_block = reverse.iter_blocks().next().unwrap();
        assert_eq!(reverse_block.value().r#type().as_ref(), &ArrayType::scalar(DataType::F32));
        // The derivative payload is honestly `f32`-encoded, so the comparison happens at `f32` precision.
        assert_abs_diff_eq!(reverse_block.value().values()[0], 2.0f64.cos(), epsilon = 1e-6);

        let hessian = context.hessian(|value| value.sin(), input).unwrap();
        let hessian_block = hessian.iter_blocks().next().unwrap();
        assert_eq!(hessian_block.value().r#type().as_ref(), &ArrayType::scalar(DataType::F32));
        // The derivative payload is honestly `f32`-encoded, so the comparison happens at `f32` precision.
        assert_abs_diff_eq!(hessian_block.value().values()[0], -2.0f64.sin(), epsilon = 1e-6);
    }

    #[test]
    fn test_jacrev_converts_promoted_cotangents_to_each_input_type() {
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let f32 = Array::from_f64s(ArrayType::scalar(DataType::F32), vec![2.0]);
        let f64 = Array::from_f64s(ArrayType::scalar(DataType::F64), vec![3.0]);

        let add = context.jacrev(|(left, right)| Ok(left + right), (f32.clone(), f64.clone())).unwrap();
        let add_blocks = add.iter_blocks().collect::<Vec<_>>();
        assert_eq!(add_blocks[0].value().r#type().data_type(), DataType::F32);
        assert_eq!(add_blocks[1].value().r#type().data_type(), DataType::F64);
        assert_abs_diff_eq!(add_blocks[0].value().values()[0], 1.0, epsilon = 1e-9);
        assert_abs_diff_eq!(add_blocks[1].value().values()[0], 1.0, epsilon = 1e-9);

        let sub = context.jacrev(|(left, right)| Ok(left - right), (f32.clone(), f64.clone())).unwrap();
        let sub_blocks = sub.iter_blocks().collect::<Vec<_>>();
        assert_abs_diff_eq!(sub_blocks[0].value().values()[0], 1.0, epsilon = 1e-9);
        assert_abs_diff_eq!(sub_blocks[1].value().values()[0], -1.0, epsilon = 1e-9);

        let mul = context.jacrev(|(left, right)| Ok(left * right), (f32, f64)).unwrap();
        let mul_blocks = mul.iter_blocks().collect::<Vec<_>>();
        assert_abs_diff_eq!(mul_blocks[0].value().values()[0], 3.0, epsilon = 1e-9);
        assert_abs_diff_eq!(mul_blocks[1].value().values()[0], 2.0, epsilon = 1e-9);
    }

    #[test]
    fn test_vjp_restores_each_elementwise_input_sharding() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharded_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2)]))
            .with_sharding(Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        let replicated_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2)]))
            .with_sharding(Sharding::replicated(mesh, 1))
            .unwrap();
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let (output, pullback) = context
            .vjp(
                |(left, right)| Ok(left + right),
                (
                    Array::from_f64s(sharded_type.clone(), vec![1.0, 2.0]),
                    Array::from_f64s(replicated_type.clone(), vec![3.0, 4.0]),
                ),
            )
            .unwrap();
        assert!(pullback.program().to_string().contains("reshard"));
        let (left, right) = pullback.apply(Array::from_f64s(output.r#type().into_owned(), vec![1.0, 1.0])).unwrap();

        assert_eq!(left.r#type().as_ref(), &sharded_type);
        assert_eq!(right.r#type().as_ref(), &replicated_type);
    }

    #[test]
    fn test_dense_jacobians_unbroadcast_scalar_elementwise_inputs() {
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let primals = (Array::scalar(2.0), Array::vector(vec![3.0, 4.0]));
        let forward = context.jacfwd(|(scalar, vector)| Ok(scalar.clone() * vector + scalar), primals.clone()).unwrap();
        let reverse = context.jacrev(|(scalar, vector)| Ok(scalar.clone() * vector + scalar), primals).unwrap();

        for jacobian in [forward, reverse] {
            let blocks = jacobian.iter_blocks().collect::<Vec<_>>();
            assert_eq!(blocks.len(), 2);
            assert_eq!(blocks[0].value().r#type().static_shape().unwrap().as_slice(), &[2]);
            assert_eq!(blocks[0].value().values(), &[4.0, 5.0]);
            assert_eq!(blocks[1].value().r#type().static_shape().unwrap().as_slice(), &[2, 2]);
            assert_eq!(blocks[1].value().values(), &[2.0, 0.0, 0.0, 2.0]);
        }
    }

    #[test]
    fn test_batching_composes_around_dense_jacobian() {
        let jacobian = crate::batching::batch(
            |input| {
                input
                    .context()
                    .clone()
                    .jacfwd(|value| Ok(value.clone() * value), input)
                    .map_err(|error| crate::ProgramError::MalformedProgram(error.to_string()))
            },
            Array::vector(vec![1.0, 2.0, 3.0]),
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        )
        .unwrap();

        let block = jacobian.iter_blocks().next().unwrap();
        assert_eq!(block.value().values(), &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_dense_jacobians_keep_zero_sized_blocks_as_domain_values() {
        let r#type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(0)]));
        let input = Array::from_f64s(r#type.clone(), Vec::new());
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();

        let forward = context.jacfwd(|x| Ok(x.clone() + x), input.clone()).unwrap();
        let forward_block = forward.iter_blocks().next().unwrap();
        assert_eq!(forward_block.output_type().static_shape().unwrap().as_slice(), &[0]);
        assert_eq!(forward_block.input_type().static_shape().unwrap().as_slice(), &[0]);
        let block_type = r#type.with_inserted_dimension(0, Size::Static(0)).unwrap();
        assert_eq!(forward_block.value().r#type().as_ref(), &block_type);
        assert!(forward_block.value().values().is_empty());

        let reverse = jacrev(|x| Ok(x.clone() + x), input).unwrap();
        let reverse_block = reverse.iter_blocks().next().unwrap();
        assert_eq!(reverse_block.output_type().static_shape().unwrap().as_slice(), &[0]);
        assert_eq!(reverse_block.input_type().static_shape().unwrap().as_slice(), &[0]);
        assert_eq!(reverse_block.value().r#type(), forward_block.value().r#type());
        assert!(reverse_block.value().values().is_empty());

        let input = Array::from_f64s(r#type, Vec::new());
        let hessian = context.hessian(|x| Ok(x.clone() * x), input).unwrap();
        let hessian_block = hessian.iter_blocks().next().unwrap();
        assert_eq!(hessian_block.value().r#type().static_shape().unwrap().as_slice(), &[0, 0, 0]);
        assert!(hessian_block.value().values().is_empty());
    }

    #[test]
    fn test_jacfwd_iter_blocks_yields_each_output_input_pair() {
        let jacobian = EagerContext::<Array, ArrayOperation<Array>>::new()
            .jacfwd(|(x, y)| Ok((x.clone() * y.clone() + x.sin()?, x + y)), (Array::scalar(2.0), Array::scalar(3.0)))
            .unwrap();

        let triples = jacobian
            .iter_blocks()
            .map(|block| (block.output_path().to_string(), block.input_path().to_string(), block.value().values()[0]))
            .collect::<Vec<_>>();

        assert_eq!(triples.len(), 4);
        assert_eq!(triples[0].0, "$.0");
        assert_eq!(triples[0].1, "$.0");
        assert_abs_diff_eq!(triples[0].2, 3.0 + 2.0f64.cos(), epsilon = 1e-9);
        assert_eq!(triples[1].0, "$.0");
        assert_eq!(triples[1].1, "$.1");
        assert_abs_diff_eq!(triples[1].2, 2.0, epsilon = 1e-9);
        assert_eq!(triples[2].0, "$.1");
        assert_eq!(triples[2].1, "$.0");
        assert_abs_diff_eq!(triples[2].2, 1.0, epsilon = 1e-9);
        assert_eq!(triples[3].0, "$.1");
        assert_eq!(triples[3].1, "$.1");
        assert_abs_diff_eq!(triples[3].2, 1.0, epsilon = 1e-9);
    }

    #[test]
    fn test_hessian_accepts_original_scalar_function() {
        let hessian = EagerContext::<Array, ArrayOperation<Array>>::new()
            .hessian(|(x, y)| Ok(x.clone() * y + x.sin()?), (Array::scalar(2.0), Array::scalar(3.0)))
            .unwrap();

        let blocks = hessian.iter_blocks().collect::<Vec<_>>();
        let [block_00, block_01, block_10, block_11] = blocks.as_slice() else { unreachable!() };

        assert_abs_diff_eq!(block_00.value().values()[0], -2.0f64.sin(), epsilon = 1e-9);
        assert_abs_diff_eq!(block_01.value().values()[0], 1.0, epsilon = 1e-9);
        assert_abs_diff_eq!(block_10.value().values()[0], 1.0, epsilon = 1e-9);
        assert_abs_diff_eq!(block_11.value().values()[0], 0.0, epsilon = 1e-9);
    }

    #[test]
    fn test_hessian_materializes_all_structured_output_blocks() {
        let hessian = EagerContext::<Array, ArrayOperation<Array>>::new()
            .hessian(|x| Ok((x.clone() * x.clone(), x.clone() * x.clone() * x)), Array::scalar(2.0))
            .unwrap();

        let blocks = hessian.iter_blocks().collect::<Vec<_>>();
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].output_path().to_string(), "$.0");
        assert_eq!(blocks[0].first_input_path().to_string(), "$");
        assert_eq!(blocks[0].second_input_path().to_string(), "$");
        assert_abs_diff_eq!(blocks[0].value().values()[0], 2.0, epsilon = 1e-9);
        assert_eq!(blocks[1].output_path().to_string(), "$.1");
        assert_abs_diff_eq!(blocks[1].value().values()[0], 12.0, epsilon = 1e-9);
    }

    #[test]
    fn test_hessian_materializes_structured_mixed_rank_cartesian_product() {
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let hessian = context
            .hessian(
                |(vector, scalar)| Ok((vector.clone() * vector, scalar.clone() * scalar)),
                (Array::vector(vec![1.0, 2.0]), Array::scalar(3.0)),
            )
            .unwrap();
        let blocks = hessian.iter_blocks().collect::<Vec<_>>();

        assert_eq!(blocks.len(), 8);
        assert_eq!(blocks[0].output_path().to_string(), "$.0");
        assert_eq!(blocks[0].first_input_path().to_string(), "$.0");
        assert_eq!(blocks[0].second_input_path().to_string(), "$.0");
        assert_eq!(blocks[0].value().r#type().static_shape().unwrap().as_slice(), &[2, 2, 2]);
        assert_eq!(blocks[0].value().values(), &[2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0]);
        assert_eq!(blocks[1].value().r#type().static_shape().unwrap().as_slice(), &[2, 2]);
        assert_eq!(blocks[1].value().values(), &[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(blocks[2].value().r#type().static_shape().unwrap().as_slice(), &[2, 2]);
        assert_eq!(blocks[2].value().values(), &[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(blocks[3].value().r#type().static_shape().unwrap().as_slice(), &[2]);
        assert_eq!(blocks[3].value().values(), &[0.0, 0.0]);
        assert_eq!(blocks[4].output_path().to_string(), "$.1");
        assert_eq!(blocks[4].value().r#type().static_shape().unwrap().as_slice(), &[2, 2]);
        assert_eq!(blocks[4].value().values(), &[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(blocks[5].value().r#type().static_shape().unwrap().as_slice(), &[2]);
        assert_eq!(blocks[5].value().values(), &[0.0, 0.0]);
        assert_eq!(blocks[6].value().r#type().static_shape().unwrap().as_slice(), &[2]);
        assert_eq!(blocks[6].value().values(), &[0.0, 0.0]);
        assert_eq!(blocks[7].first_input_path().to_string(), "$.1");
        assert_eq!(blocks[7].second_input_path().to_string(), "$.1");
        assert!(blocks[7].value().r#type().static_shape().unwrap().as_slice().is_empty());
        assert_eq!(blocks[7].value().values(), &[2.0]);
    }

    #[test]
    fn test_dense_differentiation_can_differentiate_hessian_values() {
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let derivative = context
            .jacfwd(
                |input| {
                    let nested_context = input.context().clone();
                    let hessian = nested_context
                        .hessian(|value| Ok(value.clone() * value.clone() * value), input)
                        .map_err(|error| crate::ProgramError::MalformedProgram(error.to_string()))?;
                    Ok(hessian.into_values().remove(0))
                },
                Array::scalar(2.0),
            )
            .unwrap();

        assert_abs_diff_eq!(derivative.iter_blocks().next().unwrap().value().values()[0], 6.0, epsilon = 1e-9);
    }

    #[test]
    fn test_reverse_mode_composes_around_forward_jacobian() {
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let derivative = context
            .jacrev(
                |input| {
                    let nested_context = input.context().clone();
                    let jacobian = nested_context
                        .jacfwd(|value| Ok(value.clone() * value), input)
                        .map_err(|error| crate::ProgramError::MalformedProgram(error.to_string()))?;
                    Ok(jacobian.into_values().remove(0))
                },
                Array::scalar(3.0),
            )
            .unwrap();

        assert_abs_diff_eq!(derivative.iter_blocks().next().unwrap().value().values()[0], 2.0, epsilon = 1e-9);
    }

    #[test]
    fn test_dense_differentiation_with_aux_preserves_auxiliary_values() {
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();

        let forward_evaluations = Cell::new(0);
        let (forward, forward_auxiliary) = context
            .jacfwd_with_aux(
                |x| {
                    forward_evaluations.set(forward_evaluations.get() + 1);
                    Ok((x.clone() * x.clone(), x))
                },
                Array::scalar(2.0),
            )
            .unwrap();
        assert_eq!(forward_evaluations.get(), 1);
        assert_abs_diff_eq!(forward.iter_blocks().next().unwrap().value().values()[0], 4.0, epsilon = 1e-9);
        assert_eq!(forward_auxiliary.values(), &[2.0]);

        let reverse_evaluations = Cell::new(0);
        let (reverse, reverse_auxiliary) = context
            .jacrev_with_aux(
                |x| {
                    reverse_evaluations.set(reverse_evaluations.get() + 1);
                    Ok((x.clone() * x.clone(), x))
                },
                Array::scalar(2.0),
            )
            .unwrap();
        assert_eq!(reverse_evaluations.get(), 1);
        assert_abs_diff_eq!(reverse.iter_blocks().next().unwrap().value().values()[0], 4.0, epsilon = 1e-9);
        assert_eq!(reverse_auxiliary.values(), &[2.0]);

        let hessian_evaluations = Cell::new(0);
        let (hessian, hessian_auxiliary) = context
            .hessian_with_aux(
                |x| {
                    hessian_evaluations.set(hessian_evaluations.get() + 1);
                    Ok((x.clone() * x.clone(), x))
                },
                Array::scalar(2.0),
            )
            .unwrap();
        assert_eq!(hessian_evaluations.get(), 1);
        assert_abs_diff_eq!(hessian.iter_blocks().next().unwrap().value().values()[0], 2.0, epsilon = 1e-9);
        assert_eq!(hessian_auxiliary.values(), &[2.0]);
    }

    #[test]
    fn test_dense_differentiation_validates_element_and_coordinate_types() {
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();

        assert_eq!(
            crate::tracing_v2::jacfwd(|inputs| Ok(inputs), Vec::<Array>::new()).unwrap_err(),
            DifferentiationError::EmptyInput,
        );

        let integer = Array::from_f64s(ArrayType::scalar(DataType::I32), vec![2.0]);
        assert_eq!(
            context.jacfwd(|x| Ok(x), integer).unwrap_err(),
            DifferentiationError::NonDifferentiableParameter {
                transform: DerivativeTransform::JacobianForward,
                role: DifferentiationParameterRole::Input,
                path: "$".to_string(),
                r#type: "i32[]".to_string(),
            },
        );

        let complex = Array::from_f64s(ArrayType::scalar(DataType::C64), vec![2.0]);
        assert_eq!(
            context.jacfwd(|x| Ok(x), complex.clone()).unwrap_err(),
            DifferentiationError::ComplexParameter {
                transform: DerivativeTransform::JacobianForward,
                role: DifferentiationParameterRole::Input,
                path: "$".to_string(),
                r#type: "c64[]".to_string(),
            },
        );
        let holomorphic = context.jacfwd_holomorphic(|x| Ok(x), complex).unwrap();
        assert_eq!(
            holomorphic.iter_blocks().next().unwrap().value().values(),
            &[Scalar::C64(ComplexNumber::new(1.0, 0.0))],
        );

        assert_eq!(
            context.jacrev_holomorphic(|x| Ok(x), Array::scalar(2.0)).unwrap_err(),
            DifferentiationError::NonComplexParameter {
                transform: DerivativeTransform::JacobianReverse,
                role: DifferentiationParameterRole::Input,
                path: "$".to_string(),
                r#type: "f64[]".to_string(),
            },
        );

        let complex_output_error = context
            .jacrev(
                |input| Ok(input.context().lift(Array::from_f64s(ArrayType::scalar(DataType::C64), vec![1.0]))?),
                Array::scalar(2.0),
            )
            .unwrap_err();
        assert_eq!(
            complex_output_error,
            DifferentiationError::ComplexParameter {
                transform: DerivativeTransform::JacobianReverse,
                role: DifferentiationParameterRole::Output,
                path: "$".to_string(),
                r#type: "c64[]".to_string(),
            },
        );

        let dynamic_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Dynamic(None)]));
        let dynamic = Array::with_unchecked_type(dynamic_type, vec![Scalar::F64(1.0)]);
        assert_eq!(
            context.jacfwd(|x| Ok(x), dynamic).unwrap_err(),
            DifferentiationError::NonFiniteCoordinateSpace {
                transform: DerivativeTransform::JacobianForward,
                role: DifferentiationParameterRole::Input,
                path: "$".to_string(),
                r#type: "f64[*]".to_string(),
            },
        );

        let dynamic_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Dynamic(None)]));
        assert_eq!(
            context
                .jacfwd(
                    |input| Ok(input
                        .context()
                        .lift(Array::with_unchecked_type(dynamic_type.clone(), vec![Scalar::F64(1.0)]))?),
                    Array::scalar(1.0),
                )
                .unwrap_err(),
            DifferentiationError::NonFiniteCoordinateSpace {
                transform: DerivativeTransform::JacobianForward,
                role: DifferentiationParameterRole::Output,
                path: "$".to_string(),
                r#type: "f64[*]".to_string(),
            },
        );

        let dynamic = Array::with_unchecked_type(dynamic_type, vec![Scalar::F64(1.0)]);
        assert_eq!(
            context.jacrev(|input| Ok(input.context().lift(Array::scalar(1.0))?), dynamic).unwrap_err(),
            DifferentiationError::NonFiniteCoordinateSpace {
                transform: DerivativeTransform::JacobianReverse,
                role: DifferentiationParameterRole::Input,
                path: "$".to_string(),
                r#type: "f64[*]".to_string(),
            },
        );
    }

    #[test]
    fn test_jacfwd_handles_function_with_independent_outputs() {
        // f(x, y) = (x*y + sin(x), y, x + y) — output[1] is independent of x.
        let jacobian = EagerContext::<Array, ArrayOperation<Array>>::new()
            .jacfwd(
                |(x, y)| Ok((x.clone() * y.clone() + x.sin()?, y.clone(), x + y)),
                (Array::scalar(2.0), Array::scalar(3.0)),
            )
            .unwrap();

        let triples = jacobian
            .iter_blocks()
            .map(|block| (block.output_path().to_string(), block.input_path().to_string(), block.value().values()[0]))
            .collect::<Vec<_>>();

        // 3 outputs * 2 inputs = 6 blocks
        assert_eq!(triples.len(), 6);
        // d(x*y + sin(x))/dx = y + cos(x) = 3 + cos(2)
        assert_abs_diff_eq!(triples[0].2, 3.0 + 2.0f64.cos(), epsilon = 1e-9);
        // d(x*y + sin(x))/dy = x = 2
        assert_abs_diff_eq!(triples[1].2, 2.0, epsilon = 1e-9);
        // dy/dx = 0  (independent of x — exercise the all-zero short-circuit downstream)
        assert_abs_diff_eq!(triples[2].2, 0.0, epsilon = 1e-9);
        // dy/dy = 1
        assert_abs_diff_eq!(triples[3].2, 1.0, epsilon = 1e-9);
        // d(x + y)/dx = 1
        assert_abs_diff_eq!(triples[4].2, 1.0, epsilon = 1e-9);
        // d(x + y)/dy = 1
        assert_abs_diff_eq!(triples[5].2, 1.0, epsilon = 1e-9);
    }

    /// Builds a replicated scalar Boolean predicate batch with the provided truth value.
    fn replicated_predicate(value: bool) -> ArrayBatch<Array> {
        ArrayBatch::replicated(Array::from_f64s(
            ArrayType::scalar(DataType::Boolean),
            vec![if value { 1.0 } else { 0.0 }],
        ))
    }

    /// Builds a single-input program that scales a vector input by `factor`.
    fn vector_scale_branch(
        size: usize,
        factor: f64,
    ) -> crate::Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(size)])));
        let factor = builder.add_constant(Array::scalar(factor));
        let output = builder.add_instruction(MulOperation, Vec::new(), vec![input, factor]).unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Builds a vector-input program that returns a replicated constant vector.
    fn constant_vector_branch(
        values: Vec<f64>,
    ) -> crate::Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(values.len())])));
        let output = builder.add_constant(Array::vector(values));
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Batches a vector-valued condition whose true and false branches scale their input by two and three.
    fn batch_vector_condition(batch_size: usize, item_size: usize, input_values: Vec<f64>) -> ArrayBatch<Array> {
        let physical_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(batch_size), Size::Static(item_size)]));
        let predicate_type = ArrayType::new(DataType::Boolean, Shape::new(vec![Size::Static(batch_size)]));
        let predicate_values = (0..batch_size).map(|index| if index == 0 { 1.0 } else { 0.0 }).collect();
        let predicate = Array::from_f64s(predicate_type.clone(), predicate_values);
        let predicate = ArrayBatch::new(predicate_type, predicate, Some(0)).unwrap();
        let operand = Array::from_f64s(physical_type.clone(), input_values);
        let operand = ArrayBatch::new(physical_type, operand, Some(0)).unwrap();
        let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), batch_size);
        let mut outputs = context
            .bind(
                ArrayOperation::Condition(ConditionOperation::new()),
                vec![vector_scale_branch(item_size, 2.0), vector_scale_branch(item_size, 3.0)],
                &[BatchingTracer::new(context.clone(), predicate), BatchingTracer::new(context.clone(), operand)],
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        outputs.remove(0).into_batch()
    }

    #[test]
    fn test_condition_batches_replicated_true_predicate_over_array_batches() {
        use crate::batching::{ArrayBatch, BatchAxis, BatchingTracer};

        // A replicated `true` predicate selects scalar_scale_branch(2.0). Pass a 3-item batched operand and
        // verify each batch item is independently scaled by 2.
        let condition_regions = vec![scalar_scale_branch(2.0), scalar_scale_branch(3.0)];
        let condition = ConditionOperation::new();
        let operation = ArrayOperation::Condition(condition);
        let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 3);

        let batched_input = {
            let value = Array::vector(vec![1.0, 4.0, 9.0]);
            ArrayBatch::new(value.r#type().into_owned(), value, Some(0))
        }
        .unwrap();
        let outputs = context
            .bind(
                operation,
                condition_regions,
                &[
                    BatchingTracer::new(context.clone(), replicated_predicate(true)),
                    BatchingTracer::new(context.clone(), batched_input),
                ],
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        let output_batch = outputs[0].batch();
        assert_eq!(output_batch.batch_axis(), BatchAxis::new(0));
        assert_eq!(output_batch.value().to_f64s(), vec![2.0, 8.0, 18.0]);
    }

    #[test]
    fn test_condition_batches_false_branch_when_replicated_predicate_is_false() {
        use crate::batching::{ArrayBatch, BatchAxis, BatchingTracer};

        let condition_regions = vec![scalar_scale_branch(2.0), scalar_scale_branch(3.0)];
        let condition = ConditionOperation::new();
        let operation = ArrayOperation::Condition(condition);
        let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 3);

        let batched_input = {
            let value = Array::vector(vec![1.0, 4.0, 9.0]);
            ArrayBatch::new(value.r#type().into_owned(), value, Some(0))
        }
        .unwrap();
        let outputs = context
            .bind(
                operation,
                condition_regions,
                &[
                    BatchingTracer::new(context.clone(), replicated_predicate(false)),
                    BatchingTracer::new(context.clone(), batched_input),
                ],
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        let output_batch = outputs[0].batch();
        assert_eq!(output_batch.batch_axis(), BatchAxis::new(0));
        assert_eq!(output_batch.value().to_f64s(), vec![3.0, 12.0, 27.0]);
    }

    #[test]
    fn test_dot_general_evaluates_batched_matmul() {
        use crate::operations::math::{Dot, DotDimensionNumbers};

        // Batched matmul: [2, 2, 3] @ [2, 3, 2] -> [2, 2, 2] with axis 0 batched.
        let lhs_values: Vec<f64> = (1..=12).map(|value| value as f64).collect();
        let rhs_values: Vec<f64> = (1..=12).map(|value| value as f64).collect();
        let lhs = Array::from_f64s(
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(2), Size::Static(3)])),
            lhs_values,
        );
        let rhs = Array::from_f64s(
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3), Size::Static(2)])),
            rhs_values,
        );

        let dimensions = DotDimensionNumbers::new(vec![2], vec![1], vec![0], vec![0]);
        let output = lhs.dot(&rhs, &dimensions);

        assert_eq!(
            output.r#type().into_owned(),
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(2), Size::Static(2)])),
        );
        // Batch 0: [[1,2,3],[4,5,6]] @ [[1,2],[3,4],[5,6]] = [[22,28],[49,64]]
        // Batch 1: [[7,8,9],[10,11,12]] @ [[7,8],[9,10],[11,12]] = [[220,244],[301,334]]
        assert_eq!(output.to_f64s(), vec![22.0, 28.0, 49.0, 64.0, 220.0, 244.0, 301.0, 334.0]);
    }

    #[test]
    fn test_transpose_evaluates_general_permutation() {
        use crate::operations::manipulation::Transpose;

        // Rank-3 transpose with permutation [2, 0, 1]: [2, 3, 4] -> [4, 2, 3].
        let values: Vec<f64> = (0..24).map(|value| value as f64).collect();
        let input = Array::from_f64s(
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3), Size::Static(4)])),
            values,
        );

        let output = input.transpose(vec![2, 0, 1]).unwrap();

        assert_eq!(
            output.r#type().into_owned(),
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(4), Size::Static(2), Size::Static(3)])),
        );
        // Spot-check: input[0, 0, 0] (= 0) goes to output[0, 0, 0]; input[0, 0, 1] (= 1) -> output[1, 0, 0];
        // input[1, 2, 3] (= 23) -> output[3, 1, 2].
        assert_eq!(output.to_f64s()[0], 0.0);
        assert_eq!(output.to_f64s()[1 * 6], 1.0);
        let output_flat_for_23 = 3 * 6 + 1 * 3 + 2;
        assert_eq!(output.to_f64s()[output_flat_for_23], 23.0);
    }

    #[test]
    fn test_jacrev_over_dot_batches_adjoint_dots() {
        use crate::operations::math::{Dot, DotDimensionNumbers};

        // jacrev internally batches the pullback's adjoint `dot` operations (their known operands riding as
        // replicated pullback inputs) through BatchableOperation::batch — exercise that path explicitly via a
        // dot-based scalar function. f(x, y) = x · y (inner product) so ∂f/∂x = y and ∂f/∂y = x.
        let jacobian = jacrev(
            |(x, y)| Ok(x.dot(&y, &DotDimensionNumbers::inner_product())),
            (Array::vector(vec![2.0, 3.0, 5.0]), Array::vector(vec![7.0, 11.0, 13.0])),
        )
        .unwrap();

        let blocks = jacobian.iter_blocks().collect::<Vec<_>>();
        let [block_x, block_y] = blocks.as_slice() else { unreachable!() };
        assert_eq!(block_x.value().values(), &[7.0, 11.0, 13.0]);
        assert_eq!(block_y.value().values(), &[2.0, 3.0, 5.0]);
    }

    #[test]
    fn test_jacfwd_over_dot_batches_basis_tangents_through_the_pushforward() {
        use crate::operations::math::{Dot, DotDimensionNumbers};

        // jacfwd linearizes the function once, then replays all input-coordinate basis tangents through the
        // pushforward in one batched pass. A dot-product scalar output exercises captured-factor (product-rule)
        // linear maps instead of only elementwise tangent arithmetic.
        let jacobian = EagerContext::<Array, ArrayOperation<Array>>::new()
            .jacfwd(
                |(x, y)| Ok(x.dot(&y, &DotDimensionNumbers::inner_product())),
                (Array::vector(vec![2.0, 3.0, 5.0]), Array::vector(vec![7.0, 11.0, 13.0])),
            )
            .unwrap();

        let blocks = jacobian.iter_blocks().collect::<Vec<_>>();
        let [block_x, block_y] = blocks.as_slice() else { unreachable!() };
        assert_eq!(block_x.value().values(), &[7.0, 11.0, 13.0]);
        assert_eq!(block_y.value().values(), &[2.0, 3.0, 5.0]);
    }

    #[test]
    fn test_batching_batch_varying_condition_selects_per_item() {
        use crate::batching::BatchingTracer;
        // Per-item scalar branches: on_true scales by 2.0, on_false scales by 3.0. Operand is a
        // [4]-vector; predicate is a [4]-vector with values [1.0, 0.0, 1.0, 0.0]. Expected per-item
        // output: [1*2, 2*3, 3*2, 4*3] = [2, 6, 6, 12].
        let condition_regions = vec![scalar_scale_branch(2.0), scalar_scale_branch(3.0)];
        let condition = ConditionOperation::new();
        let operation = ArrayOperation::Condition(condition);

        let predicate_type = ArrayType::new(DataType::Boolean, Shape::new(vec![Size::Static(4)]));
        let operand_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(4)]));
        let predicate_batch = ArrayBatch::new(
            predicate_type.clone(),
            Array::from_f64s(predicate_type, vec![1.0, 0.0, 1.0, 0.0]),
            Some(0),
        )
        .unwrap();
        let operand_batch = ArrayBatch::new(operand_type, Array::vector(vec![1.0, 2.0, 3.0, 4.0]), Some(0)).unwrap();

        let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 4);
        let outputs = context
            .bind(
                operation,
                condition_regions,
                &[
                    BatchingTracer::new(context.clone(), predicate_batch),
                    BatchingTracer::new(context.clone(), operand_batch),
                ],
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch().batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].batch().value().to_f64s(), vec![2.0, 6.0, 6.0, 12.0]);
    }

    #[test]
    fn test_batching_batch_varying_condition_selects_non_scalar_outputs_along_the_batch_axis() {
        // The batch size differs from the per-item vector length. The Boolean `[2]` predicate must become `[2, 1]`
        // before selecting between the `[2, 3]` branch values.
        let output = batch_vector_condition(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(output.batch_axis(), BatchAxis::new(0));
        assert_eq!(output.value().values(), &[2.0, 4.0, 6.0, 12.0, 15.0, 18.0]);

        // Equal batch and item sizes previously made trailing-axis broadcasting type-check while selecting columns
        // instead of rows. Pin the row-wise result explicitly.
        let output = batch_vector_condition(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(output.batch_axis(), BatchAxis::new(0));
        assert_eq!(output.value().values(), &[2.0, 4.0, 9.0, 12.0]);
    }

    #[test]
    fn test_batching_batch_varying_condition_aligns_replicated_and_mapped_branch_outputs() {
        let batch_size = 2;
        let item_size = 3;
        let predicate_type = ArrayType::new(DataType::Boolean, Shape::new(vec![Size::Static(batch_size)]));
        let predicate = Array::from_f64s(predicate_type.clone(), vec![1.0, 0.0]);
        let predicate = ArrayBatch::new(predicate_type, predicate, Some(0)).unwrap();
        let operand = Array::matrix(batch_size, item_size, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let operand = ArrayBatch::new(operand.r#type().into_owned(), operand, Some(0)).unwrap();
        let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), batch_size);

        let outputs = context
            .bind(
                ArrayOperation::Condition(ConditionOperation::new()),
                vec![constant_vector_branch(vec![10.0, 20.0, 30.0]), vector_scale_branch(item_size, 3.0)],
                &[BatchingTracer::new(context.clone(), predicate), BatchingTracer::new(context.clone(), operand)],
            )
            .unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch().batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].batch().value().values(), &[10.0, 20.0, 30.0, 12.0, 15.0, 18.0]);
    }

    #[test]
    fn test_batching_batch_varying_condition_rejects_effectful_branches_before_replay() {
        let mut effectful_builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = effectful_builder.add_input(ArrayType::scalar(DataType::F64));
        let output =
            effectful_builder.add_instruction(PrintOperation::new("branch"), Vec::new(), vec![input]).unwrap()[0];
        let effectful_branch = effectful_builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let batch_size = 2;
        let predicate_type = ArrayType::new(DataType::Boolean, Shape::new(vec![Size::Static(batch_size)]));
        let predicate = Array::from_f64s(predicate_type.clone(), vec![1.0, 0.0]);
        let predicate = ArrayBatch::new(predicate_type, predicate, Some(0)).unwrap();
        let operand = Array::vector(vec![1.0, 2.0]);
        let operand = ArrayBatch::new(operand.r#type().into_owned(), operand, Some(0)).unwrap();
        let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), batch_size);

        let error = context
            .bind(
                ArrayOperation::Condition(ConditionOperation::new()),
                vec![effectful_branch, scalar_scale_branch(3.0)],
                &[BatchingTracer::new(context.clone(), predicate), BatchingTracer::new(context.clone(), operand)],
            )
            .unwrap_err();

        assert!(matches!(
            error.downcast_custom::<BatchingError>(),
            Some(BatchingError::UnsupportedOperation { message })
                if message == "cannot batch a condition with a batch-varying predicate and effectful branches because \
                               observable effects cannot be selected per batch item",
        ));
    }

    #[test]
    fn test_select_batches_with_replicated_predicate_via_broadcast() {
        // Predicate is a rank-0 replicated scalar; on_true / on_false are mapped vectors of
        // size 3. With the JAX-style broadcasting elementwise batching rule, the replicated
        // predicate is promoted to the batched physical shape before invoking
        // `Select::select`, so the mixed-batching case succeeds with the expected per-item
        // pick.
        use crate::operations::control_flow::SelectOperation;

        let pred_type = ArrayType::scalar(DataType::Boolean);
        let operand_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)]));
        let pred_batch = ArrayBatch::new(pred_type.clone(), Array::from_f64s(pred_type, vec![1.0]), None).unwrap();
        let on_true_batch = ArrayBatch::new(operand_type.clone(), Array::vector(vec![1.0, 2.0, 3.0]), Some(0)).unwrap();
        let on_false_batch = ArrayBatch::new(operand_type, Array::vector(vec![4.0, 5.0, 6.0]), Some(0)).unwrap();

        let outputs = SelectOperation
            .batch(
                &BatchingContext::new(EagerContext::<Array>::new(), 3),
                &crate::EmptyRegionDriver,
                &[pred_batch, on_true_batch, on_false_batch],
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value().to_f64s(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_batching_rule_zero_operation_is_replicated() {
        // `ZeroOperation` takes no inputs and produces a constant of its captured type. The same
        // constant is the right value for every batch item, so the per-op rule wraps the output as
        // replicated (`batch_axis = None`) with no inserted axis.
        let scalar = ArrayType::scalar(DataType::F64);
        let operation = crate::operations::constants::ZeroOperation::new(scalar.clone());

        let outputs: Vec<ArrayBatch<Array>> = operation
            .batch(
                &BatchingContext::new(
                    EagerContext::<Array, crate::operations::constants::ConstantOperation<Array>>::new(),
                    2,
                ),
                &crate::EmptyRegionDriver,
                &[],
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::replicated());
        assert_eq!(outputs[0].r#type().into_owned(), scalar);
        assert_eq!(outputs[0].value().to_f64s(), vec![0.0]);
    }

    #[test]
    fn test_batching_rule_one_operation_is_replicated() {
        // Symmetric to `ZeroOperation`: `OneOperation` is replicated by construction.
        let scalar = ArrayType::scalar(DataType::F64);
        let operation = crate::operations::constants::OneOperation::new(scalar.clone());

        let outputs: Vec<ArrayBatch<Array>> = operation
            .batch(
                &BatchingContext::new(
                    EagerContext::<Array, crate::operations::constants::ConstantOperation<Array>>::new(),
                    2,
                ),
                &crate::EmptyRegionDriver,
                &[],
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::replicated());
        assert_eq!(outputs[0].r#type().into_owned(), scalar);
        assert_eq!(outputs[0].value().to_f64s(), vec![1.0]);
    }

    #[test]
    fn test_batching_rule_enum_zero_input_variants_delegate_to_payload_rules() {
        // The enum arms for zero-input variants delegate to the per-payload rules (interpret once under the active
        // context and wrap the outputs replicated) instead of erroring. This is the path a nested-program
        // `Zero`/`Fill` instruction takes when `BatchingContext::interpret_program` dispatches it through the enum's
        // `batch` with no inputs.
        let scalar = ArrayType::scalar(DataType::F64);
        let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 2);

        let zero = ArrayOperation::<Array>::Zero(crate::operations::constants::ZeroOperation::new(scalar.clone()));
        let outputs: Vec<ArrayBatch<Array>> = zero.batch(&context, &crate::EmptyRegionDriver, &[]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::replicated());
        assert_eq!(outputs[0].value().to_f64s(), vec![0.0]);

        let fill = ArrayOperation::<Array>::Fill(crate::operations::constants::FillOperation::new(
            scalar,
            crate::backends::scalars::Scalar::from(7.5),
        ));
        let outputs: Vec<ArrayBatch<Array>> = fill.batch(&context, &crate::EmptyRegionDriver, &[]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::replicated());
        assert_eq!(outputs[0].value().to_f64s(), vec![7.5]);
    }

    #[test]
    fn test_jacrev_through_function_using_zero_like() {
        // `f(x) = x + zero_like(x)` is functionally the identity, but exercises the
        // `ZeroLikeOperation` rule through `jacrev`'s internal Jacobian batching path. Verifies
        // that the constant-op rule composes cleanly with reverse-mode autodiff.
        let jacobian = jacrev(|x| Ok(x.clone() + x.zero_like()), Array::scalar(2.0)).unwrap();
        let block = jacobian.iter_blocks().next().unwrap();
        // d(x + 0) / dx = 1 at the scalar point.
        assert_abs_diff_eq!(block.value().values()[0], 1.0, epsilon = 1e-9);
    }

    #[test]
    fn test_jacfwd_through_function_using_one_like() {
        // `f(x) = x + one_like(x)` shifts x by a constant; the Jacobian is still 1. Exercises
        // `OneLikeOperation` through jacfwd's internal batching.
        let jacobian = EagerContext::<Array, ArrayOperation<Array>>::new()
            .jacfwd(|x| Ok(x.clone() + x.one_like()), Array::scalar(2.0))
            .unwrap();
        let block = jacobian.iter_blocks().next().unwrap();
        // d(x + 1) / dx = 1.
        assert_abs_diff_eq!(block.value().values()[0], 1.0, epsilon = 1e-9);
    }

    /// Binds `ArrayOperation::Condition` over `scalar_scale_branch(2.0)` / `scalar_scale_branch(3.0)` through the
    /// provided value's dispatch domain, feeding its predicate input with a lifted constant `true`. Generic over the
    /// value so it serves both staged closures (a [`Tracer`](crate::tracing::Tracer)) and dual-driven differentiation
    /// closures (a [`LinearizationTracer`](crate::differentiation::LinearizationTracer)).
    fn stage_constant_predicate_condition<V>(x: V) -> V
    where
        V: crate::programs::Value<Type = ArrayType>,
        V::DispatchDomain:
            crate::contexts::Context<Type = ArrayType, Constant = Array, Operation = ArrayOperation<Array>>,
    {
        let condition_regions = vec![scalar_scale_branch(2.0), scalar_scale_branch(3.0)];
        let condition = ConditionOperation::new();
        let context = x.dispatch_domain();
        let predicate = context.lift(Array::from_f64s(ArrayType::scalar(DataType::Boolean), vec![1.0])).unwrap();
        let mut outputs = context
            .bind(ArrayOperation::Condition(condition), condition_regions.clone(), &[predicate, x])
            .unwrap();
        outputs.remove(0)
    }

    /// Applies a condition whose predicate is computed from `x`, so differentiation and packed replay must retain
    /// and execute both attached branch regions instead of folding the condition from a captured predicate.
    fn stage_runtime_predicate_condition<V>(x: V) -> Result<V, crate::ProgramError>
    where
        V: crate::programs::Value<Type = ArrayType>,
        V::DispatchDomain:
            crate::contexts::Context<Type = ArrayType, Constant = Array, Operation = ArrayOperation<Array>>,
    {
        let context = x.dispatch_domain();
        let zero = context.lift(Array::scalar(0.0))?;
        let mut predicates = context.bind(
            ArrayOperation::Compare(CompareOperation::new(ComparisonDirection::GreaterThan)),
            Vec::new(),
            &[x.clone(), zero],
        )?;
        let predicate = predicates.remove(0);
        let mut outputs = context.bind(
            ArrayOperation::Condition(ConditionOperation::new()),
            vec![scalar_scale_branch(2.0), scalar_scale_branch(3.0)],
            &[predicate, x],
        )?;
        Ok(outputs.remove(0))
    }

    #[test]
    fn test_condition_composes_with_jacrev_and_jacfwd() {
        // The constant `true` predicate selects the scale-by-2 branch, so both autodiff transforms must report a
        // derivative of 2 by linearizing the selected branch through the `ArrayOperation::Condition` JVP dispatch.
        let jacobian = jacrev(|x| Ok(stage_constant_predicate_condition(x)), Array::scalar(4.0)).unwrap();
        assert_abs_diff_eq!(jacobian.iter_blocks().next().unwrap().value().values()[0], 2.0, epsilon = 1e-9);

        let jacobian = EagerContext::<Array, ArrayOperation<Array>>::new()
            .jacfwd(|x| Ok(stage_constant_predicate_condition(x)), Array::scalar(4.0))
            .unwrap();
        assert_abs_diff_eq!(jacobian.iter_blocks().next().unwrap().value().values()[0], 2.0, epsilon = 1e-9);
    }

    #[test]
    fn test_dense_jacobians_replay_runtime_condition_regions() {
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        for (input, expected) in [(4.0, 2.0), (-4.0, 3.0)] {
            let forward = context.jacfwd(stage_runtime_predicate_condition, Array::scalar(input)).unwrap();
            let reverse = context.jacrev(stage_runtime_predicate_condition, Array::scalar(input)).unwrap();
            assert_abs_diff_eq!(forward.iter_blocks().next().unwrap().value().values()[0], expected, epsilon = 1e-9,);
            assert_abs_diff_eq!(reverse.iter_blocks().next().unwrap().value().values()[0], expected, epsilon = 1e-9,);
        }
    }

    /// Builds the `while (x < 8) { x = x + x }` doubling-loop fixture used by the while differentiation tests,
    /// returning the payload-free operation together with its condition and body region programs (in region order).
    fn doubling_while_operation() -> (
        crate::operations::control_flow::WhileOperation,
        Vec<crate::Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>>>,
    ) {
        use crate::operations::compare::ComparisonDirection;
        type TestOp = ArrayOperation<Array>;

        let scalar_f64 = ArrayType::scalar(DataType::F64);
        let mut condition_builder = ProgramBuilder::<Array, TestOp>::new();
        let condition_state = condition_builder.add_input(scalar_f64.clone());
        let threshold = condition_builder.add_constant(Array::scalar(8.0));
        let predicate = condition_builder
            .add_instruction(
                CompareOperation::new(ComparisonDirection::LessThan),
                Vec::new(),
                vec![condition_state, threshold],
            )
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<Array>, Vec<Array>>(vec![predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut body_builder = ProgramBuilder::<Array, TestOp>::new();
        let body_state = body_builder.add_input(scalar_f64);
        let doubled = body_builder.add_instruction(AddOperation, Vec::new(), vec![body_state, body_state]).unwrap()[0];
        let body = body_builder
            .build::<Vec<Array>, Vec<Array>>(vec![doubled], vec![Placeholder], vec![Placeholder])
            .unwrap();

        (crate::operations::control_flow::WhileOperation::new(), vec![condition, body])
    }

    #[test]
    fn test_while_jvp_propagates_tangents_through_staged_linear_loop() {
        // `while (x < 8) { x = x + x }` starting at `x = 1` doubles three times, so the primal output is 8 and the
        // forward-mode derivative is 2^3 = 8. Exercises the `ArrayOperation::While` JVP dispatch in an eager domain:
        // the rule stages the doubled-state linear loop, which direct JVP execution interprets immediately.
        let (while_operation, while_regions) = doubling_while_operation();
        let (primal, tangent) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .jvp(
                move |x| {
                    let mut outputs = x.context().bind(
                        ArrayOperation::While(while_operation),
                        while_regions.clone(),
                        &[x.clone()],
                    )?;
                    Ok(outputs.remove(0))
                },
                Array::scalar(1.0),
                Array::scalar(1.0),
            )
            .unwrap();
        assert_eq!(primal.to_f64s(), vec![8.0]);
        assert_eq!(tangent.to_f64s(), vec![8.0]);
    }

    #[test]
    fn test_jacfwd_through_while_batches_basis_tangents() {
        // `jacfwd` linearizes the loop once into a pushforward containing a staged linear while, then replays
        // the batched basis tangents through it: the pushforward is instantiated into a direct linear program and
        // interpreted over the batching rules with concrete values, so the linear while runs once over the stacked
        // basis tangents. The derivative of the doubling loop at `x = 1` is `2^3 = 8`.
        let (while_operation, while_regions) = doubling_while_operation();
        let jacobian = EagerContext::<Array, ArrayOperation<Array>>::new()
            .jacfwd(
                move |x| {
                    let mut outputs = x.context().bind(
                        ArrayOperation::While(while_operation),
                        while_regions.clone(),
                        &[x.clone()],
                    )?;
                    Ok(outputs.remove(0))
                },
                Array::scalar(1.0),
            )
            .unwrap();
        assert_abs_diff_eq!(jacobian.iter_blocks().next().unwrap().value().values()[0], 8.0, epsilon = 1e-9);
    }

    #[test]
    fn test_jacrev_through_bounded_while_batches_basis_cotangents() {
        let (while_operation, while_regions) = doubling_while_operation();
        let while_operation = while_operation.with_iteration_bound(5).unwrap();
        let jacobian = EagerContext::<Array, ArrayOperation<Array>>::new()
            .jacrev(
                move |x| {
                    let mut outputs = x.context().bind(
                        ArrayOperation::While(while_operation),
                        while_regions.clone(),
                        &[x.clone()],
                    )?;
                    Ok(outputs.remove(0))
                },
                Array::scalar(1.0),
            )
            .unwrap();
        assert_abs_diff_eq!(jacobian.iter_blocks().next().unwrap().value().values()[0], 8.0, epsilon = 1e-9);
    }

    #[test]
    fn test_while_value_and_grad_computes_gradient_through_bounded_loop() {
        // Eager reverse mode through a *bounded* while loop whose bound does not bind: the doubling loop at `x = 1`
        // runs three iterations, below the bound of 5, so the eager rule executes all three iterations (the bound only
        // truncates once it is reached). Linearization records a straight-line pushforward, and locally `f(x) = 8 x`, so
        // the value is 8 and the gradient is 8.
        let (while_operation, while_regions) = doubling_while_operation();
        let while_operation = while_operation.with_iteration_bound(5).unwrap();
        let (value, gradient) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .value_and_gradient(
                move |x| {
                    let mut outputs = x
                        .context()
                        .bind(ArrayOperation::While(while_operation), while_regions.clone(), &[x.clone()])
                        .unwrap();
                    outputs.remove(0)
                },
                Array::scalar(1.0),
            )
            .unwrap();
        assert_eq!(value.to_f64s(), vec![8.0]);
        assert_eq!(gradient.to_f64s(), vec![8.0]);
    }

    #[test]
    fn test_while_value_and_grad_computes_gradient_through_unrolled_loop() {
        // Eager reverse mode through an *unbounded* while loop: the JVP rule executes the doubling loop at `x = 1`
        // (three iterations), so linearization records a straight-line linear program that transposes. Locally
        // `f(x) = 8 x`, so the value is 8 and the gradient is 8. JAX cannot do this even under eager execution,
        // because it always traces `while_loop`.
        let (while_operation, while_regions) = doubling_while_operation();
        let (value, gradient) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .value_and_gradient(
                move |x| {
                    let mut outputs = x
                        .context()
                        .bind(ArrayOperation::While(while_operation), while_regions.clone(), &[x.clone()])
                        .unwrap();
                    outputs.remove(0)
                },
                Array::scalar(1.0),
            )
            .unwrap();
        assert_eq!(value.to_f64s(), vec![8.0]);
        assert_eq!(gradient.to_f64s(), vec![8.0]);
    }

    #[test]
    fn test_while_vjp_computes_cotangents_through_bounded_loop() {
        // Eager `vjp` through a *bounded* while loop whose bound does not bind: the doubling loop at `x = 1` runs
        // three iterations, below the bound of 5, so the eager rule executes the loop in full and linearization
        // transposes the resulting straight-line pushforward into a reusable pullback (no `while` remains). The loop
        // is locally `f(x) = 8 x`, so every output cotangent is scaled by 8.
        let (while_operation, while_regions) = doubling_while_operation();
        let while_operation = while_operation.with_iteration_bound(5).unwrap();
        let (output, pullback) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .vjp(
                move |x| {
                    let mut outputs = x.context().bind(
                        ArrayOperation::While(while_operation),
                        while_regions.clone(),
                        &[x.clone()],
                    )?;
                    Ok(outputs.remove(0))
                },
                Array::scalar(1.0),
            )
            .unwrap();
        let (pullback, residuals) = pullback.into_parts();
        assert_eq!(output.to_f64s(), vec![8.0]);
        assert!(!pullback.to_string().contains("while"), "{pullback}");
        let pullback_inputs = |cotangent: Array| {
            let mut inputs = vec![cotangent];
            inputs.extend(residuals.iter().cloned());
            inputs
        };
        assert_eq!(
            pullback.interpret(pullback_inputs(Array::scalar(1.0))).map(|cotangents| cotangents[0].to_f64s()),
            Ok(vec![8.0]),
        );
        assert_eq!(
            pullback.interpret(pullback_inputs(Array::scalar(5.0))).map(|cotangents| cotangents[0].to_f64s()),
            Ok(vec![40.0]),
        );
    }

    #[test]
    fn test_while_vjp_computes_cotangents_through_unrolled_loop() {
        // Eager `vjp` transposes the unrolled straight-line pushforward of an *unbounded* loop into a reusable
        // pullback: the doubling loop at `x = 1` is locally `f(x) = 8 x`, so every output cotangent is scaled by 8.
        let (while_operation, while_regions) = doubling_while_operation();
        let (output, pullback) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .vjp(
                move |x| {
                    let mut outputs = x.context().bind(
                        ArrayOperation::While(while_operation),
                        while_regions.clone(),
                        &[x.clone()],
                    )?;
                    Ok(outputs.remove(0))
                },
                Array::scalar(1.0),
            )
            .unwrap();
        let (pullback, residuals) = pullback.into_parts();
        assert_eq!(output.to_f64s(), vec![8.0]);
        let pullback_inputs = |cotangent: Array| {
            let mut inputs = vec![cotangent];
            inputs.extend(residuals.iter().cloned());
            inputs
        };
        assert_eq!(
            pullback.interpret(pullback_inputs(Array::scalar(1.0))).map(|cotangents| cotangents[0].to_f64s()),
            Ok(vec![8.0]),
        );
        assert_eq!(
            pullback.interpret(pullback_inputs(Array::scalar(5.0))).map(|cotangents| cotangents[0].to_f64s()),
            Ok(vec![40.0]),
        );
    }

    /// Builds the three-iteration cumulative-product [`ScanOperation`] (body `[carry, x] -> [carry * x, carry * x]`)
    /// used by the scan differentiation tests, optionally visiting the iterations in reverse order.
    fn product_scan_operation(
        reverse: bool,
    ) -> (
        crate::operations::control_flow::ScanOperation<Array>,
        crate::Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>>,
    ) {
        type TestOp = ArrayOperation<Array>;
        let mut body_builder = ProgramBuilder::<Array, TestOp>::new();
        let carry = body_builder.add_input(ArrayType::scalar(DataType::F64));
        let x = body_builder.add_input(ArrayType::scalar(DataType::F64));
        let product = body_builder.add_instruction(MulOperation, Vec::new(), vec![carry, x]).unwrap()[0];
        let body = body_builder
            .build::<Vec<Array>, Vec<Array>>(
                vec![product, product],
                vec![Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();
        (crate::operations::control_flow::ScanOperation::<Array>::new(1, 3).with_reverse(reverse), body)
    }

    #[test]
    fn test_scan_value_and_grad_computes_gradient_through_reversed_linear_scan() {
        // The headline scan capability: end-to-end reverse mode. `f(init, xs)` is the final carry of the
        // cumulative-product scan, so `f = init * xs[0] * xs[1] * xs[2] = 24` at `init = 1, xs = [2, 3, 4]`, with
        // gradient `24` w.r.t. `init` and `[12, 8, 6]` w.r.t. `xs`. The pullback runs the transposed linear scan
        // (same residual stacks, `reverse` flipped) — the static trip count is what makes this total, where the
        // staged linear `while` rejects transposition.
        let (scan, scan_body) = product_scan_operation(false);
        let (value, (init_gradient, xs_gradient)) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .value_and_gradient(
                move |(init, xs)| {
                    let mut outputs = init
                        .context()
                        .bind(ArrayOperation::Scan(scan), vec![scan_body.clone()], &[init.clone(), xs.clone()])
                        .unwrap();
                    outputs.remove(0)
                },
                (Array::scalar(1.0), Array::vector(vec![2.0, 3.0, 4.0])),
            )
            .unwrap();
        assert_eq!(value.to_f64s(), vec![24.0]);
        assert_eq!(init_gradient.to_f64s(), vec![24.0]);
        assert_eq!(xs_gradient.to_f64s(), vec![12.0, 8.0, 6.0]);
    }

    #[test]
    fn test_scan_vjp_stages_reversed_linear_scan_in_reusable_pullback() {
        // `vjp` through the cumulative-product scan produces a reusable pullback containing the transposed linear
        // scan: the same residual stacks with `reverse` flipped to `true`. Each cotangent seed scales the
        // hand-computed gradients `(24, [12, 8, 6])`.
        let (scan, scan_body) = product_scan_operation(false);
        let (output, pullback) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .vjp(
                move |(init, xs)| {
                    let mut outputs = init.context().bind(
                        ArrayOperation::Scan(scan),
                        vec![scan_body.clone()],
                        &[init.clone(), xs.clone()],
                    )?;
                    Ok(outputs.remove(0))
                },
                (Array::scalar(1.0), Array::vector(vec![2.0, 3.0, 4.0])),
            )
            .unwrap();
        let (pullback, residuals) = pullback.into_parts();
        assert_eq!(output.to_f64s(), vec![24.0]);
        let rendered_pullback = pullback.to_string();
        assert!(rendered_pullback.contains("scan"), "{rendered_pullback}");
        assert!(rendered_pullback.contains("reverse=true"), "{rendered_pullback}");
        let pullback_inputs = |cotangent: Array| {
            let mut inputs = vec![cotangent];
            inputs.extend(residuals.iter().cloned());
            inputs
        };
        let cotangents = pullback.interpret(pullback_inputs(Array::scalar(1.0))).unwrap();
        assert_eq!(cotangents[0].to_f64s(), vec![24.0]);
        assert_eq!(cotangents[1].to_f64s(), vec![12.0, 8.0, 6.0]);
        let cotangents = pullback.interpret(pullback_inputs(Array::scalar(2.0))).unwrap();
        assert_eq!(cotangents[0].to_f64s(), vec![48.0]);
        assert_eq!(cotangents[1].to_f64s(), vec![24.0, 16.0, 12.0]);
    }

    #[test]
    fn test_dense_jacobians_replay_scan_regions() {
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let primals = (Array::scalar(1.0), Array::vector(vec![2.0, 3.0, 4.0]));
        let (scan, body) = product_scan_operation(false);
        let forward = context
            .jacfwd(
                move |(initial, values)| {
                    let mut outputs = initial.context().bind(
                        ArrayOperation::Scan(scan),
                        vec![body.clone()],
                        &[initial.clone(), values],
                    )?;
                    Ok(outputs.remove(0))
                },
                primals.clone(),
            )
            .unwrap();
        let (scan, body) = product_scan_operation(false);
        let reverse = context
            .jacrev(
                move |(initial, values)| {
                    let mut outputs = initial.context().bind(
                        ArrayOperation::Scan(scan),
                        vec![body.clone()],
                        &[initial.clone(), values],
                    )?;
                    Ok(outputs.remove(0))
                },
                primals,
            )
            .unwrap();

        for jacobian in [forward, reverse] {
            let blocks = jacobian.iter_blocks().collect::<Vec<_>>();
            assert_eq!(blocks.len(), 2);
            assert_eq!(blocks[0].value().values(), &[24.0]);
            assert_eq!(blocks[1].value().values(), &[12.0, 8.0, 6.0]);
        }
    }

    #[test]
    fn test_hessian_replays_scan_regions() {
        // For `f(initial, values) = initial * product(values)` at `(1, [2, 3, 4])`, all same-variable second
        // derivatives vanish. Mixed derivatives with `initial` are the products excluding the corresponding value,
        // and mixed derivatives between values are `initial` times the remaining value. This exercises nested
        // forward-over-reverse replay of the scan body rather than only first-order region replay.
        let (scan, body) = product_scan_operation(false);
        let hessian = EagerContext::<Array, ArrayOperation<Array>>::new()
            .hessian(
                move |(initial, values)| {
                    let mut outputs = initial.context().bind(
                        ArrayOperation::Scan(scan),
                        vec![body.clone()],
                        &[initial.clone(), values],
                    )?;
                    Ok(outputs.remove(0))
                },
                (Array::scalar(1.0), Array::vector(vec![2.0, 3.0, 4.0])),
            )
            .unwrap();

        let blocks = hessian.iter_blocks().collect::<Vec<_>>();
        assert_eq!(blocks.len(), 4);
        assert_eq!(blocks[0].value().values(), &[0.0]);
        assert_eq!(blocks[1].value().values(), &[12.0, 8.0, 6.0]);
        assert_eq!(blocks[2].value().values(), &[12.0, 8.0, 6.0]);
        assert_eq!(
            blocks[3].value().values(),
            &[
                0.0, 4.0, 3.0, //
                4.0, 0.0, 2.0, //
                3.0, 2.0, 0.0, //
            ],
        );
    }

    #[test]
    fn test_reversed_scan_jvp_and_grad_align_items() {
        // Pins the alignment invariant for `reverse = true`: the primal scan visits iterations from the back while
        // output iteration `i` stays aligned with input iteration `i`, and the linear scan runs with the same direction so
        // residual iteration `i` is consumed exactly when tangent iteration `i` is processed. The reversed cumulative
        // product has `ys = [x0 x1 x2, x1 x2, x2] = [24, 12, 4]`, so a unit tangent on `x1` gives
        // `dys = [x0 x2, x2, 0] = [8, 4, 0]`.
        let (scan, scan_body) = product_scan_operation(true);
        let ((carry, ys), (carry_tangent, ys_tangent)) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .jvp(
                move |(init, xs)| {
                    let mut outputs = init.context().bind(
                        ArrayOperation::Scan(scan),
                        vec![scan_body.clone()],
                        &[init.clone(), xs.clone()],
                    )?;
                    let ys = outputs.remove(1);
                    Ok((outputs.remove(0), ys))
                },
                (Array::scalar(1.0), Array::vector(vec![2.0, 3.0, 4.0])),
                (Array::scalar(0.0), Array::vector(vec![0.0, 1.0, 0.0])),
            )
            .unwrap();
        assert_eq!(carry.to_f64s(), vec![24.0]);
        assert_eq!(ys.to_f64s(), vec![24.0, 12.0, 4.0]);
        assert_eq!(carry_tangent.to_f64s(), vec![8.0]);
        assert_eq!(ys_tangent.to_f64s(), vec![8.0, 4.0, 0.0]);

        // Reverse mode through the reversed scan flips `reverse` back to `false` in the pullback and produces the
        // same product-rule gradients (multiplication commutes across the visit order).
        let (scan, scan_body) = product_scan_operation(true);
        let (output, pullback) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .vjp(
                move |(init, xs)| {
                    let mut outputs = init.context().bind(
                        ArrayOperation::Scan(scan),
                        vec![scan_body.clone()],
                        &[init.clone(), xs.clone()],
                    )?;
                    Ok(outputs.remove(0))
                },
                (Array::scalar(1.0), Array::vector(vec![2.0, 3.0, 4.0])),
            )
            .unwrap();
        let (pullback, residuals) = pullback.into_parts();
        assert_eq!(output.to_f64s(), vec![24.0]);
        let rendered_pullback = pullback.to_string();
        assert!(rendered_pullback.contains("reverse=false"), "{rendered_pullback}");
        let mut pullback_inputs = vec![Array::scalar(1.0)];
        pullback_inputs.extend(residuals);
        let cotangents = pullback.interpret(pullback_inputs).unwrap();
        assert_eq!(cotangents[0].to_f64s(), vec![24.0]);
        assert_eq!(cotangents[1].to_f64s(), vec![12.0, 8.0, 6.0]);
    }

    #[test]
    fn test_array_operation_condition_interprets_runtime_predicate() {
        let condition_regions = vec![scalar_scale_branch(2.0), scalar_scale_branch(3.0)];
        let condition = ConditionOperation::new();
        let operation = ArrayOperation::Condition(condition);

        let predicate = Array::from_f64s(ArrayType::scalar(DataType::Boolean), vec![0.0]);
        assert_eq!(
            EagerContext::<Array, ArrayOperation<Array>>::new()
                .bind(operation, condition_regions, &[predicate, Array::scalar(4.0)])
                .map(|outputs| outputs[0].to_f64s()[0]),
            Ok(12.0),
        );
    }

    #[test]
    fn test_condition_vjp_computes_branch_cotangents_for_runtime_predicate() {
        // f(p, x) = if p { 2 * x } else { 3 * x }. Reverse mode composes through the total linear-condition
        // transpose rule: the pullback runs the transposed branch program selected by the captured predicate, so
        // the operand cotangent is 2 * output cotangent at a TRUE-predicate primal point and 3 * output cotangent
        // at a FALSE one. The Boolean predicate has no tangent space, so its cotangent slot is always zero.
        let (output, pullback) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .vjp(
                |(predicate, operand)| {
                    let condition_regions = vec![scalar_scale_branch(2.0), scalar_scale_branch(3.0)];
                    let condition = ConditionOperation::new();
                    let mut outputs = predicate.context().bind(
                        ArrayOperation::Condition(condition),
                        condition_regions.clone(),
                        &[predicate.clone(), operand.clone()],
                    )?;
                    Ok(outputs.remove(0))
                },
                (Array::from_f64s(ArrayType::scalar(DataType::Boolean), vec![1.0]), Array::scalar(4.0)),
            )
            .unwrap();
        let (pullback, residuals) = pullback.into_parts();
        assert_eq!(output.to_f64s(), vec![8.0]);
        let mut pullback_inputs = vec![Array::scalar(5.0)];
        pullback_inputs.extend(residuals);
        let cotangents = pullback.interpret(pullback_inputs).unwrap();
        assert_eq!(cotangents[1].to_f64s(), vec![10.0]);
        assert_eq!(cotangents[0].values(), &[Scalar::Zero]);

        let (output, pullback) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .vjp(
                |(predicate, operand)| {
                    let condition_regions = vec![scalar_scale_branch(2.0), scalar_scale_branch(3.0)];
                    let condition = ConditionOperation::new();
                    let mut outputs = predicate.context().bind(
                        ArrayOperation::Condition(condition),
                        condition_regions.clone(),
                        &[predicate.clone(), operand.clone()],
                    )?;
                    Ok(outputs.remove(0))
                },
                (Array::from_f64s(ArrayType::scalar(DataType::Boolean), vec![0.0]), Array::scalar(4.0)),
            )
            .unwrap();
        let (pullback, residuals) = pullback.into_parts();
        assert_eq!(output.to_f64s(), vec![12.0]);
        let mut pullback_inputs = vec![Array::scalar(5.0)];
        pullback_inputs.extend(residuals);
        let cotangents = pullback.interpret(pullback_inputs).unwrap();
        assert_eq!(cotangents[1].to_f64s(), vec![15.0]);
        assert_eq!(cotangents[0].values(), &[Scalar::Zero]);
    }
}
