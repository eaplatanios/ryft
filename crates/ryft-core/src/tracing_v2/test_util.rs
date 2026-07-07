use crate::operations::arithmetic::MulOperation;
use crate::parameters::Placeholder;
use crate::programs::ProgramBuilder;
use crate::scalars::Scalar;
use crate::tests::TestArray;
use crate::tracing_v2::ArrayOperation;
use crate::types::{ArrayType, DataType, Typed};

/// Asserts that `actual` is within absolute tolerance `1e-9` of `expected`.
pub(crate) fn assert_close(actual: f64, expected: f64) {
    let delta = (actual - expected).abs();
    assert!(delta <= 1e-9, "expected {actual} ~= {expected}; absolute error {delta} exceeded tolerance");
}

/// Asserts that the floating-point payload of `actual` is within absolute tolerance `1e-9` of `expected`. This is the
/// [`Scalar`] counterpart of [`assert_close`] used by the scalar-domain differentiation tests, whose results are
/// [`Scalar`] values rather than bare `f64`s. It accepts any floating-point [`Scalar`] variant and panics on a
/// non-floating-point variant, which would indicate the test produced an unexpected data type.
pub(crate) fn assert_scalar_close(actual: Scalar, expected: f64) {
    let value = match actual {
        Scalar::BF16(value) => value.to_f64(),
        Scalar::F16(value) => value.to_f64(),
        Scalar::F32(value) => value as f64,
        Scalar::F64(value) => value,
        other => panic!("expected a floating-point scalar but got {}", other.r#type().into_owned()),
    };
    assert_close(value, expected);
}

/// Builds a single-input flat program that scales its scalar input by `factor`, multiplying the input by a captured
/// constant carrying `factor`.
pub(crate) fn scalar_scale_branch(
    factor: f64,
) -> crate::programs::Program<TestArray, ArrayOperation<TestArray>, Vec<TestArray>, Vec<TestArray>> {
    let mut builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
    let input = builder.add_input(ArrayType::scalar(DataType::F64));
    let factor = builder.add_constant(TestArray::scalar(factor));
    let output = builder.add_instruction(MulOperation, vec![input, factor]).unwrap()[0];
    builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::batching::ArrayBatch;
    use crate::batching::BatchAxis;
    use crate::batching::BatchableOperation;
    use crate::contexts::{Context, EagerContext, StagingContext};
    use crate::interpretation::InterpretableOperation;
    use crate::operations::arithmetic::{AddOperation, MulOperation, SubOperation};
    use crate::operations::compare::CompareOperation;
    use crate::operations::constants::{OneLike, OneLikeOperation, ZeroLike, ZeroLikeOperation};
    use crate::operations::control_flow::ConditionOperation;
    use crate::operations::trigonometric::Sin;
    use crate::parameters::Placeholder;
    use crate::programs::ProgramBuilder;
    use crate::tracing_v2::{DifferentiableDomainExtension, Differentiate, jacrev};
    use crate::types::{Shape, Size, Typed};

    use super::*;

    #[test]
    fn test_dot_batches_mixed_lhs_batched_rhs_replicated() {
        // LHS is mapped at axis 0 with per-item shape [3]; RHS is replicated with shape [3].
        // Per-item semantics: dot(lhs_row, rhs) over the shared K=3 dimension. The batching rule
        // should broadcast the RHS to gain a singleton batch axis at position 0, then thread the
        // batch axis through `lift_dot_dimensions`.
        use crate::tracing_v2::operations::dot::{DotDimensionNumbers, DotOperation};
        let lhs = {
            let value = TestArray::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
            ArrayBatch::new(value.r#type().into_owned(), value, Some(0))
        }
        .unwrap();
        let rhs = ArrayBatch::replicated(TestArray::vector(vec![10.0, 100.0, 1000.0]));
        let dimensions = DotDimensionNumbers::new(vec![0], vec![0], vec![], vec![]);
        let outputs = DotOperation::new(dimensions).batch(&EagerContext::<TestArray>::new(), &[lhs, rhs]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        // Batch item 0: 1*10 + 2*100 + 3*1000 = 3210; batch item 1: 4*10 + 5*100 + 6*1000 = 6540.
        assert_eq!(outputs[0].value().values(), &[3210.0, 6540.0]);
    }

    #[test]
    fn test_reduce_sum_jvp_linearizes_to_itself() {
        // Verify the linear reduce rule directly over a concrete tangent value.
        use crate::tracing_v2::operations::reduce::{ReduceOperation, ReductionKind};
        let primal = TestArray::vector(vec![1.0, 2.0, 3.0, 4.0]);
        let tangent_value = TestArray::vector(vec![0.5, 0.5, 0.5, 0.5]);

        let operation = ReduceOperation::new(vec![0], ReductionKind::Sum);

        // Primal: reduce(x, [0], Sum) on `TestArray` directly.
        let primal_output = operation
            .interpret(&EagerContext::<TestArray>::new(), std::slice::from_ref(&primal))
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        assert_eq!(primal_output.values(), &[10.0]);

        // Tangent: linearizes to itself (Sum is linear), so the tangent of the reduce is the
        // reduce of the tangent.
        let tangent_outputs = operation
            .interpret(&EagerContext::<TestArray>::new(), std::slice::from_ref(&tangent_value))
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
        type TestOp = ArrayOperation<TestArray>;

        let scalar_f64 = ArrayType::scalar(DataType::F64);

        // Condition program: state -> (state > 0). Returns a scalar Boolean.
        let mut condition_builder = ProgramBuilder::<TestArray, TestOp>::new();
        let cond_input = condition_builder.add_input(scalar_f64.clone());
        let cond_zero = condition_builder.add_instruction(ZeroLikeOperation, vec![cond_input]).unwrap()[0];
        let cond_output = condition_builder
            .add_instruction(CompareOperation::new(ComparisonDirection::GreaterThan), vec![cond_input, cond_zero])
            .unwrap()[0];
        let condition: Program<TestArray, TestOp, Vec<TestArray>, Vec<TestArray>> = condition_builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![cond_output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        // Body program: state -> state - 1.
        let mut body_builder = ProgramBuilder::<TestArray, TestOp>::new();
        let body_input = body_builder.add_input(scalar_f64);
        let body_one = body_builder.add_instruction(OneLikeOperation, vec![body_input]).unwrap()[0];
        let body_output = body_builder.add_instruction(SubOperation, vec![body_input, body_one]).unwrap()[0];
        let body: Program<TestArray, TestOp, Vec<TestArray>, Vec<TestArray>> = body_builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![body_output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let while_op = WhileOperation::<TestArray, TestOp>::new(condition, body).unwrap();
        let context = EagerContext::<TestArray, TestOp>::new();

        let initial_state = {
            let value = TestArray::vector(vec![3.0, 1.0, 2.0]);
            ArrayBatch::new(value.r#type().into_owned(), value, Some(0))
        }
        .unwrap();
        let outputs = while_op.batch(&context, &[initial_state]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        // Each batch item terminates when its value reaches 0; inactive batch items retain their last value.
        assert_eq!(outputs[0].value().values(), &[0.0, 0.0, 0.0]);

        // A semantic iteration bound truncates the batched loop too: every batch item performs at most two body
        // applications, so batch item 0 (initial 3.0) is cut off at 1.0 while the other batch items terminate through
        // their own predicates first.
        let bounded_while_op = while_op.with_iteration_bound(2).unwrap();
        let initial_state = {
            let value = TestArray::vector(vec![3.0, 1.0, 2.0]);
            ArrayBatch::new(value.r#type().into_owned(), value, Some(0))
        }
        .unwrap();
        let outputs = bounded_while_op.batch(&context, &[initial_state]).unwrap();
        assert_eq!(outputs[0].value().values(), &[1.0, 0.0, 0.0]);

        // The replicated batched loop respects the bound as well: an unbatched initial state of 5.0 stops at 3.0.
        let initial_state = ArrayBatch::replicated(TestArray::scalar(5.0));
        let outputs = bounded_while_op.batch(&context, &[initial_state]).unwrap();
        assert_eq!(outputs[0].value().values(), &[3.0]);
    }

    #[test]
    fn test_jacfwd_batches_basis_tangents() {
        let jacobian = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .jacfwd(
                |(x, y)| Ok((x.clone() * y.clone() + x.sin()?, x + y)),
                (TestArray::scalar(2.0), TestArray::scalar(3.0)),
            )
            .unwrap();

        let (row_0, row_1) = jacobian.rows();
        let (block_00, block_01) = row_0.partials();
        let (block_10, block_11) = row_1.partials();

        assert_eq!(block_00.output_shape(), &[] as &[usize]);
        assert_eq!(block_00.input_shape(), &[] as &[usize]);

        assert_close(block_00.values()[0], 3.0 + 2.0f64.cos());
        assert_close(block_01.values()[0], 2.0);
        assert_close(block_10.values()[0], 1.0);
        assert_close(block_11.values()[0], 1.0);
    }

    #[test]
    fn test_jacrev_batches_basis_cotangents() {
        let jacobian = jacrev(
            &EagerContext::<TestArray, ArrayOperation<TestArray>>::new(),
            |(x, y)| Ok((x.clone() * y.clone() + x.sin()?, x + y)),
            (TestArray::scalar(2.0), TestArray::scalar(3.0)),
        )
        .unwrap();

        let (row_0, row_1) = jacobian.rows();
        let (block_00, block_01) = row_0.partials();
        let (block_10, block_11) = row_1.partials();

        assert_close(block_00.values()[0], 3.0 + 2.0f64.cos());
        assert_close(block_01.values()[0], 2.0);
        assert_close(block_10.values()[0], 1.0);
        assert_close(block_11.values()[0], 1.0);
    }

    #[test]
    fn test_jacfwd_iter_blocks_yields_each_output_input_pair() {
        let jacobian = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .jacfwd(
                |(x, y)| Ok((x.clone() * y.clone() + x.sin()?, x + y)),
                (TestArray::scalar(2.0), TestArray::scalar(3.0)),
            )
            .unwrap();

        let triples = jacobian
            .iter_blocks()
            .map(|(output_path, input_path, block)| {
                (output_path.to_string(), input_path.to_string(), block.values()[0])
            })
            .collect::<Vec<_>>();

        assert_eq!(triples.len(), 4);
        assert_eq!(triples[0].0, "$.0");
        assert_eq!(triples[0].1, "$.0");
        assert_close(triples[0].2, 3.0 + 2.0f64.cos());
        assert_eq!(triples[1].0, "$.0");
        assert_eq!(triples[1].1, "$.1");
        assert_close(triples[1].2, 2.0);
        assert_eq!(triples[2].0, "$.1");
        assert_eq!(triples[2].1, "$.0");
        assert_close(triples[2].2, 1.0);
        assert_eq!(triples[3].0, "$.1");
        assert_eq!(triples[3].1, "$.1");
        assert_close(triples[3].2, 1.0);
    }

    #[test]
    fn test_hessian_accepts_original_scalar_function() {
        let hessian = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .hessian(|(x, y)| x.clone() * y + x.sin().unwrap(), (TestArray::scalar(2.0), TestArray::scalar(3.0)))
            .unwrap();

        let (row_0, row_1) = hessian.rows();
        let (block_00, block_01) = row_0.partials();
        let (block_10, block_11) = row_1.partials();

        assert_close(block_00.values()[0], -2.0f64.sin());
        assert_close(block_01.values()[0], 1.0);
        assert_close(block_10.values()[0], 1.0);
        assert_close(block_11.values()[0], 0.0);
    }

    #[test]
    fn test_jacfwd_handles_function_with_independent_outputs() {
        // f(x, y) = (x*y + sin(x), y, x + y) — output[1] is independent of x.
        let jacobian = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .jacfwd(
                |(x, y)| Ok((x.clone() * y.clone() + x.sin()?, y.clone(), x + y)),
                (TestArray::scalar(2.0), TestArray::scalar(3.0)),
            )
            .unwrap();

        let triples = jacobian
            .iter_blocks()
            .map(|(output_path, input_path, block)| {
                (output_path.to_string(), input_path.to_string(), block.values()[0])
            })
            .collect::<Vec<_>>();

        // 3 outputs * 2 inputs = 6 blocks
        assert_eq!(triples.len(), 6);
        // d(x*y + sin(x))/dx = y + cos(x) = 3 + cos(2)
        assert_close(triples[0].2, 3.0 + 2.0f64.cos());
        // d(x*y + sin(x))/dy = x = 2
        assert_close(triples[1].2, 2.0);
        // dy/dx = 0  (independent of x — exercise the all-zero short-circuit downstream)
        assert_close(triples[2].2, 0.0);
        // dy/dy = 1
        assert_close(triples[3].2, 1.0);
        // d(x + y)/dx = 1
        assert_close(triples[4].2, 1.0);
        // d(x + y)/dy = 1
        assert_close(triples[5].2, 1.0);
    }

    /// Builds a replicated scalar Boolean predicate batch with the provided truth value.
    fn replicated_predicate(value: bool) -> ArrayBatch<TestArray> {
        ArrayBatch::replicated(TestArray::new(
            ArrayType::scalar(DataType::Boolean),
            vec![if value { 1.0 } else { 0.0 }],
        ))
    }

    #[test]
    fn test_condition_batches_replicated_true_predicate_over_array_batches() {
        use crate::batching::ArrayBatch;
        use crate::batching::BatchAxis;
        use crate::batching::BatchableOperation;

        // A replicated `true` predicate selects scalar_scale_branch(2.0). Pass a 3-item batched operand and
        // verify each batch item is independently scaled by 2.
        let condition = ConditionOperation::new(scalar_scale_branch(2.0), scalar_scale_branch(3.0)).unwrap();
        let operation = ArrayOperation::Condition(Box::new(condition));
        let context = EagerContext::<TestArray, ArrayOperation<TestArray>>::new();

        let batched_input = {
            let value = TestArray::vector(vec![1.0, 4.0, 9.0]);
            ArrayBatch::new(value.r#type().into_owned(), value, Some(0))
        }
        .unwrap();
        let outputs = operation.batch(&context, &[replicated_predicate(true), batched_input]).unwrap();
        assert_eq!(outputs.len(), 1);
        let output_batch = &outputs[0];
        assert_eq!(output_batch.batch_axis(), BatchAxis::new(0));
        assert_eq!(output_batch.value().values, vec![2.0, 8.0, 18.0]);
    }

    #[test]
    fn test_condition_batches_false_branch_when_replicated_predicate_is_false() {
        use crate::batching::ArrayBatch;
        use crate::batching::BatchAxis;
        use crate::batching::BatchableOperation;

        let condition = ConditionOperation::new(scalar_scale_branch(2.0), scalar_scale_branch(3.0)).unwrap();
        let operation = ArrayOperation::Condition(Box::new(condition));
        let context = EagerContext::<TestArray, ArrayOperation<TestArray>>::new();

        let batched_input = {
            let value = TestArray::vector(vec![1.0, 4.0, 9.0]);
            ArrayBatch::new(value.r#type().into_owned(), value, Some(0))
        }
        .unwrap();
        let outputs = operation.batch(&context, &[replicated_predicate(false), batched_input]).unwrap();
        assert_eq!(outputs.len(), 1);
        let output_batch = &outputs[0];
        assert_eq!(output_batch.batch_axis(), BatchAxis::new(0));
        assert_eq!(output_batch.value().values, vec![3.0, 12.0, 27.0]);
    }

    #[test]
    fn test_dot_general_evaluates_batched_matmul() {
        use crate::tracing_v2::operations::dot::{Dot, DotDimensionNumbers};

        // Batched matmul: [2, 2, 3] @ [2, 3, 2] -> [2, 2, 2] with axis 0 batched.
        let lhs_values: Vec<f64> = (1..=12).map(|value| value as f64).collect();
        let rhs_values: Vec<f64> = (1..=12).map(|value| value as f64).collect();
        let lhs = TestArray {
            r#type: ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(2), Size::Static(3)])),
            values: lhs_values,
        };
        let rhs = TestArray {
            r#type: ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3), Size::Static(2)])),
            values: rhs_values,
        };

        let dimensions = DotDimensionNumbers::new(vec![2], vec![1], vec![0], vec![0]);
        let output = lhs.dot(&rhs, &dimensions);

        assert_eq!(
            output.r#type,
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(2), Size::Static(2)])),
        );
        // Batch 0: [[1,2,3],[4,5,6]] @ [[1,2],[3,4],[5,6]] = [[22,28],[49,64]]
        // Batch 1: [[7,8,9],[10,11,12]] @ [[7,8],[9,10],[11,12]] = [[220,244],[301,334]]
        assert_eq!(output.values, vec![22.0, 28.0, 49.0, 64.0, 220.0, 244.0, 301.0, 334.0]);
    }

    #[test]
    fn test_transpose_evaluates_general_permutation() {
        use crate::operations::manipulation::Transpose;

        // Rank-3 transpose with permutation [2, 0, 1]: [2, 3, 4] -> [4, 2, 3].
        let values: Vec<f64> = (0..24).map(|value| value as f64).collect();
        let input = TestArray {
            r#type: ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3), Size::Static(4)])),
            values,
        };

        let output = input.transpose(vec![2, 0, 1]).unwrap();

        assert_eq!(
            output.r#type,
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(4), Size::Static(2), Size::Static(3)])),
        );
        // Spot-check: input[0, 0, 0] (= 0) goes to output[0, 0, 0]; input[0, 0, 1] (= 1) -> output[1, 0, 0];
        // input[1, 2, 3] (= 23) -> output[3, 1, 2].
        assert_eq!(output.values[0], 0.0);
        assert_eq!(output.values[1 * 6], 1.0);
        let output_flat_for_23 = 3 * 6 + 1 * 3 + 2;
        assert_eq!(output.values[output_flat_for_23], 23.0);
    }

    #[test]
    fn test_jacrev_over_dot_batches_adjoint_dots() {
        use crate::tracing_v2::operations::dot::{Dot, DotDimensionNumbers};

        // jacrev internally batches the pullback's adjoint `dot` operations (their known operands riding as
        // replicated pullback inputs) through BatchableOperation::batch — exercise that path explicitly via a
        // dot-based scalar function. f(x, y) = x · y (inner product) so ∂f/∂x = y and ∂f/∂y = x.
        let jacobian = jacrev(
            &EagerContext::<TestArray, ArrayOperation<TestArray>>::new(),
            |(x, y)| Ok(x.dot(&y, &DotDimensionNumbers::inner_product())),
            (TestArray::vector(vec![2.0, 3.0, 5.0]), TestArray::vector(vec![7.0, 11.0, 13.0])),
        )
        .unwrap();

        let row = jacobian.rows();
        let (block_x, block_y) = row.partials();
        assert_eq!(block_x.values(), &[7.0, 11.0, 13.0]);
        assert_eq!(block_y.values(), &[2.0, 3.0, 5.0]);
    }

    #[test]
    fn test_jacfwd_over_dot_batches_basis_tangents_through_the_pushforward() {
        use crate::tracing_v2::operations::dot::{Dot, DotDimensionNumbers};

        // jacfwd linearizes the function once, then replays all input-coordinate basis tangents through the
        // pushforward in one batched pass. A dot-product scalar output exercises captured-factor (product-rule)
        // linear maps instead of only elementwise tangent arithmetic.
        let jacobian = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .jacfwd(
                |(x, y)| Ok(x.dot(&y, &DotDimensionNumbers::inner_product())),
                (TestArray::vector(vec![2.0, 3.0, 5.0]), TestArray::vector(vec![7.0, 11.0, 13.0])),
            )
            .unwrap();

        let row = jacobian.rows();
        let (block_x, block_y) = row.partials();
        assert_eq!(block_x.values(), &[7.0, 11.0, 13.0]);
        assert_eq!(block_y.values(), &[2.0, 3.0, 5.0]);
    }

    #[test]
    fn test_batching_batch_varying_condition_selects_per_item() {
        // Per-item scalar branches: on_true scales by 2.0, on_false scales by 3.0. Operand is a
        // [4]-vector; predicate is a [4]-vector with values [1.0, 0.0, 1.0, 0.0]. Expected per-item
        // output: [1*2, 2*3, 3*2, 4*3] = [2, 6, 6, 12].
        let condition = ConditionOperation::new(scalar_scale_branch(2.0), scalar_scale_branch(3.0)).unwrap();
        let operation = ArrayOperation::Condition(Box::new(condition));

        let predicate_type = ArrayType::new(DataType::Boolean, Shape::new(vec![Size::Static(4)]));
        let operand_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(4)]));
        let predicate_batch =
            ArrayBatch::new(predicate_type.clone(), TestArray::new(predicate_type, vec![1.0, 0.0, 1.0, 0.0]), Some(0))
                .unwrap();
        let operand_batch =
            ArrayBatch::new(operand_type, TestArray::vector(vec![1.0, 2.0, 3.0, 4.0]), Some(0)).unwrap();

        let context = EagerContext::<TestArray, ArrayOperation<TestArray>>::new();
        let outputs = operation.batch(&context, &[predicate_batch, operand_batch]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value().values, vec![2.0, 6.0, 6.0, 12.0]);
    }

    #[test]
    fn test_broadcast_replicates_across_added_axes() {
        use crate::operations::manipulation::Broadcast;

        // A length-3 vector broadcast to shape [2, 3] with output_axes=[1]: the input
        // axis maps to output axis 1, so the value replicates across output axis 0.
        let input = TestArray::vector(vec![1.0, 2.0, 3.0]);
        let target = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]));
        let output = input.broadcast(target, &[1]).unwrap();
        assert_eq!(output.values, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_broadcast_leading_prepends_axes() {
        use crate::operations::manipulation::Broadcast;

        // `t.broadcast_leading([2])` prepends a leading axis of size 2 and replicates the original
        // values across it. Matches `jax.lax.broadcast(t, [2])`.
        let input = TestArray::vector(vec![1.0, 2.0, 3.0]);
        let output = input.broadcast_leading(vec![2]).unwrap();
        assert_eq!(output.r#type, ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)])),);
        assert_eq!(output.values, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_broadcast_to_uses_numpy_right_alignment() {
        use crate::operations::manipulation::Broadcast;

        // A scalar (rank-0) broadcasts to shape [2, 3] by replicating across both axes.
        let scalar = TestArray::scalar(7.0);
        let output = scalar.broadcast_to(Shape::new(vec![Size::Static(2), Size::Static(3)])).unwrap();
        assert_eq!(output.values, vec![7.0; 6]);

        // A rank-1 `[3]` vector broadcasts to `[2, 3]` by right-aligning: input axis 0 maps
        // to output axis 1, replicating across output axis 0 — matches NumPy's
        // `np.broadcast_to(x, (2, 3))`.
        let vector = TestArray::vector(vec![10.0, 20.0, 30.0]);
        let output = vector.broadcast_to(Shape::new(vec![Size::Static(2), Size::Static(3)])).unwrap();
        assert_eq!(output.values, vec![10.0, 20.0, 30.0, 10.0, 20.0, 30.0]);
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
        let pred_batch = ArrayBatch::new(pred_type.clone(), TestArray::new(pred_type, vec![1.0]), None).unwrap();
        let on_true_batch =
            ArrayBatch::new(operand_type.clone(), TestArray::vector(vec![1.0, 2.0, 3.0]), Some(0)).unwrap();
        let on_false_batch = ArrayBatch::new(operand_type, TestArray::vector(vec![4.0, 5.0, 6.0]), Some(0)).unwrap();

        let outputs = SelectOperation
            .batch(&EagerContext::<TestArray>::new(), &[pred_batch, on_true_batch, on_false_batch])
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value().values, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_batching_rule_zero_operation_is_replicated() {
        // `ZeroOperation` takes no inputs and produces a constant of its captured type. The same
        // constant is the right value for every batch item, so the per-op rule wraps the output as
        // replicated (`batch_axis = None`) with no inserted axis.
        let scalar = ArrayType::scalar(DataType::F64);
        let operation = crate::operations::constants::ZeroOperation::new(scalar.clone());

        let outputs: Vec<ArrayBatch<TestArray>> = operation
            .batch(&EagerContext::<TestArray, crate::operations::constants::ConstantOperation<TestArray>>::new(), &[])
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::replicated());
        assert_eq!(outputs[0].r#type().into_owned(), scalar);
        assert_eq!(outputs[0].value().values, vec![0.0]);
    }

    #[test]
    fn test_batching_rule_one_operation_is_replicated() {
        // Symmetric to `ZeroOperation`: `OneOperation` is replicated by construction.
        let scalar = ArrayType::scalar(DataType::F64);
        let operation = crate::operations::constants::OneOperation::new(scalar.clone());

        let outputs: Vec<ArrayBatch<TestArray>> = operation
            .batch(&EagerContext::<TestArray, crate::operations::constants::ConstantOperation<TestArray>>::new(), &[])
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::replicated());
        assert_eq!(outputs[0].r#type().into_owned(), scalar);
        assert_eq!(outputs[0].value().values, vec![1.0]);
    }

    #[test]
    fn test_batching_rule_enum_zero_input_variants_delegate_to_payload_rules() {
        // The enum arms for zero-input variants delegate to the per-payload rules (interpret once under the active
        // context and wrap the outputs replicated) instead of erroring. This is the path a nested-program
        // `Zero`/`Fill` instruction takes when `BatchingContext::interpret_program` dispatches it through the enum's
        // `batch` with no inputs.
        let scalar = ArrayType::scalar(DataType::F64);
        let context = EagerContext::<TestArray, ArrayOperation<TestArray>>::new();

        let zero = ArrayOperation::<TestArray>::Zero(crate::operations::constants::ZeroOperation::new(scalar.clone()));
        let outputs: Vec<ArrayBatch<TestArray>> = zero.batch(&context, &[]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::replicated());
        assert_eq!(outputs[0].value().values, vec![0.0]);

        let fill = ArrayOperation::<TestArray>::Fill(crate::operations::constants::FillOperation::new(scalar, 7.5));
        let outputs: Vec<ArrayBatch<TestArray>> = fill.batch(&context, &[]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::replicated());
        assert_eq!(outputs[0].value().values, vec![7.5]);
    }

    #[test]
    fn test_jacrev_through_function_using_zero_like() {
        // `f(x) = x + zero_like(x)` is functionally the identity, but exercises the
        // `ZeroLikeOperation` rule through `jacrev`'s internal Jacobian batching path. Verifies
        // that the constant-op rule composes cleanly with reverse-mode autodiff.
        let jacobian = jacrev(
            &EagerContext::<TestArray, ArrayOperation<TestArray>>::new(),
            |x| Ok(x.clone() + x.zero_like()),
            TestArray::scalar(2.0),
        )
        .unwrap();
        let row = jacobian.rows();
        let block = row.partials();
        // d(x + 0) / dx = 1 at the scalar point.
        assert_close(block.values()[0], 1.0);
    }

    #[test]
    fn test_jacfwd_through_function_using_one_like() {
        // `f(x) = x + one_like(x)` shifts x by a constant; the Jacobian is still 1. Exercises
        // `OneLikeOperation` through jacfwd's internal batching.
        let jacobian = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .jacfwd(|x| Ok(x.clone() + x.one_like()), TestArray::scalar(2.0))
            .unwrap();
        let row = jacobian.rows();
        let block = row.partials();
        // d(x + 1) / dx = 1.
        assert_close(block.values()[0], 1.0);
    }

    /// Stages `ArrayOperation::Condition` over `scalar_scale_branch(2.0)` / `scalar_scale_branch(3.0)` inside an
    /// active differentiation trace, feeding its predicate input with a staged constant `true`.
    fn stage_constant_predicate_condition<C>(x: crate::tracing::Tracer<C>) -> crate::tracing::Tracer<C>
    where
        C: crate::contexts::StagingContext<
                Type = ArrayType,
                Constant = TestArray,
                Operation = ArrayOperation<TestArray>,
            >,
    {
        let condition = ConditionOperation::new(scalar_scale_branch(2.0), scalar_scale_branch(3.0)).unwrap();
        let predicate = x.context().constant(TestArray::new(ArrayType::scalar(DataType::Boolean), vec![1.0]));
        let mut outputs = x
            .context()
            .stage_operation(ArrayOperation::Condition(Box::new(condition)), &[&predicate, &x])
            .unwrap();
        outputs.remove(0)
    }

    #[test]
    fn test_condition_composes_with_jacrev_and_jacfwd() {
        // The constant `true` predicate selects the scale-by-2 branch, so both autodiff transforms must report a
        // derivative of 2 by linearizing the selected branch through the `ArrayOperation::Condition` JVP dispatch.
        let jacobian = jacrev(
            &EagerContext::<TestArray, ArrayOperation<TestArray>>::new(),
            |x| Ok(stage_constant_predicate_condition(x)),
            TestArray::scalar(4.0),
        )
        .unwrap();
        assert_close(jacobian.rows().partials().values()[0], 2.0);

        let jacobian = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .jacfwd(|x| Ok(stage_constant_predicate_condition(x)), TestArray::scalar(4.0))
            .unwrap();
        assert_close(jacobian.rows().partials().values()[0], 2.0);
    }

    /// Builds the `while (x < 8) { x = x + x }` doubling-loop fixture used by the while differentiation tests.
    fn doubling_while_operation()
    -> crate::operations::control_flow::WhileOperation<TestArray, ArrayOperation<TestArray>> {
        use crate::operations::compare::ComparisonDirection;
        type TestOp = ArrayOperation<TestArray>;

        let scalar_f64 = ArrayType::scalar(DataType::F64);
        let mut condition_builder = ProgramBuilder::<TestArray, TestOp>::new();
        let condition_state = condition_builder.add_input(scalar_f64.clone());
        let threshold = condition_builder.add_constant(TestArray::scalar(8.0));
        let predicate = condition_builder
            .add_instruction(CompareOperation::new(ComparisonDirection::LessThan), vec![condition_state, threshold])
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut body_builder = ProgramBuilder::<TestArray, TestOp>::new();
        let body_state = body_builder.add_input(scalar_f64);
        let doubled = body_builder.add_instruction(AddOperation, vec![body_state, body_state]).unwrap()[0];
        let body = body_builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![doubled], vec![Placeholder], vec![Placeholder])
            .unwrap();

        crate::operations::control_flow::WhileOperation::new(condition, body).unwrap()
    }

    #[test]
    fn test_while_jvp_propagates_tangents_through_staged_linear_loop() {
        // `while (x < 8) { x = x + x }` starting at `x = 1` doubles three times, so the primal output is 8 and the
        // forward-mode derivative is 2^3 = 8. Exercises the `ArrayOperation::While` JVP dispatch in an eager domain:
        // the rule stages the doubled-state linear loop, which direct JVP execution interprets immediately.
        let while_operation = doubling_while_operation();
        let (primal, tangent) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .jvp(
                move |x| {
                    let mut outputs =
                        x.context().bind(ArrayOperation::While(Box::new(while_operation)), &[x.clone()])?;
                    Ok(outputs.remove(0))
                },
                TestArray::scalar(1.0),
                TestArray::scalar(1.0),
            )
            .unwrap();
        assert_eq!(primal.values, vec![8.0]);
        assert_eq!(tangent.values, vec![8.0]);
    }

    #[test]
    fn test_jacfwd_through_while_batches_basis_tangents() {
        // `jacfwd` linearizes the loop once into a pushforward containing a staged linear while, then replays
        // the batched basis tangents through it: the pushforward is instantiated into a direct linear program and
        // interpreted over the value-level batching rules, so the linear while runs once over the stacked
        // basis tangents. The derivative of the doubling loop at `x = 1` is `2^3 = 8`.
        let while_operation = doubling_while_operation();
        let jacobian = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .jacfwd(
                move |x| {
                    let mut outputs =
                        x.context().stage_operation(ArrayOperation::While(Box::new(while_operation)), &[&x])?;
                    Ok(outputs.remove(0))
                },
                TestArray::scalar(1.0),
            )
            .unwrap();
        assert_close(jacobian.rows().partials().values()[0], 8.0);
    }

    #[test]
    fn test_while_value_and_grad_computes_gradient_through_bounded_loop() {
        // Eager reverse mode through a *bounded* while loop whose bound does not bind: the doubling loop at `x = 1`
        // runs three iterations, below the bound of 5, so the eager hybrid rule unrolls the loop in full (the bound
        // only truncates once it is reached). The straight-line pushforward transposes, and locally `f(x) = 8 x`, so
        // the value is 8 and the gradient is 8.
        let while_operation = doubling_while_operation().with_iteration_bound(5).unwrap();
        let (value, gradient) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .value_and_grad(
                move |x| {
                    let mut outputs =
                        x.context().stage_operation(ArrayOperation::While(Box::new(while_operation)), &[&x]).unwrap();
                    outputs.remove(0)
                },
                TestArray::scalar(1.0),
            )
            .unwrap();
        assert_eq!(value.values, vec![8.0]);
        assert_eq!(gradient.values, vec![8.0]);
    }

    #[test]
    fn test_while_value_and_grad_computes_gradient_through_unrolled_loop() {
        // Eager reverse mode through an *unbounded* while loop: the hybrid JVP rule unrolls the doubling loop at
        // `x = 1` (three iterations), so the pushforward is a straight-line linear program that transposes. Locally
        // `f(x) = 8 x`, so the value is 8 and the gradient is 8. JAX cannot do this even under eager execution,
        // because it always traces `while_loop`.
        let while_operation = doubling_while_operation();
        let (value, gradient) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .value_and_grad(
                move |x| {
                    let mut outputs =
                        x.context().stage_operation(ArrayOperation::While(Box::new(while_operation)), &[&x]).unwrap();
                    outputs.remove(0)
                },
                TestArray::scalar(1.0),
            )
            .unwrap();
        assert_eq!(value.values, vec![8.0]);
        assert_eq!(gradient.values, vec![8.0]);
    }

    #[test]
    fn test_while_vjp_computes_cotangents_through_bounded_loop() {
        // Eager `vjp` through a *bounded* while loop whose bound does not bind: the doubling loop at `x = 1` runs
        // three iterations, below the bound of 5, so the eager hybrid rule unrolls the loop in full and transposes the
        // straight-line pushforward into a reusable pullback (no `while` remains). The loop is locally `f(x) = 8 x`,
        // so every output cotangent is scaled by 8.
        let while_operation = doubling_while_operation().with_iteration_bound(5).unwrap();
        let (output, pullback, residuals) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .vjp(
                move |x| {
                    let mut outputs =
                        x.context().stage_operation(ArrayOperation::While(Box::new(while_operation)), &[&x])?;
                    Ok(outputs.remove(0))
                },
                TestArray::scalar(1.0),
            )
            .unwrap();
        assert_eq!(output.values, vec![8.0]);
        assert!(!pullback.to_string().contains("while"), "{pullback}");
        let pullback_inputs = |cotangent: TestArray| {
            let mut inputs = vec![cotangent];
            inputs.extend(residuals.iter().cloned());
            inputs
        };
        assert_eq!(
            pullback
                .interpret(pullback_inputs(TestArray::scalar(1.0)))
                .map(|cotangents| cotangents[0].values.clone()),
            Ok(vec![8.0]),
        );
        assert_eq!(
            pullback
                .interpret(pullback_inputs(TestArray::scalar(5.0)))
                .map(|cotangents| cotangents[0].values.clone()),
            Ok(vec![40.0]),
        );
    }

    #[test]
    fn test_while_vjp_computes_cotangents_through_unrolled_loop() {
        // Eager `vjp` transposes the unrolled straight-line pushforward of an *unbounded* loop into a reusable
        // pullback: the doubling loop at `x = 1` is locally `f(x) = 8 x`, so every output cotangent is scaled by 8.
        let while_operation = doubling_while_operation();
        let (output, pullback, residuals) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .vjp(
                move |x| {
                    let mut outputs =
                        x.context().stage_operation(ArrayOperation::While(Box::new(while_operation)), &[&x])?;
                    Ok(outputs.remove(0))
                },
                TestArray::scalar(1.0),
            )
            .unwrap();
        assert_eq!(output.values, vec![8.0]);
        let pullback_inputs = |cotangent: TestArray| {
            let mut inputs = vec![cotangent];
            inputs.extend(residuals.iter().cloned());
            inputs
        };
        assert_eq!(
            pullback
                .interpret(pullback_inputs(TestArray::scalar(1.0)))
                .map(|cotangents| cotangents[0].values.clone()),
            Ok(vec![8.0]),
        );
        assert_eq!(
            pullback
                .interpret(pullback_inputs(TestArray::scalar(5.0)))
                .map(|cotangents| cotangents[0].values.clone()),
            Ok(vec![40.0]),
        );
    }

    /// Builds the three-iteration cumulative-product [`ScanOperation`] (body `[carry, x] -> [carry * x, carry * x]`)
    /// used by the scan differentiation tests, optionally visiting the iterations in reverse order.
    fn product_scan_operation(
        reverse: bool,
    ) -> crate::operations::control_flow::ScanOperation<TestArray, ArrayOperation<TestArray>> {
        type TestOp = ArrayOperation<TestArray>;
        let mut body_builder = ProgramBuilder::<TestArray, TestOp>::new();
        let carry = body_builder.add_input(ArrayType::scalar(DataType::F64));
        let x = body_builder.add_input(ArrayType::scalar(DataType::F64));
        let product = body_builder.add_instruction(MulOperation, vec![carry, x]).unwrap()[0];
        let body = body_builder
            .build::<Vec<TestArray>, Vec<TestArray>>(
                vec![product, product],
                vec![Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();
        crate::operations::control_flow::ScanOperation::<TestArray, ArrayOperation<TestArray>>::new(body, 1, 3)
            .unwrap()
            .with_reverse(reverse)
    }

    #[test]
    fn test_scan_value_and_grad_computes_gradient_through_reversed_linear_scan() {
        // The headline scan capability: end-to-end reverse mode. `f(init, xs)` is the final carry of the
        // cumulative-product scan, so `f = init * xs[0] * xs[1] * xs[2] = 24` at `init = 1, xs = [2, 3, 4]`, with
        // gradient `24` w.r.t. `init` and `[12, 8, 6]` w.r.t. `xs`. The pullback runs the transposed linear scan
        // (same residual stacks, `reverse` flipped) — the static trip count is what makes this total, where the
        // staged linear `while` rejects transposition.
        let scan = product_scan_operation(false);
        let (value, (init_gradient, xs_gradient)) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .value_and_grad(
                move |(init, xs)| {
                    let mut outputs =
                        init.context().stage_operation(ArrayOperation::Scan(Box::new(scan)), &[&init, &xs]).unwrap();
                    outputs.remove(0)
                },
                (TestArray::scalar(1.0), TestArray::vector(vec![2.0, 3.0, 4.0])),
            )
            .unwrap();
        assert_eq!(value.values, vec![24.0]);
        assert_eq!(init_gradient.values, vec![24.0]);
        assert_eq!(xs_gradient.values, vec![12.0, 8.0, 6.0]);
    }

    #[test]
    fn test_scan_vjp_stages_reversed_linear_scan_in_reusable_pullback() {
        // `vjp` through the cumulative-product scan produces a reusable pullback containing the transposed linear
        // scan: the same residual stacks with `reverse` flipped to `true`. Each cotangent seed scales the
        // hand-computed gradients `(24, [12, 8, 6])`.
        let scan = product_scan_operation(false);
        let (output, pullback, residuals) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .vjp(
                move |(init, xs)| {
                    let mut outputs =
                        init.context().stage_operation(ArrayOperation::Scan(Box::new(scan)), &[&init, &xs])?;
                    Ok(outputs.remove(0))
                },
                (TestArray::scalar(1.0), TestArray::vector(vec![2.0, 3.0, 4.0])),
            )
            .unwrap();
        assert_eq!(output.values, vec![24.0]);
        let rendered_pullback = pullback.to_string();
        assert!(rendered_pullback.contains("scan"), "{rendered_pullback}");
        assert!(rendered_pullback.contains("reverse=true"), "{rendered_pullback}");
        let pullback_inputs = |cotangent: TestArray| {
            let mut inputs = vec![cotangent];
            inputs.extend(residuals.iter().cloned());
            inputs
        };
        let cotangents = pullback.interpret(pullback_inputs(TestArray::scalar(1.0))).unwrap();
        assert_eq!(cotangents[0].values, vec![24.0]);
        assert_eq!(cotangents[1].values, vec![12.0, 8.0, 6.0]);
        let cotangents = pullback.interpret(pullback_inputs(TestArray::scalar(2.0))).unwrap();
        assert_eq!(cotangents[0].values, vec![48.0]);
        assert_eq!(cotangents[1].values, vec![24.0, 16.0, 12.0]);
    }

    #[test]
    fn test_reversed_scan_jvp_and_grad_align_items() {
        // Pins the alignment invariant for `reverse = true`: the primal scan visits iterations from the back while
        // output iteration `i` stays aligned with input iteration `i`, and the linear scan runs with the same direction so
        // residual iteration `i` is consumed exactly when tangent iteration `i` is processed. The reversed cumulative
        // product has `ys = [x0 x1 x2, x1 x2, x2] = [24, 12, 4]`, so a unit tangent on `x1` gives
        // `dys = [x0 x2, x2, 0] = [8, 4, 0]`.
        let scan = product_scan_operation(true);
        let ((carry, ys), (carry_tangent, ys_tangent)) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .jvp(
                move |(init, xs)| {
                    let mut outputs =
                        init.context().bind(ArrayOperation::Scan(Box::new(scan)), &[init.clone(), xs.clone()])?;
                    let ys = outputs.remove(1);
                    Ok((outputs.remove(0), ys))
                },
                (TestArray::scalar(1.0), TestArray::vector(vec![2.0, 3.0, 4.0])),
                (TestArray::scalar(0.0), TestArray::vector(vec![0.0, 1.0, 0.0])),
            )
            .unwrap();
        assert_eq!(carry.values, vec![24.0]);
        assert_eq!(ys.values, vec![24.0, 12.0, 4.0]);
        assert_eq!(carry_tangent.values, vec![8.0]);
        assert_eq!(ys_tangent.values, vec![8.0, 4.0, 0.0]);

        // Reverse mode through the reversed scan flips `reverse` back to `false` in the pullback and produces the
        // same product-rule gradients (multiplication commutes across the visit order).
        let scan = product_scan_operation(true);
        let (output, pullback, residuals) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .vjp(
                move |(init, xs)| {
                    let mut outputs =
                        init.context().stage_operation(ArrayOperation::Scan(Box::new(scan)), &[&init, &xs])?;
                    Ok(outputs.remove(0))
                },
                (TestArray::scalar(1.0), TestArray::vector(vec![2.0, 3.0, 4.0])),
            )
            .unwrap();
        assert_eq!(output.values, vec![24.0]);
        let rendered_pullback = pullback.to_string();
        assert!(rendered_pullback.contains("reverse=false"), "{rendered_pullback}");
        let mut pullback_inputs = vec![TestArray::scalar(1.0)];
        pullback_inputs.extend(residuals);
        let cotangents = pullback.interpret(pullback_inputs).unwrap();
        assert_eq!(cotangents[0].values, vec![24.0]);
        assert_eq!(cotangents[1].values, vec![12.0, 8.0, 6.0]);
    }

    #[test]
    fn test_array_operation_condition_interprets_runtime_predicate() {
        let condition = ConditionOperation::new(scalar_scale_branch(2.0), scalar_scale_branch(3.0)).unwrap();
        let operation = ArrayOperation::Condition(Box::new(condition));

        let predicate = TestArray::new(ArrayType::scalar(DataType::Boolean), vec![0.0]);
        assert_eq!(
            operation
                .interpret(&EagerContext::<TestArray>::new(), &[predicate, TestArray::scalar(4.0)])
                .map(|outputs| outputs[0].values[0]),
            Ok(12.0),
        );
    }

    #[test]
    fn test_condition_vjp_computes_branch_cotangents_for_runtime_predicate() {
        // f(p, x) = if p { 2 * x } else { 3 * x }. Reverse mode composes through the total linear-condition
        // transpose rule: the pullback runs the transposed branch program selected by the captured predicate, so
        // the operand cotangent is 2 * output cotangent at a TRUE-predicate primal point and 3 * output cotangent
        // at a FALSE one. The Boolean predicate has no tangent space, so its cotangent slot is always zero.
        let (output, pullback, residuals) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .vjp(
                |(predicate, operand)| {
                    let condition = ConditionOperation::new(scalar_scale_branch(2.0), scalar_scale_branch(3.0))?;
                    let mut outputs = predicate
                        .context()
                        .stage_operation(ArrayOperation::Condition(Box::new(condition)), &[&predicate, &operand])?;
                    Ok(outputs.remove(0))
                },
                (TestArray::new(ArrayType::scalar(DataType::Boolean), vec![1.0]), TestArray::scalar(4.0)),
            )
            .unwrap();
        assert_eq!(output.values, vec![8.0]);
        let mut pullback_inputs = vec![TestArray::scalar(5.0)];
        pullback_inputs.extend(residuals);
        let cotangents = pullback.interpret(pullback_inputs).unwrap();
        assert_eq!(cotangents[1].values, vec![10.0]);
        assert_eq!(cotangents[0].values, vec![0.0]);

        let (output, pullback, residuals) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .vjp(
                |(predicate, operand)| {
                    let condition = ConditionOperation::new(scalar_scale_branch(2.0), scalar_scale_branch(3.0))?;
                    let mut outputs = predicate
                        .context()
                        .stage_operation(ArrayOperation::Condition(Box::new(condition)), &[&predicate, &operand])?;
                    Ok(outputs.remove(0))
                },
                (TestArray::new(ArrayType::scalar(DataType::Boolean), vec![0.0]), TestArray::scalar(4.0)),
            )
            .unwrap();
        assert_eq!(output.values, vec![12.0]);
        let mut pullback_inputs = vec![TestArray::scalar(5.0)];
        pullback_inputs.extend(residuals);
        let cotangents = pullback.interpret(pullback_inputs).unwrap();
        assert_eq!(cotangents[1].values, vec![15.0]);
        assert_eq!(cotangents[0].values, vec![0.0]);
    }
}
