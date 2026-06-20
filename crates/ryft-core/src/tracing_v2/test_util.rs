use crate::operations::arithmetic::ScaleOperation;
use crate::parameters::Placeholder;
use crate::programs::ProgramBuilder;
use crate::tests::{TestArray, TestArrayDomain};
use crate::tracing_v2::ArrayOperation;
use crate::types::{ArrayType, DataType};

/// Asserts that `actual` is within absolute tolerance `1e-9` of `expected`.
pub(crate) fn assert_close(actual: f64, expected: f64) {
    let delta = (actual - expected).abs();
    assert!(delta <= 1e-9, "expected {actual} ~= {expected}; absolute error {delta} exceeded tolerance");
}

/// Builds a single-input flat program that scales its scalar input by `factor`.
pub(crate) fn scalar_scale_branch(
    factor: f64,
) -> crate::programs::Program<ArrayType, TestArray, ArrayOperation<ArrayType, TestArray>, Vec<TestArray>, Vec<TestArray>>
{
    let mut builder = ProgramBuilder::<ArrayType, TestArray, ArrayOperation<ArrayType, TestArray>>::new();
    let input = builder.add_input(ArrayType::scalar(DataType::F64));
    let output = builder.add_instruction(ScaleOperation::new(TestArray::scalar(factor)), vec![input]).unwrap()[0];
    builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::convert::Infallible;
    use std::rc::Rc;

    use pretty_assertions::assert_eq;

    use crate::ProvidesContext;
    use crate::contexts::{EagerContext, StagingContext};
    use crate::operations::InterpretableOperation;
    use crate::operations::arithmetic::{AddOperation, MulOperation, SubOperation};
    use crate::operations::compare::CompareOperation;
    use crate::operations::constants::{OneLike, OneLikeOperation, ZeroLike, ZeroLikeOperation};
    use crate::operations::control_flow::ConditionOperation;
    use crate::operations::trigonometric::Sin;
    use crate::parameters::Placeholder;
    use crate::programs::ProgramBuilder;
    use crate::tracing_v2::operations::control_flow::LinearConditionOperation;
    use crate::tracing_v2::{
        ArrayBatch, BatchableOperation, CapturedFactor, DifferentiableDomainExtension, DifferentiableOperation,
        DifferentiationContext, JvpTracer, LinearArrayOperation, ResidualizedOperation, TangentContext, jacrev,
    };
    use crate::types::{Shape, Size, Typed};

    use super::*;

    #[test]
    fn test_dot_batches_mixed_lhs_batched_rhs_lane_uniform() {
        // LHS is mapped at axis 0 with per-lane shape [3]; RHS is lane-uniform with shape [3].
        // Per-lane semantics: dot(lhs_row, rhs) over the shared K=3 dimension. The batching rule
        // should broadcast the RHS to gain a singleton batch axis at position 0, then thread the
        // batch axis through `lift_dot_dimensions`.
        use crate::tracing_v2::operations::dot::{DotDimensionNumbers, DotOperation};
        let lhs = ArrayBatch::mapped(TestArray::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), 0).unwrap();
        let rhs = ArrayBatch::unbatched(TestArray::vector(vec![10.0, 100.0, 1000.0]));
        let dimensions = DotDimensionNumbers::new(vec![0], vec![0], vec![], vec![]);
        let outputs = DotOperation::new(dimensions).batch(&crate::EagerContext::new(), &[lhs, rhs]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), Some(0));
        // Lane 0: 1*10 + 2*100 + 3*1000 = 3210; lane 1: 4*10 + 5*100 + 6*1000 = 6540.
        assert_eq!(outputs[0].value().values(), &[3210.0, 6540.0]);
    }

    #[test]
    fn test_reduce_sum_jvp_linearizes_to_itself() {
        // Verify the JVP rule for `ReduceOperation::Sum`: the tangent of `sum(x)` is `sum(Δx)`.
        // We exercise the rule directly on a `Tangent::Value` over a TestArray vector. Result
        // should match summing the values directly.
        use crate::differentiation::Tangent;
        use crate::tracing_v2::operations::reduce::{ReduceOperation, ReductionKind};
        let primal = TestArray::vector(vec![1.0, 2.0, 3.0, 4.0]);
        let tangent_value = TestArray::vector(vec![0.5, 0.5, 0.5, 0.5]);
        let tangent: Tangent<ArrayType, TestArray> = Tangent::Value(tangent_value);

        let operation = ReduceOperation::new(vec![0], ReductionKind::Sum);

        // Primal: reduce(x, [0], Sum) on `TestArray` directly.
        let primal_output = operation
            .interpret(&crate::EagerContext::new(), std::slice::from_ref(&primal))
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        assert_eq!(primal_output.values(), &[10.0]);

        // Tangent: linearizes to itself (Sum is linear), so the tangent of the reduce is the
        // reduce of the tangent.
        let tangent_outputs = operation.interpret(&crate::EagerContext::new(), std::slice::from_ref(&tangent)).unwrap();
        let tangent_output = tangent_outputs.into_iter().next().unwrap();
        match tangent_output {
            Tangent::Value(value) => assert_eq!(value.values(), &[2.0]),
            Tangent::Zero(_) => panic!("expected non-zero tangent output"),
        }
    }

    #[test]
    fn test_lane_varying_while_terminates_lanes_independently() {
        // Build a batched while loop with a per-lane termination predicate. Each lane starts at a
        // different value and decrements by 1 until it reaches 0. Lane 0 (initial 3.0) iterates
        // three times, lane 1 (initial 1.0) iterates once, lane 2 (initial 2.0) iterates twice;
        // inactive lanes retain their final state via per-lane `Select` masking.
        use crate::operations::compare::ComparisonDirection;
        use crate::operations::control_flow::WhileOperation;
        use crate::programs::Program;
        type TestOp = ArrayOperation<ArrayType, TestArray>;

        let scalar_f64 = ArrayType::scalar(DataType::F64);

        // Condition program: state -> (state > 0). Returns a scalar Boolean.
        let mut condition_builder = ProgramBuilder::<ArrayType, TestArray, TestOp>::new();
        let cond_input = condition_builder.add_input(scalar_f64.clone());
        let cond_zero = condition_builder.add_instruction(ZeroLikeOperation, vec![cond_input]).unwrap()[0];
        let cond_output = condition_builder
            .add_instruction(CompareOperation::new(ComparisonDirection::GreaterThan), vec![cond_input, cond_zero])
            .unwrap()[0];
        let condition: Program<ArrayType, TestArray, TestOp, Vec<TestArray>, Vec<TestArray>> = condition_builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![cond_output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        // Body program: state -> state - 1.
        let mut body_builder = ProgramBuilder::<ArrayType, TestArray, TestOp>::new();
        let body_input = body_builder.add_input(scalar_f64);
        let body_one = body_builder.add_instruction(OneLikeOperation, vec![body_input]).unwrap()[0];
        let body_output = body_builder.add_instruction(SubOperation, vec![body_input, body_one]).unwrap()[0];
        let body: Program<ArrayType, TestArray, TestOp, Vec<TestArray>, Vec<TestArray>> = body_builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![body_output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let while_op = WhileOperation::<TestArray, TestOp, ArrayType>::new(condition, body).unwrap();
        let context = EagerContext::<ArrayType, TestArray, TestOp>::new();

        let initial_state = ArrayBatch::mapped(TestArray::vector(vec![3.0, 1.0, 2.0]), 0).unwrap();
        let outputs = while_op.batch(&context, &[initial_state]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), Some(0));
        // Each lane terminates when its value reaches 0; inactive lanes retain their last value.
        assert_eq!(outputs[0].value().values(), &[0.0, 0.0, 0.0]);

        // A semantic iteration bound truncates the batched loop too: every lane performs at most two body
        // applications, so lane 0 (initial 3.0) is cut off at 1.0 while the other lanes terminate through their
        // own predicates first.
        let bounded_while_op = while_op.with_iteration_bound(2).unwrap();
        let initial_state = ArrayBatch::mapped(TestArray::vector(vec![3.0, 1.0, 2.0]), 0).unwrap();
        let outputs = bounded_while_op.batch(&context, &[initial_state]).unwrap();
        assert_eq!(outputs[0].value().values(), &[1.0, 0.0, 0.0]);

        // The lane-uniform batched loop respects the bound as well: an unbatched initial state of 5.0 stops at 3.0.
        let initial_state = ArrayBatch::unbatched(TestArray::scalar(5.0));
        let outputs = bounded_while_op.batch(&context, &[initial_state]).unwrap();
        assert_eq!(outputs[0].value().values(), &[3.0]);
    }

    #[test]
    fn test_jacfwd_batches_basis_tangents() {
        let jacobian = TestArrayDomain
            .jacfwd(
                |(x, y)| Ok((x.clone() * y.clone() + x.sin(), x + y)),
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
            &TestArrayDomain,
            |(x, y)| Ok((x.clone() * y.clone() + x.sin(), x + y)),
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
        let jacobian = TestArrayDomain
            .jacfwd(
                |(x, y)| Ok((x.clone() * y.clone() + x.sin(), x + y)),
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
        let hessian = TestArrayDomain
            .hessian(|(x, y)| x.clone() * y + x.sin(), (TestArray::scalar(2.0), TestArray::scalar(3.0)))
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
        let jacobian = TestArrayDomain
            .jacfwd(
                |(x, y)| Ok((x.clone() * y.clone() + x.sin(), y.clone(), x + y)),
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

    /// Builds a lane-uniform scalar Boolean predicate batch with the provided truth value.
    fn lane_uniform_predicate(value: bool) -> ArrayBatch<TestArray> {
        ArrayBatch::unbatched(TestArray::new(ArrayType::scalar(DataType::Boolean), vec![if value { 1.0 } else { 0.0 }]))
    }

    #[test]
    fn test_condition_batches_lane_uniform_true_predicate_over_array_batches() {
        use crate::tracing_v2::batching::{ArrayBatch, BatchableOperation};

        // A lane-uniform `true` predicate selects scalar_scale_branch(2.0). Pass a 3-lane batched operand and
        // verify each lane is independently scaled by 2.
        let condition = ConditionOperation::new(scalar_scale_branch(2.0), scalar_scale_branch(3.0)).unwrap();
        let operation = ArrayOperation::Condition(Box::new(condition));
        let context = EagerContext::<ArrayType, TestArray, ArrayOperation<ArrayType, TestArray>>::new();

        let batched_input = ArrayBatch::mapped(TestArray::vector(vec![1.0, 4.0, 9.0]), 0).unwrap();
        let outputs = operation.batch(&context, &[lane_uniform_predicate(true), batched_input]).unwrap();
        assert_eq!(outputs.len(), 1);
        let output_batch = &outputs[0];
        assert_eq!(output_batch.batch_axis(), Some(0));
        assert_eq!(output_batch.value().values, vec![2.0, 8.0, 18.0]);
    }

    #[test]
    fn test_condition_batches_false_branch_when_lane_uniform_predicate_is_false() {
        use crate::tracing_v2::batching::{ArrayBatch, BatchableOperation};

        let condition = ConditionOperation::new(scalar_scale_branch(2.0), scalar_scale_branch(3.0)).unwrap();
        let operation = ArrayOperation::Condition(Box::new(condition));
        let context = EagerContext::<ArrayType, TestArray, ArrayOperation<ArrayType, TestArray>>::new();

        let batched_input = ArrayBatch::mapped(TestArray::vector(vec![1.0, 4.0, 9.0]), 0).unwrap();
        let outputs = operation.batch(&context, &[lane_uniform_predicate(false), batched_input]).unwrap();
        assert_eq!(outputs.len(), 1);
        let output_batch = &outputs[0];
        assert_eq!(output_batch.batch_axis(), Some(0));
        assert_eq!(output_batch.value().values, vec![3.0, 12.0, 27.0]);
    }

    #[test]
    fn test_linear_condition_batches_through_symbolic_zero_path() {
        use crate::differentiation::Tangent;
        use crate::tracing_v2::batching::{ArrayBatch, BatchableOperation};

        // Build a LinearArrayOperation::Condition with a captured `true` predicate factor and a linear
        // scale branch. Pass an all-`Tangent::Zero` batched input and verify the symbolic-zero
        // short-circuit fires (no concrete arithmetic, output is Tangent::Zero).
        let mut builder =
            ProgramBuilder::<ArrayType, TestArray, LinearArrayOperation<ArrayType, TestArray, TestArray>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(ScaleOperation::new(TestArray::scalar(5.0)), vec![input]).unwrap()[0];
        let linear_branch = builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        let operation: LinearArrayOperation<ArrayType, TestArray, TestArray> =
            LinearArrayOperation::Condition(LinearConditionOperation::new(
                TestArray::new(ArrayType::scalar(DataType::Boolean), vec![1.0]),
                Box::new(linear_branch.clone()),
                Box::new(linear_branch),
            ));

        let batched_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(4)]));
        let zero_input =
            ArrayBatch::new(batched_type.clone(), Tangent::<ArrayType, TestArray>::zero(batched_type), Some(0))
                .unwrap();
        let context = EagerContext::<
            ArrayType,
            Tangent<ArrayType, TestArray>,
            LinearArrayOperation<ArrayType, TestArray, TestArray>,
        >::new();
        let outputs = <LinearArrayOperation<ArrayType, TestArray, TestArray> as BatchableOperation<
            Tangent<ArrayType, TestArray>,
            EagerContext<
                ArrayType,
                Tangent<ArrayType, TestArray>,
                LinearArrayOperation<ArrayType, TestArray, TestArray>,
            >,
        >>::batch(&operation, &context, &[zero_input])
        .unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(outputs[0].value().is_zero(), "expected symbolic-zero output from all-zero linear condition inputs");
    }

    fn linear_scalar_scale_branch(
        factor: f64,
    ) -> crate::programs::Program<
        ArrayType,
        TestArray,
        LinearArrayOperation<ArrayType, TestArray, TestArray>,
        Vec<TestArray>,
        Vec<TestArray>,
    > {
        let mut builder =
            ProgramBuilder::<ArrayType, TestArray, LinearArrayOperation<ArrayType, TestArray, TestArray>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(ScaleOperation::new(TestArray::scalar(factor)), vec![input]).unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
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
    fn test_jacrev_over_dot_uses_left_right_dot_batching() {
        use crate::tracing_v2::operations::dot::{Dot, DotDimensionNumbers};

        // jacrev internally batches cotangents of the form LeftDot/RightDot through
        // BatchableOperation::batch — exercise that path explicitly via a dot-based scalar
        // function. f(x, y) = x · y (inner product) so ∂f/∂x = y and ∂f/∂y = x.
        let jacobian = jacrev(
            &TestArrayDomain,
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
        let jacobian = TestArrayDomain
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
    fn test_batching_lane_varying_condition_selects_per_lane() {
        // Per-lane scalar branches: on_true scales by 2.0, on_false scales by 3.0. Operand is a
        // [4]-vector; predicate is a [4]-vector with values [1.0, 0.0, 1.0, 0.0]. Expected per-lane
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

        let context = EagerContext::<ArrayType, TestArray, ArrayOperation<ArrayType, TestArray>>::new();
        let outputs = operation.batch(&context, &[predicate_batch, operand_batch]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), Some(0));
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
        use crate::operations::manipulation::BroadcastLeading;

        // `t.broadcast_leading([2])` prepends a leading axis of size 2 and replicates the original
        // values across it. Matches `jax.lax.broadcast(t, [2])`.
        let input = TestArray::vector(vec![1.0, 2.0, 3.0]);
        let output = input.broadcast_leading(vec![2]).unwrap();
        assert_eq!(output.r#type, ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)])),);
        assert_eq!(output.values, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_broadcast_to_uses_numpy_right_alignment() {
        use crate::operations::manipulation::BroadcastTo;

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
    fn test_select_batches_with_lane_uniform_predicate_via_broadcast() {
        // Predicate is a rank-0 lane-uniform scalar; on_true / on_false are mapped vectors of
        // size 3. With the JAX-style broadcasting elementwise rule, `apply_elementwise_batch`
        // promotes the lane-uniform predicate to the batched physical shape before invoking
        // `Select::select`, so the mixed-batching case succeeds with the expected per-lane
        // pick.
        use crate::operations::control_flow::SelectOperation;

        let pred_type = ArrayType::scalar(DataType::Boolean);
        let operand_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)]));
        let pred_batch = ArrayBatch::new(pred_type.clone(), TestArray::new(pred_type, vec![1.0]), None).unwrap();
        let on_true_batch =
            ArrayBatch::new(operand_type.clone(), TestArray::vector(vec![1.0, 2.0, 3.0]), Some(0)).unwrap();
        let on_false_batch = ArrayBatch::new(operand_type, TestArray::vector(vec![4.0, 5.0, 6.0]), Some(0)).unwrap();

        let outputs = SelectOperation
            .batch(&crate::EagerContext::new(), &[pred_batch, on_true_batch, on_false_batch])
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), Some(0));
        assert_eq!(outputs[0].value().values, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_batching_rule_zero_operation_is_lane_uniform() {
        // `ZeroOperation` takes no inputs and produces a constant of its captured type. The same
        // constant is the right value for every lane, so the per-op rule wraps the output as
        // lane-uniform (`batch_axis = None`) with no inserted axis.
        let scalar = ArrayType::scalar(DataType::F64);
        let operation = crate::operations::constants::ZeroOperation::new(scalar.clone());

        let outputs: Vec<ArrayBatch<TestArray>> = operation
            .batch(&crate::EagerContext::<ArrayType, TestArray, std::convert::Infallible>::new(), &[])
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), None);
        assert_eq!(outputs[0].r#type().into_owned(), scalar);
        assert_eq!(outputs[0].value().values, vec![0.0]);
    }

    #[test]
    fn test_batching_rule_one_operation_is_lane_uniform() {
        // Symmetric to `ZeroOperation`: `OneOperation` is lane-uniform by construction.
        let scalar = ArrayType::scalar(DataType::F64);
        let operation = crate::operations::constants::OneOperation::new(scalar.clone());

        let outputs: Vec<ArrayBatch<TestArray>> = operation
            .batch(&crate::EagerContext::<ArrayType, TestArray, std::convert::Infallible>::new(), &[])
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), None);
        assert_eq!(outputs[0].r#type().into_owned(), scalar);
        assert_eq!(outputs[0].value().values, vec![1.0]);
    }

    #[test]
    fn test_tangent_condition_captured_predicate_materializes_and_dispatches() {
        use crate::differentiation::Tangent;
        use crate::tracing_v2::batching::{ArrayBatch, BatchableOperation};

        // Tangent + Condition with a captured `true` predicate factor: materialize-then-dispatch routes through the
        // V-level Condition rule, which prepends the lane-uniform predicate and interprets the selected branch over
        // the batched operand. Per-lane output is `[1*2, 2*2, 3*2, 4*2] = [2, 4, 6, 8]`.
        let operation: LinearArrayOperation<ArrayType, TestArray, TestArray> =
            LinearArrayOperation::Condition(LinearConditionOperation::new(
                TestArray::new(ArrayType::scalar(DataType::Boolean), vec![1.0]),
                Box::new(linear_scalar_scale_branch(2.0)),
                Box::new(linear_scalar_scale_branch(3.0)),
            ));

        let operand_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(4)]));
        let operand_batch = ArrayBatch::new(
            operand_type,
            Tangent::<ArrayType, TestArray>::Value(TestArray::vector(vec![1.0, 2.0, 3.0, 4.0])),
            Some(0),
        )
        .unwrap();

        let context = EagerContext::<
            ArrayType,
            Tangent<ArrayType, TestArray>,
            LinearArrayOperation<ArrayType, TestArray, TestArray>,
        >::new();
        let outputs = <LinearArrayOperation<ArrayType, TestArray, TestArray> as BatchableOperation<
            Tangent<ArrayType, TestArray>,
            EagerContext<
                ArrayType,
                Tangent<ArrayType, TestArray>,
                LinearArrayOperation<ArrayType, TestArray, TestArray>,
            >,
        >>::batch(&operation, &context, &[operand_batch])
        .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), Some(0));
        match outputs[0].value() {
            Tangent::Value(v) => assert_eq!(v.values, vec![2.0, 4.0, 6.0, 8.0]),
            Tangent::Zero(_) => panic!("expected a Tangent::Value output from a non-zero operand"),
        }
    }

    #[test]
    fn test_tangent_condition_with_all_zero_tangents_materializes_correctly() {
        use crate::differentiation::Tangent;
        use crate::tracing_v2::batching::{ArrayBatch, BatchableOperation};

        // Tangent + Condition with a captured predicate factor and an all-zero tangent operand: the linear-operation
        // tangent batching rule short-circuits all-zero inputs, using the V-level Condition rule only to lift the
        // output types and axes, so the result stays a symbolic zero (a `Tangent::Value(zero)` is also accepted).
        let operation: LinearArrayOperation<ArrayType, TestArray, TestArray> =
            LinearArrayOperation::Condition(LinearConditionOperation::new(
                TestArray::new(ArrayType::scalar(DataType::Boolean), vec![1.0]),
                Box::new(linear_scalar_scale_branch(2.0)),
                Box::new(linear_scalar_scale_branch(3.0)),
            ));

        let operand_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(4)]));
        let zero_operand_batch =
            ArrayBatch::new(operand_type.clone(), Tangent::<ArrayType, TestArray>::zero(operand_type), Some(0))
                .unwrap();

        let context = EagerContext::<
            ArrayType,
            Tangent<ArrayType, TestArray>,
            LinearArrayOperation<ArrayType, TestArray, TestArray>,
        >::new();
        let outputs = <LinearArrayOperation<ArrayType, TestArray, TestArray> as BatchableOperation<
            Tangent<ArrayType, TestArray>,
            EagerContext<
                ArrayType,
                Tangent<ArrayType, TestArray>,
                LinearArrayOperation<ArrayType, TestArray, TestArray>,
            >,
        >>::batch(&operation, &context, &[zero_operand_batch])
        .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), Some(0));
        match outputs[0].value() {
            Tangent::Value(v) => assert_eq!(v.values, vec![0.0, 0.0, 0.0, 0.0]),
            Tangent::Zero(_) => {
                // The all-zero short-circuit reports a symbolic-zero output. Either representation is correct
                // for downstream consumers — accept both.
            }
        }
    }

    #[test]
    fn test_jacrev_through_function_using_zero_like() {
        // `f(x) = x + zero_like(x)` is functionally the identity, but exercises the
        // `ZeroLikeOperation` rule through `jacrev`'s internal Jacobian batching path. Verifies
        // that the constant-op rule composes cleanly with reverse-mode autodiff.
        let jacobian = jacrev(&TestArrayDomain, |x| Ok(x.clone() + x.zero_like()), TestArray::scalar(2.0)).unwrap();
        let row = jacobian.rows();
        let block = row.partials();
        // d(x + 0) / dx = 1 at the scalar point.
        assert_close(block.values()[0], 1.0);
    }

    #[test]
    fn test_jacfwd_through_function_using_one_like() {
        // `f(x) = x + one_like(x)` shifts x by a constant; the Jacobian is still 1. Exercises
        // `OneLikeOperation` through jacfwd's internal batching.
        let jacobian = TestArrayDomain.jacfwd(|x| Ok(x.clone() + x.one_like()), TestArray::scalar(2.0)).unwrap();
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
                Operation = ArrayOperation<ArrayType, TestArray>,
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
        let jacobian =
            jacrev(&TestArrayDomain, |x| Ok(stage_constant_predicate_condition(x)), TestArray::scalar(4.0)).unwrap();
        assert_close(jacobian.rows().partials().values()[0], 2.0);

        let jacobian = TestArrayDomain
            .jacfwd(|x| Ok(stage_constant_predicate_condition(x)), TestArray::scalar(4.0))
            .unwrap();
        assert_close(jacobian.rows().partials().values()[0], 2.0);
    }

    /// Builds the `while (x < 8) { x = x + x }` doubling-loop fixture used by the while differentiation tests.
    fn doubling_while_operation()
    -> crate::operations::control_flow::WhileOperation<TestArray, ArrayOperation<ArrayType, TestArray>, ArrayType> {
        use crate::operations::compare::ComparisonDirection;
        type TestOp = ArrayOperation<ArrayType, TestArray>;

        let scalar_f64 = ArrayType::scalar(DataType::F64);
        let mut condition_builder = ProgramBuilder::<ArrayType, TestArray, TestOp>::new();
        let condition_state = condition_builder.add_input(scalar_f64.clone());
        let threshold = condition_builder.add_constant(TestArray::scalar(8.0));
        let predicate = condition_builder
            .add_instruction(CompareOperation::new(ComparisonDirection::LessThan), vec![condition_state, threshold])
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut body_builder = ProgramBuilder::<ArrayType, TestArray, TestOp>::new();
        let body_state = body_builder.add_input(scalar_f64);
        let doubled = body_builder.add_instruction(AddOperation, vec![body_state, body_state]).unwrap()[0];
        let body = body_builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![doubled], vec![Placeholder], vec![Placeholder])
            .unwrap();

        crate::operations::control_flow::WhileOperation::new(condition, body).unwrap()
    }

    #[test]
    fn test_while_jvp_propagates_tangents_through_staged_fused_loop() {
        // `while (x < 8) { x = x + x }` starting at `x = 1` doubles three times, so the primal output is 8 and the
        // forward-mode derivative is 2^3 = 8. Exercises the `ArrayOperation::While` JVP dispatch in an eager domain:
        // the rule stages the doubled-state fused linear loop, which direct JVP execution interprets immediately.
        let while_operation = doubling_while_operation();
        let (primal, tangent) = TestArrayDomain
            .jvp(
                move |x| {
                    let mut outputs =
                        x.context().stage_operation(ArrayOperation::While(Box::new(while_operation)), &[&x]).unwrap();
                    outputs.remove(0)
                },
                TestArray::scalar(1.0),
                TestArray::scalar(1.0),
            )
            .unwrap();
        assert_eq!(primal.values, vec![8.0]);
        assert_eq!(tangent.values, vec![8.0]);
    }

    #[test]
    fn test_while_linearize_reuses_pushforward_with_fresh_tangents() {
        // Linearizing eagerly through the doubling loop unrolls its three iterations (at `x = 1`) into a
        // straight-line pushforward — no `while` remains in the staged tangent program — that is reusable: each
        // fresh tangent replay is scaled by the captured per-iteration linear maps (`2^3 = 8`).
        let while_operation = doubling_while_operation();
        let (primal, pushforward) = TestArrayDomain
            .linearize(
                move |x| {
                    let mut outputs =
                        x.context().stage_operation(ArrayOperation::While(Box::new(while_operation)), &[&x])?;
                    Ok(outputs.remove(0))
                },
                TestArray::scalar(1.0),
            )
            .unwrap();
        assert_eq!(primal.values, vec![8.0]);
        assert!(!pushforward.program().to_string().contains("while"));
        let tangent_context = TestArrayDomain.context();
        assert_eq!(
            pushforward.apply(&tangent_context, TestArray::scalar(1.0)).map(|tangent| tangent.values),
            Ok(vec![8.0])
        );
        assert_eq!(
            pushforward.apply(&tangent_context, TestArray::scalar(2.5)).map(|tangent| tangent.values),
            Ok(vec![20.0])
        );
    }

    #[test]
    fn test_while_linearize_unrolls_state_dependent_products() {
        // State `(counter, value)` with body `(counter - 1, value * value)` and condition `counter > 0`: the body
        // pushforward of `value * value` captures `value` on both sides of the product rule, and eager linearization
        // unrolls the two iterations, closing each iteration's captured primal value into the inlined linear maps.
        // For `x_{n+1} = x_n^2` the tangent map is `t_{n+1} = 2 x_n t_n`: starting at `value = 3` with
        // `counter = 2`, the primal is `3^4 = 81` and the tangent of `(0, 1)` is `(2 * 3) * (2 * 9) = 108` (the
        // derivative of `x^4` at `x = 3`).
        use crate::operations::compare::ComparisonDirection;
        use crate::operations::control_flow::WhileOperation;
        type TestOp = ArrayOperation<ArrayType, TestArray>;

        let scalar_f64 = ArrayType::scalar(DataType::F64);
        let mut condition_builder = ProgramBuilder::<ArrayType, TestArray, TestOp>::new();
        let condition_counter = condition_builder.add_input(scalar_f64.clone());
        let _condition_value = condition_builder.add_input(scalar_f64.clone());
        let condition_zero = condition_builder.add_instruction(ZeroLikeOperation, vec![condition_counter]).unwrap()[0];
        let predicate = condition_builder
            .add_instruction(
                CompareOperation::new(ComparisonDirection::GreaterThan),
                vec![condition_counter, condition_zero],
            )
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![predicate], vec![Placeholder, Placeholder], vec![Placeholder])
            .unwrap();

        let mut body_builder = ProgramBuilder::<ArrayType, TestArray, TestOp>::new();
        let body_counter = body_builder.add_input(scalar_f64.clone());
        let body_value = body_builder.add_input(scalar_f64);
        let one = body_builder.add_instruction(OneLikeOperation, vec![body_counter]).unwrap()[0];
        let next_counter = body_builder.add_instruction(SubOperation, vec![body_counter, one]).unwrap()[0];
        let squared = body_builder.add_instruction(MulOperation, vec![body_value, body_value]).unwrap()[0];
        let body = body_builder
            .build::<Vec<TestArray>, Vec<TestArray>>(
                vec![next_counter, squared],
                vec![Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();
        let while_operation = WhileOperation::<TestArray, TestOp, ArrayType>::new(condition, body).unwrap();

        let ((counter_primal, value_primal), pushforward) = TestArrayDomain
            .linearize(
                move |(counter, value)| {
                    let mut outputs = counter
                        .context()
                        .stage_operation(ArrayOperation::While(Box::new(while_operation)), &[&counter, &value])?;
                    let value_output = outputs.remove(1);
                    Ok((outputs.remove(0), value_output))
                },
                (TestArray::scalar(2.0), TestArray::scalar(3.0)),
            )
            .unwrap();
        assert_eq!(counter_primal.values, vec![0.0]);
        assert_eq!(value_primal.values, vec![81.0]);

        // The unrolled pushforward is straight-line: the per-iteration product rules appear as captured-factor
        // `scale` maps and no `while` remains in the staged tangent program.
        assert!(pushforward.program().to_string().contains("scale"));
        assert!(!pushforward.program().to_string().contains("while"));

        let tangent_context = TestArrayDomain.context();
        let (counter_tangent, value_tangent) =
            pushforward.apply(&tangent_context, (TestArray::scalar(0.0), TestArray::scalar(1.0))).unwrap();
        assert_eq!(counter_tangent.values, vec![0.0]);
        assert_eq!(value_tangent.values, vec![108.0]);
    }

    #[test]
    fn test_jacfwd_through_while_batches_basis_tangents() {
        // `jacfwd` linearizes the loop once into a pushforward containing a staged fused linear while, then replays
        // the batched basis tangents through it: the pushforward is instantiated into a direct linear program and
        // interpreted over the value-level batching rules, so the fused linear while runs once over the stacked
        // basis tangents. The derivative of the doubling loop at `x = 1` is `2^3 = 8`.
        let while_operation = doubling_while_operation();
        let jacobian = TestArrayDomain
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
        let (value, gradient) = TestArrayDomain
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
        let (value, gradient) = TestArrayDomain
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
        let (output, pullback) = TestArrayDomain
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
        assert_eq!(pullback.interpret(TestArray::scalar(1.0)).map(|cotangent| cotangent.values), Ok(vec![8.0]));
        assert_eq!(pullback.interpret(TestArray::scalar(5.0)).map(|cotangent| cotangent.values), Ok(vec![40.0]));
    }

    #[test]
    fn test_while_vjp_computes_cotangents_through_unrolled_loop() {
        // Eager `vjp` transposes the unrolled straight-line pushforward of an *unbounded* loop into a reusable
        // pullback: the doubling loop at `x = 1` is locally `f(x) = 8 x`, so every output cotangent is scaled by 8.
        let while_operation = doubling_while_operation();
        let (output, pullback) = TestArrayDomain
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
        assert_eq!(pullback.interpret(TestArray::scalar(1.0)).map(|cotangent| cotangent.values), Ok(vec![8.0]));
        assert_eq!(pullback.interpret(TestArray::scalar(5.0)).map(|cotangent| cotangent.values), Ok(vec![40.0]));
    }

    /// Builds the three-lane cumulative-product [`ScanOperation`] (body `[carry, x] -> [carry * x, carry * x]`)
    /// used by the scan differentiation tests, optionally visiting the lanes in reverse order.
    fn product_scan_operation(
        reverse: bool,
    ) -> crate::operations::control_flow::ScanOperation<TestArray, ArrayOperation<ArrayType, TestArray>, ArrayType>
    {
        type TestOp = ArrayOperation<ArrayType, TestArray>;
        let mut body_builder = ProgramBuilder::<ArrayType, TestArray, TestOp>::new();
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
        crate::operations::control_flow::ScanOperation::new(body, 1, 3).unwrap().with_reverse(reverse)
    }

    #[test]
    fn test_scan_value_and_grad_computes_gradient_through_reversed_linear_scan() {
        // The headline scan capability: end-to-end reverse mode. `f(init, xs)` is the final carry of the
        // cumulative-product scan, so `f = init * xs[0] * xs[1] * xs[2] = 24` at `init = 1, xs = [2, 3, 4]`, with
        // gradient `24` w.r.t. `init` and `[12, 8, 6]` w.r.t. `xs`. The pullback runs the transposed linear scan
        // (same residual stacks, `reverse` flipped) — the static trip count is what makes this total, where the
        // staged linear `while` rejects transposition.
        let scan = product_scan_operation(false);
        let (value, (init_gradient, xs_gradient)) = TestArrayDomain
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
        let (output, pullback) = TestArrayDomain
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
        let (init_cotangent, xs_cotangent) = pullback.interpret(TestArray::scalar(1.0)).unwrap();
        assert_eq!(init_cotangent.values, vec![24.0]);
        assert_eq!(xs_cotangent.values, vec![12.0, 8.0, 6.0]);
        let (init_cotangent, xs_cotangent) = pullback.interpret(TestArray::scalar(2.0)).unwrap();
        assert_eq!(init_cotangent.values, vec![48.0]);
        assert_eq!(xs_cotangent.values, vec![24.0, 16.0, 12.0]);
    }

    #[test]
    fn test_scan_linearize_reuses_pushforward_with_fresh_tangents() {
        // Linearizing through the scan stages one linear scan over stored residual stacks — no unrolling, unlike
        // `while` in eager domains — and the resulting pushforward replays with fresh tangents:
        // `df = 24 d_init + 12 d_x0 + 8 d_x1 + 6 d_x2`.
        let scan = product_scan_operation(false);
        let (primal, pushforward) = TestArrayDomain
            .linearize(
                move |(init, xs)| {
                    let mut outputs =
                        init.context().stage_operation(ArrayOperation::Scan(Box::new(scan)), &[&init, &xs])?;
                    Ok(outputs.remove(0))
                },
                (TestArray::scalar(1.0), TestArray::vector(vec![2.0, 3.0, 4.0])),
            )
            .unwrap();
        assert_eq!(primal.values, vec![24.0]);
        assert!(pushforward.program().to_string().contains("scan"));
        let tangent_context = TestArrayDomain.context();
        let tangent = pushforward
            .apply(&tangent_context, (TestArray::scalar(1.0), TestArray::vector(vec![0.0, 0.0, 0.0])))
            .unwrap();
        assert_eq!(tangent.values, vec![24.0]);
        let tangent = pushforward
            .apply(&tangent_context, (TestArray::scalar(0.0), TestArray::vector(vec![1.0, 1.0, 1.0])))
            .unwrap();
        assert_eq!(tangent.values, vec![26.0]);
    }

    #[test]
    fn test_reversed_scan_jvp_and_grad_align_lanes() {
        // Pins the alignment invariant for `reverse = true`: the primal scan visits lanes from the back while
        // output lane `i` stays aligned with input lane `i`, and the linear scan runs with the same direction so
        // residual lane `i` is consumed exactly when tangent lane `i` is processed. The reversed cumulative
        // product has `ys = [x0 x1 x2, x1 x2, x2] = [24, 12, 4]`, so a unit tangent on `x1` gives
        // `dys = [x0 x2, x2, 0] = [8, 4, 0]`.
        let scan = product_scan_operation(true);
        let ((carry, ys), (carry_tangent, ys_tangent)) = TestArrayDomain
            .jvp(
                move |(init, xs)| {
                    let mut outputs =
                        init.context().stage_operation(ArrayOperation::Scan(Box::new(scan)), &[&init, &xs]).unwrap();
                    let ys = outputs.remove(1);
                    (outputs.remove(0), ys)
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
        let (output, pullback) = TestArrayDomain
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
        let (init_cotangent, xs_cotangent) = pullback.interpret(TestArray::scalar(1.0)).unwrap();
        assert_eq!(init_cotangent.values, vec![24.0]);
        assert_eq!(xs_cotangent.values, vec![12.0, 8.0, 6.0]);
    }

    #[test]
    fn test_array_operation_condition_interprets_runtime_predicate() {
        let condition = ConditionOperation::new(scalar_scale_branch(2.0), scalar_scale_branch(3.0)).unwrap();
        let operation = ArrayOperation::Condition(Box::new(condition));

        let predicate = TestArray::new(ArrayType::scalar(DataType::Boolean), vec![0.0]);
        assert_eq!(
            operation
                .interpret(&crate::EagerContext::new(), &[predicate, TestArray::scalar(4.0)])
                .map(|outputs| outputs[0].values[0]),
            Ok(12.0),
        );
    }

    #[test]
    fn test_condition_jvp_linearizes_predicate_chosen_branch() {
        let condition = ConditionOperation::new(scalar_scale_branch(2.0), scalar_scale_branch(3.0)).unwrap();
        let builder = Rc::new(RefCell::new(ProgramBuilder::<
            ArrayType,
            TestArray,
            LinearArrayOperation<ArrayType, TestArray, TestArray, Infallible, CapturedFactor<ArrayType, TestArray>>,
        >::new()));
        let residuals = Rc::new(RefCell::new(Vec::new()));
        let residual_atoms = Rc::new(RefCell::new(HashMap::new()));
        let mut context =
            TangentContext::new_with_residuals(&TestArrayDomain, builder.clone(), residuals.clone(), residual_atoms);
        let tangent_input = context.input(ArrayType::scalar(DataType::F64));
        let predicate = JvpTracer::from_zero_tangent(
            TestArray::new(ArrayType::scalar(DataType::Boolean), vec![1.0]),
            ArrayType::scalar(DataType::Boolean),
        );
        let outputs = condition
            .jvp(&mut context, &[predicate, JvpTracer::from_value(TestArray::scalar(4.0), tangent_input)])
            .unwrap();

        assert_eq!(outputs[0].primal().values[0], 8.0);
        let tangent_output = match outputs[0].tangent().clone() {
            crate::differentiation::Tangent::Value(tracer) => tracer.atom_id().unwrap(),
            crate::differentiation::Tangent::Zero(_) => {
                panic!("expected a concrete tangent output for the captured branch")
            }
        };
        drop(outputs);
        drop(context);
        let builder = Rc::try_unwrap(builder).unwrap().into_inner();
        let tangent_program =
            builder.build::<TestArray, TestArray>(vec![tangent_output], Placeholder, Placeholder).unwrap();
        let residuals = residuals.borrow();
        let tangent_program = tangent_program
            .map_operations(|operation| {
                ResidualizedOperation::<TestArrayDomain>::instantiate_residuals(operation, residuals.as_slice())
            })
            .unwrap();
        assert_eq!(tangent_program.interpret(TestArray::scalar(10.0)).map(|output| output.values[0]), Ok(20.0));
    }

    #[test]
    fn test_condition_jvp_linearizes_false_predicate_branch_and_reuses_pushforward() {
        // Mirror of `test_condition_jvp_linearizes_predicate_chosen_branch` at a FALSE-predicate primal point: the
        // bound primal condition evaluates the scale-by-3 branch (3 * 4 = 12), and the captured-predicate linear
        // condition replays that branch pushforward (tangent * 3) for any fresh tangent, including a second replay
        // that reuses the pushforward staged at the original primal point.
        let condition = ConditionOperation::new(scalar_scale_branch(2.0), scalar_scale_branch(3.0)).unwrap();
        let builder = Rc::new(RefCell::new(ProgramBuilder::<
            ArrayType,
            TestArray,
            LinearArrayOperation<ArrayType, TestArray, TestArray, Infallible, CapturedFactor<ArrayType, TestArray>>,
        >::new()));
        let residuals = Rc::new(RefCell::new(Vec::new()));
        let residual_atoms = Rc::new(RefCell::new(HashMap::new()));
        let mut context =
            TangentContext::new_with_residuals(&TestArrayDomain, builder.clone(), residuals.clone(), residual_atoms);
        let tangent_input = context.input(ArrayType::scalar(DataType::F64));
        let predicate = JvpTracer::from_zero_tangent(
            TestArray::new(ArrayType::scalar(DataType::Boolean), vec![0.0]),
            ArrayType::scalar(DataType::Boolean),
        );
        let outputs = condition
            .jvp(&mut context, &[predicate, JvpTracer::from_value(TestArray::scalar(4.0), tangent_input)])
            .unwrap();

        assert_eq!(outputs[0].primal().values[0], 12.0);
        let tangent_output = match outputs[0].tangent().clone() {
            crate::differentiation::Tangent::Value(tracer) => tracer.atom_id().unwrap(),
            crate::differentiation::Tangent::Zero(_) => {
                panic!("expected a concrete tangent output for the captured branch")
            }
        };
        drop(outputs);
        drop(context);
        let builder = Rc::try_unwrap(builder).unwrap().into_inner();
        let tangent_program =
            builder.build::<TestArray, TestArray>(vec![tangent_output], Placeholder, Placeholder).unwrap();
        let residuals = residuals.borrow();
        let tangent_program = tangent_program
            .map_operations(|operation| {
                ResidualizedOperation::<TestArrayDomain>::instantiate_residuals(operation, residuals.as_slice())
            })
            .unwrap();
        assert_eq!(tangent_program.interpret(TestArray::scalar(10.0)).map(|output| output.values[0]), Ok(30.0));
        assert_eq!(tangent_program.interpret(TestArray::scalar(-2.0)).map(|output| output.values[0]), Ok(-6.0));
    }

    #[test]
    fn test_condition_vjp_computes_branch_cotangents_for_runtime_predicate() {
        // f(p, x) = if p { 2 * x } else { 3 * x }. Reverse mode composes through the total linear-condition
        // transpose rule: the pullback runs the transposed branch program selected by the captured predicate, so
        // the operand cotangent is 2 * output cotangent at a TRUE-predicate primal point and 3 * output cotangent
        // at a FALSE one. The Boolean predicate has no tangent space, so its cotangent slot is always zero.
        let (output, pullback) = TestArrayDomain
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
        let (predicate_cotangent, operand_cotangent) = pullback.interpret(TestArray::scalar(5.0)).unwrap();
        assert_eq!(operand_cotangent.values, vec![10.0]);
        assert_eq!(predicate_cotangent.values, vec![0.0]);

        let (output, pullback) = TestArrayDomain
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
        let (predicate_cotangent, operand_cotangent) = pullback.interpret(TestArray::scalar(5.0)).unwrap();
        assert_eq!(operand_cotangent.values, vec![15.0]);
        assert_eq!(predicate_cotangent.values, vec![0.0]);
    }
}
