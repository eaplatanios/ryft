use approx::assert_abs_diff_eq;
use indoc::indoc;
use pretty_assertions::assert_eq;

use crate::arrays::{
    Array, ArrayBatch, ArrayOperation, ArrayType, DataType, Dimension, DimensionBounds, DimensionVariable, LogicalMesh,
    MeshAxis, MeshAxisType, RaggedAxis, Shape, Sharding, ShardingDimension,
};
use crate::batching::{BatchAxis, BatchableOperation, BatchedProgram, BatchingContext, batch};
use crate::contexts::EagerContext;
use crate::differentiation::differentiate_at;
use crate::macros::{check_operation_transposition, check_operation_type_inference};
use crate::programs::{Operation, TypeError};

use super::*;

fn test_mesh() -> LogicalMesh {
    LogicalMesh::new(vec![
        MeshAxis::new("b", 2, MeshAxisType::Explicit).unwrap(),
        MeshAxis::new("m", 2, MeshAxisType::Explicit).unwrap(),
        MeshAxis::new("n", 2, MeshAxisType::Explicit).unwrap(),
        MeshAxis::new("k", 2, MeshAxisType::Explicit).unwrap(),
    ])
    .unwrap()
}

fn plain_array(sizes: &[usize]) -> ArrayType {
    ArrayType::new(DataType::F32, Shape::new(sizes.iter().map(|size| Dimension::Static(*size)).collect()))
}

fn sharded_array(mesh: &LogicalMesh, sizes: &[usize], dimensions: Vec<ShardingDimension>) -> ArrayType {
    plain_array(sizes).with_sharding(Sharding::new(mesh.clone(), dimensions).unwrap()).unwrap()
}

#[test]
fn test_dot_accumulation_type() {
    // Type inference widens the output to the accumulation type for promotable operand types and rejects
    // non-promotable ones, combining with a requested output sharding, and differentiation.
    let operation = DotOperation::matmul().with_accumulation_type(DataType::F32);
    assert_eq!(operation.accumulation_type(), Some(DataType::F32));
    let lhs = ArrayType::new(DataType::F8E4M3FN, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]));
    let rhs = lhs.clone();
    let bf16_operand = ArrayType::new(DataType::BF16, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]));
    let output_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]));
    check_operation_type_inference!(
        operation = operation,
        cases = [
            {
                input_types = [lhs.clone(), rhs.clone()],
                output_types = [output_type.clone()],
            },
            {
                input_types = [bf16_operand.clone(), bf16_operand],
                output_types = [output_type],
            },
        ],
    );
    let narrowing = DotOperation::matmul().with_accumulation_type(DataType::F16);
    let f32_operand = plain_array(&[2, 2]);
    check_operation_type_inference!(
        operation = narrowing,
        cases = [{
            input_types = [f32_operand.clone(), f32_operand],
            error = "`dot` operand data type f32 cannot accumulate at data type f16",
        }],
    );
    let mesh = test_mesh();
    let sharded = DotOperation::matmul().with_accumulation_type(DataType::F32).with_output_sharding(
        Sharding::new(mesh, vec![ShardingDimension::Replicated, ShardingDimension::Replicated]).unwrap(),
    );
    check_operation_type_inference!(
        operation = sharded,
        cases = [{
            input_types = [lhs.clone(), rhs.clone()],
            error = "`dot` does not support combining an accumulation type with a requested output sharding yet",
        }],
    );

    // The eager reference backend upcasts the operands and accumulates at the accumulation type: every value
    // below is exactly representable in `f8e4m3fn`, so the `f32` results are exact.
    let lhs_values = Array::from_f64s(lhs.clone(), vec![0.5, 1.0, 1.5, 2.0]);
    let rhs_values = Array::from_f64s(rhs.clone(), vec![1.0, 0.5, 0.5, 1.0]);
    let product = lhs_values.dot_with_accumulation_type(&rhs_values, &DotDimensionNumbers::matmul(), DataType::F32);
    assert_eq!(
        product.r#type().as_ref(),
        &ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]))
    );
    assert_eq!(product.to_f64s(), vec![1.0, 1.25, 2.5, 2.75]);

    // Forward-mode differentiation stages accumulation-typed tangent dots over the operand-typed tangents, so
    // the output tangent lives at the accumulation type exactly like the primal output. Every value below is
    // exactly representable in `f8e4m3fn` and every product sum is exact in `f32`.
    let mut builder = crate::programs::builders::ProgramBuilder::<Array, ArrayOperation<Array>>::new();
    let lhs_input = builder.add_input(lhs.clone());
    let rhs_input = builder.add_input(rhs.clone());
    let output = builder
        .add_instruction(
            DotOperation::matmul().with_accumulation_type(DataType::F32),
            Vec::new(),
            vec![lhs_input, rhs_input],
            None,
        )
        .unwrap()[0];
    let program = builder
        .build::<Vec<Array>, Vec<Array>>(
            vec![output],
            vec![crate::parameters::Placeholder; 2],
            vec![crate::parameters::Placeholder],
        )
        .unwrap();
    let jvp = program.jvp().unwrap();
    assert_eq!(
        jvp.to_string(),
        indoc! {"
                lambda %0:f8e4m3fn[2, 2], %1:f8e4m3fn[2, 2], %2:f8e4m3fn[2, 2], %3:f8e4m3fn[2, 2] .
                let %4:f32[2, 2] = dot [
                    dimensions=(lhs_contracting=[1], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                    accumulation_type=f32,
                ] %0 %1
                    %5:f32[2, 2] = dot [
                        dimensions=(lhs_contracting=[1], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                        accumulation_type=f32,
                    ] %2 %1
                    %6:f32[2, 2] = dot [
                        dimensions=(lhs_contracting=[1], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                        accumulation_type=f32,
                    ] %0 %3
                    %7:f32[2, 2] = add %5 %6
                in (%4, %7)
            "}
        .trim_end(),
    );
    let jvp_outputs = jvp
        .interpret(vec![
            lhs_values.clone(),
            rhs_values.clone(),
            Array::from_f64s(lhs.clone(), vec![1.0, 1.0, 1.0, 1.0]),
            Array::from_f64s(rhs.clone(), vec![0.5, 0.5, 0.5, 0.5]),
        ])
        .unwrap();
    assert_eq!(jvp_outputs[0].to_f64s(), vec![1.0, 1.25, 2.5, 2.75]);
    assert_eq!(jvp_outputs[1].r#type().data_type(), DataType::F32);
    // Tangent = d_lhs · rhs + lhs · d_rhs = [[1.5, 1.5], [1.5, 1.5]] + [[0.75, 0.75], [1.75, 1.75]].
    assert_eq!(jvp_outputs[1].to_f64s(), vec![2.25, 2.25, 3.25, 3.25]);

    // The transpose rule contracts the adjoint at the accumulation type and converts the result back to the
    // linear operand's `f8e4m3fn` cotangent representation. With an identity output cotangent, the adjoint of
    // the linear RHS is exactly `lhsᵀ`.
    check_operation_transposition!(
        @exact,
        operation = DotOperation::matmul().with_accumulation_type(DataType::F32),
        cases = [{
            inputs = [
                (@known, lhs_values),
                (@linear(type = rhs.clone())),
            ],
            output_cotangents = [Array::from_f64s(
                ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)])),
                vec![1.0, 0.0, 0.0, 1.0],
            )],
            input_cotangents = [Array::from_f64s(rhs, vec![0.5, 1.5, 1.0, 2.0])],
        }],
    );

    // Batching lifts the dimension numbers while carrying the accumulation type, so per-item products still
    // accumulate at the widened type.
    let lifted = program
        .batched(
            2,
            ShardingDimension::Replicated,
            &[BatchAxis::new(0), BatchAxis::new(0)],
            crate::batching::ProgramBatchingOutputAxesPolicy::Natural,
        )
        .unwrap()
        .into_parts()
        .0;
    let batched_lhs_type = ArrayType::new(
        DataType::F8E4M3FN,
        Shape::new(vec![Dimension::Static(2), Dimension::Static(2), Dimension::Static(2)]),
    );
    let batched_lhs = Array::from_f64s(batched_lhs_type.clone(), vec![0.5, 1.0, 1.5, 2.0, 0.5, 1.0, 1.5, 2.0]);
    let batched_rhs = Array::from_f64s(batched_lhs_type, vec![1.0, 0.5, 0.5, 1.0, 1.0, 0.5, 0.5, 1.0]);
    let outputs = lifted.interpret(vec![batched_lhs, batched_rhs]).unwrap();
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].r#type().data_type(), DataType::F32);
    // Both batch items repeat the unbatched case, whose exact product is [[1, 1.25], [2.5, 2.75]].
    assert_eq!(outputs[0].to_f64s(), vec![1.0, 1.25, 2.5, 2.75, 1.0, 1.25, 2.5, 2.75]);
}

#[test]
fn test_dot_inference_with_dynamic_dimensions() {
    // Batched matrix multiplication contracting axis 2 of the LHS with axis 1 of the RHS over batching axis 0.
    let operation = DotOperation::new(DotDimensionNumbers::new(vec![2], vec![1], vec![0], vec![0]));

    // Dynamic dimension sizes that compare equal flow through inference into the output type: the dynamic
    // batching dimension is preserved and the equal bounded dynamic contracting dimensions are dropped.
    let batch = DimensionVariable::new("batch", DimensionBounds::unbounded());
    let contracting = DimensionVariable::new("contracting", DimensionBounds::non_negative(Some(4)).unwrap());
    let lhs = ArrayType::new(
        DataType::F64,
        Shape::new(vec![
            Dimension::Dynamic(batch.clone()),
            Dimension::Static(2),
            Dimension::Dynamic(contracting.clone()),
        ]),
    );
    let rhs = ArrayType::new(
        DataType::F64,
        Shape::new(vec![Dimension::Dynamic(batch.clone()), Dimension::Dynamic(contracting), Dimension::Static(3)]),
    );
    assert_eq!(
        operation.infer_output_types(&[lhs.clone(), rhs.clone()], &[]),
        Ok(vec![ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Dynamic(batch.clone()), Dimension::Static(2), Dimension::Static(3)]),
        )]),
    );

    // Static-vs-dynamic and unequal dynamic dimension pairs keep erroring under the strict size equality used
    // for batching and contracting dimensions.
    let static_rhs = ArrayType::new(
        DataType::F64,
        Shape::new(vec![Dimension::Dynamic(batch), Dimension::Static(4), Dimension::Static(3)]),
    );
    assert_eq!(
        operation.infer_output_types(&[lhs.clone(), static_rhs], &[]),
        Err(TypeError::invalid("`dot` contracting dimension sizes do not match (LHS axis 2, RHS axis 1)".to_string())),
    );
    let mismatched_batch_rhs = ArrayType::new(
        DataType::F64,
        Shape::new(vec![
            Dimension::Dynamic(DimensionVariable::new("dynamic", DimensionBounds::non_negative(Some(8)).unwrap())),
            Dimension::Dynamic(DimensionVariable::new("dynamic", DimensionBounds::non_negative(Some(4)).unwrap())),
            Dimension::Static(3),
        ]),
    );
    assert_eq!(
        operation.infer_output_types(&[lhs, mismatched_batch_rhs], &[]),
        Err(TypeError::invalid("`dot` batching dimension sizes do not match (LHS axis 0, RHS axis 0)".to_string())),
    );
}

#[test]
fn test_dot_inference_batched_sharding_propagation() {
    let mesh = test_mesh();
    let operation = DotOperation::new(DotDimensionNumbers::new(vec![2], vec![1], vec![0], vec![0]));
    // The batch dimension merges the more informative entry (LHS `b` over RHS replicated), and the result
    // dimensions are copied from their owning operands.
    let lhs = sharded_array(
        &mesh,
        &[2, 4, 8],
        vec![ShardingDimension::sharded(["b"]), ShardingDimension::sharded(["m"]), ShardingDimension::replicated()],
    );
    let rhs = sharded_array(
        &mesh,
        &[2, 8, 16],
        vec![ShardingDimension::replicated(), ShardingDimension::replicated(), ShardingDimension::sharded(["n"])],
    );
    assert_eq!(
        operation.infer_output_types(&[lhs, rhs], &[]),
        Ok(vec![sharded_array(
            &mesh,
            &[2, 4, 16],
            vec![
                ShardingDimension::sharded(["b"]),
                ShardingDimension::sharded(["m"]),
                ShardingDimension::sharded(["n"]),
            ],
        )]),
    );
}

#[test]
fn test_dot_inference_matmul_sharding_propagation() {
    let mesh = test_mesh();
    let operation = DotOperation::matmul();
    // Fully replicated operands stay fully replicated.
    let replicated_lhs =
        sharded_array(&mesh, &[4, 8], vec![ShardingDimension::replicated(), ShardingDimension::replicated()]);
    let replicated_rhs =
        sharded_array(&mesh, &[8, 16], vec![ShardingDimension::replicated(), ShardingDimension::replicated()]);
    assert_eq!(
        operation.infer_output_types(&[replicated_lhs, replicated_rhs], &[]),
        Ok(vec![sharded_array(
            &mesh,
            &[4, 16],
            vec![ShardingDimension::replicated(), ShardingDimension::replicated()],
        )]),
    );

    // `[M@m, K] · [K, N@n] -> [M@m, N@n]`.
    let lhs = sharded_array(&mesh, &[4, 8], vec![ShardingDimension::sharded(["m"]), ShardingDimension::replicated()]);
    let rhs = sharded_array(&mesh, &[8, 16], vec![ShardingDimension::replicated(), ShardingDimension::sharded(["n"])]);
    assert_eq!(
        operation.infer_output_types(&[lhs, rhs], &[]),
        Ok(vec![sharded_array(
            &mesh,
            &[4, 16],
            vec![ShardingDimension::sharded(["m"]), ShardingDimension::sharded(["n"])],
        )]),
    );
}

#[test]
fn test_dot_inference_one_sided_sharding_propagation() {
    let mesh = test_mesh();
    let operation = DotOperation::matmul();
    // A missing operand sharding is treated as fully replicated on the present operand's mesh.
    let lhs = sharded_array(&mesh, &[4, 8], vec![ShardingDimension::sharded(["m"]), ShardingDimension::replicated()]);
    assert_eq!(
        operation.infer_output_types(&[lhs, plain_array(&[8, 16])], &[]),
        Ok(vec![sharded_array(
            &mesh,
            &[4, 16],
            vec![ShardingDimension::sharded(["m"]), ShardingDimension::replicated()],
        )]),
    );
    // Without any operand shardings, the output carries none.
    assert_eq!(
        operation.infer_output_types(&[plain_array(&[4, 8]), plain_array(&[8, 16])], &[]),
        Ok(vec![plain_array(&[4, 16])]),
    );
}

#[test]
fn test_dot_inference_batch_sharding_conflict() {
    let mesh = test_mesh();
    let operation = DotOperation::new(DotDimensionNumbers::new(vec![2], vec![1], vec![0], vec![0]));
    let lhs = sharded_array(
        &mesh,
        &[2, 4, 8],
        vec![ShardingDimension::sharded(["b"]), ShardingDimension::replicated(), ShardingDimension::replicated()],
    );
    let rhs = sharded_array(
        &mesh,
        &[2, 8, 16],
        vec![ShardingDimension::sharded(["m"]), ShardingDimension::replicated(), ShardingDimension::replicated()],
    );
    assert_eq!(
        operation.infer_output_types(&[lhs, rhs], &[]),
        Err(TypeError::invalid(
            "`dot` batching dimensions must have consistent shardings, but got {'b'} and {'m'}".to_string()
        )),
    );
}

#[test]
fn test_dot_inference_contracting_sharding_errors() {
    let mesh = test_mesh();
    let operation = DotOperation::matmul();
    // Identically sharded contracting dimensions make the output sharding ambiguous.
    let lhs = sharded_array(&mesh, &[4, 8], vec![ShardingDimension::replicated(), ShardingDimension::sharded(["k"])]);
    let rhs = sharded_array(&mesh, &[8, 16], vec![ShardingDimension::sharded(["k"]), ShardingDimension::replicated()]);
    assert_eq!(
        operation.infer_output_types(&[lhs.clone(), rhs], &[]),
        Err(TypeError::invalid(
            "`dot` contracting dimensions are sharded, making the output sharding ambiguous; request an \
                          explicit output sharding (e.g., one with unreduced axes) to resolve it"
                .to_string()
        )),
    );
    // Differently sharded contracting dimensions are inconsistent.
    let mismatched_rhs =
        sharded_array(&mesh, &[8, 16], vec![ShardingDimension::sharded(["m"]), ShardingDimension::replicated()]);
    assert_eq!(
        operation.infer_output_types(&[lhs.clone(), mismatched_rhs], &[]),
        Err(TypeError::invalid(
            "`dot` contracting dimensions must have consistent shardings, but got {'k'} and {'m'}".to_string()
        )),
    );
    // A contracting dimension sharded on only one operand is allowed, and its sharding is dropped.
    let replicated_rhs =
        sharded_array(&mesh, &[8, 16], vec![ShardingDimension::replicated(), ShardingDimension::replicated()]);
    assert_eq!(
        operation.infer_output_types(&[lhs, replicated_rhs], &[]),
        Ok(vec![sharded_array(
            &mesh,
            &[4, 16],
            vec![ShardingDimension::replicated(), ShardingDimension::replicated()],
        )]),
    );
}

#[test]
fn test_dot_inference_mesh_mismatch() {
    let mesh = test_mesh();
    let other_mesh = LogicalMesh::new(vec![MeshAxis::new("m", 4, MeshAxisType::Explicit).unwrap()]).unwrap();
    let operation = DotOperation::matmul();
    let lhs = sharded_array(&mesh, &[4, 8], vec![ShardingDimension::sharded(["m"]), ShardingDimension::replicated()]);
    let rhs =
        sharded_array(&other_mesh, &[8, 16], vec![ShardingDimension::sharded(["m"]), ShardingDimension::replicated()]);
    assert_eq!(
        operation.infer_output_types(&[lhs, rhs], &[]),
        Err(TypeError::invalid("`dot` operand shardings must use the same mesh".to_string())),
    );
}

#[test]
fn test_dot_inference_unreduced_and_reduced_operands() {
    let mesh = test_mesh();
    let operation = DotOperation::matmul();
    // Unreduced operands are rejected: the pending reduction must be discharged before the contraction.
    let unreduced_lhs = plain_array(&[4, 8])
        .with_sharding(
            Sharding::new(mesh.clone(), vec![ShardingDimension::replicated(), ShardingDimension::replicated()])
                .unwrap()
                .with_unreduced_axes(["k"])
                .unwrap(),
        )
        .unwrap();
    assert_eq!(
        operation.infer_output_types(&[unreduced_lhs, plain_array(&[8, 16])], &[]),
        Err(TypeError::invalid("`dot` operands cannot be unreduced".to_string())),
    );

    // Reduced operands are legal (this is what lets adjoint dots consume reduced cotangents), and their reduced
    // axes are unioned into the output sharding.
    let reduced_lhs = plain_array(&[4, 8])
        .with_sharding(
            Sharding::new(mesh.clone(), vec![ShardingDimension::replicated(), ShardingDimension::replicated()])
                .unwrap()
                .with_reduced_axes(["k"])
                .unwrap(),
        )
        .unwrap();
    assert_eq!(
        operation.infer_output_types(&[reduced_lhs, plain_array(&[8, 16])], &[]),
        Ok(vec![
            plain_array(&[4, 16])
                .with_sharding(
                    Sharding::new(mesh, vec![ShardingDimension::replicated(), ShardingDimension::replicated()],)
                        .unwrap()
                        .with_reduced_axes(["k"])
                        .unwrap(),
                )
                .unwrap()
        ]),
    );
}

#[test]
fn test_dot_inference_strips_auto_axes() {
    let mesh = LogicalMesh::new(vec![
        MeshAxis::new("a", 2, MeshAxisType::Auto).unwrap(),
        MeshAxis::new("m", 2, MeshAxisType::Explicit).unwrap(),
    ])
    .unwrap();
    let operation = DotOperation::matmul();
    let lhs = sharded_array(&mesh, &[4, 8], vec![ShardingDimension::sharded(["a"]), ShardingDimension::replicated()]);
    let rhs = sharded_array(&mesh, &[8, 16], vec![ShardingDimension::replicated(), ShardingDimension::sharded(["m"])]);
    assert_eq!(
        operation.infer_output_types(&[lhs, rhs], &[]),
        Ok(vec![sharded_array(
            &mesh,
            &[4, 16],
            vec![ShardingDimension::replicated(), ShardingDimension::sharded(["m"])],
        )]),
    );
}

#[test]
fn test_dot_inference_output_sharding_bypass_and_validation() {
    let mesh = test_mesh();
    // The requested output sharding bypasses the batch consistency checks (here, conflicting batch shardings).
    let lhs = sharded_array(
        &mesh,
        &[2, 4, 8],
        vec![ShardingDimension::sharded(["b"]), ShardingDimension::replicated(), ShardingDimension::replicated()],
    );
    let rhs = sharded_array(
        &mesh,
        &[2, 8, 16],
        vec![ShardingDimension::sharded(["m"]), ShardingDimension::replicated(), ShardingDimension::replicated()],
    );
    let requested = Sharding::new(
        mesh.clone(),
        vec![ShardingDimension::sharded(["b"]), ShardingDimension::replicated(), ShardingDimension::sharded(["n"])],
    )
    .unwrap();
    let operation = DotOperation::new(DotDimensionNumbers::new(vec![2], vec![1], vec![0], vec![0]))
        .with_output_sharding(requested.clone());
    assert_eq!(
        operation.infer_output_types(&[lhs.clone(), rhs.clone()], &[]),
        Ok(vec![plain_array(&[2, 4, 16]).with_sharding(requested).unwrap()]),
    );

    // Rank validation.
    let rank_mismatched = Sharding::new(mesh.clone(), vec![ShardingDimension::replicated()]).unwrap();
    let operation = DotOperation::new(DotDimensionNumbers::new(vec![2], vec![1], vec![0], vec![0]))
        .with_output_sharding(rank_mismatched);
    assert_eq!(
        operation.infer_output_types(&[lhs.clone(), rhs.clone()], &[]),
        Err(TypeError::invalid("`dot` output sharding rank (1) does not match the output rank (3)".to_string())),
    );

    // Mesh validation.
    let other_mesh = LogicalMesh::new(vec![MeshAxis::new("m", 4, MeshAxisType::Explicit).unwrap()]).unwrap();
    let other_mesh_sharding = Sharding::replicated(other_mesh, 3);
    let operation = DotOperation::new(DotDimensionNumbers::new(vec![2], vec![1], vec![0], vec![0]))
        .with_output_sharding(other_mesh_sharding);
    assert_eq!(
        operation.infer_output_types(&[lhs, rhs], &[]),
        Err(TypeError::invalid("`dot` output sharding must use the same mesh as the operands".to_string())),
    );

    // Auto mesh axes cannot be requested explicitly.
    let auto_mesh = LogicalMesh::new(vec![MeshAxis::new("a", 2, MeshAxisType::Auto).unwrap()]).unwrap();
    let auto_sharding =
        Sharding::new(auto_mesh, vec![ShardingDimension::sharded(["a"]), ShardingDimension::replicated()]).unwrap();
    let operation = DotOperation::matmul().with_output_sharding(auto_sharding);
    assert_eq!(
        operation.infer_output_types(&[plain_array(&[4, 8]), plain_array(&[8, 16])], &[]),
        Err(TypeError::invalid("`dot` output sharding cannot reference auto mesh axes".to_string())),
    );
}

#[test]
fn test_dot_inference_unreduced_output_sharding() {
    let mesh = test_mesh();
    let lhs = sharded_array(&mesh, &[4, 8], vec![ShardingDimension::replicated(), ShardingDimension::sharded(["k"])]);
    let rhs = sharded_array(&mesh, &[8, 16], vec![ShardingDimension::sharded(["k"]), ShardingDimension::replicated()]);
    // Identically sharded contracting dimensions plus a matching unreduced set produce an unreduced output.
    let unreduced = Sharding::new(mesh.clone(), vec![ShardingDimension::replicated(), ShardingDimension::replicated()])
        .unwrap()
        .with_unreduced_axes(["k"])
        .unwrap();
    let operation = DotOperation::matmul().with_output_sharding(unreduced.clone());
    assert_eq!(
        operation.infer_output_types(&[lhs.clone(), rhs.clone()], &[]),
        Ok(vec![plain_array(&[4, 16]).with_sharding(unreduced.clone()).unwrap()]),
    );

    // The contracting dimensions must be sharded identically.
    let replicated_rhs =
        sharded_array(&mesh, &[8, 16], vec![ShardingDimension::replicated(), ShardingDimension::replicated()]);
    let operation = DotOperation::matmul().with_output_sharding(unreduced.clone());
    assert_eq!(
        operation.infer_output_types(&[lhs.clone(), replicated_rhs.clone()], &[]),
        Err(TypeError::invalid(
            "`dot` contracting dimensions must be sharded identically when the output sharding is unreduced"
                .to_string()
        )),
    );

    // The unreduced set must equal the axes that shard the contracting dimensions.
    let mismatched =
        Sharding::new(mesh.clone(), vec![ShardingDimension::replicated(), ShardingDimension::replicated()])
            .unwrap()
            .with_unreduced_axes(["n"])
            .unwrap();
    let operation = DotOperation::matmul().with_output_sharding(mismatched);
    assert_eq!(
        operation.infer_output_types(&[lhs, rhs], &[]),
        Err(TypeError::invalid(
            "`dot` output sharding unreduced axes must equal the axes that shard the contracting dimensions"
                .to_string()
        )),
    );

    // Unsharded contracting dimensions cannot produce an unreduced output.
    let operation = DotOperation::matmul().with_output_sharding(unreduced);
    assert_eq!(
        operation.infer_output_types(
            &[
                replicated_rhs.clone(),
                sharded_array(&mesh, &[16, 4], vec![ShardingDimension::replicated(), ShardingDimension::replicated()],)
            ],
            &[]
        ),
        Err(TypeError::invalid(
            "`dot` output sharding unreduced axes must equal the axes that shard the contracting dimensions"
                .to_string()
        )),
    );
}

#[test]
fn test_dot_operation_output_sharding_builder_and_render() {
    let mesh = test_mesh();
    let sharding =
        Sharding::new(mesh, vec![ShardingDimension::sharded(["m"]), ShardingDimension::replicated()]).unwrap();
    let operation = DotOperation::matmul().with_output_sharding(sharding.clone());
    assert_eq!(operation.output_sharding(), Some(&sharding));
    assert_eq!(DotOperation::matmul().output_sharding(), None);
    // The output sharding is rendered only when present.
    assert!(!DotOperation::matmul().to_string().contains("output_sharding="));
    assert!(operation.to_string().contains(&format!("output_sharding={sharding}")));
}

#[test]
fn test_dot_batching_stages_the_lifted_output_sharding() {
    use std::rc::Rc;

    use crate::arrays::ArrayBatch;
    use crate::batching::{BatchAxis, BatchableOperation, BatchingContext};
    use crate::parameters::Placeholder;
    use crate::tracing::TracingContext;

    let mesh = test_mesh();
    let output_sharding =
        Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["m"]), ShardingDimension::sharded(["n"])])
            .unwrap();
    let operation = DotOperation::matmul().with_output_sharding(output_sharding.clone());

    // Batch the operation over tracer inputs, which is how program batching applies lifted operations: the
    // staged batched dot must carry the lifted output sharding instead of dropping it.
    let context = TracingContext::<ArrayType, ArrayOperation<ArrayType>>::new();
    let builder = context.builder().clone();
    let lhs_atom = builder.borrow_mut().add_input(plain_array(&[2, 4, 8]));
    let rhs_atom = builder.borrow_mut().add_input(plain_array(&[2, 8, 16]));
    let batching_context = BatchingContext::new(context.clone(), 2);
    let lhs = {
        let value = context.tracer(lhs_atom, None);
        ArrayBatch::new(value, Some(0))
    }
    .unwrap();
    let rhs = {
        let value = context.tracer(rhs_atom, None);
        ArrayBatch::new(value, Some(0))
    }
    .unwrap();
    let outputs = operation.batch(&batching_context, &crate::EmptyRegionDriver, &[lhs, rhs]).unwrap().into_parts().0;
    assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
    let output_atom = outputs[0].value().atom_id().unwrap();
    drop(outputs);
    drop(batching_context);
    drop(context);

    let builder = Rc::try_unwrap(builder).expect("batching should not hold on to the builder").into_inner();
    let program = builder
        .build::<Vec<ArrayType>, Vec<ArrayType>>(vec![output_atom], vec![Placeholder; 2], vec![Placeholder])
        .unwrap();
    let lifted_sharding = output_sharding.with_inserted_dimension(0, ShardingDimension::replicated()).unwrap();
    assert!(program.to_string().contains(&format!("output_sharding={lifted_sharding}")));
}

#[test]
fn test_dot_batching_preserves_materialized_batch_placement() {
    use std::rc::Rc;

    use crate::arrays::{Array, ArrayBatch, ArrayOperation};
    use crate::batching::{BatchAxis, BatchableOperation, BatchingContext};
    use crate::parameters::Placeholder;
    use crate::tracing::TracingContext;

    for axis_type in [MeshAxisType::Explicit, MeshAxisType::Manual] {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, axis_type).unwrap()]).unwrap();
        let lhs_sharding = Sharding::new(
            mesh.clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated(), ShardingDimension::replicated()],
        )
        .unwrap()
        .with_varying_manual_axes((axis_type == MeshAxisType::Manual).then_some("x"))
        .unwrap();
        let lhs_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Static(2), Dimension::Static(2), Dimension::Static(2)]),
        )
        .with_sharding(lhs_sharding)
        .unwrap();
        let rhs_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(1)]))
            .with_sharding(Sharding::replicated(mesh, 2))
            .unwrap();
        let parent = TracingContext::<Array, ArrayOperation<Array>>::new();
        let builder = parent.builder().clone();
        let lhs_atom = builder.borrow_mut().add_input(lhs_type.clone());
        let rhs_atom = builder.borrow_mut().add_input(rhs_type);
        let lhs = ArrayBatch::new(parent.tracer(lhs_atom, None), BatchAxis::new(0)).unwrap();
        let rhs = ArrayBatch::replicated(parent.tracer(rhs_atom, None));
        let context = BatchingContext::new(parent.clone(), 2).with_axis_sharding(ShardingDimension::sharded(["x"]));

        let outputs = DotOperation::matmul()
            .batch(&context, &crate::EmptyRegionDriver, &[lhs, rhs])
            .unwrap()
            .into_parts()
            .0;

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        let output_atom = outputs[0].value().atom_id().unwrap();
        drop(outputs);
        drop(context);
        drop(parent);

        let builder = Rc::try_unwrap(builder).expect("batching should not retain the tracing builder").into_inner();
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output_atom], vec![Placeholder, Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.output_types()[0].sharding().unwrap().dimensions(),
            &[ShardingDimension::sharded(["x"]), ShardingDimension::replicated(), ShardingDimension::replicated(),],
        );
    }
}

#[test]
fn test_dot_batching_lifts_dimension_numbers() {
    // x has shape [3, 4]; outer batch over axis 0 produces per-item rank-1 vectors. Inside,
    // we want every per-item vector dotted with itself, giving a per-item scalar; batch
    // over the leading axis then yields a length-3 vector of dot products.
    let x_data: Vec<f64> = (1..=12).map(|value| value as f64).collect();
    let x = Array::matrix(3, 4, x_data);

    let output: Array = batch(
        |row| Ok(row.dot(&row, &DotDimensionNumbers::inner_product())),
        x,
        BatchAxis::new(0),
        BatchAxis::new(0),
        None,
    )
    .unwrap();

    assert_eq!(output.r#type().into_owned(), ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)])),);
    // Batch item 0: [1,2,3,4]·[1,2,3,4] = 30. Batch item 1: [5,6,7,8]·[5,6,7,8] = 174. Batch item 2: 446.
    for (actual, expected) in output.to_f64s().iter().zip([30.0_f64, 174.0, 446.0].iter()) {
        assert_abs_diff_eq!(*actual, *expected, epsilon = 1e-9);
    }

    // A replicated operand is broadcast across the mapped operand's batch axis before the dot dimensions are
    // lifted. Each row therefore contracts against the same right-hand vector.
    let lhs = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let lhs = ArrayBatch::new(lhs, BatchAxis::new(0)).unwrap();
    let rhs = ArrayBatch::replicated(Array::vector(vec![10.0, 100.0, 1000.0]));
    let outputs = DotOperation::new(DotDimensionNumbers::inner_product())
        .batch(&BatchingContext::new(EagerContext::<Array>::new(), 2), &crate::EmptyRegionDriver, &[lhs, rhs])
        .unwrap()
        .into_parts()
        .0;
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
    assert_eq!(outputs[0].value().to_f64s(), vec![3210.0, 6540.0]);
}

#[test]
fn test_dot_batching_validates_mapped_extents() {
    use crate::batching::BatchingContext;
    use crate::tracing::TracingContext;

    let context = TracingContext::<ArrayType, ArrayOperation<ArrayType>>::new();
    let batching_context = BatchingContext::new(context.clone(), 2);
    let input = |r#type: ArrayType| {
        let atom = context.builder().borrow_mut().add_input(r#type.clone());
        context.tracer(atom, Some(r#type))
    };
    let operation = DotOperation::new(DotDimensionNumbers::inner_product());
    let dynamic_rows = |variable: &DimensionVariable| {
        input(ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Dynamic(variable.clone()), Dimension::Static(3)]),
        ))
    };

    // Two mapped operands whose mapped axes carry different statically known extents cannot describe one batch.
    assert_eq!(
        operation
            .batch(
                &batching_context,
                &crate::EmptyRegionDriver,
                &[
                    ArrayBatch::new(input(plain_array(&[2, 3])), BatchAxis::new(0)).unwrap(),
                    ArrayBatch::new(input(plain_array(&[3, 3])), BatchAxis::new(0)).unwrap(),
                ],
            )
            .map(|outputs| outputs.into_parts().0)
            .unwrap_err(),
        BatchingError::MismatchedBatchSizes { expected: 2, actual: 3 },
    );

    // Two mapped operands sharing one dynamic mapped extent describe the same batch, so the lifted contraction is
    // staged with that dimension on its batching dimension.
    let variable = DimensionVariable::new("batch", DimensionBounds::new(1, Some(5)).unwrap());
    let outputs = operation
        .batch(
            &batching_context,
            &crate::EmptyRegionDriver,
            &[
                ArrayBatch::new(dynamic_rows(&variable), BatchAxis::new(0)).unwrap(),
                ArrayBatch::new(dynamic_rows(&variable), BatchAxis::new(0)).unwrap(),
            ],
        )
        .unwrap()
        .into_parts()
        .0;
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
    assert_eq!(
        outputs[0].r#type().into_owned(),
        ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(variable.clone())])),
    );

    // Dynamic extents are compared by variable identity, so two independently declared variables are two
    // independent extents even when their bounds agree.
    let other = DimensionVariable::new("other", DimensionBounds::new(1, Some(5)).unwrap());
    assert_eq!(
        operation
            .batch(
                &batching_context,
                &crate::EmptyRegionDriver,
                &[
                    ArrayBatch::new(dynamic_rows(&variable), BatchAxis::new(0)).unwrap(),
                    ArrayBatch::new(dynamic_rows(&other), BatchAxis::new(0)).unwrap(),
                ],
            )
            .map(|outputs| outputs.into_parts().0)
            .unwrap_err(),
        BatchingError::MisalignedBatchAxes {
            message: "`dot` operands map different batch extents `batch` and `other`".to_string(),
        },
    );
}

#[test]
fn test_dot_batching_propagates_free_ragged_axes() {
    // Per item, a ragged `[length, 2]` matrix contracts its dense trailing axis against a `[2]` vector, so the
    // ragged axis is a free axis of the contraction and survives into the `[length]` per-item result. The lifted
    // dot lays its result out as the batching dimension followed by the LHS free axes, so the ragged axis moves
    // from packed axis 1 to output axis 1 and the mapped axis its extents index stays at output axis 0.
    let variable = DimensionVariable::new("length", DimensionBounds::new(0, Some(3)).unwrap());
    let extents = Array::vector(vec![1_i32, 3]);
    let lhs = ArrayBatch::new(
        Array::from_f64s(plain_array(&[2, 3, 2]), (1..=12).map(f64::from).collect()),
        BatchAxis::new(0),
    )
    .unwrap()
    .with_ragged_axes(vec![RaggedAxis::new(1, extents.clone(), variable.clone(), vec![0])])
    .unwrap();
    let rhs = ArrayBatch::new(Array::matrix(2, 2, vec![1.0_f32, 10.0, 100.0, 1000.0]), BatchAxis::new(0)).unwrap();
    let (outputs, evidence) = DotOperation::new(DotDimensionNumbers::new(vec![1], vec![0], Vec::new(), Vec::new()))
        .batch(&BatchingContext::new(EagerContext::<Array>::new(), 2), &crate::EmptyRegionDriver, &[lhs, rhs])
        .unwrap()
        .into_parts();

    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
    assert_eq!(outputs[0].r#type().into_owned(), plain_array(&[2, 3]));
    assert_eq!(outputs[0].ragged_axes(), &[RaggedAxis::new(1, extents, variable.clone(), vec![0])]);
    assert_eq!(
        outputs[0].unbatched_type(),
        ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(variable)])),
    );
    // Nothing was contracted away, so the rule claims no consumption evidence and the padded rows keep whatever
    // the dense contraction produced for them.
    assert!(evidence.is_empty());
    assert_eq!(outputs[0].value().to_f64s(), vec![21.0, 43.0, 65.0, 8700.0, 10900.0, 13100.0]);
}

#[test]
fn test_dot_batching_rejects_unsupported_ragged_configurations() {
    let variable = DimensionVariable::new("length", DimensionBounds::new(0, Some(3)).unwrap());
    let extents = Array::vector(vec![1_i32, 3]);
    let ragged_matrix = || {
        ArrayBatch::new(Array::from_f64s(plain_array(&[2, 3, 2]), (1..=12).map(f64::from).collect()), BatchAxis::new(0))
            .unwrap()
            .with_ragged_axes(vec![RaggedAxis::new(1, extents.clone(), variable.clone(), vec![0])])
            .unwrap()
    };
    let context = BatchingContext::new(EagerContext::<Array>::new(), 2);

    // Contracting the ragged axis requires zeroing its padding, which static array batching cannot stage.
    assert_eq!(
        DotOperation::new(DotDimensionNumbers::new(vec![0], vec![0], Vec::new(), Vec::new()))
            .batch(&context, &crate::EmptyRegionDriver, &[ragged_matrix(), ragged_matrix()])
            .map(|outputs| outputs.into_parts().0)
            .unwrap_err(),
        BatchingError::UnsupportedOperation {
            message: "static array batching cannot zero-pad bounded ragged axes".to_string(),
        },
    );

    // A ragged axis declared as a batching dimension of the dot itself would require both operands to agree on
    // per-item extents along paired batch dimensions, which nothing here establishes.
    assert_eq!(
        DotOperation::new(DotDimensionNumbers::new(vec![1], vec![1], vec![0], vec![0]))
            .batch(&context, &crate::EmptyRegionDriver, &[ragged_matrix(), ragged_matrix()])
            .map(|outputs| outputs.into_parts().0)
            .unwrap_err(),
        BatchingError::UnsupportedOperation {
            message: "`dot` does not support bounded ragged dimension `length` on a batching dimension".to_string(),
        },
    );

    // A replicated ragged operand gains its batch axis through a broadcast that carries no per-item extents.
    let replicated = ArrayBatch::replicated(Array::from_f64s(plain_array(&[3, 2]), (1..=6).map(f64::from).collect()))
        .with_ragged_axes(vec![RaggedAxis::new(0, extents, variable, vec![])])
        .unwrap();
    assert_eq!(
        DotOperation::matmul()
            .batch(
                &context,
                &crate::EmptyRegionDriver,
                &[
                    ArrayBatch::new(Array::matrix(2, 3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]), BatchAxis::new(0))
                        .unwrap(),
                    replicated,
                ],
            )
            .map(|outputs| outputs.into_parts().0)
            .unwrap_err(),
        BatchingError::UnsupportedOperation {
            message: "`dot` does not support bounded ragged dimension `length` on a replicated operand".to_string(),
        },
    );
}

#[test]
fn test_dot_dense_jacobians() {
    let inputs = (Array::vector(vec![2.0, 3.0, 5.0]), Array::vector(vec![7.0, 11.0, 13.0]));

    // Reverse mode batches the pullback's adjoint dots over output-coordinate cotangents.
    let jacobian = differentiate_at(inputs.clone())
        .jacobian_reverse(|(left, right)| Ok(left.dot(&right, &DotDimensionNumbers::inner_product())))
        .unwrap();
    let blocks = jacobian.iter_blocks().collect::<Vec<_>>();
    let [left, right] = blocks.as_slice() else { unreachable!() };
    assert_eq!(left.value().to_f64s(), vec![7.0, 11.0, 13.0]);
    assert_eq!(right.value().to_f64s(), vec![2.0, 3.0, 5.0]);

    // Forward mode batches input-coordinate basis tangents through the dot pushforward.
    let jacobian = differentiate_at(inputs)
        .jacobian_forward(|(left, right)| Ok(left.dot(&right, &DotDimensionNumbers::inner_product())))
        .unwrap();
    let blocks = jacobian.iter_blocks().collect::<Vec<_>>();
    let [left, right] = blocks.as_slice() else { unreachable!() };
    assert_eq!(left.value().to_f64s(), vec![7.0, 11.0, 13.0]);
    assert_eq!(right.value().to_f64s(), vec![2.0, 3.0, 5.0]);
}

#[test]
fn test_dot_partitioned_transpose_computes_operand_adjoints() {
    let matmul = DotDimensionNumbers::matmul();
    let left = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let right = Array::matrix(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    let cotangent = Array::matrix(2, 2, vec![1.0, -2.0, 0.5, 3.0]);
    check_operation_transposition!(
        @exact,
        operation = DotOperation::new(matmul),
        cases = [
            {
                inputs = [
                    (@known, left),
                    (@linear(type = right.r#type().into_owned())),
                ],
                output_cotangents = [cotangent.clone()],
                input_cotangents = [Array::matrix(3, 2, vec![3.0, 10.0, 4.5, 11.0, 6.0, 12.0])],
            },
            {
                inputs = [
                    (@linear(type = ArrayType::new(
                        DataType::F64,
                        Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]),
                    ))),
                    (@known, right),
                ],
                output_cotangents = [cotangent],
                input_cotangents = [Array::matrix(2, 3, vec![-9.0, -11.0, -13.0, 27.5, 34.5, 41.5])],
            },
        ],
    );
}

#[test]
fn test_ragged_dot_inference_modes_and_group_prefixes() {
    let group_sizes = ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Static(3)]));
    let dimensions = RaggedDotDimensionNumbers::matmul();
    assert_eq!(
        dimensions.to_string(),
        "(dot=(lhs_contracting=[1], rhs_contracting=[1], lhs_batching=[], rhs_batching=[]), lhs_ragged=[0], \
         rhs_group=[0])",
    );
    assert_eq!(
        format!("{dimensions:?}"),
        "RaggedDotDimensionNumbers { dot_dimensions: DotDimensionNumbers { lhs_contracting_dimensions: [1], \
         rhs_contracting_dimensions: [1], lhs_batching_dimensions: [], rhs_batching_dimensions: [] }, \
         lhs_ragged_dimensions: [0], rhs_group_dimensions: [0] }",
    );
    assert_eq!(RaggedDotMode::NonContracting.to_string(), "non-contracting");
    assert_eq!(RaggedDotMode::Contracting.to_string(), "contracting");
    assert_eq!(RaggedDotMode::Batch.to_string(), "batch");
    assert_eq!(
        format!("{:?}", [RaggedDotMode::NonContracting, RaggedDotMode::Contracting, RaggedDotMode::Batch]),
        "[NonContracting, Contracting, Batch]",
    );
    check_operation_type_inference!(
        operation = RaggedDotOperation::new(dimensions),
        cases = [
            {
                input_types = [plain_array(&[5, 2]), plain_array(&[3, 2, 4]), group_sizes.clone()],
                output_types = [plain_array(&[5, 4])],
            },
            {
                input_types = [
                    plain_array(&[5, 2]),
                    plain_array(&[3, 2, 4]),
                    ArrayType::new_static(DataType::F32, [3]),
                ],
                error = "`ragged_dot_general` group sizes must have an integer data type",
            },
            {
                input_types = [
                    plain_array(&[5, 2]),
                    plain_array(&[3, 2, 4]),
                    ArrayType::scalar(DataType::I32),
                ],
                error = "`ragged_dot_general` group sizes must have rank at least one",
            },
            {
                input_types = [plain_array(&[5, 2]), plain_array(&[2, 2, 4]), group_sizes.clone()],
                error = "`ragged_dot_general` RHS group dimension has extent 2 but group sizes describe 3",
            },
            {
                input_types = [
                    ArrayType::new_static(DataType::F8E8M0FNU, [5, 2]),
                    ArrayType::new_static(DataType::F8E8M0FNU, [3, 2, 4]),
                    group_sizes.clone(),
                ],
                error = "`ragged_dot_general` does not support element data type `f8e8m0fnu` in grouped expansion \
                         modes because it cannot represent zero",
            },
        ],
    );

    check_operation_type_inference!(
        operation = RaggedDotOperation::new(RaggedDotDimensionNumbers::new(
            DotDimensionNumbers::new(vec![1], vec![1], Vec::new(), Vec::new()),
            vec![0],
            Vec::new(),
        )),
        cases = [{
            input_types = [plain_array(&[5, 2]), plain_array(&[3, 2, 4]), group_sizes.clone()],
            error = "`ragged_dot_general` requires exactly one RHS group dimension when the LHS ragged dimension is \
                     non-contracting",
        }],
    );

    let contracting = RaggedDotOperation::new(RaggedDotDimensionNumbers::new(
        DotDimensionNumbers::new(vec![1], vec![0], Vec::new(), Vec::new()),
        vec![1],
        Vec::new(),
    ));
    check_operation_type_inference!(
        operation = contracting,
        cases = [
            {
                input_types = [plain_array(&[2, 5]), plain_array(&[5, 4]), group_sizes.clone()],
                output_types = [plain_array(&[3, 2, 4])],
            },
            {
                input_types = [plain_array(&[2, 5]), plain_array(&[5, 4]), ArrayType::new_static(DataType::I64, [3])],
                output_types = [plain_array(&[3, 2, 4])],
            },
            {
                input_types = [
                    ArrayType::new_static(DataType::F8E8M0FNU, [2, 5]),
                    ArrayType::new_static(DataType::F8E8M0FNU, [5, 4]),
                    group_sizes.clone(),
                ],
                error = "`ragged_dot_general` does not support element data type `f8e8m0fnu` in grouped expansion \
                         modes because it cannot represent zero",
            },
        ],
    );
    check_operation_type_inference!(
        operation = RaggedDotOperation::new(RaggedDotDimensionNumbers::new(
            DotDimensionNumbers::new(vec![1], vec![0], Vec::new(), Vec::new()),
            vec![1],
            vec![1],
        )),
        cases = [{
            input_types = [plain_array(&[2, 5]), plain_array(&[5, 4]), group_sizes.clone()],
            error = "`ragged_dot_general` requires zero RHS group dimensions when the LHS ragged dimension is \
                     contracting or batching",
        }],
    );

    let batch = RaggedDotOperation::new(RaggedDotDimensionNumbers::new(
        DotDimensionNumbers::new(vec![1], vec![1], vec![0], vec![0]),
        vec![0],
        Vec::new(),
    ));
    check_operation_type_inference!(
        operation = batch,
        cases = [
            {
                input_types = [plain_array(&[5, 2]), plain_array(&[5, 2, 4]), group_sizes.clone()],
                output_types = [plain_array(&[5, 4])],
            },
            {
                input_types = [
                    ArrayType::new_static(DataType::F8E8M0FNU, [5, 2]),
                    ArrayType::new_static(DataType::F8E8M0FNU, [5, 2, 4]),
                    group_sizes.clone(),
                ],
                output_types = [ArrayType::new_static(DataType::F8E8M0FNU, [5, 4])],
            },
        ],
    );

    let prefixed = RaggedDotOperation::new(RaggedDotDimensionNumbers::new(
        DotDimensionNumbers::new(vec![2], vec![2], vec![0], vec![1]),
        vec![1],
        vec![0],
    ));
    check_operation_type_inference!(
        operation = prefixed,
        cases = [
            {
                input_types = [
                    plain_array(&[2, 5, 3]),
                    plain_array(&[4, 2, 3, 6]),
                    ArrayType::new(
                        DataType::I32,
                        Shape::new(vec![Dimension::Static(2), Dimension::Static(4)]),
                    ),
                ],
                output_types = [plain_array(&[2, 5, 6])],
            },
            {
                input_types = [
                    plain_array(&[2, 5, 3]),
                    plain_array(&[4, 2, 3, 6]),
                    ArrayType::new(
                        DataType::I32,
                        Shape::new(vec![Dimension::Static(3), Dimension::Static(4)]),
                    ),
                ],
                error = "`ragged_dot_general` group sizes prefix must be `[2]`, but got `[3]`",
            },
        ],
    );

    let prefixed_contracting = RaggedDotOperation::new(RaggedDotDimensionNumbers::new(
        DotDimensionNumbers::new(vec![2], vec![1], vec![0], vec![0]),
        vec![2],
        Vec::new(),
    ));
    check_operation_type_inference!(
        operation = prefixed_contracting,
        cases = [{
            input_types = [
                plain_array(&[2, 3, 4]),
                plain_array(&[2, 4, 5]),
                ArrayType::new_static(DataType::I32, [2, 3]),
            ],
            output_types = [plain_array(&[3, 2, 3, 5])],
        }],
    );

    let prefixed_batch = RaggedDotOperation::new(RaggedDotDimensionNumbers::new(
        DotDimensionNumbers::new(vec![2], vec![2], vec![0, 1], vec![0, 1]),
        vec![1],
        Vec::new(),
    ));
    check_operation_type_inference!(
        operation = prefixed_batch,
        cases = [{
            input_types = [
                plain_array(&[2, 4, 3]),
                plain_array(&[2, 4, 3, 5]),
                ArrayType::new_static(DataType::I32, [2, 3]),
            ],
            output_types = [plain_array(&[2, 4, 5])],
        }],
    );
}

#[test]
fn test_ragged_dot_eager_zero_groups_and_uncovered_rows() {
    let lhs = Array::matrix(5, 2, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    let rhs = Array::from_f64s(plain_array(&[3, 2, 1]), vec![10.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    let output = lhs.ragged_dot(&rhs, &Array::vector(vec![2_i32, 0, 2])).unwrap();
    assert_eq!(output.r#type().into_owned(), plain_array(&[5, 1]));
    assert_eq!(output.to_f64s(), vec![12.0, 34.0, 50.0, 68.0, 0.0]);

    // Over-cover clips the intersecting group and leaves every later raw cumulative interval empty.
    let lhs = Array::matrix(4, 2, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let output = lhs.ragged_dot(&rhs, &Array::vector(vec![3_i32, 3, 2])).unwrap();
    assert_eq!(output, Array::matrix(4, 1, vec![12.0_f32, 34.0, 56.0, 38.0]));

    // Sizes in intervals that begin after the physical extent are unobservable, even when their cumulative sum would
    // overflow the host index type.
    let lhs = Array::matrix(1, 1, vec![2.0_f32]);
    let rhs = Array::from_f64s(plain_array(&[2, 1, 1]), vec![3.0, 100.0]);
    let output = lhs.ragged_dot(&rhs, &Array::vector(vec![1_u64, u64::MAX])).unwrap();
    assert_eq!(output, Array::matrix(1, 1, vec![6.0_f32]));

    let lhs = Array::from_f64s(ArrayType::new_static(DataType::F8E8M0FNU, [2, 1]), vec![1.0, 2.0]);
    let rhs = Array::from_f64s(ArrayType::new_static(DataType::F8E8M0FNU, [1, 1, 1]), vec![1.0]);
    assert!(matches!(
        lhs.ragged_dot(&rhs, &Array::vector(vec![1_i32])),
        Err(ProgramError::Type(TypeError::Invalid { message }))
            if message == "`ragged_dot_general` does not support element data type `f8e8m0fnu` in grouped expansion \
                           modes because it cannot represent zero",
    ));
}

#[test]
fn test_ragged_dot_eager_contracting_batch_and_group_prefixes() {
    let contracting_dimensions = RaggedDotDimensionNumbers::new(
        DotDimensionNumbers::new(vec![1], vec![0], Vec::new(), Vec::new()),
        vec![1],
        Vec::new(),
    );
    let lhs = Array::matrix(2, 4, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let rhs = Array::matrix(4, 1, vec![10.0_f32, 20.0, 30.0, 40.0]);
    let output = lhs.ragged_dot_general(&rhs, &Array::vector(vec![1_i32, 0, 2]), &contracting_dimensions).unwrap();
    assert_eq!(output.r#type().into_owned(), plain_array(&[3, 2, 1]));
    assert_eq!(output.to_f64s(), vec![10.0, 50.0, 0.0, 0.0, 130.0, 330.0]);

    // Contracting groups use the same clipped raw intervals, with empty trailing groups retaining zero output slices.
    let output = lhs.ragged_dot_general(&rhs, &Array::vector(vec![3_i32, 3, 2]), &contracting_dimensions).unwrap();
    assert_eq!(
        output,
        Array::from_elements(plain_array(&[3, 2, 1]), &[140.0_f32, 380.0, 160.0, 320.0, 0.0, 0.0]).unwrap(),
    );

    let prefixed_contracting_dimensions = RaggedDotDimensionNumbers::new(
        DotDimensionNumbers::new(vec![0, 1], vec![0, 1], Vec::new(), Vec::new()),
        vec![1],
        Vec::new(),
    );
    let lhs = Array::from_f64s(plain_array(&[2, 3, 1]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let rhs = Array::from_f64s(plain_array(&[2, 3, 1]), vec![10.0, 20.0, 30.0, 1.0, 2.0, 3.0]);
    let group_sizes = Array::matrix(2, 2, vec![1_i32, 2, 2, 1]);
    let output = lhs.ragged_dot_general(&rhs, &group_sizes, &prefixed_contracting_dimensions).unwrap();
    assert_eq!(output.r#type().into_owned(), plain_array(&[2, 1, 1]));
    assert_eq!(output.to_f64s(), vec![24.0, 148.0]);

    let group_sizes = Array::matrix(2, 2, vec![2_i32, 3, 4, 1]);
    let output = lhs.ragged_dot_general(&rhs, &group_sizes, &prefixed_contracting_dimensions).unwrap();
    assert_eq!(output, Array::from_elements(plain_array(&[2, 1, 1]), &[82.0_f32, 90.0]).unwrap());

    let batch_dimensions = RaggedDotDimensionNumbers::new(
        DotDimensionNumbers::new(vec![1], vec![1], vec![0], vec![0]),
        vec![0],
        Vec::new(),
    );
    let lhs = Array::matrix(4, 1, vec![1.0_f32, 2.0, 3.0, 4.0]);
    let rhs = Array::from_f64s(plain_array(&[4, 1, 1]), vec![10.0, 20.0, 30.0, 40.0]);
    // Batch mode is the ordinary batched dot; group-size values do not participate in its runtime semantics.
    let output = lhs.ragged_dot_general(&rhs, &Array::vector(vec![5_i32, -1, 99]), &batch_dimensions).unwrap();
    assert_eq!(output.r#type().into_owned(), plain_array(&[4, 1]));
    assert_eq!(output.to_f64s(), vec![10.0, 40.0, 90.0, 160.0]);

    let lhs = Array::from_f64s(ArrayType::new_static(DataType::F8E8M0FNU, [2, 1]), vec![1.0, 2.0]);
    let rhs = Array::from_f64s(ArrayType::new_static(DataType::F8E8M0FNU, [2, 1, 1]), vec![2.0, 4.0]);
    let output = lhs.ragged_dot_general(&rhs, &Array::vector(vec![-1_i32]), &batch_dimensions).unwrap();
    assert_eq!(output.to_f64s(), vec![2.0, 8.0]);

    let prefixed_dimensions =
        RaggedDotDimensionNumbers::new(DotDimensionNumbers::new(vec![2], vec![2], vec![0], vec![1]), vec![1], vec![0]);
    let lhs = Array::from_f64s(plain_array(&[2, 3, 1]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let rhs = Array::from_f64s(plain_array(&[2, 2, 1, 1]), vec![10.0, 100.0, 20.0, 200.0]);
    let group_sizes = Array::matrix(2, 2, vec![1_i32, 1, 2, 0]);
    let output = lhs.ragged_dot_general(&rhs, &group_sizes, &prefixed_dimensions).unwrap();
    assert_eq!(output.r#type().into_owned(), plain_array(&[2, 3, 1]));
    assert_eq!(output.to_f64s(), vec![10.0, 40.0, 0.0, 400.0, 500.0, 0.0]);

    let prefixed_batch_dimensions = RaggedDotDimensionNumbers::new(
        DotDimensionNumbers::new(vec![2], vec![2], vec![0, 1], vec![0, 1]),
        vec![1],
        Vec::new(),
    );
    let lhs = Array::from_f64s(plain_array(&[2, 4, 1]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let rhs = Array::from_f64s(plain_array(&[2, 4, 1, 1]), vec![10.0, 20.0, 30.0, 40.0, 1.0, 2.0, 3.0, 4.0]);
    let group_sizes = Array::matrix(2, 3, vec![1_i32, 0, 2, 0, 2, 2]);
    let output = lhs.ragged_dot_general(&rhs, &group_sizes, &prefixed_batch_dimensions).unwrap();
    assert_eq!(output.r#type().into_owned(), plain_array(&[2, 4, 1]));
    assert_eq!(output.to_f64s(), vec![10.0, 40.0, 90.0, 160.0, 5.0, 12.0, 21.0, 32.0]);
}

#[test]
fn test_ragged_dot_batching_leading_axis_and_ragged_axis_rejection() {
    let operation = RaggedDotOperation::new(RaggedDotDimensionNumbers::matmul());
    let lhs = ArrayBatch::new(
        Array::from_f64s(plain_array(&[2, 2, 2]), vec![1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 4.0, 3.0]),
        BatchAxis::new(0),
    )
    .unwrap();
    let rhs = ArrayBatch::new(
        Array::from_f64s(plain_array(&[2, 2, 2, 1]), vec![10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
        BatchAxis::new(0),
    )
    .unwrap();
    let group_sizes = ArrayBatch::new(Array::matrix(2, 2, vec![1_i32, 1, 1, 1]), BatchAxis::new(0)).unwrap();
    let context = BatchingContext::new(EagerContext::<Array>::new(), 2);
    let outputs = operation
        .batch(&context, &crate::EmptyRegionDriver, &[lhs.clone(), rhs.clone(), group_sizes.clone()])
        .unwrap()
        .into_parts()
        .0;
    assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
    assert_eq!(outputs[0].value().to_f64s(), vec![12.0, 18.0, 13.0, 45.0]);

    let contracting_operation = RaggedDotOperation::new(RaggedDotDimensionNumbers::new(
        DotDimensionNumbers::new(vec![1], vec![0], Vec::new(), Vec::new()),
        vec![1],
        Vec::new(),
    ));
    let contracting_lhs = ArrayBatch::new(
        Array::from_f64s(
            plain_array(&[2, 2, 4]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 2.0, 1.0, 0.0, 1.0, 1.0, 2.0, 1.0, 0.0],
        ),
        BatchAxis::new(0),
    )
    .unwrap();
    let contracting_rhs = ArrayBatch::new(
        Array::from_f64s(plain_array(&[2, 4, 1]), vec![10.0, 20.0, 30.0, 40.0, 1.0, 2.0, 3.0, 4.0]),
        BatchAxis::new(0),
    )
    .unwrap();
    let contracting_groups = ArrayBatch::new(Array::matrix(2, 2, vec![1_i32, 2, 2, 2]), BatchAxis::new(0)).unwrap();
    let outputs = contracting_operation
        .batch(&context, &crate::EmptyRegionDriver, &[contracting_lhs, contracting_rhs, contracting_groups])
        .unwrap()
        .into_parts()
        .0;
    assert_eq!(outputs[0].batch_axis(), BatchAxis::new(1));
    assert_eq!(outputs[0].value().to_f64s(), vec![10.0, 50.0, 4.0, 5.0, 130.0, 330.0, 4.0, 3.0]);

    let variable = DimensionVariable::new("length", DimensionBounds::new(0, Some(2)).unwrap());
    let ragged_lhs = lhs
        .with_ragged_axes(vec![RaggedAxis::new(1, Array::vector(vec![1_i32, 2]), variable, vec![0])])
        .unwrap();
    assert_eq!(
        operation.batch(&context, &crate::EmptyRegionDriver, &[ragged_lhs, rhs, group_sizes]).unwrap_err(),
        crate::batching::BatchingError::UnsupportedOperation {
            message: "`ragged_dot_general` does not accept bounded ragged dimension `length`; use explicit group \
                      sizes instead"
                .to_string(),
        },
    );
}

#[test]
fn test_ragged_dot_jvp_and_noncontracting_transpose() {
    let lhs = Array::matrix(3, 2, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let rhs = Array::from_f64s(plain_array(&[2, 2, 1]), vec![10.0, 1.0, 2.0, 3.0]);
    let group_sizes = Array::vector(vec![1_i32, 2]);

    let mut builder = crate::ProgramBuilder::<Array, ArrayOperation<Array>>::new();
    let lhs_input = builder.add_input(lhs.r#type().into_owned());
    let rhs_input = builder.add_input(rhs.r#type().into_owned());
    let groups = builder.add_constant(group_sizes.clone());
    let output = builder
        .add_instruction(
            RaggedDotOperation::new(RaggedDotDimensionNumbers::matmul()),
            Vec::new(),
            vec![lhs_input, rhs_input, groups],
            None,
        )
        .unwrap()[0];
    let program = builder
        .build::<Vec<Array>, Vec<Array>>(vec![output], vec![crate::Placeholder; 2], vec![crate::Placeholder])
        .unwrap();
    let lhs_tangent = Array::matrix(3, 2, vec![1.0_f32; 6]);
    let rhs_tangent = Array::from_f64s(plain_array(&[2, 2, 1]), vec![1.0; 4]);
    let outputs = program
        .clone()
        .jvp()
        .unwrap()
        .interpret(vec![lhs.clone(), rhs.clone(), lhs_tangent.clone(), rhs_tangent.clone()])
        .unwrap();
    assert_eq!(outputs[0].to_f64s(), vec![12.0, 18.0, 28.0]);
    assert_eq!(outputs[1].to_f64s(), vec![14.0, 12.0, 16.0]);
    let step = 1e-3;
    let plus = (lhs.clone() + lhs_tangent.clone() * step)
        .ragged_dot(&(rhs.clone() + rhs_tangent.clone() * step), &group_sizes)
        .unwrap();
    let minus = (lhs.clone() - lhs_tangent * step)
        .ragged_dot(&(rhs.clone() - rhs_tangent * step), &group_sizes)
        .unwrap();
    assert_abs_diff_eq!(outputs[1], (plus - minus) * (0.5 / step), epsilon = 1e-3);

    let transpose = program.transpose_with_respect_to(&[0]).unwrap();
    assert_eq!(
        transpose.to_string(),
        indoc! {"
            lambda %0:f32[3, 1], %1:f32[2, 2, 1] .
            let %2:i32[2] = const [1, 2]
                %3:f32[3, 2] = ragged_dot_general [
                    dimensions=(dot=(lhs_contracting=[1], rhs_contracting=[2], lhs_batching=[], \
                    rhs_batching=[]), lhs_ragged=[0], rhs_group=[0]),
                ] %0 %1 %2
            in (%3)
        "}
        .trim_end(),
    );

    check_operation_transposition!(
        @exact,
        operation = RaggedDotOperation::new(RaggedDotDimensionNumbers::matmul()),
        cases = [
            {
                inputs = [
                    (@linear(type = lhs.r#type().into_owned())),
                    (@known, rhs.clone()),
                    (@known, group_sizes.clone()),
                ],
                output_cotangents = [Array::matrix(3, 1, vec![2.0_f32, 3.0, 5.0])],
                input_cotangents = [Array::matrix(3, 2, vec![20.0_f32, 2.0, 6.0, 9.0, 10.0, 15.0])],
            },
            {
                inputs = [
                    (@known, lhs.clone()),
                    (@linear(type = rhs.r#type().into_owned())),
                    (@known, group_sizes),
                ],
                output_cotangents = [Array::matrix(3, 1, vec![2.0_f32, 3.0, 5.0])],
                input_cotangents = [Array::from_f64s(plain_array(&[2, 2, 1]), vec![2.0, 4.0, 34.0, 42.0])],
            },
        ],
    );
}

#[test]
fn test_ragged_dot_batch_widened_differential_staging() {
    let lhs_type = ArrayType::new_static(DataType::F8E8M0FNU, [3, 2]);
    let rhs_type = ArrayType::new_static(DataType::F8E8M0FNU, [3, 2, 1]);
    let dimensions = RaggedDotDimensionNumbers::new(
        DotDimensionNumbers::new(vec![1], vec![1], vec![0], vec![0]),
        vec![0],
        Vec::new(),
    );
    let mut builder = crate::ProgramBuilder::<Array, ArrayOperation<Array>>::new();
    let lhs = builder.add_input(lhs_type.clone());
    let rhs = builder.add_input(rhs_type.clone());
    let group_sizes = builder.add_constant(Array::vector(vec![1_i32, 2]));
    let output = builder
        .add_instruction(RaggedDotOperation::new(dimensions), Vec::new(), vec![lhs, rhs, group_sizes], None)
        .unwrap()[0];
    let program = builder
        .build::<Vec<Array>, Vec<Array>>(vec![output], vec![crate::Placeholder; 2], vec![crate::Placeholder])
        .unwrap();

    let jvp = program.clone().jvp().unwrap();
    assert_eq!(
        jvp.input_types(),
        vec![
            lhs_type.clone(),
            rhs_type.clone(),
            ArrayType::new_static(DataType::F32, [3, 2]),
            ArrayType::new_static(DataType::F32, [3, 2, 1]),
        ],
    );
    assert_eq!(
        jvp.output_types(),
        vec![ArrayType::new_static(DataType::F8E8M0FNU, [3, 1]), ArrayType::new_static(DataType::F32, [3, 1]),],
    );
}

#[test]
fn test_ragged_dot_transpose_rejects_contracting_and_batch_modes() {
    let cases = [
        (
            RaggedDotOperation::new(RaggedDotDimensionNumbers::new(
                DotDimensionNumbers::new(vec![1], vec![0], Vec::new(), Vec::new()),
                vec![1],
                Vec::new(),
            )),
            [plain_array(&[2, 5]), plain_array(&[5, 4]), ArrayType::new_static(DataType::I32, [3])],
            RaggedDotMode::Contracting,
        ),
        (
            RaggedDotOperation::new(RaggedDotDimensionNumbers::new(
                DotDimensionNumbers::new(vec![1], vec![1], vec![0], vec![0]),
                vec![0],
                Vec::new(),
            )),
            [plain_array(&[5, 2]), plain_array(&[5, 2, 4]), ArrayType::new_static(DataType::I32, [3])],
            RaggedDotMode::Batch,
        ),
    ];
    for (operation, input_types, mode) in cases {
        let mut builder = crate::ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let inputs = input_types.map(|input_type| builder.add_input(input_type));
        let outputs = builder.add_instruction(operation, Vec::new(), inputs.to_vec(), None).unwrap().to_vec();
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(outputs, vec![crate::Placeholder; 3], vec![crate::Placeholder])
            .unwrap();
        assert_eq!(
            program.transpose_with_respect_to(&[0]).unwrap_err(),
            crate::differentiation::DifferentiationError::Program(crate::ProgramError::UnsupportedOperation {
                message: format!("`{RAGGED_DOT_OPERATION_NAME}` transposition is unsupported in `{mode}` mode"),
            }),
        );
    }
}
