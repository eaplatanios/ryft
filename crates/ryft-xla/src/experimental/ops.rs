//! Backend-owned staged op universe for traced XLA programs.

use std::{
    fmt::{Debug, Display},
    sync::Arc,
};

use ryft_core::{
    batching::{BatchingError, FlatTracedVMap, LinearVMapOperation, VMapOperation, VMapTracingOperation},
    tracing_v2::{
        AtomId, CustomPrimitive, DifferentiableOperation, DifferentiationError, InterpretableOperation,
        LinearPrimitiveOperation, LinearTerm, Operation, TracingError,
        engine::Engine,
        forward::JvpTracer,
        linear::{Linearized, linearize_program, transpose_linear_program_with_output_examples},
        operations::{
            AddOperation, AddTracingOperation, CosOperation, CosTracingOperation, CustomTracingOperation,
            FlatTracedRematerialize, LeftMatMulOperation, LeftMatMulTracingOperation, LinearRematerializeOperation,
            MatMulOperation, MatMulTracingOperation, MatrixTransposeOperation, MatrixTransposeTracingOperation,
            MulOperation, MulTracingOperation, NegOperation, NegTracingOperation, RematerializeOperation,
            RematerializeTracingOperation, ReshapeOperation, ReshapeTracingOperation, RightMatMulOperation,
            RightMatMulTracingOperation, ScaleOperation, ScaleTracingOperation, SinOperation, SinTracingOperation,
            left_matmul::left_matmul_abstract_eval, right_matmul::right_matmul_abstract_eval,
        },
    },
    types::{ArrayType, TypeError, Typed},
};

use crate::experimental::{
    operations::{ShardMapOperation, WithShardingConstraintOperation},
    shard_map::{ShardMapTensor, ShardMapTracer},
};

type XlaLinearOperation = LinearPrimitiveOperation<ArrayType, ShardMapTensor>;

fn make_linear_xla_vmap(
    engine: &dyn Engine<
        Type = ArrayType,
        Value = ShardMapTensor,
        TracingOperation = XlaPrimitiveOperation,
        LinearOperation = XlaLinearOperation,
    >,
    body: &FlatTracedVMap<ArrayType, ShardMapTensor, XlaPrimitiveOperation>,
    input_primals: Vec<ShardMapTensor>,
) -> Result<LinearVMapOperation<ArrayType, ShardMapTensor>, TracingError> {
    let body_program = body.program();
    let output_primals = body_program.interpret(input_primals.clone())?;
    let pushforward = linearize_program(engine, &body_program, input_primals)?;
    let pullback = transpose_linear_program_with_output_examples(engine, &pushforward, output_primals.as_slice())?;
    Ok(LinearVMapOperation::new(
        FlatTracedVMap::from_parts(
            body.lane_count(),
            body.input_types().to_vec(),
            body.output_types().to_vec(),
            pushforward,
        ),
        FlatTracedVMap::from_parts(
            body.lane_count(),
            body.output_types().to_vec(),
            body.input_types().to_vec(),
            pullback,
        ),
    ))
}

fn make_linear_xla_rematerialize(
    engine: &dyn Engine<
        Type = ArrayType,
        Value = ShardMapTensor,
        TracingOperation = XlaPrimitiveOperation,
        LinearOperation = XlaLinearOperation,
    >,
    body: &FlatTracedRematerialize<ArrayType, ShardMapTensor, XlaPrimitiveOperation>,
    input_primals: Vec<ShardMapTensor>,
) -> Result<LinearRematerializeOperation<ArrayType, ShardMapTensor>, TracingError> {
    let body_program = body.program();
    let output_primals = body_program.interpret(input_primals.clone())?;
    let pushforward = linearize_program(engine, &body_program, input_primals)?;
    let pullback = transpose_linear_program_with_output_examples(engine, &pushforward, output_primals.as_slice())?;
    Ok(LinearRematerializeOperation::new(
        FlatTracedRematerialize::from_parts(body.input_types().to_vec(), body.output_types().to_vec(), pushforward),
        FlatTracedRematerialize::from_parts(body.output_types().to_vec(), body.input_types().to_vec(), pullback),
    ))
}

/// Closed ordinary staged-op universe owned by the XLA backend.
#[allow(private_interfaces)]
#[derive(Clone)]
pub enum XlaPrimitiveOperation {
    /// Elementwise addition.
    Add,

    /// Elementwise multiplication.
    Mul,

    /// Elementwise negation.
    Neg,

    /// Elementwise sine.
    Sin,

    /// Elementwise cosine.
    Cos,

    /// Matrix multiplication.
    MatMul,

    /// Matrix transpose.
    MatrixTranspose,

    /// Scaling by one captured factor.
    Scale { factor: ShardMapTensor },

    /// Left matrix multiplication by one captured factor.
    LeftMatMul { factor: ShardMapTensor },

    /// Right matrix multiplication by one captured factor.
    RightMatMul { factor: ShardMapTensor },

    /// Reshape.
    Reshape { input_type: ArrayType, output_type: ArrayType },

    /// Higher-order `vmap`.
    VMap(Box<VMapOperation<ArrayType, ShardMapTensor, XlaPrimitiveOperation, XlaLinearOperation>>),

    /// Higher-order rematerialization.
    Rematerialize(Box<RematerializeOperation<ArrayType, ShardMapTensor, XlaPrimitiveOperation, XlaLinearOperation>>),

    /// XLA-specific `shard_map`.
    ShardMap(Box<ShardMapOperation<ShardMapTensor>>),

    /// XLA-specific sharding constraint.
    WithShardingConstraint(WithShardingConstraintOperation),

    /// Explicit escape hatch for custom XLA ops.
    Custom(Arc<CustomPrimitive<ArrayType, ShardMapTensor>>),
}

impl Debug for XlaPrimitiveOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Add => write!(formatter, "Add"),
            Self::Mul => write!(formatter, "Mul"),
            Self::Neg => write!(formatter, "Neg"),
            Self::Sin => write!(formatter, "Sin"),
            Self::Cos => write!(formatter, "Cos"),
            Self::MatMul => write!(formatter, "MatMul"),
            Self::MatrixTranspose => write!(formatter, "MatrixTranspose"),
            Self::Scale { .. } => write!(formatter, "Scale"),
            Self::LeftMatMul { .. } => write!(formatter, "LeftMatMul"),
            Self::RightMatMul { .. } => write!(formatter, "RightMatMul"),
            Self::Reshape { input_type, output_type } => write!(formatter, "Reshape({input_type} -> {output_type})"),
            Self::VMap(vmap) => Debug::fmt(vmap, formatter),
            Self::Rematerialize(remat) => Debug::fmt(remat, formatter),
            Self::ShardMap(op) => Debug::fmt(op, formatter),
            Self::WithShardingConstraint(op) => Debug::fmt(op, formatter),
            Self::Custom(op) => Debug::fmt(op.as_ref(), formatter),
        }
    }
}

impl Display for XlaPrimitiveOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Reshape { output_type, .. } => write!(formatter, "reshape{}", output_type.shape),
            _ => write!(formatter, "{}", self.name()),
        }
    }
}

impl Operation for XlaPrimitiveOperation {
    fn name(&self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Mul => "mul",
            Self::Neg => "neg",
            Self::Sin => "sin",
            Self::Cos => "cos",
            Self::MatMul => "matmul",
            Self::MatrixTranspose => "matrix_transpose",
            Self::Scale { .. } => "scale",
            Self::LeftMatMul { .. } => "left_matmul",
            Self::RightMatMul { .. } => "right_matmul",
            Self::Reshape { .. } => "reshape",
            Self::VMap(vmap) => vmap.name(),
            Self::Rematerialize(remat) => remat.name(),
            Self::ShardMap(op) => op.name(),
            Self::WithShardingConstraint(op) => op.name(),
            Self::Custom(op) => op.name(),
        }
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        match self {
            Self::Add => AddOperation.infer_output_types(input_types),
            Self::Mul => MulOperation.infer_output_types(input_types),
            Self::Neg => NegOperation.infer_output_types(input_types),
            Self::Sin => SinOperation.infer_output_types(input_types),
            Self::Cos => CosOperation.infer_output_types(input_types),
            Self::MatMul => MatMulOperation.infer_output_types(input_types),
            Self::MatrixTranspose => MatrixTransposeOperation.infer_output_types(input_types),
            Self::Scale { .. } => ScaleOperation::<ArrayType, ShardMapTensor>::abstract_eval_static(input_types),
            Self::LeftMatMul { factor } => left_matmul_abstract_eval(&Typed::r#type(factor), input_types),
            Self::RightMatMul { factor } => right_matmul_abstract_eval(&Typed::r#type(factor), input_types),
            Self::Reshape { input_type, output_type } => {
                ReshapeOperation::new(input_type.clone(), output_type.clone()).infer_output_types(input_types)
            }
            Self::VMap(vmap) => vmap.infer_output_types(input_types),
            Self::Rematerialize(remat) => remat.infer_output_types(input_types),
            Self::ShardMap(op) => op.infer_output_types(input_types),
            Self::WithShardingConstraint(op) => op.infer_output_types(input_types),
            Self::Custom(op) => op.infer_output_types(input_types),
        }
    }

    fn try_simplify(
        &self,
        inputs: &[AtomId],
        is_zero_constant: &dyn Fn(AtomId) -> bool,
        is_one_constant: &dyn Fn(AtomId) -> bool,
    ) -> Option<Vec<AtomId>> {
        match self {
            Self::Add => AddOperation.try_simplify(inputs, is_zero_constant, is_one_constant),
            Self::Mul => MulOperation.try_simplify(inputs, is_zero_constant, is_one_constant),
            Self::Neg => NegOperation.try_simplify(inputs, is_zero_constant, is_one_constant),
            Self::Scale { factor } => ScaleOperation::<ArrayType, ShardMapTensor>::new(factor.clone()).try_simplify(
                inputs,
                is_zero_constant,
                is_one_constant,
            ),
            Self::Custom(op) => op.try_simplify(inputs, is_zero_constant, is_one_constant),
            _ => None,
        }
    }
}

impl InterpretableOperation<ArrayType, ShardMapTensor> for XlaPrimitiveOperation {
    fn interpret(&self, inputs: &[ShardMapTensor]) -> Result<Vec<ShardMapTensor>, TracingError> {
        match self {
            Self::Add => AddOperation.interpret(inputs),
            Self::Mul => MulOperation.interpret(inputs),
            Self::Neg => NegOperation.interpret(inputs),
            Self::Sin => SinOperation.interpret(inputs),
            Self::Cos => CosOperation.interpret(inputs),
            Self::MatMul => MatMulOperation.interpret(inputs),
            Self::MatrixTranspose => MatrixTransposeOperation.interpret(inputs),
            Self::Scale { factor } => ScaleOperation::new(factor.clone()).interpret(inputs),
            Self::LeftMatMul { factor } => LeftMatMulOperation::new(factor.clone()).interpret(inputs),
            Self::RightMatMul { factor } => RightMatMulOperation::new(factor.clone()).interpret(inputs),
            Self::Reshape { input_type, output_type } => {
                ReshapeOperation::new(input_type.clone(), output_type.clone()).interpret(inputs)
            }
            Self::VMap(vmap) => vmap.interpret(inputs),
            Self::Rematerialize(remat) => remat.interpret(inputs),
            Self::ShardMap(op) => op.interpret(inputs),
            Self::WithShardingConstraint(op) => op.interpret(inputs),
            Self::Custom(op) => op.interpret(inputs),
        }
    }
}

impl
    DifferentiableOperation<
        ArrayType,
        ShardMapTensor,
        LinearTerm<ArrayType, ShardMapTensor, XlaLinearOperation>,
        XlaPrimitiveOperation,
        XlaLinearOperation,
    > for XlaPrimitiveOperation
{
    fn jvp(
        &self,
        engine: &dyn Engine<
            Type = ArrayType,
            Value = ShardMapTensor,
            TracingOperation = XlaPrimitiveOperation,
            LinearOperation = XlaLinearOperation,
        >,
        inputs: &[JvpTracer<ShardMapTensor, LinearTerm<ArrayType, ShardMapTensor, XlaLinearOperation>>],
    ) -> Result<Vec<JvpTracer<ShardMapTensor, LinearTerm<ArrayType, ShardMapTensor, XlaLinearOperation>>>, TracingError>
    {
        match self {
            Self::Add => AddOperation.jvp(engine, inputs),
            Self::Mul => MulOperation.jvp(engine, inputs),
            Self::Neg => NegOperation.jvp(engine, inputs),
            Self::Sin => SinOperation.jvp(engine, inputs),
            Self::Cos => CosOperation.jvp(engine, inputs),
            Self::MatMul => MatMulOperation.jvp(engine, inputs),
            Self::MatrixTranspose => MatrixTransposeOperation.jvp(engine, inputs),
            Self::Scale { factor } => ScaleOperation::new(factor.clone()).jvp(engine, inputs),
            Self::LeftMatMul { factor } => LeftMatMulOperation::new(factor.clone()).jvp(engine, inputs),
            Self::RightMatMul { factor } => RightMatMulOperation::new(factor.clone()).jvp(engine, inputs),
            Self::Reshape { input_type, output_type } => {
                ReshapeOperation::new(input_type.clone(), output_type.clone()).jvp(engine, inputs)
            }
            Self::VMap(vmap) => {
                let primal_inputs = inputs.iter().map(|input| input.primal.clone()).collect::<Vec<_>>();
                let tangent_inputs = inputs.iter().map(|input| input.tangent.clone()).collect::<Vec<_>>();
                let primal_outputs = vmap.interpret(primal_inputs.as_slice())?;
                let tangent_builder = if let Some(first_tangent) = tangent_inputs.first() {
                    first_tangent.builder.clone()
                } else if vmap.body().total_output_count() == 0 {
                    return Ok(Vec::new());
                } else {
                    return Err(BatchingError::VMapMissingTangentStagingContext.into());
                };
                let lane_input_count = vmap.body().input_types().len();
                let lane_primals = primal_inputs.iter().take(lane_input_count).cloned().collect::<Vec<_>>();
                let tangent_outputs = LinearTerm::apply_staged_op(
                    tangent_builder,
                    tangent_inputs.as_slice(),
                    LinearPrimitiveOperation::VMap(Box::new(make_linear_xla_vmap(engine, vmap.body(), lane_primals)?)),
                    vmap.body().total_output_count(),
                )?;
                Ok(primal_outputs
                    .into_iter()
                    .zip(tangent_outputs)
                    .map(|(primal, tangent)| JvpTracer { primal, tangent })
                    .collect::<Vec<_>>())
            }
            Self::Rematerialize(remat) => {
                let primal_inputs = inputs.iter().map(|input| input.primal.clone()).collect::<Vec<_>>();
                let tangent_inputs = inputs.iter().map(|input| input.tangent.clone()).collect::<Vec<_>>();
                let primal_outputs = remat.interpret(primal_inputs.as_slice())?;
                let tangent_builder = if let Some(first_tangent) = tangent_inputs.first() {
                    first_tangent.builder.clone()
                } else if remat.body().output_types().is_empty() {
                    return Ok(Vec::new());
                } else {
                    return Err(DifferentiationError::MissingLinearRematerializeReplayTangentLeaves.into());
                };
                let tangent_outputs = LinearTerm::apply_staged_op(
                    tangent_builder,
                    tangent_inputs.as_slice(),
                    LinearPrimitiveOperation::Rematerialize(Box::new(make_linear_xla_rematerialize(
                        engine,
                        remat.body(),
                        primal_inputs,
                    )?)),
                    remat.body().output_types().len(),
                )?;
                Ok(primal_outputs
                    .into_iter()
                    .zip(tangent_outputs)
                    .map(|(primal, tangent)| JvpTracer { primal, tangent })
                    .collect::<Vec<_>>())
            }
            Self::ShardMap(op) => op.jvp(engine, inputs),
            Self::WithShardingConstraint(op) => op.jvp(engine, inputs),
            Self::Custom(op) => op.jvp(engine, inputs),
        }
    }
}

impl InterpretableOperation<ArrayType, Linearized<ShardMapTracer>> for XlaPrimitiveOperation {
    fn interpret(
        &self,
        inputs: &[Linearized<ShardMapTracer>],
    ) -> Result<Vec<Linearized<ShardMapTracer>>, TracingError> {
        match self {
            Self::Add => AddOperation.interpret(inputs),
            Self::Mul => MulOperation.interpret(inputs),
            Self::Neg => NegOperation.interpret(inputs),
            Self::Sin => SinOperation.interpret(inputs),
            Self::Cos => CosOperation.interpret(inputs),
            Self::MatMul => MatMulOperation.interpret(inputs),
            Self::MatrixTranspose => MatrixTransposeOperation.interpret(inputs),
            Self::Scale { factor } => ScaleOperation::new(factor.clone()).interpret(inputs),
            Self::LeftMatMul { factor } => LeftMatMulOperation::new(factor.clone()).interpret(inputs),
            Self::RightMatMul { factor } => RightMatMulOperation::new(factor.clone()).interpret(inputs),
            Self::Reshape { input_type, output_type } => {
                ReshapeOperation::new(input_type.clone(), output_type.clone()).interpret(inputs)
            }
            Self::VMap(vmap) => vmap.interpret(inputs),
            Self::Rematerialize(remat) => remat.interpret(inputs),
            Self::ShardMap(op) => op.interpret(inputs),
            Self::WithShardingConstraint(op) => op.interpret(inputs),
            Self::Custom(op) => op.interpret(inputs),
        }
    }
}

impl AddTracingOperation<ArrayType, ShardMapTensor> for XlaPrimitiveOperation {
    fn add_op() -> Self {
        XlaPrimitiveOperation::Add
    }
}

impl MulTracingOperation<ArrayType, ShardMapTensor> for XlaPrimitiveOperation {
    fn mul_op() -> Self {
        XlaPrimitiveOperation::Mul
    }
}

impl NegTracingOperation<ArrayType, ShardMapTensor> for XlaPrimitiveOperation {
    fn neg_op() -> Self {
        XlaPrimitiveOperation::Neg
    }
}

impl SinTracingOperation<ArrayType, ShardMapTensor> for XlaPrimitiveOperation {
    fn sin_op() -> Self {
        XlaPrimitiveOperation::Sin
    }
}

impl CosTracingOperation<ArrayType, ShardMapTensor> for XlaPrimitiveOperation {
    fn cos_op() -> Self {
        XlaPrimitiveOperation::Cos
    }
}

impl MatMulTracingOperation<ArrayType, ShardMapTensor> for XlaPrimitiveOperation {
    fn matmul_op() -> Self {
        XlaPrimitiveOperation::MatMul
    }
}

impl MatrixTransposeTracingOperation<ArrayType, ShardMapTensor> for XlaPrimitiveOperation {
    fn matrix_transpose_op() -> Self {
        XlaPrimitiveOperation::MatrixTranspose
    }
}

impl CustomTracingOperation<ArrayType, ShardMapTensor> for XlaPrimitiveOperation {
    fn custom_op(primitive: Arc<CustomPrimitive<ArrayType, ShardMapTensor>>) -> Self {
        XlaPrimitiveOperation::Custom(primitive)
    }
}

impl VMapTracingOperation<ArrayType, ShardMapTensor, XlaLinearOperation> for XlaPrimitiveOperation {
    fn vmap_op(op: VMapOperation<ArrayType, ShardMapTensor, XlaPrimitiveOperation, XlaLinearOperation>) -> Self {
        XlaPrimitiveOperation::VMap(Box::new(op))
    }
}

impl RematerializeTracingOperation<ArrayType, ShardMapTensor, XlaLinearOperation> for XlaPrimitiveOperation {
    fn rematerialize_op(
        op: RematerializeOperation<ArrayType, ShardMapTensor, XlaPrimitiveOperation, XlaLinearOperation>,
    ) -> Self {
        XlaPrimitiveOperation::Rematerialize(Box::new(op))
    }
}

impl ScaleTracingOperation<ArrayType, ShardMapTensor> for XlaPrimitiveOperation {
    fn scale_op(factor: ShardMapTensor) -> Self {
        XlaPrimitiveOperation::Scale { factor }
    }
}

impl LeftMatMulTracingOperation<ArrayType, ShardMapTensor> for XlaPrimitiveOperation {
    fn left_matmul_op(factor: ShardMapTensor) -> Self {
        XlaPrimitiveOperation::LeftMatMul { factor }
    }
}

impl RightMatMulTracingOperation<ArrayType, ShardMapTensor> for XlaPrimitiveOperation {
    fn right_matmul_op(factor: ShardMapTensor) -> Self {
        XlaPrimitiveOperation::RightMatMul { factor }
    }
}

impl ReshapeTracingOperation<ArrayType, ShardMapTensor> for XlaPrimitiveOperation {
    fn reshape_op(input_type: ArrayType, output_type: ArrayType) -> Self {
        XlaPrimitiveOperation::Reshape { input_type, output_type }
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, fmt::Display, rc::Rc, sync::Arc};

    use pretty_assertions::assert_eq;

    use ryft_core::parameters::Placeholder;
    use ryft_core::tracing_v2::ProgramBuilder;
    use ryft_core::tracing_v2::operations::CustomOperationError;
    use ryft_core::types::{DataType, Typed};

    use super::*;

    #[derive(Clone, Debug)]
    struct TestCustomXlaOp;

    impl Display for TestCustomXlaOp {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(formatter, "test_custom_xla")
        }
    }

    impl Operation for TestCustomXlaOp {
        fn name(&self) -> &'static str {
            "test_custom_xla"
        }

        fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
            Ok(input_types.to_vec())
        }
    }

    impl InterpretableOperation<ArrayType, ShardMapTensor> for TestCustomXlaOp {
        fn interpret(&self, inputs: &[ShardMapTensor]) -> Result<Vec<ShardMapTensor>, TracingError> {
            Ok(inputs.to_vec())
        }
    }

    fn scalar_type() -> ArrayType {
        ArrayType::scalar(DataType::F32)
    }

    fn unary_vmap_body() -> FlatTracedVMap<ArrayType, ShardMapTensor, XlaPrimitiveOperation> {
        let mut builder = ProgramBuilder::<ArrayType, ShardMapTensor, XlaPrimitiveOperation>::new();
        let input = builder.add_input(scalar_type());
        let output = builder
            .add_instruction(XlaPrimitiveOperation::Sin, vec![input])
            .expect("vmap body should stage one sine op")
            .into_iter()
            .next()
            .expect("sine should produce one output");
        let program = builder.build::<Vec<ShardMapTensor>, Vec<ShardMapTensor>>(
            vec![output],
            vec![Placeholder],
            vec![Placeholder],
        );
        FlatTracedVMap::from_parts(2, vec![scalar_type()], vec![scalar_type()], program)
    }

    fn unary_rematerialize_body() -> FlatTracedRematerialize<ArrayType, ShardMapTensor, XlaPrimitiveOperation> {
        let mut builder = ProgramBuilder::<ArrayType, ShardMapTensor, XlaPrimitiveOperation>::new();
        let input = builder.add_input(scalar_type());
        let output = builder
            .add_instruction(XlaPrimitiveOperation::Sin, vec![input])
            .expect("rematerialize body should stage one sine op")
            .into_iter()
            .next()
            .expect("sine should produce one output");
        let program = builder.build::<Vec<ShardMapTensor>, Vec<ShardMapTensor>>(
            vec![output],
            vec![Placeholder],
            vec![Placeholder],
        );
        FlatTracedRematerialize::from_parts(vec![scalar_type()], vec![scalar_type()], program)
    }

    #[test]
    fn test_custom_xla_op_missing_linearized_jit_rule_reports_missing_rule() {
        let operation = XlaPrimitiveOperation::Custom(Arc::new(CustomPrimitive::new(TestCustomXlaOp)));
        let inputs: Vec<Linearized<ShardMapTracer>> = vec![];

        assert!(matches!(
            operation.interpret(&inputs),
            Err(TracingError::CustomOperation(CustomOperationError::MissingRule {
                op: "test_custom_xla",
                transform: "linearized JIT replay",
            }))
        ));
    }

    #[test]
    fn test_xla_vmap_jvp_stages_a_linear_vmap() {
        let operation = XlaPrimitiveOperation::VMap(Box::new(VMapOperation::new(unary_vmap_body())));
        let tangent_builder =
            Rc::new(RefCell::new(ProgramBuilder::<ArrayType, ShardMapTensor, XlaLinearOperation>::new()));
        let first_tangent_atom = tangent_builder.borrow_mut().add_input(scalar_type());
        let second_tangent_atom = tangent_builder.borrow_mut().add_input(scalar_type());
        let outputs = operation
            .jvp(
                crate::experimental::engine::XlaEngine::token(),
                &[
                    JvpTracer {
                        primal: ShardMapTensor::new(scalar_type()),
                        tangent: LinearTerm::from_staged_parts(first_tangent_atom, tangent_builder.clone()),
                    },
                    JvpTracer {
                        primal: ShardMapTensor::new(scalar_type()),
                        tangent: LinearTerm::from_staged_parts(second_tangent_atom, tangent_builder.clone()),
                    },
                ],
            )
            .expect("xla vmap jvp should succeed");
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].primal.r#type().into_owned(), scalar_type());
        assert_eq!(outputs[1].primal.r#type().into_owned(), scalar_type());

        let output_atoms = outputs.into_iter().map(|output| output.tangent.atom).collect::<Vec<_>>();
        let tangent_builder = Rc::try_unwrap(tangent_builder)
            .expect("vmap jvp builder should not have outstanding linear terms")
            .into_inner();
        let tangent_program = tangent_builder.build::<Vec<ShardMapTensor>, Vec<ShardMapTensor>>(
            output_atoms,
            vec![Placeholder, Placeholder],
            vec![Placeholder, Placeholder],
        );
        assert!(
            tangent_program.to_string().contains("vmap"),
            "expected linearized xla vmap jvp to stage a linear vmap op: {}",
            tangent_program
        );
    }

    #[test]
    fn test_xla_rematerialize_jvp_stages_a_linear_rematerialize() {
        let operation =
            XlaPrimitiveOperation::Rematerialize(Box::new(RematerializeOperation::new(unary_rematerialize_body())));
        let tangent_builder =
            Rc::new(RefCell::new(ProgramBuilder::<ArrayType, ShardMapTensor, XlaLinearOperation>::new()));
        let tangent_atom = tangent_builder.borrow_mut().add_input(scalar_type());
        let outputs = operation
            .jvp(
                crate::experimental::engine::XlaEngine::token(),
                &[JvpTracer {
                    primal: ShardMapTensor::new(scalar_type()),
                    tangent: LinearTerm::from_staged_parts(tangent_atom, tangent_builder.clone()),
                }],
            )
            .expect("xla rematerialize jvp should succeed");
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].primal.r#type().into_owned(), scalar_type());

        let output_atoms = outputs.into_iter().map(|output| output.tangent.atom).collect::<Vec<_>>();
        let tangent_builder = Rc::try_unwrap(tangent_builder)
            .expect("rematerialize jvp builder should not have outstanding linear terms")
            .into_inner();
        let tangent_program = tangent_builder.build::<Vec<ShardMapTensor>, Vec<ShardMapTensor>>(
            output_atoms,
            vec![Placeholder],
            vec![Placeholder],
        );
        assert!(
            tangent_program.to_string().contains("rematerialize"),
            "expected linearized xla rematerialize jvp to stage a linear rematerialize op: {}",
            tangent_program
        );
    }
}
