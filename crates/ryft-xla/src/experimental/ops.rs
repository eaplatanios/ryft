use std::{
    fmt::{Debug, Display},
    sync::Arc,
};

use ryft_core::{
    tracing::{InterpretableOperation, Operation, TracingError},
    tracing_v2::{
        CustomPrimitive, DifferentiableOperation, DifferentiationError, LinearPrimitiveOperation, LinearTerm,
        engine::Engine,
        forward::{Differentiable, EngineTangent, JvpTracer},
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
    operations::{LinearShardMapOperation, ShardMapOperation, WithShardingConstraintOperation},
    shard_map::{ShardMapTensor, ShardMapTracer},
};

type XlaLinearOperation = LinearPrimitiveOperation<ShardMapTensor>;

fn make_linear_xla_rematerialize<E>(
    engine: &E,
    body: &FlatTracedRematerialize<ArrayType, ShardMapTensor, XlaPrimitiveOperation>,
    input_primals: Vec<ShardMapTensor>,
) -> Result<LinearRematerializeOperation<ArrayType, ShardMapTensor>, TracingError>
where
    E: Engine<
            Type = ArrayType,
            Value = ShardMapTensor,
            TracingOperation = XlaPrimitiveOperation,
            LinearOperation = XlaLinearOperation,
        > + 'static,
{
    let body_program = body.program();
    let output_primals = body_program.interpret(input_primals.clone())?;
    let pushforward = linearize_program(engine, body_program, input_primals)?;
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

    /// Higher-order rematerialization.
    Rematerialize(Box<RematerializeOperation<ArrayType, ShardMapTensor, XlaPrimitiveOperation, XlaLinearOperation>>),

    /// XLA-specific `shard_map`.
    ShardMap(Box<ShardMapOperation<ShardMapTensor>>),

    /// XLA-specific `linear_shard_map`.
    LinearShardMap(Box<LinearShardMapOperation<ShardMapTensor>>),

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
            Self::Rematerialize(remat) => Debug::fmt(remat, formatter),
            Self::ShardMap(op) => Debug::fmt(op, formatter),
            Self::LinearShardMap(op) => Debug::fmt(op, formatter),
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

impl Operation<ArrayType> for XlaPrimitiveOperation {
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
            Self::Rematerialize(remat) => remat.name(),
            Self::ShardMap(op) => op.name(),
            Self::LinearShardMap(op) => op.name(),
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
            Self::Rematerialize(remat) => remat.infer_output_types(input_types),
            Self::ShardMap(op) => op.infer_output_types(input_types),
            Self::LinearShardMap(op) => op.infer_output_types(input_types),
            Self::WithShardingConstraint(op) => op.infer_output_types(input_types),
            Self::Custom(op) => op.infer_output_types(input_types),
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
            Self::Rematerialize(remat) => remat.interpret(inputs),
            Self::ShardMap(op) => op.interpret(inputs),
            Self::LinearShardMap(op) => op.interpret(inputs),
            Self::WithShardingConstraint(op) => op.interpret(inputs),
            Self::Custom(op) => op.interpret(inputs),
        }
    }
}

impl<E> DifferentiableOperation<E> for XlaPrimitiveOperation
where
    E: Engine<
            Type = ArrayType,
            Value = ShardMapTensor,
            TracingOperation = XlaPrimitiveOperation,
            LinearOperation = XlaLinearOperation,
        > + 'static,
    ShardMapTensor: Differentiable<
            ArrayType,
            Tangent<XlaLinearOperation> = LinearTerm<ArrayType, ShardMapTensor, XlaLinearOperation>,
        >,
{
    fn jvp(
        &self,
        engine: &E,
        inputs: &[JvpTracer<ShardMapTensor, EngineTangent<E>>],
    ) -> Result<Vec<JvpTracer<ShardMapTensor, EngineTangent<E>>>, TracingError> {
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
            Self::LinearShardMap(op) => op.jvp(engine, inputs),
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
            Self::Rematerialize(remat) => remat.interpret(inputs),
            Self::ShardMap(op) => op.interpret(inputs),
            Self::LinearShardMap(op) => op.interpret(inputs),
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
    use ryft_core::tracing::ProgramBuilder;
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

    impl Operation<ArrayType> for TestCustomXlaOp {
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
