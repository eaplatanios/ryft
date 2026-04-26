use std::{
    fmt::{Debug, Display},
    sync::Arc,
};

use ryft_core::{
    tracing::{InterpretableOperation, Operation, TracingError},
    tracing_v2::{
        CustomPrimitive, DifferentiableOperation, DifferentiationError, LinearPrimitiveOperation, LinearTerm,
        engines::DifferentiableEngine,
        forward::{Differentiable, EngineTangent, JvpTracer},
        linear::{Linearized, linearize_program, transpose_linear_program_with_output_examples},
        operations::{
            AddOperation, ConditionOperation, ConditionPredicate, ControlFlowError, CosOperation,
            FlatTracedRematerialize, LeftMatMulOperation, LinearRematerializeOperation, MatMulOperation,
            MatrixTransposeOperation, MulOperation, NegOperation, RematerializeOperation, ReshapeOperation,
            RightMatMulOperation, ScaleOperation, SinOperation, SupportsAdd, SupportsCos, SupportsCustom,
            SupportsLeftMatMul, SupportsMatMul, SupportsMatrixTranspose, SupportsMul, SupportsNeg,
            SupportsRematerialize, SupportsReshape, SupportsRightMatMul, SupportsScale, SupportsSin, WhileOperation,
            left_matmul::left_matmul_abstract_eval, lift_jit_constant, right_matmul::right_matmul_abstract_eval,
        },
    },
    types::{ArrayType, TypeError, Typed},
};

use crate::experimental::{
    operations::{
        LinearShardMapOperation, ShardMapCustomReplayExtension, ShardMapOperation, ShardMapReplayContext,
        WithShardingConstraintOperation,
    },
    shard_map::{ShardMapTensor, ShardMapTracer},
};

type XlaLinearOperation = LinearPrimitiveOperation<ShardMapTensor>;

fn make_linear_xla_rematerialize<E>(
    engine: &E,
    body: &FlatTracedRematerialize<ArrayType, ShardMapTensor, XlaPrimitiveOperation>,
    input_primals: Vec<ShardMapTensor>,
) -> Result<LinearRematerializeOperation<ArrayType, ShardMapTensor>, TracingError>
where
    E: DifferentiableEngine<Type = ArrayType, Value = ShardMapTensor, LinearOperation = XlaLinearOperation>,
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

#[cfg(test)]
fn replay_xla_program_with_tracers(
    program: &ryft_core::tracing::Program<
        ArrayType,
        ShardMapTensor,
        XlaPrimitiveOperation,
        Vec<ShardMapTensor>,
        Vec<ShardMapTensor>,
    >,
    inputs: Vec<ShardMapTracer>,
) -> Result<Vec<ShardMapTracer>, TracingError> {
    let exemplar = inputs.first().cloned();
    let mut values = vec![None; program.atoms.len()];
    for (atom_id, value) in program.input_ids.iter().copied().zip(inputs) {
        values[atom_id.index] = Some(value);
    }
    for (atom_index, atom) in program.atoms.iter().enumerate() {
        if let ryft_core::tracing::Atom::Constant(value) = atom {
            let Some(exemplar) = exemplar.as_ref() else {
                return Err(TracingError::InvalidInputCount { expected: 1, got: 0 });
            };
            values[atom_index] = Some(lift_jit_constant(value, exemplar));
        }
    }
    for instruction in program.instructions.iter() {
        let inputs = instruction
            .inputs
            .iter()
            .map(|input| values[input.index].clone().ok_or(TracingError::UnboundAtomId { id: *input }))
            .collect::<Result<Vec<_>, _>>()?;
        let outputs = instruction.operation.interpret(inputs.as_slice())?;
        if outputs.len() != instruction.outputs.len() {
            return Err(TracingError::InvalidOutputCount { expected: instruction.outputs.len(), got: outputs.len() });
        }
        for (output, value) in instruction.outputs.iter().copied().zip(outputs) {
            values[output.index] = Some(value);
        }
    }
    program
        .output_ids
        .iter()
        .map(|output| values[output.index].clone().ok_or(TracingError::UnboundAtomId { id: *output }))
        .collect()
}

fn interpret_xla_condition_jvp<E>(
    condition: &ConditionOperation<ShardMapTensor, XlaPrimitiveOperation>,
    inputs: &[JvpTracer<ShardMapTensor, EngineTangent<E>>],
    engine: &E,
) -> Result<Vec<JvpTracer<ShardMapTensor, EngineTangent<E>>>, TracingError>
where
    E: DifferentiableEngine<Type = ArrayType, Value = ShardMapTensor, LinearOperation = XlaLinearOperation>,
{
    let ConditionPredicate::Captured(predicate) = condition.predicate() else {
        return Err(ControlFlowError::MissingTransformRule { transform: "runtime-predicate condition jvp" }.into());
    };
    let primal_inputs = inputs.iter().map(|input| input.primal.clone()).collect::<Vec<_>>();
    let tangent_inputs = inputs.iter().map(|input| input.tangent.clone()).collect::<Vec<_>>();
    let Some(tangent_builder) = tangent_inputs.first().map(|input| input.builder.clone()) else {
        return if condition.output_types().is_empty() {
            Ok(Vec::new())
        } else {
            Err(TracingError::InvalidInputCount { expected: 1, got: 0 })
        };
    };

    let selected_branch = if *predicate { condition.true_branch() } else { condition.false_branch() };
    let primal_outputs = selected_branch.interpret(primal_inputs.clone())?;
    let true_pushforward = linearize_program(engine, condition.true_branch(), primal_inputs.clone())?;
    let false_pushforward = linearize_program(engine, condition.false_branch(), primal_inputs)?;
    let linear_condition = ConditionOperation::with_captured_predicate(*predicate, true_pushforward, false_pushforward)
        .map_err(TracingError::from)?;
    let tangent_outputs = LinearTerm::apply_staged_op(
        tangent_builder,
        tangent_inputs.as_slice(),
        LinearPrimitiveOperation::Condition(Box::new(linear_condition)),
        condition.output_types().len(),
    )?;
    Ok(primal_outputs
        .into_iter()
        .zip(tangent_outputs)
        .map(|(primal, tangent)| JvpTracer { primal, tangent })
        .collect())
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

    /// Higher-order conditional.
    Condition(Box<ConditionOperation<ShardMapTensor, XlaPrimitiveOperation>>),

    /// Higher-order while loop.
    While(Box<WhileOperation<ShardMapTensor, XlaPrimitiveOperation>>),

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
            Self::Condition(condition) => Debug::fmt(condition, formatter),
            Self::While(while_operation) => Debug::fmt(while_operation, formatter),
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
            Self::Condition(condition) => condition.name(),
            Self::While(while_operation) => while_operation.name(),
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
            Self::Condition(condition) => condition.infer_output_types(input_types),
            Self::While(while_operation) => while_operation.infer_output_types(input_types),
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
            Self::Condition(condition) => condition.interpret(inputs),
            Self::While(while_operation) => while_operation.interpret(inputs),
            Self::ShardMap(op) => op.interpret(inputs),
            Self::LinearShardMap(op) => op.interpret(inputs),
            Self::WithShardingConstraint(op) => op.interpret(inputs),
            Self::Custom(op) => op.interpret(inputs),
        }
    }
}

impl InterpretableOperation<ArrayType, ShardMapTracer> for XlaPrimitiveOperation {
    fn interpret(&self, inputs: &[ShardMapTracer]) -> Result<Vec<ShardMapTracer>, TracingError> {
        match self {
            Self::Add => AddOperation.interpret(inputs),
            Self::Mul => MulOperation.interpret(inputs),
            Self::Neg => NegOperation.interpret(inputs),
            Self::Sin => SinOperation.interpret(inputs),
            Self::Cos => CosOperation.interpret(inputs),
            Self::MatMul => MatMulOperation.interpret(inputs),
            Self::MatrixTranspose => MatrixTransposeOperation.interpret(inputs),
            Self::Scale { factor } => {
                let exemplar = inputs.first().ok_or(TracingError::InvalidInputCount { expected: 1, got: 0 })?;
                ScaleOperation::new(lift_jit_constant(factor, exemplar)).interpret(inputs)
            }
            Self::LeftMatMul { factor } => {
                let exemplar = inputs.first().ok_or(TracingError::InvalidInputCount { expected: 1, got: 0 })?;
                LeftMatMulOperation::new(lift_jit_constant(factor, exemplar)).interpret(inputs)
            }
            Self::RightMatMul { factor } => {
                let exemplar = inputs.first().ok_or(TracingError::InvalidInputCount { expected: 1, got: 0 })?;
                RightMatMulOperation::new(lift_jit_constant(factor, exemplar)).interpret(inputs)
            }
            Self::Reshape { input_type, output_type } => {
                ReshapeOperation::new(input_type.clone(), output_type.clone()).interpret(inputs)
            }
            Self::Rematerialize(remat) => remat.interpret(inputs),
            Self::Condition(condition) => {
                let exemplar = inputs.first().ok_or(TracingError::InvalidInputCount { expected: 1, got: 0 })?;
                ryft_core::tracing_v2::Tracer::apply_staged_op(
                    exemplar.engine,
                    exemplar.builder.clone(),
                    inputs,
                    XlaPrimitiveOperation::Condition(condition.clone()),
                )
            }
            Self::While(while_operation) => {
                let exemplar = inputs.first().ok_or(TracingError::InvalidInputCount { expected: 1, got: 0 })?;
                ryft_core::tracing_v2::Tracer::apply_staged_op(
                    exemplar.engine,
                    exemplar.builder.clone(),
                    inputs,
                    XlaPrimitiveOperation::While(while_operation.clone()),
                )
            }
            Self::ShardMap(op) => {
                let exemplar = inputs.first().ok_or(TracingError::InvalidInputCount { expected: 1, got: 0 })?;
                op.interpret_traced_with_context(exemplar.builder.clone(), inputs)
            }
            Self::LinearShardMap(op) => {
                let exemplar = inputs.first().ok_or(TracingError::InvalidInputCount { expected: 1, got: 0 })?;
                op.interpret_traced_with_context(exemplar.builder.clone(), inputs)
            }
            Self::WithShardingConstraint(op) => op.interpret(inputs),
            Self::Custom(op) => {
                let exemplar = inputs.first().ok_or(TracingError::InvalidInputCount { expected: 1, got: 0 })?;
                let replay_context = ShardMapReplayContext::new(exemplar.builder.clone());
                op.extensions()
                    .get::<ShardMapCustomReplayExtension<ShardMapTracer>>()
                    .ok_or_else(|| {
                        TracingError::CustomOperation(ryft_core::tracing_v2::CustomOperationError::MissingRule {
                            op: op.name(),
                            transform: "traced replay",
                        })
                    })?
                    .replay(&replay_context, inputs.to_vec())
                    .map_err(|error| TracingError::Type(TypeError { message: error.to_string() }))
            }
        }
    }
}

impl<E> DifferentiableOperation<E> for XlaPrimitiveOperation
where
    E: DifferentiableEngine<Type = ArrayType, Value = ShardMapTensor, LinearOperation = XlaLinearOperation>,
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
            Self::Condition(condition) => interpret_xla_condition_jvp::<E>(condition, inputs, engine),
            Self::While(_) => Err(ControlFlowError::MissingTransformRule { transform: "while jvp" }.into()),
            Self::ShardMap(op) => op.jvp(engine, inputs),
            Self::LinearShardMap(op) => op.jvp(engine, inputs),
            Self::WithShardingConstraint(op) => op.jvp(engine, inputs),
            Self::Custom(op) => {
                Err(ryft_core::tracing_v2::CustomOperationError::MissingRule { op: op.name(), transform: "jvp" }.into())
            }
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
            Self::Condition(_) | Self::While(_) => {
                Err(ControlFlowError::MissingTransformRule { transform: "linearized JIT replay" }.into())
            }
            Self::ShardMap(op) => op.interpret(inputs),
            Self::LinearShardMap(op) => op.interpret(inputs),
            Self::WithShardingConstraint(op) => op.interpret(inputs),
            Self::Custom(op) => op.interpret(inputs),
        }
    }
}

impl SupportsAdd<ArrayType, ShardMapTensor> for XlaPrimitiveOperation {
    fn add_operation() -> Self {
        XlaPrimitiveOperation::Add
    }
}

impl SupportsMul<ArrayType, ShardMapTensor> for XlaPrimitiveOperation {
    fn mul_operation() -> Self {
        XlaPrimitiveOperation::Mul
    }
}

impl SupportsNeg<ArrayType, ShardMapTensor> for XlaPrimitiveOperation {
    fn neg_operation() -> Self {
        XlaPrimitiveOperation::Neg
    }
}

impl SupportsSin<ArrayType, ShardMapTensor> for XlaPrimitiveOperation {
    fn sin_operation() -> Self {
        XlaPrimitiveOperation::Sin
    }
}

impl SupportsCos<ArrayType, ShardMapTensor> for XlaPrimitiveOperation {
    fn cos_operation() -> Self {
        XlaPrimitiveOperation::Cos
    }
}

impl SupportsMatMul<ArrayType, ShardMapTensor> for XlaPrimitiveOperation {
    fn matmul_operation() -> Self {
        XlaPrimitiveOperation::MatMul
    }
}

impl SupportsMatrixTranspose<ArrayType, ShardMapTensor> for XlaPrimitiveOperation {
    fn matrix_transpose_operation() -> Self {
        XlaPrimitiveOperation::MatrixTranspose
    }
}

impl SupportsCustom<ArrayType, ShardMapTensor> for XlaPrimitiveOperation {
    fn custom_operation(primitive: Arc<CustomPrimitive<ArrayType, ShardMapTensor>>) -> Self {
        XlaPrimitiveOperation::Custom(primitive)
    }
}

impl SupportsRematerialize<ArrayType, ShardMapTensor, XlaLinearOperation> for XlaPrimitiveOperation {
    fn rematerialize_operation(
        op: RematerializeOperation<ArrayType, ShardMapTensor, XlaPrimitiveOperation, XlaLinearOperation>,
    ) -> Self {
        XlaPrimitiveOperation::Rematerialize(Box::new(op))
    }
}

impl SupportsScale<ArrayType, ShardMapTensor> for XlaPrimitiveOperation {
    fn scale_operation(factor: ShardMapTensor) -> Self {
        XlaPrimitiveOperation::Scale { factor }
    }
}

impl SupportsLeftMatMul<ArrayType, ShardMapTensor> for XlaPrimitiveOperation {
    fn left_matmul_operation(factor: ShardMapTensor) -> Self {
        XlaPrimitiveOperation::LeftMatMul { factor }
    }
}

impl SupportsRightMatMul<ArrayType, ShardMapTensor> for XlaPrimitiveOperation {
    fn right_matmul_operation(factor: ShardMapTensor) -> Self {
        XlaPrimitiveOperation::RightMatMul { factor }
    }
}

impl SupportsReshape<ArrayType, ShardMapTensor> for XlaPrimitiveOperation {
    fn reshape_operation(input_type: ArrayType, output_type: ArrayType) -> Self {
        XlaPrimitiveOperation::Reshape { input_type, output_type }
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, fmt::Display, rc::Rc, sync::Arc};

    use pretty_assertions::assert_eq;

    use ryft_core::parameters::Placeholder;
    use ryft_core::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding};
    use ryft_core::tracing::ProgramBuilder;
    use ryft_core::tracing_v2::Tracer;
    use ryft_core::tracing_v2::operations::CustomOperationError;
    use ryft_core::types::{DataType, Typed};

    use crate::experimental::engines::XlaEngine;

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

    fn test_mesh() -> LogicalMesh {
        LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap()
    }

    fn unary_rematerialize_body() -> FlatTracedRematerialize<ArrayType, ShardMapTensor, XlaPrimitiveOperation> {
        let mut builder = ProgramBuilder::<ArrayType, ShardMapTensor, XlaPrimitiveOperation>::new();
        let input = builder.add_input(scalar_type());
        let output = builder
            .add_instruction(XlaPrimitiveOperation::Sin, vec![input])
            .expect("rematerialize body should stage one sine op")
            .into_iter()
            .copied()
            .next()
            .expect("sine should produce one output");
        let program = builder
            .build::<Vec<ShardMapTensor>, Vec<ShardMapTensor>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
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
                crate::experimental::engines::XlaEngine::token(),
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
        let tangent_program = tangent_builder
            .build::<Vec<ShardMapTensor>, Vec<ShardMapTensor>>(output_atoms, vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert!(
            tangent_program.to_string().contains("rematerialize"),
            "expected linearized xla rematerialize jvp to stage a linear rematerialize op: {}",
            tangent_program
        );
    }

    #[test]
    fn test_replay_xla_program_with_tracers_uses_custom_replay_extension() {
        let sharding = Sharding::replicated(test_mesh(), 0);
        let custom = WithShardingConstraintOperation::new(sharding).to_tensor_custom_primitive();
        let mut program_builder = ProgramBuilder::<ArrayType, ShardMapTensor, XlaPrimitiveOperation>::new();
        let input = program_builder.add_input(scalar_type());
        let output = program_builder
            .add_instruction(XlaPrimitiveOperation::Custom(Arc::new(custom)), vec![input])
            .expect("custom op should stage")
            .into_iter()
            .copied()
            .next()
            .expect("custom op should produce one output");
        let program = program_builder
            .build::<Vec<ShardMapTensor>, Vec<ShardMapTensor>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let tracing_builder =
            Rc::new(RefCell::new(ProgramBuilder::<ArrayType, ShardMapTensor, XlaPrimitiveOperation>::new()));
        let traced_input_atom = tracing_builder.borrow_mut().add_input(scalar_type());
        let traced_input =
            Tracer::from_staged_parts(traced_input_atom, scalar_type(), tracing_builder, XlaEngine::token());

        let outputs = replay_xla_program_with_tracers(&program, vec![traced_input]).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].r#type().into_owned(), scalar_type());
    }
}
