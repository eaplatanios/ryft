use std::fmt::{Debug, Display};
use std::sync::Arc;

use ryft_core::operations::{InterpretableOperation, Operation};
use ryft_core::tracing::engines::TracingContext;
use ryft_core::tracing::{AtomId, TracingError};
use ryft_core::tracing_v2::differentiation::JvpTracer;
use ryft_core::tracing_v2::operations::{
    AddOperation, ConditionOperation, ConditionPredicate, ControlFlowError, CosOperation, FlatTracedRematerialize,
    LinearRematerializeOperation, MatMulOperation, MatrixTransposeOperation, MulOperation, NegOperation,
    OneLikeOperation, OneOperation, RematerializeOperation, ReshapeOperation, ScaleOperation, SinOperation,
    SupportsAdd, SupportsCos, SupportsCustom, SupportsMatMul, SupportsMatrixTranspose, SupportsMul, SupportsNeg,
    SupportsOne, SupportsOneLike, SupportsRematerialize, SupportsReshape, SupportsScale, SupportsSin, SupportsZero,
    SupportsZeroLike, WhileOperation, ZeroLikeOperation, ZeroOperation,
};
use ryft_core::tracing_v2::{
    CustomOperationError, CustomPrimitive, DifferentiableEngine, DifferentiableOperation, DifferentiationError,
    JvpContext, LinearArrayOperation,
};
use ryft_core::types::{ArrayType, Shape, TypeError};

use crate::experimental::engines::XlaEngine;
use crate::experimental::operations::{
    LinearShardMapOperation, ShardMapCustomReplayExtension, ShardMapOperation, ShardMapReplayContext,
    WithShardingConstraintOperation,
};
use crate::experimental::shard_map::{ShardMapTensor, ShardMapTracer};

/// Linear staged operation carrier used by the XLA backend.
pub type LinearXlaOperation<V = ShardMapTensor> = LinearArrayOperation<V>;

fn make_linear_xla_rematerialize<
    E: DifferentiableEngine<Type = ArrayType, Value = ShardMapTensor, LinearOperationCarrier = LinearXlaOperation>,
>(
    engine: &E,
    body: &FlatTracedRematerialize<ArrayType, ShardMapTensor, XlaOperation>,
    input_primals: Vec<ShardMapTensor>,
) -> Result<LinearRematerializeOperation<ArrayType, ShardMapTensor>, TracingError>
where
    XlaOperation: DifferentiableOperation<E>,
{
    let body_program = &body.program;
    let output_primals = body_program.interpret(input_primals.clone())?;
    let pushforward = body_program.linearize(engine, input_primals)?;
    let pullback = pushforward.transpose(output_primals.as_slice())?;
    Ok(LinearRematerializeOperation::new(
        FlatTracedRematerialize::from_parts(body.input_types.clone(), body.output_types.clone(), pushforward),
        FlatTracedRematerialize::from_parts(body.output_types.clone(), body.input_types.clone(), pullback),
    ))
}

#[cfg(test)]
fn replay_xla_program_with_tracers(
    program: &ryft_core::tracing::Program<
        ArrayType,
        ShardMapTensor,
        XlaOperation,
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
            values[atom_index] = Some(exemplar.context.constant(value.clone()));
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

fn interpret_xla_condition_jvp<
    E: DifferentiableEngine<Type = ArrayType, Value = ShardMapTensor, LinearOperationCarrier = LinearXlaOperation>,
>(
    condition: &ConditionOperation<ShardMapTensor, XlaOperation>,
    context: &mut JvpContext<'_, E>,
    inputs: &[JvpTracer<ShardMapTensor, AtomId>],
) -> Result<Vec<JvpTracer<ShardMapTensor, AtomId>>, TracingError>
where
    XlaOperation: DifferentiableOperation<E>,
{
    let ConditionPredicate::Captured(predicate) = &condition.predicate else {
        return Err(ControlFlowError::MissingTransformRule { transform: "runtime-predicate condition jvp" }.into());
    };
    let primal_inputs = inputs.iter().map(|input| input.primal.clone()).collect::<Vec<_>>();
    let tangent_inputs = inputs.iter().map(|input| input.tangent).collect::<Vec<_>>();
    if tangent_inputs.is_empty() && !condition.output_types().is_empty() {
        return Err(TracingError::InvalidInputCount { expected: 1, got: 0 });
    }

    let selected_branch = if *predicate { &condition.true_branch } else { &condition.false_branch };
    let primal_outputs = selected_branch.interpret(primal_inputs.clone())?;
    let true_pushforward = condition.true_branch.linearize(context.engine, primal_inputs.clone())?;
    let false_pushforward = condition.false_branch.linearize(context.engine, primal_inputs)?;
    let linear_condition = ConditionOperation::with_captured_predicate(*predicate, true_pushforward, false_pushforward)
        .map_err(TracingError::from)?;
    let tangent_outputs = context.apply_operation(
        tangent_inputs.as_slice(),
        LinearArrayOperation::Condition(Box::new(linear_condition)),
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
pub enum XlaOperation {
    /// Typed zero with no inputs and one output.
    Zero(ZeroOperation<ArrayType>),

    /// Typed one with no inputs and one output.
    One(OneOperation<ArrayType>),

    /// Exemplar-derived zero.
    ZeroLike,

    /// Exemplar-derived one.
    OneLike,

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
    MatrixMultiply,

    /// Matrix transpose.
    Transpose,

    /// Scaling by one captured factor.
    Scale { factor: ShardMapTensor },

    /// Reshape from one shape to another.
    Reshape { input_shape: Shape, output_shape: Shape },

    /// Higher-order rematerialization.
    Rematerialize(Box<RematerializeOperation<ArrayType, ShardMapTensor, XlaOperation, LinearXlaOperation>>),

    /// Higher-order conditional.
    Condition(Box<ConditionOperation<ShardMapTensor, XlaOperation>>),

    /// Higher-order while loop.
    While(Box<WhileOperation<ShardMapTensor, XlaOperation>>),

    /// XLA-specific `shard_map`.
    ShardMap(Box<ShardMapOperation<ShardMapTensor>>),

    /// XLA-specific `linear_shard_map`.
    LinearShardMap(Box<LinearShardMapOperation<ShardMapTensor>>),

    /// XLA-specific sharding constraint.
    WithShardingConstraint(WithShardingConstraintOperation),

    /// Explicit escape hatch for custom XLA ops.
    Custom(Arc<CustomPrimitive<ArrayType, ShardMapTensor>>),
}

impl Debug for XlaOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Zero(zero) => Debug::fmt(zero, formatter),
            Self::One(one) => Debug::fmt(one, formatter),
            Self::ZeroLike => write!(formatter, "ZeroLike"),
            Self::OneLike => write!(formatter, "OneLike"),
            Self::Add => write!(formatter, "Add"),
            Self::Mul => write!(formatter, "Mul"),
            Self::Neg => write!(formatter, "Neg"),
            Self::Sin => write!(formatter, "Sin"),
            Self::Cos => write!(formatter, "Cos"),
            Self::MatrixMultiply => write!(formatter, "MatrixMultiply"),
            Self::Transpose => write!(formatter, "Transpose"),
            Self::Scale { .. } => write!(formatter, "Scale"),
            Self::Reshape { input_shape, output_shape } => {
                write!(formatter, "Reshape({input_shape} -> {output_shape})")
            }
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

impl Display for XlaOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Reshape { output_shape, .. } => write!(formatter, "reshape{output_shape}"),
            _ => write!(formatter, "{}", self.name()),
        }
    }
}

impl Operation<ArrayType> for XlaOperation {
    fn name(&self) -> &'static str {
        match self {
            Self::Zero(zero) => zero.name(),
            Self::One(one) => one.name(),
            Self::ZeroLike => "zero_like",
            Self::OneLike => "one_like",
            Self::Add => "add",
            Self::Mul => "mul",
            Self::Neg => "neg",
            Self::Sin => "sin",
            Self::Cos => "cos",
            Self::MatrixMultiply => "matmul",
            Self::Transpose => "matrix_transpose",
            Self::Scale { .. } => "scale",
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
            Self::Zero(zero) => zero.infer_output_types(input_types),
            Self::One(one) => one.infer_output_types(input_types),
            Self::ZeroLike => ZeroLikeOperation.infer_output_types(input_types),
            Self::OneLike => OneLikeOperation.infer_output_types(input_types),
            Self::Add => AddOperation.infer_output_types(input_types),
            Self::Mul => MulOperation.infer_output_types(input_types),
            Self::Neg => NegOperation.infer_output_types(input_types),
            Self::Sin => SinOperation.infer_output_types(input_types),
            Self::Cos => CosOperation.infer_output_types(input_types),
            Self::MatrixMultiply => MatMulOperation.infer_output_types(input_types),
            Self::Transpose => MatrixTransposeOperation.infer_output_types(input_types),
            Self::Scale { .. } => ScaleOperation::<ArrayType, ShardMapTensor>::abstract_eval_static(input_types),
            Self::Reshape { input_shape, output_shape } => {
                ReshapeOperation::new(input_shape.clone(), output_shape.clone()).infer_output_types(input_types)
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

impl InterpretableOperation<ArrayType, ShardMapTensor> for XlaOperation {
    fn interpret(&self, inputs: &[ShardMapTensor]) -> Result<Vec<ShardMapTensor>, TracingError> {
        match self {
            Self::Zero(zero) => zero.interpret(inputs),
            Self::One(one) => one.interpret(inputs),
            Self::ZeroLike => ZeroLikeOperation.interpret(inputs),
            Self::OneLike => OneLikeOperation.interpret(inputs),
            Self::Add => AddOperation.interpret(inputs),
            Self::Mul => MulOperation.interpret(inputs),
            Self::Neg => NegOperation.interpret(inputs),
            Self::Sin => SinOperation.interpret(inputs),
            Self::Cos => CosOperation.interpret(inputs),
            Self::MatrixMultiply => MatMulOperation.interpret(inputs),
            Self::Transpose => MatrixTransposeOperation.interpret(inputs),
            Self::Scale { factor } => ScaleOperation::new(factor.clone()).interpret(inputs),
            Self::Reshape { input_shape, output_shape } => {
                ReshapeOperation::new(input_shape.clone(), output_shape.clone()).interpret(inputs)
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

impl InterpretableOperation<ArrayType, ShardMapTracer> for XlaOperation {
    fn interpret(&self, inputs: &[ShardMapTracer]) -> Result<Vec<ShardMapTracer>, TracingError> {
        match self {
            Self::Zero(zero) => Err(TypeError {
                message: format!(
                    "typed zero operation over tracer values was not materialized before interpretation for {}",
                    &zero.output_type
                ),
            }
            .into()),
            Self::One(one) => Err(TypeError {
                message: format!(
                    "typed one operation over tracer values was not materialized before interpretation for {}",
                    &one.output_type
                ),
            }
            .into()),
            Self::ZeroLike => ZeroLikeOperation.interpret(inputs),
            Self::OneLike => OneLikeOperation.interpret(inputs),
            Self::Add => AddOperation.interpret(inputs),
            Self::Mul => MulOperation.interpret(inputs),
            Self::Neg => NegOperation.interpret(inputs),
            Self::Sin => SinOperation.interpret(inputs),
            Self::Cos => CosOperation.interpret(inputs),
            Self::MatrixMultiply => MatMulOperation.interpret(inputs),
            Self::Transpose => MatrixTransposeOperation.interpret(inputs),
            Self::Scale { factor } => {
                let exemplar = inputs.first().ok_or(TracingError::InvalidInputCount { expected: 1, got: 0 })?;
                ScaleOperation::new(exemplar.context.constant(factor.clone())).interpret(inputs)
            }
            Self::Reshape { input_shape, output_shape } => {
                ReshapeOperation::new(input_shape.clone(), output_shape.clone()).interpret(inputs)
            }
            Self::Rematerialize(remat) => remat.interpret(inputs),
            Self::Condition(condition) => {
                let exemplar = inputs.first().ok_or(TracingError::InvalidInputCount { expected: 1, got: 0 })?;
                let input_refs = inputs.iter().collect::<Vec<_>>();
                exemplar.context.trace(XlaOperation::Condition(condition.clone()), input_refs.as_slice())
            }
            Self::While(while_operation) => {
                let exemplar = inputs.first().ok_or(TracingError::InvalidInputCount { expected: 1, got: 0 })?;
                let input_refs = inputs.iter().collect::<Vec<_>>();
                exemplar.context.trace(XlaOperation::While(while_operation.clone()), input_refs.as_slice())
            }
            Self::ShardMap(op) => {
                let exemplar = inputs.first().ok_or(TracingError::InvalidInputCount { expected: 1, got: 0 })?;
                op.interpret_traced_with_context(exemplar.builder().clone(), inputs)
            }
            Self::LinearShardMap(op) => {
                let exemplar = inputs.first().ok_or(TracingError::InvalidInputCount { expected: 1, got: 0 })?;
                op.interpret_traced_with_context(exemplar.builder().clone(), inputs)
            }
            Self::WithShardingConstraint(op) => op.interpret(inputs),
            Self::Custom(op) => {
                let exemplar = inputs.first().ok_or(TracingError::InvalidInputCount { expected: 1, got: 0 })?;
                let replay_context = ShardMapReplayContext::new(exemplar.builder().clone());
                op.extensions
                    .get::<ShardMapCustomReplayExtension>()
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

impl<'c> DifferentiableOperation<XlaEngine<'c>> for XlaOperation {
    fn jvp(
        &self,
        context: &mut JvpContext<'_, XlaEngine<'c>>,
        inputs: &[JvpTracer<ShardMapTensor, AtomId>],
    ) -> Result<Vec<JvpTracer<ShardMapTensor, AtomId>>, TracingError> {
        match self {
            Self::Zero(zero) => zero.jvp(context, inputs),
            Self::One(one) => one.jvp(context, inputs),
            Self::ZeroLike => ZeroLikeOperation.jvp(context, inputs),
            Self::OneLike => OneLikeOperation.jvp(context, inputs),
            Self::Add => AddOperation.jvp(context, inputs),
            Self::Mul => MulOperation.jvp(context, inputs),
            Self::Neg => NegOperation.jvp(context, inputs),
            Self::Sin => SinOperation.jvp(context, inputs),
            Self::Cos => CosOperation.jvp(context, inputs),
            Self::MatrixMultiply => MatMulOperation.jvp(context, inputs),
            Self::Transpose => MatrixTransposeOperation.jvp(context, inputs),
            Self::Scale { factor } => ScaleOperation::new(factor.clone()).jvp(context, inputs),
            Self::Reshape { input_shape, output_shape } => {
                ReshapeOperation::new(input_shape.clone(), output_shape.clone()).jvp(context, inputs)
            }
            Self::Rematerialize(remat) => {
                let primal_inputs = inputs.iter().map(|input| input.primal.clone()).collect::<Vec<_>>();
                let tangent_inputs = inputs.iter().map(|input| input.tangent).collect::<Vec<_>>();
                let primal_outputs = remat.interpret(primal_inputs.as_slice())?;
                if tangent_inputs.is_empty() && !remat.body.output_types.as_slice().is_empty() {
                    return Err(DifferentiationError::MissingLinearRematerializeReplayTangentLeaves.into());
                }
                let tangent_outputs = context.apply_operation(
                    tangent_inputs.as_slice(),
                    LinearArrayOperation::Rematerialize(Box::new(make_linear_xla_rematerialize(
                        context.engine,
                        &remat.body,
                        primal_inputs,
                    )?)),
                    remat.body.output_types.as_slice().len(),
                )?;
                Ok(primal_outputs
                    .into_iter()
                    .zip(tangent_outputs)
                    .map(|(primal, tangent)| JvpTracer { primal, tangent })
                    .collect::<Vec<_>>())
            }
            Self::Condition(condition) => interpret_xla_condition_jvp(condition, context, inputs),
            Self::While(_) => Err(ControlFlowError::MissingTransformRule { transform: "while jvp" }.into()),
            Self::ShardMap(op) => op.jvp(context, inputs),
            Self::LinearShardMap(op) => op.jvp(context, inputs),
            Self::WithShardingConstraint(op) => op.jvp(context, inputs),
            Self::Custom(op) => {
                Err(ryft_core::tracing_v2::CustomOperationError::MissingRule { op: op.name(), transform: "jvp" }.into())
            }
        }
    }
}

impl DifferentiableOperation<TracingContext<'static, XlaEngine<'static>>> for XlaOperation {
    fn jvp(
        &self,
        context: &mut JvpContext<'_, TracingContext<'static, XlaEngine<'static>>>,
        inputs: &[JvpTracer<ShardMapTracer, AtomId>],
    ) -> Result<Vec<JvpTracer<ShardMapTracer, AtomId>>, TracingError> {
        match self {
            Self::Zero(zero) => zero.jvp(context, inputs),
            Self::One(one) => one.jvp(context, inputs),
            Self::ZeroLike => ZeroLikeOperation.jvp(context, inputs),
            Self::OneLike => OneLikeOperation.jvp(context, inputs),
            Self::Add => AddOperation.jvp(context, inputs),
            Self::Mul => MulOperation.jvp(context, inputs),
            Self::Neg => NegOperation.jvp(context, inputs),
            Self::Sin => SinOperation.jvp(context, inputs),
            Self::Cos => CosOperation.jvp(context, inputs),
            Self::MatrixMultiply => MatMulOperation.jvp(context, inputs),
            Self::Transpose => MatrixTransposeOperation.jvp(context, inputs),
            Self::Scale { factor } => ScaleOperation::new(factor.clone()).jvp(context, inputs),
            Self::Reshape { input_shape, output_shape } => {
                ReshapeOperation::new(input_shape.clone(), output_shape.clone()).jvp(context, inputs)
            }
            Self::Rematerialize(remat) => remat.jvp(context, inputs),
            Self::Condition(condition) => condition.jvp(context, inputs),
            Self::While(while_operation) => while_operation.jvp(context, inputs),
            Self::ShardMap(op) => {
                let traced_op = ShardMapOperation::<ShardMapTracer>::new(op.body.clone());
                traced_op.jvp_with_builders(context.engine.builder.clone(), context, inputs)
            }
            Self::LinearShardMap(op) => op.jvp_traced_with_builders(context.engine.builder.clone(), context, inputs),
            Self::WithShardingConstraint(op) => {
                let input = inputs.first().ok_or(TracingError::InvalidInputCount { expected: 1, got: 0 })?;
                let primal = input
                    .primal
                    .context
                    .trace(XlaOperation::WithShardingConstraint(op.clone()), &[&input.primal])?
                    .into_iter()
                    .next()
                    .expect("with_sharding_constraint should produce one primal output");
                let tangent = context
                    .apply_operation(
                        &[input.tangent],
                        LinearArrayOperation::Custom(Arc::new(op.to_tracer_linear_custom_primitive())),
                        1,
                    )?
                    .into_iter()
                    .next()
                    .expect("with_sharding_constraint should produce one tangent output");
                Ok(vec![JvpTracer { primal, tangent }])
            }
            Self::Custom(op) => {
                Err(CustomOperationError::MissingRule { op: op.name(), transform: "traced linearization" }.into())
            }
        }
    }
}

impl SupportsAdd<ArrayType, ShardMapTensor> for XlaOperation {
    fn add_operation() -> Self {
        XlaOperation::Add
    }
}

impl SupportsMul<ArrayType, ShardMapTensor> for XlaOperation {
    fn mul_operation() -> Self {
        XlaOperation::Mul
    }
}

impl SupportsNeg<ArrayType, ShardMapTensor> for XlaOperation {
    fn neg_operation() -> Self {
        XlaOperation::Neg
    }
}

impl SupportsSin<ArrayType, ShardMapTensor> for XlaOperation {
    fn sin_operation() -> Self {
        XlaOperation::Sin
    }
}

impl SupportsCos<ArrayType, ShardMapTensor> for XlaOperation {
    fn cos_operation() -> Self {
        XlaOperation::Cos
    }
}

impl SupportsZero<ArrayType, ShardMapTensor> for XlaOperation {
    fn zero_operation(r#type: ArrayType) -> Self {
        XlaOperation::Zero(ZeroOperation::new(r#type))
    }

    fn as_zero_operation(&self) -> Option<&ZeroOperation<ArrayType>> {
        match self {
            Self::Zero(zero) => Some(zero),
            _ => None,
        }
    }
}

impl SupportsOne<ArrayType, ShardMapTensor> for XlaOperation {
    fn one_operation(r#type: ArrayType) -> Self {
        XlaOperation::One(OneOperation::new(r#type))
    }
}

impl SupportsZeroLike<ArrayType, ShardMapTensor> for XlaOperation {
    fn zero_like_operation() -> Self {
        XlaOperation::ZeroLike
    }
}

impl SupportsOneLike<ArrayType, ShardMapTensor> for XlaOperation {
    fn one_like_operation() -> Self {
        XlaOperation::OneLike
    }
}

impl SupportsMatMul<ArrayType, ShardMapTensor> for XlaOperation {
    fn matmul_operation() -> Self {
        XlaOperation::MatrixMultiply
    }
}

impl SupportsMatrixTranspose<ArrayType, ShardMapTensor> for XlaOperation {
    fn matrix_transpose_operation() -> Self {
        XlaOperation::Transpose
    }
}

impl SupportsCustom<ArrayType, ShardMapTensor> for XlaOperation {
    fn custom_operation(primitive: Arc<CustomPrimitive<ArrayType, ShardMapTensor>>) -> Self {
        XlaOperation::Custom(primitive)
    }
}

impl SupportsRematerialize<ArrayType, ShardMapTensor, LinearXlaOperation> for XlaOperation {
    fn rematerialize_operation(
        op: RematerializeOperation<ArrayType, ShardMapTensor, XlaOperation, LinearXlaOperation>,
    ) -> Self {
        XlaOperation::Rematerialize(Box::new(op))
    }
}

impl SupportsScale<ArrayType, ShardMapTensor> for XlaOperation {
    fn scale_operation(factor: ShardMapTensor) -> Self {
        XlaOperation::Scale { factor }
    }
}

impl SupportsReshape<ArrayType, ShardMapTensor> for XlaOperation {
    fn reshape_operation(input_shape: Shape, output_shape: Shape) -> Self {
        XlaOperation::Reshape { input_shape, output_shape }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;
    use std::sync::Arc;

    use pretty_assertions::assert_eq;

    use ryft_core::parameters::Placeholder;
    use ryft_core::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding};
    use ryft_core::tracing::ProgramBuilder;
    use ryft_core::types::{DataType, Typed};

    use crate::experimental::engines::XlaEngine;

    use super::*;

    fn scalar_type() -> ArrayType {
        ArrayType::scalar(DataType::F32)
    }

    fn test_mesh() -> LogicalMesh {
        LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap()
    }

    fn unary_rematerialize_body() -> FlatTracedRematerialize<ArrayType, ShardMapTensor, XlaOperation> {
        let mut builder = ProgramBuilder::<ArrayType, ShardMapTensor, XlaOperation>::new();
        let input = builder.add_input(scalar_type());
        let output = builder
            .add_instruction(XlaOperation::Sin, vec![input])
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
    fn test_xla_rematerialize_jvp_stages_a_linear_rematerialize() {
        let operation = XlaOperation::Rematerialize(Box::new(RematerializeOperation::new(unary_rematerialize_body())));
        let tangent_builder =
            Rc::new(RefCell::new(ProgramBuilder::<ArrayType, ShardMapTensor, LinearXlaOperation>::new()));
        let tangent_atom = tangent_builder.borrow_mut().add_input(scalar_type());
        let engine = crate::experimental::engines::XlaEngine::token();
        let mut context = JvpContext::new(engine, tangent_builder.clone());
        let outputs = operation
            .jvp(&mut context, &[JvpTracer { primal: ShardMapTensor::new(scalar_type()), tangent: tangent_atom }])
            .expect("xla rematerialize jvp should succeed");
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].primal.r#type().into_owned(), scalar_type());

        let output_atoms = outputs.into_iter().map(|output| output.tangent).collect::<Vec<_>>();
        drop(context);
        let tangent_builder = Rc::try_unwrap(tangent_builder)
            .expect("rematerialize jvp builder should not have outstanding linear terms")
            .into_inner();
        let tangent_program = tangent_builder
            .build::<Vec<ShardMapTensor>, Vec<ShardMapTensor>>(output_atoms, vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert!(
            tangent_program.to_string().contains("rematerialize"),
            "expected xla rematerialize jvp to stage a linear rematerialize op: {}",
            tangent_program
        );
    }

    #[test]
    fn test_replay_xla_program_with_tracers_uses_custom_replay_extension() {
        let sharding = Sharding::replicated(test_mesh(), 0);
        let custom = WithShardingConstraintOperation::new(sharding).to_tensor_custom_primitive();
        let mut program_builder = ProgramBuilder::<ArrayType, ShardMapTensor, XlaOperation>::new();
        let input = program_builder.add_input(scalar_type());
        let output = program_builder
            .add_instruction(XlaOperation::Custom(Arc::new(custom)), vec![input])
            .expect("custom op should stage")
            .into_iter()
            .copied()
            .next()
            .expect("custom op should produce one output");
        let program = program_builder
            .build::<Vec<ShardMapTensor>, Vec<ShardMapTensor>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let tracing_builder = Rc::new(RefCell::new(ProgramBuilder::<ArrayType, ShardMapTensor, XlaOperation>::new()));
        let traced_input_atom = tracing_builder.borrow_mut().add_input(scalar_type());
        let traced_input =
            TracingContext::new(XlaEngine::token(), tracing_builder).tracer(traced_input_atom, Some(scalar_type()));

        let outputs = replay_xla_program_with_tracers(&program, vec![traced_input]).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].r#type().into_owned(), scalar_type());
    }
}
