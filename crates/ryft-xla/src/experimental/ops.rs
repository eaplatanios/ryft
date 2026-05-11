use std::fmt::{Debug, Display};
use std::ops::{Add, Mul, Neg};
use std::sync::Arc;

use ryft_core::differentiation::{Cotangent, LinearOperation};
use ryft_core::macros::check_count;
use ryft_core::operations::arithmetic::{
    ADD_OPERATION_NAME, AddOperation, DIV_OPERATION_NAME, DivOperation, MUL_OPERATION_NAME, MulOperation,
    NEG_OPERATION_NAME, NegOperation, SCALE_OPERATION_NAME, SUB_OPERATION_NAME, ScaleOperation, SubOperation,
    SupportsAdd, SupportsDiv, SupportsMul, SupportsNeg, SupportsScale, SupportsSub,
};
use ryft_core::operations::constants::{
    ONE_LIKE_OPERATION_NAME, OneLikeOperation, OneOperation, SupportsOne, SupportsOneLike, SupportsZero,
    SupportsZeroLike, ZERO_LIKE_OPERATION_NAME, ZeroLike, ZeroLikeOperation, ZeroOperation,
};
use ryft_core::operations::trigonometric::{CosOperation, SinOperation, SupportsCos, SupportsSin};
use ryft_core::operations::{InterpretableOperation, Operation};
use ryft_core::parameters::Parameterized;
use ryft_core::tracing::domains::{Tracer, TracingContext};
use ryft_core::tracing::{ProgramTracingContext, Traceable, TracingError};
use ryft_core::tracing_v2::differentiation::JvpTracer;
use ryft_core::tracing_v2::operations::{
    ConditionOperation, ControlFlowValue, LeftMatMulOperation, MatMulOperation, MatrixOps, MatrixTransposeOperation,
    ReshapeOperation, ReshapeOps, RightMatMulOperation, SupportsCustom, SupportsLeftMatMul, SupportsMatMul,
    SupportsMatrixTranspose, SupportsReshape, SupportsRightMatMul, WhileOperation,
};
use ryft_core::tracing_v2::{
    CustomOperationError, CustomPrimitive, DifferentiableOperation, JvpContext, LinearArrayOperation,
    LinearOperationCarrierFamily,
};
use ryft_core::types::{ArrayType, Shape, TypeError};

use crate::experimental::domains::{LinearXlaDomain, XlaDomain};
use crate::experimental::operations::{
    LinearShardMapOperation, ShardMapCustomReplayExtension, ShardMapOperation, ShardMapReplayContext,
    WithShardingConstraintOperation,
};
use crate::experimental::shard_map::{ShardMapTensor, ShardMapTracer};

/// Closed linear staged-op universe owned by the XLA backend.
#[derive(Clone, Debug)]
pub enum LinearXlaOperation<V>
where
    V: Traceable<ArrayType>,
{
    /// Generic array linear operation reused from `ryft-core`.
    Array(LinearArrayOperation<V, ArrayType>),

    /// Higher-order conditional whose nested linear programs can contain XLA linear operations.
    Condition(Box<ConditionOperation<V, LinearXlaOperation<V>, ArrayType>>),

    /// Higher-order while loop whose nested linear programs can contain XLA linear operations.
    While(Box<WhileOperation<V, LinearXlaOperation<V>, ArrayType>>),

    /// XLA-specific linear `shard_map`.
    LinearShardMap(Box<LinearShardMapOperation<V>>),

    /// XLA-specific sharding constraint in tangent/cotangent programs.
    WithShardingConstraint(WithShardingConstraintOperation),
}

impl<V> Display for LinearXlaOperation<V>
where
    V: Traceable<ArrayType>,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Array(op) => Display::fmt(op, formatter),
            Self::Condition(condition) => Display::fmt(condition, formatter),
            Self::While(while_operation) => Display::fmt(while_operation, formatter),
            Self::LinearShardMap(op) => Display::fmt(op, formatter),
            Self::WithShardingConstraint(op) => Display::fmt(op, formatter),
        }
    }
}

impl<V> Operation<ArrayType> for LinearXlaOperation<V>
where
    V: Traceable<ArrayType>,
{
    #[inline]
    fn name(&self) -> &'static str {
        match self {
            Self::Array(op) => op.name(),
            Self::Condition(condition) => condition.name(),
            Self::While(while_operation) => while_operation.name(),
            Self::LinearShardMap(op) => op.name(),
            Self::WithShardingConstraint(op) => op.name(),
        }
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        match self {
            Self::Array(op) => op.infer_output_types(input_types),
            Self::Condition(condition) => condition.infer_output_types(input_types),
            Self::While(while_operation) => while_operation.infer_output_types(input_types),
            Self::LinearShardMap(op) => op.infer_output_types(input_types),
            Self::WithShardingConstraint(op) => op.infer_output_types(input_types),
        }
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        match self {
            Self::Array(op) => op.render(formatter, indentation),
            Self::Condition(condition) => condition.render(formatter, indentation),
            Self::While(while_operation) => while_operation.render(formatter, indentation),
            Self::LinearShardMap(op) => op.render(formatter, indentation),
            Self::WithShardingConstraint(op) => op.render(formatter, indentation),
        }
    }
}

impl<V> InterpretableOperation<ArrayType, V> for LinearXlaOperation<V>
where
    V: Traceable<ArrayType> + ControlFlowValue,
    LinearArrayOperation<V, ArrayType>: InterpretableOperation<ArrayType, V>,
    LinearShardMapOperation<V>: InterpretableOperation<ArrayType, V>,
    WithShardingConstraintOperation: InterpretableOperation<ArrayType, V>,
{
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        match self {
            Self::Array(op) => op.interpret(inputs),
            Self::Condition(condition) => condition.interpret(inputs),
            Self::While(while_operation) => while_operation.interpret(inputs),
            Self::LinearShardMap(op) => op.interpret(inputs),
            Self::WithShardingConstraint(op) => op.interpret(inputs),
        }
    }
}

impl<V> LinearOperation<ArrayType, V, LinearXlaOperation<V>> for LinearXlaOperation<V>
where
    V: Traceable<ArrayType>
        + Add<Output = V>
        + Neg<Output = V>
        + Mul<Output = V>
        + ZeroLike
        + MatrixOps
        + ReshapeOps
        + ControlFlowValue,
    Vec<V>: Parameterized<V, ParameterStructure: Debug + PartialEq>,
    LinearShardMapOperation<V>: LinearOperation<ArrayType, V, LinearXlaOperation<V>>,
    WithShardingConstraintOperation: LinearOperation<ArrayType, V, LinearXlaOperation<V>>,
{
    fn transpose<'transpose>(
        &self,
        context: &mut ProgramTracingContext<'transpose, ArrayType, V, LinearXlaOperation<V>>,
        output_cotangents: &[Cotangent<'transpose, ArrayType, V, LinearXlaOperation<V>>],
    ) -> Result<Vec<Cotangent<'transpose, ArrayType, V, LinearXlaOperation<V>>>, TracingError> {
        match self {
            Self::Array(op) => match op {
                LinearArrayOperation::Zero(zero) => zero.transpose(context, output_cotangents),
                LinearArrayOperation::One(one) => one.transpose(context, output_cotangents),
                LinearArrayOperation::ZeroLike => ZeroLikeOperation.transpose(context, output_cotangents),
                LinearArrayOperation::OneLike => OneLikeOperation.transpose(context, output_cotangents),
                LinearArrayOperation::Add => AddOperation.transpose(context, output_cotangents),
                LinearArrayOperation::Sub => SubOperation.transpose(context, output_cotangents),
                LinearArrayOperation::Neg => NegOperation.transpose(context, output_cotangents),
                LinearArrayOperation::Transpose => MatrixTransposeOperation.transpose(context, output_cotangents),
                LinearArrayOperation::Scale { factor } => {
                    ScaleOperation::new(factor.clone()).transpose(context, output_cotangents)
                }
                LinearArrayOperation::LeftMatMul { factor } => {
                    LeftMatMulOperation::new(factor.clone()).transpose(context, output_cotangents)
                }
                LinearArrayOperation::RightMatMul { factor } => {
                    RightMatMulOperation::new(factor.clone()).transpose(context, output_cotangents)
                }
                LinearArrayOperation::Reshape { input_shape, output_shape } => {
                    ReshapeOperation::new(input_shape.clone(), output_shape.clone())
                        .transpose(context, output_cotangents)
                }
                LinearArrayOperation::Condition(_)
                | LinearArrayOperation::While(_)
                | LinearArrayOperation::Custom(_) => {
                    Err(CustomOperationError::MissingRule { op: op.name(), transform: "XLA linear transpose" }.into())
                }
            },
            Self::Condition(condition) => condition.transpose(context, output_cotangents),
            Self::While(while_operation) => while_operation.transpose(context, output_cotangents),
            Self::LinearShardMap(op) => op.transpose(context, output_cotangents),
            Self::WithShardingConstraint(op) => op.transpose(context, output_cotangents),
        }
    }
}

impl<D, V> LinearOperationCarrierFamily<D, V> for LinearXlaOperation<V>
where
    D: ryft_core::tracing::domains::TracingDomain<
            Type = ArrayType,
            Value = ShardMapTensor,
            OperationCarrier = XlaOperation,
        >,
    V: Traceable<ArrayType>,
    D::OperationCarrier: SupportsAdd<ArrayType, D::Value>
        + SupportsSub<ArrayType, D::Value>
        + SupportsNeg<ArrayType, D::Value>
        + SupportsMul<ArrayType, D::Value>
        + SupportsZeroLike<ArrayType, D::Value>
        + SupportsOneLike<ArrayType, D::Value>
        + SupportsMatMul<ArrayType, D::Value>
        + SupportsMatrixTranspose<ArrayType, D::Value>
        + SupportsReshape<ArrayType, D::Value>,
{
    type ForTracer<'domain>
        = LinearXlaOperation<Tracer<'domain, D>>
    where
        D: 'domain;
}

impl<V> SupportsAdd<ArrayType, V> for LinearXlaOperation<V>
where
    V: Traceable<ArrayType>,
{
    fn add_operation() -> Self {
        Self::Array(LinearArrayOperation::Add)
    }
}

impl<V> SupportsSub<ArrayType, V> for LinearXlaOperation<V>
where
    V: Traceable<ArrayType>,
{
    fn sub_operation() -> Self {
        Self::Array(LinearArrayOperation::Sub)
    }
}

impl<V> SupportsNeg<ArrayType, V> for LinearXlaOperation<V>
where
    V: Traceable<ArrayType>,
{
    fn neg_operation() -> Self {
        Self::Array(LinearArrayOperation::Neg)
    }
}

impl<V> SupportsZero<ArrayType, V> for LinearXlaOperation<V>
where
    V: Traceable<ArrayType>,
{
    fn zero_operation(r#type: ArrayType) -> Self {
        Self::Array(LinearArrayOperation::Zero(ZeroOperation::new(r#type)))
    }

    fn as_zero_operation(&self) -> Option<&ZeroOperation<ArrayType>> {
        match self {
            Self::Array(op) => op.as_zero_operation(),
            _ => None,
        }
    }
}

impl<V> SupportsOne<ArrayType, V> for LinearXlaOperation<V>
where
    V: Traceable<ArrayType>,
{
    fn one_operation(r#type: ArrayType) -> Self {
        Self::Array(LinearArrayOperation::One(OneOperation::new(r#type)))
    }
}

impl<V> SupportsZeroLike<ArrayType, V> for LinearXlaOperation<V>
where
    V: Traceable<ArrayType>,
{
    fn zero_like_operation() -> Self {
        Self::Array(LinearArrayOperation::ZeroLike)
    }
}

impl<V> SupportsOneLike<ArrayType, V> for LinearXlaOperation<V>
where
    V: Traceable<ArrayType>,
{
    fn one_like_operation() -> Self {
        Self::Array(LinearArrayOperation::OneLike)
    }
}

impl<V> SupportsScale<ArrayType, V> for LinearXlaOperation<V>
where
    V: Traceable<ArrayType>,
{
    fn scale_operation(factor: V) -> Self {
        Self::Array(LinearArrayOperation::Scale { factor })
    }
}

impl<V> SupportsLeftMatMul<ArrayType, V> for LinearXlaOperation<V>
where
    V: Traceable<ArrayType>,
{
    fn left_matmul_operation(factor: V) -> Self {
        Self::Array(LinearArrayOperation::LeftMatMul { factor })
    }
}

impl<V> SupportsRightMatMul<ArrayType, V> for LinearXlaOperation<V>
where
    V: Traceable<ArrayType>,
{
    fn right_matmul_operation(factor: V) -> Self {
        Self::Array(LinearArrayOperation::RightMatMul { factor })
    }
}

impl<V> SupportsMatrixTranspose<ArrayType, V> for LinearXlaOperation<V>
where
    V: Traceable<ArrayType>,
{
    fn matrix_transpose_operation() -> Self {
        Self::Array(LinearArrayOperation::Transpose)
    }
}

impl<V> SupportsReshape<ArrayType, V> for LinearXlaOperation<V>
where
    V: Traceable<ArrayType>,
{
    fn reshape_operation(input_shape: Shape, output_shape: Shape) -> Self {
        Self::Array(LinearArrayOperation::Reshape { input_shape, output_shape })
    }
}

impl<V> From<ConditionOperation<V, LinearXlaOperation<V>, ArrayType>> for LinearXlaOperation<V>
where
    V: Traceable<ArrayType>,
{
    fn from(op: ConditionOperation<V, LinearXlaOperation<V>, ArrayType>) -> Self {
        Self::Condition(Box::new(op))
    }
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
        check_count!("output", outputs, instruction.outputs.len(), TracingError);
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

/// Closed ordinary staged-op universe owned by the XLA backend.
#[allow(private_interfaces)]
#[derive(Clone, Debug)]
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

    /// Elementwise subtraction.
    Sub,

    /// Elementwise multiplication.
    Mul,

    /// Elementwise division.
    Div,

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

    /// Higher-order conditional.
    Condition(Box<ConditionOperation<ShardMapTensor, XlaOperation, ArrayType>>),

    /// Higher-order while loop.
    While(Box<WhileOperation<ShardMapTensor, XlaOperation, ArrayType>>),

    /// XLA-specific `shard_map`.
    ShardMap(Box<ShardMapOperation<ShardMapTensor>>),

    /// XLA-specific `linear_shard_map`.
    LinearShardMap(Box<LinearShardMapOperation<ShardMapTensor>>),

    /// XLA-specific sharding constraint.
    WithShardingConstraint(WithShardingConstraintOperation),

    /// Explicit escape hatch for custom XLA ops.
    Custom(Arc<CustomPrimitive<ArrayType, ShardMapTensor>>),
}

impl Display for XlaOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Reshape { output_shape, .. } => write!(formatter, "{}{output_shape}", self.name()),
            _ => write!(formatter, "{}", self.name()),
        }
    }
}

impl Operation<ArrayType> for XlaOperation {
    #[inline]
    fn name(&self) -> &'static str {
        match self {
            Self::Zero(zero) => zero.name(),
            Self::One(one) => one.name(),
            Self::ZeroLike => ZERO_LIKE_OPERATION_NAME,
            Self::OneLike => ONE_LIKE_OPERATION_NAME,
            Self::Add => ADD_OPERATION_NAME,
            Self::Sub => SUB_OPERATION_NAME,
            Self::Mul => MUL_OPERATION_NAME,
            Self::Div => DIV_OPERATION_NAME,
            Self::Neg => NEG_OPERATION_NAME,
            Self::Sin => "sin",
            Self::Cos => "cos",
            Self::MatrixMultiply => "matmul",
            Self::Transpose => "matrix_transpose",
            Self::Scale { .. } => SCALE_OPERATION_NAME,
            Self::Reshape { .. } => "reshape",
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
            Self::Sub => SubOperation.infer_output_types(input_types),
            Self::Mul => MulOperation.infer_output_types(input_types),
            Self::Div => DivOperation.infer_output_types(input_types),
            Self::Neg => NegOperation.infer_output_types(input_types),
            Self::Sin => SinOperation.infer_output_types(input_types),
            Self::Cos => CosOperation.infer_output_types(input_types),
            Self::MatrixMultiply => MatMulOperation.infer_output_types(input_types),
            Self::Transpose => MatrixTransposeOperation.infer_output_types(input_types),
            Self::Scale { factor } => ScaleOperation::new(factor.clone()).infer_output_types(input_types),
            Self::Reshape { input_shape, output_shape } => {
                ReshapeOperation::new(input_shape.clone(), output_shape.clone()).infer_output_types(input_types)
            }
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
            Self::Sub => SubOperation.interpret(inputs),
            Self::Mul => MulOperation.interpret(inputs),
            Self::Div => DivOperation.interpret(inputs),
            Self::Neg => NegOperation.interpret(inputs),
            Self::Sin => SinOperation.interpret(inputs),
            Self::Cos => CosOperation.interpret(inputs),
            Self::MatrixMultiply => MatMulOperation.interpret(inputs),
            Self::Transpose => MatrixTransposeOperation.interpret(inputs),
            Self::Scale { factor } => ScaleOperation::new(factor.clone()).interpret(inputs),
            Self::Reshape { input_shape, output_shape } => {
                ReshapeOperation::new(input_shape.clone(), output_shape.clone()).interpret(inputs)
            }
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
                    &zero.r#type
                ),
            }
            .into()),
            Self::One(one) => Err(TypeError {
                message: format!(
                    "typed one operation over tracer values was not materialized before interpretation for {}",
                    &one.r#type
                ),
            }
            .into()),
            Self::ZeroLike => ZeroLikeOperation.interpret(inputs),
            Self::OneLike => OneLikeOperation.interpret(inputs),
            Self::Add => AddOperation.interpret(inputs),
            Self::Sub => SubOperation.interpret(inputs),
            Self::Mul => MulOperation.interpret(inputs),
            Self::Div => DivOperation.interpret(inputs),
            Self::Neg => NegOperation.interpret(inputs),
            Self::Sin => SinOperation.interpret(inputs),
            Self::Cos => CosOperation.interpret(inputs),
            Self::MatrixMultiply => MatMulOperation.interpret(inputs),
            Self::Transpose => MatrixTransposeOperation.interpret(inputs),
            Self::Scale { factor } => {
                check_count!("input", inputs, 1, TracingError);
                let factor = inputs[0].context.constant(factor.clone());
                Ok(vec![factor * inputs[0].clone()])
            }
            Self::Reshape { input_shape, output_shape } => {
                ReshapeOperation::new(input_shape.clone(), output_shape.clone()).interpret(inputs)
            }
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

impl<'c> DifferentiableOperation<XlaDomain<'c>> for XlaOperation {
    fn jvp<'jvp>(
        &self,
        context: &mut JvpContext<'jvp, XlaDomain<'c>>,
        inputs: &[JvpTracer<ShardMapTensor, Tracer<'jvp, LinearXlaDomain>>],
    ) -> Result<Vec<JvpTracer<ShardMapTensor, Tracer<'jvp, LinearXlaDomain>>>, TracingError> {
        match self {
            Self::Zero(zero) => zero.jvp(context, inputs),
            Self::One(one) => one.jvp(context, inputs),
            Self::ZeroLike => ZeroLikeOperation.jvp(context, inputs),
            Self::OneLike => OneLikeOperation.jvp(context, inputs),
            Self::Add => AddOperation.jvp(context, inputs),
            Self::Sub => SubOperation.jvp(context, inputs),
            Self::Mul => MulOperation.jvp(context, inputs),
            Self::Div => DivOperation.jvp(context, inputs),
            Self::Neg => NegOperation.jvp(context, inputs),
            Self::Sin => SinOperation.jvp(context, inputs),
            Self::Cos => CosOperation.jvp(context, inputs),
            Self::MatrixMultiply => MatMulOperation.jvp(context, inputs),
            Self::Transpose => MatrixTransposeOperation.jvp(context, inputs),
            Self::Scale { factor } => ScaleOperation::new(factor.clone()).jvp(context, inputs),
            Self::Reshape { input_shape, output_shape } => {
                ReshapeOperation::new(input_shape.clone(), output_shape.clone()).jvp(context, inputs)
            }
            Self::Condition(condition) => condition.jvp(context, inputs),
            Self::While(while_operation) => while_operation.jvp(context, inputs),
            Self::ShardMap(op) => op.jvp(context, inputs),
            Self::LinearShardMap(op) => op.jvp(context, inputs),
            Self::WithShardingConstraint(op) => op.jvp(context, inputs),
            Self::Custom(op) => {
                Err(CustomOperationError::MissingRule { op: op.name(), transform: "concrete linearization" }.into())
            }
        }
    }
}

impl DifferentiableOperation<TracingContext<'static, XlaDomain<'static>>> for XlaOperation {
    fn jvp<'jvp>(
        &self,
        context: &mut JvpContext<'jvp, TracingContext<'static, XlaDomain<'static>>>,
        inputs: &[JvpTracer<ShardMapTracer, Tracer<'jvp, TracingContext<'static, XlaDomain<'static>>>>],
    ) -> Result<Vec<JvpTracer<ShardMapTracer, Tracer<'jvp, TracingContext<'static, XlaDomain<'static>>>>>, TracingError>
    {
        match self {
            Self::Zero(zero) => zero.jvp(context, inputs),
            Self::One(one) => one.jvp(context, inputs),
            Self::ZeroLike => ZeroLikeOperation.jvp(context, inputs),
            Self::OneLike => OneLikeOperation.jvp(context, inputs),
            Self::Add => AddOperation.jvp(context, inputs),
            Self::Sub => SubOperation.jvp(context, inputs),
            Self::Mul => MulOperation.jvp(context, inputs),
            Self::Div => DivOperation.jvp(context, inputs),
            Self::Neg => NegOperation.jvp(context, inputs),
            Self::Sin => SinOperation.jvp(context, inputs),
            Self::Cos => CosOperation.jvp(context, inputs),
            Self::MatrixMultiply => MatMulOperation.jvp(context, inputs),
            Self::Transpose => MatrixTransposeOperation.jvp(context, inputs),
            Self::Scale { factor } => ScaleOperation::new(factor.clone()).jvp(context, inputs),
            Self::Reshape { input_shape, output_shape } => {
                ReshapeOperation::new(input_shape.clone(), output_shape.clone()).jvp(context, inputs)
            }
            Self::Condition(condition) => condition.jvp(context, inputs),
            Self::While(while_operation) => while_operation.jvp(context, inputs),
            Self::ShardMap(op) => {
                let traced_op = ShardMapOperation::<ShardMapTracer>::new(op.body.clone());
                traced_op.jvp_with_builders(context.domain.builder.clone(), context, inputs)
            }
            Self::LinearShardMap(op) => op.jvp_traced_with_builders(context.domain.builder.clone(), context, inputs),
            Self::WithShardingConstraint(op) => {
                check_count!("input", inputs, 1, TracingError);
                let input = &inputs[0];
                let primal_outputs =
                    input.primal.context.trace(XlaOperation::WithShardingConstraint(op.clone()), &[&input.primal])?;
                check_count!("output", primal_outputs, 1, TracingError);
                let mut tangent_outputs =
                    context.stage(LinearXlaOperation::WithShardingConstraint(op.clone()), &[input.tangent.clone()])?;
                check_count!("output", tangent_outputs, 1, TracingError);
                Ok(vec![JvpTracer { primal: primal_outputs[0].clone(), tangent: tangent_outputs.remove(0) }])
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

impl SupportsSub<ArrayType, ShardMapTensor> for XlaOperation {
    fn sub_operation() -> Self {
        XlaOperation::Sub
    }
}

impl SupportsMul<ArrayType, ShardMapTensor> for XlaOperation {
    fn mul_operation() -> Self {
        XlaOperation::Mul
    }
}

impl SupportsDiv<ArrayType, ShardMapTensor> for XlaOperation {
    fn div_operation() -> Self {
        XlaOperation::Div
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
    use ryft_core::tracing::{ProgramBuilder, TracingContext};
    use ryft_core::types::{DataType, Typed};

    use crate::experimental::domains::XlaDomain;

    use super::*;

    fn scalar_type() -> ArrayType {
        ArrayType::scalar(DataType::F32)
    }

    fn test_mesh() -> LogicalMesh {
        LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap()
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
            TracingContext::new(XlaDomain::token(), tracing_builder).tracer(traced_input_atom, Some(scalar_type()));

        let outputs = replay_xla_program_with_tracers(&program, vec![traced_input]).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].r#type().into_owned(), scalar_type());
    }
}
