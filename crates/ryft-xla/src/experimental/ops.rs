//! Backend-owned staged op universe for traced XLA programs.

use std::{
    fmt::{Debug, Display},
    sync::Arc,
};

use ryft_core::{
    tracing_v2::{
        AtomId, CustomPrimitive, DifferentiableOperation, InterpretableOperation, LinearPrimitiveOperation, LinearTerm,
        Operation, TracingError,
        engine::Engine,
        forward::JvpTracer,
        linear::Linearized,
        operations::{
            AddOperation, AddTracingOperation, CosOperation, CosTracingOperation, CustomTracingOperation,
            LeftMatMulOperation, LeftMatMulTracingOperation, MatMulOperation, MatMulTracingOperation,
            MatrixTransposeOperation, MatrixTransposeTracingOperation, MulOperation, MulTracingOperation, NegOperation,
            NegTracingOperation, RematerializeOperation, RematerializeTracingOperation, ReshapeOperation,
            ReshapeTracingOperation, RightMatMulOperation, RightMatMulTracingOperation, ScaleOperation,
            ScaleTracingOperation, SinOperation, SinTracingOperation, VMapOperation, VMapTracingOperation,
            left_matmul::left_matmul_abstract_eval, right_matmul::right_matmul_abstract_eval,
        },
    },
    types::{ArrayType, Typed},
};

use crate::experimental::{
    operations::{ShardMapOperation, WithShardingConstraintOperation},
    shard_map::{ShardMapTensor, ShardMapTracer},
};

type XlaLinearOperation = LinearPrimitiveOperation<ArrayType, ShardMapTensor>;

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

    fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TracingError> {
        match self {
            Self::Add => AddOperation.abstract_eval(inputs),
            Self::Mul => MulOperation.abstract_eval(inputs),
            Self::Neg => NegOperation.abstract_eval(inputs),
            Self::Sin => SinOperation.abstract_eval(inputs),
            Self::Cos => CosOperation.abstract_eval(inputs),
            Self::MatMul => MatMulOperation.abstract_eval(inputs),
            Self::MatrixTranspose => MatrixTransposeOperation.abstract_eval(inputs),
            Self::Scale { .. } => ScaleOperation::<ArrayType, ShardMapTensor>::abstract_eval_static(inputs),
            Self::LeftMatMul { factor } => left_matmul_abstract_eval(&Typed::r#type(factor), inputs),
            Self::RightMatMul { factor } => right_matmul_abstract_eval(&Typed::r#type(factor), inputs),
            Self::Reshape { input_type, output_type } => {
                ReshapeOperation::new(input_type.clone(), output_type.clone()).abstract_eval(inputs)
            }
            Self::VMap(vmap) => vmap.abstract_eval(inputs),
            Self::Rematerialize(remat) => remat.abstract_eval(inputs),
            Self::ShardMap(op) => op.abstract_eval(inputs),
            Self::WithShardingConstraint(op) => op.abstract_eval(inputs),
            Self::Custom(op) => op.abstract_eval(inputs),
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
            Self::VMap(vmap) => Err(TracingError::HigherOrderOpFailure {
                op: "linearize_program",
                message: format!("JVP rule for staged op '{}' is not implemented", vmap.name()),
            }),
            Self::Rematerialize(remat) => Err(TracingError::HigherOrderOpFailure {
                op: "linearize_program",
                message: format!("JVP rule for staged op '{}' is not implemented", remat.name()),
            }),
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
            Self::Custom(_) => Err(TracingError::HigherOrderOpFailure {
                op: "eval_linearized_jit",
                message: "linearized JIT replay for custom XLA ops is not supported".to_string(),
            }),
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
