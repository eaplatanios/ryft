//! Backend-owned staged op universe for traced XLA programs.

use std::{
    fmt::{Debug, Display},
    sync::Arc,
};

use ryft_core::{
    tracing_v2::{
        CustomPrimitive, DifferentiableOp, InterpretableOp, LinearPrimitiveOp, LinearTerm, Op, OperationSet,
        SupportsAdd, SupportsCos, SupportsCustom, SupportsLeftMatMul, SupportsLinearAdd, SupportsLinearCustom,
        SupportsLinearLeftMatMul, SupportsLinearMatrixTranspose, SupportsLinearNeg, SupportsLinearReshape,
        SupportsLinearRightMatMul, SupportsLinearScale, SupportsMatMul, SupportsMatrixTranspose, SupportsMul,
        SupportsNeg, SupportsRematerialize, SupportsReshape, SupportsRightMatMul, SupportsScale, SupportsSin,
        SupportsVMap, TraceError,
        engine::Engine,
        forward::JvpTracer,
        linear::Linearized,
        operations::{
            AddOp, CosOp, LeftMatMulOp, MatMulOp, MatrixTransposeOp, MulOp, NegOp, RematerializeOp, ReshapeOp,
            RightMatMulOp, ScaleOp, SinOp, VMapOp, left_matmul::left_matmul_abstract_eval,
            right_matmul::right_matmul_abstract_eval,
        },
    },
    types::{ArrayType, Typed},
};

use crate::experimental::{
    operations::{ShardMapOp, WithShardingConstraintOp},
    shard_map::{ShardMapTensor, ShardMapTracer},
};

/// Closed ordinary staged-op universe owned by the XLA backend.
#[allow(private_interfaces)]
#[derive(Clone)]
pub enum XlaPrimitiveOp {
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
    VMap(Box<VMapOp<ArrayType, ShardMapTensor, XlaOperationSet>>),

    /// Higher-order rematerialization.
    Rematerialize(Box<RematerializeOp<ArrayType, ShardMapTensor, XlaOperationSet>>),

    /// XLA-specific `shard_map`.
    ShardMap(Box<ShardMapOp<ShardMapTensor>>),

    /// XLA-specific sharding constraint.
    WithShardingConstraint(WithShardingConstraintOp),

    /// Explicit escape hatch for custom XLA ops.
    Custom(Arc<CustomPrimitive<ArrayType, ShardMapTensor>>),
}

/// Op-set marker selecting [`XlaPrimitiveOp`] for ordinary traced XLA programs.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct XlaOperationSet;

impl Debug for XlaPrimitiveOp {
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

impl Display for XlaPrimitiveOp {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Reshape { output_type, .. } => write!(formatter, "reshape{}", output_type.shape),
            _ => write!(formatter, "{}", self.name()),
        }
    }
}

impl Op for XlaPrimitiveOp {
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

    fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TraceError> {
        match self {
            Self::Add => AddOp.abstract_eval(inputs),
            Self::Mul => MulOp.abstract_eval(inputs),
            Self::Neg => NegOp.abstract_eval(inputs),
            Self::Sin => SinOp.abstract_eval(inputs),
            Self::Cos => CosOp.abstract_eval(inputs),
            Self::MatMul => MatMulOp.abstract_eval(inputs),
            Self::MatrixTranspose => MatrixTransposeOp.abstract_eval(inputs),
            Self::Scale { .. } => ScaleOp::<ArrayType, ShardMapTensor>::abstract_eval_static(inputs),
            Self::LeftMatMul { factor } => left_matmul_abstract_eval(&Typed::tpe(factor), inputs),
            Self::RightMatMul { factor } => right_matmul_abstract_eval(&Typed::tpe(factor), inputs),
            Self::Reshape { input_type, output_type } => {
                ReshapeOp::new(input_type.clone(), output_type.clone()).abstract_eval(inputs)
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
        inputs: &[usize],
        is_zero_constant: &dyn Fn(usize) -> bool,
        is_one_constant: &dyn Fn(usize) -> bool,
    ) -> Option<Vec<usize>> {
        match self {
            Self::Add => AddOp.try_simplify(inputs, is_zero_constant, is_one_constant),
            Self::Mul => MulOp.try_simplify(inputs, is_zero_constant, is_one_constant),
            Self::Neg => NegOp.try_simplify(inputs, is_zero_constant, is_one_constant),
            Self::Scale { factor } => ScaleOp::<ArrayType, ShardMapTensor>::new(factor.clone()).try_simplify(
                inputs,
                is_zero_constant,
                is_one_constant,
            ),
            Self::Custom(op) => op.try_simplify(inputs, is_zero_constant, is_one_constant),
            _ => None,
        }
    }
}

impl InterpretableOp<ArrayType, ShardMapTensor> for XlaPrimitiveOp {
    fn interpret(&self, inputs: &[ShardMapTensor]) -> Result<Vec<ShardMapTensor>, TraceError> {
        match self {
            Self::Add => AddOp.interpret(inputs),
            Self::Mul => MulOp.interpret(inputs),
            Self::Neg => NegOp.interpret(inputs),
            Self::Sin => SinOp.interpret(inputs),
            Self::Cos => CosOp.interpret(inputs),
            Self::MatMul => MatMulOp.interpret(inputs),
            Self::MatrixTranspose => MatrixTransposeOp.interpret(inputs),
            Self::Scale { factor } => ScaleOp::new(factor.clone()).interpret(inputs),
            Self::LeftMatMul { factor } => LeftMatMulOp::new(factor.clone()).interpret(inputs),
            Self::RightMatMul { factor } => RightMatMulOp::new(factor.clone()).interpret(inputs),
            Self::Reshape { input_type, output_type } => {
                ReshapeOp::new(input_type.clone(), output_type.clone()).interpret(inputs)
            }
            Self::VMap(vmap) => vmap.interpret(inputs),
            Self::Rematerialize(remat) => remat.interpret(inputs),
            Self::ShardMap(op) => op.interpret(inputs),
            Self::WithShardingConstraint(op) => op.interpret(inputs),
            Self::Custom(op) => op.interpret(inputs),
        }
    }
}

impl DifferentiableOp<ArrayType, ShardMapTensor, LinearTerm<ArrayType, ShardMapTensor>, XlaOperationSet>
    for XlaPrimitiveOp
{
    fn jvp(
        &self,
        engine: &dyn Engine<Type = ArrayType, Value = ShardMapTensor, OperationSet = XlaOperationSet>,
        inputs: &[JvpTracer<ShardMapTensor, LinearTerm<ArrayType, ShardMapTensor>>],
    ) -> Result<Vec<JvpTracer<ShardMapTensor, LinearTerm<ArrayType, ShardMapTensor>>>, TraceError> {
        match self {
            Self::Add => AddOp.jvp(engine, inputs),
            Self::Mul => MulOp.jvp(engine, inputs),
            Self::Neg => NegOp.jvp(engine, inputs),
            Self::Sin => SinOp.jvp(engine, inputs),
            Self::Cos => CosOp.jvp(engine, inputs),
            Self::MatMul => MatMulOp.jvp(engine, inputs),
            Self::MatrixTranspose => MatrixTransposeOp.jvp(engine, inputs),
            Self::Scale { factor } => ScaleOp::new(factor.clone()).jvp(engine, inputs),
            Self::LeftMatMul { factor } => LeftMatMulOp::new(factor.clone()).jvp(engine, inputs),
            Self::RightMatMul { factor } => RightMatMulOp::new(factor.clone()).jvp(engine, inputs),
            Self::Reshape { input_type, output_type } => {
                ReshapeOp::new(input_type.clone(), output_type.clone()).jvp(engine, inputs)
            }
            Self::VMap(vmap) => vmap.jvp(engine, inputs),
            Self::Rematerialize(remat) => remat.jvp(engine, inputs),
            Self::ShardMap(op) => op.jvp(engine, inputs),
            Self::WithShardingConstraint(op) => op.jvp(engine, inputs),
            Self::Custom(op) => op.jvp(engine, inputs),
        }
    }
}

impl InterpretableOp<ArrayType, Linearized<ShardMapTracer>> for XlaPrimitiveOp {
    fn interpret(&self, inputs: &[Linearized<ShardMapTracer>]) -> Result<Vec<Linearized<ShardMapTracer>>, TraceError> {
        match self {
            Self::Add => AddOp.interpret(inputs),
            Self::Mul => MulOp.interpret(inputs),
            Self::Neg => NegOp.interpret(inputs),
            Self::Sin => SinOp.interpret(inputs),
            Self::Cos => CosOp.interpret(inputs),
            Self::MatMul => MatMulOp.interpret(inputs),
            Self::MatrixTranspose => MatrixTransposeOp.interpret(inputs),
            Self::Scale { factor } => ScaleOp::new(factor.clone()).interpret(inputs),
            Self::LeftMatMul { factor } => LeftMatMulOp::new(factor.clone()).interpret(inputs),
            Self::RightMatMul { factor } => RightMatMulOp::new(factor.clone()).interpret(inputs),
            Self::Reshape { input_type, output_type } => {
                ReshapeOp::new(input_type.clone(), output_type.clone()).interpret(inputs)
            }
            Self::VMap(vmap) => vmap.interpret(inputs),
            Self::Rematerialize(remat) => remat.interpret(inputs),
            Self::ShardMap(op) => op.interpret(inputs),
            Self::WithShardingConstraint(op) => op.interpret(inputs),
            Self::Custom(_) => Err(TraceError::HigherOrderOpFailure {
                op: "eval_linearized_jit",
                message: "linearized JIT replay for custom XLA ops is not supported".to_string(),
            }),
        }
    }
}

impl OperationSet<ArrayType, ShardMapTensor> for XlaOperationSet {
    type TracingOperation = XlaPrimitiveOp;
    type LinearOperation = LinearPrimitiveOp<ArrayType, ShardMapTensor>;
}

impl SupportsAdd<ArrayType, ShardMapTensor> for XlaOperationSet {
    fn add_op() -> Self::TracingOperation {
        XlaPrimitiveOp::Add
    }
}

impl SupportsMul<ArrayType, ShardMapTensor> for XlaOperationSet {
    fn mul_op() -> Self::TracingOperation {
        XlaPrimitiveOp::Mul
    }
}

impl SupportsNeg<ArrayType, ShardMapTensor> for XlaOperationSet {
    fn neg_op() -> Self::TracingOperation {
        XlaPrimitiveOp::Neg
    }
}

impl SupportsSin<ArrayType, ShardMapTensor> for XlaOperationSet {
    fn sin_op() -> Self::TracingOperation {
        XlaPrimitiveOp::Sin
    }
}

impl SupportsCos<ArrayType, ShardMapTensor> for XlaOperationSet {
    fn cos_op() -> Self::TracingOperation {
        XlaPrimitiveOp::Cos
    }
}

impl SupportsMatMul<ArrayType, ShardMapTensor> for XlaOperationSet {
    fn matmul_op() -> Self::TracingOperation {
        XlaPrimitiveOp::MatMul
    }
}

impl SupportsMatrixTranspose<ArrayType, ShardMapTensor> for XlaOperationSet {
    fn matrix_transpose_op() -> Self::TracingOperation {
        XlaPrimitiveOp::MatrixTranspose
    }
}

impl SupportsCustom<ArrayType, ShardMapTensor> for XlaOperationSet {
    fn custom_op(primitive: Arc<CustomPrimitive<ArrayType, ShardMapTensor>>) -> Self::TracingOperation {
        XlaPrimitiveOp::Custom(primitive)
    }
}

impl SupportsVMap<ArrayType, ShardMapTensor> for XlaOperationSet {
    fn vmap_op(op: VMapOp<ArrayType, ShardMapTensor, Self>) -> Self::TracingOperation {
        XlaPrimitiveOp::VMap(Box::new(op))
    }
}

impl SupportsRematerialize<ArrayType, ShardMapTensor> for XlaOperationSet {
    fn rematerialize_op(op: RematerializeOp<ArrayType, ShardMapTensor, Self>) -> Self::TracingOperation {
        XlaPrimitiveOp::Rematerialize(Box::new(op))
    }
}

impl SupportsScale<ArrayType, ShardMapTensor> for XlaOperationSet {
    fn scale_op(factor: ShardMapTensor) -> Self::TracingOperation {
        XlaPrimitiveOp::Scale { factor }
    }
}

impl SupportsLeftMatMul<ArrayType, ShardMapTensor> for XlaOperationSet {
    fn left_matmul_op(factor: ShardMapTensor) -> Self::TracingOperation {
        XlaPrimitiveOp::LeftMatMul { factor }
    }
}

impl SupportsRightMatMul<ArrayType, ShardMapTensor> for XlaOperationSet {
    fn right_matmul_op(factor: ShardMapTensor) -> Self::TracingOperation {
        XlaPrimitiveOp::RightMatMul { factor }
    }
}

impl SupportsReshape<ArrayType, ShardMapTensor> for XlaOperationSet {
    fn reshape_op(input_type: ArrayType, output_type: ArrayType) -> Self::TracingOperation {
        XlaPrimitiveOp::Reshape { input_type, output_type }
    }
}

impl SupportsLinearAdd<ArrayType, ShardMapTensor> for XlaOperationSet {
    fn linear_add_op() -> Self::LinearOperation {
        LinearPrimitiveOp::Add
    }
}

impl SupportsLinearNeg<ArrayType, ShardMapTensor> for XlaOperationSet {
    fn linear_neg_op() -> Self::LinearOperation {
        LinearPrimitiveOp::Neg
    }
}

impl SupportsLinearMatrixTranspose<ArrayType, ShardMapTensor> for XlaOperationSet {
    fn linear_matrix_transpose_op() -> Self::LinearOperation {
        LinearPrimitiveOp::LinearMatrixTranspose
    }
}

impl SupportsLinearScale<ArrayType, ShardMapTensor> for XlaOperationSet {
    fn linear_scale_op(factor: ShardMapTensor) -> Self::LinearOperation {
        LinearPrimitiveOp::Scale { factor }
    }
}

impl SupportsLinearLeftMatMul<ArrayType, ShardMapTensor> for XlaOperationSet {
    fn linear_left_matmul_op(factor: ShardMapTensor) -> Self::LinearOperation {
        LinearPrimitiveOp::LeftMatMul { factor }
    }
}

impl SupportsLinearRightMatMul<ArrayType, ShardMapTensor> for XlaOperationSet {
    fn linear_right_matmul_op(factor: ShardMapTensor) -> Self::LinearOperation {
        LinearPrimitiveOp::RightMatMul { factor }
    }
}

impl SupportsLinearReshape<ArrayType, ShardMapTensor> for XlaOperationSet {
    fn linear_reshape_op(input_type: ArrayType, output_type: ArrayType) -> Self::LinearOperation {
        LinearPrimitiveOp::Reshape { input_type, output_type }
    }
}

impl SupportsLinearCustom<ArrayType, ShardMapTensor> for XlaOperationSet {
    fn linear_custom_op(
        primitive: CustomPrimitive<ArrayType, ShardMapTensor>,
    ) -> Result<Self::LinearOperation, TraceError> {
        LinearPrimitiveOp::custom(primitive)
    }

    fn linear_custom_arc_op(
        primitive: Arc<CustomPrimitive<ArrayType, ShardMapTensor>>,
    ) -> Result<Self::LinearOperation, TraceError> {
        LinearPrimitiveOp::custom_arc(primitive)
    }
}
