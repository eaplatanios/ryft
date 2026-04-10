//! Primitive operation traits for `tracing_v2`.
//!
//! The staged op set is intentionally open: each primitive is represented by its own concrete type implementing one
//! or more transform-specific traits. This module keeps only the operation-neutral dispatch interfaces.

use std::{
    any::Any,
    fmt::{Debug, Display},
    sync::Arc,
};

use crate::tracing_v2::{
    FloatExt, MatrixOps, OneLike, TraceError, TraceValue, TransformLeaf, ZeroLike,
    batch::Batch,
    forward::{JvpTracer, TangentSpace},
    graph::AtomId,
    jit::JitTracer,
    linear::LinearTerm,
    program::ProgramBuilder,
};
use crate::types::ArrayType;

/// Core primitive operation interface understood by staged graphs.
///
/// This trait covers the minimum surface required for JIT tracing: abstract evaluation for shape propagation and
/// concrete evaluation for eager execution. Operations that additionally support differentiation should implement
/// [`DifferentiableOp`].
pub trait Op<V: TraceValue>: Debug + Display {
    /// Returns this operation as [`Any`] for downcasting.
    fn as_any(&self) -> &dyn Any;

    /// Returns the stable primitive name used in diagnostics and pretty-printing.
    fn name(&self) -> &'static str;

    /// Computes abstract outputs from abstract inputs without executing the operation.
    fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TraceError>;

    /// Executes the operation on concrete values.
    fn eval(&self, inputs: &[V]) -> Result<Vec<V>, TraceError>;
}

/// Extension of [`Op`] for operations that participate in program-level differentiation.
///
/// Operations implementing this trait provide rules for forward-mode linearization ([`apply_program_jvp_rule`]),
/// reverse-mode transposition ([`transpose_program_op`]), and higher-order JIT replay
/// ([`replay_linearized_jit`]). Default implementations return [`TraceError::HigherOrderOpFailure`] so that
/// operations only need to override the methods relevant to their transform support.
///
/// [`apply_program_jvp_rule`]: DifferentiableOp::apply_program_jvp_rule
/// [`transpose_program_op`]: DifferentiableOp::transpose_program_op
/// [`replay_linearized_jit`]: DifferentiableOp::replay_linearized_jit
pub trait DifferentiableOp<V: TraceValue>: Op<V> {
    /// Replays this staged op while tracing a linearized JIT program.
    fn replay_linearized_jit(
        &self,
        _inputs: Vec<JvpTracer<JitTracer<V>, LinearTerm<JitTracer<V>>>>,
    ) -> Result<Vec<JvpTracer<JitTracer<V>, LinearTerm<JitTracer<V>>>>, TraceError>
    where
        V: TransformLeaf,
    {
        Err(TraceError::HigherOrderOpFailure {
            op: "replay_program_graph",
            message: format!("replaying linearized values through staged op '{}' is not implemented", self.name()),
        })
    }

    /// Applies this op's program-level JVP rule while linearizing a staged program.
    fn apply_program_jvp_rule(
        &self,
        _inputs: &[JvpTracer<V, LinearTerm<V>>],
    ) -> Result<Vec<JvpTracer<V, LinearTerm<V>>>, TraceError>
    where
        V: FloatExt + ZeroLike + OneLike + MatrixOps + operations::reshape::ReshapeOps,
    {
        Err(TraceError::HigherOrderOpFailure {
            op: "linearize_program",
            message: format!("JVP rule for staged op '{}' is not implemented", self.name()),
        })
    }

    /// Applies this op's transpose rule while transposing a linearized staged program.
    fn transpose_program_op(
        &self,
        _builder: &mut ProgramBuilder<V>,
        _inputs: &[AtomId],
        _outputs: &[AtomId],
        _output_cotangents: &[AtomId],
    ) -> Result<Vec<Option<AtomId>>, TraceError>
    where
        V: FloatExt + ZeroLike + OneLike + MatrixOps + operations::reshape::ReshapeOps,
    {
        Err(TraceError::HigherOrderOpFailure {
            op: "transpose_linear_program",
            message: format!("transpose rule for staged op '{}' is not implemented", self.name()),
        })
    }
}

/// Closed set of built-in staged operations.
///
/// Every known primitive is a zero-cost enum variant. Operations originating outside
/// `ryft-core` (e.g., shard-map ops in `ryft-xla`) go through the [`Custom`](PrimitiveOp::Custom) escape
/// hatch, which still uses dynamic dispatch.
#[derive(Clone)]
pub enum PrimitiveOp<V: TraceValue> {
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

    /// Matrix transposition.
    MatrixTranspose,

    /// Linear matrix transposition used in cotangent programs.
    LinearMatrixTranspose,

    /// Scalar or tensor scaling by a captured factor.
    Scale { factor: V },

    /// Left matrix multiplication by a captured factor: `factor @ input`.
    LeftMatMul { factor: V },

    /// Right matrix multiplication by a captured factor: `input @ factor`.
    RightMatMul { factor: V },

    /// Reshape between two statically known shapes.
    Reshape {
        input_type: ArrayType,
        output_type: ArrayType,
    },

    /// Higher-order `vmap` carrying a compiled per-lane body and optional transpose body.
    VMap(Box<crate::tracing_v2::operations::VMapOp<V>>),

    /// Escape hatch for user- or crate-defined operations outside `ryft-core`.
    Custom(Arc<dyn DifferentiableOp<V>>),
}

/// Canonical operation type used by the staged program IR.
pub type PrimitiveOpRef<V> = PrimitiveOp<V>;

/// Shared reference to a dynamically dispatched staged operation that supports differentiation.
///
/// NOTE: This alias is kept for backward compatibility with code that still wraps ops in `Arc`.
/// New code should prefer [`PrimitiveOp`] directly.
pub type StagedOpRef<V> = Arc<dyn DifferentiableOp<V>>;

/// Primitive operation with a forward-mode differentiation rule.
pub(crate) trait JvpOp<V: TraceValue>: Op<V> {
    /// Applies the primitive's forward-mode rule to traced inputs.
    fn jvp<T>(&self, inputs: &[JvpTracer<V, T>]) -> Result<Vec<JvpTracer<V, T>>, TraceError>
    where
        T: TangentSpace<V>;
}

/// Primitive operation with a batching rule used by `vmap`.
pub(crate) trait BatchOp<V: TraceValue>: Op<V> {
    /// Applies the primitive's batching rule to batched inputs.
    fn batch(&self, inputs: &[Batch<V>]) -> Result<Vec<Batch<V>>, TraceError>;
}

impl<T: Op<V> + ?Sized, V: TraceValue> Op<V> for Arc<T> {
    #[inline]
    fn as_any(&self) -> &dyn Any {
        (**self).as_any()
    }

    #[inline]
    fn name(&self) -> &'static str {
        (**self).name()
    }

    #[inline]
    fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TraceError> {
        (**self).abstract_eval(inputs)
    }

    #[inline]
    fn eval(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        (**self).eval(inputs)
    }
}

impl<T: DifferentiableOp<V> + ?Sized, V: TraceValue> DifferentiableOp<V> for Arc<T> {
    #[inline]
    fn replay_linearized_jit(
        &self,
        inputs: Vec<JvpTracer<JitTracer<V>, LinearTerm<JitTracer<V>>>>,
    ) -> Result<Vec<JvpTracer<JitTracer<V>, LinearTerm<JitTracer<V>>>>, TraceError>
    where
        V: TransformLeaf,
    {
        (**self).replay_linearized_jit(inputs)
    }

    #[inline]
    fn apply_program_jvp_rule(
        &self,
        inputs: &[JvpTracer<V, LinearTerm<V>>],
    ) -> Result<Vec<JvpTracer<V, LinearTerm<V>>>, TraceError>
    where
        V: FloatExt + ZeroLike + OneLike + MatrixOps + operations::reshape::ReshapeOps,
    {
        (**self).apply_program_jvp_rule(inputs)
    }

    #[inline]
    fn transpose_program_op(
        &self,
        builder: &mut ProgramBuilder<V>,
        inputs: &[AtomId],
        outputs: &[AtomId],
        output_cotangents: &[AtomId],
    ) -> Result<Vec<Option<AtomId>>, TraceError>
    where
        V: FloatExt + ZeroLike + OneLike + MatrixOps + operations::reshape::ReshapeOps,
    {
        (**self).transpose_program_op(builder, inputs, outputs, output_cotangents)
    }
}

impl<T: JvpOp<V> + ?Sized, V: TraceValue> JvpOp<V> for Arc<T> {
    #[inline]
    fn jvp<U>(&self, inputs: &[JvpTracer<V, U>]) -> Result<Vec<JvpTracer<V, U>>, TraceError>
    where
        U: TangentSpace<V>,
    {
        (**self).jvp(inputs)
    }
}

impl<T: BatchOp<V> + ?Sized, V: TraceValue> BatchOp<V> for Arc<T> {
    #[inline]
    fn batch(&self, inputs: &[Batch<V>]) -> Result<Vec<Batch<V>>, TraceError> {
        (**self).batch(inputs)
    }
}

// ---------------------------------------------------------------------------
// PrimitiveOp — Debug, Display, Op, DifferentiableOp, JvpOp, BatchOp impls
// ---------------------------------------------------------------------------

use crate::tracing_v2::operations::{
    self,
    AddOp, CosOp, LeftMatMulOp, LinearMatrixTransposeOp, MatMulOp, MatrixTransposeOp, MulOp, NegOp, ReshapeOp,
    RightMatMulOp, ScaleOp, SinOp,
};

impl<V: TraceValue> Debug for PrimitiveOp<V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Add => write!(formatter, "Add"),
            Self::Mul => write!(formatter, "Mul"),
            Self::Neg => write!(formatter, "Neg"),
            Self::Sin => write!(formatter, "Sin"),
            Self::Cos => write!(formatter, "Cos"),
            Self::MatMul => write!(formatter, "MatMul"),
            Self::MatrixTranspose => write!(formatter, "MatrixTranspose"),
            Self::LinearMatrixTranspose => write!(formatter, "LinearMatrixTranspose"),
            Self::Scale { .. } => write!(formatter, "Scale"),
            Self::LeftMatMul { .. } => write!(formatter, "LeftMatMul"),
            Self::RightMatMul { .. } => write!(formatter, "RightMatMul"),
            Self::Reshape { input_type, output_type } => {
                write!(formatter, "Reshape({input_type} -> {output_type})")
            }
            Self::VMap(vmap) => Debug::fmt(vmap, formatter),
            Self::Custom(op) => Debug::fmt(op.as_ref(), formatter),
        }
    }
}

impl<V: TraceValue> Display for PrimitiveOp<V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Reshape { output_type, .. } => write!(formatter, "reshape{}", output_type.shape),
            _ => write!(formatter, "{}", self.name()),
        }
    }
}

impl<V: TraceValue> PrimitiveOp<V> {
    /// Returns the stable primitive name for this operation.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Mul => "mul",
            Self::Neg => "neg",
            Self::Sin => "sin",
            Self::Cos => "cos",
            Self::MatMul => "matmul",
            Self::MatrixTranspose => "matrix_transpose",
            Self::LinearMatrixTranspose => "linear_matrix_transpose",
            Self::Scale { .. } => "scale",
            Self::LeftMatMul { .. } => "left_matmul",
            Self::RightMatMul { .. } => "right_matmul",
            Self::Reshape { .. } => "reshape",
            Self::VMap(_) => "vmap",
            Self::Custom(op) => op.name(),
        }
    }
}

impl<V: TraceValue + FloatExt + ZeroLike + OneLike + MatrixOps + operations::reshape::ReshapeOps> Op<V>
    for PrimitiveOp<V>
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &'static str {
        PrimitiveOp::name(self)
    }

    fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TraceError> {
        match self {
            Self::Add => Ok(vec![operations::binary_same_abstract("add", inputs)?]),
            Self::Mul => Ok(vec![operations::binary_same_abstract("mul", inputs)?]),
            Self::Neg => Ok(vec![operations::unary_abstract(inputs)?]),
            Self::Sin => Ok(vec![operations::unary_abstract(inputs)?]),
            Self::Cos => Ok(vec![operations::unary_abstract(inputs)?]),
            Self::MatMul => {
                operations::expect_input_count(inputs.len(), 2)?;
                Ok(vec![operations::matrix::matmul_abstract(&inputs[0], &inputs[1], "matmul")?])
            }
            Self::MatrixTranspose => {
                operations::expect_input_count(inputs.len(), 1)?;
                Ok(vec![operations::matrix::transpose_abstract(&inputs[0], "matrix_transpose")?])
            }
            Self::LinearMatrixTranspose => {
                operations::expect_input_count(inputs.len(), 1)?;
                Ok(vec![operations::matrix::transpose_abstract(&inputs[0], "linear_matrix_transpose")?])
            }
            Self::Scale { .. } => Ok(vec![operations::unary_abstract(inputs)?]),
            Self::LeftMatMul { factor } => {
                operations::expect_input_count(inputs.len(), 1)?;
                Ok(vec![operations::matrix::matmul_abstract(
                    &<V as crate::types::Typed<ArrayType>>::tpe(factor),
                    &inputs[0],
                    "left_matmul",
                )?])
            }
            Self::RightMatMul { factor } => {
                operations::expect_input_count(inputs.len(), 1)?;
                Ok(vec![operations::matrix::matmul_abstract(
                    &inputs[0],
                    &<V as crate::types::Typed<ArrayType>>::tpe(factor),
                    "right_matmul",
                )?])
            }
            Self::Reshape { input_type, output_type } => {
                <ReshapeOp as Op<V>>::abstract_eval(
                    &ReshapeOp::new(input_type.clone(), output_type.clone()),
                    inputs,
                )
            }
            Self::VMap(vmap) => {
                let expected_inputs = vmap.body().repeated_input_types();
                if inputs.len() != expected_inputs.len() {
                    return Err(TraceError::InvalidInputCount { expected: expected_inputs.len(), got: inputs.len() });
                }
                if inputs != expected_inputs.as_slice() {
                    return Err(TraceError::IncompatibleAbstractValues { op: "vmap" });
                }
                Ok(vmap.body().repeated_output_types())
            }
            Self::Custom(op) => op.abstract_eval(inputs),
        }
    }

    fn eval(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        match self {
            Self::Add => {
                operations::expect_input_count(inputs.len(), 2)?;
                Ok(vec![inputs[0].clone() + inputs[1].clone()])
            }
            Self::Mul => {
                operations::expect_input_count(inputs.len(), 2)?;
                Ok(vec![inputs[0].clone() * inputs[1].clone()])
            }
            Self::Neg => {
                operations::expect_input_count(inputs.len(), 1)?;
                Ok(vec![-inputs[0].clone()])
            }
            Self::Sin => {
                operations::expect_input_count(inputs.len(), 1)?;
                Ok(vec![inputs[0].clone().sin()])
            }
            Self::Cos => {
                operations::expect_input_count(inputs.len(), 1)?;
                Ok(vec![inputs[0].clone().cos()])
            }
            Self::MatMul => {
                operations::expect_input_count(inputs.len(), 2)?;
                Ok(vec![inputs[0].clone().matmul(inputs[1].clone())])
            }
            Self::MatrixTranspose => {
                operations::expect_input_count(inputs.len(), 1)?;
                Ok(vec![inputs[0].clone().transpose_matrix()])
            }
            Self::LinearMatrixTranspose => {
                operations::expect_input_count(inputs.len(), 1)?;
                Ok(vec![inputs[0].clone().transpose_matrix()])
            }
            Self::Scale { factor } => {
                operations::expect_input_count(inputs.len(), 1)?;
                Ok(vec![factor.clone() * inputs[0].clone()])
            }
            Self::LeftMatMul { factor } => {
                operations::expect_input_count(inputs.len(), 1)?;
                Ok(vec![factor.clone().matmul(inputs[0].clone())])
            }
            Self::RightMatMul { factor } => {
                operations::expect_input_count(inputs.len(), 1)?;
                Ok(vec![inputs[0].clone().matmul(factor.clone())])
            }
            Self::Reshape { input_type: _, output_type } => {
                operations::expect_input_count(inputs.len(), 1)?;
                Ok(vec![inputs[0].clone().reshape(output_type.shape.clone())?])
            }
            Self::VMap(vmap) => vmap.body().eval_lanes(inputs),
            Self::Custom(op) => op.eval(inputs),
        }
    }
}

impl<V: TraceValue + FloatExt + ZeroLike + OneLike + MatrixOps + operations::reshape::ReshapeOps>
    DifferentiableOp<V> for PrimitiveOp<V>
{
    // NOTE: The VMap arms in the methods below delegate to `VMapOp<V>: DifferentiableOp<V>` which requires
    // `V: TransformLeaf`. This is fine because vmap ops are only ever staged for leaf value types. When the
    // dispatch is used with non-leaf types (e.g. `JitTracer<V>`) the VMap arm is unreachable and the error
    // fallback fires safely.
    fn replay_linearized_jit(
        &self,
        inputs: Vec<JvpTracer<JitTracer<V>, LinearTerm<JitTracer<V>>>>,
    ) -> Result<Vec<JvpTracer<JitTracer<V>, LinearTerm<JitTracer<V>>>>, TraceError>
    where
        V: TransformLeaf,
    {
        match self {
            Self::Add => AddOp.replay_linearized_jit(inputs),
            Self::Mul => MulOp.replay_linearized_jit(inputs),
            Self::Neg => NegOp.replay_linearized_jit(inputs),
            Self::Sin => SinOp.replay_linearized_jit(inputs),
            Self::Cos => CosOp.replay_linearized_jit(inputs),
            Self::MatMul => MatMulOp.replay_linearized_jit(inputs),
            Self::MatrixTranspose => MatrixTransposeOp.replay_linearized_jit(inputs),
            Self::LinearMatrixTranspose => LinearMatrixTransposeOp.replay_linearized_jit(inputs),
            Self::Scale { factor } => ScaleOp::new(factor.clone()).replay_linearized_jit(inputs),
            Self::LeftMatMul { factor } => LeftMatMulOp::new(factor.clone()).replay_linearized_jit(inputs),
            Self::RightMatMul { factor } => RightMatMulOp::new(factor.clone()).replay_linearized_jit(inputs),
            Self::Reshape { input_type, output_type } => {
                ReshapeOp::new(input_type.clone(), output_type.clone()).replay_linearized_jit(inputs)
            }
            Self::VMap(vmap) => vmap.replay_linearized_jit(inputs),
            Self::Custom(op) => op.replay_linearized_jit(inputs),
        }
    }

    fn apply_program_jvp_rule(
        &self,
        inputs: &[JvpTracer<V, LinearTerm<V>>],
    ) -> Result<Vec<JvpTracer<V, LinearTerm<V>>>, TraceError>
    where
        V: FloatExt + ZeroLike + OneLike + MatrixOps + operations::reshape::ReshapeOps,
    {
        match self {
            Self::Add => AddOp.apply_program_jvp_rule(inputs),
            Self::Mul => MulOp.apply_program_jvp_rule(inputs),
            Self::Neg => NegOp.apply_program_jvp_rule(inputs),
            Self::Sin => SinOp.apply_program_jvp_rule(inputs),
            Self::Cos => CosOp.apply_program_jvp_rule(inputs),
            Self::MatMul => MatMulOp.apply_program_jvp_rule(inputs),
            Self::MatrixTranspose => MatrixTransposeOp.apply_program_jvp_rule(inputs),
            Self::LinearMatrixTranspose => LinearMatrixTransposeOp.apply_program_jvp_rule(inputs),
            Self::Scale { factor } => ScaleOp::new(factor.clone()).apply_program_jvp_rule(inputs),
            Self::LeftMatMul { factor } => LeftMatMulOp::new(factor.clone()).apply_program_jvp_rule(inputs),
            Self::RightMatMul { factor } => RightMatMulOp::new(factor.clone()).apply_program_jvp_rule(inputs),
            Self::Reshape { input_type, output_type } => {
                ReshapeOp::new(input_type.clone(), output_type.clone()).apply_program_jvp_rule(inputs)
            }
            Self::VMap(_) => Err(TraceError::HigherOrderOpFailure {
                op: "linearize_program",
                message: "vmap JVP rule requires TransformLeaf values; use the replay path instead".to_string(),
            }),
            Self::Custom(op) => op.apply_program_jvp_rule(inputs),
        }
    }

    fn transpose_program_op(
        &self,
        builder: &mut ProgramBuilder<V>,
        inputs: &[AtomId],
        outputs: &[AtomId],
        output_cotangents: &[AtomId],
    ) -> Result<Vec<Option<AtomId>>, TraceError>
    where
        V: FloatExt + ZeroLike + OneLike + MatrixOps + operations::reshape::ReshapeOps,
    {
        match self {
            Self::Add => AddOp.transpose_program_op(builder, inputs, outputs, output_cotangents),
            Self::Mul => MulOp.transpose_program_op(builder, inputs, outputs, output_cotangents),
            Self::Neg => NegOp.transpose_program_op(builder, inputs, outputs, output_cotangents),
            Self::Sin => SinOp.transpose_program_op(builder, inputs, outputs, output_cotangents),
            Self::Cos => CosOp.transpose_program_op(builder, inputs, outputs, output_cotangents),
            Self::MatMul => MatMulOp.transpose_program_op(builder, inputs, outputs, output_cotangents),
            Self::MatrixTranspose => {
                MatrixTransposeOp.transpose_program_op(builder, inputs, outputs, output_cotangents)
            }
            Self::LinearMatrixTranspose => {
                LinearMatrixTransposeOp.transpose_program_op(builder, inputs, outputs, output_cotangents)
            }
            Self::Scale { factor } => {
                ScaleOp::new(factor.clone()).transpose_program_op(builder, inputs, outputs, output_cotangents)
            }
            Self::LeftMatMul { factor } => {
                LeftMatMulOp::new(factor.clone()).transpose_program_op(builder, inputs, outputs, output_cotangents)
            }
            Self::RightMatMul { factor } => {
                RightMatMulOp::new(factor.clone()).transpose_program_op(builder, inputs, outputs, output_cotangents)
            }
            Self::Reshape { input_type, output_type } => ReshapeOp::new(input_type.clone(), output_type.clone())
                .transpose_program_op(builder, inputs, outputs, output_cotangents),
            Self::VMap(_) => Err(TraceError::HigherOrderOpFailure {
                op: "transpose_linear_program",
                message: "vmap transpose rule requires TransformLeaf values; use the replay path instead".to_string(),
            }),
            Self::Custom(op) => op.transpose_program_op(builder, inputs, outputs, output_cotangents),
        }
    }
}

impl<V: TraceValue> JvpOp<V> for PrimitiveOp<V>
where
    V: TransformLeaf,
{
    fn jvp<T>(&self, inputs: &[JvpTracer<V, T>]) -> Result<Vec<JvpTracer<V, T>>, TraceError>
    where
        T: TangentSpace<V>,
    {
        match self {
            Self::Add => AddOp.jvp(inputs),
            Self::Mul => MulOp.jvp(inputs),
            Self::Neg => NegOp.jvp(inputs),
            Self::Sin => SinOp.jvp(inputs),
            Self::Cos => CosOp.jvp(inputs),
            // MatMul and MatrixTranspose JVP rules are handled through `MatrixOps for JvpTracer<V, T>` in
            // `operations::matrix` and are never dispatched through the `JvpOp` trait on `PrimitiveOp`.
            _ => Err(TraceError::HigherOrderOpFailure {
                op: "jvp",
                message: format!("eager JVP rule for staged op '{}' is not implemented", self.name()),
            }),
        }
    }
}

impl<V: TraceValue> BatchOp<V> for PrimitiveOp<V>
where
    V: TransformLeaf,
{
    fn batch(&self, inputs: &[Batch<V>]) -> Result<Vec<Batch<V>>, TraceError> {
        match self {
            Self::Add => AddOp.batch(inputs),
            Self::Mul => MulOp.batch(inputs),
            Self::Neg => NegOp.batch(inputs),
            Self::Sin => SinOp.batch(inputs),
            Self::Cos => CosOp.batch(inputs),
            Self::MatMul => MatMulOp.batch(inputs),
            Self::MatrixTranspose => MatrixTransposeOp.batch(inputs),
            _ => Err(TraceError::HigherOrderOpFailure {
                op: "batch",
                message: format!("batching rule for staged op '{}' is not implemented", self.name()),
            }),
        }
    }
}
