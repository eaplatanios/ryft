//! Primitive operation traits for `tracing_v2`.
//!
//! The staged op set is intentionally open: each primitive is represented by its own concrete type implementing one
//! or more transform-specific traits. This module keeps only the operation-neutral dispatch interfaces.
//!
//! # Trait hierarchy
//!
//! ```text
//! Op (shape-level, NOT generic over V)
//! ├─ Eval<V>               — concrete execution on values of type V
//! ├─ LinearOp<V>           — transpose + replay rules for linear programs
//! ├─ DifferentiableOp<V, T> — forward-mode JVP rule, generic over tangent type T
//! └─ BatchOp<V>            — batching rule for vmap
//! ```
//!
//! [`Op`] is intentionally non-generic: `abstract_eval`, `name`, and `as_any` operate on [`ArrayType`] metadata
//! only. This means `Graph<O: Op, V, ...>` can be constructed, displayed, and simplified for *any* value type
//! without requiring operation-specific value bounds.

use std::{
    any::Any,
    fmt::{Debug, Display},
    sync::Arc,
};

use crate::tracing_v2::{
    FloatExt, MatrixOps, OneLike, TraceError, TraceValue, TransformLeaf, ZeroLike,
    batch::Batch,
    forward::JvpTracer,
    graph::AtomId,
    jit::JitTracer,
    linear::LinearTerm,
    program::ProgramBuilder,
};
use crate::types::{ArrayType, Typed};

/// Shape-level operation interface for staged graphs.
///
/// This trait covers the metadata surface needed for graph construction, display, simplification, and MLIR lowering.
/// Concrete execution is provided by the separate [`Eval`] trait. Staged-program differentiation rules are split
/// between [`LinearOp`] (transpose/replay) and [`DifferentiableOp`] (forward-mode JVP).
pub trait Op: Debug + Display {
    /// Returns this operation as [`Any`] for downcasting.
    fn as_any(&self) -> &dyn Any;

    /// Returns the stable primitive name used in diagnostics and pretty-printing.
    fn name(&self) -> &'static str;

    /// Computes abstract output types from abstract input types without executing the operation.
    fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TraceError>;
}

/// Concrete execution capability for staged operations.
///
/// Separated from [`Op`] so that graph construction, display, and simplification can work without value-type bounds.
/// Only code paths that actually execute operations (graph replay, JIT example propagation) require this trait.
pub trait Eval<V: TraceValue>: Op {
    /// Executes the operation on concrete values.
    fn eval(&self, inputs: &[V]) -> Result<Vec<V>, TraceError>;
}

/// Operations that can appear in tangent/cotangent programs and support reverse-mode transposition.
///
/// This trait covers the staged-program side of differentiation: transposing linearized programs
/// ([`transpose_program_op`]) and replaying them inside a JIT scope ([`replay_linearized_jit`]).
/// Default implementations return [`TraceError::HigherOrderOpFailure`] so that operations only need
/// to override the methods relevant to their transform support.
///
/// [`transpose_program_op`]: LinearOp::transpose_program_op
/// [`replay_linearized_jit`]: LinearOp::replay_linearized_jit
pub trait LinearOp<V: TraceValue>: Op {
    /// Applies the transpose rule for reverse-mode differentiation.
    fn transpose_program_op(
        &self,
        _builder: &mut ProgramBuilder<V>,
        _inputs: &[AtomId],
        _outputs: &[AtomId],
        _output_cotangents: &[AtomId],
    ) -> Result<Vec<Option<AtomId>>, TraceError> {
        Err(TraceError::HigherOrderOpFailure {
            op: "transpose_linear_program",
            message: format!("transpose rule for staged op '{}' is not implemented", self.name()),
        })
    }

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
}

/// Forward-mode differentiation rule, generic over the tangent type `T`.
///
/// Each operation implements this trait with the exact bounds on `T` that its JVP rule requires.
/// For example, [`AddOp`](crate::tracing_v2::operations::AddOp) only needs `T: TangentSpace<V>`,
/// while [`MatMulOp`](crate::tracing_v2::operations::MatMulOp) needs
/// `T: TangentSpace<V> + MatrixTangentSpace<V>`.
///
/// [`TangentSpace`]: crate::tracing_v2::forward::TangentSpace
/// [`MatrixTangentSpace`]: crate::tracing_v2::MatrixTangentSpace
pub trait DifferentiableOp<V: TraceValue, T>: Op {
    /// Applies the forward-mode JVP rule.
    fn jvp(&self, inputs: &[JvpTracer<V, T>]) -> Result<Vec<JvpTracer<V, T>>, TraceError>;
}

/// Combined trait for custom operations that support evaluation, linearization, and differentiation.
///
/// This is the required capability set for the [`PrimitiveOp::Custom`] escape hatch. External
/// operations must implement [`Eval<V>`] (concrete execution), [`LinearOp<V>`]
/// (transpose/replay rules), [`DifferentiableOp<V, LinearTerm<V>>`] (program-level JVP), and
/// [`Op`] (shape-level metadata).
pub trait CustomOp<V: TraceValue>: Eval<V> + LinearOp<V> + DifferentiableOp<V, LinearTerm<V>> {}

impl<V: TraceValue, T: Eval<V> + LinearOp<V> + DifferentiableOp<V, LinearTerm<V>>> CustomOp<V> for T {}

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
    Reshape { input_type: ArrayType, output_type: ArrayType },

    /// Higher-order `vmap` carrying a compiled per-lane body and optional transpose body.
    VMap(Box<crate::tracing_v2::operations::VMapOp<V>>),

    /// Escape hatch for user- or crate-defined operations outside `ryft-core`.
    Custom(Arc<dyn CustomOp<V>>),
}

/// Canonical operation type used by the staged program IR.
pub type PrimitiveOpRef<V> = PrimitiveOp<V>;

/// Shared reference to a dynamically dispatched staged operation that supports differentiation.
///
/// NOTE: This alias is kept for backward compatibility with code that still wraps ops in `Arc`.
/// New code should prefer [`PrimitiveOp`] directly.
pub type StagedOpRef<V> = Arc<dyn CustomOp<V>>;

/// Primitive operation with a batching rule used by `vmap`.
pub(crate) trait BatchOp<V: TraceValue>: Op {
    /// Applies the primitive's batching rule to batched inputs.
    fn batch(&self, inputs: &[Batch<V>]) -> Result<Vec<Batch<V>>, TraceError>;
}

// ---------------------------------------------------------------------------
// Arc forwarding impls
// ---------------------------------------------------------------------------

impl<T: Op + ?Sized> Op for Arc<T> {
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
}

impl<T: Eval<V> + ?Sized, V: TraceValue> Eval<V> for Arc<T> {
    #[inline]
    fn eval(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        (**self).eval(inputs)
    }
}

impl<T: LinearOp<V> + ?Sized, V: TraceValue> LinearOp<V> for Arc<T> {
    #[inline]
    fn transpose_program_op(
        &self,
        builder: &mut ProgramBuilder<V>,
        inputs: &[AtomId],
        outputs: &[AtomId],
        output_cotangents: &[AtomId],
    ) -> Result<Vec<Option<AtomId>>, TraceError> {
        (**self).transpose_program_op(builder, inputs, outputs, output_cotangents)
    }

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
}

impl<T: DifferentiableOp<V, U> + ?Sized, V: TraceValue, U> DifferentiableOp<V, U> for Arc<T> {
    #[inline]
    fn jvp(&self, inputs: &[JvpTracer<V, U>]) -> Result<Vec<JvpTracer<V, U>>, TraceError> {
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
// PrimitiveOp — Debug, Display, Op, Eval, LinearOp, DifferentiableOp, BatchOp
// ---------------------------------------------------------------------------

use crate::tracing_v2::operations::{
    AddOp, CosOp, LeftMatMulOp, LinearMatrixTransposeOp, MatMulOp, MatrixTransposeOp, MulOp, NegOp, ReshapeOp,
    RightMatMulOp, ScaleOp, SinOp, left_matmul::left_matmul_abstract_eval, right_matmul::right_matmul_abstract_eval,
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

/// [`Op`] for [`PrimitiveOp`] requires NO value-type bounds — shape validation works for any `V: TraceValue`.
impl<V: TraceValue> Op for PrimitiveOp<V> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &'static str {
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
            Self::VMap(vmap) => vmap.name(),
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
            Self::LinearMatrixTranspose => LinearMatrixTransposeOp.abstract_eval(inputs),
            Self::Scale { .. } => ScaleOp::<V>::abstract_eval_static(inputs),
            Self::LeftMatMul { factor } => left_matmul_abstract_eval(&Typed::tpe(factor), inputs),
            Self::RightMatMul { factor } => right_matmul_abstract_eval(&Typed::tpe(factor), inputs),
            Self::Reshape { input_type, output_type } => {
                <ReshapeOp as Op>::abstract_eval(&ReshapeOp::new(input_type.clone(), output_type.clone()), inputs)
            }
            Self::VMap(vmap) => vmap.abstract_eval(inputs),
            Self::Custom(op) => op.abstract_eval(inputs),
        }
    }
}

/// [`Eval`] for [`PrimitiveOp`] requires the full value capability set.
impl<V: TraceValue + FloatExt + ZeroLike + OneLike + MatrixOps + crate::tracing_v2::operations::reshape::ReshapeOps>
    Eval<V> for PrimitiveOp<V>
{
    fn eval(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        match self {
            Self::Add => AddOp.eval(inputs),
            Self::Mul => MulOp.eval(inputs),
            Self::Neg => NegOp.eval(inputs),
            Self::Sin => SinOp.eval(inputs),
            Self::Cos => CosOp.eval(inputs),
            Self::MatMul => MatMulOp.eval(inputs),
            Self::MatrixTranspose => MatrixTransposeOp.eval(inputs),
            Self::LinearMatrixTranspose => LinearMatrixTransposeOp.eval(inputs),
            Self::Scale { factor } => ScaleOp::new(factor.clone()).eval(inputs),
            Self::LeftMatMul { factor } => LeftMatMulOp::new(factor.clone()).eval(inputs),
            Self::RightMatMul { factor } => RightMatMulOp::new(factor.clone()).eval(inputs),
            Self::Reshape { input_type, output_type } => {
                ReshapeOp::new(input_type.clone(), output_type.clone()).eval(inputs)
            }
            Self::VMap(vmap) => vmap.eval(inputs),
            Self::Custom(op) => op.eval(inputs),
        }
    }
}

impl<V: TraceValue + FloatExt + ZeroLike + OneLike + MatrixOps + crate::tracing_v2::operations::reshape::ReshapeOps>
    LinearOp<V> for PrimitiveOp<V>
{
    fn transpose_program_op(
        &self,
        builder: &mut ProgramBuilder<V>,
        inputs: &[AtomId],
        outputs: &[AtomId],
        output_cotangents: &[AtomId],
    ) -> Result<Vec<Option<AtomId>>, TraceError> {
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
}

impl<V: TraceValue + FloatExt + ZeroLike + OneLike + MatrixOps + crate::tracing_v2::operations::reshape::ReshapeOps>
    DifferentiableOp<V, LinearTerm<V>> for PrimitiveOp<V>
{
    fn jvp(
        &self,
        inputs: &[JvpTracer<V, LinearTerm<V>>],
    ) -> Result<Vec<JvpTracer<V, LinearTerm<V>>>, TraceError> {
        match self {
            Self::Add => DifferentiableOp::<V, LinearTerm<V>>::jvp(&AddOp, inputs),
            Self::Mul => DifferentiableOp::<V, LinearTerm<V>>::jvp(&MulOp, inputs),
            Self::Neg => DifferentiableOp::<V, LinearTerm<V>>::jvp(&NegOp, inputs),
            Self::Sin => DifferentiableOp::<V, LinearTerm<V>>::jvp(&SinOp, inputs),
            Self::Cos => DifferentiableOp::<V, LinearTerm<V>>::jvp(&CosOp, inputs),
            Self::Scale { factor } => DifferentiableOp::<V, LinearTerm<V>>::jvp(&ScaleOp::new(factor.clone()), inputs),
            Self::MatMul => DifferentiableOp::<V, LinearTerm<V>>::jvp(&MatMulOp, inputs),
            Self::MatrixTranspose => DifferentiableOp::<V, LinearTerm<V>>::jvp(&MatrixTransposeOp, inputs),
            Self::LinearMatrixTranspose => {
                DifferentiableOp::<V, LinearTerm<V>>::jvp(&LinearMatrixTransposeOp, inputs)
            }
            Self::LeftMatMul { factor } => {
                DifferentiableOp::<V, LinearTerm<V>>::jvp(&LeftMatMulOp::new(factor.clone()), inputs)
            }
            Self::RightMatMul { factor } => {
                DifferentiableOp::<V, LinearTerm<V>>::jvp(&RightMatMulOp::new(factor.clone()), inputs)
            }
            Self::Reshape { input_type, output_type } => {
                DifferentiableOp::<V, LinearTerm<V>>::jvp(&ReshapeOp::new(input_type.clone(), output_type.clone()), inputs)
            }
            Self::VMap(vmap) => Err(TraceError::HigherOrderOpFailure {
                op: "linearize_program",
                message: format!("JVP rule for staged op '{}' is not implemented", vmap.name()),
            }),
            Self::Custom(op) => op.jvp(inputs),
        }
    }
}

impl<V: TransformLeaf> BatchOp<V> for PrimitiveOp<V> {
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
