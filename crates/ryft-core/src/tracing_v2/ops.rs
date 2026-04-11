//! Primitive operation traits for `tracing_v2`.
//!
//! The staged op set is intentionally open: each primitive is represented by its own concrete type implementing one
//! or more transform-specific traits. This module keeps only the operation-neutral dispatch interfaces.
//!
//! # Trait hierarchy
//!
//! ```text
//! Op (shape-level, NOT generic over V)
//! ├─ InterpretableOp<V>               — concrete execution on values of type V
//! ├─ LinearOp<V>           — transpose rule for linear programs
//! ├─ DifferentiableOp<V, T> — forward-mode JVP rule, generic over tangent type T
//! └─ BatchOp<V>            — batching rule for vmap
//! ```
//!
//! [`Op`] is intentionally non-generic: `abstract_eval` and `name` operate on [`ArrayType`] metadata only.
//! This means `Graph<O: Op, V, ...>` can be constructed, displayed, and simplified for *any* value type
//! without requiring operation-specific value bounds.

use std::{
    fmt::{Debug, Display},
    ops::{Add, Mul, Neg},
    sync::Arc,
};

use crate::tracing_v2::{
    FloatExt, Linearized, MatrixOps, OneLike, TraceError, TraceValue, ZeroLike, batch::Batch, forward::JvpTracer,
    graph::AtomId, jit::JitTracer, linear::LinearTerm, program::ProgramBuilder,
};
use crate::types::{ArrayType, Typed};

/// Shape-level operation interface for staged graphs.
///
/// This trait covers the metadata surface needed for graph construction, display, simplification, and MLIR lowering.
/// Concrete execution is provided by the separate [`InterpretableOp`] trait. Staged-program differentiation rules are split
/// between [`LinearOp`] (transpose/replay) and [`DifferentiableOp`] (forward-mode JVP).
pub trait Op: Debug + Display {
    /// Returns the stable primitive name used in diagnostics and pretty-printing.
    fn name(&self) -> &'static str;

    /// Computes abstract output types from abstract input types without executing the operation.
    fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TraceError>;

    /// Returns simplified output atoms if this operation is a trivial algebraic identity.
    ///
    /// Called during graph construction to eliminate no-op operations like `x + 0`, `x * 1`,
    /// or `scale(x, 1)`. The callbacks check whether an input atom is a constant zero or one.
    /// Returns `None` if no simplification applies.
    fn try_simplify(
        &self,
        _inputs: &[usize],
        _is_zero_constant: &dyn Fn(usize) -> bool,
        _is_one_constant: &dyn Fn(usize) -> bool,
    ) -> Option<Vec<usize>> {
        None
    }
}

/// Concrete execution capability for staged operations.
///
/// Separated from [`Op`] so that graph construction, display, and simplification can work without value-type bounds.
/// Only code paths that actually execute operations (graph replay, JIT example propagation) require this trait.
pub trait InterpretableOp<V: TraceValue>: Op {
    /// Executes the operation on concrete values.
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TraceError>;
}

/// Operations that can appear in tangent/cotangent programs and support reverse-mode transposition.
pub trait LinearOp<V: TraceValue>: Op {
    /// Applies the transpose rule for reverse-mode differentiation.
    fn transpose(
        &self,
        builder: &mut ProgramBuilder<V>,
        inputs: &[AtomId],
        outputs: &[AtomId],
        output_cotangents: &[AtomId],
    ) -> Result<Vec<Option<AtomId>>, TraceError>;
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

/// Bundle of all capabilities required by the [`PrimitiveOp::Custom`] escape hatch.
///
/// This trait exists solely to give Rust a single name for `dyn` dispatch. It is automatically
/// implemented for any type that satisfies the four underlying trait bounds — user code should
/// never write `impl CustomOp<V>` directly.
///
/// [`Linearized<JitTracer<V>>`]: crate::tracing_v2::linear::Linearized
pub trait CustomOp<V: TraceValue>:
    InterpretableOp<V>
    + LinearOp<V>
    + DifferentiableOp<V, LinearTerm<V>>
    + InterpretableOp<Linearized<JitTracer<V>>>
    + AsAny
where
    Linearized<JitTracer<V>>: TraceValue,
{
}

impl<
    V: TraceValue,
    T: 'static
        + InterpretableOp<V>
        + LinearOp<V>
        + DifferentiableOp<V, LinearTerm<V>>
        + InterpretableOp<Linearized<JitTracer<V>>>,
> CustomOp<V> for T
where
    Linearized<JitTracer<V>>: TraceValue,
{
}

/// Upcasting helper for downcasting custom operations behind `dyn CustomOp<V>`.
///
/// Automatically implemented for all `'static` types. Used by crate-external code (e.g., XLA
/// lowering) that needs to recover the concrete type of a custom operation via
/// [`Any::downcast_ref`](std::any::Any::downcast_ref).
pub trait AsAny {
    /// Returns `self` as a `&dyn Any` for downcasting.
    fn as_any(&self) -> &dyn std::any::Any;
}

impl<T: 'static> AsAny for T {
    fn as_any(&self) -> &dyn std::any::Any {
        self
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
    Reshape { input_type: ArrayType, output_type: ArrayType },

    /// Higher-order `vmap` carrying a compiled per-lane body and optional transpose body.
    VMap(Box<crate::tracing_v2::operations::VMapOp<V>>),

    /// Higher-order rematerialization boundary carrying a compiled body and optional transpose body.
    Rematerialize(Box<crate::tracing_v2::operations::RematerializeOp<V>>),

    /// Escape hatch for user- or crate-defined operations outside `ryft-core`.
    Custom(Arc<dyn CustomOp<V>>),
}

/// Canonical operation type used by the staged program IR.
pub type PrimitiveOpRef<V> = PrimitiveOp<V>;

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
    fn name(&self) -> &'static str {
        (**self).name()
    }

    #[inline]
    fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TraceError> {
        (**self).abstract_eval(inputs)
    }

    #[inline]
    fn try_simplify(
        &self,
        inputs: &[usize],
        is_zero_constant: &dyn Fn(usize) -> bool,
        is_one_constant: &dyn Fn(usize) -> bool,
    ) -> Option<Vec<usize>> {
        (**self).try_simplify(inputs, is_zero_constant, is_one_constant)
    }
}

impl<T: InterpretableOp<V> + ?Sized, V: TraceValue> InterpretableOp<V> for Arc<T> {
    #[inline]
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        (**self).interpret(inputs)
    }
}

impl<T: LinearOp<V> + ?Sized, V: TraceValue> LinearOp<V> for Arc<T> {
    #[inline]
    fn transpose(
        &self,
        builder: &mut ProgramBuilder<V>,
        inputs: &[AtomId],
        outputs: &[AtomId],
        output_cotangents: &[AtomId],
    ) -> Result<Vec<Option<AtomId>>, TraceError> {
        (**self).transpose(builder, inputs, outputs, output_cotangents)
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
// PrimitiveOp — Debug, Display, Op, InterpretableOp, LinearOp, DifferentiableOp, BatchOp
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
            Self::Rematerialize(remat) => Debug::fmt(remat, formatter),
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
            Self::Rematerialize(remat) => remat.name(),
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
            Self::Rematerialize(remat) => remat.abstract_eval(inputs),
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
            Self::Scale { factor } => {
                ScaleOp::new(factor.clone()).try_simplify(inputs, is_zero_constant, is_one_constant)
            }
            Self::LeftMatMul { factor } => {
                if crate::tracing_v2::graph::is_identity_one(factor) {
                    Some(inputs.to_vec())
                } else {
                    None
                }
            }
            Self::RightMatMul { factor } => {
                if crate::tracing_v2::graph::is_identity_one(factor) {
                    Some(inputs.to_vec())
                } else {
                    None
                }
            }
            Self::Custom(op) => op.try_simplify(inputs, is_zero_constant, is_one_constant),
            _ => None,
        }
    }
}

/// [`InterpretableOp`] for [`PrimitiveOp`] requires the full value capability set.
impl<V: TraceValue + FloatExt + ZeroLike + OneLike + MatrixOps + crate::tracing_v2::operations::reshape::ReshapeOps>
    InterpretableOp<V> for PrimitiveOp<V>
{
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        match self {
            Self::Add => AddOp.interpret(inputs),
            Self::Mul => MulOp.interpret(inputs),
            Self::Neg => NegOp.interpret(inputs),
            Self::Sin => SinOp.interpret(inputs),
            Self::Cos => CosOp.interpret(inputs),
            Self::MatMul => MatMulOp.interpret(inputs),
            Self::MatrixTranspose => MatrixTransposeOp.interpret(inputs),
            Self::LinearMatrixTranspose => LinearMatrixTransposeOp.interpret(inputs),
            Self::Scale { factor } => ScaleOp::new(factor.clone()).interpret(inputs),
            Self::LeftMatMul { factor } => LeftMatMulOp::new(factor.clone()).interpret(inputs),
            Self::RightMatMul { factor } => RightMatMulOp::new(factor.clone()).interpret(inputs),
            Self::Reshape { input_type, output_type } => {
                ReshapeOp::new(input_type.clone(), output_type.clone()).interpret(inputs)
            }
            Self::VMap(vmap) => vmap.interpret(inputs),
            Self::Rematerialize(remat) => remat.interpret(inputs),
            Self::Custom(op) => InterpretableOp::<V>::interpret(op.as_ref(), inputs),
        }
    }
}

impl<V: TraceValue + FloatExt + ZeroLike + OneLike + MatrixOps + crate::tracing_v2::operations::reshape::ReshapeOps>
    LinearOp<V> for PrimitiveOp<V>
{
    fn transpose(
        &self,
        builder: &mut ProgramBuilder<V>,
        inputs: &[AtomId],
        outputs: &[AtomId],
        output_cotangents: &[AtomId],
    ) -> Result<Vec<Option<AtomId>>, TraceError> {
        match self {
            Self::Add => AddOp.transpose(builder, inputs, outputs, output_cotangents),
            Self::Mul => MulOp.transpose(builder, inputs, outputs, output_cotangents),
            Self::Neg => NegOp.transpose(builder, inputs, outputs, output_cotangents),
            Self::Sin => SinOp.transpose(builder, inputs, outputs, output_cotangents),
            Self::Cos => CosOp.transpose(builder, inputs, outputs, output_cotangents),
            Self::MatMul => MatMulOp.transpose(builder, inputs, outputs, output_cotangents),
            Self::MatrixTranspose => MatrixTransposeOp.transpose(builder, inputs, outputs, output_cotangents),
            Self::LinearMatrixTranspose => {
                LinearMatrixTransposeOp.transpose(builder, inputs, outputs, output_cotangents)
            }
            Self::Scale { factor } => {
                ScaleOp::new(factor.clone()).transpose(builder, inputs, outputs, output_cotangents)
            }
            Self::LeftMatMul { factor } => {
                LeftMatMulOp::new(factor.clone()).transpose(builder, inputs, outputs, output_cotangents)
            }
            Self::RightMatMul { factor } => {
                RightMatMulOp::new(factor.clone()).transpose(builder, inputs, outputs, output_cotangents)
            }
            Self::Reshape { input_type, output_type } => ReshapeOp::new(input_type.clone(), output_type.clone())
                .transpose(builder, inputs, outputs, output_cotangents),
            Self::VMap(_) => Err(TraceError::HigherOrderOpFailure {
                op: "transpose_linear_program",
                message: "vmap transpose rule requires concrete leaf values; use the replay path instead".to_string(),
            }),
            Self::Rematerialize(remat) => remat.transpose(builder, inputs, outputs, output_cotangents),
            Self::Custom(op) => op.transpose(builder, inputs, outputs, output_cotangents),
        }
    }
}

/// Linearized JIT replay: evaluates staged operations on [`Linearized<JitTracer<V>>`] values.
///
/// For pure (non-capturing) ops, this is covered by their generic [`InterpretableOp<V>`] implementations
/// because [`JvpTracer`] already implements all necessary arithmetic, matrix, and reshape traits.
/// Capturing ops ([`ScaleOp`], [`LeftMatMulOp`], [`RightMatMulOp`]) and higher-order ops
/// ([`VMapOp`](crate::tracing_v2::operations::VMapOp),
/// [`RematerializeOp`](crate::tracing_v2::operations::RematerializeOp)) provide dedicated
/// [`InterpretableOp`] implementations that lift captured constants into the JIT trace.
///
/// [`Linearized<JitTracer<V>>`]: crate::tracing_v2::linear::Linearized
/// [`ScaleOp`]: crate::tracing_v2::operations::ScaleOp
/// [`LeftMatMulOp`]: crate::tracing_v2::operations::LeftMatMulOp
/// [`RightMatMulOp`]: crate::tracing_v2::operations::RightMatMulOp
impl<
    V: TraceValue
        + FloatExt
        + ZeroLike
        + OneLike
        + Add<Output = V>
        + Mul<Output = V>
        + Neg<Output = V>
        + MatrixOps
        + crate::tracing_v2::operations::reshape::ReshapeOps,
> InterpretableOp<crate::tracing_v2::linear::Linearized<JitTracer<V>>> for PrimitiveOp<V>
{
    fn interpret(
        &self,
        inputs: &[crate::tracing_v2::linear::Linearized<JitTracer<V>>],
    ) -> Result<Vec<crate::tracing_v2::linear::Linearized<JitTracer<V>>>, TraceError> {
        match self {
            Self::Add => AddOp.interpret(inputs),
            Self::Mul => MulOp.interpret(inputs),
            Self::Neg => NegOp.interpret(inputs),
            Self::Sin => SinOp.interpret(inputs),
            Self::Cos => CosOp.interpret(inputs),
            Self::MatMul => MatMulOp.interpret(inputs),
            Self::MatrixTranspose => MatrixTransposeOp.interpret(inputs),
            Self::LinearMatrixTranspose => LinearMatrixTransposeOp.interpret(inputs),
            Self::Scale { factor } => ScaleOp::new(factor.clone()).interpret(inputs),
            Self::LeftMatMul { factor } => LeftMatMulOp::new(factor.clone()).interpret(inputs),
            Self::RightMatMul { factor } => RightMatMulOp::new(factor.clone()).interpret(inputs),
            Self::Reshape { input_type, output_type } => {
                ReshapeOp::new(input_type.clone(), output_type.clone()).interpret(inputs)
            }
            Self::VMap(vmap) => vmap.interpret(inputs),
            Self::Rematerialize(remat) => remat.interpret(inputs),
            Self::Custom(op) => {
                InterpretableOp::<crate::tracing_v2::linear::Linearized<JitTracer<V>>>::interpret(op.as_ref(), inputs)
            }
        }
    }
}

impl<V: TraceValue + FloatExt + ZeroLike + OneLike + MatrixOps + crate::tracing_v2::operations::reshape::ReshapeOps>
    DifferentiableOp<V, LinearTerm<V>> for PrimitiveOp<V>
{
    fn jvp(&self, inputs: &[JvpTracer<V, LinearTerm<V>>]) -> Result<Vec<JvpTracer<V, LinearTerm<V>>>, TraceError> {
        match self {
            Self::Add => DifferentiableOp::<V, LinearTerm<V>>::jvp(&AddOp, inputs),
            Self::Mul => DifferentiableOp::<V, LinearTerm<V>>::jvp(&MulOp, inputs),
            Self::Neg => DifferentiableOp::<V, LinearTerm<V>>::jvp(&NegOp, inputs),
            Self::Sin => DifferentiableOp::<V, LinearTerm<V>>::jvp(&SinOp, inputs),
            Self::Cos => DifferentiableOp::<V, LinearTerm<V>>::jvp(&CosOp, inputs),
            Self::Scale { factor } => DifferentiableOp::<V, LinearTerm<V>>::jvp(&ScaleOp::new(factor.clone()), inputs),
            Self::MatMul => DifferentiableOp::<V, LinearTerm<V>>::jvp(&MatMulOp, inputs),
            Self::MatrixTranspose => DifferentiableOp::<V, LinearTerm<V>>::jvp(&MatrixTransposeOp, inputs),
            Self::LinearMatrixTranspose => DifferentiableOp::<V, LinearTerm<V>>::jvp(&LinearMatrixTransposeOp, inputs),
            Self::LeftMatMul { factor } => {
                DifferentiableOp::<V, LinearTerm<V>>::jvp(&LeftMatMulOp::new(factor.clone()), inputs)
            }
            Self::RightMatMul { factor } => {
                DifferentiableOp::<V, LinearTerm<V>>::jvp(&RightMatMulOp::new(factor.clone()), inputs)
            }
            Self::Reshape { input_type, output_type } => DifferentiableOp::<V, LinearTerm<V>>::jvp(
                &ReshapeOp::new(input_type.clone(), output_type.clone()),
                inputs,
            ),
            Self::VMap(vmap) => Err(TraceError::HigherOrderOpFailure {
                op: "linearize_program",
                message: format!("JVP rule for staged op '{}' is not implemented", vmap.name()),
            }),
            Self::Rematerialize(remat) => DifferentiableOp::<V, LinearTerm<V>>::jvp(remat.as_ref(), inputs),
            Self::Custom(op) => DifferentiableOp::<V, LinearTerm<V>>::jvp(op.as_ref(), inputs),
        }
    }
}

impl<V: TraceValue + Add<Output = V> + Mul<Output = V> + Neg<Output = V> + FloatExt + MatrixOps> BatchOp<V>
    for PrimitiveOp<V>
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
