//! Closed default carriers for the built-in `tracing_v2` operation set.
//!
//! [`PrimitiveOp`] is the ordinary staged-operation carrier and [`LinearPrimitiveOp`] is the
//! linear-only sibling used by linear programs. Both enums are zero-cost wrappers around the
//! per-primitive op types in [`crate::tracing_v2::operations`] and use the
//! [`Custom`](PrimitiveOp::Custom) escape hatch for operations defined outside this crate.
//!
//! These carriers are the default backend choice for `ryft-core`. Other backends (for example
//! `ryft-xla`) own their own carrier enums and implement the same staging traits from the
//! per-operation modules to slot into the generic transform code.

use std::{
    fmt::{Debug, Display},
    ops::{Add, Mul, Neg},
    sync::Arc,
};

use crate::{
    parameters::{Parameter, Parameterized},
    tracing_v2::{
        Cos, MatrixOps, OneLike, Sin, TraceError, Traceable, Value, ZeroLike,
        batch::Batch,
        engine::Engine,
        forward::JvpTracer,
        jit::Tracer,
        linear::LinearTerm,
        operations::{
            AddOp, CosOp, LeftMatMulOp, MatMulOp, MatrixTransposeOp, MulOp, NegOp, ReshapeOp, RightMatMulOp, ScaleOp,
            SinOp, left_matmul::left_matmul_abstract_eval, right_matmul::right_matmul_abstract_eval,
        },
    },
    types::{ArrayType, Type, Typed},
};

use super::{
    DifferentiableOp, InterpretableOp, LinearOperation, Op, VectorizableOp,
    add::{AddTracingOperation, LinearAddOperation},
    cos::CosTracingOperation,
    custom::{CustomPrimitive, CustomTracingOperation, LinearCustomOperation, LinearCustomPrimitive},
    left_matmul::{LeftMatMulTracingOperation, LinearLeftMatMulOperation},
    matmul::MatMulTracingOperation,
    matrix_transpose::{LinearMatrixTransposeOperation, MatrixTransposeTracingOperation},
    mul::MulTracingOperation,
    neg::{LinearNegOperation, NegTracingOperation},
    rematerialize::{LinearRematerializeOperation, RematerializeTracingOperation},
    reshape::{LinearReshapeOperation, ReshapeTracingOperation},
    right_matmul::{LinearRightMatMulOperation, RightMatMulTracingOperation},
    scale::{LinearScaleOperation, ScaleTracingOperation},
    sin::SinTracingOperation,
    vmap::{LinearVMapOperation, VMapTracingOperation},
};

/// Closed set of built-in staged operations.
///
/// Every known primitive is a zero-cost enum variant. Operations originating outside
/// `ryft-core` (e.g., shard-map ops in `ryft-xla`) go through the [`Custom`](PrimitiveOp::Custom) escape
/// hatch, which still uses dynamic dispatch.
pub enum PrimitiveOp<T: Type + Display, V: Traceable<T> + Parameter> {
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

    /// Scalar or tensor scaling by a captured factor.
    Scale { factor: V },

    /// Left matrix multiplication by a captured factor: `factor @ input`.
    LeftMatMul { factor: V },

    /// Right matrix multiplication by a captured factor: `input @ factor`.
    RightMatMul { factor: V },

    /// Reshape between two statically known shapes.
    Reshape { input_type: T, output_type: T },

    /// Higher-order `vmap` carrying a compiled per-lane body and optional transpose body.
    VMap(Box<crate::tracing_v2::operations::VMapOp<T, V, PrimitiveOp<T, V>, LinearPrimitiveOp<T, V>>>),

    /// Higher-order rematerialization boundary carrying a compiled body and optional transpose body.
    Rematerialize(
        Box<crate::tracing_v2::operations::RematerializeOp<T, V, PrimitiveOp<T, V>, LinearPrimitiveOp<T, V>>>,
    ),

    /// Escape hatch for user- or crate-defined operations outside `ryft-core`.
    Custom(Arc<CustomPrimitive<T, V>>),
}

impl<T: Type + Display, V: Traceable<T>> Clone for PrimitiveOp<T, V> {
    fn clone(&self) -> Self {
        match self {
            Self::Add => Self::Add,
            Self::Mul => Self::Mul,
            Self::Neg => Self::Neg,
            Self::Sin => Self::Sin,
            Self::Cos => Self::Cos,
            Self::MatMul => Self::MatMul,
            Self::MatrixTranspose => Self::MatrixTranspose,
            Self::Scale { factor } => Self::Scale { factor: factor.clone() },
            Self::LeftMatMul { factor } => Self::LeftMatMul { factor: factor.clone() },
            Self::RightMatMul { factor } => Self::RightMatMul { factor: factor.clone() },
            Self::Reshape { input_type, output_type } => {
                Self::Reshape { input_type: input_type.clone(), output_type: output_type.clone() }
            }
            Self::VMap(vmap) => Self::VMap(vmap.clone()),
            Self::Rematerialize(remat) => Self::Rematerialize(remat.clone()),
            Self::Custom(op) => Self::Custom(op.clone()),
        }
    }
}

/// Closed set of operations that may appear in staged linear programs.
pub enum LinearPrimitiveOp<T: Type + Display, V: Traceable<T> + Parameter> {
    /// Elementwise addition.
    Add,

    /// Elementwise negation.
    Neg,

    /// Matrix transposition.
    MatrixTranspose,

    /// Scalar or tensor scaling by a captured factor.
    Scale { factor: V },

    /// Left matrix multiplication by a captured factor: `factor @ input`.
    LeftMatMul { factor: V },

    /// Right matrix multiplication by a captured factor: `input @ factor`.
    RightMatMul { factor: V },

    /// Reshape between two statically known shapes.
    Reshape { input_type: T, output_type: T },

    /// Higher-order `vmap` restricted to linear bodies and linear transpose bodies.
    VMap(Box<crate::tracing_v2::operations::LinearVMapOp<T, V, LinearPrimitiveOp<T, V>>>),

    /// Higher-order rematerialization boundary restricted to linear bodies and transpose bodies.
    Rematerialize(Box<crate::tracing_v2::operations::LinearRematerializeOp<T, V, LinearPrimitiveOp<T, V>>>),

    /// Escape hatch for user- or crate-defined linear custom operations.
    Custom(Arc<LinearCustomPrimitive<T, V>>),
}

impl<T: Type + Display, V: Traceable<T>> Clone for LinearPrimitiveOp<T, V> {
    fn clone(&self) -> Self {
        match self {
            Self::Add => Self::Add,
            Self::Neg => Self::Neg,
            Self::MatrixTranspose => Self::MatrixTranspose,
            Self::Scale { factor } => Self::Scale { factor: factor.clone() },
            Self::LeftMatMul { factor } => Self::LeftMatMul { factor: factor.clone() },
            Self::RightMatMul { factor } => Self::RightMatMul { factor: factor.clone() },
            Self::Reshape { input_type, output_type } => {
                Self::Reshape { input_type: input_type.clone(), output_type: output_type.clone() }
            }
            Self::VMap(vmap) => Self::VMap(vmap.clone()),
            Self::Rematerialize(remat) => Self::Rematerialize(remat.clone()),
            Self::Custom(op) => Self::Custom(op.clone()),
        }
    }
}

impl<V: Traceable<ArrayType>> LinearPrimitiveOp<ArrayType, V> {
    /// Wraps one custom primitive in the linear-only operation universe after verifying transpose support.
    pub fn custom(primitive: CustomPrimitive<ArrayType, V>) -> Result<Self, TraceError> {
        Ok(Self::Custom(Arc::new(primitive.into_linear()?)))
    }

    /// Wraps one shared custom primitive in the linear-only operation universe after verifying transpose support.
    pub fn custom_arc(primitive: Arc<CustomPrimitive<ArrayType, V>>) -> Result<Self, TraceError> {
        Ok(Self::Custom(Arc::new(LinearCustomPrimitive::from_custom_primitive(primitive)?)))
    }
}

impl<T: Type + Display, V: Traceable<T>> AddTracingOperation<T, V> for PrimitiveOp<T, V> {
    #[inline]
    fn add_op() -> Self {
        PrimitiveOp::Add
    }
}

impl<T: Type + Display, V: Traceable<T>> MulTracingOperation<T, V> for PrimitiveOp<T, V> {
    #[inline]
    fn mul_op() -> Self {
        PrimitiveOp::Mul
    }
}

impl<T: Type + Display, V: Traceable<T>> NegTracingOperation<T, V> for PrimitiveOp<T, V> {
    #[inline]
    fn neg_op() -> Self {
        PrimitiveOp::Neg
    }
}

impl<T: Type + Display, V: Traceable<T>> SinTracingOperation<T, V> for PrimitiveOp<T, V> {
    #[inline]
    fn sin_op() -> Self {
        PrimitiveOp::Sin
    }
}

impl<T: Type + Display, V: Traceable<T>> CosTracingOperation<T, V> for PrimitiveOp<T, V> {
    #[inline]
    fn cos_op() -> Self {
        PrimitiveOp::Cos
    }
}

impl<T: Type + Display, V: Traceable<T>> MatMulTracingOperation<T, V> for PrimitiveOp<T, V> {
    #[inline]
    fn matmul_op() -> Self {
        PrimitiveOp::MatMul
    }
}

impl<T: Type + Display, V: Traceable<T>> MatrixTransposeTracingOperation<T, V> for PrimitiveOp<T, V> {
    #[inline]
    fn matrix_transpose_op() -> Self {
        PrimitiveOp::MatrixTranspose
    }
}

impl<T: Type + Display, V: Traceable<T>> ScaleTracingOperation<T, V> for PrimitiveOp<T, V> {
    #[inline]
    fn scale_op(factor: V) -> Self {
        PrimitiveOp::Scale { factor }
    }
}

impl<T: Type + Display, V: Traceable<T>> LeftMatMulTracingOperation<T, V> for PrimitiveOp<T, V> {
    #[inline]
    fn left_matmul_op(factor: V) -> Self {
        PrimitiveOp::LeftMatMul { factor }
    }
}

impl<T: Type + Display, V: Traceable<T>> RightMatMulTracingOperation<T, V> for PrimitiveOp<T, V> {
    #[inline]
    fn right_matmul_op(factor: V) -> Self {
        PrimitiveOp::RightMatMul { factor }
    }
}

impl<T: Type + Display, V: Traceable<T>> ReshapeTracingOperation<T, V> for PrimitiveOp<T, V> {
    #[inline]
    fn reshape_op(input_type: T, output_type: T) -> Self {
        PrimitiveOp::Reshape { input_type, output_type }
    }
}

impl<T: Type + Display, V: Traceable<T>> VMapTracingOperation<T, V, LinearPrimitiveOp<T, V>> for PrimitiveOp<T, V> {
    #[inline]
    fn vmap_op(op: crate::tracing_v2::operations::VMapOp<T, V, Self, LinearPrimitiveOp<T, V>>) -> Self {
        PrimitiveOp::VMap(Box::new(op))
    }
}

impl<T: Type + Display, V: Traceable<T>> RematerializeTracingOperation<T, V, LinearPrimitiveOp<T, V>>
    for PrimitiveOp<T, V>
{
    #[inline]
    fn rematerialize_op(
        op: crate::tracing_v2::operations::RematerializeOp<T, V, Self, LinearPrimitiveOp<T, V>>,
    ) -> Self {
        PrimitiveOp::Rematerialize(Box::new(op))
    }
}

impl<T: Type + Display, V: Traceable<T>> CustomTracingOperation<T, V> for PrimitiveOp<T, V> {
    #[inline]
    fn custom_op(primitive: Arc<CustomPrimitive<T, V>>) -> Self {
        PrimitiveOp::Custom(primitive)
    }
}

impl<T: Type + Display, V: Traceable<T>> LinearAddOperation<T, V> for LinearPrimitiveOp<T, V> {
    #[inline]
    fn linear_add_op() -> Self {
        LinearPrimitiveOp::Add
    }
}

impl<T: Type + Display, V: Traceable<T>> LinearNegOperation<T, V> for LinearPrimitiveOp<T, V> {
    #[inline]
    fn linear_neg_op() -> Self {
        LinearPrimitiveOp::Neg
    }
}

impl<T: Type + Display, V: Traceable<T>> LinearMatrixTransposeOperation<T, V> for LinearPrimitiveOp<T, V> {
    #[inline]
    fn linear_matrix_transpose_op() -> Self {
        LinearPrimitiveOp::MatrixTranspose
    }
}

impl<T: Type + Display, V: Traceable<T>> LinearScaleOperation<T, V> for LinearPrimitiveOp<T, V> {
    #[inline]
    fn linear_scale_op(factor: V) -> Self {
        LinearPrimitiveOp::Scale { factor }
    }
}

impl<T: Type + Display, V: Traceable<T>> LinearLeftMatMulOperation<T, V> for LinearPrimitiveOp<T, V> {
    #[inline]
    fn linear_left_matmul_op(factor: V) -> Self {
        LinearPrimitiveOp::LeftMatMul { factor }
    }
}

impl<T: Type + Display, V: Traceable<T>> LinearRightMatMulOperation<T, V> for LinearPrimitiveOp<T, V> {
    #[inline]
    fn linear_right_matmul_op(factor: V) -> Self {
        LinearPrimitiveOp::RightMatMul { factor }
    }
}

impl<T: Type + Display, V: Traceable<T>> LinearReshapeOperation<T, V> for LinearPrimitiveOp<T, V> {
    #[inline]
    fn linear_reshape_op(input_type: T, output_type: T) -> Self {
        LinearPrimitiveOp::Reshape { input_type, output_type }
    }
}

impl<T: Type + Display, V: Traceable<T>> LinearVMapOperation<T, V> for LinearPrimitiveOp<T, V> {
    #[inline]
    fn linear_vmap_op(op: crate::tracing_v2::operations::LinearVMapOp<T, V, Self>) -> Self {
        LinearPrimitiveOp::VMap(Box::new(op))
    }
}

impl<T: Type + Display, V: Traceable<T>> LinearRematerializeOperation<T, V> for LinearPrimitiveOp<T, V> {
    #[inline]
    fn linear_rematerialize_op(op: crate::tracing_v2::operations::LinearRematerializeOp<T, V, Self>) -> Self {
        LinearPrimitiveOp::Rematerialize(Box::new(op))
    }
}

impl<T: Type + Display + 'static, V: Traceable<T>> LinearCustomOperation<T, V> for LinearPrimitiveOp<T, V> {
    #[inline]
    fn linear_custom_op(primitive: CustomPrimitive<T, V>) -> Result<Self, TraceError> {
        Ok(LinearPrimitiveOp::Custom(Arc::new(primitive.into_linear()?)))
    }

    #[inline]
    fn linear_custom_arc_op(primitive: Arc<CustomPrimitive<T, V>>) -> Result<Self, TraceError> {
        Ok(LinearPrimitiveOp::Custom(Arc::new(LinearCustomPrimitive::from_custom_primitive(primitive)?)))
    }
}

impl<T: Type + Display, V: Traceable<T>> Debug for PrimitiveOp<T, V> {
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
            Self::Reshape { input_type, output_type } => {
                write!(formatter, "Reshape({input_type} -> {output_type})")
            }
            Self::VMap(vmap) => Debug::fmt(vmap, formatter),
            Self::Rematerialize(remat) => Debug::fmt(remat, formatter),
            Self::Custom(op) => Debug::fmt(op.as_ref(), formatter),
        }
    }
}

impl<V: Traceable<ArrayType>> Display for PrimitiveOp<ArrayType, V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Reshape { output_type, .. } => write!(formatter, "reshape{}", output_type.shape),
            _ => write!(formatter, "{}", self.name()),
        }
    }
}

impl<T: Type + Display, V: Traceable<T>> Debug for LinearPrimitiveOp<T, V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Add => write!(formatter, "Add"),
            Self::Neg => write!(formatter, "Neg"),
            Self::MatrixTranspose => write!(formatter, "MatrixTranspose"),
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

impl<V: Traceable<ArrayType>> Display for LinearPrimitiveOp<ArrayType, V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Reshape { output_type, .. } => write!(formatter, "reshape{}", output_type.shape),
            _ => write!(formatter, "{}", self.name()),
        }
    }
}

/// [`Op`] for [`PrimitiveOp`] requires NO value-type bounds Ã¢â‚¬â€ shape validation works for any `V: Traceable<ArrayType>`.
impl<V: Traceable<ArrayType>> Op for PrimitiveOp<ArrayType, V> {
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
            Self::Scale { .. } => ScaleOp::<ArrayType, V>::abstract_eval_static(inputs),
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
                ScaleOp::<ArrayType, V>::new(factor.clone()).try_simplify(inputs, is_zero_constant, is_one_constant)
            }
            Self::LeftMatMul { factor } => {
                if crate::tracing_v2::is_identity_one(factor) {
                    Some(inputs.to_vec())
                } else {
                    None
                }
            }
            Self::RightMatMul { factor } => {
                if crate::tracing_v2::is_identity_one(factor) {
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

/// [`Op`] for [`LinearPrimitiveOp`] requires NO value-type bounds Ã¢â‚¬â€ shape validation works for any `V: Traceable<ArrayType>`.
impl<V: Traceable<ArrayType>> Op for LinearPrimitiveOp<ArrayType, V> {
    fn name(&self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Neg => "neg",
            Self::MatrixTranspose => "matrix_transpose",
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
            Self::Neg => NegOp.abstract_eval(inputs),
            Self::MatrixTranspose => MatrixTransposeOp.abstract_eval(inputs),
            Self::Scale { .. } => ScaleOp::<ArrayType, V>::abstract_eval_static(inputs),
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
            Self::Neg => NegOp.try_simplify(inputs, is_zero_constant, is_one_constant),
            Self::Scale { factor } => {
                ScaleOp::<ArrayType, V>::new(factor.clone()).try_simplify(inputs, is_zero_constant, is_one_constant)
            }
            Self::LeftMatMul { factor } => {
                if crate::tracing_v2::is_identity_one(factor) {
                    Some(inputs.to_vec())
                } else {
                    None
                }
            }
            Self::RightMatMul { factor } => {
                if crate::tracing_v2::is_identity_one(factor) {
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

/// [`InterpretableOp`] for [`PrimitiveOp`] requires the full union of value capabilities used by
/// the closed default ordinary-op carrier.
///
/// That broad union is local to [`PrimitiveOp`] itself. The higher-level tracing APIs avoid
/// exposing it as one public value-bundle trait and instead express their requirements through the
/// specific staged op carrier bounds they actually exercise.
impl<
    V: Value<ArrayType>
        + Add<Output = V>
        + Mul<Output = V>
        + Neg<Output = V>
        + Sin
        + Cos
        + ZeroLike
        + OneLike
        + MatrixOps
        + crate::tracing_v2::operations::reshape::ReshapeOps,
> InterpretableOp<ArrayType, V> for PrimitiveOp<ArrayType, V>
where
    Vec<V>: Parameterized<V, ParameterStructure: Clone + PartialEq>,
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
            Self::Scale { factor } => ScaleOp::new(factor.clone()).interpret(inputs),
            Self::LeftMatMul { factor } => LeftMatMulOp::new(factor.clone()).interpret(inputs),
            Self::RightMatMul { factor } => RightMatMulOp::new(factor.clone()).interpret(inputs),
            Self::Reshape { input_type, output_type } => {
                ReshapeOp::new(input_type.clone(), output_type.clone()).interpret(inputs)
            }
            Self::VMap(vmap) => vmap.interpret(inputs),
            Self::Rematerialize(remat) => remat.interpret(inputs),
            Self::Custom(op) => op.interpret(inputs),
        }
    }
}

impl<
    V: Traceable<ArrayType>
        + Add<Output = V>
        + Neg<Output = V>
        + Mul<Output = V>
        + ZeroLike
        + MatrixOps
        + crate::tracing_v2::operations::reshape::ReshapeOps,
> InterpretableOp<ArrayType, V> for LinearPrimitiveOp<ArrayType, V>
where
    Vec<V>: Parameterized<V, ParameterStructure: Clone + PartialEq>,
{
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        match self {
            Self::Add => AddOp.interpret(inputs),
            Self::Neg => NegOp.interpret(inputs),
            Self::MatrixTranspose => MatrixTransposeOp.interpret(inputs),
            Self::Scale { factor } => ScaleOp::new(factor.clone()).interpret(inputs),
            Self::LeftMatMul { factor } => LeftMatMulOp::new(factor.clone()).interpret(inputs),
            Self::RightMatMul { factor } => RightMatMulOp::new(factor.clone()).interpret(inputs),
            Self::Reshape { input_type, output_type } => {
                ReshapeOp::new(input_type.clone(), output_type.clone()).interpret(inputs)
            }
            Self::VMap(vmap) => vmap.interpret(inputs),
            Self::Rematerialize(remat) => remat.interpret(inputs),
            Self::Custom(op) => op.interpret(inputs),
        }
    }
}

impl<
    V: Traceable<ArrayType>
        + Add<Output = V>
        + Neg<Output = V>
        + Mul<Output = V>
        + ZeroLike
        + OneLike
        + MatrixOps
        + crate::tracing_v2::operations::reshape::ReshapeOps,
> LinearOperation<ArrayType, V> for LinearPrimitiveOp<ArrayType, V>
where
    Vec<V>: Parameterized<V, ParameterStructure: Clone + PartialEq>,
{
    fn transpose(
        &self,
        output_cotangents: &[LinearTerm<ArrayType, V>],
    ) -> Result<Vec<Option<LinearTerm<ArrayType, V>>>, TraceError> {
        match self {
            Self::Add => AddOp.transpose(output_cotangents),
            Self::Neg => NegOp.transpose(output_cotangents),
            Self::MatrixTranspose => MatrixTransposeOp.transpose(output_cotangents),
            Self::Scale { factor } => ScaleOp::new(factor.clone()).transpose(output_cotangents),
            Self::LeftMatMul { factor } => LeftMatMulOp::new(factor.clone()).transpose(output_cotangents),
            Self::RightMatMul { factor } => RightMatMulOp::new(factor.clone()).transpose(output_cotangents),
            Self::Reshape { input_type, output_type } => {
                ReshapeOp::new(input_type.clone(), output_type.clone()).transpose(output_cotangents)
            }
            Self::VMap(vmap) => vmap.transpose(output_cotangents),
            Self::Rematerialize(remat) => remat.transpose(output_cotangents),
            Self::Custom(op) => op.transpose(output_cotangents),
        }
    }
}

/// Linearized JIT replay: evaluates staged operations on [`Linearized<Tracer<V>>`] values.
///
/// For pure (non-capturing) ops, this is covered by their generic [`InterpretableOp<V>`] implementations
/// because [`JvpTracer`] already implements all necessary arithmetic, matrix, and reshape traits.
/// Capturing ops ([`ScaleOp`], [`LeftMatMulOp`], [`RightMatMulOp`]) and higher-order ops
/// ([`VMapOp`](crate::tracing_v2::operations::VMapOp),
/// [`RematerializeOp`](crate::tracing_v2::operations::RematerializeOp)) provide dedicated
/// [`InterpretableOp`] implementations that lift captured constants into the JIT trace.
///
/// [`Linearized<Tracer<V>>`]: crate::tracing_v2::linear::Linearized
/// [`ScaleOp`]: crate::tracing_v2::operations::ScaleOp
/// [`LeftMatMulOp`]: crate::tracing_v2::operations::LeftMatMulOp
/// [`RightMatMulOp`]: crate::tracing_v2::operations::RightMatMulOp
impl<
    V: Value<ArrayType>
        + Add<Output = V>
        + Mul<Output = V>
        + Neg<Output = V>
        + Sin
        + Cos
        + ZeroLike
        + OneLike
        + Parameterized<V>
        + MatrixOps
        + crate::tracing_v2::operations::reshape::ReshapeOps,
    E: Engine<
            Type = ArrayType,
            Value = V,
            TracingOperation = PrimitiveOp<ArrayType, V>,
            LinearOperation = LinearPrimitiveOp<ArrayType, V>,
        > + ?Sized
        + 'static,
> InterpretableOp<ArrayType, crate::tracing_v2::linear::Linearized<Tracer<E>>> for PrimitiveOp<ArrayType, V>
where
    V::ParameterStructure: Clone + PartialEq,
    Vec<V>: Parameterized<V, ParameterStructure: Clone + PartialEq>,
{
    fn interpret(
        &self,
        inputs: &[crate::tracing_v2::linear::Linearized<Tracer<E>>],
    ) -> Result<Vec<crate::tracing_v2::linear::Linearized<Tracer<E>>>, TraceError> {
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
            Self::Custom(op) => op.interpret(inputs),
        }
    }
}

impl<
    V: Value<ArrayType>
        + Add<Output = V>
        + Mul<Output = V>
        + Neg<Output = V>
        + Sin
        + Cos
        + ZeroLike
        + OneLike
        + Parameterized<V>
        + MatrixOps
        + crate::tracing_v2::operations::reshape::ReshapeOps,
> DifferentiableOp<ArrayType, V, LinearTerm<ArrayType, V>, PrimitiveOp<ArrayType, V>, LinearPrimitiveOp<ArrayType, V>>
    for PrimitiveOp<ArrayType, V>
where
    V::ParameterStructure: Clone + PartialEq,
    Vec<V>: Parameterized<V, ParameterStructure: Clone + PartialEq>,
{
    fn jvp(
        &self,
        engine: &dyn Engine<
            Type = ArrayType,
            Value = V,
            TracingOperation = PrimitiveOp<ArrayType, V>,
            LinearOperation = LinearPrimitiveOp<ArrayType, V>,
        >,
        inputs: &[JvpTracer<V, LinearTerm<ArrayType, V>>],
    ) -> Result<Vec<JvpTracer<V, LinearTerm<ArrayType, V>>>, TraceError> {
        match self {
            Self::Add => DifferentiableOp::<
                ArrayType,
                V,
                LinearTerm<ArrayType, V>,
                PrimitiveOp<ArrayType, V>,
                LinearPrimitiveOp<ArrayType, V>,
            >::jvp(&AddOp, engine, inputs),
            Self::Mul => DifferentiableOp::<
                ArrayType,
                V,
                LinearTerm<ArrayType, V>,
                PrimitiveOp<ArrayType, V>,
                LinearPrimitiveOp<ArrayType, V>,
            >::jvp(&MulOp, engine, inputs),
            Self::Neg => DifferentiableOp::<
                ArrayType,
                V,
                LinearTerm<ArrayType, V>,
                PrimitiveOp<ArrayType, V>,
                LinearPrimitiveOp<ArrayType, V>,
            >::jvp(&NegOp, engine, inputs),
            Self::Sin => DifferentiableOp::<
                ArrayType,
                V,
                LinearTerm<ArrayType, V>,
                PrimitiveOp<ArrayType, V>,
                LinearPrimitiveOp<ArrayType, V>,
            >::jvp(&SinOp, engine, inputs),
            Self::Cos => DifferentiableOp::<
                ArrayType,
                V,
                LinearTerm<ArrayType, V>,
                PrimitiveOp<ArrayType, V>,
                LinearPrimitiveOp<ArrayType, V>,
            >::jvp(&CosOp, engine, inputs),
            Self::Scale { factor } => DifferentiableOp::<
                ArrayType,
                V,
                LinearTerm<ArrayType, V>,
                PrimitiveOp<ArrayType, V>,
                LinearPrimitiveOp<ArrayType, V>,
            >::jvp(&ScaleOp::new(factor.clone()), engine, inputs),
            Self::MatMul => DifferentiableOp::<
                ArrayType,
                V,
                LinearTerm<ArrayType, V>,
                PrimitiveOp<ArrayType, V>,
                LinearPrimitiveOp<ArrayType, V>,
            >::jvp(&MatMulOp, engine, inputs),
            Self::MatrixTranspose => DifferentiableOp::<
                ArrayType,
                V,
                LinearTerm<ArrayType, V>,
                PrimitiveOp<ArrayType, V>,
                LinearPrimitiveOp<ArrayType, V>,
            >::jvp(&MatrixTransposeOp, engine, inputs),
            Self::LeftMatMul { factor } => DifferentiableOp::<
                ArrayType,
                V,
                LinearTerm<ArrayType, V>,
                PrimitiveOp<ArrayType, V>,
                LinearPrimitiveOp<ArrayType, V>,
            >::jvp(&LeftMatMulOp::new(factor.clone()), engine, inputs),
            Self::RightMatMul { factor } => DifferentiableOp::<
                ArrayType,
                V,
                LinearTerm<ArrayType, V>,
                PrimitiveOp<ArrayType, V>,
                LinearPrimitiveOp<ArrayType, V>,
            >::jvp(&RightMatMulOp::new(factor.clone()), engine, inputs),
            Self::Reshape { input_type, output_type } => {
                DifferentiableOp::<
                    ArrayType,
                    V,
                    LinearTerm<ArrayType, V>,
                    PrimitiveOp<ArrayType, V>,
                    LinearPrimitiveOp<ArrayType, V>,
                >::jvp(&ReshapeOp::new(input_type.clone(), output_type.clone()), engine, inputs)
            }
            Self::VMap(vmap) => Err(TraceError::HigherOrderOpFailure {
                op: "linearize_program",
                message: format!("JVP rule for staged op '{}' is not implemented", vmap.name()),
            }),
            Self::Rematerialize(remat) => DifferentiableOp::<
                ArrayType,
                V,
                LinearTerm<ArrayType, V>,
                PrimitiveOp<ArrayType, V>,
                LinearPrimitiveOp<ArrayType, V>,
            >::jvp(remat.as_ref(), engine, inputs),
            Self::Custom(op) => op.jvp(engine, inputs),
        }
    }
}

impl<V: Traceable<ArrayType> + Add<Output = V> + Mul<Output = V> + Neg<Output = V> + Sin + Cos + MatrixOps>
    VectorizableOp<ArrayType, V> for PrimitiveOp<ArrayType, V>
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
            Self::Custom(op) => op.batch(inputs),
            _ => Err(TraceError::HigherOrderOpFailure {
                op: "vectorize",
                message: format!("vectorization rule for staged op '{}' is not implemented", self.name()),
            }),
        }
    }
}
