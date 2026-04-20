//! Closed default carriers for the built-in `tracing_v2` operation set.
//!
//! [`PrimitiveOperation`] is the ordinary staged-operation carrier and [`LinearPrimitiveOperation`] is the
//! linear-only sibling used by linear programs. Both enums are zero-cost wrappers around the
//! per-primitive op types in [`crate::tracing_v2::operations`] and use the
//! [`Custom`](PrimitiveOperation::Custom) escape hatch for operations defined outside this crate.
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
        AtomId, Cos, MatrixOps, OneLike, Sin, Traceable, TracingError, Value, ZeroLike,
        batch::Batch,
        engine::Engine,
        forward::JvpTracer,
        jit::Tracer,
        linear::LinearTerm,
        operations::{
            AddOperation, CosOperation, LeftMatMulOperation, MatMulOperation, MatrixTransposeOperation, MulOperation,
            NegOperation, ReshapeOperation, RightMatMulOperation, ScaleOperation, SinOperation,
            left_matmul::left_matmul_abstract_eval, right_matmul::right_matmul_abstract_eval,
        },
    },
    types::{ArrayType, Type, Typed},
};

use super::{
    DifferentiableOperation, InterpretableOperation, LinearOperation, Operation, VectorizableOperation,
    add::{AddTracingOperation, LinearAddOperation},
    cos::CosTracingOperation,
    custom::{CustomPrimitive, CustomTracingOperation, LinearCustomOperation, LinearCustomPrimitive},
    left_matmul::{LeftMatMulTracingOperation, LinearLeftMatMulOperation},
    matmul::MatMulTracingOperation,
    matrix_transpose::{LinearMatrixTransposeOperation, MatrixTransposeTracingOperation},
    mul::MulTracingOperation,
    neg::{LinearNegOperation, NegTracingOperation},
    rematerialize::{LinearRematerializeCarrierOperation, RematerializeTracingOperation},
    reshape::{LinearReshapeOperation, ReshapeTracingOperation},
    right_matmul::{LinearRightMatMulOperation, RightMatMulTracingOperation},
    scale::{LinearScaleOperation, ScaleTracingOperation},
    sin::SinTracingOperation,
    vmap::{LinearVMapCarrierOperation, VMapTracingOperation},
};

/// Closed set of built-in staged operations.
///
/// [`PrimitiveOperation`] is the default ordinary-program carrier for `ryft-core`. Each variant is a
/// thin tag around one semantic primitive defined elsewhere in [`super`], and the carrier exists so
/// tracing entry points can store "one of the built-in operations" without resorting to trait
/// objects for the common case.
#[derive(Clone)]
pub enum PrimitiveOperation<T: Type + Display, V: Traceable<T> + Parameter> {
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
    VMap(
        Box<
            crate::tracing_v2::operations::VMapOperation<
                T,
                V,
                PrimitiveOperation<T, V>,
                LinearPrimitiveOperation<T, V>,
            >,
        >,
    ),

    /// Higher-order rematerialization boundary carrying a compiled body and optional transpose body.
    Rematerialize(
        Box<
            crate::tracing_v2::operations::RematerializeOperation<
                T,
                V,
                PrimitiveOperation<T, V>,
                LinearPrimitiveOperation<T, V>,
            >,
        >,
    ),

    /// Escape hatch for user- or crate-defined operations outside `ryft-core`.
    Custom(Arc<CustomPrimitive<T, V>>),
}

/// Closed set of operations that may appear in staged linear programs.
///
/// [`LinearPrimitiveOperation`] is the linear-program sibling of [`PrimitiveOperation`]. It contains only the
/// operations that make sense in tangent and cotangent programs plus the linearized higher-order
/// ops needed by `vmap` and rematerialization.
#[derive(Clone)]
pub enum LinearPrimitiveOperation<T: Type + Display, V: Traceable<T> + Parameter> {
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
    VMap(Box<crate::tracing_v2::operations::LinearVMapOperation<T, V, LinearPrimitiveOperation<T, V>>>),

    /// Higher-order rematerialization boundary restricted to linear bodies and transpose bodies.
    Rematerialize(
        Box<crate::tracing_v2::operations::LinearRematerializeOperation<T, V, LinearPrimitiveOperation<T, V>>>,
    ),

    /// Escape hatch for user- or crate-defined linear custom operations.
    Custom(Arc<LinearCustomPrimitive<T, V>>),
}

impl<V: Traceable<ArrayType> + 'static> LinearPrimitiveOperation<ArrayType, V> {
    /// Wraps one custom primitive in the linear-only operation universe after verifying transpose support.
    pub fn custom(primitive: CustomPrimitive<ArrayType, V>) -> Result<Self, TracingError> {
        Ok(Self::Custom(Arc::new(primitive.into_linear()?)))
    }

    /// Wraps one shared custom primitive in the linear-only operation universe after verifying transpose support.
    pub fn custom_arc(primitive: Arc<CustomPrimitive<ArrayType, V>>) -> Result<Self, TracingError> {
        Ok(Self::Custom(Arc::new(LinearCustomPrimitive::from_custom_primitive(primitive)?)))
    }
}

impl<T: Type + Display, V: Traceable<T>> AddTracingOperation<T, V> for PrimitiveOperation<T, V> {
    #[inline]
    fn add_op() -> Self {
        PrimitiveOperation::Add
    }
}

impl<T: Type + Display, V: Traceable<T>> MulTracingOperation<T, V> for PrimitiveOperation<T, V> {
    #[inline]
    fn mul_op() -> Self {
        PrimitiveOperation::Mul
    }
}

impl<T: Type + Display, V: Traceable<T>> NegTracingOperation<T, V> for PrimitiveOperation<T, V> {
    #[inline]
    fn neg_op() -> Self {
        PrimitiveOperation::Neg
    }
}

impl<T: Type + Display, V: Traceable<T>> SinTracingOperation<T, V> for PrimitiveOperation<T, V> {
    #[inline]
    fn sin_op() -> Self {
        PrimitiveOperation::Sin
    }
}

impl<T: Type + Display, V: Traceable<T>> CosTracingOperation<T, V> for PrimitiveOperation<T, V> {
    #[inline]
    fn cos_op() -> Self {
        PrimitiveOperation::Cos
    }
}

impl<T: Type + Display, V: Traceable<T>> MatMulTracingOperation<T, V> for PrimitiveOperation<T, V> {
    #[inline]
    fn matmul_op() -> Self {
        PrimitiveOperation::MatMul
    }
}

impl<T: Type + Display, V: Traceable<T>> MatrixTransposeTracingOperation<T, V> for PrimitiveOperation<T, V> {
    #[inline]
    fn matrix_transpose_op() -> Self {
        PrimitiveOperation::MatrixTranspose
    }
}

impl<T: Type + Display, V: Traceable<T>> ScaleTracingOperation<T, V> for PrimitiveOperation<T, V> {
    #[inline]
    fn scale_op(factor: V) -> Self {
        PrimitiveOperation::Scale { factor }
    }
}

impl<T: Type + Display, V: Traceable<T>> LeftMatMulTracingOperation<T, V> for PrimitiveOperation<T, V> {
    #[inline]
    fn left_matmul_op(factor: V) -> Self {
        PrimitiveOperation::LeftMatMul { factor }
    }
}

impl<T: Type + Display, V: Traceable<T>> RightMatMulTracingOperation<T, V> for PrimitiveOperation<T, V> {
    #[inline]
    fn right_matmul_op(factor: V) -> Self {
        PrimitiveOperation::RightMatMul { factor }
    }
}

impl<T: Type + Display, V: Traceable<T>> ReshapeTracingOperation<T, V> for PrimitiveOperation<T, V> {
    #[inline]
    fn reshape_op(input_type: T, output_type: T) -> Self {
        PrimitiveOperation::Reshape { input_type, output_type }
    }
}

impl<T: Type + Display, V: Traceable<T>> VMapTracingOperation<T, V, LinearPrimitiveOperation<T, V>>
    for PrimitiveOperation<T, V>
{
    #[inline]
    fn vmap_op(op: crate::tracing_v2::operations::VMapOperation<T, V, Self, LinearPrimitiveOperation<T, V>>) -> Self {
        PrimitiveOperation::VMap(Box::new(op))
    }
}

impl<T: Type + Display, V: Traceable<T>> RematerializeTracingOperation<T, V, LinearPrimitiveOperation<T, V>>
    for PrimitiveOperation<T, V>
{
    #[inline]
    fn rematerialize_op(
        op: crate::tracing_v2::operations::RematerializeOperation<T, V, Self, LinearPrimitiveOperation<T, V>>,
    ) -> Self {
        PrimitiveOperation::Rematerialize(Box::new(op))
    }
}

impl<T: Type + Display, V: Traceable<T>> CustomTracingOperation<T, V> for PrimitiveOperation<T, V> {
    #[inline]
    fn custom_op(primitive: Arc<CustomPrimitive<T, V>>) -> Self {
        PrimitiveOperation::Custom(primitive)
    }
}

impl<T: Type + Display, V: Traceable<T>> LinearAddOperation<T, V> for LinearPrimitiveOperation<T, V> {
    #[inline]
    fn linear_add_op() -> Self {
        LinearPrimitiveOperation::Add
    }
}

impl<T: Type + Display, V: Traceable<T>> LinearNegOperation<T, V> for LinearPrimitiveOperation<T, V> {
    #[inline]
    fn linear_neg_op() -> Self {
        LinearPrimitiveOperation::Neg
    }
}

impl<T: Type + Display, V: Traceable<T>> LinearMatrixTransposeOperation<T, V> for LinearPrimitiveOperation<T, V> {
    #[inline]
    fn linear_matrix_transpose_op() -> Self {
        LinearPrimitiveOperation::MatrixTranspose
    }
}

impl<T: Type + Display, V: Traceable<T>> LinearScaleOperation<T, V> for LinearPrimitiveOperation<T, V> {
    #[inline]
    fn linear_scale_op(factor: V) -> Self {
        LinearPrimitiveOperation::Scale { factor }
    }
}

impl<T: Type + Display, V: Traceable<T>> LinearLeftMatMulOperation<T, V> for LinearPrimitiveOperation<T, V> {
    #[inline]
    fn linear_left_matmul_op(factor: V) -> Self {
        LinearPrimitiveOperation::LeftMatMul { factor }
    }
}

impl<T: Type + Display, V: Traceable<T>> LinearRightMatMulOperation<T, V> for LinearPrimitiveOperation<T, V> {
    #[inline]
    fn linear_right_matmul_op(factor: V) -> Self {
        LinearPrimitiveOperation::RightMatMul { factor }
    }
}

impl<T: Type + Display, V: Traceable<T>> LinearReshapeOperation<T, V> for LinearPrimitiveOperation<T, V> {
    #[inline]
    fn linear_reshape_op(input_type: T, output_type: T) -> Self {
        LinearPrimitiveOperation::Reshape { input_type, output_type }
    }
}

impl<T: Type + Display, V: Traceable<T>> LinearVMapCarrierOperation<T, V> for LinearPrimitiveOperation<T, V> {
    #[inline]
    fn linear_vmap_op(op: crate::tracing_v2::operations::LinearVMapOperation<T, V, Self>) -> Self {
        LinearPrimitiveOperation::VMap(Box::new(op))
    }
}

impl<T: Type + Display, V: Traceable<T>> LinearRematerializeCarrierOperation<T, V> for LinearPrimitiveOperation<T, V> {
    #[inline]
    fn linear_rematerialize_op(op: crate::tracing_v2::operations::LinearRematerializeOperation<T, V, Self>) -> Self {
        LinearPrimitiveOperation::Rematerialize(Box::new(op))
    }
}

impl<T: Type + Display + 'static, V: Traceable<T> + 'static> LinearCustomOperation<T, V>
    for LinearPrimitiveOperation<T, V>
{
    #[inline]
    fn linear_custom_op(primitive: CustomPrimitive<T, V>) -> Result<Self, TracingError> {
        Ok(LinearPrimitiveOperation::Custom(Arc::new(primitive.into_linear()?)))
    }

    #[inline]
    fn linear_custom_arc_op(primitive: Arc<CustomPrimitive<T, V>>) -> Result<Self, TracingError> {
        Ok(LinearPrimitiveOperation::Custom(Arc::new(LinearCustomPrimitive::from_custom_primitive(primitive)?)))
    }
}

impl<T: Type + Display, V: Traceable<T>> Debug for PrimitiveOperation<T, V> {
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

impl<V: Traceable<ArrayType>> Display for PrimitiveOperation<ArrayType, V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Reshape { output_type, .. } => write!(formatter, "reshape{}", output_type.shape),
            _ => write!(formatter, "{}", self.name()),
        }
    }
}

impl<T: Type + Display, V: Traceable<T>> Debug for LinearPrimitiveOperation<T, V> {
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

impl<V: Traceable<ArrayType>> Display for LinearPrimitiveOperation<ArrayType, V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Reshape { output_type, .. } => write!(formatter, "reshape{}", output_type.shape),
            _ => write!(formatter, "{}", self.name()),
        }
    }
}

/// [`Operation`] for [`PrimitiveOperation`] requires NO value-type bounds Ã¢â‚¬â€ shape validation works for any `V: Traceable<ArrayType>`.
impl<V: Traceable<ArrayType>> Operation for PrimitiveOperation<ArrayType, V> {
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

    fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TracingError> {
        match self {
            Self::Add => AddOperation.abstract_eval(inputs),
            Self::Mul => MulOperation.abstract_eval(inputs),
            Self::Neg => NegOperation.abstract_eval(inputs),
            Self::Sin => SinOperation.abstract_eval(inputs),
            Self::Cos => CosOperation.abstract_eval(inputs),
            Self::MatMul => MatMulOperation.abstract_eval(inputs),
            Self::MatrixTranspose => MatrixTransposeOperation.abstract_eval(inputs),
            Self::Scale { .. } => ScaleOperation::<ArrayType, V>::abstract_eval_static(inputs),
            Self::LeftMatMul { factor } => left_matmul_abstract_eval(&Typed::r#type(factor), inputs),
            Self::RightMatMul { factor } => right_matmul_abstract_eval(&Typed::r#type(factor), inputs),
            Self::Reshape { input_type, output_type } => <ReshapeOperation as Operation>::abstract_eval(
                &ReshapeOperation::new(input_type.clone(), output_type.clone()),
                inputs,
            ),
            Self::VMap(vmap) => vmap.abstract_eval(inputs),
            Self::Rematerialize(remat) => remat.abstract_eval(inputs),
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
            Self::Scale { factor } => ScaleOperation::<ArrayType, V>::new(factor.clone()).try_simplify(
                inputs,
                is_zero_constant,
                is_one_constant,
            ),
            Self::LeftMatMul { factor } => {
                if factor.is_one() {
                    Some(inputs.to_vec())
                } else {
                    None
                }
            }
            Self::RightMatMul { factor } => {
                if factor.is_one() {
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

/// [`Operation`] for [`LinearPrimitiveOperation`] requires NO value-type bounds Ã¢â‚¬â€ shape validation works for any `V: Traceable<ArrayType>`.
impl<V: Traceable<ArrayType>> Operation for LinearPrimitiveOperation<ArrayType, V> {
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

    fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TracingError> {
        match self {
            Self::Add => AddOperation.abstract_eval(inputs),
            Self::Neg => NegOperation.abstract_eval(inputs),
            Self::MatrixTranspose => MatrixTransposeOperation.abstract_eval(inputs),
            Self::Scale { .. } => ScaleOperation::<ArrayType, V>::abstract_eval_static(inputs),
            Self::LeftMatMul { factor } => left_matmul_abstract_eval(&Typed::r#type(factor), inputs),
            Self::RightMatMul { factor } => right_matmul_abstract_eval(&Typed::r#type(factor), inputs),
            Self::Reshape { input_type, output_type } => <ReshapeOperation as Operation>::abstract_eval(
                &ReshapeOperation::new(input_type.clone(), output_type.clone()),
                inputs,
            ),
            Self::VMap(vmap) => vmap.abstract_eval(inputs),
            Self::Rematerialize(remat) => remat.abstract_eval(inputs),
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
            Self::Neg => NegOperation.try_simplify(inputs, is_zero_constant, is_one_constant),
            Self::Scale { factor } => ScaleOperation::<ArrayType, V>::new(factor.clone()).try_simplify(
                inputs,
                is_zero_constant,
                is_one_constant,
            ),
            Self::LeftMatMul { factor } => {
                if factor.is_one() {
                    Some(inputs.to_vec())
                } else {
                    None
                }
            }
            Self::RightMatMul { factor } => {
                if factor.is_one() {
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

/// [`InterpretableOperation`] for [`PrimitiveOperation`] requires the full union of value capabilities used by
/// the closed default ordinary-op carrier.
///
/// That broad union is local to [`PrimitiveOperation`] itself. The higher-level tracing APIs avoid
/// exposing it as one public value-bundle trait and instead express their requirements through the
/// specific staged op carrier bounds they actually exercise.
impl<
    'engine,
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
> InterpretableOperation<ArrayType, V> for PrimitiveOperation<ArrayType, V>
where
    Vec<V>: Parameterized<V, ParameterStructure: Clone + PartialEq>,
{
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
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
> InterpretableOperation<ArrayType, V> for LinearPrimitiveOperation<ArrayType, V>
where
    Vec<V>: Parameterized<V, ParameterStructure: Clone + PartialEq>,
{
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        match self {
            Self::Add => AddOperation.interpret(inputs),
            Self::Neg => NegOperation.interpret(inputs),
            Self::MatrixTranspose => MatrixTransposeOperation.interpret(inputs),
            Self::Scale { factor } => ScaleOperation::new(factor.clone()).interpret(inputs),
            Self::LeftMatMul { factor } => LeftMatMulOperation::new(factor.clone()).interpret(inputs),
            Self::RightMatMul { factor } => RightMatMulOperation::new(factor.clone()).interpret(inputs),
            Self::Reshape { input_type, output_type } => {
                ReshapeOperation::new(input_type.clone(), output_type.clone()).interpret(inputs)
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
> LinearOperation<ArrayType, V> for LinearPrimitiveOperation<ArrayType, V>
where
    Vec<V>: Parameterized<V, ParameterStructure: Clone + PartialEq>,
{
    fn transpose(
        &self,
        output_cotangents: &[LinearTerm<ArrayType, V>],
    ) -> Result<Vec<Option<LinearTerm<ArrayType, V>>>, TracingError> {
        match self {
            Self::Add => AddOperation.transpose(output_cotangents),
            Self::Neg => NegOperation.transpose(output_cotangents),
            Self::MatrixTranspose => MatrixTransposeOperation.transpose(output_cotangents),
            Self::Scale { factor } => ScaleOperation::new(factor.clone()).transpose(output_cotangents),
            Self::LeftMatMul { factor } => LeftMatMulOperation::new(factor.clone()).transpose(output_cotangents),
            Self::RightMatMul { factor } => RightMatMulOperation::new(factor.clone()).transpose(output_cotangents),
            Self::Reshape { input_type, output_type } => {
                ReshapeOperation::new(input_type.clone(), output_type.clone()).transpose(output_cotangents)
            }
            Self::VMap(vmap) => vmap.transpose(output_cotangents),
            Self::Rematerialize(remat) => remat.transpose(output_cotangents),
            Self::Custom(op) => op.transpose(output_cotangents),
        }
    }
}

/// Linearized JIT replay: evaluates staged operations on [`Linearized<Tracer<V>>`] values.
///
/// For pure (non-capturing) ops, this is covered by their generic [`InterpretableOperation<V>`] implementations
/// because [`JvpTracer`] already implements all necessary arithmetic, matrix, and reshape traits.
/// Capturing ops ([`ScaleOperation`], [`LeftMatMulOperation`], [`RightMatMulOperation`]) and higher-order ops
/// ([`VMapOperation`](crate::tracing_v2::operations::VMapOperation),
/// [`RematerializeOperation`](crate::tracing_v2::operations::RematerializeOperation)) provide dedicated
/// [`InterpretableOperation`] implementations that lift captured constants into the JIT trace.
///
/// [`Linearized<Tracer<V>>`]: crate::tracing_v2::linear::Linearized
/// [`ScaleOperation`]: crate::tracing_v2::operations::ScaleOperation
/// [`LeftMatMulOperation`]: crate::tracing_v2::operations::LeftMatMulOperation
/// [`RightMatMulOperation`]: crate::tracing_v2::operations::RightMatMulOperation
impl<
    'engine,
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
        + crate::tracing_v2::operations::reshape::ReshapeOps
        + 'static,
    E: Engine<
            Type = ArrayType,
            Value = V,
            TracingOperation = PrimitiveOperation<ArrayType, V>,
            LinearOperation = LinearPrimitiveOperation<ArrayType, V>,
        > + ?Sized
        + 'static,
> InterpretableOperation<ArrayType, crate::tracing_v2::linear::Linearized<Tracer<'engine, E>>>
    for PrimitiveOperation<ArrayType, V>
where
    V::ParameterStructure: Clone + PartialEq,
    Vec<V>: Parameterized<V, ParameterStructure: Clone + PartialEq>,
{
    fn interpret(
        &self,
        inputs: &[crate::tracing_v2::linear::Linearized<Tracer<'engine, E>>],
    ) -> Result<Vec<crate::tracing_v2::linear::Linearized<Tracer<'engine, E>>>, TracingError> {
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
            Self::Custom(op) => op.interpret_linearized_jit(inputs),
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
        + crate::tracing_v2::operations::reshape::ReshapeOps
        + 'static,
>
    DifferentiableOperation<
        ArrayType,
        V,
        LinearTerm<ArrayType, V>,
        PrimitiveOperation<ArrayType, V>,
        LinearPrimitiveOperation<ArrayType, V>,
    > for PrimitiveOperation<ArrayType, V>
where
    V::ParameterStructure: Clone + PartialEq,
    Vec<V>: Parameterized<V, ParameterStructure: Clone + PartialEq>,
{
    fn jvp(
        &self,
        engine: &dyn Engine<
            Type = ArrayType,
            Value = V,
            TracingOperation = PrimitiveOperation<ArrayType, V>,
            LinearOperation = LinearPrimitiveOperation<ArrayType, V>,
        >,
        inputs: &[JvpTracer<V, LinearTerm<ArrayType, V>>],
    ) -> Result<Vec<JvpTracer<V, LinearTerm<ArrayType, V>>>, TracingError> {
        match self {
            Self::Add => DifferentiableOperation::<
                ArrayType,
                V,
                LinearTerm<ArrayType, V>,
                PrimitiveOperation<ArrayType, V>,
                LinearPrimitiveOperation<ArrayType, V>,
            >::jvp(&AddOperation, engine, inputs),
            Self::Mul => DifferentiableOperation::<
                ArrayType,
                V,
                LinearTerm<ArrayType, V>,
                PrimitiveOperation<ArrayType, V>,
                LinearPrimitiveOperation<ArrayType, V>,
            >::jvp(&MulOperation, engine, inputs),
            Self::Neg => DifferentiableOperation::<
                ArrayType,
                V,
                LinearTerm<ArrayType, V>,
                PrimitiveOperation<ArrayType, V>,
                LinearPrimitiveOperation<ArrayType, V>,
            >::jvp(&NegOperation, engine, inputs),
            Self::Sin => DifferentiableOperation::<
                ArrayType,
                V,
                LinearTerm<ArrayType, V>,
                PrimitiveOperation<ArrayType, V>,
                LinearPrimitiveOperation<ArrayType, V>,
            >::jvp(&SinOperation, engine, inputs),
            Self::Cos => DifferentiableOperation::<
                ArrayType,
                V,
                LinearTerm<ArrayType, V>,
                PrimitiveOperation<ArrayType, V>,
                LinearPrimitiveOperation<ArrayType, V>,
            >::jvp(&CosOperation, engine, inputs),
            Self::Scale { factor } => DifferentiableOperation::<
                ArrayType,
                V,
                LinearTerm<ArrayType, V>,
                PrimitiveOperation<ArrayType, V>,
                LinearPrimitiveOperation<ArrayType, V>,
            >::jvp(&ScaleOperation::new(factor.clone()), engine, inputs),
            Self::MatMul => DifferentiableOperation::<
                ArrayType,
                V,
                LinearTerm<ArrayType, V>,
                PrimitiveOperation<ArrayType, V>,
                LinearPrimitiveOperation<ArrayType, V>,
            >::jvp(&MatMulOperation, engine, inputs),
            Self::MatrixTranspose => DifferentiableOperation::<
                ArrayType,
                V,
                LinearTerm<ArrayType, V>,
                PrimitiveOperation<ArrayType, V>,
                LinearPrimitiveOperation<ArrayType, V>,
            >::jvp(&MatrixTransposeOperation, engine, inputs),
            Self::LeftMatMul { factor } => DifferentiableOperation::<
                ArrayType,
                V,
                LinearTerm<ArrayType, V>,
                PrimitiveOperation<ArrayType, V>,
                LinearPrimitiveOperation<ArrayType, V>,
            >::jvp(&LeftMatMulOperation::new(factor.clone()), engine, inputs),
            Self::RightMatMul { factor } => {
                DifferentiableOperation::<
                    ArrayType,
                    V,
                    LinearTerm<ArrayType, V>,
                    PrimitiveOperation<ArrayType, V>,
                    LinearPrimitiveOperation<ArrayType, V>,
                >::jvp(&RightMatMulOperation::new(factor.clone()), engine, inputs)
            }
            Self::Reshape { input_type, output_type } => {
                DifferentiableOperation::<
                    ArrayType,
                    V,
                    LinearTerm<ArrayType, V>,
                    PrimitiveOperation<ArrayType, V>,
                    LinearPrimitiveOperation<ArrayType, V>,
                >::jvp(&ReshapeOperation::new(input_type.clone(), output_type.clone()), engine, inputs)
            }
            Self::VMap(vmap) => Err(TracingError::HigherOrderOpFailure {
                op: "linearize_program",
                message: format!("JVP rule for staged op '{}' is not implemented", vmap.name()),
            }),
            Self::Rematerialize(remat) => DifferentiableOperation::<
                ArrayType,
                V,
                LinearTerm<ArrayType, V>,
                PrimitiveOperation<ArrayType, V>,
                LinearPrimitiveOperation<ArrayType, V>,
            >::jvp(remat.as_ref(), engine, inputs),
            Self::Custom(op) => op.jvp(engine, inputs),
        }
    }
}

impl<V: Traceable<ArrayType> + Add<Output = V> + Mul<Output = V> + Neg<Output = V> + Sin + Cos + MatrixOps + 'static>
    VectorizableOperation<ArrayType, V> for PrimitiveOperation<ArrayType, V>
{
    fn batch(&self, inputs: &[Batch<V>]) -> Result<Vec<Batch<V>>, TracingError> {
        match self {
            Self::Add => AddOperation.batch(inputs),
            Self::Mul => MulOperation.batch(inputs),
            Self::Neg => NegOperation.batch(inputs),
            Self::Sin => SinOperation.batch(inputs),
            Self::Cos => CosOperation.batch(inputs),
            Self::MatMul => MatMulOperation.batch(inputs),
            Self::MatrixTranspose => MatrixTransposeOperation.batch(inputs),
            Self::Custom(op) => op.batch(inputs),
            _ => Err(TracingError::HigherOrderOpFailure {
                op: "vectorize",
                message: format!("vectorization rule for staged op '{}' is not implemented", self.name()),
            }),
        }
    }
}
