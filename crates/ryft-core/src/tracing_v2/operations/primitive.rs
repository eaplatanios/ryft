use std::fmt::{Debug, Display};
use std::ops::{Add, Mul, Neg};
use std::sync::Arc;

use crate::operations::{InterpretableOperation, Operation, OperationFormatter};
use crate::parameters::{Parameter, Parameterized};
use crate::tracing::engines::Tracer;
use crate::tracing::transposition::LinearOperation;
use crate::tracing::{AtomId, Traceable, TracingError, Value};
use crate::tracing_v2::differentiation::{Differentiable, JvpContext, JvpTracer};
use crate::tracing_v2::operations::constants::{
    One, OneLike, OneLikeOperation, OneOperation, Zero, ZeroLike, ZeroLikeOperation, ZeroOperation,
};
use crate::tracing_v2::operations::control_flow::{ConditionOperation, ControlFlowValue, WhileOperation};
use crate::tracing_v2::operations::left_matmul::left_matmul_abstract_eval;
use crate::tracing_v2::operations::right_matmul::right_matmul_abstract_eval;
use crate::tracing_v2::operations::{
    AddOperation, CosOperation, LeftMatMulOperation, MatMulOperation, MatrixTransposeOperation, MulOperation,
    NegOperation, ReshapeOperation, RightMatMulOperation, ScaleOperation, SinOperation,
};
use crate::tracing_v2::{
    Cos, DifferentiableOperation, DifferentiableTracingEngine, LinearizableEngine, MatrixOps, Sin,
};
use crate::types::{ArrayType, DataType, Shape, Type, TypeError, Typed};

use super::add::SupportsAdd;
use super::constants::{SupportsOne, SupportsOneLike, SupportsZero, SupportsZeroLike};
use super::cos::SupportsCos;
use super::custom::{CustomPrimitive, LinearCustomPrimitive, SupportsCustom, SupportsLinearCustom};
use super::left_matmul::SupportsLeftMatMul;
use super::matmul::SupportsMatMul;
use super::matrix_transpose::SupportsMatrixTranspose;
use super::mul::SupportsMul;
use super::neg::SupportsNeg;
use super::rematerialize::{SupportsLinearRematerialize, SupportsRematerialize};
use super::reshape::SupportsReshape;
use super::right_matmul::SupportsRightMatMul;
use super::scale::SupportsScale;
use super::sin::SupportsSin;

/// Default closed carrier for ordinary staged programs.
///
/// [`ArrayOperation`] is the reusable array operation enum for core tests and backend crates that do not need a fully
/// custom carrier. Most variants are thin tags around one semantic primitive defined elsewhere in [`super`]. The
/// [`Custom`](Self::Custom) variant is the explicit escape hatch for operations outside that default set, so the
/// carrier remains closed for normal dispatch while still allowing user- or backend-defined extensions.
#[derive(Clone)]
pub enum ArrayOperation<V, T = ArrayType>
where
    T: Type + PartialEq,
    V: Traceable<T> + Parameter,
{
    /// Typed zero with no inputs and one output, carrying a [`ZeroOperation`].
    Zero(ZeroOperation<T>),

    /// Typed one with no inputs and one output, carrying a [`OneOperation`].
    One(OneOperation<T>),

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

    /// Matrix transposition.
    Transpose,

    /// Scalar or tensor scaling by a captured factor.
    Scale { factor: V },

    /// Reshape from one shape to another.
    Reshape { input_shape: Shape, output_shape: Shape },

    /// Higher-order rematerialization boundary carrying a compiled body and optional transpose body.
    Rematerialize(
        Box<
            crate::tracing_v2::operations::RematerializeOperation<
                T,
                V,
                ArrayOperation<V, T>,
                LinearArrayOperation<V, T>,
            >,
        >,
    ),

    /// Higher-order conditional carrying true and false branch programs.
    Condition(Box<ConditionOperation<V, ArrayOperation<V, T>, T>>),

    /// Higher-order while loop carrying condition and body programs.
    While(Box<WhileOperation<V, ArrayOperation<V, T>, T>>),

    /// Escape hatch for user- or crate-defined operations outside `ryft-core`.
    Custom(Arc<CustomPrimitive<T, V>>),
}

/// Default closed carrier for staged linear programs.
///
/// [`LinearArrayOperation`] is the linear-program sibling of [`ArrayOperation`]. It contains
/// operations that can appear in tangent and cotangent programs, including captured-factor linear
/// maps such as [`LeftMatMul`](Self::LeftMatMul) and [`RightMatMul`](Self::RightMatMul), and the
/// linearized higher-order operations needed by rematerialization and control flow.
#[derive(Clone)]
pub enum LinearArrayOperation<V, T = ArrayType>
where
    T: Type + PartialEq,
    V: Traceable<T> + Parameter,
{
    /// Typed zero with no inputs and one output, carrying a [`ZeroOperation`].
    ///
    /// Emitted by the transpose pass at the boundary of pullbacks for primal inputs that receive
    /// no cotangent contribution from any output. Interpreting it requires
    /// [`Zero<ArrayType>`](crate::tracing_v2::operations::constants::Zero) on the value type;
    /// staged tracer programs must materialize these ops away before being interpreted.
    Zero(ZeroOperation<T>),

    /// Typed one with no inputs and one output, carrying a [`OneOperation`].
    One(OneOperation<T>),

    /// Exemplar-derived zero map.
    ZeroLike,

    /// Exemplar-derived one map.
    OneLike,

    /// Elementwise addition.
    Add,

    /// Elementwise negation.
    Neg,

    /// Matrix transposition.
    Transpose,

    /// Scalar or tensor scaling by a captured factor.
    Scale { factor: V },

    /// Left matrix multiplication by a captured factor: `factor @ input`.
    LeftMatMul { factor: V },

    /// Right matrix multiplication by a captured factor: `input @ factor`.
    RightMatMul { factor: V },

    /// Reshape from one shape to another.
    Reshape { input_shape: Shape, output_shape: Shape },

    /// Higher-order rematerialization boundary restricted to linear bodies and transpose bodies.
    Rematerialize(Box<crate::tracing_v2::operations::LinearRematerializeOperation<T, V, LinearArrayOperation<V, T>>>),

    /// Higher-order conditional restricted to linear branch programs.
    Condition(Box<ConditionOperation<V, LinearArrayOperation<V, T>, T>>),

    /// Higher-order while loop restricted to linear condition and body programs.
    While(Box<WhileOperation<V, LinearArrayOperation<V, T>, T>>),

    /// Escape hatch for user- or crate-defined linear custom operations.
    Custom(Arc<LinearCustomPrimitive<T, V>>),
}

impl<T, V> LinearArrayOperation<V, T>
where
    T: Type + PartialEq + 'static,
    V: Traceable<T> + Parameter + 'static,
{
    /// Wraps one custom primitive in the linear-only operation universe after verifying transpose support.
    pub fn custom(primitive: CustomPrimitive<T, V>) -> Result<Self, TracingError> {
        Ok(Self::Custom(Arc::new(primitive.into_linear()?)))
    }

    /// Wraps one shared custom primitive in the linear-only operation universe after verifying transpose support.
    pub fn custom_arc(primitive: Arc<CustomPrimitive<T, V>>) -> Result<Self, TracingError> {
        Ok(Self::Custom(Arc::new(LinearCustomPrimitive::from_custom_primitive(primitive)?)))
    }
}

/// Closed scalar operation carrier for ordinary staged scalar programs.
///
/// [`ScalarOperation`] is intentionally limited to operations that are valid for scalar
/// [`DataType`] metadata. Array-only primitives such as reshaping and matrix multiplication remain
/// available as standalone operations and through array backend carriers, but they are not variants
/// of this enum.
#[derive(Clone)]
pub enum ScalarOperation<V>
where
    V: Traceable<DataType> + Parameter,
{
    /// Typed scalar zero with no inputs and one output.
    Zero(ZeroOperation<DataType>),

    /// Typed scalar one with no inputs and one output.
    One(OneOperation<DataType>),

    /// Scalar exemplar-derived zero.
    ZeroLike,

    /// Scalar exemplar-derived one.
    OneLike,

    /// Scalar addition.
    Add,

    /// Scalar multiplication.
    Mul,

    /// Scalar negation.
    Neg,

    /// Scalar sine.
    Sin,

    /// Scalar cosine.
    Cos,

    /// Scalar scaling by a captured factor.
    Scale { factor: V },

    /// Escape hatch for user- or crate-defined scalar operations.
    Custom(Arc<CustomPrimitive<DataType, V>>),
}

/// Closed scalar operation carrier for staged linear scalar programs.
#[derive(Clone)]
pub enum LinearScalarOperation<V>
where
    V: Traceable<DataType> + Parameter,
{
    /// Typed scalar zero with no inputs and one output.
    Zero(ZeroOperation<DataType>),

    /// Typed scalar one with no inputs and one output.
    One(OneOperation<DataType>),

    /// Scalar exemplar-derived zero map.
    ZeroLike,

    /// Scalar exemplar-derived one map.
    OneLike,

    /// Scalar addition.
    Add,

    /// Scalar negation.
    Neg,

    /// Scalar scaling by a captured factor.
    Scale { factor: V },

    /// Escape hatch for user- or crate-defined linear scalar operations.
    Custom(Arc<LinearCustomPrimitive<DataType, V>>),
}

impl<V> LinearScalarOperation<V>
where
    V: Traceable<DataType> + Parameter + 'static,
{
    /// Wraps one custom primitive in the scalar linear operation universe after verifying transpose support.
    pub fn custom(primitive: CustomPrimitive<DataType, V>) -> Result<Self, TracingError> {
        Ok(Self::Custom(Arc::new(primitive.into_linear()?)))
    }

    /// Wraps one shared custom primitive in the scalar linear operation universe after verifying transpose support.
    pub fn custom_arc(primitive: Arc<CustomPrimitive<DataType, V>>) -> Result<Self, TracingError> {
        Ok(Self::Custom(Arc::new(LinearCustomPrimitive::from_custom_primitive(primitive)?)))
    }
}

impl<V> SupportsAdd<DataType, V> for ScalarOperation<V>
where
    V: Traceable<DataType> + Parameter,
{
    #[inline]
    fn add_operation() -> Self {
        Self::Add
    }
}

impl<V> SupportsMul<DataType, V> for ScalarOperation<V>
where
    V: Traceable<DataType> + Parameter,
{
    #[inline]
    fn mul_operation() -> Self {
        Self::Mul
    }
}

impl<V> SupportsNeg<DataType, V> for ScalarOperation<V>
where
    V: Traceable<DataType> + Parameter,
{
    #[inline]
    fn neg_operation() -> Self {
        Self::Neg
    }
}

impl<V> SupportsSin<DataType, V> for ScalarOperation<V>
where
    V: Traceable<DataType> + Parameter,
{
    #[inline]
    fn sin_operation() -> Self {
        Self::Sin
    }
}

impl<V> SupportsCos<DataType, V> for ScalarOperation<V>
where
    V: Traceable<DataType> + Parameter,
{
    #[inline]
    fn cos_operation() -> Self {
        Self::Cos
    }
}

impl<V> SupportsZero<DataType, V> for ScalarOperation<V>
where
    V: Traceable<DataType> + Parameter,
{
    #[inline]
    fn zero_operation(r#type: DataType) -> Self {
        Self::Zero(ZeroOperation::new(r#type))
    }

    #[inline]
    fn as_zero(&self) -> Option<&DataType> {
        match self {
            Self::Zero(zero) => Some(zero.output_type()),
            _ => None,
        }
    }
}

impl<V> SupportsOne<DataType, V> for ScalarOperation<V>
where
    V: Traceable<DataType> + Parameter,
{
    #[inline]
    fn one_operation(r#type: DataType) -> Self {
        Self::One(OneOperation::new(r#type))
    }
}

impl<V> SupportsZeroLike<DataType, V> for ScalarOperation<V>
where
    V: Traceable<DataType> + Parameter,
{
    #[inline]
    fn zero_like_operation() -> Self {
        Self::ZeroLike
    }
}

impl<V> SupportsOneLike<DataType, V> for ScalarOperation<V>
where
    V: Traceable<DataType> + Parameter,
{
    #[inline]
    fn one_like_operation() -> Self {
        Self::OneLike
    }
}

impl<V> SupportsScale<DataType, V> for ScalarOperation<V>
where
    V: Traceable<DataType> + Parameter,
{
    #[inline]
    fn scale_operation(factor: V) -> Self {
        Self::Scale { factor }
    }
}

impl<V> SupportsCustom<DataType, V> for ScalarOperation<V>
where
    V: Traceable<DataType> + Parameter,
{
    #[inline]
    fn custom_operation(primitive: Arc<CustomPrimitive<DataType, V>>) -> Self {
        Self::Custom(primitive)
    }
}

impl<V> SupportsAdd<DataType, V> for LinearScalarOperation<V>
where
    V: Traceable<DataType> + Parameter,
{
    #[inline]
    fn add_operation() -> Self {
        Self::Add
    }
}

impl<V> SupportsZero<DataType, V> for LinearScalarOperation<V>
where
    V: Traceable<DataType> + Parameter,
{
    #[inline]
    fn zero_operation(r#type: DataType) -> Self {
        Self::Zero(ZeroOperation::new(r#type))
    }

    #[inline]
    fn as_zero(&self) -> Option<&DataType> {
        match self {
            Self::Zero(zero) => Some(zero.output_type()),
            _ => None,
        }
    }
}

impl<V> SupportsOne<DataType, V> for LinearScalarOperation<V>
where
    V: Traceable<DataType> + Parameter,
{
    #[inline]
    fn one_operation(r#type: DataType) -> Self {
        Self::One(OneOperation::new(r#type))
    }
}

impl<V> SupportsZeroLike<DataType, V> for LinearScalarOperation<V>
where
    V: Traceable<DataType> + Parameter,
{
    #[inline]
    fn zero_like_operation() -> Self {
        Self::ZeroLike
    }
}

impl<V> SupportsOneLike<DataType, V> for LinearScalarOperation<V>
where
    V: Traceable<DataType> + Parameter,
{
    #[inline]
    fn one_like_operation() -> Self {
        Self::OneLike
    }
}

impl<V> SupportsNeg<DataType, V> for LinearScalarOperation<V>
where
    V: Traceable<DataType> + Parameter,
{
    #[inline]
    fn neg_operation() -> Self {
        Self::Neg
    }
}

impl<V> SupportsScale<DataType, V> for LinearScalarOperation<V>
where
    V: Traceable<DataType> + Parameter,
{
    #[inline]
    fn scale_operation(factor: V) -> Self {
        Self::Scale { factor }
    }
}

impl<V> SupportsLinearCustom<DataType, V> for LinearScalarOperation<V>
where
    V: Traceable<DataType> + Parameter + 'static,
{
    #[inline]
    fn custom_operation(primitive: CustomPrimitive<DataType, V>) -> Result<Self, TracingError> {
        Ok(Self::Custom(Arc::new(primitive.into_linear()?)))
    }

    #[inline]
    fn custom_arc_operation(primitive: Arc<CustomPrimitive<DataType, V>>) -> Result<Self, TracingError> {
        Ok(Self::Custom(Arc::new(LinearCustomPrimitive::from_custom_primitive(primitive)?)))
    }
}

impl<T, V> SupportsAdd<T, V> for ArrayOperation<V, T>
where
    T: Type + PartialEq,
    V: Traceable<T> + Parameter,
{
    #[inline]
    fn add_operation() -> Self {
        ArrayOperation::Add
    }
}

impl<T, V> SupportsMul<T, V> for ArrayOperation<V, T>
where
    T: Type + PartialEq,
    V: Traceable<T> + Parameter,
{
    #[inline]
    fn mul_operation() -> Self {
        ArrayOperation::Mul
    }
}

impl<T, V> SupportsNeg<T, V> for ArrayOperation<V, T>
where
    T: Type + PartialEq,
    V: Traceable<T> + Parameter,
{
    #[inline]
    fn neg_operation() -> Self {
        ArrayOperation::Neg
    }
}

impl<T, V> SupportsSin<T, V> for ArrayOperation<V, T>
where
    T: Type + PartialEq,
    V: Traceable<T> + Parameter,
{
    #[inline]
    fn sin_operation() -> Self {
        ArrayOperation::Sin
    }
}

impl<T, V> SupportsCos<T, V> for ArrayOperation<V, T>
where
    T: Type + PartialEq,
    V: Traceable<T> + Parameter,
{
    #[inline]
    fn cos_operation() -> Self {
        ArrayOperation::Cos
    }
}

impl<T, V> SupportsZero<T, V> for ArrayOperation<V, T>
where
    T: Type + PartialEq,
    V: Traceable<T> + Parameter,
{
    #[inline]
    fn zero_operation(r#type: T) -> Self {
        ArrayOperation::Zero(ZeroOperation::new(r#type))
    }

    #[inline]
    fn as_zero(&self) -> Option<&T> {
        match self {
            Self::Zero(zero) => Some(zero.output_type()),
            _ => None,
        }
    }
}

impl<T, V> SupportsOne<T, V> for ArrayOperation<V, T>
where
    T: Type + PartialEq,
    V: Traceable<T> + Parameter,
{
    #[inline]
    fn one_operation(r#type: T) -> Self {
        ArrayOperation::One(OneOperation::new(r#type))
    }
}

impl<T, V> SupportsZeroLike<T, V> for ArrayOperation<V, T>
where
    T: Type + PartialEq,
    V: Traceable<T> + Parameter,
{
    #[inline]
    fn zero_like_operation() -> Self {
        ArrayOperation::ZeroLike
    }
}

impl<T, V> SupportsOneLike<T, V> for ArrayOperation<V, T>
where
    T: Type + PartialEq,
    V: Traceable<T> + Parameter,
{
    #[inline]
    fn one_like_operation() -> Self {
        ArrayOperation::OneLike
    }
}

impl<V: Traceable<ArrayType> + Parameter> SupportsMatMul<ArrayType, V> for ArrayOperation<V> {
    #[inline]
    fn matmul_operation() -> Self {
        ArrayOperation::MatrixMultiply
    }
}

impl<V: Traceable<ArrayType> + Parameter> SupportsMatrixTranspose<ArrayType, V> for ArrayOperation<V> {
    #[inline]
    fn matrix_transpose_operation() -> Self {
        ArrayOperation::Transpose
    }
}

impl<T, V> SupportsScale<T, V> for ArrayOperation<V, T>
where
    T: Type + PartialEq,
    V: Traceable<T> + Parameter,
{
    #[inline]
    fn scale_operation(factor: V) -> Self {
        ArrayOperation::Scale { factor }
    }
}

impl<V: Traceable<ArrayType> + Parameter> SupportsReshape<ArrayType, V> for ArrayOperation<V> {
    #[inline]
    fn reshape_operation(input_shape: Shape, output_shape: Shape) -> Self {
        ArrayOperation::Reshape { input_shape, output_shape }
    }
}

impl<V: Traceable<ArrayType> + Parameter> SupportsRematerialize<ArrayType, V, LinearArrayOperation<V>>
    for ArrayOperation<V>
{
    #[inline]
    fn rematerialize_operation(
        op: crate::tracing_v2::operations::RematerializeOperation<ArrayType, V, Self, LinearArrayOperation<V>>,
    ) -> Self {
        ArrayOperation::Rematerialize(Box::new(op))
    }
}

impl<T, V> SupportsCustom<T, V> for ArrayOperation<V, T>
where
    T: Type + PartialEq,
    V: Traceable<T> + Parameter,
{
    #[inline]
    fn custom_operation(primitive: Arc<CustomPrimitive<T, V>>) -> Self {
        ArrayOperation::Custom(primitive)
    }
}

impl<T, V> SupportsAdd<T, V> for LinearArrayOperation<V, T>
where
    T: Type + PartialEq,
    V: Traceable<T> + Parameter,
{
    #[inline]
    fn add_operation() -> Self {
        LinearArrayOperation::Add
    }
}

impl<T, V> SupportsZero<T, V> for LinearArrayOperation<V, T>
where
    T: Type + PartialEq,
    V: Traceable<T> + Parameter,
{
    #[inline]
    fn zero_operation(r#type: T) -> Self {
        LinearArrayOperation::Zero(ZeroOperation::new(r#type))
    }

    #[inline]
    fn as_zero(&self) -> Option<&T> {
        match self {
            Self::Zero(zero) => Some(zero.output_type()),
            _ => None,
        }
    }
}

impl<T, V> SupportsOne<T, V> for LinearArrayOperation<V, T>
where
    T: Type + PartialEq,
    V: Traceable<T> + Parameter,
{
    #[inline]
    fn one_operation(r#type: T) -> Self {
        LinearArrayOperation::One(OneOperation::new(r#type))
    }
}

impl<T, V> SupportsZeroLike<T, V> for LinearArrayOperation<V, T>
where
    T: Type + PartialEq,
    V: Traceable<T> + Parameter,
{
    #[inline]
    fn zero_like_operation() -> Self {
        LinearArrayOperation::ZeroLike
    }
}

impl<T, V> SupportsOneLike<T, V> for LinearArrayOperation<V, T>
where
    T: Type + PartialEq,
    V: Traceable<T> + Parameter,
{
    #[inline]
    fn one_like_operation() -> Self {
        LinearArrayOperation::OneLike
    }
}

impl<T, V> SupportsNeg<T, V> for LinearArrayOperation<V, T>
where
    T: Type + PartialEq,
    V: Traceable<T> + Parameter,
{
    #[inline]
    fn neg_operation() -> Self {
        LinearArrayOperation::Neg
    }
}

impl<V: Traceable<ArrayType> + Parameter> SupportsMatrixTranspose<ArrayType, V> for LinearArrayOperation<V> {
    #[inline]
    fn matrix_transpose_operation() -> Self {
        LinearArrayOperation::Transpose
    }
}

impl<T, V> SupportsScale<T, V> for LinearArrayOperation<V, T>
where
    T: Type + PartialEq,
    V: Traceable<T> + Parameter,
{
    #[inline]
    fn scale_operation(factor: V) -> Self {
        LinearArrayOperation::Scale { factor }
    }
}

impl<V: Traceable<ArrayType> + Parameter> SupportsLeftMatMul<ArrayType, V> for LinearArrayOperation<V> {
    #[inline]
    fn left_matmul_operation(factor: V) -> Self {
        LinearArrayOperation::LeftMatMul { factor }
    }
}

impl<V: Traceable<ArrayType> + Parameter> SupportsRightMatMul<ArrayType, V> for LinearArrayOperation<V> {
    #[inline]
    fn right_matmul_operation(factor: V) -> Self {
        LinearArrayOperation::RightMatMul { factor }
    }
}

impl<V: Traceable<ArrayType> + Parameter> SupportsReshape<ArrayType, V> for LinearArrayOperation<V> {
    #[inline]
    fn reshape_operation(input_shape: Shape, output_shape: Shape) -> Self {
        LinearArrayOperation::Reshape { input_shape, output_shape }
    }
}

impl<V: Traceable<ArrayType> + Parameter> SupportsLinearRematerialize<ArrayType, V> for LinearArrayOperation<V> {
    #[inline]
    fn rematerialize_operation(
        op: crate::tracing_v2::operations::LinearRematerializeOperation<ArrayType, V, Self>,
    ) -> Self {
        LinearArrayOperation::Rematerialize(Box::new(op))
    }
}

impl<V: Traceable<ArrayType> + Parameter> From<ConditionOperation<V, LinearArrayOperation<V>>>
    for LinearArrayOperation<V>
{
    #[inline]
    fn from(op: ConditionOperation<V, LinearArrayOperation<V>>) -> Self {
        LinearArrayOperation::Condition(Box::new(op))
    }
}

impl<T, V> SupportsLinearCustom<T, V> for LinearArrayOperation<V, T>
where
    T: Type + PartialEq + 'static,
    V: Traceable<T> + Parameter + 'static,
{
    #[inline]
    fn custom_operation(primitive: CustomPrimitive<T, V>) -> Result<Self, TracingError> {
        Ok(LinearArrayOperation::Custom(Arc::new(primitive.into_linear()?)))
    }

    #[inline]
    fn custom_arc_operation(primitive: Arc<CustomPrimitive<T, V>>) -> Result<Self, TracingError> {
        Ok(LinearArrayOperation::Custom(Arc::new(LinearCustomPrimitive::from_custom_primitive(primitive)?)))
    }
}

impl<T, V> ArrayOperation<V, T>
where
    T: Type + PartialEq,
    V: Traceable<T> + Parameter,
{
    #[inline]
    fn operation_name(&self) -> &'static str {
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
            Self::Rematerialize(_) => "rematerialize",
            Self::Condition(_) => "condition",
            Self::While(_) => "while",
            Self::Custom(op) => op.name(),
        }
    }
}

impl<T, V> LinearArrayOperation<V, T>
where
    T: Type + PartialEq,
    V: Traceable<T> + Parameter,
{
    #[inline]
    fn operation_name(&self) -> &'static str {
        match self {
            Self::Zero(zero) => zero.name(),
            Self::One(one) => one.name(),
            Self::ZeroLike => "zero_like",
            Self::OneLike => "one_like",
            Self::Add => "add",
            Self::Neg => "neg",
            Self::Transpose => "matrix_transpose",
            Self::Scale { .. } => "scale",
            Self::LeftMatMul { .. } => "left_matmul",
            Self::RightMatMul { .. } => "right_matmul",
            Self::Reshape { .. } => "reshape",
            Self::Rematerialize(_) => "rematerialize",
            Self::Condition(_) => "condition",
            Self::While(_) => "while",
            Self::Custom(op) => op.name(),
        }
    }
}

impl<V> ScalarOperation<V>
where
    V: Traceable<DataType> + Parameter,
{
    #[inline]
    fn operation_name(&self) -> &'static str {
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
            Self::Scale { .. } => "scale",
            Self::Custom(op) => op.name(),
        }
    }
}

impl<V> LinearScalarOperation<V>
where
    V: Traceable<DataType> + Parameter,
{
    #[inline]
    fn operation_name(&self) -> &'static str {
        match self {
            Self::Zero(zero) => zero.name(),
            Self::One(one) => one.name(),
            Self::ZeroLike => "zero_like",
            Self::OneLike => "one_like",
            Self::Add => "add",
            Self::Neg => "neg",
            Self::Scale { .. } => "scale",
            Self::Custom(op) => op.name(),
        }
    }
}

impl<V> Debug for ScalarOperation<V>
where
    V: Traceable<DataType> + Parameter,
{
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
            Self::Scale { .. } => write!(formatter, "Scale"),
            Self::Custom(op) => Debug::fmt(op.as_ref(), formatter),
        }
    }
}

impl<V> Display for ScalarOperation<V>
where
    V: Traceable<DataType> + Parameter,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.operation_name())
    }
}

impl<V> Debug for LinearScalarOperation<V>
where
    V: Traceable<DataType> + Parameter,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Zero(zero) => Debug::fmt(zero, formatter),
            Self::One(one) => Debug::fmt(one, formatter),
            Self::ZeroLike => write!(formatter, "ZeroLike"),
            Self::OneLike => write!(formatter, "OneLike"),
            Self::Add => write!(formatter, "Add"),
            Self::Neg => write!(formatter, "Neg"),
            Self::Scale { .. } => write!(formatter, "Scale"),
            Self::Custom(op) => Debug::fmt(op.as_ref(), formatter),
        }
    }
}

impl<V> Display for LinearScalarOperation<V>
where
    V: Traceable<DataType> + Parameter,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.operation_name())
    }
}

impl<T, V> Debug for ArrayOperation<V, T>
where
    T: Type + PartialEq,
    V: Traceable<T> + Parameter,
{
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
            Self::Custom(op) => Debug::fmt(op.as_ref(), formatter),
        }
    }
}

impl<T, V> Display for ArrayOperation<V, T>
where
    T: Type + PartialEq,
    V: Traceable<T> + Parameter,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Reshape { output_shape, .. } => write!(formatter, "reshape{output_shape}"),
            _ => write!(formatter, "{}", self.operation_name()),
        }
    }
}

impl<T, V> Debug for LinearArrayOperation<V, T>
where
    T: Type + PartialEq,
    V: Traceable<T> + Parameter,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Zero(zero) => Debug::fmt(zero, formatter),
            Self::One(one) => Debug::fmt(one, formatter),
            Self::ZeroLike => write!(formatter, "ZeroLike"),
            Self::OneLike => write!(formatter, "OneLike"),
            Self::Add => write!(formatter, "Add"),
            Self::Neg => write!(formatter, "Neg"),
            Self::Transpose => write!(formatter, "Transpose"),
            Self::Scale { .. } => write!(formatter, "Scale"),
            Self::LeftMatMul { .. } => write!(formatter, "LeftMatMul"),
            Self::RightMatMul { .. } => write!(formatter, "RightMatMul"),
            Self::Reshape { input_shape, output_shape } => {
                write!(formatter, "Reshape({input_shape} -> {output_shape})")
            }
            Self::Rematerialize(remat) => Debug::fmt(remat, formatter),
            Self::Condition(condition) => Debug::fmt(condition, formatter),
            Self::While(while_operation) => Debug::fmt(while_operation, formatter),
            Self::Custom(op) => Debug::fmt(op.as_ref(), formatter),
        }
    }
}

impl<T, V> Display for LinearArrayOperation<V, T>
where
    T: Type + PartialEq,
    V: Traceable<T> + Parameter,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Reshape { output_shape, .. } => write!(formatter, "reshape{output_shape}"),
            _ => write!(formatter, "{}", self.operation_name()),
        }
    }
}

fn unsupported_scalar_metadata_operation(operation_name: &'static str) -> TypeError {
    TypeError { message: format!("{operation_name} is not supported for scalar data type metadata") }
}

impl<V: Traceable<DataType> + Parameter> Operation<DataType> for ScalarOperation<V> {
    fn name(&self) -> &'static str {
        self.operation_name()
    }

    fn infer_output_types(&self, input_types: &[DataType]) -> Result<Vec<DataType>, TypeError> {
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
            Self::Scale { .. } => ScaleOperation::<DataType, V>::abstract_eval_static(input_types),
            Self::Custom(op) => op.infer_output_types(input_types),
        }
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        match self {
            Self::Zero(zero) => zero.render(formatter, indentation),
            Self::One(one) => one.render(formatter, indentation),
            Self::Scale { factor } => OperationFormatter::new(formatter, indentation, self.operation_name())?
                .bracketed(|operation| operation.field("factor", factor)),
            Self::Custom(op) => op.render(formatter, indentation),
            _ => Display::fmt(self, formatter),
        }
    }
}

impl<V: Traceable<DataType> + Parameter> Operation<DataType> for LinearScalarOperation<V> {
    fn name(&self) -> &'static str {
        self.operation_name()
    }

    fn infer_output_types(&self, input_types: &[DataType]) -> Result<Vec<DataType>, TypeError> {
        match self {
            Self::Zero(zero) => zero.infer_output_types(input_types),
            Self::One(one) => one.infer_output_types(input_types),
            Self::ZeroLike => ZeroLikeOperation.infer_output_types(input_types),
            Self::OneLike => OneLikeOperation.infer_output_types(input_types),
            Self::Add => AddOperation.infer_output_types(input_types),
            Self::Neg => NegOperation.infer_output_types(input_types),
            Self::Scale { .. } => ScaleOperation::<DataType, V>::abstract_eval_static(input_types),
            Self::Custom(op) => op.infer_output_types(input_types),
        }
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        match self {
            Self::Zero(zero) => zero.render(formatter, indentation),
            Self::One(one) => one.render(formatter, indentation),
            Self::Scale { factor } => OperationFormatter::new(formatter, indentation, self.operation_name())?
                .bracketed(|operation| operation.field("factor", factor)),
            Self::Custom(op) => op.render(formatter, indentation),
            _ => Display::fmt(self, formatter),
        }
    }
}

impl<V: Traceable<ArrayType> + Parameter> Operation<ArrayType> for ArrayOperation<V> {
    fn name(&self) -> &'static str {
        self.operation_name()
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
            Self::Scale { .. } => ScaleOperation::<ArrayType, V>::abstract_eval_static(input_types),
            Self::Reshape { input_shape, output_shape } => {
                ReshapeOperation::new(input_shape.clone(), output_shape.clone()).infer_output_types(input_types)
            }
            Self::Rematerialize(remat) => remat.infer_output_types(input_types),
            Self::Condition(condition) => condition.infer_output_types(input_types),
            Self::While(while_operation) => while_operation.infer_output_types(input_types),
            Self::Custom(op) => op.infer_output_types(input_types),
        }
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        match self {
            Self::Zero(zero) => zero.render(formatter, indentation),
            Self::One(one) => one.render(formatter, indentation),
            Self::Reshape { input_shape, output_shape } => {
                ReshapeOperation::new(input_shape.clone(), output_shape.clone()).render(formatter, indentation)
            }
            Self::Scale { factor } => OperationFormatter::new(formatter, indentation, self.operation_name())?
                .bracketed(|operation| operation.field("factor", factor)),
            Self::Rematerialize(remat) => remat.render(formatter, indentation),
            Self::Condition(condition) => condition.render(formatter, indentation),
            Self::While(while_operation) => while_operation.render(formatter, indentation),
            Self::Custom(op) => op.render(formatter, indentation),
            _ => Display::fmt(self, formatter),
        }
    }
}

impl<V: Traceable<DataType> + Parameter> Operation<DataType> for ArrayOperation<V, DataType> {
    fn name(&self) -> &'static str {
        self.operation_name()
    }

    fn infer_output_types(&self, input_types: &[DataType]) -> Result<Vec<DataType>, TypeError> {
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
            Self::Scale { .. } => ScaleOperation::<DataType, V>::abstract_eval_static(input_types),
            Self::Custom(op) => op.infer_output_types(input_types),
            Self::MatrixMultiply
            | Self::Transpose
            | Self::Reshape { .. }
            | Self::Rematerialize(_)
            | Self::Condition(_)
            | Self::While(_) => Err(unsupported_scalar_metadata_operation(self.operation_name())),
        }
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        match self {
            Self::Zero(zero) => zero.render(formatter, indentation),
            Self::One(one) => one.render(formatter, indentation),
            Self::Reshape { input_shape, output_shape } => {
                ReshapeOperation::new(input_shape.clone(), output_shape.clone()).render(formatter, indentation)
            }
            Self::Scale { factor } => OperationFormatter::new(formatter, indentation, self.operation_name())?
                .bracketed(|operation| operation.field("factor", factor)),
            Self::Rematerialize(remat) => remat.render(formatter, indentation),
            Self::Condition(condition) => condition.render(formatter, indentation),
            Self::While(while_operation) => while_operation.render(formatter, indentation),
            Self::Custom(op) => op.render(formatter, indentation),
            _ => Display::fmt(self, formatter),
        }
    }
}

impl<V: Traceable<ArrayType> + Parameter> Operation<ArrayType> for LinearArrayOperation<V> {
    fn name(&self) -> &'static str {
        self.operation_name()
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        match self {
            Self::Zero(zero) => zero.infer_output_types(input_types),
            Self::One(one) => one.infer_output_types(input_types),
            Self::ZeroLike => ZeroLikeOperation.infer_output_types(input_types),
            Self::OneLike => OneLikeOperation.infer_output_types(input_types),
            Self::Add => AddOperation.infer_output_types(input_types),
            Self::Neg => NegOperation.infer_output_types(input_types),
            Self::Transpose => MatrixTransposeOperation.infer_output_types(input_types),
            Self::Scale { .. } => ScaleOperation::<ArrayType, V>::abstract_eval_static(input_types),
            Self::LeftMatMul { factor } => {
                let factor_type = <V as Typed<ArrayType>>::r#type(factor);
                left_matmul_abstract_eval(factor_type.as_ref(), input_types)
            }
            Self::RightMatMul { factor } => {
                let factor_type = <V as Typed<ArrayType>>::r#type(factor);
                right_matmul_abstract_eval(factor_type.as_ref(), input_types)
            }
            Self::Reshape { input_shape, output_shape } => {
                ReshapeOperation::new(input_shape.clone(), output_shape.clone()).infer_output_types(input_types)
            }
            Self::Rematerialize(remat) => remat.infer_output_types(input_types),
            Self::Condition(condition) => condition.infer_output_types(input_types),
            Self::While(while_operation) => while_operation.infer_output_types(input_types),
            Self::Custom(op) => op.infer_output_types(input_types),
        }
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        match self {
            Self::Zero(zero) => zero.render(formatter, indentation),
            Self::One(one) => one.render(formatter, indentation),
            Self::Reshape { input_shape, output_shape } => {
                ReshapeOperation::new(input_shape.clone(), output_shape.clone()).render(formatter, indentation)
            }
            Self::Scale { factor } => OperationFormatter::new(formatter, indentation, self.operation_name())?
                .bracketed(|operation| operation.field("factor", factor)),
            Self::LeftMatMul { factor } | Self::RightMatMul { factor } => {
                OperationFormatter::new(formatter, indentation, self.operation_name())?
                    .bracketed(|operation| operation.field("factor", factor))
            }
            Self::Rematerialize(remat) => remat.render(formatter, indentation),
            Self::Condition(condition) => condition.render(formatter, indentation),
            Self::While(while_operation) => while_operation.render(formatter, indentation),
            Self::Custom(op) => op.render(formatter, indentation),
            _ => Display::fmt(self, formatter),
        }
    }
}

impl<V: Traceable<DataType> + Parameter> Operation<DataType> for LinearArrayOperation<V, DataType> {
    fn name(&self) -> &'static str {
        self.operation_name()
    }

    fn infer_output_types(&self, input_types: &[DataType]) -> Result<Vec<DataType>, TypeError> {
        match self {
            Self::Zero(zero) => zero.infer_output_types(input_types),
            Self::One(one) => one.infer_output_types(input_types),
            Self::ZeroLike => ZeroLikeOperation.infer_output_types(input_types),
            Self::OneLike => OneLikeOperation.infer_output_types(input_types),
            Self::Add => AddOperation.infer_output_types(input_types),
            Self::Neg => NegOperation.infer_output_types(input_types),
            Self::Scale { .. } => ScaleOperation::<DataType, V>::abstract_eval_static(input_types),
            Self::Custom(op) => op.infer_output_types(input_types),
            Self::Transpose
            | Self::LeftMatMul { .. }
            | Self::RightMatMul { .. }
            | Self::Reshape { .. }
            | Self::Rematerialize(_)
            | Self::Condition(_)
            | Self::While(_) => Err(unsupported_scalar_metadata_operation(self.operation_name())),
        }
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        match self {
            Self::Zero(zero) => zero.render(formatter, indentation),
            Self::One(one) => one.render(formatter, indentation),
            Self::Reshape { input_shape, output_shape } => {
                ReshapeOperation::new(input_shape.clone(), output_shape.clone()).render(formatter, indentation)
            }
            Self::Scale { factor } => OperationFormatter::new(formatter, indentation, self.operation_name())?
                .bracketed(|operation| operation.field("factor", factor)),
            Self::LeftMatMul { factor } | Self::RightMatMul { factor } => {
                OperationFormatter::new(formatter, indentation, self.operation_name())?
                    .bracketed(|operation| operation.field("factor", factor))
            }
            Self::Rematerialize(remat) => remat.render(formatter, indentation),
            Self::Condition(condition) => condition.render(formatter, indentation),
            Self::While(while_operation) => while_operation.render(formatter, indentation),
            Self::Custom(op) => op.render(formatter, indentation),
            _ => Display::fmt(self, formatter),
        }
    }
}

impl<
    V: Traceable<DataType>
        + Parameter
        + Add<Output = V>
        + Mul<Output = V>
        + Neg<Output = V>
        + Sin
        + Cos
        + Zero<DataType>
        + One<DataType>
        + ZeroLike
        + OneLike,
> InterpretableOperation<DataType, V> for ScalarOperation<V>
where
    Vec<V>: Parameterized<V, ParameterStructure: std::fmt::Debug + PartialEq>,
{
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        match self {
            Self::Zero(zero) => zero.interpret(inputs),
            Self::One(one) => one.interpret(inputs),
            Self::ZeroLike => ZeroLikeOperation.interpret(inputs),
            Self::OneLike => OneLikeOperation.interpret(inputs),
            Self::Add => <AddOperation as InterpretableOperation<DataType, V>>::interpret(&AddOperation, inputs),
            Self::Mul => <MulOperation as InterpretableOperation<DataType, V>>::interpret(&MulOperation, inputs),
            Self::Neg => <NegOperation as InterpretableOperation<DataType, V>>::interpret(&NegOperation, inputs),
            Self::Sin => <SinOperation as InterpretableOperation<DataType, V>>::interpret(&SinOperation, inputs),
            Self::Cos => <CosOperation as InterpretableOperation<DataType, V>>::interpret(&CosOperation, inputs),
            Self::Scale { factor } => ScaleOperation::new(factor.clone()).interpret(inputs),
            Self::Custom(op) => op.interpret(inputs),
        }
    }
}

impl<
    V: Traceable<DataType>
        + Parameter
        + Add<Output = V>
        + Neg<Output = V>
        + Mul<Output = V>
        + Zero<DataType>
        + One<DataType>
        + ZeroLike
        + OneLike,
> InterpretableOperation<DataType, V> for LinearScalarOperation<V>
where
    Vec<V>: Parameterized<V, ParameterStructure: std::fmt::Debug + PartialEq>,
{
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        match self {
            Self::Zero(zero) => zero.interpret(inputs),
            Self::One(one) => one.interpret(inputs),
            Self::ZeroLike => ZeroLikeOperation.interpret(inputs),
            Self::OneLike => OneLikeOperation.interpret(inputs),
            Self::Add => <AddOperation as InterpretableOperation<DataType, V>>::interpret(&AddOperation, inputs),
            Self::Neg => <NegOperation as InterpretableOperation<DataType, V>>::interpret(&NegOperation, inputs),
            Self::Scale { factor } => ScaleOperation::new(factor.clone()).interpret(inputs),
            Self::Custom(op) => op.interpret(inputs),
        }
    }
}

impl<'engine, E> InterpretableOperation<DataType, Tracer<'engine, E>> for LinearScalarOperation<Tracer<'engine, E>>
where
    E: DifferentiableTracingEngine<Type = DataType> + ?Sized + 'static,
    Tracer<'engine, E>: Add<Output = Tracer<'engine, E>>
        + Neg<Output = Tracer<'engine, E>>
        + Mul<Output = Tracer<'engine, E>>
        + ZeroLike
        + OneLike,
    Vec<Tracer<'engine, E>>: Parameterized<Tracer<'engine, E>, ParameterStructure: std::fmt::Debug + PartialEq>,
{
    fn interpret(&self, inputs: &[Tracer<'engine, E>]) -> Result<Vec<Tracer<'engine, E>>, TracingError> {
        match self {
            Self::Zero(zero) => Err(TypeError {
                message: format!(
                    "linear zero operation over tracer values was not materialized before interpretation for {}",
                    zero.output_type()
                ),
            }
            .into()),
            Self::One(one) => Err(TypeError {
                message: format!(
                    "linear one operation over tracer values was not materialized before interpretation for {}",
                    one.output_type()
                ),
            }
            .into()),
            Self::ZeroLike => ZeroLikeOperation.interpret(inputs),
            Self::OneLike => OneLikeOperation.interpret(inputs),
            Self::Add => {
                <AddOperation as InterpretableOperation<DataType, Tracer<'engine, E>>>::interpret(&AddOperation, inputs)
            }
            Self::Neg => {
                <NegOperation as InterpretableOperation<DataType, Tracer<'engine, E>>>::interpret(&NegOperation, inputs)
            }
            Self::Scale { factor } => ScaleOperation::new(factor.clone()).interpret(inputs),
            Self::Custom(op) => op.interpret(inputs),
        }
    }
}

/// [`InterpretableOperation`] for [`ArrayOperation`] requires the full union of value capabilities used by
/// the closed default ordinary-op carrier.
///
/// That broad union is local to [`ArrayOperation`] itself. The higher-level tracing APIs avoid
/// exposing it as one public value-bundle trait and instead express their requirements through the
/// specific staged op carrier bounds they actually exercise.
impl<
    V: Traceable<ArrayType>
        + Parameter
        + Add<Output = V>
        + Mul<Output = V>
        + Neg<Output = V>
        + Sin
        + Cos
        + Zero<ArrayType>
        + One<ArrayType>
        + ZeroLike
        + OneLike
        + MatrixOps
        + crate::tracing_v2::operations::reshape::ReshapeOps
        + ControlFlowValue,
> InterpretableOperation<ArrayType, V> for ArrayOperation<V>
where
    Vec<V>: Parameterized<V, ParameterStructure: std::fmt::Debug + PartialEq>,
{
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
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
            Self::Custom(op) => op.interpret(inputs),
        }
    }
}

impl<
    V: Traceable<DataType>
        + Parameter
        + Add<Output = V>
        + Mul<Output = V>
        + Neg<Output = V>
        + Sin
        + Cos
        + Zero<DataType>
        + One<DataType>
        + ZeroLike
        + OneLike,
> InterpretableOperation<DataType, V> for ArrayOperation<V, DataType>
where
    Vec<V>: Parameterized<V, ParameterStructure: std::fmt::Debug + PartialEq>,
{
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        match self {
            Self::Zero(zero) => zero.interpret(inputs),
            Self::One(one) => one.interpret(inputs),
            Self::ZeroLike => ZeroLikeOperation.interpret(inputs),
            Self::OneLike => OneLikeOperation.interpret(inputs),
            Self::Add => <AddOperation as InterpretableOperation<DataType, V>>::interpret(&AddOperation, inputs),
            Self::Mul => <MulOperation as InterpretableOperation<DataType, V>>::interpret(&MulOperation, inputs),
            Self::Neg => <NegOperation as InterpretableOperation<DataType, V>>::interpret(&NegOperation, inputs),
            Self::Sin => <SinOperation as InterpretableOperation<DataType, V>>::interpret(&SinOperation, inputs),
            Self::Cos => <CosOperation as InterpretableOperation<DataType, V>>::interpret(&CosOperation, inputs),
            Self::Scale { factor } => ScaleOperation::new(factor.clone()).interpret(inputs),
            Self::Custom(op) => op.interpret(inputs),
            Self::MatrixMultiply
            | Self::Transpose
            | Self::Reshape { .. }
            | Self::Rematerialize(_)
            | Self::Condition(_)
            | Self::While(_) => Err(unsupported_scalar_metadata_operation(self.operation_name()).into()),
        }
    }
}

impl<
    V: Traceable<ArrayType>
        + Parameter
        + Add<Output = V>
        + Neg<Output = V>
        + Mul<Output = V>
        + Zero<ArrayType>
        + One<ArrayType>
        + ZeroLike
        + OneLike
        + MatrixOps
        + crate::tracing_v2::operations::reshape::ReshapeOps
        + ControlFlowValue,
> InterpretableOperation<ArrayType, V> for LinearArrayOperation<V>
where
    Vec<V>: Parameterized<V, ParameterStructure: std::fmt::Debug + PartialEq>,
{
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        match self {
            Self::Zero(zero) => zero.interpret(inputs),
            Self::One(one) => one.interpret(inputs),
            Self::ZeroLike => ZeroLikeOperation.interpret(inputs),
            Self::OneLike => OneLikeOperation.interpret(inputs),
            Self::Add => AddOperation.interpret(inputs),
            Self::Neg => NegOperation.interpret(inputs),
            Self::Transpose => MatrixTransposeOperation.interpret(inputs),
            Self::Scale { factor } => ScaleOperation::new(factor.clone()).interpret(inputs),
            Self::LeftMatMul { factor } => LeftMatMulOperation::new(factor.clone()).interpret(inputs),
            Self::RightMatMul { factor } => RightMatMulOperation::new(factor.clone()).interpret(inputs),
            Self::Reshape { input_shape, output_shape } => {
                ReshapeOperation::new(input_shape.clone(), output_shape.clone()).interpret(inputs)
            }
            Self::Rematerialize(remat) => remat.interpret(inputs),
            Self::Condition(condition) => condition.interpret(inputs),
            Self::While(while_operation) => while_operation.interpret(inputs),
            Self::Custom(op) => op.interpret(inputs),
        }
    }
}

impl<
    V: Traceable<DataType>
        + Parameter
        + Add<Output = V>
        + Neg<Output = V>
        + Mul<Output = V>
        + Zero<DataType>
        + One<DataType>
        + ZeroLike
        + OneLike,
> InterpretableOperation<DataType, V> for LinearArrayOperation<V, DataType>
where
    Vec<V>: Parameterized<V, ParameterStructure: std::fmt::Debug + PartialEq>,
{
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        match self {
            Self::Zero(zero) => zero.interpret(inputs),
            Self::One(one) => one.interpret(inputs),
            Self::ZeroLike => ZeroLikeOperation.interpret(inputs),
            Self::OneLike => OneLikeOperation.interpret(inputs),
            Self::Add => <AddOperation as InterpretableOperation<DataType, V>>::interpret(&AddOperation, inputs),
            Self::Neg => <NegOperation as InterpretableOperation<DataType, V>>::interpret(&NegOperation, inputs),
            Self::Scale { factor } => ScaleOperation::new(factor.clone()).interpret(inputs),
            Self::Custom(op) => op.interpret(inputs),
            Self::Transpose
            | Self::LeftMatMul { .. }
            | Self::RightMatMul { .. }
            | Self::Reshape { .. }
            | Self::Rematerialize(_)
            | Self::Condition(_)
            | Self::While(_) => Err(unsupported_scalar_metadata_operation(self.operation_name()).into()),
        }
    }
}

impl<'engine, E> InterpretableOperation<ArrayType, Tracer<'engine, E>> for LinearArrayOperation<Tracer<'engine, E>>
where
    E: DifferentiableTracingEngine<Type = ArrayType> + ?Sized + 'static,
    Tracer<'engine, E>: Add<Output = Tracer<'engine, E>>
        + Neg<Output = Tracer<'engine, E>>
        + Mul<Output = Tracer<'engine, E>>
        + ZeroLike
        + OneLike
        + MatrixOps
        + crate::tracing_v2::operations::reshape::ReshapeOps
        + ControlFlowValue,
    Vec<Tracer<'engine, E>>: Parameterized<Tracer<'engine, E>, ParameterStructure: std::fmt::Debug + PartialEq>,
{
    fn interpret(&self, inputs: &[Tracer<'engine, E>]) -> Result<Vec<Tracer<'engine, E>>, TracingError> {
        match self {
            Self::Zero(zero) => Err(TypeError {
                message: format!(
                    "linear zero operation over tracer values was not materialized before interpretation for {}",
                    zero.output_type()
                ),
            }
            .into()),
            Self::One(one) => Err(TypeError {
                message: format!(
                    "linear one operation over tracer values was not materialized before interpretation for {}",
                    one.output_type()
                ),
            }
            .into()),
            Self::ZeroLike => ZeroLikeOperation.interpret(inputs),
            Self::OneLike => OneLikeOperation.interpret(inputs),
            Self::Add => <AddOperation as InterpretableOperation<ArrayType, Tracer<'engine, E>>>::interpret(
                &AddOperation,
                inputs,
            ),
            Self::Neg => <NegOperation as InterpretableOperation<ArrayType, Tracer<'engine, E>>>::interpret(
                &NegOperation,
                inputs,
            ),
            Self::Transpose => MatrixTransposeOperation.interpret(inputs),
            Self::Scale { factor } => ScaleOperation::new(factor.clone()).interpret(inputs),
            Self::LeftMatMul { factor } => LeftMatMulOperation::new(factor.clone()).interpret(inputs),
            Self::RightMatMul { factor } => RightMatMulOperation::new(factor.clone()).interpret(inputs),
            Self::Reshape { input_shape, output_shape } => {
                ReshapeOperation::new(input_shape.clone(), output_shape.clone()).interpret(inputs)
            }
            Self::Rematerialize(remat) => remat.interpret(inputs),
            Self::Condition(condition) => condition.interpret(inputs),
            Self::While(while_operation) => while_operation.interpret(inputs),
            Self::Custom(op) => op.interpret(inputs),
        }
    }
}

impl<'engine, E> InterpretableOperation<DataType, Tracer<'engine, E>>
    for LinearArrayOperation<Tracer<'engine, E>, DataType>
where
    E: DifferentiableTracingEngine<Type = DataType> + ?Sized + 'static,
    Tracer<'engine, E>: Add<Output = Tracer<'engine, E>>
        + Neg<Output = Tracer<'engine, E>>
        + Mul<Output = Tracer<'engine, E>>
        + ZeroLike
        + OneLike,
    Vec<Tracer<'engine, E>>: Parameterized<Tracer<'engine, E>, ParameterStructure: std::fmt::Debug + PartialEq>,
{
    fn interpret(&self, inputs: &[Tracer<'engine, E>]) -> Result<Vec<Tracer<'engine, E>>, TracingError> {
        match self {
            Self::Zero(zero) => Err(TypeError {
                message: format!(
                    "linear zero operation over tracer values was not materialized before interpretation for {}",
                    zero.output_type()
                ),
            }
            .into()),
            Self::One(one) => Err(TypeError {
                message: format!(
                    "linear one operation over tracer values was not materialized before interpretation for {}",
                    one.output_type()
                ),
            }
            .into()),
            Self::ZeroLike => ZeroLikeOperation.interpret(inputs),
            Self::OneLike => OneLikeOperation.interpret(inputs),
            Self::Add => {
                <AddOperation as InterpretableOperation<DataType, Tracer<'engine, E>>>::interpret(&AddOperation, inputs)
            }
            Self::Neg => {
                <NegOperation as InterpretableOperation<DataType, Tracer<'engine, E>>>::interpret(&NegOperation, inputs)
            }
            Self::Scale { factor } => ScaleOperation::new(factor.clone()).interpret(inputs),
            Self::Custom(op) => op.interpret(inputs),
            Self::Transpose
            | Self::LeftMatMul { .. }
            | Self::RightMatMul { .. }
            | Self::Reshape { .. }
            | Self::Rematerialize(_)
            | Self::Condition(_)
            | Self::While(_) => Err(unsupported_scalar_metadata_operation(self.operation_name()).into()),
        }
    }
}

impl<V: Traceable<DataType> + Parameter + Add<Output = V> + Neg<Output = V> + ZeroLike + OneLike>
    LinearOperation<DataType, V, LinearScalarOperation<V>> for LinearScalarOperation<V>
where
    Vec<V>: Parameterized<V, ParameterStructure: std::fmt::Debug + PartialEq>,
{
    fn transpose(
        &self,
        context: &mut crate::tracing::transposition::TranspositionContext<DataType, V, LinearScalarOperation<V>>,
        output_cotangents: &[Option<crate::tracing::AtomId>],
    ) -> Result<Vec<Option<crate::tracing::AtomId>>, TracingError> {
        match self {
            Self::Zero(zero) => zero.transpose(context, output_cotangents),
            Self::One(one) => one.transpose(context, output_cotangents),
            Self::ZeroLike => ZeroLikeOperation.transpose(context, output_cotangents),
            Self::OneLike => OneLikeOperation.transpose(context, output_cotangents),
            Self::Add => {
                if output_cotangents.len() != 1 {
                    return Err(TracingError::InvalidInputCount { expected: 1, got: output_cotangents.len() });
                }
                Ok(vec![output_cotangents[0], output_cotangents[0]])
            }
            Self::Neg => {
                if output_cotangents.len() != 1 {
                    return Err(TracingError::InvalidInputCount { expected: 1, got: output_cotangents.len() });
                }
                output_cotangents[0]
                    .map(|atom| context.stage(Self::Neg, &[atom]).map(|outputs| vec![Some(outputs[0])]))
                    .unwrap_or_else(|| Ok(vec![None]))
            }
            Self::Scale { factor } => {
                if output_cotangents.len() != 1 {
                    return Err(TracingError::InvalidInputCount { expected: 1, got: output_cotangents.len() });
                }
                output_cotangents[0]
                    .map(|atom| {
                        context
                            .stage(Self::Scale { factor: factor.clone() }, &[atom])
                            .map(|outputs| vec![Some(outputs[0])])
                    })
                    .unwrap_or_else(|| Ok(vec![None]))
            }
            Self::Custom(_) => Err(TypeError {
                message: "custom scalar linear transpose requires a carrier-specific transpose rule".to_string(),
            }
            .into()),
        }
    }
}

impl<
    V: Traceable<ArrayType>
        + Parameter
        + Add<Output = V>
        + Neg<Output = V>
        + Mul<Output = V>
        + ZeroLike
        + OneLike
        + MatrixOps
        + crate::tracing_v2::operations::reshape::ReshapeOps
        + ControlFlowValue,
> LinearOperation<ArrayType, V, LinearArrayOperation<V>> for LinearArrayOperation<V>
where
    Vec<V>: Parameterized<V, ParameterStructure: std::fmt::Debug + PartialEq>,
{
    fn transpose(
        &self,
        context: &mut crate::tracing::transposition::TranspositionContext<ArrayType, V, LinearArrayOperation<V>>,
        output_cotangents: &[Option<crate::tracing::AtomId>],
    ) -> Result<Vec<Option<crate::tracing::AtomId>>, TracingError> {
        match self {
            Self::Zero(zero) => zero.transpose(context, output_cotangents),
            Self::One(one) => one.transpose(context, output_cotangents),
            Self::ZeroLike => ZeroLikeOperation.transpose(context, output_cotangents),
            Self::OneLike => OneLikeOperation.transpose(context, output_cotangents),
            Self::Add => AddOperation.transpose(context, output_cotangents),
            Self::Neg => NegOperation.transpose(context, output_cotangents),
            Self::Transpose => MatrixTransposeOperation.transpose(context, output_cotangents),
            Self::Scale { factor } => ScaleOperation::new(factor.clone()).transpose(context, output_cotangents),
            Self::LeftMatMul { factor } => {
                LeftMatMulOperation::new(factor.clone()).transpose(context, output_cotangents)
            }
            Self::RightMatMul { factor } => {
                RightMatMulOperation::new(factor.clone()).transpose(context, output_cotangents)
            }
            Self::Reshape { input_shape, output_shape } => {
                ReshapeOperation::new(input_shape.clone(), output_shape.clone()).transpose(context, output_cotangents)
            }
            Self::Rematerialize(remat) => remat.transpose(context, output_cotangents),
            Self::Condition(condition) => condition.transpose(context, output_cotangents),
            Self::While(while_operation) => while_operation.transpose(context, output_cotangents),
            Self::Custom(op) => op.transpose(context, output_cotangents),
        }
    }
}

impl<V: Traceable<DataType> + Parameter + Add<Output = V> + Neg<Output = V> + Mul<Output = V> + ZeroLike + OneLike>
    LinearOperation<DataType, V, LinearArrayOperation<V, DataType>> for LinearArrayOperation<V, DataType>
where
    Vec<V>: Parameterized<V, ParameterStructure: std::fmt::Debug + PartialEq>,
{
    fn transpose(
        &self,
        context: &mut crate::tracing::transposition::TranspositionContext<
            DataType,
            V,
            LinearArrayOperation<V, DataType>,
        >,
        output_cotangents: &[Option<crate::tracing::AtomId>],
    ) -> Result<Vec<Option<crate::tracing::AtomId>>, TracingError> {
        match self {
            Self::Zero(zero) => zero.transpose(context, output_cotangents),
            Self::One(one) => one.transpose(context, output_cotangents),
            Self::ZeroLike => ZeroLikeOperation.transpose(context, output_cotangents),
            Self::OneLike => OneLikeOperation.transpose(context, output_cotangents),
            Self::Add => AddOperation.transpose(context, output_cotangents),
            Self::Neg => NegOperation.transpose(context, output_cotangents),
            Self::Scale { factor } => ScaleOperation::new(factor.clone()).transpose(context, output_cotangents),
            Self::Custom(op) => op.transpose(context, output_cotangents),
            Self::Transpose
            | Self::LeftMatMul { .. }
            | Self::RightMatMul { .. }
            | Self::Reshape { .. }
            | Self::Rematerialize(_)
            | Self::Condition(_)
            | Self::While(_) => Err(unsupported_scalar_metadata_operation(self.operation_name()).into()),
        }
    }
}

impl<
    V: Value<DataType>
        + Traceable<ArrayType>
        + Add<Output = V>
        + Mul<Output = V>
        + Neg<Output = V>
        + Sin
        + Cos
        + ZeroLike
        + OneLike
        + Zero<DataType>
        + One<DataType>
        + Parameterized<V>
        + Differentiable<DataType, Tangent = V>
        + 'static,
    E: LinearizableEngine<Type = DataType, Value = V, LinearOperationCarrier = LinearScalarOperation<V>> + 'static,
> DifferentiableOperation<E> for ScalarOperation<V>
where
    V::ParameterStructure: std::fmt::Debug + PartialEq,
    Vec<V>: Parameterized<V, ParameterStructure: std::fmt::Debug + PartialEq>,
    LinearScalarOperation<V>: super::SupportsAdd<DataType, V>
        + super::SupportsNeg<DataType, V>
        + super::SupportsScale<DataType, V>
        + SupportsZero<DataType, V>
        + SupportsZeroLike<DataType, V>,
{
    fn jvp(
        &self,
        context: &mut JvpContext<'_, E>,
        inputs: &[JvpTracer<V, AtomId>],
    ) -> Result<Vec<JvpTracer<V, AtomId>>, TracingError> {
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
            Self::Scale { factor } => ScaleOperation::new(factor.clone()).jvp(context, inputs),
            Self::Custom(_) => {
                Err(TypeError { message: format!("{} is not supported for scalar data type metadata", self.name()) }
                    .into())
            }
        }
    }
}

impl<'engine, V, EInner> DifferentiableOperation<crate::tracing::engines::TracingContext<'engine, EInner>>
    for ScalarOperation<V>
where
    V: Value<DataType>
        + Traceable<ArrayType>
        + Add<Output = V>
        + Mul<Output = V>
        + Neg<Output = V>
        + Sin
        + Cos
        + ZeroLike
        + OneLike
        + Zero<DataType>
        + One<DataType>
        + Parameterized<V>
        + Differentiable<DataType, Tangent = V>
        + 'static,
    EInner: DifferentiableTracingEngine<Type = DataType, Value = V, OperationCarrier = ScalarOperation<V>>
        + ?Sized
        + 'static,
    V::ParameterStructure: std::fmt::Debug + PartialEq,
    Vec<V>: Parameterized<V, ParameterStructure: std::fmt::Debug + PartialEq>,
    LinearScalarOperation<V>: super::SupportsAdd<DataType, V>
        + super::SupportsNeg<DataType, V>
        + super::SupportsScale<DataType, V>
        + SupportsZero<DataType, V>
        + SupportsZeroLike<DataType, V>
        + Clone
        + InterpretableOperation<DataType, V>
        + LinearOperation<DataType, V, LinearScalarOperation<V>>,
    Tracer<'engine, EInner>: Add<Output = Tracer<'engine, EInner>>
        + Mul<Output = Tracer<'engine, EInner>>
        + Neg<Output = Tracer<'engine, EInner>>
        + Sin
        + Cos
        + ZeroLike
        + OneLike,
    EInner::LinearOperationCarrier<'engine>: Clone
        + InterpretableOperation<DataType, Tracer<'engine, EInner>>
        + LinearOperation<DataType, Tracer<'engine, EInner>, EInner::LinearOperationCarrier<'engine>>
        + SupportsScale<DataType, Tracer<'engine, EInner>>
        + SupportsZeroLike<DataType, Tracer<'engine, EInner>>
        + SupportsZero<DataType, Tracer<'engine, EInner>>,
{
    fn jvp(
        &self,
        context: &mut JvpContext<'_, crate::tracing::engines::TracingContext<'engine, EInner>>,
        inputs: &[JvpTracer<Tracer<'engine, EInner>, AtomId>],
    ) -> Result<Vec<JvpTracer<Tracer<'engine, EInner>, AtomId>>, TracingError> {
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
            Self::Scale { factor } => {
                if inputs.len() != 1 {
                    return Err(TracingError::InvalidInputCount { expected: 1, got: inputs.len() });
                }
                let input = &inputs[0];
                let factor_tracer = context.engine.constant(factor.clone());
                let tangent =
                    context
                        .apply_operation(
                            &[input.tangent],
                            <EInner::LinearOperationCarrier<'engine> as SupportsScale<
                                DataType,
                                Tracer<'engine, EInner>,
                            >>::scale_operation(factor_tracer.clone()),
                            1,
                        )?
                        .into_iter()
                        .next()
                        .expect("scale jvp should produce one tangent");
                Ok(vec![JvpTracer { primal: factor_tracer * input.primal.clone(), tangent }])
            }
            Self::Custom(_) => {
                Err(TypeError { message: format!("{} is not supported for scalar data type metadata", self.name()) }
                    .into())
            }
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
        + Zero<ArrayType>
        + One<ArrayType>
        + Parameterized<V>
        + MatrixOps
        + crate::tracing_v2::operations::reshape::ReshapeOps
        + ControlFlowValue
        + Differentiable<ArrayType, Tangent = V>
        + 'static,
    E: LinearizableEngine<Type = ArrayType, Value = V, LinearOperationCarrier = LinearArrayOperation<V>> + 'static,
> DifferentiableOperation<E> for ArrayOperation<V>
where
    V: Differentiable<ArrayType, Tangent = V>,
    V::ParameterStructure: std::fmt::Debug + PartialEq,
    Vec<V>: Parameterized<V, ParameterStructure: std::fmt::Debug + PartialEq>,
    LinearArrayOperation<V>: super::SupportsAdd<ArrayType, V>
        + super::SupportsNeg<ArrayType, V>
        + super::SupportsScale<ArrayType, V>
        + super::SupportsLeftMatMul<ArrayType, V>
        + super::SupportsRightMatMul<ArrayType, V>
        + super::SupportsMatrixTranspose<ArrayType, V>
        + super::SupportsReshape<ArrayType, V>
        + SupportsZero<ArrayType, V>
        + SupportsZeroLike<ArrayType, V>,
{
    fn jvp(
        &self,
        context: &mut JvpContext<'_, E>,
        inputs: &[JvpTracer<V, AtomId>],
    ) -> Result<Vec<JvpTracer<V, AtomId>>, TracingError> {
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
            Self::Scale { factor } => ScaleOperation::new(factor.clone()).jvp(context, inputs),
            Self::MatrixMultiply => MatMulOperation.jvp(context, inputs),
            Self::Transpose => MatrixTransposeOperation.jvp(context, inputs),
            Self::Reshape { input_shape, output_shape } => {
                ReshapeOperation::new(input_shape.clone(), output_shape.clone()).jvp(context, inputs)
            }
            Self::Rematerialize(remat) => remat.as_ref().jvp(context, inputs),
            Self::Condition(condition) => condition.as_ref().jvp(context, inputs),
            Self::While(while_operation) => while_operation.as_ref().jvp(context, inputs),
            Self::Custom(op) => op.jvp(context, inputs),
        }
    }
}

impl<
    V: Value<DataType>
        + Traceable<ArrayType>
        + Add<Output = V>
        + Mul<Output = V>
        + Neg<Output = V>
        + Sin
        + Cos
        + ZeroLike
        + OneLike
        + Zero<DataType>
        + One<DataType>
        + Parameterized<V>
        + Differentiable<DataType, Tangent = V>
        + 'static,
    E: LinearizableEngine<Type = DataType, Value = V, LinearOperationCarrier = LinearArrayOperation<V, DataType>>
        + 'static,
> DifferentiableOperation<E> for ArrayOperation<V, DataType>
where
    V::ParameterStructure: std::fmt::Debug + PartialEq,
    Vec<V>: Parameterized<V, ParameterStructure: std::fmt::Debug + PartialEq>,
    LinearArrayOperation<V, DataType>: super::SupportsAdd<DataType, V>
        + super::SupportsNeg<DataType, V>
        + super::SupportsScale<DataType, V>
        + SupportsZero<DataType, V>
        + SupportsZeroLike<DataType, V>,
{
    fn jvp(
        &self,
        context: &mut JvpContext<'_, E>,
        inputs: &[JvpTracer<V, AtomId>],
    ) -> Result<Vec<JvpTracer<V, AtomId>>, TracingError> {
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
            Self::Scale { factor } => ScaleOperation::new(factor.clone()).jvp(context, inputs),
            Self::MatrixMultiply
            | Self::Transpose
            | Self::Reshape { .. }
            | Self::Rematerialize(_)
            | Self::Condition(_)
            | Self::While(_)
            | Self::Custom(_) => {
                Err(TypeError { message: format!("{} is not supported for scalar data type metadata", self.name()) }
                    .into())
            }
        }
    }
}

/// Linearization-engine dispatcher for [`ArrayOperation`] under the traced-linearization path.
///
/// Forwards each variant to the per-op JVP rule, picking up the
/// [`TracingContext`](crate::tracing::engines::TracingContext)-keyed impl for captured
/// [`Scale`](Self::Scale), the [`Rematerialize`](Self::Rematerialize) impl that recurses via
/// [`TracingContext::linearize`](crate::tracing::engines::TracingContext::linearize), the
/// [`Condition`](Self::Condition) / [`While`](Self::While) stub impls (predicate extraction does
/// not work at trace time), and the [`Custom`](Self::Custom) bridge to the registered traced
/// linearization rule.
impl<'engine, V, EInner> DifferentiableOperation<crate::tracing::engines::TracingContext<'engine, EInner>>
    for ArrayOperation<V>
where
    V: Value<ArrayType>
        + Add<Output = V>
        + Mul<Output = V>
        + Neg<Output = V>
        + Sin
        + Cos
        + ZeroLike
        + OneLike
        + Zero<ArrayType>
        + One<ArrayType>
        + Parameterized<V>
        + MatrixOps
        + crate::tracing_v2::operations::reshape::ReshapeOps
        + ControlFlowValue
        + Differentiable<ArrayType, Tangent = V>
        + 'static,
    EInner: DifferentiableTracingEngine<Type = ArrayType, Value = V, OperationCarrier = ArrayOperation<V>>
        + ?Sized
        + 'static,
    V::ParameterStructure: std::fmt::Debug + PartialEq,
    Vec<V>: Parameterized<V, ParameterStructure: std::fmt::Debug + PartialEq>,
    LinearArrayOperation<V>: super::SupportsAdd<ArrayType, V>
        + super::SupportsNeg<ArrayType, V>
        + super::SupportsScale<ArrayType, V>
        + super::SupportsLeftMatMul<ArrayType, V>
        + super::SupportsRightMatMul<ArrayType, V>
        + super::SupportsMatrixTranspose<ArrayType, V>
        + super::SupportsReshape<ArrayType, V>
        + SupportsZero<ArrayType, V>
        + SupportsZeroLike<ArrayType, V>
        + Clone
        + InterpretableOperation<ArrayType, V>
        + LinearOperation<ArrayType, V, LinearArrayOperation<V>>,
    Tracer<'engine, EInner>: Add<Output = Tracer<'engine, EInner>>
        + Mul<Output = Tracer<'engine, EInner>>
        + Neg<Output = Tracer<'engine, EInner>>
        + Sin
        + Cos
        + MatrixOps
        + ZeroLike
        + OneLike,
    EInner::LinearOperationCarrier<'engine>: Clone
        + InterpretableOperation<ArrayType, Tracer<'engine, EInner>>
        + LinearOperation<ArrayType, Tracer<'engine, EInner>, EInner::LinearOperationCarrier<'engine>>
        + SupportsLeftMatMul<ArrayType, Tracer<'engine, EInner>>
        + SupportsRightMatMul<ArrayType, Tracer<'engine, EInner>>
        + SupportsMatrixTranspose<ArrayType, Tracer<'engine, EInner>>
        + SupportsReshape<ArrayType, Tracer<'engine, EInner>>
        + SupportsZeroLike<ArrayType, Tracer<'engine, EInner>>
        + SupportsZero<ArrayType, Tracer<'engine, EInner>>
        + SupportsLinearRematerialize<ArrayType, Tracer<'engine, EInner>>,
{
    fn jvp(
        &self,
        context: &mut JvpContext<'_, crate::tracing::engines::TracingContext<'engine, EInner>>,
        inputs: &[JvpTracer<Tracer<'engine, EInner>, AtomId>],
    ) -> Result<Vec<JvpTracer<Tracer<'engine, EInner>, AtomId>>, TracingError> {
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
            Self::Scale { factor } => ScaleOperation::new(factor.clone()).jvp(context, inputs),
            Self::MatrixMultiply => MatMulOperation.jvp(context, inputs),
            Self::Transpose => MatrixTransposeOperation.jvp(context, inputs),
            Self::Reshape { input_shape, output_shape } => {
                ReshapeOperation::new(input_shape.clone(), output_shape.clone()).jvp(context, inputs)
            }
            Self::Rematerialize(remat) => remat.as_ref().jvp(context, inputs),
            Self::Condition(condition) => condition.as_ref().jvp(context, inputs),
            Self::While(while_operation) => while_operation.as_ref().jvp(context, inputs),
            Self::Custom(op) => op.jvp(context, inputs),
        }
    }
}

impl<'engine, V, EInner> DifferentiableOperation<crate::tracing::engines::TracingContext<'engine, EInner>>
    for ArrayOperation<V, DataType>
where
    V: Value<DataType>
        + Traceable<ArrayType>
        + Add<Output = V>
        + Mul<Output = V>
        + Neg<Output = V>
        + Sin
        + Cos
        + ZeroLike
        + OneLike
        + Zero<DataType>
        + One<DataType>
        + Parameterized<V>
        + Differentiable<DataType, Tangent = V>
        + 'static,
    EInner: DifferentiableTracingEngine<Type = DataType, Value = V, OperationCarrier = ArrayOperation<V, DataType>>
        + ?Sized
        + 'static,
    V::ParameterStructure: std::fmt::Debug + PartialEq,
    Vec<V>: Parameterized<V, ParameterStructure: std::fmt::Debug + PartialEq>,
    LinearArrayOperation<V, DataType>: super::SupportsAdd<DataType, V>
        + super::SupportsNeg<DataType, V>
        + super::SupportsScale<DataType, V>
        + SupportsZero<DataType, V>
        + SupportsZeroLike<DataType, V>
        + Clone
        + InterpretableOperation<DataType, V>
        + LinearOperation<DataType, V, LinearArrayOperation<V, DataType>>,
    Tracer<'engine, EInner>: Add<Output = Tracer<'engine, EInner>>
        + Mul<Output = Tracer<'engine, EInner>>
        + Neg<Output = Tracer<'engine, EInner>>
        + Sin
        + Cos
        + ZeroLike
        + OneLike,
    EInner::LinearOperationCarrier<'engine>: Clone
        + InterpretableOperation<DataType, Tracer<'engine, EInner>>
        + LinearOperation<DataType, Tracer<'engine, EInner>, EInner::LinearOperationCarrier<'engine>>
        + SupportsScale<DataType, Tracer<'engine, EInner>>
        + SupportsZeroLike<DataType, Tracer<'engine, EInner>>
        + SupportsZero<DataType, Tracer<'engine, EInner>>,
{
    fn jvp(
        &self,
        context: &mut JvpContext<'_, crate::tracing::engines::TracingContext<'engine, EInner>>,
        inputs: &[JvpTracer<Tracer<'engine, EInner>, AtomId>],
    ) -> Result<Vec<JvpTracer<Tracer<'engine, EInner>, AtomId>>, TracingError> {
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
            Self::Scale { factor } => {
                if inputs.len() != 1 {
                    return Err(TracingError::InvalidInputCount { expected: 1, got: inputs.len() });
                }
                let input = &inputs[0];
                let factor_tracer = context.engine.constant(factor.clone());
                let tangent =
                    context
                        .apply_operation(
                            &[input.tangent],
                            <EInner::LinearOperationCarrier<'engine> as SupportsScale<
                                DataType,
                                Tracer<'engine, EInner>,
                            >>::scale_operation(factor_tracer.clone()),
                            1,
                        )?
                        .into_iter()
                        .next()
                        .expect("scale jvp should produce one tangent");
                Ok(vec![JvpTracer { primal: factor_tracer * input.primal.clone(), tangent }])
            }
            Self::MatrixMultiply
            | Self::Transpose
            | Self::Reshape { .. }
            | Self::Rematerialize(_)
            | Self::Condition(_)
            | Self::While(_)
            | Self::Custom(_) => {
                Err(TypeError { message: format!("{} is not supported for scalar data type metadata", self.name()) }
                    .into())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    // Primitive-operation behavior is exercised through the per-operation modules and transform tests.
}
